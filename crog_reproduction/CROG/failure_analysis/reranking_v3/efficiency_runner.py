from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .experiment_config import ENSEMBLE_SEEDS
from .feature_store import FeatureCatalog, prior_array_to_lookup
from .models.fullchain_ranker import FullChainRanker
from .oof import (
    _normalizers_from_json,
    model_kwargs,
    required_array_keys,
    resolve_device,
    torch_feature_batch,
)
from .precision_efficiency import array_storage_statistics, measure_efficiency
from .schema import (
    artifact_identity,
    atomic_write_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


_CATALOG_KEYS = {"artifact_dirs", "head_override_dirs", "candidate_feature_paths"}


def _resolve_descriptor_paths(base: Path, values: Sequence[str]) -> list[Path]:
    result = []
    for value in values:
        path = Path(value).expanduser()
        result.append((base / path).resolve() if not path.is_absolute() else path.resolve())
    return result


def load_feature_catalog_descriptor(
    descriptor_path: str | Path,
) -> tuple[FeatureCatalog, dict[str, Any]]:
    """Load the label-free paths needed to reconstruct a :class:`FeatureCatalog`."""
    path = Path(descriptor_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("feature catalog descriptor must be a JSON object")
    unknown = set(payload) - _CATALOG_KEYS
    if unknown:
        raise ValueError(f"unknown feature catalog descriptor fields: {sorted(unknown)}")
    artifacts = payload.get("artifact_dirs")
    candidates = payload.get("candidate_feature_paths")
    overlays = payload.get("head_override_dirs", [])
    if not isinstance(artifacts, list) or not artifacts or not all(isinstance(v, str) for v in artifacts):
        raise ValueError("artifact_dirs must be a non-empty string list")
    if not isinstance(overlays, list) or not all(isinstance(v, str) for v in overlays):
        raise ValueError("head_override_dirs must be a string list")
    if not isinstance(candidates, list) or not candidates or not all(isinstance(v, str) for v in candidates):
        raise ValueError("candidate_feature_paths must be a non-empty string list")
    resolved = {
        "artifact_dirs": _resolve_descriptor_paths(path.parent, artifacts),
        "head_override_dirs": _resolve_descriptor_paths(path.parent, overlays),
        "candidate_feature_paths": _resolve_descriptor_paths(path.parent, candidates),
    }
    catalog = FeatureCatalog(
        resolved["artifact_dirs"],
        head_override_dirs=resolved["head_override_dirs"],
        candidate_feature_paths=resolved["candidate_feature_paths"],
    )
    evidence = {
        "descriptor": artifact_identity(path),
        "resolved": {key: [str(value) for value in values] for key, values in resolved.items()},
        "catalog_provenance": catalog.provenance(),
    }
    return catalog, evidence


def load_sample_ids(path: str | Path) -> set[str]:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = payload.get("sample_ids") if isinstance(payload, dict) else payload
    else:
        values = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("sample ID input must contain a non-empty string list")
    if len(values) != len(set(values)):
        raise ValueError("sample ID input contains duplicates")
    return set(values)


def load_v2_prior_npz(
    path: str | Path, *, sample_ids: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        if "sample_ids" not in payload.files or "prior" not in payload.files:
            raise ValueError("V2 prior NPZ must contain sample_ids and prior")
        source_ids = np.asarray(payload["sample_ids"]).astype(str)
        prior = np.asarray(payload["prior"], dtype=np.float32)
        valid = np.asarray(payload["valid"], dtype=bool) if "valid" in payload.files else np.ones(len(source_ids), bool)
    if len(source_ids) == 0 or len(set(source_ids.tolist())) != len(source_ids):
        raise ValueError("V2 prior sample identity must be non-empty and unique")
    if prior.shape != (len(source_ids), 5, 80) or not np.isfinite(prior).all():
        raise ValueError("V2 prior must be finite [samples,5,80]")
    if valid.shape != (len(source_ids),):
        raise ValueError("V2 prior valid mask must be [samples]")
    lookup = {value: index for index, value in enumerate(source_ids.tolist())}
    missing = sample_ids - set(lookup)
    if missing:
        raise ValueError(f"V2 prior is missing requested samples: {sorted(missing)[:5]}")
    invalid = sorted(value for value in sample_ids if not valid[lookup[value]])
    if invalid:
        raise ValueError(f"V2 prior marks requested samples invalid: {invalid[:5]}")
    ordered = sorted(sample_ids)
    values = np.stack([prior[lookup[value]] for value in ordered]).astype(np.float32)
    return prior_array_to_lookup(np.asarray(ordered), values), {
        "artifact": artifact_identity(source),
        "source_row_count": len(source_ids),
        "selected_row_count": len(ordered),
        "unused_source_row_count": len(source_ids) - len(ordered),
        "all_selected_rows_valid": True,
    }


def _path_storage(paths: Sequence[str | Path]) -> dict[str, Any]:
    files: dict[str, int] = {}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            files[str(path)] = int(path.stat().st_size)
        elif path.is_dir():
            for value in path.rglob("*"):
                if value.is_file() and not value.is_symlink():
                    files[str(value.resolve())] = int(value.stat().st_size)
        else:
            raise FileNotFoundError(path)
    return {
        "file_count": len(files),
        "total_bytes": sum(files.values()),
        "paths": [str(Path(value).expanduser().resolve()) for value in paths],
    }


def _mps_memory_snapshot(device: torch.device) -> dict[str, int] | None:
    if device.type != "mps":
        return None
    result: dict[str, int] = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        function = getattr(torch.mps, name, None)
        if function is not None:
            try:
                result[f"{name}_bytes"] = int(function())
            except RuntimeError:
                pass
    return result or None


def _ensemble_output_storage(arrays: Mapping[str, np.ndarray], *, sample_count: int) -> dict[str, Any]:
    seed_stacked = {
        "scores", "residual", "probabilities", "any_probability", "embeddings", "token_attention",
    }
    sample_first = {
        name: np.swapaxes(np.asarray(value), 0, 1) if name in seed_stacked else np.asarray(value)
        for name, value in arrays.items()
    }
    return array_storage_statistics(sample_first, sample_count=sample_count)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _load_models(
    checkpoint_paths: Sequence[str | Path], *, catalog: FeatureCatalog, device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(checkpoint_paths) != len(ENSEMBLE_SEEDS):
        raise ValueError("selected ensemble requires exactly three checkpoints")
    if len({str(Path(value).expanduser().resolve()) for value in checkpoint_paths}) != 3:
        raise ValueError("selected ensemble checkpoints must be distinct")
    loaded = []
    records = []
    reference_config: str | None = None
    reference_preprocessing: str | None = None
    for expected_seed, raw in zip(ENSEMBLE_SEEDS, checkpoint_paths, strict=True):
        path = Path(raw).expanduser().resolve()
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("status") != "complete" or artifact.get("kind") != "fcer_checkpoint":
            raise ValueError(f"invalid FCER checkpoint: {path}")
        if int(artifact.get("seed", -1)) != expected_seed:
            raise ValueError("selected ensemble checkpoint seed/order mismatch")
        config = dict(artifact["config"])
        config_json = canonical_json(config)
        if reference_config is None:
            reference_config = config_json
        elif config_json != reference_config:
            raise ValueError("selected ensemble checkpoint configurations differ")
        if int(config.get("head_dim", -1)) != len(catalog.schema["head_feature_names"]):
            raise ValueError("checkpoint/catalog head feature dimension mismatch")
        if int(config.get("depth_dim", -1)) != len(catalog.schema["depth_feature_names"]):
            raise ValueError("checkpoint/catalog depth feature dimension mismatch")
        mask = np.asarray(artifact["head_mask"], dtype=bool)
        if mask.shape != (len(catalog.schema["head_feature_names"]),) or not mask.any():
            raise ValueError("checkpoint head mask is incompatible with catalog")
        preprocessing = canonical_json({
            "head_mask": mask.tolist(),
            "normalizers": artifact["normalizers"],
        })
        if reference_preprocessing is None:
            reference_preprocessing = preprocessing
        elif preprocessing != reference_preprocessing:
            raise ValueError("selected ensemble checkpoint preprocessing differs")
        model = FullChainRanker(**model_kwargs(config))
        model.load_state_dict(artifact["state_dict"], strict=True)
        parameter_count = sum(int(value.numel()) for value in model.parameters())
        if parameter_count != int(artifact.get("parameter_count", -1)):
            raise ValueError("checkpoint parameter count does not match state_dict")
        model = model.to(device).eval()
        loaded.append({
            "model": model,
            "config": config,
            "normalizers": _normalizers_from_json(artifact["normalizers"]),
            "head_mask": mask,
            "seed": expected_seed,
            "sha256": sha256_file(path),
        })
        records.append({
            **artifact_identity(path),
            "seed": expected_seed,
            "parameter_count": parameter_count,
        })
    return loaded, records


@torch.inference_mode()
def _predict_loaded_ensemble(
    *,
    catalog: FeatureCatalog,
    sample_ids: set[str],
    priors: Mapping[str, np.ndarray],
    loaded_models: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    seed_outputs = []
    for loaded in loaded_models:
        collected: dict[str, list[np.ndarray]] = {
            "scores": [], "residual": [], "probabilities": [], "any_probability": [],
            "embeddings": [], "token_attention": [], "q": [],
        }
        ordered: list[str] = []
        candidate_ids: list[list[str]] = []
        candidate_checksums: list[list[str]] = []
        q_ranks: list[list[int]] = []
        for batch in catalog.iter_batches(
            allowed_ids=sample_ids,
            labels=None,
            priors=priors,
            normalizers=loaded["normalizers"],
            batch_size=batch_size,
            seed=0,
            shuffle=False,
            array_keys=required_array_keys(loaded["config"]),
        ):
            inputs, target = torch_feature_batch(
                batch, device=device, head_mask=loaded["head_mask"], config=loaded["config"],
            )
            if target is not None:
                raise AssertionError("efficiency inference unexpectedly received labels")
            prediction = loaded["model"](**inputs)
            for name, tensor in (
                ("scores", prediction["scores"]),
                ("residual", prediction["residual"]),
                ("probabilities", prediction["absolute_probability"]),
                ("any_probability", prediction["any_probability"]),
                ("embeddings", prediction["embedding"]),
                ("token_attention", prediction["token_attention"]),
                ("q", inputs["q"]),
            ):
                collected[name].append(tensor.detach().float().cpu().numpy())
            ordered.extend(map(str, batch["sample_ids"]))
            for sample_id, record in zip(batch["sample_ids"], batch["records"], strict=True):
                frozen = catalog.candidate_records.get(str(sample_id))
                if frozen is None:
                    raise ValueError(f"missing frozen candidate record: {sample_id}")
                candidates = frozen["candidates"]
                ids = [str(value["candidate_id"]) for value in candidates]
                checksums = [str(value["candidate_checksum"]) for value in candidates]
                ranks = [int(value["q_rank"]) for value in candidates]
                if ids != list(map(str, record["candidate_ids"])) or checksums != list(map(str, record["candidate_checksums"])):
                    raise ValueError(f"frozen candidate identity mismatch: {sample_id}")
                if len(set(ids)) != 5 or any(not value for value in checksums) or sorted(ranks) != list(range(5)):
                    raise ValueError(f"invalid five-candidate coverage: {sample_id}")
                candidate_ids.append(ids)
                candidate_checksums.append(checksums)
                q_ranks.append(ranks)
        if len(ordered) != len(sample_ids) or set(ordered) != sample_ids or len(set(ordered)) != len(ordered):
            raise ValueError("selected-model prediction did not exactly cover the requested cohort")
        seed_outputs.append({
            **{name: np.concatenate(values) for name, values in collected.items()},
            "sample_ids": np.asarray(ordered),
            "candidate_ids": np.asarray(candidate_ids),
            "candidate_checksums": np.asarray(candidate_checksums),
            "q_ranks": np.asarray(q_ranks, dtype=np.int64),
        })
    reference = seed_outputs[0]
    for observed in seed_outputs[1:]:
        for name in ("sample_ids", "candidate_ids", "candidate_checksums", "q_ranks", "q"):
            if not np.array_equal(observed[name], reference[name]):
                raise ValueError(f"selected ensemble changed frozen identity/evidence across seeds: {name}")
    return {
        "sample_ids": reference["sample_ids"],
        "candidate_ids": reference["candidate_ids"],
        "candidate_checksums": reference["candidate_checksums"],
        "q_ranks": reference["q_ranks"],
        "q": reference["q"],
        **{
            name: np.stack([value[name] for value in seed_outputs])
            for name in ("scores", "residual", "probabilities", "any_probability", "embeddings", "token_attention")
        },
    }


def run_selected_model_efficiency_audit(
    *,
    catalog: FeatureCatalog,
    sample_ids: set[str],
    priors: Mapping[str, np.ndarray],
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
    device: str = "auto",
    batch_size: int = 32,
    warmup: int = 2,
    repeat: int = 10,
    catalog_evidence: Mapping[str, Any] | None = None,
    prior_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit the locked three-seed FCER ensemble without loading any labels."""
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    ids = set(map(str, sample_ids))
    if not ids or len(ids) != len(sample_ids):
        raise ValueError("sample IDs must be non-empty and unique")
    if set(priors) != ids:
        raise ValueError("V2 prior lookup must exactly cover the requested cohort")
    for sample_id, value in priors.items():
        prior = np.asarray(value)
        if prior.shape != (5, 80) or not np.isfinite(prior).all():
            raise ValueError(f"invalid V2 prior for {sample_id}")
    locations = set(getattr(catalog, "locations", {}))
    if locations and not ids.issubset(locations):
        raise ValueError(f"feature catalog is missing requested samples: {sorted(ids - locations)[:5]}")
    identity = catalog.candidate_identity(ids)
    if set(identity) != ids:
        raise ValueError("feature catalog candidate identity coverage mismatch")
    for sample_id, value in identity.items():
        candidate_ids = list(map(str, value.get("candidate_ids", ())))
        candidate_checksums = list(map(str, value.get("candidate_checksums", ())))
        if len(candidate_ids) != 5 or len(set(candidate_ids)) != 5:
            raise ValueError(f"candidate identity must contain five unique candidates: {sample_id}")
        if len(candidate_checksums) != 5 or any(not item for item in candidate_checksums):
            raise ValueError(f"candidate identity must contain five non-empty checksums: {sample_id}")
    if int(batch_size) <= 0 or int(warmup) < 1 or int(repeat) <= 0:
        raise ValueError("batch_size/repeat must be positive and at least one warmup is required")

    torch_device = resolve_device(device)
    setup_started = time.perf_counter()
    loaded, checkpoints = _load_models(checkpoint_paths, catalog=catalog, device=torch_device)
    _synchronize(torch_device)
    setup_seconds = time.perf_counter() - setup_started
    memory_samples: list[dict[str, int]] = []
    initial_memory = _mps_memory_snapshot(torch_device)
    if initial_memory is not None:
        memory_samples.append(initial_memory)
    latest: dict[str, np.ndarray] | None = None

    def operation() -> None:
        nonlocal latest
        latest = _predict_loaded_ensemble(
            catalog=catalog,
            sample_ids=ids,
            priors=priors,
            loaded_models=loaded,
            device=torch_device,
            batch_size=int(batch_size),
        )
        snapshot = _mps_memory_snapshot(torch_device)
        if snapshot is not None:
            memory_samples.append(snapshot)

    efficiency = measure_efficiency(
        operation,
        sample_count=len(ids),
        warmup=int(warmup),
        repeat=int(repeat),
        synchronize=lambda: _synchronize(torch_device),
    )
    if latest is None:
        raise AssertionError("efficiency operation produced no output")
    latency = efficiency["latency_seconds"]
    efficiency["per_expression_latency"] = {
        "unit": "seconds_per_referring_expression",
        "median": latency["median"] / len(ids),
        "p95_nearest_rank": latency["p95_nearest_rank"] / len(ids),
        "mean": latency["mean"] / len(ids),
        "median_milliseconds": 1000.0 * latency["median"] / len(ids),
        "p95_milliseconds_nearest_rank": 1000.0 * latency["p95_nearest_rank"] / len(ids),
    }
    output_storage = _ensemble_output_storage(latest, sample_count=len(ids))
    catalog_paths = [
        *map(str, catalog.artifact_dirs),
        *map(str, catalog.head_override_dirs),
        *map(str, catalog.candidate_feature_paths),
    ]
    mps_max = None
    if memory_samples:
        keys = sorted(set().union(*(sample.keys() for sample in memory_samples)))
        mps_max = {key: max(sample[key] for sample in memory_samples if key in sample) for key in keys}
    parameter_counts = [int(value["parameter_count"]) for value in checkpoints]
    if len(set(parameter_counts)) != 1:
        raise ValueError("selected ensemble checkpoints have different parameter counts")
    report: dict[str, Any] = {
        "schema_version": "3.0.0",
        "kind": "v3_selected_model_efficiency_audit",
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "labels_read": False,
        "protocol": {
            "scope": "three-seed selected FCER ensemble feature loading, preprocessing and forward inference",
            "checkpoint_loading_in_timed_region": False,
            "setup_checkpoint_load_seconds": setup_seconds,
            "warmup": int(warmup),
            "repeat": int(repeat),
            "batch_size": int(batch_size),
            "requested_device": device,
            "resolved_device": str(torch_device),
            "candidate_count_per_expression": 5,
            "cohort_reused_for_repeats": True,
        },
        "coverage": {
            "expression_count": len(ids),
            "unique_expression_count": len(ids),
            "candidate_count": len(ids) * 5,
            "unique_candidates_per_expression": True,
            "candidate_checksums_nonempty": True,
            "q_ranks_are_permutations_0_to_4": True,
            "ensemble_identity_equal_across_seeds": True,
            "v2_prior_exact_coverage": True,
        },
        "efficiency": efficiency,
        "memory": {
            "peak_rss": efficiency["rss"],
            "mps_unified_memory_boundary_maximum": mps_max,
            "limitations": [
                "ru_maxrss is the process-lifetime peak and may include setup or earlier allocations.",
                "PyTorch exposes MPS current/driver allocation at observation boundaries, not a true operation peak.",
                "CPU has no separate accelerator unified-memory counter; MPS fields are null on CPU.",
            ],
        },
        "model": {
            "ensemble_size": len(checkpoints),
            "parameter_count_per_seed": parameter_counts[0],
            "parameter_count_all_seed_instances": sum(parameter_counts),
            "checkpoints": checkpoints,
        },
        "disk": {
            "checkpoints": _path_storage(checkpoint_paths),
            "feature_catalog": _path_storage(catalog_paths),
            "in_memory_output": output_storage,
            "output_bytes": output_storage["total_bytes"],
        },
        "inputs": {
            "catalog": dict(catalog.provenance()) if catalog_evidence is None else dict(catalog_evidence),
            "v2_prior": {} if prior_evidence is None else dict(prior_evidence),
            "sample_ids_sha256": sha256_bytes("\n".join(sorted(ids)).encode("utf-8")),
        },
    }
    if not all(math.isfinite(float(value)) for value in (
        efficiency["per_expression_latency"]["median"],
        efficiency["per_expression_latency"]["p95_nearest_rank"],
        efficiency["throughput_samples_per_second"]["mean"],
    )):
        raise FloatingPointError("non-finite efficiency result")
    report["content_sha256"] = sha256_bytes(canonical_json(report).encode("utf-8"))
    atomic_write_json(output, report)
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a label-free efficiency audit of a selected FCER ensemble.")
    parser.add_argument("--catalog-descriptor", required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--v2-priors", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    catalog, catalog_evidence = load_feature_catalog_descriptor(args.catalog_descriptor)
    sample_ids = load_sample_ids(args.sample_ids)
    priors, prior_evidence = load_v2_prior_npz(args.v2_priors, sample_ids=sample_ids)
    report = run_selected_model_efficiency_audit(
        catalog=catalog,
        sample_ids=sample_ids,
        priors=priors,
        checkpoint_paths=args.checkpoint,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeat=args.repeat,
        catalog_evidence=catalog_evidence,
        prior_evidence=prior_evidence,
    )
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "content_sha256": report["content_sha256"],
        "median_ms_per_expression": report["efficiency"]["per_expression_latency"]["median_milliseconds"],
        "p95_ms_per_expression": report["efficiency"]["per_expression_latency"]["p95_milliseconds_nearest_rank"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
