"""Truthful runtime profiling with explicit full-pipeline completeness gates.

The frozen artifacts allow several deployment components to be measured without
re-running expensive proposal networks.  A route receives a deployment total
only if every pre-registered stage is executed.  Cache/component timings are
still useful, but are deliberately marked partial and never relabelled as an
end-to-end latency.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import pickle
import pickletools
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil

from .io import atomic_csv, atomic_json, atomic_parquet, atomic_text


FOUR_D_STAGES = (
    "input_io",
    "preprocess",
    "visual_grounding_or_e2e_model",
    "grasp_backend",
    "candidate_decode",
    "nms",
    "feature_extraction",
    "reranker",
    "gate",
    "final_selection",
)
SIX_D_STAGES = (
    "input_io",
    "mask_inference",
    "depth_backprojection",
    "workspace_construction",
    "tsdf_integration",
    "vgn_inference",
    "candidate_decode",
    "pose_nms",
    "feature_extraction",
    "reranker",
    "gate",
    "final_selection",
)


@dataclass(frozen=True)
class RouteSource:
    route: str
    dimension: str
    device: str
    candidate_path: Path
    feature_path: Path | None
    prediction_path: Path | None
    retrospective: bool = False
    condition: str | None = None


def synchronize_device(device: str) -> None:
    """Wait for asynchronous work on a timed accelerator, if present."""

    name = str(device).lower()
    if name == "cpu":
        return
    import torch

    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS synchronisation requested but MPS is unavailable")
        torch.mps.synchronize()
        return
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA synchronisation requested but CUDA is unavailable")
        torch.cuda.synchronize()
        return
    raise ValueError(f"unsupported timing device: {device}")


def timed_call_ns(function: Callable[[], Any], *, device: str = "cpu") -> tuple[int, Any]:
    synchronize_device(device)
    start = time.perf_counter_ns()
    value = function()
    synchronize_device(device)
    elapsed = time.perf_counter_ns() - start
    if elapsed < 0:
        raise AssertionError("monotonic timer returned a negative interval")
    return int(elapsed), value


def deployment_stage_total_ns(frame: pd.DataFrame) -> int:
    """Sum only explicitly deployment-marked stages for one sample."""

    required = {"elapsed_ns", "deployment_stage"}
    if not required.issubset(frame.columns):
        raise ValueError(f"timing rows miss columns: {sorted(required - set(frame.columns))}")
    values = frame.loc[frame["deployment_stage"].astype(bool), "elapsed_ns"].astype(np.int64)
    if (values < 0).any():
        raise ValueError("timing rows contain a negative interval")
    return int(values.sum())


def stage_sum_matches_total(
    frame: pd.DataFrame, total_ns: int, *, relative_tolerance: float = 0.01
) -> bool:
    if int(total_ns) < 0 or not 0 <= float(relative_tolerance) < 1:
        raise ValueError("invalid total or tolerance")
    observed = deployment_stage_total_ns(frame)
    tolerance = max(1, int(round(int(total_ns) * float(relative_tolerance))))
    return abs(observed - int(total_ns)) <= tolerance


def _sequence_id(scene_id: str) -> str:
    match = re.search(r"(?:^|/)(seq[^/,]+)", str(scene_id), flags=re.IGNORECASE)
    if match is None:
        return str(scene_id).rsplit(",", 1)[0]
    prefix = str(scene_id)[: match.end(1)]
    return prefix.lstrip("/")


def _round_robin_sample(
    frame: pd.DataFrame, cluster_column: str, count: int
) -> pd.DataFrame:
    work = frame.sort_values([cluster_column, "sample_id"], kind="mergesort").copy()
    groups = [part for _, part in work.groupby(cluster_column, sort=True)]
    selected: list[pd.Series] = []
    offset = 0
    while len(selected) < min(count, len(work)):
        changed = False
        for group in groups:
            if offset < len(group):
                selected.append(group.iloc[offset])
                changed = True
                if len(selected) == min(count, len(work)):
                    break
        if not changed:
            break
        offset += 1
    return pd.DataFrame(selected).reset_index(drop=True)


def build_subset_manifest(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    four_manifest_path = (
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet"
    )
    paired = pd.read_parquet(four_manifest_path)
    paired["sequence_id"] = paired["scene_id"].map(_sequence_id)
    four = _round_robin_sample(paired, "sequence_id", 100)

    input_roots = list(
        (
            repo_root
            / "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart/analysis_inputs/oracle_gt_mask"
        ).glob("*/test_group_universe.parquet")
    )
    if len(input_roots) != 1:
        raise FileNotFoundError("cannot resolve unique frozen 6D test group universe")
    six = pd.read_parquet(input_roots[0]).rename(columns={"group_id": "sample_id"})
    six = _round_robin_sample(six, "scene_id", 100)

    manifest = {
        "schema_version": 1,
        "selection_uses_runtime": False,
        "selection_uses_reranking_outcome": False,
        "four_d": {
            "count": len(four),
            "sequence_count": int(four["sequence_id"].nunique()),
            "sample_ids": four["sample_id"].astype(str).tolist(),
            "sequence_ids": four["sequence_id"].astype(str).tolist(),
            "source_manifest": str(four_manifest_path.resolve()),
        },
        "six_d": {
            "count": len(six),
            "scene_count": int(six["scene_id"].nunique()),
            "group_ids": six["sample_id"].astype(str).tolist(),
            "scene_ids": six["scene_id"].astype(str).tolist(),
            "source_manifest": str(input_roots[0].resolve()),
        },
        "note": (
            "Deterministic round-robin cluster sampling. Success/failure was not used "
            "because the protocol forbids outcome-driven subset selection."
        ),
    }
    atomic_json(output_dir / "subset_manifest.json", manifest)
    atomic_json(output_dir / "profile_subset_manifest.json", manifest)
    return manifest


def _route_sources(repo_root: Path) -> list[RouteSource]:
    primary = repo_root / "runs/fair_unified_reranking_20260809_103012"
    candidates = primary / "02_candidates"
    features = primary / "03_features/tracks/T2_matched_common"
    score_root = primary / "08_lock/label_free_test_rankers"
    routes = [
        RouteSource(
            route=name.upper(),
            dimension="4D",
            device="cpu",
            candidate_path=candidates / f"{name}_test_top5.parquet",
            feature_path=features / f"{name}_test/candidate_features.parquet",
            prediction_path=score_root / name / "per_candidate_scores.parquet",
        )
        for name in ("crog", "g1", "c1")
    ]
    d1 = repo_root / "runs/fair_d1_reranking_extension_20260811T145515Z"
    routes.append(
        RouteSource(
            route="D1",
            dimension="4D",
            device="cpu",
            candidate_path=d1 / "02_candidates/d1_test_top5.parquet",
            feature_path=d1
            / "03_features/test/top5/matched_common_raw/candidate_features.parquet",
            prediction_path=d1
            / "08_lock/label_free_test_rankers/d1/per_candidate_scores.parquet",
            retrospective=True,
        )
    )
    six_root = (
        repo_root
        / "artifacts/graspnet6d/20260819_221819_graspnet6d_vgn_lambdamart"
    )
    routes.append(
        RouteSource(
            route="6D_ORACLE",
            dimension="6D",
            device="cpu",
            candidate_path=six_root / "candidate_features.parquet",
            feature_path=six_root / "candidate_features.parquet",
            prediction_path=six_root / "reranked_predictions.parquet",
            condition="oracle_gt_mask",
        )
    )
    return routes


def _read_inputs_for_samples(
    manifest: pd.DataFrame, sample_ids: set[str]
) -> tuple[list[bytes], list[bytes]]:
    rows = manifest.loc[manifest["sample_id"].astype(str).isin(sample_ids)]
    rgb = [Path(path).read_bytes() for path in rows["source_rgb_path"]]
    depth = [Path(path).read_bytes() for path in rows["source_depth_path"]]
    return rgb, depth


def _decode_inputs(rgb: bytes, depth: bytes) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    colour = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
    depth_image = cv2.imdecode(
        np.frombuffer(depth, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if colour is None or depth_image is None:
        raise RuntimeError("OpenCV failed to decode a profile input")
    return colour.astype(np.float32) / 255.0, depth_image.astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_completion_valid(
    repo_root: Path, run_dir: Path
) -> dict[str, Any] | None:
    """Return a verified partial-profile completion record for safe resume."""

    output = run_dir / "runtime_profile"
    completion_path = output / "completion.json"
    if not completion_path.is_file():
        return None
    try:
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        signature = payload.pop("completion_signature_sha256")
        if signature != _sha256_json(payload):
            return None
        if payload.get("status") != "PARTIAL_RUNTIME_PROFILE":
            return None
        if payload.get("runtime_code_sha256") != _sha256_file(Path(__file__)):
            return None
        for relative, expected in payload.get("output_hashes", {}).items():
            path = output / relative
            if not path.is_file() or _sha256_file(path) != expected:
                return None
        for relative, expected in payload.get("source_hashes", {}).items():
            path = repo_root / relative
            if not path.is_file() or _sha256_file(path) != expected:
                return None
        for relative, expected in payload.get("run_local_hashes", {}).items():
            path = run_dir / relative
            if not path.is_file() or _sha256_file(path) != expected:
                return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    payload["completion_signature_sha256"] = signature
    return payload


def _verified_record_path(record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError(f"{label} is not a path/SHA-256 artifact record")
    path = Path(str(record["path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = _sha256_file(path)
    if observed != str(record["sha256"]):
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {record['sha256']}, observed {observed}"
        )
    return path


def _load_native_lightgbm_booster(path: Path) -> Any:
    """Load a native LightGBM model without executing a legacy pickle.

    The formal 4-DoF checkpoints predate native ``.txt`` persistence.  Their
    pickles contain exactly one portable LightGBM model string.  Parsing pickle
    opcodes is data-only; unlike ``pickle.load``, it does not invoke arbitrary
    constructors.
    """

    import lightgbm as lgb

    if path.suffix == ".txt":
        booster = lgb.Booster(model_file=str(path))
    else:
        model_strings = [
            argument
            for _opcode, argument, _position in pickletools.genops(path.read_bytes())
            if isinstance(argument, str) and argument.startswith("tree\nversion=")
        ]
        if len(model_strings) != 1:
            raise RuntimeError(
                "formal LambdaMART pickle does not contain exactly one native model string"
            )
        booster = lgb.Booster(model_str=model_strings[0])
    if int(booster.num_feature()) <= 0 or int(booster.num_trees()) <= 0:
        raise RuntimeError("formal LambdaMART checkpoint is empty")
    return booster


@dataclass(frozen=True)
class _RankerBundle:
    seeds: tuple[int, ...]
    preprocessors: tuple[Any, ...]
    boosters: tuple[Any, ...]
    model_paths: tuple[Path, ...]
    partition_calibrators: tuple[dict[str, Any] | None, ...]


def _load_formal_ranker_bundle(repo_root: Path, source: RouteSource) -> _RankerBundle:
    """Resolve the formally selected three-seed ranker through locked manifests."""

    # LightGBM must initialise its OpenMP runtime before importing the dataset
    # module (which imports Torch/OpenMP) on macOS.  Reversing this order can
    # crash in LightGBM's C API instead of raising a Python exception.
    import lightgbm  # noqa: F401

    from unified_reranking.datasets import FoldPreprocessor

    route = source.route.lower()
    preprocessors: list[Any] = []
    boosters: list[Any] = []
    model_paths: list[Path] = []
    partition_calibrators: list[dict[str, Any] | None] = []
    seeds: list[int] = []
    if route == "d1":
        manifest_path = (
            repo_root
            / "runs/fair_d1_reranking_extension_20260811T145515Z/08_lock/label_free_test_rankers/d1/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE" or manifest.get("selected_method") != "R5":
            raise RuntimeError("D1 formal ranker manifest is not the locked R5 ensemble")
        raw_feature_path = _verified_record_path(
            manifest.get("sources", {}).get("raw_features"),
            label="D1 formal raw Test features",
        )
        if source.feature_path is None or raw_feature_path != source.feature_path.resolve():
            raise RuntimeError("D1 runtime feature source is not the locked raw feature table")
        entries = manifest.get("seed_applications", {})
        for seed in (42, 123, 2026):
            entry = entries.get(str(seed))
            if not isinstance(entry, dict):
                raise RuntimeError(f"D1 formal ranker misses seed {seed}")
            cell_path = _verified_record_path(
                entry.get("cell"), label=f"D1 seed-{seed} Validation cell"
            )
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            configuration = cell.get("configuration", {})
            if (
                cell.get("status") != "COMPLETE"
                or configuration.get("method") != "R5"
                or configuration.get("encoder") != "lambdamart"
                or int(configuration.get("seed", -1)) != seed
            ):
                raise RuntimeError(f"D1 seed-{seed} cell is not the locked R5 cell")
            model_path = _verified_record_path(entry.get("model"), label=f"D1 seed-{seed} model")
            preprocessor_path = _verified_record_path(
                entry.get("preprocessor"), label=f"D1 seed-{seed} preprocessor"
            )
            artifact = json.loads(preprocessor_path.read_text(encoding="utf-8"))
            if artifact != cell.get("preprocessor"):
                raise RuntimeError(f"D1 seed-{seed} persisted preprocessor differs from cell")
            preprocessors.append(FoldPreprocessor.from_artifact(artifact))
            boosters.append(_load_native_lightgbm_booster(model_path))
            model_paths.append(model_path)
            calibrator = configuration.get("fold_local_calibrator")
            if not isinstance(calibrator, dict):
                raise RuntimeError(f"D1 seed-{seed} fold-local calibrator is absent")
            partition_calibrators.append(calibrator)
            seeds.append(seed)
    elif route in {"crog", "g1", "c1"}:
        lock = repo_root / "runs/fair_unified_reranking_20260809_103012/08_lock"
        manifest_path = lock / "label_free_test_rankers" / route / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = manifest.get("configuration", {})
        configured_seeds = tuple(map(int, configuration.get("seeds", ())))
        if (
            manifest.get("status") != "COMPLETE"
            or configuration.get("encoder") != "lambdamart"
            or configured_seeds != (42, 123, 2026)
        ):
            raise RuntimeError(f"{source.route} formal ranker is not the locked ensemble")
        applications = manifest.get("sources", {}).get("applications", {})
        for seed in configured_seeds:
            application_path = _verified_record_path(
                applications.get(str(seed)),
                label=f"{source.route} seed-{seed} application manifest",
            )
            application = json.loads(application_path.read_text(encoding="utf-8"))
            cell_path = _verified_record_path(
                application.get("sources", {}).get("cell_manifest"),
                label=f"{source.route} seed-{seed} Validation cell",
            )
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            cell_configuration = cell.get("configuration", {})
            if (
                cell.get("status") != "COMPLETE"
                or cell_configuration.get("mode") != "validation"
                or cell_configuration.get("encoder") != "lambdamart"
                or int(cell_configuration.get("seed", -1)) != seed
            ):
                raise RuntimeError(f"{source.route} seed-{seed} cell is not Validation-locked")
            model_path = _verified_record_path(
                application.get("sources", {}).get("model"),
                label=f"{source.route} seed-{seed} model",
            )
            cell_model = cell.get("artifacts", {}).get("model", {})
            if (
                Path(str(cell_model.get("path", ""))).resolve() != model_path
                or cell_model.get("sha256") != _sha256_file(model_path)
            ):
                raise RuntimeError(f"{source.route} seed-{seed} model chain differs")
            feature_path = _verified_record_path(
                application.get("sources", {}).get("features"),
                label=f"{source.route} formal Test features",
            )
            if source.feature_path is None or feature_path != source.feature_path.resolve():
                raise RuntimeError(f"{source.route} runtime feature source is not formal T2")
            preprocessors.append(FoldPreprocessor.from_artifact(cell["preprocessor"]))
            boosters.append(_load_native_lightgbm_booster(model_path))
            model_paths.append(model_path)
            partition_calibrators.append(None)
            seeds.append(seed)
    else:
        raise ValueError(f"no formal 4D ranker bundle for {source.route}")
    return _RankerBundle(
        seeds=tuple(seeds),
        preprocessors=tuple(preprocessors),
        boosters=tuple(boosters),
        model_paths=tuple(model_paths),
        partition_calibrators=tuple(partition_calibrators),
    )


def _ranker_scores_for_group(
    bundle: _RankerBundle, group: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    ordered = group.sort_values(["native_rank", "candidate_id"], kind="mergesort")
    seed_scores: list[np.ndarray] = []
    for preprocessor, booster, calibrator in zip(
        bundle.preprocessors,
        bundle.boosters,
        bundle.partition_calibrators,
        strict=True,
    ):
        if calibrator is None:
            matrix = preprocessor.transform(ordered)
        else:
            # D1's locked R5 cells apply a seed/fold-local calibrator before
            # their persisted Train-only preprocessor.  Replaying the final
            # feature table directly is not equivalent and must fail parity.
            from d1_reranking.fold_calibration import apply_partition_calibrator
            from unified_reranking.datasets import build_inference_query_arrays

            calibrated = apply_partition_calibrator(ordered, calibrator)
            arrays = build_inference_query_arrays(
                calibrated, preprocessor=preprocessor, max_candidates=5
            )
            candidate_ids = tuple(ordered["candidate_id"].astype(str))
            if arrays.candidate_ids != (candidate_ids,):
                raise RuntimeError("D1 ranker preprocessing changed candidate order")
            matrix = arrays.features.numpy()[~arrays.padding_mask.numpy()]
        scores = np.asarray(booster.predict(matrix, num_threads=1), dtype=np.float64)
        if scores.shape != (len(ordered),) or not np.isfinite(scores).all():
            raise RuntimeError("formal LambdaMART returned invalid group scores")
        seed_scores.append(scores)
    stacked = np.column_stack(seed_scores)
    return stacked, stacked.mean(axis=1)


@dataclass(frozen=True)
class _GateBundle:
    model: Any
    feature_columns: tuple[str, ...]
    operating_point: Any | None
    inputs: pd.DataFrame
    decisions: pd.DataFrame
    model_path: Path


def _load_formal_gate_bundle(repo_root: Path, source: RouteSource) -> _GateBundle:
    from unified_reranking.gate import (
        ConservativeTransitionModel,
        GateOperatingPoint,
    )

    route = source.route.lower()
    if route == "d1":
        run = repo_root / "runs/fair_d1_reranking_extension_20260811T145515Z"
        selection_path = run / "07_validation/gate/d1/gate_selection.json"
    elif route in {"crog", "g1", "c1"}:
        run = repo_root / "runs/fair_unified_reranking_20260809_103012"
        selection_path = run / "08_lock/gates" / route / "gate_selection.json"
    else:
        raise ValueError(f"no formal 4D gate bundle for {source.route}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    configured_route = str(selection.get("configuration", {}).get("route", "")).lower()
    if selection.get("status") != "COMPLETE" or configured_route != route:
        raise RuntimeError(f"{source.route} gate is not the completed locked selection")
    model_path = _verified_record_path(
        selection.get("artifacts", {}).get("transition_model"),
        label=f"{source.route} transition gate",
    )
    # This is a byte-verified, repository-owned formal checkpoint.  Unlike the
    # legacy LightGBM pickle above, sklearn's calibrated ensemble has no safe
    # portable text representation, so the repository's own inference path
    # necessarily uses pickle for this one trusted artifact.
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    if not isinstance(model, ConservativeTransitionModel):
        raise RuntimeError(f"{source.route} transition gate has an unexpected type")
    feature_columns = tuple(selection["configuration"]["feature_columns"])
    if tuple(model.feature_names_ or ()) != feature_columns:
        raise RuntimeError(f"{source.route} gate model/schema mismatch")
    point_payload = selection.get("selection", {}).get("selected_operating_point")
    point = None if point_payload is None else GateOperatingPoint(**point_payload)
    gate_root = run / "08_lock/label_free_test_gates" / route
    manifest = json.loads((gate_root / "manifest.json").read_text(encoding="utf-8"))
    input_path = _verified_record_path(
        manifest.get("artifacts", {}).get("inputs"),
        label=f"{source.route} formal gate inputs",
    )
    decision_path = _verified_record_path(
        manifest.get("artifacts", {}).get("decisions"),
        label=f"{source.route} formal gate decisions",
    )
    return _GateBundle(
        model=model,
        feature_columns=feature_columns,
        operating_point=point,
        inputs=pd.read_parquet(input_path),
        decisions=pd.read_parquet(decision_path),
        model_path=model_path,
    )


def _gate_result_for_row(bundle: _GateBundle, row: pd.DataFrame) -> tuple[float, float, bool]:
    from unified_reranking.gate import GateEvidence, gate_switch_mask

    matrix = row.loc[:, bundle.feature_columns].to_numpy(dtype=np.float64)
    recover, harm = bundle.model.predict_probabilities(matrix)
    if bundle.operating_point is None:
        switch = False
    else:
        evidence = GateEvidence(
            score_margin=row["score_margin"].to_numpy(),
            challenger_reliability=row["challenger_reliability"].to_numpy(),
            perturbation_stability=row["perturbation_stability"].to_numpy(),
            seed_challenger_votes=row["seed_challenger_votes"].to_numpy(),
            candidate_id_unchanged=row["candidate_id_unchanged"].to_numpy(),
            geometry_hash_unchanged=row["geometry_hash_unchanged"].to_numpy(),
            challenger_exists=row["challenger_exists"].to_numpy(),
        )
        switch = bool(
            gate_switch_mask(recover, harm, evidence, bundle.operating_point)[0]
        )
    return float(recover[0]), float(harm[0]), switch


def _timing_row(
    source: RouteSource,
    sample_id: str,
    stage: str,
    elapsed_ns: int,
    *,
    status: str,
    deployment_stage: bool,
    variant: str,
    cache_policy: str = "warm_in_memory",
    candidate_count: int | None = None,
) -> dict[str, Any]:
    return {
        "route": source.route,
        "dimension": source.dimension,
        "device": source.device,
        "sample_id": str(sample_id),
        "cache_policy": cache_policy,
        "stage": stage,
        "elapsed_ns": int(elapsed_ns),
        "elapsed_ms": int(elapsed_ns) / 1e6,
        "deployment_stage": bool(deployment_stage),
        "status": status,
        "variant": variant,
        "batch_size": 1,
        "warmup_samples": 5,
        "candidate_count": (
            int(candidate_count) if candidate_count is not None else np.nan
        ),
    }


def _load_feature_assets(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and verify frozen probability/mask/depth for a common-feature call."""

    import cv2
    from PIL import Image

    for path_column, hash_column, label in (
        ("predicted_probability_path", "predicted_probability_sha256", "probability"),
        ("predicted_mask_path", "predicted_mask_sha256", "mask"),
        ("source_depth_path", "source_depth_sha256", "depth"),
    ):
        path = Path(str(row[path_column])).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != str(row[hash_column]):
            raise RuntimeError(f"frozen feature {label} artifact failed hash validation")
    loaded = np.load(Path(str(row["predicted_probability_path"])), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if loaded.files != ["probability"]:
                raise ValueError("unexpected probability archive schema")
            probability = np.asarray(loaded["probability"], dtype=np.float32)
        finally:
            loaded.close()
    else:
        probability = np.asarray(loaded, dtype=np.float32)
    mask = np.asarray(Image.open(str(row["predicted_mask_path"]))) > 0
    depth_mm = np.asarray(Image.open(str(row["source_depth_path"])))
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2 or mask.shape != depth_mm.shape:
        raise ValueError("frozen 4D mask/depth contract differs from formal extraction")
    if probability.shape != depth_mm.shape:
        probability = cv2.resize(
            probability,
            (depth_mm.shape[1], depth_mm.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
    if (
        probability.ndim != 2
        or not np.isfinite(probability).all()
        or probability.min(initial=0.0) < 0.0
        or probability.max(initial=0.0) > 1.0
    ):
        raise ValueError("frozen probability violates the formal extractor contract")
    return probability, mask, depth_mm.astype(np.float32) / np.float32(1000.0)


def _common_extractor_hash(repo_root: Path, source: RouteSource) -> str:
    if source.route == "D1":
        manifest_path = (
            repo_root
            / "runs/fair_d1_reranking_extension_20260811T145515Z/03_features/test/top5/matched_common_raw/manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["sources"]["extractor"]["shared_numerical_extractor"]
        _verified_record_path(record, label="D1 shared numerical extractor")
        return str(record["sha256"])
    manifest_path = (
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/03_features/common"
        / f"{source.route.lower()}_test/feature_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest["feature_extractor_sha256"])


def _profile_four_d_learned_components(
    repo_root: Path,
    source: RouteSource,
    four_manifest: pd.DataFrame,
    source_ids: list[str],
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure exact available 4D components without running a grasp network."""

    result_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"component_failures": {}}
    candidate_groups = {
        str(key): part for key, part in candidates.groupby("sample_id", sort=False)
    }
    feature_groups = {
        str(key): part for key, part in features.groupby("sample_id", sort=False)
    }
    prediction_groups = {
        str(key): part for key, part in predictions.groupby("sample_id", sort=False)
    }
    available_ids = [
        sample_id
        for sample_id in source_ids
        if sample_id in candidate_groups
        and sample_id in feature_groups
        and sample_id in prediction_groups
    ]
    metadata["learned_component_sample_count"] = len(available_ids)
    metadata["batch_size"] = 1
    metadata["warmup_samples"] = min(5, len(available_ids))
    if len(available_ids) != len(source_ids):
        metadata["component_failures"]["coverage"] = (
            f"only {len(available_ids)}/{len(source_ids)} common-subset samples have "
            "candidate, feature, and prediction rows"
        )

    feature_component_rows: list[dict[str, Any]] = []
    try:
        from unified_reranking.feature_extractors.common import extract_common_evidence

        extractor_path = repo_root / "src/unified_reranking/feature_extractors/common.py"
        expected_extractor = _common_extractor_hash(repo_root, source)
        if _sha256_file(extractor_path) != expected_extractor:
            raise RuntimeError("current common feature extractor differs from frozen source")
        manifest_by_id = (
            four_manifest.assign(sample_id=four_manifest["sample_id"].astype(str))
            .drop_duplicates("sample_id")
            .set_index("sample_id")
        )
        assets = {
            sample_id: _load_feature_assets(manifest_by_id.loc[sample_id])
            for sample_id in available_ids
        }

        def extract(sample_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
            probability, mask, depth = assets[sample_id]
            return extract_common_evidence(
                candidate_groups[sample_id],
                probability=probability,
                binary_mask=mask,
                depth_m=depth,
            )

        for sample_id in available_ids[:5]:
            extract(sample_id)
        for sample_id in available_ids:
            elapsed, extracted = timed_call_ns(
                lambda sample_id=sample_id: extract(sample_id), device=source.device
            )
            extracted_ids = set(extracted[0]["candidate_id"].astype(str))
            expected_ids = set(candidate_groups[sample_id]["candidate_id"].astype(str))
            if extracted_ids != expected_ids:
                raise RuntimeError("common feature extraction changed candidate membership")
            feature_component_rows.append(
                _timing_row(
                    source,
                    sample_id,
                    "feature_extraction_common_component",
                    elapsed,
                    status="MEASURED_EXACT_FROZEN_COMMON_EXTRACTOR_COMPONENT_ONLY",
                    deployment_stage=False,
                    variant="native_plus_feature_extraction",
                    candidate_count=len(candidate_groups[sample_id]),
                )
            )
        metadata["common_feature_component_status"] = "MEASURED_COMPONENT_ONLY"
        metadata["common_feature_component_sample_count"] = len(available_ids)
        metadata["common_feature_extractor_sha256"] = expected_extractor
        result_rows.extend(feature_component_rows)
    except Exception as error:  # fail closed per component, retain exact reason
        metadata["component_failures"]["feature_extraction_common_component"] = (
            f"{type(error).__name__}: {error}"
        )

    actual_scores: dict[str, np.ndarray] = {}
    actual_candidate_ids: dict[str, np.ndarray] = {}
    ranker_component_rows: list[dict[str, Any]] = []
    try:
        ranker = _load_formal_ranker_bundle(repo_root, source)
        for sample_id in available_ids[:5]:
            _ranker_scores_for_group(ranker, feature_groups[sample_id])
        for sample_id in available_ids:
            ordered = feature_groups[sample_id].sort_values(
                ["native_rank", "candidate_id"], kind="mergesort"
            )
            elapsed, values = timed_call_ns(
                lambda sample_id=sample_id: _ranker_scores_for_group(
                    ranker, feature_groups[sample_id]
                ),
                device=source.device,
            )
            seed_values, ensemble = values
            locked = prediction_groups[sample_id].sort_values(
                ["native_rank", "candidate_id"], kind="mergesort"
            )
            if not np.array_equal(
                ordered["candidate_id"].astype(str).to_numpy(),
                locked["candidate_id"].astype(str).to_numpy(),
            ):
                raise RuntimeError("ranker inference candidate order differs from locked scores")
            for index, seed in enumerate(ranker.seeds):
                expected_column = f"score_seed_{seed}"
                if expected_column not in locked or not np.allclose(
                    seed_values[:, index],
                    locked[expected_column].to_numpy(float),
                    rtol=1e-10,
                    atol=1e-10,
                ):
                    raise RuntimeError(f"seed-{seed} ranker inference failed score parity")
            if not np.allclose(
                ensemble,
                locked["ensemble_score"].to_numpy(float),
                rtol=1e-10,
                atol=1e-10,
            ):
                raise RuntimeError("formal ensemble inference failed score parity")
            actual_scores[sample_id] = ensemble
            actual_candidate_ids[sample_id] = ordered["candidate_id"].astype(str).to_numpy()
            ranker_component_rows.append(
                _timing_row(
                    source,
                    sample_id,
                    "reranker",
                    elapsed,
                    status="MEASURED_EXACT_LOCKED_THREE_SEED_ENSEMBLE",
                    deployment_stage=True,
                    variant="native_plus_raw_reranker",
                    candidate_count=len(candidate_groups[sample_id]),
                )
            )
        for sample_id in available_ids[:5]:
            int(np.argmax(actual_scores[sample_id]))
        for sample_id in available_ids:
            elapsed, selected_index = timed_call_ns(
                lambda sample_id=sample_id: int(np.argmax(actual_scores[sample_id])),
                device=source.device,
            )
            selected_id = actual_candidate_ids[sample_id][selected_index]
            locked = prediction_groups[sample_id]
            expected = (
                locked.sort_values(
                    ["ensemble_score", "native_rank", "candidate_id"],
                    ascending=[False, True, True],
                    kind="mergesort",
                )
                .iloc[0]["candidate_id"]
            )
            if str(selected_id) != str(expected):
                raise RuntimeError("raw final selection differs from locked decision")
            ranker_component_rows.append(
                _timing_row(
                    source,
                    sample_id,
                    "final_selection",
                    elapsed,
                    status="MEASURED_FROM_EXACT_LOCKED_ENSEMBLE_OUTPUT",
                    deployment_stage=True,
                    variant="native_plus_raw_reranker",
                    candidate_count=len(candidate_groups[sample_id]),
                )
            )
        metadata["ranker_status"] = "MEASURED_EXACT_LOCKED_ENSEMBLE"
        metadata["ranker_sample_count"] = len(available_ids)
        metadata["ranker_seed_count"] = len(ranker.seeds)
        metadata["ranker_model_paths"] = [str(path) for path in ranker.model_paths]
        result_rows.extend(ranker_component_rows)
    except Exception as error:  # fail closed; never substitute frozen score lookup
        metadata["component_failures"]["reranker"] = f"{type(error).__name__}: {error}"

    gate_component_rows: list[dict[str, Any]] = []
    try:
        gate = _load_formal_gate_bundle(repo_root, source)
        gate_inputs = {
            str(key): part for key, part in gate.inputs.groupby("sample_id", sort=False)
        }
        gate_decisions = gate.decisions.set_index(gate.decisions["sample_id"].astype(str))
        gate_ids = [sample_id for sample_id in available_ids if sample_id in gate_inputs]
        if len(gate_ids) != len(source_ids):
            raise RuntimeError(
                f"formal gate inputs cover {len(gate_ids)}/{len(source_ids)} profile samples"
            )
        for sample_id in gate_ids[:5]:
            _gate_result_for_row(gate, gate_inputs[sample_id])
        gate_outputs: dict[str, tuple[float, float, bool]] = {}
        for sample_id in gate_ids:
            elapsed, output = timed_call_ns(
                lambda sample_id=sample_id: _gate_result_for_row(
                    gate, gate_inputs[sample_id]
                ),
                device=source.device,
            )
            expected = gate_decisions.loc[sample_id]
            if isinstance(expected, pd.DataFrame) or not (
                np.isclose(output[0], float(expected["probability_recover"]), rtol=1e-10, atol=1e-10)
                and np.isclose(output[1], float(expected["probability_harm"]), rtol=1e-10, atol=1e-10)
                and output[2] == bool(expected["switch"])
            ):
                raise RuntimeError("gate inference failed locked decision parity")
            gate_outputs[sample_id] = output
            gate_component_rows.append(
                _timing_row(
                    source,
                    sample_id,
                    "gate",
                    elapsed,
                    status="MEASURED_EXACT_LOCKED_TRANSITION_GATE",
                    deployment_stage=True,
                    variant="native_plus_reranker_plus_gate",
                    candidate_count=len(candidate_groups[sample_id]),
                )
            )

        def gated_selection(sample_id: str) -> str:
            row = gate_inputs[sample_id].iloc[0]
            return str(
                row["challenger_candidate_id"]
                if gate_outputs[sample_id][2]
                else row["native_candidate_id"]
            )

        for sample_id in gate_ids[:5]:
            gated_selection(sample_id)
        for sample_id in gate_ids:
            elapsed, selected = timed_call_ns(
                lambda sample_id=sample_id: gated_selection(sample_id),
                device=source.device,
            )
            if selected != str(gate_decisions.loc[sample_id, "selected_candidate_id"]):
                raise RuntimeError("gated final selection differs from locked decision")
            gate_component_rows.append(
                _timing_row(
                    source,
                    sample_id,
                    "final_selection",
                    elapsed,
                    status="MEASURED_FROM_EXACT_LOCKED_GATE_OUTPUT",
                    deployment_stage=True,
                    variant="native_plus_reranker_plus_gate",
                    candidate_count=len(candidate_groups[sample_id]),
                )
            )
        metadata["gate_status"] = "MEASURED_EXACT_LOCKED_GATE"
        metadata["gate_sample_count"] = len(gate_ids)
        metadata["gate_model_path"] = str(gate.model_path)
        result_rows.extend(gate_component_rows)
    except Exception as error:  # fail closed; never time a lookup as gate inference
        metadata["component_failures"]["gate"] = f"{type(error).__name__}: {error}"
    return result_rows, metadata


def _stage_rows_for_route(
    repo_root: Path,
    source: RouteSource,
    four_manifest: pd.DataFrame,
    four_ids: list[str],
    six_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stages = FOUR_D_STAGES if source.dimension == "4D" else SIX_D_STAGES
    source_ids = four_ids if source.dimension == "4D" else six_ids
    sample_ids = set(source_ids)
    missing = [
        str(path)
        for path in (source.candidate_path, source.feature_path, source.prediction_path)
        if path is not None and not path.is_file()
    ]
    if missing:
        return rows, {
            "route": source.route,
            "status": "FAILED_PREFLIGHT",
            "reason": f"missing source artifacts: {missing}",
            "complete_deployment": False,
        }

    candidate_ns, candidates = timed_call_ns(
        lambda: pd.read_parquet(source.candidate_path), device=source.device
    )
    feature_ns, features = timed_call_ns(
        lambda: pd.read_parquet(source.feature_path), device=source.device
    )
    prediction_ns, predictions = timed_call_ns(
        lambda: pd.read_parquet(source.prediction_path), device=source.device
    )
    identifier = "group_id" if source.dimension == "6D" else "sample_id"
    candidates = candidates.loc[candidates[identifier].astype(str).isin(sample_ids)]
    features = features.loc[features[identifier].astype(str).isin(sample_ids)]
    predictions = predictions.loc[predictions[identifier].astype(str).isin(sample_ids)]
    if source.condition is not None:
        if "condition" not in candidates or "condition" not in features:
            raise RuntimeError(f"{source.route} source lacks a grounding-condition field")
        candidates = candidates.loc[candidates["condition"].eq(source.condition)].copy()
        features = features.loc[features["condition"].eq(source.condition)].copy()
        prediction_condition = (
            "grounding_condition"
            if "grounding_condition" in predictions
            else "condition"
        )
        if prediction_condition not in predictions:
            raise RuntimeError(f"{source.route} predictions lack a grounding condition")
        predictions = predictions.loc[
            predictions[prediction_condition].eq(source.condition)
        ].copy()

    cold_components = {
        "candidate_cache_read": candidate_ns,
        "feature_cache_read": feature_ns,
        "prediction_cache_read": prediction_ns,
    }
    for stage, elapsed in cold_components.items():
        rows.append(
            {
                "route": source.route,
                "dimension": source.dimension,
                "device": source.device,
                "sample_id": "__whole_profile_subset__",
                "cache_policy": "cold_file_open_os_cache_uncontrolled",
                "stage": stage,
                "elapsed_ns": elapsed,
                "elapsed_ms": elapsed / 1e6,
                "deployment_stage": False,
                "status": "MEASURED_COMPONENT_ONLY",
                "variant": "shared",
                "batch_size": np.nan,
                "warmup_samples": 0,
            }
        )

    if source.dimension == "4D":
        subset_manifest = four_manifest.loc[
            four_manifest["sample_id"].astype(str).isin(sample_ids)
        ].drop_duplicates("sample_id")
        io_payload: dict[str, tuple[bytes, bytes]] = {}
        for row in subset_manifest.itertuples(index=False):
            elapsed, payload = timed_call_ns(
                lambda row=row: (
                    Path(row.source_rgb_path).read_bytes(),
                    Path(row.source_depth_path).read_bytes(),
                )
            )
            io_payload[str(row.sample_id)] = payload
            rows.append(
                {
                    "route": source.route,
                    "dimension": source.dimension,
                    "device": source.device,
                    "sample_id": str(row.sample_id),
                    "cache_policy": "cold_file_open_os_cache_uncontrolled",
                    "stage": "input_io",
                    "elapsed_ns": elapsed,
                    "elapsed_ms": elapsed / 1e6,
                    "deployment_stage": True,
                    "status": "MEASURED",
                    "variant": "shared",
                    "batch_size": 1,
                    "warmup_samples": 5,
                }
            )
        for _ in range(5):
            for payload in list(io_payload.values())[:1]:
                _decode_inputs(*payload)
        for sample_id, payload in io_payload.items():
            elapsed, _ = timed_call_ns(lambda payload=payload: _decode_inputs(*payload))
            rows.append(
                {
                    "route": source.route,
                    "dimension": source.dimension,
                    "device": source.device,
                    "sample_id": sample_id,
                    "cache_policy": "warm_in_memory",
                    "stage": "preprocess",
                    "elapsed_ns": elapsed,
                    "elapsed_ms": elapsed / 1e6,
                    "deployment_stage": True,
                    "status": "MEASURED",
                    "variant": "shared",
                    "batch_size": 1,
                    "warmup_samples": 5,
                }
            )

    candidate_groups = {
        str(key): part for key, part in candidates.groupby(identifier, sort=False)
    }
    feature_groups = {
        str(key): part for key, part in features.groupby(identifier, sort=False)
    }
    prediction_groups = {
        str(key): part for key, part in predictions.groupby(identifier, sort=False)
    }
    for _ in range(5):
        for sample_id in source_ids[:1]:
            if sample_id in candidate_groups:
                candidate_groups[sample_id][["native_rank", "native_score"]].to_numpy()
    for sample_id in source_ids:
        if sample_id not in candidate_groups:
            continue
        elapsed, _ = timed_call_ns(
            lambda sample_id=sample_id: candidate_groups[sample_id][
                ["native_rank", "native_score"]
            ].to_numpy(copy=True)
        )
        rows.append(
            {
                "route": source.route,
                "dimension": source.dimension,
                "device": source.device,
                "sample_id": sample_id,
                "cache_policy": "warm_in_memory",
                "stage": "candidate_decode",
                "elapsed_ns": elapsed,
                "elapsed_ms": elapsed / 1e6,
                "deployment_stage": True,
                "status": "MEASURED_FROZEN_CACHE",
                "variant": "shared",
                "batch_size": 1,
                "warmup_samples": 5,
            }
        )
        if sample_id in feature_groups:
            elapsed, _ = timed_call_ns(
                lambda sample_id=sample_id: feature_groups[sample_id].select_dtypes(
                    include=[np.number]
                ).to_numpy(copy=True)
            )
            rows.append(
                {
                    "route": source.route,
                    "dimension": source.dimension,
                    "device": source.device,
                    "sample_id": sample_id,
                    "cache_policy": "warm_in_memory",
                    "stage": "feature_cache_materialisation",
                    "elapsed_ns": elapsed,
                    "elapsed_ms": elapsed / 1e6,
                    "deployment_stage": False,
                    "status": "MEASURED_COMPONENT_ONLY",
                    "variant": "shared",
                    "batch_size": 1,
                    "warmup_samples": 5,
                }
            )
        if source.dimension == "6D" and sample_id in prediction_groups:
            score_column = next(
                (
                    name
                    for name in ("score", "raw_rerank_score", "rerank_score")
                    if name in prediction_groups[sample_id]
                ),
                None,
            )
            if score_column is not None:
                values = prediction_groups[sample_id][score_column].to_numpy()
                elapsed, _ = timed_call_ns(lambda values=values: int(np.argmax(values)))
                rows.append(
                    {
                        "route": source.route,
                        "dimension": source.dimension,
                        "device": source.device,
                        "sample_id": sample_id,
                        "cache_policy": "warm_in_memory",
                        "stage": "final_selection",
                        "elapsed_ns": elapsed,
                        "elapsed_ms": elapsed / 1e6,
                        "deployment_stage": True,
                        "status": "MEASURED_FROM_FROZEN_SCORES",
                        "variant": "native_plus_raw_reranker",
                        "batch_size": 1,
                        "warmup_samples": 5,
                    }
                )

    component_metadata: dict[str, Any] = {}
    if source.dimension == "4D":
        learned_rows, component_metadata = _profile_four_d_learned_components(
            repo_root,
            source,
            four_manifest,
            source_ids,
            candidates,
            features,
            predictions,
        )
        rows.extend(learned_rows)

    measured_stages = {
        row["stage"]
        for row in rows
        if row["deployment_stage"] and str(row["status"]).startswith("MEASURED")
    }
    missing_stages = sorted(set(stages).difference(measured_stages))
    return rows, {
        "route": source.route,
        "status": "PARTIAL_COMPONENT_PROFILE",
        "reason": (
            "Frozen artifacts avoid prohibited repeat proposal inference, but no existing "
            "entry point can execute every pre-registered stage in one compatible process."
        ),
        "complete_deployment": False,
        "missing_deployment_stages": missing_stages,
        "retrospective": source.retrospective,
        "candidate_count_mean": float(
            candidates.groupby(identifier).size().reindex(source_ids).fillna(0).mean()
        ),
        **component_metadata,
    }


def _startup_timings(repo_root: Path, routes: list[RouteSource]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    python = repo_root / ".venv-graspnet6d/bin/python"
    code = (
        "import json,os,psutil,time;"
        "p=psutil.Process();b=p.memory_info().rss;t=time.perf_counter_ns();"
        "import numpy,pandas,pyarrow,lightgbm;"
        "print(json.dumps({'inner_ns':time.perf_counter_ns()-t,'rss_before':b,"
        "'rss_after':p.memory_info().rss}))"
    )
    for route in routes:
        for repeat in range(3):
            start = time.perf_counter_ns()
            result = subprocess.run(
                [str(python), "-c", code],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            )
            outer = time.perf_counter_ns() - start
            payload: dict[str, Any] = {}
            if result.returncode == 0 and result.stdout.strip():
                payload = json.loads(result.stdout.strip().splitlines()[-1])
            rows.append(
                {
                    "route": route.route,
                    "repeat": repeat,
                    "device": route.device,
                    "startup_ns": outer,
                    "startup_ms": outer / 1e6,
                    "inner_import_ns": payload.get("inner_ns"),
                    "rss_before_bytes": payload.get("rss_before"),
                    "rss_after_import_bytes": payload.get("rss_after"),
                    "returncode": result.returncode,
                    "status": (
                        "PARTIAL_IMPORT_STARTUP"
                        if result.returncode == 0
                        else "FAILED_PREFLIGHT"
                    ),
                    "model_checkpoint_loaded": False,
                }
            )
    return pd.DataFrame(rows)


def _memory_rows(routes: list[RouteSource]) -> pd.DataFrame:
    process = psutil.Process()
    rows: list[dict[str, Any]] = []
    for route in routes:
        before = process.memory_info().rss
        candidate_size = route.candidate_path.stat().st_size if route.candidate_path.exists() else 0
        feature_size = (
            route.feature_path.stat().st_size
            if route.feature_path is not None and route.feature_path.exists()
            else 0
        )
        prediction_size = (
            route.prediction_path.stat().st_size
            if route.prediction_path is not None and route.prediction_path.exists()
            else 0
        )
        after = process.memory_info().rss
        rows.append(
            {
                "route": route.route,
                "device": route.device,
                "rss_before_load_bytes": before,
                "rss_after_checkpoint_load_bytes": np.nan,
                "peak_rss_bytes": np.nan,
                "partial_process_rss_checkpoint_bytes": max(before, after),
                "model_load_memory_bytes": np.nan,
                "candidate_cache_size_bytes": candidate_size,
                "feature_cache_size_bytes": feature_size,
                "prediction_cache_size_bytes": prediction_size,
                "mps_current_allocated_bytes": np.nan,
                "mps_driver_allocated_bytes": np.nan,
                "unified_memory_peak": "not directly measurable",
                "status": "PARTIAL_METADATA_ONLY_NO_ROUTE_SUBPROCESS_MODEL_LOAD",
            }
        )
    return pd.DataFrame(rows)


def _offline_evaluator_timings(repo_root: Path, sample_ids: list[str]) -> pd.DataFrame:
    evaluator_path = (
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py"
    )
    spec = importlib.util.spec_from_file_location("_robustness_runtime_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    candidates = pd.read_parquet(
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/02_candidates/crog_test_all.parquet"
    )
    candidates = candidates.loc[
        candidates["sample_id"].astype(str).isin(sample_ids)
        & candidates["native_rank"].eq(1)
    ]
    labels = pd.read_parquet(
        repo_root
        / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/manifests/test_labels.parquet",
        columns=["sample_id", "gt_grasp_rectangles"],
    ).set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        prediction = module.CanonicalGrasp(
            cx_px=float(row.cx_px),
            cy_px=float(row.cy_px),
            theta_deg=float(row.theta_deg),
            jaw_width_px=float(row.width_px),
            rectangle_height_px=float(row.height_px),
        )
        ground_truth = tuple(
            module.gt_from_corners(
                np.stack(
                    [np.asarray(point, dtype=np.float64) for point in rectangle]
                )
            )
            for rectangle in labels.loc[row.sample_id, "gt_grasp_rectangles"]
        )
        elapsed, _ = timed_call_ns(
            lambda prediction=prediction, ground_truth=ground_truth: module.evaluate_candidate(
                prediction, ground_truth
            )
        )
        rows.append(
            {
                "route": "CROG",
                "sample_id": str(row.sample_id),
                "stage": "offline_matching_kernel_only",
                "elapsed_ns": elapsed,
                "elapsed_ms": elapsed / 1e6,
                "excluded_from_deployment_total": True,
                "status": "PARTIAL_PRELOADED_GT_SINGLE_CANDIDATE_MATCH",
            }
        )
    return pd.DataFrame(rows)


def _summarise(
    timings: pd.DataFrame,
    startup: pd.DataFrame,
    memory: pd.DataFrame,
    statuses: list[dict[str, Any]],
    evaluator: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    status_by_route = {item["route"]: item for item in statuses}
    for route in sorted(status_by_route):
        local = timings.loc[timings["route"].eq(route)]
        per_sample = (
            local.loc[local["deployment_stage"]]
            .groupby("sample_id", as_index=False)["elapsed_ms"]
            .sum()
        )
        per_sample = per_sample.loc[~per_sample["sample_id"].str.startswith("__")]
        startup_local = startup.loc[startup["route"].eq(route), "startup_ms"]
        memory_local = memory.loc[memory["route"].eq(route)]
        reranker = local.loc[local["stage"].eq("reranker"), "elapsed_ms"]
        gate = local.loc[local["stage"].eq("gate"), "elapsed_ms"]
        common_feature = local.loc[
            local["stage"].eq("feature_extraction_common_component"), "elapsed_ms"
        ]
        raw_selection = local.loc[
            local["stage"].eq("final_selection")
            & local.get("variant", pd.Series(index=local.index, dtype=object)).eq(
                "native_plus_raw_reranker"
            ),
            "elapsed_ms",
        ]
        gated_selection = local.loc[
            local["stage"].eq("final_selection")
            & local.get("variant", pd.Series(index=local.index, dtype=object)).eq(
                "native_plus_reranker_plus_gate"
            ),
            "elapsed_ms",
        ]
        complete = bool(status_by_route[route]["complete_deployment"])
        rows.append(
            {
                "route": route,
                "device": status_by_route[route].get("device", "cpu"),
                "profile_status": status_by_route[route]["status"],
                "startup_median_ms_partial_import_only": float(startup_local.median()),
                "deployment_median_ms": (
                    float(per_sample["elapsed_ms"].median()) if complete else np.nan
                ),
                "deployment_p95_ms": (
                    float(per_sample["elapsed_ms"].quantile(0.95)) if complete else np.nan
                ),
                "throughput_samples_per_s": (
                    1000.0 / float(per_sample["elapsed_ms"].median())
                    if complete and float(per_sample["elapsed_ms"].median()) > 0
                    else np.nan
                ),
                # Rows span mutually exclusive route variants, so their sum is not
                # a deployable path and must not be exposed as a latency total.
                "partial_measured_component_median_ms": np.nan,
                "reranker_overhead_ms": float(reranker.median()) if len(reranker) else np.nan,
                "reranker_overhead_percent": np.nan,
                "gate_overhead_ms": float(gate.median()) if len(gate) else np.nan,
                "common_feature_component_median_ms": (
                    float(common_feature.median()) if len(common_feature) else np.nan
                ),
                "common_feature_component_p95_ms": (
                    float(common_feature.quantile(0.95))
                    if len(common_feature)
                    else np.nan
                ),
                "raw_final_selection_median_ms": (
                    float(raw_selection.median()) if len(raw_selection) else np.nan
                ),
                "gated_final_selection_median_ms": (
                    float(gated_selection.median()) if len(gated_selection) else np.nan
                ),
                "reranker_measured_samples": int(len(reranker)),
                "gate_measured_samples": int(len(gate)),
                "common_feature_measured_samples": int(len(common_feature)),
                "peak_rss_bytes": (
                    memory_local["peak_rss_bytes"].iloc[0] if len(memory_local) else np.nan
                ),
                "partial_process_rss_checkpoint_bytes": (
                    memory_local["partial_process_rss_checkpoint_bytes"].iloc[0]
                    if len(memory_local)
                    else np.nan
                ),
                "offline_evaluator_median_ms": np.nan,
                "offline_matching_kernel_median_ms_partial": (
                    float(evaluator["elapsed_ms"].median()) if route == "CROG" else np.nan
                ),
                "reason": status_by_route[route].get("reason"),
            }
        )
    return pd.DataFrame(rows)


def run_runtime_profile(
    repo_root: Path, run_dir: Path, *, resume: bool = True
) -> dict[str, Any]:
    repo_root = Path(repo_root).expanduser().resolve()
    run_dir = Path(run_dir).expanduser().resolve()
    allowed = (repo_root / "artifacts/robustness_suite").resolve()
    if run_dir == allowed or allowed not in run_dir.parents:
        raise ValueError(
            "runtime output must be an isolated child of artifacts/robustness_suite"
        )
    output = run_dir / "runtime_profile"
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        completion = _runtime_completion_valid(repo_root, run_dir)
        if completion is not None:
            return {
                "status": "PARTIAL_RUNTIME_PROFILE",
                "resumed": True,
                "completion_signature_sha256": completion[
                    "completion_signature_sha256"
                ],
                "routes": json.loads(
                    (output / "route_status.json").read_text(encoding="utf-8")
                ),
                **completion["row_counts"],
                "subset_protocol_deviations": completion[
                    "subset_protocol_deviations"
                ],
            }
    subset = build_subset_manifest(repo_root, output)
    four_manifest = pd.read_parquet(
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet"
    )
    sources = _route_sources(repo_root)
    all_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for source in sources:
        route_rows, status = _stage_rows_for_route(
            repo_root,
            source,
            four_manifest,
            subset["four_d"]["sample_ids"],
            subset["six_d"]["group_ids"],
        )
        status["device"] = source.device
        all_rows.extend(route_rows)
        statuses.append(status)
    for status in statuses:
        if status["route"] == "6D_ORACLE":
            status["required_variants"] = ["native", "raw_reranker"]
        else:
            status["required_variants"] = [
                "native",
                "native_plus_feature_extraction",
                "native_plus_raw_reranker",
                "native_plus_reranker_plus_gate",
            ]
        status["complete_variant_count"] = 0
    statuses.append(
        {
            "route": "6D_HIFICS_ADAPTED",
            "status": "NOT_PROFILED_PROHIBITED_REPEAT_INFERENCE",
            "reason": (
                "The formal adapted-mask condition exists, but no frozen timing trace "
                "covers mask inference through VGN and the run forbids repeating the "
                "frozen VGN proposal inference."
            ),
            "complete_deployment": False,
            "missing_deployment_stages": list(SIX_D_STAGES),
            "retrospective": False,
            "device": "cpu",
            "candidate_count_mean": np.nan,
            "required_variants": ["native", "raw_reranker"],
            "complete_variant_count": 0,
        }
    )
    timings = pd.DataFrame(all_rows)
    startup = _startup_timings(repo_root, sources)
    memory = _memory_rows(sources)
    evaluator = _offline_evaluator_timings(
        repo_root, subset["four_d"]["sample_ids"]
    )
    summary = _summarise(timings, startup, memory, statuses, evaluator)
    atomic_parquet(output / "raw_stage_timings.parquet", timings)
    atomic_parquet(output / "offline_evaluator_timings.parquet", evaluator)
    atomic_csv(output / "startup_timings.csv", startup)
    atomic_csv(output / "memory.csv", memory)
    atomic_csv(output / "summary.csv", summary)
    atomic_json(output / "route_status.json", statuses)
    methods = """# Runtime profile methods

This run used `time.perf_counter_ns()` with device synchronisation hooks, five
distinct warm-up samples, deterministic 100-sample manifests, batch size one,
single-threaded LightGBM prediction and separate cache policies. For the 4-DoF
routes, the byte-locked three-seed formal LambdaMART checkpoints were restored,
their Train-fitted preprocessors were applied per candidate group, and every
measured score was checked against the frozen formal score table. The locked
transition gate was then executed and checked against its frozen probability
and switch decisions. Raw and gated final-selection operations were timed from
those actual outputs.

The exact frozen common-evidence extractor was also re-executed on the cached
predicted probability, mask and depth arrays. It is deliberately named
`feature_extraction_common_component`: CROG/G1/C1's complete T2 feature path
also contains RGB and calibration components, so this measured component is
not relabelled as total feature-extraction latency. Frozen artifacts further
permit measurement of input I/O, preprocessing, candidate-cache decode, cache
materialisation and a preloaded single-candidate offline matching kernel without
re-running any grasp network.

No compatible existing entry point could execute every pre-registered model
stage. The frozen 6-DoF source did not retain a loadable formal ranker
checkpoint, so its cached scores are not misrepresented as model inference.
Therefore deployment p50/p95, throughput and re-ranker percentage are
intentionally null. A cross-variant component sum is also intentionally null;
the individually named component medians must not be cited as full-pipeline
latency. Startup rows time a fresh Python
process plus scientific-runtime imports, but are also marked partial because
they do not load every route checkpoint. Unified-memory peak is not directly
measurable on this system. The same-process RSS checkpoint is retained under
the explicitly partial field `partial_process_rss_checkpoint_bytes`; required
peak route RSS and model-load memory remain null.

The offline matching kernel is timed separately and excluded from all
deployment component sums. It is not labelled a complete offline-evaluator
latency because GT loading and metric aggregation were outside the timed call.

Protocol deviation: the deterministic subsets are sequence/scene round-robin.
They are not additionally stratified by 4-DoF object and success/failure or by
6-DoF target visibility. The pre-registration chose not to use model outcomes
for sampling; this conflicts with the master profile's simultaneous
stratification request. Because the full profile is already fail-closed, the
subset is not changed after observing measurements, and the deviation remains
a completion blocker.

No source model, candidate cache, checkpoint or locked result was written or
mutated.
"""
    atomic_text(output / "PROFILE_METHODS.md", methods)

    output_names = (
        "raw_stage_timings.parquet",
        "offline_evaluator_timings.parquet",
        "startup_timings.csv",
        "memory.csv",
        "summary.csv",
        "route_status.json",
        "subset_manifest.json",
        "profile_subset_manifest.json",
        "PROFILE_METHODS.md",
    )
    source_paths: set[Path] = set()
    for source in sources:
        source_paths.add(source.candidate_path.resolve())
        if source.feature_path is not None:
            source_paths.add(source.feature_path.resolve())
        if source.prediction_path is not None:
            source_paths.add(source.prediction_path.resolve())
    source_paths.update(
        {
            (
                repo_root
                / "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet"
            ).resolve(),
            (
                repo_root
                / "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py"
            ).resolve(),
            (
                repo_root
                / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/manifests/test_labels.parquet"
            ).resolve(),
            (repo_root / "src/unified_reranking/feature_extractors/common.py").resolve(),
        }
    )
    for route in ("crog", "g1", "c1"):
        source_paths.update(
            {
                (
                    repo_root
                    / "runs/fair_unified_reranking_20260809_103012/08_lock/label_free_test_rankers"
                    / route
                    / "manifest.json"
                ).resolve(),
                (
                    repo_root
                    / "runs/fair_unified_reranking_20260809_103012/08_lock/gates"
                    / route
                    / "gate_selection.json"
                ).resolve(),
                (
                    repo_root
                    / "runs/fair_unified_reranking_20260809_103012/08_lock/label_free_test_gates"
                    / route
                    / "manifest.json"
                ).resolve(),
            }
        )
    source_paths.update(
        {
            (
                repo_root
                / "runs/fair_d1_reranking_extension_20260811T145515Z/08_lock/label_free_test_rankers/d1/manifest.json"
            ).resolve(),
            (
                repo_root
                / "runs/fair_d1_reranking_extension_20260811T145515Z/07_validation/gate/d1/gate_selection.json"
            ).resolve(),
            (
                repo_root
                / "runs/fair_d1_reranking_extension_20260811T145515Z/08_lock/label_free_test_gates/d1/manifest.json"
            ).resolve(),
        }
    )
    for status_row in statuses:
        for model_path in status_row.get("ranker_model_paths", []):
            source_paths.add(Path(model_path).resolve())
        gate_model_path = status_row.get("gate_model_path")
        if gate_model_path:
            source_paths.add(Path(gate_model_path).resolve())
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            f"runtime completion source inventory is incomplete: {missing_sources}"
        )
    source_hashes = {
        str(path.relative_to(repo_root)): _sha256_file(path)
        for path in sorted(source_paths)
    }
    completion: dict[str, Any] = {
        "schema_version": 1,
        "status": "PARTIAL_RUNTIME_PROFILE",
        "run_id": run_dir.name,
        "analysis_nature": "post-hoc robustness and sensitivity analysis",
        "runtime_code_sha256": _sha256_file(Path(__file__)),
        "source_hashes": source_hashes,
        "run_local_hashes": {
            "PRE_REGISTRATION.md": _sha256_file(run_dir / "PRE_REGISTRATION.md"),
            "source_artifact_inventory.json": _sha256_file(
                run_dir / "source_artifact_inventory.json"
            ),
        },
        "output_hashes": {
            name: _sha256_file(output / name) for name in output_names
        },
        "row_counts": {
            "measured_rows": len(timings),
            "startup_rows": len(startup),
            "offline_evaluator_rows": len(evaluator),
        },
        "subset_protocol_deviations": [
            "4D subset not stratified by object and success/failure",
            "6D subset not stratified by target visibility",
        ],
    }
    completion["completion_signature_sha256"] = _sha256_json(completion)
    atomic_json(output / "completion.json", completion)
    return {
        "status": "PARTIAL_RUNTIME_PROFILE",
        "resumed": False,
        "completion_signature_sha256": completion[
            "completion_signature_sha256"
        ],
        "routes": statuses,
        "measured_rows": len(timings),
        "startup_rows": len(startup),
        "offline_evaluator_rows": len(evaluator),
        "subset_protocol_deviations": [
            "4D subset not stratified by object and success/failure",
            "6D subset not stratified by target visibility",
        ],
    }


__all__ = [
    "FOUR_D_STAGES",
    "SIX_D_STAGES",
    "build_subset_manifest",
    "deployment_stage_total_ns",
    "run_runtime_profile",
    "stage_sum_matches_total",
    "synchronize_device",
    "timed_call_ns",
]
