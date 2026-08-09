#!/usr/bin/env python3
"""Run one frozen 4-DoF method on validation or the locked formal test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends import (  # noqa: E402
    AnalyticGraspConfig,
    BackendSample,
    GGCNN2Backend,
    GGCNN2Config,
    GRConvNetBackend,
    GRConvNetConfig,
    MaskDepthAnalyticBackend,
)
from src.grasping.backends.training import load_finetuned_model  # noqa: E402
from src.grasping.common import NMSConfig  # noqa: E402
from src.grasping.common.experiment_lock import verify_lock  # noqa: E402
from src.grasping.common.results import (  # noqa: E402
    aggregate_method_metrics,
    assert_metric_consistency,
    evaluate_prediction_records,
)
from src.grasping.common.sample_io import (  # noqa: E402
    CompactSampleLoader,
    aligned_labels,
    read_deployment_manifest,
    read_label_manifest,
)


METHOD_NAMES = {
    "G0": "repeatedfilm_grconvnet_pretrained_transfer",
    "G1": "repeatedfilm_grconvnet_ocidvlg_finetuned",
    "C0": "repeatedfilm_ggcnn2_pretrained_transfer",
    "C1": "repeatedfilm_ggcnn2_ocidvlg_finetuned",
    "A0": "repeatedfilm_mask_depth_analytic",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must contain a JSON object")
    return value


def _nms(value: Mapping[str, Any] | None) -> NMSConfig:
    return NMSConfig(**dict(value or {}))


def _build_backend(
    *, method_id: str, config: Mapping[str, Any], oracle: bool
) -> tuple[object, dict[str, Any]]:
    values = dict(config)
    values.pop("method", None)
    values.pop("method_id", None)
    values.pop("selection", None)
    finetuned_checkpoint = values.pop("finetuned_checkpoint", None)
    expected_finetuned_sha = values.pop("finetuned_checkpoint_sha256", None)
    expected_training_lineage = values.pop("training_data_lineage", None)
    expected_training_lineage_sha = values.pop(
        "training_data_lineage_sha256", None
    )
    if "nms" in values:
        values["nms"] = _nms(values["nms"])
    values["allow_oracle"] = bool(oracle)
    if method_id in ("G0", "G1"):
        network_config = GRConvNetConfig(**values)
        model = None
        fine_sha = None
        if method_id == "G1":
            if not finetuned_checkpoint:
                raise ValueError("G1 requires finetuned_checkpoint")
            model, fine_sha, payload = load_finetuned_model(
                finetuned_checkpoint,
                backend="grconvnet",
                device=network_config.device,
            )
        backend = GRConvNetBackend(network_config, model=model)
    elif method_id in ("C0", "C1"):
        network_config = GGCNN2Config(**values)
        model = None
        fine_sha = None
        if method_id == "C1":
            if not finetuned_checkpoint:
                raise ValueError("C1 requires finetuned_checkpoint")
            model, fine_sha, payload = load_finetuned_model(
                finetuned_checkpoint,
                backend="ggcnn2",
                device=network_config.device,
            )
        backend = GGCNN2Backend(network_config, model=model)
    elif method_id == "A0":
        backend = MaskDepthAnalyticBackend(AnalyticGraspConfig(**values))
        network_config = backend.config
        fine_sha = None
    else:
        raise ValueError(f"unsupported method ID: {method_id}")
    if finetuned_checkpoint:
        if (
            expected_finetuned_sha != fine_sha
            or not isinstance(expected_training_lineage, Mapping)
            or payload.get("data_lineage") != expected_training_lineage
            or payload.get("data_lineage_sha256") != expected_training_lineage_sha
        ):
            raise ValueError("fine-tuned checkpoint/config lineage mismatch")
        backend.checkpoint_path = str(Path(finetuned_checkpoint).resolve())
        backend.checkpoint_sha256 = fine_sha
    return backend, {
        "method_id": method_id,
        "method": METHOD_NAMES[method_id],
        "oracle": oracle,
        "backend_config": asdict(network_config),
        "finetuned_checkpoint": None
        if not finetuned_checkpoint
        else str(Path(finetuned_checkpoint).resolve()),
        "finetuned_checkpoint_sha256": fine_sha,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--method", choices=tuple(METHOD_NAMES), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-list", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_prefix = (PROJECT_ROOT / ".venv-grasp4dof").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(
            f"run_method requires isolated environment {expected_prefix}; "
            f"observed {Path(sys.prefix).resolve()}"
        )
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    run_dir = args.run_dir.expanduser().resolve()
    lock: Mapping[str, Any] | None = None
    if args.split == "test":
        if args.limit is not None or args.sample_list is not None:
            raise ValueError("formal test forbids --limit and --sample-list")
        lock = verify_lock(run_dir)
        config_record = lock["artifacts"].get(f"config_{args.method}")
        if not isinstance(config_record, Mapping):
            raise ValueError(f"locked config artifact missing for {args.method}")
        locked_config = (run_dir / str(config_record["path"])).resolve()
        if args.config.expanduser().resolve() != locked_config:
            raise ValueError("formal config path differs from the locked selected config")
        if _sha256(locked_config) != str(config_record["sha256"]):
            raise ValueError("formal config digest differs from the experiment lock")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.relative_to(run_dir)
    if args.split == "test":
        expected_output = run_dir / (
            f"oracle/{args.method}-O" if args.oracle else f"formal_test/{args.method}"
        )
        if output_dir != expected_output.resolve():
            raise ValueError(f"formal output directory must be {expected_output.resolve()}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    backend, run_config = _build_backend(
        method_id=args.method, config=_load_json(args.config), oracle=args.oracle
    )
    method_name = METHOD_NAMES[args.method] + ("_gt_mask_oracle" if args.oracle else "")
    manifests = run_dir / "manifests"
    samples_path = manifests / f"{args.split}_samples.parquet"
    labels_path = manifests / f"{args.split}_labels.parquet"
    deployment = read_deployment_manifest(samples_path)
    labels = read_label_manifest(labels_path)
    labels = list(aligned_labels(deployment, labels))
    if args.sample_list is not None:
        payload = _load_json(args.sample_list)
        requested = [str(row["sample_id"]) for row in payload.get("samples", [])]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("sample list must contain unique sample IDs")
        by_id = {
            str(row["sample_id"]): (row, label)
            for row, label in zip(deployment, labels, strict=True)
        }
        if any(sample_id not in by_id for sample_id in requested):
            raise ValueError("sample list contains IDs outside the selected split")
        pairs = [by_id[sample_id] for sample_id in requested]
        deployment = [pair[0] for pair in pairs]
        labels = [pair[1] for pair in pairs]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        deployment, labels = deployment[: args.limit], labels[: args.limit]
    loader = CompactSampleLoader()
    sample_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    peak_rss = initial_rss
    peak_mps = 0
    started = time.perf_counter()
    for index, (row, label) in enumerate(zip(deployment, labels, strict=True), 1):
        mask_source = "gt_mask_oracle" if args.oracle else "predicted"
        arrays = loader.load(
            row,
            mask_source=mask_source,
            labels=label if args.oracle else None,
            load_intrinsics=args.method == "A0",
        )
        if args.method == "A0":
            prediction = backend.predict(arrays)
        else:
            sample = BackendSample(
                sample_id=arrays.sample_id,
                rgb=arrays.rgb,
                depth_m=arrays.depth_m,
                predicted_mask=None if args.oracle else arrays.binary_mask,
                probability_map=None if args.oracle else arrays.probability,
                mask_source=mask_source,
                oracle_mask=arrays.binary_mask if args.oracle else None,
                metadata={"scene_id": arrays.scene_id},
            )
            prediction = backend.predict(sample)
        sample_record, candidates = evaluate_prediction_records(
            method=method_name, prediction=prediction, label=label
        )
        sample_rows.append(sample_record)
        candidate_rows.extend(candidates)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if torch.backends.mps.is_available():
            peak_mps = max(peak_mps, int(torch.mps.current_allocated_memory()))
        if index % 100 == 0 or index == len(deployment):
            _atomic_json(
                output_dir / "progress.json",
                {
                    "status": "RUNNING" if index < len(deployment) else "INFERENCE_COMPLETE",
                    "completed": index,
                    "expected": len(deployment),
                    "candidate_rows": len(candidate_rows),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(
                json.dumps(
                    {"method": method_name, "completed": index, "expected": len(deployment)}
                ),
                flush=True,
            )
    metrics = aggregate_method_metrics(sample_rows)
    assert_metric_consistency(metrics)
    sample_output = output_dir / "per_sample_predictions.parquet"
    candidate_output = output_dir / "per_candidate_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(sample_rows), sample_output, compression="zstd")
    pq.write_table(pa.Table.from_pylist(candidate_rows), candidate_output, compression="zstd")
    elapsed = time.perf_counter() - started
    runtime = {
        "method": method_name,
        "sample_count": len(sample_rows),
        "backend_latency_excludes_source_file_io": True,
        "total_wall_seconds_including_io_and_evaluation": elapsed,
        "throughput_samples_per_second": len(sample_rows) / elapsed,
        "p50_backend_latency_seconds": metrics["p50_latency_seconds"],
        "p95_backend_latency_seconds": metrics["p95_latency_seconds"],
    }
    memory = {
        "method": method_name,
        "initial_rss_bytes": initial_rss,
        "peak_rss_bytes": peak_rss,
        "rss_increase_bytes": peak_rss - initial_rss,
        "peak_mps_allocated_bytes": peak_mps,
    }
    provenance = {
        **run_config,
        "method": method_name,
        "split": args.split,
        "sample_count": len(sample_rows),
        "samples_manifest": str(samples_path),
        "samples_manifest_sha256": _sha256(samples_path),
        "labels_manifest": str(labels_path),
        "labels_manifest_sha256": _sha256(labels_path),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config.resolve()),
        "per_sample_sha256": _sha256(sample_output),
        "per_candidate_sha256": _sha256(candidate_output),
        "experiment_lock_sha256": None
        if lock is None
        else lock["manifest_content_sha256"],
    }
    for name, value in (
        ("metrics.json", metrics),
        ("runtime_metrics.json", runtime),
        ("memory_metrics.json", memory),
        ("run_config.json", provenance),
    ):
        _atomic_json(output_dir / name, value)
    artifact_hashes = {
        name: _sha256(output_dir / name)
        for name in (
            "metrics.json",
            "runtime_metrics.json",
            "memory_metrics.json",
            "run_config.json",
            "per_sample_predictions.parquet",
            "per_candidate_predictions.parquet",
        )
    }
    _atomic_json(
        output_dir / "COMPLETE.json",
        {
            "schema_version": 2,
            "status": "COMPLETE",
            "method_id": args.method,
            "method": method_name,
            "split": args.split,
            "oracle": bool(args.oracle),
            "sample_count": len(sample_rows),
            "config_sha256": _sha256(args.config.resolve()),
            "samples_manifest_sha256": _sha256(samples_path),
            "labels_manifest_sha256": _sha256(labels_path),
            "experiment_lock_sha256": None
            if lock is None
            else lock["manifest_content_sha256"],
            "artifacts": artifact_hashes,
        },
    )
    print(json.dumps({"status": "COMPLETE", "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
