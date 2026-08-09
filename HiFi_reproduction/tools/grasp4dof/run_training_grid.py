#!/usr/bin/env python3
"""Run the preregistered OCID-VLG fine-tuning grids and select on validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends.training import (  # noqa: E402
    TrainingConfig,
    load_finetuned_model,
    train_finetuned_backend,
)
from src.grasping.common.sample_io import (  # noqa: E402
    read_deployment_manifest,
    read_label_manifest,
)


EXPECTED_SPLIT_COUNTS = {"train": 26295, "validation": 3778}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_ids(rows: Sequence[Mapping[str, Any]], *, label: str) -> tuple[str, ...]:
    ids = tuple(str(row["sample_id"]) for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate sample IDs")
    return ids


def _ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()


def _manifest_artifact(path: Path, ids: Sequence[str]) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "row_count": len(ids),
        "ordered_sample_ids_sha256": _ids_sha256(ids),
    }


def load_training_data_contract(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load only train/validation and bind them to the audited preflight."""

    manifests = run_dir / "manifests"
    paths = {
        split: {
            "samples": (manifests / f"{split}_samples.parquet").resolve(),
            "labels": (manifests / f"{split}_labels.parquet").resolve(),
        }
        for split in EXPECTED_SPLIT_COUNTS
    }
    train_deployment = read_deployment_manifest(paths["train"]["samples"])
    train_labels = read_label_manifest(paths["train"]["labels"])
    validation_deployment = read_deployment_manifest(paths["validation"]["samples"])
    validation_labels = read_label_manifest(paths["validation"]["labels"])
    rows = {
        "train": (train_deployment, train_labels),
        "validation": (validation_deployment, validation_labels),
    }
    identity: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    split_ids: dict[str, tuple[str, ...]] = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        deployment, labels = rows[split]
        sample_ids = _ordered_ids(deployment, label=f"{split} samples")
        label_ids = _ordered_ids(labels, label=f"{split} labels")
        if (
            len(sample_ids) != expected_count
            or len(label_ids) != expected_count
            or sample_ids != label_ids
        ):
            raise ValueError(f"{split} manifests violate the ordered identity contract")
        split_ids[split] = sample_ids
        identity[split] = {
            "count": expected_count,
            "ordered_sample_ids_sha256": _ids_sha256(sample_ids),
        }
        artifacts[split] = {
            "samples": _manifest_artifact(paths[split]["samples"], sample_ids),
            "labels": _manifest_artifact(paths[split]["labels"], label_ids),
        }
    overlap = set(split_ids["train"]) & set(split_ids["validation"])
    if overlap:
        raise ValueError("train/validation sample IDs overlap")

    preflight_path = (run_dir / "audit/formal_input_reference_preflight.json").resolve()
    preflight = _load_json(preflight_path)
    references = preflight.get("split_references")
    if preflight.get("status") != "PASS" or not isinstance(references, Mapping):
        raise ValueError("formal input preflight is missing or invalid")
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        reference = references.get(split)
        expected_reference = {
            "sample_count": expected_count,
            "samples_manifest": str(paths[split]["samples"]),
            "samples_manifest_sha256": artifacts[split]["samples"]["sha256"],
            "labels_manifest": str(paths[split]["labels"]),
            "labels_manifest_sha256": artifacts[split]["labels"]["sha256"],
        }
        if not isinstance(reference, Mapping) or any(
            reference.get(key) != value for key, value in expected_reference.items()
        ):
            raise ValueError(f"{split} manifests drifted from the audited preflight")
    lineage: dict[str, Any] = {
        "schema_version": 1,
        "source": "audited_formal_manifests",
        "identity": identity,
        "train_validation_overlap": 0,
        "artifacts": artifacts,
        "formal_input_preflight": {
            "path": str(preflight_path),
            "bytes": preflight_path.stat().st_size,
            "sha256": _sha256_file(preflight_path),
        },
        "test_manifest_read": False,
    }
    lineage["content_sha256"] = _canonical_sha256(lineage)
    return (
        train_deployment,
        train_labels,
        validation_deployment,
        validation_labels,
        lineage,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value, indent=2, sort_keys=True, default=str, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _log_invocation(run_dir: Path, argv: Sequence[str] | None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with (run_dir / "commands.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp}\t{shlex.join(command)}\n")


def preregistered_grid(backend: str) -> tuple[tuple[float, float], ...]:
    """Return exactly the hyperparameters stated in the experiment protocol."""

    if backend == "grconvnet":
        return ((1e-4, 0.0), (1e-4, 1e-5), (5e-5, 0.0), (5e-5, 1e-5))
    if backend == "ggcnn2":
        return ((1e-4, 0.0), (5e-5, 0.0))
    raise ValueError(f"unsupported backend: {backend}")


def _nms_value(base: Mapping[str, Any], key: str, default: float) -> float:
    raw = base.get("nms", {})
    if not isinstance(raw, Mapping):
        raise ValueError("transfer config nms must be a mapping")
    return float(raw.get(key, default))


def build_training_config(
    *,
    backend: str,
    transfer_config: Mapping[str, Any],
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    batch_size: int,
    device: str,
    num_workers: int = 0,
) -> TrainingConfig:
    """Map the validation-selected transfer decoder to the training validator."""

    base = dict(transfer_config)
    if bool(base.get("allow_oracle", False)):
        raise ValueError("fine-tuning base config must use predicted masks")
    if base.get("finetuned_checkpoint"):
        raise ValueError("fine-tuning must initialize from an official transfer checkpoint")
    common: dict[str, Any] = {
        "backend": backend,
        "input_size": int(base["input_size"]),
        "conditioning_variant": str(base["conditioning_variant"]),
        "dilation_fraction": float(base.get("dilation_fraction", 0.15)),
        "minimum_crop_side_px": int(base.get("minimum_crop_side_px", 64)),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "seed": int(seed),
        "device": str(device),
        "quality_threshold": float(base.get("quality_threshold", 0.2)),
        "min_peak_distance_px": int(base.get("min_peak_distance_px", 20)),
        "max_raw_candidates": int(base.get("max_raw_candidates", 100)),
        "fixed_height_px": float(base.get("fixed_height_px", 20.0)),
        "center_gate_exponent": float(base.get("center_gate_exponent", 1.0)),
        "jaw_gate_exponent": float(base.get("jaw_gate_exponent", 0.0)),
        "nms_center_distance_px": _nms_value(base, "center_distance_px", 8.0),
        "nms_angle_distance_deg": _nms_value(base, "angle_distance_deg", 15.0),
        "nms_width_distance_px": _nms_value(base, "width_distance_px", 10.0),
        "nms_iou_threshold": _nms_value(base, "rectangle_iou_threshold", 0.25),
    }
    if backend == "grconvnet":
        common.update(
            {
                "grconvnet_source_checkpoint": str(base["checkpoint_path"]),
                "grconvnet_source_checkpoint_sha256": str(base["checkpoint_sha256"]),
                "grconvnet_input_channels": int(base.get("input_channels", 4)),
            }
        )
    return TrainingConfig(**common)


def _job_id(config: TrainingConfig) -> str:
    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    short = hashlib.sha256(payload).hexdigest()[:12]
    return (
        f"lr{config.learning_rate:g}_wd{config.weight_decay:g}_"
        f"seed{config.seed}_{short}"
    )


def _selection_tuple(result: Mapping[str, Any]) -> tuple[float, ...]:
    row = result["best_validation"]
    score = (
        float(row["j_at_1"]),
        float(row["j_at_5"]),
        float(row["non_empty_rate"]),
        -float(row["validation_loss"]),
        -float(row.get("validation_elapsed_seconds", float("inf"))),
    )
    if not all(math.isfinite(value) for value in score):
        raise ValueError("training selection metrics must be finite")
    return score


def _read_completed_job(
    path: Path,
    expected: TrainingConfig,
    data_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load_json(path / "training_complete.json")
    observed = result.get("training_config")
    expected_value = json.loads(json.dumps(asdict(expected), default=str))
    expected_lineage = json.loads(json.dumps(data_lineage, default=str))
    if (
        result.get("schema_version") != 2
        or result.get("status") != "COMPLETE"
        or result.get("backend") != expected.backend
        or observed != expected_value
        or result.get("data_lineage") != expected_lineage
        or result.get("data_lineage_sha256") != data_lineage.get("content_sha256")
    ):
        raise ValueError(f"completed training config drift: {path}")
    checkpoint = Path(str(result["best_checkpoint"])).expanduser().resolve()
    curves = path.resolve() / "training_curves.csv"
    artifacts = result.get("artifacts")
    if (
        checkpoint != path.resolve() / "best_state_dict.pt"
        or not checkpoint.is_file()
        or not curves.is_file()
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {"best_state_dict.pt", "training_curves.csv"}
        or artifacts.get("best_state_dict.pt") != _sha256_file(checkpoint)
        or artifacts.get("training_curves.csv") != _sha256_file(curves)
        or result.get("best_checkpoint_sha256") != artifacts.get("best_state_dict.pt")
    ):
        raise ValueError(f"completed training checkpoint missing or outside job: {path}")
    try:
        _, checkpoint_sha, payload = load_finetuned_model(
            checkpoint, backend=expected.backend, device="cpu"
        )
    except Exception as error:
        raise ValueError(f"completed training checkpoint cannot be strictly loaded: {path}") from error
    if (
        checkpoint_sha != result.get("best_checkpoint_sha256")
        or payload.get("schema_version") != 2
        or payload.get("training_config") != expected_value
        or payload.get("data_lineage") != expected_lineage
        or payload.get("data_lineage_sha256") != data_lineage.get("content_sha256")
        or payload.get("best_validation") != result.get("best_validation")
        or int(payload.get("best_epoch", -1))
        != int(result.get("best_validation", {}).get("epoch", -2))
        or payload.get("source_checkpoint_sha256")
        != result.get("source_checkpoint_sha256")
        or payload.get("training_curves_sha256")
        != artifacts.get("training_curves.csv")
        or int(payload.get("epochs_completed", -1))
        != int(result.get("epochs_completed", -2))
        or int(payload.get("last_epoch", -1)) != int(result.get("last_epoch", -2))
        or payload.get("termination_reason") != result.get("termination_reason")
    ):
        raise ValueError(f"completed training checkpoint evidence mismatch: {path}")
    with curves.open(newline="", encoding="utf-8") as stream:
        curve_rows = list(csv.DictReader(stream))
    if (
        not curve_rows
        or len(curve_rows) != int(result.get("epochs_completed", -1))
        or int(curve_rows[-1]["epoch"]) != int(result.get("last_epoch", -1))
        or [int(row["epoch"]) for row in curve_rows]
        != list(range(1, len(curve_rows) + 1))
    ):
        raise ValueError(f"completed training curve evidence mismatch: {path}")
    try:
        curve_best = max(
            curve_rows, key=lambda row: _selection_tuple({"best_validation": row})
        )
        saved_best = result["best_validation"]
        if not isinstance(saved_best, Mapping) or set(curve_best) != set(saved_best):
            raise ValueError("best row schema mismatch")
        for key, saved_value in saved_best.items():
            if not math.isfinite(float(curve_best[key])) or float(
                curve_best[key]
            ) != float(saved_value):
                raise ValueError(f"best row field mismatch: {key}")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"completed training curve/best mismatch: {path}") from error
    _selection_tuple(result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("grconvnet", "ggcnn2"), required=True)
    parser.add_argument("--transfer-config", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("mps", "cpu", "auto"), default="mps")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_prefix = (PROJECT_ROOT / ".venv-grasp4dof").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(
            f"training grid requires isolated environment {expected_prefix}; "
            f"observed {Path(sys.prefix).resolve()}"
        )
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    run_dir = args.run_dir.expanduser().resolve()
    if (run_dir / "manifests/experiment_lock.json").exists():
        raise RuntimeError("training is forbidden after the formal experiment lock")
    transfer_path = args.transfer_config.expanduser().resolve()
    transfer = _load_json(transfer_path)
    method_id = "G1" if args.backend == "grconvnet" else "C1"
    backend_dir = "grconvnet" if args.backend == "grconvnet" else "ggcnn2"
    grid_root = run_dir / backend_dir / "finetuned" / "grid"
    configs = [
        build_training_config(
            backend=args.backend,
            transfer_config=transfer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
        for learning_rate, weight_decay in preregistered_grid(args.backend)
    ]
    (
        train_deployment,
        train_labels,
        validation_deployment,
        validation_labels,
        data_lineage,
    ) = load_training_data_contract(run_dir)
    plan = {
        "schema_version": 2,
        "selection_split": "validation",
        "test_manifest_read": False,
        "backend": args.backend,
        "method_id": method_id,
        "transfer_config": {
            "path": str(transfer_path),
            "bytes": transfer_path.stat().st_size,
            "sha256": _sha256_file(transfer_path),
        },
        "data_lineage": data_lineage,
        "jobs": [
            {"job_id": _job_id(config), "training_config": asdict(config)}
            for config in configs
        ],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        return 0
    _log_invocation(run_dir, argv)
    plan_path = grid_root / "training_grid_plan.json"
    _atomic_json(plan_path, plan)
    results_path = grid_root / "training_grid_results.json"
    results: list[dict[str, Any]] = []
    for index, config in enumerate(configs, 1):
        job_id = _job_id(config)
        destination = grid_root / job_id
        complete = destination / "training_complete.json"
        print(
            json.dumps(
                {"status": "JOB_START", "job": job_id, "index": index, "total": len(configs)}
            ),
            flush=True,
        )
        if complete.is_file():
            result = _read_completed_job(destination, config, data_lineage)
        else:
            result = train_finetuned_backend(
                train_deployment=train_deployment,
                train_labels=train_labels,
                validation_deployment=validation_deployment,
                validation_labels=validation_labels,
                output_dir=destination,
                config=config,
                data_lineage=data_lineage,
            )
        results.append(result)
        _atomic_json(results_path, results)
        print(
            json.dumps(
                {"status": "JOB_COMPLETE", "job": job_id, "best": result["best_validation"]},
                sort_keys=True,
            ),
            flush=True,
        )

    selected = max(results, key=_selection_tuple)
    selected_checkpoint = Path(str(selected["best_checkpoint"])).resolve()
    selected_config = dict(transfer)
    selected_config.pop("method", None)
    selected_config.pop("method_id", None)
    selected_config["method_id"] = method_id
    selected_config["finetuned_checkpoint"] = str(selected_checkpoint)
    selected_config["finetuned_checkpoint_sha256"] = _sha256_file(
        selected_checkpoint
    )
    selected_config["training_data_lineage"] = data_lineage
    selected_config["training_data_lineage_sha256"] = data_lineage[
        "content_sha256"
    ]
    selected_config["selection"] = {
        "split": "validation",
        "rule": ["j_at_1", "j_at_5", "non_empty_rate", "validation_loss", "runtime"],
        "training_job": selected_checkpoint.parent.name,
        "best_epoch": int(selected["best_validation"]["epoch"]),
        "best_validation": selected["best_validation"],
    }
    selected_path = run_dir / "selected_configs" / f"{method_id}.json"
    _atomic_json(selected_path, selected_config)
    _atomic_json(
        grid_root / "TRAINING_GRID_COMPLETE.json",
        {
            "schema_version": 2,
            "status": "COMPLETE",
            "method_id": method_id,
            "selected_config": str(selected_path),
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": _sha256_file(selected_checkpoint),
            "data_lineage_sha256": data_lineage["content_sha256"],
            "jobs": len(results),
            "artifacts": {
                "training_grid_plan.json": _sha256_file(plan_path),
                "training_grid_results.json": _sha256_file(results_path),
                "selected_config": _sha256_file(selected_path),
            },
        },
    )
    print(json.dumps({"status": "COMPLETE", "selected_config": str(selected_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
