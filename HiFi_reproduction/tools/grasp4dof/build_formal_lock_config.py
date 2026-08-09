#!/usr/bin/env python3
"""Materialize the final lineage manifests and candidate config for the lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.grasp4dof.select_validation_primary import (  # noqa: E402
    PRIMARY_METHODS,
    choose_primary,
    validate_existing_selection,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"immutable lock-preparation artifact drift: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _checkpoint_inventory(manifest: Mapping[str, Any]) -> dict[str, str]:
    values = manifest.get("checkpoints")
    if not isinstance(values, list) or not values:
        raise ValueError("third-party manifest has no checkpoint inventory")
    return {str(row["checkpoint_id"]): str(row["sha256"]) for row in values}


def _selection_trace_matches(observed: Any, expected: Any) -> bool:
    """Compare a CSV-replayed trace without requiring lossless float text I/O."""

    if not isinstance(observed, list) or not isinstance(expected, list):
        return False
    if len(observed) != len(expected):
        return False
    for observed_step, expected_step in zip(observed, expected, strict=True):
        if not isinstance(observed_step, Mapping) or not isinstance(
            expected_step, Mapping
        ):
            return False
        if set(observed_step) != set(expected_step):
            return False
        for key, expected_value in expected_step.items():
            observed_value = observed_step[key]
            if key == "best":
                try:
                    if not math.isclose(
                        float(observed_value),
                        float(expected_value),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        return False
                except (TypeError, ValueError, OverflowError):
                    return False
            elif observed_value != expected_value:
                return False
    return True


def _selected_config_paths(run: Path) -> dict[str, Path]:
    records = _json(run / "selected_configs.json")
    expected = {"G0", "G1", "C0", "C1", "A0"}
    if set(records) != expected:
        raise ValueError("selected_configs.json must contain G0,G1,C0,C1,A0")
    result: dict[str, Path] = {}
    for method_id, raw in records.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"selected config record is invalid: {method_id}")
        path = Path(str(raw["path"])).expanduser().resolve()
        if not path.is_file() or _sha256(path) != str(raw["sha256"]):
            raise ValueError(f"selected config drift: {method_id}")
        path.relative_to(run)
        result[method_id] = path
    return result


def _validate_preflight_splits(run: Path, preflight: Mapping[str, Any]) -> None:
    split_references = preflight.get("split_references")
    expected_counts = {"train": 26295, "validation": 3778, "test": 7675}
    if not isinstance(split_references, Mapping) or set(split_references) != set(
        expected_counts
    ):
        raise ValueError("formal preflight split coverage mismatch")
    for split, expected_count in expected_counts.items():
        samples_path = (run / f"manifests/{split}_samples.parquet").resolve()
        labels_path = (run / f"manifests/{split}_labels.parquet").resolve()
        samples = pd.read_parquet(samples_path, columns=["sample_id"])[
            "sample_id"
        ].astype(str)
        labels = pd.read_parquet(labels_path, columns=["sample_id"])[
            "sample_id"
        ].astype(str)
        if (
            len(samples) != expected_count
            or len(labels) != expected_count
            or samples.duplicated().any()
            or labels.duplicated().any()
            or set(samples) != set(labels)
        ):
            raise ValueError(f"{split} manifests violate the locked sample-ID contract")
        reference = split_references[split]
        expected = {
            "sample_count": expected_count,
            "samples_manifest": str(samples_path),
            "samples_manifest_sha256": _sha256(samples_path),
            "labels_manifest": str(labels_path),
            "labels_manifest_sha256": _sha256(labels_path),
        }
        if not isinstance(reference, Mapping) or any(
            reference.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"{split} manifests do not match the formal preflight")


def _source_files(
    repeated: Mapping[str, Any], vendor_manifests: Sequence[Mapping[str, Any]]
) -> list[str]:
    sources = sorted(
        list((PROJECT_ROOT / "src/grasping/common").glob("*.py"))
        + list((PROJECT_ROOT / "src/grasping/backends").glob("*.py"))
        + list((PROJECT_ROOT / "tools/grasp4dof").glob("*.py"))
    )
    sources.extend(
        [
            PROJECT_ROOT / "requirements-grasp4dof-macos.txt",
            Path(str(repeated["checkpoint_path"])),
            Path(str(repeated["config_path"])),
        ]
    )
    for manifest in vendor_manifests:
        for row in manifest.get("source_files", {}).values():
            sources.append(Path(str(row["path"])))
        for row in manifest.get("checkpoints", []):
            sources.append(Path(str(row["path"])))
        license_path = manifest.get("license", {}).get("path")
        if license_path:
            sources.append(Path(str(license_path)))
    unique: dict[str, Path] = {}
    for path in sources:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        unique[_relative(resolved)] = resolved
    return sorted(unique)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    if (run / "manifests/experiment_lock.json").exists():
        raise RuntimeError("formal lock already exists")
    repeated_path = run / "audit/repeatedfilm_source_manifest.json"
    gr_path = run / "third_party/grconvnet_source_manifest.json"
    gg_path = run / "third_party/ggcnn2_source_manifest.json"
    primary_path = run / "primary_validation_selection.json"
    validation_results_path = run / "validation_results.csv"
    preflight_path = run / "audit/formal_input_reference_preflight.json"
    for path in (
        repeated_path,
        gr_path,
        gg_path,
        primary_path,
        validation_results_path,
        preflight_path,
        run / "selected_configs.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    repeated, gr_vendor, gg_vendor, primary, preflight = map(
        _json, (repeated_path, gr_path, gg_path, primary_path, preflight_path)
    )
    validated_primary = validate_existing_selection(run)
    if validated_primary != primary:
        raise ValueError("source-backed primary validation replay drifted")
    if (
        primary.get("selection_split") != "validation"
        or primary.get("test_metrics_read") is not False
    ):
        raise ValueError("primary was not selected under the validation-only contract")
    if primary.get("validation_results_sha256") != _sha256(validation_results_path):
        raise ValueError("primary selection validation-results digest mismatch")
    if preflight.get("status") != "PASS" or preflight.get("run_id") != run.name:
        raise ValueError("formal input reference preflight did not pass")
    _validate_preflight_splits(run, preflight)
    if int(preflight.get("r0", {}).get("sample_count", -1)) != 7675:
        raise ValueError("R0 preflight sample count mismatch")
    configs = _selected_config_paths(run)
    config_records = {
        method_id: {"path": str(path), "sha256": _sha256(path)}
        for method_id, path in configs.items()
    }
    if primary.get("selected_configs") != config_records:
        raise ValueError("primary selection selected-config records drifted")
    config_values = {method_id: _json(path) for method_id, path in configs.items()}
    selected_primary = str(primary["primary_method_id"])
    if selected_primary not in {"G1", "C1", "A0"}:
        raise ValueError("locked primary must be G1, C1, or A0")
    validation_rows = pd.read_csv(validation_results_path)
    if len(validation_rows) != 5 or set(validation_rows["method_id"].astype(str)) != {
        "G0",
        "G1",
        "C0",
        "C1",
        "A0",
    }:
        raise ValueError("validation result method coverage mismatch")
    replayed_primary, replayed_trace = choose_primary(
        validation_rows.loc[
            validation_rows["method_id"].astype(str).isin(PRIMARY_METHODS)
        ].to_dict(orient="records"),
        rate_tolerance=float(primary["rate_tolerance"]),
    )
    if replayed_primary != selected_primary or not _selection_trace_matches(
        replayed_trace, primary.get("trace")
    ):
        raise ValueError(
            "primary selection cannot be replayed from locked validation results"
        )

    model_manifest_path = run / "manifests/selected_model_manifest.json"
    evaluator_manifest_path = run / "manifests/evaluator_manifest.json"
    candidate_config_path = run / "manifests/formal_lock_candidate_config.json"
    fine_checkpoints: dict[str, dict[str, Any]] = {}
    for method_id in ("G1", "C1"):
        checkpoint = Path(
            str(config_values[method_id]["finetuned_checkpoint"])
        ).resolve()
        checkpoint.relative_to(run)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        fine_checkpoints[method_id] = {
            "path": checkpoint.relative_to(run).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        }
    model_manifest = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_metrics_read": False,
        "primary_method_id": selected_primary,
        "configurations": {
            method_id: {
                "path": path.relative_to(run).as_posix(),
                "sha256": _sha256(path),
            }
            for method_id, path in configs.items()
        },
        "fine_tuned_checkpoints": fine_checkpoints,
    }
    evaluator_sources = [
        PROJECT_ROOT / "src/grasping/common/evaluator.py",
        PROJECT_ROOT / "src/grasping/common/geometry.py",
        PROJECT_ROOT / "src/grasping/common/results.py",
        PROJECT_ROOT / "src/grasping/common/candidate_decoder.py",
    ]
    evaluator_manifest = {
        "schema_version": 1,
        "name": "corrected_ocid_vlg_rectangle_evaluator",
        "rectangle_iou_operator": ">",
        "rectangle_iou_threshold": 0.25,
        "angle_operator": "<=",
        "angle_threshold_degrees": 30.0,
        "angle_period_degrees": 180.0,
        "same_gt_pair_required": True,
        "fixed_grasp_height_px": 20.0,
        "all_sample_denominator": True,
        "source_hashes": {_relative(path): _sha256(path) for path in evaluator_sources},
    }
    source_files = _source_files(repeated, (gr_vendor, gg_vendor))
    _write_exclusive(model_manifest_path, model_manifest)
    _write_exclusive(evaluator_manifest_path, evaluator_manifest)

    artifacts: dict[str, dict[str, str]] = {
        "repeatedfilm_source": {
            "role": "source",
            "path": repeated_path.relative_to(run).as_posix(),
        },
        "train_samples": {"role": "data", "path": "manifests/train_samples.parquet"},
        "train_labels": {"role": "data", "path": "manifests/train_labels.parquet"},
        "validation_samples": {
            "role": "data",
            "path": "manifests/validation_samples.parquet",
        },
        "validation_labels": {
            "role": "data",
            "path": "manifests/validation_labels.parquet",
        },
        "test_samples": {"role": "data", "path": "manifests/test_samples.parquet"},
        "test_labels": {"role": "data", "path": "manifests/test_labels.parquet"},
        "grconvnet_vendor": {
            "role": "third_party",
            "path": gr_path.relative_to(run).as_posix(),
        },
        "ggcnn2_vendor": {
            "role": "third_party",
            "path": gg_path.relative_to(run).as_posix(),
        },
        "selected_models": {
            "role": "model",
            "path": model_manifest_path.relative_to(run).as_posix(),
        },
        "selected_configs": {
            "role": "selected_config",
            "path": "selected_configs.json",
        },
        "primary_selection": {
            "role": "selected_config",
            "path": primary_path.relative_to(run).as_posix(),
        },
        "validation_results": {
            "role": "selected_config",
            "path": validation_results_path.relative_to(run).as_posix(),
        },
        "formal_input_preflight": {
            "role": "source",
            "path": preflight_path.relative_to(run).as_posix(),
        },
        "evaluator": {
            "role": "evaluator",
            "path": evaluator_manifest_path.relative_to(run).as_posix(),
        },
    }
    for method_id, path in configs.items():
        artifacts[f"config_{method_id}"] = {
            "role": "selected_config",
            "path": path.relative_to(run).as_posix(),
        }
    for method_id, record in fine_checkpoints.items():
        artifacts[f"checkpoint_{method_id}"] = {"role": "model", "path": record["path"]}

    gr_checkpoints = _checkpoint_inventory(gr_vendor)
    gg_checkpoints = _checkpoint_inventory(gg_vendor)
    formal_device = {
        method_id: str(config.get("device", "cpu"))
        for method_id, config in config_values.items()
    }
    candidate_config = {
        "run_id": run.name,
        "artifacts": artifacts,
        "source_files": source_files,
        "lineage": {
            "repeated_film": {
                "source_manifest_artifact": "repeatedfilm_source",
                "checkpoint_sha256": str(repeated["checkpoint_sha256"]),
            },
            "splits": {
                "train": {"manifest_artifact": "train_samples", "sample_count": 26295},
                "validation": {
                    "manifest_artifact": "validation_samples",
                    "sample_count": 3778,
                },
                "test": {"manifest_artifact": "test_samples", "sample_count": 7675},
            },
            "prediction_manifest": {"artifact": "formal_input_preflight"},
            "reference": {
                "manifest_artifact": "formal_input_preflight",
                "reference_run": str(preflight["r0"]["reference_run"]),
                "direct_inputs": {
                    name: str(record["sha256"])
                    for name, record in preflight["r0"]["direct_inputs"].items()
                },
            },
            "vendors": {
                "grconvnet": {
                    "manifest_artifact": "grconvnet_vendor",
                    "commit": str(gr_vendor["pinned_commit"]),
                    "checkpoints": gr_checkpoints,
                },
                "ggcnn2": {
                    "manifest_artifact": "ggcnn2_vendor",
                    "commit": str(gg_vendor["pinned_commit"]),
                    "checkpoints": gg_checkpoints,
                },
            },
        },
        "protocol": {
            "preprocess": {
                "rgb": "official per-image mean",
                "depth": "metres; official mean/clip",
            },
            "input": {
                "G0_G1": "depth-first RGB-D or selected depth",
                "C0_C1": "depth",
                "A0": "mask_depth_intrinsics",
            },
            "device": formal_device,
            "conditioning": {
                method_id: value.get("conditioning_variant", "mask_depth_analytic")
                for method_id, value in config_values.items()
            },
            "crop": {
                method_id: {
                    "input_size": value.get("input_size"),
                    "dilation_fraction": value.get("dilation_fraction"),
                }
                for method_id, value in config_values.items()
            },
            "gate": {
                method_id: {
                    "center": value.get("center_gate_exponent"),
                    "jaw": value.get("jaw_gate_exponent"),
                    "quality_threshold": value.get("quality_threshold"),
                }
                for method_id, value in config_values.items()
            },
            "width": {"decode_scale_px": 150.0, "training_target_scale_px": 150.0},
            "angle": {
                "representation": "half_atan2_sin2_cos2",
                "period_degrees": 180.0,
            },
            "fixed_grasp_height_px": 20.0,
            "peak": {
                method_id: {
                    "threshold": value.get("quality_threshold"),
                    "min_distance_px": value.get("min_peak_distance_px"),
                }
                for method_id, value in config_values.items()
            },
            "nms": {
                method_id: value.get("nms")
                for method_id, value in config_values.items()
            },
            "analytic": config_values["A0"],
            "evaluator": {"artifact": "evaluator", "version": 1},
            "primary_method": selected_primary,
            "seed": 20260803,
            "expected_test_sample_count": 7675,
        },
        "selection_inputs": {
            "checkpoint": {"split": "validation", "artifact": "primary_selection"},
            "preprocessing": {"split": "validation", "artifact": "selected_configs"},
            "thresholds": {"split": "validation", "artifact": "selected_configs"},
            "analytic_weights": {"split": "validation", "artifact": "selected_configs"},
            "validation_results": {
                "split": "validation",
                "artifact": "validation_results",
            },
        },
    }
    _write_exclusive(candidate_config_path, candidate_config)
    print(
        json.dumps(
            {"status": "READY", "candidate_config": str(candidate_config_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
