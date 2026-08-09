#!/usr/bin/env python3
"""Consolidate full validation results and freeze the primary 4-DoF backend."""

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
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.results import (  # noqa: E402
    assert_aggregate_matches_sample_rows,
)


PREDICTED_METHODS = ("G0", "G1", "C0", "C1", "A0")
PRIMARY_METHODS = ("G1", "C1", "A0")
SIMPLICITY_RANK = {"A0": 0, "C1": 1, "G1": 2}
EXPECTED_VALIDATION_COUNT = 3778


def _assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected ID=PATH")
    method_id, raw = value.split("=", 1)
    if not method_id or not raw:
        raise argparse.ArgumentTypeError("expected non-empty ID=PATH")
    return method_id, Path(raw).expanduser().resolve()


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


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _completion_marker(directory: Path) -> Path:
    certified = directory / "VALIDATED_COMPLETE.json"
    return certified if certified.is_file() else directory / "COMPLETE.json"


def _validated_output(
    directory: Path,
    *,
    run: Path,
    method_id: str,
    config_path: Path,
    oracle: bool,
    expected_count: int,
    expected_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate content-addressed provenance and recompute saved aggregates."""

    directory.relative_to(run)
    names = (
        "metrics.json",
        "runtime_metrics.json",
        "memory_metrics.json",
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
        "run_config.json",
    )
    complete_path = _completion_marker(directory)
    if not complete_path.is_file() or not all((directory / name).is_file() for name in names):
        raise FileNotFoundError(f"incomplete validation output for {method_id}: {directory}")
    complete = _json(complete_path)
    config_sha256 = _sha256(config_path)
    if (
        complete.get("schema_version") != 2
        or complete.get("status") != "COMPLETE"
        or complete.get("method_id") != method_id
        or complete.get("split") != "validation"
        or complete.get("oracle") is not oracle
        or int(complete.get("sample_count", -1)) != expected_count
        or complete.get("config_sha256") != config_sha256
    ):
        raise ValueError(f"validation completion marker mismatch for {method_id}")
    artifacts = complete.get("artifacts")
    if not isinstance(artifacts, Mapping) or any(
        artifacts.get(name) != _sha256(directory / name) for name in names
    ):
        raise ValueError(f"validation completion artifact mismatch for {method_id}")
    sample_path = directory / "per_sample_predictions.parquet"
    candidate_path = directory / "per_candidate_predictions.parquet"
    run_config_path = directory / "run_config.json"
    run_config = _json(run_config_path)
    if (
        run_config.get("method_id") != method_id
        or run_config.get("split") != "validation"
        or run_config.get("oracle") is not oracle
        or int(run_config.get("sample_count", -1)) != expected_count
        or run_config.get("config_sha256") != config_sha256
        or run_config.get("samples_manifest_sha256")
        != _sha256(run / "manifests/validation_samples.parquet")
        or run_config.get("labels_manifest_sha256")
        != _sha256(run / "manifests/validation_labels.parquet")
        or run_config.get("per_sample_sha256") != _sha256(sample_path)
        or run_config.get("per_candidate_sha256") != _sha256(candidate_path)
    ):
        raise ValueError(f"validation provenance mismatch for {method_id}")
    sample_table = pq.read_table(sample_path)
    ids = {str(value) for value in sample_table.column("sample_id").to_pylist()}
    if sample_table.num_rows != expected_count or ids != expected_ids:
        raise ValueError(f"validation coverage mismatch for {method_id}")
    pq.ParquetFile(candidate_path)
    metrics = _json(directory / "metrics.json")
    assert_aggregate_matches_sample_rows(sample_table.to_pylist(), metrics)
    return metrics, run_config


def mask_gap_recovery(predicted_j1: float, oracle_j1: float) -> float:
    """Fraction of the GT-mask-oracle J@1 retained with a predicted mask."""

    if oracle_j1 <= 0.0:
        return 1.0 if predicted_j1 <= 0.0 else 0.0
    return min(max(predicted_j1 / oracle_j1, 0.0), 1.0)


def _same_saved_value(observed: Any, expected: Any) -> bool:
    if expected is None:
        return bool(pd.isna(observed))
    if isinstance(expected, bool):
        return isinstance(observed, (bool, int)) and bool(observed) is expected
    if isinstance(expected, int):
        try:
            return int(observed) == expected and float(observed).is_integer()
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(expected, float):
        try:
            value = float(observed)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and math.isclose(
            value, expected, rel_tol=0.0, abs_tol=1e-12
        )
    return str(observed) == str(expected)


def _validated_validation_ids(run: Path) -> set[str]:
    samples_path = run / "manifests/validation_samples.parquet"
    labels_path = run / "manifests/validation_labels.parquet"
    samples = pd.read_parquet(samples_path, columns=["sample_id"])["sample_id"].astype(
        str
    )
    labels = pd.read_parquet(labels_path, columns=["sample_id"])["sample_id"].astype(str)
    if (
        len(samples) != EXPECTED_VALIDATION_COUNT
        or len(labels) != EXPECTED_VALIDATION_COUNT
        or samples.duplicated().any()
        or labels.duplicated().any()
        or set(samples) != set(labels)
    ):
        raise ValueError("validation sample/label manifests violate the 3778-ID contract")
    preflight = _json(run / "audit/formal_input_reference_preflight.json")
    reference = preflight.get("split_references", {}).get("validation")
    expected = {
        "sample_count": EXPECTED_VALIDATION_COUNT,
        "samples_manifest": str(samples_path.resolve()),
        "samples_manifest_sha256": _sha256(samples_path),
        "labels_manifest": str(labels_path.resolve()),
        "labels_manifest_sha256": _sha256(labels_path),
    }
    if not isinstance(reference, Mapping) or any(
        reference.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("validation manifests do not match the formal-input preflight")
    return set(samples)


def validate_existing_selection(run_dir: Path) -> dict[str, Any]:
    """Rebuild validation rows from source outputs and replay selection exactly."""

    run = run_dir.expanduser().resolve()
    selection_path = run / "primary_validation_selection.json"
    validation_path = run / "validation_results.csv"
    selected_configs_path = run / "selected_configs.json"
    selection = _json(selection_path)
    selected_configs = _json(selected_configs_path)
    if (
        selection.get("selection_split") != "validation"
        or selection.get("test_metrics_read") is not False
        or selection.get("validation_results_sha256") != _sha256(validation_path)
        or set(selected_configs) != set(PREDICTED_METHODS)
        or selection.get("selected_configs") != selected_configs
    ):
        raise ValueError("existing primary selection provenance mismatch")
    expected_ids = _validated_validation_ids(run)
    sources = selection.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(PREDICTED_METHODS):
        raise ValueError("existing primary selection source coverage mismatch")
    rebuilt: list[dict[str, Any]] = []
    for method_id in PREDICTED_METHODS:
        config_record = selected_configs[method_id]
        config = Path(str(config_record["path"])).resolve()
        config.relative_to(run)
        if _sha256(config) != str(config_record["sha256"]):
            raise ValueError(f"existing selected-config drift: {method_id}")
        source = sources[method_id]
        directory = Path(str(source["directory"])).resolve()
        oracle_source = source.get("oracle")
        if not isinstance(oracle_source, Mapping):
            raise ValueError(f"existing primary oracle source missing: {method_id}")
        oracle_directory = Path(str(oracle_source["directory"])).resolve()
        metrics, _ = _validated_output(
            directory,
            run=run,
            method_id=method_id,
            config_path=config,
            oracle=False,
            expected_count=len(expected_ids),
            expected_ids=expected_ids,
        )
        oracle_metrics, _ = _validated_output(
            oracle_directory,
            run=run,
            method_id=method_id,
            config_path=config,
            oracle=True,
            expected_count=len(expected_ids),
            expected_ids=expected_ids,
        )

        def check_source(
            record: Mapping[str, Any], output: Path, *, include_config: bool
        ) -> None:
            expected = {
                "directory": str(output),
                "metrics_sha256": _sha256(output / "metrics.json"),
                "per_sample_sha256": _sha256(
                    output / "per_sample_predictions.parquet"
                ),
                "per_candidate_sha256": _sha256(
                    output / "per_candidate_predictions.parquet"
                ),
                "run_config_sha256": _sha256(output / "run_config.json"),
                "complete_sha256": _sha256(_completion_marker(output)),
                "complete_filename": _completion_marker(output).name,
            }
            if include_config:
                expected["config_sha256"] = _sha256(config)
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(f"existing selection source drift: {method_id}/{key}")

        check_source(source, directory, include_config=True)
        check_source(oracle_source, oracle_directory, include_config=False)
        row = {"method_id": method_id, **metrics}
        row["predicted_mask_oracle_gap_recovery"] = mask_gap_recovery(
            float(metrics["j_at_1"]), float(oracle_metrics["j_at_1"])
        )
        row["gt_mask_oracle_j_at_1"] = float(oracle_metrics["j_at_1"])
        rebuilt.append(row)
    saved = pd.read_csv(validation_path)
    if len(saved) != len(rebuilt) or set(saved["method_id"].astype(str)) != set(
        PREDICTED_METHODS
    ):
        raise ValueError("existing validation results method coverage mismatch")
    saved_by_id = saved.set_index(saved["method_id"].astype(str), drop=False)
    for row in rebuilt:
        observed = saved_by_id.loc[row["method_id"]]
        for field, expected in row.items():
            if field not in observed or not _same_saved_value(observed[field], expected):
                raise ValueError(
                    f"validation result/source mismatch: {row['method_id']}/{field}"
                )
    primary_rows = [row for row in rebuilt if row["method_id"] in PRIMARY_METHODS]
    selected, trace = choose_primary(
        primary_rows, rate_tolerance=float(selection["rate_tolerance"])
    )
    if selected != selection.get("primary_method_id") or trace != selection.get("trace"):
        raise ValueError("existing primary selection does not replay exactly")
    return selection


def choose_primary(
    rows: Sequence[Mapping[str, Any]], *, rate_tolerance: float
) -> tuple[str, list[dict[str, Any]]]:
    """Apply the preregistered ordered rule without consulting test outcomes."""

    if not 0.0 <= float(rate_tolerance) < 1.0:
        raise ValueError("rate tolerance must be in [0, 1)")
    by_id = {str(row["method_id"]): dict(row) for row in rows}
    if set(by_id) != set(PRIMARY_METHODS):
        raise ValueError("primary selection requires exactly G1, C1, and A0")
    eligible = list(by_id.values())
    trace: list[dict[str, Any]] = []

    best_j1 = max(float(row["j_at_1"]) for row in eligible)
    eligible = [
        row for row in eligible if best_j1 - float(row["j_at_1"]) <= rate_tolerance
    ]
    trace.append({"criterion": "j_at_1", "best": best_j1, "eligible": sorted(row["method_id"] for row in eligible)})

    best_j5 = max(float(row["j_at_5"]) for row in eligible)
    eligible = [
        row for row in eligible if best_j5 - float(row["j_at_5"]) <= rate_tolerance
    ]
    trace.append({"criterion": "j_at_5", "best": best_j5, "eligible": sorted(row["method_id"] for row in eligible)})

    ordered = sorted(
        eligible,
        key=lambda row: (
            float(row["no_grasp_rate"]),
            -float(row["predicted_mask_oracle_gap_recovery"]),
            float(row["p95_latency_seconds"]),
            SIMPLICITY_RANK[str(row["method_id"])],
            str(row["method_id"]),
        ),
    )
    selected = str(ordered[0]["method_id"])
    trace.append(
        {
            "criterion": "no_grasp_then_gap_recovery_then_p95_then_simplicity",
            "eligible_order": [str(row["method_id"]) for row in ordered],
            "selected": selected,
        }
    )
    return selected, trace


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method-dir", action="append", type=_assignment, required=True)
    parser.add_argument("--oracle-dir", action="append", type=_assignment, required=True)
    parser.add_argument("--config", action="append", type=_assignment, required=True)
    parser.add_argument("--rate-tolerance", type=float, default=0.005)
    parser.add_argument("--expected-count", type=int, default=3778)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    if args.expected_count != EXPECTED_VALIDATION_COUNT:
        raise ValueError("primary selection requires exactly 3778 validation samples")
    if (run / "manifests/experiment_lock.json").exists():
        raise RuntimeError("primary selection is forbidden after experiment lock")
    methods, oracles, configs = (
        dict(args.method_dir),
        dict(args.oracle_dir),
        dict(args.config),
    )
    if set(methods) != set(PREDICTED_METHODS):
        raise ValueError("--method-dir must provide exactly G0,G1,C0,C1,A0")
    if set(oracles) != set(PREDICTED_METHODS):
        raise ValueError("--oracle-dir must provide exactly G0,G1,C0,C1,A0")
    if set(configs) != set(PREDICTED_METHODS):
        raise ValueError("--config must provide exactly G0,G1,C0,C1,A0")
    expected_ids = _validated_validation_ids(run)
    if len(expected_ids) != args.expected_count:
        raise ValueError("validation manifest count differs from preregistered count")

    result_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for method_id in PREDICTED_METHODS:
        directory = methods[method_id]
        config_path = configs[method_id]
        config_path.relative_to(run)
        sample_path = directory / "per_sample_predictions.parquet"
        candidate_path = directory / "per_candidate_predictions.parquet"
        metrics_path = directory / "metrics.json"
        run_config_path = directory / "run_config.json"
        metrics, _ = _validated_output(
            directory,
            run=run,
            method_id=method_id,
            config_path=config_path,
            oracle=False,
            expected_count=args.expected_count,
            expected_ids=expected_ids,
        )
        oracle_metrics: dict[str, Any] | None = None
        if method_id in oracles:
            oracle_directory = oracles[method_id]
            oracle_metrics, _ = _validated_output(
                oracle_directory,
                run=run,
                method_id=method_id,
                config_path=config_path,
                oracle=True,
                expected_count=args.expected_count,
                expected_ids=expected_ids,
            )
        row = {"method_id": method_id, **metrics}
        row["predicted_mask_oracle_gap_recovery"] = (
            None
            if oracle_metrics is None
            else mask_gap_recovery(float(metrics["j_at_1"]), float(oracle_metrics["j_at_1"]))
        )
        row["gt_mask_oracle_j_at_1"] = (
            None if oracle_metrics is None else float(oracle_metrics["j_at_1"])
        )
        result_rows.append(row)
        if method_id in PRIMARY_METHODS:
            primary_rows.append(row)
        sources[method_id] = {
            "directory": str(directory),
            "config_sha256": _sha256(config_path),
            "metrics_sha256": _sha256(metrics_path),
            "per_sample_sha256": _sha256(sample_path),
            "per_candidate_sha256": _sha256(candidate_path),
            "run_config_sha256": _sha256(run_config_path),
            "complete_sha256": _sha256(_completion_marker(directory)),
            "complete_filename": _completion_marker(directory).name,
            "oracle": None
            if oracle_metrics is None
            else {
                "directory": str(oracles[method_id]),
                "metrics_sha256": _sha256(oracles[method_id] / "metrics.json"),
                "per_sample_sha256": _sha256(
                    oracles[method_id] / "per_sample_predictions.parquet"
                ),
                "per_candidate_sha256": _sha256(
                    oracles[method_id] / "per_candidate_predictions.parquet"
                ),
                "run_config_sha256": _sha256(oracles[method_id] / "run_config.json"),
                "complete_sha256": _sha256(_completion_marker(oracles[method_id])),
                "complete_filename": _completion_marker(oracles[method_id]).name,
            },
        }

    selected, trace = choose_primary(primary_rows, rate_tolerance=args.rate_tolerance)
    validation_path = run / "validation_results.csv"
    selected_configs_path = run / "selected_configs.json"
    primary_path = run / "primary_validation_selection.json"
    for path in (validation_path, selected_configs_path, primary_path):
        if path.exists():
            raise FileExistsError(path)
    pd.DataFrame(result_rows).to_csv(validation_path, index=False)
    config_records = {
        method_id: {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for method_id, path in configs.items()
    }
    _atomic_json(selected_configs_path, config_records)
    selection = {
        "schema_version": 1,
        "selection_split": "validation",
        "test_metrics_read": False,
        "primary_candidates": list(PRIMARY_METHODS),
        "primary_method_id": selected,
        "rate_tolerance": args.rate_tolerance,
        "selection_rule": [
            "j_at_1_with_tolerance",
            "j_at_5_with_tolerance",
            "lower_no_grasp_rate",
            "higher_predicted_mask_oracle_gap_recovery",
            "lower_p95_latency",
            "simpler_method",
        ],
        "trace": trace,
        "validation_results": str(validation_path),
        "validation_results_sha256": _sha256(validation_path),
        "selected_configs": config_records,
        "sources": sources,
    }
    _atomic_json(primary_path, selection)
    # Read everything back through the independent, source-backed validator so
    # even this initial materialization cannot emit an internally consistent
    # but source-inconsistent CSV/trace pair.
    validated = validate_existing_selection(run)
    print(json.dumps(validated, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
