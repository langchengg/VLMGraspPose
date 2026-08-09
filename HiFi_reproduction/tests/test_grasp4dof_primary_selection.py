from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.grasping.common.results import aggregate_method_metrics
from tools.grasp4dof.build_formal_lock_config import _selection_trace_matches

SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools/grasp4dof/select_validation_primary.py"
)
SPEC = importlib.util.spec_from_file_location("select_validation_primary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(method: str, j1: float, j5: float, *, latency: float = 0.1) -> dict:
    return {
        "method_id": method,
        "j_at_1": j1,
        "j_at_5": j5,
        "no_grasp_rate": 0.0,
        "predicted_mask_oracle_gap_recovery": 0.9,
        "p95_latency_seconds": latency,
    }


def test_primary_selector_honours_j1_outside_tolerance() -> None:
    rows = [_row("G1", 0.61, 0.7), _row("C1", 0.60, 0.9), _row("A0", 0.4, 0.9)]
    selected, trace = MODULE.choose_primary(rows, rate_tolerance=0.005)
    assert selected == "G1"
    assert trace[0]["eligible"] == ["G1"]


def test_lock_trace_replay_tolerates_only_csv_scale_float_roundoff() -> None:
    stored = [{"criterion": "j_at_1", "best": 0.9200635256749603, "eligible": ["G1"]}]
    replayed = [{"criterion": "j_at_1", "best": 0.9200635256749604, "eligible": ["G1"]}]
    assert _selection_trace_matches(replayed, stored)

    replayed[0]["best"] = 0.9200645256749603
    assert not _selection_trace_matches(replayed, stored)
    replayed[0]["best"] = stored[0]["best"]
    replayed[0]["eligible"] = ["C1"]
    assert not _selection_trace_matches(replayed, stored)


def test_primary_selector_uses_j5_within_registered_tolerance() -> None:
    rows = [_row("G1", 0.601, 0.7), _row("C1", 0.60, 0.8), _row("A0", 0.4, 0.9)]
    selected, _ = MODULE.choose_primary(rows, rate_tolerance=0.005)
    assert selected == "C1"


def test_primary_selector_prefers_latency_then_simplicity_after_outcome_ties() -> None:
    rows = [
        _row("G1", 0.6, 0.8, latency=0.2),
        _row("C1", 0.6, 0.8, latency=0.1),
        _row("A0", 0.6, 0.8, latency=1.0),
    ]
    selected, _ = MODULE.choose_primary(rows, rate_tolerance=0.005)
    assert selected == "C1"


def test_gap_recovery_is_bounded_and_handles_zero_oracle() -> None:
    assert MODULE.mask_gap_recovery(0.4, 0.5) == 0.8
    assert MODULE.mask_gap_recovery(0.6, 0.5) == 1.0
    assert MODULE.mask_gap_recovery(0.0, 0.0) == 1.0


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validation_output(
    run: Path, *, method_id: str, config: Path, oracle: bool
) -> Path:
    suffix = "oracle" if oracle else "predicted"
    output = run / f"validation/{suffix}/{method_id}"
    output.mkdir(parents=True)
    sample_rows = [
        {
            "method": method_id,
            "sample_id": f"sample-{index}",
            "scene_id": f"scene-{index // 10}",
            "j_at_1": False,
            "j_at_5": False,
            "candidate_pool_oracle": False,
            "first_valid_rank": None,
            "reciprocal_rank": 0.0,
            "non_empty": False,
            "raw_candidate_count": 0,
            "nms_candidate_count": 0,
            "empty_reason": "no_candidate",
            "latency_seconds": 0.01,
        }
        for index in range(MODULE.EXPECTED_VALIDATION_COUNT)
    ]
    sample_path = output / "per_sample_predictions.parquet"
    candidate_path = output / "per_candidate_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(sample_rows), sample_path)
    pd.DataFrame(columns=["method", "sample_id", "candidate_id", "rank"]).to_parquet(
        candidate_path, index=False
    )
    _json(output / "metrics.json", aggregate_method_metrics(sample_rows))
    _json(
        output / "runtime_metrics.json",
        {"sample_count": MODULE.EXPECTED_VALIDATION_COUNT},
    )
    _json(output / "memory_metrics.json", {"peak_rss_bytes": 1})
    run_config = {
        "method_id": method_id,
        "split": "validation",
        "oracle": oracle,
        "sample_count": MODULE.EXPECTED_VALIDATION_COUNT,
        "config_sha256": MODULE._sha256(config),
        "samples_manifest_sha256": MODULE._sha256(
            run / "manifests/validation_samples.parquet"
        ),
        "labels_manifest_sha256": MODULE._sha256(
            run / "manifests/validation_labels.parquet"
        ),
        "per_sample_sha256": MODULE._sha256(sample_path),
        "per_candidate_sha256": MODULE._sha256(candidate_path),
    }
    _json(output / "run_config.json", run_config)
    names = (
        "metrics.json",
        "runtime_metrics.json",
        "memory_metrics.json",
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
        "run_config.json",
    )
    _json(
        output / "COMPLETE.json",
        {
            "schema_version": 2,
            "status": "COMPLETE",
            "method_id": method_id,
            "split": "validation",
            "oracle": oracle,
            "sample_count": MODULE.EXPECTED_VALIDATION_COUNT,
            "config_sha256": MODULE._sha256(config),
            "artifacts": {name: MODULE._sha256(output / name) for name in names},
        },
    )
    return output


def test_source_backed_replay_rejects_self_consistent_fabricated_csv(
    tmp_path: Path,
) -> None:
    run = tmp_path.resolve()
    manifest = pd.DataFrame(
        [
            {"sample_id": f"sample-{index}", "scene_id": f"scene-{index // 10}"}
            for index in range(MODULE.EXPECTED_VALIDATION_COUNT)
        ]
    )
    (run / "manifests").mkdir(parents=True)
    manifest.to_parquet(run / "manifests/validation_samples.parquet", index=False)
    manifest.to_parquet(run / "manifests/validation_labels.parquet", index=False)
    _json(
        run / "audit/formal_input_reference_preflight.json",
        {
            "status": "PASS",
            "split_references": {
                "validation": {
                    "sample_count": MODULE.EXPECTED_VALIDATION_COUNT,
                    "samples_manifest": str(
                        (run / "manifests/validation_samples.parquet").resolve()
                    ),
                    "samples_manifest_sha256": MODULE._sha256(
                        run / "manifests/validation_samples.parquet"
                    ),
                    "labels_manifest": str(
                        (run / "manifests/validation_labels.parquet").resolve()
                    ),
                    "labels_manifest_sha256": MODULE._sha256(
                        run / "manifests/validation_labels.parquet"
                    ),
                }
            },
        },
    )
    configs = {}
    predicted = {}
    oracles = {}
    for method_id in MODULE.PREDICTED_METHODS:
        config = run / f"configs/{method_id}.json"
        _json(config, {"method_id": method_id})
        configs[method_id] = config
        predicted[method_id] = _validation_output(
            run, method_id=method_id, config=config, oracle=False
        )
        oracles[method_id] = _validation_output(
            run, method_id=method_id, config=config, oracle=True
        )
    argv = [
        "--run-dir",
        str(run),
        "--expected-count",
        str(MODULE.EXPECTED_VALIDATION_COUNT),
    ]
    for method_id in MODULE.PREDICTED_METHODS:
        argv.extend(["--method-dir", f"{method_id}={predicted[method_id]}"])
        argv.extend(["--oracle-dir", f"{method_id}={oracles[method_id]}"])
        argv.extend(["--config", f"{method_id}={configs[method_id]}"])
    assert MODULE.main(argv) == 0

    validation_path = run / "validation_results.csv"
    rows = pd.read_csv(validation_path)
    rows.loc[rows["method_id"] == "G1", ["j_at_1", "j_at_5"]] = 0.9
    rows.to_csv(validation_path, index=False)
    selection_path = run / "primary_validation_selection.json"
    selection = json.loads(selection_path.read_text())
    primary_rows = rows.loc[rows["method_id"].isin(MODULE.PRIMARY_METHODS)].to_dict(
        orient="records"
    )
    selected, trace = MODULE.choose_primary(
        primary_rows, rate_tolerance=float(selection["rate_tolerance"])
    )
    selection["primary_method_id"] = selected
    selection["trace"] = trace
    selection["validation_results_sha256"] = MODULE._sha256(validation_path)
    _json(selection_path, selection)

    with pytest.raises(ValueError, match="validation result/source mismatch"):
        MODULE.validate_existing_selection(run)


def test_selector_rejects_noncanonical_validation_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 3778"):
        MODULE.main(
            [
                "--run-dir",
                str(tmp_path),
                "--expected-count",
                "1",
                "--method-dir",
                "G0=/tmp/G0",
                "--method-dir",
                "G1=/tmp/G1",
                "--method-dir",
                "C0=/tmp/C0",
                "--method-dir",
                "C1=/tmp/C1",
                "--method-dir",
                "A0=/tmp/A0",
                "--oracle-dir",
                "G0=/tmp/G0-O",
                "--oracle-dir",
                "G1=/tmp/G1-O",
                "--oracle-dir",
                "C0=/tmp/C0-O",
                "--oracle-dir",
                "C1=/tmp/C1-O",
                "--oracle-dir",
                "A0=/tmp/A0-O",
                "--config",
                "G0=/tmp/G0.json",
                "--config",
                "G1=/tmp/G1.json",
                "--config",
                "C0=/tmp/C0.json",
                "--config",
                "C1=/tmp/C1.json",
                "--config",
                "A0=/tmp/A0.json",
            ]
        )
