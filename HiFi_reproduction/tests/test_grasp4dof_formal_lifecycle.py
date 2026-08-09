from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from src.grasping.common.results import aggregate_method_metrics
from tools.grasp4dof.consolidate_results import main as consolidate_main
from tools.grasp4dof.run_final_validation import (
    EXPECTED_VALIDATION_COUNT,
    _complete as validation_complete,
)
from tools.grasp4dof.run_formal_inference import EXPECTED_TEST_COUNT, _complete
from tools.grasp4dof.recompute_reference import _legacy_outcome_comparison


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _formal_output(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    output = tmp_path / "G0"
    output.mkdir()
    config = tmp_path / "G0.json"
    _json(config, {"method_id": "G0"})
    sample_rows = [
        {
            "method": "repeatedfilm_grconvnet_pretrained_transfer",
            "sample_id": f"s{i}",
            "scene_id": f"scene-{i // 10}",
            "j_at_1": False,
            "j_at_5": False,
            "candidate_pool_oracle": False,
            "first_valid_rank": None,
            "reciprocal_rank": 0.0,
            "non_empty": False,
            "raw_candidate_count": 0,
            "nms_candidate_count": 0,
            "empty_reason": "synthetic_empty",
            "latency_seconds": 0.01,
        }
        for i in range(EXPECTED_TEST_COUNT)
    ]
    pd.DataFrame(sample_rows).to_parquet(
        output / "per_sample_predictions.parquet", index=False
    )
    pd.DataFrame({"candidate_id": []}).to_parquet(
        output / "per_candidate_predictions.parquet", index=False
    )
    test_samples_path = tmp_path / "test_samples.parquet"
    pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(EXPECTED_TEST_COUNT)]}
    ).to_parquet(test_samples_path, index=False)
    lock = {
        "run_dir": str(tmp_path),
        "manifest_content_sha256": "a" * 64,
        "artifacts": {
            "test_samples": {
                "path": test_samples_path.name,
                "bytes": test_samples_path.stat().st_size,
                "sha256": _sha(test_samples_path),
            },
            "test_labels": {"sha256": "c" * 64},
        },
    }
    _json(output / "metrics.json", aggregate_method_metrics(sample_rows))
    _json(output / "runtime_metrics.json", {"sample_count": EXPECTED_TEST_COUNT})
    _json(output / "memory_metrics.json", {"sample_count": EXPECTED_TEST_COUNT})
    run_config = {
        "method_id": "G0",
        "split": "test",
        "oracle": False,
        "experiment_lock_sha256": lock["manifest_content_sha256"],
        "samples_manifest_sha256": lock["artifacts"]["test_samples"]["sha256"],
        "labels_manifest_sha256": lock["artifacts"]["test_labels"]["sha256"],
        "config_sha256": _sha(config),
        "per_sample_sha256": _sha(output / "per_sample_predictions.parquet"),
        "per_candidate_sha256": _sha(output / "per_candidate_predictions.parquet"),
    }
    _json(output / "run_config.json", run_config)
    names = (
        "metrics.json",
        "runtime_metrics.json",
        "memory_metrics.json",
        "run_config.json",
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
    )
    _json(
        output / "COMPLETE.json",
        {
            "schema_version": 2,
            "status": "COMPLETE",
            "method_id": "G0",
            "split": "test",
            "oracle": False,
            "sample_count": EXPECTED_TEST_COUNT,
            "experiment_lock_sha256": lock["manifest_content_sha256"],
            "config_sha256": _sha(config),
            "artifacts": {name: _sha(output / name) for name in names},
        },
    )
    return output, config, lock


def test_formal_completion_requires_content_addressed_provenance(
    tmp_path: Path,
) -> None:
    output, config, lock = _formal_output(tmp_path)
    assert _complete(output, method_id="G0", oracle=False, lock=lock, config=config)


def test_formal_completion_rejects_wrong_method_marker(tmp_path: Path) -> None:
    output, config, lock = _formal_output(tmp_path)
    marker = json.loads((output / "COMPLETE.json").read_text())
    marker["method_id"] = "C0"
    _json(output / "COMPLETE.json", marker)
    with pytest.raises(ValueError, match="invalid formal completion marker"):
        _complete(output, method_id="G0", oracle=False, lock=lock, config=config)


def test_formal_completion_rejects_output_drift(tmp_path: Path) -> None:
    output, config, lock = _formal_output(tmp_path)
    _json(output / "metrics.json", {"sample_count": 1})
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        _complete(output, method_id="G0", oracle=False, lock=lock, config=config)


def test_formal_completion_requires_exact_locked_test_ids(tmp_path: Path) -> None:
    output, config, lock = _formal_output(tmp_path)
    manifest_path = tmp_path / "test_samples.parquet"
    pd.DataFrame(
        {"sample_id": [f"real-{i}" for i in range(EXPECTED_TEST_COUNT)]}
    ).to_parquet(manifest_path, index=False)
    lock["artifacts"]["test_samples"] = {
        "path": manifest_path.name,
        "bytes": manifest_path.stat().st_size,
        "sha256": _sha(manifest_path),
    }
    run_config_path = output / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["samples_manifest_sha256"] = _sha(manifest_path)
    _json(run_config_path, run_config)
    marker_path = output / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["artifacts"]["run_config.json"] = _sha(run_config_path)
    _json(marker_path, marker)

    with pytest.raises(ValueError, match="per-sample identity mismatch"):
        _complete(output, method_id="G0", oracle=False, lock=lock, config=config)


def test_formal_completion_rejects_rehashed_reversed_sample_order(
    tmp_path: Path,
) -> None:
    output, config, lock = _formal_output(tmp_path)
    samples_path = output / "per_sample_predictions.parquet"
    samples = pd.read_parquet(samples_path)
    samples.iloc[::-1].reset_index(drop=True).to_parquet(samples_path, index=False)

    run_config_path = output / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["per_sample_sha256"] = _sha(samples_path)
    _json(run_config_path, run_config)

    marker_path = output / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["artifacts"]["per_sample_predictions.parquet"] = _sha(samples_path)
    marker["artifacts"]["run_config.json"] = _sha(run_config_path)
    _json(marker_path, marker)

    with pytest.raises(ValueError, match="per-sample identity mismatch"):
        _complete(output, method_id="G0", oracle=False, lock=lock, config=config)


def test_formal_completion_rejects_rehashed_fabricated_metrics(tmp_path: Path) -> None:
    output, config, lock = _formal_output(tmp_path)
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["j_at_1"] = 0.5
    _json(metrics_path, metrics)
    marker_path = output / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["artifacts"]["metrics.json"] = _sha(metrics_path)
    _json(marker_path, marker)

    with pytest.raises(ValueError, match="metrics/per-sample mismatch"):
        _complete(output, method_id="G0", oracle=False, lock=lock, config=config)


def test_formal_completion_rejects_fabricated_r0_evidence(tmp_path: Path) -> None:
    output, _, lock = _formal_output(tmp_path)
    run_config_path = output / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["method_id"] = "R0"
    _json(run_config_path, run_config)
    evidence_path = output / "independent_reference_recompute.json"
    _json(evidence_path, {"status": "FABRICATED"})
    marker_path = output / "COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["method_id"] = "R0"
    marker["artifacts"]["run_config.json"] = _sha(run_config_path)
    marker["artifacts"]["independent_reference_recompute.json"] = _sha(evidence_path)
    _json(marker_path, marker)

    with pytest.raises(ValueError, match="locked R0 reference lineage"):
        _complete(output, method_id="R0", oracle=False, lock=lock)


def test_r0_legacy_success_flags_are_audited_but_not_used_as_ground_truth() -> None:
    comparison = _legacy_outcome_comparison(
        [
            ("same", False, False, False, False, False, False),
            ("changed", False, False, False, True, True, True),
        ]
    )

    assert comparison["status"] == "REEVALUATED_UNDER_LOCKED_EVALUATOR"
    assert comparison["compatible_with_locked_evaluator"] is False
    assert comparison["mismatch_count"] == 1
    assert comparison["mismatch_examples"] == ["changed"]
    assert comparison["legacy_metrics"] == {
        "j_at_1": 0.0,
        "j_at_5": 0.0,
        "candidate_pool_oracle": 0.0,
    }


def test_validation_completion_rejects_fabricated_metrics_and_placeholders(
    tmp_path: Path,
) -> None:
    output = tmp_path / "validation"
    output.mkdir()
    config = tmp_path / "G0.json"
    _json(config, {"method_id": "G0"})
    for name in ("per_sample_predictions.parquet", "per_candidate_predictions.parquet"):
        (output / name).write_bytes(b"not parquet")
    _json(
        output / "metrics.json",
        {"sample_count": EXPECTED_VALIDATION_COUNT, "j_at_1": 999},
    )
    _json(output / "runtime_metrics.json", {"sample_count": EXPECTED_VALIDATION_COUNT})
    _json(output / "memory_metrics.json", {"sample_count": EXPECTED_VALIDATION_COUNT})
    run_config = {
        "method_id": "G0",
        "split": "validation",
        "oracle": False,
        "sample_count": EXPECTED_VALIDATION_COUNT,
        "config_sha256": _sha(config),
        "per_sample_sha256": _sha(output / "per_sample_predictions.parquet"),
        "per_candidate_sha256": _sha(output / "per_candidate_predictions.parquet"),
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
            "method_id": "G0",
            "split": "validation",
            "oracle": False,
            "sample_count": EXPECTED_VALIDATION_COUNT,
            "config_sha256": _sha(config),
            "artifacts": {name: _sha(output / name) for name in names},
        },
    )

    with pytest.raises(pa.ArrowInvalid, match="Parquet|parquet|magic bytes"):
        validation_complete(output, method_id="G0", config_path=config, oracle=False)


def test_consolidation_without_lock_leaves_no_root_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    methods = (
        "R0",
        "G0",
        "G1",
        "C0",
        "C1",
        "A0",
        "G0-O",
        "G1-O",
        "C0-O",
        "C1-O",
        "A0-O",
    )
    argv = [
        "--run-dir",
        str(run),
        "--primary-method-id",
        "C1",
        "--expected-count",
        "1",
    ]
    for method in methods:
        argv.extend(["--method-dir", f"{method}={run / 'missing'}"])
    with pytest.raises(FileNotFoundError, match="experiment_lock"):
        consolidate_main(argv)
    output_names = (
        "per_sample_predictions.parquet",
        "per_candidate_predictions.parquet",
        "formal_test_results.csv",
        "per_method_metrics.csv",
        "oracle_results.csv",
        "common_subset_comparison.csv",
        "training_curves.csv",
        "runtime_metrics.json",
        "memory_metrics.json",
        "results_bundle.json",
    )
    assert not any((run / name).exists() for name in output_names)
