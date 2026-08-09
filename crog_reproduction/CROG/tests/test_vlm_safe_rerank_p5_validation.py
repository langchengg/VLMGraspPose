from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from failure_analysis.vlm_safe_rerank.p5_validation import _verify_p5_dependencies


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_p5(tmp_path: Path) -> tuple[Path, Path, Path]:
    calibration = tmp_path / "calibration/API_SAFE_GATE_CALIBRATION.json"
    local = tmp_path / "calibration/local_only_model.json"
    phase = tmp_path / "p5_validation"
    manifest = phase / "inference_manifest.json"
    query_gate = phase / "sample_query_gate.parquet"
    local_oof = tmp_path / "calibration/local_only_oof_pairs.parquet"
    validation_features = tmp_path / "validation_features.jsonl"
    stability = tmp_path / "diagnostic_expanded/DIAGNOSTIC_RESULTS.json"
    methods = {
        name: {"threshold_state": "no_beneficial_switch"}
        for name in ("P5_er2_safe", "P5_flash_safe", "P5_joint_safe")
    }
    local_oof.parent.mkdir(parents=True, exist_ok=True)
    local_oof.write_bytes(b"frozen-oof")
    validation_features.write_bytes(b"frozen-validation-features")
    _write_json(stability, {"stability": "frozen"})
    _write_json(
        calibration,
        {
            "methods": methods,
            "query_gate_contract": {
                "local_only_model_sha256": "pending",
                "local_only_oof_pairs_sha256": _sha(local_oof),
            },
            "diagnostic_stability": {"sha256": _sha(stability)},
        },
    )
    _write_json(local, {"query_model": "frozen"})
    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    calibration_payload["query_gate_contract"]["local_only_model_sha256"] = _sha(local)
    _write_json(calibration, calibration_payload)
    _write_json(
        manifest,
        {"eligible_methods": [], "needed_models": [], "rows": []},
    )
    query_gate.write_bytes(b"frozen-query-gate")
    _write_json(
        phase / "INFERENCE_MANIFEST_IDENTITY.json",
        {
            "files": {
                "inference_manifest": {"path": str(manifest), "sha256": _sha(manifest)},
                "sample_query_gate": {"path": str(query_gate), "sha256": _sha(query_gate)},
                "api_safe_gate_calibration": {"path": str(calibration), "sha256": _sha(calibration)},
                "local_query_model": {"path": str(local), "sha256": _sha(local)},
                "local_query_oof": {"path": str(local_oof), "sha256": _sha(local_oof)},
                "validation_features": {"path": str(validation_features), "sha256": _sha(validation_features)},
                "diagnostic_stability": {"path": str(stability), "sha256": _sha(stability)},
            }
        },
    )
    return phase, calibration, query_gate


def test_p5_dependency_sidecar_binds_all_inputs(tmp_path: Path) -> None:
    phase, _, _ = _frozen_p5(tmp_path)
    assert _verify_p5_dependencies(tmp_path, phase)["rows"] == []


@pytest.mark.parametrize("target", ["calibration", "query_gate"])
def test_p5_dependency_drift_hard_stops(tmp_path: Path, target: str) -> None:
    phase, calibration, query_gate = _frozen_p5(tmp_path)
    path = calibration if target == "calibration" else query_gate
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="frozen dependency changed"):
        _verify_p5_dependencies(tmp_path, phase)
