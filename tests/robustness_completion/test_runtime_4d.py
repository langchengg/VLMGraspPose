from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robustness_completion.common import PREREGISTRATION_SHA256, sha256_file
from robustness_completion.runtime_4d import (
    MEASURED_COUNT,
    PARITY_COUNT,
    SUPPORTED_MODES,
    WARMUP_COUNT,
    PipelineResult,
    RuntimeContractError,
    _StageRecorder,
    _angle_error_degrees,
    _d1_status,
    _gate_input,
    _geometry_hash,
    _measurement_rows,
    _optional_id,
    parse_args,
    validate_parity_proof,
)


REPO = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO
    / "artifacts/robustness_completion/20260822_194405_remaining_robustness"
)


def _candidate_frame(sample_id: str = "sample") -> pd.DataFrame:
    rows = []
    for index in range(2):
        row = {
            "route": "G1",
            "sample_id": sample_id,
            "candidate_id": f"candidate_{index}",
            "native_rank": index + 1,
            "native_score": 0.8 - index * 0.1,
            "cx_px": 10.0 + index,
            "cy_px": 20.0 + index,
            "theta_deg": 5.0 + index,
            "width_px": 30.0,
            "height_px": 15.0,
        }
        row["candidate_geometry_sha256"] = _geometry_hash(row)
        rows.append(row)
    return pd.DataFrame(rows)


def test_locked_preregistration_hash() -> None:
    assert sha256_file(RUN_DIR / "PRE_REGISTRATION.md") == PREREGISTRATION_SHA256
    assert (RUN_DIR / "PRE_REGISTRATION.sha256").read_text().split()[0] == PREREGISTRATION_SHA256


def test_cli_exposes_required_modes() -> None:
    assert SUPPORTED_MODES == ("parity", "cold", "warm-disk", "warm-preloaded")
    parsed = parse_args(
        [
            "--repo-root",
            str(REPO),
            "--run-dir",
            str(RUN_DIR),
            "--route",
            "g1",
            "--method",
            "gated",
            "--mode",
            "warm-preloaded",
            "--output",
            "/tmp/runtime.json",
        ]
    )
    assert parsed.route == "g1"
    assert parsed.mode == "warm-preloaded"


@pytest.mark.parametrize("route", ["crog", "g1", "c1"])
def test_live_geometry_hash_contract_matches_frozen_source(route: str) -> None:
    source = (
        REPO
        / "runs/fair_unified_reranking_20260809_103012/02_candidates"
        / f"{route}_test_top5.parquet"
    )
    row = pd.read_parquet(source).iloc[0]
    assert _geometry_hash(row) == row["candidate_geometry_sha256"]


def test_stage_rows_have_orchestrator_schema_and_nonnegative_time() -> None:
    recorder = _StageRecorder("G1", "raw", "sample", "warm_disk")
    recorder.add("input_io", 7)
    recorder.call("reranker", lambda: 3, device="cpu")
    rows = recorder.finish(candidate_count=2)
    required = {
        "route",
        "method",
        "mode",
        "sample_id",
        "stage",
        "elapsed_ns",
        "status",
        "candidate_count",
    }
    assert all(required <= set(row) for row in rows)
    assert all(row["elapsed_ns"] >= 0 for row in rows)
    assert all(row["candidate_count"] == 2 for row in rows)
    assert {row["mode"] for row in rows} == {"warm-disk"}
    with pytest.raises(AssertionError):
        recorder.add("gate", -1)


def test_selection_signature_covers_candidate_geometry_and_scores() -> None:
    candidates = _candidate_frame()
    result = PipelineResult(
        sample_id="sample",
        route="G1",
        method="native",
        candidates=candidates,
        features=pd.DataFrame(),
        seed_scores=None,
        ensemble_scores=None,
        native_candidate_id="candidate_0",
        raw_candidate_id="candidate_0",
        gated_candidate_id="candidate_0",
        selected_candidate_id="candidate_0",
        gate_probability_recover=None,
        gate_probability_harm=None,
        gate_switch=False,
    )
    baseline = result.selection_signature()
    result.candidates.loc[0, "native_score"] += 0.1
    assert result.selection_signature() != baseline


def test_live_gate_input_uses_native_and_three_seed_challenger() -> None:
    features = _candidate_frame()
    features["calibrated_native_probability"] = [0.7, 0.6]
    features["native_score_raw"] = [0.8, 0.7]
    features["overall_feature_reliability"] = [0.9, 0.8]
    features["peak_retention_rate"] = [0.9, 0.7]
    features["perturbed_valid_fraction"] = [0.8, 0.6]
    features["mask_reliability"] = [0.95, 0.75]
    scores = np.asarray([[0.1, 0.0, 0.2], [0.9, 1.0, 0.8]])
    ensemble = scores.mean(axis=1)
    gate, native, challenger, votes = _gate_input(features, scores, ensemble)
    assert (native, challenger, votes) == ("candidate_0", "candidate_1", 3)
    assert gate.loc[0, "ranker_score_margin"] == pytest.approx(0.8)
    assert gate.loc[0, "challenger_exists_numeric"] == 1.0


def test_candidate_angle_is_pi_periodic_and_optional_ids_are_fail_closed() -> None:
    assert _angle_error_degrees(89.0, -89.0) == pytest.approx(2.0)
    assert _angle_error_degrees(10.0, 190.0) == pytest.approx(0.0)
    assert _optional_id(None) is None
    assert _optional_id(np.nan) is None
    assert _optional_id("") is None
    assert _optional_id("candidate") == "candidate"


def test_frozen_prediction_caches_are_confined_to_parity_method() -> None:
    parity_source = inspect.getsource(
        __import__(
            "robustness_completion.runtime_4d", fromlist=["FourDRuntimeWorker"]
        ).FourDRuntimeWorker.run_parity
    )
    deployment_source = inspect.getsource(
        __import__(
            "robustness_completion.runtime_4d", fromlist=["FourDRuntimeWorker"]
        ).FourDRuntimeWorker.run_observation
    )
    assert 'self.paths["candidate"]' in parity_source
    assert 'self.paths["feature"]' in parity_source
    assert 'self.paths["score"]' in parity_source
    assert "read_parquet" not in deployment_source


def test_parity_proof_is_20_sample_fail_closed(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "status": "PASS",
                "complete_deployment": True,
                "route": "G1",
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "parity_count": PARITY_COUNT,
            }
        )
    )
    assert validate_parity_proof(proof, "G1")["status"] == "PASS"
    payload = json.loads(proof.read_text())
    payload["parity_count"] = PARITY_COUNT - 1
    proof.write_text(json.dumps(payload))
    with pytest.raises(RuntimeContractError):
        validate_parity_proof(proof, "G1")


def test_d1_is_deployable_but_unexecuted_and_never_has_cached_timings() -> None:
    status = _d1_status(REPO, probe_docker=False)
    assert status["status"] == "NOT_EXECUTED_IMPLEMENTATION_INCOMPLETE"
    assert status["deployable_source_route"] is True
    assert status["retrospective"] is True
    assert status["complete_deployment"] is False
    assert status["timings"] == []
    assert status["results"] == []
    hashes = status["checkpoint_and_model_hashes"]
    assert hashes["hifics_checkpoint"]["sha256"].startswith("b19a6493")
    assert hashes["gqcnn_model_file_manifest_sha256"]


class _FakeWorker:
    route = "G1"
    method = "raw"

    def run_row(self, row: dict[str, str], *, instrument: bool, cache_policy: str) -> PipelineResult:
        candidates = _candidate_frame(row["sample_id"])
        stage_rows = []
        if instrument:
            recorder = _StageRecorder(self.route, self.method, row["sample_id"], cache_policy)
            recorder.add("input_io", 1)
            stage_rows = recorder.finish(candidate_count=len(candidates))
        return PipelineResult(
            sample_id=row["sample_id"],
            route=self.route,
            method=self.method,
            candidates=candidates,
            features=pd.DataFrame(),
            seed_scores=None,
            ensemble_scores=None,
            native_candidate_id="candidate_0",
            raw_candidate_id="candidate_0",
            gated_candidate_id="candidate_0",
            selected_candidate_id="candidate_0",
            gate_probability_recover=None,
            gate_probability_harm=None,
            gate_switch=False,
            stage_rows=stage_rows,
            instrumented_total_ns=sum(row["elapsed_ns"] for row in stage_rows),
        )


def test_warm_result_rows_have_machine_readable_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "robustness_completion.runtime_4d.synchronize_device", lambda _device: None
    )
    rows = [
        {"sample_id": f"sample-{index:03d}"}
        for index in range(WARMUP_COUNT + MEASURED_COUNT)
    ]
    stages, payload = _measurement_rows(_FakeWorker(), rows, mode="warm-disk")
    samples = payload[0]["samples"]
    required = {
        "route",
        "method",
        "mode",
        "sample_id",
        "device",
        "status",
        "whole_elapsed_ns",
        "candidate_count",
    }
    assert len(samples) == MEASURED_COUNT
    assert all(required <= set(row) for row in samples)
    assert all(required - {"whole_elapsed_ns"} <= set(row) for row in stages)

