from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from failure_analysis.reranking_v3 import artifacts, pipeline
from failure_analysis.reranking_v3.calibration import calibrate_gate_bundle
from failure_analysis.reranking_v3.cli import _parser, _run_initial
from failure_analysis.reranking_v3.experiment_config import ENSEMBLE_SEEDS
from failure_analysis.reranking_v3.inference import save_gate_bundle, write_v3_predictions


def _identity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_ids = np.asarray([["base", "array-first", "q-first", "z", "a"]])
    checksums = np.asarray([[f"sha-{index}" for index in range(5)]])
    q_ranks = np.asarray([[0, 3, 1, 4, 2]], dtype=np.int64)
    return candidate_ids, checksums, q_ranks


def _bundle() -> dict[str, Any]:
    candidate_ids, checksums, q_ranks = _identity()
    gate = np.zeros((1, 5, 3), dtype=np.float32)
    gate[..., 2] = 1.0
    return {
        "sample_ids": np.asarray(["multiple:val:00000001"]),
        "candidate_ids": candidate_ids,
        "candidate_checksums": checksums,
        "q_ranks": q_ranks,
        "q": np.asarray([[1.0, 0.8, 0.9, 0.2, 0.3]], dtype=np.float32),
        "scores": np.asarray([[0.0, 0.9, 0.9, 0.1, 0.1]], dtype=np.float32),
        "seed_scores": np.zeros((3, 1, 5), dtype=np.float32),
        "probabilities": np.zeros((1, 5), dtype=np.float32),
        "seed_probabilities": np.zeros((3, 1, 5), dtype=np.float32),
        "any_probability": np.zeros(1, dtype=np.float32),
        "embeddings": np.zeros((1, 5, 2), dtype=np.float32),
        "token_attention": np.zeros((1, 5, 2), dtype=np.float32),
        "baseline_indices": np.asarray([0], dtype=np.int64),
        "v2_records": [],
        "gate_probabilities": gate,
        "seed_gate_probabilities": np.repeat(gate[None], 3, axis=0),
        "uncertainty": {
            "score_std": np.zeros((1, 5), dtype=np.float32),
            "valid_fraction": np.ones((1, 5), dtype=np.float32),
            "ranking_consistency": np.ones(1, dtype=np.float32),
            "ensemble_disagreement": np.zeros((1, 5), dtype=np.float32),
        },
    }


def _labels(path: Path, *, bad_checksum: bool = False) -> Path:
    candidate_ids, checksums, _ = _identity()
    rows = []
    for index in reversed(range(5)):
        rows.append({
            "candidate_id": str(candidate_ids[0, index]),
            "candidate_checksum": "wrong" if bad_checksum and index == 1 else str(checksums[0, index]),
            "candidate_correct": index == 1,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "sample_id": "multiple:val:00000001", "candidate_labels": rows,
    }) + "\n", encoding="utf-8")
    return path


def test_gate_bundle_persists_identity_q_ranks_and_explanation_evidence(tmp_path: Path) -> None:
    output = tmp_path / "gate_bundle.npz"
    manifest = save_gate_bundle(output, _bundle())
    with np.load(output) as payload:
        assert np.array_equal(payload["candidate_checksums"].astype(str), _identity()[1])
        assert np.array_equal(payload["q_ranks"], _identity()[2])
        assert np.array_equal(payload["token_attention"], _bundle()["token_attention"])
    assert manifest["candidate_checksums_persisted"] is True
    assert manifest["q_ranks_persisted"] is True
    assert manifest["token_attention_persisted"] is True


def test_calibration_rejects_checksum_drift_after_candidate_id_alignment(tmp_path: Path) -> None:
    bundle = tmp_path / "gate_bundle.npz"
    save_gate_bundle(bundle, _bundle())
    with pytest.raises(ValueError, match="candidate checksum differs"):
        calibrate_gate_bundle(
            bundle_path=bundle,
            labels_path=_labels(tmp_path / "labels.jsonl", bad_checksum=True),
            output_path=tmp_path / "policy.json",
        )
    assert not (tmp_path / "policy.json").exists()


def test_locked_prediction_remaining_order_uses_q_rank_not_array_index(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gate.pt"
    checkpoint.write_bytes(b"synthetic-gate")
    summary = write_v3_predictions(
        bundle=_bundle(),
        policy={
            "harm_cost": 2.0, "threshold": 1.0, "uncertainty_kappa": 0.0,
            "consensus": 3, "minimum_valid_fraction": 1.0,
        },
        output_dir=tmp_path / "predictions",
        gate_checkpoint_paths=[checkpoint],
    )
    row = json.loads((tmp_path / "predictions/predictions.jsonl").read_text().splitlines()[0])
    assert row["candidate_order"][:3] == ["base", "q-first", "array-first"]
    assert row["candidate_checksums"] == _identity()[1][0].tolist()
    assert row["candidate_q_ranks"] == _identity()[2][0].tolist()
    assert summary["ranking_tie_break"] == "score_desc_then_q_rank_asc_then_candidate_id_asc"


def test_final_ensemble_artifact_propagates_identity_across_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints = []
    for seed in ENSEMBLE_SEEDS:
        path = tmp_path / f"seed-{seed}.pt"
        path.write_bytes(str(seed).encode())
        checkpoints.append(path)

    monkeypatch.setattr(torch, "load", lambda path, **_: {"seed": int(Path(path).stem.split("-")[-1])})

    def fake_predict(**kwargs: Any) -> dict[str, Any]:
        candidate_ids, checksums, q_ranks = _identity()
        return {
            "sample_ids": np.asarray(["multiple:val:00000001"]),
            "candidate_ids": candidate_ids,
            "candidate_checksums": checksums,
            "q_ranks": q_ranks,
            "scores": np.zeros((1, 5), dtype=np.float32),
            "residual": np.zeros((1, 5), dtype=np.float32),
            "probabilities": np.zeros((1, 5), dtype=np.float32),
            "any_probability": np.zeros(1, dtype=np.float32),
            "embeddings": np.zeros((1, 5, 2), dtype=np.float32),
            "token_attention": np.zeros((1, 5, 2), dtype=np.float32),
            "q": np.zeros((1, 5), dtype=np.float32),
        }

    monkeypatch.setattr(pipeline, "predict_ranker_streaming", fake_predict)
    output = tmp_path / "final.npz"
    manifest = pipeline.predict_final_ensemble(
        catalog=object(), sample_ids={"multiple:val:00000001"}, priors={},
        checkpoint_paths=checkpoints, output_path=output, device="cpu",
    )
    with np.load(output) as payload:
        assert np.array_equal(payload["candidate_checksums"].astype(str), _identity()[1])
        assert np.array_equal(payload["q_ranks"], _identity()[2])
    assert manifest["candidate_checksums_persisted"] is True
    assert manifest["q_ranks_persisted"] is True


def test_train_gate_cli_runs_synthetic_calibration_and_keeps_summary_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts, "OUTPUT_ROOT", tmp_path)
    run = tmp_path / "v3_fullchain_identity"
    run.mkdir()
    gate_bundle = run / "inference/calibration/gate_bundle.npz"
    calibration_bundle = _bundle()
    calibration_bundle["gate_probabilities"][:, 1] = [1.0, 0.0, 0.0]
    calibration_bundle["seed_gate_probabilities"] = np.repeat(
        calibration_bundle["gate_probabilities"][None], 3, axis=0
    )
    save_gate_bundle(gate_bundle, calibration_bundle)
    labels = run / "labels/partitions/calibration/corrected_scientific/labels.jsonl"
    _labels(labels)
    policy = run / "calibration/gate_policy.json"
    argv = [
        "train-gate", "--output-dir", str(run), "--scope", "calibration",
        "--features", str(gate_bundle), "--policy", str(policy),
    ]
    result = _run_initial(
        _parser().parse_args(argv),
        ["python", "-m", "failure_analysis.reranking_v3.cli", *argv],
    )
    assert result["status"] == "locked_on_calibration"
    assert result["selected_correct"] == 1
    assert policy.exists()

    summary = run / "training/gate/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"status": "complete", "compatibility": True}), encoding="utf-8")
    query_argv = ["train-gate", "--output-dir", str(run)]
    queried = _run_initial(
        _parser().parse_args(query_argv),
        ["python", "-m", "failure_analysis.reranking_v3.cli", *query_argv],
    )
    assert queried == {"status": "complete", "compatibility": True}
