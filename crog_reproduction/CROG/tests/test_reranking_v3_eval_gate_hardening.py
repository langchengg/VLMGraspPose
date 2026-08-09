from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from failure_analysis.reranking_v3.calibration import (
    calibrate_gate_bundle,
    decide_safe_gate,
    tune_safe_gate,
)
from failure_analysis.reranking_v3.evaluation import _cohort, _labels, _method_arrays
from failure_analysis.reranking_v3.inference import _align_v2_baseline, apply_locked_gate


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(value) + "\n" for value in records), encoding="utf-8")
    return path


def _candidates() -> list[dict]:
    return [
        {
            "candidate_id": f"c{index}",
            "candidate_checksum": f"sum-{index}",
            "q_raw": 0.9 - index * 0.1,
        }
        for index in range(5)
    ]


def _feature(source_id: int) -> dict:
    return {"split": "val", "sample_id": source_id, "candidates": _candidates()}


def _sample_id(source_id: int) -> str:
    return f"multiple:val:{source_id:08d}"


def _label(source_id: int, *, reverse: bool = False) -> dict:
    values = [
        {
            "candidate_id": f"c{index}",
            "candidate_checksum": f"sum-{index}",
            "candidate_correct": index == 4,
        }
        for index in range(5)
    ]
    return {
        "sample_id": _sample_id(source_id),
        "frame_id": f"frame-{source_id}",
        "sequence_id": f"scene-{source_id}",
        "candidate_labels": list(reversed(values)) if reverse else values,
    }


def _prediction(source_id: int) -> dict:
    return {
        "sample_id": _sample_id(source_id),
        "candidate_order": ["c4", "c3", "c2", "c1", "c0"],
        "candidate_probability_ids": ["c4", "c3", "c2", "c1", "c0"],
        "candidate_correctness_probabilities": [0.5, 0.4, 0.3, 0.2, 0.1],
        "selection": {"selected_candidate_id": "c4"},
    }


def test_allowed_evaluation_cohort_rejects_extra_features(tmp_path: Path) -> None:
    features = _write_jsonl(tmp_path / "features.jsonl", [_feature(1), _feature(2)])
    labels = _write_jsonl(tmp_path / "labels.jsonl", [_label(1)])
    with pytest.raises(ValueError, match="feature cohort differs"):
        _cohort(
            features_path=features,
            label_path=labels,
            allowed_ids={_sample_id(1)},
        )


@pytest.mark.parametrize("artifact", ["feature", "label"])
def test_evaluation_cohort_rejects_duplicate_records(tmp_path: Path, artifact: str) -> None:
    feature_rows = [_feature(1), _feature(1)] if artifact == "feature" else [_feature(1)]
    label_rows = [_label(1), _label(1)] if artifact == "label" else [_label(1)]
    features = _write_jsonl(tmp_path / "features.jsonl", feature_rows)
    labels = _write_jsonl(tmp_path / "labels.jsonl", label_rows)
    with pytest.raises(ValueError, match=f"duplicate {artifact}"):
        _cohort(
            features_path=features,
            label_path=labels,
            allowed_ids={_sample_id(1)},
        )


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "missing"])
def test_prediction_cohort_must_be_exact_and_unique(tmp_path: Path, mutation: str) -> None:
    ids = [_sample_id(1)]
    features = {ids[0]: _feature(1)}
    records = [_prediction(1)]
    if mutation == "duplicate":
        records.append(_prediction(1))
    elif mutation == "extra":
        records.append(_prediction(2))
    else:
        records = []
    path = _write_jsonl(tmp_path / "predictions.jsonl", records)
    with pytest.raises(ValueError, match="duplicate prediction|prediction cohort differs"):
        _method_arrays(ids=ids, features=features, prediction_path=path)


def test_label_join_uses_candidate_id_and_checksum() -> None:
    sample_id = _sample_id(1)
    features = {sample_id: _feature(1)}
    labels = {sample_id: _label(1, reverse=True)}
    joined = _labels([sample_id], labels, features)
    assert joined.tolist() == [[0.0, 0.0, 0.0, 0.0, 1.0]]

    labels[sample_id]["candidate_labels"][0]["candidate_checksum"] = "changed"
    with pytest.raises(ValueError, match="checksum differs"):
        _labels([sample_id], labels, features)


def test_probability_vector_is_aligned_by_candidate_id(tmp_path: Path) -> None:
    sample_id = _sample_id(1)
    features = {sample_id: _feature(1)}
    path = _write_jsonl(tmp_path / "predictions.jsonl", [_prediction(1)])
    ranking, probabilities = _method_arrays(
        ids=[sample_id], features=features, prediction_path=path,
    )
    assert ranking.tolist() == [[4, 3, 2, 1, 0]]
    assert probabilities.tolist() == [[0.1, 0.2, 0.3, 0.4, 0.5]]

    invalid = _prediction(1)
    invalid["candidate_probability_ids"][-1] = "c4"
    _write_jsonl(path, [invalid])
    with pytest.raises(ValueError, match="probability identity differs"):
        _method_arrays(ids=[sample_id], features=features, prediction_path=path)


def test_v2_baseline_accepts_superset_but_rejects_identity_failures(tmp_path: Path) -> None:
    requested = _sample_id(1)
    path = _write_jsonl(tmp_path / "v2.jsonl", [_prediction(1), _prediction(2)])
    baseline, records = _align_v2_baseline(
        sample_ids=np.asarray([requested]),
        candidate_ids=np.asarray([["c0", "c1", "c2", "c3", "c4"]]),
        v2_predictions_path=path,
    )
    assert baseline.tolist() == [4]
    assert [value["sample_id"] for value in records] == [requested]

    _write_jsonl(path, [_prediction(1), _prediction(1)])
    with pytest.raises(ValueError, match="duplicate V2 baseline"):
        _align_v2_baseline(
            sample_ids=np.asarray([requested]),
            candidate_ids=np.asarray([["c0", "c1", "c2", "c3", "c4"]]),
            v2_predictions_path=path,
        )

    _write_jsonl(path, [_prediction(2)])
    with pytest.raises(ValueError, match="missing FCER samples"):
        _align_v2_baseline(
            sample_ids=np.asarray([requested]),
            candidate_ids=np.asarray([["c0", "c1", "c2", "c3", "c4"]]),
            v2_predictions_path=path,
        )

    changed = _prediction(1)
    changed["candidate_order"][-1] = "other"
    _write_jsonl(path, [changed])
    with pytest.raises(ValueError, match="candidate pool differs"):
        _align_v2_baseline(
            sample_ids=np.asarray([requested]),
            candidate_ids=np.asarray([["c0", "c1", "c2", "c3", "c4"]]),
            v2_predictions_path=path,
        )


def _gate_probabilities(samples: int) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((samples, 5, 3), dtype=np.float64)
    probabilities[..., 2] = 1.0
    probabilities[:, 1] = [0.8, 0.0, 0.2]
    probabilities[:, 2] = [0.8, 0.0, 0.2]
    return probabilities, np.repeat(probabilities[None, ...], 3, axis=0)


def test_gate_tie_break_and_seed_consensus_are_deterministic() -> None:
    probabilities, seed_probabilities = _gate_probabilities(3)
    scores = np.asarray([
        [0.0, 0.4, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0, 0.0],
    ])
    q_ranks = np.asarray([
        [0, 1, 2, 3, 4],
        [0, 2, 1, 3, 4],
        [0, 1, 1, 3, 4],
    ])
    candidate_ids = np.asarray([
        ["base", "b", "a", "d", "e"],
        ["base", "b", "a", "d", "e"],
        ["base", "z", "a", "d", "e"],
    ])
    result = decide_safe_gate(
        baseline_indices=np.zeros(3, dtype=np.int64),
        gate_probabilities=probabilities,
        seed_gate_probabilities=seed_probabilities,
        uncertainty=np.zeros((3, 5)), valid_fraction=np.ones((3, 5)),
        fcer_scores=scores, candidate_ids=candidate_ids, q_ranks=q_ranks,
        harm_cost=2.0, threshold=0.0, uncertainty_kappa=0.0,
        consensus=3, minimum_valid_fraction=1.0,
    )
    assert result["proposed_indices"].tolist() == [2, 2, 2]
    assert result["selected_indices"].tolist() == [2, 2, 2]
    assert result["seed_selected_indices"].tolist() == [[2, 2, 2]] * 3
    assert result["consensus"].tolist() == [3, 3, 3]


def test_calibration_and_deployment_share_coverage_decision() -> None:
    probabilities, seed_probabilities = _gate_probabilities(1)
    # Candidate 1 wins every gain/tie-break, but its perturbation coverage is
    # below the locked minimum so both calibration and deployment must fall back.
    probabilities[:, 2] = [0.1, 0.0, 0.9]
    seed_probabilities = np.repeat(probabilities[None, ...], 3, axis=0)
    candidate_ids = np.asarray([["c0", "c1", "c2", "c3", "c4"]])
    scores = np.asarray([[0.0, 1.0, 0.5, 0.0, 0.0]])
    valid = np.asarray([[1.0, 0.5, 1.0, 1.0, 1.0]])
    calibrated = tune_safe_gate(
        labels=np.asarray([[0, 1, 0, 0, 0]], dtype=np.float64),
        baseline_indices=np.asarray([0]),
        gate_probabilities=probabilities,
        seed_gate_probabilities=seed_probabilities,
        uncertainty=np.zeros((1, 5)), valid_fraction=valid,
        fcer_scores=scores, candidate_ids=candidate_ids,
        q_ranks=np.asarray([[0, 1, 2, 3, 4]]),
        harm_costs=(2.0,), thresholds=(0.0,), kappas=(0.0,),
        consensus_values=(3,), minimum_valid_fractions=(1.0,),
    )
    assert calibrated["minimum_valid_fraction"] == 1.0
    assert calibrated["selected_indices"].tolist() == [0]
    policy = {key: value for key, value in calibrated.items() if key != "selected_indices"}
    deployed = apply_locked_gate(
        {
            "baseline_indices": np.asarray([0]),
            "gate_probabilities": probabilities,
            "seed_gate_probabilities": seed_probabilities,
            "scores": scores,
            "candidate_ids": candidate_ids,
            "candidate_checksums": np.asarray([["s0", "s1", "s2", "s3", "s4"]]),
            "sample_ids": np.asarray(["sample"]),
            "q_ranks": np.asarray([[0, 1, 2, 3, 4]]),
            "uncertainty": {"score_std": np.zeros((1, 5)), "valid_fraction": valid},
        },
        policy,
    )
    assert deployed["selected_indices"].tolist() == [0]
    assert deployed["coverage_ok"].tolist() == [False]


def test_calibration_bundle_aligns_labels_by_candidate_id(tmp_path: Path) -> None:
    sample_id = _sample_id(1)
    candidate_ids = np.asarray([["c0", "c1", "c2", "c3", "c4"]])
    probabilities = np.zeros((1, 5, 3), dtype=np.float32)
    probabilities[..., 2] = 1.0
    probabilities[:, 1] = [1.0, 0.0, 0.0]
    bundle = tmp_path / "bundle.npz"
    np.savez_compressed(
        bundle,
        sample_ids=np.asarray([sample_id]),
        candidate_ids=candidate_ids,
        candidate_checksums=np.asarray([[f"sha-{index}" for index in range(5)]]),
        q_ranks=np.asarray([[0, 1, 2, 3, 4]], dtype=np.int64),
        baseline_indices=np.asarray([0], dtype=np.int64),
        gate_probabilities=probabilities,
        seed_gate_probabilities=np.repeat(probabilities[None], 3, axis=0),
        uncertainty_score_std=np.zeros((1, 5), dtype=np.float32),
        uncertainty_valid_fraction=np.ones((1, 5), dtype=np.float32),
        scores=np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    values = [
        {
            "candidate_id": f"c{index}",
            "candidate_checksum": f"sha-{index}",
            "candidate_correct": index == 1,
        }
        for index in range(5)
    ]
    labels = _write_jsonl(
        tmp_path / "labels.jsonl",
        [{"sample_id": sample_id, "candidate_labels": list(reversed(values))}],
    )
    result = calibrate_gate_bundle(
        bundle_path=bundle,
        labels_path=labels,
        output_path=tmp_path / "policy.json",
    )
    assert result["baseline_correct"] == 0
    assert result["selected_correct"] == 1
    assert result["recovered"] == 1

    invalid = list(reversed(values))
    invalid[0] = {"candidate_id": "other", "candidate_checksum": "sha-other", "candidate_correct": True}
    _write_jsonl(labels, [{"sample_id": sample_id, "candidate_labels": invalid}])
    with pytest.raises(ValueError, match="candidate IDs differ"):
        calibrate_gate_bundle(
            bundle_path=bundle,
            labels_path=labels,
            output_path=tmp_path / "unused.json",
        )
