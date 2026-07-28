from copy import deepcopy

import pytest

from failure_analysis.reranking.evaluate import evaluate_feature_records
from failure_analysis.reranking.geometry import geometry_checksum
from failure_analysis.reranking.rankers import q_only_matches_q_rank, rank_candidates
from failure_analysis.reranking.train_mlp import load_mlp_scorer, save_artifact, train_mlp


def _candidate(candidate_id, legacy_rank, q, mask_score):
    candidate = {
        "candidate_id": candidate_id,
        "legacy_rank": legacy_rank,
        "q_rank": legacy_rank,
        "row": 10 + legacy_rank,
        "col": 20 + legacy_rank,
        "cx": float(20 + legacy_rank),
        "cy": float(10 + legacy_rank),
        "angle_rad": 0.0,
        "angle_deg": 0.0,
        "width_px": 30.0,
        "height_px": 20,
        "polygon": [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]],
        "q_raw": q,
        "candidate_checksum": None,
        "legacy_grasp": [float(20 + legacy_rank), float(10 + legacy_rank), 30.0, 20, 0.0],
        "features": {
            "q": {"value": q, "reliability": 1.0, "missing_reason": None},
            "mask_consistency": {
                "value": mask_score,
                "reliability": 1.0,
                "missing_reason": None,
            },
            "width_compatibility": {
                "value": mask_score,
                "reliability": 1.0,
                "missing_reason": None,
            },
        },
    }
    candidate["candidate_checksum"] = geometry_checksum(candidate)
    return candidate


def _records():
    feature_records = []
    label_records = []
    specifications = [
        ("s0", "frame0", [False, True]),
        ("s1", "frame1", [True, False]),
    ]
    for sample_id, scene_id, valid in specifications:
        candidates = [
            _candidate("candidate_0", 0, 0.9, 0.1),
            _candidate("candidate_1", 1, 0.8, 1.0),
        ]
        feature_records.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "split": "val",
                "predicted_mask_area": 10,
                "candidates": candidates,
            }
        )
        label_records.append(
            {
                "sample_id": sample_id,
                "candidate_labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_checksum": candidate["candidate_checksum"],
                        "candidate_valid": validity,
                    }
                    for candidate, validity in zip(candidates, valid)
                ],
            }
        )
    return feature_records, label_records


def test_q_only_reproduces_q_order_and_candidate_checksums_are_frozen():
    features, _ = _records()
    candidates = features[0]["candidates"]
    assert q_only_matches_q_rank(candidates)
    before = {item["candidate_id"]: item["candidate_checksum"] for item in candidates}
    for ranker in ("legacy", "q_only", "rule_2d_equal", "rule_fixed_v1"):
        ranked = rank_candidates(candidates, ranker)
        assert {item["candidate_id"]: item["candidate_checksum"] for item in ranked} == before


def test_oracle_is_invariant_and_recovered_minus_harmful_equals_delta():
    features, labels = _records()
    summary, outcomes, _ = evaluate_feature_records(
        features,
        labels,
        ranker="mlp",
        mlp_scorer=lambda candidate: candidate["features"]["mask_consistency"]["value"],
        bootstrap_iterations=20,
        seed=3,
    )
    assert summary["oracle_at_5"] == 1.0
    assert summary["net_gain"] == summary["recovered"] - summary["harmful_flip"]
    assert summary["reranked_success_count"] - summary["original_success_count"] == summary["net_gain"]
    assert all(item["candidate_count"] == 2 for item in outcomes)


def test_same_seed_produces_identical_bootstrap_summary():
    features, labels = _records()
    first = evaluate_feature_records(
        features, labels, ranker="q_only", bootstrap_iterations=50, seed=11
    )[0]
    second = evaluate_feature_records(
        features, labels, ranker="q_only", bootstrap_iterations=50, seed=11
    )[0]
    assert first == second


def test_empty_candidate_set_is_consistent_and_not_counted_as_ranking_change():
    features = [
        {
            "sample_id": "empty",
            "scene_id": "frame-empty",
            "split": "val",
            "predicted_mask_area": 0,
            "candidates": [],
        }
    ]
    labels = [{"sample_id": "empty", "candidate_labels": []}]
    summary, outcomes, _ = evaluate_feature_records(
        features,
        labels,
        ranker="legacy",
        bootstrap_iterations=2,
        seed=3,
    )
    assert outcomes[0]["same_top1"] is True
    assert summary["top1_consistency"] == 1.0
    assert summary["ranking_changed_count"] == 0


def test_mlp_uses_group_split_rejects_test_and_score_ignores_gt(tmp_path):
    features, labels = _records()
    extra_features = []
    extra_labels = []
    for index in range(4):
        feature = deepcopy(features[index % 2])
        feature["sample_id"] = f"train-{index}"
        feature["scene_id"] = f"frame-{index}"
        feature["split"] = "train"
        label = deepcopy(labels[index % 2])
        label["sample_id"] = feature["sample_id"]
        extra_features.append(feature)
        extra_labels.append(label)
    artifact = train_mlp(extra_features, extra_labels, seed=5, epochs=2, patience=2)
    model_path, _ = save_artifact(artifact, tmp_path / "ranker.pt")
    scorer = load_mlp_scorer(model_path)
    candidate = extra_features[0]["candidates"][0]
    contaminated = deepcopy(candidate)
    contaminated["gt_grasps"] = [[0, 0, 0, 0, 0]]
    contaminated["candidate_validity"] = not extra_labels[0]["candidate_labels"][0]["candidate_valid"]
    assert scorer(candidate) == scorer(contaminated)

    locked = deepcopy(extra_features)
    locked[0]["split"] = "test"
    with pytest.raises(ValueError):
        train_mlp(locked, extra_labels, seed=5, epochs=1)
