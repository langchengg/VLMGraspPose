from copy import deepcopy

import numpy as np

from .geometry import assert_candidate_set_unchanged, neutral_expectation
from .schema import INFERENCE_FEATURE_ALLOWLIST


RANKER_NAMES = (
    "legacy",
    "q_only",
    "q_mask",
    "rule_2d_equal",
    "q_mask_width_angle",
    "q_mask_width_depth",
    "rule_fixed_v1",
    "rule_val_tuned",
    "mlp",
)

FIXED_WEIGHTS = {
    "q": 0.45,
    "mask_consistency": 0.25,
    "width_compatibility": 0.10,
    "angle_consistency": 0.05,
    "depth_geometry": 0.05,
    "safety": 0.10,
}


def _allowlisted_feature(candidate, name):
    if name not in INFERENCE_FEATURE_ALLOWLIST:
        raise ValueError(f"ranker attempted to read non-allowlisted feature: {name}")
    feature = candidate.get("features", {}).get(name, {})
    value = feature.get("value")
    reliability = float(feature.get("reliability", 0.0) or 0.0)
    return value, reliability


def expected_feature(candidate, name):
    value, reliability = _allowlisted_feature(candidate, name)
    return neutral_expectation(value, reliability)


def _weighted_score(candidate, weights):
    if not weights or any(float(weight) < 0 for weight in weights.values()):
        raise ValueError("ranker weights must be non-negative")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("ranker weights must have positive sum")
    components = {
        name: expected_feature(candidate, name)
        for name in weights
    }
    score = sum(float(weights[name]) * components[name] for name in weights) / total
    return float(score), {
        "weights": {name: float(value) / total for name, value in weights.items()},
        "expected_features": components,
    }


def score_candidate(candidate, ranker, *, tuned_weights=None, mlp_scorer=None):
    if ranker == "legacy":
        return -float(candidate["legacy_rank"]), {"legacy_order": True}
    if ranker == "q_only":
        value, _ = _allowlisted_feature(candidate, "q")
        return float(value), {"q": float(value)}
    if ranker == "q_mask":
        return _weighted_score(candidate, {"q": 1.0, "mask_consistency": 1.0})
    if ranker == "rule_2d_equal":
        return _weighted_score(
            candidate,
            {"q": 1.0, "mask_consistency": 1.0, "width_compatibility": 1.0},
        )
    if ranker == "q_mask_width_angle":
        return _weighted_score(
            candidate,
            {
                "q": 1.0,
                "mask_consistency": 1.0,
                "width_compatibility": 1.0,
                "angle_consistency": 1.0,
            },
        )
    if ranker == "q_mask_width_depth":
        return _weighted_score(
            candidate,
            {
                "q": 1.0,
                "mask_consistency": 1.0,
                "width_compatibility": 1.0,
                "depth_geometry": 1.0,
            },
        )
    if ranker == "rule_fixed_v1":
        return _weighted_score(candidate, FIXED_WEIGHTS)
    if ranker == "rule_val_tuned":
        if tuned_weights is None:
            raise ValueError("rule_val_tuned requires validation-only tuned weights")
        return _weighted_score(candidate, tuned_weights)
    if ranker == "mlp":
        if mlp_scorer is None:
            raise ValueError("mlp ranker requires a trained MLP scorer")
        return float(mlp_scorer(candidate)), {"model": "mlp"}
    raise ValueError(f"unknown ranker: {ranker}")


def rank_candidates(candidates, ranker, *, tuned_weights=None, mlp_scorer=None):
    before = deepcopy(list(candidates))
    scored = []
    for candidate in candidates:
        score, decomposition = score_candidate(
            candidate,
            ranker,
            tuned_weights=tuned_weights,
            mlp_scorer=mlp_scorer,
        )
        if not np.isfinite(score):
            raise AssertionError(f"non-finite {ranker} score for {candidate['candidate_id']}")
        scored.append((candidate, float(score), decomposition))
    if ranker == "legacy":
        scored.sort(key=lambda item: (item[0]["legacy_rank"], item[0]["candidate_id"]))
    else:
        scored.sort(
            key=lambda item: (
                -item[1],
                -float(item[0]["q_raw"]),
                int(item[0]["legacy_rank"]),
                str(item[0]["candidate_id"]),
            )
        )
    ranked = []
    for new_rank, (candidate, score, decomposition) in enumerate(scored):
        item = deepcopy(candidate)
        item["rerank_score"] = score
        item["rerank_rank"] = new_rank
        item["score_decomposition"] = decomposition
        ranked.append(item)
    assert_candidate_set_unchanged(before, ranked)
    return ranked


def q_only_matches_q_rank(candidates):
    ranked = rank_candidates(candidates, "q_only")
    return all(candidate["q_rank"] == index for index, candidate in enumerate(ranked))
