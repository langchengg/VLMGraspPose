from __future__ import annotations

from typing import Any

import numpy as np


def ranking_metrics(
    labels: np.ndarray,
    rankings: np.ndarray,
    *,
    probabilities: np.ndarray | None = None,
    bins: int = 15,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.float64)
    order = np.asarray(rankings, dtype=np.int64)
    if y.ndim != 2 or y.shape[1] != 5 or order.shape != y.shape:
        raise ValueError("labels and rankings must both have shape [N,5]")
    if not np.array_equal(np.sort(order, axis=1), np.broadcast_to(np.arange(5), order.shape)):
        raise ValueError("ranking rows must be permutations of the frozen five candidates")
    ranked = np.take_along_axis(y, order, axis=1)
    top_correct = ranked[:, 0] > 0.5
    oracle = y.max(axis=1) > 0.5
    positive = ranked > 0.5
    first = np.where(positive.any(axis=1), positive.argmax(axis=1) + 1, 0)
    reciprocal = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    discounts = 1.0 / np.log2(np.arange(2, 7))
    dcg = (ranked * discounts).sum(axis=1)
    ideal = np.sort(y, axis=1)[:, ::-1]
    idcg = (ideal * discounts).sum(axis=1)
    ndcg = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    result: dict[str, Any] = {
        "sample_count": int(len(y)),
        "correct": int(top_correct.sum()),
        "j_at_1": float(top_correct.mean()),
        "oracle_correct": int(oracle.sum()),
        "oracle_at_5": float(oracle.mean()),
        "oracle_gap": float(oracle.mean() - top_correct.mean()),
        "mrr_at_5": float(reciprocal.mean()),
        "ndcg_at_5": float(ndcg.mean()),
        "mean_first_correct_rank": float(first[first > 0].mean()) if (first > 0).any() else None,
        "top_correct": top_correct,
        "oracle": oracle,
        "first_correct_rank": first,
    }
    if probabilities is not None:
        p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
        if p.shape != y.shape:
            raise ValueError("candidate probabilities must have shape [N,5]")
        result["candidate_brier"] = float(np.square(p - y).mean())
        result["candidate_nll"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        flattened_p = p.ravel(); flattened_y = y.ravel()
        boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
        ece = 0.0
        reliability = []
        for index in range(int(bins)):
            member = (flattened_p >= boundaries[index]) & (
                flattened_p <= boundaries[index + 1] if index == bins - 1 else flattened_p < boundaries[index + 1]
            )
            if not member.any():
                reliability.append({"lower": float(boundaries[index]), "upper": float(boundaries[index + 1]), "count": 0, "confidence": None, "accuracy": None})
                continue
            confidence = float(flattened_p[member].mean())
            accuracy = float(flattened_y[member].mean())
            ece += member.mean() * abs(confidence - accuracy)
            reliability.append({"lower": float(boundaries[index]), "upper": float(boundaries[index + 1]), "count": int(member.sum()), "confidence": confidence, "accuracy": accuracy})
        result["ece"] = float(ece)
        result["reliability"] = reliability
    return result


def paired_switch_metrics(
    labels: np.ndarray,
    reference_rankings: np.ndarray,
    challenger_rankings: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(labels)
    reference = np.asarray(reference_rankings, dtype=np.int64)
    challenger = np.asarray(challenger_rankings, dtype=np.int64)
    reference_selected = reference[:, 0]
    challenger_selected = challenger[:, 0]
    rows = np.arange(len(y))
    reference_correct = y[rows, reference_selected] > 0.5
    challenger_correct = y[rows, challenger_selected] > 0.5
    switched = reference_selected != challenger_selected
    recovered = (~reference_correct) & challenger_correct
    harmful = reference_correct & (~challenger_correct)
    neutral = switched & (reference_correct == challenger_correct)
    changed = int(recovered.sum() + harmful.sum())
    reference_j = float(reference_correct.mean())
    challenger_j = float(challenger_correct.mean())
    oracle = float((y.max(axis=1) > 0.5).mean())
    denominator = oracle - reference_j
    return {
        "sample_count": int(len(y)),
        "reference_correct": int(reference_correct.sum()),
        "challenger_correct": int(challenger_correct.sum()),
        "recovered": int(recovered.sum()),
        "harmful": int(harmful.sum()),
        "net_recovered": int(recovered.sum() - harmful.sum()),
        "delta_j_at_1": challenger_j - reference_j,
        "neutral_switch": int(neutral.sum()),
        "switch_count": int(switched.sum()),
        "switch_coverage": float(switched.mean()),
        "outcome_changing_precision": None if changed == 0 else float(recovered.sum() / changed),
        "headroom_recovered": None if denominator <= 0 else float((challenger_j - reference_j) / denominator),
        "reference_correct_mask": reference_correct,
        "challenger_correct_mask": challenger_correct,
        "recovered_mask": recovered,
        "harmful_mask": harmful,
        "switched_mask": switched,
    }


def strip_metric_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not isinstance(value, np.ndarray)}
