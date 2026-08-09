from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _rank_vector(ranking: list[str]) -> list[int]:
    if len(ranking) != 5 or set(ranking) != {"A", "B", "C", "D", "E"}:
        raise ValueError("stability ranking must contain A-E exactly once")
    positions = {candidate: index for index, candidate in enumerate(ranking)}
    return [positions[candidate] for candidate in ("A", "B", "C", "D", "E")]


def compute_stability(
    repeated_decisions: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in repeated_decisions:
        grouped[(str(row["model_id"]), str(row["sample_id"]))].append(dict(row))
    per_sample: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model_id, sample_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: int(item["replicate_id"]))
        if [int(row["replicate_id"]) for row in rows] != [1, 2, 3]:
            raise ValueError(f"stability requires replicate_id 1,2,3 for {model_id}/{sample_id}")
        selected = [str(row["selected_candidate_id"]) for row in rows]
        rankings = [[str(value) for value in row["ranking"]] for row in rows]
        valid_flags = [bool(row.get("valid", False)) for row in rows]
        all_valid = all(valid_flags)
        kendall: list[float] = []
        spearman: list[float] = []
        if all_valid:
            vectors = [_rank_vector(ranking) for ranking in rankings]
            for left, right in combinations(vectors, 2):
                if left == right:
                    # scipy may return 0.9999999999999999 for an identical
                    # permutation.  Preserve the mathematically exact agreement
                    # so downstream JSON and strict independent checks do not
                    # report a spurious instability.
                    kendall.append(1.0)
                    spearman.append(1.0)
                else:
                    kendall.append(float(kendalltau(left, right).statistic))
                    spearman.append(float(spearmanr(left, right).statistic))
        confidences = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)
        margins = np.asarray([float(row["score_margin_top1_top2"]) for row in rows], dtype=np.float64)
        decisions = [str(row["decision"]) for row in rows]
        counts = Counter(selected)
        record = {
            "model_id": model_id,
            "sample_id": sample_id,
            "valid_replicate_count": sum(valid_flags),
            "all_replicates_valid": all_valid,
            "selected_candidate_exact_agreement": (len(counts) == 1) if all_valid else None,
            "selected_candidate_majority_agreement": (max(counts.values()) / 3.0) if all_valid else None,
            "complete_ranking_exact_agreement": (
                len({tuple(value) for value in rankings}) == 1
            ) if all_valid else None,
            "kendall_tau_mean": float(np.mean(kendall)) if all_valid else None,
            "spearman_rank_correlation_mean": float(np.mean(spearman)) if all_valid else None,
            "confidence_mean": float(confidences.mean()),
            "confidence_std": float(confidences.std(ddof=0)),
            "score_margin_mean": float(margins.mean()),
            "score_margin_std": float(margins.std(ddof=0)),
            "decision_consistency": (len(set(decisions)) == 1) if all_valid else None,
            "technical_fallback_consistency": (not all_valid) and len(set(decisions)) == 1,
            "decision_values": decisions,
        }
        per_sample.append(record)
        by_model[model_id].append(record)
    summary: dict[str, Any] = {}
    for model_id, rows in sorted(by_model.items()):
        complete = [row for row in rows if row["all_replicates_valid"]]
        summary[model_id] = {
            "sample_count": len(rows),
            "all_replicates_valid_sample_count": len(complete),
            "all_replicates_valid_coverage": len(complete) / len(rows),
            "selected_candidate_exact_agreement_rate": float(
                np.mean([row["selected_candidate_exact_agreement"] for row in complete])
            ) if complete else None,
            "selected_candidate_majority_agreement_mean": float(
                np.mean([row["selected_candidate_majority_agreement"] for row in complete])
            ) if complete else None,
            "complete_ranking_exact_agreement_rate": float(
                np.mean([row["complete_ranking_exact_agreement"] for row in complete])
            ) if complete else None,
            "kendall_tau_mean": float(np.mean([row["kendall_tau_mean"] for row in complete]))
            if complete else None,
            "spearman_rank_correlation_mean": float(
                np.mean([row["spearman_rank_correlation_mean"] for row in complete])
            ) if complete else None,
            "decision_consistency_rate": float(
                np.mean([row["decision_consistency"] for row in complete])
            ) if complete else None,
            "technical_fallback_consistency_rate": float(
                np.mean([row["technical_fallback_consistency"] for row in rows])
            ),
            "confidence_within_sample_std_mean": float(
                np.mean([row["confidence_std"] for row in complete])
            ) if complete else None,
            "score_margin_within_sample_std_mean": float(
                np.mean([row["score_margin_std"] for row in complete])
            ) if complete else None,
        }
    return per_sample, summary
