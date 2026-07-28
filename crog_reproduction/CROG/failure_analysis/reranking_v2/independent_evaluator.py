"""Independent count-based checker for a frozen V2 evaluation.

This intentionally does not call ``metrics.evaluate_rankings``. It is a second
implementation of the key invariants used to catch accidental cohort, ordering,
or sign errors in the primary evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from .datasets import load_joined
from .evaluation import load_prediction_rankings
from .schema import atomic_write_json


def recompute_counts(
    *,
    features: str | Path,
    labels: str | Path,
    predictions: str | Path | None,
) -> dict:
    samples = load_joined(features, labels)
    rankings, _ = load_prediction_rankings(predictions, samples)
    original_correct = selected_correct = oracle = 0
    recovered = harmful = neutral_switch = switched = 0
    for sample in samples:
        original_order = [
            str(candidate["candidate_id"])
            for candidate in sample.feature["candidates"]
        ]
        order = rankings[sample.sample_id]
        if len(order) != 5 or set(order) != set(original_order):
            raise AssertionError("frozen candidate set/order membership changed")
        by_id = {
            str(value["candidate_id"]): bool(value["candidate_correct"])
            for value in sample.label["candidate_labels"]
        }
        before = by_id[original_order[0]]
        after = by_id[order[0]]
        original_correct += int(before)
        selected_correct += int(after)
        oracle += int(any(by_id[candidate_id] for candidate_id in original_order))
        switched_here = order[0] != original_order[0]
        switched += int(switched_here)
        recovered += int((not before) and after)
        harmful += int(before and (not after))
        neutral_switch += int(switched_here and before == after)
    count = len(samples)
    discordant = recovered + harmful
    return {
        "sample_count": count,
        "q_only_success_count": original_correct,
        "selected_success_count": selected_correct,
        "q_only_j1": original_correct / count,
        "selected_j1": selected_correct / count,
        "delta_j1_percentage_points": 100
        * (selected_correct - original_correct)
        / count,
        "oracle_success_count": oracle,
        "oracle_at_5": oracle / count,
        "recovered": recovered,
        "harmful": harmful,
        "net_recovered": recovered - harmful,
        "neutral_switch": neutral_switch,
        "switch_coverage": switched / count,
        "mcnemar_exact_two_sided_pvalue": (
            float(binomtest(recovered, discordant, p=0.5).pvalue)
            if discordant
            else 1.0
        ),
        "identity_passed": (
            selected_correct - original_correct == recovered - harmful
        ),
    }


def assert_matches_primary(
    primary_summary: dict,
    independent_summary: dict,
) -> None:
    mapping = {
        "sample_count": "sample_count",
        "q_only_j1": "q_only_j1",
        "legacy_or_corrected_j1": "selected_j1",
        "delta_j1_percentage_points": "delta_j1_percentage_points",
        "oracle_at_5": "oracle_at_5",
        "recovered": "recovered",
        "harmful": "harmful",
        "net_recovered": "net_recovered",
        "neutral_switch": "neutral_switch",
        "switch_coverage": "switch_coverage",
        "mcnemar_exact_two_sided_pvalue": "mcnemar_exact_two_sided_pvalue",
    }
    for primary_name, independent_name in mapping.items():
        left = primary_summary[primary_name]
        right = independent_summary[independent_name]
        if isinstance(left, float) or isinstance(right, float):
            if abs(float(left) - float(right)) > 1e-12:
                raise AssertionError(
                    f"independent evaluator mismatch: {primary_name}"
                )
        elif left != right:
            raise AssertionError(
                f"independent evaluator mismatch: {primary_name}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = recompute_counts(
        features=args.features,
        labels=args.labels,
        predictions=args.predictions,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
