from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .dataset import load_label_index


def independent_recompute_local_validation(
    run_dir: str | Path,
    *,
    corrected_labels: str | Path,
) -> dict[str, Any]:
    """Recompute only from stable IDs and frozen labels, not main metrics."""

    root = Path(run_dir)
    decisions = pq.read_table(root / "validation/local_only_decisions.parquet").to_pylist()
    sample_ids = {str(row["sample_id"]) for row in decisions}
    labels = load_label_index(corrected_labels, sample_ids)
    recovered = harmful = baseline_successes = selected_successes = switches = 0
    row_mismatches = []
    for row in decisions:
        sample_id = str(row["sample_id"]); truth = labels[sample_id]
        baseline = str(row["baseline_candidate_id"]); selected = str(row["selected_candidate_id"])
        if baseline not in truth or selected not in truth:
            raise AssertionError("saved decision selected outside frozen candidate labels")
        baseline_correct = truth[baseline]; selected_correct = truth[selected]
        baseline_successes += baseline_correct; selected_successes += selected_correct
        switches += selected != baseline
        recovered += (not baseline_correct) and selected_correct
        harmful += baseline_correct and (not selected_correct)
        if bool(row["baseline_correct"]) != baseline_correct or bool(row["selected_correct"]) != selected_correct:
            row_mismatches.append(sample_id)
    total = len(decisions)
    result = {
        "schema_version": "1.0.0", "method": "P6_local_only_safe_gate",
        "total": total, "baseline_successes": baseline_successes,
        "selected_successes": selected_successes, "baseline_j1": baseline_successes / total,
        "selected_j1": selected_successes / total, "recovered": recovered,
        "harmful": harmful, "net": recovered - harmful, "switches": switches,
        "per_sample_correctness_mismatches": len(row_mismatches),
        "selected_candidate_ids_verified": len(row_mismatches) == 0,
    }
    primary = json.loads((root / "validation/VALIDATION_RESULTS.json").read_text())
    expected = primary["corrected"]
    result["aggregate_matches_main"] = all([
        total == expected["total"], baseline_successes == expected["baseline_successes"],
        selected_successes == expected["final_successes"], recovered == expected["recovered"],
        harmful == expected["harmful"], recovered - harmful == expected["net"], switches == expected["switches"],
    ])
    temporary = root / "independent_recompute_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(root / "independent_recompute_results.json")
    return result


def independent_recompute_p5_validation(
    run_dir: str | Path,
    *,
    corrected_labels: str | Path | None = None,
    legacy_labels: str | Path | None = None,
) -> dict[str, Any]:
    """Independently recompute frozen P5 selections against both label tracks.

    The selected candidate IDs are the only decision output consumed from the
    primary evaluator.  Ground-truth labels are loaded again from the frozen
    label files and all outcome counts are implemented here, independently of
    :mod:`p5_validation` and :func:`policy.full_denominator_metrics`.
    """

    if corrected_labels is None or legacy_labels is None:
        from .runner import VALIDATION_CORRECTED, VALIDATION_LEGACY

        corrected_labels = corrected_labels or VALIDATION_CORRECTED
        legacy_labels = legacy_labels or VALIDATION_LEGACY

    root = Path(run_dir)
    decision_path = root / "p5_validation/p5_decisions.parquet"
    main_path = root / "p5_validation/P5_VALIDATION_RESULTS.json"
    rows = pq.read_table(decision_path).to_pylist()
    if not rows:
        raise AssertionError("P5 decisions are empty")
    sample_ids = {str(row["sample_id"]) for row in rows}
    corrected = load_label_index(corrected_labels, sample_ids)
    legacy = load_label_index(legacy_labels, sample_ids)
    methods = sorted({str(row["method"]) for row in rows})
    main = json.loads(main_path.read_text(encoding="utf-8"))
    expected_total = int(main["expected_denominator"])
    output_methods: dict[str, Any] = {}
    all_match = True

    for method in methods:
        method_rows = [row for row in rows if str(row["method"]) == method]
        if len(method_rows) != expected_total:
            raise AssertionError(f"{method} does not cover the frozen P5 denominator")
        if len({str(row["sample_id"]) for row in method_rows}) != expected_total:
            raise AssertionError(f"{method} contains duplicate or missing sample IDs")

        track_results: dict[str, Any] = {}
        saved_field_mismatches = 0
        for track, labels, saved_baseline, saved_selected in (
            ("corrected", corrected, "baseline_correct", "selected_correct"),
            (
                "legacy",
                legacy,
                "legacy_baseline_correct",
                "legacy_selected_correct",
            ),
        ):
            baseline_successes = selected_successes = recovered = harmful = switches = 0
            for row in method_rows:
                sample_id = str(row["sample_id"])
                baseline_id = str(row["baseline_candidate_id"])
                selected_id = str(row["selected_candidate_id"])
                truth = labels[sample_id]
                if baseline_id not in truth or selected_id not in truth:
                    raise AssertionError("saved P5 decision selected outside frozen candidates")
                baseline_correct = bool(truth[baseline_id])
                selected_correct = bool(truth[selected_id])
                baseline_successes += int(baseline_correct)
                selected_successes += int(selected_correct)
                recovered += int((not baseline_correct) and selected_correct)
                harmful += int(baseline_correct and (not selected_correct))
                switches += int(selected_id != baseline_id)
                saved_field_mismatches += int(
                    bool(row[saved_baseline]) != baseline_correct
                    or bool(row[saved_selected]) != selected_correct
                )
            total = len(method_rows)
            changed = recovered + harmful
            track_results[track] = {
                "total": total,
                "baseline_successes": baseline_successes,
                "final_successes": selected_successes,
                "baseline_j1": baseline_successes / total,
                "final_j1": selected_successes / total,
                "recovered": recovered,
                "harmful": harmful,
                "net": recovered - harmful,
                "switches": switches,
                "switch_rate": switches / total,
                "harm_rate": harmful / baseline_successes if baseline_successes else 0.0,
                "outcome_changing_precision": recovered / changed if changed else 0.0,
            }

        main_method = main["methods"][method]
        aggregate_matches = all(
            track_results[track][field] == main_method[track][field]
            for track in ("corrected", "legacy")
            for field in (
                "total",
                "baseline_successes",
                "final_successes",
                "recovered",
                "harmful",
                "net",
                "switches",
            )
        )
        selected_ids_match_q_only = all(
            str(row["selected_candidate_id"]) == str(row["baseline_candidate_id"])
            for row in method_rows
        )
        method_result = {
            **track_results,
            "selected_candidate_ids_verified": True,
            "selected_ids_match_q_only": selected_ids_match_q_only,
            "saved_correctness_field_mismatches": saved_field_mismatches,
            "aggregate_matches_main": aggregate_matches,
        }
        output_methods[method] = method_result
        all_match = all_match and aggregate_matches and saved_field_mismatches == 0

    result = {
        "schema_version": "1.0.0",
        "recompute_kind": "independent_p5_validation_from_frozen_ids_and_labels",
        "expected_denominator_per_method": expected_total,
        "methods": output_methods,
        "all_methods_match_main": all_match,
        "selected_candidate_ids_verified": all(
            value["selected_candidate_ids_verified"] for value in output_methods.values()
        ),
        "formal_evidence_used": False,
    }
    temporary = root / "independent_recompute_results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(root / "independent_recompute_results.json")
    return result
