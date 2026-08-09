from __future__ import annotations

from copy import deepcopy

from scripts.run_full_gqcnn_scoring import completion_checks


def _complete_accounting() -> tuple[list[dict[str, int]], dict, dict]:
    selected = [{"candidate_count": 2}, {"candidate_count": 0}]
    progress = {
        "total_samples": 2,
        "terminal_samples": 2,
        "completed_nonempty_samples": 1,
        "skipped_empty_samples": 1,
        "failed_samples": 0,
        "scored_candidates": 2,
        "remaining_candidates": 0,
    }
    statistics = {
        "total_samples": 2,
        "terminal_samples": 2,
        "scored_nonempty_samples": 1,
        "skipped_valid_empty_samples": 1,
        "failed_samples": 0,
        "corrupt_committed_samples": 0,
        "expected_candidates": 2,
        "scored_candidates": 2,
        "finite_q_values": 2,
        "invalid_q_values": 0,
    }
    return selected, progress, statistics


def test_completion_checks_accept_exact_selected_workload_accounting() -> None:
    selected, progress, statistics = _complete_accounting()
    assert all(completion_checks(progress, statistics, selected).values())


def test_completion_checks_reject_corrupt_missing_and_nonfinite_results() -> None:
    selected, progress, statistics = _complete_accounting()
    corrupt = deepcopy(statistics)
    corrupt["corrupt_committed_samples"] = 1
    assert not all(completion_checks(progress, corrupt, selected).values())

    missing = deepcopy(progress)
    missing["terminal_samples"] = 1
    assert not all(completion_checks(missing, statistics, selected).values())

    nonfinite = deepcopy(statistics)
    nonfinite["finite_q_values"] = 1
    nonfinite["invalid_q_values"] = 1
    assert not all(completion_checks(progress, nonfinite, selected).values())
