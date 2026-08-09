"""Deterministic acceptance rules. These never compute a local candidate score."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _candidate(response: Mapping[str, Any] | None, baseline: str) -> str:
    return baseline if response is None else str(response.get("selected_internal_candidate_id", baseline))


def direct(response: Mapping[str, Any] | None, baseline: str) -> str:
    return _candidate(response, baseline)


def baseline_aware(response: Mapping[str, Any] | None, baseline: str) -> str:
    if response is None or response.get("decision") != "SELECT_CANDIDATE":
        return baseline
    return _candidate(response, baseline)


def confidence_accept(response: Mapping[str, Any] | None, baseline: str, threshold: int) -> str:
    if response is None:
        return baseline
    selected = _candidate(response, baseline)
    if (
        response.get("decision") == "SELECT_CANDIDATE"
        and selected != baseline
        and int(response.get("switch_confidence", -1)) >= int(threshold)
        and response.get("evidence_reliability") == "HIGH"
    ):
        return selected
    return baseline


def self_consistent(
    first: Mapping[str, Any] | None, second: Mapping[str, Any] | None,
    baseline: str, threshold: int,
) -> str:
    left = confidence_accept(first, baseline, threshold)
    right = confidence_accept(second, baseline, threshold)
    return left if left != baseline and left == right else baseline


def cross_model_consensus(
    er2: Mapping[str, Any] | None, flash: Mapping[str, Any] | None,
    baseline: str, er2_threshold: int, flash_threshold: int,
) -> str:
    left = confidence_accept(er2, baseline, er2_threshold)
    right = confidence_accept(flash, baseline, flash_threshold)
    return left if left != baseline and left == right else baseline


def assert_selected_is_frozen(selected: str | None, frozen_ids: Sequence[str]) -> None:
    if selected is not None and str(selected) not in set(map(str, frozen_ids)):
        raise ValueError("selected candidate is not in the frozen Top-5")
