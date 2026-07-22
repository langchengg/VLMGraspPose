"""Mutually exclusive P0--P6 target-consistency failure taxonomy.

The labels describe candidate availability and GT *target consistency*.  They
must not be interpreted as physical grasp success or 6-DoF pose accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import pandas as pd


P0_TECHNICAL_FAILURE = "P0_technical_failure"
P1_NO_OFFICIAL_CANDIDATE = "P1_no_official_candidate"
P2_NO_GT_POSITIVE_IN_OFFICIAL_POOL = "P2_no_gt_positive_in_official_pool"
P3_HARD_FILTER_REJECTS_ALL_GT_POSITIVE = (
    "P3_hard_filter_rejects_all_gt_positive"
)
P4_HARD_FILTER_MISDIRECTION = "P4_hard_filter_misdirection"
P5_RANKING_ERROR_WITHIN_FILTERED_POOL = (
    "P5_ranking_error_within_filtered_pool"
)
P6_TARGET_CONSISTENT_TOP1 = "P6_target_consistent_top1"

PRIMARY_FAILURE_CLASSES = (
    P0_TECHNICAL_FAILURE,
    P1_NO_OFFICIAL_CANDIDATE,
    P2_NO_GT_POSITIVE_IN_OFFICIAL_POOL,
    P3_HARD_FILTER_REJECTS_ALL_GT_POSITIVE,
    P4_HARD_FILTER_MISDIRECTION,
    P5_RANKING_ERROR_WITHIN_FILTERED_POOL,
    P6_TARGET_CONSISTENT_TOP1,
)

SCIENTIFIC_STATUSES = frozenset({"ok", "no_target_grasp", "no_official_grasp"})


class FailureTaxonomyError(RuntimeError):
    """Raised when frozen candidate artifacts imply contradictory classes."""


@dataclass(frozen=True)
class TaxonomyFacts:
    """Minimal Boolean state that uniquely determines a P0--P6 class."""

    technical_failure: bool
    has_official_candidate: bool
    has_pred_filtered_candidate: bool
    has_gt_positive_anywhere: bool
    has_gt_positive_after_pred_filter: bool
    hard_filter_top1_is_gt_positive: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def primary_failure_from_facts(facts: TaxonomyFacts) -> str:
    """Assign exactly one primary class using the pre-registered precedence."""

    if facts.technical_failure:
        return P0_TECHNICAL_FAILURE
    if not facts.has_official_candidate:
        return P1_NO_OFFICIAL_CANDIDATE
    if not facts.has_gt_positive_anywhere:
        return P2_NO_GT_POSITIVE_IN_OFFICIAL_POOL
    if not facts.has_pred_filtered_candidate:
        return P3_HARD_FILTER_REJECTS_ALL_GT_POSITIVE
    if not facts.has_gt_positive_after_pred_filter:
        return P4_HARD_FILTER_MISDIRECTION
    if not facts.hard_filter_top1_is_gt_positive:
        return P5_RANKING_ERROR_WITHIN_FILTERED_POOL
    return P6_TARGET_CONSISTENT_TOP1


def _optional_count(sample: Mapping[str, Any], name: str) -> int | None:
    value = sample.get(name)
    if value in (None, "") or pd.isna(value):
        return None
    try:
        result = int(float(value))
    except (TypeError, ValueError) as error:
        raise FailureTaxonomyError(
            f"sample {sample.get('sample_id')} has invalid {name}={value!r}"
        ) from error
    if result < 0:
        raise FailureTaxonomyError(
            f"sample {sample.get('sample_id')} has negative {name}"
        )
    return result


def _facts_for_sample(
    sample: Mapping[str, Any],
    candidates: pd.DataFrame,
    *,
    positive_column: str,
) -> TaxonomyFacts:
    sample_id = str(sample.get("sample_id", ""))
    status = str(sample.get("pred_status", sample.get("status", ""))).strip()
    technical = status not in SCIENTIFIC_STATUSES
    if technical:
        # Candidate-state fields of failed preprocessing/infrastructure rows
        # are not evidence and therefore cannot override P0.
        return TaxonomyFacts(True, False, False, False, False, False)

    if not candidates.empty and positive_column not in candidates:
        raise FailureTaxonomyError(
            f"candidate rows for {sample_id} lack {positive_column}"
        )
    for required in ("pred_filter_pass", "is_baseline_top1"):
        if not candidates.empty and required not in candidates:
            raise FailureTaxonomyError(
                f"candidate rows for {sample_id} lack {required}"
            )

    official_count = len(candidates)
    saved_official = _optional_count(sample, "n_official_candidates")
    if saved_official is not None and saved_official != official_count:
        raise FailureTaxonomyError(
            f"sample {sample_id} official count mismatch: "
            f"sample={saved_official}, candidates={official_count}"
        )
    if (status == "no_official_grasp") != (official_count == 0):
        raise FailureTaxonomyError(
            f"sample {sample_id} status/official-pool contradiction: "
            f"status={status}, candidates={official_count}"
        )

    if official_count:
        positive = candidates[positive_column].fillna(False).astype(bool)
        passed = candidates["pred_filter_pass"].fillna(False).astype(bool)
        selected = candidates["is_baseline_top1"].fillna(False).astype(bool)
    else:
        positive = pd.Series([], dtype=bool)
        passed = pd.Series([], dtype=bool)
        selected = pd.Series([], dtype=bool)
    filtered_count = int(passed.sum())
    saved_filtered = _optional_count(sample, "n_pred_filtered_candidates")
    if saved_filtered is not None and saved_filtered != filtered_count:
        raise FailureTaxonomyError(
            f"sample {sample_id} filtered count mismatch: "
            f"sample={saved_filtered}, candidates={filtered_count}"
        )
    if int(selected.sum()) > 1:
        raise FailureTaxonomyError(f"sample {sample_id} has multiple baseline top-1 rows")
    if bool((selected & ~passed).any()):
        raise FailureTaxonomyError(
            f"sample {sample_id} baseline top-1 did not pass the predicted filter"
        )
    if filtered_count > 0 and int(selected.sum()) != 1:
        raise FailureTaxonomyError(
            f"sample {sample_id} has a filtered pool but no unique baseline top-1"
        )
    if filtered_count == 0 and selected.any():
        raise FailureTaxonomyError(
            f"sample {sample_id} selects a top-1 from an empty filtered pool"
        )
    expected_status = "ok" if filtered_count else "no_target_grasp"
    if official_count and status != expected_status:
        raise FailureTaxonomyError(
            f"sample {sample_id} status/filter contradiction: "
            f"status={status}, expected={expected_status}"
        )

    top1_positive = bool((selected & positive).any())
    return TaxonomyFacts(
        technical_failure=False,
        has_official_candidate=official_count > 0,
        has_pred_filtered_candidate=filtered_count > 0,
        has_gt_positive_anywhere=bool(positive.any()),
        has_gt_positive_after_pred_filter=bool((positive & passed).any()),
        hard_filter_top1_is_gt_positive=top1_positive,
    )


def assign_failure_taxonomy(
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    positive_column: str = "gt_target_positive_primary",
) -> pd.DataFrame:
    """Augment every sample with facts and exactly one P0--P6 primary class."""

    if "sample_id" not in samples:
        raise FailureTaxonomyError("sample table lacks sample_id")
    if samples["sample_id"].astype(str).duplicated().any():
        raise FailureTaxonomyError("sample table contains duplicate sample_id values")
    if "sample_id" not in candidates:
        if len(candidates):
            raise FailureTaxonomyError("candidate table lacks sample_id")
        candidates = candidates.assign(sample_id=pd.Series(dtype=str))
    sample_ids = set(samples["sample_id"].astype(str))
    unknown = set(candidates["sample_id"].astype(str)) - sample_ids
    if unknown:
        raise FailureTaxonomyError(
            f"candidate table refers to unknown samples: {sorted(unknown)[:5]}"
        )
    groups = {
        str(sample_id): group
        for sample_id, group in candidates.groupby("sample_id", sort=False)
    }
    empty = candidates.iloc[0:0]
    rows: list[dict[str, Any]] = []
    for sample in samples.to_dict(orient="records"):
        facts = _facts_for_sample(
            sample,
            groups.get(str(sample["sample_id"]), empty),
            positive_column=positive_column,
        )
        rows.append(
            {
                **facts.to_dict(),
                "primary_failure_class": primary_failure_from_facts(facts),
            }
        )
    addition = pd.DataFrame(rows, index=samples.index)
    overlapping = sorted(set(addition) & set(samples))
    if overlapping:
        raise FailureTaxonomyError(
            f"taxonomy columns already exist in sample table: {overlapping}"
        )
    result = pd.concat([samples.copy(), addition], axis=1)
    validate_failure_taxonomy(result)
    return result


def validate_failure_taxonomy(samples: pd.DataFrame) -> None:
    """Prove that every row has one known primary class and consistent facts."""

    if "primary_failure_class" not in samples:
        raise FailureTaxonomyError("sample table lacks primary_failure_class")
    invalid = sorted(
        set(samples["primary_failure_class"].astype(str)) - set(PRIMARY_FAILURE_CLASSES)
    )
    if invalid:
        raise FailureTaxonomyError(f"unknown primary failure classes: {invalid}")
    required = tuple(TaxonomyFacts.__dataclass_fields__)
    missing = [name for name in required if name not in samples]
    if missing:
        raise FailureTaxonomyError(f"sample table lacks taxonomy facts: {missing}")
    for row in samples.to_dict(orient="records"):
        facts = TaxonomyFacts(**{name: bool(row[name]) for name in required})
        expected = primary_failure_from_facts(facts)
        if row["primary_failure_class"] != expected:
            raise FailureTaxonomyError(
                f"sample {row.get('sample_id')} taxonomy mismatch: "
                f"{row['primary_failure_class']} != {expected}"
            )


def primary_failure_counts(samples: pd.DataFrame) -> dict[str, int]:
    """Return all P0--P6 counts, including zero-count classes."""

    validate_failure_taxonomy(samples)
    observed = samples["primary_failure_class"].value_counts().to_dict()
    return {name: int(observed.get(name, 0)) for name in PRIMARY_FAILURE_CLASSES}


__all__ = [
    "FailureTaxonomyError",
    "P0_TECHNICAL_FAILURE",
    "P1_NO_OFFICIAL_CANDIDATE",
    "P2_NO_GT_POSITIVE_IN_OFFICIAL_POOL",
    "P3_HARD_FILTER_REJECTS_ALL_GT_POSITIVE",
    "P4_HARD_FILTER_MISDIRECTION",
    "P5_RANKING_ERROR_WITHIN_FILTERED_POOL",
    "P6_TARGET_CONSISTENT_TOP1",
    "PRIMARY_FAILURE_CLASSES",
    "SCIENTIFIC_STATUSES",
    "TaxonomyFacts",
    "assign_failure_taxonomy",
    "primary_failure_counts",
    "primary_failure_from_facts",
    "validate_failure_taxonomy",
]
