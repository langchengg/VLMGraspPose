"""Mutually exclusive native and post-R7 bottleneck taxonomies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


NATIVE_CLASSES = (
    "T0_technical_or_annotation_failure",
    "T1_deployed_native_success",
    "T2_native_ranking_limited_within_top5",
    "T3_correct_candidate_below_top5",
    "T4_grounding_limited",
    "T5_grounding_plus_selection_within_top5",
    "T6_grounding_plus_deep_ranking",
    "T7_grasper_or_candidate_generation_limited",
)

POST_R7_CLASSES = (
    "R0_technical",
    "R1_reranker_limited",
    "R2_residual_grounding_limited",
    "R3_residual_grounding_plus_selection",
    "R4_residual_grounding_plus_deep_ranking",
    "R5_residual_generator_limited_under_GT",
)

_NATIVE_REQUIRED = {
    "sample_id",
    "technical_failure",
    "pred_native_correct",
    "pred_top5_positive",
    "pred_all_positive",
    "gt_native_correct",
    "gt_top5_positive",
    "gt_all_positive",
}


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} misses columns: {missing}")


def _binary(frame: pd.DataFrame, columns: set[str], *, name: str) -> pd.DataFrame:
    result = frame.copy()
    for column in sorted(columns):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError(f"{name}.{column} must be finite binary")
        result[column] = values.astype(bool)
    return result


def _validate_nesting(frame: pd.DataFrame, *, route_column: str) -> None:
    nontechnical = ~frame["technical_failure"]
    implications = (
        ("pred_native_correct", "pred_top5_positive"),
        ("pred_top5_positive", "pred_all_positive"),
        ("gt_native_correct", "gt_top5_positive"),
        ("gt_top5_positive", "gt_all_positive"),
    )
    for narrow, broad in implications:
        if (nontechnical & frame[narrow] & ~frame[broad]).any():
            raise ValueError(f"invalid taxonomy flags: {narrow} must imply {broad}")
    d1 = nontechnical & frame[route_column].astype(str).str.upper().eq("D1")
    if d1.any():
        for column in ("pred_top10_positive", "gt_top10_positive"):
            if column not in frame:
                raise ValueError(f"D1 taxonomy requires {column}")
        if (d1 & frame["pred_top5_positive"] & ~frame["pred_top10_positive"]).any():
            raise ValueError("D1 pred Top-5 must imply pred Top-10")
        if (d1 & frame["pred_top10_positive"] & ~frame["pred_all_positive"]).any():
            raise ValueError("D1 pred Top-10 must imply pred all")
        if (d1 & frame["gt_top5_positive"] & ~frame["gt_top10_positive"]).any():
            raise ValueError("D1 GT Top-5 must imply GT Top-10")
        if (d1 & frame["gt_top10_positive"] & ~frame["gt_all_positive"]).any():
            raise ValueError("D1 GT Top-10 must imply GT all")


def classify_native_taxonomy(
    frame: pd.DataFrame, *, route_column: str = "route"
) -> pd.DataFrame:
    """Assign exactly one T0--T7 class to every denominator sample.

    D1 rows additionally receive ``pred_deep_rank_flag`` and
    ``gt_deep_rank_flag`` when their positive is below Top-5.
    """

    _require_columns(frame, _NATIVE_REQUIRED | {route_column}, name="native taxonomy")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("native taxonomy input must be non-empty and unique by sample_id")
    binary_columns = _NATIVE_REQUIRED - {"sample_id"}
    for optional in ("pred_top10_positive", "gt_top10_positive"):
        if optional in frame:
            binary_columns.add(optional)
    work = _binary(frame, binary_columns, name="native taxonomy")
    _validate_nesting(work, route_column=route_column)

    technical = work["technical_failure"]
    a1 = work["pred_native_correct"]
    p5 = work["pred_top5_positive"]
    pa = work["pred_all_positive"]
    g1 = work["gt_native_correct"]
    g5 = work["gt_top5_positive"]
    ga = work["gt_all_positive"]
    conditions = (
        technical,
        ~technical & a1,
        ~technical & ~a1 & p5,
        ~technical & ~a1 & ~p5 & pa,
        ~technical & ~pa & g1,
        ~technical & ~pa & ~g1 & g5,
        ~technical & ~pa & ~g5 & ga,
        ~technical & ~pa & ~ga,
    )
    membership = np.column_stack([condition.to_numpy(bool) for condition in conditions])
    totals = membership.sum(axis=1)
    if not np.equal(totals, 1).all():
        bad = work.loc[totals != 1, "sample_id"].astype(str).head().tolist()
        raise RuntimeError(f"native taxonomy is not mutually exclusive/exhaustive: {bad}")
    work["native_taxonomy"] = np.asarray(NATIVE_CLASSES, dtype=object)[
        membership.argmax(axis=1)
    ]
    work["pred_deep_rank_flag"] = ""
    work["gt_deep_rank_flag"] = ""
    d1 = work[route_column].astype(str).str.upper().eq("D1")
    pred_deep = d1 & work["native_taxonomy"].eq(NATIVE_CLASSES[3])
    gt_deep = d1 & work["native_taxonomy"].eq(NATIVE_CLASSES[6])
    if d1.any():
        work.loc[pred_deep & work["pred_top10_positive"], "pred_deep_rank_flag"] = (
            "rank_6_to_10"
        )
        work.loc[pred_deep & ~work["pred_top10_positive"], "pred_deep_rank_flag"] = (
            "rank_below_10"
        )
        work.loc[gt_deep & work["gt_top10_positive"], "gt_deep_rank_flag"] = (
            "rank_6_to_10"
        )
        work.loc[gt_deep & ~work["gt_top10_positive"], "gt_deep_rank_flag"] = (
            "rank_below_10"
        )
    return work


def classify_post_r7_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify only frozen-final residual failures into R0--R5.

    Frozen-final successes remain in the returned denominator with an empty
    class and ``is_residual_failure=False``.  This prevents their accidental
    inclusion in residual percentages while preserving pair alignment.
    """

    required = {
        "sample_id",
        "technical_failure",
        "final_correct",
        "pred_top5_positive",
        "pred_all_positive",
        "gt_native_correct",
        "gt_top5_positive",
        "gt_all_positive",
    }
    _require_columns(frame, required, name="post-R7 taxonomy")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("post-R7 input must be non-empty and unique by sample_id")
    work = _binary(frame, required - {"sample_id"}, name="post-R7 taxonomy")
    residual = ~work["final_correct"]
    evaluable = residual & ~work["technical_failure"]
    implications = (
        ("pred_top5_positive", "pred_all_positive"),
        ("gt_native_correct", "gt_top5_positive"),
        ("gt_top5_positive", "gt_all_positive"),
    )
    for narrow, broad in implications:
        if (evaluable & work[narrow] & ~work[broad]).any():
            raise ValueError(
                f"invalid post-R7 flags: {narrow} must imply {broad}"
            )
    tech = work["technical_failure"]
    pa = work["pred_all_positive"]
    g1 = work["gt_native_correct"]
    g5 = work["gt_top5_positive"]
    ga = work["gt_all_positive"]
    conditions = (
        residual & tech,
        residual & ~tech & pa,
        residual & ~tech & ~pa & g1,
        residual & ~tech & ~pa & ~g1 & g5,
        residual & ~tech & ~pa & ~g5 & ga,
        residual & ~tech & ~pa & ~ga,
    )
    membership = np.column_stack([condition.to_numpy(bool) for condition in conditions])
    if not np.equal(membership[residual.to_numpy()].sum(axis=1), 1).all():
        bad = work.loc[
            residual & pd.Series(membership.sum(axis=1), index=work.index).ne(1),
            "sample_id",
        ].astype(str).head().tolist()
        raise RuntimeError(f"post-R7 residual taxonomy is not exhaustive: {bad}")
    labels = np.full(len(work), "", dtype=object)
    labels[residual.to_numpy()] = np.asarray(POST_R7_CLASSES, dtype=object)[
        membership[residual.to_numpy()].argmax(axis=1)
    ]
    work["is_residual_failure"] = residual
    work["post_r7_taxonomy"] = labels
    work["reranker_depth_flag"] = ""
    reranker = work["post_r7_taxonomy"].eq(POST_R7_CLASSES[1])
    work.loc[reranker & work["pred_top5_positive"], "reranker_depth_flag"] = (
        "positive_in_top5"
    )
    work.loc[reranker & ~work["pred_top5_positive"], "reranker_depth_flag"] = (
        "positive_below_top5"
    )
    return work


def add_secondary_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add preregistered non-primary flags without changing the T/R classes."""

    required = {
        "pred_all_positive",
        "gt_all_positive",
        "pred_no_output",
        "gt_no_output",
        "pred_candidate_count",
        "gt_candidate_count",
        "pred_first_positive_rank",
        "gt_first_positive_rank",
    }
    _require_columns(frame, required, name="secondary flags")
    work = frame.copy()
    pa = work["pred_all_positive"].astype(bool)
    ga = work["gt_all_positive"].astype(bool)
    work["pred_positive_gt_negative"] = pa & ~ga
    work["pred_negative_gt_positive"] = ~pa & ga
    work["GT_mask_regression"] = pa & ~ga
    work["candidate_count_increased"] = (
        work["gt_candidate_count"] > work["pred_candidate_count"]
    )
    work["candidate_count_decreased"] = (
        work["gt_candidate_count"] < work["pred_candidate_count"]
    )
    both_ranked = work["pred_first_positive_rank"].notna() & work[
        "gt_first_positive_rank"
    ].notna()
    work["first_positive_rank_improved"] = both_ranked & (
        work["gt_first_positive_rank"] < work["pred_first_positive_rank"]
    )
    work["first_positive_rank_worsened"] = both_ranked & (
        work["gt_first_positive_rank"] > work["pred_first_positive_rank"]
    )
    for column in (
        "predicted_mask_empty",
        "gt_mask_empty_or_invalid",
        "annotation_suspect",
    ):
        if column not in work:
            work[column] = False
    return work


def taxonomy_counts(frame: pd.DataFrame, *, column: str) -> dict[str, int]:
    """Return stable counts and reject unknown/non-exhaustive primary labels."""

    if column not in frame:
        raise ValueError(f"taxonomy frame misses {column}")
    allowed: tuple[str, ...]
    if column == "native_taxonomy":
        allowed = NATIVE_CLASSES
        expected = len(frame)
    elif column == "post_r7_taxonomy":
        allowed = POST_R7_CLASSES
        expected = int(frame.get("is_residual_failure", pd.Series(False, index=frame.index)).sum())
    else:
        raise ValueError(f"unsupported taxonomy column: {column}")
    values = frame[column].astype(str)
    unknown = sorted(set(values).difference({*allowed, ""}))
    if unknown:
        raise ValueError(f"unknown taxonomy labels: {unknown}")
    counts = {label: int(values.eq(label).sum()) for label in allowed}
    if sum(counts.values()) != expected:
        raise RuntimeError(
            f"taxonomy count mismatch: observed={sum(counts.values())} expected={expected}"
        )
    return counts


def taxonomy_definitions() -> Mapping[str, Any]:
    """Serializable definitions that the independent audit may bind as input."""

    return {
        "native_classes": list(NATIVE_CLASSES),
        "post_r7_classes": list(POST_R7_CLASSES),
        "native_priority": "T0>T1>T2>T3>T4>T5>T6>T7",
        "post_r7_priority": "R0>R1>R2>R3>R4>R5 on F=0 only",
    }
