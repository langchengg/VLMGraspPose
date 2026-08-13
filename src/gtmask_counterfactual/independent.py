"""Independent NumPy/Pandas recomputation from saved counterfactual frames.

This module intentionally imports no candidate generator, ranker, gate, report
builder, evaluator adapter, metrics implementation, or taxonomy producer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


IMAGE_SHAPE = (480, 640)
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0
GT_HEIGHT_PX = 20.0
GT_WIDTH_CLIP_PX = 100.0

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


def normalize_angle_deg(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    result = (value + 90.0) % 180.0 - 90.0
    return 0.0 if result == 0.0 else float(result)


def periodic_angle_error_deg(first: float, second: float) -> float:
    return abs(normalize_angle_deg(float(first) - float(second)))


def _value(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    raise ValueError(f"geometry misses every alias: {names}")


def canonical_corners(item: Any) -> np.ndarray:
    """Reproduce OpenCV boxPoints geometry using only NumPy."""

    cx = float(_value(item, "cx_px"))
    cy = float(_value(item, "cy_px"))
    theta = normalize_angle_deg(float(_value(item, "theta_deg")))
    width = float(_value(item, "jaw_width_px", "width_px"))
    height = float(_value(item, "rectangle_height_px", "height_px"))
    if not all(math.isfinite(value) for value in (cx, cy, width, height)):
        raise ValueError("canonical rectangle values must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("canonical rectangle dimensions must be positive")
    angle = math.radians(-theta)
    along = np.asarray(
        [math.cos(angle) * width / 2.0, math.sin(angle) * width / 2.0]
    )
    across = np.asarray(
        [-math.sin(angle) * height / 2.0, math.cos(angle) * height / 2.0]
    )
    centre = np.asarray([cx, cy])
    return np.asarray(
        [
            centre - along + across,
            centre - along - across,
            centre + along - across,
            centre + along + across,
        ],
        dtype=np.float64,
    )


def gt_corners_to_canonical(values: Any) -> dict[str, float]:
    corners = np.asarray(values, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("GT corners must be finite 4x2 x/y coordinates")
    centre = 0.5 * (corners[0] + corners[2])
    jaw = corners[3] - corners[0]
    width = min(float(np.linalg.norm(jaw)), GT_WIDTH_CLIP_PX)
    if width <= 0.0:
        raise ValueError("GT jaw width must be positive")
    raw = math.degrees(math.atan2(float(jaw[0]), float(jaw[1])))
    official = raw - 90.0 if raw > 0.0 else raw + 90.0
    return {
        "cx_px": float(centre[0]),
        "cy_px": float(centre[1]),
        "theta_deg": normalize_angle_deg(official),
        "jaw_width_px": width,
        "rectangle_height_px": GT_HEIGHT_PX,
    }


def _polygon_flat_pixels(
    vertices: np.ndarray, shape: tuple[int, int] = IMAGE_SHAPE
) -> np.ndarray:
    """Inclusive integer-grid fill equivalent to the frozen convex rasterizer."""

    height, width = map(int, shape)
    if height <= 0 or width <= 0:
        raise ValueError("image shape must be positive")
    polygon = np.asarray(vertices, dtype=np.intp)
    if polygon.shape != (4, 2):
        raise ValueError("rectangle polygon must be 4x2")
    x_min = max(0, int(polygon[:, 0].min()))
    x_max = min(width - 1, int(polygon[:, 0].max()))
    y_min = max(0, int(polygon[:, 1].min()))
    y_max = min(height - 1, int(polygon[:, 1].max()))
    if x_min > x_max or y_min > y_max:
        return np.empty(0, dtype=np.int64)
    rows, columns = np.mgrid[y_min : y_max + 1, x_min : x_max + 1]
    crosses = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        crosses.append(
            (end[0] - start[0]) * (rows - start[1])
            - (end[1] - start[1]) * (columns - start[0])
        )
    values = np.stack(crosses)
    inside = np.all(values >= 0, axis=0) | np.all(values <= 0, axis=0)
    return np.unique(rows[inside].astype(np.int64) * width + columns[inside])


def raster_iou(
    first: Any, second: Any, shape: tuple[int, int] = IMAGE_SHAPE
) -> float:
    a = _polygon_flat_pixels(canonical_corners(first), shape)
    b = _polygon_flat_pixels(canonical_corners(second), shape)
    if a.size == 0 and b.size == 0:
        return 0.0
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = int(a.size + b.size - intersection)
    return 0.0 if union <= 0 else float(intersection / union)


def evaluate_same_gt_candidate(
    candidate: Any,
    gt_grasp_rectangles: Sequence[Any],
    *,
    shape: tuple[int, int] = IMAGE_SHAPE,
) -> dict[str, Any]:
    """Apply strict IoU and inclusive angle to the same GT rectangle."""

    candidate_theta = normalize_angle_deg(float(_value(candidate, "theta_deg")))
    pairs: list[dict[str, Any]] = []
    for index, corners in enumerate(gt_grasp_rectangles):
        ground_truth = gt_corners_to_canonical(corners)
        iou = raster_iou(candidate, ground_truth, shape)
        angle = periodic_angle_error_deg(candidate_theta, ground_truth["theta_deg"])
        pairs.append(
            {
                "gt_index": index,
                "iou": iou,
                "angle_error_deg": angle,
                "iou_ok": iou > IOU_THRESHOLD,
                "angle_ok": angle <= ANGLE_THRESHOLD_DEG,
                "success": iou > IOU_THRESHOLD and angle <= ANGLE_THRESHOLD_DEG,
            }
        )
    successes = [item for item in pairs if item["success"]]
    ranked = sorted(
        successes or pairs,
        key=lambda item: (
            -float(item["iou"]),
            float(item["angle_error_deg"]),
            int(item["gt_index"]),
        ),
    )
    best = ranked[0] if ranked else None
    return {
        "candidate_success": bool(successes),
        "matched_gt_index": None if best is None else int(best["gt_index"]),
        "best_same_gt_iou": math.nan if best is None else float(best["iou"]),
        "best_same_gt_angle_error_deg": (
            math.nan if best is None else float(best["angle_error_deg"])
        ),
        "pairwise": pairs,
    }


def independent_evaluate_candidates(
    candidate_geometry: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    shape: tuple[int, int] = IMAGE_SHAPE,
) -> pd.DataFrame:
    """Re-label saved geometry; invalid rows remain explicit technical failures."""

    candidate_required = {
        "sample_id",
        "route",
        "branch",
        "candidate_id",
        "native_rank",
        "cx_px",
        "cy_px",
        "theta_deg",
    }
    missing = sorted(candidate_required.difference(candidate_geometry.columns))
    if missing:
        raise ValueError(f"candidate geometry misses columns: {missing}")
    if not ({"width_px", "jaw_width_px"} & set(candidate_geometry.columns)):
        raise ValueError("candidate geometry misses width")
    if not ({"height_px", "rectangle_height_px"} & set(candidate_geometry.columns)):
        raise ValueError("candidate geometry misses height")
    if candidate_geometry.duplicated(
        ["sample_id", "route", "branch", "candidate_id"]
    ).any():
        raise ValueError("candidate identities are not unique")
    if not {"sample_id", "gt_grasp_rectangles"}.issubset(ground_truth.columns):
        raise ValueError("ground truth misses sample_id/gt_grasp_rectangles")
    labels = ground_truth[["sample_id", "gt_grasp_rectangles"]].copy()
    labels["sample_id"] = labels["sample_id"].astype(str)
    if labels["sample_id"].duplicated().any() or labels["sample_id"].eq("").any():
        raise ValueError("ground-truth sample IDs must be unique and non-empty")
    work = candidate_geometry.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    joined = work.merge(labels, on="sample_id", how="left", validate="many_to_one")
    if joined["gt_grasp_rectangles"].isna().any():
        raise ValueError("candidate geometry references missing ground truth")
    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        try:
            result = evaluate_same_gt_candidate(
                row, row.gt_grasp_rectangles, shape=shape
            )
            result.pop("pairwise")
            rows.append(
                {"independent_evaluator_valid": True, "independent_error": "", **result}
            )
        except (TypeError, ValueError) as error:
            rows.append(
                {
                    "independent_evaluator_valid": False,
                    "independent_error": f"{type(error).__name__}: {error}",
                    "candidate_success": False,
                    "matched_gt_index": None,
                    "best_same_gt_iou": math.nan,
                    "best_same_gt_angle_error_deg": math.nan,
                }
            )
    return pd.concat(
        [work.reset_index(drop=True), pd.DataFrame.from_records(rows)], axis=1
    )


def _sample_outcomes(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    route: str,
    branch: str,
    ks: Sequence[int],
) -> pd.DataFrame:
    selected = candidates.loc[
        candidates["route"].astype(str).eq(route)
        & candidates["branch"].astype(str).eq(branch)
    ].copy()
    if len(selected):
        selected["native_rank"] = pd.to_numeric(
            selected["native_rank"], errors="raise"
        ).astype(int)
        if selected["native_rank"].le(0).any() or selected.duplicated(
            ["sample_id", "native_rank"]
        ).any():
            raise ValueError("independent native ranks are invalid")
        if not selected.groupby("sample_id")["native_rank"].min().eq(1).all():
            raise ValueError("independent non-empty pools must start at rank 1")
    counts = selected.groupby("sample_id").size().to_dict()
    positive_rows = selected.loc[selected["candidate_success"].astype(bool)]
    first = positive_rows.groupby("sample_id")["native_rank"].min().to_dict()
    positive_counts = (
        selected.groupby("sample_id")["candidate_success"].sum().astype(int).to_dict()
    )
    invalid = (
        selected.loc[~selected["independent_evaluator_valid"].astype(bool), "sample_id"]
        .astype(str)
        .unique()
    )
    invalid_ids = set(invalid)
    rows = []
    for sample in manifest.itertuples(index=False):
        sample_id = str(sample.sample_id)
        rank = first.get(sample_id)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "route": route,
            "branch": branch,
            "candidate_count": int(counts.get(sample_id, 0)),
            "no_output": sample_id not in counts,
            "native_correct": rank == 1,
            "first_positive_rank": pd.NA if rank is None else int(rank),
            "positive_candidate_count": int(positive_counts.get(sample_id, 0)),
            "oracle_all": rank is not None,
            "reciprocal_rank": 0.0 if rank is None else 1.0 / float(rank),
            "technical_failure": bool(
                getattr(sample, "technical_failure", False) or sample_id in invalid_ids
            ),
        }
        for k in ks:
            hit = rank is not None and int(rank) <= k
            row[f"j_at_{k}"] = bool(hit)
            row[f"oracle_at_{k}"] = bool(hit)
        rows.append(row)
    result = pd.DataFrame(rows)
    result["first_positive_rank"] = result["first_positive_rank"].astype("Int64")
    return result


def _distribution(values: pd.Series) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = "no_positive" if pd.isna(value) else str(int(value))
        result[key] = result.get(key, 0) + 1
    return result


def _metrics(outcomes: pd.DataFrame) -> dict[str, Any]:
    count = len(outcomes)
    candidate_counts = outcomes["candidate_count"].astype(int)
    positive_counts = outcomes["positive_candidate_count"].astype(int)
    result: dict[str, Any] = {
        "route": str(outcomes["route"].iloc[0]),
        "branch": str(outcomes["branch"].iloc[0]),
        "N": count,
        "no_output": int(outcomes["no_output"].sum()),
        "no_output_rate": float(outcomes["no_output"].mean()),
        "candidate_count_mean": float(candidate_counts.mean()),
        "candidate_count_median": float(candidate_counts.median()),
        "candidate_count_p95": float(candidate_counts.quantile(0.95)),
        "native_j_at_1_numerator": int(outcomes["native_correct"].sum()),
        "native_j_at_1": float(outcomes["native_correct"].mean()),
        "oracle_all_numerator": int(outcomes["oracle_all"].sum()),
        "oracle_all": float(outcomes["oracle_all"].mean()),
        "mrr": float(outcomes["reciprocal_rank"].mean()),
        "first_positive_rank_distribution": _distribution(
            outcomes["first_positive_rank"]
        ),
        "positive_candidates_per_sample_mean": float(positive_counts.mean()),
        "positive_candidates_per_sample_median": float(positive_counts.median()),
        "positive_candidates_per_sample_p95": float(positive_counts.quantile(0.95)),
        "positive_candidates_per_sample_distribution": {
            str(key): int(value)
            for key, value in positive_counts.value_counts().sort_index().items()
        },
    }
    for column in sorted(name for name in outcomes if name.startswith("oracle_at_")):
        suffix = column.removeprefix("oracle_at_")
        numerator = int(outcomes[column].sum())
        result[f"oracle_at_{suffix}_numerator"] = numerator
        result[f"oracle_at_{suffix}"] = numerator / count
        result[f"j_at_{suffix}_numerator"] = numerator
        result[f"j_at_{suffix}"] = numerator / count
    return result


def _native_taxonomy(paired: pd.DataFrame) -> pd.DataFrame:
    work = paired.copy()
    tech = work["technical_failure"].astype(bool)
    a1 = work["pred_native_correct"].astype(bool)
    p5 = work["pred_top5_positive"].astype(bool)
    pa = work["pred_all_positive"].astype(bool)
    g1 = work["gt_native_correct"].astype(bool)
    g5 = work["gt_top5_positive"].astype(bool)
    ga = work["gt_all_positive"].astype(bool)
    conditions = (
        tech,
        ~tech & a1,
        ~tech & ~a1 & p5,
        ~tech & ~a1 & ~p5 & pa,
        ~tech & ~pa & g1,
        ~tech & ~pa & ~g1 & g5,
        ~tech & ~pa & ~g5 & ga,
        ~tech & ~pa & ~ga,
    )
    membership = np.column_stack([item.to_numpy(bool) for item in conditions])
    if not np.equal(membership.sum(axis=1), 1).all():
        raise RuntimeError("independent native taxonomy is not exhaustive")
    work["native_taxonomy"] = np.asarray(NATIVE_CLASSES, dtype=object)[
        membership.argmax(axis=1)
    ]
    work["pred_deep_rank_flag"] = ""
    work["gt_deep_rank_flag"] = ""
    d1 = work["route"].str.upper().eq("D1")
    pred_deep = d1 & work["native_taxonomy"].eq(NATIVE_CLASSES[3])
    gt_deep = d1 & work["native_taxonomy"].eq(NATIVE_CLASSES[6])
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


def _post_r7_taxonomy(native: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    work = native.merge(
        final[["sample_id", "route", "final_correct"]],
        on=["sample_id", "route"],
        how="left",
        validate="one_to_one",
    )
    if work["final_correct"].isna().any():
        raise ValueError("final outcomes do not cover the independent denominator")
    residual = ~work["final_correct"].astype(bool)
    tech = work["technical_failure"].astype(bool)
    pa = work["pred_all_positive"].astype(bool)
    g1 = work["gt_native_correct"].astype(bool)
    g5 = work["gt_top5_positive"].astype(bool)
    ga = work["gt_all_positive"].astype(bool)
    conditions = (
        residual & tech,
        residual & ~tech & pa,
        residual & ~tech & ~pa & g1,
        residual & ~tech & ~pa & ~g1 & g5,
        residual & ~tech & ~pa & ~g5 & ga,
        residual & ~tech & ~pa & ~ga,
    )
    membership = np.column_stack([item.to_numpy(bool) for item in conditions])
    if not np.equal(membership[residual].sum(axis=1), 1).all():
        raise RuntimeError("independent post-R7 taxonomy is not exhaustive")
    labels = np.full(len(work), "", dtype=object)
    labels[residual] = np.asarray(POST_R7_CLASSES, dtype=object)[
        membership[residual].argmax(axis=1)
    ]
    work["is_residual_failure"] = residual
    work["post_r7_taxonomy"] = labels
    return work


def _paired_inputs(
    predicted: pd.DataFrame, gt: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    pred_source = predicted.copy()
    gt_source = gt.copy()
    if "oracle_at_10" not in pred_source:
        pred_source["oracle_at_10"] = False
    if "oracle_at_10" not in gt_source:
        gt_source["oracle_at_10"] = False
    pred = pred_source.rename(
        columns={
            "native_correct": "pred_native_correct",
            "oracle_at_5": "pred_top5_positive",
            "oracle_at_10": "pred_top10_positive",
            "oracle_all": "pred_all_positive",
            "technical_failure": "pred_technical_failure",
        }
    )
    counterfactual = gt_source.rename(
        columns={
            "native_correct": "gt_native_correct",
            "oracle_at_5": "gt_top5_positive",
            "oracle_at_10": "gt_top10_positive",
            "oracle_all": "gt_all_positive",
            "technical_failure": "gt_technical_failure",
        }
    )
    columns = [
        "sample_id",
        "route",
        "pred_native_correct",
        "pred_top5_positive",
        "pred_top10_positive",
        "pred_all_positive",
        "pred_technical_failure",
    ]
    gt_columns = [
        "sample_id",
        "route",
        "gt_native_correct",
        "gt_top5_positive",
        "gt_top10_positive",
        "gt_all_positive",
        "gt_technical_failure",
    ]
    paired = pred[columns].merge(
        counterfactual[gt_columns],
        on=["sample_id", "route"],
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(manifest):
        raise RuntimeError("independent branch pair lost denominator samples")
    paired["technical_failure"] = paired[
        ["pred_technical_failure", "gt_technical_failure"]
    ].any(axis=1)
    return paired.drop(
        columns=["pred_technical_failure", "gt_technical_failure"]
    )


def _statistical_input(
    paired: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    metadata = manifest[["sample_id", "scene_id", "frame_id"]].copy()
    result = paired.merge(metadata, on="sample_id", validate="many_to_one")
    result["pred_native_j_at_1"] = result["pred_native_correct"]
    result["gt_native_j_at_1"] = result["gt_native_correct"]
    result["pred_oracle_at_5"] = result["pred_top5_positive"]
    result["gt_oracle_at_5"] = result["gt_top5_positive"]
    result["pred_oracle_at_10"] = result["pred_top10_positive"]
    result["gt_oracle_at_10"] = result["gt_top10_positive"]
    result["pred_oracle_all"] = result["pred_all_positive"]
    result["gt_oracle_all"] = result["gt_all_positive"]
    return result[
        [
            "sample_id",
            "route",
            "scene_id",
            "frame_id",
            "pred_native_j_at_1",
            "gt_native_j_at_1",
            "pred_oracle_at_5",
            "gt_oracle_at_5",
            "pred_oracle_at_10",
            "gt_oracle_at_10",
            "pred_oracle_all",
            "gt_oracle_all",
        ]
    ].sort_values(["route", "sample_id"], kind="mergesort").reset_index(drop=True)


def _paired_transition_counts(paired: pd.DataFrame) -> dict[str, Any]:
    pred_all = paired["pred_all_positive"].astype(bool)
    gt_all = paired["gt_all_positive"].astype(bool)
    pred_negative = ~pred_all
    recovered = pred_negative & gt_all
    transitions: dict[str, Any] = {
        "N": len(paired),
        "pred_no_positive_to_gt_positive": int(recovered.sum()),
        "pred_positive_to_gt_no_positive": int((pred_all & ~gt_all).sum()),
        "both_positive": int((pred_all & gt_all).sum()),
        "neither_positive": int((~pred_all & ~gt_all).sum()),
        "grounding_candidate_recovery_denominator": int(pred_negative.sum()),
        "grounding_candidate_recovery_rate": (
            None
            if not pred_negative.any()
            else float(recovered.sum() / pred_negative.sum())
        ),
        "oracle_grounding_ceiling": float(gt_all.mean() - pred_all.mean()),
        "residual_generator_failure_count": int((~gt_all).sum()),
        "residual_generator_failure_rate": float((~gt_all).mean()),
    }
    for metric in (
        "native_correct",
        "top5_positive",
        "top10_positive",
        "all_positive",
    ):
        reference = paired[f"pred_{metric}"].astype(bool)
        counterfactual = paired[f"gt_{metric}"].astype(bool)
        transitions[f"{metric}_b_reference_only"] = int(
            (reference & ~counterfactual).sum()
        )
        transitions[f"{metric}_c_counterfactual_only"] = int(
            (~reference & counterfactual).sum()
        )
        transitions[f"delta_{metric}"] = float(
            counterfactual.mean() - reference.mean()
        )
    return transitions


def assert_frame_exact(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    columns: Sequence[str] | None = None,
) -> None:
    """Exact row/key/value comparison with explicit NaN equality."""

    key_columns = list(keys)
    compare = list(columns) if columns is not None else sorted(set(observed) & set(expected))
    required = set(key_columns) | set(compare)
    for name, frame in (("observed", observed), ("expected", expected)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise AssertionError(f"{name} exact frame misses columns: {missing}")
        if frame.duplicated(key_columns).any():
            raise AssertionError(f"{name} exact frame contains duplicate keys")
    left = observed[key_columns + [c for c in compare if c not in key_columns]].sort_values(
        key_columns, kind="mergesort"
    ).reset_index(drop=True)
    right = expected[key_columns + [c for c in compare if c not in key_columns]].sort_values(
        key_columns, kind="mergesort"
    ).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        raise AssertionError(
            f"exact frame shape/schema differs: {left.shape} vs {right.shape}"
        )
    for column in left:
        a, b = left[column], right[column]
        equal = a.eq(b) | (a.isna() & b.isna())
        if not bool(equal.all()):
            index = int(np.flatnonzero(~equal.to_numpy())[0])
            raise AssertionError(
                f"exact frame differs at {column}[{index}]: {a.iloc[index]!r} != {b.iloc[index]!r}"
            )


def _assert_nested_exact(observed: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        if set(observed) != set(expected):
            raise AssertionError(f"{path} mapping keys differ")
        for key in observed:
            _assert_nested_exact(observed[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(observed, float) or isinstance(expected, float):
        if math.isnan(float(observed)) and math.isnan(float(expected)):
            return
        if float(observed) != float(expected):
            raise AssertionError(f"{path} differs: {observed!r} != {expected!r}")
        return
    if observed != expected:
        raise AssertionError(f"{path} differs: {observed!r} != {expected!r}")


def independent_recompute_from_frames(
    sample_manifest: pd.DataFrame,
    candidate_geometry: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    predicted_branch: str = "predicted",
    gt_branch: str = "gt_oracle",
    k_by_route: Mapping[str, Sequence[int]] | None = None,
    final_outcomes: pd.DataFrame | None = None,
    taxonomy_definitions: Mapping[str, Any] | None = None,
    saved_candidate_labels: pd.DataFrame | None = None,
    saved_sample_outcomes: pd.DataFrame | None = None,
    saved_branch_metrics: Mapping[str, Any] | None = None,
    saved_native_taxonomy: pd.DataFrame | None = None,
    saved_post_r7_taxonomy: pd.DataFrame | None = None,
    saved_statistical_inputs: pd.DataFrame | None = None,
    saved_paired_transitions: Mapping[str, Any] | None = None,
    shape: tuple[int, int] = IMAGE_SHAPE,
) -> dict[str, Any]:
    """Recompute saved labels, metrics, taxonomies and paired statistic inputs.

    Every supplied saved artifact is checked exactly.  The function is purely
    in-memory and performs no Test access, file write, generator/ranker/gate
    invocation, or report construction.
    """

    required_manifest = {"sample_id", "scene_id", "frame_id"}
    missing = sorted(required_manifest.difference(sample_manifest.columns))
    if missing:
        raise ValueError(f"sample manifest misses columns: {missing}")
    manifest = sample_manifest.copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    if manifest.empty or manifest["sample_id"].duplicated().any():
        raise ValueError("sample manifest must be non-empty and unique")
    if taxonomy_definitions is not None:
        if tuple(taxonomy_definitions.get("native_classes", ())) != NATIVE_CLASSES:
            raise ValueError("independent native taxonomy definitions differ")
        if tuple(taxonomy_definitions.get("post_r7_classes", ())) != POST_R7_CLASSES:
            raise ValueError("independent post-R7 taxonomy definitions differ")

    evaluated = independent_evaluate_candidates(
        candidate_geometry, ground_truth, shape=shape
    )
    if saved_candidate_labels is not None:
        assert_frame_exact(
            evaluated,
            saved_candidate_labels,
            keys=["sample_id", "route", "branch", "candidate_id"],
            columns=["candidate_success"],
        )

    routes = sorted(evaluated["route"].astype(str).unique())
    branches = set(evaluated["branch"].astype(str))
    if predicted_branch not in branches or gt_branch not in branches:
        raise ValueError("independent geometry misses predicted or GT branch")
    outcomes: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, Any]] = {}
    paired_transitions: dict[str, dict[str, Any]] = {}
    native_rows: list[pd.DataFrame] = []
    stat_rows: list[pd.DataFrame] = []
    for route in routes:
        ks = tuple((k_by_route or {}).get(route, (5, 10)))
        pred = _sample_outcomes(
            evaluated,
            manifest,
            route=route,
            branch=predicted_branch,
            ks=ks,
        )
        gt = _sample_outcomes(
            evaluated,
            manifest,
            route=route,
            branch=gt_branch,
            ks=ks,
        )
        outcomes.extend([pred, gt])
        metrics[f"{route}|{predicted_branch}"] = _metrics(pred)
        metrics[f"{route}|{gt_branch}"] = _metrics(gt)
        paired = _paired_inputs(pred, gt, manifest)
        paired_transitions[route] = _paired_transition_counts(paired)
        native_rows.append(_native_taxonomy(paired))
        stat_rows.append(_statistical_input(paired, manifest))
    sample_outcomes = pd.concat(outcomes, ignore_index=True)
    native_taxonomy = pd.concat(native_rows, ignore_index=True)
    statistical_inputs = pd.concat(stat_rows, ignore_index=True)

    if saved_sample_outcomes is not None:
        columns = [
            "candidate_count",
            "no_output",
            "native_correct",
            "first_positive_rank",
            "positive_candidate_count",
            "oracle_all",
            "reciprocal_rank",
            *sorted(
                set(name for name in sample_outcomes if name.startswith("oracle_at_"))
                & set(saved_sample_outcomes.columns)
            ),
        ]
        assert_frame_exact(
            sample_outcomes,
            saved_sample_outcomes,
            keys=["sample_id", "route", "branch"],
            columns=columns,
        )
    if saved_branch_metrics is not None:
        _assert_nested_exact(metrics, saved_branch_metrics, path="branch_metrics")
    if saved_native_taxonomy is not None:
        assert_frame_exact(
            native_taxonomy,
            saved_native_taxonomy,
            keys=["sample_id", "route"],
            columns=["native_taxonomy", "pred_deep_rank_flag", "gt_deep_rank_flag"],
        )
    if saved_statistical_inputs is not None:
        assert_frame_exact(
            statistical_inputs,
            saved_statistical_inputs,
            keys=["sample_id", "route"],
        )
    if saved_paired_transitions is not None:
        _assert_nested_exact(
            paired_transitions,
            saved_paired_transitions,
            path="paired_transitions",
        )

    post_r7 = None
    if final_outcomes is not None:
        required_final = {"sample_id", "route", "final_correct"}
        missing_final = sorted(required_final.difference(final_outcomes.columns))
        if missing_final or final_outcomes.duplicated(["sample_id", "route"]).any():
            raise ValueError(
                f"final outcomes are invalid; missing={missing_final}"
            )
        post_r7 = _post_r7_taxonomy(native_taxonomy, final_outcomes)
        if saved_post_r7_taxonomy is not None:
            assert_frame_exact(
                post_r7,
                saved_post_r7_taxonomy,
                keys=["sample_id", "route"],
                columns=["is_residual_failure", "post_r7_taxonomy"],
            )
    elif saved_post_r7_taxonomy is not None:
        raise ValueError("saved post-R7 taxonomy requires final_outcomes")

    return {
        "status": "PASS",
        "candidate_labels": evaluated,
        "sample_outcomes": sample_outcomes,
        "branch_metrics": metrics,
        "paired_transitions": paired_transitions,
        "native_taxonomy": native_taxonomy,
        "post_r7_taxonomy": post_r7,
        "statistical_inputs": statistical_inputs,
        "exact_checks": {
            "candidate_labels": saved_candidate_labels is not None,
            "sample_outcomes": saved_sample_outcomes is not None,
            "branch_metrics": saved_branch_metrics is not None,
            "native_taxonomy": saved_native_taxonomy is not None,
            "post_r7_taxonomy": saved_post_r7_taxonomy is not None,
            "statistical_inputs": saved_statistical_inputs is not None,
            "paired_transitions": saved_paired_transitions is not None,
        },
    }


recompute_saved_frames = independent_recompute_from_frames
