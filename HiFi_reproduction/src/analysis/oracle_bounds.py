"""Target-consistency ceilings over frozen VGN candidate pools.

The same-pool analyses in this module only select rows that already exist in
the predicted-mask run.  They never invoke VGN, rebuild a TSDF, or alter an
official VGN quality.  The GT-regenerated pool is reported separately because
its target mask changed the task frame and therefore candidate generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.experiments.metrics import wilson_interval


PRIMARY_LABEL_COLUMN = "gt_target_positive_primary"
LABEL_TOLERANCES: Mapping[str, str] = {
    "raw_mask": "gt_inside_raw_mask",
    "3_px": "gt_inside_dilated_mask_3px",
    "5_px": "gt_inside_dilated_mask_5px",
    "10_px": "gt_inside_dilated_mask_10px",
    "3d_near_10mm": "gt_3d_near_10mm",
    "3d_near_20mm": "gt_3d_near_20mm",
    "3d_near_30mm": "gt_3d_near_30mm",
}

ORACLE_COLUMNS: Mapping[str, str] = {
    "current_baseline": "baseline_target_consistent",
    "same_pool_post_filter_oracle": "oracle_same_pool_post_filter",
    "same_pool_pre_filter_oracle": "oracle_same_pool_pre_filter",
    "gt_regenerated_pool_oracle": "oracle_gt_regenerated_pool",
    "union_diagnostic_ceiling": "oracle_union_pool",
}

DENOMINATOR_SCOPES: Mapping[str, str] = {
    "all_samples": "eligible_all_samples",
    "samples_with_official_candidates": "eligible_with_official_candidates",
    "samples_with_baseline_selection": "eligible_with_baseline_selection",
}


class OracleBoundsError(ValueError):
    """Raised when candidate tables cannot support a comparable analysis."""


@dataclass(frozen=True)
class OracleAnalysisTables:
    """All sample- and aggregate-level outputs for oracle analysis."""

    samples: pd.DataFrame
    oracle_upper_bounds: pd.DataFrame
    oracle_sensitivity: pd.DataFrame
    union_candidates: pd.DataFrame


def _position(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            row["position_camera_x"],
            row["position_camera_y"],
            row["position_camera_z"],
        ],
        dtype=np.float64,
    )


def _quaternion(row: Mapping[str, Any]) -> np.ndarray:
    quaternion = np.asarray(row["quaternion_camera_xyzw"], dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise OracleBoundsError("camera-frame quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise OracleBoundsError("camera-frame quaternion has zero norm")
    return quaternion / norm


def pose_translation_distance_m(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    """Euclidean distance between two camera-frame grasp origins."""

    return float(np.linalg.norm(_position(left) - _position(right)))


def pose_rotation_geodesic_deg(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    """SO(3) geodesic distance, invariant to quaternion sign."""

    dot = float(abs(np.dot(_quaternion(left), _quaternion(right))))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def poses_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    translation_threshold_m: float = 0.01,
    rotation_threshold_deg: float = 15.0,
    width_threshold_m: float = 0.01,
) -> bool:
    """Return whether two candidates satisfy the preregistered union thresholds."""

    if min(translation_threshold_m, rotation_threshold_deg, width_threshold_m) < 0:
        raise ValueError("pose-matching thresholds must be non-negative")
    return bool(
        pose_translation_distance_m(left, right) <= translation_threshold_m
        and pose_rotation_geodesic_deg(left, right) <= rotation_threshold_deg
        and abs(float(left["width_m"]) - float(right["width_m"]))
        <= width_threshold_m
    )


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    # Prefer the frozen predicted-mask row for exact ties.  This makes union
    # construction reproducible without changing the quality ordering.
    source_order = 0 if str(row.get("pool_source")) == "predicted_mask" else 1
    return (
        -float(row["vgn_quality"]),
        source_order,
        int(row["candidate_index_original"]),
    )


def deduplicate_union_pool(
    predicted_candidates: pd.DataFrame,
    gt_regenerated_candidates: pd.DataFrame,
    *,
    translation_threshold_m: float = 0.01,
    rotation_threshold_deg: float = 15.0,
    width_threshold_m: float = 0.01,
) -> pd.DataFrame:
    """Greedy analysis-only 6-DoF NMS in the shared camera frame.

    Rows are processed by descending unmodified VGN quality.  A lower-ranked
    row is suppressed when it matches any retained row from the same sample.
    The returned representative keeps its own GT label; suppression does not
    transfer labels between nearby but non-identical poses.
    """

    frames = []
    for frame, expected_source in (
        (predicted_candidates, "predicted_mask"),
        (gt_regenerated_candidates, "gt_regenerated"),
    ):
        copy = frame.copy()
        if "pool_source" not in copy:
            copy["pool_source"] = expected_source
        frames.append(copy)
    if not frames or all(frame.empty for frame in frames):
        columns = list(
            dict.fromkeys(
                list(predicted_candidates.columns)
                + list(gt_regenerated_candidates.columns)
                + ["union_candidate_index", "union_member_count", "union_member_sources"]
            )
        )
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "sample_id",
        "pool_source",
        "candidate_index_original",
        "vgn_quality",
        "width_m",
        "position_camera_x",
        "position_camera_y",
        "position_camera_z",
        "quaternion_camera_xyzw",
    }
    missing = sorted(required - set(combined.columns))
    if missing:
        raise OracleBoundsError(f"union candidates lack required columns: {missing}")

    retained_rows: list[dict[str, Any]] = []
    for _, group in combined.groupby("sample_id", sort=True):
        records = sorted(group.to_dict(orient="records"), key=_candidate_sort_key)
        retained: list[dict[str, Any]] = []
        memberships: list[list[dict[str, Any]]] = []
        for record in records:
            match_index = next(
                (
                    index
                    for index, representative in enumerate(retained)
                    if poses_match(
                        record,
                        representative,
                        translation_threshold_m=translation_threshold_m,
                        rotation_threshold_deg=rotation_threshold_deg,
                        width_threshold_m=width_threshold_m,
                    )
                ),
                None,
            )
            identity = {
                "pool_source": str(record["pool_source"]),
                "candidate_index_original": int(record["candidate_index_original"]),
            }
            if match_index is None:
                retained.append(record)
                memberships.append([identity])
            else:
                memberships[match_index].append(identity)
        for union_index, (record, members) in enumerate(
            zip(retained, memberships, strict=True)
        ):
            record["union_candidate_index"] = int(union_index)
            record["union_member_count"] = len(members)
            record["union_member_sources"] = sorted(
                {str(member["pool_source"]) for member in members}
            )
            record["union_members"] = members
            retained_rows.append(record)
    return pd.DataFrame(retained_rows).sort_values(
        ["sample_id", "union_candidate_index"]
    ).reset_index(drop=True)


def _validate_inputs(
    samples: pd.DataFrame,
    predicted_candidates: pd.DataFrame,
    gt_regenerated_candidates: pd.DataFrame,
    label_columns: Iterable[str],
) -> None:
    if "sample_id" not in samples or "scene_id" not in samples:
        raise OracleBoundsError("sample table requires sample_id and scene_id")
    if samples["sample_id"].isna().any() or samples["sample_id"].duplicated().any():
        raise OracleBoundsError("sample_id must be non-empty and unique")
    sample_ids = set(samples["sample_id"].astype(str))
    for name, candidates in (
        ("predicted", predicted_candidates),
        ("gt_regenerated", gt_regenerated_candidates),
    ):
        required = {
            "sample_id",
            "candidate_index_original",
            "vgn_quality",
            "pred_filter_pass",
            "is_baseline_top1",
        }
        missing = sorted(required - set(candidates.columns))
        if missing:
            raise OracleBoundsError(f"{name} candidates lack required columns: {missing}")
        unknown = set(candidates["sample_id"].astype(str)) - sample_ids
        if unknown:
            raise OracleBoundsError(f"{name} candidates refer to unknown samples")
        duplicate_key = candidates.duplicated(
            ["sample_id", "candidate_index_original"], keep=False
        )
        if duplicate_key.any():
            raise OracleBoundsError(f"{name} pool has duplicate per-sample candidate IDs")
        missing_labels = sorted(set(label_columns) - set(candidates.columns))
        if missing_labels:
            raise OracleBoundsError(f"{name} candidates lack GT labels: {missing_labels}")


def _best_positive(
    candidates: pd.DataFrame, label_column: str, *, filter_column: str | None = None
) -> tuple[bool, int | None, float | None]:
    selected = candidates.loc[candidates[label_column].fillna(False).astype(bool)]
    if filter_column is not None:
        selected = selected.loc[selected[filter_column].fillna(False).astype(bool)]
    if selected.empty:
        return False, None, None
    selected = selected.sort_values(
        ["vgn_quality", "candidate_index_original"],
        ascending=[False, True],
        kind="stable",
    )
    row = selected.iloc[0]
    return True, int(row["candidate_index_original"]), float(row["vgn_quality"])


def compute_sample_oracles(
    samples: pd.DataFrame,
    predicted_candidates: pd.DataFrame,
    gt_regenerated_candidates: pd.DataFrame,
    *,
    label_column: str = PRIMARY_LABEL_COLUMN,
    union_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute candidate-selection ceilings once per manifest sample."""

    _validate_inputs(
        samples,
        predicted_candidates,
        gt_regenerated_candidates,
        [label_column],
    )
    pred_groups = {
        str(sample_id): group
        for sample_id, group in predicted_candidates.groupby("sample_id", sort=False)
    }
    regenerated_groups = {
        str(sample_id): group
        for sample_id, group in gt_regenerated_candidates.groupby("sample_id", sort=False)
    }
    union_groups = (
        {
            str(sample_id): group
            for sample_id, group in union_candidates.groupby("sample_id", sort=False)
        }
        if union_candidates is not None and not union_candidates.empty
        else {}
    )
    empty_pred = predicted_candidates.iloc[0:0]
    empty_regenerated = gt_regenerated_candidates.iloc[0:0]
    empty_union = (
        union_candidates.iloc[0:0]
        if union_candidates is not None
        else predicted_candidates.iloc[0:0]
    )

    rows: list[dict[str, Any]] = []
    for sample in samples.to_dict(orient="records"):
        sample_id = str(sample["sample_id"])
        pred = pred_groups.get(sample_id, empty_pred)
        regenerated = regenerated_groups.get(sample_id, empty_regenerated)
        union = union_groups.get(sample_id, empty_union)

        pre, pre_index, pre_quality = _best_positive(pred, label_column)
        post, post_index, post_quality = _best_positive(
            pred, label_column, filter_column="pred_filter_pass"
        )
        regenerated_value, regenerated_index, regenerated_quality = _best_positive(
            regenerated, label_column
        )
        if union_candidates is None:
            union_value, union_index, union_quality = None, None, None
        else:
            union_value, union_index, union_quality = _best_positive(union, label_column)

        baseline_rows = pred.loc[pred["is_baseline_top1"].fillna(False).astype(bool)]
        if len(baseline_rows) > 1:
            raise OracleBoundsError(f"{sample_id} has multiple baseline top-1 rows")
        baseline_selected = not baseline_rows.empty
        baseline_value = bool(
            baseline_selected and baseline_rows.iloc[0][label_column]
        )
        if baseline_value and not post:
            raise OracleBoundsError(
                f"{sample_id} baseline target consistency exceeds post-filter oracle"
            )
        row = dict(sample)
        row.update(
            n_official_candidates=int(len(pred)),
            n_gt_regenerated_official_candidates=int(len(regenerated)),
            has_baseline_selection=baseline_selected,
            baseline_target_consistent=baseline_value,
            oracle_same_pool_post_filter=post,
            oracle_same_pool_pre_filter=pre,
            oracle_gt_regenerated_pool=regenerated_value,
            oracle_union_pool=union_value,
            same_pool_post_filter_candidate_index=post_index,
            same_pool_post_filter_vgn_quality=post_quality,
            same_pool_pre_filter_candidate_index=pre_index,
            same_pool_pre_filter_vgn_quality=pre_quality,
            gt_regenerated_candidate_index=regenerated_index,
            gt_regenerated_vgn_quality=regenerated_quality,
            union_candidate_index=union_index,
            union_vgn_quality=union_quality,
            eligible_all_samples=True,
            eligible_with_official_candidates=bool(len(pred)),
            eligible_with_baseline_selection=baseline_selected,
            gt_label_column=label_column,
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if not (
        result["baseline_target_consistent"]
        <= result["oracle_same_pool_post_filter"]
    ).all():
        raise OracleBoundsError("baseline exceeds same-pool post-filter oracle")
    if not (
        result["oracle_same_pool_post_filter"]
        <= result["oracle_same_pool_pre_filter"]
    ).all():
        raise OracleBoundsError("post-filter oracle exceeds pre-filter oracle")
    return result


class _SceneBootstrap:
    """Reusable deterministic scene-resampling weights for many proportions."""

    def __init__(
        self,
        samples: pd.DataFrame,
        *,
        replicates: int,
        seed: int,
        confidence: float = 0.95,
    ) -> None:
        if replicates < 1:
            raise ValueError("bootstrap replicates must be positive")
        self.scene_ids = sorted(samples["scene_id"].astype(str).unique())
        if not self.scene_ids:
            raise ValueError("scene-cluster bootstrap requires at least one scene")
        self.replicates = int(replicates)
        self.seed = int(seed)
        self.confidence = float(confidence)
        generator = np.random.default_rng(self.seed)
        probabilities = np.full(len(self.scene_ids), 1.0 / len(self.scene_ids))
        self.weights = generator.multinomial(
            len(self.scene_ids), probabilities, size=self.replicates
        )

    def interval(
        self,
        samples: pd.DataFrame,
        value_column: str,
        eligibility_column: str,
        *,
        baseline_column: str | None = None,
    ) -> dict[str, Any]:
        eligible = samples[eligibility_column].fillna(False).astype(bool)
        values = samples[value_column].fillna(False).astype(float)
        if baseline_column is not None:
            values = values - samples[baseline_column].fillna(False).astype(float)
        scratch = pd.DataFrame(
            {
                "scene_id": samples["scene_id"].astype(str),
                "numerator": np.where(eligible, values, 0.0),
                "denominator": eligible.astype(float),
            }
        )
        grouped = scratch.groupby("scene_id")[["numerator", "denominator"]].sum()
        grouped = grouped.reindex(self.scene_ids, fill_value=0.0)
        numerator = self.weights @ grouped["numerator"].to_numpy(dtype=float)
        denominator = self.weights @ grouped["denominator"].to_numpy(dtype=float)
        finite = denominator > 0
        estimates = numerator[finite] / denominator[finite]
        if not len(estimates):
            return {
                "estimate": None,
                "ci_lower": None,
                "ci_upper": None,
                "confidence": self.confidence,
                "method": "paired_scene_cluster_percentile_bootstrap"
                if baseline_column is not None
                else "scene_cluster_percentile_bootstrap",
                "replicates": self.replicates,
                "seed": self.seed,
                "cluster_count": len(self.scene_ids),
            }
        alpha = 1.0 - self.confidence
        source = values[eligible]
        return {
            "estimate": float(source.mean()) if len(source) else None,
            "ci_lower": float(np.quantile(estimates, alpha / 2.0)),
            "ci_upper": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
            "confidence": self.confidence,
            "method": "paired_scene_cluster_percentile_bootstrap"
            if baseline_column is not None
            else "scene_cluster_percentile_bootstrap",
            "replicates": self.replicates,
            "seed": self.seed,
            "cluster_count": len(self.scene_ids),
        }


def summarize_oracle_bounds(
    sample_oracles: pd.DataFrame,
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create denominator-explicit rate, Wilson, and paired-bootstrap rows."""

    bootstrap = _SceneBootstrap(
        sample_oracles, replicates=bootstrap_replicates, seed=seed
    )
    rows: list[dict[str, Any]] = []
    for denominator_scope, eligibility_column in DENOMINATOR_SCOPES.items():
        eligible = sample_oracles[eligibility_column].fillna(False).astype(bool)
        denominator = int(eligible.sum())
        baseline_count = int(
            sample_oracles.loc[eligible, "baseline_target_consistent"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        baseline_rate = baseline_count / denominator if denominator else None
        for metric, value_column in ORACLE_COLUMNS.items():
            if sample_oracles[value_column].isna().all():
                continue
            numerator = int(
                sample_oracles.loc[eligible, value_column]
                .fillna(False)
                .astype(bool)
                .sum()
            )
            rate = numerator / denominator if denominator else None
            wilson = wilson_interval(numerator, denominator)
            interval = bootstrap.interval(
                sample_oracles, value_column, eligibility_column
            )
            paired = bootstrap.interval(
                sample_oracles,
                value_column,
                eligibility_column,
                baseline_column="baseline_target_consistent",
            )
            gain = None if rate is None or baseline_rate is None else rate - baseline_rate
            relative_error_reduction = (
                gain / (1.0 - baseline_rate)
                if gain is not None and baseline_rate is not None and baseline_rate < 1.0
                else None
            )
            rows.append(
                {
                    "metric": metric,
                    "denominator_scope": denominator_scope,
                    "numerator": numerator,
                    "denominator": denominator,
                    "rate": rate,
                    "percentage": None if rate is None else 100.0 * rate,
                    "wilson_ci_lower": wilson["ci_lower"],
                    "wilson_ci_upper": wilson["ci_upper"],
                    "scene_bootstrap_ci_lower": interval["ci_lower"],
                    "scene_bootstrap_ci_upper": interval["ci_upper"],
                    "baseline_numerator": baseline_count,
                    "absolute_gain": gain,
                    "absolute_gain_percentage_points": (
                        None if gain is None else 100.0 * gain
                    ),
                    "relative_error_reduction": relative_error_reduction,
                    "paired_scene_delta_ci_lower": paired["ci_lower"],
                    "paired_scene_delta_ci_upper": paired["ci_upper"],
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_seed": seed,
                    "score_source": "official_vgn_processed_quality",
                    "custom_reranking": False,
                }
            )
    return pd.DataFrame(rows)


def build_oracle_analysis(
    samples: pd.DataFrame,
    predicted_candidates: pd.DataFrame,
    gt_regenerated_candidates: pd.DataFrame,
    *,
    include_union: bool = True,
    bootstrap_replicates: int = 10_000,
    seed: int = 42,
    union_translation_threshold_m: float = 0.01,
    union_rotation_threshold_deg: float = 15.0,
    union_width_threshold_m: float = 0.01,
) -> OracleAnalysisTables:
    """Run primary and tolerance-sensitive oracle analyses."""

    label_columns = list(LABEL_TOLERANCES.values())
    _validate_inputs(
        samples, predicted_candidates, gt_regenerated_candidates, label_columns
    )
    union = (
        deduplicate_union_pool(
            predicted_candidates,
            gt_regenerated_candidates,
            translation_threshold_m=union_translation_threshold_m,
            rotation_threshold_deg=union_rotation_threshold_deg,
            width_threshold_m=union_width_threshold_m,
        )
        if include_union
        else pd.DataFrame()
    )
    primary = compute_sample_oracles(
        samples,
        predicted_candidates,
        gt_regenerated_candidates,
        label_column=PRIMARY_LABEL_COLUMN,
        union_candidates=union if include_union else None,
    )
    upper_bounds = summarize_oracle_bounds(
        primary, bootstrap_replicates=bootstrap_replicates, seed=seed
    )

    sensitivity_frames: list[pd.DataFrame] = []
    for tolerance, label_column in LABEL_TOLERANCES.items():
        outcomes = compute_sample_oracles(
            samples,
            predicted_candidates,
            gt_regenerated_candidates,
            label_column=label_column,
            union_candidates=union if include_union else None,
        )
        summary = summarize_oracle_bounds(
            outcomes, bootstrap_replicates=bootstrap_replicates, seed=seed
        )
        summary = summary.loc[summary["denominator_scope"] == "all_samples"].copy()
        summary.insert(0, "label_tolerance", tolerance)
        summary.insert(1, "label_column", label_column)
        sensitivity_frames.append(summary)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    return OracleAnalysisTables(primary, upper_bounds, sensitivity, union)


__all__ = [
    "DENOMINATOR_SCOPES",
    "LABEL_TOLERANCES",
    "ORACLE_COLUMNS",
    "OracleAnalysisTables",
    "OracleBoundsError",
    "build_oracle_analysis",
    "compute_sample_oracles",
    "deduplicate_union_pool",
    "pose_rotation_geodesic_deg",
    "pose_translation_distance_m",
    "poses_match",
    "summarize_oracle_bounds",
]
