"""Candidate multiplicity and analysis-only 6-DoF pose-mode clustering.

The functions in this module never alter a candidate pool or VGN quality.
Pose clustering is a deterministic, transitive single-link diagnostic used to
distinguish raw candidate count from the number of meaningfully different pose
modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


PRIMARY_TRANSLATION_THRESHOLD_M = 0.010
PRIMARY_ROTATION_THRESHOLD_DEG = 15.0
PRIMARY_WIDTH_THRESHOLD_M = 0.010
TRANSLATION_SENSITIVITY_M = (0.005, 0.010, 0.020)
ROTATION_SENSITIVITY_DEG = (10.0, 15.0, 30.0)
COUNT_BIN_LABELS = ("0", "1", "2", "3", "4", "5–9", "10–19", "20–49", "50+")


class MultiplicityError(ValueError):
    """Raised when candidate tables cannot support multiplicity analysis."""


@dataclass(frozen=True)
class PoseModeThresholds:
    translation_m: float = PRIMARY_TRANSLATION_THRESHOLD_M
    rotation_deg: float = PRIMARY_ROTATION_THRESHOLD_DEG
    width_m: float = PRIMARY_WIDTH_THRESHOLD_M

    def __post_init__(self) -> None:
        values = (self.translation_m, self.rotation_deg, self.width_m)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise MultiplicityError("pose-mode thresholds must be finite and non-negative")


class _UnionFind:
    """Union-find whose representative is always the smallest input ordinal."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        root, child = sorted((left_root, right_root))
        self.parent[child] = root


def rotation_geodesic_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Return the shortest SO(3) geodesic angle between two rotations."""

    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != (3, 3) or second.shape != (3, 3):
        raise MultiplicityError("rotations must be 3x3 matrices")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise MultiplicityError("rotations must be finite")
    cosine = (float(np.trace(first.T @ second)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _rotation_from_row(row: object) -> np.ndarray:
    rotation = getattr(row, "rotation_camera_3x3", None)
    if rotation is not None:
        result = np.asarray(rotation, dtype=np.float64)
    else:
        quaternion = np.asarray(
            getattr(row, "quaternion_camera_xyzw", None), dtype=np.float64
        )
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise MultiplicityError(
                "candidate requires rotation_camera_3x3 or a finite xyzw quaternion"
            )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 0:
            raise MultiplicityError("candidate quaternion has zero norm")
        x, y, z, w = quaternion / norm
        result = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise MultiplicityError("candidate rotation is not a finite 3x3 matrix")
    return result


def _required_candidate_columns(candidates: pd.DataFrame) -> None:
    required = {
        "candidate_index_original",
        "position_camera_x",
        "position_camera_y",
        "position_camera_z",
        "width_m",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise MultiplicityError(f"candidate table lacks columns: {missing}")
    if not ({"rotation_camera_3x3", "quaternion_camera_xyzw"} & set(candidates.columns)):
        raise MultiplicityError("candidate table lacks a camera-frame orientation")


def cluster_pose_modes(
    candidates: pd.DataFrame,
    *,
    translation_threshold_m: float = PRIMARY_TRANSLATION_THRESHOLD_M,
    rotation_threshold_deg: float = PRIMARY_ROTATION_THRESHOLD_DEG,
    width_threshold_m: float = PRIMARY_WIDTH_THRESHOLD_M,
) -> pd.Series:
    """Assign deterministic single-link pose-mode IDs to one sample's candidates.

    Rows are first ordered by ``candidate_index_original``. Two rows are
    connected when translation, SO(3) geodesic rotation, and gripper-width
    differences all satisfy their inclusive thresholds. Union-find supplies
    the transitive closure. Returned IDs start at zero in increasing order of
    the smallest original candidate index in each component and retain the
    caller's original DataFrame index.
    """

    thresholds = PoseModeThresholds(
        float(translation_threshold_m),
        float(rotation_threshold_deg),
        float(width_threshold_m),
    )
    if candidates.empty:
        return pd.Series(index=candidates.index, dtype="int64", name="pose_mode_id")
    _required_candidate_columns(candidates)
    if candidates["candidate_index_original"].duplicated().any():
        raise MultiplicityError("candidate_index_original must be unique within a sample")
    ordered = candidates.sort_values("candidate_index_original", kind="mergesort")
    positions = ordered[
        ["position_camera_x", "position_camera_y", "position_camera_z"]
    ].to_numpy(dtype=np.float64)
    widths = ordered["width_m"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(widths)):
        raise MultiplicityError("candidate positions and widths must be finite")
    rotations = [_rotation_from_row(row) for row in ordered.itertuples(index=False)]
    union_find = _UnionFind(len(ordered))
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            if (
                float(np.linalg.norm(positions[left] - positions[right]))
                <= thresholds.translation_m
                and abs(float(widths[left] - widths[right])) <= thresholds.width_m
                and rotation_geodesic_degrees(rotations[left], rotations[right])
                <= thresholds.rotation_deg
            ):
                union_find.union(left, right)
    roots = [union_find.find(index) for index in range(len(ordered))]
    root_to_mode = {
        root: mode for mode, root in enumerate(sorted(set(roots)))
    }
    ordered_labels = pd.Series(
        [root_to_mode[root] for root in roots], index=ordered.index, dtype="int64"
    )
    return ordered_labels.reindex(candidates.index).rename("pose_mode_id")


def count_distinct_pose_modes(
    candidates: pd.DataFrame, **thresholds: float
) -> int:
    """Return the number of deterministic pose components in one sample."""

    labels = cluster_pose_modes(candidates, **thresholds)
    return int(labels.nunique()) if len(labels) else 0


def candidate_count_bin(value: int) -> str:
    """Map an exact non-negative candidate count to the preregistered bin."""

    count = int(value)
    if count < 0:
        raise MultiplicityError("candidate count cannot be negative")
    if count <= 4:
        return str(count)
    if count <= 9:
        return "5–9"
    if count <= 19:
        return "10–19"
    if count <= 49:
        return "20–49"
    return "50+"


def _count_by_sample(
    samples: pd.DataFrame, candidates: pd.DataFrame, column: str | None = None
) -> pd.Series:
    sample_ids = pd.Index(samples["sample_id"].astype(str), name="sample_id")
    if candidates.empty:
        return pd.Series(0, index=sample_ids, dtype="int64")
    if column is None:
        counts = candidates.groupby(candidates["sample_id"].astype(str), sort=False).size()
    else:
        if column not in candidates:
            raise MultiplicityError(f"candidate table lacks label column {column!r}")
        positive = candidates[column].fillna(False).astype(bool)
        counts = positive.groupby(candidates["sample_id"].astype(str), sort=False).sum()
    return counts.reindex(sample_ids, fill_value=0).astype("int64")


def build_multiplicity_table(
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    primary_positive_column: str = "gt_target_positive_primary",
    include_pose_sensitivity: bool = True,
) -> pd.DataFrame:
    """Build one multiplicity/rankability row for every manifest sample."""

    if "sample_id" not in samples:
        raise MultiplicityError("sample table lacks sample_id")
    if samples["sample_id"].astype(str).duplicated().any():
        raise MultiplicityError("sample table contains duplicate sample_id values")
    if not candidates.empty and "sample_id" not in candidates:
        raise MultiplicityError("candidate table lacks sample_id")
    known = set(samples["sample_id"].astype(str))
    unknown = set(candidates.get("sample_id", pd.Series(dtype=str)).astype(str)) - known
    if unknown:
        raise MultiplicityError(f"candidates refer to unknown samples: {sorted(unknown)[:3]}")

    result = samples.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    result = result.set_index("sample_id", drop=False)
    result["n_official_candidates"] = _count_by_sample(samples, candidates)
    result["n_pred_filtered_candidates"] = _count_by_sample(
        samples, candidates, "pred_filter_pass"
    )
    labels = {
        "n_gt_positive_primary": primary_positive_column,
        "n_gt_positive_strict": "gt_inside_raw_mask",
        "n_gt_positive_5px": "gt_inside_dilated_mask_5px",
        "n_gt_positive_10px": "gt_inside_dilated_mask_10px",
    }
    for output, source in labels.items():
        result[output] = _count_by_sample(samples, candidates, source)

    result["pre_filter_rankable"] = result["n_official_candidates"] >= 2
    result["post_filter_rankable"] = result["n_pred_filtered_candidates"] >= 2
    result["gt_rankable"] = (
        result["pre_filter_rankable"] & (result["n_gt_positive_primary"] >= 1)
    )
    result["official_candidate_count_bin"] = result["n_official_candidates"].map(
        candidate_count_bin
    )
    result["pred_filtered_candidate_count_bin"] = result[
        "n_pred_filtered_candidates"
    ].map(candidate_count_bin)
    result["official_multiplicity_class"] = np.select(
        [
            result["n_official_candidates"] == 0,
            result["n_official_candidates"] == 1,
        ],
        ["zero_candidate", "single_candidate"],
        default="multiple_candidates",
    )
    count = result["n_official_candidates"]
    positive = result["n_gt_positive_primary"]
    result["gt_positive_multiplicity_class"] = np.select(
        [
            count == 0,
            (count == 1) & (positive == 1),
            (count == 1) & (positive == 0),
            (count >= 2) & (positive == 0),
            (count >= 2) & (positive == 1),
        ],
        [
            "zero_candidate",
            "single_gt_positive",
            "single_gt_negative",
            "multiple_with_no_gt_positive",
            "multiple_with_one_gt_positive",
        ],
        default="multiple_with_multiple_gt_positive",
    )

    settings: list[tuple[float, float, str]] = [
        (
            PRIMARY_TRANSLATION_THRESHOLD_M,
            PRIMARY_ROTATION_THRESHOLD_DEG,
            "n_distinct_pose_modes",
        )
    ]
    if include_pose_sensitivity:
        settings.extend(
            (
                translation,
                rotation,
                f"n_distinct_modes_t{int(round(translation * 1000)):03d}mm_r{int(rotation):02d}deg",
            )
            for translation in TRANSLATION_SENSITIVITY_M
            for rotation in ROTATION_SENSITIVITY_DEG
            if not (
                math.isclose(translation, PRIMARY_TRANSLATION_THRESHOLD_M)
                and math.isclose(rotation, PRIMARY_ROTATION_THRESHOLD_DEG)
            )
        )
    grouped = {
        str(sample_id): group
        for sample_id, group in candidates.groupby("sample_id", sort=False)
    }
    empty = candidates.iloc[0:0]
    for translation, rotation, output in settings:
        result[output] = [
            count_distinct_pose_modes(
                grouped.get(sample_id, empty),
                translation_threshold_m=translation,
                rotation_threshold_deg=rotation,
                width_threshold_m=PRIMARY_WIDTH_THRESHOLD_M,
            )
            for sample_id in result.index
        ]
    return result.reset_index(drop=True)


def multiplicity_distribution(
    table: pd.DataFrame, count_column: str
) -> pd.DataFrame:
    """Return the fixed-bin distribution for a per-sample count column."""

    if count_column not in table:
        raise MultiplicityError(f"table lacks count column {count_column!r}")
    bins = table[count_column].map(candidate_count_bin)
    counts = bins.value_counts().reindex(COUNT_BIN_LABELS, fill_value=0)
    denominator = len(table)
    return pd.DataFrame(
        {
            "count_bin": COUNT_BIN_LABELS,
            "sample_count": counts.to_numpy(dtype=int),
            "denominator": denominator,
            "percentage": (
                counts.to_numpy(dtype=float) * 100.0 / denominator
                if denominator
                else np.zeros(len(counts), dtype=float)
            ),
        }
    )


def rankability_table(table: pd.DataFrame) -> pd.DataFrame:
    """Report the preregistered pre/post-filter rankability denominators."""

    required = {
        "pre_filter_rankable",
        "post_filter_rankable",
        "gt_rankable",
        "n_pred_filtered_candidates",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise MultiplicityError(f"multiplicity table lacks columns: {missing}")
    all_count = len(table)
    selected_count = int((table["n_pred_filtered_candidates"] >= 1).sum())
    rows = [
        ("pre_filter_rankable_all", int(table["pre_filter_rankable"].sum()), all_count),
        ("post_filter_rankable_all", int(table["post_filter_rankable"].sum()), all_count),
        (
            "post_filter_rankable_given_pred_filtered",
            int(table["post_filter_rankable"].sum()),
            selected_count,
        ),
        ("gt_rankable_all", int(table["gt_rankable"].sum()), all_count),
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "numerator": numerator,
                "denominator": denominator,
                "percentage": (
                    100.0 * numerator / denominator if denominator else float("nan")
                ),
            }
            for metric, numerator, denominator in rows
        ]
    )


compute_multiplicity = build_multiplicity_table
compute_sample_multiplicity = build_multiplicity_table


__all__ = [
    "COUNT_BIN_LABELS",
    "MultiplicityError",
    "PoseModeThresholds",
    "PRIMARY_ROTATION_THRESHOLD_DEG",
    "PRIMARY_TRANSLATION_THRESHOLD_M",
    "PRIMARY_WIDTH_THRESHOLD_M",
    "ROTATION_SENSITIVITY_DEG",
    "TRANSLATION_SENSITIVITY_M",
    "build_multiplicity_table",
    "candidate_count_bin",
    "cluster_pose_modes",
    "compute_multiplicity",
    "compute_sample_multiplicity",
    "count_distinct_pose_modes",
    "multiplicity_distribution",
    "rankability_table",
    "rotation_geodesic_degrees",
]
