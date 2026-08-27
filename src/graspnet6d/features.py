"""Leakage-safe feature contracts for the 6-DoF ranking experiment.

This module deliberately separates row metadata (candidate/group identifiers),
runtime features, and evaluator labels.  Ground-truth masks may be used to
*compute the ordinary mask-derived columns* in the declared oracle condition;
the mask source and every other ground-truth/evaluator field remain metadata
and are never model inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_GROUPS = (
    "native_quality",
    "semantic_mask",
    "width",
    "orientation",
    "collision_risk",
    "local_geometry",
)

NATIVE_QUALITY_FEATURES = (
    "native_score",
    "native_rank",
    "score_to_top1",
    "score_to_previous",
    "score_to_next",
    "score_zscore_within_group",
    "score_percentile",
)

SEMANTIC_MASK_FEATURES = (
    "center_in_mask",
    "center_mask_probability",
    "distance_to_mask_boundary",
    "normalised_distance_to_mask_centroid",
    "closing_region_mask_coverage",
    "left_finger_projected_mask_coverage",
    "right_finger_projected_mask_coverage",
    "approach_region_mask_coverage",
    "target_point_fraction_inside_closing_volume",
    "target_points_between_fingers",
    "left_contact_target_support",
    "right_contact_target_support",
    "target_support_symmetry",
    "candidate_center_to_target_centroid",
    "candidate_center_relative_to_target_bbox_x",
    "candidate_center_relative_to_target_bbox_y",
    "candidate_center_relative_to_target_bbox_z",
    "mask_area_fraction",
    "mask_entropy",
    "mean_mask_confidence",
    "boundary_uncertainty",
    "mask_depth_valid_fraction",
    "number_of_mask_components",
    "largest_component_fraction",
)

WIDTH_FEATURES = (
    "gripper_width_m",
    "width_over_target_bbox_x",
    "width_over_target_bbox_y",
    "width_over_target_bbox_diagonal",
    "estimated_closing_margin",
    "width_feasibility_flag",
)

ORIENTATION_FEATURES = (
    *(f"rotation_6d_{index}" for index in range(6)),
    "approach_vs_gravity_angle",
    "approach_vs_table_normal_angle",
    "closing_axis_vs_target_pca_axis_0",
    "closing_axis_vs_target_pca_axis_1",
    "closing_axis_vs_target_pca_axis_2",
    "approach_vs_local_surface_normal",
    "roll_consistency",
)

COLLISION_RISK_FEATURES = (
    "left_finger_occupancy",
    "right_finger_occupancy",
    "palm_occupancy",
    "closing_volume_obstacle_fraction",
    "approach_swept_volume_occupancy",
    "table_clearance_m",
    "nearest_obstacle_distance_m",
    "collision_proxy_flag",
)

LOCAL_GEOMETRY_FEATURES = (
    "local_point_density",
    "target_point_density",
    "depth_variance",
    "normal_consistency",
    "surface_curvature",
    "contact_region_planarity",
    "left_right_contact_balance",
    "valid_depth_fraction_near_grasp",
)

DEFAULT_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "native_quality": NATIVE_QUALITY_FEATURES,
    "semantic_mask": SEMANTIC_MASK_FEATURES,
    "width": WIDTH_FEATURES,
    "orientation": ORIENTATION_FEATURES,
    "collision_risk": COLLISION_RISK_FEATURES,
    "local_geometry": LOCAL_GEOMETRY_FEATURES,
}

DEFAULT_FEATURE_TO_GROUP = {
    feature: group
    for group, features in DEFAULT_FEATURE_GROUPS.items()
    for feature in features
}

# Match evaluator/GT/identity fields, not runtime proxies.  In particular,
# ``collision_proxy_flag`` is permitted while an official collision outcome is
# prohibited.  Mask-derived runtime features are also permitted, but raw GT
# mask identity, mask IoU, and the oracle/predicted condition flag are not.
_FORBIDDEN_FEATURE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|_)gt($|_)",
        r"ground[_-]?truth",
        r"(^|_)[a-z0-9_-]+[_-]id$",
        r"target[_-]?object",
        r"associated[_-]?object",
        r"object[_-]?(mesh|model|pose|transform|category|class)",
        r"(category|class)[_-]?(label|index|name)$",
        r"(^|_)(split|partition|fold)($|_)",
        r"(^|_)(camera|camera_name|scene_name|frame_name)$",
        r"(^|_)mesh($|_)",
        r"instance[_-]?(mask|label|pose)",
        r"(^|_)(relevance|label|candidate_success|target_success)($|_)",
        r"first[_-]?valid[_-]?target[_-]?rank",
        r"(official|evaluator)[_-]?(score|success|collision|friction|valid|output)",
        r"friction[_-]?(required|requirement|score|label|threshold|mu)",
        r"friction[_-]?coefficient",
        r"force[_-]?closure",
        r"(^|_)grasp[_-]?(score|valid|success)($|_)",
        r"(^|_)(is_)?collision($|_label|_truth|_status|_score)",
        r"(^|_)(is_)?collision[_-]?free$",
        r"collision[_-]?(gt|label|truth|status|score)",
        r"pose[_-]?(valid|label|error)",
        r"(mask|segmentation)[_-]?iou",
        r"mask[_-]?(source|condition|kind)",
        r"(oracle|solvable|recovered|harmful)",
        r"(^|_)(rgb|depth|mask|probability|mesh|model|source)[_-]?path$",
        r"(^|_)path$",
    )
)


@dataclass(frozen=True)
class FeatureSpec:
    """One ordered numeric input in a versioned feature schema."""

    name: str
    group: str
    allow_missing: bool = True
    unit: str | None = None


def forbidden_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Return sorted model columns that expose identity, GT, or labels."""

    return tuple(
        sorted(
            {
                str(column)
                for column in columns
                if any(
                    pattern.search(str(column))
                    for pattern in _FORBIDDEN_FEATURE_PATTERNS
                )
            }
        )
    )


def assert_no_gt_leakage(columns: Iterable[str]) -> tuple[str, ...]:
    """Validate an ordered model-input column list and preserve its order."""

    selected = tuple(str(column) for column in columns)
    if not selected or any(not column for column in selected):
        raise ValueError("model feature columns must be non-empty strings")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate model feature columns")
    forbidden = forbidden_feature_columns(selected)
    if forbidden:
        raise ValueError(f"forbidden GT/identity/label feature columns: {list(forbidden)}")
    return selected


def _normalise_schema_records(schema: Any) -> list[Mapping[str, Any]]:
    if isinstance(schema, Mapping):
        records = schema.get("features")
        if records is None and isinstance(schema.get("groups"), Mapping):
            aliases = {
                "native_confidence": "native_quality",
                "target_support_2d": "semantic_mask",
                "target_support_3d": "semantic_mask",
                "grounding_quality": "semantic_mask",
            }
            records = []
            for declared_group, names in schema["groups"].items():
                # ``rotation_6d_only`` is a named A6 subset of the full
                # orientation group in the checked-in JSON, not a second copy.
                if declared_group == "rotation_6d_only":
                    continue
                if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
                    raise ValueError(f"schema group {declared_group!r} must be a feature list")
                group = aliases.get(str(declared_group), str(declared_group))
                records.extend(
                    {"name": str(name), "group": group} for name in names
                )
        if records is None:
            # A concise {feature: group} mapping is useful in tests and tools.
            records = [
                {"name": str(name), "group": str(group)}
                for name, group in schema.items()
                if name not in {"schema_version", "description"}
            ]
    else:
        records = schema
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("feature schema must contain an ordered features list")
    return list(records)


def validate_feature_schema(schema: Any) -> tuple[FeatureSpec, ...]:
    """Parse and validate an ordered schema without silently adding features."""

    specs: list[FeatureSpec] = []
    for raw in _normalise_schema_records(schema):
        if isinstance(raw, FeatureSpec):
            raw = asdict(raw)
        if not isinstance(raw, Mapping):
            raise ValueError("every feature schema entry must be an object")
        name = str(raw.get("name", ""))
        group = str(raw.get("group", ""))
        if not name or group not in FEATURE_GROUPS:
            raise ValueError(f"invalid feature schema entry: name={name!r}, group={group!r}")
        allow_missing = raw.get("allow_missing", True)
        if not isinstance(allow_missing, bool):
            raise ValueError(f"allow_missing must be boolean for {name}")
        unit_value = raw.get("unit")
        unit = None if unit_value is None else str(unit_value)
        specs.append(
            FeatureSpec(
                name=name,
                group=group,
                allow_missing=allow_missing,
                unit=unit,
            )
        )
    assert_no_gt_leakage(spec.name for spec in specs)
    return tuple(specs)


def default_feature_schema() -> tuple[FeatureSpec, ...]:
    """Return the prompt-declared v1 feature ordering."""

    return tuple(
        FeatureSpec(name=feature, group=group)
        for group in FEATURE_GROUPS
        for feature in DEFAULT_FEATURE_GROUPS[group]
    )


def load_feature_schema(path: str | Path) -> tuple[FeatureSpec, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_feature_schema(value)


def assert_feature_frame_matches_schema(
    frame: pd.DataFrame, schema: Any
) -> tuple[str, ...]:
    """Require the feature table to match the versioned schema exactly."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("feature table must be a pandas DataFrame")
    expected = tuple(spec.name for spec in validate_feature_schema(schema))
    observed = tuple(map(str, frame.columns))
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise ValueError(
            "feature table does not exactly match schema/order; "
            f"missing={missing}, extra={extra}"
        )
    return expected


def feature_schema_sha256(schema: Any) -> str:
    specs = validate_feature_schema(schema)
    payload = json.dumps(
        [asdict(spec) for spec in specs],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_VARIANT_EXCLUDED_GROUPS = {
    "all": frozenset(),
    "q_only": frozenset(FEATURE_GROUPS[1:]),
    "drop_semantic": frozenset({"semantic_mask"}),
    "drop_collision": frozenset({"collision_risk"}),
    "drop_local_geometry": frozenset({"local_geometry"}),
    # A6 removes this group, with rotation-6D explicitly restored below as the
    # candidate's intrinsic pose representation.
    "drop_orientation": frozenset({"orientation"}),
}

_VARIANT_ALIASES = {
    "a1": "q_only",
    "a2": "all",
    "a3": "drop_semantic",
    "a4": "drop_collision",
    "a5": "drop_local_geometry",
    "a6": "drop_orientation",
    "a1_q_only": "q_only",
    "a2_all": "all",
    "a3_without_semantic_mask": "drop_semantic",
    "a4_without_collision_risk": "drop_collision",
    "a5_without_local_geometry": "drop_local_geometry",
    "a6_without_orientation_relative": "drop_orientation",
    "q-only": "q_only",
}


def select_feature_columns(schema: Any, variant: str) -> tuple[str, ...]:
    """Select one predeclared ablation without consulting labels or outcomes."""

    specs = validate_feature_schema(schema)
    key = str(variant).strip().lower()
    key = _VARIANT_ALIASES.get(key, key)
    if key not in _VARIANT_EXCLUDED_GROUPS:
        raise ValueError(
            f"unknown feature variant {variant!r}; expected one of "
            f"{sorted(_VARIANT_EXCLUDED_GROUPS)}"
        )
    excluded = _VARIANT_EXCLUDED_GROUPS[key]
    selected = tuple(
        spec.name
        for spec in specs
        if spec.group not in excluded
        # The declared A6 retains raw rotation-6D and removes target-relative
        # orientation features.  This matches configs/graspnet6d/features.yaml.
        or (key == "drop_orientation" and spec.name.startswith("rotation_6d_"))
    )
    if not selected:
        raise ValueError(f"feature variant {variant!r} selected no columns")
    return assert_no_gt_leakage(selected)


class StableMissingValueImputer:
    """Median imputation fitted on training rows with explicit indicators.

    Indicators are emitted for *every* input feature.  This makes the output
    schema independent of whether a missing value happened to occur in a small
    training split and prevents validation/test-only missingness from becoming
    invisible.
    """

    def __init__(
        self, *, all_missing_fill_values: Mapping[str, float] | None = None
    ) -> None:
        self.feature_names_in_: tuple[str, ...] | None = None
        self.fill_values_: dict[str, float] | None = None
        self.all_missing_fill_values = {
            str(column): float(value)
            for column, value in (all_missing_fill_values or {}).items()
        }
        if not all(
            np.isfinite(value) for value in self.all_missing_fill_values.values()
        ):
            raise ValueError("explicit all-missing fill values must be finite")

    @staticmethod
    def _numeric_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        observed = tuple(map(str, frame.columns))
        if observed != columns:
            raise ValueError(
                "feature columns/order differ from fitted schema: "
                f"expected {list(columns)}, observed {list(observed)}"
            )
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        nonnumeric = frame.notna() & numeric.isna()
        if bool(nonnumeric.any().any()):
            bad = sorted(nonnumeric.columns[nonnumeric.any()].astype(str).tolist())
            raise ValueError(f"non-numeric feature values in columns: {bad}")
        finite_or_missing = np.isfinite(numeric.to_numpy(dtype=float)) | numeric.isna().to_numpy()
        if not bool(finite_or_missing.all()):
            raise ValueError("infinite feature values are invalid; use NaN for missing values")
        return numeric.astype(float)

    def fit(self, training_features: pd.DataFrame) -> "StableMissingValueImputer":
        columns = assert_no_gt_leakage(training_features.columns)
        numeric = self._numeric_frame(training_features, columns)
        fills: dict[str, float] = {}
        for column in columns:
            finite = numeric[column].dropna().to_numpy(dtype=float)
            if not len(finite):
                if column not in self.all_missing_fill_values:
                    raise ValueError(
                        f"training feature {column!r} is entirely missing; "
                        "declare an explicit condition-specific fill value"
                    )
                fills[column] = self.all_missing_fill_values[column]
            else:
                fills[column] = float(np.median(finite))
        self.feature_names_in_ = columns
        self.fill_values_ = fills
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_in_ is None or self.fill_values_ is None:
            raise RuntimeError("imputer is not fitted")
        numeric = self._numeric_frame(features, self.feature_names_in_)
        missing = numeric.isna()
        transformed = numeric.fillna(self.fill_values_)
        for column in self.feature_names_in_:
            transformed[f"{column}__missing"] = missing[column].astype(np.float64)
        values = transformed.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError("imputation failed to produce finite values")
        return transformed

    def fit_transform(self, training_features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training_features).transform(training_features)

    def artifact(self) -> dict[str, Any]:
        if self.feature_names_in_ is None or self.fill_values_ is None:
            raise RuntimeError("imputer is not fitted")
        return {
            "kind": "train_median_with_all_missing_indicators",
            "feature_names_in": list(self.feature_names_in_),
            "fill_values": dict(self.fill_values_),
            "indicator_columns": [
                f"{column}__missing" for column in self.feature_names_in_
            ],
            "fit_scope": "training_only",
            "explicit_all_missing_fill_values": dict(self.all_missing_fill_values),
        }


__all__ = [
    "COLLISION_RISK_FEATURES",
    "DEFAULT_FEATURE_GROUPS",
    "FEATURE_GROUPS",
    "FeatureSpec",
    "LOCAL_GEOMETRY_FEATURES",
    "NATIVE_QUALITY_FEATURES",
    "ORIENTATION_FEATURES",
    "SEMANTIC_MASK_FEATURES",
    "StableMissingValueImputer",
    "WIDTH_FEATURES",
    "assert_no_gt_leakage",
    "assert_feature_frame_matches_schema",
    "default_feature_schema",
    "feature_schema_sha256",
    "forbidden_feature_columns",
    "load_feature_schema",
    "select_feature_columns",
    "validate_feature_schema",
]
