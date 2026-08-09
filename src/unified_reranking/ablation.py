"""Predeclared, label-independent feature families for Validation ablations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from .contracts import assert_model_feature_columns


FEATURE_FAMILY_ORDER = (
    "native_calibration",
    "soft_target_support",
    "jaw_geometry",
    "angle_agreement",
    "depth_contact",
    "collision_proxy",
    "backend_consensus",
    "reliability_context",
)


def feature_family(column: str) -> str:
    """Map a frozen column name to one predeclared semantic family."""

    name = str(column).lower()
    # Precedence disambiguates names such as contact_depth and angle_to_mask.
    if any(token in name for token in ("collision", "obstacle", "sweep", "intrusion")):
        return "collision_proxy"
    if any(
        token in name
        for token in ("depth", "plane", "surface", "z_center", "z_m", "normal")
    ):
        return "depth_contact"
    if any(token in name for token in ("angle", "theta", "orientation")):
        return "angle_agreement"
    if any(
        token in name
        for token in ("jaw", "contact", "width", "closing", "grasp_span", "finger")
    ):
        return "jaw_geometry"
    if any(token in name for token in ("backend", "consensus", "agreement", "dense_")):
        return "backend_consensus"
    if any(
        token in name
        for token in (
            "probability",
            "p_center",
            "inside_mask",
            "binary_coverage",
            "mask_boundary",
            "mask_support",
            "component",
        )
    ):
        return "soft_target_support"
    if any(
        token in name
        for token in ("native", "calibrated", "base_logit", "original_score", "rank")
    ):
        return "native_calibration"
    return "reliability_context"


def feature_families(columns: Iterable[str]) -> OrderedDict[str, tuple[str, ...]]:
    """Return present families in fixed order and preserve frozen column order."""

    selected = assert_model_feature_columns(columns)
    grouped = OrderedDict(
        (
            family,
            tuple(column for column in selected if feature_family(column) == family),
        )
        for family in FEATURE_FAMILY_ORDER
    )
    result = OrderedDict(
        (family, values) for family, values in grouped.items() if values
    )
    if len(result) < 2:
        raise ValueError(
            "feature-family ablation requires at least two present families"
        )
    if tuple(column for values in result.values() for column in values) != tuple(
        sorted(
            selected,
            key=lambda column: (
                FEATURE_FAMILY_ORDER.index(feature_family(column)),
                selected.index(column),
            ),
        )
    ):
        # The family map is exhaustive; this assertion protects future edits.
        raise AssertionError("feature-family partition is not deterministic")
    return result


def ablation_feature_sets(
    columns: Iterable[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build cumulative and leave-one-family-out feature-set contracts."""

    selected = assert_model_feature_columns(columns)
    families = feature_families(selected)
    cumulative: list[dict[str, object]] = []
    included_families: list[str] = []
    for family in families:
        included_families.append(family)
        included = tuple(
            column for column in selected if feature_family(column) in included_families
        )
        cumulative.append(
            {
                "ablation_type": "cumulative",
                "family": family,
                "family_order": FEATURE_FAMILY_ORDER.index(family),
                "included_families": list(included_families),
                "included_features": list(included),
            }
        )
    leave_one: list[dict[str, object]] = []
    for family in families:
        included = tuple(
            column for column in selected if feature_family(column) != family
        )
        leave_one.append(
            {
                "ablation_type": "leave_one_family_out",
                "family": family,
                "family_order": FEATURE_FAMILY_ORDER.index(family),
                "included_families": [value for value in families if value != family],
                "included_features": list(included),
            }
        )
    return cumulative, leave_one


__all__ = [
    "FEATURE_FAMILY_ORDER",
    "ablation_feature_sets",
    "feature_families",
    "feature_family",
]
