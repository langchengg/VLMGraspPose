"""Deterministic, auditable referring expressions derived from visible objects."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


_FAMILIES = {
    "catalog_name",
    "leftmost",
    "rightmost",
    "frontmost",
    "rearmost",
    "nearest",
    "farthest",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
}


def _canonical_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _point(value: ArrayLike, name: str) -> tuple[float, float, float]:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return tuple(float(item) for item in point)


@dataclass(frozen=True, slots=True)
class VisibleObject:
    object_id: str | int
    catalog_name: str
    center_camera_m: tuple[float, float, float] | ArrayLike
    center_table_m: tuple[float, float, float] | ArrayLike
    color: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.object_id, bool) or not str(self.object_id).strip():
            raise ValueError("object_id must be non-empty")
        if not _canonical_name(self.catalog_name):
            raise ValueError("catalog_name must be non-empty")
        object.__setattr__(
            self, "center_camera_m", _point(self.center_camera_m, "center_camera_m")
        )
        object.__setattr__(
            self, "center_table_m", _point(self.center_table_m, "center_table_m")
        )
        if self.color is not None:
            normalized_color = _canonical_name(self.color)
            object.__setattr__(self, "color", normalized_color or None)


@dataclass(frozen=True, slots=True)
class LanguageExpression:
    template_family: str
    attributes: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.template_family not in _FAMILIES:
            raise ValueError(f"unsupported template family: {self.template_family}")
        normalized = json.loads(
            json.dumps(
                dict(self.attributes),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        forbidden_keys = {
            str(key).casefold()
            for key in normalized
            if str(key).casefold()
            in {"target_id", "target_object_id", "object_id", "instance_id"}
        }
        if forbidden_keys:
            raise ValueError(
                "language predicates may not encode hidden target identifiers: "
                f"{sorted(forbidden_keys)}"
            )
        object.__setattr__(self, "attributes", normalized)


@dataclass(frozen=True, slots=True)
class ResolvedLanguageQuery:
    query: str
    template_family: str
    attributes: Mapping[str, Any]
    resolver_result: tuple[str | int, ...]
    is_unique: bool
    provenance: str = "derived"


def _validated_objects(objects: Iterable[VisibleObject]) -> tuple[VisibleObject, ...]:
    values = tuple(objects)
    identifiers = [str(item.object_id) for item in values]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("visible object IDs must be unique")
    return values


def _matching_name(
    objects: Iterable[VisibleObject], name: str, color: str | None = None
) -> list[VisibleObject]:
    canonical = _canonical_name(name)
    result = [
        item for item in objects if _canonical_name(item.catalog_name) == canonical
    ]
    if color is not None:
        canonical_color = _canonical_name(color)
        result = [item for item in result if item.color == canonical_color]
    return result


def _extrema(
    objects: tuple[VisibleObject, ...],
    values: np.ndarray,
    *,
    minimum: bool,
    tolerance_m: float,
) -> tuple[str | int, ...]:
    if not objects:
        return ()
    extremum = float(np.min(values) if minimum else np.max(values))
    matches = [
        item.object_id
        for item, value in zip(objects, values, strict=True)
        if abs(float(value) - extremum) <= tolerance_m
    ]
    return tuple(sorted(matches, key=str))


def resolve_expression(
    expression: LanguageExpression,
    objects: Iterable[VisibleObject],
    *,
    tolerance_m: float = 1e-6,
) -> tuple[str | int, ...]:
    """Resolve structured semantics over all visible objects, without ID tie-breaks."""

    values = _validated_objects(objects)
    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("tolerance_m must be finite and non-negative")
    family = expression.template_family
    attributes = expression.attributes
    color = attributes.get("color")
    named = (
        tuple(_matching_name(values, str(attributes["catalog_name"]), color))
        if "catalog_name" in attributes
        else values
    )
    if family == "catalog_name":
        return tuple(sorted((item.object_id for item in named), key=str))

    if family in {"leftmost", "rightmost", "frontmost", "rearmost"}:
        if attributes.get("frame", "camera") != "camera":
            raise ValueError(f"{family} is defined only in the camera frame")
        axis = 0 if family in {"leftmost", "rightmost"} else 2
        coordinate = np.asarray([item.center_camera_m[axis] for item in named])
        return _extrema(
            named,
            coordinate,
            minimum=family in {"leftmost", "frontmost"},
            tolerance_m=tolerance_m,
        )
    if family in {"nearest", "farthest"}:
        if attributes.get("frame", "camera") != "camera":
            raise ValueError(f"{family} is defined only in the camera frame")
        distance = np.asarray(
            [np.linalg.norm(item.center_camera_m) for item in named], dtype=np.float64
        )
        return _extrema(
            named,
            distance,
            minimum=family == "nearest",
            tolerance_m=tolerance_m,
        )

    if attributes.get("frame", "camera") != "camera":
        raise ValueError(f"{family} is defined only in the camera frame")
    reference_name = str(attributes.get("reference_catalog_name", ""))
    references = _matching_name(values, reference_name, attributes.get("reference_color"))
    if len(references) != 1:
        return ()
    reference = references[0]
    candidates = [item for item in named if item.object_id != reference.object_id]
    if family == "left_of":
        candidates = [
            item
            for item in candidates
            if item.center_camera_m[0] < reference.center_camera_m[0] - tolerance_m
        ]
    elif family == "right_of":
        candidates = [
            item
            for item in candidates
            if item.center_camera_m[0] > reference.center_camera_m[0] + tolerance_m
        ]
    elif family == "in_front_of":
        candidates = [
            item
            for item in candidates
            if item.center_camera_m[2] < reference.center_camera_m[2] - tolerance_m
        ]
    elif family == "behind":
        candidates = [
            item
            for item in candidates
            if item.center_camera_m[2] > reference.center_camera_m[2] + tolerance_m
        ]
    return tuple(sorted((item.object_id for item in candidates), key=str))


def render_expression(expression: LanguageExpression) -> str:
    family = expression.template_family
    attributes = expression.attributes
    name = str(attributes.get("catalog_name", "object"))
    color = attributes.get("color")
    description = f"{color} {name}" if color else name
    if family == "catalog_name":
        return f"Pick the {description}."
    if family in {"leftmost", "rightmost", "frontmost", "rearmost"}:
        return f"Grasp the {family} {description}."
    if family in {"nearest", "farthest"}:
        direction = "closest to" if family == "nearest" else "farthest from"
        return f"Pick the {description} {direction} the camera."
    relation_text = {
        "left_of": "to the left of",
        "right_of": "to the right of",
        "in_front_of": "in front of",
        "behind": "behind",
    }[family]
    reference = str(attributes["reference_catalog_name"])
    reference_color = attributes.get("reference_color")
    reference_description = (
        f"{reference_color} {reference}" if reference_color else reference
    )
    return f"Grasp the {description} {relation_text} the {reference_description}."


def resolve_query(
    expression: LanguageExpression,
    objects: Iterable[VisibleObject],
    *,
    tolerance_m: float = 1e-6,
) -> ResolvedLanguageQuery:
    result = resolve_expression(expression, objects, tolerance_m=tolerance_m)
    return ResolvedLanguageQuery(
        query=render_expression(expression),
        template_family=expression.template_family,
        attributes=dict(expression.attributes),
        resolver_result=result,
        is_unique=len(result) == 1,
    )


def generate_unique_queries_for_target(
    objects: Iterable[VisibleObject],
    target_object_id: str | int,
    *,
    tolerance_m: float = 1e-6,
) -> tuple[ResolvedLanguageQuery, ...]:
    """Generate only predicates whose ordinary resolver uniquely selects target.

    ``target_object_id`` controls which valid outputs are retained; it is never
    inserted into the expression or used by the resolver to break a tie.
    """

    values = _validated_objects(objects)
    matches = [item for item in values if item.object_id == target_object_id]
    if len(matches) != 1:
        raise KeyError(f"target object {target_object_id!r} is not uniquely visible")
    target = matches[0]
    base = {"catalog_name": target.catalog_name}
    if target.color is not None:
        base["color"] = target.color
    predicates = [LanguageExpression("catalog_name", base)]
    predicates.extend(
        LanguageExpression(family, {**base, "frame": "camera"})
        for family in ("leftmost", "rightmost", "frontmost", "rearmost")
    )
    predicates.extend(
        LanguageExpression(family, {**base, "frame": "camera"})
        for family in ("nearest", "farthest")
    )
    # A reference must itself be unique by observable catalog name/color.
    for reference in sorted(values, key=lambda item: str(item.object_id)):
        if reference.object_id == target.object_id:
            continue
        reference_matches = _matching_name(
            values, reference.catalog_name, reference.color
        )
        if len(reference_matches) != 1:
            continue
        reference_attributes: dict[str, Any] = {
            **base,
            "reference_catalog_name": reference.catalog_name,
            "frame": "camera",
        }
        if reference.color is not None:
            reference_attributes["reference_color"] = reference.color
        predicates.extend(
            LanguageExpression(family, reference_attributes)
            for family in ("left_of", "right_of", "in_front_of", "behind")
        )

    generated: list[ResolvedLanguageQuery] = []
    seen_queries: set[str] = set()
    for predicate in predicates:
        query = resolve_query(predicate, values, tolerance_m=tolerance_m)
        if query.resolver_result == (target.object_id,) and query.query not in seen_queries:
            generated.append(query)
            seen_queries.add(query.query)
    return tuple(generated)


def assert_query_resolves_uniquely(
    query: ResolvedLanguageQuery, target_object_id: str | int
) -> None:
    if not query.is_unique or query.resolver_result != (target_object_id,):
        raise ValueError(
            f"query does not uniquely resolve target {target_object_id!r}: "
            f"{query.resolver_result}"
        )
