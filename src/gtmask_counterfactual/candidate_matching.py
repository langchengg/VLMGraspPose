"""Branch-local candidate identities and deterministic geometry matching."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from typing import Any


ROUTES = frozenset({"g1", "c1", "d1"})
BRANCHES = frozenset({"predicted", "gt_oracle", "gt_shape_only"})
GEOMETRY_FIELDS = ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")


class CandidateIdentityError(ValueError):
    """Candidate identity or geometry violates the locked contract."""


def _finite(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CandidateIdentityError(f"invalid {field}: {value!r}") from error
    if not math.isfinite(result):
        raise CandidateIdentityError(f"non-finite {field}: {value!r}")
    return result


def canonical_geometry(
    candidate: Mapping[str, Any], *, decimal_places: int = 6
) -> dict[str, str]:
    """Return serializer-stable geometry for branch-local ID construction."""

    if decimal_places < 0 or decimal_places > 12:
        raise CandidateIdentityError("decimal_places must be between 0 and 12")
    values = {field: _finite(candidate.get(field), field=field) for field in GEOMETRY_FIELDS}
    if values["width_px"] <= 0 or values["height_px"] <= 0:
        raise CandidateIdentityError("candidate width and height must be positive")
    values["theta_deg"] %= 180.0
    return {
        field: f"{values[field]:.{decimal_places}f}" for field in GEOMETRY_FIELDS
    }


def stable_candidate_id(
    *,
    sample_id: str,
    route: str,
    branch: str,
    source_candidate_index: int | str,
    candidate: Mapping[str, Any],
    decimal_places: int = 6,
) -> str:
    """Hash the required sample/route/branch/source-index/geometry namespace."""

    sample = str(sample_id).strip()
    route_name = str(route).lower().strip()
    branch_name = str(branch).lower().strip()
    source_index = str(source_candidate_index).strip()
    if not sample or not source_index:
        raise CandidateIdentityError("sample_id and source_candidate_index are required")
    if route_name not in ROUTES:
        raise CandidateIdentityError(f"unsupported route: {route}")
    if branch_name not in BRANCHES:
        raise CandidateIdentityError(f"unsupported branch: {branch}")
    payload = {
        "branch": branch_name,
        "geometry": canonical_geometry(candidate, decimal_places=decimal_places),
        "route": route_name,
        "sample_id": sample,
        "source_candidate_index": source_index,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def periodic_angle_difference(left: float, right: float) -> float:
    """Smallest unsigned angle difference under 180-degree grasp symmetry."""

    delta = abs(_finite(left, field="left angle") - _finite(right, field="right angle"))
    return float(min(delta % 180.0, 180.0 - (delta % 180.0)))


def _polygon(candidate: Mapping[str, Any]) -> list[tuple[float, float]]:
    values = {field: _finite(candidate.get(field), field=field) for field in GEOMETRY_FIELDS}
    width = values["width_px"]
    height = values["height_px"]
    if width <= 0 or height <= 0:
        raise CandidateIdentityError("candidate width and height must be positive")
    theta = math.radians(values["theta_deg"])
    cosine, sine = math.cos(theta), math.sin(theta)
    result: list[tuple[float, float]] = []
    for local_x, local_y in (
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ):
        result.append(
            (
                values["cx_px"] + local_x * cosine - local_y * sine,
                values["cy_px"] + local_x * sine + local_y * cosine,
            )
        )
    return result


def _cross(
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    point: tuple[float, float],
) -> float:
    return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0])


def _intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> tuple[float, float]:
    segment_x, segment_y = second[0] - first[0], second[1] - first[1]
    edge_x, edge_y = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
    denominator = segment_x * edge_y - segment_y * edge_x
    if abs(denominator) < 1e-12:
        return second
    numerator = (edge_start[0] - first[0]) * edge_y - (
        edge_start[1] - first[1]
    ) * edge_x
    scale = numerator / denominator
    return first[0] + scale * segment_x, first[1] + scale * segment_y


def _clip(
    subject: list[tuple[float, float]], clipper: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    output = subject
    for edge_start, edge_end in zip(clipper, clipper[1:] + clipper[:1]):
        source = output
        output = []
        if not source:
            break
        previous = source[-1]
        previous_inside = _cross(edge_start, edge_end, previous) >= -1e-9
        for point in source:
            inside = _cross(edge_start, edge_end, point) >= -1e-9
            if inside:
                if not previous_inside:
                    output.append(
                        _intersection(previous, point, edge_start, edge_end)
                    )
                output.append(point)
            elif previous_inside:
                output.append(_intersection(previous, point, edge_start, edge_end))
            previous, previous_inside = point, inside
    return output


def _area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(polygon, polygon[1:] + polygon[:1])
        )
    ) / 2.0


def rotated_rectangle_iou(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    """Compute IoU for two finite rotated rectangles without changing labels."""

    left_polygon = _polygon(left)
    right_polygon = _polygon(right)
    intersection = _area(_clip(left_polygon, right_polygon))
    union = _area(left_polygon) + _area(right_polygon) - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def geometry_equivalence(
    predicted: Mapping[str, Any],
    gt: Mapping[str, Any],
    *,
    max_angle_deg: float = 10.0,
    max_center_distance_px: float = 4.0,
    max_width_difference_px: float = 5.0,
    minimum_rotated_iou: float = 0.5,
) -> dict[str, Any]:
    """Evaluate the pre-registered cross-branch matching rule."""

    angle = periodic_angle_difference(predicted["theta_deg"], gt["theta_deg"])
    center = math.hypot(
        _finite(predicted["cx_px"], field="predicted cx")
        - _finite(gt["cx_px"], field="GT cx"),
        _finite(predicted["cy_px"], field="predicted cy")
        - _finite(gt["cy_px"], field="GT cy"),
    )
    width = abs(
        _finite(predicted["width_px"], field="predicted width")
        - _finite(gt["width_px"], field="GT width")
    )
    iou = rotated_rectangle_iou(predicted, gt)
    local_match = center <= max_center_distance_px and width <= max_width_difference_px
    matched = angle <= max_angle_deg and (local_match or iou >= minimum_rotated_iou)
    return {
        "matched": matched,
        "periodic_angle_difference_deg": angle,
        "center_distance_px": center,
        "width_difference_px": width,
        "rotated_rectangle_iou": iou,
        "match_basis": (
            "center_width" if matched and local_match else "rotated_iou" if matched else "none"
        ),
    }


def match_candidate_pools(
    predicted_candidates: Iterable[Mapping[str, Any]],
    gt_candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Produce a deterministic one-to-one analytic match inventory.

    Source equivalence is preferred when both rows expose the same non-empty
    ``source_equivalence_key``.  Remaining admissible edges follow the locked
    geometry rule.  Matching never changes evaluator correctness labels.
    """

    predicted = [dict(row) for row in predicted_candidates]
    gt = [dict(row) for row in gt_candidates]
    for label, rows in (("predicted", predicted), ("GT", gt)):
        identities = [str(row.get("candidate_id", "")) for row in rows]
        if any(not identity for identity in identities) or len(set(identities)) != len(
            identities
        ):
            raise CandidateIdentityError(f"{label} candidate IDs must be unique")

    edges: list[tuple[tuple[Any, ...], int, int, dict[str, Any]]] = []
    for predicted_index, predicted_row in enumerate(predicted):
        for gt_index, gt_row in enumerate(gt):
            source_key = str(predicted_row.get("source_equivalence_key", ""))
            source_equivalent = bool(source_key) and source_key == str(
                gt_row.get("source_equivalence_key", "")
            )
            diagnostics = geometry_equivalence(predicted_row, gt_row)
            if not source_equivalent and not diagnostics["matched"]:
                continue
            basis = "source_nms_equivalence" if source_equivalent else diagnostics["match_basis"]
            priority = (
                0 if source_equivalent else 1,
                -float(diagnostics["rotated_rectangle_iou"]),
                float(diagnostics["center_distance_px"]),
                float(diagnostics["periodic_angle_difference_deg"]),
                float(diagnostics["width_difference_px"]),
                str(predicted_row["candidate_id"]),
                str(gt_row["candidate_id"]),
            )
            edges.append((priority, predicted_index, gt_index, {**diagnostics, "match_basis": basis}))

    matched_predicted: set[int] = set()
    matched_gt: set[int] = set()
    result: list[dict[str, Any]] = []
    for _, predicted_index, gt_index, diagnostics in sorted(edges):
        if predicted_index in matched_predicted or gt_index in matched_gt:
            continue
        matched_predicted.add(predicted_index)
        matched_gt.add(gt_index)
        result.append(
            {
                "status": "matched_pred_gt_candidate",
                "predicted_candidate_id": predicted[predicted_index]["candidate_id"],
                "gt_candidate_id": gt[gt_index]["candidate_id"],
                **diagnostics,
            }
        )
    result.extend(
        {
            "status": "pred_only_candidate",
            "predicted_candidate_id": row["candidate_id"],
            "gt_candidate_id": None,
        }
        for index, row in enumerate(predicted)
        if index not in matched_predicted
    )
    result.extend(
        {
            "status": "gt_only_candidate",
            "predicted_candidate_id": None,
            "gt_candidate_id": row["candidate_id"],
        }
        for index, row in enumerate(gt)
        if index not in matched_gt
    )
    return sorted(
        result,
        key=lambda row: (
            str(row["status"]),
            str(row.get("predicted_candidate_id") or ""),
            str(row.get("gt_candidate_id") or ""),
        ),
    )


__all__ = [
    "CandidateIdentityError",
    "canonical_geometry",
    "geometry_equivalence",
    "match_candidate_pools",
    "periodic_angle_difference",
    "rotated_rectangle_iou",
    "stable_candidate_id",
]
