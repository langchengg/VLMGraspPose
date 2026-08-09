"""Label-free, permutation-equivariant candidate-set relation features.

The public contract is :func:`extract_candidate_relation_features`.  It returns
``(G10_FEATURE_NAMES, values)`` where ``values`` has shape
``(K, G10_FEATURE_DIM)`` and row ``i`` belongs to ``candidates[i]``.  Permuting
the candidates and every candidate-aligned input therefore permutes the rows and
does not change their contents.

The inputs are limited to frozen candidate geometry/q fields and cached CROG
head or crop evidence.  No ground-truth or correctness fields are read.  The
``rectangle_overlap`` features are continuous rotated-rectangle
intersection-over-union values.  They deliberately avoid the literal metric
token reserved by :mod:`failure_analysis.reranking_v3.schema` for evaluation
data.

``q_rank`` is an explicit candidate feature.  It is never inferred from list
position, and it must travel with its candidate under a permutation.  A false
``candidate_mask`` entry is ignored by every aggregate and receives an all-zero
output row, which also permits padded candidate batches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from .aligned_crops import CROP_CHANNELS


_EPSILON = 1e-12
_AXIS_HEIGHT_FRACTION = 0.25
_CONTACT_CENTER_FRACTION = 0.35
_CONTACT_WIDTH_FRACTION = 0.20


_CANDIDATE_FEATURE_NAMES = (
    "g10_candidate_q_rank",
    "g10_candidate_q_rank_fraction",
    "g10_candidate_is_q_top1",
    "g10_candidate_q_mass_share",
    "g10_candidate_quality_mass_share",
    "g10_candidate_mask_support_mass_share",
    "g10_candidate_q_peer_win_fraction",
    "g10_candidate_quality_peer_win_fraction",
    "g10_candidate_mask_support_peer_win_fraction",
    "g10_candidate_q_dominance_margin",
    "g10_candidate_quality_dominance_margin",
    "g10_candidate_mask_support_dominance_margin",
    "g10_candidate_q_minus_set_mean",
    "g10_candidate_quality_minus_set_mean",
    "g10_candidate_mask_support_minus_set_mean",
    "g10_candidate_head_disagreement",
    "g10_candidate_quality_evidence_available",
    "g10_candidate_mask_evidence_available",
)

_PAIR_FEATURE_NAMES = tuple(
    f"g10_pair_{metric}_{statistic}"
    for metric, statistics in (
        ("center_distance_px", ("min", "mean", "max", "std")),
        ("center_distance_fraction", ("min", "mean", "max", "std")),
        ("axial_angle_difference_fraction", ("min", "mean", "max", "std")),
        ("width_difference_fraction", ("min", "mean", "max", "std")),
        ("q_delta", ("min", "mean", "max", "std")),
        ("q_absolute_difference", ("mean", "max")),
        ("quality_delta", ("min", "mean", "max", "std")),
        ("quality_absolute_difference", ("mean", "max")),
        ("mask_support_delta", ("min", "mean", "max", "std")),
        ("mask_support_absolute_difference", ("mean", "max")),
        ("rectangle_overlap", ("mean", "max")),
        ("axis_overlap", ("mean", "max")),
        ("contact_overlap", ("mean", "max")),
        ("same_local_peak_proxy", ("mean", "max", "neighbour_fraction")),
    )
    for statistic in statistics
)

_TOP1_FEATURE_NAMES = (
    "g10_relative_top1_center_distance_fraction",
    "g10_relative_top1_axial_angle_difference_fraction",
    "g10_relative_top1_width_difference_fraction",
    "g10_relative_top1_q_delta",
    "g10_relative_top1_quality_delta",
    "g10_relative_top1_mask_support_delta",
    "g10_relative_top1_rectangle_overlap",
    "g10_relative_top1_axis_overlap",
    "g10_relative_top1_contact_overlap",
    "g10_relative_top1_same_local_peak_proxy",
)

_SET_FEATURE_NAMES = (
    "g10_set_valid_fraction",
    "g10_set_q_entropy",
    "g10_set_quality_entropy",
    "g10_set_mask_support_entropy",
    "g10_set_q_disagreement",
    "g10_set_quality_disagreement",
    "g10_set_mask_support_disagreement",
    "g10_set_head_disagreement",
    "g10_set_center_diversity",
    "g10_set_axial_angle_diversity",
    "g10_set_width_diversity",
    "g10_set_rectangle_diversity",
    "g10_set_axis_diversity",
    "g10_set_contact_diversity",
    "g10_set_same_local_peak_density",
)

G10_FEATURE_NAMES = (
    _CANDIDATE_FEATURE_NAMES
    + _PAIR_FEATURE_NAMES
    + _TOP1_FEATURE_NAMES
    + _SET_FEATURE_NAMES
)
"""Stable ordered names for the columns returned by this module."""

G10_FEATURE_DIM = len(G10_FEATURE_NAMES)
"""Number of G10 columns; output shape is always ``(K, G10_FEATURE_DIM)``."""


def _stable_mean(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if not flat.size:
        return 0.0
    return float(math.fsum(sorted(map(float, flat))) / flat.size)


def _stable_std(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if not flat.size:
        return 0.0
    mean = _stable_mean(flat)
    return float(math.sqrt(math.fsum(sorted((float(value) - mean) ** 2 for value in flat)) / flat.size))


def _mass_distribution(values: np.ndarray) -> np.ndarray:
    """Turn non-negative evidence into shares, using uniform mass at zero."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    if not values.size:
        return values
    total = math.fsum(sorted(map(float, values)))
    if total <= _EPSILON:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    probabilities = _mass_distribution(values)
    terms = [
        -float(probability) * math.log(float(probability))
        for probability in probabilities
        if probability > 0.0
    ]
    return float(math.fsum(sorted(terms)) / math.log(values.size))


def _candidate_mask_array(candidate_mask: Sequence[bool] | np.ndarray | None, count: int) -> np.ndarray:
    if candidate_mask is None:
        return np.ones(count, dtype=bool)
    raw = np.asarray(candidate_mask)
    if raw.shape != (count,):
        raise ValueError(f"candidate_mask must have shape ({count},), received {raw.shape}")
    if raw.dtype != np.bool_:
        if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
            raise ValueError("candidate_mask must contain only booleans or finite 0/1 values")
        if not np.isin(raw, (0, 1)).all():
            raise ValueError("candidate_mask must contain only booleans or 0/1 values")
    return raw.astype(bool, copy=False)


def _candidate_vectors(
    candidates: Sequence[Mapping[str, Any]],
    candidate_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(candidates)
    result = {
        "q_rank": np.zeros(count, dtype=np.float64),
        "q": np.zeros(count, dtype=np.float64),
        "cx": np.zeros(count, dtype=np.float64),
        "cy": np.zeros(count, dtype=np.float64),
        "width": np.zeros(count, dtype=np.float64),
        "height": np.zeros(count, dtype=np.float64),
        "angle": np.zeros(count, dtype=np.float64),
    }
    fields = {
        "q_rank": "q_rank",
        "q": "q_raw",
        "cx": "cx",
        "cy": "cy",
        "width": "width_px",
        "height": "height_px",
        "angle": "angle_deg",
    }
    for index in np.flatnonzero(candidate_mask):
        candidate = candidates[int(index)]
        if not isinstance(candidate, Mapping):
            raise TypeError(f"valid candidate {index} is not a mapping")
        for target, source in fields.items():
            if source not in candidate:
                raise KeyError(f"valid candidate {index} is missing {source!r}")
            value = float(candidate[source])
            if not math.isfinite(value):
                raise ValueError(f"valid candidate {index} has non-finite {source!r}")
            result[target][index] = value
        if result["width"][index] < 0.0 or result["height"][index] < 0.0:
            raise ValueError(f"valid candidate {index} has negative rectangle size")
        if result["q_rank"][index] < 0.0 or not result["q_rank"][index].is_integer():
            raise ValueError(f"valid candidate {index} has invalid q_rank")
    valid_ranks = result["q_rank"][candidate_mask]
    if len(np.unique(valid_ranks)) != len(valid_ranks):
        raise ValueError("valid candidates must have unique explicit q_rank values")
    return result


def _head_column(
    head_features: np.ndarray | None,
    feature_names: Sequence[str] | None,
    preferred_names: Sequence[str],
) -> np.ndarray | None:
    if head_features is None:
        if feature_names is not None:
            raise ValueError("head_feature_names were provided without head_features")
        return None
    if feature_names is None:
        raise ValueError("head_features require head_feature_names")
    names = tuple(map(str, feature_names))
    if len(names) != head_features.shape[1] or len(names) != len(set(names)):
        raise ValueError("head feature names must be unique and match the feature dimension")
    by_name = {name: index for index, name in enumerate(names)}
    selected = next((by_name[name] for name in preferred_names if name in by_name), None)
    return None if selected is None else np.asarray(head_features[:, selected], dtype=np.float64)


def _crop_evidence(
    crops: np.ndarray | None,
    crop_channels: Sequence[str] | None,
    count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if crops is None:
        if crop_channels is not None:
            raise ValueError("crop_channels were provided without crops")
        return None, None
    array = np.asarray(crops, dtype=np.float64)
    if array.ndim != 4 or array.shape[0] != count:
        raise ValueError(f"crops must have shape (K,C,H,W), received {array.shape}")
    channels = tuple(CROP_CHANNELS if crop_channels is None else map(str, crop_channels))
    if len(channels) != array.shape[1] or len(channels) != len(set(channels)):
        raise ValueError("crop channel names must be unique and match the channel dimension")
    height, width = array.shape[-2:]
    if height <= 0 or width <= 0:
        raise ValueError("crop spatial dimensions must be positive")
    by_name = {name: index for index, name in enumerate(channels)}
    quality = None
    if "quality_probability" in by_name:
        quality = array[:, by_name["quality_probability"], height // 2, width // 2]
    mask_support = None
    if "mask_probability" in by_name:
        rows = np.abs(np.linspace(-1.0, 1.0, height)) <= 0.22
        region = array[:, by_name["mask_probability"], rows, :]
        mask_support = np.asarray(
            [_stable_mean(values[np.isfinite(values)]) for values in region],
            dtype=np.float64,
        )
    return quality, mask_support


def _resolve_evidence(
    *,
    q: np.ndarray,
    valid: np.ndarray,
    head_features: np.ndarray | None,
    head_feature_names: Sequence[str] | None,
    crops: np.ndarray | None,
    crop_channels: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(q)
    if head_features is not None:
        head_features = np.asarray(head_features, dtype=np.float64)
        if head_features.ndim != 2 or head_features.shape[0] != count:
            raise ValueError(f"head_features must have shape (K,F), received {head_features.shape}")
    head_quality = _head_column(
        head_features,
        head_feature_names,
        ("g1_q_probability_center", "g0_q_probability"),
    )
    head_mask = _head_column(
        head_features,
        head_feature_names,
        ("g2_mask_axis_soft", "g2_mask_contact_soft", "g2_mask_roi_soft"),
    )
    crop_quality, crop_mask = _crop_evidence(crops, crop_channels, count)

    quality = np.asarray(q, dtype=np.float64).copy()
    mask_support = np.zeros(count, dtype=np.float64)
    quality_available = np.zeros(count, dtype=np.float64)
    mask_available = np.zeros(count, dtype=np.float64)
    for index in np.flatnonzero(valid):
        if head_quality is not None and math.isfinite(float(head_quality[index])):
            quality[index] = float(head_quality[index])
            quality_available[index] = 1.0
        elif crop_quality is not None and math.isfinite(float(crop_quality[index])):
            quality[index] = float(crop_quality[index])
            quality_available[index] = 1.0
        if head_mask is not None and math.isfinite(float(head_mask[index])):
            mask_support[index] = float(head_mask[index])
            mask_available[index] = 1.0
        elif crop_mask is not None and math.isfinite(float(crop_mask[index])):
            mask_support[index] = float(crop_mask[index])
            mask_available[index] = 1.0
    quality[valid] = np.clip(quality[valid], 0.0, 1.0)
    mask_support[valid] = np.clip(mask_support[valid], 0.0, 1.0)
    return quality, mask_support, quality_available, mask_available


def _rectangle_polygon(cx: float, cy: float, width: float, height: float, angle: float) -> np.ndarray:
    return np.asarray(
        cv2.boxPoints(((float(cx), float(cy)), (float(width), float(height)), -float(angle))),
        dtype=np.float32,
    )


def _candidate_regions(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle: float,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if width <= _EPSILON or height <= _EPSILON:
        return (), (), ()
    # Canonicalising the axial orientation is more than cosmetic: OpenCV's
    # convex intersection can otherwise return different areas for nearly
    # identical polygons represented at angles differing by 180 degrees.
    axial_angle = ((float(angle) + 90.0) % 180.0) - 90.0
    rectangle = (_rectangle_polygon(cx, cy, width, height, axial_angle),)
    axis = (_rectangle_polygon(cx, cy, width, height * _AXIS_HEIGHT_FRACTION, axial_angle),)
    theta = math.radians(axial_angle)
    direction_x, direction_y = math.cos(theta), -math.sin(theta)
    offset = width * _CONTACT_CENTER_FRACTION
    contact_width = width * _CONTACT_WIDTH_FRACTION
    contacts = tuple(
        _rectangle_polygon(
            cx + sign * offset * direction_x,
            cy + sign * offset * direction_y,
            contact_width,
            height,
            axial_angle,
        )
        for sign in (-1.0, 1.0)
    )
    return rectangle, axis, contacts


def _region_overlap(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]) -> float:
    """Continuous intersection-over-union for unions of disjoint rectangles."""
    left_area = math.fsum(abs(float(cv2.contourArea(polygon))) for polygon in left)
    right_area = math.fsum(abs(float(cv2.contourArea(polygon))) for polygon in right)
    if left_area <= _EPSILON or right_area <= _EPSILON:
        return 0.0
    intersections = []
    for left_polygon in left:
        for right_polygon in right:
            area, _ = cv2.intersectConvexConvex(left_polygon, right_polygon)
            if math.isfinite(float(area)) and area > 0.0:
                intersections.append(float(area))
    intersection = min(math.fsum(sorted(intersections)), left_area, right_area)
    union = left_area + right_area - intersection
    if union <= _EPSILON:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


def _aggregate(features: dict[str, np.ndarray], prefix: str, row: int, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    features[f"{prefix}_min"][row] = float(np.min(values))
    features[f"{prefix}_mean"][row] = _stable_mean(values)
    features[f"{prefix}_max"][row] = float(np.max(values))
    features[f"{prefix}_std"][row] = _stable_std(values)


def _pairwise_relations(
    vectors: Mapping[str, np.ndarray],
    quality: np.ndarray,
    mask_support: np.ndarray,
    valid_indices: np.ndarray,
    image_diagonal: float,
) -> dict[str, np.ndarray]:
    count = len(vectors["q"])
    cx, cy = vectors["cx"], vectors["cy"]
    center_px = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    center_fraction = center_px / max(image_diagonal, _EPSILON)
    angle_difference = np.abs(
        ((vectors["angle"][:, None] - vectors["angle"][None, :] + 90.0) % 180.0) - 90.0
    ) / 90.0
    width_denominator = np.maximum(
        np.maximum(vectors["width"][:, None], vectors["width"][None, :]),
        1.0,
    )
    width_difference = np.abs(vectors["width"][:, None] - vectors["width"][None, :]) / width_denominator
    q_delta = vectors["q"][:, None] - vectors["q"][None, :]
    quality_delta = quality[:, None] - quality[None, :]
    mask_delta = mask_support[:, None] - mask_support[None, :]
    rectangle_overlap = np.zeros((count, count), dtype=np.float64)
    axis_overlap = np.zeros_like(rectangle_overlap)
    contact_overlap = np.zeros_like(rectangle_overlap)
    regions: dict[int, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]] = {}
    for index in valid_indices:
        i = int(index)
        regions[i] = _candidate_regions(
            vectors["cx"][i],
            vectors["cy"][i],
            vectors["width"][i],
            vectors["height"][i],
            vectors["angle"][i],
        )
    for left_position, left_index in enumerate(valid_indices):
        i = int(left_index)
        for right_index in valid_indices[left_position:]:
            j = int(right_index)
            overlaps = tuple(_region_overlap(left, right) for left, right in zip(regions[i], regions[j], strict=True))
            rectangle_overlap[i, j] = rectangle_overlap[j, i] = overlaps[0]
            axis_overlap[i, j] = axis_overlap[j, i] = overlaps[1]
            contact_overlap[i, j] = contact_overlap[j, i] = overlaps[2]
    average_width = 0.5 * (vectors["width"][:, None] + vectors["width"][None, :])
    peak_center_scale = np.maximum(0.5 * average_width, 1.0)
    same_peak = np.exp(
        -0.5
        * (
            (center_px / peak_center_scale) ** 2
            + (angle_difference * 90.0 / 15.0) ** 2
            + (np.abs(q_delta) / 0.10) ** 2
        )
    )
    return {
        "center_px": center_px,
        "center_fraction": center_fraction,
        "angle_fraction": angle_difference,
        "width_fraction": width_difference,
        "q_delta": q_delta,
        "quality_delta": quality_delta,
        "mask_delta": mask_delta,
        "rectangle_overlap": rectangle_overlap,
        "axis_overlap": axis_overlap,
        "contact_overlap": contact_overlap,
        "same_peak": same_peak,
    }


def extract_candidate_relation_features(
    candidates: Sequence[Mapping[str, Any]],
    *,
    image_shape: tuple[int, int],
    head_features: np.ndarray | None = None,
    head_feature_names: Sequence[str] | None = None,
    crops: np.ndarray | None = None,
    crop_channels: Sequence[str] | None = None,
    candidate_mask: Sequence[bool] | np.ndarray | None = None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return label-free G10 aggregate relations with shape ``(K, G10_FEATURE_DIM)``.

    ``head_features`` must be the unnormalised cached ``(K,F)`` values.  Quality
    and mask evidence use ``g1_q_probability_center`` and
    ``g2_mask_axis_soft`` when present.  A missing/non-finite candidate value
    falls back to the corresponding cached crop channel.  Quality finally
    falls back to frozen ``q_raw``; mask support falls back to zero, with the
    two availability columns making either fallback explicit.
    """
    count = len(candidates)
    if len(image_shape) != 2:
        raise ValueError("image_shape must be (height, width)")
    image_height, image_width = map(float, image_shape)
    if not math.isfinite(image_height) or not math.isfinite(image_width) or image_height <= 0.0 or image_width <= 0.0:
        raise ValueError("image_shape dimensions must be finite and positive")
    valid = _candidate_mask_array(candidate_mask, count)
    output = np.zeros((count, G10_FEATURE_DIM), dtype=np.float32)
    if not np.any(valid):
        return G10_FEATURE_NAMES, output

    vectors = _candidate_vectors(candidates, valid)
    quality, mask_support, quality_available, mask_available = _resolve_evidence(
        q=vectors["q"],
        valid=valid,
        head_features=head_features,
        head_feature_names=head_feature_names,
        crops=crops,
        crop_channels=crop_channels,
    )
    valid_indices = np.flatnonzero(valid)
    image_diagonal = math.hypot(image_width, image_height)
    pair = _pairwise_relations(vectors, quality, mask_support, valid_indices, image_diagonal)
    features = {name: np.zeros(count, dtype=np.float64) for name in G10_FEATURE_NAMES}

    q_valid = vectors["q"][valid]
    quality_valid = quality[valid]
    mask_valid = mask_support[valid]
    distributions = {
        "q": _mass_distribution(q_valid),
        "quality": _mass_distribution(quality_valid),
        "mask_support": _mass_distribution(mask_valid),
    }
    stable_means = {
        "q": _stable_mean(q_valid),
        "quality": _stable_mean(quality_valid),
        "mask_support": _stable_mean(mask_valid),
    }
    ranks = vectors["q_rank"][valid]
    rank_denominator = max(float(np.max(ranks)), 1.0)
    top1_index = int(valid_indices[int(np.argmin(ranks))])
    source_values = {
        "q": vectors["q"],
        "quality": quality,
        "mask_support": mask_support,
    }
    head_disagreement = np.zeros(count, dtype=np.float64)

    for valid_position, raw_index in enumerate(valid_indices):
        index = int(raw_index)
        peers = valid_indices[valid_indices != index]
        features["g10_candidate_q_rank"][index] = vectors["q_rank"][index]
        features["g10_candidate_q_rank_fraction"][index] = vectors["q_rank"][index] / rank_denominator
        features["g10_candidate_is_q_top1"][index] = float(index == top1_index)
        for evidence_name in ("q", "quality", "mask_support"):
            values = source_values[evidence_name]
            features[f"g10_candidate_{evidence_name}_mass_share"][index] = distributions[evidence_name][valid_position]
            features[f"g10_candidate_{evidence_name}_minus_set_mean"][index] = values[index] - stable_means[evidence_name]
            if peers.size:
                peer_values = values[peers]
                wins = np.sum(values[index] > peer_values) + 0.5 * np.sum(values[index] == peer_values)
                features[f"g10_candidate_{evidence_name}_peer_win_fraction"][index] = float(wins / peers.size)
                features[f"g10_candidate_{evidence_name}_dominance_margin"][index] = values[index] - float(np.max(peer_values))
        head_disagreement[index] = _stable_std(
            np.asarray([np.clip(vectors["q"][index], 0.0, 1.0), quality[index], mask_support[index]])
        )
        features["g10_candidate_head_disagreement"][index] = head_disagreement[index]
        features["g10_candidate_quality_evidence_available"][index] = quality_available[index]
        features["g10_candidate_mask_evidence_available"][index] = mask_available[index]

        if peers.size:
            _aggregate(features, "g10_pair_center_distance_px", index, pair["center_px"][index, peers])
            _aggregate(features, "g10_pair_center_distance_fraction", index, pair["center_fraction"][index, peers])
            _aggregate(features, "g10_pair_axial_angle_difference_fraction", index, pair["angle_fraction"][index, peers])
            _aggregate(features, "g10_pair_width_difference_fraction", index, pair["width_fraction"][index, peers])
            for name, matrix in (
                ("q", pair["q_delta"]),
                ("quality", pair["quality_delta"]),
                ("mask_support", pair["mask_delta"]),
            ):
                differences = matrix[index, peers]
                _aggregate(features, f"g10_pair_{name}_delta", index, differences)
                absolute = np.abs(differences)
                features[f"g10_pair_{name}_absolute_difference_mean"][index] = _stable_mean(absolute)
                features[f"g10_pair_{name}_absolute_difference_max"][index] = float(np.max(absolute))
            for name, matrix in (
                ("rectangle", pair["rectangle_overlap"]),
                ("axis", pair["axis_overlap"]),
                ("contact", pair["contact_overlap"]),
            ):
                values = matrix[index, peers]
                features[f"g10_pair_{name}_overlap_mean"][index] = _stable_mean(values)
                features[f"g10_pair_{name}_overlap_max"][index] = float(np.max(values))
            peak_values = pair["same_peak"][index, peers]
            features["g10_pair_same_local_peak_proxy_mean"][index] = _stable_mean(peak_values)
            features["g10_pair_same_local_peak_proxy_max"][index] = float(np.max(peak_values))
            features["g10_pair_same_local_peak_proxy_neighbour_fraction"][index] = _stable_mean(peak_values >= 0.5)

        top1_sources = (
            pair["center_fraction"],
            pair["angle_fraction"],
            pair["width_fraction"],
            pair["q_delta"],
            pair["quality_delta"],
            pair["mask_delta"],
            pair["rectangle_overlap"],
            pair["axis_overlap"],
            pair["contact_overlap"],
            pair["same_peak"],
        )
        for name, matrix in zip(_TOP1_FEATURE_NAMES, top1_sources, strict=True):
            features[name][index] = matrix[index, top1_index]

    upper = np.triu_indices(len(valid_indices), k=1)
    pair_values = {
        name: matrix[np.ix_(valid_indices, valid_indices)][upper]
        for name, matrix in pair.items()
    }
    set_values = {
        "g10_set_valid_fraction": len(valid_indices) / max(count, 1),
        "g10_set_q_entropy": _normalized_entropy(q_valid),
        "g10_set_quality_entropy": _normalized_entropy(quality_valid),
        "g10_set_mask_support_entropy": _normalized_entropy(mask_valid),
        "g10_set_q_disagreement": _stable_std(q_valid),
        "g10_set_quality_disagreement": _stable_std(quality_valid),
        "g10_set_mask_support_disagreement": _stable_std(mask_valid),
        "g10_set_head_disagreement": _stable_mean(head_disagreement[valid]),
        "g10_set_center_diversity": _stable_mean(pair_values["center_fraction"]),
        "g10_set_axial_angle_diversity": _stable_mean(pair_values["angle_fraction"]),
        "g10_set_width_diversity": _stable_mean(pair_values["width_fraction"]),
        "g10_set_rectangle_diversity": _stable_mean(1.0 - pair_values["rectangle_overlap"]),
        "g10_set_axis_diversity": _stable_mean(1.0 - pair_values["axis_overlap"]),
        "g10_set_contact_diversity": _stable_mean(1.0 - pair_values["contact_overlap"]),
        "g10_set_same_local_peak_density": _stable_mean(pair_values["same_peak"]),
    }
    for name, value in set_values.items():
        features[name][valid] = value

    result = np.column_stack([features[name] for name in G10_FEATURE_NAMES]).astype(np.float32)
    result[~valid] = 0.0
    if result.shape != (count, G10_FEATURE_DIM):
        raise AssertionError(f"G10 feature shape changed unexpectedly: {result.shape}")
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite candidate relation feature")
    return G10_FEATURE_NAMES, result


__all__ = (
    "G10_FEATURE_DIM",
    "G10_FEATURE_NAMES",
    "extract_candidate_relation_features",
)
