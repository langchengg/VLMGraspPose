"""GT-free candidate features for strict-mask proposal selection."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from .depth_mask_features import depth_mask_features
from .proposal_deduplication import mask_iou
from .query_semantics import QuerySemantics
from .spatial_relation_features import location_ranks, pairwise_relation_features, relation_score


EPS = np.finfo(np.float64).eps


def _boundary(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & ~ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool)
    )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max()))


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _finite_or_nan(value: Any) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return float("nan")
    return float(value)


def _morphology(
    mask: np.ndarray, box: tuple[int, int, int, int] | None
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(mask))
    image_area = int(mask.size)
    box_area = 0 if box is None else (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    crop = (
        mask[box[1] : box[3] + 1, box[0] : box[2] + 1]
        if box is not None
        else mask[:1, :1]
    )
    labels, components = ndimage.label(crop, structure=np.ones((3, 3), dtype=bool))
    component_areas = (
        np.bincount(labels.ravel())[1:] if components else np.asarray([], dtype=np.int64)
    )
    boundary = _boundary(crop)
    perimeter = int(np.count_nonzero(boundary))
    uint8 = crop.astype(np.uint8)
    contours, hierarchy = cv2.findContours(uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = float(sum(abs(cv2.contourArea(value)) for value in contours))
    contour_perimeter = float(
        sum(cv2.arcLength(value, closed=True) for value in contours)
    )
    points = np.column_stack(np.nonzero(crop)[::-1]).astype(np.int32)
    hull = cv2.convexHull(points) if len(points) >= 3 else None
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    hull_perimeter = float(cv2.arcLength(hull, closed=True)) if hull is not None else 0.0
    hole_count = 0
    if hierarchy is not None:
        hole_count = int(np.count_nonzero(hierarchy[0, :, 3] >= 0))
    distance = ndimage.distance_transform_edt(crop)
    ridge = crop & (distance >= ndimage.maximum_filter(distance, size=3)) & (distance > 0.0)
    edge_contact = int(
        np.count_nonzero(mask[0])
        + np.count_nonzero(mask[-1])
        + np.count_nonzero(mask[:, 0])
        + np.count_nonzero(mask[:, -1])
    )
    aspect = (
        float((box[2] - box[0] + 1) / max(box[3] - box[1] + 1, 1))
        if box is not None
        else 0.0
    )
    return {
        "area_px": float(area),
        "area_fraction": _ratio(area, image_area),
        "bounding_box_fraction": _ratio(box_area, image_area),
        "fill_ratio": _ratio(area, box_area),
        "connected_component_count": float(components),
        "largest_component_ratio": (
            _ratio(float(component_areas.max(initial=0)), area) if area else 0.0
        ),
        "perimeter_px": float(perimeter),
        "compactness": _ratio(4.0 * np.pi * area, perimeter * perimeter),
        "solidity": _ratio(contour_area, hull_area),
        "convexity": _ratio(hull_perimeter, max(contour_perimeter, EPS)),
        "hole_count": float(hole_count),
        "distance_ridge_skeleton_length_px": float(np.count_nonzero(ridge)),
        "aspect_ratio": aspect,
        "image_edge_contact_fraction": _ratio(edge_contact, max(perimeter, 1)),
    }


def _probability_features(
    candidate: np.ndarray,
    hifi_mask: np.ndarray,
    probability: np.ndarray,
    box: tuple[int, int, int, int] | None,
    hifi_area: int,
    total_mass: float,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=bool)
    hifi_mask = np.asarray(hifi_mask, dtype=bool)
    if box is None:
        candidate_crop = candidate[:1, :1]
        hifi_crop = hifi_mask[:1, :1]
        probability_crop = probability[:1, :1]
    else:
        x1, y1, x2, y2 = box
        candidate_crop = candidate[y1 : y2 + 1, x1 : x2 + 1]
        hifi_crop = hifi_mask[y1 : y2 + 1, x1 : x2 + 1]
        probability_crop = probability[y1 : y2 + 1, x1 : x2 + 1]
    intersection = int(np.count_nonzero(candidate_crop & hifi_crop))
    candidate_area = int(np.count_nonzero(candidate))
    values = probability_crop[candidate_crop]
    candidate_mass = float(values.sum(dtype=np.float64))
    p = np.clip(values.astype(np.float64), 1e-7, 1.0 - 1e-7)
    entropy = float(np.mean(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))) if len(p) else 0.0
    expansion = candidate_crop & ~hifi_crop
    high_core = probability >= 0.80
    high_core_crop = probability_crop >= 0.80
    return {
        "hifi_candidate_iou": mask_iou(candidate, hifi_mask),
        "hifi_coarse_recall": _ratio(intersection, hifi_area),
        "hifi_coarse_precision": _ratio(intersection, candidate_area),
        "candidate_hifi_area_ratio": _ratio(candidate_area, hifi_area),
        "hifi_probability_mass_recall": _ratio(candidate_mass, total_mass),
        "hifi_probability_mass_precision": _ratio(candidate_mass, candidate_area),
        "hifi_probability_mean_inside": float(np.mean(values)) if len(values) else 0.0,
        "hifi_probability_median_inside": float(np.median(values)) if len(values) else 0.0,
        "hifi_probability_q10_inside": float(np.quantile(values, 0.10)) if len(values) else 0.0,
        "hifi_probability_q25_inside": float(np.quantile(values, 0.25)) if len(values) else 0.0,
        "hifi_probability_q75_inside": float(np.quantile(values, 0.75)) if len(values) else 0.0,
        "hifi_probability_q90_inside": float(np.quantile(values, 0.90)) if len(values) else 0.0,
        "hifi_probability_entropy_inside": entropy,
        "high_confidence_core_coverage": _ratio(
            np.count_nonzero(candidate_crop & high_core_crop), np.count_nonzero(high_core)
        ),
        "low_probability_expansion_fraction": _ratio(
            np.count_nonzero(expansion & (probability_crop < 0.10)),
            np.count_nonzero(expansion),
        ),
    }


def _appearance_features(
    mask: np.ndarray,
    rgb: np.ndarray,
    hsv: np.ndarray,
    lab: np.ndarray,
    gradient: np.ndarray,
    requested_colour: str | None,
    box: tuple[int, int, int, int] | None,
) -> dict[str, float | bool]:
    mask = np.asarray(mask, dtype=bool)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if box is None:
        candidate = mask[:1, :1]
        rgb_crop = rgb[:1, :1]
        hsv_crop = hsv[:1, :1]
        lab_crop = lab[:1, :1]
        gradient_crop = gradient[:1, :1]
    else:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1 - 3), max(0, y1 - 3)
        x2, y2 = min(mask.shape[1] - 1, x2 + 3), min(mask.shape[0] - 1, y2 + 3)
        candidate = mask[y1 : y2 + 1, x1 : x2 + 1]
        rgb_crop = rgb[y1 : y2 + 1, x1 : x2 + 1]
        hsv_crop = hsv[y1 : y2 + 1, x1 : x2 + 1]
        lab_crop = lab[y1 : y2 + 1, x1 : x2 + 1]
        gradient_crop = gradient[y1 : y2 + 1, x1 : x2 + 1]
    boundary = _boundary(candidate)
    outside = ndimage.binary_dilation(candidate, iterations=3) & ~candidate
    inside_rgb = rgb_crop[candidate].astype(np.float32)
    outside_rgb = rgb_crop[outside].astype(np.float32)
    features: dict[str, float | bool] = {
        "rgb_boundary_gradient_support": (
            float(np.mean(gradient_crop[boundary]) / max(float(np.quantile(gradient, 0.95)), EPS))
            if np.any(boundary)
            else 0.0
        ),
        "inside_outside_rgb_contrast": (
            float(np.linalg.norm(inside_rgb.mean(axis=0) - outside_rgb.mean(axis=0)) / 441.673)
            if len(inside_rgb) and len(outside_rgb)
            else 0.0
        ),
        "appearance_features_valid": bool(np.any(candidate)),
        "appearance_features_missing": bool(not np.any(candidate)),
    }
    for name, image in (("hsv", hsv_crop), ("lab", lab_crop)):
        values = image[candidate].astype(np.float32)
        for channel in range(3):
            features[f"{name}_mean_{channel}"] = (
                float(values[:, channel].mean()) if len(values) else float("nan")
            )
            features[f"{name}_std_{channel}"] = (
                float(values[:, channel].std()) if len(values) else float("nan")
            )
    colour_valid = bool(requested_colour and np.any(candidate))
    colour_score = float("nan")
    if colour_valid:
        colour_tokens = [value.strip() for value in requested_colour.split(" and ")]
        hue_centres = {
            "red": (0.0, 179.0),
            "orange": (15.0,),
            "yellow": (30.0,),
            "green": (60.0,),
            "blue": (110.0,),
            "pink": (165.0,),
            "brown": (12.0,),
        }
        values = hsv_crop[candidate]
        scores = []
        for token in colour_tokens:
            if token == "black":
                scores.append(float(np.mean(values[:, 2] < 70)))
            elif token == "white":
                scores.append(float(np.mean((values[:, 1] < 45) & (values[:, 2] > 150))))
            elif token in {"gray", "grey", "beige", "transparent"}:
                scores.append(float(np.mean(values[:, 1] < 80)))
            elif token in hue_centres:
                distances = [
                    np.minimum(abs(values[:, 0].astype(float) - centre), 180.0 - abs(values[:, 0].astype(float) - centre))
                    for centre in hue_centres[token]
                ]
                scores.append(float(np.mean(np.min(distances, axis=0) <= 18.0)))
        colour_score = float(np.mean(scores)) if scores else float("nan")
    features.update(
        {
            "requested_colour_match_fraction": colour_score,
            "requested_colour_feature_valid": colour_valid,
            "requested_colour_feature_missing": not colour_valid,
        }
    )
    return features


def _centroid_distance(
    left: np.ndarray,
    right_centroid: tuple[float, float] | None,
    box: tuple[int, int, int, int] | None,
) -> float:
    if box is None:
        return float("nan")
    x1, y1, x2, y2 = box
    ly, lx = np.nonzero(left[y1 : y2 + 1, x1 : x2 + 1])
    if not len(lx) or right_centroid is None:
        return float("nan")
    return float(
        np.hypot(
            lx.mean() + x1 - right_centroid[0],
            ly.mean() + y1 - right_centroid[1],
        )
    )


def _boundary_coordinates(
    mask: np.ndarray,
    box: tuple[int, int, int, int] | None = None,
    maximum: int = 2000,
) -> np.ndarray:
    if box is None:
        values = np.column_stack(np.nonzero(_boundary(mask))).astype(np.float64)
    else:
        x1, y1, x2, y2 = box
        crop = mask[y1 : y2 + 1, x1 : x2 + 1]
        values = np.column_stack(np.nonzero(_boundary(crop))).astype(np.float64)
        if len(values):
            values[:, 0] += y1
            values[:, 1] += x1
    if len(values) > maximum:
        values = values[:: int(np.ceil(len(values) / maximum))]
    return values


def _independent_provenance_count(values: Any) -> int:
    records = values if isinstance(values, list) else [values]
    keys: set[tuple[Any, ...]] = set()
    for value in records:
        if not isinstance(value, dict):
            continue
        text = value.get("text", value.get("canonical_text_prompt"))
        if text:
            keys.add(("text", str(text).strip().lower()))
            continue
        prompt_id = value.get("sample_prompt_id", value.get("prompt_id"))
        if prompt_id:
            keys.add(("prompt", str(prompt_id)))
            continue
        if value.get("transformation"):
            keys.add(("transformation", str(value["transformation"])))
            continue
        if value.get("source"):
            keys.add(
                (
                    "source",
                    str(value["source"]),
                    value.get("threshold"),
                    value.get("source_rank"),
                )
            )
            continue
        if value.get("official_interface"):
            keys.add(
                (
                    "official_interface",
                    str(value["official_interface"]),
                    value.get("source_rank"),
                )
            )
            continue
        keys.add(
            (
                "family",
                str(value.get("source_family", "unknown")),
                str(value.get("source_variant", "unknown")),
            )
        )
    return max(1, len(keys))


def extract_candidate_features(
    index: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    hifi_mask: np.ndarray,
    hifi_probability: np.ndarray,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
    semantics: QuerySemantics,
    reference_candidate_ids: set[str] | None = None,
    candidate_provenance: dict[str, Any] | None = None,
    invariant_cache: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if not len(index):
        raise ValueError("candidate index is empty")
    hifi_boundary_coordinates = _boundary_coordinates(hifi_mask)
    hifi_boundary_tree = cKDTree(hifi_boundary_coordinates) if len(hifi_boundary_coordinates) else None
    hifi_yy, hifi_xx = np.nonzero(hifi_mask)
    hifi_centroid = (
        (float(hifi_xx.mean()), float(hifi_yy.mean())) if len(hifi_xx) else None
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
    required_intrinsics = {"fx", "fy", "cx", "cy"}
    if not required_intrinsics.issubset(intrinsics):
        raise ValueError(
            f"candidate features missing intrinsics {sorted(required_intrinsics - set(intrinsics))}"
        )
    hifi_depth = depth_mask_features(hifi_mask, depth_m, intrinsics=intrinsics)
    hifi_depth_median = _finite_or_nan(hifi_depth["median_depth"])
    hifi_point_cloud_centroid = np.asarray(
        [
            hifi_depth["point_cloud_centroid_x"],
            hifi_depth["point_cloud_centroid_y"],
            hifi_depth["point_cloud_centroid_z"],
        ],
        dtype=np.float64,
    )
    hifi_area = int(np.count_nonzero(hifi_mask))
    total_hifi_mass = float(hifi_probability.sum(dtype=np.float64))
    base_rows: list[dict[str, Any]] = []
    depths: list[float | None] = []
    eligible_masks: list[np.ndarray] = []
    eligible_positions: list[int] = []
    for position, row in enumerate(index.to_dict(orient="records")):
        candidate_id = str(row["candidate_id"])
        mask = masks[candidate_id]
        cache_key = str(row.get("mask_sha256", candidate_id))
        cached = None if invariant_cache is None else invariant_cache.get(cache_key)
        if cached is None:
            box = _bbox(mask)
            candidate_boundary_coordinates = _boundary_coordinates(mask, box)
            if box is None:
                empty_intrinsics = {
                    **intrinsics,
                    "cx": float(intrinsics["cx"]),
                    "cy": float(intrinsics["cy"]),
                }
                invariant_depth = depth_mask_features(
                    mask[:1, :1], depth_m[:1, :1], intrinsics=empty_intrinsics
                )
            else:
                x1, y1, x2, y2 = box
                x1p, y1p = max(0, x1 - 2), max(0, y1 - 2)
                x2p, y2p = min(mask.shape[1] - 1, x2 + 2), min(
                    mask.shape[0] - 1, y2 + 2
                )
                invariant_depth = depth_mask_features(
                    mask[y1p : y2p + 1, x1p : x2p + 1],
                    depth_m[y1p : y2p + 1, x1p : x2p + 1],
                    intrinsics={
                        **intrinsics,
                        "cx": float(intrinsics["cx"]) - x1p,
                        "cy": float(intrinsics["cy"]) - y1p,
                    },
                )
            cached = {
                "box": box,
                "boundary_coordinates": candidate_boundary_coordinates,
                "depth": invariant_depth,
                "morphology": _morphology(mask, box),
                "appearance_by_colour": {},
            }
            if invariant_cache is not None:
                invariant_cache[cache_key] = cached
        box = cached["box"]
        candidate_boundary_coordinates = cached["boundary_coordinates"]
        depth = dict(cached["depth"])
        colour_key = semantics.target_color or ""
        appearance_by_colour = cached["appearance_by_colour"]
        if colour_key not in appearance_by_colour:
            appearance_by_colour[colour_key] = _appearance_features(
                mask,
                rgb,
                hsv,
                lab,
                gradient,
                semantics.target_color,
                box,
            )
        appearance = appearance_by_colour[colour_key]
        displacement = (
            0.5
            * (
                float(
                    np.mean(hifi_boundary_tree.query(candidate_boundary_coordinates)[0])
                )
                + float(
                    np.mean(
                        cKDTree(candidate_boundary_coordinates).query(
                            hifi_boundary_coordinates
                        )[0]
                    )
                )
            )
            if len(candidate_boundary_coordinates)
            and len(hifi_boundary_coordinates)
            and hifi_boundary_tree is not None
            else float(max(mask.shape))
        )
        candidate_point_cloud_centroid = np.asarray(
            [
                depth["point_cloud_centroid_x"],
                depth["point_cloud_centroid_y"],
                depth["point_cloud_centroid_z"],
            ],
            dtype=np.float64,
        )
        depth["reference_centroid_3d_distance"] = (
            float(np.linalg.norm(candidate_point_cloud_centroid - hifi_point_cloud_centroid))
            if np.isfinite(candidate_point_cloud_centroid).all()
            and np.isfinite(hifi_point_cloud_centroid).all()
            else float("nan")
        )
        depth_median = _finite_or_nan(depth["median_depth"])
        prompt_points = json.loads(str(row.get("prompt_points_json", "[]")))
        point_hits = [bool(mask[int(y), int(x)]) for x, y, label in prompt_points if int(label) == 1]
        negative_hits = [bool(mask[int(y), int(x)]) for x, y, label in prompt_points if int(label) == 0]
        record: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "candidate_id": candidate_id,
            "source_family": str(row["source_family"]),
            "source_variant": str(row["source_variant"]),
            "eligible_final": bool(row["eligible_final"]),
            "sam_score": _finite_or_nan(row.get("sam_score")),
            "sam_score_valid": bool(not pd.isna(row.get("sam_score"))),
            "sam_score_missing": bool(pd.isna(row.get("sam_score"))),
            "presence_score": _finite_or_nan(row.get("presence_score")),
            "presence_score_valid": bool(not pd.isna(row.get("presence_score"))),
            "presence_score_missing": bool(pd.isna(row.get("presence_score"))),
            "mask_quality_score": _finite_or_nan(row.get("mask_quality_score")),
            "mask_threshold": _finite_or_nan(row.get("mask_threshold")),
            "instance_threshold": _finite_or_nan(row.get("instance_threshold")),
            "source_rank": _finite_or_nan(row.get("source_rank")),
            "source_consensus_count": float(
                _independent_provenance_count(
                    (candidate_provenance or {}).get(candidate_id, [])
                )
            ),
            "raw_provenance_count": float(row.get("provenance_count", 1)),
            "positive_prompt_point_inclusion": float(np.mean(point_hits)) if point_hits else float("nan"),
            "negative_prompt_point_violation": float(np.mean(negative_hits)) if negative_hits else float("nan"),
            "query_type": semantics.query_type,
            "target_category": semantics.target_category,
            "absolute_location_type": semantics.absolute_location,
            "relation_type": semantics.pairwise_relation,
            "parser_confidence": semantics.parser_confidence,
            "has_relation": semantics.pairwise_relation is not None,
            "has_absolute_location": semantics.absolute_location is not None,
            "has_colour": semantics.target_color is not None,
            "boundary_displacement_px": displacement,
            "hifi_candidate_centroid_distance_px": _centroid_distance(
                mask, hifi_centroid, box
            ),
            "depth_median_difference_from_hifi": (
                depth_median - hifi_depth_median
                if np.isfinite(depth_median) and np.isfinite(hifi_depth_median)
                else float("nan")
            ),
            "clip_full_query_similarity": float("nan"),
            "clip_target_category_similarity": float("nan"),
            "clip_target_attribute_similarity": float("nan"),
            "clip_box_crop_similarity": float("nan"),
            "clip_candidate_crop_background_contrast": float("nan"),
            "clip_features_valid": False,
            "clip_features_missing": True,
            **_probability_features(
                mask,
                hifi_mask,
                hifi_probability,
                box,
                hifi_area,
                total_hifi_mass,
            ),
            **cached["morphology"],
            **appearance,
            **depth,
        }
        base_rows.append(record)
        depths.append(depth_median if np.isfinite(depth_median) else None)
        if bool(row["eligible_final"]):
            eligible_positions.append(position)
            eligible_masks.append(mask)

    locations = location_ranks(eligible_masks, [depths[pos] for pos in eligible_positions], hifi_mask.shape)
    for position, values in zip(eligible_positions, locations, strict=True):
        base_rows[position].update(values)
    location_keys = list(locations[0]) if locations else []
    for position in set(range(len(base_rows))) - set(eligible_positions):
        base_rows[position].update({key: float("nan") for key in location_keys})

    reference_indices = [
        pos
        for pos, row in enumerate(base_rows)
        if row["source_family"].startswith("REFERENCE_")
        or row["candidate_id"] in (reference_candidate_ids or set())
    ]
    reference_indices.sort(
        key=lambda pos: (
            -(base_rows[pos]["sam_score"] if np.isfinite(base_rows[pos]["sam_score"]) else -1.0),
            base_rows[pos]["candidate_id"],
        )
    )
    reference_indices = reference_indices[:3]
    for position, record in enumerate(base_rows):
        relation_values: list[tuple[float, int, dict[str, float]]] = []
        if semantics.pairwise_relation and record["eligible_final"]:
            for reference_position in reference_indices:
                if (
                    base_rows[reference_position]["candidate_id"]
                    == record["candidate_id"]
                ):
                    continue
                features = pairwise_relation_features(
                    masks[record["candidate_id"]],
                    masks[base_rows[reference_position]["candidate_id"]],
                    target_depth=depths[position],
                    reference_depth=depths[reference_position],
                    image_shape=hifi_mask.shape,
                )
                relation_values.append(
                    (relation_score(features, semantics.pairwise_relation), reference_position, features)
                )
        if relation_values:
            relation_values.sort(key=lambda value: (-value[0], base_rows[value[1]]["candidate_id"]))
            scores = np.asarray([item[0] for item in relation_values], dtype=np.float64)
            weights = np.exp(scores - scores.max())
            best_score, best_position, best_features = relation_values[0]
            record.update(best_features)
            record.update(
                {
                    "relation_max_consistency": float(best_score),
                    "relation_mean_consistency": float(scores.mean()),
                    "relation_softmax_consistency": float(np.sum(scores * weights) / weights.sum()),
                    "relation_ambiguity_count": float(np.count_nonzero(scores >= best_score - 0.05)),
                    "best_reference_candidate_id": base_rows[best_position]["candidate_id"],
                    "relation_features_valid": True,
                    "relation_features_missing": False,
                }
            )
        else:
            record.update(
                {
                    "relation_max_consistency": float("nan"),
                    "relation_mean_consistency": float("nan"),
                    "relation_softmax_consistency": float("nan"),
                    "relation_ambiguity_count": float("nan"),
                    "best_reference_candidate_id": None,
                    "relation_features_valid": False,
                    "relation_features_missing": True,
                }
            )

    frame = pd.DataFrame(base_rows)
    eligible = frame["eligible_final"]
    for column, rank_name, ascending in (
        ("area_px", "candidate_area_rank", False),
        ("sam_score", "candidate_sam_score_rank", False),
        ("hifi_candidate_iou", "candidate_hifi_overlap_rank", False),
        ("depth_reliability", "candidate_depth_consistency_rank", False),
        ("relation_max_consistency", "candidate_spatial_relation_rank", False),
    ):
        frame[rank_name] = float("nan")
        frame.loc[eligible, rank_name] = frame.loc[eligible, column].rank(
            method="min", ascending=ascending, na_option="bottom"
        )
    frame["same_category_alternative_count"] = float(max(int(eligible.sum()) - 1, 0))
    return frame


__all__ = ["extract_candidate_features"]
