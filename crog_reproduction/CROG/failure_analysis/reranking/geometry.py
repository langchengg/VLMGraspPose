import hashlib
import json
import math

import cv2
import numpy as np


def grasp_polygon(cx, cy, width_px, height_px, angle_deg):
    """Use the same OpenCV rectangle convention as CROG evaluation/plots."""
    return cv2.boxPoints(
        (
            (float(cx), float(cy)),
            (float(width_px), float(height_px)),
            -float(angle_deg),
        )
    )


def rasterize_polygon(polygon, shape):
    mask = np.zeros(tuple(shape), dtype=np.uint8)
    points = np.asarray(polygon, dtype=np.intp)
    if points.shape == (4, 2):
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def rasterize_candidate(candidate, shape):
    polygon = grasp_polygon(
        candidate["cx"],
        candidate["cy"],
        candidate["width_px"],
        candidate["height_px"],
        candidate["angle_deg"],
    )
    visible = rasterize_polygon(polygon, shape)

    pad = int(math.ceil(max(candidate["width_px"], candidate["height_px"]))) + 4
    shifted = polygon.copy()
    shifted[:, 0] += pad
    shifted[:, 1] += pad
    full_shape = (int(2 * pad + shape[0]), int(2 * pad + shape[1]))
    full_count = int(rasterize_polygon(shifted, full_shape).sum())
    return visible, max(full_count, 1), polygon


def local_coordinates(candidate, shape):
    """Return opening-axis (u) and perpendicular (v) pixel coordinates."""
    rows, cols = np.indices(tuple(shape), dtype=np.float32)
    dx = cols - float(candidate["cx"])
    dy = rows - float(candidate["cy"])
    theta = math.radians(float(candidate["angle_deg"]))
    # The repo passes -theta to OpenCV because image y increases downwards.
    ux, uy = math.cos(theta), -math.sin(theta)
    vx, vy = -uy, ux
    u = dx * ux + dy * uy
    v = dx * vx + dy * vy
    return u, v


def candidate_contact_bands(candidate, shape, thickness_ratio=0.15):
    u, v = local_coordinates(candidate, shape)
    half_width = max(float(candidate["width_px"]) / 2.0, 1.0)
    half_height = max(float(candidate["height_px"]) / 2.0, 1.0)
    band = max(1.0, half_width * float(thickness_ratio))
    within_height = np.abs(v) <= half_height
    left = within_height & (u >= -half_width) & (u <= -half_width + band)
    right = within_height & (u <= half_width) & (u >= half_width - band)
    return left, right


def candidate_jaw_bands(candidate, shape, thickness_ratio=0.15):
    return candidate_contact_bands(candidate, shape, thickness_ratio)


def angle_difference_deg(angle_a, angle_b):
    """Smallest difference for a parallel-jaw grasp (180-degree symmetry)."""
    return abs(((float(angle_a) - float(angle_b) + 90.0) % 180.0) - 90.0)


def neutral_expectation(value, reliability):
    reliability = float(np.clip(reliability, 0.0, 1.0))
    if value is None or not np.isfinite(value):
        reliability = 0.0
        value = 0.5
    return reliability * float(value) + (1.0 - reliability) * 0.5


def frozen_candidate_signature(candidates):
    signature = {}
    for candidate in candidates:
        actual_checksum = geometry_checksum(candidate)
        declared_checksum = str(candidate["candidate_checksum"])
        if actual_checksum != declared_checksum:
            raise AssertionError(
                f"candidate geometry no longer matches checksum: {candidate['candidate_id']}"
            )
        signature[str(candidate["candidate_id"])] = {
            "checksum": declared_checksum,
            "geometry": {
                key: candidate[key]
                for key in (
                    "row",
                    "col",
                    "cx",
                    "cy",
                    "angle_rad",
                    "angle_deg",
                    "width_px",
                    "height_px",
                    "polygon",
                )
            },
        }
    return signature


def assert_candidate_set_unchanged(before, after):
    before_signature = frozen_candidate_signature(before)
    after_signature = frozen_candidate_signature(after)
    if set(before_signature) != set(after_signature):
        raise AssertionError("reranker changed candidate ids")
    if before_signature != after_signature:
        raise AssertionError("reranker changed candidate geometry/checksum")


def geometry_checksum(candidate):
    keys = ("row", "col", "cx", "cy", "angle_rad", "angle_deg", "width_px", "height_px", "polygon")
    payload = {key: candidate[key] for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
