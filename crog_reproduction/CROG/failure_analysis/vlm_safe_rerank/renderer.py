from __future__ import annotations

import hashlib
import io
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from failure_analysis.failure_utils import rle_to_mask
from failure_analysis.reranking.feature_extraction import load_depth_m
from failure_analysis.reranking.geometry import candidate_contact_bands, grasp_polygon

from .features import build_pair_evidence, ordered_candidates


class PerturbationVariant(str, Enum):
    ORIGINAL = "original"
    PANEL_SWAP = "panel_swap"
    NEUTRAL_COLOR_REMAP = "neutral_color_remap"
    DISPLAY_ID_RENAME = "display_id_rename"
    PANEL_SWAP_TABLE_FIXED = "panel_swap_table_fixed"
    OVERLAY_ORDER_SWAP = "overlay_order_swap"


def _rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _aligned_crop(array: np.ndarray, candidate: Mapping[str, Any], size: int = 144) -> np.ndarray:
    center = (float(candidate["cx"]), float(candidate["cy"]))
    matrix = cv2.getRotationMatrix2D(center, -float(candidate["angle_deg"]), 1.0)
    rotated = cv2.warpAffine(
        array,
        matrix,
        (array.shape[1], array.shape[0]),
        flags=cv2.INTER_NEAREST if array.ndim == 2 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return cv2.getRectSubPix(rotated, (size, size), center)


def _draw_candidate(
    image: np.ndarray,
    candidate: Mapping[str, Any],
    label: str,
    color: tuple[int, int, int],
) -> None:
    polygon = np.rint(grasp_polygon(
        candidate["cx"], candidate["cy"], candidate["width_px"],
        candidate["height_px"], candidate["angle_deg"]
    )).astype(np.int32)
    cv2.polylines(image, [polygon], True, color, 3, cv2.LINE_AA)
    center = (int(round(float(candidate["cx"]))), int(round(float(candidate["cy"]))))
    cv2.circle(image, center, 5, color, -1, cv2.LINE_AA)
    theta = np.deg2rad(float(candidate["angle_deg"]))
    axis = np.asarray([np.cos(theta), -np.sin(theta)])
    half = max(float(candidate["width_px"]) / 2.0, 5.0)
    p0 = tuple(np.rint(np.asarray(center) - half * axis).astype(int))
    p1 = tuple(np.rint(np.asarray(center) + half * axis).astype(int))
    cv2.arrowedLine(image, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.12)
    cv2.putText(image, label, (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _depth_display(depth: np.ndarray | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if depth is None:
        return np.zeros((*shape, 3), dtype=np.uint8), np.zeros(shape, dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0.0)
    scaled = np.zeros(shape, dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        scaled[valid] = np.clip((depth[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(255 - scaled, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color, (valid.astype(np.uint8) * 255)


def _table_text(
    pair: Mapping[str, Any], display_ids: Mapping[str, str],
    side_order: tuple[str, str] = ("baseline", "challenger"),
) -> str:
    fields = [
        ("rank", "original_rank"), ("q", "q"), ("delta q", "delta_q_to_baseline"),
        ("mask rect", "mask_rectangle_coverage"), ("mask axis", "mask_axis_support"),
        ("contact L", "mask_contact_support_left"), ("contact R", "mask_contact_support_right"),
        ("depth valid", "valid_depth_fraction"), ("centre depth m", "centre_depth_m"),
        ("contact dz m", "absolute_contact_depth_difference_m"),
        ("object width px", "estimated_object_width_px"),
        ("width compat", "grasp_width_compatibility"), ("clearance", "clearance_proxy"),
        ("collision", "collision_proxy"), ("reliability", "aggregate_reliability"),
    ]
    headers = [f"{side.upper()} {display_ids[side]}" for side in side_order]
    lines = [f"{'feature':<18} {headers[0]:<20} {headers[1]:<20}"]
    for label, key in fields:
        values = []
        for side in side_order:
            value = pair[side].get(key)
            values.append("missing" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value))
        lines.append(f"{label:<18} {values[0]:<20} {values[1]:<20}")
    lines.append("collision is a relative 2.5D obstacle proxy; metric 3D collision is unavailable")
    return "\n".join(lines)


def render_pairwise_board(
    feature: Mapping[str, Any],
    challenger_id: str,
    *,
    variant: PerturbationVariant = PerturbationVariant.ORIGINAL,
    output_path: str | Path | None = None,
    include_numeric: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Render GT-free pair evidence. Perturbations always receive distinct hashes."""

    pair = build_pair_evidence(feature, challenger_id)
    candidates = ordered_candidates(feature)
    by_id = {str(row["candidate_id"]): row for row in candidates}
    baseline = candidates[0]
    challenger = by_id[challenger_id]
    rgb = _rgb(feature["image_path"])
    mask = np.asarray(rle_to_mask(feature["predicted_mask_rle"]), dtype=bool)
    depth = None
    if feature.get("depth_path"):
        depth, _ = load_depth_m(feature["depth_path"], expected_shape=mask.shape)
    depth_rgb, valid = _depth_display(depth, mask.shape)

    display_ids = {"baseline": "A", "challenger": "B"}
    if variant is PerturbationVariant.DISPLAY_ID_RENAME:
        display_ids = {"baseline": "X7", "challenger": "Q2"}
    colors = {"baseline": (32, 180, 245), "challenger": (235, 170, 45)}
    if variant is PerturbationVariant.NEUTRAL_COLOR_REMAP:
        colors = {"baseline": (210, 210, 210), "challenger": (110, 110, 110)}

    overview = rgb.copy()
    overlay = rgb.copy()
    overlay[mask] = np.rint(0.55 * overlay[mask] + 0.45 * np.asarray([70, 210, 120])).astype(np.uint8)
    order = [("baseline", baseline), ("challenger", challenger)]
    if variant is PerturbationVariant.OVERLAY_ORDER_SWAP:
        order.reverse()
    for side, candidate in order:
        _draw_candidate(overview, candidate, f"{side.upper()} {display_ids[side]}", colors[side])
        _draw_candidate(overlay, candidate, f"{side.upper()} {display_ids[side]}", colors[side])

    panel_sides = ["baseline", "challenger"]
    if variant in (PerturbationVariant.PANEL_SWAP, PerturbationVariant.PANEL_SWAP_TABLE_FIXED):
        panel_sides.reverse()
    side_candidate = {"baseline": baseline, "challenger": challenger}
    fig = plt.figure(figsize=(18, 12), dpi=110)
    grid = fig.add_gridspec(4, 6, height_ratios=[1.55, 1, 1, 0.9])
    top = [(overview, "RGB + frozen baseline/challenger"), (overlay, "Predicted target mask overlay"), (depth_rgb, "Metric depth (invalid=black)")]
    for index, (image, title) in enumerate(top):
        ax = fig.add_subplot(grid[0, 2 * index:2 * index + 2])
        ax.imshow(image); ax.set_title(title); ax.axis("off")
    for column, side in enumerate(panel_sides):
        candidate = side_candidate[side]
        rgb_crop = _aligned_crop(rgb, candidate)
        mask_crop = _aligned_crop((mask.astype(np.uint8) * 255), candidate)
        depth_crop = _aligned_crop(depth_rgb, candidate)
        valid_crop = _aligned_crop(valid, candidate)
        for row, (image, title, cmap) in enumerate([
            (rgb_crop, "aligned RGB", None), (mask_crop, "predicted mask", "gray"),
            (depth_crop, "metric depth", None), (valid_crop, "depth validity", "gray")
        ]):
            ax = fig.add_subplot(grid[1 + row // 2, 3 * column + (row % 2):3 * column + (row % 2) + 1])
            ax.imshow(image, cmap=cmap, vmin=0, vmax=255); ax.set_title(f"{side.upper()} {display_ids[side]}: {title}"); ax.axis("off")
        # Explicit jaw pad display is derived from the same authoritative geometry.
        left, right = candidate_contact_bands(candidate, mask.shape)
        pads = np.zeros((*mask.shape, 3), dtype=np.uint8)
        pads[left] = colors[side]; pads[right] = colors[side]
        ax = fig.add_subplot(grid[1:3, 3 * column + 2])
        ax.imshow(_aligned_crop(pads, candidate)); ax.set_title(f"{side.upper()} jaw contact pads + sweep"); ax.axis("off")
    ax = fig.add_subplot(grid[3, :])
    ax.axis("off")
    table_order = (
        tuple(panel_sides)
        if variant is PerturbationVariant.PANEL_SWAP
        else ("baseline", "challenger")
    )
    table_text = (
        _table_text(pair, display_ids, table_order)
        if include_numeric
        else "P3 visual-only pairwise critic: deterministic numeric evidence table intentionally withheld"
    )
    ax.text(0.01, 0.98, table_text, family="monospace", fontsize=8.7, va="top")
    fig.suptitle(
        "Advisory pairwise evidence — default KEEP BASELINE\n"
        f"Expression: {feature['language_instruction']} | origin top-left; x right; y down; angle deg; θ≡θ+180°",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stream = io.BytesIO()
    fig.savefig(stream, format="png", metadata={"Software": "vlm_safe_rerank"})
    plt.close(fig)
    png = stream.getvalue()
    if output_path is not None:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(png)
    metadata = {
        "sample_id": pair["sample_id"], "baseline_candidate_id": pair["baseline"]["candidate_id"],
        "challenger_candidate_id": pair["challenger"]["candidate_id"], "variant": variant.value,
        "evidence_hash": pair["evidence_hash"], "image_sha256": hashlib.sha256(png).hexdigest(),
        "renderer_contract_hash": hashlib.sha256(
            json.dumps(
                {"renderer": "pairwise_safe_renderer_v1", "variant": variant.value}
                if include_numeric
                else {"renderer": "pairwise_visual_p3_renderer_v1", "variant": variant.value},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "display_ids": display_ids,
        "numeric_evidence_included": include_numeric,
    }
    return png, metadata
