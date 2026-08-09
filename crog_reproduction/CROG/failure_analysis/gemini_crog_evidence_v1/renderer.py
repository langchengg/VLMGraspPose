from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from . import RENDERER_VERSION
from .evidence import angle_vector_magnitude, decode_angle_map
from .security import assert_no_gt_leak


DISPLAY_IDS = ("A", "B", "C", "D", "E")
DISPLAY_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#9467bd",
    "D": "#17becf",
    "E": "#bcbd22",
}


def deterministic_candidate_mapping(
    candidate_ids: list[str], sample_id: str, seed: int = 47
) -> dict[str, Any]:
    if len(candidate_ids) != 5 or len(set(candidate_ids)) != 5:
        raise ValueError("candidate mapping requires five unique stable IDs")
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "big")
    shuffled = list(candidate_ids)
    random.Random(derived_seed).shuffle(shuffled)
    display_to_candidate = dict(zip(DISPLAY_IDS, shuffled, strict=True))
    return {
        "renderer_version": RENDERER_VERSION,
        "seed": int(seed),
        "derived_seed": derived_seed,
        "display_to_candidate": display_to_candidate,
        "candidate_to_display": {value: key for key, value in display_to_candidate.items()},
    }


def reverse_display_id(display_id: str, mapping: dict[str, Any]) -> str:
    try:
        return str(mapping["display_to_candidate"][str(display_id)])
    except KeyError as exc:
        raise ValueError(f"unknown display ID: {display_id}") from exc


def _draw_candidates(
    axis,
    candidates: list[dict[str, Any]],
    mapping: dict[str, Any],
    *,
    rectangles: bool,
) -> None:
    by_id = {str(item["candidate_id"]): item for item in candidates}
    for display_id in DISPLAY_IDS:
        candidate = by_id[mapping["display_to_candidate"][display_id]]
        color = DISPLAY_COLORS[display_id]
        polygon = np.asarray(candidate["polygon"], dtype=np.float64)
        if rectangles:
            closed = np.vstack((polygon, polygon[0]))
            axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.2)
        axis.scatter([candidate["cx"]], [candidate["cy"]], color=color, s=28, marker="o")
        axis.text(
            float(candidate["cx"]) + 6,
            float(candidate["cy"]) - 6,
            display_id,
            color="white",
            fontsize=10,
            weight="bold",
            bbox={"facecolor": color, "edgecolor": "white", "pad": 1.5},
        )


def _rgb_crop(image: np.ndarray, candidate: dict[str, Any], half_extent: int) -> np.ndarray:
    x, y = int(round(candidate["cx"])), int(round(candidate["cy"]))
    padded = cv2.copyMakeBorder(
        image,
        half_extent,
        half_extent,
        half_extent,
        half_extent,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    xp, yp = x + half_extent, y + half_extent
    return padded[yp - half_extent : yp + half_extent, xp - half_extent : xp + half_extent].copy()


def _scalar_crop(array: np.ndarray, candidate: dict[str, Any], half_extent: int) -> np.ndarray:
    x, y = int(round(candidate["cx"])), int(round(candidate["cy"]))
    padded = cv2.copyMakeBorder(
        np.asarray(array, dtype=np.float32),
        half_extent,
        half_extent,
        half_extent,
        half_extent,
        cv2.BORDER_CONSTANT,
        value=0.0,
    )
    xp, yp = x + half_extent, y + half_extent
    return padded[yp - half_extent : yp + half_extent, xp - half_extent : xp + half_extent].copy()


def render_evidence_board(
    *,
    rgb: np.ndarray,
    candidates: list[dict[str, Any]],
    candidate_evidence: list[dict[str, Any]],
    mask_probability: np.ndarray,
    quality_probability: np.ndarray,
    sin_2theta: np.ndarray,
    cos_2theta: np.ndarray,
    width_probability: np.ndarray,
    sample_id: str,
    output_path: str | Path,
    mapping: dict[str, Any] | None = None,
    mask_threshold: float = 0.35,
) -> dict[str, Any]:
    request_side = {
        "sample_id": sample_id,
        "candidates": candidates,
        "candidate_evidence": candidate_evidence,
    }
    assert_no_gt_leak(request_side)
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")
    shape = image.shape[:2]
    maps = [mask_probability, quality_probability, sin_2theta, cos_2theta, width_probability]
    if any(np.asarray(value).shape != shape for value in maps):
        raise ValueError("renderer maps must match RGB shape")
    if len(candidates) != 5 or len(candidate_evidence) != 5:
        raise ValueError("renderer requires exactly five candidates/evidence cards")
    mapping = mapping or deterministic_candidate_mapping(
        [str(item["candidate_id"]) for item in candidates], sample_id
    )
    by_id = {str(item["candidate_id"]): item for item in candidates}
    evidence_by_id = {str(item["candidate_id"]): item for item in candidate_evidence}
    if set(by_id) != set(mapping["candidate_to_display"]):
        raise ValueError("mapping does not match frozen candidate set")

    fig = plt.figure(figsize=(20, 20), dpi=110, constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.12))
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    ax_rgb, ax_m, ax_q, ax_angle = axes
    ax_width = fig.add_subplot(grid[2, 0])
    cards_grid = grid[2, 1].subgridspec(1, 5, wspace=0.06)
    card_axes = [fig.add_subplot(cards_grid[0, index]) for index in range(5)]

    ax_rgb.imshow(image)
    binary = np.asarray(mask_probability) > float(mask_threshold)
    ax_rgb.contour(binary.astype(np.uint8), levels=[0.5], colors=["white"], linewidths=1.2)
    _draw_candidates(ax_rgb, candidates, mapping, rectangles=True)
    ax_rgb.set_title("A — Original RGB + CROG predicted contour + frozen Top-5", fontsize=13)

    m_plot = ax_m.imshow(mask_probability, vmin=0.0, vmax=1.0, cmap="magma")
    ax_m.contour(binary.astype(np.uint8), levels=[0.5], colors=["white"], linewidths=1.0)
    _draw_candidates(ax_m, candidates, mapping, rectangles=False)
    ax_m.set_title("B — CROG predicted target probability — not ground truth", fontsize=13)
    fig.colorbar(m_plot, ax=ax_m, fraction=0.035, pad=0.01, label="M probability")

    q_plot = ax_q.imshow(quality_probability, vmin=0.0, vmax=1.0, cmap="viridis")
    _draw_candidates(ax_q, candidates, mapping, rectangles=False)
    ax_q.set_title("C — CROG predicted grasp quality prior", fontsize=13)
    fig.colorbar(q_plot, ax=ax_q, fraction=0.035, pad=0.01, label="Q probability")

    angles = decode_angle_map(np.asarray(sin_2theta), np.asarray(cos_2theta))
    magnitude = angle_vector_magnitude(np.asarray(sin_2theta), np.asarray(cos_2theta))
    hue = (angles + 90.0) / 180.0
    saturation = np.ones_like(hue)
    value = np.ones_like(hue)
    angle_rgb = hsv_to_rgb(np.stack((hue, saturation, value), axis=-1))
    confidence = np.clip(magnitude / max(float(np.percentile(magnitude, 95)), 1e-6), 0.0, 1.0)
    angle_rgb = angle_rgb * confidence[..., None] + (1.0 - confidence[..., None]) * 0.18
    ax_angle.imshow(angle_rgb)
    _draw_candidates(ax_angle, candidates, mapping, rectangles=False)
    for candidate in candidates:
        theta = math.radians(float(candidate["angle_deg"]))
        length = 30.0
        ax_angle.arrow(
            float(candidate["cx"]) - 0.5 * length * math.cos(theta),
            float(candidate["cy"]) + 0.5 * length * math.sin(theta),
            length * math.cos(theta),
            -length * math.sin(theta),
            color="white",
            width=1.0,
            head_width=5.0,
            length_includes_head=True,
        )
    ax_angle.text(
        0.01,
        0.99,
        "Hue: axial angle −90°…+90°\nBrightness: vector magnitude",
        transform=ax_angle.transAxes,
        va="top",
        color="white",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.65, "pad": 4},
    )
    ax_angle.set_title("D — CROG predicted axial angle evidence", fontsize=13)

    width_decoded = np.asarray(width_probability) * 100.0
    w_plot = ax_width.imshow(width_decoded, vmin=0.0, vmax=100.0, cmap="cividis")
    _draw_candidates(ax_width, candidates, mapping, rectangles=False)
    for display_id, candidate_id in mapping["display_to_candidate"].items():
        candidate = by_id[candidate_id]
        ax_width.text(
            candidate["cx"] + 6,
            candidate["cy"] + 18,
            f"{float(candidate['width_px']):.1f}px",
            color="white",
            fontsize=8,
        )
    ax_width.set_title("E — CROG predicted width (0–100 px)", fontsize=13)
    fig.colorbar(w_plot, ax=ax_width, fraction=0.035, pad=0.01, label="decoded width (px)")

    half_extent = max(48, int(math.ceil(max(float(item["width_px"]) for item in candidates) * 0.9)))
    for axis, display_id in zip(card_axes, DISPLAY_IDS, strict=True):
        candidate_id = mapping["display_to_candidate"][display_id]
        candidate = by_id[candidate_id]
        evidence = evidence_by_id[candidate_id]
        crop = _rgb_crop(image, candidate, half_extent)
        crop_size = 160
        rgb_card = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        mask_card = cv2.resize(
            _scalar_crop(mask_probability, candidate, half_extent),
            (crop_size // 2, crop_size // 2),
            interpolation=cv2.INTER_LINEAR,
        )
        q_card = cv2.resize(
            _scalar_crop(quality_probability, candidate, half_extent),
            (crop_size // 2, crop_size // 2),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_color = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(np.clip(mask_card, 0, 1) * 255), cv2.COLORMAP_MAGMA),
            cv2.COLOR_BGR2RGB,
        )
        q_color = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(np.clip(q_card, 0, 1) * 255), cv2.COLORMAP_VIRIDIS),
            cv2.COLOR_BGR2RGB,
        )
        card = np.zeros((crop_size + crop_size // 2, crop_size, 3), dtype=np.uint8)
        rgb_uint8 = np.uint8(np.clip(rgb_card * 255.0 if rgb_card.max() <= 1.0 else rgb_card, 0, 255))
        card[:crop_size] = rgb_uint8
        card[crop_size:, : crop_size // 2] = mask_color
        card[crop_size:, crop_size // 2 :] = q_color
        polygon = np.asarray(candidate["polygon"], dtype=np.float64).copy()
        polygon[:, 0] -= float(candidate["cx"]) - half_extent
        polygon[:, 1] -= float(candidate["cy"]) - half_extent
        polygon *= crop_size / float(2 * half_extent)
        closed = np.vstack((polygon, polygon[0]))
        axis.imshow(card)
        axis.plot(closed[:, 0], closed[:, 1], color=DISPLAY_COLORS[display_id], linewidth=2)
        centre = crop_size / 2.0
        axis.scatter([centre], [centre], color=DISPLAY_COLORS[display_id], s=20)
        theta = math.radians(float(candidate["angle_deg"]))
        arrow_length = crop_size * 0.24
        axis.arrow(
            centre - 0.5 * arrow_length * math.cos(theta),
            centre + 0.5 * arrow_length * math.sin(theta),
            arrow_length * math.cos(theta),
            -arrow_length * math.sin(theta),
            color="white",
            width=0.8,
            head_width=5,
            length_includes_head=True,
        )
        axis.text(3, crop_size + 12, "M", color="white", fontsize=7, weight="bold")
        axis.text(crop_size // 2 + 3, crop_size + 12, "Q", color="white", fontsize=7, weight="bold")
        axis.set_title(f"{display_id}", color=DISPLAY_COLORS[display_id], weight="bold", fontsize=14)
        axis.set_xlabel(
            "\n".join(
                (
                    f"q-rank {int(evidence['original_q_rank']) + 1}  q={float(evidence['q_probability_at_center']):.3f}",
                    f"mask={float(evidence['rectangle_mask_probability_mean']):.2f}  jaws={float(evidence['min_jaw_mask_support']):.2f}",
                    f"angle={float(evidence['angle_consistency_score']):.2f}",
                    f"width={float(evidence['width_consistency_score']):.2f}",
                )
            ),
            fontsize=7.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "CROG frozen Top-5 evidence board — model predictions only; no ground truth",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.755,
        0.355,
        "F — Candidate evidence cards (identical scale)",
        ha="center",
        fontsize=12,
        weight="bold",
    )
    for axis in (ax_rgb, ax_m, ax_q, ax_angle, ax_width):
        axis.set_xlim(0, shape[1] - 1)
        axis.set_ylim(shape[0] - 1, 0)
        axis.set_xticks([])
        axis.set_yticks([])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="png", facecolor="white")
    plt.close(fig)
    image_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    layer_manifest = {
        "renderer_version": RENDERER_VERSION,
        "sample_id": str(sample_id),
        "image_sha256": image_sha,
        "width_px": 2200,
        "height_px": 2200,
        "panels": ["rgb", "predicted_m", "predicted_q", "predicted_angle", "predicted_width", "candidate_cards"],
        "evaluation_overlay_included": False,
        "candidate_mapping": mapping,
        "mask_threshold": float(mask_threshold),
    }
    assert_no_gt_leak(layer_manifest)
    return layer_manifest


def metadata_for_prompt(
    *,
    referring_expression_block: str,
    candidate_evidence: list[dict[str, Any]],
    mapping: dict[str, Any],
    original_q_top1_candidate_id: str,
) -> str:
    evidence_by_id = {str(item["candidate_id"]): item for item in candidate_evidence}
    top1_display = mapping["candidate_to_display"][str(original_q_top1_candidate_id)]
    lines = [
        referring_expression_block,
        "",
        "<original_q_top1_display_id>",
        top1_display,
        "</original_q_top1_display_id>",
        "",
        "<candidates>",
    ]
    for display_id in DISPLAY_IDS:
        evidence = evidence_by_id[mapping["display_to_candidate"][display_id]]
        lines.extend(
            (
                f"{display_id}:",
                f"  original_q_rank: {int(evidence['original_q_rank']) + 1}",
                f"  q_probability: {float(evidence['q_probability_at_center']):.6f}",
                f"  q_relative: {float(evidence['q_relative_minmax']):.4f}",
                f"  q_peak_prominence: {float(evidence['q_peak_prominence']):.4f}",
                f"  mask_center_probability: {float(evidence['mask_probability_at_center']):.4f}",
                f"  rectangle_mask_support: {float(evidence['rectangle_mask_probability_mean']):.4f}",
                f"  jaw_mask_support_min: {float(evidence['min_jaw_mask_support']):.4f}",
                f"  mask_boundary_distance_normalized: {float(evidence['signed_distance_to_predicted_mask_boundary']) / 640.0:.4f}",
                f"  angle_degrees: {float(evidence['angle_deg_periodic_180']):.2f}",
                f"  angle_confidence: {float(evidence['angle_vector_magnitude']):.4f}",
                f"  local_angle_consistency: {float(evidence['angle_consistency_score']):.4f}",
                f"  width_pixels: {float(evidence['width_px']):.2f}",
                f"  width_consistency: {float(evidence['width_consistency_score']):.4f}",
                f"  candidate_uniqueness: {float(evidence['candidate_uniqueness']):.4f}",
            )
        )
    lines.append("</candidates>")
    result = "\n".join(lines)
    assert_no_gt_leak({"metadata": result})
    return result
