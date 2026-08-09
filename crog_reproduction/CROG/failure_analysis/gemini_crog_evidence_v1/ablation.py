from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from .evidence import angle_vector_magnitude, decode_angle_map
from .renderer import DISPLAY_IDS, _draw_candidates
from .security import assert_display_mapping, assert_no_gt_leak


AblationProtocol = Literal["A0", "A1", "A3"]
ABLATION_RENDERER_VERSION = "crog-gemini-evidence-ablation-v1"


def _candidate_by_display(
    candidates: list[dict[str, Any]], mapping: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    by_id = {str(item["candidate_id"]): item for item in candidates}
    assert_display_mapping(mapping, candidate_ids=by_id)
    return [(display, by_id[mapping["display_to_candidate"][display]]) for display in DISPLAY_IDS]


def metadata_for_ablation(
    *,
    protocol: AblationProtocol,
    referring_expression_block: str,
    candidates: list[dict[str, Any]],
    candidate_evidence: list[dict[str, Any]],
    mapping: dict[str, Any],
    original_q_top1_candidate_id: str,
) -> str:
    """Build the strict modality-removal contracts without changing the system prompt."""

    if protocol not in {"A0", "A1", "A3"}:
        raise ValueError(f"unsupported development protocol: {protocol}")
    ordered = _candidate_by_display(candidates, mapping)
    evidence_by_id = {str(item["candidate_id"]): item for item in candidate_evidence}
    lines = [
        referring_expression_block,
        "",
        f"<development_ablation_protocol>{protocol}</development_ablation_protocol>",
    ]
    if protocol == "A0":
        lines.extend(
            (
                "Only RGB and the five frozen candidate rectangles are supplied in this ablation.",
                "No CROG M/Q/angle/W map or numeric evidence is supplied.",
                "<candidates>",
            )
        )
        lines.extend(f"{display}: frozen_candidate" for display, _ in ordered)
    elif protocol == "A1":
        lines.extend(
            (
                "Only RGB, frozen candidates, and the original q prior are supplied in this ablation.",
                "No CROG M/angle/W map or candidate evidence is supplied.",
                "<original_q_top1_display_id>",
                mapping["candidate_to_display"][str(original_q_top1_candidate_id)],
                "</original_q_top1_display_id>",
                "<candidates>",
            )
        )
        for display, candidate in ordered:
            evidence = evidence_by_id[str(candidate["candidate_id"])]
            lines.extend(
                (
                    f"{display}:",
                    f"  original_q_rank: {int(evidence['original_q_rank']) + 1}",
                    f"  q_probability: {float(evidence['q_probability_at_center']):.6f}",
                )
            )
    else:
        lines.extend(
            (
                "CROG M, angle, W, and non-Q candidate evidence are supplied.",
                "The original q prior and Q-map are deliberately withheld.",
                "<candidates>",
            )
        )
        for display, candidate in ordered:
            evidence = evidence_by_id[str(candidate["candidate_id"])]
            lines.extend(
                (
                    f"{display}:",
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
    if protocol == "A0" and any(token in result for token in ("q_probability", "mask_", "angle_", "width_")):
        raise AssertionError("A0 metadata contains a removed modality")
    if protocol == "A3" and any(token in result for token in ("q_probability", "q_rank", "q_peak")):
        raise AssertionError("A3 metadata contains q evidence")
    return result


def render_ablation_board(
    *,
    protocol: AblationProtocol,
    rgb: np.ndarray,
    candidates: list[dict[str, Any]],
    candidate_evidence: list[dict[str, Any]],
    mapping: dict[str, Any],
    sample_id: str,
    output_path: str | Path,
    mask_probability: np.ndarray | None = None,
    sin_2theta: np.ndarray | None = None,
    cos_2theta: np.ndarray | None = None,
    width_probability: np.ndarray | None = None,
) -> dict[str, Any]:
    if protocol not in {"A0", "A1", "A3"}:
        raise ValueError(f"unsupported development protocol: {protocol}")
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")
    if len(candidates) != 5 or len(candidate_evidence) != 5:
        raise ValueError("ablation renderer requires exactly five candidates")
    ordered = _candidate_by_display(candidates, mapping)
    evidence_by_id = {str(item["candidate_id"]): item for item in candidate_evidence}
    if protocol == "A3":
        required = (mask_probability, sin_2theta, cos_2theta, width_probability)
        if any(value is None or np.asarray(value).shape != image.shape[:2] for value in required):
            raise ValueError("A3 requires M/sin/cos/W maps matching the RGB shape")

    fig = plt.figure(figsize=(20, 20), dpi=110, constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_rgb = fig.add_subplot(grid[0, 0])
    ax_rgb.imshow(image)
    _draw_candidates(ax_rgb, candidates, mapping, rectangles=True)
    ax_rgb.set_title("RGB + frozen Top-5", fontsize=14)
    panels = ["rgb", "candidate_cards"]

    if protocol in {"A0", "A1"}:
        ax_context = fig.add_subplot(grid[0, 1])
        ax_context.imshow(image)
        _draw_candidates(ax_context, candidates, mapping, rectangles=False)
        ax_context.set_title(
            "Visual-only candidate centres" if protocol == "A0" else "Visual + original q prior",
            fontsize=14,
        )
        ax_note = fig.add_subplot(grid[1, :])
        ax_note.axis("off")
        rows = []
        for display, candidate in ordered:
            if protocol == "A0":
                rows.append([display, "frozen rectangle", "withheld", "withheld", "withheld", "withheld"])
            else:
                evidence = evidence_by_id[str(candidate["candidate_id"])]
                rows.append(
                    [
                        display,
                        "frozen rectangle",
                        str(int(evidence["original_q_rank"]) + 1),
                        f"{float(evidence['q_probability_at_center']):.6f}",
                        "withheld",
                        "withheld",
                    ]
                )
        table = ax_note.table(
            cellText=rows,
            colLabels=("ID", "geometry", "q rank", "q score", "M/angle/W", "GT"),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.0, 2.0)
        panels.append("q_prior_table" if protocol == "A1" else "modality_withholding_table")
    else:
        ax_m = fig.add_subplot(grid[0, 1])
        m_plot = ax_m.imshow(mask_probability, vmin=0.0, vmax=1.0, cmap="magma")
        _draw_candidates(ax_m, candidates, mapping, rectangles=False)
        ax_m.set_title("CROG predicted target probability M", fontsize=14)
        fig.colorbar(m_plot, ax=ax_m, fraction=0.035, pad=0.01)
        ax_angle = fig.add_subplot(grid[1, 0])
        angles = decode_angle_map(np.asarray(sin_2theta), np.asarray(cos_2theta))
        magnitude = angle_vector_magnitude(np.asarray(sin_2theta), np.asarray(cos_2theta))
        hue = (angles + 90.0) / 180.0
        confidence = np.clip(magnitude / max(float(np.percentile(magnitude, 95)), 1e-6), 0.0, 1.0)
        angle_rgb = hsv_to_rgb(np.stack((hue, np.ones_like(hue), np.ones_like(hue)), axis=-1))
        ax_angle.imshow(angle_rgb * confidence[..., None] + (1.0 - confidence[..., None]) * 0.18)
        _draw_candidates(ax_angle, candidates, mapping, rectangles=False)
        ax_angle.set_title("CROG predicted angle evidence", fontsize=14)
        ax_w = fig.add_subplot(grid[1, 1])
        w_plot = ax_w.imshow(np.asarray(width_probability) * 100.0, vmin=0.0, vmax=100.0, cmap="cividis")
        _draw_candidates(ax_w, candidates, mapping, rectangles=False)
        ax_w.set_title("CROG predicted width W — Q withheld", fontsize=14)
        fig.colorbar(w_plot, ax=ax_w, fraction=0.035, pad=0.01)
        panels.extend(("predicted_m", "predicted_angle", "predicted_width"))

    for axis in fig.axes:
        if hasattr(axis, "set_xticks") and axis.has_data():
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        f"Development ablation {protocol} — frozen candidates; no ground truth",
        fontsize=16,
        weight="bold",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="png", facecolor="white")
    plt.close(fig)
    image_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "renderer_version": ABLATION_RENDERER_VERSION,
        "protocol": protocol,
        "sample_id": str(sample_id),
        "image_sha256": image_sha,
        "width_px": 2200,
        "height_px": 2200,
        "panels": panels,
        "q_map_included": False,
        "evaluation_overlay_included": False,
        "candidate_mapping": mapping,
    }
    assert_no_gt_leak(manifest)
    return manifest


def ablation_renderer_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
