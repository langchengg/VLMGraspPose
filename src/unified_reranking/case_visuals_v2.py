"""Presentation-grade, read-only qualitative case visualizations.

The functions in this module consume already-frozen candidates and evaluator
diagnostics.  They never select, train, or mutate a formal experiment.
"""

from __future__ import annotations

import html
import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle


ROUTES = ("crog", "g1", "c1")
DISPLAY_NAMES = {
    "crog": "CROG",
    "g1": "HiFi-CS → G1",
    "c1": "HiFi-CS → C1",
}
COLORS = {
    "candidate": "#A7ADB4",
    "native": "#00A6D6",
    "challenger": "#E69F00",
    "final": "#CC79A7",
    "gt": "#0057B8",
    "gt_only": "#009E73",
    "pred_only": "#D55E00",
    "overlap": "#F0E442",
    "pass": "#0072B2",
    "fail": "#D55E00",
    "borderline": "#E69F00",
    "ink": "#15202B",
    "muted": "#55616F",
    "panel": "#F4F6F8",
}
ALL_CANDIDATE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#332288",
    "#44AA99",
    "#999933",
)


def rle_decode(record: Mapping[str, Any]) -> np.ndarray:
    """Decode the frozen CROG alternating-run mask format."""

    size = tuple(map(int, record["size"]))
    counts = list(map(int, record["counts"]))
    values = np.empty(sum(counts), dtype=np.uint8)
    cursor = 0
    value = int(record.get("start_value", 0))
    for count in counts:
        values[cursor : cursor + count] = value
        cursor += count
        value = 1 - value
    if cursor != int(np.prod(size)):
        raise ValueError("invalid CROG RLE length")
    return values.reshape(size).astype(bool)


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise ValueError("mask shapes differ")
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else float(np.count_nonzero(a & b) / union)


def grounding_verdict(mask_iou: float, *, empty: bool = False) -> str:
    if empty or not math.isfinite(float(mask_iou)) or float(mask_iou) < 0.25:
        return "FAIL"
    if float(mask_iou) < 0.50:
        return "BORDERLINE"
    return "PASS"


def offline_failure_reason(
    iou: float, angle_error_deg: float, *, valid: bool = True
) -> str:
    if not valid:
        return "invalid"
    iou_bad = not math.isfinite(float(iou)) or float(iou) <= 0.25
    angle_bad = (
        not math.isfinite(float(angle_error_deg))
        or float(angle_error_deg) > 30.0
    )
    if iou_bad and angle_bad:
        return "IoU + angle"
    if iou_bad:
        return "IoU"
    if angle_bad:
        return "angle"
    return "none"


def candidate_generation_verdict(
    *, candidate_count: int, top5_positive: bool, full_pool_positive: bool
) -> str:
    if int(candidate_count) == 0:
        return "NO OUTPUT"
    if bool(top5_positive):
        return "PASS"
    if bool(full_pool_positive):
        return "PARTIAL"
    return "FAIL"


def native_ranking_verdict(
    *, native_correct: bool, full_pool_positive: bool
) -> str:
    if bool(native_correct):
        return "PASS"
    if bool(full_pool_positive):
        return "FAIL"
    return "N.A."


def reranker_verdict(
    *, native_correct: bool, final_correct: bool, native_id: str, final_id: str
) -> str:
    if str(native_id) == str(final_id):
        return "NO SWITCH"
    if not native_correct and final_correct:
        return "RECOVERED"
    if native_correct and not final_correct:
        return "HARMFUL"
    if native_correct and final_correct:
        return "CORRECT→CORRECT"
    return "WRONG→WRONG"


def gate_verdict(
    *,
    native_correct: bool,
    ungated_correct: bool,
    final_correct: bool,
    native_id: str,
    ungated_id: str,
    final_id: str,
) -> str:
    accepted = str(final_id) == str(ungated_id) and str(final_id) != str(native_id)
    rejected = str(final_id) == str(native_id) and str(ungated_id) != str(native_id)
    if accepted and not native_correct and final_correct:
        return "GOOD ACCEPT"
    if accepted and native_correct and not final_correct:
        return "BAD ACCEPT"
    if rejected and native_correct and not ungated_correct:
        return "GOOD REJECT"
    if rejected and not native_correct and ungated_correct:
        return "MISSED RECOVERY"
    return "NEUTRAL"


def earliest_observable_issue(row: Mapping[str, Any]) -> str:
    """Apply the frozen operational diagnosis ordering from the prompt."""

    route = str(row["route"]).lower()
    grounding = str(row.get("grounding_verdict", "FAIL"))
    bridge = str(row.get("bridge_category", ""))
    generation = str(row.get("candidate_generation_verdict", "FAIL"))
    native = str(row.get("native_ranking_verdict", "FAIL"))
    reranker = str(row.get("reranker_verdict", "WRONG→WRONG"))
    gate = str(row.get("gate_verdict", "NEUTRAL"))
    if route in {"g1", "c1"} and bridge == "grounding_limited":
        return "visual grounding"
    if route != "crog" and grounding == "FAIL" and generation in {"FAIL", "NO OUTPUT"}:
        return "Indeterminate from available artifacts"
    if generation in {"FAIL", "NO OUTPUT", "PARTIAL"}:
        if route == "crog" and grounding == "FAIL":
            return "Indeterminate from available artifacts"
        return "candidate generation"
    if native == "FAIL":
        return "native ranking"
    if reranker == "HARMFUL" or gate == "GOOD REJECT":
        return "learned reranker"
    if gate == "MISSED RECOVERY":
        return "conservative gate"
    if bool(row.get("final_correct", row.get("gated_correct", False))):
        return "No observable failure under the offline criterion"
    return "Indeterminate from available artifacts"


def rectangle_points(row: Mapping[str, Any]) -> np.ndarray:
    return cv2.boxPoints(
        (
            (float(row["cx_px"]), float(row["cy_px"])),
            (float(row["width_px"]), float(row["height_px"])),
            -float(row["theta_deg"]),
        )
    ).astype(float)


def gt_geometry(corners: Sequence[Sequence[float]]) -> dict[str, float]:
    value = np.asarray(corners, dtype=float)
    if value.shape != (4, 2) or not np.all(np.isfinite(value)):
        raise ValueError("GT corners must be finite 4x2")
    centre = 0.5 * (value[0] + value[2])
    jaw = value[3] - value[0]
    width = min(float(np.linalg.norm(jaw)), 100.0)
    raw = math.degrees(math.atan2(float(jaw[0]), float(jaw[1])))
    theta = raw - 90.0 if raw > 0.0 else raw + 90.0
    theta = (theta + 90.0) % 180.0 - 90.0
    return {
        "cx_px": float(centre[0]),
        "cy_px": float(centre[1]),
        "theta_deg": float(theta),
        "width_px": width,
        "height_px": 20.0,
    }


def crop_extent(
    *,
    image_shape: tuple[int, int],
    target_bbox: Sequence[float],
    candidates: pd.DataFrame,
    gt_rectangles: Sequence[Sequence[Sequence[float]]],
    padding_fraction: float = 0.20,
) -> tuple[float, float, float, float]:
    """Return one context crop shared by every panel in a case."""

    height, width = map(int, image_shape)
    x, y, w, h = map(float, target_bbox)
    points = [[x, y], [x + w, y + h]]
    if not candidates.empty:
        for record in candidates.to_dict("records"):
            points.extend(rectangle_points(record).tolist())
    for rectangle in gt_rectangles:
        array = np.asarray(rectangle, dtype=float)
        if array.shape == (4, 2):
            points.extend(array.tolist())
    values = np.asarray(points, dtype=float)
    x0, y0 = values.min(axis=0)
    x1, y1 = values.max(axis=0)
    span = max(x1 - x0, y1 - y0, 40.0)
    pad = padding_fraction * span
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(float(width), x1 + pad), min(float(height), y1 + pad)
    # Preserve a readable 4:3 evidence crop while retaining the union.
    desired = 4.0 / 3.0
    current = (x1 - x0) / max(y1 - y0, 1.0)
    if current < desired:
        grow = ((y1 - y0) * desired - (x1 - x0)) / 2.0
        x0, x1 = max(0.0, x0 - grow), min(float(width), x1 + grow)
    else:
        grow = ((x1 - x0) / desired - (y1 - y0)) / 2.0
        y0, y1 = max(0.0, y0 - grow), min(float(height), y1 + grow)
    return x0, y0, x1, y1


def _show_crop(ax: plt.Axes, image: np.ndarray, crop: Sequence[float]) -> None:
    ax.imshow(image)
    x0, y0, x1, y1 = map(float, crop)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D3D8DE")
        spine.set_linewidth(0.8)


def _mask_overlay(rgb: np.ndarray, mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    result = np.asarray(rgb, dtype=float).copy()
    shade = np.asarray(tuple(int(color[index : index + 2], 16) for index in (1, 3, 5)))
    active = np.asarray(mask, bool)
    result[active] = (1.0 - alpha) * result[active] + alpha * shade
    return np.clip(result, 0, 255).astype(np.uint8)


def _mask_error_overlay(
    rgb: np.ndarray, gt_mask: np.ndarray, predicted_mask: np.ndarray
) -> np.ndarray:
    result = np.asarray(rgb, dtype=float).copy()
    gt = np.asarray(gt_mask, bool)
    pred = np.asarray(predicted_mask, bool)
    regions = {
        "gt_only": gt & ~pred,
        "pred_only": pred & ~gt,
        "overlap": gt & pred,
    }
    for name, active in regions.items():
        color = COLORS[name]
        shade = np.asarray(
            tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
        )
        result[active] = 0.40 * result[active] + 0.60 * shade
    return np.clip(result, 0, 255).astype(np.uint8)


def _draw_rectangle(
    ax: plt.Axes,
    row: Mapping[str, Any],
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 3,
    outlined: bool = False,
) -> None:
    patch = Polygon(
        rectangle_points(row),
        closed=True,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    if outlined:
        patch.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=linewidth + 2.8,
                    foreground="white",
                    alpha=0.95,
                ),
                path_effects.Normal(),
            ]
        )
    ax.add_patch(patch)


def _panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", fontsize=15.5, weight="bold", pad=3)
def _candidate_row(candidates: pd.DataFrame, candidate_id: Any) -> dict[str, Any] | None:
    match = candidates[candidates["candidate_id"].astype(str) == str(candidate_id)]
    return None if match.empty else match.iloc[0].to_dict()


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N.A."
    return f"{number:.{digits}f}" if math.isfinite(number) else "N.A."


def _diagnosis_rows(sample: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    grounding_label = (
        "Grounding-output evidence"
        if str(sample["route"]).lower() == "crog"
        else "Visual grounding"
    )
    return [
        (
            grounding_label,
            str(sample["grounding_verdict"]),
            f"mask IoU {_fmt(sample['mask_iou'])}",
        ),
        (
            "Candidate generation",
            str(sample["candidate_generation_verdict"]),
            f"Top-5 +{int(sample['positive_count_top5'])}; All +{int(sample['positive_count_all'])}",
        ),
        (
            "Native Top-1 ranking",
            str(sample["native_ranking_verdict"]),
            f"first positive rank {sample.get('first_positive_rank', 'N.A.')}",
        ),
        (
            "Learned reranker",
            str(sample["reranker_verdict"]),
            f"margin {_fmt(sample.get('score_margin'))}",
        ),
        (
            "Conservative gate",
            str(sample["gate_verdict"]),
            str(sample.get("gate_decision_reason", "")),
        ),
    ]


def render_route_case_board(
    *,
    sample: Mapping[str, Any],
    candidates: pd.DataFrame,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    predicted_mask: np.ndarray,
    gt_rectangles: Sequence[Sequence[Sequence[float]]],
    output_png: Path,
    output_svg: Path,
) -> None:
    """Render the required six-panel board and diagnosis/table columns."""

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "text.color": COLORS["ink"],
    })
    full = candidates.copy()
    crop = tuple(json.loads(str(sample["crop_extent_json"])))
    top5 = full[full["in_top5"].astype(bool)].sort_values("native_rank")
    final = _candidate_row(top5, sample.get("gated_candidate_id"))
    matched_index = int(sample.get("final_matched_gt_index", -1))
    matched_gt = (
        rectangle_points(gt_geometry(gt_rectangles[matched_index]))
        if 0 <= matched_index < len(gt_rectangles)
        else None
    )
    fig = plt.figure(figsize=(40.0 / 3.0, 7.5), dpi=180)
    header_text = (
        f"{DISPLAY_NAMES[str(sample['route'])]} | "
        f"{str(sample['presentation_outcome']).replace('_', ' ').upper()} | "
        f"Sample {sample['sample_id']}"
    )
    header_size = 20.5 if len(header_text) <= 78 else 17.5
    fig.text(
        0.018,
        0.985,
        header_text,
        fontsize=header_size,
        weight="bold",
        va="top",
    )
    fig.text(
        0.018,
        0.935,
        f"Language prompt: \u201c{str(sample.get('language', ''))}\u201d",
        fontsize=16,
        weight="semibold",
        va="top",
    )
    left, right, bottom, top = 0.018, 0.785, 0.285, 0.835
    gap_x, gap_y = 0.011, 0.050
    panel_w = (right - left - 2 * gap_x) / 3
    panel_h = (top - bottom - gap_y) / 2
    axes: list[plt.Axes] = []
    for row in range(2):
        for col in range(3):
            y = top - (row + 1) * panel_h - row * gap_y
            axes.append(fig.add_axes([left + col * (panel_w + gap_x), y, panel_w, panel_h]))
    # A: full scene with crop and inset.
    ax = axes[0]
    ax.imshow(rgb)
    x0, y0, x1, y1 = crop
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#FFFFFF", linewidth=2.0))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=COLORS["ink"], linewidth=0.8, linestyle="--"))
    ax.set_xticks([])
    ax.set_yticks([])
    inset = ax.inset_axes([0.58, 0.03, 0.40, 0.40])
    _show_crop(inset, rgb, crop)
    inset.set_title("target crop", fontsize=8, pad=1)
    _panel_title(ax, "A", "RGB + crop")
    # B: GT target mask.
    ax = axes[1]
    _show_crop(ax, _mask_overlay(rgb, gt_mask, COLORS["gt"], 0.55), crop)
    _panel_title(ax, "B", f"GT mask | {int(np.count_nonzero(gt_mask)):,} px")
    # C: mask error overlay.
    ax = axes[2]
    _show_crop(ax, _mask_error_overlay(rgb, gt_mask, predicted_mask), crop)
    _panel_title(
        ax,
        "C",
        f"Predicted | IoU {_fmt(sample['mask_iou'])} | {sample['grounding_verdict']}",
    )
    ax.text(
        0.01,
        0.01,
        "GT-only green  |  overlap yellow  |  prediction-only red",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 2},
        va="bottom",
    )
    # D: all frozen candidates.
    ax = axes[3]
    _show_crop(ax, rgb, crop)
    for record in full.sort_values(["native_rank", "candidate_id"]).to_dict(
        "records"
    ):
        rank = int(record["native_rank"])
        color = ALL_CANDIDATE_COLORS[(rank - 1) % len(ALL_CANDIDATE_COLORS)]
        _draw_rectangle(
            ax,
            record,
            color=color,
            linewidth=2.2,
            alpha=1.0,
            zorder=4 + rank,
            outlined=True,
        )
        label = ax.text(
            float(record["cx_px"]),
            float(record["cy_px"]),
            f"r{rank}",
            color=color,
            fontsize=9.2,
            weight="bold",
            ha="center",
            va="center",
            zorder=20 + rank,
        )
        label.set_path_effects(
            [path_effects.Stroke(linewidth=3.2, foreground="white"), path_effects.Normal()]
        )
    _panel_title(ax, "D", f"All candidates | n={len(full)}")
    # E: Top-5 and original Top-1.
    ax = axes[4]
    _show_crop(ax, rgb, crop)
    rank_colors = [COLORS["native"], "#7B8794", "#8D99AE", "#A5ADB8", "#BCC2CA"]
    for record in top5.to_dict("records"):
        rank = int(record["native_rank"])
        _draw_rectangle(
            ax,
            record,
            color=rank_colors[min(rank - 1, 4)],
            linewidth=2.6 if rank == 1 else 1.1,
            alpha=1.0 if rank == 1 else 0.75,
        )
    _panel_title(ax, "E", "Top-5 | original #1")
    ax.text(
        0.01,
        0.01,
        "ranks: " + "  ".join(f"{int(r.native_rank)}={r.candidate_id}" for r in top5.itertuples(index=False)),
        transform=ax.transAxes,
        fontsize=8.8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 2},
    )
    # F: final and matched GT.
    ax = axes[5]
    _show_crop(ax, rgb, crop)
    if final is not None:
        _draw_rectangle(ax, final, color=COLORS["final"], linewidth=3.0)
    if matched_gt is not None:
        ax.add_patch(Polygon(matched_gt, fill=False, edgecolor=COLORS["gt"], linewidth=2.5, linestyle="--", zorder=4))
    no_switch = str(sample.get("native_candidate_id")) == str(sample.get("gated_candidate_id"))
    suffix = " | No candidate switch" if no_switch else ""
    final_rank = sample.get("final_native_rank")
    final_rank_label = "N.A." if pd.isna(final_rank) else str(int(float(final_rank)))
    _panel_title(ax, "F", f"Final #{final_rank_label} + matched GT{suffix}")
    # Diagnosis column.
    diag = fig.add_axes([0.80, bottom, 0.186, top - bottom])
    diag.axis("off")
    diag.text(0, 1.0, "Module diagnosis", fontsize=17, weight="bold", va="top")
    y = 0.925
    for label, verdict, fact in _diagnosis_rows(sample):
        color = COLORS["pass"] if verdict in {"PASS", "RECOVERED", "GOOD ACCEPT", "GOOD REJECT"} else COLORS["borderline"] if verdict in {"BORDERLINE", "PARTIAL", "NEUTRAL", "N.A.", "NO SWITCH", "CORRECT→CORRECT"} else COLORS["fail"]
        diag.text(0.0, y, label, fontsize=11.5, weight="bold", va="top")
        diag.text(
            0.0,
            y - 0.057,
            textwrap.fill(f"{verdict} · {fact}", 31),
            fontsize=10.3,
            weight="bold",
            color=color,
            va="top",
        )
        y -= 0.158
    issue = str(sample["earliest_observable_issue"])
    diag.text(
        0.0,
        0.015,
        textwrap.fill(f"Earliest observable issue: {issue}", 29),
        fontsize=11.8,
        weight="bold",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EAF0F6", "edgecolor": "#49657A"},
    )
    # Bottom comparison table and contribution note.
    table_ax = fig.add_axes([0.018, 0.045, 0.968, 0.185])
    table_ax.axis("off")
    rows = [
        ("candidate ID", sample.get("native_candidate_id", "NO OUTPUT"), sample.get("gated_candidate_id", "NO OUTPUT")),
        ("native rank", sample.get("native_native_rank"), sample.get("final_native_rank")),
        ("native / reranker score", f"{_fmt(sample.get('native_native_score'))} / {_fmt(sample.get('native_ensemble_score'))}", f"{_fmt(sample.get('final_native_score'))} / {_fmt(sample.get('final_ensemble_score'))}"),
        ("same-GT IoU / angle", f"{_fmt(sample.get('native_diagnostic_iou'))} / {_fmt(sample.get('native_diagnostic_angle_error_deg'),1)}°", f"{_fmt(sample.get('final_diagnostic_iou'))} / {_fmt(sample.get('final_diagnostic_angle_error_deg'),1)}°"),
        ("centre / width error", f"{_fmt(sample.get('native_center_error_px'),1)} / {_fmt(sample.get('native_width_error_px'),1)} px", f"{_fmt(sample.get('final_center_error_px'),1)} / {_fmt(sample.get('final_width_error_px'),1)} px"),
        ("offline 4-DoF criterion", "PASS" if bool(sample.get("native_correct")) else "FAIL", "PASS" if bool(sample.get("gated_correct")) else "FAIL"),
    ]
    table = table_ax.table(
        cellText=[[label, str(native_value), str(final_value)] for label, native_value, final_value in rows],
        colLabels=["quantity", "original Top-1", "improved Top-1"],
        colWidths=[0.23, 0.36, 0.36],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.06)
    for (row_index, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DCE2")
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_facecolor("#EAF0F6")
            cell.set_text_props(weight="bold")
    contribution = str(sample.get("contribution_summary", "No candidate switch."))
    fig.text(
        0.5,
        0.016,
        textwrap.shorten(
            f"Post-hoc score decomposition: {contribution} Not causal proof.",
            width=195,
            placeholder="…",
        ),
        ha="center",
        va="bottom",
        fontsize=9.4,
        color=COLORS["muted"],
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, facecolor="white")
    fig.savefig(output_svg, facecolor="white")
    plt.close(fig)


def render_cross_route_board(
    *,
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    rgb: np.ndarray,
    gt_mask: np.ndarray,
    hifi_mask: np.ndarray,
    crog_mask: np.ndarray,
    gt_rectangles: Sequence[Sequence[Sequence[float]]],
    output_png: Path,
) -> None:
    """Render one identical-crop, three-route native/final comparison."""

    ordered = samples.set_index("route").loc[list(ROUTES)].reset_index()
    first = ordered.iloc[0]
    common_gt = int(first["cross_route_gt_index"])
    gt = rectangle_points(gt_geometry(gt_rectangles[common_gt]))
    sample_candidates = candidates[
        candidates["sample_id"].astype(str) == str(first.sample_id)
    ]
    selected_ids = {
        (str(row.route), str(candidate_id))
        for row in ordered.itertuples(index=False)
        for candidate_id in (row.native_candidate_id, row.gated_candidate_id)
    }
    selected_candidates = sample_candidates[
        [
            (str(route), str(candidate_id)) in selected_ids
            for route, candidate_id in zip(
                sample_candidates["route"],
                sample_candidates["candidate_id"],
                strict=True,
            )
        ]
    ]
    crop = crop_extent(
        image_shape=rgb.shape[:2],
        target_bbox=(
            float(first.target_bbox_x),
            float(first.target_bbox_y),
            float(first.target_bbox_width),
            float(first.target_bbox_height),
        ),
        candidates=selected_candidates,
        gt_rectangles=[gt],
        padding_fraction=0.18,
    )
    fig = plt.figure(figsize=(40.0 / 3.0, 7.5), dpi=180, facecolor="white")
    fig.text(0.018, 0.985, f"Three-route comparison | Sample {first.sample_id}", fontsize=21, weight="bold", va="top")
    fig.text(0.018, 0.935, f"Language prompt: \u201c{first.language}\u201d", fontsize=15.5, weight="semibold", va="top")
    strip_x, strip_w = 0.018, 0.185
    strip_titles = ["RGB + crop", "GT target mask", "CROG predicted mask", "HiFi-CS predicted mask"]
    strip_images = [rgb, _mask_overlay(rgb, gt_mask, COLORS["gt"], 0.55), _mask_error_overlay(rgb, gt_mask, crog_mask), _mask_error_overlay(rgb, gt_mask, hifi_mask)]
    for index, (title, image) in enumerate(zip(strip_titles, strip_images, strict=True)):
        ax = fig.add_axes([strip_x, 0.69 - index * 0.165, strip_w, 0.125])
        if index == 0:
            ax.imshow(image)
            x0, y0, x1, y1 = crop
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="white", linewidth=2))
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            _show_crop(ax, image, crop)
        ax.set_title(title, fontsize=11.5, loc="left", weight="bold", pad=2)
    grid_left, col_gap = 0.245, 0.016
    col_w = (0.985 - grid_left - col_gap) / 2
    row_h = 0.180
    row_positions = (0.635, 0.370, 0.105)
    for row_index, route in enumerate(ROUTES):
        sample = ordered[ordered["route"] == route].iloc[0]
        route_candidates = candidates[(candidates["route"] == route) & (candidates["sample_id"] == sample.sample_id)]
        for col_index, (kind, candidate_id, color) in enumerate((
            ("Original Top-1", sample.native_candidate_id, COLORS["native"]),
            ("Improved final Top-1", sample.gated_candidate_id, COLORS["final"]),
        )):
            y = row_positions[row_index]
            ax = fig.add_axes([grid_left + col_index * (col_w + col_gap), y, col_w, row_h])
            _show_crop(ax, rgb, crop)
            record = _candidate_row(route_candidates, candidate_id)
            if record is not None:
                _draw_rectangle(ax, record, color=color, linewidth=3.0)
            ax.add_patch(Polygon(gt, fill=False, edgecolor=COLORS["gt"], linewidth=2.4, linestyle="--"))
            prefix = "native" if col_index == 0 else "final"
            passed = bool(sample.native_correct if col_index == 0 else sample.gated_correct)
            ax.set_title(
                f"{DISPLAY_NAMES[route]} | {kind}\nrank {sample[prefix + '_native_rank']:.0f} | IoU {sample[prefix + '_diagnostic_iou']:.3f} | angle {sample[prefix + '_diagnostic_angle_error_deg']:.1f}° | {'PASS' if passed else 'FAIL'}",
                fontsize=11.2,
                loc="left",
                weight="bold",
                color=COLORS["pass"] if passed else COLORS["fail"],
                pad=3,
            )
        fig.text(
            grid_left,
            row_positions[row_index] - 0.025,
            f"{DISPLAY_NAMES[route]} earliest observable issue: {sample.earliest_observable_issue}",
            fontsize=11.2,
            weight="bold",
            va="top",
        )
    fig.text(0.245, 0.018, "Cyan = native Top-1 | magenta = final Top-1 | dashed deep blue = one common matched GT. Offline 4-DoF criterion only.", fontsize=10.8, color=COLORS["muted"])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, facecolor="white")
    plt.close(fig)


def write_gallery_html(
    output_path: Path, cases: pd.DataFrame, cross_route: pd.DataFrame
) -> None:
    cards = []
    for row in cases.to_dict("records"):
        rel = html.escape(str(row["board_relative_path"]))
        cards.append(
            f'<article data-route="{html.escape(str(row["route"]))}" data-outcome="{html.escape(str(row["presentation_outcome"]))}">'
            f'<a href="{rel}"><img loading="lazy" src="{rel}" alt="case board"></a>'
            f'<h2>{html.escape(DISPLAY_NAMES[str(row["route"])])} · {html.escape(str(row["presentation_outcome"]))}</h2>'
            f'<p>{html.escape(str(row["sample_id"]))}<br>clarity {float(row["presentation_clarity_score"]):.3f}</p></article>'
        )
    for row in cross_route.to_dict("records"):
        rel = html.escape(str(row["board_relative_path"]))
        cards.append(
            f'<article data-route="cross" data-outcome="cross-route"><a href="{rel}"><img loading="lazy" src="{rel}" alt="three-route board"></a>'
            f'<h2>Three-route comparison</h2><p>{html.escape(str(row["sample_id"]))}</p></article>'
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Three-route qualitative audit</title>
<style>body{{font:15px system-ui;margin:24px;background:#f4f6f8;color:#15202b}}.controls{{position:sticky;top:0;background:white;padding:12px;z-index:2;border-radius:8px}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px;margin-top:18px}}article{{background:white;padding:10px;border-radius:10px;box-shadow:0 1px 5px #bac1c8}}img{{width:100%;height:auto}}button{{margin-right:7px;padding:7px 12px}}h2{{font-size:17px;margin:8px 0 3px}}</style>
<script>function f(v){{document.querySelectorAll('article').forEach(x=>x.style.display=(v==='all'||x.dataset.route===v||x.dataset.outcome===v)?'block':'none')}}</script></head><body><h1>Best three-route reranking cases</h1><p>Post-formal, read-only qualitative analysis. Colours never replace PASS/FAIL text.</p><div class="controls"><button onclick="f('all')">all</button><button onclick="f('crog')">CROG</button><button onclick="f('g1')">G1</button><button onclick="f('c1')">C1</button><button onclick="f('cross')">cross-route</button></div><main>{''.join(cards)}</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)


__all__ = [
    "ALL_CANDIDATE_COLORS",
    "COLORS",
    "DISPLAY_NAMES",
    "ROUTES",
    "binary_iou",
    "candidate_generation_verdict",
    "crop_extent",
    "earliest_observable_issue",
    "gate_verdict",
    "grounding_verdict",
    "gt_geometry",
    "native_ranking_verdict",
    "offline_failure_reason",
    "rectangle_points",
    "render_cross_route_board",
    "render_route_case_board",
    "reranker_verdict",
    "rle_decode",
    "write_gallery_html",
]
