#!/usr/bin/env python3
"""Build the Chapter 4 qualitative evidence figure from frozen real cases.

The figure is deliberately reconstructed from canonical tables, immutable RGB /
mask assets, and frozen candidate geometry.  Existing case boards are used only
as QA/selection evidence; they are not pasted into the thesis figure.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
OUT = REPO / "images" / "chapter4"
SUBMISSION_OUT = REPO / "figures" / "ch4"
sys.path.insert(0, str(REPO))

from src.unified_reranking.case_visuals_v2 import (  # noqa: E402
    binary_iou,
    rectangle_points,
    rle_decode,
)


CASE_VISUAL_HELPERS = REPO / "src" / "unified_reranking" / "case_visuals_v2.py"
VISUAL_ROOT = REPO / "runs" / "reranking_case_visuals_v2_20260810T203154Z"
CANONICAL_CASES = VISUAL_ROOT / "01_data" / "canonical_case_table.parquet"
CANONICAL_CANDIDATES = VISUAL_ROOT / "01_data" / "canonical_candidate_table.parquet"
SELECTED_AUDIT = VISUAL_ROOT / "02_selection" / "SELECTED_AUDIT_CASES.csv"
ALL_ELIGIBLE_AUDIT = VISUAL_ROOT / "02_selection" / "ALL_ELIGIBLE_CASES.csv"
CROG_FEATURES = (
    REPO
    / "crog_reproduction"
    / "CROG"
    / "failure_analysis"
    / "reranking_outputs"
    / "full_test_17749_v1"
    / "features.jsonl"
)
D1_ROOT = REPO / "runs" / "fair_d1_reranking_extension_20260811T145515Z"
D1_MANIFEST = D1_ROOT / "01_manifests" / "d1_paired_manifest.parquet"
D1_TOP5 = D1_ROOT / "02_candidates" / "d1_test_top5.parquet"
D1_OUTCOMES = D1_ROOT / "09_formal_test" / "formal_candidate_outcomes.parquet"
D1_DECISIONS = D1_ROOT / "09_formal_test" / "formal_candidate_score_decision_bundle.parquet"
D1_FAILURES = D1_ROOT / "tables" / "d1_failure_decomposition.csv"
D1_SELECTION = D1_ROOT / "tables" / "d1_case_selection.csv"
G1_RECOVERY_BOARD = (
    REPO
    / "runs"
    / "reranking_case_visuals_v2_20260810T203154Z"
    / "04_boards"
    / "g1"
    / "recovered"
    / "q0017659_64789b8613f18b5d_case_board.png"
)
UNIFIED_TOP5 = {
    route: (
        REPO
        / "runs"
        / "fair_unified_reranking_20260809_103012"
        / "02_candidates"
        / f"{route}_test_top5.parquet"
    )
    for route in ("crog", "g1", "c1")
}
UNIFIED_G1_TOP5 = (
    REPO
    / "runs"
    / "fair_unified_reranking_20260809_103012"
    / "02_candidates"
    / "g1_test_top5.parquet"
)
UNIFIED_FINAL_LOCK = (
    REPO
    / "runs"
    / "fair_unified_reranking_20260809_103012"
    / "FINAL_RUN_LOCK.json"
)
BASELINE_MANIFEST = (
    REPO
    / "runs"
    / "fair_crog_hifics_g1_c1_no_rerank_20260807_091523"
    / "01_manifest"
    / "paired_manifest.parquet"
)
CONTRACT = (
    REPO
    / "runs"
    / "four_route_evidence_consolidation_20260813T155455Z"
    / "00_audit"
    / "EVALUATOR_AUDIT.md"
)

MAIN_KEY = ("crog", "q0007634_3a4a159ceef589b4")
CASE_ROWS = [
    ("crog", "q0007634_3a4a159ceef589b4", "ranking_recovery"),
    ("crog", "q0005395_a60bc13237137645", "harmful_replacement"),
    ("g1", "q0017659_64789b8613f18b5d", "ranking_recovery"),
    ("c1", "q0001344_5d5bba2ea6dc982e", "candidate_pool_failure"),
]

ROUTE_DISPLAY = {
    "crog": "CROG",
    "d1": "D1 (HiFi-CS → Dex-Net/GQ-CNN)",
    "g1": "G1 (HiFi-CS → GR-ConvNet)",
    "c1": "C1 (HiFi-CS → GG-CNN2)",
}
ROUTE_COLORS = {
    "crog": "#0072B2",
    "d1": "#E69F00",
    "g1": "#009E73",
    "c1": "#CC79A7",
}
COLORS = {
    "gt": "#6F2DA8",
    "native": "#0072B2",
    "final": "#D55E00",
    "mask": "#009E73",
    "candidate": "#6B7280",
    "pass": "#0072B2",
    "fail": "#B33A3A",
    "ink": "#18212B",
    "muted": "#55616F",
    "panel": "#F4F6F8",
}
CANDIDATE_COLORS = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#332288"]
CANDIDATE_ALIAS_BY_CANONICAL_ID = {
    f"candidate_{index}": f"K{index + 1}" for index in range(5)
}
TARGET_PRINT_WIDTH_MM = 168.3
MAIN_WIDTH_IN = TARGET_PRINT_WIDTH_MM / 25.4
MAIN_ASPECT_RATIO = 6.2 / 7.2
MAIN_SIZE_IN = (MAIN_WIDTH_IN, MAIN_WIDTH_IN * MAIN_ASPECT_RATIO)
APPENDIX_SIZE_IN = (7.2, 2.45)
MIN_FONT_PT = 9.3
VERDICT_FONT_PT = 9.2


@dataclass
class CaseEvidence:
    route: str
    sample_id: str
    row: Mapping[str, Any]
    candidates: pd.DataFrame
    rgb: np.ndarray
    gt_mask: np.ndarray
    predicted_mask: np.ndarray
    gt_grasps: list[list[list[float]]]
    crop: tuple[float, float, float, float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def publish_submission_mirrors(paths: list[Path]) -> list[Path]:
    """Copy byte-identical build artefacts to the dissertation contract path."""
    SUBMISSION_OUT.mkdir(parents=True, exist_ok=True)
    mirrors: list[Path] = []
    for source in paths:
        target = SUBMISSION_OUT / source.name
        shutil.copyfile(source, target)
        if sha256(target) != sha256(source):
            raise AssertionError(f"submission mirror differs from active output: {target}")
        mirrors.append(target)
    return mirrors


def resolve_asset(raw: Any) -> Path:
    path = Path(str(raw))
    if path.is_file():
        return path
    parts = list(path.parts)
    if "VLMGraspPose" in parts:
        candidate = REPO.joinpath(*parts[parts.index("VLMGraspPose") + 1 :])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def load_target_mask(path: Path, target_instance_id: int) -> np.ndarray:
    values = np.asarray(Image.open(path))
    return values == int(target_instance_id)


def read_crog_rle(sample_index: int) -> tuple[np.ndarray, str]:
    with CROG_FEATURES.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if int(record.get("sample_index", -1)) == int(sample_index):
                mask = rle_decode(record["predicted_mask_rle"])
                identity = f"{rel(CROG_FEATURES)}:{line_number}"
                return mask, identity + ":sha256=" + sha256_bytes(line.encode("utf-8"))
    raise KeyError(f"CROG sample_index {sample_index} not found")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": MIN_FONT_PT,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
        }
    )


def expand_crop(
    crop: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    *,
    min_width: float,
    target_aspect: float,
) -> tuple[float, float, float, float]:
    """Expand a real source crop without resampling, alteration, or clipping."""

    image_h, image_w = image_shape[:2]
    x0, y0, x1, y1 = map(float, crop)
    width = max(x1 - x0, min_width)
    height = max(y1 - y0, width / target_aspect)
    width = max(width, height * target_aspect)
    width = min(width, float(image_w))
    height = min(height, float(image_h))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    new_x0 = min(max(0.0, cx - 0.5 * width), float(image_w) - width)
    new_y0 = min(max(0.0, cy - 0.5 * height), float(image_h) - height)
    return (new_x0, new_y0, new_x0 + width, new_y0 + height)


def crop_from_points(
    image_shape: tuple[int, ...], masks: list[np.ndarray], points: list[np.ndarray]
) -> tuple[float, float, float, float]:
    height, width = image_shape[:2]
    xs: list[float] = []
    ys: list[float] = []
    for mask in masks:
        yy, xx = np.nonzero(mask)
        if len(xx):
            xs.extend([float(xx.min()), float(xx.max())])
            ys.extend([float(yy.min()), float(yy.max())])
    for value in points:
        array = np.asarray(value, dtype=float)
        if array.size:
            xs.extend([float(array[:, 0].min()), float(array[:, 0].max())])
            ys.extend([float(array[:, 1].min()), float(array[:, 1].max())])
    if not xs:
        return (0.0, 0.0, float(width), float(height))
    pad_x = max(30.0, 0.28 * (max(xs) - min(xs) + 1.0))
    pad_y = max(30.0, 0.38 * (max(ys) - min(ys) + 1.0))
    x0 = max(0.0, min(xs) - pad_x)
    y0 = max(0.0, min(ys) - pad_y)
    x1 = min(float(width), max(xs) + pad_x)
    y1 = min(float(height), max(ys) + pad_y)
    # Small thesis panels need context around a tall target; expand a narrow
    # crop horizontally while retaining every real candidate and mask pixel.
    target_aspect = 1.08
    if x1 - x0 < target_aspect * (y1 - y0):
        centre = 0.5 * (x0 + x1)
        half = 0.5 * target_aspect * (y1 - y0)
        x0 = max(0.0, centre - half)
        x1 = min(float(width), centre + half)
    return (x0, y0, x1, y1)


def load_canonical_case(
    case_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    route: str,
    sample_id: str,
    *,
    crog_rle: bool = False,
) -> tuple[CaseEvidence, str | None]:
    selected = case_table[
        case_table["route"].eq(route) & case_table["sample_id"].eq(sample_id)
    ]
    if len(selected) != 1:
        raise AssertionError(f"expected one canonical case row: {route}/{sample_id}")
    row = selected.iloc[0]
    candidates = candidate_table[
        candidate_table["route"].eq(route)
        & candidate_table["sample_id"].eq(sample_id)
    ].sort_values("native_rank")
    if len(candidates) != int(row["candidate_count_top5"]):
        raise AssertionError(f"candidate count mismatch: {route}/{sample_id}")
    rgb_path = resolve_asset(row["rgb_path"])
    gt_path = resolve_asset(row["gt_mask_path"])
    rgb = load_rgb(rgb_path)
    gt_mask = load_target_mask(gt_path, int(row["target_instance_id"]))
    rle_identity = None
    if crog_rle:
        # The CROG feature JSONL is keyed by the original expression/question
        # index, whereas the paired formal manifest has its own row index.
        predicted, rle_identity = read_crog_rle(int(row["question_index"]))
        expected_iou = float(row["crog_mask_iou"])
    else:
        predicted = load_binary_mask(resolve_asset(row["predicted_mask_path"]))
        expected_iou = float(row["hifics_mask_iou"])
    if predicted.shape != gt_mask.shape:
        raise AssertionError(f"mask shape mismatch: {route}/{sample_id}")
    observed_iou = binary_iou(predicted, gt_mask)
    if not math.isclose(observed_iou, expected_iou, abs_tol=1e-12):
        raise AssertionError(
            f"mask IoU mismatch for {route}/{sample_id}: {observed_iou} vs {expected_iou}"
        )
    crop = tuple(map(float, json.loads(row["crop_extent_json"])))
    gt_grasps = json.loads(row["gt_grasp_list_json"])
    return (
        CaseEvidence(
            route=route,
            sample_id=sample_id,
            row=row.to_dict(),
            candidates=candidates.reset_index(drop=True),
            rgb=rgb,
            gt_mask=gt_mask,
            predicted_mask=predicted,
            gt_grasps=gt_grasps,
            crop=crop,
        ),
        rle_identity,
    )


def tint_mask(rgb: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    base = rgb.astype(float) * 0.68
    value = np.array(matplotlib.colors.to_rgb(color)) * 255.0
    base[mask] = 0.40 * rgb[mask].astype(float) + 0.60 * value
    return np.clip(base, 0, 255).astype(np.uint8)


def wrap_prompt(text: str, width: int = 64) -> str:
    """Wrap the complete prompt; never truncate it with an ellipsis."""

    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def set_image_axis(
    ax: plt.Axes,
    image: np.ndarray,
    crop: tuple[float, float, float, float],
    title: str,
    *,
    title_fontsize: float = MIN_FONT_PT,
) -> None:
    ax.imshow(image, interpolation="nearest")
    x0, y0, x1, y1 = crop
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C3CAD2")
        spine.set_linewidth(0.6)
    ax.set_title(
        title,
        loc="left",
        pad=2,
        fontsize=title_fontsize,
        color=COLORS["ink"],
        linespacing=1.0,
    )


def draw_polygon(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str | None = None,
    label_offset: tuple[float, float] = (2.0, -4.0),
    zorder: int = 5,
) -> None:
    value = np.asarray(points, dtype=float)
    patch = Polygon(
        value,
        closed=True,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=linewidth,
        joinstyle="round",
        zorder=zorder,
    )
    patch.set_path_effects(
        [path_effects.Stroke(linewidth=linewidth + 1.15, foreground="white"), path_effects.Normal()]
    )
    ax.add_patch(patch)
    if label:
        anchor = value[np.argmin(value[:, 1])]
        text = ax.text(
            anchor[0] + label_offset[0],
            anchor[1] + label_offset[1],
            label,
            color=color,
            fontsize=MIN_FONT_PT,
            weight="bold",
            ha="left",
            va="bottom",
            zorder=zorder + 1,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": color, "linewidth": 0.45, "alpha": 0.90},
        )
        text.set_path_effects([path_effects.withStroke(linewidth=0.35, foreground="white")])


def draw_matched_gt(ax: plt.Axes, candidate: Mapping[str, Any], label: bool = True) -> None:
    if "matched_gt_corners_json" in candidate and pd.notna(candidate["matched_gt_corners_json"]):
        points = np.asarray(json.loads(candidate["matched_gt_corners_json"]), dtype=float)
    else:
        raise KeyError("matched GT corners unavailable")
    draw_polygon(
        ax,
        points,
        color=COLORS["gt"],
        linestyle="--",
        linewidth=1.55,
        label="matched GT" if label else None,
        label_offset=(2.0, -12.0),
        zorder=4,
    )


def draw_gt_from_index(
    ax: plt.Axes, gt_grasps: list[list[list[float]]], index: int, label: bool = True
) -> None:
    draw_polygon(
        ax,
        np.asarray(gt_grasps[int(index)], dtype=float),
        color=COLORS["gt"],
        linestyle="--",
        linewidth=1.55,
        label="matched GT" if label else None,
        label_offset=(2.0, -12.0),
        zorder=4,
    )


def draw_candidate(
    ax: plt.Axes,
    candidate: Mapping[str, Any],
    role: str,
    label: str | None,
    *,
    zorder: int = 6,
) -> None:
    styles = {
        "native": (COLORS["native"], "-."),
        "final": (COLORS["final"], "-"),
        "candidate": (COLORS["candidate"], "-"),
    }
    color, linestyle = styles[role]
    draw_polygon(
        ax,
        rectangle_points(candidate),
        color=color,
        linestyle=linestyle,
        linewidth=1.75 if role != "candidate" else 1.15,
        label=label,
        zorder=zorder,
    )


def draw_mask_contour(ax: plt.Axes, mask: np.ndarray) -> None:
    ax.contour(
        mask.astype(float),
        levels=[0.5],
        colors=[COLORS["gt"]],
        linewidths=[1.15],
        linestyles=["--"],
        zorder=4,
    )


def add_candidate_key(
    ax: plt.Axes,
    count: int,
    *,
    y: float = 1.02,
    prefix: str = "K",
) -> None:
    """Place the numbered Top-5 key outside the image region."""

    for index in range(count):
        ax.text(
            0.03 + 0.19 * index,
            y,
            f"{prefix}{index + 1}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=MIN_FONT_PT,
            weight="bold",
            color=CANDIDATE_COLORS[index],
            zorder=20,
            clip_on=False,
        )


def candidate_display_alias(candidate_id: str) -> str:
    """Return the immutable display alias for a canonical CROG candidate ID."""

    try:
        return CANDIDATE_ALIAS_BY_CANONICAL_ID[candidate_id]
    except KeyError as exc:
        raise AssertionError(f"unexpected canonical CROG candidate ID: {candidate_id}") from exc


def candidate_by_id(case: CaseEvidence, candidate_id: str) -> Mapping[str, Any]:
    selected = case.candidates[case.candidates["candidate_id"].eq(candidate_id)]
    if len(selected) != 1:
        raise AssertionError(f"candidate identity mismatch: {case.sample_id}/{candidate_id}")
    return selected.iloc[0]


def add_main_pipeline(fig: plt.Figure, spec, main: CaseEvidence) -> list[plt.Axes]:
    canonical_ids = main.candidates["candidate_id"].astype(str).tolist()
    expected_ids = list(CANDIDATE_ALIAS_BY_CANONICAL_ID)
    if canonical_ids != expected_ids:
        raise AssertionError(
            f"CROG candidate display order changed: {canonical_ids} != {expected_ids}"
        )

    grid = spec.subgridspec(
        3,
        4,
        height_ratios=[0.51, 1.0, 1.0],
        hspace=0.32,
        wspace=0.08,
    )
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    header.text(
        0.0,
        0.84,
        "(A) Complete fixed-candidate pipeline recovery — CROG",
        fontsize=10.6,
        weight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    header.text(
        0.0,
        0.08,
        f"Instruction: “{main.row['language']}”",
        fontsize=MIN_FONT_PT,
        color=COLORS["muted"],
        ha="left",
        va="bottom",
    )
    axes_top = [fig.add_subplot(grid[1, index]) for index in range(4)]
    set_image_axis(axes_top[0], main.rgb, main.crop, "1  RGB")
    set_image_axis(
        axes_top[1],
        tint_mask(main.rgb, main.gt_mask, COLORS["gt"]),
        main.crop,
        f"2  GT mask · instance {int(main.row['target_instance_id'])}",
    )
    set_image_axis(
        axes_top[2],
        tint_mask(main.rgb, main.predicted_mask, COLORS["mask"]),
        main.crop,
        f"3  CROG mask · IoU {float(main.row['crog_mask_iou']):.3f}",
    )
    set_image_axis(
        axes_top[3],
        main.rgb,
        main.crop,
        f"4  Top-5 fixed · {int(main.row['positive_count_top5'])}/5 PASS",
    )
    for index, (_, candidate) in enumerate(main.candidates.iterrows()):
        draw_polygon(
            axes_top[3],
            rectangle_points(candidate),
            color=CANDIDATE_COLORS[index],
            linestyle="-",
            linewidth=1.25,
            label=None,
            zorder=5 + index,
        )
    add_candidate_key(axes_top[3], len(main.candidates), y=0.02)

    native = candidate_by_id(main, str(main.row["native_candidate_id"]))
    final = candidate_by_id(main, str(main.row["gated_candidate_id"]))
    axes_bottom = [fig.add_subplot(grid[2, index]) for index in range(4)]
    set_image_axis(
        axes_bottom[0],
        main.rgb,
        main.crop,
        "5  Native Top-1 · FAIL",
    )
    draw_matched_gt(axes_bottom[0], native, label=False)
    draw_candidate(axes_bottom[0], native, "native", None)
    set_image_axis(
        axes_bottom[1],
        main.rgb,
        main.crop,
        "6  Challenger · PASS",
    )
    draw_matched_gt(axes_bottom[1], final, label=False)
    draw_candidate(axes_bottom[1], final, "final", None)

    gate_ax = axes_bottom[2]
    gate_ax.set_facecolor(COLORS["panel"])
    gate_ax.set_xticks([])
    gate_ax.set_yticks([])
    for spine in gate_ax.spines.values():
        spine.set_color("#C3CAD2")
        spine.set_linewidth(0.7)
    gate_ax.set_title("7  Gate", loc="left", pad=2, fontsize=9.5)
    gate_ax.text(0.5, 0.76, "SWITCH", transform=gate_ax.transAxes, ha="center", va="center", fontsize=10.6, weight="bold", color=COLORS["final"])
    gate_ax.text(0.5, 0.46, f"margin +{float(main.row['score_margin']):.3f} · votes {int(main.row['seed_challenger_votes'])}/3", transform=gate_ax.transAxes, ha="center", va="center", fontsize=MIN_FONT_PT, color=COLORS["ink"])
    gate_ax.text(0.5, 0.17, f"utility {float(main.row['utility']):+.3f} · IDs/geometry frozen", transform=gate_ax.transAxes, ha="center", va="center", fontsize=MIN_FONT_PT, color=COLORS["muted"])

    set_image_axis(
        axes_bottom[3],
        main.rgb,
        main.crop,
        "8  Final + GT · PASS",
    )
    draw_matched_gt(axes_bottom[3], final, label=False)
    draw_candidate(axes_bottom[3], final, "final", None)
    return [*axes_top, axes_bottom[0], axes_bottom[1], axes_bottom[3]]


def decision_rows(main: CaseEvidence, *, all_candidates: bool) -> list[list[str]]:
    selected_ids = {
        str(main.row["native_candidate_id"]),
        str(main.row["gated_candidate_id"]),
    }
    rows: list[list[str]] = []
    for _, candidate in main.candidates.iterrows():
        if not all_candidates and str(candidate["candidate_id"]) not in selected_ids:
            continue
        rows.append(
            [
                f"{candidate_display_alias(str(candidate['candidate_id']))} "
                f"{candidate['candidate_id']}",
                str(int(candidate["native_rank"])),
                str(int(candidate["improved_rank"])),
                f"{candidate['native_score']:.4f}",
                f"{candidate['ensemble_score']:.3f}",
                f"{candidate['diagnostic_iou']:.3f}",
                f"{candidate['diagnostic_angle_error_deg']:.1f}°",
                "PASS" if bool(candidate["candidate_success"]) else "FAIL",
            ]
        )
    return rows


def style_decision_table(table) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(MIN_FONT_PT)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#C8D0D8")
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_facecolor("#E8EDF2")
            cell.get_text().set_weight("bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F7F8FA")
        if row_index > 0 and col_index == 7:
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(
                COLORS["pass"] if cell.get_text().get_text() == "PASS" else COLORS["fail"]
            )


def add_candidate_strip(fig: plt.Figure, spec, main: CaseEvidence) -> None:
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.set_title(
        "Decision strip (candidate membership and geometry fixed)",
        loc="left",
        pad=2,
        fontsize=9.5,
        weight="bold",
    )
    columns = ["Candidate", "Nat. rank", "Re-rank", "Nat. score", "Rerank score", "IoU", "Angle", "Verdict"]
    table = ax.table(
        cellText=decision_rows(main, all_candidates=False),
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, -0.12, 1.0, 1.00],
        colWidths=[0.19, 0.10, 0.09, 0.12, 0.13, 0.10, 0.11, 0.10],
    )
    style_decision_table(table)


def gallery_titles(kind: str, case: CaseEvidence) -> tuple[str, str]:
    if kind == "native_success":
        return (
            "Native success",
            "Failing challenger rejected; native retained.",
        )
    if kind == "angle_only_recovery":
        return (
            "Angle-only recovery",
            "Native overlap passed; angle threshold failed.",
        )
    if kind == "candidate_pool_failure":
        return (
            "Candidate-pool failure",
            "No positive candidate was available to rerank.",
        )
    if kind == "harmful_replacement":
        return (
            "Harmful replacement",
            "The gate accepted a harmful switch.",
        )
    raise KeyError(kind)


def add_gallery_card(
    fig: plt.Figure,
    spec,
    case: CaseEvidence,
    kind: str,
    index: int,
) -> list[plt.Axes]:
    background = fig.add_subplot(spec)
    background.set_facecolor("#FAFBFC")
    background.set_xticks([])
    background.set_yticks([])
    for spine in background.spines.values():
        spine.set_color("#BFC7D0")
        spine.set_linewidth(0.8)
    background.spines["top"].set_color(ROUTE_COLORS[case.route])
    background.spines["top"].set_linewidth(1.5)
    background.set_zorder(0)
    nested = spec.subgridspec(
        3,
        3,
        # Reserve a true two-line header band.  At the final 0.99-textwidth
        # scale, a 10 pt title plus a 9.3 pt prompt cannot share the former
        # shallow row without their glyph boxes touching.
        height_ratios=[0.72, 0.60, 0.50],
        hspace=0.03,
        wspace=0.05,
    )
    title_ax = fig.add_subplot(nested[0, :])
    title_ax.axis("off")
    title, verdict = gallery_titles(kind, case)
    title_ax.text(
        0.015,
        0.98,
        f"({chr(65 + index)}) {title} · {case.route.upper()}{'*' if case.route == 'd1' else ''}",
        ha="left",
        va="top",
        fontsize=10.0,
        weight="bold",
        color=ROUTE_COLORS[case.route],
    )
    title_ax.text(
        0.015,
        0.02,
        f"“{wrap_prompt(str(case.row['language']), 62)}”",
        ha="left",
        va="bottom",
        fontsize=MIN_FONT_PT,
        color=COLORS["muted"],
        style="italic",
        linespacing=1.0,
    )
    image_axes = [fig.add_subplot(nested[1, column]) for column in range(3)]

    cue = tint_mask(case.rgb, case.predicted_mask, COLORS["mask"])
    if case.route == "d1":
        cue_title = "Cue"
        set_image_axis(
            image_axes[0],
            cue,
            case.crop,
            "",
        )
        draw_mask_contour(image_axes[0], case.gt_mask)
        for candidate_index, (_, candidate) in enumerate(case.candidates.iterrows()):
            draw_polygon(
                image_axes[0],
                rectangle_points(candidate),
                color=CANDIDATE_COLORS[candidate_index],
                linestyle="-",
                linewidth=0.95,
                label=None,
                zorder=5 + candidate_index,
            )
        add_candidate_key(image_axes[0], len(case.candidates), y=0.14, prefix="")
    else:
        cue_title = "Cue"
        set_image_axis(
            image_axes[0],
            cue,
            case.crop,
            "",
        )
        draw_mask_contour(image_axes[0], case.gt_mask)

    native_id = str(case.row["native_candidate_id"])
    final_id = str(case.row["gated_candidate_id"])
    native = candidate_by_id(case, native_id)
    final = candidate_by_id(case, final_id)
    if case.route == "d1":
        native_iou = float(native["best_same_gt_iou"])
        final_iou = float(final["best_same_gt_iou"])
        native_title = "Native"
        final_title = "Final"
        matched_native = int(native["matched_gt_index"])
        matched_final = int(final["matched_gt_index"])
    else:
        native_iou = float(native["diagnostic_iou"])
        final_iou = float(final["diagnostic_iou"])
        native_status = "PASS" if bool(native["candidate_success"]) else "FAIL"
        final_status = "PASS" if bool(final["candidate_success"]) else "FAIL"
        native_title = "Native"
        final_title = "Final"
        matched_native = matched_final = -1

    set_image_axis(image_axes[1], case.rgb, case.crop, "")
    if case.route == "d1":
        draw_gt_from_index(image_axes[1], case.gt_grasps, matched_native, label=False)
    else:
        draw_matched_gt(image_axes[1], native, label=False)
    draw_candidate(image_axes[1], native, "native", None)

    set_image_axis(image_axes[2], case.rgb, case.crop, "")
    if case.route == "d1":
        draw_gt_from_index(image_axes[2], case.gt_grasps, matched_final, label=False)
    else:
        draw_matched_gt(image_axes[2], final, label=False)
    draw_candidate(image_axes[2], final, "final", None)

    for image_ax, heading in zip(
        image_axes, [cue_title, native_title, final_title], strict=True
    ):
        # Keep the prompt in its own header band and put the view label in the
        # generous inter-column gutter.  This avoids both an extra text row and
        # any overlap with masks or grasp rectangles.
        image_ax.text(
            -0.08,
            0.50,
            heading,
            transform=image_ax.transAxes,
            ha="right",
            va="center",
            fontsize=MIN_FONT_PT,
            weight="bold",
            color=COLORS["ink"],
            clip_on=False,
            zorder=30,
        )

    note_ax = fig.add_subplot(nested[2, :])
    note_ax.axis("off")
    if case.route == "d1":
        evidence = (
            f"N/F FAIL · IoU {native_iou:.3f} · mask {float(case.row['mask_iou']):.3f}"
            " · 0/5 and 0/19 positives"
        )
    else:
        evidence = (
            f"N {native_status} {native_iou:.3f}/{float(native['diagnostic_angle_error_deg']):.1f}°"
            f" → F {final_status} {final_iou:.3f}/{float(final['diagnostic_angle_error_deg']):.1f}°"
        )
    note_ax.text(
        0.02,
        0.50,
        evidence + "\n" + verdict,
        ha="left",
        va="center",
        fontsize=MIN_FONT_PT,
        weight="bold",
        linespacing=1.40,
        color=COLORS["fail"] if kind in {"candidate_pool_failure", "harmful_replacement"} else COLORS["ink"],
    )
    return image_axes


COLUMN_WIDTHS = [1.0, 1.0, 1.0, 1.0, 1.0, 2.13]


def row_title(kind: str) -> tuple[str, str]:
    if kind == "ranking_recovery":
        return "Ranking recovery", "RECOVERED"
    if kind == "harmful_replacement":
        return "Harmful replacement", "HARMFUL"
    if kind == "candidate_pool_failure":
        return "No-positive frozen pool", "NO PASS"
    raise KeyError(kind)


def short_candidate_id(case: CaseEvidence, candidate_id: str) -> str:
    if case.route == "crog":
        return candidate_display_alias(candidate_id)
    return candidate_id.removeprefix("native_")


def add_column_headings(fig: plt.Figure, spec) -> None:
    grid = spec.subgridspec(1, 6, width_ratios=COLUMN_WIDTHS, wspace=0.055)
    headings = [
        "RGB +\nexpression",
        "Target\nevidence",
        "Frozen\ncandidates",
        "Native\nTop-1",
        "Final\nTop-1",
        "Evaluator\nverdict",
    ]
    for index, heading in enumerate(headings):
        ax = fig.add_subplot(grid[0, index])
        ax.axis("off")
        ax.text(
            0.5,
            0.48,
            heading,
            ha="center",
            va="center",
            fontsize=MIN_FONT_PT,
            weight="bold",
            color=COLORS["ink"],
            linespacing=1.35,
        )
        if index < len(headings) - 1:
            ax.text(
                1.04,
                0.48,
                "→",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10.2,
                weight="bold",
                color="#7B8794",
                clip_on=False,
            )


def add_pool_key(ax: plt.Axes, case: CaseEvidence) -> None:
    positions = np.linspace(0.08, 0.92, len(case.candidates))
    for position, (_, candidate) in zip(
        positions, case.candidates.iterrows(), strict=True
    ):
        rank = int(candidate["native_rank"])
        label = (
            candidate_display_alias(str(candidate["candidate_id"]))
            if case.route == "crog"
            else f"r{rank}"
        )
        ax.text(
            position,
            1.015,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=MIN_FONT_PT,
            weight="bold",
            color=CANDIDATE_COLORS[rank - 1],
            clip_on=False,
            zorder=20,
        )


def verdict_card(
    ax: plt.Axes,
    case: CaseEvidence,
    kind: str,
    native: Mapping[str, Any],
    final: Mapping[str, Any],
) -> None:
    ax.set_facecolor("#F7F9FB")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C3CAD2")
        spine.set_linewidth(0.7)
    title, outcome = row_title(kind)
    del title
    outcome_color = (
        COLORS["fail"]
        if kind in {"harmful_replacement", "candidate_pool_failure"}
        else COLORS["pass"]
    )
    native_id = str(native["candidate_id"])
    final_id = str(final["candidate_id"])
    native_status = "PASS" if bool(native["candidate_success"]) else "FAIL"
    final_status = "PASS" if bool(final["candidate_success"]) else "FAIL"
    gate = "KEEP" if native_id == final_id else "SWITCH"
    ax.text(
        0.04,
        0.94,
        f"{outcome} · {gate}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        weight="bold",
        color=outcome_color,
    )
    native_label = (
        f"{native_id} ({short_candidate_id(case, native_id)})"
        if case.route == "crog"
        else native_id
    )
    final_label = (
        f"{final_id} ({short_candidate_id(case, final_id)})"
        if case.route == "crog"
        else final_id
    )
    # Keep the full evaluator trace at print-readable size without forcing a
    # long single line.  Candidate/metric fields are left aligned; rank and
    # PASS/FAIL fields are right aligned in the same fixed four-row grammar.
    rows = [
        (
            0.665,
            f"N  {native_label}",
            f"r{int(native['native_rank'])}→{int(native['improved_rank'])}",
        ),
        (
            0.475,
            f"IoU {float(native['diagnostic_iou']):.3f} · "
            f"Δθ {float(native['diagnostic_angle_error_deg']):.1f}°",
            native_status,
        ),
        (
            0.275,
            f"F  {final_label}",
            f"r{int(final['native_rank'])}→{int(final['improved_rank'])}",
        ),
        (
            0.085,
            f"IoU {float(final['diagnostic_iou']):.3f} · "
            f"Δθ {float(final['diagnostic_angle_error_deg']):.1f}°",
            final_status,
        ),
    ]
    for y, left_text, right_text in rows:
        ax.text(
            0.04,
            y,
            left_text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=VERDICT_FONT_PT,
            color=COLORS["ink"],
        )
        ax.text(
            0.96,
            y,
            right_text,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=VERDICT_FONT_PT,
            color=COLORS["ink"],
        )


def add_case_row(
    fig: plt.Figure,
    spec,
    case: CaseEvidence,
    kind: str,
    index: int,
) -> list[plt.Axes]:
    nested = spec.subgridspec(
        2,
        6,
        height_ratios=[0.42, 1.0],
        width_ratios=COLUMN_WIDTHS,
        hspace=0.08,
        wspace=0.055,
    )
    header = fig.add_subplot(nested[0, :])
    header.axis("off")
    header.set_xlim(0.0, 1.0)
    header.set_ylim(0.0, 1.0)
    title, _ = row_title(kind)
    route_label = case.route.upper()
    mask_source = "CROG mask" if case.route == "crog" else "HiFi-CS mask"
    header.axhline(
        1.05,
        color=ROUTE_COLORS[case.route],
        linewidth=1.4,
        clip_on=False,
    )
    header.text(
        0.0,
        0.80,
        f"({chr(65 + index)}) {route_label} · {title}",
        ha="left",
        va="center",
        fontsize=10.3,
        weight="bold",
        color=ROUTE_COLORS[case.route],
    )
    header.text(
        0.0,
        0.13,
        f"“{case.row['language']}”",
        ha="left",
        va="bottom",
        fontsize=MIN_FONT_PT,
        style="italic",
        color=COLORS["muted"],
    )
    header.text(
        1.0,
        0.80,
        f"{mask_source} IoU {float(case.row['mask_iou']):.3f} · "
        f"frozen pool {int(case.row['positive_count_top5'])}/{int(case.row['candidate_count_top5'])} PASS",
        ha="right",
        va="center",
        fontsize=MIN_FONT_PT,
        weight="bold",
        color=COLORS["ink"],
    )

    axes = [fig.add_subplot(nested[1, column]) for column in range(6)]
    set_image_axis(axes[0], case.rgb, case.crop, "")

    evidence = tint_mask(case.rgb, case.predicted_mask, COLORS["mask"])
    set_image_axis(axes[1], evidence, case.crop, "")
    draw_mask_contour(axes[1], case.gt_mask)

    set_image_axis(axes[2], case.rgb, case.crop, "")
    for _, candidate in case.candidates.iterrows():
        rank = int(candidate["native_rank"])
        draw_polygon(
            axes[2],
            rectangle_points(candidate),
            color=CANDIDATE_COLORS[rank - 1],
            linestyle="-",
            linewidth=1.15,
            label=None,
            zorder=4 + rank,
        )
    add_pool_key(axes[2], case)

    native = candidate_by_id(case, str(case.row["native_candidate_id"]))
    final = candidate_by_id(case, str(case.row["gated_candidate_id"]))
    set_image_axis(axes[3], case.rgb, case.crop, "")
    draw_matched_gt(axes[3], native, label=False)
    draw_candidate(axes[3], native, "native", None)
    set_image_axis(axes[4], case.rgb, case.crop, "")
    draw_matched_gt(axes[4], final, label=False)
    draw_candidate(axes[4], final, "final", None)
    verdict_card(axes[5], case, kind, native, final)
    return axes[:5]


def build_figure(
    cases: list[tuple[CaseEvidence, str]],
) -> tuple[Path, Path, dict[str, Any]]:
    configure_style()
    fig = plt.figure(figsize=MAIN_SIZE_IN)
    # Reserve a small print-safe right gutter for the longest verdict labels.
    fig.subplots_adjust(left=0.025, right=0.975, top=0.985, bottom=0.066)
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.19, 1.0, 1.0, 1.0, 1.0],
        hspace=0.12,
    )
    add_column_headings(fig, outer[0])
    case_axes: list[tuple[CaseEvidence, list[plt.Axes]]] = []
    for index, (case, kind) in enumerate(cases):
        axes = add_case_row(fig, outer[index + 1], case, kind, index)
        case_axes.append((case, axes))
    fig.text(
        0.5,
        0.010,
        "GT: purple dashed · Native: blue dash–dot · Final: orange solid\n"
        "candidate colours: frozen native rank",
        ha="center",
        va="bottom",
        fontsize=MIN_FONT_PT,
        linespacing=1.35,
        color=COLORS["muted"],
    )

    fig.canvas.draw()
    print_scale = (TARGET_PRINT_WIDTH_MM / 25.4) / MAIN_SIZE_IN[0]
    raster_limit_ppi = 300.0 / print_scale
    case_ppi: dict[str, float] = {}
    for case, axes in case_axes:
        axis_width = max(
            axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
            for axis in axes
        )
        case_ppi[case.sample_id] = min(
            (case.crop[2] - case.crop[0]) / (axis_width * print_scale),
            raster_limit_ppi,
        )
    quality = {
        "target_print_width_mm": TARGET_PRINT_WIDTH_MM,
        "print_scale": print_scale,
        "minimum_effective_font_pt": VERDICT_FONT_PT * print_scale,
        "case_effective_ppi": case_ppi,
        "pdf_raster_limit_ppi_at_print_width": raster_limit_ppi,
    }

    pdf = OUT / "stage_failure_cases.pdf"
    png = OUT / "stage_failure_cases.png"
    metadata = {
        "Title": "Chapter 4 stage-wise qualitative evidence",
        "Author": "VLMGraspPose reproducible evidence pipeline",
        "Subject": "Four frozen real cases spanning CROG and modular routes",
        "Keywords": f"script_sha256={sha256(SCRIPT)}; offline same-GT evaluator",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        pdf,
        format="pdf",
        dpi=300,
        bbox_inches=None,
        pad_inches=0,
        metadata=metadata,
    )
    fig.savefig(
        png,
        format="png",
        dpi=300,
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)
    return pdf, png, quality


def build_candidate_appendix(main: CaseEvidence) -> Path:
    configure_style()
    fig, ax = plt.subplots(figsize=APPENDIX_SIZE_IN)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.68, bottom=0.08)
    ax.axis("off")
    fig.text(
        0.035,
        0.91,
        "CROG complete frozen Top-5 candidate trace",
        ha="left",
        va="top",
        fontsize=10.6,
        weight="bold",
    )
    fig.text(
        0.035,
        0.79,
        f"Tuple {main.sample_id} · instruction: “{main.row['language']}”",
        ha="left",
        va="top",
        fontsize=MIN_FONT_PT,
        color=COLORS["muted"],
    )
    columns = ["Candidate", "Nat. rank", "Re-rank", "Nat. score", "Rerank score", "IoU", "Angle", "Verdict"]
    table = ax.table(
        cellText=decision_rows(main, all_candidates=True),
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.19, 0.10, 0.09, 0.12, 0.13, 0.10, 0.11, 0.10],
    )
    style_decision_table(table)
    table.scale(1.0, 1.08)
    pdf = OUT / "crog_candidate_trace_appendix.pdf"
    metadata = {
        "Title": "CROG frozen Top-5 candidate trace",
        "Author": "VLMGraspPose reproducible evidence pipeline",
        "Subject": "Full candidate-level trace for the Chapter 4 recovery case",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, format="pdf", dpi=300, bbox_inches=None, metadata=metadata)
    plt.close(fig)
    return pdf


def finite_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def canonical_case_manifest(case: CaseEvidence, kind: str) -> dict[str, Any]:
    native = candidate_by_id(case, str(case.row["native_candidate_id"]))
    final = candidate_by_id(case, str(case.row["gated_candidate_id"]))
    return {
        "role": kind,
        "route": ROUTE_DISPLAY[case.route],
        "sample_id": case.sample_id,
        "prompt": str(case.row["language"]),
        "source_rgb_size_px": [int(case.rgb.shape[1]), int(case.rgb.shape[0])],
        "display_crop_xyxy": [finite_float(value) for value in case.crop],
        "display_crop_size_px": [
            finite_float(case.crop[2] - case.crop[0]),
            finite_float(case.crop[3] - case.crop[1]),
        ],
        "mask_iou": finite_float(case.row["mask_iou"]),
        "native": {
            "candidate_id": str(native["candidate_id"]),
            "native_rank": int(native["native_rank"]),
            "reranked_rank": int(native["improved_rank"]),
            "pass": bool(native["candidate_success"]),
            "same_gt_iou": finite_float(native["diagnostic_iou"]),
            "angle_error_deg": finite_float(native["diagnostic_angle_error_deg"]),
        },
        "final": {
            "candidate_id": str(final["candidate_id"]),
            "native_rank": int(final["native_rank"]),
            "reranked_rank": int(final["improved_rank"]),
            "pass": bool(final["candidate_success"]),
            "same_gt_iou": finite_float(final["diagnostic_iou"]),
            "angle_error_deg": finite_float(final["diagnostic_angle_error_deg"]),
        },
        "frozen_pool": {
            "candidate_count": int(case.row["candidate_count_top5"]),
            "positive_count": int(case.row["positive_count_top5"]),
        },
        "gate_decision": str(case.row["gate_decision_reason"]),
        "earliest_observable_stage": str(case.row["earliest_observable_issue"]),
        "source_scope": "FORMAL_PRIMARY_POST_FORMAL_DIAGNOSTIC_FROM_FROZEN_OUTPUTS",
    }


def validate_case_selection_evidence() -> None:
    audit = pd.read_csv(SELECTED_AUDIT)
    expected = {
        ("crog", CASE_ROWS[0][1]): "recovered",
        ("crog", CASE_ROWS[1][1]): "harmful",
        ("g1", CASE_ROWS[2][1]): "recovered",
    }
    for (route, sample_id), outcome in expected.items():
        selected = audit[
            audit["route"].eq(route)
            & audit["sample_id"].eq(sample_id)
            & audit["presentation_outcome"].eq(outcome)
            & audit["selection_status"].eq("SELECTED")
            & audit["alternative_rank"].eq(1)
        ]
        if len(selected) != 1:
            raise AssertionError(
                f"case is no longer the deterministic QA selection: {route}/{sample_id}"
            )

    # Row D was deliberately changed after print-preview review.  It remains a
    # locked, mandatory-eligible same-GT case, but it is not misrepresented as
    # the rank-1 QA selection retained for rows A--C.
    eligible = pd.read_csv(ALL_ELIGIBLE_AUDIT)
    route, sample_id, _ = CASE_ROWS[3]
    matched = eligible[
        eligible["route"].eq(route)
        & eligible["sample_id"].eq(sample_id)
        & eligible["presentation_outcome"].eq("candidate_generation_irreparable")
    ]
    if len(matched) != 1:
        raise AssertionError(
            f"replacement case is not uniquely present in the locked eligible audit: "
            f"{route}/{sample_id}"
        )
    record = matched.iloc[0]
    if str(record["mandatory_eligible"]).strip().lower() != "true" or str(
        record["same_gt_native_final"]
    ).strip().lower() != "true":
        raise AssertionError(
            f"replacement case no longer satisfies mandatory/same-GT QA: "
            f"{route}/{sample_id}"
        )


def validate_frozen_case(case: CaseEvidence, kind: str) -> None:
    frozen = pd.read_parquet(UNIFIED_TOP5[case.route])
    frozen = frozen[frozen["sample_id"].eq(case.sample_id)].sort_values("native_rank")
    canonical = case.candidates.sort_values("native_rank")
    if len(frozen) != len(canonical):
        raise AssertionError(f"frozen-pool size changed: {case.route}/{case.sample_id}")
    frozen_by_id = frozen.set_index("candidate_id")
    canonical_by_id = canonical.set_index("candidate_id")
    if set(frozen_by_id.index) != set(canonical_by_id.index):
        raise AssertionError(f"frozen-pool membership changed: {case.route}/{case.sample_id}")
    for candidate_id in canonical_by_id.index:
        frozen_row = frozen_by_id.loc[candidate_id]
        canonical_row = canonical_by_id.loc[candidate_id]
        if int(frozen_row["native_rank"]) != int(canonical_row["native_rank"]):
            raise AssertionError(f"native rank changed: {case.sample_id}/{candidate_id}")
        if not math.isclose(
            float(frozen_row["native_score"]),
            float(canonical_row["native_score"]),
            abs_tol=1e-12,
        ):
            raise AssertionError(f"native score changed: {case.sample_id}/{candidate_id}")
        if str(frozen_row["candidate_geometry_sha256"]) != str(
            canonical_row["candidate_geometry_sha256"]
        ):
            raise AssertionError(f"candidate geometry changed: {case.sample_id}/{candidate_id}")

    native = candidate_by_id(case, str(case.row["native_candidate_id"]))
    final = candidate_by_id(case, str(case.row["gated_candidate_id"]))
    if kind == "ranking_recovery":
        valid = not bool(native["candidate_success"]) and bool(final["candidate_success"])
    elif kind == "harmful_replacement":
        valid = bool(native["candidate_success"]) and not bool(final["candidate_success"])
    elif kind == "candidate_pool_failure":
        valid = (
            not bool(native["candidate_success"])
            and not bool(final["candidate_success"])
            and int(case.row["positive_count_top5"]) == 0
            and not bool(canonical["candidate_success"].any())
        )
    else:
        raise KeyError(kind)
    if not valid:
        raise AssertionError(f"case role changed: {case.route}/{case.sample_id}/{kind}")


def write_manifest(
    main: CaseEvidence,
    cases: list[tuple[CaseEvidence, str]],
    rle_identities: dict[str, str],
    pdf: Path,
    png: Path,
    appendix_pdf: Path,
    quality: dict[str, Any],
) -> Path:
    source_paths = [
        CASE_VISUAL_HELPERS,
        CANONICAL_CASES,
        CANONICAL_CANDIDATES,
        SELECTED_AUDIT,
        ALL_ELIGIBLE_AUDIT,
        CROG_FEATURES,
        CONTRACT,
        G1_RECOVERY_BOARD,
        *UNIFIED_TOP5.values(),
        UNIFIED_FINAL_LOCK,
    ]
    assets: dict[str, dict[str, str]] = {}
    for case, _ in cases:
        rgb_path = resolve_asset(case.row["rgb_path"])
        gt_path = resolve_asset(case.row["gt_mask_path"])
        mask_path = (
            CROG_FEATURES
            if case.route == "crog"
            else resolve_asset(case.row["predicted_mask_path"])
        )
        assets[case.sample_id] = {
            "rgb": rel(rgb_path),
            "rgb_sha256": sha256(rgb_path),
            "gt_instance_mask": rel(gt_path),
            "gt_instance_mask_sha256": sha256(gt_path),
            "predicted_evidence": rel(mask_path),
            "predicted_evidence_sha256": sha256(mask_path),
        }

    manifest = {
        "schema_version": 3,
        "status": "COMPLETE",
        "script": {"path": rel(SCRIPT), "sha256": sha256(SCRIPT)},
        "main_canvas_inches": list(MAIN_SIZE_IN),
        "appendix_canvas_inches": list(APPENDIX_SIZE_IN),
        "font_family": "Arial",
        "minimum_source_font_pt": VERDICT_FONT_PT,
        "layout": {
            "case_rows": 4,
            "columns": [
                "RGB + expression",
                "target evidence",
                "frozen candidates",
                "native Top-1",
                "final Top-1",
                "evaluator verdict",
            ],
            "full_candidate_trace_location": "appendix-only",
        },
        "candidate_alias_mapping": {
            alias: candidate_id
            for candidate_id, alias in CANDIDATE_ALIAS_BY_CANONICAL_ID.items()
        },
        "candidate_alias_semantics": (
            "K denotes display membership in the frozen CROG Top-5 list; "
            "canonical IDs, membership and geometry are unchanged. Modular "
            "rows retain their native_peak candidate IDs."
        ),
        "layout_and_quality": quality,
        "offline_evaluator": {
            "criterion": "same GT rectangle; rotated IoU strictly > 0.25 and 180-degree-periodic angle error <= 30 degrees",
            "boundary": "offline 2D rectangle consistency, not a physical grasp trial",
            "source": rel(CONTRACT),
            "source_sha256": sha256(CONTRACT),
        },
        "cases": [
            {"row": chr(65 + index), **canonical_case_manifest(case, kind)}
            for index, (case, kind) in enumerate(cases)
        ],
        "appendix_candidate_trace": [
            {
                "display_id": row[0],
                "native_rank": int(candidate["native_rank"]),
                "reranked_rank": int(candidate["improved_rank"]),
                "native_score": finite_float(candidate["native_score"]),
                "reranker_score": finite_float(candidate["ensemble_score"]),
                "same_gt_iou": finite_float(candidate["diagnostic_iou"]),
                "angle_error_deg": finite_float(candidate["diagnostic_angle_error_deg"]),
                "pass": bool(candidate["candidate_success"]),
            }
            for row, (_, candidate) in zip(
                decision_rows(main, all_candidates=True),
                main.candidates.iterrows(),
                strict=True,
            )
        ],
        "crog_rle_records": rle_identities,
        "sources": [
            {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "assets": assets,
        "outputs": [
            {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in [pdf, png, appendix_pdf]
        ],
        "submission_contract_mirrors": [
            str((Path("figures") / "ch4" / path.name).as_posix())
            for path in [pdf, png, appendix_pdf]
        ]
        + ["figures/ch4/stage_failure_cases_manifest.json"],
        "selection_notes": [
            "Rows A--C are deterministic rank-1 SELECTED cases from the QA audit table.",
            "Row D is a locked mandatory-eligible same-GT audit case selected after the user-requested print-readability review; it is not represented as a rank-1 selection.",
            "Rows A and B decode route-correct CROG internal masks from frozen RLE records; Rows C and D use HiFi-CS masks.",
            "Row D is labelled as an earliest-observable frozen candidate-pool limitation. It does not claim that localisation and candidate generation have been causally separated.",
            "All candidate rectangles, masks, prompts, verdicts and metrics are read from frozen repository artefacts; no values are fabricated.",
            "Displayed crops are expanded only within the original image coordinate system; no grasp geometry or image content is altered.",
        ],
    }
    output = OUT / "stage_failure_cases_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source in [
        CASE_VISUAL_HELPERS,
        CANONICAL_CASES,
        CANONICAL_CANDIDATES,
        SELECTED_AUDIT,
        ALL_ELIGIBLE_AUDIT,
        CROG_FEATURES,
        CONTRACT,
        G1_RECOVERY_BOARD,
        *UNIFIED_TOP5.values(),
        UNIFIED_FINAL_LOCK,
    ]:
        if not source.is_file():
            raise FileNotFoundError(source)
    validate_case_selection_evidence()
    case_table = pd.read_parquet(CANONICAL_CASES)
    candidate_table = pd.read_parquet(CANONICAL_CANDIDATES)
    cases: list[tuple[CaseEvidence, str]] = []
    rle_identities: dict[str, str] = {}
    for route, sample_id, kind in CASE_ROWS:
        case, rle_identity = load_canonical_case(
            case_table,
            candidate_table,
            route,
            sample_id,
            crog_rle=route == "crog",
        )
        if route == "crog":
            if rle_identity is None:
                raise AssertionError(f"CROG RLE record identity missing: {sample_id}")
            rle_identities[sample_id] = rle_identity
        case.crop = expand_crop(
            case.crop,
            case.rgb.shape,
            min_width=330.0,
            target_aspect=4.0 / 3.0,
        )
        validate_frozen_case(case, kind)
        cases.append((case, kind))
    main = cases[0][0]
    pdf, png, quality = build_figure(cases)
    appendix = build_candidate_appendix(main)
    manifest = write_manifest(
        main,
        cases,
        rle_identities,
        pdf,
        png,
        appendix,
        quality,
    )
    mirrors = publish_submission_mirrors([pdf, png, appendix, manifest])
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(f"wrote {appendix}")
    print(f"wrote {manifest}")
    for mirror in mirrors:
        print(f"mirrored {mirror}")


if __name__ == "__main__":
    main()
