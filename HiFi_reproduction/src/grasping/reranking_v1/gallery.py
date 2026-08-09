"""Auditable qualitative galleries for frozen-candidate re-ranking.

The rendering path deliberately separates inference-visible data from
post-hoc evaluation data.  Ground-truth rectangles, correctness labels, IoU,
and angle errors are accepted only by :func:`_render_evaluation_panel`; they
cannot enter the RGB/mask/Top-5/selection panels.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ruamel.yaml import YAML

from src.grasping.geometric_ranker import make_ocid_vlg_evaluation_rectangles

from .vlm_visualization import (
    NEUTRAL_CANDIDATE_COLORS,
    _draw_candidate,
    _mask_boundary,
)


SCHEMA_VERSION = 1
DEFAULT_GALLERY_QUOTAS: Mapping[str, int] = {
    "recovered": 25,
    "harmful": 25,
    "both_wrong": 15,
    "vlm_fallback": 10,
    "vlm_learned_disagreement": 10,
}
CATEGORY_LABELS: Mapping[str, str] = {
    "recovered": "Recovered",
    "harmful": "Harmful",
    "both_wrong": "Both wrong",
    "vlm_fallback": "VLM abstain / fallback",
    "vlm_learned_disagreement": "VLM vs learned disagreement",
}
GT_COLOR = (0, 229, 255)
INFERENCE_BANNER_HEIGHT = 44
HEADER_HEIGHT = 118
PANEL_WIDTH = 480
PANEL_HEIGHT = 360
PANEL_GAP = 16
CANVAS_MARGIN = 22
INFERENCE_ROW_BOTTOM = HEADER_HEIGHT + INFERENCE_BANNER_HEIGHT + PANEL_HEIGHT


class GalleryError(ValueError):
    """Raised when gallery inputs violate the frozen-candidate contract."""


@dataclass(frozen=True)
class InferenceCandidate:
    """The only candidate fields allowed into inference-only panels."""

    candidate_id: str
    original_rank: int
    q_score: float
    reranker_rank: int | None
    reranker_score: float | None
    geometry: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationCandidate:
    """Post-hoc label data, kept out of inference rendering."""

    candidate_id: str
    positive: bool
    iou: float
    angle_error_deg: float
    best_gt_id: int | None


@dataclass(frozen=True)
class GalleryCase:
    sample_id: str
    scene_id: str
    category: str
    original_candidate_id: str
    new_candidate_id: str
    original_correct: bool
    new_correct: bool
    failure_category: str
    vlm_result: Mapping[str, Any] | None


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp.png"
    )
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canonical_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "method": "reranker_method",
        "rank": "reranker_rank",
        "score": "reranker_score",
    }
    result = frame.rename(
        columns={
            old: new
            for old, new in aliases.items()
            if old in frame.columns and new not in frame.columns
        }
    ).copy()
    required = {
        "sample_id",
        "candidate_id",
        "reranker_method",
        "reranker_rank",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise GalleryError(f"predictions missing columns: {missing}")
    if "reranker_score" not in result:
        result["reranker_score"] = np.nan
    for column in ("sample_id", "candidate_id", "reranker_method"):
        result[column] = result[column].astype(str)
    result["reranker_rank"] = pd.to_numeric(
        result["reranker_rank"], errors="coerce"
    )
    if (
        result["reranker_rank"].isna().any()
        or bool((result["reranker_rank"] < 1).any())
        or not np.allclose(
            result["reranker_rank"], np.round(result["reranker_rank"])
        )
    ):
        raise GalleryError("reranker ranks must be positive integers")
    result["reranker_rank"] = result["reranker_rank"].astype(np.int64)
    result["reranker_score"] = pd.to_numeric(
        result["reranker_score"], errors="coerce"
    )
    if result.duplicated(
        ["reranker_method", "sample_id", "candidate_id"]
    ).any():
        raise GalleryError("predictions contain duplicate method/sample/candidate IDs")
    return result


def _validated_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "scene_id",
        "candidate_id",
        "original_gqcnn_rank",
        "q_raw",
        "candidate_positive",
        "candidate_gt_iou",
        "candidate_gt_angle_error_deg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise GalleryError(f"per-candidate table missing columns: {missing}")
    result = frame.copy()
    for column in ("sample_id", "scene_id", "candidate_id"):
        result[column] = result[column].astype(str)
    if result.duplicated(["sample_id", "candidate_id"]).any():
        raise GalleryError("per-candidate table contains duplicate candidate IDs")
    result["original_gqcnn_rank"] = pd.to_numeric(
        result["original_gqcnn_rank"], errors="coerce"
    )
    if result["original_gqcnn_rank"].isna().any():
        raise GalleryError("original_gqcnn_rank contains non-numeric values")
    result["original_gqcnn_rank"] = result["original_gqcnn_rank"].astype(int)
    result["q_raw"] = pd.to_numeric(result["q_raw"], errors="coerce")
    if not np.all(np.isfinite(result["q_raw"].to_numpy(float))):
        raise GalleryError("q_raw contains non-finite values")
    result["candidate_positive"] = result["candidate_positive"].astype(bool)
    for name, group in result.groupby("sample_id", sort=False):
        ranks = np.sort(group["original_gqcnn_rank"].to_numpy(int))
        if not np.array_equal(ranks, np.arange(1, len(group) + 1)):
            raise GalleryError(f"{name}: original ranks are not a 1..N permutation")
        if group["scene_id"].nunique() != 1:
            raise GalleryError(f"{name}: multiple scene IDs")
    return result


def _vlm_index(rows: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for source in rows or ():
        row = dict(source)
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in index:
            raise GalleryError("VLM rows require unique non-empty sample_id values")
        index[sample_id] = row
    return index


def _derive_failure_category(
    group: pd.DataFrame,
    *,
    original_correct: bool,
    new_correct: bool,
) -> str:
    for column in ("failure_category", "funnel_category"):
        if column in group and group[column].notna().any():
            return str(group.loc[group[column].notna(), column].iloc[0])
    if not original_correct and new_correct:
        return "recovered_by_reranking"
    if original_correct and not new_correct:
        return "reranker_harmful_switch"
    if original_correct and new_correct:
        return "both_correct"
    top5_positive = bool(
        group.loc[
            group["original_gqcnn_rank"] <= 5, "candidate_positive"
        ].any()
    )
    if top5_positive:
        return "ranking_loss_top5"
    if bool(group["candidate_positive"].any()):
        return "ranking_loss_beyond_top5"
    return "all_negative_nms_pool"


def classify_gallery_cases(
    per_candidate: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    method: str,
    vlm_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[GalleryCase]:
    """Classify every comparable sample without cherry-picked ordering."""

    candidates = _validated_candidates(per_candidate)
    canonical = _canonical_predictions(predictions)
    method_rows = canonical.loc[canonical["reranker_method"].eq(str(method))].copy()
    if method_rows.empty:
        available = sorted(canonical["reranker_method"].unique())
        raise GalleryError(f"method {method!r} not found; available={available}")
    candidate_keys = set(
        zip(candidates["sample_id"], candidates["candidate_id"], strict=False)
    )
    prediction_keys = set(
        zip(method_rows["sample_id"], method_rows["candidate_id"], strict=False)
    )
    unknown = sorted(prediction_keys - candidate_keys)
    if unknown:
        raise GalleryError(f"predictions reference unknown candidates: {unknown[:3]}")
    vlm = _vlm_index(vlm_rows)
    top1 = method_rows.loc[method_rows["reranker_rank"].eq(1)]
    if top1["sample_id"].duplicated().any():
        raise GalleryError("method has multiple rank-1 candidates for a sample")
    top1_by_sample = {
        str(row.sample_id): str(row.candidate_id)
        for row in top1.itertuples(index=False)
    }
    cases: list[GalleryCase] = []
    for sample_id, group in candidates.groupby("sample_id", sort=False):
        if sample_id not in top1_by_sample:
            continue
        old_row = group.loc[group["original_gqcnn_rank"].eq(1)]
        if len(old_row) != 1:
            raise GalleryError(f"{sample_id}: expected exactly one original Top-1")
        old = old_row.iloc[0]
        new_id = top1_by_sample[sample_id]
        new_rows = group.loc[group["candidate_id"].eq(new_id)]
        if len(new_rows) != 1:
            raise GalleryError(f"{sample_id}: selected candidate is not unique")
        new = new_rows.iloc[0]
        old_correct = bool(old["candidate_positive"])
        new_correct = bool(new["candidate_positive"])
        categories: list[str] = []
        if not old_correct and new_correct:
            categories.append("recovered")
        if old_correct and not new_correct:
            categories.append("harmful")
        if not old_correct and not new_correct:
            categories.append("both_wrong")
        vlm_row = vlm.get(str(sample_id))
        if vlm_row is not None:
            if bool(vlm_row.get("fallback")) or bool(vlm_row.get("abstain")):
                categories.append("vlm_fallback")
            vlm_selected = str(vlm_row.get("selected_candidate_id") or "")
            if (
                vlm_selected
                and not bool(vlm_row.get("fallback"))
                and not bool(vlm_row.get("abstain"))
                and vlm_selected != new_id
            ):
                categories.append("vlm_learned_disagreement")
        failure = _derive_failure_category(
            group,
            original_correct=old_correct,
            new_correct=new_correct,
        )
        for category in categories:
            cases.append(
                GalleryCase(
                    sample_id=str(sample_id),
                    scene_id=str(old["scene_id"]),
                    category=category,
                    original_candidate_id=str(old["candidate_id"]),
                    new_candidate_id=new_id,
                    original_correct=old_correct,
                    new_correct=new_correct,
                    failure_category=failure,
                    vlm_result=vlm_row,
                )
            )
    return cases


def select_gallery_cases(
    cases: Sequence[GalleryCase],
    *,
    quotas: Mapping[str, int] = DEFAULT_GALLERY_QUOTAS,
    strict_quotas: bool = True,
) -> tuple[list[GalleryCase], dict[str, Any]]:
    """Select deterministic per-category prefixes and report every shortfall."""

    unknown = sorted(set(quotas) - set(DEFAULT_GALLERY_QUOTAS))
    if unknown:
        raise GalleryError(f"unknown gallery categories: {unknown}")
    negative = {key: value for key, value in quotas.items() if int(value) < 0}
    if negative:
        raise GalleryError(f"gallery quotas cannot be negative: {negative}")
    by_category: dict[str, list[GalleryCase]] = {
        category: [] for category in quotas
    }
    for case in cases:
        if case.category in by_category:
            by_category[case.category].append(case)
    selected: list[GalleryCase] = []
    counts: dict[str, dict[str, int]] = {}
    shortfalls: dict[str, int] = {}
    for category, requested_value in quotas.items():
        requested = int(requested_value)
        available = sorted(
            by_category[category],
            key=lambda item: (item.scene_id, item.sample_id),
        )
        chosen = available[:requested]
        selected.extend(chosen)
        counts[category] = {
            "requested": requested,
            "available": len(available),
            "selected": len(chosen),
        }
        if len(chosen) < requested:
            shortfalls[category] = requested - len(chosen)
    if strict_quotas and shortfalls:
        raise GalleryError(f"gallery quota shortfall: {shortfalls}")
    return selected, {
        "schema_version": SCHEMA_VERSION,
        "selection_order": "scene_id ascending, then sample_id ascending",
        "counts": counts,
        "shortfalls": shortfalls,
        "strict_quotas": bool(strict_quotas),
    }


def _load_rgb(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("RGB"))
    if value.ndim != 3 or value.shape[2] != 3:
        raise GalleryError(f"invalid RGB image: {path}")
    return value


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L")) > 0
    if value.shape != shape:
        raise GalleryError(
            f"predicted mask shape {value.shape} does not match RGB {shape}: {path}"
        )
    return value


def _resolve_root_image(root: Path, sample_id: str) -> Path:
    suffix_prefix = sample_id.rsplit("_", 1)[-1][:2]
    candidates = (
        root / f"{sample_id}.png",
        root / suffix_prefix / f"{sample_id}.png",
        root / sample_id / "rgb.png",
        root / sample_id / "hifics_mask_processed.png",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise GalleryError(
        f"cannot resolve {sample_id} below {root}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def _candidate_payload(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise GalleryError(f"candidate file has no list: {path}")
    result = {str(row["candidate_id"]): dict(row) for row in rows}
    if len(result) != len(rows):
        raise GalleryError(f"duplicate candidate IDs: {path}")
    return result


def _safe_candidates(
    group: pd.DataFrame,
    predictions: pd.DataFrame,
    geometry: Mapping[str, Mapping[str, Any]],
    *,
    method: str,
) -> list[InferenceCandidate]:
    method_rows = predictions.loc[
        predictions["reranker_method"].eq(method)
        & predictions["sample_id"].eq(str(group["sample_id"].iloc[0]))
    ]
    prediction_by_id = {
        str(row.candidate_id): row for row in method_rows.itertuples(index=False)
    }
    result: list[InferenceCandidate] = []
    for row in group.sort_values("original_gqcnn_rank").itertuples(index=False):
        candidate_id = str(row.candidate_id)
        if candidate_id not in geometry:
            raise GalleryError(f"missing geometry for candidate {candidate_id}")
        prediction = prediction_by_id.get(candidate_id)
        score = None
        rank = None
        if prediction is not None:
            rank = int(prediction.reranker_rank)
            value = float(prediction.reranker_score)
            score = value if math.isfinite(value) else None
        result.append(
            InferenceCandidate(
                candidate_id=candidate_id,
                original_rank=int(row.original_gqcnn_rank),
                q_score=float(row.q_raw),
                reranker_rank=rank,
                reranker_score=score,
                geometry=geometry[candidate_id],
            )
        )
    return result


def _evaluation_candidates(group: pd.DataFrame) -> dict[str, EvaluationCandidate]:
    result: dict[str, EvaluationCandidate] = {}
    for row in group.itertuples(index=False):
        best_gt = getattr(row, "best_gt_id", None)
        result[str(row.candidate_id)] = EvaluationCandidate(
            candidate_id=str(row.candidate_id),
            positive=bool(row.candidate_positive),
            iou=float(row.candidate_gt_iou),
            angle_error_deg=float(row.candidate_gt_angle_error_deg),
            best_gt_id=None
            if best_gt is None or pd.isna(best_gt)
            else int(best_gt),
        )
    return result


def _panel_base(rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(rgb).resize(
        (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.BILINEAR
    )
    return image


def _scaled_geometry(
    candidate: InferenceCandidate,
    source_shape: tuple[int, int],
) -> dict[str, Any]:
    source_h, source_w = source_shape
    geometry = dict(candidate.geometry)
    geometry["center_u_px"] = (
        float(geometry["center_u_px"]) * PANEL_WIDTH / source_w
    )
    geometry["center_v_px"] = (
        float(geometry["center_v_px"]) * PANEL_HEIGHT / source_h
    )
    geometry["width_px"] = (
        float(geometry["width_px"]) * PANEL_WIDTH / source_w
    )
    geometry["rectangle_height_px"] = 20.0 * PANEL_HEIGHT / source_h
    return geometry


def _title_panel(image: Image.Image, title: str, *, evaluation: bool = False) -> None:
    draw = ImageDraw.Draw(image)
    color = (118, 31, 38) if evaluation else (20, 27, 38)
    draw.rectangle((0, 0, image.width, 32), fill=color)
    draw.text((10, 7), title, fill="white", font=_font(16, bold=True))
    draw.rectangle(
        (0, 0, image.width - 1, image.height - 1),
        outline=(239, 68, 68) if evaluation else (92, 107, 130),
        width=3 if evaluation else 1,
    )


def _draw_candidate_label(
    draw: ImageDraw.ImageDraw,
    candidate: InferenceCandidate,
    *,
    source_shape: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    width: int = 3,
) -> None:
    geometry = _scaled_geometry(candidate, source_shape)
    _draw_candidate(draw, geometry, color=color, width=width)
    x = float(geometry["center_u_px"]) + 4
    y = max(35.0, float(geometry["center_v_px"]) - 18)
    draw.text(
        (x, y),
        label,
        fill="white",
        font=_font(13, bold=True),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


def _render_inference_panels(
    *,
    rgb: np.ndarray,
    predicted_mask: np.ndarray,
    instruction: str,
    candidates: Sequence[InferenceCandidate],
    original_candidate_id: str,
    new_candidate_id: str,
    method: str,
    vlm_result: Mapping[str, Any] | None,
) -> tuple[list[Image.Image], Image.Image]:
    """Render GT-free panels; this function has no GT or label parameters."""

    del instruction  # The caller renders the instruction once in the header.
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    source_shape = rgb.shape[:2]
    rgb_panel = _panel_base(rgb)
    _title_panel(rgb_panel, "RGB (inference input)")

    overlay = rgb.astype(np.float32)
    tint = np.asarray([45.0, 180.0, 196.0])
    overlay[predicted_mask] = 0.67 * overlay[predicted_mask] + 0.33 * tint
    mask_panel = _panel_base(np.clip(overlay, 0, 255).astype(np.uint8))
    mask_draw = ImageDraw.Draw(mask_panel)
    boundary_y, boundary_x = np.nonzero(_mask_boundary(predicted_mask))
    sx = PANEL_WIDTH / rgb.shape[1]
    sy = PANEL_HEIGHT / rgb.shape[0]
    for x, y in zip(boundary_x.tolist(), boundary_y.tolist(), strict=False):
        mask_draw.point((x * sx, y * sy), fill=(255, 235, 59))
    _title_panel(mask_panel, "Predicted HiFi mask (no GT)")

    top5_panel = _panel_base(rgb)
    top5_draw = ImageDraw.Draw(top5_panel)
    top5 = sorted(candidates, key=lambda item: item.original_rank)[:5]
    for index, candidate in enumerate(top5):
        color = NEUTRAL_CANDIDATE_COLORS[index]
        _draw_candidate(
            top5_draw,
            _scaled_geometry(candidate, source_shape),
            color=color,
            width=3,
        )
    # Candidate centres can be only a few pixels apart.  Drawing five text
    # labels at those centres makes the IDs unreadable, so identity stays in a
    # fixed, non-overlapping legend while the rectangles retain the same
    # neutral colors used by the VLM visualization.
    legend_height = 17 * len(top5) + 9
    legend_top = PANEL_HEIGHT - legend_height
    top5_draw.rectangle(
        (0, legend_top, PANEL_WIDTH, PANEL_HEIGHT), fill=(10, 13, 18)
    )
    for index, candidate in enumerate(top5):
        color = NEUTRAL_CANDIDATE_COLORS[index]
        y = legend_top + 5 + index * 17
        top5_draw.rectangle((9, y + 2, 21, y + 13), fill=color)
        top5_draw.text(
            (29, y),
            (
                f"r{candidate.original_rank}  {candidate.candidate_id}"
                f"  q={candidate.q_score:.6g}"
            ),
            fill=(240, 243, 247),
            font=_font(13),
        )
    _title_panel(top5_panel, "Original GQ-CNN Top-5 (q order)")

    comparison = _panel_base(rgb)
    comparison_draw = ImageDraw.Draw(comparison)
    old = by_id[original_candidate_id]
    new = by_id[new_candidate_id]
    _draw_candidate_label(
        comparison_draw,
        old,
        source_shape=source_shape,
        color=(255, 76, 92),
        label=f"OLD {old.candidate_id}",
        width=4,
    )
    _draw_candidate_label(
        comparison_draw,
        new,
        source_shape=source_shape,
        color=(54, 211, 153),
        label=f"NEW {new.candidate_id}",
        width=4,
    )
    _title_panel(comparison, "Inference choice: original vs reranker")

    audit = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), (19, 25, 35))
    draw = ImageDraw.Draw(audit)
    _title_panel(audit, "Inference audit (GT-free)")
    top_lines = [
        f"method: {method}",
        f"original Top-1: {old.candidate_id}   q={old.q_score:.6g}",
        (
            f"new Top-1: {new.candidate_id}   q={new.q_score:.6g}   "
            f"score={new.reranker_score:.6g}"
            if new.reranker_score is not None
            else f"new Top-1: {new.candidate_id}   q={new.q_score:.6g}   score=n/a"
        ),
        "",
        "Original Top-5:",
    ]
    for candidate in top5:
        score = (
            "n/a"
            if candidate.reranker_score is None
            else f"{candidate.reranker_score:.5g}"
        )
        top_lines.append(
            f"  r{candidate.original_rank} {candidate.candidate_id}"
            f"  q={candidate.q_score:.5g}  rerank={score}"
        )
    top_lines.extend(["", "VLM ranking:"])
    if vlm_result is None:
        top_lines.append("  not available")
    else:
        ranking = vlm_result.get("ranking") or []
        if ranking:
            for index, item in enumerate(ranking[:5], start=1):
                top_lines.append(
                    f"  {index}. {item.get('candidate_id', '?')}"
                    f"  score={item.get('score', 'n/a')}"
                )
        else:
            top_lines.append("  no parsed ranking")
        if bool(vlm_result.get("fallback")) or bool(vlm_result.get("abstain")):
            top_lines.append(
                "  FALLBACK: "
                + str(vlm_result.get("fallback_reason") or "abstain")
            )
    y = 42
    for line in top_lines:
        draw.text((12, y), line, fill=(225, 231, 239), font=_font(14))
        y += 20
    return [rgb_panel, mask_panel, top5_panel, comparison], audit


def _draw_gt_geometry(
    draw: ImageDraw.ImageDraw,
    *,
    grasps: Sequence[Any],
    evaluation_config: Mapping[str, Any],
    source_shape: tuple[int, int],
) -> None:
    source_h, source_w = source_shape
    sx = PANEL_WIDTH / source_w
    sy = PANEL_HEIGHT / source_h
    for index, geometry in enumerate(
        make_ocid_vlg_evaluation_rectangles(grasps, evaluation_config),
        start=1,
    ):
        polygon = [
            (float(point[0]) * sx, float(point[1]) * sy)
            for point in geometry["polygon"]
        ]
        draw.line(polygon + [polygon[0]], fill=GT_COLOR, width=3, joint="curve")
        center = geometry["center_uv"]
        draw.text(
            (float(center[0]) * sx + 3, float(center[1]) * sy + 3),
            f"GT{index}",
            fill=GT_COLOR,
            font=_font(13, bold=True),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )


def _render_evaluation_panel(
    *,
    rgb: np.ndarray,
    candidates: Sequence[InferenceCandidate],
    labels: Mapping[str, EvaluationCandidate],
    original_candidate_id: str,
    new_candidate_id: str,
    grasps: Sequence[Any],
    evaluation_config: Mapping[str, Any],
    failure_category: str,
) -> Image.Image:
    """Render the only panel allowed to receive GT and correctness labels."""

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    panel = _panel_base(rgb)
    draw = ImageDraw.Draw(panel)
    _draw_gt_geometry(
        draw,
        grasps=grasps,
        evaluation_config=evaluation_config,
        source_shape=rgb.shape[:2],
    )
    _draw_candidate_label(
        draw,
        by_id[original_candidate_id],
        source_shape=rgb.shape[:2],
        color=(255, 76, 92),
        label=f"OLD {original_candidate_id}",
        width=4,
    )
    _draw_candidate_label(
        draw,
        by_id[new_candidate_id],
        source_shape=rgb.shape[:2],
        color=(54, 211, 153),
        label=f"NEW {new_candidate_id}",
        width=4,
    )
    _title_panel(panel, "EVALUATION ONLY — cyan is GT", evaluation=True)
    old = labels[original_candidate_id]
    new = labels[new_candidate_id]
    angle_threshold = float(evaluation_config["angle_threshold_deg"])
    iou_threshold = float(evaluation_config["iou_threshold"])
    lines = [
        f"failure: {failure_category}",
        (
            f"OLD {old.candidate_id}: {'PASS' if old.positive else 'FAIL'} "
            f"IoU={old.iou:.3f} angle={old.angle_error_deg:.1f}°"
        ),
        (
            f"NEW {new.candidate_id}: {'PASS' if new.positive else 'FAIL'} "
            f"IoU={new.iou:.3f} angle={new.angle_error_deg:.1f}°"
        ),
        (
            f"rule: angle ≤ {angle_threshold:g}° and rectangle "
            f"IoU > {iou_threshold:g}"
        ),
    ]
    y = PANEL_HEIGHT - 82
    draw.rectangle((0, y - 5, PANEL_WIDTH, PANEL_HEIGHT), fill=(15, 17, 22))
    for line in lines:
        draw.text((8, y), line, fill=(244, 246, 250), font=_font(13))
        y += 19
    return panel


def render_gallery_case(
    *,
    case: GalleryCase,
    instruction: str,
    rgb: np.ndarray,
    predicted_mask: np.ndarray,
    candidates: Sequence[InferenceCandidate],
    labels: Mapping[str, EvaluationCandidate],
    grasps: Sequence[Any],
    evaluation_config: Mapping[str, Any],
    method: str,
    output_path: Path,
) -> Path:
    """Render one six-panel case PNG with a hard inference/evaluation boundary."""

    inference_panels, audit_panel = _render_inference_panels(
        rgb=rgb,
        predicted_mask=predicted_mask,
        instruction=instruction,
        candidates=candidates,
        original_candidate_id=case.original_candidate_id,
        new_candidate_id=case.new_candidate_id,
        method=method,
        vlm_result=case.vlm_result,
    )
    evaluation_panel = _render_evaluation_panel(
        rgb=rgb,
        candidates=candidates,
        labels=labels,
        original_candidate_id=case.original_candidate_id,
        new_candidate_id=case.new_candidate_id,
        grasps=grasps,
        evaluation_config=evaluation_config,
        failure_category=case.failure_category,
    )
    width = 3 * PANEL_WIDTH + 2 * PANEL_GAP + 2 * CANVAS_MARGIN
    height = (
        HEADER_HEIGHT
        + 2 * INFERENCE_BANNER_HEIGHT
        + 2 * PANEL_HEIGHT
        + PANEL_GAP
        + 2 * CANVAS_MARGIN
    )
    canvas = Image.new("RGB", (width, height), (11, 15, 22))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (CANVAS_MARGIN, 16),
        f"Case  ·  {case.sample_id}",
        fill=(247, 250, 252),
        font=_font(25, bold=True),
    )
    wrapped = textwrap.wrap(instruction, width=105) or [""]
    draw.text(
        (CANVAS_MARGIN, 52),
        "Instruction: " + wrapped[0],
        fill=(215, 224, 235),
        font=_font(18),
    )
    if len(wrapped) > 1:
        draw.text(
            (CANVAS_MARGIN + 102, 76),
            wrapped[1],
            fill=(215, 224, 235),
            font=_font(18),
        )
    banner_y = HEADER_HEIGHT
    draw.rectangle(
        (CANVAS_MARGIN, banner_y, width - CANVAS_MARGIN, banner_y + 34),
        fill=(22, 78, 99),
    )
    draw.text(
        (CANVAS_MARGIN + 10, banner_y + 8),
        "INFERENCE-ONLY PANELS — ground truth and labels are not present",
        fill="white",
        font=_font(16, bold=True),
    )
    top_y = banner_y + INFERENCE_BANNER_HEIGHT
    for index, panel in enumerate(inference_panels[:3]):
        x = CANVAS_MARGIN + index * (PANEL_WIDTH + PANEL_GAP)
        canvas.paste(panel, (x, top_y))

    second_banner_y = top_y + PANEL_HEIGHT + PANEL_GAP
    draw.rectangle(
        (
            CANVAS_MARGIN,
            second_banner_y,
            width - CANVAS_MARGIN,
            second_banner_y + 34,
        ),
        fill=(57, 65, 79),
    )
    draw.text(
        (CANVAS_MARGIN + 10, second_banner_y + 8),
        "LEFT + CENTRE: inference evidence     |     RIGHT: post-hoc evaluation only",
        fill="white",
        font=_font(16, bold=True),
    )
    bottom_y = second_banner_y + INFERENCE_BANNER_HEIGHT
    canvas.paste(inference_panels[3], (CANVAS_MARGIN, bottom_y))
    canvas.paste(
        audit_panel, (CANVAS_MARGIN + PANEL_WIDTH + PANEL_GAP, bottom_y)
    )
    canvas.paste(
        evaluation_panel,
        (CANVAS_MARGIN + 2 * (PANEL_WIDTH + PANEL_GAP), bottom_y),
    )
    _atomic_save_png(canvas, output_path)
    return output_path


def _annotation_index(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise GalleryError("annotation file has no data list")
    result = {int(row["question_index"]): dict(row) for row in rows}
    if len(result) != len(rows):
        raise GalleryError("duplicate annotation question_index values")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_index(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    summary: Mapping[str, Any],
    output_path: Path,
) -> None:
    cards: list[str] = []
    for row in manifest_rows:
        category = str(row["category"])
        cards.append(
            f"""
<article class="card" data-category="{html.escape(category)}">
  <div class="meta"><span class="pill">{html.escape(CATEGORY_LABELS[category])}</span>
  <code>{html.escape(str(row["sample_id"]))}</code></div>
  <a href="{html.escape(str(row["image_path"]))}">
    <img loading="lazy" src="{html.escape(str(row["image_path"]))}"
         alt="{html.escape(CATEGORY_LABELS[category])}: {html.escape(str(row["sample_id"]))}">
  </a>
  <p>{html.escape(str(row["instruction"]))}</p>
  <p class="small">old <code>{html.escape(str(row["original_candidate_id"]))}</code>
  → new <code>{html.escape(str(row["new_candidate_id"]))}</code> ·
  {html.escape(str(row["failure_category"]))}</p>
  <a class="details" href="{html.escape(str(row["case_json_path"]))}">machine-readable case</a>
</article>"""
        )
    count_items = "".join(
        "<li><b>"
        + html.escape(CATEGORY_LABELS[key])
        + "</b>: "
        + str(value["selected"])
        + " selected / "
        + str(value["available"])
        + " available / "
        + str(value["requested"])
        + " requested</li>"
        for key, value in summary["counts"].items()
    )
    buttons = "".join(
        f'<button data-filter="{html.escape(category)}">'
        f"{html.escape(CATEGORY_LABELS[category])}</button>"
        for category in summary["counts"]
    )
    markup = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Modular re-ranking failure gallery</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #0b0f16; color: #eef2f7; }}
header {{ position: sticky; top: 0; z-index: 2; padding: 18px 24px;
  background: rgba(11,15,22,.96); border-bottom: 1px solid #334155; }}
h1 {{ margin: 0 0 8px; font-size: 1.5rem; }}
.note {{ color: #aebdce; max-width: 1100px; }}
button {{ margin: 4px; padding: 7px 11px; border: 1px solid #64748b;
  border-radius: 999px; background: #182233; color: #eef2f7; cursor: pointer; }}
button.active {{ background: #0e7490; }}
.summary {{ margin: 18px 24px; color: #cbd5e1; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(430px,1fr));
  gap: 18px; padding: 0 24px 36px; }}
.card {{ background: #111827; border: 1px solid #334155; border-radius: 12px;
  padding: 12px; }}
.card img {{ width: 100%; height: auto; border-radius: 7px; background: #030712; }}
.meta {{ display: flex; justify-content: space-between; gap: 8px; margin-bottom: 8px; }}
.pill {{ background: #164e63; border-radius: 999px; padding: 4px 9px; }}
.small, .details {{ color: #aebdce; font-size: .88rem; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<header>
  <h1>Frozen-candidate failure gallery · {html.escape(method)}</h1>
  <div class="note">The top and lower-left/centre panels are inference-only.
  Cyan ground truth, IoU, angle error, and pass/fail labels appear only inside
  the red-bordered EVALUATION ONLY panel.</div>
  <div><button class="active" data-filter="all">All</button>{buttons}</div>
</header>
<section class="summary"><ul>{count_items}</ul></section>
<main class="grid">{"".join(cards)}</main>
<script>
for (const button of document.querySelectorAll("button[data-filter]")) {{
  button.addEventListener("click", () => {{
    const category = button.dataset.filter;
    for (const card of document.querySelectorAll(".card")) {{
      card.hidden = category !== "all" && card.dataset.category !== category;
    }}
    for (const other of document.querySelectorAll("button")) other.classList.remove("active");
    button.classList.add("active");
  }});
}}
</script>
</body>
</html>
"""
    _atomic_write_text(output_path, markup)


def generate_failure_gallery(
    *,
    per_candidate_path: Path,
    per_sample_path: Path,
    predictions_path: Path,
    method: str,
    hifi_manifest_path: Path,
    candidate_root: Path,
    annotation_file: Path,
    evaluation_config_path: Path,
    output_dir: Path,
    vlm_results_path: Path | None = None,
    rgb_root: Path | None = None,
    predicted_mask_root: Path | None = None,
    quotas: Mapping[str, int] = DEFAULT_GALLERY_QUOTAS,
    strict_quotas: bool = True,
) -> dict[str, Any]:
    """Select, render, and index an auditable qualitative gallery."""

    candidates = _validated_candidates(pd.read_parquet(per_candidate_path))
    per_sample = pd.read_parquet(per_sample_path)
    if "sample_id" not in per_sample or per_sample["sample_id"].duplicated().any():
        raise GalleryError("per-sample table requires unique sample_id values")
    missing_sample_rows = sorted(
        set(candidates["sample_id"]) - set(per_sample["sample_id"].astype(str))
    )
    if missing_sample_rows:
        raise GalleryError(
            f"per-sample table is missing candidate samples: {missing_sample_rows[:3]}"
        )
    predictions = _canonical_predictions(pd.read_parquet(predictions_path))
    vlm_rows = _read_jsonl(vlm_results_path) if vlm_results_path else []
    classified = classify_gallery_cases(
        candidates,
        predictions,
        method=method,
        vlm_rows=vlm_rows,
    )
    selected, selection_summary = select_gallery_cases(
        classified, quotas=quotas, strict_quotas=strict_quotas
    )
    hifi_rows = _read_jsonl(hifi_manifest_path)
    hifi_index = {str(row["sample_id"]): row for row in hifi_rows}
    if len(hifi_index) != len(hifi_rows):
        raise GalleryError("HiFi manifest contains duplicate sample IDs")
    annotations = _annotation_index(annotation_file)
    evaluation_config = YAML(typ="safe").load(
        evaluation_config_path.read_text(encoding="utf-8")
    )
    destination = Path(output_dir)
    case_rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(selected, start=1):
        hifi = hifi_index.get(case.sample_id)
        if hifi is None:
            raise GalleryError(f"HiFi manifest missing {case.sample_id}")
        rgb_path = (
            _resolve_root_image(rgb_root, case.sample_id)
            if rgb_root is not None
            else Path(hifi["source_rgb_path"])
        )
        mask_path = (
            _resolve_root_image(predicted_mask_root, case.sample_id)
            if predicted_mask_root is not None
            else Path(hifi["native_mask_path"])
        )
        rgb = _load_rgb(rgb_path)
        predicted_mask = _load_mask(mask_path, rgb.shape[:2])
        group = candidates.loc[candidates["sample_id"].eq(case.sample_id)]
        geometry = _candidate_payload(
            candidate_root / case.sample_id / "candidates.json"
        )
        safe = _safe_candidates(group, predictions, geometry, method=method)
        labels = _evaluation_candidates(group)
        question_index = int(hifi["question_index"])
        annotation = annotations.get(question_index)
        if annotation is None:
            raise GalleryError(f"annotation missing question {question_index}")
        instruction = str(hifi.get("query") or hifi.get("instruction") or "")
        if (
            annotation.get("image_filename") not in {None, case.scene_id}
            or annotation.get("question") not in {None, instruction}
        ):
            raise GalleryError(f"{case.sample_id}: annotation identity mismatch")
        grasps = annotation.get("grasps")
        if not isinstance(grasps, list) or not grasps:
            raise GalleryError(f"{case.sample_id}: annotation has no GT grasps")
        case_name = f"{case_index:03d}_{case.sample_id}"
        case_dir = destination / "cases" / case.category
        image_path = case_dir / f"{case_name}.png"
        render_gallery_case(
            case=case,
            instruction=instruction,
            rgb=rgb,
            predicted_mask=predicted_mask,
            candidates=safe,
            labels=labels,
            grasps=grasps,
            evaluation_config=evaluation_config,
            method=method,
            output_path=image_path,
        )
        safe_by_id = {item.candidate_id: item for item in safe}
        label_by_id = labels
        top5 = sorted(safe, key=lambda item: item.original_rank)[:5]
        inference_payload = {
            "gt_fields_included": False,
            "instruction": instruction,
            "rgb_path": str(rgb_path),
            "predicted_mask_path": str(mask_path),
            "method": method,
            "original_top5": [
                {
                    "candidate_id": item.candidate_id,
                    "original_rank": item.original_rank,
                    "q_score": item.q_score,
                    "reranker_rank": item.reranker_rank,
                    "reranker_score": item.reranker_score,
                }
                for item in top5
            ],
            "original_top1_candidate_id": case.original_candidate_id,
            "new_top1_candidate_id": case.new_candidate_id,
            "vlm_ranking": None
            if case.vlm_result is None
            else case.vlm_result.get("ranking"),
            "vlm_fallback": None
            if case.vlm_result is None
            else bool(case.vlm_result.get("fallback")),
            "vlm_fallback_reason": None
            if case.vlm_result is None
            else case.vlm_result.get("fallback_reason"),
        }
        evaluation_payload = {
            "evaluation_only": True,
            "failure_category": case.failure_category,
            "metric_rule": (
                "angle <= "
                f"{float(evaluation_config['angle_threshold_deg']):g} deg and "
                "rectangle IoU > "
                f"{float(evaluation_config['iou_threshold']):g}"
            ),
            "ground_truth_rectangle_count": len(grasps),
            "original_top1": label_by_id[case.original_candidate_id].__dict__,
            "new_top1": label_by_id[case.new_candidate_id].__dict__,
        }
        case_json_path = case_dir / f"{case_name}.json"
        case_payload = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": case.sample_id,
            "scene_id": case.scene_id,
            "category": case.category,
            "inference": inference_payload,
            "evaluation": evaluation_payload,
            "image_sha256": _sha256_file(image_path),
            "candidate_geometry_source": str(
                candidate_root / case.sample_id / "candidates.json"
            ),
            "selected_candidate_geometry_present": (
                case.original_candidate_id in safe_by_id
                and case.new_candidate_id in safe_by_id
            ),
        }
        _atomic_write_text(
            case_json_path,
            json.dumps(case_payload, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
        )
        case_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": case.sample_id,
                "scene_id": case.scene_id,
                "category": case.category,
                "instruction": instruction,
                "original_candidate_id": case.original_candidate_id,
                "new_candidate_id": case.new_candidate_id,
                "original_correct": case.original_correct,
                "new_correct": case.new_correct,
                "failure_category": case.failure_category,
                "image_path": str(image_path.relative_to(destination)),
                "case_json_path": str(case_json_path.relative_to(destination)),
            }
        )
    manifest_path = destination / "gallery_manifest.jsonl"
    _atomic_write_text(
        manifest_path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in case_rows
        ),
    )
    result = {
        **selection_summary,
        "method": str(method),
        "rendered_case_count": len(case_rows),
        "classified_case_count": len(classified),
        "unique_rendered_sample_count": len(
            {row["sample_id"] for row in case_rows}
        ),
        "gt_isolation": {
            "inference_panel_gt_fields_included": False,
            "evaluation_panel_only": [
                "GT rectangles",
                "candidate correctness",
                "rectangle IoU",
                "angle error",
                "failure category",
            ],
        },
        "input_hashes": {
            "per_candidate": _sha256_file(per_candidate_path),
            "per_sample": _sha256_file(per_sample_path),
            "predictions": _sha256_file(predictions_path),
            "hifi_manifest": _sha256_file(hifi_manifest_path),
            "annotations": _sha256_file(annotation_file),
            "evaluation_config": _sha256_file(evaluation_config_path),
            "vlm_results": None
            if vlm_results_path is None
            else _sha256_file(vlm_results_path),
        },
        "category_render_counts": dict(
            sorted(Counter(row["category"] for row in case_rows).items())
        ),
    }
    summary_path = destination / "gallery_summary.json"
    _atomic_write_text(
        summary_path,
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    index_path = destination / "index.html"
    _render_index(case_rows, method=method, summary=result, output_path=index_path)
    result["index_path"] = str(index_path)
    result["manifest_path"] = str(manifest_path)
    result["summary_path"] = str(summary_path)
    return result


__all__ = [
    "CATEGORY_LABELS",
    "DEFAULT_GALLERY_QUOTAS",
    "GT_COLOR",
    "GalleryCase",
    "GalleryError",
    "InferenceCandidate",
    "EvaluationCandidate",
    "classify_gallery_cases",
    "generate_failure_gallery",
    "render_gallery_case",
    "select_gallery_cases",
]
