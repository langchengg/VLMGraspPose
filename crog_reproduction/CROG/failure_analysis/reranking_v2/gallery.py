from __future__ import annotations

import html
import json
from pathlib import Path
from textwrap import fill
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from failure_analysis.failure_utils import (
    draw_grasps,
    load_gt_mask,
    load_rgb,
    overlay_mask,
    rle_to_mask,
)

from .datasets import load_joined
from .evaluation import load_prediction_rankings
from .schema import atomic_write_json, atomic_write_text, read_jsonl


ORIGINAL_COLOR = (30, 190, 255)
SELECTED_COLOR = (215, 70, 190)
OTHER_COLOR = (245, 185, 45)
GT_GRASP_COLOR = (40, 95, 230)


def _draw_candidate(
    image: np.ndarray,
    candidate: dict[str, Any],
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    result = image.copy()
    polygon = np.asarray(candidate["polygon"], dtype=np.int32)
    cv2.polylines(result, [polygon], True, color, thickness, cv2.LINE_AA)
    return result


def classify_failure_stage(
    raw_prediction: dict[str, Any],
    *,
    original_correct: bool,
    oracle: bool,
) -> str:
    mask_iou = raw_prediction.get("mask_iou")
    if mask_iou is not None and float(mask_iou) < 0.5:
        return "grounding_failure"
    if not oracle:
        return "candidate_set_failure"
    if not original_correct:
        return "ranking_failure"
    return "original_success"


def render_evaluation_case(
    *,
    sample,
    raw_prediction: dict[str, Any],
    candidate_order: list[str],
    output_path: str | Path,
) -> dict[str, Any]:
    rgb = load_rgb(sample.feature["image_path"])
    candidates = sample.feature["candidates"]
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    labels = {
        str(value["candidate_id"]): bool(value["candidate_correct"])
        for value in sample.label["candidate_labels"]
    }
    original_id = str(candidates[0]["candidate_id"])
    selected_id = str(candidate_order[0])
    predicted_mask = rle_to_mask(sample.feature["predicted_mask_rle"])
    prediction_panel = overlay_mask(
        rgb, predicted_mask, (40, 205, 125), alpha=0.32
    )
    for candidate in candidates:
        prediction_panel = _draw_candidate(
            prediction_panel, candidate, OTHER_COLOR, 1
        )
    prediction_panel = _draw_candidate(
        prediction_panel, by_id[original_id], ORIGINAL_COLOR, 4
    )
    prediction_panel = _draw_candidate(
        prediction_panel, by_id[selected_id], SELECTED_COLOR, 3
    )
    gt_mask = load_gt_mask(
        raw_prediction["mask_path"], raw_prediction["obj_id"]
    )
    gt_panel = overlay_mask(rgb, gt_mask, (235, 70, 70), alpha=0.34)
    gt_panel = draw_grasps(
        gt_panel,
        raw_prediction.get("gt_grasps", []),
        GT_GRASP_COLOR,
        thickness=2,
    )
    stage = classify_failure_stage(
        raw_prediction,
        original_correct=labels[original_id],
        oracle=any(labels.values()),
    )
    fig = plt.figure(figsize=(18, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, height_ratios=(2.3, 1.4))
    top_left = fig.add_subplot(grid[0, :3])
    top_right = fig.add_subplot(grid[0, 3:])
    top_left.imshow(prediction_panel)
    top_left.set_title(
        "Prediction view — gold: all, cyan: original Top-1, magenta: selected"
    )
    top_right.imshow(gt_panel)
    top_right.set_title("Evaluation-only GT view — red mask, blue GT grasps")
    for axis in (top_left, top_right):
        axis.axis("off")
    rank_by_id = {
        candidate_id: index + 1
        for index, candidate_id in enumerate(candidate_order)
    }
    for index, candidate in enumerate(candidates):
        axis = fig.add_subplot(grid[1, index])
        half = max(48, int(round(candidate["width_px"] * 1.2)))
        x, y = int(candidate["cx"]), int(candidate["cy"])
        local = _draw_candidate(
            rgb,
            candidate,
            (
                SELECTED_COLOR
                if candidate["candidate_id"] == selected_id
                else (
                    ORIGINAL_COLOR
                    if candidate["candidate_id"] == original_id
                    else OTHER_COLOR
                )
            ),
            4,
        )
        crop = local[
            max(0, y - half) : min(local.shape[0], y + half),
            max(0, x - half) : min(local.shape[1], x + half),
        ]
        axis.imshow(crop)
        roles = []
        if candidate["candidate_id"] == original_id:
            roles.append("original")
        if candidate["candidate_id"] == selected_id:
            roles.append("selected")
        axis.set_title(
            f"{candidate['candidate_id']} → rank {rank_by_id[candidate['candidate_id']]}\n"
            f"Q={candidate['q_raw']:.3f}; correct={labels[candidate['candidate_id']]}\n"
            + ", ".join(roles),
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle(
        fill(sample.feature["language_instruction"], 95)
        + "\n"
        + (
            f"stage={stage}; original_correct={labels[original_id]}; "
            f"selected_correct={labels[selected_id]}; "
            f"Oracle@5={any(labels.values())}; mask_IoU="
            f"{raw_prediction.get('mask_iou', float('nan')):.3f}"
        ),
        fontsize=12,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return {
        "sample_id": sample.sample_id,
        "language_prompt": sample.feature["language_instruction"],
        "stage": stage,
        "original_candidate_id": original_id,
        "selected_candidate_id": selected_id,
        "original_correct": labels[original_id],
        "selected_correct": labels[selected_id],
        "oracle_at_5": any(labels.values()),
        "image_path": str(output_path.resolve()),
    }


def build_failure_gallery(
    *,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    raw_predictions_path: str | Path,
    reranker_predictions_path: str | Path,
    output_dir: str | Path,
    per_group: int = 5,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = load_joined(features_path, legacy_labels_path)
    rankings, _ = load_prediction_rankings(
        reranker_predictions_path, samples
    )
    raw_by_key = {
        (str(row["split"]), int(row["sample_id"])): row
        for row in read_jsonl(raw_predictions_path)
    }
    grouped: dict[str, list] = {
        "recovered": [],
        "harmful": [],
        "grounding_failure": [],
        "candidate_set_failure": [],
        "ranking_failure": [],
    }
    for sample in samples:
        by_id = {
            str(value["candidate_id"]): bool(value["candidate_correct"])
            for value in sample.label["candidate_labels"]
        }
        original = str(sample.feature["candidates"][0]["candidate_id"])
        selected = rankings[sample.sample_id][0]
        raw = raw_by_key[
            (
                str(sample.feature["split"]),
                int(sample.feature["sample_id"]),
            )
        ]
        before, after = by_id[original], by_id[selected]
        if (not before) and after:
            grouped["recovered"].append((sample, raw))
        if before and (not after):
            grouped["harmful"].append((sample, raw))
        stage = classify_failure_stage(
            raw,
            original_correct=before,
            oracle=any(by_id.values()),
        )
        if stage in grouped:
            grouped[stage].append((sample, raw))
    records = []
    for group, cases in grouped.items():
        group_dir = output_dir / group
        for index, (sample, raw) in enumerate(
            sorted(cases, key=lambda value: value[0].sample_id)[
                : int(per_group)
            ]
        ):
            record = render_evaluation_case(
                sample=sample,
                raw_prediction=raw,
                candidate_order=rankings[sample.sample_id],
                output_path=group_dir / f"{index:02d}_{sample.sample_id.replace(':', '_')}.png",
            )
            record["gallery_group"] = group
            records.append(record)
    rows = "\n".join(
        (
            "<article><h2>"
            + html.escape(record["gallery_group"])
            + " — "
            + html.escape(record["sample_id"])
            + "</h2><p>"
            + html.escape(record["language_prompt"])
            + "</p><img loading=\"lazy\" src=\""
            + html.escape(
                str(Path(record["image_path"]).relative_to(output_dir))
            )
            + "\" alt=\"CROG failure-analysis case\"></article>"
        )
        for record in records
    )
    atomic_write_text(
        output_dir / "index.html",
        (
            "<!doctype html><meta charset=\"utf-8\"><title>CROG V2 failure "
            "gallery</title><style>body{font-family:system-ui;max-width:1500px;"
            "margin:auto;padding:24px}img{width:100%;height:auto}article{"
            "margin:0 0 48px}</style><h1>CROG Re-ranking V2 failure gallery</h1>"
            "<p>Ground truth appears only in the clearly labelled evaluation "
            "panel.</p>"
            + rows
        ),
    )
    result = {
        "case_count": len(records),
        "requested_per_group": int(per_group),
        "available_counts": {
            group: len(cases) for group, cases in grouped.items()
        },
        "cases": records,
        "index_html": str((output_dir / "index.html").resolve()),
    }
    atomic_write_json(output_dir / "gallery.json", result)
    return result
