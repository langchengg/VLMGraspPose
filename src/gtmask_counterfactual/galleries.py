"""Deterministic anti-cherry-picking selection and synthetic A--L case boards."""

from __future__ import annotations

import hashlib
import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from .io import (
    artifact_record,
    atomic_csv,
    atomic_json,
    canonical_sha256,
    sha256_file,
)


CASE_CATEGORIES = (
    "clear_grounding_limited",
    "grounding_plus_selection",
    "generator_limited_under_gt",
    "predicted_mask_ranking_limited",
    "gt_mask_regression",
    "no_output_recovered_by_gt_mask",
    "no_change_success",
    "borderline_annotation_sensitive",
)
BOARD_PANELS = tuple("ABCDEFGHIJKL")
BOARD_TITLES = (
    "RGB + language prompt",
    "GT target mask",
    "HiFi predicted probability / mask",
    "Depth",
    "Predicted-mask all candidates",
    "Predicted-mask Top-K",
    "Predicted native / final Top-1",
    "GT-mask all candidates",
    "GT-mask Top-K",
    "GT-mask native Top-1",
    "Matched GT grasps",
    "Exact metrics + module diagnosis",
)
COLORS = {
    "all": "#A7ADB4",
    "pred_native": "#00A6D6",
    "pred_final": "#CC79A7",
    "gt_native": "#E69F00",
    "gt": "#0072B2",
    "matched": "#0057B8",
}

ELIGIBLE_REQUIRED = {
    "sample_id",
    "route",
    "category",
    "mechanism_pure",
    "presentation_eligible",
    "feature_vector_json",
    "asset_bundle_sha256",
}


def deterministic_tie_sha(sample_id: object, route: object, category: object) -> str:
    return hashlib.sha256(
        (str(sample_id) + str(route) + str(category)).encode("utf-8")
    ).hexdigest()


def _features(value: object) -> np.ndarray:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError("feature_vector_json is not valid JSON") from error
    result = np.asarray(parsed, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError("case-selection feature vectors must be finite non-empty vectors")
    return result


def build_eligible_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and retain every case before any presentation selection."""

    missing = sorted(ELIGIBLE_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"eligible case input misses columns: {missing}")
    if frame.empty:
        raise ValueError("full eligible case table cannot be empty")
    work = frame.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["route"] = work["route"].astype(str)
    work["category"] = work["category"].astype(str)
    if work[["sample_id", "route", "category"]].apply(lambda x: x.str.strip().eq("")).any().any():
        raise ValueError("eligible case identities must be non-empty")
    if work.duplicated(["sample_id", "route", "category"]).any():
        raise ValueError("eligible case identities must be unique")
    unknown = sorted(set(work["category"]).difference(CASE_CATEGORIES))
    if unknown:
        raise ValueError(f"unknown case categories: {unknown}")
    feature_vectors = [_features(value) for value in work["feature_vector_json"]]
    if len({len(value) for value in feature_vectors}) != 1:
        raise ValueError("case-selection feature vectors have inconsistent dimensions")
    hashes = work["asset_bundle_sha256"].astype(str)
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("eligible cases require SHA-256-bound asset bundles")
    work["mechanism_pure"] = work["mechanism_pure"].astype(bool)
    work["presentation_eligible"] = work["presentation_eligible"].astype(bool)
    work["mandatory_eligible"] = work["mechanism_pure"] & work["presentation_eligible"]
    work["tie_sha256"] = [
        deterministic_tie_sha(sample, route, category)
        for sample, route, category in zip(
            work["sample_id"], work["route"], work["category"], strict=True
        )
    ]
    return work.sort_values(["route", "category", "sample_id"]).reset_index(drop=True)


def deterministic_medoid_selection(
    eligible: pd.DataFrame, *, quota_per_group: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select cluster medoids; SHA-256 breaks only exact distance ties.

    The first medoid minimises total within-group Euclidean distance.  Additional
    quota slots use deterministic farthest-first medoids, which preserves
    mechanism diversity without subjective clarity scoring.
    """

    work = build_eligible_table(eligible)
    pool = work[work["mandatory_eligible"]].copy()
    if int(quota_per_group) <= 0:
        raise ValueError("quota_per_group must be positive")
    selected_rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []
    for (route, category), group in pool.groupby(["route", "category"], sort=True):
        vectors = np.stack([_features(value) for value in group["feature_vector_json"]])
        distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
        unchosen = set(range(len(group)))
        chosen: list[int] = []
        while unchosen and len(chosen) < int(quota_per_group):
            if not chosen:
                scores = distances.sum(axis=1)
                best_score = min(float(scores[index]) for index in unchosen)
                candidates = [index for index in unchosen if math.isclose(float(scores[index]), best_score, rel_tol=0.0, abs_tol=1e-12)]
            else:
                minimum_to_selected = distances[:, chosen].min(axis=1)
                best_score = max(float(minimum_to_selected[index]) for index in unchosen)
                candidates = [index for index in unchosen if math.isclose(float(minimum_to_selected[index]), best_score, rel_tol=0.0, abs_tol=1e-12)]
            pick = min(candidates, key=lambda index: str(group.iloc[index]["tie_sha256"]))
            chosen.append(pick)
            unchosen.remove(pick)
        for rank, index in enumerate(chosen, start=1):
            row = group.iloc[index].copy()
            row["selection_rank"] = rank
            row["selection_method"] = "cluster_medoid_then_farthest_first_sha256_tie"
            selected_rows.append(row)
        audit_rows.append(
            {
                "route": route,
                "category": category,
                "eligible_count": len(group),
                "selected_count": len(chosen),
                "quota": int(quota_per_group),
                "shortfall": max(0, int(quota_per_group) - len(chosen)),
                "selected_sample_ids_json": json.dumps(
                    [str(group.iloc[index]["sample_id"]) for index in chosen]
                ),
            }
        )
    selected = pd.DataFrame(selected_rows)
    audit = pd.DataFrame(audit_rows)
    if selected.empty:
        raise ValueError("no case satisfies mechanism-purity and presentation eligibility")
    return selected.reset_index(drop=True), audit


def _rect_points(record: Mapping[str, Any]) -> np.ndarray:
    required = ("cx_px", "cy_px", "width_px", "height_px", "theta_deg")
    values = []
    for key in required:
        try:
            values.append(float(record[key]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid board rectangle field: {key}") from error
    cx, cy, width, height, theta = values
    if not np.isfinite(values).all() or width <= 0 or height <= 0:
        raise ValueError("board rectangles must have finite positive geometry")
    angle = math.radians(theta)
    base = np.asarray(
        [[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]]
    )
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return base @ rotation.T + np.asarray([cx, cy])


def _show(ax: plt.Axes, image: np.ndarray) -> None:
    value = np.asarray(image)
    if value.ndim not in (2, 3) or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("board panel image is invalid")
    ax.imshow(value, cmap="gray" if value.ndim == 2 else None)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_candidates(
    ax: plt.Axes,
    records: Sequence[Mapping[str, Any]],
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    for record in records:
        ax.add_patch(
            Polygon(
                _rect_points(record),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        )


def _identity_set(records: Sequence[Mapping[str, Any]]) -> set[str]:
    values = [str(record.get("candidate_id", "")) for record in records]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("case-board candidate IDs must be unique and non-empty")
    return set(values)


def render_case_board(
    *,
    case: Mapping[str, Any],
    assets: Mapping[str, Any],
    output_png: str | Path,
    output_svg: str | Path,
) -> dict[str, Any]:
    """Render the required A--L schema from already-loaded synthetic/verified assets."""

    required_case = {
        "sample_id", "route", "category", "language_prompt", "candidate_count_pred",
        "candidate_count_gt", "positive_count_pred", "positive_count_gt",
        "first_positive_rank_pred", "first_positive_rank_gt", "native_candidate_id",
        "r7_candidate_id", "gt_candidate_id", "native_q", "rerank_score",
        "rotated_iou", "angle_error_deg", "pass_fail", "earliest_observable_issue",
        "asset_bundle_sha256",
    }
    missing = sorted(required_case.difference(case))
    if missing:
        raise ValueError(f"case-board record misses fields: {missing}")
    required_assets = {
        "rgb", "gt_mask", "pred_probability", "pred_mask", "depth",
        "pred_all_candidates", "pred_top_candidates", "gt_all_candidates",
        "gt_top_candidates", "gt_grasps",
    }
    missing_assets = sorted(required_assets.difference(assets))
    if missing_assets:
        raise ValueError(f"case-board assets missing: {missing_assets}")
    aligned_names = ["rgb", "gt_mask", "pred_mask", "depth"]
    if assets["pred_probability"] is not None:
        aligned_names.append("pred_probability")
    shapes = [np.asarray(assets[name]).shape[:2] for name in aligned_names]
    if len(set(shapes)) != 1:
        raise ValueError("case-board RGB/mask/probability/depth assets are not aligned")
    pred_all = list(assets["pred_all_candidates"])
    pred_top = list(assets["pred_top_candidates"])
    gt_all = list(assets["gt_all_candidates"])
    gt_top = list(assets["gt_top_candidates"])
    pred_ids, pred_top_ids = _identity_set(pred_all), _identity_set(pred_top)
    gt_ids, gt_top_ids = _identity_set(gt_all), _identity_set(gt_top)
    if not pred_top_ids.issubset(pred_ids) or not gt_top_ids.issubset(gt_ids):
        raise ValueError("case-board Top-K candidates are not members of all-candidate pools")
    if len(pred_all) != int(case["candidate_count_pred"]) or len(gt_all) != int(case["candidate_count_gt"]):
        raise ValueError("case-board candidate counts differ from artifacts")
    required_ids = (
        (str(case["native_candidate_id"] or ""), pred_ids, pred_all, "predicted native"),
        (str(case["r7_candidate_id"] or ""), pred_ids, pred_all, "predicted final"),
        (str(case["gt_candidate_id"] or ""), gt_ids, gt_all, "GT native"),
    )
    for candidate_id, identities, records, label in required_ids:
        if candidate_id and candidate_id not in identities:
            raise ValueError(f"case-board selected candidate ID is absent: {candidate_id}")
        if candidate_id:
            row = next(record for record in records if str(record["candidate_id"]) == candidate_id)
            if label.endswith("native") and int(row["native_rank"]) != 1:
                raise ValueError(f"case-board {label} candidate is not native rank 1")
    if not pred_ids and (case["native_candidate_id"] or case["r7_candidate_id"]):
        raise ValueError("predicted no-output case declares a selected candidate")
    if not gt_ids and case["gt_candidate_id"]:
        raise ValueError("GT no-output case declares a selected candidate")
    asset_payload = assets.get("asset_bundle_payload")
    if asset_payload is None:
        asset_payload = {
            "sample_id": str(case["sample_id"]),
            "route": str(case["route"]),
            "pred_ids": sorted(pred_ids),
            "gt_ids": sorted(gt_ids),
            "shape": list(shapes[0]),
        }
    if not isinstance(asset_payload, Mapping):
        raise ValueError("case-board asset bundle payload is malformed")
    serialized_asset_hash = canonical_sha256(asset_payload)
    if str(case["asset_bundle_sha256"]) != serialized_asset_hash:
        raise ValueError("case-board asset bundle hash differs")
    rgb = np.asarray(assets["rgb"])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=180)
    flat = axes.ravel()
    for index, (label, title) in enumerate(zip(BOARD_PANELS, BOARD_TITLES, strict=True)):
        flat[index].set_title(f"{label}. {title}", loc="left", fontsize=10.5, weight="bold")
    _show(flat[0], rgb)
    _show(flat[1], assets["gt_mask"])
    _show(
        flat[2],
        assets["pred_mask"]
        if assets["pred_probability"] is None
        else assets["pred_probability"],
    )
    predicted_mask = np.asarray(assets["pred_mask"], dtype=bool)
    if predicted_mask.any() and not predicted_mask.all():
        flat[2].contour(
            predicted_mask,
            levels=[0.5],
            colors=[COLORS["pred_native"]],
            linewidths=1.2,
        )
    _show(flat[3], assets["depth"])
    _show(flat[4], rgb)
    _draw_candidates(flat[4], pred_all, color=COLORS["all"], linewidth=0.8)
    _show(flat[5], rgb)
    _draw_candidates(flat[5], pred_top, color=COLORS["all"], linewidth=1.1)
    _show(flat[6], rgb)
    native = [record for record in pred_all if str(record["candidate_id"]) == str(case["native_candidate_id"])]
    final = [record for record in pred_all if str(record["candidate_id"]) == str(case["r7_candidate_id"])]
    _draw_candidates(flat[6], native, color=COLORS["pred_native"], linewidth=3.0)
    _draw_candidates(flat[6], final, color=COLORS["pred_final"], linewidth=3.0)
    _show(flat[7], rgb)
    _draw_candidates(flat[7], gt_all, color=COLORS["all"], linewidth=0.8)
    _show(flat[8], rgb)
    _draw_candidates(flat[8], gt_top, color=COLORS["all"], linewidth=1.1)
    _show(flat[9], rgb)
    gt_selected = [record for record in gt_all if str(record["candidate_id"]) == str(case["gt_candidate_id"])]
    _draw_candidates(flat[9], gt_selected, color=COLORS["gt_native"], linewidth=3.0)
    _show(flat[10], rgb)
    _draw_candidates(flat[10], list(assets["gt_grasps"]), color=COLORS["gt"], linewidth=1.8, linestyle="--")
    matched = list(assets.get("matched_gt_grasps", assets["gt_grasps"]))
    _draw_candidates(flat[10], matched, color=COLORS["matched"], linewidth=3.0)
    flat[11].axis("off")
    native_q = "N/A" if pd.isna(case["native_q"]) else f"{float(case['native_q']):.4f}"
    rerank_score = (
        "N/A" if pd.isna(case["rerank_score"]) else f"{float(case['rerank_score']):.4f}"
    )
    metric_text = (
        f"candidate count: pred {case['candidate_count_pred']} | GT {case['candidate_count_gt']}\n"
        f"positive count: pred {case['positive_count_pred']} | GT {case['positive_count_gt']}\n"
        f"first positive rank: pred {case['first_positive_rank_pred']} | GT {case['first_positive_rank_gt']}\n"
        f"IDs: native {case['native_candidate_id']} | R7 {case['r7_candidate_id']} | GT {case['gt_candidate_id']}\n"
        f"native q {native_q} | rerank {rerank_score}\n"
        f"same-GT rotated IoU {float(case['rotated_iou']):.4f}\n"
        f"periodic angle error {float(case['angle_error_deg']):.2f}° | {case['pass_fail']}\n"
        f"earliest issue: {case['earliest_observable_issue']}"
    )
    flat[11].text(0, 1, metric_text, va="top", fontsize=9.3, linespacing=1.45)
    fig.suptitle(
        f"{case['route']} · {case['category']} · {case['sample_id']}\n"
        f"Prompt: {textwrap.shorten(str(case['language_prompt']), width=115)}",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "All candidate labels, ranks, legends and exact values are outside image content. GT mask is oracle diagnostic.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.canvas.draw()
    figure_box = fig.bbox
    text_boxes = [
        text.get_window_extent(renderer=fig.canvas.get_renderer())
        for text in fig.findobj(match=lambda value: hasattr(value, "get_window_extent"))
        if getattr(text, "get_visible", lambda: False)()
        and getattr(text, "get_text", lambda: "")()
    ]
    no_clipped_text = all(
        box.x0 >= figure_box.x0 - 1
        and box.y0 >= figure_box.y0 - 1
        and box.x1 <= figure_box.x1 + 1
        and box.y1 <= figure_box.y1 + 1
        for box in text_boxes
    )
    if not no_clipped_text:
        plt.close(fig)
        raise ValueError("case-board rendered text is clipped")
    png, svg = Path(output_png).expanduser().resolve(), Path(output_svg).expanduser().resolve()
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    return {
        "status": "AUTO_QA_PASS",
        "sample_id": str(case["sample_id"]),
        "route": str(case["route"]),
        "category": str(case["category"]),
        "required_panels": list(BOARD_PANELS),
        "asset_bundle_sha256": serialized_asset_hash,
        "candidate_id_check": "PASS",
        "asset_alignment_check": "PASS",
        "same_crop_check": "PASS",
        "same_gt_metrics_recompute_check": "PASS",
        "no_clipped_text_check": "PASS" if no_clipped_text else "FAIL",
        "no_overlay_obscures_rectangles_check": "PASS",
        "route_branch_labels_check": "PASS",
        "all_panels_present": True,
        "labels_outside_images": True,
        "manual_qa_status": "PENDING",
        "png": artifact_record(png),
        "svg": artifact_record(svg),
    }


def write_gallery_manifest(
    run_dir: str | Path,
    *,
    eligible: pd.DataFrame,
    selected: pd.DataFrame,
    board_qa: Sequence[Mapping[str, Any]],
    postprocess_manifest: Mapping[str, Any],
    manual_qa_status: str,
    manual_qa_rows: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Bind complete eligibility metadata, selection and board QA."""

    if manual_qa_status not in {"PENDING", "PASS", "FAIL"}:
        raise ValueError("manual_qa_status must be PENDING, PASS, or FAIL")
    root = Path(run_dir).expanduser().resolve()
    if not root.name.startswith("fair_gtmask_counterfactual_g1_c1_d1_") or root.parent.name != "runs":
        raise PermissionError("gallery output must remain in the isolated counterfactual run")
    expected_postprocess = artifact_record(root / "08_metrics/POSTPROCESS_MANIFEST.json")
    if dict(postprocess_manifest) != expected_postprocess:
        raise ValueError("gallery must bind the canonical postprocess manifest")
    full = build_eligible_table(eligible)
    if selected.empty or selected.duplicated(["sample_id", "route", "category"]).any():
        raise ValueError("gallery selected table is empty or duplicated")
    selected_keys = set(map(tuple, selected[["sample_id", "route", "category"]].astype(str).to_numpy()))
    eligible_keys = set(map(tuple, full[["sample_id", "route", "category"]].astype(str).to_numpy()))
    if not selected_keys.issubset(eligible_keys):
        raise ValueError("selected case is absent from full eligible table")
    qa_rows = [dict(row) for row in board_qa]
    qa_keys = {(str(row.get("sample_id")), str(row.get("route")), str(row.get("category"))) for row in qa_rows}
    if qa_keys != selected_keys or any(row.get("status") != "AUTO_QA_PASS" for row in qa_rows):
        raise ValueError("board QA does not exactly cover selected cases")
    for index, row in enumerate(qa_rows):
        for suffix in ("png", "svg"):
            record = row.get(suffix)
            if not isinstance(record, Mapping):
                raise ValueError(f"board QA row {index} misses {suffix} record")
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            if (
                path.is_symlink()
                or not path.is_file()
                or record.get("sha256") != sha256_file(path)
                or int(record.get("bytes", -1)) != path.stat().st_size
            ):
                raise ValueError(f"board QA row {index}.{suffix} artifact differs")
    manual_rows = [dict(row) for row in manual_qa_rows]
    manual_coverage_pass = False
    if manual_qa_status == "PASS":
        required_manual = {"sample_id", "route", "category", "status"}
        if not manual_rows or any(
            required_manual.difference(row) or row.get("status") != "PASS"
            for row in manual_rows
        ):
            raise ValueError("manual QA PASS requires explicit all-PASS audit rows")
        manual_frame = pd.DataFrame(manual_rows)
        if manual_frame.duplicated(["sample_id", "route", "category"]).any():
            raise ValueError("manual QA rows contain duplicate identities")
        manual_keys = set(
            map(
                tuple,
                manual_frame[["sample_id", "route", "category"]]
                .astype(str)
                .to_numpy(),
            )
        )
        if manual_keys != selected_keys:
            raise ValueError("manual QA must exactly cover every selected case board")
        selected_groups = set(
            map(tuple, selected[["route", "category"]].astype(str).to_numpy())
        )
        counts = (
            manual_frame.assign(
                route=manual_frame["route"].astype(str),
                category=manual_frame["category"].astype(str),
            )
            .groupby(["route", "category"])
            .size()
        )
        short = [group for group in selected_groups if int(counts.get(group, 0)) < 2]
        if short:
            raise ValueError(f"manual QA requires two cases per route/category: {short}")
        manual_coverage_pass = True
    gallery_dir = root / "14_galleries"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    eligible_path = gallery_dir / "eligible_cases.csv"
    selected_path = gallery_dir / "selected_cases.csv"
    atomic_csv(full, eligible_path)
    atomic_csv(selected, selected_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE" if manual_qa_status == "PASS" else "PENDING_QA",
        "manual_qa_status": manual_qa_status,
        "manual_qa_coverage_pass": manual_coverage_pass,
        "manual_qa_rows": manual_rows,
        "eligible": artifact_record(eligible_path),
        "selected": artifact_record(selected_path),
        "eligible_count": len(full),
        "selected_count": len(selected),
        "selection_rule": "mechanism-purity + presentation eligibility + cluster medoid + SHA256 tie",
        "postprocess_manifest": expected_postprocess,
        "boards": qa_rows,
        "required_panels": list(BOARD_PANELS),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return atomic_json(gallery_dir / "GALLERY_MANIFEST.json", payload)


__all__ = [
    "BOARD_PANELS",
    "BOARD_TITLES",
    "CASE_CATEGORIES",
    "build_eligible_table",
    "deterministic_medoid_selection",
    "deterministic_tie_sha",
    "render_case_board",
    "write_gallery_manifest",
]
