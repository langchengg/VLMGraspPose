"""Evaluation-only failure galleries for the frozen CROG candidate pool."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from failure_analysis.failure_utils import draw_grasps, load_gt_mask, load_rgb, overlay_mask, rle_to_mask
from failure_analysis.reranking_v2.datasets import load_joined
from failure_analysis.reranking_v2.schema import read_jsonl

from .schema import artifact_identity, atomic_write_json, atomic_write_text


# Values are RGB because ``load_rgb`` returns an RGB array even though OpenCV is
# used only as a raster drawing primitive.
COLORS = {
    "other": (170, 170, 170),
    "q": (86, 180, 233),
    "v2": (230, 159, 0),
    "v3": (213, 94, 0),
    "correct": (0, 158, 115),
    "gt": (0, 114, 178),
    "mask": (204, 121, 167),
}

GALLERY_GROUPS: tuple[tuple[str, str], ...] = (
    ("01_v3_recovered_from_v2", "V3 recovered from V2"),
    ("02_v3_harmful_vs_v2", "V3 harmful relative to V2"),
    ("03_v2_recovered_v3_reverted", "V2 recovered but V3 reverted incorrectly"),
    ("04_q_wrong_v2_wrong_v3_correct", "q-only wrong, V2 wrong, V3 correct"),
    ("05_q_correct_v2_harmful_v3_restores", "q-only correct, V2 harmful, V3 restores"),
    ("06_corrected_only_recovery", "Corrected-only recovery"),
    ("07_legacy_corrected_disagreement", "Legacy/corrected evaluator disagreement"),
    ("08_top5_all_wrong", "Top-5 all wrong"),
    ("09_grounding_failure", "Grounding failure"),
    ("10_ambiguous_annotation_benchmark_sensitivity", "Ambiguous annotation / benchmark sensitivity"),
)
GALLERY_DESCRIPTIONS = dict(GALLERY_GROUPS)
GALLERY_EVIDENCE_FIELDS = (
    "candidate_scores",
    "gate_gains",
    "uncertainty",
    "candidate_evidence",
    "mask_support",
    "angle_confidence",
    "width_consistency",
    "depth_evidence",
    "attention",
)


def _prediction_lookup(path: str | Path, *, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(path):
        sample_id = str(value["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate {name} sample ID: {sample_id}")
        result[sample_id] = value
    return result


def _draw(image: np.ndarray, candidate: Mapping[str, Any], color: tuple[int, int, int], thickness: int) -> np.ndarray:
    polygon = np.asarray(candidate["polygon"], dtype=np.float64)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError(f"invalid candidate polygon: {candidate.get('candidate_id')}")
    result = image.copy()
    cv2.polylines(result, [np.rint(polygon).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
    return result


def _candidate_labels(record: Mapping[str, Any], candidate_ids: Sequence[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for value in record.get("candidate_labels", []):
        candidate_id = str(value["candidate_id"])
        if candidate_id in result:
            raise ValueError(f"duplicate candidate label: {candidate_id}")
        result[candidate_id] = bool(value["candidate_correct"])
    if set(result) != set(candidate_ids):
        raise ValueError("label candidate IDs differ from the frozen candidate pool")
    return result


def classify_gallery_groups(
    *,
    q_correct: bool,
    v2_correct: bool,
    v3_correct: bool,
    oracle: bool,
    legacy_v3_correct: bool,
    corrected_v3_correct: bool,
    mask_iou: float | None,
    grounding_failure: bool = False,
) -> list[str]:
    """Assign all applicable predeclared gallery categories."""
    result: list[str] = []
    if not v2_correct and v3_correct:
        result.append("01_v3_recovered_from_v2")
    if v2_correct and not v3_correct:
        result.append("02_v3_harmful_vs_v2")
    if not q_correct and v2_correct and not v3_correct:
        result.append("03_v2_recovered_v3_reverted")
    if not q_correct and not v2_correct and v3_correct:
        result.append("04_q_wrong_v2_wrong_v3_correct")
    if q_correct and not v2_correct and v3_correct:
        result.append("05_q_correct_v2_harmful_v3_restores")
    if not v2_correct and corrected_v3_correct and not legacy_v3_correct:
        result.append("06_corrected_only_recovery")
    if corrected_v3_correct != legacy_v3_correct:
        result.append("07_legacy_corrected_disagreement")
    if not oracle:
        result.append("08_top5_all_wrong")
    finite_iou = None if mask_iou is None else float(mask_iou)
    if finite_iou is not None and not math.isfinite(finite_iou):
        finite_iou = None
    if grounding_failure or (finite_iou is not None and finite_iou < 0.5):
        result.append("09_grounding_failure")
    if corrected_v3_correct != legacy_v3_correct:
        result.append("10_ambiguous_annotation_benchmark_sensitivity")
    return result


# Backward-compatible private name used by early callers.
_groups = classify_gallery_groups


def _selection_id(record: Mapping[str, Any], *, name: str, candidate_ids: Sequence[str]) -> str:
    order = [str(value) for value in record.get("candidate_order", [])]
    if len(order) != 5 or len(set(order)) != 5 or set(order) != set(candidate_ids):
        raise ValueError(f"{name} ranking changed or corrupted the frozen candidate pool")
    selected = str(record.get("selection", {}).get("selected_candidate_id", order[0]))
    if selected != order[0]:
        raise ValueError(f"{name} selected candidate differs from ranking top")
    return selected


def _candidate_vector_with_coverage(
    record: Mapping[str, Any], key: str, candidate_ids: Sequence[str]
) -> tuple[list[Any], dict[str, Any]]:
    present = key in record and record.get(key) is not None
    values = list(record[key]) if present else [None] * 5
    if len(values) != 5:
        raise ValueError(f"{key} must contain five candidate-aligned values")
    id_field = f"{key}_ids"
    vector_ids = [
        str(value)
        for value in record.get(id_field, record.get("candidate_probability_ids", candidate_ids))
    ]
    if len(vector_ids) != 5 or len(set(vector_ids)) != 5 or set(vector_ids) != set(candidate_ids):
        raise ValueError(f"{key} candidate identity differs from frozen candidates")
    by_id = dict(zip(vector_ids, values, strict=True))
    aligned = [by_id[candidate_id] for candidate_id in candidate_ids]
    available = sum(value is not None for value in aligned)
    return aligned, {
        "available": available,
        "total": len(candidate_ids),
        "coverage": available / len(candidate_ids),
        "status": "complete" if available == len(candidate_ids) else "n/a" if available == 0 else "partial",
    }


def _candidate_vector(record: Mapping[str, Any], key: str, candidate_ids: Sequence[str]) -> list[Any]:
    """Backward-compatible value-only view of a candidate-aligned vector."""
    return _candidate_vector_with_coverage(record, key, candidate_ids)[0]


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _evidence_value(candidate: Mapping[str, Any], evidence: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    features = candidate.get("features", {})
    for name in aliases:
        if name in evidence:
            return evidence[name]
        if isinstance(features, Mapping) and name in features:
            return features[name]
    return None


def _top_token(evidence: Mapping[str, Any]) -> str:
    if evidence.get("top_token") is not None:
        return str(evidence["top_token"])
    if evidence.get("top_token_index") is not None:
        return f"token #{int(evidence['top_token_index'])}"
    return "n/a"


def render_case(
    *,
    sample: Any,
    legacy_label: Mapping[str, Any],
    raw_prediction: Mapping[str, Any],
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    group: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Render one explicitly labelled evaluation panel."""
    rgb = load_rgb(sample.feature["image_path"])
    candidates = list(sample.feature["candidates"])
    candidate_ids = [str(value["candidate_id"]) for value in candidates]
    if len(candidate_ids) != 5 or len(set(candidate_ids)) != 5:
        raise ValueError("gallery requires exactly five unique frozen candidates")
    by_id = dict(zip(candidate_ids, candidates, strict=True))
    corrected = _candidate_labels(sample.label, candidate_ids)
    legacy = _candidate_labels(legacy_label, candidate_ids)
    q_id = candidate_ids[0]
    v2_id = _selection_id(v2, name="V2", candidate_ids=candidate_ids)
    v3_id = _selection_id(v3, name="V3", candidate_ids=candidate_ids)
    scores, score_coverage = _candidate_vector_with_coverage(v3, "candidate_scores", candidate_ids)
    gains, gain_coverage = _candidate_vector_with_coverage(v3, "gate_gains", candidate_ids)
    uncertainty, uncertainty_coverage = _candidate_vector_with_coverage(v3, "uncertainty", candidate_ids)
    evidence, evidence_coverage = _candidate_vector_with_coverage(v3, "candidate_evidence", candidate_ids)
    if any(value is not None and not isinstance(value, Mapping) for value in evidence):
        raise ValueError("candidate_evidence values must be mappings")
    semantic_available = {
        "mask_support": 0,
        "angle_confidence": 0,
        "width_consistency": 0,
        "depth_evidence": 0,
        "attention": 0,
    }

    predicted_mask = rle_to_mask(sample.feature["predicted_mask_rle"])
    predicted = overlay_mask(rgb, predicted_mask, COLORS["mask"], alpha=0.28)
    for candidate in candidates:
        predicted = _draw(predicted, candidate, COLORS["other"], 2)
    # Nested line widths preserve visibility if two or three methods select the
    # same rectangle: q is outermost, V2 middle, V3 innermost.
    predicted = _draw(predicted, by_id[q_id], COLORS["q"], 10)
    predicted = _draw(predicted, by_id[v2_id], COLORS["v2"], 7)
    predicted = _draw(predicted, by_id[v3_id], COLORS["v3"], 4)

    gt_mask = load_gt_mask(raw_prediction["mask_path"], raw_prediction["obj_id"])
    gt = overlay_mask(rgb, gt_mask, (220, 220, 220), alpha=0.32)
    gt = draw_grasps(gt, raw_prediction.get("gt_grasps", []), COLORS["gt"], thickness=2)

    figure = plt.figure(figsize=(17.5, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 5, height_ratios=(2.4, 1.55))
    original_axis = figure.add_subplot(grid[0, 0:2])
    prediction_axis = figure.add_subplot(grid[0, 2:4])
    evaluation_axis = figure.add_subplot(grid[0, 4])
    original_axis.imshow(rgb)
    original_axis.set_title("Original RGB (no evaluation annotations)")
    prediction_axis.imshow(predicted)
    prediction_axis.set_title("Prediction panel: frozen candidates + predicted mask")
    evaluation_axis.imshow(gt)
    evaluation_axis.set_title("Evaluation-only GT panel")
    for axis in (original_axis, prediction_axis, evaluation_axis):
        axis.axis("off")
    prediction_axis.legend(
        handles=[
            Line2D([0], [0], color=np.asarray(COLORS["q"]) / 255.0, lw=6, label="q-only selection"),
            Line2D([0], [0], color=np.asarray(COLORS["v2"]) / 255.0, lw=5, label="V2 selection"),
            Line2D([0], [0], color=np.asarray(COLORS["v3"]) / 255.0, lw=4, label="V3 selection"),
            Patch(facecolor=np.asarray(COLORS["mask"]) / 255.0, alpha=0.4, label="Predicted mask"),
            Line2D([0], [0], color=np.asarray(COLORS["correct"]) / 255.0, lw=4, label="Benchmark-correct candidate frame"),
        ],
        loc="lower left",
        fontsize=7,
    )

    for index, candidate in enumerate(candidates):
        axis = figure.add_subplot(grid[1, index])
        half = max(48, int(round(float(candidate["width_px"]) * 1.2)))
        x, y = int(round(float(candidate["cx"]))), int(round(float(candidate["cy"])))
        local = _draw(rgb, candidate, COLORS["v3"] if candidate_ids[index] == v3_id else COLORS["other"], 3)
        crop = local[max(0, y - half) : min(local.shape[0], y + half), max(0, x - half) : min(local.shape[1], x + half)]
        if crop.size == 0:
            crop = np.zeros((2, 2, 3), dtype=np.uint8)
        axis.imshow(crop)
        item = evidence[index] or {}
        mask_support = _evidence_value(candidate, item, ("mask_support", "g2_mask_probability_mean", "mask_probability_mean"))
        angle_confidence = _evidence_value(candidate, item, ("angle_confidence", "g3_axial_concentration", "axial_concentration"))
        width_consistency = _evidence_value(candidate, item, ("width_consistency", "g4_width_consistency"))
        depth_evidence = _evidence_value(candidate, item, ("depth_evidence", "depth_valid_fraction", "g9_depth_valid_fraction"))
        semantic_available["mask_support"] += int(mask_support is not None)
        semantic_available["angle_confidence"] += int(angle_confidence is not None)
        semantic_available["width_consistency"] += int(width_consistency is not None)
        semantic_available["depth_evidence"] += int(depth_evidence is not None)
        semantic_available["attention"] += int(
            item.get("top_token") is not None or item.get("top_token_index") is not None
        )
        candidate_id = candidate_ids[index]
        status = "CORRECT" if corrected[candidate_id] else "wrong"
        axis.set_title(
            f"{candidate_id} — {status}\n"
            f"q={_fmt(candidate.get('q_raw'))}  V3={_fmt(scores[index])}\n"
            f"gain={_fmt(gains[index])}  U={_fmt(uncertainty[index])}\n"
            f"C/L={int(corrected[candidate_id])}/{int(legacy[candidate_id])}\n"
            f"mask={_fmt(mask_support)}  angle={_fmt(angle_confidence)}\n"
            f"width={_fmt(width_consistency)}  depth={_fmt(depth_evidence)}\n"
            f"attention={_top_token(item)}",
            fontsize=7.2,
            color="#006B4F" if corrected[candidate_id] else "#333333",
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4 if corrected[candidate_id] else 1)
            spine.set_edgecolor(np.asarray(COLORS["correct"]) / 255.0 if corrected[candidate_id] else "#AAAAAA")
        axis.set_xticks([])
        axis.set_yticks([])

    prompt = str(sample.feature["language_instruction"])
    figure.suptitle(
        f"Prompt: {prompt}\n{GALLERY_DESCRIPTIONS.get(group, group)}; "
        f"q={q_id}, V2={v2_id}, V3={v3_id}; corrected q/V2/V3="
        f"{int(corrected[q_id])}/{int(corrected[v2_id])}/{int(corrected[v3_id])}; "
        f"legacy V3={int(legacy[v3_id])}; Oracle@5={int(any(corrected.values()))}",
        fontsize=10.5,
    )
    output = Path(output_path).with_suffix(".png")
    pdf_output = output.with_suffix(".pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or pdf_output.exists():
        plt.close(figure)
        raise FileExistsError(output if output.exists() else pdf_output)
    from .plotting import _atomic_figure_write

    _atomic_figure_write(figure, output, format_name="png")
    _atomic_figure_write(figure, pdf_output, format_name="pdf")
    plt.close(figure)
    vector_coverage = {
        "candidate_scores": score_coverage,
        "gate_gains": gain_coverage,
        "uncertainty": uncertainty_coverage,
        "candidate_evidence": evidence_coverage,
    }
    for name, available in semantic_available.items():
        vector_coverage[name] = {
            "available": available,
            "total": len(candidate_ids),
            "coverage": available / len(candidate_ids),
            "status": "complete" if available == len(candidate_ids) else "n/a" if available == 0 else "partial",
        }
    return {
        "sample_id": str(sample.sample_id),
        "group": group,
        "prompt": prompt,
        "q_candidate_id": q_id,
        "v2_candidate_id": v2_id,
        "v3_candidate_id": v3_id,
        "corrected": {"q": corrected[q_id], "v2": corrected[v2_id], "v3": corrected[v3_id]},
        "legacy_v3": legacy[v3_id],
        "oracle_at_5": any(corrected.values()),
        "image_path": str(output.resolve()),
        "pdf_path": str(pdf_output.resolve()),
        "image_artifacts": {
            "png": artifact_identity(output),
            "pdf": artifact_identity(pdf_output),
        },
        "png_dpi": 300,
        "evidence_coverage": vector_coverage,
        "gt_usage": "evaluation_panel_only",
    }


def build_v3_galleries(
    *,
    features_path: str | Path,
    corrected_labels_path: str | Path,
    legacy_labels_path: str | Path,
    raw_predictions_path: str | Path,
    v2_predictions_path: str | Path,
    v3_predictions_path: str | Path,
    output_dir: str | Path,
    per_group: int = 5,
    independent_evaluation_completion_path: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reusable ``build-gallery`` backend for the ten predeclared categories."""
    if int(per_group) < 0:
        raise ValueError("per_group must be non-negative")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    independent_identity: dict[str, Any] | None = None
    if independent_evaluation_completion_path is not None:
        from .report_validation import verify_independent_evaluation_completion

        independent_identity = verify_independent_evaluation_completion(
            independent_evaluation_completion_path
        )
    input_identities = {
        "features": artifact_identity(features_path),
        "corrected_labels": artifact_identity(corrected_labels_path),
        "legacy_labels": artifact_identity(legacy_labels_path),
        "raw_predictions": artifact_identity(raw_predictions_path),
        "v2_predictions": artifact_identity(v2_predictions_path),
        "v3_predictions": artifact_identity(v3_predictions_path),
    }
    samples = list(load_joined(features_path, corrected_labels_path))
    sample_ids = [str(sample.sample_id) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate joined gallery sample ID")
    expected = set(sample_ids)
    legacy_by_id = _prediction_lookup(legacy_labels_path, name="legacy label")
    v2 = _prediction_lookup(v2_predictions_path, name="V2 prediction")
    v3 = _prediction_lookup(v3_predictions_path, name="V3 prediction")
    raw: dict[tuple[str, int], dict[str, Any]] = {}
    for value in read_jsonl(raw_predictions_path):
        key = (str(value["split"]), int(value["sample_id"]))
        if key in raw:
            raise ValueError(f"duplicate raw prediction identity: {key}")
        raw[key] = value
    for name, lookup in (("legacy labels", legacy_by_id), ("V2 predictions", v2), ("V3 predictions", v3)):
        if set(lookup) != expected:
            raise ValueError(f"{name} cohort differs: missing={sorted(expected-set(lookup))[:5]}, extra={sorted(set(lookup)-expected)[:5]}")

    grouped: dict[str, list[tuple[Any, Mapping[str, Any], Mapping[str, Any]]]] = {name: [] for name, _ in GALLERY_GROUPS}
    for sample in samples:
        sample_id = str(sample.sample_id)
        candidate_ids = [str(value["candidate_id"]) for value in sample.feature["candidates"]]
        corrected = _candidate_labels(sample.label, candidate_ids)
        legacy = _candidate_labels(legacy_by_id[sample_id], candidate_ids)
        q_id = candidate_ids[0]
        v2_id = _selection_id(v2[sample_id], name="V2", candidate_ids=candidate_ids)
        v3_id = _selection_id(v3[sample_id], name="V3", candidate_ids=candidate_ids)
        raw_key = (str(sample.feature["split"]), int(sample.feature["sample_id"]))
        if raw_key not in raw:
            raise ValueError(f"raw prediction missing gallery sample: {raw_key}")
        raw_value = raw[raw_key]
        failure_type = str(raw_value.get("failure_type", "")).lower()
        groups = classify_gallery_groups(
            q_correct=corrected[q_id],
            v2_correct=corrected[v2_id],
            v3_correct=corrected[v3_id],
            oracle=any(corrected.values()),
            legacy_v3_correct=legacy[v3_id],
            corrected_v3_correct=corrected[v3_id],
            mask_iou=raw_value.get("mask_iou"),
            grounding_failure="ground" in failure_type,
        )
        for group in groups:
            grouped[group].append((sample, legacy_by_id[sample_id], raw_value))

    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for group, _ in GALLERY_GROUPS:
        group_dir = output / group
        group_dir.mkdir()
        cases = sorted(grouped[group], key=lambda value: str(value[0].sample_id))[: int(per_group)]
        for index, (sample, legacy, raw_value) in enumerate(cases):
            records.append(
                render_case(
                    sample=sample,
                    legacy_label=legacy,
                    raw_prediction=raw_value,
                    v2=v2[str(sample.sample_id)],
                    v3=v3[str(sample.sample_id)],
                    group=group,
                    output_path=group_dir / f"{index:02d}_{str(sample.sample_id).replace(':', '_')}.png",
                )
            )

    sections = []
    for group, description in GALLERY_GROUPS:
        cases = [value for value in records if value["group"] == group]
        cards = "".join(
            f"<article><h3>{html.escape(value['sample_id'])}</h3><p>{html.escape(value['prompt'])}</p>"
            f"<img loading='lazy' alt='{html.escape(description)}' src='{html.escape(str(Path(value['image_path']).relative_to(output)))}'></article>"
            for value in cases
        )
        if not cases:
            cards = "<p class='empty'>No samples met this category on the evaluated cohort.</p>"
        sections.append(f"<section><h2>{html.escape(group)} — {html.escape(description)}</h2>{cards}</section>")
    atomic_write_text(
        output / "index.html",
        "<!doctype html><meta charset='utf-8'><title>CROG V3 galleries</title>"
        "<style>body{font-family:system-ui;max-width:1500px;margin:auto;padding:24px}"
        "img{width:100%}article{margin-bottom:48px}.empty{color:#666}</style>"
        "<h1>CROG V3 evaluation galleries</h1>"
        "<p>Ground truth and correctness are used only in panels explicitly labelled evaluation-only. "
        "q-only is sky blue, V2 orange, V3 vermillion, and benchmark-correct candidates have green frames.</p>"
        + "".join(sections),
    )
    aggregate_coverage: dict[str, Any] = {}
    for name in GALLERY_EVIDENCE_FIELDS:
        available = sum(int(record["evidence_coverage"][name]["available"]) for record in records)
        total = sum(int(record["evidence_coverage"][name]["total"]) for record in records)
        aggregate_coverage[name] = {
            "available": available,
            "total": total,
            "coverage": available / total if total else None,
            "status": "complete" if total and available == total else "n/a" if available == 0 else "partial",
        }
    result = {
        "schema_version": "3.0.0",
        "kind": "v3_failure_galleries",
        "status": "complete",
        "case_count": len(records),
        "requested_per_group": int(per_group),
        "groups": [{"name": name, "description": description} for name, description in GALLERY_GROUPS],
        "available_counts": {key: len(value) for key, value in grouped.items()},
        "rendered_counts": {key: sum(record["group"] == key for record in records) for key in grouped},
        "cases": records,
        "evidence_coverage": aggregate_coverage,
        "index_html": artifact_identity(output / "index.html"),
        "inputs": input_identities,
        "input_hashes": {name: identity["sha256"] for name, identity in input_identities.items()},
        "independent_evaluation_completion": independent_identity,
        "formal_evidence_ready": independent_identity is not None,
        "render_formats": ["png", "pdf"],
        "png_dpi": 300,
        "labels_read": True,
        "ground_truth_usage": "evaluation_panels_only",
    }
    atomic_write_json(output / "gallery.json", result)
    return result


def build_gallery(**kwargs: Any) -> dict[str, Any]:
    """Public backend name intended for the CLI ``build-gallery`` command."""
    return build_v3_galleries(**kwargs)


def build_v3_galleries_strict(
    *, independent_evaluation_completion_path: str | Path | Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Build a formal gallery bound to a completed independent evaluation.

    ``build_v3_galleries`` remains callable for legacy/diagnostic callers, but
    only this strict wrapper produces an artifact accepted by the report layer.
    """
    return build_v3_galleries(
        independent_evaluation_completion_path=independent_evaluation_completion_path,
        **kwargs,
    )


__all__ = (
    "COLORS",
    "GALLERY_EVIDENCE_FIELDS",
    "GALLERY_GROUPS",
    "build_gallery",
    "build_v3_galleries",
    "build_v3_galleries_strict",
    "classify_gallery_groups",
    "render_case",
)
