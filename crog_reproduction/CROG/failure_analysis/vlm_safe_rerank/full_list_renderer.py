from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from failure_analysis.failure_utils import rle_to_mask
from failure_analysis.reranking.feature_extraction import load_depth_m

from .features import COORDINATE_CONVENTION, build_pair_evidence, ordered_candidates
from .dataset import stable_sample_id
from .renderer import _depth_display, _draw_candidate, _rgb
from .security import assert_no_ground_truth


FULL_LIST_EVIDENCE_VERSION = "full_list_deterministic_v1"
FULL_LIST_RENDERER_VERSION = "full_list_keep_prior_renderer_v1"
EXPECTED_CANDIDATE_IDS = tuple(f"candidate_{index}" for index in range(5))

_RENDERER_CONTRACT = {
    "renderer_version": FULL_LIST_RENDERER_VERSION,
    "candidate_order": list(EXPECTED_CANDIDATE_IDS),
    "baseline": "candidate_0",
    "baseline_label": "candidate_0 (FROZEN BASELINE / KEEP PRIOR)",
    "panels": ["rgb", "predicted_mask", "metric_depth", "numeric_evidence"],
    "layout": "three_global_panels_five_local_crops_numeric_table",
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def full_list_renderer_hash() -> str:
    return hashlib.sha256(_canonical_json(_RENDERER_CONTRACT).encode("utf-8")).hexdigest()


def _evidence_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("evidence_hash", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def validate_full_list_evidence(evidence: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate a GT-free, immutable full-list evidence payload and its digest."""

    assert_no_ground_truth(evidence)
    if evidence.get("feature_schema_version") != FULL_LIST_EVIDENCE_VERSION:
        raise ValueError("unexpected full-list evidence schema version")
    if evidence.get("baseline_candidate_id") != "candidate_0":
        raise ValueError("full-list baseline must be candidate_0")
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("full-list evidence requires exactly five candidates")
    candidate_ids = tuple(str(row.get("candidate_id")) for row in candidates)
    if candidate_ids != EXPECTED_CANDIDATE_IDS:
        raise ValueError("full-list candidate order or identity changed")
    for index, row in enumerate(candidates):
        if int(row.get("original_rank", -1)) != index:
            raise ValueError("full-list q rank changed")
        if not row.get("candidate_checksum"):
            raise ValueError("full-list candidate checksum is missing")
    claimed = str(evidence.get("evidence_hash", ""))
    if len(claimed) != 64 or claimed != _evidence_digest(evidence):
        raise ValueError("full-list evidence hash mismatch")
    return candidate_ids


def build_full_list_evidence(
    feature: Mapping[str, Any],
    *,
    load_depth: bool = True,
) -> dict[str, Any]:
    """Build deterministic numeric evidence for all five frozen q-ranked candidates."""

    frozen = ordered_candidates(feature)
    candidate_ids = tuple(str(row["candidate_id"]) for row in frozen)
    if candidate_ids != EXPECTED_CANDIDATE_IDS:
        raise ValueError("P2 requires frozen candidate_0..candidate_4 q order")

    pair_rows = [
        build_pair_evidence(feature, candidate_id, load_depth=load_depth)
        for candidate_id in EXPECTED_CANDIDATE_IDS[1:]
    ]
    candidates = [dict(pair_rows[0]["baseline"])] + [
        dict(pair["challenger"]) for pair in pair_rows
    ]
    payload: dict[str, Any] = {
        "feature_schema_version": FULL_LIST_EVIDENCE_VERSION,
        "sample_id": stable_sample_id(feature),
        "language_instruction": str(feature["language_instruction"]),
        "coordinate_convention": COORDINATE_CONVENTION,
        "baseline_candidate_id": "candidate_0",
        "candidate_ordering": "frozen_q_rank",
        "candidate_count": 5,
        "candidates": candidates,
    }
    payload["evidence_hash"] = _evidence_digest(payload)
    validate_full_list_evidence(payload)
    return payload


def _format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _numeric_table(evidence: Mapping[str, Any]) -> str:
    fields = (
        ("q rank", "original_rank"),
        ("q", "q"),
        ("mask rect", "mask_rectangle_coverage"),
        ("mask axis", "mask_axis_support"),
        ("contact L", "mask_contact_support_left"),
        ("contact R", "mask_contact_support_right"),
        ("depth valid", "valid_depth_fraction"),
        ("centre depth m", "centre_depth_m"),
        ("contact dz m", "absolute_contact_depth_difference_m"),
        ("width compat", "grasp_width_compatibility"),
        ("clearance", "clearance_proxy"),
        ("collision", "collision_proxy"),
        ("reliability", "aggregate_reliability"),
    )
    headings = "feature          " + " ".join(
        f"{candidate_id:>13}" for candidate_id in EXPECTED_CANDIDATE_IDS
    )
    lines = [
        "candidate_0 is the FROZEN BASELINE and explicit KEEP prior",
        headings,
    ]
    for label, key in fields:
        values = [
            _format_value(candidate.get(key))
            for candidate in evidence["candidates"]
        ]
        lines.append(f"{label:<16}" + " ".join(f"{value:>13}" for value in values))
    lines.append(
        "collision is a relative 2.5D proxy; missing/unreliable evidence favors KEEP/INSUFFICIENT"
    )
    return "\n".join(lines)


def render_full_list_board(
    feature: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Render RGB, predicted-mask, depth, and numeric evidence for frozen Top-5."""

    evidence = build_full_list_evidence(feature)
    assert_no_ground_truth({"evidence": evidence})
    frozen = ordered_candidates(feature)
    rgb = _rgb(feature["image_path"])
    mask = np.asarray(rle_to_mask(feature["predicted_mask_rle"]), dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("predicted mask is not aligned with RGB")
    depth = None
    depth_path = feature.get("depth_path")
    if depth_path and Path(str(depth_path)).is_file():
        depth, _ = load_depth_m(depth_path, expected_shape=mask.shape)
    depth_rgb, _ = _depth_display(depth, mask.shape)

    colors = (
        (255, 170, 40),
        (55, 180, 235),
        (90, 205, 120),
        (190, 120, 230),
        (225, 210, 80),
    )
    overview = rgb.copy()
    mask_overlay = rgb.copy()
    mask_overlay[mask] = np.rint(
        0.55 * mask_overlay[mask] + 0.45 * np.asarray([70, 210, 120])
    ).astype(np.uint8)
    for index, candidate in enumerate(frozen):
        label = "0B" if index == 0 else str(index)
        _draw_candidate(overview, candidate, label, colors[index])
        _draw_candidate(mask_overlay, candidate, label, colors[index])

    figure = plt.figure(figsize=(20, 13), dpi=110)
    grid = figure.add_gridspec(3, 10, height_ratios=(1.6, 1.0, 1.35))
    for columns, image, title in (
        (slice(0, 3), overview, "RGB + frozen Top-5 (candidate_0 = BASELINE)"),
        (slice(3, 7), mask_overlay, "Predicted target mask + frozen Top-5"),
        (slice(7, 10), depth_rgb, "Metric depth (invalid/missing = black)"),
    ):
        axis = figure.add_subplot(grid[0, columns])
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    for index, candidate in enumerate(frozen):
        axis = figure.add_subplot(grid[1, 2 * index : 2 * index + 2])
        individual = rgb.copy()
        individual[mask] = np.rint(
            0.55 * individual[mask] + 0.45 * np.asarray([70, 210, 120])
        ).astype(np.uint8)
        _draw_candidate(
            individual,
            candidate,
            f"c{index}" + (" BASE" if index == 0 else ""),
            colors[index],
        )
        row, col = int(candidate["row"]), int(candidate["col"])
        radius = 72
        y0, y1 = max(0, row - radius), min(rgb.shape[0], row + radius)
        x0, x1 = max(0, col - radius), min(rgb.shape[1], col + radius)
        crop = individual[y0:y1, x0:x1]
        axis.imshow(crop)
        role = "FROZEN BASELINE / KEEP PRIOR" if index == 0 else f"q-rank {index}"
        axis.set_title(f"candidate_{index}\n{role}", fontsize=10)
        axis.axis("off")

    axis = figure.add_subplot(grid[2, :])
    axis.axis("off")
    axis.text(
        0.01,
        0.98,
        _numeric_table(evidence),
        family="monospace",
        fontsize=8.5,
        va="top",
    )
    figure.suptitle(
        "P2 full-list advisory critic — explicit KEEP candidate_0 prior\n"
        f"Expression: {feature['language_instruction']} | frozen geometry only; no new grasp",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        metadata={"Software": FULL_LIST_RENDERER_VERSION},
    )
    plt.close(figure)
    board_png = stream.getvalue()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(board_png)

    metadata = {
        "sample_id": evidence["sample_id"],
        "evidence_hash": evidence["evidence_hash"],
        "image_sha256": hashlib.sha256(board_png).hexdigest(),
        "renderer_version": FULL_LIST_RENDERER_VERSION,
        "renderer_contract_hash": full_list_renderer_hash(),
        "visible_candidate_ids": list(EXPECTED_CANDIDATE_IDS),
        "baseline_candidate_id": "candidate_0",
        "baseline_label": "candidate_0 (FROZEN BASELINE / KEEP PRIOR)",
        "input_panels": ["rgb", "predicted_mask", "metric_depth", "numeric_evidence"],
        "depth_available": depth is not None,
    }
    assert_no_ground_truth({"evidence": evidence, "board_metadata": metadata})
    return board_png, evidence, metadata
