from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from unified_reranking.case_analysis_v2 import (
    SCORE_WEIGHTS,
    clarity_score,
    cross_route_candidates,
    make_eligible_cases,
    select_audit_cases,
)
from unified_reranking.case_visuals_v2 import (
    ALL_CANDIDATE_COLORS,
    DISPLAY_NAMES,
    binary_iou,
    candidate_generation_verdict,
    crop_extent,
    gate_verdict,
    grounding_verdict,
    gt_geometry,
    offline_failure_reason,
    render_route_case_board,
    reranker_verdict,
    rle_decode,
)


def test_route_mapping_is_exact_and_d1_is_not_inferred() -> None:
    assert DISPLAY_NAMES == {
        "crog": "CROG",
        "g1": "HiFi-CS → G1",
        "c1": "HiFi-CS → C1",
    }
    assert "D1" not in DISPLAY_NAMES and "d1" not in DISPLAY_NAMES


def test_crog_rle_and_mask_iou_contract() -> None:
    mask = rle_decode({"size": [2, 4], "start_value": 0, "counts": [2, 3, 3]})
    assert mask.tolist() == [[False, False, True, True], [True, False, False, False]]
    assert binary_iou(mask, mask) == 1.0
    try:
        rle_decode({"size": [2, 4], "counts": [2, 2]})
    except ValueError as error:
        assert "length" in str(error)
    else:
        raise AssertionError("invalid RLE length was accepted")


def test_same_gt_geometry_and_strict_threshold_reasons() -> None:
    gt = gt_geometry([[10, 10], [30, 10], [30, 20], [10, 20]])
    assert gt["cx_px"] == 20.0 and gt["cy_px"] == 15.0
    assert gt["height_px"] == 20.0 and gt["width_px"] == 10.0
    assert offline_failure_reason(0.25, 30.0) == "IoU"
    assert offline_failure_reason(np.nextafter(0.25, 1.0), 30.0) == "none"
    assert offline_failure_reason(0.4, 30.1) == "angle"


def test_verdicts_cover_generation_reranker_and_gate() -> None:
    assert candidate_generation_verdict(candidate_count=0, top5_positive=False, full_pool_positive=False) == "NO OUTPUT"
    assert candidate_generation_verdict(candidate_count=3, top5_positive=False, full_pool_positive=True) == "PARTIAL"
    assert grounding_verdict(0.24) == "FAIL"
    assert grounding_verdict(0.25) == "BORDERLINE"
    assert grounding_verdict(0.50) == "PASS"
    assert reranker_verdict(native_correct=False, final_correct=True, native_id="a", final_id="b") == "RECOVERED"
    assert gate_verdict(native_correct=True, ungated_correct=False, final_correct=True, native_id="a", ungated_id="b", final_id="a") == "GOOD REJECT"


def test_crop_contains_target_candidates_and_gt() -> None:
    candidates = pd.DataFrame(
        [{"cx_px": 100, "cy_px": 80, "width_px": 30, "height_px": 20, "theta_deg": 25}]
    )
    extent = crop_extent(
        image_shape=(480, 640),
        target_bbox=(70, 50, 80, 60),
        candidates=candidates,
        gt_rectangles=[[[75, 55], [125, 55], [125, 95], [75, 95]]],
    )
    x0, y0, x1, y1 = extent
    assert 0 <= x0 < 70 < 150 < x1 <= 640
    assert 0 <= y0 < 50 < 110 < y1 <= 480


def _case(route: str, category: str, sample_id: str, score_bias: float = 0.0) -> dict:
    native_correct = category in {"harmful", "gate_prevented_harmful"}
    final_correct = category in {"recovered", "gate_prevented_harmful"}
    return {
        "route": route,
        "sample_id": sample_id,
        "scene_id": sample_id + "-scene",
        "analysis_category": "no_positive_pool" if category == "candidate_generation_irreparable" else category,
        "outcome": "recovered" if category == "recovered" else "wrong_retained",
        "native_correct": native_correct,
        "ungated_correct": final_correct,
        "gated_correct": final_correct,
        "native_candidate_id": "a",
        "ungated_candidate_id": "b",
        "gated_candidate_id": "b" if category in {"recovered", "harmful"} else "a",
        "full_pool_positive": category != "candidate_generation_irreparable",
        "top5_positive": category != "candidate_generation_irreparable",
        "candidate_count_all": 5,
        "candidate_generation_verdict": "FAIL" if category == "candidate_generation_irreparable" else "PASS",
        "native_matched_gt_index": 0,
        "final_matched_gt_index": 0,
        "native_cx_px": 100,
        "native_cy_px": 100,
        "native_theta_deg": 0,
        "native_width_px": 40,
        "final_cx_px": 140 + score_bias,
        "final_cy_px": 100,
        "final_theta_deg": 45,
        "final_width_px": 45,
        "final_matched_gt_width_px": 60,
        "final_matched_gt_height_px": 20,
        "native_diagnostic_iou": 0.10,
        "native_diagnostic_angle_error_deg": 10,
        "final_diagnostic_iou": 0.50,
        "final_diagnostic_angle_error_deg": 10,
        "target_bbox_width": 100,
        "target_bbox_height": 80,
        "crop_width_px": 220,
        "crop_height_px": 180,
        "mask_iou": 0.8,
        "language": "grasp the clearly visible object",
        "dominant_abs_share": 0.8,
        "rgb_exists": True,
        "gt_mask_exists": True,
        "predicted_mask_available": True,
        "candidate_geometry_valid": True,
        "image_quality_score": 0.8,
        "highlight_clip_fraction": 0.01,
        "grounding_verdict": "PASS",
        "bridge_category": "candidate_generation_limited",
    }


def test_integer_clarity_formula_and_selection_are_order_invariant() -> None:
    components = {name: 1.0 for name in SCORE_WEIGHTS}
    numerator, score = clarity_score(components)
    assert numerator == 1_000_000_000 and score == 1.0
    rows = []
    for route in ("crog", "g1", "c1"):
        for category in (
            "recovered",
            "harmful",
            "gate_prevented_harmful",
            "gate_missed_recoverable",
            "wrong_to_wrong_solvable",
            "candidate_generation_irreparable",
        ):
            rows.extend([_case(route, category, f"{route}-{category}-{index}", index) for index in range(7)])
    eligible = make_eligible_cases(pd.DataFrame(rows))
    selected, alternatives = select_audit_cases(eligible)
    shuffled, shuffled_alternatives = select_audit_cases(eligible.sample(frac=1, random_state=7))
    assert len(selected) == 18
    assert alternatives.groupby(["route", "presentation_outcome"]).size().eq(6).all()
    assert selected[["route", "presentation_outcome", "sample_id"]].sort_values(["route", "presentation_outcome"]).reset_index(drop=True).equals(
        shuffled[["route", "presentation_outcome", "sample_id"]].sort_values(["route", "presentation_outcome"]).reset_index(drop=True)
    )
    assert len(shuffled_alternatives) == len(alternatives)


def test_severely_overexposed_scene_is_not_presentation_eligible() -> None:
    row = _case("crog", "candidate_generation_irreparable", "clipped")
    row["highlight_clip_fraction"] = 0.35
    eligible = make_eligible_cases(pd.DataFrame([row]))
    assert not eligible["mandatory_eligible"].any()


def test_cross_route_selection_requires_common_gt_and_distinct_scenes() -> None:
    rows = []
    for sample_index in range(4):
        for route_index, route in enumerate(("crog", "g1", "c1")):
            row = _case(route, "recovered", f"sample-{sample_index}", route_index * 5)
            row.update(
                {
                    "scene_id": f"scene-{sample_index}",
                    "final_matched_gt_index": 2,
                    "earliest_observable_issue": ["native ranking", "candidate generation", "visual grounding"][route_index],
                    "presentation_clarity_score": 0.8,
                    "gated_correct": route_index == 0,
                }
            )
            rows.append(row)
    selected = cross_route_candidates(pd.DataFrame(rows))
    assert len(selected) == 2 and selected["sample_id"].nunique() == 2
    assert selected["cross_route_gt_index"].eq(2).all()


def test_route_board_renders_png_svg_with_required_resolution(tmp_path: Path) -> None:
    rgb = np.full((480, 640, 3), 180, np.uint8)
    gt_mask = np.zeros((480, 640), bool)
    gt_mask[150:300, 200:400] = True
    pred = gt_mask.copy()
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "native_rank": 1, "native_score": 0.5, "ensemble_score": 0.1, "cx_px": 280, "cy_px": 220, "theta_deg": 0, "width_px": 80, "height_px": 20, "in_top5": True},
            {"candidate_id": "b", "native_rank": 2, "native_score": 0.4, "ensemble_score": 0.9, "cx_px": 330, "cy_px": 230, "theta_deg": 45, "width_px": 70, "height_px": 20, "in_top5": True},
        ]
    )
    sample = {
        **_case("crog", "recovered", "sample"),
        "presentation_outcome": "recovered",
        "language": "grasp the red object",
        "crop_extent_json": json.dumps([150, 100, 450, 350]),
        "native_candidate_id": "a",
        "gated_candidate_id": "b",
        "native_native_rank": 1,
        "final_native_rank": 2,
        "native_native_score": 0.5,
        "final_native_score": 0.4,
        "native_ensemble_score": 0.1,
        "final_ensemble_score": 0.9,
        "native_center_error_px": 20,
        "final_center_error_px": 10,
        "native_width_error_px": 5,
        "final_width_error_px": 2,
        "positive_count_top5": 1,
        "positive_count_all": 1,
        "first_positive_rank": 2,
        "native_ranking_verdict": "FAIL",
        "reranker_verdict": "RECOVERED",
        "gate_verdict": "GOOD ACCEPT",
        "gate_decision_reason": "accepted_all_conditions",
        "score_margin": 0.8,
        "earliest_observable_issue": "native ranking",
        "final_matched_gt_index": 0,
        "contribution_summary": "angle agreement +0.4.",
    }
    png, svg = tmp_path / "board.png", tmp_path / "board.svg"
    render_route_case_board(
        sample=sample,
        candidates=candidates,
        rgb=rgb,
        gt_mask=gt_mask,
        predicted_mask=pred,
        gt_rectangles=[[[250, 210], [350, 210], [350, 230], [250, 230]]],
        output_png=png,
        output_svg=svg,
    )
    assert png.is_file() and svg.is_file()
    with Image.open(png) as image:
        assert image.size == (2400, 1350)
        pixels = np.asarray(image.convert("RGB"))
    # Panel D occupies the lower-left board cell.  Both synthetic candidates
    # must leave a substantial, high-contrast rank-colour trace there; this
    # prevents a regression to nearly invisible pale/transparent outlines.
    all_candidates_panel = pixels[625:970, 40:645]
    for value in ALL_CANDIDATE_COLORS[:2]:
        color = np.asarray(
            [int(value[index : index + 2], 16) for index in (1, 3, 5)],
            dtype=np.int16,
        )
        distance = np.abs(all_candidates_panel.astype(np.int16) - color)
        assert int(np.count_nonzero(np.max(distance, axis=2) <= 35)) > 40
