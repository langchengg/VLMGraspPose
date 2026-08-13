from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gtmask_counterfactual.independent import (
    assert_frame_exact,
    canonical_corners,
    evaluate_same_gt_candidate,
    independent_evaluate_candidates,
    independent_recompute_from_frames,
    periodic_angle_error_deg,
    raster_iou,
)
from gtmask_counterfactual.metrics import (
    compare_branch_outcomes,
    compute_branch_metrics,
    evaluate_candidate_rows,
)
from gtmask_counterfactual.statistics import (
    align_paired_frames,
    apply_holm_family,
    cluster_bootstrap_delta,
    exact_mcnemar,
    holm_adjust,
    paired_metric_statistics,
)
from gtmask_counterfactual.taxonomy import (
    NATIVE_CLASSES,
    POST_R7_CLASSES,
    classify_native_taxonomy,
    classify_post_r7_taxonomy,
    taxonomy_counts,
    taxonomy_definitions,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = (
    ROOT
    / "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py"
)


def _rect(
    cx: float = 100.0,
    cy: float = 100.0,
    theta: float = 0.0,
    width: float = 80.0,
    height: float = 20.0,
) -> dict[str, float]:
    return {
        "cx_px": cx,
        "cy_px": cy,
        "theta_deg": theta,
        "width_px": width,
        "height_px": height,
    }


def _gt_corners(**changes: float) -> list[list[float]]:
    return canonical_corners(_rect(**changes)).tolist()


def _candidate(
    sample_id: str,
    branch: str,
    candidate_id: str,
    rank: int,
    *,
    route: str = "G1",
    success_geometry: bool = True,
) -> dict[str, object]:
    geometry = _rect(cx=100.0 if success_geometry else 400.0)
    return {
        "sample_id": sample_id,
        "route": route,
        "branch": branch,
        "candidate_id": candidate_id,
        "native_rank": rank,
        "native_score": 1.0 / rank,
        **geometry,
    }


def test_same_gt_evaluator_strict_iou_angle_and_empty_contract() -> None:
    identical = evaluate_same_gt_candidate(_rect(), [_gt_corners()])
    assert identical["candidate_success"] is True
    assert identical["best_same_gt_iou"] == 1.0

    equivalent = evaluate_same_gt_candidate(_rect(theta=180.0), [_gt_corners()])
    assert equivalent["candidate_success"] is True
    assert periodic_angle_error_deg(89.0, -89.0) == 2.0
    assert periodic_angle_error_deg(0.0, 30.0) == 30.0
    assert periodic_angle_error_deg(0.0, 30.000001) > 30.0

    # Five inclusive x pixels shifted by three gives intersection/union=2/8.
    at_strict_boundary = raster_iou(
        _rect(cx=100.0, width=4.0), _rect(cx=103.0, width=4.0)
    )
    assert at_strict_boundary == 0.25
    boundary = evaluate_same_gt_candidate(
        _rect(cx=100.0, width=4.0),
        [_gt_corners(cx=103.0, width=4.0)],
    )
    assert boundary["candidate_success"] is False

    angle_fail = evaluate_same_gt_candidate(
        _rect(theta=31.0), [_gt_corners(theta=0.0)]
    )
    assert angle_fail["pairwise"][0]["iou_ok"] is True
    assert angle_fail["pairwise"][0]["angle_ok"] is False
    different_gt_conjunction = evaluate_same_gt_candidate(
        _rect(theta=31.0),
        [_gt_corners(theta=0.0), _gt_corners(cx=400.0, theta=31.0)],
    )
    assert different_gt_conjunction["candidate_success"] is False
    assert evaluate_same_gt_candidate(_rect(), [])["candidate_success"] is False
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_same_gt_candidate(_rect(width=-1.0), [_gt_corners()])


def test_locked_adapter_matches_independent_and_ignores_branch_identity() -> None:
    candidates = pd.DataFrame(
        [
            _candidate("s1", "predicted", "p1", 1),
            _candidate("s1", "gt_oracle", "g1", 1),
            _candidate("s2", "predicted", "p2", 1, success_geometry=False),
        ]
    )
    gt = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "gt_grasp_rectangles": [[_gt_corners()], [_gt_corners()]],
        }
    )
    locked = evaluate_candidate_rows(
        candidates, gt, evaluator_path=EVALUATOR
    )
    independent = independent_evaluate_candidates(candidates, gt)
    assert locked["candidate_success"].tolist() == [True, True, False]
    assert_frame_exact(
        independent,
        locked,
        keys=["sample_id", "route", "branch", "candidate_id"],
        columns=["candidate_success", "matched_gt_index", "best_same_gt_iou"],
    )
    invalid = candidates.iloc[[0]].copy()
    invalid["width_px"] = -1.0
    marked = evaluate_candidate_rows(
        invalid, gt, evaluator_path=EVALUATOR, on_invalid="mark"
    )
    assert marked.loc[0, "evaluator_valid"] == False  # noqa: E712
    assert marked.loc[0, "candidate_success"] == False  # noqa: E712


def test_branch_metrics_keep_empty_samples_and_compute_oracle_mrr() -> None:
    manifest = pd.DataFrame({"sample_id": ["s1", "s2", "s3"]})
    candidates = pd.DataFrame(
        [
            {
                **_candidate("s1", "predicted", "p1", 1),
                "candidate_success": True,
            },
            {
                **_candidate("s2", "predicted", "p2", 1),
                "candidate_success": False,
            },
            {
                **_candidate("s2", "predicted", "p3", 2),
                "candidate_success": True,
            },
        ]
    )
    outcomes, metrics = compute_branch_metrics(
        candidates, manifest, route="G1", branch="predicted", k_values=(5,)
    )
    assert metrics["N"] == 3
    assert metrics["no_output"] == 1
    assert metrics["native_j_at_1"] == pytest.approx(1 / 3)
    assert metrics["oracle_at_5"] == pytest.approx(2 / 3)
    assert metrics["oracle_all"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert outcomes.set_index("sample_id").loc["s3", "no_output"]

    gt_outcomes = outcomes.copy()
    gt_outcomes["branch"] = "gt_oracle"
    gt_outcomes.loc[gt_outcomes["sample_id"].eq("s2"), "native_correct"] = True
    gt_outcomes.loc[gt_outcomes["sample_id"].eq("s2"), "first_positive_rank"] = 1
    gt_outcomes.loc[gt_outcomes["sample_id"].eq("s3"), "oracle_all"] = True
    gt_outcomes.loc[gt_outcomes["sample_id"].eq("s3"), "first_positive_rank"] = 3
    _, comparison = compare_branch_outcomes(outcomes, gt_outcomes)
    assert comparison["pred_no_positive_to_gt_positive"] == 1
    assert comparison["grounding_candidate_recovery_denominator"] == 1
    assert comparison["grounding_candidate_recovery_rate"] == 1.0


def _taxonomy_row(label: str) -> dict[str, object]:
    values: dict[str, object] = {
        "sample_id": label,
        "route": "D1",
        "technical_failure": False,
        "pred_native_correct": False,
        "pred_top5_positive": False,
        "pred_top10_positive": False,
        "pred_all_positive": False,
        "gt_native_correct": False,
        "gt_top5_positive": False,
        "gt_top10_positive": False,
        "gt_all_positive": False,
    }
    if label == "T0":
        values["technical_failure"] = True
    elif label == "T1":
        values.update(
            pred_native_correct=True,
            pred_top5_positive=True,
            pred_top10_positive=True,
            pred_all_positive=True,
        )
    elif label == "T2":
        values.update(
            pred_top5_positive=True,
            pred_top10_positive=True,
            pred_all_positive=True,
        )
    elif label == "T3":
        values.update(pred_top10_positive=True, pred_all_positive=True)
    elif label == "T4":
        values.update(
            gt_native_correct=True,
            gt_top5_positive=True,
            gt_top10_positive=True,
            gt_all_positive=True,
        )
    elif label == "T5":
        values.update(
            gt_top5_positive=True, gt_top10_positive=True, gt_all_positive=True
        )
    elif label == "T6":
        values.update(gt_top10_positive=True, gt_all_positive=True)
    return values


def test_native_taxonomy_is_mutually_exclusive_exhaustive_with_d1_flags() -> None:
    classified = classify_native_taxonomy(
        pd.DataFrame([_taxonomy_row(f"T{index}") for index in range(8)])
    )
    assert classified["native_taxonomy"].tolist() == list(NATIVE_CLASSES)
    assert taxonomy_counts(classified, column="native_taxonomy") == {
        label: 1 for label in NATIVE_CLASSES
    }
    assert classified.set_index("sample_id").loc["T3", "pred_deep_rank_flag"] == (
        "rank_6_to_10"
    )
    assert classified.set_index("sample_id").loc["T6", "gt_deep_rank_flag"] == (
        "rank_6_to_10"
    )

    invalid = pd.DataFrame([_taxonomy_row("T2")])
    invalid["pred_all_positive"] = False
    with pytest.raises(ValueError, match="must imply"):
        classify_native_taxonomy(invalid)


def test_post_r7_taxonomy_is_exhaustive_only_over_residual_failures() -> None:
    rows = []
    for index in range(6):
        row = _taxonomy_row("T7")
        row["sample_id"] = f"R{index}"
        row["final_correct"] = False
        if index == 0:
            row["technical_failure"] = True
        elif index == 1:
            row.update(pred_top5_positive=True, pred_all_positive=True)
        elif index == 2:
            row.update(
                gt_native_correct=True, gt_top5_positive=True, gt_all_positive=True
            )
        elif index == 3:
            row.update(gt_top5_positive=True, gt_all_positive=True)
        elif index == 4:
            row["gt_all_positive"] = True
        rows.append(row)
    success = _taxonomy_row("T7")
    success.update(sample_id="success", final_correct=True)
    rows.append(success)
    classified = classify_post_r7_taxonomy(pd.DataFrame(rows))
    assert classified["post_r7_taxonomy"].tolist() == [*POST_R7_CLASSES, ""]
    assert taxonomy_counts(classified, column="post_r7_taxonomy") == {
        label: 1 for label in POST_R7_CLASSES
    }


def test_statistical_pair_alignment_exact_mcnemar_holm_and_bootstrap() -> None:
    reference = [0, 1, 0, 1, 0, 1]
    counterfactual = [1, 1, 0, 0, 1, 1]
    exact = exact_mcnemar(reference, counterfactual)
    assert exact["b_reference_only"] == 1
    assert exact["c_counterfactual_only"] == 2
    assert exact["N"] == 6
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    family = apply_holm_family([{"raw_p": 0.01}, {"raw_p": 0.04}])
    assert [row["holm_adjusted_p"] for row in family] == pytest.approx([0.02, 0.04])

    clusters = ["a", "a", "b", "b", "c", "c"]
    first = cluster_bootstrap_delta(
        reference, counterfactual, clusters, iterations=200, seed=20260813
    )
    second = cluster_bootstrap_delta(
        reference, counterfactual, clusters, iterations=200, seed=20260813
    )
    assert first == second
    assert first["point_estimate"] == pytest.approx(1 / 6)
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "scene_id": clusters,
            "frame_id": ["f1", "f1", "f2", "f2", "f3", "f3"],
            "pred": reference,
            "gt": counterfactual,
        }
    )
    stats = paired_metric_statistics(
        frame,
        reference_column="pred",
        counterfactual_column="gt",
        iterations=200,
    )
    assert stats["scene_cluster_bootstrap"]["seed"] == 20260813
    assert stats["frame_cluster_bootstrap_sensitivity"]["iterations"] == 200

    left = pd.DataFrame({"sample_id": ["a", "b"], "correct": [0, 1]})
    right = pd.DataFrame({"sample_id": ["b", "a"], "correct": [1, 1]})
    aligned = align_paired_frames(left, right)
    assert aligned["sample_id"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="identity sets differ"):
        align_paired_frames(left, right.iloc[[0]])


def test_independent_recompute_exactly_checks_saved_frames_and_aggregates() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "scene_id": ["scene-a", "scene-b"],
            "frame_id": ["frame-a", "frame-b"],
        }
    )
    gt = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "gt_grasp_rectangles": [[_gt_corners()], [_gt_corners()]],
        }
    )
    geometry = pd.DataFrame(
        [
            _candidate("s1", "predicted", "p1", 1),
            _candidate("s1", "gt_oracle", "g1", 1),
            _candidate("s2", "predicted", "p2", 1, success_geometry=False),
            _candidate("s2", "gt_oracle", "g2", 1),
        ]
    )
    final = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "route": ["G1", "G1"],
            "final_correct": [True, False],
        }
    )
    first = independent_recompute_from_frames(
        manifest,
        geometry,
        gt,
        k_by_route={"G1": (5,)},
        final_outcomes=final,
        taxonomy_definitions=taxonomy_definitions(),
    )
    assert first["status"] == "PASS"
    assert first["native_taxonomy"].set_index("sample_id").loc[
        "s2", "native_taxonomy"
    ] == NATIVE_CLASSES[4]
    assert first["post_r7_taxonomy"].set_index("sample_id").loc[
        "s2", "post_r7_taxonomy"
    ] == POST_R7_CLASSES[2]

    second = independent_recompute_from_frames(
        manifest,
        geometry,
        gt,
        k_by_route={"G1": (5,)},
        final_outcomes=final,
        taxonomy_definitions=taxonomy_definitions(),
        saved_candidate_labels=first["candidate_labels"],
        saved_sample_outcomes=first["sample_outcomes"],
        saved_branch_metrics=first["branch_metrics"],
        saved_native_taxonomy=first["native_taxonomy"],
        saved_post_r7_taxonomy=first["post_r7_taxonomy"],
        saved_statistical_inputs=first["statistical_inputs"],
        saved_paired_transitions=first["paired_transitions"],
    )
    assert all(second["exact_checks"].values())

    corrupted = first["candidate_labels"].copy()
    corrupted.loc[0, "candidate_success"] = False
    with pytest.raises(AssertionError, match="candidate_success"):
        independent_recompute_from_frames(
            manifest,
            geometry,
            gt,
            k_by_route={"G1": (5,)},
            saved_candidate_labels=corrupted,
        )
