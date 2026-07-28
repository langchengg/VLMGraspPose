from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.analysis.diagnostic_features import (
    DiagnosticThresholds,
    diagnose_no_official_sample,
    grounding_diagnostic_features,
    official_postprocessing_stage_diagnostics,
    ranking_diagnostic_features,
)
from src.analysis.failure_taxonomy import (
    PRIMARY_FAILURE_CLASSES,
    FailureTaxonomyError,
    assign_failure_taxonomy,
    primary_failure_counts,
)
from src.grasping.vgn_adapter import PredictionResult


def _candidate(
    sample_id: str,
    index: int,
    *,
    positive: bool,
    passed: bool,
    selected: bool = False,
    quality: float = 0.95,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "candidate_index_original": index,
        "rank_vgn_all": index + 1,
        "vgn_quality": quality,
        "width_m": 0.04,
        "pred_filter_pass": passed,
        "is_baseline_top1": selected,
        "gt_target_positive_primary": positive,
        "projected_depth_difference_m": 0.0,
    }


def _complete_taxonomy_result() -> pd.DataFrame:
    samples = pd.DataFrame(
        [
            {"sample_id": "p0", "pred_status": "vgn_inference_error"},
            {
                "sample_id": "p1",
                "pred_status": "no_official_grasp",
                "n_official_candidates": 0,
                "n_pred_filtered_candidates": 0,
            },
            {
                "sample_id": "p2",
                "pred_status": "no_target_grasp",
                "n_official_candidates": 1,
                "n_pred_filtered_candidates": 0,
            },
            {
                "sample_id": "p3",
                "pred_status": "no_target_grasp",
                "n_official_candidates": 1,
                "n_pred_filtered_candidates": 0,
            },
            {
                "sample_id": "p4",
                "pred_status": "ok",
                "n_official_candidates": 2,
                "n_pred_filtered_candidates": 1,
            },
            {
                "sample_id": "p5",
                "pred_status": "ok",
                "n_official_candidates": 2,
                "n_pred_filtered_candidates": 2,
            },
            {
                "sample_id": "p6",
                "pred_status": "ok",
                "n_official_candidates": 1,
                "n_pred_filtered_candidates": 1,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            _candidate("p2", 0, positive=False, passed=False),
            _candidate("p3", 0, positive=True, passed=False),
            _candidate("p4", 0, positive=False, passed=True, selected=True),
            _candidate("p4", 1, positive=True, passed=False),
            _candidate("p5", 0, positive=False, passed=True, selected=True),
            _candidate("p5", 1, positive=True, passed=True, quality=0.94),
            _candidate("p6", 0, positive=True, passed=True, selected=True),
        ]
    )

    return assign_failure_taxonomy(samples, candidates)


def test_primary_taxonomy_mutually_exclusive() -> None:
    result = _complete_taxonomy_result()

    assert result["primary_failure_class"].tolist() == list(PRIMARY_FAILURE_CLASSES)
    assert primary_failure_counts(result) == {
        name: 1 for name in PRIMARY_FAILURE_CLASSES
    }


def test_primary_taxonomy_covers_all_samples() -> None:
    result = _complete_taxonomy_result()

    assert len(result) == len(PRIMARY_FAILURE_CLASSES)
    assert result["primary_failure_class"].notna().all()


def test_current_status_crosswalk() -> None:
    result = _complete_taxonomy_result()
    crosswalk = pd.crosstab(result["pred_status"], result["primary_failure_class"])

    assert int(crosswalk.to_numpy().sum()) == len(result)
    assert set(crosswalk.index) == {"no_official_grasp", "no_target_grasp", "ok", "vgn_inference_error"}
    assert set(crosswalk.columns) == set(PRIMARY_FAILURE_CLASSES)


def test_failure_taxonomy_rejects_inconsistent_frozen_top1() -> None:
    samples = pd.DataFrame(
        [
            {
                "sample_id": "bad",
                "pred_status": "ok",
                "n_official_candidates": 1,
                "n_pred_filtered_candidates": 1,
            }
        ]
    )
    candidates = pd.DataFrame(
        [_candidate("bad", 0, positive=True, passed=True, selected=False)]
    )
    with pytest.raises(FailureTaxonomyError, match="no unique baseline top-1"):
        assign_failure_taxonomy(samples, candidates)


def test_official_stage_diagnostics_localizes_surface_filter() -> None:
    tsdf = np.zeros((1, 40, 40, 40), dtype=np.float32)
    quality = np.ones((40, 40, 40), dtype=np.float32)
    width = np.full((40, 40, 40), 2.0, dtype=np.float32)

    result = official_postprocessing_stage_diagnostics(
        tsdf, quality, width, sample_id="surface"
    )

    assert result.raw_quality_max == 1.0
    assert result.count_raw_quality_above_0_90 == 40**3
    assert result.count_after_gaussian == 40**3
    assert result.count_after_surface_filter == 0
    assert result.count_after_width_filter == 0
    assert result.count_after_threshold == 0
    assert result.count_after_3d_local_maximum == 0
    assert result.first_zero_stage == "surface_filter"
    assert result.candidate_generation_secondary_flag == "S_removed_by_surface_filter"
    assert result.S_empty_or_sparse_tsdf is True


def test_official_stage_diagnostics_localizes_width_filter() -> None:
    tsdf = np.ones((1, 40, 40, 40), dtype=np.float32)
    quality = np.ones((40, 40, 40), dtype=np.float32)
    width = np.zeros((40, 40, 40), dtype=np.float32)

    result = official_postprocessing_stage_diagnostics(tsdf, quality, width)

    assert result.count_after_surface_filter == 40**3
    assert result.count_after_width_filter == 0
    assert result.first_zero_stage == "width_filter"
    assert result.candidate_generation_secondary_flag == "S_removed_by_width_filter"


def test_no_official_diagnostic_rebuilds_from_saved_frame_and_depth(tmp_path) -> None:
    depth_path = tmp_path / "depth.png"
    Image.fromarray(np.full((2, 2), 1000, dtype=np.uint16)).save(depth_path)
    workspace_path = tmp_path / "workspace_frame.json"
    workspace_path.write_text(
        json.dumps(
            {
                "T_camera_task": np.eye(4).tolist(),
                "intrinsics": {
                    "width": 2,
                    "height": 2,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.5,
                    "cy": 0.5,
                },
                "depth": {"depth_unit": "mm", "depth_scale": 1000.0},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_builder(depth_m, intrinsics, transform, **kwargs):
        captured["depth"] = depth_m.copy()
        captured["transform"] = np.asarray(transform).copy()
        captured["kwargs"] = kwargs
        return SimpleNamespace(grid=np.zeros((1, 40, 40, 40), dtype=np.float32))

    def fake_predictor(grid, net, device):
        del grid, net
        return PredictionResult(
            qual_vol=np.ones((40, 40, 40), dtype=np.float32),
            rot_vol=np.zeros((4, 40, 40, 40), dtype=np.float32),
            width_vol=np.full((40, 40, 40), 2.0, dtype=np.float32),
            requested_device=device,
            used_device=device,
        )

    result = diagnose_no_official_sample(
        {
            "sample_id": "failed",
            "pred_status": "no_official_grasp",
            "depth_path": str(depth_path),
            "workspace_frame_path": str(workspace_path),
        },
        net=object(),
        device="cpu",
        vgn_root=tmp_path,
        tsdf_builder=fake_builder,
        predictor=fake_predictor,
    )

    np.testing.assert_array_equal(captured["depth"], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(captured["transform"], np.eye(4))
    assert captured["kwargs"]["workspace_size_m"] == pytest.approx(0.30)
    assert captured["kwargs"]["resolution"] == 40
    assert result.sample_id == "failed"
    assert result.first_zero_stage == "surface_filter"
    assert result.device_used == "cpu"


def test_grounding_and_ranking_secondary_flags_are_explicit() -> None:
    target = np.zeros((20, 20), dtype=bool)
    target[5:15, 5:15] = True
    predicted = np.zeros_like(target)
    predicted[0:2, 0:2] = True
    predicted[18:20, 18:20] = True
    grounding = grounding_diagnostic_features(
        predicted, target, thresholds=DiagnosticThresholds()
    )
    assert grounding["S_mask_zero_overlap"] is True
    assert grounding["S_mask_undersegmented"] is True
    assert grounding["S_mask_fragmented"] is True
    assert grounding["S_mask_wrong_object_likely"] is True

    candidates = pd.DataFrame(
        [
            _candidate("x", 0, positive=False, passed=True, selected=True),
            _candidate("x", 1, positive=True, passed=False, quality=0.94),
        ]
    )
    ranking = ranking_diagnostic_features({"sample_id": "x"}, candidates)
    assert ranking["S_multiple_official_candidates"] is True
    assert ranking["S_gt_positive_rank_2_3"] is True
    assert ranking["S_pred_filter_false_negative"] is True
    assert ranking["S_pred_filter_false_positive"] is True
    assert ranking["S_small_quality_gap"] is True
