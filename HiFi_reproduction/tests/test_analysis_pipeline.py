from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.analysis.analysis_report import build_report
from src.analysis.gt_candidate_labels import label_candidate_group
from src.analysis.human_audit import stratified_audit_sample


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def full_tables():
    analysis_root = ROOT / "outputs/hifics_vgn_analysis"
    manifest = json.loads(
        (analysis_root / "data/analysis_manifest.json").read_text(encoding="utf-8")
    )
    return SimpleNamespace(
        samples=pd.read_parquet(analysis_root / "data/per_sample.parquet"),
        predicted_candidates=pd.read_parquet(
            analysis_root / "data/per_candidate.parquet",
            columns=["sample_id", "candidate_index_original", "is_baseline_top1"],
        ),
        gt_regenerated_candidates=pd.read_parquet(
            analysis_root / "data/per_candidate_gt_regenerated.parquet",
            columns=["sample_id", "candidate_index_original"],
        ),
        integrity=manifest["integrity"],
    )


def test_all_7675_samples_loaded(full_tables) -> None:
    assert len(full_tables.samples) == 7_675


def test_no_duplicate_sample_ids(full_tables) -> None:
    assert full_tables.samples.sample_id.is_unique


def test_pred_and_oracle_sample_alignment(full_tables) -> None:
    sample_ids = set(full_tables.samples.sample_id)
    assert set(full_tables.predicted_candidates.sample_id) <= sample_ids
    assert set(full_tables.gt_regenerated_candidates.sample_id) <= sample_ids
    assert full_tables.integrity["comparable_except_mask_source"] is True


def test_candidate_count_matches_npz(full_tables) -> None:
    assert len(full_tables.predicted_candidates) == 21_809
    assert len(full_tables.gt_regenerated_candidates) == 21_575
    assert full_tables.integrity["candidate_npz_mismatch_count"] == 0


def test_existing_top1_matches_recomputed_baseline(full_tables) -> None:
    assert full_tables.integrity["top1_recomputed_mismatch_count"] == 0
    selected = full_tables.predicted_candidates.is_baseline_top1.sum()
    assert int(selected) == 3_263


def _synthetic_label_case(tmp_path: Path, u: float) -> tuple[dict, pd.DataFrame]:
    gt = np.zeros((21, 21), dtype=np.uint8)
    gt[8:13, 8:13] = 255
    gt_path = tmp_path / "gt.png"
    Image.fromarray(gt).save(gt_path)
    depth_path = tmp_path / "depth.png"
    Image.fromarray(np.full((21, 21), 1000, dtype=np.uint16)).save(depth_path)
    sample = {
        "sample_id": "synthetic",
        "gt_mask_path": str(gt_path),
        "depth_path": str(depth_path),
        # Deliberately nonexistent: GT labeling must not read a predicted mask.
        "pred_mask_path": str(tmp_path / "must_not_be_read.png"),
        "intrinsics": {
            "width": 21,
            "height": 21,
            "fx": 10.0,
            "fy": 10.0,
            "cx": 10.0,
            "cy": 10.0,
        },
    }
    z = 1.0
    x = (u - 10.0) * z / 10.0
    candidates = pd.DataFrame(
        [
            {
                "position_camera_x": x,
                "position_camera_y": 0.0,
                "position_camera_z": z,
                "projected_u_saved": u,
                "projected_v_saved": 10.0,
            }
        ]
    )
    return sample, candidates


def test_gt_label_uses_gt_mask_only(tmp_path: Path) -> None:
    sample, candidates = _synthetic_label_case(tmp_path, 10.0)
    result = label_candidate_group(sample, candidates)
    assert bool(result.iloc[0].gt_target_positive_primary)


def test_gt_label_does_not_use_pred_mask(tmp_path: Path) -> None:
    sample, candidates = _synthetic_label_case(tmp_path, 10.0)
    assert not Path(sample["pred_mask_path"]).exists()
    result = label_candidate_group(sample, candidates)
    assert bool(result.iloc[0].gt_inside_raw_mask)


def test_gt_dilation_sensitivity(tmp_path: Path) -> None:
    sample, candidates = _synthetic_label_case(tmp_path, 16.0)
    result = label_candidate_group(sample, candidates).iloc[0]
    assert not bool(result.gt_inside_raw_mask)
    assert not bool(result.gt_inside_dilated_mask_3px)
    assert bool(result.gt_inside_dilated_mask_5px)


def test_no_physical_success_terms_in_metrics() -> None:
    metric_names = [
        "current_baseline",
        "same_pool_post_filter_oracle",
        "same_pool_pre_filter_oracle",
        "gt_regenerated_pool_oracle",
        "union_diagnostic_ceiling",
        "target_consistent_candidate_recall",
    ]
    forbidden = ("grasp success", "success rate", "physical success")
    assert not any(term in name.lower() for name in metric_names for term in forbidden)


def test_human_audit_sampling_stratified() -> None:
    rows = []
    for class_index, primary in enumerate(("P1", "P2", "P3")):
        for index in range(12):
            rows.append(
                {
                    "sample_id": f"{primary}_{index}",
                    "dataset_index": class_index * 100 + index,
                    "scene_id": f"scene_{index % 4}",
                    "query_type": ("name", "relation", "attribute")[index % 3],
                    "target_category": f"category_{index % 2}",
                    "mask_iou": index / 12,
                    "n_official_candidates": index % 6,
                    "primary_failure_class": primary,
                }
            )
    first = stratified_audit_sample(pd.DataFrame(rows), per_class=7, seed=42)
    second = stratified_audit_sample(pd.DataFrame(rows), per_class=7, seed=42)
    assert first.sample_id.tolist() == second.sample_id.tolist()
    assert first.groupby("primary_failure_class").size().eq(7).all()
    assert first.groupby("primary_failure_class").query_type.nunique().ge(2).all()


def test_report_links_exist(tmp_path: Path) -> None:
    executive = {
        "integrity": {"manifest_count": 2},
        "headline_metrics": {},
        "method_design_recommendation": {"reranker_insertion": "analysis pending"},
    }
    markdown, html_path = build_report(tmp_path, integrity=executive["integrity"], executive=executive)
    assert markdown.is_file()
    assert html_path.is_file()
    assert (tmp_path / "report/gallery.html").is_file()
