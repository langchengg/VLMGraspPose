from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_full_ocid_vgn import _paired_oracle


def _row(
    sample_id: str,
    scene_id: str,
    status: str,
    official: int,
    target: int,
    quality: float | None,
    runtime: float,
    dataset_index: int,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "dataset_index": dataset_index,
        "status": status,
        "official_candidate_count": official,
        "target_candidate_count": target,
        "top1_vgn_quality": quality,
        "processing_time_total": runtime,
    }


def test_paired_oracle_exports_all_required_comparisons(tmp_path: Path) -> None:
    predicted = [
        _row("a", "scene-1", "ok", 4, 2, 0.95, 0.5, 0),
        _row("b", "scene-2", "no_target_grasp", 3, 0, None, 0.6, 1),
        _row("c", "scene-3", "no_official_grasp", 0, 0, None, 0.7, 2),
    ]
    oracle = [
        _row("a", "scene-1", "ok", 5, 1, 0.96, 0.4, 0),
        _row("b", "scene-2", "ok", 2, 1, 0.92, 0.5, 1),
        _row("c", "scene-3", "no_official_grasp", 0, 0, None, 0.6, 2),
    ]
    result = _paired_oracle(
        predicted,
        oracle,
        output=tmp_path,
        replicates=20,
        seed=42,
    )
    assert result["paired_sample_count"] == 3
    assert result["delta_definition"] == "gt_oracle minus predicted"
    for name in (
        "official_candidate_coverage",
        "target_candidate_coverage",
        "no_official_grasp_rate",
        "no_target_grasp_rate",
        "target_given_official_availability",
        "candidate_count_retention_ratio",
        "top1_vgn_quality",
        "processing_time_total",
    ):
        assert name in result
        assert "paired_scene_cluster_delta" in result[name]
    assert result["top1_vgn_quality"]["paired_finite_count"] == 1
    assert result["candidate_count_retention_ratio"]["paired_finite_count"] == 2
    assert result["processing_time_total"]["paired_finite_count"] == 3
    assert (tmp_path / "pred_vs_gt_oracle.csv").is_file()
    persisted = json.loads((tmp_path / "oracle_delta.json").read_text())
    assert persisted == result
