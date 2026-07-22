from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from src.experiments.render_gallery import build_gallery, render_sample_webp
from src.experiments.report_builder import REAL_ROBOT_ABSENCE_REASON, build_report


def _image_inputs(root: Path) -> dict[str, object]:
    rgb_path = root / "rgb.png"
    mask_path = root / "mask.png"
    gt_path = root / "gt.png"
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[15:35, 20:45] = 255
    gt = np.zeros_like(mask)
    gt[14:34, 21:46] = 255
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(gt).save(gt_path)
    return {
        "sample_id": "sample-a",
        "instruction": "grasp the test object",
        "rgb_path": rgb_path,
        "mask_path": mask_path,
        "intrinsics_path": root / "intrinsics.json",
        "gt_path": gt_path,
    }


def _candidate() -> dict[str, object]:
    return {
        "projected_uv": [32.0, 25.0],
        "inside_dilated_target_mask": True,
        "vgn_quality": 0.96,
        "width_m": 0.04,
        "T_camera_grasp": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_render_sample_webp_direct_and_failure_safe(tmp_path: Path) -> None:
    sample = _image_inputs(tmp_path)
    sample_dir = tmp_path / "samples" / "sample-a"
    sample_dir.mkdir(parents=True)
    sample["intrinsics_path"].write_text(  # type: ignore[union-attr]
        json.dumps({"width": 64, "height": 48, "fx": 60, "fy": 60, "cx": 32, "cy": 24}),
        encoding="utf-8",
    )
    candidate = _candidate()
    (sample_dir / "candidates.json").write_text(
        json.dumps({"all_official_vgn_candidates": [candidate]}), encoding="utf-8"
    )
    (sample_dir / "top1.json").write_text(
        json.dumps({"status": "ok", "candidate": candidate}), encoding="utf-8"
    )
    rendered = render_sample_webp(
        sample_dir, sample, gt_mask_path=sample["gt_path"], quality=75  # type: ignore[arg-type]
    )
    assert set(rendered) == {
        "rgb_mask_overlay",
        "candidates_2d_overlay",
        "top1_2d_overlay",
    }
    for path in rendered.values():
        assert path.is_file()
        with Image.open(path) as image:
            assert image.format == "WEBP"
            assert image.size == (64, 48)

    # A scientific/model failure is still a renderable terminal outcome.
    (sample_dir / "candidates.json").write_text(
        json.dumps({"all_official_vgn_candidates": []}), encoding="utf-8"
    )
    (sample_dir / "top1.json").write_text(
        json.dumps({"status": "no_official_grasp", "candidate": None}),
        encoding="utf-8",
    )
    failed = render_sample_webp(sample_dir, sample)
    assert all(path.is_file() for path in failed.values())


def test_gallery_links_exist_and_controls_are_present(tmp_path: Path) -> None:
    report = tmp_path / "report"
    sample_dir = tmp_path / "samples" / "sample-a"
    sample_dir.mkdir(parents=True)
    for name in (
        "rgb_mask_overlay.webp",
        "candidates_2d_overlay.webp",
        "top1_2d_overlay.webp",
    ):
        Image.new("RGB", (8, 8), "navy").save(sample_dir / name, format="WEBP")
    (sample_dir / "top1.json").write_text("{}\n", encoding="utf-8")
    (sample_dir / "grasps_3d.ply").write_text("ply\n", encoding="utf-8")
    gallery = build_gallery(
        [
            {
                "sample_id": "sample-a",
                "instruction": "pick blue object",
                "status": "ok",
                "query_type": "attribute",
                "target_category": "bottle",
                "pred_mask_iou": 0.75,
                "top1_vgn_quality": 0.96,
                "official_candidate_count": 3,
            }
        ],
        tmp_path / "samples",
        report / "gallery.html",
    )
    markup = gallery.read_text(encoding="utf-8")
    assert all(token in markup for token in ('id="status"', 'id="query"', 'id="category"', 'id="sort"'))
    links = re.findall(r'(?:src|href)="([^"#]+)"', markup)
    assert links
    for relative in links:
        assert (gallery.parent / relative).resolve().is_file(), relative


def test_report_keeps_offline_simulated_and_real_metrics_separate(tmp_path: Path) -> None:
    ocid = tmp_path / "ocid"
    (ocid / "metrics").mkdir(parents=True)
    (ocid / "samples").mkdir()
    aggregate = {
        "manifest_count": 4,
        "registered_row_count": 4,
        "status_counts": {"ok": 1, "no_target_grasp": 1, "support_plane_failed": 2},
        "proportions": {
            "official_candidate_availability": {
                "numerator": 2,
                "denominator": 2,
                "estimate": 1.0,
                "ci_lower": 0.34,
                "ci_upper": 1.0,
                "method": "wilson",
            },
            "target_candidate_availability": {
                "numerator": 1,
                "denominator": 2,
                "estimate": 0.5,
                "ci_lower": 0.09,
                "ci_upper": 0.91,
                "method": "wilson",
            },
        },
        "truthfulness": {
            "all_scores_from_official_processed_quality": True,
            "any_custom_reranking": False,
        },
    }
    (ocid / "metrics" / "aggregate_metrics.json").write_text(
        json.dumps(aggregate), encoding="utf-8"
    )
    (ocid / "metrics" / "per_sample.csv").write_text(
        "sample_id,status,official_candidate_count,target_candidate_count\n"
        "sample-a,ok,2,1\n",
        encoding="utf-8",
    )
    simulation = tmp_path / "simulation"
    simulation.mkdir()
    (simulation / "aggregate.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "metric_scope": "pybullet_simulated_physical_execution",
                "simulated_grasp_success_rate": None,
                "reason": "simulation preflight failed",
            }
        ),
        encoding="utf-8",
    )
    paths = build_report(ocid, simulation, tmp_path / "report")
    executive = json.loads(paths["executive_summary"].read_text(encoding="utf-8"))
    offline = executive["offline_candidate_coverage"]
    assert offline["official"]["metric_name"] == "official_vgn_candidate_coverage"
    assert offline["physical_success_claimed"] is False
    assert executive["simulated_physical_success"]["value"] is None
    assert executive["simulated_physical_success"]["reason"] == "simulation preflight failed"
    assert executive["real_robot_success"]["value"] is None
    assert executive["real_robot_success"]["reason"] == REAL_ROBOT_ABSENCE_REASON
    assert executive["scope_separation_verified"] is True

    report = paths["report_md"].read_text(encoding="utf-8")
    assert "offline grasp success rate" not in report.lower()
    assert "`simulated_grasp_success_rate`" in report
    assert "`real_robot_grasp_success_rate`: **null**" in report
    assert "auxiliary cross-representation metric" in report
    assert "not 6-DoF ground truth" in report

