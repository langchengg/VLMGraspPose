from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import TargetAwareGraspPipeline, load_config
from utils.data_types import DatasetSample


def test_full_synthetic_pipeline_smoke(tmp_path: Path) -> None:
    h, w = 96, 96
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = [30, 30, 30]
    rgb[28:68, 30:70] = [220, 30, 30]
    yy, xx = np.mgrid[0:h, 0:w]
    depth = np.full((h, w), 1200, dtype=np.uint16)
    depth[28:68, 30:70] = (900 + ((xx[28:68, 30:70] + yy[28:68, 30:70]) % 9)).astype(np.uint16)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[28:68, 30:70] = 1

    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(depth_path), depth)
    cv2.imwrite(str(mask_path), mask)

    sample = DatasetSample(
        dataset_name="synthetic",
        sample_id="synthetic_000",
        rgb_path=rgb_path,
        depth_path=depth_path,
        sentence="pick the red box",
        target_label="red box",
        target_id=1,
        target_index=1,
        target_bbox_gt=[30, 28, 69, 67],
        target_mask_path=mask_path,
        output_dir=tmp_path / "outputs" / "synthetic_000" / "target_001",
    )
    config = load_config(ROOT / "configs")
    config["ocid_vlg"]["depth_scale"] = 1000.0
    config["ocid_vlg"]["fallback_intrinsics"] = {
        "width": w,
        "height": h,
        "fx": 100.0,
        "fy": 100.0,
        "cx": w / 2.0,
        "cy": h / 2.0,
    }
    result = TargetAwareGraspPipeline(config).run_dataset_sample(
        sample,
        target_source="oracle",
        scorer="rule_based",
        top_k=3,
        overwrite=True,
    )
    assert result.status == "success", result.error_message
    assert result.best_grasp is not None
    assert np.isfinite(result.best_grasp.final_score)
    assert (sample.output_dir / "best_grasp.json").exists()
    assert (sample.output_dir / "ranked_grasps.json").exists()
    assert (sample.output_dir / "visualization_rgb.png").exists()
    assert (sample.output_dir / "target_pointcloud.ply").exists()
