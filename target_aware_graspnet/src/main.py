from __future__ import annotations

from pathlib import Path
import traceback

import cv2
import numpy as np
import pandas as pd
import yaml

from association.feature_extractor import CandidateFeatureExtractor
from dataset.graspnet_loader import GraspNetLoader
from grasp_sampler.geometric_sampler import GeometricGraspSampler
from pointcloud.processor import PointCloudProcessor
from pointcloud.rgbd_to_pointcloud import save_pointcloud
from scoring.rule_based_scorer import RuleBasedScorer
from target.target_selector import TargetSelector
from utils.data_types import FrameResult, GraspNetSample
from utils.io_utils import ensure_dir, save_json
from utils.timing import timed
from visualization.visualize_pointcloud import save_pointcloud_figure
from visualization.visualize_rgb import save_rgb_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_dir: Path | None = None) -> dict:
    config_dir = config_dir or PROJECT_ROOT / "configs"
    merged = {}
    for name in ["default.yaml", "dataset.yaml", "sampler.yaml", "scoring.yaml", "evaluation.yaml"]:
        path = config_dir / name
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            key = name.replace(".yaml", "")
            merged[key] = data
    if "processing" not in merged:
        merged["processing"] = merged.get("default", {}).get("processing", {})
    return merged


class TargetAwareGraspPipeline:
    def __init__(self, config: dict):
        self.config = config
        default = config.get("default", {})
        dataset_cfg = config.get("dataset", {})
        self.loader = GraspNetLoader(
            depth_scale=dataset_cfg.get("depth_scale", default.get("dataset", {}).get("depth_scale", 1000.0)),
            fallback_intrinsics=dataset_cfg.get("fallback_intrinsics", default.get("dataset", {}).get("fallback_intrinsics")),
        )
        processing_cfg = default.get("processing", {}) | config.get("processing", {})
        processing_cfg["depth_trunc"] = dataset_cfg.get("depth_trunc", default.get("dataset", {}).get("depth_trunc", 2.0))
        self.pointcloud = PointCloudProcessor(processing_cfg)
        self.sampler_cfg = default.get("sampler", {}) | config.get("sampler", {})
        self.sampler = GeometricGraspSampler(self.sampler_cfg)
        self.features = CandidateFeatureExtractor(self.sampler_cfg)
        scoring_cfg = default.get("scoring", {}).get("weights", {}) | config.get("scoring", {}).get("weights", {})
        self.scorer = RuleBasedScorer(scoring_cfg)

    def run_sample(
        self,
        sample: GraspNetSample,
        target_mode: str = "pseudo",
        target_id: int | None = None,
        target_label: str | None = None,
        target_bbox: list[int] | None = None,
        target_mask: Path | None = None,
        command: str | None = None,
        top_k: int = 5,
        overwrite: bool = False,
    ) -> FrameResult:
        out = sample.output_dir
        best_path = out / "best_grasp.json"
        if best_path.exists() and not overwrite:
            return FrameResult(
                sample,
                None,
                [],
                None,
                {},
                "skipped",
                metadata={"target_id": target_id, "target_label": target_label, "command": command},
            )
        ensure_dir(out)
        runtime = {}
        try:
            with timed("load_sample", runtime):
                data = self.loader.load_sample(sample)
            with timed("target_selection", runtime):
                selector = TargetSelector(target_mode)
                target = selector.select(
                    sample,
                    data["rgb"],
                    data.get("label"),
                    target_id=target_id,
                    bbox=target_bbox,
                    mask_path=target_mask,
                )
                if target_label:
                    target.label = target_label
                if command:
                    target.command = command
                    target.metadata["language_source"] = "object_language_mapping"
            with timed("pointcloud_processing", runtime):
                pcr = self.pointcloud.process(data["rgb"], data["depth"], data["intrinsics"], target)
                if len(pcr.clean_target_pcd.points) == 0:
                    raise ValueError("Target point cloud is empty.")
                target.center_3d = pcr.target_center_3d
            with timed("sampling", runtime):
                candidates = self.sampler.sample(pcr, top_k=max(top_k * 3, top_k))
                if not candidates:
                    raise ValueError("No grasp candidates generated.")
            with timed("feature_extraction", runtime):
                feats = self.features.extract(candidates, target, pcr, data["depth"], data["intrinsics"])
            with timed("scoring", runtime):
                scored = self.scorer.top_k(candidates, feats, top_k)
                best = scored[0] if scored else None
                if best is None:
                    raise ValueError("No scored grasps available.")
            result = FrameResult(
                sample,
                target,
                scored,
                best,
                runtime,
                "success",
                metadata={"num_candidates": len(candidates), "grasp_candidates": [c.to_json() for c in candidates]},
            )
            self._save_outputs(result, data, pcr)
            return result
        except Exception as exc:
            result = FrameResult(
                sample=sample,
                target_region=None,
                top_k=[],
                best_grasp=None,
                runtime=runtime,
                status="failed",
                error_message=f"{type(exc).__name__}: {exc}",
                metadata={"traceback": traceback.format_exc()},
            )
            save_json(out / "error.json", result.to_json())
            return result

    def _save_outputs(self, result: FrameResult, data: dict, pcr) -> None:
        out = result.sample.output_dir
        target = result.target_region
        top_k = result.top_k
        best = result.best_grasp
        if target.mask is not None:
            cv2.imwrite(str(out / "target_mask.png"), (target.mask.astype(np.uint8) * 255))
        else:
            cv2.imwrite(str(out / "target_mask.png"), np.zeros(data["depth"].shape, dtype=np.uint8))
        save_pointcloud(out / "target_pointcloud.ply", pcr.clean_target_pcd)
        save_json(out / "grasp_candidates.json", result.metadata.get("grasp_candidates", [sg.candidate.to_json() for sg in top_k]))
        save_json(out / "ranked_grasps.json", [sg.to_json() for sg in top_k])
        save_json(out / "score_breakdown.json", {
            "features": [sg.features.to_json() for sg in top_k],
            "score_breakdown": [sg.metadata.get("score_breakdown", {}) for sg in top_k],
        })
        save_json(out / "best_grasp.json", self._best_grasp_json(result))
        save_rgb_overlay(out / "visualization_rgb.png", data["rgb"], target, best, data["intrinsics"])
        save_pointcloud_figure(out / "visualization_3d.png", pcr.clean_target_pcd, top_k)

    def _best_grasp_json(self, result: FrameResult) -> dict:
        sample = result.sample
        target = result.target_region
        best = result.best_grasp
        c = best.candidate
        fallback = [sg.to_json() for sg in result.top_k[1:]]
        return {
            "split": sample.split,
            "scene_id": sample.scene_id,
            "camera": sample.camera,
            "frame_id": sample.frame_id,
            "target_id": target.target_id,
            "target_label": target.label,
            "command": target.command,
            "target_bbox": target.bbox,
            "best_grasp_position": c.position.tolist(),
            "best_grasp_orientation_quaternion": c.orientation.tolist(),
            "approach_vector": c.approach_vector.tolist(),
            "closing_direction": c.closing_direction.tolist(),
            "gripper_width": float(c.gripper_width),
            "grasp_type": c.grasp_type,
            "final_score": float(best.final_score),
            "feature_breakdown": best.features.to_json(),
            "top_k_fallback_candidates": fallback,
            "runtime": result.runtime,
        }


def frame_result_row(result: FrameResult) -> dict:
    sample = result.sample
    row = {
        "split": sample.split,
        "scene_id": sample.scene_id,
        "camera": sample.camera,
        "frame_id": sample.frame_id,
        "target_id": result.target_region.target_id if result.target_region else result.metadata.get("target_id"),
        "command": result.target_region.command if result.target_region else result.metadata.get("command"),
        "status": result.status,
        "error": result.error_message,
        "runtime_total": sum(result.runtime.values()) if result.runtime else 0.0,
    }
    if result.best_grasp is not None:
        row["final_score"] = result.best_grasp.final_score
        row["target_overlap"] = result.best_grasp.features.target_overlap
        row["center_distance"] = result.best_grasp.features.distance_to_target_center
        row["collision_penalty"] = result.best_grasp.features.collision_penalty
    return row


def write_summary_csv(path: Path, results: list[FrameResult]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame([frame_result_row(r) for r in results]).to_csv(path, index=False)
