from __future__ import annotations

from pathlib import Path
import traceback

import cv2
import numpy as np
import pandas as pd
import yaml

from association.feature_extractor import CandidateFeatureExtractor
from dataset.graspnet_loader import GraspNetLoader
from dataset.ocid_vlg_loader import OCIDVLGLoader
from grasp_sampler.geometric_sampler import GeometricGraspSampler
from pointcloud.processor import PointCloudProcessor
from pointcloud.rgbd_to_pointcloud import save_pointcloud
from scoring.factory import build_scorer
from target.florence2_grounder import Florence2Grounder
from target.target_selector import TargetSelector
from utils.data_types import FrameResult, GraspNetSample, OCIDVLGSample
from utils.geometry import grasp_rectangle_2d, project_points
from utils.io_utils import ensure_dir, save_json
from utils.timing import timed
from visualization.visualize_pointcloud import save_pointcloud_figure
from visualization.visualize_rgb import save_rgb_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_dir: Path | None = None) -> dict:
    config_dir = config_dir or PROJECT_ROOT / "configs"
    merged = {}
    for name in [
        "default.yaml",
        "dataset.yaml",
        "ocid_vlg.yaml",
        "target_grounding.yaml",
        "sampler.yaml",
        "scoring.yaml",
        "evaluation.yaml",
    ]:
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
        ocid_cfg = config.get("ocid_vlg", {})
        self.ocid_loader = OCIDVLGLoader(
            depth_scale=ocid_cfg.get("depth_scale", 1000.0),
            fallback_intrinsics=ocid_cfg.get("fallback_intrinsics"),
        )
        processing_cfg = default.get("processing", {}) | config.get("processing", {})
        processing_cfg["depth_trunc"] = dataset_cfg.get("depth_trunc", default.get("dataset", {}).get("depth_trunc", 2.0))
        self.pointcloud = PointCloudProcessor(processing_cfg)
        self.sampler_cfg = default.get("sampler", {}) | config.get("sampler", {})
        self.sampler = GeometricGraspSampler(self.sampler_cfg)
        self.features = CandidateFeatureExtractor(self.sampler_cfg)
        self.scoring_cfg = default.get("scoring", {}) | config.get("scoring", {})
        self.scorer = build_scorer(self.scoring_cfg)
        self.target_grounding_cfg = default.get("target_grounding", {}) | config.get("target_grounding", {})
        self._florence2_grounder = None

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
                target = self._maybe_apply_florence2_grounding(data["rgb"], target, target.command or command or target.label)
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

    def run_ocid_sample(
        self,
        sample: OCIDVLGSample,
        top_k: int = 5,
        overwrite: bool = False,
    ) -> FrameResult:
        out = sample.output_dir
        dataset_name = sample.metadata.get("dataset", "OCID-VLG")
        best_path = out / "best_grasp.json"
        if best_path.exists() and not overwrite:
            return FrameResult(sample, None, [], None, {}, "skipped", metadata={"command": sample.command})
        ensure_dir(out)
        runtime = {}
        try:
            with timed("load_sample", runtime):
                data = self.ocid_loader.load_sample(sample)
                target = data["target"]
                target = self._maybe_apply_florence2_grounding(data["rgb"], target, sample.command)
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
                metadata={
                    "num_candidates": len(candidates),
                    "grasp_candidates": [c.to_json() for c in candidates],
                    "gt_grasp_rectangles": data.get("grasp_rectangles", []),
                    "intrinsics": data["intrinsics"].tolist(),
                    "dataset": dataset_name,
                },
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
                metadata={"traceback": traceback.format_exc(), "dataset": dataset_name},
            )
            save_json(out / "error.json", result.to_json())
            return result

    def _maybe_apply_florence2_grounding(self, rgb: np.ndarray, annotation_target, command: str):
        method = str(self.target_grounding_cfg.get("method", "annotation")).lower()
        if method not in {"florence2", "florence-2"}:
            annotation_target.metadata["target_grounder"] = "annotation"
            return annotation_target
        try:
            if self._florence2_grounder is None:
                florence_cfg = self.target_grounding_cfg.get("florence2", {})
                self._florence2_grounder = Florence2Grounder(florence_cfg)
            grounded = self._florence2_grounder.ground(
                rgb,
                command,
                target_id=annotation_target.target_id,
            )
            grounded.metadata["target_grounder"] = "florence2"
            grounded.metadata["annotation_bbox"] = annotation_target.bbox
            grounded.metadata["annotation_label"] = annotation_target.label
            return grounded
        except Exception as exc:
            if self.target_grounding_cfg.get("fallback_to_annotation", True):
                annotation_target.metadata["target_grounder"] = "annotation_fallback"
                annotation_target.metadata["florence2_error"] = f"{type(exc).__name__}: {exc}"
                return annotation_target
            raise

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
            "gt_grasp_rectangles": result.metadata.get("gt_grasp_rectangles", []),
        })
        save_json(out / "best_grasp.json", self._best_grasp_json(result))
        save_rgb_overlay(out / "visualization_rgb.png", data["rgb"], target, best, data["intrinsics"], top_k=top_k)
        save_pointcloud_figure(out / "visualization_3d.png", pcr.clean_target_pcd, top_k)

    def _best_grasp_json(self, result: FrameResult) -> dict:
        sample = result.sample
        target = result.target_region
        best = result.best_grasp
        c = best.candidate
        fallback = [sg.to_json() for sg in result.top_k[1:]]
        center_2d = None
        top_k_centers_2d = []
        best_rectangle_2d = None
        top_k_rectangles_2d = []
        intrinsics = result.metadata.get("intrinsics")
        if intrinsics is not None:
            K = np.asarray(intrinsics, dtype=float)
            center_2d = project_points(c.position.reshape(1, 3), K)[0].tolist()
            top_k_centers_2d = [
                project_points(sg.candidate.position.reshape(1, 3), K)[0].tolist()
                for sg in result.top_k
            ]
            best_rectangle_2d = grasp_rectangle_2d(c.position, c.closing_direction, c.gripper_width, K)
            top_k_rectangles_2d = [
                grasp_rectangle_2d(
                    sg.candidate.position,
                    sg.candidate.closing_direction,
                    sg.candidate.gripper_width,
                    K,
                )
                for sg in result.top_k
            ]
        return {
            "dataset": result.metadata.get("dataset") or sample.metadata.get("dataset"),
            "split": sample.split,
            "scene_id": sample.scene_id,
            "camera": sample.camera,
            "frame_id": sample.frame_id,
            "image_id": getattr(sample, "image_id", None),
            "target_id": target.target_id,
            "target_label": target.label,
            "command": target.command,
            "target_bbox": target.bbox,
            "grounding_score": float(target.grounding_score),
            "target_metadata": target.metadata,
            "best_grasp_position": c.position.tolist(),
            "best_grasp_center_2d": center_2d,
            "best_grasp_rectangle_2d": best_rectangle_2d,
            "top_k_grasp_centers_2d": top_k_centers_2d,
            "top_k_grasp_rectangles_2d": top_k_rectangles_2d,
            "best_grasp_orientation_quaternion": c.orientation.tolist(),
            "approach_vector": c.approach_vector.tolist(),
            "closing_direction": c.closing_direction.tolist(),
            "gripper_width": float(c.gripper_width),
            "grasp_type": c.grasp_type,
            "final_score": float(best.final_score),
            "feature_breakdown": best.features.to_json(),
            "gt_grasp_rectangles": result.metadata.get("gt_grasp_rectangles", []),
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
        "image_id": getattr(sample, "image_id", None),
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
