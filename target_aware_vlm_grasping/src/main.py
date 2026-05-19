from __future__ import annotations

from pathlib import Path
import traceback

import cv2
import numpy as np
import pandas as pd
import yaml

from association.feature_extractor import CandidateFeatureExtractor
from dataset.ocid_vlg_loader import OCIDVLGLoader
from grasp_sampler.geometric_sampler import GeometricGraspSampler
from pointcloud.pointcloud_processor import PointCloudProcessor
from pointcloud.rgbd_to_pointcloud import save_pointcloud
from scoring.factory import build_scorer
from target.oracle_grounder import OracleTargetGrounder
from target.vlm_grounder import VLMTargetGrounder
from utils.data_types import DatasetSample, FrameResult
from utils.geometry import grasp_rectangle_2d, project_points
from utils.io_utils import ensure_dir, save_json
from utils.timing import timed
from visualization.visualize_pointcloud import save_pointcloud_figure
from visualization.visualize_rgb import save_rgb_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge_dicts(*items: dict | None) -> dict:
    merged: dict = {}
    for item in items:
        if item:
            merged.update(item)
    return merged


def load_config(config_dir: Path | None = None) -> dict:
    config_dir = config_dir or PROJECT_ROOT / "configs"
    merged = {}
    for name in [
        "default.yaml",
        "dataset.yaml",
        "ocid_vlg.yaml",
        "target_grounding.yaml",
        "pointcloud.yaml",
        "sampler.yaml",
        "scoring.yaml",
        "mlp.yaml",
        "evaluation.yaml",
    ]:
        path = config_dir / name
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            merged[name.replace(".yaml", "")] = data
    if "processing" not in merged:
        merged["processing"] = merged.get("default", {}).get("processing", {})
    return merged


class TargetAwareGraspPipeline:
    """Executable OCID language-conditioned RGB-D grasping pipeline.

    The core path is CPU-only: target grounding can run in oracle mode without
    optional VLM packages, while VLM mode is lazy-loaded only when selected.
    """

    def __init__(self, config: dict):
        self.config = config
        default = config.get("default", {})
        dataset_cfg = config.get("dataset", {})
        ocid_cfg = config.get("ocid_vlg", {})

        self.ocid_loader = OCIDVLGLoader(
            depth_scale=ocid_cfg.get("depth_scale", dataset_cfg.get("depth_scale", 1000.0)),
            fallback_intrinsics=ocid_cfg.get(
                "fallback_intrinsics",
                dataset_cfg.get("fallback_intrinsics", default.get("dataset", {}).get("fallback_intrinsics")),
            ),
        )

        processing_cfg = _merge_dicts(
            default.get("processing", {}),
            config.get("pointcloud", {}),
            config.get("processing", {}),
        )
        processing_cfg.setdefault(
            "depth_trunc",
            dataset_cfg.get("depth_trunc", default.get("dataset", {}).get("depth_trunc", 2.0)),
        )
        self.pointcloud = PointCloudProcessor(processing_cfg)

        self.sampler_cfg = _merge_dicts(default.get("sampler", {}), config.get("sampler", {}))
        self.sampler = GeometricGraspSampler(self.sampler_cfg)
        self.features = CandidateFeatureExtractor(_merge_dicts(self.sampler_cfg, processing_cfg))

        self.scoring_cfg = _merge_dicts(default.get("scoring", {}), config.get("scoring", {}))
        self.scorer = build_scorer(self.scoring_cfg)
        self.target_grounding_cfg = _merge_dicts(default.get("target_grounding", {}), config.get("target_grounding", {}))
        self._vlm_grounders: dict[str, VLMTargetGrounder] = {}
        self._oracle_grounder = OracleTargetGrounder()

    def run_dataset_sample(
        self,
        sample: DatasetSample,
        target_source: str = "oracle",
        vlm_backend: str | None = None,
        scorer: str | None = None,
        mlp_checkpoint: Path | None = None,
        top_k: int = 5,
        overwrite: bool = False,
    ) -> FrameResult:
        out = sample.output_dir
        best_path = out / "best_grasp.json"
        if best_path.exists() and not overwrite:
            return FrameResult(sample, None, [], None, {}, "skipped", metadata={"output_dir": str(out)})
        ensure_dir(out)

        runtime: dict = {}
        data: dict = {}
        try:
            with timed("load_sample", runtime):
                data = self.ocid_loader.load_sample(sample)
            with timed("target_grounding", runtime):
                target = self._predict_target(sample, data["rgb"], target_source, vlm_backend)
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
                scorer_impl = self._build_scorer(scorer, mlp_checkpoint)
                scored = scorer_impl.top_k(candidates, feats, top_k)
                best = scored[0] if scored else None
                if best is None:
                    raise ValueError("No scored grasps available.")
            result = FrameResult(
                sample=sample,
                target_region=target,
                top_k=scored,
                best_grasp=best,
                runtime=runtime,
                status="success",
                metadata={
                    "dataset": sample.dataset_name,
                    "num_candidates": len(candidates),
                    "grasp_candidates": [c.to_json() for c in candidates],
                    "gt_grasp_rectangles": data.get("grasp_rectangles", []),
                    "intrinsics": data["intrinsics"].tolist(),
                    "target_source_requested": target_source,
                    "scorer": scorer or self.scoring_cfg.get("method", "rule_based"),
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
                metadata={
                    "traceback": traceback.format_exc(),
                    "dataset": sample.dataset_name,
                    "target_source_requested": target_source,
                },
            )
            save_json(out / "error.json", result.to_json())
            return result

    def run_ocid_sample(
        self,
        sample: DatasetSample,
        top_k: int = 5,
        overwrite: bool = False,
    ) -> FrameResult:
        target_source = self.target_grounding_cfg.get("target_source")
        if target_source is None:
            method = str(self.target_grounding_cfg.get("method", "oracle")).lower()
            target_source = "vlm" if method in {"florence2", "florence-2", "vlm"} else "oracle"
        return self.run_dataset_sample(
            sample,
            target_source=target_source,
            vlm_backend=self.target_grounding_cfg.get("vlm_backend") or self.target_grounding_cfg.get("backend"),
            top_k=top_k,
            overwrite=overwrite,
        )

    def _predict_target(
        self,
        sample: DatasetSample,
        rgb: np.ndarray,
        target_source: str,
        vlm_backend: str | None,
    ):
        source = str(target_source or "oracle").lower()
        if source == "oracle":
            return self._oracle_grounder.predict(sample, rgb)
        if source == "vlm":
            backend = vlm_backend or self.target_grounding_cfg.get("vlm_backend") or "florence2"
            if backend not in self._vlm_grounders:
                backend_cfg = self.target_grounding_cfg.get(str(backend), {})
                if backend in {"florence2_sam", "florence2", "florence-2"}:
                    backend_cfg = self.target_grounding_cfg.get("florence2", backend_cfg)
                self._vlm_grounders[backend] = VLMTargetGrounder(
                    backend_name=backend,
                    backend_config=backend_cfg,
                    fallback_to_oracle=bool(self.target_grounding_cfg.get("fallback_to_oracle", False)),
                )
            return self._vlm_grounders[backend].predict(sample, rgb)
        raise ValueError("target_source must be 'oracle' or 'vlm'.")

    def _build_scorer(self, scorer: str | None, mlp_checkpoint: Path | None):
        if not scorer and not mlp_checkpoint:
            return self.scorer
        cfg = dict(self.scoring_cfg)
        if scorer:
            cfg["method"] = scorer
        if mlp_checkpoint:
            cfg.setdefault("mlp", {})["checkpoint_path"] = str(mlp_checkpoint)
        return build_scorer(cfg)

    def _save_outputs(self, result: FrameResult, data: dict, pcr) -> None:
        out = result.sample.output_dir
        target = result.target_region
        top_k = result.top_k
        best = result.best_grasp
        if target is None or best is None:
            return
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
        if target is None or best is None:
            return result.to_json()
        c = best.candidate
        fallback = [sg.to_json() for sg in result.top_k[1:]]
        intrinsics = result.metadata.get("intrinsics")
        center_2d = None
        top_k_centers_2d = []
        best_rectangle_2d = None
        top_k_rectangles_2d = []
        if intrinsics is not None:
            K = np.asarray(intrinsics, dtype=float)
            center_2d = project_points(c.position.reshape(1, 3), K)[0].tolist()
            top_k_centers_2d = [
                project_points(sg.candidate.position.reshape(1, 3), K)[0].tolist()
                for sg in result.top_k
            ]
            best_rectangle_2d = grasp_rectangle_2d(c.position, c.closing_direction, c.gripper_width, K)
            top_k_rectangles_2d = [
                grasp_rectangle_2d(sg.candidate.position, sg.candidate.closing_direction, sg.candidate.gripper_width, K)
                for sg in result.top_k
            ]
        target_bbox_gt = sample.target_bbox_gt
        target_bbox_pred = target.metadata.get("target_bbox_pred") if target.target_source == "vlm" else None
        best_grasp = {
            "position": c.position.tolist(),
            "orientation_quaternion": c.orientation.tolist(),
            "approach_vector": c.approach_vector.tolist(),
            "closing_direction": c.closing_direction.tolist(),
            "gripper_width": float(c.gripper_width),
            "grasp_type": c.grasp_type,
            "final_score": float(best.final_score),
        }
        return {
            "dataset_name": sample.dataset_name,
            "dataset": sample.dataset_name,
            "split": sample.split,
            "sample_id": sample.sample_id,
            "image_id": sample.image_id,
            "scene_id": sample.scene_id,
            "camera": sample.camera,
            "frame_id": sample.frame_id,
            "command": target.command,
            "sentence": sample.sentence,
            "target_id": target.target_id,
            "target_label": target.label,
            "target_source": target.target_source,
            "target_bbox": target.bbox,
            "target_bbox_gt": target_bbox_gt,
            "target_bbox_pred": target_bbox_pred,
            "grounding_score": float(target.grounding_score),
            "target_metadata": target.metadata,
            "best_grasp": best_grasp,
            "best_grasp_position": best_grasp["position"],
            "best_grasp_center_2d": center_2d,
            "best_grasp_rectangle_2d": best_rectangle_2d,
            "top_k_grasp_centers_2d": top_k_centers_2d,
            "top_k_grasp_rectangles_2d": top_k_rectangles_2d,
            "best_grasp_orientation_quaternion": best_grasp["orientation_quaternion"],
            "approach_vector": best_grasp["approach_vector"],
            "closing_direction": best_grasp["closing_direction"],
            "gripper_width": best_grasp["gripper_width"],
            "grasp_type": best_grasp["grasp_type"],
            "final_score": best_grasp["final_score"],
            "scorer": best.scorer_type,
            "scorer_type": best.scorer_type,
            "feature_breakdown": best.features.to_json(),
            "gt_grasp_rectangles": result.metadata.get("gt_grasp_rectangles", []),
            "top_k_fallback_candidates": fallback,
            "runtime": result.runtime,
            "status": result.status,
        }


def frame_result_row(result: FrameResult) -> dict:
    sample = result.sample
    row = {
        "dataset_name": sample.dataset_name,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "camera": sample.camera,
        "frame_id": sample.frame_id,
        "image_id": sample.image_id,
        "target_source": result.target_source,
        "target_id": result.target_region.target_id if result.target_region else result.metadata.get("target_id"),
        "target_label": result.target_label,
        "command": result.command,
        "status": result.status,
        "error": result.error_message,
        "runtime_total": sum(result.runtime.values()) if result.runtime else 0.0,
    }
    if result.best_grasp is not None:
        row["scorer_type"] = result.best_grasp.scorer_type
        row["final_score"] = result.best_grasp.final_score
        row["target_overlap"] = result.best_grasp.features.target_overlap
        row["center_distance"] = result.best_grasp.features.distance_to_target_center
        row["collision_penalty"] = result.best_grasp.features.collision_penalty
    return row


def write_summary_csv(path: Path, results: list[FrameResult]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame([frame_result_row(r) for r in results]).to_csv(path, index=False)
