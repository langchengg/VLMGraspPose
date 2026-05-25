from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from evaluation.metrics import (
    is_proxy_valid,
    mean_or_zero,
    topk_2d_grasp_center_hit_rate,
    topk_2d_grasp_rectangle_match_rate,
    topk_valid_rate,
)


class OutputEvaluator:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds

    def load_best_grasps(self, output_root: Path) -> list[dict]:
        records = []
        for path in sorted(Path(output_root).glob("**/best_grasp.json")):
            if "paper_figures" in path.parts:
                continue
            with open(path) as f:
                rec = json.load(f)
            rec["_path"] = str(path)
            records.append(rec)
        return records

    def load_error_records(self, output_root: Path) -> list[dict]:
        records = []
        for path in sorted(Path(output_root).glob("**/error.json")):
            if "paper_figures" in path.parts:
                continue
            with open(path) as f:
                rec = json.load(f)
            sample = rec.get("sample", {})
            metadata = rec.get("metadata", {})
            sample_metadata = sample.get("metadata", {})
            records.append({
                "dataset": sample.get("dataset_name") or sample_metadata.get("dataset") or metadata.get("dataset") or "unknown",
                "split": sample.get("split") or "unknown",
                "scene_id": sample.get("scene_id") or "unknown",
                "target_source": metadata.get("target_source_requested") or "unknown",
                "scorer": metadata.get("scorer") or "unknown",
                "runtime": rec.get("runtime", {}),
                "error": rec.get("error_message"),
                "_path": str(path),
            })
        return records

    def evaluate_records(self, records: list[dict], mode: str = "proxy", failures: int = 0) -> dict:
        n = len(records)
        processed = n + int(failures)
        features = [r.get("feature_breakdown", {}) for r in records]
        row = {
            "processed_frames": processed,
            "successful_frames": n,
            "failure_rate": (float(failures) / processed) if processed else 0.0,
            "mean_final_score": mean_or_zero([r.get("final_score", 0.0) for r in records]),
            "mean_target_overlap": mean_or_zero([f.get("target_overlap", 0.0) for f in features]),
            "mean_center_distance": mean_or_zero([f.get("distance_to_target_center", 0.0) for f in features]),
            "mean_collision_penalty": mean_or_zero([f.get("collision_penalty", 1.0) for f in features]),
            "collision_free_proxy_rate": mean_or_zero([
                1.0 if f.get("collision_penalty", 1.0) < self.thresholds.get("collision_penalty", 0.5) else 0.0
                for f in features
            ]),
            "top1_proxy_valid_rate": topk_valid_rate(records, 1, self.thresholds),
            "top3_proxy_valid_rate": topk_valid_rate(records, 3, self.thresholds),
            "top5_proxy_valid_rate": topk_valid_rate(records, 5, self.thresholds),
            "mean_runtime_per_frame": mean_or_zero([
                sum(r.get("runtime", {}).values()) for r in records
            ]),
        }
        if mode == "annotation":
            row.update({
                "top1_annotation_valid_rate": self._annotation_topk_rate(records, 1),
                "top3_annotation_valid_rate": self._annotation_topk_rate(records, 3),
                "top5_annotation_valid_rate": self._annotation_topk_rate(records, 5),
            })
        if mode == "ocid_2d":
            iou_threshold = self.thresholds.get("grasp_rectangle_iou", self.thresholds.get("rectangle_iou", 0.25))
            angle_threshold = self.thresholds.get("grasp_angle_threshold_deg", self.thresholds.get("angle_threshold_deg", 30.0))
            row.update({
                "top1_2d_grasp_center_hit_rate": topk_2d_grasp_center_hit_rate(records, 1),
                "top3_2d_grasp_center_hit_rate": topk_2d_grasp_center_hit_rate(records, 3),
                "top5_2d_grasp_center_hit_rate": topk_2d_grasp_center_hit_rate(records, 5),
                "top1_2d_rectangle_match_rate": topk_2d_grasp_rectangle_match_rate(records, 1, iou_threshold, angle_threshold),
                "top3_2d_rectangle_match_rate": topk_2d_grasp_rectangle_match_rate(records, 3, iou_threshold, angle_threshold),
                "top5_2d_rectangle_match_rate": topk_2d_grasp_rectangle_match_rate(records, 5, iou_threshold, angle_threshold),
            })
        return row

    def _annotation_topk_rate(self, records: list[dict], k: int) -> float:
        if not records:
            return 0.0
        ok = 0
        for record in records:
            candidates = [self._candidate_from_record(record)] + [
                self._candidate_from_scored(c) for c in record.get("top_k_fallback_candidates", [])[: max(k - 1, 0)]
            ]
            annotations = record.get("annotation_valid_grasps", []) or record.get("valid_annotated_grasps", [])
            ok += any(
                self._matches_any_annotation(candidate, annotations)
                for candidate in candidates[:k]
                if candidate is not None
            )
        return ok / len(records)

    def _candidate_from_record(self, record: dict) -> dict:
        return {
            "position": record.get("best_grasp_position"),
            "orientation": record.get("best_grasp_orientation_quaternion"),
            "gripper_width": record.get("gripper_width"),
        }

    def _candidate_from_scored(self, scored: dict) -> dict | None:
        candidate = scored.get("candidate")
        if not candidate:
            return None
        return {
            "position": candidate.get("position"),
            "orientation": candidate.get("orientation"),
            "gripper_width": candidate.get("gripper_width"),
        }

    def _matches_any_annotation(self, candidate: dict, annotations: list[dict]) -> bool:
        for ann in annotations:
            if self._matches_annotation(candidate, ann):
                return True
        return False

    def _matches_annotation(self, candidate: dict, ann: dict) -> bool:
        if not ann:
            return False
        try:
            pos = np.asarray(candidate["position"], dtype=float)
            ann_pos = np.asarray(ann.get("position") or ann.get("translation"), dtype=float)
            quat = np.asarray(candidate["orientation"], dtype=float)
            ann_quat = np.asarray(ann.get("orientation") or ann.get("quaternion"), dtype=float)
            width = float(candidate.get("gripper_width", 0.0))
            ann_width = float(ann.get("gripper_width", ann.get("width", width)))
        except (TypeError, ValueError):
            return False
        if np.linalg.norm(pos - ann_pos) > self.thresholds.get("position_distance", 0.03):
            return False
        angle = Rotation.from_quat(quat).inv() * Rotation.from_quat(ann_quat)
        if abs(angle.magnitude()) > np.deg2rad(self.thresholds.get("orientation_angle_deg", 30.0)):
            return False
        return abs(width - ann_width) <= self.thresholds.get("width_tolerance", 0.02)
