from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "to_json"):
        return value.to_json()
    return value


@dataclass
class GraspNetSample:
    split: str
    scene_id: str
    camera: str
    frame_id: str
    rgb_path: Path
    depth_path: Path
    annotation_path: Optional[Path]
    camera_intrinsic_path: Optional[Path]
    output_dir: Path
    label_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return _to_jsonable(asdict(self))


@dataclass
class TargetRegion:
    target_id: Optional[int]
    label: str
    bbox: Optional[list[int]]
    mask: Optional[np.ndarray]
    grounding_score: float
    center_2d: Optional[tuple[float, float]]
    center_3d: Optional[np.ndarray] = None
    command: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_json(self, include_mask: bool = False) -> dict:
        data = asdict(self)
        if not include_mask:
            data["mask"] = None
            data["has_mask"] = self.mask is not None
        return _to_jsonable(data)


@dataclass
class PointCloudRepresentation:
    scene_pcd: Any
    target_pcd: Any
    clean_target_pcd: Any
    table_plane: Optional[np.ndarray]
    target_center_3d: Optional[np.ndarray]
    target_aabb: Optional[Any]
    target_obb: Optional[Any]
    surface_normals: Optional[np.ndarray]
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "table_plane": _to_jsonable(self.table_plane),
            "target_center_3d": _to_jsonable(self.target_center_3d),
            "num_scene_points": len(self.scene_pcd.points) if self.scene_pcd is not None else 0,
            "num_target_points": len(self.target_pcd.points) if self.target_pcd is not None else 0,
            "num_clean_target_points": len(self.clean_target_pcd.points) if self.clean_target_pcd is not None else 0,
            "metadata": _to_jsonable(self.metadata),
        }


@dataclass
class GraspCandidate:
    position: np.ndarray
    orientation: np.ndarray
    approach_vector: np.ndarray
    closing_direction: np.ndarray
    gripper_width: float
    grasp_type: str
    initial_geometric_score: float
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return _to_jsonable(asdict(self))


@dataclass
class CandidateFeatureVector:
    target_overlap: float
    center_alignment: float
    distance_to_target_center: float
    gripper_width_match: float
    approach_direction_score: float
    depth_stability: float
    collision_penalty: float
    boundary_penalty: float
    initial_geometric_score: float
    grounding_score: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return _to_jsonable(asdict(self))


@dataclass
class ScoredGrasp:
    candidate: GraspCandidate
    features: CandidateFeatureVector
    final_score: float
    rank: int
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return _to_jsonable(asdict(self))


@dataclass
class FrameResult:
    sample: GraspNetSample
    target_region: Optional[TargetRegion]
    top_k: list[ScoredGrasp]
    best_grasp: Optional[ScoredGrasp]
    runtime: dict
    status: str
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return _to_jsonable(asdict(self))
