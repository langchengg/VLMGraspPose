"""
src/data_utils.py — Scene loading and low-level data accessors
================================================================
Migrated from data/dataset.py.

Provides:
  • Scene discovery and metadata loading
  • Per-view image / depth / label / meta loaders
  • object_id_list.txt parser
  • bbox_from_mask helper
  • Grasp label and collision label loaders

Sample generation logic is NOT here — it lives in the step scripts.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ═════════════════════════════════════════════════════════════════════
#  Scene discovery
# ═════════════════════════════════════════════════════════════════════

def discover_scenes(
    scenes_dir: Path = config.SCENES_DIR,
) -> List[Path]:
    """Return sorted list of scene directories."""
    if not scenes_dir.exists():
        return []
    return sorted(
        d for d in scenes_dir.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    )


def scene_id_from_path(scene_path: Path) -> int:
    """Extract integer scene ID from scene directory name."""
    return int(scene_path.name.split("_")[1])


def split_for_scene(scene_id: int) -> str:
    """Return split name for a given scene ID."""
    for split, (lo, hi) in config.SPLIT_SCENE_RANGES.items():
        if lo <= scene_id < hi:
            return split
    raise ValueError(f"Scene {scene_id} does not belong to any split")


# ═════════════════════════════════════════════════════════════════════
#  Object list
# ═════════════════════════════════════════════════════════════════════

def load_object_id_list(scene_dir: Path) -> List[int]:
    """Load object_id_list.txt → list of int object IDs.

    GraspNet stores one integer per line in ``scene_xxxx/object_id_list.txt``.
    """
    path = scene_dir / "object_id_list.txt"
    if not path.exists():
        return []
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


# ═════════════════════════════════════════════════════════════════════
#  Camera intrinsics & poses
# ═════════════════════════════════════════════════════════════════════

def load_camera_intrinsics(
    scene_dir: Path,
    camera: str = config.CAMERA_TYPE,
) -> np.ndarray:
    """Load 3×3 camera intrinsic matrix."""
    path = scene_dir / camera / "camK.npy"
    return np.load(str(path))


def load_camera_poses(
    scene_dir: Path,
    camera: str = config.CAMERA_TYPE,
) -> np.ndarray:
    """Load (N, 4, 4) camera-to-world poses."""
    path = scene_dir / camera / "camera_poses.npy"
    return np.load(str(path))


# ═════════════════════════════════════════════════════════════════════
#  Per-view loaders
# ═════════════════════════════════════════════════════════════════════

def load_rgb(
    scene_dir: Path,
    frame_id: int,
    camera: str = config.CAMERA_TYPE,
) -> np.ndarray:
    """Load RGB image as (H, W, 3) uint8."""
    path = scene_dir / camera / "rgb" / f"{frame_id:04d}.png"
    return np.array(Image.open(str(path)))


def load_depth(
    scene_dir: Path,
    frame_id: int,
    camera: str = config.CAMERA_TYPE,
    factor: float = 1000.0,
) -> np.ndarray:
    """Load depth image as (H, W) float32 in metres."""
    path = scene_dir / camera / "depth" / f"{frame_id:04d}.png"
    raw = np.array(Image.open(str(path))).astype(np.float32)
    return raw / factor


def load_label(
    scene_dir: Path,
    frame_id: int,
    camera: str = config.CAMERA_TYPE,
) -> np.ndarray:
    """Load instance segmentation mask as (H, W) uint16/uint8."""
    path = scene_dir / camera / "label" / f"{frame_id:04d}.png"
    return np.array(Image.open(str(path)))


def load_meta(
    scene_dir: Path,
    frame_id: int,
    camera: str = config.CAMERA_TYPE,
) -> dict:
    """Load per-view .mat metadata → dict with poses, cls_indexes, etc."""
    path = scene_dir / camera / "meta" / f"{frame_id:04d}.mat"
    return sio.loadmat(str(path))


def get_factor_depth(
    scene_dir: Path,
    camera: str = config.CAMERA_TYPE,
) -> float:
    """Read depth scale factor from first meta file."""
    meta = load_meta(scene_dir, 0, camera)
    return float(meta["factor_depth"].squeeze())


# ═════════════════════════════════════════════════════════════════════
#  View enumeration
# ═════════════════════════════════════════════════════════════════════

def count_views(
    scene_dir: Path,
    camera: str = config.CAMERA_TYPE,
) -> int:
    """Count available RGB frames for a (scene, camera)."""
    rgb_dir = scene_dir / camera / "rgb"
    if not rgb_dir.exists():
        return 0
    return len(list(rgb_dir.glob("*.png")))


def list_frame_ids(
    scene_dir: Path,
    camera: str = config.CAMERA_TYPE,
    stride: int = 1,
) -> List[int]:
    """Return sorted frame IDs, optionally sub-sampled by stride."""
    n = count_views(scene_dir, camera)
    return list(range(0, n, stride))


# ═════════════════════════════════════════════════════════════════════
#  GT helpers
# ═════════════════════════════════════════════════════════════════════

def visible_object_ids(
    scene_dir: Path,
    frame_id: int,
    camera: str = config.CAMERA_TYPE,
) -> List[int]:
    """Return list of object IDs visible in a given view.

    Uses cls_indexes from the meta .mat file.  These are the 1-based
    label-mask values;  object_id = cls_index − 1.
    """
    meta = load_meta(scene_dir, frame_id, camera)
    cls_indexes = meta["cls_indexes"].squeeze().tolist()
    if isinstance(cls_indexes, int):
        cls_indexes = [cls_indexes]
    return [c - 1 for c in cls_indexes]


def bbox_from_mask(
    label: np.ndarray,
    mask_val: int,
) -> Optional[List[int]]:
    """Compute tight [x1, y1, x2, y2] bbox for pixels == mask_val.

    Returns None if the object is not visible.
    """
    ys, xs = np.where(label == mask_val)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_pixel_count(label: np.ndarray, mask_val: int) -> int:
    """Count visible pixels for an object in the label mask."""
    return int(np.sum(label == mask_val))


# ═════════════════════════════════════════════════════════════════════
#  Grasp label loaders
# ═════════════════════════════════════════════════════════════════════

def load_grasp_labels(
    obj_id: int,
    grasp_label_dir: Path = config.GRASP_LABEL_DIR,
) -> Optional[dict]:
    """Load GraspNet per-object grasp labels.

    Returns dict with keys like 'points', 'offsets', 'scores' as stored
    by the official dataset, or None if file not found.
    """
    path = grasp_label_dir / f"{obj_id:03d}_labels.npz"
    if not path.exists():
        return None
    return dict(np.load(str(path), allow_pickle=True))


def load_collision_labels(
    scene_id: int,
    camera: str = config.CAMERA_TYPE,
    frame_id: int = 0,
    collision_label_dir: Path = config.COLLISION_LABEL_DIR,
) -> Optional[np.ndarray]:
    """Load per-scene collision labels.

    Returns the collision array for a specific scene/camera/view,
    or None if file not found.
    """
    path = collision_label_dir / f"scene_{scene_id:04d}" / camera / f"{frame_id:04d}.npz"
    if not path.exists():
        # Try flat structure
        path = collision_label_dir / f"scene_{scene_id:04d}" / f"collision_labels.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    # The exact key depends on GraspNet version
    for key in ["collision_label", "collision_labels", "arr_0"]:
        if key in data:
            return data[key]
    return None


# ═════════════════════════════════════════════════════════════════════
#  Grasp candidate loading (shared by step07, step08, step10)
# ═════════════════════════════════════════════════════════════════════

def load_grasp_candidates(
    view_sample_id: str,
    detector: str = "antipodal",
    candidates_dir: Path = None,
) -> list:
    """Load cached grasp candidates for a view.

    Searches detector-specific subdirectory first (new layout),
    then falls back to flat layout (legacy).

    Returns a list of GraspCandidate objects (or empty list).
    """
    from src.grasp_detector import GraspCandidate

    if candidates_dir is None:
        candidates_dir = config.GRASP_CANDIDATES_DIR

    # New layout: derived/grasp_candidates/{detector}/{sample_id}.npz
    path = candidates_dir / detector / f"{view_sample_id}.npz"
    if not path.exists():
        # Legacy fallback: derived/grasp_candidates/{sample_id}.npz
        path = candidates_dir / f"{view_sample_id}.npz"
    if not path.exists():
        return []

    data = np.load(str(path), allow_pickle=True)
    candidates = []
    n = int(data.get("num_candidates", 0))
    for i in range(n):
        candidates.append(GraspCandidate(
            candidate_id=i,
            position=data["positions"][i].tolist(),
            rotation=data["rotations"][i].tolist(),
            width=float(data["widths"][i]),
            detector_score=float(data["detector_scores"][i]),
            source=str(data["sources"][i]),
        ))
    return candidates

