"""Thin adapter around the pinned official GQ-CNN image sampler.

This module deliberately keeps TensorFlow scoring separate from geometry-only
candidate generation.  The official sampler consumes full-scene metric depth
and a target segmask; it returns planar ``Grasp2D`` objects.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_GQCNN_ROOT = REPO_ROOT / "third_party" / "gqcnn-official"
OFFICIAL_GQCNN_COMMIT = "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21"
OFFICIAL_GQCNN_RELEASE = "v1.3.0"


def ensure_official_gqcnn_path(root: Path | None = None) -> Path:
    """Place the pinned official checkout on ``sys.path`` without installing TF."""
    root = Path(root or OFFICIAL_GQCNN_ROOT).resolve()
    if not (root / "gqcnn" / "grasping" / "image_grasp_sampler.py").is_file():
        raise FileNotFoundError(f"Pinned official GQ-CNN checkout missing: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def gqcnn_runtime() -> dict[str, Any]:
    ensure_official_gqcnn_path()
    import gqcnn
    from gqcnn.grasping import AntipodalDepthImageGraspSampler

    return {
        "release": OFFICIAL_GQCNN_RELEASE,
        "commit": OFFICIAL_GQCNN_COMMIT,
        "version": gqcnn.__version__,
        "scoring_import_available": bool(gqcnn.SCORING_IMPORT_AVAILABLE),
        "sampler_class": AntipodalDepthImageGraspSampler.__name__,
        "checkout": str(OFFICIAL_GQCNN_ROOT),
    }


def make_camera_intrinsics(
    values: Mapping[str, Any],
    *,
    frame: str,
):
    """Construct official ``autolab_core.CameraIntrinsics`` from verified values."""
    ensure_official_gqcnn_path()
    from autolab_core import CameraIntrinsics

    required = ("fx", "fy", "cx", "cy", "width", "height")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing camera intrinsics fields: {missing}")
    floats = {key: float(values[key]) for key in ("fx", "fy", "cx", "cy")}
    if not all(np.isfinite(value) for value in floats.values()):
        raise ValueError("Camera intrinsics must be finite")
    if floats["fx"] <= 0 or floats["fy"] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    width = int(values["width"])
    height = int(values["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Camera image dimensions must be positive")
    return CameraIntrinsics(
        frame,
        fx=floats["fx"],
        fy=floats["fy"],
        cx=floats["cx"],
        cy=floats["cy"],
        skew=float(values.get("skew", 0.0)),
        height=height,
        width=width,
    )


def make_rgbd_and_segmask(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    *,
    frame: str,
):
    """Create official full-scene RGB-D and target ``BinaryImage`` objects."""
    ensure_official_gqcnn_path()
    from autolab_core import BinaryImage, ColorImage, DepthImage, RgbdImage

    rgb = np.asarray(rgb)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(target_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"RGB must be HxWx3 uint8, got {rgb.shape} {rgb.dtype}")
    if depth_m.shape != rgb.shape[:2] or mask.shape != depth_m.shape:
        raise ValueError(
            f"RGB/depth/mask dimensions differ: {rgb.shape}, {depth_m.shape}, {mask.shape}"
        )
    if not np.all(np.isfinite(depth_m)) or np.any(depth_m < 0):
        raise ValueError("Depth must contain finite, non-negative metric values")
    color_im = ColorImage(rgb, frame=frame)
    depth_im = DepthImage(depth_m, frame=frame)
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    segmask = BinaryImage(mask.astype(np.uint8) * 255, frame=frame)
    return rgbd_im, segmask


def export_intrinsics_file(camera_intrinsics: Any, path: Path) -> Path:
    """Save an official autolab-core ``.intr`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    camera_intrinsics.save(str(path))
    return path


def sampling_config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_antipodal_grasps(
    rgbd_image: Any,
    camera_intrinsics: Any,
    target_segmask: Any,
    sampling_config: Mapping[str, Any],
    *,
    num_samples: int,
    seed: int,
    visualize: bool = False,
) -> list[Any]:
    """Run the official v1.3.0 antipodal sampler with its actual public API."""
    ensure_official_gqcnn_path()
    from gqcnn.grasping import AntipodalDepthImageGraspSampler

    config = dict(sampling_config)
    required = (
        "gripper_width", "friction_coef", "depth_grad_thresh",
        "depth_grad_gaussian_sigma", "downsample_rate",
        "max_rejection_samples", "max_dist_from_center",
        "min_dist_from_boundary", "min_grasp_dist", "angle_dist_weight",
        "depth_sampling_mode", "depth_samples_per_grasp",
        "depth_sample_win_height", "depth_sample_win_width",
        "min_depth_offset", "max_depth_offset",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing official sampler configuration: {missing}")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    # v1.3.0 accepts a gripper-width positional argument but ignores it and
    # reads config['gripper_width']; keeping it in the mapping is mandatory.
    sampler = AntipodalDepthImageGraspSampler(config)
    return list(
        sampler.sample(
            rgbd_image,
            camera_intrinsics,
            int(num_samples),
            segmask=target_segmask,
            seed=int(seed),
            visualize=bool(visualize),
        )
    )
