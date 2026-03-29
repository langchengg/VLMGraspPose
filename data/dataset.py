"""
data/dataset.py — Scene loader and sample generator
=====================================================
Sample granularity: (scene_id, view_id, object_instance_id)

A single scene contains multiple views and multiple objects.
For target-directed grasping we iterate over every
(scene, view, target_object) triple.
"""

import json
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class ObjectInfo:
    """Metadata for a single object instance in a scene."""
    obj_id: int                  # 0-indexed class id used in label mask
    obj_name: str                # raw name, e.g. "037_scissors"
    friendly_name: str           # human name, e.g. "scissors"
    pos_in_world: List[float]    # [x, y, z]
    ori_in_world: List[float]    # [qx, qy, qz, qw]


@dataclass
class SceneMeta:
    """Everything we know about one scene."""
    scene_id: str
    scene_dir: Path
    camera_type: str
    num_views: int
    intrinsics: np.ndarray       # 3×3
    camera_poses: np.ndarray     # (num_views, 4, 4)
    factor_depth: float
    objects: List[ObjectInfo]


@dataclass
class Sample:
    """One training / evaluation sample."""
    sample_id: str
    scene_id: str
    view_id: int
    image_path: str
    depth_path: str
    label_path: str
    intrinsics_path: str
    extrinsics_index: int        # index into camera_poses.npy
    target_class: str            # friendly name
    target_obj_id: int           # label mask value = cls_indexes entry
    text_query: str
    gt_bbox: Optional[List[int]] = None       # [x1,y1,x2,y2]
    gt_mask_info: Optional[Dict] = None       # {label_path, instance_id}
    frame: str = config.COORD_FRAME


# ── Scene Loader ─────────────────────────────────────────────────────

def load_scene(scene_dir: Path, camera: str = config.CAMERA_TYPE) -> SceneMeta:
    """Load all metadata for a single scene."""
    cam_dir = scene_dir / camera

    # Intrinsics
    intrinsics = np.load(str(cam_dir / "camK.npy"))

    # Extrinsics (256 camera poses)
    camera_poses = np.load(str(cam_dir / "camera_poses.npy"))

    # Factor depth from first meta file
    first_meta = sio.loadmat(str(cam_dir / "meta" / "0000.mat"))
    factor_depth = float(first_meta["factor_depth"].squeeze())

    # Objects from first annotation file
    ann_file = cam_dir / "annotations" / "0000.xml"
    objects = _parse_annotation(ann_file)

    # Number of views = number of RGB files
    rgb_dir = cam_dir / "rgb"
    num_views = len(list(rgb_dir.glob("*.png")))

    return SceneMeta(
        scene_id=scene_dir.name,
        scene_dir=scene_dir,
        camera_type=camera,
        num_views=num_views,
        intrinsics=intrinsics,
        camera_poses=camera_poses,
        factor_depth=factor_depth,
        objects=objects,
    )


def _parse_annotation(xml_path: Path) -> List[ObjectInfo]:
    """Parse annotation XML into ObjectInfo list."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    objects = []
    for obj_elem in root.findall("obj"):
        obj_id = int(obj_elem.find("obj_id").text)
        raw_name = obj_elem.find("obj_name").text.replace(".ply", "")
        friendly = config.OBJECT_NAME_MAP.get(raw_name, raw_name)
        pos = [float(x) for x in obj_elem.find("pos_in_world").text.split()]
        ori = [float(x) for x in obj_elem.find("ori_in_world").text.split()]
        objects.append(ObjectInfo(obj_id, raw_name, friendly, pos, ori))
    return objects


# ── Image / Depth Loaders ────────────────────────────────────────────

def load_rgb(scene_dir: Path, view_id: int,
             camera: str = config.CAMERA_TYPE) -> np.ndarray:
    """Load RGB image as HxWx3 uint8."""
    path = scene_dir / camera / "rgb" / f"{view_id:04d}.png"
    return np.array(Image.open(str(path)))


def load_depth(scene_dir: Path, view_id: int,
               camera: str = config.CAMERA_TYPE,
               factor: float = 1000.0) -> np.ndarray:
    """Load depth image as HxW float32 in metres."""
    path = scene_dir / camera / "depth" / f"{view_id:04d}.png"
    raw = np.array(Image.open(str(path))).astype(np.float32)
    return raw / factor


def load_label(scene_dir: Path, view_id: int,
               camera: str = config.CAMERA_TYPE) -> np.ndarray:
    """Load instance label mask as HxW uint8."""
    path = scene_dir / camera / "label" / f"{view_id:04d}.png"
    return np.array(Image.open(str(path)))


def load_meta(scene_dir: Path, view_id: int,
              camera: str = config.CAMERA_TYPE) -> dict:
    """Load per-view meta .mat → dict with poses, cls_indexes, etc."""
    path = scene_dir / camera / "meta" / f"{view_id:04d}.mat"
    return sio.loadmat(str(path))


# ── GT Bounding Box from Label Mask ──────────────────────────────────

def bbox_from_mask(label: np.ndarray, instance_id: int) -> Optional[List[int]]:
    """Compute tight [x1,y1,x2,y2] bbox for *instance_id* in label mask.

    The label mask stores cls_indexes values (1-indexed in the mask,
    matching meta['cls_indexes']).  instance_id here is the actual
    pixel value in the label image.
    """
    ys, xs = np.where(label == instance_id)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ── Sample Generator ─────────────────────────────────────────────────

def generate_samples(
    scene_meta: SceneMeta,
    view_stride: int = config.VIEW_STRIDE,
    seed: int = 42,
) -> List[Sample]:
    """Generate samples for every (view, object) pair in a scene.

    Uses *view_stride* to subsample views (e.g. every 16th frame).
    """
    rng = random.Random(seed)
    samples: List[Sample] = []

    cam_dir = scene_meta.scene_dir / scene_meta.camera_type

    for view_id in range(0, scene_meta.num_views, view_stride):
        # Load label mask for this view
        label = load_label(scene_meta.scene_dir, view_id,
                           scene_meta.camera_type)

        # Load meta to get cls_indexes (these are the label-mask values)
        meta = load_meta(scene_meta.scene_dir, view_id,
                         scene_meta.camera_type)
        cls_indexes = meta["cls_indexes"].squeeze().tolist()
        if isinstance(cls_indexes, int):
            cls_indexes = [cls_indexes]

        for obj in scene_meta.objects:
            # The label mask uses cls_indexes values.
            # obj.obj_id is the 0-based object id from object_id_list.txt.
            # cls_indexes are 1-based: cls_indexes = obj_id + 1
            mask_val = obj.obj_id + 1
            if mask_val not in cls_indexes:
                continue

            # Check object is actually visible in this view
            gt_bbox = bbox_from_mask(label, mask_val)
            if gt_bbox is None:
                continue

            # Minimum object size filter (at least 20×20 pixels)
            bw = gt_bbox[2] - gt_bbox[0]
            bh = gt_bbox[3] - gt_bbox[1]
            if bw < 20 or bh < 20:
                continue

            # Generate text query from template
            template = rng.choice(config.TEXT_TEMPLATES)
            text_query = template.format(obj=obj.friendly_name)

            sample_id = f"{scene_meta.scene_id}_{view_id:04d}_{obj.obj_name}"

            samples.append(Sample(
                sample_id=sample_id,
                scene_id=scene_meta.scene_id,
                view_id=view_id,
                image_path=str(cam_dir / "rgb" / f"{view_id:04d}.png"),
                depth_path=str(cam_dir / "depth" / f"{view_id:04d}.png"),
                label_path=str(cam_dir / "label" / f"{view_id:04d}.png"),
                intrinsics_path=str(cam_dir / "camK.npy"),
                extrinsics_index=view_id,
                target_class=obj.friendly_name,
                target_obj_id=obj.obj_id,
                text_query=text_query,
                gt_bbox=gt_bbox,
                gt_mask_info={
                    "label_path": str(cam_dir / "label" / f"{view_id:04d}.png"),
                    "instance_id": int(mask_val),
                },
            ))

    return samples


# ── Convenience ──────────────────────────────────────────────────────

def discover_scenes(data_dir: Path) -> List[Path]:
    """Return sorted list of scene directories under *data_dir*."""
    if not data_dir.exists():
        return []
    scenes = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("scene_")
    ])
    return scenes
