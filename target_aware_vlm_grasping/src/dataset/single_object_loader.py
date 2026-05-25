from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from dataset.camera_loader import intrinsics_from_dict, load_depth, load_rgb
from target.mask_utils import clean_binary_mask, compute_mask_center, mask_to_bbox
from utils.data_types import DatasetSample, TargetRegion


DEFAULT_SINGLE_OBJECT_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 579.411,
    "fy": 579.411,
    "cx": 320.0,
    "cy": 240.0,
}


OBJECT_SPECS = {
    "001_chips_can": {
        "object_name": "chips can",
        "category_name": "can",
        "command": "pick the chips can",
    },
    "002_master_chef_can": {
        "object_name": "master chef can",
        "category_name": "can",
        "command": "pick the master chef can",
    },
    "003_cracker_box": {
        "object_name": "cracker box",
        "category_name": "box",
        "command": "pick the cracker box",
    },
}


class SingleObjectIndexBuilder:
    """Index turntable-style single-object RGB-D folders.

    Expected layout:

    data/
      001_chips_can/
        NP1_0.jpg
        NP1_0.h5              # contains dataset "depth"
        masks/NP1_0_mask.pbm  # optional binary object mask

    The processing unit is one view of one object. The language command is
    deterministic: "pick the {object_name}".
    """

    def __init__(self, dataset_root: Path, output_root: Path):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

    def build(
        self,
        objects: Iterable[str] | None = None,
        max_samples: int | None = None,
        samples_per_object: int | None = None,
    ) -> list[DatasetSample]:
        object_keys = list(objects) if objects else self._discover_object_keys()
        samples: list[DatasetSample] = []
        for object_key in object_keys:
            object_dir = self._resolve_object_dir(object_key)
            object_samples = self._build_for_object(object_dir, samples_per_object)
            samples.extend(object_samples)
            if max_samples is not None and len(samples) >= max_samples:
                return samples[:max_samples]
        return samples

    def _discover_object_keys(self) -> list[str]:
        keys = [key for key in OBJECT_SPECS if (self.dataset_root / key).is_dir()]
        if keys:
            return keys
        return [
            path.name
            for path in sorted(self.dataset_root.iterdir())
            if path.is_dir() and any(path.glob("*.jpg")) and any(path.glob("*.h5"))
        ]

    def _resolve_object_dir(self, object_key: str) -> Path:
        if object_key in OBJECT_SPECS:
            path = self.dataset_root / object_key
            if path.exists():
                return path
        matches = sorted(self.dataset_root.glob(f"{object_key}*"))
        matches = [path for path in matches if path.is_dir()]
        if matches:
            return matches[0]
        path = self.dataset_root / object_key
        raise FileNotFoundError(f"Missing single-object folder: {path}")

    def _build_for_object(self, object_dir: Path, samples_per_object: int | None) -> list[DatasetSample]:
        spec = OBJECT_SPECS.get(object_dir.name, _spec_from_dir_name(object_dir.name))
        samples = []
        stems = sorted(path.stem for path in object_dir.glob("*.jpg"))
        if samples_per_object is not None:
            stems = stems[:samples_per_object]
        for stem in stems:
            rgb_path = object_dir / f"{stem}.jpg"
            depth_path = object_dir / f"{stem}.h5"
            if not depth_path.exists():
                continue
            mask_path = object_dir / "masks" / f"{stem}_mask.pbm"
            bbox = None
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]
                    # Pipeline RGB is resized to the 640x480 depth frame.
                    mask = _binary_object_mask(mask)
                    mask = cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
                    bbox = mask_to_bbox(mask)
            sample_id = f"{object_dir.name}__{stem}"
            samples.append(DatasetSample(
                dataset_name="SingleObjectRGBD",
                sample_id=sample_id,
                rgb_path=rgb_path,
                depth_path=depth_path,
                sentence=spec["command"],
                target_label=spec["object_name"],
                split=object_dir.name,
                image_id=sample_id,
                scene_id=object_dir.name,
                camera="turntable",
                frame_id=stem,
                command=spec["command"],
                target_id=object_dir.name,
                target_index=None,
                target_bbox=bbox,
                target_bbox_gt=bbox,
                target_mask_path=mask_path if mask_path.exists() else None,
                grasp_rectangles=[],
                grasp_annotations=[],
                output_dir=self.output_root / "single_object" / object_dir.name / stem,
                label_path=mask_path if mask_path.exists() else None,
                camera_intrinsics=DEFAULT_SINGLE_OBJECT_INTRINSICS,
                metadata={
                    "dataset": "SingleObjectRGBD",
                    "object_key": object_dir.name,
                    "object_name": spec["object_name"],
                    "category_name": spec["category_name"],
                    "depth_scale": 10000.0,
                    "rgb_resized_to_depth": True,
                    "source_layout": "single_object_h5_depth_pbm_mask",
                },
            ))
        return samples


class SingleObjectRGBDLoader:
    def __init__(
        self,
        depth_scale: float = 10000.0,
        fallback_intrinsics: dict | None = None,
    ):
        self.depth_scale = depth_scale
        self.fallback_intrinsics = fallback_intrinsics or DEFAULT_SINGLE_OBJECT_INTRINSICS

    def load_sample(self, sample: DatasetSample) -> dict:
        depth_scale = float(sample.metadata.get("depth_scale", self.depth_scale))
        rgb = load_rgb(sample.rgb_path)
        depth = load_depth(sample.depth_path, depth_scale)
        rgb = _resize_rgb_to_depth(rgb, depth.shape)
        mask = self._load_mask(sample, depth.shape)
        bbox = mask_to_bbox(mask) if mask is not None else sample.target_bbox
        if bbox is not None:
            sample.target_bbox = bbox
            sample.target_bbox_gt = bbox
        intrinsics = intrinsics_from_dict(sample.camera_intrinsics or self.fallback_intrinsics)
        target = TargetRegion(
            target_id=sample.target_id,
            label=sample.target_label,
            bbox=bbox,
            mask=mask,
            grounding_score=1.0,
            center_2d=compute_mask_center(mask) if mask is not None else None,
            command=sample.command,
            target_source="oracle",
            metadata={
                "dataset": "SingleObjectRGBD",
                "object_key": sample.metadata.get("object_key"),
                "object_name": sample.metadata.get("object_name"),
                "category_name": sample.metadata.get("category_name"),
                "target_mask_source": "dataset" if mask is not None else "missing",
            },
        )
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "target": target,
            "grasp_rectangles": [],
        }

    def _load_mask(self, sample: DatasetSample, depth_shape: tuple[int, int]) -> np.ndarray | None:
        path = sample.target_mask_path
        if not path or not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = _binary_object_mask(mask)
        if mask.shape != depth_shape:
            mask = cv2.resize(mask.astype(np.uint8), (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        return clean_binary_mask(mask, kernel_size=3)


def _resize_rgb_to_depth(rgb: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    h, w = depth_shape
    if rgb.shape[:2] == (h, w):
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def _binary_object_mask(mask: np.ndarray) -> np.ndarray:
    """Return the likely object foreground for binary masks.

    The single-object PBM files in this dataset encode the object as black and
    the background as white. Choosing the minority side is a conservative way to
    support that layout while still accepting conventional white-foreground
    masks.
    """

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    positive = mask > 0
    negative = ~positive
    return negative if negative.mean() < positive.mean() else positive


def _spec_from_dir_name(name: str) -> dict:
    clean = name
    if "_" in clean and clean.split("_", 1)[0].isdigit():
        clean = clean.split("_", 1)[1]
    object_name = clean.replace("_", " ")
    return {
        "object_name": object_name,
        "category_name": object_name.split()[-1] if object_name else "object",
        "command": f"pick the {object_name}",
    }
