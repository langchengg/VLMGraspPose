from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from dataset.camera_loader import intrinsics_from_dict, load_depth, load_rgb
from target.mask_utils import bbox_to_mask, clean_binary_mask, compute_mask_center
from utils.data_types import DatasetSample, TargetRegion


DEFAULT_OCID_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 579.411,
    "fy": 579.411,
    "cx": 320.0,
    "cy": 240.0,
}


class OCIDVLGIndexBuilder:
    def __init__(self, dataset_root: Path, output_root: Path):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

    def build(
        self,
        refer_split: str = "multiple",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> list[DatasetSample]:
        expressions_path = self.dataset_root / "refer" / refer_split / f"{split}_expressions.json"
        if not expressions_path.exists():
            raise FileNotFoundError(f"Missing OCID-VLG expressions file: {expressions_path}")
        with open(expressions_path) as f:
            payload = json.load(f)
        rows = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        samples = [self._sample_from_row(row, refer_split, split, idx) for idx, row in enumerate(rows)]
        if max_samples is not None:
            samples = samples[:max_samples]
        return samples

    def _sample_from_row(self, row: dict, refer_split: str, split: str, idx: int) -> DatasetSample:
        scene_rel, filename = row["image_filename"].split(",", 1)
        scene_dir = self.dataset_root / scene_rel
        stem = Path(filename).stem
        x, y, w, h = [int(round(v)) for v in row["box"]]
        bbox = [x, y, x + max(w - 1, 0), y + max(h - 1, 0)]
        image_id = f"{scene_rel.replace('/', '__')}__{stem}__{idx:06d}"
        output_dir = self.output_root / "ocid_vlg" / refer_split / split / image_id
        mask_path = scene_dir / "seg_mask_instances_combi" / filename
        target_index = int(row.get("answer", row.get("target_index", 0)))
        sentence = str(row.get("question") or row.get("sentence") or "").strip()
        target_label = str(row.get("target") or row.get("target_label") or "target")
        grasp_rectangles = _parse_grasp_rectangles(row.get("grasps", []))
        return DatasetSample(
            dataset_name="OCID-VLG",
            sample_id=image_id,
            rgb_path=scene_dir / "rgb" / filename,
            depth_path=scene_dir / "depth" / filename,
            sentence=sentence,
            target_label=target_label,
            split=split,
            image_id=image_id,
            scene_id=scene_rel,
            camera="ocid",
            frame_id=stem,
            command=sentence,
            target_id=target_index,
            target_index=target_index,
            target_bbox=bbox,
            target_bbox_gt=bbox,
            target_mask_path=mask_path if mask_path.exists() else None,
            grasp_rectangles=grasp_rectangles,
            grasp_annotations=grasp_rectangles,
            output_dir=output_dir,
            label_path=mask_path if mask_path.exists() else None,
            metadata={
                "dataset": "OCID-VLG",
                "refer_split": refer_split,
                "image_filename": row["image_filename"],
                "raw_box_xywh": row.get("box"),
                "template": row.get("template"),
                "concept_map": row.get("concept_map", {}),
            },
        )


class OCIDGraspIndexBuilder:
    """Fallback indexer for OCID-Grasp frames without referring expressions."""

    def __init__(self, dataset_root: Path, output_root: Path):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.catalog = self._load_catalog()

    def build(self, max_samples: Optional[int] = None) -> list[DatasetSample]:
        samples = self._build_from_box_files(max_samples)
        if samples:
            return samples
        return self._build_from_instance_masks(max_samples)

    def _build_from_box_files(self, max_samples: Optional[int] = None) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for boxes_path in sorted(self.dataset_root.glob("ARID*/**/Boxes_per_instance/*.txt")):
            scene_dir = boxes_path.parent.parent
            stem = boxes_path.stem
            rgb_path = scene_dir / "rgb" / f"{stem}.png"
            depth_path = scene_dir / "depth" / f"{stem}.png"
            mask_path = scene_dir / "seg_mask_instances_combi" / f"{stem}.png"
            if not rgb_path.exists() or not depth_path.exists():
                continue
            rows = self._read_boxes(boxes_path)
            commands = self._commands_for_rows(rows)
            for row in rows:
                inst_idx = int(row["instance_index"])
                subclass_id = int(row["subclass_id"])
                bbox = row["bbox"]
                grasp_path = scene_dir / "Grasps_per_instance" / stem / f"{inst_idx}_{subclass_id}.txt"
                grasps = _parse_grasp_rectangle_txt(grasp_path)
                if not grasps:
                    continue
                scene_rel = scene_dir.relative_to(self.dataset_root).as_posix()
                image_id = f"{scene_rel.replace('/', '__')}__{stem}__inst_{inst_idx:03d}"
                command = commands[inst_idx]
                target_label = self._target_label(subclass_id)
                samples.append(DatasetSample(
                    dataset_name="OCID-Grasp",
                    sample_id=image_id,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    sentence=command,
                    target_label=target_label,
                    split="ocid_grasp",
                    image_id=image_id,
                    scene_id=scene_rel,
                    camera="ocid",
                    frame_id=stem,
                    command=command,
                    target_id=inst_idx,
                    target_index=inst_idx,
                    target_bbox=bbox,
                    target_bbox_gt=bbox,
                    target_mask_path=mask_path if mask_path.exists() else None,
                    grasp_rectangles=grasps,
                    grasp_annotations=grasps,
                    output_dir=self.output_root / "ocid_grasp" / image_id,
                    label_path=mask_path if mask_path.exists() else None,
                    metadata={
                        "dataset": "OCID-Grasp",
                        "subclass_id": subclass_id,
                        "class_name": self._class_name(subclass_id),
                    },
                ))
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
        return samples

    def _build_from_instance_masks(self, max_samples: Optional[int] = None) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for rgb_path in sorted(self.dataset_root.glob("ARID*/**/rgb/*.png")):
            scene_dir = rgb_path.parent.parent
            stem = rgb_path.stem
            depth_path = scene_dir / "depth" / rgb_path.name
            mask_path = scene_dir / "seg_mask_instances_combi" / rgb_path.name
            label_path = scene_dir / "label" / rgb_path.name
            if not depth_path.exists() or not mask_path.exists():
                continue
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_image is None:
                continue
            if mask_image.ndim == 3:
                mask_image = mask_image[:, :, 0]
            label_image = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED) if label_path.exists() else None
            if label_image is not None and label_image.ndim == 3:
                label_image = label_image[:, :, 0]
            rows = []
            for inst_idx in sorted(int(v) for v in np.unique(mask_image) if int(v) > 0):
                instance_mask = mask_image == inst_idx
                if not instance_mask.any():
                    continue
                ys, xs = np.where(instance_mask)
                subclass_id = self._class_id_from_label_image(label_image, instance_mask)
                if subclass_id is None:
                    subclass_id = inst_idx
                rows.append({
                    "instance_index": inst_idx,
                    "subclass_id": int(subclass_id),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                })
            commands = self._commands_for_rows(rows)
            for row in rows:
                inst_idx = int(row["instance_index"])
                subclass_id = int(row["subclass_id"])
                class_grasp_path = scene_dir / "Annotations_per_class" / stem / str(subclass_id) / f"{stem}.txt"
                grasp_path = class_grasp_path if class_grasp_path.exists() else scene_dir / "Annotations" / f"{stem}.txt"
                grasps = _parse_grasp_rectangle_txt(grasp_path)
                if not grasps:
                    continue
                scene_rel = scene_dir.relative_to(self.dataset_root).as_posix()
                image_id = f"{scene_rel.replace('/', '__')}__{stem}__inst_{inst_idx:03d}"
                target_label = self._target_label(subclass_id)
                samples.append(DatasetSample(
                    dataset_name="OCID-Grasp",
                    sample_id=image_id,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    sentence=commands.get(inst_idx, f"pick the {target_label.replace('_', ' ')}"),
                    target_label=target_label,
                    split="ocid_grasp",
                    image_id=image_id,
                    scene_id=scene_rel,
                    camera="ocid",
                    frame_id=stem,
                    command=commands.get(inst_idx, f"pick the {target_label.replace('_', ' ')}"),
                    target_id=inst_idx,
                    target_index=inst_idx,
                    target_bbox=row["bbox"],
                    target_bbox_gt=row["bbox"],
                    target_mask_path=mask_path,
                    grasp_rectangles=grasps,
                    grasp_annotations=grasps,
                    output_dir=self.output_root / "ocid_grasp" / image_id,
                    label_path=mask_path,
                    metadata={
                        "dataset": "OCID-Grasp",
                        "subclass_id": subclass_id,
                        "class_name": self._class_name(subclass_id),
                        "source_layout": "instance_masks",
                    },
                ))
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
        return samples

    def _load_catalog(self) -> dict[int, dict]:
        path = self.dataset_root / "catalog.csv"
        if path.exists():
            table = pd.read_csv(path, sep="\t")
            return {int(row["ID"]): row.to_dict() for _, row in table.iterrows()}
        class_dict = self.dataset_root / "OCID_class_dict.py"
        if not class_dict.exists():
            return {}
        text = class_dict.read_text()
        rows = {}
        for name, value in re.findall(r"'([^']+)'\s*:\s*'(\d+)'", text):
            rows[int(value)] = {"ID": int(value), "class": name, "label": name}
        return rows

    def _class_id_from_label_image(self, label_image: np.ndarray | None, instance_mask: np.ndarray) -> int | None:
        if label_image is None:
            return None
        values = label_image[instance_mask]
        values = values[values > 0]
        if values.size == 0:
            return None
        labels, counts = np.unique(values.astype(int), return_counts=True)
        return int(labels[int(np.argmax(counts))])

    def _read_boxes(self, path: Path) -> list[dict]:
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            left, coords = line.split(";", 2)[0:2], line.split(";", 2)[2]
            instance_index = int(left[0])
            subclass_id = int(left[1])
            x1, y1, x2, y2 = [int(float(v)) for v in coords.split()]
            rows.append({
                "instance_index": instance_index,
                "subclass_id": subclass_id,
                "bbox": [x1, y1, x2, y2],
            })
        return rows

    def _commands_for_rows(self, rows: list[dict]) -> dict[int, str]:
        by_class: dict[str, list[dict]] = {}
        for row in rows:
            class_name = self._class_name(row["subclass_id"])
            by_class.setdefault(class_name, []).append(row)
        commands = {}
        for class_name, class_rows in by_class.items():
            ordered = sorted(class_rows, key=lambda item: (item["bbox"][0] + item["bbox"][2]) * 0.5)
            for idx, row in enumerate(ordered):
                phrase = class_name.replace("_", " ")
                if len(ordered) == 1:
                    command = f"pick the {phrase}"
                elif len(ordered) == 2:
                    command = f"pick the {'left' if idx == 0 else 'right'} {phrase}"
                elif idx == 0:
                    command = f"pick the left {phrase}"
                elif idx == len(ordered) - 1:
                    command = f"pick the right {phrase}"
                else:
                    command = f"pick the center {phrase}"
                commands[int(row["instance_index"])] = command
        return commands

    def _target_label(self, subclass_id: int) -> str:
        row = self.catalog.get(int(subclass_id), {})
        return str(row.get("label") or row.get("class") or f"object_{subclass_id:03d}")

    def _class_name(self, subclass_id: int) -> str:
        row = self.catalog.get(int(subclass_id), {})
        return str(row.get("class") or row.get("label") or f"object_{subclass_id:03d}")


class OCIDVLGLoader:
    def __init__(
        self,
        depth_scale: float = 1000.0,
        fallback_intrinsics: dict | None = None,
    ):
        self.depth_scale = depth_scale
        self.fallback_intrinsics = fallback_intrinsics or DEFAULT_OCID_INTRINSICS

    def load_sample(self, sample: DatasetSample) -> dict:
        rgb = load_rgb(sample.rgb_path)
        depth = load_depth(sample.depth_path, self.depth_scale)
        mask = self._load_target_mask(sample, rgb.shape[:2])
        dataset_name = sample.metadata.get("dataset", "OCID-VLG")
        target = TargetRegion(
            target_id=sample.target_index,
            label=sample.target_label,
            bbox=sample.target_bbox,
            mask=mask,
            grounding_score=1.0,
            center_2d=compute_mask_center(mask) if mask is not None else None,
            command=sample.command,
            target_source="oracle",
            metadata={
                "dataset": dataset_name,
                "image_id": sample.image_id,
                "sentence": sample.sentence,
                "grasp_rectangles": sample.grasp_rectangles,
            },
        )
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics_from_dict(self.fallback_intrinsics),
            "target": target,
            "grasp_rectangles": sample.grasp_rectangles,
        }

    def _load_target_mask(self, sample: DatasetSample, shape: tuple[int, int]) -> np.ndarray:
        if sample.target_mask_path and sample.target_mask_path.exists():
            mask_image = cv2.imread(str(sample.target_mask_path), cv2.IMREAD_UNCHANGED)
            if mask_image is not None:
                if mask_image.ndim == 3:
                    mask_image = mask_image[:, :, 0]
                if sample.target_index is not None:
                    mask = mask_image.astype(np.int32) == int(sample.target_index)
                    if mask.any():
                        return clean_binary_mask(mask, kernel_size=3)
                binary_mask = mask_image.astype(np.uint8) > 0
                if binary_mask.any():
                    return clean_binary_mask(binary_mask, kernel_size=3)
        return bbox_to_mask(sample.target_bbox, shape)


def _parse_grasp_rectangles(value) -> list[list[list[float]]]:
    rects: list[list[list[float]]] = []
    for rect in value or []:
        arr = np.asarray(rect, dtype=float)
        if arr.shape == (4, 2):
            rects.append(arr.tolist())
    return rects


def _parse_grasp_rectangle_txt(path: Path) -> list[list[list[float]]]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            values.append([float(parts[0]), float(parts[1])])
    rects = []
    for i in range(0, len(values) - 3, 4):
        rect = values[i:i + 4]
        if len(rect) == 4:
            rects.append(rect)
    return rects
