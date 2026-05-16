from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from target.mask_utils import compute_mask_center, mask_to_bbox
from utils.data_types import GraspNetSample
from utils.io_utils import ensure_dir, save_json


@dataclass
class ObjectLanguageEntry:
    split: str
    scene_id: str
    camera: str
    frame_id: str
    target_id: int
    target_label: str
    command: str
    mask_path: str
    bbox: list[int]
    disambiguation_type: str
    visibility_score: Optional[float] = None
    metadata: dict | None = None

    def to_json(self) -> dict:
        data = asdict(self)
        data["metadata"] = self.metadata or {}
        return data


class ObjectLanguageMapper:
    """Build explicit language-target rows for each visible object instance."""

    def __init__(
        self,
        output_root: Path,
        category_labels: dict[int, str] | None = None,
        command_mode: str = "auto",
        save_masks: bool = True,
    ):
        self.output_root = Path(output_root)
        self.category_labels = category_labels or {}
        self.command_mode = command_mode
        self.save_masks = save_masks

    def entries_for_sample(
        self,
        sample: GraspNetSample,
        label: np.ndarray | None,
        rgb: np.ndarray | None = None,
        all_targets: bool = True,
        target_ids: list[int] | None = None,
    ) -> list[ObjectLanguageEntry]:
        if label is None:
            raise ValueError("Object-language mapping requires a label image.")

        instances = self._visible_instances(label)
        if target_ids is not None:
            wanted = set(target_ids)
            instances = [inst for inst in instances if inst["target_id"] in wanted]
        if not instances:
            return []
        if not all_targets:
            instances = [max(instances, key=lambda inst: inst["pixel_count"])]

        labels = {inst["target_id"]: self._target_label(inst["target_id"]) for inst in instances}
        label_counts: dict[str, int] = {}
        for label_text in labels.values():
            label_counts[label_text] = label_counts.get(label_text, 0) + 1

        total_visible_pixels = float(sum(inst["pixel_count"] for inst in instances)) or 1.0
        entries = []
        for inst in instances:
            target_id = inst["target_id"]
            target_label = labels[target_id]
            duplicate_label = label_counts[target_label] > 1
            same_label_instances = [item for item in instances if labels[item["target_id"]] == target_label]
            descriptor = self._position_descriptor(inst, same_label_instances, duplicate_label)
            color = self._dominant_color(rgb, inst["mask"]) if rgb is not None else None
            command, disambiguation_type = self._build_command(
                inst=inst,
                target_label=target_label,
                duplicate_label=duplicate_label,
                descriptor=descriptor,
                color=color,
                instances=instances,
            )
            mask_path = self._mask_path(sample, target_id)
            if self.save_masks:
                ensure_dir(Path(mask_path).parent)
                cv2.imwrite(mask_path, inst["mask"].astype(np.uint8) * 255)
            entries.append(ObjectLanguageEntry(
                split=sample.split,
                scene_id=sample.scene_id,
                camera=sample.camera,
                frame_id=sample.frame_id,
                target_id=target_id,
                target_label=target_label,
                command=command,
                mask_path=mask_path,
                bbox=inst["bbox"],
                disambiguation_type=disambiguation_type,
                visibility_score=inst["pixel_count"] / total_visible_pixels,
                metadata={
                    "mask_val": inst["mask_val"],
                    "pixel_count": inst["pixel_count"],
                    "center_2d": list(inst["center"]),
                    "alternative_commands": self._alternative_commands(target_id, target_label, descriptor, color),
                },
            ))
        return entries

    def build_for_samples(
        self,
        samples: list[GraspNetSample],
        labels: dict[tuple[str, str, str, str], np.ndarray],
        all_targets: bool = True,
    ) -> list[ObjectLanguageEntry]:
        entries: list[ObjectLanguageEntry] = []
        for sample in samples:
            key = (sample.split, sample.scene_id, sample.camera, sample.frame_id)
            entries.extend(self.entries_for_sample(sample, labels.get(key), all_targets=all_targets))
        return entries

    def save_mapping(self, entries: list[ObjectLanguageEntry]) -> dict[str, Path]:
        mapping_dir = self.output_root / "mappings"
        ensure_dir(mapping_dir)
        rows = [entry.to_json() for entry in entries]
        csv_path = mapping_dir / "object_language_mapping.csv"
        json_path = mapping_dir / "object_language_mapping.json"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        save_json(json_path, rows)
        return {"csv": csv_path, "json": json_path}

    def _visible_instances(self, label: np.ndarray) -> list[dict]:
        instances = []
        for mask_val in sorted(int(value) for value in np.unique(label) if int(value) > 0):
            mask = label == mask_val
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            center = compute_mask_center(mask)
            instances.append({
                "target_id": mask_val - 1,
                "mask_val": mask_val,
                "mask": mask,
                "bbox": bbox,
                "center": center,
                "pixel_count": int(mask.sum()),
            })
        return instances

    def _target_label(self, target_id: int) -> str:
        return self.category_labels.get(target_id, f"object_{target_id:03d}")

    def _build_command(
        self,
        inst: dict,
        target_label: str,
        duplicate_label: bool,
        descriptor: str | None,
        color: str | None,
        instances: list[dict],
    ) -> tuple[str, str]:
        has_category_label = not target_label.startswith("object_")
        if self.command_mode == "attribute":
            attribute = self._attribute_descriptor(inst, instances, color)
            if attribute:
                return f"pick the {attribute} object", "attribute"

        if has_category_label and duplicate_label and descriptor:
            return f"pick the {descriptor} {target_label}", "position"
        if has_category_label and color and duplicate_label:
            return f"pick the {color} {target_label}", "attribute"
        if has_category_label:
            return f"pick the {target_label}", "category"
        return f"pick object_{inst['target_id']:03d}", "object_id"

    def _position_descriptor(self, inst: dict, instances: list[dict], duplicate_label: bool) -> str | None:
        if not duplicate_label:
            return None
        ordered = sorted(instances, key=lambda item: item["center"][0])
        rank = next(idx for idx, item in enumerate(ordered) if item is inst)
        if len(ordered) == 2:
            return "left" if rank == 0 else "right"
        if rank == 0:
            return "left"
        if rank == len(ordered) - 1:
            return "right"
        return "center"

    def _attribute_descriptor(self, inst: dict, instances: list[dict], color: str | None) -> str | None:
        if color:
            return color
        largest = max(instances, key=lambda item: item["pixel_count"])
        if largest is inst:
            return "largest"
        ordered = sorted(instances, key=lambda item: item["center"][0])
        if ordered[0] is inst:
            return "leftmost"
        if ordered[-1] is inst:
            return "rightmost"
        return None

    def _dominant_color(self, rgb: np.ndarray, mask: np.ndarray) -> str | None:
        pixels = rgb[mask.astype(bool)]
        if len(pixels) == 0:
            return None
        mean = pixels.mean(axis=0)
        colors = {
            "red": np.array([180, 70, 70]),
            "green": np.array([70, 150, 70]),
            "blue": np.array([70, 90, 180]),
            "yellow": np.array([190, 170, 70]),
            "white": np.array([220, 220, 220]),
            "black": np.array([40, 40, 40]),
        }
        name, color_vec = min(colors.items(), key=lambda item: float(np.linalg.norm(mean - item[1])))
        distance = float(np.linalg.norm(mean - color_vec))
        return name if distance < 110.0 else None

    def _alternative_commands(
        self,
        target_id: int,
        target_label: str,
        descriptor: str | None,
        color: str | None,
    ) -> list[str]:
        commands = [
            f"pick object_{target_id:03d}",
            f"pick the object with id {target_id:03d}",
        ]
        if descriptor and not target_label.startswith("object_"):
            commands.append(f"pick the {descriptor} {target_label}")
        if color:
            commands.append(f"pick the {color} object")
        return commands

    def _mask_path(self, sample: GraspNetSample, target_id: int) -> str:
        return str(
            self.output_root
            / "mappings"
            / "masks"
            / sample.split
            / sample.scene_id
            / sample.camera
            / sample.frame_id
            / f"target_{target_id:03d}.png"
        )
