from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from target.mask_utils import mask_to_bbox
from utils.data_types import DatasetSample


@dataclass
class ObjectLanguageEntry:
    split: Optional[str]
    scene_id: str
    camera: str
    frame_id: str
    target_id: int | str | None
    target_label: str
    command: str
    mask_path: str | None
    bbox: list[int] | None
    disambiguation_type: str
    visibility_score: float | None = None

    def to_json(self) -> dict:
        return asdict(self)


class ObjectLanguageMapper:
    """Create explicit target-language entries without relying on object ids as names."""

    def entry_from_sample(self, sample: DatasetSample) -> ObjectLanguageEntry:
        bbox = sample.target_bbox_gt or sample.target_bbox
        mask_path = str(sample.target_mask_path) if sample.target_mask_path else None
        return ObjectLanguageEntry(
            split=sample.split,
            scene_id=sample.scene_id,
            camera=sample.camera,
            frame_id=sample.frame_id,
            target_id=sample.target_id,
            target_label=sample.target_label,
            command=sample.command or sample.sentence or self.command_from_label(sample.target_label),
            mask_path=mask_path,
            bbox=bbox,
            disambiguation_type="dataset_sentence" if sample.sentence else "category_label",
            visibility_score=self._bbox_area_score(bbox),
        )

    def entries_from_instances(
        self,
        sample: DatasetSample,
        instance_mask: np.ndarray,
        labels: dict[int, str] | None = None,
    ) -> list[ObjectLanguageEntry]:
        labels = labels or {}
        entries = []
        ids = [int(v) for v in np.unique(instance_mask) if int(v) > 0]
        centers = {}
        for target_id in ids:
            mask = instance_mask == target_id
            bbox = mask_to_bbox(mask)
            centers[target_id] = (bbox[0] + bbox[2]) * 0.5 if bbox else 0.0
        by_label: dict[str, list[int]] = {}
        for target_id in ids:
            label = labels.get(target_id, f"object_{target_id:03d}")
            by_label.setdefault(label, []).append(target_id)
        for label, target_ids in by_label.items():
            ordered = sorted(target_ids, key=lambda value: centers[value])
            for idx, target_id in enumerate(ordered):
                mask = instance_mask == target_id
                bbox = mask_to_bbox(mask)
                command, disambiguation = self._disambiguated_command(label, idx, len(ordered))
                entries.append(ObjectLanguageEntry(
                    split=sample.split,
                    scene_id=sample.scene_id,
                    camera=sample.camera,
                    frame_id=sample.frame_id,
                    target_id=target_id,
                    target_label=label,
                    command=command,
                    mask_path=str(sample.target_mask_path) if sample.target_mask_path else None,
                    bbox=bbox,
                    disambiguation_type=disambiguation,
                    visibility_score=float(mask.mean()),
                ))
        return entries

    @staticmethod
    def command_from_label(label: str) -> str:
        label = (label or "target object").replace("_", " ")
        return f"pick the {label}"

    @staticmethod
    def _disambiguated_command(label: str, idx: int, total: int) -> tuple[str, str]:
        phrase = label.replace("_", " ")
        if total <= 1:
            return f"pick the {phrase}", "category_label"
        if total == 2:
            pos = "left" if idx == 0 else "right"
        elif idx == 0:
            pos = "left"
        elif idx == total - 1:
            pos = "right"
        else:
            pos = "center"
        return f"pick the {pos} {phrase}", "spatial_disambiguation"

    @staticmethod
    def _bbox_area_score(bbox: list[int] | None) -> float | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return float(max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1))


def save_mapping(entries: list[ObjectLanguageEntry], output_root: Path) -> None:
    import json
    import pandas as pd

    out = Path(output_root) / "mappings"
    out.mkdir(parents=True, exist_ok=True)
    rows = [entry.to_json() for entry in entries]
    pd.DataFrame(rows).to_csv(out / "object_language_mapping.csv", index=False)
    with open(out / "object_language_mapping.json", "w") as f:
        json.dump(rows, f, indent=2)
