from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from dataset.camera_loader import load_label
from target.mask_utils import mask_to_bbox


def load_object_ids(scene_dir: Path) -> list[int]:
    path = scene_dir / "object_id_list.txt"
    if not path.exists():
        return []
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            ids.append(int(line))
    return ids


def visible_instances_from_label(label_path: Path) -> list[dict]:
    if not label_path or not label_path.exists():
        return []
    label = load_label(label_path)
    instances = []
    for mask_val in sorted(int(x) for x in np.unique(label) if int(x) > 0):
        mask = label == mask_val
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        instances.append({
            "target_id": mask_val - 1,
            "mask_val": mask_val,
            "bbox": bbox,
            "pixel_count": int(mask.sum()),
        })
    return instances


def select_largest_visible_instance(label_path: Optional[Path]) -> Optional[dict]:
    instances = visible_instances_from_label(label_path) if label_path else []
    if not instances:
        return None
    return max(instances, key=lambda x: x["pixel_count"])
