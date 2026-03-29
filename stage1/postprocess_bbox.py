"""
stage1/postprocess_bbox.py — Bbox utilities & saving Stage-1 outputs
=====================================================================
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from stage1.grounding import GroundingResult


def save_stage1_output(
    sample_id: str,
    text_query: str,
    result: GroundingResult,
    output_dir: Path = config.STAGE1_OUTPUT_DIR,
) -> Path:
    """Persist Stage-1 grounding result as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_id}.json"

    record = {
        "sample_id": sample_id,
        "text_query": text_query,
        "bbox": result.bbox,
        "confidence": result.confidence,
        "has_mask": result.mask is not None,
    }

    # Optionally save mask as .npy alongside the JSON
    if result.mask is not None:
        mask_path = output_dir / f"{sample_id}_mask.npy"
        np.save(str(mask_path), result.mask)
        record["mask_path"] = str(mask_path)

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return out_path


def load_stage1_output(
    sample_id: str,
    output_dir: Path = config.STAGE1_OUTPUT_DIR,
) -> Dict:
    """Load a previously saved Stage-1 result."""
    path = output_dir / f"{sample_id}.json"
    with open(path) as f:
        record = json.load(f)

    # Load mask if available
    if record.get("mask_path"):
        record["mask"] = np.load(record["mask_path"])
    else:
        record["mask"] = None

    return record


def pad_bbox(bbox, pad=10, img_w=config.IMAGE_WIDTH, img_h=config.IMAGE_HEIGHT):
    """Expand bounding box by *pad* pixels, clipping to image bounds."""
    x1, y1, x2, y2 = bbox
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(img_w - 1, x2 + pad),
        min(img_h - 1, y2 + pad),
    ]
