from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from target.mask_utils import bbox_to_mask, compute_mask_center, mask_to_bbox


def test_bbox_mask_roundtrip() -> None:
    mask = bbox_to_mask([2, 3, 7, 8], (12, 14))
    assert mask_to_bbox(mask) == [2, 3, 7, 8]
    assert compute_mask_center(mask) == (4.5, 5.5)
