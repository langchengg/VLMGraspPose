from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from target.florence2_grounder import Florence2Grounder


class FakeFlorenceBackend:
    def ground(self, rgb: np.ndarray, command: str) -> dict:
        return {
            "bbox": [10, 12, 30, 32],
            "label": "red cup",
            "score": 0.83,
            "raw": {"source": "fake"},
        }


def test_florence2_grounder_converts_bbox_to_target_region() -> None:
    rgb = np.zeros((64, 80, 3), dtype=np.uint8)
    grounder = Florence2Grounder(backend=FakeFlorenceBackend())

    target = grounder.ground(rgb, "pick the red cup")

    assert target.label == "red cup"
    assert target.command == "pick the red cup"
    assert target.bbox == [10, 12, 30, 32]
    assert target.mask is not None
    assert target.mask[12:33, 10:31].all()
    assert target.grounding_score == 0.83
    assert target.metadata["grounding_model"] == "Florence-2"

