from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from target.florence2_grounder import Florence2Grounder
from target.command_parser import parse_command


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


class MultiCandidateBackend:
    def ground_all(self, rgb: np.ndarray, command: str) -> list[dict]:
        query = command.lower()
        if "blue marker" in query:
            return [{"bbox": [90, 20, 110, 80], "label": "blue marker", "score": 0.9}]
        if "food box" in query:
            return [
                {"bbox": [20, 20, 70, 80], "label": "food box", "score": 0.85},
                {"bbox": [150, 20, 210, 80], "label": "food box", "score": 0.8},
            ]
        if "marker" in query:
            return [
                {"bbox": [20, 90, 50, 130], "label": "marker", "score": 0.8},
                {"bbox": [170, 90, 200, 130], "label": "marker", "score": 0.9},
            ]
        return []


def test_command_parser_uses_target_label_for_queries() -> None:
    parsed = parse_command("Pick the left marker", "marker_058")

    assert parsed.target_phrase == "marker"
    assert parsed.target_queries == ["Pick the left marker", "marker", "the marker"]
    assert parsed.ordinal == "leftmost"


def test_florence2_grounder_selects_leftmost_candidate() -> None:
    rgb = np.zeros((160, 240, 3), dtype=np.uint8)
    grounder = Florence2Grounder(backend=MultiCandidateBackend())

    target = grounder.ground(rgb, "pick the leftmost marker", target_label="marker")

    assert target.bbox == [20, 90, 50, 130]
    assert target.metadata["selection_reason"].startswith("leftmost")


def test_florence2_grounder_uses_reference_relation() -> None:
    rgb = np.zeros((160, 240, 3), dtype=np.uint8)
    grounder = Florence2Grounder(backend=MultiCandidateBackend())

    target = grounder.ground(
        rgb,
        "pick the food box right of the blue marker",
        target_label="food_box_003",
    )

    assert target.bbox == [150, 20, 210, 80]
    assert "right_of" in target.metadata["selection_reason"]
    assert target.metadata["reference_bbox"] == [90, 20, 110, 80]
