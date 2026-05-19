from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.split_evaluator import SplitEvaluator


def _write_best(path: Path, *, split: str, scene_id: str, dataset: str = "OCID-VLG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset": dataset,
        "split": split,
        "scene_id": scene_id,
        "camera": "ocid",
        "frame_id": "frame_0001",
        "target_id": 2,
        "command": "Grasp the flashlight",
        "final_score": 0.7,
        "feature_breakdown": {
            "target_overlap": 1.0,
            "center_alignment": 1.0,
            "distance_to_target_center": 0.01,
            "collision_penalty": 0.0,
            "depth_stability": 1.0,
            "gripper_width_match": 1.0,
        },
        "top_k_fallback_candidates": [],
        "runtime": {"total": 0.1},
    }))


def test_split_evaluator_groups_by_record_split_not_output_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_best(
        output_root / "ocid_vlg" / "multiple" / "test" / "sample_a" / "best_grasp.json",
        split="test",
        scene_id="ARID10/floor/top/non-fruits/seq09",
    )
    _write_best(
        output_root / "ocid_vlg" / "multiple" / "train" / "sample_b" / "best_grasp.json",
        split="train",
        scene_id="ARID10/floor/top/non-fruits/seq09",
    )

    rows = SplitEvaluator({}).evaluate_by_split(output_root)

    assert {row["split"] for row in rows} == {"test", "train"}


def test_split_evaluator_groups_by_true_scene_id_from_best_json(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_best(
        output_root / "ocid_vlg" / "multiple" / "test" / "sample_a" / "best_grasp.json",
        split="test",
        scene_id="ARID10/floor/top/non-fruits/seq09",
    )

    rows = SplitEvaluator({}).evaluate_by_scene(output_root)

    assert rows[0]["split"] == "test"
    assert rows[0]["scene_id"] == "ARID10/floor/top/non-fruits/seq09"

