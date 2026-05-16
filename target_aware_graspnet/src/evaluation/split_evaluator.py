from __future__ import annotations

from pathlib import Path

from evaluation.evaluator import OutputEvaluator


class SplitEvaluator:
    def __init__(self, thresholds: dict, mode: str = "proxy"):
        self.evaluator = OutputEvaluator(thresholds)
        self.mode = mode

    def evaluate_by_split(self, output_root: Path) -> list[dict]:
        rows = []
        for split_dir in sorted(Path(output_root).iterdir()):
            if not split_dir.is_dir() or split_dir.name == "paper_figures":
                continue
            records = self.evaluator.load_best_grasps(split_dir)
            if not records:
                continue
            row = {"split": split_dir.name}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode))
            rows.append(row)
        return rows

    def evaluate_by_scene(self, output_root: Path) -> list[dict]:
        rows = []
        for split_dir in sorted(Path(output_root).iterdir()):
            if not split_dir.is_dir():
                continue
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                records = self.evaluator.load_best_grasps(scene_dir)
                if not records:
                    continue
                row = {"split": split_dir.name, "scene_id": scene_dir.name}
                row.update(self.evaluator.evaluate_records(records, mode=self.mode))
                rows.append(row)
        return rows
