from __future__ import annotations

from pathlib import Path
from collections import defaultdict

from evaluation.evaluator import OutputEvaluator


class SplitEvaluator:
    def __init__(self, thresholds: dict, mode: str = "proxy"):
        self.evaluator = OutputEvaluator(thresholds)
        self.mode = mode

    def evaluate_by_split(self, output_root: Path) -> list[dict]:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[(record.get("dataset") or "unknown", record.get("split") or "unknown")].append(record)
        rows = []
        for (dataset, split), records in sorted(groups.items()):
            row = {"dataset": dataset, "split": split}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode))
            rows.append(row)
        return rows

    def evaluate_by_scene(self, output_root: Path) -> list[dict]:
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[(
                record.get("dataset") or "unknown",
                record.get("split") or "unknown",
                record.get("scene_id") or "unknown",
            )].append(record)
        rows = []
        for (dataset, split, scene_id), records in sorted(groups.items()):
            row = {"dataset": dataset, "split": split, "scene_id": scene_id}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode))
            rows.append(row)
        return rows

    def evaluate_by_dataset(self, output_root: Path) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[record.get("dataset") or "unknown"].append(record)
        rows = []
        for dataset, records in sorted(groups.items()):
            row = {"dataset": dataset}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode))
            rows.append(row)
        return rows
