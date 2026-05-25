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
        failures: dict[tuple[str, str], int] = defaultdict(int)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[(record.get("dataset") or "unknown", record.get("split") or "unknown")].append(record)
        for record in self.evaluator.load_error_records(output_root):
            failures[(record.get("dataset") or "unknown", record.get("split") or "unknown")] += 1
        rows = []
        for dataset, split in sorted(set(groups) | set(failures)):
            records = groups[(dataset, split)]
            row = {"dataset": dataset, "split": split}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode, failures=failures[(dataset, split)]))
            rows.append(row)
        return rows

    def evaluate_by_scene(self, output_root: Path) -> list[dict]:
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        failures: dict[tuple[str, str, str], int] = defaultdict(int)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[(
                record.get("dataset") or "unknown",
                record.get("split") or "unknown",
                record.get("scene_id") or "unknown",
            )].append(record)
        for record in self.evaluator.load_error_records(output_root):
            failures[(
                record.get("dataset") or "unknown",
                record.get("split") or "unknown",
                record.get("scene_id") or "unknown",
            )] += 1
        rows = []
        for dataset, split, scene_id in sorted(set(groups) | set(failures)):
            records = groups[(dataset, split, scene_id)]
            row = {"dataset": dataset, "split": split, "scene_id": scene_id}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode, failures=failures[(dataset, split, scene_id)]))
            rows.append(row)
        return rows

    def evaluate_by_dataset(self, output_root: Path) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        failures: dict[str, int] = defaultdict(int)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[record.get("dataset") or "unknown"].append(record)
        for record in self.evaluator.load_error_records(output_root):
            failures[record.get("dataset") or "unknown"] += 1
        rows = []
        for dataset in sorted(set(groups) | set(failures)):
            records = groups[dataset]
            row = {"dataset": dataset}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode, failures=failures[dataset]))
            rows.append(row)
        return rows

    def evaluate_by_target_source(self, output_root: Path) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        failures: dict[str, int] = defaultdict(int)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[record.get("target_source") or "unknown"].append(record)
        for record in self.evaluator.load_error_records(output_root):
            failures[record.get("target_source") or "unknown"] += 1
        rows = []
        for target_source in sorted(set(groups) | set(failures)):
            records = groups[target_source]
            row = {"target_source": target_source}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode, failures=failures[target_source]))
            rows.append(row)
        return rows

    def evaluate_by_scorer(self, output_root: Path) -> list[dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        failures: dict[str, int] = defaultdict(int)
        for record in self.evaluator.load_best_grasps(output_root):
            groups[record.get("scorer") or record.get("scorer_type") or "unknown"].append(record)
        for record in self.evaluator.load_error_records(output_root):
            failures[record.get("scorer") or "unknown"] += 1
        rows = []
        for scorer in sorted(set(groups) | set(failures)):
            records = groups[scorer]
            row = {"scorer": scorer}
            row.update(self.evaluator.evaluate_records(records, mode=self.mode, failures=failures[scorer]))
            rows.append(row)
        return rows
