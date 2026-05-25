from __future__ import annotations

from pathlib import Path

import pandas as pd
from evaluation.split_evaluator import SplitEvaluator


def generate_reports(output_root: Path, thresholds: dict, mode: str = "proxy") -> None:
    output_root = Path(output_root)
    evaluator = SplitEvaluator(thresholds, mode=mode)
    dataset_rows = evaluator.evaluate_by_dataset(output_root)
    split_rows = evaluator.evaluate_by_split(output_root)
    scene_rows = evaluator.evaluate_by_scene(output_root)
    target_source_rows = evaluator.evaluate_by_target_source(output_root)
    scorer_rows = evaluator.evaluate_by_scorer(output_root)
    pd.DataFrame(dataset_rows).to_csv(output_root / "metrics_by_dataset.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_root / "metrics_by_split.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(output_root / "metrics_by_scene.csv", index=False)
    pd.DataFrame(target_source_rows).to_csv(output_root / "metrics_by_target_source.csv", index=False)
    pd.DataFrame(scorer_rows).to_csv(output_root / "metrics_by_scorer.csv", index=False)
    runtime_rows = []
    for best in output_root.glob("**/best_grasp.json"):
        import json
        with open(best) as f:
            rec = json.load(f)
        runtime_rows.append({
            "split": rec.get("split"),
            "scene_id": rec.get("scene_id"),
            "camera": rec.get("camera"),
            "frame_id": rec.get("frame_id"),
            "target_id": rec.get("target_id"),
            "command": rec.get("command"),
            "runtime_total": sum(rec.get("runtime", {}).values()),
            **{f"runtime_{k}": v for k, v in rec.get("runtime", {}).items()},
        })
    pd.DataFrame(runtime_rows).to_csv(output_root / "runtime_report.csv", index=False)
    failure_columns = ["dataset", "split", "scene_id", "camera", "frame_id", "target_id", "command", "error"]
    failure_rows = []
    for err in output_root.glob("**/error.json"):
        import json
        with open(err) as f:
            rec = json.load(f)
        sample = rec.get("sample", {})
        sample_metadata = sample.get("metadata", {})
        failure_rows.append({
            "dataset": sample.get("dataset_name") or sample_metadata.get("dataset"),
            "split": sample.get("split"),
            "scene_id": sample.get("scene_id"),
            "camera": sample.get("camera"),
            "frame_id": sample.get("frame_id"),
            "target_id": sample.get("target_id") or sample_metadata.get("target_id"),
            "command": sample.get("command") or sample.get("sentence") or sample_metadata.get("command"),
            "error": rec.get("error_message"),
        })
    pd.DataFrame(failure_rows, columns=failure_columns).to_csv(output_root / "failure_cases.csv", index=False)
