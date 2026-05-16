from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _common import ROOT


def _best_records(output_root: Path) -> list[dict]:
    records = []
    for path in sorted(output_root.glob("**/best_grasp.json")):
        with open(path) as f:
            rec = json.load(f)
        rec["_path"] = path
        records.append(rec)
    return records


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _score_bar(record: dict, out_path: Path) -> None:
    features = record.get("feature_breakdown", {})
    names = [
        "initial_geometric_score",
        "target_overlap",
        "center_alignment",
        "gripper_width_match",
        "depth_stability",
        "approach_direction_score",
        "collision_penalty",
        "boundary_penalty",
    ]
    values = [features.get(name, 0.0) for name in names]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.bar(range(len(names)), values, color="#4a7ebb")
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _export_case(record: dict, case_dir: Path) -> None:
    best_path = Path(record["_path"])
    frame_dir = best_path.parent
    label = f"{record.get('split')}_{record.get('scene_id')}_{record.get('camera')}_{record.get('frame_id')}_target_{int(record.get('target_id', 0)):03d}"
    target_dir = case_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ["visualization_rgb.png", "visualization_3d.png", "target_mask.png", "score_breakdown.json", "best_grasp.json"]:
        _copy_if_exists(frame_dir / name, target_dir / name)
    _score_bar(record, target_dir / "score_breakdown_bar.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export representative figures and result tables for a paper draft.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--num-success", type=int, default=12)
    parser.add_argument("--num-failure", type=int, default=6)
    args = parser.parse_args()

    output_root = args.output_root
    figure_root = output_root / "paper_figures"
    records = _best_records(output_root)
    records = sorted(records, key=lambda rec: rec.get("final_score", 0.0), reverse=True)
    for record in records[: args.num_success]:
        _export_case(record, figure_root / "success_cases")

    failure_csv = output_root / "failure_cases.csv"
    if failure_csv.exists() and failure_csv.stat().st_size > 0:
        failures = pd.read_csv(failure_csv).head(args.num_failure)
        fail_dir = figure_root / "failure_cases"
        fail_dir.mkdir(parents=True, exist_ok=True)
        failures.to_csv(fail_dir / "selected_failure_cases.csv", index=False)

    metric_csv = output_root / "metrics_by_split.csv"
    runtime_csv = output_root / "runtime_report.csv"
    if metric_csv.exists():
        shutil.copy2(metric_csv, figure_root / "table_quantitative_results.csv")
        shutil.copy2(metric_csv, figure_root / "table_ablation.csv")
    if runtime_csv.exists():
        shutil.copy2(runtime_csv, figure_root / "table_runtime.csv")
    print(f"paper_figures: {figure_root}")


if __name__ == "__main__":
    main()
