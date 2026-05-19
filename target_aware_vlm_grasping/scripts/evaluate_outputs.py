from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.report_generator import generate_reports
from main import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target-aware grasp outputs.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--mode", choices=["proxy", "annotation", "ocid_2d"], default="proxy")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config_dir)
    if args.mode == "annotation":
        thresholds = config.get("evaluation", {}).get("annotation_thresholds", {})
    elif args.mode == "ocid_2d":
        thresholds = config.get("evaluation", {}).get("ocid_2d_thresholds", {})
    else:
        thresholds = config.get("evaluation", {}).get("proxy_thresholds", {})
    if args.mode == "annotation":
        print("Annotation mode compares against annotation_valid_grasps if present in best_grasp.json; otherwise annotation rates are 0.")
    if args.mode == "ocid_2d":
        print("OCID 2D mode reports whether projected grasp centers fall inside GT grasp rectangles.")
    generate_reports(args.output_root, thresholds, mode=args.mode)
    print(f"metrics_by_dataset: {args.output_root / 'metrics_by_dataset.csv'}")
    print(f"metrics_by_split: {args.output_root / 'metrics_by_split.csv'}")
    print(f"metrics_by_scene: {args.output_root / 'metrics_by_scene.csv'}")
    print(f"metrics_by_target_source: {args.output_root / 'metrics_by_target_source.csv'}")
    print(f"metrics_by_scorer: {args.output_root / 'metrics_by_scorer.csv'}")
    print(f"runtime_report: {args.output_root / 'runtime_report.csv'}")
    print(f"failure_cases: {args.output_root / 'failure_cases.csv'}")


if __name__ == "__main__":
    main()
