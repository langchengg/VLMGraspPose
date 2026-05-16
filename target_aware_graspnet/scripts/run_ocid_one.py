from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT
from dataset.ocid_vlg_loader import OCIDVLGIndexBuilder
from main import TargetAwareGraspPipeline, load_config


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one OCID-VLG language-conditioned target sample.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--refer-split", default=None, help="unique/multiple/novel-classes/novel-instances")
    parser.add_argument("--split", default=None, help="train/val/test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config_dir)
    ocid_cfg = config.get("ocid_vlg", {})
    sampler_cfg = config.get("sampler", {})
    dataset_root = _resolve(args.dataset_root or Path(ocid_cfg.get("root", "../data/raw/OCID-VLG")))
    output_root = _resolve(args.output_root or Path(ocid_cfg.get("output_root", "outputs")))
    refer_split = args.refer_split or ocid_cfg.get("refer_split", "multiple")
    split = args.split or ocid_cfg.get("split", "test")
    top_k = args.top_k or sampler_cfg.get("top_k", 5)

    samples = OCIDVLGIndexBuilder(dataset_root, output_root).build(refer_split, split, max_samples=args.index + 1)
    if args.index >= len(samples):
        raise SystemExit(f"Index {args.index} is out of range for {refer_split}/{split}.")
    sample = samples[args.index]
    result = TargetAwareGraspPipeline(config).run_ocid_sample(sample, top_k=top_k, overwrite=args.overwrite)
    print(f"{sample.image_id}: {result.status}")
    print(f"  command: {sample.command}")
    print(f"  target: {sample.target_label} index={sample.target_index} bbox={sample.target_bbox}")
    print(f"  gt_grasps: {len(sample.grasp_rectangles)}")
    if result.best_grasp:
        print(f"  final_score: {result.best_grasp.final_score:.4f}")
    if result.error_message:
        print(f"  error: {result.error_message}")
    print(f"  output_dir: {sample.output_dir}")


if __name__ == "__main__":
    main()
