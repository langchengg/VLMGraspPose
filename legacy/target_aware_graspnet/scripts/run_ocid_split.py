from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from _common import ROOT, log_failure
from dataset.ocid_vlg_loader import OCIDVLGIndexBuilder
from main import TargetAwareGraspPipeline, load_config, write_summary_csv


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCID-VLG split over language-conditioned target samples.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--refer-split", default=None, choices=["unique", "multiple", "novel-classes", "novel-instances"])
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--target-grounder", choices=["annotation", "florence2"], default=None)
    parser.add_argument("--florence-model-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config_dir)
    if args.target_grounder:
        config.setdefault("target_grounding", {})["method"] = args.target_grounder
    if args.florence_model_id:
        config.setdefault("target_grounding", {}).setdefault("florence2", {})["model_id"] = args.florence_model_id
    ocid_cfg = config.get("ocid_vlg", {})
    sampler_cfg = config.get("sampler", {})
    dataset_root = _resolve(args.dataset_root or Path(ocid_cfg.get("root", "../data/raw/OCID-VLG")))
    output_root = _resolve(args.output_root or Path(ocid_cfg.get("output_root", "outputs")))
    refer_split = args.refer_split or ocid_cfg.get("refer_split", "multiple")
    split = args.split or ocid_cfg.get("split", "test")
    top_k = args.top_k or sampler_cfg.get("top_k", 5)

    samples = OCIDVLGIndexBuilder(dataset_root, output_root).build(refer_split, split, max_samples=args.max_samples)
    pipeline = TargetAwareGraspPipeline(config)
    results = []
    for sample in tqdm(samples, desc=f"ocid_vlg/{refer_split}/{split}"):
        result = pipeline.run_ocid_sample(sample, top_k=top_k, overwrite=args.overwrite)
        log_failure(output_root, result)
        results.append(result)
    summary_path = output_root / "ocid_vlg" / refer_split / split / "summary.csv"
    write_summary_csv(summary_path, results)
    write_summary_csv(output_root / "summary.csv", results)
    print(f"processed_units: {len(results)}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
