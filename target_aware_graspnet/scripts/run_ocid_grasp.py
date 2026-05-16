from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from _common import ROOT, log_failure
from dataset.ocid_vlg_loader import OCIDGraspIndexBuilder
from main import TargetAwareGraspPipeline, load_config, write_summary_csv


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCID-Grasp fallback samples with generated class commands.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config_dir)
    ocid_cfg = config.get("ocid_vlg", {})
    sampler_cfg = config.get("sampler", {})
    dataset_root = _resolve(args.dataset_root or Path(ocid_cfg.get("root", "../data/raw/OCID-VLG")))
    output_root = _resolve(args.output_root or Path(ocid_cfg.get("output_root", "outputs")))
    top_k = args.top_k or sampler_cfg.get("top_k", 5)

    samples = OCIDGraspIndexBuilder(dataset_root, output_root).build(max_samples=args.max_samples)
    pipeline = TargetAwareGraspPipeline(config)
    results = []
    for sample in tqdm(samples, desc="ocid_grasp"):
        result = pipeline.run_ocid_sample(sample, top_k=top_k, overwrite=args.overwrite)
        log_failure(output_root, result)
        results.append(result)
    summary_path = output_root / "ocid_grasp" / "summary.csv"
    write_summary_csv(summary_path, results)
    print(f"processed_units: {len(results)}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
