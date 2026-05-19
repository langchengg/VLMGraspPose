from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.ocid_grasp_loader import OCIDGraspIndexBuilder
from dataset.ocid_vlg_loader import OCIDVLGIndexBuilder
from main import TargetAwareGraspPipeline, frame_result_row, load_config, write_summary_csv
from utils.io_utils import ensure_dir


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _default_dataset_root(dataset: str) -> Path:
    if dataset == "ocid_grasp":
        return Path("data/OCID-Grasp")
    return Path("data/OCID-VLG")


def build_samples(args, output_root: Path):
    dataset_root = _resolve(args.dataset_root or _default_dataset_root(args.dataset))
    if args.dataset == "ocid_vlg":
        builder = OCIDVLGIndexBuilder(dataset_root, output_root)
        refer_splits = _resolve_refer_splits(dataset_root, args.refer_split)
        splits = _resolve_splits(args.split)
        samples = []
        for refer_split in refer_splits:
            for split in splits:
                expressions = dataset_root / "refer" / refer_split / f"{split}_expressions.json"
                if not expressions.exists():
                    continue
                remaining = None if args.max_samples is None else max(args.max_samples - len(samples), 0)
                if remaining == 0:
                    return samples
                samples.extend(builder.build(refer_split=refer_split, split=split, max_samples=remaining))
        return samples
    return OCIDGraspIndexBuilder(dataset_root, output_root).build(max_samples=args.max_samples)


def _resolve_refer_splits(dataset_root: Path, value: str) -> list[str]:
    if value != "all":
        return [value]
    refer_root = dataset_root / "refer"
    return sorted(path.name for path in refer_root.iterdir() if path.is_dir())


def _resolve_splits(value: str) -> list[str]:
    if value != "all":
        return [value]
    return ["train", "val", "test"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run OCID language-conditioned target-aware RGB-D grasping.")
    parser.add_argument("--dataset", choices=["ocid_vlg", "ocid_grasp"], default="ocid_vlg")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--refer-split", default="multiple", help="OCID-VLG refer split, or 'all'.")
    parser.add_argument("--split", default="test", help="OCID-VLG split: train/val/test, or 'all'.")
    parser.add_argument("--target-source", choices=["oracle", "vlm"], default="oracle")
    parser.add_argument("--vlm-backend", default="florence2")
    parser.add_argument("--scorer", choices=["rule_based", "mlp"], default="rule_based")
    parser.add_argument("--mlp-checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    output_root = _resolve(args.output_root)
    ensure_dir(output_root)
    samples = build_samples(args, output_root)
    pipeline = TargetAwareGraspPipeline(load_config(args.config_dir))
    results = []
    failures = []

    for sample in tqdm(samples, desc=f"{args.dataset}:{args.target_source}", unit="sample"):
        if args.resume and (sample.output_dir / "best_grasp.json").exists() and not args.overwrite:
            continue
        result = pipeline.run_dataset_sample(
            sample,
            target_source=args.target_source,
            vlm_backend=args.vlm_backend,
            scorer=args.scorer,
            mlp_checkpoint=args.mlp_checkpoint,
            top_k=args.top_k,
            overwrite=args.overwrite,
        )
        results.append(result)
        if result.status == "failed":
            failures.append(frame_result_row(result))

    write_summary_csv(output_root / "summary.csv", results)
    if failures:
        pd.DataFrame(failures).to_csv(output_root / "failure_cases.csv", index=False)
    elif not (output_root / "failure_cases.csv").exists():
        pd.DataFrame([]).to_csv(output_root / "failure_cases.csv", index=False)
    print(f"processed={len(results)} failures={len(failures)} output_root={output_root}")


if __name__ == "__main__":
    main()
