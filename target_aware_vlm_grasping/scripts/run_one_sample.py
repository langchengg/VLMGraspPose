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

from dataset.ocid_grasp_loader import OCIDGraspIndexBuilder
from dataset.ocid_vlg_loader import OCIDVLGIndexBuilder
from dataset.single_object_loader import SingleObjectIndexBuilder
from main import TargetAwareGraspPipeline, load_config


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _default_dataset_root(dataset: str) -> Path:
    if dataset == "single_object":
        return Path("data")
    if dataset == "ocid_grasp":
        return Path("data/OCID-Grasp")
    return Path("data/OCID-VLG")


def _build_samples(args, output_root: Path):
    dataset_root = _resolve(args.dataset_root or _default_dataset_root(args.dataset))
    if args.dataset == "ocid_vlg":
        return OCIDVLGIndexBuilder(dataset_root, output_root).build(
            refer_split=args.refer_split,
            split=args.split,
            max_samples=None if args.sample_id else args.index + 1,
        )
    if args.dataset == "single_object":
        objects = _parse_objects(args.objects)
        return SingleObjectIndexBuilder(dataset_root, output_root).build(
            objects=objects,
            max_samples=None if args.sample_id else args.index + 1,
            samples_per_object=args.samples_per_object,
        )
    return OCIDGraspIndexBuilder(dataset_root, output_root).build(
        max_samples=None if args.sample_id else args.index + 1
    )


def _parse_objects(value: str | None) -> list[str] | None:
    if not value or value == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one language-conditioned RGB-D grasping sample.")
    parser.add_argument("--dataset", choices=["ocid_vlg", "ocid_grasp", "single_object"], default="ocid_vlg")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/debug"))
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--refer-split", default="multiple")
    parser.add_argument("--split", default="test")
    parser.add_argument("--objects", default="all", help="For single_object: comma-separated object folders or prefixes.")
    parser.add_argument("--samples-per-object", type=int, default=None)
    parser.add_argument("--target-source", choices=["oracle", "vlm"], default="oracle")
    parser.add_argument("--vlm-backend", default="florence2")
    parser.add_argument("--scorer", choices=["rule_based", "mlp", "xgboost"], default="rule_based")
    parser.add_argument("--mlp-checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    output_root = _resolve(args.output_root)
    samples = _build_samples(args, output_root)
    if args.sample_id:
        matches = [sample for sample in samples if sample.sample_id == args.sample_id or sample.image_id == args.sample_id]
        if not matches:
            raise SystemExit(f"Sample id not found: {args.sample_id}")
        sample = matches[0]
    else:
        if args.index >= len(samples):
            raise SystemExit(f"Index {args.index} is out of range.")
        sample = samples[args.index]

    config = load_config(args.config_dir)
    result = TargetAwareGraspPipeline(config).run_dataset_sample(
        sample,
        target_source=args.target_source,
        vlm_backend=args.vlm_backend,
        scorer=args.scorer,
        mlp_checkpoint=args.mlp_checkpoint,
        top_k=args.top_k,
        overwrite=args.overwrite,
    )
    print(f"{sample.sample_id}: {result.status}")
    print(f"  command: {sample.command}")
    print(f"  target: {sample.target_label} id={sample.target_id} bbox={sample.target_bbox_gt}")
    if result.best_grasp:
        print(f"  final_score: {result.best_grasp.final_score:.4f}")
        print(f"  position: {result.best_grasp.candidate.position.tolist()}")
    if result.error_message:
        print(f"  error: {result.error_message}")
    print(f"  output_dir: {sample.output_dir}")


if __name__ == "__main__":
    main()
