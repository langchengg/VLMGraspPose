#!/usr/bin/env python3
"""Fine-tune one preregistered GR-ConvNet or GG-CNN2 configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends.training import TrainingConfig, train_finetuned_backend  # noqa: E402
from src.grasping.common.sample_io import (  # noqa: E402
    read_deployment_manifest,
    read_label_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("grconvnet", "ggcnn2"), required=True)
    parser.add_argument(
        "--conditioning", choices=("hard_mask", "dilated_crop"), required=True
    )
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--dilation", type=float, default=0.15)
    parser.add_argument("--minimum-crop-side", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--quality-threshold", type=float, default=0.0)
    parser.add_argument("--min-peak-distance", type=int, default=20)
    parser.add_argument("--center-gate-exponent", type=float, default=1.0)
    parser.add_argument("--jaw-gate-exponent", type=float, default=0.0)
    parser.add_argument("--nms-center-distance", type=float, default=8.0)
    parser.add_argument("--nms-angle-distance", type=float, default=15.0)
    parser.add_argument("--nms-width-distance", type=float, default=10.0)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--grconvnet-source-checkpoint")
    parser.add_argument("--grconvnet-source-checkpoint-sha256")
    parser.add_argument("--grconvnet-input-channels", type=int, default=4)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_prefix = (PROJECT_ROOT / ".venv-grasp4dof").resolve()
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError(
            f"training requires isolated environment {expected_prefix}; "
            f"observed {Path(sys.prefix).resolve()}"
        )
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    run_dir = args.run_dir.expanduser().resolve()
    manifests = run_dir / "manifests"
    train_rows = read_deployment_manifest(manifests / "train_samples.parquet")
    train_labels = read_label_manifest(manifests / "train_labels.parquet")
    val_rows = read_deployment_manifest(manifests / "validation_samples.parquet")
    val_labels = read_label_manifest(manifests / "validation_labels.parquet")
    if args.limit_train is not None:
        train_rows, train_labels = (
            train_rows[: args.limit_train],
            train_labels[: args.limit_train],
        )
    if args.limit_validation is not None:
        val_rows, val_labels = (
            val_rows[: args.limit_validation],
            val_labels[: args.limit_validation],
        )
    job_name = (
        f"{args.conditioning}_s{args.input_size}_lr{args.learning_rate:g}"
        f"_wd{args.weight_decay:g}_seed{args.seed}"
    )
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / args.backend / "finetuned" / job_name
    )
    result = train_finetuned_backend(
        train_deployment=train_rows,
        train_labels=train_labels,
        validation_deployment=val_rows,
        validation_labels=val_labels,
        output_dir=output,
        config=TrainingConfig(
            backend=args.backend,
            input_size=args.input_size,
            conditioning_variant=args.conditioning,
            dilation_fraction=args.dilation,
            minimum_crop_side_px=args.minimum_crop_side,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            quality_threshold=args.quality_threshold,
            min_peak_distance_px=args.min_peak_distance,
            center_gate_exponent=args.center_gate_exponent,
            jaw_gate_exponent=args.jaw_gate_exponent,
            nms_center_distance_px=args.nms_center_distance,
            nms_angle_distance_deg=args.nms_angle_distance,
            nms_width_distance_px=args.nms_width_distance,
            nms_iou_threshold=args.nms_iou_threshold,
            grconvnet_source_checkpoint=args.grconvnet_source_checkpoint,
            grconvnet_source_checkpoint_sha256=args.grconvnet_source_checkpoint_sha256,
            grconvnet_input_channels=args.grconvnet_input_channels,
        ),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
