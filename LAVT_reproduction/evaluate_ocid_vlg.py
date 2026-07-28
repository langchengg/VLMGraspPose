"""Evaluate an OCID-VLG LAVT checkpoint and export prediction-only artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ocid_vlg.checkpoint import load_checkpoint
from ocid_vlg.device import resolve_device
from ocid_vlg.engine import (
    build_dataset,
    build_loader,
    build_model,
    configure_logging,
    evaluate_loader,
    PredictionExporter,
    save_json,
    save_metrics_tables,
    verify_prediction_export,
)
from train_ocid_vlg import parse_args


def _manifest_count(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _render_row(row: dict, destination: Path) -> None:
    with Image.open(row["image_path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(row["mask_path"]) as image:
        gt = np.asarray(image) == int(row["objID"])
    probability = np.load(row["target_probability"])
    with Image.open(row["target_mask"]) as image:
        prediction = np.asarray(image) != 0
    overlay = rgb.astype(np.float32).copy()
    overlay[prediction] = 0.45 * overlay[prediction] + 0.55 * np.array(
        [255, 64, 64], dtype=np.float32
    )
    figure, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    entries = (
        (rgb, "RGB", None),
        (gt, "Original GT", "gray"),
        (probability, "LAVT foreground probability", "viridis"),
        (prediction, "LAVT binary mask", "gray"),
        (overlay.astype(np.uint8), "Overlay", None),
    )
    for axis, (image, title, cmap) in zip(axes, entries):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{row['sentence']}\nIoU={row['IoU']:.4f} | "
        f"sent_id={row['sent_id']} | scene={row['scene_id']}",
        fontsize=10,
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(figure)


def _qualitative_figures(output_dir: Path, rows: list[dict]) -> dict[str, int]:
    if not rows:
        return {}
    ranked = sorted(rows, key=lambda row: (row["IoU"], row["sent_id"]))
    count = min(10, len(ranked))
    midpoint = max(0, len(ranked) // 2 - count // 2)
    groups = {
        "best": list(reversed(ranked[-count:])),
        "worst": ranked[:count],
        "median": ranked[midpoint : midpoint + count],
    }
    for group, selected in groups.items():
        for index, row in enumerate(selected):
            _render_row(
                row,
                output_dir / "figures" / group / f"{index:02d}_{row['sent_id']}.png",
            )
    return {name: len(values) for name, values in groups.items()}


def main() -> int:
    args = parse_args()
    if not args.resume:
        raise ValueError("--resume checkpoint is required")
    if not args.test_manifest:
        raise ValueError("--test_manifest is required")
    if not args.ocid_root or not args.ocid_api_root:
        raise ValueError("--ocid_root and --ocid_api_root are required")
    device = resolve_device(args.device)
    checkpoint_path = Path(args.resume).expanduser().resolve()
    output_dir = (
        Path(args.resolved_run_dir).expanduser().resolve()
        if args.resolved_run_dir
        else checkpoint_path.parent.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "evaluation.log")
    logger.info("checkpoint=%s device=%s", checkpoint_path, device)

    dataset = build_dataset(
        args,
        "test",
        args.test_manifest,
        args.limit_test_samples,
    )
    loader = build_loader(
        dataset,
        batch_size=1,
        train=False,
        num_workers=args.num_workers,
        device=device,
    )
    model = build_model(args, device, initialize_backbone=False)
    payload = load_checkpoint(
        checkpoint_path,
        model,
        device=device,
        strict=True,
        restore_rng=False,
    )
    checkpoint_config = payload.get("config") or {}
    for key in ("model", "swin_type", "img_size", "max_tokens"):
        if key in checkpoint_config and checkpoint_config[key] != getattr(args, key):
            raise ValueError(
                f"checkpoint/config mismatch for {key}: "
                f"{checkpoint_config[key]!r} != {getattr(args, key)!r}"
            )
    exporter = PredictionExporter(
        output_dir,
        checkpoint_path=checkpoint_path,
        model_name=args.model,
        backbone=f"swin_{args.swin_type}",
        prediction_policy=args.prediction_policy,
        threshold=args.threshold,
    )
    metrics_model, metrics_original, rows = evaluate_loader(
        model,
        loader,
        device,
        prediction_policy=args.prediction_policy,
        threshold=args.threshold,
        collect_arrays=True,
        sample_callback=exporter.write,
    )
    save_json(output_dir / "metrics_test_model_resolution.json", metrics_model)
    save_json(output_dir / "metrics_test_original_resolution.json", metrics_original)
    save_metrics_tables(output_dir, rows)

    prediction_manifest = exporter.finalize()
    verification = verify_prediction_export(prediction_manifest, len(dataset))
    save_json(output_dir / "prediction_export_audit.json", verification)
    exported_by_id = {
        row["sent_id"]: row for row in exporter.manifest_rows
    }
    figure_rows = [
        {**row, **exported_by_id[row["sent_id"]]} for row in rows
    ]
    figure_counts = _qualitative_figures(output_dir, figure_rows)

    expected_full = _manifest_count(args.test_manifest)
    checkpoint_completed_epochs = int(payload.get("next_epoch", 0))
    run_status_path = output_dir / "RUN_STATUS.json"
    run_status = (
        json.loads(run_status_path.read_text(encoding="utf-8"))
        if run_status_path.is_file()
        else {}
    )
    training_run_completed_epochs = int(
        run_status.get("completed_epochs", checkpoint_completed_epochs)
    )
    same_run_directory = checkpoint_path.parent.parent.resolve() == output_dir.resolve()
    full_test = args.limit_test_samples is None and len(dataset) == expected_full
    full_success = (
        full_test
        and same_run_directory
        and training_run_completed_epochs >= 40
        and int(checkpoint_config.get("epochs", 0)) >= 40
        and not verification["errors"]
    )
    status = (
        "SUCCESS_FULL"
        if full_success and args.swin_type == "base"
        else "SUCCESS_FULL_FALLBACK_BACKBONE"
        if full_success
        else "SUCCESS_SMOKE_ONLY"
        if args.limit_test_samples is not None
        else "PARTIAL_TRAINING"
    )
    save_json(
        output_dir / "EVALUATION_STATUS.json",
        {
            "status": status,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "checkpoint_completed_epochs": checkpoint_completed_epochs,
            "training_run_completed_epochs": training_run_completed_epochs,
            "completed_epochs": training_run_completed_epochs,
            "same_run_directory": same_run_directory,
            "evaluated_samples": len(dataset),
            "expected_full_test_samples": expected_full,
            "prediction_manifest": str(prediction_manifest.resolve()),
            "model_resolution": metrics_model,
            "original_resolution": metrics_original,
            "qualitative_figure_counts": figure_counts,
        },
    )
    logger.info(
        "status=%s N=%d original_mIoU=%.6f original_oIoU=%.6f",
        status,
        len(dataset),
        metrics_original["mean_iou"],
        metrics_original["overall_iou"],
    )
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
