#!/usr/bin/env python3
"""Fail closed unless a limited training run demonstrates real learning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--minimum-loss-drop-ratio", type=float, default=0.20)
    parser.add_argument("--minimum-miou-gain", type=float, default=0.05)
    args = parser.parse_args()

    history_path = args.run_dir / "metrics_history.jsonl"
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise RuntimeError("overfit gate requires at least two recorded epochs")
    first, final = rows[0], rows[-1]
    losses = [float(row["train_loss"]) for row in rows]
    mious = [
        float(row["validation_original_resolution"]["mean_iou"]) for row in rows
    ]
    all_finite = all(math.isfinite(value) for value in losses + mious)
    loss_drop_ratio = (losses[0] - losses[-1]) / losses[0]
    miou_gain = mious[-1] - mious[0]
    conditions = {
        "all_losses_and_mious_finite": all_finite,
        "loss_drop_ratio_at_least_threshold": (
            loss_drop_ratio >= args.minimum_loss_drop_ratio
        ),
        "final_miou_gain_at_least_threshold": (
            miou_gain >= args.minimum_miou_gain
        ),
        "no_missing_predictions": all(
            row["validation_original_resolution"]["missing_prediction_count"] == 0
            for row in rows
        ),
    }
    result = {
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "epochs": len(rows),
        "first_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "loss_drop_ratio": loss_drop_ratio,
        "first_original_miou": mious[0],
        "final_original_miou": mious[-1],
        "original_miou_gain": miou_gain,
        "minimum_loss_drop_ratio": args.minimum_loss_drop_ratio,
        "minimum_miou_gain": args.minimum_miou_gain,
    }
    output_path = args.run_dir / "OVERFIT_STATUS.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
