#!/usr/bin/env python3
"""Evaluate frozen selected masks against OCID-VLG GT after inference and selection."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.mask_metrics import evaluate_mask  # noqa: E402
from src.segmentation.sam3_serialization import save_strict_json  # noqa: E402


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refined-root", type=Path, default=REPO_ROOT / "outputs" / "sam3_refined_masks"
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=REPO_ROOT / "runs" / "hifics_ocidvlg_20260711_112921" / "predictions",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "outputs" / "sam3_mask_evaluation"
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "sam3_refinement.yaml"
    )
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mask evaluation: {output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["evaluation"]
    tolerance = int(config["boundary_tolerance_px"])
    margin = float(config["change_margin_iou"])
    rows_manifest = [
        json.loads(line)
        for line in (args.refined_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows: list[dict] = []
    temporary = output.with_name(output.name + ".incomplete")
    temporary.mkdir(parents=True)
    for category in ("improved_cases", "degraded_cases", "fallback_cases"):
        (temporary / category).mkdir()
    try:
        for manifest_row in rows_manifest:
            sample_id = str(manifest_row["sample_id"])
            sample = args.refined_root / sample_id
            metadata = json.loads((sample / "refinement_metadata.json").read_text(encoding="utf-8"))
            coarse = np.asarray(Image.open(sample / "coarse_mask.png")) > 0
            refined = np.asarray(Image.open(sample / "refined_mask.png")) > 0
            # This is the only stage that opens evaluation-only target masks.
            target_path = args.prediction_root / sample_id / "ground_truth_mask_original_resolution.png"
            target = np.asarray(Image.open(target_path)) > 0
            coarse_metrics = evaluate_mask(coarse, target, boundary_tolerance_px=tolerance)
            refined_metrics = evaluate_mask(refined, target, boundary_tolerance_px=tolerance)
            delta = float(refined_metrics["iou"] - coarse_metrics["iou"])
            classification = "improved" if delta > margin else "degraded" if delta < -margin else "unchanged"
            row = {
                "sample_id": sample_id,
                "selected_mask_source": metadata["selected_mask_source"],
                "fallback": bool(metadata["fallback"]),
                "classification": classification,
                "delta_iou": delta,
            }
            for prefix, metrics in (("coarse", coarse_metrics), ("refined", refined_metrics)):
                row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            rows.append(row)
            if classification in {"improved", "degraded"}:
                shutil.copy2(sample / "coarse_vs_refined.png", temporary / f"{classification}_cases" / f"{sample_id}.png")
            if metadata["fallback"]:
                shutil.copy2(sample / "coarse_vs_refined.png", temporary / "fallback_cases" / f"{sample_id}.png")
        if not rows:
            raise ValueError("refinement manifest contains no samples")
        fields = list(rows[0])
        with (temporary / "per_sample_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        aggregate: dict[str, object] = {
            "schema_version": 1,
            "sample_count": len(rows),
            "change_margin_iou": margin,
            "improved_count": sum(row["classification"] == "improved" for row in rows),
            "unchanged_count": sum(row["classification"] == "unchanged" for row in rows),
            "degraded_count": sum(row["classification"] == "degraded" for row in rows),
            "improvement_rate": float(np.mean([row["classification"] == "improved" for row in rows])),
            "degradation_rate": float(np.mean([row["classification"] == "degraded" for row in rows])),
            "fallback_rate": float(np.mean([row["fallback"] for row in rows])),
        }
        for prefix in ("coarse", "refined"):
            ious = [float(row[f"{prefix}_iou"]) for row in rows]
            aggregate[f"{prefix}_mean_iou"] = float(np.mean(ious))
            aggregate[f"{prefix}_median_iou"] = float(statistics.median(ious))
            aggregate[f"{prefix}_mean_dice"] = _mean(rows, f"{prefix}_dice")
            aggregate[f"{prefix}_mean_boundary_fscore"] = _mean(rows, f"{prefix}_boundary_fscore")
            aggregate[f"{prefix}_mean_false_positive_area_px"] = _mean(rows, f"{prefix}_false_positive_area_px")
            aggregate[f"{prefix}_mean_false_negative_area_px"] = _mean(rows, f"{prefix}_false_negative_area_px")
            aggregate[f"{prefix}_mean_connected_component_count"] = _mean(rows, f"{prefix}_connected_component_count")
            aggregate[f"{prefix}_mean_main_component_ratio"] = _mean(rows, f"{prefix}_main_component_ratio")
            for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
                aggregate[f"{prefix}_precision_at_{int(threshold * 100)}"] = float(
                    np.mean(np.asarray(ious) >= threshold)
                )
        save_strict_json(temporary / "summary.json", aggregate)
        with (temporary / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregate))
            writer.writeheader()
            writer.writerow(aggregate)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": "EVALUATED", "samples": len(rows), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

