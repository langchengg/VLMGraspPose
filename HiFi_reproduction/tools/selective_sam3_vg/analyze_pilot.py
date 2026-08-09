#!/usr/bin/env python3
"""Validation-GT analysis kept strictly separate from pilot inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.io import load_binary_mask, resize_binary_mask  # noqa: E402
from segmentation.selective_sam3_vg.metrics import evaluator_iou_float32  # noqa: E402


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def _feature_value(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.startswith(("[", "{")):
        return json.loads(text)
    try:
        return float(text)
    except ValueError:
        return text


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stratification-manifest",
        type=Path,
        default=ROOT / "outputs/selective_sam3_vg/validation_pilot/pilot_stratification_manifest.parquet",
    )
    parser.add_argument("--input-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-configurations", type=int, default=1500)
    parser.add_argument(
        "--diagnostic-label",
        default="Validation GT-selected upper bound; not deployable.",
    )
    return parser.parse_args()


def _metrics(values: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": int(values.size),
        "mean_iou": float(np.mean(values)),
        "median_iou": float(np.median(values)),
    }
    for threshold in THRESHOLDS:
        successes = int(np.count_nonzero(values > threshold))
        result[f"p_at_{int(threshold * 100)}"] = successes / values.size
        result[f"p_at_{int(threshold * 100)}_numerator"] = successes
        result[f"p_at_{int(threshold * 100)}_denominator"] = int(values.size)
    return result


def main() -> None:
    args = _arguments()
    manifest = pd.read_parquet(args.stratification_manifest)
    if len(manifest) < 300:
        raise RuntimeError("pilot must contain at least 300 validation samples")
    if "iou_band" not in manifest:
        manifest["iou_band"] = pd.cut(
            manifest["iou"],
            bins=[-1.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 1.000001],
            right=False,
            labels=["00_25", "25_50", "50_60", "60_70", "70_80", "80_90", "90_100"],
        ).astype(str)
    if "target_area_band" not in manifest:
        manifest["target_area_band"] = pd.cut(
            manifest["target_area_fraction"],
            bins=[-1.0, 0.005, 0.015, 0.04, 1.0],
            right=False,
            labels=["tiny", "small", "medium", "large"],
        ).astype(str)
    sample_lookup = manifest.set_index("sample_id").to_dict(orient="index")
    configuration_dirs = sorted(
        path.parent
        for root in args.input_root
        for path in root.rglob("status.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETED"
    )
    if len(configuration_dirs) != args.expected_configurations:
        raise RuntimeError(
            f"pilot incomplete: {len(configuration_dirs)} completed configurations, "
            f"expected {args.expected_configurations}"
        )
    rows: list[dict[str, Any]] = []
    for number, directory in enumerate(configuration_dirs, start=1):
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        sample_id = str(provenance["sample_id"])
        family = str(provenance["prompt_family"])
        metadata = sample_lookup[sample_id]
        gt = load_binary_mask(metadata["gt_mask_path"])
        archive = np.load(directory / "all_sam_hypotheses.npz", allow_pickle=False)
        try:
            masks = archive["masks"]
            probabilities = archive["probabilities"]
            qualities = archive["qualities"]
            ids = archive["candidate_ids"].astype(str)
        finally:
            archive.close()
        if (
            masks.ndim != 3
            or probabilities.shape != masks.shape
            or masks.shape[1:] != (480, 640)
            or len(masks) != 3
            or not np.isfinite(probabilities).all()
            or float(probabilities.min()) < 0.0
            or float(probabilities.max()) > 1.0
            or not np.isfinite(qualities).all()
        ):
            raise RuntimeError(f"invalid SAM hypothesis archive: {directory}")
        feature_rows = {
            row["candidate_id"]: row
            for row in csv.DictReader((directory / "candidate_features.csv").open(encoding="utf-8"))
        }
        coarse_iou = float(metadata["iou"])
        rows.append(
            {
                "sample_id": sample_id,
                "scene_id": metadata["scene_id"],
                "query_type": metadata["query_type"],
                "target_category": metadata["target_category"],
                "target_area_band": metadata["target_area_band"],
                "error_type": metadata["error_type"],
                "baseline_iou_band": metadata["iou_band"],
                "prompt_family": family,
                "box_expansion_fraction": float(provenance["box_expansion_fraction"]),
                "candidate_id": "coarse_0",
                "sam_quality": np.nan,
                "candidate_iou": coarse_iou,
                "delta_iou": 0.0,
                "oracle": True,
                **{f"feature_{key}": _feature_value(value) for key, value in feature_rows["coarse_0"].items() if key != "candidate_id"},
            }
        )
        if not (len(masks) == len(probabilities) == len(qualities) == len(ids)):
            raise RuntimeError(f"misaligned hypotheses in {directory}")
        for candidate_id, mask, quality in zip(ids, masks, qualities, strict=True):
            restored = resize_binary_mask(np.asarray(mask, dtype=bool), gt.shape)
            candidate_iou = evaluator_iou_float32(restored, gt)
            rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": metadata["scene_id"],
                    "query_type": metadata["query_type"],
                    "target_category": metadata["target_category"],
                    "target_area_band": metadata["target_area_band"],
                    "error_type": metadata["error_type"],
                    "baseline_iou_band": metadata["iou_band"],
                    "prompt_family": family,
                    "box_expansion_fraction": float(provenance["box_expansion_fraction"]),
                    "candidate_id": candidate_id,
                    "sam_quality": float(quality),
                    "candidate_iou": candidate_iou,
                    "delta_iou": candidate_iou - coarse_iou,
                    "oracle": True,
                    **{f"feature_{key}": _feature_value(value) for key, value in feature_rows[candidate_id].items() if key != "candidate_id"},
                }
            )
        if number % 100 == 0:
            print(f"evaluated {number}/{len(configuration_dirs)} configurations", flush=True)
    candidates = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    grouping = ["prompt_family", "box_expansion_fraction"]
    for keys, group in candidates.groupby(grouping, sort=True):
        family, expansion = keys
        coarse = group[group["candidate_id"] == "coarse_0"].set_index("sample_id")["candidate_iou"]
        sam = group[group["candidate_id"] != "coarse_0"]
        best_sam = sam.groupby("sample_id")["candidate_iou"].max().reindex(coarse.index)
        hybrid = np.maximum(coarse.to_numpy(), best_sam.to_numpy())
        best_sam_values = best_sam.to_numpy()
        delta = best_sam_values - coarse.to_numpy()
        row: dict[str, Any] = {
            "prompt_family": family,
            "box_expansion_fraction": float(expansion),
            "oracle": True,
            "diagnostic_label": args.diagnostic_label,
            "fraction_any_positive_sam_gain": float(np.mean(delta > 0.0)),
            "fraction_gain_ge_0_01": float(np.mean(delta >= 0.01)),
            "fraction_gain_ge_0_02": float(np.mean(delta >= 0.02)),
            "fraction_gain_ge_0_05": float(np.mean(delta >= 0.05)),
            "sam_harmful_rate": float(np.mean(delta < 0.0)),
            **{f"best_sam_{key}": value for key, value in _metrics(best_sam_values).items()},
            **{f"hybrid_oracle_{key}": value for key, value in _metrics(hybrid).items()},
        }
        for threshold in (0.70, 0.80, 0.90):
            row[f"crossing_below_{int(threshold*100)}_to_above"] = int(
                np.count_nonzero((coarse.to_numpy() <= threshold) & (best_sam_values > threshold))
            )
        summary_rows.append(row)
    comparison = pd.DataFrame(summary_rows).sort_values(
        [
            "hybrid_oracle_p_at_90", "hybrid_oracle_p_at_80",
            "hybrid_oracle_p_at_70", "hybrid_oracle_mean_iou",
            "prompt_family", "box_expansion_fraction",
        ],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    comparison.insert(0, "validation_rank", np.arange(1, len(comparison) + 1))
    winner = comparison.iloc[0]
    band_rows: list[dict[str, Any]] = []
    for keys, group in candidates.groupby(grouping, sort=True):
        family, expansion = keys
        coarse_rows = group[group["candidate_id"] == "coarse_0"].set_index("sample_id")
        best_sam = group[group["candidate_id"] != "coarse_0"].groupby("sample_id")["candidate_iou"].max()
        coarse_values = coarse_rows["candidate_iou"]
        analysis_bands = pd.cut(
            coarse_values,
            bins=[-1.0, 0.60, 0.70, 0.80, 0.90, 1.000001],
            right=False,
            labels=["lt_0_60", "0_60_to_0_70", "0_70_to_0_80", "0_80_to_0_90", "ge_0_90"],
        )
        for band in analysis_bands.cat.categories:
            ids = analysis_bands[analysis_bands == band].index
            if not len(ids):
                continue
            original = coarse_values.reindex(ids).to_numpy()
            sam_values = best_sam.reindex(ids).to_numpy()
            delta = sam_values - original
            hybrid_values = np.maximum(original, sam_values)
            band_rows.append(
                {
                    "prompt_family": family,
                    "box_expansion_fraction": float(expansion),
                    "baseline_iou_band": str(band),
                    "samples": len(ids),
                    "fraction_any_positive_sam_gain": float(np.mean(delta > 0.0)),
                    "fraction_gain_ge_0_01": float(np.mean(delta >= 0.01)),
                    "fraction_gain_ge_0_02": float(np.mean(delta >= 0.02)),
                    "fraction_gain_ge_0_05": float(np.mean(delta >= 0.05)),
                    "sam_harmful_rate": float(np.mean(delta < 0.0)),
                    **{f"hybrid_oracle_{key}": value for key, value in _metrics(hybrid_values).items()},
                    "oracle": True,
                    "diagnostic_label": args.diagnostic_label,
                }
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.output_root / "pilot_candidate_evaluation.parquet", index=False)
    candidates.to_csv(args.output_root / "pilot_candidate_evaluation.csv", index=False)
    comparison.to_csv(args.output_root / "prompt_comparison.csv", index=False)
    pd.DataFrame(band_rows).to_csv(args.output_root / "oracle_by_baseline_iou_band.csv", index=False)
    result = {
        "status": "COMPLETED",
        "oracle": True,
        "diagnostic_label": args.diagnostic_label,
        "configuration_count": len(configuration_dirs),
        "candidate_row_count": len(candidates),
        "sample_count": int(manifest["sample_id"].nunique()),
        "selected_prompt_family": winner["prompt_family"],
        "selected_box_expansion_fraction": float(winner["box_expansion_fraction"]),
        "selection_objective": ["P@90", "P@80", "P@70", "mIoU"],
        "selected_metrics": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in winner.to_dict().items()
        },
    }
    (args.output_root / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
