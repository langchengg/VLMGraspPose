#!/usr/bin/env python3
"""Evaluate formal masks only after their immutable output lock exists."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.io import load_binary_mask, load_probability, resize_binary_mask, sha256_file  # noqa: E402
from segmentation.selective_sam3_vg.metrics import binary_mask_metrics, summarize_ious  # noqa: E402


PAPER = {"mean_iou": 0.8826, "p_at_50": 0.9268, "p_at_60": 0.9213, "p_at_70": 0.9153, "p_at_80": 0.8969, "p_at_90": 0.8321}
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def main() -> None:
    output_lock_path = ROOT / "artifacts/selective_sam3_vg/formal_output_lock.json"
    output_lock = json.loads(output_lock_path.read_text(encoding="utf-8"))
    if output_lock.get("status") != "LOCKED_BEFORE_GT_EVALUATION":
        raise RuntimeError("formal output masks were not locked before evaluation")
    output_manifest_path = Path(output_lock["output_manifest_path"])
    if sha256_file(output_manifest_path) != output_lock["output_manifest_sha256"]:
        raise RuntimeError("formal output manifest changed after locking")
    outputs = pd.read_parquet(output_manifest_path).sort_values("sample_index")
    baseline = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/baseline_manifest.parquet").sort_values("sample_index")
    if len(outputs) != 7675 or not outputs["sample_id"].tolist() == baseline["sample_id"].tolist():
        raise RuntimeError("formal outputs and authoritative baseline are misaligned")
    rows = []
    for number, (output, source) in enumerate(zip(outputs.itertuples(index=False), baseline.itertuples(index=False), strict=True), start=1):
        if sha256_file(output.final_mask_path) != output.final_mask_sha256:
            raise RuntimeError(f"final mask changed after output lock: {output.sample_id}")
        final_native = load_binary_mask(output.final_mask_path, expected_shape=(480, 640))
        # Preserve the frozen evaluator contract exactly for every HiFi
        # fallback. SAM masks originate at native resolution and are mapped to
        # 352x352 only for the common evaluator.
        final = (
            load_probability(source.probability_path) >= 0.5
            if output.selected_source == "hifics"
            else resize_binary_mask(final_native, (352, 352))
        )
        gt = load_binary_mask(source.gt_mask_path, expected_shape=(352, 352))
        metrics = binary_mask_metrics(final, gt, boundary_tolerance_px=2)
        provenance = json.loads((Path(output.final_mask_path).parent / "provenance.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "sample_index": int(source.sample_index), "sample_id": source.sample_id,
                "scene_id": source.scene_id, "query": source.query, "query_type": source.query_type,
                "target_category": source.target_category, "baseline_iou": float(source.original_iou),
                "hybrid_iou": float(metrics["evaluator_iou_float32"]),
                "delta_iou": float(metrics["evaluator_iou_float32"] - source.original_iou),
                "selected_source": output.selected_source, "sam_invoked": bool(output.sam_invoked),
                "trigger_score": provenance["trigger_score"], "selector_score": provenance["selector_score"],
                "predicted_gain": provenance["predicted_gain"], "fallback_reason": provenance["fallback_reason"],
                **{key: value for key, value in metrics.items() if key not in {"iou", "evaluator_iou_float32"}},
            }
        )
        if number % 500 == 0:
            print(f"evaluated {number}/7675 locked formal masks", flush=True)
    per_sample = pd.DataFrame(rows)
    baseline_values = per_sample["baseline_iou"].to_numpy(dtype=np.float64)
    hybrid_values = per_sample["hybrid_iou"].to_numpy(dtype=np.float64)
    baseline_summary = summarize_ious(baseline_values)
    hybrid_summary = summarize_ious(hybrid_values)
    baseline_diagnostics = pd.read_parquet(
        ROOT / "outputs/selective_sam3_vg/diagnostics/baseline_formal_test_per_sample_metrics.parquet"
    ).sort_values("sample_index")
    for key in ("dice", "mask_precision", "mask_recall", "boundary_fscore"):
        baseline_summary[f"mean_{key}"] = float(baseline_diagnostics[key].mean())
        hybrid_summary[f"mean_{key}"] = float(per_sample[key].mean())
    baseline_summary["low_precision_count_lt_0_5"] = int(np.count_nonzero(baseline_diagnostics["mask_precision"] < 0.5))
    baseline_summary["low_recall_count_lt_0_5"] = int(np.count_nonzero(baseline_diagnostics["mask_recall"] < 0.5))
    hybrid_summary["low_precision_count_lt_0_5"] = int(np.count_nonzero(per_sample["mask_precision"] < 0.5))
    hybrid_summary["low_recall_count_lt_0_5"] = int(np.count_nonzero(per_sample["mask_recall"] < 0.5))
    transitions = []
    for threshold in THRESHOLDS:
        original_success = baseline_values > threshold
        hybrid_success = hybrid_values > threshold
        recovered = int(np.count_nonzero(~original_success & hybrid_success))
        harmed = int(np.count_nonzero(original_success & ~hybrid_success))
        transitions.append(
            {
                "threshold": threshold,
                "baseline_successes": int(np.count_nonzero(original_success)),
                "hybrid_successes": int(np.count_nonzero(hybrid_success)),
                "denominator": len(per_sample), "recovered": recovered, "harmed": harmed,
                "net": recovered - harmed,
                "outcome_changing_precision": recovered / (recovered + harmed) if recovered + harmed else 1.0,
            }
        )
    accepted = per_sample["selected_source"] == "sam3"
    harmful = accepted & (per_sample["delta_iou"] < 0.0)
    improved = accepted & (per_sample["delta_iou"] > 0.0)
    gap_closed = {
        "mean_iou": (hybrid_summary["mean_iou"] - baseline_summary["mean_iou"]) / (PAPER["mean_iou"] - baseline_summary["mean_iou"]),
    }
    for threshold in THRESHOLDS:
        key = f"p_at_{int(threshold*100)}"
        gap_closed[key] = (hybrid_summary[key] - baseline_summary[key]) / (PAPER[key] - baseline_summary[key])
    summary: dict[str, Any] = {
        "status": "COMPLETED",
        "evaluation_started_after_output_lock": True,
        "paper_values_label": "paper numeric reference",
        "paper_numeric_reference": PAPER,
        "baseline": baseline_summary,
        "locked_selective_sam3": hybrid_summary,
        "selective_counts": {
            "trigger_count": int(per_sample["sam_invoked"].sum()),
            "sam_invocation_rate": float(per_sample["sam_invoked"].mean()),
            "sam_acceptance_count": int(accepted.sum()),
            "sam_acceptance_rate": float(accepted.mean()),
            "hifi_fallback_count": int((~accepted).sum()),
            "accepted_sam_improvement_count": int(improved.sum()),
            "accepted_sam_degradation_count": int(harmful.sum()),
            "mean_gain_on_accepted_samples": float(per_sample.loc[accepted, "delta_iou"].mean()) if accepted.any() else 0.0,
            "mean_loss_on_harmful_samples": float(per_sample.loc[harmful, "delta_iou"].mean()) if harmful.any() else 0.0,
        },
        "threshold_transitions": transitions,
        "fraction_of_paper_gap_closed": gap_closed,
    }
    report_root = ROOT / "outputs/selective_sam3_vg/report"
    report_root.mkdir(parents=True, exist_ok=True)
    per_sample.to_parquet(report_root / "formal_per_sample_metrics.parquet", index=False)
    per_sample.to_csv(report_root / "formal_per_sample_metrics.csv", index=False)
    pd.DataFrame(transitions).to_csv(report_root / "threshold_transitions.csv", index=False)
    (report_root / "formal_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
