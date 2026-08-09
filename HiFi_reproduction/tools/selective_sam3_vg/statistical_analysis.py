#!/usr/bin/env python3
"""Wilson, scene-clustered bootstrap, paired differences, and McNemar tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm


ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
SEED = 42
REPLICATES = 10_000


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = float(norm.ppf(0.975))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def main() -> None:
    table = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/report/formal_per_sample_metrics.parquet")
    scenes = sorted(table["scene_id"].unique())
    grouped = list(table.groupby("scene_id", sort=True))
    if len(table) != 7675 or len(scenes) != 325:
        raise RuntimeError("unexpected formal sample/scene count")
    count = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    baseline_sum = np.asarray([group["baseline_iou"].sum() for _, group in grouped], dtype=np.float64)
    hybrid_sum = np.asarray([group["hybrid_iou"].sum() for _, group in grouped], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(grouped), size=(REPLICATES, len(grouped)))
    denominator = count[indices].sum(axis=1)
    baseline_bootstrap = baseline_sum[indices].sum(axis=1) / denominator
    hybrid_bootstrap = hybrid_sum[indices].sum(axis=1) / denominator
    rows = [
        {
            "metric": "mean_iou", "method": "baseline",
            "point_estimate": float(table["baseline_iou"].mean()),
            "scene_clustered_bootstrap_ci_low": float(np.quantile(baseline_bootstrap, 0.025)),
            "scene_clustered_bootstrap_ci_high": float(np.quantile(baseline_bootstrap, 0.975)),
            "paired_difference": None,
        },
        {
            "metric": "mean_iou", "method": "locked_selective_sam3",
            "point_estimate": float(table["hybrid_iou"].mean()),
            "scene_clustered_bootstrap_ci_low": float(np.quantile(hybrid_bootstrap, 0.025)),
            "scene_clustered_bootstrap_ci_high": float(np.quantile(hybrid_bootstrap, 0.975)),
            "paired_difference": float((table["hybrid_iou"] - table["baseline_iou"]).mean()),
            "paired_difference_ci_low": float(np.quantile(hybrid_bootstrap - baseline_bootstrap, 0.025)),
            "paired_difference_ci_high": float(np.quantile(hybrid_bootstrap - baseline_bootstrap, 0.975)),
        },
    ]
    mcnemar_rows = []
    for threshold in THRESHOLDS:
        percent = int(threshold * 100)
        baseline_success = table["baseline_iou"].to_numpy() > threshold
        hybrid_success = table["hybrid_iou"].to_numpy() > threshold
        baseline_scene_success = np.asarray([np.count_nonzero(group["baseline_iou"].to_numpy() > threshold) for _, group in grouped], dtype=np.float64)
        hybrid_scene_success = np.asarray([np.count_nonzero(group["hybrid_iou"].to_numpy() > threshold) for _, group in grouped], dtype=np.float64)
        baseline_dist = baseline_scene_success[indices].sum(axis=1) / denominator
        hybrid_dist = hybrid_scene_success[indices].sum(axis=1) / denominator
        for method, success, distribution in (
            ("baseline", baseline_success, baseline_dist),
            ("locked_selective_sam3", hybrid_success, hybrid_dist),
        ):
            wilson_low, wilson_high = _wilson(int(np.count_nonzero(success)), len(success))
            row = {
                "metric": f"p_at_{percent}", "method": method,
                "numerator": int(np.count_nonzero(success)), "denominator": len(success),
                "point_estimate": float(np.mean(success)),
                "wilson_ci_low": wilson_low, "wilson_ci_high": wilson_high,
                "scene_clustered_bootstrap_ci_low": float(np.quantile(distribution, 0.025)),
                "scene_clustered_bootstrap_ci_high": float(np.quantile(distribution, 0.975)),
            }
            if method == "locked_selective_sam3":
                row.update(
                    {
                        "paired_difference": float(np.mean(hybrid_success) - np.mean(baseline_success)),
                        "paired_difference_ci_low": float(np.quantile(hybrid_dist - baseline_dist, 0.025)),
                        "paired_difference_ci_high": float(np.quantile(hybrid_dist - baseline_dist, 0.975)),
                    }
                )
            rows.append(row)
        recovered = int(np.count_nonzero(~baseline_success & hybrid_success))
        harmed = int(np.count_nonzero(baseline_success & ~hybrid_success))
        discordant = recovered + harmed
        p_value = float(binomtest(min(recovered, harmed), discordant, p=0.5, alternative="two-sided").pvalue) if discordant else 1.0
        mcnemar_rows.append(
            {
                "threshold": threshold, "recovered": recovered, "harmed": harmed,
                "net_change": recovered - harmed, "discordant": discordant,
                "outcome_changing_precision": recovered / discordant if discordant else 1.0,
                "exact_mcnemar_p_value": p_value,
            }
        )
    report = ROOT / "outputs/selective_sam3_vg/report"
    pd.DataFrame(rows).to_csv(report / "confidence_intervals.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(report / "mcnemar_tests.csv", index=False)
    summary = {
        "status": "COMPLETED", "seed": SEED, "bootstrap_replicates": REPLICATES,
        "cluster_unit": "scene_id", "scene_count": len(scenes),
        "confidence_intervals": rows, "mcnemar": mcnemar_rows,
    }
    (report / "statistical_analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "rows": len(rows), "mcnemar_tests": len(mcnemar_rows)}, indent=2))


if __name__ == "__main__":
    main()
