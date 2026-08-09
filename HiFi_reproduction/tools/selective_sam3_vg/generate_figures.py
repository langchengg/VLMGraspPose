#!/usr/bin/env python3
"""Generate the eleven required publication-style result figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/selective_sam3_vg/report/figures"
COLORS = {"paper": "#4C78A8", "baseline": "#6B7280", "hybrid": "#0F9D8A", "oracle": "#F59E0B", "harm": "#C44E52"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
            "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 160,
            "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def _group_plot(path: Path, title: str, stem: str) -> None:
    frame = pd.read_csv(path)
    frame = frame.sort_values("group")
    x = np.arange(len(frame))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(5.2, len(frame) * 0.55), 3.4))
    ax.bar(x - width / 2, 100 * frame["baseline_mean_iou"], width, color=COLORS["baseline"], label="Baseline")
    ax.bar(x + width / 2, 100 * frame["hybrid_mean_iou"], width, color=COLORS["hybrid"], label="Selective SAM 3")
    ax.set_xticks(x, frame["group"], rotation=35, ha="right")
    ax.set_ylabel("Mean IoU (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, stem)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    metrics = json.loads((ROOT / "outputs/selective_sam3_vg/report/formal_metrics.json").read_text())
    formal = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/report/formal_per_sample_metrics.parquet")
    baseline, hybrid, paper = metrics["baseline"], metrics["locked_selective_sam3"], metrics["paper_numeric_reference"]
    oracle = json.loads((ROOT / "outputs/selective_sam3_vg/oracle/ORACLE_GT_SELECTED_summary.json").read_text())

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    names = ["Paper numeric\nreference", "Local repeated-FiLM", "Locked selective\nSAM 3"]
    values = [paper["mean_iou"], baseline["mean_iou"], hybrid["mean_iou"]]
    bars = ax.bar(names, np.asarray(values) * 100, color=[COLORS["paper"], COLORS["baseline"], COLORS["hybrid"]])
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_ylabel("Mean IoU (%)")
    ax.set_title("OCID-VLG visual-grounding mean IoU")
    ax.set_ylim(0, max(values) * 112)
    _save(fig, "01_paper_baseline_selective_miou")

    thresholds = [50, 60, 70, 80, 90]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, source, color, marker in (
        ("Paper numeric reference", paper, COLORS["paper"], "o"),
        ("Local repeated-FiLM", baseline, COLORS["baseline"], "s"),
        ("Locked selective SAM 3", hybrid, COLORS["hybrid"], "^"),
    ):
        ax.plot(thresholds, [100 * source[f"p_at_{x}"] for x in thresholds], marker=marker, color=color, label=label)
    ax.set_xlabel("IoU threshold (%)")
    ax.set_ylabel("Precision at threshold (%)")
    ax.set_xticks(thresholds)
    ax.set_title("Threshold precision profile")
    ax.legend(frameon=False)
    _save(fig, "02_paper_baseline_selective_precision_curve")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(formal["baseline_iou"], bins=bins, histtype="step", linewidth=1.8, color=COLORS["baseline"], label="Baseline")
    ax.hist(formal["hybrid_iou"], bins=bins, histtype="step", linewidth=1.8, color=COLORS["hybrid"], label="Selective SAM 3")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Samples")
    ax.set_title("IoU distribution before and after")
    ax.legend(frameon=False)
    _save(fig, "03_iou_distribution_before_after")

    transition = pd.DataFrame(metrics["threshold_transitions"])
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    colors = [COLORS["hybrid"] if value >= 0 else COLORS["harm"] for value in transition["net"]]
    ax.bar([f"P@{int(x*100)}" for x in transition["threshold"]], transition["net"], color=colors)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("Net threshold transitions")
    ax.set_title("Recovered minus harmed samples")
    _save(fig, "04_threshold_transition_waterfall")

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    x = np.arange(len(transition))
    width = 0.36
    ax.bar(x - width / 2, transition["recovered"], width, color=COLORS["hybrid"], label="Recovered")
    ax.bar(x + width / 2, transition["harmed"], width, color=COLORS["harm"], label="Harmed")
    ax.set_xticks(x, [f"P@{int(t*100)}" for t in transition["threshold"]])
    ax.set_ylabel("Samples")
    ax.set_title("Outcome-changing samples")
    ax.legend(frameon=False)
    _save(fig, "05_recovered_versus_harmed")

    trigger = pd.read_parquet(ROOT / "artifacts/selective_sam3_vg/validation_training/trigger_oof_predictions.parquet")
    precision, recall, _ = precision_recall_curve(trigger["boundary_recoverable_label"], trigger["trigger_score_oof"])
    trigger_calibration = json.loads(
        (ROOT / "artifacts/selective_sam3_vg/locked_trigger/calibration.json").read_text()
    )["selected"]["calibration"]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(recall, precision, color=COLORS["hybrid"])
    ax.scatter(
        [trigger_calibration["recall"]],
        [trigger_calibration["precision"]],
        color="#111111", s=24, label="Locked threshold (validation precision)", zorder=3,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation trigger precision–recall")
    ax.legend(frameon=False)
    _save(fig, "06_trigger_precision_recall")

    selector = pd.read_parquet(ROOT / "artifacts/selective_sam3_vg/validation_training/selector_oof_predictions.parquet")
    coarse = selector[selector["candidate_id"] == "coarse_0"].set_index("sample_id")
    sam = selector[selector["candidate_id"] != "coarse_0"].copy()
    sam["predicted_gain"] = sam["selector_regression_oof"] - sam["sample_id"].map(coarse["selector_regression_oof"])
    sam["true_gain"] = sam["candidate_iou"] - sam["sample_id"].map(coarse["candidate_iou"])
    fig, ax = plt.subplots(figsize=(4.7, 3.7))
    plot = ax.hexbin(sam["predicted_gain"], sam["true_gain"], gridsize=45, mincnt=1, cmap="viridis")
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.axvline(0, color="#777777", linewidth=0.7)
    ax.set_xlabel("OOF predicted gain")
    ax.set_ylabel("True validation gain")
    ax.set_title("Candidate gain calibration")
    fig.colorbar(plot, ax=ax, label="Candidates")
    _save(fig, "07_predicted_vs_true_validation_gain")

    grouped = ROOT / "outputs/selective_sam3_vg/report/grouped"
    _group_plot(grouped / "results_by_baseline_iou_bin.csv", "Results by baseline IoU bin", "08_results_by_baseline_iou_bin")
    _group_plot(grouped / "results_by_query_type.csv", "Results by query type", "09_results_by_query_type")
    _group_plot(grouped / "results_by_target_area_bin.csv", "Results by target-mask size", "10_results_by_mask_size")

    oracle_metrics = oracle["ORACLE_GT_SELECTED"]
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    names = [
        "Baseline",
        "Formal\nselective",
        "GT-selected oracle\n(non-deployable)",
        "Paper numeric\nreference",
    ]
    values = [baseline["mean_iou"], hybrid["mean_iou"], oracle_metrics["mean_iou"], paper["mean_iou"]]
    bars = ax.bar(names, np.asarray(values) * 100, color=[COLORS["baseline"], COLORS["hybrid"], COLORS["oracle"], COLORS["paper"]])
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_ylabel("Mean IoU (%)")
    ax.set_title("Formal result and diagnostic oracle ceiling")
    _save(fig, "11_oracle_ceiling_versus_formal")
    print(json.dumps({"status": "COMPLETED", "figure_stems": 11, "formats": ["png", "pdf", "svg"]}, indent=2))


if __name__ == "__main__":
    main()
