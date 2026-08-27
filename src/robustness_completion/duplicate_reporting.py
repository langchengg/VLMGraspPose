"""Paper-ready figures and table for the locked RGB-D duplicate audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    atomic_json,
    atomic_text,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)
from .duplicate_map import _verify_locked_outputs


ROUTE_COLOURS = {
    "CROG": "#0072B2",
    "G1": "#E69F00",
    "C1": "#009E73",
    "D1": "#CC79A7",
}
ROUTE_LABELS = {"CROG": "CROG", "G1": "G1", "C1": "C1", "D1": "D1 (retrospective)"}


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for extension in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        paths[extension] = sha256_file(path)
    plt.close(fig)
    return paths


def _route_specific(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["analysis_set"] == "route_specific"].copy()


def _plot_full_filtered(metrics: pd.DataFrame, output: Path) -> dict[str, str]:
    selected = _route_specific(metrics)
    selected = selected[selected["subset"].isin(["full", "strict_excluded"])]
    routes = ["CROG", "G1", "C1", "D1"]
    x = np.arange(len(routes), dtype=float)
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.1, 3.8))
    for offset, subset, hatch in (
        (-width / 2, "full", ""),
        (width / 2, "strict_excluded", "///"),
    ):
        values = [
            float(selected[(selected["route"] == route) & (selected["subset"] == subset)]["raw_gain_pp"].iloc[0])
            for route in routes
        ]
        axis.bar(
            x + offset,
            values,
            width,
            facecolor=[ROUTE_COLOURS[route] for route in routes],
            edgecolor="black",
            linewidth=0.65,
            hatch=hatch,
            alpha=0.92,
            label="Full formal set" if subset == "full" else "Strict-excluded",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, [ROUTE_LABELS[route] for route in routes])
    axis.set_ylabel("Raw reranking gain (pp)")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.set_title("Full versus preregistered strict-excluded paired gain")
    fig.tight_layout()
    return _save_figure(fig, output / "near_duplicate_full_vs_filtered")


def _plot_gain_shift(metrics: pd.DataFrame, output: Path) -> dict[str, str]:
    selected = _route_specific(metrics)
    selected = selected[selected["subset"] == "strict_excluded"].set_index("route")
    routes = ["CROG", "G1", "C1", "D1"]
    estimates = np.asarray([selected.loc[route, "raw_gain_shift_pp"] for route in routes], float)
    lower = np.asarray([selected.loc[route, "raw_gain_shift_ci_low_pp"] for route in routes], float)
    upper = np.asarray([selected.loc[route, "raw_gain_shift_ci_high_pp"] for route in routes], float)
    y = np.arange(len(routes))
    fig, axis = plt.subplots(figsize=(6.7, 3.5))
    axis.errorbar(
        estimates,
        y,
        xerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.0,
    )
    for index, route in enumerate(routes):
        axis.scatter(estimates[index], y[index], color=ROUTE_COLOURS[route], s=48, zorder=3)
    axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
    axis.set_yticks(y, [ROUTE_LABELS[route] for route in routes])
    axis.invert_yaxis()
    axis.set_xlabel("Strict-excluded minus full raw gain (pp)")
    axis.set_title("Sequence-cluster bootstrap gain shift (95% CI)")
    if np.allclose(lower, upper) and np.allclose(estimates, 0):
        axis.set_xlim(-0.1, 0.1)
        axis.text(
            0.02,
            0.96,
            "No observations met the locked strict tier; subsets are identical.",
            transform=axis.transAxes,
            fontsize=8,
            va="top",
        )
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return _save_figure(fig, output / "near_duplicate_gain_shift")


def _plot_pair_distribution(pairs: pd.DataFrame, output: Path) -> dict[str, str]:
    exact = pairs["exact_match"].to_numpy(bool)
    strict_only = pairs["strict_match"].to_numpy(bool) & ~exact
    moderate_only = pairs["moderate_match"].to_numpy(bool) & ~exact & ~strict_only
    rejected = ~(exact | strict_only | moderate_only)
    counts = np.asarray([exact.sum(), strict_only.sum(), moderate_only.sum(), rejected.sum()])
    labels = ["Exact", "Strict only", "Moderate only", "Rejected"]
    colours = ["#009E73", "#0072B2", "#E69F00", "#999999"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    axes[0].bar(np.arange(4), counts, color=colours, edgecolor="black", linewidth=0.6)
    axes[0].set_xticks(np.arange(4), labels, rotation=22, ha="right")
    axes[0].set_ylabel("Retrieved train–test pairs")
    axes[0].set_title("Locked tier membership")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    finite = pairs[np.isfinite(pairs["luminance_ssim"])].copy()
    for same_sequence, colour, label in (
        (True, "#0072B2", "Same sequence"),
        (False, "#D55E00", "Different sequence"),
    ):
        group = finite[finite["same_sequence"] == same_sequence]
        axes[1].scatter(
            group["normalised_rgb_mae"],
            group["luminance_ssim"],
            color=colour,
            s=7,
            alpha=0.32,
            linewidths=0,
            label=label,
        )
    axes[1].axvline(0.015, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.990, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Normalised RGB MAE")
    axes[1].set_ylabel("Luminance SSIM")
    axes[1].set_title("RGB diagnostics (strict bounds dashed)")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    axes[1].grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return _save_figure(fig, output / "duplicate_pair_distribution")


def _latex_table(metrics: pd.DataFrame) -> str:
    selected = _route_specific(metrics)
    selected = selected[selected["subset"].isin(["full", "strict_excluded"])].copy()
    selected["_route_order"] = selected["route"].map(
        {"CROG": 0, "G1": 1, "C1": 2, "D1": 3}
    )
    selected["_subset_order"] = selected["subset"].map(
        {"full": 0, "strict_excluded": 1}
    )
    selected = selected.sort_values(["_route_order", "_subset_order"])
    selected["Route"] = selected["route"].map(ROUTE_LABELS)
    selected["Subset"] = selected["subset"].map(
        {"full": "Full", "strict_excluded": "Strict-excluded"}
    )
    selected["N"] = selected["n_tuples"].astype(int)
    selected["Native J@1"] = selected.apply(
        lambda row: f"{int(row.native_num)}/{int(row.n_tuples)} ({100*row.native_j1:.2f}\\%)",
        axis=1,
    )
    selected["Raw J@1"] = selected.apply(
        lambda row: f"{int(row.raw_num)}/{int(row.n_tuples)} ({100*row.raw_j1:.2f}\\%)",
        axis=1,
    )
    selected["Gain (pp)"] = selected["raw_gain_pp"].map(lambda value: f"{value:+.2f}")
    selected["95\\% CI (pp)"] = selected.apply(
        lambda row: f"[{row.raw_ci_low_pp:.2f}, {row.raw_ci_high_pp:.2f}]", axis=1
    )
    selected["R/H"] = selected.apply(
        lambda row: f"{int(row.raw_recovered)}/{int(row.raw_harmful)}", axis=1
    )
    selected["Shift (pp)"] = selected["raw_gain_shift_pp"].map(lambda value: f"{value:+.2f}")
    table = selected[
        ["Route", "Subset", "N", "Native J@1", "Raw J@1", "Gain (pp)", "95\\% CI (pp)", "R/H", "Shift (pp)"]
    ]
    body = table.to_latex(index=False, escape=False, column_format="llrllllll")
    return (
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\caption{Paired 4-DoF results before and after the preregistered strict RGB-D near-duplicate exclusion. "
        "No test observation met the strict tier, so the two estimands are identical. D1 is retrospective.}\n"
        "\\label{tab:near-duplicate-exclusion}\n"
        + body
        + "\\end{table}\n"
    )


def report_duplicate_results(repo: Path, run_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    repo = repo.resolve()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    root = run_dir / "duplicate_audit"
    lock = json.loads((root / "DUPLICATE_MAP_LOCK.json").read_text(encoding="utf-8"))
    _verify_locked_outputs(root, lock)
    completion = json.loads((run_dir / "duplicate_exclusion" / "completion.json").read_text(encoding="utf-8"))
    for relative, expected in completion["output_sha256"].items():
        if sha256_file(run_dir / "duplicate_exclusion" / relative) != expected:
            raise RuntimeError(f"duplicate exclusion output hash mismatch: {relative}")
    completion_path = run_dir / "duplicate_reporting_completion.json"
    if completion_path.exists() and resume:
        current = json.loads(completion_path.read_text(encoding="utf-8"))
        for relative, expected in current["output_sha256"].items():
            if sha256_file(run_dir / relative) != expected:
                raise RuntimeError(f"duplicate reporting output hash mismatch: {relative}")
        return {"status": "COMPLETE", "resumed": True}

    metrics = pd.read_csv(run_dir / "duplicate_exclusion" / "full_vs_filtered_metrics.csv")
    pairs = pd.read_parquet(root / "all_candidate_pairs.parquet")
    figures = run_dir / "figures"
    manifests = {
        "near_duplicate_full_vs_filtered": _plot_full_filtered(metrics, figures),
        "near_duplicate_gain_shift": _plot_gain_shift(metrics, figures),
        "duplicate_pair_distribution": _plot_pair_distribution(pairs, figures),
    }
    table = run_dir / "tables" / "near_duplicate_exclusion.tex"
    atomic_text(table, _latex_table(metrics))
    relative_names = [
        f"figures/{stem}.{extension}"
        for stem in manifests
        for extension in ("pdf", "svg", "png")
    ] + ["tables/near_duplicate_exclusion.tex"]
    payload = {
        "status": "COMPLETE",
        "source_duplicate_lock_sha256": sha256_file(root / "DUPLICATE_MAP_LOCK.json"),
        "source_metrics_sha256": sha256_file(run_dir / "duplicate_exclusion" / "full_vs_filtered_metrics.csv"),
        "figure_captions": {
            "near_duplicate_full_vs_filtered": "Full and strict-excluded raw paired gains; the locked strict audit excluded zero observations.",
            "near_duplicate_gain_shift": "Sequence-cluster bootstrap shift in raw paired gain after strict exclusion.",
            "duplicate_pair_distribution": "Membership of 6,944 outcome-blind retrieved RGB-D pairs and their RGB diagnostics.",
        },
        "output_sha256": {name: sha256_file(run_dir / name) for name in relative_names},
    }
    atomic_json(completion_path, payload)
    return {"status": "COMPLETE", "resumed": False, "figures": 3, "table": 1}
