from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "baseline": "#B0BEC5", "safe": "#E76F51", "recovered": "#009E73",
    "harmful": "#D55E00", "net": "#0072B2", "neutral": "#8C8C8C",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .15, "grid.linestyle": "-",
        "lines.linewidth": 1.8, "lines.markersize": 4,
    })


def _save(fig: plt.Figure, directory: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.png", dpi=300)
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def generate_calibration_plots(run_dir: str | Path) -> list[str]:
    _style(); root = Path(run_dir); directory = root / "plots"; directory.mkdir(parents=True, exist_ok=True)
    sweep = pq.read_table(root / "calibration/local_only_threshold_sweep.parquet").to_pylist()
    oof = pq.read_table(root / "calibration/local_only_oof_pairs.parquet").to_pylist()
    taus = sorted({float(row["tau"]) for row in sweep})
    best = [max((row for row in sweep if float(row["tau"]) == tau), key=lambda row: (row["utility"], -row["harmful"], -row["switches"])) for tau in taus]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(taus, [row["net"] for row in best], color=COLORS["net"], marker="o", label="Corrected Net")
    ax.plot(taus, [row["utility"] for row in best], color=COLORS["safe"], marker="s", label="Recovered − 2×Harmful")
    ax.axhline(0, color="#444", linewidth=.8); ax.set(xlabel="Benefit threshold τ", ylabel="Count", title="Calibration threshold utility envelope")
    ax.legend(); _save(fig, directory, "threshold_net_curve")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(taus, [row["recovered"] for row in best], color=COLORS["recovered"], marker="o", label="Recovered")
    ax.plot(taus, [row["harmful"] for row in best], color=COLORS["harmful"], marker="s", label="Harmful")
    ax.set(xlabel="Benefit threshold τ", ylabel="Outcome-changing samples", title="Recovered versus harmful switches")
    ax.legend(); _save(fig, directory, "recovered_harmful_curve")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(taus, [100 * row["switch_rate"] for row in best], color=COLORS["safe"], marker="o")
    ax.axhline(3.667, color=COLORS["neutral"], linestyle="--", label="Calibration recoverable prevalence")
    ax.set(xlabel="Benefit threshold τ", ylabel="Switch rate (%)", title="Safe-gate switch exposure")
    ax.legend(); _save(fig, directory, "switch_rate_curve")

    scores = np.asarray([float(row["p_benefit"]) for row in oof]); labels = np.asarray([not row["baseline_correct"] and row["challenger_correct"] for row in oof], dtype=float)
    bins = np.linspace(0, 1, 11); centers = []; observed = []; counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if mask.any():
            centers.append(float(scores[mask].mean())); observed.append(float(labels[mask].mean())); counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(3.25, 3.0))
    ax.plot([0, 1], [0, 1], color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.scatter(centers, observed, s=np.sqrt(counts) * 5, color=COLORS["safe"], edgecolor="white", linewidth=.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Calibrated $p_{benefit}$", ylabel="Observed beneficial rate", title="OOF calibration")
    _save(fig, directory, "calibration_curve")

    validation = __import__("json").loads((root / "validation/VALIDATION_RESULTS.json").read_text())
    baseline = 100 * validation["corrected"]["baseline_j1"]; final = 100 * validation["corrected"]["final_j1"]
    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    bars = ax.bar(["q-only", "P6 safe"], [baseline, final], color=[COLORS["baseline"], COLORS["safe"]], width=.55)
    for bar, value in zip(bars, [baseline, final]):
        ax.text(bar.get_x() + bar.get_width()/2, value + .08, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylim(min(baseline, final) - 1, max(baseline, final) + 1); ax.set_ylabel("Corrected J@1 (%)"); ax.set_title("Untouched validation")
    _save(fig, directory, "cohort_results")
    return [str(path) for path in sorted(directory.glob("*.png"))]


def generate_final_plots(run_dir: str | Path) -> list[str]:
    """Render the preregistered final plot set from frozen saved artifacts."""

    _style()
    root = Path(run_dir)
    directory = root / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(
        (root / "calibration/API_SAFE_GATE_CALIBRATION.json").read_text(encoding="utf-8")
    )
    method = "P5_er2_safe"
    selected = calibration["methods"][method]["selected_thresholds"]
    sweep = pq.read_table(root / "calibration/api_safe_gate_threshold_sweep.parquet").to_pylist()
    query_threshold = float(selected.get("query_threshold", 0.0))
    eta = float(selected.get("eta", 0.05))
    rows = sorted(
        (
            row for row in sweep
            if row["method"] == method
            and abs(float(row["query_threshold"]) - query_threshold) < 1e-15
            and abs(float(row["eta"]) - eta) < 1e-12
        ),
        key=lambda row: float(row["tau"]),
    )
    if not rows:
        raise RuntimeError("frozen API calibration curve is empty")
    taus = [float(row["tau"]) for row in rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(taus, [row["net"] for row in rows], color=COLORS["net"], marker="o", label="Weighted Net")
    ax.plot(taus, [row["utility"] for row in rows], color=COLORS["safe"], marker="s", label="Recovered − 2×Harmful")
    ax.axhline(0, color="#444", linewidth=.8)
    ax.set(xlabel="Benefit threshold τ", ylabel="Weighted count", title="Exploratory safe-gate calibration")
    ax.legend(); _save(fig, directory, "threshold_net_curve")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(taus, [row["recovered"] for row in rows], color=COLORS["recovered"], marker="o", label="Recovered")
    ax.plot(taus, [row["harmful"] for row in rows], color=COLORS["harmful"], marker="s", label="Harmful")
    ax.set(xlabel="Benefit threshold τ", ylabel="Weighted outcome count", title="Exploratory recovered versus harmful")
    ax.legend(); _save(fig, directory, "recovered_harmful_curve")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(taus, [100 * float(row["switch_rate"]) for row in rows], color=COLORS["safe"], marker="o")
    ax.set(xlabel="Benefit threshold τ", ylabel="Weighted switch rate (%)", title="Exploratory safe-gate exposure")
    _save(fig, directory, "switch_rate_curve")

    oof = [
        row for row in pq.read_table(root / "calibration/api_safe_gate_oof_pairs.parquet").to_pylist()
        if row["method"] == method
    ]
    scores = np.asarray([float(row["p_benefit"]) for row in oof])
    labels = np.asarray([
        (not bool(row["baseline_correct"])) and bool(row["challenger_correct"])
        for row in oof
    ], dtype=float)
    weights = np.asarray([float(row["sample_weight"]) for row in oof])
    bins = np.linspace(0, max(.10, float(scores.max(initial=0))), 9)
    centers: list[float] = []
    observed: list[float] = []
    sizes: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi if hi < bins[-1] else scores <= hi)
        if mask.any():
            centers.append(float(np.average(scores[mask], weights=weights[mask])))
            observed.append(float(np.average(labels[mask], weights=weights[mask])))
            sizes.append(float(mask.sum()))
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    maximum = max([.1, *centers, *observed])
    ax.plot([0, maximum], [0, maximum], color=COLORS["neutral"], linestyle="--", linewidth=1)
    ax.scatter(centers, observed, s=np.sqrt(sizes) * 8, color=COLORS["safe"], edgecolor="white", linewidth=.5)
    ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="OOF $p_{benefit}$", ylabel="Weighted beneficial rate", title="Calibration diagnostic")
    _save(fig, directory, "calibration_curve")

    p1 = json.loads((root / "p1_direct_diagnostic/P1_DIAGNOSTIC_FREEZE.json").read_text(encoding="utf-8"))
    p3 = json.loads((root / "p3_diagnostic/DIAGNOSTIC_RESULTS.json").read_text(encoding="utf-8"))
    p5 = json.loads((root / "p5_validation/P5_VALIDATION_RESULTS.json").read_text(encoding="utf-8"))
    values = [
        100 * p1["models"]["gemini-robotics-er-2-preview"]["corrected"]["selected_j1"],
        100 * p1["models"]["gemini-3.6-flash"]["corrected"]["selected_j1"],
        100 * p3["models"]["gemini-robotics-er-2-preview"]["corrected_hard_rule"]["final_j1"],
        100 * p3["models"]["gemini-3.6-flash"]["corrected_hard_rule"]["final_j1"],
        100 * p5["methods"]["P5_er2_safe"]["corrected"]["final_j1"],
    ]
    labels_short = ["Direct\nER2", "Direct\nFlash", "Pairwise\nER2", "Pairwise\nFlash", "P5\nq-only"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(labels_short, values, color=[COLORS["harmful"], COLORS["harmful"], COLORS["safe"], COLORS["safe"], COLORS["baseline"]])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .5, f"{value:.1f}", ha="center", fontsize=8)
    ax.set(ylabel="Corrected J@1 (%)", title="Protocol diagnostics (different cohort prevalences)")
    ax.set_ylim(0, max(values) + 10)
    _save(fig, directory, "cohort_results")

    stability = json.loads(
        (root / "diagnostic_expanded/DIAGNOSTIC_RESULTS.json").read_text(encoding="utf-8")
    )["models"]
    variants = list(calibration["diagnostic_stability"]["contract"]["required_variants"])
    x = np.arange(len(variants)); width = .36
    er2 = [100 * stability["gemini-robotics-er-2-preview"]["stability"][variant]["hard_rule_consistency"] for variant in variants]
    flash_rows = [
        stability["gemini-3.6-flash"]["stability"][variant]
        for variant in variants
    ]
    flash_values = [
        100 * float(row["hard_rule_consistency"])
        if row["hard_rule_consistency"] is not None
        and float(row["valid_pair_coverage"]) >= .98
        else 0.0
        for row in flash_rows
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - width/2, er2, width, label="ER2", color=COLORS["safe"])
    ax.bar(x + width/2, flash_values, width, label="Flash", color=COLORS["baseline"])
    for position, row in zip(x + width/2, flash_rows):
        if float(row["valid_pair_coverage"]) < .98:
            ax.text(position, 2, "coverage\nfailed", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.axhline(90, color=COLORS["neutral"], linestyle="--", linewidth=1, label="required")
    ax.set(xticks=x, xticklabels=[v.replace("_", "\n") for v in variants], ylabel="Hard-rule consistency (%)", title="Perturbation stability")
    ax.legend(); _save(fig, directory, "perturbation_stability")

    connection = sqlite3.connect(f"file:{(root / 'pairwise_cache.sqlite').resolve()}?mode=ro", uri=True)
    try:
        attempt_rows = connection.execute(
            "SELECT r.requested_model,a.latency_seconds,a.estimated_cost_usd "
            "FROM attempts a JOIN requests r USING(request_hash) "
            "WHERE a.latency_seconds IS NOT NULL"
        ).fetchall()
    finally:
        connection.close()
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    for model_id, label, color in (
        ("gemini-robotics-er-2-preview", "ER2", COLORS["safe"]),
        ("gemini-3.6-flash", "Flash", COLORS["net"]),
    ):
        latency = [float(row[1]) for row in attempt_rows if row[0] == model_id]
        if latency:
            ax.hist(latency, bins=40, histtype="step", density=True, label=f"{label} (n={len(latency)})", color=color)
    ax.set(xlabel="Attempt latency (s)", ylabel="Density", title="Provider-attempt latency")
    ax.legend(); _save(fig, directory, "latency_distribution")

    actual_by_model = {
        model_id: sum(float(row[2]) for row in attempt_rows if row[0] == model_id)
        for model_id in ("gemini-robotics-er-2-preview", "gemini-3.6-flash")
    }
    formal_samples = int(json.loads((root / "DATA_MANIFEST.json").read_text())["expected_denominator"])
    pair_requests_per_model = 2 * formal_samples
    er2_cap = max([float(row[2]) for row in attempt_rows if row[0] == "gemini-robotics-er-2-preview"] or [1.0])
    flash_cap = max([float(row[2]) for row in attempt_rows if row[0] == "gemini-3.6-flash"] or [.10])
    original_upper = pair_requests_per_model * (er2_cap + flash_cap)
    confirmation_upper = 2 * original_upper
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    cost_values = [sum(actual_by_model.values()), original_upper, confirmation_upper]
    bars = ax.bar(["Observed\nattempt reserve", "Formal original\nupper bound", "With confirmation\nupper bound"], cost_values, color=[COLORS["baseline"], COLORS["safe"], COLORS["harmful"]])
    for bar, value in zip(bars, cost_values):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"${value:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="Conservative USD reserve", title="Cost exposure (formal run not authorized)")
    _save(fig, directory, "cost_projection")
    return [str(path) for path in sorted(directory.glob("*.png"))]
