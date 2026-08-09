"""Reproducible, colorblind-safe statistical figures for CROG V3."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .schema import artifact_identity, atomic_write_json


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
    "gray": "#8C8C8C",
}
METHOD_COLORS = {
    "q_only": OKABE_ITO["sky"],
    "q-only": OKABE_ITO["sky"],
    "v2_locked_primary": OKABE_ITO["orange"],
    "v2": OKABE_ITO["orange"],
    "v3": OKABE_ITO["vermillion"],
    "v3_primary": OKABE_ITO["vermillion"],
    "oracle": OKABE_ITO["green"],
    "correct": OKABE_ITO["green"],
}
OUR_COLOR = OKABE_ITO["vermillion"]
BASELINE_COLOR = "#AAB7C4"
FIG_SINGLE = (3.25, 2.5)
FIG_FULL = (6.75, 2.8)


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4,
        }
    )


def _atomic_figure_write(fig: Any, path: Path, *, format_name: str) -> None:
    if path.exists():
        raise FileExistsError(f"immutable figure already exists: {path}")
    buffer = io.BytesIO()
    metadata = {"Creator": "CROG reranking V3 plotting.py"}
    if format_name == "pdf":
        metadata["CreationDate"] = None
    fig.savefig(buffer, format=format_name, dpi=300, metadata=metadata)
    content = buffer.getvalue()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save(fig: Any, output_stem: Path) -> list[str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    if pdf.exists() or png.exists():
        plt.close(fig)
        raise FileExistsError(f"immutable figure target exists: {output_stem}")
    try:
        _atomic_figure_write(fig, pdf, format_name="pdf")
        _atomic_figure_write(fig, png, format_name="png")
    finally:
        plt.close(fig)
    return [str(pdf.resolve()), str(png.resolve())]


def _empty(axis: Any, message: str = "No observations in this analysis group") -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color=OKABE_ITO["gray"])
    axis.set_xticks([])
    axis.set_yticks([])


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _method_color(name: str, index: int = 0) -> str:
    lowered = str(name).lower()
    if lowered in METHOD_COLORS:
        return METHOD_COLORS[lowered]
    if lowered.startswith("v3") or "fcer" in lowered:
        return METHOD_COLORS["v3"]
    palette = [OKABE_ITO["blue"], OKABE_ITO["pink"], OKABE_ITO["green"], OKABE_ITO["gray"]]
    return palette[index % len(palette)]


def plot_reliability(
    reliability: Mapping[str, Sequence[Mapping[str, Any]]], output_stem: str | Path,
) -> list[str]:
    publication_style()
    fig, axis = plt.subplots(figsize=FIG_SINGLE)
    axis.plot([0, 1], [0, 1], color=BASELINE_COLOR, linestyle="--", label="Perfect calibration")
    plotted = 0
    for index, (method, bins) in enumerate(reliability.items()):
        available = [
            value
            for value in bins
            if int(value.get("count", 0)) > 0
            and _finite(value.get("confidence")) is not None
            and _finite(value.get("accuracy")) is not None
        ]
        if not available:
            continue
        axis.plot(
            [float(value["confidence"]) for value in available],
            [float(value["accuracy"]) for value in available],
            marker="o",
            label=str(method),
            color=_method_color(str(method), index),
        )
        plotted += 1
    axis.set(xlabel="Mean predicted probability", ylabel="Observed correctness", xlim=(0, 1), ylim=(0, 1), title="Candidate reliability")
    if plotted:
        axis.legend(loc="upper left")
    else:
        axis.legend(loc="upper left")
        axis.text(0.5, 0.12, "No non-empty calibration bins", ha="center", transform=axis.transAxes, color=OKABE_ITO["gray"])
    return _save(fig, Path(output_stem))


def plot_risk_coverage(
    curves: Mapping[str, Sequence[Mapping[str, Any]]], output_stem: str | Path,
) -> list[str]:
    publication_style()
    fig, axis = plt.subplots(figsize=FIG_SINGLE)
    plotted = 0
    for index, (method, points) in enumerate(curves.items()):
        available = [
            value
            for value in points
            if _finite(value.get("coverage")) is not None and _finite(value.get("risk")) is not None
        ]
        if not available:
            continue
        axis.plot(
            [float(value["coverage"]) for value in available],
            [float(value["risk"]) for value in available],
            label=str(method),
            color=_method_color(str(method), index),
        )
        plotted += 1
    axis.set(xlabel="Coverage", ylabel="Risk (error rate)", xlim=(0, 1), ylim=(0, None), title="Risk–coverage")
    if plotted:
        axis.legend()
    else:
        _empty(axis)
        axis.set_title("Risk–coverage")
    return _save(fig, Path(output_stem))


def plot_ablation(
    rows: Sequence[Mapping[str, Any]], output_stem: str | Path, *, metric: str = "delta_vs_v2",
) -> list[str]:
    publication_style()
    available = [(value, _finite(value.get(metric))) for value in rows]
    available = [(value, score) for value, score in available if score is not None]
    fig, axis = plt.subplots(figsize=(6.75, max(2.8, 0.25 * len(available) + 0.7)))
    if not available:
        _empty(axis, f"No finite {metric} ablation results")
        axis.set_title("Predeclared full-chain ablation")
        return _save(fig, Path(output_stem))
    ordered = sorted(available, key=lambda value: value[1])
    names = [str(value.get("configuration") or value.get("method") or "unnamed") for value, _ in ordered]
    scores = np.asarray([100.0 * score for _, score in ordered])
    best_index = int(np.argmax(scores))
    colors = [OUR_COLOR if index == best_index else (OKABE_ITO["blue"] if value >= 0 else BASELINE_COLOR) for index, value in enumerate(scores)]
    y = np.arange(len(names))
    bars = axis.barh(y, scores, color=colors, height=0.62, edgecolor="white", linewidth=0.4)
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_yticks(y, labels=names)
    axis.set_xlabel("Corrected ΔJ@1 vs V2 (percentage points)")
    axis.set_title("Predeclared full-chain ablation")
    for bar, value in zip(bars, scores, strict=True):
        axis.text(value + (0.02 if value >= 0 else -0.02), bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=7)
    return _save(fig, Path(output_stem))


def plot_method_scores(
    rows: Sequence[Mapping[str, Any]], output_stem: str | Path, *, metric: str = "j_at_1",
) -> list[str]:
    """Plot q/V2/V3 and the fixed correct-candidate Oracle@5 ceiling."""
    publication_style()
    preferred = ("q_only", "v2_locked_primary", "v3_primary")
    by_method = {str(row.get("method")): row for row in rows if _finite(row.get(metric)) is not None}
    ordered = [name for name in preferred if name in by_method]
    ordered.extend(sorted(set(by_method) - set(ordered)))
    labels = ["q-only" if name == "q_only" else "V2" if name == "v2_locked_primary" else "V3" if name.startswith("v3") else name for name in ordered]
    values = [100.0 * float(by_method[name][metric]) for name in ordered]
    oracle = next(
        (_finite(row.get("oracle_at_5")) for row in rows if _finite(row.get("oracle_at_5")) is not None),
        None,
    )
    if oracle is None:
        for row in rows:
            count = _finite(row.get("oracle_correct"))
            sample_count = _finite(row.get("sample_count"))
            if count is not None and sample_count is not None and sample_count > 0:
                oracle = count / sample_count
                break
    if oracle is not None:
        labels.append("Correct in Top-5\n(Oracle)")
        values.append(100.0 * oracle)
        ordered.append("oracle")
    fig, axis = plt.subplots(figsize=FIG_FULL)
    if not values:
        _empty(axis, f"No finite {metric} method results")
    else:
        x = np.arange(len(values))
        bars = axis.bar(x, values, color=[_method_color(name, index) for index, name in enumerate(ordered)], width=0.65)
        axis.set_xticks(x, labels=labels)
        axis.set_ylabel(f"{metric} (%)")
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axis.set_title("Frozen-candidate benchmark correctness")
    return _save(fig, Path(output_stem))


def plot_pairwise_intervals(
    rows: Sequence[Mapping[str, Any]], output_stem: str | Path,
) -> list[str]:
    publication_style()
    parsed = []
    for row in rows:
        effect = _finite(row.get("effect"))
        lower = _finite(row.get("scene_ci_lower", row.get("ci_lower")))
        upper = _finite(row.get("scene_ci_upper", row.get("ci_upper")))
        if effect is not None and lower is not None and upper is not None and lower <= upper:
            label = f"{row.get('method', 'challenger')} vs {row.get('reference', 'reference')}"
            parsed.append((label, effect, lower, upper, str(row.get("method", ""))))
    fig, axis = plt.subplots(figsize=(6.75, max(2.8, 0.35 * len(parsed) + 0.8)))
    if not parsed:
        _empty(axis, "No valid pairwise confidence intervals")
    else:
        y = np.arange(len(parsed))
        center = 100.0 * np.asarray([value[1] for value in parsed])
        lower = 100.0 * np.asarray([value[2] for value in parsed])
        upper = 100.0 * np.asarray([value[3] for value in parsed])
        axis.errorbar(center, y, xerr=np.vstack((center - lower, upper - center)), fmt="none", ecolor=OKABE_ITO["gray"], capsize=3, zorder=1)
        axis.scatter(center, y, color=[_method_color(value[4], index) for index, value in enumerate(parsed)], zorder=2)
        axis.axvline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
        axis.set_yticks(y, labels=[value[0] for value in parsed])
        axis.set_xlabel("Paired ΔJ@1 (percentage points; scene-cluster 95% CI)")
    axis.set_title("Pairwise benchmark effects")
    return _save(fig, Path(output_stem))


def plot_outcome_distribution(
    rows: Sequence[Mapping[str, Any]], output_stem: str | Path,
) -> list[str]:
    publication_style()
    global_rows = [row for row in rows if str(row.get("subgroup_field")) == "__all__"]
    names = ("recovered", "harmful", "stable_correct", "stable_incorrect")
    lookup = {str(row.get("outcome")): int(row.get("count", 0)) for row in global_rows}
    counts = [max(0, lookup.get(name, 0)) for name in names]
    fig, axis = plt.subplots(figsize=FIG_FULL)
    if not global_rows:
        _empty(axis, "No recovered/harmful outcome records")
    else:
        bars = axis.bar(np.arange(4), counts, color=[OKABE_ITO["green"], OKABE_ITO["vermillion"], OKABE_ITO["blue"], OKABE_ITO["gray"]])
        axis.set_xticks(np.arange(4), labels=["Recovered", "Harmful", "Stable correct", "Stable incorrect"])
        axis.set_ylabel("Samples")
        for bar, count in zip(bars, counts, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom")
    axis.set_title("V3 outcomes relative to locked V2")
    return _save(fig, Path(output_stem))


def plot_feature_distributions(
    rows: Sequence[Mapping[str, Any]], output_stem: str | Path,
) -> list[str]:
    """Plot pre-aggregated recovered/harmful feature means without label IO."""
    publication_style()
    parsed = []
    for row in rows:
        recovered = _finite(row.get("recovered_mean"))
        harmful = _finite(row.get("harmful_mean"))
        if recovered is not None and harmful is not None:
            parsed.append((str(row.get("feature_group", row.get("feature", "unnamed"))), recovered, harmful))
    fig, axis = plt.subplots(figsize=(6.75, max(2.8, 0.3 * len(parsed) + 0.8)))
    if not parsed:
        _empty(axis, "No recovered/harmful feature summaries")
    else:
        y = np.arange(len(parsed))
        height = 0.36
        axis.barh(y - height / 2, [row[1] for row in parsed], height=height, color=OKABE_ITO["green"], label="Recovered")
        axis.barh(y + height / 2, [row[2] for row in parsed], height=height, color=OKABE_ITO["vermillion"], label="Harmful")
        axis.set_yticks(y, labels=[row[0] for row in parsed])
        axis.set_xlabel("Mean normalized feature value")
        axis.legend()
    axis.set_title("Recovered versus harmful feature distributions")
    return _save(fig, Path(output_stem))


def plot_gate_uncertainty(
    distributions: Mapping[str, Mapping[str, Sequence[float]]], output_stem: str | Path,
) -> list[str]:
    publication_style()
    metrics = list(distributions)
    fig, axes = plt.subplots(1, max(1, len(metrics)), figsize=(max(3.25, 3.25 * max(1, len(metrics))), 2.5), squeeze=False)
    if not metrics:
        _empty(axes[0, 0], "No gate/uncertainty observations")
        axes[0, 0].set_title("Gate and uncertainty")
    for index, metric in enumerate(metrics):
        axis = axes[0, index]
        recovered = np.asarray(distributions[metric].get("recovered", ()), dtype=float)
        harmful = np.asarray(distributions[metric].get("harmful", ()), dtype=float)
        recovered = recovered[np.isfinite(recovered)]
        harmful = harmful[np.isfinite(harmful)]
        values = [value for value in (recovered, harmful) if len(value)]
        labels = [label for label, value in (("Recovered", recovered), ("Harmful", harmful)) if len(value)]
        if values:
            box = axis.boxplot(values, labels=labels, patch_artist=True, widths=0.55)
            colors = [OKABE_ITO["green"] if label == "Recovered" else OKABE_ITO["vermillion"] for label in labels]
            for patch, color in zip(box["boxes"], colors, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
        else:
            _empty(axis)
        axis.set_title(str(metric).replace("_", " ").title())
    return _save(fig, Path(output_stem))


def plot_switch_precision_coverage(
    points: Sequence[Mapping[str, Any]], output_stem: str | Path,
) -> list[str]:
    publication_style()
    parsed = [
        (_finite(row.get("switch_coverage", row.get("coverage"))), _finite(row.get("outcome_changing_precision", row.get("precision"))), str(row.get("method", "V3")))
        for row in points
    ]
    parsed = [value for value in parsed if value[0] is not None and value[1] is not None]
    fig, axis = plt.subplots(figsize=FIG_SINGLE)
    if not parsed:
        _empty(axis, "No defined switch precision values")
    else:
        for index, (coverage, precision, method) in enumerate(parsed):
            axis.scatter(coverage, precision, color=_method_color(method, index), label=method)
        if len({value[2] for value in parsed}) > 1:
            axis.legend()
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Switch coverage", ylabel="Outcome-changing precision")
    axis.set_title("Switch precision versus coverage")
    return _save(fig, Path(output_stem))


def build_statistical_figures(
    output_dir: str | Path,
    *,
    result_rows: Sequence[Mapping[str, Any]] | None = None,
    pairwise_rows: Sequence[Mapping[str, Any]] | None = None,
    reliability: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    risk_coverage: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ablation_rows: Sequence[Mapping[str, Any]] | None = None,
    outcome_rows: Sequence[Mapping[str, Any]] | None = None,
    feature_distribution_rows: Sequence[Mapping[str, Any]] | None = None,
    gate_uncertainty: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    switch_precision_coverage: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reusable ``build-report`` plotting backend using precomputed metrics."""
    output = Path(output_dir)
    manifest_path = output / "figure_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable figure manifest exists: {manifest_path}")
    generated: dict[str, list[str]] = {}
    jobs = (
        ("method_scores", result_rows, plot_method_scores),
        ("pairwise_intervals", pairwise_rows, plot_pairwise_intervals),
        ("reliability", reliability, plot_reliability),
        ("risk_coverage", risk_coverage, plot_risk_coverage),
        ("ablation", ablation_rows, plot_ablation),
        ("outcomes", outcome_rows, plot_outcome_distribution),
        ("feature_distributions", feature_distribution_rows, plot_feature_distributions),
        ("gate_uncertainty", gate_uncertainty, plot_gate_uncertainty),
        ("switch_precision_coverage", switch_precision_coverage, plot_switch_precision_coverage),
    )
    for name, payload, callback in jobs:
        if payload is not None:
            generated[name] = callback(payload, output / f"fig_{name}")
    artifacts = {
        name: [artifact_identity(path) for path in paths]
        for name, paths in generated.items()
    }
    manifest = {
        "schema_version": "3.0.0",
        "kind": "v3_statistical_figures",
        "status": "complete",
        "palette": "Okabe-Ito",
        "png_dpi": 300,
        "formats": ["pdf", "png"],
        "figures": artifacts,
        "labels_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    return manifest | {"manifest": artifact_identity(manifest_path)}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_result_figures(run_dir: str | Path) -> dict[str, Any]:
    """Compatibility wrapper that discovers conventional V3 result artifacts."""
    run = Path(run_dir)
    selection_path = run / "selection/selection_summary.json"
    calibration_path = run / "evaluation/calibration_curves.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    return build_statistical_figures(
        run / "figures",
        result_rows=_read_csv(run / "results_test.csv") if (run / "results_test.csv").exists() else None,
        pairwise_rows=_read_csv(run / "pairwise_statistics.csv") if (run / "pairwise_statistics.csv").exists() else None,
        reliability=calibration.get("reliability") if calibration else None,
        risk_coverage=calibration.get("risk_coverage") if calibration else None,
        ablation_rows=selection.get("results") if selection else None,
    )


__all__ = (
    "METHOD_COLORS",
    "OKABE_ITO",
    "build_statistical_figures",
    "generate_result_figures",
    "plot_ablation",
    "plot_feature_distributions",
    "plot_gate_uncertainty",
    "plot_method_scores",
    "plot_outcome_distribution",
    "plot_pairwise_intervals",
    "plot_reliability",
    "plot_risk_coverage",
    "plot_switch_precision_coverage",
    "publication_style",
)
