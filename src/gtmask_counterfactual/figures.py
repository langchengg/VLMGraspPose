"""Publication figures rendered only from the verified GT-mask table bundle."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from .io import artifact_record, atomic_copy, atomic_json, canonical_sha256
from .reporting import TABLE_MANIFEST_RELATIVE_PATH, load_bound_tables


OKABE_ITO = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)

FIGURE_SPECS: dict[str, tuple[str, ...]] = {
    "01_pred_vs_gt_oracle_all": ("pred_vs_gt_paired_metrics.csv",),
    "02_candidate_funnel": ("branch_metrics.csv",),
    "03_bottleneck_shift_before_after_r7": (
        "native_failure_taxonomy.csv",
        "post_r7_bottleneck_taxonomy.csv",
    ),
    "04_grounding_recovery_vs_generator_residual": (
        "pred_vs_gt_paired_metrics.csv",
        "native_failure_taxonomy.csv",
    ),
    "05_first_positive_rank_transition": ("first_positive_rank_transitions.csv",),
    "06_candidate_count_transition": ("candidate_pool_transitions.csv",),
    "07_mask_iou_stratified_recovery": ("stratified_results.csv",),
    "08_query_type_recovery": ("stratified_results.csv",),
    "09_gt_mask_regression_count": ("pred_vs_gt_paired_metrics.csv",),
    "10_route_comparison_forest": ("statistical_tests.csv",),
    "11_stage_replacement_sankey": ("candidate_pool_transitions.csv",),
    "12_d1_top5_top10_allnms_curve": ("branch_metrics.csv",),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "svg.hashsalt": "gtmask-counterfactual-v1",
        }
    )


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"figure input misses {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"figure input {column} must be finite")
    return values


def _nonempty(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"{name} has no observed rows; blank figures are forbidden")
    return frame


def _rate(frame: pd.DataFrame, numerator: str) -> np.ndarray:
    n = _numeric(frame, "N")
    if (n <= 0).any():
        raise ValueError("figure denominators must be positive")
    return _numeric(frame, numerator) / n


def _finish(fig: plt.Figure, title: str) -> None:
    fig.suptitle(title, fontsize=12.5, weight="bold", y=1.01)
    fig.text(
        0.5,
        -0.02,
        "GT mask is oracle diagnostic (not available at deployment).",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )


def _oracle_all(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = _nonempty(tables["pred_vs_gt_paired_metrics.csv"], name="paired metrics")
    routes = data["route"].astype(str).tolist()
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    width = 0.35
    ax.bar(
        x - width / 2,
        _rate(data, "pred_oracle_all"),
        width,
        label="Predicted mask",
        color=OKABE_ITO[1],
    )
    ax.bar(
        x + width / 2,
        _rate(data, "gt_oracle_all"),
        width,
        label="GT mask",
        color=OKABE_ITO[0],
    )
    ax.set_xticks(
        x,
        [
            f"{r}\nN={int(n):,}"
            for r, n in zip(routes, _numeric(data, "N"), strict=True)
        ],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Oracle@All rate")
    ax.legend(ncol=2)
    _finish(fig, "Predicted-mask versus GT-mask Oracle@All")
    return fig


def _candidate_funnel(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = _nonempty(tables["branch_metrics.csv"], name="branch metrics").copy()
    labels = (data["route"].astype(str) + " · " + data["branch"].astype(str)).tolist()
    stages = ("native_correct", "oracle_at_5", "oracle_all")
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    bottom = np.zeros(len(data))
    previous = np.zeros(len(data))
    for index, stage in enumerate(stages):
        current = _rate(data, stage)
        increment = current - previous
        if (increment < -1e-12).any():
            raise ValueError("candidate funnel stages must be nested")
        ax.bar(
            x,
            increment,
            bottom=bottom,
            color=OKABE_ITO[index],
            label=stage.replace("_", " "),
        )
        bottom += increment
        previous = current
    ax.set_xticks(
        x,
        [
            f"{label}\nN={int(n):,}"
            for label, n in zip(labels, _numeric(data, "N"), strict=True)
        ],
        rotation=15,
        ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Denominator fraction")
    ax.legend(ncol=3)
    _finish(fig, "Candidate funnel by route and intervention branch")
    return fig


def _taxonomy_shift(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    native = _nonempty(tables["native_failure_taxonomy.csv"], name="native taxonomy")
    post = _nonempty(tables["post_r7_bottleneck_taxonomy.csv"], name="post-R7 taxonomy")
    routes = sorted(set(native["route"].astype(str)) | set(post["route"].astype(str)))
    fig, axes = plt.subplots(
        1, len(routes), figsize=(max(6.75, 2.5 * len(routes)), 3.4), squeeze=False
    )
    for ax, route in zip(axes[0], routes, strict=True):
        rows = [
            native[native["route"].astype(str).eq(route)],
            post[post["route"].astype(str).eq(route)],
        ]
        for stage_index, frame in enumerate(rows):
            total = float(_numeric(frame, "count").sum())
            if total <= 0:
                raise ValueError(f"taxonomy has zero denominator for {route}")
            bottom = 0.0
            for category_index, record in enumerate(
                frame.sort_values("taxonomy").to_dict("records")
            ):
                height = float(record["count"]) / total
                ax.bar(
                    stage_index,
                    height,
                    bottom=bottom,
                    color=OKABE_ITO[category_index % 7],
                    width=0.65,
                )
                bottom += height
        ax.set_xticks([0, 1], ["Native\nT0–T7", "Post-R7\nR0–R5"])
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{route} (N={int(native[native['route'].astype(str).eq(route)]['N'].max()):,})"
        )
        if ax is axes[0, 0]:
            ax.set_ylabel("Taxonomy fraction")
    _finish(fig, "Operational bottleneck composition before versus after R7")
    return fig


def _recovery_generator(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    paired = _nonempty(tables["pred_vs_gt_paired_metrics.csv"], name="paired metrics")
    taxonomy = tables["native_failure_taxonomy.csv"]
    routes = paired["route"].astype(str).tolist()
    recovered = _numeric(paired, "grounding_recovered")
    generator = np.asarray(
        [
            pd.to_numeric(
                taxonomy.loc[
                    taxonomy["route"].astype(str).eq(route)
                    & taxonomy["taxonomy"].astype(str).str.startswith("T7_"),
                    "count",
                ],
                errors="coerce",
            ).sum()
            for route in routes
        ],
        dtype=float,
    )
    x = np.arange(len(routes))
    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    ax.bar(x - 0.18, recovered, 0.36, color=OKABE_ITO[2], label="Grounding recovered")
    ax.bar(
        x + 0.18,
        generator,
        0.36,
        color=OKABE_ITO[5],
        label="Generator residual under GT",
    )
    ax.set_xticks(
        x,
        [
            f"{r}\nN={int(n):,}"
            for r, n in zip(routes, _numeric(paired, "N"), strict=True)
        ],
    )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Sample count")
    ax.legend()
    _finish(fig, "Grounding recovery versus candidate-generation residual")
    return fig


def _first_rank(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = _nonempty(
        tables["first_positive_rank_transitions.csv"], name="first-rank transitions"
    ).copy()
    routes = sorted(data["route"].astype(str).unique())
    fig, axes = plt.subplots(
        1, len(routes), figsize=(max(6.75, 2.8 * len(routes)), 3.2), squeeze=False
    )
    for ax, route in zip(axes[0], routes, strict=True):
        part = data[data["route"].astype(str).eq(route)]
        pivot = part.pivot_table(
            index="pred_first_positive_rank",
            columns="gt_first_positive_rank",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        matrix = pivot.to_numpy(float)
        image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks(
            range(len(pivot.columns)),
            [str(v) for v in pivot.columns],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
        ax.set_xlabel("GT first-positive rank")
        ax.set_ylabel("Pred first-positive rank")
        ax.set_title(f"{route} · N={int(matrix.sum()):,}")
        fig.colorbar(image, ax=ax, shrink=0.7, label="count")
    _finish(fig, "First-positive-rank transition")
    return fig


def _candidate_transitions(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = _nonempty(
        tables["candidate_pool_transitions.csv"], name="candidate transitions"
    )
    data = data[data["transition_family"].astype(str).eq("candidate_count")]
    _nonempty(data, name="candidate-count transitions")
    pivot = data.pivot_table(
        index="transition", columns="route", values="count", aggfunc="sum", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.35 * len(pivot) + 1.2)))
    left = np.zeros(len(pivot))
    for index, route in enumerate(pivot.columns):
        values = pivot[route].to_numpy(float)
        route_rows = data[data["route"].astype(str).eq(str(route))]
        declared = np.unique(_numeric(route_rows, "N"))
        if len(declared) != 1 or not math.isclose(values.sum(), declared[0]):
            raise ValueError(f"candidate transitions do not sum to N for {route}")
        ax.barh(
            np.arange(len(pivot)),
            values,
            left=left,
            color=OKABE_ITO[index % 7],
            label=f"{route} (N={int(declared[0]):,})",
        )
        left += values
    ax.set_yticks(np.arange(len(pivot)), pivot.index.astype(str))
    ax.set_xlabel("Sample count")
    ax.set_xlim(left=0)
    ax.legend(ncol=min(3, len(pivot.columns)))
    _finish(fig, "Candidate-pool transition counts")
    return fig


def _stratified(
    tables: Mapping[str, pd.DataFrame], keyword: str, title: str
) -> plt.Figure:
    data = tables["stratified_results.csv"]
    selected = data[
        data["stratum_name"].astype(str).str.lower().str.contains(keyword, regex=False)
    ]
    _nonempty(selected, name=title)
    labels = (
        selected["route"].astype(str)
        + " · "
        + selected["branch"].astype(str)
        + " · "
        + selected["stratum_value"].astype(str)
    ).tolist()
    delta = _numeric(selected, "delta")
    fig, ax = plt.subplots(figsize=(max(6.75, 0.65 * len(selected)), 3.2))
    colors = [OKABE_ITO[2] if value >= 0 else OKABE_ITO[5] for value in delta]
    ax.bar(np.arange(len(selected)), delta, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(
        np.arange(len(selected)),
        [
            f"{label}\nN={int(n):,}"
            for label, n in zip(labels, _numeric(selected, "N"), strict=True)
        ],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Paired recovery delta")
    _finish(fig, title)
    return fig


def _regressions(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = _nonempty(tables["pred_vs_gt_paired_metrics.csv"], name="paired metrics")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    values = _numeric(data, "gt_regression")
    ax.bar(np.arange(len(data)), values, color=OKABE_ITO[5])
    ax.set_xticks(
        np.arange(len(data)),
        [
            f"{r}\nN={int(n):,}"
            for r, n in zip(data["route"], _numeric(data, "N"), strict=True)
        ],
    )
    ax.set_ylim(0, max(1.0, float(values.max()) * 1.15))
    ax.set_ylabel("Regression count")
    _finish(fig, "GT-mask regression count")
    return fig


def _forest(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = _nonempty(tables["statistical_tests.csv"], name="statistical tests").copy()
    delta, low, high = (_numeric(data, name) for name in ("delta", "ci_low", "ci_high"))
    if ((low > delta) | (delta > high)).any():
        raise ValueError("forest interval does not contain its point estimate")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(6.75, max(3.0, 0.38 * len(data) + 1.3)))
    ax.errorbar(
        delta,
        y,
        xerr=np.vstack([delta - low, high - delta]),
        fmt="o",
        color=OKABE_ITO[4],
        ecolor="#555555",
        capsize=3,
    )
    ax.axvline(0, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_yticks(
        y,
        [
            f"{r} · {m} · N={int(n):,}"
            for r, m, n in zip(
                data["route"], data["metric"], _numeric(data, "N"), strict=True
            )
        ],
    )
    ax.set_xlabel("Paired delta (95% cluster interval)")
    ax.invert_yaxis()
    _finish(fig, "Route comparison forest plot")
    return fig


def _sankey(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = _nonempty(tables["candidate_pool_transitions.csv"], name="stage transitions")
    data = data[data["transition_family"].astype(str).eq("all_oracle")]
    _nonempty(data, name="all-oracle stage transitions")
    routes = sorted(data["route"].astype(str).unique())
    fig, axes = plt.subplots(
        len(routes), 1, figsize=(7.2, max(3.0, 1.8 * len(routes))), squeeze=False
    )
    for ax, route in zip(axes[:, 0], routes, strict=True):
        part = data[data["route"].astype(str).eq(route)].sort_values("transition")
        values = _numeric(part, "count")
        total = values.sum()
        if total <= 0:
            raise ValueError(f"Sankey route {route} has zero observed transitions")
        declared = np.unique(_numeric(part, "N"))
        if len(declared) != 1 or not math.isclose(total, declared[0]):
            raise ValueError(f"Sankey route {route} transitions do not sum to N")
        lower = 0.0
        for index, (label, value) in enumerate(
            zip(part["transition"].astype(str), values, strict=True)
        ):
            height = value / total
            color = OKABE_ITO[index % 7]
            ax.bar(0, height, bottom=lower, width=0.28, color=color)
            ax.bar(1, height, bottom=lower, width=0.28, color=color, alpha=0.75)
            polygon = Polygon(
                [
                    (0.14, lower),
                    (0.86, lower),
                    (0.86, lower + height),
                    (0.14, lower + height),
                ],
                color=color,
                alpha=0.20,
                linewidth=0,
            )
            ax.add_patch(polygon)
            if height >= 0.035:
                ax.text(
                    0.5,
                    lower + height / 2,
                    f"{label}: {int(value):,}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
            lower += height
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1], ["Predicted-mask stage", "GT-mask stage"])
        ax.set_ylabel(f"{route}\nN={int(total):,}")
    _finish(fig, "Stage-replacement transition flow")
    return fig


def _d1_curve(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    data = tables["branch_metrics.csv"]
    selected = data[data["route"].astype(str).str.upper().eq("D1")].copy()
    _nonempty(selected, name="D1 branch metrics")
    branches = set(selected["branch"].astype(str).str.lower())
    if not {"predicted", "gt_oracle"}.issubset(branches):
        raise ValueError(
            "D1 curve requires both predicted and primary GT-oracle branches"
        )
    if "oracle_at_10" not in selected:
        raise ValueError("D1 counterfactual curve requires oracle_at_10")
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for index, row in enumerate(selected.sort_values("branch").to_dict("records")):
        n = float(row["N"])
        if n <= 0:
            raise ValueError("D1 branch denominator must be positive")
        values = (
            np.asarray(
                [row["oracle_at_5"], row["oracle_at_10"], row["oracle_all"]],
                dtype=float,
            )
            / n
        )
        if not np.isfinite(values).all() or (np.diff(values) < -1e-12).any():
            raise ValueError("D1 Oracle@K curve must be finite and nondecreasing")
        ax.plot(
            x,
            values,
            marker=("o", "s", "^")[index % 3],
            color=OKABE_ITO[index % 7],
            label=f"{row['branch']} · N={int(n):,}",
        )
    ax.set_xticks(x, ["Top-5", "Top-10", "All NMS"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Oracle rate")
    ax.legend()
    _finish(fig, "D1 Top-5 / Top-10 / All-NMS counterfactual curve")
    return fig


_RENDERERS: dict[str, Callable[[Mapping[str, pd.DataFrame]], plt.Figure]] = {
    "01_pred_vs_gt_oracle_all": _oracle_all,
    "02_candidate_funnel": _candidate_funnel,
    "03_bottleneck_shift_before_after_r7": _taxonomy_shift,
    "04_grounding_recovery_vs_generator_residual": _recovery_generator,
    "05_first_positive_rank_transition": _first_rank,
    "06_candidate_count_transition": _candidate_transitions,
    "07_mask_iou_stratified_recovery": lambda tables: _stratified(
        tables, "mask_iou", "Mask-IoU-stratified recovery"
    ),
    "08_query_type_recovery": lambda tables: _stratified(
        tables, "query_type", "Query-type recovery"
    ),
    "09_gt_mask_regression_count": _regressions,
    "10_route_comparison_forest": _forest,
    "11_stage_replacement_sankey": _sankey,
    "12_d1_top5_top10_allnms_curve": _d1_curve,
}


def render_all_figures(
    run_dir: str | Path,
    table_manifest_path: str | Path | None = None,
    *,
    allow_missing_d1_primary: bool = False,
) -> Path:
    """Render all 12 preregistered figures as PDF, SVG and 400-DPI PNG."""

    tables, table_manifest = load_bound_tables(run_dir, table_manifest_path)
    root = Path(run_dir).expanduser().resolve()
    output = root / "13_figures"
    output.mkdir(parents=True, exist_ok=True)
    _style()
    records: dict[str, dict[str, Any]] = {}
    for name, required in FIGURE_SPECS.items():
        if allow_missing_d1_primary and name == "12_d1_top5_top10_allnms_curve":
            continue
        if any(table not in tables for table in required):
            raise ValueError(f"figure {name} misses a bound source table")
        fig = _RENDERERS[name](tables)
        try:
            files: dict[str, Any] = {}
            for suffix in ("pdf", "svg", "png"):
                path = output / f"{name}.{suffix}"
                fig.savefig(
                    path, dpi=400 if suffix == "png" else None, facecolor="white"
                )
                files[suffix] = artifact_record(path)
            records[name] = files
        finally:
            plt.close(fig)
    summary_path = atomic_copy(
        output / "01_pred_vs_gt_oracle_all.pdf",
        output / "gtmask_counterfactual_summary.pdf",
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "core_status": "COMPLETE",
        "d1_secondary_status": (
            "PENDING_AFTER_CORE" if allow_missing_d1_primary else "COMPLETE"
        ),
        "oracle_diagnostic": True,
        "palette": "Okabe-Ito",
        "png_dpi": 400,
        "figure_count": len(records),
        "missing_figures": (
            [
                {
                    "name": "12_d1_top5_top10_allnms_curve",
                    "reason": "D1 is a post-core secondary extension; no core figure was omitted",
                }
            ]
            if allow_missing_d1_primary
            else []
        ),
        "formats": ["pdf", "svg", "png"],
        "core_summary_figure": artifact_record(summary_path),
        "table_bundle": artifact_record(
            root / TABLE_MANIFEST_RELATIVE_PATH
            if table_manifest_path is None
            else Path(table_manifest_path).expanduser().resolve()
        ),
        "table_bundle_content_sha256": table_manifest["content_sha256"],
        "figures": records,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    return atomic_json(output / "FIGURES_MANIFEST.json", manifest)


__all__ = ["FIGURE_SPECS", "OKABE_ITO", "render_all_figures"]
