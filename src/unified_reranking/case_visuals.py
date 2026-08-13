"""Publication figures, case boards, and a static gallery for case analysis."""

from __future__ import annotations

import html
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from .ablation import FEATURE_FAMILY_ORDER


PALETTE = {
    "crog": "#0072B2",
    "g1": "#E69F00",
    "c1": "#009E73",
    "native": "#00BFC4",
    "challenger": "#E69F00",
    "final": "#CC79A7",
    "gt": "#0072B2",
    "gray": "#8C8C8C",
    "red": "#D55E00",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _route_grouped_bars(
    frame: pd.DataFrame, columns: Sequence[str], title: str, ylabel: str
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(frame))
    width = 0.8 / len(columns)
    for index, column in enumerate(columns):
        values = frame[column].to_numpy(float)
        bars = ax.bar(
            x + (index - (len(columns) - 1) / 2) * width,
            values,
            width,
            label=column.replace("_", " "),
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, frame["route"].str.upper())
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def build_figures(
    output_dir: Path,
    performance: pd.DataFrame,
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    pairs: pd.DataFrame,
    importance: pd.DataFrame,
    ablation: pd.DataFrame,
) -> list[str]:
    """Write a compact set of vector-first paper figures (16 unique views)."""

    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    def emit(name: str, fig: plt.Figure) -> None:
        save_figure(fig, output_dir / name)
        created.append(name)

    emit(
        "01_native_ungated_gated_j1",
        _route_grouped_bars(
            performance,
            ["native_j_at_1", "ungated_j_at_1", "gated_j_at_1"],
            "Frozen Test Top-1 performance",
            "J@1",
        ),
    )
    emit(
        "02_oracle_headroom",
        _route_grouped_bars(
            performance,
            ["native_j_at_1", "gated_j_at_1", "oracle_at_5"],
            "Order-only headroom",
            "fraction of Test samples",
        ),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(3)
    ax.bar(x - 0.18, performance["recovered"], 0.36, label="recovered", color="#009E73")
    ax.bar(x + 0.18, -performance["harmful"], 0.36, label="harmful", color="#D55E00")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, performance["route"].str.upper())
    ax.set_ylabel("sample count")
    ax.set_title("Recovered decisions versus harmful switches")
    ax.legend(frameon=False)
    fig.tight_layout()
    emit("03_recovery_harm_waterfall", fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for route in ("crog", "g1", "c1"):
        sub = (
            candidates[candidates["route"] == route]
            .groupby("native_rank")["candidate_success"]
            .mean()
        )
        ax.plot(
            sub.index, sub.values, marker="o", label=route.upper(), color=PALETTE[route]
        )
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("frozen native rank")
    ax.set_ylabel("candidate positive rate")
    ax.set_title("Where positive candidates occur in the frozen Top-5")
    ax.legend(frameon=False)
    fig.tight_layout()
    emit("04_positive_rate_by_native_rank", fig)

    transition = (
        candidates.groupby(["route", "native_rank", "improved_rank"])
        .size()
        .rename("n")
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, route in zip(axes, ("crog", "g1", "c1"), strict=True):
        pivot = (
            transition[transition["route"] == route]
            .pivot(index="native_rank", columns="improved_rank", values="n")
            .fillna(0)
        )
        image = ax.imshow(pivot, cmap="Blues", origin="upper")
        ax.set_title(route.upper())
        ax.set_xlabel("improved rank")
        ax.set_ylabel("native rank")
        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(5), range(1, 6))
    fig.colorbar(image, ax=axes, shrink=0.72, label="candidates")
    fig.suptitle("Native-to-improved rank transition")
    emit("05_rank_transition_heatmap", fig)

    switched = samples[
        samples["native_candidate_id"] != samples["ungated_candidate_id"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for route in ("crog", "g1", "c1"):
        sub = switched[switched["route"] == route]
        ax.hist(
            sub["score_margin"].dropna(),
            bins=40,
            density=True,
            histtype="step",
            lw=2,
            label=route.upper(),
            color=PALETTE[route],
        )
    ax.set_xlabel("ensemble challenger-native margin")
    ax.set_ylabel("density")
    ax.set_title("Margin distribution for changed ungated Top-1")
    ax.legend(frameon=False)
    fig.tight_layout()
    emit("06_switch_margin_distribution", fig)

    gain = (
        importance.groupby(["route", "feature"], observed=True)["gain"]
        .mean()
        .reset_index()
    )
    gain["within_route_gain"] = gain["gain"] / gain.groupby("route")["gain"].transform(
        "sum"
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, route in zip(axes, ("crog", "g1", "c1"), strict=True):
        sub = (
            gain[gain["route"] == route]
            .nlargest(10, "within_route_gain")
            .sort_values("within_route_gain")
        )
        ax.barh(
            sub["feature"].str.replace("_", " ").str.slice(0, 30),
            sub["within_route_gain"],
            color=PALETTE[route],
        )
        ax.set_title(route.upper())
        ax.set_xlabel("normalized gain")
    fig.suptitle("Frozen LightGBM gain importance (top 10 per route)")
    fig.tight_layout()
    emit("07_gain_importance", fig)

    family_gain = (
        importance.groupby(["route", "family"], observed=True)["gain"]
        .sum()
        .reset_index()
    )
    family_gain["share"] = family_gain["gain"] / family_gain.groupby("route")[
        "gain"
    ].transform("sum")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    pivot = (
        family_gain.pivot(index="route", columns="family", values="share")
        .reindex(["crog", "g1", "c1"])
        .fillna(0)
    )
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c")
    ax.set_ylabel("within-route gain share")
    ax.set_xlabel("")
    ax.set_title("Feature-family share of split gain")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    emit("08_family_gain_share", fig)

    changed_pairs = pairs[
        pairs["native_candidate_id"] != pairs["challenger_candidate_id"]
    ]
    family_cols = [f"family_delta::{family}" for family in FEATURE_FAMILY_ORDER]
    abs_family = (
        changed_pairs.groupby(["route", "outcome"])[family_cols]
        .apply(lambda x: x.abs().mean())
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, route in zip(axes, ("crog", "g1", "c1"), strict=True):
        sub = abs_family[
            (abs_family["route"] == route)
            & abs_family["outcome"].isin(["recovered", "harmful"])
        ]
        x = np.arange(len(FEATURE_FAMILY_ORDER))
        width = 0.36
        for j, outcome in enumerate(("recovered", "harmful")):
            row = sub[sub["outcome"] == outcome]
            values = (
                row[family_cols].iloc[0].to_numpy(float)
                if len(row)
                else np.zeros(len(x))
            )
            ax.bar(x + (j - 0.5) * width, values, width, label=outcome)
        ax.set_title(route.upper())
        ax.set_xticks(
            x,
            [v.replace("_", "\n") for v in FEATURE_FAMILY_ORDER],
            rotation=45,
            ha="right",
            fontsize=6,
        )
    axes[0].set_ylabel("mean |challenger-native contribution delta|")
    axes[-1].legend(frameon=False)
    fig.suptitle("Contribution magnitude in recovered and harmful switches")
    fig.tight_layout()
    emit("09_recovered_harmful_family_contributions", fig)

    mech = (
        changed_pairs.groupby(["route", "mechanism"]).size().rename("n").reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, route in zip(axes, ("crog", "g1", "c1"), strict=True):
        sub = mech[mech["route"] == route].sort_values("n")
        ax.barh(sub["mechanism"].str.slice(0, 34), sub["n"], color=PALETTE[route])
        ax.set_title(route.upper())
    fig.suptitle("Dominant post-hoc mechanism among changed rankings")
    fig.tight_layout()
    emit("10_dominant_mechanisms", fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for route in ("crog", "g1", "c1"):
        sub = switched[switched["route"] == route]
        ax.scatter(
            sub["score_margin"],
            sub["utility"],
            s=5,
            alpha=0.2,
            label=route.upper(),
            color=PALETTE[route],
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("ranker margin")
    ax.set_ylabel("gate utility")
    ax.set_title("Ranker confidence and conservative-gate utility")
    ax.legend(frameon=False)
    fig.tight_layout()
    emit("11_gate_margin_utility", fig)

    gate_reasons = (
        samples.groupby(["route", "gate_decision_reason"])
        .size()
        .rename("n")
        .reset_index()
    )
    top_reasons = (
        gate_reasons.groupby("gate_decision_reason")["n"].sum().nlargest(10).index
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = (
        gate_reasons[gate_reasons["gate_decision_reason"].isin(top_reasons)]
        .pivot(index="gate_decision_reason", columns="route", values="n")
        .fillna(0)
    )
    pivot.plot(kind="barh", ax=ax, color=[PALETTE[x] for x in pivot.columns])
    ax.set_xlabel("samples")
    ax.set_ylabel("")
    ax.set_title("Most frequent gate outcomes/rejection conjunctions")
    ax.legend(frameon=False)
    fig.tight_layout()
    emit("12_gate_rejection_reasons", fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        performance["route"].str.upper(),
        performance["irreparable"] / performance["sample_count"],
        color=[PALETTE[r] for r in performance["route"]],
    )
    ax.set_ylabel("fraction of Test samples")
    ax.set_title("Reranking-irreparable samples (E0+E1+E2)")
    fig.tight_layout()
    emit("13_irreparable_bottleneck", fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, label in (
        (axes[0], "final_diagnostic_iou", "matched rectangle IoU"),
        (axes[1], "final_diagnostic_angle_error_deg", "angle error (deg)"),
    ):
        for outcome, color in (("recovered", "#009E73"), ("harmful", "#D55E00")):
            ax.hist(
                samples[samples["outcome"] == outcome][metric].dropna(),
                bins=35,
                density=True,
                histtype="step",
                lw=2,
                label=outcome,
                color=color,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(frameon=False)
    fig.suptitle("Formal matched-rectangle diagnostics by outcome")
    fig.tight_layout()
    emit("14_iou_angle_diagnostics", fig)

    if not ablation.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
        for ax, route in zip(axes, ("crog", "g1", "c1"), strict=True):
            sub = ablation[
                ablation["route"].astype(str).str.lower() == route
            ].sort_values("family")
            value_col = "delta_j_at_1" if "delta_j_at_1" in sub else "j_at_1"
            ax.barh(
                sub["family"].str.replace("_", " "),
                sub[value_col],
                color=PALETTE[route],
            )
            ax.set_title(route.upper())
        fig.suptitle("Validation-only leave-one-family-out ablation")
        fig.tight_layout()
        emit("15_validation_ablation_triangulation", fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, "Frozen Top-5\ncandidate pool", "#D9D9D9"),
        (0.22, "105 locked T2\nevidence features", "#56B4E9"),
        (0.43, "3-seed LightGBM\nLambdaMART", "#E69F00"),
        (0.65, "Conservative\ntransition gate", "#CC79A7"),
        (0.84, "Final Top-1\nformal outcome", "#009E73"),
    ]
    for x, text, color in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, 0.35), 0.14, 0.3, color=color, alpha=0.8, transform=ax.transAxes
            )
        )
        ax.text(
            x + 0.07,
            0.5,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )
    for x in (0.17, 0.38, 0.60, 0.79):
        ax.annotate(
            "",
            xy=(x + 0.04, 0.5),
            xytext=(x, 0.5),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 1.4},
        )
    ax.set_title("Frozen order-only decision chain (post-formal analysis)")
    emit("16_decision_pipeline", fig)
    return created


def _read_image(
    path: Any, *, fallback_shape: tuple[int, int] = (480, 640)
) -> np.ndarray:
    if isinstance(path, str) and Path(path).is_file():
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
    return np.zeros((*fallback_shape, 3), dtype=np.uint8)


def rectangle_points(row: Mapping[str, Any]) -> np.ndarray:
    return cv2.boxPoints(
        (
            (float(row["cx_px"]), float(row["cy_px"])),
            (float(row["width_px"]), float(row["height_px"])),
            -float(row["theta_deg"]),
        )
    )


def _draw(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    selected_id: str | None,
    selected_color: str,
    title: str,
    gt: Sequence[Sequence[float]] = (),
) -> None:
    for candidate in rows.to_dict("records"):
        color = (
            selected_color
            if str(candidate["candidate_id"]) == str(selected_id)
            else PALETTE["gray"]
        )
        linestyle = "-" if color != PALETTE["gray"] else ":"
        ax.add_patch(
            Polygon(
                rectangle_points(candidate),
                fill=False,
                edgecolor=color,
                linewidth=2.5 if color != PALETTE["gray"] else 0.8,
                linestyle=linestyle,
            )
        )
    for rectangle in gt:
        values = np.asarray(rectangle, dtype=float)
        if values.shape == (4, 2):
            ax.add_patch(
                Polygon(
                    values,
                    fill=False,
                    edgecolor=PALETTE["gt"],
                    linewidth=1.2,
                    linestyle="--",
                )
            )
    ax.set_title(title, fontsize=10, loc="left")
    ax.axis("off")


def render_case_board(
    sample: Mapping[str, Any],
    candidate_rows: pd.DataFrame,
    output_path: Path,
    *,
    gt_rectangles: Sequence[Sequence[Sequence[float]]] = (),
) -> None:
    """Render a label-free-image overlay board; text stays outside imagery."""

    _style()
    rgb = _read_image(sample.get("source_rgb_path"))
    mask = _read_image(sample.get("predicted_mask_path"), fallback_shape=rgb.shape[:2])
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(
            mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    overlay = rgb.copy()
    binary = mask[..., 0] > 0
    overlay[binary] = (0.68 * overlay[binary] + 0.32 * np.array([0, 158, 115])).astype(
        np.uint8
    )
    fig = plt.figure(figsize=(12, 6.75), dpi=200)
    grid = fig.add_gridspec(2, 4, height_ratios=[4.5, 1.4], hspace=0.08, wspace=0.02)
    titles = [
        (
            "A  frozen native Top-1",
            sample.get("native_candidate_id"),
            PALETTE["native"],
        ),
        (
            "B  ungated learned Top-1",
            sample.get("ungated_candidate_id"),
            PALETTE["challenger"],
        ),
        (
            "C  conservative-gated final",
            sample.get("gated_candidate_id"),
            PALETTE["final"],
        ),
        ("D  target-mask context", sample.get("gated_candidate_id"), PALETTE["final"]),
    ]
    for index, (title, candidate_id, color) in enumerate(titles):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(overlay if index == 3 else rgb)
        _draw(
            ax,
            candidate_rows,
            selected_id=str(candidate_id),
            selected_color=color,
            title=title,
            gt=gt_rectangles,
        )
    text_ax = fig.add_subplot(grid[1, :])
    text_ax.axis("off")
    first = (
        f"{str(sample.get('route')).upper()}  |  {sample.get('analysis_category')}  |  "
        f"sample {sample.get('sample_id')}  |  query: {str(sample.get('language', ''))[:120]}"
    )
    second = (
        f"native {sample.get('native_candidate_id')} ({bool(sample.get('native_correct'))})  →  "
        f"challenger {sample.get('ungated_candidate_id')} ({bool(sample.get('ungated_correct'))})  →  "
        f"final {sample.get('gated_candidate_id')} ({bool(sample.get('gated_correct'))});  "
        f"margin={float(sample.get('score_margin', float('nan'))):.3f}, utility={float(sample.get('utility', float('nan'))):.3f}, "
        f"gate={sample.get('gate_decision_reason')}"
    )
    third = (
        f"post-hoc contribution association: {sample.get('mechanism')}  |  "
        f"dominant family={sample.get('dominant_family')} (|share|={float(sample.get('dominant_abs_share', float('nan'))):.2f}).  "
        "This is a model-score decomposition, not a causal proof or physical-grasp trial."
    )
    text_ax.text(
        0.01, 0.92, textwrap.fill(first, 170), fontsize=8.2, weight="bold", va="top"
    )
    text_ax.text(0.01, 0.59, textwrap.fill(second, 180), fontsize=7.7, va="top")
    text_ax.text(0.01, 0.27, textwrap.fill(third, 180), fontsize=7.4, va="top")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


def write_gallery_html(output_path: Path, gallery: pd.DataFrame, *, title: str) -> None:
    cards = []
    for row in gallery.to_dict("records"):
        image = html.escape(str(row["board_relative_path"]))
        cards.append(
            f'<article data-route="{html.escape(str(row["route"]))}" data-category="{html.escape(str(row["category"]))}">'
            f'<a href="{image}"><img loading="lazy" src="{image}" alt="case board"></a>'
            f"<p><b>{html.escape(str(row['route']).upper())}</b> · {html.escape(str(row['category']))}<br>"
            f"{html.escape(str(row['sample_id']))}</p></article>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:14px system-ui;margin:24px;background:#f6f6f6}}.controls{{position:sticky;top:0;background:white;padding:12px;z-index:2}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}article{{background:white;padding:8px;border-radius:8px;box-shadow:0 1px 4px #bbb}}img{{width:100%;height:auto}}button{{margin-right:6px}}</style>
<script>function f(v){{document.querySelectorAll('article').forEach(x=>x.style.display=(v==='all'||x.dataset.route===v||x.dataset.category===v)?'block':'none')}}</script></head>
<body><h1>{html.escape(title)}</h1><div class="controls"><button onclick="f('all')">all</button><button onclick="f('crog')">CROG</button><button onclick="f('g1')">G1</button><button onclick="f('c1')">C1</button></div><main>{"".join(cards)}</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document)


__all__ = [
    "PALETTE",
    "build_figures",
    "rectangle_points",
    "render_case_board",
    "save_figure",
    "write_gallery_html",
]
