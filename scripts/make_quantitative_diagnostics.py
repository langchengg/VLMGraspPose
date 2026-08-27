#!/usr/bin/env python3
"""Build the Chapter 4 quantitative evidence figures.

All displayed values are read from final locked artefacts.  The main figure is
a deliberately sparse two-panel summary for 0.99\textwidth placement in the
dissertation; the D1 R0--R7 Validation trace is exported separately for
appendix use.  Validation LOFO and gate diagnostics remain in the tidy export
for auditability, but are not squeezed into the main evidence figure.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[1]
OUT = REPO / "images" / "chapter4"
SUBMISSION_OUT = REPO / "figures" / "ch4"
CONSOLIDATED = REPO / "runs" / "four_route_evidence_consolidation_20260813T155455Z"
TABLES = CONSOLIDATED / "03_tables"
FIGURE_SOURCES = CONSOLIDATED / "04_figures"
UNIFIED = REPO / "runs" / "fair_unified_reranking_20260809_103012"

ROUTE_ORDER = ["CROG", "D1", "G1", "C1"]
ROUTE_DISPLAY_NAMES = {
    "CROG": "CROG",
    "D1": "D1 (HiFi-CS → Dex-Net/GQ-CNN)",
    "G1": "G1 (HiFi-CS → GR-ConvNet)",
    "C1": "C1 (HiFi-CS → GG-CNN2)",
}
PALETTE = {
    "CROG": "#0072B2",
    "D1": "#E69F00",
    "G1": "#009E73",
    "C1": "#CC79A7",
}
MARKERS = {"CROG": "o", "D1": "*", "G1": "s", "C1": "D"}
N = 7_675
MAIN_SIZE_IN = (7.2, 3.8)
APPENDIX_SIZE_IN = (7.2, 2.45)
MIN_FONT_PT = 9.3

SOURCE_FILES = {
    "native": TABLES / "table_02_formal_native.csv",
    "final": TABLES / "table_04_native_vs_gated.csv",
    "topk": TABLES / "table_05_topk_oracle.csv",
    "decomposition": TABLES / "table_06_recovered_harmful.csv",
    "d1_validation": TABLES / "table_09_d1_r0_r7_validation.csv",
    "statistics": TABLES / "table_10_statistical_checks.csv",
    "gate": FIGURE_SOURCES / "fig_10_gate_risk_coverage_source.csv",
    "lofo": UNIFIED / "07_validation" / "ablations" / "leave_one_family_out_ablation.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def publish_submission_mirrors(paths: list[Path]) -> list[Path]:
    """Copy byte-identical build artefacts to the dissertation contract path."""
    SUBMISSION_OUT.mkdir(parents=True, exist_ok=True)
    mirrors: list[Path] = []
    for source in paths:
        target = SUBMISSION_OUT / source.name
        shutil.copyfile(source, target)
        if sha256(target) != sha256(source):
            raise AssertionError(f"submission mirror differs from active output: {target}")
        mirrors.append(target)
    return mirrors


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": MIN_FONT_PT,
            "axes.titlesize": 10.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": MIN_FONT_PT,
            "ytick.labelsize": MIN_FONT_PT,
            "legend.fontsize": MIN_FONT_PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#D8DEE5", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#68727D")
    ax.spines["bottom"].set_color("#68727D")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def load_data() -> dict[str, pd.DataFrame]:
    for path in SOURCE_FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    native_raw = pd.read_csv(SOURCE_FILES["native"])
    final_raw = pd.read_csv(SOURCE_FILES["final"])
    topk_raw = pd.read_csv(SOURCE_FILES["topk"])
    stats_raw = pd.read_csv(SOURCE_FILES["statistics"])
    decomp_raw = pd.read_csv(SOURCE_FILES["decomposition"])
    gate_raw = pd.read_csv(SOURCE_FILES["gate"])
    d1_validation = pd.read_csv(SOURCE_FILES["d1_validation"])
    lofo = pd.read_csv(SOURCE_FILES["lofo"])

    top5_rows = topk_raw[
        topk_raw["route"].isin(["CROG", "G1", "C1", "D1 Top-5"])
    ].copy()
    top5_rows["route"] = top5_rows["route"].replace({"D1 Top-5": "D1"})
    native = native_raw[["route", "j_at_1"]].rename(
        columns={"j_at_1": "native_j_at_1"}
    )
    native = pd.concat(
        [
            native,
            top5_rows[top5_rows["route"].eq("D1")][["route", "j_at_1"]].rename(
                columns={"j_at_1": "native_j_at_1"}
            ),
        ],
        ignore_index=True,
    )
    performance = (
        native.merge(
            final_raw[["route", "j_at_1", "delta_percentage_points", "scope"]].rename(
                columns={"j_at_1": "final_j_at_1"}
            ),
            on="route",
            validate="one_to_one",
        )
        .merge(
            top5_rows[["route", "j_at_5", "oracle_all"]].rename(
                columns={"j_at_5": "oracle_at_5"}
            ),
            on="route",
            validate="one_to_one",
        )
        .merge(
            stats_raw[["route", "scene_ci95_lower", "scene_ci95_upper", "holm_p"]],
            on="route",
            validate="one_to_one",
        )
        .set_index("route")
        .loc[ROUTE_ORDER]
        .reset_index()
    )
    decomp = decomp_raw.set_index("route").loc[ROUTE_ORDER].reset_index()
    gate = gate_raw.set_index("route").loc[ROUTE_ORDER].reset_index()
    lofo = lofo[lofo["route"].isin(["crog", "g1", "c1"])].copy()

    # Locked arithmetic invariants.  Any stale or mixed-scope source fails
    # before plotting rather than producing a plausible-looking figure.
    assert performance["route"].tolist() == ROUTE_ORDER
    assert decomp["route"].tolist() == ROUTE_ORDER
    for row in performance.itertuples(index=False):
        gain = 100.0 * (row.final_j_at_1 - row.native_j_at_1)
        if not math.isclose(gain, row.delta_percentage_points, abs_tol=1e-11):
            raise AssertionError(f"gain mismatch for {row.route}: {gain}")
    for row in decomp.itertuples(index=False):
        if int(row.net) != int(row.recovered - row.harmful):
            raise AssertionError(f"net mismatch for {row.route}")
        expected_gain = 100.0 * row.net / N
        observed = float(
            performance.loc[
                performance["route"].eq(row.route), "delta_percentage_points"
            ].iloc[0]
        )
        if not math.isclose(expected_gain, observed, abs_tol=1e-11):
            raise AssertionError(f"paired identity mismatch for {row.route}")
        expected_precision = row.recovered / (row.recovered + row.harmful)
        if not math.isclose(
            expected_precision, row.outcome_changing_precision, abs_tol=1e-12
        ):
            raise AssertionError(f"precision mismatch for {row.route}")
        performance_row = performance[performance["route"].eq(row.route)].iloc[0]
        top5_headroom = round(
            N
            * (
                float(performance_row["oracle_at_5"])
                - float(performance_row["native_j_at_1"])
            )
        )
        expected_headroom = row.net / top5_headroom
        if not math.isclose(
            expected_headroom, row.headroom_recovery_at_5, abs_tol=1e-12
        ):
            raise AssertionError(f"headroom recovery mismatch for {row.route}")
    if set(d1_validation["method"]) != {f"R{i}" for i in range(8)}:
        raise AssertionError("D1 R0-R7 table is incomplete")
    if int(d1_validation["validation_sample_count"].nunique()) != 1:
        raise AssertionError("D1 validation denominators differ")
    if set(lofo["family"]) != {
        "native_calibration",
        "soft_target_support",
        "jaw_geometry",
        "angle_agreement",
        "depth_contact",
        "collision_proxy",
        "reliability_context",
    }:
        raise AssertionError("unexpected LOFO feature-family schema")

    return {
        "performance": performance,
        "decomposition": decomp,
        "gate": gate,
        "d1_validation": d1_validation,
        "lofo": lofo,
    }


def write_tidy_data(data: dict[str, pd.DataFrame]) -> Path:
    records: list[dict[str, object]] = []

    def add(
        panel: str,
        subpanel: str,
        route: str,
        metric: str,
        value: float,
        unit: str,
        source: Path,
        scope: str,
        note: str = "",
    ) -> None:
        records.append(
            {
                "panel": panel,
                "subpanel": subpanel,
                "route": route,
                "metric": metric,
                "value": value,
                "unit": unit,
                "scope": scope,
                "source_file": rel(source),
                "note": note,
            }
        )

    for row in data["performance"].itertuples(index=False):
        scope = str(row.scope)
        native_source = SOURCE_FILES["topk"] if row.route == "D1" else SOURCE_FILES["native"]
        add("A", "performance", row.route, "native_j_at_1", row.native_j_at_1, "proportion", native_source, scope)
        add("A", "performance", row.route, "final_j_at_1", row.final_j_at_1, "proportion", SOURCE_FILES["final"], scope)
        add("A", "performance", row.route, "oracle_at_5", row.oracle_at_5, "proportion", SOURCE_FILES["topk"], scope, "frozen candidate-set ceiling")
        add("A", "performance", row.route, "oracle_all", row.oracle_all, "proportion", SOURCE_FILES["topk"], scope, "reported in data export; not drawn in main figure")
        add("B", "paired_gain", row.route, "gain", row.delta_percentage_points, "percentage_points", SOURCE_FILES["statistics"], scope)
        add("B", "paired_gain", row.route, "gain_scene_ci95_lower", 100.0 * row.scene_ci95_lower, "percentage_points", SOURCE_FILES["statistics"], scope, "scene-cluster bootstrap")
        add("B", "paired_gain", row.route, "gain_scene_ci95_upper", 100.0 * row.scene_ci95_upper, "percentage_points", SOURCE_FILES["statistics"], scope, "scene-cluster bootstrap")
        add("B", "paired_gain", row.route, "holm_adjusted_p", row.holm_p, "p_value", SOURCE_FILES["statistics"], scope, "supportive exact McNemar with Holm correction; reported in export, not drawn")
    for row in data["decomposition"].itertuples(index=False):
        for metric in [
            "recovered",
            "harmful",
            "net",
            "switch_count",
            "switch_rate",
            "outcome_changing_precision",
            "headroom_recovery_at_5",
        ]:
            unit = "count" if metric in {"recovered", "harmful", "net", "switch_count"} else "proportion"
            add("B", "decomposition", row.route, metric, float(getattr(row, metric)), unit, SOURCE_FILES["decomposition"], row.scope)
    for row in data["lofo"].itertuples(index=False):
        add("Export-only", "unified_lofo", row.route.upper(), f"remove_{row.family}", 100.0 * row.delta_j_at_1, "percentage_points", SOURCE_FILES["lofo"], "T2_MATCHED_COMMON_VALIDATION", "ablated minus full; three seeds; not drawn in the two-panel main figure")
    for row in data["d1_validation"].itertuples(index=False):
        add("Appendix", "d1_model_selection", "D1", row.method, row.validation_j_at_1, "proportion", SOURCE_FILES["d1_validation"], "D1_VALIDATION", str(row.role))
    for row in data["gate"].itertuples(index=False):
        for metric in ["switch_rate", "outcome_changing_precision", "delta_percentage_points"]:
            unit = "percentage_points" if metric == "delta_percentage_points" else "proportion"
            scope = "D1_RETROSPECTIVE" if row.route == "D1" else "FORMAL_PRIMARY"
            add("Export-only", "locked_gate_operating_point", row.route, metric, float(getattr(row, metric)), unit, SOURCE_FILES["gate"], scope, "not drawn in the two-panel main figure")

    output = OUT / "quantitative_diagnostics_data.csv"
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def route_tick_label(route: str) -> str:
    return "D1*" if route == "D1" else route


def plot_panel_a(ax: plt.Axes, performance: pd.DataFrame) -> None:
    y = np.arange(len(performance))
    for index, row in performance.iterrows():
        route = str(row["route"])
        color = PALETTE[route]
        if route == "D1":
            ax.axhspan(index - 0.43, index + 0.43, color="#F1F3F5", zorder=0)
        values = 100.0 * np.array(
            [row["native_j_at_1"], row["final_j_at_1"], row["oracle_at_5"]]
        )
        ax.plot(values[:2], [index, index], color=color, linewidth=2.1, zorder=2)
        ax.plot(
            values[1:],
            [index, index],
            color=color,
            linewidth=1.25,
            linestyle=":",
            zorder=2,
        )
        ax.scatter(
            values[0],
            index,
            s=42,
            marker="o",
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=4,
        )
        ax.scatter(
            values[1],
            index,
            s=44,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=5,
        )
        ax.scatter(
            values[2],
            index,
            s=46,
            marker="D",
            facecolor="white",
            edgecolor=color,
            linewidth=1.35,
            zorder=4,
        )
        # Alternate labels above and below the trace so even CROG's closely
        # spaced Final and Oracle markers remain distinct at A4 print size.
        label_specs = [
            ((-2, -9), "right"),
            ((0, 9), "center"),
            ((4, -9), "left"),
        ]
        for value, (offset, alignment) in zip(values, label_specs, strict=True):
            ax.annotate(
                f"{value:.1f}",
                (value, index),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                va="center",
                fontsize=MIN_FONT_PT,
                color=color,
                weight="bold",
            )
        if route == "D1":
            ax.text(
                106.0,
                index,
                "retrospective\nseparate scope",
                ha="right",
                va="center",
                fontsize=8.9,
                color="#55616F",
                linespacing=1.0,
            )

    ax.set_yticks(y, [route_tick_label(route) for route in performance["route"]])
    for tick, route in zip(ax.get_yticklabels(), performance["route"], strict=True):
        tick.set_color(PALETTE[route])
        tick.set_fontweight("bold")
    ax.set_ylim(3.48, -0.72)
    ax.set_xlim(27.0, 108.0)
    ax.set_xticks([30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel("Offline J@1 / frozen Top-5 candidate-set success (%)", labelpad=2)
    ax.text(
        0.0,
        1.12,
        "(A) Native, Final and Frozen Top-5 Ceiling",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.6,
        weight="bold",
    )
    style_axis(ax, "x")
    legend = [
        Line2D([0], [0], marker="o", color="#40464D", markerfacecolor="white", markeredgewidth=1.4, linewidth=0, label="Native"),
        Line2D([0], [0], marker="o", color="#40464D", markerfacecolor="#40464D", linewidth=0, label="Final / gated"),
        Line2D([0], [0], marker="D", color="#40464D", markerfacecolor="white", linewidth=0, label="Oracle@5"),
    ]
    ax.legend(
        handles=legend,
        frameon=False,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.04),
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0.0,
    )


def plot_panel_b(
    ax: plt.Axes,
    decomp: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    y = np.arange(len(decomp))
    for index, row in decomp.iterrows():
        route = str(row["route"])
        color = PALETTE[route]
        if route == "D1":
            ax.axhspan(index - 0.43, index + 0.43, color="#F1F3F5", zorder=0)
        hatch = "///" if route == "D1" else None
        recovered_rate = 100.0 * float(row["recovered"]) / N
        harmful_rate = 100.0 * float(row["harmful"]) / N
        ax.barh(
            index,
            recovered_rate,
            height=0.50,
            color=color,
            edgecolor="white" if route != "D1" else "#6F4E00",
            linewidth=0.65,
            hatch=hatch,
            zorder=3,
        )
        ax.barh(
            index,
            -harmful_rate,
            height=0.50,
            color="white",
            edgecolor=color,
            linewidth=1.15,
            hatch="////",
            zorder=3,
        )
        if recovered_rate > 15.0:
            ax.text(
                recovered_rate - 0.25,
                index,
                f"{int(row['recovered']):,}",
                ha="right",
                va="center",
                fontsize=MIN_FONT_PT,
                color=color,
                weight="bold",
                zorder=5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                    "pad": 0.35,
                },
            )
        else:
            ax.annotate(
                f"{int(row['recovered']):,}",
                (recovered_rate, index),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=MIN_FONT_PT,
                color=color,
                weight="bold",
            )
        if harmful_rate > 1.0:
            ax.text(
                -0.5 * harmful_rate,
                index,
                f"{int(row['harmful']):,}",
                ha="center",
                va="center",
                fontsize=MIN_FONT_PT,
                color=color,
                weight="bold",
                zorder=5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                    "pad": 0.35,
                },
            )
        else:
            ax.annotate(
                f"{int(row['harmful']):,}",
                (-harmful_rate, index),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=MIN_FONT_PT,
                color=color,
                weight="bold",
            )
        perf = performance[performance["route"].eq(route)].iloc[0]
        lower = 100.0 * float(perf["scene_ci95_lower"])
        upper = 100.0 * float(perf["scene_ci95_upper"])
        ax.text(
            29.8,
            index,
            f"+{int(row['net']):,}  |  +{float(perf['delta_percentage_points']):.2f} "
            f"[{lower:.2f}, {upper:.2f}]",
            ha="center",
            va="center",
            fontsize=MIN_FONT_PT,
            color="#27313B",
            weight="bold",
        )
        ax.text(
            40.0,
            index,
            f"{100.0 * row['headroom_recovery_at_5']:.1f}",
            ha="center",
            va="center",
            fontsize=MIN_FONT_PT,
            color="#27313B",
            weight="bold",
        )

    ax.axvline(0, color="#39414A", linewidth=0.85)
    ax.axvline(23.2, color="#C8D0D8", linewidth=0.65)
    ax.axvline(37.3, color="#C8D0D8", linewidth=0.65)
    ax.set_yticks(y, [route_tick_label(route) for route in ROUTE_ORDER])
    for tick, route in zip(ax.get_yticklabels(), ROUTE_ORDER, strict=True):
        tick.set_color(PALETTE[route])
        tick.set_fontweight("bold")
    ax.set_ylim(3.55, -0.67)
    ax.set_xlim(-3.35, 42.2)
    ax.set_xticks([-2, 0, 5, 10, 15, 20])
    ax.set_xlabel(
        "Harmful  ←  paired transitions (% of 7,675 tuples)  →  Recovered",
        labelpad=2,
    )
    ax.text(
        29.8,
        -0.54,
        "Net count  |  paired gain [95% CI] (pp)",
        ha="center",
        va="bottom",
        fontsize=MIN_FONT_PT,
        weight="bold",
    )
    ax.text(
        40.0,
        -0.54,
        "HR5 (%)",
        ha="center",
        va="bottom",
        fontsize=MIN_FONT_PT,
        weight="bold",
    )
    ax.text(
        0.0,
        1.10,
        "(B) Recovered, Harmful and Paired Gain",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.6,
        weight="bold",
    )
    style_axis(ax, "x")
    ax.legend(
        handles=[
            Patch(facecolor="#66717D", edgecolor="white", label="Recovered"),
            Patch(
                facecolor="white",
                edgecolor="#66717D",
                hatch="////",
                label="Harmful",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.04),
        handlelength=1.3,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def plot_panel_c(fig: plt.Figure, spec, data: dict[str, pd.DataFrame]) -> None:
    outer = spec.subgridspec(
        2,
        2,
        height_ratios=[0.45, 1.0],
        width_ratios=[2.10, 1.0],
        hspace=0.03,
        wspace=0.34,
    )
    heading = fig.add_subplot(outer[0, :])
    heading.axis("off")
    heading.text(
        0.0,
        0.96,
        "(C) Feature Ablation and Gate Diagnostics",
        ha="left",
        va="top",
        fontsize=10.6,
        weight="bold",
    )
    heading.text(
        0.0,
        0.02,
        "LOFO · Validation (3 seeds)",
        ha="left",
        va="bottom",
        fontsize=9.5,
        weight="bold",
    )
    heading.text(
        0.79,
        0.02,
        "Gate OCP",
        ha="left",
        va="bottom",
        fontsize=9.5,
        weight="bold",
    )

    heat_grid = outer[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.10)
    ax_heat = fig.add_subplot(heat_grid[0, 0])
    ax_cbar = fig.add_subplot(heat_grid[0, 1])
    ax_gate = fig.add_subplot(outer[1, 1])

    lofo = data["lofo"].copy()
    family_order = [
        "native_calibration",
        "soft_target_support",
        "jaw_geometry",
        "angle_agreement",
        "depth_contact",
        "collision_proxy",
        "reliability_context",
    ]
    family_labels = [
        "Cal\n",
        "\nSupp",
        "Jaw\n",
        "\nAng",
        "D/C\n",
        "\nColl.",
        "Rel.\n",
    ]
    route_order = ["crog", "g1", "c1"]
    matrix = (
        lofo.pivot(index="route", columns="family", values="delta_j_at_1")
        .loc[route_order, family_order]
        .to_numpy()
        * 100.0
    )
    full_by_route: dict[str, float] = {}
    for route in route_order:
        subset = lofo[lofo["route"].eq(route)]
        full = subset["j_at_1"] - subset["delta_j_at_1"]
        if float(full.max() - full.min()) > 1e-10:
            raise AssertionError(f"LOFO full-model inconsistency for {route}")
        full_by_route[route] = float(full.mean())
    bound = 0.42
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    image = ax_heat.imshow(matrix, cmap="BrBG", norm=norm, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax_heat.text(
                col,
                row,
                ("+" if value >= 0 else "−") + f"{abs(value):.2f}".replace("0.", "."),
                ha="center",
                va="center",
                fontsize=MIN_FONT_PT,
                color="white" if abs(value) > 0.23 else "#18222C",
                weight="bold",
            )
    ax_heat.set_xticks(np.arange(len(family_labels)), family_labels)
    ax_heat.set_yticks(
        np.arange(3),
        [
            f"{route.upper()}\nFull {100.0 * full_by_route[route]:.1f}%"
            for route in route_order
        ],
    )
    for tick, route in zip(ax_heat.get_yticklabels(), ["CROG", "G1", "C1"], strict=True):
        tick.set_color(PALETTE[route])
        tick.set_fontweight("bold")
    ax_heat.tick_params(length=0, pad=2)
    ax_heat.set_xticks(np.arange(-0.5, len(family_labels), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(route_order), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.9)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, cax=ax_cbar, orientation="vertical")
    colorbar.set_ticks([-0.4, 0.0, 0.4])
    colorbar.ax.tick_params(labelleft=False, labelright=False, length=2)
    colorbar.ax.axhline(0.0, color="#27313B", linewidth=0.7)
    colorbar.ax.set_title("pp", fontsize=MIN_FONT_PT, pad=2)

    gate = data["gate"]
    offsets = {"CROG": (3, 6), "D1": (-19, 7), "G1": (3, 2), "C1": (3, -10)}
    for _, row in gate.iterrows():
        route = str(row["route"])
        x = 100.0 * float(row["switch_rate"])
        y = 100.0 * float(row["outcome_changing_precision"])
        ax_gate.scatter(
            x,
            y,
            s=52,
            color=PALETTE[route],
            marker=MARKERS[route],
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        ax_gate.annotate(
            route_tick_label(route),
            (x, y),
            xytext=offsets[route],
            textcoords="offset points",
            fontsize=MIN_FONT_PT,
            color=PALETTE[route],
            weight="bold",
        )
    ax_gate.set_xlim(30, 82)
    ax_gate.set_ylim(87, 98)
    ax_gate.set_xticks([40, 60, 80])
    ax_gate.set_yticks([88, 92, 96])
    ax_gate.set_xlabel("Coverage (%)", labelpad=2)
    ax_gate.set_ylabel("")
    style_axis(ax_gate, "both")


def build_main_figure(data: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    configure_style()
    fig = plt.figure(figsize=MAIN_SIZE_IN)
    fig.subplots_adjust(left=0.082, right=0.985, top=0.915, bottom=0.145)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.96, 1.04],
        hspace=0.72,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[1, 0])
    plot_panel_a(ax_a, data["performance"])
    plot_panel_b(ax_b, data["decomposition"], data["performance"])

    pdf = OUT / "quantitative_diagnostics.pdf"
    png = OUT / "quantitative_diagnostics.png"
    metadata = {
        "Title": "Chapter 4 quantitative diagnostics",
        "Author": "VLMGraspPose reproducible evidence pipeline",
        "Subject": "Locked 7,675-tuple performance and paired transition evidence",
        "Keywords": f"script_sha256={sha256(SCRIPT)}; D1 retrospective Top-5; offline 4-DoF",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, format="pdf", dpi=300, bbox_inches=None, metadata=metadata)
    fig.savefig(png, format="png", dpi=300, bbox_inches=None)
    plt.close(fig)
    return pdf, png


def build_d1_appendix(data: dict[str, pd.DataFrame]) -> Path:
    configure_style()
    d1 = data["d1_validation"].copy()
    d1["order"] = d1["method"].str[1:].astype(int)
    d1 = d1.sort_values("order")
    x = np.arange(len(d1))
    y = 100.0 * d1["validation_j_at_1"].to_numpy()

    fig, ax = plt.subplots(figsize=APPENDIX_SIZE_IN)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.78, bottom=0.24)
    ax.plot(x, y, color=PALETTE["D1"], linewidth=2.0, marker="o", markersize=5.2)
    selected = int(np.flatnonzero(d1["method"].eq("R5").to_numpy())[0])
    gated = int(np.flatnonzero(d1["method"].eq("R7").to_numpy())[0])
    ax.scatter(selected, y[selected], s=105, marker="*", color=PALETTE["D1"], edgecolor="#5A3E00", linewidth=0.7, zorder=4)
    ax.scatter(gated, y[gated], s=58, marker="D", facecolor="white", edgecolor=PALETTE["D1"], linewidth=1.5, zorder=4)
    ax.annotate("selected ranker", (selected, y[selected]), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom", fontsize=MIN_FONT_PT, weight="bold")
    ax.annotate(
        "R5 + route gate",
        (gated, y[gated]),
        xytext=(-4, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=MIN_FONT_PT,
        weight="bold",
    )
    ax.set_xticks(x, d1["method"])
    ax.set_ylim(34, 56)
    ax.set_ylabel("Validation J@1 (%)")
    ax.set_xlabel("D1 Validation configuration")
    ax.set_title("D1 R0–R7 Validation trace (N=3,778)", loc="left", fontsize=10.6, pad=8)
    style_axis(ax, "y")
    pdf = OUT / "d1_validation_trace_appendix.pdf"
    metadata = {
        "Title": "D1 R0-R7 Validation trace",
        "Author": "VLMGraspPose reproducible evidence pipeline",
        "Subject": "R5 selected base ranker and R7 expected-gain gating policy",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, format="pdf", dpi=300, bbox_inches=None, metadata=metadata)
    plt.close(fig)
    return pdf


def write_manifest(
    data_csv: Path,
    pdf: Path,
    png: Path,
    appendix_pdf: Path,
) -> Path:
    manifest = {
        "schema_version": 2,
        "status": "COMPLETE",
        "script": {"path": rel(SCRIPT), "sha256": sha256(SCRIPT)},
        "formal_denominator": N,
        "route_order": ROUTE_ORDER,
        "route_display_names": ROUTE_DISPLAY_NAMES,
        "main_canvas_inches": list(MAIN_SIZE_IN),
        "appendix_canvas_inches": list(APPENDIX_SIZE_IN),
        "font_family": "Arial",
        "minimum_source_font_pt": MIN_FONT_PT,
        "sources": [
            {"role": role, "path": rel(path), "sha256": sha256(path)}
            for role, path in SOURCE_FILES.items()
        ],
        "outputs": [
            {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in [pdf, png, data_csv, appendix_pdf]
        ],
        "submission_contract_mirrors": [
            str((Path("figures") / "ch4" / path.name).as_posix())
            for path in [pdf, png, data_csv, appendix_pdf]
        ]
        + ["figures/ch4/quantitative_diagnostics_manifest.json"],
        "figure_table_trace": {
            "panel_A": {
                "claim": "Native-to-final change and the frozen Oracle@5 ceiling for four scope-labelled routes.",
                "sources": [rel(SOURCE_FILES[key]) for key in ["native", "final", "topk"]],
            },
            "panel_B": {
                "claim": "Recovered outcomes exceed harmful replacements; net exactly reproduces paired J@1 gain, and HR5 reports the realised fraction of frozen Top-5 headroom.",
                "uncertainty": "The displayed 95% scene-cluster bootstrap interval applies to paired gain only.",
                "sources": [
                    rel(SOURCE_FILES["decomposition"]),
                    rel(SOURCE_FILES["statistics"]),
                ],
            },
            "export_only_diagnostics": {
                "claim": "Unified-route Validation LOFO and locked Test gate operating points are retained in the tidy data export but deliberately omitted from the two-panel main figure.",
                "sources": [rel(SOURCE_FILES[key]) for key in ["lofo", "gate"]],
            },
            "appendix": {
                "claim": "D1 R0-R7 Validation trace; R7 is R5 plus a route-specific expected-gain gate, not a distinct ranker.",
                "sources": [rel(SOURCE_FILES["d1_validation"])],
            },
        },
        "scope_boundaries": [
            "CROG/G1/C1 are FORMAL_PRIMARY route-local frozen Top-5 results.",
            "D1 is D1_RETROSPECTIVE under its own frozen post-NMS Top-5 contract and is starred in the figure.",
            "LOFO is Validation-only T2 matched-common evidence with three seeds and is export-only in this revision.",
            "The offline same-GT rectangle criterion is not physical grasp success.",
        ],
    }
    output = OUT / "quantitative_diagnostics_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    data_csv = write_tidy_data(data)
    pdf, png = build_main_figure(data)
    appendix = build_d1_appendix(data)
    manifest = write_manifest(data_csv, pdf, png, appendix)
    mirrors = publish_submission_mirrors([pdf, png, data_csv, appendix, manifest])
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(f"wrote {data_csv}")
    print(f"wrote {appendix}")
    print(f"wrote {manifest}")
    for mirror in mirrors:
        print(f"mirrored {mirror}")


if __name__ == "__main__":
    main()
