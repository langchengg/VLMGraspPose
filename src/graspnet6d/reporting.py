"""Reproducible paper-plot helpers that reject placeholder result tables."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd


# Color-blind-safe Okabe-Ito palette.  Native is intentionally neutral while
# the intervention uses blue; additional systems receive distinct hues.
OKABE_ITO = (
    "#999999",
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
)

_PLACEHOLDER = re.compile(
    r"^(placeholder|todo|tbd|dummy|fake|synthetic|not[_ -]?run|pending)$",
    re.IGNORECASE,
)


def validate_result_rows(
    rows: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    """Return a validated copy or refuse empty/blocked/placeholder data."""

    if not isinstance(rows, pd.DataFrame) or rows.empty:
        raise ValueError("cannot generate a report from empty result rows")
    required = tuple(map(str, required_columns))
    numeric = tuple(map(str, numeric_columns))
    missing = sorted(set(required).difference(rows.columns))
    if missing:
        raise ValueError(f"result rows missing columns: {missing}")
    work = rows.copy()
    if "is_placeholder" in work and _truthy(work["is_placeholder"]).any():
        raise ValueError("placeholder rows are not reportable")
    if "status" in work:
        blocked = work["status"].astype(str).str.upper().isin(
            {"BLOCKED", "PENDING", "NOT_RUN", "FAILED"}
        )
        if blocked.any():
            raise ValueError("blocked, pending, or failed rows are not reportable")
    for column in work.select_dtypes(include=["object", "string"]).columns:
        values = work[column].dropna().astype(str).str.strip()
        if values.map(lambda value: bool(_PLACEHOLDER.fullmatch(value))).any():
            raise ValueError(f"placeholder token found in result column {column!r}")
    if work[list(required)].isna().any().any():
        raise ValueError("required result fields contain null values")
    for column in numeric:
        values = pd.to_numeric(work[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"result column {column!r} must be finite numeric data")
        work[column] = values
    return work


def _truthy(values: pd.Series) -> np.ndarray:
    lowered = values.fillna(False).astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"}).to_numpy(bool)


def load_result_rows(
    path: str | Path,
    *,
    required_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"result CSV is missing or empty: {source}")
    return validate_result_rows(
        pd.read_csv(source),
        required_columns=required_columns,
        numeric_columns=numeric_columns,
    )


def plot_system_comparison(
    rows: pd.DataFrame,
    output_stem: str | Path,
    *,
    condition_column: str = "condition",
    system_column: str = "system",
    value_column: str = "value",
    seed_column: str = "seed",
    ylabel: str = "Target P@1",
    title: str | None = None,
) -> dict[str, Path]:
    """Plot means and seed variation from a tidy, non-placeholder result table.

    The input rows, rather than hand-entered chart values, determine every bar
    and error bar.  Both vector PDF and high-resolution PNG are written.
    """

    required = [condition_column, system_column, value_column]
    if seed_column in rows:
        required.append(seed_column)
    work = validate_result_rows(
        rows, required_columns=required, numeric_columns=[value_column]
    )
    if work.duplicated(required).any():
        raise ValueError("result rows contain duplicate condition/system/seed keys")
    conditions = list(dict.fromkeys(work[condition_column].astype(str)))
    systems = list(dict.fromkeys(work[system_column].astype(str)))
    if len(systems) < 2:
        raise ValueError("a system-comparison plot requires at least two systems")
    if len(systems) > len(OKABE_ITO):
        raise ValueError("too many systems for the locked Okabe-Ito palette")

    grouped = (
        work.assign(
            **{
                condition_column: work[condition_column].astype(str),
                system_column: work[system_column].astype(str),
            }
        )
        .groupby([condition_column, system_column], sort=False)[value_column]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    expected = pd.MultiIndex.from_product(
        [conditions, systems], names=[condition_column, system_column]
    )
    observed = pd.MultiIndex.from_frame(grouped[[condition_column, system_column]])
    if set(observed) != set(expected):
        raise ValueError("every condition must contain every plotted system")
    grouped = grouped.set_index([condition_column, system_column]).loc[expected].reset_index()
    grouped["sem"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])

    # Import lazily so metric-only and smoke-test stages do not initialize a
    # plotting backend.  Agg is deterministic and headless-safe on macOS CI.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        width = 0.8 / len(systems)
        x = np.arange(len(conditions), dtype=float)
        fig, axis = plt.subplots(figsize=(max(3.35, 0.85 * len(conditions)), 2.35))
        for index, system in enumerate(systems):
            subset = grouped[grouped[system_column].eq(system)]
            positions = x - 0.4 + width / 2.0 + index * width
            axis.bar(
                positions,
                subset["mean"].to_numpy(float),
                width=width,
                yerr=subset["sem"].to_numpy(float),
                capsize=2.0,
                label=system,
                color=OKABE_ITO[index],
                edgecolor="black",
                linewidth=0.45,
                error_kw={"elinewidth": 0.65, "capthick": 0.65},
            )
        axis.set_xticks(x, conditions)
        axis.set_ylabel(ylabel)
        if title:
            axis.set_title(str(title))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=min(3, len(systems)))
        fig.tight_layout(pad=0.5)

        stem = Path(output_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        pdf = stem.with_suffix(".pdf")
        png = stem.with_suffix(".png")
        fig.savefig(
            pdf,
            bbox_inches="tight",
            metadata={"Creator": "graspnet6d.reporting", "CreationDate": None},
        )
        fig.savefig(
            png,
            dpi=300,
            bbox_inches="tight",
            metadata={"Software": "graspnet6d.reporting"},
        )
        plt.close(fig)
    for artifact in (pdf, png):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"plot writer failed to create {artifact}")
    return {"pdf": pdf, "png": png}


__all__ = [
    "OKABE_ITO",
    "load_result_rows",
    "plot_system_comparison",
    "validate_result_rows",
]
