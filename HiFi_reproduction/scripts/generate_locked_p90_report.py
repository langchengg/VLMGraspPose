#!/usr/bin/env python3
"""Reproduce the locked P@90 report with robust presentation serialization.

This is a post-lock analysis wrapper.  It does not participate in proposal
generation, selector inference, gating, or output locking.  The underlying
report module intentionally remains byte-identical to the formal method lock.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = PROJECT_ROOT / "scripts/generate_sam3_p90_report.py"


def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_report_module():
    specification = importlib.util.spec_from_file_location(
        "sam3_p90_report_locked", REPORT_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load report module: {REPORT_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _redraw_method_comparison(module) -> None:
    payload = json.loads((module.REPORT / "formal_metrics.json").read_text())
    metrics = payload["metrics"]
    names = [
        "Local HiFi-CS baseline",
        "Existing selective SAM3",
        "SAM3 full-query highest score",
        "Maximum HiFi overlap",
        "Deterministic relation-aware",
        "Learned Stage-1 selector",
        "Stage-1 + Stage-2 selector (pre-gate)",
        "Locked conservative method",
        "Stage-1 GT oracle",
        "Stage-2 GT oracle",
    ]
    thresholds = [50, 60, 70, 80, 90]
    colours = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    styles = ["-"] * 5 + ["--"] * 5
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, name in enumerate(names):
        axis.plot(
            thresholds,
            [metrics[name][f"p_at_{threshold}"] for threshold in thresholds],
            marker=markers[index],
            linestyle=styles[index],
            linewidth=1.8,
            markersize=6,
            label=name,
            color=colours[index],
        )
    axis.set(
        xlabel="IoU threshold (%)",
        ylabel="Precision at threshold",
        ylim=(0, 1.02),
    )
    axis.legend(fontsize=7, ncol=2, loc="lower left")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            module.FIGURES / f"01_p50_p90_comparison.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(figure)


def main() -> int:
    module = _load_report_module()

    def save(figure, name: str, data: dict) -> None:
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(
                module.FIGURES / f"{name}.{suffix}",
                dpi=300 if suffix == "png" else None,
            )
        (module.FIGURES / f"{name}.data.json").write_text(
            json.dumps(
                _clean_json(data), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        module.plt.close(figure)

    original_read_csv = module.pd.read_csv

    def read_csv(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        if (
            args
            and Path(args[0]).name == "grouped_results.csv"
            and "value" in frame
        ):
            frame["value"] = frame["value"].fillna("none").astype(str)
        return frame

    module._save = save
    module.pd.read_csv = read_csv
    result = int(module.main())
    _redraw_method_comparison(module)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
