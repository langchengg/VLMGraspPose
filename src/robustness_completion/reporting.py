"""Evidence-constrained reports for the robustness-completion child run.

Missing runtime evidence is rendered as ``not estimable`` and remains a
completion blocker.  It is never silently converted to a zero-valued result.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    PREREGISTRATION_SHA256,
    RUN_ID,
    atomic_json,
    atomic_text,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)
from .runtime_aggregation import OUTPUT_NAMES, _input_signature, _source_inventory


PARENT_RUN_ID = "20260822_085125_robustness_suite"
PARENT_RELATIVE = Path("artifacts/robustness_suite") / PARENT_RUN_ID

# Okabe--Ito. Grey is reserved for unavailable/diagnostic values.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}
ROUTE_COLOURS = {
    "CROG": PALETTE["blue"],
    "G1": PALETTE["orange"],
    "C1": PALETTE["green"],
    "D1": PALETTE["purple"],
    "6D_ORACLE": PALETTE["vermillion"],
    "6D_ADAPTED": PALETTE["sky"],
}
ROUTE_ORDER = ("CROG", "G1", "C1", "D1", "6D_ORACLE", "6D_ADAPTED")
METHOD_ORDER = ("native", "raw", "gated")
FOUR_D_ROUTES = frozenset({"CROG", "G1", "C1", "D1"})
PRIMARY_FOUR_D_ROUTES = frozenset({"CROG", "G1", "C1"})
SIX_D_ROUTES = frozenset({"6D_ORACLE", "6D_ADAPTED"})
EXPECTED_WARM_SAMPLES = {
    **{route: 100 for route in FOUR_D_ROUTES},
    **{route: 98 for route in SIX_D_ROUTES},
}


@dataclass(frozen=True)
class CompletionAssessment:
    status: str
    emitted: bool
    duplicate_complete: bool
    runtime_complete: bool
    blockers: tuple[str, ...]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _verify_runtime_completion(
    runtime_root: Path, repo: Path
) -> dict[str, Any]:
    """Verify the aggregate lock before any aggregate table is consumed."""

    completion_path = runtime_root / "runtime_completion.json"
    if not completion_path.is_file():
        raise RuntimeError("runtime aggregate completion lock is missing")
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "runtime aggregate completion lock is unreadable"
        ) from error
    if not isinstance(completion, dict):
        raise RuntimeError("runtime aggregate completion lock is not an object")
    schema = completion.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < 2:
        raise RuntimeError("runtime aggregate completion schema must be >= 2")
    if completion.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise RuntimeError(
            "runtime aggregate preregistration SHA-256 does not match the lock"
        )
    outputs = completion.get("output_sha256")
    if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_NAMES):
        raise RuntimeError(
            "runtime aggregate completion does not bind the exact output set"
        )
    for name in OUTPUT_NAMES:
        path = runtime_root / name
        if not path.is_file():
            raise RuntimeError(f"runtime aggregate output is missing: {name}")
        if sha256_file(path) != outputs[name]:
            raise RuntimeError(f"runtime aggregate output SHA-256 mismatch: {name}")
    current_signature = _input_signature(_source_inventory(runtime_root, repo))
    if completion.get("input_signature") != current_signature:
        raise RuntimeError("runtime aggregate input signature mismatch")
    return completion


def _first_column(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _normalise_route(value: Any) -> str:
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ORACLE": "6D_ORACLE",
        "ORACLE_GT_MASK": "6D_ORACLE",
        "6D_ORACLE_GT_MASK": "6D_ORACLE",
        "ADAPTED": "6D_ADAPTED",
        "HIFICS_ADAPTED": "6D_ADAPTED",
        "HIFICS_ADAPTED_MASK": "6D_ADAPTED",
        "6D_HIFICS_ADAPTED": "6D_ADAPTED",
        "6D_PREDICTED": "6D_ADAPTED",
    }
    return aliases.get(text, text)


def _normalise_method(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"reranked", "raw_reranked", "raw_lambdamart", "lambdamart"}:
        return "raw"
    if text in {"gate", "expected_gain_gated", "reranker_gate", "raw_gate"}:
        return "gated"
    return text


def _normalise_mode(value: Any, default: str) -> str:
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "warm-disk-input": "warm-disk",
        "disk": "warm-disk",
        "warm": "warm-disk",
        "preloaded": "warm-preloaded",
        "warm-memory": "warm-preloaded",
        "cold-start": "cold",
    }
    return aliases.get(text, text if text and text != "nan" else default)


def _elapsed_ms(frame: pd.DataFrame) -> pd.Series:
    for name in (
        "cold_start_total_ns",
        "process_spawn_to_output_ns",
        "parent_wall_ns",
        "cold_start_elapsed_ns",
        "process_elapsed_ns",
        "total_elapsed_ns",
        "whole_elapsed_ns",
        "uninstrumented_total_ns",
        "elapsed_ns",
        "total_ns",
    ):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce") / 1e6
    for name in (
        "cold_start_ms",
        "whole_elapsed_ms",
        "uninstrumented_total_ms",
        "elapsed_ms",
        "total_ms",
    ):
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _canonical_timing(frame: pd.DataFrame, *, default_mode: str) -> pd.DataFrame:
    """Map worker/orchestrator timing schemas to the reporting contract."""

    columns = [
        "route",
        "method",
        "mode",
        "sample_id",
        "device",
        "elapsed_ms",
        "candidate_count",
        "status",
        "complete_deployment",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    route = _first_column(frame, ("route", "route_id", "deployment_route"))
    method = _first_column(frame, ("method", "variant", "system"))
    mode = _first_column(frame, ("mode", "timing_mode", "cache_policy"))
    sample = _first_column(
        frame, ("sample_id", "group_id", "tuple_id", "repetition")
    )
    device = _first_column(frame, ("device", "formal_device"))
    candidates = _first_column(frame, ("candidate_count", "n_candidates"))
    status = _first_column(frame, ("status", "measurement_status"))
    complete = _first_column(
        frame, ("complete_deployment", "deployment_complete")
    ).map(_as_bool)
    canonical = pd.DataFrame(
        {
            "route": route.map(_normalise_route),
            "method": method.map(_normalise_method),
            "mode": mode.map(lambda item: _normalise_mode(item, default_mode)),
            "sample_id": sample.astype(str),
            "device": device.astype(str).str.lower(),
            "elapsed_ms": _elapsed_ms(frame),
            "candidate_count": pd.to_numeric(candidates, errors="coerce"),
            "status": status.astype(str),
            "complete_deployment": complete,
        }
    )
    canonical.loc[canonical["method"].isin({"", "nan"}), "method"] = "native"
    canonical.loc[canonical["device"].isin({"", "nan"}), "device"] = (
        "not recorded"
    )
    missing_sample = canonical["sample_id"] == "nan"
    canonical.loc[missing_sample, "sample_id"] = canonical.index[missing_sample].astype(
        str
    )
    return canonical[columns]


def _canonical_stages(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "route",
        "method",
        "mode",
        "sample_id",
        "device",
        "stage",
        "elapsed_ms",
        "candidate_count",
        "status",
        "complete_deployment",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    base = _canonical_timing(frame, default_mode="warm-disk")
    # Stage rows also carry the independently timed whole-pipeline duration.
    # A stage decomposition must use the stage timer, never that repeated
    # whole-pipeline value.
    if "elapsed_ns" in frame.columns:
        base["elapsed_ms"] = pd.to_numeric(
            frame["elapsed_ns"], errors="coerce"
        ) / 1e6
    elif "elapsed_ms" in frame.columns:
        base["elapsed_ms"] = pd.to_numeric(
            frame["elapsed_ms"], errors="coerce"
        )
    if not {"complete_deployment", "deployment_complete"}.intersection(
        frame.columns
    ):
        base["complete_deployment"] = base["status"].astype(str).str.lower().isin(
            {"measured", "combined_formal_api"}
        )
    base.insert(
        5, "stage", _first_column(frame, ("stage", "stage_name", "component")).astype(str)
    )
    return base[columns]


def _memory_value(
    frame: pd.DataFrame, byte_names: Sequence[str], mib_names: Sequence[str]
) -> pd.Series:
    for name in byte_names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce") / (1024**2)
    for name in mib_names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _canonical_memory(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "route",
        "method",
        "device",
        "status",
        "complete_deployment",
        "peak_rss_mib",
        "model_load_delta_mib",
        "mps_mib",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    status = _first_column(frame, ("status", "measurement_status")).astype(str)
    complete = _first_column(
        frame, ("complete_deployment", "deployment_complete")
    ).map(_as_bool)
    if not {"complete_deployment", "deployment_complete"}.intersection(
        frame.columns
    ):
        complete = status.str.lower().isin(
            {"complete_deployment_memory", "measured_full_deployment", "measured"}
        )
    result = pd.DataFrame(
        {
            "route": _first_column(
                frame, ("route", "route_id", "deployment_route")
            ).map(_normalise_route),
            "method": _first_column(frame, ("method", "variant", "system")).map(
                _normalise_method
            ),
            "device": _first_column(frame, ("device", "formal_device"))
            .astype(str)
            .str.lower(),
            "status": status,
            "complete_deployment": complete,
            "peak_rss_mib": _memory_value(
                frame,
                ("peak_total_rss_bytes", "peak_rss_bytes", "peak_recursive_rss_bytes"),
                ("peak_rss_mib", "peak_total_rss_mib", "peak_rss_mb"),
            ),
            "model_load_delta_mib": _memory_value(
                frame,
                ("model_load_rss_delta_bytes", "rss_load_delta_bytes"),
                ("model_load_rss_delta_mib", "rss_load_delta_mib"),
            ),
            "mps_mib": _memory_value(
                frame,
                ("mps_peak_allocated_bytes", "mps_driver_allocated_bytes"),
                ("mps_peak_allocated_mib", "mps_allocated_mib"),
            ),
        }
    )
    result.loc[result["method"].isin({"", "nan"}), "method"] = "native"
    result.loc[result["device"].isin({"", "nan"}), "device"] = "not recorded"
    return result[columns]


def _as_bool(value: Any) -> bool | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed", "complete"}:
        return True
    if text in {"false", "0", "no", "fail", "failed"}:
        return False
    return np.nan


def _all_known_true(series: pd.Series) -> bool | float:
    known = series.dropna()
    return bool(known.astype(bool).all()) if len(known) else np.nan


def _canonical_parity(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "route",
        "method",
        "n",
        "source_rows",
        "status",
        "candidate_parity",
        "native_top1_parity",
        "reranked_top1_parity",
        "gate_parity",
        "final_top1_parity",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    work = frame.copy()
    work["route"] = _first_column(work, ("route", "route_id")).map(
        _normalise_route
    )
    work["method"] = _first_column(work, ("method", "variant")).map(
        _normalise_method
    )
    work.loc[work["method"].isin({"", "nan"}), "method"] = "gated"

    def bool_col(names: Sequence[str]) -> pd.Series:
        return _first_column(work, names).map(_as_bool)

    work["candidate_parity"] = bool_col(
        ("candidate_parity", "candidate_ids_parity", "candidate_ids_equal")
    )
    work["native_top1_parity"] = bool_col(
        ("native_top1_parity", "native_parity", "top1_equal", "candidate_parity")
    )
    work["reranked_top1_parity"] = bool_col(
        ("reranked_top1_parity", "ranker_parity")
    )
    work["gate_parity"] = bool_col(("gate_parity", "gate_decision_parity"))
    work["final_top1_parity"] = bool_col(
        ("final_top1_parity", "final_parity", "top1_equal", "gate_parity")
    )
    raw_status = _first_column(work, ("status", "parity_status")).astype(str).str.lower()
    work["status"] = raw_status.isin(
        {"true", "pass", "passed", "complete", "parity_complete"}
    )
    source_n = pd.to_numeric(
        _first_column(
            work,
            ("parity_count", "parity_groups_checked", "n", "n_samples"),
        ),
        errors="coerce",
    )
    overall = bool_col(("passed", "parity_passed"))
    for name in columns[4:]:
        work[name] = work[name].where(work[name].notna(), overall)
    records: list[dict[str, Any]] = []
    for (route, method), group in work.groupby(["route", "method"], dropna=False):
        records.append(
            {
                "route": route,
                "method": method,
                "n": int(source_n.loc[group.index].max())
                if source_n.loc[group.index].notna().any()
                else len(group),
                "source_rows": len(group),
                "status": bool(group["status"].all()),
                **{name: _all_known_true(group[name]) for name in columns[5:]},
            }
        )
    return pd.DataFrame(records, columns=columns)


def _variant_key(route: str, method: str) -> tuple[int, int, str, str]:
    route_index = ROUTE_ORDER.index(route) if route in ROUTE_ORDER else len(ROUTE_ORDER)
    method_index = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return route_index, method_index, route, method


def _quantile(series: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.quantile(quantile)) if len(clean) else np.nan


def _nanmax(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.max()) if len(clean) else np.nan


def _first_nonempty(*series: pd.Series) -> str:
    for values in series:
        cleaned = [
            str(value)
            for value in values
            if str(value) not in {"", "nan", "not recorded"}
        ]
        if cleaned:
            return cleaned[0]
    return "not recorded"


def _eligible_timing_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    status = frame["status"].astype(str).str.lower()
    return (
        frame["elapsed_ms"].notna()
        & frame["complete_deployment"].eq(True)
        & status.isin({"measured", "measured_full_deployment"})
    )


def _measured_stage_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    # Some formal APIs expose a measured combined stage without a truthful
    # internal boundary (for example depth/workspace/TSDF construction).  It
    # is still deployment work and must remain in the stage decomposition;
    # only its finer-grained split is not estimable.
    status = frame["status"].astype(str).str.lower()
    return (
        frame["elapsed_ms"].notna()
        & frame["complete_deployment"].eq(True)
        & status.isin({"measured", "combined_formal_api"})
    )


def _eligible_memory_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["complete_deployment"].eq(True) & frame["status"].astype(
        str
    ).str.lower().isin(
        {"complete_deployment_memory", "measured_full_deployment", "measured"}
    )


def _runtime_summary(
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
    stages: pd.DataFrame,
    memory: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute report values from raw timings; missing cells remain NaN."""

    cold_c = _canonical_timing(cold, default_mode="cold")
    disk_c = _canonical_timing(disk, default_mode="warm-disk")
    preloaded_c = _canonical_timing(preloaded, default_mode="warm-preloaded")
    stages_c = _canonical_stages(stages)
    memory_c = _canonical_memory(memory)
    keys: set[tuple[str, str]] = set()
    for frame in (cold_c, disk_c, preloaded_c, stages_c, memory_c):
        if not frame.empty:
            keys.update(
                map(tuple, frame[["route", "method"]].drop_duplicates().to_numpy())
            )
    rows: list[dict[str, Any]] = []
    for route, method in sorted(keys, key=lambda item: _variant_key(*item)):
        cold_all = cold_c[(cold_c.route == route) & (cold_c.method == method)]
        disk_all = disk_c[(disk_c.route == route) & (disk_c.method == method)]
        preload_all = preloaded_c[
            (preloaded_c.route == route) & (preloaded_c.method == method)
        ]
        stage_all = stages_c[
            (stages_c.route == route)
            & (stages_c.method == method)
            & (stages_c["mode"] == "warm-disk")
        ]
        memory_all = memory_c[
            (memory_c.route == route) & (memory_c.method == method)
        ]
        cold_rows = cold_all[_eligible_timing_mask(cold_all)]
        disk_rows = disk_all[_eligible_timing_mask(disk_all)]
        preload_rows = preload_all[_eligible_timing_mask(preload_all)]
        stage_rows = stage_all[_measured_stage_mask(stage_all)]
        memory_rows = memory_all[_eligible_memory_mask(memory_all)]
        overhead_rows = stage_rows[
            stage_rows.stage.str.lower().isin({"reranker", "gate"})
        ]
        per_sample_overhead = (
            overhead_rows.groupby("sample_id", dropna=False)["elapsed_ms"].sum()
            if not overhead_rows.empty
            else pd.Series(dtype=float)
        )
        overhead = (
            float(per_sample_overhead.median())
            if len(per_sample_overhead)
            else np.nan
        )
        disk_median = _quantile(disk_rows.elapsed_ms, 0.5)
        rows.append(
            {
                "route": route,
                "method": method,
                "device": _first_nonempty(
                    disk_all.device,
                    preload_all.device,
                    cold_all.device,
                    memory_all.device,
                ),
                "cold_n": int(cold_rows.elapsed_ms.notna().sum()),
                "cold_median_ms": _quantile(cold_rows.elapsed_ms, 0.5),
                "cold_min_ms": _quantile(cold_rows.elapsed_ms, 0.0),
                "cold_max_ms": _quantile(cold_rows.elapsed_ms, 1.0),
                "warm_disk_n": int(
                    disk_rows.loc[
                        disk_rows.elapsed_ms.notna(), "sample_id"
                    ].nunique()
                ),
                "warm_disk_median_ms": disk_median,
                "warm_disk_p95_ms": _quantile(disk_rows.elapsed_ms, 0.95),
                "warm_preloaded_n": int(
                    preload_rows.loc[
                        preload_rows.elapsed_ms.notna(), "sample_id"
                    ].nunique()
                ),
                "warm_preloaded_median_ms": _quantile(
                    preload_rows.elapsed_ms, 0.5
                ),
                "warm_preloaded_p95_ms": _quantile(
                    preload_rows.elapsed_ms, 0.95
                ),
                "throughput_samples_s": (
                    1000.0 / disk_median
                    if np.isfinite(disk_median) and disk_median > 0
                    else np.nan
                ),
                "reranker_gate_overhead_ms": overhead,
                "overhead_pct": (
                    100.0 * overhead / disk_median
                    if np.isfinite(overhead)
                    and np.isfinite(disk_median)
                    and disk_median > 0
                    else np.nan
                ),
                "peak_rss_mib": _nanmax(memory_rows.peak_rss_mib),
                "model_load_delta_mib": _nanmax(
                    memory_rows.model_load_delta_mib
                ),
                "mps_mib": _nanmax(memory_rows.mps_mib),
            }
        )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for extension in ("pdf", "svg", "png"):
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{stem.name}.", suffix=f".{extension}", dir=stem.parent
        )
        os.close(descriptor)
        try:
            options: dict[str, Any] = {
                "bbox_inches": "tight",
                "format": extension,
            }
            if extension == "png":
                options["dpi"] = 300
            fig.savefig(temporary, **options)
            final = stem.with_suffix(f".{extension}")
            os.replace(temporary, final)
            hashes[extension] = sha256_file(final)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    plt.close(fig)
    return hashes


def _empty_axis(axis: plt.Axes, message: str) -> None:
    axis.set_axis_off()
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )


def _variant_label(route: str, method: str) -> str:
    route_label = (
        "D1 (retrospective)" if route == "D1" else route.replace("_", " ")
    )
    return f"{route_label} — {method}"


def _plot_p50_p95(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    needed = {"warm_disk_median_ms", "warm_disk_p95_ms"}
    usable = (
        summary.dropna(subset=list(needed)).copy()
        if needed.issubset(summary.columns)
        else pd.DataFrame()
    )
    fig, axis = plt.subplots(figsize=(8.0, max(3.4, 0.38 * max(len(usable), 5))))
    if usable.empty:
        _empty_axis(
            axis, "Complete warm-disk deployment latency was not estimable."
        )
    else:
        usable["_key"] = [
            _variant_key(route, method)
            for route, method in zip(usable.route, usable.method)
        ]
        usable = usable.sort_values("_key")
        y = np.arange(len(usable))
        medians = usable["warm_disk_median_ms"].to_numpy(float)
        p95 = usable["warm_disk_p95_ms"].to_numpy(float)
        colours = [
            ROUTE_COLOURS.get(route, PALETTE["grey"]) for route in usable.route
        ]
        axis.hlines(y, medians, p95, color=colours, linewidth=2.2)
        axis.scatter(
            medians,
            y,
            marker="o",
            color=colours,
            edgecolor="black",
            linewidth=0.4,
            label="p50",
        )
        axis.scatter(
            p95,
            y,
            marker="|",
            color=colours,
            s=120,
            linewidth=2,
            label="p95",
        )
        axis.set_yticks(
            y,
            [
                _variant_label(row.route, row.method)
                for row in usable.itertuples()
            ],
        )
        axis.invert_yaxis()
        axis.set_xlabel("Complete warm-disk deployment latency (ms/sample)")
        axis.grid(axis="x", alpha=0.24, linewidth=0.6)
        axis.legend(frameon=False, ncol=2, loc="lower right")
    axis.set_title("End-to-end deployment latency")
    fig.tight_layout()
    return _save_figure(fig, output / "full_runtime_p50_p95")


def _preferred_variant(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    subset = frame[frame["mode"] == mode].copy()
    # Select the highest-preference *measured* variant. Blocked raw/gated
    # placeholders must not hide an eligible native measurement for the route.
    subset = subset[_measured_stage_mask(subset)]
    chosen: list[pd.DataFrame] = []
    for _, group in subset.groupby("route"):
        methods = set(group.method)
        method = next(
            (item for item in ("gated", "raw", "native") if item in methods),
            None,
        )
        if method is not None:
            chosen.append(group[group.method == method])
    return (
        pd.concat(chosen, ignore_index=True)
        if chosen
        else subset.iloc[0:0].copy()
    )


def _plot_stages(
    stages: pd.DataFrame, output: Path, *, mode: str, stem: str
) -> dict[str, str]:
    canonical = _preferred_variant(_canonical_stages(stages), mode)
    canonical = canonical[_measured_stage_mask(canonical)]
    fig, axis = plt.subplots(figsize=(8.5, 4.4))
    if canonical.empty:
        _empty_axis(axis, f"Complete {mode} stage timings were not estimable.")
    else:
        per_sample = canonical.groupby(
            ["route", "method", "sample_id", "stage"], as_index=False
        )["elapsed_ms"].sum()
        values = per_sample.groupby(
            ["route", "method", "stage"], as_index=False
        )["elapsed_ms"].median()
        variants = sorted(
            set(zip(values.route, values.method)),
            key=lambda item: _variant_key(*item),
        )
        stages_order = list(dict.fromkeys(values.stage.astype(str)))
        colour_names = (
            "blue",
            "orange",
            "green",
            "vermillion",
            "purple",
            "sky",
            "yellow",
            "grey",
        )
        x = np.arange(len(variants))
        bottoms = np.zeros(len(variants), dtype=float)
        for index, stage in enumerate(stages_order):
            heights = []
            for route, method in variants:
                found = values[
                    (values.route == route)
                    & (values.method == method)
                    & (values.stage == stage)
                ]
                heights.append(
                    float(found.elapsed_ms.iloc[0]) if len(found) else 0.0
                )
            axis.bar(
                x,
                heights,
                bottom=bottoms,
                color=PALETTE[colour_names[index % len(colour_names)]],
                edgecolor="white",
                linewidth=0.3,
                label=stage.replace("_", " "),
            )
            bottoms += np.asarray(heights)
        axis.set_xticks(
            x,
            [_variant_label(route, method) for route, method in variants],
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("Median measured stage latency (ms/sample)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.legend(
            frameon=False,
            fontsize=7,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
        )
    axis.set_title(f"Stage decomposition: {mode}")
    fig.tight_layout()
    return _save_figure(fig, output / stem)


def _plot_memory(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    usable = (
        summary.dropna(subset=["peak_rss_mib"]).copy()
        if "peak_rss_mib" in summary
        else pd.DataFrame()
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    if usable.empty:
        _empty_axis(axis, "Peak process RSS was not estimable.")
    else:
        usable["_key"] = [
            _variant_key(route, method)
            for route, method in zip(usable.route, usable.method)
        ]
        usable = usable.sort_values("_key")
        labels = [
            _variant_label(row.route, row.method) for row in usable.itertuples()
        ]
        x = np.arange(len(usable))
        axis.bar(
            x,
            usable.peak_rss_mib,
            color=[
                ROUTE_COLOURS.get(route, PALETTE["grey"])
                for route in usable.route
            ],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylabel("Peak recursive process RSS (MiB)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axis.set_title("Deployment memory peak (separate route processes)")
    fig.tight_layout()
    return _save_figure(fig, output / "full_runtime_memory")


def _plot_overhead(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    needed = {"reranker_gate_overhead_ms", "overhead_pct"}
    usable = (
        summary.dropna(subset=list(needed)).copy()
        if needed.issubset(summary.columns)
        else pd.DataFrame()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    if usable.empty:
        for axis in axes:
            _empty_axis(
                axis,
                "End-to-end reranker + gate overhead was not estimable.",
            )
    else:
        usable["_key"] = [
            _variant_key(route, method)
            for route, method in zip(usable.route, usable.method)
        ]
        usable = usable.sort_values("_key")
        x = np.arange(len(usable))
        colours = [
            ROUTE_COLOURS.get(route, PALETTE["grey"]) for route in usable.route
        ]
        labels = [
            _variant_label(row.route, row.method) for row in usable.itertuples()
        ]
        axes[0].bar(
            x,
            usable.reranker_gate_overhead_ms,
            color=colours,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].set_ylabel("Median overhead (ms/sample)")
        axes[1].bar(
            x,
            usable.overhead_pct,
            color=colours,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].set_ylabel("Share of complete warm-disk latency (%)")
        for axis in axes:
            axis.set_xticks(x, labels, rotation=32, ha="right", fontsize=7)
            axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].set_title("Absolute overhead")
    axes[1].set_title("End-to-end share")
    fig.tight_layout()
    return _save_figure(fig, output / "reranker_overhead")


def _plot_cold(cold: pd.DataFrame, output: Path) -> dict[str, str]:
    canonical = _canonical_timing(cold, default_mode="cold")
    usable = canonical[_eligible_timing_mask(canonical)]
    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    if usable.empty:
        _empty_axis(
            axis, "Fresh-process cold-start measurements were not estimable."
        )
    else:
        values = usable.groupby(
            ["route", "method"], as_index=False
        ).elapsed_ms.median()
        values["_key"] = [
            _variant_key(route, method)
            for route, method in zip(values.route, values.method)
        ]
        values = values.sort_values("_key")
        x = np.arange(len(values))
        axis.bar(
            x,
            values.elapsed_ms,
            color=[
                ROUTE_COLOURS.get(route, PALETTE["grey"])
                for route in values.route
            ],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(
            x,
            [
                _variant_label(row.route, row.method)
                for row in values.itertuples()
            ],
            rotation=28,
            ha="right",
        )
        axis.set_ylabel("Cold start to first complete Top-1 (ms)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axis.set_title("Fresh-process cold-start latency")
    fig.tight_layout()
    return _save_figure(fig, output / "full_runtime_cold_start")


def _plot_sixd(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    needed = {"route", "warm_disk_median_ms"}
    usable = (
        summary[
            summary.route.isin(SIX_D_ROUTES)
            & summary.warm_disk_median_ms.notna()
        ].copy()
        if needed.issubset(summary.columns)
        else pd.DataFrame()
    )
    fig, axis = plt.subplots(figsize=(6.5, 3.7))
    if usable.empty:
        _empty_axis(
            axis,
            "Complete 6-DoF oracle/adapted-mask comparison was not estimable.",
        )
    else:
        usable["_key"] = [
            _variant_key(route, method)
            for route, method in zip(usable.route, usable.method)
        ]
        usable = usable.sort_values("_key")
        x = np.arange(len(usable))
        axis.bar(
            x,
            usable.warm_disk_median_ms,
            color=[
                ROUTE_COLOURS.get(route, PALETTE["grey"])
                for route in usable.route
            ],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(
            x,
            [
                _variant_label(row.route, row.method)
                for row in usable.itertuples()
            ],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel("Complete warm-disk latency (ms/sample)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axis.set_title("6-DoF oracle versus adapted-mask deployment")
    fig.tight_layout()
    return _save_figure(fig, output / "sixd_oracle_vs_predicted_latency")


def _plot_candidate_scaling(
    stages: pd.DataFrame, output: Path
) -> dict[str, str]:
    canonical = _canonical_stages(stages)
    canonical = canonical[
        (canonical["mode"] == "warm-disk")
        & canonical["stage"].str.lower().isin(
            {
                "runtime_feature_extraction",
                "feature_extraction",
                "reranker",
            }
        )
        & canonical["candidate_count"].notna()
        & _measured_stage_mask(canonical)
    ]
    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    if canonical.empty:
        _empty_axis(
            axis, "Candidate-count feature/reranker scaling was not estimable."
        )
    else:
        values = canonical.groupby(
            ["route", "method", "sample_id", "candidate_count"], as_index=False
        ).elapsed_ms.sum()
        for route, group in values.groupby("route"):
            axis.scatter(
                group.candidate_count,
                group.elapsed_ms,
                s=14,
                alpha=0.45,
                linewidth=0,
                color=ROUTE_COLOURS.get(route, PALETTE["grey"]),
                label=(
                    "D1 (retrospective)"
                    if route == "D1"
                    else route.replace("_", " ")
                ),
            )
        axis.set_xlabel("Post-NMS candidate count")
        axis.set_ylabel("Feature extraction + reranker latency (ms/sample)")
        axis.legend(frameon=False, fontsize=7)
        axis.grid(alpha=0.2, linewidth=0.5)
    axis.set_title("Candidate count versus reranking-path cost")
    fig.tight_layout()
    return _save_figure(fig, output / "candidate_count_vs_reranking_time")


def _fmt_value(value: Any, template: str = "{:.2f}") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "not estimable"
    return template.format(numeric) if math.isfinite(numeric) else "not estimable"


def _fmt_ms(value: Any) -> str:
    return _fmt_value(value, "{:.2f}")


def _fmt_pct_fraction(numerator: Any, denominator: Any) -> str:
    n = int(numerator)
    d = int(denominator)
    return f"{n}/{d} = {100 * n / d:.2f}%" if d else "not estimable"


def _runtime_latex(summary: pd.DataFrame) -> str:
    if summary.empty:
        body = (
            "\\multicolumn{10}{c}{No complete runtime measurements were "
            "estimable.} \\\\\n"
        )
    else:
        rows = []
        summary = summary.copy()
        summary["_key"] = [
            _variant_key(route, method)
            for route, method in zip(summary.route, summary.method)
        ]
        for row in summary.sort_values("_key").itertuples():
            route = (
                "D1 (retrospective)"
                if row.route == "D1"
                else row.route.replace("_", " ")
            )
            rows.append(
                " & ".join(
                    (
                        route,
                        str(row.method),
                        str(row.device).upper(),
                        _fmt_ms(row.cold_median_ms),
                        _fmt_ms(row.warm_disk_median_ms),
                        _fmt_ms(row.warm_disk_p95_ms),
                        _fmt_ms(row.warm_preloaded_median_ms),
                        _fmt_ms(row.warm_preloaded_p95_ms),
                        _fmt_ms(row.reranker_gate_overhead_ms),
                        _fmt_value(row.peak_rss_mib, "{:.1f}"),
                    )
                )
                + " \\\\"
            )
        body = "\n".join(rows) + "\n"
    return (
        "\\begin{table*}[t]\n\\centering\n\\small\n"
        "\\caption{Complete deployment runtime and memory measurements. "
        "Missing values are reported as not estimable rather than zero. "
        "D1 is retrospective.}\n"
        "\\label{tab:full-runtime-profile}\n"
        "\\begin{tabular}{lllrrrrrrr}\n\\toprule\n"
        "Route & Variant & Device & Cold p50 & Disk p50 & Disk p95 & "
        "Preload p50 & Preload p95 & Rerank+gate & Peak RSS \\\\\n"
        " & & & (ms) & (ms) & (ms) & (ms) & (ms) & (ms) & (MiB) \\\\\n"
        "\\midrule\n"
        + body
        + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def _required_pairs(d1_required: bool) -> set[tuple[str, str]]:
    pairs = {
        (route, method)
        for route in PRIMARY_FOUR_D_ROUTES
        for method in METHOD_ORDER
    }
    pairs.update(
        (route, method)
        for route in SIX_D_ROUTES
        for method in ("native", "raw")
    )
    if d1_required:
        pairs.update(("D1", method) for method in METHOD_ORDER)
    return pairs


def _parity_passes(
    parity: pd.DataFrame, route: str, method: str, expected_n: int
) -> bool:
    matches = parity[
        (parity.route == route) & (parity.method == method)
    ]
    if len(matches) != 1:
        return False
    row = matches.iloc[0]
    if "source_rows" in matches.columns:
        source_rows = pd.to_numeric(
            pd.Series([row.source_rows]), errors="coerce"
        ).iloc[0]
        if pd.isna(source_rows) or float(source_rows) != 1.0:
            return False
    count = pd.to_numeric(pd.Series([row.n]), errors="coerce").iloc[0]
    if pd.isna(count) or float(count) != float(expected_n):
        return False
    required = [
        "status",
        "candidate_parity",
        "native_top1_parity",
        "final_top1_parity",
    ]
    if method in {"raw", "gated"}:
        required.append("reranked_top1_parity")
    if method == "gated":
        required.append("gate_parity")
    for name in required:
        if pd.isna(row[name]) or not bool(row[name]):
            return False
    return True


def _duplicate_complete(run_dir: Path) -> tuple[bool, list[str]]:
    required = (
        "duplicate_audit/DUPLICATE_MAP_LOCK.json",
        "duplicate_audit/duplicate_audit.html",
        "duplicate_audit/duplicate_audit.pdf",
        "duplicate_audit/AI_ASSISTED_AUDIT.md",
        "duplicate_exclusion/full_vs_filtered_metrics.csv",
        "duplicate_exclusion/bootstrap_results.csv",
        "duplicate_exclusion/significance_tests.json",
        "duplicate_exclusion/strict_exclusion_manifest.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return False, [f"duplicate output missing: {name}" for name in missing]
    try:
        lock = json.loads(
            (run_dir / "duplicate_audit/DUPLICATE_MAP_LOCK.json").read_text()
        )
        metrics = pd.read_csv(
            run_dir / "duplicate_exclusion/full_vs_filtered_metrics.csv"
        )
    except (json.JSONDecodeError, OSError, pd.errors.ParserError) as error:
        return False, [f"duplicate output could not be validated: {error}"]
    blockers = []
    if lock.get("status") != "LOCKED":
        blockers.append("duplicate map is not locked")
    expected_subsets = {
        "full",
        "exact_excluded",
        "strict_excluded",
        "moderate_excluded",
    }
    if "subset" not in metrics or not expected_subsets.issubset(
        set(metrics["subset"])
    ):
        blockers.append("duplicate evaluation does not contain all four locked subsets")
    return not blockers, blockers


def assess_completion(
    run_dir: Path,
    runtime_summary: pd.DataFrame,
    parity: pd.DataFrame,
    *,
    d1_required: bool,
) -> CompletionAssessment:
    """Apply the user's COMPLETE gate without inferring absent measurements."""

    duplicate_ok, blockers = _duplicate_complete(run_dir)
    test_report = run_dir / "test_report.txt"
    if not test_report.is_file() or "OVERALL_STATUS=PASS" not in test_report.read_text(
        encoding="utf-8"
    ):
        blockers.append(
            "required regression/static checks are not recorded as PASS in test_report.txt"
        )
    contracts_path = run_dir / "runtime_full/ROUTE_CONTRACTS.md"
    contracts = (
        contracts_path.read_text(encoding="utf-8")
        if contracts_path.is_file()
        else ""
    )
    if (
        "did **not**\nserialize any LightGBM Booster" in contracts
        or "NOT_MEASURABLE_MISSING_FORMAL_CHECKPOINT" in contracts
    ):
        blockers.append(
            "6D raw runtime blocked: the formal LambdaMART Booster was not "
            "serialized, and frozen-score replay is not deployment inference"
        )
    summary_pairs = set(
        zip(runtime_summary.get("route", []), runtime_summary.get("method", []))
    )
    for route, method in sorted(
        _required_pairs(d1_required), key=lambda item: _variant_key(*item)
    ):
        if (route, method) not in summary_pairs:
            blockers.append(f"full runtime missing: {route}/{method}")
            continue
        row = runtime_summary[
            (runtime_summary.route == route) & (runtime_summary.method == method)
        ].iloc[0]
        expected = EXPECTED_WARM_SAMPLES[route]
        if int(row.warm_disk_n) < expected:
            blockers.append(
                f"warm-disk runtime incomplete: {route}/{method} "
                f"({int(row.warm_disk_n)}/{expected})"
            )
        if int(row.warm_preloaded_n) < expected:
            blockers.append(
                f"warm-preloaded runtime incomplete: {route}/{method} "
                f"({int(row.warm_preloaded_n)}/{expected})"
            )
        if int(row.cold_n) < 5:
            blockers.append(
                f"cold-start repetitions incomplete: {route}/{method} "
                f"({int(row.cold_n)}/5)"
            )
        if not np.isfinite(float(row.peak_rss_mib)):
            blockers.append(f"peak RSS missing: {route}/{method}")
        expected_parity = 20
        if not _parity_passes(parity, route, method, expected_parity):
            blockers.append(
                f"{expected_parity}-sample live parity not demonstrated: "
                f"{route}/{method}"
            )
    if d1_required and not {
        ("D1", method) for method in METHOD_ORDER
    }.issubset(summary_pairs):
        blockers.append(
            "D1 is an independently executable retrospective deployment route, "
            "but its full native/raw/gated profile is incomplete"
        )
    runtime_prefixes = (
        "full runtime",
        "warm-",
        "cold-",
        "peak RSS",
        "20-sample",
        "6D raw runtime",
        "D1 is an independently",
        "required regression",
    )
    runtime_ok = not any(
        blocker.startswith(runtime_prefixes) for blocker in blockers
    )
    emitted = duplicate_ok and runtime_ok
    return CompletionAssessment(
        status=(
            "COMPLETE_REMAINING_ROBUSTNESS_EXPERIMENTS"
            if emitted
            else "PARTIAL_REMAINING_ROBUSTNESS_EXPERIMENTS"
        ),
        emitted=emitted,
        duplicate_complete=duplicate_ok,
        runtime_complete=runtime_ok,
        blockers=tuple(blockers),
    )


def _d1_required(route_contracts: Path) -> bool:
    # Fail closed if the audit is missing or ambiguous. The current audit
    # explicitly establishes an independently executable D1 backend.
    if not route_contracts.is_file():
        return True
    text = route_contracts.read_text(encoding="utf-8")
    if "NOT_A_SEPARATE_DEPLOYMENT_ROUTE" in text:
        return False
    return "independently executable" in text or "HiFi-CS → D1" in text


def _parent_findings(parent: Path) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    folds = _read_csv(parent / "6d_scene_cv/fold_results.csv")
    if not folds.empty:
        findings["sixd"] = {
            "folds_positive": int(
                (folds["delta_p_at_1_mu_1.2_pp"] > 0).sum()
            ),
            "folds_n": len(folds),
            "mean_pp": float(folds["delta_p_at_1_mu_1.2_pp"].mean()),
        }
    oof = _read_csv(parent / "6d_scene_cv/oof_metrics.csv")
    if not oof.empty:
        pooled = oof[oof["scope"] == "pooled_oof"]
        if pooled.empty and "fold" in oof:
            pooled = oof[oof["fold"].isna()]
        native = pooled[pooled.system == "native"]
        raw = pooled[pooled.system == "raw_lambdamart"]
        if len(native) and len(raw):
            native_row = native.iloc[0]
            raw_row = raw.iloc[0]
            findings.setdefault("sixd", {}).update(
                {
                    "n": int(native_row["n_groups"]),
                    "native_num": int(native_row["p_at_1_mu_1.2_n"]),
                    "raw_num": int(raw_row["p_at_1_mu_1.2_n"]),
                    "delta_pp": 100
                    * (
                        float(raw_row["p_at_1_mu_1.2"])
                        - float(native_row["p_at_1_mu_1.2"])
                    ),
                }
            )
    threshold = _read_csv(parent / "4d_threshold_sensitivity/results.csv")
    if not threshold.empty:
        findings["threshold"] = {
            "positive": int((threshold.delta_pp > 0).sum()),
            "n": len(threshold),
            "min_pp": float(threshold.delta_pp.min()),
            "max_pp": float(threshold.delta_pp.max()),
        }
    topk = _read_csv(parent / "4d_topk_sensitivity/ranker_results.csv")
    if not topk.empty:
        k5 = topk[topk.k == 5]
        findings["topk"] = {
            "routes": int(k5.route.nunique()),
            "absence_min": float(k5.candidate_absence_rate.min()),
            "absence_max": float(k5.candidate_absence_rate.max()),
            "headroom_min": float(k5.headroom_recovery.min()),
            "headroom_max": float(k5.headroom_recovery.max()),
        }
    return findings


def _duplicate_paragraphs(
    run_dir: Path,
) -> tuple[str, str, pd.DataFrame, Mapping[str, Any]]:
    lock = json.loads(
        (run_dir / "duplicate_audit/DUPLICATE_MAP_LOCK.json").read_text()
    )
    metrics = pd.read_csv(
        run_dir / "duplicate_exclusion/full_vs_filtered_metrics.csv"
    )
    counts = lock["counts"]
    if int(counts.get("strict_excluded_observations", 0)) == 0:
        interpretation = (
            "The independently preregistered high-precision RGB-D protocol "
            "found no exact, strict, or moderate train–test matches among "
            "6,944 outcome-blind retrieved candidate pairs. Consequently, all "
            "filtered subsets equal the full set and every gain shift is zero "
            "by identity. This result is non-informative about sensitivity to "
            "duplicate exclusion: it does not demonstrate duplicate robustness "
            "or unseen-scene generalisation."
        )
    else:
        interpretation = (
            "The strict tier removed observations before outcomes were loaded. "
            "The filtered paired estimate and its sequence-cluster gain-shift "
            "interval quantify sensitivity to the locked high-similarity "
            "definition; they do not establish unseen-scene generalisation."
        )
    legacy = (
        "The previous approximate count could not be reproduced. A new "
        "independently preregistered high-precision RGB-D near-duplicate audit "
        "was therefore conducted."
        if not lock.get("legacy_map_recovered", False)
        else "The legacy map was recovered with immutable provenance and was "
        "analysed separately."
    )
    return interpretation, legacy, metrics, lock


def _runtime_markdown(summary: pd.DataFrame) -> str:
    header = (
        "| Route | Variant | Device | Cold p50 | Warm disk p50 / p95 | "
        "Warm preloaded p50 / p95 | Reranker + gate | Overhead | Peak RSS |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    if summary.empty:
        return (
            header
            + "| not estimable | — | — | — | — | — | — | — | — |\n"
        )
    summary = summary.copy()
    summary["_key"] = [
        _variant_key(route, method)
        for route, method in zip(summary.route, summary.method)
    ]
    lines = []
    for row in summary.sort_values("_key").itertuples():
        route = (
            "D1 (retrospective)"
            if row.route == "D1"
            else row.route.replace("_", " ")
        )
        lines.append(
            f"| {route} | {row.method} | {str(row.device).upper()} | "
            f"{_fmt_ms(row.cold_median_ms)} ms | "
            f"{_fmt_ms(row.warm_disk_median_ms)} / "
            f"{_fmt_ms(row.warm_disk_p95_ms)} ms | "
            f"{_fmt_ms(row.warm_preloaded_median_ms)} / "
            f"{_fmt_ms(row.warm_preloaded_p95_ms)} ms | "
            f"{_fmt_ms(row.reranker_gate_overhead_ms)} ms | "
            f"{_fmt_value(row.overhead_pct)}% | "
            f"{_fmt_value(row.peak_rss_mib, '{:.1f}')} MiB |"
        )
    return header + "\n".join(lines) + "\n"


def _stage_bottlenecks(stages: pd.DataFrame) -> str:
    canonical = _canonical_stages(stages)
    canonical = canonical[
        (canonical["mode"] == "warm-disk") & _measured_stage_mask(canonical)
    ]
    if canonical.empty:
        return "- Stage bottlenecks were not estimable from complete stage timings."
    per_sample = canonical.groupby(
        ["route", "method", "sample_id", "stage"], as_index=False
    ).elapsed_ms.sum()
    medians = per_sample.groupby(
        ["route", "method", "stage"], as_index=False
    ).elapsed_ms.median()
    lines = []
    for (route, method), group in medians.groupby(["route", "method"]):
        top = group.nlargest(3, "elapsed_ms")
        values = ", ".join(
            f"{row.stage.replace('_', ' ')} ({row.elapsed_ms:.2f} ms)"
            for row in top.itertuples()
        )
        lines.append(f"- {_variant_label(route, method)}: {values}.")
    return "\n".join(lines)


def _duplicate_markdown(metrics: pd.DataFrame) -> str:
    selected = metrics[
        (metrics.analysis_set == "route_specific")
        & metrics.subset.isin({"full", "strict_excluded"})
    ].copy()
    header = (
        "| Route | Subset | N | Native J@1 | Raw J@1 | Gated J@1 | "
        "Raw gain | Recovered / harmful | Gain shift (95% CI) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = []
    for row in selected.sort_values(["route", "subset"]).itertuples():
        route = "D1 (retrospective)" if row.route == "D1" else row.route
        lines.append(
            f"| {route} | {row.subset} | {int(row.n_tuples)} | "
            f"{_fmt_pct_fraction(row.native_num, row.n_tuples)} | "
            f"{_fmt_pct_fraction(row.raw_num, row.n_tuples)} | "
            f"{_fmt_pct_fraction(row.gated_num, row.n_tuples)} | "
            f"{row.raw_gain_pp:+.2f} pp | "
            f"{int(row.raw_recovered)} / {int(row.raw_harmful)} | "
            f"{row.raw_gain_shift_pp:+.2f} pp "
            f"[{row.raw_gain_shift_ci_low_pp:+.2f}, "
            f"{row.raw_gain_shift_ci_high_pp:+.2f}] |"
        )
    return header + "\n".join(lines) + "\n"


def _scaled(findings: Mapping[str, Any], key: str) -> float:
    try:
        return 100.0 * float(findings[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _combined_summary(
    assessment: CompletionAssessment,
    parent_findings: Mapping[str, Any],
    duplicate_interpretation: str,
    legacy: str,
    duplicate_metrics: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    stage_timings: pd.DataFrame,
    sequence: Mapping[str, Any],
) -> str:
    sixd = parent_findings.get("sixd", {})
    threshold = parent_findings.get("threshold", {})
    topk = parent_findings.get("topk", {})
    blockers = (
        "\n".join(f"- {item}" for item in assessment.blockers) or "- None."
    )
    runtime_conclusion = (
        "established for every required route/variant under the locked profile "
        "protocol"
        if assessment.runtime_complete
        else "not established for every required route/variant; see the blockers"
    )
    return f"""# Combined robustness-completion summary

This document links the three completed analyses in parent run `{PARENT_RUN_ID}`
to the two new **post-hoc robustness completion** analyses in child run `{RUN_ID}`.
It does not replace or relabel any locked primary result.

## 1. 6-DoF scene-grouped stability

Raw LambdaMART was positive in {sixd.get('folds_positive', 'not estimable')}/
{sixd.get('folds_n', 'not estimable')} outer folds. The pooled oracle-mask OOF
effect was {_fmt_value(sixd.get('delta_pp'))} pp
({sixd.get('native_num', 'not estimable')} native and
{sixd.get('raw_num', 'not estimable')} raw successes among
{sixd.get('n', 'not estimable')} groups). The parent suite's scene-cluster
inference supports split stability only for the locked oracle-mask subset.

## 2. 4-DoF evaluator-threshold sensitivity

{threshold.get('positive', 'not estimable')}/
{threshold.get('n', 'not estimable')} locked route/threshold cells had a positive
effect; magnitudes ranged from {_fmt_value(threshold.get('min_pp'))} to
{_fmt_value(threshold.get('max_pp'))} pp. Direction was not tied to the formal
0.25/30° cell, although magnitude remained evaluator-dependent.

## 3. Independent near-duplicate audit

{legacy}

{duplicate_interpretation}

{_duplicate_markdown(duplicate_metrics)}

All {sequence.get('overlapping_test_sequence_count', 'not estimable')}/
{sequence.get('test_sequence_count', 'not estimable')} test sequences also appeared
in training. Sequence-disjoint evaluation is therefore **not estimable**,
independently of the frame-level near-duplicate result.

## 4. 4-DoF Top-K sensitivity

At K=5, valid-candidate absence ranged from
{_fmt_value(_scaled(topk, 'absence_min'))}% to
{_fmt_value(_scaled(topk, 'absence_max'))}% across routes, while raw rankers
recovered {_fmt_value(_scaled(topk, 'headroom_min'))}% to
{_fmt_value(_scaled(topk, 'headroom_max'))}% of available headroom. Top-5 is a
reasonable locked comparison point, but grounding/candidate generation remains
the dominant residual limitation for routes with large Oracle@5 absence;
ordering remains a smaller, non-zero limitation.

## 5. Complete deployment runtime and memory

{_runtime_markdown(runtime_summary)}

Only rows backed by fresh-input, cache-disabled route execution and a passed
parity gate are eligible as complete deployment measurements. Missing entries
are not zero. Component replay timings from the parent run remain component-only
evidence and are not used as end-to-end latency.

### Largest measured stages

{_stage_bottlenecks(stage_timings)}

## 6. Integrated answer

- **Scene stability:** supported for the 6-DoF oracle-mask raw ranker in the locked 30-scene subset.
- **Threshold stability:** direction was stable over the fixed 3×3 grid; magnitude was threshold-dependent.
- **Duplicate sensitivity:** the zero-match exclusion was non-informative, not evidence of duplicate robustness.
- **Sequence generalisation:** not supported because every formal test sequence overlaps training.
- **Candidate coverage:** Top-K evidence identifies candidate absence as a major ceiling for G1, C1 and retrospective D1.
- **Computational practicality:** {runtime_conclusion}.

## Final status: {assessment.status}

`formal_remaining_results_emitted={'true' if assessment.emitted else 'false'}`

### Blocking conditions

{blockers}
"""


def _paper_insert(
    assessment: CompletionAssessment,
    interpretation: str,
    legacy: str,
    runtime_summary: pd.DataFrame,
) -> str:
    needed = {"warm_disk_median_ms", "warm_disk_p95_ms"}
    measured = (
        runtime_summary.dropna(subset=list(needed))
        if needed.issubset(runtime_summary.columns)
        else pd.DataFrame()
    )
    if measured.empty:
        runtime_sentence = (
            "No route satisfied the complete parity-gated end-to-end protocol, "
            "so deployment p50/p95 and the end-to-end reranker overhead remain "
            "not estimable."
        )
    else:
        runtime_sentence = (
            f"Complete warm-disk measurements were available for {len(measured)} "
            "route variants; Table~\\ref{tab:full-runtime-profile} reports "
            "p50/p95 latency, incremental reranking cost, and separately sampled "
            "process RSS."
        )
    return f"""# Paper insert: remaining robustness experiments

## Suggested main-text paragraphs

**Near-duplicate sensitivity.** {legacy} {interpretation} Exclusion was defined
at the visual-observation level, so all language queries for any matched RGB-D
frame would have been removed together. Sequence-cluster bootstrap intervals
used 10,000 replicates with seed 20260815. D1 is reported only as a retrospective
route.

**Deployment runtime.** We profiled fresh-process cold start, warm local-disk
input, and warm preloaded-input compute at batch size one. Frozen candidate,
feature, and score caches were disabled in deployment timing and used only for
output parity. Whole-pipeline timers were separate from stage instrumentation,
and offline evaluator work was excluded. {runtime_sentence}

## Tables and figures

- Main text: `tables/near_duplicate_exclusion.tex`,
  `figures/near_duplicate_full_vs_filtered.pdf`, and
  `figures/near_duplicate_gain_shift.pdf`.
- Main text when complete runtime rows exist: `tables/full_runtime_profile.tex`,
  `figures/full_runtime_p50_p95.pdf`, `figures/full_runtime_stages.pdf`,
  `figures/full_runtime_memory.pdf`, and `figures/reranker_overhead.pdf`.
- Appendix: the duplicate visual audit, exact/moderate descriptive tiers,
  warm-preloaded stage plot, cold-start plot, candidate-scaling plot, raw timing
  distributions, route parity details, and memory tables.

## Claims not supported

- The audit does not establish unseen-scene or sequence-disjoint generalisation.
- Zero locked duplicate matches do not demonstrate robustness to duplicate exclusion.
- D1 is not a primary prospective route.
- Component-only replay is not full-pipeline latency.
- No route is called real-time without an explicit target frequency and complete p95 evidence.

## Emission status

`{assessment.status}`;
`formal_remaining_results_emitted={'true' if assessment.emitted else 'false'}`.

The dissertation source is intentionally not modified automatically.
"""


def _limitations(assessment: CompletionAssessment, interpretation: str) -> str:
    blockers = (
        "\n".join(f"- {item}" for item in assessment.blockers) or "- None."
    )
    return f"""# Limitations update

- These are post-hoc robustness and sensitivity analyses; locked primary results are unchanged.
- {interpretation}
- All formal 4-DoF test sequences overlap training, so sequence-disjoint evaluation is not estimable.
- Strict RGB-D thresholds prioritise precision; visually related observations outside the locked retrieval set or thresholds may remain.
- D1 is retrospective and must not be pooled with primary prospective routes without that label.
- Runtime rows are eligible only after live output parity. Frozen-score/candidate replay remains component-only evidence.
- Process RSS is distinct from Mac unified-memory use. Unavailable MPS/unified-memory values remain not directly measurable.
- No real-time claim follows without an explicit frequency target and complete p95 latency.

## Outstanding completion blockers

{blockers}
"""


def _reproduce() -> str:
    return f"""# Reproduce the robustness-completion child run

Run from the repository root. The preregistration and duplicate-map lock are
immutable; use `--resume` to verify or continue existing stages.

```bash
PYTHONPATH=src .venv-graspnet6d/bin/python -m robustness_completion.cli audit --run-id {RUN_ID}
PYTHONPATH=src .venv-graspnet6d/bin/python -m robustness_completion.cli preregister --run-id {RUN_ID}
PYTHONPATH=src HiFi_reproduction/.venv-grasp4dof/bin/python -m robustness_completion.cli build-duplicate-map --run-id {RUN_ID} --resume
PYTHONPATH=src HiFi_reproduction/.venv-grasp4dof/bin/python -m robustness_completion.cli audit-duplicate-map --run-id {RUN_ID} --resume
PYTHONPATH=src HiFi_reproduction/.venv-grasp4dof/bin/python -m robustness_completion.cli evaluate-duplicate-exclusion --run-id {RUN_ID} --resume
PYTHONPATH=src HiFi_reproduction/.venv-grasp4dof/bin/python -m robustness_completion.cli report-duplicate-results --run-id {RUN_ID} --resume
PYTHONPATH=src .venv-graspnet6d/bin/python -m robustness_completion.cli build-runtime-adapters --run-id {RUN_ID} --resume
PYTHONPATH=src .venv-graspnet6d/bin/python -m robustness_completion.cli validate-runtime-parity --run-id {RUN_ID} --resume
PYTHONPATH=src .venv-graspnet6d/bin/python -m robustness_completion.cli profile-runtime-full --run-id {RUN_ID} --resume
PYTHONPATH=src .venv-graspnet6d/bin/python -m robustness_completion.cli report --run-id {RUN_ID}
```

Run ID: `{RUN_ID}`. Parent run: `{PARENT_RUN_ID}`. Full route workers execute in
their locked virtual environments and separate subprocesses. Runtime aggregation
must retain the raw CSV, Parquet, parity, and memory records supporting each value.
The exact fresh-process worker commands and output destinations used for this run
are retained verbatim in `runtime_full/monitors/**.monitor.json`; diagnostic
attempts remain separately labelled and are excluded from formal aggregation.
"""


def _final_status(assessment: CompletionAssessment) -> str:
    blockers = (
        "\n".join(f"- {item}" for item in assessment.blockers) or "- None."
    )
    return f"""# {assessment.status}

formal_remaining_results_emitted={'true' if assessment.emitted else 'false'}

This is an independent post-hoc robustness-completion child run. It does not
alter the parent suite or any locked formal result.

## Stage status

- near-duplicate audit and exclusion: {'COMPLETE' if assessment.duplicate_complete else 'PARTIAL'}
- complete end-to-end runtime/memory profile: {'COMPLETE' if assessment.runtime_complete else 'PARTIAL'}

## Blocking conditions

{blockers}
"""


def _hash_existing(paths: Iterable[Path], base: Path) -> dict[str, str]:
    return {
        str(path.relative_to(base)): sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _completion_source_paths(
    run_dir: Path, parent: Path, runtime: Path
) -> list[Path]:
    return [
        run_dir / "PRE_REGISTRATION.md",
        run_dir / "duplicate_audit/DUPLICATE_MAP_LOCK.json",
        run_dir / "duplicate_exclusion/full_vs_filtered_metrics.csv",
        parent / "6d_scene_cv/fold_results.csv",
        parent / "4d_threshold_sensitivity/results.csv",
        parent / "4d_topk_sensitivity/ranker_results.csv",
        runtime / "route_parity.csv",
        runtime / "cold_start_raw.csv",
        runtime / "warm_disk_raw.parquet",
        runtime / "warm_preloaded_raw.parquet",
        runtime / "stage_timings.parquet",
        runtime / "memory_summary.csv",
        runtime / "runtime_summary.csv",
        runtime / "runtime_completion.json",
        runtime / "4D_PARITY_FAILURE_AUDIT.md",
        run_dir / "source_integrity_final.json",
        run_dir / "COMMAND_LOG.md",
        run_dir / "test_report.txt",
    ]


def _atomic_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".csv", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def generate_completion_report(repo: Path, run_dir: Path) -> dict[str, Any]:
    """Generate all artifacts while preserving fail-closed status semantics."""

    repo = repo.expanduser().resolve()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    parent = repo / PARENT_RELATIVE
    runtime = run_dir / "runtime_full"
    _verify_runtime_completion(runtime, repo)
    cold_raw = _read_csv(runtime / "cold_start_raw.csv")
    disk_raw = _read_parquet(runtime / "warm_disk_raw.parquet")
    preloaded_raw = _read_parquet(runtime / "warm_preloaded_raw.parquet")
    stage_raw = _read_parquet(runtime / "stage_timings.parquet")
    memory_raw = _read_csv(runtime / "memory_summary.csv")
    parity_raw = _read_csv(runtime / "route_parity.csv")
    canonical_parity = _canonical_parity(parity_raw)
    summary = _runtime_summary(
        cold_raw, disk_raw, preloaded_raw, stage_raw, memory_raw
    )
    # Preserve the orchestrator-owned runtime_summary.csv.  The reporter keeps
    # its independent raw-derived values alongside the paper table so a report
    # can never erase a richer benchmark summary or turn a partial run into a
    # seemingly complete one.
    derived_summary_path = runtime / "report_derived_summary.csv"
    _atomic_frame_csv(derived_summary_path, summary)

    d1_required = _d1_required(runtime / "ROUTE_CONTRACTS.md")
    assessment = assess_completion(
        run_dir,
        summary,
        canonical_parity,
        d1_required=d1_required,
    )
    (
        duplicate_interpretation,
        legacy,
        duplicate_metrics,
        duplicate_lock,
    ) = _duplicate_paragraphs(run_dir)
    sequence_path = run_dir / "duplicate_exclusion/sequence_overlap_diagnostic.json"
    sequence = (
        json.loads(sequence_path.read_text()) if sequence_path.is_file() else {}
    )
    parent_findings = _parent_findings(parent)

    figures = run_dir / "figures"
    figure_hashes = {
        "full_runtime_p50_p95": _plot_p50_p95(summary, figures),
        "full_runtime_stages": _plot_stages(
            stage_raw,
            figures,
            mode="warm-disk",
            stem="full_runtime_stages",
        ),
        "full_runtime_preloaded_stages": _plot_stages(
            stage_raw,
            figures,
            mode="warm-preloaded",
            stem="full_runtime_preloaded_stages",
        ),
        "full_runtime_cold_start": _plot_cold(cold_raw, figures),
        "reranker_overhead": _plot_overhead(summary, figures),
        "full_runtime_memory": _plot_memory(summary, figures),
        "sixd_oracle_vs_predicted_latency": _plot_sixd(summary, figures),
        "candidate_count_vs_reranking_time": _plot_candidate_scaling(
            stage_raw, figures
        ),
    }
    table_path = run_dir / "tables/full_runtime_profile.tex"
    atomic_text(table_path, _runtime_latex(summary))
    documents = {
        "COMBINED_SUMMARY.md": _combined_summary(
            assessment,
            parent_findings,
            duplicate_interpretation,
            legacy,
            duplicate_metrics,
            summary,
            stage_raw,
            sequence,
        ),
        "PAPER_INSERT_COMPLETION.md": _paper_insert(
            assessment, duplicate_interpretation, legacy, summary
        ),
        "LIMITATIONS_UPDATE.md": _limitations(
            assessment, duplicate_interpretation
        ),
        "REPRODUCE.md": _reproduce(),
        "FINAL_STATUS.md": _final_status(assessment),
    }
    for name, text in documents.items():
        atomic_text(run_dir / name, text)

    figure_manifest = {
        "palette": "Okabe-Ito",
        "data_sources": {
            "latency": "runtime_full/warm_disk_raw.parquet",
            "preloaded_latency": "runtime_full/warm_preloaded_raw.parquet",
            "cold_start": "runtime_full/cold_start_raw.csv",
            "stages": "runtime_full/stage_timings.parquet",
            "memory": "runtime_full/memory_summary.csv",
        },
        "figures": figure_hashes,
        "missing_values": "rendered as not estimable; never zero-filled",
    }
    figure_manifest_path = run_dir / "completion_figures_manifest.json"
    atomic_json(figure_manifest_path, figure_manifest)

    output_paths = [
        *(run_dir / name for name in documents),
        table_path,
        derived_summary_path,
        figure_manifest_path,
        *(
            figures / f"{stem}.{extension}"
            for stem in figure_hashes
            for extension in ("pdf", "svg", "png")
        ),
    ]
    source_paths = _completion_source_paths(run_dir, parent, runtime)
    result_manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN_ID,
        "analysis_type": "post-hoc robustness completion",
        "status": assessment.status,
        "formal_remaining_results_emitted": assessment.emitted,
        "duplicate_complete": assessment.duplicate_complete,
        "runtime_complete": assessment.runtime_complete,
        "d1_required_as_separate_deployment_route": d1_required,
        "blockers": list(assessment.blockers),
        "legacy_duplicate_map_recovered": bool(
            duplicate_lock.get("legacy_map_recovered", False)
        ),
        "strict_excluded_observations": int(
            duplicate_lock["counts"].get("strict_excluded_observations", 0)
        ),
        "source_sha256": _hash_existing(source_paths, repo),
        "output_sha256": _hash_existing(output_paths, run_dir),
    }
    atomic_json(run_dir / "completion_results_manifest.json", result_manifest)
    return result_manifest


__all__ = [
    "CompletionAssessment",
    "assess_completion",
    "generate_completion_report",
]
