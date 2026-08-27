"""Aggregate parity-gated deployment workers into locked runtime tables.

This module never executes a model.  It accepts only structured worker and
parent-monitor outputs, rejects diagnostic files from formal timing, and emits
explicit null-valued blocker rows when a route cannot be measured.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import (
    PREREGISTRATION_SHA256,
    RUN_ID,
    atomic_frame,
    atomic_json,
    atomic_text,
    canonical_sha256,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)


class RuntimeAggregationError(RuntimeError):
    """Raised when structured runtime evidence violates its locked contract."""


@dataclass(frozen=True)
class Variant:
    route: str
    worker_route: str
    method: str
    device: str
    expected_samples: int
    retrospective: bool = False


FOUR_D_ROUTES = ("CROG", "G1", "C1")
FOUR_D_METHODS = ("native", "raw", "gated")
VARIANTS = tuple(
    Variant(route, route.lower(), method, "mps", 100)
    for route in FOUR_D_ROUTES
    for method in FOUR_D_METHODS
) + tuple(
    Variant("D1", "d1", method, "cpu (Docker amd64 emulation)", 100, True)
    for method in FOUR_D_METHODS
) + tuple(
    Variant(route, worker, method, "cpu", 98)
    for route, worker in (("6D_ORACLE", "oracle"), ("6D_ADAPTED", "adapted"))
    for method in ("native", "raw")
)

OUTPUT_NAMES = (
    "route_parity.csv",
    "cold_start_raw.csv",
    "warm_disk_raw.parquet",
    "warm_preloaded_raw.parquet",
    "stage_timings.parquet",
    "memory_samples.parquet",
    "runtime_summary.csv",
    "memory_summary.csv",
    "RUNTIME_METHODS.md",
    "RUNTIME_LIMITATIONS.md",
)

PARITY_COLUMNS = (
    "route",
    "method",
    "retrospective",
    "device",
    "parity_n",
    "n",
    "parity_count",
    "status",
    "candidate_parity",
    "native_top1_parity",
    "runtime_feature_parity",
    "reranked_top1_parity",
    "gate_parity",
    "final_top1_parity",
    "failure_count",
    "blocker_code",
    "blocker",
    "source_path",
    "source_sha256",
)

WHOLE_COLUMNS = (
    "route",
    "method",
    "retrospective",
    "mode",
    "device",
    "sample_id",
    "sample_index",
    "scene_id",
    "status",
    "complete_deployment",
    "whole_elapsed_ns",
    "candidate_count",
    "candidate_bytes",
    "feature_bytes",
    "worker_segment_id",
    "blocker_code",
    "blocker",
    "source_path",
    "source_sha256",
)

COLD_COLUMNS = (
    "route",
    "method",
    "retrospective",
    "mode",
    "device",
    "repetition",
    "sample_id",
    "status",
    "complete_deployment",
    "process_spawn_to_output_ns",
    "model_load_ns",
    "first_complete_output_ns",
    "cold_inner_ns",
    "candidate_count",
    "checkpoint_bytes",
    "peak_total_rss_bytes",
    "pid",
    "blocker_code",
    "blocker",
    "worker_source_path",
    "worker_source_sha256",
    "monitor_source_path",
    "monitor_source_sha256",
)

STAGE_COLUMNS = (
    "route",
    "method",
    "retrospective",
    "mode",
    "device",
    "sample_id",
    "sample_index",
    "scene_id",
    "stage",
    "elapsed_ns",
    "status",
    "stage_status",
    "combined_members",
    "candidate_count",
    "instrumented_wall_ns",
    "whole_elapsed_ns",
    "instrumentation_overhead_ns",
    "worker_segment_id",
    "blocker_code",
    "blocker",
    "source_path",
    "source_sha256",
)

MEMORY_SAMPLE_COLUMNS = (
    "route",
    "method",
    "retrospective",
    "mode",
    "device",
    "repetition",
    "timestamp_ns",
    "elapsed_since_spawn_ns",
    "rss_total_bytes",
    "recursive_process_count",
    "child_pid",
    "eligible_deployment_memory",
    "status",
    "source_path",
    "source_sha256",
)


def _relative(path: Path, repo: Path) -> str:
    return str(path.resolve().relative_to(repo.resolve()))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeAggregationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeAggregationError(f"expected JSON object: {path}")
    return value


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _frame(records: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    rows = list(records)
    if not rows:
        return _empty(columns)
    result = pd.DataFrame(rows)
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result.loc[:, list(columns)]


def _bool_all(values: Iterable[Any]) -> bool:
    materialised = [bool(value) for value in values]
    return bool(materialised) and all(materialised)


def _variant(route: str, method: str) -> Variant:
    for item in VARIANTS:
        if item.route == route and item.method == method:
            return item
    raise KeyError((route, method))


def _route_from_monitor(value: str) -> tuple[str, str]:
    text = str(value)
    if text.startswith("6d_oracle_"):
        return "6D_ORACLE", text.removeprefix("6d_oracle_")
    if text.startswith("6d_adapted_"):
        return "6D_ADAPTED", text.removeprefix("6d_adapted_")
    route, _, method = text.partition("_")
    return route.upper(), method.lower()


def _source_inventory(runtime_root: Path, repo: Path) -> dict[str, str]:
    """Hash structured inputs only; generated aggregate outputs are excluded."""

    inputs: list[Path] = []
    workers = runtime_root / "workers"
    monitors = runtime_root / "monitors"
    if workers.is_dir():
        inputs.extend(workers.rglob("*.json"))
        inputs.extend(workers.rglob("*.parquet"))
    if monitors.is_dir():
        inputs.extend(monitors.rglob("*.monitor.json"))
        inputs.extend(monitors.rglob("*.memory.parquet"))
    for name in (
        "profile_subset_manifest.json",
        "ROUTE_CONTRACTS.md",
        "environment_grasp4dof.json",
        "environment_graspnet6d.json",
    ):
        path = runtime_root / name
        if path.is_file():
            inputs.append(path)
    return {
        _relative(path, repo): sha256_file(path)
        for path in sorted(set(inputs))
        if path.is_file()
    }


def _input_signature(inventory: Mapping[str, str]) -> str:
    module_root = Path(__file__).resolve().parent
    return canonical_sha256(
        {
            "schema_version": 2,
            "run_id": RUN_ID,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "implementation_sha256": {
                name: sha256_file(module_root / name)
                for name in (
                    "runtime_aggregation.py",
                    "runtime_4d.py",
                    "runtime_6d.py",
                    "runtime_common.py",
                )
            },
            "inputs": dict(inventory),
        }
    )


def _validate_monitor(monitor_path: Path, repo: Path) -> dict[str, Any]:
    monitor = _read_json(monitor_path)
    samples_path_text = monitor.get("memory_samples_path")
    if samples_path_text:
        samples_path = Path(str(samples_path_text)).resolve()
        runtime_root = (
            repo / "artifacts/robustness_completion" / RUN_ID / "runtime_full"
        ).resolve()
        if runtime_root not in samples_path.parents:
            raise RuntimeAggregationError("monitor memory path escapes child runtime root")
        if not samples_path.is_file():
            raise RuntimeAggregationError(f"monitor memory samples missing: {samples_path}")
        expected = monitor.get("memory_samples_sha256")
        if expected and sha256_file(samples_path) != expected:
            raise RuntimeAggregationError(
                f"monitor memory hash mismatch: {samples_path}"
            )
    return monitor


def _blocker_for(variant: Variant, parity_status: Mapping[str, str]) -> tuple[str, str]:
    if variant.route in {"CROG", "G1", "C1"}:
        detail = parity_status.get(variant.route, "FAILED_PARITY")
        return (
            "FAILED_PARITY",
            f"{variant.route} live parity did not pass ({detail}); formal timing is prohibited",
        )
    if variant.route == "D1":
        return (
            "NOT_EXECUTED_IMPLEMENTATION_INCOMPLETE",
            "D1 is an independently deployable retrospective route, but its thin adapter is incomplete",
        )
    if variant.method == "raw":
        return (
            "MISSING_FORMAL_CHECKPOINT",
            "formal 6D three-seed LambdaMART Booster ensemble was not serialized; retraining and cached-score timing are prohibited",
        )
    return "MISSING_REQUIRED_RUNTIME_OUTPUT", "required runtime output is absent"


def _four_d_parity(
    runtime_root: Path, repo: Path, subset: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    expected_ids = list(subset.get("parity_4d", {}).get("sample_ids", []))
    for route in FOUR_D_ROUTES:
        path = runtime_root / "workers/4d" / route.lower() / "parity.json"
        if not path.is_file():
            statuses[route] = "MISSING_PARITY_OUTPUT"
            for method in FOUR_D_METHODS:
                variant = _variant(route, method)
                code, blocker = _blocker_for(variant, statuses)
                rows.append(
                    {
                        "route": route,
                        "method": method,
                        "retrospective": False,
                        "device": "mps",
                        "parity_n": 0,
                        "status": "MISSING_PARITY_OUTPUT",
                        "blocker_code": code,
                        "blocker": blocker,
                    }
                )
            continue
        value = _read_json(path)
        parity_rows = value.get("parity_rows", [])
        if not isinstance(parity_rows, list):
            raise RuntimeAggregationError(f"invalid parity rows: {path}")
        observed_ids = [str(row.get("sample_id")) for row in parity_rows]
        count = int(value.get("parity_count", len(parity_rows)))
        ids_exact = count == 20 and observed_ids == expected_ids
        candidate = ids_exact and bool(value.get("candidate_parity", False))
        features = ids_exact and bool(value.get("feature_parity", False))
        ranker = ids_exact and bool(value.get("ranker_parity", False))
        gate = ids_exact and bool(value.get("gate_parity", False))
        passed = (
            str(value.get("status")) == "PASS"
            and value.get("complete_deployment") is True
            and ids_exact
            and candidate
            and features
            and ranker
            and gate
        )
        status = "PASS" if passed else "FAILED_PARITY"
        statuses[route] = status
        source_hash = sha256_file(path)
        failure_count = int(value.get("failure_count", 0 if passed else count))
        for method in FOUR_D_METHODS:
            variant = _variant(route, method)
            code, blocker = ("", "") if passed else _blocker_for(variant, statuses)
            method_pass = passed
            rows.append(
                {
                    "route": route,
                    "method": method,
                    "retrospective": False,
                    "device": "mps",
                    "parity_n": count,
                    "status": "PASS" if method_pass else status,
                    "candidate_parity": candidate,
                    "native_top1_parity": candidate,
                    "runtime_feature_parity": features
                    if method in {"raw", "gated"}
                    else np.nan,
                    "reranked_top1_parity": ranker
                    if method in {"raw", "gated"}
                    else np.nan,
                    "gate_parity": gate if method == "gated" else np.nan,
                    "final_top1_parity": (
                        candidate
                        if method == "native"
                        else ranker
                        if method == "raw"
                        else gate
                    ),
                    "failure_count": failure_count,
                    "blocker_code": code or np.nan,
                    "blocker": blocker or np.nan,
                    "source_path": _relative(path, repo),
                    "source_sha256": source_hash,
                }
            )
    return rows, statuses


def _six_d_native_parity(
    runtime_root: Path,
    repo: Path,
    subset: Mapping[str, Any],
    route: str,
    worker_route: str,
) -> dict[str, Any]:
    variant = _variant(route, "native")
    root = runtime_root / "workers/6d" / worker_route / "native"
    path = root / "parity_results.parquet"
    status_path = root / "parity_result.json"
    expected = list(subset.get("parity_6d", {}).get("group_ids", []))
    if not path.is_file() or not status_path.is_file():
        code, blocker = _blocker_for(variant, {})
        return {
            "route": route,
            "method": "native",
            "retrospective": False,
            "device": "cpu",
            "parity_n": 0,
            "status": "MISSING_PARITY_OUTPUT",
            "blocker_code": code,
            "blocker": blocker,
        }
    frame = pd.read_parquet(path)
    value = _read_json(status_path)
    observed = frame.get("group_id", pd.Series(dtype=str)).astype(str).tolist()
    exact = len(frame) == 20 and observed == expected
    candidate = exact and _bool_all(frame["candidate_ids_equal"])
    native_top1 = exact and _bool_all(frame["top1_equal"])
    features = exact and _bool_all(frame["features_equal"])
    passed = (
        exact
        and _bool_all(frame["passed"])
        and value.get("parity_complete") is True
        and value.get("parity_passed") is True
        and str(value.get("status")) == "PARITY_COMPLETE"
        and value.get("preregistration_sha256") == PREREGISTRATION_SHA256
        and value.get("worker_implementation_sha256")
        == sha256_file(repo / "src/robustness_completion/runtime_6d.py")
    )
    return {
        "route": route,
        "method": "native",
        "retrospective": False,
        "device": "cpu",
        "parity_n": len(frame),
        "status": "PASS" if passed else "FAILED_PARITY",
        "candidate_parity": candidate,
        "native_top1_parity": native_top1,
        "runtime_feature_parity": features,
        "reranked_top1_parity": np.nan,
        "gate_parity": np.nan,
        "final_top1_parity": native_top1,
        "failure_count": int((~frame["passed"].astype(bool)).sum()),
        "blocker_code": np.nan if passed else "FAILED_PARITY",
        "blocker": np.nan
        if passed
        else f"{route} native 20-group parity did not pass",
        "source_path": _relative(path, repo),
        "source_sha256": sha256_file(path),
    }


def _parity_table(
    runtime_root: Path, repo: Path, subset: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, four_d_status = _four_d_parity(runtime_root, repo, subset)
    for route, worker_route in (
        ("6D_ORACLE", "oracle"),
        ("6D_ADAPTED", "adapted"),
    ):
        native = _six_d_native_parity(
            runtime_root, repo, subset, route, worker_route
        )
        rows.append(native)
        raw_variant = _variant(route, "raw")
        code, blocker = _blocker_for(raw_variant, four_d_status)
        blocker_path = runtime_root / "workers/6d" / worker_route / "raw" / "parity_result.json"
        if not blocker_path.is_file() and route == "6D_ADAPTED":
            blocker_path = runtime_root / "workers/6d/oracle/raw/parity_result.json"
        rows.append(
            {
                "route": route,
                "method": "raw",
                "retrospective": False,
                "device": "cpu",
                "parity_n": 0,
                "status": "NOT_MEASURABLE_MISSING_FORMAL_CHECKPOINT",
                "failure_count": 0,
                "blocker_code": code,
                "blocker": blocker,
                "source_path": _relative(blocker_path, repo)
                if blocker_path.is_file()
                else np.nan,
                "source_sha256": sha256_file(blocker_path)
                if blocker_path.is_file()
                else np.nan,
            }
        )
    for method in FOUR_D_METHODS:
        variant = _variant("D1", method)
        code, blocker = _blocker_for(variant, four_d_status)
        rows.append(
            {
                "route": "D1",
                "method": method,
                "retrospective": True,
                "device": variant.device,
                "parity_n": 0,
                "status": code,
                "failure_count": 0,
                "blocker_code": code,
                "blocker": blocker,
            }
        )
    result = _frame(rows, PARITY_COLUMNS)
    result["n"] = result["parity_n"]
    result["parity_count"] = result["parity_n"]
    return result.loc[:, PARITY_COLUMNS], four_d_status


def _parity_pass(parity: pd.DataFrame, route: str, method: str) -> bool:
    row = parity[(parity.route == route) & (parity.method == method)]
    return bool(len(row) == 1 and row.iloc[0]["status"] == "PASS")


def _blocker_whole(
    variant: Variant,
    mode: str,
    code: str,
    blocker: str,
) -> dict[str, Any]:
    return {
        "route": variant.route,
        "method": variant.method,
        "retrospective": variant.retrospective,
        "mode": mode,
        "device": variant.device,
        "status": code,
        "complete_deployment": False,
        "blocker_code": code,
        "blocker": blocker,
    }


def _blocker_cold(
    variant: Variant, code: str, blocker: str
) -> dict[str, Any]:
    return {
        "route": variant.route,
        "method": variant.method,
        "retrospective": variant.retrospective,
        "mode": "cold",
        "device": variant.device,
        "status": code,
        "complete_deployment": False,
        "blocker_code": code,
        "blocker": blocker,
    }


def _blocker_stage(
    variant: Variant, mode: str, code: str, blocker: str
) -> dict[str, Any]:
    return {
        "route": variant.route,
        "method": variant.method,
        "retrospective": variant.retrospective,
        "mode": mode,
        "device": variant.device,
        "stage": "not_measured",
        "status": code,
        "stage_status": code,
        "blocker_code": code,
        "blocker": blocker,
    }


def _monitor_index(runtime_root: Path, repo: Path) -> dict[tuple[str, str, str, int], tuple[Path, dict[str, Any]]]:
    index: dict[tuple[str, str, str, int], tuple[Path, dict[str, Any]]] = {}
    monitors_root = runtime_root / "monitors"
    if not monitors_root.is_dir():
        return index
    for path in sorted(monitors_root.rglob("*.monitor.json")):
        value = _validate_monitor(path, repo)
        route, method = _route_from_monitor(str(value.get("route", "")))
        mode = str(value.get("mode", ""))
        repetition = int(value.get("repetition", 0))
        key = (route, method, mode, repetition)
        if key in index:
            # A repeated failed parity attempt is allowed, but formal deployment
            # modes must have exactly one monitor per repetition.
            if mode != "parity":
                raise RuntimeAggregationError(f"duplicate monitor identity: {key}")
            previous = index[key]
            if int(value.get("elapsed_ns", 0)) > int(previous[1].get("elapsed_ns", 0)):
                index[key] = (path, value)
        else:
            index[key] = (path, value)
    return index


def _cold_six_d(
    runtime_root: Path,
    repo: Path,
    monitor_index: Mapping[tuple[str, str, str, int], tuple[Path, dict[str, Any]]],
    variant: Variant,
) -> list[dict[str, Any]]:
    root = runtime_root / "workers/6d" / variant.worker_route / "native"
    rows: list[dict[str, Any]] = []
    for repetition in range(5):
        worker_path = root / f"cold_result_{repetition}.json"
        monitor_entry = monitor_index.get(
            (variant.route, "native", "cold", repetition)
        )
        if not worker_path.is_file() or monitor_entry is None:
            continue
        worker = _read_json(worker_path)
        monitor_path, monitor = monitor_entry
        if (
            worker.get("complete_deployment") is not True
            or str(worker.get("status")) != "COMPLETE"
            or str(worker.get("mode")) != "cold"
            or int(monitor.get("return_code", 1)) != 0
            or worker.get("worker_implementation_sha256")
            != sha256_file(repo / "src/robustness_completion/runtime_6d.py")
            or int(worker.get("feature_bytes", -1)) != 0
        ):
            continue
        rows.append(
            {
                "route": variant.route,
                "method": "native",
                "retrospective": False,
                "mode": "cold",
                "device": "cpu",
                "repetition": repetition,
                "sample_id": worker.get("sample_id"),
                "status": "MEASURED_FULL_DEPLOYMENT",
                "complete_deployment": True,
                "process_spawn_to_output_ns": int(monitor["elapsed_ns"]),
                "model_load_ns": int(worker["model_load_ns"]),
                "first_complete_output_ns": int(worker["first_complete_output_ns"]),
                "cold_inner_ns": int(worker["cold_inner_ns"]),
                "candidate_count": int(worker.get("candidate_count", 0)),
                "checkpoint_bytes": int(worker.get("checkpoint_bytes", 0)),
                "peak_total_rss_bytes": int(monitor["peak_total_rss_bytes"]),
                "pid": int(worker["pid"]),
                "worker_source_path": _relative(worker_path, repo),
                "worker_source_sha256": sha256_file(worker_path),
                "monitor_source_path": _relative(monitor_path, repo),
                "monitor_source_sha256": sha256_file(monitor_path),
            }
        )
    return rows


def _four_d_result_jsons(runtime_root: Path, route: str) -> list[tuple[Path, dict[str, Any]]]:
    root = runtime_root / "workers/4d" / route.lower()
    results: list[tuple[Path, dict[str, Any]]] = []
    if not root.is_dir():
        return results
    for path in sorted(root.rglob("*.json")):
        if path.name == "parity.json":
            continue
        value = _read_json(path)
        if str(value.get("mode")) in {"cold", "warm-disk", "warm-preloaded"}:
            results.append((path, value))
    return results


def _cold_four_d(
    runtime_root: Path,
    repo: Path,
    monitor_index: Mapping[tuple[str, str, str, int], tuple[Path, dict[str, Any]]],
    variant: Variant,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = [
        (path, value)
        for path, value in _four_d_result_jsons(runtime_root, variant.route)
        if str(value.get("method")) == variant.method
        and str(value.get("mode")) == "cold"
        and value.get("complete_deployment") is True
        and str(value.get("status", "")).startswith("COMPLETE_COLD")
    ]
    for repetition, (worker_path, worker) in enumerate(candidates[:5]):
        monitor_entry = monitor_index.get(
            (variant.route, variant.method, "cold", repetition)
        )
        monitor_path, monitor = monitor_entry if monitor_entry else (None, None)
        results = worker.get("results", [])
        result = results[0] if isinstance(results, list) and results else worker
        rows.append(
            {
                "route": variant.route,
                "method": variant.method,
                "retrospective": False,
                "mode": "cold",
                "device": "mps",
                "repetition": repetition,
                "sample_id": result.get("sample_id"),
                "status": "MEASURED_FULL_DEPLOYMENT",
                "complete_deployment": True,
                "process_spawn_to_output_ns": int(monitor["elapsed_ns"])
                if monitor and int(monitor.get("return_code", 1)) == 0
                else np.nan,
                "model_load_ns": worker.get("startup_model_load_ns"),
                "first_complete_output_ns": worker.get("first_full_top1_ns"),
                "cold_inner_ns": (
                    int(worker.get("startup_model_load_ns", 0))
                    + int(worker.get("first_full_top1_ns", 0))
                ),
                "candidate_count": result.get("candidate_count"),
                "peak_total_rss_bytes": monitor.get("peak_total_rss_bytes")
                if monitor
                else np.nan,
                "worker_source_path": _relative(worker_path, repo),
                "worker_source_sha256": sha256_file(worker_path),
                "monitor_source_path": _relative(monitor_path, repo)
                if monitor_path
                else np.nan,
                "monitor_source_sha256": sha256_file(monitor_path)
                if monitor_path
                else np.nan,
            }
        )
    return rows


def _cold_table(
    runtime_root: Path,
    repo: Path,
    parity: pd.DataFrame,
    parity_status: Mapping[str, str],
    monitor_index: Mapping[tuple[str, str, str, int], tuple[Path, dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        if not _parity_pass(parity, variant.route, variant.method):
            code, blocker = _blocker_for(variant, parity_status)
            rows.append(_blocker_cold(variant, code, blocker))
            continue
        measured = (
            _cold_six_d(runtime_root, repo, monitor_index, variant)
            if variant.route in {"6D_ORACLE", "6D_ADAPTED"}
            else _cold_four_d(runtime_root, repo, monitor_index, variant)
        )
        if measured:
            rows.extend(measured)
        else:
            rows.append(
                _blocker_cold(
                    variant,
                    "MISSING_REQUIRED_RUNTIME_OUTPUT",
                    "five fresh-process cold-start measurements are absent",
                )
            )
    return _frame(rows, COLD_COLUMNS)


def _validate_warm_ids(
    frame: pd.DataFrame, expected_ids: Sequence[str], variant: Variant, mode: str
) -> None:
    if "sample_id" not in frame:
        raise RuntimeAggregationError(
            f"{variant.route}/{variant.method}/{mode} lacks sample_id"
        )
    observed = frame["sample_id"].astype(str).tolist()
    if (
        len(observed) != variant.expected_samples
        or len(set(observed)) != variant.expected_samples
        or set(observed) != set(map(str, expected_ids))
    ):
        raise RuntimeAggregationError(
            f"{variant.route}/{variant.method}/{mode} does not cover the locked "
            f"{variant.expected_samples}-sample subset exactly"
        )


def _normalise_whole(
    source: pd.DataFrame,
    variant: Variant,
    mode: str,
    path: Path,
    repo: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    digest = sha256_file(path)
    for row in source.to_dict("records"):
        elapsed = row.get("whole_elapsed_ns", row.get("uninstrumented_total_ns"))
        if elapsed is None or not math.isfinite(float(elapsed)) or float(elapsed) <= 0:
            raise RuntimeAggregationError(f"non-positive whole timer in {path}")
        rows.append(
            {
                "route": variant.route,
                "method": variant.method,
                "retrospective": variant.retrospective,
                "mode": mode,
                "device": variant.device,
                "sample_id": row.get("sample_id", row.get("group_id")),
                "sample_index": row.get("sample_index"),
                "scene_id": row.get("scene_id"),
                "status": "MEASURED_FULL_DEPLOYMENT",
                "complete_deployment": True,
                "whole_elapsed_ns": int(elapsed),
                "candidate_count": row.get("candidate_count"),
                "candidate_bytes": row.get("candidate_bytes"),
                "feature_bytes": row.get("feature_bytes"),
                "worker_segment_id": row.get("worker_segment_id"),
                "source_path": _relative(path, repo),
                "source_sha256": digest,
            }
        )
    return _frame(rows, WHOLE_COLUMNS)


def _normalise_stages(
    source: pd.DataFrame,
    whole: pd.DataFrame,
    variant: Variant,
    mode: str,
    path: Path,
    repo: Path,
) -> pd.DataFrame:
    digest = sha256_file(path)
    whole_lookup = whole.set_index("sample_id")["whole_elapsed_ns"].to_dict()
    rows: list[dict[str, Any]] = []
    for row in source.to_dict("records"):
        sample_id = str(row.get("sample_id", row.get("group_id")))
        elapsed = row.get("elapsed_ns")
        if elapsed is not None and not pd.isna(elapsed) and float(elapsed) < 0:
            raise RuntimeAggregationError(f"negative stage timer in {path}")
        wall = row.get("instrumented_wall_ns", row.get("instrumented_total_ns"))
        whole_elapsed = whole_lookup.get(sample_id)
        combined = row.get("combined_members", [])
        if isinstance(combined, str):
            combined_text = combined
        else:
            if isinstance(combined, np.ndarray):
                combined = combined.tolist()
            elif combined is None or (isinstance(combined, float) and pd.isna(combined)):
                combined = []
            combined_text = json.dumps(combined, separators=(",", ":"))
        rows.append(
            {
                "route": variant.route,
                "method": variant.method,
                "retrospective": variant.retrospective,
                "mode": mode,
                "device": variant.device,
                "sample_id": sample_id,
                "sample_index": row.get("sample_index"),
                "scene_id": row.get("scene_id"),
                "stage": row.get("stage"),
                "elapsed_ns": elapsed,
                "status": row.get("status", row.get("stage_status", "measured")),
                "stage_status": row.get(
                    "stage_status", row.get("status", "measured")
                ),
                "combined_members": combined_text,
                "candidate_count": row.get("candidate_count"),
                "instrumented_wall_ns": wall,
                "whole_elapsed_ns": whole_elapsed,
                "instrumentation_overhead_ns": (
                    int(wall) - int(whole_elapsed)
                    if wall is not None
                    and not pd.isna(wall)
                    and whole_elapsed is not None
                    and not pd.isna(whole_elapsed)
                    else np.nan
                ),
                "worker_segment_id": row.get("worker_segment_id"),
                "source_path": _relative(path, repo),
                "source_sha256": digest,
            }
        )
    result = _frame(rows, STAGE_COLUMNS)
    if set(result.sample_id.astype(str)) != set(whole.sample_id.astype(str)):
        raise RuntimeAggregationError(f"whole/stage sample IDs disagree: {path}")
    for sample_id, group in result.groupby("sample_id", sort=False):
        walls = pd.to_numeric(group["instrumented_wall_ns"], errors="coerce").dropna()
        elapsed = pd.to_numeric(group["elapsed_ns"], errors="coerce").dropna()
        if not len(walls):
            raise RuntimeAggregationError(
                f"instrumented wall timer missing for {sample_id}: {path}"
            )
        wall = int(walls.iloc[0])
        if not (walls.astype(np.int64) == wall).all():
            raise RuntimeAggregationError(
                f"inconsistent instrumented wall timer for {sample_id}: {path}"
            )
        # Formal workers include an explicit orchestration row, making the
        # stage sum an integrity identity.  The distinct paired overhead is
        # wall minus the separately executed uninstrumented whole timer.
        if int(elapsed.sum()) != wall:
            raise RuntimeAggregationError(
                f"stage sum does not equal instrumented wall for {sample_id}: {path}"
            )
    return result


def _warm_six_d(
    runtime_root: Path,
    repo: Path,
    variant: Variant,
    mode: str,
    expected_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    root = runtime_root / "workers/6d" / variant.worker_route / "native"
    whole_path = root / f"{mode}_whole_timings.parquet"
    stage_path = root / f"{mode}_stage_timings.parquet"
    result_path = root / f"{mode.replace('-', '_')}_result.json"
    if not (whole_path.is_file() and stage_path.is_file() and result_path.is_file()):
        return None
    result = _read_json(result_path)
    if (
        str(result.get("status")) != "COMPLETE"
        or result.get("complete_deployment") is not True
        or int(result.get("timed_samples", -1)) != variant.expected_samples
        or int(result.get("warmup_count", -1)) != 5
        or result.get("candidate_cache_used_for_timing") is not False
        or result.get("feature_cache_used_for_timing") is not False
        or result.get("score_cache_used_for_timing") is not False
        or result.get("worker_implementation_sha256")
        != sha256_file(repo / "src/robustness_completion/runtime_6d.py")
    ):
        raise RuntimeAggregationError(
            f"invalid complete warm worker result: {result_path}"
        )
    source_whole = pd.read_parquet(whole_path)
    feature_bytes = pd.to_numeric(
        source_whole.get("feature_bytes", pd.Series(dtype=float)), errors="coerce"
    )
    if len(feature_bytes) != variant.expected_samples or not feature_bytes.eq(0).all():
        raise RuntimeAggregationError(
            f"native timing unexpectedly materialised runtime features: {whole_path}"
        )
    _validate_warm_ids(source_whole, expected_ids, variant, mode)
    whole = _normalise_whole(source_whole, variant, mode, whole_path, repo)
    source_stages = pd.read_parquet(stage_path)
    feature_stage = source_stages[
        source_stages["stage"].astype(str) == "runtime_feature_extraction"
    ]
    if (
        len(feature_stage) != variant.expected_samples
        or pd.to_numeric(feature_stage["elapsed_ns"], errors="coerce").notna().any()
        or not feature_stage["stage_status"]
        .astype(str)
        .eq("not_applicable_native")
        .all()
    ):
        raise RuntimeAggregationError(
            f"native runtime_feature_extraction must be null/not-applicable: {stage_path}"
        )
    stages = _normalise_stages(source_stages, whole, variant, mode, stage_path, repo)
    return whole, stages


def _warm_four_d(
    runtime_root: Path,
    repo: Path,
    variant: Variant,
    mode: str,
    expected_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    matches = [
        (path, value)
        for path, value in _four_d_result_jsons(runtime_root, variant.route)
        if str(value.get("method")) == variant.method
        and str(value.get("mode")) == mode
        and str(value.get("status")) == "COMPLETE_WARM_FULL_DEPLOYMENT"
        and value.get("complete_deployment") is True
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeAggregationError(
            f"multiple 4D warm outputs: {variant.route}/{variant.method}/{mode}"
        )
    path, result = matches[0]
    if (
        int(result.get("measured_count", -1)) != 100
        or int(result.get("warmup_count", -1)) != 5
    ):
        raise RuntimeAggregationError(f"invalid 4D warm counts: {path}")
    samples = result.get("samples", [])
    timings = result.get("timings", [])
    if not isinstance(samples, list) or not isinstance(timings, list):
        raise RuntimeAggregationError(f"invalid 4D timing arrays: {path}")
    source_whole = pd.DataFrame(samples)
    _validate_warm_ids(source_whole, expected_ids, variant, mode)
    whole = _normalise_whole(source_whole, variant, mode, path, repo)
    stages = _normalise_stages(
        pd.DataFrame(timings), whole, variant, mode, path, repo
    )
    return whole, stages


def _warm_tables(
    runtime_root: Path,
    repo: Path,
    subset: Mapping[str, Any],
    parity: pd.DataFrame,
    parity_status: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    disk_rows: list[pd.DataFrame] = []
    preloaded_rows: list[pd.DataFrame] = []
    stage_rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        for mode, destination in (
            ("warm-disk", disk_rows),
            ("warm-preloaded", preloaded_rows),
        ):
            if not _parity_pass(parity, variant.route, variant.method):
                code, blocker = _blocker_for(variant, parity_status)
                destination.append(
                    _frame([_blocker_whole(variant, mode, code, blocker)], WHOLE_COLUMNS)
                )
                stage_rows.append(
                    _frame([_blocker_stage(variant, mode, code, blocker)], STAGE_COLUMNS)
                )
                continue
            expected_ids = (
                subset["profile_6d"]["group_ids"]
                if variant.route in {"6D_ORACLE", "6D_ADAPTED"}
                else subset["profile_4d"]["sample_ids"]
            )
            measured = (
                _warm_six_d(runtime_root, repo, variant, mode, expected_ids)
                if variant.route in {"6D_ORACLE", "6D_ADAPTED"}
                else _warm_four_d(runtime_root, repo, variant, mode, expected_ids)
            )
            if measured is None:
                code = "MISSING_REQUIRED_RUNTIME_OUTPUT"
                blocker = f"complete {mode} worker output is absent"
                destination.append(
                    _frame([_blocker_whole(variant, mode, code, blocker)], WHOLE_COLUMNS)
                )
                stage_rows.append(
                    _frame([_blocker_stage(variant, mode, code, blocker)], STAGE_COLUMNS)
                )
            else:
                destination.append(measured[0])
                stage_rows.append(measured[1])
    disk = pd.concat(disk_rows, ignore_index=True) if disk_rows else _empty(WHOLE_COLUMNS)
    preloaded = (
        pd.concat(preloaded_rows, ignore_index=True)
        if preloaded_rows
        else _empty(WHOLE_COLUMNS)
    )
    stages = pd.concat(stage_rows, ignore_index=True) if stage_rows else _empty(STAGE_COLUMNS)
    return disk.loc[:, WHOLE_COLUMNS], preloaded.loc[:, WHOLE_COLUMNS], stages.loc[:, STAGE_COLUMNS]


def _memory_samples(
    runtime_root: Path,
    repo: Path,
    parity: pd.DataFrame,
    parity_status: Mapping[str, str],
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    monitor_values: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    seen_variants: set[tuple[str, str]] = set()
    root = runtime_root / "monitors"
    if root.is_dir():
        for monitor_path in sorted(root.rglob("*.monitor.json")):
            monitor = _validate_monitor(monitor_path, repo)
            route, method = _route_from_monitor(str(monitor.get("route", "")))
            if not any(item.route == route and item.method == method for item in VARIANTS):
                continue
            variant = _variant(route, method)
            mode = str(monitor.get("mode", ""))
            repetition = int(monitor.get("repetition", 0))
            key = (route, method, mode, repetition)
            # Retain each attempt's raw samples. Summary peak selection uses the
            # largest value across monitor attempts rather than overwriting.
            monitor_values[
                (route, method, mode, len([item for item in monitor_values if item[:3] == key[:3]]))
            ] = monitor
            memory_path = Path(str(monitor["memory_samples_path"])).resolve()
            source = pd.read_parquet(memory_path)
            if mode == "cold":
                accepted_worker = bool(
                    len(
                        cold[
                            (cold.route == route)
                            & (cold.method == method)
                            & (cold.repetition == repetition)
                            & cold.complete_deployment.eq(True)
                        ]
                    )
                    == 1
                )
            elif mode == "warm-disk":
                accepted_worker = bool(
                    len(
                        disk[
                            (disk.route == route)
                            & (disk.method == method)
                            & disk.complete_deployment.eq(True)
                        ]
                    )
                    == variant.expected_samples
                )
            elif mode == "warm-preloaded":
                accepted_worker = bool(
                    len(
                        preloaded[
                            (preloaded.route == route)
                            & (preloaded.method == method)
                            & preloaded.complete_deployment.eq(True)
                        ]
                    )
                    == variant.expected_samples
                )
            else:
                accepted_worker = False
            worker_bound = accepted_worker
            if route in {"6D_ORACLE", "6D_ADAPTED"} and method == "native":
                worker_route = "oracle" if route == "6D_ORACLE" else "adapted"
                worker_root = (
                    runtime_root / "workers/6d" / worker_route / "native"
                )
                result_path = (
                    worker_root / f"cold_result_{repetition}.json"
                    if mode == "cold"
                    else worker_root / f"{mode.replace('-', '_')}_result.json"
                )
                command = [str(item) for item in monitor.get("command", [])]
                output_from_command: Path | None = None
                if "--output" in command:
                    output_index = command.index("--output") + 1
                    if output_index < len(command):
                        output_from_command = Path(command[output_index]).resolve()
                worker = _read_json(result_path) if result_path.is_file() else {}
                observed_pids = set(
                    pd.to_numeric(
                        source.get("child_pid", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                )
                worker_bound = bool(
                    accepted_worker
                    and output_from_command == result_path.resolve()
                    and worker.get("complete_deployment") is True
                    and str(worker.get("status")) == "COMPLETE"
                    and worker.get("worker_implementation_sha256")
                    == sha256_file(repo / "src/robustness_completion/runtime_6d.py")
                    and observed_pids == {int(worker.get("pid", -1))}
                )
            eligible = (
                mode in {"cold", "warm-disk", "warm-preloaded"}
                and int(monitor.get("return_code", 1)) == 0
                and _parity_pass(parity, route, method)
                and worker_bound
            )
            status = (
                "MEASURED_FULL_DEPLOYMENT"
                if eligible
                else "DIAGNOSTIC_NOT_FORMAL_DEPLOYMENT"
            )
            digest = sha256_file(memory_path)
            for row in source.to_dict("records"):
                rows.append(
                    {
                        "route": route,
                        "method": method,
                        "retrospective": variant.retrospective,
                        "mode": mode,
                        "device": variant.device,
                        "repetition": repetition,
                        "timestamp_ns": row.get("timestamp_ns"),
                        "elapsed_since_spawn_ns": row.get(
                            "elapsed_since_spawn_ns"
                        ),
                        "rss_total_bytes": row.get("rss_total_bytes"),
                        "recursive_process_count": row.get(
                            "recursive_process_count"
                        ),
                        "child_pid": row.get("child_pid"),
                        "eligible_deployment_memory": eligible,
                        "status": status,
                        "source_path": _relative(memory_path, repo),
                        "source_sha256": digest,
                    }
                )
            seen_variants.add((route, method))
    for variant in VARIANTS:
        if (variant.route, variant.method) in seen_variants:
            continue
        code, blocker = _blocker_for(variant, parity_status)
        rows.append(
            {
                "route": variant.route,
                "method": variant.method,
                "retrospective": variant.retrospective,
                "mode": "not_measured",
                "device": variant.device,
                "eligible_deployment_memory": False,
                "status": code,
                "source_path": np.nan,
                "source_sha256": np.nan,
            }
        )
    return _frame(rows, MEMORY_SAMPLE_COLUMNS), monitor_values


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(quantile)) if len(clean) else np.nan


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else np.nan


def _ns_to_ms(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric / 1e6 if math.isfinite(numeric) else np.nan


def _memory_events(runtime_root: Path, variant: Variant) -> list[dict[str, Any]]:
    if variant.route not in {"6D_ORACLE", "6D_ADAPTED"} or variant.method != "native":
        return []
    path = (
        runtime_root
        / "workers/6d"
        / variant.worker_route
        / "native/memory_events.json"
    )
    if not path.is_file():
        return []
    status_path = path.with_name("route_status.json")
    if not status_path.is_file():
        return []
    status = _read_json(status_path)
    expected_worker_hash = sha256_file(
        Path(__file__).with_name("runtime_6d.py")
    )
    if (
        status.get("complete_deployment") is not True
        or str(status.get("status")) != "COMPLETE"
        or status.get("worker_implementation_sha256") != expected_worker_hash
    ):
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise RuntimeAggregationError(f"memory events must be a list: {path}")
    events = [dict(item) for item in value]
    if not events or any(int(item.get("pid", -1)) != int(status["pid"]) for item in events):
        raise RuntimeAggregationError(
            f"memory events are not bound to their accepted worker: {path}"
        )
    return events


def _memory_summary(
    runtime_root: Path,
    memory_samples: pd.DataFrame,
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
    parity: pd.DataFrame,
    parity_status: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        selected = memory_samples[
            (memory_samples.route == variant.route)
            & (memory_samples.method == variant.method)
        ]
        eligible = selected[selected.eligible_deployment_memory.eq(True)]
        diagnostic = selected[
            ~selected.eligible_deployment_memory.eq(True)
            & selected.rss_total_bytes.notna()
        ]
        events = _memory_events(runtime_root, variant)
        before = next(
            (item for item in events if item.get("event") == "before_model_load"),
            {},
        )
        after = next(
            (item for item in events if item.get("event") == "after_model_load"),
            {},
        )
        cold_rows = cold[
            (cold.route == variant.route)
            & (cold.method == variant.method)
            & cold.complete_deployment.eq(True)
        ]
        warm_rows = pd.concat(
            [
                disk[
                    (disk.route == variant.route)
                    & (disk.method == variant.method)
                    & disk.complete_deployment.eq(True)
                ],
                preloaded[
                    (preloaded.route == variant.route)
                    & (preloaded.method == variant.method)
                    & preloaded.complete_deployment.eq(True)
                ],
            ],
            ignore_index=True,
        )
        eligible_modes = set(eligible["mode"].astype(str))
        valid = (
            _parity_pass(parity, variant.route, variant.method)
            and {"cold", "warm-disk", "warm-preloaded"}.issubset(eligible_modes)
            and bool(before)
            and bool(after)
        )
        code, blocker = (
            ("COMPLETE_DEPLOYMENT_MEMORY", "")
            if valid
            else _blocker_for(variant, parity_status)
        )
        before_rss = before.get("rss_bytes")
        after_rss = after.get("rss_bytes")
        checkpoint_bytes = pd.to_numeric(
            cold_rows.get("checkpoint_bytes", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "route": variant.route,
                "method": variant.method,
                "retrospective": variant.retrospective,
                "device": variant.device,
                "status": code,
                "blocker_code": np.nan if valid else code,
                "blocker": np.nan if valid else blocker,
                "peak_total_rss_bytes": _safe_quantile(
                    eligible.rss_total_bytes, 1.0
                ),
                "peak_rss_mib": _safe_quantile(
                    eligible.rss_total_bytes, 1.0
                )
                / (1024**2)
                if not eligible.empty
                else np.nan,
                "rss_before_model_load_bytes": before_rss
                if valid and before_rss is not None
                else np.nan,
                "rss_after_checkpoint_load_bytes": after_rss
                if valid and after_rss is not None
                else np.nan,
                "model_load_rss_delta_bytes": (
                    int(after_rss) - int(before_rss)
                    if valid and before_rss is not None and after_rss is not None
                    else np.nan
                ),
                "model_load_rss_delta_mib": (
                    (int(after_rss) - int(before_rss)) / (1024**2)
                    if valid and before_rss is not None and after_rss is not None
                    else np.nan
                ),
                "mps_peak_allocated_bytes": np.nan,
                "mps_driver_allocated_bytes": np.nan,
                "unified_memory_peak": "not directly measurable",
                "checkpoint_bytes": int(checkpoint_bytes.max())
                if len(checkpoint_bytes)
                else np.nan,
                "candidate_bytes_max": _safe_quantile(
                    warm_rows.get("candidate_bytes", pd.Series(dtype=float)), 1.0
                ),
                "feature_bytes_max": _safe_quantile(
                    warm_rows.get("feature_bytes", pd.Series(dtype=float)), 1.0
                ),
                "memory_sample_count": int(len(eligible)),
                "memory_sampling_interval_ms": 5.0 if not eligible.empty else np.nan,
                "diagnostic_parity_peak_rss_bytes": _safe_quantile(
                    diagnostic.rss_total_bytes, 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _stage_component_median(
    stages: pd.DataFrame, names: set[str]
) -> float:
    selected = stages[
        stages.stage.astype(str).str.lower().isin(names)
        & stages.elapsed_ns.notna()
        & ~stages.stage_status.astype(str).str.lower().str.contains(
            "not_applicable|blocked|missing", regex=True, na=False
        )
    ]
    if selected.empty:
        return np.nan
    per_sample = selected.groupby("sample_id")["elapsed_ns"].sum()
    return float(per_sample.median())


def _stage_component_sum_median(
    stages: pd.DataFrame, names: set[str]
) -> float:
    selected = stages[
        stages.stage.astype(str).str.lower().isin(names)
        & stages.elapsed_ns.notna()
        & ~stages.stage_status.astype(str).str.lower().str.contains(
            "not_applicable|blocked|missing", regex=True, na=False
        )
    ]
    if selected.empty:
        return np.nan
    per_sample_stage = selected.groupby(
        ["sample_id", "stage"], as_index=False
    )["elapsed_ns"].sum()
    required = set(map(str.lower, names))
    observed = per_sample_stage.groupby("sample_id")["stage"].agg(
        lambda values: set(values.astype(str).str.lower())
    )
    complete_ids = observed[observed.map(lambda values: required <= values)].index
    if len(complete_ids) == 0:
        return np.nan
    return float(
        per_sample_stage[
            per_sample_stage.sample_id.isin(complete_ids)
        ].groupby("sample_id")["elapsed_ns"].sum().median()
    )


def _runtime_summary_table(
    cold: pd.DataFrame,
    disk: pd.DataFrame,
    preloaded: pd.DataFrame,
    stages: pd.DataFrame,
    parity: pd.DataFrame,
    memory_summary: pd.DataFrame,
    parity_status: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        parity_row = parity[
            (parity.route == variant.route) & (parity.method == variant.method)
        ].iloc[0]
        cold_rows = cold[
            (cold.route == variant.route)
            & (cold.method == variant.method)
            & cold.complete_deployment.eq(True)
        ]
        disk_rows = disk[
            (disk.route == variant.route)
            & (disk.method == variant.method)
            & disk.complete_deployment.eq(True)
        ]
        preload_rows = preloaded[
            (preloaded.route == variant.route)
            & (preloaded.method == variant.method)
            & preloaded.complete_deployment.eq(True)
        ]
        stage_rows = stages[
            (stages.route == variant.route)
            & (stages.method == variant.method)
            & (stages["mode"] == "warm-disk")
            & stages.elapsed_ns.notna()
        ]
        cold_n = int(len(cold_rows))
        disk_n = int(disk_rows.sample_id.astype(str).nunique())
        preload_n = int(preload_rows.sample_id.astype(str).nunique())
        complete = (
            parity_row["status"] == "PASS"
            and cold_n == 5
            and disk_n == variant.expected_samples
            and preload_n == variant.expected_samples
        )
        combined = stage_rows.combined_members.astype(str).map(
            lambda value: value not in {"", "[]", "nan", "None"}
        )
        stage_profile_complete = bool(len(stage_rows) and not combined.any())
        if complete:
            status = (
                "COMPLETE_DEPLOYMENT_TIMING"
                if stage_profile_complete
                else "COMPLETE_DEPLOYMENT_TIMING_STAGE_COMBINED"
            )
            code = blocker = np.nan
        else:
            code, blocker = _blocker_for(variant, parity_status)
            if parity_row["status"] == "PASS" and code == "MISSING_REQUIRED_RUNTIME_OUTPUT":
                blocker = (
                    f"required measurements incomplete: cold {cold_n}/5, "
                    f"warm-disk {disk_n}/{variant.expected_samples}, "
                    f"warm-preloaded {preload_n}/{variant.expected_samples}"
                )
            status = code
        cold_total = pd.to_numeric(
            cold_rows.process_spawn_to_output_ns, errors="coerce"
        )
        warm_disk = pd.to_numeric(disk_rows.whole_elapsed_ns, errors="coerce")
        warm_preloaded = pd.to_numeric(
            preload_rows.whole_elapsed_ns, errors="coerce"
        )
        overhead_per_sample = (
            stage_rows.groupby("sample_id")["instrumentation_overhead_ns"].first()
            if len(stage_rows)
            else pd.Series(dtype=float)
        )
        reranker_ns = _stage_component_median(stage_rows, {"reranker"})
        gate_ns = _stage_component_median(stage_rows, {"gate"})
        reranker_gate_ns = _stage_component_sum_median(
            stage_rows, {"reranker", "gate"}
        )
        disk_median_ns = _safe_quantile(warm_disk, 0.5)
        memory_row = memory_summary[
            (memory_summary.route == variant.route)
            & (memory_summary.method == variant.method)
        ].iloc[0]
        rows.append(
            {
                "route": variant.route,
                "method": variant.method,
                "retrospective": variant.retrospective,
                "device": variant.device,
                "status": status,
                "blocker_code": code,
                "blocker": blocker,
                "complete_deployment": complete,
                "parity_status": parity_row["status"],
                "parity_n": int(parity_row["parity_n"]),
                "cold_n": cold_n,
                "cold_start_median_ms": _ns_to_ms(_safe_quantile(cold_total, 0.5)),
                "cold_start_min_ms": _ns_to_ms(_safe_quantile(cold_total, 0.0)),
                "cold_start_max_ms": _ns_to_ms(_safe_quantile(cold_total, 1.0)),
                "cold_start_p95_ms": _ns_to_ms(_safe_quantile(cold_total, 0.95)),
                "checkpoint_load_median_ms": _ns_to_ms(
                    _safe_quantile(cold_rows.model_load_ns, 0.5)
                ),
                "first_inference_median_ms": _ns_to_ms(
                    _safe_quantile(cold_rows.first_complete_output_ns, 0.5)
                ),
                "warm_disk_n": disk_n,
                "warm_disk_median_ms": _ns_to_ms(_safe_quantile(warm_disk, 0.5)),
                "warm_disk_p95_ms": _ns_to_ms(_safe_quantile(warm_disk, 0.95)),
                "warm_preloaded_n": preload_n,
                "warm_preloaded_median_ms": _ns_to_ms(
                    _safe_quantile(warm_preloaded, 0.5)
                ),
                "warm_preloaded_p95_ms": _ns_to_ms(
                    _safe_quantile(warm_preloaded, 0.95)
                ),
                "throughput_samples_s": (
                    1e9 / disk_median_ns
                    if np.isfinite(disk_median_ns) and disk_median_ns > 0
                    else np.nan
                ),
                "reranker_median_ms": _ns_to_ms(reranker_ns),
                "gate_median_ms": _ns_to_ms(gate_ns),
                "reranker_gate_overhead_ms": _ns_to_ms(reranker_gate_ns),
                "reranker_gate_overhead_pct": (
                    100 * reranker_gate_ns / disk_median_ns
                    if np.isfinite(reranker_gate_ns)
                    and np.isfinite(disk_median_ns)
                    and disk_median_ns > 0
                    else np.nan
                ),
                # This signed paired delta is intentionally not clamped.
                "instrumentation_overhead_median_ns": _safe_quantile(
                    overhead_per_sample, 0.5
                ),
                "instrumentation_overhead_p95_ns": _safe_quantile(
                    overhead_per_sample, 0.95
                ),
                "stage_profile_complete": stage_profile_complete,
                "offline_evaluator_excluded": True,
                "mean_candidate_count": _safe_mean(disk_rows.candidate_count),
                "checkpoint_bytes": memory_row["checkpoint_bytes"],
                "peak_rss_mib": memory_row["peak_rss_mib"],
                "model_load_rss_delta_mib": memory_row[
                    "model_load_rss_delta_mib"
                ],
                "mps_peak_allocated_bytes": memory_row[
                    "mps_peak_allocated_bytes"
                ],
                "unified_memory_peak": memory_row["unified_memory_peak"],
            }
        )
    return pd.DataFrame(rows)


def _methods_markdown(summary: pd.DataFrame, memory_samples: pd.DataFrame) -> str:
    complete = summary[summary.complete_deployment.eq(True)]
    complete_labels = ", ".join(
        f"{row.route}/{row.method}" for row in complete.itertuples()
    ) or "none"
    return f"""# Complete runtime aggregation methods

This file describes a **post-hoc deployment benchmark**. It does not change any
locked accuracy result.

## Source eligibility

- Model execution occurred only in the route workers. Aggregation does not run a model.
- Candidate, feature and score caches were disabled in timed workers and used only for parity.
- A 4-DoF worker is ineligible for formal timing unless its locked 20-tuple live parity status is `PASS`.
- The two 6-DoF native CPU routes require exact 20-group parity plus exactly 98 unique locked profile groups.
- Native 6-DoF deployment ends at native Top-1 selection; runtime feature extraction, reranking and gating are explicit null/not-applicable stages rather than timed work.
- Diagnostic files containing `_diagnostic_` are never discovered as formal timings.
- Blocked routes receive explicit rows with null timing fields; null is never converted to zero.

Complete route variants found: {complete_labels}.

## Timing definitions

- Cold start is the parent monitor's process-spawn-to-worker-exit elapsed time for a fresh process. Five repetitions are required. Checkpoint load and first complete inference are retained as separate child-worker fields.
- Warm disk-input latency is the uninstrumented whole-pipeline timer for each locked sample, including local input decode.
- Warm preloaded compute uses the same live computation with input arrays loaded before the timed region; it never uses frozen candidates.
- Stage timings are a second instrumented execution. For every included sample, the sum of stage rows is asserted equal to `instrumented_wall_ns`.
- `instrumentation_overhead_ns` is the signed paired delta `instrumented_wall_ns - whole_elapsed_ns`. It may be negative because the two live executions are subject to run-to-run noise. It is not clamped and is not treated as a fixed framework cost.
- Offline evaluators, metrics, bootstrap and report generation are excluded from deployment totals.

## Memory

The parent sampled recursive child-process RSS every 5 ms. The aggregate contains
{len(memory_samples):,} raw monitor rows, including separately labelled diagnostic
parity samples. Only cold/warm samples behind a passed parity gate are eligible for
deployment peak RSS. Worker `before_model_load` and `after_model_load` events are
used for model-load RSS deltas where available. Process RSS is distinct from Mac
unified memory; unavailable unified/MPS peaks remain `not directly measurable`.
"""


def _limitations_markdown(summary: pd.DataFrame) -> str:
    blockers = summary[~summary.complete_deployment.eq(True)]
    blocker_lines = "\n".join(
        f"- {row.route}/{row.method}: {row.blocker}"
        for row in blockers.itertuples()
    ) or "- None."
    return f"""# Full-runtime limitations

- CROG, G1 and C1 timings are not formal deployment results if live parity failed; diagnostic parity process time and RSS are not substituted.
- D1 is a retrospective but independently executable route. Its incomplete adapter remains a blocker and component-only numbers are not reported as full latency.
- The formal 6-DoF run did not serialize the three-seed LambdaMART Booster ensemble. Raw reranker timing is therefore not measurable without prohibited retraining or cached-score replay.
- The retained 6-DoF formal APIs combine HiFi mask inference/postprocessing and depth/workspace/TSDF construction. Whole-pipeline timing is valid, while the corresponding internal stage decomposition is explicitly marked combined and incomplete.
- Warm-disk timings use the local OS page-cache state as observed; no unsupported cache-flush mechanism was used.
- Parent-monitor RSS is a sampled process metric, not a direct unified-memory peak.
- No real-time claim is made because no control-frequency target was preregistered.

## Route blockers

{blocker_lines}
"""


def _validate_subset(subset: Mapping[str, Any]) -> None:
    parity_4d = list(subset.get("parity_4d", {}).get("sample_ids", []))
    parity_6d = list(subset.get("parity_6d", {}).get("group_ids", []))
    profile_4d = list(subset.get("profile_4d", {}).get("sample_ids", []))
    profile_6d = list(subset.get("profile_6d", {}).get("group_ids", []))
    expected = ((parity_4d, 20), (parity_6d, 20), (profile_4d, 100), (profile_6d, 98))
    if any(len(values) != count or len(set(values)) != count for values, count in expected):
        raise RuntimeAggregationError("profile subset manifest violates locked counts/uniqueness")


def _validate_resume(
    completion: Mapping[str, Any], runtime_root: Path, input_signature: str
) -> bool:
    if completion.get("input_signature") != input_signature:
        return False
    outputs = completion.get("output_sha256", {})
    if not isinstance(outputs, dict):
        raise RuntimeAggregationError("runtime completion output hashes are invalid")
    if set(outputs) != set(OUTPUT_NAMES):
        raise RuntimeAggregationError(
            "runtime completion does not bind the exact required output set"
        )
    for name, expected in outputs.items():
        path = runtime_root / str(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeAggregationError(
                f"runtime aggregate output changed with unchanged inputs: {name}"
            )
    return True


def aggregate_runtime(
    repo: Path, run_dir: Path, resume: bool = True
) -> dict[str, Any]:
    """Aggregate all structured runtime evidence into the required outputs."""

    repo = repo.expanduser().resolve()
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    runtime_root = run_dir / "runtime_full"
    completion_path = runtime_root / "runtime_completion.json"
    inventory_before = _source_inventory(runtime_root, repo)
    input_signature = _input_signature(inventory_before)
    if completion_path.is_file():
        previous = _read_json(completion_path)
        if not resume:
            raise RuntimeAggregationError(
                "runtime aggregation already exists; pass resume=True or use a new run"
            )
        if _validate_resume(previous, runtime_root, input_signature):
            return {**previous, "resumed": True}

    subset_path = runtime_root / "profile_subset_manifest.json"
    if not subset_path.is_file():
        raise RuntimeAggregationError("locked runtime profile subset is missing")
    subset = _read_json(subset_path)
    _validate_subset(subset)
    parity, parity_status = _parity_table(runtime_root, repo, subset)
    monitor_index = _monitor_index(runtime_root, repo)
    cold = _cold_table(
        runtime_root, repo, parity, parity_status, monitor_index
    )
    disk, preloaded, stages = _warm_tables(
        runtime_root, repo, subset, parity, parity_status
    )
    memory_samples, _ = _memory_samples(
        runtime_root,
        repo,
        parity,
        parity_status,
        cold,
        disk,
        preloaded,
    )
    memory_summary = _memory_summary(
        runtime_root,
        memory_samples,
        cold,
        disk,
        preloaded,
        parity,
        parity_status,
    )
    summary = _runtime_summary_table(
        cold,
        disk,
        preloaded,
        stages,
        parity,
        memory_summary,
        parity_status,
    )
    # The source evidence must remain bit-for-bit unchanged during aggregation.
    if _source_inventory(runtime_root, repo) != inventory_before:
        raise RuntimeAggregationError("runtime source evidence changed during aggregation")

    frames = {
        "route_parity.csv": parity,
        "cold_start_raw.csv": cold,
        "warm_disk_raw.parquet": disk,
        "warm_preloaded_raw.parquet": preloaded,
        "stage_timings.parquet": stages,
        "memory_samples.parquet": memory_samples,
        "runtime_summary.csv": summary,
        "memory_summary.csv": memory_summary,
    }
    for name, frame in frames.items():
        atomic_frame(runtime_root / name, frame)
    atomic_text(
        runtime_root / "RUNTIME_METHODS.md",
        _methods_markdown(summary, memory_samples),
    )
    atomic_text(
        runtime_root / "RUNTIME_LIMITATIONS.md",
        _limitations_markdown(summary),
    )
    output_hashes = {
        name: sha256_file(runtime_root / name) for name in OUTPUT_NAMES
    }
    complete_variants = [
        f"{row.route}/{row.method}"
        for row in summary[summary.complete_deployment.eq(True)].itertuples()
    ]
    blockers = [
        {
            "route": row.route,
            "method": row.method,
            "code": row.blocker_code,
            "reason": row.blocker,
        }
        for row in summary[~summary.complete_deployment.eq(True)].itertuples()
    ]
    all_complete = len(complete_variants) == len(VARIANTS)
    completion = {
        "schema_version": 2,
        "run_id": RUN_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "status": "COMPLETE_RUNTIME_FULL_PROFILE"
        if all_complete
        else "PARTIAL_RUNTIME_FULL_PROFILE",
        "complete": all_complete,
        "input_signature": input_signature,
        "source_sha256": inventory_before,
        "output_sha256": output_hashes,
        "complete_route_variants": complete_variants,
        "blockers": blockers,
        "counts": {
            "route_variants": len(VARIANTS),
            "complete_route_variants": len(complete_variants),
            "parity_rows": len(parity),
            "cold_rows": len(cold),
            "warm_disk_measured_rows": int(disk.complete_deployment.eq(True).sum()),
            "warm_preloaded_measured_rows": int(
                preloaded.complete_deployment.eq(True).sum()
            ),
            "stage_rows": len(stages),
            "memory_samples": len(memory_samples),
        },
        "resumed": False,
    }
    atomic_json(completion_path, completion)
    return completion


__all__ = ["RuntimeAggregationError", "aggregate_runtime"]
