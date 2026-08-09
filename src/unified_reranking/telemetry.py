"""Shared, explicit telemetry contracts for reranking artifacts."""

from __future__ import annotations

import math
import resource
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .artifacts import load_verified_json, verify_artifact_records_recursive
from .hashing import sha256_file


TELEMETRY_FIELDS = (
    "parameter_count",
    "ranker_latency_ms",
    "feature_latency_ms",
    "peak_memory_mb",
    "missing_feature_rate",
)

FEATURE_EXTRACTION_LATENCY_FIELD = "feature_extraction_latency_ms"
T3_ASSEMBLY_LATENCY_FIELD = "feature_track_assembly_latency_ms"


def t3_composite_extraction_latency_ms(
    benchmark: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> float:
    """Compose end-to-end T3 latency from its measured source components.

    T3 is the matched-common track plus fresh tri-backend dense evidence.  Its
    comparable extraction latency is therefore the route's persisted common
    and RGB extraction, the fixed-128 dense benchmark, and the current T3
    assembly.  Keeping these terms explicit avoids reporting the dense-only
    benchmark as if it were the complete superset track.
    """

    route = str(feature_manifest.get("route", "")).lower()
    if route not in {"crog", "g1", "c1"}:
        raise RuntimeError("T3 feature manifest route is invalid")
    assembly = feature_manifest.get(T3_ASSEMBLY_LATENCY_FIELD)
    if (
        not isinstance(assembly, (int, float))
        or not math.isfinite(float(assembly))
        or float(assembly) < 0
    ):
        raise RuntimeError("T3 feature-track assembly latency is invalid")
    dense = benchmark.get(FEATURE_EXTRACTION_LATENCY_FIELD)
    if (
        not isinstance(dense, (int, float))
        or not math.isfinite(float(dense))
        or float(dense) <= 0
    ):
        raise RuntimeError("T3 dense extraction benchmark latency is invalid")
    measurements = benchmark.get("component_measurements")
    if not isinstance(measurements, list):
        raise RuntimeError("T3 benchmark component measurements are missing")
    wanted = {f"common/{route}", f"rgb/{route}"}
    observed: dict[str, float] = {}
    for row in measurements:
        if not isinstance(row, Mapping) or str(row.get("name", "")) not in wanted:
            continue
        name = str(row["name"])
        latency = row.get(FEATURE_EXTRACTION_LATENCY_FIELD)
        if name in observed:
            raise RuntimeError(f"T3 benchmark duplicates component {name}")
        if (
            not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0
        ):
            raise RuntimeError(f"T3 benchmark component {name} is invalid")
        observed[name] = float(latency)
    if set(observed) != wanted:
        raise RuntimeError("T3 benchmark lacks route-specific common/RGB timing")
    return float(dense) + float(assembly) + sum(observed.values())


def resolved_track_extraction_latency(
    run_dir: str | Path,
    track: str,
    feature_manifest: Mapping[str, Any],
) -> tuple[float, dict[str, str] | None]:
    """Resolve direct T1/T2 timing or the locked T3 benchmark timing.

    T3 extraction includes common and RGB evidence, all three dense backends,
    and T3 assembly.  Its dense term is measured by the fixed-128 Validation
    benchmark and its other terms are hash-bound persisted measurements.  Bind
    the benchmark bytes so resume identities cannot silently retain stale
    timing provenance.
    """

    direct = feature_manifest.get(FEATURE_EXTRACTION_LATENCY_FIELD)
    if str(track) != "T3_tri_backend":
        if isinstance(direct, (int, float)) and math.isfinite(float(direct)):
            if float(direct) < 0:
                raise RuntimeError("feature extraction latency must be non-negative")
            return float(direct), None
        raise RuntimeError(f"{track} feature extraction latency is missing")
    if direct is not None:
        raise RuntimeError(
            "T3 track manifests must not replace the fixed extraction benchmark"
        )

    benchmark_path = (
        Path(run_dir).resolve()
        / "07_validation"
        / "telemetry"
        / "feature_extraction_benchmark.json"
    )
    benchmark = load_verified_json(
        benchmark_path, name="T3 feature extraction benchmark"
    )
    configuration = benchmark.get("configuration")
    if (
        benchmark.get("analysis") != "validation_feature_extraction_runtime"
        or benchmark.get("candidate_test_labels_read") is not False
        or not isinstance(configuration, Mapping)
        or configuration.get("split") != "validation"
        or configuration.get("tag") != "latency_benchmark_128"
        or int(configuration.get("sample_limit", -1)) != 128
    ):
        raise RuntimeError("T3 feature extraction benchmark contract is invalid")
    verify_artifact_records_recursive(
        {
            "sources": benchmark.get("sources"),
            "artifacts": benchmark.get("artifacts"),
        },
        name="T3 feature extraction benchmark",
        require_at_least_one=True,
    )
    latency = t3_composite_extraction_latency_ms(benchmark, feature_manifest)
    return latency, {
        "path": str(benchmark_path),
        "sha256": sha256_file(benchmark_path),
    }


def per_candidate_extraction_latency_ms(
    elapsed_seconds: float, candidate_rows: int
) -> float:
    """Validate and normalise persisted extractor wall time per candidate row."""

    elapsed = float(elapsed_seconds)
    rows = int(candidate_rows)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(
            "feature extraction elapsed time must be finite and non-negative"
        )
    if rows <= 0:
        raise ValueError("feature extraction latency requires positive candidate rows")
    return elapsed * 1000.0 / rows


def peak_memory_mb() -> float:
    """Return process peak resident memory using the platform RSS contract."""

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other supported POSIX builders use KiB.
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return rss / divisor


def missing_feature_rate(frame: pd.DataFrame, columns: Iterable[str]) -> float:
    """Fraction of selected numeric feature cells that are null or non-finite."""

    selected = tuple(map(str, columns))
    if not selected or frame.empty:
        raise ValueError("missing-feature telemetry requires rows and feature columns")
    missing_columns = sorted(set(selected).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"telemetry columns are absent: {missing_columns}")
    numeric = (
        frame.loc[:, selected]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    return float((~np.isfinite(numeric)).sum() / numeric.size)


def torch_parameter_count(model: torch.nn.Module) -> int:
    """Count all learned tensor scalars, matching ``Module.parameters()``."""

    return int(sum(parameter.numel() for parameter in model.parameters()))


def lightgbm_parameter_count(model: Any) -> int:
    """Count fitted LightGBM leaf values as learned scalar parameters."""

    wrapped = getattr(model, "model", None)
    booster = getattr(wrapped, "booster_", None)
    if booster is None:
        raise ValueError("LightGBM model is not fitted")

    def leaves(node: Mapping[str, Any]) -> int:
        if "leaf_value" in node:
            return 1
        left = node.get("left_child")
        right = node.get("right_child")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError("LightGBM dump contains an invalid tree")
        return leaves(left) + leaves(right)

    dump = booster.dump_model()
    trees = dump.get("tree_info") if isinstance(dump, Mapping) else None
    if not isinstance(trees, list) or not trees:
        raise ValueError("LightGBM dump contains no fitted trees")
    count = sum(leaves(tree["tree_structure"]) for tree in trees)
    if count <= 0:
        raise ValueError("LightGBM learned-parameter count is empty")
    return int(count)


def telemetry_payload(
    *,
    phase: str,
    parameter_count: int | None,
    ranker_latency_ms: float | None,
    feature_latency_ms: float,
    missing_feature_rate_value: float,
    not_applicable_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Build and validate the common machine-readable telemetry payload."""

    not_applicable = sorted(set(map(str, not_applicable_fields)))
    values: dict[str, int | float | None] = {
        "parameter_count": parameter_count,
        "ranker_latency_ms": ranker_latency_ms,
        "feature_latency_ms": feature_latency_ms,
        "peak_memory_mb": peak_memory_mb(),
        "missing_feature_rate": missing_feature_rate_value,
    }
    for field, value in values.items():
        if field in not_applicable:
            if value is not None:
                raise ValueError(f"{field} is declared not applicable but has a value")
            continue
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    if not 0.0 <= float(missing_feature_rate_value) <= 1.0:
        raise ValueError("missing_feature_rate must lie in [0, 1]")
    return {
        "schema_version": 1,
        "phase": str(phase),
        **values,
        "not_applicable_fields": not_applicable,
        "measurement_protocols": {
            "parameter_count": "learned tensor scalars (Torch) or fitted leaf values (LightGBM)",
            "ranker_latency_ms": "perf_counter inference wall time divided by candidate rows",
            "feature_latency_ms": "perf_counter feature load/preprocess wall time divided by candidate rows",
            "peak_memory_mb": "process ru_maxrss converted to MiB",
            "missing_feature_rate": "non-finite selected numeric feature cells divided by all selected cells",
        },
    }


def flatten_telemetry(payload: Mapping[str, Any]) -> dict[str, int | float | None]:
    """Project the five report-facing telemetry fields into a manifest."""

    return {field: payload.get(field) for field in TELEMETRY_FIELDS}
