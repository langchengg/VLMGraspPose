"""Offline finalization for a completed protected safe-rerank run.

This module never imports the provider client and cannot issue API requests. It
materializes the human- and machine-readable artifacts required by the frozen
experiment contract from already persisted calibration and P5 evidence.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from .gallery import generate_failure_gallery
from .independent_recompute import independent_recompute_p5_validation
from .plots import generate_final_plots
from .security import secret_pattern_hits


def _atomic_text(path: Path, text: str) -> None:
    if secret_pattern_hits(text):
        raise RuntimeError(f"secret scan failed before writing {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_parquet_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pq.write_table(pq.read_table(source), temporary, compression="zstd")
    temporary.replace(destination)


def _threshold_csv(source: Path) -> str:
    rows = pq.read_table(source).to_pylist()
    if not rows:
        raise RuntimeError("threshold sweep is empty")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _calibration_report(calibration: Mapping[str, Any]) -> str:
    query = calibration["query_gate_contract"]
    rows = []
    for method, payload in sorted(calibration["methods"].items()):
        selected = payload["selected_thresholds"]
        rows.append(
            "| {method} | {state} | {sampling} | {stability} | {coverage:.2%} | "
            "{tau} | {eta} | {net} |".format(
                method=method,
                state=payload["threshold_state"],
                sampling=str(bool(payload.get("sampling_contract_passed"))).lower(),
                stability=str(bool(payload.get("stability_passed"))).lower(),
                coverage=float(payload.get("original_response_coverage", 0.0)),
                tau=selected.get("tau", "—"),
                eta=selected.get("eta", "—"),
                net=selected.get("net", "—"),
            )
        )
    unsupported = ", ".join(query.get("unsupported_natural_joint_strata", [])) or "none"
    return """# Calibration report

## Frozen calibration decision

The 150-sample natural calibration cohort failed the preregistered sampling
support contract. The natural population contains a protected-correct/relation
cell, but the frozen sample contains no member of that cell. Consequently the
joint post-stratification system has no finite solution and no method is
eligible to authorize P5 provider calls. Cohort-only weighted sweeps are kept
as exploratory diagnostics and cannot be used as a lock.

| Method | Threshold state | Sampling contract | Stability | Original coverage | Exploratory τ | η | Weighted Net |
|---|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + f"""

## Query gate

- Natural denominator: {query['call_rate_denominator']:,}
- Natural recoverable denominator: {query['recall_denominator']:,}
- Minimum recoverable recall: {query['minimum_calibration_recoverable_recall']:.1%}
- Unsupported natural strata: `{unsupported}`
- Raking converged: {str(bool(query['raking_converged'])).lower()}
- Sampling contract passed: {str(bool(query['sampling_contract_passed'])).lower()}

No validation or test labels were used to repair, expand, or tune this frozen
calibration cohort.
"""


def _validation_report(validation: Mapping[str, Any]) -> str:
    rows = []
    for method, payload in sorted(validation["methods"].items()):
        corrected = payload["corrected"]
        legacy = payload["legacy"]
        ci = payload["scene_sequence_bootstrap_delta_j1"]
        rows.append(
            "| {method} | {cj:.6f} | {cr}/{ch}/{cn} | {lj:.6f} | {lr}/{lh}/{ln} | "
            "{sw:.2%} | {prec:.2%} | {p:.4g} | [{lo:.6f}, {hi:.6f}] | {status} |".format(
                method=method,
                cj=corrected["final_j1"], cr=corrected["recovered"],
                ch=corrected["harmful"], cn=corrected["net"],
                lj=legacy["final_j1"], lr=legacy["recovered"],
                lh=legacy["harmful"], ln=legacy["net"],
                sw=corrected["switch_rate"],
                prec=corrected["outcome_changing_precision"],
                p=payload["exact_mcnemar_p"], lo=ci["lower"], hi=ci["upper"],
                status=payload["validation_status"],
            )
        )
    return f"""# P5 untouched validation report

## Result

**{validation['validation_status']}**. The frozen calibration sampling contract
made every API method ineligible before validation, so P5 correctly issued zero
provider requests and every one of the {validation['expected_denominator']:,}
samples retained q-only. This is a protected fallback result, not evidence that
an API safe gate improved accuracy.

| Method | Corrected J@1 | Corrected R/H/Net | Legacy J@1 | Legacy R/H/Net | Switch | Outcome precision | McNemar p | Scene CI ΔJ@1 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
""" + "\n".join(rows) + f"""

Primary method: **{validation['primary_method']}**. No validation tuning:
**{str(bool(validation['no_validation_tuning'])).lower()}**. The final formal
gate is closed because validation is not exact GO and the selected primary is
q-only.
"""


def _cost_projection(root: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{(root / 'pairwise_cache.sqlite').resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT r.requested_model,a.estimated_cost_usd "
            "FROM attempts a JOIN requests r USING(request_hash)"
        ).fetchall()
    finally:
        connection.close()
    actual_by_model: dict[str, float] = {}
    request_caps: dict[str, float] = {}
    latency_by_model: dict[str, list[float]] = {}
    for model, cost in rows:
        model = str(model or "unknown")
        actual_by_model[model] = actual_by_model.get(model, 0.0) + float(cost or 0.0)
        request_caps[model] = max(request_caps.get(model, 0.0), float(cost or 0.0))
    connection = sqlite3.connect(f"file:{(root / 'pairwise_cache.sqlite').resolve()}?mode=ro", uri=True)
    try:
        latency_rows = connection.execute(
            "SELECT r.requested_model,a.latency_seconds FROM attempts a "
            "JOIN requests r USING(request_hash) WHERE a.latency_seconds IS NOT NULL"
        ).fetchall()
    finally:
        connection.close()
    for model, latency in latency_rows:
        latency_by_model.setdefault(str(model or "unknown"), []).append(float(latency))
    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    samples = int(_json(root / "DATA_MANIFEST.json")["expected_denominator"])
    per_model_original = 2 * samples
    er2_cap = request_caps.get("gemini-robotics-er-2-preview", 1.0) or 1.0
    flash_cap = request_caps.get("gemini-3.6-flash", 0.10) or 0.10
    original_reserve = per_model_original * (er2_cap + flash_cap)
    er2_latency = latency_by_model.get("gemini-robotics-er-2-preview", [])
    flash_latency = latency_by_model.get("gemini-3.6-flash", [])
    sequential_p50_hours = per_model_original * (
        percentile(er2_latency, .50) + percentile(flash_latency, .50)
    ) / 3600.0
    sequential_p95_hours = per_model_original * (
        percentile(er2_latency, .95) + percentile(flash_latency, .95)
    ) / 3600.0
    return {
        "schema_version": "1.0.0",
        "formal_api_run_authorized": False,
        "formal_samples": samples,
        "maximum_challengers": 2,
        "original_requests_per_model_upper_bound": per_model_original,
        "all_models_original_requests_upper_bound": 2 * per_model_original,
        "confirmation_requests_upper_bound": 2 * per_model_original,
        "all_requests_with_confirmation_upper_bound": 4 * per_model_original,
        "observed_attempt_reserve_usd_by_model": actual_by_model,
        "observed_attempt_reserve_usd": sum(actual_by_model.values()),
        "original_only_reserve_upper_bound_usd": original_reserve,
        "with_confirmation_reserve_upper_bound_usd": 2 * original_reserve,
        "sequential_original_wall_time_projection_hours": {
            "observed_p50_attempt_basis": sequential_p50_hours,
            "observed_p95_attempt_basis": sequential_p95_hours,
        },
        "sequential_with_confirmation_wall_time_upper_hours": {
            "observed_p50_attempt_basis": 2 * sequential_p50_hours,
            "observed_p95_attempt_basis": 2 * sequential_p95_hours,
        },
        "wall_time_projection_caveat": (
            "Rough sequential projection from observed attempt latency; quota, retries, "
            "circuit breakers, and provider drift can dominate actual wall time."
        ),
        "er2_request_cap_usd": er2_cap,
        "flash_request_cap_usd": flash_cap,
        "er2_price_disclaimer": (
            "Actual ER2 monetary cost could not be independently verified; "
            "the experiment used a conservative configurable per-request budget reserve."
        ),
    }


def finalize_safe_rerank(run_dir: str | Path) -> dict[str, Any]:
    """Create final offline evidence artifacts for the completed P5 NO-GO run."""

    root = Path(run_dir).resolve()
    calibration = _json(root / "calibration/API_SAFE_GATE_CALIBRATION.json")
    validation = _json(root / "p5_validation/P5_VALIDATION_RESULTS.json")
    if validation.get("validation_status") != "NO_GO":
        raise RuntimeError("this finalizer is restricted to the frozen P5 NO-GO run")
    if (root / "LOCKED_MANIFEST.json").exists() or any(root.glob("**/FORMAL_TEST_REPORT*")):
        raise RuntimeError("NO-GO run contains a forbidden formal artifact")

    _write_parquet_copy(
        root / "p5_validation/p5_decisions.parquet",
        root / "PER_SAMPLE_DECISIONS.parquet",
    )
    _write_parquet_copy(
        root / "p5_validation/p5_pair_scores.parquet",
        root / "PAIRWISE_DECISIONS.parquet",
    )
    _atomic_text(
        root / "THRESHOLD_SWEEP.csv",
        _threshold_csv(root / "calibration/api_safe_gate_threshold_sweep.parquet"),
    )
    _atomic_text(root / "CALIBRATION_REPORT.md", _calibration_report(calibration))
    _atomic_text(root / "VALIDATION_REPORT.md", _validation_report(validation))
    projection = _cost_projection(root)
    _atomic_text(
        root / "COST_PROJECTION.json",
        json.dumps(projection, indent=2, sort_keys=True) + "\n",
    )
    independent = independent_recompute_p5_validation(root)
    plots = generate_final_plots(root)
    gallery = generate_failure_gallery(root, phase="diagnostic_expanded", limit_per_category=10)
    return {
        "schema_version": "1.0.0",
        "validation_status": validation["validation_status"],
        "primary_method": validation["primary_method"],
        "independent_recompute_matches": independent["all_methods_match_main"],
        "plots": plots,
        "gallery_counts": gallery,
        "formal_artifacts_created": False,
        "cost_projection": projection,
    }


__all__ = ["finalize_safe_rerank"]
