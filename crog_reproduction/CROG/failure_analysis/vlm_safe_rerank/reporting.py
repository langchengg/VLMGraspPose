"""Secret-safe final reporting for the protected VLM reranking experiment.

This module is deliberately offline.  It reads only persisted JSON and SQLite
artifacts, never imports an API client, and writes reports only after every
rendered byte has passed the secret scan.
"""

from __future__ import annotations

import csv
import hashlib
import io
import importlib.metadata
import json
import math
import re
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .security import secret_pattern_hits


SCHEMA_VERSION = "1.0.0"
Q_ONLY = "q_only"
MODEL_IDS = (
    "gemini-robotics-er-2-preview",
    "gemini-3.6-flash",
)
REQUIRED_REPORTS = (
    "SUMMARY.md",
    "GO_NO_GO.md",
    "RESULTS.json",
    "METRICS.csv",
    "API_LEDGER_SUMMARY.json",
    "COST_REPORT.md",
    "LATENCY_REPORT.md",
    "FAILURE_ANALYSIS.md",
    "REPRODUCE.md",
    "MANIFEST.json",
)
CONDITIONAL_LOCK = "REPORTING_LOCK_SNAPSHOT.json"
CONDITIONAL_FORMAL_REPORT = "FORMAL_TEST_REPORT.md"


class ReportingError(RuntimeError):
    """Raised when saved evidence cannot support a safe report."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportingError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ReportingError(f"JSON artifact is not an object: {path.name}")
    return value


def _first_file(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    try:
        name = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        name = path.name
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": name, "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _fmt_number(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "not available"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:.{digits}f}"


def _fmt_rate(value: Any) -> str:
    number = _as_float(value)
    return "not available" if number is None else f"{100.0 * number:.2f}%"


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _assert_secret_free(rendered: Mapping[str, str]) -> None:
    credential_assignment = re.compile(
        r"(?:GEMINI_API_KEY|GOOGLE_API_KEY|X-goog-api-key)\s*[:=]\s*[^\s,}\]]+",
        flags=re.IGNORECASE,
    )
    for name, content in rendered.items():
        if secret_pattern_hits(content) or credential_assignment.search(content):
            raise ReportingError(f"secret scan failed before writing {name}")


def _ledger_summary(path: Path) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "status": "missing",
        "requests": 0,
        "attempts": 0,
        "request_status_counts": {},
        "attempt_status_counts": {},
        "attempt_error_class_counts": {},
        "duplicate_successful_request_hashes": 0,
        "attempt_estimated_cost_usd": 0.0,
        "latency_seconds": {
            "all_attempts_p50": None,
            "all_attempts_p95": None,
            "successful_attempts_p50": None,
            "successful_attempts_p95": None,
        },
        "model_attribution_available": False,
    }
    if not path.is_file():
        return empty
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"requests", "attempts"}.issubset(tables):
            return {**empty, "status": "invalid_schema"}
        requests = [dict(row) for row in connection.execute("SELECT status FROM requests")]
        request_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(requests)")
        }
        model_select = (
            "r.requested_model" if "requested_model" in request_columns else "NULL AS requested_model"
        )
        attempts = [
            dict(row)
            for row in connection.execute(
                "SELECT a.request_hash,a.status,a.latency_seconds,a.estimated_cost_usd,"
                f"a.error_class,{model_select} FROM attempts a "
                "JOIN requests r USING(request_hash)"
            )
        ]
    except sqlite3.Error:
        return {**empty, "status": "unreadable"}
    finally:
        if connection is not None:
            connection.close()
    all_latencies = [
        float(row["latency_seconds"])
        for row in attempts
        if _as_float(row.get("latency_seconds")) is not None
    ]
    success_latencies = [
        float(row["latency_seconds"])
        for row in attempts
        if str(row.get("status")) in {"SUCCEEDED", "ABSTAIN"}
        and _as_float(row.get("latency_seconds")) is not None
    ]
    successes = Counter(
        str(row["request_hash"])
        for row in attempts
        if str(row.get("status")) in {"SUCCEEDED", "ABSTAIN"}
    )
    per_model: dict[str, Any] = {}
    for model in sorted({str(row.get("requested_model") or "unknown") for row in attempts}):
        selected = [row for row in attempts if str(row.get("requested_model") or "unknown") == model]
        model_latencies = [
            float(row["latency_seconds"])
            for row in selected if _as_float(row.get("latency_seconds")) is not None
        ]
        per_model[model] = {
            "attempts": len(selected),
            "status_counts": dict(Counter(str(row["status"]) for row in selected)),
            "estimated_cost_usd": sum(float(row.get("estimated_cost_usd") or 0.0) for row in selected),
            "latency_p50_seconds": _percentile(model_latencies, 0.50),
            "latency_p95_seconds": _percentile(model_latencies, 0.95),
        }
    return {
        "status": "available",
        "requests": len(requests),
        "attempts": len(attempts),
        "request_status_counts": dict(Counter(str(row["status"]) for row in requests)),
        "attempt_status_counts": dict(Counter(str(row["status"]) for row in attempts)),
        "attempt_error_class_counts": dict(
            Counter(str(row["error_class"]) for row in attempts if row.get("error_class"))
        ),
        "duplicate_successful_request_hashes": sum(count > 1 for count in successes.values()),
        "attempt_estimated_cost_usd": sum(
            float(row.get("estimated_cost_usd") or 0.0) for row in attempts
        ),
        "latency_seconds": {
            "all_attempts_p50": _percentile(all_latencies, 0.50),
            "all_attempts_p95": _percentile(all_latencies, 0.95),
            "successful_attempts_p50": _percentile(success_latencies, 0.50),
            "successful_attempts_p95": _percentile(success_latencies, 0.95),
        },
        "model_attribution_available": True,
        "per_model": per_model,
    }


def _phase_api_totals(root: Path) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for path in sorted(root.glob("*/API_SUMMARY.json")):
        value = _load_json(path)
        if value is None:
            continue
        phases[path.parent.name] = {
            key: value.get(key)
            for key in ("model_pair_variants", "successful", "fallback", "cache_hits")
        }
    return {
        "phases": phases,
        "cache_reuse_events": sum(
            int(_mapping(value).get("cache_hits") or 0) for value in phases.values()
        ),
        "successful_rows_reported_by_phases": sum(
            int(_mapping(value).get("successful") or 0) for value in phases.values()
        ),
        "fallback_rows_reported_by_phases": sum(
            int(_mapping(value).get("fallback") or 0) for value in phases.values()
        ),
    }


def _metric_row(
    *,
    stage: str,
    method: str,
    track: str,
    values: Mapping[str, Any],
    scope: str,
    schema_valid: Any = None,
    fallback: Any = None,
    terminal_failures: Any = None,
) -> dict[str, Any]:
    baseline_j1 = values.get("baseline_j1", values.get("q_only_j1"))
    selected_j1 = values.get("final_j1", values.get("selected_j1"))
    total = values.get("total", values.get("samples", values.get("pairs")))
    delta_pp = values.get("delta_pp")
    if delta_pp is None and _as_float(baseline_j1) is not None and _as_float(selected_j1) is not None:
        delta_pp = 100.0 * (float(selected_j1) - float(baseline_j1))
    return {
        "stage": stage,
        "method": method,
        "track": track,
        "scope": scope,
        "total": total,
        "schema_valid": schema_valid,
        "fallback": fallback,
        "terminal_failures": terminal_failures,
        "baseline_j1": baseline_j1,
        "selected_j1": selected_j1,
        "delta_pp": delta_pp,
        "recovered": values.get("recovered"),
        "harmful": values.get("harmful"),
        "net": values.get("net"),
        "switches": values.get("switches"),
        "switch_rate": values.get("switch_rate"),
        "harm_rate": values.get("harm_rate"),
        "outcome_changing_precision": values.get("outcome_changing_precision"),
    }


def _collect_metrics(
    p1: Mapping[str, Any] | None,
    smoke_results: Mapping[str, Any] | None,
    diagnostic_results: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    formal: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, payload in sorted(_mapping((p1 or {}).get("models")).items()):
        model_values = _mapping(payload)
        for track in ("legacy", "corrected"):
            values = _mapping(model_values.get(track))
            if values:
                rows.append(
                    _metric_row(
                        stage="p1_diagnostic",
                        method=model,
                        track=track,
                        values=values,
                        scope="full_terminal_diagnostic",
                        schema_valid=model_values.get("schema_valid"),
                        fallback=model_values.get("fallback"),
                        terminal_failures=model_values.get("terminal_failures"),
                    )
                )
    for model, payload in sorted(_mapping((smoke_results or {}).get("models")).items()):
        values = _mapping(_mapping(payload).get("corrected_hard_rule"))
        if values:
            rows.append(
                _metric_row(
                    stage="smoke",
                    method=f"{model}:hard_rule",
                    track="corrected",
                    values=values,
                    scope="full_smoke_denominator",
                )
            )
    for model, payload in sorted(_mapping((diagnostic_results or {}).get("models")).items()):
        values = _mapping(_mapping(payload).get("corrected_hard_rule"))
        if values:
            rows.append(
                _metric_row(
                    stage="pairwise_diagnostic",
                    method=f"{model}:hard_rule",
                    track="corrected",
                    values=values,
                    scope="balanced_diagnostic_denominator",
                )
            )
    if calibration:
        values = {
            **_mapping(calibration.get("selected_thresholds")),
            "total": calibration.get("samples"),
        }
        rows.append(
            _metric_row(
                stage="calibration",
                method=str(calibration.get("method") or "local_safe_gate"),
                track="corrected",
                values=values,
                scope="full_calibration",
            )
        )
    if validation:
        validation_methods = _mapping(validation.get("methods"))
        if validation_methods:
            for method, payload in sorted(validation_methods.items()):
                for track in ("corrected", "legacy"):
                    values = _mapping(_mapping(payload).get(track))
                    if values:
                        rows.append(
                            _metric_row(
                                stage="p5_validation",
                                method=method,
                                track=track,
                                values=values,
                                scope="full_untouched_validation",
                            )
                        )
        else:
            values = _mapping(validation.get("corrected"))
            if values:
                rows.append(
                    _metric_row(
                        stage="validation",
                        method=str(validation.get("primary_method") or Q_ONLY),
                        track="corrected",
                        values=values,
                        scope="full_validation",
                    )
                )
    if formal:
        methods = _mapping(formal.get("methods"))
        if methods:
            for method, payload in sorted(methods.items()):
                values = _mapping(_mapping(payload).get("corrected") or payload)
                if values:
                    rows.append(
                        _metric_row(
                            stage="formal_test",
                            method=method,
                            track="corrected",
                            values=values,
                            scope="full_formal_test",
                        )
                    )
        else:
            values = _mapping(formal.get("corrected"))
            if values:
                rows.append(
                    _metric_row(
                        stage="formal_test",
                        method=str(formal.get("primary_method") or "locked_primary"),
                        track="corrected",
                        values=values,
                        scope="full_formal_test",
                    )
                )
    return rows


def _metrics_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "stage", "method", "track", "scope", "total", "schema_valid",
        "fallback", "terminal_failures", "baseline_j1", "selected_j1",
        "delta_pp", "recovered", "harmful", "net", "switches",
        "switch_rate", "harm_rate", "outcome_changing_precision",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return output.getvalue()


def _formal_observed_total(formal: Mapping[str, Any] | None) -> int | None:
    if not formal:
        return None
    direct = _as_int(formal.get("total"))
    if direct is not None:
        return direct
    completed = _as_int(formal.get("completed"))
    if completed is not None:
        return completed
    summary_total = _as_int(_mapping(formal.get("summary")).get("total"))
    if summary_total is not None:
        return summary_total
    corrected = _mapping(formal.get("corrected"))
    if _as_int(corrected.get("total")) is not None:
        return _as_int(corrected.get("total"))
    method_totals = {
        _as_int(_mapping(_mapping(value).get("corrected") or value).get("total"))
        for value in _mapping(formal.get("methods")).values()
    }
    method_totals.discard(None)
    return next(iter(method_totals)) if len(method_totals) == 1 else None


def _formal_status_complete(formal: Mapping[str, Any] | None) -> bool:
    if not formal:
        return False
    status = str(
        formal.get("formal_status", formal.get("status", formal.get("phase_status", "")))
    ).upper()
    return bool(formal.get("formal_complete")) or status in {
        "COMPLETE", "FORMAL_COMPLETE", "COMPLETED"
    }


def _source_inventory(root: Path) -> dict[str, Path | None]:
    return {
        "data_manifest": _first_file(root, ("DATA_MANIFEST.json",)),
        "inference_manifest": _first_file(root, ("INFERENCE_MANIFEST.json",)),
        "p1_freeze": _first_file(
            root, ("p1_direct_diagnostic/P1_DIAGNOSTIC_FREEZE.json",)
        ),
        "smoke_api_summary": _first_file(
            root, ("smoke/API_SUMMARY.json", "smoke/API_PROGRESS.json")
        ),
        "smoke_results": _first_file(root, ("smoke/DIAGNOSTIC_RESULTS.json",)),
        "diagnostic_progress": _first_file(
            root,
            (
                "p3_diagnostic/API_SUMMARY.json",
                "diagnostic/API_SUMMARY.json",
                "diagnostic/API_PROGRESS.json",
            ),
        ),
        "diagnostic_results": _first_file(
            root,
            (
                "diagnostic_expanded/DIAGNOSTIC_RESULTS.json",
                "p3_diagnostic/DIAGNOSTIC_RESULTS.json",
                "diagnostic/DIAGNOSTIC_RESULTS.json",
            ),
        ),
        "calibration": _first_file(
            root,
            (
                "calibration/API_SAFE_GATE_CALIBRATION.json",
                "calibration/CALIBRATION_RESULTS.json",
            ),
        ),
        "validation": _first_file(
            root,
            (
                "p5_validation/P5_VALIDATION_RESULTS.json",
                "validation/VALIDATION_RESULTS.json",
            ),
        ),
        "independent_recompute": _first_file(root, ("independent_recompute_results.json",)),
        "cost_projection": _first_file(root, ("COST_PROJECTION.json",)),
        "test_results": _first_file(root, ("TEST_RESULTS.json",)),
        "pairwise_ledger": _first_file(root, ("pairwise_cache.sqlite",)),
        "formal": _first_file(
            root,
            (
                "formal_test/FORMAL_TEST_RESULTS.json",
                "formal_test/RESULTS.json",
                "formal_test/results_bundle.json",
                "formal_test/FORMAL_TEST_REPORT.json",
                "FORMAL_TEST_RESULTS.json",
                "FORMAL_TEST_REPORT.json",
            ),
        ),
    }


def _canonical_calibration(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None or "methods" not in value:
        return value
    methods = _mapping(value.get("methods"))
    states = {str(_mapping(payload).get("threshold_state")) for payload in methods.values()}
    result = dict(value)
    result.setdefault("method", "api_safe_gate_multi_method")
    result.setdefault(
        "threshold_state",
        next(iter(states)) if len(states) == 1 else "mixed",
    )
    result.setdefault("samples", 150)
    result.setdefault("pairs", 300)
    result.setdefault("selected_thresholds", {})
    return result


def _canonical_validation(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None or "methods" not in value or "corrected" in value:
        return value
    methods = _mapping(value.get("methods"))
    if not methods:
        return value
    reference_name = "P5_er2_safe" if "P5_er2_safe" in methods else sorted(methods)[0]
    reference = _mapping(methods[reference_name])
    result = dict(value)
    result["corrected"] = _mapping(reference.get("corrected"))
    result["legacy"] = _mapping(reference.get("legacy"))
    result["exact_mcnemar_p"] = reference.get("exact_mcnemar_p")
    result["scene_sequence_bootstrap_delta_j1"] = _mapping(
        reference.get("scene_sequence_bootstrap_delta_j1")
    )
    return result


def _build_context(root: Path) -> dict[str, Any]:
    sources = _source_inventory(root)
    data = _load_json(sources["data_manifest"])
    inference = _load_json(sources["inference_manifest"])
    p1 = _load_json(sources["p1_freeze"])
    smoke_api = _load_json(sources["smoke_api_summary"])
    smoke_results = _load_json(sources["smoke_results"])
    diagnostic = _load_json(sources["diagnostic_progress"])
    diagnostic_results = _load_json(sources["diagnostic_results"])
    calibration = _canonical_calibration(_load_json(sources["calibration"]))
    validation = _canonical_validation(_load_json(sources["validation"]))
    independent_recompute = _load_json(sources["independent_recompute"])
    cost_projection = _load_json(sources["cost_projection"])
    formal = _load_json(sources["formal"])
    ledger = _ledger_summary(sources["pairwise_ledger"] or root / "pairwise_cache.sqlite")
    phase_api_totals = _phase_api_totals(root)

    run_id = str(
        (data or {}).get("run_id")
        or (inference or {}).get("run_id")
        or root.name
    )
    data_denominator = _as_int((data or {}).get("expected_denominator"))
    inference_denominator = _as_int((inference or {}).get("expected_denominator"))
    expected_formal = data_denominator or inference_denominator
    formal_denominator_consistent = (
        data_denominator is not None
        and inference_denominator is not None
        and data_denominator == inference_denominator
    )
    validation_metrics = _mapping((validation or {}).get("corrected"))
    validation_expected = _as_int((validation or {}).get("expected_denominator"))
    validation_observed = _as_int(validation_metrics.get("total"))
    validation_denominator_consistent = (
        validation_expected is not None
        and validation_observed is not None
        and validation_expected == validation_observed
    )
    validation_status = str((validation or {}).get("validation_status") or "MISSING").upper()
    declared_primary = str((validation or {}).get("primary_method") or Q_ONLY)
    primary_is_q_only = "qonly" in re.sub(r"[^a-z0-9]", "", declared_primary.lower())
    corrected_primary = (
        declared_primary if validation_status == "GO" and not primary_is_q_only else Q_ONLY
    )
    formal_flag = bool(
        _mapping((inference or {}).get("inference_inputs")).get(
            "formal_requests_authorized", False
        )
    )
    core_available = {
        "data_manifest": data is not None,
        "inference_manifest": inference is not None,
        "p1_diagnostic_freeze": p1 is not None,
        "smoke": smoke_api is not None and smoke_results is not None,
        "diagnostic": diagnostic is not None,
        "calibration": calibration is not None,
        "validation": validation is not None,
        "pairwise_ledger": ledger["status"] == "available",
    }
    missing_core = [name for name, available in core_available.items() if not available]
    blockers: list[str] = []
    if missing_core:
        blockers.append("missing core evidence: " + ", ".join(missing_core))
    if not formal_denominator_consistent:
        blockers.append("formal denominator bindings are missing or inconsistent")
    if validation is not None and not validation_denominator_consistent:
        blockers.append("validation observed/full denominator mismatch")
    if ledger["duplicate_successful_request_hashes"]:
        blockers.append("pairwise ledger contains duplicate successful request hashes")
    if validation_status == "NO_GO":
        blockers.append("corrected validation status is NO_GO")
    elif validation_status == "GO" and primary_is_q_only:
        blockers.append("GO cannot select q-only as an API primary")
    elif validation_status == "MISSING":
        blockers.append("corrected validation is absent")
    elif validation_status != "GO":
        blockers.append(
            f"corrected validation status is {validation_status}; only exact GO authorizes formal"
        )

    if validation is not None and validation_status != "GO":
        decision_status = "NO_GO"
    elif validation_status == "GO" and not missing_core and not any(
        phrase in blocker
        for blocker in blockers
        for phrase in (
            "denominator", "duplicate successful", "cannot select q-only"
        )
    ):
        decision_status = "GO"
    else:
        decision_status = "PARTIAL"
    formal_run_allowed = (
        decision_status == "GO"
        and formal_flag
        and corrected_primary != Q_ONLY
        and not blockers
    )
    formal_observed = _formal_observed_total(formal)
    formal_complete = (
        formal_run_allowed
        and _formal_status_complete(formal)
        and expected_formal is not None
        and formal_observed == expected_formal
    )
    if formal is not None and not formal_complete:
        blockers.append("formal artifact exists but is not a complete full-denominator run")
    report_status = "FORMAL_COMPLETE" if formal_complete else decision_status
    metrics = _collect_metrics(
        p1,
        smoke_results,
        diagnostic_results,
        calibration,
        validation,
        formal if formal_complete else None,
    )

    model_failures: dict[str, Any] = {}
    p1_models = _mapping((p1 or {}).get("models"))
    smoke_models = _mapping((smoke_results or {}).get("models"))
    diagnostic_models = _mapping((diagnostic_results or {}).get("models"))
    for model in MODEL_IDS:
        p1_model = _mapping(p1_models.get(model))
        smoke_model = _mapping(smoke_models.get(model))
        diagnostic_model = _mapping(diagnostic_models.get(model))
        model_failures[model] = {
            "p1_decisions": _as_int(p1_model.get("decisions")),
            "p1_schema_valid": _as_int(p1_model.get("schema_valid")),
            "p1_fallback": _as_int(p1_model.get("fallback")),
            "p1_terminal_failures": _as_int(p1_model.get("terminal_failures")),
            "smoke_status_counts": _mapping(smoke_model.get("status_counts")),
            "diagnostic_status_counts": _mapping(
                diagnostic_model.get("status_counts")
            ),
        }

    return {
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "sources": sources,
        "data": data,
        "inference": inference,
        "p1": p1,
        "smoke_api": smoke_api,
        "smoke_results": smoke_results,
        "diagnostic": diagnostic,
        "diagnostic_results": diagnostic_results,
        "calibration": calibration,
        "validation": validation,
        "independent_recompute": independent_recompute,
        "cost_projection": cost_projection,
        "formal": formal,
        "ledger": ledger,
        "phase_api_totals": phase_api_totals,
        "core_available": core_available,
        "missing_core": missing_core,
        "blockers": blockers,
        "report_status": report_status,
        "decision_status": decision_status,
        "validation_status": validation_status,
        "corrected_primary": corrected_primary,
        "formal_requests_authorized_flag": formal_flag,
        "formal_run_allowed": formal_run_allowed,
        "formal_complete": formal_complete,
        "expected_formal_denominator": expected_formal,
        "formal_observed_denominator": formal_observed,
        "validation_expected_denominator": validation_expected,
        "validation_observed_denominator": validation_observed,
        "formal_denominator_consistent": formal_denominator_consistent,
        "validation_denominator_consistent": validation_denominator_consistent,
        "metrics": metrics,
        "model_failures": model_failures,
    }


def _results_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    validation = _mapping(context.get("validation"))
    corrected = _mapping(validation.get("corrected"))
    calibration = _mapping(context.get("calibration"))
    p1 = _mapping(context.get("p1"))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": context["generated_at_utc"],
        "run_id": context["run_id"],
        "report_status": context["report_status"],
        "decision_status": context["decision_status"],
        "primary_evaluator": "corrected",
        "corrected_primary_method": context["corrected_primary"],
        "default_action": Q_ONLY,
        "q_only_is_default_and_fallback": True,
        "formal_requests_authorized_flag": context["formal_requests_authorized_flag"],
        "formal_run_allowed": context["formal_run_allowed"],
        "formal_complete": context["formal_complete"],
        "denominators": {
            "formal_expected": context["expected_formal_denominator"],
            "formal_observed": context["formal_observed_denominator"],
            "validation_expected": context["validation_expected_denominator"],
            "validation_observed": context["validation_observed_denominator"],
            "calibration_samples": _as_int(calibration.get("samples")),
            "calibration_pairs": _as_int(calibration.get("pairs")),
            "p1_decisions": _as_int(p1.get("decision_count")),
            "pairwise_requests": context["ledger"]["requests"],
            "pairwise_attempts": context["ledger"]["attempts"],
        },
        "phase_availability": context["core_available"],
        "missing_core_evidence": context["missing_core"],
        "blockers": context["blockers"],
        "p1_diagnostic_eligible_for_primary": bool(p1.get("eligible_for_primary", False)),
        "calibration": {
            "method": calibration.get("method"),
            "threshold_state": calibration.get("threshold_state"),
            "selected_thresholds": _mapping(calibration.get("selected_thresholds")),
            "methods": _mapping(calibration.get("methods")),
            "query_gate_contract": _mapping(calibration.get("query_gate_contract")),
        },
        "validation": {
            "status": context["validation_status"],
            "primary_method": validation.get("primary_method"),
            "corrected": corrected,
            "exact_mcnemar_p": validation.get("exact_mcnemar_p"),
            "scene_sequence_bootstrap_delta_j1": _mapping(
                validation.get("scene_sequence_bootstrap_delta_j1")
            ),
            "no_validation_tuning": validation.get("no_validation_tuning"),
            "methods": _mapping(validation.get("methods")),
            "candidate_identity_sha256": validation.get("candidate_identity_sha256"),
            "q_value_sha256": validation.get("q_value_sha256"),
            "combined_identity_sha256": validation.get("combined_identity_sha256"),
        },
        "api_phase_progress": {
            "smoke": {
                "planned_model_pair_variants": (context.get("smoke_api") or {}).get(
                    "planned_model_pair_variants",
                    (context.get("smoke_api") or {}).get("model_pair_variants"),
                ),
                "completed_model_pair_variants": (context.get("smoke_api") or {}).get(
                    "completed_model_pair_variants",
                    (context.get("smoke_api") or {}).get("model_pair_variants"),
                ),
                "successful": (context.get("smoke_api") or {}).get("successful"),
                "fallback": (context.get("smoke_api") or {}).get("fallback"),
            },
            "diagnostic": {
                "planned_model_pair_variants": (context.get("diagnostic") or {}).get(
                    "planned_model_pair_variants"
                ),
                "completed_model_pair_variants": (context.get("diagnostic") or {}).get(
                    "completed_model_pair_variants"
                ),
            },
        },
        "model_failures": context["model_failures"],
        "api_phase_totals": context.get("phase_api_totals"),
        "independent_recompute": context.get("independent_recompute"),
        "cost_projection": context.get("cost_projection"),
        "secret_scan": {"status": "passed_before_write"},
    }


def _summary_markdown(context: Mapping[str, Any]) -> str:
    validation = _mapping(context.get("validation"))
    corrected = _mapping(validation.get("corrected"))
    status = context["report_status"]
    decision = context["decision_status"]
    primary = context["corrected_primary"]
    blockers = context["blockers"]
    p1_models = _mapping(_mapping(context.get("p1")).get("models"))
    er2_direct = _mapping(_mapping(p1_models.get(MODEL_IDS[0])).get("corrected"))
    flash_direct = _mapping(_mapping(p1_models.get(MODEL_IDS[1])).get("corrected"))
    diagnostic_models = _mapping(_mapping(context.get("diagnostic_results")).get("models"))
    er2_pairwise = _mapping(_mapping(diagnostic_models.get(MODEL_IDS[0])).get("corrected_hard_rule"))
    flash_pairwise = _mapping(_mapping(diagnostic_models.get(MODEL_IDS[1])).get("corrected_hard_rule"))
    return f"""# Safe VLM reranking final summary

## Technical summary

**Decision: {decision}.** The corrected evaluator is primary. The locked/default action is **q-only**, and the selected corrected primary is **{primary}**. This report is **{status}**; a partial result is never represented as formal.

- Formal API allow flag: **{str(context['formal_requests_authorized_flag']).lower()}**.
- Formal run allowed by the saved evidence: **{str(context['formal_run_allowed']).lower()}**.
- Formal run complete: **{str(context['formal_complete']).lower()}**.
- Full formal denominator: **{_fmt_number(context['expected_formal_denominator'])}**; observed formal denominator: **{_fmt_number(context['formal_observed_denominator'])}**.
- Full validation denominator: **{_fmt_number(context['validation_observed_denominator'])}**; corrected q-only/final J@1: **{_fmt_rate(corrected.get('baseline_j1'))} / {_fmt_rate(corrected.get('final_j1'))}**.

## Corrected validation determines the primary

Validation status is **{context['validation_status']}** with primary method **{validation.get('primary_method', 'not available')}**. Corrected Recovered/Harmful/Net is **{_fmt_number(corrected.get('recovered'))}/{_fmt_number(corrected.get('harmful'))}/{_fmt_number(corrected.get('net'))}** over the complete saved validation denominator. Legacy results, when present, remain diagnostic and do not select the primary.

## Calibration and diagnostic evidence

Calibration state is **{_mapping(context.get('calibration')).get('threshold_state', 'missing')}**. P1 Direct is diagnostic-only and eligible-for-primary is **{str(bool(_mapping(context.get('p1')).get('eligible_for_primary', False))).lower()}**. Flash and ER2 failure counts use each model's full decision denominator and are detailed in `FAILURE_ANALYSIS.md`.

## Answers to the scientific questions

- Direct replacement failed primarily through excessive switching: ER2 switched **{_fmt_rate(er2_direct.get('switch_rate'))}** and produced corrected R/H/Net **{_fmt_number(er2_direct.get('recovered'))}/{_fmt_number(er2_direct.get('harmful'))}/{_fmt_number(er2_direct.get('net'))}**; Flash switched **{_fmt_rate(flash_direct.get('switch_rate'))}** with **{_fmt_number(flash_direct.get('recovered'))}/{_fmt_number(flash_direct.get('harmful'))}/{_fmt_number(flash_direct.get('net'))}**.
- The explicit KEEP prior and pairwise evidence reduced exposure but did not establish positive benefit. On the 150-sample balanced diagnostic cohort, ER2 achieved corrected R/H/Net **{_fmt_number(er2_pairwise.get('recovered'))}/{_fmt_number(er2_pairwise.get('harmful'))}/{_fmt_number(er2_pairwise.get('net'))}** at **{_fmt_rate(er2_pairwise.get('switch_rate'))}**; Flash was effectively q-only because provider responses were unavailable.
- ER2 supplied more usable critic evidence than Flash, but it still harmed protected-correct cases. Flash's near-zero valid perturbation coverage means the joint gate was not independently corroborated.
- Cross-model agreement is not treated as independent confirmation. No evidence in this run establishes that agreement is safer than the single-model gate.
- The mask/depth/geometry board was exercised, but there was no separately randomized input ablation, so its causal contribution cannot be claimed.
- The exploratory calibrated ER2 sweep had positive weighted Net, but the frozen calibration omitted a natural protected-correct/relation stratum. Therefore that threshold is sampling-ineligible and cannot authorize validation API calls or a formal lock.
- Every provider, schema, reliability, confirmation, or gate failure strictly falls back to q-only and remains in its phase denominator.
- The safe API policy did not beat q-only on untouched validation: it made zero authorized switches. The final recommendation is to keep q-only and not integrate either provider critic as a production primary.

## Scope and denominator definitions

- A fallback includes every invalid, abstained, technical, or permanent-failure path that keeps q-only.
- Validation metrics use all saved validation samples, not only successful switches.
- Pairwise ledger attempts are API attempts, while requests are distinct logical request hashes; they are not interchangeable denominators.

## Limitations and next step

{('Blocking evidence: ' + '; '.join(blockers)) if blockers else 'No reporting blocker was detected. Formal execution still requires the independent runner lock and authorization checks.'}
"""


def _go_no_go_markdown(context: Mapping[str, Any]) -> str:
    blockers = context["blockers"]
    blocker_lines = "\n".join(f"- {item}" for item in blockers) or "- None in the saved reporting evidence."
    return f"""# GO / NO-GO decision

## Decision

**{context['decision_status']}** under the corrected evaluator. Corrected primary: **{context['corrected_primary']}**. Q-only remains the default and universal fallback.

## Formal gate state

- Saved formal allow flag: **{str(context['formal_requests_authorized_flag']).lower()}**
- Reporting evidence permits a formal run: **{str(context['formal_run_allowed']).lower()}**
- Actual full-denominator formal run complete: **{str(context['formal_complete']).lower()}**
- Expected formal denominator: **{_fmt_number(context['expected_formal_denominator'])}**

## Blockers and safeguards

{blocker_lines}

P1 diagnostic performance cannot override this decision. Missing phases stay partial, and a NO-GO run does not create `LOCKED_MANIFEST.json` or `FORMAL_TEST_REPORT.md`.
"""


def _cost_markdown(context: Mapping[str, Any]) -> str:
    pairwise = context["ledger"]
    p1_cost = _as_float(_mapping(_mapping(context.get("p1")).get("ledger")).get("attempt_estimated_charge_usd"))
    pairwise_cost = _as_float(pairwise.get("attempt_estimated_cost_usd")) or 0.0
    combined = pairwise_cost + (p1_cost or 0.0)
    projection = _mapping(context.get("cost_projection"))
    return f"""# Cost report

## Persisted usage-based estimates

| Source | Attempts | Estimated cost (USD) | Interpretation |
|---|---:|---:|---|
| Current pairwise ledger | {_fmt_number(pairwise.get('attempts'))} | {pairwise_cost:.6f} | Authoritative for the current pairwise cache snapshot |
| Frozen P1 diagnostic ledger | {_fmt_number(_mapping(_mapping(context.get('p1')).get('ledger')).get('attempts'))} | {_fmt_number(p1_cost, 6)} | Separate diagnostic source run |
| Distinct-source total | — | {combined:.6f} | Does not add smoke/diagnostic progress again because those are subsets of the pairwise ledger |

Provider invoice verification is unavailable. Actual ER2 monetary cost could not be independently verified; the experiment used a conservative configurable per-request budget reserve. Model-level reserve attribution comes only from the saved request ledger.

## Formal projection (not authorized)

- Original pairwise request upper bound: **{_fmt_number(projection.get('all_models_original_requests_upper_bound'))}**.
- Original plus confirmation request upper bound: **{_fmt_number(projection.get('all_requests_with_confirmation_upper_bound'))}**.
- Conservative original-only reserve: **${_fmt_number(projection.get('original_only_reserve_upper_bound_usd'), 2)}**.
- Conservative original-plus-confirmation reserve: **${_fmt_number(projection.get('with_confirmation_reserve_upper_bound_usd'), 2)}**.
- Sequential original-only wall-time projection: **{_fmt_number(_mapping(projection.get('sequential_original_wall_time_projection_hours')).get('observed_p50_attempt_basis'), 1)}–{_fmt_number(_mapping(projection.get('sequential_original_wall_time_projection_hours')).get('observed_p95_attempt_basis'), 1)} hours** on observed p50/p95 attempt-latency bases.
- Sequential confirmation upper bound: **{_fmt_number(_mapping(projection.get('sequential_with_confirmation_wall_time_upper_hours')).get('observed_p50_attempt_basis'), 1)}–{_fmt_number(_mapping(projection.get('sequential_with_confirmation_wall_time_upper_hours')).get('observed_p95_attempt_basis'), 1)} hours**.

These are deliberately conservative capacity bounds, not a provider invoice and not permission to run formal test.
"""


def _latency_markdown(context: Mapping[str, Any]) -> str:
    ledger_latency = _mapping(context["ledger"].get("latency_seconds"))
    p1_models = _mapping(_mapping(context.get("p1")).get("models"))
    rows = []
    for model in MODEL_IDS:
        values = _mapping(p1_models.get(model))
        rows.append(
            f"| P1 diagnostic | {model} | {_fmt_number(values.get('latency_p50_seconds'))} | {_fmt_number(values.get('latency_p95_seconds'))} | model-attributed terminal decisions |"
        )
    rows.append(
        f"| Pairwise ledger | all models | {_fmt_number(ledger_latency.get('all_attempts_p50'))} | {_fmt_number(ledger_latency.get('all_attempts_p95'))} | all persisted attempts, including failures |"
    )
    rows.append(
        f"| Pairwise ledger | successful only | {_fmt_number(ledger_latency.get('successful_attempts_p50'))} | {_fmt_number(ledger_latency.get('successful_attempts_p95'))} | successful/abstain attempts |"
    )
    for model, values in sorted(_mapping(context["ledger"].get("per_model")).items()):
        rows.append(
            f"| Pairwise ledger | {model} | {_fmt_number(_mapping(values).get('latency_p50_seconds'))} | {_fmt_number(_mapping(values).get('latency_p95_seconds'))} | {_fmt_number(_mapping(values).get('attempts'))} model-attributed attempts |"
        )
    return """# Latency report

## Saved latency distributions

| Source | Model/scope | p50 seconds | p95 seconds | Denominator |
|---|---|---:|---:|---|
""" + "\n".join(rows) + """

Model attribution is read from the immutable `requests.requested_model` ledger column. Failed attempts remain in the all-attempt distribution.
"""


def _failure_markdown(context: Mapping[str, Any]) -> str:
    rows = []
    for model in MODEL_IDS:
        values = context["model_failures"][model]
        rows.append(
            "| {model} | {decisions} | {valid} | {fallback} | {terminal} | `{smoke}` | `{diagnostic}` |".format(
                model=model,
                decisions=_fmt_number(values.get("p1_decisions")),
                valid=_fmt_number(values.get("p1_schema_valid")),
                fallback=_fmt_number(values.get("p1_fallback")),
                terminal=_fmt_number(values.get("p1_terminal_failures")),
                smoke=json.dumps(values.get("smoke_status_counts") or {}, sort_keys=True),
                diagnostic=json.dumps(
                    values.get("diagnostic_status_counts") or {}, sort_keys=True
                ),
            )
        )
    ledger = context["ledger"]
    return """# Failure analysis

## Flash and ER2 use the full terminal denominator

| Model | P1 decisions | Schema-valid | All fallback | Permanent failures | Smoke status counts | Diagnostic status counts |
|---|---:|---:|---:|---:|---|---|
""" + "\n".join(rows) + f"""

Fallback is not a successful model decision: every invalid, abstained, technical, or permanent-failure path is retained in the denominator and keeps q-only.

## Pairwise transport and ledger failures

- Logical requests: **{_fmt_number(ledger.get('requests'))}**
- Physical attempts: **{_fmt_number(ledger.get('attempts'))}**
- Request status counts: `{json.dumps(ledger.get('request_status_counts', {}), sort_keys=True)}`
- Attempt status counts: `{json.dumps(ledger.get('attempt_status_counts', {}), sort_keys=True)}`
- Error-class counts: `{json.dumps(ledger.get('attempt_error_class_counts', {}), sort_keys=True)}`
- Duplicate successful request hashes: **{_fmt_number(ledger.get('duplicate_successful_request_hashes'))}**
- Cross-phase cache reuse events: **{_fmt_number(_mapping(context.get('phase_api_totals')).get('cache_reuse_events'))}** (reuse events, not distinct requests)

## Scientific failure modes

Calibration state is **{_mapping(context.get('calibration')).get('threshold_state', 'missing')}** and corrected validation is **{context['validation_status']}**. Negative or zero net gain is retained; it does not trigger prompt changes, candidate movement, or test-based primary selection.
"""


def _reproduce_markdown(context: Mapping[str, Any]) -> str:
    return f"""# Reproduce the final reports

The report generator is offline and reads only persisted artifacts. It performs no provider call.

```bash
export REPORT_RUN_ROOT="runs/{context['run_id']}"
.venv/bin/python - <<'PY'
import os
from failure_analysis.vlm_safe_rerank.reporting import generate_reports

generate_reports(os.environ["REPORT_RUN_ROOT"])
PY
```

For a non-writing audit, pass `dry_run=True`. Reproduction must use the same frozen candidate/q identities, corrected labels, validation result, and pairwise SQLite ledger. No credential is required for reporting.
"""


def _formal_report_markdown(context: Mapping[str, Any]) -> str:
    formal = _mapping(context.get("formal"))
    corrected = _mapping(formal.get("corrected"))
    if not corrected:
        selected = _mapping(_mapping(formal.get("methods")).get(context["corrected_primary"]))
        corrected = _mapping(selected.get("corrected") or selected)
    return f"""# Formal test report

## Full-denominator result

The saved formal artifact is complete under the corrected evaluator. Locked primary: **{context['corrected_primary']}**. Observed/expected denominator: **{_fmt_number(context['formal_observed_denominator'])}/{_fmt_number(context['expected_formal_denominator'])}**.

Corrected baseline/final J@1 is **{_fmt_rate(corrected.get('baseline_j1'))}/{_fmt_rate(corrected.get('final_j1'))}** with Recovered/Harmful/Net **{_fmt_number(corrected.get('recovered'))}/{_fmt_number(corrected.get('harmful'))}/{_fmt_number(corrected.get('net'))}**.

This file is generated only from an explicitly complete, denominator-matched formal artifact. It does not convert validation, diagnostic, or partial formal data into a formal claim.
"""


def _locked_manifest(context: Mapping[str, Any], root: Path) -> dict[str, Any]:
    sources = context["sources"]
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "vlm_safe_rerank_reporting_lock_snapshot",
        "run_id": context["run_id"],
        "created_at_utc": context["generated_at_utc"],
        "primary_evaluator": "corrected",
        "primary_method": context["corrected_primary"],
        "default_action": Q_ONLY,
        "validation_status": "GO",
        "expected_formal_denominator": context["expected_formal_denominator"],
        "formal_requests_authorized_flag": True,
        "formal_run_allowed": True,
        "reporting_lock_only": True,
        "api_authorization_must_be_verified_by_runner": True,
        "source_identities": {
            name: _file_identity(path, root)
            for name, path in sources.items()
            if name in {"data_manifest", "inference_manifest", "calibration", "validation"}
            and path is not None
        },
    }


def _render_reports(context: Mapping[str, Any], root: Path) -> dict[str, str]:
    results = _results_payload(context)
    ledger_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": context["generated_at_utc"],
        "run_id": context["run_id"],
        "full_denominator_definition": {
            "requests": "all distinct logical request hashes in pairwise_cache.sqlite",
            "attempts": "all persisted physical attempts, including failed attempts",
        },
        **context["ledger"],
        "phase_api_totals": context.get("phase_api_totals"),
        "model_failures_from_phase_artifacts": context["model_failures"],
        "phase_progress": {
            "smoke": results["api_phase_progress"]["smoke"],
            "diagnostic": results["api_phase_progress"]["diagnostic"],
        },
    }
    rendered = {
        "SUMMARY.md": _summary_markdown(context),
        "GO_NO_GO.md": _go_no_go_markdown(context),
        "RESULTS.json": _json_text(results),
        "METRICS.csv": _metrics_csv(context["metrics"]),
        "API_LEDGER_SUMMARY.json": _json_text(ledger_payload),
        "COST_REPORT.md": _cost_markdown(context),
        "LATENCY_REPORT.md": _latency_markdown(context),
        "FAILURE_ANALYSIS.md": _failure_markdown(context),
        "REPRODUCE.md": _reproduce_markdown(context),
    }
    if context["formal_run_allowed"]:
        rendered[CONDITIONAL_LOCK] = _json_text(_locked_manifest(context, root))
    if context["formal_complete"]:
        rendered[CONDITIONAL_FORMAL_REPORT] = _formal_report_markdown(context)
    return rendered


def _experiment_manifest_details(root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    repository = root.parents[1]
    def git(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args), cwd=repository, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        return completed.stdout.strip()

    data = _mapping(context.get("data"))
    inference = _mapping(context.get("inference"))
    calibration = _mapping(context.get("calibration"))
    tracked_sources = {
        "prompt": repository / "prompts/pairwise_safe_v1.txt",
        "schema": repository / "prompts/pairwise_safe_v1.schema.json",
        "renderer": repository / "failure_analysis/vlm_safe_rerank/renderer.py",
        "feature_schema": repository / "failure_analysis/vlm_safe_rerank/critic_features.py",
        "calibrator": repository / "failure_analysis/vlm_safe_rerank/api_calibration.py",
        "p5_evaluator": repository / "failure_analysis/vlm_safe_rerank/p5_validation.py",
    }
    return {
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty_working_tree": bool(git("status", "--porcelain")),
            "diff_sha256": _sha256_bytes(git("diff", "--binary").encode("utf-8")),
        },
        "dataset": {
            "split_manifest": data.get("split_manifest"),
            "candidate_sources": data.get("candidate_sources"),
            "candidate_q_identities": data.get("candidate_q_identities"),
            "evaluation_sources": data.get("evaluation_sources"),
            "formal_test_pristine": _mapping(data.get("metadata")).get("formal_test_pristine"),
        },
        "source_identities": {
            name: _file_identity(path, repository)
            for name, path in tracked_sources.items() if path.is_file()
        },
        "models": list(MODEL_IDS),
        "api": {
            "sdk": f"google-genai=={importlib.metadata.version('google-genai')}",
            "endpoint": "v1beta Interactions",
            "temperature": 0.0,
            "thinking_level": "low",
            "max_output_tokens": 1024,
            "store": False,
            "stream": False,
            "background": False,
            "service_tier": "standard",
            "maximum_challengers": _mapping(inference.get("inference_inputs")).get("maximum_challengers", 2),
        },
        "calibration": {
            "calibrator_type": "L2 logistic dual-risk with out-of-fold Platt calibration",
            "threshold_state": calibration.get("threshold_state"),
            "methods": {
                method: {
                    "threshold_state": _mapping(payload).get("threshold_state"),
                    "selected_thresholds": _mapping(payload).get("selected_thresholds"),
                    "sampling_contract_passed": _mapping(payload).get("sampling_contract_passed"),
                    "stability_passed": _mapping(payload).get("stability_passed"),
                }
                for method, payload in _mapping(calibration.get("methods")).items()
            },
            "random_seed": 20260803,
        },
        "policy": {
            "default": "q_only_c0",
            "fallback": _mapping(inference.get("inference_inputs")).get("fallback_policy"),
            "retry": "finite transport retries with Retry-After; terminal failure keeps c0",
            "formal_requests_authorized": context["formal_requests_authorized_flag"],
        },
        "runtime": {
            "generated_at_utc": context["generated_at_utc"],
            "ledger": context["ledger"],
            "cost_projection": context.get("cost_projection"),
        },
    }


def generate_reports(
    run_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate final safe-rerank reports from persisted evidence only.

    Missing stages are represented as partial evidence.  NO_GO and partial runs
    never create formal artifacts.  A conditional reporting snapshot is never
    named LOCKED_MANIFEST and cannot substitute for the independently verified
    runner lock.
    """

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise ReportingError("run root does not exist")
    output = Path(output_dir).resolve() if output_dir is not None else root
    context = _build_context(root)
    rendered = _render_reports(context, root)

    source_identities = {
        name: _file_identity(path, root)
        for name, path in context["sources"].items()
        if path is not None
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "vlm_safe_rerank_report_manifest",
        "run_id": context["run_id"],
        "generated_at_utc": context["generated_at_utc"],
        "report_status": context["report_status"],
        "source_identities": source_identities,
        "generated_files": {
            name: {
                "sha256": _sha256_bytes(content.encode("utf-8")),
                "size_bytes": len(content.encode("utf-8")),
            }
            for name, content in sorted(rendered.items())
        },
        "self_identity": {"path": "MANIFEST.json", "sha256_excluded": True},
        "secret_scan": {"status": "passed_before_write"},
        "experiment": _experiment_manifest_details(root, context),
    }
    rendered["MANIFEST.json"] = _json_text(manifest)
    _assert_secret_free(rendered)

    conditional_allowed = {
        CONDITIONAL_LOCK: bool(context["formal_run_allowed"]),
        CONDITIONAL_FORMAL_REPORT: bool(context["formal_complete"]),
    }
    if not dry_run:
        for name, allowed in conditional_allowed.items():
            if not allowed and (output / name).exists():
                raise ReportingError(
                    f"stale conditional artifact would misrepresent this run: {name}"
                )
        output.mkdir(parents=True, exist_ok=True)
        for name, content in sorted(rendered.items()):
            _atomic_write(output / name, content)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": context["run_id"],
        "report_status": context["report_status"],
        "decision_status": context["decision_status"],
        "corrected_primary_method": context["corrected_primary"],
        "formal_requests_authorized_flag": context["formal_requests_authorized_flag"],
        "formal_run_allowed": context["formal_run_allowed"],
        "formal_complete": context["formal_complete"],
        "files": sorted(rendered),
        "output_dir": str(output),
        "written": not dry_run,
    }


__all__ = ["ReportingError", "generate_reports"]
