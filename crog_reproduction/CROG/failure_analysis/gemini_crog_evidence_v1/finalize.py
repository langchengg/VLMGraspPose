"""Offline publication and integrity finalizer for Gemini CROG Evidence V1.

This module never imports the provider client and never sends API requests.  It
only derives publication artifacts from durable phase outputs, the canonical
SQLite cache, and frozen local inputs.  Incomplete runs remain explicitly
incomplete: formal claims require a formal completion marker, a valid lock, and
successful independent recomputation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .api import estimate_usage_cost_usd
from .audit import (
    PREEXISTING_V2_TREE_SHA256,
    audit_frozen_test_baseline,
)
from .run_state import atomic_write_json, utc_now
from .protocol import sha256_file, verify_experiment_lock


ER2_COST_DISCLAIMER = (
    "Published ER2 token prices were available from the official Gemini pricing table; "
    "the reported ER2 amount is a usage-based estimate rather than an independently "
    "verified provider invoice, and the experiment retained a conservative configurable "
    "per-request budget reserve."
)
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"

MODEL_IDS = ("gemini-robotics-er-2-preview", "gemini-3.6-flash")
CANONICAL_PHASE_COUNTS = {
    "pilot": 100,
    "stability": 20,
    "ablation": 500,
    "calibration": 9_790,
    "validation": 8_669,
    "formal_test": 17_749,
}
PHASE_MANIFESTS = {
    phase: f"{phase}_manifest.json" for phase in CANONICAL_PHASE_COUNTS
}
PHASE_ORDER = tuple(CANONICAL_PHASE_COUNTS)
MAX_PLANNED_REQUEST_SLOTS = 76_736

REQUIRED_ROOT_ARTIFACTS = (
    "results_bundle.json",
    "per_method_metrics.csv",
    "per_sample_predictions.parquet",
    "per_model_decisions.parquet",
    "per_candidate_evidence.parquet",
    "per_candidate_gemini_scores.parquet",
    "request_manifest.parquet",
    "candidate_mapping.parquet",
    "statistical_tests.json",
    "bootstrap_intervals.json",
    "threshold_sweeps.csv",
    "model_agreement.json",
    "stability_results.json",
    "api_runtime_metrics.json",
    "api_token_usage.csv",
    "cost_estimate.json",
    "actual_cost_estimate.json",
    "fallback_summary.json",
    "environment.json",
    "commands.log",
    "independent_recompute_results.json",
    "gemini_cache.sqlite",
)

_GENERIC_SECRET_PATTERNS = (
    re.compile(rb"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(rb"AQ\.[0-9A-Za-z_-]{20,}"),
)
_SCAN_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".parquet",
    ".py",
    ".sqlite",
    ".txt",
    ".yaml",
    ".yml",
}


class FinalizationError(RuntimeError):
    """Raised when publication would violate an experiment invariant."""


@dataclass(frozen=True)
class FrozenInputPaths:
    features: Path
    legacy_labels: Path
    corrected_labels: Path
    v2_root: Path


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _atomic_copy(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    _atomic_bytes(destination, source.read_bytes())


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist([dict(row) for row in rows], schema=schema), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _table_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist() if path.is_file() else []


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def audit_cache(cache_path: str | Path) -> dict[str, Any]:
    """Independently check cache integrity and successful-hash uniqueness."""

    path = Path(cache_path)
    if not path.is_file():
        raise FinalizationError(f"canonical cache is missing: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise FinalizationError("SQLite integrity_check failed")
        tables = _sqlite_tables(connection)
        if "responses" not in tables:
            raise FinalizationError("canonical cache has no responses table")
        response_count = int(connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
        distinct_response_hashes = int(
            connection.execute("SELECT COUNT(DISTINCT request_hash) FROM responses").fetchone()[0]
        )
        if response_count != distinct_response_hashes:
            raise FinalizationError("duplicate response request_hash detected")
        valid_responses = int(
            connection.execute("SELECT COUNT(*) FROM responses WHERE valid = 1").fetchone()[0]
        )
        invalid_responses = response_count - valid_responses
        attempt_counts: Counter[str] = Counter()
        duplicate_successful_hashes = 0
        ledger_attempts = retries = 0
        imported_response_without_attempt_count = response_count
        if "request_attempts" in tables:
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM request_attempts GROUP BY status"
            ):
                attempt_counts[str(row["status"])] = int(row["count"])
            ledger_attempts = sum(attempt_counts.values())
            imported_response_without_attempt_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM responses r WHERE NOT EXISTS ("
                    "SELECT 1 FROM request_attempts a WHERE a.request_hash=r.request_hash)"
                ).fetchone()[0]
            )
            retries = int(
                connection.execute(
                    "SELECT COALESCE(SUM(CASE WHEN attempts > 1 THEN attempts - 1 ELSE 0 END), 0) "
                    "FROM (SELECT COUNT(*) AS attempts FROM request_attempts GROUP BY request_hash)"
                ).fetchone()[0]
            )
            duplicate_successful_hashes = int(
                connection.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT request_hash FROM request_attempts WHERE status='SUCCEEDED' "
                    "GROUP BY request_hash HAVING COUNT(*) > 1)"
                ).fetchone()[0]
            )
        if duplicate_successful_hashes:
            raise FinalizationError("duplicate successful request_hash detected")
        state_counts: Counter[str] = Counter()
        if "request_states" in tables:
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM request_states GROUP BY status"
            ):
                state_counts[str(row["status"])] = int(row["count"])
        return {
            "status": "passed",
            "sqlite_integrity_check": integrity,
            "response_count": response_count,
            "distinct_response_hash_count": distinct_response_hashes,
            "valid_response_count": valid_responses,
            "invalid_response_count": invalid_responses,
            "attempt_count": ledger_attempts + imported_response_without_attempt_count,
            "ledger_attempt_count": ledger_attempts,
            "imported_response_without_attempt_count": imported_response_without_attempt_count,
            "retry_count": retries,
            "attempt_status_counts": dict(attempt_counts),
            "request_state_counts": dict(state_counts),
            "duplicate_successful_request_hash_count": 0,
        }
    finally:
        connection.close()


def _manifest_row_count(payload: Mapping[str, Any]) -> int:
    if isinstance(payload.get("rows"), list):
        return len(payload["rows"])
    return int(payload.get("sample_count", -1))


def audit_manifest_request_counts(
    run_root: str | Path,
    *,
    enforce_canonical_counts: bool = True,
) -> dict[str, Any]:
    """Check frozen cohort sizes and request-count arithmetic without requiring completion."""

    root = Path(run_root)
    plan = _read_json(root / "full_run_plan.json")
    if not isinstance(plan, dict):
        raise FinalizationError("full_run_plan.json is missing or invalid")
    phase_plan = {str(row["phase"]): row for row in plan.get("phases", [])}
    observed: dict[str, Any] = {}
    slot_sum = 0
    errors: list[str] = []
    for phase, manifest_name in PHASE_MANIFESTS.items():
        manifest = _read_json(root / manifest_name)
        if not isinstance(manifest, dict):
            errors.append(f"missing {manifest_name}")
            continue
        count = _manifest_row_count(manifest)
        expected = CANONICAL_PHASE_COUNTS[phase]
        if enforce_canonical_counts and count != expected:
            errors.append(f"{phase} manifest count {count} != {expected}")
        planned = phase_plan.get(phase)
        if planned is None:
            errors.append(f"missing {phase} in full_run_plan")
            continue
        if int(planned.get("sample_count", -1)) != count:
            errors.append(f"{phase} plan/manifest sample count mismatch")
        protocols = len(planned.get("protocols", []))
        replicates = len(planned.get("replicates", []))
        models = len(planned.get("model_ids", []))
        computed_slots = count * protocols * replicates * models
        declared_slots = int(planned.get("planned_request_slots", -1))
        if computed_slots != declared_slots:
            errors.append(
                f"{phase} request slots {declared_slots} != {count}x{protocols}x{replicates}x{models}"
            )
        slot_sum += max(0, declared_slots)
        decision_path = root / phase / "per_model_decisions.parquet"
        marker = _read_json(root / phase / "PHASE_COMPLETE.json")
        terminal_rows = pq.read_metadata(decision_path).num_rows if decision_path.is_file() else 0
        if isinstance(marker, dict) and marker.get("status") == "complete" and terminal_rows != declared_slots:
            errors.append(f"{phase} completion marker has {terminal_rows}/{declared_slots} decisions")
        observed[phase] = {
            "sample_count": count,
            "planned_request_slots": declared_slots,
            "new_requests": int(planned.get("new_requests", 0)),
            "cache_hits": int(planned.get("cache_hits", 0)),
            "terminal_decision_rows": terminal_rows,
            "complete": bool(isinstance(marker, dict) and marker.get("status") == "complete"),
        }
    if slot_sum > MAX_PLANNED_REQUEST_SLOTS:
        errors.append(f"planned request slots {slot_sum} exceed cap {MAX_PLANNED_REQUEST_SLOTS}")
    totals = plan.get("totals", {})
    if int(totals.get("planned_request_slots", -1)) != slot_sum:
        errors.append("full_run_plan totals.planned_request_slots mismatch")
    planned_unique = int(totals.get("planned_unique_requests", -1))
    new_requests = int(totals.get("new_requests", -1))
    existing_hits = int(totals.get("existing_cache_hits", 0))
    if planned_unique != new_requests + existing_hits:
        errors.append("planned_unique_requests != new_requests + existing_cache_hits")
    if planned_unique > MAX_PLANNED_REQUEST_SLOTS:
        errors.append("planned_unique_requests exceeds registered cap")
    if errors:
        raise FinalizationError("manifest/request audit failed: " + "; ".join(errors))
    return {
        "status": "passed",
        "phases": observed,
        "planned_request_slots": slot_sum,
        "planned_unique_requests": planned_unique,
        "new_requests": new_requests,
        "existing_cache_hits": existing_hits,
        "cross_phase_cache_hits": int(totals.get("cross_phase_cache_hits", 0)),
        "request_cap": MAX_PLANNED_REQUEST_SLOTS,
    }


def _v2_tree_digest(repo_root: Path, v2_root: Path) -> str:
    try:
        relative = v2_root.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise FinalizationError("V2 tree must be below repository root") from exc
    command = (
        f"find {shlex.quote(str(relative))} -type f -print0 | "
        "LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256"
    )
    result = subprocess.run(
        command,
        shell=True,
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.split()[0]


def frozen_input_paths_from_passport(run_root: str | Path, repo_root: str | Path) -> FrozenInputPaths:
    root = Path(run_root)
    repo = Path(repo_root)
    passport = _read_json(root / "phase0" / "material_passport.json")
    if not isinstance(passport, dict) or not isinstance(passport.get("sources"), dict):
        raise FinalizationError("phase0 material passport is missing")
    sources = passport["sources"]
    return FrozenInputPaths(
        features=Path(sources["frozen_test_candidates"]),
        legacy_labels=Path(sources["legacy_evaluation_labels"]),
        corrected_labels=Path(sources["corrected_evaluation_labels"]),
        v2_root=repo / "failure_analysis/reranking_outputs/v2_20260727T174412+0100",
    )


def audit_frozen_inputs(
    *,
    paths: FrozenInputPaths,
    repo_root: str | Path,
    expected_candidate_sha256: str,
    expected_v2_sha256: str = PREEXISTING_V2_TREE_SHA256,
    verify_v2_tree: bool = True,
) -> dict[str, Any]:
    baseline = audit_frozen_test_baseline(
        features_path=paths.features,
        legacy_labels_path=paths.legacy_labels,
        corrected_labels_path=paths.corrected_labels,
    )
    if baseline["candidate_identity_stream_sha256"] != expected_candidate_sha256:
        raise FinalizationError("frozen candidate identity changed")
    v2_sha = _v2_tree_digest(Path(repo_root), paths.v2_root) if verify_v2_tree else None
    if verify_v2_tree and v2_sha != expected_v2_sha256:
        raise FinalizationError("pre-existing V2 tree identity changed")
    return {
        "status": "passed",
        "baseline": baseline,
        "candidate_identity_sha256": baseline["candidate_identity_stream_sha256"],
        "v2_tree_sha256": v2_sha,
        "expected_v2_tree_sha256": expected_v2_sha256,
        "v2_tree_verified": verify_v2_tree,
    }


def _secret_scan_files(scopes: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for scope in scopes:
        if scope.is_file():
            candidates = [scope]
        elif scope.is_dir():
            candidates = scope.rglob("*")
        else:
            continue
        for path in candidates:
            if not path.is_file() or path.name in {".env", ".env.local"}:
                continue
            if path.suffix.lower() not in _SCAN_SUFFIXES and path.name != "gemini_cache.sqlite-wal":
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def secret_scan(
    scopes: Sequence[str | Path],
    *,
    secret_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Scan publication surfaces without ever returning secret material."""

    exact = [value.encode("utf-8") for value in secret_values if value]
    hit_files: list[str] = []
    files_scanned = 0
    for path in _secret_scan_files([Path(scope) for scope in scopes]):
        files_scanned += 1
        data = path.read_bytes()
        if any(value in data for value in exact) or any(pattern.search(data) for pattern in _GENERIC_SECRET_PATTERNS):
            hit_files.append(str(path))
    if hit_files:
        # Deliberately omit values, hashes, prefixes, suffixes, and length data.
        raise FinalizationError(f"secret scan found credentials in {len(hit_files)} file(s)")
    return {"status": "passed", "files_scanned": files_scanned, "credential_match_file_count": 0}


def _valid_evidence_directory(path: Path) -> bool:
    if any((ancestor / marker).exists() for ancestor in (path, *path.parents) for marker in ("INVALIDATED.json", "SUPERSEDED.json")):
        return False
    summary = _read_json(path / "export_summary.json")
    return bool(
        isinstance(summary, dict)
        and summary.get("status") == "complete"
        and (path / "candidate_evidence.parquet").is_file()
        and (path / "candidate_mapping.parquet").is_file()
        and (path / "request_manifest.parquet").is_file()
    )


def discover_evidence_sources(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    candidates = {path.parent for path in root.rglob("export_summary.json")}
    return sorted(path for path in candidates if _valid_evidence_directory(path))


def _aggregate_parquet(
    sources: Sequence[Path],
    destination: Path,
    *,
    filename: str,
    key_fields: Sequence[str],
) -> dict[str, Any]:
    rows_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    schema: pa.Schema | None = None
    duplicate_rows = 0
    for source in sources:
        path = source / filename
        table = pq.read_table(path)
        if schema is None:
            schema = table.schema
        elif table.schema != schema:
            raise FinalizationError(f"evidence schema drift in {filename}: {source}")
        for row in table.to_pylist():
            key = tuple(str(row[field]) for field in key_fields)
            previous = rows_by_key.get(key)
            if previous is not None:
                duplicate_rows += 1
                if previous != row:
                    raise FinalizationError(f"conflicting repeated evidence row in {filename}")
            else:
                rows_by_key[key] = row
    if schema is None:
        raise FinalizationError(f"no valid evidence source for {filename}")
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    _atomic_parquet(destination, rows, schema)
    return {
        "source_count": len(sources),
        "row_count": len(rows),
        "deduplicated_identical_row_count": duplicate_rows,
    }


def aggregate_evidence(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    sources = discover_evidence_sources(root)
    if not sources:
        raise FinalizationError("no valid candidate evidence sources were found")
    candidate = _aggregate_parquet(
        sources,
        root / "per_candidate_evidence.parquet",
        filename="candidate_evidence.parquet",
        key_fields=("stable_candidate_id",),
    )
    mapping = _aggregate_parquet(
        sources,
        root / "candidate_mapping.parquet",
        filename="candidate_mapping.parquet",
        key_fields=("sample_id",),
    )
    requests = _aggregate_parquet(
        sources,
        root / "evidence_request_manifest.parquet",
        filename="request_manifest.parquet",
        key_fields=("sample_id",),
    )
    return {
        "status": "passed",
        "sources": [str(path.relative_to(root)) for path in sources],
        "candidate_evidence": candidate,
        "candidate_mapping": mapping,
        "evidence_request_manifest": requests,
    }


def build_logical_request_manifest(run_root: str | Path) -> dict[str, Any]:
    """Materialize one row for every preregistered logical request slot."""

    root = Path(run_root)
    plan = _read_json(root / "full_run_plan.json")
    if not isinstance(plan, dict):
        raise FinalizationError("full_run_plan.json is missing or invalid")
    rows: list[dict[str, Any]] = []
    for phase_plan in plan.get("phases", []):
        phase = str(phase_plan["phase"])
        manifest = _read_json(root / f"{phase}_manifest.json")
        if not isinstance(manifest, dict) or not isinstance(manifest.get("rows"), list):
            raise FinalizationError(f"logical request manifest lacks {phase}_manifest.json")
        for sample in manifest["rows"]:
            sample_id = str(sample["sample_id"])
            for protocol in phase_plan.get("protocols", []):
                for replicate in phase_plan.get("replicates", []):
                    for model in phase_plan.get("model_ids", []):
                        logical_key = "|".join(
                            (phase, str(protocol), str(model), sample_id, str(int(replicate)))
                        )
                        decision_name = hashlib.sha256(logical_key.encode("utf-8")).hexdigest() + ".json"
                        decision = _read_json(root / phase / "decisions" / decision_name, {})
                        if not isinstance(decision, dict):
                            decision = {}
                        rows.append(
                            {
                                "phase": phase,
                                "protocol_id": str(protocol),
                                "replicate_id": int(replicate),
                                "model_id": str(model),
                                "sample_id": sample_id,
                                "logical_request_key": logical_key,
                                "request_hash": str(decision.get("request_hash", "")),
                                "lifecycle_status": str(decision.get("lifecycle_status", "PLANNED")),
                                "terminal": bool(decision),
                                "valid_response": bool(decision.get("valid", False)),
                                "cache_hit": bool(decision.get("cache_hit", False)),
                                "api_attempted": bool(decision.get("api_attempted", False)),
                                "retry_count": int(decision.get("retry_count", 0)),
                                "request_id": str(decision.get("request_id") or ""),
                                "technical_fallback": bool(decision.get("technical_fallback", False)),
                                "permanent_api_failure": bool(decision.get("permanent_api_failure", False)),
                            }
                        )
    expected = int((plan.get("totals") or {}).get("planned_request_slots", -1))
    if len(rows) != expected or len({row["logical_request_key"] for row in rows}) != len(rows):
        raise FinalizationError("logical request manifest count or identity mismatch")
    schema = pa.schema(
        [
            ("phase", pa.string()), ("protocol_id", pa.string()),
            ("replicate_id", pa.int64()), ("model_id", pa.string()),
            ("sample_id", pa.string()), ("logical_request_key", pa.string()),
            ("request_hash", pa.string()), ("lifecycle_status", pa.string()),
            ("terminal", pa.bool_()), ("valid_response", pa.bool_()),
            ("cache_hit", pa.bool_()), ("api_attempted", pa.bool_()),
            ("retry_count", pa.int64()), ("request_id", pa.string()),
            ("technical_fallback", pa.bool_()), ("permanent_api_failure", pa.bool_()),
        ]
    )
    _atomic_parquet(root / "request_manifest.parquet", rows, schema)
    return {
        "status": "complete",
        "planned_slots": len(rows),
        "terminal_slots": sum(bool(row["terminal"]) for row in rows),
        "planned_slots_without_request_hash": sum(not row["request_hash"] for row in rows),
    }


def _usage_value(usage: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if name in usage and usage[name] is not None:
            return int(usage[name])
    return 0


def cache_derived_artifacts(run_root: str | Path) -> dict[str, Any]:
    """Derive attempt-level usage/cost and response-level candidate scores."""

    root = Path(run_root)
    connection = sqlite3.connect(f"file:{root / 'gemini_cache.sqlite'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    score_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    response_fallbacks: Counter[str] = Counter()
    try:
        responses = connection.execute("SELECT * FROM responses ORDER BY request_hash").fetchall()
        response_by_hash = {str(row["request_hash"]): row for row in responses}
        for response in responses:
            model = str(response["model_id"])
            protocol = str(response["protocol_id"] or "p1_full_crog_evidence")
            response_fallbacks[str(response["fallback_reason"] or "none")] += 1
            try:
                parsed = json.loads(response["parsed_output_json"] or "null")
                display_to_candidate = json.loads(response["mapping_json"])["display_to_candidate"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if not isinstance(parsed, dict) or not isinstance(parsed.get("ranking"), list):
                continue
            for rank, item in enumerate(parsed["ranking"], start=1):
                if not isinstance(item, dict):
                    continue
                display_id = str(item.get("candidate_id", ""))
                candidate_id = display_to_candidate.get(display_id)
                if candidate_id is None:
                    continue
                score_rows.append(
                    {
                        "request_hash": str(response["request_hash"]),
                        "sample_id": str(response["sample_id"]),
                        "model_id": model,
                        "protocol_id": protocol,
                        "replicate_id": "" if response["replicate_id"] is None else str(response["replicate_id"]),
                        "rank": rank,
                        "display_candidate_id": display_id,
                        "candidate_id": str(candidate_id),
                        "stable_candidate_id": f"{response['sample_id']}/{candidate_id}",
                        **{
                            name: float(item.get(name, 0.0))
                            for name in (
                                "target_alignment_score", "mask_support_score",
                                "quality_evidence_score", "angle_consistency_score",
                                "width_consistency_score", "edge_safety_score", "overall_score",
                            )
                        },
                        "reason_codes_json": json.dumps(item.get("reason_codes", []), sort_keys=True),
                    }
                )

        tables = _sqlite_tables(connection)
        attempts = []
        if "request_attempts" in tables:
            attempts = connection.execute(
                "SELECT * FROM request_attempts ORDER BY request_hash, attempt_number"
            ).fetchall()
        states = {
            str(row["request_hash"]): dict(row)
            for row in (
                connection.execute("SELECT * FROM request_states").fetchall()
                if "request_states" in tables
                else []
            )
        }
        ledgered = {str(row["request_hash"]) for row in attempts}
        for attempt in attempts:
            request_hash = str(attempt["request_hash"])
            response = response_by_hash.get(request_hash)
            state = states.get(request_hash, {})
            usage_rows.append(
                {
                    "source": "request_attempts",
                    "request_hash": request_hash,
                    "attempt_number": int(attempt["attempt_number"]),
                    "attempt_status": str(attempt["status"]),
                    "sample_id": str(state.get("sample_id") or (response["sample_id"] if response else "")),
                    "model_id": str(state.get("model_id") or (response["model_id"] if response else "")),
                    "protocol_id": str(state.get("protocol_id") or (response["protocol_id"] if response else "") or "p1_full_crog_evidence"),
                    "replicate_id": str(state.get("replicate_id") or ""),
                    "namespace": str(state.get("namespace") or ""),
                    "valid": bool(response["valid"]) if response and attempt["status"] == "SUCCEEDED" else False,
                    "latency_seconds": float(
                        attempt["latency_seconds"]
                        if "latency_seconds" in attempt.keys()
                        else (response["latency_seconds"] if response else 0.0)
                    ),
                    "usage": json.loads(
                        attempt["usage_json"]
                        if "usage_json" in attempt.keys()
                        else (response["usage_json"] if response else "{}")
                    ),
                    "estimated_charge_usd": float(
                        attempt["estimated_charge_usd"]
                        if "estimated_charge_usd" in attempt.keys()
                        else (response["estimated_charge_usd"] if response else 0.0)
                    ),
                }
            )
        # The imported Phase-B smoke cache predates the attempt ledger.  Each
        # such response is still one real provider attempt and remains billable.
        for request_hash, response in response_by_hash.items():
            if request_hash in ledgered:
                continue
            usage_rows.append(
                {
                    "source": "imported_response",
                    "request_hash": request_hash,
                    "attempt_number": 1,
                    "attempt_status": "SUCCEEDED" if bool(response["valid"]) else "TERMINAL_RESPONSE",
                    "sample_id": str(response["sample_id"]),
                    "model_id": str(response["model_id"]),
                    "protocol_id": str(response["protocol_id"] or "p1_full_crog_evidence"),
                    "replicate_id": str(response["replicate_id"] or ""),
                    "namespace": str(response["namespace"] or ""),
                    "valid": bool(response["valid"]),
                    "latency_seconds": float(response["latency_seconds"] or 0.0),
                    "usage": json.loads(response["usage_json"] or "{}"),
                    "estimated_charge_usd": float(response["estimated_charge_usd"] or 0.0),
                }
            )
    finally:
        connection.close()

    token_totals: Counter[str] = Counter()
    model_totals: dict[str, Counter[str]] = defaultdict(Counter)
    latency_by_model: dict[str, list[float]] = defaultdict(list)
    model_charges: Counter[str] = Counter()
    request_hashes_by_model: dict[str, set[str]] = defaultdict(set)
    csv_usage_rows: list[dict[str, Any]] = []
    for row in usage_rows:
        usage = row.pop("usage")
        input_tokens = _usage_value(usage, "total_input_tokens", "input_tokens", "prompt_token_count")
        output_tokens = _usage_value(usage, "total_output_tokens", "output_tokens", "candidates_token_count")
        thought_tokens = _usage_value(usage, "total_thought_tokens", "thought_tokens", "thoughts_token_count")
        total_tokens = _usage_value(usage, "total_tokens", "total_token_count") or input_tokens + output_tokens + thought_tokens
        published_cost = (
            estimate_usage_cost_usd(str(row["model_id"]), usage)
            if input_tokens + output_tokens + thought_tokens > 0
            else float(row["estimated_charge_usd"])
        )
        enriched = {
            **row,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thought_tokens": thought_tokens,
            "total_tokens": total_tokens,
            "estimated_charge_usd": published_cost,
        }
        csv_usage_rows.append(enriched)
        model = str(row["model_id"])
        token_totals.update(input_tokens=input_tokens, output_tokens=output_tokens, thought_tokens=thought_tokens, total_tokens=total_tokens)
        model_totals[model].update(attempts=1, valid=int(bool(row["valid"])), input_tokens=input_tokens, output_tokens=output_tokens, thought_tokens=thought_tokens)
        latency_by_model[model].append(float(row["latency_seconds"]))
        model_charges[model] += published_cost
        request_hashes_by_model[model].add(str(row["request_hash"]))
    retry_count = sum(max(0, count - 1) for count in Counter(row["request_hash"] for row in usage_rows).values())
    usage_fields = (
        "source", "request_hash", "attempt_number", "attempt_status", "sample_id", "model_id",
        "protocol_id", "replicate_id", "namespace", "valid", "latency_seconds", "input_tokens",
        "output_tokens", "thought_tokens", "total_tokens", "estimated_charge_usd",
    )
    _atomic_csv(root / "api_token_usage.csv", csv_usage_rows, usage_fields)
    score_schema = pa.schema(
        [
            ("request_hash", pa.string()), ("sample_id", pa.string()), ("model_id", pa.string()),
            ("protocol_id", pa.string()), ("replicate_id", pa.string()), ("rank", pa.int64()),
            ("display_candidate_id", pa.string()), ("candidate_id", pa.string()),
            ("stable_candidate_id", pa.string()),
            ("target_alignment_score", pa.float64()), ("mask_support_score", pa.float64()),
            ("quality_evidence_score", pa.float64()), ("angle_consistency_score", pa.float64()),
            ("width_consistency_score", pa.float64()), ("edge_safety_score", pa.float64()),
            ("overall_score", pa.float64()), ("reason_codes_json", pa.string()),
        ]
    )
    _atomic_parquet(root / "per_candidate_gemini_scores.parquet", score_rows, score_schema)
    all_latencies = [value for values in latency_by_model.values() for value in values]
    runtime = {
        "status": "complete_from_persisted_cache",
        "attempt_count": len(usage_rows),
        "retry_count": retry_count,
        "p50_latency_seconds": _percentile(all_latencies, 0.50),
        "p95_latency_seconds": _percentile(all_latencies, 0.95),
        "by_model": {
            model: {
                **dict(model_totals[model]),
                "unique_request_count": len(request_hashes_by_model[model]),
                "p50_latency_seconds": _percentile(latency_by_model[model], 0.50),
                "p95_latency_seconds": _percentile(latency_by_model[model], 0.95),
            }
            for model in sorted(model_totals)
        },
    }
    runtime_progress = _read_json(root / "runtime_progress.json", {})
    if isinstance(runtime_progress, dict):
        runtime.update(
            {
                "wall_time_seconds": runtime_progress.get("wall_time_seconds"),
                "requests_per_hour": runtime_progress.get("requests_per_hour"),
                "eta_seconds": runtime_progress.get("eta_seconds"),
                "run_id": runtime_progress.get("run_id"),
            }
        )
    atomic_write_json(root / "api_runtime_metrics.json", runtime)

    plan = _read_json(root / "full_run_plan.json", {})
    expected_cost = (plan.get("totals") or {}).get("expected_cost", {})
    atomic_write_json(
        root / "cost_estimate.json",
        {
            "status": "planned_budget_estimate",
            "source": "full_run_plan.json",
            "planned_unique_requests": (plan.get("totals") or {}).get("planned_unique_requests"),
            "expected_cost": expected_cost,
            "pricing_source": PRICING_SOURCE,
            "er2_cost_note": ER2_COST_DISCLAIMER,
        },
    )
    preflight = _read_json(root / "full_run_preflight.json", {})
    er2_cap = ((preflight.get("environment") or {}).get("gemini_er2_cost_cap_per_request_usd"))
    max_budget = ((preflight.get("environment") or {}).get("gemini_max_spend_usd"))
    er2_unique_request_count = len(request_hashes_by_model[MODEL_IDS[0]])
    er2_reserve = None if er2_cap is None else er2_unique_request_count * float(er2_cap)
    total_estimated_charge = sum(model_charges.values())
    actual_cost = {
        "status": "estimated_from_persisted_usage",
        "attempt_count": len(usage_rows),
        "retry_count": retry_count,
        "unique_request_count": len({row["request_hash"] for row in usage_rows}),
        "token_usage": dict(token_totals),
        "flash_estimated_cost_usd": model_charges[MODEL_IDS[1]],
        "er2_published_rate_estimated_cost_usd": model_charges[MODEL_IDS[0]],
        "er2_unique_request_count": er2_unique_request_count,
        "er2_conservative_cost_cap_per_request_usd": er2_cap,
        "er2_conservative_budget_reserve_usd": er2_reserve,
        "published_rate_estimated_total_usd": total_estimated_charge,
        "conservative_budget_use_usd": (
            None
            if er2_reserve is None
            else er2_reserve + float(model_charges[MODEL_IDS[1]])
        ),
        "max_budget_usd": max_budget,
        "remaining_budget_after_conservative_reserve_usd": (
            None
            if er2_reserve is None or max_budget is None
            else float(max_budget) - er2_reserve - float(model_charges[MODEL_IDS[1]])
        ),
        "pricing_source": PRICING_SOURCE,
        "provider_invoice_verified": False,
        "er2_cost_note": ER2_COST_DISCLAIMER,
    }
    atomic_write_json(root / "actual_cost_estimate.json", actual_cost)
    logical_rows = _table_rows(root / "request_manifest.parquet")
    terminal_rows = [row for row in logical_rows if row.get("terminal")]
    fallback = {
        "planned_logical_request_count": len(logical_rows),
        "terminal_logical_request_count": len(terminal_rows),
        "valid_response_count": sum(bool(row.get("valid_response")) for row in terminal_rows),
        "technical_fallback_count": sum(bool(row.get("technical_fallback")) for row in terminal_rows),
        "permanent_api_failure_count": sum(bool(row.get("permanent_api_failure")) for row in terminal_rows),
        "by_lifecycle_status": dict(Counter(str(row.get("lifecycle_status")) for row in terminal_rows)),
        "cached_response_fallback_reasons": dict(response_fallbacks),
    }
    atomic_write_json(root / "fallback_summary.json", fallback)
    return {"runtime": runtime, "cost": actual_cost, "fallback": fallback, "score_rows": len(score_rows)}


def _empty_publication_artifacts(root: Path) -> None:
    if not (root / "per_sample_predictions.parquet").is_file():
        _atomic_parquet(
            root / "per_sample_predictions.parquet",
            [],
            pa.schema(
                [
                    ("sample_id", pa.string()), ("method", pa.string()),
                    ("selected_stable_candidate_id", pa.string()), ("status", pa.string()),
                ]
            ),
        )
    if not (root / "per_model_decisions.parquet").is_file():
        _atomic_parquet(
            root / "per_model_decisions.parquet",
            [],
            pa.schema(
                [
                    ("sample_id", pa.string()), ("model_id", pa.string()),
                    ("request_hash", pa.string()), ("selected_stable_candidate_id", pa.string()),
                    ("valid", pa.bool_()),
                ]
            ),
        )
    if not (root / "per_method_metrics.csv").is_file():
        _atomic_csv(root / "per_method_metrics.csv", [], ("method", "status"))
    defaults = {
        "statistical_tests.json": {"status": "not_run"},
        "bootstrap_intervals.json": {"status": "not_run", "draws": 10_000},
        "model_agreement.json": {"status": "not_run"},
        "stability_results.json": {"status": "not_run"},
        "independent_recompute_results.json": {"status": "not_run"},
    }
    for filename, payload in defaults.items():
        if not (root / filename).is_file():
            atomic_write_json(root / filename, payload)
    if not (root / "threshold_sweeps.csv").is_file():
        _atomic_csv(root / "threshold_sweeps.csv", [], ("model_id", "status"))


def _phase_complete(root: Path, phase: str) -> bool:
    marker = _read_json(root / phase / "PHASE_COMPLETE.json")
    complete = bool(
        isinstance(marker, dict)
        and marker.get("phase") == phase
        and marker.get("status") == "complete"
    )
    if not complete:
        return False
    if phase == "formal_test":
        identities = marker.get("artifact_identities")
        if not marker.get("run_id") or not isinstance(identities, list) or not identities:
            return False
        for identity in identities:
            path = Path(str(identity.get("path", "")))
            if (
                identity.get("identity_kind") != "file_sha256"
                or not path.is_file()
                or sha256_file(path) != identity.get("sha256")
                or path.stat().st_size != int(identity.get("size_bytes", -1))
            ):
                return False
    return True


def _independent_recompute_passed(root: Path, lock_payload: Mapping[str, Any] | None = None) -> bool:
    payload = _read_json(root / "independent_recompute_results.json")
    if not isinstance(payload, dict):
        return False
    comparison = payload.get("comparison_to_main", payload)
    passed = bool(
        isinstance(comparison, dict)
        and comparison.get("selected_candidate_id_match_rate") == 1.0
        and comparison.get("legacy_correctness_match_rate") == 1.0
        and comparison.get("corrected_correctness_match_rate") == 1.0
        and comparison.get("aggregate_counts_match") is True
    )
    if not passed or lock_payload is None:
        return passed
    input_hashes = payload.get("input_sha256")
    if not isinstance(input_hashes, dict):
        return False
    expected_paths = {
        "predictions": root / "formal_test" / "per_sample_predictions.parquet",
        "features": Path(str(lock_payload["baseline_candidates"]["features"]["path"])),
        "raw_predictions_with_gt": Path(
            str(lock_payload["ground_truth_inputs"]["raw_predictions_with_gt"]["path"])
        ),
        "evaluator_source": Path(str(lock_payload["evaluator"]["corrected"]["source"]["path"])),
    }
    return all(
        path.is_file() and input_hashes.get(name) == sha256_file(path)
        for name, path in expected_paths.items()
    )


def _lock_valid_for_publication(
    root: Path, *, repo_root: Path
) -> tuple[bool, str | None, dict[str, Any] | None]:
    lock_path = root / "frozen_gemini_crog_evidence_manifest.json"
    if not lock_path.is_file():
        return False, None, None
    payload = _read_json(lock_path)
    if not isinstance(payload, dict):
        return False, None, None
    expected = payload.get("lock_sha256")
    marker = _read_json(root / "formal_test" / "PHASE_COMPLETE.json", {})
    run_id = marker.get("run_id") if isinstance(marker, dict) else None
    if not expected or not run_id:
        return False, None, None
    try:
        verified = verify_experiment_lock(
            lock_path,
            repo_root=repo_root,
            expected_run_id=str(run_id),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False, str(expected), None
    claim = _read_json(root / "formal_test" / "formal_run_claim.json", {})
    if (
        not isinstance(claim, dict)
        or claim.get("run_id") != run_id
        or claim.get("lock_sha256") != expected
    ):
        return False, str(expected), None
    return True, str(verified["lock_sha256"]), payload


def publish_highest_complete_phase(
    run_root: str | Path, *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    root = Path(run_root)
    repository = Path(repo_root).resolve() if repo_root is not None else root.resolve()
    formal_complete = _phase_complete(root, "formal_test")
    lock_valid, lock_sha, lock_payload = _lock_valid_for_publication(
        root, repo_root=repository
    )
    recompute_passed = _independent_recompute_passed(root, lock_payload)
    if formal_complete and not (lock_valid and recompute_passed):
        raise FinalizationError("formal completion cannot be published without valid lock and recomputation")
    if formal_complete:
        source = root / "formal_test"
        scope = "formal_test"
        mappings = {
            "per_sample_predictions.parquet": "per_sample_predictions.parquet",
            "per_model_decisions.parquet": "per_model_decisions.parquet",
            "per_method_metrics.csv": "per_method_metrics.csv",
            "statistical_tests.json": "statistical_tests.json",
            "bootstrap_intervals.json": "bootstrap_intervals.json",
            "model_agreement.json": "model_agreement.json",
            "query_type_metrics.csv": "query_type_metrics.csv",
            "recovered_harmful_analysis.json": "recovered_harmful_analysis.json",
        }
    elif _phase_complete(root, "validation"):
        source = root / "validation"
        scope = "validation"
        mappings = {
            "per_model_decisions.parquet": "per_model_decisions.parquet",
            "per_method_metrics.csv": "per_method_metrics.csv",
            "statistical_tests.json": "statistical_tests.json",
            "bootstrap_intervals.json": "bootstrap_intervals.json",
            "model_agreement.json": "model_agreement.json",
            "query_type_metrics.csv": "query_type_metrics.csv",
            "recovered_harmful_analysis.json": "recovered_harmful_analysis.json",
        }
    else:
        source = None
        scope = "partial"
        mappings = {}
    copied = []
    for destination_name, source_name in mappings.items():
        path = source / source_name  # type: ignore[operator]
        if not path.is_file():
            raise FinalizationError(f"completed {scope} artifact is missing: {path.name}")
        _atomic_copy(path, root / destination_name)
        copied.append(destination_name)
    if (root / "calibration").is_dir():
        sweep_rows: list[dict[str, Any]] = []
        sweep_fields: list[str] = []
        for path in sorted((root / "calibration").glob("*_threshold_sweep.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    enriched = {"source": path.name, **row}
                    for field in enriched:
                        if field not in sweep_fields:
                            sweep_fields.append(field)
                    sweep_rows.append(enriched)
        if sweep_rows:
            _atomic_csv(root / "threshold_sweeps.csv", sweep_rows, sweep_fields)
    stability = root / "stability" / "per_model_summary.json"
    if stability.is_file():
        _atomic_copy(stability, root / "stability_results.json")
    return {
        "publication_scope": scope,
        "formal_complete": formal_complete,
        "formal_lock_valid": lock_valid,
        "formal_lock_sha256": lock_sha,
        "independent_recompute_passed": recompute_passed,
        "copied_artifacts": copied,
    }


def _safe_env_status(name: str) -> str:
    return "SET" if os.environ.get(name) else "UNSET"


def write_environment(run_root: str | Path) -> dict[str, Any]:
    payload = {
        "generated_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "google_genai": "2.16.0",
        "environment_variable_status": {
            "GEMINI_API_KEY": _safe_env_status("GEMINI_API_KEY"),
            "GEMINI_MAX_SPEND_USD": _safe_env_status("GEMINI_MAX_SPEND_USD"),
            "GEMINI_MAX_CONCURRENCY": _safe_env_status("GEMINI_MAX_CONCURRENCY"),
            "GEMINI_ER2_COST_CAP_PER_REQUEST_USD": _safe_env_status(
                "GEMINI_ER2_COST_CAP_PER_REQUEST_USD"
            ),
        },
        "secrets_included": False,
    }
    atomic_write_json(Path(run_root) / "environment.json", payload)
    return payload


def _phase_execution_status(root: Path) -> dict[str, str]:
    statuses = {}
    phase_status = _read_json(root / "phase_status.json", {})
    detail = phase_status.get("phases", {}) if isinstance(phase_status, dict) else {}
    for phase in PHASE_ORDER:
        if _phase_complete(root, phase):
            statuses[phase] = "complete"
        elif isinstance(detail.get(phase), dict):
            statuses[phase] = str(detail[phase].get("status", "pending"))
        else:
            statuses[phase] = "pending"
    statuses["lock"] = "complete" if (root / "frozen_gemini_crog_evidence_manifest.json").is_file() else "pending"
    return statuses


def _metrics_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "per_method_metrics.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _locked_primary(root: Path) -> str | None:
    for path in (
        root / "frozen_gemini_crog_evidence_manifest.json",
        root / "validation" / "primary_selection.json",
    ):
        payload = _read_json(path)
        if isinstance(payload, dict):
            for key in ("primary_method", "locked_primary", "selected_method"):
                if payload.get(key):
                    return str(payload[key])
    return None


def build_results_bundle(
    run_root: str | Path,
    *,
    publication: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    manifest_audit: Mapping[str, Any],
    frozen_audit: Mapping[str, Any],
    security_audit: Mapping[str, Any],
    evidence_audit: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(run_root)
    phase_status = _read_json(root / "phase_status.json", {})
    scope = str(publication["publication_scope"])
    run_status = str(phase_status.get("status", "partial")) if isinstance(phase_status, dict) else "partial"
    status = "complete" if scope == "formal_test" else run_status
    bundle = {
        "schema_version": "1.0",
        "experiment_id": root.name,
        "status": status,
        "publication_scope": scope,
        "formal_results_available": scope == "formal_test",
        "partial_results_are_not_formal_conclusions": scope != "formal_test",
        "generated_at_utc": utc_now(),
        "execution_status": _phase_execution_status(root),
        "locked_primary": _locked_primary(root),
        "formal_lock": {
            "valid": bool(publication["formal_lock_valid"]),
            "sha256": publication.get("formal_lock_sha256"),
        },
        "baseline_integrity": _json_safe(frozen_audit),
        "manifest_integrity": _json_safe(manifest_audit),
        "cache_integrity": _json_safe(cache_audit),
        "secret_scan": _json_safe(security_audit),
        "evidence_aggregation": _json_safe(evidence_audit),
        "metrics": _metrics_rows(root),
        "query_type_metrics": (
            list(csv.DictReader((root / "query_type_metrics.csv").open(newline="", encoding="utf-8")))
            if (root / "query_type_metrics.csv").is_file()
            else []
        ),
        "recovered_harmful_analysis": _read_json(
            root / "recovered_harmful_analysis.json", {"status": "not_run"}
        ),
        "independent_recompute": _read_json(root / "independent_recompute_results.json", {"status": "not_run"}),
        "runtime": derived.get("runtime"),
        "cost": derived.get("cost"),
        "fallback": derived.get("fallback"),
        "er2_cost_note": ER2_COST_DISCLAIMER,
        "interpretation_limits": [
            "The formal test split is not pristine.",
            "ER2 is a preview model and provider snapshots may change.",
            "API outputs are not fully deterministic.",
            "Predicted M/Q/angle/W evidence can be wrong.",
            "RGB evidence cannot verify physical contact.",
            "J@1 is not physical grasp success.",
        ],
    }
    atomic_write_json(root / "results_bundle.json", bundle)
    return bundle


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "No completed method metrics are available."
    preferred = [
        "method", "legacy_j1", "legacy_delta_pp", "legacy_recovered", "legacy_harmful",
        "legacy_net", "corrected_j1", "corrected_delta_pp", "corrected_recovered",
        "corrected_harmful", "corrected_net", "switch_coverage", "outcome_precision",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    if not fields:
        fields = list(rows[0])[:8]
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join((header, separator, *body))


def _generated_markdown(bundle: Mapping[str, Any]) -> str:
    execution = bundle["execution_status"]
    phase_lines = "\n".join(f"- {phase}: {status}" for phase, status in execution.items())
    baseline = (bundle.get("baseline_integrity") or {}).get("baseline", {})
    scope = bundle["publication_scope"]
    if scope == "formal_test":
        summary = "Phase C–H formal artifacts are complete and independently recomputed."
    else:
        summary = (
            f"The persisted run is currently `{bundle['status']}` with publication scope `{scope}`. "
            "No incomplete phase is presented as a formal result."
        )
    return f"""## Generated offline finalization status

Generated: {bundle['generated_at_utc']}

### Summary

{summary}

### Execution status

{phase_lines}

### Baseline integrity

- Samples: {baseline.get('sample_count', 'unavailable')}
- Candidate rows: {baseline.get('candidate_count', 'unavailable')}
- Exactly five candidates per sample: {baseline.get('five_candidates_per_sample_count', 'unavailable')}
- Legacy q-only successes: {baseline.get('legacy_q_only_success_count', 'unavailable')}
- Legacy Oracle@5 successes: {baseline.get('legacy_oracle_success_count', 'unavailable')}
- Corrected q-only successes: {baseline.get('corrected_q_only_success_count', 'unavailable')}
- Corrected Oracle@5 successes: {baseline.get('corrected_oracle_success_count', 'unavailable')}
- Candidate identity: {bundle.get('baseline_integrity', {}).get('candidate_identity_sha256', 'unavailable')}
- Old V2 tree: {bundle.get('baseline_integrity', {}).get('v2_tree_sha256', 'not recomputed')}

### Method results

{_markdown_table(bundle.get('metrics', []))}

### Independent recomputation

Formal publication gate passed: {bundle.get('formal_results_available', False) and bundle.get('formal_lock', {}).get('valid', False)}.

### Cost and runtime

{ER2_COST_DISCLAIMER}

Runtime and token totals are derived only from persisted cache records. They do not imply provider billing finality.

### Verified assumptions

- Secret scan: {bundle.get('secret_scan', {}).get('status', 'unknown')}
- Duplicate successful request hashes: {bundle.get('cache_integrity', {}).get('duplicate_successful_request_hash_count', 'unknown')}
- Candidate identity audit: {bundle.get('baseline_integrity', {}).get('status', 'unknown')}
- Manifest/request arithmetic: {bundle.get('manifest_integrity', {}).get('status', 'unknown')}
- Formal test used for selection: no; formal publication requires the frozen validation lock.

### Remaining risks

- The test split is not pristine.
- ER2 is a preview model and the provider may update or withdraw it.
- API outputs are not completely deterministic.
- CROG predicted M/Q/angle/W evidence may be wrong.
- RGB alone cannot verify real contact.
- J@1 is not physical grasp success.
- API cost, quota, and provider data-handling remain operational risks.
"""


def _upsert_generated_section(path: Path, content: str) -> None:
    start = "<!-- GEMINI-OFFLINE-FINALIZER:START -->"
    end = "<!-- GEMINI-OFFLINE-FINALIZER:END -->"
    generated = f"{start}\n{content.rstrip()}\n{end}\n"
    original = path.read_text(encoding="utf-8") if path.is_file() else f"# {path.stem}\n"
    if start in original and end in original:
        prefix = original.split(start, 1)[0]
        suffix = original.split(end, 1)[1].lstrip("\n")
        updated = prefix.rstrip() + "\n\n" + generated + ("\n" + suffix if suffix else "")
    else:
        updated = original.rstrip() + "\n\n" + generated
    _atomic_text(path, updated)


def write_docs(bundle: Mapping[str, Any], docs_dir: str | Path) -> list[str]:
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    content = _generated_markdown(bundle)
    paths = [
        docs / "CROG_GEMINI_EVIDENCE_V1_VALIDATION.md",
        docs / "CROG_GEMINI_EVIDENCE_V1_RESULTS.md",
    ]
    for path in paths:
        _upsert_generated_section(path, content)
    return [str(path) for path in paths]


def append_command_log(run_root: str | Path) -> None:
    path = Path(run_root) / "commands.log"
    line = "python -m failure_analysis.gemini_crog_evidence_v1.finalize --run-root <run-root>"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if line not in existing.splitlines():
        _atomic_text(path, existing.rstrip() + ("\n" if existing.strip() else "") + line + "\n")


def finalize_experiment(
    run_root: str | Path,
    *,
    repo_root: str | Path,
    docs_dir: str | Path | None = None,
    frozen_paths: FrozenInputPaths | None = None,
    expected_candidate_sha256: str | None = None,
    expected_v2_sha256: str = PREEXISTING_V2_TREE_SHA256,
    verify_v2_tree: bool = True,
    secret_values: Sequence[str] = (),
    enforce_canonical_counts: bool = True,
) -> dict[str, Any]:
    """Finalize persisted artifacts atomically after all read-only audits pass."""

    root = Path(run_root).resolve()
    repo = Path(repo_root).resolve()
    if not root.is_dir():
        raise FinalizationError(f"run root is missing: {root}")
    cache_audit = audit_cache(root / "gemini_cache.sqlite")
    manifest_audit = audit_manifest_request_counts(
        root, enforce_canonical_counts=enforce_canonical_counts
    )
    preflight = _read_json(root / "full_run_preflight.json", {})
    candidate_sha = expected_candidate_sha256 or preflight.get("candidate_identity_stream_sha256")
    if not candidate_sha:
        raise FinalizationError("expected frozen candidate identity is unavailable")
    paths = frozen_paths or frozen_input_paths_from_passport(root, repo)
    frozen_audit = audit_frozen_inputs(
        paths=paths,
        repo_root=repo,
        expected_candidate_sha256=str(candidate_sha),
        expected_v2_sha256=expected_v2_sha256,
        verify_v2_tree=verify_v2_tree,
    )
    current_secret = os.environ.get("GEMINI_API_KEY")
    exact_secrets = tuple(value for value in (*secret_values, current_secret) if value)
    security_audit = secret_scan(
        (root, repo / "failure_analysis/gemini_crog_evidence_v1", repo / "configs", repo / "prompts", repo / "docs"),
        secret_values=exact_secrets,
    )
    evidence_audit = aggregate_evidence(root)
    logical_request_audit = build_logical_request_manifest(root)
    publication = publish_highest_complete_phase(root, repo_root=repo)
    _empty_publication_artifacts(root)
    derived = cache_derived_artifacts(root)
    write_environment(root)
    append_command_log(root)
    bundle = build_results_bundle(
        root,
        publication=publication,
        cache_audit=cache_audit,
        manifest_audit=manifest_audit,
        frozen_audit=frozen_audit,
        security_audit=security_audit,
        evidence_audit={**evidence_audit, "logical_requests": logical_request_audit},
        derived=derived,
    )
    if docs_dir is not None:
        bundle["docs_updated"] = write_docs(bundle, docs_dir)
        atomic_write_json(root / "results_bundle.json", bundle)
    missing = [name for name in REQUIRED_ROOT_ARTIFACTS if not (root / name).is_file()]
    if publication["formal_complete"] and not (root / "frozen_gemini_crog_evidence_manifest.json").is_file():
        missing.append("frozen_gemini_crog_evidence_manifest.json")
    if missing:
        raise FinalizationError("required publication artifacts are missing: " + ", ".join(missing))
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--skip-v2-tree", action="store_true", help="Development-only: skip the large V2 tree digest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = finalize_experiment(
        args.run_root,
        repo_root=args.repo_root,
        docs_dir=args.docs_dir,
        verify_v2_tree=not args.skip_v2_tree,
    )
    print(json.dumps({"status": bundle["status"], "publication_scope": bundle["publication_scope"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
