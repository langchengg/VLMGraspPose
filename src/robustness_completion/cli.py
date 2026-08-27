"""Unified, resumable entry point for the remaining robustness child run.

The CLI deliberately separates model execution from evidence aggregation.
Runtime parity validation reads fixed worker artifacts only, and runtime
aggregation is imported lazily so importing this module can never launch a
worker or depend on an aggregation module that is still being assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import pandas as pd

from .common import (
    PREREGISTRATION_SHA256,
    RUN_ID,
    atomic_json,
    canonical_sha256,
    require_run_dir,
    sha256_file,
    verify_preregistration,
)


FOUR_D_ROUTES = ("CROG", "G1", "C1", "D1")
SIX_D_NATIVE_ROUTES = ("oracle", "adapted")
PARITY_BOOLEAN_COLUMNS = (
    "passed",
    "probability_equal",
    "binary_mask_equal",
    "candidate_ids_equal",
    "top1_equal",
    "feature_schema_equal",
    "features_equal",
)

DUPLICATE_ARTIFACTS = (
    "duplicate_audit/DUPLICATE_MAP_LOCK.json",
    "duplicate_exclusion/completion.json",
    "duplicate_reporting_completion.json",
)
RUNTIME_ARTIFACTS = (
    "runtime_full/profile_subset_manifest.json",
    "runtime_full/adapter_preflight.json",
    "runtime_full/parity_evidence_summary.json",
    "runtime_full/route_parity.csv",
    "runtime_full/cold_start_raw.csv",
    "runtime_full/warm_disk_raw.parquet",
    "runtime_full/warm_preloaded_raw.parquet",
    "runtime_full/stage_timings.parquet",
    "runtime_full/memory_summary.csv",
    "runtime_full/runtime_summary.csv",
    "runtime_full/runtime_completion.json",
    "runtime_full/RUNTIME_METHODS.md",
    "runtime_full/RUNTIME_LIMITATIONS.md",
)
REPORT_ARTIFACTS = (
    "completion_results_manifest.json",
    "completion_figures_manifest.json",
    "COMBINED_SUMMARY.md",
    "FINAL_STATUS.md",
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_dir(repo: Path, run_id: str) -> Path:
    return repo / "artifacts" / "robustness_completion" / run_id


def _prepare(repo: Path, run_dir: Path) -> Path:
    locked = require_run_dir(repo, run_dir)
    verify_preregistration(locked)
    return locked


def _run_duplicate_map(
    repo: Path, run_dir: Path, *, resume: bool
) -> dict[str, Any]:
    from .duplicate_map import build_duplicate_map

    return build_duplicate_map(repo, run_dir, resume=resume)


def _run_duplicate_visual(
    repo: Path, run_dir: Path, *, resume: bool
) -> dict[str, Any]:
    from .duplicate_visual import audit_duplicate_map

    return audit_duplicate_map(repo, run_dir, resume=resume)


def _run_duplicate_evaluation(
    repo: Path, run_dir: Path, *, resume: bool
) -> dict[str, Any]:
    from .duplicate_evaluation import evaluate_duplicate_exclusion

    return evaluate_duplicate_exclusion(repo, run_dir, resume=resume)


def _run_duplicate_report(
    repo: Path, run_dir: Path, *, resume: bool
) -> dict[str, Any]:
    from .duplicate_reporting import report_duplicate_results

    return report_duplicate_results(repo, run_dir, resume=resume)


def _compact_preflight(result: Mapping[str, Any]) -> dict[str, Any]:
    records = result.get("records", result.get("source_records", []))
    complete_deployment = result.get("complete_deployment")
    cache_read = result.get("candidate_feature_score_caches_read")
    return {
        "status": result.get("status", "UNKNOWN"),
        "device": result.get("device"),
        "complete_deployment": (
            bool(complete_deployment) if complete_deployment is not None else None
        ),
        "retrospective": bool(result.get("retrospective", False)),
        "cache_disabled": cache_read is False if cache_read is not None else None,
        "blockers": list(result.get("blockers", [])),
        "asset_hashes": {
            str(record["label"]): record.get("sha256")
            for record in records
            if isinstance(record, Mapping) and "label" in record
        },
    }


def _build_runtime_adapters(repo: Path, run_dir: Path) -> dict[str, Any]:
    """Lock subsets and summarize route contracts without executing inference."""

    from .runtime_4d import asset_preflight
    from .runtime_6d import (
        MissingFormalCheckpointError,
        assert_formal_reranker_available,
        select_profile_groups,
        validate_source_contract,
    )
    from .runtime_common import (
        FOUR_D_REQUIRED_STAGES,
        SIX_D_REQUIRED_STAGES,
        build_combined_subset_manifest,
    )

    run_dir = _prepare(repo, run_dir)
    subset = build_combined_subset_manifest(repo, run_dir)
    routes_4d: dict[str, Any] = {}
    for route in FOUR_D_ROUTES:
        try:
            preflight = _compact_preflight(
                asset_preflight(
                    repo,
                    run_dir,
                    route,
                    method="gated",
                    verify_hashes=True,
                    probe_docker=False,
                )
            )
        except Exception as error:  # fail closed, while preserving other routes
            preflight = {
                "status": "PREFLIGHT_FAILED",
                "complete_deployment": False,
                "retrospective": route == "D1",
                "cache_disabled": None,
                "blockers": [f"{type(error).__name__}: {error}"],
                "asset_hashes": {},
            }
        routes_4d[route] = {
            "methods": ["native", "raw", "gated"],
            "required_stages": list(FOUR_D_REQUIRED_STAGES[route]),
            "preflight": preflight,
        }

    routes_6d: dict[str, Any] = {}
    try:
        contract = validate_source_contract(repo)
        selection = select_profile_groups(contract)
        source = {
            "status": "PASS",
            "profile_groups": len(selection.groups),
            "parity_groups": len(selection.parity_groups),
            "warmup_groups": len(selection.warmup_groups),
            "subset_ordering_sha256": selection.ordering_sha256,
            "source_hashes": dict(contract.hashes),
        }
        try:
            assert_formal_reranker_available(contract)
        except MissingFormalCheckpointError as error:
            raw = {
                "status": "BLOCKED",
                "complete_deployment": False,
                "blocker_code": error.code,
                "blocker": str(error),
            }
        else:
            raw = {"status": "PASS", "complete_deployment": True}
        for route in SIX_D_NATIVE_ROUTES:
            routes_6d[route] = {
                "device": "cpu",
                "cache_disabled": True,
                "required_stages": list(SIX_D_REQUIRED_STAGES[route]),
                "source_preflight": source,
                "methods": {
                    "native": {"status": "PASS", "complete_deployment": True},
                    "raw": raw,
                },
            }
    except Exception as error:  # source-contract failures block both routes
        for route in SIX_D_NATIVE_ROUTES:
            routes_6d[route] = {
                "device": "cpu",
                "cache_disabled": True,
                "required_stages": list(SIX_D_REQUIRED_STAGES[route]),
                "source_preflight": {
                    "status": "PREFLIGHT_FAILED",
                    "blocker": f"{type(error).__name__}: {error}",
                },
                "methods": {
                    "native": {"status": "BLOCKED", "complete_deployment": False},
                    "raw": {"status": "BLOCKED", "complete_deployment": False},
                },
            }

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "subset_manifest_sha256": sha256_file(
            run_dir / "runtime_full/profile_subset_manifest.json"
        ),
        "subset_counts": {
            "4d_profile": int(subset["profile_4d"]["count"]),
            "4d_parity": int(subset["parity_4d"]["count"]),
            "6d_profile": int(subset["profile_6d"]["count"]),
            "6d_parity": int(subset["parity_6d"]["count"]),
        },
        "route_contracts": {"4d": routes_4d, "6d": routes_6d},
        "executes_inference": False,
    }
    if (
        any(route["preflight"]["status"] != "PASS" for route in routes_4d.values())
        or any(
            route["source_preflight"]["status"] != "PASS"
            for route in routes_6d.values()
        )
        or any(
            route["methods"]["raw"]["status"] != "PASS"
            for route in routes_6d.values()
        )
    ):
        payload["status"] = "PARTIAL_RUNTIME_PREFLIGHT"
    atomic_json(run_dir / "runtime_full/adapter_preflight.json", payload)
    return payload


def _parity_record_base(path: Path, route: str, method: str) -> dict[str, Any]:
    return {
        "route": route,
        "method": method,
        "retrospective": route == "D1",
        "required": True,
        "classification": "MISSING",
        "status": "MISSING_PARITY_EVIDENCE",
        "passed": False,
        "expected_count": 20,
        "observed_count": 0,
        "unique_count": 0,
        "ordered_ids_exact": False,
        "preregistration_match": False,
        "route_match": False,
        "method_match": False,
        "mode_match": False,
        "device_match": False,
        "cache_contract_passed": False,
        "checks": {},
        "failure_count": None,
        "failures": [],
        "blocker_code": "MISSING_PARITY_EVIDENCE",
        "blocker": "required fixed parity evidence is absent",
        "source_path": str(path),
        "source_sha256": None,
        "detail_path": None,
        "detail_sha256": None,
    }


def _four_d_parity_record(
    path: Path, route: str, expected_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    record = _parity_record_base(path, route, "gated")
    if not path.is_file():
        return record
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("parity_rows", [])
    rows = rows if isinstance(rows, list) else []
    observed_ids = [str(row.get("sample_id")) for row in rows]
    declared_ids = [str(value) for value in payload.get("parity_sample_ids", [])]
    expected = list(expected_ids) if expected_ids is not None else declared_ids
    count = int(payload.get("parity_count") or len(rows))
    checks = {
        name: payload.get(name)
        for name in ("candidate_parity", "feature_parity", "ranker_parity", "gate_parity")
    }
    row_contract = bool(
        len(rows) == 20
        and all(
            row.get("route") == route
            and row.get("method") == "gated"
            and row.get("mode") == "parity"
            and row.get("device") == "mps"
            and row.get("status") == "PASS"
            and all(
                row.get(name) is True
                for name in (
                    "candidate_parity",
                    "feature_parity",
                    "ranker_parity",
                    "gate_parity",
                )
            )
            for row in rows
        )
    )
    ordered_ids_exact = bool(
        count == 20
        and len(observed_ids) == 20
        and len(set(observed_ids)) == 20
        and observed_ids == declared_ids == expected
    )
    failure_count = int(payload.get("failure_count") or 0)
    passed = (
        payload.get("status") == "PASS"
        and payload.get("complete_deployment") is True
        and count == 20
        and ordered_ids_exact
        and payload.get("route") == route
        and payload.get("method") == "gated"
        and payload.get("preregistration_sha256") == PREREGISTRATION_SHA256
        and payload.get("cache_use") == "parity_comparison_only"
        and failure_count == 0
        and not payload.get("failures", [])
        and row_contract
        and all(value is True for value in checks.values())
    )
    blocked = str(payload.get("status", "")).startswith("NOT_EXECUTED")
    blockers = payload.get("blockers", [])
    blocker_list = blockers if isinstance(blockers, list) else []
    record.update(
        {
            "classification": "PASS" if passed else ("BLOCKED" if blocked else "FAILED"),
            "status": str(payload.get("status", "UNKNOWN")),
            "passed": passed,
            "observed_count": count,
            "unique_count": len(set(observed_ids)),
            "ordered_ids_exact": ordered_ids_exact,
            "preregistration_match": (
                payload.get("preregistration_sha256") == PREREGISTRATION_SHA256
            ),
            "route_match": payload.get("route") == route,
            "method_match": payload.get("method") == "gated",
            "mode_match": all(row.get("mode") == "parity" for row in rows),
            "device_match": all(row.get("device") == "mps" for row in rows),
            "cache_contract_passed": (
                payload.get("cache_use") == "parity_comparison_only"
            ),
            "checks": {**checks, "row_contract": row_contract},
            "failure_count": failure_count,
            "failures": list(payload.get("failures", [])),
            "blocker_code": (
                None if passed else ("NOT_EXECUTED_IMPLEMENTATION_INCOMPLETE" if blocked else "FAILED_PARITY")
            ),
            "blocker": "; ".join(str(value) for value in blocker_list) or None,
            "source_sha256": sha256_file(path),
        }
    )
    return record


def _six_d_parity_record(
    path: Path,
    route: str,
    method: str,
    expected_ids: Sequence[str] | None = None,
    expected_ordering_sha256: str | None = None,
    known_blocker: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_route = f"6D_{route.upper()}"
    record = _parity_record_base(path, canonical_route, method)
    if not path.is_file():
        if known_blocker and known_blocker.get("blocker_code") == "MISSING_FORMAL_CHECKPOINT":
            record.update(
                {
                    "classification": "BLOCKED",
                    "status": "NOT_MEASURABLE_MISSING_FORMAL_LAMBDAMART_CHECKPOINT",
                    "blocker_code": "MISSING_FORMAL_CHECKPOINT",
                    "blocker": known_blocker.get("blocker"),
                    "blocker_source_path": known_blocker.get("source_path"),
                    "blocker_source_sha256": known_blocker.get("source_sha256"),
                }
            )
        return record
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("blocker_code") == "MISSING_FORMAL_CHECKPOINT":
        record.update(
            {
                "classification": "BLOCKED",
                "status": str(payload.get("status")),
                "preregistration_match": (
                    payload.get("preregistration_sha256") == PREREGISTRATION_SHA256
                ),
                "route_match": payload.get("route") == route,
                "method_match": payload.get("method") == method,
                "mode_match": payload.get("mode") == "parity",
                "device_match": payload.get("device") == "cpu",
                "cache_contract_passed": all(
                    payload.get(name) is False
                    for name in (
                        "candidate_cache_used_for_timing",
                        "feature_cache_used_for_timing",
                        "score_cache_used_for_timing",
                    )
                )
                and payload.get("cache_disabled") is True,
                "blocker_code": "MISSING_FORMAL_CHECKPOINT",
                "blocker": payload.get("blocker"),
                "source_sha256": sha256_file(path),
            }
        )
        return record
    detail_path = path.with_name("parity_results.parquet")
    detail_passed = False
    detail_count = 0
    unique_count = 0
    ordered_ids_exact = False
    row_contract = False
    geometry_contract = False
    detail_sha256: str | None = None
    if detail_path.is_file():
        detail = pd.read_parquet(detail_path)
        missing = set(PARITY_BOOLEAN_COLUMNS) - set(detail.columns)
        if not missing:
            identifiers = detail["group_id"].astype(str).tolist()
            detail_count = len(detail)
            unique_count = len(set(identifiers))
            expected = list(expected_ids) if expected_ids is not None else identifiers
            ordered_ids_exact = bool(
                detail_count == 20 and unique_count == 20 and identifiers == expected
            )
            bool_columns_strict = all(
                pd.api.types.is_bool_dtype(detail[name])
                for name in PARITY_BOOLEAN_COLUMNS
            )
            bool_checks_pass = bool_columns_strict and bool(
                detail.loc[:, list(PARITY_BOOLEAN_COLUMNS)].all(axis=None)
            )
            row_contract = bool(
                (detail["sample_id"].astype(str) == detail["group_id"].astype(str)).all()
                and detail["route"].eq(route).all()
                and detail["method"].eq(method).all()
                and detail["mode"].eq("parity").all()
                and detail["device"].eq("cpu").all()
                and detail["status"].eq("passed").all()
                and detail["parity_index"].tolist() == list(range(20))
                and detail["candidate_count_actual"].equals(
                    detail["candidate_count_frozen"]
                )
            )
            geometry_contract = bool(
                detail["max_translation_error_m"].fillna(float("inf")).le(1e-5).all()
                and detail["max_rotation_error_rad"].fillna(float("inf")).le(1e-4).all()
                and detail["max_width_error_m"].fillna(float("inf")).le(1e-5).all()
            )
            detail_passed = bool(
                ordered_ids_exact
                and bool_checks_pass
                and row_contract
                and geometry_contract
            )
        detail_sha256 = sha256_file(detail_path)
    count = int(payload.get("parity_groups_checked") or 0)
    cache_contract = bool(
        payload.get("cache_disabled") is True
        and all(
            payload.get(name) is False
            for name in (
                "candidate_cache_used_for_timing",
                "feature_cache_used_for_timing",
                "score_cache_used_for_timing",
            )
        )
    )
    subset_match = (
        expected_ordering_sha256 is None
        or payload.get("subset_ordering_sha256") == expected_ordering_sha256
    )
    passed = (
        payload.get("status") == "PARITY_COMPLETE"
        and payload.get("parity_passed") is True
        and payload.get("parity_complete") is True
        and count == 20
        and detail_count == 20
        and detail_passed
        and payload.get("preregistration_sha256") == PREREGISTRATION_SHA256
        and payload.get("route") == route
        and payload.get("method") == method
        and payload.get("mode") == "parity"
        and payload.get("device") == "cpu"
        and subset_match
        and cache_contract
    )
    record.update(
        {
            "classification": "PASS" if passed else "FAILED",
            "status": str(payload.get("status", "UNKNOWN")),
            "passed": passed,
            "observed_count": count,
            "unique_count": unique_count,
            "ordered_ids_exact": ordered_ids_exact,
            "preregistration_match": (
                payload.get("preregistration_sha256") == PREREGISTRATION_SHA256
            ),
            "subset_ordering_match": subset_match,
            "route_match": payload.get("route") == route,
            "method_match": payload.get("method") == method,
            "mode_match": payload.get("mode") == "parity",
            "device_match": payload.get("device") == "cpu",
            "cache_contract_passed": cache_contract,
            "checks": {
                "detail_booleans": detail_passed,
                "row_contract": row_contract,
                "geometry_tolerances": geometry_contract,
            },
            "failure_count": 0 if passed else max(20 - unique_count, 1),
            "blocker_code": None if passed else "FAILED_PARITY",
            "blocker": payload.get("blocker"),
            "source_sha256": sha256_file(path),
            "detail_path": str(detail_path),
            "detail_sha256": detail_sha256,
        }
    )
    return record


def _validate_runtime_parity(repo: Path, run_dir: Path) -> dict[str, Any]:
    """Summarize fixed parity artifacts; never import or execute workers."""

    run_dir = _prepare(repo, run_dir)
    workers = run_dir / "runtime_full/workers"
    subset_path = run_dir / "runtime_full/profile_subset_manifest.json"
    if not subset_path.is_file():
        raise RuntimeError("locked runtime profile subset is missing")
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    expected_4d = [str(value) for value in subset["parity_4d"]["sample_ids"]]
    expected_6d = [str(value) for value in subset["parity_6d"]["group_ids"]]
    profile_6d = [str(value) for value in subset["profile_6d"]["group_ids"]]
    ordered_profile_6d = sorted(
        profile_6d,
        key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value),
    )
    ordering_sha256 = canonical_sha256(ordered_profile_6d)
    records = [
        _four_d_parity_record(
            workers / "4d" / route.lower() / "parity.json", route, expected_4d
        )
        for route in FOUR_D_ROUTES
    ]
    records.extend(
        _six_d_parity_record(
            workers / "6d" / route / "native/parity_result.json",
            route,
            "native",
            expected_6d,
            ordering_sha256,
        )
        for route in SIX_D_NATIVE_ROUTES
    )
    oracle_raw = _six_d_parity_record(
        workers / "6d/oracle/raw/parity_result.json",
        "oracle",
        "raw",
        expected_6d,
        ordering_sha256,
    )
    records.append(oracle_raw)
    records.append(
        _six_d_parity_record(
            workers / "6d/adapted/raw/parity_result.json",
            "adapted",
            "raw",
            expected_6d,
            ordering_sha256,
            known_blocker=oracle_raw,
        )
    )
    classifications = [str(record["classification"]) for record in records]
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": (
            "PASS" if all(record["passed"] for record in records) else "PARTIAL_RUNTIME_PARITY"
        ),
        "parity_gate_passed": all(record["passed"] for record in records),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "subset_manifest_path": str(subset_path),
        "subset_manifest_sha256": sha256_file(subset_path),
        "evidence_only": True,
        "workers_executed": False,
        "records": records,
        "passed_records": sum(record["passed"] for record in records),
        "failed_records": classifications.count("FAILED"),
        "blocked_records": classifications.count("BLOCKED"),
        "missing_records": classifications.count("MISSING"),
        "total_records": len(records),
    }
    atomic_json(run_dir / "runtime_full/parity_evidence_summary.json", payload)
    return payload


def _runtime_aggregation_module() -> ModuleType:
    """Import only when aggregation was explicitly requested."""

    try:
        return importlib.import_module("robustness_completion.runtime_aggregation")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("runtime aggregation adapter is unavailable") from error


def _aggregate_runtime(
    repo: Path, run_dir: Path, *, resume: bool
) -> dict[str, Any]:
    module = _runtime_aggregation_module()
    aggregate = getattr(module, "aggregate_runtime", None)
    if aggregate is None:
        raise RuntimeError("runtime_aggregation.aggregate_runtime is unavailable")
    return aggregate(repo, run_dir, resume)


def _generate_report(repo: Path, run_dir: Path) -> dict[str, Any]:
    from .reporting import generate_completion_report

    return generate_completion_report(repo, run_dir)


def _run_all(repo: Path, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the locked resumable orchestration order; PARTIAL is a valid result."""

    stages = {
        "duplicate_map": _run_duplicate_map(repo, run_dir, resume=True),
        "duplicate_visual_audit": _run_duplicate_visual(repo, run_dir, resume=True),
        "duplicate_evaluation": _run_duplicate_evaluation(repo, run_dir, resume=True),
        "duplicate_reporting": _run_duplicate_report(repo, run_dir, resume=True),
        "runtime_subset_preflight": _build_runtime_adapters(repo, run_dir),
        "runtime_aggregation": _aggregate_runtime(repo, run_dir, resume=True),
    }
    report = _generate_report(repo, run_dir)
    stages["report"] = report
    result = {
        "status": report.get("status", "PARTIAL_ROBUSTNESS_COMPLETION"),
        "formal_remaining_results_emitted": bool(
            report.get("formal_remaining_results_emitted", False)
        ),
        "stages": stages,
    }
    return result, stages


def _existing_hashes(run_dir: Path, names: Sequence[str]) -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name in names
        if (path := run_dir / name).is_file()
    }


def _stage_status(value: Mapping[str, Any]) -> str:
    return str(value.get("status", "UNKNOWN"))


def _record_final_manifest(
    run_dir: Path,
    result: Mapping[str, Any],
    command_argv: Sequence[str],
    stage_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Atomically merge final status, evidence hashes, and this invocation."""

    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"child run manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = str(result.get("status", "PARTIAL_ROBUSTNESS_COMPLETION"))
    manifest["formal_remaining_results_emitted"] = bool(
        result.get("formal_remaining_results_emitted", False)
    )
    stages = dict(manifest.get("stages", {}))
    if stage_results:
        stages.update(
            {
                name: {"status": _stage_status(value)}
                for name, value in stage_results.items()
            }
        )
    stages.update(
        {
            "duplicate": {
                "status": (
                    "COMPLETE" if result.get("duplicate_complete") else "PARTIAL"
                ),
                "artifact_sha256": _existing_hashes(run_dir, DUPLICATE_ARTIFACTS),
            },
            "runtime": {
                "status": (
                    "COMPLETE" if result.get("runtime_complete") else "PARTIAL"
                ),
                "artifact_sha256": _existing_hashes(run_dir, RUNTIME_ARTIFACTS),
            },
            "report": {
                "status": str(result.get("status", "PARTIAL_ROBUSTNESS_COMPLETION")),
                "artifact_sha256": _existing_hashes(run_dir, REPORT_ARTIFACTS),
            },
        }
    )
    manifest["stages"] = stages
    commands = list(manifest.get("commands", []))
    command_record = {"argv": [str(part) for part in command_argv]}
    if command_record not in commands:
        commands.append(command_record)
    manifest["commands"] = commands
    atomic_json(path, manifest)


def _record_command(run_dir: Path, command_argv: Sequence[str]) -> None:
    """Append a successfully executed CLI invocation without changing run status."""

    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"child run manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    commands = list(manifest.get("commands", []))
    command_record = {"argv": [str(part) for part in command_argv]}
    if command_record not in commands:
        commands.append(command_record)
    manifest["commands"] = commands
    atomic_json(path, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robustness_completion")
    parser.add_argument(
        "command",
        choices=[
            "audit",
            "preregister",
            "build-duplicate-map",
            "audit-duplicate-map",
            "evaluate-duplicate-exclusion",
            "report-duplicate-results",
            "build-runtime-adapters",
            "validate-runtime-parity",
            "profile-runtime-full",
            "report",
            "all",
        ],
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    process_argv = list(sys.argv if argv is None else ["robustness_completion", *argv])
    args = _parser().parse_args(effective_argv)
    repo = _repo()
    run_dir = _run_dir(repo, args.run_id)
    stage_results: dict[str, Mapping[str, Any]] | None = None
    if args.command in {"audit", "preregister"}:
        _prepare(repo, run_dir)
        result: dict[str, Any] = {
            "status": "PASS",
            "run_id": args.run_id,
            "preregistration_verified": True,
        }
    elif args.command == "build-duplicate-map":
        result = _run_duplicate_map(repo, run_dir, resume=args.resume)
    elif args.command == "audit-duplicate-map":
        result = _run_duplicate_visual(repo, run_dir, resume=args.resume)
    elif args.command == "evaluate-duplicate-exclusion":
        result = _run_duplicate_evaluation(repo, run_dir, resume=args.resume)
    elif args.command == "report-duplicate-results":
        result = _run_duplicate_report(repo, run_dir, resume=args.resume)
    elif args.command == "build-runtime-adapters":
        result = _build_runtime_adapters(repo, run_dir)
    elif args.command == "validate-runtime-parity":
        result = _validate_runtime_parity(repo, run_dir)
    elif args.command == "profile-runtime-full":
        result = _aggregate_runtime(repo, run_dir, resume=args.resume)
    elif args.command == "report":
        result = _generate_report(repo, run_dir)
        _record_final_manifest(run_dir, result, process_argv)
    else:
        result, stage_results = _run_all(repo, run_dir)
        report = stage_results["report"]
        _record_final_manifest(run_dir, report, process_argv, stage_results)
    if args.command not in {"report", "all"}:
        _record_command(run_dir, process_argv)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
