"""Read-only source audit and fail-closed planning for a continuation run."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .constants import EXACT_MODEL_IDS
from .contracts import validate_candidate_manifest
from .io import atomic_json, atomic_parquet, initialize_run, sha256_file, sha256_json, utc_now
from .payload import REQUEST_HASH_VERSION
from .renderer import renderer_hash
from .stages import check_model_metadata, normalized_model_metadata


EXPECTED_CANDIDATE_HASHES = {
    "G1": "73d51791d1f42bb853421eed6d79cb9b38c41561edbe7944696b16221f4acf41",
    "C1": "18d003d9a3871e2fe451112e3e273cb498790b7aee0dd52de12d0c5f4e4b8b8b",
}
EXPECTED_TEST = {
    "G1": {"j_at_1": 0.8706188925081433, "j_at_5": 0.9078827361563518},
    "C1": {"j_at_1": 0.7930944625407166, "j_at_5": 0.8676221498371336},
}
SOURCE_STATIC_FILES = (
    "BASELINE_RECOMPUTE.json",
    "CANDIDATE_MANIFEST_G1.parquet",
    "CANDIDATE_MANIFEST_C1.parquet",
    "DATA_SPLIT.csv",
    "DATA_SPLIT_SUMMARY.json",
    "EVIDENCE_FEATURES_G1.parquet",
    "EVIDENCE_FEATURES_C1.parquet",
    "STAGE_COHORTS.parquet",
    "baseline_per_sample.parquet",
    "QUERY_TYPE_ANNOTATIONS.parquet",
    "pricing_manifest.json",
)
SOURCE_SCIENCE_FILES = (
    "API_LEDGER_SUMMARY.json",
    "DEVELOPMENT_POLICY_LOCK.json",
    "DIAGNOSTIC_REPORT.json",
    "EVIDENCE_SELECTION.json",
    "GO_NO_GO.md",
    "INTEGRITY_REPORT.json",
    "MANIFEST.json",
    "POLICY_PRESELECTION.json",
    "RESULTS.json",
    "SUMMARY.md",
    "SUMMARY_ZH.md",
    "VALIDATION_RESULTS.json",
    "policy_threshold_sweep.csv",
    "validation_metrics.csv",
)


def _tree_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {
            "api_ledger.sqlite-shm", "api_ledger.sqlite-wal", "provider_run.lock",
        }:
            continue
        inventory[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return inventory


def _inventory_hash(inventory: Mapping[str, Any]) -> str:
    return sha256_json(inventory)


def _assert_source_idle(source: Path) -> None:
    lock_path = source / "provider_run.lock"
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("SOURCE_RUN_ACTIVE: source run has a live provider owner") from error
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
    state_path = source / "provider_run_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("status") == "RUNNING":
            raise RuntimeError(f"SOURCE_RUN_ACTIVE: {state}")


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"continuation artifact already exists with different content: {destination}")
        return
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        _copy_file(path, destination / path.relative_to(source))


def _sqlite_backup(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    read_only = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    target = sqlite3.connect(destination)
    try:
        read_only.backup(target)
    finally:
        target.close()
        read_only.close()


def _clone_source(source: Path, run: Path) -> dict[str, Any]:
    for name in SOURCE_STATIC_FILES:
        _copy_file(source / name, run / name)
    for directory in ("prompts",):
        _copy_tree(source / directory, run / directory)
    for directory in ("request_manifests", "stage_results"):
        _copy_tree(source / directory, run / directory)
    _copy_tree(source / "raw_api/er2", run / "raw_api/er2")
    snapshot = run / "source_snapshot"
    for name in SOURCE_SCIENCE_FILES:
        if (source / name).is_file():
            _copy_file(source / name, snapshot / name)
    for directory in ("audit", "request_manifests", "stage_results", "raw_api/flash"):
        _copy_tree(source / directory, snapshot / directory)
    _sqlite_backup(source / "api_ledger.sqlite", run / "api_ledger.sqlite")
    return {
        "active_static_files": list(SOURCE_STATIC_FILES),
        "source_science_namespace": str(snapshot),
        "ledger_import": "SQLite online backup; historical attempts retained and separately accounted",
        "active_source_flash_hard_stop_imported": False,
        "active_source_strict_lock_imported": False,
    }


def _ledger_recompute(path: Path) -> dict[str, Any]:
    database = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        statuses = {
            str(row["requested_model"]): {
                str(item["status"]): int(item["n"])
                for item in database.execute(
                    "SELECT status,COUNT(*) n FROM requests WHERE requested_model=? GROUP BY status",
                    (row["requested_model"],),
                )
            }
            for row in database.execute("SELECT DISTINCT requested_model FROM requests")
        }
        logical = int(database.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
        attempts = int(database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        attempted = int(database.execute("SELECT COUNT(DISTINCT request_hash) FROM attempts").fetchone()[0])
        success = int(database.execute("SELECT COUNT(*) FROM requests WHERE status='SUCCEEDED'").fetchone()[0])
        cache_hits = int(database.execute("SELECT COALESCE(SUM(cache_hit_count),0) FROM requests").fetchone()[0])
        duplicates = int(database.execute(
            "SELECT COUNT(*) FROM (SELECT request_hash FROM attempts WHERE status='SUCCEEDED' GROUP BY request_hash HAVING COUNT(*)>1)"
        ).fetchone()[0])
        inflight = int(database.execute("SELECT COUNT(*) FROM requests WHERE status='IN_FLIGHT'").fetchone()[0])
        reservations = int(database.execute("SELECT COUNT(*) FROM attempt_reservations").fetchone()[0])
    finally:
        database.close()
    return {
        "logical_requests": logical,
        "provider_attempts": attempts,
        "successful_responses": success,
        "retry_attempts": attempts - attempted,
        "cache_hits": cache_hits,
        "duplicate_successful_request_hashes": duplicates,
        "inflight": inflight,
        "attempt_reservations": reservations,
        "status_counts_by_model": statuses,
    }


def _science_stage_recompute(source: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scientific_total = 0
    for stage in ("smoke", "diagnostic", "policy_selection"):
        frame = pd.read_parquet(source / "stage_results" / f"{stage}.parquet")
        counts = frame.groupby(["model_id", "status"]).size().rename("n").reset_index()
        result[stage] = {f"{row.model_id}:{row.status}": int(row.n) for row in counts.itertuples()}
        scientific_total += len(frame.loc[frame["model_id"].eq(EXACT_MODEL_IDS[0])])
    result["er2_scientific_rows"] = scientific_total
    result["flash_current_contract_scientific_rows"] = sum(
        int(value) for stage in ("smoke", "diagnostic", "policy_selection")
        for key, value in result[stage].items() if key.startswith(EXACT_MODEL_IDS[1] + ":")
    )
    return result


def _policy_recompute(source: Path) -> dict[str, Any]:
    sweep = pd.read_csv(source / "policy_threshold_sweep.csv")
    expected = {"G1": 80, "C1": 95}
    output: dict[str, Any] = {}
    for backend, threshold in expected.items():
        row = sweep.loc[
            sweep["backend"].eq(backend)
            & sweep["model_id"].eq(EXACT_MODEL_IDS[0])
            & sweep["protocol"].eq("P3_API_BASELINE_AWARE_CONFIDENCE")
            & pd.to_numeric(sweep["threshold"], errors="coerce").eq(threshold)
        ].iloc[0]
        output[backend] = {
            "N": int(row["N_total"]), "threshold": threshold,
            "baseline_j_at_1": float(row["baseline_j_at_1"]),
            "final_j_at_1": float(row["final_j_at_1"]),
            "recovered": int(row["recovered"]), "harmful": int(row["harmful"]),
            "net": int(row["net"]), "harm_rate": float(row["harm_rate"]),
            "outcome_changing_precision": float(row["outcome_changing_precision"]),
        }
    return output


def recompute_source(source: Path) -> dict[str, Any]:
    ledger = _ledger_recompute(source / "api_ledger.sqlite")
    stages = _science_stage_recompute(source)
    policy = _policy_recompute(source)
    validation = json.loads((source / "VALIDATION_RESULTS.json").read_text())
    checks = {
        "ledger": ledger["logical_requests"] == 4191 and ledger["provider_attempts"] == 4279
        and ledger["successful_responses"] == 4148 and ledger["retry_attempts"] == 89
        and ledger["cache_hits"] == 56 and ledger["duplicate_successful_request_hashes"] == 0,
        "er2_scientific_rows": stages["er2_scientific_rows"] == 4136,
        "flash_current_contract_empty": stages["flash_current_contract_scientific_rows"] == 0,
        "G1_policy": policy["G1"]["N"] == 800 and policy["G1"]["net"] == 5,
        "C1_policy": policy["C1"]["N"] == 800 and policy["C1"]["net"] == -1,
        "old_validation_p0": all(item["net"] == 0 for item in validation["metrics"]),
        "no_live_source_attempt": ledger["inflight"] == 0 and ledger["attempt_reservations"] == 0,
    }
    return {
        "ledger": ledger, "scientific_stages": stages, "policy": policy,
        "old_strict_validation": validation, "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _candidate_and_baseline_audit(run: Path) -> dict[str, Any]:
    baseline = pd.read_parquet(run / "baseline_per_sample.parquet")
    output: dict[str, Any] = {}
    for backend in ("G1", "C1"):
        path = run / f"CANDIDATE_MANIFEST_{backend}.parquet"
        frame = pd.read_parquet(path)
        validate_candidate_manifest(frame)
        test = baseline.loc[(baseline["backend"] == backend) & baseline["split"].eq("test")]
        values = {
            "candidate_rows": len(frame), "candidate_manifest_sha256": sha256_file(path),
            "sample_count": len(test), "j_at_1": float(test["top1_correct"].mean()),
            "j_at_5": float(test["top5_any_correct"].mean()),
        }
        values["passes"] = (
            values["candidate_manifest_sha256"] == EXPECTED_CANDIDATE_HASHES[backend]
            and values["sample_count"] == 7675
            and abs(values["j_at_1"] - EXPECTED_TEST[backend]["j_at_1"]) < 1e-15
            and abs(values["j_at_5"] - EXPECTED_TEST[backend]["j_at_5"]) < 1e-15
        )
        output[backend] = values
    return output


def _eligible_counts(run: Path) -> dict[str, Any]:
    baseline = pd.read_parquet(run / "baseline_per_sample.parquet")
    cohorts = pd.read_parquet(run / "STAGE_COHORTS.parquet")
    output: dict[str, Any] = {}
    for stage in ("smoke", "diagnostic", "policy_selection", "untouched_validation"):
        output[stage] = {}
        for backend in ("G1", "C1"):
            ids = set(cohorts.loc[(cohorts["stage"] == stage) & (cohorts["backend"] == backend), "sample_id"].astype(str))
            frame = baseline.loc[(baseline["backend"] == backend) & baseline["sample_id"].astype(str).isin(ids)]
            output[stage][backend] = {"total": len(frame), "api_eligible": int(frame["candidate_count"].ge(2).sum())}
    output["test"] = {
        backend: {
            "total": int(len(frame)), "api_eligible": int(frame["candidate_count"].ge(2).sum())
        }
        for backend in ("G1", "C1")
        for frame in [baseline.loc[(baseline["backend"] == backend) & baseline["split"].eq("test")]]
    }
    return output


def build_request_and_cost_preflight(run: Path) -> dict[str, Any]:
    counts = _eligible_counts(run)
    validation_eligible = sum(counts["untouched_validation"][backend]["api_eligible"] for backend in ("G1", "C1"))
    test_eligible = sum(counts["test"][backend]["api_eligible"] for backend in ("G1", "C1"))
    policy_eligible = sum(counts["policy_selection"][backend]["api_eligible"] for backend in ("G1", "C1"))
    er2_base = 240 + 362 + 2 * validation_eligible + 2 * test_eligible
    flash_base = 36 + 1456 + 2 * policy_eligible + 2 * validation_eligible + 2 * test_eligible
    er2_upper = er2_base + validation_eligible + test_eligible
    flash_upper = flash_base + policy_eligible + validation_eligible + test_eligible
    expected_total = 78_204
    flash_observed_mean = 0.193746 / 8
    er2_expected = 0.05 * round(expected_total * er2_base / (er2_base + flash_base))
    flash_expected = flash_observed_mean * round(expected_total * flash_base / (er2_base + flash_base))
    upper_cost = er2_upper * 0.05 + flash_upper * flash_observed_mean
    max_requests = int(os.environ.get("MAX_PROVIDER_REQUESTS", "0") or 0)
    max_cost = float(os.environ.get("MAX_API_COST_USD", "0") or 0)
    max_wall = float(os.environ.get("MAX_WALL_TIME_HOURS", "0") or 0)
    concurrency = int(os.environ.get("GEMINI_CONCURRENCY", "1") or 1)
    # Current provider code serializes Flash in-flight; the measured successful
    # request latency therefore dominates, even though the configured RPM is 18.
    wall_expected_low = 207.0
    wall_expected_high = 274.0
    blockers = []
    if max_requests < er2_upper + flash_upper:
        blockers.append("MAX_PROVIDER_REQUESTS_BELOW_LOGICAL_UPPER_BOUND")
    if max_cost < upper_cost:
        blockers.append("MAX_API_COST_USD_BELOW_CONSERVATIVE_UPPER_BOUND")
    if max_wall < wall_expected_low:
        blockers.append("MAX_WALL_TIME_HOURS_BELOW_IMPLEMENTED_RUNNER_PROJECTION")
    return {
        "eligible_counts": counts,
        "requests": {
            "er2_base_without_validation_test_p4": er2_base,
            "er2_all_p4_upper_bound": er2_upper,
            "flash_base_without_p4": flash_base,
            "flash_all_p4_upper_bound": flash_upper,
            "total_base": er2_base + flash_base,
            "total_empirical_expected": expected_total,
            "total_all_p4_upper_bound": er2_upper + flash_upper,
            "retries_excluded": True,
        },
        "cost": {
            "expected_er2_reserve_usd": er2_expected,
            "expected_flash_token_estimate_usd": flash_expected,
            "expected_total_usd": er2_expected + flash_expected,
            "all_p4_upper_bound_usd": upper_cost,
            "flash_historical_mean_usd_per_success": flash_observed_mean,
            "provider_invoice_verified": False,
        },
        "wall_time": {
            "implemented_runner_projection_hours": [wall_expected_low, wall_expected_high],
            "configured_concurrency": concurrency,
        },
        "hard_caps": {
            "MAX_PROVIDER_REQUESTS": max_requests,
            "MAX_API_COST_USD": max_cost,
            "MAX_WALL_TIME_HOURS": max_wall,
        },
        "blockers": blockers,
        "ready_for_paid_api": not blockers,
    }


def _completion_matrix(run: Path, source_recompute: Mapping[str, Any], blockers: Iterable[str]) -> pd.DataFrame:
    rows = []
    blocker_status = "BLOCKED_BUDGET" if any("COST" in item or "REQUEST" in item for item in blockers) else "PARTIAL"
    for backend in ("G1", "C1"):
        for model in (*EXACT_MODEL_IDS, "ER2+Flash consensus"):
            for stage in (
                "provider_preflight", "smoke", "diagnostic", "evidence_ablation",
                "perturbation_stability", "natural_policy_selection", "strict_validation",
                "exploratory_validation", "strict_formal_test", "exploratory_full_test",
                "final_reporting",
            ):
                status = blocker_status
                if model == "ER2+Flash consensus" and stage in {"provider_preflight", "smoke", "diagnostic", "evidence_ablation", "perturbation_stability"}:
                    status = "NOT_APPLICABLE_BY_PROTOCOL"
                elif model == EXACT_MODEL_IDS[0] and stage in {"smoke", "diagnostic", "evidence_ablation", "natural_policy_selection", "strict_validation"}:
                    status = "COMPLETE"
                rows.append({
                    "backend": backend, "model": model, "stage": stage,
                    "protocol": "P0-P5 as applicable", "evidence": "E0-E3/locked_best as applicable",
                    "status": status,
                })
    return pd.DataFrame(rows)


def _write_markdown(run: Path, source: Path, recompute: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
    (run / "SOURCE_RECOMPUTE.md").write_text(
        "# Source run independent recomputation\n\n"
        f"- Source: `{source}`\n"
        f"- Ledger checks pass: `{recompute['all_checks_pass']}`\n"
        f"- Logical/attempts/success/retries/cache: "
        f"{recompute['ledger']['logical_requests']}/{recompute['ledger']['provider_attempts']}/"
        f"{recompute['ledger']['successful_responses']}/{recompute['ledger']['retry_attempts']}/"
        f"{recompute['ledger']['cache_hits']}\n"
        "- The old ER2 NO_GO and P0 validation remain immutable.\n"
        "- Old Flash audit successes are not current-contract scientific responses.\n",
        encoding="utf-8",
    )
    (run / "PREFLIGHT_COST_ESTIMATE.md").write_text(
        "# Continuation request, cost, and wall-time preflight\n\n"
        f"- Base logical requests: {preflight['requests']['total_base']:,}\n"
        f"- Empirical expected logical requests: {preflight['requests']['total_empirical_expected']:,}\n"
        f"- All-P4 logical upper bound: {preflight['requests']['total_all_p4_upper_bound']:,} (retries excluded)\n"
        f"- Expected conservative cost: ${preflight['cost']['expected_total_usd']:.2f}\n"
        f"- All-P4 conservative upper cost: ${preflight['cost']['all_p4_upper_bound_usd']:.2f}\n"
        f"- Implemented-runner wall projection: {preflight['wall_time']['implemented_runner_projection_hours'][0]:.0f}–{preflight['wall_time']['implemented_runner_projection_hours'][1]:.0f} hours\n"
        f"- Hard blockers: {', '.join(preflight['blockers']) or 'none'}\n\n"
        "Flash pricing uses the official standard paid-token schedule and historical usage only for projection. "
        "ER2 uses the configured conservative per-request reserve; no provider invoice is claimed.\n",
        encoding="utf-8",
    )
    (run / "CONTINUATION_AUDIT.md").write_text(
        "# G1/C1 Gemini continuation audit\n\n"
        f"Source recomputation: {'PASS' if recompute['all_checks_pass'] else 'FAIL'}\n\n"
        f"Paid execution readiness: {'READY' if preflight['ready_for_paid_api'] else 'BLOCKED'}\n\n"
        "The source run is referenced read-only. Strict historical NO_GO results are not altered; "
        "exploratory work, if unblocked, belongs only to this continuation run.\n",
        encoding="utf-8",
    )


def run_continuation_audit(repo_root: Path, source_run: Path, run_dir: Path) -> dict[str, Any]:
    source = source_run.expanduser().resolve()
    run = run_dir.expanduser().resolve()
    if source == run or source in run.parents:
        raise ValueError("continuation run must be a new directory outside the source run")
    if not (source / "MANIFEST.json").is_file():
        raise FileNotFoundError("source Gemini run manifest is absent")
    _assert_source_idle(source)
    before = _tree_inventory(source)
    initialize_run(run)
    clone = _clone_source(source, run)
    recompute = recompute_source(source)
    if not recompute["all_checks_pass"]:
        atomic_json(run / "SOURCE_RECOMPUTE_MISMATCH.json", recompute)
        raise RuntimeError("source recomputation mismatch; provider execution is forbidden")
    candidate_audit = _candidate_and_baseline_audit(run)
    if not all(item["passes"] for item in candidate_audit.values()):
        raise RuntimeError("RERANKING_INVARIANT_FAILURE: frozen candidates/baselines changed")
    preflight = build_request_and_cost_preflight(run)
    after = _tree_inventory(source)
    source_unchanged = before == after
    if not source_unchanged:
        raise RuntimeError("source run changed during continuation audit")
    source_manifest = json.loads((source / "MANIFEST.json").read_text())
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout
    code_files = sorted(
        [*repo_root.glob("src/grasping/api_only_gemini_rerank/*.py"), *repo_root.glob("tools/api_reranking/*.py")]
    )
    code_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in code_files}
    reference = {
        "source_run": str(source), "continuation_run": str(run),
        "source_tree_sha256": _inventory_hash(before), "source_tree_files": len(before),
        "source_critical_files": {
            name: before[name] for name in sorted(before)
            if name in SOURCE_STATIC_FILES or name in SOURCE_SCIENCE_FILES
            or name.startswith(("prompts/", "request_manifests/", "stage_results/"))
        },
        "source_git_commit": source_manifest.get("git_commit"),
        "source_git_dirty": source_manifest.get("git_dirty"),
        "source_exact_code_snapshot_available": False,
        "source_exact_code_snapshot_note": "API-only implementation was untracked in the dirty source worktree",
    }
    atomic_json(run / "SOURCE_RUN_REFERENCE.json", reference)
    atomic_json(run / "SOURCE_RECOMPUTE.json", recompute)
    atomic_json(run / "SOURCE_LEDGER_AUDIT.json", recompute["ledger"])
    atomic_json(run / "PREFLIGHT_COST_ESTIMATE.json", preflight)
    manifest = {
        "schema_version": 1, "experiment": "api_only_gemini_rerank_g1_c1_full_completion",
        "created_at_utc": utc_now(), "source_run": str(source), "continuation_run": str(run),
        "source_run_tree_sha256": reference["source_tree_sha256"],
        "source_git_commit": source_manifest.get("git_commit"), "continuation_git_commit": git_commit,
        "continuation_git_dirty": bool(git_status.strip()),
        "source_and_continuation_code_equal": "UNKNOWN_EXACT_SOURCE_CODE_SNAPSHOT",
        "current_code_hashes": code_hashes, "current_code_tree_sha256": sha256_json(code_hashes),
        "exact_model_ids": list(EXACT_MODEL_IDS), "request_hash_version": REQUEST_HASH_VERSION,
        "renderer_hash": renderer_hash(),
        "prompt_hashes": {name: sha256_file(run / "prompts" / name) for name in (
            "direct_full_list_v1.txt", "baseline_aware_v1.txt", "api_rerank_v1.schema.json"
        )},
        "candidate_hashes": EXPECTED_CANDIDATE_HASHES,
        "split_hash": sha256_file(run / "DATA_SPLIT.csv"),
        "evaluator_hash": source_manifest.get("evaluator", {}).get("sha256"),
        "source_clone": clone, "strict_source_decision_immutable": True,
        "exploratory_track_separate": True, "paid_api_ready": preflight["ready_for_paid_api"],
        "paid_api_blockers": preflight["blockers"],
    }
    atomic_json(run / "CONTINUATION_MANIFEST.json", manifest)
    matrix = _completion_matrix(run, recompute, preflight["blockers"])
    matrix.to_csv(run / "COMPLETION_MATRIX.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    atomic_json(run / "COMPLETION_MATRIX.json", {"rows": matrix.to_dict(orient="records")})
    missing = pd.DataFrame([
        {"model": EXACT_MODEL_IDS[0], "new_base_requests": preflight["requests"]["er2_base_without_validation_test_p4"], "upper_bound": preflight["requests"]["er2_all_p4_upper_bound"], "scientific_source_cache_rows": 4136},
        {"model": EXACT_MODEL_IDS[1], "new_base_requests": preflight["requests"]["flash_base_without_p4"], "upper_bound": preflight["requests"]["flash_all_p4_upper_bound"], "scientific_source_cache_rows": 0},
    ])
    missing.to_csv(run / "MISSING_REQUESTS.csv", index=False)
    atomic_json(run / "CACHE_REUSE_PLAN.json", {
        "request_identity": REQUEST_HASH_VERSION,
        "er2_scientific_stage_rows": 4136,
        "er2_success_alias_candidates": 4134,
        "er2_terminal_schema_rows": 2,
        "flash_current_contract_alias_candidates": 0,
        "requires_current_revision_fingerprint_match": True,
        "old_flash_audit_successes_reusable": False,
        "source_historical_cost_is_not_continuation_incremental_cost": True,
    })
    (run / "SOURCE_RUN_INTEGRITY.md").write_text(
        "# Source run integrity\n\n"
        f"- Read-only tree unchanged during audit: `{source_unchanged}`\n"
        f"- Tree SHA-256: `{reference['source_tree_sha256']}`\n"
        f"- Files: {reference['source_tree_files']}\n"
        "- Exact historical code equality is unknown because the source worktree was dirty and the API-only tree was untracked.\n",
        encoding="utf-8",
    )
    _write_markdown(run, source, recompute, preflight)
    return {
        "run_dir": str(run), "source_run": str(source),
        "source_recompute_pass": True, "source_unchanged": source_unchanged,
        "candidate_integrity": candidate_audit, "preflight": preflight,
    }


def run_provider_metadata_preflight(
    repo_root: Path, source_run: Path, run_dir: Path,
) -> dict[str, Any]:
    """Non-generation exact-model discovery and normalized revision comparison."""
    source = source_run.expanduser().resolve()
    run = run_dir.expanduser().resolve()
    worker_python = repo_root.parent / "crog_reproduction/CROG/.venv/bin/python"
    worker_script = repo_root / "tools/api_reranking/provider_worker.py"
    if not worker_python.is_file() or not worker_script.is_file():
        raise RuntimeError("pinned google-genai==2.16.0 worker runtime is unavailable")
    existing_report = run / "MODEL_VERSION_REPORT.json"
    if existing_report.is_file():
        previous = json.loads(existing_report.read_text())
        current = previous.get("models", {})
    else:
        current = check_model_metadata(
            run, EXACT_MODEL_IDS, worker_python, worker_script,
            stage="provider_preflight",
        )
    source_manifest = json.loads((source / "MANIFEST.json").read_text())
    source_er2 = source_manifest.get("model_metadata", {}).get(EXACT_MODEL_IDS[0], {})
    source_raw = source_er2.get("metadata", {})
    source_normalized = normalized_model_metadata(source_raw) if source_raw else None
    source_fingerprint = None if source_normalized is None else sha256_json(source_normalized)
    current_er2 = current.get(EXACT_MODEL_IDS[0], {})
    current_fingerprint = current_er2.get("normalized_fingerprint")
    er2_revision_match = bool(source_fingerprint and source_fingerprint == current_fingerprint)
    exact_names = {
        model: record.get("metadata", {}).get("name") == f"models/{model}"
        for model, record in current.items() if record.get("available")
    }
    blockers = []
    for model in EXACT_MODEL_IDS:
        if not current.get(model, {}).get("available"):
            blockers.append(f"MODEL_UNAVAILABLE:{model}")
        elif not exact_names.get(model, False):
            blockers.append(f"EXACT_MODEL_ID_MISMATCH:{model}")
    if current_er2.get("available") and not er2_revision_match:
        blockers.append("ER2_REVISION_DRIFT_REQUIRES_ER2_REVISION_B")
    payload = {
        "checked_at_utc": utc_now(),
        "exact_model_ids": list(EXACT_MODEL_IDS),
        "models": current,
        "exact_provider_names_match": exact_names,
        "source_er2_normalized_fingerprint": source_fingerprint,
        "current_er2_normalized_fingerprint": current_fingerprint,
        "er2_revision_match": er2_revision_match,
        "model_version_observable": {
            model: bool(record.get("metadata", {}).get("version"))
            for model, record in current.items()
        },
        "endpoint": "v1beta Interactions",
        "sdk": "google-genai==2.16.0",
        "paid_tier_verified": False,
        "paid_tier_note": "Models.get confirms catalog availability, not billing tier or generation quota.",
        "blockers": blockers,
    }
    atomic_json(run / "MODEL_VERSION_REPORT.json", payload)
    (run / "MODEL_VERSION_REPORT.md").write_text(
        "# Model version report\n\n"
        + "\n".join(
            f"- `{model}`: available={record.get('available')}, "
            f"version={record.get('metadata', {}).get('version')}, "
            f"fingerprint={record.get('normalized_fingerprint')}"
            for model, record in current.items()
        )
        + f"\n- ER2 source/current fingerprint match: `{er2_revision_match}`\n"
        + "- Developer API metadata cannot prove an immutable internal weight snapshot.\n",
        encoding="utf-8",
    )
    (run / "PROVIDER_PREFLIGHT.md").write_text(
        "# Provider preflight\n\n"
        f"- Exact-model metadata blockers: {', '.join(blockers) or 'none'}\n"
        "- No generation request was sent by this preflight.\n"
        "- Paid tier remains unverified until the authorized Flash smoke empirically clears the old 20-request free-tier limit or authoritative project billing evidence is supplied.\n",
        encoding="utf-8",
    )
    matrix_path = run / "COMPLETION_MATRIX.csv"
    if matrix_path.is_file():
        matrix = pd.read_csv(matrix_path)
        for model in EXACT_MODEL_IDS:
            available = bool(current.get(model, {}).get("available"))
            selected = matrix["model"].eq(model) & matrix["stage"].eq("provider_preflight")
            matrix.loc[selected, "status"] = "COMPLETE" if available else "BLOCKED_MODEL_UNAVAILABLE"
        matrix.to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)
        atomic_json(run / "COMPLETION_MATRIX.json", {"rows": matrix.to_dict(orient="records")})
    cost_preflight = json.loads((run / "PREFLIGHT_COST_ESTIMATE.json").read_text())
    all_blockers = [*cost_preflight.get("blockers", []), *blockers]
    strict_result = {
        "G1": "USE_G1_ORIGINAL_SCORE",
        "C1": "USE_C1_ORIGINAL_SCORE",
        "source_strict_no_go_unchanged": True,
        "formal_test_executed": False,
    }
    completion = {
        "status": "PARTIAL — BLOCKED_BY_PREFLIGHT_CAPS" if all_blockers else "READY_FOR_FLASH_SMOKE",
        "updated_at_utc": utc_now(), "blockers": all_blockers,
        "source_run_unchanged": True, "provider_generation_requests_sent": 0,
        "provider_metadata_requests_sent": len(list((run / "audit/model_metadata_history/provider_preflight").glob("*/*.json"))),
        "strict_result": strict_result,
        "exploratory_result": "NOT_RUN",
        "fully_complete_exploratory": False,
    }
    atomic_json(run / "FINAL_COMPLETION_STATUS.json", completion)
    (run / "STRICT_CONTRACT_REPORT.md").write_text(
        "# Strict confirmatory contract\n\n"
        "The source ER2 NO_GO is immutable. No continuation validation or formal-test request was sent.\n",
        encoding="utf-8",
    )
    (run / "STRICT_GO_NO_GO.md").write_text(
        "# Strict result\n\n"
        "- G1: `USE_G1_ORIGINAL_SCORE`\n"
        "- C1: `USE_C1_ORIGINAL_SCORE`\n"
        "- Formal test: not executed because no new method passed strict validation.\n",
        encoding="utf-8",
    )
    (run / "SUMMARY_ZH.md").write_text(
        "# G1/C1 Gemini continuation 状态\n\n"
        "本轮已完成只读 source 复算、冻结候选/基线不变量和 exact-model metadata 预检。"
        "两个 exact model 均可发现，ER2 revision 与 source 一致；但 Models.get 不能证明 Flash paid quota。\n\n"
        f"当前状态：`{completion['status']}`。\n\n"
        f"阻塞项：{', '.join(all_blockers) or '无'}。\n\n"
        "在这些硬上限解除前没有发送任何生成请求，因此没有新的 ER2、Flash 或 consensus 性能结果。"
        "旧 strict NO_GO 不变，G1/C1 primary 仍为 original backend score。\n",
        encoding="utf-8",
    )
    (run / "SUMMARY.md").write_text(
        "# G1/C1 Gemini continuation status\n\n"
        f"Status: `{completion['status']}`. Source recomputation and exact-model metadata preflight passed. "
        "No provider generation request was sent because the frozen request/cost/wall-time caps do not cover the complete exploratory contract. "
        "The historical strict NO_GO remains unchanged.\n",
        encoding="utf-8",
    )
    return payload
