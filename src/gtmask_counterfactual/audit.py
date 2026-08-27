"""Source-lock verification and independent run lifecycle controls."""

from __future__ import annotations

import json
import fcntl
import os
import platform
import sqlite3
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.ledger import initialize_ledger, render_ledger_commands

from .contracts import RUN_DIRECTORIES, RUN_STATE_ORDER, SCIENTIFIC_NAME, RunState
from .io import artifact_record, atomic_json, atomic_text, canonical_sha256, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} is not a JSON object: {path}")
    return value


def _inventory_hash(lock: Mapping[str, Any], inventory: list[Any]) -> str:
    keys = [
        key for key in ("inventory_sha256", "inventory_content_sha256") if key in lock
    ]
    if len(keys) != 1:
        raise RuntimeError("source final lock must declare exactly one inventory hash")
    expected = lock.get(keys[0])
    observed = canonical_sha256(inventory)
    if expected != observed:
        raise RuntimeError("source final-lock canonical inventory hash differs")
    return observed


def rehash_inventory_bytes(
    source_run: str | Path,
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rehash every recorded inventory file without opening scientific rows."""

    root = Path(source_run).expanduser().resolve()
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(inventory):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"source inventory row {index} is not an object")
        relative = row.get("relative_path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise RuntimeError(
                f"source inventory relative path is invalid: {relative!r}"
            )
        seen.add(relative)
        unresolved_path = root / relative
        if unresolved_path.is_symlink():
            raise RuntimeError(f"source inventory path is a symlink: {relative}")
        expected_path = unresolved_path.resolve()
        try:
            expected_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"source inventory escapes its run: {relative}"
            ) from error
        recorded_path = Path(str(row.get("path", ""))).expanduser().resolve()
        if recorded_path != expected_path:
            raise RuntimeError(
                f"source inventory path/relative_path disagree: {relative}"
            )
        observed = artifact_record(expected_path)
        if (
            row.get("sha256") != observed["sha256"]
            or row.get("bytes") != observed["bytes"]
        ):
            raise RuntimeError(f"source inventory byte record differs: {relative}")
        verified.append({"relative_path": relative, **observed})
    return verified


def verify_source_final_lock(
    source_run: str | Path,
    *,
    expected_file_sha256: str,
    full_inventory_rehash: bool,
) -> dict[str, Any]:
    """Verify file, self, canonical inventory, and optionally every inventory byte."""

    root = Path(source_run).expanduser().resolve()
    lock_path = root / "FINAL_RUN_LOCK.json"
    observed_file_sha256 = sha256_file(lock_path)
    if observed_file_sha256 != expected_file_sha256:
        raise RuntimeError("source FINAL_RUN_LOCK.json file SHA-256 differs")
    lock = _read_object(lock_path, name="source final lock")
    unsigned = dict(lock)
    recorded_self = unsigned.pop("self_sha256", None)
    if lock.get("status") != "COMPLETE" or recorded_self != canonical_sha256(unsigned):
        raise RuntimeError("source final-lock status/self hash differs")
    inventory = lock.get("inventory")
    if not isinstance(inventory, list) or lock.get("inventory_count") != len(inventory):
        raise RuntimeError("source final-lock inventory count differs")
    inventory_sha256 = _inventory_hash(lock, inventory)
    if not full_inventory_rehash:
        raise PermissionError("full source inventory byte rehash is mandatory")
    verified = rehash_inventory_bytes(root, inventory)
    return {
        "status": "PASS",
        "source_run": str(root),
        "final_lock": artifact_record(lock_path),
        "final_lock_self_sha256": recorded_self,
        "inventory_sha256": inventory_sha256,
        "inventory_count": len(inventory),
        "inventory_files_rehashed": len(verified),
        "scientific_rows_opened": 0,
    }


def verify_source_locks(
    sources: Mapping[str, tuple[str | Path, str]],
    *,
    full_inventory_rehash: bool,
) -> dict[str, Any]:
    records = {
        name: verify_source_final_lock(
            source_run,
            expected_file_sha256=expected,
            full_inventory_rehash=full_inventory_rehash,
        )
        for name, (source_run, expected) in sorted(sources.items())
    }
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "verified_at_utc": utc_now(),
        "full_inventory_byte_rehash": True,
        "sources": records,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def source_immutability_snapshot(
    source_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce a full byte audit to the exact immutable source controls."""

    sources = source_verification.get("sources")
    if source_verification.get("status") != "PASS" or not isinstance(sources, Mapping):
        raise RuntimeError("source immutability snapshot requires a PASS verification")
    records: dict[str, dict[str, Any]] = {}
    rehashes: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for name, value in sorted(sources.items()):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"source verification is invalid: {name}")
        root = Path(str(value.get("source_run", ""))).expanduser().resolve()
        final_lock = artifact_record(root / "FINAL_RUN_LOCK.json")
        if value.get("final_lock") != final_lock:
            raise RuntimeError(f"source final-lock authority differs: {name}")
        manifest_path = root / "manifest.json"
        manifest = _read_object(manifest_path, name=f"{name} root manifest")
        if manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"source root manifest is not COMPLETE: {name}")
        try:
            formal_count = int(manifest["formal_test_execution_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"source formal count is invalid: {name}") from error
        if formal_count != 1:
            raise RuntimeError(f"source formal count is not exactly one: {name}")
        records[f"{name}_final_lock"] = final_lock
        records[f"{name}_root_manifest"] = artifact_record(manifest_path)
        rehashes[name] = {
            "inventory_sha256": value.get("inventory_sha256"),
            "inventory_count": value.get("inventory_count"),
            "inventory_files_rehashed": value.get("inventory_files_rehashed"),
            "final_lock_self_sha256": value.get("final_lock_self_sha256"),
        }
        if (
            not isinstance(rehashes[name]["inventory_sha256"], str)
            or rehashes[name]["inventory_count"]
            != rehashes[name]["inventory_files_rehashed"]
            or not isinstance(rehashes[name]["final_lock_self_sha256"], str)
        ):
            raise RuntimeError(f"source full-rehash evidence is invalid: {name}")
        counts[name] = formal_count
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "full_inventory_byte_rehash": True,
        "sources": records,
        "inventory_rehashes": rehashes,
        "formal_test_counts": counts,
        "scientific_rows_opened": 0,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def write_source_immutability_snapshot(
    run_dir: str | Path,
    source_verification: Mapping[str, Any],
    *,
    phase: str,
) -> Path:
    """Write the mandatory BEFORE or AFTER source immutability snapshot."""

    normalized = phase.upper()
    if normalized not in {"BEFORE", "AFTER"}:
        raise ValueError("source immutability phase must be BEFORE or AFTER")
    root = Path(run_dir).expanduser().resolve()
    payload = source_immutability_snapshot(source_verification)
    if normalized == "AFTER":
        before = _read_object(
            root / "00_audit/SOURCE_RUN_IMMUTABILITY_BEFORE.json",
            name="source immutability BEFORE",
        )
        if (
            payload["sources"] != before.get("sources")
            or payload["inventory_rehashes"]
            != before.get("inventory_rehashes")
            or payload["formal_test_counts"] != before.get("formal_test_counts")
        ):
            raise RuntimeError("source immutability AFTER differs from BEFORE")
    destination = root / f"00_audit/SOURCE_RUN_IMMUTABILITY_{normalized}.json"
    atomic_json(destination, payload)
    alias = root / f"source_hashes_{normalized.lower()}.json"
    atomic_json(alias, payload)
    return destination


def write_reproduction_controls(run_dir: str | Path) -> dict[str, Path]:
    """Persist environment, Git state and reproducible command documentation."""

    root = Path(run_dir).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    audit_commands = {
        "pwd": ["pwd"],
        "head": ["git", "rev-parse", "HEAD"],
        "status_porcelain": ["git", "status", "--porcelain=v1"],
        "diff_stat": ["git", "diff", "--stat"],
        "python_version": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "hardware": ["system_profiler", "SPHardwareDataType"],
        "disk": ["df", "-h"],
        "processes": ["ps", "aux"],
        "swap": ["sysctl", "vm.swapusage"],
    }
    command_results: dict[str, Any] = {}
    for name, command in audit_commands.items():
        completed = subprocess.run(
            command,
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        command_results[name] = {
            "command": command,
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    dependency_script = (
        "import importlib, json\n"
        "names=['torch','numpy','pandas','sklearn','lightgbm','cv2','scipy','shapely']\n"
        "out={}\n"
        "for name in names:\n"
        " try:\n"
        "  mod=importlib.import_module(name); out[name]=getattr(mod,'__version__','UNKNOWN')\n"
        " except Exception as exc: out[name]=f'UNAVAILABLE:{type(exc).__name__}:{exc}'\n"
        "try:\n"
        " import torch; out['mps_built']=torch.backends.mps.is_built(); out['mps_available']=torch.backends.mps.is_available()\n"
        "except Exception: pass\n"
        "print(json.dumps(out,sort_keys=True))\n"
    )
    dependency_probe = subprocess.run(
        [sys.executable, "-c", dependency_script],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    command_results["dependency_versions"] = {
        "command": [sys.executable, "-c", "<version probe>"],
        "return_code": dependency_probe.returncode,
        "stdout": dependency_probe.stdout,
        "stderr": dependency_probe.stderr,
    }
    environment = (
        f"pwd={repository}\n"
        f"python_executable={Path(sys.executable).resolve()}\n"
        f"python_version={sys.version}\n"
        f"platform={platform.platform()}\n"
        f"machine={platform.machine()}\n"
    )
    environment_path = atomic_text(root / "environment.txt", environment)
    git_path = atomic_json(root / "git_state.json", command_results)
    registry_rows: list[dict[str, Any]] = []
    status = _read_object(
        root / "pipeline_status.json", name="counterfactual pipeline status"
    )
    verification_record = status.get("source_lock_verification")
    if isinstance(verification_record, Mapping):
        verification = _read_object(
            Path(str(verification_record.get("path", ""))),
            name="counterfactual source verification",
        )
        for name, value in sorted(verification.get("sources", {}).items()):
            if isinstance(value, Mapping):
                registry_rows.append(
                    {
                        "source_name": name,
                        "source_run": value.get("source_run"),
                        "final_lock_sha256": value.get("final_lock", {}).get("sha256"),
                        "inventory_sha256": value.get("inventory_sha256"),
                        "inventory_count": value.get("inventory_count"),
                        "inventory_files_rehashed": value.get(
                            "inventory_files_rehashed"
                        ),
                    }
                )
    registry_text = "source_name,source_run,final_lock_sha256,inventory_sha256,inventory_count,inventory_files_rehashed\n"
    for row in registry_rows:
        registry_text += (
            ",".join(
                json.dumps(row[key], ensure_ascii=False)
                for key in (
                    "source_name",
                    "source_run",
                    "final_lock_sha256",
                    "inventory_sha256",
                    "inventory_count",
                    "inventory_files_rehashed",
                )
            )
            + "\n"
        )
    registry_path = atomic_text(
        root / "00_audit/source_run_registry.csv", registry_text
    )
    immutability_path = atomic_text(
        root / "00_audit/SOURCE_IMMUTABILITY_AUDIT.md",
        "# Source immutability audit\n\n"
        "Both frozen source runs were verified by final-lock file SHA, self hash, "
        "canonical inventory hash, and a byte rehash of every inventoried file. "
        "No scientific table rows were opened by this audit. A second full byte "
        "rehash is mandatory before terminal finalization.\n",
    )
    readme = atomic_text(
        root / "README_REPRODUCE.md",
        "# GT-mask counterfactual reproduction\n\n"
        "This run is an isolated post-formal oracle diagnostic. Never invoke a "
        "source formal runner. Every heavy command requires a fresh PASS 3×5-minute "
        "resource gate and the repository-global D1 flock.\n\n"
        "The exact executed commands are exported from `run_ledger.sqlite` to "
        "`commands.log`. Resume only with the same `--run-dir` and the recorded "
        "command plus `--resume`. A legal D1 Case-B runtime blocker produces "
        "`PARTIAL`, never `COMPLETE`.\n",
    )
    return {
        "environment": environment_path,
        "git_state": git_path,
        "readme": readme,
        "source_registry": registry_path,
        "immutability_audit": immutability_path,
    }


def initialize_counterfactual_ledger(path: str | Path) -> Path:
    destination = initialize_ledger(path)
    with sqlite3.connect(destination) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS counterfactual_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                route TEXT NOT NULL DEFAULT '',
                branch TEXT NOT NULL DEFAULT '',
                sample_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                command TEXT NOT NULL DEFAULT '',
                start_time TEXT NOT NULL,
                end_time TEXT,
                artifact_path TEXT NOT NULL DEFAULT '',
                artifact_sha256 TEXT NOT NULL DEFAULT '',
                error_summary TEXT NOT NULL DEFAULT '',
                UNIQUE(stage, route, branch, sample_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS counterfactual_jobs_status_idx "
            "ON counterfactual_jobs(status)"
        )
    return destination


def export_counterfactual_commands(run_dir: str | Path) -> Path:
    """Export both authoritative ledger tables as deterministic JSONL.

    Unified ``ledger_stage`` exports only its ``stages`` table.  GT-mask work
    also has per-route/per-sample rows, so terminal provenance must include
    both tables rather than silently dropping the counterfactual jobs.
    """

    root = Path(run_dir).expanduser().resolve()
    ledger = root / "run_ledger.sqlite"
    if ledger.is_symlink() or not ledger.is_file():
        raise RuntimeError(f"counterfactual ledger is absent or unsafe: {ledger}")
    lock_path = root / ".commands.log.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        stages = render_ledger_commands(ledger)
        with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """SELECT id, stage, route, branch, sample_id, status, command,
                          start_time, end_time, artifact_path, artifact_sha256,
                          error_summary
                   FROM counterfactual_jobs
                   ORDER BY stage, route, branch, sample_id, id"""
            ).fetchall()
        jobs = "".join(
            json.dumps(
                {"ledger_table": "counterfactual_jobs", **dict(row)},
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )
        destination = atomic_text(root / "commands.log", stages + jobs)
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
        return destination
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def bootstrap_run(
    run_dir: str | Path,
    *,
    source_verification: Mapping[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    if source_verification.get("status") != "PASS" or not source_verification.get(
        "full_inventory_byte_rehash"
    ):
        raise PermissionError("bootstrap requires a full PASS source verification")
    nonempty = root.exists() and any(root.iterdir())
    if nonempty and not resume:
        raise FileExistsError(f"counterfactual run is non-empty; pass --resume: {root}")
    if (root / "FINAL_COUNTERFACTUAL_RUN_LOCK.json").exists() or (
        root / "COMPLETE"
    ).exists():
        raise PermissionError("counterfactual run is terminal")
    if nonempty and resume:
        current = _read_object(
            root / "pipeline_status.json", name="counterfactual pipeline status"
        )
        record = current.get("source_lock_verification")
        verified = verify_artifact_records_recursive(
            record,
            name="counterfactual source verification",
            require_at_least_one=True,
        )
        if len(verified) != 1:
            raise RuntimeError("counterfactual source verification binding differs")
        return current
    for relative in RUN_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    initialize_counterfactual_ledger(root / "run_ledger.sqlite")
    source_path = atomic_json(
        root / "00_audit" / "source_lock_verification.json",
        source_verification,
    )
    status = {
        "schema_version": 1,
        "scientific_name": SCIENTIFIC_NAME,
        "status": RunState.P0_AUDIT.value,
        "first_incomplete_stage": RunState.P1_BASELINE_REPLAY_PASS.value,
        "counterfactual_execution_count": 0,
        "raw_test_ground_truth_rows_read": 0,
        "created_at_utc": utc_now(),
        "source_lock_verification": artifact_record(source_path),
    }
    status_path = root / "pipeline_status.json"
    atomic_json(status_path, status)
    atomic_text(root / "commands.log", "")
    write_source_immutability_snapshot(root, source_verification, phase="BEFORE")
    write_reproduction_controls(root)
    atomic_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "scientific_name": SCIENTIFIC_NAME,
            "status": RunState.P0_AUDIT.value,
            "counterfactual_execution_count": 0,
            "formal_test_execution_count": 0,
            "source_formal_test_execution_modified": False,
        },
    )
    return status


def require_verified_source(
    run_dir: str | Path,
    *,
    source_name: str,
    source_run: str | Path,
) -> dict[str, Any]:
    """Require a bootstrap-bound source run and an unchanged final-lock file."""

    root = Path(run_dir).expanduser().resolve()
    status = _read_object(
        root / "pipeline_status.json", name="counterfactual pipeline status"
    )
    record = status.get("source_lock_verification")
    verified = verify_artifact_records_recursive(
        record,
        name="counterfactual source verification",
        require_at_least_one=True,
    )
    if len(verified) != 1:
        raise RuntimeError("counterfactual source verification binding differs")
    verification = _read_object(
        Path(verified[0]["path"]), name="counterfactual source verification"
    )
    sources = verification.get("sources")
    source = sources.get(source_name) if isinstance(sources, Mapping) else None
    expected_root = Path(source_run).expanduser().resolve()
    if (
        verification.get("status") != "PASS"
        or verification.get("full_inventory_byte_rehash") is not True
        or not isinstance(source, Mapping)
        or source.get("source_run") != str(expected_root)
    ):
        raise PermissionError(f"source is not bootstrap-authorized: {source_name}")
    lock_record = source.get("final_lock")
    if not isinstance(lock_record, Mapping):
        raise RuntimeError(f"source final-lock record is absent: {source_name}")
    observed = artifact_record(expected_root / "FINAL_RUN_LOCK.json")
    if (
        lock_record.get("path") != observed["path"]
        or lock_record.get("sha256") != observed["sha256"]
        or lock_record.get("bytes") != observed["bytes"]
    ):
        raise RuntimeError(f"source final-lock bytes changed: {source_name}")
    return dict(source)


def write_source_immutability_after(run_dir: str | Path) -> Path:
    """Rehash the complete source inventories and write the terminal snapshot.

    This function is intentionally expensive.  Production callers must invoke
    it only while holding the repository-global heavy-work lease after a fresh
    resource gate.  Merely re-reading the original verification JSON would not
    detect an inventoried source artifact changed behind an unchanged lock.
    """

    root = Path(run_dir).expanduser().resolve()
    status = _read_object(
        root / "pipeline_status.json", name="counterfactual pipeline status"
    )
    record = status.get("source_lock_verification")
    verified = verify_artifact_records_recursive(
        record,
        name="counterfactual source verification",
        require_at_least_one=True,
    )
    if len(verified) != 1:
        raise RuntimeError("counterfactual source verification binding differs")
    original = _read_object(
        Path(verified[0]["path"]), name="counterfactual source verification"
    )
    sources = original.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise RuntimeError("counterfactual source verification has no sources")
    requested: dict[str, tuple[str, str]] = {}
    for name, value in sorted(sources.items()):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"counterfactual source record is invalid: {name}")
        final_lock = value.get("final_lock")
        if not isinstance(final_lock, Mapping):
            raise RuntimeError(f"counterfactual source lock record is absent: {name}")
        requested[str(name)] = (
            str(value.get("source_run", "")),
            str(final_lock.get("sha256", "")),
        )
    refreshed = verify_source_locks(requested, full_inventory_rehash=True)
    return write_source_immutability_snapshot(root, refreshed, phase="AFTER")


def transition_pipeline_status(
    run_dir: str | Path,
    target: RunState | str,
    *,
    first_incomplete_stage: str | None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    status_path = root / "pipeline_status.json"
    current = _read_object(status_path, name="counterfactual pipeline status")
    observed = str(current.get("status", ""))
    target_value = str(target)
    terminal = {
        RunState.COMPLETE.value,
        RunState.PARTIAL.value,
        RunState.FAILED.value,
    }
    if observed in terminal:
        if observed != target_value:
            raise PermissionError(
                f"terminal status cannot transition: {observed} -> {target_value}"
            )
        # A terminal transition is an idempotent observation, not a new
        # transition.  Rewriting it would replace the original pre-terminal
        # provenance with ``COMPLETE -> COMPLETE`` and make a crash before
        # final-lock publication impossible to reconcile safely.
        return current
    elif target_value in {RunState.PARTIAL.value, RunState.FAILED.value}:
        pass
    elif target_value not in RUN_STATE_ORDER or observed not in RUN_STATE_ORDER:
        raise ValueError(f"unknown lifecycle transition: {observed} -> {target_value}")
    elif (
        observed == RunState.P5B_G1_FULL_COMPLETE.value
        and target_value == RunState.P7_TAXONOMY_COMPLETE.value
    ):
        # D1 is a secondary extension attempted only after the complete core
        # report/recompute chain.  The historical P6 label must not force D1
        # into the G1/C1 critical path.
        pass
    elif RUN_STATE_ORDER[target_value] not in {
        RUN_STATE_ORDER[observed],
        RUN_STATE_ORDER[observed] + 1,
    }:
        raise PermissionError(
            f"pipeline status must advance exactly one stage: {observed} -> {target_value}"
        )
    updated = {
        **current,
        "previous_status": observed,
        "status": target_value,
        "first_incomplete_stage": first_incomplete_stage,
        "updated_at_utc": utc_now(),
    }
    atomic_json(status_path, updated)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_object(manifest_path, name="counterfactual manifest")
        manifest["status"] = target_value
        atomic_json(manifest_path, manifest)
    return updated


def verify_bound_artifacts(value: Any, *, name: str) -> list[dict[str, str]]:
    """Expose the project recursive artifact verifier for protocol consumers."""

    return verify_artifact_records_recursive(
        value,
        name=name,
        require_at_least_one=True,
    )
