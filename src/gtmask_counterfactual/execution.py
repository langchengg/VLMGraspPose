"""Execution guards and atomic adapters for GT-mask counterfactual routes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from typing import Any, Iterator

import pandas as pd

from unified_reranking.hashing import canonical_sha256, sha256_file
from gtmask_counterfactual.protocol import load_execution_authority

from .candidate_matching import stable_candidate_id
from .io import artifact_record, atomic_json, atomic_parquet

from .resource import validate_fresh_gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_NATIVE_INFERENCE_SHA256 = (
    "5bbae71830db8264494b20651ff6c62fd87210011614b159c48f11bfd4586544"
)
FROZEN_D1_CANDIDATE_SCRIPT_SHA256 = (
    "86c84e1ff1b7674dc4f089c21f6bd6fb67cc7925f3be16b14ce5638eb6da7623"
)
FROZEN_D1_SCORER_SCRIPT_SHA256 = (
    "f8b875cf91a97eb0884f836a5b0303da31deffb01606d4bd75b794b8c36fc02b"
)
FROZEN_D1_CONFIG_SHA256 = (
    "96b0f85bd053cb59c28566ce55b5b4b1d12c9db89240370d35d6c24d083e40fb"
)
FROZEN_DOCKER_IMAGE = "vlmgrasp/gqcnn-score:1.3.0"
FROZEN_DOCKER_IMAGE_ID = (
    "sha256:3d1158ca83197d55808454b718d0a328d3f27c57c80baaaea7031e21a9134ebd"
)
FROZEN_D1_SOURCE = (
    REPOSITORY_ROOT
    / "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
)
FROZEN_G1_C1_SOURCE = (
    REPOSITORY_ROOT / "HiFi_reproduction/runs/"
    "modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500"
)
FROZEN_D1_CANDIDATE_SCRIPT = (
    FROZEN_D1_SOURCE / "source_snapshot/scripts/run_hifics_dexnet_candidates.py"
)
FROZEN_D1_SCORER_SCRIPT = (
    FROZEN_D1_SOURCE / "source_snapshot/scripts/run_full_gqcnn_scoring.py"
)
FROZEN_D1_CONFIG = (
    FROZEN_D1_SOURCE
    / "source_snapshot/configs/dexnet_candidates_formal_no_refinement.yaml"
)
NATIVE_INFERENCE = (
    REPOSITORY_ROOT / "experiments/fair_crog_hifics_g1_c1_no_rerank/native_inference.py"
)


class ExecutionContractError(RuntimeError):
    """A route launch differs from the locked counterfactual contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ExecutionContractError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionContractError(f"cannot read {label}: {source}") from error
    if not isinstance(value, dict):
        raise ExecutionContractError(f"{label} must be a JSON object")
    expected_self = value.get("self_sha256")
    if expected_self is not None:
        unsigned = {key: item for key, item in value.items() if key != "self_sha256"}
        if expected_self != canonical_sha256(unsigned):
            raise ExecutionContractError(f"{label} self hash differs")
    expected_content = value.get("content_sha256")
    if expected_content is not None:
        unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
        if expected_content != canonical_sha256(unsigned):
            raise ExecutionContractError(f"{label} content hash differs")
    return value


def _verify_artifact(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    expected = str(record.get("sha256", ""))
    if not expected or sha256_file(path) != expected:
        raise ExecutionContractError(f"{label} hash differs")
    return path


def authorize_gt_candidate_generation(
    *,
    protocol_lock_path: Path,
    registry_path: Path,
    route: str,
    branch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authorize the only transition that may feed GT pixels to a generator."""

    route_name = route.lower()
    branch_name = branch.lower()
    if route_name not in {"g1", "c1", "d1"}:
        raise ExecutionContractError(f"unsupported route: {route}")
    if branch_name not in {"gt_oracle", "gt_shape_only"}:
        raise ExecutionContractError("this guard authorizes GT branches only")
    lock_path = protocol_lock_path.expanduser().resolve()
    run_dir = lock_path.parents[1]
    expected_lock_path = run_dir / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json"
    if lock_path != expected_lock_path:
        raise ExecutionContractError("protocol lock is outside the canonical run path")
    try:
        authority = load_execution_authority(lock_path)
    except (ValueError, RuntimeError) as error:
        raise PermissionError(
            "GT candidate generation is forbidden before P4 lock"
        ) from error
    authorized = (
        authority.get("d1_candidate_generation_authorized")
        if route_name == "d1"
        else authority.get("gt_candidate_generation_authorized")
    )
    if authorized is not True:
        raise PermissionError(
            "protocol lock does not authorize GT candidate generation"
        )
    registry = authority.get("gt_mask_registry")
    if not isinstance(registry, Mapping):
        raise ExecutionContractError("protocol lock has no GT registry artifact")
    verified_registry = _verify_artifact(registry, label="locked GT registry")
    if verified_registry != registry_path.expanduser().resolve():
        raise ExecutionContractError("requested GT registry differs from protocol lock")
    routes = authority.get("routes")
    if not isinstance(routes, Mapping) or not isinstance(
        routes.get(route_name), Mapping
    ):
        raise ExecutionContractError(
            f"protocol lock lacks route contract: {route_name}"
        )
    # The repository protocol creates this sole claim and increments the new
    # run's counter before any bulk generator may start.
    claim_path = lock_path.parent / "COUNTERFACTUAL_EXECUTION.json"
    claim = _load_json(claim_path, label="counterfactual execution claim")
    pipeline = _load_json(run_dir / "pipeline_status.json", label="pipeline status")
    if (
        claim.get("status") != "RUNNING"
        or int(claim.get("execution_count", -1)) != 1
        or claim.get("protocol_lock_file_sha256") != sha256_file(lock_path)
        or int(pipeline.get("counterfactual_execution_count", -1)) != 1
    ):
        raise PermissionError("counterfactual execution claim differs")
    route_contract = dict(routes[route_name])
    allowed = route_contract.get("allowed_gt_branches", ["gt_oracle"])
    if branch_name not in allowed:
        raise PermissionError(f"{route_name}/{branch_name} is not locked")
    return authority, route_contract


def authorize_retrospective_import(
    *,
    protocol_lock_path: Path,
    registry_path: Path,
    route: str,
    branch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authorize a verified import without a current-run execution claim."""

    route_name = route.lower()
    branch_name = branch.lower()
    if route_name not in {"g1", "c1"} or branch_name != "gt_oracle":
        raise ExecutionContractError("retrospective import supports G1/C1 GT-oracle")
    lock_path = protocol_lock_path.expanduser().resolve()
    run_dir = lock_path.parents[1]
    if lock_path != run_dir / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json":
        raise ExecutionContractError("protocol lock is outside the canonical run path")
    authority = load_execution_authority(lock_path)
    registry = authority.get("gt_mask_registry")
    if not isinstance(registry, Mapping) or (
        _verify_artifact(registry, label="locked GT registry")
        != registry_path.expanduser().resolve()
    ):
        raise ExecutionContractError("requested GT registry differs from protocol lock")
    routes = authority.get("routes")
    if not isinstance(routes, Mapping) or not isinstance(routes.get(route_name), Mapping):
        raise ExecutionContractError(f"protocol lock lacks route: {route_name}")
    route_contract = dict(routes[route_name])
    if route_contract.get("execution_mode") != "retrospective_verified_import":
        raise PermissionError("route is not locked for retrospective verified import")
    pipeline = _load_json(run_dir / "pipeline_status.json", label="pipeline status")
    if int(pipeline.get("counterfactual_execution_count", -1)) != 0:
        raise PermissionError(
            "retrospective import requires current-run execution count zero"
        )
    if (lock_path.parent / "COUNTERFACTUAL_EXECUTION.json").exists():
        raise PermissionError("retrospective import must not create an execution claim")
    return authority, route_contract


def append_gt_access_log(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Durably append one auditable GT access event under an advisory lock."""

    root = run_dir.expanduser().resolve()
    destination = root / "logs/gt_mask_access.log"
    lock_path = root / "logs/.gt_mask_access.lock"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **dict(payload),
        }
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return destination


@contextmanager
def counterfactual_job(
    run_dir: Path,
    *,
    stage: str,
    route: str,
    branch: str,
    sample_id: str = "",
    command: str = "",
) -> Iterator[dict[str, str]]:
    """Record one aggregate or per-sample real-GT job in the core ledger."""

    ledger = run_dir.expanduser().resolve() / "run_ledger.sqlite"
    started = datetime.now(timezone.utc).isoformat()
    identity = (stage, route, branch, sample_id)
    existing_complete: tuple[str, str, str] | None = None
    with sqlite3.connect(ledger) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='counterfactual_jobs'"
        ).fetchone()
        if table is None:
            raise ExecutionContractError(
                "counterfactual_jobs ledger is not initialized"
            )
        previous = connection.execute(
            """SELECT status, command, artifact_path, artifact_sha256
               FROM counterfactual_jobs
               WHERE stage=? AND route=? AND branch=? AND sample_id=?""",
            identity,
        ).fetchone()
        if previous is not None and str(previous[0]) == "COMPLETE":
            previous_command = str(previous[1] or "")
            artifact_path = str(previous[2] or "")
            artifact_sha = str(previous[3] or "")
            if previous_command != command:
                raise ExecutionContractError(
                    "completed counterfactual job command identity differs"
                )
            if artifact_path:
                path = Path(artifact_path).expanduser().resolve()
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or artifact_sha != sha256_file(path)
                ):
                    raise ExecutionContractError(
                        "completed counterfactual job artifact differs"
                    )
            elif artifact_sha:
                raise ExecutionContractError(
                    "completed counterfactual job has an incomplete artifact record"
                )
            existing_complete = (previous_command, artifact_path, artifact_sha)
        else:
            if previous is not None and str(previous[1] or "") != command:
                raise ExecutionContractError(
                    "resumed counterfactual job command identity differs"
                )
            connection.execute(
                """
                INSERT INTO counterfactual_jobs
                  (stage, route, branch, sample_id, status, command, start_time)
                VALUES (?, ?, ?, ?, 'RUNNING', ?, ?)
                ON CONFLICT(stage, route, branch, sample_id) DO UPDATE SET
                  status='RUNNING', start_time=excluded.start_time, end_time=NULL,
                  artifact_path='', artifact_sha256='', error_summary=''
                """,
                (*identity, command, started),
            )
    state: dict[str, str] = {}
    if existing_complete is not None:
        state["artifact_path"] = existing_complete[1]
        state["artifact_sha256"] = existing_complete[2]
    try:
        yield state
    except BaseException as error:
        if existing_complete is not None:
            raise ExecutionContractError(
                "completed counterfactual job resume body raised; ledger remains immutable"
            ) from error
        with sqlite3.connect(ledger) as connection:
            connection.execute(
                """UPDATE counterfactual_jobs SET status='FAILED', end_time=?,
                   error_summary=? WHERE stage=? AND route=? AND branch=? AND sample_id=?""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    f"{type(error).__name__}: {error}",
                    *identity,
                ),
            )
        raise
    else:
        artifact_path = state.get("artifact_path", "")
        artifact_sha = state.get("artifact_sha256", "")
        final_status = state.get("status", "COMPLETE")
        if final_status not in {"COMPLETE", "BLOCKED", "MACHINE_BLOCKED"}:
            raise ExecutionContractError(
                f"invalid counterfactual job status: {final_status}"
            )
        if existing_complete is not None:
            if (
                final_status != "COMPLETE"
                or artifact_path != existing_complete[1]
                or artifact_sha != existing_complete[2]
            ):
                raise ExecutionContractError(
                    "completed counterfactual job cannot be rewritten on resume"
                )
            return
        with sqlite3.connect(ledger) as connection:
            connection.execute(
                """UPDATE counterfactual_jobs SET status=?, end_time=?,
                   artifact_path=?, artifact_sha256=?
                   WHERE stage=? AND route=? AND branch=? AND sample_id=?""",
                (
                    final_status,
                    datetime.now(timezone.utc).isoformat(),
                    artifact_path,
                    artifact_sha,
                    *identity,
                ),
            )


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_sample_json(
    *,
    run_dir: Path,
    route: str,
    branch: str,
    sample_id: str,
    payload: Mapping[str, Any],
    output_scope: str | None = None,
) -> Path:
    """Write one immutable sample shard via temporary/fsync/atomic rename."""

    sample = str(sample_id).strip()
    if not sample:
        raise ExecutionContractError("sample_id is empty")
    digest = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    prediction_root = run_dir.expanduser().resolve() / "06_gtmask_predictions"
    if output_scope is not None:
        scope = str(output_scope).strip().lower()
        if not scope or "/" in scope or scope in {".", ".."}:
            raise ExecutionContractError("output scope is unsafe")
        prediction_root /= scope
    destination = (
        prediction_root
        / route.lower()
        / branch.lower()
        / "samples"
        / digest[:2]
        / f"{digest}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {**dict(payload), "sample_id": sample}
    encoded = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, indent=2)
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != encoded:
            raise ExecutionContractError(
                f"existing sample shard differs: {destination}"
            )
        return destination
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    _fsync_parent(destination)
    return destination


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, indent=2)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_parent(path)
    return path


def _exact_or_write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Publish once; a resume must prove exact logical equality."""

    destination = path.expanduser().resolve()
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ExecutionContractError(f"existing Parquet is unsafe: {destination}")
        existing = pd.read_parquet(destination)
        if list(existing.columns) != list(frame.columns) or not existing.equals(frame):
            raise ExecutionContractError(f"existing Parquet differs: {destination}")
        return destination
    return atomic_parquet(frame, destination)


def adapt_native_cumulative_to_sample_shards(
    *,
    run_dir: Path,
    route: str,
    branch: str,
    native_output: Path,
    expected_executed_sample_ids: set[str] | None = None,
    technical_complement_ids: set[str] | None = None,
    source_adapter_manifest: Path | None = None,
    allow_source_superset: bool = False,
    output_scope: str | None = None,
    stage: str | None = None,
) -> Path:
    """Safely adapt the frozen runner's cumulative parquets to atomic shards."""

    import pyarrow.parquet as pq

    sample_path = native_output / "per_sample.parquet"
    candidate_path = native_output / "candidates.parquet"
    manifest_path = native_output / "run_manifest.json"
    manifest = _load_json(manifest_path, label="native inference manifest")
    if manifest.get("status") != "COMPLETE":
        raise ExecutionContractError("native inference output is incomplete")
    samples = pq.read_table(sample_path).to_pylist()
    candidates = pq.read_table(candidate_path).to_pylist()
    source_identities = [str(row["sample_id"]) for row in samples]
    if len(set(source_identities)) != len(source_identities):
        raise ExecutionContractError("native per-sample output has duplicate IDs")
    if expected_executed_sample_ids is not None:
        expected = set(expected_executed_sample_ids)
        source_set = set(source_identities)
        if not expected.issubset(source_set):
            raise ExecutionContractError("native output omits an expected sample")
        if not allow_source_superset and source_set != expected:
            raise ExecutionContractError(
                "native output differs from the expected execution set"
            )
        samples = [row for row in samples if str(row["sample_id"]) in expected]
        candidates = [
            row for row in candidates if str(row["sample_id"]) in expected
        ]
    for candidate in candidates:
        candidate["source_candidate_id"] = str(candidate.get("candidate_id", ""))
        candidate["source_candidate_index"] = int(candidate.get("native_rank", 0))
        candidate["candidate_id"] = stable_candidate_id(
            sample_id=str(candidate["sample_id"]),
            route=route,
            branch=branch,
            source_candidate_index=candidate["source_candidate_index"],
            candidate={
                **candidate,
                "width_px": candidate.get("width_px", candidate.get("jaw_width_px")),
                "height_px": candidate.get(
                    "height_px", candidate.get("rectangle_height_px")
                ),
            },
        )
        candidate["route"] = route.upper()
        candidate["branch"] = branch.lower()
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_sample[str(candidate["sample_id"])].append(dict(candidate))
    identities = [str(row["sample_id"]) for row in samples]
    if set(by_sample).difference(identities):
        raise ExecutionContractError("candidate output references an unknown sample")
    if expected_executed_sample_ids is not None and set(identities) != set(
        expected_executed_sample_ids
    ):
        raise ExecutionContractError("filtered native output differs from expected IDs")
    complement = set(technical_complement_ids or set())
    if set(identities).intersection(complement):
        raise ExecutionContractError("native output overlaps the P2 technical complement")
    normalized_samples: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        count = len(by_sample.get(sample_id, []))
        native_status = sample.get("status")
        normalized_samples.append(
            {
                **dict(sample),
                "route": route.upper(),
                "branch": branch.lower(),
                "candidate_count": count,
                "no_output": count == 0,
                "technical_failure": False,
                "native_status": native_status,
                "status": "NO_OUTPUT" if count == 0 else "COMPLETE",
            }
        )
    for sample_id in sorted(complement):
        normalized_samples.append(
            {
                "sample_id": sample_id,
                "route": route.upper(),
                "branch": branch.lower(),
                "candidate_count": 0,
                "no_output": True,
                "technical_failure": True,
                "native_status": None,
                "status": "TECHNICAL_FAILURE",
                "failure_reason": "P2_UNRESOLVED_MAPPING",
            }
        )
    samples = sorted(normalized_samples, key=lambda row: str(row["sample_id"]))

    inventory: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda row: str(row["sample_id"])):
        sample_id = str(sample["sample_id"])
        job_stage = stage or (
            "P5B_G1_FULL_COMPLETE" if route == "g1" else "P5_C1_FULL_COMPLETE"
        )
        with counterfactual_job(
            run_dir,
            stage=job_stage,
            route=route,
            branch=branch,
            sample_id=sample_id,
            command="adapt_frozen_native_output",
        ) as job:
            shard = atomic_sample_json(
                run_dir=run_dir,
                route=route,
                branch=branch,
                sample_id=sample_id,
                payload={
                    "sample": dict(sample),
                    "candidates": sorted(
                        by_sample.get(sample_id, []),
                        key=lambda row: (
                            int(row.get("native_rank", 0)),
                            str(row.get("candidate_id", "")),
                        ),
                    ),
                    "source_cumulative": {
                        "per_sample_sha256": sha256_file(sample_path),
                        "candidates_sha256": sha256_file(candidate_path),
                        "run_manifest_sha256": sha256_file(manifest_path),
                    },
                },
                output_scope=output_scope,
            )
            job["artifact_path"] = str(shard)
            job["artifact_sha256"] = sha256_file(shard)
        inventory.append(
            {
                "sample_id": sample_id,
                "path": str(shard),
                "sha256": sha256_file(shard),
                "candidate_count": len(by_sample.get(sample_id, [])),
            }
        )
    canonical_root = run_dir.expanduser().resolve() / "06_gtmask_predictions"
    if output_scope is not None:
        canonical_root /= output_scope.lower()
    canonical_root = canonical_root / route.lower() / branch.lower()
    canonical_sample_path = _exact_or_write_parquet(
        pd.DataFrame(samples), canonical_root / "per_sample.parquet"
    )
    canonical_candidate_path = _exact_or_write_parquet(
        pd.DataFrame(candidates),
        canonical_root / "per_candidate.parquet",
    )
    dense_map_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "NOT_PERSISTED_BY_FROZEN_ORACLE_RUNNER",
        "route": route.lower(),
        "branch": branch.lower(),
        "dense_maps_used_for_candidate_generation": True,
        "dense_maps_persisted": False,
        "reason": "frozen native oracle CLI does not expose --save-raw-maps",
        "source_native_manifest": artifact_record(manifest_path),
    }
    dense_map_manifest["content_sha256"] = canonical_sha256(dense_map_manifest)
    dense_map_path = _atomic_json(
        canonical_root / "dense_map_manifest.json", dense_map_manifest
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "route": route.lower(),
        "branch": branch.lower(),
        "sample_count": len(inventory),
        "executed_sample_count": len(identities),
        "technical_complement_count": len(complement),
        "candidate_count": len(candidates),
        "source_sample_count": len(source_identities),
        "source_subset_import": allow_source_superset,
        "atomic_write_contract": "temporary_fsync_atomic_rename_v1",
        "source_native_manifest": artifact_record(manifest_path),
        "per_sample": artifact_record(canonical_sample_path),
        "per_candidate": artifact_record(canonical_candidate_path),
        # Canonical consumer name; ``per_candidate`` is retained for the
        # execution-facing schema and both records must remain identical.
        "candidates": artifact_record(canonical_candidate_path),
        "dense_map_manifest": artifact_record(dense_map_path),
        "candidate_id_namespace": "sha256(sample,route,branch,source_index,geometry)",
        "sample_shards": inventory,
    }
    if source_adapter_manifest is not None:
        result["execution_source_adapter"] = artifact_record(source_adapter_manifest)
    if output_scope is not None:
        result["output_scope"] = output_scope.lower()
    result["content_sha256"] = canonical_sha256(result)
    destination = canonical_root / "manifest.json"
    return _atomic_json(destination, result)


def _retrospective_native_source(
    route: str, route_contract: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Verify every locked byte required for a Case-A retrospective import."""

    if route_contract.get("execution_mode") != "retrospective_verified_import":
        raise ExecutionContractError("G1/C1 contract does not authorize inference")
    for name in (
        "retrospective_source_lock",
        "retrospective_finalization",
        "retrospective_result_hashes",
        "retrospective_canonical_candidates",
    ):
        record = route_contract.get(name)
        if not isinstance(record, Mapping):
            raise ExecutionContractError(f"route lacks locked {name}")
        _verify_artifact(record, label=name)
    snapshot_record = route_contract.get("retrospective_native_inference")
    if not isinstance(snapshot_record, Mapping):
        raise ExecutionContractError("route lacks stored native source snapshot")
    snapshot = _verify_artifact(snapshot_record, label="stored native source snapshot")
    if (
        sha256_file(snapshot) != FROZEN_NATIVE_INFERENCE_SHA256
        or sha256_file(NATIVE_INFERENCE) != FROZEN_NATIVE_INFERENCE_SHA256
    ):
        raise ExecutionContractError("stored/current native inference bytes differ")
    source = route_contract.get("retrospective_native_output")
    if not isinstance(source, Mapping) or set(source) != {
        "run_manifest",
        "per_sample",
        "candidates",
    }:
        raise ExecutionContractError("retrospective native output inventory differs")
    paths = {
        name: _verify_artifact(record, label=f"retrospective {route} {name}")
        for name, record in source.items()
        if isinstance(record, Mapping)
    }
    if set(paths) != set(source) or len({path.parent for path in paths.values()}) != 1:
        raise ExecutionContractError("retrospective native output paths differ")
    manifest = _load_json(paths["run_manifest"], label=f"stored {route} manifest")
    expected = route_contract.get("native_manifest_fields")
    if not isinstance(expected, Mapping) or any(
        manifest.get(name) != value for name, value in expected.items()
    ):
        raise ExecutionContractError("stored native manifest differs from protocol")
    if int(manifest.get("sample_count", -1)) != 7_675:
        raise ExecutionContractError("stored native output is not the full denominator")
    result_hashes_path = _verify_artifact(
        route_contract["retrospective_result_hashes"],
        label="retrospective result hashes",
    )
    result_hashes = _load_json(result_hashes_path, label="retrospective result hashes")
    canonical_record = route_contract["retrospective_canonical_candidates"]
    recorded = result_hashes.get("files", {}).get(
        "03_canonical/canonical_candidates.parquet"
    )
    if not isinstance(recorded, Mapping) or (
        recorded.get("sha256") != canonical_record.get("sha256")
        or recorded.get("bytes") != canonical_record.get("bytes")
    ):
        raise ExecutionContractError("final result hashes do not bind candidates")
    raw = pd.read_parquet(paths["candidates"])
    canonical = pd.read_parquet(
        route_contract["retrospective_canonical_candidates"]["path"]
    )
    canonical = canonical.loc[
        canonical["method"].astype(str).eq(f"{route.upper()}-ORACLE")
    ].copy()
    core_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "jaw_width_px",
        "rectangle_height_px",
        "source_row",
        "source_column",
        "status",
        "failure_reason",
        "transform_json",
        "checkpoint_sha256",
        "selected_config_sha256",
        "native_decoder_config_sha256",
    ]
    if any(column not in raw or column not in canonical for column in core_columns):
        raise ExecutionContractError("raw/canonical candidate core schema differs")
    order = ["sample_id", "native_rank", "candidate_id"]
    raw_core = raw.loc[:, core_columns].sort_values(order, kind="mergesort").reset_index(drop=True)
    canonical_core = canonical.loc[:, core_columns].sort_values(
        order, kind="mergesort"
    ).reset_index(drop=True)
    if not raw_core.equals(canonical_core):
        raise ExecutionContractError(
            "stored raw candidates differ from formally locked canonical core fields"
        )
    return paths["run_manifest"].parent, manifest


def _pilot_acceptance_payload(
    *,
    route_contract: Mapping[str, Any],
    pilot_manifest: Mapping[str, Any],
    imported_manifest_path: Path,
) -> dict[str, Any]:
    """Independently recount the 200 stored raw and finalized C1 outputs."""

    imported = _load_json(imported_manifest_path, label="C1 pilot import manifest")
    selection = pd.read_parquet(pilot_manifest["selection"]["path"])
    selected = selection["sample_id"].astype(str).tolist()
    selected_set = set(selected)
    native_records = route_contract["retrospective_native_output"]
    raw_samples = pd.read_parquet(native_records["per_sample"]["path"])
    raw_candidates = pd.read_parquet(native_records["candidates"]["path"])
    raw_samples = raw_samples.loc[
        raw_samples["sample_id"].astype(str).isin(selected_set)
    ].copy()
    raw_candidates = raw_candidates.loc[
        raw_candidates["sample_id"].astype(str).isin(selected_set)
    ].copy()
    imported_samples = pd.read_parquet(imported["per_sample"]["path"])
    imported_candidates = pd.read_parquet(imported["per_candidate"]["path"])
    finalized = pd.read_parquet(
        route_contract["retrospective_canonical_candidates"]["path"],
        columns=["sample_id", "method"],
    )
    finalized = finalized.loc[
        finalized["sample_id"].astype(str).isin(selected_set)
        & finalized["method"].astype(str).eq("C1-ORACLE")
    ]
    if (
        len(selected) != 200
        or len(selected_set) != 200
        or set(raw_samples["sample_id"].astype(str)) != selected_set
        or set(imported_samples["sample_id"].astype(str)) != selected_set
        or int(imported.get("technical_complement_count", -1)) != 0
        or len(raw_candidates) != len(imported_candidates)
        or len(finalized) != len(raw_candidates)
    ):
        raise ExecutionContractError("C1 pilot retrospective population differs")
    raw_counts = raw_candidates.groupby(
        raw_candidates["sample_id"].astype(str)
    ).size()
    declared_counts = raw_samples.set_index(
        raw_samples["sample_id"].astype(str)
    )["candidate_count"].astype(int)
    if not declared_counts.equals(raw_counts.reindex(declared_counts.index, fill_value=0)):
        raise ExecutionContractError("C1 pilot native candidate recount differs")
    predicted_counts = {
        str(name): int(count)
        for name, count in selection["predicted_outcome_class"]
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    required = {
        "predicted_success",
        "predicted_no_positive",
        "predicted_no_output",
    }
    if not required.issubset(predicted_counts):
        raise ExecutionContractError("C1 pilot lacks required predicted outcomes")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "P4_C1_PILOT_PASS",
        "acceptance_method": "independent_raw_manifest_and_finalized_recount_v1",
        "sample_count": 200,
        "candidate_count": len(raw_candidates),
        "no_output_count": int((declared_counts == 0).sum()),
        "ordered_selected_ids_sha256": canonical_sha256(selected),
        "predicted_outcome_counts": predicted_counts,
        "query_type_counts": {
            str(name): int(count)
            for name, count in selection["query_type"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "target_size_quartile_counts": {
            str(name): int(count)
            for name, count in selection["target_size_quartile"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        },
        "raw_and_finalized_candidate_count_match": True,
        "pilot_source_adapter": dict(route_contract["c1_pilot_source_adapter"]),
        "import_manifest": artifact_record(imported_manifest_path),
        "stored_native_manifest": dict(native_records["run_manifest"]),
        "stored_native_per_sample": dict(native_records["per_sample"]),
        "stored_native_candidates": dict(native_records["candidates"]),
        "stored_finalized_candidates": dict(
            route_contract["retrospective_canonical_candidates"]
        ),
        "retrospective_reconstruction": artifact_record(
            imported_manifest_path.parent / "RETROSPECTIVE_RECONSTRUCTION.json"
        ),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def verify_c1_pilot_acceptance(
    path: Path, *, route_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute and verify the independently accepted C1 audit pilot."""

    observed = _load_json(path, label="C1 pilot acceptance")
    unsigned = dict(observed)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise ExecutionContractError("C1 pilot acceptance content hash differs")
    pilot_record = route_contract.get("c1_pilot_source_adapter")
    if not isinstance(pilot_record, Mapping):
        raise ExecutionContractError("C1 route lacks a pilot adapter")
    from .pilot import verify_c1_pilot_source_adapter

    pilot = verify_c1_pilot_source_adapter(
        _verify_artifact(pilot_record, label="C1 pilot source adapter")
    )
    imported_path = _verify_artifact(
        observed.get("import_manifest", {}), label="C1 pilot import manifest"
    )
    expected = _pilot_acceptance_payload(
        route_contract=route_contract,
        pilot_manifest=pilot,
        imported_manifest_path=imported_path,
    )
    if observed != expected:
        raise ExecutionContractError("C1 pilot independent acceptance differs")
    return observed


def import_g1_c1_retrospective(
    *,
    run_dir: Path,
    route: str,
    scope: str,
    registry_path: Path,
    protocol_lock_path: Path,
) -> Path:
    """Import immutable Case-A outputs; this function never launches a model."""

    route_name = route.lower()
    scope_name = scope.lower()
    if route_name not in {"g1", "c1"} or scope_name not in {"pilot", "full"}:
        raise ExecutionContractError("unsupported retrospective route/scope")
    if scope_name == "pilot" and route_name != "c1":
        raise ExecutionContractError("the audit pilot is defined only for C1")
    _, route_contract = authorize_retrospective_import(
        protocol_lock_path=protocol_lock_path,
        registry_path=registry_path,
        route=route_name,
        branch="gt_oracle",
    )
    native_output, _ = _retrospective_native_source(route_name, route_contract)
    complement: set[str] = set()
    output_scope: str | None = None
    if scope_name == "pilot":
        from .pilot import verify_c1_pilot_source_adapter

        adapter_path = _verify_artifact(
            route_contract["c1_pilot_source_adapter"],
            label="C1 pilot source adapter",
        )
        adapter = verify_c1_pilot_source_adapter(adapter_path)
        expected_ids = set(
            pd.read_parquet(
                adapter["test_samples"]["path"], columns=["sample_id"]
            )["sample_id"].astype(str)
        )
        stage = "P4_C1_PILOT_PASS"
        output_scope = "pilot"
    else:
        from .g1_c1_adapter import verify_g1_c1_source_adapter

        adapter_path = _verify_artifact(
            route_contract["execution_source_adapter"],
            label=f"{route_name} full source adapter",
        )
        adapter = verify_g1_c1_source_adapter(adapter_path)
        expected_ids = set(
            pd.read_parquet(
                adapter["test_samples"]["path"], columns=["sample_id"]
            )["sample_id"].astype(str)
        )
        complement = set(
            str(value)
            for value in _load_json(
                Path(str(adapter["unresolved_partition"]["path"])),
                label="G1/C1 unresolved partition",
            ).get("sample_ids", [])
        )
        stage = "P5_C1_FULL_COMPLETE" if route_name == "c1" else "P5B_G1_FULL_COMPLETE"
    append_gt_access_log(
        run_dir,
        {
            "stage": stage,
            "route": route_name,
            "branch": "gt_oracle",
            "purpose": "hash_bound_retrospective_case_a_import",
            "model_inference_performed": False,
            "protocol_lock": artifact_record(protocol_lock_path),
            "gt_registry": artifact_record(registry_path),
        },
    )
    imported = adapt_native_cumulative_to_sample_shards(
        run_dir=run_dir,
        route=route_name,
        branch="gt_oracle",
        native_output=native_output,
        expected_executed_sample_ids=expected_ids,
        technical_complement_ids=complement,
        source_adapter_manifest=adapter_path,
        allow_source_superset=True,
        output_scope=output_scope,
        stage=stage,
    )
    reconstruction: dict[str, Any] = {
        "schema_version": 1,
        "status": "VERIFIED",
        "execution_mode": "retrospective_verified_import",
        "route": route_name,
        "scope": scope_name,
        "current_run_model_inference_performed": False,
        "current_run_counterfactual_execution_count": 0,
        "historical_source_execution_preexisting": True,
        "test_outcomes_exposed_before_current_protocol_binding": True,
        "protocol_lock": artifact_record(protocol_lock_path),
        "source_run": route_contract["retrospective_source_run"],
        "source_lock": dict(route_contract["retrospective_source_lock"]),
        "source_finalization": dict(route_contract["retrospective_finalization"]),
        "source_result_hashes": dict(route_contract["retrospective_result_hashes"]),
        "source_native_output": dict(route_contract["retrospective_native_output"]),
        "source_canonical_candidates": dict(
            route_contract["retrospective_canonical_candidates"]
        ),
        "import_manifest": artifact_record(imported),
        "raw_to_locked_canonical_core_fields_exact": True,
    }
    reconstruction["content_sha256"] = canonical_sha256(reconstruction)
    reconstruction_path = imported.parent / "RETROSPECTIVE_RECONSTRUCTION.json"
    if reconstruction_path.exists():
        if _load_json(
            reconstruction_path, label="retrospective reconstruction"
        ) != reconstruction:
            raise ExecutionContractError("existing retrospective reconstruction differs")
    else:
        atomic_json(reconstruction_path, reconstruction)
    if scope_name == "full":
        return imported
    pilot = verify_c1_pilot_source_adapter(adapter_path)
    acceptance = _pilot_acceptance_payload(
        route_contract=route_contract,
        pilot_manifest=pilot,
        imported_manifest_path=imported,
    )
    destination = imported.parent / "C1_PILOT_ACCEPTANCE.json"
    if destination.exists():
        observed = _load_json(destination, label="C1 pilot acceptance")
        if observed != acceptance:
            raise ExecutionContractError("existing C1 pilot acceptance differs")
    else:
        atomic_json(destination, acceptance)
    verify_c1_pilot_acceptance(destination, route_contract=route_contract)
    return destination


def run_g1_c1_oracle(
    *,
    run_dir: Path,
    route: str,
    source_run: Path,
    registry_path: Path,
    protocol_lock_path: Path,
    resume: bool,
    python: Path | None = None,
    command_runner: Any = subprocess.run,
) -> Path:
    """Run the exact-hash native G1/C1 oracle and adapt atomic sample shards."""

    route_name = route.lower()
    if route_name not in {"g1", "c1"}:
        raise ExecutionContractError("native oracle wrapper supports G1/C1 only")
    _, route_contract = authorize_gt_candidate_generation(
        protocol_lock_path=protocol_lock_path,
        registry_path=registry_path,
        route=route_name,
        branch="gt_oracle",
    )
    if sha256_file(NATIVE_INFERENCE) != FROZEN_NATIVE_INFERENCE_SHA256:
        raise ExecutionContractError("G1/C1 native inference source hash differs")
    from .g1_c1_adapter import verify_g1_c1_source_adapter

    adapter_record = route_contract.get("execution_source_adapter")
    if not isinstance(adapter_record, Mapping):
        raise ExecutionContractError("G1/C1 route lacks its P2 source adapter")
    adapter_manifest_path = _verify_artifact(
        adapter_record, label=f"{route_name} execution source adapter"
    )
    adapter_manifest = verify_g1_c1_source_adapter(adapter_manifest_path)
    execution_source = Path(
        str(route_contract.get("execution_source_root", ""))
    ).expanduser().resolve()
    if execution_source != adapter_manifest_path.parent:
        raise ExecutionContractError("G1/C1 execution source root differs")
    expected_source = (
        Path(str(route_contract.get("source_run", ""))).expanduser().resolve()
    )
    if expected_source != source_run.expanduser().resolve():
        raise ExecutionContractError("G1/C1 source run differs from protocol lock")
    label_record = route_contract.get("oracle_label_manifest")
    if not isinstance(label_record, Mapping):
        raise ExecutionContractError("G1/C1 route lock lacks oracle label manifest")
    label_path = _verify_artifact(label_record, label=f"{route_name} oracle labels")
    if label_path != expected_source / "manifests/test_labels.parquet":
        raise ExecutionContractError(
            "G1/C1 oracle label path differs from frozen runner"
        )
    launcher = Path(python or sys.executable)
    if not launcher.is_file():
        raise ExecutionContractError(f"Python launcher is absent: {launcher}")
    adapter_root = (
        run_dir.expanduser().resolve() / "logs/native_oracle_adapter" / route_name
    )
    command = [
        str(launcher),
        str(NATIVE_INFERENCE),
        "--run-dir",
        str(adapter_root),
        "--source-run",
        str(execution_source),
        "--method",
        route_name,
        "--oracle",
    ]
    if resume:
        command.append("--resume")
    append_gt_access_log(
        run_dir,
        {
            "stage": "GT_CANDIDATE_GENERATION",
            "route": route_name,
            "branch": "gt_oracle",
            "purpose": "locked_stage_replacement_counterfactual",
            "protocol_lock": artifact_record(protocol_lock_path),
            "gt_registry": artifact_record(registry_path),
        },
    )
    completed = command_runner(command, check=False, capture_output=True, text=True)
    log = run_dir.expanduser().resolve() / "logs" / f"{route_name}_native_oracle.json"
    _atomic_json(
        log,
        {
            "command": command,
            "return_code": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "source_sha256": sha256_file(NATIVE_INFERENCE),
        },
    )
    if completed.returncode != 0:
        raise ExecutionContractError(f"{route_name} native oracle failed; see {log}")
    native_output = (
        adapter_root / "02_predictions/native_work" / f"{route_name}_gtmask_oracle"
    )
    native_manifest = _load_json(
        native_output / "run_manifest.json", label=f"{route_name} native manifest"
    )
    expected_fields = route_contract.get("native_manifest_fields")
    if not isinstance(expected_fields, Mapping) or not expected_fields:
        raise ExecutionContractError("route lock lacks exact native manifest fields")
    required_manifest_fields = {
        "status",
        "variant",
        "sample_count",
        "checkpoint",
        "checkpoint_sha256",
        "selected_config",
        "selected_config_sha256",
        "raw_maps_saved",
        "source_samples_sha256",
        "source_labels_sha256",
    }
    if set(expected_fields) != required_manifest_fields:
        raise ExecutionContractError(
            "route lock native manifest field inventory differs"
        )
    differences = {
        key: (native_manifest.get(key), expected)
        for key, expected in expected_fields.items()
        if native_manifest.get(key) != expected
    }
    if differences:
        raise ExecutionContractError(
            f"native oracle source contract differs: {differences}"
        )
    return adapt_native_cumulative_to_sample_shards(
        run_dir=run_dir,
        route=route_name,
        branch="gt_oracle",
        native_output=native_output,
        expected_executed_sample_ids=set(
            pd.read_parquet(
                Path(str(adapter_manifest["test_samples"]["path"])),
                columns=["sample_id"],
            )["sample_id"].astype(str)
        ),
        technical_complement_ids=set(
            str(value)
            for value in _load_json(
                Path(str(adapter_manifest["unresolved_partition"]["path"])),
                label="G1/C1 unresolved partition",
            ).get("sample_ids", [])
        ),
        source_adapter_manifest=adapter_manifest_path,
    )


def probe_frozen_docker_image(
    *, command_runner: Any = subprocess.run
) -> dict[str, Any]:
    """Read-only Docker probe; never pulls or starts a container."""

    docker = shutil.which("docker")
    if docker is None:
        return {
            "status": "BLOCKED",
            "blocker_code": "D1_DOCKER_CLI_MISSING",
            "execution_attempted": False,
        }
    command = [
        docker,
        "image",
        "inspect",
        FROZEN_DOCKER_IMAGE,
        "--format",
        "{{.Id}}|{{.Architecture}}|{{.Os}}",
    ]
    completed = command_runner(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return {
            "status": "BLOCKED",
            "blocker_code": "D1_DOCKER_DAEMON_OR_IMAGE_UNAVAILABLE",
            "docker_image": FROZEN_DOCKER_IMAGE,
            "expected_image_id": FROZEN_DOCKER_IMAGE_ID,
            "diagnostic": completed.stderr.strip(),
            "execution_attempted": False,
        }
    pieces = completed.stdout.strip().split("|")
    if pieces != [FROZEN_DOCKER_IMAGE_ID, "amd64", "linux"]:
        return {
            "status": "BLOCKED",
            "blocker_code": "D1_DOCKER_IMAGE_ID_OR_PLATFORM_MISMATCH",
            "observed": pieces,
            "expected": [FROZEN_DOCKER_IMAGE_ID, "amd64", "linux"],
            "execution_attempted": False,
        }
    return {
        "status": "PASS",
        "docker_image": FROZEN_DOCKER_IMAGE,
        "docker_image_id": pieces[0],
        "architecture": pieces[1],
        "os": pieces[2],
        "execution_attempted": False,
    }


def d1_case_b_preflight(
    *,
    run_dir: Path,
    registry_path: Path,
    protocol_lock_path: Path,
    resource_gate: Mapping[str, Any],
    command_runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Verify D1 Case B and emit a machine blocker instead of unsafe work."""

    _, route_contract = authorize_gt_candidate_generation(
        protocol_lock_path=protocol_lock_path,
        registry_path=registry_path,
        route="d1",
        branch="gt_oracle",
    )
    if route_contract.get("case") != "B":
        raise ExecutionContractError("D1 must be locked as Case B")
    if route_contract.get("mask_affects_raw_sampling") is not True:
        raise ExecutionContractError("D1 Case B must regenerate raw candidates")
    if route_contract.get("raw_candidate_regeneration_required") is not True:
        raise ExecutionContractError("D1 raw regeneration is not locked")
    if route_contract.get("filter_only_primary_allowed") is not False:
        raise ExecutionContractError("filter-only D1 cannot masquerade as primary")
    expected_sources = {
        FROZEN_D1_CANDIDATE_SCRIPT: FROZEN_D1_CANDIDATE_SCRIPT_SHA256,
        FROZEN_D1_SCORER_SCRIPT: FROZEN_D1_SCORER_SCRIPT_SHA256,
        FROZEN_D1_CONFIG: FROZEN_D1_CONFIG_SHA256,
    }
    for path, expected in expected_sources.items():
        if sha256_file(path) != expected:
            raise ExecutionContractError(f"frozen D1 source hash differs: {path}")
    validate_fresh_gate(resource_gate)
    docker = probe_frozen_docker_image(command_runner=command_runner)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": docker["status"],
        "route": "d1",
        "branch": "gt_oracle",
        "d1_case": "B",
        "raw_candidate_regeneration_required": True,
        "filter_only_primary_allowed": False,
        "frozen_sources": [artifact_record(path) for path in expected_sources],
        "docker": docker,
        "execution_attempted": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if docker["status"] != "PASS":
        result["blocker_code"] = docker["blocker_code"]
        result["status"] = "MACHINE_BLOCKED"
        result["content_sha256"] = canonical_sha256(result)
        destination = (
            run_dir.expanduser().resolve()
            / "00_audit/machine_blockers/D1_CASE_B_MACHINE_BLOCKER.json"
        )
        _atomic_json(destination, result)
        result["artifact_path"] = str(destination)
        return result
    result.update(
        {
            "status": "BLOCKED",
            "blocker_code": "D1_ISOLATED_GT_INPUT_ADAPTER_REQUIRED",
            "reason": (
                "The byte-frozen candidate runner is prediction-only. An isolated "
                "GT loader must replace only mask_input before invoking the same Case B "
                "sampler; filter-only reuse is forbidden."
            ),
        }
    )
    result["content_sha256"] = canonical_sha256(result)
    destination = (
        run_dir.expanduser().resolve()
        / "00_audit/machine_blockers/D1_CASE_B_ADAPTER_BLOCKER.json"
    )
    _atomic_json(destination, result)
    result["artifact_path"] = str(destination)
    return result


__all__ = [
    "ExecutionContractError",
    "FROZEN_D1_CANDIDATE_SCRIPT_SHA256",
    "FROZEN_D1_CONFIG_SHA256",
    "FROZEN_D1_SCORER_SCRIPT_SHA256",
    "FROZEN_DOCKER_IMAGE_ID",
    "FROZEN_NATIVE_INFERENCE_SHA256",
    "adapt_native_cumulative_to_sample_shards",
    "artifact_record",
    "atomic_sample_json",
    "authorize_gt_candidate_generation",
    "authorize_retrospective_import",
    "d1_case_b_preflight",
    "import_g1_c1_retrospective",
    "probe_frozen_docker_image",
    "run_g1_c1_oracle",
    "verify_c1_pilot_acceptance",
]
