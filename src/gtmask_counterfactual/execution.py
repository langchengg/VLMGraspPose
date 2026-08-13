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
from .io import atomic_parquet

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


def artifact_record(path: Path) -> dict[str, str]:
    source = path.expanduser().resolve()
    return {"path": str(source), "sha256": sha256_file(source)}


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
    if authority.get("gt_candidate_generation_authorized") is not True:
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
) -> Path:
    """Write one immutable sample shard via temporary/fsync/atomic rename."""

    sample = str(sample_id).strip()
    if not sample:
        raise ExecutionContractError("sample_id is empty")
    digest = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    destination = (
        run_dir.expanduser().resolve()
        / "06_gtmask_predictions"
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
    if len(set(identities)) != len(identities):
        raise ExecutionContractError("native per-sample output has duplicate IDs")
    if set(by_sample).difference(identities):
        raise ExecutionContractError("candidate output references an unknown sample")
    if expected_executed_sample_ids is not None and set(identities) != set(
        expected_executed_sample_ids
    ):
        raise ExecutionContractError("native output differs from the P2 executable set")
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
        stage = (
            "P4_G1_COUNTERFACTUAL_COMPLETE"
            if route == "g1"
            else "P5_C1_COUNTERFACTUAL_COMPLETE"
        )
        with counterfactual_job(
            run_dir,
            stage=stage,
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
    canonical_root = (
        run_dir.expanduser().resolve()
        / "06_gtmask_predictions"
        / route.lower()
        / branch.lower()
    )
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
    result["content_sha256"] = canonical_sha256(result)
    destination = (
        run_dir.expanduser().resolve()
        / "06_gtmask_predictions"
        / route.lower()
        / branch.lower()
        / "manifest.json"
    )
    return _atomic_json(destination, result)


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
    "d1_case_b_preflight",
    "probe_frozen_docker_image",
    "run_g1_c1_oracle",
]
