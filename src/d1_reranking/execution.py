"""Immutable resource-gate and execution-authorization contracts for D1."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import canonical_sha256, sha256_file

from .resource_gate import (
    evaluate_resource_gate,
    host_contract,
    resource_thresholds,
)


AUTHORIZATION_START_WINDOW_SECONDS = 300
LEASE_HEARTBEAT_MAX_AGE_SECONDS = 180
GATE_CLOCK_TOLERANCE_SECONDS = 5
CANDIDATE_RESOURCE_SCOPE = {
    "route": "D1",
    "stage": "P2",
    "operation": "canonical_existing_artifact_projection",
    "device": "cpu",
    "max_parallel": 1,
    "split_count": 3,
}
FEATURE_RESOURCE_SCOPE = {
    "route": "D1",
    "stage": "P5",
    "operation": "raw_matched_common_feature_extraction",
    "device": "cpu",
    "max_parallel": 1,
    "job_count": 9,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RANK1_RUN_DIR = (
    REPO_ROOT / "runs" / "reranking_complete_20260803_094159"
).resolve()


@contextmanager
def exclusive_heavy_resource_lease(run_dir: Path, *, purpose: str) -> Iterator[Path]:
    """Hold the repository-wide nonblocking kernel lease for heavy D1 work.

    The stable lock inode lives beside all run directories, so candidate and
    primary jobs from sibling runs cannot execute concurrently.  The kernel
    releases the lease on normal exit, exceptions, or process death; JSON
    heartbeat files remain audit metadata and are never used as mutexes.
    """

    if not purpose.strip():
        raise ValueError("D1 heavy resource lease requires a purpose")
    root = run_dir.expanduser().resolve()
    lock_path = root.parent / ".d1_heavy_resource.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(
            f"cannot open D1 heavy resource lock: {lock_path}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"D1 heavy resource lock is not regular: {lock_path}")
        os.set_inheritable(descriptor, False)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another candidate/training process holds the D1 heavy resource lease: {lock_path}"
            ) from error
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def artifact_record(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    return {"path": str(source), "sha256": sha256_file(source)}


def python_invocation_path(path: str | Path) -> Path:
    """Return an absolute Python launcher path without resolving venv symlinks.

    CPython discovers a virtual environment from the launcher path. Resolving
    ``venv/bin/python`` to the base interpreter before ``subprocess`` changes
    ``sys.prefix`` and silently drops the venv site-packages. Artifact records
    may still bind the resolved executable bytes, but child commands must keep
    the launcher path supplied by the orchestration contract.
    """

    launcher = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if not launcher.is_file():
        raise FileNotFoundError(f"D1 Python launcher is absent: {launcher}")
    return launcher


def load_content_manifest(
    path: str | Path, *, name: str, statuses: tuple[str, ...]
) -> dict[str, Any]:
    value = load_verified_json(path, name=name, statuses=statuses)
    unsigned = dict(value)
    expected = unsigned.pop("content_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash mismatch")
    return value


def resource_policy(
    *, plan_path: Path, source_paths: tuple[Path, ...]
) -> dict[str, Any]:
    # Delayed import avoids a module cycle: primary_execution reuses the
    # generic gate/artifact primitives defined here.
    from .primary_execution import primary_resource_policy

    return primary_resource_policy(plan_path=plan_path, source_paths=source_paths)


def candidate_resource_policy(
    *, source_closure_path: Path, source_paths: tuple[Path, ...]
) -> dict[str, Any]:
    """Freeze the pre-P2 projection gate without depending on the P9 plan."""

    source_closure = load_content_manifest(
        source_closure_path, name="D1 source closure", statuses=("PASS",)
    )
    if (
        source_closure.get("canonical_snapshot") != "A"
        or source_closure.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 candidate resource policy requires canonical P1 closure")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY",
        "gate_type": "d1_candidate_projection_three_continuous_five_minute_windows_v1",
        "scope": CANDIDATE_RESOURCE_SCOPE,
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "prerequisite": artifact_record(source_closure_path),
        "thresholds": resource_thresholds(),
        "candidate_test_labels_read": False,
        "sources": [artifact_record(path) for path in source_paths],
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def load_candidate_resource_policy(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the active immutable P2 policy and its source-closure binding."""

    from .provenance import load_source_closure

    root = Path(run_dir).expanduser().resolve()
    pointer = load_content_manifest(
        root / "configs" / "d1_candidate_resource_gate_policy_active.json",
        name="D1 active candidate resource-policy pointer",
        statuses=("LOCKED_POLICY_POINTER",),
    )
    policy_path = verified_artifact_path(
        pointer.get("active_policy", {}), name="D1 active candidate resource policy"
    )
    expected_parent = (root / "configs" / "candidate_resource_policies").resolve()
    if policy_path.parent != expected_parent:
        raise RuntimeError(
            "D1 candidate resource policy is outside its immutable registry"
        )
    policy = load_content_manifest(
        policy_path, name="D1 candidate resource policy", statuses=("LOCKED_POLICY",)
    )
    closure_path, _closure = load_source_closure(root)
    closure_record = artifact_record(closure_path)
    policy_record = artifact_record(policy_path)
    pointer_closure = pointer.get("source_closure")
    pointer_policy = pointer.get("active_policy")
    if (
        policy.get("prerequisite") != closure_record
        or not isinstance(pointer_closure, Mapping)
        or any(
            pointer_closure.get(key) != value for key, value in closure_record.items()
        )
        or not isinstance(pointer_policy, Mapping)
        or any(pointer_policy.get(key) != value for key, value in policy_record.items())
    ):
        raise RuntimeError("D1 active candidate policy/source-closure binding differs")
    return policy_path, policy


def feature_resource_policy(
    *, plan_path: Path, source_paths: tuple[Path, ...]
) -> dict[str, Any]:
    """Freeze the independent gate for the exact nine raw-feature jobs."""

    plan = load_content_manifest(
        plan_path, name="D1 raw-feature extraction plan", statuses=("PLANNED",)
    )
    if (
        plan.get("job_count") != 9
        or plan.get("scope") != FEATURE_RESOURCE_SCOPE
        or plan.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 feature resource policy requires the exact 9-job plan")
    source_closure = plan.get("sources", {}).get("source_closure")  # type: ignore[union-attr]
    if not isinstance(source_closure, Mapping):
        raise RuntimeError("D1 feature plan source closure is absent")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY",
        "gate_type": "d1_raw_features_three_continuous_five_minute_windows_v1",
        "scope": FEATURE_RESOURCE_SCOPE,
        "rank1_run_dir": str(CANONICAL_RANK1_RUN_DIR),
        "plan": artifact_record(plan_path),
        "prerequisite": dict(source_closure),
        "thresholds": resource_thresholds(),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
        "sources": [artifact_record(path) for path in source_paths],
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def load_feature_resource_policy(
    run_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the versioned active feature policy and exact current plan."""

    from .feature_plan import load_active_feature_extraction_plan
    from .provenance import load_source_closure

    root = Path(run_dir).expanduser().resolve()
    pointer = load_content_manifest(
        root / "configs/d1_feature_resource_gate_policy_active.json",
        name="D1 active feature resource-policy pointer",
        statuses=("LOCKED_POLICY_POINTER",),
    )
    policy_path = verified_artifact_path(
        pointer.get("active_policy", {}), name="D1 active feature resource policy"
    )
    if policy_path.parent != (root / "configs/feature_resource_policies").resolve():
        raise RuntimeError("D1 feature resource policy is outside its registry")
    policy = load_content_manifest(
        policy_path, name="D1 feature resource policy", statuses=("LOCKED_POLICY",)
    )
    plan_path, _plan = load_active_feature_extraction_plan(root)
    closure_path, _closure = load_source_closure(root)
    expected_plan = artifact_record(plan_path)
    expected_closure = artifact_record(closure_path)
    expected_policy = artifact_record(policy_path)
    if (
        policy.get("scope") != FEATURE_RESOURCE_SCOPE
        or policy.get("plan") != expected_plan
        or policy.get("prerequisite") != expected_closure
        or pointer.get("active_policy") != expected_policy
        or pointer.get("plan") != expected_plan
        or pointer.get("source_closure") != expected_closure
    ):
        raise RuntimeError("D1 active feature policy/plan/source binding differs")
    return policy_path, policy


def validate_feature_resource_gate(
    run_dir: Path, *, require_fresh: bool = True
) -> dict[str, Any]:
    """Replay the current PASS gate for the exact raw-feature plan."""

    from .feature_plan import load_active_feature_extraction_plan
    from .provenance import load_source_closure

    root = run_dir.expanduser().resolve()
    policy_path, _policy = load_feature_resource_policy(root)
    closure_path, _closure = load_source_closure(root)
    plan_path, _plan = load_active_feature_extraction_plan(root)
    pointer = load_content_manifest(
        root / "00_audit/RESOURCE_AUDIT.json",
        name="D1 resource gate pointer",
        statuses=("PASS",),
    )
    gate_path = verified_artifact_path(
        pointer.get("latest_gate", {}), name="D1 current feature resource gate"
    )
    return validate_resource_gate(
        gate_path,
        run_dir=root,
        plan_path=plan_path,
        policy_path=policy_path,
        expected_scope=FEATURE_RESOURCE_SCOPE,
        prerequisite=artifact_record(closure_path),
        require_fresh=require_fresh,
    )


def validate_resource_gate(
    gate_path: str | Path,
    *,
    run_dir: Path,
    plan_path: Path | None = None,
    policy_path: Path | None = None,
    expected_scope: Mapping[str, Any] | None = None,
    prerequisite: Mapping[str, Any] | None = None,
    require_fresh: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    source = Path(gate_path).expanduser().resolve()
    expected_parent = root / "00_audit" / "resource_gates"
    if source.parent != expected_parent:
        raise RuntimeError(
            "D1 resource gate is outside the run-local immutable registry"
        )
    gate = load_content_manifest(
        source, name="D1 primary resource gate", statuses=("PASS",)
    )
    if gate.get("candidate_test_labels_read") is not False:
        raise RuntimeError("D1 resource gate violates Test isolation")
    if gate.get("run_dir") != str(root):
        raise RuntimeError("D1 resource gate is bound to another run")
    pointer_path = root / "00_audit" / "RESOURCE_AUDIT.json"
    pointer = load_content_manifest(
        pointer_path, name="D1 resource gate pointer", statuses=("PASS",)
    )
    if pointer.get("latest_gate") != artifact_record(source):
        raise RuntimeError("D1 resource gate is not the current run-local gate")
    if gate.get("host") != host_contract():
        raise RuntimeError("D1 resource gate host/boot session differs")
    if plan_path is not None and gate.get("plan") != artifact_record(plan_path):
        raise RuntimeError("D1 resource gate plan binding differs")
    if prerequisite is not None and gate.get("prerequisite") != dict(prerequisite):
        raise RuntimeError("D1 resource gate prerequisite binding differs")
    policy_record = gate.get("policy")
    bound_policy_path = verified_artifact_path(
        policy_record if isinstance(policy_record, Mapping) else {},
        name="D1 resource gate policy",
    )
    if (
        policy_path is not None
        and bound_policy_path != policy_path.expanduser().resolve()
    ):
        raise RuntimeError("D1 resource gate policy path differs")
    policy = load_content_manifest(
        bound_policy_path, name="D1 resource gate policy", statuses=("LOCKED_POLICY",)
    )
    if gate.get("rank1_run_dir") != str(CANONICAL_RANK1_RUN_DIR) or policy.get(
        "rank1_run_dir"
    ) != str(CANONICAL_RANK1_RUN_DIR):
        raise RuntimeError("D1 resource gate rank1 interlock path differs")
    if pointer.get("scope") != gate.get("scope") or pointer.get(
        "policy"
    ) != artifact_record(bound_policy_path):
        raise RuntimeError("D1 resource gate pointer scope/policy differs")
    if plan_path is not None and policy.get("plan") != artifact_record(plan_path):
        raise RuntimeError("D1 resource policy plan binding differs")
    if prerequisite is not None and policy.get("prerequisite") != dict(prerequisite):
        raise RuntimeError("D1 resource policy prerequisite differs")
    if expected_scope is not None and (
        gate.get("scope") != dict(expected_scope)
        or policy.get("scope") != dict(expected_scope)
    ):
        raise RuntimeError("D1 resource gate scope differs")
    if policy.get("thresholds") != resource_thresholds():
        raise RuntimeError("D1 resource policy thresholds differ from code")
    if gate.get("thresholds") != policy.get("thresholds"):
        raise RuntimeError("D1 resource gate thresholds differ from locked policy")
    verify_artifact_records_recursive(
        {"gate_sources": gate.get("sources"), "policy_sources": policy.get("sources")},
        name="D1 resource gate provenance",
        require_at_least_one=True,
    )
    windows = gate.get("windows")
    if not isinstance(windows, list):
        raise RuntimeError("D1 resource gate windows are missing")
    passed, reasons = evaluate_resource_gate(windows)
    if not passed or gate.get("failure_reasons") != []:
        raise RuntimeError(f"D1 resource gate semantic replay failed: {reasons}")
    started = datetime.fromisoformat(str(gate.get("started_at_utc")))
    finished = datetime.fromisoformat(str(gate.get("finished_at_utc")))
    if started.tzinfo is None or finished.tzinfo is None or finished < started:
        raise RuntimeError("D1 resource gate wall-clock interval is invalid")
    current = datetime.now(timezone.utc) if now is None else now
    if current.tzinfo is None:
        raise RuntimeError("D1 resource gate validation requires timezone-aware time")
    if finished > current + timedelta(seconds=GATE_CLOCK_TOLERANCE_SECONDS):
        raise RuntimeError("D1 resource gate finish time is in the future")
    elapsed = float(gate.get("monotonic_elapsed_seconds", -1.0))
    minimum_elapsed = float(resource_thresholds()["window_count"]) * float(
        resource_thresholds()["window_duration_seconds"]
    )
    if elapsed < minimum_elapsed:
        raise RuntimeError("D1 resource gate monotonic interval is incomplete")
    if require_fresh:
        if current > finished + timedelta(seconds=AUTHORIZATION_START_WINDOW_SECONDS):
            raise RuntimeError("D1 resource gate is stale; run a fresh gate")
    return gate


def validate_candidate_resource_gate(
    run_dir: Path, *, require_fresh: bool = True
) -> dict[str, Any]:
    """Validate the latest PASS gate for P2 without introducing a P9 dependency."""

    from .provenance import load_source_closure

    root = run_dir.expanduser().resolve()
    closure_path, _closure = load_source_closure(root)
    policy_path, _policy = load_candidate_resource_policy(root)
    pointer = load_content_manifest(
        root / "00_audit" / "RESOURCE_AUDIT.json",
        name="D1 resource gate pointer",
        statuses=("PASS",),
    )
    gate_path = verified_artifact_path(
        pointer.get("latest_gate", {}), name="D1 current candidate resource gate"
    )
    return validate_resource_gate(
        gate_path,
        run_dir=root,
        policy_path=policy_path,
        expected_scope=CANDIDATE_RESOURCE_SCOPE,
        prerequisite=artifact_record(closure_path),
        require_fresh=require_fresh,
    )
