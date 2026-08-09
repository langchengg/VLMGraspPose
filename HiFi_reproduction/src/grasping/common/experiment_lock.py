"""Immutable, content-addressed experiment lock for formal 4-DoF evaluation.

The lock is deliberately independent from model implementations.  A candidate
configuration names the frozen protocol and the manifest files already present
inside a run directory; this module validates and hashes those inputs, records
the repository state, and can create the effective lock exactly once.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_SCHEMA_VERSION = 1
DEFAULT_LOCK_RELATIVE_PATH = Path("manifests/experiment_lock.json")
DEFAULT_MARKER_NAME = ".EXPERIMENT_LOCKED"
DEFAULT_FROZEN_ALIAS_NAME = "frozen_4dof_backends_experiment_manifest.json"

REQUIRED_ARTIFACT_ROLES = frozenset(
    {"source", "data", "third_party", "model", "selected_config", "evaluator"}
)
REQUIRED_LINEAGE_KEYS = (
    "repeated_film",
    "splits",
    "prediction_manifest",
    "reference",
    "vendors",
)
REQUIRED_PROTOCOL_KEYS = (
    "preprocess",
    "input",
    "device",
    "conditioning",
    "crop",
    "gate",
    "width",
    "angle",
    "fixed_grasp_height_px",
    "peak",
    "nms",
    "analytic",
    "evaluator",
    "primary_method",
    "seed",
    "expected_test_sample_count",
)


class ExperimentLockError(ValueError):
    """Base class for an invalid lock candidate or effective lock."""


class LockDriftError(ExperimentLockError):
    """Raised when a locked input no longer matches its recorded digest."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ExperimentLockError(f"{name} must be a non-empty mapping")
    return value


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ExperimentLockError(f"{name} is missing required fields: {missing}")


def _require_sha256(value: Any, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ExperimentLockError(f"{name} must be a 64-character SHA-256 digest")
    return digest


def _resolve_contained(path: Path, root: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ExperimentLockError(f"{name} must be contained in {root}: {resolved}") from exc
    return resolved


def _resolve_run_path(raw: Any, run_dir: Path, name: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = run_dir / path
    resolved = _resolve_contained(path, run_dir, name)
    if not resolved.is_file():
        raise ExperimentLockError(f"{name} does not exist or is not a file: {resolved}")
    return resolved


def _resolve_source_path(raw: Any, repository_root: Path, name: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = repository_root / path
    resolved = _resolve_contained(path, repository_root, name)
    if not resolved.is_file():
        raise ExperimentLockError(f"{name} does not exist or is not a file: {resolved}")
    return resolved


def _git_output(repository_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise ExperimentLockError(
            f"failed to inspect git repository {repository_root}: {str(detail).strip()}"
        ) from exc
    return completed.stdout


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = _git_output(repository_root, "rev-parse", "HEAD").decode().strip()
    diff = _git_output(
        repository_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
    )
    return {
        "root": str(repository_root),
        "git_commit": commit,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_diff_bytes": len(diff),
        "git_dirty": bool(diff),
    }


def _artifact_records(
    artifacts: Mapping[str, Any], run_dir: Path
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    roles: set[str] = set()
    for logical_name, raw in artifacts.items():
        item = _as_mapping(raw, f"artifacts.{logical_name}")
        _require_keys(item, ("role", "path"), f"artifacts.{logical_name}")
        role = str(item["role"])
        if role not in REQUIRED_ARTIFACT_ROLES:
            raise ExperimentLockError(
                f"artifacts.{logical_name}.role is unsupported: {role!r}"
            )
        path = _resolve_run_path(
            item["path"], run_dir, f"artifacts.{logical_name}.path"
        )
        roles.add(role)
        records[str(logical_name)] = {
            "role": role,
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    missing_roles = sorted(REQUIRED_ARTIFACT_ROLES - roles)
    if missing_roles:
        raise ExperimentLockError(
            "artifacts must include source/data/third_party/model/selected_config/"
            f"evaluator manifests; missing roles: {missing_roles}"
        )
    return records


def _artifact_reference(
    artifacts: Mapping[str, Mapping[str, Any]],
    raw_name: Any,
    field: str,
    *,
    role: str | None = None,
) -> str:
    name = str(raw_name)
    if name not in artifacts:
        raise ExperimentLockError(f"{field} references unknown artifact {name!r}")
    actual_role = str(artifacts[name]["role"])
    if role is not None and actual_role != role:
        raise ExperimentLockError(
            f"{field} must reference a {role!r} artifact, got {actual_role!r}"
        )
    return name


def _validate_lineage(
    lineage: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> None:
    _require_keys(lineage, REQUIRED_LINEAGE_KEYS, "lineage")

    repeated_film = _as_mapping(lineage["repeated_film"], "lineage.repeated_film")
    _require_keys(
        repeated_film,
        ("source_manifest_artifact", "checkpoint_sha256"),
        "lineage.repeated_film",
    )
    _artifact_reference(
        artifacts,
        repeated_film["source_manifest_artifact"],
        "lineage.repeated_film.source_manifest_artifact",
        role="source",
    )
    _require_sha256(
        repeated_film["checkpoint_sha256"],
        "lineage.repeated_film.checkpoint_sha256",
    )

    splits = _as_mapping(lineage["splits"], "lineage.splits")
    _require_keys(splits, ("train", "validation", "test"), "lineage.splits")
    for split in ("train", "validation", "test"):
        item = _as_mapping(splits[split], f"lineage.splits.{split}")
        _require_keys(item, ("manifest_artifact", "sample_count"), f"lineage.splits.{split}")
        _artifact_reference(
            artifacts,
            item["manifest_artifact"],
            f"lineage.splits.{split}.manifest_artifact",
            role="data",
        )
        if not isinstance(item["sample_count"], int) or isinstance(
            item["sample_count"], bool
        ) or item["sample_count"] <= 0:
            raise ExperimentLockError(
                f"lineage.splits.{split}.sample_count must be a positive integer"
            )

    prediction = _as_mapping(
        lineage["prediction_manifest"], "lineage.prediction_manifest"
    )
    _require_keys(prediction, ("artifact",), "lineage.prediction_manifest")
    _artifact_reference(
        artifacts,
        prediction["artifact"],
        "lineage.prediction_manifest.artifact",
        role="source",
    )

    reference = _as_mapping(lineage["reference"], "lineage.reference")
    _require_keys(
        reference,
        ("manifest_artifact", "reference_run", "direct_inputs"),
        "lineage.reference",
    )
    _artifact_reference(
        artifacts,
        reference["manifest_artifact"],
        "lineage.reference.manifest_artifact",
        role="source",
    )
    if not str(reference["reference_run"]).strip():
        raise ExperimentLockError("lineage.reference.reference_run must be non-empty")
    direct_inputs = _as_mapping(
        reference["direct_inputs"], "lineage.reference.direct_inputs"
    )
    for name, digest in direct_inputs.items():
        _require_sha256(digest, f"lineage.reference.direct_inputs.{name}")

    vendors = _as_mapping(lineage["vendors"], "lineage.vendors")
    for vendor_name, raw_vendor in vendors.items():
        vendor = _as_mapping(raw_vendor, f"lineage.vendors.{vendor_name}")
        _require_keys(
            vendor,
            ("manifest_artifact", "commit", "checkpoints"),
            f"lineage.vendors.{vendor_name}",
        )
        _artifact_reference(
            artifacts,
            vendor["manifest_artifact"],
            f"lineage.vendors.{vendor_name}.manifest_artifact",
            role="third_party",
        )
        if not str(vendor["commit"]).strip():
            raise ExperimentLockError(
                f"lineage.vendors.{vendor_name}.commit must be non-empty"
            )
        checkpoints = _as_mapping(
            vendor["checkpoints"], f"lineage.vendors.{vendor_name}.checkpoints"
        )
        for checkpoint_name, digest in checkpoints.items():
            _require_sha256(
                digest,
                f"lineage.vendors.{vendor_name}.checkpoints.{checkpoint_name}",
            )


def _validate_protocol(
    protocol: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]], lineage: Mapping[str, Any]
) -> None:
    _require_keys(protocol, REQUIRED_PROTOCOL_KEYS, "protocol")
    if not isinstance(protocol["seed"], int) or isinstance(protocol["seed"], bool):
        raise ExperimentLockError("protocol.seed must be an integer")
    expected = protocol["expected_test_sample_count"]
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise ExperimentLockError(
            "protocol.expected_test_sample_count must be a positive integer"
        )
    actual = lineage["splits"]["test"]["sample_count"]
    if expected != actual:
        raise ExperimentLockError(
            "protocol.expected_test_sample_count must equal the locked test split count"
        )
    height = protocol["fixed_grasp_height_px"]
    if not isinstance(height, (int, float)) or isinstance(height, bool) or height <= 0:
        raise ExperimentLockError("protocol.fixed_grasp_height_px must be positive")
    if not str(protocol["primary_method"]).strip():
        raise ExperimentLockError("protocol.primary_method must be non-empty")
    evaluator = _as_mapping(protocol["evaluator"], "protocol.evaluator")
    _require_keys(evaluator, ("artifact",), "protocol.evaluator")
    _artifact_reference(
        artifacts,
        evaluator["artifact"],
        "protocol.evaluator.artifact",
        role="evaluator",
    )


def _validate_selection_inputs(
    selection_inputs: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> None:
    if "checkpoint" not in selection_inputs:
        raise ExperimentLockError("selection_inputs must include checkpoint selection")
    for selection_name, raw in selection_inputs.items():
        selection = _as_mapping(raw, f"selection_inputs.{selection_name}")
        _require_keys(
            selection, ("split", "artifact"), f"selection_inputs.{selection_name}"
        )
        split = str(selection["split"]).lower()
        if split != "validation":
            raise ExperimentLockError(
                f"selection_inputs.{selection_name} must use validation; "
                f"{split!r} is forbidden for model/config selection"
            )
        _artifact_reference(
            artifacts,
            selection["artifact"],
            f"selection_inputs.{selection_name}.artifact",
            role="selected_config",
        )


def build_lock_candidate(
    config: Mapping[str, Any], *, run_dir: Path, repository_root: Path
) -> dict[str, Any]:
    """Validate and build a non-effective lock candidate without writing files."""

    run = run_dir.expanduser().resolve()
    repository = repository_root.expanduser().resolve()
    if not run.is_dir():
        raise ExperimentLockError(f"run directory does not exist: {run}")
    _resolve_contained(run, repository, "run_dir")
    if not isinstance(config, Mapping):
        raise ExperimentLockError("config must be a JSON object")
    _require_keys(
        config,
        ("run_id", "artifacts", "source_files", "lineage", "protocol", "selection_inputs"),
        "config",
    )
    run_id = str(config["run_id"])
    if not run_id or run_id != run.name:
        raise ExperimentLockError(
            f"config.run_id must equal run directory name {run.name!r}"
        )

    artifact_config = _as_mapping(config["artifacts"], "artifacts")
    artifacts = _artifact_records(artifact_config, run)
    lineage = _as_mapping(config["lineage"], "lineage")
    protocol = _as_mapping(config["protocol"], "protocol")
    selection_inputs = _as_mapping(config["selection_inputs"], "selection_inputs")
    _validate_lineage(lineage, artifacts)
    _validate_protocol(protocol, artifacts, lineage)
    _validate_selection_inputs(selection_inputs, artifacts)

    raw_sources = config["source_files"]
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ExperimentLockError("source_files must be a non-empty list")
    if not raw_sources:
        raise ExperimentLockError("source_files must be a non-empty list")
    source_hashes: dict[str, str] = {}
    for index, raw_path in enumerate(raw_sources):
        source = _resolve_source_path(raw_path, repository, f"source_files[{index}]")
        relative = source.relative_to(repository).as_posix()
        if relative in source_hashes:
            raise ExperimentLockError(f"source_files contains duplicate path: {relative}")
        source_hashes[relative] = sha256_file(source)

    candidate: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "lock_status": "CANDIDATE_NOT_EFFECTIVE",
        "effective": False,
        "candidate_created_utc": _utc_now(),
        "run_id": run_id,
        "run_dir": str(run),
        "repository": _git_state(repository),
        "lineage": copy.deepcopy(dict(lineage)),
        "protocol": copy.deepcopy(dict(protocol)),
        "selection_inputs": copy.deepcopy(dict(selection_inputs)),
        "artifacts": artifacts,
        "source_hashes": source_hashes,
        "lock_relative_path": DEFAULT_LOCK_RELATIVE_PATH.as_posix(),
        "marker_relative_path": DEFAULT_MARKER_NAME,
    }
    candidate["candidate_content_sha256"] = canonical_json_sha256(candidate)
    return candidate


def _effective_manifest(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("lock_status") != "CANDIDATE_NOT_EFFECTIVE" or candidate.get(
        "effective"
    ) is not False:
        raise ExperimentLockError("only a non-effective candidate can be formalized")
    expected = candidate.get("candidate_content_sha256")
    unhashed_candidate = dict(candidate)
    unhashed_candidate.pop("candidate_content_sha256", None)
    if expected != canonical_json_sha256(unhashed_candidate):
        raise ExperimentLockError("candidate content hash mismatch")
    manifest = copy.deepcopy(unhashed_candidate)
    manifest["lock_status"] = "LOCKED"
    manifest["effective"] = True
    manifest["locked_utc"] = _utc_now()
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_lock_exclusive(
    run_dir: Path, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the effective lock and marker once, refusing every overwrite."""

    run = run_dir.expanduser().resolve()
    lock_path = run / DEFAULT_LOCK_RELATIVE_PATH
    marker_path = run / DEFAULT_MARKER_NAME
    frozen_alias_path = run / DEFAULT_FROZEN_ALIAS_NAME
    if lock_path.exists():
        raise FileExistsError(f"experiment lock already exists: {lock_path}")
    if marker_path.exists():
        raise FileExistsError(f"experiment lock marker already exists: {marker_path}")
    if frozen_alias_path.exists():
        raise FileExistsError(f"frozen experiment manifest already exists: {frozen_alias_path}")
    if str(candidate.get("run_dir")) != str(run):
        raise ExperimentLockError("candidate run_dir does not match lock destination")

    manifest = _effective_manifest(candidate)
    marker = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "lock_status": "LOCKED",
        "run_id": manifest["run_id"],
        "lock_relative_path": DEFAULT_LOCK_RELATIVE_PATH.as_posix(),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "locked_utc": manifest["locked_utc"],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_write(lock_path, manifest)
    try:
        _exclusive_write(frozen_alias_path, manifest)
        _exclusive_write(marker_path, marker)
    except BaseException:
        for path in (marker_path, frozen_alias_path, lock_path):
            if path.exists():
                path.chmod(0o644)
                path.unlink()
        raise
    _fsync_directory(lock_path.parent)
    _fsync_directory(run)
    return manifest


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockDriftError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LockDriftError(f"{name} must contain a JSON object: {path}")
    return value


def verify_lock(run_dir: Path) -> dict[str, Any]:
    """Verify the lock, marker, repository state, and every locked file hash."""

    run = run_dir.expanduser().resolve()
    lock_path = run / DEFAULT_LOCK_RELATIVE_PATH
    marker_path = run / DEFAULT_MARKER_NAME
    frozen_alias_path = run / DEFAULT_FROZEN_ALIAS_NAME
    manifest = _load_json_object(lock_path, "experiment lock")
    marker = _load_json_object(marker_path, "experiment lock marker")
    if not frozen_alias_path.is_file() or frozen_alias_path.read_bytes() != lock_path.read_bytes():
        raise LockDriftError("frozen experiment manifest differs from effective lock")

    expected_manifest_hash = manifest.pop("manifest_content_sha256", None)
    actual_manifest_hash = canonical_json_sha256(manifest)
    manifest["manifest_content_sha256"] = expected_manifest_hash
    if expected_manifest_hash != actual_manifest_hash:
        raise LockDriftError("experiment lock content hash mismatch")
    if manifest.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise LockDriftError("unsupported experiment lock schema")
    if manifest.get("lock_status") != "LOCKED" or manifest.get("effective") is not True:
        raise LockDriftError("experiment lock is not effective")
    if str(manifest.get("run_dir")) != str(run):
        raise LockDriftError("experiment lock run_dir does not match its location")
    if marker.get("lock_status") != "LOCKED" or marker.get(
        "manifest_content_sha256"
    ) != expected_manifest_hash:
        raise LockDriftError("experiment lock marker does not match the lock")
    if marker.get("run_id") != manifest.get("run_id"):
        raise LockDriftError("experiment lock marker run_id mismatch")

    for logical_name, item in manifest["artifacts"].items():
        path = _resolve_run_path(
            item["path"], run, f"locked artifact {logical_name}"
        )
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise LockDriftError(f"locked artifact changed: {logical_name} ({path})")

    repository_root = Path(manifest["repository"]["root"]).resolve()
    for relative, expected_hash in manifest["source_hashes"].items():
        path = _resolve_source_path(relative, repository_root, f"locked source {relative}")
        if sha256_file(path) != expected_hash:
            raise LockDriftError(f"locked source changed: {path}")
    current_git = _git_state(repository_root)
    for field in ("git_commit", "git_diff_sha256", "git_diff_bytes", "git_dirty"):
        if current_git[field] != manifest["repository"][field]:
            raise LockDriftError(f"locked repository state changed: {field}")
    return manifest


__all__ = [
    "DEFAULT_LOCK_RELATIVE_PATH",
    "DEFAULT_FROZEN_ALIAS_NAME",
    "DEFAULT_MARKER_NAME",
    "ExperimentLockError",
    "LOCK_SCHEMA_VERSION",
    "LockDriftError",
    "build_lock_candidate",
    "canonical_json_sha256",
    "sha256_file",
    "verify_lock",
    "write_lock_exclusive",
]
