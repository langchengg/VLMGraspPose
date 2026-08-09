from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import code_fingerprint, write_sha_sidecar
from .schema import atomic_write_json, canonical_json, sha256_bytes, sha256_file
from .test_access_guard import verify_manifest_sidecar


SCHEMA_VERSION = "3.0.0"
LOCKED_MANIFEST_RESERVED_FIELDS = frozenset(
    {"schema_version", "kind", "status", "locked_at", "content_sha256"}
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load_json_object(path: str | Path, *, description: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description} JSON: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {source}")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    observed = str(value)
    if len(observed) != 64 or any(character not in "0123456789abcdef" for character in observed):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return observed


def _verify_artifact_identity(identity: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{field} must be an artifact identity object")
    if "path" not in identity or "sha256" not in identity:
        raise ValueError(f"{field} requires path and sha256")
    path = Path(str(identity["path"])).expanduser().resolve()
    expected = _require_sha256(identity["sha256"], field=f"{field}.sha256")
    if not path.is_file():
        raise FileNotFoundError(f"{field} is not a regular file: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{field} changed: {path}")
    size = int(path.stat().st_size)
    if "size_bytes" in identity and int(identity["size_bytes"]) != size:
        raise ValueError(f"{field} size changed: {path}")
    return {"path": str(path), "sha256": expected, "size_bytes": size}


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Create one immutable JSON file without a check-then-replace race."""
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return path


def _validate_stage(stage: str) -> str:
    value = str(stage).strip().lower()
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in value):
        raise ValueError("stage must contain only lowercase letters, digits, '_' or '-'")
    return value


def _validate_claim(
    claim_path: Path,
    *,
    stage: str,
    manifest_path: Path,
    manifest_file_sha256: str,
    manifest_content_sha256: str,
) -> dict[str, Any]:
    verify_manifest_sidecar(claim_path)
    claim = _load_json_object(claim_path, description=f"{stage} claim")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{stage}_run_claim",
        "status": "claimed",
        "stage": stage,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_content_sha256": manifest_content_sha256,
    }
    for field, value in expected.items():
        if claim.get(field) != value:
            raise ValueError(f"{stage} claim {field} mismatch")
    token = claim.get("claim_token")
    if not isinstance(token, str) or len(token) != 64:
        raise ValueError(f"{stage} claim token is invalid")
    _require_sha256(token, field=f"{stage}.claim_token")
    if not isinstance(claim.get("claimed_at"), str) or not claim["claimed_at"]:
        raise ValueError(f"{stage} claim timestamp is invalid")
    if not isinstance(claim.get("pid"), int) or claim["pid"] <= 0:
        raise ValueError(f"{stage} claim PID is invalid")
    return claim


def write_locked_manifest(
    path: str | Path, payload: dict[str, Any], *, kind: str
) -> dict[str, Any]:
    output = Path(path)
    if not isinstance(payload, dict):
        raise TypeError("locked manifest payload must be a dictionary")
    conflicting = LOCKED_MANIFEST_RESERVED_FIELDS & set(payload)
    if conflicting:
        raise ValueError(f"locked manifest payload overrides reserved fields: {sorted(conflicting)}")
    manifest_kind = str(kind).strip()
    if not manifest_kind:
        raise ValueError("locked manifest kind must be non-empty")
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"immutable manifest already exists: {output}")
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": manifest_kind,
        "status": "locked",
        "locked_at": _timestamp(),
        **payload,
    }
    value["content_sha256"] = sha256_bytes(canonical_json(value).encode())
    atomic_write_json(output, value)
    write_sha_sidecar(output)
    return value


def verify_locked_manifest(
    path: str | Path,
    *,
    verify_code: bool = True,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    manifest = Path(path).expanduser().resolve()
    verify_manifest_sidecar(manifest)
    value = _load_json_object(manifest, description="locked manifest")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("locked manifest schema version mismatch")
    kind = value.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("locked manifest kind is invalid")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"locked manifest kind mismatch: {kind!r} != {expected_kind!r}")
    if value.get("status") != "locked":
        raise ValueError("locked manifest status is not locked")
    if not isinstance(value.get("locked_at"), str) or not value["locked_at"]:
        raise ValueError("locked manifest timestamp is invalid")
    expected = _require_sha256(value.get("content_sha256"), field="content_sha256")
    unsigned = {field: item for field, item in value.items() if field != "content_sha256"}
    observed = sha256_bytes(canonical_json(unsigned).encode())
    if observed != expected:
        raise ValueError("locked manifest internal content hash mismatch")
    if verify_code:
        fingerprint = value.get("code_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint != code_fingerprint():
            raise ValueError("V3 code changed after manifest lock")
    identities = value.get("locked_artifacts", [])
    if not isinstance(identities, list):
        raise ValueError("locked_artifacts must be a list")
    for index, identity in enumerate(identities):
        _verify_artifact_identity(identity, field=f"locked_artifacts[{index}]")
    return value


def claim_stage_once(
    stage_dir: str | Path,
    *,
    stage: str,
    manifest_path: str | Path,
    resume: bool = False,
) -> Path:
    stage_name = _validate_stage(stage)
    root = Path(stage_dir)
    claim_path = root / f"{stage_name.upper()}_RUN_CLAIM.json"
    complete_path = root / f"{stage_name.upper()}_RUN_COMPLETE.json"
    manifest = Path(manifest_path).expanduser().resolve()

    manifest_sha_before = sha256_file(manifest)
    locked = verify_locked_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    if manifest_sha_before != manifest_sha:
        raise ValueError("locked manifest changed while it was being verified")
    manifest_content_sha = str(locked["content_sha256"])

    if complete_path.exists():
        raise FileExistsError(f"{stage_name} has already completed")
    if claim_path.exists():
        if not resume:
            raise FileExistsError(f"{stage_name} has already been claimed")
        _validate_claim(
            claim_path,
            stage=stage_name,
            manifest_path=manifest,
            manifest_file_sha256=manifest_sha,
            manifest_content_sha256=manifest_content_sha,
        )
        return claim_path

    claim = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{stage_name}_run_claim",
        "status": "claimed",
        "stage": stage_name,
        "claimed_at": _timestamp(),
        "pid": os.getpid(),
        "claim_token": secrets.token_hex(32),
        "manifest_path": str(manifest),
        "manifest_file_sha256": manifest_sha,
        "manifest_content_sha256": manifest_content_sha,
    }
    try:
        _exclusive_write_json(claim_path, claim)
        write_sha_sidecar(claim_path)
        return claim_path
    except FileExistsError:
        if not resume:
            raise
        _validate_claim(
            claim_path,
            stage=stage_name,
            manifest_path=manifest,
            manifest_file_sha256=manifest_sha,
            manifest_content_sha256=manifest_content_sha,
        )
        return claim_path


def complete_stage_once(
    stage_dir: str | Path,
    *,
    stage: str,
    result_artifacts: Sequence[Mapping[str, Any]],
) -> Path:
    stage_name = _validate_stage(stage)
    artifacts = list(result_artifacts)
    if not artifacts:
        raise ValueError(f"{stage_name} completion requires at least one result artifact")
    root = Path(stage_dir)
    claim_path = root / f"{stage_name.upper()}_RUN_CLAIM.json"
    complete_path = root / f"{stage_name.upper()}_RUN_COMPLETE.json"
    if not claim_path.is_file():
        raise FileNotFoundError(f"{stage_name} was not claimed")
    if complete_path.exists() or complete_path.with_suffix(complete_path.suffix + ".sha256").exists():
        raise FileExistsError(f"{stage_name} already complete")

    raw_claim = _load_json_object(claim_path, description=f"{stage_name} claim")
    manifest_path = Path(str(raw_claim.get("manifest_path", ""))).expanduser().resolve()
    manifest_sha_before = sha256_file(manifest_path)
    locked = verify_locked_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha_before != manifest_sha:
        raise ValueError("locked manifest changed while it was being verified")
    claim = _validate_claim(
        claim_path,
        stage=stage_name,
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_sha,
        manifest_content_sha256=str(locked["content_sha256"]),
    )
    verified_results = [
        _verify_artifact_identity(identity, field=f"result_artifacts[{index}]")
        for index, identity in enumerate(artifacts)
    ]
    completion: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"{stage_name}_run_complete",
        "status": "complete",
        "stage": stage_name,
        "completed_at": _timestamp(),
        "claim_path": str(claim_path.resolve()),
        "claim_sha256": sha256_file(claim_path),
        "claim_token": claim["claim_token"],
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_sha,
        "manifest_content_sha256": locked["content_sha256"],
        "result_artifacts": verified_results,
    }
    completion["content_sha256"] = sha256_bytes(canonical_json(completion).encode())
    _exclusive_write_json(complete_path, completion)
    write_sha_sidecar(complete_path)
    return complete_path


def assert_lockcheck_complete(path: str | Path) -> dict[str, Any]:
    completion_path = Path(path).expanduser().resolve()
    verify_manifest_sidecar(completion_path)
    completion = _load_json_object(completion_path, description="lockcheck completion")
    expected_completion = {
        "schema_version": SCHEMA_VERSION,
        "kind": "lockcheck_run_complete",
        "status": "complete",
        "stage": "lockcheck",
    }
    for field, value in expected_completion.items():
        if completion.get(field) != value:
            raise ValueError(f"lockcheck completion {field} mismatch")
    if not isinstance(completion.get("completed_at"), str) or not completion["completed_at"]:
        raise ValueError("lockcheck completion timestamp is invalid")
    expected_content = _require_sha256(
        completion.get("content_sha256"), field="lockcheck completion content_sha256"
    )
    unsigned = {
        field: value for field, value in completion.items() if field != "content_sha256"
    }
    if sha256_bytes(canonical_json(unsigned).encode()) != expected_content:
        raise ValueError("lockcheck completion internal content hash mismatch")

    claim_path = Path(str(completion.get("claim_path", ""))).expanduser().resolve()
    expected_claim_path = completion_path.parent / "LOCKCHECK_RUN_CLAIM.json"
    if claim_path != expected_claim_path:
        raise ValueError("lockcheck completion claim path mismatch")
    if not claim_path.is_file():
        raise FileNotFoundError("lockcheck completion claim is missing")
    if sha256_file(claim_path) != _require_sha256(
        completion.get("claim_sha256"), field="lockcheck completion claim_sha256"
    ):
        raise ValueError("lockcheck claim changed after completion")

    manifest_path = Path(str(completion.get("manifest_path", ""))).expanduser().resolve()
    manifest_sha_before = sha256_file(manifest_path)
    locked = verify_locked_manifest(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha_before != manifest_sha:
        raise ValueError("locked manifest changed while it was being verified")
    expected_manifest_sha = _require_sha256(
        completion.get("manifest_file_sha256"),
        field="lockcheck completion manifest_file_sha256",
    )
    if manifest_sha != expected_manifest_sha:
        raise ValueError("lockcheck completion manifest SHA mismatch")
    if completion.get("manifest_content_sha256") != locked["content_sha256"]:
        raise ValueError("lockcheck completion manifest content SHA mismatch")

    claim = _validate_claim(
        claim_path,
        stage="lockcheck",
        manifest_path=manifest_path,
        manifest_file_sha256=manifest_sha,
        manifest_content_sha256=str(locked["content_sha256"]),
    )
    if completion.get("claim_token") != claim["claim_token"]:
        raise ValueError("lockcheck completion claim token mismatch")

    identities = completion.get("result_artifacts")
    if not isinstance(identities, list) or not identities:
        raise ValueError("lockcheck completion requires result artifacts")
    for index, identity in enumerate(identities):
        _verify_artifact_identity(identity, field=f"result_artifacts[{index}]")
    return completion
