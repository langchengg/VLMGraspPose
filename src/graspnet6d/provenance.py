"""Immutable run and stage provenance without importing experiment dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import repository_root, sha256_file


RUN_MANIFEST_SCHEMA_VERSION = 2
IMMUTABLE_IDENTITY_SCHEMA_VERSION = 1

_PROFILE_CONFIGS = {
    "smoke": "smoke.yaml",
    "paper-lite": "paper_lite.yaml",
    "paper-lite-train3": "paper_lite_train3.yaml",
    "paper-extended": "paper_extended.yaml",
}
_LOCKED_CONFIG_PATHS = (
    "configs/graspnet6d/feature_schema_6d_v1.json",
    "configs/graspnet6d/features.yaml",
    "configs/graspnet6d/ranker.yaml",
    "configs/graspnet6d/language_templates.yaml",
    "configs/graspnet6d/graspnet_object_catalog_v1.json",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def dirty_tree_hash() -> str:
    root = repository_root()
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=root, capture_output=True, check=True
    ).stdout
    digest.update(diff)
    untracked = _git(["ls-files", "--others", "--exclude-standard"]).splitlines()
    for relative in sorted(untracked):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_graspnet6d_vgn_lambdamart"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolved_profile_config(root: Path, profile: str) -> tuple[dict[str, Any], str]:
    """Return the canonical resolved config and its exact YAML representation."""

    try:
        filename = _PROFILE_CONFIGS[profile]
    except KeyError as error:
        raise ValueError(f"unknown run profile {profile!r}") from error
    try:
        import yaml

        path = root / "configs" / "graspnet6d" / filename
        overlay = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(overlay, dict):
            raise TypeError("profile document is not a mapping")
        extends = overlay.get("extends")
        if extends:
            base_path = (root / str(extends)).resolve()
            try:
                base_path.relative_to(root.resolve())
            except ValueError as error:
                raise ValueError(
                    f"profile extends path escapes the repository: {extends!r}"
                ) from error
            base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            if not isinstance(base, dict) or base.get("extends"):
                raise ValueError(
                    "only one explicit config inheritance level is supported"
                )
            overlay = _deep_merge(base, overlay)
        if overlay.get("profile") != profile:
            raise ValueError(
                f"resolved profile is {overlay.get('profile')!r}, expected {profile!r}"
            )
        serialised = yaml.safe_dump(overlay, sort_keys=True)
    except Exception as error:
        raise RuntimeError(
            f"cannot resolve the required {profile!r} profile: {error}"
        ) from error
    return overlay, serialised


def resolve_profile_config(
    profile: str, *, root: Path | None = None
) -> tuple[dict[str, Any], str]:
    """Resolve one locked profile for all CLI/workflow consumers.

    Keeping this public boundary prevents download, split, provenance, and
    reporting code from silently using different profile interpretations.
    """

    return _resolved_profile_config((root or repository_root()).resolve(), profile)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_file_hash(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"immutable run input is missing or not a regular file: {path}"
        )
    return sha256_file(path)


def _checkpoint_record(
    root: Path, config: dict[str, Any], section: str, key: str
) -> dict[str, Any]:
    value = config.get(section, {}).get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"resolved config is missing {section}.{key}")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"checkpoint path escapes the repository: {value!r}"
        ) from error
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"required checkpoint is missing or not a regular file: {path}"
        )
    return {"path": value, "sha256": sha256_file(path)}


def _load_upstream_lock(root: Path) -> tuple[str, dict[str, str]]:
    lock = root / "configs" / "graspnet6d" / "upstream_versions.lock"
    try:
        import yaml

        locked = yaml.safe_load(lock.read_text(encoding="utf-8"))
        if not isinstance(locked, dict) or not isinstance(locked.get("sources"), dict):
            raise TypeError("lock document does not contain a sources mapping")
        upstream_shas = {
            name: source["commit"]
            for name, source in locked["sources"].items()
            if isinstance(source, dict) and isinstance(source.get("commit"), str)
        }
        if not upstream_shas:
            raise ValueError("lock document contains no upstream commits")
    except Exception as error:
        raise RuntimeError(
            f"cannot parse the required upstream lock {lock}: {error}"
        ) from error
    return sha256_file(lock), upstream_shas


def _current_immutable_identity(root: Path, profile: str) -> tuple[dict[str, Any], str]:
    resolved_config, resolved_yaml = _resolved_profile_config(root, profile)
    upstream_lock_hash, upstream_shas = _load_upstream_lock(root)
    config_hashes = {
        relative: _required_file_hash(root, relative)
        for relative in _LOCKED_CONFIG_PATHS
    }
    checkpoints = {
        "vgn": _checkpoint_record(root, resolved_config, "vgn", "checkpoint"),
        "hifics": _checkpoint_record(
            root, resolved_config, "grounding", "hifics_checkpoint"
        ),
    }
    repository_git_sha = _git(["rev-parse", "HEAD"])
    repository_dirty = bool(_git(["status", "--porcelain"]))
    identity = {
        "schema_version": IMMUTABLE_IDENTITY_SCHEMA_VERSION,
        "profile": profile,
        "resolved_config": resolved_config,
        "resolved_config_sha256": _sha256_bytes(resolved_yaml.encode("utf-8")),
        "repository": {
            "git_sha": repository_git_sha,
            "dirty": repository_dirty,
            "dirty_working_tree_sha256": dirty_tree_hash(),
        },
        "upstream": {
            "lock_sha256": upstream_lock_hash,
            "commits": upstream_shas,
        },
        "config_file_hashes": config_hashes,
        "checkpoint_hashes": checkpoints,
    }
    return identity, resolved_yaml


def _identity_mismatches(
    expected: Any, observed: Any, prefix: str = "identity"
) -> list[str]:
    """Return precise paths without dumping potentially large config values."""

    if isinstance(expected, dict) and isinstance(observed, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            child = f"{prefix}.{key}"
            if key not in expected or key not in observed:
                differences.append(child)
            else:
                differences.extend(
                    _identity_mismatches(expected[key], observed[key], child)
                )
        return differences
    if expected != observed:
        return [prefix]
    return []


def _top_level_identity_fields(identity: dict[str, Any]) -> dict[str, Any]:
    repository = identity["repository"]
    upstream = identity["upstream"]
    checkpoints = identity["checkpoint_hashes"]
    config_hashes = identity["config_file_hashes"]
    return {
        "repository_git_sha": repository["git_sha"],
        "repository_dirty": repository["dirty"],
        "dirty_working_tree_hash": repository["dirty_working_tree_sha256"],
        "upstream_lock_sha256": upstream["lock_sha256"],
        "upstream_shas": upstream["commits"],
        "resolved_config_sha256": identity["resolved_config_sha256"],
        "feature_schema_hash": config_hashes[
            "configs/graspnet6d/feature_schema_6d_v1.json"
        ],
        "features_config_hash": config_hashes["configs/graspnet6d/features.yaml"],
        "ranker_config_hash": config_hashes["configs/graspnet6d/ranker.yaml"],
        "language_templates_hash": config_hashes[
            "configs/graspnet6d/language_templates.yaml"
        ],
        "object_catalog_hash": config_hashes[
            "configs/graspnet6d/graspnet_object_catalog_v1.json"
        ],
        "vgn_checkpoint_hash": checkpoints["vgn"]["sha256"],
        "hifics_checkpoint_hash": checkpoints["hifics"]["sha256"],
    }


def _validate_resolved_config_snapshot(
    path: Path, identity: dict[str, Any], resolved_yaml: str
) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"immutable resolved config snapshot is missing or invalid: {path}"
        )
    raw = path.read_bytes()
    actual_hash = _sha256_bytes(raw)
    expected_hash = identity["resolved_config_sha256"]
    if actual_hash != expected_hash or raw != resolved_yaml.encode("utf-8"):
        raise ValueError(
            "immutable resolved config snapshot differs from the run manifest/current profile"
        )


def create_run(run_id: str, *, profile: str, command: list[str]) -> Path:
    root = repository_root()
    run_dir = root / "artifacts" / "graspnet6d" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    identity, resolved_yaml = _current_immutable_identity(root, profile)
    identity_fields = _top_level_identity_fields(identity)
    resolved_path = run_dir / "resolved_config.yaml"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("profile") != profile:
            raise ValueError(
                f"run {run_id!r} was created for profile {payload.get('profile')!r}, not {profile!r}"
            )
        if payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"run {run_id!r} has legacy/unknown manifest schema "
                f"{payload.get('schema_version')!r}; immutable resume cannot be proven"
            )
        stored_identity = payload.get("immutable_identity")
        differences = _identity_mismatches(stored_identity, identity)
        for field, current_value in identity_fields.items():
            if payload.get(field) != current_value:
                differences.append(f"manifest.{field}")
        if differences:
            changed = ", ".join(sorted(set(differences)))
            raise ValueError(
                f"run {run_id!r} immutable identity changed; refusing resume: {changed}"
            )
        _validate_resolved_config_snapshot(resolved_path, identity, resolved_yaml)
        commands = payload.setdefault("commands_executed", [])
        if command not in commands:
            commands.append(command)
        resumed_at = datetime.now(timezone.utc).isoformat()
        previous_status = str(payload.get("status", "IN_PROGRESS"))
        if previous_status in {"BLOCKED", "FAILED", "COMPLETE"}:
            history = payload.setdefault("status_history", [])
            if not isinstance(history, list):
                raise ValueError(
                    f"run {run_id!r} has an invalid status_history; refusing resume"
                )
            history.append(
                {
                    "status": previous_status,
                    "blocked_stage": payload.get("blocked_stage"),
                    "blocked_reason": payload.get("blocked_reason"),
                    "failed_stage": payload.get("failed_stage"),
                    "failure_type": payload.get("failure_type"),
                    "failure_message": payload.get("failure_message"),
                    "ended_at_utc": payload.get("ended_at_utc"),
                    "resumed_at_utc": resumed_at,
                }
            )
            payload["status"] = "IN_PROGRESS"
            payload["formal_results_emitted"] = False
            payload.pop("blocked_stage", None)
            payload.pop("blocked_reason", None)
            payload.pop("failed_stage", None)
            payload.pop("failure_type", None)
            payload.pop("failure_message", None)
            payload.pop("ended_at_utc", None)
        payload["last_resumed_at_utc"] = resumed_at
        atomic_json(manifest_path, payload)
        return run_dir
    audit_environment = root / "artifacts/graspnet6d/audit/hardware_environment.json"
    hardware = (
        json.loads(audit_environment.read_text(encoding="utf-8"))
        if audit_environment.is_file()
        else None
    )
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "profile": profile,
        "status": "IN_PROGRESS",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_identity": identity,
        **identity_fields,
        "commands_executed": [command],
        "random_seeds": [20260815, 20260816, 20260817],
        "dataset_archive_hashes": {},
        "dataset_manifest_hash": None,
        "split_hash": None,
        "candidate_cache_hash": None,
        "hardware_environment": hardware,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "sample_counts": {
            "scenes": 0,
            "frames": 0,
            "target_groups": 0,
            "excluded_groups": 0,
        },
        "formal_results_emitted": False,
    }
    lineage = identity["resolved_config"].get("lineage", {})
    parent_run_id = lineage.get("parent_run_id") if isinstance(lineage, dict) else None
    if parent_run_id is not None:
        if not isinstance(parent_run_id, str) or not parent_run_id.strip():
            raise RuntimeError("lineage.parent_run_id must be a non-empty string")
        if parent_run_id == run_id:
            raise RuntimeError("a run cannot name itself as parent_run_id")
        parent_manifest = (
            root / "artifacts" / "graspnet6d" / parent_run_id / "run_manifest.json"
        )
        if not parent_manifest.is_file() or parent_manifest.is_symlink():
            raise RuntimeError(
                f"configured parent run manifest is missing: {parent_manifest}"
            )
        payload["parent_run_id"] = parent_run_id
        payload["parent_run_manifest_sha256"] = sha256_file(parent_manifest)
        parent_parent_run_id = lineage.get("parent_parent_run_id")
        if parent_parent_run_id is not None:
            if not isinstance(parent_parent_run_id, str) or not parent_parent_run_id.strip():
                raise RuntimeError(
                    "lineage.parent_parent_run_id must be a non-empty string"
                )
            parent_payload = json.loads(parent_manifest.read_text(encoding="utf-8"))
            if parent_payload.get("parent_run_id") != parent_parent_run_id:
                raise RuntimeError(
                    "lineage.parent_parent_run_id does not match the configured "
                    "parent run manifest"
                )
            parent_parent_manifest = (
                root
                / "artifacts"
                / "graspnet6d"
                / parent_parent_run_id
                / "run_manifest.json"
            )
            if not parent_parent_manifest.is_file() or parent_parent_manifest.is_symlink():
                raise RuntimeError(
                    "configured parent-parent run manifest is missing: "
                    f"{parent_parent_manifest}"
                )
            payload["parent_parent_run_id"] = parent_parent_run_id
            payload["parent_parent_run_manifest_sha256"] = sha256_file(
                parent_parent_manifest
            )
        change_reason = lineage.get("change_reason")
        if change_reason is not None:
            if not isinstance(change_reason, str) or not change_reason.strip():
                raise RuntimeError("lineage.change_reason must be a non-empty string")
            payload["change_reason"] = change_reason.strip()
    resolved = identity["resolved_config"]
    experiment_scope = resolved.get("experiment_scope", {})
    dataset = resolved.get("dataset", {})
    payload.update(
        {
            "source_archive_identity": list(
                dataset.get("required_scene_archives", [])
            ),
            "camera": str(dataset.get("camera", resolved.get("camera", ""))),
            "fixed_seed": int(resolved["seed"]),
            "official_test_benchmark": bool(
                experiment_scope.get("official_graspnet_test_benchmark", False)
            ),
            "experiment_scope_label": experiment_scope.get("scope_label"),
        }
    )
    atomic_text(resolved_path, resolved_yaml)
    atomic_json(manifest_path, payload)
    atomic_text(root / "artifacts/graspnet6d/ACTIVE_RUN_ID", run_id + "\n")
    return run_dir


def update_manifest(run_dir: Path, **changes: Any) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    atomic_json(path, payload)
    return payload


_STAGE_OPERATIONAL_FIELDS = frozenset(
    {
        "recorded_at_utc",
        "resume_requested",
        "resumed",
        "resumed_groups",
        "completed_groups",
        "executed_groups",
    }
)


def _stable_stage_evidence(value: Any) -> Any:
    """Remove per-attempt bookkeeping from a stage-record comparison."""

    if isinstance(value, dict):
        return {
            key: _stable_stage_evidence(item)
            for key, item in value.items()
            if key not in _STAGE_OPERATIONAL_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_stage_evidence(item) for item in value]
    return value


def record_stage(run_dir: Path, stage: str, status: str, **details: Any) -> None:
    payload = {
        "stage": stage,
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    path = run_dir / "stages" / f"{stage}.json"
    if status == "COMPLETE" and path.is_file() and not path.is_symlink():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("stage") == stage
            and existing.get("status") == "COMPLETE"
            and _stable_stage_evidence(existing) == _stable_stage_evidence(payload)
        ):
            return
    atomic_json(path, payload)


__all__ = [
    "IMMUTABLE_IDENTITY_SCHEMA_VERSION",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "atomic_json",
    "atomic_text",
    "create_run",
    "dirty_tree_hash",
    "new_run_id",
    "record_stage",
    "update_manifest",
]
