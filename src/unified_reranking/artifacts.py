"""Fail-closed helpers for content-addressed experiment artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def verified_artifact_path(record: Mapping[str, Any], *, name: str) -> Path:
    """Resolve one ``{path, sha256}`` record and verify its current bytes."""

    path_value = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{name} is not a path/SHA-256 artifact record")
    path = Path(path_value).resolve()
    try:
        observed = sha256_file(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"{name} is missing or not a regular file: {path}"
        ) from error
    if observed != expected:
        raise RuntimeError(f"{name} SHA-256 mismatch: {path}")
    return path


def load_verified_json(
    manifest_path: str | Path,
    *,
    name: str,
    statuses: tuple[str, ...] = ("COMPLETE",),
) -> dict[str, Any]:
    """Load a JSON manifest only after its status contract is satisfied."""

    path = Path(manifest_path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is unreadable: {path}") from error
    if not isinstance(value, dict) or value.get("status") not in statuses:
        raise RuntimeError(f"{name} status is not one of {statuses}: {path}")
    return value


def verified_manifest_artifact(
    manifest: Mapping[str, Any],
    *,
    key: str = "artifact",
    name: str,
) -> Path:
    """Verify a named artifact from either singular or plural manifest storage."""

    record: Any
    if key in manifest:
        record = manifest[key]
    else:
        artifacts = manifest.get("artifacts")
        record = artifacts.get(key) if isinstance(artifacts, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} record is absent from its manifest")
    return verified_artifact_path(record, name=name)


def verify_artifact_records_recursive(
    value: Any,
    *,
    name: str,
    require_at_least_one: bool = False,
) -> list[dict[str, str]]:
    """Verify every nested ``{path, sha256}`` record in an artifact tree.

    Manifests in this project store source/output records in dictionaries and
    lists at several depths.  Resume checks must validate every such child,
    rather than merely trusting the parent manifest bytes.  Mappings with only
    one of ``path``/``sha256`` are rejected because they are malformed records.
    """

    verified: list[dict[str, str]] = []

    def visit(node: Any, location: str) -> None:
        if isinstance(node, Mapping):
            has_path = "path" in node
            has_sha = "sha256" in node
            if has_path != has_sha:
                raise ValueError(f"{location} has an incomplete artifact record")
            if has_path:
                path = verified_artifact_path(node, name=location)
                verified.append(
                    {"name": location, "path": str(path), "sha256": str(node["sha256"])}
                )
                return
            for key, child in node.items():
                visit(child, f"{location}.{key}")
        elif isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")

    visit(value, name)
    if require_at_least_one and not verified:
        raise ValueError(f"{name} contains no path/SHA-256 artifact records")
    return verified
