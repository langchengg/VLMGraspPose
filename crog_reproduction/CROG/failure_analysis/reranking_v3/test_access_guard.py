from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Literal

from .schema import canonical_json, sha256_bytes, sha256_file


TRAINING_SCOPES = frozenset({"development", "calibration", "select"})
EVALUATION_SCOPES = frozenset({"lockcheck", "test"})
TEST_PATH_MARKERS = ("formal_test", "labels_test", "test_labels", "/test/")
TEST_RANKING_MARKERS = ("primary_predictions", "test_predictions", "test_ranking")


def _normalized(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve()).lower()


def classify_sensitive_path(path: str | Path) -> tuple[str, ...]:
    value = _normalized(path)
    reasons = []
    if any(marker in value for marker in TEST_PATH_MARKERS):
        reasons.append("formal_test_or_test_label")
    if any(marker in value for marker in TEST_RANKING_MARKERS):
        reasons.append("test_ranking")
    return tuple(reasons)


class TestAccessGuard:
    """Fail-closed path guard and append-only artifact-access journal."""

    def __init__(
        self,
        scope: str,
        log_path: str | Path,
        *,
        formal_manifest: str | Path | None = None,
        manifest_access_class: Literal["v3_final", "v2_frozen"] = "v3_final",
    ):
        if scope not in TRAINING_SCOPES | EVALUATION_SCOPES:
            raise ValueError(f"unknown scope: {scope}")
        self.scope = scope
        self.log_path = Path(log_path)
        self.formal_manifest = Path(formal_manifest).resolve() if formal_manifest else None
        self.manifest_access_class = manifest_access_class

    def check(self, paths: Iterable[str | Path], *, purpose: str, label_access: bool = False) -> None:
        resolved = [Path(path).expanduser().resolve() for path in paths]
        if self.scope in TRAINING_SCOPES:
            for path in resolved:
                reasons = classify_sensitive_path(path)
                if reasons:
                    raise PermissionError(
                        f"scope={self.scope} cannot read {path} ({','.join(reasons)})"
                    )
        if self.scope == "test" and self.formal_manifest is None:
            raise PermissionError("formal test access requires a frozen manifest")
        if self.scope == "test":
            verify_test_access_manifest(
                self.formal_manifest,
                access_class=self.manifest_access_class,
            )
        record = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            "scope": self.scope,
            "purpose": str(purpose),
            "label_access": bool(label_access),
            "paths": [
                {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
                for path in resolved
            ],
            "formal_manifest": str(self.formal_manifest) if self.formal_manifest else None,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = canonical_json(record) + "\n"
        # Append-only journal: one write syscall plus fsync.  The log is not an
        # immutable scientific artifact until its enclosing stage is complete.
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def verify_manifest_sidecar(manifest_path: str | Path) -> str:
    manifest = Path(manifest_path).resolve()
    sidecar = manifest.with_suffix(manifest.suffix + ".sha256")
    if not sidecar.exists():
        raise FileNotFoundError(f"missing manifest SHA sidecar: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(manifest)
    if expected != observed:
        raise ValueError("frozen manifest SHA mismatch")
    return observed


def verify_test_access_manifest(
    manifest_path: str | Path,
    *,
    access_class: Literal["v3_final", "v2_frozen"] = "v3_final",
) -> dict[str, object]:
    """Fail closed on the kind and internal hash of a test-access manifest.

    ``v2_frozen`` is a deliberately separate clearance for reproducing the
    already-frozen, label-free V2 replay/prior.  It must be requested
    explicitly; all V3 test inference/evaluation access defaults to the final
    V3 lifecycle manifest.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    verify_manifest_sidecar(manifest)
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid frozen manifest JSON: {manifest}") from error
    if not isinstance(value, dict):
        raise ValueError("frozen manifest must be a JSON object")
    if value.get("status") != "locked":
        raise ValueError("test-access manifest status is not locked")
    if access_class == "v3_final":
        if (
            value.get("schema_version") != "3.0.0"
            or value.get("kind") != "v3_final_experiment_manifest"
        ):
            raise PermissionError("V3 test access requires the final V3 manifest")
        expected = str(value.get("content_sha256", ""))
        unsigned = {
            field: item for field, item in value.items() if field != "content_sha256"
        }
        observed = sha256_bytes(canonical_json(unsigned).encode())
        if expected != observed:
            raise ValueError("final V3 manifest internal content hash mismatch")
    elif access_class == "v2_frozen":
        if (
            value.get("schema_version") != "2.0.0"
            or value.get("kind") != "frozen_experiment_manifest"
        ):
            raise PermissionError("V2 replay access requires the frozen V2 manifest")
        expected = str(value.get("lock_sha256", ""))
        unsigned = {field: item for field, item in value.items() if field != "lock_sha256"}
        # V2's immutable protocol predates V3 canonical_json and used the
        # json.dumps default ``ensure_ascii=True`` in its lock calculation.
        observed = sha256_bytes(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        )
        if expected != observed:
            raise ValueError("frozen V2 manifest internal lock hash mismatch")
    else:
        raise ValueError(f"unknown manifest_access_class: {access_class}")
    return value
