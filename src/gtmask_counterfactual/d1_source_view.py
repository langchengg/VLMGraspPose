"""Content-addressed executable source view for frozen D1 replay tools."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from .execution import FROZEN_D1_SOURCE, REPOSITORY_ROOT
from .io import artifact_record, atomic_json, canonical_sha256, sha256_file


GQCNN_COMMIT = "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21"
GQCNN_OVERRIDE_HASHES = {
    "gqcnn/__init__.py": (
        "dd924f1c533c7ecac3b5f6aaa0feffdd1bfc4329f85ba06b84446f440a6346dc"
    ),
    "gqcnn/grasping/__init__.py": (
        "d14dbf3519598e02ca8453363d91387e4323f106b7545de4237f4a3e395dd63d"
    ),
}
SUPPORT_FILES = {
    "scripts/gtmask_oracle_candidate_bootstrap.py": (
        REPOSITORY_ROOT
        / "tools/gtmask_counterfactual/d1_oracle_candidate_bootstrap.py",
        "3412992989cf459c73f9d36f926d64356d95f0e1b52b42c56429c1bc66159fd6",
    ),
    "src/grasping/camera_geometry.py": (
        REPOSITORY_ROOT / "HiFi_reproduction/src/grasping/camera_geometry.py",
        "22daa97137b7143139d185f72d2564309667b73219684380ae949b0b8b3240c3",
    ),
    "src/grasping/dexnet_scoring.py": (
        REPOSITORY_ROOT / "HiFi_reproduction/src/grasping/dexnet_scoring.py",
        "833d05d25ba25ee4d4ab7facd9eb3ceb5d469e722c79406cb8fbb98fab34c641",
    ),
    "src/grasping/grasp_visualization.py": (
        REPOSITORY_ROOT / "HiFi_reproduction/src/grasping/grasp_visualization.py",
        "17a52c77edc614b63f4b408b4473a5e37fd9e00ce7be9d2aecd810582295a423",
    ),
    "scripts/score_existing_dexnet_candidates.py": (
        REPOSITORY_ROOT
        / "HiFi_reproduction/scripts/score_existing_dexnet_candidates.py",
        "9d7215576db238648334f08d8b5ec03e3e3f2f4e7c743b195864cd8b69952cc5",
    ),
}


class D1SourceViewError(RuntimeError):
    """The executable source view is missing, changed, or unsafe."""


def _safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise D1SourceViewError(f"unsafe source-view path: {value!r}")
    return Path(*pure.parts)


def _regular(path: Path, *, label: str) -> Path:
    source = path.expanduser().resolve(strict=False)
    if path.expanduser().is_symlink() or not source.is_file():
        raise D1SourceViewError(f"{label} must be a regular file: {source}")
    return source


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise D1SourceViewError(
            f"cannot inspect pinned GQ-CNN checkout: {completed.stderr.strip()}"
        )
    return completed.stdout


def production_source_files() -> dict[str, Path]:
    """Resolve and hash the complete executable D1 source dependency graph."""

    snapshot = FROZEN_D1_SOURCE / "source_snapshot"
    result: dict[str, Path] = {}
    for source in sorted(snapshot.rglob("*")):
        if (
            not source.is_file()
            or source.is_symlink()
            or "__pycache__" in source.parts
            or source.suffix in {".pyc", ".pyo"}
            or "third_party/gqcnn-official" in source.as_posix()
        ):
            continue
        result[source.relative_to(snapshot).as_posix()] = source.resolve()

    checkout = REPOSITORY_ROOT / "HiFi_reproduction/third_party/gqcnn-official"
    if _git(checkout, "rev-parse", "HEAD").strip() != GQCNN_COMMIT:
        raise D1SourceViewError("GQ-CNN checkout commit differs")
    status = {
        line[3:]
        for line in _git(checkout, "status", "--porcelain", "--untracked-files=no").splitlines()
        if line.strip()
    }
    if status != set(GQCNN_OVERRIDE_HASHES):
        raise D1SourceViewError(f"GQ-CNN checkout modifications differ: {sorted(status)}")
    for relative in _git(checkout, "ls-files").splitlines():
        if not relative:
            continue
        source = _regular(checkout / _safe_relative(relative), label="GQ-CNN source")
        result[f"third_party/gqcnn-official/{relative}"] = source
    for relative, expected in GQCNN_OVERRIDE_HASHES.items():
        source = result[f"third_party/gqcnn-official/{relative}"]
        if sha256_file(source) != expected:
            raise D1SourceViewError(f"GQ-CNN override hash differs: {relative}")
    for relative, (source, expected) in SUPPORT_FILES.items():
        resolved = _regular(source, label=f"D1 support source {relative}")
        if sha256_file(resolved) != expected:
            raise D1SourceViewError(f"D1 support source hash differs: {relative}")
        result[relative] = resolved
    return result


def _source_records(files: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative, path in sorted(files.items()):
        safe = _safe_relative(relative).as_posix()
        if safe in records:
            raise D1SourceViewError(f"duplicate source-view path: {safe}")
        records[safe] = artifact_record(_regular(path, label=f"source {safe}"))
    if not records:
        raise D1SourceViewError("D1 source view cannot be empty")
    return records


def verify_d1_source_view(path: str | Path) -> dict[str, Any]:
    manifest_path = _regular(Path(path), label="D1 source-view manifest")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D1SourceViewError("cannot parse D1 source-view manifest") from error
    if not isinstance(value, dict):
        raise D1SourceViewError("D1 source-view manifest is not an object")
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise D1SourceViewError("D1 source-view content hash differs")
    root = manifest_path.parent
    if manifest_path != root / "SOURCE_VIEW_MANIFEST.json":
        raise D1SourceViewError("D1 source-view manifest name differs")
    files = value.get("files")
    if value.get("status") != "COMPLETE" or not isinstance(files, Mapping):
        raise D1SourceViewError("D1 source-view manifest is incomplete")
    for relative, record in files.items():
        if not isinstance(record, Mapping):
            raise D1SourceViewError(f"source-view record is malformed: {relative}")
        destination = _regular(root / _safe_relative(str(relative)), label="source-view member")
        observed = artifact_record(destination)
        if any(record.get(key) != observed[key] for key in ("path", "sha256", "bytes")):
            raise D1SourceViewError(f"source-view member differs: {relative}")
    if int(value.get("file_count", -1)) != len(files):
        raise D1SourceViewError("D1 source-view file count differs")
    return value


def build_d1_source_view(
    base_dir: str | Path,
    *,
    source_files: Mapping[str, Path] | None = None,
) -> Path:
    """Materialise a non-symlink executable view and verify every output byte."""

    base = Path(base_dir).expanduser().resolve(strict=False)
    records = _source_records(source_files or production_source_files())
    identity = canonical_sha256(
        {
            "gqcnn_commit": GQCNN_COMMIT,
            "files": records,
            "source_view_role": "adapter_support_then_exact_replay_required",
        }
    )
    destination = base / identity[:24]
    manifest_path = destination / "SOURCE_VIEW_MANIFEST.json"
    if manifest_path.exists():
        value = verify_d1_source_view(manifest_path)
        if value.get("source_identity_sha256") != identity:
            raise D1SourceViewError("existing D1 source-view identity differs")
        return manifest_path
    base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{identity[:12]}.", dir=base))
    try:
        output_records: dict[str, dict[str, Any]] = {}
        for relative, source_record in records.items():
            target = staging / _safe_relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(str(source_record["path"])), target, follow_symlinks=False)
            if target.is_symlink() or sha256_file(target) != source_record["sha256"]:
                raise D1SourceViewError(f"copied source-view member differs: {relative}")
            output_records[relative] = artifact_record(target)
        os.replace(staging, destination)
        final_records = {
            relative: artifact_record(destination / _safe_relative(relative))
            for relative in sorted(output_records)
        }
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE",
            "gqcnn_commit": GQCNN_COMMIT,
            "source_identity_sha256": identity,
            "source_view_role": "adapter_support_then_exact_replay_required",
            "file_count": len(final_records),
            "sources": records,
            "files": final_records,
        }
        payload["content_sha256"] = canonical_sha256(payload)
        atomic_json(manifest_path, payload)
        verify_d1_source_view(manifest_path)
        return manifest_path
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "D1SourceViewError",
    "build_d1_source_view",
    "production_source_files",
    "verify_d1_source_view",
]
