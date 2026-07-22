"""Strict serialization and provenance helpers for SAM 3 refinement."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


FORBIDDEN_PROMPT_KEYS = {
    "ground_truth",
    "ground_truth_mask",
    "ground_truth_box",
    "grasps",
    "grasp_annotations",
    "ocid_vlg_2d_consistency",
    "answer_instance_value",
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_strict_json(path: Path | str, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_strict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(_strict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
        for row in rows
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def assert_no_ground_truth_leakage(payload: Any, *, context: str = "prompt payload") -> None:
    """Reject prompt/input metadata that contains evaluation-only information."""

    def visit(value: Any, trail: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in FORBIDDEN_PROMPT_KEYS or lowered.startswith("gt_"):
                    raise ValueError(f"{context} contains forbidden key {'.'.join((*trail, str(key)))}")
                visit(item, (*trail, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, (*trail, str(index)))
        elif isinstance(value, str):
            name = Path(value).name.lower()
            if "ground_truth" in name or "grasp_annotation" in name:
                raise ValueError(f"{context} references evaluation-only artifact at {'.'.join(trail)}")

    visit(payload, ())


def file_manifest(paths: Mapping[str, Path | str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, source in sorted(paths.items()):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def verify_file_manifest(root: Path | str, manifest: Mapping[str, Mapping[str, Any]]) -> None:
    root = Path(root)
    for logical_name, record in manifest.items():
        filename = str(record["filename"])
        if Path(filename).name != filename:
            raise ValueError(f"unsafe manifest filename for {logical_name}: {filename}")
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if int(record["bytes"]) != path.stat().st_size or str(record["sha256"]) != sha256_file(path):
            raise ValueError(f"manifest mismatch for {logical_name}: {path}")
