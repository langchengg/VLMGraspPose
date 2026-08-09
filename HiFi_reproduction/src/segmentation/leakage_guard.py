"""Fail-closed static and runtime guards for locked GT-free inference."""

from __future__ import annotations

import ast
import builtins
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any


FORBIDDEN_PATH_TOKENS = (
    "ground_truth",
    "groundtruth",
    "gt_mask",
    "candidate_iou",
    "candidate_correctness",
    "threshold_success",
    "answer_instance",
    "target_instance_annotation",
)
FORBIDDEN_KEYS = {
    "answer_instance_value",
    "target_instance_id",
    "candidate_iou",
    "test_iou",
    "test_candidate_correctness",
    "threshold_success_label",
    "gt_mask_path",
    "ground_truth_mask",
}
FORBIDDEN_INFERENCE_IMPORT_PARTS = (
    "proposal_oracle",
    "selective_sam3_vg.evaluation",
    "compute_proposal_oracle",
    "evaluate_locked",
)


def assert_no_forbidden_mapping(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS:
                raise RuntimeError(f"{context}: forbidden inference key {key!r}")
            assert_no_forbidden_mapping(child, context=context)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_forbidden_mapping(child, context=context)


def static_scan(path: str | Path) -> list[str]:
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if any(token in name for token in FORBIDDEN_INFERENCE_IMPORT_PARTS):
                violations.append(f"line {node.lineno}: forbidden inference import {name}")
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = str(node.slice.value).lower()
            if key in FORBIDDEN_KEYS:
                violations.append(f"line {node.lineno}: forbidden inference key {key}")
    return violations


class RuntimeFileAccessGuard(AbstractContextManager):
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.records: list[dict[str, Any]] = []
        self._open = builtins.open
        self._path_open = Path.open

    def _check(self, value: Any, mode: str) -> None:
        path = str(value)
        lowered = path.lower()
        forbidden = next((token for token in FORBIDDEN_PATH_TOKENS if token in lowered), None)
        record = {"path": path, "mode": mode, "forbidden_token": forbidden}
        self.records.append(record)
        if forbidden:
            raise RuntimeError(f"forbidden locked-inference file access: {path}")

    def __enter__(self):
        guard = self

        def guarded_open(file, mode="r", *args, **kwargs):
            guard._check(file, str(mode))
            return guard._open(file, mode, *args, **kwargs)

        def guarded_path_open(path, mode="r", *args, **kwargs):
            guard._check(path, str(mode))
            return guard._path_open(path, mode, *args, **kwargs)

        builtins.open = guarded_open
        Path.open = guarded_path_open
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.open = self._open
        Path.open = self._path_open
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._open(self.log_path, "a", encoding="utf-8") as stream:
            for record in self.records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return False


__all__ = [
    "RuntimeFileAccessGuard",
    "assert_no_forbidden_mapping",
    "static_scan",
]
