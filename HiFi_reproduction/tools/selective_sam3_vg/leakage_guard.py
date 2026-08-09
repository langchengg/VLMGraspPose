#!/usr/bin/env python3
"""Static fail-closed guard for deployable inference modules."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    ROOT / "src/segmentation/selective_sam3_vg/prompts.py",
    ROOT / "src/segmentation/selective_sam3_vg/features.py",
    ROOT / "src/segmentation/selective_sam3_vg/models.py",
    ROOT / "tools/selective_sam3_vg/run_formal_inference.py",
)
FORBIDDEN_IMPORT_PARTS = ("evaluation", "metrics", "diagnose_baseline", "analyze_pilot")
FORBIDDEN_SYMBOLS = (
    "load_ground_truth", "gt_mask", "ground_truth_mask", "candidate_iou",
    "delta_iou", "threshold_success", "validation_target",
)


def audit_module(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(part in alias.name for part in FORBIDDEN_IMPORT_PARTS):
                    violations.append(f"line {node.lineno}: forbidden import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(part in module for part in FORBIDDEN_IMPORT_PARTS):
                violations.append(f"line {node.lineno}: forbidden import {module}")
        elif isinstance(node, ast.Call):
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name in {"open", "read_csv", "read_parquet", "load", "read_text"}:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        lowered = argument.value.lower()
                        if any(symbol in lowered for symbol in FORBIDDEN_SYMBOLS):
                            violations.append(f"line {node.lineno}: forbidden data access {argument.value!r}")
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = str(node.slice.value).lower()
            if any(symbol == key or key.startswith(symbol + "_") for symbol in FORBIDDEN_SYMBOLS):
                # The formal runner's fail-closed manifest-key denylist is a
                # set literal, not a Subscript, so it remains allowed.
                violations.append(f"line {node.lineno}: forbidden feature/key access {key!r}")
    return violations


def main() -> None:
    all_violations = {str(path): audit_module(path) for path in MODULES}
    all_violations = {path: values for path, values in all_violations.items() if values}
    if all_violations:
        raise SystemExit(json.dumps({"status": "FAILED", "violations": all_violations}, indent=2))
    print(json.dumps({"status": "PASSED", "modules": [str(path) for path in MODULES]}, indent=2))


if __name__ == "__main__":
    main()
