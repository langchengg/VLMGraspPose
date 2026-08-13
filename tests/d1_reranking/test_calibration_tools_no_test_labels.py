from __future__ import annotations

import ast
from pathlib import Path


def test_label_free_calibration_application_has_no_evaluator_or_label_import() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "d1_reranking"
        / "apply_calibration.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert all("evaluator" not in name and "label" not in name for name in imports)
