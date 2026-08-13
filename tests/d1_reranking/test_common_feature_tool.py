from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.d1_reranking.extract_common_features import _verified_common_manifest
from unified_reranking.hashing import sha256_file


def test_d1_common_feature_tool_has_no_label_loader_import() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "d1_reranking"
        / "extract_common_features.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert all("evaluator" not in name and "label" not in name for name in imports)


def test_common_wrapper_verifies_upstream_candidate_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    compatibility = tmp_path / "candidates.parquet"
    paired = tmp_path / "paired.parquet"
    compatibility.write_bytes(b"candidates")
    paired.write_bytes(b"paired")
    artifacts = {}
    for name in ("candidate_features", "candidate_relations", "sample_context"):
        path = tmp_path / f"{name}.parquet"
        path.write_bytes(name.encode("utf-8"))
        artifacts[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    manifest_path = tmp_path / "feature_manifest.json"
    payload = {
        "status": "COMPLETE",
        "route": "d1",
        "split": "train",
        "pool": "top5",
        "tag": "formal",
        "candidate_manifest_sha256": sha256_file(compatibility),
        "paired_manifest_sha256": sha256_file(paired),
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        _verified_common_manifest(
            manifest_path,
            split="train",
            pool="top5",
            compatibility=compatibility,
            paired_alias=paired,
        )["route"]
        == "d1"
    )
    compatibility.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="semantics differ"):
        _verified_common_manifest(
            manifest_path,
            split="train",
            pool="top5",
            compatibility=compatibility,
            paired_alias=paired,
        )
