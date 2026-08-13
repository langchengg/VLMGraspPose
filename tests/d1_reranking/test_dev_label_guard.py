from __future__ import annotations

from pathlib import Path

import pytest

from d1_reranking.candidates import artifact_record
from tools.d1_reranking.build_dev_labels import _candidate_contract
from unified_reranking.hashing import atomic_json, canonical_sha256
from unified_reranking.evaluator_adapter import build_candidate_labels


def test_development_label_builder_refuses_test_before_formal(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="Train/Validation labels only"):
        build_candidate_labels(
            tmp_path / "candidates.parquet",
            tmp_path / "test_labels.parquet",
            tmp_path / "output.parquet",
            tmp_path / "evaluator.py",
            "0" * 64,
            split="test",
        )


def test_development_label_builder_requires_current_source_closure(
    tmp_path: Path,
) -> None:
    closure = tmp_path / "closure.json"
    other_closure = tmp_path / "other_closure.json"
    closure.write_text("closure", encoding="utf-8")
    other_closure.write_text("other", encoding="utf-8")
    split_dir = tmp_path / "02_candidates" / "train"
    candidates = split_dir / "d1_top5_candidates.parquet"
    hashes = split_dir / "candidate_hashes.parquet"
    candidates.parent.mkdir(parents=True)
    candidates.write_bytes(b"candidate rows")
    hashes.write_bytes(b"candidate hashes")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": {
            "route": "D1",
            "split": "train",
            "source_contract": {"source_closure": artifact_record(other_closure)},
        },
        "artifacts": {
            "top5": artifact_record(candidates),
            "candidate_hashes": artifact_record(hashes),
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(split_dir / "manifest.json", manifest)

    with pytest.raises(RuntimeError, match="source-closure binding differs"):
        _candidate_contract(tmp_path, "train", "top5", closure)

    closure_record = artifact_record(closure)
    closure_record.pop("bytes")
    manifest["configuration"] = {
        "route": "D1",
        "split": "train",
        "source_contract": {"source_closure": closure_record},
    }
    manifest.pop("content_sha256")
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(split_dir / "manifest.json", manifest)

    observed, observed_manifest, observed_hashes = _candidate_contract(
        tmp_path, "train", "top5", closure
    )
    assert observed == candidates.resolve()
    assert observed_manifest == (split_dir / "manifest.json")
    assert observed_hashes == hashes.resolve()
