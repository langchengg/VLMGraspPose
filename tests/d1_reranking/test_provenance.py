from __future__ import annotations

from pathlib import Path

import pytest

import d1_reranking.provenance as provenance
from d1_reranking.provenance import (
    _closure_identity,
    _verify,
    load_source_closure,
    publish_source_closure,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage


def test_source_hash_verifier_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"source")
    with pytest.raises(RuntimeError, match="source drift"):
        _verify(source, "0" * 64, "synthetic")


def _write_content(path: Path, value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def test_source_closure_is_versioned_and_recursively_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"frozen")
    source_sha = sha256_file(source)
    record = {
        "name": "synthetic",
        "path": str(source.resolve()),
        "sha256": source_sha,
        "bytes": source.stat().st_size,
    }
    monkeypatch.setattr(
        provenance,
        "_authoritative_named_paths",
        lambda: {"synthetic": source.resolve()},
    )
    monkeypatch.setattr(provenance, "EXPECTED_SOURCE_HASHES", {"synthetic": source_sha})
    monkeypatch.setattr(
        provenance,
        "_expected_canonical_input_paths",
        lambda: {
            split: {
                "candidate_sources": [str(source.resolve())],
                "score_sources": [str(source.resolve())],
                "paired_manifest": str(source.resolve()),
                "native_features": str(source.resolve()),
                "feature_allowlist": str(source.resolve()),
                "feature_manifest": str(source.resolve()),
                **(
                    {"development_labels": str(source.resolve())}
                    if split in {"train", "validation"}
                    else {
                        "opaque_ground_truth": str(source.resolve()),
                        "opaque_visual_ground_truth": str(source.resolve()),
                    }
                ),
            }
            for split in ("train", "validation", "test")
        },
    )
    input_record = {"path": str(source.resolve()), "sha256": source_sha}
    closure: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS",
        "canonical_snapshot": "A",
        "selection_used_test_metrics": False,
        "candidate_test_labels_read": False,
        "repo_root": str(provenance.REPO_ROOT),
        "verified_artifacts": [record],
        "development": {"train": {}, "validation": {}},
        "test": {"key_audit": {}, "model_contract": []},
        "canonical_inputs": {
            split: {
                "candidate_sources": [input_record],
                "score_sources": [input_record],
                "paired_manifest": input_record,
                "native_features": input_record,
                "feature_allowlist": input_record,
                "feature_manifest": input_record,
                **(
                    {"development_labels": input_record}
                    if split in {"train", "validation"}
                    else {
                        "opaque_ground_truth": input_record,
                        "opaque_visual_ground_truth": input_record,
                    }
                ),
            }
            for split in ("train", "validation", "test")
        },
    }
    closure_id = _closure_identity(closure)
    closure["closure_id"] = closure_id
    closure_path = (
        tmp_path / "00_audit" / "source_reconciliations" / closure_id / "manifest.json"
    )
    _write_content(closure_path, closure)
    with ledger_stage(
        tmp_path / "run_ledger.sqlite",
        stage="P1",
        substage=f"synthetic_{closure_id}",
        route="D1",
        method="provenance",
        command="synthetic source reconciliation",
    ) as state:
        state["artifact_path"] = str(closure_path.resolve())
        state["artifact_sha256"] = sha256_file(closure_path)
    publish_source_closure(tmp_path, closure_path)
    observed_path, observed = load_source_closure(tmp_path)
    assert observed_path == closure_path.resolve()
    assert observed["canonical_snapshot"] == "A"

    source.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_source_closure(tmp_path)
