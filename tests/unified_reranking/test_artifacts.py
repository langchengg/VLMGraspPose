import json

import pytest

from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verified_manifest_artifact,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import sha256_file


def test_verified_artifact_rejects_byte_drift(tmp_path):
    artifact = tmp_path / "features.bin"
    artifact.write_bytes(b"frozen")
    record = {"path": str(artifact), "sha256": sha256_file(artifact)}
    assert verified_artifact_path(record, name="features") == artifact.resolve()
    artifact.write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verified_artifact_path(record, name="features")


def test_manifest_helpers_verify_status_and_nested_artifact(tmp_path):
    artifact = tmp_path / "features.bin"
    artifact.write_bytes(b"frozen")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "artifacts": {
                    "features": {
                        "path": str(artifact),
                        "sha256": sha256_file(artifact),
                    }
                },
            }
        )
    )
    manifest = load_verified_json(manifest_path, name="feature manifest")
    assert (
        verified_manifest_artifact(manifest, key="features", name="features")
        == artifact.resolve()
    )


def test_recursive_artifact_verification_rejects_nested_child_drift(tmp_path):
    source = tmp_path / "source.bin"
    child = tmp_path / "child.bin"
    source.write_bytes(b"source")
    child.write_bytes(b"child")
    tree = {
        "sources": [{"path": str(source), "sha256": sha256_file(source)}],
        "artifacts": {
            "nested": [{"path": str(child), "sha256": sha256_file(child)}]
        },
    }
    assert len(
        verify_artifact_records_recursive(
            tree, name="synthetic tree", require_at_least_one=True
        )
    ) == 2
    child.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_artifact_records_recursive(tree, name="synthetic tree")
