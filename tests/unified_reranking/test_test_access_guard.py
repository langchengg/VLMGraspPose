from __future__ import annotations

from pathlib import Path

import pytest

from src.unified_reranking.test_access_guard import assert_test_labels_unlocked
from src.unified_reranking.hashing import atomic_json
from src.unified_reranking.lock import (
    claim_formal_test_execution,
    create_formal_test_lock,
    finalize_formal_test_execution,
)


def test_test_label_access_is_denied_without_lock(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="locked before"):
        assert_test_labels_unlocked(tmp_path)
    log = tmp_path / "09_formal_test/test_access.log"
    assert "candidate_test_label_access_denied" in log.read_text()


def test_test_label_access_requires_hash_valid_lock_and_single_claim(tmp_path: Path) -> None:
    source = tmp_path / "selected.json"
    source.write_text("{}\n")
    atomic_json(
        tmp_path / "manifest.json",
        {"formal_test_execution_count": 0, "test_label_state": "LOCKED_PREVALIDATION"},
    )
    create_formal_test_lock(
        tmp_path,
        declaration={"primary": "m"},
        locked_files={"selection": source},
    )
    with pytest.raises(PermissionError, match="claimed"):
        assert_test_labels_unlocked(tmp_path)
    claim_formal_test_execution(tmp_path)
    assert_test_labels_unlocked(tmp_path)


def test_tampered_locked_file_denies_access(tmp_path: Path) -> None:
    source = tmp_path / "selected.json"
    source.write_text("{}\n")
    atomic_json(tmp_path / "manifest.json", {"formal_test_execution_count": 0})
    create_formal_test_lock(tmp_path, declaration={}, locked_files={"selection": source})
    source.write_text('{"changed":true}\n')
    with pytest.raises(PermissionError, match="hash mismatch"):
        claim_formal_test_execution(tmp_path)


def test_completed_execution_does_not_reauthorize_label_reader(tmp_path: Path) -> None:
    source = tmp_path / "selected.json"
    source.write_text("{}\n")
    atomic_json(
        tmp_path / "manifest.json",
        {"formal_test_execution_count": 0, "test_label_state": "LOCKED_PREVALIDATION"},
    )
    create_formal_test_lock(
        tmp_path, declaration={"primary": "m"}, locked_files={"selection": source}
    )
    claim_formal_test_execution(tmp_path)
    artifact = tmp_path / "result.json"
    artifact.write_text("{}\n")
    finalize_formal_test_execution(tmp_path, {"result": artifact})
    with pytest.raises(PermissionError, match="invalid formal Test execution claim"):
        assert_test_labels_unlocked(tmp_path)
