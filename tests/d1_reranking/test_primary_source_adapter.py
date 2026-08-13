from __future__ import annotations

from pathlib import Path

import pytest

from d1_reranking.primary_source_adapter import (
    NEW_SNIPPET,
    OLD_SNIPPET,
    verify_exact_rank_dtype_patch,
    verify_primary_cell_records_with_source_adapter,
    verify_records_with_primary_source_adapter,
)
from unified_reranking.hashing import sha256_file


def test_rank_dtype_patch_accepts_only_the_exact_source_change() -> None:
    old = f"before\n{OLD_SNIPPET}after\n".encode()
    current = f"before\n{NEW_SNIPPET}after\n".encode()
    verify_exact_rank_dtype_patch(old, current)

    with pytest.raises(RuntimeError, match="beyond"):
        verify_exact_rank_dtype_patch(old, current + b"# other change\n")

    with pytest.raises(RuntimeError, match="not unique"):
        verify_exact_rank_dtype_patch(b"unrelated\n", current)


def test_primary_cell_verifier_allows_only_two_audited_source_records(
    tmp_path: Path,
) -> None:
    frozen_path = tmp_path / "datasets.py"
    frozen_path.write_text("live numeric equality\n", encoding="utf-8")
    other_path = tmp_path / "model.bin"
    other_path.write_bytes(b"model")
    frozen_record = {
        "path": str(frozen_path),
        "sha256": "a" * 64,
        "bytes": 17,
    }
    live_record = {
        "path": str(frozen_path),
        "sha256": sha256_file(frozen_path),
        "bytes": frozen_path.stat().st_size,
    }
    other_record = {
        "path": str(other_path),
        "sha256": sha256_file(other_path),
        "bytes": other_path.stat().st_size,
    }
    cell = {
        "configuration": {
            "sources": {"training_code": [dict(frozen_record), dict(other_record)]}
        },
        "sources": {"training_code": [dict(frozen_record), dict(other_record)]},
        "artifacts": {"model": dict(other_record)},
    }
    adapter = {
        "frozen_source": {
            "path": str(frozen_path),
            "sha256": "a" * 64,
        },
        "live_source": live_record,
    }
    verify_primary_cell_records_with_source_adapter(
        cell, adapter=adapter, name="synthetic primary cell"
    )

    cell["sources"]["training_code"][0]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="frozen datasets.py record differs"):
        verify_primary_cell_records_with_source_adapter(
            cell, adapter=adapter, name="synthetic primary cell"
        )

    downstream = {
        "sources": {"inference_code": [{"path": str(frozen_path), "sha256": "a" * 64}]},
        "artifacts": {"prediction": other_record},
    }
    assert (
        verify_records_with_primary_source_adapter(
            downstream, adapter=adapter, name="synthetic downstream application"
        )
        == 1
    )
