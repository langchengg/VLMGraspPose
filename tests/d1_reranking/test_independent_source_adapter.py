from __future__ import annotations

import json
from pathlib import Path

import pytest

from d1_reranking import formal
from d1_reranking.independent_source_adapter import (
    FORMAL_CONTRACT_FRAGMENT,
    OLD_FRAGMENT,
    run_adapted_independent_recompute,
    transform_locked_source,
)


def test_transform_removes_only_unique_native_rank_predicate() -> None:
    source = "before\n" + OLD_FRAGMENT + "after\n"
    assert transform_locked_source(source) == "before\nafter\n"
    with pytest.raises(RuntimeError, match="not unique"):
        transform_locked_source("before\nafter\n")
    with pytest.raises(RuntimeError, match="not unique"):
        transform_locked_source(source + OLD_FRAGMENT)


def test_formal_validator_preserves_native_rank_provenance() -> None:
    text = Path(formal.__file__).read_text(encoding="utf-8")
    assert text.count(FORMAL_CONTRACT_FRAGMENT) == 1
    assert "or not universe_ranks.eq(1).all()" not in text


def test_adapter_requires_absent_final_output(tmp_path: Path) -> None:
    (tmp_path / "17_independent_recompute").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_adapted_independent_recompute(tmp_path)


def test_adapter_audit_schema_is_json_serializable() -> None:
    value = {
        "fragment": OLD_FRAGMENT,
        "contract": FORMAL_CONTRACT_FRAGMENT,
        "scientific_values_changed": False,
    }
    assert json.loads(json.dumps(value)) == value
