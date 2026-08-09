from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from failure_analysis.vlm_safe_rerank.api_calibration import (
    _assert_response_coverage,
    _response_index,
)


def _write(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_calibration_index_rejects_non_original_variant(tmp_path: Path) -> None:
    path = tmp_path / "responses.parquet"
    _write(path, [{
        "sample_id": "s", "challenger_candidate_id": "candidate_1",
        "model_id": "gemini-3.6-flash", "variant": "panel_swap", "parsed": None,
    }])
    with pytest.raises(ValueError, match="variant"):
        _response_index(path, expected_variant="original")


def test_calibration_index_rejects_duplicate_pair_model(tmp_path: Path) -> None:
    path = tmp_path / "responses.parquet"
    row = {
        "sample_id": "s", "challenger_candidate_id": "candidate_1",
        "model_id": "gemini-3.6-flash", "variant": "original", "parsed": None,
    }
    _write(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        _response_index(path, expected_variant="original")


def test_response_coverage_requires_every_pair_model_and_rejects_extras() -> None:
    pairs = [{"sample_id": "s", "challenger_candidate_id": "candidate_1"}]
    complete = {
        ("s", "candidate_1", "er2"): None,
        ("s", "candidate_1", "flash"): None,
    }
    _assert_response_coverage(complete, pairs, ["er2", "flash"], context="test")

    with pytest.raises(ValueError, match="missing=1"):
        _assert_response_coverage(
            {("s", "candidate_1", "er2"): None},
            pairs,
            ["er2", "flash"],
            context="test",
        )
    with pytest.raises(ValueError, match="extra=1"):
        _assert_response_coverage(
            {**complete, ("extra", "candidate_1", "er2"): None},
            pairs,
            ["er2", "flash"],
            context="test",
        )
