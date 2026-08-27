"""Plotting tests use clearly marked unit fixtures, never paper metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from graspnet6d.reporting import plot_system_comparison, validate_result_rows


def _unit_plot_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": ["oracle-mask", "oracle-mask", "predicted-mask", "predicted-mask"] * 2,
            "system": ["native", "reranked", "native", "reranked"] * 2,
            "seed": [1] * 4 + [2] * 4,
            "value": [0.31, 0.36, 0.22, 0.27, 0.32, 0.35, 0.21, 0.28],
            "fixture_scope": ["unit_fixture_only"] * 8,
            "status": ["COMPLETE"] * 8,
        }
    )


def test_reporting_refuses_empty_placeholder_and_blocked_rows() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_result_rows(
            pd.DataFrame(), required_columns=["value"], numeric_columns=["value"]
        )
    placeholder = pd.DataFrame(
        {"condition": ["placeholder"], "system": ["native"], "value": [0.0]}
    )
    with pytest.raises(ValueError, match="placeholder"):
        validate_result_rows(
            placeholder,
            required_columns=["condition", "system", "value"],
            numeric_columns=["value"],
        )
    blocked = pd.DataFrame({"value": [0.1], "status": ["BLOCKED"]})
    with pytest.raises(ValueError, match="blocked"):
        validate_result_rows(
            blocked, required_columns=["value"], numeric_columns=["value"]
        )


def test_plot_is_generated_from_tidy_rows_as_pdf_and_png(tmp_path) -> None:
    artifacts = plot_system_comparison(
        _unit_plot_fixture(),
        tmp_path / "unit-system-comparison",
        ylabel="Unit-fixture target P@1",
    )
    assert artifacts["pdf"].read_bytes().startswith(b"%PDF")
    assert artifacts["png"].read_bytes().startswith(b"\x89PNG")
    assert artifacts["pdf"].stat().st_size > 1_000
    assert artifacts["png"].stat().st_size > 1_000

