from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.unified_reranking.run_postlock_r1_r6_test_comparison import (
    ROUTES,
    _formal_route_labels,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    candidates_root = source / "02_candidates"
    candidates_root.mkdir(parents=True)
    formal_rows: list[dict[str, object]] = []
    for route in ROUTES:
        candidates = pd.DataFrame(
            {
                "sample_id": ["sample-0", "sample-0"],
                "candidate_id": ["candidate-0", "candidate-1"],
                "native_rank": [1, 2],
                "candidate_geometry_sha256": [f"{route}-geometry-0", f"{route}-geometry-1"],
            }
        )
        candidates.to_parquet(candidates_root / f"{route}_test_top5.parquet", index=False)
        for row in candidates.itertuples(index=False):
            formal_rows.append(
                {
                    "system_name": f"{route}_native",
                    "route": route,
                    "sample_id": row.sample_id,
                    "candidate_id": row.candidate_id,
                    "native_rank": row.native_rank,
                    "candidate_geometry_sha256": row.candidate_geometry_sha256,
                    "candidate_success": int(row.native_rank == 2),
                }
            )
        # A second formal system is deliberately present; the diagnostic
        # extractor must not duplicate labels from it.
        for row in candidates.itertuples(index=False):
            formal_rows.append(
                {
                    "system_name": f"{route}_ungated_primary",
                    "route": route,
                    "sample_id": row.sample_id,
                    "candidate_id": row.candidate_id,
                    "native_rank": row.native_rank,
                    "candidate_geometry_sha256": row.candidate_geometry_sha256,
                    "candidate_success": int(row.native_rank == 2),
                }
            )
    formal_path = tmp_path / "per_candidate_scores.parquet"
    pd.DataFrame(formal_rows).to_parquet(formal_path, index=False)
    return source, formal_path


def test_formal_route_labels_use_only_native_rows_and_exact_top5(tmp_path: Path) -> None:
    source, formal_path = _fixture(tmp_path)
    labels = _formal_route_labels(formal_path, source)
    assert set(labels) == set(ROUTES)
    for route in ROUTES:
        assert len(labels[route]) == 2
        assert labels[route]["candidate_success"].tolist() == [0, 1]
        assert labels[route]["native_rank"].tolist() == [1, 2]


def test_formal_route_labels_reject_candidate_universe_drift(tmp_path: Path) -> None:
    source, formal_path = _fixture(tmp_path)
    formal = pd.read_parquet(formal_path)
    formal = formal.loc[
        ~(
            formal["system_name"].eq("g1_native")
            & formal["candidate_id"].eq("candidate-1")
        )
    ]
    formal.to_parquet(formal_path, index=False)
    with pytest.raises(RuntimeError, match="formal g1 labels do not exactly cover"):
        _formal_route_labels(formal_path, source)
