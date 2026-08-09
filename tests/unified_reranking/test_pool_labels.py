from __future__ import annotations

import pandas as pd

from src.unified_reranking.evaluator_adapter import annotate_pool_labels


def test_pool_solvability_is_recomputed_after_truncation() -> None:
    full = pd.DataFrame(
        {
            "sample_id": ["a"] * 6,
            "native_rank": [1, 2, 3, 4, 5, 6],
            "candidate_success": [False, False, False, False, False, True],
            "pool_solvable": [True] * 6,
            "first_positive_rank": [6] * 6,
        }
    )
    top5 = annotate_pool_labels(full.loc[full.native_rank <= 5])
    assert not top5["pool_solvable"].any()
    assert top5["first_positive_rank"].isna().all()
