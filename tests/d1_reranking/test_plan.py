from __future__ import annotations

from pathlib import Path

import torch

from d1_reranking.models import D1ResidualMLPScorer
from d1_reranking.plan import primary_plan


def test_primary_plan_has_exact_equal_budget_job_universe() -> None:
    source = Path(__file__)
    first = primary_plan(tool_paths=(source,))
    second = primary_plan(tool_paths=(source,))
    assert first == second
    assert first["job_count"] == 360
    assert len({job["job_id"] for job in first["jobs"]}) == 360
    assert all(
        job["worker_argv"]
        == [
            "-m",
            "tools.d1_reranking.train_primary_cell",
            "--job-id",
            job["job_id"],
            "--resume",
        ]
        for job in first["jobs"]
    )
    assert first["equal_validation_trial_budget"] == {
        "R2": 4,
        "R3": 4,
        "R4": 4,
        "R5": 4,
        "R6": 4,
    }
    assert first["r7_selection"]["eligible"] == ["R3", "R5", "R6"]
    assert (
        first["fold_local_calibration"]["global_train_oof_base_logit_for_ranker_cells"]
        is False
    )


def test_d1_mlp_alpha_zero_is_exact_native_control() -> None:
    model = D1ResidualMLPScorer(3, hidden_dims=(8, 4), dropout=0.1, alpha=0.0)
    features = torch.randn(2, 5, 3)
    native = torch.randn(2, 5)
    padding = torch.tensor(
        [[False, False, False, True, True], [False, False, True, True, True]]
    )
    assert torch.equal(model(features, native, padding), native)
