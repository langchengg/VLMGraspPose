from __future__ import annotations

import pandas as pd

from d1_reranking.fold_calibration import (
    apply_partition_calibrator,
    fit_partition_calibrator,
)


def _rows(prefix: str, label_flip: bool = False) -> pd.DataFrame:
    rows = []
    for sample in range(4):
        for rank, score in ((1, 0.8), (2, 0.2)):
            rows.append(
                {
                    "sample_id": f"{prefix}{sample}",
                    "candidate_id": f"c{rank}",
                    "native_rank": rank,
                    "native_score_raw": score,
                    "candidate_success": int((rank == 1) ^ label_flip),
                }
            )
    return pd.DataFrame(rows)


def test_fold_calibrator_does_not_depend_on_held_labels() -> None:
    fit = _rows("fit")
    held = _rows("held")
    payload = fit_partition_calibrator(fit, method="platt", fit_fold_ids=(1, 2, 3))
    first = apply_partition_calibrator(held, payload)
    mutated = held.copy()
    mutated["candidate_success"] = 1 - mutated["candidate_success"]
    second = apply_partition_calibrator(mutated, payload)
    assert first["base_logit"].equals(second["base_logit"])
    assert (
        first.sort_values(["sample_id", "native_rank"])["candidate_id"].tolist()
        == second.sort_values(
            ["sample_id", "calibrated_native_probability"], ascending=[True, False]
        )["candidate_id"].tolist()
    )
