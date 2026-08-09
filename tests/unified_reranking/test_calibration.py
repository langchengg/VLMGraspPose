from __future__ import annotations

import numpy as np
import pandas as pd

from src.unified_reranking.calibration import (
    MonotoneIsotonicCalibrator,
    MonotonePlattCalibrator,
    assert_order_invariant,
    grouped_oof_calibration,
)


def test_platt_slope_is_positive_even_for_adversarial_labels() -> None:
    model = MonotonePlattCalibrator().fit([0, 1, 2, 3], [1, 1, 0, 0])
    prediction = model.predict([0, 1, 2, 3])
    assert model.slope_ is not None and model.slope_ > 0
    assert np.all(np.diff(prediction) >= 0)


def test_isotonic_ties_use_native_rank() -> None:
    model = MonotoneIsotonicCalibrator().fit([0, 1, 2, 3], [0, 1, 0, 1])
    frame = pd.DataFrame(
        {"sample_id": ["q"] * 4, "candidate_id": list("abcd"), "native_rank": [4, 3, 2, 1], "p": model.predict([0, 1, 2, 3])}
    )
    assert_order_invariant(frame, "p")


def test_grouped_oof_predicts_every_row_once() -> None:
    rows = []
    folds = []
    for sample in range(10):
        folds.append({"sample_id": f"q{sample}", "fold": sample % 5})
        for rank in (1, 2):
            rows.append({"sample_id": f"q{sample}", "candidate_id": f"c{rank}", "native_rank": rank, "native_score": 1.0 / rank + sample * 0.001, "candidate_success": rank == 1})
    train = pd.DataFrame(rows)
    validation = train.iloc[:8].copy()
    oof, val, _, metadata = grouped_oof_calibration(train, pd.DataFrame(folds), validation)
    assert len(oof) == len(train)
    assert oof["calibrated_native_probability"].notna().all()
    assert val["base_logit"].notna().all()
    assert metadata["selected_method"] in {"platt", "isotonic"}
