from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from d1_reranking.rules import r1_score


def test_r1_rule_matches_predeclared_log_formula() -> None:
    frame = pd.DataFrame(
        {
            "base_logit": [0.2],
            "p_center": [0.5],
            "rectangle_probability_mean": [0.25],
            "jaw_probability_min": [0.125],
        }
    )
    trial = {"beta_center": 0.5, "beta_rect": 0.5, "beta_jaw": 0.5}
    expected = 0.2 + 0.5 * (np.log(0.5) + np.log(0.25) + np.log(0.125))
    assert r1_score(frame, trial)[0] == pytest.approx(expected)


def test_r1_rule_rejects_nonprobability_support() -> None:
    frame = pd.DataFrame(
        {
            "base_logit": [0.2],
            "p_center": [1.2],
            "rectangle_probability_mean": [0.25],
            "jaw_probability_min": [0.125],
        }
    )
    with pytest.raises(ValueError, match="lie in"):
        r1_score(frame, {"beta_center": 0.5})
