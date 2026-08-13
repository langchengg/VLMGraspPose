from __future__ import annotations

import pandas as pd
import pytest

from d1_reranking.splits import validate_fold_assignments


def test_grouped_fold_validator_rejects_same_frame_crossing_folds() -> None:
    train = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "f"],
            "scene_id": ["s"] * 6,
            "rgbd_pair_sha256": ["frame", "frame", "c", "d", "e", "f"],
        }
    )
    assignments = train.copy()
    assignments["fold"] = [0, 1, 0, 2, 3, 4]
    with pytest.raises(ValueError, match="RGB-D frame crosses"):
        validate_fold_assignments(assignments, train)
