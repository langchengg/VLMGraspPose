from __future__ import annotations

import hashlib

import pandas as pd

from gtmask_counterfactual.pilot import (
    PILOT_SALT,
    PREDICTED_CLASSES,
    _balanced_selection,
    _predicted_class,
    _target_quartiles,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_target_quartiles_are_deterministic_and_balanced() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{index:03d}" for index in range(20)],
            "foreground_fraction": [0.25] * 20,
        }
    )
    first = _target_quartiles(frame)
    second = _target_quartiles(frame.sample(frac=1, random_state=9))

    assert first == second
    assert pd.Series(first).value_counts().to_dict() == {
        "Q1": 5,
        "Q2": 5,
        "Q3": 5,
        "Q4": 5,
    }


def test_joint_stratum_pilot_is_sha_stable_and_covers_outcomes() -> None:
    rows = []
    index = 0
    for query_type in ("thing", "stuff"):
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            for outcome in PREDICTED_CLASSES:
                for _ in range(9):
                    sample_id = f"sample-{index:04d}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "query_type": query_type,
                            "target_size_quartile": quartile,
                            "predicted_outcome_class": outcome,
                            "selection_sha256": _sha(
                                f"{PILOT_SALT}|sample|{sample_id}"
                            ),
                        }
                    )
                    index += 1
    frame = pd.DataFrame(rows)
    first = _balanced_selection(frame, size=200)
    second = _balanced_selection(
        frame.sample(frac=1, random_state=27), size=200
    )

    assert first["sample_id"].tolist() == second["sample_id"].tolist()
    assert first["selection_rank"].tolist() == list(range(1, 201))
    assert set(first["query_type"]) == {"thing", "stuff"}
    assert set(first["target_size_quartile"]) == {"Q1", "Q2", "Q3", "Q4"}
    assert set(first["predicted_outcome_class"]) == set(PREDICTED_CLASSES)


def test_predicted_outcome_class_precedence_includes_no_output() -> None:
    row = type(
        "Row",
        (),
        {
            "candidate_count_all": 0,
            "native_correct": True,
            "full_pool_positive": True,
        },
    )()
    assert _predicted_class(row) == "predicted_no_output"

