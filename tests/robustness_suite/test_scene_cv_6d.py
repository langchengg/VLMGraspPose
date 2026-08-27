from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robustness_suite.scene_cv_6d import (
    EXPECTED_EARLY_STOPPING_ROUNDS,
    EXPECTED_HYPERPARAMETERS,
    EXPECTED_SCENES,
    PRIMARY_CONDITION,
    SOURCE_RUN_ID,
    _assert_isolated_run_dir,
    assert_frozen_candidate_pool,
    assign_group_splits,
    fit_fold_preprocessor,
    make_scene_folds,
)


@pytest.fixture(scope="module")
def folds() -> list[dict[str, object]]:
    return make_scene_folds(EXPECTED_SCENES)


def test_all_30_scenes_appear_once_in_outer_test(
    folds: list[dict[str, object]],
) -> None:
    appearances = [scene for fold in folds for scene in fold["test_scenes"]]
    assert len(appearances) == 30
    assert len(set(appearances)) == 30
    assert set(appearances) == set(EXPECTED_SCENES)


def test_6d_fold_train_val_test_disjoint(
    folds: list[dict[str, object]],
) -> None:
    for fold in folds:
        train = set(fold["train_scenes"])
        validation = set(fold["validation_scenes"])
        test = set(fold["test_scenes"])
        assert (len(train), len(validation), len(test)) == (19, 5, 6)
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)


def test_6d_group_inherits_scene_split(
    folds: list[dict[str, object]],
) -> None:
    universe = pd.DataFrame(
        {
            "group_id": [f"group-{index}" for index in range(30)],
            "scene_id": list(EXPECTED_SCENES),
        }
    )
    assigned = assign_group_splits(universe, folds[0])
    expected = {
        str(scene): partition
        for partition in ("train", "validation", "test")
        for scene in folds[0][f"{partition}_scenes"]
    }
    assert assigned["outer_partition"].tolist() == [
        expected[scene] for scene in universe["scene_id"]
    ]


def test_6d_preprocessing_fit_on_train_only() -> None:
    train = pd.DataFrame({"native_score": [1.0, 3.0, np.nan]})
    validation = pd.DataFrame({"native_score": [1000.0, np.nan]})
    test = pd.DataFrame({"native_score": [-1000.0, np.nan]})
    train_x, validation_x, test_x, artifact = fit_fold_preprocessor(
        train, validation, test, ("native_score",)
    )
    assert artifact["fit_scope"] == "training_only"
    assert artifact["fill_values"] == {"native_score": 2.0}
    assert train_x["native_score"].tolist() == [1.0, 3.0, 2.0]
    assert validation_x["native_score"].tolist() == [1000.0, 2.0]
    assert test_x["native_score"].tolist() == [-1000.0, 2.0]


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": ["g0", "g0", "g1"],
            "candidate_id": ["c0", "c1", "c2"],
            "geometry_sha256": ["a" * 64, "b" * 64, "c" * 64],
            "gripper_width_m": [0.04, 0.05, np.nan],
        }
    )


def test_6d_candidate_ids_frozen() -> None:
    source = _candidate_frame()
    assert assert_frozen_candidate_pool(source, source.copy(), system="raw")[
        "status"
    ] == "PASS"
    changed = source.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="membership, or row order"):
        assert_frozen_candidate_pool(source, changed, system="raw")


def test_6d_candidate_geometry_frozen() -> None:
    source = _candidate_frame()
    changed_geometry = source.copy()
    changed_geometry.loc[0, "geometry_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="geometry"):
        assert_frozen_candidate_pool(source, changed_geometry, system="raw")
    changed_width = source.copy()
    changed_width.loc[1, "gripper_width_m"] = 0.06
    with pytest.raises(ValueError, match="width"):
        assert_frozen_candidate_pool(source, changed_width, system="raw")


def test_primary_6d_hyperparameters_unchanged() -> None:
    assert EXPECTED_HYPERPARAMETERS == {
        "num_leaves": 15,
        "learning_rate": 0.03,
        "n_estimators": 200,
        "min_child_samples": 10,
        "feature_fraction": 0.8,
    }
    assert EXPECTED_EARLY_STOPPING_ROUNDS == 50


def test_adapted_masks_not_reused_across_cv_folds() -> None:
    assert PRIMARY_CONDITION == "oracle_gt_mask"
    assert "adapted" not in PRIMARY_CONDITION


def test_source_artifacts_are_read_only(tmp_path: Path) -> None:
    source = tmp_path / "artifacts" / "graspnet6d" / SOURCE_RUN_ID
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="robustness-suite"):
        _assert_isolated_run_dir(tmp_path, source)

