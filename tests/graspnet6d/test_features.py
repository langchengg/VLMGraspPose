"""Deterministic unit fixtures only; these are not experiment observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graspnet6d.features import (
    StableMissingValueImputer,
    assert_feature_frame_matches_schema,
    assert_no_gt_leakage,
    default_feature_schema,
    feature_schema_sha256,
    load_feature_schema,
    select_feature_columns,
)


def test_feature_table_contains_no_gt_leakage() -> None:
    # Runtime mask support and geometric collision proxies are allowed even in
    # the explicitly named GT-mask oracle condition; raw GT/evaluator fields are not.
    assert assert_no_gt_leakage(
        [
            "native_score",
            "center_in_mask",
            "left_finger_occupancy",
            "collision_proxy_flag",
        ]
    )
    forbidden = (
        "target_object_id",
        "associated_object_id",
        "object_mesh",
        "gt_collision_label",
        "friction_required",
        "evaluator_success",
        "relevance",
        "mask_iou",
        "mask_source",
    )
    for column in forbidden:
        with pytest.raises(ValueError, match="forbidden"):
            assert_no_gt_leakage(["native_score", column])


def test_predeclared_feature_group_selectors_are_label_independent() -> None:
    schema = default_feature_schema()
    all_features = select_feature_columns(schema, "all")
    q_only = select_feature_columns(schema, "q_only")
    assert q_only == (
        "native_score",
        "native_rank",
        "score_to_top1",
        "score_to_previous",
        "score_to_next",
        "score_zscore_within_group",
        "score_percentile",
    )
    assert select_feature_columns(schema, "A1") == q_only
    assert "center_in_mask" not in select_feature_columns(schema, "drop_semantic")
    assert "collision_proxy_flag" not in select_feature_columns(schema, "drop_collision")
    assert "surface_curvature" not in select_feature_columns(
        schema, "drop_local_geometry"
    )
    orientation_drop = select_feature_columns(schema, "drop_orientation")
    assert "rotation_6d_0" in orientation_drop
    assert "approach_vs_gravity_angle" not in orientation_drop
    assert set(q_only).issubset(all_features)
    assert feature_schema_sha256(schema) == feature_schema_sha256(schema)


def test_checked_in_grouped_json_schema_is_supported() -> None:
    schema = load_feature_schema("configs/graspnet6d/feature_schema_6d_v1.json")
    names = [spec.name for spec in schema]
    assert names.count("rotation_6d_0") == 1
    assert select_feature_columns(schema, "A1_q_only") == select_feature_columns(
        schema, "q_only"
    )
    a6 = select_feature_columns(schema, "A6_without_orientation_relative")
    assert "rotation_6d_5" in a6
    assert "roll_consistency" not in a6
    exact = pd.DataFrame(columns=[spec.name for spec in schema])
    assert assert_feature_frame_matches_schema(exact, schema) == tuple(exact.columns)
    with pytest.raises(ValueError, match="exactly match"):
        assert_feature_frame_matches_schema(
            exact.assign(evaluator_success=pd.Series(dtype=float)), schema
        )


def test_train_fitted_imputer_is_stable_and_emits_all_indicators() -> None:
    training = pd.DataFrame(
        {
            "native_score": [0.1, np.nan, 0.9],
            "center_mask_probability": [np.nan, np.nan, np.nan],
        }
    )
    imputer = StableMissingValueImputer(
        all_missing_fill_values={"center_mask_probability": 0.0}
    )
    train_result = imputer.fit_transform(training)
    assert train_result.loc[1, "native_score"] == pytest.approx(0.5)
    assert train_result["center_mask_probability"].eq(0.0).all()
    assert list(train_result.columns) == [
        "native_score",
        "center_mask_probability",
        "native_score__missing",
        "center_mask_probability__missing",
    ]
    validation = pd.DataFrame(
        {
            "native_score": [100.0, np.nan],
            "center_mask_probability": [0.7, np.nan],
        }
    )
    transformed = imputer.transform(validation)
    # Validation's 100 must not refit the training median.
    assert transformed.loc[1, "native_score"] == pytest.approx(0.5)
    assert transformed.loc[1, "native_score__missing"] == 1.0
    assert transformed.loc[0, "center_mask_probability__missing"] == 0.0
    assert np.isfinite(transformed.to_numpy()).all()
    artifact = imputer.artifact()
    assert artifact["fit_scope"] == "training_only"
    assert artifact["fill_values"]["native_score"] == pytest.approx(0.5)
    assert artifact["explicit_all_missing_fill_values"] == {
        "center_mask_probability": 0.0
    }


def test_imputer_rejects_column_drift_nonnumeric_and_infinity() -> None:
    imputer = StableMissingValueImputer().fit(pd.DataFrame({"native_score": [0.5]}))
    with pytest.raises(ValueError, match="columns/order"):
        imputer.transform(pd.DataFrame({"center_in_mask": [1.0]}))
    with pytest.raises(ValueError, match="non-numeric"):
        imputer.transform(pd.DataFrame({"native_score": ["bad"]}))
    with pytest.raises(ValueError, match="infinite"):
        imputer.transform(pd.DataFrame({"native_score": [np.inf]}))
    with pytest.raises(ValueError, match="entirely missing"):
        StableMissingValueImputer().fit(
            pd.DataFrame({"center_mask_probability": [np.nan, np.nan]})
        )
