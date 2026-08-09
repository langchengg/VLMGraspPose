from __future__ import annotations

import csv
import inspect
import json

import numpy as np
import pytest

from failure_analysis.reranking_v3.analysis import (
    MISSING_CATEGORY,
    NATIVE_COMPARISON_VARIANTS,
    feature_group_permutation_sensitivity,
    leave_one_group_out_table,
    native_enhancement_comparison_table,
    recovered_harmful_distribution,
    subgroup_metrics,
    write_analysis_csv,
    write_analysis_json,
)


def _subgroup_lookup(rows):
    return {(row["subgroup_field"], row["subgroup_value"]): row for row in rows}


def test_subgroup_wilson_metrics_support_rates_delta_and_missing_category():
    reference = np.asarray([1, 0, 1, 0, 0], dtype=bool)
    challenger = np.asarray([1, 1, 0, 0, 1], dtype=bool)
    metadata = {
        "scene_type": ["table", "table", "floor", "floor", None],
        "difficulty": ["easy", "hard", "easy", "hard", "hard"],
    }
    rows = subgroup_metrics(
        analysis_metadata=metadata,
        reference_correct=reference,
        challenger_correct=challenger,
        subgroup_fields=("scene_type",),
        ci_method="wilson",
    )
    lookup = _subgroup_lookup(rows)
    table = lookup[("scene_type", "table")]
    assert table["support"] == 2
    assert table["reference_rate"] == pytest.approx(0.5)
    assert table["challenger_rate"] == pytest.approx(1.0)
    assert table["delta"] == pytest.approx(0.5)
    assert table["recovered"] == 1 and table["harmful"] == 0
    assert lookup[("scene_type", MISSING_CATEGORY)]["support"] == 1
    for row in rows:
        assert 0 <= row["reference_ci_lower"] <= row["reference_ci_upper"] <= 1
        assert 0 <= row["challenger_ci_lower"] <= row["challenger_ci_upper"] <= 1
        assert -1 <= row["delta_ci_lower"] <= row["delta_ci_upper"] <= 1
        assert row["bootstrap_iterations"] is None


def test_cluster_bootstrap_subgroups_are_seeded_and_deterministic():
    kwargs = {
        "analysis_metadata": {"lighting": ["bright", "bright", "dark", "dark", "dark", "bright"]},
        "reference_correct": [1, 0, 1, 0, 1, 0],
        "challenger_correct": [1, 1, 0, 1, 1, 1],
        "cluster_ids": ["f0", "f0", "f1", "f1", "f2", "f3"],
        "ci_method": "bootstrap",
        "bootstrap_iterations": 250,
        "seed": 17,
    }
    first = subgroup_metrics(**kwargs)
    second = subgroup_metrics(**kwargs)
    assert first == second
    assert {row["subgroup_value"] for row in first} == {"bright", "dark"}
    for row in first:
        assert row["bootstrap_iterations"] == 250
        assert row["resampling_unit_count"] >= 1
        assert row["delta_ci_lower"] <= row["delta"] <= row["delta_ci_upper"]


def test_recovered_harmful_distribution_is_long_form_and_subgrouped():
    rows = recovered_harmful_distribution(
        reference_correct=[1, 0, 1, 0],
        challenger_correct=[1, 1, 0, 0],
        analysis_metadata={"location": ["left", "left", "right", "right"]},
        subgroup_fields=("location",),
    )
    overall = {row["outcome"]: row for row in rows if row["subgroup_field"] == "__all__"}
    assert overall["recovered"]["count"] == 1
    assert overall["harmful"]["count"] == 1
    assert overall["stable_correct"]["count"] == 1
    assert overall["stable_incorrect"]["count"] == 1
    assert overall["recovered"]["outcome_changing_share"] == pytest.approx(0.5)
    assert len([row for row in rows if row["subgroup_field"] == "location"]) == 8

    empty = recovered_harmful_distribution(reference_correct=[], challenger_correct=[])
    assert len(empty) == 4
    assert all(row["support"] == 0 and row["rate"] is None for row in empty)


def test_permutation_sensitivity_is_deterministic_masked_and_prediction_only():
    sample_count, candidate_count = 4, 3
    mask = np.tile(np.asarray([True, True, False]), (sample_count, 1))
    signal = np.tile(np.asarray([[[1.0], [0.0], [99.0]]]), (sample_count, 1, 1))
    sample_context = np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float64)
    true_selected = np.zeros(sample_count, dtype=np.int64)
    callback_calls = []

    def predict(features, candidate_mask):
        assert set(features) == {"signal", "sample_context"}
        assert not features["signal"].flags.writeable
        assert not candidate_mask.flags.writeable
        assert np.all(features["signal"][:, 2, 0] == 99.0)
        callback_calls.append(1)
        context = features["sample_context"][:, 0, None]
        scores = features["signal"][..., 0] + context * np.asarray([0.1, -0.1, 0.0])
        return np.where(candidate_mask, scores, -1000.0)

    def accuracy(predictions):
        return float(np.mean(np.argmax(predictions, axis=1) == true_selected))

    kwargs = {
        "feature_groups": {"signal": signal, "sample_context": sample_context},
        "group_semantics": {"signal": "candidate", "sample_context": "sample"},
        "prediction_callback": predict,
        "metric_callback": accuracy,
        "candidate_mask": mask,
        "seed": 23,
    }
    first = feature_group_permutation_sensitivity(**kwargs)
    second = feature_group_permutation_sensitivity(**kwargs)
    assert first == second
    assert first["baseline_metric"] == 1.0
    assert first["analysis_metadata_passed_to_prediction"] is False
    assert first["correctness_passed_to_prediction"] is False
    by_group = {row["feature_group"]: row for row in first["groups"]}
    assert by_group["signal"]["permutation_semantics"] == "candidate"
    assert by_group["signal"]["permuted_metric"] == 0.0
    assert by_group["signal"]["delta"] == -1.0
    assert by_group["signal"]["performance_drop"] == 1.0
    assert by_group["signal"]["selection_change_rate"] == 1.0
    assert by_group["sample_context"]["permutation_semantics"] == "sample"
    assert len(callback_calls) == 2 * (1 + len(kwargs["feature_groups"]))

    signature = inspect.signature(feature_group_permutation_sensitivity)
    assert "analysis_metadata" not in signature.parameters
    assert "correct" not in " ".join(signature.parameters)


@pytest.mark.parametrize("forbidden", ["ground_truth", "candidate_label", "j1_success", "matched_gt"])
def test_permutation_predictor_rejects_evaluation_feature_names(forbidden):
    with pytest.raises(ValueError, match="forbidden inference"):
        feature_group_permutation_sensitivity(
            feature_groups={forbidden: np.zeros((2, 2, 1))},
            group_semantics={forbidden: "candidate"},
            prediction_callback=lambda features, mask: features[forbidden][..., 0],
            metric_callback=lambda predictions: predictions.mean(),
        )


def test_leave_one_out_and_native_comparison_tables_are_explicit():
    leave_one_out = leave_one_group_out_table(
        full_metrics={"j_at_1": 0.8, "harmful": 2},
        leave_one_out_results={
            "text": {"j_at_1": 0.7, "harmful": 3},
            "latent": {"j_at_1": 0.75, "harmful": 1},
        },
        metric_names=("j_at_1", "harmful"),
    )
    assert [row["configuration"] for row in leave_one_out] == ["all_groups", "without_latent", "without_text"]
    assert leave_one_out[1]["delta_vs_full_j_at_1"] == pytest.approx(-0.05)
    assert leave_one_out[1]["delta_vs_full_harmful"] == pytest.approx(-1)

    results = {
        name: {"j_at_1": 0.70 + 0.01 * index, "harmful": 5 - index}
        for index, name in enumerate(NATIVE_COMPARISON_VARIANTS)
    }
    comparison = native_enhancement_comparison_table(
        variant_results=results,
        metric_names=("j_at_1", "harmful"),
    )
    assert [row["variant"] for row in comparison] == list(NATIVE_COMPARISON_VARIANTS)
    assert comparison[0]["is_native"] is True
    assert comparison[0]["delta_vs_native_j_at_1"] == 0
    assert comparison[-1]["delta_vs_native_j_at_1"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="missing comparison variants"):
        native_enhancement_comparison_table(variant_results={"native": {"j_at_1": 0.7}})


def test_machine_readable_json_csv_helpers_are_atomic_and_strict(tmp_path):
    json_path = tmp_path / "analysis.json"
    write_analysis_json(
        json_path,
        {"array": np.asarray([1, 2]), "score": np.float32(0.5), "ok": np.bool_(True)},
    )
    assert json.loads(json_path.read_text()) == {"array": [1, 2], "ok": True, "score": 0.5}
    with pytest.raises(FileExistsError):
        write_analysis_json(json_path, {})
    with pytest.raises(ValueError, match="NaN"):
        write_analysis_json(tmp_path / "bad.json", {"bad": np.nan})

    csv_path = tmp_path / "analysis.csv"
    write_analysis_csv(
        csv_path,
        [
            {"group": "native", "score": np.float32(0.7), "details": {"seed": 1}},
            {"group": "rgbd", "score": 0.8, "extra": [1, 2]},
        ],
    )
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["details"] == '{"seed":1}'
    assert rows[1]["extra"] == "[1,2]"
    with pytest.raises(FileExistsError):
        write_analysis_csv(csv_path, [])


def test_analysis_boundaries_fail_closed():
    with pytest.raises(ValueError, match="equal length"):
        subgroup_metrics(
            analysis_metadata={"x": [1]},
            reference_correct=[1],
            challenger_correct=[1, 0],
        )
    with pytest.raises(ValueError, match="0/1"):
        subgroup_metrics(
            analysis_metadata={"x": [1, 2]},
            reference_correct=[1, 2],
            challenger_correct=[1, 0],
        )
    with pytest.raises(ValueError, match="length"):
        subgroup_metrics(
            analysis_metadata={"x": [1]},
            reference_correct=[1, 0],
            challenger_correct=[1, 0],
        )
    assert subgroup_metrics(
        analysis_metadata={"x": []},
        reference_correct=[],
        challenger_correct=[],
    ) == []

    with pytest.raises(ValueError, match="finite numeric"):
        feature_group_permutation_sensitivity(
            feature_groups={"native": np.ones((2, 2, 1))},
            group_semantics={"native": "candidate"},
            prediction_callback=lambda features, mask: np.full((2, 2), np.nan),
            metric_callback=lambda predictions: 0.0,
        )
    all_masked = feature_group_permutation_sensitivity(
        feature_groups={"native": np.ones((2, 2, 1))},
        group_semantics={"native": "candidate"},
        candidate_mask=np.zeros((2, 2), dtype=bool),
        prediction_callback=lambda features, mask: features["native"][..., 0],
        metric_callback=lambda predictions: predictions.mean(),
    )
    assert all_masked["groups"][0]["selection_change_rate"] is None
    assert all_masked["groups"][0]["permutation"] == [[], []]
    with pytest.raises(ValueError, match="must not be empty"):
        feature_group_permutation_sensitivity(
            feature_groups={"native": np.ones((2, 2, 1))},
            group_semantics={"native": "candidate"},
            prediction_callback=lambda features, mask: np.empty((2, 0)),
            metric_callback=lambda predictions: 0.0,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        leave_one_group_out_table(full_metrics={"j_at_1": 1.0}, leave_one_out_results={})
