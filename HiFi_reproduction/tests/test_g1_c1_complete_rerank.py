from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.grasping.g1_c1_safe_rerank.models import (
    NeuralConfig,
    NeuralRanker,
    manual_method_score,
    manual_peak_score,
)
from src.grasping.g1_c1_safe_rerank.crop_model import CropResidualNetwork
from src.grasping.g1_c1_safe_rerank.pools import adapt_source_candidates
from src.grasping.g1_c1_safe_rerank.gate import (
    _top_pair,
    apply_expected_gain_gate,
    expected_gain_sweep,
)
from src.grasping.g1_c1_safe_rerank.evaluation import evaluate_selected_ids, oracle_metrics
from tools.g1_c1_rerank import run_formal_test_once as formal_runner
from tools.g1_c1_rerank import run_local_matrix as local_matrix
from tools.g1_c1_rerank import run_validation_followups as validation_followups
from tools.g1_c1_rerank import build_failure_galleries as failure_galleries


def _source_rows(count: int = 7) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "G1",
                "sample_id": "sample",
                "scene_id": "scene",
                "candidate_id": f"candidate-{rank}",
                "rank": rank,
                "center_x": float(rank),
                "center_y": float(rank + 1),
                "angle_deg": float(rank),
                "width_px": 30.0,
                "height_px": 20.0,
                "score": 1.0 / rank,
                "candidate_metadata_json": (
                    '{"network_quality":0.8,"center_mask_support":0.5,'
                    '"jaw_mask_support":0.4,"source_row":1,"source_column":2}'
                ),
            }
            for rank in range(1, count + 1)
        ]
    )


def test_allnms_is_canonical_and_top5_is_a_derived_identity_subset() -> None:
    source = _source_rows()
    allnms = adapt_source_candidates(source, backend="G1", split="train")
    top5 = adapt_source_candidates(source, backend="G1", split="train", top_k=5)
    assert len(allnms) == 7
    assert len(top5) == 5
    assert top5["stable_candidate_id"].tolist() == allnms.iloc[:5]["stable_candidate_id"].tolist()
    assert top5["candidate_identity_sha256"].tolist() == allnms.iloc[:5]["candidate_identity_sha256"].tolist()


def test_gallery_palette_remains_unique_beyond_tab20() -> None:
    colors = failure_galleries._candidate_colors(64)
    assert colors.shape == (64, 4)
    assert np.unique(np.round(colors, 12), axis=0).shape[0] == 64


@pytest.mark.parametrize("method", ["deepsets", "set_transformer", "candidate_gnn"])
def test_set_models_are_permutation_equivariant_over_100_shuffles(method: str) -> None:
    torch.manual_seed(4)
    config = NeuralConfig(hidden=16, embedding=8, dropout=0.0, residual_alpha=0.5)
    ranker = NeuralRanker(method, ("x0", "x1", "x2"), config=config)
    ranker.model = ranker._make_model().cpu().eval()
    x = torch.randn(6, 3)
    baseline = torch.randn(6)
    ids = ["sample"] * 6
    with torch.no_grad():
        expected = ranker.model(x, baseline, ids)
        for seed in range(100):
            permutation = torch.randperm(6, generator=torch.Generator().manual_seed(seed))
            observed = ranker.model(x[permutation], baseline[permutation], ids)
            restored = torch.empty_like(observed)
            restored[permutation] = observed
            assert torch.allclose(expected, restored, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("method", ["deepsets", "set_transformer", "candidate_gnn"])
def test_set_models_handle_single_and_empty_candidate_sets(method: str) -> None:
    config = NeuralConfig(hidden=16, embedding=8, dropout=0.0)
    ranker = NeuralRanker(method, ("x0", "x1"), config=config)
    ranker.model = ranker._make_model().cpu().eval()
    with torch.no_grad():
        single = ranker.model(torch.zeros(1, 2), torch.zeros(1), ["one"])
        empty = ranker.model(torch.zeros(0, 2), torch.zeros(0), [])
    assert single.shape == (1,) and torch.isfinite(single).all()
    assert empty.shape == (0,)


def test_alpha_zero_is_exact_baseline_pass_through() -> None:
    config = NeuralConfig(hidden=16, embedding=8, dropout=0.0, residual_alpha=0.0)
    ranker = NeuralRanker("residual_mlp_bce", ("x0", "x1"), config=config)
    ranker.model = ranker._make_model().cpu().eval()
    x = torch.randn(5, 2)
    baseline = torch.randn(5)
    with torch.no_grad():
        observed = ranker.model(x, baseline, ["sample"] * 5)
    assert torch.equal(observed, baseline)


def test_bce_objective_gives_each_query_equal_total_mass() -> None:
    config = NeuralConfig(hidden=8, embedding=4, dropout=0.0)
    ranker = NeuralRanker("residual_mlp_bce", ("x",), config=config)
    scores = torch.zeros(5)
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    combined = ranker._objective(scores, labels, ["short", "long", "long", "long", "long"])
    short = ranker._objective(scores[:1], labels[:1], ["short"])
    long = ranker._objective(scores[1:], labels[1:], ["long"] * 4)
    assert float(combined) == pytest.approx((float(short) + float(long)) / 2.0)


def test_directional_manual_evidence_penalizes_risk_and_full_includes_peak() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s", "s", "s"],
            "local_probability_mean": [0.2, 0.5, 0.8],
            "local_probability_max": [0.3, 0.6, 0.9],
            "local_probability_min": [0.1, 0.4, 0.7],
            "local_probability_std": [0.3, 0.2, 0.1],
            "grasp_axis_mask_support": [0.2, 0.5, 0.8],
            "approach_collision_proxy": [0.9, 0.5, 0.1],
            "sweep_minimum_clearance_proxy": [0.1, 0.5, 0.9],
            "gripper_sweep_valid_fraction": [1.0, 1.0, 1.0],
            "sweep_foreground_fraction": [0.5, 0.5, 0.5],
            "background_intrusion_ratio": [0.5, 0.5, 0.5],
        }
    )
    peak_columns = [
        "local_probability_mean",
        "local_probability_max",
        "local_probability_min",
        "local_probability_std",
        "grasp_axis_mask_support",
    ]
    peak = manual_peak_score(frame, peak_columns)
    full = manual_method_score(
        frame, ("clearance", "F2_peak"), peak_columns
    )
    assert np.all(np.diff(peak) > 0)
    assert np.all(np.diff(full) > 0)
    riskier = frame.copy()
    riskier.loc[1, "approach_collision_proxy"] = 0.95
    riskier_full = manual_method_score(
        riskier, ("clearance", "F2_peak"), peak_columns
    )
    assert riskier_full[1] < full[1]


def test_formal_r1_prediction_replays_locked_alpha_without_labels(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "stable_candidate_id": ["a", "b"],
            "candidate_identity_sha256": ["ha", "hb"],
            "source_score_calibrated": [0.6, 0.4],
            "local_probability_mean": [0.0, 1.0],
            "local_probability_max": [0.0, 1.0],
            "local_probability_min": [0.0, 1.0],
            "local_probability_std": [1.0, 0.0],
            "grasp_axis_mask_support": [0.0, 1.0],
        }
    )
    monkeypatch.setattr(formal_runner, "BACKENDS", ("g1",))
    monkeypatch.setattr(formal_runner, "POOLS", ("top5",))
    monkeypatch.setattr(formal_runner, "MANUAL_METHODS", {"r1_peak": ("F2_peak",)})
    monkeypatch.setattr(
        formal_runner,
        "FEATURE_FAMILIES",
        {
            "F2": (
                "local_probability_mean",
                "local_probability_max",
                "local_probability_min",
                "local_probability_std",
                "grasp_axis_mask_support",
            )
        },
    )
    monkeypatch.setattr(formal_runner, "_test_frame", lambda *_: frame.copy())
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {"manual_alphas": {"g1": {"top5": {"r1_peak": 0.1}}}}
        ),
        encoding="utf-8",
    )
    records = formal_runner._write_manual_predictions(
        tmp_path, {"prediction_plan": {"path": str(plan_path)}}
    )
    assert len(records) == 1 and records[0]["alpha"] == 0.1
    scored = pd.read_parquet(
        tmp_path
        / "09_formal_test"
        / "candidate_scores"
        / "g1"
        / "top5"
        / "r1_peak_fixed.parquet"
    )
    assert "candidate_correct" not in scored
    assert np.isfinite(scored["reranker_score"]).all()


def test_r1_alpha_selection_is_explicit_scene_grouped_held_fold_cv() -> None:
    sample_ids = [f"s{index}" for index in range(10)]
    universe = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "scene_id": [f"scene-{index // 2}" for index in range(10)],
            "fold": [index // 2 for index in range(10)],
        }
    )
    train = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "scene_id": universe.loc[index, "scene_id"],
                "fold": universe.loc[index, "fold"],
                "stable_candidate_id": f"{sample_id}-{candidate}",
                "original_rank": candidate + 1,
                "candidate_correct": (candidate == (index % 2)),
            }
            for index, sample_id in enumerate(sample_ids)
            for candidate in range(2)
        ]
    )
    base_scores = np.tile(np.array([1.0, 0.0]), len(sample_ids))
    evidence = np.tile(np.array([0.0, 20.0]), len(sample_ids))
    alpha, records, selection = local_matrix._select_manual_alpha_held_fold_cv(
        train,
        universe,
        base_scores,
        evidence,
        "r1_test",
        alphas=(0.1,),
    )
    _, full_metric = local_matrix._evaluate(
        train,
        universe,
        base_scores + alpha * evidence,
        "r1_test_full_train_equivalence",
    )
    aggregate = records.loc[records["record_type"].eq("aggregate")].iloc[0]
    assert alpha == 0.1
    assert selection["selection_scope"] == "5-fold scene-grouped held-fold Train CV"
    assert records.loc[records["record_type"].eq("held_fold"), "total"].sum() == len(universe)
    assert int(aggregate["net"]) == full_metric["net"]
    assert float(aggregate["delta_j_at_1"]) == full_metric["delta_j_at_1"]
    assert float(aggregate["switch_rate"]) == full_metric["switch_rate"]

    leaked = universe.copy()
    leaked.loc[1, "fold"] = 1
    with pytest.raises(AssertionError, match="scene leakage"):
        local_matrix._select_manual_alpha_held_fold_cv(
            train, leaked, base_scores, evidence, "r1_test", alphas=(0.1,)
        )


def test_outer_fold_calibration_excludes_outer_held_and_cross_fits_outer_train() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "scene_id": f"scene{sample}",
                "stable_candidate_id": f"s{sample}-c{candidate}",
                "fold": sample // 2,
                "original_score": 0.8 if candidate == 0 else 0.2,
                "candidate_correct": int(candidate == sample % 2),
                "source_score_calibrated": 0.5,
            }
            for sample in range(10)
            for candidate in range(2)
        ]
    )
    fit = frame.loc[frame["fold"].ne(4)]
    held = frame.loc[frame["fold"].eq(4)]
    calibrated_fit, calibrated_held, audit = local_matrix._outer_fold_calibration(
        fit, held, outer_fold=4, kind="platt"
    )
    held_label_mutation = held.assign(
        candidate_correct=1 - held["candidate_correct"].astype(int)
    )
    repeated_fit, repeated_held, _ = local_matrix._outer_fold_calibration(
        fit, held_label_mutation, outer_fold=4, kind="platt"
    )
    assert np.array_equal(
        calibrated_fit["source_score_calibrated"],
        repeated_fit["source_score_calibrated"],
    )
    assert np.array_equal(
        calibrated_held["source_score_calibrated"],
        repeated_held["source_score_calibrated"],
    )
    inner_mutation = fit.copy()
    selected_inner = inner_mutation["fold"].eq(0)
    inner_mutation.loc[selected_inner, "candidate_correct"] = (
        1 - inner_mutation.loc[selected_inner, "candidate_correct"].astype(int)
    )
    mutated_fit, _, _ = local_matrix._outer_fold_calibration(
        inner_mutation, held, outer_fold=4, kind="platt"
    )
    assert np.array_equal(
        calibrated_fit.loc[
            calibrated_fit["fold"].eq(0), "source_score_calibrated"
        ].to_numpy(),
        mutated_fit.loc[
            mutated_fit["fold"].eq(0), "source_score_calibrated"
        ].to_numpy(),
    )
    assert audit["outer_held_labels_used"] is False
    assert audit["fit_scope"] == "outer-fit-only nested cross-fit source calibration"


def test_ranker_probability_calibration_accepts_merged_oof_fold_and_cross_fits() -> None:
    oof = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "stable_candidate_id": f"s{sample}-c{candidate}",
                "candidate_correct": int(candidate == sample % 2),
                "reranker_score": float((1 if candidate == sample % 2 else -1) + sample / 100),
                "ranker_probability_outer_pure": 0.8 if candidate == sample % 2 else 0.2,
                "oof_fold": sample // 2,
            }
            for sample in range(10)
            for candidate in range(2)
        ]
    )
    validation = oof.drop(columns="oof_fold").copy()
    calibrator, calibrated_oof, calibrated_validation, audit = (
        validation_followups._ranker_calibration(oof, validation)
    )
    assert calibrator.kind in {"platt", "isotonic"}
    assert audit["folds"] == list(range(5))
    assert np.isfinite(calibrated_oof["ranker_probability"]).all()
    assert np.isfinite(calibrated_validation["ranker_probability"]).all()


def test_outer_pure_ranker_probability_excludes_held_fold_labels() -> None:
    train = pd.DataFrame(
        [
            {
                "sample_id": f"s{sample}",
                "scene_id": f"scene{sample}",
                "stable_candidate_id": f"s{sample}-c{candidate}",
                "candidate_identity_sha256": f"h{sample}-{candidate}",
                "candidate_correct": int(candidate == sample % 2),
                "fold": sample // 2,
            }
            for sample in range(10)
            for candidate in range(2)
        ]
    )
    held = train[
        ["sample_id", "scene_id", "stable_candidate_id", "candidate_identity_sha256", "fold"]
    ].copy()
    held["reranker_score"] = np.where(
        train["stable_candidate_id"].str.endswith("c0"), 0.6, 0.4
    )
    outer_by_seed = []
    for seed in (17, 29, 43):
        rows = []
        for fold in range(5):
            local = train.loc[train["fold"].ne(fold), [
                "sample_id", "scene_id", "stable_candidate_id", "candidate_identity_sha256"
            ]].copy()
            local["outer_fold"] = fold
            local[f"score_seed{seed}"] = np.where(
                local["stable_candidate_id"].str.endswith("c0"),
                0.6 + seed / 10_000,
                0.4 - seed / 10_000,
            )
            rows.append(local)
        outer_by_seed.append(pd.concat(rows, ignore_index=True))
    probability, audit = validation_followups._outer_pure_ranker_probability(
        train, held, outer_by_seed
    )
    mutated = train.copy()
    fold_zero = mutated["fold"].eq(0)
    mutated.loc[fold_zero, "candidate_correct"] = (
        1 - mutated.loc[fold_zero, "candidate_correct"].astype(int)
    )
    repeated, _ = validation_followups._outer_pure_ranker_probability(
        mutated, held, outer_by_seed
    )
    assert np.array_equal(probability[held["fold"].eq(0)], repeated[held["fold"].eq(0)])
    assert len(audit) == 5
    assert all(record["outer_held_labels_used"] is False for record in audit)


def test_gate_does_not_manufacture_challenger_when_reranker_keeps_top1() -> None:
    frame = pd.DataFrame(
        {
            "stable_candidate_id": ["first", "second"],
            "original_rank": [1, 2],
            "reranker_score": [9.0, 1.0],
        }
    )
    baseline, challenger = _top_pair(frame)
    assert baseline["stable_candidate_id"] == "first"
    assert challenger["stable_candidate_id"] == "first"


def test_expected_gain_gate_is_fail_closed_and_never_switch_is_in_sweep() -> None:
    pairs = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "scene_id": ["x", "y"],
            "baseline_candidate_id": ["a0", "b0"],
            "challenger_candidate_id": ["a1", "b1"],
            "challenger_available": [True, True],
            "p_recovered": [0.8, 0.1],
            "p_harmful": [0.1, 0.8],
            "reranker_score_delta": [0.2, 0.2],
            "feature_reliability_minimum": [0.9, 0.9],
            "seed_agreement_fraction": [2 / 3, 2 / 3],
            "baseline_correct": [False, True],
            "challenger_correct": [True, False],
        }
    )
    decisions = apply_expected_gain_gate(
        pairs,
        lambda_h=2.0,
        tau_u=0.0,
        tau_margin=0.0,
        tau_reliability=0.6,
    )
    assert decisions["switch"].tolist() == [True, False]
    sweep, selection = expected_gain_sweep(
        pairs,
        universe=pairs[["sample_id", "scene_id"]],
        lambda_values=[1.0],
        tau_values=[0.0],
        margin_values=[0.0],
        reliability_values=[0.0],
        bootstrap_draws=100,
        bootstrap_seed=7,
    )
    assert "never_switch" in set(sweep["operating_point"])
    assert selection["safe_lcb"]["net"] >= 0


def test_evaluator_fails_closed_on_missing_selected_label() -> None:
    universe = pd.DataFrame({"sample_id": ["s"], "scene_id": ["scene"]})
    selections = pd.DataFrame(
        {
            "sample_id": ["s"],
            "baseline_candidate_id": ["c1"],
            "selected_candidate_id": ["missing"],
        }
    )
    labels = pd.DataFrame(
        {"sample_id": ["s"], "stable_candidate_id": ["c1"], "candidate_correct": [True]}
    )
    with pytest.raises(ValueError, match="labels are missing"):
        evaluate_selected_ids(selections, labels, universe, method="bad")


def test_evaluator_allows_empty_reference_but_not_empty_selection() -> None:
    universe = pd.DataFrame({"sample_id": ["s"], "scene_id": ["scene"]})
    labels = pd.DataFrame(
        {"sample_id": ["s"], "stable_candidate_id": ["c1"], "candidate_correct": [True]}
    )
    valid = pd.DataFrame(
        {"sample_id": ["s"], "baseline_candidate_id": [""], "selected_candidate_id": ["c1"]}
    )
    outcomes, metrics = evaluate_selected_ids(valid, labels, universe, method="router")
    assert not bool(outcomes.iloc[0]["baseline_correct"])
    assert bool(outcomes.iloc[0]["final_correct"])
    assert metrics["recovered"] == 1
    with pytest.raises(ValueError, match="empty selected"):
        evaluate_selected_ids(valid.assign(selected_candidate_id=""), labels, universe, method="router")
    kept_empty = valid.assign(selected_candidate_id="")
    empty_outcomes, empty_metrics = evaluate_selected_ids(
        kept_empty, labels, universe, method="router", allow_empty_selected=True
    )
    assert not bool(empty_outcomes.iloc[0]["final_correct"])
    assert empty_metrics["recovered"] == 0


def test_evaluator_normalizes_identifier_dtypes_and_rejects_nonbinary_truth() -> None:
    universe = pd.DataFrame({"sample_id": [1], "scene_id": [7]})
    selections = pd.DataFrame(
        {"sample_id": ["1"], "baseline_candidate_id": [10], "selected_candidate_id": [10]}
    )
    labels = pd.DataFrame(
        {"sample_id": [1], "stable_candidate_id": ["10"], "candidate_correct": [1]}
    )
    outcomes, metrics = evaluate_selected_ids(
        selections, labels, universe, method="mixed_identifier_types"
    )
    assert bool(outcomes.iloc[0]["final_correct"])
    assert metrics["j_at_1"] == 1.0
    for invalid in (np.nan, 2, -1):
        with pytest.raises(ValueError, match="finite binary"):
            evaluate_selected_ids(
                selections,
                labels.assign(candidate_correct=invalid),
                universe,
                method="invalid_truth",
            )


def test_expected_gain_sweep_uses_complete_universe_denominator() -> None:
    pairs = pd.DataFrame(
        {
            "sample_id": ["s1"], "scene_id": ["a"],
            "baseline_candidate_id": ["b"], "challenger_candidate_id": ["c"],
            "challenger_available": [True], "p_recovered": [0.9], "p_harmful": [0.0],
            "reranker_score_delta": [1.0], "feature_reliability_minimum": [1.0],
            "seed_agreement_fraction": [1.0], "baseline_correct": [False],
            "challenger_correct": [True],
        }
    )
    universe = pd.DataFrame({"sample_id": ["s1", "s2"], "scene_id": ["a", "b"]})
    sweep, _ = expected_gain_sweep(
        pairs, universe=universe, lambda_values=[1.0], tau_values=[0.0],
        margin_values=[0.0], reliability_values=[0.0], bootstrap_draws=100,
        bootstrap_seed=3,
    )
    switched = sweep.loc[sweep["operating_point"].eq("candidate")].iloc[0]
    assert switched["complete_denominator"] == 2
    assert switched["delta_j_at_1"] == 0.5
    assert switched["switch_rate"] == 0.5


def test_union_oracle_uses_pool_rank_and_cannot_exceed_denominator() -> None:
    universe = pd.DataFrame({"sample_id": ["s"], "scene_id": ["scene"]})
    candidates = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "stable_candidate_id": ["g1:a", "c1:b"],
            "original_rank": [1, 1],
            "pool_rank": [1, 2],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "stable_candidate_id": ["g1:a", "c1:b"],
            "candidate_correct": [False, True],
        }
    )
    result = oracle_metrics(candidates, labels, universe)
    assert result["j_at_1_count"] == 0
    assert result["oracle_at_5_count"] == 1


def test_crop_cnn_is_below_parameter_budget_and_candidate_equivariant() -> None:
    torch.manual_seed(9)
    model = CropResidualNetwork(5).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) < 500_000
    crop = torch.rand(7, 4, 64, 64)
    scalar = torch.rand(7, 5)
    base = torch.rand(7)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    with torch.no_grad():
        expected = model(crop, scalar, base)
        observed = model(crop[permutation], scalar[permutation], base[permutation])
    restored = torch.empty_like(observed)
    restored[permutation] = observed
    assert torch.allclose(expected, restored, atol=1e-6, rtol=1e-6)
