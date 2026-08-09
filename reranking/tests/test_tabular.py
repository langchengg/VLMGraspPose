from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reranking.models.tabular import (
    HIST_GRADIENT_BOOSTING_FALLBACK_REASON,
    HistGradientBoostingRanker,
    LinearPairwiseRankNet,
    LinearResidualBCE,
    LogisticRegressionRanker,
    PointwiseRanker,
    RandomForestRanker,
    TrainOnlyScoreCalibrator,
    XGBoostLambdaMARTRanker,
    XGBoostRanker,
    assert_no_forbidden_columns,
    build_pairwise_batch,
    derive_q_features,
    make_tabular_ranker,
    scan_forbidden_columns,
    stable_rank_order,
)
import reranking.models.tabular as tabular_module


def _training_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # Four queries with three candidates each.  The positive position changes,
    # preventing every model from succeeding by memorizing a fixed row index.
    labels = np.asarray(
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0], dtype=np.int64
    )
    query_ids = np.repeat(["q0", "q1", "q2", "q3"], 3)
    baseline = np.asarray(
        [0.75, 0.55, 0.25, 0.70, 0.65, 0.30, 0.80, 0.60, 0.35, 0.72, 0.50, 0.20]
    )
    features = pd.DataFrame(
        {
            "mask_support": 0.1 + 0.8 * labels,
            "width_match": np.asarray(
                [0.9, 0.2, 0.3, 0.1, 0.8, 0.2, 0.3, 0.1, 0.9, 0.8, 0.4, 0.2]
            ),
            "q_logit": np.log(np.clip(baseline, 1e-6, 1 - 1e-6))
            - np.log1p(-np.clip(baseline, 1e-6, 1 - 1e-6)),
        }
    )
    return features, labels, query_ids, baseline


def test_forbidden_column_scanner_fails_closed_without_substring_false_positive() -> None:
    assert scan_forbidden_columns(["height", "target_probability", "q_logit"]) == ()
    rejected = scan_forbidden_columns(
        [
            "candidate_gt_iou",
            "trainingLabel",
            "candidate_correctness",
            "successful_grasp",
            "groundtruthMask",
            "target_gt_mask",
        ]
    )
    assert set(rejected) == {
        "candidate_gt_iou",
        "trainingLabel",
        "candidate_correctness",
        "successful_grasp",
        "groundtruthMask",
        "target_gt_mask",
    }
    with pytest.raises(ValueError, match="forbidden"):
        LogisticRegressionRanker().fit(
            pd.DataFrame({"candidate_correct": [0.0, 1.0]}),
            [0, 1],
            query_ids=["a", "a"],
        )
    assert_no_forbidden_columns(["grasp_axis_mask_support", "candidate_count"])


def test_q_features_are_finite_and_ties_are_deterministic() -> None:
    source = pd.DataFrame(
        {
            "query_id": ["a", "a", "a", "b"],
            "candidate_id": ["c2", "c1", "c0", "only"],
            "q_raw": [0.8, 0.8, 0.2, 0.0],
        }
    )
    derived = derive_q_features(source)
    feature_names = [
        "q_clipped",
        "q_log",
        "q_logit",
        "rank_percentile",
        "delta_q_top1",
        "delta_q_previous",
        "delta_q_next",
        "q_zscore_within_query",
        "q_percentile_within_query",
        "top1_margin",
        "top2_margin",
        "score_entropy",
        "score_concentration",
        "score_prominence",
    ]
    assert np.isfinite(derived[feature_names].to_numpy()).all()
    assert derived.loc[0, "rank_percentile"] == derived.loc[1, "rank_percentile"]
    assert derived.loc[0, "top1_margin"] == 0.0
    assert derived.loc[0, "score_prominence"] == 0.0
    assert derived.loc[1, "score_prominence"] == 0.0
    assert stable_rank_order([1.0, 1.0, 0.5], ["c2", "c1", "c0"]).tolist() == [1, 0, 2]


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_is_fit_only_on_supplied_training_rows(method: str) -> None:
    train_scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    train_labels = np.asarray([0, 0, 1, 1])
    calibrator = TrainOnlyScoreCalibrator(method).fit(
        train_scores,
        train_labels,
        fit_sample_ids=["train0", "train1", "train2", "train3"],
    )
    before = calibrator.predict_proba([-100.0, 100.0])
    # Prediction rows and their (hypothetical) labels are never supplied to
    # fit, and a transform cannot mutate fit provenance or parameters.
    held_out_labels = np.asarray([1, 0])
    after = calibrator.predict_proba([-100.0, 100.0])
    assert held_out_labels.tolist() == [1, 0]
    assert np.array_equal(before, after)
    assert calibrator.metadata["fit_sample_ids"] == [
        "train0",
        "train1",
        "train2",
        "train3",
    ]
    assert calibrator.metadata["n_fit_samples"] == 4
    assert np.all((before > 0.0) & (before < 1.0))


def test_pair_builder_never_crosses_queries_and_skips_ineligible_queries() -> None:
    features = pd.DataFrame({"signal": [3.0, 2.0, 1.0, 8.0, 7.0, 5.0, 4.0]})
    labels = np.asarray([1, 0, 0, 0, 0, 1, 1])
    queries = np.asarray(["eligible"] * 3 + ["no_positive"] * 2 + ["all_positive"] * 2)
    batch = build_pairwise_batch(features, labels, query_ids=queries)

    assert batch.n_pairs == 2
    assert batch.skipped_no_positive == ("no_positive",)
    assert batch.skipped_no_negative == ("all_positive",)
    for positive, negative, pair_query in zip(
        batch.positive_indices, batch.negative_indices, batch.query_ids
    ):
        assert queries[positive] == queries[negative] == pair_query
        assert labels[positive] == 1
        assert labels[negative] == 0
    assert sum(batch.weights) == pytest.approx(1.0)


def test_pair_weights_are_equal_per_query_even_with_different_pair_counts() -> None:
    features = pd.DataFrame({"signal": np.arange(8, dtype=float)})
    labels = np.asarray([1, 0, 1, 0, 0, 0, 1, 0])
    queries = np.asarray(["two_pairs"] * 3 + ["three_pairs"] * 4 + ["one_pair"])
    # Put the last row into one of the earlier queries so the query is eligible.
    queries[-1] = "one_pair"
    features = pd.concat([features, pd.DataFrame({"signal": [9.0]})], ignore_index=True)
    labels = np.append(labels, 1)
    queries = np.append(queries, "one_pair")
    batch = build_pairwise_batch(features, labels, query_ids=queries)
    totals = {
        query: float(batch.weights[np.asarray(batch.query_ids) == query].sum())
        for query in batch.pairs_per_query
    }
    assert totals
    assert len(set(round(value, 12) for value in totals.values())) == 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LogisticRegressionRanker(max_iter=100),
        lambda: RandomForestRanker(n_estimators=20),
        lambda: HistGradientBoostingRanker(max_iter=30),
    ],
)
def test_pointwise_models_have_common_finite_interface(factory) -> None:
    features, labels, queries, baseline = _training_data()
    model = factory().fit(
        features,
        labels,
        query_ids=queries,
        baseline_scores=baseline,
        sample_ids=[f"train-{index}" for index in range(len(labels))],
    )
    scores = model.predict_scores(features, query_ids=queries, baseline_scores=baseline)
    importance = model.feature_importance()
    assert scores.shape == (len(labels),)
    assert np.isfinite(scores).all()
    assert set(importance) == set(features.columns)
    assert np.isfinite(list(importance.values())).all()
    assert model.metadata["fit_scope"] == "rows explicitly supplied to fit only"


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: PointwiseRanker("logistic", penalty=value),
        lambda value: LogisticRegressionRanker(penalty=value),
    ],
)
@pytest.mark.parametrize(
    ("penalty", "solver"),
    [("l1", "liblinear"), ("l2", "lbfgs")],
)
def test_logistic_pointwise_rankers_support_explicit_l1_l2_with_valid_solvers(
    factory, penalty: str, solver: str
) -> None:
    features, labels, queries, baseline = _training_data()
    model = factory(penalty).fit(
        features,
        labels,
        query_ids=queries,
        baseline_scores=baseline,
    )

    assert model.estimator_.solver == solver
    assert model.metadata["penalty"] == penalty
    assert model.metadata["solver"] == solver
    assert model.metadata["C"] == 1.0
    assert model.metadata["regularization"] == {
        "penalty": penalty,
        "inverse_strength_C": 1.0,
        "solver": solver,
        "sklearn_parameter": model.sklearn_regularization_parameter_,
    }
    assert np.isfinite(model.predict_scores(features, query_ids=queries)).all()


def test_logistic_pointwise_ranker_preserves_l2_default_and_validates_options() -> None:
    model = LogisticRegressionRanker()
    assert model.penalty == "l2"
    assert model._build_estimator().solver == "lbfgs"

    with pytest.raises(ValueError, match="penalty must be 'l1' or 'l2'"):
        LogisticRegressionRanker(penalty="elasticnet")
    with pytest.raises(ValueError, match="strictly positive"):
        LogisticRegressionRanker(c=0.0)
    with pytest.raises(ValueError, match="only configurable for the logistic"):
        PointwiseRanker("random_forest", penalty="l1")


def test_hist_gradient_boosting_is_explicitly_pointwise_not_lambdamart() -> None:
    features, labels, queries, _ = _training_data()
    model = HistGradientBoostingRanker(max_iter=10).fit(
        features, labels, query_ids=queries
    )
    assert model.metadata["lambda_mart"] is False
    assert model.metadata["is_learning_to_rank"] is False
    assert model.metadata["ranking_backend_unavailable_reason"] == (
        HIST_GRADIENT_BOOSTING_FALLBACK_REASON
    )
    assert "not LambdaMART" in model.metadata["ranking_backend_unavailable_reason"]


class _FakeXGBRanker:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.feature_importances_ = np.asarray([2.0, 1.0, 0.0])
        type(self).last_instance = self

    def fit(self, features, labels, *, qid):
        self.fit_features = np.asarray(features)
        self.fit_labels = np.asarray(labels)
        self.fit_qid = np.asarray(qid)
        return self

    def predict(self, features):
        return np.asarray(features)[:, 0]


@pytest.mark.parametrize(
    ("objective", "lambda_mart_kind"),
    [
        ("rank:ndcg", "ndcg_scaled_lambda_rank"),
        ("rank:pairwise", "unscaled_pairwise_logistic"),
    ],
)
def test_xgboost_ranker_uses_sorted_qid_and_truthful_ranking_metadata(
    monkeypatch, objective: str, lambda_mart_kind: str
) -> None:
    fake_xgboost = SimpleNamespace(XGBRanker=_FakeXGBRanker, __version__="test-double")
    monkeypatch.setattr(tabular_module, "_load_xgboost_module", lambda: fake_xgboost)
    features, labels, queries, _ = _training_data()
    interleaved = np.asarray([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    features = features.iloc[interleaved].reset_index(drop=True)
    labels = labels[interleaved]
    queries = queries[interleaved]

    model = XGBoostRanker(
        objective=objective, n_estimators=7, random_state=13
    ).fit(
        features,
        labels,
        query_ids=queries,
        sample_ids=[f"row-{index}" for index in range(len(labels))],
    )
    backend = _FakeXGBRanker.last_instance

    assert backend.init_kwargs["objective"] == objective
    assert backend.init_kwargs["n_jobs"] == 1
    assert backend.init_kwargs["random_state"] == 13
    assert np.all(backend.fit_qid[:-1] <= backend.fit_qid[1:])
    assert np.bincount(backend.fit_qid).tolist() == [3, 3, 3, 3]
    assert model.metadata["backend"] == "xgboost.XGBRanker"
    assert model.metadata["backend_version"] == "test-double"
    assert model.metadata["is_learning_to_rank"] is True
    assert model.metadata["is_pointwise"] is False
    assert model.metadata["lambda_mart"] is True
    assert model.metadata["lambda_mart_kind"] == lambda_mart_kind
    assert model.metadata["query_group_parameter"] == "qid"
    assert model.metadata["query_group_sizes"] == [3, 3, 3, 3]
    assert model.metadata["fallback_backend"] is None
    assert set(model.feature_importance()) == set(features.columns)
    assert sum(model.feature_importance().values()) == pytest.approx(1.0)
    assert np.array_equal(model.predict_scores(features), features["mask_support"])


def test_xgboost_ranker_alias_validation_and_missing_dependency(monkeypatch) -> None:
    assert XGBoostLambdaMARTRanker is XGBoostRanker
    assert isinstance(make_tabular_ranker("lambdamart"), XGBoostRanker)
    with pytest.raises(ValueError, match="rank:ndcg.*rank:pairwise"):
        XGBoostRanker(objective="binary:logistic")

    def unavailable():
        raise RuntimeError(
            "XGBoostRanker requires the optional 'xgboost' package; "
            "no pointwise fallback will be substituted"
        )

    monkeypatch.setattr(tabular_module, "_load_xgboost_module", unavailable)
    features, labels, queries, _ = _training_data()
    with pytest.raises(RuntimeError, match="no pointwise fallback"):
        XGBoostRanker().fit(features, labels, query_ids=queries)


@pytest.mark.parametrize(
    "model",
    [
        LinearResidualBCE(max_iter=100),
        LinearPairwiseRankNet(max_iter=100),
    ],
)
def test_linear_residual_and_ranknet_scores_are_finite_and_train_only(model) -> None:
    features, labels, queries, baseline = _training_data()
    train_ids = [f"train-{index}" for index in range(len(labels))]
    fitted = model.fit(
        features,
        labels,
        query_ids=queries,
        baseline_scores=baseline,
        sample_ids=train_ids,
    )
    assert np.isfinite(
        fitted.predict_scores(features, query_ids=queries, baseline_scores=baseline)
    ).all()
    assert fitted.metadata["calibration"]["fit_sample_ids"] == train_ids
    assert set(fitted.feature_importance()) == set(features.columns)
