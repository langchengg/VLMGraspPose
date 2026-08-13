import pandas as pd
import pytest
import torch

from unified_reranking.datasets import (
    FoldPreprocessor,
    build_inference_query_arrays,
    build_query_arrays,
    join_development_features_and_labels,
    with_query_edge_features,
)


def test_query_arrays_pad_without_duplicating_candidates():
    features = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b"],
            "candidate_id": ["a0", "a1", "b0"],
            "native_rank": [1, 2, 1],
            "base_logit": [1.0, 0.0, 0.5],
            "x": [1.0, None, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b"],
            "candidate_id": ["a0", "a1", "b0"],
            "candidate_success": [0, 1, 1],
            "jacquard_margin": [-0.1, 0.2, 0.3],
        }
    )
    joined = join_development_features_and_labels(features, labels)
    preprocessor = FoldPreprocessor.fit(joined, ["x"])
    arrays = build_query_arrays(joined, preprocessor=preprocessor)
    assert arrays.features.shape == (2, 5, 1)
    assert arrays.padding_mask[0].tolist() == [False, False, True, True, True]
    assert arrays.padding_mask[1].tolist() == [False, True, True, True, True]
    assert arrays.candidate_ids == (("a0", "a1"), ("b0",))


def test_label_join_fails_closed_on_missing_candidate():
    features = pd.DataFrame({"sample_id": ["s"], "candidate_id": ["c"], "x": [1.0]})
    labels = pd.DataFrame(
        {
            "sample_id": ["other"],
            "candidate_id": ["c"],
            "candidate_success": [1],
            "jacquard_margin": [0.2],
        }
    )
    with pytest.raises(ValueError, match="exactly cover"):
        join_development_features_and_labels(features, labels)


def test_label_join_restores_or_verifies_frozen_native_rank() -> None:
    features = pd.DataFrame(
        {"sample_id": ["s"], "candidate_id": ["c"], "base_logit": [0.0]}
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "native_rank": [1],
            "candidate_success": [1],
            "jacquard_margin": [0.2],
        }
    )
    joined = join_development_features_and_labels(features, labels)
    assert joined["native_rank"].tolist() == [1]
    with pytest.raises(ValueError, match="native rank mismatch"):
        join_development_features_and_labels(
            features.assign(native_rank=[2]), labels
        )


def test_label_join_accepts_equal_native_rank_across_integer_dtypes() -> None:
    features = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "native_rank": pd.Series([1], dtype="int32"),
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["s"],
            "candidate_id": ["c"],
            "native_rank": pd.Series([1], dtype="int64"),
            "candidate_success": [1],
            "jacquard_margin": [0.2],
        }
    )
    joined = join_development_features_and_labels(features, labels)
    assert joined["native_rank"].tolist() == [1]


def test_preprocessor_rejects_label_as_feature():
    frame = pd.DataFrame({"candidate_success": [0, 1]})
    with pytest.raises(ValueError, match="forbidden"):
        FoldPreprocessor.fit(frame, ["candidate_success"])


def test_query_edges_follow_source_target_candidate_identity() -> None:
    features = pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "candidate_id": ["a0", "a1"],
            "native_rank": [1, 2],
            "base_logit": [1.0, 0.0],
            "x": [0.0, 1.0],
            "candidate_success": [0, 1],
            "jacquard_margin": [-0.5, 0.5],
        }
    )
    arrays = build_query_arrays(
        features, preprocessor=FoldPreprocessor.fit(features, ["x"])
    )
    relations = pd.DataFrame(
        {
            "sample_id": ["a", "a"],
            "source_candidate_id": ["a1", "a0"],
            "target_candidate_id": ["a0", "a1"],
            "distance": [1.0, 3.0],
        }
    )
    relation_preprocessor = FoldPreprocessor.fit(relations, ["distance"])
    result = with_query_edge_features(
        arrays, relations, preprocessor=relation_preprocessor
    )
    assert result.edge_features is not None
    assert result.edge_features.shape == (1, 5, 5, 1)
    assert result.edge_features[0, 0, 1, 0] > result.edge_features[0, 1, 0, 0]
    assert torch.equal(result.subset([0]).edge_features, result.edge_features)


def test_inference_query_builder_requires_no_labels_and_rejects_mixed_table() -> None:
    features = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "candidate_id": ["c0", "c1"],
            "native_rank": [1, 2],
            "base_logit": [1.0, 0.0],
            "x": [2.0, 3.0],
        }
    )
    arrays = build_inference_query_arrays(
        features, preprocessor=FoldPreprocessor.fit(features, ["x"])
    )
    assert arrays.candidate_ids == (("c0", "c1"),)
    assert not bool(arrays.labels.any())
    with pytest.raises(ValueError, match="contains supervision"):
        build_inference_query_arrays(
            features.assign(candidate_success=[0, 1]),
            preprocessor=FoldPreprocessor.fit(features, ["x"]),
        )
    with pytest.raises(ValueError, match="contains supervision"):
        build_inference_query_arrays(
            features.assign(first_positive_rank=[1, 2]),
            preprocessor=FoldPreprocessor.fit(features, ["x"]),
        )
