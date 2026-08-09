import numpy as np
import torch

from unified_reranking.datasets import QueryArrays
from unified_reranking.training import _edge_features, predict_neural_ranker


class IdentityModel(torch.nn.Module):
    def forward(self, features, native_scores, padding_mask=None):
        return native_scores


def _arrays():
    return QueryArrays(
        sample_ids=("a", "b"),
        candidate_ids=(("a0", "a1"), ("b0",)),
        features=torch.zeros(2, 5, 2),
        labels=torch.zeros(2, 5),
        jacquard_margins=torch.zeros(2, 5),
        native_scores=torch.tensor([[2.0, 1.0, 0, 0, 0], [3.0, 0, 0, 0, 0]]),
        native_ranks=torch.tensor([[1, 2, 0, 0, 0], [1, 0, 0, 0, 0]]),
        padding_mask=torch.tensor([[False, False, True, True, True], [False, True, True, True, True]]),
    )


def test_prediction_does_not_emit_padding_rows():
    predictions = predict_neural_ranker(IdentityModel(), _arrays())
    assert predictions[["sample_id", "candidate_id"]].to_records(index=False).tolist() == [
        ("a", "a0"),
        ("a", "a1"),
        ("b", "b0"),
    ]


def test_directed_edge_feature_is_target_minus_source():
    native = torch.tensor([[1.0, 3.0]])
    edge = _edge_features(native)
    assert edge.shape == (1, 2, 2, 1)
    assert np.isclose(edge[0, 0, 1, 0], 2.0)
    assert np.isclose(edge[0, 1, 0, 0], -2.0)
