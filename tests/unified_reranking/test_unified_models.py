from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
from torch import nn

from unified_reranking.losses import (
    all_pairs_ranknet_loss,
    jacquard_margin_ranknet_loss,
    multi_positive_listwise_loss,
    query_equal_bce_loss,
)
from unified_reranking.models import (
    CompleteGraphGNNResidualScorer,
    DeepSetsResidualScorer,
    LightGBMLambdaRank,
    LinearResidualScorer,
    ResidualMLPScorer,
    SetTransformerResidualScorer,
    contiguous_group_order,
)


def _candidate_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260808)
    features = torch.randn(2, 5, 7, generator=generator)
    native = torch.randn(2, 5, generator=generator)
    padding = torch.tensor(
        [[False, False, False, False, False], [False, False, False, True, True]]
    )
    return features, native, padding


def test_residual_mlp_has_exact_controlled_architecture() -> None:
    model = ResidualMLPScorer(7)
    assert [type(layer) for layer in model.network] == [
        nn.LayerNorm,
        nn.Linear,
        nn.GELU,
        nn.Dropout,
        nn.Linear,
        nn.GELU,
        nn.Linear,
    ]
    assert model.network[1].in_features == 7
    assert model.network[1].out_features == 64
    assert model.network[3].p == pytest.approx(0.1)
    assert model.network[4].in_features == 64
    assert model.network[4].out_features == 32
    assert model.network[6].in_features == 32
    assert model.network[6].out_features == 1


def test_linear_residual_is_single_affine_layer() -> None:
    model = LinearResidualScorer(7)
    assert isinstance(model.linear, nn.Linear)
    assert model.linear.in_features == 7
    assert model.linear.out_features == 1
    assert sum(parameter.numel() for parameter in model.parameters()) == 8


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ResidualMLPScorer(7, alpha=0.0),
        lambda: LinearResidualScorer(7, alpha=0.0),
        lambda: DeepSetsResidualScorer(7, alpha=0.0),
        lambda: SetTransformerResidualScorer(7, alpha=0.0, num_blocks=1),
    ],
)
def test_alpha_zero_is_bit_exact_native_control(factory) -> None:
    features, native, padding = _candidate_batch()
    model = factory()
    model.train()
    assert torch.equal(model(features, native, padding), native)


def test_gnn_alpha_zero_is_bit_exact_native_control() -> None:
    features, native, padding = _candidate_batch()
    edges = torch.randn(2, 5, 5, 3)
    model = CompleteGraphGNNResidualScorer(7, 3, alpha=0.0)
    assert torch.equal(
        model(features, native, padding, edge_features=edges), native
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ResidualMLPScorer(7),
        lambda: DeepSetsResidualScorer(7),
        lambda: SetTransformerResidualScorer(7, num_blocks=2),
    ],
)
def test_padded_candidates_are_exact_native_passthrough(factory) -> None:
    features, native, padding = _candidate_batch()
    model = factory().eval()
    output = model(features, native, padding)
    assert output.shape == native.shape
    assert torch.equal(output[padding], native[padding])
    assert torch.isfinite(output).all()


def test_set_transformer_supports_fully_padded_batch_row() -> None:
    features, native, padding = _candidate_batch()
    padding[1] = True
    model = SetTransformerResidualScorer(7).eval()
    output = model(features, native, padding)
    assert torch.equal(output[1], native[1])
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ResidualMLPScorer(7),
        lambda: DeepSetsResidualScorer(7),
        lambda: SetTransformerResidualScorer(7, num_blocks=1),
        lambda: SetTransformerResidualScorer(7, num_blocks=2),
    ],
)
def test_candidate_scorers_are_permutation_equivariant(factory) -> None:
    features, native, padding = _candidate_batch()
    permutation = torch.tensor([3, 0, 4, 1, 2])
    model = factory().eval()
    expected = model(features, native, padding)[:, permutation]
    actual = model(
        features[:, permutation], native[:, permutation], padding[:, permutation]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_complete_graph_gnn_contract_and_permutation_equivariance() -> None:
    features, native, padding = _candidate_batch()
    edges = torch.randn(2, 5, 5, 4, generator=torch.Generator().manual_seed(91))
    permutation = torch.tensor([4, 1, 3, 0, 2])
    model = CompleteGraphGNNResidualScorer(7, 4).eval()
    assert model.node_hidden_dim == 64
    assert model.edge_hidden_dim == 32
    assert len(model.message_passes) == 2
    expected = model(
        features, native, padding, edge_features=edges
    )[:, permutation]
    permuted_edges = edges[:, permutation][:, :, permutation]
    actual = model(
        features[:, permutation],
        native[:, permutation],
        padding[:, permutation],
        edge_features=permuted_edges,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual[padding[:, permutation]], native[:, permutation][padding[:, permutation]])


def test_query_equal_bce_averages_queries_not_candidates() -> None:
    scores = torch.tensor([[0.0, 0.0, 99.0], [2.0, -1.0, 0.5]])
    labels = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    padding = torch.tensor([[False, False, True], [False, False, False]])
    first = torch.nn.functional.binary_cross_entropy_with_logits(
        scores[0, :2], labels[0, :2]
    )
    second = torch.nn.functional.binary_cross_entropy_with_logits(
        scores[1], labels[1]
    )
    assert query_equal_bce_loss(scores, labels, padding) == pytest.approx(
        ((first + second) / 2).item()
    )


def test_ranknet_uses_all_positive_negative_pairs_and_equal_queries() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.0], [0.5, -0.5, 9.0]])
    labels = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    padding = torch.tensor([[False, False, False], [False, False, True]])
    first = (
        torch.nn.functional.softplus(torch.tensor(-2.0))
        + torch.nn.functional.softplus(torch.tensor(-1.0))
    ) / 2
    second = torch.nn.functional.softplus(torch.tensor(-1.0))
    assert all_pairs_ranknet_loss(scores, labels, padding) == pytest.approx(
        ((first + second) / 2).item()
    )


def test_multi_positive_listwise_temperature_formula_and_exclusions() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.0], [4.0, 3.0, 2.0]])
    labels = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    temperature = 0.5
    expected = torch.logsumexp(scores[0] / temperature, dim=0) - torch.logsumexp(
        scores[0, :2] / temperature, dim=0
    )
    assert multi_positive_listwise_loss(
        scores, labels, temperature=temperature
    ) == pytest.approx(expected.item())


def test_jacquard_margin_ranknet_matches_weighted_pair_formula() -> None:
    scores = torch.tensor([[1.5, 0.5, -0.5]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    margins = torch.tensor([[0.8, 0.2, -0.7]])
    beta = 2.0
    weights = torch.tensor([1 + beta * 0.6, 1 + beta * 1.5])
    losses = torch.nn.functional.softplus(-torch.tensor([1.0, 2.0]))
    expected = (weights * losses).mean()
    actual = jacquard_margin_ranknet_loss(
        scores, labels, margins, beta=beta
    )
    assert actual == pytest.approx(expected.item())
    with pytest.raises(ValueError, match="clipped"):
        jacquard_margin_ranknet_loss(scores, labels, margins + 1.0)


@pytest.mark.parametrize(
    "loss",
    [
        lambda scores, labels: all_pairs_ranknet_loss(scores, labels),
        lambda scores, labels: multi_positive_listwise_loss(scores, labels),
        lambda scores, labels: jacquard_margin_ranknet_loss(
            scores, labels, torch.zeros_like(scores)
        ),
    ],
)
def test_ranking_losses_return_differentiable_zero_without_mixed_query(loss) -> None:
    scores = torch.tensor([[1.0, 0.0], [2.0, 3.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    result = loss(scores, labels)
    assert result.item() == 0.0
    result.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_contiguous_group_order_is_stable_and_complete() -> None:
    order, groups = contiguous_group_order(["q2", "q1", "q2", "q1", "q3"], length=5)
    np.testing.assert_array_equal(order, [0, 2, 1, 3, 4])
    assert groups == [2, 2, 1]


def test_lightgbm_lambdarank_locked_contract_and_fit() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.2],
            [1.1, 1.2],
            [0.2, 0.1],
            [1.2, 1.1],
        ]
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    query_ids = ["a", "b", "a", "b", "c", "c"]
    model = LightGBMLambdaRank(
        seed=20260808, n_estimators=5, min_child_samples=1, num_leaves=3
    ).fit(features, labels, query_ids)
    parameters = model.model.get_params()
    assert parameters["objective"] == "lambdarank"
    assert parameters["label_gain"] == [0, 1]
    assert model.artifact()["eval_at"] == [1]
    assert parameters["force_col_wise"] is True
    assert parameters["deterministic"] is True
    assert parameters["device_type"] == "cpu"
    assert parameters["n_jobs"] == 1
    assert model.group_sizes_ == [2, 2, 2]
    np.testing.assert_array_equal(model.training_order_, [0, 2, 1, 3, 4, 5])
    prediction = model.predict(features)
    assert prediction.shape == (6,)
    assert np.isfinite(prediction).all()
    assert model.artifact()["integer_binary_labels"] is True


def test_locked_lightgbm_pickle_uses_native_restore_before_torch(tmp_path) -> None:
    script = textwrap.dedent(
        """
        import pickle
        import sys
        from pathlib import Path

        import numpy as np

        from tools.unified_reranking.apply_locked_matrix_cell import (
            _load_native_lightgbm_ranker,
        )
        from unified_reranking.models import LightGBMLambdaRank

        features = np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [0.1, 0.2], [1.1, 1.2]],
            dtype=float,
        )
        model = LightGBMLambdaRank(
            seed=7, n_estimators=3, min_child_samples=1, num_leaves=3
        ).fit(features, [0, 1, 0, 1], ["a", "a", "b", "b"])
        path = Path(sys.argv[1])
        with path.open("wb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        restored = _load_native_lightgbm_ranker(path)
        np.testing.assert_allclose(restored.predict(features), model.predict(features))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "model.pkl")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lightgbm_rejects_nonbinary_labels() -> None:
    model = LightGBMLambdaRank(seed=1, n_estimators=1)
    with pytest.raises(ValueError, match="binary"):
        model.fit(np.ones((3, 2)), [0, 1, 2], ["q", "q", "q"])
