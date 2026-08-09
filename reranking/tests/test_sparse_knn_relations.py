from __future__ import annotations

import copy
import json

import pytest
import torch

import reranking.models.set_models as set_models
import reranking.train as train_module
from reranking.models.set_models import (
    CandidateGNNScorer,
    PAIRWISE_EDGE_RELATION_FIELDS,
    build_indexed_pairwise_edge_relations,
    build_pairwise_edge_relations,
)
from reranking.train import (
    CandidateSetExample,
    TrainingConfig,
    fit_neural_ranker,
    make_candidate_dataloader,
    pad_candidate_sets,
)


def _inputs(
    *, seed: int = 17
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(3, 9, 5, generator=generator)
    baseline = torch.randn(3, 9, generator=generator)
    raw = torch.randn(3, 9, 10, generator=generator)
    raw[..., 3] = 0.2 + 4.0 * torch.rand(3, 9, generator=generator)
    raw[..., 8] = torch.rand(3, 9, generator=generator)
    raw[..., 9] = torch.randint(-1, 4, (3, 9), generator=generator).to(torch.float32)
    padding = torch.tensor(
        [
            [False, False, False, False, True, True, True, True, True],
            [False, False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    raw[padding] = torch.nan
    return features, baseline, raw, padding


def _knn_edges(raw: torch.Tensor, padding: torch.Tensor, *, k: int = 3):
    model = CandidateGNNScorer(5, edge_dim=0, graph_type="knn", k=k)
    return model._complete_or_knn_edges(~padding, raw[..., :2], None)


@pytest.mark.parametrize(
    "selected",
    [
        (0, 1, 2),
        (3, 4, 7, 8, 13),
        tuple(range(len(PAIRWISE_EDGE_RELATION_FIELDS))),
    ],
)
def test_indexed_relations_exactly_match_dense_gather(
    selected: tuple[int, ...],
) -> None:
    _, _, raw, padding = _inputs()
    edges = _knn_edges(raw, padding)
    dense = build_pairwise_edge_relations(raw, padding)
    expected = dense[edges.batch, edges.source_local, edges.destination_local][
        ..., list(selected)
    ]
    actual = build_indexed_pairwise_edge_relations(
        raw,
        edges.batch,
        edges.source_local,
        edges.destination_local,
        padding,
        relation_indices=selected,
    )

    assert actual.shape == (edges.count, len(selected))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_indexed_relations_match_dense_for_self_edges_and_reject_padding() -> None:
    _, _, raw, padding = _inputs()
    edge_batch = torch.tensor([0, 0, 1])
    source = torch.tensor([0, 1, 2])
    destination = torch.tensor([0, 1, 2])
    actual = build_indexed_pairwise_edge_relations(
        raw, edge_batch, source, destination, padding
    )
    assert actual.eq(0.0).all()

    with pytest.raises(ValueError, match="non-padding"):
        build_indexed_pairwise_edge_relations(
            raw,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([8]),
            padding,
        )


@pytest.mark.parametrize("selected", [(0, 1, 2), (3, 4, 7, 8, 13)])
def test_sparse_knn_scores_and_parameter_gradients_exactly_match_dense(
    selected: tuple[int, ...],
) -> None:
    features, baseline, raw, padding = _inputs(seed=23)
    torch.manual_seed(29)
    dense_model = CandidateGNNScorer(
        5,
        edge_dim=len(selected),
        hidden_dim=12,
        edge_hidden_dim=7,
        graph_type="knn",
        k=3,
        mode="residual",
        dropout=0.0,
    )
    dense_model.edge_feature_indices = selected
    sparse_model = copy.deepcopy(dense_model)
    dense_features = features.clone().requires_grad_()
    sparse_features = features.clone().requires_grad_()
    dense_relations = build_pairwise_edge_relations(raw, padding)[..., list(selected)]

    dense_scores = dense_model(
        dense_features,
        baseline,
        padding,
        coordinates=raw[..., :2],
        edge_features=dense_relations,
    )
    sparse_scores = sparse_model(
        sparse_features,
        baseline,
        padding,
        raw_edge_inputs=raw,
    )
    torch.testing.assert_close(sparse_scores, dense_scores, rtol=0.0, atol=0.0)

    dense_scores[~padding].square().sum().backward()
    sparse_scores[~padding].square().sum().backward()
    assert dense_features.grad is not None and sparse_features.grad is not None
    torch.testing.assert_close(
        sparse_features.grad, dense_features.grad, rtol=0.0, atol=0.0
    )
    for (dense_name, dense_parameter), (sparse_name, sparse_parameter) in zip(
        dense_model.named_parameters(), sparse_model.named_parameters(), strict=True
    ):
        assert sparse_name == dense_name
        assert dense_parameter.grad is not None and sparse_parameter.grad is not None
        torch.testing.assert_close(
            sparse_parameter.grad, dense_parameter.grad, rtol=0.0, atol=0.0
        )


def test_sparse_knn_preserves_candidate_permutation() -> None:
    features, baseline, raw, padding = _inputs(seed=31)
    selected = (0, 1, 2, 3, 4, 7, 8)
    torch.manual_seed(37)
    model = CandidateGNNScorer(
        5,
        edge_dim=len(selected),
        hidden_dim=12,
        edge_hidden_dim=7,
        graph_type="knn",
        k=3,
        mode="residual",
        dropout=0.0,
    ).eval()
    model.edge_feature_indices = selected
    permutation = torch.tensor([5, 1, 7, 0, 8, 3, 2, 6, 4])

    with torch.no_grad():
        expected = model(features, baseline, padding, raw_edge_inputs=raw)
        actual = model(
            features[:, permutation],
            baseline[:, permutation],
            padding[:, permutation],
            raw_edge_inputs=raw[:, permutation],
        )
    torch.testing.assert_close(actual, expected[:, permutation], rtol=2e-6, atol=2e-7)


def test_sparse_knn_padding_and_single_candidate_match_dense() -> None:
    generator = torch.Generator().manual_seed(41)
    features = torch.randn(2, 4, 5, generator=generator)
    baseline = torch.randn(2, 4, generator=generator)
    raw = torch.randn(2, 4, 10, generator=generator)
    raw[..., 3] = 2.0
    raw[..., 8] = 0.25
    raw[..., 9] = 1.0
    padding = torch.tensor([[False, True, True, True], [True, True, True, True]])
    raw[padding] = torch.nan
    selected = (0, 1, 2)
    torch.manual_seed(43)
    sparse_model = CandidateGNNScorer(
        5,
        edge_dim=3,
        graph_type="knn",
        k=4,
        mode="residual",
        dropout=0.0,
    ).eval()
    sparse_model.edge_feature_indices = selected
    dense_model = copy.deepcopy(sparse_model)
    dense = build_pairwise_edge_relations(raw, padding)[..., list(selected)]

    with torch.no_grad():
        expected = dense_model(
            features,
            baseline,
            padding,
            coordinates=raw[..., :2],
            edge_features=dense,
        )
        actual = sparse_model(features, baseline, padding, raw_edge_inputs=raw)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[padding], baseline[padding])


@pytest.mark.parametrize("edge_dim", [0, 3])
def test_trainer_knn_path_never_calls_dense_relation_builder(
    monkeypatch: pytest.MonkeyPatch, edge_dim: int
) -> None:
    features, baseline, raw, _ = _inputs(seed=47)
    examples = [
        CandidateSetExample(
            features=features[0, :4],
            labels=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            query_id="q0",
            baseline_scores=baseline[0, :4],
            raw_edge_inputs=raw[0, :4],
        ),
        CandidateSetExample(
            features=features[1, :6],
            labels=torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            query_id="q1",
            baseline_scores=baseline[1, :6],
            raw_edge_inputs=raw[1, :6],
        ),
    ]
    batch = pad_candidate_sets(examples)
    model = CandidateGNNScorer(
        5,
        edge_dim=edge_dim,
        graph_type="knn",
        k=2,
        mode="residual",
        dropout=0.0,
    )
    if edge_dim:
        model.edge_feature_indices = (0, 1, 2)

    def reject_dense(*args, **kwargs):
        raise AssertionError("dense relation builder must not run for kNN")

    monkeypatch.setattr(train_module, "build_pairwise_edge_relations", reject_dense)
    scores, used_raw_edges = train_module._forward_candidate_batch(model, batch)
    assert scores.shape == batch.labels.shape
    assert used_raw_edges is True


@pytest.mark.parametrize("graph_type", ["complete", "rule"])
def test_complete_and_rule_trainer_paths_keep_dense_fallback(
    monkeypatch: pytest.MonkeyPatch, graph_type: str
) -> None:
    features, baseline, raw, _ = _inputs(seed=53)
    batch = pad_candidate_sets(
        [
            CandidateSetExample(
                features=features[0, :4],
                labels=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                query_id="q0",
                baseline_scores=baseline[0, :4],
                raw_edge_inputs=raw[0, :4],
            )
        ]
    )
    edge_dim = len(PAIRWISE_EDGE_RELATION_FIELDS)
    model = CandidateGNNScorer(
        5,
        edge_dim=edge_dim,
        graph_type=graph_type,
        k=2,
        mode="residual",
        dropout=0.0,
    )
    calls = 0
    original = train_module.build_pairwise_edge_relations

    def count_dense(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "build_pairwise_edge_relations", count_dense)
    scores, _ = train_module._forward_candidate_batch(model, batch)
    assert scores.shape == batch.labels.shape
    assert calls == 1


def test_sparse_knn_raw_input_contract_fails_closed() -> None:
    features, baseline, raw, padding = _inputs(seed=59)
    model = CandidateGNNScorer(
        5, edge_dim=3, graph_type="knn", mode="residual", dropout=0.0
    )
    model.edge_feature_indices = (0, 1, 2)
    with pytest.raises(ValueError, match="edge_features are required"):
        model(features, baseline, padding)
    dense = build_pairwise_edge_relations(raw, padding)[..., :3]
    with pytest.raises(ValueError, match="either edge_features or raw_edge_inputs"):
        model(
            features,
            baseline,
            padding,
            edge_features=dense,
            raw_edge_inputs=raw,
        )
    with pytest.raises(ValueError, match="only for kNN"):
        CandidateGNNScorer(5, edge_dim=3, graph_type="complete", mode="residual")(
            features,
            baseline,
            padding,
            raw_edge_inputs=raw,
        )


def test_sparse_knn_does_not_call_dense_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, baseline, raw, padding = _inputs(seed=61)
    model = CandidateGNNScorer(
        5, edge_dim=3, graph_type="knn", mode="residual", dropout=0.0
    )
    model.edge_feature_indices = (0, 1, 2)

    def reject_dense(*args, **kwargs):
        raise AssertionError("dense relation builder was called")

    monkeypatch.setattr(set_models, "build_pairwise_edge_relations", reject_dense)
    scores = model(features, baseline, padding, raw_edge_inputs=raw)
    assert scores.shape == baseline.shape


def test_sparse_knn_training_metadata_records_materialization(
    tmp_path,
) -> None:
    features, baseline, raw, _ = _inputs(seed=67)
    examples = [
        CandidateSetExample(
            features=features[0, :4],
            labels=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            query_id="q0",
            baseline_scores=baseline[0, :4],
            raw_edge_inputs=raw[0, :4],
        ),
        CandidateSetExample(
            features=features[1, :6],
            labels=torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            query_id="q1",
            baseline_scores=baseline[1, :6],
            raw_edge_inputs=raw[1, :6],
        ),
    ]
    loader = make_candidate_dataloader(examples, batch_size=2, shuffle=False, seed=67)
    model = CandidateGNNScorer(
        5,
        edge_dim=3,
        hidden_dim=8,
        edge_hidden_dim=4,
        graph_type="knn",
        k=2,
        mode="residual",
        dropout=0.0,
    )
    model.edge_feature_indices = (0, 1, 2)
    result = fit_neural_ranker(
        model,
        loader,
        loader,
        checkpoint_path=tmp_path / "sparse-knn.pt",
        config=TrainingConfig(epochs=1, patience=1, seed=67, device="cpu"),
    )
    metadata = json.loads(
        (tmp_path / "sparse-knn.pt.metadata.json").read_text(encoding="utf-8")
    )
    schema = metadata["raw_edge_schema"]
    assert result.epochs_ran == 1
    assert schema["builder"].endswith("build_indexed_pairwise_edge_relations")
    assert schema["materialization"] == "selected_knn_edges_only"
    assert schema["relation_layout"] == "edge,relation"
