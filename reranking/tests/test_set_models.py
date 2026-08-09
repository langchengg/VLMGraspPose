from __future__ import annotations

import pytest
import torch

from reranking.models.set_models import (
    CandidateGNNScorer,
    DeepSetsScorer,
    PAIRWISE_EDGE_RELATION_FIELDS,
    RAW_EDGE_CANDIDATE_FIELDS,
    SetTransformerScorer,
    build_pairwise_edge_relations,
    index_add_mean,
    parameter_count,
)


def _candidate_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(2, 6, 7, generator=generator)
    baseline = torch.randn(2, 6, generator=generator)
    padding = torch.tensor(
        [[False, False, False, False, True, True], [False] * 6]
    )
    return features, baseline, padding


def _raw_relation_candidates() -> torch.Tensor:
    """x/y/width share units; theta is radians and clusters are explicit."""

    return torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0, 0.8, 0.2, 1.0, 0.9, 0.25, 3.0],
            [0.0, 0.0, 0.0, 2.0, 0.5, 0.7, 1.5, 0.4, 0.80, 3.0],
            [3.0, 4.0, torch.pi / 4.0, 4.0, 0.4, 0.1, 2.0, 0.2, 0.10, 4.0],
        ],
        dtype=torch.float32,
    )


def _relation_index(name: str) -> int:
    return PAIRWISE_EDGE_RELATION_FIELDS.index(name)


def test_pairwise_builder_has_explicit_schema_and_physical_relations() -> None:
    raw = _raw_relation_candidates()
    relations = build_pairwise_edge_relations(raw)
    assert len(RAW_EDGE_CANDIDATE_FIELDS) == raw.shape[-1]
    assert relations.shape == (3, 3, len(PAIRWISE_EDGE_RELATION_FIELDS))

    # Candidates 0 and 1 are distinct records with identical rectangles.
    forward = relations[0, 1]
    reverse = relations[1, 0]
    assert forward[_relation_index("approx_rectangle_iou")] == pytest.approx(1.0)
    assert forward[_relation_index("axis_overlap")] == pytest.approx(1.0)
    assert forward[_relation_index("same_cluster")] == pytest.approx(1.0)
    assert forward[_relation_index("sweep_conflict")] == pytest.approx(0.8)
    assert forward[_relation_index("delta_q")] == pytest.approx(-0.3)
    assert forward[_relation_index("mask_support_delta")] == pytest.approx(0.5)
    assert forward[_relation_index("depth_delta")] == pytest.approx(0.5)
    assert forward[_relation_index("clearance_delta")] == pytest.approx(-0.5)
    assert reverse[_relation_index("delta_q")] == pytest.approx(0.3)
    assert reverse[_relation_index("sweep_conflict")] == pytest.approx(0.8)

    separated = relations[0, 2]
    assert separated[_relation_index("delta_x")] == pytest.approx(3.0)
    assert separated[_relation_index("delta_y")] == pytest.approx(4.0)
    assert separated[_relation_index("center_distance")] == pytest.approx(5.0)
    assert separated[_relation_index("sin_2_delta_theta")] == pytest.approx(1.0)
    assert separated[_relation_index("cos_2_delta_theta")] == pytest.approx(0.0, abs=1e-6)
    assert separated[_relation_index("delta_width")] == pytest.approx(2.0)
    assert separated[_relation_index("same_cluster")] == pytest.approx(0.0)
    assert separated[_relation_index("approx_rectangle_iou")] == pytest.approx(0.0)
    torch.testing.assert_close(
        relations.diagonal(dim1=0, dim2=1),
        torch.zeros_like(relations.diagonal(dim1=0, dim2=1)),
    )


def test_pairwise_builder_and_edge_gnn_are_permutation_equivariant() -> None:
    raw = _raw_relation_candidates().unsqueeze(0)
    node_features = torch.tensor(
        [[[0.4, -0.1], [0.2, 0.7], [-0.5, 0.3]]], dtype=torch.float32
    )
    baseline = torch.tensor([[0.8, 0.5, 0.4]])
    permutation = torch.tensor([2, 0, 1])
    edges = build_pairwise_edge_relations(raw)
    permuted_edges = build_pairwise_edge_relations(raw[:, permutation])
    torch.testing.assert_close(
        permuted_edges,
        edges[:, permutation][:, :, permutation],
        rtol=1e-6,
        atol=1e-6,
    )

    torch.manual_seed(113)
    model = CandidateGNNScorer(
        2,
        edge_dim=len(PAIRWISE_EDGE_RELATION_FIELDS),
        graph_type="complete",
        mode="residual",
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        expected = model(node_features, baseline, edge_features=edges)
        actual = model(
            node_features[:, permutation],
            baseline[:, permutation],
            edge_features=permuted_edges,
        )
    torch.testing.assert_close(actual, expected[:, permutation], rtol=2e-5, atol=2e-6)


def test_pairwise_builder_masks_invalid_padded_payloads() -> None:
    raw = torch.cat(
        (_raw_relation_candidates(), torch.full((1, 10), torch.nan)), dim=0
    ).unsqueeze(0)
    padding = torch.tensor([[False, False, False, True]])
    actual = build_pairwise_edge_relations(raw, padding)
    expected = build_pairwise_edge_relations(raw[:, :3])
    torch.testing.assert_close(actual[:, :3, :3], expected)
    torch.testing.assert_close(actual[:, 3], torch.zeros_like(actual[:, 3]))
    torch.testing.assert_close(actual[:, :, 3], torch.zeros_like(actual[:, :, 3]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (3, 0.0, "widths must be positive"),
        (8, 1.1, "conflict_risk"),
        (9, 1.5, "cluster_id"),
    ],
)
def test_pairwise_builder_rejects_invalid_raw_semantics(
    column: int, value: float, message: str
) -> None:
    raw = _raw_relation_candidates()
    raw[0, column] = value
    with pytest.raises(ValueError, match=message):
        build_pairwise_edge_relations(raw)


def test_edge_conditioned_gnn_fails_closed_without_explicit_relations() -> None:
    model = CandidateGNNScorer(
        2,
        edge_dim=len(PAIRWISE_EDGE_RELATION_FIELDS),
        graph_type="complete",
    )
    with pytest.raises(ValueError, match="unstandardized candidate data"):
        model(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2),
            padding_mask=torch.ones(1, 2, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    "model",
    [
        DeepSetsScorer(7, mode="residual", dropout=0.0),
        SetTransformerScorer(
            7, mode="residual", num_blocks=2, dropout=0.0
        ),
        CandidateGNNScorer(
            7, mode="residual", graph_type="complete", dropout=0.0
        ),
        CandidateGNNScorer(
            7, mode="direct", graph_type="knn", k=4, dropout=0.0
        ),
    ],
    ids=["deepsets", "set-transformer", "gnn-complete", "gnn-knn"],
)
def test_candidate_permutation_equivariance(model: torch.nn.Module) -> None:
    features, baseline, padding = _candidate_batch()
    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    model.eval()

    with torch.no_grad():
        expected = model(features, baseline, padding)
        actual = model(
            features[:, permutation],
            baseline[:, permutation],
            padding[:, permutation],
        )

    torch.testing.assert_close(
        actual, expected[:, permutation], rtol=2e-5, atol=2e-6
    )


def _complete_edge_index(count: int) -> torch.Tensor:
    source = torch.arange(count).repeat_interleave(count)
    destination = torch.arange(count).repeat(count)
    keep = source != destination
    return torch.stack((source[keep], destination[keep]))


def _explicit_edge_features(
    features: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    source, destination = edge_index
    return torch.stack(
        (
            features[source, 0] - features[destination, 0],
            features[source, 1] + features[destination, 1],
            features[source, 2] * features[destination, 2],
        ),
        dim=-1,
    )


def test_explicit_edge_gnn_is_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(23)
    features = torch.randn(5, 6, generator=generator)
    baseline = torch.randn(5, generator=generator)
    edge_index = _complete_edge_index(5)
    edge_features = _explicit_edge_features(features, edge_index)
    model = CandidateGNNScorer(
        6,
        edge_dim=3,
        graph_type="explicit",
        mode="residual",
        dropout=0.0,
    ).eval()

    permutation = torch.tensor([2, 4, 0, 1, 3])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted_edge_index = inverse[edge_index]
    permuted_features = features[permutation]
    permuted_edge_features = _explicit_edge_features(
        permuted_features, permuted_edge_index
    )

    with torch.no_grad():
        expected = model(
            features,
            baseline,
            edge_index=edge_index,
            edge_features=edge_features,
        )
        actual = model(
            permuted_features,
            baseline[permutation],
            edge_index=permuted_edge_index,
            edge_features=permuted_edge_features,
        )
    torch.testing.assert_close(
        actual, expected[permutation], rtol=2e-5, atol=2e-6
    )


def test_dense_edge_features_follow_candidate_permutation() -> None:
    features, baseline, padding = _candidate_batch()
    generator = torch.Generator().manual_seed(29)
    dense_edges = torch.randn(2, 6, 6, 3, generator=generator)
    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    model = CandidateGNNScorer(
        7,
        edge_dim=3,
        graph_type="complete",
        mode="residual",
        dropout=0.0,
    ).eval()

    with torch.no_grad():
        expected = model(
            features,
            baseline,
            padding,
            edge_features=dense_edges,
        )
        actual = model(
            features[:, permutation],
            baseline[:, permutation],
            padding[:, permutation],
            edge_features=dense_edges[:, permutation][:, :, permutation],
        )
    torch.testing.assert_close(
        actual, expected[:, permutation], rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize(
    "model",
    [
        DeepSetsScorer(4, mode="residual"),
        SetTransformerScorer(4, mode="residual", num_blocks=1),
        CandidateGNNScorer(4, mode="residual", graph_type="complete"),
    ],
    ids=["deepsets", "set-transformer", "gnn"],
)
def test_padding_is_ignored_and_empty_row_bypasses(model: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(31)
    features = torch.randn(2, 4, 4, generator=generator)
    baseline = torch.randn(2, 4, generator=generator)
    padding = torch.tensor(
        [[False, False, True, True], [True, True, True, True]]
    )
    changed = features.clone()
    changed[padding] = 1_000_000.0
    model.eval()

    with torch.no_grad():
        expected = model(features, baseline, padding)
        actual = model(changed, baseline, padding)

    torch.testing.assert_close(actual[0, :2], expected[0, :2])
    torch.testing.assert_close(actual[padding], baseline[padding])
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "factory",
    [
        lambda mode: DeepSetsScorer(3, mode=mode),
        lambda mode: SetTransformerScorer(3, mode=mode),
        lambda mode: CandidateGNNScorer(3, mode=mode),
    ],
    ids=["deepsets", "set-transformer", "gnn"],
)
@pytest.mark.parametrize("mode", ["direct", "residual"])
def test_zero_candidate_tensor_is_supported(factory, mode: str) -> None:
    model = factory(mode)
    features = torch.empty(2, 0, 3)
    baseline = torch.empty(2, 0)
    scores = model(features, baseline if mode == "residual" else None)
    assert scores.shape == (2, 0)


def test_direct_empty_set_padding_outputs_zero() -> None:
    features = torch.randn(2, 3, 4)
    padding = torch.ones(2, 3, dtype=torch.bool)
    for model in (
        DeepSetsScorer(4, mode="direct"),
        SetTransformerScorer(4, mode="direct"),
        CandidateGNNScorer(4, mode="direct", edge_dim=0),
    ):
        scores = model(features, padding_mask=padding)
        torch.testing.assert_close(scores, torch.zeros_like(scores))


def test_index_add_mean_and_parameter_count_contracts() -> None:
    messages = torch.tensor([[1.0, 3.0], [3.0, 5.0], [7.0, 9.0]])
    destination = torch.tensor([0, 0, 2])
    aggregate = index_add_mean(messages, destination, num_nodes=4)
    torch.testing.assert_close(
        aggregate,
        torch.tensor([[2.0, 4.0], [0.0, 0.0], [7.0, 9.0], [0.0, 0.0]]),
    )

    model = DeepSetsScorer(3)
    total = parameter_count(model, trainable_only=False)
    assert total > 0
    first_parameter = next(model.parameters())
    frozen_size = first_parameter.numel()
    first_parameter.requires_grad_(False)
    assert parameter_count(model) == total - frozen_size


def test_two_message_passing_rounds_and_attention_configuration() -> None:
    gnn = CandidateGNNScorer(5)
    transformer = SetTransformerScorer(5, num_blocks=2)
    assert len(gnn.message_blocks) == 2
    assert len(transformer.blocks) == 2
    assert transformer.hidden_dim == 64
    assert transformer.blocks[0].attention.num_heads == 4


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", _available_devices())
def test_all_models_run_on_supported_device(device: str) -> None:
    torch.manual_seed(37)
    features = torch.randn(2, 4, 5, device=device, requires_grad=True)
    baseline = torch.randn(2, 4, device=device)
    padding = torch.tensor(
        [[False, False, True, True], [False, False, False, False]],
        device=device,
    )
    models = (
        DeepSetsScorer(5, mode="residual"),
        SetTransformerScorer(5, mode="residual", num_blocks=1),
        CandidateGNNScorer(5, mode="residual", graph_type="knn", k=2),
    )
    loss = features.new_zeros(())
    for model in models:
        model = model.to(device)
        scores = model(features, baseline, padding)
        assert scores.device.type == device
        assert scores.shape == baseline.shape
        assert torch.isfinite(scores).all()
        loss = loss + scores[~padding].sum()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_invalid_configuration_and_missing_residual_baseline_are_rejected() -> None:
    with pytest.raises(ValueError, match="1 or 2"):
        SetTransformerScorer(4, num_blocks=3)
    with pytest.raises(ValueError, match="divisible"):
        SetTransformerScorer(4, hidden_dim=63, num_heads=4)
    with pytest.raises(ValueError, match="baseline_scores"):
        DeepSetsScorer(4, mode="residual")(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="edge_index"):
        CandidateGNNScorer(4, graph_type="explicit")(
            torch.randn(2, 3, 4), torch.randn(2, 3)
        )
