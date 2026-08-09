from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional

from reranking.losses import (
    CompositeRerankingLoss,
    combined_reranking_loss,
    listwise_softmax_ce_loss,
    query_composition,
    query_normalized_bce_loss,
    same_query_ranknet_loss,
)
from reranking.models.neural import SharedMLPScorer
from reranking.models.set_models import (
    CandidateGNNScorer,
    DeepSetsScorer,
    PAIRWISE_EDGE_RELATION_FIELDS,
    RAW_EDGE_CANDIDATE_FIELDS,
)
from reranking.train import (
    CandidateSetExample,
    TrainingConfig,
    fit_neural_ranker,
    load_training_checkpoint,
    make_candidate_dataloader,
    pad_candidate_sets,
    resolve_device,
    set_global_seed,
)


def test_query_normalized_bce_gives_equal_weight_to_queries() -> None:
    scores = torch.zeros(2, 3, requires_grad=True)
    labels = torch.tensor([[1.0, -1.0, -1.0], [0.0, 0.0, 0.0]])
    padding = torch.tensor([[False, True, True], [False, False, False]])
    loss = query_normalized_bce_loss(scores, labels, padding)
    assert loss.item() == pytest.approx(float(torch.log(torch.tensor(2.0))))
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 1:].eq(0).all()


def test_padding_labels_are_ignored_without_changing_validation_contract() -> None:
    scores = torch.zeros(2, 3, requires_grad=True)
    labels = torch.tensor([[1.0, float("nan"), -7.0], [0.0, 1.0, 0.0]])
    padding = torch.tensor([[False, True, True], [False, False, False]])
    loss = query_normalized_bce_loss(scores, labels, padding)
    assert loss.item() == pytest.approx(float(torch.log(torch.tensor(2.0))))
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 1:].eq(0).all()


@pytest.mark.parametrize("invalid", [float("nan"), -1.0, 2.0])
def test_non_padding_labels_still_fail_closed(invalid: float) -> None:
    scores = torch.zeros(1, 2)
    labels = torch.tensor([[1.0, invalid]])
    with pytest.raises(ValueError, match="finite binary"):
        query_normalized_bce_loss(scores, labels)


def test_ranknet_uses_only_same_query_mixed_pairs() -> None:
    scores = torch.tensor(
        [[2.0, 0.0], [100.0, -100.0], [-100.0, 100.0], [0.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([[1, 0], [0, 0], [1, 1], [0, 0]])
    padding = torch.tensor(
        [[False, False], [False, False], [False, False], [True, True]]
    )
    loss = same_query_ranknet_loss(scores, labels, padding)
    assert loss.item() == pytest.approx(float(functional.softplus(torch.tensor(-2.0))))
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[1:].eq(0).all()

    composition = query_composition(labels, padding)
    assert composition.as_dict() == {
        "total": 4,
        "non_empty": 3,
        "empty": 1,
        "no_positive": 1,
        "all_positive": 1,
        "mixed": 1,
    }


def test_multiple_positive_listwise_matches_logsumexp_formula() -> None:
    scores = torch.tensor([[2.0, 1.0, 0.0], [9.0, -9.0, 2.0]])
    labels = torch.tensor([[1, 1, 0], [1, 1, 1]])
    expected = torch.logsumexp(scores[0], dim=0) - torch.logsumexp(scores[0, :2], dim=0)
    actual = listwise_softmax_ce_loss(scores, labels, temperature=1.0)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("loss_fn", [same_query_ranknet_loss, listwise_softmax_ce_loss])
def test_ranking_losses_return_differentiable_zero_without_mixed_query(loss_fn) -> None:
    scores = torch.randn(3, 2, requires_grad=True)
    labels = torch.tensor([[0, 0], [1, 1], [0, 0]])
    padding = torch.tensor([[False, False], [False, False], [True, True]])
    loss = loss_fn(scores, labels, padding)
    assert loss.item() == 0.0
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.eq(0).all()


def test_combined_loss_is_exact_weighted_sum() -> None:
    scores = torch.tensor([[1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    baseline = torch.tensor([[0.25, -0.25]])
    result = combined_reranking_loss(
        scores,
        labels,
        baseline_scores=baseline,
        bce_weight=0.5,
        ranknet_weight=0.2,
        listwise_weight=0.3,
        residual_weight=0.1,
    )
    expected = (
        0.5 * result.bce
        + 0.2 * result.ranknet
        + 0.3 * result.listwise
        + 0.1 * result.residual
    )
    torch.testing.assert_close(result.total, expected)
    assert result.active_queries == 1
    result.total.backward()
    assert torch.isfinite(scores.grad).all()


def test_active_breakdown_skips_inactive_terms_without_changing_optimization() -> None:
    full_scores = torch.tensor([[1.0, 0.0], [-0.5, 0.5]], requires_grad=True)
    active_scores = full_scores.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    criterion = CompositeRerankingLoss(bce_weight=1.0)

    full = criterion.breakdown(full_scores, labels)
    composition = query_composition(labels)
    active = criterion.active_breakdown(active_scores, labels, composition=composition)
    torch.testing.assert_close(active.total, full.total)
    torch.testing.assert_close(active.bce, full.bce)
    assert active.ranknet.item() == 0.0
    assert active.listwise.item() == 0.0
    assert active.composition == full.composition
    assert active.active_queries == full.active_queries

    full.total.backward()
    active.total.backward()
    torch.testing.assert_close(active_scores.grad, full_scores.grad)


def test_active_breakdown_preserves_empty_query_semantics() -> None:
    full_scores = torch.tensor([[1.0, 99.0], [0.5, -0.5]], requires_grad=True)
    active_scores = full_scores.detach().clone().requires_grad_(True)
    labels = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    padding = torch.tensor([[True, True], [False, False]])
    composition = query_composition(labels, padding)
    criterion = CompositeRerankingLoss(bce_weight=1.0)

    full = criterion.breakdown(full_scores, labels, padding)
    active = criterion.active_breakdown(
        active_scores,
        labels,
        padding,
        composition=composition,
    )
    torch.testing.assert_close(active.total, full.total)
    assert active.composition == full.composition
    assert active.active_queries == 1

    full.total.backward()
    active.total.backward()
    torch.testing.assert_close(active_scores.grad, full_scores.grad)


def test_shared_mlp_direct_and_residual_use_identical_raw_scorer() -> None:
    set_global_seed(5)
    direct = SharedMLPScorer(4, mode="direct", dropout=0.0)
    residual = SharedMLPScorer(4, mode="residual", residual_scale=0.25, dropout=0.0)
    residual.load_state_dict(direct.state_dict())
    features = torch.randn(2, 3, 4)
    baseline = torch.randn(2, 3)
    padding = torch.tensor([[False, False, True], [False, False, False]])
    direct.eval()
    residual.eval()
    raw = direct(features, baseline, padding)
    actual = residual(features, baseline, padding)
    expected = baseline + 0.25 * torch.tanh(raw)
    expected[padding] = baseline[padding]
    torch.testing.assert_close(actual, expected)
    assert raw[padding].eq(0).all()


def test_shared_mlp_candidate_permutation_and_empty_tensor() -> None:
    model = SharedMLPScorer(3, mode="direct", dropout=0.0).eval()
    features = torch.randn(2, 5, 3)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        expected = model(features)
        actual = model(features[:, permutation])
    torch.testing.assert_close(actual, expected[:, permutation])
    assert model(torch.empty(2, 0, 3)).shape == (2, 0)


def _examples() -> list[CandidateSetExample]:
    return [
        CandidateSetExample(
            query_id="empty",
            features=torch.empty(0, 3),
            labels=torch.empty(0),
            baseline_scores=torch.empty(0),
        ),
        CandidateSetExample(
            query_id="mixed",
            features=torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]),
            labels=torch.tensor([1.0, 0.0]),
            baseline_scores=torch.tensor([0.2, -0.2]),
        ),
    ]


def _raw_edge_inputs(count: int, *, cluster: int = 1) -> torch.Tensor:
    raw = torch.zeros(count, len(RAW_EDGE_CANDIDATE_FIELDS))
    if count:
        raw[:, 0] = torch.arange(count, dtype=torch.float32)
        raw[:, 1] = 0.5 * torch.arange(count, dtype=torch.float32)
        raw[:, 2] = torch.arange(count, dtype=torch.float32) * 0.1
        raw[:, 3] = 2.0
        raw[:, 4] = torch.linspace(0.8, 0.4, count)
        raw[:, 5] = torch.linspace(0.2, 0.9, count)
        raw[:, 6] = 1.0 + 0.1 * torch.arange(count, dtype=torch.float32)
        raw[:, 7] = 0.5
        raw[:, 8] = 0.25
        raw[:, 9] = float(cluster)
    return raw


def test_padding_collator_preserves_valid_empty_queries() -> None:
    batch = pad_candidate_sets(_examples())
    assert batch.features.shape == (2, 2, 3)
    assert batch.lengths.tolist() == [0, 2]
    assert batch.padding_mask[0].all()
    assert not batch.padding_mask[1].any()
    assert batch.query_ids == ("empty", "mixed")
    assert batch.baseline_scores is not None


def test_padding_collator_preserves_raw_edge_side_channel() -> None:
    examples = _examples()
    examples = [
        CandidateSetExample(
            example.features,
            example.labels,
            example.query_id,
            example.baseline_scores,
            _raw_edge_inputs(len(example.labels), cluster=index),
        )
        for index, example in enumerate(examples)
    ]
    batch = pad_candidate_sets(examples)
    assert batch.raw_edge_inputs is not None
    assert batch.raw_edge_inputs.shape == (
        2,
        2,
        len(RAW_EDGE_CANDIDATE_FIELDS),
    )
    torch.testing.assert_close(
        batch.raw_edge_inputs[1, :2], examples[1].raw_edge_inputs
    )
    torch.testing.assert_close(
        batch.raw_edge_inputs[0], torch.zeros_like(batch.raw_edge_inputs[0])
    )
    moved = batch.to(torch.device("cpu"))
    assert moved.raw_edge_inputs is not None
    torch.testing.assert_close(moved.raw_edge_inputs, batch.raw_edge_inputs)


def test_padding_collator_rejects_partial_baseline_presence() -> None:
    examples = _examples()
    examples[0] = CandidateSetExample(
        query_id="empty", features=torch.empty(0, 3), labels=torch.empty(0)
    )
    with pytest.raises(ValueError, match="every example or none"):
        pad_candidate_sets(examples)


def _training_examples(*, with_raw_edges: bool = False) -> list[CandidateSetExample]:
    examples: list[CandidateSetExample] = []
    for query_index, positive_index in enumerate((0, 1, 2, 0, 1, 2)):
        features = torch.zeros(3, 4)
        labels = torch.zeros(3)
        labels[positive_index] = 1.0
        features[:, 0] = labels * 2.0 - 1.0
        features[:, 1] = torch.arange(3, dtype=torch.float32) / 3.0
        features[:, 2] = query_index / 6.0
        features[:, 3] = 1.0
        baseline = torch.tensor([0.1, 0.0, -0.1])
        examples.append(
            CandidateSetExample(
                features,
                labels,
                f"q{query_index}",
                baseline,
                (_raw_edge_inputs(3, cluster=query_index) if with_raw_edges else None),
            )
        )
    return examples


def _fit_once(root: Path, run_name: str):
    set_global_seed(41)
    model = SharedMLPScorer(4, mode="residual", dropout=0.0)
    examples = _training_examples()
    train_loader = make_candidate_dataloader(
        examples[:4], batch_size=2, shuffle=True, seed=41
    )
    validation_loader = make_candidate_dataloader(
        examples[4:], batch_size=2, shuffle=False, seed=41
    )
    result = fit_neural_ranker(
        model,
        train_loader,
        validation_loader,
        checkpoint_path=root / f"{run_name}.pt",
        criterion=CompositeRerankingLoss(bce_weight=1.0, ranknet_weight=0.2),
        config=TrainingConfig(
            epochs=8,
            learning_rate=1e-2,
            patience=2,
            min_delta=1_000_000.0,
            seed=41,
            device="cpu",
        ),
        metadata={"split": "unit-test"},
    )
    return model, result


def test_training_is_seeded_early_stopped_and_checkpointed(tmp_path: Path) -> None:
    model_a, result_a = _fit_once(tmp_path, "a")
    model_b, result_b = _fit_once(tmp_path, "b")
    assert result_a.stopped_early
    assert result_a.best_epoch == 1
    assert result_a.epochs_ran == 3
    assert result_a.history == result_b.history
    for left, right in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(left, right)

    checkpoint = Path(result_a.checkpoint_path)
    metadata_path = Path(result_a.metadata_path)
    assert checkpoint.is_file() and metadata_path.is_file()
    assert Path(result_a.last_checkpoint_path).is_file()
    assert Path(result_a.history_path).is_file()
    assert Path(result_a.last_checkpoint_path + ".metadata.json").is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["last_checkpoint_sha256"]
    assert metadata["history_sha256"]
    assert metadata["seed"] == 41
    assert metadata["device"] == "cpu"
    assert metadata["user_metadata"] == {"split": "unit-test"}
    assert metadata["query_semantics"]["all_positive"].startswith("BCE eligible")
    assert (
        metadata["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )

    restored = SharedMLPScorer(4, mode="residual", dropout=0.0)
    payload = load_training_checkpoint(restored, checkpoint)
    assert payload["best_epoch"] == 1
    for expected, actual in zip(model_a.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)


def test_trainer_accepts_set_model_forward_contract(tmp_path: Path) -> None:
    examples = _training_examples()
    loader = make_candidate_dataloader(examples, batch_size=3, shuffle=False, seed=7)
    model = DeepSetsScorer(4, mode="residual", dropout=0.0)
    result = fit_neural_ranker(
        model,
        loader,
        loader,
        checkpoint_path=tmp_path / "deepsets.pt",
        config=TrainingConfig(epochs=1, patience=1, seed=7, device="cpu"),
    )
    assert result.epochs_ran == 1
    assert Path(result.checkpoint_path).is_file()


def test_trainer_builds_explicit_raw_relations_for_edge_gnn(tmp_path: Path) -> None:
    examples = _training_examples(with_raw_edges=True)
    train_loader = make_candidate_dataloader(
        examples[:4], batch_size=2, shuffle=True, seed=19
    )
    validation_loader = make_candidate_dataloader(
        examples[4:], batch_size=2, shuffle=False, seed=19
    )
    model = CandidateGNNScorer(
        4,
        edge_dim=len(PAIRWISE_EDGE_RELATION_FIELDS),
        graph_type="complete",
        mode="residual",
        dropout=0.0,
    )
    result = fit_neural_ranker(
        model,
        train_loader,
        validation_loader,
        checkpoint_path=tmp_path / "edge-gnn.pt",
        config=TrainingConfig(epochs=1, patience=1, seed=19, device="cpu"),
    )
    assert result.epochs_ran == 1
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    schema = metadata["raw_edge_schema"]
    assert schema["schema_version"] == 1
    assert schema["present"] is True
    assert schema["used_by_model"] is True
    assert schema["candidate_fields"] == list(RAW_EDGE_CANDIDATE_FIELDS)
    assert schema["relation_fields"] == list(PAIRWISE_EDGE_RELATION_FIELDS)
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["metadata"]["raw_edge_schema"] == schema


def test_raw_edge_side_channel_is_not_forwarded_to_mlp(tmp_path: Path) -> None:
    examples = _training_examples(with_raw_edges=True)
    loader = make_candidate_dataloader(examples, batch_size=3, shuffle=False, seed=23)
    model = SharedMLPScorer(4, mode="residual", dropout=0.0)
    result = fit_neural_ranker(
        model,
        loader,
        loader,
        checkpoint_path=tmp_path / "mlp-with-unused-raw.pt",
        config=TrainingConfig(epochs=1, patience=1, seed=23, device="cpu"),
    )
    metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
    assert metadata["raw_edge_schema"]["present"] is True
    assert metadata["raw_edge_schema"]["used_by_model"] is False


def test_edge_gnn_without_raw_edge_inputs_fails_closed(tmp_path: Path) -> None:
    examples = _training_examples(with_raw_edges=False)
    loader = make_candidate_dataloader(examples, batch_size=3, shuffle=False, seed=29)
    model = CandidateGNNScorer(
        4,
        edge_dim=len(PAIRWISE_EDGE_RELATION_FIELDS),
        graph_type="complete",
        mode="residual",
        dropout=0.0,
    )
    checkpoint = tmp_path / "missing-raw-edge.pt"
    with pytest.raises(ValueError, match="unstandardized candidate data"):
        fit_neural_ranker(
            model,
            loader,
            loader,
            checkpoint_path=checkpoint,
            config=TrainingConfig(epochs=1, patience=1, seed=29, device="cpu"),
        )
    assert not checkpoint.exists()


def test_ranking_only_training_rejects_no_eligible_query(tmp_path: Path) -> None:
    examples = [
        CandidateSetExample(torch.randn(2, 3), torch.zeros(2), "negative"),
        CandidateSetExample(torch.randn(2, 3), torch.ones(2), "positive"),
    ]
    loader = make_candidate_dataloader(examples, batch_size=2, shuffle=False, seed=3)
    with pytest.raises(ValueError, match="no eligible queries"):
        fit_neural_ranker(
            SharedMLPScorer(3, mode="direct", dropout=0.0),
            loader,
            loader,
            checkpoint_path=tmp_path / "ineligible.pt",
            criterion=CompositeRerankingLoss(bce_weight=0.0, ranknet_weight=1.0),
            config=TrainingConfig(epochs=1, patience=1, seed=3),
        )
    assert not (tmp_path / "ineligible.pt").exists()


def test_explicit_device_validation() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="cpu.*mps"):
        resolve_device("cuda")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_one_epoch_mps_training(tmp_path: Path) -> None:
    examples = _training_examples()[:2]
    loader = make_candidate_dataloader(examples, batch_size=2, shuffle=False, seed=13)
    model = SharedMLPScorer(4, mode="residual", dropout=0.0)
    result = fit_neural_ranker(
        model,
        loader,
        loader,
        checkpoint_path=tmp_path / "mps.pt",
        config=TrainingConfig(epochs=1, patience=1, seed=13, device="mps"),
    )
    assert result.device == "mps"
    assert next(model.parameters()).device.type == "mps"
