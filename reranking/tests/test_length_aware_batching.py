from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch
from torch.utils.data import DataLoader

from reranking.losses import CompositeRerankingLoss
from reranking.models.set_models import DeepSetsScorer
from reranking.train import (
    CandidateBatch,
    CandidateSetDataset,
    CandidateSetExample,
    LENGTH_BUCKETED_BATCHING_POLICY,
    _run_epoch,
    make_candidate_dataloader,
    pad_candidate_sets,
)


def _examples(lengths: Iterable[int]) -> list[CandidateSetExample]:
    result: list[CandidateSetExample] = []
    for index, length in enumerate(lengths):
        features = torch.arange(length * 2, dtype=torch.float32).reshape(length, 2)
        result.append(
            CandidateSetExample(
                features=features,
                labels=torch.zeros(length),
                query_id=f"q{index:03d}",
            )
        )
    return result


def _epoch(loader: DataLoader[CandidateBatch]) -> tuple[tuple[str, ...], ...]:
    return tuple(batch.query_ids for batch in loader)


def _padding_slots(loader: DataLoader[CandidateBatch]) -> int:
    padding = 0
    for batch in loader:
        padding += batch.batch_size * int(batch.lengths.max().item())
        padding -= int(batch.lengths.sum().item())
    return padding


def test_policy_off_exactly_matches_legacy_seeded_dataloader() -> None:
    examples = _examples([1, 7, 2, 9, 3, 8, 4, 6, 5])
    actual = make_candidate_dataloader(
        examples,
        batch_size=3,
        shuffle=True,
        seed=31,
    )
    dataset = CandidateSetDataset(examples)
    legacy = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_candidate_sets,
        generator=torch.Generator().manual_seed(31),
        persistent_workers=False,
    )

    assert [_epoch(actual), _epoch(actual)] == [_epoch(legacy), _epoch(legacy)]


def test_length_bucketed_policy_is_reproducible_random_and_complete_per_epoch() -> None:
    examples = _examples(range(1, 24))

    def make_loader() -> DataLoader[CandidateBatch]:
        return make_candidate_dataloader(
            examples,
            batch_size=4,
            shuffle=True,
            seed=1729,
            batching_policy=LENGTH_BUCKETED_BATCHING_POLICY,
        )

    epochs_a = [_epoch(make_loader()) for _ in range(3)]
    loader = make_loader()
    epochs_b = [_epoch(loader) for _ in range(3)]

    # Reconstructing a loader with the same seed reproduces its epoch stream.
    stream_a = make_loader()
    assert [_epoch(stream_a) for _ in range(3)] == epochs_b
    assert epochs_b[0] != epochs_b[1]
    expected = sorted(example.query_id for example in examples)
    for epoch in epochs_b:
        observed = [query_id for batch in epoch for query_id in batch]
        assert len(observed) == len(expected)
        assert sorted(observed) == expected

    # Fresh one-epoch loaders are intentionally identical at a fixed seed.
    assert epochs_a[0] == epochs_a[1] == epochs_a[2]


def test_length_bucketed_policy_materially_reduces_padding() -> None:
    examples = _examples(range(1, 65))
    random_loader = make_candidate_dataloader(
        examples,
        batch_size=8,
        shuffle=True,
        seed=19,
    )
    bucketed_loader = make_candidate_dataloader(
        examples,
        batch_size=8,
        shuffle=True,
        seed=19,
        batching_policy=LENGTH_BUCKETED_BATCHING_POLICY,
    )

    random_padding = _padding_slots(random_loader)
    bucketed_padding = _padding_slots(bucketed_loader)
    assert bucketed_padding < random_padding
    assert bucketed_padding <= random_padding * 0.25


@pytest.mark.parametrize("policy", ["", "length_bucketed", "random"])
def test_unknown_batching_policy_fails_closed(policy: str) -> None:
    with pytest.raises(ValueError, match="batching_policy"):
        make_candidate_dataloader(
            _examples([1, 2]),
            batch_size=2,
            shuffle=True,
            seed=5,
            batching_policy=policy,
        )


def test_length_bucketed_validation_order_is_stable_complete_and_length_sorted() -> (
    None
):
    examples = _examples([5, 1, 4, 2, 3, 2, 1])
    loader = make_candidate_dataloader(
        examples,
        batch_size=3,
        shuffle=False,
        seed=5,
        batching_policy=LENGTH_BUCKETED_BATCHING_POLICY,
    )

    first = _epoch(loader)
    second = _epoch(loader)
    assert first == second
    observed = [query_id for batch in first for query_id in batch]
    assert sorted(observed) == sorted(example.query_id for example in examples)
    length_by_query = {
        example.query_id: int(example.features.shape[0]) for example in examples
    }
    assert [length_by_query[query_id] for query_id in observed] == sorted(
        length_by_query.values()
    )


def test_length_bucketed_validation_scores_and_loss_match_legacy() -> None:
    examples = _examples([7, 1, 5, 2, 8, 3, 6, 4])
    examples = [
        CandidateSetExample(
            features=example.features,
            labels=torch.nn.functional.one_hot(
                torch.tensor(index % len(example.labels)),
                num_classes=len(example.labels),
            ).to(torch.float32),
            query_id=example.query_id,
        )
        for index, example in enumerate(examples)
    ]
    legacy = make_candidate_dataloader(examples, batch_size=3, shuffle=False, seed=41)
    bucketed = make_candidate_dataloader(
        examples,
        batch_size=3,
        shuffle=False,
        seed=41,
        batching_policy=LENGTH_BUCKETED_BATCHING_POLICY,
    )
    torch.manual_seed(41)
    model = DeepSetsScorer(2, hidden_dim=8, mode="direct", dropout=0.0).eval()

    def scores_by_query(
        loader: DataLoader[CandidateBatch],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for batch in loader:
                scores = model(batch.features, padding_mask=batch.padding_mask)
                for row, (query_id, length) in enumerate(
                    zip(batch.query_ids, batch.lengths.tolist(), strict=True)
                ):
                    result[query_id] = scores[row, :length].clone()
        return result

    legacy_scores = scores_by_query(legacy)
    bucketed_scores = scores_by_query(bucketed)
    assert legacy_scores.keys() == bucketed_scores.keys()
    for query_id in legacy_scores:
        torch.testing.assert_close(
            bucketed_scores[query_id],
            legacy_scores[query_id],
            rtol=2e-7,
            atol=2e-8,
        )

    criterion = CompositeRerankingLoss(bce_weight=1.0)
    legacy_result = _run_epoch(
        model,
        legacy,
        criterion,
        device=torch.device("cpu"),
        optimizer=None,
        gradient_clip_norm=None,
    )
    bucketed_result = _run_epoch(
        model,
        bucketed,
        criterion,
        device=torch.device("cpu"),
        optimizer=None,
        gradient_clip_norm=None,
    )
    assert bucketed_result.active_queries == legacy_result.active_queries
    assert bucketed_result.loss == pytest.approx(legacy_result.loss, rel=1e-7, abs=1e-8)
