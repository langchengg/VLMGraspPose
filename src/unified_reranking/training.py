"""Deterministic CPU training loop for the controlled neural comparisons."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn

from .datasets import QueryArrays
from .losses import (
    all_pairs_ranknet_loss,
    jacquard_margin_ranknet_loss,
    multi_positive_listwise_loss,
    query_equal_bce_loss,
)


FORMAL_SEEDS = (42, 123, 2026)


@dataclass(frozen=True)
class NeuralTrainingConfig:
    loss: str
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    alpha: float = 0.5
    temperature: float = 1.0
    beta: float = 1.0
    epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    seed: int = 42
    gradient_clip_norm: float = 5.0

    def validate(self) -> None:
        if self.loss not in {"bce", "ranknet", "listwise", "jacquard_margin_ranknet"}:
            raise ValueError(f"unknown loss: {self.loss}")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.alpha < 0:
            raise ValueError("invalid optimizer/residual configuration")
        if self.temperature <= 0 or self.beta < 0:
            raise ValueError("invalid listwise/Jacquard configuration")
        if self.epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("epochs, patience, and batch size must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient clip norm must be positive")


@dataclass(frozen=True)
class NeuralTrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_ran: int
    history: tuple[dict[str, float | int], ...]
    state_dict: dict[str, torch.Tensor]
    config: dict[str, Any]


def set_deterministic_cpu(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)


def _batch_indexes(length: int, batch_size: int, *, seed: int, shuffle: bool) -> Iterator[np.ndarray]:
    indexes = np.arange(length)
    if shuffle:
        indexes = np.random.default_rng(int(seed)).permutation(indexes)
    for start in range(0, length, batch_size):
        yield indexes[start : start + batch_size]


def _edge_features(native_scores: torch.Tensor) -> torch.Tensor:
    # [B, source, target, 1], target minus source, matching the directed
    # native-score-difference relation in the frozen relation contract.
    return (native_scores.unsqueeze(1) - native_scores.unsqueeze(2)).unsqueeze(-1)


def _forward(model: nn.Module, batch: QueryArrays, indexes: np.ndarray) -> torch.Tensor:
    features = batch.features[indexes]
    native = batch.native_scores[indexes]
    padding = batch.padding_mask[indexes]
    name = type(model).__name__
    if name == "CompleteGraphGNNResidualScorer":
        edges = (
            _edge_features(native)
            if batch.edge_features is None
            else batch.edge_features[indexes]
        )
        return model(
            features,
            native,
            padding,
            edge_features=edges,
        )
    return model(features, native, padding)


def _loss(
    scores: torch.Tensor,
    batch: QueryArrays,
    indexes: np.ndarray,
    config: NeuralTrainingConfig,
) -> torch.Tensor:
    labels = batch.labels[indexes]
    padding = batch.padding_mask[indexes]
    if config.loss == "bce":
        return query_equal_bce_loss(scores, labels, padding)
    if config.loss == "ranknet":
        return all_pairs_ranknet_loss(scores, labels, padding)
    if config.loss == "listwise":
        return multi_positive_listwise_loss(
            scores, labels, padding, temperature=config.temperature
        )
    return jacquard_margin_ranknet_loss(
        scores,
        labels,
        batch.jacquard_margins[indexes],
        padding,
        beta=config.beta,
    )


def _epoch(
    model: nn.Module,
    arrays: QueryArrays,
    config: NeuralTrainingConfig,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for indexes in _batch_indexes(
            len(arrays),
            config.batch_size,
            seed=config.seed * 1009 + epoch,
            shuffle=training,
        ):
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = _forward(model, arrays, indexes)
            loss = _loss(scores, arrays, indexes, config)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite neural reranking loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            total += float(loss.detach())
            batches += 1
    if batches == 0:
        raise ValueError("training split has no candidate-bearing queries")
    return total / batches


def fit_neural_ranker(
    model: nn.Module,
    train: QueryArrays,
    early_stop: QueryArrays,
    *,
    config: NeuralTrainingConfig,
) -> NeuralTrainingResult:
    config.validate()
    if config.seed not in FORMAL_SEEDS:
        raise ValueError(f"formal learned-method seed must be one of {FORMAL_SEEDS}")
    set_deterministic_cpu(config.seed)
    model = model.cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        training_loss = _epoch(model, train, config, epoch=epoch, optimizer=optimizer)
        validation_loss = _epoch(model, early_stop, config, epoch=epoch, optimizer=None)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("neural ranker did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return NeuralTrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        epochs_ran=len(history),
        history=tuple(history),
        state_dict={key: value.detach().cpu() for key, value in best_state.items()},
        config=asdict(config),
    )


def predict_neural_ranker(
    model: nn.Module,
    arrays: QueryArrays,
    *,
    batch_size: int = 2048,
) -> pd.DataFrame:
    model = model.cpu().eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for indexes in _batch_indexes(len(arrays), batch_size, seed=0, shuffle=False):
            scores = _forward(model, arrays, indexes).detach().cpu().numpy()
            for local, query_index in enumerate(indexes):
                candidate_ids = arrays.candidate_ids[int(query_index)]
                for position, candidate_id in enumerate(candidate_ids):
                    rows.append(
                        {
                            "sample_id": arrays.sample_ids[int(query_index)],
                            "candidate_id": candidate_id,
                            "score": float(scores[local, position]),
                        }
                    )
    return pd.DataFrame(rows)
