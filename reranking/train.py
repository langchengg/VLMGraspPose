"""Deterministic, device-explicit training utilities for neural rerankers."""

from __future__ import annotations

import copy
import csv
from collections.abc import Iterator
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

from reranking.losses import (
    CompositeRerankingLoss,
    LossBreakdown,
    QueryComposition,
    query_composition,
)
from reranking.models.set_models import (
    CandidateGNNScorer,
    PAIRWISE_EDGE_RELATION_FIELDS,
    RAW_EDGE_CANDIDATE_FIELDS,
    build_pairwise_edge_relations,
)


LENGTH_BUCKETED_BATCHING_POLICY = "length_bucketed_v1"


__all__ = [
    "CandidateBatch",
    "CandidateSetDataset",
    "CandidateSetExample",
    "LENGTH_BUCKETED_BATCHING_POLICY",
    "TrainingConfig",
    "TrainingResult",
    "fit_neural_ranker",
    "load_training_checkpoint",
    "make_candidate_dataloader",
    "pad_candidate_sets",
    "resolve_device",
    "set_global_seed",
    "train_model",
]


@dataclass(frozen=True)
class CandidateSetExample:
    """One variable-size query candidate set with labels kept out of features."""

    features: Tensor
    labels: Tensor
    query_id: str
    baseline_scores: Tensor | None = None
    raw_edge_inputs: Tensor | None = None


@dataclass(frozen=True)
class CandidateBatch:
    """Padded query batch; ``padding_mask=True`` marks padding."""

    features: Tensor
    labels: Tensor
    padding_mask: Tensor
    lengths: Tensor
    query_ids: tuple[str, ...]
    baseline_scores: Tensor | None = None
    raw_edge_inputs: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device) -> "CandidateBatch":
        return CandidateBatch(
            features=self.features.to(device),
            labels=self.labels.to(device),
            padding_mask=self.padding_mask.to(device),
            lengths=self.lengths.to(device),
            query_ids=self.query_ids,
            baseline_scores=(
                None
                if self.baseline_scores is None
                else self.baseline_scores.to(device)
            ),
            raw_edge_inputs=(
                None
                if self.raw_edge_inputs is None
                else self.raw_edge_inputs.to(device)
            ),
        )


def _coerce_example(
    example: CandidateSetExample | Mapping[str, Any],
) -> CandidateSetExample:
    if isinstance(example, CandidateSetExample):
        result = example
    elif isinstance(example, Mapping):
        missing = {"features", "labels", "query_id"} - set(example)
        if missing:
            raise ValueError(f"candidate example missing fields: {sorted(missing)}")
        result = CandidateSetExample(
            features=torch.as_tensor(example["features"]),
            labels=torch.as_tensor(example["labels"]),
            query_id=str(example["query_id"]),
            baseline_scores=(
                None
                if example.get("baseline_scores") is None
                else torch.as_tensor(example["baseline_scores"])
            ),
            raw_edge_inputs=(
                None
                if example.get("raw_edge_inputs") is None
                else torch.as_tensor(example["raw_edge_inputs"])
            ),
        )
    else:
        raise TypeError("examples must be CandidateSetExample or mapping objects")
    if not result.query_id:
        raise ValueError("query_id must be non-empty")
    if result.features.ndim != 2 or result.features.shape[1] == 0:
        raise ValueError("example features must have shape [N, D] with D > 0")
    if (
        not result.features.is_floating_point()
        or not torch.isfinite(result.features).all()
    ):
        raise ValueError("example features must be finite floating-point values")
    candidate_count = result.features.shape[0]
    if result.labels.ndim != 1 or result.labels.shape[0] != candidate_count:
        raise ValueError("example labels must have shape [N]")
    labels = result.labels.to(torch.float32)
    if labels.numel() and (
        not torch.isfinite(labels).all()
        or not bool(((labels == 0) | (labels == 1)).all().item())
    ):
        raise ValueError("example labels must be finite binary values")
    if result.baseline_scores is not None:
        baseline = result.baseline_scores
        if baseline.ndim != 1 or baseline.shape[0] != candidate_count:
            raise ValueError("example baseline_scores must have shape [N]")
        if not baseline.is_floating_point() or not torch.isfinite(baseline).all():
            raise ValueError("baseline_scores must be finite floating-point values")
    if result.raw_edge_inputs is not None:
        raw_edges = result.raw_edge_inputs
        expected_shape = (candidate_count, len(RAW_EDGE_CANDIDATE_FIELDS))
        if raw_edges.ndim != 2 or raw_edges.shape != expected_shape:
            raise ValueError(
                "raw_edge_inputs must have shape "
                f"[N, {len(RAW_EDGE_CANDIDATE_FIELDS)}] following "
                "RAW_EDGE_CANDIDATE_FIELDS"
            )
        if not raw_edges.is_floating_point() or not torch.isfinite(raw_edges).all():
            raise ValueError("raw_edge_inputs must be finite floating-point values")
    return result


class CandidateSetDataset(Dataset[CandidateSetExample]):
    """Validated in-memory dataset for variable-size candidate sets."""

    def __init__(self, examples: list[CandidateSetExample | Mapping[str, Any]]) -> None:
        if not examples:
            raise ValueError("at least one candidate-set example is required")
        self._examples = tuple(_coerce_example(example) for example in examples)
        raw_presence = [
            example.raw_edge_inputs is not None for example in self._examples
        ]
        if any(raw_presence) and not all(raw_presence):
            raise ValueError(
                "raw_edge_inputs must be present for every dataset example or none"
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> CandidateSetExample:
        return self._examples[index]


class _CandidateLengthBatchSampler(Sampler[list[int]]):
    """Shuffle complete length-sorted batches without padding-heavy mixing."""

    def __init__(
        self,
        candidate_lengths: tuple[int, ...],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        if not candidate_lengths:
            raise ValueError("candidate_lengths must be non-empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(length < 0 for length in candidate_lengths):
            raise ValueError("candidate lengths must be non-negative")
        self._candidate_lengths = candidate_lengths
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return math.ceil(len(self._candidate_lengths) / self._batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        if not self._shuffle:
            ordered = sorted(
                range(len(self._candidate_lengths)),
                key=self._candidate_lengths.__getitem__,
            )
            yield from (
                ordered[start : start + self._batch_size]
                for start in range(0, len(ordered), self._batch_size)
            )
            return

        # The initial permutation supplies deterministic random tie-breaking.
        # Stable sorting then keeps adjacent candidate counts together, while
        # the final permutations randomize both batch and within-batch order on
        # every new DataLoader iteration (one iteration per training epoch).
        tie_broken = torch.randperm(
            len(self._candidate_lengths), generator=self._generator
        ).tolist()
        ordered = sorted(tie_broken, key=self._candidate_lengths.__getitem__)
        batches = [
            ordered[start : start + self._batch_size]
            for start in range(0, len(ordered), self._batch_size)
        ]
        batch_order = torch.randperm(len(batches), generator=self._generator).tolist()
        for batch_index in batch_order:
            batch = batches[batch_index]
            within_batch = torch.randperm(
                len(batch), generator=self._generator
            ).tolist()
            yield [batch[index] for index in within_batch]


def pad_candidate_sets(
    examples: list[CandidateSetExample | Mapping[str, Any]],
) -> CandidateBatch:
    """Pad variable candidate counts without dropping valid-empty queries."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    prepared = [_coerce_example(example) for example in examples]
    feature_dim = prepared[0].features.shape[1]
    if any(example.features.shape[1] != feature_dim for example in prepared):
        raise ValueError("all examples must use the same feature dimension")
    baseline_presence = [example.baseline_scores is not None for example in prepared]
    if any(baseline_presence) and not all(baseline_presence):
        raise ValueError("baseline_scores must be present for every example or none")
    raw_edge_presence = [example.raw_edge_inputs is not None for example in prepared]
    if any(raw_edge_presence) and not all(raw_edge_presence):
        raise ValueError("raw_edge_inputs must be present for every example or none")

    batch_size = len(prepared)
    lengths = torch.tensor(
        [example.features.shape[0] for example in prepared], dtype=torch.long
    )
    max_candidates = int(lengths.max().item())
    dtype = prepared[0].features.dtype
    features = torch.zeros(batch_size, max_candidates, feature_dim, dtype=dtype)
    labels = torch.zeros(batch_size, max_candidates, dtype=dtype)
    padding = torch.ones(batch_size, max_candidates, dtype=torch.bool)
    baseline = (
        torch.zeros(batch_size, max_candidates, dtype=dtype)
        if all(baseline_presence)
        else None
    )
    raw_edge_inputs = (
        torch.zeros(
            batch_size,
            max_candidates,
            len(RAW_EDGE_CANDIDATE_FIELDS),
            dtype=dtype,
        )
        if all(raw_edge_presence)
        else None
    )
    for row, example in enumerate(prepared):
        count = example.features.shape[0]
        if count == 0:
            continue
        features[row, :count] = example.features.to(dtype=dtype, device="cpu")
        labels[row, :count] = example.labels.to(dtype=dtype, device="cpu")
        padding[row, :count] = False
        if baseline is not None:
            assert example.baseline_scores is not None
            baseline[row, :count] = example.baseline_scores.to(
                dtype=dtype, device="cpu"
            )
        if raw_edge_inputs is not None:
            assert example.raw_edge_inputs is not None
            raw_edge_inputs[row, :count] = example.raw_edge_inputs.to(
                dtype=dtype, device="cpu"
            )
    return CandidateBatch(
        features=features,
        labels=labels,
        padding_mask=padding,
        lengths=lengths,
        query_ids=tuple(example.query_id for example in prepared),
        baseline_scores=baseline,
        raw_edge_inputs=raw_edge_inputs,
    )


def make_candidate_dataloader(
    examples: list[CandidateSetExample | Mapping[str, Any]] | CandidateSetDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    batching_policy: str | None = None,
) -> DataLoader[CandidateBatch]:
    """Build a seeded loader with optional candidate-length-aware batching.

    ``batching_policy=None`` preserves the legacy DataLoader construction
    exactly.  ``"length_bucketed_v1"`` groups adjacent candidate counts.
    Training loaders reshuffle batch and row order on every epoch; validation
    loaders use a stable ascending-length order on every iteration.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    dataset = (
        examples
        if isinstance(examples, CandidateSetDataset)
        else CandidateSetDataset(examples)
    )
    generator = torch.Generator().manual_seed(int(seed))
    if batching_policy is not None:
        if batching_policy != LENGTH_BUCKETED_BATCHING_POLICY:
            raise ValueError(
                f"batching_policy must be None or '{LENGTH_BUCKETED_BATCHING_POLICY}'"
            )
        batch_sampler = _CandidateLengthBatchSampler(
            tuple(
                int(dataset[index].features.shape[0]) for index in range(len(dataset))
            ),
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=pad_candidate_sets,
            generator=generator,
            persistent_workers=num_workers > 0,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_candidate_sets,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve an explicit CPU/MPS device and fail clearly when unavailable."""

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "mps"}:
        raise ValueError("neural reranking supports explicit 'cpu' or 'mps' devices")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return resolved


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch CPU, and MPS random generators."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.0
    gradient_clip_norm: float | None = 5.0
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive or None")
        resolve_device(self.device)


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    epochs_ran: int
    stopped_early: bool
    history: tuple[dict[str, Any], ...]
    checkpoint_path: str
    metadata_path: str
    last_checkpoint_path: str
    history_path: str
    device: str
    seed: int


@dataclass(frozen=True)
class _EpochResult:
    loss: float
    active_queries: int
    composition: QueryComposition
    raw_edge_inputs_present: bool
    raw_edge_inputs_used: bool


def _forward_candidate_batch(
    model: nn.Module, batch: CandidateBatch
) -> tuple[Tensor, bool]:
    """Forward one batch without leaking GNN-only kwargs to other scorers."""

    common = {
        "baseline_scores": batch.baseline_scores,
        "padding_mask": batch.padding_mask,
    }
    if not isinstance(model, CandidateGNNScorer):
        return model(batch.features, **common), False
    if batch.raw_edge_inputs is None:
        # An edge-conditioned CandidateGNNScorer will fail closed in its own
        # forward contract.  An adjacency-only edge_dim=0 GNN remains valid.
        return model(batch.features, **common), False
    if model.graph_type == "explicit":
        raise ValueError(
            "the generic trainer cannot infer explicit edge_index; use complete/knn "
            "or a custom loader carrying explicit graph topology"
        )
    relation_count = len(PAIRWISE_EDGE_RELATION_FIELDS)
    selected_indices = tuple(
        int(index) for index in getattr(model, "edge_feature_indices", ())
    )
    if len(set(selected_indices)) != len(selected_indices) or any(
        index < 0 or index >= relation_count for index in selected_indices
    ):
        raise ValueError("invalid CandidateGNNScorer edge_feature_indices")
    expected_edge_dim = len(selected_indices) if selected_indices else relation_count
    if model.edge_dim not in {0, expected_edge_dim}:
        raise ValueError(
            "CandidateGNNScorer edge_dim must be zero or match its explicit raw "
            f"relation selection ({expected_edge_dim}), got {model.edge_dim}"
        )
    if model.graph_type == "knn":
        if model.edge_dim == 0:
            scores = model(
                batch.features,
                **common,
                coordinates=batch.raw_edge_inputs[..., :2],
            )
        else:
            scores = model(
                batch.features,
                **common,
                raw_edge_inputs=batch.raw_edge_inputs,
            )
        return scores, True

    edge_features = build_pairwise_edge_relations(
        batch.raw_edge_inputs, batch.padding_mask
    )
    if selected_indices:
        edge_features = edge_features[..., list(selected_indices)]
    scores = model(
        batch.features,
        **common,
        coordinates=batch.raw_edge_inputs[..., :2],
        edge_features=edge_features,
    )
    used_raw_edges = bool(model.edge_dim > 0 or model.graph_type in {"knn", "rule"})
    return scores, used_raw_edges


def _run_epoch(
    model: nn.Module,
    batches: Any,
    criterion: CompositeRerankingLoss,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
) -> _EpochResult:
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    active_queries = 0
    composition = QueryComposition(0, 0, 0, 0, 0)
    raw_edge_presence: bool | None = None
    raw_edge_used = False

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in batches:
            if not isinstance(raw_batch, CandidateBatch):
                raise TypeError("training iterables must yield CandidateBatch objects")
            batch_composition = query_composition(
                raw_batch.labels, raw_batch.padding_mask
            )
            batch = raw_batch.to(device)
            batch_has_raw_edges = batch.raw_edge_inputs is not None
            if raw_edge_presence is None:
                raw_edge_presence = batch_has_raw_edges
            elif raw_edge_presence != batch_has_raw_edges:
                raise ValueError(
                    "raw_edge_inputs presence must be consistent across an epoch"
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores, used_raw_edges = _forward_candidate_batch(model, batch)
            raw_edge_used |= used_raw_edges
            breakdown: LossBreakdown = criterion.active_breakdown(
                scores,
                batch.labels,
                batch.padding_mask,
                baseline_scores=batch.baseline_scores,
                composition=batch_composition,
            )
            composition = composition + breakdown.composition
            if breakdown.active_queries == 0:
                continue
            detached_total = float(breakdown.total.detach().cpu())
            if not math.isfinite(detached_total):
                raise RuntimeError("non-finite training loss")
            if training:
                breakdown.total.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            weighted_loss += detached_total * breakdown.active_queries
            active_queries += breakdown.active_queries
    if active_queries == 0:
        raise ValueError(
            "loss has no eligible queries: ranking-only losses require at least "
            "one query containing both positive and negative candidates"
        )
    return _EpochResult(
        loss=weighted_loss / active_queries,
        active_queries=active_queries,
        composition=composition,
        raw_edge_inputs_present=bool(raw_edge_presence),
        raw_edge_inputs_used=raw_edge_used,
    )


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _json_safe(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        # torch.__version__ is a str subclass that the safe unpickler rejects.
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checkpoint(
    path: Path,
    payload: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    metadata = dict(metadata)
    metadata["checkpoint_sha256"] = _sha256(path)
    metadata_path = path.with_name(path.name + ".metadata.json")
    metadata_temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    metadata_temporary.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    return metadata_path


def fit_neural_ranker(
    model: nn.Module,
    train_batches: Any,
    validation_batches: Any,
    *,
    checkpoint_path: str | Path,
    criterion: CompositeRerankingLoss | None = None,
    config: TrainingConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train with validation early stopping and restore the best checkpoint state."""

    config = config or TrainingConfig()
    config.validate()
    criterion = criterion or CompositeRerankingLoss()
    device = resolve_device(config.device)
    set_global_seed(config.seed)
    model = model.to(device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    stopped_early = False
    last_train_composition = QueryComposition(0, 0, 0, 0, 0)
    last_validation_composition = QueryComposition(0, 0, 0, 0, 0)

    for epoch in range(1, config.epochs + 1):
        train_result = _run_epoch(
            model,
            train_batches,
            criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_result = _run_epoch(
            model,
            validation_batches,
            criterion,
            device=device,
            optimizer=None,
            gradient_clip_norm=None,
        )
        if (
            train_result.raw_edge_inputs_present
            != validation_result.raw_edge_inputs_present
        ):
            raise ValueError(
                "train and validation loaders must agree on raw_edge_inputs presence"
            )
        if train_result.raw_edge_inputs_used != validation_result.raw_edge_inputs_used:
            raise ValueError(
                "train and validation loaders must agree on raw edge usage"
            )
        last_train_composition = train_result.composition
        last_validation_composition = validation_result.composition
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "validation_loss": validation_result.loss,
                "train_active_queries": train_result.active_queries,
                "validation_active_queries": validation_result.active_queries,
                "train_query_composition": train_result.composition.as_dict(),
                "validation_query_composition": validation_result.composition.as_dict(),
                "raw_edge_inputs_present": train_result.raw_edge_inputs_present,
                "raw_edge_inputs_used": train_result.raw_edge_inputs_used,
            }
        )

        if best_state is None or validation_result.loss < best_loss - config.min_delta:
            best_loss = validation_result.loss
            best_epoch = epoch
            best_state = _cpu_tree(model.state_dict())
            best_optimizer_state = _cpu_tree(optimizer.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                stopped_early = True
                break

    assert best_state is not None and best_optimizer_state is not None
    selected_relation_indices = tuple(
        int(index) for index in getattr(model, "edge_feature_indices", ())
    )
    sparse_knn = isinstance(model, CandidateGNNScorer) and model.graph_type == "knn"
    if sparse_knn and model.edge_dim == 0:
        relation_builder = "not_materialized"
        relation_materialization = "adjacency_only"
        relation_layout = "not_materialized"
        selected_relation_fields: list[str] = []
    elif sparse_knn:
        relation_builder = (
            "reranking.models.set_models.build_indexed_pairwise_edge_relations"
        )
        relation_materialization = "selected_knn_edges_only"
        relation_layout = "edge,relation"
        selected_relation_fields = [
            PAIRWISE_EDGE_RELATION_FIELDS[index] for index in selected_relation_indices
        ] or list(PAIRWISE_EDGE_RELATION_FIELDS)
    else:
        relation_builder = "reranking.models.set_models.build_pairwise_edge_relations"
        relation_materialization = "dense_pairwise_fallback"
        relation_layout = "batch,source,destination,relation"
        selected_relation_fields = [
            PAIRWISE_EDGE_RELATION_FIELDS[index] for index in selected_relation_indices
        ] or list(PAIRWISE_EDGE_RELATION_FIELDS)
    raw_edge_schema = {
        "schema_version": 1,
        "present": train_result.raw_edge_inputs_present,
        "used_by_model": train_result.raw_edge_inputs_used,
        "candidate_fields": list(RAW_EDGE_CANDIDATE_FIELDS),
        "relation_fields": list(PAIRWISE_EDGE_RELATION_FIELDS),
        "selected_relation_indices": list(selected_relation_indices),
        "selected_relation_fields": selected_relation_fields,
        "relation_layout": relation_layout,
        "builder": relation_builder,
        "materialization": relation_materialization,
        "coordinate_contract": "x/y/width share unstandardized units; theta is radians",
    }
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    last_checkpoint = checkpoint.with_name(f"{checkpoint.stem}.last{checkpoint.suffix}")
    history_path = checkpoint.with_name(f"{checkpoint.stem}.history.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_temporary = history_path.with_name(history_path.name + ".tmp")
    history_columns = (
        "epoch",
        "train_loss",
        "validation_loss",
        "train_active_queries",
        "validation_active_queries",
        "train_query_composition",
        "validation_query_composition",
        "raw_edge_inputs_present",
        "raw_edge_inputs_used",
    )
    with history_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history_columns)
        writer.writeheader()
        for row in history:
            encoded = dict(row)
            encoded["train_query_composition"] = json.dumps(
                encoded["train_query_composition"], sort_keys=True
            )
            encoded["validation_query_composition"] = json.dumps(
                encoded["validation_query_composition"], sort_keys=True
            )
            writer.writerow(encoded)
    history_temporary.replace(history_path)

    last_state = _cpu_tree(model.state_dict())
    last_optimizer_state = _cpu_tree(optimizer.state_dict())
    last_payload = {
        "format_version": 1,
        "model_state_dict": last_state,
        "optimizer_state_dict": last_optimizer_state,
        "epoch": len(history),
        "training_config": asdict(config),
        "loss_config": criterion.config,
        "checkpoint_role": "last_epoch",
        "raw_edge_schema": raw_edge_schema,
    }
    last_metadata_path = _write_checkpoint(
        last_checkpoint,
        last_payload,
        {
            "checkpoint_role": "last_epoch",
            "epoch": len(history),
            "seed": config.seed,
            "device": str(device),
            "history_path": str(history_path),
            "history_sha256": _sha256(history_path),
            "raw_edge_schema": raw_edge_schema,
        },
    )
    model.load_state_dict(best_state)
    run_metadata: dict[str, Any] = {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "training_config": asdict(config),
        "loss_config": criterion.config,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_ran": len(history),
        "stopped_early": stopped_early,
        "seed": config.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "raw_edge_schema": raw_edge_schema,
        "query_semantics": {
            "padding": "ignored",
            "empty": "ignored by every loss",
            "no_positive": "BCE eligible; RankNet/listwise ineligible",
            "all_positive": "BCE eligible; RankNet/listwise ineligible",
        },
        "final_train_query_composition": last_train_composition.as_dict(),
        "final_validation_query_composition": last_validation_composition.as_dict(),
        "history": history,
        "history_path": str(history_path),
        "history_sha256": _sha256(history_path),
        "last_checkpoint_path": str(last_checkpoint),
        "last_checkpoint_sha256": _sha256(last_checkpoint),
        "last_checkpoint_metadata_path": str(last_metadata_path),
        "user_metadata": dict(metadata or {}),
    }
    payload = {
        "format_version": 1,
        "model_state_dict": best_state,
        "optimizer_state_dict": best_optimizer_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_config": asdict(config),
        "loss_config": criterion.config,
        "metadata": _json_safe(run_metadata),
    }
    metadata_path = _write_checkpoint(checkpoint, payload, run_metadata)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        epochs_ran=len(history),
        stopped_early=stopped_early,
        history=tuple(history),
        checkpoint_path=str(checkpoint),
        metadata_path=str(metadata_path),
        last_checkpoint_path=str(last_checkpoint),
        history_path=str(history_path),
        device=str(device),
        seed=config.seed,
    )


def load_training_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint produced by :func:`fit_neural_ranker`."""

    resolved_device = resolve_device(device)
    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location=resolved_device, weights_only=True)
    if payload.get("format_version") != 1 or "model_state_dict" not in payload:
        raise ValueError("unsupported or malformed reranking checkpoint")
    model.to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


train_model = fit_neural_ranker
