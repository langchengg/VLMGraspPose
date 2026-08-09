"""Controlled local rankers for one immutable G1/C1 candidate table."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

from .contracts import assert_candidate_identity
from .features import CONTINUOUS_FEATURE_GROUPS, DEFAULT_FEATURE_COLUMNS


LABEL_COLUMN = "candidate_correct"
METHODS = (
    "logistic_pointwise",
    "residual_linear_listwise",
    "residual_mlp_bce",
    "residual_mlp_ranknet",
    "residual_mlp_listwise",
    "deepsets",
    "set_transformer",
    "candidate_gnn",
)


@dataclass
class TrainOnlyPreprocessor:
    feature_columns: tuple[str, ...]
    median: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    source_splits: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyPreprocessor":
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"model features missing: {missing}")
        if "split" not in frame:
            raise ValueError("preprocessor requires an explicit split")
        splits = tuple(sorted(frame["split"].astype(str).unique()))
        if not set(splits).issubset({"train", "development"}):
            raise ValueError("preprocessor may only fit development data")
        matrix = frame.loc[:, self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        matrix[~np.isfinite(matrix)] = np.nan
        safe_matrix = matrix.copy()
        safe_matrix[:, np.isnan(safe_matrix).all(axis=0)] = 0.0
        self.median = np.nanmedian(safe_matrix, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, self.median)
        self.mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.scale = np.where(std > 1e-12, std, 1.0)
        self.source_splits = splits
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.median is None or self.mean is None or self.scale is None:
            raise RuntimeError("preprocessor is not fitted")
        matrix = frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        filled = np.where(np.isfinite(matrix), matrix, self.median)
        output = (filled - self.mean) / self.scale
        if not np.isfinite(output).all():
            raise ValueError("preprocessed features are non-finite")
        return output.astype(np.float32)

    def artifact(self) -> dict[str, Any]:
        if self.median is None or self.mean is None or self.scale is None:
            raise RuntimeError("preprocessor is not fitted")
        return {
            "feature_columns": list(self.feature_columns),
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "source_splits": list(self.source_splits),
            "missing_value_rule": "train-only median plus explicit reliability flags",
        }


def _sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("sample_id")["stable_candidate_id"].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def attach_scores(frame: pd.DataFrame, scores: Sequence[float], *, method: str) -> pd.DataFrame:
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("one finite score is required for every frozen candidate")
    output = frame.copy()
    output["reranker_method"] = str(method)
    output["reranker_score"] = values
    ordered = output.sort_values(
        ["sample_id", "reranker_score", "stable_candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()
    ordered["reranker_rank"] = ordered.groupby("sample_id", sort=False).cumcount() + 1
    output["reranker_rank"] = ordered["reranker_rank"]
    assert_candidate_identity(frame, output)
    return output


MANUAL_FEATURE_DIRECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "mask": (
        ("center_probability", "positive"),
        ("rectangle_mask_coverage", "positive"),
        ("grasp_axis_mask_support", "positive"),
        ("left_contact_mask_support", "positive"),
        ("right_contact_mask_support", "positive"),
        ("contact_support_minimum", "positive"),
        ("contact_support_imbalance", "negative"),
        ("signed_distance_to_mask_boundary_normalized", "positive"),
        ("largest_component_ratio", "positive"),
        ("foreground_probability_mean", "positive"),
        ("foreground_probability_entropy", "negative"),
        ("candidate_rectangle_overflow_ratio", "negative"),
        ("sweep_region_mask_support", "positive"),
    ),
    "width": (
        ("candidate_width_over_target_extent", "ideal_one"),
        ("width_compatibility", "positive"),
    ),
    "depth": (
        ("local_valid_depth_fraction", "positive"),
        ("left_contact_valid_fraction", "positive"),
        ("right_contact_valid_fraction", "positive"),
        ("absolute_contact_depth_difference_m", "negative"),
        ("contact_depth_mad_m", "negative"),
        ("local_depth_std_m", "negative"),
        ("local_depth_mad_m", "negative"),
        ("depth_gradient_along_closing_axis", "absolute_negative"),
        ("depth_gradient_orthogonal_axis", "absolute_negative"),
    ),
    "clearance": (
        ("gripper_sweep_valid_fraction", "positive"),
        ("sweep_foreground_fraction", "positive"),
        ("background_intrusion_ratio", "negative"),
        ("approach_collision_proxy", "negative"),
        ("sweep_minimum_clearance_proxy", "positive"),
    ),
    "relations": (
        ("nearest_candidate_width_ratio", "ideal_one"),
        ("maximum_rectangle_iou_with_other_candidate", "positive"),
        ("similar_pose_fraction", "positive"),
        ("backend_consensus_count", "positive"),
    ),
    "reliability": tuple(
        (column, "positive")
        for column in CONTINUOUS_FEATURE_GROUPS["reliability"]
    ),
}

MANUAL_PEAK_FEATURE_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("local_probability_mean", "positive"),
    ("local_probability_max", "positive"),
    ("local_probability_min", "positive"),
    ("local_probability_std", "negative"),
    ("grasp_axis_mask_support", "positive"),
)


def _directional_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "positive":
        return values
    if transform == "negative":
        return -values
    if transform == "absolute_negative":
        return -np.abs(values)
    if transform == "ideal_one":
        return -np.abs(np.log(np.clip(values, 1e-6, None)))
    raise ValueError(f"unknown manual feature transform: {transform}")


def _directional_evidence(
    frame: pd.DataFrame,
    specification: Sequence[tuple[str, str]],
) -> np.ndarray:
    available = [(column, transform) for column, transform in specification if column in frame]
    if not available:
        raise ValueError("manual evidence has no available directed features")
    result = np.zeros(len(frame), dtype=float)
    for _, indices in frame.groupby("sample_id", sort=False).indices.items():
        index = np.asarray(indices, dtype=int)
        local = frame.iloc[index]
        contributions: list[np.ndarray] = []
        for column, transform in available:
            values = pd.to_numeric(local[column], errors="coerce").to_numpy(
                dtype=float, copy=True
            )
            values[~np.isfinite(values)] = np.nan
            median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            directed = _directional_transform(
                np.where(np.isfinite(values), values, median), transform
            )
            low, high = float(directed.min()), float(directed.max())
            contributions.append(
                np.zeros_like(directed)
                if high - low <= 1e-12
                else (directed - low) / (high - low)
            )
        result[index] = np.column_stack(contributions).mean(axis=1)
    return result


def manual_feature_spec(groups: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in groups:
        if str(group) == "F2_peak":
            specification = MANUAL_PEAK_FEATURE_DIRECTIONS
        else:
            specification = MANUAL_FEATURE_DIRECTIONS[str(group)]
        rows.extend(
            {"group": str(group), "feature": feature, "transform": transform}
            for feature, transform in specification
        )
    return rows


def manual_score(frame: pd.DataFrame, groups: Sequence[str], weights: Mapping[str, float] | None = None) -> np.ndarray:
    """Direction-constrained feature evidence, normalized within each sample."""

    groups = tuple(str(group) for group in groups if str(group) != "F2_peak")
    if not groups:
        raise ValueError("manual score requires at least one directional group")
    weight_map = {str(key): float(value) for key, value in (weights or {}).items()}
    result = np.zeros(len(frame), dtype=float)
    for group in groups:
        result += weight_map.get(group, 1.0) * _directional_evidence(
            frame, MANUAL_FEATURE_DIRECTIONS[group]
        )
    return result


def manual_peak_score(frame: pd.DataFrame, peak_columns: Sequence[str]) -> np.ndarray:
    """Direction-constrained within-sample F2 peak-confidence evidence."""

    requested = set(map(str, peak_columns))
    specification = tuple(
        (column, transform)
        for column, transform in MANUAL_PEAK_FEATURE_DIRECTIONS
        if column in requested
    )
    return _directional_evidence(frame, specification)


def manual_method_score(
    frame: pd.DataFrame,
    groups: Sequence[str],
    peak_columns: Sequence[str],
) -> np.ndarray:
    groups = tuple(map(str, groups))
    components: list[np.ndarray] = []
    directional_groups = tuple(group for group in groups if group != "F2_peak")
    if directional_groups:
        components.append(manual_score(frame, directional_groups))
    if "F2_peak" in groups:
        components.append(manual_peak_score(frame, peak_columns))
    if not components:
        raise ValueError("manual method requires directional or peak evidence")
    return np.sum(np.column_stack(components), axis=1)


class LogisticPointwiseRanker:
    method_name = "logistic_pointwise"

    def __init__(self, feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS, *, c: float = 0.1, seed: int = 42) -> None:
        self.feature_columns = tuple(map(str, feature_columns))
        self.c = float(c)
        self.seed = int(seed)
        self.preprocessor = TrainOnlyPreprocessor(self.feature_columns)
        self.model: LogisticRegression | None = None

    def fit(self, frame: pd.DataFrame) -> "LogisticPointwiseRanker":
        if LABEL_COLUMN not in frame:
            raise ValueError("candidate labels are required for training")
        x = self.preprocessor.fit(frame).transform(frame)
        y = frame[LABEL_COLUMN].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError("pointwise logistic needs both label classes")
        self.model = LogisticRegression(
            C=self.c,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=self.seed,
        )
        self.model.fit(x, y, sample_weight=_sample_weights(frame))
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        return self.model.decision_function(self.preprocessor.transform(frame)).astype(float)

    def artifact(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        return {
            "method": self.method_name,
            "C": self.c,
            "seed": self.seed,
            "preprocessor": self.preprocessor.artifact(),
            "coefficient": self.model.coef_[0].tolist(),
            "intercept": float(self.model.intercept_[0]),
        }


@dataclass(frozen=True)
class NeuralConfig:
    hidden: int = 64
    embedding: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 4
    residual_alpha: float = 0.5
    temperature: float = 1.0
    sample_batch_size: int = 256
    device: str = "auto"
    seed: int = 42


class _CandidateEncoder(nn.Module):
    def __init__(self, feature_count: int, config: NeuralConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.embedding),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _ControlledMLP(nn.Module):
    def __init__(self, feature_count: int, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _CandidateEncoder(feature_count, config)
        self.head = nn.Linear(config.embedding, 1)

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor, sample_ids: Sequence[str]) -> torch.Tensor:
        del sample_ids
        residual = torch.tanh(self.head(self.encoder(x)).squeeze(-1))
        return base_logit + self.config.residual_alpha * residual


class _ControlledLinear(nn.Module):
    def __init__(self, feature_count: int, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        self.normalization = nn.LayerNorm(feature_count)
        self.head = nn.Linear(feature_count, 1)

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor, sample_ids: Sequence[str]) -> torch.Tensor:
        del sample_ids
        residual = torch.tanh(self.head(self.normalization(x)).squeeze(-1))
        return base_logit + self.config.residual_alpha * residual


class _DeepSets(nn.Module):
    def __init__(self, feature_count: int, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden * 3, config.hidden), nn.GELU(), nn.Linear(config.hidden, 1)
        )

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor, sample_ids: Sequence[str]) -> torch.Tensor:
        embedding = self.encoder(x)
        ids = np.asarray(list(map(str, sample_ids)), dtype=object)
        context = torch.empty((len(ids), embedding.shape[1] * 2), device=x.device, dtype=x.dtype)
        for sample_id in dict.fromkeys(ids.tolist()):
            index = torch.as_tensor(np.flatnonzero(ids == sample_id), device=x.device)
            group = embedding[index]
            pooled = torch.cat([group.mean(dim=0), group.max(dim=0).values])
            context[index] = pooled
        residual = torch.tanh(self.head(torch.cat([embedding, context], dim=1)).squeeze(-1))
        return base_logit + self.config.residual_alpha * residual


class _SetTransformer(nn.Module):
    """Permutation-equivariant candidate scorer without positional encodings."""

    def __init__(self, feature_count: int, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.Linear(feature_count, config.hidden),
            nn.GELU(),
            nn.LayerNorm(config.hidden),
        )
        self.attention = nn.MultiheadAttention(
            config.hidden,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_one = nn.LayerNorm(config.hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden, config.hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden * 2, config.hidden),
        )
        self.norm_two = nn.LayerNorm(config.hidden)
        self.head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.GELU(), nn.Linear(config.hidden, 1)
        )

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor, sample_ids: Sequence[str]) -> torch.Tensor:
        ids = np.asarray(list(map(str, sample_ids)), dtype=object)
        output = torch.empty(len(ids), device=x.device, dtype=x.dtype)
        for sample_id in dict.fromkeys(ids.tolist()):
            index = torch.as_tensor(np.flatnonzero(ids == sample_id), device=x.device)
            hidden = self.input(x[index]).unsqueeze(0)
            attended, _ = self.attention(hidden, hidden, hidden, need_weights=False)
            hidden = self.norm_one(hidden + attended)
            hidden = self.norm_two(hidden + self.feed_forward(hidden))
            residual = torch.tanh(self.head(hidden).squeeze(0).squeeze(-1))
            output[index] = base_logit[index] + self.config.residual_alpha * residual
        return output


class _CandidateGraphSAGE(nn.Module):
    """Two-layer edge-conditioned complete candidate GNN without PyG."""

    def __init__(self, feature_count: int, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _CandidateEncoder(feature_count, config)
        message_input = config.embedding * 2 + feature_count
        self.message_one = nn.Sequential(
            nn.Linear(message_input, config.embedding), nn.GELU(), nn.Linear(config.embedding, config.embedding)
        )
        self.update_one = nn.Sequential(
            nn.Linear(config.embedding * 2, config.embedding), nn.GELU(), nn.LayerNorm(config.embedding)
        )
        self.message_two = nn.Sequential(
            nn.Linear(message_input, config.embedding), nn.GELU(), nn.Linear(config.embedding, config.embedding)
        )
        self.update_two = nn.Sequential(
            nn.Linear(config.embedding * 2, config.embedding), nn.GELU(), nn.LayerNorm(config.embedding)
        )
        self.head = nn.Linear(config.embedding, 1)

    def _layer(self, values: torch.Tensor, x: torch.Tensor, ids: np.ndarray, message: nn.Module, update: nn.Module) -> torch.Tensor:
        output = torch.empty_like(values)
        for sample_id in dict.fromkeys(ids.tolist()):
            index = torch.as_tensor(np.flatnonzero(ids == sample_id), device=values.device)
            group = values[index]
            if len(index) == 1:
                aggregate = torch.zeros_like(group)
            else:
                source = group[:, None, :].expand(-1, len(index), -1)
                destination = group[None, :, :].expand(len(index), -1, -1)
                edge_delta = x[index][None, :, :] - x[index][:, None, :]
                messages = message(torch.cat([source, destination, edge_delta], dim=-1))
                keep = ~torch.eye(len(index), dtype=torch.bool, device=x.device)
                messages = messages * keep.unsqueeze(-1)
                aggregate = messages.sum(dim=0) / float(len(index) - 1)
            output[index] = update(torch.cat([group, aggregate], dim=-1)) + group
        return output

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor, sample_ids: Sequence[str]) -> torch.Tensor:
        ids = np.asarray(list(map(str, sample_ids)), dtype=object)
        values = self.encoder(x)
        values = self._layer(values, x, ids, self.message_one, self.update_one)
        values = self._layer(values, x, ids, self.message_two, self.update_two)
        residual = torch.tanh(self.head(values).squeeze(-1))
        return base_logit + self.config.residual_alpha * residual


def _ranknet_loss(scores: torch.Tensor, labels: torch.Tensor, ids: Sequence[str]) -> torch.Tensor:
    names = np.asarray(list(map(str, ids)), dtype=object)
    losses: list[torch.Tensor] = []
    for sample_id in dict.fromkeys(names.tolist()):
        index = torch.as_tensor(np.flatnonzero(names == sample_id), device=scores.device)
        local_y = labels[index]
        positive, negative = index[local_y > 0.5], index[local_y <= 0.5]
        if len(positive) and len(negative):
            losses.append(F.softplus(scores[negative][None, :] - scores[positive][:, None]).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def _listwise_loss(scores: torch.Tensor, labels: torch.Tensor, ids: Sequence[str]) -> torch.Tensor:
    names = np.asarray(list(map(str, ids)), dtype=object)
    losses: list[torch.Tensor] = []
    for sample_id in dict.fromkeys(names.tolist()):
        index = torch.as_tensor(np.flatnonzero(names == sample_id), device=scores.device)
        local_y = labels[index] > 0.5
        if bool(local_y.any()) and not bool(local_y.all()):
            losses.append(torch.logsumexp(scores[index], dim=0) - torch.logsumexp(scores[index][local_y], dim=0))
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


class NeuralRanker:
    def __init__(self, method: str, feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS, *, config: NeuralConfig = NeuralConfig()) -> None:
        if method not in METHODS[1:]:
            raise ValueError(f"unknown neural method: {method}")
        self.method_name = str(method)
        self.feature_columns = tuple(map(str, feature_columns))
        self.config = config
        self.preprocessor = TrainOnlyPreprocessor(self.feature_columns)
        self.device = self._device(config.device)
        self.model: nn.Module | None = None
        self.history: list[dict[str, float]] = []
        self.best_epoch = -1

    @staticmethod
    def _device(requested: str) -> torch.device:
        if requested not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be auto, cpu, or mps")
        available = bool(torch.backends.mps.is_available())
        if requested == "mps" and not available:
            raise RuntimeError("MPS is unavailable")
        return torch.device("mps" if requested in {"auto", "mps"} and available else "cpu")

    def _make_model(self) -> nn.Module:
        if self.method_name == "residual_linear_listwise":
            return _ControlledLinear(len(self.feature_columns), self.config)
        if self.method_name == "deepsets":
            return _DeepSets(len(self.feature_columns), self.config)
        if self.method_name == "candidate_gnn":
            return _CandidateGraphSAGE(len(self.feature_columns), self.config)
        if self.method_name == "set_transformer":
            return _SetTransformer(len(self.feature_columns), self.config)
        return _ControlledMLP(len(self.feature_columns), self.config)

    def _objective(self, scores: torch.Tensor, labels: torch.Tensor, ids: Sequence[str]) -> torch.Tensor:
        if self.method_name == "residual_mlp_ranknet":
            return _ranknet_loss(scores, labels, ids)
        if self.method_name in {"residual_linear_listwise", "residual_mlp_listwise", "deepsets", "set_transformer", "candidate_gnn"}:
            return _listwise_loss(scores / self.config.temperature, labels, ids)
        # Per-query candidate/class weighting.  The local renormalization after
        # class weighting preserves equal total mass for every sample.
        names = pd.Series(list(map(str, ids)))
        positive = labels.sum().clamp_min(1.0)
        negative = (1.0 - labels).sum().clamp_min(1.0)
        class_weights = torch.where(labels > 0.5, negative / positive, torch.ones_like(labels))
        candidate_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none") * class_weights
        query_losses: list[torch.Tensor] = []
        names_array = names.to_numpy(dtype=object)
        for sample_id in dict.fromkeys(names_array.tolist()):
            index = torch.as_tensor(np.flatnonzero(names_array == sample_id), device=scores.device)
            query_losses.append(
                candidate_loss[index].sum()
                / class_weights[index].sum().clamp_min(1e-9)
            )
        return torch.stack(query_losses).mean()

    def _groups(self, frame: pd.DataFrame, *, shuffle: bool, epoch: int = 0) -> list[np.ndarray]:
        groups = [np.asarray(index, dtype=int) for index in frame.groupby("sample_id", sort=False).indices.values()]
        if shuffle:
            np.random.default_rng(self.config.seed + epoch).shuffle(groups)
        size = max(int(self.config.sample_batch_size), 1)
        return [np.concatenate(groups[start : start + size]) for start in range(0, len(groups), size)]

    def _base_logit(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.clip(frame["source_score_calibrated"].to_numpy(dtype=float), 1e-5, 1.0 - 1e-5)
        return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)

    def fit(self, frame: pd.DataFrame) -> "NeuralRanker":
        if LABEL_COLUMN not in frame:
            raise ValueError("candidate labels are required for neural training")
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        x = torch.as_tensor(self.preprocessor.fit(frame).transform(frame), device=self.device)
        base = torch.as_tensor(self._base_logit(frame), device=self.device)
        labels = torch.as_tensor(frame[LABEL_COLUMN].to_numpy(dtype=np.float32), device=self.device)
        ids = frame["sample_id"].astype(str).tolist()
        self.model = self._make_model().to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        stale = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            total, batches = 0.0, 0
            for index in self._groups(frame, shuffle=True, epoch=epoch):
                tensor_index = torch.as_tensor(index, device=self.device)
                local_ids = [ids[int(value)] for value in index]
                optimizer.zero_grad()
                scores = self.model(x[tensor_index], base[tensor_index], local_ids)
                loss = self._objective(scores, labels[tensor_index], local_ids)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite neural loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            monitor = total / max(batches, 1)
            self.history.append({"epoch": float(epoch), "train_loss": monitor})
            if monitor < best_loss - 1e-7:
                best_loss, stale, self.best_epoch = monitor, 0, epoch
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        if best_state is None:
            raise RuntimeError("neural ranker did not produce a checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        self.model.eval()
        x = torch.as_tensor(self.preprocessor.transform(frame), device=self.device)
        base = torch.as_tensor(self._base_logit(frame), device=self.device)
        ids = frame["sample_id"].astype(str).tolist()
        output = np.empty(len(frame), dtype=float)
        with torch.no_grad():
            for index in self._groups(frame, shuffle=False):
                tensor_index = torch.as_tensor(index, device=self.device)
                local_ids = [ids[int(value)] for value in index]
                output[index] = self.model(x[tensor_index], base[tensor_index], local_ids).cpu().numpy()
        return output

    def artifact(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        return {
            "method": self.method_name,
            "controlled_encoder": self.method_name.startswith("residual_mlp"),
            "loss": {
                "residual_mlp_bce": "pointwise_bce",
                "residual_linear_listwise": "multi_positive_listwise",
                "residual_mlp_ranknet": "pairwise_ranknet",
                "residual_mlp_listwise": "multi_positive_listwise",
                "deepsets": "multi_positive_listwise",
                "set_transformer": "multi_positive_listwise",
                "candidate_gnn": "multi_positive_listwise",
            }[self.method_name],
            "config": asdict(self.config),
            "preprocessor": self.preprocessor.artifact(),
            "best_epoch": self.best_epoch,
            "history": self.history,
            "parameter_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
            "resolved_device": str(self.device),
        }

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("ranker is not fitted")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"artifact": self.artifact(), "state_dict": self.model.state_dict()}, destination)


def make_ranker(method: str, feature_columns: Sequence[str], *, seed: int, device: str = "auto") -> LogisticPointwiseRanker | NeuralRanker:
    if method == "logistic_pointwise":
        return LogisticPointwiseRanker(feature_columns, seed=seed)
    return NeuralRanker(method, feature_columns, config=NeuralConfig(seed=seed, device=device))


def scene_grouped_oof(
    frame: pd.DataFrame,
    factory: Callable[[], LogisticPointwiseRanker | NeuralRanker],
    *,
    folds: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if frame["scene_id"].nunique() < folds:
        raise ValueError("insufficient scenes for grouped OOF")
    splitter = GroupKFold(n_splits=int(folds))
    parts: list[pd.DataFrame] = []
    artifacts: list[dict[str, Any]] = []
    for fold, (train_index, held_index) in enumerate(splitter.split(frame, groups=frame["scene_id"].astype(str))):
        train, held = frame.iloc[train_index].copy(), frame.iloc[held_index].copy()
        if set(train["scene_id"]) & set(held["scene_id"]):
            raise AssertionError("scene leakage in ranker OOF")
        ranker = factory().fit(train)
        scored = attach_scores(held, ranker.predict_scores(held), method=ranker.method_name)
        scored["oof_fold"] = fold
        parts.append(scored)
        artifacts.append(
            {
                "fold": fold,
                "train_scene_count": int(train["scene_id"].nunique()),
                "held_scene_count": int(held["scene_id"].nunique()),
                "model": ranker.artifact(),
            }
        )
    output = pd.concat(parts, ignore_index=True)
    if len(output) != len(frame) or output["candidate_identity_sha256"].nunique() != len(frame):
        raise AssertionError("OOF must score every frozen candidate exactly once")
    return output, artifacts


def feature_columns_for_groups(groups: Sequence[str]) -> tuple[str, ...]:
    columns: list[str] = []
    for group in groups:
        columns.extend(CONTINUOUS_FEATURE_GROUPS[str(group)])
    return tuple(dict.fromkeys(columns))
