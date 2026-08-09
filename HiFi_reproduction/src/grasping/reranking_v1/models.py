"""Leakage-safe learned rankers for a frozen candidate pool.

This module deliberately has no dataset-specific IO.  It consumes the
``per_candidate.parquet`` schema produced by :mod:`features`, validates every
model input against the inference allowlist, and returns scores attached to the
same ``sample_id``/``candidate_id``/identity tuple.
"""

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

# Let PyTorch move an unsupported MPS operator to CPU instead of aborting a
# multi-hour experiment.  This must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

from src.grasping.reranking_v1.features import validate_inference_allowlist


LABEL_COLUMN = "candidate_positive"
IDENTITY_COLUMNS = (
    "sample_id",
    "scene_id",
    "candidate_id",
    "candidate_identity_sha256",
)
DEFAULT_FEATURE_COLUMNS = (
    "q_rank_normalized",
    "q_gap_to_top1",
    "p_axis_mean",
    "p_contact_min",
    "p_contact_imbalance",
    "grasp_axis_mask_support",
    "width_ratio_to_max_gripper",
    "normalized_width_mismatch",
    "jaw_depth_difference",
    "normal_opposition",
    "contact_symmetry",
    "left_finger_occupancy",
    "right_finger_occupancy",
    "palm_occupancy",
    "approach_corridor_occupancy",
    "approach_clearance",
    "collision_proxy_total",
    "candidate_uniqueness",
    "cluster_q_mean",
    "cluster_q_std",
)
VALID_DEV_SPLITS = {"train", "development"}
VALID_INFERENCE_SPLITS = {"train", "development", "val", "validation", "test"}


def resolve_torch_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic training device without assuming CUDA."""

    value = str(requested).lower()
    if value not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be one of auto, mps, cpu")
    if value == "cpu":
        return torch.device("cpu")
    available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if value == "mps" and not available:
        raise RuntimeError("MPS was requested but is not available")
    return torch.device("mps" if available else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _require_frame(
    frame: pd.DataFrame,
    *,
    features: Sequence[str] = (),
    require_label: bool = False,
) -> None:
    selected = validate_inference_allowlist(features)
    required = {"sample_id", "scene_id", "candidate_id", *selected}
    if require_label:
        required.add(LABEL_COLUMN)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate frame missing required columns: {missing}")
    if frame.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError("candidate frame has duplicate sample_id+candidate_id")
    if selected:
        matrix = frame.loc[:, selected].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("candidate frame contains non-finite inference features")
    if require_label:
        labels = frame[LABEL_COLUMN].to_numpy()
        if not set(np.unique(labels)).issubset({False, True, 0, 1}):
            raise ValueError("candidate_positive must be binary")


def validate_candidate_contract(
    frame: pd.DataFrame,
    *,
    require_label: bool = False,
    allowed_splits: set[str] | None = None,
) -> None:
    """Validate rank, q-value, identity, and split invariants before scoring."""

    _require_frame(frame, require_label=require_label)
    required = {
        "candidate_identity_sha256",
        "original_gqcnn_rank",
        "q_raw",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate contract missing columns: {missing}")
    if allowed_splits is not None:
        splits = set(frame["split"].astype(str).unique())
        if not splits or not splits.issubset(set(allowed_splits)):
            raise ValueError(f"candidate frame has forbidden splits: {sorted(splits)}")
    hashes = frame["candidate_identity_sha256"].astype(str)
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("candidate identity hashes must be lowercase SHA-256")
    if hashes.duplicated().any():
        raise ValueError("candidate identity hash is not globally unique")
    q = pd.to_numeric(frame["q_raw"], errors="coerce").to_numpy(dtype=np.float64)
    ranks_numeric = pd.to_numeric(
        frame["original_gqcnn_rank"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(q))
        or not np.all(np.isfinite(ranks_numeric))
        or not np.all(ranks_numeric == np.floor(ranks_numeric))
        or np.any(ranks_numeric < 1)
    ):
        raise ValueError("q values and original ranks must be finite valid values")
    for sample_id, group in frame.groupby("sample_id", sort=False):
        ranks = group["original_gqcnn_rank"].to_numpy(dtype=np.int64)
        if sorted(ranks.tolist()) != list(range(1, len(group) + 1)):
            raise ValueError(f"non-contiguous or duplicate GQ-CNN ranks: {sample_id}")
        expected = group.sort_values(
            ["q_raw", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        )["candidate_id"].astype(str).tolist()
        actual = group.sort_values(
            ["original_gqcnn_rank", "candidate_id"],
            ascending=[True, True],
            kind="mergesort",
        )["candidate_id"].astype(str).tolist()
        if actual != expected:
            raise ValueError(
                f"GQ-CNN rank disagrees with full-precision q ordering: {sample_id}"
            )


def assert_feature_identity_invariant(
    before: pd.DataFrame, after: pd.DataFrame
) -> None:
    """Assert that scoring/ranking is a pure permutation of candidate rows."""

    columns = ["sample_id", "candidate_id"]
    if "candidate_identity_sha256" in before.columns:
        if "candidate_identity_sha256" not in after.columns:
            raise AssertionError("candidate identity hash was dropped")
        columns.append("candidate_identity_sha256")
    left = before.loc[:, columns].astype(str).sort_values(columns).reset_index(drop=True)
    right = after.loc[:, columns].astype(str).sort_values(columns).reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("candidate identity changed during re-ranking")


def attach_scores(
    frame: pd.DataFrame, scores: Sequence[float], *, method: str
) -> pd.DataFrame:
    """Attach finite scores and deterministic within-sample ranks."""

    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(frame),) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be one finite value per candidate")
    result = frame.copy()
    if "reranker_rank" in result.columns:
        result = result.drop(columns=["reranker_rank"])
    result["reranker_method"] = str(method)
    result["reranker_score"] = values
    # Stable ID tie-break is part of the frozen evaluation contract.
    ordered = result.sort_values(
        ["sample_id", "reranker_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ordered["reranker_rank"] = (
        ordered.groupby("sample_id", sort=False).cumcount() + 1
    )
    result = result.join(ordered["reranker_rank"])
    assert_feature_identity_invariant(frame, result)
    return result


def q_only_scores(frame: pd.DataFrame) -> np.ndarray:
    _require_frame(frame, features=("q_raw",))
    return frame["q_raw"].to_numpy(dtype=np.float64)


def q_softmask_rule_scores(
    frame: pd.DataFrame,
    *,
    alpha: float = 0.8,
    beta: float = 0.2,
    q_column: str = "q_rank_normalized",
    mask_column: str = "p_axis_mean",
) -> np.ndarray:
    """A1 score using two already-normalized, deployable quantities."""

    if alpha < 0.0 or beta < 0.0 or alpha + beta <= 0.0:
        raise ValueError("alpha and beta must be non-negative with positive sum")
    _require_frame(frame, features=(q_column, mask_column))
    scale = alpha + beta
    return (
        alpha * frame[q_column].to_numpy(dtype=np.float64)
        + beta * frame[mask_column].to_numpy(dtype=np.float64)
    ) / scale


@dataclass(frozen=True)
class GeometryGateConfig:
    max_width_ratio: float = 1.05
    max_normalized_width_mismatch: float = 0.75
    max_jaw_depth_difference_m: float = 0.035
    max_finger_occupancy: float = 0.45
    max_palm_occupancy: float = 0.35
    max_approach_occupancy: float = 0.45
    # ``approach_clearance`` is a dimensionless 0..1 corridor-clearance score
    # (1 means the observed approach corridor is clear), not a metric distance.
    min_approach_clearance_score: float = 0.003
    min_normal_opposition: float = -0.25
    min_contact_symmetry: float = 0.15
    risk_penalty: float = 2.0


GEOMETRY_GATE_COLUMNS = (
    "width_ratio_to_max_gripper",
    "normalized_width_mismatch",
    "jaw_depth_difference",
    "left_finger_occupancy",
    "right_finger_occupancy",
    "palm_occupancy",
    "approach_corridor_occupancy",
    "approach_clearance",
    "normal_opposition",
    "contact_symmetry",
)


def geometry_risk(
    frame: pd.DataFrame, config: GeometryGateConfig = GeometryGateConfig()
) -> tuple[np.ndarray, np.ndarray]:
    """Return a continuous visible-geometry risk and a severe-risk veto."""

    _require_frame(frame, features=GEOMETRY_GATE_COLUMNS)
    values = frame
    components = np.column_stack(
        [
            np.clip(
                values["width_ratio_to_max_gripper"].to_numpy(float)
                / config.max_width_ratio,
                0.0,
                2.0,
            ),
            np.clip(
                values["normalized_width_mismatch"].to_numpy(float)
                / config.max_normalized_width_mismatch,
                0.0,
                2.0,
            ),
            np.clip(
                np.abs(values["jaw_depth_difference"].to_numpy(float))
                / config.max_jaw_depth_difference_m,
                0.0,
                2.0,
            ),
            np.clip(
                np.maximum(
                    values["left_finger_occupancy"].to_numpy(float),
                    values["right_finger_occupancy"].to_numpy(float),
                )
                / config.max_finger_occupancy,
                0.0,
                2.0,
            ),
            np.clip(
                values["palm_occupancy"].to_numpy(float)
                / config.max_palm_occupancy,
                0.0,
                2.0,
            ),
            np.clip(
                values["approach_corridor_occupancy"].to_numpy(float)
                / config.max_approach_occupancy,
                0.0,
                2.0,
            ),
            np.clip(
                config.min_approach_clearance_score
                / np.maximum(
                    values["approach_clearance"].to_numpy(float), 1e-6
                ),
                0.0,
                2.0,
            ),
            np.clip(
                (config.min_normal_opposition
                - values["normal_opposition"].to_numpy(float))
                + 1.0,
                0.0,
                2.0,
            ),
            np.clip(
                (config.min_contact_symmetry
                - values["contact_symmetry"].to_numpy(float))
                + 1.0,
                0.0,
                2.0,
            ),
        ]
    )
    # 1.0 denotes a threshold boundary; the excess is a penalty signal.
    risk = np.mean(np.clip(components - 1.0, 0.0, None), axis=1)
    unsafe = np.any(components > 1.0, axis=1)
    return risk.astype(np.float64), unsafe


def geometry_gated_q_scores(
    frame: pd.DataFrame,
    config: GeometryGateConfig = GeometryGateConfig(),
) -> np.ndarray:
    _require_frame(frame, features=("q_rank_normalized", *GEOMETRY_GATE_COLUMNS))
    risk, unsafe = geometry_risk(frame, config)
    q = frame["q_rank_normalized"].to_numpy(dtype=np.float64)
    # A finite penalty preserves deterministic serialization and makes vetoed
    # candidates inspectable.  Safe candidates always outrank an equally-scored
    # unsafe candidate.
    return q - config.risk_penalty * (risk + unsafe.astype(np.float64))


@dataclass
class TrainOnlyScaler:
    feature_columns: tuple[str, ...]
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    source_splits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.feature_columns = validate_inference_allowlist(self.feature_columns)

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyScaler":
        _require_frame(frame, features=self.feature_columns)
        if "split" not in frame.columns:
            raise ValueError("train-only scaler requires an explicit split column")
        splits = tuple(sorted(map(str, frame["split"].unique())))
        if not splits or not set(splits).issubset({"train", "development"}):
            raise ValueError(
                f"scaler may only fit train/development, received {splits}"
            )
        matrix = frame.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        self.mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        self.scale = np.where(std > 1e-12, std, 1.0)
        self.source_splits = splits
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("scaler has not been fitted")
        _require_frame(frame, features=self.feature_columns)
        matrix = frame.loc[:, self.feature_columns].to_numpy(dtype=np.float64)
        transformed = (matrix - self.mean) / self.scale
        if not np.all(np.isfinite(transformed)):
            raise ValueError("scaled features are not finite")
        return transformed.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.scale is None:
            raise RuntimeError("scaler has not been fitted")
        return {
            "feature_columns": list(self.feature_columns),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "source_splits": list(self.source_splits),
            "fit_scope": "train/development candidates only",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainOnlyScaler":
        scaler = cls(tuple(map(str, payload["feature_columns"])))
        scaler.mean = np.asarray(payload["mean"], dtype=np.float64)
        scaler.scale = np.asarray(payload["scale"], dtype=np.float64)
        scaler.source_splits = tuple(map(str, payload["source_splits"]))
        if (
            scaler.mean.shape != (len(scaler.feature_columns),)
            or scaler.scale.shape != scaler.mean.shape
            or not np.all(np.isfinite(scaler.mean))
            or not np.all(np.isfinite(scaler.scale))
            or np.any(scaler.scale <= 0.0)
        ):
            raise ValueError("invalid serialized train-only scaler")
        if not set(scaler.source_splits).issubset(VALID_DEV_SPLITS):
            raise ValueError("serialized scaler was not fitted on development")
        return scaler


class RegularizedLinearRanker:
    """Interpretable sample-balanced logistic candidate ranker."""

    method_name = "regularized_linear_ranker"

    def __init__(
        self,
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
        *,
        c: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.feature_columns = validate_inference_allowlist(feature_columns)
        self.c = float(c)
        self.seed = int(seed)
        self.scaler = TrainOnlyScaler(self.feature_columns)
        self.model = LogisticRegression(
            C=self.c,
            solver="liblinear",
            max_iter=1000,
            random_state=self.seed,
        )

    def fit(
        self, frame: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "RegularizedLinearRanker":
        del validation
        _require_frame(frame, features=self.feature_columns, require_label=True)
        x = self.scaler.fit(frame).transform(frame)
        y = frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError("linear ranker needs both positive and negative labels")
        counts = frame.groupby("sample_id")["candidate_id"].transform("size")
        sample_weight = 1.0 / counts.to_numpy(dtype=np.float64)
        self.model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.scaler.transform(frame)
        return self.model.decision_function(x).astype(np.float64)

    def coefficients(self) -> pd.DataFrame:
        coefficient = self.model.coef_.reshape(-1)
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "coefficient_standardized": coefficient,
                "sign": np.where(coefficient >= 0.0, "positive", "negative"),
                "absolute_standardized_magnitude": np.abs(coefficient),
            }
        ).sort_values("absolute_standardized_magnitude", ascending=False)

    def artifact(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "hyperparameters": {"C": self.c, "penalty": "l2", "seed": self.seed},
            "scaler": self.scaler.to_dict(),
            "intercept": float(self.model.intercept_[0]),
            "coefficients": self.coefficients().to_dict(orient="records"),
        }

    @classmethod
    def from_artifact(
        cls, payload: Mapping[str, Any]
    ) -> "RegularizedLinearRanker":
        hyperparameters = payload["hyperparameters"]
        model = cls(
            tuple(map(str, payload["scaler"]["feature_columns"])),
            c=float(hyperparameters["C"]),
            seed=int(hyperparameters["seed"]),
        )
        model.scaler = TrainOnlyScaler.from_dict(payload["scaler"])
        coefficient_rows = list(payload["coefficients"])
        coefficient_by_feature = {
            str(row["feature"]): float(row["coefficient_standardized"])
            for row in coefficient_rows
        }
        if set(coefficient_by_feature) != set(model.feature_columns):
            raise ValueError("serialized linear coefficients are incomplete")
        model.model.coef_ = np.asarray(
            [[coefficient_by_feature[name] for name in model.feature_columns]],
            dtype=np.float64,
        )
        model.model.intercept_ = np.asarray(
            [float(payload["intercept"])], dtype=np.float64
        )
        model.model.classes_ = np.asarray([0, 1], dtype=np.int64)
        model.model.n_features_in_ = len(model.feature_columns)
        return model

    def save(self, path: str | Path) -> None:
        save_json(path, self.artifact())

    @classmethod
    def load(cls, path: str | Path) -> "RegularizedLinearRanker":
        return cls.from_artifact(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


def sample_balanced_pairwise_ranknet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: Sequence[str],
    *,
    hardness: torch.Tensor | None = None,
    max_negatives_per_sample: int | None = None,
) -> torch.Tensor:
    """RankNet loss where every eligible sample receives exactly equal weight."""

    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be matching vectors")
    if len(sample_ids) != len(scores):
        raise ValueError("sample_ids length does not match scores")
    if hardness is not None and hardness.shape != scores.shape:
        raise ValueError("hardness must match scores")
    losses: list[torch.Tensor] = []
    ids = np.asarray(list(map(str, sample_ids)), dtype=object)
    for sample_id in dict.fromkeys(ids.tolist()):
        index = np.flatnonzero(ids == sample_id)
        tensor_index = torch.as_tensor(
            index, dtype=torch.long, device=labels.device
        )
        local_labels = labels[tensor_index]
        positive = index[(local_labels > 0.5).detach().cpu().numpy()]
        negative = index[(local_labels <= 0.5).detach().cpu().numpy()]
        if not len(positive) or not len(negative):
            continue
        if (
            max_negatives_per_sample is not None
            and len(negative) > max_negatives_per_sample
        ):
            if max_negatives_per_sample <= 0:
                raise ValueError("max_negatives_per_sample must be positive")
            priority = (
                hardness[
                    torch.as_tensor(
                        negative, dtype=torch.long, device=hardness.device
                    )
                ]
                if hardness is not None
                else scores[
                    torch.as_tensor(
                        negative, dtype=torch.long, device=scores.device
                    )
                ].detach()
            )
            chosen = torch.topk(
                priority, k=int(max_negatives_per_sample), largest=True
            ).indices.detach().cpu().numpy()
            negative = negative[chosen]
        negative_index = torch.as_tensor(
            negative, dtype=torch.long, device=scores.device
        )
        positive_index = torch.as_tensor(
            positive, dtype=torch.long, device=scores.device
        )
        differences = (
            scores[negative_index][None, :]
            - scores[positive_index][:, None]
        )
        losses.append(F.softplus(differences).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def multi_positive_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: Sequence[str],
) -> torch.Tensor:
    """Negative log probability mass assigned to any positive candidate."""

    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must be matching vectors")
    ids = np.asarray(list(map(str, sample_ids)), dtype=object)
    losses: list[torch.Tensor] = []
    for sample_id in dict.fromkeys(ids.tolist()):
        index = np.flatnonzero(ids == sample_id)
        tensor_index = torch.as_tensor(
            index, dtype=torch.long, device=scores.device
        )
        local_scores = scores[tensor_index]
        local_labels = labels[tensor_index] > 0.5
        if not bool(local_labels.any()) or bool(local_labels.all()):
            continue
        losses.append(
            torch.logsumexp(local_scores, dim=0)
            - torch.logsumexp(local_scores[local_labels], dim=0)
        )
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class NeuralRankerConfig:
    hidden_dim: int = 32
    embedding_dim: int = 16
    dropout: float = 0.15
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    epochs: int = 40
    patience: int = 6
    residual_bound: float = 0.25
    q_alpha: float = 1.0
    sample_batch_size: int = 128
    hard_negative_limit: int = 32
    device: str = "auto"
    seed: int = 42


class _LinearScore(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1)

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, sample_ids: Sequence[str]
    ) -> torch.Tensor:
        del q, sample_ids
        return self.linear(x).squeeze(-1)


class _ResidualMLPScore(nn.Module):
    def __init__(self, feature_count: int, config: NeuralRankerConfig) -> None:
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(feature_count, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, sample_ids: Sequence[str]
    ) -> torch.Tensor:
        del sample_ids
        residual = self.config.residual_bound * torch.tanh(
            self.mlp(x).squeeze(-1)
        )
        return self.config.q_alpha * q + residual


class _DeepSetsResidualScore(nn.Module):
    def __init__(self, feature_count: int, config: NeuralRankerConfig) -> None:
        super().__init__()
        self.config = config
        self.phi = nn.Sequential(
            nn.Linear(feature_count, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, sample_ids: Sequence[str]
    ) -> torch.Tensor:
        embedding = self.phi(x)
        ids = np.asarray(list(map(str, sample_ids)), dtype=object)
        context = torch.empty(
            (len(ids), embedding.shape[1] * 2),
            dtype=embedding.dtype,
            device=embedding.device,
        )
        for sample_id in dict.fromkeys(ids.tolist()):
            index_np = np.flatnonzero(ids == sample_id)
            index = torch.as_tensor(index_np, dtype=torch.long, device=x.device)
            group = embedding[index]
            mean = group.mean(dim=0)
            maximum = group.max(dim=0).values
            context[index] = torch.cat([mean, maximum]).expand(len(index_np), -1)
        residual = self.config.residual_bound * torch.tanh(
            self.head(torch.cat([embedding, context], dim=1)).squeeze(-1)
        )
        return self.config.q_alpha * q + residual


class TorchCandidateRanker:
    """Shared trainer for pairwise, listwise, residual, and set-aware models."""

    def __init__(
        self,
        method_name: str,
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
        *,
        config: NeuralRankerConfig = NeuralRankerConfig(),
    ) -> None:
        valid = {
            "pairwise_ranker",
            "multi_positive_listwise_ranker",
            "residual_mlp",
            "set_aware_residual",
        }
        if method_name not in valid:
            raise ValueError(f"unknown neural ranker: {method_name}")
        self.method_name = method_name
        self.feature_columns = validate_inference_allowlist(feature_columns)
        self.config = config
        self.scaler = TrainOnlyScaler(self.feature_columns)
        self.model: nn.Module | None = None
        self.history: list[dict[str, float]] = []
        self.best_epoch = -1
        self.device = resolve_torch_device(config.device)

    def _make_model(self) -> nn.Module:
        if self.method_name in {
            "pairwise_ranker",
            "multi_positive_listwise_ranker",
        }:
            return _LinearScore(len(self.feature_columns))
        if self.method_name == "residual_mlp":
            return _ResidualMLPScore(len(self.feature_columns), self.config)
        return _DeepSetsResidualScore(len(self.feature_columns), self.config)

    def _tensors(
        self, frame: pd.DataFrame, *, labels: bool
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        list[str],
    ]:
        x = torch.as_tensor(
            self.scaler.transform(frame),
            dtype=torch.float32,
            device=self.device,
        )
        q_values = (
            frame["q_rank_normalized"].to_numpy(dtype=np.float32)
            if "q_rank_normalized" in frame.columns
            else np.zeros(len(frame), dtype=np.float32)
        )
        q = torch.as_tensor(q_values, dtype=torch.float32, device=self.device)
        y = (
            torch.as_tensor(
                frame[LABEL_COLUMN].to_numpy(dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            if labels
            else None
        )
        # Mining priority intentionally uses deployable signals only.  It
        # prioritizes high-q negatives and the two difficult disagreement modes:
        # high target support with geometric risk, and low target support with
        # otherwise plausible visible geometry.
        def column(name: str) -> np.ndarray:
            if name not in frame.columns:
                return np.zeros(len(frame), dtype=np.float32)
            return frame[name].to_numpy(dtype=np.float32)

        mask = np.clip(column("p_axis_mean"), 0.0, 1.0)
        geometry = np.clip(
            column("collision_proxy_total")
            + column("normalized_width_mismatch"),
            0.0,
            2.0,
        )
        hardness = torch.as_tensor(
            q_values + mask * geometry + (1.0 - mask) * (1.0 - geometry / 2.0),
            dtype=torch.float32,
            device=self.device,
        )
        return x, q, y, hardness, frame["sample_id"].astype(str).tolist()

    def _objective(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        ids: Sequence[str],
        hardness: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.method_name == "pairwise_ranker":
            return sample_balanced_pairwise_ranknet_loss(
                scores,
                labels,
                ids,
                hardness=hardness,
                max_negatives_per_sample=self.config.hard_negative_limit,
            )
        return multi_positive_listwise_loss(scores, labels, ids)

    def _sample_batches(
        self, frame: pd.DataFrame, *, shuffle: bool, epoch: int = 0
    ) -> list[np.ndarray]:
        groups = [
            np.asarray(index, dtype=np.int64)
            for index in frame.groupby("sample_id", sort=False).indices.values()
        ]
        if shuffle:
            rng = np.random.default_rng(self.config.seed + int(epoch))
            rng.shuffle(groups)
        batch_size = max(int(self.config.sample_batch_size), 1)
        return [
            np.concatenate(groups[start : start + batch_size])
            for start in range(0, len(groups), batch_size)
        ]

    def _loss_on(self, frame: pd.DataFrame) -> float:
        assert self.model is not None
        self.model.eval()
        weighted_loss = 0.0
        sample_count = 0
        x, q, labels, hardness, ids = self._tensors(frame, labels=True)
        assert labels is not None
        with torch.no_grad():
            for index in self._sample_batches(frame, shuffle=False):
                tensor_index = torch.as_tensor(
                    index, dtype=torch.long, device=self.device
                )
                local_ids = [ids[int(i)] for i in index]
                count = len(dict.fromkeys(local_ids))
                local_scores = self.model(
                    x[tensor_index], q[tensor_index], local_ids
                )
                local_loss = self._objective(
                    local_scores,
                    labels[tensor_index],
                    local_ids,
                    hardness[tensor_index],
                )
                weighted_loss += float(local_loss) * count
                sample_count += count
        return weighted_loss / max(sample_count, 1)

    def fit(
        self, frame: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "TorchCandidateRanker":
        try:
            return self._fit_impl(frame, validation)
        except RuntimeError as error:
            lowered = str(error).lower()
            if self.device.type != "mps" or not any(
                token in lowered
                for token in (
                    "mps",
                    "not implemented",
                    "unsupported",
                    "placeholder",
                    "out of memory",
                )
            ):
                raise
            self.device = torch.device("cpu")
            self.model = None
            self.history = []
            self.best_epoch = -1
            return self._fit_impl(frame, validation)

    def _fit_impl(
        self, frame: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "TorchCandidateRanker":
        required = tuple(
            dict.fromkeys((*self.feature_columns, "q_rank_normalized"))
        )
        _require_frame(frame, features=required, require_label=True)
        if validation is not None:
            _require_frame(validation, features=required, require_label=True)
        _seed_everything(self.config.seed)
        self.scaler.fit(frame)
        self.model = self._make_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        x, q, labels, hardness, ids = self._tensors(frame, labels=True)
        assert labels is not None
        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        stale = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_samples = 0
            for index in self._sample_batches(frame, shuffle=True, epoch=epoch):
                tensor_index = torch.as_tensor(
                    index, dtype=torch.long, device=self.device
                )
                local_ids = [ids[int(i)] for i in index]
                count = len(dict.fromkeys(local_ids))
                optimizer.zero_grad()
                scores = self.model(
                    x[tensor_index], q[tensor_index], local_ids
                )
                loss = self._objective(
                    scores,
                    labels[tensor_index],
                    local_ids,
                    hardness[tensor_index],
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite neural ranker loss")
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * count
                epoch_samples += count
            monitor = (
                self._loss_on(validation)
                if validation is not None
                else self._loss_on(frame)
            )
            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": epoch_loss / max(epoch_samples, 1),
                    "monitor_loss": monitor,
                }
            )
            if monitor < best_loss - 1e-7:
                best_loss = monitor
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stale = 0
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
            raise RuntimeError("ranker has not been fitted")
        required = tuple(
            dict.fromkeys((*self.feature_columns, "q_rank_normalized"))
        )
        _require_frame(frame, features=required)
        self.model.eval()
        result = np.empty(len(frame), dtype=np.float64)
        x, q, _, _, ids = self._tensors(frame, labels=False)
        with torch.no_grad():
            for index in self._sample_batches(frame, shuffle=False):
                tensor_index = torch.as_tensor(
                    index, dtype=torch.long, device=self.device
                )
                local_ids = [ids[int(i)] for i in index]
                result[index] = (
                    self.model(
                        x[tensor_index], q[tensor_index], local_ids
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("ranker emitted non-finite scores")
        return result

    def residuals(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.predict_scores(frame)
        if self.method_name not in {"residual_mlp", "set_aware_residual"}:
            return values
        q = frame["q_rank_normalized"].to_numpy(dtype=np.float64)
        return values - self.config.q_alpha * q

    def artifact(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("ranker has not been fitted")
        return {
            "method": self.method_name,
            "feature_columns": list(self.feature_columns),
            "hyperparameters": asdict(self.config),
            "scaler": self.scaler.to_dict(),
            "best_epoch": self.best_epoch,
            "training_history": self.history,
            "residual_bound": (
                self.config.residual_bound
                if self.method_name in {"residual_mlp", "set_aware_residual"}
                else None
            ),
            "resolved_device": str(self.device),
        }

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("ranker has not been fitted")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "artifact": self.artifact(),
                "state_dict": self.model.state_dict(),
            },
            destination,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, device: str = "auto"
    ) -> "TorchCandidateRanker":
        destination = Path(path)
        resolved = resolve_torch_device(device)
        payload = torch.load(
            destination,
            map_location=resolved,
            weights_only=False,
        )
        artifact = payload["artifact"]
        config_payload = dict(artifact["hyperparameters"])
        config_payload["device"] = device
        model = cls(
            str(artifact["method"]),
            tuple(map(str, artifact["feature_columns"])),
            config=NeuralRankerConfig(**config_payload),
        )
        model.scaler = TrainOnlyScaler.from_dict(artifact["scaler"])
        model.model = model._make_model().to(model.device)
        model.model.load_state_dict(payload["state_dict"], strict=True)
        model.model.eval()
        model.best_epoch = int(artifact["best_epoch"])
        model.history = [
            {str(key): float(value) for key, value in row.items()}
            for row in artifact["training_history"]
        ]
        return model


def make_ranker(
    method: str,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    *,
    config: NeuralRankerConfig = NeuralRankerConfig(),
) -> RegularizedLinearRanker | TorchCandidateRanker:
    if method == "regularized_linear_ranker":
        return RegularizedLinearRanker(
            feature_columns, seed=config.seed
        )
    return TorchCandidateRanker(method, feature_columns, config=config)


def scene_grouped_oof(
    frame: pd.DataFrame,
    model_factory: Callable[
        [], RegularizedLinearRanker | TorchCandidateRanker
    ],
    *,
    n_splits: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate strictly scene-held-out scores for every development candidate."""

    _require_frame(frame, require_label=True)
    scene_count = frame["scene_id"].nunique()
    if n_splits < 2 or scene_count < n_splits:
        raise ValueError("scene-grouped OOF requires at least n_splits unique scenes")
    splitter = GroupKFold(n_splits=n_splits)
    output_parts: list[pd.DataFrame] = []
    fold_artifacts: list[dict[str, Any]] = []
    for fold, (train_index, held_index) in enumerate(
        splitter.split(frame, groups=frame["scene_id"].astype(str))
    ):
        train = frame.iloc[train_index].copy()
        held = frame.iloc[held_index].copy()
        overlap = set(train["scene_id"].astype(str)) & set(
            held["scene_id"].astype(str)
        )
        if overlap:
            raise AssertionError(f"scene leakage in OOF fold {fold}: {overlap}")
        model = model_factory()
        # The held-out fold is never passed for early stopping; its labels are
        # consumed only after predictions have been frozen.
        model.fit(train)
        scored = attach_scores(
            held, model.predict_scores(held), method=getattr(model, "method_name")
        )
        scored["oof_fold"] = fold
        output_parts.append(scored)
        fold_artifacts.append(
            {
                "fold": fold,
                "train_scenes": sorted(train["scene_id"].astype(str).unique()),
                "held_out_scenes": sorted(held["scene_id"].astype(str).unique()),
                "model": model.artifact(),
            }
        )
    output = pd.concat(output_parts, ignore_index=True)
    assert_feature_identity_invariant(frame, output)
    if len(output) != len(frame):
        raise AssertionError("OOF did not produce exactly one score per candidate")
    return output, fold_artifacts


def top1_accuracy(scored: pd.DataFrame) -> dict[str, float | int]:
    _require_frame(scored, require_label=True)
    if "reranker_rank" not in scored.columns:
        raise ValueError("scored frame has no reranker_rank")
    top = scored.loc[scored["reranker_rank"] == 1]
    correct = int(top[LABEL_COLUMN].astype(bool).sum())
    total = int(top["sample_id"].nunique())
    return {
        "correct": correct,
        "total": total,
        "accuracy": float(correct / total) if total else 0.0,
    }


def save_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
