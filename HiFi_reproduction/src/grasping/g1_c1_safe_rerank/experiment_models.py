"""Serializable model adapters for the complete G1/C1 experiment matrix."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from reranking.losses import CompositeRerankingLoss, LossBreakdown, QueryComposition
from reranking.models.neural import SharedMLPScorer
from reranking.models.set_models import (
    CandidateGNNScorer,
    DeepSetsScorer,
    PAIRWISE_EDGE_RELATION_FIELDS,
    SetTransformerScorer,
)
from reranking.models.tabular import LinearPairwiseRankNet, LinearResidualBCE
from reranking.train import (
    CandidateSetExample,
    LENGTH_BUCKETED_BATCHING_POLICY,
    TrainingConfig,
    _forward_candidate_batch,
    fit_neural_ranker,
    make_candidate_dataloader,
)


class _LinearResidualScorer(torch.nn.Module):
    def __init__(self, input_dim: int, residual_scale: float = 0.5) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.residual_scale = float(residual_scale)
        self.head = torch.nn.Linear(input_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        baseline_scores: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if baseline_scores is None:
            raise ValueError("baseline_scores are required")
        scores = baseline_scores + self.residual_scale * torch.tanh(
            self.head(features).squeeze(-1)
        )
        if padding_mask is not None:
            scores = torch.where(padding_mask, baseline_scores, scores)
        return scores


class _HardTopPairLoss(CompositeRerankingLoss):
    """Secondary RankNet variant emphasizing source/current hard negatives."""

    def __init__(self) -> None:
        super().__init__(ranknet_weight=1.0)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "variant": "hard_top_pair_ranknet",
            "original_wrong_top1_weight": 2.0,
            "current_highest_negative_weight": 2.0,
            "other_pair_weight": 1.0,
            "high_iou_geometric_negative": "not_available_in_padded_loss_batch",
        }

    def active_breakdown(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        baseline_scores: torch.Tensor | None = None,
        composition: QueryComposition | None = None,
    ) -> LossBreakdown:
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
            labels = labels.unsqueeze(0)
            padding_mask = None if padding_mask is None else padding_mask.unsqueeze(0)
            baseline_scores = None if baseline_scores is None else baseline_scores.unsqueeze(0)
        padding = torch.zeros_like(scores, dtype=torch.bool) if padding_mask is None else padding_mask
        if baseline_scores is None:
            raise ValueError("hard-pair RankNet requires baseline scores")
        losses = []
        for query_scores, query_labels, query_padding, query_base in zip(
            scores, labels, padding, baseline_scores
        ):
            valid = ~query_padding
            positive = torch.where(valid & (query_labels == 1))[0]
            negative = torch.where(valid & (query_labels == 0))[0]
            if positive.numel() == 0 or negative.numel() == 0:
                continue
            pair = torch.nn.functional.softplus(
                -(query_scores[positive][:, None] - query_scores[negative][None, :])
            )
            weight = torch.ones_like(pair)
            source_hard = torch.argmax(query_base[negative])
            current_hard = torch.argmax(query_scores[negative])
            weight[:, source_hard] = 2.0
            weight[:, current_hard] = 2.0
            losses.append((pair * weight).sum() / weight.sum())
        loss = torch.stack(losses).mean() if losses else scores.sum() * 0.0
        zero = scores.sum() * 0.0
        comp = composition or QueryComposition(
            total=int(scores.shape[0]), empty=0, no_positive=0, all_positive=0, mixed=len(losses)
        )
        return LossBreakdown(
            total=loss,
            bce=zero,
            ranknet=loss,
            listwise=zero,
            residual=zero,
            composition=comp,
            active_queries=len(losses),
        )


def _finite_fill(frame: pd.DataFrame, columns: Sequence[str], median: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame.loc[:, list(columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    if median is None:
        with np.errstate(all="ignore"):
            median = np.nanmedian(matrix, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(matrix), matrix, median)
    if not np.isfinite(filled).all():
        raise ValueError("feature imputation produced non-finite values")
    return filled, np.asarray(median, dtype=float)


class TabularResidualModel:
    """Train-only imputation around audited generic linear rankers."""

    def __init__(self, kind: str, feature_columns: Sequence[str], *, seed: int) -> None:
        if kind not in {"bce", "ranknet"}:
            raise ValueError(kind)
        self.kind = str(kind)
        self.feature_columns = tuple(map(str, feature_columns))
        self.seed = int(seed)
        self.median: np.ndarray | None = None
        self.model: Any = None

    def fit(self, frame: pd.DataFrame) -> "TabularResidualModel":
        matrix, self.median = _finite_fill(frame, self.feature_columns)
        features = pd.DataFrame(matrix, columns=self.feature_columns)
        self.model = (
            LinearResidualBCE(random_state=self.seed)
            if self.kind == "bce"
            else LinearPairwiseRankNet(random_state=self.seed)
        )
        self.model.fit(
            features,
            frame["candidate_correct"].astype(int).to_numpy(),
            query_ids=frame["sample_id"].astype(str).tolist(),
            baseline_scores=frame["original_score"].to_numpy(dtype=float),
            sample_ids=frame["sample_id"].astype(str).tolist(),
        )
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.median is None or self.model is None:
            raise RuntimeError("model not fitted")
        matrix, _ = _finite_fill(frame, self.feature_columns, self.median)
        return self.model.predict_scores(
            pd.DataFrame(matrix, columns=self.feature_columns),
            query_ids=frame["sample_id"].astype(str).tolist(),
            baseline_scores=frame["original_score"].to_numpy(dtype=float),
        )

    def artifact(self) -> dict[str, Any]:
        return {
            "kind": f"linear_residual_{self.kind}",
            "seed": self.seed,
            "feature_columns": list(self.feature_columns),
            "median": None if self.median is None else self.median.tolist(),
            "model": self.model.metadata,
        }


class LightGBMLambdaMARTModel:
    """LightGBM lambdarank with explicit contiguous query groups."""

    def __init__(self, feature_columns: Sequence[str], *, seed: int) -> None:
        self.feature_columns = tuple(map(str, feature_columns))
        self.seed = int(seed)
        self.median: np.ndarray | None = None
        self.model: Any = None
        self.training_order: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "LightGBMLambdaMARTModel":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import lightgbm as lgb

        matrix, self.median = _finite_fill(frame, self.feature_columns)
        query = frame["sample_id"].astype(str).to_numpy()
        order = np.argsort(query, kind="stable")
        sorted_query = query[order]
        _, group = np.unique(sorted_query, return_counts=True)
        if int(group.sum()) != len(frame):
            raise AssertionError("LightGBM group sizes do not cover rows")
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=(1, 5),
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            min_child_samples=20,
            feature_fraction=0.8,
            random_state=self.seed,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(matrix[order], frame["candidate_correct"].astype(int).to_numpy()[order], group=group.tolist())
        self.training_order = order
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.median is None or self.model is None:
            raise RuntimeError("model not fitted")
        matrix, _ = _finite_fill(frame, self.feature_columns, self.median)
        scores = np.asarray(self.model.predict(matrix), dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError("LightGBM produced non-finite scores")
        return scores

    def artifact(self) -> dict[str, Any]:
        import lightgbm as lgb

        return {
            "kind": "lightgbm_lambdarank",
            "version": lgb.__version__,
            "objective": "lambdarank",
            "metric": ["ndcg@1", "ndcg@5"],
            "seed": self.seed,
            "feature_columns": list(self.feature_columns),
            "parameters": self.model.get_params(),
            "single_threaded_execution": True,
        }


class GenericNeuralModel:
    """Vectorized padded-query trainer for controlled MLP/set/GNN models."""

    KINDS = {"linear_listwise", "mlp_bce", "mlp_ranknet", "mlp_ranknet_hard_top", "mlp_listwise", "mlp_listwise_t05", "mlp_listwise_t20", "mlp_listwise_hybrid", "deepsets", "set_transformer", "gnn"}

    def __init__(
        self,
        kind: str,
        feature_columns: Sequence[str],
        *,
        seed: int,
        device: str = "mps",
        epochs: int = 30,
        patience: int = 6,
        temperature: float = 1.0,
        objective: str | None = None,
    ) -> None:
        if kind not in self.KINDS:
            raise ValueError(kind)
        self.kind = str(kind)
        self.feature_columns = tuple(map(str, feature_columns))
        self.seed = int(seed)
        self.device = str(device)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.temperature = (
            0.5
            if kind == "mlp_listwise_t05"
            else 2.0
            if kind == "mlp_listwise_t20"
            else float(temperature)
        )
        default_objective = (
            "bce"
            if kind == "mlp_bce"
            else "ranknet"
            if kind == "mlp_ranknet"
            else "hard_top_ranknet"
            if kind == "mlp_ranknet_hard_top"
            else "hybrid"
            if kind == "mlp_listwise_hybrid"
            else "listwise"
        )
        self.objective = default_objective if objective is None else str(objective)
        if self.objective not in {
            "bce",
            "ranknet",
            "hard_top_ranknet",
            "listwise",
            "hybrid",
        }:
            raise ValueError(f"unsupported neural objective: {self.objective}")
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.model: torch.nn.Module | None = None
        self.training_result: dict[str, Any] | None = None
        self.pos_weight: float | None = None

    def _normalize(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        matrix, median = _finite_fill(frame, self.feature_columns, None if fit else self.median)
        if fit:
            self.median = median
            self.mean = matrix.mean(axis=0)
            scale = matrix.std(axis=0)
            self.scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        if self.mean is None or self.scale is None:
            raise RuntimeError("preprocessor not fitted")
        return ((matrix - self.mean) / self.scale).astype(np.float32)

    @staticmethod
    def _base(frame: pd.DataFrame) -> np.ndarray:
        probability = np.clip(frame["source_score_calibrated"].to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)
        return np.log(probability / (1.0 - probability)).astype(np.float32)

    @staticmethod
    def _raw_edge(group: pd.DataFrame) -> np.ndarray:
        cluster = np.floor(group["center_x"].to_numpy(dtype=float) / 32.0) + 32.0 * np.floor(group["center_y"].to_numpy(dtype=float) / 32.0)
        values = np.column_stack(
            [
                group["center_x"].to_numpy(dtype=float),
                group["center_y"].to_numpy(dtype=float),
                np.radians(group["angle_deg"].to_numpy(dtype=float)),
                group["width_px"].to_numpy(dtype=float),
                group["original_score"].to_numpy(dtype=float),
                group["grasp_axis_mask_support"].fillna(0).to_numpy(dtype=float),
                group["center_depth_m"].fillna(0).to_numpy(dtype=float),
                group["sweep_minimum_clearance_proxy"].fillna(0).to_numpy(dtype=float),
                group["approach_collision_proxy"].fillna(0).to_numpy(dtype=float),
                cluster,
            ]
        )
        values[~np.isfinite(values)] = 0.0
        return values.astype(np.float32)

    def _examples(self, frame: pd.DataFrame, matrix: np.ndarray, *, labels: bool) -> list[CandidateSetExample]:
        examples: list[CandidateSetExample] = []
        base = self._base(frame)
        for sample_id, indices in frame.groupby("sample_id", sort=False).indices.items():
            index = np.asarray(indices, dtype=int)
            local = frame.iloc[index]
            examples.append(
                CandidateSetExample(
                    features=torch.from_numpy(matrix[index]),
                    labels=torch.from_numpy(
                        local["candidate_correct"].to_numpy(dtype=np.float32)
                        if labels
                        else np.zeros(len(local), dtype=np.float32)
                    ),
                    query_id=str(sample_id),
                    baseline_scores=torch.from_numpy(base[index]),
                    raw_edge_inputs=(
                        torch.from_numpy(self._raw_edge(local)) if self.kind == "gnn" else None
                    ),
                )
            )
        return examples

    def _make_model(self) -> torch.nn.Module:
        width = len(self.feature_columns)
        if self.kind == "linear_listwise":
            return _LinearResidualScorer(width, residual_scale=0.5)
        if self.kind.startswith("mlp_"):
            return SharedMLPScorer(width, hidden_dims=(64, 32), dropout=0.1, mode="residual", residual_scale=0.5)
        if self.kind == "deepsets":
            return DeepSetsScorer(width, hidden_dim=64, dropout=0.0, mode="residual", residual_scale=0.5)
        if self.kind == "set_transformer":
            return SetTransformerScorer(width, hidden_dim=64, num_heads=4, num_blocks=2, dropout=0.0, mode="residual", residual_scale=0.5)
        return CandidateGNNScorer(
            width,
            edge_dim=len(PAIRWISE_EDGE_RELATION_FIELDS),
            hidden_dim=64,
            edge_hidden_dim=32,
            num_message_passing=2,
            graph_type="rule",
            k=8,
            mode="residual",
            residual_scale=0.5,
        )

    def _criterion(self) -> CompositeRerankingLoss:
        if self.objective == "bce":
            return CompositeRerankingLoss(bce_weight=1.0, pos_weight=self.pos_weight)
        if self.objective == "ranknet":
            return CompositeRerankingLoss(ranknet_weight=1.0)
        if self.objective == "hard_top_ranknet":
            return _HardTopPairLoss()
        if self.objective == "hybrid":
            return CompositeRerankingLoss(
                listwise_weight=1.0,
                bce_weight=0.2,
                pos_weight=self.pos_weight,
                temperature=self.temperature,
            )
        return CompositeRerankingLoss(listwise_weight=1.0, temperature=self.temperature)

    def fit(self, frame: pd.DataFrame, checkpoint_path: str | Path) -> "GenericNeuralModel":
        frame = frame.reset_index(drop=True)
        scenes = frame[["sample_id", "scene_id"]].drop_duplicates("sample_id")
        splitter = GroupKFold(n_splits=5)
        sample_train, sample_held = next(splitter.split(scenes, groups=scenes["scene_id"].astype(str)))
        train_ids = set(scenes.iloc[sample_train]["sample_id"].astype(str))
        held_ids = set(scenes.iloc[sample_held]["sample_id"].astype(str))
        train_mask = frame["sample_id"].astype(str).isin(train_ids).to_numpy()
        held_mask = frame["sample_id"].astype(str).isin(held_ids).to_numpy()
        train_frame, held_frame = frame.loc[train_mask].reset_index(drop=True), frame.loc[held_mask].reset_index(drop=True)
        # Inner-fold preprocessing and class weighting are fitted on inner
        # train only.  The held scenes remain a pure epoch-selection set.
        train_raw, tune_median = _finite_fill(train_frame, self.feature_columns)
        tune_mean = train_raw.mean(axis=0)
        tune_scale = train_raw.std(axis=0)
        tune_scale = np.where(np.isfinite(tune_scale) & (tune_scale > 1e-12), tune_scale, 1.0)
        held_raw, _ = _finite_fill(held_frame, self.feature_columns, tune_median)
        train_matrix = ((train_raw - tune_mean) / tune_scale).astype(np.float32)
        held_matrix = ((held_raw - tune_mean) / tune_scale).astype(np.float32)
        tune_positive = int(train_frame["candidate_correct"].astype(bool).sum())
        tune_negative = int((~train_frame["candidate_correct"].astype(bool)).sum())
        self.pos_weight = (
            float(tune_negative / tune_positive)
            if tune_positive and tune_negative
            else None
        )
        train_loader = make_candidate_dataloader(
            self._examples(train_frame, train_matrix, labels=True),
            batch_size=128,
            shuffle=True,
            seed=self.seed,
            batching_policy=LENGTH_BUCKETED_BATCHING_POLICY,
        )
        held_loader = make_candidate_dataloader(
            self._examples(held_frame, held_matrix, labels=True),
            batch_size=256,
            shuffle=False,
            seed=self.seed,
            batching_policy=LENGTH_BUCKETED_BATCHING_POLICY,
        )
        model = self._make_model()
        tune_checkpoint = Path(checkpoint_path).with_name(Path(checkpoint_path).stem + ".tune.pt")
        result = fit_neural_ranker(
            model,
            train_loader,
            held_loader,
            checkpoint_path=tune_checkpoint,
            criterion=self._criterion(),
            config=TrainingConfig(
                epochs=self.epochs,
                learning_rate=1e-3,
                weight_decay=1e-4,
                patience=self.patience,
                seed=self.seed,
                device=self.device,
            ),
            metadata={"stage": "inner_scene_grouped_epoch_selection", "kind": self.kind},
        )
        # Refit on all Train for the scene-held selected number of epochs.
        positive = int(frame["candidate_correct"].astype(bool).sum())
        negative = int((~frame["candidate_correct"].astype(bool)).sum())
        self.pos_weight = float(negative / positive) if positive and negative else None
        matrix = self._normalize(frame, fit=True)
        full_loader = make_candidate_dataloader(
            self._examples(frame.reset_index(drop=True), matrix, labels=True),
            batch_size=128,
            shuffle=True,
            seed=self.seed,
            batching_policy=LENGTH_BUCKETED_BATCHING_POLICY,
        )
        self.model = self._make_model()
        final = fit_neural_ranker(
            self.model,
            full_loader,
            full_loader,
            checkpoint_path=checkpoint_path,
            criterion=self._criterion(),
            config=TrainingConfig(
                epochs=max(int(result.best_epoch), 1),
                learning_rate=1e-3,
                weight_decay=1e-4,
                patience=max(int(result.best_epoch) + 1, 2),
                seed=self.seed,
                device=self.device,
            ),
            metadata={"stage": "full_train_refit", "kind": self.kind, "selected_epochs": result.best_epoch},
        )
        self.training_result = {
            "inner": asdict(result),
            "full_refit": asdict(final),
            "inner_train_scenes": int(train_frame["scene_id"].nunique()),
            "inner_held_scenes": int(held_frame["scene_id"].nunique()),
        }
        self.model.to("cpu")
        self.device = "cpu"
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        matrix = self._normalize(frame, fit=False)
        examples = self._examples(frame.reset_index(drop=True), matrix, labels=False)
        loader = make_candidate_dataloader(
            examples,
            batch_size=256,
            shuffle=False,
            seed=self.seed,
            batching_policy=None,
        )
        output: dict[str, np.ndarray] = {}
        self.model.eval().to(self.device)
        with torch.no_grad():
            for raw_batch in loader:
                batch = raw_batch.to(torch.device(self.device))
                scores, _ = _forward_candidate_batch(self.model, batch)
                values = scores.detach().cpu().numpy()
                for row, query_id in enumerate(batch.query_ids):
                    count = int(batch.lengths[row])
                    output[str(query_id)] = values[row, :count].astype(float)
        result = np.empty(len(frame), dtype=float)
        for sample_id, indices in frame.groupby("sample_id", sort=False).indices.items():
            index = np.asarray(indices, dtype=int)
            result[index] = output[str(sample_id)]
        if not np.isfinite(result).all():
            raise RuntimeError("neural ranker produced non-finite scores")
        return result

    def artifact(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "device_after_serialization": self.device,
            "epochs_budget": self.epochs,
            "patience": self.patience,
            "temperature": self.temperature,
            "objective": self.objective,
            "bce_pos_weight": self.pos_weight,
            "bce_query_mass_renormalized_after_class_weight": True,
            "feature_columns": list(self.feature_columns),
            "parameter_count": None if self.model is None else sum(parameter.numel() for parameter in self.model.parameters()),
            "training_result": self.training_result,
            "raw_edge_contract": (
                None
                if self.kind != "gnn"
                else {
                    "candidate_fields": ["x", "y", "theta", "width", "q", "mask_support", "depth", "clearance", "conflict_risk", "cluster_id"],
                    "relation_fields": list(PAIRWISE_EDGE_RELATION_FIELDS),
                    "graph": "rule complete-candidate relations; N<=14 in frozen pools",
                }
            ),
        }


__all__ = ["GenericNeuralModel", "LightGBMLambdaMARTModel", "TabularResidualModel"]
