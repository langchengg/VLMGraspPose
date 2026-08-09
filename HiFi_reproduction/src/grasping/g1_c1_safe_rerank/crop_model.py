"""Small candidate-aligned crop residual scorer (R13, no geometry refinement)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from reranking.losses import CompositeRerankingLoss

from .experiment_models import _finite_fill


class CropResidualNetwork(torch.nn.Module):
    def __init__(self, scalar_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(64 + scalar_dim),
            torch.nn.Linear(64 + scalar_dim, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1),
        )

    def forward(self, crop: torch.Tensor, scalar: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(crop).flatten(1)
        residual = self.head(torch.cat([latent, scalar], dim=1)).squeeze(1)
        return base + 0.5 * torch.tanh(residual)


@dataclass
class CropResidualModel:
    feature_columns: tuple[str, ...]
    seed: int = 17
    device: str = "mps"
    epochs: int = 8
    objective: str = "bce"
    temperature: float = 1.0
    median: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    model: CropResidualNetwork | None = None
    best_epoch: int | None = None

    def _scalar(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        values, median = _finite_fill(frame, self.feature_columns, None if fit else self.median)
        if fit:
            self.median = median
            self.mean = values.mean(axis=0)
            std = values.std(axis=0)
            self.scale = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
        if self.mean is None or self.scale is None:
            raise RuntimeError("crop scalar preprocessor not fitted")
        return ((values - self.mean) / self.scale).astype(np.float32)

    @staticmethod
    def _base(frame: pd.DataFrame) -> np.ndarray:
        probability = np.clip(frame["source_score_calibrated"].to_numpy(dtype=float), 1e-4, 1 - 1e-4)
        return np.log(probability / (1 - probability)).astype(np.float32)

    @staticmethod
    def _weights(frame: pd.DataFrame) -> np.ndarray:
        labels = frame["candidate_correct"].astype(bool).to_numpy()
        positive, negative = int(labels.sum()), int((~labels).sum())
        pos_weight = negative / positive if positive and negative else 1.0
        raw = np.where(labels, pos_weight, 1.0)
        sample = frame["sample_id"].astype(str)
        denominator = pd.Series(raw).groupby(sample, sort=False).transform("sum").to_numpy(dtype=float)
        return (raw / denominator).astype(np.float32)

    def _train_epochs(
        self,
        frame: pd.DataFrame,
        crops: np.ndarray,
        scalar: np.ndarray,
        *,
        epochs: int,
        held: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None,
    ) -> tuple[CropResidualNetwork, int]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(self.device)
        model = CropResidualNetwork(len(self.feature_columns)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        labels = frame["candidate_correct"].astype(float).to_numpy(dtype=np.float32)
        base = self._base(frame)
        positive, negative = int(labels.sum()), int((labels == 0).sum())
        pos_weight = float(negative / positive) if positive and negative else None
        criterion = (
            CompositeRerankingLoss(bce_weight=1.0, pos_weight=pos_weight)
            if self.objective == "bce"
            else CompositeRerankingLoss(ranknet_weight=1.0)
            if self.objective == "ranknet"
            else CompositeRerankingLoss(
                listwise_weight=1.0,
                temperature=self.temperature,
            )
        )
        if self.objective not in {"bce", "ranknet", "listwise"}:
            raise ValueError(f"unsupported crop objective: {self.objective}")
        query_indices = [
            np.asarray(indices, dtype=int)
            for indices in frame.groupby("sample_id", sort=False).indices.values()
        ]
        rng = np.random.default_rng(self.seed)
        best_state, best_epoch, best_j = copy.deepcopy(model.state_dict()), 1, -1.0
        for epoch in range(1, int(epochs) + 1):
            model.train()
            order = rng.permutation(len(query_indices))
            for start in range(0, len(order), 64):
                groups = [query_indices[position] for position in order[start : start + 64]]
                index = np.concatenate(groups)
                crop_tensor = torch.from_numpy(crops[index]).to(device=device, dtype=torch.float32) / 255.0
                scalar_tensor = torch.from_numpy(scalar[index]).to(device)
                base_tensor = torch.from_numpy(base[index]).to(device)
                optimizer.zero_grad(set_to_none=True)
                score = model(crop_tensor, scalar_tensor, base_tensor)
                maximum = max(map(len, groups))
                score_rows, label_rows, base_rows, padding_rows = [], [], [], []
                offset = 0
                for group in groups:
                    count = len(group)
                    pad = maximum - count
                    score_rows.append(torch.nn.functional.pad(score[offset : offset + count], (0, pad)))
                    label_rows.append(
                        torch.nn.functional.pad(
                            torch.from_numpy(labels[group]).to(device), (0, pad)
                        )
                    )
                    base_rows.append(
                        torch.nn.functional.pad(
                            torch.from_numpy(base[group]).to(device), (0, pad)
                        )
                    )
                    padding_rows.append(
                        torch.cat(
                            [
                                torch.zeros(count, dtype=torch.bool, device=device),
                                torch.ones(pad, dtype=torch.bool, device=device),
                            ]
                        )
                    )
                    offset += count
                loss = criterion.active_breakdown(
                    torch.stack(score_rows),
                    torch.stack(label_rows),
                    torch.stack(padding_rows),
                    baseline_scores=torch.stack(base_rows),
                ).total
                loss.backward()
                optimizer.step()
            if held is None:
                best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
                continue
            held_frame, held_crop, held_scalar = held
            scores = self._predict_network(model, held_frame, held_crop, held_scalar, device=device)
            scored = held_frame[["sample_id", "stable_candidate_id", "candidate_correct"]].copy()
            scored["score"] = scores
            selected = scored.sort_values(["sample_id", "score", "stable_candidate_id"], ascending=[True, False, True], kind="mergesort").groupby("sample_id", sort=False).head(1)
            j = float(selected["candidate_correct"].mean())
            if j > best_j + 1e-12:
                best_j, best_epoch, best_state = j, epoch, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        return model, best_epoch

    def fit(self, frame: pd.DataFrame, crops: np.ndarray, checkpoint: str | Path) -> "CropResidualModel":
        frame = frame.reset_index(drop=True)
        if crops.shape != (len(frame), 4, 64, 64) or crops.dtype != np.uint8:
            raise ValueError("crop tensor contract mismatch")
        scenes = frame[["sample_id", "scene_id"]].drop_duplicates("sample_id")
        fit_index, held_index = next(GroupKFold(n_splits=5).split(scenes, groups=scenes["scene_id"].astype(str)))
        fit_ids = set(scenes.iloc[fit_index]["sample_id"].astype(str))
        held_ids = set(scenes.iloc[held_index]["sample_id"].astype(str))
        train_mask = frame["sample_id"].astype(str).isin(fit_ids).to_numpy()
        held_mask = frame["sample_id"].astype(str).isin(held_ids).to_numpy()
        tune_frame, tune_held = frame.loc[train_mask].reset_index(drop=True), frame.loc[held_mask].reset_index(drop=True)
        tune_raw, tune_median = _finite_fill(tune_frame, self.feature_columns)
        tune_mean = tune_raw.mean(axis=0)
        tune_scale = tune_raw.std(axis=0)
        tune_scale = np.where(np.isfinite(tune_scale) & (tune_scale > 1e-12), tune_scale, 1.0)
        held_raw, _ = _finite_fill(tune_held, self.feature_columns, tune_median)
        tune_scalar = ((tune_raw - tune_mean) / tune_scale).astype(np.float32)
        held_scalar = ((held_raw - tune_mean) / tune_scale).astype(np.float32)
        _, selected_epoch = self._train_epochs(
            tune_frame,
            crops[train_mask],
            tune_scalar,
            epochs=self.epochs,
            held=(tune_held, crops[held_mask], held_scalar),
        )
        full_scalar = self._scalar(frame, fit=True)
        self.model, _ = self._train_epochs(frame, crops, full_scalar, epochs=selected_epoch, held=None)
        self.best_epoch = selected_epoch
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "artifact": self.artifact()}, checkpoint)
        self.model.to("cpu")
        self.device = "cpu"
        return self

    @staticmethod
    def _predict_network(model: CropResidualNetwork, frame: pd.DataFrame, crops: np.ndarray, scalar: np.ndarray, *, device: torch.device) -> np.ndarray:
        model.eval()
        base = CropResidualModel._base(frame)
        scores = np.empty(len(frame), dtype=float)
        with torch.no_grad():
            for start in range(0, len(frame), 512):
                stop = min(len(frame), start + 512)
                value = model(
                    torch.from_numpy(crops[start:stop]).to(device=device, dtype=torch.float32) / 255.0,
                    torch.from_numpy(scalar[start:stop]).to(device),
                    torch.from_numpy(base[start:stop]).to(device),
                )
                scores[start:stop] = value.detach().cpu().numpy()
        return scores

    def predict_scores(self, frame: pd.DataFrame, crops: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("crop model is not fitted")
        if crops.shape != (len(frame), 4, 64, 64):
            raise ValueError("crop prediction coverage mismatch")
        device = torch.device(self.device)
        self.model.to(device)
        return self._predict_network(self.model, frame.reset_index(drop=True), crops, self._scalar(frame, fit=False), device=device)

    def artifact(self) -> dict[str, Any]:
        model = self.model or CropResidualNetwork(len(self.feature_columns))
        return {
            "kind": "candidate_aligned_crop_cnn_residual",
            "seed": self.seed,
            "feature_columns": list(self.feature_columns),
            "crop_shape": [4, 64, 64],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_limit": 500_000,
            "best_inner_scene_held_epoch": self.best_epoch,
            "objective": self.objective,
            "temperature": self.temperature,
            "geometry_refinement": False,
        }


__all__ = ["CropResidualModel", "CropResidualNetwork"]
