from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import (
    batches,
    clone_state_dict,
    early_stopping_update,
    resolve_device,
    seed_everything,
)


def quality_logit(q: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    clipped = torch.clamp(q, epsilon, 1.0 - epsilon)
    return torch.log(clipped) - torch.log1p(-clipped)


class LatentResidualRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.normalization = nn.LayerNorm(input_dim)
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, features: torch.Tensor, q: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = torch.tanh(
            self.residual(self.normalization(features)).squeeze(-1)
        )
        score = quality_logit(q) + float(alpha) * residual
        return score, residual


def residual_scores(
    residual: np.ndarray, q: np.ndarray, alpha: float
) -> np.ndarray:
    """Recompose scores from the model's already-tanh-bounded residual."""
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return (
        np.log(q / (1.0 - q))
        + float(alpha) * np.asarray(residual, dtype=np.float64)
    )


def _residual_loss(
    scores: torch.Tensor,
    residual: torch.Tensor,
    labels: torch.Tensor,
    *,
    residual_penalty: float,
) -> torch.Tensor:
    positive_count = labels.sum(dim=1)
    valid = positive_count > 0
    targets = labels / positive_count.clamp_min(1.0)[:, None]
    per_sample = -(
        targets * F.log_softmax(scores, dim=1)
    ).sum(dim=1)
    listwise = (
        per_sample[valid].mean()
        if valid.any()
        else scores.sum() * 0.0
    )
    bce = F.binary_cross_entropy_with_logits(scores, labels)
    return listwise + 0.2 * bce + float(residual_penalty) * residual.square().mean()


def train_latent_residual_arrays(
    train_features: np.ndarray,
    train_q: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_q: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int = 37,
    device: str = "auto",
    epochs: int = 25,
    patience: int = 5,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    residual_penalty: float = 0.02,
) -> dict[str, Any]:
    seed_everything(seed)
    torch_device = resolve_device(device)
    input_dim = int(train_features.shape[-1])
    model = LatentResidualRanker(input_dim).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_value, best_state, stale = math.inf, None, 0
    history = []
    for epoch in range(int(epochs)):
        model.train()
        train_losses = []
        for indices in batches(
            len(train_features), batch_size, shuffle=True, seed=seed + epoch
        ):
            features = (
                torch.from_numpy(np.asarray(train_features[indices], dtype=np.float32))
                .to(torch_device)
            )
            q = torch.from_numpy(np.asarray(train_q[indices], dtype=np.float32)).to(
                torch_device
            )
            labels = torch.from_numpy(
                np.asarray(train_labels[indices], dtype=np.float32)
            ).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            scores, residual = model(features, q, alpha=1.0)
            loss = _residual_loss(
                scores,
                residual,
                labels,
                residual_penalty=residual_penalty,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for indices in batches(
                len(validation_features), batch_size, shuffle=False, seed=0
            ):
                features = torch.from_numpy(
                    np.asarray(validation_features[indices], dtype=np.float32)
                ).to(torch_device)
                q = torch.from_numpy(
                    np.asarray(validation_q[indices], dtype=np.float32)
                ).to(torch_device)
                labels = torch.from_numpy(
                    np.asarray(validation_labels[indices], dtype=np.float32)
                ).to(torch_device)
                scores, residual = model(features, q, alpha=1.0)
                validation_losses.append(
                    float(
                        _residual_loss(
                            scores,
                            residual,
                            labels,
                            residual_penalty=residual_penalty,
                        ).cpu()
                    )
                )
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
            }
        )
        best_value, best_state, stale = early_stopping_update(
            validation_loss,
            best_value=best_value,
            best_state=best_state,
            model=model,
            stale=stale,
        )
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("latent residual training produced no checkpoint")
    return {
        "kind": "frozen_crog_latent_roi_residual",
        "input_dim": input_dim,
        "model_state_dict": best_state,
        "residual_penalty": float(residual_penalty),
        "history": history,
        "seed": int(seed),
    }


def predict_latent_residual_arrays(
    artifact: dict[str, Any],
    features: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float,
    device: str = "auto",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    torch_device = resolve_device(device)
    model = LatentResidualRanker(int(artifact["input_dim"]))
    model.load_state_dict(artifact["model_state_dict"])
    model.to(torch_device).eval()
    scores, residuals = [], []
    with torch.no_grad():
        for indices in batches(len(features), batch_size, shuffle=False, seed=0):
            value = torch.from_numpy(
                np.asarray(features[indices], dtype=np.float32)
            ).to(torch_device)
            quality = torch.from_numpy(np.asarray(q[indices], dtype=np.float32)).to(
                torch_device
            )
            score, residual = model(value, quality, alpha=alpha)
            scores.append(score.cpu().numpy())
            residuals.append(residual.cpu().numpy())
    return np.concatenate(scores), np.concatenate(residuals)
