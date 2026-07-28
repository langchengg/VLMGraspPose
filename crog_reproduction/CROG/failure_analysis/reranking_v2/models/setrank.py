from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .latent_residual import quality_logit
from .common import (
    batches,
    early_stopping_update,
    resolve_device,
    seed_everything,
)


class ResidualSetRank(nn.Module):
    """Permutation-equivariant K-candidate encoder with explicit q residual."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.residual = nn.Linear(model_dim, 1)
        self.correctness = nn.Linear(model_dim, 1)
        self.any_positive = nn.Linear(model_dim, 1)

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        q: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
        alpha: float = 1.0,
    ):
        padding_mask = None if candidate_mask is None else ~candidate_mask.bool()
        encoded = self.encoder(
            self.input(candidate_tokens), src_key_padding_mask=padding_mask
        )
        residual = torch.tanh(self.residual(encoded).squeeze(-1))
        scores = quality_logit(q) + float(alpha) * residual
        correctness_logits = self.correctness(encoded).squeeze(-1)
        if candidate_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            weights = candidate_mask.float().unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        any_positive_logit = self.any_positive(pooled).squeeze(-1)
        return scores, correctness_logits, any_positive_logit, residual


def setrank_loss(
    scores: torch.Tensor,
    correctness_logits: torch.Tensor,
    any_positive_logit: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    mask = candidate_mask.bool()
    positives = (labels > 0.5) & mask
    has_positive = positives.any(dim=1)
    positive_count = (labels * mask).sum(dim=1)
    target = (labels * mask) / positive_count.clamp_min(1.0)[:, None]
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    per_sample = -(
        target * F.log_softmax(masked_scores, dim=1)
    ).sum(dim=1)
    listwise = (
        per_sample[has_positive].mean()
        if has_positive.any()
        else scores.sum() * 0.0
    )
    candidate_bce = F.binary_cross_entropy_with_logits(
        correctness_logits[mask], labels[mask]
    )
    any_bce = F.binary_cross_entropy_with_logits(
        any_positive_logit, has_positive.float()
    )
    return listwise + 0.2 * candidate_bce + 0.2 * any_bce


def train_setrank_arrays(
    train_tokens: np.ndarray,
    train_q: np.ndarray,
    train_labels: np.ndarray,
    validation_tokens: np.ndarray,
    validation_q: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int = 41,
    device: str = "auto",
    epochs: int = 25,
    patience: int = 5,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
) -> dict[str, Any]:
    seed_everything(seed)
    token_mean = np.asarray(train_tokens, dtype=np.float64).mean(axis=(0, 1))
    token_scale = np.asarray(train_tokens, dtype=np.float64).std(axis=(0, 1))
    token_scale[token_scale < 1e-8] = 1.0
    train_tokens = (
        (np.asarray(train_tokens, dtype=np.float32) - token_mean) / token_scale
    ).astype(np.float32)
    validation_tokens = (
        (np.asarray(validation_tokens, dtype=np.float32) - token_mean)
        / token_scale
    ).astype(np.float32)
    torch_device = resolve_device(device)
    model = ResidualSetRank(int(train_tokens.shape[-1])).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_value, best_state, stale = math.inf, None, 0
    history = []
    for epoch in range(int(epochs)):
        model.train()
        train_losses = []
        for indices in batches(
            len(train_tokens), batch_size, shuffle=True, seed=seed + epoch
        ):
            tokens = torch.from_numpy(
                np.asarray(train_tokens[indices], dtype=np.float32)
            ).to(torch_device)
            q = torch.from_numpy(np.asarray(train_q[indices], dtype=np.float32)).to(
                torch_device
            )
            labels = torch.from_numpy(
                np.asarray(train_labels[indices], dtype=np.float32)
            ).to(torch_device)
            mask = torch.ones_like(labels, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            scores, correctness, any_positive, _ = model(
                tokens, q, candidate_mask=mask, alpha=1.0
            )
            loss = setrank_loss(scores, correctness, any_positive, labels, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for indices in batches(
                len(validation_tokens), batch_size, shuffle=False, seed=0
            ):
                tokens = torch.from_numpy(
                    np.asarray(validation_tokens[indices], dtype=np.float32)
                ).to(torch_device)
                q = torch.from_numpy(
                    np.asarray(validation_q[indices], dtype=np.float32)
                ).to(torch_device)
                labels = torch.from_numpy(
                    np.asarray(validation_labels[indices], dtype=np.float32)
                ).to(torch_device)
                mask = torch.ones_like(labels, dtype=torch.bool)
                scores, correctness, any_positive, _ = model(
                    tokens, q, candidate_mask=mask, alpha=1.0
                )
                validation_losses.append(
                    float(
                        setrank_loss(
                            scores, correctness, any_positive, labels, mask
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
        raise RuntimeError("SetRank training produced no checkpoint")
    return {
        "kind": "residual_listwise_setrank",
        "input_dim": int(train_tokens.shape[-1]),
        "model_state_dict": best_state,
        "history": history,
        "seed": int(seed),
        "token_mean": token_mean.astype(np.float32),
        "token_scale": token_scale.astype(np.float32),
    }


def predict_setrank_arrays(
    artifact: dict[str, Any],
    tokens: np.ndarray,
    q: np.ndarray,
    *,
    alpha: float = 1.0,
    device: str = "auto",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    torch_device = resolve_device(device)
    tokens = (
        (
            np.asarray(tokens, dtype=np.float32)
            - np.asarray(artifact["token_mean"], dtype=np.float32)
        )
        / np.asarray(artifact["token_scale"], dtype=np.float32)
    )
    model = ResidualSetRank(int(artifact["input_dim"]))
    model.load_state_dict(artifact["model_state_dict"])
    model.to(torch_device).eval()
    scores, probabilities, residuals = [], [], []
    with torch.no_grad():
        for indices in batches(len(tokens), batch_size, shuffle=False, seed=0):
            token = torch.from_numpy(
                np.asarray(tokens[indices], dtype=np.float32)
            ).to(torch_device)
            quality = torch.from_numpy(np.asarray(q[indices], dtype=np.float32)).to(
                torch_device
            )
            mask = torch.ones_like(quality, dtype=torch.bool)
            score, correctness, _, residual = model(
                token, quality, candidate_mask=mask, alpha=alpha
            )
            scores.append(score.cpu().numpy())
            probabilities.append(torch.sigmoid(correctness).cpu().numpy())
            residuals.append(residual.cpu().numpy())
    return (
        np.concatenate(scores),
        np.concatenate(probabilities),
        np.concatenate(residuals),
    )
