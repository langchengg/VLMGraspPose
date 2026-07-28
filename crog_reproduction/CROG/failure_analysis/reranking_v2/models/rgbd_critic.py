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


class RGBDGraspCritic(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, crops: torch.Tensor):
        encoded = self.encoder(crops).flatten(1)
        embedding = torch.tanh(self.embedding(encoded))
        return self.classifier(embedding).squeeze(-1), embedding


def grouped_candidate_batches(
    sample_index: np.ndarray,
    maximum_candidates: int,
    *,
    shuffle: bool,
    seed: int,
):
    """Yield whole candidate lists so within-list pairwise loss is observable."""
    sample_index = np.asarray(sample_index)
    order = (
        np.arange(len(sample_index))
        if np.all(sample_index[:-1] <= sample_index[1:])
        else np.argsort(sample_index, kind="stable")
    )
    ordered_groups = sample_index[order]
    boundaries = np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1
    groups = list(np.split(order, boundaries))
    if shuffle:
        permutation = np.random.default_rng(seed).permutation(len(groups))
        groups = [groups[index] for index in permutation]
    pending: list[np.ndarray] = []
    pending_count = 0
    for candidate_indices in groups:
        if pending and pending_count + len(candidate_indices) > maximum_candidates:
            yield np.concatenate(pending)
            pending, pending_count = [], 0
        pending.append(candidate_indices)
        pending_count += len(candidate_indices)
    if pending:
        yield np.concatenate(pending)


def critic_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_index: torch.Tensor,
    q_values: torch.Tensor,
    *,
    positive_weight: float,
    pairwise_weight: float = 0.25,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(float(positive_weight), device=logits.device),
    )
    unique, counts = torch.unique_consecutive(
        sample_index, return_counts=True
    )
    if len(unique) and torch.all(counts == counts[0]):
        candidates = int(counts[0])
        grouped_logits = logits.reshape(-1, candidates)
        grouped_labels = labels.reshape(-1, candidates)
        grouped_q = q_values.reshape(-1, candidates)
        positive = grouped_labels > 0.5
        negative = ~positive
        pair_mask = positive[:, :, None] & negative[:, None, :]
        margin = (
            1.0
            - grouped_logits[:, :, None]
            + grouped_logits[:, None, :]
        )
        hard = (
            grouped_q[:, None, :] > grouped_q[:, :, None]
        ).float()
        weighted = F.relu(margin) * (1.0 + hard) * pair_mask
        pair_count = pair_mask.sum(dim=(1, 2))
        valid = pair_count > 0
        pairwise = (
            (
                weighted.sum(dim=(1, 2))[valid]
                / pair_count[valid]
            ).mean()
            if valid.any()
            else bce * 0.0
        )
    else:
        # Defensive fallback for callers that do not provide complete lists.
        pair_losses = []
        for sample in torch.unique(sample_index):
            keep = sample_index == sample
            positive = keep & (labels > 0.5)
            negative = keep & (labels <= 0.5)
            if positive.any() and negative.any():
                positive_logits = logits[positive]
                negative_logits = logits[negative]
                positive_q = q_values[positive]
                negative_q = q_values[negative]
                margin = (
                    1.0
                    - positive_logits[:, None]
                    + negative_logits[None, :]
                )
                hard = (
                    negative_q[None, :] > positive_q[:, None]
                ).float()
                pair_losses.append(
                    (F.relu(margin) * (1.0 + hard)).mean()
                )
        pairwise = (
            torch.stack(pair_losses).mean()
            if pair_losses
            else bce * 0.0
        )
    return bce + float(pairwise_weight) * pairwise


def train_critic_arrays(
    train_crops: np.ndarray,
    train_labels: np.ndarray,
    train_sample_index: np.ndarray,
    train_q: np.ndarray,
    validation_crops: np.ndarray,
    validation_labels: np.ndarray,
    validation_sample_index: np.ndarray,
    validation_q: np.ndarray,
    *,
    channels: tuple[int, ...] | None = None,
    seed: int = 31,
    device: str = "auto",
    epochs: int = 20,
    patience: int = 4,
    learning_rate: float = 5e-4,
    batch_size: int = 512,
    train_candidate_indices: np.ndarray | None = None,
    validation_candidate_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    if channels is None:
        channels = tuple(range(train_crops.shape[1]))
    # Keep the shared crop artifact in float16 host memory. Converting the
    # complete 63k-expression cohort to float32 would temporarily require well
    # over the 24 GB target-machine budget; each minibatch is converted below.
    train_crops = np.asarray(train_crops)
    validation_crops = np.asarray(validation_crops)
    channel_index = np.asarray(channels, dtype=np.int64)
    train_candidate_indices = (
        np.arange(len(train_crops), dtype=np.int64)
        if train_candidate_indices is None
        else np.asarray(train_candidate_indices, dtype=np.int64)
    )
    validation_candidate_indices = (
        np.arange(len(validation_crops), dtype=np.int64)
        if validation_candidate_indices is None
        else np.asarray(validation_candidate_indices, dtype=np.int64)
    )
    train_group_index = np.asarray(train_sample_index)[
        train_candidate_indices
    ]
    validation_group_index = np.asarray(validation_sample_index)[
        validation_candidate_indices
    ]
    positives = float(
        np.sum(np.asarray(train_labels)[train_candidate_indices] > 0.5)
    )
    positive_weight = (
        float((len(train_candidate_indices) - positives) / positives)
        if positives
        else 1.0
    )
    torch_device = resolve_device(device)
    model = RGBDGraspCritic(len(channels)).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    best_value, best_state, stale = math.inf, None, 0
    history = []
    for epoch in range(int(epochs)):
        model.train()
        train_losses = []
        for local_indices in grouped_candidate_batches(
            train_group_index,
            batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            indices = train_candidate_indices[local_indices]
            crop = torch.from_numpy(
                np.asarray(
                    train_crops[indices][:, channel_index],
                    dtype=np.float32,
                )
            ).to(torch_device)
            label = torch.from_numpy(train_labels[indices]).float().to(torch_device)
            sample = torch.from_numpy(train_sample_index[indices]).long().to(torch_device)
            q_value = torch.from_numpy(train_q[indices]).float().to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(crop)
            loss = critic_loss(
                logits,
                label,
                sample,
                q_value,
                positive_weight=positive_weight,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for local_indices in grouped_candidate_batches(
                validation_group_index,
                batch_size,
                shuffle=False,
                seed=0,
            ):
                indices = validation_candidate_indices[local_indices]
                crop = torch.from_numpy(
                    np.asarray(
                        validation_crops[indices][:, channel_index],
                        dtype=np.float32,
                    )
                ).to(torch_device)
                label = (
                    torch.from_numpy(validation_labels[indices])
                    .float()
                    .to(torch_device)
                )
                sample = (
                    torch.from_numpy(validation_sample_index[indices])
                    .long()
                    .to(torch_device)
                )
                q_value = (
                    torch.from_numpy(validation_q[indices]).float().to(torch_device)
                )
                logits, _ = model(crop)
                validation_losses.append(
                    float(
                        critic_loss(
                            logits,
                            label,
                            sample,
                            q_value,
                            positive_weight=positive_weight,
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
        raise RuntimeError("critic training produced no finite checkpoint")
    return {
        "kind": "candidate_aligned_rgbd_critic",
        "model_state_dict": best_state,
        "input_channels": len(channels),
        "channel_indices": channels,
        "embedding_dim": 64,
        "positive_weight": positive_weight,
        "history": history,
        "seed": int(seed),
    }


def predict_critic_arrays(
    artifact: dict[str, Any],
    crops: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 1024,
    candidate_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    channels = tuple(int(value) for value in artifact["channel_indices"])
    values = np.asarray(crops)
    channel_index = np.asarray(channels, dtype=np.int64)
    candidate_indices = (
        np.arange(len(values), dtype=np.int64)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=np.int64)
    )
    torch_device = resolve_device(device)
    model = RGBDGraspCritic(
        int(artifact["input_channels"]), int(artifact.get("embedding_dim", 64))
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.to(torch_device).eval()
    scores, embeddings = [], []
    with torch.no_grad():
        for positions in batches(
            len(candidate_indices),
            batch_size,
            shuffle=False,
            seed=0,
        ):
            indices = candidate_indices[positions]
            logits, embedding = model(
                torch.from_numpy(
                    np.asarray(
                        values[indices][:, channel_index],
                        dtype=np.float32,
                    )
                ).to(torch_device)
            )
            scores.append(logits.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
    return np.concatenate(scores), np.concatenate(embeddings)
