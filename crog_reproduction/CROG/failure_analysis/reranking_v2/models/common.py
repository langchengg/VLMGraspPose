from __future__ import annotations

import random
from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def batches(
    size: int, batch_size: int, *, shuffle: bool, seed: int
) -> Iterable[np.ndarray]:
    indices = np.arange(int(size))
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return result


def early_stopping_update(
    value: float,
    *,
    best_value: float,
    best_state,
    model: nn.Module,
    stale: int,
    tolerance: float = 1e-8,
):
    if np.isfinite(value) and value < best_value - tolerance:
        return value, clone_state_dict(model), 0
    return best_value, best_state, stale + 1

