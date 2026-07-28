"""Small compatibility helpers replacing legacy training-only dependencies.

LAVT originally imported three tensor helpers from ``timm`` and used MMCV only
to read a local Swin checkpoint.  Keeping those old binary packages would make
the code unusable on current Apple Silicon Python builds.  The implementations
below use public PyTorch APIs and preserve the original model computation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn


def to_2tuple(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple):
        return value
    return value, value


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


class DropPath(nn.Module):
    """Per-sample stochastic depth with the same semantics used by Swin."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep_prob = 1.0 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=inputs.dtype, device=inputs.device
        )
        return inputs.div(keep_prob) * random_tensor.floor()


def get_root_logger() -> logging.Logger:
    logger = logging.getLogger("lavt")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _safe_torch_load(filename: str | Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(filename, map_location=map_location)


def load_checkpoint(
    model: nn.Module,
    filename: str | Path,
    map_location: str = "cpu",
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Load a local PyTorch checkpoint and retain a machine-readable audit."""

    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = _safe_torch_load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint is not a mapping: {path}")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise RuntimeError(f"checkpoint has no state dictionary: {path}")

    normalized = OrderedDict()
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    incompatible = model.load_state_dict(normalized, strict=strict)
    compatible_loaded_key_count = len(normalized) - len(incompatible.unexpected_keys)
    audit = {
        "path": str(path),
        "strict": bool(strict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_key_count": len(normalized),
        "loaded_key_count": compatible_loaded_key_count,
    }
    model.pretrained_load_audit = audit
    target_logger = logger or get_root_logger()
    target_logger.info(
        "Loaded %s keys from %s (missing=%s, unexpected=%s)",
        compatible_loaded_key_count,
        path,
        len(audit["missing_keys"]),
        len(audit["unexpected_keys"]),
    )
    return checkpoint
