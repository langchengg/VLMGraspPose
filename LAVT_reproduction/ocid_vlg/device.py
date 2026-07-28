"""Device selection and CUDA-only multi-GPU helpers."""

from __future__ import annotations

import os
from typing import Union

# PyTorch reads this environment variable when dispatching unsupported MPS ops.
# setdefault preserves an explicit user choice.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

DeviceLike = Union[str, torch.device]
_SUPPORTED_DEVICES = ("auto", "cuda", "mps", "cpu")


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(requested: DeviceLike = "auto") -> torch.device:
    """Resolve ``auto|cuda|mps|cpu`` without silently changing explicit choices.

    ``auto`` follows the required priority CUDA, then MPS, then CPU.  Indexed
    CUDA strings such as ``cuda:1`` are accepted for programmatic use even
    though the command-line interface only needs the four canonical choices.
    """

    if isinstance(requested, torch.device):
        requested_value = str(requested)
    else:
        requested_value = str(requested).strip().lower()

    if requested_value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(requested_value)
    except (RuntimeError, ValueError) as exc:
        choices = ", ".join(_SUPPORTED_DEVICES)
        raise ValueError(
            f"Unsupported device {requested_value!r}; use one of: {choices}."
        ) from exc

    if device.type not in {"cuda", "mps", "cpu"}:
        choices = ", ".join(_SUPPORTED_DEVICES)
        raise ValueError(
            f"Unsupported device {requested_value!r}; use one of: {choices}."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


# Friendly aliases used by command-line and training code.
select_device = resolve_device
get_device = resolve_device


def is_cuda_device(device: DeviceLike) -> bool:
    """Return whether *device* names the CUDA backend."""

    value = device if isinstance(device, torch.device) else torch.device(device)
    return value.type == "cuda"


def should_pin_memory(device: DeviceLike) -> bool:
    """Pin DataLoader memory only for CUDA host-to-device transfers."""

    return is_cuda_device(device)


def should_use_cuda_ddp(
    device: DeviceLike, *, single_process: bool = False
) -> bool:
    """Return whether CUDA DistributedDataParallel is useful on this host.

    CPU and MPS intentionally never enter the distributed path.  This helper
    only expresses eligibility; process-group initialization remains the
    caller's responsibility.
    """

    return (
        not single_process
        and is_cuda_device(device)
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )


def cuda_device_ids(
    device: DeviceLike, *, single_process: bool = False
) -> list[int]:
    """Return visible CUDA IDs for DDP, or an empty list for single-device use."""

    if not should_use_cuda_ddp(device, single_process=single_process):
        return []
    return list(range(torch.cuda.device_count()))


# Compatibility aliases for training code that describes the condition rather
# than the PyTorch implementation.
use_distributed_training = should_use_cuda_ddp
is_cuda_multi_gpu = should_use_cuda_ddp


__all__ = [
    "DeviceLike",
    "cuda_device_ids",
    "get_device",
    "is_cuda_device",
    "is_cuda_multi_gpu",
    "resolve_device",
    "select_device",
    "should_pin_memory",
    "should_use_cuda_ddp",
    "use_distributed_training",
]
