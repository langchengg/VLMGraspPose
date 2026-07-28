from __future__ import annotations

from typing import Any

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available single-process device."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if prefer_mps and mps_available:
        return torch.device("mps")
    if cuda_available:
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors while preserving container structure."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def empty_cache(device: torch.device) -> None:
    """Release unused accelerator cache for the selected backend."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def get_memory_stats(device: torch.device) -> dict[str, int]:
    """Return allocator statistics in bytes for the selected accelerator."""
    if device.type == "mps" and hasattr(torch, "mps"):
        stats = {
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_bytes": int(torch.mps.driver_allocated_memory()),
        }
        if hasattr(torch.mps, "recommended_max_memory"):
            stats["recommended_bytes"] = int(torch.mps.recommended_max_memory())
        return stats
    if device.type == "cuda":
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "driver_bytes": int(torch.cuda.memory_reserved(device)),
        }
    return {}


def record_memory_sample(samples: list[dict[str, Any]], device: torch.device,
                         phase: str, step: int) -> dict[str, Any]:
    stats = get_memory_stats(device)
    if not stats:
        return {}
    sample = {"phase": phase, "step": int(step), **stats}
    samples.append(sample)
    return sample
