"""Atomic, full-state checkpoints for continuous training resume."""

from __future__ import annotations

import dataclasses
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .device import DeviceLike, resolve_device

CHECKPOINT_FORMAT_VERSION = 1


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _unwrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    parallel_types = (
        torch.nn.DataParallel,
        torch.nn.parallel.DistributedDataParallel,
    )
    return model.module if isinstance(model, parallel_types) else model


def _plain_config(value: Any) -> Any:
    """Convert common resolved-config containers into stable built-in values."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    elif not isinstance(value, Mapping) and hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_config(item) for item in value)
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def device_information(device: DeviceLike) -> dict[str, Any]:
    """Return auditable backend information without initializing distributed mode."""

    resolved = resolve_device(device)
    return {
        "device": str(resolved),
        "type": resolved.type,
        "index": resolved.index,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        ),
        "mps_available": _mps_is_available(),
        "pytorch_enable_mps_fallback": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ),
    }


def capture_rng_state(device: DeviceLike = "cpu") -> dict[str, Any]:
    """Capture Python, NumPy, CPU PyTorch, and active accelerator RNG state."""

    resolved = resolve_device(device)
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
        "mps": None,
    }
    if resolved.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif (
        resolved.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(
    state: Mapping[str, Any], *, device: DeviceLike = "cpu"
) -> None:
    """Restore saved RNG state for exact same-backend continuation."""

    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(torch.as_tensor(state["torch"], device="cpu"))

    resolved = resolve_device(device)
    if (
        resolved.type == "cuda"
        and state.get("cuda") is not None
        and torch.cuda.is_available()
    ):
        cuda_states = [
            torch.as_tensor(item, device="cpu") for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    elif (
        resolved.type == "mps"
        and state.get("mps") is not None
        and _mps_is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(
            torch.as_tensor(state["mps"], device="cpu")
        )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None

        # Best effort: make the directory entry durable as well as atomically visible.
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    epoch: int,
    best_metrics: Mapping[str, Any] | float | None = None,
    best: Mapping[str, Any] | float | None = None,
    config: Any = None,
    seed: int = 42,
    device: DeviceLike | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a full training state.

    ``epoch`` is the epoch that has just completed.  Consequently the returned
    checkpoint resumes at ``next_epoch == epoch + 1``.  Scheduler state is
    captured after that completed epoch, so its counters and learning rate
    continue without an off-by-one step.
    """

    if best_metrics is not None and best is not None:
        raise ValueError("Specify only one of best_metrics or best.")
    selected_best = best if best_metrics is None else best_metrics
    resolved_device = resolve_device(
        _model_device(_unwrapped_model(model)) if device is None else device
    )
    rng_state = capture_rng_state(resolved_device)
    config_value = _plain_config({} if config is None else config)
    best_value = _plain_config({} if selected_best is None else selected_best)
    info = device_information(resolved_device)

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": _unwrapped_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "best": best_value,
        "best_metrics": best_value,
        "config": config_value,
        "seed": int(seed),
        "rng_state": rng_state,
        # Named RNG entries keep the on-disk contract self-evident.
        "python_rng_state": rng_state["python"],
        "numpy_rng_state": rng_state["numpy"],
        "torch_rng_state": rng_state["torch"],
        "device": str(resolved_device),
        "device_info": info,
    }
    if extra is not None:
        overlap = set(payload).intersection(extra)
        if overlap:
            raise ValueError(
                f"extra cannot overwrite checkpoint fields: {sorted(overlap)}"
            )
        payload.update(dict(extra))

    destination = Path(path)
    _atomic_torch_save(payload, destination)
    return destination


def _trusted_torch_load(
    path: str | os.PathLike[str], *, map_location: str | torch.device = "cpu"
) -> Any:
    # Full resumable state includes Python and NumPy RNG objects, so PyTorch's
    # tensor-only unpickler is intentionally unsuitable.  Training checkpoints
    # must therefore come from a trusted local run.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions predating ``weights_only``.
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    device: DeviceLike | None = None,
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load model/training/RNG state and return a payload with ``next_epoch``.

    Tensor storage is first mapped to CPU, avoiding an accelerator-memory spike.
    ``load_state_dict`` copies model tensors to the model's current device.
    """

    destination_device = resolve_device(
        _model_device(_unwrapped_model(model)) if device is None else device
    )
    payload = _trusted_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")

    if "model" not in payload:
        # Accept an upstream weight-only state dict for evaluation, but make its
        # non-resumable status explicit to callers.
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            _unwrapped_model(model).load_state_dict(payload, strict=strict)
            return {
                "model": dict(payload),
                "epoch": -1,
                "next_epoch": 0,
                "best": {},
                "best_metrics": {},
                "config": {},
                "seed": None,
                "device": str(destination_device),
                "legacy_weight_only": True,
            }
        raise ValueError("Checkpoint is missing required 'model' state.")

    _unwrapped_model(model).load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])

    rng_state = payload.get("rng_state")
    if rng_state is None:
        rng_state = {
            "python": payload.get("python_rng_state"),
            "numpy": payload.get("numpy_rng_state"),
            "torch": payload.get("torch_rng_state"),
            "cuda": payload.get("cuda_rng_state"),
            "mps": payload.get("mps_rng_state"),
        }
    if restore_rng:
        restore_rng_state(rng_state, device=destination_device)

    loaded = dict(payload)
    loaded["next_epoch"] = int(payload.get("next_epoch", int(payload["epoch"]) + 1))
    loaded.setdefault("best", payload.get("best_metrics", {}))
    loaded.setdefault("best_metrics", payload.get("best", {}))
    return loaded


# Training-oriented aliases.
save_training_checkpoint = save_checkpoint
load_training_checkpoint = load_checkpoint


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "capture_rng_state",
    "device_information",
    "load_checkpoint",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "save_training_checkpoint",
]
