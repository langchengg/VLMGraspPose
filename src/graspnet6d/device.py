"""Conservative CPU/MPS selection and reproducible VGN device benchmarks.

The formal experiment must not infer that MPS is usable merely because PyTorch
reports an available backend.  :func:`benchmark_devices` executes the same
network and the same TSDF tensors on CPU and MPS, records numerical agreement
and timing, and :func:`choose_formal_device` applies the experiment's explicit
CPU-first decision rule.
"""

from __future__ import annotations

import os
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np


SUPPORTED_DEVICES = frozenset({"auto", "cpu", "mps"})


@dataclass(frozen=True)
class DeviceResolution:
    """A requested device and the device that is safe to execute now."""

    requested: str
    resolved: str
    mps_built: bool
    mps_available: bool
    reason: str


@dataclass(frozen=True)
class DeviceBenchmark:
    """One device's aggregate benchmark measurements."""

    device: str
    available: bool
    successful: bool
    sample_count: int
    finite_rate: float | None
    max_absolute_error: float | None
    mean_absolute_error: float | None
    median_latency_s: float | None
    p95_latency_s: float | None
    peak_memory_bytes: int | None
    fallback_environment_enabled: bool
    error: str | None = None
    fallback_detected: bool = False
    fallback_details: str | None = None

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceDecision:
    """The formal inference-device decision and its auditable reason."""

    device: str
    reason: str
    parity_passed: bool
    speed_passed: bool


def _mps_capability() -> tuple[bool, bool]:
    import torch

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False, False
    return bool(backend.is_built()), bool(backend.is_available())


def resolve_device_info(requested: str = "auto") -> DeviceResolution:
    """Resolve ``auto|cpu|mps`` without silently opting into MPS.

    ``auto`` deliberately resolves to CPU.  The caller may select MPS only
    after a same-checkpoint, same-input benchmark passes both parity and speed
    checks.  An explicit unavailable ``mps`` request fails rather than falling
    back invisibly.
    """

    name = str(requested).strip().lower()
    if name not in SUPPORTED_DEVICES:
        allowed = ", ".join(sorted(SUPPORTED_DEVICES))
        raise ValueError(f"device must be one of {allowed}; got {requested!r}")
    built, available = _mps_capability()
    if name == "cpu":
        return DeviceResolution(name, "cpu", built, available, "CPU explicitly requested")
    if name == "auto":
        return DeviceResolution(
            name,
            "cpu",
            built,
            available,
            "auto is CPU-first until a recorded CPU/MPS parity-and-speed benchmark passes",
        )
    if not built or not available:
        raise RuntimeError(
            "MPS was explicitly requested but torch.backends.mps is not both built "
            "and available"
        )
    return DeviceResolution(name, "mps", built, available, "MPS explicitly requested")


def resolve_device(requested: str = "auto") -> Any:
    """Return a :class:`torch.device` under the conservative experiment policy."""

    import torch

    return torch.device(resolve_device_info(requested).resolved)


def _synchronize(device: str) -> None:
    if device != "mps":
        return
    import torch

    synchronize = getattr(torch.mps, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _peak_memory(device: str) -> int | None:
    if device != "mps":
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except (ImportError, OSError):
            return None
    import torch

    value = getattr(torch.mps, "driver_allocated_memory", None)
    return int(value()) if value is not None else None


def _normalise_outputs(outputs: Any) -> tuple[np.ndarray, ...]:
    import torch

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError("VGN benchmark model must return quality, rotation, and width")
    arrays: list[np.ndarray] = []
    for output in outputs:
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        arrays.append(np.asarray(output))
    return tuple(arrays)


def _run_samples(
    model: Any,
    tensors: Sequence[np.ndarray],
    device: str,
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[tuple[np.ndarray, ...]], list[float], int | None, tuple[str, ...]]:
    import torch

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def run_one(array: np.ndarray) -> tuple[np.ndarray, ...]:
        value = torch.from_numpy(array).unsqueeze(0).to(torch_device)
        with torch.inference_mode():
            outputs = model(value)
        _synchronize(device)
        return _normalise_outputs(outputs)

    observed_peak = _peak_memory(device)
    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always")
        for _ in range(max(0, int(warmup))):
            for tensor in tensors:
                run_one(tensor)
                current = _peak_memory(device)
                if current is not None:
                    observed_peak = current if observed_peak is None else max(observed_peak, current)

        outputs: list[tuple[np.ndarray, ...]] = []
        latencies: list[float] = []
        for repeat in range(max(1, int(repeats))):
            for tensor in tensors:
                started = time.perf_counter()
                result = run_one(tensor)
                latencies.append(time.perf_counter() - started)
                if repeat == 0:
                    outputs.append(result)
                current = _peak_memory(device)
                if current is not None:
                    observed_peak = current if observed_peak is None else max(observed_peak, current)
    warning_messages = tuple(str(item.message) for item in observed_warnings)
    return outputs, latencies, observed_peak, warning_messages


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def _failure_record(device: str, sample_count: int, error: BaseException) -> DeviceBenchmark:
    built, available = _mps_capability()
    is_available = device == "cpu" or (built and available)
    return DeviceBenchmark(
        device=device,
        available=is_available,
        successful=False,
        sample_count=sample_count,
        finite_rate=None,
        max_absolute_error=None,
        mean_absolute_error=None,
        median_latency_s=None,
        p95_latency_s=None,
        peak_memory_bytes=None,
        fallback_environment_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        error=f"{type(error).__name__}: {error}",
    )


def benchmark_devices(
    model_factory: Callable[[str], Any],
    tsdf_tensors: Iterable[np.ndarray],
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> tuple[DeviceBenchmark, DeviceBenchmark]:
    """Benchmark CPU and MPS with the same model weights and TSDF tensors.

    ``model_factory(device)`` must construct a fresh model loaded from the same
    immutable checkpoint for each device.  Each TSDF tensor must use the
    official VGN shape ``(1, 40, 40, 40)``.  MPS failures are recorded; this
    function never enables or relies on automatic CPU fallback.
    """

    tensors = [np.ascontiguousarray(value, dtype=np.float32) for value in tsdf_tensors]
    if not tensors:
        raise ValueError("at least one TSDF tensor is required for a device benchmark")
    for tensor in tensors:
        if tensor.shape != (1, 40, 40, 40):
            raise ValueError(f"expected TSDF shape (1, 40, 40, 40), got {tensor.shape}")
        if not np.all(np.isfinite(tensor)):
            raise ValueError("benchmark TSDF contains NaN or Inf")

    try:
        cpu_outputs, cpu_times, cpu_memory, cpu_warnings = _run_samples(
            model_factory("cpu"), tensors, "cpu", warmup=warmup, repeats=repeats
        )
        cpu_finite = np.concatenate(
            [array.reshape(-1) for sample in cpu_outputs for array in sample]
        )
        cpu = DeviceBenchmark(
            device="cpu",
            available=True,
            successful=True,
            sample_count=len(tensors),
            finite_rate=float(np.mean(np.isfinite(cpu_finite))),
            max_absolute_error=0.0,
            mean_absolute_error=0.0,
            median_latency_s=float(statistics.median(cpu_times)),
            p95_latency_s=_percentile(cpu_times, 95),
            peak_memory_bytes=cpu_memory,
            fallback_environment_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            fallback_detected=any("fallback" in item.lower() or "fall back" in item.lower() for item in cpu_warnings),
            fallback_details="; ".join(cpu_warnings) or None,
        )
    except Exception as error:
        cpu = _failure_record("cpu", len(tensors), error)
        return cpu, _failure_record(
            "mps", len(tensors), RuntimeError("CPU reference failed; MPS comparison not run")
        )

    built, available = _mps_capability()
    if not built or not available:
        return cpu, _failure_record(
            "mps", len(tensors), RuntimeError("MPS backend is unavailable")
        )
    try:
        mps_outputs, mps_times, mps_memory, mps_warnings = _run_samples(
            model_factory("mps"), tensors, "mps", warmup=warmup, repeats=repeats
        )
        differences: list[np.ndarray] = []
        flat_outputs: list[np.ndarray] = []
        for reference, observed in zip(cpu_outputs, mps_outputs):
            for cpu_array, mps_array in zip(reference, observed):
                if cpu_array.shape != mps_array.shape:
                    raise ValueError(
                        f"CPU/MPS output shape mismatch: {cpu_array.shape} vs {mps_array.shape}"
                    )
                differences.append(np.abs(cpu_array.astype(np.float64) - mps_array))
                flat_outputs.append(mps_array.reshape(-1))
        difference = np.concatenate([value.reshape(-1) for value in differences])
        output = np.concatenate(flat_outputs)
        mps = DeviceBenchmark(
            device="mps",
            available=True,
            successful=True,
            sample_count=len(tensors),
            finite_rate=float(np.mean(np.isfinite(output))),
            max_absolute_error=float(np.nanmax(difference)),
            mean_absolute_error=float(np.nanmean(difference)),
            median_latency_s=float(statistics.median(mps_times)),
            p95_latency_s=_percentile(mps_times, 95),
            peak_memory_bytes=mps_memory,
            fallback_environment_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            fallback_detected=any("fallback" in item.lower() or "fall back" in item.lower() for item in mps_warnings),
            fallback_details="; ".join(mps_warnings) or None,
        )
    except Exception as error:
        mps = _failure_record("mps", len(tensors), error)
    return cpu, mps


def choose_formal_device(
    cpu: DeviceBenchmark,
    mps: DeviceBenchmark,
    *,
    max_absolute_error: float = 1e-4,
    max_mean_absolute_error: float = 1e-5,
    minimum_speedup: float = 1.05,
) -> DeviceDecision:
    """Apply the recorded parity, finite-output, fallback, and speed rules."""

    if not cpu.successful or cpu.finite_rate != 1.0:
        raise RuntimeError("a successful finite CPU reference is required")
    parity = bool(
        mps.successful
        and mps.finite_rate == 1.0
        and mps.max_absolute_error is not None
        and mps.mean_absolute_error is not None
        and mps.max_absolute_error <= max_absolute_error
        and mps.mean_absolute_error <= max_mean_absolute_error
        and not mps.fallback_environment_enabled
        and not mps.fallback_detected
    )
    speed = bool(
        parity
        and cpu.median_latency_s is not None
        and mps.median_latency_s is not None
        and mps.median_latency_s > 0
        and cpu.median_latency_s / mps.median_latency_s >= minimum_speedup
    )
    if parity and speed:
        return DeviceDecision(
            "mps",
            "MPS passed finite-output/numerical parity checks and exceeded the required speedup",
            True,
            True,
        )
    reasons: list[str] = []
    if not mps.available:
        reasons.append("MPS unavailable")
    elif not mps.successful:
        reasons.append(f"MPS failed: {mps.error}")
    elif mps.fallback_environment_enabled:
        reasons.append("PYTORCH_ENABLE_MPS_FALLBACK=1 prevents a clean device attribution")
    elif mps.fallback_detected:
        reasons.append(f"MPS execution emitted an operation fallback: {mps.fallback_details}")
    elif not parity:
        reasons.append("MPS numerical or finite-output parity failed")
    elif not speed:
        reasons.append("MPS did not meet the required measured speedup")
    return DeviceDecision("cpu", "; ".join(reasons), parity, speed)
