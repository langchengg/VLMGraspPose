"""Reproducible host/device inventory for the deployment benchmark."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from .common import atomic_json, require_run_dir, verify_preregistration


def _read_command(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def capture_runtime_environment(
    repo: Path, run_dir: Path, *, label: str = "orchestrator"
) -> dict[str, Any]:
    run_dir = require_run_dir(repo, run_dir)
    verify_preregistration(run_dir)
    import numpy as np
    import torch

    mps_available = bool(torch.backends.mps.is_available())
    mps_allocated = None
    mps_driver = None
    if mps_available:
        try:
            mps_allocated = int(torch.mps.current_allocated_memory())
            mps_driver = int(torch.mps.driver_allocated_memory())
        except (AttributeError, RuntimeError):
            pass
    payload = {
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "hardware_model": _read_command(["sysctl", "-n", "hw.model"]),
        "cpu_brand": _read_command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_thread_protocol": 8,
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": mps_available,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_current_allocated_bytes_at_inventory": mps_allocated,
        "mps_driver_allocated_bytes_at_inventory": mps_driver,
        "unified_memory_peak": "not directly measurable",
        "power_configuration": _read_command(["pmset", "-g", "custom"]),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
    }
    if not label.replace("_", "").isalnum():
        raise ValueError("environment label must be alphanumeric/underscore")
    atomic_json(run_dir / f"runtime_full/environment_{label}.json", payload)
    return payload
