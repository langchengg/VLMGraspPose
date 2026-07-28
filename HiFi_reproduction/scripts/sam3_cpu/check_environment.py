#!/usr/bin/env python3
"""Print a secret-free audit of the isolated SAM 3 CPU environment."""

from __future__ import annotations

import json
import os
import platform
import sys

import psutil


def main() -> int:
    result: dict[str, object] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
        "environment_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }
    imports: dict[str, object] = {}
    try:
        import huggingface_hub
        import torch
        import transformers
        from transformers import (
            Sam3Model,
            Sam3Processor,
            Sam3TrackerModel,
            Sam3TrackerProcessor,
        )

        classes = (
            Sam3Model,
            Sam3Processor,
            Sam3TrackerModel,
            Sam3TrackerProcessor,
        )
        imports = {
            "success": True,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "huggingface_hub_version": huggingface_hub.__version__,
            "classes": [item.__name__ for item in classes],
            "cuda_available_but_unused": bool(torch.cuda.is_available()),
            "mps_available_but_unused": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
    except Exception as error:
        imports = {"success": False, "error_type": type(error).__name__}
    result["sam3_imports"] = imports
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if imports["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

