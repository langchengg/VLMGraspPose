#!/usr/bin/env python3
import argparse
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Check PyTorch MPS availability")
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()

    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    selected = get_device(prefer_mps=True)
    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"mps_built: {built}")
    print(f"mps_available: {available}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"selected_device: {selected}")

    probe = torch.ones(4, device=selected) * 2
    if selected.type == "mps":
        torch.mps.synchronize()
    print(f"tensor_probe: {probe.cpu().tolist()}")
    if args.require_mps and not available:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
