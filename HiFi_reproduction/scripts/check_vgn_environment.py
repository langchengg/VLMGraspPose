"""Minimal offline VGN environment and real-checkpoint smoke check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.grasping.vgn_adapter import (
    OFFICIAL_TSDF_INPUT_SHAPE,
    load_official_network,
    predict_official,
    resolve_device_info,
    runtime_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgn-root", type=Path, default=Path("third_party/vgn"))
    parser.add_argument(
        "--vgn-weights", type=Path, default=Path("third_party/vgn/data/models/vgn_conv.pth")
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    selection = resolve_device_info(args.device)
    metadata = runtime_metadata(vgn_root=args.vgn_root, weights_path=args.vgn_weights)
    network = load_official_network(
        args.vgn_weights, device=selection.resolved, vgn_root=args.vgn_root
    )
    result = predict_official(
        np.zeros(OFFICIAL_TSDF_INPUT_SHAPE, dtype=np.float32),
        network,
        selection.resolved,
    )
    metadata.update(
        requested_device=args.device,
        resolved_device=selection.resolved,
        device_fallback_reason=selection.fallback_reason,
        network_forward_used_device=result.used_device,
        input_shape=list(OFFICIAL_TSDF_INPUT_SHAPE),
        output_shapes={
            "quality": list(result.qual_vol.shape),
            "orientation": list(result.rot_vol.shape),
            "width": list(result.width_vol.shape),
        },
        status="ok",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
