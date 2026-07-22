#!/usr/bin/env python3
"""Run the pinned official GQ-CNN antipodal sampler on its bundled example."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "gqcnn-official"))

from autolab_core import CameraIntrinsics, YamlConfig  # noqa: E402

from src.grasping.camera_geometry import (  # noqa: E402
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
)
from src.grasping.dexnet_adapter import (  # noqa: E402
    OFFICIAL_GQCNN_COMMIT,
    OFFICIAL_GQCNN_RELEASE,
    gqcnn_runtime,
    make_rgbd_and_segmask,
    sample_antipodal_grasps,
)
from src.grasping.grasp_serialization import save_candidate_bundle  # noqa: E402
from src.grasping.grasp_visualization import (  # noqa: E402
    save_candidate_overlay,
    save_depth_visualization,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "dexnet_official_example",
    )
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    official_root = REPO_ROOT / "third_party" / "gqcnn-official"
    example_root = official_root / "data" / "examples" / "single_object" / "primesense"
    color_path = example_root / "color_0.png"
    depth_path = example_root / "depth_0.npy"
    mask_path = example_root / "segmask_0.png"
    intrinsics_path = official_root / "data" / "calib" / "primesense" / "primesense.intr"
    config_path = REPO_ROOT / "configs" / "dexnet_candidates.yaml"

    rgb = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.uint8)
    depth_m = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32).squeeze()
    mask = np.asarray(Image.open(mask_path)) > 0
    intrinsics = CameraIntrinsics.load(str(intrinsics_path))
    config = dict(YamlConfig(str(config_path))["sampling"])
    rgbd, segmask = make_rgbd_and_segmask(
        rgb, depth_m, mask, frame=intrinsics.frame
    )

    started = time.perf_counter()
    grasps = sample_antipodal_grasps(
        rgbd,
        intrinsics,
        segmask,
        config,
        num_samples=args.num_candidates,
        seed=args.seed,
        visualize=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    candidates = []
    for index, grasp in enumerate(grasps):
        center = np.asarray(grasp.center.data, dtype=np.float64)
        endpoints = np.asarray(grasp.endpoints, dtype=np.float64)
        candidates.append(
            {
                "candidate_id": f"official_{index:04d}",
                "sample_id": "official_primesense_0",
                "query": None,
                "representation": "planar_parallel_jaw_4dof",
                "approach_constraint": "fixed_camera_optical_axis",
                "center_uv": center.tolist(),
                "center_depth_m": float(grasp.depth),
                "angle_rad": float(grasp.angle),
                "width_m": float(grasp.width),
                "width_px": float(grasp.width_px),
                "endpoints_uv": endpoints.tolist(),
                "camera_frame": intrinsics.frame,
                "seed": args.seed,
                "sampler_rank": index + 1,
                "gqcnn_q_value": None,
                "rejection_reason": None,
                T_CAMERA_GRASP_FIXED_APPROACH_KEY: np.asarray(
                    grasp.pose().matrix, dtype=np.float64
                ).tolist(),
            }
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source": "bundled official BerkeleyAutomation/gqcnn example data",
        "release": OFFICIAL_GQCNN_RELEASE,
        "commit": OFFICIAL_GQCNN_COMMIT,
        "inputs": {
            "color": str(color_path),
            "depth": str(depth_path),
            "segmask": str(mask_path),
            "intrinsics": str(intrinsics_path),
        },
        "input_sha256": {
            path.name: _sha256(path)
            for path in (color_path, depth_path, mask_path, intrinsics_path)
        },
        "requested_candidate_count": args.num_candidates,
        "returned_candidate_count": len(candidates),
        "seed": args.seed,
        "generation_time_ms": elapsed_ms,
        "runtime": gqcnn_runtime(),
        "scoring_used": False,
    }
    save_candidate_bundle(
        candidates,
        json_path=output / "candidates.json",
        npz_path=output / "candidates.npz",
        csv_path=output / "candidates.csv",
        metadata=metadata,
    )
    save_candidate_overlay(
        rgb,
        candidates,
        output / "candidates_overlay.png",
        mask=mask,
        title="Official GQ-CNN v1.3.0 example: antipodal candidates",
        show_scores=False,
    )
    save_depth_visualization(
        depth_m,
        output / "depth_candidates_overlay.png",
        candidates=candidates,
        mask=mask,
        title="Official example metric depth + candidates",
    )
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0 if len(candidates) >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
