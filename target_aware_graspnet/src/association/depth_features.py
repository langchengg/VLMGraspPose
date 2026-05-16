from __future__ import annotations

import numpy as np

from utils.geometry import project_points


def depth_stability(candidate, depth: np.ndarray | None, intrinsics: np.ndarray | None) -> float:
    if depth is None or intrinsics is None:
        return 0.5
    H, W = depth.shape
    uv = project_points(candidate.position.reshape(1, 3), intrinsics)[0]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < W and 0 <= v < H):
        return 0.0
    r = 4
    patch = depth[max(0, v - r):min(H, v + r + 1), max(0, u - r):min(W, u + r + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid) < 4:
        return 0.3
    return float(np.clip(1.0 - np.std(valid) / max(np.mean(valid), 1e-3), 0.0, 1.0))
