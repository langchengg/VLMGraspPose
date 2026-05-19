from __future__ import annotations

import numpy as np

from utils.geometry import project_points


def target_overlap(candidate, target, intrinsics: np.ndarray | None) -> float:
    if target.mask is None or intrinsics is None:
        return 0.5 if target.bbox is not None else 0.0
    H, W = target.mask.shape[:2]
    uv = project_points(candidate.position.reshape(1, 3), intrinsics)[0]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < W and 0 <= v < H):
        return 0.0
    center_hit = 1.0 if target.mask[v, u] else 0.0
    radius = max(int(candidate.gripper_width * intrinsics[0, 0] / max(candidate.position[2], 1e-3) * 0.5), 2)
    x1, x2 = max(0, u - radius), min(W, u + radius + 1)
    y1, y2 = max(0, v - radius), min(H, v + radius + 1)
    patch = target.mask[y1:y2, x1:x2]
    patch_score = float(patch.mean()) if patch.size else 0.0
    return float(np.clip(0.5 * center_hit + 0.5 * patch_score, 0.0, 1.0))
