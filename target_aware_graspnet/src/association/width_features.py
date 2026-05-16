from __future__ import annotations

import numpy as np


def gripper_width_match(candidate, pcr, max_width: float = 0.10) -> float:
    if pcr.target_obb is None:
        return 0.5
    extent = np.asarray(pcr.target_obb.extent, dtype=float)
    if len(extent) == 0:
        return 0.5
    desired = np.clip(np.min(extent) + 0.01, 0.0, max_width)
    return float(np.clip(1.0 - abs(candidate.gripper_width - desired) / max(max_width, 1e-6), 0.0, 1.0))
