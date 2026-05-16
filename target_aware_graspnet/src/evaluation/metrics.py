from __future__ import annotations

import numpy as np


def is_proxy_valid(feature: dict, thresholds: dict) -> bool:
    return (
        feature.get("target_overlap", 0.0) > thresholds.get("target_overlap", 0.10)
        and feature.get("center_alignment", 0.0) > thresholds.get("center_alignment", 0.30)
        and feature.get("collision_penalty", 1.0) < thresholds.get("collision_penalty", 0.50)
        and feature.get("depth_stability", 0.0) > thresholds.get("depth_stability", 0.30)
        and feature.get("gripper_width_match", 0.0) > thresholds.get("gripper_width_match", 0.30)
    )


def topk_valid_rate(records: list[dict], k: int, thresholds: dict) -> float:
    if not records:
        return 0.0
    ok = 0
    for rec in records:
        candidates = [rec.get("feature_breakdown", {})] + [
            c.get("features", {}) for c in rec.get("top_k_fallback_candidates", [])[: max(k - 1, 0)]
        ]
        ok += any(is_proxy_valid(c, thresholds) for c in candidates[:k])
    return ok / len(records)


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
