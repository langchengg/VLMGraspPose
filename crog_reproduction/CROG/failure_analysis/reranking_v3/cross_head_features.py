"""Cross-head evidence is implemented in :mod:`output_map_features`.

This module exposes the stable subset name boundary used by ablations.
"""

from __future__ import annotations


def cross_head_indices(feature_names: list[str]) -> list[int]:
    return [index for index, name in enumerate(feature_names) if name.startswith("g5_")]

