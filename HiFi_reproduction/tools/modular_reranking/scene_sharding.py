"""Shared scene-grouped shard identity used by the Dex-Net streaming pipeline."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


SCENE_SHARD_ASSIGNMENT = (
    "uint256_be(sha256(scene_id)) modulo num_shards"
)


def scene_shard_index(scene_id: str, *, num_shards: int) -> int:
    """Return the exact bucket used by generate_compact_dexnet_candidates.py."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    digest = hashlib.sha256(str(scene_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % int(num_shards)


def assigned_to_scene_shard(
    scene_id: str, *, num_shards: int, shard_index: int
) -> bool:
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")
    return scene_shard_index(scene_id, num_shards=num_shards) == shard_index


def select_scene_shard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    """Select a shard while preserving the input manifest order exactly."""

    return [
        dict(row)
        for row in rows
        if assigned_to_scene_shard(
            str(row["scene_id"]),
            num_shards=num_shards,
            shard_index=shard_index,
        )
    ]
