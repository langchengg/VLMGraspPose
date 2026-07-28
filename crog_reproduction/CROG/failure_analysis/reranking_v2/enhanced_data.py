from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .datasets import JoinedSample, sample_matrix
from .schema import read_jsonl, stable_sample_id


def _index_lookup(enhanced_dir: str | Path):
    root = Path(enhanced_dir)
    return root, {
        str(record["sample_id"]): record
        for record in read_jsonl(root / "index.jsonl")
    }


def load_enhanced_arrays(
    enhanced_dir: str | Path,
    samples: list[JoinedSample],
    *,
    include_crops: bool = True,
    include_labels: bool = True,
) -> dict[str, np.ndarray]:
    root, lookup = _index_lookup(enhanced_dir)
    sample_count = len(samples)
    if not sample_count:
        raise ValueError("no enhanced samples requested")
    first = lookup[samples[0].sample_id]
    with np.load(root / "shards" / first["shard"]) as first_shard:
        crop_shape = first_shard["crops"].shape[2:]
        latent_shape = first_shard["latent_pre"].shape[2:]
    crops = (
        np.empty((sample_count, 5, *crop_shape), dtype=np.float16)
        if include_crops
        else None
    )
    latent_pre = np.empty((sample_count, 5, *latent_shape), dtype=np.float16)
    latent_post = np.empty_like(latent_pre)
    by_shard: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for output_index, sample in enumerate(samples):
        record = lookup.get(sample.sample_id)
        if record is None:
            raise ValueError(f"missing enhanced features for {sample.sample_id}")
        expected_ids = [
            candidate["candidate_id"] for candidate in sample.feature["candidates"]
        ]
        expected_checksums = [
            candidate["candidate_checksum"]
            for candidate in sample.feature["candidates"]
        ]
        if (
            record["candidate_ids"] != expected_ids
            or record["candidate_checksums"] != expected_checksums
        ):
            raise ValueError(f"enhanced candidate join mismatch for {sample.sample_id}")
        by_shard.setdefault(record["shard"], []).append((output_index, record))
    for shard_name, requested in sorted(by_shard.items()):
        with np.load(root / "shards" / shard_name) as shard:
            for output_index, record in requested:
                offset = int(record["offset"])
                if str(shard["sample_ids"][offset]) != str(record["sample_id"]):
                    raise ValueError("enhanced shard offset/sample mismatch")
                if crops is not None:
                    crops[output_index] = shard["crops"][offset]
                latent_pre[output_index] = shard["latent_pre"][offset]
                latent_post[output_index] = shard["latent_post"][offset]
    q = np.asarray(
        [
            [candidate["q_raw"] for candidate in sample.feature["candidates"]]
            for sample in samples
        ],
        dtype=np.float32,
    )
    scalar = np.stack([sample_matrix(sample.feature) for sample in samples])
    result = {
        "latent_pre": latent_pre,
        "latent_post": latent_post,
        "q": q,
        "scalar": scalar,
        "sample_ids": np.asarray([sample.sample_id for sample in samples]),
    }
    if include_labels:
        result["labels"] = np.asarray(
            [
                [
                    float(item["candidate_correct"])
                    for item in sample.label["candidate_labels"]
                ]
                for sample in samples
            ],
            dtype=np.float32,
        )
    if crops is not None:
        result["crops"] = crops
    return result


def flatten_candidate_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sample_count, candidates = arrays["q"].shape
    result = {
        "q": arrays["q"].reshape(-1),
        "labels": arrays["labels"].reshape(-1),
        "sample_index": np.repeat(np.arange(sample_count), candidates),
        "latent_pre": arrays["latent_pre"].reshape(
            sample_count * candidates, -1
        ),
        "latent_post": arrays["latent_post"].reshape(
            sample_count * candidates, -1
        ),
        "scalar": arrays["scalar"].reshape(sample_count * candidates, -1),
    }
    if "crops" in arrays:
        result["crops"] = arrays["crops"].reshape(
            sample_count * candidates, *arrays["crops"].shape[2:]
        )
    return result
