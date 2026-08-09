"""Content-addressed cache contracts for common feature extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .hashing import canonical_sha256, sha256_file


def common_asset_records(
    rows: list[dict[str, Any]], cache: dict[str, str]
) -> list[dict[str, Any]]:
    columns = (
        ("source_rgb_path", "source_rgb_sha256"),
        ("source_depth_path", "source_depth_sha256"),
        ("predicted_mask_path", "predicted_mask_sha256"),
        ("predicted_probability_path", "predicted_probability_sha256"),
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        sample: dict[str, Any] = {
            "sample_id": str(row["sample_id"]),
            "language_sha256": str(row["language_sha256"]),
            "assets": {},
        }
        observed_language = hashlib.sha256(str(row["language"]).encode("utf-8")).hexdigest()
        if observed_language != sample["language_sha256"]:
            raise RuntimeError(f"paired manifest language hash drift: {row['sample_id']}")
        for path_column, hash_column in columns:
            path = str(Path(str(row[path_column])).resolve())
            observed = cache.get(path)
            if observed is None:
                observed = sha256_file(path)
                cache[path] = observed
            if observed != str(row[hash_column]):
                raise RuntimeError(
                    f"paired manifest asset hash drift ({path_column}): {row['sample_id']}"
                )
            sample["assets"][path_column] = {"path": path, "sha256": observed}
        records.append(sample)
    return records


def valid_common_shard_manifest(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("status") != "COMPLETE" or any(
        value.get(key) != item for key, item in expected.items()
    ):
        return False
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    key_columns = {
        "candidate_features": ["sample_id", "candidate_id"],
        "candidate_relations": [
            "sample_id",
            "source_candidate_id",
            "target_candidate_id",
        ],
        "sample_context": ["sample_id"],
    }
    for kind, columns in key_columns.items():
        record = artifacts.get(kind, {})
        artifact_path = path.parent / f"{kind}.parquet"
        if not artifact_path.is_file() or record.get("sha256") != sha256_file(artifact_path):
            return False
        frame = pd.read_parquet(artifact_path, columns=columns)
        keys = frame[columns].astype(str).sort_values(columns).to_dict("records")
        if record.get("key_sha256") != canonical_sha256(keys) or record.get("rows") != len(frame):
            return False
    return True
