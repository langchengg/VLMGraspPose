"""Scene/RGB-D-disjoint split audit with immutable provenance."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .identity import sha256_file, stable_sample_id


def _resolve_data_path(hifics_root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = hifics_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _pair_hash(rgb_sha256: str, depth_sha256: str) -> str:
    return hashlib.sha256(
        f"{rgb_sha256}\0{depth_sha256}".encode("ascii")
    ).hexdigest()


def _sequence_id(scene_id: str) -> str:
    frame_path, _ = str(scene_id).split(",", 1)
    return str(Path(frame_path).parent)


def _file_hashes(paths: list[Path], workers: int) -> dict[Path, str]:
    unique = sorted(set(paths), key=str)
    if workers <= 0:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(sha256_file, unique))
    return dict(zip(unique, values))


def build_split_audit(
    *,
    manifest_paths: Mapping[str, Path],
    hifics_root: Path,
    provenance: Mapping[str, str],
    workers: int = 8,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Hash every sample asset and fail if any split identity intersects."""

    expected = {"train", "val", "test"}
    if set(manifest_paths) != expected:
        raise ValueError(f"manifest_paths must contain exactly {sorted(expected)}")
    raw_rows: list[dict[str, Any]] = []
    asset_paths: list[Path] = []
    manifest_identities = {}
    for split in ("train", "val", "test"):
        path = Path(manifest_paths[split]).expanduser().resolve()
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{path} is not a list")
        manifest_identities[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
        }
        for position, row in enumerate(rows):
            scene_id = str(row["scene_id"])
            question_index = int(row["question_index"])
            rgb_path = _resolve_data_path(hifics_root, str(row["rgb_path"]))
            depth_path = _resolve_data_path(hifics_root, str(row["depth_path"]))
            asset_paths.extend((rgb_path, depth_path))
            raw_rows.append(
                {
                    "sample_index": int(row.get("num", position)),
                    "sample_id": stable_sample_id(scene_id, question_index),
                    "question_index": question_index,
                    "scene_id": scene_id,
                    "sequence_id": _sequence_id(scene_id),
                    "language": str(row["text"]),
                    "dataset_split": split,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                }
            )
    hashes = _file_hashes(asset_paths, workers)
    rows = []
    for row in raw_rows:
        rgb_sha = hashes[Path(row["rgb_path"])]
        depth_sha = hashes[Path(row["depth_path"])]
        rows.append(
            {
                **row,
                "rgb_sha256": rgb_sha,
                "depth_sha256": depth_sha,
                "rgbd_pair_sha256": _pair_hash(rgb_sha, depth_sha),
                "hifi_checkpoint_sha256": provenance[
                    "hifi_checkpoint_sha256"
                ],
                "dexnet_config_sha256": provenance["dexnet_config_sha256"],
                "gqcnn_model_manifest_sha256": provenance[
                    "gqcnn_model_manifest_sha256"
                ],
            }
        )
    frame = pd.DataFrame(rows)
    duplicate_ids = frame.duplicated(["dataset_split", "sample_id"], keep=False)
    if bool(duplicate_ids.any()):
        examples = frame.loc[
            duplicate_ids, ["dataset_split", "sample_id"]
        ].head()
        raise AssertionError(f"duplicate split-local sample IDs: {examples.to_dict('records')}")

    intersections: dict[str, dict[str, int]] = {}
    keys = (
        "sample_id",
        "scene_id",
        "rgb_sha256",
        "depth_sha256",
        "rgbd_pair_sha256",
    )
    sequence_intersections = {}
    split_sets = {
        split: {
            key: set(frame.loc[frame["dataset_split"] == split, key].astype(str))
            for key in (*keys, "sequence_id")
        }
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pair = f"{left}__{right}"
        intersections[pair] = {
            key: len(split_sets[left][key] & split_sets[right][key]) for key in keys
        }
        sequence_intersections[pair] = len(
            split_sets[left]["sequence_id"] & split_sets[right]["sequence_id"]
        )
    violations = {
        pair: values
        for pair, values in intersections.items()
        if any(values.values())
    }
    summary = {
        "schema_version": 1,
        "title": "OCID-VLG unique split scene/RGB-D leakage audit",
        "sample_counts": {
            split: int((frame["dataset_split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "scene_frame_counts": {
            split: int(
                frame.loc[
                    frame["dataset_split"] == split, "scene_id"
                ].nunique()
            )
            for split in ("train", "val", "test")
        },
        "sequence_counts": {
            split: int(
                frame.loc[
                    frame["dataset_split"] == split, "sequence_id"
                ].nunique()
            )
            for split in ("train", "val", "test")
        },
        "required_intersections": intersections,
        "required_intersections_all_zero": not violations,
        "sequence_path_intersections_diagnostic_only": sequence_intersections,
        "split_semantics": (
            "scene-frame/RGB-D disjoint; sequence paths may recur across splits"
        ),
        "manifests": manifest_identities,
        "provenance": dict(provenance),
    }
    if violations:
        raise AssertionError(f"split leakage detected: {violations}")
    return summary, frame
