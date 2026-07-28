from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import atomic_write_json, sha256_file, stable_sample_id


def sequence_from_image_filename(image_filename: str) -> str:
    return str(image_filename).split(",", 1)[0]


def _content_hashes(rgb_path: Path, depth_path: Path) -> dict[str, str]:
    rgb_sha256 = sha256_file(rgb_path)
    depth_sha256 = sha256_file(depth_path)
    digest = hashlib.sha256()
    digest.update(rgb_sha256.encode())
    digest.update(b":")
    digest.update(depth_sha256.encode())
    return {
        "rgb_sha256": rgb_sha256,
        "depth_sha256": depth_sha256,
        "rgbd_content_sha256": digest.hexdigest(),
    }


def read_official_split(
    dataset_root: str | Path,
    split: str,
    *,
    version: str = "multiple",
    hash_content: bool = True,
) -> list[dict[str, Any]]:
    root = Path(dataset_root).resolve()
    source = root / "refer" / version / f"{split}_expressions.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    frame_hashes: dict[str, dict[str, str]] = {}
    for item in payload["data"]:
        local_id = int(item["question_index"])
        frame_id = str(item["image_filename"])
        sequence_id, image_name = frame_id.split(",", 1)
        if hash_content and frame_id not in frame_hashes:
            frame_hashes[frame_id] = _content_hashes(
                root / sequence_id / "rgb" / image_name,
                root / sequence_id / "depth" / image_name,
            )
        hashes = frame_hashes.get(frame_id, {})
        rows.append(
            {
                "sample_id": stable_sample_id(split, local_id),
                "source_sample_id": local_id,
                "official_split": split,
                "frame_id": frame_id,
                "scene_id": frame_id,
                "sequence_id": sequence_id,
                "rgb_sha256": hashes.get("rgb_sha256"),
                "depth_sha256": hashes.get("depth_sha256"),
                "rgbd_content_sha256": hashes.get(
                    "rgbd_content_sha256"
                ),
            }
        )
    return rows


def grouped_partition(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    heldout_fraction: float,
    seed: int,
    fit_name: str,
    heldout_name: str,
) -> list[dict[str, Any]]:
    groups = sorted({str(row[group_key]) for row in rows})
    rng = random.Random(int(seed))
    rng.shuffle(groups)
    heldout_count = max(1, round(len(groups) * float(heldout_fraction)))
    heldout = set(groups[:heldout_count])
    result = []
    for row in rows:
        result.append(
            {
                **row,
                "development_partition": (
                    heldout_name if str(row[group_key]) in heldout else fit_name
                ),
            }
        )
    return result


def assign_group_folds(
    rows: list[dict[str, Any]],
    *,
    group_key: str = "sequence_id",
    folds: int = 3,
    seed: int = 17,
) -> list[dict[str, Any]]:
    if folds < 2:
        raise ValueError("OOF requires at least two folds")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row[group_key])].append(row)
    groups = sorted(by_group)
    rng = random.Random(int(seed))
    rng.shuffle(groups)
    # Greedy balancing by expression count while keeping a group intact.
    groups.sort(key=lambda key: len(by_group[key]), reverse=True)
    fold_sizes = [0] * folds
    group_fold = {}
    for group in groups:
        fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
        group_fold[group] = fold
        fold_sizes[fold] += len(by_group[group])
    return [{**row, "oof_fold": group_fold[str(row[group_key])]} for row in rows]


def _pairwise_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]], key: str
) -> int:
    left_values = {
        row.get(key) for row in left if row.get(key) is not None
    }
    right_values = {
        row.get(key) for row in right if row.get(key) is not None
    }
    return len(left_values & right_values)


def audit_partitions(
    partitions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    keys = (
        "sample_id",
        "frame_id",
        "scene_id",
        "rgb_sha256",
        "depth_sha256",
        "rgbd_content_sha256",
    )
    pairs = {}
    names = sorted(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = partitions[left_name], partitions[right_name]
            pairs[f"{left_name}__{right_name}"] = {
                f"{key}_overlap": _pairwise_overlap(left, right, key)
                for key in keys
            }
    return {
        "partitions": {
            name: {
                "expressions": len(rows),
                "frames": len({row["frame_id"] for row in rows}),
                "scenes": len({row["scene_id"] for row in rows}),
                "sequences": len({row["sequence_id"] for row in rows}),
                "content_hashes": len(
                    {
                        row["rgbd_content_sha256"]
                        for row in rows
                        if row["rgbd_content_sha256"] is not None
                    }
                ),
                "rgb_hashes": len(
                    {
                        row["rgb_sha256"]
                        for row in rows
                        if row["rgb_sha256"] is not None
                    }
                ),
                "depth_hashes": len(
                    {
                        row["depth_sha256"]
                        for row in rows
                        if row["depth_sha256"] is not None
                    }
                ),
            }
            for name, rows in sorted(partitions.items())
        },
        "pairwise_overlap": pairs,
        "required_zero_overlap_passed": all(
            count == 0
            for values in pairs.values()
            for name, count in values.items()
            if not name.startswith("sequence")
        ),
        "sequence_overlap_is_reported_not_required_for_official_splits": {
            f"{left}__{right}": _pairwise_overlap(
                partitions[left], partitions[right], "sequence_id"
            )
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        },
    }


def build_split_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    calibration_fraction: float = 0.15,
    calibration_seed: int = 1701,
    oof_folds: int = 3,
    oof_seed: int = 1702,
    hash_content: bool = True,
) -> dict[str, Any]:
    train = read_official_split(
        dataset_root, "train", hash_content=hash_content
    )
    val = read_official_split(dataset_root, "val", hash_content=hash_content)
    test = read_official_split(dataset_root, "test", hash_content=hash_content)
    train = grouped_partition(
        train,
        group_key="sequence_id",
        heldout_fraction=calibration_fraction,
        seed=calibration_seed,
        fit_name="train",
        heldout_name="calibration",
    )
    train_fit = [row for row in train if row["development_partition"] == "train"]
    calibration = [
        row for row in train if row["development_partition"] == "calibration"
    ]
    train_fit = assign_group_folds(
        train_fit, folds=oof_folds, seed=oof_seed, group_key="sequence_id"
    )
    validation = [
        {**row, "development_partition": "validation", "oof_fold": None}
        for row in val
    ]
    test_rows = [
        {**row, "development_partition": "test", "oof_fold": None}
        for row in test
    ]
    calibration = [{**row, "oof_fold": None} for row in calibration]
    partitions = {
        "train": train_fit,
        "calibration": calibration,
        "validation": validation,
        "test": test_rows,
    }
    audit = audit_partitions(partitions)
    if not audit["required_zero_overlap_passed"]:
        raise AssertionError("split audit failed required zero-overlap checks")
    rows = train_fit + calibration + validation + test_rows
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise AssertionError("stable sample_id is not globally unique")
    payload = {
        "schema_version": "2.0.0",
        "dataset_root": str(Path(dataset_root).resolve()),
        "version": "multiple",
        "group_policy": {
            "official_split_unit": "RGB-D frame (official OCID-VLG scene)",
            "internal_calibration_group": "capture sequence",
            "oof_group": "capture sequence",
            "calibration_fraction": float(calibration_fraction),
            "calibration_seed": int(calibration_seed),
            "oof_folds": int(oof_folds),
            "oof_seed": int(oof_seed),
            "content_hashing_enabled": bool(hash_content),
        },
        "audit": audit,
        "rows": rows,
    }
    atomic_write_json(output_path, payload)
    return payload
