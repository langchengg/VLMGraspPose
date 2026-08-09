"""Audit and reuse the existing leakage-free grouped five-fold assignment."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .hashing import atomic_json, atomic_text, sha256_file


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


def freeze_grouped_folds(run_dir: Path, source_assignments: Path) -> dict[str, Any]:
    train = pd.read_parquet(
        run_dir / "01_manifests" / "paired_train.parquet",
        columns=["sample_id", "scene_id", "rgbd_pair_sha256"],
    )
    folds = pd.read_parquet(source_assignments)
    required = {"sample_id", "scene_id", "rgbd_pair_sha256", "fold"}
    if set(folds.columns) != required:
        raise ValueError(f"unexpected fold schema: {folds.columns.tolist()}")
    joined = train.merge(folds, on="sample_id", how="outer", validate="one_to_one", indicator=True, suffixes=("_train", "_fold"))
    if not (joined["_merge"] == "both").all():
        raise ValueError("fold assignment does not cover the exact Train universe")
    scene_mismatch = int((joined["scene_id_train"] != joined["scene_id_fold"]).sum())
    frame_mismatch = int((joined["rgbd_pair_sha256_train"] != joined["rgbd_pair_sha256_fold"]).sum())
    if scene_mismatch or frame_mismatch:
        raise ValueError(f"fold identity mismatch: scene={scene_mismatch}, frame={frame_mismatch}")
    if sorted(joined["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("fold assignment must contain exactly folds 0..4")
    scene_cross = int(joined.groupby("scene_id_train")["fold"].nunique().gt(1).sum())
    frame_cross = int(joined.groupby("rgbd_pair_sha256_train")["fold"].nunique().gt(1).sum())
    if scene_cross or frame_cross:
        raise ValueError(f"group leakage: scenes={scene_cross}, frames={frame_cross}")
    destination = run_dir / "04_splits" / "fold_assignments.parquet"
    _atomic_copy(source_assignments, destination)
    fold_counts = {str(int(key)): int(value) for key, value in folds["fold"].value_counts().sort_index().items()}
    audit = {
        "status": "PASS",
        "source": str(source_assignments.resolve()),
        "source_sha256": sha256_file(source_assignments),
        "destination": str(destination.resolve()),
        "destination_sha256": sha256_file(destination),
        "train_rows": len(train),
        "fold_counts": fold_counts,
        "scene_cross_fold_overlap": scene_cross,
        "frame_cross_fold_overlap": frame_cross,
        "group_rule": "scene_id, which strictly contains repeated RGB-D frame queries in this dataset",
    }
    atomic_json(run_dir / "04_splits" / "split_leakage_audit.json", audit)
    report = f"""# Split leakage audit

Status: **PASS**.

- Exact Train universe: {len(train):,} samples.
- Fold counts: {fold_counts}.
- Scene groups crossing folds: {scene_cross}.
- RGB-D frame hashes crossing folds: {frame_cross}.
- Group rule: scene ID; every repeated language query for a frame remains in one fold.
- Frozen assignment SHA-256: `{audit['destination_sha256']}`.

The assignment is reused byte-for-byte from the previous grouped development run after an
independent identity and leakage cross-check. It does not expose labels or Test outcomes.
"""
    atomic_text(run_dir / "04_splits" / "SPLIT_LEAKAGE_AUDIT.md", report)
    return audit
