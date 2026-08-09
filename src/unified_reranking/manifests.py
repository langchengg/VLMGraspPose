"""Build and verify the common inference-only split manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .hashing import atomic_json, canonical_sha256, sha256_file


SAFE_SAMPLE_COLUMNS = (
    "schema_version",
    "sample_id",
    "sample_index",
    "scene_id",
    "expression_index",
    "question_index",
    "source_rgb_path",
    "source_rgb_sha256",
    "source_depth_path",
    "source_depth_sha256",
    "rgbd_pair_sha256",
    "language",
    "language_sha256",
    "predicted_mask_path",
    "predicted_mask_sha256",
    "predicted_probability_path",
    "predicted_probability_sha256",
    "intrinsics_path",
    "intrinsics_sha256",
    "intrinsics_status",
    "intrinsics_provenance",
    "split",
    "checkpoint_sha256",
    "config_sha256",
    "inference_contract_sha256",
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _load_safe_samples(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(SAFE_SAMPLE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"sample manifest is missing columns: {missing}: {path}")
    result = frame.loc[:, SAFE_SAMPLE_COLUMNS].copy()
    result.insert(4, "frame_id", result["rgbd_pair_sha256"])
    result.insert(5, "group_id", result["rgbd_pair_sha256"])
    if result["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample_id in {path}")
    if result["sample_index"].duplicated().any():
        raise ValueError(f"duplicate sample_index in {path}")
    if not result["sample_index"].is_monotonic_increasing:
        raise ValueError(f"sample manifest is not in sample-index order: {path}")
    return result


def _verify_test_against_fair(test: pd.DataFrame, fair_path: Path) -> dict[str, Any]:
    fair = pd.read_parquet(
        fair_path,
        columns=[
            "sample_id",
            "scene_id",
            "image_sha256",
            "depth_sha256",
            "rgbd_pair_sha256",
            "expression",
            "expression_normalized_sha256",
            "predicted_hifics_mask_sha256",
            "predicted_hifics_probability_sha256",
        ],
    )
    joined = test.merge(fair, on="sample_id", how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"modular/fair Test sample identities differ: {counts}")
    comparisons = {
        "scene_id": ("scene_id_x", "scene_id_y"),
        "rgb_sha256": ("source_rgb_sha256", "image_sha256"),
        "depth_sha256": ("source_depth_sha256", "depth_sha256"),
        "rgbd_pair_sha256": ("rgbd_pair_sha256_x", "rgbd_pair_sha256_y"),
        "language_sha256": ("language_sha256", "expression_normalized_sha256"),
        "language_text": ("language", "expression"),
        "hifics_mask_sha256": ("predicted_mask_sha256", "predicted_hifics_mask_sha256"),
        "hifics_probability_sha256": (
            "predicted_probability_sha256",
            "predicted_hifics_probability_sha256",
        ),
    }
    mismatches = {
        name: int((joined[left].astype(str) != joined[right].astype(str)).sum())
        for name, (left, right) in comparisons.items()
    }
    # The fair manifest hashes the normalised expression; the modular source
    # hashes the exact expression.  Record, but do not treat, that representation
    # difference as an identity failure when the text itself agrees downstream.
    hard_mismatches = {k: v for k, v in mismatches.items() if k != "language_sha256" and v}
    if hard_mismatches:
        raise ValueError(f"modular/fair Test input mismatch: {hard_mismatches}")
    return {
        "paired_rows": len(joined),
        "identity_match": True,
        "field_mismatch_counts": mismatches,
        "note": "language hashes use different source normalisation contracts; exact text is audited separately",
    }


def build_paired_manifests(run_dir: Path, modular_run: Path, fair_run: Path) -> dict[str, Any]:
    output: dict[str, Any] = {"splits": {}}
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        source = modular_run / "manifests" / f"{split}_samples.parquet"
        frame = _load_safe_samples(source)
        source_token = "val" if split == "validation" else split
        if set(frame["split"].astype(str)) != {source_token}:
            raise ValueError(f"wrong split marker in {source}")
        frame["split"] = split
        destination = run_dir / "01_manifests" / f"paired_{split}.parquet"
        _atomic_parquet(frame, destination)
        frames[split] = frame
        output["splits"][split] = {
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "rows": len(frame),
            "unique_frames": int(frame["frame_id"].nunique()),
            "unique_scenes": int(frame["scene_id"].nunique()),
            "sample_identity_sha256": canonical_sha256(frame["sample_id"].tolist()),
        }
    output["test_fair_crosscheck"] = _verify_test_against_fair(
        frames["test"], fair_run / "01_manifest" / "paired_manifest.parquet"
    )
    atomic_json(run_dir / "01_manifests" / "paired_manifest_audit.json", output)
    return output
