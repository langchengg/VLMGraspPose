#!/usr/bin/env python3
"""Create a deterministic scene-sequence development/calibration split.

The official OCID-VLG train split remains the only source.  Entire capture
sequences are assigned together, so no expression, RGB-D frame, RGB asset,
depth asset, or RGB-D pair can cross the development/calibration boundary.
No formal validation or test artifact is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.reranking_v1.identity import (  # noqa: E402
    sha256_file,
    stable_sample_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--frozen-train-manifest", type=Path, required=True)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _asset_paths(ocid_root: Path, scene_id: str) -> tuple[Path, Path]:
    sequence, image_name = scene_id.split(",", 1)
    root = (ocid_root / sequence).resolve()
    rgb = root / "rgb" / image_name
    depth = root / "depth" / image_name
    if not rgb.is_file() or not depth.is_file():
        raise FileNotFoundError(f"missing RGB-D assets for {scene_id}")
    return rgb, depth


def _atomic_path(destination: Path, tmp_root: Path) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root / f"{destination.name}.{uuid.uuid4().hex}.tmp"


def _atomic_json(destination: Path, payload: Any, tmp_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(destination, tmp_root)
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _atomic_parquet(
    destination: Path, frame: pd.DataFrame, tmp_root: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_path(destination, tmp_root)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)


def _sequence_priority(seed: int, sequence_id: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}\0{sequence_id}".encode("utf-8")
    ).hexdigest()


def _intersection_report(
    mapping: pd.DataFrame, column: str
) -> dict[str, Any]:
    development = set(
        mapping.loc[mapping["partition"] == "development", column].astype(str)
    )
    calibration = set(
        mapping.loc[mapping["partition"] == "calibration", column].astype(str)
    )
    overlap = sorted(development & calibration)
    return {
        "column": column,
        "development_unique": len(development),
        "calibration_unique": len(calibration),
        "intersection_count": len(overlap),
        "intersection_preview": overlap[:10],
        "passed": not overlap,
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be in (0, 1)")
    candidate_path = args.per_candidate.expanduser().resolve()
    sample_path = args.per_sample.expanduser().resolve()
    manifest_path = args.frozen_train_manifest.expanduser().resolve()
    ocid_root = args.ocid_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tmp_root = args.tmp_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite split: {output_root}")

    candidates = pd.read_parquet(candidate_path)
    samples = pd.read_parquet(sample_path)
    required_candidates = {
        "sample_id",
        "scene_id",
        "candidate_id",
        "split",
    }
    required_samples = {"sample_id", "scene_id", "split", "candidate_count"}
    if missing := sorted(required_candidates - set(candidates.columns)):
        raise ValueError(f"candidate table missing columns: {missing}")
    if missing := sorted(required_samples - set(samples.columns)):
        raise ValueError(f"sample table missing columns: {missing}")
    if candidates.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError("candidate keys are not unique")
    if samples["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample IDs are not unique")
    if set(candidates["split"].astype(str)) - {"train", "development"}:
        raise ValueError("candidate input is not official train/development")
    if set(samples["split"].astype(str)) - {"train", "development"}:
        raise ValueError("sample input is not official train/development")

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("frozen train manifest must be a non-empty list")
    hash_cache: dict[Path, str] = {}

    def cached_hash(path: Path) -> str:
        value = hash_cache.get(path)
        if value is None:
            value = sha256_file(path)
            hash_cache[path] = value
        return value

    mapping_rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for record in records:
        scene_id = str(record["scene_id"])
        sample_id = stable_sample_id(
            scene_id, int(record["question_index"])
        )
        if sample_id in seen_samples:
            raise ValueError(f"duplicate manifest sample: {sample_id}")
        seen_samples.add(sample_id)
        rgb, depth = _asset_paths(ocid_root, scene_id)
        rgb_sha = cached_hash(rgb)
        depth_sha = cached_hash(depth)
        mapping_rows.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "scene_sequence_id": scene_id.split(",", 1)[0],
                "question_index": int(record["question_index"]),
                "rgb_sha256": rgb_sha,
                "depth_sha256": depth_sha,
                "rgbd_pair_sha256": hashlib.sha256(
                    f"{rgb_sha}\0{depth_sha}".encode("ascii")
                ).hexdigest(),
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    observed_sample_ids = set(samples["sample_id"].astype(str))
    if observed_sample_ids != seen_samples:
        raise ValueError(
            "feature per-sample universe disagrees with frozen train manifest"
        )
    sample_scene = samples.assign(
        sample_id=samples["sample_id"].astype(str),
        scene_id=samples["scene_id"].astype(str),
    ).set_index("sample_id")["scene_id"]
    manifest_scene = mapping.set_index("sample_id")["scene_id"]
    if not sample_scene.sort_index().equals(manifest_scene.sort_index()):
        raise ValueError("feature scene IDs disagree with frozen train manifest")
    observed_candidate_samples = set(candidates["sample_id"].astype(str))
    expected_nonempty_samples = set(
        samples.loc[
            samples["candidate_count"].astype(int) > 0, "sample_id"
        ].astype(str)
    )
    if observed_candidate_samples != expected_nonempty_samples:
        raise ValueError("candidate samples disagree with non-empty universe")

    sequence_sizes = (
        mapping.groupby("scene_sequence_id", sort=False)["sample_id"]
        .nunique()
        .to_dict()
    )
    ordered_sequences = sorted(
        sequence_sizes,
        key=lambda value: (_sequence_priority(args.seed, value), value),
    )
    target = int(
        math.ceil(len(mapping) * float(args.calibration_fraction))
    )
    calibration_sequences: set[str] = set()
    calibration_samples = 0
    for sequence_id in ordered_sequences:
        if calibration_samples >= target and calibration_sequences:
            break
        calibration_sequences.add(sequence_id)
        calibration_samples += int(sequence_sizes[sequence_id])
    if not calibration_sequences or len(calibration_sequences) == len(
        ordered_sequences
    ):
        raise ValueError("deterministic partition left one side empty")
    mapping["partition"] = mapping["scene_sequence_id"].map(
        lambda value: (
            "calibration"
            if value in calibration_sequences
            else "development"
        )
    )
    partition_by_sample = mapping.set_index("sample_id")["partition"]

    candidates = candidates.copy()
    samples = samples.copy()
    candidates["sample_id"] = candidates["sample_id"].astype(str)
    samples["sample_id"] = samples["sample_id"].astype(str)
    candidates["split"] = candidates["sample_id"].map(partition_by_sample)
    samples["split"] = samples["sample_id"].map(partition_by_sample)
    if candidates["split"].isna().any() or samples["split"].isna().any():
        raise ValueError("partition mapping is incomplete")
    development_candidates = candidates.loc[
        candidates["split"] == "development"
    ].copy()
    calibration_candidates = candidates.loc[
        candidates["split"] == "calibration"
    ].copy()
    development_samples = samples.loc[
        samples["split"] == "development"
    ].copy()
    calibration_sample_frame = samples.loc[
        samples["split"] == "calibration"
    ].copy()

    checks = [
        _intersection_report(mapping, column)
        for column in (
            "sample_id",
            "scene_id",
            "scene_sequence_id",
            "rgb_sha256",
            "depth_sha256",
            "rgbd_pair_sha256",
        )
    ]
    if not all(check["passed"] for check in checks):
        raise AssertionError(f"development/calibration leakage: {checks}")

    output_root.mkdir(parents=True)
    outputs = {
        "development_per_candidate": (
            output_root / "development_per_candidate.parquet",
            development_candidates,
        ),
        "calibration_per_candidate": (
            output_root / "calibration_per_candidate.parquet",
            calibration_candidates,
        ),
        "development_per_sample": (
            output_root / "development_per_sample.parquet",
            development_samples,
        ),
        "calibration_per_sample": (
            output_root / "calibration_per_sample.parquet",
            calibration_sample_frame,
        ),
        "partition_mapping": (
            output_root / "partition_mapping.parquet",
            mapping.sort_values("sample_id", kind="mergesort"),
        ),
    }
    for _, (path, frame) in outputs.items():
        _atomic_parquet(path, frame, tmp_root)
    manifest = {
        "schema_version": 1,
        "source_split": "official_unique_train",
        "assignment_unit": "scene_sequence_id",
        "assignment_rule": (
            "ascending sha256(seed NUL scene_sequence_id) until the "
            "calibration sample target is met"
        ),
        "seed": int(args.seed),
        "requested_calibration_fraction": float(
            args.calibration_fraction
        ),
        "frozen_train_manifest": str(manifest_path),
        "frozen_train_manifest_sha256": sha256_file(manifest_path),
        "source_per_candidate": str(candidate_path),
        "source_per_candidate_sha256": sha256_file(candidate_path),
        "source_per_sample": str(sample_path),
        "source_per_sample_sha256": sha256_file(sample_path),
        "counts": {
            "development_samples_all": int(len(development_samples)),
            "development_samples_nonempty": int(
                development_candidates["sample_id"].nunique()
            ),
            "development_candidates": int(len(development_candidates)),
            "development_frames": int(
                development_samples["scene_id"].nunique()
            ),
            "development_sequences": int(
                mapping.loc[
                    mapping["partition"] == "development",
                    "scene_sequence_id",
                ].nunique()
            ),
            "calibration_samples_all": int(
                len(calibration_sample_frame)
            ),
            "calibration_samples_nonempty": int(
                calibration_candidates["sample_id"].nunique()
            ),
            "calibration_candidates": int(len(calibration_candidates)),
            "calibration_frames": int(
                calibration_sample_frame["scene_id"].nunique()
            ),
            "calibration_sequences": int(len(calibration_sequences)),
        },
        "disjointness_checks": checks,
        "formal_validation_consumed": False,
        "formal_test_consumed": False,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
            }
            for name, (path, frame) in outputs.items()
        },
    }
    _atomic_json(output_root / "split_manifest.json", manifest, tmp_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
