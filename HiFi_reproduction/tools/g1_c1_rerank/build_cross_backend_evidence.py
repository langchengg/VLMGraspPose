#!/usr/bin/env python3
"""Build label-free G1/C1 candidate agreement evidence in original coordinates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.g1_c1_safe_rerank.contracts import (  # noqa: E402
    assert_inference_columns,
    sha256_file,
)
from src.grasping.g1_c1_safe_rerank.pools import cross_backend_match  # noqa: E402


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    return parser.parse_args(argv)


def _nearest(source: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    other_by_sample = {
        str(sample_id): group.to_dict(orient="records")
        for sample_id, group in other.groupby("sample_id", sort=False)
    }
    for candidate in source.to_dict(orient="records"):
        alternatives = other_by_sample.get(str(candidate["sample_id"]), [])
        best: dict[str, Any] | None = None
        best_key: tuple[float, str] | None = None
        for alternative in alternatives:
            evidence = cross_backend_match(candidate, alternative)
            key = (
                float(evidence["cross_backend_center_distance"]),
                str(alternative["stable_candidate_id"]),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = {**alternative, **evidence}
        if best is None:
            row = {
                "sample_id": str(candidate["sample_id"]),
                "stable_candidate_id": str(candidate["stable_candidate_id"]),
                "other_backend_candidate_id": "",
                "cross_backend_candidate_available": 0.0,
                "cross_backend_center_distance": math.nan,
                "cross_backend_angle_difference": math.nan,
                "cross_backend_width_ratio": math.nan,
                "cross_backend_nearest_rank": math.nan,
                "cross_backend_nearest_score": math.nan,
                "cross_backend_rotated_iou": math.nan,
                "cross_backend_nms_match": 0.0,
                "cross_backend_agreement_score": 0.0,
            }
        else:
            width_ratio = min(float(candidate["width_px"]), float(best["width_px"])) / max(
                float(candidate["width_px"]), float(best["width_px"]), 1e-9
            )
            distance = float(best["cross_backend_center_distance"])
            angle = float(best["cross_backend_angle_difference"])
            iou = float(best["cross_backend_rotated_iou"])
            agreement = (
                math.exp(-distance / 50.0)
                * max(math.cos(math.radians(angle)), 0.0)
                * width_ratio
                * (0.5 + 0.5 * iou)
            )
            row = {
                "sample_id": str(candidate["sample_id"]),
                "stable_candidate_id": str(candidate["stable_candidate_id"]),
                "other_backend_candidate_id": str(best["stable_candidate_id"]),
                "cross_backend_candidate_available": 1.0,
                "cross_backend_center_distance": distance,
                "cross_backend_angle_difference": angle,
                "cross_backend_width_ratio": width_ratio,
                "cross_backend_nearest_rank": float(best["original_rank"]),
                "cross_backend_nearest_score": float(best["original_score"]),
                "cross_backend_rotated_iou": iou,
                "cross_backend_nms_match": float(bool(best["matched"])),
                "cross_backend_agreement_score": float(agreement),
            }
        rows.append(row)
    result = pd.DataFrame(rows)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    run = args.run_dir.expanduser().resolve()
    split = str(args.split)
    paths = {
        backend: run / "data" / f"frozen_{backend}_{split}_candidates.parquet"
        for backend in ("g1", "c1")
    }
    frames = {backend: pd.read_parquet(path) for backend, path in paths.items()}
    assert_inference_columns(frames["g1"].columns)
    assert_inference_columns(frames["c1"].columns)
    evidence = {
        "g1": _nearest(frames["g1"], frames["c1"]),
        "c1": _nearest(frames["c1"], frames["g1"]),
    }
    nearest_pairs = {
        backend: {
            (
                str(row.sample_id),
                str(row.stable_candidate_id),
                str(row.other_backend_candidate_id),
            )
            for row in evidence[backend].itertuples(index=False)
            if str(row.other_backend_candidate_id)
        }
        for backend in ("g1", "c1")
    }
    for backend in ("g1", "c1"):
        evidence[backend]["cross_backend_mutual_nearest"] = [
            float(
                (
                    str(row.sample_id),
                    str(row.other_backend_candidate_id),
                    str(row.stable_candidate_id),
                )
                in nearest_pairs["c1" if backend == "g1" else "g1"]
            )
            for row in evidence[backend].itertuples(index=False)
        ]
        destination = run / "02_features" / split / backend / "cross_backend_evidence.parquet"
        _atomic_parquet(destination, evidence[backend])
    overlap = {
        backend.upper(): {
            "candidate_rows": len(frame),
            "samples_with_candidates": int(frame["sample_id"].nunique()),
            "nearest_available_rate": float(evidence[backend]["cross_backend_candidate_available"].mean()),
            "nms_match_rate": float(evidence[backend]["cross_backend_nms_match"].mean()),
            "mutual_nearest_rate": float(evidence[backend]["cross_backend_mutual_nearest"].mean()),
            "median_nearest_distance_px": float(evidence[backend]["cross_backend_center_distance"].median()),
            "median_nearest_angle_difference_deg": float(evidence[backend]["cross_backend_angle_difference"].median()),
        }
        for backend, frame in frames.items()
    }
    manifest = {
        "status": "COMPLETE",
        "split": split,
        "coordinate_contract": "native original-image pixel coordinates; no 224/300 scaling",
        "nearest_rule": "minimum original-image centre distance, candidate-id tie break",
        "nms_match_rule": "angle<=10 and ((center<=4 and width_diff<=5) or rotated_iou>=0.5)",
        "source_sha256": {key: sha256_file(value) for key, value in paths.items()},
        "overlap": overlap,
    }
    _atomic_json(run / "02_features" / split / "cross_backend_overlap_audit.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
