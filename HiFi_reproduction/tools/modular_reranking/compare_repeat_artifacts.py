#!/usr/bin/env python3
"""Compare two independently generated compact reranking artifact trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest-a", type=Path, required=True)
    parser.add_argument("--prediction-manifest-b", type=Path, required=True)
    parser.add_argument("--candidate-root-a", type=Path)
    parser.add_argument("--candidate-root-b", type=Path)
    parser.add_argument("--scored-root-a", type=Path)
    parser.add_argument("--scored-root-b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prediction_comparison(
    manifest_a: Path, manifest_b: Path
) -> tuple[list[str], dict[str, Any]]:
    rows_a = read_jsonl(manifest_a)
    rows_b = read_jsonl(manifest_b)
    ids_a = [str(row["sample_id"]) for row in rows_a]
    ids_b = [str(row["sample_id"]) for row in rows_b]
    if ids_a != ids_b:
        raise ValueError("prediction manifests do not contain the same ordered IDs")
    max_abs = 0.0
    mask_mismatch = 0
    exact_probability_arrays = 0
    exact_masks = 0
    for row_a, row_b in zip(rows_a, rows_b):
        probability_a = np.load(
            row_a["probability_path"], allow_pickle=False
        )["probability"]
        probability_b = np.load(
            row_b["probability_path"], allow_pickle=False
        )["probability"]
        if probability_a.shape != probability_b.shape:
            raise ValueError(f"probability shape mismatch: {row_a['sample_id']}")
        difference = np.abs(
            probability_a.astype(np.float64) - probability_b.astype(np.float64)
        )
        current = float(np.max(difference)) if difference.size else 0.0
        max_abs = max(max_abs, current)
        exact_probability_arrays += int(np.array_equal(probability_a, probability_b))
        mask_a = np.asarray(Image.open(row_a["native_mask_path"]).convert("L"))
        mask_b = np.asarray(Image.open(row_b["native_mask_path"]).convert("L"))
        mismatch = int(np.count_nonzero(mask_a != mask_b))
        mask_mismatch += mismatch
        exact_masks += int(mismatch == 0)
    return ids_a, {
        "samples": len(ids_a),
        "probability_max_abs_difference": max_abs,
        "exact_probability_arrays": exact_probability_arrays,
        "native_mask_mismatch_pixels": mask_mismatch,
        "exact_native_masks": exact_masks,
    }


def candidate_comparison(
    sample_ids: list[str], root_a: Path, root_b: Path
) -> dict[str, Any]:
    exact_candidate_sets = 0
    total_candidates = 0
    max_pose_abs = 0.0
    mismatched_ids: list[str] = []
    fields = (
        "center_uv",
        "center_depth_m",
        "center_camera_xyz_m",
        "angle_rad",
        "width_m",
        "width_px",
        "endpoints_uv",
    )
    for sample_id in sample_ids:
        payload_a = json.loads(
            (root_a / sample_id / "candidates.json").read_text(encoding="utf-8")
        )
        payload_b = json.loads(
            (root_b / sample_id / "candidates.json").read_text(encoding="utf-8")
        )
        ids_a = [
            str(candidate["candidate_id"])
            for candidate in payload_a["candidates"]
        ]
        ids_b = [
            str(candidate["candidate_id"])
            for candidate in payload_b["candidates"]
        ]
        with np.load(root_a / sample_id / "candidates.npz", allow_pickle=False) as a:
            with np.load(
                root_b / sample_id / "candidates.npz", allow_pickle=False
            ) as b:
                total_candidates += len(ids_a)
                exact = ids_a == ids_b
                for field in fields:
                    exact = exact and np.array_equal(a[field], b[field])
                transform_delta = np.max(
                    np.abs(
                        a["T_camera_grasp_fixed_approach"].astype(np.float64)
                        - b["T_camera_grasp_fixed_approach"].astype(np.float64)
                    ),
                    initial=0.0,
                )
                max_pose_abs = max(max_pose_abs, float(transform_delta))
                exact = exact and transform_delta == 0.0
                if exact:
                    exact_candidate_sets += 1
                else:
                    mismatched_ids.append(sample_id)
    return {
        "samples": len(sample_ids),
        "total_candidates": total_candidates,
        "exact_candidate_sets": exact_candidate_sets,
        "max_pose_abs_difference": max_pose_abs,
        "mismatched_sample_ids": mismatched_ids,
    }


def scored_comparison(
    sample_ids: list[str], root_a: Path, root_b: Path
) -> dict[str, Any]:
    exact_q_arrays = 0
    max_q_abs = 0.0
    total_candidates = 0
    mismatched_ids: list[str] = []
    for sample_id in sample_ids:
        path_a = root_a / sample_id / "gqcnn_scored_candidates.npz"
        path_b = root_b / sample_id / "gqcnn_scored_candidates.npz"
        with np.load(path_a, allow_pickle=False) as a:
            with np.load(path_b, allow_pickle=False) as b:
                ids_a = [str(value) for value in a["candidate_id"]]
                ids_b = [str(value) for value in b["candidate_id"]]
                q_a = a["gqcnn_q_value"]
                q_b = b["gqcnn_q_value"]
                total_candidates += len(q_a)
                delta = float(
                    np.max(np.abs(q_a.astype(np.float64) - q_b.astype(np.float64)))
                )
                max_q_abs = max(max_q_abs, delta)
                exact = ids_a == ids_b and np.array_equal(q_a, q_b)
                exact_q_arrays += int(exact)
                if not exact:
                    mismatched_ids.append(sample_id)
    return {
        "samples": len(sample_ids),
        "total_candidates": total_candidates,
        "exact_q_arrays": exact_q_arrays,
        "q_max_abs_difference": max_q_abs,
        "mismatched_sample_ids": mismatched_ids,
    }


def main() -> int:
    args = parse_args()
    sample_ids, predictions = prediction_comparison(
        args.prediction_manifest_a.resolve(),
        args.prediction_manifest_b.resolve(),
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "sample_ids_sha256": canonical_sha256(sample_ids),
        "predictions": predictions,
    }
    if (args.candidate_root_a is None) != (args.candidate_root_b is None):
        raise ValueError("both candidate roots are required together")
    if args.candidate_root_a is not None:
        result["candidates"] = candidate_comparison(
            sample_ids,
            args.candidate_root_a.resolve(),
            args.candidate_root_b.resolve(),
        )
    if (args.scored_root_a is None) != (args.scored_root_b is None):
        raise ValueError("both scored roots are required together")
    if args.scored_root_a is not None:
        result["gqcnn_scores"] = scored_comparison(
            sample_ids,
            args.scored_root_a.resolve(),
            args.scored_root_b.resolve(),
        )
    result["deterministic"] = (
        predictions["exact_probability_arrays"] == predictions["samples"]
        and predictions["exact_native_masks"] == predictions["samples"]
        and (
            "candidates" not in result
            or result["candidates"]["exact_candidate_sets"]
            == result["candidates"]["samples"]
        )
        and (
            "gqcnn_scores" not in result
            or result["gqcnn_scores"]["exact_q_arrays"]
            == result["gqcnn_scores"]["samples"]
        )
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
