#!/usr/bin/env python3
"""Freeze grouped splits and independently reproduce HiFi-CS baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.query_semantics import parse_query  # noqa: E402
from src.segmentation.selective_sam3_vg.evaluation import (  # noqa: E402
    load_frozen_ground_truth_manifest,
    load_ground_truth_mask,
)
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
    load_probability,
    sha256_file,
    stable_json_sha256,
)
from src.segmentation.selective_sam3_vg.metrics import (  # noqa: E402
    binary_mask_metrics,
    summarize_ious,
)


CHECKPOINT_SHA256 = "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
SPLITS = {
    "train": {
        "count": 26295,
        "manifest_sha256": "a986bcce3e1961be816a295c3ae0942e64e61275524a85c0a8957563e7f920c1",
    },
    "val": {
        "count": 3778,
        "manifest_sha256": "573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a",
    },
    "test": {
        "count": 7675,
        "manifest_sha256": "915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409",
    },
}
EXPECTED_TEST = {
    "mean_iou": 0.8074137115168442,
    "p_at_50_numerator": 6997,
    "p_at_60_numerator": 6818,
    "p_at_70_numerator": 6475,
    "p_at_80_numerator": 5655,
    "p_at_90_numerator": 3363,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact-root",
        type=Path,
        default=PROJECT_ROOT
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs",
    )
    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/data_audit/frozen_manifests",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=PROJECT_ROOT.parent / "crog_reproduction/OCID-VLG/refer/unique",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    result = summarize_ious(frame["evaluator_iou_float32"].to_numpy())
    result.update(
        {
            "mean_dice": float(frame["dice"].mean()),
            "mean_mask_precision": float(frame["mask_precision"].mean()),
            "mean_mask_recall": float(frame["mask_recall"].mean()),
            "mean_boundary_fscore": float(frame["boundary_fscore"].mean()),
            "low_precision_count_lt_0_50": int((frame["mask_precision"] < 0.5).sum()),
            "low_recall_count_lt_0_50": int((frame["mask_recall"] < 0.5).sum()),
            "false_positive_area_px": int(frame["false_positive_area_px"].sum()),
            "false_negative_area_px": int(frame["false_negative_area_px"].sum()),
        }
    )
    return result


def hash_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def main() -> int:
    args = parse_args()
    compact_root = args.compact_root.expanduser().resolve()
    frozen_root = args.frozen_root.expanduser().resolve()
    annotations_root = args.annotations_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    baseline_root = output_root / "baseline_reproduction"
    split_root = output_root / "splits"

    all_metrics: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    protected: dict[str, str] = {}
    split_sets: dict[str, dict[str, set[str]]] = {}

    for split, specification in SPLITS.items():
        compact_manifest = compact_root / split / "manifest.jsonl"
        frozen_manifest = frozen_root / f"ocidvlg_unique_{split}.json"
        annotations_path = annotations_root / f"{split}_expressions.json"
        protected[str(compact_manifest)] = sha256_file(compact_manifest)
        protected[str(frozen_manifest)] = sha256_file(frozen_manifest)
        protected[str(annotations_path)] = sha256_file(annotations_path)
        if protected[str(frozen_manifest)] != specification["manifest_sha256"]:
            raise ValueError(f"frozen {split} manifest SHA-256 mismatch")

        predictions = load_compact_manifest(
            compact_manifest,
            expected_split=split,
            expected_count=specification["count"],
            expected_checkpoint_sha256=CHECKPOINT_SHA256,
            expected_manifest_sha256=specification["manifest_sha256"],
        )
        ground_truth = load_frozen_ground_truth_manifest(
            frozen_manifest,
            hifics_root=PROJECT_ROOT / "hifics",
            expected_count=specification["count"],
        )
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["data"]
        if len(annotations) != specification["count"]:
            raise ValueError(f"annotation count mismatch for {split}")

        unique_rgb_hashes: dict[str, str] = {}
        split_metric_rows: list[dict[str, Any]] = []
        for index, (prediction, gt_row, annotation) in enumerate(
            zip(predictions, ground_truth, annotations, strict=True)
        ):
            if (
                prediction.sample_index != index
                or prediction.scene_id != str(gt_row["scene_id"])
                or prediction.query != str(gt_row["text"])
                or annotation["question"] != prediction.query
                or annotation["image_filename"] != prediction.scene_id
            ):
                raise ValueError(f"{split} sample identity mismatch at {index}")
            rgb_sha = str(prediction.raw["source_rgb_sha256"])
            previous = unique_rgb_hashes.setdefault(prediction.scene_id, rgb_sha)
            if previous != rgb_sha:
                raise ValueError(f"frame/RGB identity conflict: {prediction.scene_id}")
            probability = load_probability(prediction.probability_path)
            prediction_mask = probability >= prediction.foreground_threshold
            target = load_ground_truth_mask(gt_row)
            semantics = parse_query(prediction.query)
            detailed = binary_mask_metrics(prediction_mask, target)
            metric_row = {
                "split": split,
                "sample_id": prediction.sample_id,
                "sample_index": index,
                "question_index": prediction.question_index,
                "scene_id": prediction.scene_id,
                "frame_id": prediction.scene_id,
                "rgb_sha256": rgb_sha,
                "query": prediction.query,
                "query_type": semantics.query_type,
                "target_category_offline_audit_only": str(annotation["target"]).rsplit("_", 1)[0],
                **detailed,
            }
            split_metric_rows.append(metric_row)
            all_metrics.append(metric_row)
            split_rows.append(
                {
                    "split": split,
                    "protocol_role": (
                        "development_train"
                        if split == "train"
                        else "development_validation"
                        if split == "val"
                        else "known_posthoc_benchmark"
                    ),
                    "sample_id": prediction.sample_id,
                    "sample_index": index,
                    "scene_group": prediction.scene_id,
                    "frame_group": prediction.scene_id,
                    "rgb_sha256": rgb_sha,
                    "query_type": semantics.query_type,
                }
            )
            if (index + 1) % 2000 == 0:
                print(f"baseline {split}: {index + 1}/{specification['count']}", flush=True)

        split_frame = pd.DataFrame(split_metric_rows)
        summaries[split] = aggregate_metrics(split_frame)
        split_sets[split] = {
            "scene": set(split_frame["scene_id"]),
            "frame": set(split_frame["frame_id"]),
            "rgb": set(split_frame["rgb_sha256"]),
        }
        print(json.dumps({split: summaries[split]}, sort_keys=True), flush=True)

    for key, expected in EXPECTED_TEST.items():
        observed = summaries["test"][key]
        if isinstance(expected, float):
            if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"authoritative baseline mismatch: {key}")
        elif int(observed) != int(expected):
            raise RuntimeError(f"authoritative baseline mismatch: {key}")

    intersections: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        intersections[f"{left}_vs_{right}"] = {
            name: sorted(split_sets[left][name] & split_sets[right][name])
            for name in ("scene", "frame", "rgb")
        }
    overlap_audit = {
        "schema_version": 1,
        "group_unit": "exact OCID-VLG image_filename and verified RGB SHA-256",
        "all_queries_from_same_frame_kept_together": True,
        "fresh_untouched_holdout_available": False,
        "fresh_holdout_reason": (
            "Official validation and test artifacts influenced earlier local development; "
            "creating a new subset now would not make it historically untouched."
        ),
        "intersections": intersections,
        "zero_overlap": all(
            not values
            for pair in intersections.values()
            for values in pair.values()
        ),
    }
    if not overlap_audit["zero_overlap"]:
        raise RuntimeError("official split overlap detected")

    split_frame = pd.DataFrame(split_rows)
    atomic_parquet(baseline_root / "per_sample_metrics.parquet", pd.DataFrame(all_metrics))
    atomic_json(
        baseline_root / "summary.json",
        {
            "schema_version": 1,
            "status": "BASELINE_REPRODUCED_EXACTLY",
            "foreground_rule": "probability >= 0.5",
            "strict_p_at_rule": "IoU > threshold",
            "boundary_tolerance_px_at_352": 2,
            "splits": summaries,
        },
    )
    atomic_parquet(split_root / "split_manifest.parquet", split_frame)
    split_frame.to_csv(split_root / "split_manifest.csv", index=False, lineterminator="\n")
    group_files = {
        "train_groups.txt": sorted(split_sets["train"]["scene"]),
        "validation_groups.txt": sorted(split_sets["val"]["scene"]),
        "fresh_holdout_groups.txt": [],
        "posthoc_benchmark_groups.txt": sorted(split_sets["test"]["scene"]),
    }
    for filename, groups in group_files.items():
        atomic_text(split_root / filename, "" if not groups else "\n".join(groups) + "\n")
    atomic_json(split_root / "overlap_audit.json", overlap_audit)

    output_hashes = {
        "summary.json": sha256_file(baseline_root / "summary.json"),
        "per_sample_metrics.parquet": sha256_file(
            baseline_root / "per_sample_metrics.parquet"
        ),
        "split_manifest.csv": sha256_file(split_root / "split_manifest.csv"),
        "split_manifest.parquet": sha256_file(split_root / "split_manifest.parquet"),
        "overlap_audit.json": sha256_file(split_root / "overlap_audit.json"),
    }
    checksums = {
        "schema_version": 1,
        "status": "BASELINE_REPRODUCED_EXACTLY",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "protected_inputs": dict(sorted(protected.items())),
        "protected_inputs_identity_sha256": stable_json_sha256(dict(sorted(protected.items()))),
        "group_file_sha256": {
            filename: hash_lines(groups) for filename, groups in group_files.items()
        },
        "outputs": output_hashes,
        "original_artifacts_modified": False,
    }
    atomic_json(baseline_root / "baseline_checksums.json", checksums)
    print(json.dumps(checksums, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
