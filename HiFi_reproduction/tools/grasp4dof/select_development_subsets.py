#!/usr/bin/env python3
"""Select deterministic validation-only 10-sample smoke and 100-sample pilot sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.backends.conditioning import resize_probability_to_native  # noqa: E402
from src.grasping.common.sample_io import (  # noqa: E402
    CompactSampleLoader,
    aligned_labels,
    read_deployment_manifest,
    read_label_manifest,
)


RELATION = re.compile(
    r"\b(left|right|behind|front|next|near|far|closest|between|beside)\b", re.I
)
LOCATION = re.compile(r"\b(leftmost|rightmost|top|bottom|middle|center)\b", re.I)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _features(
    deployment: list[dict], labels: list[dict]
) -> list[dict[str, object]]:
    loader = CompactSampleLoader()
    rows: list[dict[str, object]] = []
    for index, (row, label) in enumerate(zip(deployment, labels, strict=True)):
        sample = loader.load(row, mask_source="predicted")
        mask = sample.binary_mask
        probability = resize_probability_to_native(sample.probability, mask.shape)
        gt_small = np.asarray(Image.open(str(label["prepared_gt_mask_path"]))) > 0
        gt = np.asarray(
            Image.fromarray(gt_small).resize(
                (mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST
            )
        ).astype(bool)
        union = int(np.count_nonzero(mask | gt))
        intersection = int(np.count_nonzero(mask & gt))
        component_labels, _ = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8
        )
        valid = np.isfinite(sample.depth_m) & (sample.depth_m > 0)
        area = int(np.count_nonzero(mask))
        rows.append(
            {
                "sample_index": index,
                "sample_id": sample.sample_id,
                "scene_id": sample.scene_id,
                "language": sample.language,
                "predicted_mask_area_px": area,
                "gt_mask_area_px": int(np.count_nonzero(gt)),
                "mask_iou": 0.0 if union == 0 else intersection / union,
                "mask_mean_confidence": 0.0
                if area == 0
                else float(probability[mask].mean()),
                "valid_target_depth_fraction": 0.0
                if area == 0
                else float(np.count_nonzero(valid & mask) / area),
                "predicted_component_count": int(max(component_labels - 1, 0)),
                "is_relation_query": bool(RELATION.search(sample.language)),
                "is_location_query": bool(LOCATION.search(sample.language)),
            }
        )
    return rows


def _choose_smoke(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used: set[str] = set()

    def take(tag: str, ordered: list[dict[str, object]]) -> None:
        for row in ordered:
            if str(row["sample_id"]) not in used:
                selected.append({"coverage_tag": tag, **row})
                used.add(str(row["sample_id"]))
                return
        raise RuntimeError(f"unable to select distinct smoke sample for {tag}")

    take("wrong_mask", sorted(rows, key=lambda row: float(row["mask_iou"])))
    take("small_object", sorted(rows, key=lambda row: int(row["gt_mask_area_px"])))
    take("large_object", sorted(rows, key=lambda row: -int(row["gt_mask_area_px"])))
    take(
        "relation_query",
        sorted(rows, key=lambda row: (not bool(row["is_relation_query"]), row["sample_id"])),
    )
    take(
        "location_query",
        sorted(rows, key=lambda row: (not bool(row["is_location_query"]), row["sample_id"])),
    )
    take(
        "clutter_fragmented",
        sorted(rows, key=lambda row: -int(row["predicted_component_count"])),
    )
    take(
        "invalid_depth_stress",
        sorted(rows, key=lambda row: float(row["valid_target_depth_fraction"])),
    )
    take(
        "low_mask_confidence",
        sorted(rows, key=lambda row: float(row["mask_mean_confidence"])),
    )
    take(
        "high_mask_confidence",
        sorted(rows, key=lambda row: -float(row["mask_mean_confidence"])),
    )
    areas = np.asarray([int(row["predicted_mask_area_px"]) for row in rows])
    median = float(np.median(areas))
    take(
        "median_case",
        sorted(rows, key=lambda row: abs(int(row["predicted_mask_area_px"]) - median)),
    )
    return selected


def _choose_pilot(rows: list[dict[str, object]], count: int, seed: int) -> list[dict]:
    by_scene: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    ranked_scenes = sorted(
        by_scene,
        key=lambda scene: hashlib.sha256(f"{seed}\0{scene}".encode()).hexdigest(),
    )
    chosen: list[dict] = []
    round_index = 0
    while len(chosen) < count:
        progressed = False
        for scene in ranked_scenes:
            scene_rows = sorted(by_scene[scene], key=lambda row: str(row["sample_id"]))
            if round_index < len(scene_rows):
                chosen.append(dict(scene_rows[round_index]))
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            raise RuntimeError("insufficient validation samples for pilot")
        round_index += 1
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    manifests = run_dir / "manifests"
    deployment = read_deployment_manifest(manifests / "validation_samples.parquet")
    labels = read_label_manifest(manifests / "validation_labels.parquet")
    labels = list(aligned_labels(deployment, labels))
    feature_rows = _features(deployment, labels)
    smoke = _choose_smoke(feature_rows)
    pilot = _choose_pilot(feature_rows, 100, args.seed)
    outputs = {
        "development_smoke_10.json": {
            "split": "validation",
            "sample_count": 10,
            "synthetic_empty_mask_required": True,
            "synthetic_empty_mask_reason": "validation contains no naturally empty predicted mask",
            "samples": smoke,
        },
        "pilot_100.json": {
            "split": "validation",
            "sample_count": 100,
            "scene_count": len({row["scene_id"] for row in pilot}),
            "seed": args.seed,
            "samples": pilot,
        },
    }
    for name, value in outputs.items():
        path = manifests / name
        if path.exists():
            raise FileExistsError(path)
        _atomic_json(path, value)
    print(json.dumps({"status": "COMPLETE", "outputs": list(outputs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
