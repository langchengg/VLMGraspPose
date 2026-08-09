#!/usr/bin/env python3
"""Resumable, GT-free SAM 3 validation-pilot inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.sam3_cpu_model import TransformersSam3Cpu  # noqa: E402
from segmentation.sam3_cpu_serialization import (  # noqa: E402
    atomic_output_directory,
    atomic_write_json,
    sha256_file,
)
from segmentation.selective_sam3_vg.features import (  # noqa: E402
    post_sam_features,
    pre_sam_features,
)
from segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_binary_mask,
    load_probability,
    resize_probability,
)
from segmentation.selective_sam3_vg.prompts import (  # noqa: E402
    PROMPT_FAMILIES,
    build_selective_prompt,
)


FORBIDDEN_INFERENCE_KEYS = {
    "gt_mask_path", "iou", "delta_iou", "threshold_success", "target_label",
    "mask_precision", "mask_recall", "boundary_fscore",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/selective_sam3_vg/validation_pilot/pilot_inference_manifest.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/selective_sam3_vg_validation.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/selective_sam3_vg/validation_pilot/initial_prompt_comparison",
    )
    parser.add_argument("--families", nargs="+", default=list(PROMPT_FAMILIES))
    parser.add_argument("--box-expansion", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--oracle-diagnostic", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                forbidden = FORBIDDEN_INFERENCE_KEYS & set(row)
                if forbidden:
                    raise RuntimeError(f"GT/evaluation leakage in inference manifest: {sorted(forbidden)}")
                rows.append(row)
    return rows


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, tuple, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def _completed(path: Path) -> bool:
    status = path / "status.json"
    if not status.is_file():
        return False
    try:
        value = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("status") == "COMPLETED" and (path / "all_sam_hypotheses.npz").is_file()


def main() -> None:
    args = _arguments()
    unknown = set(args.families) - set(PROMPT_FAMILIES)
    if unknown:
        raise ValueError(f"unknown prompt families: {sorted(unknown)}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = _load_rows(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    work: list[tuple[dict[str, Any], str, Path]] = []
    for row in rows:
        for family in args.families:
            destination = args.output_root / str(row["sample_id"]) / family
            if not _completed(destination):
                work.append((row, family, destination))
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    atomic_write_json(
        args.output_root / "run_contract.json",
        {
            "status": "RUNNING" if work else "COMPLETED",
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "families": args.families,
            "box_expansion_fraction": args.box_expansion,
            "sample_count": len(rows),
            "expected_configuration_count": len(rows) * len(args.families),
            "pending_configuration_count_at_start": len(work),
            "ground_truth_loaded": False,
            "oracle": bool(args.oracle_diagnostic),
            "start_unix": run_started,
        },
    )
    if not work:
        print("all requested pilot configurations already completed")
        return
    model_config = config["model"]
    model = TransformersSam3Cpu(
        ROOT / model_config["local_path"],
        revision=str(model_config["revision"]),
        backend=str(model_config["backend"]),
        processor_size=int(model_config["processor_size"]),
        num_threads=int(config["runtime"]["num_threads"]),
        interop_threads=int(config["runtime"]["interop_threads"]),
    )
    for completed, (row, family, destination) in enumerate(work, start=1):
        sample_started = time.perf_counter()
        rgb = np.asarray(Image.open(row["rgb_path"]).convert("RGB"), dtype=np.uint8)
        depth = np.asarray(Image.open(row["depth_path"]), dtype=np.float32)
        coarse = load_binary_mask(row["native_mask_path"], expected_shape=rgb.shape[:2])
        probability = resize_probability(load_probability(row["probability_path"]), rgb.shape[:2])
        pre_features = pre_sam_features(coarse, probability, rgb, depth, str(row["query"]))
        built = build_selective_prompt(
            family,
            probability,
            coarse,
            depth,
            box_expansion_fraction=float(args.box_expansion),
            config=config["prompt"],
        )
        result = model.infer(
            Image.fromarray(rgb, mode="RGB"),
            built.visual_prompt,
            prompt_mode=built.prompt_mode,
            image_cache_key=str(Path(row["rgb_path"]).resolve()),
        )
        candidate_rows = [
            post_sam_features(
                mask, candidate_probability, quality, coarse, probability, rgb, depth,
                built.visual_prompt, candidate_id=f"sam3_{index:03d}",
            )
            for index, (mask, candidate_probability, quality) in enumerate(
                zip(result.masks, result.probabilities, result.qualities, strict=True)
            )
        ]
        candidate_rows.insert(
            0,
            post_sam_features(
                coarse, probability, None, coarse, probability, rgb, depth,
                built.visual_prompt, candidate_id="coarse_0",
            ),
        )
        with atomic_output_directory(destination) as temporary:
            np.savez_compressed(
                temporary / "all_sam_hypotheses.npz",
                candidate_ids=np.asarray([f"sam3_{index:03d}" for index in range(len(result.masks))]),
                masks=np.asarray(result.masks, dtype=np.uint8),
                probabilities=np.asarray(result.probabilities, dtype=np.float32),
                qualities=np.asarray([np.nan if value is None else value for value in result.qualities], dtype=np.float32),
            )
            _save_csv(temporary / "candidate_features.csv", candidate_rows)
            (temporary / "pre_sam_features.json").write_text(
                json.dumps(pre_features, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (temporary / "prompt_metadata.json").write_text(
                json.dumps(built.metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (temporary / "timing.json").write_text(
                json.dumps(result.timings, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (temporary / "memory.json").write_text(
                json.dumps(result.memory, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            provenance = {
                "sample_id": row["sample_id"],
                "scene_id": row["scene_id"],
                "query": row["query"],
                "prompt_family": family,
                "box_expansion_fraction": float(args.box_expansion),
                "model_revision": model.runtime_metadata["model_revision"],
                "model_backend": model.backend,
                "config_sha256": sha256_file(args.config),
                "prediction_probability_path": str(Path(row["probability_path"]).resolve()),
                "prediction_mask_path": str(Path(row["native_mask_path"]).resolve()),
                "ground_truth_loaded": False,
                "oracle": bool(args.oracle_diagnostic),
            }
            (temporary / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (temporary / "status.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "hypothesis_count": len(result.masks),
                        "wall_seconds": time.perf_counter() - sample_started,
                        "oracle": bool(args.oracle_diagnostic),
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
        print(
            f"pilot {completed}/{len(work)} sample={row['sample_id']} family={family} "
            f"hypotheses={len(result.masks)} seconds={time.perf_counter() - sample_started:.3f}",
            flush=True,
        )
    atomic_write_json(
        args.output_root / "run_contract.json",
        {
            "status": "COMPLETED",
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "families": args.families,
            "box_expansion_fraction": args.box_expansion,
            "sample_count": len(rows),
            "configuration_count": len(rows) * len(args.families),
            "ground_truth_loaded": False,
            "oracle": bool(args.oracle_diagnostic),
            "start_unix": run_started,
            "end_unix": time.time(),
            "wall_seconds": time.time() - run_started,
            "model_runtime": model.runtime_metadata,
        },
    )


if __name__ == "__main__":
    main()
