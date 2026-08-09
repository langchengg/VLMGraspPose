#!/usr/bin/env python3
"""Locked formal selective SAM 3 inference with no test-GT access."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
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
from segmentation.sam3_cpu_serialization import atomic_output_directory, sha256_file  # noqa: E402
from segmentation.sam3_cpu_visualization import (  # noqa: E402
    save_coarse_vs_refined,
    save_prompt_visualization,
)
from segmentation.selective_sam3_vg.features import post_sam_features, pre_sam_features  # noqa: E402
from segmentation.selective_sam3_vg.io import load_binary_mask, load_probability, resize_probability  # noqa: E402
from segmentation.selective_sam3_vg.models import select_candidate, trigger_probability  # noqa: E402
from segmentation.selective_sam3_vg.prompts import build_selective_prompt  # noqa: E402


FORBIDDEN_KEYS = {"gt_mask_path", "iou", "delta_iou", "threshold_success", "target_label"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/selective_sam3_vg_LOCKED.yaml",
    )
    parser.add_argument(
        "--lock", type=Path,
        default=ROOT / "artifacts/selective_sam3_vg/formal_lock.json",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs/selective_sam3_vg/formal_test_masks",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                forbidden = FORBIDDEN_KEYS & set(row)
                if forbidden:
                    raise RuntimeError(f"formal inference manifest contains forbidden fields: {sorted(forbidden)}")
                result.append(row)
    return result


def _completed(directory: Path) -> bool:
    try:
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "COMPLETED" and (directory / "final_mask.png").is_file()


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()}
            )


def _fallback_decision(reason: str) -> dict[str, Any]:
    return {
        "selected_index": 0,
        "selected_candidate_id": "coarse_0",
        "selected_source": "hifics",
        "selector_score": None,
        "coarse_score": None,
        "predicted_gain": None,
        "fallback_reason": reason,
        "rejected_candidates": {},
    }


def main() -> None:
    args = _arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if not config.get("LOCKED_BEFORE_TEST") or not lock.get("LOCKED_BEFORE_TEST"):
        raise RuntimeError("formal method is not locked")
    if sha256_file(args.config) != lock["config_sha256"]:
        raise RuntimeError("locked formal configuration hash mismatch")
    with Path(config["trigger"]["artifact_path"]).open("rb") as stream:
        trigger = pickle.load(stream)
    with Path(config["selector"]["artifact_path"]).open("rb") as stream:
        selector = pickle.load(stream)
    if sha256_file(config["trigger"]["artifact_path"]) != config["trigger"]["artifact_sha256"]:
        raise RuntimeError("trigger artifact hash mismatch")
    if sha256_file(config["selector"]["artifact_path"]) != config["selector"]["artifact_sha256"]:
        raise RuntimeError("selector artifact hash mismatch")
    manifest_path = Path(config["formal_test"]["manifest"])
    rows = _rows(manifest_path)
    if args.limit is None and len(rows) != int(config["formal_test"]["sample_count"]):
        raise RuntimeError("formal inference manifest count mismatch")
    if args.limit is not None:
        rows = rows[: args.limit]
    pending = [row for row in rows if not _completed(args.output_root / row["sample_id"])]
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_start = time.time()
    (args.output_root / "inference_contract.json").write_text(
        json.dumps(
            {
                "status": "RUNNING" if pending else "COMPLETED",
                "LOCKED_BEFORE_TEST": True,
                "config_sha256": lock["config_sha256"],
                "manifest_sha256": sha256_file(manifest_path),
                "requested_samples": len(rows),
                "pending_at_start": len(pending),
                "ground_truth_loaded": False,
                "start_unix": run_start,
            }, indent=2, sort_keys=True, allow_nan=False
        ) + "\n", encoding="utf-8"
    )
    model: TransformersSam3Cpu | None = None
    model_load_error: str | None = None
    trigger_count = acceptance_count = 0
    for completed, row in enumerate(pending, start=1):
        started = time.perf_counter()
        rgb = np.asarray(Image.open(row["rgb_path"]).convert("RGB"), dtype=np.uint8)
        depth = np.asarray(Image.open(row["depth_path"]), dtype=np.float32)
        coarse = load_binary_mask(row["native_mask_path"], expected_shape=rgb.shape[:2])
        coarse_probability = resize_probability(load_probability(row["probability_path"]), rgb.shape[:2])
        empty_coarse = not np.any(coarse)
        pre_features = pre_sam_features(coarse, coarse_probability, rgb, depth, row["query"])
        trigger_score = trigger_probability(trigger, pre_features)
        triggered = (not empty_coarse) and trigger_score >= float(config["trigger"]["threshold"])
        trigger_count += int(triggered)
        prompt = None
        result = None
        candidate_rows: list[dict[str, Any]] = []
        decision: dict[str, Any]
        if triggered:
            inference_error: str | None = None
            try:
                prompt = build_selective_prompt(
                    config["prompt"]["family"], coarse_probability, coarse, depth,
                    box_expansion_fraction=float(config["prompt"]["box_expansion_fraction"]),
                    config=config["prompt"],
                )
                if model is None and model_load_error is None:
                    model_config = config["model"]
                    try:
                        model = TransformersSam3Cpu(
                            ROOT / model_config["local_path"],
                            revision=model_config["revision"],
                            backend=model_config["backend"],
                            processor_size=int(model_config["processor_size"]),
                            num_threads=int(config["runtime"]["num_threads"]),
                            interop_threads=int(config["runtime"]["interop_threads"]),
                        )
                    except Exception as error:
                        model_load_error = f"{type(error).__name__}: {error}"
                if model is None:
                    raise RuntimeError(f"persistent_model_load_failed: {model_load_error}")
                result = model.infer(
                    Image.fromarray(rgb, mode="RGB"), prompt.visual_prompt,
                    prompt_mode=prompt.prompt_mode,
                    image_cache_key=str(Path(row["rgb_path"]).resolve()),
                )
                candidate_rows = [
                    post_sam_features(
                        coarse, coarse_probability, None, coarse, coarse_probability, rgb, depth,
                        prompt.visual_prompt, candidate_id="coarse_0",
                    )
                ]
                candidate_rows.extend(
                    post_sam_features(
                        mask, probability, quality, coarse, coarse_probability, rgb, depth,
                        prompt.visual_prompt, candidate_id=f"sam3_{index:03d}",
                    )
                    for index, (mask, probability, quality) in enumerate(
                        zip(result.masks, result.probabilities, result.qualities, strict=True)
                    )
                )
                decision = select_candidate(selector, candidate_rows)
                selected_index = int(decision["selected_index"])
                if selected_index:
                    final_mask = np.asarray(result.masks[selected_index - 1], dtype=bool)
                    final_probability = np.asarray(result.probabilities[selected_index - 1], dtype=np.float32)
                    acceptance_count += 1
                else:
                    final_mask = coarse.copy()
                    final_probability = coarse_probability.copy()
            except Exception as error:
                inference_error = f"{type(error).__name__}: {error}"
                result = None
                candidate_rows = []
                decision = _fallback_decision(f"sam3_or_selector_failed|{inference_error}")
                final_mask = coarse.copy()
                final_probability = coarse_probability.copy()
        else:
            decision = _fallback_decision(
                "empty_coarse_prediction" if empty_coarse else "pre_sam_trigger_not_activated"
            )
            final_mask = coarse.copy()
            final_probability = coarse_probability.copy()
        destination = args.output_root / row["sample_id"]
        with atomic_output_directory(destination) as temporary:
            shutil.copy2(row["native_mask_path"], temporary / "coarse_mask.png")
            np.save(temporary / "coarse_probability.npy", coarse_probability.astype(np.float32), allow_pickle=False)
            (temporary / "pre_sam_features.json").write_text(json.dumps(pre_features, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            (temporary / "trigger_decision.json").write_text(
                json.dumps({"score": trigger_score, "threshold": float(config["trigger"]["threshold"]), "triggered": triggered}, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if triggered and prompt is not None:
                save_prompt_visualization(rgb, prompt.visual_prompt, temporary / "prompt_overlay.png")
                (temporary / "prompt_metadata.json").write_text(json.dumps(prompt.metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            if triggered and prompt is not None and result is not None:
                np.savez_compressed(
                    temporary / "all_sam_hypotheses.npz",
                    candidate_ids=np.asarray([f"sam3_{index:03d}" for index in range(len(result.masks))]),
                    masks=np.asarray(result.masks, dtype=np.uint8),
                    probabilities=np.asarray(result.probabilities, dtype=np.float32),
                    qualities=np.asarray([np.nan if value is None else value for value in result.qualities], dtype=np.float32),
                )
                _csv(temporary / "candidate_features.csv", candidate_rows)
                (temporary / "memory.json").write_text(
                    json.dumps(result.memory, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
            (temporary / "selector_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            Image.fromarray(final_mask.astype(np.uint8) * 255, mode="L").save(temporary / "final_mask.png")
            np.save(temporary / "final_probability.npy", final_probability.astype(np.float32), allow_pickle=False)
            save_coarse_vs_refined(rgb, coarse, final_mask, temporary / "coarse_vs_final.png")
            total_seconds = time.perf_counter() - started
            timing = {"total_sample_seconds": total_seconds, "sam_invoked": triggered}
            if result is not None:
                timing.update({f"sam_{key}": value for key, value in result.timings.items()})
            (temporary / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            provenance = {
                "sample_id": row["sample_id"], "scene_id": row["scene_id"],
                "selected_source": decision["selected_source"],
                "trigger_score": trigger_score, "trigger_threshold": float(config["trigger"]["threshold"]),
                "prompt_family": config["prompt"]["family"] if triggered else None,
                "selected_hypothesis": decision["selected_candidate_id"],
                "selector_score": decision["selector_score"], "predicted_gain": decision["predicted_gain"],
                "fallback_reason": decision["fallback_reason"],
                "sam_model_revision": config["model"]["revision"],
                "trigger_hash": config["trigger"]["artifact_sha256"],
                "selector_hash": config["selector"]["artifact_sha256"],
                "configuration_hash": lock["config_sha256"],
                "no_gt_inference_confirmation": True,
            }
            (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            (temporary / "status.json").write_text(json.dumps({"status": "COMPLETED", "selected_source": decision["selected_source"], "sam_invoked": triggered}, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"formal {completed}/{len(pending)} sample={row['sample_id']} trigger={triggered} selected={decision['selected_source']} seconds={time.perf_counter()-started:.3f}", flush=True)
    terminal_count = sum(_completed(args.output_root / row["sample_id"]) for row in rows)
    contract = {
        "status": "COMPLETED" if terminal_count == len(rows) else "INCOMPLETE",
        "LOCKED_BEFORE_TEST": True,
        "config_sha256": lock["config_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "requested_samples": len(rows), "terminal_samples": terminal_count,
        "trigger_count_this_invocation": trigger_count,
        "acceptance_count_this_invocation": acceptance_count,
        "ground_truth_loaded": False,
        "start_unix": run_start, "end_unix": time.time(), "wall_seconds": time.time() - run_start,
        "model_runtime": None if model is None else model.runtime_metadata,
        "model_load_error": model_load_error,
    }
    (args.output_root / "inference_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if terminal_count != len(rows):
        raise RuntimeError(f"only {terminal_count}/{len(rows)} formal samples reached terminal status")


if __name__ == "__main__":
    main()
