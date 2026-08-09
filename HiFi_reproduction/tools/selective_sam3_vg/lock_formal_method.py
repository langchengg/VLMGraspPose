#!/usr/bin/env python3
"""Create the immutable formal-method lock before any test inference."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import stat
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.io import sha256_file, stable_json_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-analysis", type=Path, required=True)
    parser.add_argument(
        "--trigger-root", type=Path,
        default=ROOT / "artifacts/selective_sam3_vg/locked_trigger",
    )
    parser.add_argument(
        "--selector-root", type=Path,
        default=ROOT / "artifacts/selective_sam3_vg/locked_selector",
    )
    parser.add_argument(
        "--output-config", type=Path,
        default=ROOT / "configs/selective_sam3_vg_LOCKED.yaml",
    )
    parser.add_argument(
        "--output-lock", type=Path,
        default=ROOT / "artifacts/selective_sam3_vg/formal_lock.json",
    )
    args = parser.parse_args()
    if args.output_config.exists() or args.output_lock.exists():
        raise FileExistsError("formal lock already exists; refusing to overwrite it")
    analysis = json.loads(args.prompt_analysis.read_text(encoding="utf-8"))
    if analysis.get("status") != "COMPLETED" or not analysis.get("oracle"):
        raise RuntimeError("prompt analysis is not a completed validation-only oracle diagnostic")
    with (args.trigger_root / "model.pkl").open("rb") as stream:
        import pickle
        trigger = pickle.load(stream)
    with (args.selector_root / "model.pkl").open("rb") as stream:
        selector = pickle.load(stream)
    validation_config = yaml.safe_load(
        (ROOT / "configs/selective_sam3_vg_validation.yaml").read_text(encoding="utf-8")
    )
    config = {
        "schema_version": 1,
        "LOCKED_BEFORE_TEST": True,
        "seed": 42,
        "model": validation_config["model"],
        "runtime": validation_config["runtime"],
        "prompt": {
            **validation_config["prompt"],
            "family": str(analysis["selected_prompt_family"]),
            "box_expansion_fraction": float(analysis["selected_box_expansion_fraction"]),
            "output_threshold": 0.5,
        },
        "trigger": {
            "artifact_path": str((args.trigger_root / "model.pkl").resolve()),
            "artifact_sha256": sha256_file(args.trigger_root / "model.pkl"),
            "model_name": trigger["model_name"],
            "threshold": float(trigger["threshold"]),
        },
        "selector": {
            "artifact_path": str((args.selector_root / "model.pkl").resolve()),
            "artifact_sha256": sha256_file(args.selector_root / "model.pkl"),
            "selector_type": selector["selector_type"],
            "acceptance_margin": float(selector["acceptance_margin"]),
            "conservative_gate": selector["conservative_gate"],
        },
        "formal_test": {
            "manifest": str((ROOT / "outputs/selective_sam3_vg/splits/formal_test_inference_manifest.jsonl").resolve()),
            "sample_count": 7675,
            "ground_truth_access_during_inference": False,
        },
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    lock = {
        "schema_version": 1,
        "LOCKED_BEFORE_TEST": True,
        "locked_timestamp_utc": timestamp,
        "config_path": str(args.output_config.resolve()),
        "config_sha256": sha256_file(args.output_config),
        "prompt_analysis_path": str(args.prompt_analysis.resolve()),
        "prompt_analysis_sha256": sha256_file(args.prompt_analysis),
        "trigger_artifact_sha256": config["trigger"]["artifact_sha256"],
        "selector_artifact_sha256": config["selector"]["artifact_sha256"],
        "formal_test_results_visible_at_lock_time": False,
        "ground_truth_used_to_select_formal_test_masks": False,
    }
    lock["lock_payload_sha256"] = stable_json_sha256(lock)
    args.output_lock.parent.mkdir(parents=True, exist_ok=True)
    args.output_lock.write_text(
        json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Read-only permissions make accidental post-lock edits explicit. They do
    # not replace the cryptographic hashes, which remain the source of truth.
    args.output_config.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    args.output_lock.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
