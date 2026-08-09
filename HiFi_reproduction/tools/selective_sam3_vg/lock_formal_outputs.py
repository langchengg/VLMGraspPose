#!/usr/bin/env python3
"""Checksum and freeze every formal output before the evaluator may load GT."""

from __future__ import annotations

import datetime as dt
import json
import stat
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.io import sha256_file, stable_json_sha256  # noqa: E402


def main() -> None:
    inference_manifest = ROOT / "outputs/selective_sam3_vg/splits/formal_test_inference_manifest.jsonl"
    output_root = ROOT / "outputs/selective_sam3_vg/formal_test_masks"
    output_manifest = ROOT / "outputs/selective_sam3_vg/formal_output_manifest.parquet"
    output_csv = ROOT / "outputs/selective_sam3_vg/formal_output_manifest.csv"
    output_lock = ROOT / "artifacts/selective_sam3_vg/formal_output_lock.json"
    if output_manifest.exists() or output_csv.exists() or output_lock.exists():
        raise FileExistsError("formal output lock already exists; refusing to overwrite")
    inference_rows = [json.loads(line) for line in inference_manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(inference_rows) != 7675:
        raise RuntimeError("formal inference manifest does not contain 7,675 samples")
    rows = []
    for number, source in enumerate(inference_rows, start=1):
        directory = output_root / source["sample_id"]
        required = (
            "status.json", "provenance.json", "trigger_decision.json", "selector_decision.json",
            "coarse_mask.png", "coarse_probability.npy", "final_mask.png", "final_probability.npy",
        )
        if any(not (directory / name).is_file() for name in required):
            raise RuntimeError(f"incomplete formal output: {directory}")
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETED" or not provenance.get("no_gt_inference_confirmation"):
            raise RuntimeError(f"invalid terminal provenance: {directory}")
        rows.append(
            {
                "sample_index": int(source["sample_index"]),
                "sample_id": source["sample_id"],
                "scene_id": source["scene_id"],
                "selected_source": provenance["selected_source"],
                "sam_invoked": bool(status["sam_invoked"]),
                "final_mask_path": str((directory / "final_mask.png").resolve()),
                "final_mask_sha256": sha256_file(directory / "final_mask.png"),
                "final_probability_path": str((directory / "final_probability.npy").resolve()),
                "final_probability_sha256": sha256_file(directory / "final_probability.npy"),
                "provenance_sha256": sha256_file(directory / "provenance.json"),
                "status_sha256": sha256_file(directory / "status.json"),
            }
        )
        if number % 500 == 0:
            print(f"checksummed {number}/7675 formal outputs", flush=True)
    frame = pd.DataFrame(rows).sort_values("sample_index")
    if frame["sample_id"].nunique() != 7675 or not frame["sample_index"].tolist() == list(range(7675)):
        raise RuntimeError("formal terminal outputs are not unique and contiguous")
    frame.to_parquet(output_manifest, index=False)
    frame.to_csv(output_csv, index=False)
    formal_lock = json.loads((ROOT / "artifacts/selective_sam3_vg/formal_lock.json").read_text(encoding="utf-8"))
    lock = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_GT_EVALUATION",
        "locked_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_count": 7675,
        "formal_method_lock_sha256": sha256_file(ROOT / "artifacts/selective_sam3_vg/formal_lock.json"),
        "formal_config_sha256": formal_lock["config_sha256"],
        "output_manifest_path": str(output_manifest.resolve()),
        "output_manifest_sha256": sha256_file(output_manifest),
        "output_manifest_csv_sha256": sha256_file(output_csv),
        "final_masks_frozen_before_gt_evaluation": True,
    }
    lock["lock_payload_sha256"] = stable_json_sha256(lock)
    output_lock.parent.mkdir(parents=True, exist_ok=True)
    output_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (output_manifest, output_csv, output_lock):
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
