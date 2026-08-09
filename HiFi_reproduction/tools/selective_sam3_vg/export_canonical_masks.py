#!/usr/bin/env python3
"""Publish canonical masks only when the locked hybrid passes its safety gate."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.selective_sam3_vg.io import sha256_file  # noqa: E402


def _link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    metrics = json.loads(
        (ROOT / "outputs/selective_sam3_vg/report/formal_metrics.json").read_text(encoding="utf-8")
    )
    baseline, hybrid = metrics["baseline"], metrics["locked_selective_sam3"]
    safe = (
        hybrid["mean_iou"] > baseline["mean_iou"]
        and hybrid["p_at_50"] - baseline["p_at_50"] >= -0.001
        and hybrid["p_at_60"] - baseline["p_at_60"] >= -0.001
    )
    decision_path = ROOT / "artifacts/selective_sam3_vg/canonical_export_decision.json"
    if not safe:
        decision = {
            "status": "NOT_EXPORTED", "reason": "formal visual-grounding safety gate not satisfied",
            "baseline_mean_iou": baseline["mean_iou"], "hybrid_mean_iou": hybrid["mean_iou"],
        }
        decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(decision, indent=2, sort_keys=True))
        return
    destination = ROOT / "runs/hifics_selective_sam3_LOCKED"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite canonical root: {destination}")
    destination.mkdir(parents=True)
    output_manifest = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/formal_output_manifest.parquet")
    baseline_manifest = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/baseline_manifest.parquet").set_index("sample_id")
    config = json.loads((ROOT / "artifacts/selective_sam3_vg/formal_lock.json").read_text(encoding="utf-8"))
    manifest_rows = []
    for number, row in enumerate(output_manifest.itertuples(index=False), start=1):
        source_dir = Path(row.final_mask_path).parent
        sample_dir = destination / row.sample_id
        sample_dir.mkdir()
        _link(source_dir / "final_mask.png", sample_dir / "target_mask.png")
        _link(source_dir / "final_probability.npy", sample_dir / "target_probability.npy")
        provenance = json.loads((source_dir / "provenance.json").read_text(encoding="utf-8"))
        original = baseline_manifest.loc[row.sample_id]
        metadata = {
            "sample_id": row.sample_id, "scene_id": row.scene_id,
            "source_checkpoint_sha256": original["checkpoint_sha256"],
            "original_hifi_mask": original["coarse_mask_path"],
            "selected_source": row.selected_source,
            "sam_model_revision": provenance["sam_model_revision"],
            "trigger_hash": provenance["trigger_hash"], "selector_hash": provenance["selector_hash"],
            "formal_config_hash": config["config_sha256"],
            "final_visual_grounding_metrics": hybrid,
            "probability_resolution": [480, 640], "mask_resolution": [480, 640],
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "sample_index": int(row.sample_index), "sample_id": row.sample_id,
                "target_mask_path": str((sample_dir / "target_mask.png").resolve()),
                "target_probability_path": str((sample_dir / "target_probability.npy").resolve()),
                "metadata_path": str((sample_dir / "metadata.json").resolve()),
                "selected_source": row.selected_source,
            }
        )
        if number % 500 == 0:
            print(f"canonical export {number}/7675", flush=True)
    with (destination / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    decision = {
        "status": "EXPORTED", "canonical_root": str(destination.resolve()),
        "sample_count": len(manifest_rows), "manifest_sha256": sha256_file(destination / "manifest.jsonl"),
        "dexnet_run": False,
    }
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
