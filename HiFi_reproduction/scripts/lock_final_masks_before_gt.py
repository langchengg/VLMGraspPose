#!/usr/bin/env python3
"""Verify and checksum all GT-free final masks before enabling GT evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
    sha256_file,
    stable_json_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = args.experiment_root.expanduser().resolve()
    artifact = args.artifact_root.expanduser().resolve()
    formal_lock = json.loads((artifact / "formal_lock.json").read_text(encoding="utf-8"))
    compact = load_compact_manifest(
        PROJECT_ROOT
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/test/manifest.jsonl",
        expected_split="test",
        expected_count=7675,
    )
    output_root = experiment / "locked_benchmark_masks"
    marker = output_root / "LOCKED_BEFORE_GT_EVALUATION"
    if marker.exists():
        raise FileExistsError("benchmark outputs are already locked")
    expected_ids = {prediction.sample_id for prediction in compact}
    actual_ids = {
        path.name
        for path in output_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    if actual_ids != expected_ids:
        raise RuntimeError(
            "benchmark output directory identity mismatch: "
            f"missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)}"
        )
    access_log = experiment / "leakage_audit/runtime_file_access_benchmark.jsonl"
    if not access_log.is_file() or access_log.stat().st_size == 0:
        raise RuntimeError("benchmark runtime file-access audit is missing")
    access_records = [
        json.loads(line)
        for line in access_log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    forbidden_records = [
        record for record in access_records if record.get("forbidden_token") is not None
    ]
    if forbidden_records:
        raise RuntimeError("forbidden file access occurred during locked inference")
    rows = []
    outcomes: dict[str, int] = {}
    for number, prediction in enumerate(compact, start=1):
        directory = output_root / prediction.sample_id
        status_path = directory / "terminal_status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"missing terminal status: {prediction.sample_id}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE_GT_FREE_INFERENCE":
            raise RuntimeError(f"non-terminal sample: {prediction.sample_id}: {status}")
        for name, expected in status.get("file_sha256", {}).items():
            path = directory / name
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"checksum failure: {prediction.sample_id}/{name}")
        mask_path = directory / "final_mask.png"
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if mask.shape != (480, 640) or not np.isfinite(mask).all():
            raise ValueError(f"invalid final mask: {prediction.sample_id}: {mask.shape}")
        if not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"non-binary final mask: {prediction.sample_id}")
        provenance = json.loads((directory / "provenance.json").read_text())
        if provenance.get("formal_lock_sha256") != formal_lock["formal_lock_sha256"]:
            raise RuntimeError(f"formal-lock drift: {prediction.sample_id}")
        outcome = str(status["terminal_outcome"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        rows.append(
            {
                "sample_index": prediction.sample_index,
                "sample_id": prediction.sample_id,
                "rgb_sha256": str(prediction.raw["source_rgb_sha256"]),
                "final_mask_path": str(mask_path),
                "final_mask_sha256": sha256_file(mask_path),
                "provenance_sha256": sha256_file(directory / "provenance.json"),
                "terminal_status_sha256": sha256_file(status_path),
                "terminal_outcome": outcome,
                "final_probability_available": bool(
                    provenance["final_probability_available"]
                ),
            }
        )
        if number % 250 == 0:
            print(f"pre-GT lock verification: {number}/7675", flush=True)
    manifest_path = output_root / "immutable_output_manifest.jsonl"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(manifest_path)
    lock_payload = {
        "status": "LOCKED_BEFORE_GT_EVALUATION",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "sample_ids_sha256": stable_json_sha256([row["sample_id"] for row in rows]),
        "immutable_output_manifest_sha256": sha256_file(manifest_path),
        "formal_lock_sha256": formal_lock["formal_lock_sha256"],
        "terminal_outcomes": outcomes,
        "ground_truth_opened_during_inference": False,
        "runtime_file_access_record_count": len(access_records),
        "runtime_file_access_log_sha256": sha256_file(access_log),
    }
    marker.write_text(
        json.dumps(lock_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(lock_payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
