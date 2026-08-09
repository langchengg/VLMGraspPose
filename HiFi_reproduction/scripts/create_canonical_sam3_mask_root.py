#!/usr/bin/env python3
"""Export the locked method as a new canonical root only if safeguards pass."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
    sha256_file,
)
from src.segmentation.camera_intrinsics import resolve_camera_intrinsics  # noqa: E402


EXPERIMENT = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1"
ARTIFACT = PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1"
CANONICAL = PROJECT_ROOT / "runs/hifics_sam3_proposal_selector_LOCKED"


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    metrics_payload = json.loads(
        (EXPERIMENT / "report/formal_metrics.json").read_text(encoding="utf-8")
    )
    metrics = metrics_payload["metrics"]
    baseline = metrics["Local HiFi-CS baseline"]
    locked = metrics["Locked conservative method"]
    margin = 0.001
    safeguards = {
        "p90_improves": locked["p_at_90"] > baseline["p_at_90"],
        "miou_noninferior": locked["mean_iou"] >= baseline["mean_iou"] - margin,
        "p50_noninferior": locked["p_at_50"] >= baseline["p_at_50"] - margin,
        "p60_noninferior": locked["p_at_60"] >= baseline["p_at_60"] - margin,
    }
    decision = {
        "safeguards": safeguards,
        "passed": all(safeguards.values()),
        "noninferiority_margin_absolute": margin,
        "canonical_root": str(CANONICAL),
    }
    decision_path = EXPERIMENT / "report/canonical_export_decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not decision["passed"]:
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    if CANONICAL.exists():
        raise FileExistsError(f"refusing to overwrite canonical root: {CANONICAL}")
    building = CANONICAL.with_name(f".{CANONICAL.name}.building.{os.getpid()}")
    if building.exists():
        raise FileExistsError(f"stale canonical build root requires inspection: {building}")
    building.mkdir(parents=True)
    compact = load_compact_manifest(
        PROJECT_ROOT
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/test/manifest.jsonl",
        expected_split="test",
        expected_count=7675,
    )
    formal_lock = json.loads((ARTIFACT / "formal_lock.json").read_text())
    gate_hash = sha256_file(ARTIFACT / "final_selector/gate.json")
    stage1_hash = sha256_file(ARTIFACT / "stage1_selector/model.pkl")
    stage2_hash = sha256_file(ARTIFACT / "final_selector/model.pkl")
    manifest_rows = []
    for number, row in enumerate(compact, start=1):
        source = EXPERIMENT / "locked_benchmark_masks" / row.sample_id
        destination = building / row.sample_id
        destination.mkdir()
        _, intrinsics, intrinsics_payload = resolve_camera_intrinsics(
            row,
            cache_root=EXPERIMENT / "cache/intrinsics",
            image_shape=(480, 640),
        )
        _copy(row.rgb_path, destination / "color.png")
        _copy(row.depth_path, destination / "depth.png")
        _copy(source / "final_mask.png", destination / "target_mask.png")
        probability_available = (source / "final_probability.npy").is_file()
        if probability_available:
            _copy(source / "final_probability.npy", destination / "target_probability.npy")
        _copy(intrinsics, destination / "intrinsics.json")
        (destination / "language.txt").write_text(row.query + "\n", encoding="utf-8")
        provenance = json.loads((source / "provenance.json").read_text())
        _copy(source / "provenance.json", destination / "provenance.json")
        metadata = {
            "sample_id": row.sample_id,
            "split": "posthoc_benchmark_test",
            "original_hifi_source": str(row.native_mask_path),
            "selected_proposal_source": provenance["source_family"],
            "final_probability_available": probability_available,
            "sam_model_revision": formal_lock["model"]["revision"],
            "camera_intrinsics_sha256": sha256_file(intrinsics),
            "camera_intrinsics_source": intrinsics_payload.get("source"),
            "camera_factory_calibration": intrinsics_payload.get(
                "factory_calibration"
            ),
            "query_parse": json.loads((source / "query_parse.json").read_text()),
            "stage1_selector_hash": stage1_hash,
            "stage2_selector_hash": stage2_hash,
            "stage2_configuration_hash": formal_lock["config_hashes"].get(
                "configs/sam3_proposal_bank_p90_v1/second_refinement.yaml"
            ),
            "final_gate_hash": gate_hash,
            "formal_lock_sha256": formal_lock["formal_lock_sha256"],
            "benchmark_protocol_caveat": formal_lock["posthoc_benchmark_caveat"],
            "physical_grasp_success_evaluated": False,
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        checksums = {
            path.name: sha256_file(path)
            for path in sorted(destination.iterdir())
            if path.is_file()
        }
        (destination / "checksums.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "sample_index": row.sample_index,
                "sample_id": row.sample_id,
                "path": str(CANONICAL / row.sample_id),
                "target_mask_sha256": checksums["target_mask.png"],
                "selected_source": provenance["source_family"],
                "final_probability_available": probability_available,
            }
        )
        if number % 250 == 0:
            print(f"canonical export: {number}/7675", flush=True)
    with (building / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    model_manifest = {
        "formal_lock": formal_lock,
        "stage1_selector_sha256": stage1_hash,
        "final_selector_sha256": stage2_hash,
        "gate_sha256": gate_hash,
        "NO_TEST_GT_USED_FOR_SELECTION": True,
        "canonical_export_triggered_by_post_lock_metrics": True,
    }
    (building / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (building / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    building.replace(CANONICAL)
    decision["exported_samples"] = len(manifest_rows)
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
