#!/usr/bin/env python3
"""Create the formal run, audit immutable inputs, and build the paired manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import skimage
import statsmodels
import torch
import torchvision
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
METHODS = ("CROG-native", "HiFi-CS→G1", "HiFi-CS→C1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    process = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return (process.stdout or process.stderr).strip()


def git_state(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "commit": command("git", "rev-parse", "HEAD", cwd=path),
        "branch": command("git", "branch", "--show-current", cwd=path) or "DETACHED",
        "dirty": bool(command("git", "status", "--porcelain", cwd=path)),
        "status_porcelain": command("git", "status", "--porcelain", cwd=path).splitlines(),
    }


def checkpoint_rows(source_run: Path) -> list[dict[str, Any]]:
    g1 = json.loads((source_run / "selected_configs/G1.json").read_text())
    c1 = json.loads((source_run / "selected_configs/C1.json").read_text())
    hifi = Path(g1["training_data_lineage"]["formal_input_preflight"]["path"]).parents[2]
    del hifi
    records = [
        {
            "method": "CROG-native",
            "path": ROOT / "crog_reproduction/CROG/exp/OCID-VLG_multiple_mac/CROG_mac_mps_official_params_50epoch_bs8/best_jindex_model.pth",
            "training_source": "official CROG OCID-VLG multiple training run; local strict-load audit",
            "validation_basis": "epoch 50 best validation J-index checkpoint (checkpoint payload and local audit)",
            "optimizer_state": True,
            "fine_tuned": True,
        },
        {
            "method": "HiFi-CS shared frontend",
            "path": ROOT / "HiFi_reproduction/runs/hifics_ocidvlg_hierfilm_20260727_214615/checkpoints/best.pth",
            "training_source": "local repeated/hierarchical-FiLM OCID-VLG training run",
            "validation_basis": "best validation mIoU 0.820741 at step 19728",
            "optimizer_state": True,
            "fine_tuned": True,
        },
        {
            "method": "G1 GR-ConvNet",
            "path": Path(g1["finetuned_checkpoint"]),
            "training_source": "local OCID-VLG fine-tuning over official GR-ConvNet initializer",
            "validation_basis": f"epoch {g1['selection']['best_epoch']}; val J@1={g1['selection']['best_validation']['j_at_1']:.12f}",
            "optimizer_state": False,
            "fine_tuned": True,
        },
        {
            "method": "C1 GG-CNN2",
            "path": Path(c1["finetuned_checkpoint"]),
            "training_source": "local OCID-VLG fine-tuning over official GG-CNN2 initializer",
            "validation_basis": f"epoch {c1['selection']['best_epoch']}; val J@1={c1['selection']['best_validation']['j_at_1']:.12f}",
            "optimizer_state": False,
            "fine_tuned": True,
        },
    ]
    rows = []
    for record in records:
        path = Path(record.pop("path")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                **record,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def expression_type(row: dict[str, Any]) -> str:
    template = str(row.get("template_filename", "")).lower()
    for label in ("name", "attribute", "relation", "location"):
        if template == f"{label}.json":
            return label
    return "mixed"


def object_category(name: str) -> str:
    parts = str(name).rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(name)


def prepare(args: argparse.Namespace) -> None:
    run = args.run_dir.resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"refusing non-empty formal run: {run}")
    directories = [
        "00_audit", "01_manifest", "02_predictions", "03_canonical", "04_metrics",
        "05_statistics", "06_failure_analysis", "07_figures", "08_qualitative",
        "09_reports", "logs", "config", "tests", "source_snapshot",
    ]
    for name in directories:
        (run / name).mkdir(parents=True, exist_ok=True)
    source_run = args.source_run.resolve()
    crog_jsonl = args.crog_predictions.resolve()
    if json.loads((source_run / "FINALIZATION_COMPLETE.json").read_text()).get("status") != "COMPLETE":
        raise RuntimeError("source modular run is not finalized")

    # Snapshot this experiment implementation and the exact frozen geometry.
    for source in sorted(PACKAGE.glob("*.py")):
        shutil.copy2(source, run / "source_snapshot" / source.name)
    shutil.copy2(PACKAGE / "geometry.py", run / "config" / "canonical_evaluator.py")
    shutil.copy2(source_run / "selected_configs/G1.json", run / "config/G1_validation_selected.json")
    shutil.copy2(source_run / "selected_configs/C1.json", run / "config/C1_validation_selected.json")

    geometry_contract = """schema_version: 1
protocol: fair-4dof-no-rerank-v1
coordinate_frame: original_OCID_VLG_640x480
x_semantics: image_column
y_semantics: image_row
angle_domain_deg: '[-90, 90) with 180-degree parallel-jaw periodicity'
success:
  same_gt_required: true
  rotated_raster_iou: '> 0.25 (strict)'
  periodic_angle_error_deg: '<= 30'
ground_truth:
  conversion: 'centre=(corner0+corner2)/2; jaw axis=corner3-corner0'
  jaw_width_clip_px: 100
  rectangle_height_px: 20
predictions:
  geometry_is_not_modified_by_evaluator: true
  crog_rectangle_height_px: 20
  grconvnet_ggcnn2_rectangle_height: 'native jaw_width_px / 2 (upstream Grasp semantics)'
primary_iou: corrected clipped raster polygon on shape [480,640]
continuous_cross_check: OpenCV intersectConvexConvex and Shapely Polygon
"""
    (run / "config/canonical_geometry_contract.yaml").write_text(geometry_contract)

    checkpoints = checkpoint_rows(source_run)
    pd.DataFrame(checkpoints).to_csv(run / "00_audit/checkpoint_registry.csv", index=False)
    git = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "repositories": [
            git_state(ROOT),
            git_state(ROOT / "HiFi_reproduction"),
            git_state(ROOT / "HiFi_reproduction/third_party_src/grconvnet"),
            git_state(ROOT / "HiFi_reproduction/third_party_src/ggcnn"),
            git_state(ROOT / "crog_reproduction/CROG"),
        ],
    }
    (run / "00_audit/git_state.json").write_text(json.dumps(git, indent=2) + "\n")
    memory_bytes = int(command("sysctl", "-n", "hw.memsize") or 0)
    environment = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "apple_model": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "ram_bytes": memory_bytes,
        "macos": platform.mac_ver()[0],
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "shapely": shapely.__version__,
        "scikit_image": skimage.__version__,
        "statsmodels": statsmodels.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    (run / "00_audit/environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    protected_paths = [
        source_run / "FINALIZATION_COMPLETE.json",
        source_run / "manifests/test_samples.parquet",
        source_run / "manifests/test_labels.parquet",
        source_run / "selected_configs/G1.json",
        source_run / "selected_configs/C1.json",
        crog_jsonl,
        args.crog_features.resolve(),
        PACKAGE / "geometry.py",
        PACKAGE / "native_inference.py",
    ] + [Path(item["path"]) for item in checkpoints]
    protected = {
        str(path.resolve()): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in protected_paths
    }
    (run / "00_audit/source_run_hashes.json").write_text(
        json.dumps({"phase": "before", "files": protected}, indent=2) + "\n"
    )

    samples = pq.read_table(source_run / "manifests/test_samples.parquet").to_pylist()
    labels = {str(row["sample_id"]): row for row in pq.read_table(source_run / "manifests/test_labels.parquet").to_pylist()}
    train = pq.read_table(source_run / "manifests/train_samples.parquet").to_pylist()
    validation = pq.read_table(source_run / "manifests/validation_samples.parquet").to_pylist()
    train_val_rgb = {row["source_rgb_sha256"] for row in train + validation}
    train_val_rgbd = {row["rgbd_pair_sha256"] for row in train + validation}
    # In the frozen OCID-VLG manifests ``scene_id`` is a sequence+frame record.
    # Sequences are reused across official splits, but frames/RGBD pairs are not.
    train_val_scenes = {str(row["scene_id"]) for row in train + validation}
    annotations = json.loads((ROOT / "crog_reproduction/OCID-VLG/refer/multiple/test_expressions.json").read_text())["data"]
    crog_by_id: dict[int, dict[str, Any]] = {}
    with crog_jsonl.open() as stream:
        for line in stream:
            row = json.loads(line)
            crog_by_id[int(row["sample_index"])] = row
    if len(crog_by_id) != 17749:
        raise ValueError("unexpected CROG prediction count")
    manifest = []
    failures = []
    for source_row in samples:
        sid = str(source_row["sample_id"])
        label = labels[sid]
        try:
            query_id = int(sid[1:8])
            crog = crog_by_id[query_id]
            official = annotations[query_id]
            assertions = {
                "question_index": int(official["question_index"]) == query_id,
                "scene": official["image_filename"] == source_row["scene_id"] == crog["scene_id"],
                "expression": official["question"] == source_row["language"] == crog["language_instruction"],
                "object": int(official["answer"]) == int(label["target_object_id"]) == int(crog["obj_id"]),
                "rgb_path": Path(source_row["source_rgb_path"]).resolve() == Path(crog["image_path"]).resolve(),
                "depth_path": Path(source_row["source_depth_path"]).resolve() == Path(crog["depth_path"]).resolve(),
                "grasps": official["grasps"] == label["gt_grasp_rectangles"],
            }
            if not all(assertions.values()):
                raise ValueError(f"identity cross-check failed: {assertions}")
            with Image.open(source_row["source_rgb_path"]) as image:
                width, height = image.size
            with Image.open(crog["mask_path"]) as mask:
                if mask.size != (width, height):
                    raise ValueError("official instance mask/image shape mismatch")
            scene_record = str(source_row["scene_id"])
            scene_family = scene_record.split(",", 1)[0]
            leakage = (
                source_row["source_rgb_sha256"] in train_val_rgb
                or source_row["rgbd_pair_sha256"] in train_val_rgbd
                or scene_record in train_val_scenes
            )
            if leakage:
                raise ValueError("train_or_validation_overlap")
            normalised = " ".join(str(source_row["language"]).lower().split())
            box = official.get("box") or [0, 0, 0, 0]
            manifest.append(
                {
                    "sample_id": sid,
                    "scene_id": scene_record,
                    "frame_id": str(Path(source_row["source_rgb_path"]).resolve()),
                    "scene_family": scene_family,
                    "query_id": query_id,
                    "target_instance_id": int(label["target_object_id"]),
                    "expression": str(source_row["language"]),
                    "expression_type": expression_type(official),
                    "rgb_path": str(Path(source_row["source_rgb_path"]).resolve()),
                    "depth_path": str(Path(source_row["source_depth_path"]).resolve()),
                    "gt_mask_path": str(Path(crog["mask_path"]).resolve()),
                    "gt_grasp_path": str(Path(label["official_annotations_path"]).resolve()),
                    "gt_grasp_list_json": json.dumps(label["gt_grasp_rectangles"], separators=(",", ":")),
                    "image_width": width,
                    "image_height": height,
                    "object_category": object_category(official["target"]),
                    "split": "strict_common_heldout_test",
                    "image_sha256": str(source_row["source_rgb_sha256"]),
                    "depth_sha256": str(source_row["source_depth_sha256"]),
                    "rgbd_pair_sha256": str(source_row["rgbd_pair_sha256"]),
                    "expression_normalized_sha256": hashlib.sha256(normalised.encode()).hexdigest(),
                    "predicted_hifics_mask_path": str(Path(source_row["predicted_mask_path"]).resolve()),
                    "predicted_hifics_mask_sha256": str(source_row["predicted_mask_sha256"]),
                    "predicted_hifics_probability_path": str(Path(source_row["predicted_probability_path"]).resolve()),
                    "predicted_hifics_probability_sha256": str(source_row["predicted_probability_sha256"]),
                    "gt_grasp_count": int(label["gt_grasp_count"]),
                    "scene_instance_count": int(crog.get("scene_instance_count", 0)),
                    "target_bbox_x": int(box[0]),
                    "target_bbox_y": int(box[1]),
                    "target_bbox_width": int(box[2]),
                    "target_bbox_height": int(box[3]),
                    "inclusion_status": "included",
                    "exclusion_reason": None,
                }
            )
        except Exception as error:
            failures.append({"sample_id": sid, "source": "modular", "exclusion_reason": f"{type(error).__name__}:{error}"})
    included_queries = {int(row["query_id"]) for row in manifest}
    excluded = failures + [
        {
            "sample_id": int(query_id),
            "source": "CROG-original-test",
            "exclusion_reason": "not_in_strict_modular_common_heldout_manifest",
        }
        for query_id in sorted(set(crog_by_id) - included_queries)
    ]
    if len(manifest) != len(samples) or failures:
        raise RuntimeError(f"paired manifest incomplete: {len(manifest)}/{len(samples)}; failures={len(failures)}")
    frame = pd.DataFrame(manifest).sort_values("sample_id").reset_index(drop=True)
    if frame["sample_id"].duplicated().any() or frame["query_id"].duplicated().any():
        raise RuntimeError("paired manifest contains duplicate IDs")
    frame.to_parquet(run / "01_manifest/paired_manifest.parquet", index=False)
    frame.to_csv(run / "01_manifest/paired_manifest.csv", index=False)
    pd.DataFrame(excluded).to_csv(run / "01_manifest/excluded_samples.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pairing_audit = f"""# Pairing audit

- CROG original test: {len(crog_by_id):,}
- Modular held-out test: {len(samples):,}
- Strict three-system paired intersection: {len(frame):,}
- Excluded CROG-only rows: {len(crog_by_id) - len(frame):,}
- Identity failures/missing files/no GT: 0
- Duplicate paired sample/query IDs: 0
- Train/validation overlap by dataset-native scene record (sequence+frame), RGB SHA-256, or RGBD SHA-256: 0
- Official splits reuse sequence families; therefore sequence-family isolation is not claimed and `scene_family` is retained as a sensitivity stratum.
- Pairing key: numeric original CROG question index parsed from `qNNNNNNN_*`, followed by exact scene, expression, object, RGB/depth path, and GT-grasp cross-checks.
- A scene/expression/object tuple alone is not used because repeated queries make it ambiguous.
- Main and broader paired intersection are identical: all {len(frame):,} rows are strict common-held-out.
"""
    (run / "01_manifest/pairing_audit.md").write_text(pairing_audit)
    source_audit = f"""# Source and environment audit

This run compares three concrete systems on the same paired OCID-VLG samples under one evaluator, each in its native input configuration. It is not a causal architecture comparison.

## Models and inputs

- CROG-native: RGB + exact language; native sigmoid quality peaks (K=5, threshold 0.4, min-distance 2), sigmoid width ×100 px, 20 px rectangle height, no added NMS or reranking.
- HiFi-CS shared frontend: RGB + exact language, repeated/hierarchical FiLM, 352×352 preprocessing, frozen validation-selected checkpoint, sigmoid mask threshold 0.5 from the frozen local config. Its cached mask path and SHA-256 are shared byte-for-byte by G1 and C1.
- G1: validation-selected dilated target crop (10%, 224×224), depth+RGB ordered `[depth,R,G,B]`, official mean normalization, fine-tuned GR-ConvNet. Native Gaussian postprocessing σ=(2,2,1), q>0.2, min-distance 20, up to 100 peaks, width×150, rectangle short side=jaw width/2.
- C1: validation-selected dilated target crop (15%, 300×300), normalized depth only, fine-tuned GG-CNN2. It uses the same native decoder as G1.

## No-rerank audit

The older modular formal predictions are not reused as outcomes: their backend applied a probability-gated quality map, candidate mask-support rescoring and project NMS. This run reruns G1/C1 without those operations. The mask remains part of the already selected crop/input-conditioning contract only.

## Environment and lineage

Machine-readable details are in `environment.json`, `git_state.json`, `checkpoint_registry.csv`, and `source_run_hashes.json`. Dirty worktrees are recorded and never cleaned; exact source snapshots and hashes, rather than cleanliness claims, define this run.
"""
    (run / "00_audit/SOURCE_AND_ENVIRONMENT_AUDIT.md").write_text(source_audit)
    (run / ".RUN_ACTIVE").write_text(f"pid={os.getpid()}\nstarted={datetime.now().astimezone().isoformat()}\n")
    (run / "config/run_config.json").write_text(
        json.dumps(
            {
                "protocol": "fair-4dof-no-rerank-v1",
                "paired_n": len(frame),
                "methods": METHODS,
                "statistics_seed": 20260806,
                "bootstrap_replicates": 10000,
                "source_run": str(source_run),
                "crog_predictions": str(crog_jsonl),
            }, indent=2
        ) + "\n"
    )
    print(run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--crog-predictions", type=Path, required=True)
    parser.add_argument("--crog-features", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
