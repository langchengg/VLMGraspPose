#!/usr/bin/env python3
"""Run the frozen hierarchical/single-FiLM modular comparison to completion."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
HIER_RUN = REPO_ROOT / "runs" / "hifics_ocidvlg_hierfilm_20260727_214615"
OLD_RUN = REPO_ROOT / "runs" / "hifics_ocidvlg_20260711_112921"
TEST_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "data_audit"
    / "frozen_manifests"
    / "ocidvlg_unique_test.json"
)
ANNOTATIONS = PROJECT_ROOT / "crog_reproduction" / "OCID-VLG" / "refer" / "unique" / "test_expressions.json"
DATASET_ROOT = PROJECT_ROOT / "crog_reproduction" / "OCID-VLG"
TEMPLATE_BUNDLES = OLD_RUN / "anygrasp_input_predicted_mask"
OLD_PREDICTIONS = OLD_RUN / "predictions"
MODEL_DIR = REPO_ROOT / "models" / "gqcnn-official" / "GQCNN-2.1"
CLIP_CACHE = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
DOCKER_IMAGE = "vlmgrasp/gqcnn-score:1.3.0"
DOCKER_IMAGE_ID = (
    "sha256:3d1158ca83197d55808454b718d0a328d3f27c57c80baaaea7031e21a9134ebd"
)
EXPECTED_TEST_SHA = (
    "915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409"
)
EXPECTED_CHECKPOINT_SHA = (
    "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
)
EXPECTED_OLD_CHECKPOINT_SHA = (
    "436a54ecc159a36664f55f762463c54fc9b082f44205cee8020bed59fb5280d0"
)
EXPECTED_MODEL_CONFIG_SHA = (
    "eb5bc17089a39bd8fe6c801010c25a6a79a898d64181180feb5cf69aa630ff6f"
)
EXPECTED_MODEL_MANIFEST_SHA = (
    "8201961abe3a09d90c6c66e582a3bfeb181d7095a2ebcc3a9d90e68fc12e8614"
)
EXPECTED_CLIP_SHA = (
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
)
EXPECTED_SAMPLES = 7675
CODE_FILES = (
    "hifics/tools/export_modular_standard_masks.py",
    "hifics/models/hifics.py",
    "hifics/datasets/dataloader.py",
    "scripts/run_hifics_dexnet_candidates.py",
    "scripts/run_full_gqcnn_scoring.py",
    "scripts/evaluate_corrected_gqcnn_pipeline.py",
    "scripts/finalize_hierfilm_modular_experiment.py",
    "scripts/run_hierfilm_modular_experiment.py",
    "src/grasping/ocid_vlg_grasp_adapter.py",
    "src/grasping/mask_processing.py",
    "src/grasping/dexnet_adapter.py",
    "src/grasping/dexnet_candidate_generator.py",
    "src/grasping/dexnet_run_reliability.py",
    "src/grasping/grasp_serialization.py",
    "src/grasping/gqcnn_full_scoring.py",
    "src/grasping/geometric_ranker.py",
    "third_party/gqcnn-official/gqcnn/__init__.py",
    "third_party/gqcnn-official/gqcnn/grasping/__init__.py",
    "third_party/gqcnn-official/gqcnn/grasping/grasp.py",
    "third_party/gqcnn-official/gqcnn/grasping/image_grasp_sampler.py",
    "configs/dexnet_candidates_formal_no_refinement.yaml",
    "configs/dexnet_grasp_consistency_corrected.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def run_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def stable_sample_id(scene_id: str, question_index: int) -> str:
    identity = f"{scene_id}\t{int(question_index)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"q{int(question_index):07d}_{digest}"


def path_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def model_directory_identity(path: Path) -> dict[str, Any]:
    files = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
            "size_bytes": item.stat().st_size,
        }
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]
    encoded = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "path": str(path.resolve()),
        "files": files,
        "file_count": len(files),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def current_code_hashes() -> dict[str, str]:
    values = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in CODE_FILES
    }
    crog = PROJECT_ROOT / "crog_reproduction" / "CROG" / "utils" / "grasp_metrics.py"
    values[str(crog)] = sha256_file(crog)
    return values


def git_evidence(root: Path) -> dict[str, Any]:
    commit = run_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = run_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ]
    )
    diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--binary"], stderr=subprocess.STDOUT
    )
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_size_bytes": len(diff),
    }


def initialize_run(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(f"fresh formal run path already exists: {run_dir}")
    if sha256_file(TEST_MANIFEST) != EXPECTED_TEST_SHA:
        raise ValueError("frozen test manifest SHA mismatch")
    checkpoint = HIER_RUN / "checkpoints" / "best.pth"
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("hierarchical best checkpoint SHA mismatch")
    old_export = json.loads(
        (OLD_PREDICTIONS / "export_manifest.json").read_text(encoding="utf-8")
    )
    old_checkpoint = Path(old_export["checkpoint"]).resolve()
    if sha256_file(old_checkpoint) != EXPECTED_OLD_CHECKPOINT_SHA:
        raise ValueError("old single-FiLM checkpoint SHA mismatch")
    model_identity = model_directory_identity(MODEL_DIR)
    if sha256_file(MODEL_DIR / "config.json") != EXPECTED_MODEL_CONFIG_SHA:
        raise ValueError("GQCNN-2.1 config SHA mismatch")
    if model_identity["manifest_sha256"] != EXPECTED_MODEL_MANIFEST_SHA:
        raise ValueError("GQCNN-2.1 full model manifest SHA mismatch")
    if sha256_file(CLIP_CACHE) != EXPECTED_CLIP_SHA:
        raise ValueError("OpenAI CLIP ViT-B/16 weight SHA mismatch")
    docker_identity = run_output(
        ["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Id}}"]
    )
    if docker_identity != DOCKER_IMAGE_ID:
        raise ValueError(f"Docker image ID mismatch: {docker_identity}")
    for path in (
        run_dir,
        run_dir / "logs",
        run_dir / "masks",
        run_dir / "candidates",
        run_dir / "scores",
        run_dir / "evaluation",
        run_dir / "reports",
        run_dir / "artifacts" / "qualitative_audit",
        run_dir / "source_snapshot",
    ):
        path.mkdir(parents=True, exist_ok=True)
    code_hashes = current_code_hashes()
    for relative in CODE_FILES:
        source = REPO_ROOT / relative
        destination = run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    crog_source = (
        PROJECT_ROOT / "crog_reproduction" / "CROG" / "utils" / "grasp_metrics.py"
    )
    shutil.copy2(crog_source, run_dir / "source_snapshot" / "crog_grasp_metrics.py")
    manifest = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    if len(manifest) != EXPECTED_SAMPLES:
        raise ValueError("frozen test record count mismatch")
    annotations_payload = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["data"]
    annotations_by_question = {
        int(row["question_index"]): row for row in annotations_payload
    }
    test_sample_ids = {
        stable_sample_id(row["scene_id"], int(row["question_index"]))
        for row in manifest
    }
    test_question_ids = {int(row["question_index"]) for row in manifest}
    if (
        len(test_sample_ids) != EXPECTED_SAMPLES
        or len(test_question_ids) != EXPECTED_SAMPLES
    ):
        raise ValueError("frozen test sample/question identities are not unique")
    split_overlap: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        split_path = (
            REPO_ROOT
            / "artifacts"
            / "data_audit"
            / "frozen_manifests"
            / f"ocidvlg_unique_{split}.json"
        )
        split_records = json.loads(split_path.read_text(encoding="utf-8"))
        split_samples = {
            stable_sample_id(row["scene_id"], int(row["question_index"]))
            for row in split_records
        }
        split_questions = {int(row["question_index"]) for row in split_records}
        split_overlap[split] = {
            "sample_id_overlap": len(test_sample_ids & split_samples),
            "question_index_overlap": len(test_question_ids & split_questions),
        }
    if any(
        overlap["sample_id_overlap"] for overlap in split_overlap.values()
    ):
        raise ValueError(f"frozen test manifest crosses train/val: {split_overlap}")
    input_path = run_dir / "input_manifest.csv"
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "sample_index",
            "sample_id",
            "question_index",
            "scene_id",
            "query",
            "rgb_path",
            "depth_path",
            "gt_mask_path",
            "target_object_id",
            "official_gt_grasp_count",
            "native_rgb_path",
            "native_depth_path",
            "official_instance_mask_path",
            "camera_intrinsics_path",
        )
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(manifest):
            sample_id = stable_sample_id(
                row["scene_id"], int(row["question_index"])
            )
            annotation = annotations_by_question.get(int(row["question_index"]))
            if (
                annotation is None
                or annotation["image_filename"] != row["scene_id"]
                or annotation["question"] != row["text"]
            ):
                raise ValueError(
                    f"official annotation join mismatch: {row['question_index']}"
                )
            template = TEMPLATE_BUNDLES / sample_id
            template_metadata_path = template / "metadata.json"
            intrinsics_path = template / "intrinsics.json"
            if not template_metadata_path.is_file() or not intrinsics_path.is_file():
                raise FileNotFoundError(
                    f"template bundle/intrinsics missing: {sample_id}"
                )
            template_metadata = json.loads(
                template_metadata_path.read_text(encoding="utf-8")
            )
            if (
                template_metadata["sample_id"] != sample_id
                or int(template_metadata["question_index"])
                != int(row["question_index"])
                or template_metadata["scene_id"] != row["scene_id"]
                or template_metadata["query"] != row["text"]
            ):
                raise ValueError(f"template stable join mismatch: {sample_id}")
            _, image_name = row["scene_id"].split(",", 1)
            instance_mask = (
                Path(template_metadata["source_rgb"]).parent.parent
                / "seg_mask_instances_combi"
                / image_name
            )
            required_assets = (
                REPO_ROOT / "hifics" / Path(row["rgb_path"]),
                REPO_ROOT / "hifics" / Path(row["depth_path"]),
                REPO_ROOT / "hifics" / Path(row["mask_path"]),
                Path(template_metadata["source_rgb"]),
                Path(template_metadata["source_depth"]),
                instance_mask,
                intrinsics_path,
            )
            missing_assets = [str(path) for path in required_assets if not path.is_file()]
            if missing_assets:
                raise FileNotFoundError(
                    f"stable sample assets missing for {sample_id}: {missing_assets}"
                )
            writer.writerow(
                {
                    "sample_index": index,
                    "sample_id": sample_id,
                    "question_index": int(row["question_index"]),
                    "scene_id": row["scene_id"],
                    "query": row["text"],
                    "rgb_path": row["rgb_path"],
                    "depth_path": row["depth_path"],
                    "gt_mask_path": row["mask_path"],
                    "target_object_id": int(annotation["answer"]),
                    "official_gt_grasp_count": len(annotation["grasps"]),
                    "native_rgb_path": template_metadata["source_rgb"],
                    "native_depth_path": template_metadata["source_depth"],
                    "official_instance_mask_path": str(instance_mask),
                    "camera_intrinsics_path": str(intrinsics_path),
                }
            )
    protocol = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FORMAL_EXECUTION",
        "sample_count": EXPECTED_SAMPLES,
        "mask": {
            "hierarchical_checkpoint": str(checkpoint),
            "model_input_output": [352, 352],
            "foreground_probability": "sigmoid(-background_logit)",
            "threshold": 0.5,
            "threshold_comparison": ">=",
            "native_mapping": "nearest-neighbor from 352x352 binary mask",
            "author_compatible_mask_forbidden": True,
            "clip": {
                "implementation": "OpenAI CLIP Python package clip==1.0",
                "backbone": "ViT-B/16",
                "weight_path": str(CLIP_CACHE),
                "weight_sha256": EXPECTED_CLIP_SHA,
                "official_url": (
                    "https://openaipublic.azureedge.net/clip/models/"
                    f"{EXPECTED_CLIP_SHA}/ViT-B-16.pt"
                ),
                "frozen": True,
            },
        },
        "candidate_generation": {
            "sampler": "official AntipodalDepthImageGraspSampler at pinned GQ-CNN v1.3.0 commit",
            "base_seed": 42,
            "sample_seed_mode": "stable-sha256",
            "seed_namespace": "hierfilm-modular-formal-v1",
            "sample_seed_derivation": "uint64_be(sha256(namespace\\0base_seed\\0stable_sample_id)[:8]) mod (2**32-1)",
            "requested_candidates": 256,
            "nms_center_distance_px": 8.0,
            "nms_angle_distance_deg": 15.0,
            "mask_component_filter": False,
            "retain_largest_component": False,
            "mask_erode_px": 0,
            "mask_dilate_px": 0,
            "contact_support_dilation_px": 0,
            "valid_depth_intersection": True,
            "ground_truth_used": False,
        },
        "gqcnn": {
            "model": "GQCNN-2.1",
            "model_config_sha256": EXPECTED_MODEL_CONFIG_SHA,
            "docker_image": DOCKER_IMAGE,
            "docker_image_id": DOCKER_IMAGE_ID,
            "ranking": "raw full-precision Q descending",
            "tie_break": "candidate_id ascending for exact Q ties",
            "crop_height_px": 96,
            "crop_width_px": 96,
            "network_input_height_px": 32,
            "network_input_width_px": 32,
            "inpaint_rescale_factor": 0.5,
            "input_dtype": "float32",
            "device": "CPU",
            "depth_normalization": (
                "official GQ-CNN v1.3.0 GraspQualityFunction preprocessing "
                "using the frozen model mean/std tensors"
            ),
            "quality_interpretation": (
                "model-predicted grasp robustness/quality score; not a "
                "calibrated physical success probability"
            ),
            "cem": False,
            "ground_truth_or_mask_iou_used": False,
        },
        "evaluation": {
            "version": "corrected_geometric_v2",
            "polygon_coordinates": "[x,y]",
            "rasterization": "row=y,column=x",
            "angle_periodicity_deg": 180,
            "iou_threshold": 0.25,
            "iou_comparison": ">",
            "angle_threshold_deg": 30.0,
            "angle_comparison": "<=",
            "same_gt_joint_match": True,
            "primary_denominator": EXPECTED_SAMPLES,
            "empty_and_failed_count_as_failure": True,
            "physical_success_claim": False,
        },
        "comparison": {
            "only_planned_pipeline_input_change": (
                "old single-FiLM predicted binary mask -> five-stage repeated-FiLM Standard binary mask"
            ),
            "old_baseline": (
                "recover model-resolution old binary predictions, map them with the "
                "same nearest adapter, and rerun the frozen current downstream"
            ),
            "pure_repeated_film_causal_claim_allowed": False,
        },
        "bootstrap": {
            "cluster": "scene_id/RGB frame",
            "replicates": 10000,
            "seed": 20260728,
        },
    }
    atomic_text(
        run_dir / "frozen_protocol.yaml",
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
    )
    run_manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "inputs": {
            "hierfilm_checkpoint": path_identity(checkpoint),
            "test_manifest": path_identity(TEST_MANIFEST),
            "official_annotations": path_identity(ANNOTATIONS),
            "old_checkpoint": path_identity(old_checkpoint),
            "clip_vit_b16_weights": path_identity(CLIP_CACHE),
            "old_prediction_export_manifest": path_identity(
                OLD_PREDICTIONS / "export_manifest.json"
            ),
            "dexnet_config": path_identity(
                REPO_ROOT
                / "configs"
                / "dexnet_candidates_formal_no_refinement.yaml"
            ),
            "corrected_evaluation_config": path_identity(
                REPO_ROOT
                / "configs"
                / "dexnet_grasp_consistency_corrected.yaml"
            ),
            "gqcnn_model_config": path_identity(MODEL_DIR / "config.json"),
            "gqcnn_model_directory": model_identity,
            "frozen_train_manifest": path_identity(
                REPO_ROOT
                / "artifacts"
                / "data_audit"
                / "frozen_manifests"
                / "ocidvlg_unique_train.json"
            ),
            "frozen_val_manifest": path_identity(
                REPO_ROOT
                / "artifacts"
                / "data_audit"
                / "frozen_manifests"
                / "ocidvlg_unique_val.json"
            ),
        },
        "code_hashes": code_hashes,
        "git": {
            "parent": git_evidence(PROJECT_ROOT),
            "hifics": git_evidence(REPO_ROOT),
            "gqcnn_official_checkout": git_evidence(
                REPO_ROOT / "third_party" / "gqcnn-official"
            ),
            "corrected_crog_checkout": git_evidence(
                PROJECT_ROOT / "crog_reproduction" / "CROG"
            ),
        },
        "immutable_old_artifacts": {
            "template_bundles": str(TEMPLATE_BUNDLES),
            "old_predictions": str(OLD_PREDICTIONS),
            "historical_candidates": str(
                REPO_ROOT / "outputs" / "dexnet_candidates_full_hifics"
            ),
            "historical_scores": str(
                REPO_ROOT / "outputs" / "gqcnn_scored_full_hifics"
            ),
        },
        "input_audit": {
            "test_samples": len(test_sample_ids),
            "unique_sample_ids": len(test_sample_ids),
            "unique_question_indices": len(test_question_ids),
            "unique_scenes": len({row["scene_id"] for row in manifest}),
            "official_annotation_joins": EXPECTED_SAMPLES,
            "template_bundle_and_intrinsics_joins": EXPECTED_SAMPLES,
            "split_overlap": split_overlap,
            "question_index_scope": (
                "split-local; overlap across splits is expected and is not "
                "used as a cross-split sample identity"
            ),
            "join_key": "stable SHA-256 sample_id(scene_id, question_index)",
            "directory_or_dataframe_order_used": False,
        },
    }
    atomic_json(run_dir / "run_manifest.json", run_manifest)
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": run_output(
            [
                str(REPO_ROOT / "hifics" / ".venv" / "bin" / "python"),
                "-c",
                "import torch; print(torch.__version__)",
            ]
        ),
        "mps_available": run_output(
            [
                str(REPO_ROOT / "hifics" / ".venv" / "bin" / "python"),
                "-c",
                "import torch; print(torch.backends.mps.is_available())",
            ]
        ),
        "docker_server": run_output(
            ["docker", "info", "--format", "{{.ServerVersion}}"]
        ),
        "docker_image": DOCKER_IMAGE,
        "docker_image_id": docker_identity,
        "gqcnn_runtime": {
            "platform": "linux/amd64",
            "python": "3.7.17",
            "tensorflow": "1.15.0",
            "device": "CPU",
            "network": "none",
        },
        "analysis_python": run_output(
            ["/opt/anaconda3/bin/python", "-c", "import sys; print(sys.version)"]
        ),
        "disk": run_output(["df", "-h", str(run_dir)]),
        "missing_skill_capabilities": [
            "bash_exec",
            "memory.list_recent",
            "memory.search",
            "artifact.*",
            "record_main_experiment",
        ],
    }
    atomic_json(run_dir / "environment.json", environment)
    plan = f"""# PLAN — {run_dir.name}

## Goal

Run both recoverable segmenters through the same frozen mask-to-depth adapter,
sample-derived Dex-Net sampler, official GQCNN-2.1 scorer, and corrected offline
rectangle evaluator for all 7,675 official unique test expressions.

## Pre-registered stages

1. Regenerate hierarchical Standard masks and exact 352×352 metrics.
2. Recover old model-resolution binary masks and apply the same nearest mapping.
3. Regenerate all Dex-Net raw and post-NMS candidates independently for each mask.
4. Score every post-NMS candidate with frozen GQCNN-2.1 raw Q.
5. Add GT correctness only after ranking with corrected_geometric_v2.
6. Independently recompute, pair, bootstrap, visualize, and report.

No test-derived tuning, fallback mask, candidate reuse, CEM, reranker, or training
is permitted.
"""
    atomic_text(run_dir / "PLAN.md", plan)
    checklist = """# CHECKLIST

- [x] checkpoint / manifest / model / code hashes frozen
- [x] protocol frozen before formal execution
- [ ] 7,675 hierarchical masks regenerated
- [ ] hierarchical Standard metrics exactly reproduced
- [ ] old masks recovered and mapped with current nearest adapter
- [ ] both Dex-Net candidate runs complete with zero technical failures
- [ ] both GQCNN-2.1 scoring runs complete with finite Q values
- [ ] corrected evaluation complete for both pipelines
- [ ] paired McNemar and clustered bootstrap complete
- [ ] at least 25 qualitative cases visually audited
- [ ] independent recomputation passes
- [ ] required artifact manifest and COMPLETED marker written
"""
    atomic_text(run_dir / "CHECKLIST.md", checklist)


def validate_frozen_code(run_dir: Path) -> None:
    recorded = json.loads((run_dir / "run_manifest.json").read_text())["code_hashes"]
    observed = current_code_hashes()
    if observed != recorded:
        changed = sorted(
            key
            for key in set(recorded) | set(observed)
            if recorded.get(key) != observed.get(key)
        )
        atomic_json(
            run_dir / "INVALID",
            {
                "reason": "frozen code changed after formal initialization",
                "changed": changed,
                "time_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise RuntimeError(f"frozen code drift: {changed}")


def heartbeat(run_dir: Path, stage: str, process: subprocess.Popen | None = None) -> None:
    atomic_json(
        run_dir / "heartbeat.json",
        {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "driver_pid": os.getpid(),
            "stage": stage,
            "child_pid": None if process is None else process.pid,
            "child_alive": None if process is None else process.poll() is None,
        },
    )


def run_stage(
    run_dir: Path,
    stage: str,
    command: list[str],
    *,
    complete: callable,
) -> None:
    validate_frozen_code(run_dir)
    if complete():
        heartbeat(run_dir, f"{stage}:verified_existing")
        return
    stdout_path = run_dir / "logs" / f"{stage}.stdout.log"
    stderr_path = run_dir / "logs" / f"{stage}.stderr.log"
    atomic_json(
        run_dir / "logs" / f"{stage}.command.json",
        {
            "stage": stage,
            "command": command,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    started = time.time()
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        last_heartbeat = 0.0
        while process.poll() is None:
            now = time.time()
            if now - last_heartbeat >= 30:
                heartbeat(run_dir, stage, process)
                last_heartbeat = now
            time.sleep(5)
        return_code = process.wait()
    atomic_json(
        run_dir / "logs" / f"{stage}.result.json",
        {
            "stage": stage,
            "return_code": return_code,
            "elapsed_seconds": time.time() - started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    if return_code != 0:
        raise RuntimeError(
            f"stage {stage} failed with exit {return_code}; see {stderr_path}"
        )
    if not complete():
        raise RuntimeError(f"stage {stage} returned zero but completion validation failed")


def json_complete(path: Path, expected: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        return all(payload.get(key) == value for key, value in expected.items())
    except Exception:
        return False


def candidate_counts(root: Path) -> tuple[int, int, int]:
    with (root / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    nonempty = sum(int(row["post_nms_count"]) > 0 for row in rows)
    empty = len(rows) - nonempty
    candidates = sum(int(row["post_nms_count"]) for row in rows)
    return nonempty, empty, candidates


def candidate_complete(root: Path) -> bool:
    path = root / "progress.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        terminal = int(payload["success_nonempty"]) + int(payload["success_empty"])
        return (
            int(payload["total_expected"]) == EXPECTED_SAMPLES
            and terminal == EXPECTED_SAMPLES
            and int(payload["failed"]) == 0
            and int(payload["remaining"]) == 0
        )
    except Exception:
        return False


def score_command(
    run_dir: Path,
    pipeline: str,
    nonempty: int,
    empty: int,
    candidates: int,
) -> list[str]:
    candidate_root = run_dir / "candidates" / pipeline
    scored_root = run_dir / "scores" / pipeline
    scored_root.mkdir(parents=True, exist_ok=True)
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        f"{run_dir.name}-{pipeline}-gqcnn",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--init",
        "-v",
        f"{REPO_ROOT}:/workspace/HiFi_reproduction:ro",
        "-v",
        f"{candidate_root}:/candidates:ro",
        "-v",
        f"{MODEL_DIR}:/models/GQCNN-2.1:ro",
        "-v",
        f"{scored_root}:/scored:rw",
        "-w",
        "/workspace/HiFi_reproduction",
        DOCKER_IMAGE,
        "python",
        "scripts/run_full_gqcnn_scoring.py",
        "--candidate-root",
        "/candidates",
        "--output-root",
        "/scored",
        "--model-dir",
        "/models/GQCNN-2.1",
        "--model-name",
        "GQCNN-2.1",
        "--docker-image",
        DOCKER_IMAGE,
        "--docker-image-id",
        DOCKER_IMAGE_ID,
        "--expected-model-config-hash",
        EXPECTED_MODEL_CONFIG_SHA,
        "--seed",
        "42",
        "--resume",
        "--verify-existing",
        "--retry-failed",
        "--batch-size",
        "100",
        "--log-every",
        "100",
        "--expected-samples",
        str(EXPECTED_SAMPLES),
        "--expected-nonempty",
        str(nonempty),
        "--expected-empty",
        str(empty),
        "--expected-candidates",
        str(candidates),
    ]


def write_sample_status(run_dir: Path) -> None:
    source = (
        run_dir
        / "evaluation"
        / "hierfilm_per_sample_pipeline_metrics.csv"
    )
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError("hierfilm status source count mismatch")
    text = "".join(
        json.dumps(
            {
                "sample_index": int(row["sample_index"]),
                "sample_id": row["sample_id"],
                "terminal_status": row["terminal_status"],
                "technical_failure": row["technical_failure"].lower() == "true",
                "failure_category": row["failure_category"],
            },
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )
    atomic_text(run_dir / "sample_status.jsonl", text)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if (run_dir / "COMPLETED").is_file():
        print(
            json.dumps(
                {
                    "status": "ALREADY_COMPLETED",
                    "run_dir": str(run_dir),
                    "completed": str(run_dir / "COMPLETED"),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    if args.initialize:
        initialize_run(run_dir)
        if args.initialize_only:
            heartbeat(run_dir, "INITIALIZED_PROTOCOL_FROZEN")
            print(
                json.dumps(
                    {
                        "status": "INITIALIZED_PROTOCOL_FROZEN",
                        "run_dir": str(run_dir),
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0
    elif args.initialize_only:
        raise ValueError("--initialize-only requires --initialize")
    if not (run_dir / "frozen_protocol.yaml").is_file():
        raise FileNotFoundError("formal run is not initialized")
    if (run_dir / "INVALID").exists():
        raise RuntimeError("formal run is marked INVALID")
    lock_path = run_dir / "formal_pipeline.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"another formal pipeline process owns {lock_path}"
        ) from exc
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "acquired_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    lock_stream.flush()
    heartbeat(run_dir, "driver_start")
    hific_python = str(REPO_ROOT / "hifics" / ".venv" / "bin" / "python")
    candidate_python = str(REPO_ROOT / ".venv-gqcnn" / "bin" / "python")
    analysis_python = "/opt/anaconda3/bin/python"
    common_mask = [
        "--frozen-manifest",
        str(TEST_MANIFEST),
        "--official-annotations",
        str(ANNOTATIONS),
        "--template-bundle-root",
        str(TEMPLATE_BUNDLES),
        "--expected-samples",
        str(EXPECTED_SAMPLES),
        "--resume",
    ]
    run_stage(
        run_dir,
        "masks_hierfilm",
        [
            hific_python,
            "hifics/tools/export_modular_standard_masks.py",
            "--pipeline",
            "hierfilm",
            "--output-root",
            str(run_dir / "masks" / "hierfilm"),
            *common_mask,
            "--hierfilm-run",
            str(HIER_RUN),
            "--device",
            "mps",
            "--batch-size",
            "32",
        ],
        complete=lambda: json_complete(
            run_dir / "masks" / "hierfilm" / "MASKS_COMPLETED.json",
            {"pipeline": "hierfilm"},
        ),
    )
    run_stage(
        run_dir,
        "masks_old_singlefilm",
        [
            hific_python,
            "hifics/tools/export_modular_standard_masks.py",
            "--pipeline",
            "old_singlefilm",
            "--output-root",
            str(run_dir / "masks" / "old_singlefilm"),
            *common_mask,
            "--old-predictions-root",
            str(OLD_PREDICTIONS),
        ],
        complete=lambda: json_complete(
            run_dir / "masks" / "old_singlefilm" / "MASKS_COMPLETED.json",
            {"pipeline": "old_singlefilm"},
        ),
    )
    for pipeline in ("hierfilm", "old_singlefilm"):
        candidate_root = run_dir / "candidates" / pipeline
        run_stage(
            run_dir,
            f"candidates_{pipeline}",
            [
                candidate_python,
                "scripts/run_hifics_dexnet_candidates.py",
                "--dataset-root",
                str(DATASET_ROOT),
                "--mask-root",
                str(run_dir / "masks" / pipeline / "bundles"),
                "--output-dir",
                str(candidate_root),
                "--config",
                str(
                    REPO_ROOT
                    / "configs"
                    / "dexnet_candidates_formal_no_refinement.yaml"
                ),
                "--mode",
                "candidate-only",
                "--num-candidates",
                "256",
                "--top-k",
                "30",
                "--seed",
                "42",
                "--sample-seed-mode",
                "stable-sha256",
                "--seed-namespace",
                "hierfilm-modular-formal-v1",
                "--resume",
                "--verify-existing",
                "--retry-failures",
                "--visualize-policy",
                "none",
                "--status-every",
                "25",
                "--checkpoint-every",
                "25",
                "--max-failures",
                "1",
            ],
            complete=lambda root=candidate_root: candidate_complete(root),
        )
        nonempty, empty, candidates = candidate_counts(candidate_root)
        scored_root = run_dir / "scores" / pipeline
        run_stage(
            run_dir,
            f"scores_{pipeline}",
            score_command(
                run_dir, pipeline, nonempty, empty, candidates
            ),
            complete=lambda root=scored_root, expected=candidates: json_complete(
                root / "progress.json",
                {
                    "terminal_samples": EXPECTED_SAMPLES,
                    "failed_samples": 0,
                    "scored_candidates": expected,
                },
            ),
        )
        evaluation_path = (
            run_dir / "evaluation" / f"{pipeline}_pipeline_metrics.json"
        )
        run_stage(
            run_dir,
            f"evaluate_{pipeline}",
            [
                analysis_python,
                "scripts/evaluate_corrected_gqcnn_pipeline.py",
                "--pipeline",
                pipeline,
                "--candidate-root",
                str(candidate_root),
                "--scored-root",
                str(scored_root),
                "--mask-metadata",
                str(
                    run_dir
                    / "masks"
                    / pipeline
                    / "per_sample_mask_metadata.csv"
                ),
                "--official-annotations",
                str(ANNOTATIONS),
                "--evaluation-config",
                str(
                    REPO_ROOT
                    / "configs"
                    / "dexnet_grasp_consistency_corrected.yaml"
                ),
                "--crog-root",
                str(PROJECT_ROOT / "crog_reproduction" / "CROG"),
                "--output-dir",
                str(run_dir / "evaluation"),
                "--expected-samples",
                str(EXPECTED_SAMPLES),
            ],
            complete=lambda path=evaluation_path: json_complete(
                path,
                {
                    "primary_denominator": EXPECTED_SAMPLES,
                    "technical_failure_count": 0,
                },
            ),
        )
    write_sample_status(run_dir)
    run_stage(
        run_dir,
        "finalize",
        [
            analysis_python,
            "scripts/finalize_hierfilm_modular_experiment.py",
            "--run-dir",
            str(run_dir),
            "--official-annotations",
            str(ANNOTATIONS),
            "--bootstrap-replicates",
            "10000",
            "--bootstrap-seed",
            "20260728",
        ],
        complete=lambda: json_complete(
            run_dir / "COMPLETED",
            {
                "status": "SUCCESS",
                "samples_per_pipeline": EXPECTED_SAMPLES,
                "technical_failures": 0,
                "independent_verification_passed": True,
            },
        ),
    )
    checklist = (run_dir / "CHECKLIST.md").read_text(encoding="utf-8")
    checklist = checklist.replace("- [ ]", "- [x]")
    atomic_text(run_dir / "CHECKLIST.md", checklist)
    heartbeat(run_dir, "COMPLETED")
    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "run_dir": str(run_dir),
                "completed": str(run_dir / "COMPLETED"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
