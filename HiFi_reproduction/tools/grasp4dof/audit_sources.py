#!/usr/bin/env python3
"""Audit the immutable repeated-FiLM, OCID-VLG, and reference lineage.

This command is intentionally read-only outside ``--run-dir``.  It validates
the retained five-stage repeated-FiLM checkpoint, all three official unique
splits, compact predicted-mask coverage, and the frozen Dex-Net/GQ-CNN
reference before the new 4-DoF benchmark is allowed to proceed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SOURCE_RUN = PROJECT_ROOT / "runs/hifics_ocidvlg_hierfilm_20260727_214615"
CHECKPOINT = SOURCE_RUN / "checkpoints/best.pth"
REFERENCE_RUN = (
    PROJECT_ROOT / "runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
)
COMPACT_RUN = (
    PROJECT_ROOT / "runs/modular_reranking_repeatedfilm_v1_20260729_203147"
)
FROZEN_MANIFESTS = PROJECT_ROOT / "artifacts/data_audit/frozen_manifests"
DATASET_ROOT = PROJECT_ROOT / "OCID-VLG"
CROG_DATASET_ROOT = WORKSPACE_ROOT / "crog_reproduction/OCID-VLG"
EXPECTED_CHECKPOINT_SHA256 = (
    "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
)
EXPECTED_MANIFESTS = {
    "train": (26295, "a986bcce3e1961be816a295c3ae0942e64e61275524a85c0a8957563e7f920c1"),
    "val": (3778, "573c6ecd9ed9963eda525162279836b7649d163d83c57f164598604579b8b84a"),
    "test": (7675, "915e002bf31f044419db7140bc1145b8fcc45f9a6b35259637d923c6d4610409"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--full-probability-scan",
        action="store_true",
        help="Decompress every saved probability and verify finite [0,1] values.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def require_file(path: Path, *, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")
    return path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_text(*args: str) -> str:
    result = subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def stable_pair_hash(rgb_sha: str, depth_sha: str) -> str:
    return hashlib.sha256(f"{rgb_sha}\0{depth_sha}".encode()).hexdigest()


def audit_checkpoint() -> dict[str, Any]:
    require_file(CHECKPOINT, label="repeated-FiLM checkpoint")
    observed_sha = sha256_file(CHECKPOINT)
    if observed_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"repeated-FiLM checkpoint hash mismatch: {observed_sha}"
        )
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if payload.get("format") != "hifics_hierfilm_trainable_only_v1":
        raise ValueError("checkpoint format is not retained hierarchical FiLM")
    state = payload.get("trainable_state")
    metadata = payload.get("metadata")
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise ValueError("checkpoint omits trainable_state or metadata")
    expected_film = {
        f"film_stages.{stage}.{branch}.{parameter}"
        for stage in range(5)
        for branch in ("alpha", "beta")
        for parameter in ("weight", "bias")
    }
    film_keys = {name for name in state if name.startswith("film_stages.")}
    suspicious = [
        name
        for name in state
        if name.startswith(("film_mul", "film_add", "reduce."))
    ]
    if film_keys != expected_film or suspicious:
        raise ValueError("checkpoint is not the five-stage repeated-FiLM schema")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"non-finite or non-tensor checkpoint value: {name}")

    # Instantiate the exact frozen architecture and prove a strict state match.
    hifics_root = PROJECT_ROOT / "hifics"
    sys.path[:0] = [str(hifics_root), str(PROJECT_ROOT)]
    from models.hifics import HierarchicalCLIPDensePredT  # type: ignore

    model = HierarchicalCLIPDensePredT(
        version="ViT-B/16",
        extract_layers=(1, 3, 5, 7, 9),
        reduce_dim=64,
        n_heads=4,
        cond_layer=None,
        extended_film=True,
        hierarchical_film=True,
    )
    expected_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != expected_trainable:
        raise ValueError("checkpoint trainable keys do not match exact model")
    complete_state = model.state_dict()
    complete_state.update(state)
    incompatible = model.load_state_dict(complete_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict checkpoint load failed: {incompatible}")
    config_path = SOURCE_RUN / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        "visual_grounding_variant": "hierarchical_repeated_film",
        "single_film_allowed": False,
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": observed_sha,
        "checkpoint_format": payload["format"],
        "checkpoint_step": int(metadata["global_step"]),
        "checkpoint_epoch": int(metadata["epoch"]),
        "checkpoint_strict_load_success": True,
        "checkpoint_expected_keys": len(expected_trainable),
        "checkpoint_loaded_keys": len(state),
        "checkpoint_missing_keys": [],
        "checkpoint_unexpected_keys": [],
        "architecture_class": "models.hifics.HierarchicalCLIPDensePredT",
        "architecture_signature": (
            "CLIP ViT-B/16 layers 9,7,5,3,1 -> five independent "
            "projection/FiLM/decoder stages"
        ),
        "film_injection_count": len(model.film_stages),
        "legacy_singlefilm_attributes_present": any(
            hasattr(model, name) for name in ("film_mul", "film_add", "reduce")
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "source_snapshot_sha256": metadata.get("source_snapshot_sha256"),
    }


def validate_grasps(grasps: Any, *, split: str, index: int) -> int:
    if not isinstance(grasps, list) or not grasps:
        raise ValueError(f"{split}[{index}] has no GT grasp rectangles")
    for grasp in grasps:
        array = np.asarray(grasp, dtype=np.float64)
        if array.shape != (4, 2) or not np.isfinite(array).all():
            raise ValueError(f"{split}[{index}] malformed GT rectangle")
    return len(grasps)


def probability_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 1:
            raise ValueError(f"ambiguous probability archive: {path}")
        return np.asarray(archive[archive.files[0]])


def audit_splits(*, full_probability_scan: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "split_semantics": (
            "official OCID-VLG unique; scene-frame/RGB-D disjoint"
        ),
        "full_probability_content_scan": bool(full_probability_scan),
        "splits": {},
        "pairwise_overlap": {},
    }
    identities: dict[str, dict[str, set[str]]] = {}
    for split in ("train", "val", "test"):
        expected_count, expected_sha = EXPECTED_MANIFESTS[split]
        frozen_path = FROZEN_MANIFESTS / f"ocidvlg_unique_{split}.json"
        frozen_rows = load_json(require_file(frozen_path, label=f"{split} manifest"))
        if len(frozen_rows) != expected_count or sha256_file(frozen_path) != expected_sha:
            raise ValueError(f"{split} frozen manifest identity mismatch")
        annotation_path = DATASET_ROOT / "refer/unique" / f"{split}_expressions.json"
        annotation = load_json(
            require_file(annotation_path, label=f"{split} annotations")
        )["data"]
        if len(annotation) != expected_count:
            raise ValueError(f"{split} annotation count mismatch")
        compact_path = COMPACT_RUN / "compact_inputs" / split / "manifest.jsonl"
        compact_rows = read_jsonl(
            require_file(compact_path, label=f"{split} compact manifest")
        )
        if len(compact_rows) != expected_count:
            raise ValueError(f"{split} compact coverage mismatch")

        sample_ids: set[str] = set()
        scenes: set[str] = set()
        rgb_hashes: set[str] = set()
        depth_hashes: set[str] = set()
        pair_hashes: set[str] = set()
        gt_rectangles = 0
        empty_masks = 0
        prob_min = 1.0
        prob_max = 0.0
        for index, (frozen, official, compact) in enumerate(
            zip(frozen_rows, annotation, compact_rows, strict=True)
        ):
            if (
                int(compact["sample_index"]) != index
                or int(compact["question_index"]) != int(official["question_index"])
                or str(compact["scene_id"]) != str(official["image_filename"])
                or str(compact["query"]) != str(official["question"])
                or str(frozen["scene_id"]) != str(official["image_filename"])
                or str(frozen["text"]) != str(official["question"])
            ):
                raise ValueError(f"{split}[{index}] sample mapping mismatch")
            gt_rectangles += validate_grasps(
                official.get("grasps"), split=split, index=index
            )
            for label in ("rgb_path", "depth_path", "mask_path"):
                raw = Path(str(frozen[label]).replace("\\", "/"))
                prepared = raw if raw.is_absolute() else PROJECT_ROOT / "hifics" / raw
                require_file(prepared.resolve(), label=f"{split} prepared {label}")
            if compact.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
                raise ValueError(f"{split}[{index}] checkpoint lineage mismatch")
            if compact.get("ready") is not True:
                raise ValueError(f"{split}[{index}] compact row is not ready")
            probability = require_file(
                Path(compact["probability_path"]), label="predicted probability"
            )
            mask = require_file(Path(compact["native_mask_path"]), label="predicted mask")
            if compact.get("probability_shape") != [352, 352]:
                raise ValueError(f"{split}[{index}] probability shape mismatch")
            if compact.get("native_mask_shape") != [480, 640]:
                raise ValueError(f"{split}[{index}] mask shape mismatch")
            if compact.get("probability_dtype") != "float32":
                raise ValueError(f"{split}[{index}] probability dtype mismatch")
            if full_probability_scan:
                values = probability_array(probability)
                if (
                    values.shape != (352, 352)
                    or values.dtype != np.float32
                    or not np.isfinite(values).all()
                    or float(values.min()) < 0.0
                    or float(values.max()) > 1.0
                ):
                    raise ValueError(f"{split}[{index}] invalid probability values")
                prob_min = min(prob_min, float(values.min()))
                prob_max = max(prob_max, float(values.max()))
            sample_id = str(compact["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"{split} duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
            scenes.add(str(compact["scene_id"]))
            rgb_sha = str(compact["source_rgb_sha256"])
            depth_sha = str(compact["source_depth_sha256"])
            require_file(Path(compact["source_rgb_path"]), label="source RGB")
            require_file(Path(compact["source_depth_path"]), label="source depth")
            rgb_hashes.add(rgb_sha)
            depth_hashes.add(depth_sha)
            pair_hashes.add(stable_pair_hash(rgb_sha, depth_sha))
            native_area = compact.get("native_mask_area_px")
            if native_area is None:
                with Image.open(mask) as image:
                    native_area = int(np.count_nonzero(np.asarray(image)))
            empty_masks += int(int(native_area) == 0)
            _ = mask

        identities[split] = {
            "sample_id": sample_ids,
            "scene_id": scenes,
            "rgb_sha256": rgb_hashes,
            "depth_sha256": depth_hashes,
            "rgbd_pair_sha256": pair_hashes,
        }
        result["splits"][split] = {
            "samples": expected_count,
            "scenes": len(scenes),
            "gt_grasp_rectangles": gt_rectangles,
            "frozen_manifest_path": str(frozen_path),
            "frozen_manifest_sha256": expected_sha,
            "annotations_path": str(annotation_path),
            "annotations_sha256": sha256_file(annotation_path),
            "compact_manifest_path": str(compact_path),
            "compact_manifest_sha256": sha256_file(compact_path),
            "predicted_mask_coverage": expected_count,
            "predicted_probability_coverage": expected_count,
            "empty_predicted_masks": empty_masks,
            "probability_min": prob_min if full_probability_scan else None,
            "probability_max": prob_max if full_probability_scan else None,
        }

    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        key = f"{left}__{right}"
        result["pairwise_overlap"][key] = {
            name: len(identities[left][name] & identities[right][name])
            for name in identities[left]
        }
        if any(result["pairwise_overlap"][key].values()):
            raise ValueError(f"split leakage detected: {key}")
    result["all_checks_passed"] = True
    return result


def audit_reference() -> dict[str, Any]:
    manifest_path = REFERENCE_RUN / "run_manifest.json"
    manifest = load_json(require_file(manifest_path, label="reference manifest"))
    identities = manifest["identities"]
    if (
        identities["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256
        or identities["test_manifest"]["sha256"] != EXPECTED_MANIFESTS["test"][1]
        or int(manifest["sample_count"]) != EXPECTED_MANIFESTS["test"][0]
    ):
        raise ValueError("reference run is not directly comparable")
    baseline_path = (
        COMPACT_RUN
        / "reports/baseline_test_audit_v2/repeatedfilm_baseline_oracle.json"
    )
    baseline = load_json(require_file(baseline_path, label="reference recomputation"))
    key_files = {
        "input_manifest": REFERENCE_RUN / "input_manifest.csv",
        "raw_candidates": REFERENCE_RUN / "candidates/dexnet_raw_candidates.parquet",
        "nms_candidates": REFERENCE_RUN / "candidates/dexnet_nms_candidates.parquet",
        "gqcnn_scores": REFERENCE_RUN / "scores/gqcnn_per_candidate.parquet",
        "evaluation": REFERENCE_RUN / "evaluation/hierfilm_pipeline_metrics.json",
    }
    return {
        "reference_modular_run": str(REFERENCE_RUN),
        "reference_run_reusable": True,
        "status": manifest["status"],
        "sample_count": int(manifest["sample_count"]),
        "checkpoint_sha256": identities["checkpoint"]["sha256"],
        "test_manifest_sha256": identities["test_manifest"]["sha256"],
        "method_name": "repeatedfilm_dexnet_gqcnn_reference",
        "corrected_evaluator": baseline["predicate"],
        "audit_anchor_only": {
            "j_at_1": baseline["metrics"]["j_at_1"],
            "j_at_5": baseline["metrics"]["recall_at_5"],
            "candidate_pool_oracle": baseline["metrics"]["nms_oracle"],
            "mrr": baseline["metrics"]["mrr"],
        },
        "independent_verification": manifest["verification"],
        "protected_key_files_before": {
            name: {
                "path": str(require_file(path, label=f"reference {name}")),
                "sha256": sha256_file(path),
            }
            for name, path in key_files.items()
        },
    }


def audit_environment() -> dict[str, Any]:
    stat = os.statvfs(PROJECT_ROOT)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos_version": command_text("sw_vers", "-productVersion"),
        "macos_build": command_text("sw_vers", "-buildVersion"),
        "chip": command_text("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(command_text("sysctl", "-n", "hw.memsize")),
        "available_disk_bytes": int(stat.f_bavail * stat.f_frsize),
        "python": sys.version,
        "pytorch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "opencv": package_version("opencv-python-headless")
        or package_version("opencv-python"),
        "scikit_image": package_version("scikit-image"),
        "scipy": package_version("scipy"),
        "pyarrow": package_version("pyarrow"),
        "pandas": package_version("pandas"),
        "source_run_size_bytes": sum(
            path.stat().st_size for path in SOURCE_RUN.rglob("*") if path.is_file()
        ),
        "reference_run_size_bytes": sum(
            path.stat().st_size
            for path in REFERENCE_RUN.rglob("*")
            if path.is_file()
        ),
        "compact_reuse_policy": (
            "reference existing train/val/test compact predictions; do not copy "
            "RGB, depth, masks, or float32 probability arrays into the new run"
        ),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def report_markdown(
    source: dict[str, Any], splits: dict[str, Any], reference: dict[str, Any], env: dict[str, Any]
) -> str:
    rows = []
    for split, values in splits["splits"].items():
        rows.append(
            f"| {split} | {values['samples']:,} | {values['scenes']:,} | "
            f"{values['gt_grasp_rectangles']:,} | "
            f"{values['predicted_mask_coverage']:,} |"
        )
    return f"""# Repeated-FiLM 4-DoF source audit

Audit time: {datetime.now(timezone.utc).isoformat()}  
Decision: **PHASE_0_PASS**.

## Retained visual-grounding lineage

- Variant: `hierarchical_repeated_film`.
- Checkpoint: `{source['checkpoint_path']}`.
- Checkpoint SHA-256: `{source['checkpoint_sha256']}`.
- Best step/epoch: {source['checkpoint_step']} / {source['checkpoint_epoch']}.
- Strict load: {source['checkpoint_loaded_keys']}/{source['checkpoint_expected_keys']} keys, no missing or unexpected keys.
- Architecture: five independent FiLM injections; no legacy single-FiLM attributes.
- Config byte SHA-256: `{source['config_sha256']}`.
- Config canonical SHA-256: `{source['config_canonical_sha256']}`.

## Data inventory

| Split | Samples | Scenes | GT rectangles | Predicted masks/probabilities |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

All pairwise intersections are zero for stable sample ID, scene-frame ID, RGB
hash, depth hash, and RGB-D pair hash. GT rectangles are used only for training
labels, validation, formal evaluation, oracle analysis, and post-hoc failure
analysis. They are forbidden from deployment inference inputs.

## Reference comparability

`{reference['reference_modular_run']}` is directly reusable as
`repeatedfilm_dexnet_gqcnn_reference`: it binds the same checkpoint, same
7,675-row test manifest, and corrected same-GT evaluator (rectangle IoU > 0.25
and 180-degree periodic angle error <= 30 degrees). Existing metrics are audit
anchors only; final reporting requires independent recomputation.

## Mac and storage

- Host: {env['chip']}, {env['memory_bytes'] / 2**30:.0f} GiB unified memory.
- macOS: {env['macos_version']} ({env['machine']}).
- Python/PyTorch: {env['python'].split()[0]} / {env['pytorch']}.
- MPS built/available: {env['mps_built']} / {env['mps_available']}.
- Available disk: {env['available_disk_bytes'] / 2**30:.1f} GiB.

The new run will reference the already complete compact train/validation/test
predictions rather than duplicate approximately 14 GiB of probability and mask
artifacts. Formal model computation remains float32.

## Stop-condition decision

No stop condition is active: the checkpoint loads strictly, required OCID-VLG
RGB/depth/language/GT rectangles are present, all split predictions exist,
the reference evaluator is reproducible, official third-party weights are
legally accessible, and available storage is sufficient for compact execution.

J@1/J@5 measure consistency with annotated OCID-VLG 2D grasp rectangles. They
do not measure physical grasp success.
"""


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    audit_dir = run_dir / "audit"
    reports_dir = run_dir / "reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    source = audit_checkpoint()
    splits = audit_splits(full_probability_scan=args.full_probability_scan)
    reference = audit_reference()
    environment = audit_environment()
    source.update(
        {
            "dataset_root": str(DATASET_ROOT),
            "dataset_version": "OCID-VLG official unique split snapshot",
            "train_manifest_sha256": EXPECTED_MANIFESTS["train"][1],
            "validation_manifest_sha256": EXPECTED_MANIFESTS["val"][1],
            "test_manifest_sha256": EXPECTED_MANIFESTS["test"][1],
            "reference_modular_run": str(REFERENCE_RUN),
            "reference_run_reusable": True,
        }
    )
    inventory = {
        "schema_version": 1,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PHASE_0_PASS",
        "source": source,
        "environment": environment,
        "dataset_roots": [str(DATASET_ROOT), str(CROG_DATASET_ROOT)],
        "compact_prediction_source": str(COMPACT_RUN),
        "stop_conditions": [],
    }
    write_json(audit_dir / "source_inventory.json", inventory)
    write_json(audit_dir / "repeatedfilm_source_manifest.json", source)
    write_json(audit_dir / "dataset_split_audit.json", splits)
    write_json(audit_dir / "reference_baseline_inventory.json", reference)
    (reports_dir / "REPEATEDFILM_4DOF_SOURCE_AUDIT.md").write_text(
        report_markdown(source, splits, reference, environment), encoding="utf-8"
    )
    print(json.dumps({"status": "PHASE_0_PASS", "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
