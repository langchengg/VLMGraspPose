"""Source-run audit and GT-free Top-5 materialisation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .constants import SOURCE_INFERENCE_COLUMNS
from .contracts import candidate_set_sha256, geometry_sha256, validate_candidate_manifest
from .io import atomic_json, atomic_parquet, sha256_file, utc_now


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def source_candidate_path(source_run: Path, split: str, backend: str) -> Path:
    prefix = "validation/final" if split == "validation" else "formal_test"
    return source_run / prefix / backend / "per_candidate_predictions.parquet"


def source_sample_path(source_run: Path, split: str, backend: str) -> Path:
    prefix = "validation/final" if split == "validation" else "formal_test"
    return source_run / prefix / backend / "per_sample_predictions.parquet"


def _metadata(value: Any) -> dict[str, Any]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("candidate metadata is not an object")
    return parsed


def freeze_candidate_table(source_run: Path, split: str, backend: str) -> pd.DataFrame:
    # Explicit column projection prevents the mixed source table's GT fields
    # from entering the frozen deployment artifact.
    source = pd.read_parquet(
        source_candidate_path(source_run, split, backend),
        columns=list(SOURCE_INFERENCE_COLUMNS),
    )
    selected = source.loc[source["rank"].astype(int) <= 5].copy()
    selected = selected.sort_values(["sample_id", "rank", "candidate_id"], kind="mergesort")
    metadata = selected["candidate_metadata_json"].map(_metadata)
    selected["backend"] = backend
    selected["split"] = split
    selected["original_rank"] = selected["rank"].astype(int)
    selected["original_score"] = selected["score"].astype(float)
    selected["center_mask_support"] = metadata.map(lambda row: float(row.get("center_mask_support", np.nan)))
    selected["jaw_mask_support"] = metadata.map(lambda row: float(row.get("jaw_mask_support", np.nan)))
    selected["candidate_geometry_sha256"] = selected.apply(geometry_sha256, axis=1)
    set_hashes = {
        str(sample_id): candidate_set_sha256(group)
        for sample_id, group in selected.groupby("sample_id", sort=False)
    }
    selected["candidate_set_sha256"] = selected["sample_id"].astype(str).map(set_hashes)
    output = selected[[
        "backend", "split", "sample_id", "scene_id", "candidate_id",
        "original_rank", "original_score", "center_x", "center_y", "angle_deg",
        "width_px", "height_px", "center_mask_support", "jaw_mask_support",
        "candidate_geometry_sha256", "candidate_set_sha256",
    ]].reset_index(drop=True)
    validate_candidate_manifest(output)
    return output


def _candidate_stats(source_run: Path, split: str, backend: str) -> dict[str, Any]:
    candidate_path = source_candidate_path(source_run, split, backend)
    sample_path = source_sample_path(source_run, split, backend)
    schema = pq.read_schema(candidate_path)
    raw = pd.read_parquet(candidate_path, columns=list(SOURCE_INFERENCE_COLUMNS))
    samples = pd.read_parquet(sample_path, columns=["sample_id", "scene_id"])
    frozen = raw.loc[raw["rank"].astype(int) <= 5]
    counts = frozen.groupby("sample_id").size()
    numeric = raw[["center_x", "center_y", "angle_deg", "width_px", "height_px", "score"]].to_numpy(dtype=float)
    return {
        "source_candidate_path": str(candidate_path.resolve()),
        "source_candidate_sha256": sha256_file(candidate_path),
        "source_sample_path": str(sample_path.resolve()),
        "source_sample_sha256": sha256_file(sample_path),
        "source_candidate_schema": [str(name) for name in schema.names],
        "label_and_deployment_fields_mixed": "candidate_success" in schema.names,
        "sample_count": int(len(samples)),
        "scene_count": int(samples["scene_id"].nunique()),
        "source_full_candidate_rows": int(len(raw)),
        "frozen_top5_candidate_rows": int(len(frozen)),
        "frozen_candidate_count_distribution": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
        "empty_samples": int(len(samples) - counts.size),
        "single_candidate_samples": int((counts == 1).sum()),
        "duplicate_candidate_ids": int(raw.duplicated(["sample_id", "candidate_id"]).sum()),
        "rank_origin": int(raw["rank"].min()) if len(raw) else None,
        "score_descending": bool(all(group["score"].tolist() == sorted(group["score"].tolist(), reverse=True) for _, group in raw.groupby("sample_id"))),
        "finite_geometry_and_score": bool(np.isfinite(numeric).all()),
    }


def run_audit(repo_root: Path, source_run: Path, run_dir: Path) -> dict[str, Any]:
    configs: dict[str, Any] = {}
    for backend in ("G1", "C1"):
        path = source_run / "selected_configs" / f"{backend}.json"
        payload = json.loads(path.read_text())
        checkpoint = Path(payload["finetuned_checkpoint"]).resolve()
        if sha256_file(checkpoint) != payload["finetuned_checkpoint_sha256"]:
            raise RuntimeError(f"{backend} checkpoint hash mismatch")
        configs[backend] = {
            "config_path": str(path.resolve()),
            "config_sha256": sha256_file(path),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    packages = {}
    for name in ("google-genai", "torch", "pandas", "pyarrow", "opencv-python", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    candidates = {
        f"{backend}_{split}": _candidate_stats(source_run, split, backend)
        for split in ("validation", "test") for backend in ("G1", "C1")
    }
    inventory = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "repo_root": str(repo_root.resolve()),
        "source_run": str(source_run.resolve()),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "git_status": _git(repo_root, "status", "--short").splitlines(),
        "git_dirty": bool(_git(repo_root, "status", "--short")),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "backends": configs,
        "candidates": candidates,
        "evaluator": {
            "path": str((repo_root / "src/grasping/common/evaluator.py").resolve()),
            "sha256": sha256_file(repo_root / "src/grasping/common/evaluator.py"),
            "rectangle_iou": "rasterized rectangle IoU > 0.25",
            "angle": "180-degree periodic difference <= 30 degrees",
            "same_gt_required": True,
            "denominator": "all samples including empty predictions",
        },
        "coordinate_contract": {
            "image_origin": "top-left",
            "x": "column, increases rightward",
            "y": "row, increases downward",
            "image_size": [640, 480],
            "angle_units": "degrees",
            "angle_range": "[-90, 90)",
            "positive_rotation_in_image_coordinates": "clockwise visually because y increases downward",
            "angle_axis": "rectangle width / gripper closing axis",
            "periodicity_degrees": 180,
            "height_px": 20.0,
            "contact_points": "centre +/- 0.5*width*(cos(theta), sin(theta))",
        },
        "api_preflight": {
            "api_key_present": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
            "max_api_cost_usd_present": bool(os.environ.get("MAX_API_COST_USD")),
            "max_provider_requests_present": bool(os.environ.get("MAX_PROVIDER_REQUESTS")),
            "allow_formal": os.environ.get("ALLOW_FORMAL_GEMINI_RERANK") == "1" and os.environ.get("ALLOW_PAID_API_RUN") == "1",
        },
        "current_official_model_evidence": {
            "gemini_3_6_flash": "officially documented",
            "gemini_robotics_er_2_preview": "not present on the public models page; exact metadata check required, no substitution",
        },
    }
    atomic_json(run_dir / "audit_inventory.json", inventory)
    lines = [
        "# Pure Gemini API G1/C1 Re-ranking Audit",
        "",
        f"- Repository: `{repo_root.resolve()}`",
        f"- Locked source run: `{source_run.resolve()}`",
        f"- Git: `{inventory['git_branch']}` at `{inventory['git_commit']}`; dirty={inventory['git_dirty']}",
        "- The source candidate tables mix inference and GT/evaluator fields. New frozen manifests are created only through an explicit inference-column projection.",
        "- The source tables contain complete NMS pools; API inputs are strictly frozen to source ranks 1–5.",
        "- Reused modules: CompactSampleLoader, corrected evaluator/geometry, and read-only source manifests.",
        "- New code required: API-only evidence extraction, two-board renderer, strict schema/parser, SQLite ledger, provider adapters, policy selection, lock, evaluation and reporting.",
        "- Main leakage risk: accidental materialisation of candidate_success/best IoU/GT paths from mixed source Parquet files. Payload construction is allowlist-only and scanned before every request.",
        "- Board volume depends on policy gates; K=0 and K=1 never produce provider requests.",
        "- Cost estimation uses observed token usage and dated official Flash prices; ER2 cost remains null unless a reliable price is available.",
        "- Current blocker for paid calls: API key, MAX_API_COST_USD, and MAX_PROVIDER_REQUESTS are absent.",
        "",
        "All reported J@1 values are OCID-VLG offline 2D grasp-rectangle consistency, not physical grasp success.",
    ]
    (run_dir / "AUDIT.md").write_text("\n".join(lines) + "\n")
    return inventory
