"""Read-only audit and candidate projection from the locked HiFi source run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .artifacts import atomic_json, atomic_parquet, utc_now
from .contracts import canonical_sha256, identity_table_sha256, sha256_file
from .pools import SOURCE_INFERENCE_COLUMNS, adapt_source_candidates


EXPECTED_SPLIT_COUNTS = {"train": 26_295, "validation": 3_778, "test": 7_675}
EXPECTED_SOURCE_BASELINES = {
    "G1": {
        "candidate_rows": 25_527,
        "j_at_1": 0.8706188925081433,
        "j_at_5": 0.9078827361563518,
        "non_empty_rate": 0.9850162866449511,
    },
    "C1": {
        "candidate_rows": 22_913,
        "j_at_1": 0.7930944625407166,
        "j_at_5": 0.8676221498371335,
        "non_empty_rate": 0.9953094462540717,
    },
}
EXPECTED_SOURCE_POOLS = {
    ("G1", "validation"): {"rows": 12_274, "max_rank": 10},
    ("C1", "validation"): {"rows": 11_085, "max_rank": 10},
    ("G1", "test"): {"rows": 25_527, "max_rank": 14},
    ("C1", "test"): {"rows": 22_913, "max_rank": 11},
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "pyarrow", "scikit-learn", "torch", "google-genai")
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return {}
    values: dict[str, str] = {}
    for name in names:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "not-installed"
    return values


def _source_candidate_path(base: Path, split: str, backend: str) -> Path:
    phase = "validation/final" if split == "validation" else "formal_test"
    return base / phase / backend / "per_candidate_predictions.parquet"


def _source_metrics(base: Path, backend: str) -> dict[str, Any]:
    metrics = json.loads((base / "formal_test" / backend / "metrics.json").read_text())
    return {
        "candidate_rows": int(
            pq.ParquetFile(_source_candidate_path(base, "test", backend)).metadata.num_rows
        ),
        "j_at_1": float(metrics["j_at_1"]),
        "j_at_5": float(metrics["j_at_5"]),
        "non_empty_rate": float(metrics["non_empty_rate"]),
    }


def _split_inventory(base: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}
    for split in EXPECTED_SPLIT_COUNTS:
        path = base / "manifests" / f"{split}_samples.parquet"
        frame = pd.read_parquet(
            path,
            columns=[
                "sample_id",
                "scene_id",
                "source_rgb_sha256",
                "source_depth_sha256",
                "rgbd_pair_sha256",
                "split",
            ],
        )
        if len(frame) != EXPECTED_SPLIT_COUNTS[split]:
            raise AssertionError(f"{split} sample count drift: {len(frame)}")
        expected_label = "val" if split == "validation" else split
        if not frame["split"].astype(str).eq(expected_label).all():
            raise AssertionError(f"{split} manifest contains another split")
        frame["split"] = split
        records.append(frame)
        audit[split] = {
            "sample_count": int(len(frame)),
            "scene_count": int(frame["scene_id"].nunique()),
            "manifest_sha256": sha256_file(path),
            "sample_ids_sha256": canonical_sha256(sorted(frame["sample_id"].astype(str))),
        }
    mapping = pd.concat(records, ignore_index=True)
    for key in ("sample_id", "scene_id", "source_rgb_sha256", "source_depth_sha256", "rgbd_pair_sha256"):
        counts = mapping.groupby(key, dropna=False)["split"].nunique()
        if int((counts > 1).sum()) != 0:
            raise AssertionError(f"split leakage detected through {key}")
    audit["cross_split_overlap"] = {
        key: 0
        for key in ("sample_id", "scene_id", "source_rgb_sha256", "source_depth_sha256", "rgbd_pair_sha256")
    }
    return mapping, audit


def run_audit(project_root: Path, base_run: Path, run_dir: Path) -> dict[str, Any]:
    finalization = base_run / "FINALIZATION_COMPLETE.json"
    lock = base_run / "frozen_4dof_backends_experiment_manifest.json"
    if not finalization.is_file() or not lock.is_file():
        raise FileNotFoundError("locked source run is incomplete")
    source_final = json.loads(finalization.read_text())
    if source_final.get("status") != "COMPLETE":
        raise RuntimeError("source run is not complete")
    mapping, split_audit = _split_inventory(base_run)
    atomic_parquet(run_dir / "data/split_mapping.parquet", mapping)
    baselines: dict[str, Any] = {}
    for backend, expected in EXPECTED_SOURCE_BASELINES.items():
        observed = _source_metrics(base_run, backend)
        for key, value in expected.items():
            if abs(float(observed[key]) - float(value)) > 1e-12:
                raise AssertionError(f"{backend} {key} regression: {observed[key]} != {value}")
        baselines[backend] = observed
    baseline_frame = pd.DataFrame(
        [{"backend": backend, **metrics} for backend, metrics in baselines.items()]
    )
    atomic_parquet(run_dir / "audit/source_baselines.parquet", baseline_frame)
    baseline_frame.to_csv(run_dir / "tables/baseline_reproduction.csv", index=False)
    dirty = _git(project_root, "status", "--short")
    inventory = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "project_root": str(project_root),
        "base_run": str(base_run),
        "source_finalization_sha256": sha256_file(finalization),
        "source_lock_sha256": sha256_file(lock),
        "source_experiment_lock_sha256": source_final.get("experiment_lock_sha256"),
        "source_primary": source_final.get("primary_method_id"),
        "split_audit": split_audit,
        "source_baselines": baselines,
        "baseline_regression_match": True,
        "api_authorization": {
            "allow_new_gemini_calls": os.environ.get("ALLOW_NEW_GEMINI_CALLS") == "1",
            "max_api_cost_usd_present": bool(os.environ.get("MAX_API_COST_USD")),
            "allow_formal_api_run": os.environ.get("ALLOW_FORMAL_API_RUN") == "1",
            "api_key_present": bool(os.environ.get("GEMINI_API_KEY")),
        },
        "test_contract": {
            "status": "SEALED_BEFORE_NEW_FORMAL",
            "warning": "source aggregate test metrics were already published; the new formal test is not pristine",
            "per_sample_test_labels_loaded": False,
        },
    }
    atomic_json(run_dir / "AUDIT_INVENTORY.json", inventory)
    manifest = {
        "schema_version": 1,
        "experiment": "g1_c1_safe_rerank",
        "created_at_utc": utc_now(),
        "git_branch": _git(project_root, "branch", "--show-current"),
        "git_commit": _git(project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "git_status_short": dirty.splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "base_run": str(base_run),
        "source_lock_sha256": sha256_file(lock),
        "split_hashes": {key: value["manifest_sha256"] for key, value in split_audit.items() if isinstance(value, dict) and "manifest_sha256" in value},
        "seeds": [17, 29, 43],
        "scene_grouped_folds": 5,
        "union_nms": {"angle_deg": 10.0, "center_px": 4.0, "width_px": 5.0, "rotated_iou": 0.5},
        "formal_run_count": 0,
        "formal_api_allow_flag_at_creation": os.environ.get("ALLOW_FORMAL_API_RUN") == "1",
    }
    atomic_json(run_dir / "MANIFEST.json", manifest)
    (run_dir / "AUDIT.md").write_text(
        "# G1/C1 Safe Re-ranking Audit\n\n"
        "The locked HiFi source run passed its finalization marker. The train, validation, and test manifests contain no sample, scene, RGB, depth, or RGB-D hash overlap. "
        "Published source aggregate baselines match the preregistered regression values exactly. New per-sample test labels remain sealed.\n",
        encoding="utf-8",
    )
    return inventory


def freeze_inference_pools(base_run: Path, run_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for split in ("validation", "test"):
        for backend in ("G1", "C1"):
            source = _source_candidate_path(base_run, split, backend)
            # Column projection is deliberate: GT/evaluator columns in the old
            # result table are never materialized in the new inference artifact.
            frame = pd.read_parquet(source, columns=list(SOURCE_INFERENCE_COLUMNS))
            frozen = adapt_source_candidates(frame, backend=backend, split=split)
            expected = EXPECTED_SOURCE_POOLS[(backend, split)]
            observed_max_rank = int(frozen["original_rank"].max())
            if len(frozen) != expected["rows"] or observed_max_rank != expected["max_rank"]:
                raise AssertionError(
                    f"{backend} {split} AllNMS pool drift: "
                    f"rows={len(frozen)}, max_rank={observed_max_rank}, expected={expected}"
                )
            path = run_dir / "data" / f"frozen_{backend.lower()}_{split}_allnms_candidates.parquet"
            atomic_parquet(path, frozen)
            # Keep the unqualified artifact as an explicit canonical AllNMS alias
            # for older local consumers, and derive Top-5 without changing IDs.
            canonical_path = run_dir / "data" / f"frozen_{backend.lower()}_{split}_candidates.parquet"
            atomic_parquet(canonical_path, frozen)
            top5 = frozen.loc[frozen["original_rank"].astype(int) <= 5].copy()
            top5_path = run_dir / "data" / f"frozen_{backend.lower()}_{split}_top5_candidates.parquet"
            atomic_parquet(top5_path, top5)
            records[f"{backend}_{split}"] = {
                "path": str(path),
                "canonical_alias_path": str(canonical_path),
                "top5_path": str(top5_path),
                "candidate_rows": int(len(frozen)),
                "top5_candidate_rows": int(len(top5)),
                "max_rank": observed_max_rank,
                "nonempty_samples": int(frozen["sample_id"].nunique()),
                "candidate_identity_sha256": identity_table_sha256(frozen),
                "artifact_sha256": sha256_file(path),
                "top5_artifact_sha256": sha256_file(top5_path),
                "source_columns_loaded": list(SOURCE_INFERENCE_COLUMNS),
            }
    atomic_json(run_dir / "audit/frozen_pool_inventory.json", records)
    return records
