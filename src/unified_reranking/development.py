"""Prepare leakage-safe development supervision and grouped folds."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluator_adapter import annotate_pool_labels, build_candidate_labels
from .hashing import atomic_json, atomic_text, sha256_file
from .splits import freeze_grouped_folds


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)


def _reuse_label_artifact(
    candidates_path: Path,
    destination: Path,
    previous: dict[str, Any] | None,
    evaluator_sha256: str,
) -> dict[str, Any] | None:
    """Reuse an audited label table only after exact identity verification."""

    if not previous or not destination.is_file():
        return None
    if previous.get("evaluator_sha256") != evaluator_sha256:
        return None
    if previous.get("sha256") != sha256_file(destination):
        return None
    candidate_keys = pd.read_parquet(
        candidates_path, columns=["sample_id", "candidate_id", "native_rank"]
    ).sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
    label_keys = pd.read_parquet(
        destination, columns=["sample_id", "candidate_id", "native_rank"]
    ).sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
    if len(candidate_keys) != len(label_keys) or not candidate_keys.reset_index(drop=True).equals(
        label_keys.reset_index(drop=True)
    ):
        return None
    descriptor = dict(previous)
    descriptor["reused_after_identity_verification"] = True
    descriptor["candidate_manifest_sha256"] = sha256_file(candidates_path)
    return descriptor


def prepare_development_contracts(
    run_dir: Path,
    fair_run: Path,
    modular_run: Path,
    fold_source: Path,
) -> dict[str, Any]:
    evaluator_source = fair_run / "config" / "canonical_evaluator.py"
    evaluator_destination = run_dir / "configs" / "canonical_evaluator.py"
    _atomic_copy(evaluator_source, evaluator_destination)
    evaluator_sha = sha256_file(evaluator_source)
    if sha256_file(evaluator_destination) != evaluator_sha:
        raise RuntimeError("frozen evaluator copy is not byte-identical")
    split_audit = freeze_grouped_folds(run_dir, fold_source)
    previous_audit_path = run_dir / "03_features" / "development_label_audit.json"
    previous_audit = (
        json.loads(previous_audit_path.read_text(encoding="utf-8"))
        if previous_audit_path.is_file()
        else {}
    )
    previous_labels = previous_audit.get("labels", {})
    result: dict[str, Any] = {
        "status": "COMPLETE_FOR_AVAILABLE_CANDIDATES",
        "evaluator": {
            "source": str(evaluator_source.resolve()),
            "destination": str(evaluator_destination.resolve()),
            "sha256": evaluator_sha,
        },
        "splits": split_audit,
        "labels": {},
        "missing_candidate_pools": [],
    }
    for route in ("crog", "g1", "c1"):
        for split in ("train", "validation"):
            candidates = run_dir / "02_candidates" / f"{route}_{split}_all.parquet"
            if not candidates.is_file():
                result["missing_candidate_pools"].append(f"{route}/{split}")
                continue
            label_token = "validation" if split == "validation" else "train"
            sample_labels = modular_run / "manifests" / f"{label_token}_labels.parquet"
            all_destination = run_dir / "03_features" / f"candidate_labels_{route}_{split}_all.parquet"
            all_key = f"{route}/{split}/all"
            descriptor = _reuse_label_artifact(
                candidates,
                all_destination,
                previous_labels.get(all_key),
                evaluator_sha,
            )
            if descriptor is None:
                descriptor = build_candidate_labels(
                    candidates,
                    sample_labels,
                    all_destination,
                    evaluator_destination,
                    evaluator_sha,
                    split=split,
                )
                descriptor["candidate_manifest_sha256"] = sha256_file(candidates)
                descriptor["sample_labels_sha256"] = sha256_file(sample_labels)
            top5_destination = run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
            top5_candidates = run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
            top5_key = f"{route}/{split}/top5"
            top5_descriptor = _reuse_label_artifact(
                top5_candidates,
                top5_destination,
                previous_labels.get(top5_key),
                evaluator_sha,
            )
            if top5_descriptor is None:
                all_labels = pd.read_parquet(all_destination)
                top5_labels = annotate_pool_labels(
                    all_labels.loc[all_labels["native_rank"] <= 5].copy()
                )
                temporary = top5_destination.with_name(f".{top5_destination.name}.{os.getpid()}.tmp")
                top5_labels.to_parquet(temporary, index=False)
                os.replace(temporary, top5_destination)
                top5_descriptor = {
                    "candidate_rows": len(top5_labels),
                    "samples": int(top5_labels["sample_id"].nunique()),
                    "positive_candidates": int(top5_labels["candidate_success"].sum()),
                    "solvable_samples": int(top5_labels.loc[top5_labels["pool_solvable"], "sample_id"].nunique()),
                    "destination": str(top5_destination.resolve()),
                    "sha256": sha256_file(top5_destination),
                    "candidate_manifest_sha256": sha256_file(top5_candidates),
                    "evaluator_sha256": evaluator_sha,
                }
            result["labels"][all_key] = descriptor
            result["labels"][top5_key] = top5_descriptor
    if result["missing_candidate_pools"]:
        result["status"] = "PARTIAL_AWAITING_G1_C1_DEVELOPMENT_INFERENCE"
    atomic_json(run_dir / "03_features" / "development_label_audit.json", result)
    missing = ", ".join(result["missing_candidate_pools"]) or "none"
    report = f"""# Development-label contract

Status: **{result['status']}**.

- The byte-frozen fair evaluator is used at SHA-256 `{evaluator_sha}`.
- Only Train and Validation sample-label files are accepted by this builder.
- Candidate labels are stored separately from inference features.
- Missing fair candidate pools: {missing}.
- Test candidate labels were not loaded.
"""
    atomic_text(run_dir / "03_features" / "DEVELOPMENT_LABEL_AUDIT.md", report)
    return result
