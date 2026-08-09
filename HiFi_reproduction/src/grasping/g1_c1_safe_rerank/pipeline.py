"""Deterministic preparation pipeline for calibrated single/union pools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import atomic_json, atomic_parquet
from .calibration import apply_backend_calibrators, fit_backend_oof_calibration
from .contracts import identity_table_sha256, sha256_file, validate_frozen_candidates
from .pools import build_deduplicated_union, build_raw_union, pool_manifest


TRACKS = ("g1", "c1", "union")
SPLITS = ("train", "validation", "test")


def _read_backend(run_dir: Path, backend: str, split: str) -> pd.DataFrame:
    path = run_dir / "data" / f"frozen_{backend}_{split}_candidates.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    validate_frozen_candidates(frame, require_top5=False)
    return frame


def _train_labels(run_dir: Path, backend: str) -> pd.DataFrame:
    path = run_dir / "data" / f"{backend}_train_candidate_labels.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    labels = pd.read_parquet(path)
    required = {"sample_id", "stable_candidate_id", "candidate_correct"}
    if not required.issubset(labels.columns):
        raise ValueError(f"invalid train label artifact: {path}")
    return labels


def _attach_pool_contract(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["pool_rank"] = output["original_rank"].astype(int)
    output["pool_score"] = output["original_score"].astype(float)
    return output


def build_calibrated_pools(run_dir: str | Path, *, folds: int = 5) -> dict[str, Any]:
    """Fit calibration on train OOF only, then build all frozen inference pools."""

    run = Path(run_dir).expanduser().resolve()
    train_parts: list[pd.DataFrame] = []
    calibrators = {}
    calibration_artifacts: dict[str, Any] = {}
    for backend in ("g1", "c1"):
        candidates = _read_backend(run, backend, "train")
        labels = _train_labels(run, backend)
        labelled = candidates.merge(
            labels[["sample_id", "stable_candidate_id", "candidate_correct"]],
            on=["sample_id", "stable_candidate_id"],
            how="left",
            validate="one_to_one",
        )
        if labelled["candidate_correct"].isna().any():
            raise AssertionError(f"{backend}: missing train candidate labels")
        oof, artifact, full = fit_backend_oof_calibration(
            labelled, backend=backend.upper(), folds=int(folds)
        )
        oof["source_score_calibrated"] = oof["source_score_calibrated_oof"]
        train_parts.append(_attach_pool_contract(oof))
        calibrators[backend.upper()] = full
        calibration_artifacts[backend.upper()] = artifact
        atomic_json(run / "models" / f"score_calibrator_{backend}.json", artifact)
    calibrated_train = pd.concat(train_parts, ignore_index=True)
    # Preserve the original source-table order for deterministic downstream hashes.
    calibrated_train = calibrated_train.sort_values(
        ["sample_id", "backend", "original_rank", "stable_candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    records: dict[str, Any] = {"calibration": calibration_artifacts, "pools": {}}
    for split in SPLITS:
        if split == "train":
            combined = calibrated_train
        else:
            parts = [
                apply_backend_calibrators(_read_backend(run, backend, split), calibrators)
                for backend in ("g1", "c1")
            ]
            combined = pd.concat(parts, ignore_index=True)
        singles: dict[str, pd.DataFrame] = {}
        for backend in ("g1", "c1"):
            single = _attach_pool_contract(
                combined.loc[combined["backend"].eq(backend.upper())].copy()
            )
            path = run / "data" / f"calibrated_{backend}_{split}_candidates.parquet"
            # Drop development labels before any inference artifact is saved.
            inference = single.drop(
                columns=["candidate_correct", "source_score_calibrated_oof", "calibration_fold"],
                errors="ignore",
            )
            atomic_parquet(path, inference)
            singles[backend] = inference
            records["pools"][f"{backend}_{split}"] = {
                **pool_manifest(inference),
                "artifact_sha256": sha256_file(path),
                "path": str(path),
            }
        raw = build_raw_union(singles["g1"], singles["c1"])
        raw_path = run / "data" / f"raw_union_{split}_candidates.parquet"
        atomic_parquet(raw_path, raw)
        dedup = build_deduplicated_union(raw)
        dedup_path = run / "data" / f"dedup_union_{split}_candidates.parquet"
        atomic_parquet(dedup_path, dedup)
        records["pools"][f"raw_union_{split}"] = {
            "candidate_rows": int(len(raw)),
            "artifact_sha256": sha256_file(raw_path),
            "candidate_identity_sha256": identity_table_sha256(raw),
            "path": str(raw_path),
        }
        records["pools"][f"dedup_union_{split}"] = {
            **pool_manifest(dedup),
            "artifact_sha256": sha256_file(dedup_path),
            "path": str(dedup_path),
        }
    atomic_json(run / "audit/calibration_and_pool_manifest.json", records)
    return records
