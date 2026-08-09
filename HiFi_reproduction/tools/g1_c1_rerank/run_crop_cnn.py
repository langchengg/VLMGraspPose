#!/usr/bin/env python3
"""Train/evaluate the R13 candidate-aligned crop CNN on both frozen pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json, atomic_parquet  # noqa: E402
from src.grasping.g1_c1_safe_rerank.crop_model import CropResidualModel  # noqa: E402
from tools.g1_c1_rerank.run_local_matrix import (  # noqa: E402
    BACKENDS,
    POOLS,
    SEEDS,
    _attach_calibration,
    _columns,
    _evaluate,
    _load_joined,
    _universe,
)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def load_crops(run: Path, backend: str, split: str, frame: pd.DataFrame) -> np.ndarray:
    root = run / "02_features" / split / backend / "candidate_crops_64"
    manifest = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"crop store incomplete: {root}")
    keys, arrays = [], []
    for path in sorted((root / "shards").glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as loaded:
            local = pd.DataFrame(
                {
                    "sample_id": loaded["sample_id"].astype(str),
                    "stable_candidate_id": loaded["candidate_id"].astype(str),
                    "crop_index": np.arange(len(loaded["crop"])) + sum(len(value) for value in arrays),
                }
            )
            keys.append(local)
            arrays.append(np.asarray(loaded["crop"], dtype=np.uint8))
    crop = np.concatenate(arrays, axis=0)
    index = pd.concat(keys, ignore_index=True)
    if index.duplicated(["sample_id", "stable_candidate_id"]).any():
        raise AssertionError("duplicate crop candidate keys")
    aligned = frame[["sample_id", "stable_candidate_id"]].merge(
        index,
        on=["sample_id", "stable_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if aligned["crop_index"].isna().any():
        raise AssertionError("crop store does not cover requested candidates")
    return crop[aligned["crop_index"].to_numpy(dtype=int)]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    args.base_run = args.base_run.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    universe = _universe(args.base_run, "validation")
    records = []
    for backend in BACKENDS:
        train_all = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, "train", "allnms"), backend, "train")
        validation_all = _attach_calibration(args.run_dir, _load_joined(args.run_dir, backend, "validation", "allnms"), backend, "validation")
        train_crop_all = load_crops(args.run_dir, backend, "train", train_all)
        validation_crop_all = load_crops(args.run_dir, backend, "validation", validation_all)
        for pool in POOLS:
            loss_selection_path = (
                args.run_dir
                / "07_validation"
                / "loss_selection"
                / backend
                / f"{pool}.json"
            )
            loss_selection = json.loads(
                loss_selection_path.read_text(encoding="utf-8")
            )
            if loss_selection.get("status") != "LOCKED_FROM_OFFICIAL_VALIDATION":
                raise RuntimeError(f"controlled loss is not locked: {loss_selection_path}")
            objective = str(loss_selection["selected_objective"])
            temperature = float(loss_selection["selected_temperature"])
            train_mask = train_all["original_rank"].le(5).to_numpy() if pool == "top5" else np.ones(len(train_all), dtype=bool)
            validation_mask = validation_all["original_rank"].le(5).to_numpy() if pool == "top5" else np.ones(len(validation_all), dtype=bool)
            train = train_all.loc[train_mask].reset_index(drop=True)
            validation = validation_all.loc[validation_mask].reset_index(drop=True)
            train_crop = train_crop_all[train_mask]
            validation_crop = validation_crop_all[validation_mask]
            columns = _columns(train, through="F6")
            score_parts = []
            for seed in SEEDS:
                root = args.run_dir / "07_validation" / "crop_cnn" / backend / pool
                model_path = args.run_dir / "05_models" / backend / pool / f"r13_crop_cnn_seed{seed}.joblib"
                score_path = root / f"scores_seed{seed}.parquet"
                metric_path = root / f"metrics_seed{seed}.json"
                if args.resume and all(path.is_file() for path in (model_path, score_path, metric_path)):
                    score_parts.append(pd.read_parquet(score_path).rename(columns={"reranker_score": f"score_seed{seed}"}))
                    records.append(json.loads(metric_path.read_text(encoding="utf-8")))
                    continue
                model = CropResidualModel(columns, seed=seed, device=args.device, epochs=8, objective=objective, temperature=temperature)
                try:
                    model.fit(train, train_crop, args.run_dir / "checkpoints" / backend / pool / f"r13_crop_cnn_seed{seed}.pt")
                except RuntimeError:
                    if args.device != "mps":
                        raise
                    model = CropResidualModel(columns, seed=seed, device="cpu", epochs=8, objective=objective, temperature=temperature).fit(
                        train, train_crop, args.run_dir / "checkpoints" / backend / pool / f"r13_crop_cnn_seed{seed}.pt"
                    )
                score = model.predict_scores(validation, validation_crop)
                outcomes, metric = _evaluate(validation, universe, score, f"r13_crop_cnn_seed{seed}")
                metric.update({"rung": "R13", "method": "r13_crop_cnn", "seed": seed, "backend": backend.upper(), "pool": pool, "feature_set": "F0-F6+candidate_crop", "parameter_count": model.artifact()["parameter_count"], "objective": objective, "temperature": temperature, "controlled_loss_source": str(loss_selection_path)})
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path, compress=3)
                scored = validation[["sample_id", "stable_candidate_id", "candidate_identity_sha256"]].copy()
                scored["reranker_score"] = score
                atomic_parquet(score_path, scored)
                atomic_parquet(root / f"outcomes_seed{seed}.parquet", outcomes)
                atomic_json(metric_path, metric)
                atomic_json(model_path.with_suffix(".json"), model.artifact())
                records.append(metric)
                score_parts.append(scored.rename(columns={"reranker_score": f"score_seed{seed}"}))
            ensemble = score_parts[0]
            for score in score_parts[1:]:
                ensemble = ensemble.merge(score.drop(columns=["candidate_identity_sha256"]), on=["sample_id", "stable_candidate_id"], validate="one_to_one")
            seed_columns = [f"score_seed{seed}" for seed in SEEDS]
            ensemble["reranker_score"] = ensemble[seed_columns].mean(axis=1)
            votes = np.zeros(len(ensemble), dtype=float)
            for column in seed_columns:
                selected_index = (
                    ensemble.sort_values(
                        ["sample_id", column, "stable_candidate_id"],
                        ascending=[True, False, True],
                        kind="mergesort",
                    )
                    .groupby("sample_id", sort=False)
                    .head(1)
                    .index
                )
                votes[selected_index] += 1.0
            ensemble["seed_agreement_fraction"] = votes / len(SEEDS)
            aligned_score = validation[["sample_id", "stable_candidate_id"]].merge(ensemble[["sample_id", "stable_candidate_id", "reranker_score"]], on=["sample_id", "stable_candidate_id"], validate="one_to_one")["reranker_score"].to_numpy(dtype=float)
            outcomes, metric = _evaluate(validation, universe, aligned_score, "r13_crop_cnn_ensemble")
            metric.update({"rung": "R13", "method": "r13_crop_cnn", "seed": "ensemble", "backend": backend.upper(), "pool": pool, "feature_set": "F0-F6+candidate_crop", "objective": objective, "temperature": temperature, "controlled_loss_source": str(loss_selection_path)})
            atomic_parquet(args.run_dir / "07_validation" / "crop_cnn" / backend / pool / "scores_ensemble.parquet", ensemble)
            atomic_parquet(args.run_dir / "07_validation" / "crop_cnn" / backend / pool / "outcomes_ensemble.parquet", outcomes)
            atomic_json(args.run_dir / "07_validation" / "crop_cnn" / backend / pool / "metrics_ensemble.json", metric)
            records.append(metric)
    pd.DataFrame(records).to_csv(args.run_dir / "07_validation" / "R13_CROP_CNN_RESULTS.csv", index=False)
    status_path = args.run_dir / "02_features" / "FEATURE_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    for family in status["families"]:
        if family["family"] == "F8":
            family.update({"status": "COMPLETE_WITH_FALLBACK", "fallback": "R13 candidate-aligned crop CNN complete; R12 backend/HiFi latent unavailable", "evidence": "audited 64x64 four-channel crop stores and R13 validation results; no stable pre-freeze latent hook contract"})
    atomic_json(status_path, status)
    pd.DataFrame(status["families"]).to_csv(args.run_dir / "02_features" / "FEATURE_STATUS.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
