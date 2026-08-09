"""Assemble leakage-safe per-sample inputs for the selected route gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verified_manifest_artifact,
    verify_artifact_records_recursive,
)
from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS
from unified_reranking.hashing import atomic_json, sha256_file
from unified_reranking.ledger import ledger_stage


ROUTES = ("crog", "g1", "c1")
FEATURE_COLUMNS = SAFE_GATE_FEATURE_COLUMNS


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _manifest(path: Path) -> dict[str, Any]:
    value = load_verified_json(path, name="selected ensemble manifest")
    verify_artifact_records_recursive(
        value.get("sources", {}),
        name=f"selected ensemble sources {path}",
        require_at_least_one=True,
    )
    verify_artifact_records_recursive(
        value.get("artifacts", {}),
        name=f"selected ensemble artifacts {path}",
        require_at_least_one=True,
    )
    return value


def _candidate_features(run_dir: Path, route: str, split: str) -> pd.DataFrame:
    feature_dir = (
        run_dir
        / "03_features"
        / "tracks"
        / "T2_matched_common"
        / f"{route}_{split}"
    )
    manifest = load_verified_json(
        feature_dir / "feature_manifest.json",
        name=f"{route}/{split} T2 gate feature manifest",
    )
    path = verified_manifest_artifact(
        manifest, name=f"{route}/{split} T2 gate candidate features"
    )
    columns = [
        "sample_id",
        "candidate_id",
        "calibrated_native_probability",
        "native_score_raw",
        "overall_feature_reliability",
        "peak_retention_rate",
        "perturbed_valid_fraction",
        "mask_reliability",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["perturbation_stability"] = frame[
        ["peak_retention_rate", "perturbed_valid_fraction"]
    ].min(axis=1)
    return frame.drop(columns=["peak_retention_rate", "perturbed_valid_fraction"])


def _native_correct(run_dir: Path, route: str, split: str) -> pd.DataFrame:
    candidates = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_{split}_top5.parquet",
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_geometry_sha256",
        ],
    )
    labels = pd.read_parquet(
        run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet",
        columns=["sample_id", "candidate_id", "candidate_success"],
    )
    top = candidates.loc[candidates["native_rank"].eq(1)].merge(
        labels, on=["sample_id", "candidate_id"], validate="one_to_one"
    )
    return top.rename(
        columns={
            "candidate_id": "native_candidate_id",
            "candidate_geometry_sha256": "native_geometry_sha256",
            "candidate_success": "native_correct",
        }
    )[
        [
            "sample_id",
            "native_candidate_id",
            "native_geometry_sha256",
            "native_correct",
        ]
    ]


def _prefixed_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            "candidate_id": f"{prefix}_candidate_id",
            "calibrated_native_probability": f"{prefix}_calibrated_probability",
            "native_score_raw": f"{prefix}_native_score",
            "overall_feature_reliability": f"{prefix}_overall_reliability",
            "perturbation_stability": f"{prefix}_perturbation_stability",
            "mask_reliability": f"{prefix}_mask_reliability",
        }
    )


def _build_split(
    run_dir: Path,
    route: str,
    split: str,
    ensemble_manifest: dict[str, Any],
) -> pd.DataFrame:
    decisions = pd.read_parquet(
        verified_artifact_path(
            ensemble_manifest["artifacts"]["decisions"],
            name=f"{route}/{split} ensemble decisions",
        )
    )
    paired = pd.read_parquet(
        run_dir / "01_manifests" / f"paired_{split}.parquet",
        columns=["sample_id", "scene_id"],
    )
    output = paired.merge(decisions, on="sample_id", how="left", validate="one_to_one")
    if output["selected_correct"].isna().any():
        raise RuntimeError("ensemble decisions do not preserve the full sample denominator")
    output = output.merge(
        _native_correct(run_dir, route, split)[["sample_id", "native_correct"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    output["native_correct"] = output["native_correct"].fillna(False).astype(bool)
    output["challenger_correct"] = output["selected_correct"].astype(bool)
    output["challenger_candidate_id"] = output["selected_candidate_id"]
    features = _candidate_features(run_dir, route, split)
    native = _prefixed_features(features, "native")
    challenger = _prefixed_features(features, "challenger")
    output = output.merge(
        native,
        on=["sample_id", "native_candidate_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        challenger,
        on=["sample_id", "challenger_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    candidates = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_{split}_top5.parquet",
        columns=["sample_id", "candidate_id", "candidate_geometry_sha256"],
    ).rename(
        columns={
            "candidate_id": "challenger_candidate_id",
            "candidate_geometry_sha256": "locked_challenger_geometry_sha256",
        }
    )
    output = output.merge(
        candidates,
        on=["sample_id", "challenger_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    exists = output["challenger_exists"].fillna(False).astype(bool)
    output["candidate_id_unchanged"] = (
        output["challenger_candidate_id"].notna()
        & output["locked_challenger_geometry_sha256"].notna()
    )
    output["geometry_hash_unchanged"] = (
        output["selected_geometry_sha256"].fillna("").astype(str)
        == output["locked_challenger_geometry_sha256"].fillna("").astype(str)
    ) & output["candidate_id_unchanged"]
    # No-output and no-challenger rows are forced to zero evidence and cannot switch.
    scalar = [
        "native_calibrated_probability",
        "native_native_score",
        "native_overall_reliability",
        "native_perturbation_stability",
        "native_mask_reliability",
        "challenger_calibrated_probability",
        "challenger_native_score",
        "challenger_overall_reliability",
        "challenger_perturbation_stability",
        "challenger_mask_reliability",
    ]
    output[scalar] = output[scalar].fillna(0.0)
    output["ranker_score_margin"] = output["ensemble_score_margin"].fillna(0.0)
    output["calibrated_probability_delta"] = (
        output["challenger_calibrated_probability"]
        - output["native_calibrated_probability"]
    )
    output["native_score_delta"] = (
        output["challenger_native_score"] - output["native_native_score"]
    )
    output["overall_reliability_delta"] = (
        output["challenger_overall_reliability"]
        - output["native_overall_reliability"]
    )
    output["perturbation_stability_delta"] = (
        output["challenger_perturbation_stability"]
        - output["native_perturbation_stability"]
    )
    output["mask_reliability_delta"] = (
        output["challenger_mask_reliability"] - output["native_mask_reliability"]
    )
    output["challenger_exists_numeric"] = exists.astype(float)
    output["score_margin"] = output["ranker_score_margin"]
    output["challenger_reliability"] = output[
        "challenger_overall_reliability"
    ].clip(0.0, 1.0)
    output["perturbation_stability"] = output[
        "challenger_perturbation_stability"
    ].clip(0.0, 1.0)
    output["prediction_source"] = "train_oof" if split == "train" else "validation"
    if split == "train":
        folds = pd.read_parquet(
            run_dir / "04_splits" / "fold_assignments.parquet",
            columns=["sample_id", "fold"],
        ).rename(columns={"fold": "oof_fold"})
        output = output.merge(folds, on="sample_id", validate="one_to_one")
    columns = [
        "sample_id",
        "scene_id",
        "prediction_source",
        *( ["oof_fold"] if split == "train" else [] ),
        "native_correct",
        "challenger_correct",
        "native_candidate_id",
        "challenger_candidate_id",
        "score_margin",
        "challenger_reliability",
        "perturbation_stability",
        "seed_challenger_votes",
        "candidate_id_unchanged",
        "geometry_hash_unchanged",
        "challenger_exists",
        *FEATURE_COLUMNS,
    ]
    result = output.loc[:, list(dict.fromkeys(columns))].copy()
    result["challenger_candidate_id"] = result["challenger_candidate_id"].fillna("")
    result["native_candidate_id"] = result["native_candidate_id"].fillna("")
    numeric = result.loc[:, FEATURE_COLUMNS].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("gate inference features are not finite")
    return result


def run(run_dir: Path) -> dict[str, Any]:
    selection_path = run_dir / "07_validation" / "selected_primary_ungated.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "VALIDATION_LOCKED":
        raise RuntimeError("primary ungated ranker selection is not Validation-locked")
    selections = selection["selections"]
    if set(selections) != set(ROUTES):
        raise RuntimeError("primary ungated selection must contain exactly three routes")
    artifacts: dict[str, Any] = {}
    for route in ROUTES:
        selected = selections[route]
        train_manifest_path = verified_artifact_path(
            {
                "path": selected.get("oof_manifest"),
                "sha256": selected.get("oof_manifest_sha256"),
            },
            name=f"{route} selected OOF ensemble manifest",
        )
        validation_manifest_path = verified_artifact_path(
            {
                "path": selected.get("validation_manifest"),
                "sha256": selected.get("validation_manifest_sha256"),
            },
            name=f"{route} selected Validation ensemble manifest",
        )
        train = _build_split(run_dir, route, "train", _manifest(train_manifest_path))
        validation = _build_split(
            run_dir, route, "validation", _manifest(validation_manifest_path)
        )
        output = run_dir / "07_validation" / "gate_inputs" / route
        train_path = output / "train_oof.parquet"
        validation_path = output / "validation.parquet"
        _atomic_parquet(train_path, train)
        _atomic_parquet(validation_path, validation)
        manifest = {
            "status": "COMPLETE",
            "route": route,
            "feature_columns": list(FEATURE_COLUMNS),
            "test_access": "NONE",
            "sources": {
                "selection": {"path": str(selection_path.resolve()), "sha256": sha256_file(selection_path)},
                "oof_ensemble": {"path": str(train_manifest_path.resolve()), "sha256": sha256_file(train_manifest_path)},
                "validation_ensemble": {"path": str(validation_manifest_path.resolve()), "sha256": sha256_file(validation_manifest_path)},
            },
            "artifacts": {
                "train_oof": {"path": str(train_path.resolve()), "sha256": sha256_file(train_path)},
                "validation": {"path": str(validation_path.resolve()), "sha256": sha256_file(validation_path)},
            },
        }
        atomic_json(output / "manifest.json", manifest)
        artifacts[route] = manifest["artifacts"]
    summary = {
        "status": "COMPLETE",
        "feature_columns": list(FEATURE_COLUMNS),
        "routes": artifacts,
        "test_access": "NONE",
    }
    atomic_json(run_dir / "07_validation" / "gate_inputs" / "manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P8",
        substage="prepare_gate_inputs",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(run_dir)
        artifact = run_dir / "07_validation" / "gate_inputs" / "manifest.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
