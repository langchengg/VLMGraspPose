"""Apply the selected three-seed union ranker to label-free Top-15 Test features."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import torch

from tools.unified_reranking.prepare_union_features import TRACK, run_split
from unified_reranking.datasets import FoldPreprocessor, build_inference_query_arrays
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import select_order_only
from unified_reranking.models import DeepSetsResidualScorer
from unified_reranking.training import FORMAL_SEEDS, predict_neural_ranker, set_deterministic_cpu
from unified_reranking.test_access_guard import append_access_log


_FORBIDDEN_TOKENS = (
    "candidate_success",
    "jacquard_margin",
    "matched_gt",
    "ground_truth",
    "_correct",
)


def _native_lightgbm_scores(model_path: Path, features: np.ndarray) -> np.ndarray:
    """Predict in a fresh process whose native runtime imports LightGBM first."""

    script = """
import sys
from pathlib import Path

import numpy as np

from tools.unified_reranking.apply_locked_matrix_cell import _load_native_lightgbm_ranker

model = _load_native_lightgbm_ranker(Path(sys.argv[1]))
features = np.load(sys.argv[2], allow_pickle=False)
scores = model.predict(features)
np.save(sys.argv[3], scores, allow_pickle=False)
"""
    matrix = np.asarray(features, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="union-lightgbm-") as directory:
        temporary = Path(directory)
        feature_path = temporary / "features.npy"
        score_path = temporary / "scores.npy"
        np.save(feature_path, matrix, allow_pickle=False)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(model_path),
                str(feature_path),
                str(score_path),
            ],
            text=True,
            capture_output=True,
            check=False,
            cwd=ROOT,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                "isolated LightGBM union prediction failed"
                + (f": {detail}" if detail else "")
            )
        if not score_path.is_file():
            raise RuntimeError("isolated LightGBM union prediction wrote no scores")
        scores = np.asarray(
            np.load(score_path, allow_pickle=False), dtype=np.float64
        )
    if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
        raise RuntimeError("isolated LightGBM union prediction returned invalid scores")
    return scores


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _predict_cell(cell: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    configuration = cell["configuration"]
    encoder = str(configuration["encoder"])
    seed = int(configuration["seed"])
    preprocessor = FoldPreprocessor.from_artifact(cell["preprocessor"])
    if tuple(map(str, cell["feature_columns"])) != preprocessor.columns:
        raise RuntimeError("union cell feature/preprocessor schemas differ")
    arrays = build_inference_query_arrays(
        features,
        preprocessor=preprocessor,
        max_candidates=15,
    )
    model_record = cell["artifacts"]["model"]
    model_path = Path(model_record["path"])
    if sha256_file(model_path) != model_record["sha256"]:
        raise RuntimeError("locked union model hash mismatch")
    if encoder == "lambdamart":
        valid = ~arrays.padding_mask.numpy()
        scores = _native_lightgbm_scores(
            model_path, arrays.features.numpy()[valid]
        )
        rows: list[dict[str, object]] = []
        cursor = 0
        for sample_id, candidate_ids in zip(arrays.sample_ids, arrays.candidate_ids, strict=True):
            for candidate_id in candidate_ids:
                rows.append({"sample_id": sample_id, "candidate_id": candidate_id, "score": float(scores[cursor])})
                cursor += 1
        if cursor != len(scores):
            raise RuntimeError("union LambdaMART Test scores do not cover candidates")
        return pd.DataFrame(rows)
    if encoder != "deepsets":
        raise ValueError(f"unsupported locked union encoder: {encoder}")
    set_deterministic_cpu(seed)
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    model = DeepSetsResidualScorer(int(payload["input_dim"]), alpha=float(payload["alpha"]))
    model.load_state_dict(payload["state_dict"])
    return predict_neural_ranker(model, arrays)


def _resume(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE" or value.get("signature_sha256") != signature:
        raise RuntimeError("immutable union Test output exists with a different signature")
    for record in value.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("resumable union Test artifact hash mismatch")
    return value


def run(
    run_dir: Path,
    *,
    selection_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    selection_path = (
        selection_path or run_dir / "08_lock" / "union_ranker" / "selected_union_ranker.json"
    ).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "VALIDATION_LOCKED" or selection.get("test_access") != "NONE":
        raise RuntimeError("union ranker is not Validation-locked")
    test_manifest = run_split(run_dir, "test")
    if test_manifest.get("candidate_test_labels_read") is not False:
        raise PermissionError("union Test feature artifact lacks a label-free certificate")
    feature_path = Path(test_manifest["artifacts"]["features"]["path"])
    features = pd.read_parquet(feature_path)
    forbidden = sorted(
        column
        for column in features.columns
        if any(token in column.lower() for token in _FORBIDDEN_TOKENS)
    )
    if forbidden:
        raise PermissionError(f"union Test features contain supervision: {forbidden}")
    selected_validation_record = selection.get("selected_validation_manifest")
    if not isinstance(selected_validation_record, dict):
        raise RuntimeError("union selection lacks its locked Validation manifest")
    selected_validation_path = Path(
        str(selected_validation_record.get("path", ""))
    ).resolve()
    if (
        not selected_validation_path.is_file()
        or sha256_file(selected_validation_path)
        != selected_validation_record.get("sha256")
    ):
        raise RuntimeError("selected union Validation manifest hash mismatch")
    selected_validation = json.loads(
        selected_validation_path.read_text(encoding="utf-8")
    )
    if (
        selected_validation.get("status") != "COMPLETE"
        or selected_validation.get("identity", {}).get("encoder")
        != selection.get("selected_encoder")
        or canonical_sha256(selection.get("selected_validation_ensemble"))
        != canonical_sha256(selected_validation)
    ):
        raise RuntimeError("embedded union ensemble differs from its locked manifest")
    cell_records = selected_validation["sources"]["cells"]
    cells_by_seed: dict[int, tuple[dict[str, Any], Path]] = {}
    for record in cell_records:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("selected union cell manifest hash mismatch")
        cell = json.loads(path.read_text(encoding="utf-8"))
        config = cell.get("configuration", {})
        if (
            cell.get("status") != "COMPLETE"
            or config.get("mode") != "validation"
            or config.get("encoder") != selection["selected_encoder"]
        ):
            raise RuntimeError("selected union cell contract mismatch")
        seed = int(config["seed"])
        if seed in cells_by_seed:
            raise RuntimeError("duplicate selected union Validation seed")
        cells_by_seed[seed] = (cell, path)
    if set(cells_by_seed) != set(FORMAL_SEEDS):
        raise RuntimeError("selected union ranker must contain exactly three formal seeds")
    denominator_path = run_dir / "01_manifests" / "paired_test.parquet"
    sources = {
        "selection": _record(selection_path),
        "test_feature_manifest": _record(
            run_dir / "03_features" / "tracks" / TRACK / "union_test" / "feature_manifest.json"
        ),
        "test_features": _record(feature_path),
        "implementation_tool": _record(Path(__file__)),
        "cells": {str(seed): _record(path) for seed, (_, path) in cells_by_seed.items()},
        "denominator": _record(denominator_path),
    }
    configuration = {
        "encoder": selection["selected_encoder"],
        "seeds": list(FORMAL_SEEDS),
        "pool": "primary_union_top15_no_dedup",
        "label_free_test": True,
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output_dir = (output_dir or run_dir / "08_lock" / "union_ranker_test").resolve()
    marker = output_dir / "manifest.json"
    resumed = _resume(marker, signature)
    if resumed is not None:
        return resumed
    predictions = features[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ]
    ].copy()
    for seed in FORMAL_SEEDS:
        frame = _predict_cell(cells_by_seed[seed][0], features).rename(
            columns={"score": f"score_seed_{seed}"}
        )
        predictions = predictions.merge(
            frame,
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
    score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
    predictions["ensemble_score"] = predictions[score_columns].mean(axis=1)
    denominator = pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"].astype(str).tolist()
    decisions = select_order_only(
        denominator,
        predictions,
        score_column="ensemble_score",
    )
    selected = predictions[
        [
            "sample_id",
            "candidate_id",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ]
    ].rename(columns={"candidate_id": "selected_candidate_id"})
    decisions = decisions.merge(
        selected,
        on=["sample_id", "selected_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    decisions["prediction_source"] = "test_label_free"
    prediction_path = output_dir / "per_candidate_scores.parquet"
    decision_path = output_dir / "per_sample_decisions.parquet"
    _atomic_parquet(prediction_path, predictions)
    _atomic_parquet(decision_path, decisions)
    manifest = {
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": configuration,
        "sources": sources,
        "sample_count": int(len(decisions)),
        "candidate_rows": int(len(predictions)),
        "artifacts": {
            "predictions": _record(prediction_path),
            "decisions": _record(decision_path),
        },
        "candidate_test_labels_read": False,
        "test_access": "LABEL_FREE_INFERENCE_ONLY",
    }
    atomic_json(marker, manifest)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "union_ranker_test_application",
            "output_manifest": str(marker.resolve()),
            "output_manifest_sha256": sha256_file(marker),
            "candidate_labels_opened_as_table": False,
        },
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P11_PRELOCK",
        substage="apply_locked_union_ranker",
        route="cross_route",
        evidence_track=TRACK,
        pool="primary_union_top15_no_dedup",
        method="selected_union_ranker",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(run_dir, selection_path=args.selection, output_dir=args.output_dir)
        marker = Path(args.output_dir or run_dir / "08_lock" / "union_ranker_test") / "manifest.json"
        state["artifact_path"] = str(marker.resolve())
        state["artifact_sha256"] = sha256_file(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
