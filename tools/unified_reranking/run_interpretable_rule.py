"""Run one resumable grouped-OOF or Validation R1 rule cell."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.datasets import FoldPreprocessor, join_development_features_and_labels
from unified_reranking.artifacts import (
    load_verified_json,
    verified_manifest_artifact,
    verify_artifact_records_recursive,
)
from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import evaluate_order_only
from unified_reranking.rules import RULE_ALIASES, SignConstrainedLinearUtility, single_rule_scores
from unified_reranking.telemetry import (
    flatten_telemetry,
    missing_feature_rate,
    resolved_track_extraction_latency,
    telemetry_payload,
)
from unified_reranking.training import FORMAL_SEEDS


METHODS = (*RULE_ALIASES, "sign_constrained_linear")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    parser.add_argument("--track", required=True, choices=("T1_native", "T2_matched_common", "T3_tri_backend"))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--alpha", required=True, type=float, choices=(0.25, 0.5, 1.0))
    parser.add_argument("--seed", required=True, type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--mode", required=True, choices=("oof", "validation"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--l2", type=float, default=1e-4)
    return parser.parse_args()


def _load(
    run_dir: Path, route: str, track: str, split: str
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    Path,
    Path,
    Path,
    float,
    dict[str, str] | None,
]:
    feature_dir = run_dir / "03_features" / "tracks" / track / f"{route}_{split}"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    feature_manifest = load_verified_json(
        feature_manifest_path, name=f"{route}/{track}/{split} rule feature manifest"
    )
    feature_path = verified_manifest_artifact(
        feature_manifest, name=f"{route}/{track}/{split} rule candidate features"
    )
    columns = assert_model_feature_columns(feature_manifest["model_feature_columns"])
    extraction_latency, extraction_source = resolved_track_extraction_latency(
        run_dir, track, feature_manifest
    )
    label_path = run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
    joined = join_development_features_and_labels(pd.read_parquet(feature_path), pd.read_parquet(label_path))
    return (
        joined,
        columns,
        feature_path,
        label_path,
        feature_manifest_path,
        extraction_latency,
        extraction_source,
    )


def _training_code_records() -> list[dict[str, str]]:
    paths = (
        Path(__file__).resolve(),
        SRC / "unified_reranking/datasets.py",
        SRC / "unified_reranking/metrics.py",
        SRC / "unified_reranking/rules.py",
        SRC / "unified_reranking/telemetry.py",
    )
    return [
        {"path": str(path), "sha256": sha256_file(path)} for path in paths
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    if (args.mode == "oof") != (args.fold is not None):
        raise ValueError("OOF mode requires --fold; Validation mode forbids it")
    run_dir = args.run_dir.resolve()
    feature_started = time.perf_counter()
    held_fold = int(args.fold) if args.mode == "oof" else None
    (
        train,
        columns,
        train_feature_path,
        train_label_path,
        train_manifest_path,
        train_extraction_latency,
        train_extraction_source,
    ) = _load(run_dir, args.route, args.track, "train")
    fold_path = run_dir / "04_splits" / "fold_assignments.parquet"
    prediction_feature_path = train_feature_path
    prediction_label_path = train_label_path
    prediction_manifest_path = train_manifest_path
    denominator_path: Path | None = None
    validation_rows: pd.DataFrame | None = None
    validation_extraction_latency: float | None = None
    validation_extraction_source: dict[str, str] | None = None
    if args.mode == "validation":
        (
            validation_rows,
            validation_columns,
            prediction_feature_path,
            prediction_label_path,
            prediction_manifest_path,
            validation_extraction_latency,
            validation_extraction_source,
        ) = _load(run_dir, args.route, args.track, "validation")
        if validation_columns != columns:
            raise RuntimeError("Train/Validation feature schemas differ")
        denominator_path = run_dir / "01_manifests" / "paired_validation.parquet"
    prediction_extraction_source = (
        train_extraction_source
        if args.mode == "oof"
        else validation_extraction_source
    )
    code_records = _training_code_records()
    source_identity = {
        "train_feature_manifest_sha256": sha256_file(train_manifest_path),
        "train_features_sha256": sha256_file(train_feature_path),
        "train_labels_sha256": sha256_file(train_label_path),
        "prediction_feature_manifest_sha256": sha256_file(prediction_manifest_path),
        "prediction_features_sha256": sha256_file(prediction_feature_path),
        "prediction_labels_sha256": sha256_file(prediction_label_path),
        "folds_sha256": sha256_file(fold_path),
        "feature_schema_sha256": canonical_sha256(columns),
        "training_code_sha256": canonical_sha256(code_records),
        "train_feature_extraction_benchmark_sha256": None
        if train_extraction_source is None
        else train_extraction_source["sha256"],
        "prediction_feature_extraction_benchmark_sha256": None
        if prediction_extraction_source is None
        else prediction_extraction_source["sha256"],
    }
    if denominator_path is not None:
        source_identity["validation_denominator_sha256"] = sha256_file(denominator_path)
    configuration = {
        "route": args.route,
        "track": args.track,
        "method": args.method,
        "alpha": args.alpha,
        "l2": args.l2,
        "seed": args.seed,
        "mode": args.mode,
        "held_fold": held_fold,
        "source_identity": source_identity,
    }
    cell_key = canonical_sha256(configuration)[:16]
    root = run_dir / ("06_oof" if args.mode == "oof" else "07_validation") / "rule_cells" / cell_key
    marker = root / "manifest.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("status") == "COMPLETE" and previous.get("configuration") == configuration:
            verify_artifact_records_recursive(
                previous.get("sources", {}),
                name="rule cell resumable sources",
                require_at_least_one=True,
            )
            verify_artifact_records_recursive(
                previous.get("artifacts", {}),
                name="rule cell resumable outputs",
                require_at_least_one=True,
            )
            return previous
    folds = pd.read_parquet(fold_path)[["sample_id", "fold"]]
    train = train.merge(folds, on="sample_id", how="left", validate="many_to_one")
    if train["fold"].isna().any():
        raise RuntimeError("fold assignment does not cover all training candidates")
    fit_rows = train.loc[train["fold"] != held_fold].copy() if held_fold is not None else train.copy()
    if args.mode == "oof":
        predict_rows = train.loc[train["fold"] == held_fold].copy()
        denominator = folds.loc[folds["fold"] == held_fold, "sample_id"].astype(str).tolist()
        prediction_feature_path = train_feature_path
        prediction_label_path = train_label_path
    else:
        if validation_rows is None or denominator_path is None:
            raise RuntimeError("Validation inputs were not initialized")
        predict_rows = validation_rows
        denominator = pd.read_parquet(
            denominator_path, columns=["sample_id"]
        )["sample_id"].astype(str).tolist()

    preprocessor = FoldPreprocessor.fit(fit_rows, columns)
    fit_matrix = preprocessor.transform(fit_rows)
    predict_matrix = preprocessor.transform(predict_rows)
    predict_base = pd.to_numeric(predict_rows["base_logit"], errors="raise").to_numpy(float)
    processed_rows = len(fit_rows) + len(predict_rows)
    if processed_rows <= 0:
        raise RuntimeError("rule cell has no candidates to measure")
    feature_latency_ms = (
        (time.perf_counter() - feature_started) * 1000.0 / processed_rows
    )
    feature_missing_rate = missing_feature_rate(
        pd.concat((fit_rows, predict_rows), ignore_index=True), columns
    )
    model_artifact: dict[str, object]
    if args.method == "sign_constrained_linear":
        model = SignConstrainedLinearUtility(alpha=args.alpha, l2=args.l2).fit(
            fit_matrix,
            fit_rows["candidate_success"].astype(int),
            fit_rows["sample_id"],
            pd.to_numeric(fit_rows["base_logit"], errors="raise"),
            columns,
        )
        model_artifact = model.artifact()
        parameter_count = len(model_artifact["weights"]) + 1
        ranker_started = time.perf_counter()
        scores = model.predict(predict_matrix, predict_base, columns)
    else:
        ranker_started = time.perf_counter()
        scores = single_rule_scores(
            predict_base,
            predict_matrix,
            columns,
            family=args.method,
            alpha=args.alpha,
        )
        model_artifact = {"method": args.method, "alpha": args.alpha}
        parameter_count = 0
    ranker_latency_ms = (
        (time.perf_counter() - ranker_started) * 1000.0 / len(predict_rows)
    )
    telemetry = telemetry_payload(
        phase=f"interpretable_rule_{args.mode}",
        parameter_count=parameter_count,
        ranker_latency_ms=ranker_latency_ms,
        feature_latency_ms=feature_latency_ms,
        missing_feature_rate_value=feature_missing_rate,
    )
    feature_extraction_latency_ms = (
        train_extraction_latency
        if args.mode == "oof"
        else validation_extraction_latency
    )
    if feature_extraction_latency_ms is None:
        raise RuntimeError("rule feature extraction latency was not resolved")

    predictions = predict_rows[["sample_id", "candidate_id"]].copy()
    predictions["score"] = np.asarray(scores, dtype=float)
    evaluation_input = predict_rows[["sample_id", "candidate_id", "native_rank", "candidate_success"]].merge(
        predictions, on=["sample_id", "candidate_id"], validate="one_to_one"
    )
    metrics, decisions = evaluate_order_only(denominator, evaluation_input, score_column="score")
    prediction_path = root / "predictions.parquet"
    decision_path = root / "per_sample_decisions.parquet"
    _atomic_parquet(prediction_path, predictions)
    _atomic_parquet(decision_path, decisions)
    result: dict[str, object] = {
        "status": "COMPLETE",
        "configuration": configuration,
        "cell_key": cell_key,
        "feature_columns": list(columns),
        "feature_schema_sha256": canonical_sha256(columns),
        "preprocessor": preprocessor.artifact(),
        "model": model_artifact,
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "feature_extraction_latency_ms": float(feature_extraction_latency_ms),
        "metrics": metrics,
        "fit_candidate_rows": len(fit_rows),
        "prediction_candidate_rows": len(predictions),
        "sources": {
            "train_features": {"path": str(train_feature_path), "sha256": sha256_file(train_feature_path)},
            "train_feature_manifest": {"path": str(train_manifest_path), "sha256": sha256_file(train_manifest_path)},
            "train_labels": {"path": str(train_label_path), "sha256": sha256_file(train_label_path)},
            "prediction_features": {"path": str(prediction_feature_path), "sha256": sha256_file(prediction_feature_path)},
            "prediction_feature_manifest": {"path": str(prediction_manifest_path), "sha256": sha256_file(prediction_manifest_path)},
            "train_feature_extraction_benchmark": train_extraction_source,
            "prediction_feature_extraction_benchmark": prediction_extraction_source,
            "prediction_labels": {"path": str(prediction_label_path), "sha256": sha256_file(prediction_label_path)},
            "folds": {"path": str(fold_path), "sha256": sha256_file(fold_path)},
            "validation_denominator": None
            if denominator_path is None
            else {"path": str(denominator_path), "sha256": sha256_file(denominator_path)},
            "training_code": code_records,
        },
        "artifacts": {
            "predictions": {"path": str(prediction_path.resolve()), "sha256": sha256_file(prediction_path)},
            "decisions": {"path": str(decision_path.resolve()), "sha256": sha256_file(decision_path)},
        },
    }
    atomic_json(marker, result)
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    substage = f"rule_{args.mode}_{args.route}_{args.track}_{args.method}_{args.alpha}_{args.seed}_{args.fold}"
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P7",
        substage=substage,
        route=args.route,
        evidence_track=args.track,
        pool="top5",
        method=args.method,
        feature_set="interpretable_r1",
        loss="fixed_rule" if args.method != "sign_constrained_linear" else "query_equal_bce",
        encoder="rule",
        seed=args.seed,
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args)
        artifact = Path(str(result["artifacts"]["predictions"]["path"]))
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    manifest_path = artifact.parent / "manifest.json"
    print(
        json.dumps(
            {
                "status": result["status"],
                "cell_key": result["cell_key"],
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
