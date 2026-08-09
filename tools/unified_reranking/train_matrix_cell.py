"""Train one resumable grouped-OOF or Validation matrix cell on CPU."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.datasets import (
    FoldPreprocessor,
    build_query_arrays,
    join_development_features_and_labels,
    with_query_edge_features,
)
from unified_reranking.artifacts import (
    load_verified_json,
    verified_manifest_artifact,
    verify_artifact_records_recursive,
)
from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import evaluate_order_only
from unified_reranking.models import (
    CompleteGraphGNNResidualScorer,
    DeepSetsResidualScorer,
    LightGBMLambdaRank,
    LinearResidualScorer,
    ResidualMLPScorer,
    SetTransformerResidualScorer,
)
from unified_reranking.training import (
    FORMAL_SEEDS,
    NeuralTrainingConfig,
    fit_neural_ranker,
    predict_neural_ranker,
    set_deterministic_cpu,
)
from unified_reranking.telemetry import (
    flatten_telemetry,
    lightgbm_parameter_count,
    missing_feature_rate,
    resolved_track_extraction_latency,
    telemetry_payload,
    torch_parameter_count,
)


NEURAL_ENCODERS = ("linear", "mlp", "deepsets", "set_transformer", "gnn")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    parser.add_argument(
        "--track",
        required=True,
        choices=("T1_native", "T2_matched_common", "T3_tri_backend"),
    )
    parser.add_argument(
        "--encoder", required=True, choices=(*NEURAL_ENCODERS, "lambdamart")
    )
    parser.add_argument(
        "--loss",
        required=True,
        choices=("bce", "ranknet", "listwise", "jacquard_margin_ranknet", "lambdarank"),
    )
    parser.add_argument("--seed", required=True, type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--mode", required=True, choices=("oof", "validation"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--tree-learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--num-attention-blocks", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def _build_model(
    encoder: str,
    input_dim: int,
    alpha: float,
    *,
    num_attention_blocks: int = 2,
    edge_dim: int = 1,
) -> torch.nn.Module:
    if encoder == "linear":
        return LinearResidualScorer(input_dim, alpha=alpha)
    if encoder == "mlp":
        return ResidualMLPScorer(input_dim, alpha=alpha)
    if encoder == "deepsets":
        return DeepSetsResidualScorer(input_dim, alpha=alpha)
    if encoder == "set_transformer":
        return SetTransformerResidualScorer(
            input_dim, alpha=alpha, num_blocks=num_attention_blocks
        )
    if encoder == "gnn":
        return CompleteGraphGNNResidualScorer(input_dim, edge_dim, alpha=alpha)
    raise ValueError(f"not a neural encoder: {encoder}")


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
    feature_path = feature_dir / "candidate_features.parquet"
    manifest_path = feature_dir / "feature_manifest.json"
    manifest = load_verified_json(
        manifest_path, name=f"{route}/{track}/{split} feature manifest"
    )
    feature_path = verified_manifest_artifact(
        manifest,
        name=f"{route}/{track}/{split} candidate features",
    )
    columns = tuple(map(str, manifest["model_feature_columns"]))
    extraction_latency, extraction_latency_source = (
        resolved_track_extraction_latency(run_dir, track, manifest)
    )
    labels_path = (
        run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
    )
    joined = join_development_features_and_labels(
        pd.read_parquet(feature_path), pd.read_parquet(labels_path)
    )
    return (
        joined,
        columns,
        feature_path,
        labels_path,
        manifest_path,
        float(extraction_latency),
        extraction_latency_source,
    )


def _training_code_records() -> list[dict[str, str]]:
    paths = (
        Path(__file__).resolve(),
        SRC / "unified_reranking" / "datasets.py",
        SRC / "unified_reranking" / "metrics.py",
        SRC / "unified_reranking" / "training.py",
        SRC / "unified_reranking" / "models" / "core.py",
        SRC / "unified_reranking" / "models" / "lightgbm_ranker.py",
        SRC / "unified_reranking" / "telemetry.py",
    )
    return [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


def _flat_matrix(arrays) -> tuple[np.ndarray, np.ndarray, list[str]]:
    valid = ~arrays.padding_mask.numpy()
    features = arrays.features.numpy()[valid]
    labels = arrays.labels.numpy()[valid].astype(np.int32)
    query_ids = [
        sample_id
        for sample_id, ids in zip(arrays.sample_ids, arrays.candidate_ids)
        for _ in ids
    ]
    return features, labels, query_ids


def _relation_columns(
    frame: pd.DataFrame, *, route: str, track: str
) -> tuple[str, ...]:
    identity = {"sample_id", "source_candidate_id", "target_candidate_id"}
    columns = [
        str(column)
        for column in frame.columns
        if str(column) not in identity and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if route == "crog" and track == "T1_native":
        columns = [column for column in columns if "depth" not in column.lower()]
    return assert_model_feature_columns(columns)


def _load_relations(run_dir: Path, route: str, split: str) -> tuple[pd.DataFrame, Path]:
    path = (
        run_dir
        / "03_features"
        / "common"
        / f"{route}_{split}"
        / "candidate_relations.parquet"
    )
    return pd.read_parquet(path), path


def run(
    args: argparse.Namespace,
    *,
    feature_columns_override: Sequence[str] | None = None,
    output_parent: Path | None = None,
    analysis_contract: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    if (args.mode == "oof") != (args.fold is not None):
        raise ValueError("OOF mode requires --fold; Validation mode forbids it")
    if args.encoder == "lambdamart" and args.loss != "lambdarank":
        raise ValueError("LambdaMART requires lambdarank loss")
    if args.encoder != "lambdamart" and args.loss == "lambdarank":
        raise ValueError("lambdarank loss requires LambdaMART")
    run_dir = args.run_dir.resolve()
    feature_started = time.perf_counter()
    (
        train,
        columns,
        train_feature_path,
        train_labels_path,
        train_feature_manifest_path,
        train_feature_extraction_latency_ms,
        train_feature_extraction_latency_source,
    ) = _load(run_dir, args.route, args.track, "train")
    full_feature_columns = columns
    if feature_columns_override is not None:
        requested = assert_model_feature_columns(feature_columns_override)
        unknown = sorted(set(requested).difference(full_feature_columns))
        if unknown:
            raise ValueError(
                f"feature override is not a subset of the locked schema: {unknown}"
            )
        columns = requested
    folds_path = run_dir / "04_splits" / "fold_assignments.parquet"
    folds = pd.read_parquet(folds_path)[["sample_id", "fold"]]
    train = train.merge(folds, on="sample_id", how="left", validate="many_to_one")
    if train["fold"].isna().any():
        raise RuntimeError("fold assignment does not cover all training candidates")

    held_fold = int(args.fold) if args.mode == "oof" else None
    early_fold = (held_fold + 1) % 5 if held_fold is not None else 0
    fit_rows = train.loc[
        (train["fold"] != early_fold)
        & ((train["fold"] != held_fold) if held_fold is not None else True)
    ].copy()
    early_rows = train.loc[train["fold"] == early_fold].copy()
    validation_feature_path: Path | None = None
    validation_labels_path: Path | None = None
    validation_feature_manifest_path: Path | None = None
    validation_feature_extraction_latency_source: dict[str, str] | None = None
    denominator_path: Path | None = None
    if args.mode == "oof":
        predict_rows = train.loc[train["fold"] == held_fold].copy()
        denominator_ids = (
            folds.loc[folds["fold"] == held_fold, "sample_id"].astype(str).tolist()
        )
    else:
        (
            predict_rows,
            validation_columns,
            validation_feature_path,
            validation_labels_path,
            validation_feature_manifest_path,
            validation_feature_extraction_latency_ms,
            validation_feature_extraction_latency_source,
        ) = _load(run_dir, args.route, args.track, "validation")
        if validation_columns != full_feature_columns:
            raise RuntimeError("Train/Validation feature schemas differ")
        denominator_path = run_dir / "01_manifests" / "paired_validation.parquet"
        denominator_ids = (
            pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
            .astype(str)
            .tolist()
        )
    feature_extraction_latency_ms = (
        train_feature_extraction_latency_ms
        if args.mode == "oof"
        else validation_feature_extraction_latency_ms
    )

    preprocessor = FoldPreprocessor.fit(fit_rows, columns)
    fit_arrays = build_query_arrays(fit_rows, preprocessor=preprocessor)
    early_arrays = build_query_arrays(early_rows, preprocessor=preprocessor)
    predict_arrays = build_query_arrays(predict_rows, preprocessor=preprocessor)

    relation_metadata: dict[str, object] | None = None
    relation_path: Path | None = None
    relation_columns: tuple[str, ...] = ()
    if args.encoder == "gnn":
        train_relations, train_relation_path = _load_relations(
            run_dir, args.route, "train"
        )
        fit_ids = set(fit_rows["sample_id"].astype(str))
        early_ids = set(early_rows["sample_id"].astype(str))
        fit_relations = train_relations.loc[
            train_relations["sample_id"].astype(str).isin(fit_ids)
        ].copy()
        early_relations = train_relations.loc[
            train_relations["sample_id"].astype(str).isin(early_ids)
        ].copy()
        relation_columns = _relation_columns(
            train_relations, route=args.route, track=args.track
        )
        relation_preprocessor = FoldPreprocessor.fit(fit_relations, relation_columns)
        fit_arrays = with_query_edge_features(
            fit_arrays, fit_relations, preprocessor=relation_preprocessor
        )
        early_arrays = with_query_edge_features(
            early_arrays, early_relations, preprocessor=relation_preprocessor
        )
        if args.mode == "oof":
            predict_ids = set(predict_rows["sample_id"].astype(str))
            predict_relations = train_relations.loc[
                train_relations["sample_id"].astype(str).isin(predict_ids)
            ].copy()
            relation_path = train_relation_path
        else:
            predict_relations, relation_path = _load_relations(
                run_dir, args.route, "validation"
            )
        predict_arrays = with_query_edge_features(
            predict_arrays, predict_relations, preprocessor=relation_preprocessor
        )
        relation_metadata = {
            "columns": list(relation_columns),
            "schema_sha256": canonical_sha256(relation_columns),
            "preprocessor": relation_preprocessor.artifact(),
            "train_path": str(train_relation_path.resolve()),
            "train_sha256": sha256_file(train_relation_path),
            "prediction_path": str(relation_path.resolve()),
            "prediction_sha256": sha256_file(relation_path),
        }

    processed_candidate_rows = len(fit_rows) + len(early_rows) + len(predict_rows)
    if processed_candidate_rows <= 0:
        raise RuntimeError("training cell has no candidates to measure")
    feature_latency_ms = (
        (time.perf_counter() - feature_started) * 1000.0 / processed_candidate_rows
    )
    selected_missing_rate = missing_feature_rate(
        pd.concat((fit_rows, early_rows, predict_rows), ignore_index=True), columns
    )

    code_records = _training_code_records()

    source_identity: dict[str, Any] = {
        "train_features_sha256": sha256_file(train_feature_path),
        "train_feature_manifest_sha256": sha256_file(train_feature_manifest_path),
        "train_feature_columns": list(full_feature_columns),
        "train_feature_schema_sha256": canonical_sha256(full_feature_columns),
        "train_labels_sha256": sha256_file(train_labels_path),
        "folds_sha256": sha256_file(folds_path),
        "selected_feature_columns": list(columns),
        "selected_feature_schema_sha256": canonical_sha256(columns),
        "tool_sha256": code_records[0]["sha256"],
        "training_code_sha256": canonical_sha256(code_records),
    }
    if train_feature_extraction_latency_source is not None:
        source_identity["train_feature_extraction_benchmark_sha256"] = (
            train_feature_extraction_latency_source["sha256"]
        )
    if validation_feature_path is not None and validation_labels_path is not None:
        source_identity.update(
            {
                "validation_features_sha256": sha256_file(validation_feature_path),
                "validation_feature_manifest_sha256": sha256_file(
                    validation_feature_manifest_path
                ),
                "validation_feature_columns": list(validation_columns),
                "validation_feature_schema_sha256": canonical_sha256(
                    validation_columns
                ),
                "validation_labels_sha256": sha256_file(validation_labels_path),
                "validation_denominator_sha256": sha256_file(denominator_path),
            }
        )
        if validation_feature_extraction_latency_source is not None:
            source_identity["validation_feature_extraction_benchmark_sha256"] = (
                validation_feature_extraction_latency_source["sha256"]
            )
    if relation_metadata is not None:
        source_identity.update(
            {
                "train_relations_sha256": str(relation_metadata["train_sha256"]),
                "prediction_relations_sha256": str(
                    relation_metadata["prediction_sha256"]
                ),
            }
        )
    configuration = {
        "route": args.route,
        "track": args.track,
        "encoder": args.encoder,
        "loss": args.loss,
        "seed": args.seed,
        "mode": args.mode,
        "held_fold": held_fold,
        "early_stop_fold": early_fold,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "beta": args.beta,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_leaves": args.num_leaves,
        "tree_learning_rate": args.tree_learning_rate,
        "n_estimators": args.n_estimators,
        "num_attention_blocks": args.num_attention_blocks,
        "analysis_contract": None
        if analysis_contract is None
        else dict(analysis_contract),
        "source_identity": source_identity,
    }
    cell_key = canonical_sha256(configuration)[:16]
    root = (
        Path(output_parent).resolve()
        if output_parent is not None
        else run_dir
        / ("06_oof" if args.mode == "oof" else "07_validation")
        / "matrix_cells"
    ) / cell_key
    marker = root / "manifest.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "COMPLETE"
            and previous.get("configuration") == configuration
        ):
            verify_artifact_records_recursive(
                {
                    "sources": previous.get("sources"),
                    "artifacts": previous.get("artifacts"),
                },
                name=f"completed matrix cell {cell_key}",
                require_at_least_one=True,
            )
            return previous

    if args.encoder == "lambdamart":
        fit_x, fit_y, fit_q = _flat_matrix(fit_arrays)
        early_x, early_y, early_q = _flat_matrix(early_arrays)
        predict_x, _, _ = _flat_matrix(predict_arrays)
        model = LightGBMLambdaRank(
            seed=args.seed,
            num_leaves=args.num_leaves,
            learning_rate=args.tree_learning_rate,
            n_estimators=args.n_estimators,
        ).fit(fit_x, fit_y, fit_q, eval_set=(early_x, early_y, early_q))
        ranker_started = time.perf_counter()
        score = model.predict(predict_x)
        ranker_latency_ms = (
            (time.perf_counter() - ranker_started) * 1000.0 / len(predict_x)
        )
        prediction_rows = []
        cursor = 0
        for sample_id, ids in zip(
            predict_arrays.sample_ids, predict_arrays.candidate_ids
        ):
            for candidate_id in ids:
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "score": float(score[cursor]),
                    }
                )
                cursor += 1
        predictions = pd.DataFrame(prediction_rows)
        _atomic_pickle(root / "model.pkl", model)
        training_metadata: dict[str, object] = model.artifact()
        model_path = root / "model.pkl"
        parameter_count = lightgbm_parameter_count(model)
    else:
        set_deterministic_cpu(args.seed)
        model = _build_model(
            args.encoder,
            len(columns),
            args.alpha,
            num_attention_blocks=args.num_attention_blocks,
            edge_dim=len(relation_columns) if relation_columns else 1,
        )
        training_config = NeuralTrainingConfig(
            loss=args.loss,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            alpha=args.alpha,
            temperature=args.temperature,
            beta=args.beta,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        training = fit_neural_ranker(
            model, fit_arrays, early_arrays, config=training_config
        )
        model.load_state_dict(training.state_dict)
        ranker_started = time.perf_counter()
        predictions = predict_neural_ranker(model, predict_arrays)
        ranker_latency_ms = (
            (time.perf_counter() - ranker_started) * 1000.0 / len(predictions)
        )
        model_path = root / "model.pt"
        _atomic_torch(
            model_path,
            {
                "state_dict": training.state_dict,
                "encoder": args.encoder,
                "input_dim": len(columns),
                "alpha": args.alpha,
                "num_attention_blocks": args.num_attention_blocks,
                "edge_dim": len(relation_columns) if relation_columns else 1,
                "edge_columns": list(relation_columns),
            },
        )
        training_metadata = {
            "best_epoch": training.best_epoch,
            "best_validation_loss": training.best_validation_loss,
            "epochs_ran": training.epochs_ran,
            "history": list(training.history),
        }
        parameter_count = torch_parameter_count(model)

    telemetry = telemetry_payload(
        phase=f"matrix_cell_{args.mode}",
        parameter_count=parameter_count,
        ranker_latency_ms=ranker_latency_ms,
        feature_latency_ms=feature_latency_ms,
        missing_feature_rate_value=selected_missing_rate,
    )

    evaluation_input = predict_rows[
        ["sample_id", "candidate_id", "native_rank", "candidate_success"]
    ].merge(predictions, on=["sample_id", "candidate_id"], validate="one_to_one")
    metrics, decisions = evaluate_order_only(
        denominator_ids, evaluation_input, score_column="score"
    )
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
        "training": training_metadata,
        "relations": relation_metadata,
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "feature_extraction_latency_ms": feature_extraction_latency_ms,
        "metrics": metrics,
        "fit_candidate_rows": len(fit_rows),
        "early_stop_candidate_rows": len(early_rows),
        "prediction_candidate_rows": len(predictions),
        "sources": {
            "train_features": {
                "path": str(train_feature_path),
                "sha256": sha256_file(train_feature_path),
            },
            "train_feature_manifest": {
                "path": str(train_feature_manifest_path.resolve()),
                "sha256": sha256_file(train_feature_manifest_path),
            },
            "train_feature_extraction_benchmark": (
                train_feature_extraction_latency_source
            ),
            "train_labels": {
                "path": str(train_labels_path),
                "sha256": sha256_file(train_labels_path),
            },
            "folds": {"path": str(folds_path), "sha256": sha256_file(folds_path)},
            "validation_features": None
            if validation_feature_path is None
            else {
                "path": str(validation_feature_path),
                "sha256": sha256_file(validation_feature_path),
            },
            "validation_labels": None
            if validation_labels_path is None
            else {
                "path": str(validation_labels_path),
                "sha256": sha256_file(validation_labels_path),
            },
            "validation_feature_manifest": None
            if validation_feature_manifest_path is None
            else {
                "path": str(validation_feature_manifest_path.resolve()),
                "sha256": sha256_file(validation_feature_manifest_path),
            },
            "validation_feature_extraction_benchmark": (
                validation_feature_extraction_latency_source
            ),
            "validation_denominator": None
            if denominator_path is None
            else {
                "path": str(denominator_path),
                "sha256": sha256_file(denominator_path),
            },
            "train_relations": None
            if relation_metadata is None
            else {
                "path": str(relation_metadata["train_path"]),
                "sha256": str(relation_metadata["train_sha256"]),
            },
            "prediction_relations": None
            if relation_metadata is None
            else {
                "path": str(relation_metadata["prediction_path"]),
                "sha256": str(relation_metadata["prediction_sha256"]),
            },
            "training_code": code_records,
        },
        "artifacts": {
            "model": {
                "path": str(model_path.resolve()),
                "sha256": sha256_file(model_path),
            },
            "predictions": {
                "path": str(prediction_path.resolve()),
                "sha256": sha256_file(prediction_path),
            },
            "decisions": {
                "path": str(decision_path.resolve()),
                "sha256": sha256_file(decision_path),
            },
        },
    }
    atomic_json(marker, result)
    return result


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    substage = f"matrix_{args.mode}_{args.route}_{args.track}_{args.encoder}_{args.loss}_{args.seed}_{args.fold}"
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P7",
        substage=substage,
        route=args.route,
        evidence_track=args.track,
        pool="top5",
        method=args.encoder,
        feature_set="all",
        loss=args.loss,
        encoder=args.encoder,
        seed=args.seed,
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args)
        artifact = Path(
            str(result.get("artifacts", {}).get("predictions", {}).get("path", ""))
        )
        if artifact.is_file():
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
