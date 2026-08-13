"""Train one strict fold-local D1 Top5/T2 R2-R6 cell on CPU."""

from __future__ import annotations

# ruff: noqa: E402 -- native thread limits must be set before numeric imports

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

# The frozen macOS runtime can crash inside LightGBM when its OpenMP runtime is
# initialized after PyTorch's.  R5 imports LightGBM lazily, so initialize the
# optional native runtime first for every worker while retaining the explicit
# missing-dependency error at the point where R5 is selected.
try:  # pragma: no cover - availability is environment-specific
    import lightgbm as _lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _lightgbm = None

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.candidates import artifact_record  # noqa: E402
from d1_reranking.fold_calibration import (  # noqa: E402
    apply_partition_calibrator,
    fit_partition_calibrator,
)
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.models import D1ResidualMLPScorer  # noqa: E402
from d1_reranking.plan import load_active_primary_plan  # noqa: E402
from d1_reranking.primary_execution import validate_worker_context  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.datasets import (  # noqa: E402
    FoldPreprocessor,
    build_query_arrays,
    join_development_features_and_labels,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.metrics import evaluate_order_only  # noqa: E402
from unified_reranking.models import LightGBMLambdaRank  # noqa: E402
from unified_reranking.telemetry import (  # noqa: E402
    flatten_telemetry,
    lightgbm_parameter_count,
    missing_feature_rate,
    telemetry_payload,
    torch_parameter_count,
)
from unified_reranking.training import (  # noqa: E402
    FORMAL_SEEDS,
    NeuralTrainingConfig,
    fit_neural_ranker,
    predict_neural_ranker,
    set_deterministic_cpu,
)


METHOD_LOSSES = {
    "R2": "bce",
    "R3": "ranknet",
    "R4": "listwise",
    "R6": "jacquard_margin_ranknet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _verified_content(
    path: Path, *, name: str, statuses: tuple[str, ...] = ("COMPLETE",)
) -> dict[str, Any]:
    value = load_verified_json(path, name=name, statuses=statuses)
    unsigned = dict(value)
    expected = unsigned.pop("content_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash mismatch")
    return value


def _same_artifact_record(
    observed: object, expected: dict[str, object], *, name: str
) -> None:
    """Compare normalized artifact identity while allowing extra metadata."""

    if not isinstance(observed, dict) or any(
        observed.get(key) != expected.get(key) for key in ("path", "sha256")
    ):
        raise RuntimeError(f"{name} does not bind the current artifact")
    if (
        "bytes" in observed
        and "bytes" in expected
        and observed.get("bytes") != expected.get("bytes")
    ):
        raise RuntimeError(f"{name} byte count differs")


def _load_split(
    root: Path, split: str
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, object]]:
    raw_manifest_path = (
        root / "03_features" / split / "top5" / "matched_common_raw" / "manifest.json"
    )
    final_manifest_path = (
        root / "03_features" / split / "top5" / "T2_matched_common" / "manifest.json"
    )
    label_manifest_path = (
        root / "03_features" / split / "top5" / "labels" / "manifest.json"
    )
    raw_manifest = _verified_content(raw_manifest_path, name=f"D1 {split} raw T2")
    final_manifest = _verified_content(final_manifest_path, name=f"D1 {split} final T2")
    extraction_latency = float(final_manifest.get("feature_extraction_latency_ms", -1))
    if extraction_latency < 0:
        raise RuntimeError(f"D1 {split} final T2 extraction latency is absent")
    label_manifest = _verified_content(label_manifest_path, name=f"D1 {split} labels")
    candidate_manifest_path = root / "02_candidates" / split / "manifest.json"
    candidate_manifest = _verified_content(
        candidate_manifest_path, name=f"D1 {split} candidates"
    )
    candidate_configuration = candidate_manifest.get("configuration")
    if not isinstance(candidate_configuration, dict) or (
        candidate_configuration.get("route") != "D1"
        or candidate_configuration.get("split") != split
        or candidate_manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {split} candidate manifest semantics differ")
    verify_artifact_records_recursive(
        candidate_manifest.get("artifacts"),
        name=f"D1 {split} candidate artifacts",
        require_at_least_one=True,
    )
    candidate_record = candidate_manifest.get("artifacts", {}).get("top5", {})
    candidate_path = verified_artifact_path(
        candidate_record, name=f"D1 {split} Top5 candidates"
    )
    candidate_hashes_path = verified_artifact_path(
        candidate_manifest.get("artifacts", {}).get("candidate_hashes", {}),
        name=f"D1 {split} candidate hashes",
    )
    candidate_manifest_record = artifact_record(candidate_manifest_path)
    candidates_record = artifact_record(candidate_path)
    candidate_hashes_record = artifact_record(candidate_hashes_path)
    raw_sources = raw_manifest.get("sources")
    raw_extractor_sources = (
        raw_sources.get("extractor") if isinstance(raw_sources, dict) else None
    )
    final_sources = final_manifest.get("sources")
    label_sources = label_manifest.get("sources")
    final_configuration = final_manifest.get("configuration")
    if (
        raw_manifest.get("route") != "D1"
        or raw_manifest.get("split") != split
        or raw_manifest.get("pool") != "top5"
        or raw_manifest.get("track") != "matched_common_raw"
        or not isinstance(raw_sources, dict)
        or not isinstance(raw_extractor_sources, dict)
    ):
        raise RuntimeError(f"D1 {split} raw T2 candidate binding differs")
    _same_artifact_record(
        raw_extractor_sources.get("candidate_manifest"),
        candidate_manifest_record,
        name=f"D1 {split} raw T2 candidate manifest",
    )
    _same_artifact_record(
        raw_extractor_sources.get("canonical_candidates"),
        candidates_record,
        name=f"D1 {split} raw T2 candidates",
    )
    _same_artifact_record(
        raw_extractor_sources.get("candidate_hashes"),
        candidate_hashes_record,
        name=f"D1 {split} raw T2 candidate hashes",
    )
    if (
        not isinstance(final_configuration, dict)
        or final_configuration.get("route") != "D1"
        or final_configuration.get("split") != split
        or final_configuration.get("pool") != "top5"
        or final_configuration.get("track") != "T2_matched_common"
        or not isinstance(final_sources, dict)
    ):
        raise RuntimeError(f"D1 {split} final T2 source binding differs")
    _same_artifact_record(
        final_sources.get("candidate_manifest"),
        candidate_manifest_record,
        name=f"D1 {split} final T2 candidate manifest",
    )
    _same_artifact_record(
        final_sources.get("candidates"),
        candidates_record,
        name=f"D1 {split} final T2 candidates",
    )
    _same_artifact_record(
        final_sources.get("candidate_hashes"),
        candidate_hashes_record,
        name=f"D1 {split} final T2 candidate hashes",
    )
    _same_artifact_record(
        final_sources.get("raw_common_manifest"),
        artifact_record(raw_manifest_path),
        name=f"D1 {split} final T2 raw common manifest",
    )
    if (
        label_manifest.get("split") != split
        or label_manifest.get("pool") != "top5"
        or not isinstance(label_sources, dict)
    ):
        raise RuntimeError(f"D1 {split} label candidate binding differs")
    _same_artifact_record(
        label_sources.get("candidate_manifest"),
        candidate_manifest_record,
        name=f"D1 {split} label candidate manifest",
    )
    _same_artifact_record(
        label_sources.get("candidates"),
        candidates_record,
        name=f"D1 {split} label candidates",
    )
    _same_artifact_record(
        label_sources.get("candidate_hashes"),
        candidate_hashes_record,
        name=f"D1 {split} label candidate hashes",
    )
    verify_artifact_records_recursive(
        {"sources": raw_sources, "artifacts": raw_manifest.get("artifacts")},
        name=f"D1 {split} raw T2",
        require_at_least_one=True,
    )
    verify_artifact_records_recursive(
        {"sources": final_sources, "artifacts": final_manifest.get("artifacts")},
        name=f"D1 {split} final T2",
        require_at_least_one=True,
    )
    verify_artifact_records_recursive(
        {"sources": label_sources, "artifact": label_manifest.get("artifact")},
        name=f"D1 {split} labels",
        require_at_least_one=True,
    )
    if any(
        manifest.get("candidate_test_labels_read") is not False
        for manifest in (raw_manifest, final_manifest, label_manifest)
    ):
        raise RuntimeError("D1 development input violates Test isolation")
    raw_path = verified_artifact_path(
        raw_manifest.get("artifacts", {}).get("candidate_features", {}),
        name=f"D1 {split} raw common features",
    )
    label_path = verified_artifact_path(
        label_manifest.get("artifact", {}), name=f"D1 {split} labels"
    )
    columns = tuple(map(str, final_manifest.get("model_feature_columns", ())))
    if not columns or not {"base_logit", "calibrated_native_probability"}.issubset(
        columns
    ):
        raise RuntimeError("D1 final T2 schema lacks calibration fields")
    joined = join_development_features_and_labels(
        pd.read_parquet(raw_path), pd.read_parquet(label_path)
    )
    candidates = pd.read_parquet(
        candidate_path,
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    ).rename(columns={"native_rank": "frozen_native_rank"})
    joined = joined.merge(
        candidates, on=["sample_id", "candidate_id"], how="inner", validate="one_to_one"
    )
    if len(joined) != len(candidates):
        raise RuntimeError(f"D1 {split} primary cell candidate membership differs")
    if (
        not joined["native_rank"]
        .astype(int)
        .equals(joined["frozen_native_rank"].astype(int))
    ):
        raise RuntimeError(f"D1 {split} primary cell native rank differs")
    joined = joined.drop(columns="frozen_native_rank")
    return (
        joined,
        columns,
        {
            "raw_feature_manifest": artifact_record(raw_manifest_path),
            "raw_features": artifact_record(raw_path),
            "final_feature_manifest": artifact_record(final_manifest_path),
            "feature_extraction_latency_ms": extraction_latency,
            "label_manifest": artifact_record(label_manifest_path),
            "labels": artifact_record(label_path),
            "candidate_manifest": artifact_record(candidate_manifest_path),
            "candidate_hashes": candidate_hashes_record,
            "candidates": artifact_record(candidate_path),
            "final_calibration_manifest": final_sources.get("calibration_manifest"),
        },
    )


def _atomic_torch(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    return path


def _atomic_pickle(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _flat_arrays(arrays: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    valid = ~arrays.padding_mask.numpy()
    features = arrays.features.numpy()[valid]
    labels = arrays.labels.numpy()[valid].astype(np.int32)
    query_ids = [
        sample_id
        for sample_id, candidate_ids in zip(arrays.sample_ids, arrays.candidate_ids)
        for _ in candidate_ids
    ]
    return features, labels, query_ids


def _cell_code_records() -> list[dict[str, object]]:
    return [
        artifact_record(path)
        for path in (
            Path(__file__),
            ROOT / "src/d1_reranking/plan.py",
            ROOT / "src/d1_reranking/primary_execution.py",
            ROOT / "src/d1_reranking/models.py",
            ROOT / "src/d1_reranking/fold_calibration.py",
            ROOT / "src/unified_reranking/datasets.py",
            ROOT / "src/unified_reranking/training.py",
            ROOT / "src/unified_reranking/losses.py",
            ROOT / "src/unified_reranking/models/lightgbm_ranker.py",
        )
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    plan_path, plan = load_active_primary_plan(root)
    if plan.get("status") != "PLANNED" or plan.get("execution_authorized") is not False:
        raise RuntimeError("D1 primary plan contract is invalid")
    matching_jobs = [
        job
        for job in plan.get("jobs", [])
        if isinstance(job, dict) and job.get("job_id") == args.job_id
    ]
    if len(matching_jobs) != 1:
        raise RuntimeError("D1 requested cell is outside the predeclared job universe")
    planned_job = matching_jobs[0]
    planned_configuration = planned_job.get("configuration")
    if not isinstance(planned_configuration, dict):
        raise RuntimeError("D1 planned job configuration is invalid")
    if canonical_sha256(planned_configuration)[:16] != args.job_id:
        raise RuntimeError("D1 planned job identifier differs from its configuration")
    (
        execution_path,
        execution,
        _event_path,
        _event,
        claim_path,
        _claim,
    ) = validate_worker_context(
        root,
        plan_path=plan_path,
        plan=plan,
        job=planned_job,
    )
    method = str(planned_configuration.get("method"))
    mode = str(planned_configuration.get("mode"))
    held_fold_value = planned_configuration.get("held_fold")
    held_fold = None if held_fold_value is None else int(held_fold_value)
    seed = int(planned_configuration.get("seed", -1))
    if (
        method not in {"R2", "R3", "R4", "R5", "R6"}
        or mode not in {"oof", "validation"}
        or seed not in FORMAL_SEEDS
        or (mode == "oof") != (held_fold is not None)
    ):
        raise RuntimeError("D1 planned job semantics are invalid")
    feature_started = time.perf_counter()
    train, columns, train_sources = _load_split(root, "train")
    folds_path = root / "04_splits" / "fold_assignments.parquet"
    folds = pd.read_parquet(folds_path, columns=["sample_id", "fold"])
    train = train.merge(folds, on="sample_id", how="left", validate="many_to_one")
    if train["fold"].isna().any():
        raise RuntimeError("D1 folds do not cover all Train candidates")
    early_fold = (held_fold + 1) % 5 if held_fold is not None else 0
    fit_precal = train.loc[
        (train["fold"] != early_fold)
        & ((train["fold"] != held_fold) if held_fold is not None else True)
    ].copy()
    early_precal = train.loc[train["fold"] == early_fold].copy()
    validation_sources: dict[str, object] | None = None
    denominator_path: Path | None = None
    if mode == "oof":
        predict_precal = train.loc[train["fold"] == held_fold].copy()
        denominator_ids = (
            folds.loc[folds["fold"] == held_fold, "sample_id"].astype(str).tolist()
        )
    else:
        predict_precal, validation_columns, validation_sources = _load_split(
            root, "validation"
        )
        if validation_columns != columns:
            raise RuntimeError("D1 Train/Validation final T2 schemas differ")
        denominator_path = root / "01_manifests" / "d1_paired_validation.parquet"
        denominator_ids = (
            pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
            .astype(str)
            .tolist()
        )
    calibration_manifest_path = (
        root / "05_calibration" / "top5" / "calibration_manifest.json"
    )
    calibration_manifest = _verified_content(
        calibration_manifest_path, name="D1 Top5 calibration"
    )
    calibration_record = artifact_record(calibration_manifest_path)
    if train_sources.get("final_calibration_manifest") != calibration_record or (
        validation_sources is not None
        and validation_sources.get("final_calibration_manifest") != calibration_record
    ):
        raise RuntimeError("D1 primary cell T2/calibration binding differs")
    selected_calibration = str(calibration_manifest.get("selected_method", ""))
    fit_fold_ids = tuple(sorted(fit_precal["fold"].astype(int).unique()))
    fold_calibrator = fit_partition_calibrator(
        fit_precal, method=selected_calibration, fit_fold_ids=fit_fold_ids
    )
    fit_rows = apply_partition_calibrator(fit_precal, fold_calibrator)
    early_rows = apply_partition_calibrator(early_precal, fold_calibrator)
    predict_rows = apply_partition_calibrator(predict_precal, fold_calibrator)
    unknown = sorted(set(columns).difference(fit_rows.columns))
    if unknown:
        raise RuntimeError(f"D1 fold-calibrated features miss final schema: {unknown}")
    preprocessor = FoldPreprocessor.fit(fit_rows, columns)
    fit_arrays = build_query_arrays(
        fit_rows, preprocessor=preprocessor, max_candidates=5
    )
    early_arrays = build_query_arrays(
        early_rows, preprocessor=preprocessor, max_candidates=5
    )
    predict_arrays = build_query_arrays(
        predict_rows, preprocessor=preprocessor, max_candidates=5
    )
    sources: dict[str, object] = {
        "plan": artifact_record(plan_path),
        "train": train_sources,
        "validation": validation_sources,
        "folds": artifact_record(folds_path),
        "calibration_manifest": artifact_record(calibration_manifest_path),
        "validation_denominator": (
            None if denominator_path is None else artifact_record(denominator_path)
        ),
        "training_code": _cell_code_records(),
    }
    configuration = {
        "schema_version": 1,
        **planned_configuration,
        "early_stop_fold": early_fold,
        "fit_fold_ids": list(fit_fold_ids),
        "feature_columns": list(columns),
        "feature_schema_sha256": canonical_sha256(columns),
        "fold_local_calibrator": fold_calibrator,
        "planned_job_id": planned_job["job_id"],
        "planned_configuration": planned_configuration,
        "sources": sources,
    }
    cell_key = canonical_sha256(configuration)[:16]
    output = (
        root
        / ("06_oof" if mode == "oof" else "07_validation")
        / "primary_cells"
        / cell_key
    )
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = _verified_content(manifest_path, name=f"D1 primary cell {cell_key}")
        if (
            args.resume
            and existing.get("status") == "COMPLETE"
            and existing.get("configuration") == configuration
        ):
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name=f"D1 primary cell {cell_key}",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("D1 primary cell exists with a different/corrupt contract")

    processed = pd.concat([fit_rows, early_rows, predict_rows], ignore_index=True)
    feature_latency_ms = (
        (time.perf_counter() - feature_started) * 1000.0 / len(processed)
    )
    selected_missing = missing_feature_rate(processed, columns)
    if method == "R5":
        fit_x, fit_y, fit_q = _flat_arrays(fit_arrays)
        early_x, early_y, early_q = _flat_arrays(early_arrays)
        predict_x, _, _ = _flat_arrays(predict_arrays)
        model = LightGBMLambdaRank(
            seed=seed,
            num_leaves=int(planned_configuration["num_leaves"]),
            learning_rate=float(planned_configuration["learning_rate"]),
            n_estimators=int(planned_configuration["n_estimators"]),
        ).fit(fit_x, fit_y, fit_q, eval_set=(early_x, early_y, early_q))
        started = time.perf_counter()
        values = model.predict(predict_x)
        ranker_latency_ms = (time.perf_counter() - started) * 1000.0 / len(values)
        rows = []
        cursor = 0
        for sample_id, candidate_ids in zip(
            predict_arrays.sample_ids, predict_arrays.candidate_ids
        ):
            for candidate_id in candidate_ids:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "score": float(values[cursor]),
                    }
                )
                cursor += 1
        predictions = pd.DataFrame(rows)
        model_path = _atomic_pickle(output / "model.pkl", model)
        training_metadata: dict[str, object] = model.artifact()
        parameter_count = lightgbm_parameter_count(model)
    else:
        set_deterministic_cpu(seed)
        hidden_dims = tuple(map(int, planned_configuration["hidden_dims"]))
        model = D1ResidualMLPScorer(
            len(columns),
            hidden_dims=hidden_dims,
            dropout=float(planned_configuration["dropout"]),
            alpha=float(planned_configuration["alpha"]),
        )
        training_config = NeuralTrainingConfig(
            loss=METHOD_LOSSES[method],
            learning_rate=float(planned_configuration["learning_rate"]),
            weight_decay=float(planned_configuration["weight_decay"]),
            alpha=float(planned_configuration["alpha"]),
            temperature=float(planned_configuration["temperature"]),
            beta=float(planned_configuration["beta"]),
            epochs=int(planned_configuration["epochs"]),
            patience=int(planned_configuration["patience"]),
            batch_size=int(planned_configuration["batch_size"]),
            seed=seed,
        )
        training = fit_neural_ranker(
            model, fit_arrays, early_arrays, config=training_config
        )
        model.load_state_dict(training.state_dict)
        started = time.perf_counter()
        predictions = predict_neural_ranker(model, predict_arrays)
        ranker_latency_ms = (time.perf_counter() - started) * 1000.0 / len(predictions)
        model_path = _atomic_torch(
            output / "model.pt",
            {
                "state_dict": training.state_dict,
                "input_dim": len(columns),
                "hidden_dims": list(hidden_dims),
                "dropout": float(planned_configuration["dropout"]),
                "alpha": float(planned_configuration["alpha"]),
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
        phase=f"d1_primary_cell_{mode}",
        parameter_count=parameter_count,
        ranker_latency_ms=ranker_latency_ms,
        feature_latency_ms=feature_latency_ms,
        missing_feature_rate_value=selected_missing,
    )
    prediction_contract = predict_rows[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ]
    ]
    if predictions.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError("D1 primary cell predictions contain duplicate candidates")
    predictions = prediction_contract.merge(
        predictions[["sample_id", "candidate_id", "score"]],
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if (
        len(predictions) != len(prediction_contract)
        or not np.isfinite(predictions["score"].to_numpy(float)).all()
    ):
        raise RuntimeError(
            "D1 primary cell predictions do not exactly cover candidates"
        )
    evaluation = predict_rows[
        ["sample_id", "candidate_id", "native_rank", "candidate_success"]
    ].merge(
        predictions[["sample_id", "candidate_id", "score"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    metrics, decisions = evaluate_order_only(
        denominator_ids, evaluation, score_column="score", max_k=5
    )
    prediction_path = atomic_parquet(predictions, output / "predictions.parquet")
    decisions_path = atomic_parquet(decisions, output / "per_sample_decisions.parquet")
    preprocessor_path = output / "preprocessor.json"
    atomic_json(preprocessor_path, preprocessor.artifact())
    artifacts = {
        "model": artifact_record(model_path),
        "predictions": artifact_record(prediction_path),
        "decisions": artifact_record(decisions_path),
        "preprocessor": artifact_record(preprocessor_path),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "cell_key": cell_key,
        "configuration": configuration,
        "metrics": metrics,
        "training": training_metadata,
        "preprocessor": preprocessor.artifact(),
        "feature_extraction_latency_ms": (
            train_sources["feature_extraction_latency_ms"]
            if validation_sources is None
            else validation_sources["feature_extraction_latency_ms"]
        ),
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "execution_provenance": {
            "execution_id": execution["execution_id"],
            "execution_authority": artifact_record(execution_path),
            "resource_gate": execution["resource_gate"],
            "claim": artifact_record(claim_path),
            "candidate_test_labels_read": False,
        },
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P9",
        substage=f"d1_primary_cell_{args.job_id}",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="planned_job",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args)
        manifest_path = (
            root
            / (
                "06_oof"
                if result["configuration"]["mode"] == "oof"
                else "07_validation"
            )
            / "primary_cells"
            / str(result["cell_key"])
            / "manifest.json"
        )
        state["artifact_path"] = str(manifest_path)
        state["artifact_sha256"] = sha256_file(manifest_path)
        print(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "manifest": str(manifest_path),
                    "sha256": state["artifact_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
