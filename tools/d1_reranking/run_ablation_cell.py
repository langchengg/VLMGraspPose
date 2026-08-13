"""Run one claimed P10 ablation cell under the global heavy-resource lease."""

from __future__ import annotations

# ruff: noqa: E402 -- native thread limits must be set before numeric imports

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import sys
import time
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

try:
    import lightgbm as _lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _lightgbm = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.ablation import (  # noqa: E402
    ABLATION_PLAN_RELATIVE,
    load_ablation_plan,
    track_feature_artifact,
)
from d1_reranking.ablation_execution import validate_worker_context  # noqa: E402
from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.fold_calibration import (  # noqa: E402
    apply_partition_calibrator,
    fit_partition_calibrator,
)
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.models import D1ResidualMLPScorer  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
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
    NeuralTrainingConfig,
    fit_neural_ranker,
    predict_neural_ranker,
    set_deterministic_cpu,
)
from tools.d1_reranking.run_k_sensitivity_cell import (  # noqa: E402
    _atomic_native_lightgbm,
    _atomic_torch,
    _decision_contract,
    _flat_arrays,
    _prediction_contract,
    _prediction_rows,
)


METHOD_LOSSES = {"R3": "ranknet", "R6": "jacquard_margin_ranknet"}
PREDICTION_COLUMNS = (
    "sample_id",
    "candidate_id",
    "native_rank",
    "candidate_identity_sha256",
    "candidate_geometry_sha256",
    "score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _planned_job(plan: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    matches = [
        _mapping(job, name="D1 P10 job")
        for job in plan.get("jobs", ())
        if isinstance(job, Mapping) and job.get("job_id") == job_id
    ]
    if len(matches) != 1:
        raise RuntimeError("D1 P10 cell is outside the frozen plan")
    job = matches[0]
    configuration = _mapping(job["configuration"], name="D1 P10 configuration")
    held_fold = configuration.get("held_fold")
    if (
        canonical_sha256(configuration)[:16] != job_id
        or configuration.get("analysis") not in {"evidence_track", "feature_family"}
        or configuration.get("method") not in {"R3", "R5", "R6"}
        or configuration.get("seed") not in {42, 123, 2026}
        or configuration.get("mode") not in {"oof", "validation"}
        or (configuration.get("mode") == "oof") != (held_fold is not None)
        or (held_fold is not None and not 0 <= int(held_fold) < 5)
        or not configuration.get("feature_columns")
        or configuration.get("candidate_test_labels_read") is not False
        or configuration.get("test_inputs_referenced") is not False
    ):
        raise RuntimeError("D1 P10 planned cell semantics differ")
    return job


def _bound_manifest(
    record: Mapping[str, Any], *, name: str
) -> tuple[Path, dict[str, Any]]:
    path = verified_artifact_path(record, name=name)
    manifest = load_content_manifest(path, name=name, statuses=("COMPLETE",))
    verify_artifact_records_recursive(
        {
            "sources": manifest.get("sources"),
            "artifact": manifest.get("artifact"),
            "artifacts": manifest.get("artifacts"),
        },
        name=f"{name} closure",
        require_at_least_one=True,
    )
    return path, manifest


def _load_split(
    plan: Mapping[str, Any],
    *,
    split: str,
    track: str,
    selected_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise PermissionError("D1 P10 cell cannot represent Test inputs")
    sources = _mapping(plan["sources"], name="D1 P10 plan sources")
    feature_records = _mapping(
        _mapping(sources["track_manifests"], name="D1 P10 tracks")[track],
        name=f"D1 P10 {track} manifests",
    )
    candidate_records = _mapping(
        sources["candidate_manifests"], name="candidate manifests"
    )
    label_records = _mapping(
        sources["development_label_manifests"], name="development labels"
    )
    feature_manifest_path, feature_manifest = _bound_manifest(
        _mapping(feature_records[split], name="feature record"),
        name=f"D1 P10 {split}/{track} features",
    )
    candidate_manifest_path, candidate_manifest = _bound_manifest(
        _mapping(candidate_records[split], name="candidate record"),
        name=f"D1 P10 {split} candidates",
    )
    label_manifest_path, label_manifest = _bound_manifest(
        _mapping(label_records[split], name="label record"),
        name=f"D1 P10 {split} labels",
    )
    if any(
        manifest.get("candidate_test_labels_read") is not False
        for manifest in (feature_manifest, candidate_manifest, label_manifest)
    ):
        raise PermissionError("D1 P10 development source provenance differs")
    feature_columns = tuple(
        map(
            str,
            plan["evidence_tracks"][track]["feature_columns"],
        )
    )
    if not set(selected_columns).issubset(feature_columns):
        raise RuntimeError("D1 P10 selected feature family escapes its track schema")
    feature_path = track_feature_artifact(
        feature_manifest,
        track=track,
        name=f"D1 P10 {split}/{track} features",
    )
    candidate_path = verified_artifact_path(
        candidate_manifest["artifacts"]["top5"],
        name=f"D1 P10 {split} Top5 candidates",
    )
    label_path = verified_artifact_path(
        label_manifest["artifact"], name=f"D1 P10 {split} development labels"
    )
    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)
    joined = join_development_features_and_labels(features, labels)
    identity_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "native_score",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
    ]
    candidates = pd.read_parquet(candidate_path, columns=identity_columns)
    frozen = candidates.rename(
        columns={column: f"_frozen_{column}" for column in identity_columns[2:]}
    )
    joined = joined.merge(
        frozen, on=["sample_id", "candidate_id"], how="inner", validate="one_to_one"
    )
    if len(joined) != len(candidates):
        raise RuntimeError("D1 P10 candidate membership differs")
    for column in identity_columns[2:]:
        frozen_column = f"_frozen_{column}"
        if column in joined:
            if column in {"native_rank", "native_score"}:
                equal = np.array_equal(
                    pd.to_numeric(joined[column]).to_numpy(float),
                    pd.to_numeric(joined[frozen_column]).to_numpy(float),
                )
            else:
                equal = (
                    joined[column].astype(str).equals(joined[frozen_column].astype(str))
                )
            if not equal:
                raise RuntimeError(f"D1 P10 frozen {column} differs")
            joined = joined.drop(columns=column)
        joined = joined.rename(columns={frozen_column: column})
    if "native_score_raw" not in joined:
        joined["native_score_raw"] = joined["native_score"].to_numpy(float)
    if int(joined.groupby("sample_id", sort=False).size().max()) > 5:
        raise RuntimeError("D1 P10 source exceeds Top5")
    return joined, {
        "feature_manifest": artifact_record(feature_manifest_path),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "label_manifest": artifact_record(label_manifest_path),
        "features": artifact_record(feature_path),
        "candidates": artifact_record(candidate_path),
        "labels": artifact_record(label_path),
    }


def run(
    args: argparse.Namespace, *, resource_lease_path: Path | None = None
) -> dict[str, Any]:
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    expected_lease = (root.parent / ".d1_heavy_resource.lock").resolve()
    if resource_lease_path is None or resource_lease_path.resolve() != expected_lease:
        raise RuntimeError("D1 P10 cell requires the repository-wide lease")
    plan_path = root / ABLATION_PLAN_RELATIVE
    plan = load_ablation_plan(plan_path)
    job = _planned_job(plan, str(args.job_id))
    configuration = _mapping(job["configuration"], name="D1 P10 configuration")
    (
        execution_path,
        execution,
        event_path,
        _event,
        claim_path,
        _claim,
    ) = validate_worker_context(root, plan_path=plan_path, plan=plan, job=job)
    manifest_path = (root / str(job["output_manifest"])).resolve()
    output = manifest_path.parent
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name=f"D1 P10 cell {args.job_id}", statuses=("COMPLETE",)
        )
        if args.resume and existing.get("configuration") == configuration:
            verify_artifact_records_recursive(
                existing, name="D1 P10 resume cell", require_at_least_one=True
            )
            return existing
        raise RuntimeError("D1 P10 cell exists and differs")
    columns = tuple(map(str, configuration["feature_columns"]))
    feature_started = time.perf_counter()
    train, train_inputs = _load_split(
        plan, split="train", track=str(configuration["track"]), selected_columns=columns
    )
    folds_path = verified_artifact_path(
        plan["sources"]["fold_assignments"], name="D1 P10 folds"
    )
    folds = pd.read_parquet(folds_path, columns=["sample_id", "fold"])
    train = train.merge(folds, on="sample_id", how="left", validate="many_to_one")
    if train["fold"].isna().any() or set(train["fold"].astype(int)) != set(range(5)):
        raise RuntimeError("D1 P10 fold assignment differs")
    held_fold = configuration.get("held_fold")
    early_fold = (int(held_fold) + 1) % 5 if held_fold is not None else 0
    fit_precal = train.loc[
        (train["fold"] != early_fold)
        & ((train["fold"] != held_fold) if held_fold is not None else True)
    ].copy()
    early_precal = train.loc[train["fold"] == early_fold].copy()
    validation_inputs = None
    if configuration["mode"] == "oof":
        predict_precal = train.loc[train["fold"] == held_fold].copy()
        denominator = (
            folds.loc[folds["fold"] == held_fold, "sample_id"].astype(str).tolist()
        )
    else:
        predict_precal, validation_inputs = _load_split(
            plan,
            split="validation",
            track=str(configuration["track"]),
            selected_columns=columns,
        )
        denominator_path = verified_artifact_path(
            plan["sources"]["denominators"]["validation"],
            name="D1 P10 Validation denominator",
        )
        denominator = (
            pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
            .astype(str)
            .tolist()
        )
    calibration_path, calibration = _bound_manifest(
        plan["sources"]["calibration_manifest"], name="D1 P10 calibration"
    )
    method = str(calibration["selected_method"])
    fit_fold_ids = tuple(map(int, sorted(fit_precal["fold"].astype(int).unique())))
    fold_calibrator = fit_partition_calibrator(
        fit_precal, method=method, fit_fold_ids=fit_fold_ids
    )
    fit_rows = apply_partition_calibrator(fit_precal, fold_calibrator)
    early_rows = apply_partition_calibrator(early_precal, fold_calibrator)
    predict_rows = apply_partition_calibrator(predict_precal, fold_calibrator)
    if sorted(set(columns).difference(fit_rows.columns)):
        raise RuntimeError("D1 P10 selected feature schema is incomplete")
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
    processed = pd.concat([fit_rows, early_rows, predict_rows], ignore_index=True)
    feature_latency = (time.perf_counter() - feature_started) * 1000.0 / len(processed)
    missing_rate = missing_feature_rate(processed, columns)
    training_spec = _mapping(
        configuration["training_spec"], name="D1 P10 training spec"
    )
    selected_method = str(configuration["method"])
    seed = int(configuration["seed"])
    training_metadata: dict[str, Any]
    if selected_method == "R5":
        fit_x, fit_y, fit_q = _flat_arrays(fit_arrays)
        early_x, early_y, early_q = _flat_arrays(early_arrays)
        predict_x, _, _ = _flat_arrays(predict_arrays)
        model: Any = LightGBMLambdaRank(
            seed=seed,
            num_leaves=int(training_spec["num_leaves"]),
            learning_rate=float(training_spec["learning_rate"]),
            n_estimators=int(training_spec["n_estimators"]),
        ).fit(fit_x, fit_y, fit_q, eval_set=(early_x, early_y, early_q))
        started = time.perf_counter()
        score_rows = _prediction_rows(predict_arrays, model.predict(predict_x))
        ranker_latency = (time.perf_counter() - started) * 1000.0 / len(score_rows)
        training_metadata = model.artifact()
        parameter_count = lightgbm_parameter_count(model)
        model_path = _atomic_native_lightgbm(output / "model.txt", model)
        model_kind = "lightgbm_lambdarank_native"
    else:
        set_deterministic_cpu(seed)
        model = D1ResidualMLPScorer(
            len(columns),
            hidden_dims=tuple(map(int, training_spec["hidden_dims"])),
            dropout=float(training_spec["dropout"]),
            alpha=float(training_spec["alpha"]),
        )
        neural = NeuralTrainingConfig(
            loss=METHOD_LOSSES[selected_method],
            learning_rate=float(training_spec["learning_rate"]),
            weight_decay=float(training_spec["weight_decay"]),
            alpha=float(training_spec["alpha"]),
            temperature=float(training_spec["temperature"]),
            beta=float(training_spec["beta"]),
            epochs=int(training_spec["epochs"]),
            patience=int(training_spec["patience"]),
            batch_size=int(training_spec["batch_size"]),
            seed=seed,
            gradient_clip_norm=float(training_spec["gradient_clip_norm"]),
        )
        trained = fit_neural_ranker(model, fit_arrays, early_arrays, config=neural)
        model.load_state_dict(trained.state_dict)
        started = time.perf_counter()
        score_rows = predict_neural_ranker(model, predict_arrays)
        ranker_latency = (time.perf_counter() - started) * 1000.0 / len(score_rows)
        training_metadata = {
            "best_epoch": trained.best_epoch,
            "best_validation_loss": trained.best_validation_loss,
            "epochs_ran": trained.epochs_ran,
            "history": list(trained.history),
            "training_config": trained.config,
        }
        parameter_count = torch_parameter_count(model)
        model_path = _atomic_torch(
            output / "model.pt",
            {
                "state_dict": trained.state_dict,
                "input_dim": len(columns),
                "selected_primary_trial_id": configuration["selected_primary_trial_id"],
            },
        )
        model_kind = "d1_residual_mlp"
    identity = predict_rows[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ]
    ]
    predictions = identity.merge(
        score_rows[["sample_id", "candidate_id", "score"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    ).loc[:, list(PREDICTION_COLUMNS)]
    predictions = predictions.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    evaluation = predict_rows[
        ["sample_id", "candidate_id", "native_rank", "candidate_success"]
    ].merge(
        predictions[["sample_id", "candidate_id", "score"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score", max_k=5
    )
    prediction_path = atomic_parquet(predictions, output / "predictions.parquet")
    decision_path = atomic_parquet(decisions, output / "per_sample_decisions.parquet")
    preprocessor_payload = {
        "schema_version": 1,
        "job_id": str(args.job_id),
        "feature_schema_sha256": canonical_sha256(columns),
        "fold_calibrator": fold_calibrator,
        "fold_preprocessor": preprocessor.artifact(),
    }
    preprocessor_payload["content_sha256"] = canonical_sha256(preprocessor_payload)
    preprocessor_path = atomic_json(output / "preprocessor.json", preprocessor_payload)
    model_contract = {
        "kind": model_kind,
        "method": selected_method,
        "seed": seed,
        "effective_training_hyperparameters": training_spec,
        "feature_schema_sha256": canonical_sha256(columns),
        "training": training_metadata,
        "parameter_count": parameter_count,
    }
    telemetry = telemetry_payload(
        phase=f"d1_p10_{configuration['analysis']}_{configuration['mode']}",
        parameter_count=parameter_count,
        ranker_latency_ms=ranker_latency,
        feature_latency_ms=feature_latency,
        missing_feature_rate_value=missing_rate,
    )
    sources = {
        "plan": artifact_record(plan_path),
        "execution_authority": artifact_record(execution_path),
        "execution_event": artifact_record(event_path),
        "execution_claim": artifact_record(claim_path),
        "selected_primary": plan["sources"]["selected_primary"],
        "selected_trial": plan["sources"]["selected_trial"],
        "track_manifests": plan["sources"]["track_manifests"][configuration["track"]],
        "candidate_manifests": plan["sources"]["candidate_manifests"],
        "development_label_manifests": plan["sources"]["development_label_manifests"],
        "calibration_manifest": artifact_record(calibration_path),
        "fold_assignments": plan["sources"]["fold_assignments"],
        "denominators": plan["sources"]["denominators"],
        "runner": artifact_record(Path(__file__)),
    }
    artifacts = {
        "model": artifact_record(model_path),
        "preprocessor": artifact_record(preprocessor_path),
        "predictions": artifact_record(prediction_path),
        "decisions": artifact_record(decision_path),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "job_id": str(args.job_id),
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "fold_contract": {
            "early_stop_fold": early_fold,
            "fit_fold_ids": list(fit_fold_ids),
            "fold_local_preprocessing": True,
            "fold_local_calibration": True,
        },
        "input_records": {"train": train_inputs, "validation": validation_inputs},
        "metrics": metrics,
        "model_contract": model_contract,
        "preprocessor": preprocessor_payload,
        "prediction_contract": _prediction_contract(predictions),
        "decision_contract": _decision_contract(decisions),
        "telemetry": telemetry,
        "telemetry_flat": flatten_telemetry(telemetry),
        "execution_provenance": {
            "execution_id": execution["execution_id"],
            "resource_gate": execution["resource_gate"],
            "resource_lease_path": str(expected_lease),
        },
        "sources": sources,
        "artifacts": artifacts,
    }
    result["output_signature_sha256"] = canonical_sha256(
        {
            "configuration": configuration,
            "sources": sources,
            "artifacts": artifacts,
            "metrics": metrics,
            "prediction_contract": result["prediction_contract"],
            "decision_contract": result["decision_contract"],
        }
    )
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    expected_lease = (root.parent / ".d1_heavy_resource.lock").resolve()
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage="d1_ablation_cell",
        route="D1",
        pool="top5",
        method="selected_primary_fixed",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args, resource_lease_path=expected_lease)
        path = root / str(
            _planned_job(
                load_ablation_plan(root / ABLATION_PLAN_RELATIVE), args.job_id
            )["output_manifest"]
        )
        state["artifact_path"] = str(path.resolve())
        state["artifact_sha256"] = sha256_file(path)
        state["feature_set"] = str(result["configuration"]["variant_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
