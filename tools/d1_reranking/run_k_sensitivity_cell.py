"""Run one authorized, label-isolated D1 K-sensitivity cell on CPU."""

from __future__ import annotations

# ruff: noqa: E402 -- native thread limits must be set before numeric imports

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
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

# Keep LightGBM's native OpenMP runtime ahead of PyTorch on the frozen macOS
# experiment environment.  The import stays optional so non-R5 contract tests
# can still run in environments without LightGBM.
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

from d1_reranking.execution import (  # noqa: E402
    artifact_record,
    load_content_manifest,
)
from d1_reranking.fold_calibration import (  # noqa: E402
    apply_partition_calibrator,
    fit_partition_calibrator,
)
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.k_execution import (  # noqa: E402
    K_EXECUTION_ID_ENV,
    K_EXECUTION_POINTER_RELATIVE,
    K_EXECUTION_SCOPE,
    validate_worker_context,
)
from d1_reranking.k_sensitivity import (  # noqa: E402
    K_SENSITIVITY_METHODS,
    K_SENSITIVITY_OUTER_FOLDS,
    K_SENSITIVITY_SEEDS,
    load_k_sensitivity_plan,
    selected_training_spec,
    validate_k_sensitivity_result,
)
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


EXECUTION_ENVIRONMENT_VARIABLE = K_EXECUTION_ID_ENV
EXECUTION_RELATIVE_PATH = K_EXECUTION_POINTER_RELATIVE
EXECUTION_SCOPE = K_EXECUTION_SCOPE
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


def _same_artifact_record(
    observed: object, expected: Mapping[str, Any], *, name: str
) -> None:
    """Match artifact identity while allowing producer-added metadata."""

    if not isinstance(observed, Mapping) or any(
        observed.get(key) != expected.get(key) for key in ("path", "sha256")
    ):
        raise RuntimeError(f"{name} does not bind the current artifact")
    if (
        "bytes" in observed
        and "bytes" in expected
        and observed.get("bytes") != expected.get("bytes")
    ):
        raise RuntimeError(f"{name} byte count differs")


def validate_k_sensitivity_execution(
    root: Path,
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    """Validate an ancestor-owned immutable claim before opening data."""

    return validate_worker_context(root, plan_path=plan_path, plan=plan, job=job)


def _bound_manifest(
    record: Mapping[str, Any], *, name: str, statuses: tuple[str, ...] = ("COMPLETE",)
) -> tuple[Path, dict[str, Any]]:
    path = verified_artifact_path(record, name=name)
    manifest = load_content_manifest(path, name=name, statuses=statuses)
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


def _exact_local_artifact(
    record: Mapping[str, Any], *, expected: Path, name: str
) -> Path:
    observed = verified_artifact_path(record, name=name)
    if observed != expected.resolve():
        raise RuntimeError(f"{name} path differs from its split-local contract")
    return observed


def _load_split(
    *,
    split: str,
    pool: str,
    track: str,
    max_candidates: int,
    candidate_record: Mapping[str, Any],
    feature_record: Mapping[str, Any],
    label_record: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    """Open one exact Train/Validation source closure; Test is unrepresentable."""

    if split not in {"train", "validation"}:
        raise RuntimeError("D1 K-sensitivity permits only Train/Validation inputs")
    candidate_manifest_path, candidate_manifest = _bound_manifest(
        candidate_record, name=f"D1 K-sensitivity {split} candidate manifest"
    )
    feature_manifest_path, feature_manifest = _bound_manifest(
        feature_record, name=f"D1 K-sensitivity {split}/{pool}/{track} features"
    )
    label_manifest_path, label_manifest = _bound_manifest(
        label_record, name=f"D1 K-sensitivity {split}/{pool} labels"
    )
    candidate_configuration = _mapping(
        candidate_manifest.get("configuration"),
        name=f"D1 K-sensitivity {split} candidate configuration",
    )
    feature_configuration = _mapping(
        feature_manifest.get("configuration"),
        name=f"D1 K-sensitivity {split} feature configuration",
    )
    feature_sources = _mapping(
        feature_manifest.get("sources"),
        name=f"D1 K-sensitivity {split} feature sources",
    )
    label_sources = _mapping(
        label_manifest.get("sources"),
        name=f"D1 K-sensitivity {split} label sources",
    )
    if (
        candidate_configuration.get("route") != "D1"
        or candidate_configuration.get("split") != split
        or feature_configuration.get("route") != "D1"
        or feature_configuration.get("split") != split
        or feature_configuration.get("pool") != pool
        or feature_configuration.get("track") != track
        or label_manifest.get("split") != split
        or label_manifest.get("pool") != pool
        or any(
            manifest.get("candidate_test_labels_read") is not False
            for manifest in (candidate_manifest, feature_manifest, label_manifest)
        )
    ):
        raise RuntimeError(f"D1 K-sensitivity {split} source semantics differ")
    summaries = _mapping(
        candidate_manifest.get("summaries"),
        name=f"D1 K-sensitivity {split} candidate summaries",
    )
    pool_summary = _mapping(
        summaries.get(pool), name=f"D1 K-sensitivity {split}/{pool} summary"
    )
    declared_maximum = pool_summary.get("maximum_candidates")
    if (
        not isinstance(declared_maximum, int)
        or isinstance(declared_maximum, bool)
        or declared_maximum <= 0
        or declared_maximum > max_candidates
        or (pool == "top10" and max_candidates != 10)
    ):
        raise RuntimeError(f"D1 K-sensitivity {split}/{pool} maximum differs")

    candidate_artifacts = _mapping(
        candidate_manifest.get("artifacts"),
        name=f"D1 K-sensitivity {split} candidate artifacts",
    )
    candidate_path = _exact_local_artifact(
        _mapping(candidate_artifacts.get(pool), name=f"D1 {split}/{pool} candidates"),
        expected=candidate_manifest_path.parent / f"d1_{pool}_candidates.parquet",
        name=f"D1 K-sensitivity {split}/{pool} candidate table",
    )
    candidates_record = artifact_record(candidate_path)
    candidate_manifest_record = artifact_record(candidate_manifest_path)
    for source_name, observed, expected in (
        (
            "feature candidate manifest",
            feature_sources.get("candidate_manifest"),
            candidate_manifest_record,
        ),
        ("feature candidates", feature_sources.get("candidates"), candidates_record),
        (
            "label candidate manifest",
            label_sources.get("candidate_manifest"),
            candidate_manifest_record,
        ),
        ("label candidates", label_sources.get("candidates"), candidates_record),
    ):
        _same_artifact_record(
            observed,
            expected,
            name=f"D1 K-sensitivity {split} {source_name}",
        )
    feature_artifacts = _mapping(
        feature_manifest.get("artifacts"),
        name=f"D1 K-sensitivity {split} feature artifacts",
    )
    feature_path = _exact_local_artifact(
        _mapping(
            feature_artifacts.get("candidate_features"),
            name=f"D1 K-sensitivity {split} feature table",
        ),
        expected=feature_manifest_path.parent / "candidate_features.parquet",
        name=f"D1 K-sensitivity {split} feature table",
    )
    label_path = _exact_local_artifact(
        _mapping(
            label_manifest.get("artifact"),
            name=f"D1 K-sensitivity {split} label table",
        ),
        expected=label_manifest_path.parent / "candidate_labels.parquet",
        name=f"D1 K-sensitivity {split} label table",
    )
    columns = tuple(map(str, feature_manifest.get("model_feature_columns", ())))
    if not columns or not {"base_logit", "calibrated_native_probability"}.issubset(
        columns
    ):
        raise RuntimeError(f"D1 K-sensitivity {split} feature schema differs")
    joined = join_development_features_and_labels(
        pd.read_parquet(feature_path), pd.read_parquet(label_path)
    )
    identity_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
    ]
    candidates = pd.read_parquet(candidate_path, columns=identity_columns)
    if candidates.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError(f"D1 K-sensitivity {split} candidates contain duplicates")
    frozen = candidates.rename(
        columns={column: f"_frozen_{column}" for column in identity_columns[2:]}
    )
    joined = joined.merge(
        frozen,
        on=["sample_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(candidates):
        raise RuntimeError(f"D1 K-sensitivity {split} candidate membership differs")
    for column in identity_columns[2:]:
        frozen_column = f"_frozen_{column}"
        if column in joined:
            if column == "native_rank":
                equal = np.array_equal(
                    joined[column].to_numpy(int), joined[frozen_column].to_numpy(int)
                )
            else:
                equal = (
                    joined[column].astype(str).equals(joined[frozen_column].astype(str))
                )
            if not equal:
                raise RuntimeError(f"D1 K-sensitivity {split} frozen {column} differs")
            joined = joined.drop(columns=column)
        joined = joined.rename(columns={frozen_column: column})
    if "native_score_raw" not in joined:
        raise RuntimeError(f"D1 K-sensitivity {split} lacks native q")
    observed_maximum = int(joined.groupby("sample_id", sort=False).size().max())
    if observed_maximum != declared_maximum or observed_maximum > max_candidates:
        raise RuntimeError(f"D1 K-sensitivity {split} observed pool maximum differs")
    return (
        joined,
        columns,
        {
            "candidate_manifest": candidate_manifest_record,
            "feature_manifest": artifact_record(feature_manifest_path),
            "label_manifest": artifact_record(label_manifest_path),
            "candidates": candidates_record,
            "features": artifact_record(feature_path),
            "labels": artifact_record(label_path),
            "feature_extraction_latency_ms": float(
                feature_manifest.get("feature_extraction_latency_ms", 0.0)
            ),
        },
    )


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


def _prediction_rows(arrays: Any, values: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cursor = 0
    for sample_id, candidate_ids in zip(arrays.sample_ids, arrays.candidate_ids):
        for candidate_id in candidate_ids:
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "score": float(values[cursor]),
                }
            )
            cursor += 1
    if cursor != len(values):
        raise RuntimeError("D1 K-sensitivity score vector length differs")
    return pd.DataFrame(rows)


def _atomic_torch(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    return path


def _atomic_native_lightgbm(path: Path, model: Any) -> Path:
    """Persist only LightGBM native text; pickle model loading is forbidden."""

    booster = getattr(getattr(model, "model", None), "booster_", None)
    if booster is None or not callable(getattr(booster, "save_model", None)):
        raise RuntimeError("D1 K fitted R5 model has no native LightGBM booster")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    booster.save_model(str(temporary))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _numerical_code_records() -> list[dict[str, str]]:
    return [
        artifact_record(path)
        for path in (
            ROOT / "src/d1_reranking/models.py",
            ROOT / "src/d1_reranking/fold_calibration.py",
            ROOT / "src/unified_reranking/datasets.py",
            ROOT / "src/unified_reranking/training.py",
            ROOT / "src/unified_reranking/losses.py",
            ROOT / "src/unified_reranking/models/lightgbm_ranker.py",
            ROOT / "src/unified_reranking/metrics.py",
            ROOT / "src/unified_reranking/telemetry.py",
        )
    ]


def _prediction_contract(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    identity = [
        [
            str(row.sample_id),
            str(row.candidate_id),
            int(row.native_rank),
            str(row.candidate_identity_sha256),
            str(row.candidate_geometry_sha256),
        ]
        for row in ordered.itertuples(index=False)
    ]
    scores = [
        [str(row.sample_id), str(row.candidate_id), float(row.score).hex()]
        for row in ordered.itertuples(index=False)
    ]
    value: dict[str, Any] = {
        "rows": len(ordered),
        "candidate_universe_sha256": canonical_sha256(identity),
        "score_vector_sha256": canonical_sha256(scores),
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def _decision_contract(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.reset_index(drop=True)
    samples = ordered["sample_id"].astype(str).tolist()
    selections = [
        [
            str(row.sample_id),
            None
            if pd.isna(row.selected_candidate_id)
            else str(row.selected_candidate_id),
            bool(row.selected_correct),
        ]
        for row in ordered.itertuples(index=False)
    ]
    value: dict[str, Any] = {
        "rows": len(ordered),
        "sample_universe_sha256": canonical_sha256(samples),
        "selected_vector_sha256": canonical_sha256(selections),
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def _planned_job(plan: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    matching = [
        _mapping(job, name="D1 K-sensitivity planned job")
        for job in plan.get("jobs", ())
        if isinstance(job, Mapping) and job.get("job_id") == job_id
    ]
    if len(matching) != 1:
        raise RuntimeError("D1 K-sensitivity job is outside the frozen 54-job universe")
    job = matching[0]
    configuration = _mapping(
        job.get("configuration"), name="D1 K-sensitivity job configuration"
    )
    held_fold = configuration.get("held_fold")
    if (
        canonical_sha256(configuration)[:16] != job_id
        or configuration.get("method") not in K_SENSITIVITY_METHODS
        or configuration.get("seed") not in K_SENSITIVITY_SEEDS
        or configuration.get("mode") not in {"oof", "validation"}
        or (configuration.get("mode") == "oof") != (held_fold is not None)
        or (
            held_fold is not None
            and (
                not isinstance(held_fold, int)
                or isinstance(held_fold, bool)
                or not 0 <= held_fold < K_SENSITIVITY_OUTER_FOLDS
            )
        )
        or not isinstance(configuration.get("max_candidates"), int)
        or int(configuration["max_candidates"]) <= 0
    ):
        raise RuntimeError("D1 K-sensitivity planned job semantics differ")
    return job


def run(
    args: argparse.Namespace, *, resource_lease_path: Path | None = None
) -> dict[str, Any]:
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    expected_lease = (root.parent / ".d1_heavy_resource.lock").resolve()
    if resource_lease_path is None or resource_lease_path.resolve() != expected_lease:
        raise RuntimeError("D1 K-sensitivity cell requires the repository-wide lease")
    plan_path = root / "configs" / "d1_k_sensitivity_plan.json"
    plan = load_k_sensitivity_plan(plan_path)
    job = _planned_job(plan, str(args.job_id))
    configuration = _mapping(
        job.get("configuration"), name="D1 K-sensitivity configuration"
    )
    (
        execution_path,
        execution,
        execution_event_path,
        _execution_event,
        execution_claim_path,
        _execution_claim,
    ) = validate_k_sensitivity_execution(
        root,
        plan_path=plan_path,
        plan=plan,
        job=job,
    )
    output = (root / str(job["output_manifest"])).resolve().parent
    expected_output_root = (root / "11_k_sensitivity" / "cells").resolve()
    if output.parent != expected_output_root or output.name != str(args.job_id):
        raise RuntimeError("D1 K-sensitivity output path differs from the plan")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path,
            name=f"D1 K-sensitivity cell {args.job_id}",
            statuses=("COMPLETE",),
        )
        if not args.resume:
            raise FileExistsError(
                f"D1 K-sensitivity cell already exists: {manifest_path}"
            )
        return validate_k_sensitivity_result(
            existing,
            plan_path=plan_path,
            job=job,
            manifest_path=manifest_path,
        )

    sources_plan = _mapping(plan.get("sources"), name="D1 K-sensitivity sources")
    scenario_id = str(configuration["scenario_id"])
    pool = str(configuration["pool"])
    track = str(configuration["track"])
    max_candidates = int(configuration["max_candidates"])
    candidate_records = _mapping(
        sources_plan.get("candidate_manifests"),
        name="D1 K-sensitivity candidate manifests",
    )
    feature_records = _mapping(
        _mapping(
            sources_plan.get("feature_manifests"),
            name="D1 K-sensitivity feature manifests",
        ).get(scenario_id),
        name="D1 K-sensitivity scenario feature manifests",
    )
    label_records = _mapping(
        _mapping(
            sources_plan.get("development_label_manifests"),
            name="D1 K-sensitivity label manifests",
        ).get(pool),
        name="D1 K-sensitivity pool label manifests",
    )

    feature_started = time.perf_counter()
    train, columns, train_inputs = _load_split(
        split="train",
        pool=pool,
        track=track,
        max_candidates=max_candidates,
        candidate_record=_mapping(
            candidate_records.get("train"), name="Train candidates"
        ),
        feature_record=_mapping(feature_records.get("train"), name="Train features"),
        label_record=_mapping(label_records.get("train"), name="Train labels"),
    )
    folds_record = _mapping(
        sources_plan.get("fold_assignments"), name="D1 K-sensitivity folds"
    )
    folds_path = verified_artifact_path(folds_record, name="D1 K-sensitivity folds")
    folds = pd.read_parquet(folds_path, columns=["sample_id", "fold"])
    if folds["sample_id"].astype(str).duplicated().any() or set(
        folds["fold"].astype(int).unique()
    ) != set(range(K_SENSITIVITY_OUTER_FOLDS)):
        raise RuntimeError("D1 K-sensitivity fold assignment contract differs")
    train = train.merge(folds, on="sample_id", how="left", validate="many_to_one")
    if train["fold"].isna().any():
        raise RuntimeError("D1 K-sensitivity folds do not cover Train candidates")

    mode = str(configuration["mode"])
    held_fold = configuration.get("held_fold")
    early_fold = (
        (int(held_fold) + 1) % K_SENSITIVITY_OUTER_FOLDS if held_fold is not None else 0
    )
    fit_precal = train.loc[
        (train["fold"] != early_fold)
        & ((train["fold"] != held_fold) if held_fold is not None else True)
    ].copy()
    early_precal = train.loc[train["fold"] == early_fold].copy()
    validation_inputs: dict[str, Any] | None = None
    if mode == "oof":
        predict_precal = train.loc[train["fold"] == held_fold].copy()
        denominator_ids = (
            folds.loc[folds["fold"] == held_fold, "sample_id"].astype(str).tolist()
        )
    else:
        predict_precal, validation_columns, validation_inputs = _load_split(
            split="validation",
            pool=pool,
            track=track,
            max_candidates=max_candidates,
            candidate_record=_mapping(
                candidate_records.get("validation"), name="Validation candidates"
            ),
            feature_record=_mapping(
                feature_records.get("validation"), name="Validation features"
            ),
            label_record=_mapping(
                label_records.get("validation"), name="Validation labels"
            ),
        )
        if validation_columns != columns:
            raise RuntimeError("D1 K-sensitivity Train/Validation schemas differ")
        denominator_records = _mapping(
            sources_plan.get("denominators"), name="D1 K-sensitivity denominators"
        )
        denominator_path = verified_artifact_path(
            _mapping(
                denominator_records.get("validation"), name="Validation denominator"
            ),
            name="D1 K-sensitivity Validation denominator",
        )
        denominator_ids = (
            pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
            .astype(str)
            .tolist()
        )
    if not denominator_ids or len(denominator_ids) != len(set(denominator_ids)):
        raise RuntimeError("D1 K-sensitivity denominator universe is invalid")

    calibration_record = _mapping(
        _mapping(
            sources_plan.get("calibration_manifests"),
            name="D1 K-sensitivity calibration manifests",
        ).get(pool),
        name="D1 K-sensitivity calibration manifest",
    )
    calibration_path, calibration_manifest = _bound_manifest(
        calibration_record, name=f"D1 K-sensitivity {pool} calibration"
    )
    selected_calibration = str(calibration_manifest.get("selected_method", ""))
    if not selected_calibration:
        raise RuntimeError("D1 K-sensitivity selected calibration family is absent")
    fit_fold_ids = tuple(map(int, sorted(fit_precal["fold"].astype(int).unique())))
    fold_calibrator = fit_partition_calibrator(
        fit_precal, method=selected_calibration, fit_fold_ids=fit_fold_ids
    )
    fit_rows = apply_partition_calibrator(fit_precal, fold_calibrator)
    early_rows = apply_partition_calibrator(early_precal, fold_calibrator)
    predict_rows = apply_partition_calibrator(predict_precal, fold_calibrator)
    if sorted(set(columns).difference(fit_rows.columns)):
        raise RuntimeError("D1 K-sensitivity calibrated feature schema is incomplete")
    preprocessor = FoldPreprocessor.fit(fit_rows, columns)
    fit_arrays = build_query_arrays(
        fit_rows, preprocessor=preprocessor, max_candidates=max_candidates
    )
    early_arrays = build_query_arrays(
        early_rows, preprocessor=preprocessor, max_candidates=max_candidates
    )
    predict_arrays = build_query_arrays(
        predict_rows, preprocessor=preprocessor, max_candidates=max_candidates
    )
    processed = pd.concat([fit_rows, early_rows, predict_rows], ignore_index=True)
    feature_latency_ms = (
        (time.perf_counter() - feature_started) * 1000.0 / len(processed)
    )
    selected_missing = missing_feature_rate(processed, columns)

    selected_configuration = _mapping(
        configuration.get("selected_primary_configuration"),
        name="D1 K-sensitivity selected primary configuration",
    )
    if canonical_sha256(selected_configuration) != configuration.get(
        "selected_primary_configuration_sha256"
    ) or selected_configuration.get("method") != configuration.get("method"):
        raise RuntimeError("D1 K-sensitivity selected hyperparameters differ")
    method = str(configuration["method"])
    seed = int(configuration["seed"])
    training_spec = selected_training_spec(selected_configuration, method=method)
    training_metadata: dict[str, Any]
    if method == "R5":
        fit_x, fit_y, fit_q = _flat_arrays(fit_arrays)
        early_x, early_y, early_q = _flat_arrays(early_arrays)
        predict_x, _, _ = _flat_arrays(predict_arrays)
        model = LightGBMLambdaRank(
            seed=seed,
            num_leaves=int(training_spec["num_leaves"]),
            learning_rate=float(training_spec["learning_rate"]),
            n_estimators=int(training_spec["n_estimators"]),
        ).fit(fit_x, fit_y, fit_q, eval_set=(early_x, early_y, early_q))
        started = time.perf_counter()
        values = model.predict(predict_x)
        ranker_latency_ms = (time.perf_counter() - started) * 1000.0 / len(values)
        score_rows = _prediction_rows(predict_arrays, values)
        training_metadata = model.artifact()
        parameter_count = lightgbm_parameter_count(model)
        model_kind = "lightgbm_lambdarank"
    else:
        set_deterministic_cpu(seed)
        hidden_dims = tuple(map(int, training_spec["hidden_dims"]))
        model = D1ResidualMLPScorer(
            len(columns),
            hidden_dims=hidden_dims,
            dropout=float(training_spec["dropout"]),
            alpha=float(training_spec["alpha"]),
        )
        training_config = NeuralTrainingConfig(
            loss=METHOD_LOSSES[method],
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
        training = fit_neural_ranker(
            model, fit_arrays, early_arrays, config=training_config
        )
        model.load_state_dict(training.state_dict)
        started = time.perf_counter()
        score_rows = predict_neural_ranker(model, predict_arrays)
        ranker_latency_ms = (time.perf_counter() - started) * 1000.0 / len(score_rows)
        training_metadata = {
            "best_epoch": training.best_epoch,
            "best_validation_loss": training.best_validation_loss,
            "epochs_ran": training.epochs_ran,
            "history": list(training.history),
            "training_config": training.config,
        }
        parameter_count = torch_parameter_count(model)
        model_kind = "residual_mlp"

    model_contract: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "kind": model_kind,
        "seed": seed,
        "selected_primary_trial_id": configuration["selected_primary_trial_id"],
        "selected_primary_configuration_sha256": configuration[
            "selected_primary_configuration_sha256"
        ],
        "effective_training_hyperparameters": training_spec,
        "feature_schema_sha256": canonical_sha256(columns),
        "max_candidates": max_candidates,
        "training": training_metadata,
        "parameter_count": parameter_count,
        "serialization": (
            "lightgbm_native_text" if method == "R5" else "torch_state_dict"
        ),
    }
    model_contract["content_sha256"] = canonical_sha256(model_contract)
    if method == "R5":
        model_path = _atomic_native_lightgbm(output / "model.txt", model)
    else:
        model_path = _atomic_torch(
            output / "model.pt",
            {
                "state_dict": training.state_dict,
                "input_dim": len(columns),
                "model_contract": model_contract,
            },
        )

    identity = predict_rows[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ]
    ].copy()
    if score_rows.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError("D1 K-sensitivity predictions contain duplicates")
    predictions = identity.merge(
        score_rows[["sample_id", "candidate_id", "score"]],
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    predictions = (
        predictions.loc[:, list(PREDICTION_COLUMNS)]
        .sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(predictions) != len(identity)
        or not np.isfinite(predictions["score"].to_numpy(float)).all()
    ):
        raise RuntimeError("D1 K-sensitivity predictions do not cover candidates")
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
    prediction_contract = _prediction_contract(predictions)
    decision_contract = _decision_contract(decisions)
    prediction_path = atomic_parquet(predictions, output / "predictions.parquet")
    decision_path = atomic_parquet(decisions, output / "per_sample_decisions.parquet")
    preprocessor_payload: dict[str, Any] = {
        "schema_version": 1,
        "job_id": str(args.job_id),
        "feature_schema_sha256": canonical_sha256(columns),
        "fold_calibrator": fold_calibrator,
        "fold_preprocessor": preprocessor.artifact(),
    }
    preprocessor_payload["content_sha256"] = canonical_sha256(preprocessor_payload)
    preprocessor_path = output / "preprocessor.json"
    atomic_json(preprocessor_path, preprocessor_payload)

    telemetry = telemetry_payload(
        phase=f"d1_k_sensitivity_{scenario_id}_{mode}",
        parameter_count=parameter_count,
        ranker_latency_ms=ranker_latency_ms,
        feature_latency_ms=feature_latency_ms,
        missing_feature_rate_value=selected_missing,
    )
    result_sources: dict[str, Any] = {
        "plan": artifact_record(plan_path),
        "execution_manifest": artifact_record(execution_path),
        "execution_event": artifact_record(execution_event_path),
        "execution_claim": artifact_record(execution_claim_path),
        "selected_primary": sources_plan["selected_primary"],
        "selected_trial": sources_plan["selected_trial"],
        "candidate_manifests": {
            "train": candidate_records["train"],
            "validation": candidate_records["validation"],
        },
        "feature_manifests": {
            "train": feature_records["train"],
            "validation": feature_records["validation"],
        },
        "development_label_manifests": {
            "train": label_records["train"],
            "validation": label_records["validation"],
        },
        "calibration_manifest": artifact_record(calibration_path),
        "fold_assignments": folds_record,
        "denominators": sources_plan["denominators"],
        "runner_code": artifact_record(Path(__file__)),
        "numerical_code": _numerical_code_records(),
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
        "prediction_contract": prediction_contract,
        "decision_contract": decision_contract,
        "telemetry": telemetry,
        "telemetry_flat": flatten_telemetry(telemetry),
        "execution_provenance": {
            "execution_id": execution["execution_id"],
            "execution_manifest": artifact_record(execution_path),
            "execution_event": artifact_record(execution_event_path),
            "execution_claim": artifact_record(execution_claim_path),
            "resource_gate": execution["resource_gate"],
            "resource_lease_path": str(expected_lease),
            "scope": EXECUTION_SCOPE,
        },
        "sources": result_sources,
        "artifacts": artifacts,
    }
    result["output_signature_sha256"] = canonical_sha256(
        {
            "configuration": configuration,
            "sources": result_sources,
            "artifacts": artifacts,
            "model_contract": model_contract,
            "preprocessor": preprocessor_payload,
            "prediction_contract": prediction_contract,
            "decision_contract": decision_contract,
            "telemetry": telemetry,
        }
    )
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    return validate_k_sensitivity_result(
        result, plan_path=plan_path, job=job, manifest_path=manifest_path
    )


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    lease_path = (root.parent / ".d1_heavy_resource.lock").resolve()
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P10",
        substage=f"d1_k_sensitivity_{args.job_id}",
        route="D1",
        pool="planned_job",
        evidence_track="planned_job",
        method="selected_primary",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(args, resource_lease_path=lease_path)
        manifest_path = (
            root / "11_k_sensitivity" / "cells" / args.job_id / "manifest.json"
        )
        state["artifact_path"] = str(manifest_path)
        state["artifact_sha256"] = sha256_file(manifest_path)
        print(
            json.dumps(
                {
                    "job_id": result["job_id"],
                    "manifest": str(manifest_path),
                    "sha256": state["artifact_sha256"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
