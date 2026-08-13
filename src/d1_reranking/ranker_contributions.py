"""Exact label-free LightGBM contribution replay for the locked D1 R5 ranker."""

from __future__ import annotations

# LightGBM must be imported before Torch (transitively imported by datasets) on
# macOS to avoid mixing incompatible OpenMP runtimes during native inference.
try:
    import lightgbm as _lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - validated by the CLI boundary
    _lightgbm = None

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.datasets import FoldPreprocessor, build_inference_query_arrays
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.test_access_guard import append_access_log

from .contracts import assert_label_free_parquet_schema
from .execution import load_content_manifest
from .fold_calibration import apply_partition_calibrator
from .io import atomic_parquet
from .run import assert_writable_prelock


MANIFEST_RELATIVE_PATH = (
    "08_lock/postformal_sources/r5_candidate_contributions_manifest.json"
)
ARTIFACT_RELATIVE_PATH = "08_lock/postformal_sources/r5_candidate_contributions.parquet"
RANKER_RELATIVE_PATH = "08_lock/label_free_test_rankers/d1/manifest.json"
BIAS_FEATURE_NAME = "__bias__"
SEEDS = (42, 123, 2026)
KEYS = ("sample_id", "candidate_id")


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 ranker-contribution source is not a regular file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _same_record(observed: object, expected: Mapping[str, Any], *, name: str) -> None:
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


def _raw_feature_columns(
    model_columns: tuple[str, ...], schema_columns: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "sample_id",
                "candidate_id",
                "native_rank",
                "native_score_raw",
                *(column for column in model_columns if column in schema_columns),
            )
        )
    )


def _query_keys(arrays: Any) -> pd.DataFrame:
    rows = [
        {"sample_id": str(sample_id), "candidate_id": str(candidate_id)}
        for sample_id, candidate_ids in zip(
            arrays.sample_ids, arrays.candidate_ids, strict=True
        )
        for candidate_id in candidate_ids
    ]
    return pd.DataFrame(rows, columns=list(KEYS))


def _flat_features(arrays: Any) -> np.ndarray:
    return arrays.features.numpy()[~arrays.padding_mask.numpy()]


def _native_contribution_matrix(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Return audited native LightGBM SHAP contributions, including bias."""

    booster = model.model.booster_
    if matrix.ndim != 2 or matrix.shape[1] != int(booster.num_feature()):
        raise RuntimeError("D1 contribution feature width differs from locked model")
    values = np.asarray(
        booster.predict(np.asarray(matrix, dtype=np.float64), pred_contrib=True),
        dtype=np.float64,
    )
    if values.shape != (len(matrix), matrix.shape[1] + 1):
        raise RuntimeError("D1 LightGBM contribution matrix shape differs")
    if not np.isfinite(values).all():
        raise RuntimeError("D1 LightGBM contributions contain non-finite values")
    scores = np.asarray(model.predict(matrix), dtype=np.float64)
    if not np.allclose(values.sum(axis=1), scores, rtol=1e-7, atol=1e-8):
        raise RuntimeError("D1 LightGBM contribution additivity differs from prediction")
    return values


def _long_contribution_frame(
    keys: pd.DataFrame,
    values: np.ndarray,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    names = (*feature_names, BIAS_FEATURE_NAME)
    if values.shape != (len(keys), len(names)):
        raise RuntimeError("D1 contribution rows/features differ from their schema")
    return pd.DataFrame(
        {
            "sample_id": np.repeat(keys["sample_id"].astype(str).to_numpy(), len(names)),
            "candidate_id": np.repeat(
                keys["candidate_id"].astype(str).to_numpy(), len(names)
            ),
            "feature_name": np.tile(np.asarray(names, dtype=object), len(keys)),
            "contribution": values.reshape(-1),
        }
    )


def _load_sources(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ranker_path = root / RANKER_RELATIVE_PATH
    ranker = load_content_manifest(
        ranker_path, name="D1 selected Test ranker", statuses=("COMPLETE",)
    )
    if (
        ranker.get("candidate_test_labels_read") is not False
        or str(ranker.get("selected_method", "")).upper() not in {"R5", "LAMBDAMART"}
    ):
        raise RuntimeError("D1 candidate contributions require label-free selected R5")
    verify_artifact_records_recursive(
        {
            "sources": ranker.get("sources"),
            "artifacts": ranker.get("artifacts"),
            "seed_applications": ranker.get("seed_applications"),
        },
        name="D1 selected Test ranker",
        require_at_least_one=True,
    )
    sources = ranker.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("D1 selected Test ranker sources are absent")
    raw_manifest_path = verified_artifact_path(
        sources.get("raw_feature_manifest", {}), name="D1 raw Test feature manifest"
    )
    final_manifest_path = verified_artifact_path(
        sources.get("final_feature_manifest", {}), name="D1 final Test feature manifest"
    )
    raw_manifest = load_content_manifest(
        raw_manifest_path, name="D1 raw Test features", statuses=("COMPLETE",)
    )
    final_manifest = load_content_manifest(
        final_manifest_path, name="D1 final Test features", statuses=("COMPLETE",)
    )
    for name, value in (("raw", raw_manifest), ("final", final_manifest)):
        if value.get("candidate_test_labels_read") is not False:
            raise PermissionError(f"D1 {name} Test features are not label-free")
        verify_artifact_records_recursive(
            {"sources": value.get("sources"), "artifacts": value.get("artifacts")},
            name=f"D1 {name} Test features",
            require_at_least_one=True,
        )
    return ranker, raw_manifest, final_manifest


def _cell_contributions(
    *,
    application: Mapping[str, Any],
    raw_features: pd.DataFrame,
    model_columns: tuple[str, ...],
    load_native_ranker: Callable[[Path], Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cell_path = verified_artifact_path(application.get("cell", {}), name="D1 R5 cell")
    cell = load_content_manifest(cell_path, name="D1 R5 cell", statuses=("COMPLETE",))
    verify_artifact_records_recursive(
        {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
        name="D1 R5 cell",
        require_at_least_one=True,
    )
    configuration = cell.get("configuration")
    if (
        not isinstance(configuration, Mapping)
        or configuration.get("method") != "R5"
        or tuple(map(str, configuration.get("feature_columns", ()))) != model_columns
    ):
        raise RuntimeError("D1 contribution cell configuration differs from selected R5")
    model_path = verified_artifact_path(
        application.get("model", {}), name="D1 R5 application model"
    )
    preprocessor_path = verified_artifact_path(
        application.get("preprocessor", {}), name="D1 R5 application preprocessor"
    )
    _same_record(
        cell.get("artifacts", {}).get("model"),
        _record(model_path),
        name="D1 R5 cell model",
    )
    _same_record(
        cell.get("artifacts", {}).get("preprocessor"),
        _record(preprocessor_path),
        name="D1 R5 cell preprocessor",
    )
    persisted = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    if persisted != cell.get("preprocessor"):
        raise RuntimeError("D1 R5 embedded/persisted preprocessors differ")
    preprocessor = FoldPreprocessor.from_artifact(persisted)
    if preprocessor.columns != model_columns:
        raise RuntimeError("D1 R5 contribution preprocessor schema differs")
    calibrated = apply_partition_calibrator(
        raw_features, configuration["fold_local_calibrator"]
    )
    arrays = build_inference_query_arrays(
        calibrated, preprocessor=preprocessor, max_candidates=5
    )
    matrix = _flat_features(arrays)
    model = load_native_ranker(model_path)
    values = _native_contribution_matrix(model, matrix)
    scores = np.asarray(model.predict(matrix), dtype=np.float64)
    return _query_keys(arrays), values, scores


def validate_ranker_contributions(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    value = load_content_manifest(
        manifest_path, name="D1 R5 candidate contributions", statuses=("COMPLETE",)
    )
    if (
        value.get("candidate_test_labels_read") is not False
        or value.get("selected_method") != "R5"
        or value.get("seeds") != list(SEEDS)
        or value.get("source_signature_sha256")
        != canonical_sha256(value.get("sources"))
    ):
        raise RuntimeError("D1 R5 candidate-contribution contract differs")
    verify_artifact_records_recursive(
        {"sources": value.get("sources"), "artifacts": value.get("artifacts")},
        name="D1 R5 candidate contributions",
        require_at_least_one=True,
    )
    artifact = verified_artifact_path(
        value.get("artifacts", {}).get("candidate_contributions", {}),
        name="D1 R5 candidate contribution artifact",
    )
    if artifact != (root / ARTIFACT_RELATIVE_PATH).resolve():
        raise RuntimeError("D1 R5 candidate-contribution path differs")
    columns = assert_label_free_parquet_schema(
        artifact, name="D1 R5 candidate contributions"
    )
    if columns != ("sample_id", "candidate_id", "feature_name", "contribution"):
        raise RuntimeError("D1 R5 candidate-contribution schema differs")
    return manifest_path, value


def build_ranker_contributions(
    run_dir: str | Path,
    *,
    resume: bool,
    load_native_ranker: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    assert_writable_prelock(root)
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if manifest_path.exists():
        if not resume:
            raise RuntimeError("D1 R5 candidate contributions already exist")
        _, existing = validate_ranker_contributions(root)
        append_access_log(
            root,
            {
                "event": "prelock_label_free_test_stage",
                "stage": "d1_r5_candidate_contributions",
                "output_manifest": str(manifest_path.resolve()),
                "output_manifest_sha256": sha256_file(manifest_path),
                "candidate_labels_opened_as_table": False,
                "resumed": True,
            },
        )
        return existing
    ranker, raw_manifest, final_manifest = _load_sources(root)
    applications = ranker.get("seed_applications")
    if not isinstance(applications, Mapping) or set(applications) != set(map(str, SEEDS)):
        raise RuntimeError("D1 selected R5 ranker lacks exactly three seed applications")
    raw_path = verified_artifact_path(
        raw_manifest.get("artifacts", {}).get("candidate_features", {}),
        name="D1 raw Test candidate features",
    )
    schema = assert_label_free_parquet_schema(raw_path, name="D1 raw Test features")
    model_columns = tuple(map(str, final_manifest.get("model_feature_columns", ())))
    if not model_columns:
        raise RuntimeError("D1 final Test model feature schema is absent")
    raw_features = pd.read_parquet(
        raw_path, columns=list(_raw_feature_columns(model_columns, schema))
    )
    if raw_features.duplicated(list(KEYS)).any():
        raise RuntimeError("D1 raw Test features contain duplicate candidate keys")
    score_path = verified_artifact_path(
        ranker.get("artifacts", {}).get("per_candidate_scores", {}),
        name="D1 selected Test ranker scores",
    )
    scores = pd.read_parquet(score_path)
    required_score_columns = {
        *KEYS,
        *(f"score_seed_{seed}" for seed in SEEDS),
        "ensemble_score",
    }
    if required_score_columns.difference(scores.columns) or scores.duplicated(list(KEYS)).any():
        raise RuntimeError("D1 selected Test ranker score schema differs")
    if load_native_ranker is None:
        if _lightgbm is None:
            raise ModuleNotFoundError("LightGBM is required for D1 R5 contributions")
        from tools.unified_reranking.apply_locked_matrix_cell import (
            _load_native_lightgbm_ranker,
        )

        load_native_ranker = _load_native_lightgbm_ranker
    seed_values: list[np.ndarray] = []
    canonical_keys: pd.DataFrame | None = None
    for seed in SEEDS:
        keys, values, predicted = _cell_contributions(
            application=applications[str(seed)],
            raw_features=raw_features,
            model_columns=model_columns,
            load_native_ranker=load_native_ranker,
        )
        if canonical_keys is None:
            canonical_keys = keys
        elif not keys.equals(canonical_keys):
            raise RuntimeError("D1 R5 seed contribution candidate ordering differs")
        expected = keys.merge(
            scores[[*KEYS, f"score_seed_{seed}"]],
            on=list(KEYS),
            how="left",
            validate="one_to_one",
        )[f"score_seed_{seed}"].to_numpy(float)
        if not np.allclose(predicted, expected, rtol=1e-7, atol=1e-8):
            raise RuntimeError(f"D1 R5 seed {seed} replay differs from locked scores")
        seed_values.append(values)
    if canonical_keys is None:
        raise RuntimeError("D1 R5 contribution candidate universe is empty")
    averaged = np.mean(np.stack(seed_values, axis=0), axis=0)
    expected_ensemble = canonical_keys.merge(
        scores[[*KEYS, "ensemble_score"]],
        on=list(KEYS),
        how="left",
        validate="one_to_one",
    )["ensemble_score"].to_numpy(float)
    contribution_sums = averaged.sum(axis=1)
    if not np.allclose(contribution_sums, expected_ensemble, rtol=1e-7, atol=1e-8):
        raise RuntimeError("D1 averaged R5 contributions differ from ensemble scores")
    frame = _long_contribution_frame(canonical_keys, averaged, model_columns)
    artifact_path = atomic_parquet(frame, root / ARTIFACT_RELATIVE_PATH)
    sources = {
        "ranker_manifest": _record(root / RANKER_RELATIVE_PATH),
        "raw_feature_manifest": _record(
            verified_artifact_path(
                ranker.get("sources", {}).get("raw_feature_manifest", {}),
                name="D1 R5 raw feature manifest",
            )
        ),
        "final_feature_manifest": _record(
            verified_artifact_path(
                ranker.get("sources", {}).get("final_feature_manifest", {}),
                name="D1 R5 final feature manifest",
            )
        ),
        "per_candidate_scores": _record(score_path),
        "seed_applications": {
            str(seed): {
                key: _record(verified_artifact_path(applications[str(seed)][key], name=f"D1 R5 {seed} {key}"))
                for key in ("cell", "model", "preprocessor")
            }
            for seed in SEEDS
        },
        "producer_code": _record(Path(__file__)),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "selected_method": "R5",
        "selected_trial_id": ranker.get("selected_trial_id"),
        "seeds": list(SEEDS),
        "candidate_test_labels_read": False,
        "candidate_count": len(canonical_keys),
        "model_feature_columns": list(model_columns),
        "contribution_feature_names": [*model_columns, BIAS_FEATURE_NAME],
        "contribution_row_count": len(frame),
        "maximum_absolute_additivity_residual": float(
            np.max(np.abs(contribution_sums - expected_ensemble))
        ),
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": {"candidate_contributions": _record(artifact_path)},
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "d1_r5_candidate_contributions",
            "output_manifest": str(manifest_path.resolve()),
            "output_manifest_sha256": sha256_file(manifest_path),
            "candidate_labels_opened_as_table": False,
            "resumed": False,
        },
    )
    return result
