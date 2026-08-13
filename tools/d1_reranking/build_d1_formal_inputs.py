"""Build label-free D1 Test rankers/gates and five normalized formal inputs."""

from __future__ import annotations

# ruff: noqa: E402 -- native thread limits must be set before numeric imports

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

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

# macOS can crash if LightGBM's OpenMP runtime is initialized after Torch.
# Match the audited unified Test wrapper: initialize LightGBM first, while
# keeping it optional for neural-only environments.
try:
    import lightgbm as _lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - neural-only installations
    _lightgbm = None

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.candidates import verify_canonical_candidate_frame  # noqa: E402
from d1_reranking.contracts import (  # noqa: E402
    CANDIDATE_REQUIRED_COLUMNS,
    assert_label_free_parquet_schema,
)
from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.fold_calibration import apply_partition_calibrator  # noqa: E402
from d1_reranking.gate_inputs import build_label_free_test_gate_input_frame  # noqa: E402
from d1_reranking.gate_validation import validate_gate_selection_semantics  # noqa: E402
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.k_replay import validate_k_selection  # noqa: E402
from d1_reranking.models import D1ResidualMLPScorer  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from d1_reranking.selection import ensemble_seed_scores  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.datasets import (  # noqa: E402
    FoldPreprocessor,
    build_inference_query_arrays,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.metrics import rank_by_score  # noqa: E402
from unified_reranking.telemetry import (  # noqa: E402
    lightgbm_parameter_count,
    torch_parameter_count,
)
from unified_reranking.test_access_guard import append_access_log  # noqa: E402
from unified_reranking.training import (  # noqa: E402
    predict_neural_ranker,
    set_deterministic_cpu,
)
from tools.d1_reranking.apply_locked_test_gate import _apply  # noqa: E402
from tools.unified_reranking.apply_locked_matrix_cell import (  # noqa: E402
    _lambdamart_predictions,
    _load_native_lightgbm_ranker,
)


K_SCENARIOS = {
    "top10_t2": {"pool": "top10", "ranker_name": "d1_top10"},
    "allnms_t2": {"pool": "allnms", "ranker_name": "d1_allnms"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _resume_manifest(
    path: Path,
    *,
    name: str,
    resume: bool,
    sources: Mapping[str, Any],
    required: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an exact, intact prior result without touching its outputs."""

    if not path.exists():
        return None
    existing = load_content_manifest(path, name=name, statuses=("COMPLETE",))
    if not resume:
        raise FileExistsError(f"{name} already exists: {path}")
    if (
        existing.get("sources") != sources
        or existing.get("source_signature_sha256") != canonical_sha256(sources)
        or any(existing.get(key) != value for key, value in required.items())
    ):
        raise RuntimeError(f"{name} resume source/contract differs")
    verify_artifact_records_recursive(
        {"sources": existing.get("sources"), "artifacts": existing.get("artifacts")},
        name=f"{name} resume closure",
        require_at_least_one=True,
    )
    return existing


def _seed_predictions(
    *,
    cell_path: Path,
    cell: Mapping[str, Any],
    features: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[int, pd.DataFrame, dict[str, Any]]:
    configuration = _mapping(cell.get("configuration"), name="D1 K Test cell config")
    if configuration.get("mode") != "validation":
        raise RuntimeError("D1 K Test inference requires Validation cell")
    seed = int(configuration["seed"])
    preprocessor_path = verified_artifact_path(
        _mapping(
            _mapping(cell.get("artifacts"), name="D1 K Test cell artifacts").get(
                "preprocessor"
            ),
            name="D1 K Test preprocessor",
        ),
        name="D1 K Test preprocessor",
    )
    persisted = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    if persisted != cell.get("preprocessor"):
        raise RuntimeError("D1 K Test persisted/embedded preprocessor differs")
    calibrator = _mapping(
        persisted.get("fold_calibrator"), name="D1 K Test fold calibrator"
    )
    calibrated = apply_partition_calibrator(features, calibrator)
    preprocessor = FoldPreprocessor.from_artifact(
        _mapping(persisted.get("fold_preprocessor"), name="D1 K Test fold preprocessor")
    )
    model_contract = _mapping(
        cell.get("model_contract"), name="D1 K Test model contract"
    )
    if canonical_sha256(list(preprocessor.columns)) != model_contract.get(
        "feature_schema_sha256"
    ):
        raise RuntimeError("D1 K Test model/preprocessor feature schema differs")
    arrays = build_inference_query_arrays(
        calibrated,
        preprocessor=preprocessor,
        max_candidates=int(configuration["max_candidates"]),
    )
    model_path = verified_artifact_path(
        _mapping(
            _mapping(cell.get("artifacts"), name="D1 K Test cell artifacts").get(
                "model"
            ),
            name="D1 K Test model",
        ),
        name="D1 K Test model",
    )
    started = time.perf_counter()
    if configuration["method"] == "R5":
        if model_contract.get("serialization") != "lightgbm_native_text":
            raise RuntimeError(
                "D1 K R5 cell does not declare native text serialization"
            )
        if model_path.suffix != ".txt":
            raise RuntimeError("D1 K R5 Test inference requires native .txt model")
        model = _load_native_lightgbm_ranker(model_path)
        predictions = _lambdamart_predictions(model, arrays)
        parameter_count = lightgbm_parameter_count(model)
    else:
        if model_contract.get("serialization") != "torch_state_dict":
            raise RuntimeError(
                "D1 K neural cell does not declare state-dict serialization"
            )
        set_deterministic_cpu(seed)
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        training = _mapping(
            model_contract.get("effective_training_hyperparameters"),
            name="D1 K Test neural hyperparameters",
        )
        model = D1ResidualMLPScorer(
            int(payload["input_dim"]),
            hidden_dims=tuple(map(int, training["hidden_dims"])),
            dropout=float(training["dropout"]),
            alpha=float(training["alpha"]),
        )
        model.load_state_dict(payload["state_dict"])
        predictions = predict_neural_ranker(model, arrays)
        parameter_count = torch_parameter_count(model)
    latency = (time.perf_counter() - started) * 1000.0 / max(len(predictions), 1)
    identity = candidates.loc[
        :,
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    ]
    predictions = identity.merge(
        predictions,
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if (
        len(predictions) != len(identity)
        or not np.isfinite(predictions["score"].to_numpy(float)).all()
    ):
        raise RuntimeError("D1 K Test predictions do not exactly cover candidates")
    return (
        seed,
        predictions,
        {
            "cell": artifact_record(cell_path),
            "model": artifact_record(model_path),
            "preprocessor": artifact_record(preprocessor_path),
            "parameter_count": parameter_count,
            "ranker_latency_ms": latency,
            "safe_model_loader": "lightgbm_native_text_or_torch_weights_only",
        },
    )


def _ranker_decisions(
    denominator: pd.DataFrame, candidates: pd.DataFrame, ensemble: pd.DataFrame
) -> pd.DataFrame:
    ranked = rank_by_score(ensemble, score_column="ensemble_score")
    native = candidates.loc[
        candidates["native_rank"].eq(1),
        [
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    ].rename(
        columns={
            "candidate_id": "native_candidate_id",
            "candidate_identity_sha256": "native_identity_sha256",
            "candidate_geometry_sha256": "native_geometry_sha256",
        }
    )
    rows: list[dict[str, Any]] = []
    for sample_id, group in ranked.groupby("sample_id", sort=False):
        ordered = group.sort_values("rerank_rank", kind="mergesort")
        top = ordered.iloc[0]
        second = (
            float(ordered.iloc[1]["ensemble_score"])
            if len(ordered) > 1
            else float(top["ensemble_score"])
        )
        votes = sum(
            str(
                rank_by_score(group, score_column=f"score_seed_{seed}").iloc[0][
                    "candidate_id"
                ]
            )
            == str(top["candidate_id"])
            for seed in (42, 123, 2026)
        )
        rows.append(
            {
                "sample_id": str(sample_id),
                "selected_candidate_id": str(top["candidate_id"]),
                "selected_identity_sha256": str(top["candidate_identity_sha256"]),
                "selected_geometry_sha256": str(top["candidate_geometry_sha256"]),
                "ensemble_score": float(top["ensemble_score"]),
                "ensemble_score_margin": float(top["ensemble_score"]) - second,
                "seed_challenger_votes": int(votes),
                "candidate_count": len(ordered),
            }
        )
    selected = pd.DataFrame(rows)
    result = (
        denominator[["sample_id"]]
        .merge(native, on="sample_id", how="left", validate="one_to_one")
        .merge(selected, on="sample_id", how="left", validate="one_to_one")
    )
    result["candidate_count"] = result["candidate_count"].fillna(0).astype(int)
    result["challenger_exists"] = (
        result["selected_candidate_id"].notna()
        & result["native_candidate_id"].notna()
        & result["selected_candidate_id"]
        .astype(str)
        .ne(result["native_candidate_id"].astype(str))
    )
    result["prediction_source"] = "test_label_free"
    return result


def _build_k_ranker(
    root: Path,
    *,
    scenario_id: str,
    pool: str,
    ranker_name: str,
    selection: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    scenario = _mapping(
        _mapping(selection.get("scenarios"), name="D1 K scenarios").get(scenario_id),
        name=f"D1 K {scenario_id}",
    )
    job_ids = list(map(str, scenario["cell_job_ids"]))
    selection_sources = _mapping(
        selection.get("sources"), name="D1 K selection sources"
    )
    cell_records = _mapping(selection_sources.get("cells"), name="D1 K cell records")
    validation_cells: list[tuple[Path, dict[str, Any]]] = []
    for job_id in job_ids:
        path = verified_artifact_path(
            _mapping(cell_records.get(job_id), name=f"D1 K cell {job_id}"),
            name=f"D1 K cell {job_id}",
        )
        cell = load_content_manifest(
            path, name=f"D1 K cell {job_id}", statuses=("COMPLETE",)
        )
        configuration = _mapping(cell.get("configuration"), name="D1 K cell config")
        if (
            configuration.get("scenario_id") == scenario_id
            and configuration.get("mode") == "validation"
        ):
            validation_cells.append((path, cell))
    if sorted(
        int(cell["configuration"]["seed"]) for _path, cell in validation_cells
    ) != [42, 123, 2026]:
        raise RuntimeError(f"D1 K {scenario_id} lacks exact Validation seeds")
    scenario_definition = _mapping(
        scenario.get("definition"), name=f"D1 K {scenario_id} definition"
    )
    for _path, cell in validation_cells:
        configuration = _mapping(
            cell.get("configuration"), name="D1 K Validation cell configuration"
        )
        model_contract = _mapping(
            cell.get("model_contract"), name="D1 K Validation cell model contract"
        )
        if (
            configuration.get("method") != scenario.get("method")
            or configuration.get("pool") != pool
            or configuration.get("track") != "T2_matched_common"
            or configuration.get("max_candidates")
            != scenario_definition.get("max_candidates")
            or configuration.get("selected_primary_trial_id")
            != scenario.get("selected_primary_trial_id")
            or model_contract.get("effective_training_hyperparameters") is None
            or (
                configuration.get("method") == "R5"
                and model_contract.get("serialization") != "lightgbm_native_text"
            )
        ):
            raise RuntimeError(f"D1 K {scenario_id} frozen cell contract differs")

    candidate_manifest_path = root / "02_candidates/test_manifest.json"
    candidate_manifest = load_content_manifest(
        candidate_manifest_path, name="D1 Test candidates", statuses=("COMPLETE",)
    )
    candidate_path = verified_artifact_path(
        _mapping(
            _mapping(
                candidate_manifest.get("artifacts"), name="D1 Test candidate artifacts"
            ).get(pool),
            name=f"D1 Test {pool} candidates",
        ),
        name=f"D1 Test {pool} candidates",
    )
    feature_manifest_path = (
        root / f"03_features/test/{pool}/T2_matched_common/manifest.json"
    )
    feature_manifest = load_content_manifest(
        feature_manifest_path, name=f"D1 Test {pool} T2", statuses=("COMPLETE",)
    )
    feature_path = verified_artifact_path(
        _mapping(
            _mapping(
                feature_manifest.get("artifacts"), name="D1 Test feature artifacts"
            ).get("candidate_features"),
            name="D1 Test feature table",
        ),
        name=f"D1 Test {pool} feature table",
    )
    paired_path = root / "01_manifests/d1_paired_manifest.parquet"
    for source, name in (
        (candidate_path, f"D1 Test {pool} candidates"),
        (feature_path, f"D1 Test {pool} features"),
        (paired_path, "D1 Test denominator"),
    ):
        assert_label_free_parquet_schema(source, name=name)
    sources = {
        "k_selection": artifact_record(
            root / "11_k_sensitivity/selection_manifest.json"
        ),
        "scenario_gate_selection": artifact_record(
            root / "11_k_sensitivity/scenario_gate_selection.json"
        ),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "feature_manifest": artifact_record(feature_manifest_path),
        "candidate_features": artifact_record(feature_path),
        "denominator": artifact_record(paired_path),
        "validation_cells": {
            str(cell["configuration"]["seed"]): artifact_record(path)
            for path, cell in validation_cells
        },
        "inference_code": [
            artifact_record(path)
            for path in (
                ROOT / "src/d1_reranking/fold_calibration.py",
                ROOT / "src/d1_reranking/models.py",
                ROOT / "src/d1_reranking/selection.py",
                ROOT / "src/unified_reranking/datasets.py",
                ROOT / "src/unified_reranking/training.py",
                Path(__file__),
            )
        ],
    }
    output = root / "08_lock/label_free_test_rankers" / ranker_name
    manifest_path = output / "manifest.json"
    existing = _resume_manifest(
        manifest_path,
        name=f"D1 {ranker_name} ranker",
        resume=resume,
        sources=sources,
        required={
            "route": "D1",
            "scenario_id": scenario_id,
            "pool": pool,
            "track": "T2_matched_common",
            "selected_method": scenario["method"],
            "selected_primary_trial_id": scenario["selected_primary_trial_id"],
            "seed_policy": "fixed mean score ensemble over 42,123,2026",
            "candidate_test_labels_read": False,
            "test_inputs_referenced": True,
        },
    )
    if existing is not None:
        return existing

    candidates = pd.read_parquet(
        candidate_path,
        columns=[
            *CANDIDATE_REQUIRED_COLUMNS,
            "candidate_identity_precision_contract",
        ],
    )
    verify_canonical_candidate_frame(candidates, split="test")
    features = pd.read_parquet(feature_path)
    candidate_keys = set(
        map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    feature_keys = set(
        map(tuple, features[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    if candidate_keys != feature_keys or len(candidates) != len(features):
        raise RuntimeError(f"D1 Test {pool} feature/candidate membership differs")
    denominator = pd.read_parquet(paired_path, columns=["sample_id", "scene_id"])
    seed_frames: dict[int, pd.DataFrame] = {}
    applications: dict[str, Any] = {}
    for cell_path, cell in validation_cells:
        seed, predictions, application = _seed_predictions(
            cell_path=cell_path,
            cell=cell,
            features=features,
            candidates=candidates,
        )
        seed_frames[seed] = predictions
        applications[str(seed)] = application
    ensemble = ensemble_seed_scores(seed_frames)
    decisions = _ranker_decisions(denominator, candidates, ensemble)
    expected_counts = (
        denominator["sample_id"]
        .astype(str)
        .map(candidates.groupby("sample_id").size())
        .fillna(0)
        .astype(int)
    )
    if (
        not decisions["candidate_count"]
        .reset_index(drop=True)
        .equals(expected_counts.reset_index(drop=True))
    ):
        raise RuntimeError(f"D1 Test {pool} ranker denominator/counts differ")
    artifacts = {
        "per_candidate_scores": artifact_record(
            atomic_parquet(ensemble, output / "per_candidate_scores.parquet")
        ),
        "per_sample_decisions": artifact_record(
            atomic_parquet(decisions, output / "per_sample_decisions.parquet")
        ),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "route": "D1",
        "scenario_id": scenario_id,
        "pool": pool,
        "track": "T2_matched_common",
        "selected_method": scenario["method"],
        "selected_primary_trial_id": scenario["selected_primary_trial_id"],
        "seed_policy": "fixed mean score ensemble over 42,123,2026",
        "validation_cell_job_ids": sorted(
            str(cell["job_id"]) for _path, cell in validation_cells
        ),
        "seed_applications": applications,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "d1_k_scenario_test_ranker",
            "scenario_id": scenario_id,
            "pool": pool,
            "output_manifest": str(manifest_path),
            "output_manifest_sha256": sha256_file(manifest_path),
            "candidate_labels_opened_as_table": False,
        },
    )
    return result


def _build_k_gate(
    root: Path,
    *,
    scenario_id: str,
    pool: str,
    ranker_name: str,
    ranker: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    gate_path = root / f"11_k_sensitivity/gates/{scenario_id}/gate_selection.json"
    gate = load_content_manifest(
        gate_path, name=f"D1 K {scenario_id} gate", statuses=("COMPLETE",)
    )
    validated = validate_gate_selection_semantics(gate)
    ranker_path = root / f"08_lock/label_free_test_rankers/{ranker_name}/manifest.json"
    decision_path = verified_artifact_path(
        _mapping(
            _mapping(ranker.get("artifacts"), name="D1 K ranker artifacts").get(
                "per_sample_decisions"
            ),
            name="D1 K ranker decisions",
        ),
        name="D1 K ranker decisions",
    )
    feature_manifest_path = (
        root / f"03_features/test/{pool}/T2_matched_common/manifest.json"
    )
    feature_manifest = load_content_manifest(
        feature_manifest_path, name="D1 K Test T2", statuses=("COMPLETE",)
    )
    feature_path = verified_artifact_path(
        _mapping(
            _mapping(
                feature_manifest.get("artifacts"), name="K Test feature artifacts"
            ).get("candidate_features"),
            name="K Test features",
        ),
        name="D1 K Test features",
    )
    candidate_manifest_path = root / "02_candidates/test_manifest.json"
    candidate_manifest = load_content_manifest(
        candidate_manifest_path, name="D1 Test candidates", statuses=("COMPLETE",)
    )
    candidate_path = verified_artifact_path(
        _mapping(
            _mapping(
                candidate_manifest.get("artifacts"), name="D1 candidate artifacts"
            ).get(pool),
            name=f"D1 {pool}",
        ),
        name=f"D1 Test {pool}",
    )
    paired_path = root / "01_manifests/d1_paired_manifest.parquet"
    sources = {
        "gate_selection": artifact_record(gate_path),
        "transition_model": artifact_record(validated.transition_model_path),
        "ranker_application": artifact_record(ranker_path),
        "ranker_decisions": artifact_record(decision_path),
        "feature_manifest": artifact_record(feature_manifest_path),
        "candidate_features": artifact_record(feature_path),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "denominator": artifact_record(paired_path),
        "inference_code": artifact_record(Path(__file__)),
    }
    output = root / "08_lock/label_free_test_gates" / ranker_name
    manifest_path = output / "manifest.json"
    existing = _resume_manifest(
        manifest_path,
        name=f"D1 {ranker_name} gate",
        resume=resume,
        sources=sources,
        required={
            "route": "D1",
            "scenario_id": scenario_id,
            "pool": pool,
            "decision": gate["decision"],
            "prediction_source": "test_label_free",
            "candidate_test_labels_read": False,
            "test_inputs_referenced": True,
        },
    )
    if existing is not None:
        return existing

    for source, name in (
        (paired_path, "D1 Test denominator"),
        (decision_path, f"D1 Test {ranker_name} ranker decisions"),
        (feature_path, f"D1 Test {pool} T2 features"),
        (candidate_path, f"D1 Test {pool} candidates"),
    ):
        assert_label_free_parquet_schema(source, name=name)
    inputs = build_label_free_test_gate_input_frame(
        paired=pd.read_parquet(paired_path, columns=["sample_id", "scene_id"]),
        ranker_decisions=pd.read_parquet(decision_path),
        candidate_features=pd.read_parquet(feature_path),
        candidates=pd.read_parquet(candidate_path),
    )
    decisions, point = _apply(gate, validated.model, inputs)
    artifacts = {
        "inputs": artifact_record(
            atomic_parquet(inputs, output / "gate_test_inputs.parquet")
        ),
        "decisions": artifact_record(
            atomic_parquet(decisions, output / "gate_test_decisions.parquet")
        ),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "route": "D1",
        "scenario_id": scenario_id,
        "pool": pool,
        "decision": gate["decision"],
        "selected_operating_point": None if point is None else asdict(point),
        "sample_count": len(inputs),
        "switch_count": int(decisions["switch"].sum()),
        "prediction_source": "test_label_free",
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "d1_k_scenario_test_gate",
            "scenario_id": scenario_id,
            "pool": pool,
            "output_manifest": str(manifest_path),
            "output_manifest_sha256": sha256_file(manifest_path),
            "candidate_labels_opened_as_table": False,
        },
    )
    return result


def _candidate_universe(candidates: pd.DataFrame) -> pd.DataFrame:
    required = [
        "route",
        "sample_id",
        "candidate_id",
        "candidate_geometry_sha256",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ]
    missing = sorted(set(required).difference(candidates.columns))
    if missing:
        raise RuntimeError(f"D1 formal candidate universe misses {missing}")
    universe = (
        candidates.loc[:, required].rename(columns={"route": "source_route"}).copy()
    )
    numeric = universe[
        [
            "native_rank",
            "native_score",
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        universe.duplicated(["source_route", "sample_id", "candidate_id"]).any()
        or not universe["candidate_geometry_sha256"]
        .astype(str)
        .str.fullmatch(r"[0-9a-f]{64}")
        .all()
        or not np.isfinite(numeric.to_numpy(float)).all()
        or (numeric[["width_px", "height_px"]] <= 0).any().any()
    ):
        raise RuntimeError("D1 formal candidate universe contains duplicates")
    return universe


def _normalized_scores(
    candidates: pd.DataFrame, scores: pd.DataFrame | None
) -> pd.DataFrame:
    if scores is None:
        result = candidates[
            ["route", "sample_id", "candidate_id", "native_score", "native_rank"]
        ].rename(
            columns={
                "route": "source_route",
                "native_score": "score",
                "native_rank": "rank",
            }
        )
        return result
    candidate_keys = set(
        map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    score_keys = set(
        map(tuple, scores[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    if (
        score_keys != candidate_keys
        or len(scores) != len(candidates)
        or not np.isfinite(scores["ensemble_score"].to_numpy(float)).all()
    ):
        raise RuntimeError(
            "D1 formal ranker scores do not cover the candidate universe"
        )
    ranked = rank_by_score(scores, score_column="ensemble_score")
    return (
        ranked[["sample_id", "candidate_id", "ensemble_score", "rerank_rank"]]
        .assign(source_route="D1")
        .rename(columns={"ensemble_score": "score", "rerank_rank": "rank"})[
            ["source_route", "sample_id", "candidate_id", "score", "rank"]
        ]
    )


def _normalized_decisions(
    denominator: pd.DataFrame,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame | None,
    *,
    selected_column: str,
    geometry_column: str | None,
) -> pd.DataFrame:
    if decisions is None:
        selected = candidates.loc[
            candidates["native_rank"].eq(1),
            ["sample_id", "candidate_id", "candidate_geometry_sha256"],
        ].rename(
            columns={
                "candidate_id": "selected_candidate_id",
                "candidate_geometry_sha256": "selected_geometry_sha256",
            }
        )
    else:
        selected = decisions[
            [
                "sample_id",
                selected_column,
                *([geometry_column] if geometry_column else []),
            ]
        ].rename(
            columns={
                selected_column: "selected_candidate_id",
                **(
                    {geometry_column: "selected_geometry_sha256"}
                    if geometry_column
                    else {}
                ),
            }
        )
    result = denominator[["sample_id"]].merge(
        selected, on="sample_id", how="left", validate="one_to_one"
    )
    if "selected_geometry_sha256" not in result:
        geometry = candidates[
            ["sample_id", "candidate_id", "candidate_geometry_sha256"]
        ].rename(
            columns={
                "candidate_id": "selected_candidate_id",
                "candidate_geometry_sha256": "selected_geometry_sha256",
            }
        )
        result = result.merge(
            geometry,
            on=["sample_id", "selected_candidate_id"],
            how="left",
            validate="one_to_one",
        )
    lookup = candidates[
        ["sample_id", "candidate_id", "candidate_geometry_sha256"]
    ].rename(
        columns={
            "candidate_id": "selected_candidate_id",
            "candidate_geometry_sha256": "locked_geometry",
        }
    )
    checked = result.merge(
        lookup,
        on=["sample_id", "selected_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    bearing = checked["selected_candidate_id"].notna() & checked[
        "selected_candidate_id"
    ].astype(str).ne("")
    if checked.loc[bearing, "locked_geometry"].isna().any() or not checked.loc[
        bearing, "selected_geometry_sha256"
    ].astype(str).equals(checked.loc[bearing, "locked_geometry"].astype(str)):
        raise RuntimeError(
            "D1 formal selected ID/geometry differs from candidate universe"
        )
    return pd.DataFrame(
        {
            "sample_id": checked["sample_id"].astype(str),
            "selected_source_route": np.where(bearing, "D1", ""),
            "selected_candidate_id": checked["selected_candidate_id"]
            .fillna("")
            .astype(str),
            "selected_geometry_sha256": checked["selected_geometry_sha256"]
            .fillna("")
            .astype(str),
        }
    )


def _write_normalized(
    root: Path,
    *,
    name: str,
    pool: str,
    kind: str,
    source_manifests: Mapping[str, Any],
    score_record: Mapping[str, Any] | None,
    decision_record: Mapping[str, Any] | None,
    decision_key: str,
    geometry_key: str | None,
    resume: bool,
) -> dict[str, Any]:
    candidate_manifest_path = root / "02_candidates/test_manifest.json"
    candidate_manifest = load_content_manifest(
        candidate_manifest_path, name="D1 Test candidates", statuses=("COMPLETE",)
    )
    candidate_path = verified_artifact_path(
        _mapping(
            _mapping(
                candidate_manifest.get("artifacts"), name="D1 candidate artifacts"
            ).get(pool),
            name=f"D1 {pool}",
        ),
        name=f"D1 Test {pool}",
    )
    paired_path = root / "01_manifests/d1_paired_manifest.parquet"
    sources = {
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "denominator": artifact_record(paired_path),
        **dict(source_manifests),
        "normalizer": artifact_record(Path(__file__)),
    }
    output = root / "08_lock/formal_inputs" / name
    manifest_path = output / "manifest.json"
    existing = _resume_manifest(
        manifest_path,
        name=f"D1 formal {name}",
        resume=resume,
        sources=sources,
        required={
            "name": name,
            "kind": kind,
            "pool": pool,
            "routes": ["D1"],
            "candidate_test_labels_read": False,
            "test_inputs_referenced": True,
        },
    )
    if existing is not None:
        return existing

    score_path = (
        None
        if score_record is None
        else verified_artifact_path(
            _mapping(score_record, name=f"{name} score record"),
            name=f"{name} scores",
        )
    )
    decision_path = (
        None
        if decision_record is None
        else verified_artifact_path(
            _mapping(decision_record, name=f"{name} decision record"),
            name=f"{name} decisions",
        )
    )
    label_free_sources: list[tuple[Path, str]] = [
        (candidate_path, f"D1 Test {pool} candidates"),
        (paired_path, "D1 Test denominator"),
    ]
    if score_path is not None:
        label_free_sources.append((score_path, f"D1 Test {name} scores"))
    if decision_path is not None:
        label_free_sources.append((decision_path, f"D1 Test {name} decisions"))
    for source, artifact_name in label_free_sources:
        assert_label_free_parquet_schema(source, name=artifact_name)
    candidates = pd.read_parquet(candidate_path)
    denominator = pd.read_parquet(paired_path, columns=["sample_id"])
    universe = _candidate_universe(candidates)
    score_frame = None if score_path is None else pd.read_parquet(score_path)
    decision_frame = None if decision_path is None else pd.read_parquet(decision_path)
    scores = _normalized_scores(candidates, score_frame)
    decisions = _normalized_decisions(
        denominator,
        candidates,
        decision_frame,
        selected_column=decision_key,
        geometry_column=geometry_key,
    )
    if len(decisions) != len(denominator) or set(decisions["sample_id"]) != set(
        denominator["sample_id"].astype(str)
    ):
        raise RuntimeError(f"D1 formal {name} denominator differs")
    artifacts = {
        "candidate_universe": artifact_record(
            atomic_parquet(universe, output / "candidate_universe.parquet")
        ),
        "candidate_scores": artifact_record(
            atomic_parquet(scores, output / "candidate_scores.parquet")
        ),
        "per_sample_decisions": artifact_record(
            atomic_parquet(decisions, output / "per_sample_decisions.parquet")
        ),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "name": name,
        "kind": kind,
        "pool": pool,
        "routes": ["D1"],
        "candidate_test_labels_read": False,
        "test_inputs_referenced": True,
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    return result


def run(run_dir: Path, *, resume: bool) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    # Complete the development-only semantic replay before opening any Test
    # manifest, schema, or row.  This preserves the same fail-closed ordering as
    # the primary Top5 Test wrapper.
    validate_k_selection(root)
    selection_path = root / "11_k_sensitivity/selection_manifest.json"
    selection = load_content_manifest(
        selection_path, name="D1 K selection", statuses=("COMPLETE",)
    )
    scenario_gate_path = root / "11_k_sensitivity/scenario_gate_selection.json"
    scenario_gate = load_content_manifest(
        scenario_gate_path, name="D1 K scenario gates", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        scenario_gate,
        name="D1 K scenario gate aggregate",
        require_at_least_one=True,
    )
    gate_sources = _mapping(
        scenario_gate.get("sources"), name="D1 K scenario gate sources"
    )
    declared_gates = _mapping(
        gate_sources.get("gates"), name="D1 K scenario gate records"
    )
    if (
        scenario_gate.get("allnms_t3_sensitivity_only") is not True
        or gate_sources.get("k_selection") != artifact_record(selection_path)
        or set(declared_gates) != set(K_SCENARIOS)
    ):
        raise RuntimeError("D1 K scenario gate aggregate contract differs")
    for scenario_id in K_SCENARIOS:
        gate_path = root / f"11_k_sensitivity/gates/{scenario_id}/gate_selection.json"
        gate = load_content_manifest(
            gate_path, name=f"D1 K {scenario_id} gate", statuses=("COMPLETE",)
        )
        validate_gate_selection_semantics(gate)
        if _mapping(
            declared_gates.get(scenario_id), name=f"D1 K {scenario_id} gate records"
        ).get("selection") != artifact_record(gate_path):
            raise RuntimeError(f"D1 K {scenario_id} aggregate binding differs")
    rankers: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for scenario_id, definition in K_SCENARIOS.items():
        ranker = _build_k_ranker(
            root,
            scenario_id=scenario_id,
            pool=definition["pool"],
            ranker_name=definition["ranker_name"],
            selection=selection,
            resume=resume,
        )
        gate = _build_k_gate(
            root,
            scenario_id=scenario_id,
            pool=definition["pool"],
            ranker_name=definition["ranker_name"],
            ranker=ranker,
            resume=resume,
        )
        rankers[definition["ranker_name"]] = ranker
        gates[definition["ranker_name"]] = gate

    primary_gate_path = root / "07_validation/gate/d1/gate_selection.json"
    primary_gate = load_content_manifest(
        primary_gate_path, name="D1 Top5 development gate", statuses=("COMPLETE",)
    )
    validate_gate_selection_semantics(primary_gate)
    top5_ranker_path = root / "08_lock/label_free_test_rankers/d1/manifest.json"
    top5_gate_path = root / "08_lock/label_free_test_gates/d1/manifest.json"
    top5_ranker = load_content_manifest(
        top5_ranker_path, name="D1 Top5 ranker", statuses=("COMPLETE",)
    )
    top5_gate = load_content_manifest(
        top5_gate_path, name="D1 Top5 gate", statuses=("COMPLETE",)
    )
    for value, name in ((top5_ranker, "ranker"), (top5_gate, "gate")):
        if value.get("candidate_test_labels_read") is not False:
            raise RuntimeError(f"D1 Top5 {name} violates Test-label isolation")
        verify_artifact_records_recursive(
            value, name=f"D1 Top5 {name}", require_at_least_one=True
        )
    top5_ranker_artifacts = _mapping(
        top5_ranker.get("artifacts"), name="D1 Top5 ranker artifacts"
    )
    top5_gate_artifacts = _mapping(
        top5_gate.get("artifacts"), name="D1 Top5 gate artifacts"
    )
    normalized: dict[str, Any] = {}
    normalized["d1_top5_r0"] = _write_normalized(
        root,
        name="d1_top5_r0",
        pool="top5",
        kind="R0",
        source_manifests={},
        score_record=None,
        decision_record=None,
        decision_key="selected_candidate_id",
        geometry_key=None,
        resume=resume,
    )
    normalized["d1_top5_r7_ungated"] = _write_normalized(
        root,
        name="d1_top5_r7_ungated",
        pool="top5",
        kind="R7_UNGATED",
        source_manifests={
            "feature_manifest": artifact_record(
                root / "03_features/test/top5/T2_matched_common/manifest.json"
            ),
            "ranker_manifest": artifact_record(top5_ranker_path),
        },
        score_record=_mapping(
            top5_ranker_artifacts.get("per_candidate_scores"), name="Top5 scores"
        ),
        decision_record=_mapping(
            top5_ranker_artifacts.get("per_sample_decisions"), name="Top5 decisions"
        ),
        decision_key="selected_candidate_id",
        geometry_key="selected_geometry_sha256",
        resume=resume,
    )
    normalized["d1_top5_r7_gated"] = _write_normalized(
        root,
        name="d1_top5_r7_gated",
        pool="top5",
        kind="R7_GATED",
        source_manifests={
            "feature_manifest": artifact_record(
                root / "03_features/test/top5/T2_matched_common/manifest.json"
            ),
            "ranker_manifest": artifact_record(top5_ranker_path),
            "gate_manifest": artifact_record(top5_gate_path),
        },
        score_record=_mapping(
            top5_ranker_artifacts.get("per_candidate_scores"), name="Top5 scores"
        ),
        decision_record=_mapping(
            top5_gate_artifacts.get("decisions"), name="Top5 gate decisions"
        ),
        decision_key="selected_candidate_id",
        geometry_key="selected_geometry_sha256",
        resume=resume,
    )
    for scenario_id, definition in K_SCENARIOS.items():
        name = f"{definition['ranker_name']}_locked"
        ranker = rankers[definition["ranker_name"]]
        gate = gates[definition["ranker_name"]]
        normalized[name] = _write_normalized(
            root,
            name=name,
            pool=definition["pool"],
            kind="K_LOCKED",
            source_manifests={
                "k_selection": artifact_record(selection_path),
                "scenario_gate_selection": artifact_record(scenario_gate_path),
                "feature_manifest": artifact_record(
                    root
                    / f"03_features/test/{definition['pool']}/T2_matched_common/manifest.json"
                ),
                "ranker_manifest": artifact_record(
                    root
                    / f"08_lock/label_free_test_rankers/{definition['ranker_name']}/manifest.json"
                ),
                "gate_manifest": artifact_record(
                    root
                    / f"08_lock/label_free_test_gates/{definition['ranker_name']}/manifest.json"
                ),
            },
            score_record=_mapping(
                _mapping(ranker.get("artifacts"), name="K ranker artifacts").get(
                    "per_candidate_scores"
                ),
                name="K scores",
            ),
            decision_record=_mapping(
                _mapping(gate.get("artifacts"), name="K gate artifacts").get(
                    "decisions"
                ),
                name="K decisions",
            ),
            decision_key="selected_candidate_id",
            geometry_key="selected_geometry_sha256",
            resume=resume,
        )
    normalized_records = {
        name: artifact_record(root / f"08_lock/formal_inputs/{name}/manifest.json")
        for name in sorted(normalized)
    }
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "event_id": canonical_sha256(
                {
                    "stage": "d1_formal_input_normalization",
                    "outputs": normalized_records,
                }
            )[:24],
            "stage": "d1_formal_input_normalization",
            "output_manifests": normalized_records,
            "candidate_labels_opened_as_table": False,
            "candidate_test_labels_read": False,
            "resumed": resume,
        },
    )
    return {
        "rankers": rankers,
        "gates": gates,
        "normalized": normalized,
    }


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_k_test_and_normalized_formal_inputs",
        route="D1",
        pool="top5_top10_allnms",
        evidence_track="T2_matched_common",
        method="R0_R7_K_locked",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(root, resume=args.resume)
        output = root / "08_lock/formal_inputs/d1_allnms_locked/manifest.json"
        state["artifact_path"] = str(output)
        state["artifact_sha256"] = sha256_file(output)
        state["normalized_count"] = len(result["normalized"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
