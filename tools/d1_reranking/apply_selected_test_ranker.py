"""Apply the Validation-selected three-seed D1 ranker to label-free Test Top5."""

from __future__ import annotations

# ruff: noqa: E402 -- native thread limits must be set before numeric imports

import argparse
import json
import os
import sys
import time
from pathlib import Path

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

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.contracts import (  # noqa: E402
    CANDIDATE_REQUIRED_COLUMNS,
    assert_label_free_parquet_schema,
)
from d1_reranking.candidates import verify_canonical_candidate_frame  # noqa: E402
from d1_reranking.fold_calibration import apply_partition_calibrator  # noqa: E402
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.models import D1ResidualMLPScorer  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from d1_reranking.selection import ensemble_seed_scores, trial_configuration  # noqa: E402
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
    flatten_telemetry,
    lightgbm_parameter_count,
    missing_feature_rate,
    telemetry_payload,
    torch_parameter_count,
)
from unified_reranking.test_access_guard import append_access_log  # noqa: E402
from unified_reranking.training import predict_neural_ranker, set_deterministic_cpu  # noqa: E402
from tools.unified_reranking.apply_locked_matrix_cell import (  # noqa: E402
    _lambdamart_predictions,
    _load_native_lightgbm_ranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _same_artifact_record(
    observed: object, expected: dict[str, object], *, name: str
) -> None:
    """Compare artifact identity while allowing producer-added metadata."""

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


def _raw_feature_columns(
    model_columns: tuple[str, ...], schema_columns: tuple[str, ...]
) -> tuple[str, ...]:
    """Select raw evidence only; candidate identities come from the pool contract."""

    return tuple(
        dict.fromkeys(
            [
                "sample_id",
                "candidate_id",
                "native_rank",
                "native_score_raw",
                *[column for column in model_columns if column in schema_columns],
            ]
        )
    )


def _normalized_candidate_rank_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize representation while preserving exact candidate/rank values."""

    keys = ["sample_id", "candidate_id"]
    contract = frame.loc[:, [*keys, "native_rank"]].copy()
    contract[keys] = contract[keys].astype(str)
    contract["native_rank"] = pd.to_numeric(
        contract["native_rank"], errors="raise"
    ).astype("int64")
    return contract.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _validation_cells(
    trial: dict[str, object],
) -> list[tuple[int, Path, dict[str, object]]]:
    records = trial.get("sources", {}).get("cells", [])  # type: ignore[union-attr]
    selected = []
    for record in records:
        path = verified_artifact_path(record, name="D1 selected trial cell")
        cell = load_content_manifest(
            path, name="D1 selected trial cell", statuses=("COMPLETE",)
        )
        configuration = cell.get("configuration", {})
        if configuration.get("mode") == "validation":
            selected.append((int(configuration["seed"]), path, cell))
    if [seed for seed, _path, _cell in sorted(selected)] != [42, 123, 2026]:
        raise RuntimeError("D1 selected trial lacks exactly three Validation cells")
    return sorted(selected)


def _predict_cell(
    *,
    cell: dict[str, object],
    cell_path: Path,
    raw_features: pd.DataFrame,
    model_columns: tuple[str, ...],
    candidates: pd.DataFrame,
) -> tuple[int, pd.DataFrame, dict[str, object]]:
    verify_artifact_records_recursive(
        {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
        name="D1 selected Test cell",
        require_at_least_one=True,
    )
    configuration = cell["configuration"]
    seed = int(configuration["seed"])
    calibrated = apply_partition_calibrator(
        raw_features, configuration["fold_local_calibrator"]
    )
    calibrated_missing_feature_rate = missing_feature_rate(
        calibrated, model_columns
    )
    preprocessor_path = verified_artifact_path(
        cell.get("artifacts", {}).get("preprocessor", {}),
        name="D1 selected cell preprocessor",
    )
    persisted_preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    if persisted_preprocessor != cell.get("preprocessor"):
        raise RuntimeError("D1 selected cell embedded/persisted preprocessors differ")
    preprocessor = FoldPreprocessor.from_artifact(persisted_preprocessor)
    if preprocessor.columns != model_columns:
        raise RuntimeError("D1 selected cell/Test feature schemas differ")
    arrays = build_inference_query_arrays(
        calibrated, preprocessor=preprocessor, max_candidates=5
    )
    model_path = verified_artifact_path(
        cell.get("artifacts", {}).get("model", {}), name="D1 selected cell model"
    )
    started = time.perf_counter()
    if configuration["method"] == "R5":
        model = _load_native_lightgbm_ranker(model_path)
        predictions = _lambdamart_predictions(model, arrays)
        parameter_count = lightgbm_parameter_count(model)
    else:
        set_deterministic_cpu(seed)
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        model = D1ResidualMLPScorer(
            int(payload["input_dim"]),
            hidden_dims=tuple(map(int, payload["hidden_dims"])),
            dropout=float(payload["dropout"]),
            alpha=float(payload["alpha"]),
        )
        expected = configuration.get("planned_configuration", {})
        if (
            int(payload["input_dim"]) != len(model_columns)
            or tuple(map(int, payload["hidden_dims"]))
            != tuple(map(int, expected.get("hidden_dims", ())))
            or float(payload["dropout"]) != float(expected.get("dropout"))
            or float(payload["alpha"]) != float(expected.get("alpha"))
        ):
            raise RuntimeError("D1 selected neural checkpoint architecture differs")
        model.load_state_dict(payload["state_dict"])
        predictions = predict_neural_ranker(model, arrays)
        parameter_count = torch_parameter_count(model)
    latency = (time.perf_counter() - started) * 1000.0 / len(predictions)
    contract = candidates.loc[
        :,
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    ]
    prediction_keys = predictions.loc[:, ["sample_id", "candidate_id"]].astype(str)
    contract_keys = contract.loc[:, ["sample_id", "candidate_id"]].astype(str)
    if (
        predictions.duplicated(["sample_id", "candidate_id"]).any()
        or len(predictions) != len(contract)
        or set(map(tuple, prediction_keys.to_numpy()))
        != set(map(tuple, contract_keys.to_numpy()))
    ):
        raise RuntimeError("D1 Test cell prediction universe differs from Top5")
    predictions = contract.merge(
        predictions, on=["sample_id", "candidate_id"], how="left", validate="one_to_one"
    )
    if (
        len(predictions) != len(contract)
        or not np.isfinite(predictions["score"].to_numpy(float)).all()
    ):
        raise RuntimeError("D1 Test cell predictions do not exactly cover Top5")
    return (
        seed,
        predictions,
        {
            "cell": artifact_record(cell_path),
            "model": artifact_record(model_path),
            "preprocessor": cell.get("artifacts", {}).get("preprocessor"),
            "parameter_count": parameter_count,
            "ranker_latency_ms": latency,
            "missing_feature_rate": calibrated_missing_feature_rate,
        },
    )


def _decisions(
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
    rows = []
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
    selected_rows = pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "selected_candidate_id",
            "selected_identity_sha256",
            "selected_geometry_sha256",
            "ensemble_score",
            "ensemble_score_margin",
            "seed_challenger_votes",
            "candidate_count",
        ],
    )
    result = (
        denominator[["sample_id"]]
        .merge(native, on="sample_id", how="left", validate="one_to_one")
        .merge(selected_rows, on="sample_id", how="left", validate="one_to_one")
    )
    result["candidate_count"] = result["candidate_count"].fillna(0).astype(int)
    result["challenger_exists"] = (
        result["selected_candidate_id"].notna()
        & result["native_candidate_id"].notna()
        & (
            result["selected_candidate_id"].astype(str)
            != result["native_candidate_id"].astype(str)
        )
    )
    result["prediction_source"] = "test_label_free"
    return result


def run(run_dir: Path, *, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    selection_path = root / "07_validation" / "selected_primary_ungated.json"
    selection = load_content_manifest(
        selection_path, name="D1 selected primary", statuses=("COMPLETE",)
    )
    trial_path = verified_artifact_path(
        selection.get("artifacts", {}).get("selected_trial_manifest", {}),
        name="D1 selected trial manifest",
    )
    trial = load_content_manifest(
        trial_path, name="D1 selected trial", statuses=("COMPLETE",)
    )
    trial_id = str(trial.get("trial_id", ""))
    trial_config = trial.get("configuration")
    if (
        selection.get("artifacts", {}).get("selected_trial_manifest")
        != artifact_record(trial_path)
        or selection.get("selected_trial_id") != trial_id
        or selection.get("selected_method")
        != (trial_config.get("method") if isinstance(trial_config, dict) else None)
        or selection.get("selected_configuration") != trial_config
    ):
        raise RuntimeError("D1 selected primary/trial identity differs")
    if selection.get("selected_method") not in {"R3", "R5", "R6"}:
        raise RuntimeError("D1 selected primary method is outside R7 eligibility")
    cells = _validation_cells(trial)
    for _seed, _cell_path, cell in cells:
        cell_configuration = cell.get("configuration", {})
        planned = (
            cell_configuration.get("planned_configuration")
            if isinstance(cell_configuration, dict)
            else None
        )
        if (
            not isinstance(planned, dict)
            or trial_configuration(planned) != trial_config
        ):
            raise RuntimeError(
                "D1 selected trial cell differs from its trial configuration"
            )
        if any(cell_configuration.get(key) != value for key, value in planned.items()):
            raise RuntimeError(
                "D1 selected cell top-level/planned configurations differ"
            )
    raw_manifest_path = (
        root / "03_features" / "test" / "top5" / "matched_common_raw" / "manifest.json"
    )
    final_manifest_path = (
        root / "03_features" / "test" / "top5" / "T2_matched_common" / "manifest.json"
    )
    candidate_manifest_path = root / "02_candidates" / "test_manifest.json"
    raw_manifest = load_content_manifest(
        raw_manifest_path, name="D1 Test raw common", statuses=("COMPLETE",)
    )
    final_manifest = load_content_manifest(
        final_manifest_path, name="D1 Test final T2", statuses=("COMPLETE",)
    )
    candidate_manifest = load_content_manifest(
        candidate_manifest_path, name="D1 Test candidates", statuses=("COMPLETE",)
    )
    for name, value in (
        ("selection", selection),
        ("selected trial", trial),
        ("raw common", raw_manifest),
        ("final T2", final_manifest),
        ("candidates", candidate_manifest),
    ):
        if value.get("candidate_test_labels_read") is not False:
            raise RuntimeError(f"D1 {name} violates prelock Test-label isolation")
        verify_artifact_records_recursive(
            {"sources": value.get("sources"), "artifacts": value.get("artifacts")},
            name=f"D1 Test ranker {name}",
            require_at_least_one=True,
        )
    raw_path = verified_artifact_path(
        raw_manifest.get("artifacts", {}).get("candidate_features", {}),
        name="D1 Test raw common features",
    )
    candidate_path = verified_artifact_path(
        candidate_manifest.get("artifacts", {}).get("top5", {}),
        name="D1 Test Top5 candidates",
    )
    candidate_record = artifact_record(candidate_path)
    raw_sources = raw_manifest.get("sources")
    raw_extractor_sources = (
        raw_sources.get("extractor") if isinstance(raw_sources, dict) else None
    )
    if not isinstance(raw_extractor_sources, dict):
        raise RuntimeError("D1 Test raw common extractor sources are absent")
    _same_artifact_record(
        raw_extractor_sources.get("canonical_candidates"),
        candidate_record,
        name="D1 Test raw common Top5 candidates",
    )
    final_sources = final_manifest.get("sources")
    if not isinstance(final_sources, dict):
        raise RuntimeError("D1 Test final T2 sources are absent")
    _same_artifact_record(
        final_sources.get("candidates"),
        candidate_record,
        name="D1 Test final T2 Top5 candidates",
    )
    _same_artifact_record(
        final_sources.get("raw_common_manifest"),
        artifact_record(raw_manifest_path),
        name="D1 Test final T2 raw common manifest",
    )
    candidate_hash_record = candidate_manifest.get("artifacts", {}).get(
        "candidate_hashes", {}
    )
    candidate_hash_path = verified_artifact_path(
        candidate_hash_record, name="D1 Test candidate contract hashes"
    )
    paired_path = root / "01_manifests" / "d1_paired_manifest.parquet"
    cell_sources = {
        str(seed): {
            "cell": artifact_record(cell_path),
            "model": cell.get("artifacts", {}).get("model"),
            "preprocessor": cell.get("artifacts", {}).get("preprocessor"),
        }
        for seed, cell_path, cell in cells
    }
    sources = {
        "selection": artifact_record(selection_path),
        "selected_trial": artifact_record(trial_path),
        "raw_feature_manifest": artifact_record(raw_manifest_path),
        "raw_features": artifact_record(raw_path),
        "final_feature_manifest": artifact_record(final_manifest_path),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "candidate_hashes": artifact_record(candidate_hash_path),
        "denominator": artifact_record(paired_path),
        "selected_validation_cells": cell_sources,
        "inference_code": [
            artifact_record(path)
            for path in (
                ROOT / "src/d1_reranking/contracts.py",
                ROOT / "src/d1_reranking/candidates.py",
                ROOT / "src/d1_reranking/fold_calibration.py",
                ROOT / "src/d1_reranking/models.py",
                ROOT / "src/d1_reranking/selection.py",
                ROOT / "src/unified_reranking/datasets.py",
                ROOT / "src/unified_reranking/metrics.py",
                ROOT / "src/unified_reranking/telemetry.py",
                ROOT / "src/unified_reranking/training.py",
                ROOT / "tools/unified_reranking/apply_locked_matrix_cell.py",
                Path(__file__),
            )
        ],
    }
    signature = canonical_sha256(sources)
    output = root / "08_lock" / "label_free_test_rankers" / "d1"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name="D1 Test ranker application", statuses=("COMPLETE",)
        )
        if (
            resume
            and existing.get("sources") == sources
            and existing.get("source_signature_sha256") == canonical_sha256(sources)
            and existing.get("source_signature_sha256") == signature
        ):
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                    "seed_applications": existing.get("seed_applications"),
                },
                name="D1 Test ranker resume",
                require_at_least_one=True,
            )
            append_access_log(
                root,
                {
                    "event": "prelock_label_free_test_stage",
                    "stage": "d1_selected_ranker_test_application",
                    "output_manifest": str(manifest_path.resolve()),
                    "output_manifest_sha256": sha256_file(manifest_path),
                    "candidate_labels_opened_as_table": False,
                    "resumed": True,
                },
            )
            return existing
        raise RuntimeError("D1 Test ranker application differs or is corrupt")
    feature_started = time.perf_counter()
    schemas = {}
    for path, name in (
        (raw_path, "Test raw common features"),
        (candidate_path, "Test Top5 candidates"),
        (paired_path, "Test paired denominator"),
    ):
        schemas[path] = assert_label_free_parquet_schema(path, name=name)
    model_columns = tuple(map(str, final_manifest["model_feature_columns"]))
    raw_columns = _raw_feature_columns(
        model_columns, tuple(map(str, schemas[raw_path]))
    )
    raw_features = pd.read_parquet(raw_path, columns=list(raw_columns))
    candidates = pd.read_parquet(
        candidate_path,
        columns=[
            *CANDIDATE_REQUIRED_COLUMNS,
            "candidate_identity_precision_contract",
        ],
    )
    verify_canonical_candidate_frame(candidates, split="test")
    denominator = pd.read_parquet(paired_path, columns=["sample_id"])
    keys = ["sample_id", "candidate_id"]
    required_raw = {*keys, "native_rank", "native_score_raw"}
    missing = sorted(required_raw.difference(raw_features.columns))
    if missing:
        raise RuntimeError(f"D1 Test raw common features miss columns: {missing}")
    if raw_features.duplicated(keys).any() or candidates.duplicated(keys).any():
        raise RuntimeError("D1 Test raw features/candidates contain duplicate keys")
    raw_contract = _normalized_candidate_rank_contract(raw_features)
    candidate_contract = _normalized_candidate_rank_contract(candidates)
    if len(raw_contract) != len(candidate_contract) or not raw_contract.equals(
        candidate_contract
    ):
        raise RuntimeError(
            "D1 Test raw common candidate/rank universe differs from Top5"
        )
    if denominator["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("D1 Test denominator contains duplicate samples")
    foreign_samples = set(candidates["sample_id"].astype(str)).difference(
        denominator["sample_id"].astype(str)
    )
    if foreign_samples:
        raise RuntimeError("D1 Test candidates contain samples outside the denominator")
    grouped_counts = candidates.groupby("sample_id").size().astype(int)
    feature_latency_ms = (
        (time.perf_counter() - feature_started) * 1000.0 / len(raw_features)
    )
    seed_frames = {}
    applications = {}
    for _seed, cell_path, cell in cells:
        seed, predictions, application = _predict_cell(
            cell=cell,
            cell_path=cell_path,
            raw_features=raw_features,
            model_columns=model_columns,
            candidates=candidates,
        )
        seed_frames[seed] = predictions
        applications[str(seed)] = application
    ensemble = ensemble_seed_scores(seed_frames)
    decisions = _decisions(denominator, candidates, ensemble)
    observed_counts = decisions.set_index("sample_id")["candidate_count"].astype(int)
    expected_counts = (
        denominator["sample_id"].astype(str).map(grouped_counts).fillna(0).astype(int)
    )
    if (
        not observed_counts.reindex(denominator["sample_id"].astype(str))
        .reset_index(drop=True)
        .equals(expected_counts.reset_index(drop=True))
    ):
        raise RuntimeError("D1 Test ranker candidate counts differ from Top5")
    bearing = decisions["candidate_count"].gt(0)
    required_identity = [
        "native_candidate_id",
        "native_identity_sha256",
        "native_geometry_sha256",
        "selected_candidate_id",
        "selected_identity_sha256",
        "selected_geometry_sha256",
    ]
    if decisions.loc[bearing, required_identity].isna().any().any():
        raise RuntimeError("D1 candidate-bearing Test sample lacks ranker identity")
    telemetry = telemetry_payload(
        phase="d1_label_free_test_ranker",
        parameter_count=sum(
            int(value["parameter_count"]) for value in applications.values()
        ),
        ranker_latency_ms=float(
            np.mean([value["ranker_latency_ms"] for value in applications.values()])
        ),
        feature_latency_ms=feature_latency_ms,
        missing_feature_rate_value=float(
            np.mean(
                [
                    value["missing_feature_rate"]
                    for value in applications.values()
                ]
            )
        ),
    )
    artifacts = {
        "per_candidate_scores": artifact_record(
            atomic_parquet(ensemble, output / "per_candidate_scores.parquet")
        ),
        "per_sample_decisions": artifact_record(
            atomic_parquet(decisions, output / "per_sample_decisions.parquet")
        ),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_signature_sha256": signature,
        "selected_method": selection["selected_method"],
        "selected_trial_id": selection["selected_trial_id"],
        "seed_applications": applications,
        "candidate_test_labels_read": False,
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "feature_extraction_latency_ms": final_manifest[
            "feature_extraction_latency_ms"
        ],
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "d1_selected_ranker_test_application",
            "output_manifest": str(manifest_path.resolve()),
            "output_manifest_sha256": sha256_file(manifest_path),
            "candidate_labels_opened_as_table": False,
            "resumed": False,
        },
    )
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    path = root / "08_lock" / "label_free_test_rankers" / "d1" / "manifest.json"
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P13",
        substage="d1_label_free_test_ranker",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="selected_ungated",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
