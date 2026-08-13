"""Run a post-lock, secondary R1--R6 comparison on the frozen Test pool.

This command never writes to the source formal run.  It freezes the exact
Validation-selected single-seed cells, produces all label-free Test scores in
a separate output directory, and only then consumes the already-persisted
formal per-candidate outcome bundle.  Results are diagnostic and cannot feed
back into the primary R7 selection or formal claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the audited application module before importing torch directly.  On
# macOS it initializes LightGBM before PyTorch's OpenMP runtime.
from tools.unified_reranking.apply_locked_matrix_cell import (  # noqa: E402
    _lambdamart_predictions,
    _load_native_lightgbm_ranker,
    _model,
)

import torch  # noqa: E402

from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verified_manifest_artifact,
    verify_artifact_records_recursive,
)
from unified_reranking.contracts import assert_model_feature_columns  # noqa: E402
from unified_reranking.datasets import (  # noqa: E402
    FoldPreprocessor,
    build_inference_query_arrays,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.metrics import (  # noqa: E402
    compare_selections,
    evaluate_order_only,
)
from unified_reranking.rules import single_rule_scores  # noqa: E402
from unified_reranking.statistics import paired_system_statistics  # noqa: E402
from unified_reranking.training import (  # noqa: E402
    predict_neural_ranker,
    set_deterministic_cpu,
)


ROUTES = ("crog", "g1", "c1")
TRACK = "T2_matched_common"
EXPECTED_FINAL_LOCK_SHA256 = (
    "4b52eac6494e59a0f902792b794c3cca02824bf7a36bed4699b569986383f793"
)
RUNG_TO_METHOD = {
    "R1": "R1_soft_target_support",
    "R2": "R3_mlp_bce",
    "R3": "R4_mlp_ranknet",
    "R4": "R5_mlp_listwise",
    "R5": "R6_lambdamart",
    "R6": "R7_mlp_jacquard_margin",
}
RUNG_LABELS = {
    "R0": "native q",
    "R1": "calibrated q + soft target support",
    "R2": "Residual MLP + BCE",
    "R3": "Residual MLP + RankNet",
    "R4": "Residual MLP + multi-positive listwise",
    "R5": "LambdaMART",
    "R6": "Jacquard-Margin RankNet",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _append_access(output_dir: Path, payload: dict[str, Any]) -> None:
    path = output_dir / "test_access.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": _utc_now(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verified_formal_bundle(source_run: Path) -> dict[str, dict[str, str]]:
    execution_path = source_run / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("status") != "COMPLETE" or execution.get("execution_count") != 1:
        raise RuntimeError("source formal Test execution is not exactly-once COMPLETE")
    artifacts = execution.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("formal execution artifact inventory is missing")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"formal execution artifact {name} is malformed")
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"formal execution artifact {name} hash mismatch")
    manifest_record = artifacts.get("formal_test_manifest")
    score_record = artifacts.get("per_candidate_scores")
    if manifest_record is None or score_record is None:
        raise RuntimeError("formal outcome bundle is incomplete")
    canonical_manifest = source_run / "09_formal_test" / "formal_test_manifest.json"
    if Path(manifest_record["path"]).resolve() != canonical_manifest.resolve():
        raise RuntimeError("formal execution redirects the formal manifest")
    manifest = load_verified_json(canonical_manifest, name="formal Test manifest")
    nested = manifest.get("artifacts", {}).get("per_candidate_scores")
    if nested != score_record:
        raise RuntimeError("formal per-candidate score record differs across manifests")
    return {
        "execution": _record(execution_path),
        "manifest": dict(manifest_record),
        "per_candidate_scores": dict(score_record),
        "metrics": dict(artifacts["formal_test_metrics"]),
    }


def build_comparison_plan(
    source_run: Path,
    output_dir: Path,
    *,
    expected_final_lock_sha256: str = EXPECTED_FINAL_LOCK_SHA256,
) -> dict[str, Any]:
    """Freeze the complete comparison without reading candidate outcomes."""

    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    final_lock = source_run / "FINAL_RUN_LOCK.json"
    if sha256_file(final_lock) != expected_final_lock_sha256:
        raise RuntimeError("source FINAL_RUN_LOCK SHA-256 differs from the authority")
    lock_payload = json.loads(final_lock.read_text(encoding="utf-8"))
    if lock_payload.get("status") != "COMPLETE":
        raise RuntimeError("source FINAL_RUN_LOCK is not COMPLETE")
    root_manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    if (
        root_manifest.get("status") != "COMPLETE"
        or root_manifest.get("formal_test_execution_count") != 1
    ):
        raise RuntimeError("source root lifecycle is not terminal exactly-once COMPLETE")
    formal = _verified_formal_bundle(source_run)

    table_path = source_run / "07_validation" / "tables" / "screen_best_within_method.csv"
    table = pd.read_csv(table_path)
    required = {
        "route",
        "track",
        "method_code",
        "configuration_json",
        "manifest_path",
        "manifest_sha256",
        "best_within_method",
    }
    if not required.issubset(table.columns):
        raise RuntimeError("Validation screen table schema is incomplete")
    selected = table.loc[
        table["route"].isin(ROUTES)
        & table["track"].eq(TRACK)
        & table["method_code"].isin(RUNG_TO_METHOD.values())
        & table["best_within_method"].astype(bool)
    ].copy()
    if len(selected) != len(ROUTES) * len(RUNG_TO_METHOD):
        raise RuntimeError("expected exactly 18 frozen route/rung Validation winners")
    if selected.duplicated(["route", "method_code"]).any():
        raise RuntimeError("Validation winner table contains duplicate route/method rows")
    method_to_rung = {value: key for key, value in RUNG_TO_METHOD.items()}
    cells: list[dict[str, Any]] = []
    for row in selected.sort_values(["route", "method_code"]).itertuples(index=False):
        route = str(row.route)
        rung = method_to_rung[str(row.method_code)]
        manifest_path = Path(str(row.manifest_path)).resolve()
        if sha256_file(manifest_path) != str(row.manifest_sha256):
            raise RuntimeError(f"{route}/{rung} winner manifest hash mismatch")
        cell = load_verified_json(manifest_path, name=f"{route}/{rung} winner cell")
        if cell.get("status") != "COMPLETE":
            raise RuntimeError(f"{route}/{rung} winner cell is not COMPLETE")
        configuration = cell.get("configuration", {})
        expected_configuration = json.loads(str(row.configuration_json))
        if configuration != expected_configuration:
            raise RuntimeError(f"{route}/{rung} configuration differs from screen table")
        if (
            configuration.get("route") != route
            or configuration.get("track") != TRACK
            or configuration.get("mode") != "validation"
            or int(configuration.get("seed", -1)) != 42
        ):
            raise RuntimeError(f"{route}/{rung} violates the single-seed T2 contract")
        verify_artifact_records_recursive(
            {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
            name=f"{route}/{rung} winner cell",
            require_at_least_one=True,
        )
        feature_manifest_path = (
            source_run
            / "03_features"
            / "tracks"
            / TRACK
            / f"{route}_test"
            / "feature_manifest.json"
        )
        feature_manifest = load_verified_json(
            feature_manifest_path, name=f"{route} Test feature manifest"
        )
        feature_path = verified_manifest_artifact(
            feature_manifest, name=f"{route} Test features"
        )
        columns = assert_model_feature_columns(feature_manifest["model_feature_columns"])
        if tuple(map(str, cell["feature_columns"])) != tuple(columns):
            raise RuntimeError(f"{route}/{rung} Train/Test feature schema mismatch")
        candidate_path = source_run / "02_candidates" / f"{route}_test_top5.parquet"
        cells.append(
            {
                "route": route,
                "rung": rung,
                "label": RUNG_LABELS[rung],
                "method_code": str(row.method_code),
                "validation_j_at_1": float(row.j_at_1),
                "cell_manifest": _record(manifest_path),
                "test_feature_manifest": _record(feature_manifest_path),
                "test_features": _record(feature_path),
                "test_candidates": _record(candidate_path),
                "configuration": configuration,
            }
        )

    denominator = source_run / "01_manifests" / "paired_test.parquet"
    denominator_metadata = pd.read_parquet(
        denominator, columns=["sample_id", "scene_id", "frame_id"]
    )
    if (
        len(denominator_metadata) != 7_675
        or denominator_metadata["sample_id"].astype(str).duplicated().any()
    ):
        raise RuntimeError("paired Test denominator is not the frozen 7,675 samples")
    plan: dict[str, Any] = {
        "status": "LOCKED_POSTFORMAL_SECONDARY",
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "purpose": "post-lock diagnostic R1-R6 Test comparison",
        "feeds_primary_selection": False,
        "changes_formal_conclusion": False,
        "candidate_test_labels_read_during_plan": False,
        "comparison_contract": {
            "routes": list(ROUTES),
            "track": TRACK,
            "seed": 42,
            "rungs": {key: RUNG_LABELS[key] for key in RUNG_TO_METHOD},
            "baseline": "R0 native q",
            "denominator": 7_675,
            "primary_metric": "J@1",
            "primary_uncertainty": "paired scene-cluster bootstrap, 10,000 resamples",
            "supportive_test": "exact paired McNemar with Holm correction within route",
        },
        "sources": {
            "final_run_lock": _record(final_lock),
            "root_manifest": _record(source_run / "manifest.json"),
            "screen_best_within_method": _record(table_path),
            "paired_test": _record(denominator),
            "formal_execution": formal["execution"],
            "formal_manifest": formal["manifest"],
            "formal_per_candidate_scores": formal["per_candidate_scores"],
            "formal_metrics": formal["metrics"],
            "tool": _record(Path(__file__)),
            "inference_application_reference": _record(
                ROOT / "tools" / "unified_reranking" / "apply_locked_matrix_cell.py"
            ),
            "datasets_primitive": _record(SRC / "unified_reranking" / "datasets.py"),
            "models_primitive": _record(
                SRC / "unified_reranking" / "models" / "core.py"
            ),
            "rules_primitive": _record(SRC / "unified_reranking" / "rules.py"),
            "training_primitive": _record(SRC / "unified_reranking" / "training.py"),
            "metrics_primitive": _record(SRC / "unified_reranking" / "metrics.py"),
            "statistics_primitive": _record(
                SRC / "unified_reranking" / "statistics.py"
            ),
        },
        "cells": cells,
    }
    plan["content_sha256"] = canonical_sha256(plan)
    plan_path = output_dir / "00_plan" / "comparison_plan.json"
    atomic_json(plan_path, plan)
    _append_access(
        output_dir,
        {
            "event": "comparison_plan_locked",
            "candidate_outcomes_opened": False,
            "plan": _record(plan_path),
        },
    )
    return plan


def _load_test_features(source_run: Path, route: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    manifest_path = (
        source_run
        / "03_features"
        / "tracks"
        / TRACK
        / f"{route}_test"
        / "feature_manifest.json"
    )
    manifest = load_verified_json(manifest_path, name=f"{route} Test feature manifest")
    path = verified_manifest_artifact(manifest, name=f"{route} Test features")
    return pd.read_parquet(path), assert_model_feature_columns(manifest["model_feature_columns"])


def _predict_cell(source_run: Path, entry: dict[str, Any]) -> pd.DataFrame:
    route = str(entry["route"])
    rung = str(entry["rung"])
    manifest_path = Path(entry["cell_manifest"]["path"])
    if sha256_file(manifest_path) != entry["cell_manifest"]["sha256"]:
        raise RuntimeError(f"{route}/{rung} cell drifted after plan lock")
    cell = load_verified_json(manifest_path, name=f"{route}/{rung} cell")
    features, columns = _load_test_features(source_run, route)
    if tuple(map(str, cell["feature_columns"])) != tuple(columns):
        raise RuntimeError(f"{route}/{rung} feature columns changed after plan lock")
    preprocessor = FoldPreprocessor.from_artifact(cell["preprocessor"])
    if preprocessor.columns != tuple(columns):
        raise RuntimeError(f"{route}/{rung} persisted preprocessor schema mismatch")
    configuration = cell["configuration"]
    if rung == "R1":
        transformed = preprocessor.transform(features)
        base_logit = pd.to_numeric(features["base_logit"], errors="raise").to_numpy(float)
        scores = single_rule_scores(
            base_logit,
            transformed,
            columns,
            family="soft_target_support",
            alpha=float(configuration["alpha"]),
        )
        predictions = features[["sample_id", "candidate_id"]].copy()
        predictions["score"] = np.asarray(scores, dtype=float)
    else:
        arrays = build_inference_query_arrays(features, preprocessor=preprocessor)
        encoder = str(configuration["encoder"])
        model_record = cell["artifacts"]["model"]
        model_path = Path(model_record["path"])
        if sha256_file(model_path) != model_record["sha256"]:
            raise RuntimeError(f"{route}/{rung} model hash mismatch")
        if encoder == "lambdamart":
            model = _load_native_lightgbm_ranker(model_path)
            predictions = _lambdamart_predictions(model, arrays)
        else:
            set_deterministic_cpu(int(configuration["seed"]))
            payload = torch.load(model_path, map_location="cpu", weights_only=True)
            model = _model(
                encoder,
                int(payload["input_dim"]),
                float(payload["alpha"]),
                attention_blocks=int(payload.get("num_attention_blocks", 2)),
                edge_dim=int(payload.get("edge_dim", 1)),
            )
            model.load_state_dict(payload["state_dict"])
            # A small deterministic batch avoids oversubscribing Accelerate/
            # OpenMP on the shared macOS host.  The scorer is query-separable,
            # so this changes neither ordering nor floating-point operations
            # within a query.
            predictions = predict_neural_ranker(model, arrays, batch_size=256)
    if not np.isfinite(pd.to_numeric(predictions["score"], errors="coerce")).all():
        raise RuntimeError(f"{route}/{rung} produced non-finite Test scores")
    candidate_path = source_run / "02_candidates" / f"{route}_test_top5.parquet"
    candidates = pd.read_parquet(
        candidate_path,
        columns=["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"],
    )
    expected = set(map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy()))
    observed = set(map(tuple, predictions[["sample_id", "candidate_id"]].astype(str).to_numpy()))
    if expected != observed or predictions.duplicated(["sample_id", "candidate_id"]).any():
        raise RuntimeError(f"{route}/{rung} predictions changed frozen candidate membership")
    result = candidates.merge(predictions, on=["sample_id", "candidate_id"], validate="one_to_one")
    result.insert(0, "rung", rung)
    result.insert(0, "route", route)
    return result


def generate_predictions(source_run: Path, output_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Generate and freeze all 18 score tables without opening correctness."""

    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    plan_path = output_dir / "00_plan" / "comparison_plan.json"
    if (
        load_verified_json(
            plan_path,
            name="post-lock comparison plan",
            statuses=("LOCKED_POSTFORMAL_SECONDARY",),
        )
        != plan
    ):
        raise RuntimeError("in-memory comparison plan differs from frozen plan")
    artifacts: dict[str, dict[str, str]] = {}
    frames: list[pd.DataFrame] = []
    for entry in plan["cells"]:
        route, rung = str(entry["route"]), str(entry["rung"])
        predictions = _predict_cell(source_run, entry)
        path = output_dir / "01_predictions" / f"{route}_{rung.lower()}_scores.parquet"
        _atomic_parquet(path, predictions)
        artifacts[f"{route}_{rung}"] = _record(path)
        frames.append(predictions)
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "01_predictions" / "all_r1_r6_scores.parquet"
    _atomic_parquet(combined_path, combined)
    inventory: dict[str, Any] = {
        "status": "PREDICTIONS_FROZEN",
        "created_at_utc": _utc_now(),
        "candidate_outcomes_read": False,
        "plan": _record(plan_path),
        "artifacts": {**artifacts, "combined": _record(combined_path)},
        "candidate_rows": len(combined),
        "applications": len(artifacts),
    }
    inventory["content_sha256"] = canonical_sha256(inventory)
    inventory_path = output_dir / "01_predictions" / "prediction_inventory.json"
    atomic_json(inventory_path, inventory)
    _append_access(
        output_dir,
        {
            "event": "all_label_free_predictions_frozen",
            "candidate_outcomes_opened": False,
            "prediction_inventory": _record(inventory_path),
        },
    )
    return inventory


def freeze_postlock_comparison(
    output_dir: Path,
    plan: dict[str, Any],
    prediction_inventory: dict[str, Any],
) -> dict[str, Any]:
    """Predeclare the outcome read and statistics after scores are immutable."""

    output_dir = output_dir.resolve()
    lock: dict[str, Any] = {
        "status": "LOCKED_POSTFORMAL_SECONDARY",
        "designation": "POST_LOCK_COMPARATIVE_ONLY",
        "created_at_utc": _utc_now(),
        "primary_reselection_permitted": False,
        "selection_feedback_allowed": False,
        "changes_formal_conclusion": False,
        "source_formal_execution_count": 1,
        "derived_formal_execution_count": 0,
        "outcome_source": plan["sources"]["formal_per_candidate_scores"],
        "plan": _record(output_dir / "00_plan" / "comparison_plan.json"),
        "prediction_inventory": _record(
            output_dir / "01_predictions" / "prediction_inventory.json"
        ),
        "frozen_predictions": prediction_inventory["artifacts"],
        "statistics_contract": {
            "denominator": 7_675,
            "primary_metric": "J@1",
            "comparison_family": "R1-R6 versus R0 separately within each route",
            "primary_uncertainty": "scene-cluster bootstrap",
            "bootstrap_iterations": 10_000,
            "bootstrap_seed": 20260808,
            "supportive_test": "exact paired two-sided McNemar",
            "multiplicity": "Holm within each six-comparison route family",
            "frame_cluster_bootstrap": "sensitivity only",
        },
    }
    lock["content_sha256"] = canonical_sha256(lock)
    path = output_dir / "POSTLOCK_COMPARISON_LOCK.json"
    atomic_json(path, lock)
    _atomic_text(output_dir / "POSTLOCK_COMPARISON_LOCK.sha256", sha256_file(path) + "\n")
    _append_access(
        output_dir,
        {
            "event": "postlock_comparison_locked_before_outcomes",
            "candidate_outcomes_opened": False,
            "lock": _record(path),
        },
    )
    return lock


def _formal_route_labels(
    formal_scores_path: Path,
    source_run: Path,
) -> dict[str, pd.DataFrame]:
    """Read the persisted formal outcome bundle exactly once per execution."""

    formal = pd.read_parquet(
        formal_scores_path,
        columns=[
            "system_name",
            "route",
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_geometry_sha256",
            "candidate_success",
        ],
    )
    result: dict[str, pd.DataFrame] = {}
    for route in ROUTES:
        rows = formal.loc[
            formal["system_name"].eq(f"{route}_native")
            & formal["route"].astype(str).str.lower().eq(route)
        ].copy()
        required = source_run / "02_candidates" / f"{route}_test_top5.parquet"
        candidates = pd.read_parquet(
            required,
            columns=["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"],
        )
        if rows.duplicated(["sample_id", "candidate_id"]).any():
            raise RuntimeError(f"formal {route} native rows contain duplicate candidate keys")
        expected = set(map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy()))
        observed = set(map(tuple, rows[["sample_id", "candidate_id"]].astype(str).to_numpy()))
        if expected != observed:
            raise RuntimeError(f"formal {route} labels do not exactly cover frozen Top-5")
        merged = candidates.merge(
            rows[["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256", "candidate_success"]],
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
            suffixes=("_candidate", "_formal"),
        )
        if not merged["native_rank_candidate"].eq(merged["native_rank_formal"]).all():
            raise RuntimeError(f"formal {route} native ranks differ from frozen pool")
        if not merged["candidate_geometry_sha256_candidate"].eq(
            merged["candidate_geometry_sha256_formal"]
        ).all():
            raise RuntimeError(f"formal {route} geometry differs from frozen pool")
        labels = pd.to_numeric(merged["candidate_success"], errors="raise")
        if not labels.isin([0, 1]).all():
            raise RuntimeError(f"formal {route} candidate outcomes are not binary")
        result[route] = pd.DataFrame(
            {
                "sample_id": merged["sample_id"].astype(str),
                "candidate_id": merged["candidate_id"].astype(str),
                "native_rank": merged["native_rank_candidate"].astype(int),
                "candidate_success": labels.astype(int),
            }
        )
    return result


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def evaluate_predictions(
    source_run: Path,
    output_dir: Path,
    plan: dict[str, Any],
    prediction_inventory: dict[str, Any],
    comparison_lock: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen predictions using only the locked formal outcome bundle."""

    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    plan_path = output_dir / "00_plan" / "comparison_plan.json"
    inventory_path = output_dir / "01_predictions" / "prediction_inventory.json"
    comparison_lock_path = output_dir / "POSTLOCK_COMPARISON_LOCK.json"
    if (
        load_verified_json(
            plan_path,
            name="comparison plan",
            statuses=("LOCKED_POSTFORMAL_SECONDARY",),
        )
        != plan
    ):
        raise RuntimeError("comparison plan drifted before outcome read")
    if (
        load_verified_json(
            inventory_path,
            name="prediction inventory",
            statuses=("PREDICTIONS_FROZEN",),
        )
        != prediction_inventory
    ):
        raise RuntimeError("prediction inventory drifted before outcome read")
    if (
        load_verified_json(
            comparison_lock_path,
            name="post-lock comparison lock",
            statuses=("LOCKED_POSTFORMAL_SECONDARY",),
        )
        != comparison_lock
    ):
        raise RuntimeError("post-lock comparison lock drifted before outcome read")
    detached = (output_dir / "POSTLOCK_COMPARISON_LOCK.sha256").read_text(
        encoding="utf-8"
    ).strip()
    if detached != sha256_file(comparison_lock_path):
        raise RuntimeError("post-lock comparison detached digest mismatch")
    verify_artifact_records_recursive(
        prediction_inventory["artifacts"],
        name="frozen R1-R6 predictions",
        require_at_least_one=True,
    )
    formal_record = plan["sources"]["formal_per_candidate_scores"]
    formal_path = Path(formal_record["path"])
    if sha256_file(formal_path) != formal_record["sha256"]:
        raise RuntimeError("formal per-candidate outcome bundle drifted")
    _append_access(
        output_dir,
        {
            "event": "postlock_formal_candidate_outcome_bundle_open",
            "purpose": "secondary R1-R6 diagnostic comparison",
            "source": formal_record,
            "postlock_comparison_lock": _record(comparison_lock_path),
            "candidate_outcomes_opened": True,
            "feeds_primary_selection": False,
        },
    )
    labels_by_route = _formal_route_labels(formal_path, source_run)
    denominator_frame = pd.read_parquet(
        source_run / "01_manifests" / "paired_test.parquet",
        columns=["sample_id", "scene_id", "frame_id"],
    )
    denominator_frame["sample_id"] = denominator_frame["sample_id"].astype(str)
    sample_ids = denominator_frame["sample_id"].tolist()
    all_predictions = pd.read_parquet(
        prediction_inventory["artifacts"]["combined"]["path"]
    )
    metrics_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    statistics: dict[str, Any] = {"routes": {}}
    formal_metrics = json.loads(Path(plan["sources"]["formal_metrics"]["path"]).read_text(encoding="utf-8"))["systems"]
    r7_rows: list[dict[str, Any]] = []

    for route in ROUTES:
        labels = labels_by_route[route]
        native_input = labels.copy()
        native_input["score"] = -native_input["native_rank"].astype(float)
        native_metrics, native_decisions = evaluate_order_only(
            sample_ids, native_input, score_column="score"
        )
        native_decisions.insert(0, "label", RUNG_LABELS["R0"])
        native_decisions.insert(0, "rung", "R0")
        native_decisions.insert(0, "route", route)
        native_decisions = native_decisions.merge(
            denominator_frame, on="sample_id", validate="one_to_one"
        )
        decision_frames.append(native_decisions)
        metrics_rows.append(
            {
                "route": route,
                "rung": "R0",
                "label": RUNG_LABELS["R0"],
                **native_metrics,
                "native_j_at_1": native_metrics["j_at_1"],
                "delta_j_at_1": 0.0,
                "recovered": 0,
                "harmful": 0,
                "net": 0,
                "switch_count": 0,
                "switch_rate": 0.0,
                "mcnemar_raw_p": 1.0,
                "mcnemar_holm_p": 1.0,
                "scene_ci_low": 0.0,
                "scene_ci_high": 0.0,
            }
        )
        route_stats: dict[str, Any] = {}
        route_rows: list[dict[str, Any]] = []
        for rung in RUNG_TO_METHOD:
            scores = all_predictions.loc[
                all_predictions["route"].eq(route) & all_predictions["rung"].eq(rung),
                ["sample_id", "candidate_id", "native_rank", "score"],
            ]
            evaluation_input = labels.merge(
                scores,
                on=["sample_id", "candidate_id", "native_rank"],
                validate="one_to_one",
            )
            rung_metrics, decisions = evaluate_order_only(
                sample_ids, evaluation_input, score_column="score"
            )
            comparison = compare_selections(
                native_decisions,
                decisions,
                oracle_at_5=float(native_metrics["oracle_at_5"]),
            )
            decisions.insert(0, "label", RUNG_LABELS[rung])
            decisions.insert(0, "rung", rung)
            decisions.insert(0, "route", route)
            decisions = decisions.merge(denominator_frame, on="sample_id", validate="one_to_one")
            decision_frames.append(decisions)
            paired = paired_system_statistics(
                native_decisions["selected_correct"],
                decisions["selected_correct"],
                scene_ids=decisions["scene_id"],
                frame_ids=decisions["frame_id"],
            )
            raw_p = float(paired["mcnemar_conventional_supportive"]["pvalue"])
            ci = paired["scene_bootstrap"]["ci95"]
            row = {
                "route": route,
                "rung": rung,
                "label": RUNG_LABELS[rung],
                **rung_metrics,
                **comparison,
                "mcnemar_raw_p": raw_p,
                "mcnemar_holm_p": None,
                "scene_ci_low": float(ci[0]),
                "scene_ci_high": float(ci[1]),
            }
            metrics_rows.append(row)
            route_rows.append(row)
            route_stats[rung] = paired
        adjusted = multipletests(
            [row["mcnemar_raw_p"] for row in route_rows], method="holm"
        )[1]
        for row, adjusted_p in zip(route_rows, adjusted, strict=True):
            row["mcnemar_holm_p"] = float(adjusted_p)
            for metrics_row in metrics_rows:
                if metrics_row["route"] == route and metrics_row["rung"] == row["rung"]:
                    metrics_row["mcnemar_holm_p"] = float(adjusted_p)
                    break
            route_stats[row["rung"]]["mcnemar_conventional_supportive"][
                "holm_adjusted_pvalue_within_route"
            ] = float(adjusted_p)
        statistics["routes"][route] = route_stats
        for kind in ("ungated_primary", "gated_primary"):
            system = f"{route}_{kind}"
            r7_rows.append(
                {
                    "route": route,
                    "reference": f"R7_{'ungated' if kind == 'ungated_primary' else 'gated'}",
                    "j_at_1": float(formal_metrics[system]["j_at_1"]),
                    "oracle_at_5": float(formal_metrics[system]["oracle_at_5"]),
                    "source_system_name": system,
                }
            )

    metrics_frame = pd.DataFrame(metrics_rows).sort_values(["route", "rung"])
    decisions_frame = pd.concat(decision_frames, ignore_index=True)
    r7_frame = pd.DataFrame(r7_rows)
    evaluation_dir = output_dir / "02_evaluation"
    metrics_csv = evaluation_dir / "r0_r6_test_metrics.csv"
    decisions_path = evaluation_dir / "per_sample_decisions.parquet"
    r7_path = evaluation_dir / "formal_r7_reference.csv"
    _atomic_text(metrics_csv, metrics_frame.to_csv(index=False))
    _atomic_parquet(decisions_path, decisions_frame)
    _atomic_text(r7_path, r7_frame.to_csv(index=False))

    statistics["multiplicity"] = {
        "family": "six R1-R6 versus R0 comparisons within each route",
        "method": "Holm correction on exact McNemar supportive p-values",
        "scene_bootstrap_is_primary": True,
    }
    statistics["content_sha256"] = canonical_sha256(statistics)
    statistics_path = evaluation_dir / "statistics.json"
    atomic_json(statistics_path, statistics)

    conclusions: list[str] = [
        "# R1–R6 锁后 Test 对照结论",
        "",
        "本分析是正式锁定后的次级诊断实验。它不参与模型选择，不改变 R7 正式三种子集成/门控结论。",
        "所有 R1–R6 均使用 Validation 阶段已冻结的 T2、seed=42 最优配置；Test 分母固定为 7,675。",
        "",
    ]
    summary_rows: list[dict[str, Any]] = []
    for route in ROUTES:
        subset = metrics_frame.loc[
            metrics_frame["route"].eq(route) & metrics_frame["rung"].isin(RUNG_TO_METHOD)
        ].sort_values(["j_at_1", "rung"], ascending=[False, True])
        best = subset.iloc[0]
        r7_gated = r7_frame.loc[
            r7_frame["route"].eq(route) & r7_frame["reference"].eq("R7_gated"), "j_at_1"
        ].iloc[0]
        summary_rows.append(
            {
                "route": route,
                "best_rung": best["rung"],
                "best_j_at_1": float(best["j_at_1"]),
                "delta_vs_native": float(best["delta_j_at_1"]),
                "scene_ci_low": float(best["scene_ci_low"]),
                "scene_ci_high": float(best["scene_ci_high"]),
                "formal_r7_gated_j_at_1": float(r7_gated),
                "gap_to_formal_r7_gated": float(best["j_at_1"] - r7_gated),
            }
        )
        conclusions.extend(
            [
                f"## {route.upper()}",
                "",
                (
                    f"单种子最优为 **{best['rung']} ({best['label']})**："
                    f"J@1={100*float(best['j_at_1']):.3f}%，相对 R0 "
                    f"{100*float(best['delta_j_at_1']):+.3f} pp；"
                    f"scene-bootstrap 95% CI "
                    f"[{100*float(best['scene_ci_low']):+.3f}, "
                    f"{100*float(best['scene_ci_high']):+.3f}] pp。"
                ),
                (
                    f"正式 R7 gated 为 {100*float(r7_gated):.3f}%；"
                    f"该单种子最优与正式 R7 的差为 "
                    f"{100*(float(best['j_at_1'])-float(r7_gated)):+.3f} pp。"
                ),
                "",
            ]
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_path = evaluation_dir / "route_conclusions.csv"
    conclusions_path = evaluation_dir / "CONCLUSIONS_ZH.md"
    _atomic_text(summary_path, summary_frame.to_csv(index=False))
    _atomic_text(conclusions_path, "\n".join(conclusions) + "\n")

    metrics_payload: dict[str, Any] = {
        "status": "COMPLETE",
        "scope": "post-lock secondary comparison; no selection feedback",
        "sample_count": 7_675,
        "rows": [
            {key: _json_scalar(value) for key, value in row.items()}
            for row in metrics_frame.replace({np.nan: None}).to_dict(orient="records")
        ],
        "route_conclusions": summary_frame.to_dict(orient="records"),
    }
    metrics_payload["content_sha256"] = canonical_sha256(metrics_payload)
    metrics_json = evaluation_dir / "metrics.json"
    atomic_json(metrics_json, metrics_payload)
    return {
        "metrics": _record(metrics_json),
        "metrics_csv": _record(metrics_csv),
        "decisions": _record(decisions_path),
        "statistics": _record(statistics_path),
        "r7_reference": _record(r7_path),
        "route_conclusions": _record(summary_path),
        "conclusions_zh": _record(conclusions_path),
    }


def execute(
    source_run: Path,
    output_dir: Path,
    *,
    expected_final_lock_sha256: str = EXPECTED_FINAL_LOCK_SHA256,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if output_dir == source_run or source_run in output_dir.parents:
        raise ValueError("output directory must be outside the immutable source run")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("post-lock comparison output directory must be new and empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    before = sha256_file(source_run / "FINAL_RUN_LOCK.json")
    plan = build_comparison_plan(
        source_run,
        output_dir,
        expected_final_lock_sha256=expected_final_lock_sha256,
    )
    predictions = generate_predictions(source_run, output_dir, plan)
    comparison_lock = freeze_postlock_comparison(output_dir, plan, predictions)
    evaluation = evaluate_predictions(
        source_run, output_dir, plan, predictions, comparison_lock
    )
    after = sha256_file(source_run / "FINAL_RUN_LOCK.json")
    if before != after or after != expected_final_lock_sha256:
        raise RuntimeError("source FINAL_RUN_LOCK changed during post-lock comparison")
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "schema_version": 1,
        "completed_at_utc": _utc_now(),
        "source_run": str(source_run),
        "postlock_secondary_only": True,
        "feeds_primary_selection": False,
        "formal_test_execution_count_added": 0,
        "source_final_lock_sha256_before": before,
        "source_final_lock_sha256_after": after,
        "plan": _record(output_dir / "00_plan" / "comparison_plan.json"),
        "predictions": _record(output_dir / "01_predictions" / "prediction_inventory.json"),
        "postlock_comparison_lock": _record(
            output_dir / "POSTLOCK_COMPARISON_LOCK.json"
        ),
        "evaluation": evaluation,
        "access_log": _record(output_dir / "test_access.log"),
    }
    result["content_sha256"] = canonical_sha256(result)
    execution_path = output_dir / "EXECUTION.json"
    atomic_json(execution_path, result)
    _atomic_text(output_dir / "COMPLETE", sha256_file(execution_path) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-final-lock-sha256",
        default=EXPECTED_FINAL_LOCK_SHA256,
    )
    args = parser.parse_args()
    result = execute(
        args.source_run,
        args.output_dir,
        expected_final_lock_sha256=args.expected_final_lock_sha256,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
