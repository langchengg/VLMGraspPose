from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.telemetry import (
    TELEMETRY_FIELDS,
    resolved_track_extraction_latency,
)
from tools.unified_reranking.build_postformal_artifacts import (
    required_analysis_registry,
)
from tools.unified_reranking.run_validation_feature_ablations import (
    ROUTES,
    _verified_validation_metrics,
    run as run_ablations,
)
from tools.unified_reranking.train_matrix_cell import run as train_matrix_cell


SEEDS = (42, 123, 2026)
FEATURES = ("base_logit", "p_center", "jaw_probability_min", "local_depth_median")


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _training_args(run: Path) -> Namespace:
    return Namespace(
        run_dir=run,
        route="crog",
        track="T2_matched_common",
        encoder="linear",
        loss="ranknet",
        seed=42,
        mode="validation",
        fold=None,
        learning_rate=1e-3,
        weight_decay=0.0,
        alpha=0.5,
        temperature=1.0,
        beta=1.0,
        epochs=1,
        patience=1,
        batch_size=16,
        num_leaves=7,
        tree_learning_rate=0.05,
        n_estimators=5,
        num_attention_blocks=1,
    )


def test_t3_latency_uses_and_binds_fixed_validation_benchmark(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    child = run / "child.txt"
    child.parent.mkdir(parents=True)
    child.write_text("verified", encoding="utf-8")
    benchmark_path = (
        run / "07_validation/telemetry/feature_extraction_benchmark.json"
    )
    _write_json(
        benchmark_path,
        {
            "status": "COMPLETE",
            "analysis": "validation_feature_extraction_runtime",
            "candidate_test_labels_read": False,
            "configuration": {
                "split": "validation",
                "tag": "latency_benchmark_128",
                "sample_limit": 128,
            },
            "feature_extraction_latency_ms": 3.25,
            "component_measurements": [
                {
                    "name": "common/crog",
                    "feature_extraction_latency_ms": 1.5,
                },
                {
                    "name": "rgb/crog",
                    "feature_extraction_latency_ms": 2.0,
                },
            ],
            "sources": {"child": _record(child)},
            "artifacts": {},
        },
    )
    latency, source = resolved_track_extraction_latency(
        run,
        "T3_tri_backend",
        {"route": "crog", "feature_track_assembly_latency_ms": 0.25},
    )
    assert latency == pytest.approx(7.0)
    assert source == _record(benchmark_path)
    child.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        resolved_track_extraction_latency(
            run,
            "T3_tri_backend",
            {"route": "crog", "feature_track_assembly_latency_ms": 0.25},
        )


def test_t3_direct_latency_cannot_bypass_fixed_benchmark(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="must not replace"):
        resolved_track_extraction_latency(
            tmp_path,
            "T3_tri_backend",
            {"feature_extraction_latency_ms": 0.01},
        )


def test_t3_composite_latency_requires_track_assembly(tmp_path: Path) -> None:
    benchmark = tmp_path / "07_validation/telemetry/feature_extraction_benchmark.json"
    child = tmp_path / "child"
    child.write_text("x", encoding="utf-8")
    _write_json(
        benchmark,
        {
            "status": "COMPLETE",
            "analysis": "validation_feature_extraction_runtime",
            "candidate_test_labels_read": False,
            "configuration": {
                "split": "validation",
                "tag": "latency_benchmark_128",
                "sample_limit": 128,
            },
            "feature_extraction_latency_ms": 1.0,
            "component_measurements": [
                {"name": "common/g1", "feature_extraction_latency_ms": 1.0},
                {"name": "rgb/g1", "feature_extraction_latency_ms": 1.0},
            ],
            "sources": {"child": _record(child)},
            "artifacts": {},
        },
    )
    with pytest.raises(RuntimeError, match="assembly latency"):
        resolved_track_extraction_latency(
            tmp_path, "T3_tri_backend", {"route": "g1"}
        )


def _build_training_run(root: Path) -> Path:
    run = root / "train_run"
    train_ids = [f"t{index}" for index in range(10)]
    validation_ids = ["v0", "v1"]
    for split, sample_ids in (("train", train_ids), ("validation", validation_ids)):
        rows = []
        labels = []
        for sample_index, sample_id in enumerate(sample_ids):
            for native_rank, candidate_id in ((1, "a"), (2, "b")):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "native_rank": native_rank,
                        "base_logit": 1.0 - native_rank,
                        "p_center": 0.2 + 0.1 * sample_index + 0.3 * (native_rank == 2),
                        "jaw_probability_min": 0.8 - 0.2 * native_rank,
                        "local_depth_median": np.nan if sample_index == 0 else 0.5,
                    }
                )
                labels.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "candidate_success": int(native_rank == 2),
                        "jacquard_margin": 0.5 if native_rank == 2 else -0.2,
                    }
                )
        feature_path = (
            run
            / "03_features"
            / "tracks"
            / "T2_matched_common"
            / f"crog_{split}"
            / "candidate_features.parquet"
        )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(feature_path, index=False)
        _write_json(
            feature_path.parent / "feature_manifest.json",
            {
                "status": "COMPLETE",
                "model_feature_columns": list(FEATURES),
                "feature_extraction_latency_ms": 0.25,
                "artifact": _record(feature_path),
            },
        )
        label_path = run / "03_features" / f"candidate_labels_crog_{split}_top5.parquet"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(labels).to_parquet(label_path, index=False)
        candidate_path = run / "02_candidates" / f"crog_{split}_top5.parquet"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows)[["sample_id", "candidate_id", "native_rank"]].to_parquet(
            candidate_path, index=False
        )
    folds = pd.DataFrame(
        {"sample_id": train_ids, "fold": [index % 5 for index in range(len(train_ids))]}
    )
    fold_path = run / "04_splits" / "fold_assignments.parquet"
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(fold_path, index=False)
    validation_manifest = run / "01_manifests" / "paired_validation.parquet"
    validation_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": validation_ids}).to_parquet(
        validation_manifest, index=False
    )
    return run


def test_matrix_cell_persists_telemetry_and_resume_rejects_child_tamper(
    tmp_path: Path,
) -> None:
    run = _build_training_run(tmp_path)
    result = train_matrix_cell(_training_args(run))
    assert all(np.isfinite(float(result[field])) for field in TELEMETRY_FIELDS)
    assert result["parameter_count"] > 0
    assert result["feature_extraction_latency_ms"] == pytest.approx(0.25)
    assert 0 < result["missing_feature_rate"] < 1
    source_identity = result["configuration"]["source_identity"]
    assert source_identity["train_feature_columns"] == list(FEATURES)
    assert source_identity["validation_feature_columns"] == list(FEATURES)
    assert len(source_identity["tool_sha256"]) == 64
    marker = (
        run / "07_validation" / "matrix_cells" / result["cell_key"] / "manifest.json"
    )
    mtime = marker.stat().st_mtime_ns
    resumed = train_matrix_cell(_training_args(run))
    assert resumed["cell_key"] == result["cell_key"]
    assert marker.stat().st_mtime_ns == mtime
    Path(result["artifacts"]["predictions"]["path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        train_matrix_cell(_training_args(run))


def test_matrix_cell_rejects_train_validation_feature_schema_drift(
    tmp_path: Path,
) -> None:
    run = _build_training_run(tmp_path)
    manifest_path = (
        run
        / "03_features"
        / "tracks"
        / "T2_matched_common"
        / "crog_validation"
        / "feature_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_feature_columns"] = [*FEATURES, "new_schema_column"]
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="feature schemas differ"):
        train_matrix_cell(_training_args(run))


def test_ablation_metrics_are_recomputed_from_verified_predictions(
    tmp_path: Path,
) -> None:
    run = _build_training_run(tmp_path)
    result = train_matrix_cell(_training_args(run))
    marker = (
        run / "07_validation" / "matrix_cells" / result["cell_key"] / "manifest.json"
    )
    persisted = json.loads(marker.read_text(encoding="utf-8"))
    assert (
        _verified_validation_metrics(run, persisted, marker)["j_at_1"]
        == persisted["metrics"]["j_at_1"]
    )
    persisted["metrics"]["j_at_1"] = float(persisted["metrics"]["j_at_1"]) + 0.1
    _write_json(marker, persisted)
    with pytest.raises(RuntimeError, match="metrics do not match"):
        _verified_validation_metrics(run, persisted, marker)


def _build_selected_run(root: Path) -> Path:
    run = root / "ablation_run"
    selection_table = run / "07_validation" / "tables" / "selected.csv"
    selection_table.parent.mkdir(parents=True, exist_ok=True)
    selection_table.write_text("route\n", encoding="utf-8")
    selections: dict[str, Any] = {}
    for route in ROUTES:
        cell_records = []
        for seed in SEEDS:
            cell_root = run / "07_validation" / "matrix_cells" / f"{route}-{seed}"
            source = cell_root / "source.bin"
            model = cell_root / "model.bin"
            prediction = cell_root / "prediction.bin"
            decision = cell_root / "decision.bin"
            cell_root.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"source-{route}-{seed}".encode())
            model.write_bytes(b"model")
            prediction.write_bytes(b"prediction")
            decision.write_bytes(b"decision")
            cell_path = cell_root / "manifest.json"
            _write_json(
                cell_path,
                {
                    "status": "COMPLETE",
                    "configuration": {
                        "route": route,
                        "track": "T2_matched_common",
                        "encoder": "linear",
                        "loss": "ranknet",
                        "seed": seed,
                        "mode": "validation",
                        "held_fold": None,
                        "learning_rate": 0.001,
                        "weight_decay": 0.0,
                        "alpha": 0.5,
                        "temperature": 1.0,
                        "beta": 1.0,
                        "epochs": 1,
                        "patience": 1,
                        "batch_size": 16,
                        "num_leaves": 7,
                        "tree_learning_rate": 0.05,
                        "n_estimators": 5,
                        "num_attention_blocks": 1,
                    },
                    "feature_columns": list(FEATURES),
                    "metrics": {"j_at_1": 0.5, "mrr_at_5": 0.7},
                    "sources": {"features": _record(source)},
                    "artifacts": {
                        "model": _record(model),
                        "predictions": _record(prediction),
                        "decisions": _record(decision),
                    },
                },
            )
            cell_records.append(_record(cell_path))
        ensemble_path = run / "07_validation" / "ensembles" / route / "manifest.json"
        _write_json(
            ensemble_path,
            {
                "status": "COMPLETE",
                "sources": {"matrix_manifests": cell_records},
            },
        )
        selections[route] = {
            "primary_track": "T2_matched_common",
            "encoder": "linear",
            "loss": "ranknet",
            "validation_manifest": str(ensemble_path.resolve()),
            "validation_manifest_sha256": sha256_file(ensemble_path),
        }
    selection_path = run / "07_validation" / "selected_primary_ungated.json"
    _write_json(
        selection_path,
        {
            "status": "VALIDATION_LOCKED",
            "selections": selections,
            "table": _record(selection_table),
        },
    )
    return run


def _stub_cell_runner(output_values: dict[str, int]):
    def run(args: Namespace, **kwargs: Any) -> dict[str, Any]:
        included = tuple(kwargs["feature_columns_override"])
        contract = kwargs["analysis_contract"]
        key = canonical_sha256(
            {
                "route": args.route,
                "seed": args.seed,
                "included": included,
                "contract": contract,
            }
        )[:16]
        output = Path(kwargs["output_parent"]) / key
        source = output / "source.bin"
        model = output / "model.bin"
        prediction = output / "prediction.bin"
        decision = output / "decision.bin"
        output.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"source")
        model.write_bytes(b"model")
        prediction.write_bytes(b"prediction")
        decision.write_bytes(b"decision")
        result = {
            "status": "COMPLETE",
            "cell_key": key,
            "configuration": {"mode": "validation"},
            "feature_columns": list(included),
            "metrics": {
                "j_at_1": 0.4 + 0.01 * len(included) + args.seed * 1e-6,
                "mrr_at_5": 0.6 + 0.01 * len(included),
            },
            "parameter_count": len(included),
            "ranker_latency_ms": 0.1,
            "feature_latency_ms": 0.2,
            "peak_memory_mb": 1.0,
            "missing_feature_rate": 0.0,
            "sources": {"features": _record(source)},
            "artifacts": {
                "model": _record(model),
                "predictions": _record(prediction),
                "decisions": _record(decision),
            },
        }
        _write_json(output / "manifest.json", result)
        output_values[key] = output_values.get(key, 0) + 1
        return result

    return run


def _registry_tables(run: Path) -> dict[str, pd.DataFrame]:
    ablations = pd.concat(
        [
            pd.read_csv(
                run / "07_validation/ablations/cumulative_feature_ablation.csv"
            ).assign(
                source_file=str(
                    (
                        run / "07_validation/ablations/cumulative_feature_ablation.csv"
                    ).resolve()
                )
            ),
            pd.read_csv(
                run / "07_validation/ablations/leave_one_family_out_ablation.csv"
            ).assign(
                source_file=str(
                    (
                        run
                        / "07_validation/ablations/leave_one_family_out_ablation.csv"
                    ).resolve()
                )
            ),
        ],
        ignore_index=True,
    )
    return {
        "evidence_track_comparison.csv": pd.DataFrame(
            {"route": ROUTES, "track": "T2_matched_common", "j_at_1": 0.5}
        ),
        "loss_comparison.csv": pd.DataFrame(),
        "encoder_comparison.csv": pd.DataFrame(),
        "feature_ablation.csv": ablations,
        "gate_comparison.csv": pd.DataFrame(),
        "attribution_bridge.csv": pd.DataFrame(),
        "cross_route_router.csv": pd.DataFrame(),
        "union_pool.csv": pd.DataFrame(),
        "statistical_tests.csv": pd.DataFrame(),
        "runtime_complexity.csv": pd.DataFrame(
            [
                {
                    "phase": phase,
                    "cell_kind": "matrix_cells",
                    "parameter_count": 10,
                    "ranker_latency_ms": 0.1,
                    "feature_extraction_latency_ms": 0.2,
                    "feature_latency_ms": 0.2,
                    "cell_load_preprocess_latency_ms": 0.05,
                    "peak_memory_mb": 1.0,
                    "missing_feature_rate": 0.0,
                    "artifact_records_verified": True,
                }
                for phase in ("screen", "selected", "encoder")
            ]
        ),
    }


def test_validation_ablation_producer_feeds_registry_and_fails_on_mutation(
    tmp_path: Path,
) -> None:
    run = _build_selected_run(tmp_path)
    executions: dict[str, int] = {}
    result = run_ablations(
        run,
        cell_runner=_stub_cell_runner(executions),
        metric_evaluator=lambda _run, cell, _path: cell["metrics"],
    )
    assert result["status"] == "COMPLETE"
    assert result["candidate_test_labels_read"] is False
    assert len(executions) == 72  # 3 routes x 8 family analyses x 3 seeds.
    tables = _registry_tables(run)
    registry = required_analysis_registry(run, tables)
    assert registry["checks"]["cumulative_feature_ablation"]["status"] == "PASS"
    assert registry["checks"]["leave_one_family_out_ablation"]["status"] == "PASS"
    assert registry["checks"]["runtime_complexity_telemetry"]["status"] == "PASS"

    null_tables = {
        **tables,
        "runtime_complexity.csv": tables["runtime_complexity.csv"].copy(),
    }
    null_tables["runtime_complexity.csv"].loc[0, "ranker_latency_ms"] = np.nan
    null_registry = required_analysis_registry(run, null_tables)
    assert null_registry["checks"]["runtime_complexity_telemetry"]["status"] == "FAIL"

    produced_manifest = Path(result["artifacts"]["cell_manifests"][0]["path"])
    produced = json.loads(produced_manifest.read_text(encoding="utf-8"))
    Path(produced["artifacts"]["model"]["path"]).write_bytes(b"mutated")
    mutated = required_analysis_registry(run, tables)
    assert mutated["checks"]["cumulative_feature_ablation"]["status"] == "FAIL"
    assert mutated["checks"]["leave_one_family_out_ablation"]["status"] == "FAIL"
