from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.feature_cache import common_asset_records
from unified_reranking.ledger import initialize_ledger, ledger_stage, render_ledger_commands
from unified_reranking.metrics import (
    compare_selections,
    evaluate_order_only,
)
from unified_reranking.postformal_reporting import hash_inventory
from unified_reranking.candidates import regenerate_candidate_contract_hashes
from unified_reranking.datasets import FoldPreprocessor
from unified_reranking.pipeline_status import (
    active_legacy_ranker_workers,
    assert_stage_ready,
    audit_pipeline_readiness,
)
from unified_reranking.prelock import assemble_prelock_bundle
from unified_reranking.prelock_validation import (
    validate_feature_extraction_benchmark,
)
from unified_reranking.test_bridge import build_label_free_test_bridge
from unified_reranking.telemetry import TELEMETRY_FIELDS
from unified_reranking.telemetry import resolved_track_extraction_latency
from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS
from unified_reranking.cross_route_inputs import router_feature_columns
from tools.unified_reranking import (
    select_primary_rankers,
    select_validation_screen,
    train_union_rankers,
)
from tools.unified_reranking import train_matrix
from tools.unified_reranking.run_validation_feature_ablations import (
    run as run_validation_feature_ablations,
)
from tools.unified_reranking.prepare_gate_inputs import run as prepare_gate_inputs
from tools.unified_reranking.prepare_route_router_inputs import (
    run as prepare_route_router_inputs,
)
from tools.unified_reranking.analyze_union_headroom import run as run_union_headroom
from tools.unified_reranking.select_gate import run_gate_selection
from tools.unified_reranking.select_route_router import run_route_router_selection
from tools.unified_reranking.apply_locked_test_gates import (
    run as run_gate_test_application,
)
from tools.unified_reranking.apply_locked_route_router import (
    run as run_router_test_application,
)
from tools.unified_reranking.apply_locked_union_ranker import (
    run as run_union_test_application,
)
from tools.unified_reranking.prepare_union_features import (
    run_split as run_union_feature_split,
)


ROUTES = ("crog", "g1", "c1")
SEEDS = (42, 123, 2026)
FEATURES = ["native_score_raw", "overall_feature_reliability"]
MATRIX_FEATURES = ["native_score_raw", "p_center", "overall_feature_reliability"]


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_synthetic_lightgbm_model(path: Path, feature_count: int) -> None:
    script = """
import pickle
import sys
from pathlib import Path

import numpy as np

from tools.unified_reranking.apply_locked_matrix_cell import _load_native_lightgbm_ranker
from unified_reranking.models import LightGBMLambdaRank

feature_count = int(sys.argv[2])
model = LightGBMLambdaRank(
    seed=42, n_estimators=1, min_child_samples=1, num_leaves=2
).fit(
    np.asarray(
        [
            np.zeros(feature_count),
            np.ones(feature_count),
            np.full(feature_count, 0.25),
            np.full(feature_count, 0.75),
        ],
        dtype=float,
    ),
    [0, 1, 0, 1],
    ["q0", "q0", "q1", "q1"],
)
with Path(sys.argv[1]).open("wb") as stream:
    pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
_load_native_lightgbm_ranker(Path(sys.argv[1]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(feature_count)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def _telemetry(phase: str = "synthetic") -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": 1,
        "phase": phase,
        "parameter_count": 1,
        "ranker_latency_ms": 0.1,
        "feature_latency_ms": 0.1,
        "peak_memory_mb": 1.0,
        "missing_feature_rate": 0.0,
        "not_applicable_fields": [],
        "measurement_protocols": {field: "synthetic" for field in TELEMETRY_FIELDS},
    }
    return values


def _complete_cell(
    path: Path,
    *,
    configuration: dict[str, object],
    feature_columns: list[str] | None = None,
    j_at_1: float = 0.5,
) -> None:
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.bin"
    model = root / "model.bin"
    predictions = root / "predictions.bin"
    decisions = root / "decisions.bin"
    for artifact, payload in (
        (source, b"source"),
        (model, b"model"),
        (predictions, b"predictions"),
        (decisions, b"decisions"),
    ):
        artifact.write_bytes(payload)
    telemetry = _telemetry()
    _json(
        path,
        {
            "status": "COMPLETE",
            "configuration": configuration,
            "feature_columns": feature_columns or FEATURES,
            "telemetry": telemetry,
            **{field: telemetry[field] for field in TELEMETRY_FIELDS},
            "metrics": {"j_at_1": j_at_1, "mrr_at_5": j_at_1},
            "sources": {"input": _record(source)},
            "artifacts": {
                "model": _record(model),
                "predictions": _record(predictions),
                "decisions": _record(decisions),
            },
        },
    )


def _command_configuration(command: tuple[str, ...]) -> dict[str, object]:
    module = command[command.index("-m") + 1]
    options: dict[str, str] = {}
    index = command.index(module) + 1
    while index < len(command):
        options[command[index][2:].replace("-", "_")] = command[index + 1]
        index += 2
    integer_fields = {
        "seed",
        "fold",
        "epochs",
        "patience",
        "batch_size",
        "num_leaves",
        "n_estimators",
        "num_attention_blocks",
    }
    float_fields = {
        "learning_rate",
        "weight_decay",
        "alpha",
        "temperature",
        "beta",
        "tree_learning_rate",
        "l2",
    }
    configuration: dict[str, object] = {
        key: (
            int(value)
            if key in integer_fields
            else float(value)
            if key in float_fields
            else value
        )
        for key, value in options.items()
        if key != "run_dir"
    }
    configuration["held_fold"] = configuration.pop("fold", None)
    if module == "tools.unified_reranking.run_interpretable_rule":
        configuration.update({"encoder": "rule", "loss": "interpretable_rule"})
    return configuration


def _matrix_feature_manifest(run: Path, route: str, track: str, split: str) -> Path:
    return (
        run
        / "03_features/tracks"
        / track
        / f"{route}_{split}/feature_manifest.json"
    )


def _ensure_matrix_feature_manifests(run: Path) -> None:
    for route in ROUTES:
        for track in ("T1_native", "T2_matched_common", "T3_tri_backend"):
            for split in ("train", "validation"):
                path = _matrix_feature_manifest(run, route, track, split)
                if path.is_file():
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value.get("model_feature_columns") == MATRIX_FEATURES:
                        continue
                artifact = path.with_name("candidate_features.parquet")
                artifact.parent.mkdir(parents=True, exist_ok=True)
                candidates = pd.read_parquet(
                    run / f"02_candidates/{route}_{split}_top5.parquet",
                    columns=["sample_id", "candidate_id", "native_rank"],
                )
                candidates["native_score_raw"] = 1.0 / candidates["native_rank"]
                candidates["base_logit"] = -candidates["native_rank"].astype(float)
                candidates["p_center"] = 0.8
                candidates["overall_feature_reliability"] = 1.0
                candidates["calibrated_native_probability"] = 0.5
                candidates["peak_retention_rate"] = 1.0
                candidates["perturbed_valid_fraction"] = 1.0
                candidates["mask_reliability"] = 1.0
                candidates.to_parquet(artifact, index=False)
                value = {
                    "status": "COMPLETE",
                    "route": route,
                    "split": split,
                    "track": track,
                    "model_feature_columns": MATRIX_FEATURES,
                    "artifact": _record(artifact),
                }
                if track == "T3_tri_backend":
                    value["feature_track_assembly_latency_ms"] = 0.25
                else:
                    value["feature_extraction_latency_ms"] = 0.5
                _json(path, value)


def _matrix_source_identity(
    run: Path, route: str, track: str, *, validation: bool
) -> tuple[dict[str, object], dict[str, object]]:
    train_manifest_path = _matrix_feature_manifest(run, route, track, "train")
    validation_manifest_path = _matrix_feature_manifest(
        run, route, track, "validation"
    )
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    validation_manifest = json.loads(
        validation_manifest_path.read_text(encoding="utf-8")
    )
    train_artifact = train_manifest["artifact"]
    validation_artifact = validation_manifest["artifact"]
    train_labels = run / f"03_features/candidate_labels_{route}_train_top5.parquet"
    validation_labels = (
        run / f"03_features/candidate_labels_{route}_validation_top5.parquet"
    )
    folds = run / "04_splits/fold_assignments.parquet"
    tool_path = Path(__file__).resolve().parents[2] / (
        "tools/unified_reranking/train_matrix_cell.py"
    )
    tool_record = _record(tool_path)
    benchmark_path = (
        run / "07_validation/telemetry/feature_extraction_benchmark.json"
    )
    benchmark_record = _record(benchmark_path) if track == "T3_tri_backend" else None
    identity: dict[str, object] = {
        "train_features_sha256": train_artifact["sha256"],
        "train_feature_manifest_sha256": sha256_file(train_manifest_path),
        "train_feature_columns": MATRIX_FEATURES,
        "train_feature_schema_sha256": canonical_sha256(MATRIX_FEATURES),
        "train_labels_sha256": sha256_file(train_labels),
        "folds_sha256": sha256_file(folds),
        "selected_feature_columns": MATRIX_FEATURES,
        "selected_feature_schema_sha256": canonical_sha256(MATRIX_FEATURES),
        "tool_sha256": tool_record["sha256"],
        "training_code_sha256": canonical_sha256([tool_record]),
    }
    if benchmark_record is not None:
        identity["train_feature_extraction_benchmark_sha256"] = benchmark_record[
            "sha256"
        ]
    if validation:
        identity.update(
            {
                "validation_features_sha256": validation_artifact["sha256"],
                "validation_feature_manifest_sha256": sha256_file(
                    validation_manifest_path
                ),
                "validation_feature_columns": MATRIX_FEATURES,
                "validation_feature_schema_sha256": canonical_sha256(
                    MATRIX_FEATURES
                ),
                "validation_labels_sha256": sha256_file(validation_labels),
                "validation_denominator_sha256": sha256_file(
                    run / "01_manifests/paired_validation.parquet"
                ),
            }
        )
        if benchmark_record is not None:
            identity["validation_feature_extraction_benchmark_sha256"] = (
                benchmark_record["sha256"]
            )
    sources: dict[str, object] = {
        "train_feature_manifest": _record(train_manifest_path),
        "train_feature_extraction_benchmark": benchmark_record,
        "train_labels": _record(train_labels),
        "folds": _record(folds),
        "training_tool": tool_record,
    }
    if validation:
        sources.update(
            {
                "validation_feature_manifest": _record(validation_manifest_path),
                "validation_feature_extraction_benchmark": benchmark_record,
                "validation_labels": _record(validation_labels),
                "validation_denominator": _record(
                    run / "01_manifests/paired_validation.parquet"
                ),
            }
        )
    return identity, sources


def _write_replayable_matrix_cell(
    run: Path,
    path: Path,
    configuration: dict[str, object],
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    route = str(configuration["route"])
    track = str(configuration["track"])
    mode = str(configuration["mode"])
    validation = mode == "validation"
    source_identity, sources = _matrix_source_identity(
        run, route, track, validation=validation
    )
    if "method" in configuration:
        sources["prediction_feature_manifest"] = sources[
            "validation_feature_manifest"
        ]
        sources["prediction_feature_extraction_benchmark"] = sources[
            "validation_feature_extraction_benchmark"
        ]
        sources["prediction_labels"] = sources["validation_labels"]
        source_identity.update(
            {
                "prediction_feature_manifest_sha256": source_identity[
                    "validation_feature_manifest_sha256"
                ],
                "prediction_features_sha256": source_identity[
                    "validation_features_sha256"
                ],
                "prediction_labels_sha256": source_identity[
                    "validation_labels_sha256"
                ],
                "prediction_feature_extraction_benchmark_sha256": (
                    source_identity.get(
                        "validation_feature_extraction_benchmark_sha256"
                    )
                ),
            }
        )
    configuration = {**configuration, "source_identity": source_identity}
    cell_key = canonical_sha256(configuration)[:16]
    if path.parent.name != cell_key:
        path = path.parent.parent / cell_key / "manifest.json"
    split = "validation" if validation else "train"
    candidates = pd.read_parquet(
        run / f"02_candidates/{route}_{split}_top5.parquet",
        columns=["sample_id", "candidate_id", "native_rank"],
    )
    labels = pd.read_parquet(
        run / f"03_features/candidate_labels_{route}_{split}_top5.parquet",
        columns=["sample_id", "candidate_id", "candidate_success"],
    )
    if validation:
        denominator = pd.read_parquet(
            run / "01_manifests/paired_validation.parquet", columns=["sample_id"]
        )["sample_id"].astype(str).tolist()
    else:
        fold = int(configuration["held_fold"])
        assignments = pd.read_parquet(
            run / "04_splits/fold_assignments.parquet"
        )
        denominator = assignments.loc[
            assignments["fold"].astype(int).eq(fold), "sample_id"
        ].astype(str).tolist()
        wanted = set(denominator)
        candidates = candidates.loc[
            candidates["sample_id"].astype(str).isin(wanted)
        ].copy()
        labels = labels.loc[labels["sample_id"].astype(str).isin(wanted)].copy()
    predictions = candidates[["sample_id", "candidate_id"]].copy()
    predictions["score"] = predictions["candidate_id"].map({"a": 0.1, "b": 0.9})
    evaluation = candidates.merge(
        labels, on=["sample_id", "candidate_id"], validate="one_to_one"
    ).merge(predictions, on=["sample_id", "candidate_id"], validate="one_to_one")
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score"
    )
    artifact_root = run / "05_models/synthetic_matrix_artifacts" / route / mode
    fold_name = "validation" if validation else f"fold-{configuration['held_fold']}"
    prediction_path = artifact_root / fold_name / "predictions.parquet"
    decision_path = artifact_root / fold_name / "decisions.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    if not prediction_path.is_file():
        predictions.to_parquet(prediction_path, index=False)
        decisions.to_parquet(decision_path, index=False)
    feature_manifest = json.loads(
        _matrix_feature_manifest(run, route, track, split).read_text(encoding="utf-8")
    )
    extraction_latency, _ = resolved_track_extraction_latency(
        run, track, feature_manifest
    )
    telemetry = _telemetry("synthetic_matrix")
    telemetry["feature_latency_ms"] = extraction_latency
    selected_features = list(feature_columns or MATRIX_FEATURES)
    value: dict[str, object] = {
        "status": "COMPLETE",
        "cell_key": cell_key,
        "configuration": configuration,
        "feature_columns": selected_features,
        "feature_schema_sha256": canonical_sha256(tuple(selected_features)),
        "feature_extraction_latency_ms": extraction_latency,
        "telemetry": telemetry,
        **{field: telemetry[field] for field in TELEMETRY_FIELDS},
        "metrics": metrics,
        "sources": sources,
        "artifacts": {
            "predictions": _record(prediction_path),
            "decisions": _record(decision_path),
        },
    }
    _json(path, value)
    return {"cell_key": cell_key, **value}


def _build_matrix_phase(
    run: Path, phase: str, *, selection_path: Path | None = None
) -> None:
    jobs = train_matrix.build_jobs(
        Namespace(
            run_dir=run,
            phase=phase,
            routes=list(ROUTES),
            tracks=["T1_native", "T2_matched_common", "T3_tri_backend"],
            selection_json=selection_path,
        )
    )
    records_by_identifier: dict[str, dict[str, str]] = {}
    for job in jobs:
        configuration = _command_configuration(job.command)
        identity, _ = _matrix_source_identity(
            run,
            str(configuration["route"]),
            str(configuration["track"]),
            validation=str(configuration["mode"]) == "validation",
        )
        configuration["source_identity"] = identity
        cell_key = canonical_sha256(configuration)[:16]
        manifest_path = run / f"05_models/{phase}_cells/{cell_key}/manifest.json"
        written = _write_replayable_matrix_cell(
            run, manifest_path, configuration
        )
        manifest_path = (
            run
            / f"05_models/{phase}_cells/{written['cell_key']}/manifest.json"
        )
        records_by_identifier[job.identifier] = _record(manifest_path)
    selection_record = _record(selection_path) if selection_path is not None else None
    plan_path = run / f"05_models/matrix_plans/{phase}_synthetic.json"
    _json(
        plan_path,
        {
            "status": "PLANNED",
            "phase": phase,
            "routes": list(ROUTES),
            "tracks": ["T1_native", "T2_matched_common", "T3_tri_backend"],
            "job_count": len(jobs),
            "selection": selection_record,
            "planner_tool": _record(Path(train_matrix.__file__)),
            "jobs": [job.serialize() for job in jobs],
        },
    )
    results = [
        {
            "identifier": job.identifier,
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "output_manifest": records_by_identifier[job.identifier],
        }
        for job in sorted(jobs, key=lambda item: item.identifier)
    ]
    _json(
        run / f"05_models/matrix_plans/{phase}_latest_execution.json",
        {
            "status": "COMPLETE",
            "phase": phase,
            "plan": _record(plan_path),
            "selection": selection_record,
            "job_count": len(jobs),
            "results": results,
            "output_manifests": [result["output_manifest"] for result in results],
        },
    )


def _build_screen_selected_evidence(run: Path) -> Path:
    _build_matrix_phase(run, "screen")
    screen_manifest = select_validation_screen.run(run)
    _json(run / "07_validation/screen_selection_manifest.json", screen_manifest)
    finalist_path = run / "05_models/screen_finalists.json"
    _build_matrix_phase(run, "selected", selection_path=finalist_path)
    select_primary_rankers.run(run)
    return run / "07_validation/selected_primary_ungated.json"


def _build_encoder_and_ablation_evidence(
    run: Path, selection_path: Path
) -> None:
    encoder_selection_path = run / "05_models/encoder_loss_selections.json"
    _build_matrix_phase(run, "encoder", selection_path=encoder_selection_path)

    budget_fields = {
        "learning_rate",
        "weight_decay",
        "alpha",
        "temperature",
        "beta",
        "epochs",
        "patience",
        "batch_size",
        "num_leaves",
        "tree_learning_rate",
        "n_estimators",
        "num_attention_blocks",
    }

    def synthetic_cell_runner(
        args: Namespace,
        *,
        feature_columns_override: list[str],
        output_parent: Path,
        analysis_contract: dict[str, object],
    ) -> dict[str, object]:
        configuration = {
            "route": args.route,
            "track": args.track,
            "encoder": args.encoder,
            "loss": args.loss,
            "seed": args.seed,
            "mode": args.mode,
            "held_fold": args.fold,
            **{
                name: getattr(args, name)
                for name in budget_fields
                if hasattr(args, name)
            },
            "analysis_contract": analysis_contract,
        }
        return _write_replayable_matrix_cell(
            run,
            output_parent / "pending" / "manifest.json",
            configuration,
            feature_columns=list(feature_columns_override),
        )

    run_validation_feature_ablations(
        run, cell_runner=synthetic_cell_runner
    )


def _build_feature_extraction_benchmark(run: Path) -> None:
    components = (
        "common/crog",
        "rgb/crog",
        "T1_native/crog",
        "T2_matched_common/crog",
        "common/g1",
        "rgb/g1",
        "T1_native/g1",
        "T2_matched_common/g1",
        "common/c1",
        "rgb/c1",
        "T1_native/c1",
        "T2_matched_common/c1",
        "backend_maps/g1",
        "backend_maps/c1",
    )
    full_rows_by_route = {
        route: len(pd.read_parquet(run / f"02_candidates/{route}_validation_top5.parquet"))
        for route in ROUTES
    }
    paired_sample_ids = (
        pd.read_parquet(
            run / "01_manifests/paired_validation.parquet", columns=["sample_id"]
        )["sample_id"]
        .astype(str)
        .iloc[:128]
    )
    benchmark_ids = set(paired_sample_ids)
    rows_by_route = {
        route: int(
            pd.read_parquet(
                run / f"02_candidates/{route}_validation_top5.parquet",
                columns=["sample_id"],
            )["sample_id"]
            .astype(str)
            .isin(benchmark_ids)
            .sum()
        )
        for route in ROUTES
    }
    component_records: dict[str, dict[str, str]] = {}
    measurements: list[dict[str, object]] = []
    for name in components:
        component, route = name.split("/", 1)
        if component in {"T1_native", "T2_matched_common"}:
            manifest_path = (
                run
                / "03_features/tracks"
                / component
                / f"{route}_validation/feature_manifest.json"
            )
        else:
            manifest_path = (
                run
                / "03_features"
                / component
                / f"{route}_validation/feature_manifest.json"
            )
        artifact = manifest_path.with_name("candidate_features.parquet")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        component_candidates = pd.read_parquet(
            run / f"02_candidates/{route}_validation_top5.parquet",
            columns=["sample_id", "candidate_id", "native_rank"],
        )
        component_candidates["native_score_raw"] = (
            1.0 / component_candidates["native_rank"]
        )
        component_candidates["base_logit"] = -component_candidates[
            "native_rank"
        ].astype(float)
        component_candidates["p_center"] = 0.8
        component_candidates["overall_feature_reliability"] = 1.0
        component_candidates["calibrated_native_probability"] = 0.5
        component_candidates["peak_retention_rate"] = 1.0
        component_candidates["perturbed_valid_fraction"] = 1.0
        component_candidates["mask_reliability"] = 1.0
        component_candidates.to_parquet(artifact, index=False)
        _json(
            manifest_path,
            {
                "status": "COMPLETE",
                "route": route,
                "split": "validation",
                "candidate_rows": full_rows_by_route[route],
                "candidate_features": full_rows_by_route[route],
                "model_feature_columns": MATRIX_FEATURES,
                "feature_extraction_latency_ms": 0.5,
                "feature_extraction_peak_memory_mb": 12.0,
                "artifact": _record(artifact),
            },
        )
        component_records[name] = _record(manifest_path)
        measurements.append(
            {
                "name": name,
                "route": route,
                "component_or_track": component,
                "measurement_scope": "full_validation_persisted_extraction",
                "candidate_rows": full_rows_by_route[route],
                "feature_extraction_latency_ms": 0.5,
                "peak_memory_mb": 12.0,
                "manifest": component_records[name],
            }
        )
    tagged_root = run / "03_features/tri_backend_dense/validation_latency_benchmark_128"
    tagged_artifacts: dict[str, dict[str, str]] = {}
    for route in ROUTES:
        artifact = tagged_root / route / "candidate_features.parquet"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": ["synthetic"], "candidate_id": [route]}).to_parquet(
            artifact, index=False
        )
        tagged_artifacts[route] = _record(artifact)
    tagged_manifest_path = tagged_root / "manifest.json"
    _json(
        tagged_manifest_path,
        {
            "status": "COMPLETE",
            "split": "validation",
            "tag": "latency_benchmark_128",
            "candidate_test_labels_read": None,
            "artifacts": tagged_artifacts,
        },
    )
    paired_path = run / "01_manifests/paired_validation.parquet"
    candidate_records = {
        route: _record(run / f"02_candidates/{route}_validation_top5.parquet")
        for route in ROUTES
    }
    extractor = (
        Path(__file__).resolve().parents[2]
        / "tools/unified_reranking/extract_tri_backend_dense_features.py"
    )
    benchmark_tool = (
        Path(__file__).resolve().parents[2]
        / "tools/unified_reranking/benchmark_feature_extraction_latency.py"
    )
    sources = {
        "paired_validation": _record(paired_path),
        "candidate_pools": candidate_records,
        "extractor_tool": _record(extractor),
        "benchmark_tool": _record(benchmark_tool),
        "component_manifests": component_records,
    }
    sample_ids = pd.read_parquet(paired_path, columns=["sample_id"])[
        "sample_id"
    ].astype(str).tolist()
    total_rows = sum(rows_by_route.values())
    configuration = {
        "split": "validation",
        "sample_limit": 128,
        "tag": "latency_benchmark_128",
        "device": "cpu",
        "batch_size": 8,
        "chunk_size": 32,
        "sample_identity_sha256": canonical_sha256(sample_ids[:128]),
        "candidate_rows_by_route": rows_by_route,
        "total_candidate_rows": total_rows,
        "candidate_test_labels_read": False,
        "component_inventory": list(components),
    }
    command = [
        "/synthetic/python",
        "-m",
        "tools.unified_reranking.extract_tri_backend_dense_features",
        "--run-dir",
        str(run.resolve()),
        "--fair-test-source",
        str((run / "synthetic_fair_source").resolve()),
        "--split",
        "validation",
        "--device",
        "cpu",
        "--batch-size",
        "8",
        "--chunk-size",
        "32",
        "--limit",
        "128",
        "--tag",
        "latency_benchmark_128",
    ]
    elapsed = 1.28
    payload = {
        "status": "COMPLETE",
        "schema_version": 1,
        "analysis": "validation_feature_extraction_runtime",
        "source_signature_sha256": canonical_sha256(
            {"configuration": configuration, "sources": sources}
        ),
        "configuration": configuration,
        "command": command,
        "feature_extraction_elapsed_seconds": elapsed,
        "feature_extraction_latency_ms": elapsed * 1000.0 / total_rows,
        "feature_extraction_latency_ms_per_sample": elapsed * 1000.0 / 128,
        "peak_memory_mb": 64.0,
        "measurement_protocol": "synthetic Validation-only fixed-128 protocol",
        "component_measurements": measurements,
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": {"tagged_output_manifest": _record(tagged_manifest_path)},
    }
    payload["content_sha256"] = canonical_sha256(payload)
    _json(
        run / "07_validation/telemetry/feature_extraction_benchmark.json",
        payload,
    )


def _build_gate_and_router_selections(run: Path, samples: pd.DataFrame) -> None:
    del samples
    prepare_gate_inputs(run)
    gate_input_root = run / "07_validation/gate_inputs"
    grid_path = gate_input_root / "synthetic_grid.json"
    _json(
        grid_path,
        {
            "operating_points": [
                {
                    "lambda_harm": 1.0,
                    "utility_threshold": 10.0,
                    "score_margin_threshold": 10.0,
                    "reliability_threshold": 0.0,
                    "stability_threshold": 0.0,
                    "minimum_seed_votes": 2,
                }
            ]
        },
    )
    for route in ROUTES:
        route_input = gate_input_root / route
        run_gate_selection(
            oof_path=route_input / "train_oof.parquet",
            validation_path=route_input / "validation.parquet",
            grid_path=grid_path,
            output_dir=run / "08_lock/gates" / route,
            route=route,
            feature_columns=SAFE_GATE_FEATURE_COLUMNS,
            input_manifest_path=route_input / "manifest.json",
        )
    router_input_manifest = prepare_route_router_inputs(run)
    router_input_root = run / "07_validation/route_router_inputs"
    g1_features = router_feature_columns("G1")
    c1_features = router_feature_columns("C1")
    router_grid_path = router_input_root / "synthetic_grid.json"
    _json(
        router_grid_path,
        {
            "operating_points": [
                {
                    "lambda_router": 1.0,
                    "utility_threshold": 10.0,
                    "margin_threshold": 10.0,
                    "reliability_threshold": 0.0,
                    "stability_threshold": 0.0,
                }
            ]
        },
    )
    run_route_router_selection(
        oof_path=Path(router_input_manifest["artifacts"]["train_oof"]["path"]),
        validation_path=Path(
            router_input_manifest["artifacts"]["validation"]["path"]
        ),
        grid_path=router_grid_path,
        output_dir=run / "08_lock/route_router",
        g1_feature_columns=g1_features,
        c1_feature_columns=c1_features,
        input_manifest_path=router_input_root / "manifest.json",
    )


def _build_run(
    root: Path,
    *,
    include_bridges: bool = True,
    positive_union: bool = False,
    run_name: str = "run",
) -> tuple[Path, Path, Path, Path]:
    run = root / run_name
    for directory in (
        "00_audit",
        "01_manifests",
        "02_candidates",
        "03_features",
        "04_splits",
        "05_calibration",
        "05_models",
        "07_validation",
        "08_lock",
        "09_formal_test",
        "11_attribution_bridge",
        "configs",
    ):
        (run / directory).mkdir(parents=True, exist_ok=True)
    _json(
        run / "manifest.json",
        {
            "schema_version": 1,
            "status": "IN_PROGRESS",
            "test_label_state": "LOCKED_PREVALIDATION",
            "formal_test_execution_count": 0,
        },
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "scene_id": ["scene0", "scene1", "scene2"],
            "frame_id": ["frame0", "frame1", "frame2"],
            "question_index": [0, 1, 2],
            "language": ["pick zero", "pick one", "pick two"],
            "source_rgb_path": ["rgb0", "rgb1", "rgb2"],
            "source_depth_path": ["depth0", "depth1", "depth2"],
        }
    )
    asset_root = root / "synthetic_assets"
    asset_root.mkdir()
    for column in (
        "source_rgb_path",
        "source_depth_path",
        "predicted_mask_path",
        "predicted_probability_path",
    ):
        paths = []
        digests = []
        for index in range(len(samples)):
            path = asset_root / f"{column}_{index}.bin"
            path.write_bytes(f"{column}-{index}".encode())
            paths.append(str(path.resolve()))
            digests.append(sha256_file(path))
        samples[column] = paths
        samples[column.replace("_path", "_sha256")] = digests
    samples["language_sha256"] = [
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in samples["language"].astype(str)
    ]
    train_samples = pd.concat(
        [samples.iloc[[index % len(samples)]].copy() for index in range(10)],
        ignore_index=True,
    )
    train_samples["sample_id"] = [f"train-{index}" for index in range(10)]
    train_samples["scene_id"] = [f"train-scene-{index}" for index in range(10)]
    train_samples["frame_id"] = [f"train-frame-{index}" for index in range(10)]
    train_samples["question_index"] = list(range(10))
    train_samples["language"] = [f"pick train {index}" for index in range(10)]
    train_samples["language_sha256"] = [
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in train_samples["language"].astype(str)
    ]
    samples.to_parquet(run / "01_manifests/paired_test.parquet", index=False)
    train_samples.to_parquet(run / "01_manifests/paired_train.parquet", index=False)
    samples.to_parquet(run / "01_manifests/paired_validation.parquet", index=False)
    pd.DataFrame(
        {
            "sample_id": train_samples["sample_id"],
            "fold": [index // 2 for index in range(10)],
        }
    ).to_parquet(
        run / "04_splits/fold_assignments.parquet", index=False
    )
    evaluator = run / "configs/canonical_evaluator.py"
    evaluator.write_text("# frozen synthetic evaluator\n", encoding="utf-8")
    labels = root / "candidate_test_labels.parquet"
    # Deliberately not a valid Parquet payload: P11 must only hash these bytes.
    labels.write_bytes(b"opaque-candidate-test-label-bytes")
    code = root / "synthetic_pipeline.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")

    selections: dict[str, object] = {}
    for route in ROUTES:
        candidate_samples = (
            ["s0", "s1", "s2"]
            if route == "crog" or positive_union
            else ["s0", "s1"]
        )
        rows = []
        for sample_id in candidate_samples:
            for rank, candidate in ((1, "a"), (2, "b")):
                geometry = [float(rank), 2.0, -10.0 + rank, 4.0, 2.0]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "native_rank": rank,
                        "native_score": 1.0 / rank,
                        "cx_px": geometry[0],
                        "cy_px": geometry[1],
                        "theta_deg": geometry[2],
                        "width_px": geometry[3],
                        "height_px": geometry[4],
                        "candidate_geometry_sha256": canonical_sha256(
                            [
                                route.upper(),
                                sample_id,
                                candidate,
                                rank,
                                *geometry,
                            ]
                        ),
                    }
                )
        candidates = pd.DataFrame(rows)
        candidates["route"] = route.upper()
        candidates["split"] = "test"
        candidates.to_parquet(
            run / f"02_candidates/{route}_test_all.parquet", index=False
        )
        candidates.to_parquet(
            run / f"02_candidates/{route}_test_top5.parquet", index=False
        )
        for development_split in ("train", "validation"):
            if development_split == "train":
                train_rows: list[dict[str, object]] = []
                for sample_id in train_samples["sample_id"].astype(str):
                    for rank, candidate in ((1, "a"), (2, "b")):
                        geometry = [float(rank), 2.0, -10.0 + rank, 4.0, 2.0]
                        train_rows.append(
                            {
                                "sample_id": sample_id,
                                "candidate_id": candidate,
                                "native_rank": rank,
                                "native_score": 1.0 / rank,
                                "cx_px": geometry[0],
                                "cy_px": geometry[1],
                                "theta_deg": geometry[2],
                                "width_px": geometry[3],
                                "height_px": geometry[4],
                                "candidate_geometry_sha256": canonical_sha256(
                                    [
                                        route.upper(),
                                        sample_id,
                                        candidate,
                                        rank,
                                        *geometry,
                                    ]
                                ),
                                "route": route.upper(),
                            }
                        )
                development_candidates = pd.DataFrame(train_rows)
            else:
                development_candidates = candidates.copy()
            development_candidates["split"] = development_split
            development_candidates.to_parquet(
                run / f"02_candidates/{route}_{development_split}_all.parquet",
                index=False,
            )
            development_candidates.to_parquet(
                run / f"02_candidates/{route}_{development_split}_top5.parquet",
                index=False,
            )
            development_labels = pd.DataFrame(
                {
                    "sample_id": development_candidates["sample_id"],
                    "candidate_id": development_candidates["candidate_id"],
                    "candidate_success": False,
                }
            )
            if development_split == "train":
                numeric_ids = development_labels["sample_id"].str.rsplit(
                    "-", n=1
                ).str[-1].astype(int)
                even = numeric_ids.mod(2).eq(0)
                if route in {"g1", "c1"}:
                    development_labels.loc[
                        even & development_labels["candidate_id"].eq("a"),
                        "candidate_success",
                    ] = True
                    development_labels.loc[
                        ~even & development_labels["candidate_id"].eq("b"),
                        "candidate_success",
                    ] = True
                else:
                    development_labels.loc[
                        even & development_labels["candidate_id"].eq("b"),
                        "candidate_success",
                    ] = True
                    development_labels.loc[
                        ~even & development_labels["candidate_id"].eq("a"),
                        "candidate_success",
                    ] = True
            elif positive_union:
                if route == "crog":
                    development_labels.loc[
                        development_labels["sample_id"].isin(["s0", "s2"])
                        & development_labels["candidate_id"].eq("b"),
                        "candidate_success",
                    ] = True
                elif route == "g1":
                    development_labels.loc[
                        development_labels["sample_id"].eq("s1")
                        & development_labels["candidate_id"].eq("b"),
                        "candidate_success",
                    ] = True
                    development_labels.loc[
                        development_labels["sample_id"].eq("s0")
                        & development_labels["candidate_id"].eq("a"),
                        "candidate_success",
                    ] = True
                else:
                    development_labels.loc[
                        development_labels["sample_id"].eq("s0")
                        & development_labels["candidate_id"].eq("b"),
                        "candidate_success",
                    ] = True
                    development_labels.loc[
                        development_labels["sample_id"].eq("s1")
                        & development_labels["candidate_id"].eq("a"),
                        "candidate_success",
                    ] = True
            else:
                development_labels.loc[
                    development_labels["sample_id"].eq("s0")
                    & development_labels["candidate_id"].eq("b"),
                    "candidate_success",
                ] = True
                development_labels.loc[
                    development_labels["sample_id"].eq("s1")
                    & development_labels["candidate_id"].eq("a"),
                    "candidate_success",
                ] = True
            development_labels["jacquard_margin"] = development_labels[
                "candidate_success"
            ].astype(float)
            development_labels.to_parquet(
                run
                / f"03_features/candidate_labels_{route}_{development_split}_top5.parquet",
                index=False,
            )

        calibration = run / f"05_calibration/{route}_calibration_manifest.json"
        _json(calibration, {"status": "COMPLETE", "selected_method": "isotonic"})
        _json(
            run / f"05_calibration/{route}_test_application_manifest.json",
            {"status": "COMPLETE_LABEL_FREE", "candidate_labels_loaded": False},
        )
        feature_manifest = (
            run
            / f"03_features/tracks/T2_matched_common/{route}_test/feature_manifest.json"
        )
        feature_artifact = feature_manifest.parent / "candidate_features.parquet"
        feature_artifact.parent.mkdir(parents=True, exist_ok=True)
        candidate_features = candidates[
            ["sample_id", "candidate_id", "native_rank"]
        ].copy()
        candidate_features["native_score_raw"] = 1.0 / candidate_features["native_rank"]
        candidate_features["base_logit"] = -candidate_features["native_rank"].astype(
            float
        )
        candidate_features["p_center"] = 0.8
        candidate_features["overall_feature_reliability"] = 1.0
        candidate_features["calibrated_native_probability"] = 0.5
        candidate_features["peak_retention_rate"] = 1.0
        candidate_features["perturbed_valid_fraction"] = 1.0
        candidate_features["mask_reliability"] = 1.0
        candidate_features.to_parquet(feature_artifact, index=False)
        if route in {"g1", "c1"}:
            common = run / f"03_features/common/{route}_test/candidate_features.parquet"
            common.parent.mkdir(parents=True, exist_ok=True)
            bridge_features = candidates[
                ["sample_id", "candidate_id", "native_rank", "native_score"]
            ].rename(columns={"native_score": "native_score_raw"})
            bridge_features["p_center"] = 0.8
            bridge_features.to_parquet(common, index=False)
        _json(
            feature_manifest,
            {
                "status": "COMPLETE",
                "labels_physically_separate": True,
                "model_feature_columns": MATRIX_FEATURES,
                "artifact": _record(feature_artifact),
            },
        )

        matrix_records = []
        for seed in SEEDS:
            matrix_path = (
                run / f"07_validation/matrix_cells/{route}-{seed}/manifest.json"
            )
            _complete_cell(
                matrix_path,
                configuration={
                        "route": route,
                        "track": "T2_matched_common",
                        "encoder": "mlp",
                        "loss": "ranknet",
                        "seed": seed,
                        "mode": "validation",
                        "held_fold": None,
                        "learning_rate": 0.0001,
                        "weight_decay": 0.0,
                        "alpha": 0.5,
                },
            )
            matrix_records.append(_record(matrix_path))
        validation_predictions = candidates[
            ["sample_id", "candidate_id", "native_rank"]
        ].copy()
        validation_predictions["ensemble_score"] = validation_predictions[
            "candidate_id"
        ].map({"a": 0.1, "b": 0.9})
        validation_labels = pd.read_parquet(
            run / f"03_features/candidate_labels_{route}_validation_top5.parquet"
        )
        validation_evaluation = validation_predictions.merge(
            validation_labels,
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
        denominator = samples["sample_id"].astype(str).tolist()
        validation_metrics, validation_decisions = evaluate_order_only(
            denominator,
            validation_evaluation,
            score_column="ensemble_score",
        )
        native_evaluation = validation_evaluation.copy()
        native_evaluation["native_control_score"] = -native_evaluation[
            "native_rank"
        ].astype(float)
        native_metrics, native_decisions = evaluate_order_only(
            denominator,
            native_evaluation,
            score_column="native_control_score",
        )
        comparison = compare_selections(
            native_decisions,
            validation_decisions,
            oracle_at_5=float(native_metrics["oracle_at_5"]),
        )
        ensemble_path = run / f"07_validation/ensembles/{route}/manifest.json"
        ensemble_path.parent.mkdir(parents=True, exist_ok=True)
        validation_prediction_path = ensemble_path.parent / "per_candidate_scores.parquet"
        validation_decision_path = ensemble_path.parent / "per_sample_decisions.parquet"
        validation_predictions.to_parquet(validation_prediction_path, index=False)
        validation_decisions.to_parquet(validation_decision_path, index=False)
        _json(
            ensemble_path,
            {
                "status": "COMPLETE",
                "ensemble_id": route,
                "identity": {
                    "route": route,
                    "track": "T2_matched_common",
                    "method_code": "R4_mlp_ranknet",
                    "encoder": "mlp",
                    "loss": "ranknet",
                    "seeds": list(SEEDS),
                },
                "metrics": validation_metrics,
                "sources": {
                    "matrix_manifests": matrix_records,
                    "candidates": _record(
                        run / f"02_candidates/{route}_validation_top5.parquet"
                    ),
                    "labels": _record(
                        run
                        / f"03_features/candidate_labels_{route}_validation_top5.parquet"
                    ),
                },
                "artifacts": {
                    "predictions": _record(validation_prediction_path),
                    "decisions": _record(validation_decision_path),
                },
            },
        )
        oof_path = run / f"06_oof/ensembles/{route}/manifest.json"
        oof_artifact = oof_path.parent / "oof.bin"
        oof_artifact.parent.mkdir(parents=True, exist_ok=True)
        oof_artifact.write_bytes(b"synthetic-oof")
        _json(
            oof_path,
            {
                "status": "COMPLETE",
                "split": "train",
                "ensemble_id": route,
                "identity": {
                    "route": route,
                    "track": "T2_matched_common",
                    "method_code": "R4_mlp_ranknet",
                    "encoder": "mlp",
                    "loss": "ranknet",
                    "seeds": list(SEEDS),
                },
                "sources": {"validation": _record(ensemble_path)},
                "artifacts": {"synthetic": _record(oof_artifact)},
            },
        )
        selections[route] = {
            "primary_track": "T2_matched_common",
            "method_code": "R4_mlp_ranknet",
            "encoder": "mlp",
            "loss": "ranknet",
            "ensemble_id": route,
            "validation_manifest": str(ensemble_path.resolve()),
            "validation_manifest_sha256": sha256_file(ensemble_path),
            "oof_manifest": str(oof_path.resolve()),
            "oof_manifest_sha256": sha256_file(oof_path),
            "selection_metrics": {
                "j_at_1": validation_metrics["j_at_1"],
                "delta_j_at_1": comparison["delta_j_at_1"],
                "recovered": comparison["recovered"],
                "harmful": comparison["harmful"],
                "switch_rate": comparison["switch_rate"],
                "mrr_at_5": validation_metrics["mrr_at_5"],
            },
        }
        predictions = candidates[["sample_id", "candidate_id", "native_rank"]].copy()
        for seed in SEEDS:
            predictions[f"score_seed_{seed}"] = predictions["candidate_id"].map(
                {"a": 0.1, "b": 0.9}
            )
        predictions["ensemble_score"] = predictions[
            [f"score_seed_{seed}" for seed in SEEDS]
        ].mean(axis=1)
        ranker_dir = run / f"08_lock/label_free_test_rankers/{route}"
        ranker_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = ranker_dir / "per_candidate_scores.parquet"
        decision_path = ranker_dir / "per_sample_decisions.parquet"
        predictions.to_parquet(prediction_path, index=False)
        candidate_exists = [True, True, route == "crog" or positive_union]
        selected_ids = ["b" if exists else "" for exists in candidate_exists]
        native_ids = ["a" if exists else "" for exists in candidate_exists]
        geometry_by_key = candidates.set_index(["sample_id", "candidate_id"])[
            "candidate_geometry_sha256"
        ]
        selected_geometry = [
            str(geometry_by_key.loc[(sample_id, candidate_id)])
            if exists
            else ""
            for sample_id, candidate_id, exists in zip(
                samples["sample_id"], selected_ids, candidate_exists, strict=True
            )
        ]
        pd.DataFrame(
            {
                "sample_id": samples["sample_id"],
                "selected_candidate_id": selected_ids,
                "candidate_count": [2 if exists else 0 for exists in candidate_exists],
                "native_candidate_id": native_ids,
                "selected_geometry_sha256": selected_geometry,
                "ensemble_score_margin": [
                    0.8 if exists else 0.0 for exists in candidate_exists
                ],
                "seed_challenger_votes": [
                    3 if exists else 0 for exists in candidate_exists
                ],
                "challenger_exists": candidate_exists,
            }
        ).to_parquet(decision_path, index=False)
        _json(
            ranker_dir / "manifest.json",
            {
                "status": "COMPLETE",
                "route": route,
                "candidate_test_labels_read": False,
                "label_free_test_inference": True,
                "artifacts": {
                    "predictions": _record(prediction_path),
                    "decisions": _record(decision_path),
                },
            },
        )

        gate_test_dir = run / f"08_lock/label_free_test_gates/{route}"
        gate_test_dir.mkdir(parents=True, exist_ok=True)
        gate_decisions = gate_test_dir / "gate_test_decisions.parquet"
        pd.DataFrame(
            {
                "sample_id": samples["sample_id"],
                "selected_candidate_id": [
                    "a" if exists else "" for exists in candidate_exists
                ],
            }
        ).to_parquet(gate_decisions, index=False)
        _json(
            gate_test_dir / "manifest.json",
            {
                "status": "COMPLETE",
                "decision": "NO_GO_NATIVE",
                "candidate_test_labels_read": False,
                "artifacts": {"decisions": _record(gate_decisions)},
            },
        )

    fair_run = root / "audited_fair_source"
    (fair_run / "02_predictions").mkdir(parents=True)
    generator_sources = root / "generator_sources"
    generator_sources.mkdir()
    checkpoint_source = generator_sources / "checkpoint.bin"
    selected_config_source = generator_sources / "selected_config.json"
    inference_source = generator_sources / "locked_inference.py"
    checkpoint_source.write_bytes(b"synthetic-checkpoint")
    _json(selected_config_source, {"synthetic": True})
    inference_source.write_text("# synthetic locked inference\n", encoding="utf-8")
    inference_code_records = [
        {
            "path": str(inference_source.resolve()),
            "sha256": sha256_file(inference_source),
            "bytes": inference_source.stat().st_size,
        }
    ]
    inference_code_bundle_sha256 = canonical_sha256(inference_code_records)
    lineage_constants = {
        "checkpoint_sha256": sha256_file(checkpoint_source),
        "selected_config_sha256": sha256_file(selected_config_source),
        "native_decoder_config_sha256": sha256_file(inference_source),
    }
    fair_artifacts: dict[str, object] = {}
    initial_pools: dict[str, object] = {}
    for route in ROUTES:
        canonical_test = pd.read_parquet(
            run / f"02_candidates/{route}_test_all.parquet"
        )
        fair_source = canonical_test[
            [
                "sample_id",
                "candidate_id",
                "native_rank",
                "native_score",
                "cx_px",
                "cy_px",
                "theta_deg",
                "width_px",
                "height_px",
            ]
        ].rename(
            columns={
                "width_px": "jaw_width_px",
                "height_px": "rectangle_height_px",
            }
        )
        for name, value in lineage_constants.items():
            fair_source[name] = value
        fair_source_path = (
            fair_run / f"02_predictions/{route}_native_predictions.parquet"
        )
        fair_source.to_parquet(fair_source_path, index=False)
        fair_artifacts[f"{route}_candidates"] = _record(fair_source_path)
        if route in {"g1", "c1"}:
            lineage = {
                "source_kind": "audited_fair_native_predictions",
                "source": str(fair_source_path.resolve()),
                "source_sha256": sha256_file(fair_source_path),
            }
            for pool in ("all", "top5"):
                initial_pools[f"{route.upper()}/test/{pool}"] = {"lineage": lineage}

    crog_source_root = root / "audited_crog_sources"
    crog_source_root.mkdir()
    split_source_name = {"train": "train", "validation": "val", "test": "test"}
    context = (
        pd.concat([samples, train_samples], ignore_index=True)
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )
    for split in ("train", "validation", "test"):
        canonical = pd.read_parquet(run / f"02_candidates/crog_{split}_all.parquet")
        legacy_rows = []
        for row in canonical.itertuples(index=False):
            sample = context.loc[str(row.sample_id)]
            legacy_rows.append(
                {
                    "sample_id": f"legacy:{int(sample.question_index)}",
                    "scene_id": sample.scene_id,
                    "language_instruction": sample.language,
                    "image_path": sample.source_rgb_path,
                    "depth_path": sample.source_depth_path,
                    "candidate_id": row.candidate_id,
                    "original_rank": int(row.native_rank),
                    "q_raw": float(row.native_score),
                    "x_px": float(row.cx_px),
                    "y_px": float(row.cy_px),
                    "angle_rad": float(np.radians(row.theta_deg)),
                    "width_px": float(row.width_px),
                    "height_px": float(row.height_px),
                }
            )
        source_path = (
            crog_source_root
            / f"candidates_crog_frozen_top5_{split_source_name[split]}.parquet"
        )
        pd.DataFrame(legacy_rows).to_parquet(source_path, index=False)
        lineage = {
            "source_kind": "audited_crog_legacy",
            "source": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
        }
        for pool in ("all", "top5"):
            initial_pools[f"CROG/{split}/{pool}"] = {"lineage": lineage}

    for route in ("g1", "c1"):
        for split in ("train", "validation"):
            split_samples = train_samples if split == "train" else samples
            canonical = pd.read_parquet(
                run / f"02_candidates/{route}_{split}_all.parquet"
            )
            source = canonical[
                [
                    "sample_id",
                    "candidate_id",
                    "native_rank",
                    "native_score",
                    "cx_px",
                    "cy_px",
                    "theta_deg",
                    "width_px",
                    "height_px",
                ]
            ].rename(
                columns={
                    "width_px": "jaw_width_px",
                    "height_px": "rectangle_height_px",
                }
            )
            for name, value in lineage_constants.items():
                source[name] = value
            source_root = run / f"02_candidates/native_work/{route}_{split}"
            source_root.mkdir(parents=True)
            source_path = source_root / "candidates.parquet"
            source.to_parquet(source_path, index=False)
            per_sample_path = source_root / "per_sample.parquet"
            split_samples[["sample_id"]].to_parquet(per_sample_path, index=False)
            shard_root = source_root / "shards/00000"
            shard_root.mkdir(parents=True)
            shard_candidates = shard_root / "candidates.parquet"
            shard_per_sample = shard_root / "per_sample.parquet"
            source.to_parquet(shard_candidates, index=False)
            split_samples[["sample_id"]].to_parquet(
                shard_per_sample, index=False
            )

            def generated_record(path: Path) -> dict[str, object]:
                return {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "rows": int(pd.read_parquet(path).shape[0]),
                }

            shard_artifacts = {
                "per_sample": generated_record(shard_per_sample),
                "candidates": generated_record(shard_candidates),
            }
            source_samples_path = run / f"01_manifests/paired_{split}.parquet"
            shared_manifest_fields = {
                "method": route.upper(),
                "split": split,
                "tag": "formal",
                "device": "cpu",
                "locked_inference_source_sha256": sha256_file(inference_source),
                "inference_code_bundle_sha256": inference_code_bundle_sha256,
                "source_samples_sha256": sha256_file(source_samples_path),
                "checkpoint_sha256": sha256_file(checkpoint_source),
                "selected_config_sha256": sha256_file(selected_config_source),
                "native_decoder_config_sha256": sha256_file(inference_source),
                "save_raw_maps": False,
            }
            shard_manifest_path = shard_root / "manifest.json"
            shard_manifest = {
                "status": "COMPLETE",
                "start": 0,
                "stop": len(split_samples),
                "sample_rows": len(split_samples),
                "sample_identity_sha256": canonical_sha256(
                    split_samples["sample_id"].tolist()
                ),
                "input_asset_identity_sha256": canonical_sha256(
                    common_asset_records(split_samples.to_dict("records"), {})
                ),
                "artifacts": shard_artifacts,
                **shared_manifest_fields,
            }
            _json(shard_manifest_path, shard_manifest)
            manifest_path = source_root / "run_manifest.json"
            _json(
                manifest_path,
                {
                    "status": "COMPLETE",
                    **lineage_constants,
                    **shared_manifest_fields,
                    "source_samples": str(source_samples_path.resolve()),
                    "checkpoint": str(checkpoint_source.resolve()),
                    "selected_config": str(selected_config_source.resolve()),
                    "locked_inference_source": str(inference_source.resolve()),
                    "inference_code_records": inference_code_records,
                    "sample_count": len(split_samples),
                    "candidate_count": len(source),
                    "shards": 1,
                    "artifacts": {
                        "per_sample": generated_record(per_sample_path),
                        "candidates": generated_record(source_path),
                    },
                    "chunk_manifests": [
                        {
                            "path": str(shard_manifest_path.resolve()),
                            "sha256": sha256_file(shard_manifest_path),
                            "start": 0,
                            "stop": len(split_samples),
                            "sample_identity_sha256": shard_manifest[
                                "sample_identity_sha256"
                            ],
                            "artifacts": shard_artifacts,
                        }
                    ],
                },
            )
            lineage = {
                "source_kind": "generated_native_work",
                "source": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "run_manifest": str(manifest_path.resolve()),
                "run_manifest_sha256": sha256_file(manifest_path),
                "locked_inference_source_sha256": sha256_file(inference_source),
            }
            for pool in ("all", "top5"):
                initial_pools[f"{route.upper()}/{split}/{pool}"] = {
                    "lineage": lineage
                }
    _json(
        run / "00_audit/fair_source_audit.json",
        {"status": "PASS", "fair_run": str(fair_run.resolve()), "artifacts": fair_artifacts},
    )
    _json(run / "02_candidates/candidate_contract_hashes.json", {"pools": initial_pools})

    _build_feature_extraction_benchmark(run)
    _ensure_matrix_feature_manifests(run)
    selection_path = _build_screen_selected_evidence(run)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selections = selection_payload["selections"]
    _build_encoder_and_ablation_evidence(run, selection_path)
    _build_gate_and_router_selections(run, samples)
    selection_record = _record(selection_path)
    for route in ROUTES:
        path = run / f"08_lock/label_free_test_rankers/{route}/manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["selection"] = selections[route]
        validation_ensemble = json.loads(
            Path(selections[route]["validation_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        validation_cells: dict[int, dict[str, str]] = {}
        for record in validation_ensemble["sources"]["matrix_manifests"]:
            cell = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
            validation_cells[int(cell["configuration"]["seed"])] = record
        applications: dict[str, dict[str, str]] = {}
        for seed in SEEDS:
            application_path = (
                run
                / "09_formal_test"
                / "label_free_cell_predictions"
                / f"{route}-{seed}"
                / "manifest.json"
            )
            application_artifact = application_path.parent / "predictions.parquet"
            application_artifact.parent.mkdir(parents=True, exist_ok=True)
            seed_predictions = pd.read_parquet(
                run / f"02_candidates/{route}_test_top5.parquet",
                columns=["sample_id", "candidate_id", "native_rank"],
            )
            seed_predictions["score"] = seed_predictions["candidate_id"].map(
                {"a": 0.1, "b": 0.9}
            )
            seed_predictions.to_parquet(application_artifact, index=False)
            _json(
                application_path,
                {
                    "status": "COMPLETE",
                    "label_free_test_inference": True,
                    "candidate_test_labels_read": False,
                    "identity": {
                        "seed": seed,
                        "route": route,
                        "track": "T2_matched_common",
                    },
                    "sources": {
                        "cell_manifest": validation_cells[seed],
                        "feature_manifest": _record(
                            run
                            / f"03_features/tracks/T2_matched_common/{route}_test/feature_manifest.json"
                        ),
                    },
                    "artifact": _record(application_artifact),
                },
            )
            applications[str(seed)] = _record(application_path)
        ranker_sources = {
            "selection": selection_record,
            "validation_ensemble": {
                "path": selections[route]["validation_manifest"],
                "sha256": selections[route]["validation_manifest_sha256"],
            },
            "candidates": _record(
                run / f"02_candidates/{route}_test_top5.parquet"
            ),
            "denominator": _record(run / "01_manifests/paired_test.parquet"),
            "implementation_tool": _record(
                Path(__file__).resolve().parents[2]
                / "tools/unified_reranking/apply_selected_test_rankers.py"
            ),
            "applications": applications,
        }
        ranker_configuration = {
            "route": route,
            "primary_track": "T2_matched_common",
            "method_code": selections[route]["method_code"],
            "encoder": selections[route]["encoder"],
            "loss": selections[route]["loss"],
            "ensemble_id": selections[route]["ensemble_id"],
            "seeds": list(SEEDS),
            "candidate_test_labels_read": False,
        }
        manifest["sources"] = ranker_sources
        manifest["configuration"] = ranker_configuration
        manifest["signature_sha256"] = canonical_sha256(
            {"configuration": ranker_configuration, "sources": ranker_sources}
        )
        _json(path, manifest)
        gate_manifest_path = (
            run / f"08_lock/label_free_test_gates/{route}/manifest.json"
        )
        gate_manifest_path.unlink()
        run_gate_test_application(run, route)
    _json(
        run / "08_lock/label_free_test_rankers/manifest.json",
        {"status": "COMPLETE", "candidate_test_labels_read": False},
    )

    router_input_root = run / "08_lock/route_router_inputs"
    router_input_root.mkdir(parents=True, exist_ok=True)
    router_input_path = router_input_root / "test_label_free.parquet"
    router_inputs = pd.read_parquet(
        run / "07_validation/route_router_inputs/validation.parquet"
    ).drop(columns=["crog_correct", "g1_correct", "c1_correct"])
    router_inputs["prediction_source"] = "test_label_free"
    for route in ROUTES:
        geometry = pd.read_parquet(
            run / f"02_candidates/{route}_test_top5.parquet"
        ).loc[lambda frame: frame["candidate_id"].eq("a")]
        geometry_by_sample = geometry.set_index("sample_id")[
            "candidate_geometry_sha256"
        ]
        router_inputs[f"{route}_selected_candidate_geometry_sha256"] = (
            router_inputs["sample_id"].map(geometry_by_sample).fillna("")
        )
    router_inputs.to_parquet(router_input_path, index=False)
    _json(
        router_input_root / "manifest.json",
        {
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": {"test_label_free": _record(router_input_path)},
        },
    )
    run_router_test_application(run)
    run_union_headroom(run)
    if include_bridges:
        for route in ("g1", "c1"):
            for split in ("train", "validation"):
                table_path = (
                    run / "11_attribution_bridge" / f"bridge_{route}_{split}_top5.csv"
                )
                table_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "route": [route.upper()],
                        "split": [split],
                        "pool_contract": ["top5"],
                        "selector": ["fair_native"],
                        "j_at_1": [0.5],
                    }
                ).to_csv(table_path, index=False)
                _json(
                    run
                    / f"11_attribution_bridge/bridge_{route}_{split}_top5_manifest.json",
                    {
                        "status": "COMPLETE",
                        "route": route.upper(),
                        "split": split,
                        "test_access": False,
                        "artifacts": {"table": _record(table_path)},
                    },
                )
        historical_root = root / "historical_bridge_run"
        modular_root = root / "modular_bridge_run"
        (historical_root / "data").mkdir(parents=True)
        (historical_root / "audit").mkdir(parents=True)
        (modular_root / "manifests").mkdir(parents=True)
        historical_paths: dict[str, Path] = {}
        inventory: dict[str, object] = {}
        for route in ("g1", "c1"):
            rows = []
            for sample_id in ("s0", "s1"):
                for rank, candidate in ((1, "a"), (2, "b")):
                    quality = 0.9 / rank
                    support = 0.8
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "stable_candidate_id": candidate,
                            "original_rank": rank,
                            "original_score": quality * support,
                            "raw_network_quality": quality,
                            "stored_center_mask_support": support,
                            "center_x": float(rank) + 0.25,
                            "center_y": 2.0,
                            "angle_deg": -10.0 + rank,
                            "width_px": 4.0,
                            "height_px": 2.0,
                            "candidate_identity_sha256": canonical_sha256(
                                [route, sample_id, candidate, rank, "historical"]
                            ),
                            "split": "test",
                            "backend": route.upper(),
                        }
                    )
            historical_frame = pd.DataFrame(rows)
            path = historical_root / f"data/frozen_{route}_test_top5_candidates.parquet"
            historical_frame.to_parquet(path, index=False)
            historical_frame.to_parquet(
                historical_root / f"data/frozen_{route}_test_allnms_candidates.parquet",
                index=False,
            )
            historical_paths[route] = path
            inventory[f"{route.upper()}_test"] = {
                "top5_path": str(path.resolve()),
                "top5_artifact_sha256": sha256_file(path),
            }
        ground_truth = modular_root / "manifests/test_labels.parquet"
        # P11 may hash but must never parse this synthetic opaque source.
        ground_truth.write_bytes(b"opaque-historical-test-ground-truth")
        source_manifest = modular_root / "manifests/experiment_lock.json"
        experiment_lock = {
            "schema_version": 1,
            "lock_status": "LOCKED",
            "effective": True,
            "run_id": "synthetic-modular",
            "run_dir": str(modular_root.resolve()),
            "lock_relative_path": "manifests/experiment_lock.json",
            "marker_relative_path": ".EXPERIMENT_LOCKED",
            "artifacts": {
                "test_labels": {
                    "path": "manifests/test_labels.parquet",
                    "sha256": sha256_file(ground_truth),
                }
            },
        }
        experiment_lock["manifest_content_sha256"] = canonical_sha256(experiment_lock)
        _json(source_manifest, experiment_lock)
        _json(
            modular_root / ".EXPERIMENT_LOCKED",
            {
                "schema_version": 1,
                "lock_status": "LOCKED",
                "run_id": experiment_lock["run_id"],
                "lock_relative_path": "manifests/experiment_lock.json",
                "manifest_content_sha256": experiment_lock["manifest_content_sha256"],
            },
        )
        _json(
            modular_root / "FINALIZATION_COMPLETE.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "experiment_lock_sha256": experiment_lock["manifest_content_sha256"],
            },
        )
        inventory_path = historical_root / "audit/frozen_pool_inventory.json"
        _json(inventory_path, inventory)
        historical_formal_lock = historical_root / "08_lock/FORMAL_TEST_LOCK.json"
        _json(
            historical_formal_lock,
            {
                "status": "LOCKED",
                "base_run": str(modular_root.resolve()),
                "source_label_artifacts": [
                    {
                        "path": str(ground_truth.resolve()),
                        "sha256": sha256_file(ground_truth),
                    }
                ],
                "audit_artifacts": [
                    {
                        "path": str(inventory_path.resolve()),
                        "sha256": sha256_file(inventory_path),
                    }
                ],
                "candidate_artifacts": [
                    {
                        "path": str(
                            (
                                historical_root
                                / f"data/frozen_{route}_test_allnms_candidates.parquet"
                            ).resolve()
                        ),
                        "sha256": sha256_file(
                            historical_root
                            / f"data/frozen_{route}_test_allnms_candidates.parquet"
                        ),
                    }
                    for route in ("g1", "c1")
                ],
            },
        )
        authority_paths = [
            inventory_path,
            historical_formal_lock,
            *[
                historical_root / f"data/frozen_{route}_test_{pool}_candidates.parquet"
                for route in ("g1", "c1")
                for pool in ("top5", "allnms")
            ],
        ]
        run_sha_manifest = historical_root / "RUN_SHA256_MANIFEST.txt"
        run_sha_manifest.write_text(
            "".join(
                f"{sha256_file(path)}  {path.relative_to(historical_root).as_posix()}\n"
                for path in sorted(authority_paths)
            ),
            encoding="utf-8",
        )
        (historical_root / "RUN_LOCK_SHA256.txt").write_text(
            sha256_file(run_sha_manifest) + "\n", encoding="utf-8"
        )
        build_label_free_test_bridge(
            run_dir=run,
            historical_candidates=historical_paths,
            historical_ground_truth_path=ground_truth,
            historical_source_manifest_path=source_manifest,
            historical_inventory_path=inventory_path,
            evaluator_path=evaluator,
        )
    else:
        regenerate_candidate_contract_hashes(run)
    return run, labels, evaluator, code


def test_prelock_assembly_is_hash_only_semantic_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    original_read = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        if Path(path).resolve() == labels.resolve():
            raise AssertionError(
                "candidate-level Test labels were parsed before formal claim"
            )
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    result = assemble_prelock_bundle(
        run,
        candidate_test_labels_path=labels,
        evaluator_path=evaluator,
        code_roots=[code],
    )
    assert result["candidate_test_labels_read"] is False
    assert result["formal_lock_created"] is False
    assert not (run / "08_lock/FORMAL_TEST_LOCK.json").exists()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "VALIDATION_SELECTION_COMPLETE"
    assert manifest["test_label_state"] == "VALIDATION_SELECTION_COMPLETE"
    label_manifest = json.loads(
        (run / "08_lock/candidate_test_label_manifest.json").read_text(encoding="utf-8")
    )
    assert label_manifest["candidate_labels_sha256"] == sha256_file(labels)
    assert label_manifest["normalization"] == {
        "route_column": "method",
        "variant_column": "variant",
        "include_variants": ["crog_native", "g1", "c1"],
    }
    plan = json.loads(
        (run / "08_lock/formal_evaluation_plan.json").read_text(encoding="utf-8")
    )
    required_selection_provenance = {
        "primary_selection",
        "encoder_loss_selection",
        "encoder_latest_execution",
        "feature_ablation_manifest",
        "feature_extraction_benchmark",
    }
    assert required_selection_provenance.issubset(plan["bound_provenance"])
    assert required_selection_provenance.issubset(result["sources"])
    assert len(plan["systems"]) == 10
    assert {system["kind"] for system in plan["systems"]} == {
        "native",
        "ungated",
        "gated",
        "router",
    }
    bridge = pd.read_csv(run / "07_validation/bridge_train_validation.csv")
    assert sha256_file(
        run / "07_validation/bridge_train_validation.csv"
    ) == sha256_file(run / "11_attribution_bridge/bridge_train_validation.csv")
    assert set(map(tuple, bridge[["route", "split"]].to_numpy())) == {
        (route.upper(), split)
        for route in ("g1", "c1")
        for split in ("train", "validation")
    }
    for system in plan["systems"]:
        if system.get("ranking_path"):
            columns = pd.read_parquet(system["ranking_path"]).columns
            assert {
                "candidate_geometry_sha256",
                "frozen_native_rank",
                "rank",
            }.issubset(columns)
    marker = run / "08_lock/prelock_assembly_manifest.json"
    mtime = marker.stat().st_mtime_ns
    resumed = assemble_prelock_bundle(
        run,
        candidate_test_labels_path=labels,
        evaluator_path=evaluator,
        code_roots=[code],
    )
    assert resumed["signature_sha256"] == result["signature_sha256"]
    assert marker.stat().st_mtime_ns == mtime


def test_prelock_refuses_missing_bridge_without_state_transition(
    tmp_path: Path,
) -> None:
    run, labels, evaluator, code = _build_run(tmp_path, include_bridges=False)
    with pytest.raises(ValueError, match="attribution bridge"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_label_state"] == "LOCKED_PREVALIDATION"
    assert not (run / "08_lock/FORMAL_TEST_LOCK.json").exists()


def test_prelock_rejects_incomplete_regenerated_candidate_inventory(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    inventory_path = run / "02_candidates/candidate_contract_hashes.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["pools"].pop("G1/test/top5")
    inventory["status"] = "PARTIAL_AWAITING_G1_C1_DEVELOPMENT_INFERENCE"
    inventory["missing_pools"] = ["G1/test/top5"]
    _json(inventory_path, inventory)
    with pytest.raises(ValueError, match="inventory is incomplete"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_candidate_inventory_regeneration_rejects_source_valid_geometry_edit(
    tmp_path: Path,
) -> None:
    run, _labels, _evaluator, _code = _build_run(tmp_path)
    for pool in ("all", "top5"):
        path = run / f"02_candidates/g1_test_{pool}.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "cx_px"] += 0.5
        row = frame.iloc[0]
        frame.loc[0, "candidate_geometry_sha256"] = canonical_sha256(
            [
                "G1",
                row["sample_id"],
                row["candidate_id"],
                int(row["native_rank"]),
                float(row["cx_px"]),
                float(row["cy_px"]),
                float(row["theta_deg"]),
                float(row["width_px"]),
                float(row["height_px"]),
            ]
        )
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="differs from audited source"):
        regenerate_candidate_contract_hashes(run)


def test_prelock_rejects_hash_valid_selected_ensemble_swap(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    selection_path = run / "07_validation/selected_primary_ungated.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    replacement = run / "07_validation/ensembles/g1/manifest.json"
    selection["selections"]["crog"]["validation_manifest"] = str(replacement.resolve())
    selection["selections"]["crog"]["validation_manifest_sha256"] = sha256_file(
        replacement
    )
    _json(selection_path, selection)
    with pytest.raises(ValueError, match="scalar selected trial binding mismatch"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )
    assert (
        json.loads((run / "manifest.json").read_text(encoding="utf-8"))[
            "test_label_state"
        ]
        == "LOCKED_PREVALIDATION"
    )


def test_prelock_rejects_corrupted_test_feature_artifact(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    feature_path = (
        run
        / "03_features/tracks/T2_matched_common/crog_test/candidate_features.parquet"
    )
    changed = pd.read_parquet(feature_path)
    changed.loc[0, "native_score_raw"] = 999.0
    changed.to_parquet(feature_path, index=False)
    with pytest.raises(RuntimeError, match="candidate features"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_prelock_requires_encoder_and_ablation_evidence(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    (run / "05_models/matrix_plans/encoder_latest_execution.json").unlink()
    with pytest.raises(ValueError, match="encoder latest execution"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_prelock_requires_feature_extraction_benchmark(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    (run / "07_validation/telemetry/feature_extraction_benchmark.json").unlink()
    with pytest.raises(
        RuntimeError,
        match=r"train_feature_extraction_benchmark is missing",
    ):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_feature_benchmark_counts_only_fixed_first_128_samples(
    tmp_path: Path,
) -> None:
    run, _, _, _ = _build_run(tmp_path)
    paired_path = run / "01_manifests/paired_validation.parquet"
    paired = pd.read_parquet(paired_path)
    base_columns = list(paired.columns)
    extras = pd.DataFrame(
        [
            {column: (f"extra-{index}" if column == "sample_id" else None)
             for column in base_columns}
            for index in range(130)
        ]
    )
    pd.concat([paired, extras], ignore_index=True).to_parquet(
        paired_path, index=False
    )
    for route in ROUTES:
        candidate_path = run / f"02_candidates/{route}_validation_top5.parquet"
        candidates = pd.read_parquet(candidate_path)
        template = candidates.iloc[0].copy()
        additions = []
        for index in range(130):
            row = template.copy()
            row["sample_id"] = f"extra-{index}"
            row["candidate_id"] = f"extra-candidate-{index}"
            additions.append(row)
        pd.concat([candidates, pd.DataFrame(additions)], ignore_index=True).to_parquet(
            candidate_path, index=False
        )
    _build_feature_extraction_benchmark(run)
    # The helper intentionally records/counts only the fixed first 128 paired
    # samples even though the candidate pools contain later samples too.
    validate_feature_extraction_benchmark(
        run / "07_validation/telemetry/feature_extraction_benchmark.json"
    )


def test_prelock_rejects_reauthored_feature_benchmark_contract(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    benchmark_path = (
        run / "07_validation/telemetry/feature_extraction_benchmark.json"
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["configuration"]["sample_limit"] = 64
    benchmark["source_signature_sha256"] = canonical_sha256(
        {
            "configuration": benchmark["configuration"],
            "sources": benchmark["sources"],
        }
    )
    benchmark.pop("content_sha256")
    benchmark["content_sha256"] = canonical_sha256(benchmark)
    _json(benchmark_path, benchmark)
    with pytest.raises(
        RuntimeError,
        match=r"train_feature_extraction_benchmark SHA-256 mismatch",
    ):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_feature_benchmark_rejects_reauthored_component_telemetry(
    tmp_path: Path,
) -> None:
    run, _, _, _ = _build_run(tmp_path)
    benchmark_path = (
        run / "07_validation/telemetry/feature_extraction_benchmark.json"
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["component_measurements"][0][
        "feature_extraction_latency_ms"
    ] += 1.0
    benchmark.pop("content_sha256")
    benchmark["content_sha256"] = canonical_sha256(benchmark)
    _json(benchmark_path, benchmark)
    with pytest.raises(ValueError, match="component measurement mismatch"):
        validate_feature_extraction_benchmark(benchmark_path)


def test_prelock_rejects_hash_valid_encoder_cell_substitution(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    execution_path = run / "05_models/matrix_plans/encoder_latest_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    cell_path = Path(execution["output_manifests"][0]["path"])
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["configuration"]["seed"] = 999
    _json(cell_path, cell)
    replacement = _record(cell_path)
    execution["output_manifests"][0] = replacement
    execution["results"][0]["output_manifest"] = replacement
    _json(execution_path, execution)
    with pytest.raises(ValueError, match="planned command"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_prelock_rejects_rehashed_nonfinite_ablation_csv(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    manifest_path = run / "07_validation/ablations/feature_ablation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_path = Path(manifest["artifacts"]["cumulative"]["path"])
    frame = pd.read_csv(csv_path)
    frame.loc[0, "j_at_1"] = np.nan
    frame.to_csv(csv_path, index=False)
    manifest["artifacts"]["cumulative"] = _record(csv_path)
    manifest.pop("content_sha256")
    manifest["content_sha256"] = canonical_sha256(manifest)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="non-finite metrics"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_prelock_rejects_rehashed_lower_j_scalar_winner(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    selection_path = run / "07_validation/selected_primary_ungated.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    table_path = Path(selection["table"]["path"])
    table = pd.read_csv(table_path)
    lower_row = table.loc[
        table["route"].eq("crog")
        & table["track"].eq("T2_matched_common")
        & table["method_code"].eq("R6_lambdamart")
    ].iloc[0]
    selection["selections"]["crog"] = {
        "primary_track": str(lower_row["track"]),
        "method_code": str(lower_row["method_code"]),
        "encoder": str(lower_row["encoder"]),
        "loss": str(lower_row["loss"]),
        "ensemble_id": str(lower_row["ensemble_id"]),
        "validation_manifest": str(lower_row["validation_manifest"]),
        "validation_manifest_sha256": str(
            lower_row["validation_manifest_sha256"]
        ),
        "oof_manifest": str(lower_row["oof_manifest"]),
        "oof_manifest_sha256": str(lower_row["oof_manifest_sha256"]),
        "selection_metrics": {
            "j_at_1": float(lower_row["j_at_1"]),
            "mrr_at_5": float(lower_row["mrr_at_5"]),
            "delta_j_at_1": float(lower_row["delta_j_at_1"]),
            "recovered": int(lower_row["recovered"]),
            "harmful": int(lower_row["harmful"]),
            "switch_rate": float(lower_row["switch_rate"]),
        },
    }
    _json(selection_path, selection)
    with pytest.raises(ValueError, match="not the best bound trial"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_prelock_rejects_rehashed_test_ranker_output_replacement(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    manifest_path = run / "08_lock/label_free_test_rankers/crog/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = Path(manifest["artifacts"]["predictions"]["path"])
    predictions = pd.read_parquet(prediction_path)
    predictions.loc[0, "ensemble_score"] += 100.0
    predictions.to_parquet(prediction_path, index=False)
    manifest["artifacts"]["predictions"] = _record(prediction_path)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="bound seed applications"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


@pytest.mark.parametrize("application_kind", ("gate", "router"))
def test_prelock_rejects_rehashed_policy_output_replacement(
    tmp_path: Path,
    application_kind: str,
) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    if application_kind == "gate":
        manifest_path = run / "08_lock/label_free_test_gates/crog/manifest.json"
    else:
        manifest_path = run / "08_lock/route_router_test/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision_path = Path(manifest["artifacts"]["decisions"]["path"])
    decisions = pd.read_parquet(decision_path)
    decisions.loc[0, "selected_candidate_id"] = "replacement"
    decisions.to_parquet(decision_path, index=False)
    manifest["artifacts"]["decisions"] = _record(decision_path)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match=rf"{application_kind} Test decisions differs"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_prelock_rejects_reauthored_gate_decision(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    selection_path = run / "08_lock/gates/crog/gate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    point = selection["configuration"]["operating_points"][0]
    selection["decision"] = "GO"
    selection["selection"]["status"] = "GO"
    selection["selection"]["selected_operating_point"] = point
    selection.pop("content_sha256")
    selection["content_sha256"] = canonical_sha256(selection)
    _json(selection_path, selection)
    app_path = run / "08_lock/label_free_test_gates/crog/manifest.json"
    app = json.loads(app_path.read_text(encoding="utf-8"))
    app["decision"] = "GO"
    app["selected_operating_point"] = point
    app["sources"]["gate_selection"] = _record(selection_path)
    app["signature_sha256"] = canonical_sha256(
        {
            "route": "crog",
            "feature_columns": app["feature_columns"],
            "operating_point": point,
            "sources": app["sources"],
        }
    )
    _json(app_path, app)
    with pytest.raises(ValueError, match="differs from independent recomputation"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_prelock_rejects_reauthored_router_decision(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path)
    selection_path = run / "08_lock/route_router/route_router_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    point = selection["configuration"]["operating_points"][0]
    selection["decision"] = "GO"
    selection["selection"]["status"] = "GO"
    selection["selection"]["selected_operating_point"] = point
    selection.pop("content_sha256")
    selection["content_sha256"] = canonical_sha256(selection)
    _json(selection_path, selection)
    app_path = run / "08_lock/route_router_test/manifest.json"
    app = json.loads(app_path.read_text(encoding="utf-8"))
    app["configuration"]["selection_decision"] = "GO"
    app["configuration"]["selected_operating_point"] = point
    app["sources"]["router_selection"] = _record(selection_path)
    app["signature_sha256"] = canonical_sha256(
        {"configuration": app["configuration"], "sources": app["sources"]}
    )
    _json(app_path, app)
    with pytest.raises(ValueError, match="differs from independent recomputation"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def _enable_positive_union(run: Path) -> None:
    headroom = run_union_headroom(run)
    assert headroom["decision"] == "UNION_HEADROOM_AVAILABLE"
    for split in ("train", "validation"):
        run_union_feature_split(run, split)
    train_union_rankers.run_orchestrator(run, execute=False)

    train, feature_columns, train_features, train_labels = (
        train_union_rankers._load_development(run, "train")
    )
    folds_path = run / "04_splits/fold_assignments.parquet"
    folds = pd.read_parquet(folds_path, columns=["sample_id", "fold"])
    preprocessor = FoldPreprocessor.fit(train, feature_columns)
    synthetic_lambdamart_path = run / "07_validation/synthetic_union_lambdamart.pkl"
    _write_synthetic_lightgbm_model(
        synthetic_lambdamart_path, len(feature_columns)
    )
    synthetic_lambdamart_bytes = synthetic_lambdamart_path.read_bytes()
    implementation = _record(Path(train_union_rankers.__file__))
    cells: list[dict[str, object]] = []
    for planned in train_union_rankers.formal_plan(
        train_union_rankers.FORMAL_UNION_BUDGET
    ):
        mode = str(planned["mode"])
        held_fold = planned["held_fold"]
        if mode == "oof":
            prediction_frame = train.merge(
                folds, on="sample_id", validate="many_to_one"
            ).loc[lambda frame: frame["fold"].eq(held_fold)]
            prediction_features = train_features
            prediction_labels = train_labels
        else:
            prediction_frame, validation_columns, prediction_features, prediction_labels = (
                train_union_rankers._load_development(run, "validation")
            )
            assert validation_columns == feature_columns
        score = (
            prediction_frame["candidate_success"].astype(float) * 10.0
            - prediction_frame["native_rank"].astype(float)
            if planned["encoder"] == "lambdamart"
            else prediction_frame["base_logit"].astype(float)
        )
        cell_predictions = prediction_frame[["sample_id", "candidate_id"]].copy()
        cell_predictions["score"] = score.to_numpy(float)
        configuration = {
            "pool": "primary_union_top15_no_dedup",
            "track": train_union_rankers.TRACK,
            "encoder": planned["encoder"],
            "loss": (
                "lambdarank"
                if planned["encoder"] == "lambdamart"
                else train_union_rankers.FORMAL_UNION_BUDGET.deepsets_loss
            ),
            "seed": planned["seed"],
            "mode": mode,
            "held_fold": held_fold,
            "early_stop_fold": planned["early_stop_fold"],
            "deterministic_cpu": True,
            "route_candidate_identity_preserved": True,
            "budget": asdict(train_union_rankers.FORMAL_UNION_BUDGET),
            "sources": {
                "train_features": _record(train_features),
                "train_labels": _record(train_labels),
                "prediction_features": _record(prediction_features),
                "prediction_labels": _record(prediction_labels),
                "folds": _record(folds_path),
                "implementation_tool": implementation,
            },
        }
        cell_id = canonical_sha256(configuration)[:16]
        cell_root = (
            run
            / ("06_oof" if mode == "oof" else "07_validation")
            / "union_cells"
            / cell_id
        )
        cell_root.mkdir(parents=True, exist_ok=True)
        prediction_path = cell_root / "predictions.parquet"
        cell_predictions.to_parquet(prediction_path, index=False)
        model_path = cell_root / "model.pkl"
        if planned["encoder"] == "lambdamart":
            model_path.write_bytes(synthetic_lambdamart_bytes)
        else:
            model = DummyRegressor(strategy="constant", constant=0.0).fit(
                np.zeros((2, len(preprocessor.columns))), np.zeros(2)
            )
            with model_path.open("wb") as stream:
                pickle.dump(model, stream)
        cell = {
            "status": "COMPLETE",
            "cell_id": cell_id,
            "configuration": configuration,
            "feature_columns": list(feature_columns),
            "feature_schema_sha256": canonical_sha256(feature_columns),
            "preprocessor": preprocessor.artifact(),
            "artifacts": {
                "model": _record(model_path),
                "predictions": _record(prediction_path),
            },
            "test_access": "NONE",
        }
        _json(cell_root / "manifest.json", cell)
        cells.append(cell)

    train_union_rankers.select_union_ranker(run, cells)
    run_union_test_application(run)


def test_prelock_positive_union_binds_three_top5_pools_without_label_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, labels, evaluator, code = _build_run(tmp_path, positive_union=True)
    _enable_positive_union(run)
    original_read = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        if Path(path).resolve() == labels.resolve():
            raise AssertionError(
                "candidate-level Test labels were parsed before formal claim"
            )
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    assemble_prelock_bundle(
        run,
        candidate_test_labels_path=labels,
        evaluator_path=evaluator,
        code_roots=[code],
    )
    plan = json.loads(
        (run / "08_lock/formal_evaluation_plan.json").read_text(encoding="utf-8")
    )
    union = next(system for system in plan["systems"] if system["kind"] == "union")
    assert plan["union_contract"]["decision"] == "UNION_HEADROOM_AVAILABLE"
    assert union["route"] == "cross_route"
    ranking = pd.read_parquet(union["ranking_path"])
    assert ranking["rank"].between(1, 15).all()
    assert (
        ranking["candidate_id"]
        .eq(ranking["source_route"].str.upper() + ":" + ranking["source_candidate_id"])
        .all()
    )
    assert ranking.groupby("sample_id").size().to_dict() == {
        "s0": 6,
        "s1": 6,
        "s2": 6,
    }
    for snapshot in (
        "selected_methods.json",
        "selected_features.json",
        "selected_hyperparameters.json",
    ):
        assert "union" in json.loads(
            (run / "08_lock" / snapshot).read_text(encoding="utf-8")
        )


def test_prelock_rejects_rehashed_union_output_replacement(tmp_path: Path) -> None:
    run, labels, evaluator, code = _build_run(tmp_path, positive_union=True)
    _enable_positive_union(run)
    manifest_path = run / "08_lock/union_ranker_test/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = Path(manifest["artifacts"]["predictions"]["path"])
    predictions = pd.read_parquet(prediction_path)
    predictions.loc[0, "ensemble_score"] += 1.0
    predictions.to_parquet(prediction_path, index=False)
    manifest["artifacts"]["predictions"] = _record(prediction_path)
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="union Test predictions differs"):
        assemble_prelock_bundle(
            run,
            candidate_test_labels_path=labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )


def test_pipeline_status_detects_only_real_legacy_worker_and_refuses_interlock(
    tmp_path: Path,
) -> None:
    workers = active_legacy_ranker_workers(
        [
            "81876 /opt/python -m reranking.run_experiment_matrix --stage train-worker",
            "81877 caffeinate -dimsu /opt/python -m reranking.run_experiment_matrix --stage train-worker",
            "99999 python -m tools.unified_reranking.pipeline_status --run-dir x",
        ]
    )
    assert [worker["pid"] for worker in workers] == [81876]
    report = audit_pipeline_readiness(tmp_path, process_rows=[])
    assert report["candidate_test_labels_read"] is False
    assert report["stages"]["P2"]["status"] == "READY"
    blocked = {
        "legacy_ranker_workers": workers,
        "stages": {
            "P7": {
                "complete": False,
                "legacy_worker_interlock": True,
                "unmet_dependencies": [],
                "missing_or_invalid": [],
            }
        },
    }
    with pytest.raises(RuntimeError, match="81876"):
        assert_stage_ready(blocked, "P7")


def test_pipeline_dag_exposes_encoder_union_and_acyclic_postformal_stages(
    tmp_path: Path,
) -> None:
    headroom = tmp_path / "07_validation/union_headroom/manifest.json"
    _json(
        headroom,
        {
            "status": "COMPLETE",
            "decision": "UNION_HEADROOM_AVAILABLE",
            "test_access": "NONE",
        },
    )
    report = audit_pipeline_readiness(tmp_path, process_rows=[])
    stages = report["stages"]
    assert stages["P15"]["unmet_dependencies"] == ["P12"]
    assert "P15" in stages["P13"]["unmet_dependencies"]
    assert "P7_ENCODER" in stages["P11_PRELOCK"]["unmet_dependencies"]
    assert any(
        "--phase encoder" in command
        for command in stages["P7_ENCODER"]["next_commands"]
    )
    union_missing = {item["description"] for item in stages["P9"]["missing_or_invalid"]}
    assert "Validation-selected union ranker" in union_missing
    assert "label-free Test union application" in union_missing
    assert any(
        "build_postformal_artifacts" in command
        for command in stages["POSTFORMAL"]["next_commands"]
    )
    p12_missing = {item["description"] for item in stages["P12"]["missing_or_invalid"]}
    assert "formal per-candidate scores" in p12_missing
    assert "formal per-sample decisions" in p12_missing
    assert not any("table" in description.lower() for description in p12_missing)


def test_pipeline_terminal_status_rejects_stale_complete_marker(tmp_path: Path) -> None:
    _json(
        tmp_path / "manifest.json",
        {
            "status": "COMPLETE",
            "test_label_state": "FORMAL_TEST_COMPLETE",
            "formal_test_execution_count": 1,
        },
    )
    ledger_path = initialize_ledger(tmp_path / "run_ledger.sqlite")
    with ledger_stage(
        ledger_path,
        stage="POSTFORMAL",
        substage="synthetic-finalization",
        command="synthetic command",
    ):
        pass
    (tmp_path / "commands.log").write_text(
        render_ledger_commands(ledger_path), encoding="utf-8"
    )
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    inventory = hash_inventory(tmp_path)
    lock = {"schema_version": 1, "status": "COMPLETE", "inventory": inventory}
    lock["self_sha256"] = canonical_sha256(lock)
    lock_path = tmp_path / "FINAL_RUN_LOCK.json"
    _json(lock_path, lock)
    digest = sha256_file(lock_path)
    (tmp_path / "FINAL_RUN_SHA256.txt").write_text(
        f"{digest}  FINAL_RUN_LOCK.json\n", encoding="utf-8"
    )
    (tmp_path / "COMPLETE").write_text(
        f"COMPLETE\nFINAL_RUN_LOCK.json sha256={digest}\n", encoding="utf-8"
    )
    assert (
        audit_pipeline_readiness(tmp_path, process_rows=[])["stages"]["POSTFORMAL"][
            "complete"
        ]
        is True
    )
    (tmp_path / "COMPLETE").write_text(
        "COMPLETE\nFINAL_RUN_LOCK.json sha256=" + "0" * 64 + "\n",
        encoding="utf-8",
    )
    stale = audit_pipeline_readiness(tmp_path, process_rows=[])["stages"]["POSTFORMAL"]
    assert stale["complete"] is False
    assert stale["final_readiness"]["marker_binding_ok"] is False


def test_run_all_refuses_before_writing_when_formal_lock_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.unified_reranking.run_all as run_all

    run = tmp_path / "locked_run"
    (run / "08_lock").mkdir(parents=True)
    (run / "08_lock/FORMAL_TEST_LOCK.json").write_text("{}\n", encoding="utf-8")
    readme = run / "README_REPRODUCE.md"
    readme.write_text("immutable\n", encoding="utf-8")
    before = readme.read_bytes()
    monkeypatch.setattr(
        run_all,
        "parse_args",
        lambda: Namespace(run_dir=run, resume=True),
    )
    with pytest.raises(PermissionError, match="P0-P1 bootstrap"):
        run_all.main()
    assert readme.read_bytes() == before


def test_run_all_generated_readme_is_bootstrap_honest(tmp_path: Path) -> None:
    import tools.unified_reranking.run_all as run_all

    run = tmp_path / "bootstrap"
    run_all.bootstrap(run, resume=False)
    text = (run / "README_REPRODUCE.md").read_text(encoding="utf-8")
    assert "resumes only the P0-P1" in text
    assert "It does not execute the complete experiment" in text
    assert "pipeline_status" in text
    assert "prepare_test_bridge_bundle" in text
    assert "assemble_prelock" in text
    assert "independent_recompute" in text
    assert "build_postformal_artifacts" in text
    assert "candidate_contract_hashes.json" in text
