"""Exact semantic replay for P10 plan, execution, cells, gates, and tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import canonical_sha256
from unified_reranking.metrics import evaluate_order_only

from .ablation import (
    ABLATION_EXECUTION_POINTER_RELATIVE,
    ABLATION_PLAN_RELATIVE,
    EVIDENCE_TRACKS,
    load_ablation_plan,
    track_feature_artifact,
)
from .ablation_execution import (
    ablation_execution_scope,
    execution_directory,
    execution_source_records,
    load_ablation_resource_policy,
)
from .ablation_selection import (
    SELECTION_RELATIVE,
    _denominator,
    _gate_inputs,
    _gate_materials,
    _label_frame,
    aggregate_variant,
    compute_track_gate,
    feature_ablation_rows,
    gate_grid_points,
)
from .execution import artifact_record, load_content_manifest
from .resource_gate import evaluate_resource_gate, resource_thresholds


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _assert_frame(observed: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise RuntimeError(f"{name} semantic replay differs") from error


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
        "sample_universe_sha256": canonical_sha256(
            ordered["sample_id"].astype(str).tolist()
        ),
        "selected_vector_sha256": canonical_sha256(selections),
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def _input_records(
    plan: Mapping[str, Any], *, split: str, track: str
) -> dict[str, Any]:
    feature_manifest_path = verified_artifact_path(
        plan["sources"]["track_manifests"][track][split],
        name=f"D1 P10 {split}/{track} feature manifest",
    )
    candidate_manifest_path = verified_artifact_path(
        plan["sources"]["candidate_manifests"][split],
        name=f"D1 P10 {split} candidate manifest",
    )
    label_manifest_path = verified_artifact_path(
        plan["sources"]["development_label_manifests"][split],
        name=f"D1 P10 {split} label manifest",
    )
    feature_manifest = load_content_manifest(
        feature_manifest_path, name="D1 P10 features", statuses=("COMPLETE",)
    )
    candidate_manifest = load_content_manifest(
        candidate_manifest_path, name="D1 P10 candidates", statuses=("COMPLETE",)
    )
    label_manifest = load_content_manifest(
        label_manifest_path, name="D1 P10 labels", statuses=("COMPLETE",)
    )
    return {
        "feature_manifest": artifact_record(feature_manifest_path),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "label_manifest": artifact_record(label_manifest_path),
        "features": artifact_record(
            track_feature_artifact(
                feature_manifest, track=track, name="D1 P10 features"
            )
        ),
        "candidates": artifact_record(
            verified_artifact_path(
                candidate_manifest["artifacts"]["top5"],
                name="D1 P10 Top5 candidates",
            )
        ),
        "labels": artifact_record(
            verified_artifact_path(label_manifest["artifact"], name="D1 P10 labels")
        ),
    }


def validate_ablation_cell(
    value: Mapping[str, Any],
    *,
    plan_path: str | Path,
    job: Mapping[str, Any],
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Replay one COMPLETE cell without refitting its heavy ranker."""

    result = {str(key): child for key, child in value.items()}
    configuration = _mapping(job.get("configuration"), name="D1 P10 job configuration")
    job_id = str(job.get("job_id", ""))
    if (
        result.get("status") != "COMPLETE"
        or result.get("job_id") != job_id
        or result.get("configuration") != configuration
        or result.get("configuration_sha256") != canonical_sha256(configuration)
        or result.get("candidate_test_labels_read") is not False
        or result.get("test_inputs_referenced") is not False
    ):
        raise RuntimeError("D1 P10 cell/job contract differs")
    plan_source = Path(plan_path).expanduser().resolve()
    plan = load_ablation_plan(plan_source)
    output_path = Path(manifest_path).expanduser().resolve()
    expected_output = (
        plan_source.parent.parent / str(job["output_manifest"])
    ).resolve()
    if output_path != expected_output:
        raise RuntimeError("D1 P10 cell output path differs")
    sources = _mapping(result.get("sources"), name="D1 P10 cell sources")
    required_sources = {
        "plan",
        "execution_authority",
        "execution_event",
        "execution_claim",
        "selected_primary",
        "selected_trial",
        "track_manifests",
        "candidate_manifests",
        "development_label_manifests",
        "calibration_manifest",
        "fold_assignments",
        "denominators",
        "runner",
    }
    if (
        set(sources) != required_sources
        or sources.get("plan") != artifact_record(plan_source)
        or sources.get("selected_primary") != plan["sources"]["selected_primary"]
        or sources.get("selected_trial") != plan["sources"]["selected_trial"]
        or sources.get("track_manifests")
        != plan["sources"]["track_manifests"][configuration["track"]]
        or sources.get("candidate_manifests") != plan["sources"]["candidate_manifests"]
        or sources.get("development_label_manifests")
        != plan["sources"]["development_label_manifests"]
        or sources.get("calibration_manifest")
        != plan["sources"]["calibration_manifest"]
        or sources.get("fold_assignments") != plan["sources"]["fold_assignments"]
        or sources.get("denominators") != plan["sources"]["denominators"]
        or sources.get("runner")
        != artifact_record(
            Path(__file__).resolve().parents[2]
            / "tools/d1_reranking/run_ablation_cell.py"
        )
    ):
        raise RuntimeError("D1 P10 cell source closure differs")
    verify_artifact_records_recursive(
        sources, name="D1 P10 cell sources", require_at_least_one=True
    )
    artifacts = _mapping(result.get("artifacts"), name="D1 P10 cell artifacts")
    if set(artifacts) != {"model", "preprocessor", "predictions", "decisions"}:
        raise RuntimeError("D1 P10 cell artifact inventory differs")
    paths = verify_artifact_records_recursive(
        artifacts, name="D1 P10 cell artifacts", require_at_least_one=True
    )
    if any(Path(record["path"]).parent != output_path.parent for record in paths):
        raise RuntimeError("D1 P10 cell artifact escapes its output directory")
    preprocessor = _mapping(result.get("preprocessor"), name="D1 P10 preprocessor")
    persisted = json.loads(
        verified_artifact_path(
            artifacts["preprocessor"], name="D1 P10 preprocessor artifact"
        ).read_text(encoding="utf-8")
    )
    configuration_columns = tuple(map(str, configuration["feature_columns"]))
    if (
        persisted != preprocessor
        or preprocessor.get("feature_schema_sha256")
        != canonical_sha256(configuration_columns)
        or preprocessor.get("job_id") != job_id
        or preprocessor.get("fold_preprocessor", {}).get("columns")
        != list(configuration_columns)
    ):
        raise RuntimeError("D1 P10 preprocessor/schema binding differs")
    preprocessor_unsigned = dict(preprocessor)
    if preprocessor_unsigned.pop("content_sha256", None) != canonical_sha256(
        preprocessor_unsigned
    ):
        raise RuntimeError("D1 P10 preprocessor content hash differs")
    held_fold = configuration.get("held_fold")
    early_fold = (int(held_fold) + 1) % 5 if held_fold is not None else 0
    excluded = {early_fold} | ({int(held_fold)} if held_fold is not None else set())
    expected_fold_contract = {
        "early_stop_fold": early_fold,
        "fit_fold_ids": [fold for fold in range(5) if fold not in excluded],
        "fold_local_preprocessing": True,
        "fold_local_calibration": True,
    }
    model_contract = _mapping(
        result.get("model_contract"), name="D1 P10 model contract"
    )
    if (
        result.get("fold_contract") != expected_fold_contract
        or model_contract.get("method") != configuration.get("method")
        or model_contract.get("seed") != configuration.get("seed")
        or model_contract.get("effective_training_hyperparameters")
        != configuration.get("training_spec")
        or model_contract.get("feature_schema_sha256")
        != canonical_sha256(configuration_columns)
    ):
        raise RuntimeError("D1 P10 fitted model/fold contract differs")
    expected_train_inputs = _input_records(
        plan, split="train", track=str(configuration["track"])
    )
    expected_validation_inputs = (
        None
        if configuration["mode"] == "oof"
        else _input_records(plan, split="validation", track=str(configuration["track"]))
    )
    if result.get("input_records") != {
        "train": expected_train_inputs,
        "validation": expected_validation_inputs,
    }:
        raise RuntimeError("D1 P10 cell materialized input binding differs")
    predictions = pd.read_parquet(
        verified_artifact_path(artifacts["predictions"], name="D1 P10 predictions")
    )
    expected_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
        "score",
    ]
    if list(predictions.columns) != expected_columns or predictions.empty:
        raise RuntimeError("D1 P10 prediction schema differs")
    split = "train" if configuration["mode"] == "oof" else "validation"
    candidate_manifest = load_content_manifest(
        verified_artifact_path(
            plan["sources"]["candidate_manifests"][split],
            name="D1 P10 candidate manifest",
        ),
        name="D1 P10 candidates",
        statuses=("COMPLETE",),
    )
    candidate_path = verified_artifact_path(
        candidate_manifest["artifacts"]["top5"], name="D1 P10 Top5"
    )
    expected_candidates = pd.read_parquet(candidate_path, columns=expected_columns[:-1])
    if held_fold is not None:
        folds = pd.read_parquet(
            verified_artifact_path(plan["sources"]["fold_assignments"], name="folds"),
            columns=["sample_id", "fold"],
        )
        expected_candidates = expected_candidates.merge(
            folds.loc[folds["fold"].eq(int(held_fold)), ["sample_id"]],
            on="sample_id",
            validate="many_to_one",
        )
    _assert_frame(
        predictions.loc[:, expected_columns[:-1]],
        expected_candidates.sort_values(
            ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
        ).reset_index(drop=True),
        name="D1 P10 candidate universe",
    )
    decisions = pd.read_parquet(
        verified_artifact_path(artifacts["decisions"], name="D1 P10 decisions")
    )
    labels = _label_frame(plan, split)
    evaluation = (
        expected_candidates[["sample_id", "candidate_id", "native_rank"]]
        .merge(
            labels[["sample_id", "candidate_id", "candidate_success"]],
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
        .merge(
            predictions[["sample_id", "candidate_id", "score"]],
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
    )
    if len(evaluation) != len(expected_candidates):
        raise RuntimeError("D1 P10 cell prediction/label membership differs")
    if held_fold is None:
        denominator = _denominator(plan, "validation")["sample_id"].astype(str).tolist()
    else:
        denominator = (
            folds.loc[folds["fold"].eq(int(held_fold)), "sample_id"]
            .astype(str)
            .tolist()
        )
    expected_metrics, expected_decisions = evaluate_order_only(
        denominator, evaluation, score_column="score", max_k=5
    )
    if (
        result.get("metrics") != expected_metrics
        or result.get("prediction_contract") != _prediction_contract(predictions)
        or result.get("decision_contract") != _decision_contract(decisions)
    ):
        raise RuntimeError("D1 P10 cell metric/contract replay differs")
    _assert_frame(decisions, expected_decisions, name="D1 P10 decisions")
    expected_output_signature = canonical_sha256(
        {
            "configuration": configuration,
            "sources": sources,
            "artifacts": artifacts,
            "metrics": expected_metrics,
            "prediction_contract": result["prediction_contract"],
            "decision_contract": result["decision_contract"],
        }
    )
    if result.get("output_signature_sha256") != expected_output_signature:
        raise RuntimeError("D1 P10 cell output signature differs")
    return result


def _validate_bound_gate(
    *, root: Path, plan_path: Path, plan: Mapping[str, Any], gate_path: Path
) -> dict[str, Any]:
    policy_path, policy = load_ablation_resource_policy(
        root, plan_path=plan_path, plan=plan
    )
    gate = load_content_manifest(
        gate_path, name="D1 P10 resource gate", statuses=("PASS",)
    )
    passed, reasons = evaluate_resource_gate(gate.get("windows", []))
    if (
        gate.get("run_dir") != str(root)
        or gate.get("rank1_run_dir") != policy.get("rank1_run_dir")
        or gate.get("scope") != ablation_execution_scope(plan)
        or gate.get("policy") != artifact_record(policy_path)
        or gate.get("plan") != artifact_record(plan_path)
        or gate.get("thresholds") != resource_thresholds()
        or gate.get("thresholds") != policy.get("thresholds")
        or gate.get("candidate_test_labels_read") is not False
        or gate.get("failure_reasons") != []
        or not passed
        or reasons
    ):
        raise RuntimeError("D1 P10 bound resource-gate semantic replay differs")
    verify_artifact_records_recursive(
        {"policy": artifact_record(policy_path), "gate_sources": gate.get("sources")},
        name="D1 P10 resource-gate closure",
        require_at_least_one=True,
    )
    return gate


def validate_ablation_execution_results(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the COMPLETE authority and exact dynamic job/result bijection."""

    root = Path(run_dir).expanduser().resolve()
    plan_path = root / ABLATION_PLAN_RELATIVE
    plan = load_ablation_plan(plan_path)
    pointer_path = root / ABLATION_EXECUTION_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path, name="D1 P10 execution pointer", statuses=("COMPLETE",)
    )
    execution_id = str(pointer.get("execution_id", ""))
    directory = execution_directory(root, execution_id)
    execution_path = verified_artifact_path(
        pointer.get("execution", {}), name="D1 P10 execution authority"
    )
    event_path = verified_artifact_path(
        pointer.get("latest_event", {}), name="D1 P10 COMPLETE event"
    )
    execution = load_content_manifest(
        execution_path, name="D1 P10 execution authority", statuses=("ACTIVE",)
    )
    event = load_content_manifest(
        event_path, name="D1 P10 COMPLETE event", statuses=("COMPLETE",)
    )
    gate_path = verified_artifact_path(
        execution.get("resource_gate", {}), name="D1 P10 execution gate"
    )
    _validate_bound_gate(root=root, plan_path=plan_path, plan=plan, gate_path=gate_path)
    resume_record = execution.get("resume_from")
    expected_sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=(
            _mapping(resume_record, name="D1 P10 resume")
            if resume_record is not None
            else None
        ),
    )
    count = int(plan["job_count"])
    if (
        execution_path != directory / "execution.json"
        or event_path.parent != directory / "events"
        or execution.get("scope") != ablation_execution_scope(plan)
        or execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_ids_sha256") != plan.get("job_ids_sha256")
        or execution.get("max_parallel") != 1
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("sources") != expected_sources
        or execution.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
        or event.get("execution_id") != execution_id
        or event.get("current_job_id") is not None
        or event.get("claim") is not None
        or event.get("failure") is not None
        or pointer.get("completed_jobs") != count
        or pointer.get("expected_jobs") != count
        or event.get("completed_jobs") != count
        or event.get("expected_jobs") != count
    ):
        raise RuntimeError("D1 P10 COMPLETE execution semantics differ")
    jobs = list(plan["jobs"])
    planned = {str(job["job_id"]): job for job in jobs}
    outputs = _mapping(event.get("outputs"), name="D1 P10 outputs")
    inherited = _mapping(execution.get("resume_outputs"), name="D1 P10 inherited")
    if resume_record is None:
        if inherited:
            raise RuntimeError("D1 P10 outputs lack a recovery source")
    else:
        failed_path = verified_artifact_path(resume_record, name="D1 P10 failed event")
        failed = load_content_manifest(
            failed_path, name="D1 P10 failed event", statuses=("FAILED",)
        )
        if failed.get("outputs") != inherited:
            raise RuntimeError("D1 P10 inherited outputs differ from failed event")
    commands = event.get("commands")
    expected_jobs = [
        str(job["job_id"]) for job in jobs if job["job_id"] not in inherited
    ]
    if (
        set(outputs) != set(planned)
        or not set(inherited).issubset(planned)
        or not isinstance(commands, list)
        or len(commands) != len(expected_jobs)
    ):
        raise RuntimeError("D1 P10 job/result/command bijection differs")
    expected_indices = [
        index for index, job in enumerate(jobs) if job["job_id"] not in inherited
    ]
    for offset, command_value in enumerate(commands):
        command = _mapping(command_value, name=f"D1 P10 command {offset}")
        job_id = expected_jobs[offset]
        job = planned[job_id]
        argv = command.get("argv")
        expected_suffix = [*job["worker_argv"], "--run-dir", str(root)]
        claim_path = verified_artifact_path(
            command.get("claim", {}), name=f"D1 P10 claim {job_id}"
        )
        claim = load_content_manifest(
            claim_path, name=f"D1 P10 claim {job_id}", statuses=("CLAIMED",)
        )
        if (
            command.get("job_id") != job_id
            or command.get("index") != expected_indices[offset]
            or command.get("returncode") != 0
            or not isinstance(argv, list)
            or len(argv) != len(expected_suffix) + 1
            or argv[1:] != expected_suffix
            or claim_path != directory / "claims" / f"{job_id}.json"
            or claim.get("execution_id") != execution_id
            or claim.get("job_id") != job_id
            or claim.get("command") != argv
            or claim.get("configuration_sha256")
            != canonical_sha256(job["configuration"])
            or not isinstance(command.get("live_recheck"), Mapping)
        ):
            raise RuntimeError(f"D1 P10 command/claim differs for {job_id}")
    results: dict[str, dict[str, Any]] = {}
    result_records: dict[str, dict[str, str]] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        path = verified_artifact_path(outputs[job_id], name=f"D1 P10 output {job_id}")
        expected = (root / str(job["output_manifest"])).resolve()
        if path != expected:
            raise RuntimeError(f"D1 P10 output path differs for {job_id}")
        result = load_content_manifest(
            path, name=f"D1 P10 cell {job_id}", statuses=("COMPLETE",)
        )
        validate_ablation_cell(result, plan_path=plan_path, job=job, manifest_path=path)
        results[job_id] = result
        result_records[job_id] = artifact_record(path)
    records = {
        "plan": artifact_record(plan_path),
        "resource_policy": expected_sources["resource_policy"],
        "resource_gate": artifact_record(gate_path),
        "execution_pointer": artifact_record(pointer_path),
        "execution_authority": artifact_record(execution_path),
        "complete_event": artifact_record(event_path),
        "results": result_records,
    }
    return records, {
        "plan": plan,
        "planned": planned,
        "execution": execution,
        "event": event,
        "results": results,
    }


def _csv_frame(rows: Sequence[Mapping[str, Any]], path: Path) -> pd.DataFrame:
    expected = pd.DataFrame(rows)
    observed = pd.read_csv(path)
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as error:
        raise RuntimeError(f"D1 P10 table replay differs: {path.name}") from error
    return observed


def validate_track_gate_artifacts(
    gate: Mapping[str, Any],
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    points: Sequence[Any],
    expected_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute one transition model, 108 trials, winner, and decisions."""

    value = _mapping(gate, name="D1 P10 track gate")
    artifacts = _mapping(value.get("artifacts"), name="D1 P10 gate artifacts")
    replay = compute_track_gate(train=train, validation=validation, points=points)
    if (
        value.get("selection") != replay["selection"]
        or value.get("decision") != replay["decision"]
        or value.get("transition_model") != replay["model"].artifact()
        or value.get("metrics") != replay["metrics"]
    ):
        raise RuntimeError("D1 P10 track gate selection/model/metric replay differs")
    if expected_sources is not None and (
        value.get("sources") != dict(expected_sources)
        or value.get("source_signature_sha256")
        != canonical_sha256(
            {"configuration": value.get("configuration"), "sources": expected_sources}
        )
    ):
        raise RuntimeError("D1 P10 track gate source binding differs")
    _assert_frame(
        pd.read_parquet(
            verified_artifact_path(artifacts["train_gate_inputs"], name="gate Train")
        ),
        train,
        name="D1 P10 Train gate inputs",
    )
    _assert_frame(
        pd.read_parquet(
            verified_artifact_path(
                artifacts["validation_gate_inputs"], name="gate Validation"
            )
        ),
        validation,
        name="D1 P10 Validation gate inputs",
    )
    _assert_frame(
        pd.read_parquet(
            verified_artifact_path(
                artifacts["validation_decisions"], name="gate decisions"
            )
        ),
        replay["decisions"],
        name="D1 P10 gate decisions",
    )
    _assert_frame(
        pd.read_parquet(
            verified_artifact_path(artifacts["validation_trials"], name="gate trials")
        ),
        replay["trials"],
        name="D1 P10 gate trials",
    )
    model_path = verified_artifact_path(
        artifacts["transition_model"], name="D1 P10 transition model"
    )
    expected_model = pickle.dumps(replay["model"], protocol=pickle.HIGHEST_PROTOCOL)
    if model_path.read_bytes() != expected_model:
        raise RuntimeError("D1 P10 transition model bytes differ")
    return replay


def validate_ablation_selection(
    run_dir: str | Path,
    *,
    execution_records: Mapping[str, Any] | None = None,
    execution_values: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay feature deltas and every track-specific 108-point gate."""

    root = Path(run_dir).expanduser().resolve()
    if execution_records is None or execution_values is None:
        execution_records, execution_values = validate_ablation_execution_results(root)
    plan = execution_values["plan"]
    results = execution_values["results"]
    selection_path = root / SELECTION_RELATIVE
    selection = load_content_manifest(
        selection_path, name="D1 P10 selection", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        {"sources": selection.get("sources"), "artifacts": selection.get("artifacts")},
        name="D1 P10 selection closure",
        require_at_least_one=True,
    )
    selector = artifact_record(
        Path(__file__).resolve().parents[2] / "tools/d1_reranking/select_ablation.py"
    )
    expected_sources = {
        "plan": artifact_record(root / ABLATION_PLAN_RELATIVE),
        "execution": dict(execution_records),
        "selector": selector,
        "cells": execution_records["results"],
    }
    if (
        selection.get("sources") != expected_sources
        or selection.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
        or selection.get("candidate_test_labels_read") is not False
        or selection.get("test_inputs_referenced") is not False
        or selection.get("retuning_permitted") is not False
        or selection.get("selected_primary") != plan["selected_primary"]
    ):
        raise RuntimeError("D1 P10 selection source/frozen-primary contract differs")
    feature_rows = feature_ablation_rows(plan, results)
    artifacts = _mapping(selection.get("artifacts"), name="D1 P10 selection artifacts")
    feature_path = verified_artifact_path(
        artifacts["feature_ablation_table"], name="D1 P10 feature table"
    )
    _csv_frame(feature_rows, feature_path)
    r0 = load_content_manifest(
        verified_artifact_path(plan["sources"]["r0_r1_selection"], name="D1 R0"),
        name="D1 R0",
        statuses=("COMPLETE",),
    )
    primary = load_content_manifest(
        verified_artifact_path(plan["sources"]["selected_primary"], name="D1 primary"),
        name="D1 primary",
        statuses=("COMPLETE",),
    )
    primary_gate = load_content_manifest(
        verified_artifact_path(plan["sources"]["primary_gate"], name="D1 primary gate"),
        name="D1 primary gate",
        statuses=("COMPLETE",),
    )
    expected_rows: list[dict[str, Any]] = []
    t2_metrics = {
        "R0": r0["r0_validation_metrics"],
        "selected_ungated": primary["validation_metrics"],
        "R7": {
            "j_at_1": primary_gate["metrics"]["gated_j_at_1"],
            "sample_count": primary_gate["metrics"]["sample_count"],
        },
    }
    for method in ("R0", "selected_ungated", "R7"):
        metric = t2_metrics[method]
        expected_rows.append(
            {
                "track": "T2_matched_common",
                "method": method,
                "status": "REUSED_PRIMARY_EXACT",
                "sample_count": metric["sample_count"],
                "j_at_1": metric["j_at_1"],
                "delta_vs_r0": float(metric["j_at_1"] - t2_metrics["R0"]["j_at_1"]),
                "gate_decision": primary_gate["decision"] if method == "R7" else None,
            }
        )
    labels = {split: _label_frame(plan, split) for split in ("train", "validation")}
    paired = {split: _denominator(plan, split) for split in ("train", "validation")}
    gate_materials = {
        split: _gate_materials(plan, split) for split in ("train", "validation")
    }
    folds = pd.read_parquet(
        verified_artifact_path(plan["sources"]["fold_assignments"], name="D1 folds"),
        columns=["sample_id", "fold"],
    )
    grid_path = verified_artifact_path(plan["sources"]["gate_grid"], name="gate grid")
    grid = load_content_manifest(grid_path, name="D1 gate grid", statuses=("PLANNED",))
    points = gate_grid_points(grid)
    native = {
        "train_predictions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_train_predictions"], name="R0 Train"
            )
        ),
        "train_decisions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_train_decisions"], name="R0 Train decisions"
            )
        ),
        "validation_predictions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_validation_predictions"], name="R0 Validation"
            )
        ),
        "validation_decisions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_validation_decisions"],
                name="R0 Validation decisions",
            )
        ),
    }
    track_records = _mapping(artifacts.get("track_artifacts"), name="D1 P10 tracks")
    if set(track_records) != set(EVIDENCE_TRACKS):
        raise RuntimeError("D1 P10 track artifact inventory differs")
    expected_t2 = {
        "status": "REUSED_PRIMARY_EXACT",
        "primary": plan["sources"]["selected_primary"],
        "gate": plan["sources"]["primary_gate"],
    }
    if track_records["T2_matched_common"] != expected_t2:
        raise RuntimeError("D1 P10 T2 exact reuse differs")
    for track in EVIDENCE_TRACKS:
        if track == "T2_matched_common":
            continue
        members = [
            value
            for value in results.values()
            if value["configuration"]["analysis"] == "evidence_track"
            and value["configuration"]["variant_id"] == track
        ]
        if plan["evidence_tracks"][track]["status"] != "AVAILABLE":
            if track_records[track] != {"status": "NOT_AVAILABLE"}:
                raise RuntimeError(f"D1 P10 {track} availability differs")
            for method in ("R0", "selected_ungated", "R7"):
                expected_rows.append(
                    {
                        "track": track,
                        "method": method,
                        "status": "NOT_AVAILABLE",
                        "sample_count": None,
                        "j_at_1": None,
                        "delta_vs_r0": None,
                        "gate_decision": None,
                    }
                )
            continue
        aggregate = {
            "train_oof": aggregate_variant(
                cells=members,
                labels=labels["train"],
                denominator=paired["train"]["sample_id"].astype(str).tolist(),
                split="train_oof",
            ),
            "validation": aggregate_variant(
                cells=members,
                labels=labels["validation"],
                denominator=paired["validation"]["sample_id"].astype(str).tolist(),
                split="validation",
            ),
        }
        track_record = _mapping(track_records[track], name=f"D1 P10 {track}")
        ensembles = _mapping(
            track_record.get("ensembles"), name=f"D1 P10 {track} ensembles"
        )
        for split, replay in aggregate.items():
            stored = _mapping(ensembles.get(split), name=f"D1 P10 {track}/{split}")
            _assert_frame(
                pd.read_parquet(
                    verified_artifact_path(stored["predictions"], name="ensemble")
                ),
                replay["predictions"],
                name=f"D1 P10 {track}/{split} predictions",
            )
            _assert_frame(
                pd.read_parquet(
                    verified_artifact_path(stored["decisions"], name="decisions")
                ),
                replay["decisions"],
                name=f"D1 P10 {track}/{split} decisions",
            )
            if stored.get("metrics") != replay["metrics"]:
                raise RuntimeError(f"D1 P10 {track}/{split} metrics differ")
        gate_inputs = {
            "train": _gate_inputs(
                paired=paired["train"],
                native_predictions=native["train_predictions"],
                native_decisions=native["train_decisions"],
                challenger=aggregate["train_oof"],
                candidate_features=gate_materials["train"][0],
                candidates=gate_materials["train"][1],
                prediction_source="train_oof",
                folds=folds,
            ),
            "validation": _gate_inputs(
                paired=paired["validation"],
                native_predictions=native["validation_predictions"],
                native_decisions=native["validation_decisions"],
                challenger=aggregate["validation"],
                candidate_features=gate_materials["validation"][0],
                candidates=gate_materials["validation"][1],
                prediction_source="validation",
                folds=None,
            ),
        }
        gate_path = verified_artifact_path(
            track_record["gate"], name=f"D1 P10 {track} gate"
        )
        gate = load_content_manifest(
            gate_path, name=f"D1 P10 {track} gate", statuses=("COMPLETE",)
        )
        gate_artifacts = _mapping(
            gate.get("artifacts"), name=f"D1 P10 {track} gate artifacts"
        )
        _assert_frame(
            pd.read_parquet(
                verified_artifact_path(
                    gate_artifacts["train_gate_inputs"], name="gate Train"
                )
            ),
            gate_inputs["train"],
            name=f"D1 P10 {track} Train gate inputs",
        )
        _assert_frame(
            pd.read_parquet(
                verified_artifact_path(
                    gate_artifacts["validation_gate_inputs"], name="gate Validation"
                )
            ),
            gate_inputs["validation"],
            name=f"D1 P10 {track} Validation gate inputs",
        )
        expected_gate_sources = {
            "plan": artifact_record(root / ABLATION_PLAN_RELATIVE),
            "r0": plan["sources"]["r0_r1_selection"],
            "gate_grid": artifact_record(grid_path),
            "cells": [
                artifact_record(
                    Path(cell["artifacts"]["model"]["path"]).parent / "manifest.json"
                )
                for cell in members
            ],
        }
        replay = validate_track_gate_artifacts(
            gate,
            train=gate_inputs["train"],
            validation=gate_inputs["validation"],
            points=points,
            expected_sources=expected_gate_sources,
        )
        baseline = float(r0["r0_validation_metrics"]["j_at_1"])
        scores = {
            "R0": baseline,
            "selected_ungated": float(aggregate["validation"]["metrics"]["j_at_1"]),
            "R7": float(replay["metrics"]["gated_j_at_1"]),
        }
        for method, score in scores.items():
            expected_rows.append(
                {
                    "track": track,
                    "method": method,
                    "status": "COMPLETE",
                    "sample_count": len(paired["validation"]),
                    "j_at_1": score,
                    "delta_vs_r0": score - baseline,
                    "gate_decision": replay["decision"] if method == "R7" else None,
                }
            )
    evidence_path = verified_artifact_path(
        artifacts["evidence_track_table"], name="D1 P10 evidence table"
    )
    _csv_frame(expected_rows, evidence_path)
    records = {
        "selection": artifact_record(selection_path),
        "evidence_track_table": artifact_record(evidence_path),
        "feature_ablation_table": artifact_record(feature_path),
    }
    return records, {
        "selection": selection,
        "feature_rows": feature_rows,
        "evidence_rows": expected_rows,
    }


def validate_ablation_replay(run_dir: str | Path) -> dict[str, Any]:
    """P13-facing exact semantic validator for all P10 artifacts."""

    execution_records, execution_values = validate_ablation_execution_results(run_dir)
    selection_records, _selection_values = validate_ablation_selection(
        run_dir,
        execution_records=execution_records,
        execution_values=execution_values,
    )
    records = {"execution": execution_records, "selection": selection_records}
    verify_artifact_records_recursive(
        records, name="D1 P10 replay records", require_at_least_one=True
    )
    return {
        "status": "PASS",
        "schema_version": 1,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "checks": {
            "plan_exact_live_replay": True,
            "resource_policy_gate_execution_bound": True,
            "serial_claim_result_bijection": True,
            "cell_metrics_and_candidate_identity_recomputed": True,
            "feature_family_deltas_recomputed_per_seed": True,
            "three_track_108_grid_gates_recomputed": True,
            "transition_model_bytes_recomputed": True,
        },
        "sources": records,
        "records_sha256": canonical_sha256(records),
    }


__all__ = [
    "validate_ablation_cell",
    "validate_ablation_execution_results",
    "validate_ablation_replay",
    "validate_ablation_selection",
    "validate_track_gate_artifacts",
]
