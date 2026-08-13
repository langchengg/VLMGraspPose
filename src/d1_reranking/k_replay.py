"""Exact semantic replay for K plan, execution, cells, and selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import canonical_sha256
from unified_reranking.metrics import evaluate_order_only

from .execution import artifact_record, load_content_manifest
from .k_execution import (
    K_EXECUTION_POINTER_RELATIVE,
    K_EXECUTION_SCOPE,
    execution_directory,
    execution_source_records,
)
from .k_sensitivity import (
    K_SENSITIVITY_OUTER_FOLDS,
    K_SENSITIVITY_SEEDS,
    load_k_sensitivity_plan,
    validate_k_sensitivity_result,
)
from .selection import ensemble_seed_scores


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def validate_k_execution_results(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the COMPLETE event and exact 54 planned cell/result bijection."""

    root = Path(run_dir).expanduser().resolve()
    plan_path = root / "configs/d1_k_sensitivity_plan.json"
    plan = load_k_sensitivity_plan(plan_path)
    pointer_path = root / K_EXECUTION_POINTER_RELATIVE
    pointer = load_content_manifest(
        pointer_path, name="D1 K execution pointer", statuses=("COMPLETE",)
    )
    execution_id = str(pointer.get("execution_id", ""))
    execution_path = verified_artifact_path(
        _mapping(pointer.get("execution"), name="D1 K execution authority record"),
        name="D1 K execution authority",
    )
    expected_directory = execution_directory(root, execution_id)
    if execution_path != expected_directory / "execution.json":
        raise RuntimeError("D1 K COMPLETE authority path differs")
    execution = load_content_manifest(
        execution_path, name="D1 K execution authority", statuses=("ACTIVE",)
    )
    gate_path = verified_artifact_path(
        _mapping(execution.get("resource_gate"), name="D1 K execution resource gate"),
        name="D1 K execution resource gate",
    )
    resume_record = execution.get("resume_from")
    expected_sources = execution_source_records(
        plan_path=plan_path,
        resource_gate_path=gate_path,
        resume_from=(
            _mapping(resume_record, name="D1 K resume record")
            if resume_record is not None
            else None
        ),
    )
    event_path = verified_artifact_path(
        _mapping(pointer.get("latest_event"), name="D1 K COMPLETE event record"),
        name="D1 K COMPLETE event",
    )
    event = load_content_manifest(
        event_path, name="D1 K COMPLETE event", statuses=("COMPLETE",)
    )
    if (
        execution.get("execution_id") != execution_id
        or execution.get("scope") != K_EXECUTION_SCOPE
        or execution.get("plan") != artifact_record(plan_path)
        or execution.get("job_count") != 54
        or execution.get("max_parallel") != 1
        or execution.get("device") != "cpu"
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("sources") != expected_sources
        or execution.get("source_signature_sha256")
        != canonical_sha256(expected_sources)
        or event_path.parent != expected_directory / "events"
        or event.get("execution_id") != execution_id
        or event.get("current_job_id") is not None
        or event.get("claim") is not None
        or event.get("failure") is not None
        or pointer.get("completed_jobs") != 54
        or pointer.get("expected_jobs") != 54
        or event.get("completed_jobs") != 54
        or event.get("expected_jobs") != 54
    ):
        raise RuntimeError("D1 K COMPLETE execution semantics differ")
    verify_artifact_records_recursive(
        expected_sources,
        name="D1 K execution source closure",
        require_at_least_one=True,
    )
    jobs = plan.get("jobs")
    outputs = _mapping(event.get("outputs"), name="D1 K COMPLETE outputs")
    commands = event.get("commands")
    if not isinstance(jobs, list) or len(jobs) != 54 or not isinstance(commands, list):
        raise RuntimeError("D1 K COMPLETE job/command inventory is invalid")
    planned = {
        str(job["job_id"]): job
        for job in jobs
        if isinstance(job, Mapping) and isinstance(job.get("configuration"), Mapping)
    }
    inherited = _mapping(execution.get("resume_outputs"), name="D1 K inherited outputs")
    resume_from = execution.get("resume_from")
    if resume_from is None:
        if inherited:
            raise RuntimeError("D1 K execution has outputs without resume authority")
    else:
        failed_event_path = verified_artifact_path(
            _mapping(resume_from, name="D1 K resume event record"),
            name="D1 K resume event",
        )
        failed_event = load_content_manifest(
            failed_event_path, name="D1 K resume event", statuses=("FAILED",)
        )
        if failed_event.get("outputs") != inherited:
            raise RuntimeError("D1 K inherited outputs differ from failed event")
    if (
        len(planned) != 54
        or set(outputs) != set(planned)
        or not set(inherited).issubset(planned)
        or len(commands) != 54 - len(inherited)
    ):
        raise RuntimeError("D1 K COMPLETE job/result/command bijection differs")
    command_jobs: list[str] = []
    result_records: dict[str, dict[str, str]] = {}
    result_values: dict[str, dict[str, Any]] = {}
    python_paths: set[str] = set()
    planned_order = [str(job["job_id"]) for job in jobs]
    expected_command_jobs = [
        job_id for job_id in planned_order if job_id not in inherited
    ]
    expected_indices = [
        index for index, job_id in enumerate(planned_order) if job_id not in inherited
    ]
    for command_offset, command_value in enumerate(commands):
        command = _mapping(command_value, name=f"D1 K command {command_offset}")
        job_id = str(command.get("job_id", ""))
        plan_index = expected_indices[command_offset]
        if (
            job_id != expected_command_jobs[command_offset]
            or command.get("index") != plan_index
        ):
            raise RuntimeError(f"D1 K command {plan_index} differs from plan order")
        argv = command.get("argv")
        worker_argv = planned[job_id].get("worker_argv")
        expected_suffix = [*worker_argv, "--run-dir", str(root)]
        if (
            not isinstance(argv, list)
            or len(argv) != len(expected_suffix) + 1
            or argv[1:] != expected_suffix
            or command.get("returncode") != 0
            or not isinstance(command.get("live_recheck"), Mapping)
        ):
            raise RuntimeError(f"D1 K command {job_id} contract differs")
        python_paths.add(str(argv[0]))
        command_jobs.append(job_id)
        claim_path = verified_artifact_path(
            _mapping(command.get("claim"), name=f"D1 K command {job_id} claim"),
            name=f"D1 K command {job_id} claim",
        )
        claim = load_content_manifest(
            claim_path, name=f"D1 K command {job_id} claim", statuses=("CLAIMED",)
        )
        if (
            claim_path != expected_directory / "claims" / f"{job_id}.json"
            or claim.get("execution_id") != execution_id
            or claim.get("job_id") != job_id
            or claim.get("command") != argv
            or claim.get("configuration_sha256")
            != canonical_sha256(planned[job_id]["configuration"])
        ):
            raise RuntimeError(f"D1 K command/claim binding differs for {job_id}")
    expected_python_count = 1 if commands else 0
    if (
        command_jobs != expected_command_jobs
        or len(python_paths) != expected_python_count
    ):
        raise RuntimeError("D1 K command order/Python executable differs")
    for job_id in planned_order:
        result_path = verified_artifact_path(
            _mapping(outputs.get(job_id), name=f"D1 K output {job_id}"),
            name=f"D1 K output {job_id}",
        )
        expected_result = (root / str(planned[job_id]["output_manifest"])).resolve()
        if result_path != expected_result:
            raise RuntimeError(f"D1 K result path differs for {job_id}")
        result = load_content_manifest(
            result_path, name=f"D1 K result {job_id}", statuses=("COMPLETE",)
        )
        validate_k_sensitivity_result(
            result,
            plan_path=plan_path,
            job=planned[job_id],
            manifest_path=result_path,
        )
        result_records[job_id] = artifact_record(result_path)
        result_values[job_id] = result
    records = {
        "plan": artifact_record(plan_path),
        "execution_pointer": artifact_record(pointer_path),
        "execution_authority": artifact_record(execution_path),
        "complete_event": artifact_record(event_path),
        "results": result_records,
    }
    return records, {
        "plan": plan,
        "execution": execution,
        "event": event,
        "planned": planned,
        "results": result_values,
    }


def validate_k_selection(
    run_dir: str | Path,
    *,
    execution_records: Mapping[str, Any] | None = None,
    execution_values: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay ensemble artifacts and the comparison table from exact cell scores."""

    root = Path(run_dir).expanduser().resolve()
    if execution_records is None or execution_values is None:
        execution_records, execution_values = validate_k_execution_results(root)
    selection_path = root / "11_k_sensitivity/selection_manifest.json"
    selection = load_content_manifest(
        selection_path, name="D1 K selection", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        {"sources": selection.get("sources"), "artifacts": selection.get("artifacts")},
        name="D1 K selection closure",
        require_at_least_one=True,
    )
    sources = _mapping(selection.get("sources"), name="D1 K selection sources")
    repository_root = Path(__file__).resolve().parents[2]
    selector_record = artifact_record(
        repository_root / "tools/d1_reranking/select_k_sensitivity.py"
    )
    selected_primary_path = root / "07_validation/selected_primary_ungated.json"
    expected_source_signature = canonical_sha256(
        {
            **execution_records,
            "selected_primary": artifact_record(selected_primary_path),
            "selector": selector_record,
        }
    )
    if (
        set(sources)
        != {
            "plan",
            "execution_pointer",
            "execution_authority",
            "complete_event",
            "cells",
            "selected_primary",
            "development_labels",
            "selector",
        }
        or sources.get("plan") != execution_records.get("plan")
        or sources.get("execution_pointer")
        != execution_records.get("execution_pointer")
        or sources.get("complete_event") != execution_records.get("complete_event")
        or sources.get("cells") != execution_records.get("results")
        or sources.get("execution_authority")
        != execution_records.get("execution_authority")
        or sources.get("selected_primary") != artifact_record(selected_primary_path)
        or sources.get("selector") != selector_record
        or selection.get("source_signature_sha256") != expected_source_signature
        or selection.get("candidate_test_labels_read") is not False
        or selection.get("test_inputs_referenced") is not False
        or selection.get("primary_replacement_permitted") is not False
        or selection.get("seed_policy")
        != "fixed mean score ensemble over 42,123,2026; best-seed forbidden"
        or selection.get("selection_interface")
        != (
            "group exact cells by scenario; concatenate five OOF folds per seed; "
            "mean exact candidate scores across seeds; order sensitivity scenarios "
            "by Validation J@1, MRR@5, nDCG@5, scenario_id"
        )
    ):
        raise RuntimeError("D1 K selection source/policy contract differs")
    scenarios = _mapping(selection.get("scenarios"), name="D1 K selection scenarios")
    plan = execution_values["plan"]
    expected_scenarios = {
        str(item["scenario_id"]): item for item in plan["scenario_definitions"]
    }
    if set(scenarios) != set(expected_scenarios):
        raise RuntimeError("D1 K selection scenario inventory differs")
    planned = execution_values["planned"]
    results = execution_values["results"]
    label_source_values = _mapping(
        sources.get("development_labels"), name="D1 K selection label sources"
    )
    for scenario_id, definition in expected_scenarios.items():
        scenario = _mapping(
            scenarios.get(scenario_id), name=f"D1 K selection {scenario_id}"
        )
        members = [
            (job_id, job, results[job_id])
            for job_id, job in planned.items()
            if job["configuration"]["scenario_id"] == scenario_id
        ]
        expected_job_ids = sorted(job_id for job_id, _job, _result in members)
        if (
            scenario.get("definition") != definition
            or scenario.get("seeds") != list(K_SENSITIVITY_SEEDS)
            or scenario.get("cell_job_ids") != expected_job_ids
            or len(members) != 18
        ):
            raise RuntimeError(f"D1 K selection {scenario_id} semantics differ")
        artifacts = _mapping(
            scenario.get("artifacts"), name=f"D1 K selection {scenario_id} artifacts"
        )
        if set(artifacts) != {
            "validation_predictions",
            "validation_decisions",
            "oof_predictions",
            "oof_decisions",
        }:
            raise RuntimeError(f"D1 K selection {scenario_id} artifacts differ")
        validation_path = verified_artifact_path(
            _mapping(
                artifacts.get("validation_predictions"), name="K Validation ensemble"
            ),
            name=f"D1 K {scenario_id} Validation ensemble",
        )
        oof_path = verified_artifact_path(
            _mapping(artifacts.get("oof_predictions"), name="K OOF ensemble"),
            name=f"D1 K {scenario_id} OOF ensemble",
        )
        observed_frames: dict[str, pd.DataFrame] = {}
        for name, path in (("validation", validation_path), ("oof", oof_path)):
            frame = pd.read_parquet(path)
            observed_frames[name] = frame
            expected_columns = {
                "sample_id",
                "candidate_id",
                "native_rank",
                "candidate_identity_sha256",
                "candidate_geometry_sha256",
                "score_seed_42",
                "score_seed_123",
                "score_seed_2026",
                "ensemble_score",
            }
            if set(frame.columns) != expected_columns or frame.empty:
                raise RuntimeError(f"D1 K {scenario_id} {name} ensemble schema differs")
            expected_mean = frame[
                ["score_seed_42", "score_seed_123", "score_seed_2026"]
            ].mean(axis=1)
            if not expected_mean.equals(frame["ensemble_score"]):
                raise RuntimeError(f"D1 K {scenario_id} {name} ensemble mean differs")
        validation_by_seed: dict[int, pd.DataFrame] = {}
        oof_by_seed: dict[int, list[tuple[int, pd.DataFrame]]] = {
            seed: [] for seed in K_SENSITIVITY_SEEDS
        }
        for job_id, job, result in members:
            configuration = job["configuration"]
            prediction_path = verified_artifact_path(
                _mapping(
                    _mapping(
                        result.get("artifacts"), name=f"D1 K {job_id} artifacts"
                    ).get("predictions"),
                    name=f"D1 K {job_id} predictions",
                ),
                name=f"D1 K {job_id} predictions",
            )
            prediction = pd.read_parquet(prediction_path)
            seed = int(configuration["seed"])
            if configuration["mode"] == "validation":
                if seed in validation_by_seed:
                    raise RuntimeError(f"D1 K {scenario_id} duplicates Validation seed")
                validation_by_seed[seed] = prediction
            else:
                oof_by_seed[seed].append((int(configuration["held_fold"]), prediction))
        if set(validation_by_seed) != set(K_SENSITIVITY_SEEDS) or any(
            sorted(fold for fold, _frame in parts)
            != list(range(K_SENSITIVITY_OUTER_FOLDS))
            for parts in oof_by_seed.values()
        ):
            raise RuntimeError(f"D1 K {scenario_id} seed/fold inventory differs")
        expected_frames = {
            "validation": ensemble_seed_scores(validation_by_seed),
            "oof": ensemble_seed_scores(
                {
                    seed: pd.concat(
                        [frame for _fold, frame in sorted(parts)], ignore_index=True
                    )
                    for seed, parts in oof_by_seed.items()
                }
            ),
        }
        for name in ("validation", "oof"):
            try:
                pd.testing.assert_frame_equal(
                    observed_frames[name], expected_frames[name], check_exact=True
                )
            except AssertionError as error:
                raise RuntimeError(
                    f"D1 K {scenario_id} {name} ensemble semantic replay differs"
                ) from error
        pool = str(definition["pool"])
        pool_label_sources = _mapping(
            label_source_values.get(pool), name=f"D1 K {pool} label sources"
        )
        for name, split in (("validation", "validation"), ("oof", "train")):
            split_sources = _mapping(
                pool_label_sources.get(split), name=f"D1 K {pool}/{split} label sources"
            )
            planned_label_record = plan["sources"]["development_label_manifests"][pool][
                split
            ]
            planned_denominator_record = plan["sources"]["denominators"][split]
            label_manifest_path = verified_artifact_path(
                planned_label_record, name=f"D1 K {pool}/{split} label manifest"
            )
            label_manifest = load_content_manifest(
                label_manifest_path,
                name=f"D1 K {pool}/{split} labels",
                statuses=("COMPLETE",),
            )
            label_path = verified_artifact_path(
                _mapping(label_manifest.get("artifact"), name="D1 K label artifact"),
                name=f"D1 K {pool}/{split} labels",
            )
            denominator_path = verified_artifact_path(
                planned_denominator_record,
                name=f"D1 K {split} denominator",
            )
            if split_sources != {
                "manifest": planned_label_record,
                "labels": artifact_record(label_path),
                "denominator": planned_denominator_record,
            }:
                raise RuntimeError(f"D1 K {pool}/{split} label source binding differs")
            labels = pd.read_parquet(
                label_path,
                columns=["sample_id", "candidate_id", "candidate_success"],
            )
            evaluation = expected_frames[name].merge(
                labels,
                on=["sample_id", "candidate_id"],
                how="inner",
                validate="one_to_one",
            )
            if len(evaluation) != len(expected_frames[name]) or len(evaluation) != len(
                labels
            ):
                raise RuntimeError(
                    f"D1 K {scenario_id} {name} label membership differs"
                )
            denominator = (
                pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
                .astype(str)
                .tolist()
            )
            metrics, decisions = evaluate_order_only(
                denominator,
                evaluation,
                score_column="ensemble_score",
                max_k=5,
            )
            if scenario.get(f"{name}_metrics") != metrics:
                raise RuntimeError(f"D1 K {scenario_id} {name} metrics differ")
            decision_path = verified_artifact_path(
                _mapping(
                    artifacts.get(f"{name}_decisions"),
                    name=f"D1 K {scenario_id} {name} decisions",
                ),
                name=f"D1 K {scenario_id} {name} decisions",
            )
            try:
                pd.testing.assert_frame_equal(
                    pd.read_parquet(decision_path), decisions, check_exact=True
                )
            except AssertionError as error:
                raise RuntimeError(
                    f"D1 K {scenario_id} {name} decision replay differs"
                ) from error
    comparison_record = _mapping(
        _mapping(selection.get("artifacts"), name="D1 K selection artifacts").get(
            "comparison_table"
        ),
        name="D1 K comparison table",
    )
    comparison_path = verified_artifact_path(
        comparison_record, name="D1 K comparison table"
    )
    comparison = pd.read_csv(comparison_path)
    if (
        len(comparison) != 4
        or set(comparison["scenario_id"].astype(str))
        != {"top5_primary", *expected_scenarios}
        or set(comparison["pool"].astype(str)) != {"top5", "top10", "allnms"}
    ):
        raise RuntimeError("D1 K comparison table exact scenario coverage differs")
    selected_primary = load_content_manifest(
        selected_primary_path,
        name="D1 selected primary for K replay",
        statuses=("COMPLETE",),
    )
    primary_row = comparison.loc[
        comparison["scenario_id"].astype(str).eq("top5_primary")
    ]
    if (
        len(primary_row) != 1
        or primary_row.iloc[0]["role"] != "frozen_primary"
        or primary_row.iloc[0]["method"] != selected_primary.get("selected_method")
    ):
        raise RuntimeError("D1 K primary comparison row differs")
    for prefix in ("validation", "oof"):
        metrics = _mapping(
            selected_primary.get(f"{prefix}_metrics"),
            name=f"D1 selected primary {prefix} metrics",
        )
        for metric in ("j_at_1", "mrr_at_5", "ndcg_at_5"):
            if float(primary_row.iloc[0][f"{prefix}_{metric}"]) != float(
                metrics[metric]
            ):
                raise RuntimeError(f"D1 K primary {prefix}/{metric} differs")
    for scenario_id, scenario in scenarios.items():
        row = comparison.loc[comparison["scenario_id"].astype(str).eq(scenario_id)]
        if len(row) != 1:
            raise RuntimeError(f"D1 K comparison row differs for {scenario_id}")
        for prefix in ("validation", "oof"):
            metrics = _mapping(
                scenario.get(f"{prefix}_metrics"),
                name=f"D1 K {scenario_id} {prefix} metrics",
            )
            for metric in ("j_at_1", "mrr_at_5", "ndcg_at_5"):
                if float(row.iloc[0][f"{prefix}_{metric}"]) != float(metrics[metric]):
                    raise RuntimeError(
                        f"D1 K comparison metric differs: {scenario_id}/{prefix}/{metric}"
                    )
    expected_validation_order = sorted(
        scenarios,
        key=lambda scenario_id: (
            -float(scenarios[scenario_id]["validation_metrics"]["j_at_1"]),
            -float(scenarios[scenario_id]["validation_metrics"]["mrr_at_5"]),
            -float(scenarios[scenario_id]["validation_metrics"]["ndcg_at_5"]),
            scenario_id,
        ),
    )
    if selection.get("validation_order") != expected_validation_order:
        raise RuntimeError("D1 K Validation selection interface order differs")
    records = {
        "selection": artifact_record(selection_path),
        "comparison_table": comparison_record,
        "scenario_artifacts": {
            scenario_id: scenario["artifacts"]
            for scenario_id, scenario in scenarios.items()
        },
    }
    return records, {
        "scenario_count": 3,
        "cell_count": 54,
        "seed_ensemble": list(K_SENSITIVITY_SEEDS),
        "primary_replacement_permitted": False,
        "validation_order": selection.get("validation_order"),
    }


def validate_k_sensitivity_replay(
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution_records, execution_values = validate_k_execution_results(run_dir)
    selection_records, checks = validate_k_selection(
        run_dir,
        execution_records=execution_records,
        execution_values=execution_values,
    )
    return {
        "execution": execution_records,
        "selection": selection_records,
        "replay_code": artifact_record(Path(__file__)),
    }, checks


__all__ = [
    "validate_k_execution_results",
    "validate_k_selection",
    "validate_k_sensitivity_replay",
]
