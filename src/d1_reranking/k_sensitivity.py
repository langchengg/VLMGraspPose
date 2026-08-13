"""Frozen, label-isolated planning contract for D1 K-sensitivity cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256

from .execution import artifact_record, load_content_manifest
from .k_execution import K_EXECUTION_SCOPE, execution_directory


K_SENSITIVITY_SEEDS = (42, 123, 2026)
K_SENSITIVITY_OUTER_FOLDS = 5
K_SENSITIVITY_METHODS = ("R3", "R5", "R6")
K_SENSITIVITY_RUNNER_MODULE = "tools.d1_reranking.run_k_sensitivity_cell"
K_SENSITIVITY_SCENARIOS = (
    ("top10_t2", "top10", "T2_matched_common"),
    ("allnms_t2", "allnms", "T2_matched_common"),
    ("allnms_t3", "allnms", "T3_route_rich"),
)
K_NEURAL_LOSSES = {"R3": "ranknet", "R6": "jacquard_margin_ranknet"}


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def selected_training_spec(
    configuration: Mapping[str, Any], *, method: str
) -> dict[str, Any]:
    """Project every numerical training argument used by a frozen primary trial."""

    selected = _require_mapping(
        configuration, name="D1 K-sensitivity selected primary configuration"
    )
    if selected.get("method") != method:
        raise RuntimeError("D1 K-sensitivity selected method/configuration differs")
    if method == "R5":
        if (
            selected.get("encoder") != "lambdamart"
            or selected.get("loss") != "lambdarank"
        ):
            raise RuntimeError("D1 K-sensitivity selected R5 primitive differs")
        return {
            "encoder": "lambdamart",
            "loss": "lambdarank",
            "num_leaves": int(selected["num_leaves"]),
            "learning_rate": float(selected["learning_rate"]),
            "n_estimators": int(selected["n_estimators"]),
        }
    if method not in K_NEURAL_LOSSES:
        raise RuntimeError("D1 K-sensitivity selected method is unsupported")
    if (
        selected.get("encoder") != "d1_residual_mlp"
        or selected.get("loss") != K_NEURAL_LOSSES[method]
        or selected.get("optimizer") != "AdamW"
    ):
        raise RuntimeError("D1 K-sensitivity selected neural primitive differs")
    return {
        "encoder": "d1_residual_mlp",
        "loss": K_NEURAL_LOSSES[method],
        "optimizer": "AdamW",
        "hidden_dims": list(map(int, selected["hidden_dims"])),
        "dropout": float(selected["dropout"]),
        "alpha": float(selected["alpha"]),
        "learning_rate": float(selected["learning_rate"]),
        "weight_decay": float(selected["weight_decay"]),
        "temperature": float(selected["temperature"]),
        "beta": float(selected["beta"]),
        "epochs": int(selected["epochs"]),
        "patience": int(selected["patience"]),
        "batch_size": int(selected["batch_size"]),
        "gradient_clip_norm": float(selected["gradient_clip_norm"]),
    }


def _candidate_manifest_path(root: Path, split: str) -> Path:
    return root / "02_candidates" / split / "manifest.json"


def _feature_manifest_path(root: Path, split: str, pool: str, track: str) -> Path:
    return root / "03_features" / split / pool / track / "manifest.json"


def _label_manifest_path(root: Path, split: str, pool: str) -> Path:
    return root / "03_features" / split / pool / "labels" / "manifest.json"


def _load_candidate_manifest(
    root: Path, split: str
) -> tuple[Path, dict[str, Any], dict[str, int]]:
    path = _candidate_manifest_path(root, split)
    manifest = load_content_manifest(
        path, name=f"D1 {split} candidate manifest", statuses=("COMPLETE",)
    )
    configuration = _require_mapping(
        manifest.get("configuration"), name=f"D1 {split} candidate configuration"
    )
    if (
        configuration.get("route") != "D1"
        or configuration.get("split") != split
        or manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {split} candidate manifest semantics differ")
    summaries = _require_mapping(
        manifest.get("summaries"), name=f"D1 {split} candidate summaries"
    )
    maxima: dict[str, int] = {}
    for pool in ("top10", "allnms"):
        summary = _require_mapping(
            summaries.get(pool), name=f"D1 {split}/{pool} candidate summary"
        )
        value = summary.get("maximum_candidates")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(
                f"D1 {split}/{pool} maximum_candidates must be a positive integer"
            )
        maxima[pool] = value
    if maxima["top10"] > 10 or maxima["allnms"] < maxima["top10"]:
        raise RuntimeError(f"D1 {split} candidate pool maxima are inconsistent")
    return path, manifest, maxima


def _load_feature_manifest(root: Path, *, split: str, pool: str, track: str) -> Path:
    path = _feature_manifest_path(root, split, pool, track)
    manifest = load_content_manifest(
        path,
        name=f"D1 {split}/{pool}/{track} feature manifest",
        statuses=("COMPLETE",),
    )
    configuration = _require_mapping(
        manifest.get("configuration"),
        name=f"D1 {split}/{pool}/{track} feature configuration",
    )
    if (
        configuration.get("route") != "D1"
        or configuration.get("split") != split
        or configuration.get("pool") != pool
        or configuration.get("track") != track
        or manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {split}/{pool}/{track} feature semantics differ")
    return path


def _load_label_manifest(root: Path, *, split: str, pool: str) -> Path:
    path = _label_manifest_path(root, split, pool)
    manifest = load_content_manifest(
        path, name=f"D1 {split}/{pool} development labels", statuses=("COMPLETE",)
    )
    if (
        manifest.get("split") != split
        or manifest.get("pool") != pool
        or manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {split}/{pool} development-label semantics differ")
    return path


def _load_calibration_manifest(root: Path, *, pool: str) -> Path:
    path = root / "05_calibration" / pool / "calibration_manifest.json"
    manifest = load_content_manifest(
        path, name=f"D1 {pool} calibration manifest", statuses=("COMPLETE",)
    )
    configuration = _require_mapping(
        manifest.get("configuration"), name=f"D1 {pool} calibration configuration"
    )
    if (
        configuration.get("route") != "D1"
        or configuration.get("pool") != pool
        or manifest.get("candidate_test_labels_read") is not False
        or not isinstance(manifest.get("selected_method"), str)
    ):
        raise RuntimeError(f"D1 {pool} calibration semantics differ")
    return path


def _selected_primary_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_path = root / "07_validation" / "selected_primary_ungated.json"
    selected = load_content_manifest(
        selected_path, name="D1 selected primary ungated", statuses=("COMPLETE",)
    )
    if selected.get("candidate_test_labels_read") is not False:
        raise RuntimeError("D1 selected primary violates Test-label isolation")
    method = selected.get("selected_method")
    trial_id = selected.get("selected_trial_id")
    configuration = _require_mapping(
        selected.get("selected_configuration"),
        name="D1 selected primary configuration",
    )
    if (
        method not in K_SENSITIVITY_METHODS
        or not isinstance(trial_id, str)
        or not trial_id
        or configuration.get("route") != "D1"
        or configuration.get("pool") != "top5"
        or configuration.get("track") != "T2_matched_common"
        or configuration.get("method") != method
        or any(key in configuration for key in ("seed", "mode", "held_fold"))
    ):
        raise RuntimeError("D1 selected primary trial contract is invalid")
    selected_training_spec(configuration, method=str(method))

    sources = _require_mapping(
        selected.get("sources"), name="D1 selected primary sources"
    )
    plan_record = _require_mapping(
        sources.get("plan"), name="D1 selected primary plan record"
    )
    primary_plan_path = verified_artifact_path(
        plan_record, name="D1 selected primary source plan"
    )
    primary_plan = load_content_manifest(
        primary_plan_path, name="D1 primary matrix plan", statuses=("PLANNED",)
    )
    if (
        primary_plan.get("route") != "D1"
        or primary_plan.get("primary_contract")
        != {"pool": "top5", "track": "T2_matched_common"}
        or tuple(primary_plan.get("formal_seeds", ())) != K_SENSITIVITY_SEEDS
        or primary_plan.get("outer_folds") != K_SENSITIVITY_OUTER_FOLDS
    ):
        raise RuntimeError("D1 selected primary source-plan semantics differ")

    artifacts = _require_mapping(
        selected.get("artifacts"), name="D1 selected primary artifacts"
    )
    trial_record = _require_mapping(
        artifacts.get("selected_trial_manifest"),
        name="D1 selected primary trial record",
    )
    trial_path = verified_artifact_path(
        trial_record, name="D1 selected primary trial manifest"
    )
    trial = load_content_manifest(
        trial_path, name="D1 selected primary trial", statuses=("COMPLETE",)
    )
    if (
        trial.get("trial_id") != trial_id
        or trial.get("configuration") != configuration
        or trial.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 selected trial differs from the selection manifest")

    binding = {
        "method": method,
        "trial_id": trial_id,
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
        "seed_policy": "fixed seeds 42,123,2026; score ensemble; best-seed selection forbidden",
    }
    records = {
        "selected_primary": artifact_record(selected_path),
        "selected_trial": artifact_record(trial_path),
        "primary_plan": artifact_record(primary_plan_path),
    }
    return binding, records


def _scenario_definitions(allnms_max_candidates: int) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": scenario_id,
            "pool": pool,
            "track": track,
            "max_candidates": 10 if pool == "top10" else allnms_max_candidates,
        }
        for scenario_id, pool, track in K_SENSITIVITY_SCENARIOS
    ]


def _jobs(
    *,
    selected_primary: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in K_SENSITIVITY_SEEDS:
            for mode_fold in ("validation", 0, 1, 2, 3, 4):
                mode = "validation" if mode_fold == "validation" else "oof"
                held_fold = None if mode == "validation" else int(mode_fold)
                configuration: dict[str, Any] = {
                    "schema_version": 1,
                    "route": "D1",
                    "analysis": "k_sensitivity",
                    "scenario_id": scenario["scenario_id"],
                    "pool": scenario["pool"],
                    "track": scenario["track"],
                    "method": selected_primary["method"],
                    "selected_primary_trial_id": selected_primary["trial_id"],
                    "selected_primary_configuration_sha256": selected_primary[
                        "configuration_sha256"
                    ],
                    "selected_primary_configuration": selected_primary["configuration"],
                    "seed": seed,
                    "mode": mode,
                    "held_fold": held_fold,
                    "max_candidates": scenario["max_candidates"],
                    "fold_local_preprocessing": True,
                    "fold_local_calibration": True,
                    "candidate_test_labels_read": False,
                }
                job_id = canonical_sha256(configuration)[:16]
                jobs.append(
                    {
                        "job_id": job_id,
                        "worker_argv": [
                            "-m",
                            K_SENSITIVITY_RUNNER_MODULE,
                            "--job-id",
                            job_id,
                            "--resume",
                        ],
                        "output_manifest": (
                            f"11_k_sensitivity/cells/{job_id}/manifest.json"
                        ),
                        "configuration": configuration,
                    }
                )
    if len({str(job["job_id"]) for job in jobs}) != len(jobs):
        raise RuntimeError("D1 K-sensitivity job identifiers collide")
    return jobs


def build_k_sensitivity_plan(
    run_dir: str | Path, *, tool_paths: Sequence[Path]
) -> dict[str, Any]:
    """Build the deterministic Train/Validation-only K-sensitivity plan."""

    root = Path(run_dir).expanduser().resolve()
    selected_primary, selected_records = _selected_primary_contract(root)

    candidate_records: dict[str, dict[str, str]] = {}
    candidate_maxima: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        path, _manifest, maxima = _load_candidate_manifest(root, split)
        candidate_records[split] = artifact_record(path)
        candidate_maxima[split] = maxima
    allnms_max_candidates = max(
        candidate_maxima[split]["allnms"] for split in ("train", "validation")
    )
    scenarios = _scenario_definitions(allnms_max_candidates)

    feature_records: dict[str, dict[str, dict[str, str]]] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        feature_records[scenario_id] = {}
        for split in ("train", "validation"):
            path = _load_feature_manifest(
                root,
                split=split,
                pool=str(scenario["pool"]),
                track=str(scenario["track"]),
            )
            feature_records[scenario_id][split] = artifact_record(path)

    label_records: dict[str, dict[str, dict[str, str]]] = {}
    calibration_records: dict[str, dict[str, str]] = {}
    for pool in ("top10", "allnms"):
        label_records[pool] = {
            split: artifact_record(_load_label_manifest(root, split=split, pool=pool))
            for split in ("train", "validation")
        }
        calibration_records[pool] = artifact_record(
            _load_calibration_manifest(root, pool=pool)
        )

    folds_path = root / "04_splits" / "fold_assignments.parquet"
    denominator_paths = {
        "train": root / "01_manifests" / "d1_paired_train.parquet",
        "validation": root / "01_manifests" / "d1_paired_validation.parquet",
    }
    resolved_tools = sorted(
        {Path(path).expanduser().resolve() for path in tool_paths}, key=str
    )
    if not resolved_tools:
        raise ValueError("D1 K-sensitivity plan requires code source paths")
    sources: dict[str, Any] = {
        **selected_records,
        "candidate_manifests": candidate_records,
        "feature_manifests": feature_records,
        "development_label_manifests": label_records,
        "calibration_manifests": calibration_records,
        "fold_assignments": artifact_record(folds_path),
        "denominators": {
            split: artifact_record(path) for split, path in denominator_paths.items()
        },
        "code": [artifact_record(path) for path in resolved_tools],
    }
    source_signature = canonical_sha256(sources)
    jobs = _jobs(selected_primary=selected_primary, scenarios=scenarios)
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED",
        "execution_authorized": False,
        "route": "D1",
        "analysis": "K_sensitivity",
        "development_splits": ["train", "validation"],
        "test_inputs_referenced": False,
        "candidate_test_labels_read": False,
        "reporting_methods": ["R0", "selected_ungated", "R7"],
        "selected_primary": selected_primary,
        "formal_seeds": list(K_SENSITIVITY_SEEDS),
        "outer_folds": K_SENSITIVITY_OUTER_FOLDS,
        "scenario_definitions": scenarios,
        "candidate_manifest_maxima": candidate_maxima,
        "allnms_max_candidates": allnms_max_candidates,
        "job_contract": {
            "role": "selected-ungated ranker OOF plus Validation training",
            "query_membership": "frozen candidate manifests; no regeneration or re-NMS",
            "hyperparameters": "exact selected Top5/T2 primary trial; no retuning",
            "seed_policy": "fixed 42,123,2026 mean-score ensemble",
            "fold_local_preprocessing": True,
            "fold_local_calibration": True,
            "test_metrics_used": False,
        },
        "execution_contract": {
            "runner_module": K_SENSITIVITY_RUNNER_MODULE,
            "runner_implemented": True,
            "authorization_tool": "tools.d1_reranking.authorize_k_sensitivity_execution",
            "orchestrator_tool": "tools.d1_reranking.run_k_sensitivity_matrix",
            "authorization_required": True,
            "direct_cli_execution_permitted": False,
            "execution_manifest": "configs/d1_k_sensitivity_execution.json",
            "execution_state_transitions": ["ACTIVE", "COMPLETE", "FAILED"],
            "max_parallel": 1,
            "device": "cpu",
            "fresh_resource_gate_required_on_start_and_resume": True,
            "existing_train_primary_cell_compatible": False,
            "reason": "existing primary cell is frozen to Top5/T2/max_candidates=5",
        },
        "jobs": jobs,
        "job_count": len(jobs),
        "job_universe_sha256": canonical_sha256([job["configuration"] for job in jobs]),
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "source_signature_sha256": source_signature,
        "sources": sources,
    }
    plan["content_sha256"] = canonical_sha256(plan)
    return validate_k_sensitivity_plan(plan)


def validate_k_sensitivity_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the immutable job universe without reading any dataset rows."""

    plan = {str(key): child for key, child in value.items()}
    unsigned = dict(plan)
    observed_content = unsigned.pop("content_sha256", None)
    if observed_content != canonical_sha256(unsigned):
        raise RuntimeError("D1 K-sensitivity plan content hash mismatch")
    if (
        plan.get("status") != "PLANNED"
        or plan.get("execution_authorized") is not False
        or plan.get("route") != "D1"
        or plan.get("analysis") != "K_sensitivity"
        or plan.get("development_splits") != ["train", "validation"]
        or plan.get("test_inputs_referenced") is not False
        or plan.get("candidate_test_labels_read") is not False
        or tuple(plan.get("formal_seeds", ())) != K_SENSITIVITY_SEEDS
        or plan.get("outer_folds") != K_SENSITIVITY_OUTER_FOLDS
    ):
        raise RuntimeError("D1 K-sensitivity plan header contract differs")
    selected = _require_mapping(
        plan.get("selected_primary"), name="D1 K-sensitivity selected primary"
    )
    if selected.get("method") not in K_SENSITIVITY_METHODS or selected.get(
        "configuration_sha256"
    ) != canonical_sha256(selected.get("configuration")):
        raise RuntimeError("D1 K-sensitivity selected-primary binding differs")
    allnms_max = plan.get("allnms_max_candidates")
    if (
        not isinstance(allnms_max, int)
        or isinstance(allnms_max, bool)
        or allnms_max <= 0
    ):
        raise RuntimeError("D1 K-sensitivity AllNMS maximum is invalid")
    candidate_maxima = _require_mapping(
        plan.get("candidate_manifest_maxima"),
        name="D1 K-sensitivity candidate manifest maxima",
    )
    observed_allnms_maxima: list[int] = []
    for split in ("train", "validation"):
        split_maxima = _require_mapping(
            candidate_maxima.get(split),
            name=f"D1 K-sensitivity {split} candidate maxima",
        )
        top10_max = split_maxima.get("top10")
        split_allnms_max = split_maxima.get("allnms")
        if (
            not isinstance(top10_max, int)
            or isinstance(top10_max, bool)
            or top10_max <= 0
            or top10_max > 10
            or not isinstance(split_allnms_max, int)
            or isinstance(split_allnms_max, bool)
            or split_allnms_max < top10_max
        ):
            raise RuntimeError(f"D1 K-sensitivity {split} candidate maxima are invalid")
        observed_allnms_maxima.append(split_allnms_max)
    if allnms_max != max(observed_allnms_maxima):
        raise RuntimeError("D1 K-sensitivity AllNMS maximum differs from manifests")
    scenarios = _scenario_definitions(allnms_max)
    if plan.get("scenario_definitions") != scenarios:
        raise RuntimeError("D1 K-sensitivity scenarios differ from the frozen contract")
    expected_jobs = _jobs(selected_primary=selected, scenarios=scenarios)
    if plan.get("jobs") != expected_jobs or plan.get("job_count") != len(expected_jobs):
        raise RuntimeError("D1 K-sensitivity exact job universe differs")
    if plan.get("job_universe_sha256") != canonical_sha256(
        [job["configuration"] for job in expected_jobs]
    ) or plan.get("job_ids_sha256") != canonical_sha256(
        [job["job_id"] for job in expected_jobs]
    ):
        raise RuntimeError("D1 K-sensitivity job-universe hashes differ")
    execution = _require_mapping(
        plan.get("execution_contract"), name="D1 K-sensitivity execution contract"
    )
    if (
        execution.get("runner_module") != K_SENSITIVITY_RUNNER_MODULE
        or execution.get("runner_implemented") is not True
        or execution.get("authorization_tool")
        != "tools.d1_reranking.authorize_k_sensitivity_execution"
        or execution.get("orchestrator_tool")
        != "tools.d1_reranking.run_k_sensitivity_matrix"
        or execution.get("authorization_required") is not True
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("execution_manifest")
        != "configs/d1_k_sensitivity_execution.json"
        or execution.get("execution_state_transitions")
        != ["ACTIVE", "COMPLETE", "FAILED"]
        or execution.get("max_parallel") != 1
        or execution.get("device") != "cpu"
        or execution.get("fresh_resource_gate_required_on_start_and_resume") is not True
        or execution.get("existing_train_primary_cell_compatible") is not False
    ):
        raise RuntimeError("D1 K-sensitivity execution boundary differs")
    sources = _require_mapping(plan.get("sources"), name="D1 K-sensitivity sources")
    if plan.get("source_signature_sha256") != canonical_sha256(sources):
        raise RuntimeError("D1 K-sensitivity source signature differs")
    return plan


def load_k_sensitivity_plan(path: str | Path) -> dict[str, Any]:
    """Load the strict plan and replay it from every bound source byte."""

    source = Path(path).expanduser().resolve()
    plan = load_content_manifest(
        source, name="D1 K-sensitivity plan", statuses=("PLANNED",)
    )
    validated = validate_k_sensitivity_plan(plan)
    verify_artifact_records_recursive(
        validated.get("sources"),
        name="D1 K-sensitivity sources",
        require_at_least_one=True,
    )
    sources = _require_mapping(
        validated.get("sources"), name="D1 K-sensitivity sources"
    )
    code_records = sources.get("code")
    if not isinstance(code_records, Sequence) or isinstance(
        code_records, (str, bytes, bytearray)
    ):
        raise RuntimeError("D1 K-sensitivity code-source inventory is invalid")
    tool_paths = tuple(
        verified_artifact_path(
            _require_mapping(record, name=f"D1 K-sensitivity code source {index}"),
            name=f"D1 K-sensitivity code source {index}",
        )
        for index, record in enumerate(code_records)
    )
    expected = build_k_sensitivity_plan(source.parent.parent, tool_paths=tool_paths)
    if validated != expected:
        raise RuntimeError("D1 K-sensitivity plan differs from its live source replay")
    return validated


def validate_k_sensitivity_result(
    value: Mapping[str, Any],
    *,
    plan_path: str | Path,
    job: Mapping[str, Any],
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify one COMPLETE cell against its exact planned job and live artifacts."""

    result = {str(key): child for key, child in value.items()}
    unsigned = dict(result)
    observed_content = unsigned.pop("content_sha256", None)
    if observed_content != canonical_sha256(unsigned):
        raise RuntimeError("D1 K-sensitivity result content hash mismatch")
    configuration = _require_mapping(
        job.get("configuration"), name="D1 K-sensitivity planned job configuration"
    )
    job_id = str(job.get("job_id", ""))
    if (
        result.get("status") != "COMPLETE"
        or result.get("job_id") != job_id
        or result.get("configuration") != configuration
        or result.get("configuration_sha256") != canonical_sha256(configuration)
        or result.get("candidate_test_labels_read") is not False
        or result.get("test_inputs_referenced") is not False
    ):
        raise RuntimeError("D1 K-sensitivity result/job contract differs")
    sources = _require_mapping(
        result.get("sources"), name="D1 K-sensitivity result sources"
    )
    if sources.get("plan") != artifact_record(plan_path):
        raise RuntimeError("D1 K-sensitivity result plan binding differs")
    plan = load_k_sensitivity_plan(plan_path)
    plan_sources = _require_mapping(
        plan.get("sources"), name="D1 K-sensitivity plan sources"
    )
    scenario_id = str(configuration.get("scenario_id", ""))
    pool = str(configuration.get("pool", ""))
    expected_candidate_manifests = _require_mapping(
        plan_sources.get("candidate_manifests"),
        name="D1 K-sensitivity planned candidate manifests",
    )
    expected_feature_manifests = _require_mapping(
        _require_mapping(
            plan_sources.get("feature_manifests"),
            name="D1 K-sensitivity planned feature manifests",
        ).get(scenario_id),
        name="D1 K-sensitivity planned scenario feature manifests",
    )
    expected_label_manifests = _require_mapping(
        _require_mapping(
            plan_sources.get("development_label_manifests"),
            name="D1 K-sensitivity planned label manifests",
        ).get(pool),
        name="D1 K-sensitivity planned pool label manifests",
    )
    expected_calibration = _require_mapping(
        _require_mapping(
            plan_sources.get("calibration_manifests"),
            name="D1 K-sensitivity planned calibration manifests",
        ).get(pool),
        name="D1 K-sensitivity planned calibration manifest",
    )
    expected_source_values = {
        "selected_primary": plan_sources.get("selected_primary"),
        "selected_trial": plan_sources.get("selected_trial"),
        "candidate_manifests": expected_candidate_manifests,
        "feature_manifests": expected_feature_manifests,
        "development_label_manifests": expected_label_manifests,
        "calibration_manifest": expected_calibration,
        "fold_assignments": plan_sources.get("fold_assignments"),
        "denominators": plan_sources.get("denominators"),
    }
    if any(sources.get(key) != value for key, value in expected_source_values.items()):
        raise RuntimeError("D1 K-sensitivity result/plan source closure differs")
    required_sources = {
        "plan",
        "execution_manifest",
        "execution_event",
        "execution_claim",
        "selected_primary",
        "selected_trial",
        "candidate_manifests",
        "feature_manifests",
        "development_label_manifests",
        "calibration_manifest",
        "fold_assignments",
        "denominators",
        "runner_code",
        "numerical_code",
    }
    if set(sources) != required_sources:
        raise RuntimeError("D1 K-sensitivity result source inventory differs")
    runner_record = artifact_record(
        Path(__file__).resolve().parents[2]
        / "tools"
        / "d1_reranking"
        / "run_k_sensitivity_cell.py"
    )
    if sources.get("runner_code") != runner_record:
        raise RuntimeError("D1 K-sensitivity result runner binding differs")
    planned_code = plan_sources.get("code")
    if not isinstance(planned_code, Sequence) or runner_record not in planned_code:
        raise RuntimeError("D1 K-sensitivity plan does not bind the cell runner")
    repository_root = Path(__file__).resolve().parents[2]
    expected_numerical_code = [
        artifact_record(path)
        for path in (
            repository_root / "src/d1_reranking/models.py",
            repository_root / "src/d1_reranking/fold_calibration.py",
            repository_root / "src/unified_reranking/datasets.py",
            repository_root / "src/unified_reranking/training.py",
            repository_root / "src/unified_reranking/losses.py",
            repository_root / "src/unified_reranking/models/lightgbm_ranker.py",
            repository_root / "src/unified_reranking/metrics.py",
            repository_root / "src/unified_reranking/telemetry.py",
        )
    ]
    if sources.get("numerical_code") != expected_numerical_code:
        raise RuntimeError("D1 K-sensitivity numerical-code binding differs")
    verify_artifact_records_recursive(
        sources, name="D1 K-sensitivity result sources", require_at_least_one=True
    )
    artifacts = _require_mapping(
        result.get("artifacts"), name="D1 K-sensitivity result artifacts"
    )
    if set(artifacts) != {"model", "preprocessor", "predictions", "decisions"}:
        raise RuntimeError("D1 K-sensitivity result artifact inventory differs")
    verified = verify_artifact_records_recursive(
        artifacts,
        name="D1 K-sensitivity result artifacts",
        require_at_least_one=True,
    )
    if manifest_path is not None:
        expected_manifest = (
            Path(plan_path).expanduser().resolve().parent.parent
            / str(job.get("output_manifest", ""))
        ).resolve()
        observed_manifest = Path(manifest_path).expanduser().resolve()
        if observed_manifest != expected_manifest:
            raise RuntimeError("D1 K-sensitivity result manifest path differs")
        expected_parent = observed_manifest.parent
        if any(Path(record["path"]).parent != expected_parent for record in verified):
            raise RuntimeError("D1 K-sensitivity result artifact path escapes its cell")
    preprocessor = _require_mapping(
        result.get("preprocessor"), name="D1 K-sensitivity preprocessor"
    )
    preprocessor_unsigned = dict(preprocessor)
    preprocessor_content = preprocessor_unsigned.pop("content_sha256", None)
    if preprocessor_content != canonical_sha256(preprocessor_unsigned):
        raise RuntimeError("D1 K-sensitivity preprocessor content hash mismatch")
    preprocessor_path = verified_artifact_path(
        _require_mapping(
            artifacts.get("preprocessor"),
            name="D1 K-sensitivity preprocessor artifact",
        ),
        name="D1 K-sensitivity preprocessor artifact",
    )
    try:
        persisted_preprocessor = json.loads(
            preprocessor_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "D1 K-sensitivity preprocessor artifact is unreadable"
        ) from error
    if persisted_preprocessor != preprocessor:
        raise RuntimeError("D1 K-sensitivity preprocessor artifact/payload differs")
    held_fold_value = configuration.get("held_fold")
    expected_early_fold = (
        (int(held_fold_value) + 1) % K_SENSITIVITY_OUTER_FOLDS
        if held_fold_value is not None
        else 0
    )
    excluded_folds = {expected_early_fold}
    if held_fold_value is not None:
        excluded_folds.add(int(held_fold_value))
    expected_fit_folds = [
        fold for fold in range(K_SENSITIVITY_OUTER_FOLDS) if fold not in excluded_folds
    ]
    fold_contract = _require_mapping(
        result.get("fold_contract"), name="D1 K-sensitivity fold contract"
    )
    fold_calibrator = _require_mapping(
        preprocessor.get("fold_calibrator"),
        name="D1 K-sensitivity fold calibrator",
    )
    fold_calibrator_unsigned = dict(fold_calibrator)
    fold_calibrator_sha256 = fold_calibrator_unsigned.pop("content_sha256", None)
    if (
        fold_contract
        != {
            "early_stop_fold": expected_early_fold,
            "fit_fold_ids": expected_fit_folds,
            "fold_local_preprocessing": True,
            "fold_local_calibration": True,
        }
        or fold_calibrator.get("fit_fold_ids") != expected_fit_folds
        or fold_calibrator_sha256 != canonical_sha256(fold_calibrator_unsigned)
    ):
        raise RuntimeError("D1 K-sensitivity fold-local fitting contract differs")
    model_contract = _require_mapping(
        result.get("model_contract"), name="D1 K-sensitivity model contract"
    )
    model_unsigned = dict(model_contract)
    model_content = model_unsigned.pop("content_sha256", None)
    method = str(configuration.get("method", ""))
    expected_serialization = (
        "lightgbm_native_text" if method == "R5" else "torch_state_dict"
    )
    model_path = verified_artifact_path(
        _require_mapping(
            artifacts.get("model"), name="D1 K-sensitivity model artifact"
        ),
        name="D1 K-sensitivity model artifact",
    )
    if (
        model_content != canonical_sha256(model_unsigned)
        or model_contract.get("method") != configuration.get("method")
        or model_contract.get("seed") != configuration.get("seed")
        or model_contract.get("selected_primary_trial_id")
        != configuration.get("selected_primary_trial_id")
        or model_contract.get("selected_primary_configuration_sha256")
        != configuration.get("selected_primary_configuration_sha256")
        or model_contract.get("effective_training_hyperparameters")
        != selected_training_spec(
            _require_mapping(
                configuration.get("selected_primary_configuration"),
                name="D1 K-sensitivity selected primary configuration",
            ),
            method=str(configuration.get("method", "")),
        )
        or model_contract.get("serialization") != expected_serialization
        or (method == "R5" and model_path.suffix != ".txt")
        or (method != "R5" and model_path.suffix != ".pt")
    ):
        raise RuntimeError("D1 K-sensitivity fitted-model binding differs")
    prediction_contract = _require_mapping(
        result.get("prediction_contract"),
        name="D1 K-sensitivity prediction contract",
    )
    decision_contract = _require_mapping(
        result.get("decision_contract"), name="D1 K-sensitivity decision contract"
    )
    for name, contract in (
        ("prediction", prediction_contract),
        ("decision", decision_contract),
    ):
        contract_unsigned = dict(contract)
        contract_content = contract_unsigned.pop("content_sha256", None)
        if (
            not isinstance(contract.get("rows"), int)
            or contract.get("rows", -1) < 0
            or contract_content != canonical_sha256(contract_unsigned)
        ):
            raise RuntimeError(f"D1 K-sensitivity {name} contract is invalid")
    prediction_path = verified_artifact_path(
        _require_mapping(
            artifacts.get("predictions"),
            name="D1 K-sensitivity prediction artifact",
        ),
        name="D1 K-sensitivity prediction artifact",
    )
    predictions = pd.read_parquet(prediction_path)
    expected_prediction_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
        "score",
    ]
    if list(predictions.columns) != expected_prediction_columns:
        raise RuntimeError("D1 K-sensitivity prediction schema differs")
    ordered_predictions = predictions.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    prediction_identity = [
        [
            str(row.sample_id),
            str(row.candidate_id),
            int(row.native_rank),
            str(row.candidate_identity_sha256),
            str(row.candidate_geometry_sha256),
        ]
        for row in ordered_predictions.itertuples(index=False)
    ]
    prediction_scores = [
        [str(row.sample_id), str(row.candidate_id), float(row.score).hex()]
        for row in ordered_predictions.itertuples(index=False)
    ]
    prediction_split = "train" if configuration.get("mode") == "oof" else "validation"
    candidate_manifest_path = verified_artifact_path(
        _require_mapping(
            expected_candidate_manifests.get(prediction_split),
            name=f"D1 K-sensitivity planned {prediction_split} candidates",
        ),
        name=f"D1 K-sensitivity planned {prediction_split} candidates",
    )
    candidate_manifest = load_content_manifest(
        candidate_manifest_path,
        name=f"D1 K-sensitivity {prediction_split} candidate manifest",
        statuses=("COMPLETE",),
    )
    candidate_artifacts = _require_mapping(
        candidate_manifest.get("artifacts"),
        name=f"D1 K-sensitivity {prediction_split} candidate artifacts",
    )
    candidate_path = verified_artifact_path(
        _require_mapping(
            candidate_artifacts.get(pool),
            name=f"D1 K-sensitivity {prediction_split}/{pool} candidates",
        ),
        name=f"D1 K-sensitivity {prediction_split}/{pool} candidates",
    )
    expected_candidates = pd.read_parquet(
        candidate_path,
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    )
    folds_path = verified_artifact_path(
        _require_mapping(
            plan_sources.get("fold_assignments"),
            name="D1 K-sensitivity planned folds",
        ),
        name="D1 K-sensitivity planned folds",
    )
    folds = pd.read_parquet(folds_path, columns=["sample_id", "fold"])
    if prediction_split == "train":
        held_fold = configuration.get("held_fold")
        allowed_samples = set(
            folds.loc[folds["fold"] == held_fold, "sample_id"].astype(str)
        )
        expected_candidates = expected_candidates.loc[
            expected_candidates["sample_id"].astype(str).isin(allowed_samples)
        ]
    expected_candidates = expected_candidates.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    expected_prediction_identity = [
        [
            str(row.sample_id),
            str(row.candidate_id),
            int(row.native_rank),
            str(row.candidate_identity_sha256),
            str(row.candidate_geometry_sha256),
        ]
        for row in expected_candidates.itertuples(index=False)
    ]
    if (
        prediction_contract.get("rows") != len(ordered_predictions)
        or prediction_contract.get("candidate_universe_sha256")
        != canonical_sha256(prediction_identity)
        or prediction_identity != expected_prediction_identity
        or prediction_contract.get("score_vector_sha256")
        != canonical_sha256(prediction_scores)
    ):
        raise RuntimeError("D1 K-sensitivity prediction content contract differs")
    decision_path = verified_artifact_path(
        _require_mapping(
            artifacts.get("decisions"), name="D1 K-sensitivity decision artifact"
        ),
        name="D1 K-sensitivity decision artifact",
    )
    decisions = pd.read_parquet(decision_path).reset_index(drop=True)
    required_decision_columns = {
        "sample_id",
        "selected_candidate_id",
        "selected_correct",
    }
    if not required_decision_columns.issubset(decisions.columns):
        raise RuntimeError("D1 K-sensitivity decision schema differs")
    decision_samples = decisions["sample_id"].astype(str).tolist()
    if prediction_split == "train":
        expected_decision_samples = (
            folds.loc[folds["fold"] == configuration.get("held_fold"), "sample_id"]
            .astype(str)
            .tolist()
        )
    else:
        denominator_records = _require_mapping(
            plan_sources.get("denominators"),
            name="D1 K-sensitivity planned denominators",
        )
        denominator_path = verified_artifact_path(
            _require_mapping(
                denominator_records.get("validation"),
                name="D1 K-sensitivity Validation denominator",
            ),
            name="D1 K-sensitivity Validation denominator",
        )
        expected_decision_samples = (
            pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
            .astype(str)
            .tolist()
        )
    decision_selections = [
        [
            str(row.sample_id),
            None
            if pd.isna(row.selected_candidate_id)
            else str(row.selected_candidate_id),
            bool(row.selected_correct),
        ]
        for row in decisions.itertuples(index=False)
    ]
    if (
        decision_contract.get("rows") != len(decisions)
        or decision_contract.get("sample_universe_sha256")
        != canonical_sha256(decision_samples)
        or decision_samples != expected_decision_samples
        or decision_contract.get("selected_vector_sha256")
        != canonical_sha256(decision_selections)
    ):
        raise RuntimeError("D1 K-sensitivity decision content contract differs")
    telemetry = _require_mapping(
        result.get("telemetry"), name="D1 K-sensitivity telemetry"
    )
    flat = result.get("telemetry_flat")
    expected_flat = {
        key: telemetry.get(key)
        for key in (
            "parameter_count",
            "ranker_latency_ms",
            "feature_latency_ms",
            "peak_memory_mb",
            "missing_feature_rate",
        )
    }
    if flat != expected_flat:
        raise RuntimeError("D1 K-sensitivity flattened telemetry differs")
    execution_record = _require_mapping(
        sources.get("execution_manifest"),
        name="D1 K-sensitivity execution-manifest record",
    )
    execution_path = verified_artifact_path(
        execution_record, name="D1 K-sensitivity execution manifest"
    )
    execution = load_content_manifest(
        execution_path,
        name="D1 K-sensitivity execution manifest",
        statuses=("ACTIVE",),
    )
    execution_id = str(execution.get("execution_id", ""))
    run_root = Path(plan_path).expanduser().resolve().parent.parent
    expected_execution_path = (
        execution_directory(run_root, execution_id) / "execution.json"
    )
    if execution_path != expected_execution_path:
        raise RuntimeError("D1 K-sensitivity execution-manifest path differs")
    event_record = _require_mapping(
        sources.get("execution_event"), name="D1 K-sensitivity execution event"
    )
    event_path = verified_artifact_path(
        event_record, name="D1 K-sensitivity execution event"
    )
    event = load_content_manifest(
        event_path, name="D1 K-sensitivity execution event", statuses=("ACTIVE",)
    )
    claim_record = _require_mapping(
        sources.get("execution_claim"), name="D1 K-sensitivity execution claim"
    )
    claim_path = verified_artifact_path(
        claim_record, name="D1 K-sensitivity execution claim"
    )
    claim = load_content_manifest(
        claim_path, name="D1 K-sensitivity execution claim", statuses=("CLAIMED",)
    )
    expected_claim_path = (
        execution_directory(run_root, execution_id) / "claims" / f"{job_id}.json"
    )
    if (
        event_path.parent != execution_directory(run_root, execution_id) / "events"
        or event.get("execution_id") != execution_id
        or event.get("current_job_id") != job_id
        or event.get("claim") != claim_record
        or claim_path != expected_claim_path
        or claim.get("execution_id") != execution_id
        or claim.get("job_id") != job_id
        or claim.get("configuration_sha256") != canonical_sha256(configuration)
    ):
        raise RuntimeError("D1 K-sensitivity execution event/claim binding differs")
    execution_provenance = _require_mapping(
        result.get("execution_provenance"),
        name="D1 K-sensitivity execution provenance",
    )
    expected_scope = K_EXECUTION_SCOPE
    expected_lease = str(
        Path(plan_path).expanduser().resolve().parent.parent.parent
        / ".d1_heavy_resource.lock"
    )
    if (
        execution.get("plan") != artifact_record(plan_path)
        or execution.get("scope") != expected_scope
        or execution_provenance
        != {
            "execution_id": execution.get("execution_id"),
            "execution_manifest": execution_record,
            "execution_event": event_record,
            "execution_claim": claim_record,
            "resource_gate": execution.get("resource_gate"),
            "resource_lease_path": expected_lease,
            "scope": expected_scope,
        }
    ):
        raise RuntimeError("D1 K-sensitivity execution provenance differs")
    output_signature = canonical_sha256(
        {
            "configuration": configuration,
            "sources": sources,
            "artifacts": artifacts,
            "model_contract": model_contract,
            "preprocessor": preprocessor,
            "prediction_contract": prediction_contract,
            "decision_contract": decision_contract,
            "telemetry": telemetry,
        }
    )
    if result.get("output_signature_sha256") != output_signature:
        raise RuntimeError("D1 K-sensitivity output signature differs")
    return result


def write_k_sensitivity_plan(
    destination: str | Path,
    *,
    run_dir: str | Path,
    tool_paths: Sequence[Path],
    resume: bool,
) -> dict[str, Any]:
    """Write once, or resume only when the full plan and all sources are exact."""

    path = Path(destination).expanduser().resolve()
    if path.exists() and not resume:
        raise FileExistsError(f"D1 K-sensitivity plan already exists: {path}")
    expected = build_k_sensitivity_plan(run_dir, tool_paths=tool_paths)
    if path.exists():
        existing = load_k_sensitivity_plan(path)
        if existing != expected:
            raise RuntimeError("D1 K-sensitivity resume plan/source universe differs")
        return existing
    atomic_json(path, expected)
    return load_k_sensitivity_plan(path)
