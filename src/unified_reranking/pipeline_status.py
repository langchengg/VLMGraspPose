"""Read-only, label-safe readiness graph for unified reranking stages P2-P15."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hashing import atomic_json, sha256_file


ROUTES = ("crog", "g1", "c1")
SPLITS = ("train", "validation", "test")
TRACKS = ("T1_native", "T2_matched_common", "T3_tri_backend")
LEGACY_WORKER_PATTERN = re.compile(
    r"^(?:\S*/)?python(?:\d+(?:\.\d+)*)?\s+-m\s+reranking\.run_experiment_matrix(?:\s|$)"
)


@dataclass(frozen=True)
class ArtifactRequirement:
    path: str
    statuses: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class StageSpec:
    name: str
    title: str
    dependencies: tuple[str, ...]
    requirements: tuple[ArtifactRequirement, ...]
    next_commands: tuple[str, ...]
    legacy_worker_interlock: bool = False


def _requirements() -> tuple[StageSpec, ...]:
    development_labels = tuple(
        ArtifactRequirement(
            f"03_features/candidate_labels_{route}_{split}_top5.parquet",
            description=f"{route}/{split} development Top-5 labels",
        )
        for route in ROUTES
        for split in ("train", "validation")
    )
    feature_tracks = tuple(
        ArtifactRequirement(
            f"03_features/tracks/{track}/{route}_{split}/feature_manifest.json",
            ("COMPLETE",),
            f"{route}/{split}/{track} features",
        )
        for route in ROUTES
        for split in SPLITS
        for track in TRACKS
    )
    calibrations = tuple(
        requirement
        for route in ROUTES
        for requirement in (
            ArtifactRequirement(
                f"05_calibration/{route}_calibration_manifest.json",
                ("COMPLETE",),
                f"{route} grouped OOF calibration",
            ),
            ArtifactRequirement(
                f"05_calibration/{route}_test_application_manifest.json",
                ("COMPLETE_LABEL_FREE",),
                f"{route} label-free Test calibration",
            ),
        )
    )
    gates = tuple(
        requirement
        for route in ROUTES
        for requirement in (
            ArtifactRequirement(
                f"08_lock/gates/{route}/gate_selection.json",
                ("COMPLETE",),
                f"{route} Validation gate",
            ),
            ArtifactRequirement(
                f"08_lock/label_free_test_gates/{route}/manifest.json",
                ("COMPLETE",),
                f"{route} label-free Test gate application",
            ),
        )
    )
    final_reports = tuple(
        ArtifactRequirement(f"14_reports/{name}", description=name)
        for name in (
            "FINAL_SUMMARY_ZH.md",
            "FINAL_REPORT_EN.md",
            "METHODS_UNIFIED_RERANKING_EN.md",
            "RESULTS_UNIFIED_RERANKING_EN.md",
            "DISCUSSION_UNIFIED_RERANKING_EN.md",
            "LIMITATIONS_EN.md",
            "MATERIAL_PASSPORT.md",
            "EXPERIMENT_CONCLUSION.json",
        )
    )
    return (
        StageSpec(
            "P2",
            "Evaluator and physically separated development supervision",
            (),
            (
                ArtifactRequirement("configs/canonical_evaluator.py", description="frozen evaluator"),
                ArtifactRequirement(
                    "03_features/development_label_audit.json",
                    ("PASS", "COMPLETE", "COMPLETE_FOR_AVAILABLE_CANDIDATES"),
                    "development label audit",
                ),
                *development_labels,
            ),
            ("python -m tools.unified_reranking.prepare_development --run-dir {run_dir}",),
        ),
        StageSpec(
            "P3_P4",
            "Leakage-safe T1/T2/T3 feature tracks",
            ("P2",),
            feature_tracks,
            (
                "python -m tools.unified_reranking.prepare_feature_tracks --run-dir {run_dir} --route <route> --split <split> --track <T1_or_T2>",
                "python -m tools.unified_reranking.build_consensus_features --run-dir {run_dir} --split <split>",
            ),
        ),
        StageSpec(
            "P5",
            "Grouped folds and leakage audit",
            ("P2",),
            (
                ArtifactRequirement("04_splits/fold_assignments.parquet", description="fold assignments"),
                ArtifactRequirement("04_splits/split_leakage_audit.json", ("PASS",), "split leakage audit"),
            ),
            ("python -m tools.unified_reranking.prepare_development --run-dir {run_dir}",),
        ),
        StageSpec(
            "P6",
            "Route calibration and label-free Test application",
            ("P2", "P5"),
            calibrations,
            (
                "python -m tools.unified_reranking.fit_calibration --run-dir {run_dir} --route <route>",
                "python -m tools.unified_reranking.apply_locked_calibration --run-dir {run_dir} --route <route> --split test",
            ),
        ),
        StageSpec(
            "P7",
            "Controlled matrix and three-seed T2 primary selection",
            ("P3_P4", "P5", "P6"),
            (
                ArtifactRequirement("05_models/screen_finalists.json", description="screen finalists"),
                ArtifactRequirement("07_validation/selected_primary_ungated.json", ("VALIDATION_LOCKED",), "primary rankers"),
                ArtifactRequirement("08_lock/label_free_test_rankers/manifest.json", ("COMPLETE",), "label-free Test rankers"),
            ),
            (
                "python -m tools.unified_reranking.train_matrix --run-dir {run_dir} --phase screen --execute --max-parallel 1",
                "python -m tools.unified_reranking.select_validation_screen --run-dir {run_dir}",
                "python -m tools.unified_reranking.train_matrix --run-dir {run_dir} --phase selected --selection-json {run_dir}/05_models/screen_finalists.json --execute --max-parallel 1",
                "python -m tools.unified_reranking.select_primary_rankers --run-dir {run_dir}",
                "python -m tools.unified_reranking.apply_selected_test_rankers --run-dir {run_dir}",
            ),
            legacy_worker_interlock=True,
        ),
        StageSpec(
            "P7_ENCODER",
            "Matched-budget encoder comparison after loss selection",
            ("P7",),
            (
                ArtifactRequirement(
                    "05_models/matrix_plans/encoder_latest_execution.json",
                    ("COMPLETE",),
                    "complete encoder-phase execution",
                ),
            ),
            (
                "python -m tools.unified_reranking.train_matrix --run-dir {run_dir} --phase encoder --selection-json {run_dir}/05_models/encoder_loss_selections.json --execute --max-parallel 1",
            ),
            legacy_worker_interlock=True,
        ),
        StageSpec(
            "P7_ABLATION",
            "Validation-only cumulative and leave-one-family-out feature ablations",
            ("P7",),
            (
                ArtifactRequirement(
                    "07_validation/ablations/feature_ablation_manifest.json",
                    ("COMPLETE",),
                    "source-locked Validation feature ablation manifest",
                ),
                ArtifactRequirement(
                    "07_validation/ablations/cumulative_feature_ablation.csv",
                    description="cumulative Validation feature ablation table",
                ),
                ArtifactRequirement(
                    "07_validation/ablations/leave_one_family_out_ablation.csv",
                    description="leave-one-family-out Validation ablation table",
                ),
            ),
            (
                "python -m tools.unified_reranking.run_validation_feature_ablations --run-dir {run_dir}",
            ),
            legacy_worker_interlock=True,
        ),
        StageSpec(
            "P8",
            "Conservative gates",
            ("P7",),
            gates,
            (
                "python -m tools.unified_reranking.prepare_gate_inputs --run-dir {run_dir}",
                "python -m tools.unified_reranking.select_gate --run-dir {run_dir} --route <route> ...",
                "python -m tools.unified_reranking.apply_locked_test_gates --run-dir {run_dir}",
            ),
            legacy_worker_interlock=True,
        ),
        StageSpec(
            "P9",
            "CROG-default router and union-headroom decision",
            ("P8",),
            (
                ArtifactRequirement("08_lock/route_router/route_router_selection.json", ("COMPLETE",), "Validation route router"),
                ArtifactRequirement("08_lock/route_router_test/manifest.json", ("COMPLETE",), "label-free Test router"),
                ArtifactRequirement("07_validation/union_headroom/manifest.json", ("COMPLETE",), "union headroom"),
            ),
            (
                "python -m tools.unified_reranking.prepare_route_router_inputs --run-dir {run_dir}",
                "python -m tools.unified_reranking.select_route_router --run-dir {run_dir} ...",
                "python -m tools.unified_reranking.prepare_label_free_test_router_inputs --run-dir {run_dir}",
                "python -m tools.unified_reranking.apply_locked_route_router --run-dir {run_dir}",
                "python -m tools.unified_reranking.analyze_union_headroom --run-dir {run_dir}",
                "# If UNION_HEADROOM_AVAILABLE: python -m tools.unified_reranking.prepare_union_features --run-dir {run_dir} --split train && python -m tools.unified_reranking.prepare_union_features --run-dir {run_dir} --split validation && python -m tools.unified_reranking.prepare_union_features --run-dir {run_dir} --split test",
                "# If UNION_HEADROOM_AVAILABLE: python -m tools.unified_reranking.train_union_rankers --run-dir {run_dir} --execute && python -m tools.unified_reranking.apply_locked_union_ranker --run-dir {run_dir}",
            ),
            legacy_worker_interlock=True,
        ),
        StageSpec(
            "P10",
            "Validation 2x2 attribution bridges",
            ("P2",),
            tuple(
                ArtifactRequirement(
                    f"11_attribution_bridge/bridge_{route}_{split}_top5_manifest.json",
                    ("COMPLETE",),
                    f"{route} {split} bridge",
                )
                for route in ("g1", "c1")
                for split in ("train", "validation")
            )
            + (
                ArtifactRequirement(
                    "07_validation/bridge_train_validation.csv",
                    description="consolidated Train/Validation bridge",
                ),
                ArtifactRequirement(
                    "11_attribution_bridge/bridge_train_validation.csv",
                    description="prompt-facing Train/Validation bridge alias",
                ),
                ArtifactRequirement(
                    "11_attribution_bridge/test_bridge_input/manifest.json",
                    ("COMPLETE",),
                    "label-free Test bridge input",
                ),
            ),
            (
                "python -m tools.unified_reranking.run_attribution_bridge --run-dir {run_dir} --route <g1_or_c1> --split <train_or_validation> --pool top5",
                "python -m tools.unified_reranking.prepare_test_bridge_bundle --run-dir {run_dir}",
            ),
        ),
        StageSpec(
            "P11_PRELOCK",
            "Hash-bound P11 pre-lock assembly",
            ("P6", "P7", "P7_ENCODER", "P7_ABLATION", "P8", "P9", "P10"),
            (
                ArtifactRequirement("08_lock/prelock_assembly_manifest.json", ("COMPLETE",), "semantic pre-lock bundle"),
            ),
            ("python -m tools.unified_reranking.assemble_prelock --run-dir {run_dir}",),
            legacy_worker_interlock=True,
        ),
        StageSpec(
            "P11",
            "Formal Test lock",
            ("P11_PRELOCK",),
            (ArtifactRequirement("08_lock/FORMAL_TEST_LOCK.json", ("LOCKED",), "formal Test lock"),),
            (
                "python -m tools.unified_reranking.create_formal_test_lock --run-dir {run_dir} --evaluation-plan {run_dir}/08_lock/formal_evaluation_plan.json --extra-locked-file code_manifest={run_dir}/08_lock/code_manifest.json --extra-locked-file prelock_assembly={run_dir}/08_lock/prelock_assembly_manifest.json",
            ),
            legacy_worker_interlock=True,
        ),
        StageSpec(
            "P12",
            "Single formal execution",
            ("P11",),
            (
                ArtifactRequirement("09_formal_test/FORMAL_TEST_EXECUTION.json", ("COMPLETE",), "single formal execution"),
                ArtifactRequirement("09_formal_test/formal_test_manifest.json", ("COMPLETE",), "formal Test manifest"),
                ArtifactRequirement("09_formal_test/per_candidate_scores.parquet", description="formal per-candidate scores"),
                ArtifactRequirement("09_formal_test/bridge_per_candidate_scores.parquet", description="formal 2x2 bridge per-candidate scores"),
                ArtifactRequirement("09_formal_test/per_sample_decisions.parquet", description="formal per-sample decisions"),
            ),
            ("python -m tools.unified_reranking.run_formal_test_once --run-dir {run_dir}",),
        ),
        StageSpec(
            "P15",
            "Independent recompute",
            ("P12",),
            (
                ArtifactRequirement("15_independent_recompute/INDEPENDENT_RECOMPUTE.json", description="independent recompute JSON"),
                ArtifactRequirement("15_independent_recompute/INDEPENDENT_RECOMPUTE.md", description="independent recompute report"),
            ),
            ("python -m tools.unified_reranking.independent_recompute --run-dir {run_dir}",),
        ),
        StageSpec(
            "P13",
            "Failure taxonomy",
            ("P12", "P15"),
            (
                ArtifactRequirement("tables/failure_taxonomy.csv", description="failure taxonomy"),
                ArtifactRequirement(
                    "11_attribution_bridge/bridge_postlock_test.csv",
                    description="secondary post-lock Test bridge",
                ),
            ),
            ("python -m tools.unified_reranking.build_postformal_artifacts --run-dir {run_dir}",),
        ),
        StageSpec(
            "P14",
            "Figures, deterministic galleries, and bilingual reports",
            ("P12", "P13"),
            final_reports,
            ("python -m tools.unified_reranking.build_postformal_artifacts --run-dir {run_dir}",),
        ),
        StageSpec(
            "POSTFORMAL",
            "Terminal postformal bundle and final integrity lock",
            ("P13", "P14", "P15"),
            (
                ArtifactRequirement("FINAL_RUN_LOCK.json", description="final integrity lock"),
                ArtifactRequirement("FINAL_RUN_SHA256.txt", description="final run digest"),
                ArtifactRequirement("COMPLETE", description="terminal COMPLETE marker"),
            ),
            ("python -m tools.unified_reranking.build_postformal_artifacts --run-dir {run_dir}",),
        ),
    )


STAGES = _requirements()
STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


def active_legacy_ranker_workers(process_rows: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Return active legacy matrix workers without mutating or signalling them."""

    if process_rows is None:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = completed.stdout.splitlines()
    else:
        rows = list(process_rows)
    result: list[dict[str, Any]] = []
    for raw in rows:
        value = str(raw).strip()
        if not value:
            continue
        match = re.match(r"(?P<pid>\d+)\s+(?P<command>.+)", value)
        command = value if match is None else match.group("command")
        if not LEGACY_WORKER_PATTERN.search(command):
            continue
        result.append(
            {
                "pid": None if match is None else int(match.group("pid")),
                "command": command,
            }
        )
    return result


def _artifact_state(run_dir: Path, requirement: ArtifactRequirement) -> dict[str, Any]:
    path = (run_dir / requirement.path).resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "description": requirement.description,
        "exists": path.is_file() and not path.is_symlink(),
    }
    if not result["exists"]:
        result["valid"] = False
        result["reason"] = "missing_regular_file"
        return result
    result["sha256"] = sha256_file(path)
    if not requirement.statuses:
        result["valid"] = True
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result["valid"] = False
        result["reason"] = f"invalid_json:{type(error).__name__}"
        return result
    observed = value.get("status") if isinstance(value, dict) else None
    result["observed_status"] = observed
    result["expected_statuses"] = list(requirement.statuses)
    result["valid"] = observed in requirement.statuses
    if not result["valid"]:
        result["reason"] = "unexpected_status"
    return result


def audit_pipeline_readiness(
    run_dir: str | Path,
    *,
    process_rows: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit the DAG without importing or opening any candidate-label table."""

    root = Path(run_dir).expanduser().resolve()
    workers = active_legacy_ranker_workers(process_rows)
    results: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        requirements = list(stage.requirements)
        if stage.name == "P9":
            headroom_path = root / "07_validation/union_headroom/manifest.json"
            try:
                headroom = json.loads(headroom_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                headroom = {}
            if headroom.get("decision") == "UNION_HEADROOM_AVAILABLE":
                requirements.extend(
                    (
                        ArtifactRequirement(
                            "08_lock/union_ranker/selected_union_ranker.json",
                            ("VALIDATION_LOCKED",),
                            "Validation-selected union ranker",
                        ),
                        ArtifactRequirement(
                            "08_lock/union_ranker_test/manifest.json",
                            ("COMPLETE",),
                            "label-free Test union application",
                        ),
                    )
                )
        artifacts = [_artifact_state(root, requirement) for requirement in requirements]
        complete = bool(artifacts) and all(record["valid"] for record in artifacts)
        final_readiness: dict[str, Any] | None = None
        if stage.name == "POSTFORMAL" and complete:
            try:
                from tools.unified_reranking.build_postformal_artifacts import (
                    verify_final_readiness,
                )

                final_readiness = verify_final_readiness(root)
            except Exception as error:
                final_readiness = {
                    "ready": False,
                    "reason": f"verification_error:{type(error).__name__}:{error}",
                }
            if not final_readiness.get("ready"):
                complete = False
                artifacts.append(
                    {
                        "path": str(root / "FINAL_RUN_LOCK.json"),
                        "description": "cryptographically verified final readiness",
                        "exists": True,
                        "valid": False,
                        "reason": str(final_readiness.get("reason", "verification_failed")),
                    }
                )
        unmet = [dependency for dependency in stage.dependencies if not results[dependency]["complete"]]
        interlocked = bool(
            workers
            and stage.legacy_worker_interlock
            and not complete
            and not unmet
        )
        if complete:
            status = "COMPLETE"
        elif interlocked:
            status = "BLOCKED_ACTIVE_LEGACY_WORKER"
        elif unmet:
            status = "BLOCKED_DEPENDENCIES"
        else:
            status = "READY"
        results[stage.name] = {
            "title": stage.title,
            "status": status,
            "complete": complete,
            "unmet_dependencies": unmet,
            "legacy_worker_interlock": interlocked,
            "missing_or_invalid": [record for record in artifacts if not record["valid"]],
            "next_commands": [command.format(run_dir=str(root)) for command in stage.next_commands],
            **({"final_readiness": final_readiness} if final_readiness is not None else {}),
        }
    first_actionable = next(
        (
            name
            for name, state in results.items()
            if not state["complete"] and state["status"] in {"READY", "BLOCKED_ACTIVE_LEGACY_WORKER"}
        ),
        None,
    )
    execution_path = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    execution: dict[str, Any] = {}
    if execution_path.is_file():
        try:
            loaded = json.loads(execution_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                execution = loaded
        except (OSError, json.JSONDecodeError):
            execution = {}
    access_path = root / "09_formal_test" / "test_access.log"
    label_read_events = 0
    if access_path.is_file():
        for line in access_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") in {
                "candidate_test_labels_read_once",
                "formal_bridge_test_ground_truth_read_once",
            }:
                label_read_events += 1
    execution_complete = (
        execution.get("status") == "COMPLETE"
        and int(execution.get("execution_count", 0)) == 1
    )
    if execution_complete:
        label_state = "FORMAL_TEST_COMPLETE"
    elif execution.get("status") == "RUNNING":
        label_state = "FORMAL_TEST_EXECUTION_CLAIMED"
    elif (root / "08_lock" / "FORMAL_TEST_LOCK.json").is_file():
        label_state = "FORMAL_TEST_LOCKED"
    else:
        label_state = "PRELOCK_LABEL_FREE"
    return {
        "schema_version": 1,
        "status": "COMPLETE" if results["POSTFORMAL"]["complete"] else "IN_PROGRESS",
        "run_dir": str(root),
        "candidate_test_labels_read": bool(execution_complete or label_read_events),
        "candidate_test_label_access_state": label_state,
        "candidate_test_label_read_event_count": label_read_events,
        "legacy_ranker_workers": workers,
        "first_actionable_stage": first_actionable,
        "stages": results,
    }


def assert_stage_ready(report: dict[str, Any], stage_name: str) -> None:
    if stage_name not in report["stages"]:
        raise ValueError(f"unknown stage: {stage_name}")
    stage = report["stages"][stage_name]
    if stage["complete"]:
        return
    if stage["legacy_worker_interlock"]:
        pids = [str(worker.get("pid")) for worker in report["legacy_ranker_workers"]]
        raise RuntimeError(
            f"{stage_name} refused while legacy reranking.run_experiment_matrix worker(s) are active: {', '.join(pids)}"
        )
    if stage["unmet_dependencies"]:
        raise RuntimeError(
            f"{stage_name} has unmet dependencies: {', '.join(stage['unmet_dependencies'])}"
        )
    missing = [record["description"] or record["path"] for record in stage["missing_or_invalid"]]
    raise RuntimeError(f"{stage_name} is incomplete; required artifacts: {', '.join(missing)}")


def write_pipeline_status(run_dir: str | Path, report: dict[str, Any]) -> Path:
    root = Path(run_dir).expanduser().resolve()
    path = root / "_PIPELINE_STATUS.json"
    atomic_json(path, report)
    return path


__all__ = [
    "STAGES",
    "STAGE_BY_NAME",
    "active_legacy_ranker_workers",
    "assert_stage_ready",
    "audit_pipeline_readiness",
    "write_pipeline_status",
]
