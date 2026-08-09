from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from failure_analysis.reranking_v3.formal import (
    FINAL_MANIFEST_KIND,
    FORMAL_METHOD_ALLOWLIST,
    PRELIMINARY_MANIFEST_KIND,
    PRIOR_TEST_EXPOSURE_DISCLOSURE,
    evaluate_lockcheck_once,
    independent_evaluate_once,
    lock_final,
    lock_preliminary,
    run_locked_inference_once,
    verify_formal_ranking_outputs,
    verify_scope_candidate_identity,
)
from failure_analysis.reranking_v3.protocol import verify_locked_manifest
from failure_analysis.reranking_v3.experiment_config import ENSEMBLE_SEEDS, PERTURBATIONS
from failure_analysis.reranking_v3.schema import artifact_identity


METHODS = (
    "q_only",
    "v2_locked_primary",
    "v3_full_head_scalar_gate",
    "v3_fcer_native",
    "v3_fcer_rgbd",
    "v3_locked_primary",
)
INFERENCE_CALLBACK = "failure_analysis.reranking_v3.formal_backend:run_formal_inference"
EVALUATOR_CALLBACK = (
    "failure_analysis.reranking_v3.formal_backend:"
    "run_independent_dual_track_evaluation"
)
GATE_POLICY = {
    "harm_cost": 5.0,
    "threshold": 0.1,
    "uncertainty_kappa": 1.0,
    "consensus": 2,
    "minimum_valid_fraction": 1.0,
}


def write_file(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
        encoding="utf-8",
    )
    return path


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {"sample_id": sample, "candidate_ids": [f"{sample}_c{index}" for index in range(5)]}
        for sample in ("sample-a", "sample-b")
    ]


def artifact_set(tmp_path: Path) -> dict[str, Any]:
    artifacts = tmp_path / "inputs"
    candidate = write_jsonl(artifacts / "candidates.jsonl", candidate_rows())
    result = {
        "config_path": write_file(artifacts / "config.json", "{}\n"),
        "checkpoint_paths": {
            "seed_1": write_file(artifacts / "checkpoint-1.pt", "one\n"),
            "seed_2": write_file(artifacts / "checkpoint-2.pt", "two\n"),
            "seed_3": write_file(artifacts / "checkpoint-3.pt", "three\n"),
        },
        "normalizer_paths": {
            "development": write_file(artifacts / "normalizer.json", "{}\n")
        },
        "gate_paths": {
            "policy": write_file(
                artifacts / "gate-policy.json", json.dumps(GATE_POLICY) + "\n"
            ),
            "checkpoint": write_file(artifacts / "gate.pt", "gate\n"),
        },
        "contract_path": write_file(artifacts / "contract.json", "{}\n"),
        "split_path": write_file(artifacts / "split.json", "{}\n"),
        "candidate_path": candidate,
        "evaluator_paths": {
            "corrected_scientific": write_file(artifacts / "corrected.py", "# corrected\n"),
            "legacy_official_compatibility": write_file(artifacts / "legacy.py", "# legacy\n"),
        },
        "v2_paths": {
            "manifest": write_file(artifacts / "v2-manifest.json", "{}\n"),
            "ranking": write_file(artifacts / "v2-ranking.jsonl", "{}\n"),
        },
    }
    result["lockcheck_corrected_labels"] = write_file(
        artifacts / "lockcheck-corrected-labels.jsonl", "synthetic\n"
    )
    result["lockcheck_legacy_labels"] = write_file(
        artifacts / "lockcheck-legacy-labels.jsonl", "synthetic\n"
    )
    result["test_corrected_labels"] = write_file(
        artifacts / "test-corrected-labels.jsonl", "synthetic\n"
    )
    result["test_legacy_labels"] = write_file(
        artifacts / "test-legacy-labels.jsonl", "synthetic\n"
    )
    feature_schema = write_file(artifacts / "feature-schema.json", "{}\n")
    oof_manifest = write_file(artifacts / "oof-manifest.json", "{}\n")
    evaluation_descriptor = {
        "schema_version": "3.0.0",
        "kind": "v3_formal_evaluation_descriptor",
        "status": "frozen_before_evaluation",
        "cohorts": {
            "lockcheck": {
                "candidate": artifact_identity(candidate),
                "corrected_labels": artifact_identity(result["lockcheck_corrected_labels"]),
                "legacy_labels": artifact_identity(result["lockcheck_legacy_labels"]),
            },
            "test": {
                "candidate": artifact_identity(candidate),
                "corrected_labels": artifact_identity(result["test_corrected_labels"]),
                "legacy_labels": artifact_identity(result["test_legacy_labels"]),
            },
        },
        "formal_methods": list(METHODS),
        "evaluator_callback": EVALUATOR_CALLBACK,
        "bootstrap_iterations": 10_000,
        "seed": 20260801,
    }
    evaluation_path = write_file(
        artifacts / "evaluation-descriptor.json",
        json.dumps(evaluation_descriptor, sort_keys=True) + "\n",
    )
    experiment_descriptor = {
        "schema_version": "3.0.0",
        "kind": "v3_strict_experiment_descriptor",
        "status": "frozen_before_lockcheck",
        "selected_feature_groups": ["G0", "G10"],
        "excluded_feature_groups": [{"group": "G11", "reason": "unavailable"}],
        "architecture": {"name": "synthetic_fcer", "hidden_dim": 128},
        "partition_manifests": {
            name: artifact_identity(result["split_path"])
            for name in ("train", "calibration", "select", "lockcheck", "test")
        },
        "oof": {
            "group_key": "sequence_id",
            "fold_count": 3,
            "folds": [0, 1, 2],
            "manifest": artifact_identity(oof_manifest),
        },
        "seeds": list(ENSEMBLE_SEEDS),
        "feature_schema": artifact_identity(feature_schema),
        "normalizers": {
            name: artifact_identity(path)
            for name, path in result["normalizer_paths"].items()
        },
        "checkpoints": {
            name: artifact_identity(path)
            for name, path in result["checkpoint_paths"].items()
        },
        "alpha": 0.5,
        "gate_policy": GATE_POLICY,
        "perturbations": list(PERTURBATIONS),
        "inference_input_allowlist": {
            "lockcheck": ["candidate_features"],
            "test": ["candidate_features", "model_checkpoint"],
        },
        "formal_methods": list(METHODS),
        "callbacks": {
            "inference": INFERENCE_CALLBACK,
            "evaluator": EVALUATOR_CALLBACK,
        },
        "git_diff_sha256": "0" * 64,
        "prior_test_exposure_disclosure": PRIOR_TEST_EXPOSURE_DISCLOSURE,
        "artifact_bindings": {
            "config": artifact_identity(result["config_path"]),
            "contract": artifact_identity(result["contract_path"]),
            "split": artifact_identity(result["split_path"]),
            "candidate": artifact_identity(candidate),
            "gate": {
                name: artifact_identity(path) for name, path in result["gate_paths"].items()
            },
            "evaluators": {
                name: artifact_identity(path)
                for name, path in result["evaluator_paths"].items()
            },
            "v2": {
                name: artifact_identity(path) for name, path in result["v2_paths"].items()
            },
        },
        "evaluation_descriptor": artifact_identity(evaluation_path),
    }
    result["evaluation_descriptor_path"] = evaluation_path
    result["experiment_descriptor_path"] = write_file(
        artifacts / "experiment-descriptor.json",
        json.dumps(experiment_descriptor, sort_keys=True) + "\n",
    )
    result["inference_callback_identity"] = INFERENCE_CALLBACK
    result["evaluator_callback_identity"] = EVALUATOR_CALLBACK
    return result


def preliminary_lock(tmp_path: Path, **overrides: Any) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    artifacts = artifact_set(tmp_path)
    arguments = preliminary_arguments(tmp_path, artifacts)
    arguments.update(overrides)
    result = lock_preliminary(**arguments)
    return Path(arguments["output_path"]), result, artifacts


def preliminary_arguments(
    tmp_path: Path, artifacts: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "output_path": tmp_path / "preliminary.json",
        "run_id": "synthetic-v3",
        "primary_method": "v3_locked_primary",
        "formal_methods": METHODS,
        **{
            name: artifacts[name]
            for name in (
                "config_path", "checkpoint_paths", "normalizer_paths", "gate_paths",
                "contract_path", "split_path", "candidate_path", "evaluator_paths",
                "v2_paths", "experiment_descriptor_path", "evaluation_descriptor_path",
                "inference_callback_identity", "evaluator_callback_identity",
            )
        },
    }


def ranking_rows(method: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": record["sample_id"],
            "method": method,
            "candidate_order": list(record["candidate_ids"]),
        }
        for record in candidate_rows()
    ]


def inference_callback(**kwargs: Any) -> list[Path]:
    output = Path(kwargs["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    return [
        write_jsonl(output / f"{method}.jsonl", ranking_rows(method))
        for method in METHODS
        if method != "q_only"
    ]


def lockcheck_evaluator_callback(**kwargs: Any) -> list[Path]:
    output = Path(kwargs["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    results = write_file(output / "results_lockcheck.csv", "track,method\n")
    metric = {"sample_count": 2, "correct": 1, "oracle_correct": 2}
    summary = {
        "schema_version": "3.0.0",
        "kind": "v3_dual_track_evaluation",
        "status": "complete",
        "sample_count": 2,
        "methods": list(METHODS),
        "tracks": {
            track: {method: dict(metric) for method in METHODS}
            for track in ("corrected_scientific", "legacy_official_compatibility")
        },
        "oracle_unchanged": True,
    }
    summary_path = write_file(
        output / "summary.json", json.dumps(summary, sort_keys=True) + "\n"
    )
    return [results, summary_path]


def complete_lockcheck_evaluation(
    tmp_path: Path,
    *,
    preliminary: Path,
    artifacts: dict[str, Any],
    inference: dict[str, Any],
) -> dict[str, Any]:
    rankings = {
        method: None
        if method == "q_only"
        else tmp_path / f"lockcheck-output/{method}.jsonl"
        for method in METHODS
    }
    return evaluate_lockcheck_once(
        preliminary_manifest_path=preliminary,
        lockcheck_inference_completion_path=inference["completion_path"],
        stage_dir=tmp_path / "lockcheck-evaluation-stage",
        candidate_artifact=artifacts["candidate_path"],
        method_rankings=rankings,
        evaluation_artifacts={
            "corrected_labels": artifacts["lockcheck_corrected_labels"],
            "legacy_labels": artifacts["lockcheck_legacy_labels"],
        },
        output_dir=tmp_path / "lockcheck-evaluation-output",
        evaluator_callback=lockcheck_evaluator_callback,
        callback_identity=EVALUATOR_CALLBACK,
    )


def test_preliminary_lock_rejects_non_unique_or_unapproved_methods(tmp_path: Path) -> None:
    artifacts = artifact_set(tmp_path)
    base = {
        "output_path": tmp_path / "preliminary.json",
        "run_id": "synthetic-v3",
        "primary_method": "v3_locked_primary",
        **{
            name: artifacts[name]
            for name in (
                "config_path", "checkpoint_paths", "normalizer_paths", "gate_paths",
                "contract_path", "split_path", "candidate_path", "evaluator_paths",
                "v2_paths", "experiment_descriptor_path", "evaluation_descriptor_path",
                "inference_callback_identity", "evaluator_callback_identity",
            )
        },
    }
    with pytest.raises(ValueError, match="unique"):
        lock_preliminary(formal_methods=(*METHODS, "v3_locked_primary"), **base)
    with pytest.raises(ValueError, match="outside the allowlist"):
        lock_preliminary(formal_methods=(*METHODS, "test_tuned_method"), **base)
    with pytest.raises(ValueError, match="outside the allowlist"):
        lock_preliminary(
            formal_methods=(*METHODS, "v3_q_anchored_ablation"), **base
        )
    with pytest.raises(ValueError, match="disclosure"):
        lock_preliminary(
            formal_methods=METHODS,
            exposure_disclosure="V3 was unseen.",
            **base,
        )
    assert set(METHODS) <= set(FORMAL_METHOD_ALLOWLIST)


def test_lock_preliminary_captures_all_required_hash_groups(tmp_path: Path) -> None:
    path, locked, _ = preliminary_lock(tmp_path)
    verified = verify_locked_manifest(path, expected_kind=PRELIMINARY_MANIFEST_KIND)
    assert verified == locked
    assert set(verified["artifacts"]) == {
        "config",
        "checkpoints",
        "normalizers",
        "gate",
        "contract",
        "split",
        "candidate",
        "evaluators",
        "v2",
        "experiment_descriptor",
        "evaluation_descriptor",
        "evaluation_cohort_candidates",
        "descriptor_dependencies",
    }
    assert verified["formal_methods"] == list(METHODS)
    assert verified["prior_test_exposure_disclosure"] == PRIOR_TEST_EXPOSURE_DISCLOSURE
    assert len({value["path"] for value in verified["locked_artifacts"]}) == len(
        verified["locked_artifacts"]
    )


def test_scope_candidate_comes_from_evaluation_descriptor_and_flat_lock(
    tmp_path: Path,
) -> None:
    inputs = artifact_set(tmp_path)
    test_rows = [
        {
            "sample_id": row["sample_id"],
            "candidate_ids": [f"test-{row['sample_id']}-c{index}" for index in range(5)],
        }
        for row in candidate_rows()
    ]
    test_candidate = write_jsonl(
        tmp_path / "inputs/test-candidates.jsonl", test_rows
    )
    evaluation = json.loads(inputs["evaluation_descriptor_path"].read_text())
    evaluation["cohorts"]["test"]["candidate"] = artifact_identity(test_candidate)
    write_file(
        inputs["evaluation_descriptor_path"],
        json.dumps(evaluation, sort_keys=True) + "\n",
    )
    experiment = json.loads(inputs["experiment_descriptor_path"].read_text())
    experiment["evaluation_descriptor"] = artifact_identity(
        inputs["evaluation_descriptor_path"]
    )
    write_file(
        inputs["experiment_descriptor_path"],
        json.dumps(experiment, sort_keys=True) + "\n",
    )
    manifest_path = tmp_path / "preliminary.json"
    locked = lock_preliminary(**preliminary_arguments(tmp_path, inputs))

    assert verify_scope_candidate_identity(
        locked, scope="lockcheck", candidate_artifact=inputs["candidate_path"]
    ) == artifact_identity(inputs["candidate_path"])
    assert verify_scope_candidate_identity(
        locked, scope="test", candidate_artifact=test_candidate
    ) == artifact_identity(test_candidate)
    with pytest.raises(PermissionError, match="differs from the frozen"):
        verify_scope_candidate_identity(
            locked, scope="test", candidate_artifact=inputs["candidate_path"]
        )
    with pytest.raises(PermissionError, match="differs from the frozen"):
        verify_scope_candidate_identity(
            locked, scope="lockcheck", candidate_artifact=test_candidate
        )
    missing_flat_lock = dict(locked)
    missing_flat_lock["locked_artifacts"] = [
        identity for identity in locked["locked_artifacts"]
        if identity["path"] != str(test_candidate.resolve())
    ]
    with pytest.raises(PermissionError, match="flat locked_artifacts"):
        verify_scope_candidate_identity(missing_flat_lock, scope="test")

    lock_rankings = [
        write_jsonl(tmp_path / f"lock-{method}.jsonl", ranking_rows(method))
        for method in METHODS if method != "q_only"
    ]
    assert set(
        verify_formal_ranking_outputs(
            lock_rankings, locked_manifest=locked, scope="lockcheck"
        )
    ) == set(METHODS) - {"q_only"}
    with pytest.raises(ValueError, match="changed candidate identity"):
        verify_formal_ranking_outputs(
            lock_rankings, locked_manifest=locked, scope="test"
        )
    assert manifest_path.is_file()


@pytest.mark.parametrize(
    "mutation",
    ("missing_field", "extra_field", "checkpoint", "gate_policy", "callback"),
)
def test_preliminary_lock_strictly_cross_checks_experiment_descriptor(
    tmp_path: Path, mutation: str,
) -> None:
    inputs = artifact_set(tmp_path)
    descriptor_path = inputs["experiment_descriptor_path"]
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if mutation == "missing_field":
        descriptor.pop("architecture")
    elif mutation == "extra_field":
        descriptor["unreviewed_override"] = True
    elif mutation == "checkpoint":
        descriptor["checkpoints"]["seed_1"] = artifact_identity(
            inputs["normalizer_paths"]["development"]
        )
    elif mutation == "gate_policy":
        descriptor["gate_policy"]["threshold"] = 0.2
    else:
        descriptor["callbacks"]["inference"] = (
            "failure_analysis.reranking_v3.formal_backend:unlocked_inference"
        )
    write_file(
        descriptor_path, json.dumps(descriptor, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError):
        lock_preliminary(**preliminary_arguments(tmp_path, inputs))
    output = tmp_path / "preliminary.json"
    assert not output.exists()
    assert not output.with_suffix(".json.sha256").exists()


def test_preliminary_and_label_free_inference_do_not_open_descriptor_labels(
    tmp_path: Path,
) -> None:
    inputs = artifact_set(tmp_path)
    for name in (
        "lockcheck_corrected_labels",
        "lockcheck_legacy_labels",
        "test_corrected_labels",
        "test_legacy_labels",
    ):
        inputs[name].unlink()

    preliminary_path = tmp_path / "preliminary.json"
    lock_preliminary(**preliminary_arguments(tmp_path, inputs))
    run = run_locked_inference_once(
        scope="lockcheck",
        manifest_path=preliminary_path,
        stage_dir=tmp_path / "lockcheck-stage",
        input_artifacts={"candidate_features": inputs["candidate_path"]},
        output_dir=tmp_path / "lockcheck-output",
        inference_callback=inference_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    assert run["labels_read"] is False


def test_locked_inference_claims_before_rejecting_label_inputs(tmp_path: Path) -> None:
    manifest, _, artifacts = preliminary_lock(tmp_path)
    with pytest.raises(PermissionError, match="forbidden marker"):
        run_locked_inference_once(
            scope="lockcheck",
            manifest_path=manifest,
            stage_dir=tmp_path / "lockcheck-stage",
            input_artifacts={"test_labels": artifacts["candidate_path"]},
            output_dir=tmp_path / "unused",
            inference_callback=inference_callback,
            callback_identity=INFERENCE_CALLBACK,
        )
    assert (tmp_path / "lockcheck-stage/LOCKCHECK_RUN_CLAIM.json").is_file()
    assert not (tmp_path / "lockcheck-stage/LOCKCHECK_RUN_COMPLETE.json").exists()


def test_final_lock_requires_a_verified_lockcheck_completion(tmp_path: Path) -> None:
    manifest, _, artifacts = preliminary_lock(tmp_path)
    fake = write_file(tmp_path / "fake-complete.json", "{}\n")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        lock_final(
            output_path=tmp_path / "final.json",
            preliminary_manifest_path=manifest,
            lockcheck_completion_path=fake,
            lockcheck_evaluation_completion_path=fake,
            exact_test_command=("python", "-m", "formal-test"),
        )

    lockcheck = run_locked_inference_once(
        scope="lockcheck",
        manifest_path=manifest,
        stage_dir=tmp_path / "lockcheck-stage",
        input_artifacts={"candidate_features": artifacts["candidate_path"]},
        output_dir=tmp_path / "lockcheck-output",
        inference_callback=inference_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    with pytest.raises(ValueError, match="lockcheck_evaluate completion kind mismatch"):
        lock_final(
            output_path=tmp_path / "inference-only-final.json",
            preliminary_manifest_path=manifest,
            lockcheck_completion_path=lockcheck["completion_path"],
            lockcheck_evaluation_completion_path=lockcheck["completion_path"],
            exact_test_command=("python", "-m", "formal-test"),
        )
    assert not (tmp_path / "inference-only-final.json").exists()

    lockcheck_evaluation = complete_lockcheck_evaluation(
        tmp_path, preliminary=manifest, artifacts=artifacts, inference=lockcheck
    )
    final_path = tmp_path / "final.json"
    final = lock_final(
        output_path=final_path,
        preliminary_manifest_path=manifest,
        lockcheck_completion_path=lockcheck["completion_path"],
        lockcheck_evaluation_completion_path=lockcheck_evaluation["completion_path"],
        exact_test_command=("python", "-m", "formal-test"),
        test_input_paths={
            "candidate_features": artifacts["candidate_path"],
            "model_checkpoint": artifacts["checkpoint_paths"]["seed_1"],
        },
    )
    assert verify_locked_manifest(final_path, expected_kind=FINAL_MANIFEST_KIND) == final
    assert final["formal_methods"] == list(METHODS)


@pytest.mark.parametrize(
    ("invalid_summary", "message"),
    (("cohort", "summary contract differs"), ("oracle", "Oracle@5 identity differs")),
)
def test_lockcheck_evaluation_rejects_cohort_or_oracle_drift_after_claim(
    tmp_path: Path, invalid_summary: str, message: str,
) -> None:
    preliminary, _, inputs = preliminary_lock(tmp_path)
    inference = run_locked_inference_once(
        scope="lockcheck",
        manifest_path=preliminary,
        stage_dir=tmp_path / "lockcheck-stage",
        input_artifacts={"candidate_features": inputs["candidate_path"]},
        output_dir=tmp_path / "lockcheck-output",
        inference_callback=inference_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    rankings = {
        method: None
        if method == "q_only"
        else tmp_path / f"lockcheck-output/{method}.jsonl"
        for method in METHODS
    }

    def invalid_evaluator(**kwargs: Any) -> list[Path]:
        outputs = lockcheck_evaluator_callback(**kwargs)
        summary_path = next(path for path in outputs if path.name == "summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if invalid_summary == "cohort":
            summary["sample_count"] = 1
        else:
            summary["tracks"]["corrected_scientific"][METHODS[-1]][
                "oracle_correct"
            ] = 1
        write_file(summary_path, json.dumps(summary, sort_keys=True) + "\n")
        return outputs

    stage = tmp_path / "lockcheck-evaluation-stage"
    with pytest.raises(ValueError, match=message):
        evaluate_lockcheck_once(
            preliminary_manifest_path=preliminary,
            lockcheck_inference_completion_path=inference["completion_path"],
            stage_dir=stage,
            candidate_artifact=inputs["candidate_path"],
            method_rankings=rankings,
            evaluation_artifacts={
                "corrected_labels": inputs["lockcheck_corrected_labels"],
                "legacy_labels": inputs["lockcheck_legacy_labels"],
            },
            output_dir=tmp_path / "lockcheck-evaluation-output",
            evaluator_callback=invalid_evaluator,
            callback_identity=EVALUATOR_CALLBACK,
        )
    assert (stage / "LOCKCHECK_EVALUATE_RUN_CLAIM.json").is_file()
    assert not (stage / "LOCKCHECK_EVALUATE_RUN_COMPLETE.json").exists()


def test_lockcheck_evaluation_verifies_descriptor_label_identities_after_claim(
    tmp_path: Path,
) -> None:
    preliminary, _, inputs = preliminary_lock(tmp_path)
    inference = run_locked_inference_once(
        scope="lockcheck",
        manifest_path=preliminary,
        stage_dir=tmp_path / "lockcheck-stage",
        input_artifacts={"candidate_features": inputs["candidate_path"]},
        output_dir=tmp_path / "lockcheck-output",
        inference_callback=inference_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    rankings = {
        method: None
        if method == "q_only"
        else tmp_path / f"lockcheck-output/{method}.jsonl"
        for method in METHODS
    }
    replacement = write_file(tmp_path / "replacement-labels.jsonl", "different\n")
    stage = tmp_path / "lockcheck-evaluation-stage"
    called = False

    def should_not_run(**_: Any) -> list[Path]:
        nonlocal called
        called = True
        return []

    with pytest.raises(PermissionError, match="corrected_labels differs"):
        evaluate_lockcheck_once(
            preliminary_manifest_path=preliminary,
            lockcheck_inference_completion_path=inference["completion_path"],
            stage_dir=stage,
            candidate_artifact=inputs["candidate_path"],
            method_rankings=rankings,
            evaluation_artifacts={
                "corrected_labels": replacement,
                "legacy_labels": inputs["lockcheck_legacy_labels"],
            },
            output_dir=tmp_path / "lockcheck-evaluation-output",
            evaluator_callback=should_not_run,
            callback_identity=EVALUATOR_CALLBACK,
        )
    assert called is False
    assert (stage / "LOCKCHECK_EVALUATE_RUN_CLAIM.json").is_file()


def test_synthetic_formal_lifecycle_is_once_only_and_identity_safe(tmp_path: Path) -> None:
    preliminary, _, artifacts = preliminary_lock(tmp_path)
    lockcheck = run_locked_inference_once(
        scope="lockcheck",
        manifest_path=preliminary,
        stage_dir=tmp_path / "lockcheck-stage",
        input_artifacts={"candidate_features": artifacts["candidate_path"]},
        output_dir=tmp_path / "lockcheck-output",
        inference_callback=inference_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    lockcheck_evaluation = complete_lockcheck_evaluation(
        tmp_path, preliminary=preliminary, artifacts=artifacts, inference=lockcheck
    )
    final = tmp_path / "final.json"
    lock_final(
        output_path=final,
        preliminary_manifest_path=preliminary,
        lockcheck_completion_path=lockcheck["completion_path"],
        lockcheck_evaluation_completion_path=lockcheck_evaluation["completion_path"],
        exact_test_command=("python", "-m", "formal-test"),
        test_input_paths={
            "candidate_features": artifacts["candidate_path"],
            "model_checkpoint": artifacts["checkpoint_paths"]["seed_1"],
        },
    )
    formal = run_locked_inference_once(
        scope="test",
        manifest_path=final,
        stage_dir=tmp_path / "test-stage",
        input_artifacts={
            "candidate_features": artifacts["candidate_path"],
            "model_checkpoint": artifacts["checkpoint_paths"]["seed_1"],
        },
        output_dir=tmp_path / "test-output",
        inference_callback=inference_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    with pytest.raises(FileExistsError, match="already completed"):
        run_locked_inference_once(
            scope="test",
            manifest_path=final,
            stage_dir=tmp_path / "test-stage",
            input_artifacts={
                "candidate_features": artifacts["candidate_path"],
                "model_checkpoint": artifacts["checkpoint_paths"]["seed_1"],
            },
            output_dir=tmp_path / "second-test-output",
            inference_callback=inference_callback,
            callback_identity=INFERENCE_CALLBACK,
        )

    rankings = {
        method: None if method == "q_only" else tmp_path / f"test-output/{method}.jsonl"
        for method in METHODS
    }
    callback_calls: list[str] = []

    def independent_callback(**kwargs: Any) -> list[Path]:
        callback_calls.append("called")
        assert set(kwargs["method_rankings"]) == set(METHODS)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        return [write_file(output / "independent.json", '{"status":"passed"}\n')]

    independent = independent_evaluate_once(
        final_manifest_path=final,
        test_completion_path=formal["completion_path"],
        stage_dir=tmp_path / "independent-stage",
        candidate_artifact=artifacts["candidate_path"],
        method_rankings=rankings,
        evaluation_artifacts={
            "corrected_labels": artifacts["test_corrected_labels"],
            "legacy_labels": artifacts["test_legacy_labels"],
        },
        output_dir=tmp_path / "independent-output",
        evaluator_callback=independent_callback,
        callback_identity=EVALUATOR_CALLBACK,
    )
    assert callback_calls == ["called"]
    assert independent["candidate_identity_verified"] is True
    assert independent["method_identity_verified"] is True

    replacement_labels = write_file(
        tmp_path / "replacement-test-labels.jsonl", "different\n"
    )
    callback_calls.clear()
    with pytest.raises(PermissionError, match="test corrected_labels differs"):
        independent_evaluate_once(
            final_manifest_path=final,
            test_completion_path=formal["completion_path"],
            stage_dir=tmp_path / "independent-label-drift-stage",
            candidate_artifact=artifacts["candidate_path"],
            method_rankings=rankings,
            evaluation_artifacts={
                "corrected_labels": replacement_labels,
                "legacy_labels": artifacts["test_legacy_labels"],
            },
            output_dir=tmp_path / "independent-label-drift-output",
            evaluator_callback=independent_callback,
            callback_identity=EVALUATOR_CALLBACK,
        )
    assert callback_calls == []
    assert (
        tmp_path
        / "independent-label-drift-stage/INDEPENDENT_EVALUATE_RUN_CLAIM.json"
    ).is_file()


def test_independent_evaluation_rejects_candidate_or_method_identity_drift(
    tmp_path: Path,
) -> None:
    preliminary, _, artifacts = preliminary_lock(tmp_path)
    lockcheck = run_locked_inference_once(
        scope="lockcheck",
        manifest_path=preliminary,
        stage_dir=tmp_path / "lockcheck-stage",
        input_artifacts={"candidate_features": artifacts["candidate_path"]},
        output_dir=tmp_path / "lockcheck-output",
        inference_callback=inference_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    lockcheck_evaluation = complete_lockcheck_evaluation(
        tmp_path, preliminary=preliminary, artifacts=artifacts, inference=lockcheck
    )
    final = tmp_path / "final.json"
    lock_final(
        output_path=final,
        preliminary_manifest_path=preliminary,
        lockcheck_completion_path=lockcheck["completion_path"],
        lockcheck_evaluation_completion_path=lockcheck_evaluation["completion_path"],
        exact_test_command=("python", "formal-test"),
        test_input_paths={
            "candidate_features": artifacts["candidate_path"],
            "model_checkpoint": artifacts["checkpoint_paths"]["seed_1"],
        },
    )
    def wrong_method_callback(**kwargs: Any) -> list[Path]:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        results = []
        for method in METHODS:
            if method == "q_only":
                continue
            rows = ranking_rows(method)
            if method == "v3_fcer_native":
                rows[0]["method"] = "v3_fcer_rgbd"
            results.append(write_jsonl(output / f"{method}.jsonl", rows))
        return results

    formal = run_locked_inference_once(
        scope="test",
        manifest_path=final,
        stage_dir=tmp_path / "test-stage",
        input_artifacts={
            "candidate_features": artifacts["candidate_path"],
            "model_checkpoint": artifacts["checkpoint_paths"]["seed_1"],
        },
        output_dir=tmp_path / "test-output",
        inference_callback=wrong_method_callback,
        callback_identity=INFERENCE_CALLBACK,
    )
    rankings = {
        method: None if method == "q_only" else tmp_path / f"test-output/{method}.jsonl"
        for method in METHODS
    }
    untrusted_candidates = write_jsonl(
        tmp_path / "untrusted-candidates.jsonl", candidate_rows()
    )
    with pytest.raises(PermissionError, match="not locked"):
        independent_evaluate_once(
            final_manifest_path=final,
            test_completion_path=formal["completion_path"],
            stage_dir=tmp_path / "independent-stage",
            candidate_artifact=untrusted_candidates,
            method_rankings=rankings,
            evaluation_artifacts={
                "corrected_labels": artifacts["test_corrected_labels"],
                "legacy_labels": artifacts["test_legacy_labels"],
            },
            output_dir=tmp_path / "independent-output",
            evaluator_callback=lambda **_: (),
            callback_identity=EVALUATOR_CALLBACK,
        )
    with pytest.raises(ValueError, match="method identity mismatch"):
        independent_evaluate_once(
            final_manifest_path=final,
            test_completion_path=formal["completion_path"],
            stage_dir=tmp_path / "independent-stage",
            candidate_artifact=artifacts["candidate_path"],
            method_rankings=rankings,
            evaluation_artifacts={
                "corrected_labels": artifacts["test_corrected_labels"],
                "legacy_labels": artifacts["test_legacy_labels"],
            },
            output_dir=tmp_path / "independent-output",
            evaluator_callback=lambda **_: (),
            callback_identity=EVALUATOR_CALLBACK,
        )
