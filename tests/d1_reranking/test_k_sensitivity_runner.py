from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from d1_reranking.execution import artifact_record
import d1_reranking.k_execution as k_execution
from d1_reranking.k_sensitivity import write_k_sensitivity_plan
from unified_reranking.hashing import atomic_json, canonical_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "tools/d1_reranking/run_k_sensitivity_cell.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "d1_k_sensitivity_test_runner", RUNNER_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def _load_tool(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(
        name, REPOSITORY_ROOT / "tools/d1_reranking" / filename
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


authorizer = _load_tool(
    "d1_k_sensitivity_test_authorizer", "authorize_k_sensitivity_execution.py"
)
orchestrator = _load_tool(
    "d1_k_sensitivity_test_orchestrator", "run_k_sensitivity_matrix.py"
)


class _SyntheticLambdaRank:
    def __init__(self, *, seed: int, **parameters: object) -> None:
        self.seed = seed
        self.parameters = parameters

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        query_ids: list[str],
        *,
        eval_set: tuple[np.ndarray, np.ndarray, list[str]],
    ) -> "_SyntheticLambdaRank":
        assert len(features) == len(labels) == len(query_ids)
        assert len(eval_set[0]) == len(eval_set[1]) == len(eval_set[2])
        self.width = int(features.shape[1])
        self.model = self
        self.booster_ = self
        return self

    def save_model(self, path: str) -> None:
        Path(path).write_text("synthetic native LightGBM model\n", encoding="utf-8")

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[1] == self.width
        return features[:, 0].astype(float)

    def artifact(self) -> dict[str, object]:
        return {
            "kind": "synthetic_lambdarank",
            "seed": self.seed,
            "parameters": self.parameters,
        }


def _manifest(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return path


def _plain(path: Path, value: str = "synthetic\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_bound_manifest_does_not_treat_summary_sha_as_artifact(tmp_path: Path) -> None:
    artifact = _plain(tmp_path / "labels.parquet")
    manifest = _manifest(
        tmp_path / "manifest.json",
        {
            "status": "COMPLETE",
            "artifact": artifact_record(artifact),
            "sources": {},
            "summary": {
                "destination": str(artifact),
                "sha256": artifact_record(artifact)["sha256"],
                "candidate_rows": 2,
            },
        },
    )
    path, payload = runner._bound_manifest(
        artifact_record(manifest), name="synthetic development labels"
    )
    assert path == manifest.resolve()
    assert payload["summary"]["candidate_rows"] == 2


def test_k_runner_compares_normalized_artifact_identity() -> None:
    expected = {"path": "/tmp/candidates.parquet", "sha256": "a" * 64}
    runner._same_artifact_record(
        {**expected, "bytes": 123, "rows": 10},
        expected,
        name="synthetic candidates",
    )
    with pytest.raises(RuntimeError, match="does not bind"):
        runner._same_artifact_record(
            {**expected, "sha256": "b" * 64},
            expected,
            name="synthetic candidates",
        )


def _candidates(sample_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        for native_rank, native_score in ((1, 0.8), (2, 0.2)):
            candidate_id = f"{sample_id}-c{native_rank}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "native_rank": native_rank,
                    "native_score": native_score,
                    "candidate_identity_sha256": f"identity-{candidate_id}",
                    "candidate_geometry_sha256": f"geometry-{candidate_id}",
                }
            )
    return pd.DataFrame(rows)


def _write_development_sources(root: Path) -> None:
    split_samples = {
        "train": [f"train-{index}" for index in range(5)],
        "validation": ["validation-0", "validation-1"],
    }
    (root / "04_splits").mkdir(parents=True, exist_ok=True)
    (root / "01_manifests").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": split_samples["train"],
            "fold": list(range(5)),
        }
    ).to_parquet(root / "04_splits" / "fold_assignments.parquet", index=False)
    for split, sample_ids in split_samples.items():
        pd.DataFrame({"sample_id": sample_ids}).to_parquet(
            root / "01_manifests" / f"d1_paired_{split}.parquet", index=False
        )
        candidate_manifest_path = root / "02_candidates" / split / "manifest.json"
        candidate_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = _candidates(sample_ids)
        candidate_artifacts: dict[str, dict[str, str]] = {}
        for pool in ("top10", "allnms"):
            candidate_path = (
                candidate_manifest_path.parent / f"d1_{pool}_candidates.parquet"
            )
            candidates.to_parquet(candidate_path, index=False)
            candidate_artifacts[pool] = artifact_record(candidate_path)
        _manifest(
            candidate_manifest_path,
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "configuration": {"route": "D1", "split": split},
                "summaries": {
                    "top10": {"maximum_candidates": 2},
                    "allnms": {"maximum_candidates": 2},
                },
                "artifacts": candidate_artifacts,
            },
        )
        candidate_manifest_record = artifact_record(candidate_manifest_path)
        for pool, track in (
            ("top10", "T2_matched_common"),
            ("allnms", "T2_matched_common"),
            ("allnms", "T3_route_rich"),
        ):
            candidate_path = (
                candidate_manifest_path.parent / f"d1_{pool}_candidates.parquet"
            )
            candidate_record = artifact_record(candidate_path)
            feature_dir = root / "03_features" / split / pool / track
            feature_path = feature_dir / "candidate_features.parquet"
            feature_rows = candidates[
                ["sample_id", "candidate_id", "native_rank"]
            ].copy()
            feature_rows["native_score_raw"] = candidates["native_score"]
            feature_rows["calibrated_native_probability"] = candidates["native_score"]
            feature_rows["base_logit"] = np.log(
                feature_rows["native_score_raw"]
                / (1.0 - feature_rows["native_score_raw"])
            )
            feature_rows["synthetic_evidence"] = (
                feature_rows["native_rank"].astype(float) / 10.0
            )
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_rows.to_parquet(feature_path, index=False)
            feature_columns = [
                "native_score_raw",
                "calibrated_native_probability",
                "base_logit",
                "synthetic_evidence",
            ]
            _manifest(
                feature_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "status": "COMPLETE",
                    "candidate_test_labels_read": False,
                    "configuration": {
                        "route": "D1",
                        "split": split,
                        "pool": pool,
                        "track": track,
                    },
                    "model_feature_columns": feature_columns,
                    "model_feature_schema_sha256": canonical_sha256(feature_columns),
                    "feature_extraction_latency_ms": 0.01,
                    "sources": {
                        "candidate_manifest": candidate_manifest_record,
                        "candidates": candidate_record,
                    },
                    "artifacts": {"candidate_features": artifact_record(feature_path)},
                },
            )
        for pool in ("top10", "allnms"):
            candidate_path = (
                candidate_manifest_path.parent / f"d1_{pool}_candidates.parquet"
            )
            label_dir = root / "03_features" / split / pool / "labels"
            label_path = label_dir / "candidate_labels.parquet"
            labels = candidates[["sample_id", "candidate_id", "native_rank"]].copy()
            labels["candidate_success"] = labels["native_rank"].eq(1).astype(int)
            labels["jacquard_margin"] = labels["candidate_success"].astype(float)
            label_dir.mkdir(parents=True, exist_ok=True)
            labels.to_parquet(label_path, index=False)
            _manifest(
                label_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "status": "COMPLETE",
                    "split": split,
                    "pool": pool,
                    "candidate_test_labels_read": False,
                    "sources": {
                        "candidate_manifest": candidate_manifest_record,
                        "candidates": artifact_record(candidate_path),
                    },
                    "artifact": artifact_record(label_path),
                },
            )


def _synthetic_run(
    tmp_path: Path, *, with_execution: bool = True
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    _write_development_sources(root)
    primary_plan_path = _manifest(
        root / "configs" / "d1_primary_matrix_plan.json",
        {
            "schema_version": 1,
            "status": "PLANNED",
            "route": "D1",
            "primary_contract": {"pool": "top5", "track": "T2_matched_common"},
            "formal_seeds": [42, 123, 2026],
            "outer_folds": 5,
        },
    )
    selected_configuration: dict[str, object] = {
        "schema_version": 1,
        "route": "D1",
        "pool": "top5",
        "track": "T2_matched_common",
        "method": "R5",
        "encoder": "lambdamart",
        "loss": "lambdarank",
        "num_leaves": 7,
        "learning_rate": 0.05,
        "n_estimators": 3,
    }
    trial_path = _manifest(
        root
        / "07_validation"
        / "primary_selection"
        / "trials"
        / "r5"
        / "manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "trial_id": "r5",
            "configuration": selected_configuration,
            "candidate_test_labels_read": False,
        },
    )
    _manifest(
        root / "07_validation" / "selected_primary_ungated.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "selected_method": "R5",
            "selected_trial_id": "r5",
            "selected_configuration": selected_configuration,
            "candidate_test_labels_read": False,
            "sources": {"plan": artifact_record(primary_plan_path)},
            "artifacts": {"selected_trial_manifest": artifact_record(trial_path)},
        },
    )
    for pool in ("top10", "allnms"):
        marker = _plain(root / "05_calibration" / pool / "calibration.bin")
        _manifest(
            root / "05_calibration" / pool / "calibration_manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "configuration": {"route": "D1", "pool": pool},
                "selected_method": "platt",
                "candidate_test_labels_read": False,
                "artifacts": {"marker": artifact_record(marker)},
            },
        )
    plan_path = root / "configs" / "d1_k_sensitivity_plan.json"
    plan = write_k_sensitivity_plan(
        plan_path,
        run_dir=root,
        tool_paths=(
            runner.ROOT / "src/d1_reranking/k_sensitivity.py",
            runner.ROOT / "tools/d1_reranking/plan_k_sensitivity.py",
            Path(runner.__file__),
        ),
        resume=False,
    )
    job = next(
        job
        for job in plan["jobs"]
        if job["configuration"]["scenario_id"] == "top10_t2"
        and job["configuration"]["seed"] == 42
        and job["configuration"]["mode"] == "validation"
    )
    gate_path = _manifest(
        root / "00_audit" / "resource_gates" / "synthetic_gate.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "candidate_test_labels_read": False,
        },
    )
    gate_path_2 = _manifest(
        root / "00_audit" / "resource_gates" / "synthetic_gate_resume.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "candidate_test_labels_read": False,
        },
    )
    job["_synthetic_gate_path"] = gate_path
    job["_synthetic_gate_resume_path"] = gate_path_2
    if not with_execution:
        return root, job
    execution_id = "a" * 24
    execution_sources = k_execution.execution_source_records(
        plan_path=plan_path, resource_gate_path=gate_path
    )
    now = datetime.now(timezone.utc)
    execution = {
        "schema_version": 1,
        "status": "ACTIVE",
        "execution_id": execution_id,
        "authorized_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(minutes=5)).isoformat(),
        "owner": "synthetic",
        "run_dir": str(root.resolve()),
        "rank1_run_dir": str(k_execution.CANONICAL_RANK1_RUN_DIR),
        "host": {"synthetic": True},
        "scope": k_execution.K_EXECUTION_SCOPE,
        "plan": artifact_record(plan_path),
        "resource_gate": artifact_record(gate_path),
        "job_count": 54,
        "job_ids_sha256": plan["job_ids_sha256"],
        "direct_cli_execution_permitted": False,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "sources": execution_sources,
        "source_signature_sha256": canonical_sha256(execution_sources),
    }
    execution["content_sha256"] = canonical_sha256(execution)
    execution_path = (
        k_execution.execution_directory(root, execution_id) / "execution.json"
    )
    _manifest(
        execution_path,
        {key: value for key, value in execution.items() if key != "content_sha256"},
    )
    command = ["python", "-m", "tools.d1_reranking.run_k_sensitivity_cell"]
    claim_path, _claim = k_execution.create_job_claim(
        root,
        execution=execution,
        job=job,
        owner_pid=os.getpid(),
        command=command,
    )
    claim_record = artifact_record(claim_path)
    k_execution.write_execution_event(
        root,
        execution=execution,
        sequence=0,
        status="ACTIVE",
        owner_pid=os.getpid(),
        current_job_id=str(job["job_id"]),
        claim=claim_record,
        outputs={},
        commands=({"job_id": job["job_id"], "argv": command},),
    )
    job["_synthetic_execution_id"] = execution_id
    job["_synthetic_claim_sha256"] = claim_record["sha256"]
    return root, job


def test_k_sensitivity_cell_runs_and_exact_resume_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, job = _synthetic_run(tmp_path)
    monkeypatch.setenv(
        runner.EXECUTION_ENVIRONMENT_VARIABLE, job["_synthetic_execution_id"]
    )
    monkeypatch.setenv(k_execution.K_CLAIM_SHA256_ENV, job["_synthetic_claim_sha256"])
    monkeypatch.setattr(
        k_execution, "validate_resource_gate", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(k_execution, "host_contract", lambda: {"synthetic": True})
    monkeypatch.setattr(runner, "LightGBMLambdaRank", _SyntheticLambdaRank)
    monkeypatch.setattr(runner, "lightgbm_parameter_count", lambda model: 3)
    args = argparse.Namespace(
        run_dir=root,
        job_id=job["job_id"],
        execution_manifest=None,
        resume=False,
    )
    lease_path = root.parent / ".d1_heavy_resource.lock"
    result = runner.run(args, resource_lease_path=lease_path)

    assert result["status"] == "COMPLETE"
    assert result["job_id"] == job["job_id"]
    assert result["configuration"]["max_candidates"] == 10
    assert result["candidate_test_labels_read"] is False
    assert result["prediction_contract"]["rows"] == 4
    assert result["decision_contract"]["rows"] == 2
    assert result["model_contract"]["selected_primary_trial_id"] == "r5"

    args.resume = True
    assert runner.run(args, resource_lease_path=lease_path) == result

    predictions_path = Path(result["artifacts"]["predictions"]["path"])
    predictions = pd.read_parquet(predictions_path)
    predictions.loc[0, "score"] = float(predictions.loc[0, "score"]) + 1.0
    predictions.to_parquet(predictions_path, index=False)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runner.run(args, resource_lease_path=lease_path)


def test_k_sensitivity_execution_rejects_missing_identity_before_data_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, job = _synthetic_run(tmp_path)
    monkeypatch.delenv(runner.EXECUTION_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setenv(k_execution.K_CLAIM_SHA256_ENV, job["_synthetic_claim_sha256"])
    monkeypatch.setattr(
        k_execution, "validate_resource_gate", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(k_execution, "host_contract", lambda: {"synthetic": True})
    args = argparse.Namespace(
        run_dir=root,
        job_id=job["job_id"],
        execution_manifest=None,
        resume=False,
    )
    with pytest.raises(RuntimeError, match="identity differs"):
        runner.run(args, resource_lease_path=root.parent / ".d1_heavy_resource.lock")


def test_k_authorization_requires_fresh_gate_and_failed_resume_gets_new_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, job = _synthetic_run(tmp_path, with_execution=False)
    gate_path = Path(job["_synthetic_gate_path"])
    gate_path_2 = Path(job["_synthetic_gate_resume_path"])
    gate_calls: list[bool] = []

    def fake_gate(*args: object, **kwargs: object) -> dict[str, object]:
        gate_calls.append(bool(kwargs["require_fresh"]))
        return {"gate_id": "synthetic-gate"}

    clean_snapshot = {
        "rank1_workers": [],
        "rank1_claim_paths": [],
        "d1_heavy_workers": [],
        "foreign_heavy_processes": [],
    }
    monkeypatch.setattr(authorizer, "validate_resource_gate", fake_gate)
    monkeypatch.setattr(
        authorizer, "collect_resource_snapshot", lambda **kwargs: clean_snapshot
    )
    monkeypatch.setattr(authorizer, "evaluate_resource_snapshot", lambda *a, **k: [])
    monkeypatch.setattr(authorizer, "host_contract", lambda: {"synthetic": True})
    first = authorizer.run(
        root,
        gate_manifest=gate_path,
        owner="synthetic-owner",
        rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
    )
    pointer = k_execution.load_content_manifest(
        root / k_execution.K_EXECUTION_POINTER_RELATIVE,
        name="synthetic K pointer",
        statuses=("ACTIVE",),
    )
    assert pointer["execution_id"] == first["execution_id"]
    assert gate_calls == [True]

    k_execution.write_execution_event(
        root,
        execution=first,
        sequence=1,
        status="FAILED",
        owner_pid=os.getpid(),
        current_job_id=None,
        claim=None,
        outputs={},
        commands=(),
        failure="synthetic failure",
    )
    with pytest.raises(RuntimeError, match="newly completed fresh resource gate"):
        authorizer.run(
            root,
            gate_manifest=gate_path,
            owner="synthetic-owner",
            rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
        )
    second = authorizer.run(
        root,
        gate_manifest=gate_path_2,
        owner="synthetic-owner",
        rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
    )
    assert second["execution_id"] != first["execution_id"]
    assert gate_calls == [True, True]


def test_k_failure_fresh_gate_resume_runs_only_remaining_and_completes_exactly_54(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, metadata = _synthetic_run(tmp_path, with_execution=False)
    gate_path = Path(metadata["_synthetic_gate_path"])
    resume_gate_path = Path(metadata["_synthetic_gate_resume_path"])
    clean_snapshot = {
        "rank1_workers": [],
        "rank1_claim_paths": [],
        "d1_heavy_workers": [],
        "foreign_heavy_processes": [],
    }
    monkeypatch.setattr(
        authorizer,
        "validate_resource_gate",
        lambda *args, **kwargs: {"gate_id": Path(args[0]).stem},
    )
    monkeypatch.setattr(
        authorizer, "collect_resource_snapshot", lambda **kwargs: clean_snapshot
    )
    monkeypatch.setattr(authorizer, "evaluate_resource_snapshot", lambda *a, **k: [])
    monkeypatch.setattr(authorizer, "host_contract", lambda: {"synthetic": True})
    monkeypatch.setattr(
        authorizer, "validate_k_sensitivity_result", lambda *args, **kwargs: None
    )
    first = authorizer.run(
        root,
        gate_manifest=gate_path,
        owner="synthetic-owner",
        rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
    )
    plan = runner.load_k_sensitivity_plan(root / "configs/d1_k_sensitivity_plan.json")
    completed_before_failure = 7
    inherited: dict[str, dict[str, str]] = {}
    for job in plan["jobs"][:completed_before_failure]:
        result_path = _manifest(
            root / str(job["output_manifest"]),
            {"schema_version": 1, "status": "COMPLETE"},
        )
        inherited[str(job["job_id"])] = artifact_record(result_path)
    k_execution.write_execution_event(
        root,
        execution=first,
        sequence=1,
        status="FAILED",
        owner_pid=os.getpid(),
        current_job_id=None,
        claim=None,
        outputs=inherited,
        commands=(),
        failure="synthetic job 7 failure",
    )
    second = authorizer.run(
        root,
        gate_manifest=resume_gate_path,
        owner="synthetic-owner",
        rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
    )
    assert second["resume_outputs"] == inherited

    process_calls: list[str] = []
    executable = _plain(tmp_path / "synthetic-python")
    monkeypatch.setattr(orchestrator, "assert_writable_prelock", lambda root: None)
    monkeypatch.setattr(orchestrator, "_live_recheck", lambda *args, **kwargs: {})
    monkeypatch.setattr(k_execution, "validate_resource_gate", lambda *a, **k: {})
    monkeypatch.setattr(k_execution, "host_contract", lambda: {"synthetic": True})
    monkeypatch.setattr(
        orchestrator, "validate_k_sensitivity_result", lambda *args, **kwargs: None
    )

    def fake_process(command: list[str], **kwargs: object) -> SimpleNamespace:
        process_calls.append(command[command.index("--job-id") + 1])
        return SimpleNamespace(returncode=0, stdout="synthetic", stderr="")

    def fake_child_result(
        stdout: str, *, root: Path, job: dict[str, object], **kwargs: object
    ) -> dict[str, str]:
        path = _manifest(
            root / str(job["output_manifest"]),
            {"schema_version": 1, "status": "COMPLETE"},
        )
        return artifact_record(path)

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_process)
    monkeypatch.setattr(orchestrator, "_child_result", fake_child_result)
    complete = orchestrator._run_under_lease(
        root,
        python=executable,
        rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
    )
    expected_remaining = [
        str(job["job_id"]) for job in plan["jobs"][completed_before_failure:]
    ]
    assert process_calls == expected_remaining
    assert complete["status"] == "COMPLETE"
    assert complete["completed_jobs"] == 54
    assert set(complete["outputs"]) == {str(job["job_id"]) for job in plan["jobs"]}
    assert [command["job_id"] for command in complete["commands"]] == expected_remaining


def _orchestration_plan() -> dict[str, object]:
    jobs = []
    for index in range(54):
        configuration = {"schema_version": 1, "index": index}
        job_id = canonical_sha256(configuration)[:16]
        jobs.append(
            {
                "job_id": job_id,
                "configuration": configuration,
                "worker_argv": [
                    "-m",
                    "synthetic.worker",
                    "--job-id",
                    job_id,
                    "--resume",
                ],
                "output_manifest": f"11_k_sensitivity/cells/{job_id}/manifest.json",
            }
        )
    return {
        "jobs": jobs,
        "job_count": 54,
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
    }


@pytest.mark.parametrize("failure_index", [None, 2])
def test_k_orchestrator_is_strictly_serial_and_transitions_complete_or_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int | None,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    executable = _plain(tmp_path / "python")
    plan = _orchestration_plan()
    execution = {"execution_id": "b" * 24}
    execution["resume_outputs"] = {}
    initial_event = {
        "sequence": 0,
        "outputs": {},
        "current_job_id": None,
        "claim": None,
        "commands": [],
        "completed_jobs": 0,
    }
    events: list[dict[str, object]] = []
    active_children = 0
    maximum_active = 0
    process_calls = 0

    monkeypatch.setattr(orchestrator, "assert_writable_prelock", lambda root: None)
    monkeypatch.setattr(orchestrator, "load_k_sensitivity_plan", lambda path: plan)
    monkeypatch.setattr(
        orchestrator,
        "validate_active_execution",
        lambda *args, **kwargs: (
            root / "execution.json",
            execution,
            root / "event.json",
            initial_event,
        ),
    )
    monkeypatch.setattr(orchestrator, "_live_recheck", lambda *args, **kwargs: {})

    def fake_claim(
        root: Path, *, execution: object, job: dict[str, object], **kwargs: object
    ) -> tuple[Path, dict[str, object]]:
        path = _plain(
            root / "synthetic_claims" / f"{job['job_id']}.json",
            str(job["job_id"]),
        )
        return path, {"status": "CLAIMED"}

    def fake_event(root: Path, **kwargs: object) -> tuple[Path, dict[str, object]]:
        events.append(dict(kwargs))
        return root / f"event-{len(events)}.json", dict(kwargs)

    def fake_process(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal active_children, maximum_active, process_calls
        active_children += 1
        maximum_active = max(maximum_active, active_children)
        index = process_calls
        process_calls += 1
        active_children -= 1
        return SimpleNamespace(
            returncode=7 if failure_index == index else 0,
            stdout="synthetic",
            stderr="synthetic failure",
        )

    def fake_child(
        stdout: str, *, root: Path, job: dict[str, object], **kwargs: object
    ) -> dict[str, str]:
        path = _plain(root / str(job["output_manifest"]), str(job["job_id"]))
        return artifact_record(path)

    monkeypatch.setattr(orchestrator, "create_job_claim", fake_claim)
    monkeypatch.setattr(orchestrator, "write_execution_event", fake_event)
    monkeypatch.setattr(orchestrator.subprocess, "run", fake_process)
    monkeypatch.setattr(orchestrator, "_child_result", fake_child)
    monkeypatch.setattr(orchestrator, "_validate_bijection", lambda **kwargs: None)

    if failure_index is None:
        result = orchestrator._run_under_lease(
            root,
            python=executable,
            rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
        )
        assert result["status"] == "COMPLETE"
        assert process_calls == 54
        assert events[-1]["status"] == "COMPLETE"
        assert len(events[-1]["outputs"]) == 54
    else:
        with pytest.raises(RuntimeError, match="failed \(7\)"):
            orchestrator._run_under_lease(
                root,
                python=executable,
                rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
            )
        assert process_calls == failure_index + 1
        assert events[-1]["status"] == "FAILED"
        assert len(events[-1]["outputs"]) == failure_index
    assert maximum_active == 1
    assert all(event["status"] == "ACTIVE" for event in events[:-1])


def test_k_orchestrator_lease_wraps_the_entire_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    held = False

    @contextmanager
    def fake_lease(run_dir: Path, *, purpose: str):
        nonlocal held
        assert "54-job" in purpose
        held = True
        try:
            yield run_dir.parent / ".d1_heavy_resource.lock"
        finally:
            held = False

    def fake_matrix(*args: object, **kwargs: object) -> dict[str, object]:
        assert held is True
        return {"status": "COMPLETE"}

    monkeypatch.setattr(orchestrator, "assert_writable_prelock", lambda root: None)
    monkeypatch.setattr(orchestrator, "exclusive_heavy_resource_lease", fake_lease)
    monkeypatch.setattr(orchestrator, "_run_under_lease", fake_matrix)
    result = orchestrator.run(
        root,
        python=tmp_path / "python",
        rank1_run_dir=k_execution.CANONICAL_RANK1_RUN_DIR,
    )
    assert result["status"] == "COMPLETE"
    assert held is False
