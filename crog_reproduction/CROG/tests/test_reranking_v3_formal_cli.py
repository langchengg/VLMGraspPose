from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from failure_analysis.reranking_v3 import artifacts
from failure_analysis.reranking_v3.cli import _parser, _run_initial
from failure_analysis.reranking_v3.formal import verify_formal_ranking_outputs
from failure_analysis.reranking_v3.formal import PRIOR_TEST_EXPOSURE_DISCLOSURE
from failure_analysis.reranking_v3.experiment_config import ENSEMBLE_SEEDS, PERTURBATIONS
from failure_analysis.reranking_v3.protocol import write_locked_manifest
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
    "harm_cost": 5.0, "threshold": 0.1, "uncertainty_kappa": 1.0,
    "consensus": 2, "minimum_valid_fraction": 1.0,
}


def _write(path: Path, value: str = "artifact\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": sample_id,
            "candidate_ids": [f"{sample_id}-c{index}" for index in range(5)],
        }
        for sample_id in ("sample-a", "sample-b")
    ]


def _ranking_rows(method: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "method": method,
            "candidate_order": row["candidate_ids"],
        }
        for row in _candidate_rows()
    ]


def _cli_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Path]]:
    monkeypatch.setattr(artifacts, "OUTPUT_ROOT", tmp_path)
    run = tmp_path / "v3_fullchain_cli"
    run.mkdir()
    source = tmp_path / "inputs"
    values = {
        "config": _write(source / "config.json", "{}\n"),
        "checkpoint": _write(source / "model.pt"),
        "normalizer": _write(source / "normalizer.json", "{}\n"),
        "gate": _write(source / "gate.pt"),
        "gate_policy": _write(source / "gate-policy.json", json.dumps(GATE_POLICY)),
        "contract": _write(source / "contract.json", "{}\n"),
        "split": _write(source / "split.json", "{}\n"),
        "candidate": _write_jsonl(source / "candidates.jsonl", _candidate_rows()),
        "corrected": _write(source / "corrected.py", "# corrected\n"),
        "legacy": _write(source / "legacy.py", "# legacy\n"),
        "v2_manifest": _write(source / "v2-manifest.json", "{}\n"),
        "v2_ranking": _write(source / "v2-ranking.jsonl", "{}\n"),
        "lock_corrected": _write(source / "lock-corrected.jsonl"),
        "lock_legacy": _write(source / "lock-legacy.jsonl"),
        "test_corrected": _write(source / "test-corrected.jsonl"),
        "test_legacy": _write(source / "test-legacy.jsonl"),
        "feature_schema": _write(source / "feature-schema.json", "{}\n"),
        "oof": _write(source / "oof.json", "{}\n"),
    }
    evaluation = {
        "schema_version": "3.0.0", "kind": "v3_formal_evaluation_descriptor",
        "status": "frozen_before_evaluation",
        "cohorts": {
            "lockcheck": {
                "candidate": artifact_identity(values["candidate"]),
                "corrected_labels": artifact_identity(values["lock_corrected"]),
                "legacy_labels": artifact_identity(values["lock_legacy"]),
            },
            "test": {
                "candidate": artifact_identity(values["candidate"]),
                "corrected_labels": artifact_identity(values["test_corrected"]),
                "legacy_labels": artifact_identity(values["test_legacy"]),
            },
        },
        "formal_methods": list(METHODS), "evaluator_callback": EVALUATOR_CALLBACK,
        "bootstrap_iterations": 10_000, "seed": 20260801,
    }
    values["evaluation_descriptor"] = _write(
        source / "evaluation-descriptor.json", json.dumps(evaluation, sort_keys=True)
    )
    experiment = {
        "schema_version": "3.0.0", "kind": "v3_strict_experiment_descriptor",
        "status": "frozen_before_lockcheck",
        "selected_feature_groups": ["G0"],
        "excluded_feature_groups": [{"group": "G11", "reason": "unavailable"}],
        "architecture": {"name": "synthetic"},
        "partition_manifests": {
            name: artifact_identity(values["split"])
            for name in ("train", "calibration", "select", "lockcheck", "test")
        },
        "oof": {"group_key": "sequence_id", "fold_count": 3, "folds": [0, 1, 2],
                "manifest": artifact_identity(values["oof"])},
        "seeds": list(ENSEMBLE_SEEDS),
        "feature_schema": artifact_identity(values["feature_schema"]),
        "normalizers": {"development": artifact_identity(values["normalizer"])},
        "checkpoints": {"seed": artifact_identity(values["checkpoint"])},
        "alpha": 0.5, "gate_policy": GATE_POLICY,
        "perturbations": list(PERTURBATIONS),
        "inference_input_allowlist": {
            "lockcheck": ["candidate_features"],
            "test": ["candidate_features", "model_checkpoint"],
        },
        "formal_methods": list(METHODS),
        "callbacks": {"inference": INFERENCE_CALLBACK, "evaluator": EVALUATOR_CALLBACK},
        "git_diff_sha256": "0" * 64,
        "prior_test_exposure_disclosure": PRIOR_TEST_EXPOSURE_DISCLOSURE,
        "artifact_bindings": {
            "config": artifact_identity(values["config"]),
            "contract": artifact_identity(values["contract"]),
            "split": artifact_identity(values["split"]),
            "candidate": artifact_identity(values["candidate"]),
            "gate": {
                "policy": artifact_identity(values["gate_policy"]),
                "checkpoint": artifact_identity(values["gate"]),
            },
            "evaluators": {
                "corrected_scientific": artifact_identity(values["corrected"]),
                "legacy_official_compatibility": artifact_identity(values["legacy"]),
            },
            "v2": {
                "manifest": artifact_identity(values["v2_manifest"]),
                "ranking": artifact_identity(values["v2_ranking"]),
            },
        },
        "evaluation_descriptor": artifact_identity(values["evaluation_descriptor"]),
    }
    values["experiment_descriptor"] = _write(
        source / "experiment-descriptor.json", json.dumps(experiment, sort_keys=True)
    )
    return run, values


def _preliminary_argv(run: Path, values: dict[str, Path], *, dry_run: bool = False) -> list[str]:
    argv = [
        "lock-preliminary",
        "--output-dir", str(run),
        "--run-id", "cli-lifecycle",
        "--config", str(values["config"]),
        "--contract", str(values["contract"]),
        "--split-manifest", str(values["split"]),
        "--candidate-artifact", str(values["candidate"]),
        "--experiment-descriptor", str(values["experiment_descriptor"]),
        "--evaluation-descriptor", str(values["evaluation_descriptor"]),
        "--callback", INFERENCE_CALLBACK,
        "--evaluator-callback", EVALUATOR_CALLBACK,
        "--checkpoint-artifact", f"seed={values['checkpoint']}",
        "--normalizer-artifact", f"development={values['normalizer']}",
        "--gate-artifact", f"policy={values['gate_policy']}",
        "--gate-artifact", f"checkpoint={values['gate']}",
        "--evaluator-artifact", f"corrected_scientific={values['corrected']}",
        "--evaluator-artifact", f"legacy_official_compatibility={values['legacy']}",
        "--v2-artifact", f"manifest={values['v2_manifest']}",
        "--v2-artifact", f"ranking={values['v2_ranking']}",
    ]
    for method in METHODS:
        argv.extend(("--method", method))
    if dry_run:
        argv.append("--dry-run")
    return argv


def _run(argv: list[str], **callbacks: Any) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    return _run_initial(args, ["python", "-m", "failure_analysis.reranking_v3.cli", *argv], **callbacks)


def test_formal_cli_dry_run_never_creates_a_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, values = _cli_fixture(tmp_path, monkeypatch)
    result = _run(_preliminary_argv(run, values, dry_run=True))
    assert result["dry_run"] is True
    assert result["lock_or_claim_created"] is False
    assert not (run / "formal/preliminary_experiment_manifest.json").exists()
    assert not (run / "formal/preliminary_experiment_manifest.json.sha256").exists()


def test_missing_inference_callback_is_discovered_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, values = _cli_fixture(tmp_path, monkeypatch)
    _run(_preliminary_argv(run, values))
    preliminary = run / "formal/preliminary_experiment_manifest.json"
    argv = [
        "run-lockcheck", "--output-dir", str(run),
        "--formal-manifest", str(preliminary),
        "--input-artifact", f"candidate_features={values['candidate']}",
    ]
    with pytest.raises(ValueError, match="--callback is required"):
        _run(argv)
    assert not (run / "formal/lockcheck/LOCKCHECK_RUN_CLAIM.json").exists()


def test_evaluate_select_rejects_invalidated_default_and_accepts_explicit_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = _cli_fixture(tmp_path, monkeypatch)
    invalid = run / "selection"
    invalid.mkdir()
    _write(invalid / "INVALIDATED.json", "{}\n")
    with pytest.raises(PermissionError, match="explicitly invalidated"):
        _run(["evaluate-select", "--output-dir", str(run)])

    valid = run / "selection_widthfix_g10_v1"
    valid.mkdir()
    summary = _write(valid / "selection_summary.json", '{"status":"complete"}\n')
    result = _run(
        ["evaluate-select", "--output-dir", str(run), "--config", str(summary)]
    )
    assert result == {"status": "complete"}


def test_report_and_gallery_cli_dry_runs_use_real_descriptor_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, values = _cli_fixture(tmp_path, monkeypatch)
    manifest = run / "frozen_experiment_manifest.json"
    write_locked_manifest(
        manifest,
        {"code_fingerprint": "synthetic", "locked_artifacts": []},
        kind="v3_final_experiment_manifest",
    )
    independent_completion = _write(
        run / "independent_evaluate_complete.json",
        '{"kind":"independent_evaluate_run_complete","status":"complete"}\n',
    )
    gallery_descriptor = {
        "features_path": str(values["candidate"]),
        "corrected_labels_path": str(values["corrected"]),
        "legacy_labels_path": str(values["legacy"]),
        "raw_predictions_path": str(values["v2_ranking"]),
        "v2_predictions_path": str(values["v2_ranking"]),
        "v3_predictions_path": str(values["v2_ranking"]),
        "independent_evaluation_completion_path": str(independent_completion),
    }
    gallery_path = _write(
        run / "gallery_descriptor.json", json.dumps(gallery_descriptor),
    )
    gallery = _run([
        "build-gallery", "--output-dir", str(run), "--scope", "test",
        "--formal-manifest", str(manifest), "--config", str(gallery_path),
        "--dry-run",
    ])
    assert gallery["dry_run"] is True
    assert not (run / "evaluation/galleries").exists()

    machine_inputs = {
        "validation_rows": [], "lockcheck_rows": [], "test_rows": [],
        "pairwise_rows": [], "calibration_rows": [],
        "feature_ablation_rows": [], "subgroup_rows": [],
        "feature_provenance": {}, "conclusion_payload": {},
        "commands": ["synthetic"], "environment": {}, "tests": {},
    }
    report_descriptor = {
        "audit": {}, "selection": {}, "lockcheck": {}, "formal": {},
        "machine_inputs": machine_inputs,
        "evidence_artifacts": {
            "independent_evaluation_completion": str(independent_completion),
            "diagnostic": str(values["oof"]),
            "efficiency": str(values["normalizer"]),
            "subgroup": str(values["feature_schema"]),
            "gallery": str(gallery_path),
        },
    }
    report_path = _write(
        run / "report_descriptor.json", json.dumps(report_descriptor),
    )
    report = _run([
        "build-report", "--output-dir", str(run), "--scope", "test",
        "--formal-manifest", str(manifest), "--config", str(report_path),
        "--dry-run",
    ])
    assert report["dry_run"] is True
    assert not (run / "results").exists()


def test_formal_ranking_output_requires_distinct_explicit_method_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, values = _cli_fixture(tmp_path, monkeypatch)
    _run(_preliminary_argv(run, values))
    manifest = json.loads(
        (run / "formal/preliminary_experiment_manifest.json").read_text()
    )
    missing_method = _write_jsonl(
        tmp_path / "v2-source.jsonl",
        [
            {
                "sample_id": row["sample_id"],
                "candidate_order": row["candidate_ids"],
            }
            for row in _candidate_rows()
        ],
    )
    with pytest.raises(ValueError, match="unique method identity"):
        verify_formal_ranking_outputs(
            [missing_method], locked_manifest=manifest, scope="lockcheck"
        )

    ranking = _write_jsonl(
        tmp_path / "v3.jsonl", _ranking_rows("v3_locked_primary")
    )
    with pytest.raises(ValueError, match="must not share"):
        verify_formal_ranking_outputs(
            [ranking, ranking], locked_manifest=manifest, scope="lockcheck"
        )


def test_formal_cli_dispatches_complete_once_only_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, values = _cli_fixture(tmp_path, monkeypatch)
    _run(_preliminary_argv(run, values))
    preliminary = run / "formal/preliminary_experiment_manifest.json"

    def inference_callback(**kwargs: Any) -> list[Path]:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        return [
            _write_jsonl(output / f"{method}.jsonl", _ranking_rows(method))
            for method in METHODS
            if method != "q_only"
        ]

    lockcheck = _run(
        [
            "run-lockcheck", "--output-dir", str(run),
            "--formal-manifest", str(preliminary),
            "--callback", INFERENCE_CALLBACK,
            "--input-artifact", f"candidate_features={values['candidate']}",
        ],
        inference_callback=inference_callback,
    )

    def lockcheck_evaluator_callback(**kwargs: Any) -> list[Path]:
        assert kwargs["evaluation_scope"] == "lockcheck"
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        results = _write(output / "results_lockcheck.csv", "track,method\n")
        metric = {"sample_count": 2, "correct": 1, "oracle_correct": 2}
        summary = {
            "schema_version": "3.0.0",
            "kind": "v3_dual_track_evaluation",
            "status": "complete",
            "sample_count": 2,
            "methods": list(METHODS),
            "tracks": {
                track: {method: dict(metric) for method in METHODS}
                for track in (
                    "corrected_scientific",
                    "legacy_official_compatibility",
                )
            },
            "oracle_unchanged": True,
        }
        summary_path = _write(
            output / "summary.json", json.dumps(summary, sort_keys=True) + "\n"
        )
        return [results, summary_path]

    lockcheck_rankings = [
        f"{method}={run / f'inference/lockcheck_once/{method}.jsonl'}"
        for method in METHODS
        if method != "q_only"
    ]
    evaluate_lockcheck_argv = [
        "evaluate-lockcheck", "--output-dir", str(run),
        "--formal-manifest", str(preliminary),
        "--lockcheck-completion", lockcheck["completion_path"],
        "--candidate-artifact", str(values["candidate"]),
        "--callback", EVALUATOR_CALLBACK,
        "--input-artifact", f"corrected_labels={values['lock_corrected']}",
        "--input-artifact", f"legacy_labels={values['lock_legacy']}",
    ]
    for ranking in lockcheck_rankings:
        evaluate_lockcheck_argv.extend(("--method-ranking", ranking))
    lockcheck_evaluation = _run(
        evaluate_lockcheck_argv,
        evaluator_callback=lockcheck_evaluator_callback,
    )
    assert lockcheck_evaluation["oracle_identity_verified"] is True

    final = run / "frozen_experiment_manifest.json"
    formal_argv = [
        "run-test", "--output-dir", str(run),
        "--formal-manifest", str(final),
        "--callback", INFERENCE_CALLBACK,
        "--input-artifact", f"candidate_features={values['candidate']}",
        "--input-artifact", f"model_checkpoint={values['checkpoint']}",
    ]
    exact_formal_argv = [
        "python", "-m", "failure_analysis.reranking_v3.cli", *formal_argv,
    ]
    _run(
        [
            "lock-final", "--output-dir", str(run),
            "--formal-manifest", str(preliminary),
            "--lockcheck-completion", lockcheck["completion_path"],
            "--lockcheck-evaluation-completion",
            lockcheck_evaluation["completion_path"],
            "--test-input-artifact", f"candidate_features={values['candidate']}",
            "--test-input-artifact", f"model_checkpoint={values['checkpoint']}",
            "--exact-test-command", json.dumps(exact_formal_argv),
        ]
    )
    mismatched = [*formal_argv, "--resume"]
    with pytest.raises(PermissionError, match="argv differs"):
        _run(mismatched, inference_callback=inference_callback)
    assert not (run / "formal/test/TEST_RUN_CLAIM.json").exists()
    formal = _run(formal_argv, inference_callback=inference_callback)

    calls: list[str] = []

    def evaluator_callback(**kwargs: Any) -> list[Path]:
        calls.append("called")
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=False)
        return [_write(output / "dual-track-independent.json", '{"status":"passed"}\n')]

    rankings = [
        f"{method}={run / f'inference/test_once/{method}.jsonl'}"
        for method in METHODS
        if method != "q_only"
    ]
    argv = [
        "independent-evaluate", "--output-dir", str(run),
        "--formal-manifest", str(final),
        "--test-completion", formal["completion_path"],
        "--candidate-artifact", str(values["candidate"]),
        "--callback", EVALUATOR_CALLBACK,
        "--input-artifact", f"corrected_labels={values['test_corrected']}",
        "--input-artifact", f"legacy_labels={values['test_legacy']}",
    ]
    for ranking in rankings:
        argv.extend(("--method-ranking", ranking))
    independent = _run(argv, evaluator_callback=evaluator_callback)
    assert calls == ["called"]
    assert independent["candidate_identity_verified"] is True
    assert independent["method_identity_verified"] is True

    with pytest.raises(FileExistsError, match="already completed"):
        _run(formal_argv, inference_callback=inference_callback)
