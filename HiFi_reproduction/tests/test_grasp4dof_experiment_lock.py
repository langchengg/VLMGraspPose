"""Synthetic contract tests for the immutable 4-DoF experiment lock."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.grasping.common.experiment_lock import (
    DEFAULT_FROZEN_ALIAS_NAME,
    DEFAULT_LOCK_RELATIVE_PATH,
    DEFAULT_MARKER_NAME,
    ExperimentLockError,
    LockDriftError,
    build_lock_candidate,
    verify_lock,
    write_lock_exclusive,
)
from tools.grasp4dof import lock_experiment as lock_cli


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.invalid")
    _git(repository, "config", "user.name", "Experiment Lock Test")
    source = repository / "model_source.py"
    source.write_text("MODEL = 'repeated-film'\n", encoding="utf-8")
    _git(repository, "add", "model_source.py")
    _git(repository, "commit", "-qm", "synthetic source")

    run = repository / "runs" / "synthetic_run"
    run.mkdir(parents=True)
    artifact_specs = {
        "repeatedfilm_source": ("source", "audit/repeatedfilm.json"),
        "splits": ("data", "manifests/splits.json"),
        "vendor": ("third_party", "third_party/vendor.json"),
        "predictions": ("source", "predictions/prediction_manifest.json"),
        "selected_model": ("model", "selection/model.json"),
        "selected_config": ("selected_config", "selection/selected.json"),
        "evaluator": ("evaluator", "evaluation/evaluator.json"),
    }
    artifacts: dict[str, dict[str, str]] = {}
    for name, (role, relative) in artifact_specs.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        artifacts[name] = {"role": role, "path": relative}

    config = {
        "run_id": run.name,
        "artifacts": artifacts,
        "source_files": ["model_source.py"],
        "lineage": {
            "repeated_film": {
                "source_manifest_artifact": "repeatedfilm_source",
                "checkpoint_sha256": "a" * 64,
            },
            "splits": {
                split: {"manifest_artifact": "splits", "sample_count": count}
                for split, count in (("train", 8), ("validation", 3), ("test", 4))
            },
            "prediction_manifest": {"artifact": "predictions"},
            "reference": {
                "manifest_artifact": "predictions",
                "reference_run": "/synthetic/reference",
                "direct_inputs": {"candidates": "d" * 64, "samples": "e" * 64},
            },
            "vendors": {
                "ggcnn2": {
                    "manifest_artifact": "vendor",
                    "commit": "0123456789abcdef",
                    "checkpoints": {"ggcnn2": "b" * 64},
                },
                "grconvnet": {
                    "manifest_artifact": "vendor",
                    "commit": "fedcba9876543210",
                    "checkpoints": {"grconvnet": "c" * 64},
                },
            },
        },
        "protocol": {
            "preprocess": {"depth_units": "metres", "invalid_fill": "zero"},
            "input": {"height": 300, "width": 300, "channels": ["depth"]},
            "device": {"requested": "mps", "fallback": "cpu"},
            "conditioning": {"mode": "repeated_film"},
            "crop": {"mode": "center", "size": 300},
            "gate": {"threshold": 0.5},
            "width": {"minimum_px": 5, "maximum_px": 150},
            "angle": {"period_degrees": 180},
            "fixed_grasp_height_px": 20,
            "peak": {"threshold": 0.2, "top_k": 5},
            "nms": {"radius_px": 12, "angle_degrees": 20},
            "analytic": {"iou_threshold": 0.25, "angle_threshold_degrees": 30},
            "evaluator": {"artifact": "evaluator", "version": 1},
            "primary_method": "repeatedfilm_grconvnet",
            "seed": 20260803,
            "expected_test_sample_count": 4,
        },
        "selection_inputs": {
            "checkpoint": {"split": "validation", "artifact": "selected_config"},
            "thresholds": {"split": "validation", "artifact": "selected_config"},
        },
    }
    return repository, run, config


def test_dry_run_prints_non_effective_candidate_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repository, run, config = _fixture(tmp_path)
    config_path = tmp_path / "candidate.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(lock_cli, "REPOSITORY_ROOT", repository)

    assert lock_cli.main(
        ["--run-dir", str(run), "--config", str(config_path), "--dry-run"]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["lock_status"] == "CANDIDATE_NOT_EFFECTIVE"
    assert output["effective"] is False
    assert not (run / DEFAULT_LOCK_RELATIVE_PATH).exists()
    assert not (run / DEFAULT_MARKER_NAME).exists()
    assert not (run / DEFAULT_FROZEN_ALIAS_NAME).exists()


def test_formal_lock_is_immutable_and_verifiable(tmp_path: Path) -> None:
    repository, run, config = _fixture(tmp_path)
    candidate = build_lock_candidate(config, run_dir=run, repository_root=repository)
    manifest = write_lock_exclusive(run, candidate)
    original = (run / DEFAULT_LOCK_RELATIVE_PATH).read_bytes()

    assert manifest["lock_status"] == "LOCKED"
    assert manifest["effective"] is True
    assert (run / DEFAULT_FROZEN_ALIAS_NAME).read_bytes() == (
        run / DEFAULT_LOCK_RELATIVE_PATH
    ).read_bytes()
    assert verify_lock(run)["manifest_content_sha256"] == manifest[
        "manifest_content_sha256"
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        write_lock_exclusive(run, candidate)
    assert (run / DEFAULT_LOCK_RELATIVE_PATH).read_bytes() == original


def test_verify_detects_locked_artifact_hash_drift(tmp_path: Path) -> None:
    repository, run, config = _fixture(tmp_path)
    candidate = build_lock_candidate(config, run_dir=run, repository_root=repository)
    write_lock_exclusive(run, candidate)
    (run / "predictions/prediction_manifest.json").write_text(
        '{"changed": true}\n', encoding="utf-8"
    )

    with pytest.raises(LockDriftError, match="locked artifact changed"):
        verify_lock(run)


def test_test_split_cannot_select_checkpoint(tmp_path: Path) -> None:
    repository, run, config = _fixture(tmp_path)
    config["selection_inputs"]["checkpoint"]["split"] = "test"

    with pytest.raises(ExperimentLockError, match="must use validation"):
        build_lock_candidate(config, run_dir=run, repository_root=repository)
