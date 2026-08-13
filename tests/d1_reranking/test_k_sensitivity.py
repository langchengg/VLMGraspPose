from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from d1_reranking.execution import artifact_record
from d1_reranking.k_sensitivity import (
    build_k_sensitivity_plan,
    load_k_sensitivity_plan,
    validate_k_sensitivity_plan,
    write_k_sensitivity_plan,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file


def _manifest(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return path


def _plain(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _synthetic_run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "run"
    code_path = _plain(tmp_path / "planner_source.py", "VERSION = 1\n")
    _plain(root / "04_splits" / "fold_assignments.parquet", "synthetic-folds\n")
    _plain(root / "01_manifests" / "d1_paired_train.parquet", "synthetic-train\n")
    _plain(
        root / "01_manifests" / "d1_paired_validation.parquet",
        "synthetic-validation\n",
    )

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
    selected_configuration = {
        "schema_version": 1,
        "route": "D1",
        "pool": "top5",
        "track": "T2_matched_common",
        "method": "R5",
        "encoder": "lambdamart",
        "loss": "lambdarank",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
    }
    trial_path = _manifest(
        root
        / "07_validation"
        / "primary_selection"
        / "trials"
        / "trial-r5"
        / "manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "trial_id": "trial-r5",
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
            "selected_trial_id": "trial-r5",
            "selected_configuration": selected_configuration,
            "candidate_test_labels_read": False,
            "sources": {"plan": artifact_record(primary_plan_path)},
            "artifacts": {"selected_trial_manifest": artifact_record(trial_path)},
        },
    )

    allnms_maxima = {"train": 17, "validation": 23}
    for split in ("train", "validation"):
        _manifest(
            root / "02_candidates" / split / "manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "configuration": {"route": "D1", "split": split},
                "summaries": {
                    "top10": {"maximum_candidates": 10},
                    "allnms": {"maximum_candidates": allnms_maxima[split]},
                },
            },
        )
        for pool, track in (
            ("top10", "T2_matched_common"),
            ("allnms", "T2_matched_common"),
            ("allnms", "T3_route_rich"),
        ):
            _manifest(
                root / "03_features" / split / pool / track / "manifest.json",
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
                },
            )
        for pool in ("top10", "allnms"):
            _manifest(
                root / "03_features" / split / pool / "labels" / "manifest.json",
                {
                    "schema_version": 1,
                    "status": "COMPLETE",
                    "split": split,
                    "pool": pool,
                    "candidate_test_labels_read": False,
                },
            )

    for pool in ("top10", "allnms"):
        _manifest(
            root / "05_calibration" / pool / "calibration_manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "configuration": {"route": "D1", "pool": pool},
                "selected_method": "platt",
                "candidate_test_labels_read": False,
            },
        )
    return root, code_path


def test_k_sensitivity_plan_has_exact_frozen_universe_and_sources(
    tmp_path: Path,
) -> None:
    root, code_path = _synthetic_run(tmp_path)
    plan = build_k_sensitivity_plan(root, tool_paths=(code_path,))

    assert plan["status"] == "PLANNED"
    assert plan["execution_authorized"] is False
    assert plan["test_inputs_referenced"] is False
    assert plan["development_splits"] == ["train", "validation"]
    assert plan["job_count"] == 54
    assert plan["allnms_max_candidates"] == 23
    assert plan["selected_primary"]["method"] == "R5"
    assert plan["sources"]["code"] == [artifact_record(code_path)]
    assert set(plan["sources"]["candidate_manifests"]) == {"train", "validation"}

    jobs = plan["jobs"]
    assert len({job["job_id"] for job in jobs}) == 54
    for scenario_id, expected_max in (
        ("top10_t2", 10),
        ("allnms_t2", 23),
        ("allnms_t3", 23),
    ):
        scenario_jobs = [
            job for job in jobs if job["configuration"]["scenario_id"] == scenario_id
        ]
        assert len(scenario_jobs) == 18
        assert {job["configuration"]["max_candidates"] for job in scenario_jobs} == {
            expected_max
        }
        for seed in (42, 123, 2026):
            seed_jobs = [
                job for job in scenario_jobs if job["configuration"]["seed"] == seed
            ]
            assert sorted(
                (job["configuration"]["mode"], job["configuration"]["held_fold"])
                for job in seed_jobs
            ) == [
                ("oof", 0),
                ("oof", 1),
                ("oof", 2),
                ("oof", 3),
                ("oof", 4),
                ("validation", None),
            ]
            assert all(
                job["configuration"]["selected_primary_configuration"]
                == plan["selected_primary"]["configuration"]
                for job in seed_jobs
            )

    assert validate_k_sensitivity_plan(plan) == plan
    assert plan["content_sha256"] == canonical_sha256(
        {key: value for key, value in plan.items() if key != "content_sha256"}
    )


def test_k_sensitivity_validator_rejects_rehashed_universe_drift(
    tmp_path: Path,
) -> None:
    root, code_path = _synthetic_run(tmp_path)
    plan = build_k_sensitivity_plan(root, tool_paths=(code_path,))
    tampered = deepcopy(plan)
    tampered["jobs"] = tampered["jobs"][:-1]
    tampered["job_count"] = len(tampered["jobs"])
    tampered["job_universe_sha256"] = canonical_sha256(
        [job["configuration"] for job in tampered["jobs"]]
    )
    tampered["job_ids_sha256"] = canonical_sha256(
        [job["job_id"] for job in tampered["jobs"]]
    )
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )

    with pytest.raises(RuntimeError, match="exact job universe"):
        validate_k_sensitivity_plan(tampered)


def test_k_sensitivity_resume_requires_exact_live_source_hashes(
    tmp_path: Path,
) -> None:
    root, code_path = _synthetic_run(tmp_path)
    destination = root / "configs" / "d1_k_sensitivity_plan.json"
    first = write_k_sensitivity_plan(
        destination, run_dir=root, tool_paths=(code_path,), resume=False
    )
    first_file_sha256 = sha256_file(destination)

    with pytest.raises(FileExistsError):
        write_k_sensitivity_plan(
            destination, run_dir=root, tool_paths=(code_path,), resume=False
        )
    resumed = write_k_sensitivity_plan(
        destination, run_dir=root, tool_paths=(code_path,), resume=True
    )
    assert resumed == first
    assert sha256_file(destination) == first_file_sha256

    code_path.write_text("VERSION = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        write_k_sensitivity_plan(
            destination, run_dir=root, tool_paths=(code_path,), resume=True
        )


def test_k_sensitivity_resume_rejects_invalid_self_hash(tmp_path: Path) -> None:
    root, code_path = _synthetic_run(tmp_path)
    destination = root / "configs" / "d1_k_sensitivity_plan.json"
    write_k_sensitivity_plan(
        destination, run_dir=root, tool_paths=(code_path,), resume=False
    )
    tampered = json.loads(destination.read_text(encoding="utf-8"))
    tampered["job_count"] = 53
    atomic_json(destination, tampered)

    with pytest.raises(RuntimeError, match="content hash mismatch"):
        load_k_sensitivity_plan(destination)
