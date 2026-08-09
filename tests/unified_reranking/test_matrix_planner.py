from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from tools.unified_reranking.train_matrix import NEURAL_TRIALS, build_jobs
from tools.unified_reranking.select_validation_screen import _selection_entry
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.matrix_phase import load_matrix_phase_cells


def _feature_manifest(tmp_path) -> None:
    path = (
        tmp_path
        / "03_features"
        / "tracks"
        / "T2_matched_common"
        / "g1_validation"
        / "feature_manifest.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "model_feature_columns": [
                    "p_center",
                    "jaw_probability_min",
                    "angle_consistency",
                    "contact_depth_symmetry",
                    "finger_sweep_obstacle_max",
                ]
            }
        )
    )


def test_screen_plan_has_equal_loss_budget_and_validation_only(tmp_path) -> None:
    _feature_manifest(tmp_path)
    args = Namespace(
        run_dir=tmp_path,
        phase="screen",
        routes=["g1"],
        tracks=["T2_matched_common"],
        selection_json=None,
    )
    jobs = build_jobs(args)
    assert len(NEURAL_TRIALS) == 4
    assert len(jobs) == 46
    assert len({job.identifier for job in jobs}) == len(jobs)
    assert all("--mode" in job.command and job.command[job.command.index("--mode") + 1] == "validation" for job in jobs)


def test_screen_selection_entry_is_materialized_and_hash_bound(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
    configuration = {
        "encoder": "mlp",
        "loss": "ranknet",
        "source_identity": {"training_code_sha256": "a" * 64},
        "learning_rate": 3e-4,
    }
    entry = _selection_entry(configuration, manifest)
    assert entry["encoder"] == "mlp"
    assert entry["parameters"] == {"learning_rate": 3e-4}
    assert entry["screen_manifest"] == str(manifest.resolve())
    assert entry["screen_manifest_sha256"] == sha256_file(manifest)


def test_selected_and_encoder_plans_cover_three_seeds_and_five_oof_folds(tmp_path) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selections": {
                    "g1/T2_matched_common": {
                        "encoder": "mlp",
                        "loss": "ranknet",
                        "parameters": {
                            "learning_rate": 0.0003,
                            "weight_decay": 0.0001,
                            "alpha": 0.5,
                        },
                    }
                }
            }
        )
    )
    base = dict(
        run_dir=tmp_path,
        routes=["g1"],
        tracks=["T2_matched_common"],
        selection_json=selection_path,
    )
    selected = build_jobs(Namespace(phase="selected", **base))
    encoder = build_jobs(Namespace(phase="encoder", **base))
    assert len(selected) == 18
    assert len(encoder) == 90
    assert {job.command[job.command.index("--seed") + 1] for job in selected} == {
        "42",
        "123",
        "2026",
    }

    value = json.loads(selection_path.read_text())
    value["selections"]["g1/T2_matched_common"] = [
        value["selections"]["g1/T2_matched_common"],
        {
            "encoder": "lambdamart",
            "loss": "lambdarank",
            "parameters": {"num_leaves": 15},
        },
    ]
    selection_path.write_text(json.dumps(value))
    assert len(build_jobs(Namespace(phase="selected", **base))) == 36


def _phase_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    source = run / "source.bin"
    artifact = run / "predictions.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    artifact.write_bytes(b"prediction")
    configuration = {
        "route": "g1",
        "track": "T2_matched_common",
        "encoder": "mlp",
        "loss": "ranknet",
        "seed": 42,
        "mode": "validation",
        "held_fold": None,
        "source_identity": {"synthetic": True},
    }
    cell_key = canonical_sha256(configuration)[:16]
    cell = run / "07_validation" / "matrix_cells" / cell_key / "manifest.json"
    cell.parent.mkdir(parents=True)
    cell.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "configuration": configuration,
                "cell_key": cell_key,
                "sources": {"input": {"path": str(source), "sha256": sha256_file(source)}},
                "artifacts": {
                    "predictions": {"path": str(artifact), "sha256": sha256_file(artifact)}
                },
            }
        ),
        encoding="utf-8",
    )
    command = (
        "python",
        "-m",
        "tools.unified_reranking.train_matrix_cell",
        "--run-dir",
        str(run),
        "--route",
        "g1",
        "--track",
        "T2_matched_common",
        "--encoder",
        "mlp",
        "--loss",
        "ranknet",
        "--seed",
        "42",
        "--mode",
        "validation",
    )
    identifier = canonical_sha256(command)[:16]
    planner = Path(__file__).resolve().parents[2] / "tools/unified_reranking/train_matrix.py"
    plan = {
        "status": "PLANNED",
        "phase": "screen",
        "job_count": 1,
        "selection": None,
        "planner_tool": {"path": str(planner), "sha256": sha256_file(planner)},
        "jobs": [{"identifier": identifier, "command": list(command)}],
    }
    plan_path = run / "05_models" / "matrix_plans" / "screen_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    record = {"path": str(cell.resolve()), "sha256": sha256_file(cell)}
    execution = {
        "status": "COMPLETE",
        "phase": "screen",
        "job_count": 1,
        "selection": None,
        "plan": {"path": str(plan_path.resolve()), "sha256": sha256_file(plan_path)},
        "results": [
            {"identifier": identifier, "returncode": 0, "output_manifest": record}
        ],
        "output_manifests": [record],
    }
    latest = plan_path.parent / "screen_latest_execution.json"
    latest.write_text(json.dumps(execution), encoding="utf-8")
    return run, cell


def test_matrix_phase_consumes_only_exact_execution_inventory(tmp_path: Path) -> None:
    run, cell = _phase_fixture(tmp_path)
    off_plan = run / "07_validation" / "matrix_cells" / "off-plan" / "manifest.json"
    off_plan.parent.mkdir(parents=True)
    off_plan.write_text(cell.read_text(encoding="utf-8"), encoding="utf-8")
    cells = load_matrix_phase_cells(run, "screen")
    assert [path for _, path in cells] == [cell.resolve()]
    cell.write_text(cell.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_matrix_phase_cells(run, "screen")


def test_matrix_phase_rejects_cell_from_different_planned_job(
    tmp_path: Path,
) -> None:
    run, _ = _phase_fixture(tmp_path)
    plan_path = run / "05_models/matrix_plans/screen_plan.json"
    latest_path = run / "05_models/matrix_plans/screen_latest_execution.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    command = plan["jobs"][0]["command"]
    command[command.index("--route") + 1] = "c1"
    identifier = canonical_sha256(tuple(command))[:16]
    plan["jobs"][0]["identifier"] = identifier
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["plan"] = {"path": str(plan_path), "sha256": sha256_file(plan_path)}
    latest["results"][0]["identifier"] = identifier
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from command option --route"):
        load_matrix_phase_cells(run, "screen")
