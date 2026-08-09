"""Synthetic contracts for validation-only G0/C0 transfer tuning."""

from __future__ import annotations

import copy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.grasp4dof import tune_transfer_backends as tuning


def _checkpoint(
    tmp_path: Path, checkpoint_id: str, *, channels: int = 4
) -> dict:
    path = tmp_path / checkpoint_id
    path.write_bytes(checkpoint_id.encode())
    return {
        "checkpoint_id": checkpoint_id,
        "path": str(path),
        "sha256": tuning._sha256_file(path),
        "dataset": "Jacquard" if "jacquard" in checkpoint_id else "Cornell",
        "modalities": ["depth"] if channels == 1 else ["depth", "rgb"],
        "input_channels": channels,
    }


def _base(tmp_path: Path, method_id: str = "G0") -> dict:
    checkpoint = _checkpoint(tmp_path, "checkpoint", channels=4)
    return tuning._base_config(method_id, checkpoint)


def test_declared_search_covers_required_factors_without_pilot_selection(
    tmp_path: Path,
) -> None:
    checkpoints = {
        "jacquard_rgbd": _checkpoint(tmp_path, "jacquard_rgbd", channels=4),
        "jacquard_depth": _checkpoint(tmp_path, "jacquard_depth", channels=1),
        "cornell_rgbd": _checkpoint(tmp_path, "cornell_rgbd", channels=4),
    }
    stage1 = tuning.g0_checkpoint_candidates(checkpoints)
    conditioning = tuning.conditioning_candidates(
        "G0", stage1[0].config, "g0_stage2_conditioning"
    )
    gates = tuning.gate_candidates("G0", conditioning[0].config, "gate")
    thresholds = tuning.threshold_candidates("G0", gates[0].config, "threshold")
    nms = tuning.nms_candidates("G0", thresholds[0].config, "nms")

    assert [(row.design["dataset"], row.design["input_channels"]) for row in stage1] == [
        ("Jacquard", 4),
        ("Jacquard", 1),
        ("Cornell", 4),
    ]
    assert {row.config["conditioning_variant"] for row in conditioning} == {
        "hard_mask",
        "dilated_crop",
    }
    assert {row.config["input_size"] for row in conditioning} == {224, 300}
    assert {
        row.config["dilation_fraction"]
        for row in conditioning
        if row.config["conditioning_variant"] == "dilated_crop"
    } == {0.10, 0.15, 0.20}
    assert {(row.config["center_gate_exponent"], row.config["jaw_gate_exponent"]) for row in gates} == set(
        tuning.GATE_GRID
    )
    assert {row.config["quality_threshold"] for row in thresholds} == {
        0.0,
        0.05,
        0.1,
    }
    assert len({tuning._sha256_bytes(tuning._canonical_bytes(row.config)) for row in nms}) == 3


def test_run_method_command_is_unconditionally_full_validation(tmp_path: Path) -> None:
    candidate = tuning.Candidate("C0", "stage", "candidate", _base(tmp_path, "C0"), {})
    command = tuning.build_run_method_command(
        python=Path("/python"),
        run_method=Path("/run_method.py"),
        run_dir=Path("/run"),
        candidate=candidate,
        config_path=Path("/config.json"),
        output_dir=Path("/output"),
    )

    assert command[command.index("--split") + 1] == "validation"
    assert "--limit" not in command
    assert "--sample-list" not in command
    assert "--oracle" not in command
    assert command[command.index("--method") + 1] == "C0"


def test_frozen_interpreter_rejects_any_other_executable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="transfer search requires"):
        tuning.frozen_interpreter_contract(tmp_path / "python")


def test_search_runner_preserves_venv_launcher_path(tmp_path: Path) -> None:
    runner = tuning.SearchRunner(
        run_dir=tmp_path,
        python=tuning.FROZEN_PYTHON,
        run_method=tuning.DEFAULT_RUN_METHOD,
        validation_contract={},
        interpreter_contract={},
    )

    assert runner.python == tuning.FROZEN_PYTHON.absolute()
    assert runner.python != tuning.FROZEN_PYTHON.resolve()


def test_search_runner_rejects_run_method_source_drift(tmp_path: Path) -> None:
    run_method = tmp_path / "run_method.py"
    run_method.write_text("print('frozen')\n", encoding="utf-8")
    runner = tuning.SearchRunner(
        run_dir=tmp_path,
        python=tuning.FROZEN_PYTHON,
        run_method=run_method,
        validation_contract={},
        interpreter_contract={},
    )

    run_method.write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="run_method source drift detected"):
        runner._assert_run_method_unchanged("synthetic test")


def test_validation_contract_rejects_nonvalidation_rows(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    values = ["validation"] * tuning.EXPECTED_VALIDATION_COUNT
    for kind in ("samples", "labels"):
        pq.write_table(
            pa.table({"split": values}),
            manifests / f"validation_{kind}.parquet",
        )
    assert tuning.validation_manifest_contract(tmp_path)["samples"]["rows"] == 3_778

    values[-1] = "val"
    pq.write_table(
        pa.table({"split": values}),
        manifests / "validation_labels.parquet",
    )
    assert tuning.validation_manifest_contract(tmp_path)["labels"][
        "normalised_split"
    ] == "validation"

    contaminated = copy.copy(values)
    contaminated[-1] = "forbidden"
    pq.write_table(
        pa.table({"split": contaminated}),
        manifests / "validation_labels.parquet",
    )
    with pytest.raises(ValueError, match="forbidden split values"):
        tuning.validation_manifest_contract(tmp_path)


def test_selection_requires_every_candidate_to_cover_full_validation() -> None:
    good = {
        "candidate_id": "good",
        "sample_count": tuning.EXPECTED_VALIDATION_COUNT,
        "j_at_1": 0.4,
        "j_at_5": 0.6,
        "no_grasp_rate": 0.01,
        "p50_latency_seconds": 0.02,
        "p95_latency_seconds": 0.04,
        "simplicity_score": 2,
    }
    incomplete = {**good, "candidate_id": "incomplete", "sample_count": 100}

    with pytest.raises(ValueError, match="complete validation"):
        tuning.select_validation_winner([good, incomplete])


def test_selection_uses_only_declared_validation_metrics_and_simplicity() -> None:
    base = {
        "sample_count": tuning.EXPECTED_VALIDATION_COUNT,
        "j_at_1": 0.4,
        "j_at_5": 0.6,
        "no_grasp_rate": 0.01,
        "p50_latency_seconds": 0.02,
        "p95_latency_seconds": 0.04,
        "simplicity_score": 2,
    }
    lower_j1 = {**base, "candidate_id": "lower_j1", "j_at_1": 0.39}
    winner = {**base, "candidate_id": "winner", "simplicity_score": 1}
    same_metrics_complex = {**base, "candidate_id": "complex", "simplicity_score": 3}

    assert tuning.select_validation_winner(
        [lower_j1, same_metrics_complex, winner]
    )["candidate_id"] == "winner"
