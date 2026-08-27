from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from gtmask_counterfactual.contracts import RunState
from tools.gtmask_counterfactual.run_d1_case_b import _assert_command_stage
from tools.gtmask_counterfactual.run_route import _assert_route_stage


def _pipeline(root: Path, status: RunState, *, execution_count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_status.json").write_text(
        json.dumps(
            {
                "status": status.value,
                "counterfactual_execution_count": execution_count,
            }
        ),
        encoding="utf-8",
    )


def test_g1_c1_stage_guard_runs_before_heavy_gate(tmp_path: Path) -> None:
    _pipeline(tmp_path, RunState.P3_PROTOCOL_LOCKED, execution_count=0)
    _assert_route_stage(tmp_path, route="c1", phase="pilot", resume=False)

    _pipeline(tmp_path, RunState.P3_PROTOCOL_LOCKED, execution_count=0)
    _assert_route_stage(tmp_path, route="c1", phase="pilot", resume=True)

    _pipeline(tmp_path, RunState.P4_C1_PILOT_PASS, execution_count=0)
    _assert_route_stage(tmp_path, route="c1", phase="full", resume=False)
    with pytest.raises(PermissionError, match="requires --resume"):
        _assert_route_stage(tmp_path, route="c1", phase="pilot", resume=False)

    _pipeline(tmp_path, RunState.P5_C1_FULL_COMPLETE, execution_count=0)
    _assert_route_stage(tmp_path, route="g1", phase="full", resume=False)
    with pytest.raises(ValueError, match="C1-only"):
        _assert_route_stage(tmp_path, route="g1", phase="pilot", resume=False)


def test_d1_stage_guard_separates_preprotocol_replay_and_gt_work(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        run_dir=tmp_path,
        command="execute-predicted-candidates",
        resume=False,
    )
    _pipeline(tmp_path, RunState.P0_AUDIT, execution_count=0)
    _assert_command_stage(args)

    _pipeline(tmp_path, RunState.P3_PROTOCOL_LOCKED, execution_count=1)
    with pytest.raises(PermissionError, match="pre-protocol P0"):
        _assert_command_stage(args)

    args.command = "execute-candidates"
    _pipeline(tmp_path, RunState.P5B_G1_FULL_COMPLETE, execution_count=0)
    with pytest.raises(PermissionError, match="requires P10"):
        _assert_command_stage(args)

    _pipeline(tmp_path, RunState.P10_INDEPENDENT_RECOMPUTE_PASS, execution_count=0)
    for relative in (
        "08_metrics/POSTPROCESS_MANIFEST.json",
        "13_figures/FIGURES_MANIFEST.json",
        "14_galleries/GALLERY_MANIFEST.json",
        "15_reports/REPORTS_MANIFEST.json",
        "16_independent_recompute/INDEPENDENT_VALIDATION.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    _assert_command_stage(args)

    _pipeline(tmp_path, RunState.P10_INDEPENDENT_RECOMPUTE_PASS, execution_count=1)
    completion = tmp_path / "d1_secondary/D1_SECONDARY_EXECUTION_COMPLETE.json"
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="only be revalidated"):
        _assert_command_stage(args)
    args.command = "assemble"
    args.resume = True
    _assert_command_stage(args)
