from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from robustness_completion import cli
from robustness_completion.common import RUN_ID, sha256_file
from robustness_completion.runtime_6d import MissingFormalCheckpointError


def test_profile_runtime_full_uses_lazy_aggregation_contract(monkeypatch) -> None:
    calls = []

    def aggregate_runtime(repo, run_dir, resume):
        calls.append((repo, run_dir, resume))
        return {"status": "PARTIAL_RUNTIME_FULL_PROFILE"}

    monkeypatch.setattr(
        cli,
        "_runtime_aggregation_module",
        lambda: SimpleNamespace(aggregate_runtime=aggregate_runtime),
    )
    repo = Path("/repo")
    run_dir = Path("/repo/artifacts/child")
    result = cli._aggregate_runtime(repo, run_dir, resume=True)

    assert result["status"] == "PARTIAL_RUNTIME_FULL_PROFILE"
    assert calls == [(repo, run_dir, True)]


def test_all_runs_locked_order_and_partial_is_success(monkeypatch, capsys) -> None:
    order = []

    def stage(name, result):
        def run(*_args, **kwargs):
            order.append((name, kwargs.get("resume")))
            return result

        return run

    monkeypatch.setattr(cli, "_repo", lambda: Path("/repo"))
    monkeypatch.setattr(cli, "_run_duplicate_map", stage("map", {"status": "COMPLETE"}))
    monkeypatch.setattr(
        cli, "_run_duplicate_visual", stage("visual", {"status": "COMPLETE"})
    )
    monkeypatch.setattr(
        cli, "_run_duplicate_evaluation", stage("evaluation", {"status": "COMPLETE"})
    )
    monkeypatch.setattr(
        cli, "_run_duplicate_report", stage("duplicate_report", {"status": "COMPLETE"})
    )
    monkeypatch.setattr(
        cli,
        "_build_runtime_adapters",
        stage("subset", {"status": "PARTIAL_RUNTIME_PREFLIGHT"}),
    )
    monkeypatch.setattr(
        cli,
        "_aggregate_runtime",
        stage("aggregate", {"status": "PARTIAL_RUNTIME_FULL_PROFILE"}),
    )
    monkeypatch.setattr(
        cli,
        "_generate_report",
        stage(
            "report",
            {
                "status": "PARTIAL_ROBUSTNESS_COMPLETION",
                "formal_remaining_results_emitted": False,
                "duplicate_complete": True,
                "runtime_complete": False,
            },
        ),
    )
    recorded = []
    monkeypatch.setattr(cli, "_record_final_manifest", lambda *args: recorded.append(args))

    assert cli.main(["all", "--run-id", RUN_ID, "--resume"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert [name for name, _resume in order] == [
        "map",
        "visual",
        "evaluation",
        "duplicate_report",
        "subset",
        "aggregate",
        "report",
    ]
    assert [resume for name, resume in order if name in {"map", "visual", "evaluation", "duplicate_report", "aggregate"}] == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert result["status"] == "PARTIAL_ROBUSTNESS_COMPLETION"
    assert result["formal_remaining_results_emitted"] is False
    assert len(recorded) == 1


def test_build_runtime_adapters_locks_subset_and_reports_raw_blocker(
    monkeypatch, tmp_path
) -> None:
    from robustness_completion import runtime_4d, runtime_6d, runtime_common

    runtime_root = tmp_path / "runtime_full"
    runtime_root.mkdir()
    subset_path = runtime_root / "profile_subset_manifest.json"
    subset_path.write_text("{}\n", encoding="utf-8")
    subset = {
        "profile_4d": {"count": 100},
        "parity_4d": {"count": 20},
        "profile_6d": {"count": 98},
        "parity_6d": {"count": 20},
    }
    calls = []
    monkeypatch.setattr(cli, "_prepare", lambda _repo, run_dir: run_dir)
    monkeypatch.setattr(
        runtime_common,
        "build_combined_subset_manifest",
        lambda _repo, _run_dir: calls.append("subset") or subset,
    )
    monkeypatch.setattr(
        runtime_4d,
        "asset_preflight",
        lambda _repo, _run_dir, route, **_kwargs: {
            "status": (
                "NOT_EXECUTED_IMPLEMENTATION_INCOMPLETE" if route == "D1" else "PASS"
            ),
            "route": route,
            "device": "mps",
            "retrospective": route == "D1",
            "complete_deployment": False if route == "D1" else None,
            "candidate_feature_score_caches_read": False,
            "records": [],
        },
    )
    contract = SimpleNamespace(hashes={"source": "abc"})
    selection = SimpleNamespace(
        groups=tuple(range(98)),
        parity_groups=tuple(range(20)),
        warmup_groups=tuple(range(5)),
        ordering_sha256="def",
    )
    monkeypatch.setattr(runtime_6d, "validate_source_contract", lambda _repo: contract)
    monkeypatch.setattr(runtime_6d, "select_profile_groups", lambda _contract: selection)

    def missing(_contract):
        raise MissingFormalCheckpointError("MISSING_FORMAL_CHECKPOINT: absent")

    monkeypatch.setattr(runtime_6d, "assert_formal_reranker_available", missing)

    result = cli._build_runtime_adapters(tmp_path, tmp_path)

    assert calls == ["subset"]
    assert result["status"] == "PARTIAL_RUNTIME_PREFLIGHT"
    assert result["route_contracts"]["6d"]["oracle"]["methods"]["raw"][
        "blocker_code"
    ] == "MISSING_FORMAL_CHECKPOINT"
    assert result["subset_counts"] == {
        "4d_profile": 100,
        "4d_parity": 20,
        "6d_profile": 98,
        "6d_parity": 20,
    }
    assert (runtime_root / "adapter_preflight.json").is_file()


def test_parity_records_require_complete_fixed_evidence(tmp_path) -> None:
    four_path = tmp_path / "4d.json"
    sample_ids = [f"s{index}" for index in range(20)]
    parity_rows = [
        {
            "route": "CROG",
            "method": "gated",
            "mode": "parity",
            "device": "mps",
            "sample_id": sample_id,
            "status": "PASS",
            "candidate_parity": True,
            "feature_parity": True,
            "ranker_parity": True,
            "gate_parity": True,
        }
        for sample_id in sample_ids
    ]
    four_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "complete_deployment": True,
                "route": "CROG",
                "method": "gated",
                "preregistration_sha256": cli.PREREGISTRATION_SHA256,
                "cache_use": "parity_comparison_only",
                "parity_sample_ids": sample_ids,
                "parity_count": 20,
                "failure_count": 0,
                "failures": [],
                "parity_rows": parity_rows,
                "candidate_parity": True,
                "feature_parity": True,
                "ranker_parity": True,
                "gate_parity": True,
            }
        ),
        encoding="utf-8",
    )
    assert cli._four_d_parity_record(four_path, "CROG", sample_ids)["passed"] is True

    six_root = tmp_path / "six"
    six_root.mkdir()
    status_path = six_root / "parity_result.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "PARITY_COMPLETE",
                "parity_passed": True,
                "parity_complete": True,
                "parity_groups_checked": 20,
                "route": "oracle",
                "method": "native",
                "mode": "parity",
                "device": "cpu",
                "preregistration_sha256": cli.PREREGISTRATION_SHA256,
                "cache_disabled": True,
                "candidate_cache_used_for_timing": False,
                "feature_cache_used_for_timing": False,
                "score_cache_used_for_timing": False,
            }
        ),
        encoding="utf-8",
    )
    rows = {name: [True] * 20 for name in cli.PARITY_BOOLEAN_COLUMNS}
    rows["group_id"] = [f"g{index}" for index in range(20)]
    rows["sample_id"] = rows["group_id"]
    rows["route"] = ["oracle"] * 20
    rows["method"] = ["native"] * 20
    rows["mode"] = ["parity"] * 20
    rows["device"] = ["cpu"] * 20
    rows["status"] = ["passed"] * 20
    rows["parity_index"] = list(range(20))
    rows["candidate_count_actual"] = [3] * 20
    rows["candidate_count_frozen"] = [3] * 20
    rows["max_translation_error_m"] = [0.0] * 20
    rows["max_rotation_error_rad"] = [0.0] * 20
    rows["max_width_error_m"] = [0.0] * 20
    pd.DataFrame(rows).to_parquet(six_root / "parity_results.parquet", index=False)
    assert cli._six_d_parity_record(
        status_path, "oracle", "native", rows["group_id"]
    )["passed"] is True

    rows["features_equal"][-1] = False
    pd.DataFrame(rows).to_parquet(six_root / "parity_results.parquet", index=False)
    assert cli._six_d_parity_record(
        status_path, "oracle", "native", rows["group_id"]
    )["passed"] is False


def test_final_manifest_merge_is_atomic_idempotent_and_preserves_commands(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    original_command = {"argv": ["old", "command"]}
    manifest_path.write_text(
        json.dumps({"status": "IN_PROGRESS", "commands": [original_command], "stages": {}}),
        encoding="utf-8",
    )
    artifact = tmp_path / "runtime_full/runtime_summary.csv"
    artifact.parent.mkdir()
    artifact.write_text("route,status\nA,partial\n", encoding="utf-8")
    report = {
        "status": "PARTIAL_ROBUSTNESS_COMPLETION",
        "formal_remaining_results_emitted": False,
        "duplicate_complete": True,
        "runtime_complete": False,
    }
    command = ["robustness_completion", "report", "--run-id", RUN_ID]

    cli._record_final_manifest(tmp_path, report, command)
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    cli._record_final_manifest(tmp_path, report, command)
    second = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert first == second
    assert first["commands"] == [original_command, {"argv": command}]
    assert first["status"] == "PARTIAL_ROBUSTNESS_COMPLETION"
    assert first["formal_remaining_results_emitted"] is False
    assert first["stages"]["runtime"]["status"] == "PARTIAL"
    assert first["stages"]["runtime"]["artifact_sha256"][
        "runtime_full/runtime_summary.csv"
    ] == sha256_file(artifact)


def test_record_command_is_idempotent_and_preserves_run_status(tmp_path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "PARTIAL_REMAINING_ROBUSTNESS_EXPERIMENTS",
                "formal_remaining_results_emitted": False,
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    command = ["robustness_completion", "audit", "--run-id", RUN_ID, "--resume"]

    cli._record_command(tmp_path, command)
    cli._record_command(tmp_path, command)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["commands"] == [{"argv": command}]
    assert manifest["status"] == "PARTIAL_REMAINING_ROBUSTNESS_EXPERIMENTS"
    assert manifest["formal_remaining_results_emitted"] is False
