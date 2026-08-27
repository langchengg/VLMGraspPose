from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from graspnet6d import cli
from graspnet6d.paper_artifacts import PaperArtifactResult
from graspnet6d.run_outputs import FormalRunOutputs, PublishedComponent


def _args(*, profile: str = "paper-lite", resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        command="all",
        profile=profile,
        run_id=None,
        resume=resume,
        condition="all",
    )


def test_real_ranker_smoke_preserves_formal_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "20260816_graspnet6d_vgn_lambdamart"
    run_dir.mkdir()
    rows = {
        split: tmp_path / f"{split}.parquet"
        for split in ("train", "validation", "test")
    }
    universes = {
        split: tmp_path / f"{split}_universe.parquet"
        for split in ("train", "validation", "test")
    }
    observed: dict[str, object] = {}

    def fake_assemble(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        observed["run_id"] = kwargs["run_id"]
        return SimpleNamespace(
            partition_rows=rows,
            partition_group_universes=universes,
        )

    def fake_smoke(*args: object, **kwargs: object) -> dict[str, object]:
        observed["smoke_args"] = args
        observed["minimum"] = kwargs["minimum_real_groups"]
        return {"status": "PASS", "scope": "real_data_ranker_integration_only"}

    monkeypatch.setattr(
        "graspnet6d.analysis_inputs.assemble_analysis_inputs", fake_assemble
    )
    monkeypatch.setattr("graspnet6d.smoke.run_real_ranker_smoke", fake_smoke)
    monkeypatch.setattr(
        cli,
        "_manifest_paths",
        lambda path: (path / "target.jsonl", path / "language.jsonl"),
    )
    monkeypatch.setattr(cli, "record_stage", lambda *args, **kwargs: None)

    result = cli._run_real_ranker_smoke_stage(_args(resume=True), run_dir)

    assert observed["run_id"] == run_dir.name
    assert observed["minimum"] == 20
    assert result["scope"] == "real_data_ranker_integration_only"


def test_report_stage_delegates_completion_to_paper_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "formal-run"
    manifest = run_dir / "paper_artifact_manifest.json"
    order: list[str] = []
    result = PaperArtifactResult(
        run_dir=run_dir,
        manifest_path=manifest,
        manifest_sha256="a" * 64,
        input_fingerprint="b" * 64,
        selected_predicted_condition="hifics_zero_shot_mask",
        executed_figures=("F01",),
        unexecuted_figures=(),
        outputs={"RESULTS.md": "c" * 64},
        resumed=False,
    )
    component = PublishedComponent(
        manifest_path=run_dir / "environment_provenance.json",
        manifest_sha256="d" * 64,
        input_fingerprint="e" * 64,
        outputs={"environment.txt": "f" * 64},
        resumed=False,
    )
    aggregate = FormalRunOutputs(
        run_dir=run_dir,
        manifest_path=run_dir / "run_outputs_manifest.json",
        manifest_sha256="1" * 64,
        selected_predicted_condition="hifics_zero_shot_mask",
        outputs={"environment.txt": "f" * 64},
        environment=component,
        candidates=component,
        selected_analysis=component,
        resumed=False,
    )
    monkeypatch.setattr(cli, "_require_formal_profile", lambda path: None)
    monkeypatch.setattr(cli, "_require_smoke_go", lambda path: None)
    monkeypatch.setattr(
        cli, "_resume_analyses", lambda args, path: order.append("analyses")
    )

    def fake_generate(path: Path, *, resume: bool) -> PaperArtifactResult:
        assert path == run_dir
        assert resume is True
        order.append("finalizer")
        return result

    def fake_record(*args: object, **kwargs: object) -> None:
        del args, kwargs
        order.append("stage-record")

    monkeypatch.setattr(
        "graspnet6d.paper_artifacts.generate_formal_paper_artifacts", fake_generate
    )
    monkeypatch.setattr(
        "graspnet6d.run_outputs.publish_formal_run_outputs",
        lambda path, *, resume: order.append("aggregates") or aggregate,
    )
    monkeypatch.setattr(cli, "record_stage", fake_record)

    payload = cli._run_report_stage(_args(resume=True), run_dir)

    assert order == ["analyses", "aggregates", "finalizer", "stage-record"]
    assert payload["paper_artifacts"]["manifest_sha256"] == "a" * 64
    assert payload["formal_run_outputs"]["manifest_sha256"] == "1" * 64


def test_execute_one_records_report_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    args = _args(resume=True)
    args.command = "report"
    monkeypatch.setattr(cli, "_run_dir", lambda args, command: tmp_path)
    monkeypatch.setattr(cli, "_config", lambda profile: (tmp_path, {}))
    monkeypatch.setattr(cli, "_run_report_stage", lambda args, path: {"ok": True})
    monkeypatch.setattr(
        cli,
        "_record_runtime",
        lambda run_dir, stage, **kwargs: calls.append((stage, kwargs["status"])),
    )

    assert cli._execute_one(args, "report") == 0
    assert calls == [("report", "COMPLETE")]


def test_run_dir_prepublishes_and_revalidates_formal_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "formal-run"
    run_dir.mkdir()
    calls: list[bool] = []
    args = _args()
    monkeypatch.setattr(cli, "create_run", lambda *args, **kwargs: run_dir)
    monkeypatch.setattr(
        cli,
        "_config",
        lambda profile: (tmp_path / "paper_lite.yaml", {"formal_results": True}),
    )
    monkeypatch.setattr(
        "graspnet6d.run_outputs.publish_environment",
        lambda path, *, resume: calls.append(resume),
    )

    assert cli._run_dir(args, command="all") == run_dir
    (run_dir / "environment_provenance.json").write_text("{}", encoding="utf-8")
    assert cli._run_dir(args, command="all") == run_dir
    assert calls == [False, True]


def test_smoke_profile_all_stops_before_formal_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(profile="smoke")
    calls: list[str] = []
    monkeypatch.setattr(cli, "_run_dir", lambda args, command: tmp_path)
    monkeypatch.setattr(
        cli, "_config", lambda profile: (tmp_path, {"formal_results": False})
    )
    monkeypatch.setattr(cli, "run_audit", lambda: calls.append("audit"))
    monkeypatch.setattr(cli, "run_device_benchmark", lambda: calls.append("benchmark"))
    monkeypatch.setattr(
        cli, "download", lambda *args, **kwargs: calls.append("download")
    )
    monkeypatch.setattr(
        cli, "prepare_experiment", lambda *args, **kwargs: calls.append("prepare")
    )
    monkeypatch.setattr(
        cli, "_run_masks_stage", lambda *args, **kwargs: calls.append("masks")
    )
    monkeypatch.setattr(
        cli, "_run_candidates_stage", lambda *args, **kwargs: calls.append("candidates")
    )
    monkeypatch.setattr(
        cli, "_run_labels_stage", lambda *args, **kwargs: calls.append("labels")
    )
    monkeypatch.setattr(
        cli, "_run_features_stage", lambda *args, **kwargs: calls.append("features")
    )
    monkeypatch.setattr(
        cli,
        "_run_real_ranker_smoke_stage",
        lambda *args, **kwargs: calls.append("real-ranker-smoke"),
    )
    monkeypatch.setattr(
        cli,
        "run_smoke",
        lambda *args, **kwargs: [SimpleNamespace(passed=True) for _ in range(12)],
    )
    monkeypatch.setattr(cli, "_record_runtime", lambda *args, **kwargs: None)

    assert cli._run_all(args) == 0
    assert calls == [
        "audit",
        "benchmark",
        "download",
        "prepare",
        "masks",
        "candidates",
        "labels",
        "features",
        "real-ranker-smoke",
    ]


def test_all_reraises_unexpected_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(profile="smoke")
    monkeypatch.setattr(cli, "_run_dir", lambda args, command: tmp_path)
    monkeypatch.setattr(
        cli, "_config", lambda profile: (tmp_path, {"formal_results": False})
    )
    monkeypatch.setattr(cli, "run_audit", lambda: None)
    monkeypatch.setattr(cli, "run_device_benchmark", lambda: None)
    monkeypatch.setattr(
        cli,
        "download",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    monkeypatch.setattr(cli, "record_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_mark_run_failed", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_record_runtime", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="unexpected"):
        cli._run_all(args)
