from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd
import pytest

from d1_reranking import formal
from tools.d1_reranking import independent_recompute as recompute_module
from tools.d1_reranking.independent_recompute import independent_recompute
from unified_reranking.hashing import canonical_sha256


_FORMAL_TEST_PATH = Path(__file__).with_name("test_formal.py")
_SPEC = importlib.util.spec_from_file_location(
    "_d1_test_formal_fixture", _FORMAL_TEST_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE_MODULE)
_synthetic_run = _FIXTURE_MODULE._synthetic_run


@pytest.fixture
def stable_repository_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formal,
        "_repository_state",
        lambda: {
            "git_commit": "a" * 40,
            "dirty": False,
            "status_sha256": "b" * 64,
            "tracked_diff_sha256": "c" * 64,
        },
    )


def _completed_run(root: Path) -> None:
    plan = _synthetic_run(root)
    formal.create_formal_lock(root, evaluation_plan_path=plan)
    formal.run_formal_test_once(root)


def test_independent_recompute_positive(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    _completed_run(tmp_path)
    result = independent_recompute(tmp_path)
    assert result["status"] == "PASS"
    assert result["formal_execution_count"] == 1
    assert set(result["metrics"]) == set(formal.FORMAL_SYSTEMS)
    per_sample = pd.read_parquet(
        tmp_path / "17_independent_recompute" / "independent_per_sample.parquet"
    )
    assert len(per_sample) == 2 * len(formal.FORMAL_SYSTEMS)
    assert (
        tmp_path / "17_independent_recompute" / "INDEPENDENT_RECOMPUTE.md"
    ).is_file()
    events = [
        json.loads(line)
        for line in (tmp_path / "09_formal_test" / "test_access.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert (
        sum(
            event.get("event") == "d1_independent_recompute_raw_test_ground_truth_read"
            for event in events
        )
        == 1
    )
    assert independent_recompute(tmp_path, resume=True) == result


def test_independent_recompute_rejects_locked_geometry_tamper(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    _completed_run(tmp_path)
    plan = json.loads(
        (tmp_path / formal.FORMAL_PLAN_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    universe_path = Path(plan["systems"][-1]["candidate_universe"]["path"])
    universe = pd.read_parquet(universe_path)
    universe.loc[0, "cx_px"] = float(universe.loc[0, "cx_px"]) + 1.0
    universe.to_parquet(universe_path, index=False)
    with pytest.raises(RuntimeError, match="formal inventory"):
        independent_recompute(tmp_path)


def test_independent_recompute_rejects_replay_before_complete(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    plan = _synthetic_run(tmp_path)
    formal.create_formal_lock(tmp_path, evaluation_plan_path=plan)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        independent_recompute(tmp_path)


def test_independent_recompute_rejects_full_rank_metric_corruption(
    tmp_path: Path, stable_repository_state: None
) -> None:
    del stable_repository_state
    _completed_run(tmp_path)
    metrics_path = tmp_path / "09_formal_test" / "formal_test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["systems"]["d1_allnms_locked"]["rank_metrics"][-1]["ndcg_at_k"] = 0.125
    metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")

    manifest_path = tmp_path / "09_formal_test" / "formal_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("content_sha256", None)
    manifest["artifacts"]["metrics"] = formal._record(metrics_path)
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    execution_path = tmp_path / "09_formal_test" / formal.EXECUTION_NAME
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution.pop("content_sha256", None)
    execution["artifacts"]["metrics"] = formal._record(metrics_path)
    execution["artifacts"]["formal_test_manifest"] = formal._record(manifest_path)
    execution["content_sha256"] = canonical_sha256(execution)
    execution_path.write_text(json.dumps(execution, sort_keys=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="rank metric"):
        independent_recompute(tmp_path)


def test_independent_recompute_cli_records_ledger_and_commands(
    tmp_path: Path,
    stable_repository_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stable_repository_state
    _completed_run(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["independent_recompute.py", "--run-dir", str(tmp_path)],
    )
    assert recompute_module.main() == 0
    with sqlite3.connect(tmp_path / "run_ledger.sqlite") as connection:
        row = connection.execute(
            "SELECT status, command FROM stages WHERE stage='P17'"
        ).fetchone()
    assert row is not None and row[0] == "COMPLETE"
    assert "independent_recompute.py" in row[1]
    assert (tmp_path / "commands.log").is_file()
