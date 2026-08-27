from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gtmask_counterfactual.io import canonical_sha256


_TOOL = (
    Path(__file__).resolve().parents[2]
    / "tools/gtmask_counterfactual/bootstrap.py"
)
_SPEC = importlib.util.spec_from_file_location("_test_gtmask_bootstrap", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
bootstrap_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap_tool)


def test_failed_resource_gate_is_persisted_before_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_failed_gate"
    gate = {
        "schema_version": 1,
        "status": "FAIL",
        "failure_reasons": ["foreign_heavy_processes is non-empty"],
    }
    gate["content_sha256"] = canonical_sha256(gate)
    arguments = SimpleNamespace(
        full_rehash_source_inventory=True,
        run_dir=run_dir,
        unified_run=tmp_path / "unified",
        d1_run=tmp_path / "d1",
        collect_resource_gate=True,
        resource_gate=None,
        rank1_run_dir=tmp_path / "rank1",
        resume=False,
    )
    monkeypatch.setattr(bootstrap_tool, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        bootstrap_tool,
        "exclusive_d1_flock",
        lambda *_args, **_kwargs: nullcontext(tmp_path / ".lock"),
    )
    monkeypatch.setattr(
        bootstrap_tool,
        "collect_fresh_three_by_five_gate",
        lambda **_kwargs: dict(gate),
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        bootstrap_tool.main()

    failures = list(
        (run_dir / "00_audit/resource_gates").glob(
            "SOURCE_REHASH_GATE_FAILED_*.json"
        )
    )
    assert len(failures) == 1
    assert json.loads(failures[0].read_text(encoding="utf-8")) == gate
    message = json.loads(capsys.readouterr().err)
    assert message["status"] == "BLOCKED"
    assert message["resource_gate"] == str(failures[0])
    assert message["failure_reasons"] == gate["failure_reasons"]
    assert not (run_dir / "pipeline_status.json").exists()
