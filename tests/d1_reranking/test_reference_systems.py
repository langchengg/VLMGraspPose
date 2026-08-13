from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from d1_reranking.reference_systems import assemble_reference_systems
from tools.d1_reranking import assemble_reference_systems as reference_cli
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.ledger import initialize_ledger


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _content(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    (root / "01_manifests").mkdir(parents=True)
    pd.DataFrame({"sample_id": ["sample-0", "sample-1"]}).to_parquet(
        root / "01_manifests/d1_paired_manifest.parquet", index=False
    )
    rows = []
    for index, route in enumerate(("CROG", "G1", "C1", "D1"), 1):
        rows.append(
            {
                "source_route": route,
                "sample_id": "sample-0",
                "candidate_id": f"{route.lower()}-0",
                "candidate_geometry_sha256": f"{index:064x}",
                "native_rank": 1,
                "native_score": 1.0 - index / 10,
                "cx_px": float(index),
                "cy_px": float(index),
                "theta_deg": 0.0,
                "width_px": 10.0,
                "height_px": 5.0,
            }
        )
    universe = pd.DataFrame(rows)
    for name in ("four_route_crog_default_router", "top20_union"):
        output = root / "08_lock/formal_inputs" / name
        output.mkdir(parents=True, exist_ok=True)
        universe_path = output / "candidate_universe.parquet"
        universe.to_parquet(universe_path, index=False)
        _content(
            output / "manifest.json",
            {
                "status": "COMPLETE",
                "candidate_test_labels_read": False,
                "artifacts": {"candidate_universe": _record(universe_path)},
            },
        )
    old = root / "old"
    old.mkdir()
    three = old / "three.parquet"
    top15_decisions = old / "top15_decisions.parquet"
    top15_ranking = old / "top15_ranking.parquet"
    decisions = pd.DataFrame(
        {
            "sample_id": ["sample-0", "sample-1"],
            "selected_source_route": ["G1", ""],
            "selected_candidate_id": ["g1-0", ""],
        }
    )
    decisions.to_parquet(three, index=False)
    decisions.to_parquet(top15_decisions, index=False)
    universe.loc[
        universe["source_route"].ne("D1"),
        ["source_route", "sample_id", "candidate_id", "native_score"],
    ].rename(columns={"native_score": "score"}).to_parquet(top15_ranking, index=False)
    lock = root / "old/FINAL_RUN_LOCK.json"
    lock.write_text(
        json.dumps(
            {
                "inventory": [
                    _record(path) for path in (three, top15_decisions, top15_ranking)
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _content(
        root / "configs/d1_four_route_extension_plan.json",
        {
            "status": "PLANNED",
            "sources": {"completed_three_route_final_lock": _record(lock)},
        },
    )
    return three, top15_decisions, top15_ranking


def test_reference_normalizer_and_imported_cli_main_are_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    three, top15_decisions, top15_ranking = _fixture(tmp_path)
    result = assemble_reference_systems(
        tmp_path,
        three_route_decisions_path=three,
        top15_decisions_path=top15_decisions,
        top15_ranking_path=top15_ranking,
    )
    assert set(result) == {
        "three_route_crog_default_router_reference",
        "top15_union_reference",
    }
    initialize_ledger(tmp_path / "run_ledger.sqlite")
    monkeypatch.setattr(
        reference_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "run_dir": tmp_path,
                "three_route_decisions": three,
                "top15_decisions": top15_decisions,
                "top15_ranking": top15_ranking,
                "resume": True,
            },
        )(),
    )
    assert reference_cli.main() == 0
    events = [
        json.loads(line)
        for line in (tmp_path / "09_formal_test/test_access.log")
        .read_text()
        .splitlines()
    ]
    assert (
        sum(event.get("stage") == "d1_read_only_reference_systems" for event in events)
        == 1
    )


def test_reference_cli_postlock_guard_precedes_ledger_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    three, top15_decisions, top15_ranking = _fixture(tmp_path)
    lock = tmp_path / "08_lock/FORMAL_TEST_LOCK.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        reference_cli,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "run_dir": tmp_path,
                "three_route_decisions": three,
                "top15_decisions": top15_decisions,
                "top15_ranking": top15_ranking,
                "resume": False,
            },
        )(),
    )
    ledger = tmp_path / "run_ledger.sqlite"
    assert not ledger.exists()
    with pytest.raises(PermissionError):
        reference_cli.main()
    assert not ledger.exists()
