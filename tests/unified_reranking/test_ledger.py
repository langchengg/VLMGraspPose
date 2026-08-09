from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.unified_reranking.ledger import ledger_stage, render_ledger_commands


def test_resume_updates_stage_instead_of_duplicating_it(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.sqlite"
    for artifact in ("first", "second"):
        with ledger_stage(ledger, stage="P0", substage="audit") as state:
            state["artifact_path"] = artifact
    with sqlite3.connect(ledger) as connection:
        rows = connection.execute(
            "SELECT status, seed, artifact_path FROM stages"
        ).fetchall()
    assert rows == [("COMPLETE", -1, "second")]


def test_same_nonempty_command_resumes_one_command_bound_row(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.sqlite"
    command = "python -m synthetic --fixed-grid"
    for artifact in ("first", "second"):
        with ledger_stage(
            ledger,
            stage="P7",
            substage="screen-cell",
            command=command,
        ) as state:
            state["artifact_path"] = artifact
    with sqlite3.connect(ledger) as connection:
        rows = connection.execute(
            "SELECT substage,status,command,artifact_path FROM stages"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0].startswith("screen-cell__cmd_")
    assert rows[0][1:] == ("COMPLETE", command, "second")


def test_full_experiment_identity_is_recorded(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.sqlite"
    with ledger_stage(
        ledger,
        stage="P7",
        substage="fold",
        route="g1",
        evidence_track="T2_matched_common",
        pool="top5",
        method="R4",
        feature_set="all",
        loss="ranknet",
        encoder="mlp",
        seed=42,
    ):
        pass
    with sqlite3.connect(ledger) as connection:
        row = connection.execute(
            "SELECT route,evidence_track,pool,method,feature_set,loss,encoder,seed,status FROM stages"
        ).fetchone()
    assert row == (
        "g1",
        "T2_matched_common",
        "top5",
        "R4",
        "all",
        "ranknet",
        "mlp",
        42,
        "COMPLETE",
    )


def test_commands_log_is_nonempty_and_exactly_reconciles_to_ledger(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "run_ledger.sqlite"
    with ledger_stage(
        ledger,
        stage="P7",
        substage="cell",
        route="g1",
        command="python -m synthetic --route g1",
    ):
        pass
    log = tmp_path / "commands.log"
    assert log.read_text(encoding="utf-8") == render_ledger_commands(ledger)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["command"] == "python -m synthetic --route g1"
    assert records[0]["stage"] == "P7"
    assert records[0]["status"] == "COMPLETE"
    assert records[0]["start_time"] and records[0]["end_time"]


def test_matrix_hyperparameter_trials_have_distinct_concurrent_ledger_rows(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "run_ledger.sqlite"
    commands = [
        "python -m matrix --alpha 0.5",
        "python -m matrix --alpha 0.25",
    ]

    def record(index: int) -> None:
        with ledger_stage(
            ledger,
            stage="P7",
            substage="matrix_validation_g1_T2_mlp_ranknet_42_None",
            route="g1",
            evidence_track="T2_matched_common",
            pool="top5",
            method="mlp",
            feature_set="all",
            loss="ranknet",
            encoder="mlp",
            seed=42,
            command=commands[index],
        ) as state:
            state["artifact_path"] = f"trial-{index}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(record, range(2)))
    with sqlite3.connect(ledger) as connection:
        rows = connection.execute(
            "SELECT substage,status,command,artifact_path FROM stages ORDER BY substage"
        ).fetchall()
    assert len(rows) == 2
    assert all(row[1] == "COMPLETE" for row in rows)
    assert {row[2] for row in rows} == set(commands)
    assert {row[3] for row in rows} == {"trial-0", "trial-1"}
