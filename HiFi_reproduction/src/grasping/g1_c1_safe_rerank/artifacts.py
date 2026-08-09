"""Atomic artifact IO and lifecycle state for the G1/C1 experiment."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


RUN_DIRECTORIES = (
    "00_audit",
    "01_candidate_contract",
    "02_features",
    "03_splits",
    "04_calibration",
    "05_models",
    "06_oof",
    "07_validation",
    "08_lock",
    "09_formal_test",
    "10_statistics",
    "11_figures",
    "12_failure_galleries",
    "13_reports",
    "14_independent_recompute",
    "configs",
    "checkpoints",
    "predictions",
    "audit",
    "data",
    "models/local_rankers",
    "models/gates",
    "oof",
    "validation",
    "formal_test",
    "api/raw",
    "api/parsed",
    "prompts",
    "tables",
    "plots",
    "failure_gallery",
    "logs",
)


def initialize_run(run_dir: str | Path) -> Path:
    run = Path(run_dir).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    for relative in RUN_DIRECTORIES:
        (run / relative).mkdir(parents=True, exist_ok=True)
    (run / ".DO_NOT_PRUNE").touch(exist_ok=True)
    (run / ".RUN_ACTIVE").touch(exist_ok=True)
    (run / "commands.log").touch(exist_ok=True)
    if not (run / "phase_status.json").exists():
        atomic_json(
            run / "phase_status.json",
            {
                "status": "INITIALIZED",
                "current_phase": "audit",
                "created_at_utc": utc_now(),
                "updated_at_utc": utc_now(),
                "phases": {},
            },
        )
    return run


def update_phase(
    run_dir: str | Path,
    phase: str,
    status: str,
    **details: Any,
) -> None:
    run = Path(run_dir)
    path = run / "phase_status.json"
    state = read_json(path) if path.is_file() else {"phases": {}}
    phases = dict(state.get("phases", {}))
    previous = dict(phases.get(phase, {}))
    previous.update(details)
    previous["status"] = status
    previous["updated_at_utc"] = utc_now()
    phases[phase] = previous
    state.update(
        {
            "status": "RUNNING" if status != "BLOCKED" else "BLOCKED",
            "current_phase": phase,
            "updated_at_utc": utc_now(),
            "phases": phases,
        }
    )
    atomic_json(path, state)
