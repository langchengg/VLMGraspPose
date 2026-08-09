"""Run-scoped exact-model availability evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import EXACT_MODEL_IDS


def audited_unavailable_models(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Return exact models with an explicit provider hard-stop artifact."""
    run = Path(run_dir)
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((run / "audit").glob("*_HARD_STOP.json")):
        value = json.loads(path.read_text())
        model = str(value.get("model", ""))
        status = str(value.get("status", ""))
        if model in EXACT_MODEL_IDS and status.startswith("HARD_STOP_"):
            records[model] = {
                "path": str(path),
                "status": status,
                "recorded_at_utc": value.get("recorded_at_utc"),
            }
    return records
