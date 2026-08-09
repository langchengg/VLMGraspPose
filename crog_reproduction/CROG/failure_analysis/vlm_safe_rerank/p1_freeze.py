from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .manifest import file_identity
from .replay import replay_existing_direct


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8")); digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def freeze_p1_diagnostic(
    *,
    output_dir: str | Path,
    existing_run_root: str | Path,
    replay_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir); old = Path(existing_run_root)
    decision_paths = list((old / "pilot/decisions").glob("*.json"))
    if len(decision_paths) != 200:
        raise RuntimeError(f"P1 diagnostic requires exactly 200 terminal decisions, observed {len(decision_paths)}")
    replay = replay_existing_direct(output_dir=output / "replay", existing_run_root=old, **dict(replay_kwargs))
    models = replay["direct"]["models"]
    diagnostic_no_go = any(int(row["corrected"]["net"]) <= 0 for row in models.values())
    payload = {
        "schema_version": "1.0.0", "protocol": "P1_direct_full_list_diagnostic",
        "decision_count": len(decision_paths), "decision_tree_sha256": _tree_hash(decision_paths),
        "pilot_manifest": file_identity(old / "pilot_manifest.json"),
        "cache": file_identity(old / "gemini_cache.sqlite"),
        "source_run": str(old.resolve()), "models": models,
        "paired_common_valid": replay["direct"].get("paired_common_valid"),
        "paired_selected_id_agreement_rate": replay["direct"].get("paired_selected_id_agreement_rate"),
        "ledger": replay["ledger"], "diagnostic_NO_GO": diagnostic_no_go,
        "eligible_for_primary": False,
        "reason": "Direct API replacement is diagnostic only; q-only remains protected default.",
    }
    path = output / "P1_DIAGNOSTIC_FREEZE.json"
    descriptor = __import__("os").open(path, __import__("os").O_WRONLY | __import__("os").O_CREAT | __import__("os").O_EXCL, 0o600)
    try:
        __import__("os").write(descriptor, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        __import__("os").fsync(descriptor)
    finally:
        __import__("os").close(descriptor)
    return payload

