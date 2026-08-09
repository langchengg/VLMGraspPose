#!/usr/bin/env python3
"""Verify protected repeated-FiLM, reference, and vendor inputs are unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.common.experiment_lock import verify_lock  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify(path: str | Path, expected: str, *, role: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    actual = _sha256(resolved)
    if actual != str(expected):
        raise RuntimeError(f"protected {role} changed: {resolved}")
    return {"path": str(resolved), "sha256": actual, "status": "UNCHANGED"}


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _vendor(manifest: Mapping[str, Any]) -> dict[str, Any]:
    git = manifest["git"]
    checkout = Path(str(git["path"])).resolve()
    commit = _git(checkout, "rev-parse", "HEAD")
    status = _git(checkout, "status", "--porcelain", "--untracked-files=all")
    if commit != str(manifest["pinned_commit"]) or status:
        raise RuntimeError(f"protected vendor checkout changed: {checkout}")
    files: list[dict[str, Any]] = []
    for name, row in manifest.get("source_files", {}).items():
        files.append(_verify(row["path"], row["sha256"], role=f"vendor source {name}"))
    for row in manifest.get("checkpoints", []):
        files.append(
            _verify(
                row["path"],
                row["sha256"],
                role=f"vendor checkpoint {row['checkpoint_id']}",
            )
        )
    files.append(
        _verify(
            manifest["license"]["path"],
            manifest["license"]["sha256"],
            role="vendor license",
        )
    )
    return {
        "repository": manifest["repository"],
        "checkout": str(checkout),
        "commit": commit,
        "clean": True,
        "files": files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run_dir.expanduser().resolve()
    output = run / "audit/protected_source_final_verification.json"
    if output.exists():
        raise FileExistsError(output)
    lock = verify_lock(run)
    repeated = _json(run / "audit/repeatedfilm_source_manifest.json")
    reference = _json(run / "audit/reference_baseline_inventory.json")
    preflight = _json(run / "audit/formal_input_reference_preflight.json")
    checks: dict[str, Any] = {
        "repeatedfilm_checkpoint": _verify(
            repeated["checkpoint_path"],
            repeated["checkpoint_sha256"],
            role="repeated-FiLM checkpoint",
        ),
        "repeatedfilm_config": _verify(
            repeated["config_path"],
            repeated["config_sha256"],
            role="repeated-FiLM config",
        ),
        "reference": {
            name: _verify(row["path"], row["sha256"], role=f"reference {name}")
            for name, row in reference["protected_key_files_before"].items()
        },
        "reference_direct_inputs": {
            name: _verify(row["path"], row["sha256"], role=f"R0 direct input {name}")
            for name, row in preflight["r0"]["direct_inputs"].items()
        },
        "vendors": {
            "grconvnet": _vendor(
                _json(run / "third_party/grconvnet_source_manifest.json")
            ),
            "ggcnn2": _vendor(
                _json(run / "third_party/ggcnn2_source_manifest.json")
            ),
        },
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "all_protected_sources_unchanged": True,
        "experiment_lock_sha256": lock["manifest_content_sha256"],
        "checks": checks,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
