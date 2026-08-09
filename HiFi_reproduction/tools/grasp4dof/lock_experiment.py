#!/usr/bin/env python3
"""Build, create, or verify an immutable 4-DoF experiment lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMMON_SOURCE_ROOT = REPOSITORY_ROOT / "src/grasping/common"
sys.path.insert(0, str(COMMON_SOURCE_ROOT))

from experiment_lock import (  # noqa: E402
    DEFAULT_FROZEN_ALIAS_NAME,
    DEFAULT_LOCK_RELATIVE_PATH,
    DEFAULT_MARKER_NAME,
    build_lock_candidate,
    verify_lock,
    write_lock_exclusive,
)


def _load_config(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--config must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="candidate JSON; required when creating or dry-running a lock",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print a non-effective candidate and write nothing",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing effective lock and all locked inputs",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if args.verify:
        if args.config is not None:
            parser.error("--config cannot be combined with --verify")
        manifest = verify_lock(run_dir)
        print(
            json.dumps(
                {
                    "verified": True,
                    "run_id": manifest["run_id"],
                    "manifest_content_sha256": manifest["manifest_content_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.config is None:
        parser.error("--config is required unless --verify is used")

    candidate = build_lock_candidate(
        _load_config(args.config),
        run_dir=run_dir,
        repository_root=REPOSITORY_ROOT,
    )
    if args.dry_run:
        print(json.dumps(candidate, indent=2, sort_keys=True, ensure_ascii=False))
        print(
            "DRY RUN: candidate is non-effective; no lock, marker, or directory was written",
            file=sys.stderr,
        )
        return 0

    manifest = write_lock_exclusive(run_dir, candidate)
    print(
        json.dumps(
            {
                "lock": str(run_dir / DEFAULT_LOCK_RELATIVE_PATH),
                "marker": str(run_dir / DEFAULT_MARKER_NAME),
                "frozen_manifest": str(run_dir / DEFAULT_FROZEN_ALIAS_NAME),
                "manifest_content_sha256": manifest["manifest_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
