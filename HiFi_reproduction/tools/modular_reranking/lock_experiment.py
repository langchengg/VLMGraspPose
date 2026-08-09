#!/usr/bin/env python3
"""Create or dry-run the immutable pre-formal-test experiment lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.experiment_lock import (  # noqa: E402
    build_lock_manifest,
    write_lock_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.expanduser().resolve().read_text(encoding="utf-8"))
    manifest = build_lock_manifest(spec, repo_root=REPO_ROOT)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        print("DRY RUN: no lock or output directory was written", file=sys.stderr)
        return 0
    write_lock_exclusive(args.output, manifest)
    print(
        json.dumps(
            {
                "lock": str(args.output.expanduser().resolve()),
                "manifest_content_sha256": manifest["manifest_content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
