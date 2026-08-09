#!/usr/bin/env python3
"""Create or verify deterministic full-file fingerprints of frozen sources."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--root requires NAME=PATH")
    name, raw = value.split("=", 1)
    path = Path(raw).expanduser().resolve()
    if not name or not path.is_dir():
        raise ValueError(f"invalid frozen root: {value}")
    return name, path


def root_manifest(name: str, root: Path, workers: int) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())

    def one(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": int(stat.st_size),
            "sha256": sha256_file(path),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(one, files))
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "name": name,
        "root": str(root),
        "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "root_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    roots = dict(parse_root(value) for value in args.root)
    if len(roots) != len(args.root):
        raise ValueError("frozen root names must be unique")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    manifest = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "roots": [
            root_manifest(name, path, args.workers)
            for name, path in sorted(roots.items())
        ],
    }
    if args.compare_to is not None:
        before_path = args.compare_to.expanduser().resolve()
        before = json.loads(before_path.read_text(encoding="utf-8"))
        before_hashes = {
            item["name"]: item["root_content_sha256"]
            for item in before["roots"]
        }
        after_hashes = {
            item["name"]: item["root_content_sha256"]
            for item in manifest["roots"]
        }
        manifest["comparison"] = {
            "before_manifest": str(before_path),
            "before_manifest_sha256": sha256_file(before_path),
            "root_hashes_before": before_hashes,
            "root_hashes_after": after_hashes,
            "all_frozen_sources_unchanged": before_hashes == after_hashes,
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "roots": {
                    item["name"]: item["root_content_sha256"]
                    for item in manifest["roots"]
                },
                "comparison": manifest.get("comparison"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if (
        args.compare_to is not None
        and not manifest["comparison"]["all_frozen_sources_unchanged"]
    ):
        raise RuntimeError("one or more frozen source roots changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
