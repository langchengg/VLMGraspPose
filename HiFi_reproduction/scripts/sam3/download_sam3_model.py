#!/usr/bin/env python3
"""Securely download and verify the gated official Transformers SAM 3 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError


REQUIRED = (
    "config.json",
    "processor_config.json",
    "model.safetensors",
)
ALLOW_PATTERNS = (
    "config.json",
    "processor_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="facebook/sam3")
    parser.add_argument("--revision")
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def _error_message(repo_id: str) -> str:
    return (
        f"Access to gated model {repo_id!r} was denied. Visit "
        f"https://huggingface.co/{repo_id}, accept the official conditions, then run "
        "`hf auth login` or set HF_TOKEN in the environment. The token is never accepted as a "
        "command-line argument or written to the manifest."
    )


def main() -> int:
    args = parse_args()
    local_dir = args.local_dir.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else local_dir.parent / f"{local_dir.name}.download_manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite download manifest: {manifest_path}")
    expected: dict[str, dict[str, object]] = {}
    try:
        if args.local_files_only:
            if not args.revision or not SHA_PATTERN.fullmatch(args.revision):
                raise ValueError("--local-files-only requires an explicit immutable 40-character revision")
            resolved_revision = args.revision
        else:
            info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True)
            resolved_revision = str(info.sha)
            if not SHA_PATTERN.fullmatch(resolved_revision):
                raise RuntimeError(f"Hub did not resolve an immutable commit SHA: {resolved_revision}")
            for sibling in info.siblings:
                name = str(sibling.rfilename)
                if name not in ALLOW_PATTERNS:
                    continue
                lfs = getattr(sibling, "lfs", None)
                expected[name] = {
                    "hub_size": getattr(sibling, "size", None),
                    "hub_lfs_sha256": None if lfs is None else getattr(lfs, "sha256", None),
                }
            # Small authorized read fails before starting the multi-gigabyte weight transfer.
            hf_hub_download(args.repo_id, "config.json", revision=resolved_revision)
        snapshot = Path(
            snapshot_download(
                repo_id=args.repo_id,
                revision=resolved_revision,
                local_dir=local_dir,
                local_files_only=bool(args.local_files_only),
                allow_patterns=list(ALLOW_PATTERNS),
            )
        ).resolve()
    except GatedRepoError as error:
        raise SystemExit(_error_message(args.repo_id)) from error
    except LocalEntryNotFoundError as error:
        raise SystemExit(
            "The pinned SAM 3 snapshot is not complete in the local Hugging Face cache. "
            "Run once without --local-files-only on an authorized networked machine."
        ) from error
    missing = [name for name in REQUIRED if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"official Transformers snapshot is incomplete; missing {missing}")
    files: dict[str, dict[str, object]] = {}
    for name in ALLOW_PATTERNS:
        path = snapshot / name
        if not path.is_file():
            continue
        actual_hash = sha256_file(path)
        hub = expected.get(name, {})
        expected_hash = hub.get("hub_lfs_sha256")
        expected_size = hub.get("hub_size")
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise RuntimeError(f"size mismatch for {name}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": actual_hash,
            "hub_lfs_sha256": expected_hash,
        }
    payload = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "snapshot": str(snapshot),
        "local_files_only": bool(args.local_files_only),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "authentication": "implicit HF_TOKEN or Hugging Face CLI credential; secret not recorded",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "VERIFIED", "revision": resolved_revision, "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
