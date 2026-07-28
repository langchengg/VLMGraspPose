#!/usr/bin/env python3
"""Securely download one pinned official Transformers SAM 3 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import huggingface_hub
import transformers
from huggingface_hub import HfApi, snapshot_download

from check_hf_access import check_access


REQUIRED_FILES = {
    "config.json",
    "model.safetensors",
    "processor_config.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/huggingface/facebook-sam3"),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    model_root = args.model_root.expanduser().resolve()
    if args.local_files_only:
        manifest_path = model_root / "model_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "local-files-only requires an existing verified model_manifest.json"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        destination = Path(manifest["local_model_path"]).expanduser().resolve()
        for record in manifest["files"]:
            path = destination / record["path"]
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or _sha256(path) != record["sha256"]
            ):
                raise RuntimeError(f"local snapshot verification failed: {record['path']}")
        print(
            json.dumps(
                {
                    "status": "LOCAL_SNAPSHOT_VERIFIED",
                    "model_id": manifest["model_id"],
                    "revision": manifest["resolved_revision_sha"],
                    "path": str(destination),
                    "files": len(manifest["files"]),
                },
                sort_keys=True,
            )
        )
        return 0
    access, status = check_access(args.model_id)
    if status:
        print(json.dumps(access, indent=2, sort_keys=True))
        return status
    api = HfApi()
    info = api.model_info(args.model_id, files_metadata=True)
    revision = str(info.sha)
    if len(revision) != 40:
        raise RuntimeError("Hub did not resolve an immutable commit revision")
    repository_files = sorted(item.rfilename for item in info.siblings)
    allowlist = [
        name
        for name in repository_files
        if name != "sam3.pt" and not name.startswith(".")
    ]
    if not REQUIRED_FILES.issubset(allowlist):
        raise RuntimeError(
            f"official repository is missing required Transformers files: "
            f"{sorted(REQUIRED_FILES - set(allowlist))}"
        )
    destination = model_root / revision
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        revision=revision,
        local_dir=destination,
        cache_dir=args.cache_dir,
        allow_patterns=allowlist,
        local_files_only=False,
        token=True,
    )
    files = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        if ".cache" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(destination)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    present = {item["path"] for item in files}
    missing = sorted(REQUIRED_FILES - present)
    if missing:
        raise RuntimeError(f"downloaded snapshot is incomplete: {missing}")
    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "resolved_revision_sha": revision,
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "local_model_path": str(destination),
        "platform": platform.platform(),
        "transformers_version": transformers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
        "download_allowlist": allowlist,
        "excluded_files": ["sam3.pt"],
        "files": files,
    }
    model_root.mkdir(parents=True, exist_ok=True)
    manifest_path = model_root / "model_manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="model_manifest.", suffix=".tmp", dir=model_root
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, manifest_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    print(
        json.dumps(
            {
                "status": "DOWNLOADED",
                "model_id": args.model_id,
                "revision": revision,
                "path": str(destination),
                "files": len(files),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
