#!/usr/bin/env python3
"""Export the exact local-VLM prompt/schema contract into a protected run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.grasping.reranking_v1.local_vlm as local_vlm_module  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
)
from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.local_vlm import (  # noqa: E402
    ALLOWED_METADATA_FIELDS,
    FORBIDDEN_GT_KEY_FRAGMENTS,
    REASON_CODES,
    SYSTEM_PROMPT,
    ranking_json_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    return parser.parse_args()


def _protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def _validate_scopes(
    output_root: Path, tmp_root: Path
) -> tuple[Path, Path, Path]:
    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = _protected_run_root(output)
    if _protected_run_root(temporary) != run_root:
        raise ValueError("output-root and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    if output == temporary or temporary in output.parents:
        raise ValueError("persistent VLM contract must be outside run tmp")
    return output, temporary, run_root


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def export_contract(
    *, output_root: Path, tmp_root: Path
) -> dict[str, Any]:
    output, temporary, run_root = _validate_scopes(output_root, tmp_root)
    if output.exists():
        raise FileExistsError(f"local VLM contract already exists: {output}")
    temporary.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = temporary / f".vlm-contract-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        prompt_path = staging / "system_prompt.txt"
        schema_path = staging / "ranking_json_schema.json"
        prompt = SYSTEM_PROMPT.rstrip("\n") + "\n"
        schema = ranking_json_schema()
        _write_text(prompt_path, prompt)
        _write_json(schema_path, schema)

        source_path = Path(local_vlm_module.__file__).resolve()
        manifest = {
            **identity_payload(),
            "schema_version": 1,
            "contract_kind": "local_vlm_prompt_and_json_schema",
            "run_root": str(run_root),
            "local_only_required": True,
            "candidate_pool": "frozen_repeatedfilm_gqcnn_top5",
            "candidate_generation_or_geometry_modification_allowed": False,
            "ground_truth_allowed": False,
            "system_prompt": {
                "path": str(output / prompt_path.name),
                "sha256": sha256_file(prompt_path),
                "bytes": prompt_path.stat().st_size,
            },
            "ranking_json_schema": {
                "path": str(output / schema_path.name),
                "sha256": sha256_file(schema_path),
                "bytes": schema_path.stat().st_size,
            },
            "prompt_builder_source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "reason_codes": list(REASON_CODES),
            "allowed_metadata_fields": list(ALLOWED_METADATA_FIELDS),
            "forbidden_gt_key_fragments": list(FORBIDDEN_GT_KEY_FRAGMENTS),
            "dynamic_user_prompt_binding": (
                "Every request hash binds sample_id, instruction, ordered frozen "
                "candidate IDs, image hashes, optional allowlisted metadata, "
                "generation settings, and input recipe SHA-256."
            ),
        }
        _write_json(staging / "contract_manifest.json", manifest)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    args = parse_args()
    manifest = export_contract(
        output_root=args.output_root,
        tmp_root=args.tmp_root,
    )
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
