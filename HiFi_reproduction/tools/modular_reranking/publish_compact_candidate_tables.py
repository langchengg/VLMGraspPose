#!/usr/bin/env python3
"""Atomically publish compact candidate Parquets from protected run tmp.

The verbose per-sample candidate tree remains temporary.  Only the three
verified, ZSTD-compressed Parquet tables are hard-linked into persistent
storage, so publication does not duplicate their data blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq


STAGE_FILES = {
    "raw": "raw_candidates.parquet",
    "mask_validated": "mask_validated_candidates.parquet",
    "nms": "nms_candidates.parquet",
}
FORBIDDEN_GT_COLUMNS = {
    "angle_error",
    "candidate_gt_angle_error",
    "candidate_gt_iou",
    "candidate_positive",
    "correct_candidate_id",
    "first_valid_rank",
    "ground_truth",
    "gt_grasp",
    "gt_mask",
    "j_at_1",
    "j_at_any",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def validate_scopes(
    source_root: Path, output_root: Path, tmp_root: Path
) -> tuple[Path, Path, Path, Path]:
    source = source_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    source_run = protected_run_root(source)
    if (
        protected_run_root(output) != source_run
        or protected_run_root(temporary) != source_run
    ):
        raise ValueError("source, output, and tmp roots must share one active run")
    configured_tmp = (source_run / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    if source == temporary or temporary not in source.parents:
        raise ValueError("source-root must be below the current run tmp-root")
    if output == temporary or temporary in output.parents:
        raise ValueError("persistent output-root must be outside tmp-root")
    return source, output, temporary, source_run


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_source(
    source_root: Path, *, split: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config_path = source_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json_object(config_path)
    if config.get("status") != "COMPLETED" or config.get("split") != split:
        raise ValueError("candidate source is not a completed matching split")
    if int(config.get("counts", {}).get("execution_failures", -1)) != 0:
        raise ValueError("candidate source records execution failures")
    registered = config.get("candidate_stage_artifacts")
    if not isinstance(registered, Mapping):
        raise ValueError("candidate source omits stage artifacts")

    verified: dict[str, dict[str, Any]] = {}
    for stage, filename in STAGE_FILES.items():
        artifact = registered.get(stage)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"candidate source omits {stage} artifact")
        path = (source_root / filename).resolve()
        if (
            not path.is_file()
            or Path(str(artifact.get("path", ""))).resolve() != path
        ):
            raise ValueError(f"{stage} artifact path is invalid")
        digest = sha256_file(path)
        if digest != artifact.get("sha256"):
            raise ValueError(f"{stage} artifact hash is invalid")
        parquet = pq.ParquetFile(path)
        columns = list(map(str, parquet.schema_arrow.names))
        compressions = {
            str(
                parquet.metadata.row_group(row_group)
                .column(column)
                .compression
            ).lower()
            for row_group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.num_columns)
        }
        if compressions != {"zstd"}:
            raise ValueError(
                f"{stage} table is not uniformly ZSTD-compressed: "
                f"{sorted(compressions)}"
            )
        forbidden = sorted(
            set(map(str.lower, columns)) & FORBIDDEN_GT_COLUMNS
        )
        if forbidden:
            raise ValueError(f"{stage} table exposes GT columns: {forbidden}")
        rows = int(parquet.metadata.num_rows)
        if (
            rows != int(artifact.get("rows", -1))
            or list(artifact.get("primary_key", []))
            != ["sample_id", "candidate_id"]
            or not {"sample_id", "candidate_id"} <= set(columns)
        ):
            raise ValueError(f"{stage} artifact identity is invalid")
        verified[stage] = {
            "filename": filename,
            "sha256": digest,
            "rows": rows,
            "size_bytes": path.stat().st_size,
            "schema_sha256": canonical_json_sha256(
                parquet.schema_arrow.to_string()
            ),
            "columns": columns,
        }
    return config, verified


def main() -> int:
    args = parse_args()
    source, output, tmp_root, run_root = validate_scopes(
        args.source_root, args.output_root, args.tmp_root
    )
    if output.exists():
        raise FileExistsError(f"persistent candidate tables already exist: {output}")
    config, verified = validate_source(source, split=args.split)

    staging = (
        tmp_root
        / "atomic_candidate_table_publication"
        / f"{args.split}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(parents=True)
    published_artifacts: dict[str, dict[str, Any]] = {}
    for stage, artifact in verified.items():
        source_path = source / str(artifact["filename"])
        staging_path = staging / str(artifact["filename"])
        os.link(source_path, staging_path)
        source_stat = source_path.stat()
        staging_stat = staging_path.stat()
        if (
            source_stat.st_dev != staging_stat.st_dev
            or source_stat.st_ino != staging_stat.st_ino
            or sha256_file(staging_path) != artifact["sha256"]
        ):
            raise AssertionError(f"{stage} publication is not an exact hardlink")
        published_artifacts[stage] = {
            **artifact,
            "path": str((output / str(artifact["filename"])).resolve()),
            "primary_key": ["sample_id", "candidate_id"],
            "compression": "zstd",
        }

    source_config_path = source / "run_config.json"
    manifest = {
        "schema_version": 1,
        "pipeline": "hierarchical_repeated_film",
        "split": args.split,
        "status": "COMPLETED",
        "gt_free": True,
        "source_root": str(source),
        "source_run_config": str(source_config_path.resolve()),
        "source_run_config_sha256": sha256_file(source_config_path),
        "source_protocol_identity_sha256": config.get(
            "protocol_identity_sha256"
        ),
        "run_root": str(run_root),
        "storage": {
            "persistent_tables_hardlinked_from_tmp": True,
            "verbose_per_sample_tree_persistent": False,
            "data_blocks_duplicated_by_publication": False,
        },
        "artifacts": published_artifacts,
    }
    manifest_path = staging / "candidate_tables.manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (staging / "run_command.txt").write_text(
        " ".join(map(shlex.quote, [sys.executable, *sys.argv])) + "\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output)

    for stage, artifact in published_artifacts.items():
        path = Path(str(artifact["path"]))
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise AssertionError(f"{stage} persistent publication verification failed")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
