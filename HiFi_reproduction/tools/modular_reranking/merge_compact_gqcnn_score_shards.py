#!/usr/bin/env python3
"""Merge verified scene-sharded compact GQ-CNN tables.

Each input must have been produced by ``compact_gqcnn_scores.py`` with an
explicit scene-shard contract.  The merge replays the frozen prediction
manifest, rejects partition/model/provenance drift, and restores canonical
prediction-manifest row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.modular_reranking.compact_gqcnn_scores import SCORE_SCHEMA  # noqa: E402
from tools.modular_reranking import low_peak_transaction as low_peak  # noqa: E402
from tools.modular_reranking.scene_sharding import (  # noqa: E402
    SCENE_SHARD_ASSIGNMENT,
    assigned_to_scene_shard,
    select_scene_shard_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument(
        "--shard-score", type=Path, action="append", required=True
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument(
        "--max-run-bytes",
        type=int,
        default=low_peak.DEFAULT_MAX_RUN_BYTES,
        help="Refuse publication when its conservative peak exceeds this budget.",
    )
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def validate_scopes(
    output_path: Path, tmp_root: Path, shard_scores: list[Path]
) -> tuple[Path, Path, Path, list[Path]]:
    output = output_path.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = protected_run_root(output)
    if protected_run_root(temporary) != run_root:
        raise ValueError("output-path and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    if output == temporary or temporary in output.parents:
        raise ValueError("merged score table must be outside tmp-root")
    resolved_scores = [path.expanduser().resolve() for path in shard_scores]
    if any(protected_run_root(path) != run_root for path in resolved_scores):
        raise ValueError("all score shards must belong to the same active run")
    if any(
        path == configured_tmp or configured_tmp not in path.parents
        for path in resolved_scores
    ):
        raise ValueError("scene-shard score inputs must be below run tmp/")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    return output, temporary, run_root, resolved_scores


def manifest_path_for(score_path: Path) -> Path:
    return score_path.with_suffix(".manifest.json")


def _compression_set(parquet: pq.ParquetFile) -> set[str]:
    return {
        str(
            parquet.metadata.row_group(row_group).column(column).compression
        ).lower()
        for row_group in range(parquet.metadata.num_row_groups)
        for column in range(parquet.metadata.num_columns)
    }


def _validate_rank_contract(table: pa.Table, *, source: Path) -> None:
    frame = table.select(
        ["sample_id", "candidate_id", "gqcnn_q_value", "gqcnn_rank"]
    ).to_pandas()
    if (
        frame[["sample_id", "candidate_id"]].duplicated().any()
        or not np.isfinite(frame["gqcnn_q_value"].to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"invalid score primary key or q-value: {source}")
    for sample_id, group in frame.groupby("sample_id", sort=False):
        count = len(group)
        ranks = group["gqcnn_rank"].to_numpy(dtype=int)
        if sorted(ranks.tolist()) != list(range(1, count + 1)):
            raise ValueError(f"{sample_id}: score ranks are not a permutation")
        expected = group.sort_values(
            ["gqcnn_q_value", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        )["candidate_id"].tolist()
        observed = group.sort_values(
            "gqcnn_rank", kind="mergesort"
        )["candidate_id"].tolist()
        if observed != expected:
            raise ValueError(f"{sample_id}: score rank/tie-break contract differs")


def main() -> int:
    args = parse_args()
    output_path, tmp_root, run_root, shard_scores = validate_scopes(
        args.output_path, args.tmp_root, args.shard_score
    )
    output_manifest_path = manifest_path_for(output_path)
    if output_path.exists() or output_manifest_path.exists():
        raise FileExistsError(f"merged score artifact already exists: {output_path}")
    if len(shard_scores) < 2:
        raise ValueError("at least two score shards are required")

    prediction_manifest = args.prediction_manifest.expanduser().resolve()
    prediction_sha = sha256_file(prediction_manifest)
    prediction_rows = read_jsonl(prediction_manifest)
    sample_ids = [str(row["sample_id"]) for row in prediction_rows]
    sample_indices = [int(row["sample_index"]) for row in prediction_rows]
    if (
        len(sample_ids) != len(set(sample_ids))
        or sample_indices != list(range(len(prediction_rows)))
        or any(str(row.get("split")) != args.split for row in prediction_rows)
    ):
        raise ValueError("prediction manifest identity is invalid")
    prediction_by_id = {
        str(row["sample_id"]): row for row in prediction_rows
    }

    shard_items: list[tuple[Path, Path, dict[str, Any]]] = []
    expected_schema_sha256 = canonical_json_sha256(
        [
            (field.name, str(field.type), field.nullable)
            for field in SCORE_SCHEMA
        ]
    )
    for score_path in shard_scores:
        manifest_path = manifest_path_for(score_path)
        manifest = read_json(manifest_path)
        partition = manifest.get("partition")
        if (
            manifest.get("status") != "COMPLETED"
            or manifest.get("split") != args.split
            or manifest.get("gt_free") is not True
            or manifest.get("primary_key") != ["sample_id", "candidate_id"]
            or manifest.get("prediction_manifest_sha256") != prediction_sha
            or int(manifest.get("full_prediction_samples", -1))
            != len(prediction_rows)
            or manifest.get("schema_sha256") != expected_schema_sha256
            or not isinstance(
                manifest.get("independent_verification_sha256"), str
            )
            or len(manifest["independent_verification_sha256"]) != 64
            or not isinstance(manifest.get("storage"), Mapping)
            or manifest["storage"].get("ephemeral_scene_shard") is not True
            or not isinstance(partition, Mapping)
            or partition.get("assignment") != SCENE_SHARD_ASSIGNMENT
            or partition.get("scene_grouped") is not True
        ):
            raise ValueError(f"invalid scene-shard score manifest: {manifest_path}")
        if (
            Path(str(manifest.get("gqcnn_scores_parquet", ""))).resolve()
            != score_path
            or manifest.get("gqcnn_scores_parquet_sha256")
            != sha256_file(score_path)
        ):
            raise ValueError(f"score shard hash/path mismatch: {score_path}")
        shard_items.append((score_path, manifest_path, manifest))

    shard_items.sort(
        key=lambda item: int(item[2]["partition"]["shard_index"])
    )
    shard_count_values = {
        int(item[2]["partition"]["num_shards"]) for item in shard_items
    }
    shard_indices = [
        int(item[2]["partition"]["shard_index"]) for item in shard_items
    ]
    if shard_count_values != {len(shard_items)} or shard_indices != list(
        range(len(shard_items))
    ):
        raise ValueError("score shard set is incomplete or duplicated")
    num_shards = len(shard_items)
    common_keys = (
        "model",
        "schema_sha256",
        "candidate_protocol_family_identity_sha256",
    )
    for key in common_keys:
        values = {
            json.dumps(item[2].get(key), sort_keys=True, ensure_ascii=False)
            for item in shard_items
        }
        if len(values) != 1:
            raise ValueError(f"score shard contract mismatch for {key}")
    protocol_identities = [
        item[2].get("candidate_protocol_identity_sha256")
        for item in shard_items
    ]
    if (
        any(not isinstance(value, str) or len(value) != 64 for value in protocol_identities)
        or len(set(protocol_identities)) != len(protocol_identities)
    ):
        raise ValueError("score shard candidate protocol identities are invalid")

    budget_preflight = low_peak.enforce_budget_preflight(
        run_root=run_root,
        input_paths=[item[0] for item in shard_items],
        max_run_bytes=int(args.max_run_bytes),
    )

    tables: list[pa.Table] = []
    source_shards: list[dict[str, Any]] = []
    all_keys: set[tuple[str, str]] = set()
    observed_sample_ids: set[str] = set()
    empty_samples = 0
    for score_path, manifest_path, manifest in shard_items:
        partition = manifest["partition"]
        shard_index = int(partition["shard_index"])
        expected_rows = select_scene_shard_rows(
            prediction_rows,
            num_shards=num_shards,
            shard_index=shard_index,
        )
        expected_ids = {str(row["sample_id"]) for row in expected_rows}
        if int(manifest.get("samples", -1)) != len(expected_ids):
            raise ValueError(
                f"score shard {shard_index} sample accounting differs"
            )
        parquet = pq.ParquetFile(score_path)
        expected_compression = (
            {"zstd"} if int(manifest.get("rows", -1)) else set()
        )
        if (
            parquet.schema_arrow != SCORE_SCHEMA
            or int(parquet.metadata.num_rows) != int(manifest.get("rows", -1))
            or _compression_set(parquet) != expected_compression
        ):
            raise ValueError(f"score shard Parquet contract differs: {score_path}")
        table = parquet.read()
        _validate_rank_contract(table, source=score_path)
        model = manifest["model"]
        model_rows = table.select(
            ["model_name", "model_commit", "model_config_sha256"]
        ).to_pandas()
        expected_model_row = (
            str(model["model_name"]),
            str(model["model_commit"]),
            str(model["model_config_hash"]),
        )
        if any(
            tuple(map(str, row)) != expected_model_row
            for row in model_rows.itertuples(index=False, name=None)
        ):
            raise ValueError(f"score rows/model manifest differ: {score_path}")
        frame = table.select(
            [
                "sample_index",
                "sample_id",
                "question_index",
                "scene_id",
                "candidate_id",
            ]
        ).to_pandas()
        row_sample_ids = set(map(str, frame["sample_id"]))
        if not row_sample_ids <= expected_ids:
            raise ValueError(f"score rows escape scene shard {shard_index}")
        if len(row_sample_ids) != int(manifest.get("nonempty_samples", -1)):
            raise ValueError(f"score shard {shard_index} nonempty count differs")
        expected_empty = len(expected_ids) - len(row_sample_ids)
        if expected_empty != int(manifest.get("empty_samples", -1)):
            raise ValueError(f"score shard {shard_index} empty count differs")
        empty_samples += expected_empty
        for row in frame.itertuples(index=False):
            sample_id = str(row.sample_id)
            expected = prediction_by_id[sample_id]
            if (
                int(row.sample_index) != int(expected["sample_index"])
                or int(row.question_index) != int(expected["question_index"])
                or str(row.scene_id) != str(expected["scene_id"])
                or not assigned_to_scene_shard(
                    str(row.scene_id),
                    num_shards=num_shards,
                    shard_index=shard_index,
                )
            ):
                raise ValueError(f"{sample_id}: score/prediction identity differs")
            primary_key = (sample_id, str(row.candidate_id))
            if primary_key in all_keys:
                raise ValueError(
                    f"duplicate score primary key: {primary_key}"
                )
            all_keys.add(primary_key)
        observed_sample_ids.update(row_sample_ids)
        tables.append(table)
        source_shards.append(
            {
                "shard_index": shard_index,
                "score_path": str(score_path),
                "score_sha256": manifest["gqcnn_scores_parquet_sha256"],
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "samples": manifest["samples"],
                "rows": manifest["rows"],
                "candidate_protocol_identity_sha256": manifest.get(
                    "candidate_protocol_identity_sha256"
                ),
            }
        )

    if len(observed_sample_ids) + empty_samples != len(prediction_rows):
        raise ValueError("merged score sample accounting differs")
    merged = pa.concat_tables(tables).sort_by(
        [
            ("sample_index", "ascending"),
            ("source_candidate_index", "ascending"),
        ]
    )
    if merged.schema != SCORE_SCHEMA or len(merged) != len(all_keys):
        raise ValueError("merged score schema or primary-key accounting differs")
    temporary = tmp_root / f"{output_path.name}.{uuid.uuid4().hex}.tmp"
    temporary_manifest = (
        tmp_root / f"{output_manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        pq.write_table(
            merged,
            temporary,
            compression="zstd",
            use_dictionary=True,
            row_group_size=50_000,
        )
        parquet = pq.ParquetFile(temporary)
        expected_merged_compression = {"zstd"} if len(merged) else set()
        if (
            parquet.schema_arrow != SCORE_SCHEMA
            or parquet.metadata.num_rows != len(merged)
            or _compression_set(parquet) != expected_merged_compression
        ):
            raise ValueError("merged compact score Parquet verification failed")
        manifest = {
            "schema_version": 1,
            "status": "COMPLETED",
            "pipeline": "hierarchical_repeated_film",
            "split": args.split,
            "primary_key": ["sample_id", "candidate_id"],
            "row_order": (
                "prediction_manifest_then_source_candidate_index"
            ),
            "compression": "zstd",
            "gt_free": True,
            "partition_merge": {
                "assignment": SCENE_SHARD_ASSIGNMENT,
                "num_shards": num_shards,
                "complete_shard_set": True,
            },
            "samples": len(prediction_rows),
            "nonempty_samples": len(observed_sample_ids),
            "empty_samples": empty_samples,
            "rows": len(merged),
            "prediction_manifest": str(prediction_manifest),
            "prediction_manifest_sha256": prediction_sha,
            "model": shard_items[0][2]["model"],
            "candidate_protocol_family_identity_sha256": shard_items[0][
                2
            ].get("candidate_protocol_family_identity_sha256"),
            "schema_sha256": shard_items[0][2]["schema_sha256"],
            "gqcnn_scores_parquet": str(output_path),
            "gqcnn_scores_parquet_sha256": sha256_file(temporary),
            "gqcnn_scores_parquet_size_bytes": temporary.stat().st_size,
            "run_root": str(run_root),
            "source_shards": source_shards,
            "storage_budget_preflight": budget_preflight,
        }
        temporary_manifest.write_text(
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
        os.replace(temporary, output_path)
        os.replace(temporary_manifest, output_manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
