#!/usr/bin/env python3
"""Compact verified per-sample GQ-CNN outputs into one GT-free Parquet table."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.geometric_ranker import load_frozen_candidates  # noqa: E402
from src.grasping.reranking_v1.identity import (  # noqa: E402
    candidate_identity_sha256,
    sha256_file,
)
from tools.modular_reranking.scene_sharding import (  # noqa: E402
    SCENE_SHARD_ASSIGNMENT,
    select_scene_shard_rows,
)


SCORE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int16()),
        ("pipeline", pa.string()),
        ("split", pa.string()),
        ("sample_index", pa.int32()),
        ("sample_id", pa.string()),
        ("question_index", pa.int32()),
        ("scene_id", pa.string()),
        ("candidate_id", pa.string()),
        ("candidate_identity_sha256", pa.string()),
        ("gqcnn_q_value", pa.float64()),
        ("gqcnn_rank", pa.int32()),
        ("source_candidate_index", pa.int32()),
        ("source_candidates_npz_sha256", pa.string()),
        ("scored_candidates_npz_sha256", pa.string()),
        ("model_name", pa.string()),
        ("model_commit", pa.string()),
        ("model_config_sha256", pa.string()),
    ]
)

FORBIDDEN_GT_NAMES = {
    "candidate_positive",
    "candidate_gt_iou",
    "angle_error",
    "correct_candidate_id",
    "first_valid_rank",
    "gt_grasp",
    "gt_mask",
    "j_at_1",
    "j_at_any",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--tmp-root",
        type=Path,
        required=True,
        help="Atomic temporary files below the current protected run's tmp/.",
    )
    parser.add_argument("--status-every", type=int, default=500)
    parser.add_argument(
        "--scene-shard-count",
        type=int,
        help=(
            "Compact one scene-grouped candidate shard using the exact "
            "Dex-Net generator assignment."
        ),
    )
    parser.add_argument("--scene-shard-index", type=int)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def _validate_write_scope(
    output_path: Path,
    tmp_root: Path,
    *,
    ephemeral_scene_shard: bool = False,
) -> tuple[Path, Path, Path]:
    output = output_path.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = _protected_run_root(output)
    if _protected_run_root(temporary) != run_root:
        raise ValueError("output-path and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    output_is_below_tmp = output != temporary and temporary in output.parents
    if ephemeral_scene_shard:
        if not output_is_below_tmp:
            raise ValueError(
                "scene-shard compact score must be stored below tmp-root"
            )
    elif output == temporary or output_is_below_tmp:
        raise ValueError("final score table cannot be stored below tmp-root")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    return output, temporary, run_root


def _quarantine_partial_final(
    output_path: Path, manifest_path: Path, tmp_root: Path
) -> list[str]:
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if not existing:
        return []
    if len(existing) == 2:
        raise FileExistsError(f"compact score artifact already exists: {output_path}")
    quarantine = (
        tmp_root
        / "orphaned_compact_gqcnn_scores"
        / f"{output_path.stem}.{uuid.uuid4().hex}"
    )
    quarantine.mkdir(parents=True)
    moved: list[str] = []
    for path in existing:
        destination = quarantine / path.name
        os.replace(path, destination)
        moved.append(str(destination))
    return moved


def validated_score_rows(
    *,
    sample: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    q_values: np.ndarray,
    ranks: np.ndarray,
    split: str,
    source_candidates_npz_sha256: str,
    scored_candidates_npz_sha256: str,
    model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate a frozen score vector and return compact GT-free rows."""

    sample_id = str(sample["sample_id"])
    record_ids = [str(record.get("candidate_id", "")) for record in records]
    score_ids = [str(candidate_id) for candidate_id in candidate_ids]
    count = len(records)
    if (
        not sample_id
        or count == 0
        or len(record_ids) != len(set(record_ids))
        or any(not candidate_id for candidate_id in record_ids)
    ):
        raise ValueError(f"{sample_id}: invalid non-empty candidate identity")
    if record_ids != score_ids:
        raise ValueError(f"{sample_id}: scored candidate order/IDs differ")
    q = np.asarray(q_values, dtype=np.float64)
    try:
        rank_numeric = np.asarray(ranks, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{sample_id}: ranks are not numeric") from error
    if (
        rank_numeric.shape != (count,)
        or not np.all(np.isfinite(rank_numeric))
        or not np.array_equal(rank_numeric, np.rint(rank_numeric))
    ):
        raise ValueError(f"{sample_id}: ranks are not finite integers")
    rank = rank_numeric.astype(np.int64)
    if q.shape != (count,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{sample_id}: q-values are not a finite N-vector")
    if sorted(rank.tolist()) != list(range(1, count + 1)):
        raise ValueError(f"{sample_id}: ranks are not a 1..N permutation")
    expected_order = sorted(
        range(count), key=lambda index: (-float(q[index]), record_ids[index])
    )
    expected_rank = np.empty(count, dtype=np.int64)
    for position, source_index in enumerate(expected_order, start=1):
        expected_rank[source_index] = position
    if not np.array_equal(rank, expected_rank):
        raise ValueError(
            f"{sample_id}: rank differs from q-desc/candidate-id tie break"
        )

    output: list[dict[str, Any]] = []
    for source_index, (record, candidate_id) in enumerate(
        zip(records, record_ids, strict=True)
    ):
        if str(record.get("sample_id")) != sample_id:
            raise ValueError(f"{sample_id}: candidate sample identity differs")
        if FORBIDDEN_GT_NAMES & {str(name).lower() for name in record}:
            raise ValueError(f"{sample_id}: GT-derived field found in candidate")
        output.append(
            {
                "schema_version": 1,
                "pipeline": "hierarchical_repeated_film",
                "split": split,
                "sample_index": int(sample["sample_index"]),
                "sample_id": sample_id,
                "question_index": int(sample["question_index"]),
                "scene_id": str(sample["scene_id"]),
                "candidate_id": candidate_id,
                "candidate_identity_sha256": candidate_identity_sha256(record),
                "gqcnn_q_value": float(q[source_index]),
                "gqcnn_rank": int(rank[source_index]),
                "source_candidate_index": source_index,
                "source_candidates_npz_sha256": source_candidates_npz_sha256,
                "scored_candidates_npz_sha256": scored_candidates_npz_sha256,
                "model_name": str(model["model_name"]),
                "model_commit": str(model["model_commit"]),
                "model_config_sha256": str(model["model_config_hash"]),
            }
        )
    return output


def _verify_root_statistics(
    *,
    scorer_config: Mapping[str, Any],
    statistics: Mapping[str, Any],
    samples: int,
    candidates: int,
) -> None:
    source = scorer_config.get("source_identity", {})
    checks = {
        "samples": samples,
        "candidate_count": candidates,
    }
    for name, expected in checks.items():
        if int(source.get(name, -1)) != int(expected):
            raise ValueError(f"scorer source identity {name} differs")
    expected_statistics = {
        "total_samples": samples,
        "terminal_samples": samples,
        "expected_candidates": candidates,
        "scored_candidates": candidates,
        "finite_q_values": candidates,
        "invalid_q_values": 0,
        "failed_samples": 0,
        "corrupt_committed_samples": 0,
    }
    for name, expected in expected_statistics.items():
        if int(statistics.get(name, -1)) != int(expected):
            raise ValueError(f"scorer run statistic {name} differs")


def _select_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    scene_shard_count: int | None,
    scene_shard_index: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if (scene_shard_count is None) != (scene_shard_index is None):
        raise ValueError(
            "--scene-shard-count and --scene-shard-index are required together"
        )
    if scene_shard_count is None:
        return rows, None
    assert scene_shard_index is not None
    if scene_shard_count <= 0:
        raise ValueError("--scene-shard-count must be positive")
    if not 0 <= int(scene_shard_index) < scene_shard_count:
        raise ValueError(
            "--scene-shard-index must be in [0, --scene-shard-count)"
        )
    selected = select_scene_shard_rows(
        rows,
        num_shards=scene_shard_count,
        shard_index=int(scene_shard_index),
    )
    if not selected:
        raise ValueError("scene shard selects no prediction samples")
    return selected, {
        "assignment": SCENE_SHARD_ASSIGNMENT,
        "num_shards": int(scene_shard_count),
        "shard_index": int(scene_shard_index),
        "scene_grouped": True,
    }


def _validate_candidate_scene_shard(
    *,
    candidate_root: Path,
    prediction_manifest: Path,
    split: str,
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    config_path = candidate_root / "run_config.json"
    config = _read_json(config_path)
    selection = config.get("selection")
    if (
        config.get("status") != "COMPLETED"
        or config.get("split") != split
        or config.get("prediction_manifest_sha256")
        != sha256_file(prediction_manifest)
        or not isinstance(selection, Mapping)
        or selection.get("scene_grouped") is not True
        or selection.get("limit") is not None
        or int(selection.get("num_shards", -1))
        != int(partition["num_shards"])
        or int(selection.get("shard_index", -1))
        != int(partition["shard_index"])
    ):
        raise ValueError(
            "candidate root does not match the requested complete scene shard"
        )
    return config


def _atomic_json(path: Path, value: Any, tmp_root: Path) -> None:
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.status_every <= 0:
        raise ValueError("--status-every must be positive")
    output_path, tmp_root, _ = _validate_write_scope(
        args.output_path,
        args.tmp_root,
        ephemeral_scene_shard=(
            args.scene_shard_count is not None
            or args.scene_shard_index is not None
        ),
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError(
            f"another GQ-CNN compaction holds the lock: {lock_path}"
        ) from error
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(
        json.dumps({"pid": os.getpid(), "output_path": str(output_path)}) + "\n"
    )
    lock_stream.flush()
    os.fsync(lock_stream.fileno())
    quarantined_partial_final = _quarantine_partial_final(
        output_path, manifest_path, tmp_root
    )

    try:
        prediction_manifest = args.prediction_manifest.expanduser().resolve()
        candidate_root = args.candidate_root.expanduser().resolve()
        scored_root = args.scored_root.expanduser().resolve()
        full_samples = _read_jsonl(prediction_manifest)
        full_sample_ids = [str(row["sample_id"]) for row in full_samples]
        full_sample_indices = [int(row["sample_index"]) for row in full_samples]
        if (
            len(full_sample_ids) != len(set(full_sample_ids))
            or full_sample_indices != list(range(len(full_samples)))
            or any(str(row.get("split")) != args.split for row in full_samples)
        ):
            raise ValueError(
                "prediction manifest has duplicate IDs, non-canonical indices, "
                "or a split that disagrees with --split"
            )
        samples, partition = _select_prediction_rows(
            full_samples,
            scene_shard_count=args.scene_shard_count,
            scene_shard_index=args.scene_shard_index,
        )
        sample_ids = [str(row["sample_id"]) for row in samples]
        candidate_config = (
            None
            if partition is None
            else _validate_candidate_scene_shard(
                candidate_root=candidate_root,
                prediction_manifest=prediction_manifest,
                split=args.split,
                partition=partition,
            )
        )
        candidate_run_manifest_path: Path | None = None
        if partition is not None:
            candidate_run_manifest_path = candidate_root / "run_manifest.jsonl"
            candidate_rows = _read_jsonl(candidate_run_manifest_path)
            candidate_ids = [str(row.get("sample_id", "")) for row in candidate_rows]
            if (
                candidate_ids != sample_ids
                or len(candidate_ids) != len(set(candidate_ids))
            ):
                raise ValueError(
                    "prediction/candidate scene-shard sample order differs"
                )
            for sample, candidate in zip(samples, candidate_rows, strict=True):
                sample_id = str(sample["sample_id"])
                if any(
                    observed != expected
                    for observed, expected in (
                        (
                            int(candidate.get("sample_index", -1)),
                            int(sample["sample_index"]),
                        ),
                        (
                            int(candidate.get("question_index", -1)),
                            int(sample["question_index"]),
                        ),
                        (
                            str(candidate.get("scene_id", "")),
                            str(sample.get("scene_id", "")),
                        ),
                        (
                            str(candidate.get("query", "")),
                            str(sample.get("query", "")),
                        ),
                    )
                ):
                    raise ValueError(
                        f"{sample_id}: prediction/candidate sample identity differs"
                    )
        source_manifest_path = scored_root / "source_candidate_manifest.jsonl"
        scoring_manifest_path = scored_root / "scoring_manifest.jsonl"
        source_rows = _read_jsonl(source_manifest_path)
        scoring_rows = _read_jsonl(scoring_manifest_path)
        source_ids_in_order = [
            str(row.get("sample_id", "")) for row in source_rows
        ]
        source_local_indices = [
            int(row.get("sample_index", -1)) for row in source_rows
        ]
        if (
            source_ids_in_order != sample_ids
            or source_local_indices != list(range(len(source_rows)))
        ):
            raise ValueError(
                "scorer source manifest order/local indices differ from "
                "the frozen prediction selection"
            )
        source_by_id = {str(row["sample_id"]): row for row in source_rows}
        scoring_by_id = {str(row["sample_id"]): row for row in scoring_rows}
        if (
            len(source_by_id) != len(source_rows)
            or len(scoring_by_id) != len(scoring_rows)
            or set(source_by_id) != set(sample_ids)
            or set(scoring_by_id) != set(sample_ids)
        ):
            raise ValueError("prediction/source/scoring sample universes differ")

        scorer_config = _read_json(scored_root / "run_config.json")
        statistics = _read_json(scored_root / "run_statistics.json")
        model = scorer_config.get("model")
        if not isinstance(model, dict):
            raise ValueError("scorer run config has no model identity")
        required_model = {"model_name", "model_commit", "model_config_hash"}
        if not required_model <= set(model):
            raise ValueError("scorer run config has incomplete model identity")
        declared_source_manifest_sha = scorer_config.get(
            "source_identity", {}
        ).get("source_manifest_sha256")
        if declared_source_manifest_sha != sha256_file(source_manifest_path):
            raise ValueError("scorer source manifest hash differs")
        total_candidates = sum(
            int(row["candidate_count"]) for row in source_rows
        )
        _verify_root_statistics(
            scorer_config=scorer_config,
            statistics=statistics,
            samples=len(samples),
            candidates=total_candidates,
        )
        independent_verification_sha256: str | None = None
        if partition is not None:
            assert candidate_config is not None
            verification_path = scored_root / "verification_report.json"
            verification = _read_json(verification_path)
            if (
                verification.get("clean") is not True
                or Path(
                    str(verification.get("candidate_root", ""))
                ).resolve()
                != candidate_root
                or Path(str(verification.get("scored_root", ""))).resolve()
                != scored_root
                or verification.get("source_manifest_sha256")
                != declared_source_manifest_sha
                or int(verification.get("expected_total_samples", -1))
                != len(samples)
                or int(
                    verification.get("expected_frozen_candidates", -1)
                )
                != total_candidates
                or int(candidate_config["counts"]["samples"]) != len(samples)
                or int(candidate_config["counts"]["nms_candidates"])
                != total_candidates
            ):
                raise ValueError(
                    "scene-shard compaction requires a matching clean "
                    "independent verification report"
                )
            independent_verification_sha256 = sha256_file(
                verification_path
            )

        temporary = tmp_root / f"{output_path.name}.{uuid.uuid4().hex}.tmp"
        writer = pq.ParquetWriter(
            temporary, SCORE_SCHEMA, compression="zstd", use_dictionary=True
        )
        buffer: list[dict[str, Any]] = []
        written = 0
        empty_samples = 0
        try:
            for position, sample in enumerate(samples, start=1):
                sample_id = str(sample["sample_id"])
                source = source_by_id[sample_id]
                scoring = scoring_by_id[sample_id]
                count = int(source["candidate_count"])
                if (
                    int(source.get("question_index", -1))
                    != int(sample["question_index"])
                    or str(source.get("query", "")) != str(sample.get("query", ""))
                ):
                    raise ValueError(
                        f"{sample_id}: prediction/source sample identity differs"
                    )
                source_hashes = source.get("source_hashes", {})
                source_npz = candidate_root / sample_id / "candidates.npz"
                source_json = candidate_root / sample_id / "candidates.json"
                source_npz_sha = sha256_file(source_npz)
                source_json_sha = sha256_file(source_json)
                if (
                    source_hashes.get("candidates_npz_sha256")
                    != source_npz_sha
                    or source_hashes.get("candidates_json_sha256")
                    != source_json_sha
                ):
                    raise ValueError(f"{sample_id}: source candidate hash differs")
                # candidates.npz is the numeric source of truth used by both
                # GQ-CNN and feature extraction. The loader also proves that
                # the JSON sidecar agrees after conversion to the frozen NPZ
                # dtypes. Hashing the raw JSON floats here would create a
                # second, precision-dependent candidate identity namespace.
                records, _, loaded_hashes = load_frozen_candidates(
                    source_npz, source_json
                )
                if (
                    loaded_hashes.get("candidates_npz_sha256")
                    != source_npz_sha
                    or loaded_hashes.get("candidates_json_sha256")
                    != source_json_sha
                ):
                    raise ValueError(
                        f"{sample_id}: validated candidate hashes differ"
                    )
                record_ids = [
                    str(record.get("candidate_id", "")) for record in records
                ]
                if (
                    len(records) != count
                    or record_ids != list(map(str, source["candidate_ids"]))
                ):
                    raise ValueError(
                        f"{sample_id}: source candidate JSON IDs/count differ"
                    )
                if (
                    int(scoring.get("source_candidate_count", -1)) != count
                    or int(scoring.get("gqcnn_scored_count", -1)) != count
                    or scoring.get("model_config_hash")
                    != model["model_config_hash"]
                    or scoring.get("source_candidate_sha256")
                    != source_npz_sha
                ):
                    raise ValueError(
                        f"{sample_id}: scorer manifest identity differs"
                    )
                marker_path = scored_root / sample_id / "_SCORING_COMPLETE.json"
                marker = _read_json(marker_path)
                expected_status = (
                    "scored_nonempty" if count else "skipped_valid_empty"
                )
                marker_identity = {
                    "sample_id": sample_id,
                    "scoring_status": expected_status,
                    "source_candidate_count": count,
                    "gqcnn_scored_count": count,
                    "source_candidate_sha256": source_npz_sha,
                    "source_candidate_json_sha256": source_json_sha,
                    "model_name": model["model_name"],
                    "model_commit": model["model_commit"],
                    "model_config_hash": model["model_config_hash"],
                }
                if scoring.get("scoring_status") != expected_status or any(
                    marker.get(name) != expected
                    for name, expected in marker_identity.items()
                ):
                    raise ValueError(
                        f"{sample_id}: terminal scorer marker differs"
                    )
                if count == 0:
                    empty_samples += 1
                    if (
                        scored_root
                        / sample_id
                        / "gqcnn_scored_candidates.npz"
                    ).exists():
                        raise ValueError(f"{sample_id}: empty sample has score NPZ")
                else:
                    score_path = (
                        scored_root / sample_id / "gqcnn_scored_candidates.npz"
                    )
                    score_sha = sha256_file(score_path)
                    if (
                        marker.get("required_file_hashes", {}).get(
                            "gqcnn_scored_candidates.npz"
                        )
                        != score_sha
                    ):
                        raise ValueError(
                            f"{sample_id}: score NPZ marker hash differs"
                        )
                    with np.load(score_path, allow_pickle=False) as archive:
                        required = {
                            "candidate_id",
                            "gqcnn_q_value",
                            "gqcnn_rank",
                        }
                        if not required <= set(archive.files):
                            raise ValueError(f"{sample_id}: score NPZ keys missing")
                        rows = validated_score_rows(
                            sample=sample,
                            records=records,
                            candidate_ids=[
                                str(item)
                                for item in archive["candidate_id"].tolist()
                            ],
                            q_values=np.asarray(archive["gqcnn_q_value"]),
                            ranks=np.asarray(archive["gqcnn_rank"]),
                            split=args.split,
                            source_candidates_npz_sha256=source_npz_sha,
                            scored_candidates_npz_sha256=score_sha,
                            model=model,
                        )
                    if len(rows) != count:
                        raise ValueError(
                            f"{sample_id}: compact score count differs"
                        )
                    buffer.extend(rows)
                    if len(buffer) >= 50_000:
                        writer.write_table(
                            pa.Table.from_pylist(buffer, schema=SCORE_SCHEMA)
                        )
                        written += len(buffer)
                        buffer.clear()
                if position % args.status_every == 0 or position == len(samples):
                    print(
                        f"GQ compact {position}/{len(samples)} "
                        f"rows={written + len(buffer)}",
                        flush=True,
                    )
            if buffer:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=SCORE_SCHEMA)
                )
                written += len(buffer)
                buffer.clear()
        finally:
            writer.close()

        parquet = pq.ParquetFile(temporary)
        if (
            parquet.schema_arrow != SCORE_SCHEMA
            or parquet.metadata.num_rows != written
        ):
            raise ValueError("compact GQ-CNN Parquet verification failed")
        if written != total_candidates:
            raise ValueError("compact GQ-CNN row count differs from source")
        table_sha = sha256_file(temporary)
        table_size = temporary.stat().st_size
        manifest = {
            "schema_version": 1,
            "status": "COMPLETED",
            "pipeline": "hierarchical_repeated_film",
            "split": args.split,
            "primary_key": ["sample_id", "candidate_id"],
            "row_order": "prediction_manifest_then_source_candidate_order",
            "compression": "zstd",
            "gt_free": True,
            "forbidden_gt_fields": sorted(FORBIDDEN_GT_NAMES),
            "samples": len(samples),
            "full_prediction_samples": len(full_samples),
            "nonempty_samples": len(samples) - empty_samples,
            "empty_samples": empty_samples,
            "rows": written,
            "prediction_manifest": str(prediction_manifest),
            "prediction_manifest_sha256": sha256_file(prediction_manifest),
            "candidate_root": str(candidate_root),
            "scored_root": str(scored_root),
            "candidate_run_manifest_sha256": (
                None
                if candidate_run_manifest_path is None
                else sha256_file(candidate_run_manifest_path)
            ),
            "source_candidate_manifest_sha256": sha256_file(
                source_manifest_path
            ),
            "scoring_manifest_sha256": sha256_file(scoring_manifest_path),
            "scorer_run_config_sha256": sha256_file(
                scored_root / "run_config.json"
            ),
            "scorer_run_statistics_sha256": sha256_file(
                scored_root / "run_statistics.json"
            ),
            "model": model,
            "partition": partition,
            "candidate_protocol_identity_sha256": (
                None
                if candidate_config is None
                else candidate_config.get("protocol_identity_sha256")
            ),
            "candidate_protocol_family_identity_sha256": (
                None
                if candidate_config is None
                else candidate_config.get(
                    "protocol_family_identity_sha256"
                )
            ),
            "independent_verification_sha256": (
                independent_verification_sha256
            ),
            "schema_sha256": _canonical_json_hash(
                [
                    (field.name, str(field.type), field.nullable)
                    for field in SCORE_SCHEMA
                ]
            ),
            "gqcnn_scores_parquet": str(output_path),
            "gqcnn_scores_parquet_sha256": table_sha,
            "gqcnn_scores_parquet_size_bytes": table_size,
            "recovered_partial_final_artifacts": quarantined_partial_final,
            "storage": {
                "ephemeral_scene_shard": partition is not None,
                "automatic_deletion_scope": (
                    "current_run_tmp_after_verified_canonical_merge"
                    if partition is not None
                    else None
                ),
            },
        }
        temporary_manifest = (
            tmp_root / f"{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        )
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
        os.replace(temporary_manifest, manifest_path)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
