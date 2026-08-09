#!/usr/bin/env python3
"""Safely prune one verified scene shard's verbose candidate/score trees.

Only per-sample directories plus ``_scene_cache`` and ``_labels`` are eligible.
The command first proves that compact scores and extracted features preserve
the candidate primary key, identity, q-value, and rank.  Dry-run is the
default; ``--execute`` is confined to the current active run's ``tmp/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.modular_reranking.scene_sharding import (  # noqa: E402
    SCENE_SHARD_ASSIGNMENT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--compact-score", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _is_below(path: Path, parent: Path) -> bool:
    return path != parent and parent in path.parents


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_scopes(
    *,
    candidate_root: Path,
    scored_root: Path,
    compact_score: Path,
    feature_root: Path,
    receipt_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    candidate = candidate_root.expanduser().resolve()
    scored = scored_root.expanduser().resolve()
    compact = compact_score.expanduser().resolve()
    features = feature_root.expanduser().resolve()
    receipt = receipt_path.expanduser().resolve()
    run_root = protected_run_root(candidate)
    paths = (scored, compact, features, receipt)
    if any(protected_run_root(path) != run_root for path in paths):
        raise ValueError("all streaming artifacts must belong to one active run")
    tmp_root = (run_root / "tmp").resolve()
    if not _is_below(candidate, tmp_root) or not _is_below(scored, tmp_root):
        raise ValueError("candidate-root and scored-root must be below run tmp/")
    if receipt == tmp_root or tmp_root in receipt.parents:
        raise ValueError("receipt-path must be persistent outside run tmp/")
    protected_artifacts = (candidate, scored, compact, features)
    if any(
        _paths_overlap(left, right)
        for index, left in enumerate(protected_artifacts)
        for right in protected_artifacts[index + 1 :]
    ):
        raise ValueError(
            "candidate-root, scored-root, compact-score, and feature-root "
            "must be mutually disjoint"
        )
    return candidate, scored, compact, features, receipt, run_root, tmp_root


def compact_manifest_path(score_path: Path) -> Path:
    return score_path.with_suffix(".manifest.json")


def _validate_partition(
    partition: Mapping[str, Any],
    *,
    num_shards: int,
    shard_index: int,
) -> None:
    if (
        partition.get("assignment") != SCENE_SHARD_ASSIGNMENT
        or partition.get("scene_grouped") is not True
        or int(partition.get("num_shards", -1)) != num_shards
        or int(partition.get("shard_index", -1)) != shard_index
    ):
        raise ValueError("scene-shard partition identity differs")


def _validate_feature_score_semantics(
    *,
    compact_score: Path,
    feature_root: Path,
) -> tuple[int, int]:
    score = pq.read_table(
        compact_score,
        columns=[
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "gqcnn_q_value",
            "gqcnn_rank",
        ],
    ).to_pandas()
    features = pd.read_parquet(
        feature_root / "per_candidate.parquet",
        columns=[
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "q_raw",
            "original_gqcnn_rank",
        ],
    )
    score = score.sort_values(
        ["sample_id", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    features = features.sort_values(
        ["sample_id", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        score[["sample_id", "candidate_id"]].duplicated().any()
        or features[["sample_id", "candidate_id"]].duplicated().any()
        or score[["sample_id", "candidate_id"]].to_dict("records")
        != features[["sample_id", "candidate_id"]].to_dict("records")
        or score["candidate_identity_sha256"].tolist()
        != features["candidate_identity_sha256"].tolist()
        or score["gqcnn_rank"].astype(int).tolist()
        != features["original_gqcnn_rank"].astype(int).tolist()
        or not np.array_equal(
            score["gqcnn_q_value"].to_numpy(dtype=np.float64),
            features["q_raw"].to_numpy(dtype=np.float64),
        )
    ):
        raise ValueError("compact scores and extracted features differ semantically")
    samples = pd.read_parquet(feature_root / "per_sample.parquet")
    if samples["sample_id"].duplicated().any():
        raise ValueError("feature per-sample table contains duplicate IDs")
    return len(score), len(samples)


def _directory_file_manifest_sha256(path: Path) -> tuple[str, int]:
    entries = list(path.iterdir())
    files = sorted(item for item in entries if item.is_file())
    if (
        len(files) != len(entries)
        or any(item.suffix != ".json" for item in files)
        or not files
    ):
        raise ValueError(
            f"label directory must contain only nonempty JSON files: {path}"
        )
    payload = [
        {
            "relative_path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return digest, len(files)


def _validate_oracle_artifacts(
    *,
    split: str,
    candidate_root: Path,
    feature_root: Path,
    candidate_config: Mapping[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_path = candidate_root / "summary.csv"
    funnel_path = candidate_root / "funnel_labels.jsonl"
    labels_root = candidate_root / "_labels"
    if split in {"train", "val"} and not labels_root.is_dir():
        raise FileNotFoundError("train/val scene shard has no label directory")
    if not labels_root.is_dir():
        return {
            "summary_sha256": sha256_file(summary_path),
            "funnel_labels_sha256": sha256_file(funnel_path),
            "labels_manifest_sha256": None,
            "labels_file_count": 0,
        }

    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    funnel_rows = read_jsonl(funnel_path)
    summary_by_id = {str(row.get("sample_id", "")): row for row in summary_rows}
    funnel_by_id = {str(row.get("sample_id", "")): row for row in funnel_rows}
    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    if (
        len(summary_by_id) != len(summary_rows)
        or len(funnel_by_id) != len(funnel_rows)
        or set(summary_by_id) != set(sample_ids)
        or set(funnel_by_id) != set(sample_ids)
    ):
        raise ValueError("summary/funnel label sample universe differs")

    feature_labels = pd.read_parquet(
        feature_root / "per_candidate.parquet",
        columns=[
            "sample_id",
            "candidate_id",
            "candidate_positive",
            "candidate_gt_iou",
            "candidate_gt_angle_error",
        ],
    ).sort_values(["sample_id", "candidate_id"], kind="mergesort")
    if feature_labels[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("feature label table contains duplicate candidate IDs")
    expected_feature_labels: list[dict[str, Any]] = []
    label_files: list[Path] = []
    oracle_counts = {
        "raw_oracle": 0,
        "mask_validated_oracle": 0,
        "nms_oracle": 0,
    }
    candidate_counts = {
        "raw_candidates": 0,
        "mask_validated_candidates": 0,
        "nms_candidates": 0,
    }
    nonempty_samples = 0
    empty_samples = 0
    for manifest_row in manifest_rows:
        sample_id = str(manifest_row["sample_id"])
        summary = summary_by_id[sample_id]
        funnel = funnel_by_id[sample_id]
        label_path = labels_root / f"{sample_id}.json"
        label = read_json(label_path)
        label_files.append(label_path)
        label_sample = label.get("sample")
        if (
            label.get("split") != split
            or label.get("label_only_gt_artifact") is not True
            or label.get("inference_artifact_dependency") is not False
            or label.get("protocol_identity_sha256")
            != candidate_config.get("protocol_identity_sha256")
            or not isinstance(label_sample, Mapping)
            or dict(label_sample) != funnel
            or int(summary.get("question_index", -1))
            != int(manifest_row.get("question_index", -1))
            or str(summary.get("scene_id", ""))
            != str(manifest_row.get("scene_id", ""))
            or str(summary.get("query", ""))
            != str(manifest_row.get("query", ""))
        ):
            raise ValueError(f"{sample_id}: retained oracle provenance differs")
        for summary_key, funnel_key in (
            ("raw_candidate_count", "raw_candidate_count"),
            ("mask_validated_count", "mask_validated_candidate_count"),
            ("post_nms_count", "nms_candidate_count"),
        ):
            if int(summary.get(summary_key, -1)) != int(
                funnel.get(funnel_key, -2)
            ):
                raise ValueError(f"{sample_id}: summary/funnel counts differ")
        expected_status = (
            "success_empty" if bool(funnel.get("valid_empty")) else "success_nonempty"
        )
        if summary.get("status") != expected_status:
            raise ValueError(f"{sample_id}: summary/funnel status differs")
        nonempty_samples += int(expected_status == "success_nonempty")
        empty_samples += int(expected_status == "success_empty")
        candidate_counts["raw_candidates"] += int(funnel["raw_candidate_count"])
        candidate_counts["mask_validated_candidates"] += int(
            funnel["mask_validated_candidate_count"]
        )
        candidate_counts["nms_candidates"] += int(
            funnel["nms_candidate_count"]
        )
        for key in oracle_counts:
            oracle_counts[key] += int(bool(funnel[key]))
        candidate_labels = label.get("candidate_labels")
        if not isinstance(candidate_labels, list):
            raise ValueError(f"{sample_id}: candidate labels are not a list")
        if len(candidate_labels) != int(funnel["nms_candidate_count"]):
            raise ValueError(f"{sample_id}: candidate label count differs")
        candidate_ids: set[str] = set()
        for row in candidate_labels:
            candidate_id = str(row.get("candidate_id", ""))
            if (
                not candidate_id
                or candidate_id in candidate_ids
                or str(row.get("sample_id", "")) != sample_id
            ):
                raise ValueError(f"{sample_id}: candidate label identity differs")
            candidate_ids.add(candidate_id)
            expected_feature_labels.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_positive": bool(row["candidate_positive"]),
                    "candidate_gt_iou": float(row["candidate_gt_iou"]),
                    "candidate_gt_angle_error": float(
                        row["candidate_gt_angle_error_deg"]
                    ),
                }
            )

    config_counts = candidate_config.get("counts", {})
    expected_counts = {
        "samples": len(sample_ids),
        "nonempty_samples": nonempty_samples,
        "empty_samples": empty_samples,
        **candidate_counts,
        **oracle_counts,
    }
    if any(
        int(config_counts.get(key, -1)) != value
        for key, value in expected_counts.items()
    ):
        raise ValueError("summary/funnel aggregate differs from candidate config")

    label_columns = [
        "sample_id",
        "candidate_id",
        "candidate_positive",
        "candidate_gt_iou",
        "candidate_gt_angle_error",
    ]
    expected_frame = pd.DataFrame(
        expected_feature_labels, columns=label_columns
    ).sort_values(["sample_id", "candidate_id"], kind="mergesort")
    if (
        feature_labels[["sample_id", "candidate_id"]].to_dict("records")
        != expected_frame[["sample_id", "candidate_id"]].to_dict("records")
        or feature_labels["candidate_positive"].astype(bool).tolist()
        != expected_frame["candidate_positive"].astype(bool).tolist()
        or not np.array_equal(
            feature_labels["candidate_gt_iou"].to_numpy(dtype=np.float64),
            expected_frame["candidate_gt_iou"].to_numpy(dtype=np.float64),
        )
        or not np.array_equal(
            feature_labels["candidate_gt_angle_error"].to_numpy(
                dtype=np.float64
            ),
            expected_frame["candidate_gt_angle_error"].to_numpy(
                dtype=np.float64
            ),
        )
    ):
        raise ValueError("retained feature labels differ from source labels")
    labels_manifest_sha256, labels_file_count = (
        _directory_file_manifest_sha256(labels_root)
    )
    if labels_file_count != len(sample_ids) or set(label_files) != set(
        labels_root.glob("*.json")
    ):
        raise ValueError("label file universe differs from candidate samples")
    feature_manifest = read_json(feature_root / "dataset_manifest.json")
    if (
        Path(str(feature_manifest.get("labels_source", ""))).resolve()
        != labels_root.resolve()
        or feature_manifest.get("labels_source_sha256")
        != labels_manifest_sha256
        or int(feature_manifest.get("labels_source_file_count", -1))
        != labels_file_count
    ):
        raise ValueError("feature shard is not bound to the verified label source")
    return {
        "summary_sha256": sha256_file(summary_path),
        "funnel_labels_sha256": sha256_file(funnel_path),
        "labels_manifest_sha256": labels_manifest_sha256,
        "labels_file_count": labels_file_count,
        "oracle_counts": oracle_counts,
        "candidate_counts": candidate_counts,
    }


def _directory_logical_bytes(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        if root_path.is_symlink():
            raise ValueError(f"refusing to prune symlinked directory: {root_path}")
        for name in directories + files:
            item = root_path / name
            if item.is_symlink():
                raise ValueError(f"refusing to prune symlink: {item}")
        total += sum((root_path / name).stat().st_size for name in files)
    return total


def _atomic_json(path: Path, payload: Any, *, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(
            payload,
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
    (
        candidate_root,
        scored_root,
        compact_score,
        feature_root,
        receipt_path,
        run_root,
        tmp_root,
    ) = validate_scopes(
        candidate_root=args.candidate_root,
        scored_root=args.scored_root,
        compact_score=args.compact_score,
        feature_root=args.feature_root,
        receipt_path=args.receipt_path,
    )
    candidate_config_path = candidate_root / "run_config.json"
    scorer_config_path = scored_root / "run_config.json"
    verification_path = scored_root / "verification_report.json"
    source_manifest_path = scored_root / "source_candidate_manifest.jsonl"
    score_manifest_path = compact_manifest_path(compact_score)
    feature_manifest_path = feature_root / "dataset_manifest.json"
    candidate_config = read_json(candidate_config_path)
    scorer_config = read_json(scorer_config_path)
    verification = read_json(verification_path)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    score_manifest = read_json(score_manifest_path)
    feature_manifest = read_json(feature_manifest_path)
    selection = candidate_config.get("selection")
    if (
        candidate_config.get("status") != "COMPLETED"
        or candidate_config.get("split") != args.split
        or int(candidate_config.get("counts", {}).get("execution_failures", -1))
        != 0
        or not isinstance(selection, Mapping)
        or selection.get("scene_grouped") is not True
        or selection.get("limit") is not None
    ):
        raise ValueError("candidate source is not a complete clean scene shard")
    num_shards = int(selection["num_shards"])
    shard_index = int(selection["shard_index"])
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("candidate scene-shard indices are invalid")
    stage_hashes: dict[str, str] = {}
    for stage, artifact in candidate_config.get(
        "candidate_stage_artifacts", {}
    ).items():
        path = Path(str(artifact.get("path", ""))).resolve()
        if (
            not path.is_file()
            or candidate_root not in path.parents
            or sha256_file(path) != artifact.get("sha256")
            or pq.ParquetFile(path).metadata.num_rows
            != int(artifact.get("rows", -1))
        ):
            raise ValueError(f"candidate {stage} compact artifact differs")
        stage_hashes[str(stage)] = str(artifact["sha256"])
    if set(stage_hashes) != {"raw", "mask_validated", "nms"}:
        raise ValueError("candidate compact stage set is incomplete")

    source_identity = scorer_config.get("source_identity")
    if (
        verification.get("clean") is not True
        or Path(str(verification.get("candidate_root", ""))).resolve()
        != candidate_root
        or Path(str(verification.get("scored_root", ""))).resolve()
        != scored_root
        or verification.get("source_manifest_sha256")
        != source_manifest_sha256
        or not isinstance(source_identity, Mapping)
        or source_identity.get("source_manifest_sha256")
        != source_manifest_sha256
        or int(source_identity.get("samples", -1))
        != int(candidate_config["counts"]["samples"])
        or int(source_identity.get("candidate_count", -1))
        != int(candidate_config["counts"]["nms_candidates"])
    ):
        raise ValueError("scored shard lacks matching clean independent verification")

    score_partition = score_manifest.get("partition")
    if not isinstance(score_partition, Mapping):
        raise ValueError("compact score manifest has no scene partition")
    _validate_partition(
        score_partition,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if (
        score_manifest.get("status") != "COMPLETED"
        or score_manifest.get("split") != args.split
        or score_manifest.get("gt_free") is not True
        or Path(str(score_manifest.get("candidate_root", ""))).resolve()
        != candidate_root
        or Path(str(score_manifest.get("scored_root", ""))).resolve()
        != scored_root
        or score_manifest.get("candidate_protocol_identity_sha256")
        != candidate_config.get("protocol_identity_sha256")
        or score_manifest.get("candidate_run_manifest_sha256")
        != sha256_file(candidate_root / "run_manifest.jsonl")
        or score_manifest.get("gqcnn_scores_parquet_sha256")
        != sha256_file(compact_score)
        or score_manifest.get("independent_verification_sha256")
        != sha256_file(verification_path)
    ):
        raise ValueError("compact score provenance differs from the scene shard")

    if (
        feature_manifest.get("split") != args.split
        or feature_manifest.get("streaming_scene_shard") is not True
        or feature_manifest.get("shard_assignment") != SCENE_SHARD_ASSIGNMENT
        or int(feature_manifest.get("num_shards", -1)) != num_shards
        or int(feature_manifest.get("shard_index", -1)) != shard_index
        or Path(str(feature_manifest.get("candidate_root", ""))).resolve()
        != candidate_root
        or Path(str(feature_manifest.get("scored_root", ""))).resolve()
        != scored_root
        or feature_manifest.get("candidate_run_config_sha256")
        != sha256_file(candidate_config_path)
        or feature_manifest.get("scorer_run_config_sha256")
        != sha256_file(scorer_config_path)
        or feature_manifest.get("scorer_source_manifest_sha256")
        != source_manifest_sha256
        or feature_manifest.get("scoring_verification_sha256")
        != sha256_file(verification_path)
        or feature_manifest.get("scoring_verification_clean") is not True
        or (
            args.split in {"train", "val"}
            and feature_manifest.get("labels_joined_post_extraction") is not True
        )
    ):
        raise ValueError("feature shard provenance differs from the scene shard")
    for filename, hash_key in (
        ("per_candidate.parquet", "per_candidate_sha256"),
        ("per_sample.parquet", "per_sample_sha256"),
    ):
        if sha256_file(feature_root / filename) != feature_manifest.get(hash_key):
            raise ValueError(f"feature artifact changed: {filename}")

    candidate_rows, feature_samples = _validate_feature_score_semantics(
        compact_score=compact_score,
        feature_root=feature_root,
    )
    candidate_counts = candidate_config["counts"]
    expected_samples = int(candidate_counts["samples"])
    expected_candidates = int(candidate_counts["nms_candidates"])
    if (
        candidate_rows != expected_candidates
        or candidate_rows != int(score_manifest.get("rows", -1))
        or candidate_rows != int(feature_manifest.get("candidate_count", -1))
        or feature_samples != expected_samples
        or feature_samples != int(score_manifest.get("samples", -1))
        or feature_samples != int(feature_manifest.get("sample_count", -1))
        or int(verification.get("expected_total_samples", -1))
        != expected_samples
        or int(verification.get("expected_frozen_candidates", -1))
        != expected_candidates
    ):
        raise ValueError("candidate/score/feature shard accounting differs")

    manifest_rows = read_jsonl(candidate_root / "run_manifest.jsonl")
    sample_ids = [str(row["sample_id"]) for row in manifest_rows]
    if len(sample_ids) != expected_samples or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("candidate run manifest sample accounting differs")
    oracle_retention = _validate_oracle_artifacts(
        split=args.split,
        candidate_root=candidate_root,
        feature_root=feature_root,
        candidate_config=candidate_config,
        manifest_rows=manifest_rows,
    )
    targets: list[Path] = []
    for sample_id in sample_ids:
        candidate_sample = candidate_root / sample_id
        scored_sample = scored_root / sample_id
        if not candidate_sample.is_dir() or not scored_sample.is_dir():
            raise FileNotFoundError(
                f"verbose sample directory is missing: {sample_id}"
            )
        targets.extend((candidate_sample, scored_sample))
    for optional in (candidate_root / "_scene_cache", candidate_root / "_labels"):
        if optional.exists():
            if not optional.is_dir():
                raise ValueError(f"expected directory: {optional}")
            targets.append(optional)
    logical_bytes = sum(_directory_logical_bytes(path) for path in targets)
    artifacts = {
        "candidate_run_config_sha256": sha256_file(candidate_config_path),
        "candidate_run_manifest_sha256": sha256_file(
            candidate_root / "run_manifest.jsonl"
        ),
        "scorer_run_config_sha256": sha256_file(scorer_config_path),
        "scorer_source_manifest_sha256": source_manifest_sha256,
        "scoring_verification_sha256": sha256_file(verification_path),
        "compact_score_sha256": sha256_file(compact_score),
        "compact_score_manifest_sha256": sha256_file(score_manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_candidate_sha256": sha256_file(
            feature_root / "per_candidate.parquet"
        ),
        "feature_sample_sha256": sha256_file(feature_root / "per_sample.parquet"),
        "summary_sha256": oracle_retention["summary_sha256"],
        "funnel_labels_sha256": oracle_retention["funnel_labels_sha256"],
        "candidate_stage_sha256": stage_hashes,
    }
    receipt = {
        "schema_version": 1,
        "status": "VERIFIED_DRY_RUN",
        "run_root": str(run_root),
        "split": args.split,
        "partition": {
            "assignment": SCENE_SHARD_ASSIGNMENT,
            "num_shards": num_shards,
            "shard_index": shard_index,
        },
        "candidate_root": str(candidate_root),
        "scored_root": str(scored_root),
        "eligible_directories": [str(path) for path in targets],
        "eligible_directory_count": len(targets),
        "logical_bytes_eligible": logical_bytes,
        "semantic_checks": {
            "samples": expected_samples,
            "candidates": expected_candidates,
            "primary_key_equal": True,
            "candidate_identity_equal": True,
            "q_value_equal": True,
            "gqcnn_rank_equal": True,
            "ground_truth_used_for_inference": False,
        },
        "artifacts": artifacts,
        "oracle_retention": oracle_retention,
        "executed": False,
    }
    _atomic_json(receipt_path, receipt, tmp_root=tmp_root)
    if not args.execute:
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    receipt["status"] = "PRUNE_IN_PROGRESS"
    _atomic_json(receipt_path, receipt, tmp_root=tmp_root)
    for target in targets:
        if not _is_below(target.resolve(), tmp_root):
            raise RuntimeError(f"refusing to delete outside run tmp: {target}")
        shutil.rmtree(target)
    remaining = [str(path) for path in targets if path.exists()]
    if remaining:
        raise RuntimeError(f"verbose prune left directories: {remaining[:5]}")
    retained_paths = {
        "candidate_run_config_sha256": candidate_config_path,
        "candidate_run_manifest_sha256": candidate_root / "run_manifest.jsonl",
        "scorer_run_config_sha256": scorer_config_path,
        "scorer_source_manifest_sha256": source_manifest_path,
        "scoring_verification_sha256": verification_path,
        "compact_score_sha256": compact_score,
        "compact_score_manifest_sha256": score_manifest_path,
        "feature_manifest_sha256": feature_manifest_path,
        "feature_candidate_sha256": feature_root / "per_candidate.parquet",
        "feature_sample_sha256": feature_root / "per_sample.parquet",
        "summary_sha256": candidate_root / "summary.csv",
        "funnel_labels_sha256": candidate_root / "funnel_labels.jsonl",
    }
    for name, path in retained_paths.items():
        if sha256_file(path) != artifacts[name]:
            raise RuntimeError(f"retained downstream artifact changed: {path}")
    for stage, digest in stage_hashes.items():
        artifact = candidate_config["candidate_stage_artifacts"][stage]
        if sha256_file(Path(str(artifact["path"]))) != digest:
            raise RuntimeError(f"retained candidate {stage} changed during prune")
    receipt.update(
        {
            "status": "COMPLETED",
            "executed": True,
            "deleted_directory_count": len(targets),
            "remaining_eligible_directories": remaining,
        }
    )
    _atomic_json(receipt_path, receipt, tmp_root=tmp_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
