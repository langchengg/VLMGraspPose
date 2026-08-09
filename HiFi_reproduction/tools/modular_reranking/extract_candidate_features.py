#!/usr/bin/env python3
"""Extract leakage-safe features for frozen candidate and GQ-CNN artifacts."""

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
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.geometric_ranker import (  # noqa: E402
    load_frozen_candidates,
    load_intrinsics,
)
from src.grasping.reranking_v1.features import (  # noqa: E402
    FORBIDDEN_GT_COLUMNS,
    INFERENCE_FEATURE_ALLOWLIST,
    SCHEMA_VERSION,
    compute_train_only_statistics,
    extract_sample_features,
    feature_schema,
    join_candidate_labels,
    resize_probability_to_native,
    stable_rows_sha256,
)
from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from tools.modular_reranking.project_gt_free_query_metadata import (  # noqa: E402
    atomic_json,
    load_query_metadata_bundle,
    load_verified_prediction_bundle,
    validate_tmp_scope,
)
from tools.modular_reranking.scene_sharding import (  # noqa: E402
    SCENE_SHARD_ASSIGNMENT,
    assigned_to_scene_shard,
)


SAMPLE_SHARD_ASSIGNMENT = "sha256(sample_id)[0:8] modulo num_shards"
OFFICIAL_SCORER_MODEL_IDENTITY = {
    "model_name": "GQCNN-2.1",
    "model_commit": "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21",
    "model_config_hash": (
        "eb5bc17089a39bd8fe6c801010c25a6a79a898d64181180feb5cf69aa630ff6f"
    ),
    "model_file_manifest_hash": (
        "8201961abe3a09d90c6c66e582a3bfeb181d7095a2ebcc3a9d90e68fc12e8614"
    ),
    "docker_image": "vlmgrasp/gqcnn-score:1.3.0",
    "docker_image_id": (
        "sha256:3d1158ca83197d55808454b718d0a328d3f27c57c80baaaea7031e21a9134ebd"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--query-metadata",
        type=Path,
        required=True,
        help=(
            "Exact-schema GT-free query_metadata.jsonl created by "
            "project_gt_free_query_metadata.py"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tmp-root",
        type=Path,
        required=True,
        help="Atomic temporary files; must be inside the current active run's tmp/",
    )
    parser.add_argument(
        "--split", choices=("train", "development", "val", "test"), required=True
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        help="Optional post-extraction GT labels directory; never read by feature functions",
    )
    parser.add_argument(
        "--labels-parquet",
        type=Path,
        help="Optional strict labels table keyed by sample_id+candidate_id",
    )
    parser.add_argument(
        "--train-statistics",
        type=Path,
        help="Train-only statistics to reference for val/test; never refitted",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--shard-assignment",
        choices=("sample_id", "scene_id"),
        default="sample_id",
        help=(
            "sample_id preserves the original feature-only sharding; scene_id "
            "matches generate_compact_dexnet_candidates.py for streaming."
        ),
    )
    return parser.parse_args()


def atomic_parquet(path: Path, frame: pd.DataFrame, *, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        tmp_root
        / "atomic_candidate_features"
        / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if tmp_root != temporary.parent and tmp_root not in temporary.parents:
        raise RuntimeError("feature Parquet temporary escaped --tmp-root")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def load_probability(path: Path, native_shape: tuple[int, int]) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        probability = np.asarray(loaded, dtype=np.float32)
    else:
        with loaded as archive:
            if "probability" not in archive.files:
                raise ValueError(
                    f"probability archive missing probability key: {path}"
                )
            probability = np.asarray(archive["probability"], dtype=np.float32)
    return resize_probability_to_native(probability, native_shape)


def load_scores(
    scored_dir: Path, candidate_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    path = scored_dir / "gqcnn_scored_candidates.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = {"candidate_id", "gqcnn_q_value", "gqcnn_rank"}
        if not required <= set(archive.files):
            raise ValueError(
                f"scored archive missing keys {sorted(required - set(archive.files))}"
            )
        ids = [str(value) for value in archive["candidate_id"].tolist()]
        q = np.asarray(archive["gqcnn_q_value"], dtype=float)
        ranks = np.asarray(archive["gqcnn_rank"], dtype=int)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate scored candidate IDs: {scored_dir}")
    index = {candidate_id: offset for offset, candidate_id in enumerate(ids)}
    if set(index) != set(candidate_ids):
        raise ValueError(f"candidate-ID join mismatch: {scored_dir.name}")
    return (
        np.asarray([q[index[item]] for item in candidate_ids], dtype=float),
        np.asarray([ranks[index[item]] for item in candidate_ids], dtype=int),
    )


def load_empty_score_marker(scored_dir: Path, *, sample_id: str) -> dict[str, str]:
    marker_path = scored_dir / "_SCORING_COMPLETE.json"
    metadata_path = scored_dir / "scoring_metadata.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        marker.get("sample_id") != sample_id
        or metadata.get("sample_id") != sample_id
        or marker.get("scoring_status") != "skipped_valid_empty"
        or metadata.get("scoring_status") != "skipped_valid_empty"
        or int(marker.get("source_candidate_count", -1)) != 0
        or int(marker.get("gqcnn_scored_count", -1)) != 0
        or int(metadata.get("source_candidate_count", -1)) != 0
        or int(metadata.get("gqcnn_scored_count", -1)) != 0
    ):
        raise ValueError(f"invalid valid-empty score marker: {scored_dir}")
    if (scored_dir / "gqcnn_scored_candidates.npz").exists():
        raise ValueError(f"valid-empty scorer unexpectedly wrote an NPZ: {scored_dir}")
    return {
        "scored_npz_sha256": None,
        "scoring_complete_sha256": sha256_file(marker_path),
        "scoring_metadata_sha256": sha256_file(metadata_path),
        "scoring_status": "skipped_valid_empty",
    }


def load_labels(labels_root: Path, sample_ids: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        path = labels_root / f"{sample_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = payload.get("candidate_labels")
        if not isinstance(labels, list):
            raise ValueError(f"candidate_labels missing: {path}")
        output.extend(dict(item) for item in labels)
    return output


def assigned_to_shard(sample_id: str, *, num_shards: int, shard_index: int) -> bool:
    """Assign a sample deterministically without depending on directory order."""

    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % num_shards
    return bucket == shard_index


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


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_scene_streaming_roots(
    *,
    candidate_root: Path,
    scored_root: Path,
    prediction_manifest_sha256: str,
    split: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    candidate_config_path = candidate_root / "run_config.json"
    scorer_config_path = scored_root / "run_config.json"
    verification_path = scored_root / "verification_report.json"
    candidate_config = _read_json_object(candidate_config_path)
    scorer_config = _read_json_object(scorer_config_path)
    verification = _read_json_object(verification_path)
    source_manifest_path = scored_root / "source_candidate_manifest.jsonl"
    source_manifest_sha256 = sha256_file(source_manifest_path)
    selection = candidate_config.get("selection")
    if (
        candidate_config.get("status") != "COMPLETED"
        or candidate_config.get("split") != split
        or candidate_config.get("prediction_manifest_sha256")
        != prediction_manifest_sha256
        or not isinstance(selection, Mapping)
        or selection.get("scene_grouped") is not True
        or selection.get("limit") is not None
        or int(selection.get("num_shards", -1)) != num_shards
        or int(selection.get("shard_index", -1)) != shard_index
    ):
        raise ValueError("candidate root is not the requested complete scene shard")
    source_identity = scorer_config.get("source_identity")
    if (
        not isinstance(source_identity, Mapping)
        or source_identity.get("source_manifest_sha256")
        != source_manifest_sha256
        or int(source_identity.get("samples", -1))
        != int(candidate_config["counts"]["samples"])
        or int(source_identity.get("candidate_count", -1))
        != int(candidate_config["counts"]["nms_candidates"])
    ):
        raise ValueError("scorer source identity differs from the candidate shard")
    if (
        verification.get("clean") is not True
        or Path(str(verification.get("candidate_root", ""))).resolve()
        != candidate_root
        or Path(str(verification.get("scored_root", ""))).resolve()
        != scored_root
        or verification.get("source_manifest_sha256")
        != source_manifest_sha256
    ):
        raise ValueError("scene-shard scoring lacks a clean independent verification")
    model = scorer_config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("scorer run config has no model identity")
    model_identity = {
        key: model.get(key)
        for key in (
            "model_name",
            "model_commit",
            "model_config_hash",
            "model_file_manifest_hash",
            "docker_image",
            "docker_image_id",
        )
    }
    if model_identity != OFFICIAL_SCORER_MODEL_IDENTITY:
        raise ValueError("scorer model identity is not the frozen official GQCNN-2.1")
    verification_model_identity = {
        key: verification.get(key)
        for key in (
            "model_name",
            "model_commit",
            "model_config_hash",
            "model_file_manifest_hash",
        )
    }
    if verification_model_identity != {
        key: OFFICIAL_SCORER_MODEL_IDENTITY[key]
        for key in verification_model_identity
    }:
        raise ValueError(
            "independent verification model identity differs from frozen GQCNN-2.1"
        )
    return {
        "candidate_run_config_sha256": sha256_file(candidate_config_path),
        "candidate_protocol_identity_sha256": candidate_config.get(
            "protocol_identity_sha256"
        ),
        "candidate_protocol_family_identity_sha256": candidate_config.get(
            "protocol_family_identity_sha256"
        ),
        "scorer_run_config_sha256": sha256_file(scorer_config_path),
        "scorer_source_manifest_sha256": source_manifest_sha256,
        "scorer_model_identity": model_identity,
        "scorer_model_identity_sha256": _canonical_json_hash(model_identity),
        "scoring_verification_sha256": sha256_file(verification_path),
        "scoring_verification_clean": True,
    }


def labels_directory_manifest(labels_root: Path) -> tuple[str, int]:
    files = sorted(path for path in labels_root.glob("*.json") if path.is_file())
    if not files:
        raise ValueError(f"labels directory contains no JSON files: {labels_root}")
    payload = [
        {
            "relative_path": path.relative_to(labels_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return (
        hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        len(files),
    )


def main() -> int:
    args = parse_args()
    candidate_root = args.candidate_root.expanduser().resolve()
    scored_root = args.scored_root.expanduser().resolve()
    prediction_root = args.prediction_root.expanduser().resolve()
    query_metadata_path = args.query_metadata.expanduser().resolve()
    output_root, tmp_root, _ = validate_tmp_scope(
        args.output_root, args.tmp_root
    )
    if output_root.exists():
        raise FileExistsError(f"feature output already exists: {output_root}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.labels_root is not None and args.labels_parquet is not None:
        raise ValueError("--labels-root and --labels-parquet are mutually exclusive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    prediction_bundle = load_verified_prediction_bundle(
        prediction_root, split=args.split
    )
    predictions = prediction_bundle.by_sample_id
    query_metadata, query_provenance = load_query_metadata_bundle(
        query_metadata_path,
        prediction_bundle=prediction_bundle,
        split=args.split,
    )
    all_sample_dirs = sorted(
        path
        for path in candidate_root.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "candidates.npz").is_file()
    )
    if args.shard_assignment == "scene_id":
        if args.limit is not None:
            raise ValueError("--limit is not allowed for complete scene shards")
        expected_ids = {
            sample_id
            for sample_id, row in predictions.items()
            if assigned_to_scene_shard(
                str(row["scene_id"]),
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            )
        }
        observed_ids = {path.name for path in all_sample_dirs}
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            extra = sorted(observed_ids - expected_ids)
            raise ValueError(
                "candidate scene shard coverage differs from the frozen "
                f"prediction manifest: missing={missing[:5]} extra={extra[:5]}"
            )
        sample_dirs = all_sample_dirs
        streaming_provenance = _validate_scene_streaming_roots(
            candidate_root=candidate_root,
            scored_root=scored_root,
            prediction_manifest_sha256=prediction_bundle.manifest_sha256,
            split=args.split,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        shard_assignment = SCENE_SHARD_ASSIGNMENT
    else:
        sample_dirs = [
            path
            for path in all_sample_dirs
            if assigned_to_shard(
                path.name,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            )
        ]
        streaming_provenance = {}
        shard_assignment = SAMPLE_SHARD_ASSIGNMENT
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    if not sample_dirs:
        raise ValueError("no candidate samples selected")
    output_root.mkdir(parents=True)

    all_features: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    for offset, candidate_dir in enumerate(sample_dirs, start=1):
        sample_id = candidate_dir.name
        prediction = predictions.get(sample_id)
        if prediction is None:
            raise ValueError(f"prediction row missing: {sample_id}")
        if prediction["split"] != args.split:
            raise ValueError(f"prediction split mismatch: {sample_id}")
        query_row = query_metadata.get(sample_id)
        if query_row is None:
            raise ValueError(f"GT-free query metadata missing: {sample_id}")
        records, metadata, candidate_hashes = load_frozen_candidates(
            candidate_dir / "candidates.npz", candidate_dir / "candidates.json"
        )
        expected_identity = {
            "sample_id": sample_id,
            "question_index": int(query_row["question_index"]),
            "scene_id": str(query_row["scene_id"]),
            "query": str(query_row["query"]),
        }
        for field, expected in expected_identity.items():
            observed = (
                int(metadata[field])
                if field == "question_index"
                else str(metadata[field])
            )
            if observed != expected:
                raise ValueError(
                    f"candidate/query metadata {field} mismatch: {sample_id}"
                )
        for field in ("question_index", "scene_id", "query"):
            expected = expected_identity[field]
            observed = (
                int(prediction[field])
                if field == "question_index"
                else str(prediction[field])
            )
            if observed != expected:
                raise ValueError(
                    f"prediction/query metadata {field} mismatch: {sample_id}"
                )
        candidate_ids = [str(record["candidate_id"]) for record in records]
        if candidate_ids:
            q_values, q_ranks = load_scores(
                scored_root / sample_id, candidate_ids
            )
            score_provenance: dict[str, Any] = {
                "scored_npz_sha256": sha256_file(
                    scored_root
                    / sample_id
                    / "gqcnn_scored_candidates.npz"
                ),
                "scoring_status": "scored_nonempty",
            }
        else:
            q_values = np.asarray([], dtype=float)
            q_ranks = np.asarray([], dtype=int)
            score_provenance = load_empty_score_marker(
                scored_root / sample_id, sample_id=sample_id
            )
        depth_path = candidate_dir / "depth_m.npy"
        mask_path = candidate_dir / "hifics_mask_processed.png"
        depth = np.asarray(np.load(depth_path, allow_pickle=False), dtype=np.float32)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        if depth.shape != mask.shape:
            raise ValueError(f"depth/mask shape mismatch: {sample_id}")
        probability_path = Path(prediction["probability_path"])
        if (
            not probability_path.is_file()
            or sha256_file(probability_path) != prediction["probability_sha256"]
        ):
            raise ValueError(f"prediction probability identity mismatch: {sample_id}")
        probability = load_probability(probability_path, depth.shape)
        intrinsics = load_intrinsics(candidate_dir / "camera.intr")
        rows, sample_row = extract_sample_features(
            records,
            q_values=q_values,
            q_ranks=q_ranks,
            probability=probability,
            binary_mask=mask,
            depth_m=depth,
            intrinsics=intrinsics,
            sample_id=sample_id,
            scene_id=str(query_row["scene_id"]),
            split=args.split,
            query_type=str(query_row["query_type"]),
        )
        all_features.extend(rows)
        per_sample.append(sample_row)
        sample_ids.append(sample_id)
        source_files.append(
            {
                "sample_id": sample_id,
                **candidate_hashes,
                **score_provenance,
                "probability_sha256": sha256_file(probability_path),
                "depth_m_sha256": sha256_file(depth_path),
                "mask_sha256": sha256_file(mask_path),
                "intrinsics_sha256": sha256_file(candidate_dir / "camera.intr"),
            }
        )
        print(
            f"features {offset}/{len(sample_dirs)} sample={sample_id} "
            f"candidates={len(rows)}",
            flush=True,
        )

    inference_rows_hash = stable_rows_sha256(all_features)
    labels_joined = False
    labels_source_sha256: str | None = None
    labels_source_file_count: int | None = None
    if args.labels_root is not None:
        labels_root = args.labels_root.expanduser().resolve()
        labels_source_sha256, labels_source_file_count = (
            labels_directory_manifest(labels_root)
        )
        label_rows = load_labels(labels_root, sample_ids)
        all_features = join_candidate_labels(all_features, label_rows)
        labels_joined = True
    elif args.labels_parquet is not None:
        labels_path = args.labels_parquet.expanduser().resolve()
        labels_source_sha256 = sha256_file(labels_path)
        labels_source_file_count = 1
        label_frame = pd.read_parquet(labels_path)
        required_label_keys = {"sample_id", "candidate_id", "candidate_positive"}
        missing_label_keys = required_label_keys - set(label_frame.columns)
        if missing_label_keys:
            raise ValueError(
                f"labels parquet missing columns: {sorted(missing_label_keys)}"
            )
        label_frame = label_frame.loc[
            label_frame["sample_id"].astype(str).isin(sample_ids)
        ]
        all_features = join_candidate_labels(
            all_features, label_frame.to_dict("records")
        )
        labels_joined = True
    if labels_joined:
        positive_by_sample: dict[str, int] = {}
        for row in all_features:
            positive_by_sample[row["sample_id"]] = positive_by_sample.get(
                row["sample_id"], 0
            ) + int(bool(row["candidate_positive"]))
        for row in per_sample:
            row["positive_candidate_count"] = positive_by_sample.get(
                row["sample_id"], 0
            )
            row["oracle_positive"] = bool(row["positive_candidate_count"] > 0)

    if all_features:
        candidate_frame = pd.DataFrame(all_features).sort_values(
            ["sample_id", "original_gqcnn_rank", "candidate_id"],
            kind="mergesort",
        )
    else:
        candidate_frame = pd.DataFrame(
            {
                "sample_id": pd.Series(dtype="string"),
                "scene_id": pd.Series(dtype="string"),
                "split": pd.Series(dtype="string"),
                "candidate_id": pd.Series(dtype="string"),
                "candidate_identity_sha256": pd.Series(dtype="string"),
                "original_gqcnn_rank": pd.Series(dtype="int64"),
                **{
                    feature: pd.Series(dtype="float64")
                    for feature in INFERENCE_FEATURE_ALLOWLIST
                },
                **(
                    {
                        "candidate_positive": pd.Series(dtype="bool"),
                        "candidate_gt_iou": pd.Series(dtype="float64"),
                        "candidate_gt_angle_error": pd.Series(dtype="float64"),
                    }
                    if labels_joined
                    else {}
                ),
            }
        )
    sample_frame = pd.DataFrame(per_sample).sort_values("sample_id", kind="mergesort")
    atomic_parquet(
        output_root / "per_candidate.parquet",
        candidate_frame,
        tmp_root=tmp_root,
    )
    atomic_parquet(
        output_root / "per_sample.parquet", sample_frame, tmp_root=tmp_root
    )
    atomic_json(
        output_root / "feature_schema.json",
        feature_schema(candidate_frame),
        tmp_root=tmp_root,
    )
    atomic_json(
        output_root / "inference_feature_allowlist.json",
        {
            "schema_version": SCHEMA_VERSION,
            "features": list(INFERENCE_FEATURE_ALLOWLIST),
            "count": len(INFERENCE_FEATURE_ALLOWLIST),
            "ground_truth_allowed": False,
        },
        tmp_root=tmp_root,
    )
    atomic_json(
        output_root / "forbidden_gt_columns.json",
        {
            "schema_version": SCHEMA_VERSION,
            "columns": list(FORBIDDEN_GT_COLUMNS),
            "policy": "forbidden in inference features, VLM prompts, and safe-gate inputs",
        },
        tmp_root=tmp_root,
    )

    statistics_path = output_root / "feature_statistics_train_only.json"
    if args.split in {"train", "development"}:
        statistics = compute_train_only_statistics(candidate_frame, split=args.split)
    elif args.train_statistics is not None:
        source_statistics_path = args.train_statistics.expanduser().resolve()
        statistics = json.loads(source_statistics_path.read_text(encoding="utf-8"))
        if statistics.get("source_split") not in {"train", "development"}:
            raise ValueError("provided statistics are not train/development-only")
        statistics = {
            **statistics,
            "referenced_by_split": args.split,
            "source_statistics_path": str(source_statistics_path),
            "source_statistics_sha256": sha256_file(source_statistics_path),
        }
    else:
        statistics = {
            "schema_version": SCHEMA_VERSION,
            "source_split": None,
            "referenced_by_split": args.split,
            "fit_performed": False,
            "reason": "non-train split cannot fit normalization; provide --train-statistics",
            "feature_columns": list(INFERENCE_FEATURE_ALLOWLIST),
        }
    atomic_json(statistics_path, statistics, tmp_root=tmp_root)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "split": args.split,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_assignment": shard_assignment,
        "streaming_scene_shard": args.shard_assignment == "scene_id",
        "sample_count": len(sample_frame),
        "candidate_count": len(candidate_frame),
        "labels_joined_post_extraction": labels_joined,
        "labels_source": (
            str(args.labels_root.expanduser().resolve())
            if args.labels_root is not None
            else (
                str(args.labels_parquet.expanduser().resolve())
                if args.labels_parquet is not None
                else None
            )
        ),
        "labels_source_sha256": labels_source_sha256,
        "labels_source_file_count": labels_source_file_count,
        "inference_rows_sha256_before_label_join": inference_rows_hash,
        "per_candidate_sha256": sha256_file(output_root / "per_candidate.parquet"),
        "per_sample_sha256": sha256_file(output_root / "per_sample.parquet"),
        "feature_schema_sha256": sha256_file(output_root / "feature_schema.json"),
        "allowlist_sha256": sha256_file(
            output_root / "inference_feature_allowlist.json"
        ),
        "forbidden_columns_sha256": sha256_file(
            output_root / "forbidden_gt_columns.json"
        ),
        "statistics_sha256": sha256_file(statistics_path),
        # merge_feature_shards.py already freezes these two legacy contract
        # keys. Point them at the GT-free projection manifest (which also binds
        # prediction and frozen-split digests), never at *_expressions.json.
        "annotations_path": str(query_provenance["manifest_path"]),
        "annotations_sha256": query_provenance["manifest_sha256"],
        "annotations_compatibility_alias": (
            "GT-free query metadata manifest; not official expressions"
        ),
        "official_expressions_opened_by_feature_extractor": False,
        "query_metadata_path": str(query_provenance["path"]),
        "query_metadata_sha256": query_provenance["sha256"],
        "query_metadata_identity_sha256": query_provenance["identity_sha256"],
        "query_metadata_manifest_path": str(
            query_provenance["manifest_path"]
        ),
        "query_metadata_manifest_sha256": query_provenance["manifest_sha256"],
        "prediction_manifest_path": str(prediction_bundle.manifest_path),
        "prediction_manifest_sha256": prediction_bundle.manifest_sha256,
        "prediction_identity_sha256": prediction_bundle.identity_sha256,
        "frozen_split_manifest_path": str(
            prediction_bundle.frozen_manifest_path
        ),
        "frozen_split_manifest_sha256": (
            prediction_bundle.frozen_manifest_sha256
        ),
        "candidate_root": str(candidate_root),
        "scored_root": str(scored_root),
        "prediction_root": str(prediction_root),
        **streaming_provenance,
        "sources": source_files,
        "disclaimer": (
            "collision/clearance fields are visible-surface proxies from observed "
            "depth, not a complete collision guarantee"
        ),
    }
    atomic_json(
        output_root / "dataset_manifest.json", manifest, tmp_root=tmp_root
    )
    print(
        f"complete samples={len(sample_frame)} candidates={len(candidate_frame)} "
        f"labels_joined={labels_joined} output={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
