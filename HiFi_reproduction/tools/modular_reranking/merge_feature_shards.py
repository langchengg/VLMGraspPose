#!/usr/bin/env python3
"""Merge deterministic feature shards and recompute train-only statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.reranking_v1.features import (  # noqa: E402
    FORBIDDEN_GT_COLUMNS,
    INFERENCE_FEATURE_ALLOWLIST,
    SCHEMA_VERSION,
    compute_train_only_statistics,
    feature_schema,
)
from src.grasping.geometric_ranker import load_frozen_candidates  # noqa: E402
from src.grasping.reranking_v1.identity import (  # noqa: E402
    candidate_identity_sha256,
    sha256_file,
)
from src.grasping.gqcnn_full_scoring import (  # noqa: E402
    SCORED_NONEMPTY,
    SKIPPED_VALID_EMPTY,
    load_source_manifest,
    load_source_sample,
    validate_scored_output,
)
from tools.modular_reranking import low_peak_transaction as low_peak  # noqa: E402
from tools.modular_reranking.extract_candidate_features import (  # noqa: E402
    _validate_scene_streaming_roots,
)
from tools.modular_reranking.scene_sharding import (  # noqa: E402
    SCENE_SHARD_ASSIGNMENT,
    assigned_to_scene_shard,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _feature_score_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_parquet(
        root / "per_candidate.parquet",
        columns=[
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "q_raw",
            "original_gqcnn_rank",
        ],
    )
    samples = pd.read_parquet(
        root / "per_sample.parquet", columns=["sample_id", "candidate_count"]
    )
    candidates = candidates.sort_values(
        ["sample_id", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    samples = samples.sort_values("sample_id", kind="mergesort").reset_index(
        drop=True
    )
    if candidates[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("feature candidate table contains duplicate identities")
    if samples["sample_id"].duplicated().any():
        raise ValueError("feature sample table contains duplicate identities")
    if not np.all(np.isfinite(candidates["q_raw"].to_numpy(dtype=np.float64))):
        raise ValueError("feature q_raw contains non-finite values")
    return candidates, samples


def _require_exact_score_semantics(
    *, feature: pd.DataFrame, verified: pd.DataFrame
) -> None:
    verified = verified.sort_values(
        ["sample_id", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    if verified[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("verified score source contains duplicate identities")
    if (
        feature[["sample_id", "candidate_id"]].to_dict("records")
        != verified[["sample_id", "candidate_id"]].to_dict("records")
        or feature["candidate_identity_sha256"].astype(str).tolist()
        != verified["candidate_identity_sha256"].astype(str).tolist()
        or feature["original_gqcnn_rank"].astype(int).tolist()
        != verified["gqcnn_rank"].astype(int).tolist()
        or not np.array_equal(
            feature["q_raw"].to_numpy(dtype=np.float64),
            verified["gqcnn_q_value"].to_numpy(dtype=np.float64),
        )
    ):
        raise ValueError("verified scores and extracted features differ semantically")


def _source_rows_by_id(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("sources")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("feature manifest has no per-sample sources")
    output = {str(row.get("sample_id", "")): dict(row) for row in rows}
    if not output or len(output) != len(rows) or "" in output:
        raise ValueError("feature manifest source identities are empty or duplicated")
    return output


def _require_source_hashes(
    feature_source: Mapping[str, Any], source_hashes: Mapping[str, Any]
) -> None:
    expected = {
        "candidates_npz_sha256": source_hashes["candidates_npz_sha256"],
        "candidates_json_sha256": source_hashes["candidates_json_sha256"],
        "depth_m_sha256": source_hashes["depth_m_sha256"],
        "mask_sha256": source_hashes["processed_mask_sha256"],
        "intrinsics_sha256": source_hashes["camera_intrinsics_sha256"],
    }
    disagreements = sorted(
        key for key, value in expected.items() if feature_source.get(key) != value
    )
    if disagreements:
        raise ValueError(f"feature source candidate hashes differ: {disagreements}")


def _validate_live_streaming_scores(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    candidate_root: Path,
    scored_root: Path,
) -> dict[str, Any]:
    entries = load_source_manifest(scored_root / "source_candidate_manifest.jsonl")
    entry_by_id = {str(entry.get("sample_id", "")): entry for entry in entries}
    if not entry_by_id or len(entry_by_id) != len(entries) or "" in entry_by_id:
        raise ValueError("scorer source manifest identities are empty or duplicated")
    feature_sources = _source_rows_by_id(manifest)
    feature, samples = _feature_score_tables(root)
    sample_ids = samples["sample_id"].astype(str).tolist()
    expected_ids = set(entry_by_id)
    if set(sample_ids) != expected_ids or set(feature_sources) != expected_ids:
        raise ValueError("feature/scorer sample universes differ")
    candidate_dirs = {
        path.name
        for path in candidate_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    scored_dirs = {
        path.name
        for path in scored_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    if candidate_dirs != expected_ids or scored_dirs != expected_ids:
        raise ValueError("live candidate/scored directory universes differ")
    counts = {
        str(row.sample_id): int(row.candidate_count)
        for row in samples.itertuples(index=False)
    }
    if any(
        counts[sample_id] != int(entry_by_id[sample_id].get("candidate_count", -1))
        for sample_id in expected_ids
    ):
        raise ValueError("feature/scorer per-sample candidate counts differ")
    if len(feature) != sum(counts.values()):
        raise ValueError("feature candidate rows differ from per-sample counts")

    scorer_config = _read_json_object(scored_root / "run_config.json")
    model = scorer_config.get("model")
    if not isinstance(model, dict):
        raise ValueError("scorer run config has no model identity")
    seed = int(scorer_config["seed"])
    verified_rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    valid_empty = 0
    for sample_id in sample_ids:
        entry = entry_by_id[sample_id]
        source = load_source_sample(
            candidate_root / sample_id,
            expected_entry=entry,
            verify_hashes=True,
        )
        identity_records, _, _ = load_frozen_candidates(
            candidate_root / sample_id / "candidates.npz",
            candidate_root / sample_id / "candidates.json",
        )
        valid, status, _, errors = validate_scored_output(
            scored_root / sample_id,
            candidate_root / sample_id,
            entry,
            model,
            seed,
            verify_hashes=True,
        )
        if not valid or errors or status not in {SCORED_NONEMPTY, SKIPPED_VALID_EMPTY}:
            raise ValueError(
                f"official score verification failed for {sample_id}: {errors}"
            )
        feature_source = feature_sources[sample_id]
        _require_source_hashes(feature_source, source["source_hashes"])
        if status == SCORED_NONEMPTY:
            score_path = scored_root / sample_id / "gqcnn_scored_candidates.npz"
            score_sha = sha256_file(score_path)
            if (
                feature_source.get("scoring_status") != SCORED_NONEMPTY
                or feature_source.get("scored_npz_sha256") != score_sha
            ):
                raise ValueError(f"feature source scored NPZ differs: {sample_id}")
            with np.load(score_path, allow_pickle=False) as archive:
                ids = [str(value) for value in archive["candidate_id"].tolist()]
                q_values = np.asarray(archive["gqcnn_q_value"], dtype=np.float64)
                ranks = np.asarray(archive["gqcnn_rank"], dtype=np.int64)
            for record, candidate_id, q_value, rank in zip(
                identity_records, ids, q_values, ranks, strict=True
            ):
                verified_rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate_id,
                        "candidate_identity_sha256": candidate_identity_sha256(
                            record
                        ),
                        "gqcnn_q_value": float(q_value),
                        "gqcnn_rank": int(rank),
                    }
                )
        else:
            valid_empty += 1
            marker_path = scored_root / sample_id / "_SCORING_COMPLETE.json"
            metadata_path = scored_root / sample_id / "scoring_metadata.json"
            if (
                feature_source.get("scoring_status") != SKIPPED_VALID_EMPTY
                or feature_source.get("scored_npz_sha256") is not None
                or feature_source.get("scoring_complete_sha256")
                != sha256_file(marker_path)
                or feature_source.get("scoring_metadata_sha256")
                != sha256_file(metadata_path)
            ):
                raise ValueError(f"feature valid-empty score evidence differs: {sample_id}")
        evidence.append(
            {
                "sample_id": sample_id,
                "candidate_count": int(entry["candidate_count"]),
                "scoring_status": status,
                "candidate_completion_sha256": source["source_hashes"][
                    "completion_marker_sha256"
                ],
                "scoring_completion_sha256": sha256_file(
                    scored_root / sample_id / "_SCORING_COMPLETE.json"
                ),
                "scored_npz_sha256": feature_source.get("scored_npz_sha256"),
            }
        )
    verified = pd.DataFrame(
        verified_rows,
        columns=[
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "gqcnn_q_value",
            "gqcnn_rank",
        ],
    )
    _require_exact_score_semantics(feature=feature, verified=verified)
    return {
        "score_binding_mode": "LIVE_OFFICIAL_PER_SAMPLE_REPLAY",
        "score_binding_samples": len(sample_ids),
        "score_binding_candidates": len(feature),
        "score_binding_valid_empty_samples": valid_empty,
        "score_binding_sha256": _canonical_sha256(evidence),
    }


def _validate_pruned_streaming_scores(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    candidate_root: Path,
    scored_root: Path,
) -> dict[str, Any]:
    run_root = next(
        (path for path in (candidate_root, *candidate_root.parents) if (path / ".RUN_ACTIVE").is_file()),
        None,
    )
    if run_root is None:
        raise ValueError("pruned score source is not inside an active protected run")
    retained_directories = [
        path
        for verbose_root in (candidate_root, scored_root)
        for path in verbose_root.iterdir()
        if path.is_dir()
    ]
    if retained_directories:
        raise ValueError(
            f"pruned score source still has verbose directories: {retained_directories[0]}"
        )
    split = str(manifest["split"])
    shard_index = int(manifest["shard_index"])
    receipt_path = (
        run_root
        / "manifests"
        / "streaming_cleanup"
        / split
        / f"shard_{shard_index}.json"
    )
    compact_score = (
        run_root
        / "tmp"
        / split
        / "gqcnn_score_shards"
        / f"shard_{shard_index}.parquet"
    )
    compact_manifest_path = compact_score.with_suffix(".manifest.json")
    receipt = _read_json_object(receipt_path)
    artifacts = receipt.get("artifacts")
    partition = receipt.get("partition")
    semantic_checks = receipt.get("semantic_checks")
    if not isinstance(artifacts, dict) or not isinstance(partition, dict):
        raise ValueError("prune receipt is incomplete")
    expected_artifacts = {
        "candidate_run_config_sha256": sha256_file(candidate_root / "run_config.json"),
        "scorer_run_config_sha256": sha256_file(scored_root / "run_config.json"),
        "scorer_source_manifest_sha256": sha256_file(
            scored_root / "source_candidate_manifest.jsonl"
        ),
        "scoring_verification_sha256": sha256_file(
            scored_root / "verification_report.json"
        ),
        "compact_score_sha256": sha256_file(compact_score),
        "compact_score_manifest_sha256": sha256_file(compact_manifest_path),
        "feature_manifest_sha256": sha256_file(root / "dataset_manifest.json"),
        "feature_candidate_sha256": sha256_file(root / "per_candidate.parquet"),
        "feature_sample_sha256": sha256_file(root / "per_sample.parquet"),
    }
    if (
        receipt.get("status") != "COMPLETED"
        or receipt.get("executed") is not True
        or receipt.get("split") != split
        or Path(str(receipt.get("candidate_root", ""))).resolve() != candidate_root
        or Path(str(receipt.get("scored_root", ""))).resolve() != scored_root
        or receipt.get("remaining_eligible_directories") != []
        or partition.get("assignment") != SCENE_SHARD_ASSIGNMENT
        or int(partition.get("num_shards", -1)) != int(manifest["num_shards"])
        or int(partition.get("shard_index", -1)) != shard_index
        or semantic_checks
        != {
            "samples": int(manifest["sample_count"]),
            "candidates": int(manifest["candidate_count"]),
            "primary_key_equal": True,
            "candidate_identity_equal": True,
            "q_value_equal": True,
            "gqcnn_rank_equal": True,
            "ground_truth_used_for_inference": False,
        }
        or any(artifacts.get(key) != value for key, value in expected_artifacts.items())
    ):
        raise ValueError("pruned shard lacks an exact completed cleanup receipt")

    compact_manifest = _read_json_object(compact_manifest_path)
    compact_partition = compact_manifest.get("partition")
    candidate_config = _read_json_object(candidate_root / "run_config.json")
    scorer_config = _read_json_object(scored_root / "run_config.json")
    scorer_statistics_path = scored_root / "run_statistics.json"
    scoring_manifest_path = scored_root / "scoring_manifest.jsonl"
    if (
        not isinstance(compact_partition, dict)
        or compact_manifest.get("status") != "COMPLETED"
        or compact_manifest.get("split") != split
        or compact_manifest.get("gt_free") is not True
        or int(compact_manifest.get("samples", -1)) != int(manifest["sample_count"])
        or int(compact_manifest.get("rows", -1)) != int(manifest["candidate_count"])
        or int(compact_manifest.get("empty_samples", -1))
        + int(compact_manifest.get("nonempty_samples", -1))
        != int(manifest["sample_count"])
        or compact_partition.get("assignment") != SCENE_SHARD_ASSIGNMENT
        or compact_partition.get("scene_grouped") is not True
        or int(compact_partition.get("num_shards", -1))
        != int(manifest["num_shards"])
        or int(compact_partition.get("shard_index", -1)) != shard_index
        or Path(str(compact_manifest.get("candidate_root", ""))).resolve()
        != candidate_root
        or Path(str(compact_manifest.get("scored_root", ""))).resolve()
        != scored_root
        or compact_manifest.get("gqcnn_scores_parquet_sha256")
        != expected_artifacts["compact_score_sha256"]
        or compact_manifest.get("independent_verification_sha256")
        != expected_artifacts["scoring_verification_sha256"]
        or compact_manifest.get("candidate_run_manifest_sha256")
        != sha256_file(candidate_root / "run_manifest.jsonl")
        or compact_manifest.get("candidate_protocol_identity_sha256")
        != candidate_config.get("protocol_identity_sha256")
        or compact_manifest.get("candidate_protocol_family_identity_sha256")
        != candidate_config.get("protocol_family_identity_sha256")
        or compact_manifest.get("scorer_run_config_sha256")
        != expected_artifacts["scorer_run_config_sha256"]
        or compact_manifest.get("source_candidate_manifest_sha256")
        != expected_artifacts["scorer_source_manifest_sha256"]
        or compact_manifest.get("model") != scorer_config.get("model")
        or compact_manifest.get("scorer_run_statistics_sha256")
        != sha256_file(scorer_statistics_path)
        or compact_manifest.get("scoring_manifest_sha256")
        != sha256_file(scoring_manifest_path)
    ):
        raise ValueError("compact score provenance differs from pruned shard")

    feature, samples = _feature_score_tables(root)
    compact = pd.read_parquet(
        compact_score,
        columns=[
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "gqcnn_q_value",
            "gqcnn_rank",
            "source_candidate_index",
            "source_candidates_npz_sha256",
            "scored_candidates_npz_sha256",
        ],
    )
    _require_exact_score_semantics(feature=feature, verified=compact)
    entries = load_source_manifest(scored_root / "source_candidate_manifest.jsonl")
    entry_by_id = {str(entry.get("sample_id", "")): entry for entry in entries}
    sources = _source_rows_by_id(manifest)
    sample_ids = samples["sample_id"].astype(str).tolist()
    if (
        len(entry_by_id) != len(entries)
        or set(entry_by_id) != set(sample_ids)
        or set(sources) != set(sample_ids)
    ):
        raise ValueError("pruned feature/scorer sample universes differ")
    counts = dict(
        zip(
            samples["sample_id"].astype(str),
            samples["candidate_count"].astype(int),
            strict=True,
        )
    )
    for sample_id in sample_ids:
        entry = entry_by_id[sample_id]
        count = int(entry.get("candidate_count", -1))
        if counts[sample_id] != count:
            raise ValueError("pruned feature/scorer candidate counts differ")
        source = sources[sample_id]
        _require_source_hashes(source, entry["source_hashes"])
        rows = compact.loc[compact["sample_id"].astype(str) == sample_id].sort_values(
            "source_candidate_index", kind="mergesort"
        )
        if len(rows) != count or rows["candidate_id"].astype(str).tolist() != [
            str(value) for value in entry.get("candidate_ids", [])
        ]:
            raise ValueError(f"compact candidate universe differs: {sample_id}")
        if count:
            candidate_hashes = set(rows["source_candidates_npz_sha256"].astype(str))
            score_hashes = set(rows["scored_candidates_npz_sha256"].astype(str))
            if (
                candidate_hashes != {str(source["candidates_npz_sha256"])}
                or score_hashes != {str(source.get("scored_npz_sha256"))}
                or source.get("scoring_status") != SCORED_NONEMPTY
            ):
                raise ValueError(f"compact/feature source hashes differ: {sample_id}")
        elif (
            source.get("scoring_status") != SKIPPED_VALID_EMPTY
            or source.get("scored_npz_sha256") is not None
            or not SHA256_PATTERN.fullmatch(
                str(source.get("scoring_complete_sha256", ""))
            )
            or not SHA256_PATTERN.fullmatch(
                str(source.get("scoring_metadata_sha256", ""))
            )
        ):
            raise ValueError(f"pruned valid-empty evidence differs: {sample_id}")
    return {
        "score_binding_mode": "PRUNED_RECEIPT_COMPACT_SCORE_REPLAY",
        "score_binding_samples": len(samples),
        "score_binding_candidates": len(feature),
        "score_binding_valid_empty_samples": sum(value == 0 for value in counts.values()),
        "score_binding_sha256": _canonical_sha256(
            {
                "receipt_sha256": sha256_file(receipt_path),
                "compact_score_sha256": expected_artifacts["compact_score_sha256"],
                "feature_candidate_sha256": expected_artifacts[
                    "feature_candidate_sha256"
                ],
            }
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "development", "val", "test"), required=True
    )
    parser.add_argument("--train-statistics", type=Path)
    parser.add_argument(
        "--max-run-bytes",
        type=int,
        default=low_peak.DEFAULT_MAX_RUN_BYTES,
        help="Refuse publication when its conservative peak exceeds this budget.",
    )
    return parser.parse_args()


def protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def validate_tmp_scope(output_root: Path, tmp_root: Path) -> tuple[Path, Path]:
    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = protected_run_root(output)
    if protected_run_root(temporary) != run_root:
        raise ValueError("output-root and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    if output == temporary or temporary in output.parents:
        raise ValueError("final feature output cannot be stored below tmp-root")
    temporary.mkdir(parents=True, exist_ok=True)
    return output, temporary


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
    )


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def stable_frame_rows_sha256(frame: pd.DataFrame) -> str:
    """Hash canonical row JSON without materializing a second list of dicts."""

    digest = hashlib.sha256()
    digest.update(b"[")
    columns = list(map(str, frame.columns))
    for index, values in enumerate(frame.itertuples(index=False, name=None)):
        if index:
            digest.update(b",")
        row = dict(zip(columns, values, strict=True))
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


def assigned_to_shard(sample_id: str, *, num_shards: int, shard_index: int) -> bool:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards == shard_index


def _require_identical(
    manifests: list[dict[str, Any]], key: str
) -> Any:
    values = [item.get(key) for item in manifests]
    canonical = {
        json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values
    }
    if len(canonical) != 1:
        raise ValueError(f"feature shard contract mismatch for {key}: {values}")
    return values[0]


def _completion_attestation(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    status = manifest.get("status")
    if status not in {None, "COMPLETED"}:
        raise ValueError(f"feature shard has non-terminal status {status!r}: {root}")
    if manifest.get("streaming_scene_shard") is not True:
        required = (
            "per_candidate_sha256",
            "per_sample_sha256",
            "feature_schema_sha256",
            "allowlist_sha256",
            "forbidden_columns_sha256",
            "statistics_sha256",
            "prediction_manifest_path",
            "prediction_manifest_sha256",
            "prediction_identity_sha256",
            "frozen_split_manifest_sha256",
        )
        missing = [key for key in required if not manifest.get(key)]
        if missing:
            raise ValueError(
                "sample-sharded feature completion evidence is incomplete: "
                f"{root}: {missing}"
            )
        digest_keys = tuple(
            key for key in required if key != "prediction_manifest_path"
        )
        invalid_digests = [
            key
            for key in digest_keys
            if not SHA256_PATTERN.fullmatch(str(manifest[key]).lower())
        ]
        if invalid_digests:
            raise ValueError(
                "sample-sharded feature completion has invalid SHA-256 "
                f"evidence: {root}: {invalid_digests}"
            )
        prediction_path = Path(
            str(manifest["prediction_manifest_path"])
        ).expanduser()
        if prediction_path.is_symlink() or not prediction_path.is_file():
            raise ValueError(
                f"prediction manifest is missing or symlinked: {prediction_path}"
            )
        if sha256_file(prediction_path) != manifest["prediction_manifest_sha256"]:
            raise ValueError(
                f"prediction manifest SHA-256 differs: {prediction_path}"
            )
        return {
            "mode": "PENDING_COMPLETE_SET_ATTESTATION",
            "declared_status": status,
            "dataset_manifest_sha256": sha256_file(
                root / "dataset_manifest.json"
            ),
            "prediction_manifest_path": str(prediction_path.resolve()),
            "prediction_manifest_sha256": manifest[
                "prediction_manifest_sha256"
            ],
            "prediction_identity_sha256": manifest[
                "prediction_identity_sha256"
            ],
            "frozen_split_manifest_sha256": manifest[
                "frozen_split_manifest_sha256"
            ],
        }
    try:
        candidate_root = Path(str(manifest["candidate_root"])).resolve()
        scored_root = Path(str(manifest["scored_root"])).resolve()
        provenance = _validate_scene_streaming_roots(
            candidate_root=candidate_root,
            scored_root=scored_root,
            prediction_manifest_sha256=str(
                manifest["prediction_manifest_sha256"]
            ),
            split=str(manifest["split"]),
            num_shards=int(manifest["num_shards"]),
            shard_index=int(manifest["shard_index"]),
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"streaming feature shard completion cannot be attested: {root}: {error}"
        ) from error
    disagreements = sorted(
        key for key, value in provenance.items() if manifest.get(key) != value
    )
    if disagreements:
        raise ValueError(
            "streaming feature shard provenance differs from independently verified "
            f"sources: {root}: {disagreements}"
        )
    entries = load_source_manifest(scored_root / "source_candidate_manifest.jsonl")
    live_ids = {str(entry.get("sample_id", "")) for entry in entries}
    retained_candidate_ids = {
        sample_id
        for sample_id in live_ids
        if (candidate_root / sample_id).is_dir()
    }
    retained_scored_ids = {
        sample_id for sample_id in live_ids if (scored_root / sample_id).is_dir()
    }
    if retained_candidate_ids != retained_scored_ids or (
        retained_candidate_ids and retained_candidate_ids != live_ids
    ):
        raise ValueError(
            f"streaming feature shard has a partially retained live tree: {root}"
        )
    live_tree_retained = retained_candidate_ids == live_ids
    try:
        score_binding = (
            _validate_live_streaming_scores(
                root=root,
                manifest=manifest,
                candidate_root=candidate_root,
                scored_root=scored_root,
            )
            if live_tree_retained
            else _validate_pruned_streaming_scores(
                root=root,
                manifest=manifest,
                candidate_root=candidate_root,
                scored_root=scored_root,
            )
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"streaming feature shard score binding cannot be attested: {root}: {error}"
        ) from error
    return {
        "mode": "INDEPENDENT_STREAMING_COMPLETION_ATTESTATION",
        "declared_status": status,
        "dataset_manifest_sha256": sha256_file(root / "dataset_manifest.json"),
        "candidate_run_status": "COMPLETED",
        "candidate_run_config_sha256": provenance[
            "candidate_run_config_sha256"
        ],
        "scorer_run_config_sha256": provenance["scorer_run_config_sha256"],
        "scorer_source_manifest_sha256": provenance[
            "scorer_source_manifest_sha256"
        ],
        "scoring_verification_sha256": provenance[
            "scoring_verification_sha256"
        ],
        "scoring_verification_clean": True,
        **score_binding,
    }


def _attest_sample_shard_set(
    *,
    manifests: list[dict[str, Any]],
    sample_frame: pd.DataFrame,
    attestations: list[dict[str, Any]],
) -> None:
    pending = [
        item
        for item in attestations
        if item["mode"] == "PENDING_COMPLETE_SET_ATTESTATION"
    ]
    if not pending:
        return
    for key in (
        "prediction_manifest_path",
        "prediction_manifest_sha256",
        "prediction_identity_sha256",
        "frozen_split_manifest_sha256",
    ):
        values = {str(manifest.get(key, "")) for manifest in manifests}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(
                f"mixed feature shards do not share one complete {key}"
            )
    prediction_path = Path(str(manifests[0]["prediction_manifest_path"])).resolve()
    if (
        prediction_path.is_symlink()
        or not prediction_path.is_file()
        or sha256_file(prediction_path)
        != manifests[0]["prediction_manifest_sha256"]
    ):
        raise ValueError("complete-set prediction manifest identity differs")
    try:
        prediction = pd.read_json(prediction_path, lines=True)
    except Exception as error:
        raise ValueError("cannot read complete-set prediction manifest") from error
    if (
        "sample_id" not in prediction
        or prediction["sample_id"].isna().any()
        or prediction["sample_id"].astype(str).duplicated().any()
    ):
        raise ValueError("complete-set prediction manifest has invalid sample IDs")
    expected_ids = sorted(prediction["sample_id"].astype(str).tolist())
    observed_ids = sorted(sample_frame["sample_id"].astype(str).tolist())
    if observed_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(observed_ids))[:5]
        extra = sorted(set(observed_ids) - set(expected_ids))[:5]
        raise ValueError(
            "sample feature shard set does not exactly cover prediction manifest: "
            f"missing={missing}, extra={extra}"
        )
    query_ids_sha256 = hashlib.sha256(
        json.dumps(
            expected_ids,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    for item in pending:
        item.update(
            {
                "mode": "INDEPENDENT_SAMPLE_SET_COMPLETION_ATTESTATION",
                "complete_shard_set": True,
                "prediction_query_count": len(expected_ids),
                "prediction_query_ids_sha256": query_ids_sha256,
                "per_query_candidate_counts_exact": True,
            }
        )


def main() -> int:
    args = parse_args()
    shard_roots = [path.expanduser().resolve() for path in args.shard_root]
    published_root, tmp_root = validate_tmp_scope(args.output_root, args.tmp_root)
    if published_root.exists():
        raise FileExistsError(
            f"merged feature output already exists: {published_root}"
        )

    manifests: list[dict[str, Any]] = []
    schema_contracts: list[dict[str, Any]] = []
    candidates: list[pd.DataFrame] = []
    samples: list[pd.DataFrame] = []
    completion_attestations: list[dict[str, Any]] = []
    for root in shard_roots:
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        completion_attestation = _completion_attestation(root, manifest)
        if manifest["split"] != args.split:
            raise ValueError(f"split mismatch in {root}")
        artifact_contract = (
            ("per_candidate.parquet", "per_candidate_sha256"),
            ("per_sample.parquet", "per_sample_sha256"),
            ("feature_schema.json", "feature_schema_sha256"),
            ("inference_feature_allowlist.json", "allowlist_sha256"),
            ("forbidden_gt_columns.json", "forbidden_columns_sha256"),
            ("feature_statistics_train_only.json", "statistics_sha256"),
        )
        for filename, hash_key in artifact_contract:
            path = root / filename
            if sha256_file(path) != manifest.get(hash_key):
                raise ValueError(f"feature shard artifact changed: {path}")
        candidate_frame = pd.read_parquet(root / "per_candidate.parquet")
        sample_frame = pd.read_parquet(root / "per_sample.parquet")
        if (
            int(manifest.get("candidate_count", -1)) != len(candidate_frame)
            or int(manifest.get("sample_count", -1)) != len(sample_frame)
        ):
            raise ValueError(f"feature shard declared counts differ: {root}")
        numeric_counts = pd.to_numeric(
            sample_frame["candidate_count"], errors="coerce"
        )
        if (
            numeric_counts.isna().any()
            or numeric_counts.lt(0).any()
            or not numeric_counts.eq(numeric_counts.astype("int64")).all()
        ):
            raise ValueError(f"feature shard candidate counts are invalid: {root}")
        sample_frame = sample_frame.copy()
        sample_frame["candidate_count"] = numeric_counts.astype("int64")
        observed_counts = candidate_frame.groupby("sample_id", sort=True).size()
        declared_counts = sample_frame.set_index("sample_id")["candidate_count"]
        if (
            sample_frame["sample_id"].duplicated().any()
            or not observed_counts.reindex(declared_counts.index, fill_value=0).equals(
                declared_counts.astype(observed_counts.dtype)
            )
            or not set(observed_counts.index) <= set(declared_counts.index)
        ):
            raise ValueError(f"feature shard per-query candidate counts differ: {root}")
        schema = json.loads((root / "feature_schema.json").read_text())
        if int(schema.get("rows", -1)) != len(candidate_frame):
            raise ValueError(
                f"feature schema row count disagrees with shard data: {root}"
            )
        schema_contracts.append(
            {key: value for key, value in schema.items() if key != "rows"}
        )
        manifests.append(manifest)
        candidates.append(candidate_frame)
        samples.append(sample_frame)
        completion_attestations.append(completion_attestation)
        completion_attestation.update(
            {
                "verified_feature_artifacts": [
                    {"path": filename, "sha256_manifest_key": hash_key}
                    for filename, hash_key in artifact_contract
                ],
                "sample_count": len(sample_frame),
                "candidate_count": len(candidate_frame),
                "per_query_candidate_counts_exact": True,
            }
        )

    ordered = sorted(
        zip(
            shard_roots,
            manifests,
            schema_contracts,
            candidates,
            samples,
            completion_attestations,
            strict=True,
        ),
        key=lambda item: int(item[1]["shard_index"]),
    )
    shard_roots = [item[0] for item in ordered]
    manifests = [item[1] for item in ordered]
    schema_contracts = [item[2] for item in ordered]
    candidates = [item[3] for item in ordered]
    samples = [item[4] for item in ordered]
    completion_attestations = [item[5] for item in ordered]

    budget_preflight = low_peak.enforce_budget_preflight(
        run_root=protected_run_root(published_root),
        input_paths=[
            root / filename
            for root in shard_roots
            for filename in ("per_candidate.parquet", "per_sample.parquet")
        ],
        max_run_bytes=int(args.max_run_bytes),
    )

    if _require_identical(manifests, "schema_version") != SCHEMA_VERSION:
        raise ValueError("feature shard schema version is unsupported")
    shard_assignment = _require_identical(manifests, "shard_assignment")
    streaming_scene_shards = shard_assignment == SCENE_SHARD_ASSIGNMENT
    for key in (
        "allowlist_sha256",
        "forbidden_columns_sha256",
        "annotations_path",
        "annotations_sha256",
        "prediction_root",
        "labels_joined_post_extraction",
    ):
        _require_identical(manifests, key)
    if streaming_scene_shards:
        for key in (
            "streaming_scene_shard",
            "candidate_protocol_family_identity_sha256",
            "scorer_model_identity",
            "scorer_model_identity_sha256",
            "query_metadata_path",
            "query_metadata_sha256",
            "query_metadata_manifest_path",
            "query_metadata_manifest_sha256",
            "scoring_verification_clean",
            "prediction_manifest_path",
            "prediction_manifest_sha256",
            "prediction_identity_sha256",
            "frozen_split_manifest_path",
            "frozen_split_manifest_sha256",
            "query_metadata_identity_sha256",
        ):
            _require_identical(manifests, key)
        if (
            manifests[0].get("streaming_scene_shard") is not True
            or manifests[0].get("scoring_verification_clean") is not True
            or any(
                not item.get("candidate_protocol_identity_sha256")
                or not item.get("scoring_verification_sha256")
                for item in manifests
            )
        ):
            raise ValueError("streaming scene-shard provenance is incomplete")
        protocol_identities = [
            item.get("candidate_protocol_identity_sha256")
            for item in manifests
        ]
        if len(set(protocol_identities)) != len(protocol_identities):
            raise ValueError(
                "streaming scene shards reuse a candidate protocol identity"
            )
    else:
        for key in (
            "candidate_root",
            "scored_root",
            "labels_source",
            "labels_source_sha256",
            "labels_source_file_count",
        ):
            _require_identical(manifests, key)
    if len(
        {
            json.dumps(contract, sort_keys=True, ensure_ascii=False)
            for contract in schema_contracts
        }
    ) != 1:
        raise ValueError(
            "feature shard schema contract mismatch after excluding per-shard row count"
        )

    shard_counts = {int(item["num_shards"]) for item in manifests}
    shard_indices = [int(item["shard_index"]) for item in manifests]
    if shard_counts != {len(shard_roots)} or set(shard_indices) != set(
        range(len(shard_roots))
    ):
        raise ValueError("shards must form exactly one complete shard set")

    candidate_frame = pd.concat(candidates, ignore_index=True).sort_values(
        ["sample_id", "original_gqcnn_rank", "candidate_id"], kind="mergesort"
    )
    sample_frame = pd.concat(samples, ignore_index=True).sort_values(
        "sample_id", kind="mergesort"
    )
    if candidate_frame[["sample_id", "candidate_id"]].duplicated().any():
        raise ValueError("duplicate sample/candidate identity across feature shards")
    if sample_frame["sample_id"].duplicated().any():
        raise ValueError("duplicate sample identity across feature shards")
    for manifest, frame in zip(manifests, samples, strict=True):
        shard_index = int(manifest["shard_index"])
        num_shards = int(manifest["num_shards"])
        if streaming_scene_shards:
            if "scene_id" not in frame:
                raise ValueError(
                    "scene-grouped feature shard has no per-sample scene_id"
                )
            wrong = [
                str(sample_id)
                for sample_id, scene_id in zip(
                    frame["sample_id"], frame["scene_id"], strict=True
                )
                if not assigned_to_scene_shard(
                    str(scene_id),
                    num_shards=num_shards,
                    shard_index=shard_index,
                )
            ]
        else:
            wrong = [
                str(sample_id)
                for sample_id in frame["sample_id"]
                if not assigned_to_shard(
                    str(sample_id),
                    num_shards=num_shards,
                    shard_index=shard_index,
                )
            ]
        if wrong:
            raise ValueError(
                f"sample rows violate deterministic shard assignment "
                f"for shard {shard_index}: {wrong[:5]}"
            )
    _attest_sample_shard_set(
        manifests=manifests,
        sample_frame=sample_frame,
        attestations=completion_attestations,
    )
    candidate_samples = set(candidate_frame["sample_id"])
    sample_universe = set(sample_frame["sample_id"])
    if not candidate_samples <= sample_universe:
        raise ValueError("per-candidate rows contain samples outside per-sample universe")
    valid_empty = sample_frame.loc[
        ~sample_frame["sample_id"].isin(candidate_samples)
    ]
    if bool((valid_empty["candidate_count"] != 0).any()):
        raise ValueError(
            "a sample missing from per-candidate rows is not marked candidate_count=0"
        )
    missing_features = tuple(
        feature
        for feature in INFERENCE_FEATURE_ALLOWLIST
        if feature not in candidate_frame
    )
    if missing_features:
        raise ValueError(
            f"merged candidate table is missing inference features: {missing_features}"
        )
    forbidden_overlap = set(INFERENCE_FEATURE_ALLOWLIST) & set(FORBIDDEN_GT_COLUMNS)
    if forbidden_overlap:
        raise AssertionError(f"allowlist contains forbidden GT columns: {forbidden_overlap}")

    output_root = tmp_root / f".{published_root.name}.merge-{uuid.uuid4().hex}"
    output_root.mkdir()
    atomic_parquet(output_root / "per_candidate.parquet", candidate_frame)
    atomic_parquet(output_root / "per_sample.parquet", sample_frame)
    atomic_json(output_root / "feature_schema.json", feature_schema(candidate_frame))
    atomic_json(
        output_root / "inference_feature_allowlist.json",
        {
            "schema_version": SCHEMA_VERSION,
            "features": list(INFERENCE_FEATURE_ALLOWLIST),
            "count": len(INFERENCE_FEATURE_ALLOWLIST),
            "ground_truth_allowed": False,
        },
    )
    atomic_json(
        output_root / "forbidden_gt_columns.json",
        {
            "schema_version": SCHEMA_VERSION,
            "columns": list(FORBIDDEN_GT_COLUMNS),
            "policy": "forbidden in inference features, VLM prompts, and safe-gate inputs",
        },
    )

    statistics_path = output_root / "feature_statistics_train_only.json"
    if args.split in {"train", "development"}:
        statistics = compute_train_only_statistics(candidate_frame, split=args.split)
    elif args.train_statistics is not None:
        source = args.train_statistics.expanduser().resolve()
        statistics = json.loads(source.read_text())
        if statistics.get("source_split") not in {"train", "development"}:
            raise ValueError("provided statistics are not train/development-only")
        statistics = {
            **statistics,
            "referenced_by_split": args.split,
            "source_statistics_path": str(source),
            "source_statistics_sha256": sha256_file(source),
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
    atomic_json(statistics_path, statistics)

    inference_only = candidate_frame.drop(
        columns=[
            column
            for column in FORBIDDEN_GT_COLUMNS
            if column in candidate_frame.columns
        ]
    )
    labels_joined = bool(manifests[0]["labels_joined_post_extraction"])
    if streaming_scene_shards:
        labels_source = "scene-sharded; see source_shards"
        label_hashes = [
            item.get("labels_source_sha256") for item in manifests
        ]
        if labels_joined and any(not value for value in label_hashes):
            raise ValueError("a labelled scene shard omits its labels hash")
        labels_source_sha256 = (
            hashlib.sha256(
                json.dumps(
                    label_hashes,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if labels_joined
            else None
        )
        labels_source_file_count = (
            sum(int(item["labels_source_file_count"]) for item in manifests)
            if labels_joined
            else None
        )
        candidate_root = None
        scored_root = None
    else:
        labels_source = manifests[0].get("labels_source")
        labels_source_sha256 = manifests[0].get("labels_source_sha256")
        labels_source_file_count = manifests[0].get(
            "labels_source_file_count"
        )
        candidate_root = manifests[0].get("candidate_root")
        scored_root = manifests[0].get("scored_root")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "split": args.split,
        "sample_count": len(sample_frame),
        "candidate_count": len(candidate_frame),
        "shard_assignment": shard_assignment,
        "streaming_scene_shards": streaming_scene_shards,
        "labels_joined_post_extraction": labels_joined,
        "labels_source": labels_source,
        "labels_source_sha256": labels_source_sha256,
        "labels_source_file_count": labels_source_file_count,
        "annotations_path": manifests[0].get("annotations_path"),
        "annotations_sha256": manifests[0].get("annotations_sha256"),
        "candidate_root": candidate_root,
        "scored_root": scored_root,
        "prediction_root": manifests[0].get("prediction_root"),
        "prediction_manifest_path": manifests[0].get(
            "prediction_manifest_path"
        ),
        "prediction_manifest_sha256": manifests[0].get(
            "prediction_manifest_sha256"
        ),
        "prediction_identity_sha256": manifests[0].get(
            "prediction_identity_sha256"
        ),
        "frozen_split_manifest_path": manifests[0].get(
            "frozen_split_manifest_path"
        ),
        "frozen_split_manifest_sha256": manifests[0].get(
            "frozen_split_manifest_sha256"
        ),
        "query_metadata_path": manifests[0].get("query_metadata_path"),
        "query_metadata_sha256": manifests[0].get(
            "query_metadata_sha256"
        ),
        "query_metadata_manifest_path": manifests[0].get(
            "query_metadata_manifest_path"
        ),
        "query_metadata_manifest_sha256": manifests[0].get(
            "query_metadata_manifest_sha256"
        ),
        "query_metadata_identity_sha256": manifests[0].get(
            "query_metadata_identity_sha256"
        ),
        "candidate_protocol_family_identity_sha256": manifests[0].get(
            "candidate_protocol_family_identity_sha256"
        ),
        "scorer_model_identity": manifests[0].get("scorer_model_identity"),
        "scorer_model_identity_sha256": manifests[0].get(
            "scorer_model_identity_sha256"
        ),
        "inference_rows_sha256_before_label_join": stable_frame_rows_sha256(
            inference_only
        ),
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
        "source_shards": [
            {
                "path": str(root),
                "dataset_manifest_sha256": sha256_file(
                    root / "dataset_manifest.json"
                ),
                "sample_count": source_manifest["sample_count"],
                "candidate_count": source_manifest["candidate_count"],
                "shard_index": source_manifest["shard_index"],
                "candidate_root": source_manifest.get("candidate_root"),
                "scored_root": source_manifest.get("scored_root"),
                "labels_source": source_manifest.get("labels_source"),
                "labels_source_sha256": source_manifest.get(
                    "labels_source_sha256"
                ),
                "candidate_protocol_identity_sha256": source_manifest.get(
                    "candidate_protocol_identity_sha256"
                ),
                "candidate_run_config_sha256": source_manifest.get(
                    "candidate_run_config_sha256"
                ),
                "scorer_run_config_sha256": source_manifest.get(
                    "scorer_run_config_sha256"
                ),
                "scorer_source_manifest_sha256": source_manifest.get(
                    "scorer_source_manifest_sha256"
                ),
                "scoring_verification_sha256": source_manifest.get(
                    "scoring_verification_sha256"
                ),
                "completion_attestation": completion,
            }
            for root, source_manifest, completion in zip(
                shard_roots,
                manifests,
                completion_attestations,
                strict=True,
            )
        ],
        "disclaimer": (
            "collision/clearance fields are visible-surface proxies from observed "
            "depth, not a complete collision guarantee"
        ),
        "storage_budget_preflight": budget_preflight,
    }
    atomic_json(output_root / "dataset_manifest.json", manifest)
    published_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(output_root, published_root)
    print(
        f"merged samples={len(sample_frame)} candidates={len(candidate_frame)} "
        f"output={published_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
