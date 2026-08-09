from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.grasping.reranking_v1.features import (
    FORBIDDEN_GT_COLUMNS,
    INFERENCE_FEATURE_ALLOWLIST,
    feature_schema,
)
from src.grasping.reranking_v1.identity import sha256_file
from src.grasping.reranking_v1.identity import candidate_identity_sha256
from src.grasping.gqcnn_full_scoring import (
    deterministic_save_npz,
    load_source_sample,
    score_and_write_sample,
    source_manifest_entry,
)
from src.grasping.geometric_ranker import load_frozen_candidates
from tools.modular_reranking.compact_gqcnn_scores import (
    SCORE_SCHEMA,
    _validate_write_scope as validate_score_write_scope,
)
from tools.modular_reranking.merge_compact_candidate_shards import (
    STAGE_FILES,
    main as merge_candidate_shards_main,
)
from tools.modular_reranking.merge_compact_gqcnn_score_shards import (
    canonical_json_sha256,
    main as merge_score_shards_main,
)
from tools.modular_reranking.merge_feature_shards import (
    _attest_sample_shard_set,
    _completion_attestation,
    main as merge_feature_shards_main,
)
from tools.modular_reranking.extract_candidate_features import (
    OFFICIAL_SCORER_MODEL_IDENTITY,
    _validate_scene_streaming_roots,
)
from tools.modular_reranking.prune_streamed_scene_shard_verbose import (
    main as prune_scene_shard_main,
    validate_scopes as validate_prune_scopes,
)
from tools.modular_reranking.scene_sharding import (
    SCENE_SHARD_ASSIGNMENT,
    scene_shard_index,
)


def _scene_for_bucket(bucket: int, *, num_shards: int = 2) -> str:
    return next(
        f"scene-{index}"
        for index in range(10_000)
        if scene_shard_index(f"scene-{index}", num_shards=num_shards) == bucket
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_scene_assignment_is_exact_generator_full_digest_modulo() -> None:
    for scene_id in ("scene-a", "scene-b", "abc,frame.png", "你好"):
        expected = int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 8
        assert scene_shard_index(scene_id, num_shards=8) == expected


def test_scene_score_storage_is_tmp_only_while_default_is_persistent(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    tmp_root = run / "tmp"
    tmp_root.mkdir(parents=True)
    (run / ".RUN_ACTIVE").write_text("\n", encoding="utf-8")
    ephemeral = tmp_root / "scores" / "shard_0.parquet"
    persistent = run / "candidate_tables" / "train" / "gqcnn_scores.parquet"
    assert validate_score_write_scope(
        ephemeral, tmp_root, ephemeral_scene_shard=True
    )[0] == ephemeral.resolve()
    assert validate_score_write_scope(persistent, tmp_root)[0] == persistent.resolve()
    with pytest.raises(ValueError, match="must be stored below"):
        validate_score_write_scope(
            persistent, tmp_root, ephemeral_scene_shard=True
        )
    with pytest.raises(ValueError, match="cannot be stored below"):
        validate_score_write_scope(ephemeral, tmp_root)


def _candidate_merge_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[Path]]:
    run = tmp_path / "run"
    tmp_root = run / "tmp"
    tmp_root.mkdir(parents=True)
    (run / ".RUN_ACTIVE").write_text("\n", encoding="utf-8")
    prediction_manifest = run / "prediction.jsonl"
    rows = [
        {
            "sample_index": bucket,
            "sample_id": f"sample-{bucket}",
            "question_index": bucket,
            "scene_id": _scene_for_bucket(bucket),
            "query": "grasp it",
            "split": "train",
        }
        for bucket in range(2)
    ]
    prediction_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    prediction_sha = sha256_file(prediction_manifest)
    shard_roots: list[Path] = []
    schema = pa.schema([("sample_id", pa.string()), ("candidate_id", pa.string())])
    for bucket, row in enumerate(rows):
        root = tmp_root / "candidates" / f"shard_{bucket}"
        root.mkdir(parents=True)
        shard_roots.append(root)
        artifacts = {}
        counts = {"samples": 1, "execution_failures": 0}
        for stage, filename in STAGE_FILES.items():
            path = root / filename
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "sample_id": row["sample_id"],
                            "candidate_id": "g0000",
                        }
                    ],
                    schema=schema,
                ),
                path,
                compression="zstd",
            )
            artifacts[stage] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": 1,
                "primary_key": ["sample_id", "candidate_id"],
            }
            counts[
                {
                    "raw": "raw_candidates",
                    "mask_validated": "mask_validated_candidates",
                    "nms": "nms_candidates",
                }[stage]
            ] = 1
        counts.update(
            {
                "nonempty_samples": 1,
                "empty_samples": 0,
                "raw_oracle": 1,
                "mask_validated_oracle": 1,
                "nms_oracle": 1,
            }
        )
        with (root / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "sample_id",
                    "status",
                    "raw_candidate_count",
                    "mask_validated_count",
                    "post_nms_count",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "status": "success_nonempty",
                    "raw_candidate_count": 1,
                    "mask_validated_count": 1,
                    "post_nms_count": 1,
                }
            )
        (root / "funnel_labels.jsonl").write_text(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "raw_candidate_count": 1,
                    "mask_validated_candidate_count": 1,
                    "nms_candidate_count": 1,
                    "raw_oracle": True,
                    "mask_validated_oracle": True,
                    "nms_oracle": True,
                    "valid_empty": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run_config = {
            "schema_version": 2,
            "status": "COMPLETED",
            "split": "train",
            "counts": counts,
            "prediction_manifest": str(prediction_manifest.resolve()),
            "prediction_manifest_sha256": prediction_sha,
            "annotations_sha256": "annotations",
            "config_sha256": "config",
            "configuration_hash": "configuration",
            "evaluation_config_sha256": "evaluation",
            "sample_seed_mode": "stable-sha256",
            "seed_namespace": "namespace",
            "protocol_family_identity": {"family": "same"},
            "protocol_family_identity_sha256": "f" * 64,
            "protocol_identity_sha256": str(bucket) * 64,
            "candidate_stage_schema_sha256": "s" * 64,
            "candidate_stage_artifacts": artifacts,
            "fresh_samples": 1,
            "elapsed_seconds": 1.0,
            "selection": {
                "num_shards": 2,
                "shard_index": bucket,
                "scene_grouped": True,
                "limit": None,
            },
        }
        _write_json(root / "run_config.json", run_config)
        receipt = (
            run
            / "manifests"
            / "streaming_cleanup"
            / "train"
            / f"shard_{bucket}.json"
        )
        receipt.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            receipt,
            {
                "status": "COMPLETED",
                "executed": True,
                "candidate_root": str(root.resolve()),
                "artifacts": {
                    "candidate_run_config_sha256": sha256_file(
                        root / "run_config.json"
                    ),
                    "summary_sha256": sha256_file(root / "summary.csv"),
                    "funnel_labels_sha256": sha256_file(
                        root / "funnel_labels.jsonl"
                    ),
                    "candidate_stage_sha256": {
                        stage: artifacts[stage]["sha256"]
                        for stage in STAGE_FILES
                    },
                },
            },
        )
    return run, tmp_root, prediction_manifest, shard_roots


def test_candidate_compact_only_merge_accepts_pruned_verbose_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tmp_root, prediction_manifest, shard_roots = _candidate_merge_fixture(
        tmp_path
    )
    output = tmp_root / "candidates" / "compact_merged"
    argv = [
        "merge_compact_candidate_shards.py",
        "--prediction-manifest",
        str(prediction_manifest),
        "--output-root",
        str(output),
        "--tmp-root",
        str(tmp_root),
        "--compact-only",
    ]
    for root in shard_roots:
        argv.extend(["--shard-root", str(root)])
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_candidate_shards_main() == 0
    config = json.loads((output / "run_config.json").read_text())
    assert config["scorer_compatible"] is False
    assert config["merge_storage"]["verbose_per_sample_tree_retained"] is False
    assert not (output / "_labels").exists()
    assert all(
        pq.ParquetFile(output / filename).metadata.num_rows == 2
        for filename in STAGE_FILES.values()
    )

    noncompact = tmp_root / "candidates" / "noncompact"
    noncompact_argv = [
        value if value != str(output) else str(noncompact) for value in argv
    ]
    noncompact_argv.remove("--compact-only")
    monkeypatch.setattr(sys, "argv", noncompact_argv)
    with pytest.raises(FileNotFoundError, match="sample directory"):
        merge_candidate_shards_main()


def _score_row(
    prediction: dict[str, object], *, candidate_id: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "pipeline": "hierarchical_repeated_film",
        "split": prediction["split"],
        "sample_index": prediction["sample_index"],
        "sample_id": prediction["sample_id"],
        "question_index": prediction["question_index"],
        "scene_id": prediction["scene_id"],
        "candidate_id": candidate_id,
        "candidate_identity_sha256": "i" * 64,
        "gqcnn_q_value": 0.75,
        "gqcnn_rank": 1,
        "source_candidate_index": 0,
        "source_candidates_npz_sha256": "a" * 64,
        "scored_candidates_npz_sha256": "b" * 64,
        "model_name": "GQCNN-2.1",
        "model_commit": "commit",
        "model_config_sha256": "c" * 64,
    }


def _score_merge_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[Path], list[dict[str, object]]]:
    run = tmp_path / "run"
    tmp_root = run / "tmp"
    tmp_root.mkdir(parents=True)
    (run / ".RUN_ACTIVE").write_text("\n", encoding="utf-8")
    rows = [
        {
            "sample_index": bucket,
            "sample_id": f"sample-{bucket}",
            "question_index": bucket + 10,
            "scene_id": _scene_for_bucket(bucket),
            "query": "grasp it",
            "split": "train",
        }
        for bucket in range(2)
    ]
    prediction_manifest = run / "prediction.jsonl"
    prediction_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    prediction_sha = sha256_file(prediction_manifest)
    schema_sha = canonical_json_sha256(
        [
            (field.name, str(field.type), field.nullable)
            for field in SCORE_SCHEMA
        ]
    )
    model = {
        "model_name": "GQCNN-2.1",
        "model_commit": "commit",
        "model_config_hash": "c" * 64,
    }
    score_paths: list[Path] = []
    for bucket, row in enumerate(rows):
        root = tmp_root / "compact_scores" / f"shard_{bucket}"
        root.mkdir(parents=True)
        score_path = root / "gqcnn_scores.parquet"
        score_paths.append(score_path)
        pq.write_table(
            pa.Table.from_pylist(
                [_score_row(row, candidate_id=f"g{bucket:04d}")],
                schema=SCORE_SCHEMA,
            ),
            score_path,
            compression="zstd",
        )
        _write_json(
            score_path.with_suffix(".manifest.json"),
            {
                "schema_version": 1,
                "status": "COMPLETED",
                "pipeline": "hierarchical_repeated_film",
                "split": "train",
                "primary_key": ["sample_id", "candidate_id"],
                "gt_free": True,
                "samples": 1,
                "full_prediction_samples": 2,
                "nonempty_samples": 1,
                "empty_samples": 0,
                "rows": 1,
                "prediction_manifest_sha256": prediction_sha,
                "model": model,
                "schema_sha256": schema_sha,
                "gqcnn_scores_parquet": str(score_path.resolve()),
                "gqcnn_scores_parquet_sha256": sha256_file(score_path),
                "candidate_protocol_identity_sha256": str(bucket) * 64,
                "candidate_protocol_family_identity_sha256": "f" * 64,
                "independent_verification_sha256": "v" * 64,
                "storage": {"ephemeral_scene_shard": True},
                "partition": {
                    "assignment": SCENE_SHARD_ASSIGNMENT,
                    "num_shards": 2,
                    "shard_index": bucket,
                    "scene_grouped": True,
                },
            },
        )
    return run, tmp_root, prediction_manifest, score_paths, rows


def test_compact_score_shard_merge_replays_partition_and_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, tmp_root, prediction_manifest, score_paths, rows = _score_merge_fixture(
        tmp_path
    )
    output = run / "compact_inputs" / "train" / "gqcnn_scores.parquet"
    argv = [
        "merge_compact_gqcnn_score_shards.py",
        "--split",
        "train",
        "--prediction-manifest",
        str(prediction_manifest),
        "--output-path",
        str(output),
        "--tmp-root",
        str(tmp_root),
    ]
    for path in reversed(score_paths):
        argv.extend(["--shard-score", str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_score_shards_main() == 0
    frame = pq.read_table(output).to_pandas()
    assert frame["sample_id"].tolist() == [
        str(row["sample_id"]) for row in rows
    ]
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text()
    )
    assert manifest["partition_merge"]["complete_shard_set"] is True
    assert manifest["rows"] == 2


def _feature_shard(
    root: Path,
    *,
    shard_index: int,
    sample_id: str,
    scene_id: str,
) -> None:
    root.mkdir(parents=True)
    candidate_root = root.parent / "_candidate_attestation" / f"shard_{shard_index}"
    scored_root = root.parent / "_scorer_attestation" / f"shard_{shard_index}"
    candidate_sample = candidate_root / sample_id
    scored_sample = scored_root / sample_id
    candidate_sample.mkdir(parents=True)
    scored_sample.mkdir(parents=True)
    candidate_record = {
        "sample_id": sample_id,
        "candidate_id": "g0000",
        "center_u_px": 10.0,
        "center_v_px": 20.0,
        "center_depth_m": 1.0,
        "center_camera_xyz_m": [0.0, 0.0, 1.0],
        "angle_rad": 0.0,
        "angle_deg": 0.0,
        "width_m": 0.05,
        "width_px": 20.0,
        "endpoint_1_uv": [0.0, 20.0],
        "endpoint_2_uv": [20.0, 20.0],
        "contact_points_uv": [[0.0, 20.0], [20.0, 20.0]],
        "contact_normals": [[1.0, 0.0], [-1.0, 0.0]],
        "grasp_axis_mask_support": 1.0,
        "centre_boundary_distance_px": 10.0,
        "centre_inside_mask": True,
        "rejection_reason": None,
        "T_camera_grasp_fixed_approach": np.eye(4).tolist(),
    }
    arrays = {
        "center_uv": np.asarray([[10.0, 20.0]], dtype=np.float32),
        "center_depth_m": np.asarray([1.0], dtype=np.float32),
        "center_camera_xyz_m": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        "angle_rad": np.asarray([0.0], dtype=np.float32),
        "width_m": np.asarray([0.05], dtype=np.float32),
        "width_px": np.asarray([20.0], dtype=np.float32),
        "endpoints_uv": np.asarray([[[0.0, 20.0], [20.0, 20.0]]], dtype=np.float32),
        "T_camera_grasp_fixed_approach": np.asarray([np.eye(4)], dtype=np.float64),
        "gqcnn_q_value": np.asarray([np.nan], dtype=np.float32),
        "mask_support": np.asarray([1.0], dtype=np.float32),
        "boundary_distance_px": np.asarray([10.0], dtype=np.float32),
        "valid": np.asarray([True]),
    }
    deterministic_save_npz(candidate_sample / "candidates.npz", arrays)
    _write_json(
        candidate_sample / "candidates.json",
        {
            "metadata": {"sample_id": sample_id},
            "candidates": [candidate_record],
        },
    )
    _write_json(
        candidate_sample / "metadata.json",
        {
            "sample_id": sample_id,
            "question_index": shard_index,
            "query": "grasp it",
            "representation": "planar_parallel_jaw_4dof",
            "approach_constraint": "fixed_camera_optical_axis",
            "camera_frame": "ocid_camera_optical",
            "counts": {"post_nms": 1},
            "failure_reason": None,
        },
    )
    (candidate_sample / "camera.intr").write_text("test intrinsics\n")
    np.save(candidate_sample / "depth_m.npy", np.ones((2, 2), dtype=np.float32))
    (candidate_sample / "hifics_mask_processed.png").write_bytes(b"test-mask")
    required = {
        name: sha256_file(candidate_sample / name)
        for name in (
            "candidates.npz",
            "candidates.json",
            "metadata.json",
            "camera.intr",
            "depth_m.npy",
            "hifics_mask_processed.png",
        )
    }
    _write_json(
        candidate_sample / "_SUCCESS.json",
        {
            "schema_version": 1,
            "sample_id": sample_id,
            "status": "success_nonempty",
            "candidate_counts": {"post_nms": 1},
            "required_files": list(required),
            "required_file_hashes": required,
            "configuration_hash": "candidate-configuration",
            "config_file_sha256": "candidate-config",
            "seed": 42,
            "sampler_commit": "sampler-commit",
            "failure_reason": None,
        },
    )
    source = load_source_sample(candidate_sample, verify_hashes=True)
    entry = source_manifest_entry(source, 0)
    source_manifest = scored_root / "source_candidate_manifest.jsonl"
    source_manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    model = dict(OFFICIAL_SCORER_MODEL_IDENTITY)

    class _Quality:
        def __call__(self, state, grasps, params=None):
            del state, grasps, params
            return np.asarray([0.5], dtype=np.float64)

    def _state_builder(sample_dir, score_arrays, records, factor):
        del sample_dir, score_arrays, records, factor
        return object(), [object()], {"frame": "ocid_camera_optical"}

    score_and_write_sample(
        scored_sample,
        candidate_sample,
        entry,
        _Quality(),
        _state_builder,
        model,
        seed=42,
    )
    score_marker = json.loads(
        (scored_sample / "_SCORING_COMPLETE.json").read_text(encoding="utf-8")
    )
    identity_records, _, _ = load_frozen_candidates(
        candidate_sample / "candidates.npz", candidate_sample / "candidates.json"
    )
    candidate_identity = candidate_identity_sha256(identity_records[0])
    candidate_frame = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "candidate_id": "g0000",
                "split": "train",
                "query_type": "name",
                "original_gqcnn_rank": 1,
                "candidate_identity_sha256": candidate_identity,
                **{
                    feature: 0.5
                    for feature in INFERENCE_FEATURE_ALLOWLIST
                },
                "candidate_positive": True,
                "candidate_gt_iou": 0.5,
                "candidate_gt_angle_error": 5.0,
            }
        ]
    )
    sample_frame = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "candidate_count": 1,
            }
        ]
    )
    candidate_frame.to_parquet(root / "per_candidate.parquet", index=False)
    sample_frame.to_parquet(root / "per_sample.parquet", index=False)
    _write_json(root / "feature_schema.json", feature_schema(candidate_frame))
    _write_json(
        root / "inference_feature_allowlist.json",
        {"schema_version": 1, "features": list(INFERENCE_FEATURE_ALLOWLIST)},
    )
    _write_json(
        root / "forbidden_gt_columns.json",
        {"schema_version": 1, "columns": list(FORBIDDEN_GT_COLUMNS)},
    )
    _write_json(
        root / "feature_statistics_train_only.json",
        {"schema_version": 1, "source_split": "train"},
    )
    prediction_sha = "p" * 64
    _write_json(
        candidate_root / "run_config.json",
        {
            "status": "COMPLETED",
            "split": "train",
            "prediction_manifest_sha256": prediction_sha,
            "selection": {
                "scene_grouped": True,
                "limit": None,
                "num_shards": 2,
                "shard_index": shard_index,
            },
            "counts": {"samples": 1, "nms_candidates": 1},
            "protocol_identity_sha256": str(shard_index) * 64,
            "protocol_family_identity_sha256": "f" * 64,
        },
    )
    source_sha = sha256_file(source_manifest)
    _write_json(
        scored_root / "run_config.json",
        {
            "source_identity": {
                "source_manifest_sha256": source_sha,
                "samples": 1,
                "candidate_count": 1,
            },
            "model": model,
            "seed": 42,
        },
    )
    _write_json(
        scored_root / "verification_report.json",
        {
            "clean": True,
            "candidate_root": str(candidate_root.resolve()),
            "scored_root": str(scored_root.resolve()),
            "source_manifest_sha256": source_sha,
            "model_name": model["model_name"],
            "model_commit": model["model_commit"],
            "model_config_hash": model["model_config_hash"],
            "model_file_manifest_hash": model["model_file_manifest_hash"],
        },
    )
    provenance = _validate_scene_streaming_roots(
        candidate_root=candidate_root,
        scored_root=scored_root,
        prediction_manifest_sha256=prediction_sha,
        split="train",
        num_shards=2,
        shard_index=shard_index,
    )
    manifest = {
        "schema_version": 1,
        "status": "COMPLETED",
        "split": "train",
        "num_shards": 2,
        "shard_index": shard_index,
        "shard_assignment": SCENE_SHARD_ASSIGNMENT,
        "streaming_scene_shard": True,
        "sample_count": 1,
        "candidate_count": 1,
        "labels_joined_post_extraction": True,
        "labels_source": f"/labels/shard_{shard_index}",
        "labels_source_sha256": str(shard_index + 1) * 64,
        "labels_source_file_count": 1,
        "per_candidate_sha256": sha256_file(root / "per_candidate.parquet"),
        "per_sample_sha256": sha256_file(root / "per_sample.parquet"),
        "feature_schema_sha256": sha256_file(root / "feature_schema.json"),
        "allowlist_sha256": sha256_file(
            root / "inference_feature_allowlist.json"
        ),
        "forbidden_columns_sha256": sha256_file(
            root / "forbidden_gt_columns.json"
        ),
        "statistics_sha256": sha256_file(
            root / "feature_statistics_train_only.json"
        ),
        "annotations_path": "/query/manifest.json",
        "annotations_sha256": "query-manifest",
        "prediction_root": "/predictions",
        "prediction_manifest_sha256": prediction_sha,
        "prediction_identity_sha256": "prediction-identity",
        "frozen_split_manifest_sha256": "frozen",
        "query_metadata_identity_sha256": "query",
        "candidate_root": str(candidate_root.resolve()),
        "scored_root": str(scored_root.resolve()),
        "sources": [
            {
                "sample_id": sample_id,
                "candidates_npz_sha256": source["source_hashes"][
                    "candidates_npz_sha256"
                ],
                "candidates_json_sha256": source["source_hashes"][
                    "candidates_json_sha256"
                ],
                "depth_m_sha256": source["source_hashes"]["depth_m_sha256"],
                "mask_sha256": source["source_hashes"][
                    "processed_mask_sha256"
                ],
                "intrinsics_sha256": source["source_hashes"][
                    "camera_intrinsics_sha256"
                ],
                "probability_sha256": "a" * 64,
                "scored_npz_sha256": score_marker["required_file_hashes"][
                    "gqcnn_scored_candidates.npz"
                ],
                "scoring_status": "scored_nonempty",
            }
        ],
        **provenance,
    }
    _write_json(root / "dataset_manifest.json", manifest)


def _legacy_feature_completion_fixture(
    tmp_path: Path,
) -> tuple[Path, Path]:
    feature_root = tmp_path / "features" / "shard_0"
    _feature_shard(
        feature_root,
        shard_index=0,
        sample_id="sample-0",
        scene_id=_scene_for_bucket(0),
    )
    manifest_path = feature_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("status")
    _write_json(manifest_path, manifest)
    verification_path = Path(manifest["scored_root"]) / "verification_report.json"
    return feature_root, verification_path


def test_legacy_feature_completion_requires_independent_source_attestation(
    tmp_path: Path,
) -> None:
    feature_root, verification_path = _legacy_feature_completion_fixture(tmp_path)
    manifest = json.loads(
        (feature_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )

    attestation = _completion_attestation(feature_root, manifest)

    assert attestation["mode"] == "INDEPENDENT_STREAMING_COMPLETION_ATTESTATION"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["clean"] = False
    _write_json(verification_path, verification)
    with pytest.raises(ValueError, match="cannot be attested"):
        _completion_attestation(feature_root, manifest)


def test_declared_streaming_completion_rejects_fake_nonexistent_provenance(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features" / "shard_0"
    _feature_shard(
        feature_root,
        shard_index=0,
        sample_id="sample-0",
        scene_id=_scene_for_bucket(0),
    )
    manifest_path = feature_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "candidate_root": str(tmp_path / "nonexistent-candidates"),
            "scored_root": str(tmp_path / "nonexistent-scores"),
            "scorer_model_identity": "m" * 64,
            "scorer_model_identity_sha256": "m" * 64,
        }
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="cannot be attested"):
        _completion_attestation(feature_root, manifest)


def test_streaming_completion_rejects_live_but_nonofficial_scorer(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features" / "shard_0"
    _feature_shard(
        feature_root,
        shard_index=0,
        sample_id="sample-0",
        scene_id=_scene_for_bucket(0),
    )
    manifest = json.loads(
        (feature_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    scorer_config_path = Path(manifest["scored_root"]) / "run_config.json"
    scorer_config = json.loads(scorer_config_path.read_text(encoding="utf-8"))
    scorer_config["model"]["model_commit"] = "0" * 40
    _write_json(scorer_config_path, scorer_config)

    with pytest.raises(ValueError, match="cannot be attested"):
        _completion_attestation(feature_root, manifest)


def test_streaming_completion_rejects_metadata_only_roots_and_arbitrary_q(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features" / "shard_0"
    _feature_shard(
        feature_root,
        shard_index=0,
        sample_id="sample-0",
        scene_id=_scene_for_bucket(0),
    )
    manifest_path = feature_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature = pd.read_parquet(feature_root / "per_candidate.parquet")
    feature.loc[:, "q_raw"] = 0.999
    feature.to_parquet(feature_root / "per_candidate.parquet", index=False)
    manifest["per_candidate_sha256"] = sha256_file(
        feature_root / "per_candidate.parquet"
    )
    _write_json(manifest_path, manifest)
    shutil.rmtree(Path(manifest["candidate_root"]) / "sample-0")
    shutil.rmtree(Path(manifest["scored_root"]) / "sample-0")

    with pytest.raises(ValueError, match="score binding cannot be attested"):
        _completion_attestation(feature_root, manifest)


@pytest.mark.parametrize("tamper", ["scored_npz", "completion_marker"])
def test_streaming_completion_rejects_live_score_payload_tamper(
    tmp_path: Path, tamper: str
) -> None:
    feature_root = tmp_path / "features" / "shard_0"
    _feature_shard(
        feature_root,
        shard_index=0,
        sample_id="sample-0",
        scene_id=_scene_for_bucket(0),
    )
    manifest = json.loads(
        (feature_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    scored_sample = Path(manifest["scored_root"]) / "sample-0"
    if tamper == "scored_npz":
        with (scored_sample / "gqcnn_scored_candidates.npz").open("ab") as stream:
            stream.write(b"tamper")
    else:
        marker_path = scored_sample / "_SCORING_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["seed"] = 43
        _write_json(marker_path, marker)

    with pytest.raises(ValueError, match="official score verification failed"):
        _completion_attestation(feature_root, manifest)


def test_streaming_completion_rejects_feature_q_raw_mismatch(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features" / "shard_0"
    _feature_shard(
        feature_root,
        shard_index=0,
        sample_id="sample-0",
        scene_id=_scene_for_bucket(0),
    )
    manifest_path = feature_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature = pd.read_parquet(feature_root / "per_candidate.parquet")
    feature.loc[:, "q_raw"] = 0.625
    feature.to_parquet(feature_root / "per_candidate.parquet", index=False)
    manifest["per_candidate_sha256"] = sha256_file(
        feature_root / "per_candidate.parquet"
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="differ semantically"):
        _completion_attestation(feature_root, manifest)


def test_mixed_legacy_and_declared_sample_shards_require_full_prediction_universe(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(
        '{"sample_id":"sample-0"}\n{"sample_id":"sample-1"}\n',
        encoding="utf-8",
    )
    prediction_sha = sha256_file(prediction)
    common = {
        "streaming_scene_shard": False,
        "prediction_manifest_path": str(prediction.resolve()),
        "prediction_manifest_sha256": prediction_sha,
        "prediction_identity_sha256": "8" * 64,
        "frozen_split_manifest_sha256": "f" * 64,
        "per_candidate_sha256": "a" * 64,
        "per_sample_sha256": "b" * 64,
        "feature_schema_sha256": "c" * 64,
        "allowlist_sha256": "d" * 64,
        "forbidden_columns_sha256": "e" * 64,
        "statistics_sha256": "9" * 64,
    }
    roots = [tmp_path / f"feature-{index}" for index in range(2)]
    manifests = [{**common}, {**common, "status": "COMPLETED"}]
    attestations = []
    for root, manifest in zip(roots, manifests, strict=True):
        root.mkdir()
        _write_json(root / "dataset_manifest.json", manifest)
        attestations.append(_completion_attestation(root, manifest))
    samples = pd.DataFrame(
        {"sample_id": ["sample-0", "sample-1"], "candidate_count": [0, 1]}
    )

    _attest_sample_shard_set(
        manifests=manifests,
        sample_frame=samples,
        attestations=attestations,
    )

    assert (
        attestations[0]["mode"]
        == "INDEPENDENT_SAMPLE_SET_COMPLETION_ATTESTATION"
    )
    invalid = {**common, "prediction_identity_sha256": "not-a-sha256"}
    _write_json(roots[0] / "dataset_manifest.json", invalid)
    with pytest.raises(ValueError, match="invalid SHA-256"):
        _completion_attestation(roots[0], invalid)
    with pytest.raises(ValueError, match="exactly cover prediction manifest"):
        _attest_sample_shard_set(
            manifests=manifests,
            sample_frame=samples.iloc[[1]],
            attestations=[
                {**attestations[0], "mode": "PENDING_COMPLETE_SET_ATTESTATION"},
                attestations[1],
            ],
        )


def test_feature_merge_accepts_verified_scene_shards_with_distinct_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    tmp_root = run / "tmp"
    tmp_root.mkdir(parents=True)
    (run / ".RUN_ACTIVE").write_text("\n", encoding="utf-8")
    roots = [tmp_root / "features" / f"shard_{index}" for index in range(2)]
    for index, root in enumerate(roots):
        _feature_shard(
            root,
            shard_index=index,
            sample_id=f"sample-{index}",
            scene_id=_scene_for_bucket(index),
        )
    output = run / "features" / "train"
    argv = [
        "merge_feature_shards.py",
        "--output-root",
        str(output),
        "--tmp-root",
        str(tmp_root),
        "--split",
        "train",
    ]
    for root in roots:
        argv.extend(["--shard-root", str(root)])
    monkeypatch.setattr(sys, "argv", argv)
    assert merge_feature_shards_main() == 0
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["streaming_scene_shards"] is True
    assert manifest["candidate_root"] is None
    assert manifest["scored_root"] is None
    assert manifest["labels_source_file_count"] == 2
    assert manifest["prediction_manifest_sha256"] == "p" * 64
    assert manifest["prediction_identity_sha256"] == "prediction-identity"
    assert manifest["frozen_split_manifest_sha256"] == "frozen"
    assert manifest["query_metadata_identity_sha256"] == "query"
    assert manifest["candidate_protocol_family_identity_sha256"] == "f" * 64
    assert manifest["scorer_model_identity_sha256"] == json.loads(
        (roots[0] / "dataset_manifest.json").read_text(encoding="utf-8")
    )["scorer_model_identity_sha256"]
    assert {
        item["candidate_root"] for item in manifest["source_shards"]
    } == {
        json.loads(
            (root / "dataset_manifest.json").read_text(encoding="utf-8")
        )["candidate_root"]
        for root in roots
    }
    assert {
        item["completion_attestation"]["mode"]
        for item in manifest["source_shards"]
    } == {"INDEPENDENT_STREAMING_COMPLETION_ATTESTATION"}


def _prune_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Path], str]:
    run = tmp_path / "run"
    tmp_root = run / "tmp"
    candidate_root = tmp_root / "train" / "candidates" / "shard_0"
    scored_root = tmp_root / "train" / "scores" / "shard_0"
    feature_root = tmp_root / "train" / "features" / "shard_0"
    compact_score = (
        tmp_root / "train" / "gqcnn_score_shards" / "shard_0.parquet"
    )
    for path in (candidate_root, scored_root, feature_root, compact_score.parent):
        path.mkdir(parents=True, exist_ok=True)
    (run / ".RUN_ACTIVE").write_text("\n", encoding="utf-8")
    sample_id = "sample-0"
    for path in (
        candidate_root / sample_id,
        candidate_root / "_scene_cache",
        scored_root / sample_id,
    ):
        path.mkdir()
        (path / "payload.bin").write_bytes(b"verbose")
    (candidate_root / "_labels").mkdir()
    artifacts = {}
    for stage, filename in STAGE_FILES.items():
        path = candidate_root / filename
        pq.write_table(
            pa.table({"sample_id": [sample_id], "candidate_id": ["g0000"]}),
            path,
            compression="zstd",
        )
        artifacts[stage] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": 1,
        }
    candidate_config = {
        "status": "COMPLETED",
        "split": "train",
        "counts": {
            "samples": 1,
            "nonempty_samples": 1,
            "empty_samples": 0,
            "raw_candidates": 1,
            "mask_validated_candidates": 1,
            "nms_candidates": 1,
            "raw_oracle": 1,
            "mask_validated_oracle": 1,
            "nms_oracle": 1,
            "execution_failures": 0,
        },
        "protocol_identity_sha256": "p" * 64,
        "candidate_stage_artifacts": artifacts,
        "selection": {
            "num_shards": 1,
            "shard_index": 0,
            "scene_grouped": True,
            "limit": None,
        },
    }
    _write_json(candidate_root / "run_config.json", candidate_config)
    prediction = {
        "sample_index": 0,
        "sample_id": sample_id,
        "question_index": 1,
        "scene_id": _scene_for_bucket(0, num_shards=1),
        "query": "item",
        "split": "train",
    }
    (candidate_root / "run_manifest.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8"
    )
    (candidate_root / "summary.csv").write_text(
        "sample_id,query,raw_candidate_count,mask_validated_count,"
        "post_nms_count,status,question_index,scene_id\n"
        f'{sample_id},item,1,1,1,success_nonempty,1,'
        f'"{prediction["scene_id"]}"\n',
        encoding="utf-8",
    )
    funnel = {
        "sample_id": sample_id,
        "question_index": 1,
        "scene_id": prediction["scene_id"],
        "raw_candidate_count": 1,
        "mask_validated_candidate_count": 1,
        "nms_candidate_count": 1,
        "raw_oracle": True,
        "mask_validated_oracle": True,
        "nms_oracle": True,
        "valid_empty": False,
    }
    (candidate_root / "funnel_labels.jsonl").write_text(
        json.dumps(funnel) + "\n", encoding="utf-8"
    )
    _write_json(
        candidate_root / "_labels" / f"{sample_id}.json",
        {
            "schema_version": 2,
            "split": "train",
            "label_only_gt_artifact": True,
            "inference_artifact_dependency": False,
            "protocol_identity_sha256": "p" * 64,
            "sample": funnel,
            "candidate_labels": [
                {
                    "sample_id": sample_id,
                    "candidate_id": "g0000",
                    "candidate_positive": True,
                    "candidate_gt_iou": 0.5,
                    "candidate_gt_angle_error_deg": 10.0,
                }
            ],
        },
    )
    source_manifest = scored_root / "source_candidate_manifest.jsonl"
    source_manifest.write_text(
        json.dumps({"sample_id": sample_id}) + "\n", encoding="utf-8"
    )
    source_manifest_sha = sha256_file(source_manifest)
    _write_json(
        scored_root / "run_config.json",
        {
            "candidate_root": "/candidates",
            "source_identity": {
                "source_manifest_sha256": source_manifest_sha,
                "samples": 1,
                "candidate_count": 1,
            },
        },
    )
    _write_json(
        scored_root / "verification_report.json",
        {
            "clean": True,
            "candidate_root": str(candidate_root.resolve()),
            "scored_root": str(scored_root.resolve()),
            "source_manifest_sha256": source_manifest_sha,
            "expected_total_samples": 1,
            "expected_frozen_candidates": 1,
        },
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_score_row(prediction, candidate_id="g0000")],
            schema=SCORE_SCHEMA,
        ),
        compact_score,
        compression="zstd",
    )
    _write_json(
        compact_score.with_suffix(".manifest.json"),
        {
            "status": "COMPLETED",
            "split": "train",
            "gt_free": True,
            "samples": 1,
            "rows": 1,
            "candidate_root": str(candidate_root.resolve()),
            "scored_root": str(scored_root.resolve()),
            "candidate_protocol_identity_sha256": "p" * 64,
            "candidate_run_manifest_sha256": sha256_file(
                candidate_root / "run_manifest.jsonl"
            ),
            "gqcnn_scores_parquet_sha256": sha256_file(compact_score),
            "independent_verification_sha256": sha256_file(
                scored_root / "verification_report.json"
            ),
            "storage": {"ephemeral_scene_shard": True},
            "partition": {
                "assignment": SCENE_SHARD_ASSIGNMENT,
                "num_shards": 1,
                "shard_index": 0,
                "scene_grouped": True,
            },
        },
    )
    candidate_frame = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "candidate_id": "g0000",
                "candidate_identity_sha256": "i" * 64,
                "q_raw": 0.75,
                "original_gqcnn_rank": 1,
                "candidate_positive": True,
                "candidate_gt_iou": 0.5,
                "candidate_gt_angle_error": 10.0,
            }
        ]
    )
    sample_frame = pd.DataFrame(
        [{"sample_id": sample_id, "candidate_count": 1}]
    )
    candidate_frame.to_parquet(feature_root / "per_candidate.parquet", index=False)
    sample_frame.to_parquet(feature_root / "per_sample.parquet", index=False)
    labels_root = candidate_root / "_labels"
    label_files = sorted(labels_root.glob("*.json"))
    labels_payload = [
        {
            "relative_path": path.relative_to(labels_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in label_files
    ]
    labels_source_sha256 = hashlib.sha256(
        json.dumps(
            labels_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        feature_root / "dataset_manifest.json",
        {
            "split": "train",
            "streaming_scene_shard": True,
            "shard_assignment": SCENE_SHARD_ASSIGNMENT,
            "num_shards": 1,
            "shard_index": 0,
            "candidate_root": str(candidate_root.resolve()),
            "scored_root": str(scored_root.resolve()),
            "candidate_run_config_sha256": sha256_file(
                candidate_root / "run_config.json"
            ),
            "scorer_run_config_sha256": sha256_file(
                scored_root / "run_config.json"
            ),
            "scorer_source_manifest_sha256": source_manifest_sha,
            "scoring_verification_sha256": sha256_file(
                scored_root / "verification_report.json"
            ),
            "scoring_verification_clean": True,
            "labels_joined_post_extraction": True,
            "labels_source": str(labels_root.resolve()),
            "labels_source_sha256": labels_source_sha256,
            "labels_source_file_count": len(label_files),
            "sample_count": 1,
            "candidate_count": 1,
            "per_candidate_sha256": sha256_file(
                feature_root / "per_candidate.parquet"
            ),
            "per_sample_sha256": sha256_file(
                feature_root / "per_sample.parquet"
            ),
        },
    )
    return {
        "run": run,
        "candidate_root": candidate_root,
        "scored_root": scored_root,
        "feature_root": feature_root,
        "compact_score": compact_score,
        "receipt": run / "manifests" / "prune_shard_0.json",
    }, sample_id


def test_prune_requires_downstream_semantic_identity_and_defaults_to_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, sample_id = _prune_fixture(tmp_path)
    argv = [
        "prune_streamed_scene_shard_verbose.py",
        "--split",
        "train",
        "--candidate-root",
        str(paths["candidate_root"]),
        "--scored-root",
        str(paths["scored_root"]),
        "--compact-score",
        str(paths["compact_score"]),
        "--feature-root",
        str(paths["feature_root"]),
        "--receipt-path",
        str(paths["receipt"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert prune_scene_shard_main() == 0
    dry_run = json.loads(paths["receipt"].read_text())
    assert dry_run["status"] == "VERIFIED_DRY_RUN"
    assert (paths["candidate_root"] / sample_id).is_dir()
    assert (paths["scored_root"] / sample_id).is_dir()

    monkeypatch.setattr(sys, "argv", [*argv, "--execute"])
    assert prune_scene_shard_main() == 0
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["status"] == "COMPLETED"
    assert not (paths["candidate_root"] / sample_id).exists()
    assert not (paths["scored_root"] / sample_id).exists()
    assert (paths["candidate_root"] / STAGE_FILES["nms"]).is_file()
    assert sha256_file(paths["compact_score"]) == receipt["artifacts"][
        "compact_score_sha256"
    ]
    assert sha256_file(
        paths["feature_root"] / "per_candidate.parquet"
    ) == receipt["artifacts"]["feature_candidate_sha256"]


def test_prune_rejects_downstream_artifact_nested_in_deletion_tree(
    tmp_path: Path,
) -> None:
    paths, sample_id = _prune_fixture(tmp_path)
    with pytest.raises(ValueError, match="mutually disjoint"):
        validate_prune_scopes(
            candidate_root=paths["candidate_root"],
            scored_root=paths["scored_root"],
            compact_score=(
                paths["candidate_root"] / sample_id / "only_compact.parquet"
            ),
            feature_root=paths["feature_root"],
            receipt_path=paths["receipt"],
        )


def test_prune_rejects_feature_manifest_bound_to_different_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _prune_fixture(tmp_path)
    manifest_path = paths["feature_root"] / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_source_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prune_streamed_scene_shard_verbose.py",
            "--split",
            "train",
            "--candidate-root",
            str(paths["candidate_root"]),
            "--scored-root",
            str(paths["scored_root"]),
            "--compact-score",
            str(paths["compact_score"]),
            "--feature-root",
            str(paths["feature_root"]),
            "--receipt-path",
            str(paths["receipt"]),
        ],
    )
    with pytest.raises(ValueError, match="verified label source"):
        prune_scene_shard_main()
