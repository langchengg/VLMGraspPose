from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.grasping.geometric_ranker import load_frozen_candidates
from src.grasping.reranking_v1.identity import (
    candidate_identity_sha256,
    sha256_file,
)
from tools.modular_reranking.compact_gqcnn_scores import (
    SCORE_SCHEMA,
    main as compact_main,
    validated_score_rows,
)
from tools.modular_reranking.scene_sharding import scene_shard_index


def _record(candidate_id: str) -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "candidate_id": candidate_id,
        "center_uv": [100.0, 200.0],
        "center_u_px": 100.0,
        "center_v_px": 200.0,
        "center_depth_m": 0.7,
        "center_camera_xyz_m": [0.1, 0.2, 0.7],
        # Deliberately not exactly representable as float32. Main compaction
        # must use the same NPZ-canonical identity as feature extraction.
        "angle_rad": 0.2000000001,
        "width_m": 0.05,
        "width_px": 30.0,
        "endpoints_uv": [[85.0, 200.0], [115.0, 200.0]],
        "endpoint_1_uv": [85.0, 200.0],
        "endpoint_2_uv": [115.0, 200.0],
        "contact_points_uv": [[85.0, 200.0], [115.0, 200.0]],
        "contact_normals": [[-1.0, 0.0], [1.0, 0.0]],
        "grasp_axis_mask_support": 0.8,
        "centre_boundary_distance_px": 4.0,
        "gqcnn_q_value": None,
        "rejection_reason": None,
        "centre_inside_mask": True,
        "T_camera_grasp_fixed_approach": np.eye(4).tolist(),
    }


def _rows(
    records: list[dict[str, object]],
    q: list[float],
    ranks: list[int],
) -> list[dict[str, object]]:
    return validated_score_rows(
        sample={
            "sample_index": 4,
            "sample_id": "sample-1",
            "question_index": 8,
            "scene_id": "scene,frame",
        },
        records=records,
        candidate_ids=[str(record["candidate_id"]) for record in records],
        q_values=np.asarray(q),
        ranks=np.asarray(ranks),
        split="train",
        source_candidates_npz_sha256="a" * 64,
        scored_candidates_npz_sha256="b" * 64,
        model={
            "model_name": "GQCNN-2.1",
            "model_commit": "commit",
            "model_config_hash": "c" * 64,
        },
    )


def test_score_schema_is_gt_free_and_has_primary_key() -> None:
    names = set(SCORE_SCHEMA.names)
    assert {"sample_id", "candidate_id", "candidate_identity_sha256"} <= names
    assert {"gqcnn_q_value", "gqcnn_rank"} <= names
    assert not {
        "candidate_positive",
        "candidate_gt_iou",
        "angle_error",
        "gt_grasp",
    } & names


def test_validated_score_rows_preserve_source_order_and_deterministic_rank() -> None:
    records = [_record("g0000"), _record("g0001"), _record("g0002")]
    rows = _rows(records, [0.4, 0.9, 0.4], [2, 1, 3])
    assert [row["candidate_id"] for row in rows] == [
        "g0000",
        "g0001",
        "g0002",
    ]
    assert [row["gqcnn_rank"] for row in rows] == [2, 1, 3]
    assert all(len(str(row["candidate_identity_sha256"])) == 64 for row in rows)


def test_validated_score_rows_reject_id_or_rank_mismatch() -> None:
    records = [_record("g0000"), _record("g0001")]
    with pytest.raises(ValueError, match="order/IDs"):
        validated_score_rows(
            sample={
                "sample_index": 0,
                "sample_id": "sample-1",
                "question_index": 0,
                "scene_id": "scene",
            },
            records=records,
            candidate_ids=["g0001", "g0000"],
            q_values=np.asarray([0.8, 0.2]),
            ranks=np.asarray([1, 2]),
            split="val",
            source_candidates_npz_sha256="a",
            scored_candidates_npz_sha256="b",
            model={
                "model_name": "m",
                "model_commit": "c",
                "model_config_hash": "h",
            },
        )
    with pytest.raises(ValueError, match="rank differs"):
        _rows(records, [0.8, 0.2], [2, 1])


def test_validated_score_rows_reject_nonfinite_and_gt_fields() -> None:
    records = [_record("g0000")]
    with pytest.raises(ValueError, match="finite"):
        _rows(records, [float("nan")], [1])
    leaked = copy.deepcopy(records)
    leaked[0]["candidate_positive"] = True
    with pytest.raises(ValueError, match="GT-derived"):
        _rows(leaked, [0.2], [1])


def test_validated_score_rows_reject_noninteger_rank() -> None:
    records = [_record("g0000"), _record("g0001")]
    with pytest.raises(ValueError, match="finite integers"):
        _rows(records, [0.8, 0.2], [1.0, 1.9])


def _write_compaction_fixture(tmp_path: Path) -> dict[str, Path]:
    run = tmp_path / "protected_run"
    candidate_root = tmp_path / "candidates"
    scored_root = tmp_path / "scores"
    run.mkdir(parents=True)
    candidate_root.mkdir()
    scored_root.mkdir()
    (run / ".RUN_ACTIVE").write_text("test\n", encoding="utf-8")
    rows = [
        {
            "sample_index": 0,
            "sample_id": "sample-1",
            "question_index": 10,
            "scene_id": "scene-a,frame-1",
            "query": "grasp item",
            "split": "test",
        },
        {
            "sample_index": 1,
            "sample_id": "sample-empty",
            "question_index": 11,
            "scene_id": "scene-b,frame-2",
            "query": "grasp empty",
            "split": "test",
        },
    ]
    prediction_manifest = tmp_path / "prediction.jsonl"
    prediction_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (candidate_root / "run_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    model = {
        "model_name": "GQCNN-2.1",
        "model_commit": "commit",
        "model_config_hash": "c" * 64,
    }
    source_rows = []
    scoring_rows = []
    for row, records in zip(
        rows, [[_record("g0000")], []], strict=True
    ):
        sample_id = str(row["sample_id"])
        sample_candidate = candidate_root / sample_id
        sample_score = scored_root / sample_id
        sample_candidate.mkdir()
        sample_score.mkdir()
        if records:
            records[0]["sample_id"] = sample_id
        candidate_json = sample_candidate / "candidates.json"
        candidate_json.write_text(
            json.dumps({"metadata": {}, "candidates": records}),
            encoding="utf-8",
        )
        candidate_npz = sample_candidate / "candidates.npz"
        count = len(records)
        np.savez(
            candidate_npz,
            center_uv=np.asarray(
                [record["center_uv"] for record in records], dtype=np.float32
            ).reshape(count, 2),
            center_depth_m=np.asarray(
                [record["center_depth_m"] for record in records],
                dtype=np.float32,
            ),
            center_camera_xyz_m=np.asarray(
                [record["center_camera_xyz_m"] for record in records],
                dtype=np.float32,
            ).reshape(count, 3),
            angle_rad=np.asarray(
                [record["angle_rad"] for record in records], dtype=np.float32
            ),
            width_m=np.asarray(
                [record["width_m"] for record in records], dtype=np.float32
            ),
            width_px=np.asarray(
                [record["width_px"] for record in records], dtype=np.float32
            ),
            endpoints_uv=np.asarray(
                [record["endpoints_uv"] for record in records],
                dtype=np.float32,
            ).reshape(count, 2, 2),
            mask_support=np.asarray(
                [record["grasp_axis_mask_support"] for record in records],
                dtype=np.float32,
            ),
            boundary_distance_px=np.asarray(
                [record["centre_boundary_distance_px"] for record in records],
                dtype=np.float32,
            ),
            gqcnn_q_value=np.full(count, np.nan, dtype=np.float32),
            valid=np.ones(count, dtype=bool),
            T_camera_grasp_fixed_approach=np.asarray(
                [
                    record["T_camera_grasp_fixed_approach"]
                    for record in records
                ],
                dtype=np.float64,
            ).reshape(count, 4, 4),
        )
        source_row = {
            **{
                name: row[name]
                for name in (
                    "sample_index",
                    "sample_id",
                    "question_index",
                    "query",
                )
            },
            "candidate_count": count,
            "candidate_ids": [
                str(record["candidate_id"]) for record in records
            ],
            "source_hashes": {
                "candidates_npz_sha256": sha256_file(candidate_npz),
                "candidates_json_sha256": sha256_file(candidate_json),
            },
        }
        source_rows.append(source_row)
        status = "scored_nonempty" if count else "skipped_valid_empty"
        score_npz = sample_score / "gqcnn_scored_candidates.npz"
        required_hashes = {}
        if count:
            np.savez(
                score_npz,
                candidate_id=np.asarray(["g0000"], dtype="<U128"),
                gqcnn_q_value=np.asarray([0.75], dtype=np.float64),
                gqcnn_rank=np.asarray([1], dtype=np.int32),
            )
            required_hashes["gqcnn_scored_candidates.npz"] = sha256_file(
                score_npz
            )
        marker = {
            "sample_id": sample_id,
            "scoring_status": status,
            "source_candidate_count": count,
            "gqcnn_scored_count": count,
            "source_candidate_sha256": source_row["source_hashes"][
                "candidates_npz_sha256"
            ],
            "source_candidate_json_sha256": source_row["source_hashes"][
                "candidates_json_sha256"
            ],
            **model,
            "required_file_hashes": required_hashes,
        }
        (sample_score / "_SCORING_COMPLETE.json").write_text(
            json.dumps(marker), encoding="utf-8"
        )
        scoring_rows.append(
            {
                "sample_id": sample_id,
                "scoring_status": status,
                "source_candidate_count": count,
                "gqcnn_scored_count": count,
                "source_candidate_sha256": source_row["source_hashes"][
                    "candidates_npz_sha256"
                ],
                "model_config_hash": model["model_config_hash"],
            }
        )
    source_manifest = scored_root / "source_candidate_manifest.jsonl"
    source_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    (scored_root / "scoring_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scoring_rows),
        encoding="utf-8",
    )
    (scored_root / "run_config.json").write_text(
        json.dumps(
            {
                "model": model,
                "source_identity": {
                    "samples": 2,
                    "candidate_count": 1,
                    "source_manifest_sha256": sha256_file(source_manifest),
                },
            }
        ),
        encoding="utf-8",
    )
    (scored_root / "run_statistics.json").write_text(
        json.dumps(
            {
                "total_samples": 2,
                "terminal_samples": 2,
                "expected_candidates": 1,
                "scored_candidates": 1,
                "finite_q_values": 1,
                "invalid_q_values": 0,
                "failed_samples": 0,
                "corrupt_committed_samples": 0,
            }
        ),
        encoding="utf-8",
    )
    return {
        "run": run,
        "candidate_root": candidate_root,
        "scored_root": scored_root,
        "prediction_manifest": prediction_manifest,
    }


def _run_fixture(
    paths: dict[str, Path],
    output_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compact_gqcnn_scores.py",
            "--split",
            "test",
            "--prediction-manifest",
            str(paths["prediction_manifest"]),
            "--candidate-root",
            str(paths["candidate_root"]),
            "--scored-root",
            str(paths["scored_root"]),
            "--output-path",
            str(output_path),
            "--tmp-root",
            str(paths["run"] / "tmp"),
            "--status-every",
            "1",
        ],
    )
    return compact_main()


def test_compaction_main_handles_nonempty_empty_and_orphan_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_compaction_fixture(tmp_path)
    output = paths["run"] / "compact" / "gqcnn_scores.parquet"
    output.parent.mkdir()
    output.write_bytes(b"interrupted")
    assert _run_fixture(paths, output, monkeypatch) == 0
    assert pq.ParquetFile(output).metadata.num_rows == 1
    compact = pq.read_table(
        output, columns=["candidate_identity_sha256"]
    ).to_pandas()
    candidate_dir = paths["candidate_root"] / "sample-1"
    canonical_records, _, _ = load_frozen_candidates(
        candidate_dir / "candidates.npz",
        candidate_dir / "candidates.json",
    )
    raw_record = json.loads(
        (candidate_dir / "candidates.json").read_text(encoding="utf-8")
    )["candidates"][0]
    assert compact.loc[0, "candidate_identity_sha256"] == (
        candidate_identity_sha256(canonical_records[0])
    )
    assert compact.loc[0, "candidate_identity_sha256"] != (
        candidate_identity_sha256(raw_record)
    )
    manifest = json.loads(
        output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["samples"] == 2
    assert manifest["empty_samples"] == 1
    assert manifest["rows"] == 1
    recovered = manifest["recovered_partial_final_artifacts"]
    assert len(recovered) == 1 and Path(recovered[0]).read_bytes() == b"interrupted"


def test_compaction_main_rejects_wrong_split_or_candidate_json_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_compaction_fixture(tmp_path)
    prediction_rows = [
        json.loads(line)
        for line in paths["prediction_manifest"].read_text().splitlines()
    ]
    prediction_rows[0]["split"] = "train"
    paths["prediction_manifest"].write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split"):
        _run_fixture(
            paths,
            paths["run"] / "wrong_split" / "gqcnn_scores.parquet",
            monkeypatch,
        )

    paths = _write_compaction_fixture(tmp_path / "tamper")
    candidate_json = paths["candidate_root"] / "sample-1" / "candidates.json"
    candidate_json.write_text(
        candidate_json.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source candidate hash"):
        _run_fixture(
            paths,
            paths["run"] / "tampered" / "gqcnn_scores.parquet",
            monkeypatch,
        )


def test_scene_shard_compaction_requires_clean_verification_and_tmp_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_compaction_fixture(tmp_path)
    prediction_sha = sha256_file(paths["prediction_manifest"])
    candidate_config = {
        "status": "COMPLETED",
        "split": "test",
        "prediction_manifest_sha256": prediction_sha,
        "protocol_identity_sha256": "p" * 64,
        "protocol_family_identity_sha256": "f" * 64,
        "counts": {"samples": 2, "nms_candidates": 1},
        "selection": {
            "num_shards": 1,
            "shard_index": 0,
            "scene_grouped": True,
            "limit": None,
        },
    }
    (paths["candidate_root"] / "run_config.json").write_text(
        json.dumps(candidate_config), encoding="utf-8"
    )
    source_manifest = (
        paths["scored_root"] / "source_candidate_manifest.jsonl"
    )
    verification = {
        "clean": True,
        "candidate_root": str(paths["candidate_root"].resolve()),
        "scored_root": str(paths["scored_root"].resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "expected_total_samples": 2,
        "expected_frozen_candidates": 1,
    }
    (paths["scored_root"] / "verification_report.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    output = paths["run"] / "tmp" / "score_shards" / "shard_0.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compact_gqcnn_scores.py",
            "--split",
            "test",
            "--prediction-manifest",
            str(paths["prediction_manifest"]),
            "--candidate-root",
            str(paths["candidate_root"]),
            "--scored-root",
            str(paths["scored_root"]),
            "--output-path",
            str(output),
            "--tmp-root",
            str(paths["run"] / "tmp"),
            "--scene-shard-count",
            "1",
            "--scene-shard-index",
            "0",
        ],
    )
    assert compact_main() == 0
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["storage"]["ephemeral_scene_shard"] is True
    assert manifest["candidate_run_manifest_sha256"] == sha256_file(
        paths["candidate_root"] / "run_manifest.jsonl"
    )
    assert manifest["independent_verification_sha256"] == sha256_file(
        paths["scored_root"] / "verification_report.json"
    )

    paths = _write_compaction_fixture(tmp_path / "missing_verification")
    (paths["candidate_root"] / "run_config.json").write_text(
        json.dumps(
            {
                **candidate_config,
                "prediction_manifest_sha256": sha256_file(
                    paths["prediction_manifest"]
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compact_gqcnn_scores.py",
            "--split",
            "test",
            "--prediction-manifest",
            str(paths["prediction_manifest"]),
            "--candidate-root",
            str(paths["candidate_root"]),
            "--scored-root",
            str(paths["scored_root"]),
            "--output-path",
            str(paths["run"] / "tmp" / "shard.parquet"),
            "--tmp-root",
            str(paths["run"] / "tmp"),
            "--scene-shard-count",
            "1",
            "--scene-shard-index",
            "0",
        ],
    )
    with pytest.raises(FileNotFoundError, match="verification_report"):
        compact_main()


def test_scene_shard_compaction_maps_local_scorer_to_global_prediction_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_compaction_fixture(tmp_path)

    def scene_for(bucket: int) -> str:
        for index in range(1000):
            scene_id = f"scene-{bucket}-{index},frame"
            if scene_shard_index(scene_id, num_shards=2) == bucket:
                return scene_id
        raise AssertionError("could not construct deterministic scene fixture")

    selected_rows = [
        json.loads(line)
        for line in paths["prediction_manifest"].read_text().splitlines()
    ]
    for global_index, row in enumerate(selected_rows, start=1):
        row["sample_index"] = global_index
        row["scene_id"] = scene_for(0)
    full_rows = [
        {
            "sample_index": 0,
            "sample_id": "unselected-sample",
            "question_index": 9,
            "scene_id": scene_for(1),
            "query": "unselected",
            "split": "test",
        },
        *selected_rows,
    ]
    paths["prediction_manifest"].write_text(
        "".join(json.dumps(row) + "\n" for row in full_rows),
        encoding="utf-8",
    )
    (paths["candidate_root"] / "run_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    # The scorer intentionally numbers its shard-local source rows from zero.
    source_rows = [
        json.loads(line)
        for line in (
            paths["scored_root"] / "source_candidate_manifest.jsonl"
        ).read_text().splitlines()
    ]
    assert [row["sample_index"] for row in source_rows] == [0, 1]

    (paths["candidate_root"] / "run_config.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "split": "test",
                "prediction_manifest_sha256": sha256_file(
                    paths["prediction_manifest"]
                ),
                "protocol_identity_sha256": "p" * 64,
                "protocol_family_identity_sha256": "f" * 64,
                "counts": {"samples": 2, "nms_candidates": 1},
                "selection": {
                    "num_shards": 2,
                    "shard_index": 0,
                    "scene_grouped": True,
                    "limit": None,
                },
            }
        ),
        encoding="utf-8",
    )
    source_manifest = (
        paths["scored_root"] / "source_candidate_manifest.jsonl"
    )
    (paths["scored_root"] / "verification_report.json").write_text(
        json.dumps(
            {
                "clean": True,
                "candidate_root": str(paths["candidate_root"].resolve()),
                "scored_root": str(paths["scored_root"].resolve()),
                "source_manifest_sha256": sha256_file(source_manifest),
                "expected_total_samples": 2,
                "expected_frozen_candidates": 1,
            }
        ),
        encoding="utf-8",
    )
    output = paths["run"] / "tmp" / "score_shards" / "shard_0.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compact_gqcnn_scores.py",
            "--split",
            "test",
            "--prediction-manifest",
            str(paths["prediction_manifest"]),
            "--candidate-root",
            str(paths["candidate_root"]),
            "--scored-root",
            str(paths["scored_root"]),
            "--output-path",
            str(output),
            "--tmp-root",
            str(paths["run"] / "tmp"),
            "--scene-shard-count",
            "2",
            "--scene-shard-index",
            "0",
        ],
    )
    assert compact_main() == 0
    compact = pq.read_table(output).to_pandas()
    assert compact["sample_index"].tolist() == [1]
