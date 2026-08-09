from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.grasping.camera_geometry import CameraIntrinsicsData
from src.grasping.geometric_ranker import load_frozen_candidates
from src.grasping.grasp_serialization import save_candidate_bundle
from src.grasping.reranking_v1.features import (
    FORBIDDEN_GT_COLUMNS,
    INFERENCE_FEATURE_ALLOWLIST,
    compute_train_only_statistics,
    feature_schema,
    join_candidate_labels,
    quality_and_relation_features,
    reject_forbidden_features,
    resize_probability_to_native,
    soft_mask_features,
    transform_visible_points_to_grasp,
    validate_inference_allowlist,
    visible_surface_collision_features,
    width_and_depth_features,
)
from tools.modular_reranking.extract_candidate_features import (
    assigned_to_shard,
    load_empty_score_marker,
)
from tools.modular_reranking.merge_feature_shards import main as merge_features_main
from tools.modular_reranking.select_representative_vlm_pilot import (
    main as select_vlm_pilot_main,
)
from src.grasping.reranking_v1.identity import sha256_file


def intrinsics() -> CameraIntrinsicsData:
    return CameraIntrinsicsData(
        frame="camera",
        fx=100.0,
        fy=100.0,
        cx=9.5,
        cy=9.5,
        skew=0.0,
        height=20,
        width=20,
    )


def candidate(
    candidate_id: str = "g0000",
    *,
    center=(10.0, 10.0),
    contacts=((7.0, 10.0), (13.0, 10.0)),
    q=0.5,
) -> dict:
    return {
        "sample_id": "sample",
        "candidate_id": candidate_id,
        "center_uv": list(center),
        "center_depth_m": 1.0,
        "center_camera_xyz_m": [0.0, 0.0, 1.0],
        "angle_rad": 0.0,
        "width_m": 0.06,
        "width_px": 6.0,
        "endpoints_uv": [[7.0, 10.0], [13.0, 10.0]],
        "contact_points_uv": [list(contacts[0]), list(contacts[1])],
        "contact_normals": [[-1.0, 0.0], [1.0, 0.0]],
        "T_camera_grasp_fixed_approach": np.eye(4).tolist(),
        "gqcnn_q_value": q,
    }


def test_valid_empty_score_marker_requires_zero_candidate_status(
    tmp_path: Path,
) -> None:
    sample_id = "empty_sample"
    marker = {
        "sample_id": sample_id,
        "scoring_status": "skipped_valid_empty",
        "source_candidate_count": 0,
        "gqcnn_scored_count": 0,
    }
    (tmp_path / "_SCORING_COMPLETE.json").write_text(json.dumps(marker))
    (tmp_path / "scoring_metadata.json").write_text(json.dumps(marker))
    provenance = load_empty_score_marker(tmp_path, sample_id=sample_id)
    assert provenance["scoring_status"] == "skipped_valid_empty"
    assert provenance["scored_npz_sha256"] is None
    (tmp_path / "gqcnn_scored_candidates.npz").touch()
    with pytest.raises(ValueError, match="unexpectedly wrote an NPZ"):
        load_empty_score_marker(tmp_path, sample_id=sample_id)


def test_valid_empty_frozen_candidate_bundle_round_trips(tmp_path: Path) -> None:
    save_candidate_bundle(
        [],
        json_path=tmp_path / "candidates.json",
        npz_path=tmp_path / "candidates.npz",
        csv_path=tmp_path / "candidates.csv",
        metadata={"sample_id": "empty"},
    )
    records, metadata, hashes = load_frozen_candidates(
        tmp_path / "candidates.npz", tmp_path / "candidates.json"
    )
    assert records == []
    assert metadata["sample_id"] == "empty"
    assert set(hashes) == {"candidates_npz_sha256", "candidates_json_sha256"}


def test_soft_mask_interpolation_uses_bilinear_and_preserves_bounds() -> None:
    source = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    resized = resize_probability_to_native(source, (4, 4))
    assert resized.shape == (4, 4)
    assert np.all((resized >= 0.0) & (resized <= 1.0))
    assert resized[0, 0] == pytest.approx(0.0)
    assert resized[-1, -1] == pytest.approx(0.0)
    assert resized[1, 1] == pytest.approx(0.375)


def test_feature_extraction_at_image_border_is_finite() -> None:
    record = candidate(center=(0.0, 0.0), contacts=((0.0, 0.0), (4.0, 0.0)))
    record["endpoints_uv"] = [[-3.0, 0.0], [3.0, 0.0]]
    probability = np.full((20, 20), 0.6, dtype=np.float32)
    mask = probability >= 0.15
    depth = np.ones((20, 20), dtype=np.float32)
    features = soft_mask_features(
        record, probability=probability, binary_mask=mask, depth_m=depth
    )
    assert all(np.isfinite(float(value)) for value in features.values())
    assert features["p_center"] == pytest.approx(0.6)


def test_jaw_regions_are_measured_separately() -> None:
    record = candidate()
    probability = np.zeros((20, 20), dtype=np.float32)
    probability[7:14, 4:11] = 1.0
    mask = probability >= 0.15
    depth = np.ones((20, 20), dtype=np.float32)
    features = soft_mask_features(
        record, probability=probability, binary_mask=mask, depth_m=depth
    )
    assert features["p_left_jaw"] > features["p_right_jaw"]


def test_width_compatibility_and_normals_are_finite() -> None:
    record = candidate()
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:13, 7:14] = True
    depth = np.ones((20, 20), dtype=np.float32)
    features = width_and_depth_features(
        record, binary_mask=mask, depth_m=depth, intrinsics=intrinsics()
    )
    assert features["width_ratio_to_max_gripper"] == pytest.approx(0.75)
    assert features["width_margin_to_max"] == pytest.approx(0.02)
    assert features["normal_opposition"] == pytest.approx(1.0)
    assert features["normal_closing_axis_alignment"] == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in features.values())


def test_visible_point_cloud_transformation() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    camera_points = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
    grasp_points = transform_visible_points_to_grasp(camera_points, transform)
    assert np.allclose(grasp_points, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_visible_collision_proxy_is_finite_even_without_depth() -> None:
    features = visible_surface_collision_features(
        candidate(), depth_m=np.zeros((20, 20), np.float32), intrinsics=intrinsics()
    )
    assert set(features) == {
        "left_finger_occupancy",
        "right_finger_occupancy",
        "palm_occupancy",
        "approach_corridor_occupancy",
        "minimum_visible_obstacle_distance",
        "approach_clearance",
        "collision_proxy_total",
    }
    assert all(np.isfinite(value) for value in features.values())
    assert features["collision_proxy_total"] == 0.0
    assert features["approach_clearance"] == 1.0


def test_q_rank_join_and_relations_do_not_assume_npz_row_zero_is_top1() -> None:
    records = [
        candidate("g0000", center=(5.0, 5.0)),
        candidate("g0001", center=(10.0, 5.0)),
        candidate("g0002", center=(19.0, 19.0)),
    ]
    values = quality_and_relation_features(records, [0.1, 0.8, 0.2], [3, 1, 2])
    assert values[1]["q_rank_normalized"] == 1.0
    assert values[0]["q_gap_to_top1"] == pytest.approx(0.7)
    assert values[1]["nearest_candidate_center_distance"] == pytest.approx(5.0)
    assert all(np.isfinite(float(value)) for row in values for value in row.values())


def test_train_only_scaler_rejects_validation_and_test() -> None:
    frame = pd.DataFrame(
        {
            column: np.array([0.0, 1.0], dtype=float)
            for column in INFERENCE_FEATURE_ALLOWLIST
        }
    )
    statistics = compute_train_only_statistics(frame, split="train")
    assert statistics["source_split"] == "train"
    assert set(statistics["statistics"]) == set(INFERENCE_FEATURE_ALLOWLIST)
    with pytest.raises(ValueError, match="only"):
        compute_train_only_statistics(frame, split="test")


def test_inference_allowlist_and_forbidden_gt_rejection() -> None:
    assert validate_inference_allowlist(["q_raw", "p_center"]) == ("q_raw", "p_center")
    with pytest.raises(ValueError, match="forbidden"):
        reject_forbidden_features(["q_raw", "candidate_positive"])
    with pytest.raises(ValueError, match="not in"):
        validate_inference_allowlist(["q_raw", "query_length"])
    assert set(FORBIDDEN_GT_COLUMNS).isdisjoint(INFERENCE_FEATURE_ALLOWLIST)


def test_labels_are_joined_only_by_two_part_identity() -> None:
    features = [
        {"sample_id": "a", "candidate_id": "g0", "q_raw": 0.1},
        {"sample_id": "b", "candidate_id": "g0", "q_raw": 0.2},
    ]
    labels = [
        {
            "sample_id": "b",
            "candidate_id": "g0",
            "candidate_positive": True,
            "candidate_gt_angle_error_deg": 2.0,
        },
        {
            "sample_id": "a",
            "candidate_id": "g0",
            "candidate_positive": False,
            "candidate_gt_angle_error_deg": 45.0,
        },
    ]
    joined = join_candidate_labels(features, labels)
    assert [row["candidate_positive"] for row in joined] == [False, True]
    assert [row["candidate_gt_angle_error"] for row in joined] == [45.0, 2.0]
    with pytest.raises(ValueError, match="missing"):
        join_candidate_labels(features, labels[:1])


def test_feature_shard_merge_replays_assignment_and_rejects_mixed_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".RUN_ACTIVE").write_text("", encoding="utf-8")
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    sample_ids = []
    for bucket in range(2):
        sample_ids.append(
            next(
                f"sample_{index}"
                for index in range(1000)
                if assigned_to_shard(
                    f"sample_{index}", num_shards=2, shard_index=bucket
                )
            )
        )
    roots = [tmp_path / "shard0", tmp_path / "shard1"]
    prediction_manifest = tmp_path / "prediction.jsonl"
    prediction_manifest.write_text(
        "".join(
            json.dumps({"sample_id": sample_id}) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    prediction_manifest_sha256 = sha256_file(prediction_manifest)
    shared_files = {
        "inference_feature_allowlist.json": {
            "schema_version": 1,
            "features": list(INFERENCE_FEATURE_ALLOWLIST),
        },
        "forbidden_gt_columns.json": {
            "schema_version": 1,
            "columns": list(FORBIDDEN_GT_COLUMNS),
        },
    }
    for shard_index, root in enumerate(roots):
        root.mkdir()
        candidate_columns = {
            "sample_id": pd.Series(dtype=str),
            "candidate_id": pd.Series(dtype=str),
            "original_gqcnn_rank": pd.Series(dtype=int),
            **{
                feature: pd.Series(dtype=float)
                for feature in INFERENCE_FEATURE_ALLOWLIST
            },
        }
        candidate_frame = pd.DataFrame(candidate_columns)
        if shard_index == 1:
            candidate_frame = pd.DataFrame(
                [
                    {
                        "sample_id": sample_ids[1],
                        "candidate_id": "g0000",
                        "original_gqcnn_rank": 1,
                        **{feature: 0.0 for feature in INFERENCE_FEATURE_ALLOWLIST},
                    }
                ]
            )
        sample_frame = pd.DataFrame(
            [
                {
                    "sample_id": sample_ids[shard_index],
                    "candidate_count": len(candidate_frame),
                }
            ]
        )
        candidate_frame.to_parquet(root / "per_candidate.parquet", index=False)
        sample_frame.to_parquet(root / "per_sample.parquet", index=False)
        (root / "feature_schema.json").write_text(
            json.dumps(feature_schema(candidate_frame), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for filename, payload in shared_files.items():
            (root / filename).write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
        (root / "feature_statistics_train_only.json").write_text(
            json.dumps({"schema_version": 1, "source_split": None}) + "\n"
        )
        manifest = {
            "schema_version": 1,
            "status": "COMPLETED",
            "split": "val",
            "num_shards": 2,
            "shard_index": shard_index,
            "shard_assignment": "sha256(sample_id)[0:8] modulo num_shards",
            "sample_count": 1,
            "candidate_count": len(candidate_frame),
            "labels_joined_post_extraction": False,
            "labels_source": None,
            "labels_source_sha256": None,
            "labels_source_file_count": None,
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
            "annotations_path": "/frozen/annotations.json",
            "annotations_sha256": "annotations",
            "candidate_root": "/frozen/candidates",
            "scored_root": "/frozen/scores",
            "prediction_root": "/frozen/predictions",
            "prediction_manifest_path": str(prediction_manifest.resolve()),
            "prediction_manifest_sha256": prediction_manifest_sha256,
            "prediction_identity_sha256": "8" * 64,
            "frozen_split_manifest_sha256": "9" * 64,
        }
        (root / "dataset_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
    output = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_feature_shards.py",
            "--shard-root",
            str(roots[0]),
            "--shard-root",
            str(roots[1]),
            "--output-root",
            str(output),
            "--tmp-root",
            str(tmp_root),
            "--split",
            "val",
        ],
    )
    assert merge_features_main() == 0
    assert len(pd.read_parquet(output / "per_sample.parquet")) == 2
    assert len(pd.read_parquet(output / "per_candidate.parquet")) == 1
    assert (
        json.loads((roots[0] / "feature_schema.json").read_text())["rows"]
        != json.loads((roots[1] / "feature_schema.json").read_text())["rows"]
    )

    changed = json.loads((roots[1] / "dataset_manifest.json").read_text())
    changed["labels_joined_post_extraction"] = True
    changed["labels_source"] = "/different/labels"
    (roots[1] / "dataset_manifest.json").write_text(json.dumps(changed))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_feature_shards.py",
            "--shard-root",
            str(roots[0]),
            "--shard-root",
            str(roots[1]),
            "--output-root",
            str(tmp_path / "mixed"),
            "--tmp-root",
            str(tmp_root),
            "--split",
            "val",
        ],
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        merge_features_main()


def test_representative_vlm_selector_stratifies_valid_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = pd.DataFrame(
        [
                {
                    "sample_id": "nonempty",
                    "scene_id": "scene-a",
                    "split": "val",
                    "query_type": "name",
                "candidate_count": 1,
                "q_top1_gap": 0.2,
                "mask_area_px": 100,
            },
                {
                    "sample_id": "empty",
                    "scene_id": "scene-b",
                    "split": "val",
                "query_type": "relation",
                "candidate_count": 0,
                "q_top1_gap": 0.0,
                "mask_area_px": 40,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
                {
                    "sample_id": "nonempty",
                    "split": "val",
                    "original_gqcnn_rank": 1,
                "candidate_positive": True,
            }
        ]
    )
    sample_path = tmp_path / "per_sample.parquet"
    candidate_path = tmp_path / "per_candidate.parquet"
    samples.to_parquet(sample_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    output = tmp_path / "pilot"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_representative_vlm_pilot.py",
            "--per-sample",
            str(sample_path),
            "--per-candidate",
            str(candidate_path),
            "--output-root",
            str(output),
            "--count",
            "2",
            "--scene-cap",
            "1",
        ],
    )
    assert select_vlm_pilot_main() == 0
    audit = pd.read_parquet(output / "offline_selection_audit.parquet")
    empty = audit.loc[audit["sample_id"] == "empty"].iloc[0]
    assert empty["outcome_stratum"] == "valid_empty_no_candidates"
    assert not bool(empty["q_top1_correct"])
    assert not bool(empty["top5_positive"])
