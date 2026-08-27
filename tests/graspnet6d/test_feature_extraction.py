"""Deterministic geometry fixtures only; not formal candidate features."""

from __future__ import annotations

import numpy as np

from graspnet6d.contracts import Candidate6D
from graspnet6d.feature_extraction import RuntimeObservation, extract_candidate_features
from graspnet6d.features import default_feature_schema, forbidden_feature_columns
from graspnet6d.geometry import CameraIntrinsics


def _candidate() -> Candidate6D:
    provenance = {"mapping_name": "fixture", "mapping_status": "unit_test_only", "source": "deterministic test"}
    return Candidate6D(
        candidate_id="g:c0", group_id="g", native_rank=1, native_score=0.9,
        translation_local_m=(0.15, 0.15, 0.15), rotation_local=np.eye(3),
        translation_camera_m=(0.0, 0.0, 1.0), rotation_camera=np.eye(3),
        translation_table_m=(0.0, 0.0, 0.10), rotation_table=np.eye(3),
        width_m=0.06, height_m=0.02, depth_m=0.04, voxel_index=(20, 20, 20),
        conversion_provenance=provenance,
    )


def test_extractor_emits_exact_runtime_schema_without_supervision() -> None:
    probability = np.zeros((20, 20), dtype=float)
    probability[7:14, 7:14] = 1.0
    depth = np.ones((20, 20), dtype=float)
    target = np.array([[x, y, 1.0] for x in (-0.02, 0, 0.02) for y in (-0.02, 0, 0.02)])
    scene = np.vstack((target, np.array([[0.0, 0.04, 1.0], [0.0, -0.04, 1.0]])))
    observation = RuntimeObservation(
        intrinsics=CameraIntrinsics(100, 100, 10, 10, width=20, height=20),
        mask_probability=probability, depth_m=depth,
        scene_points_camera_m=scene,
        table_normal_camera=np.array([0, -1, 0]), gravity_camera=np.array([0, 1, 0]),
        grounding_condition="oracle_gt_mask",
        mask_source_sha256="0" * 64,
        depth_source_sha256="1" * 64,
        scene_points_source_sha256="2" * 64,
    )
    frame = extract_candidate_features([_candidate()], observation)
    assert tuple(frame.columns) == tuple(spec.name for spec in default_feature_schema())
    assert not forbidden_feature_columns(frame.columns)
    assert len(frame) == 1
    assert frame.loc[0, "center_in_mask"] == 1.0
    assert frame.loc[0, "gripper_width_m"] == 0.06


def test_extractor_preserves_candidate_row_count_and_order() -> None:
    candidate = _candidate()
    probability = np.ones((8, 8))
    points = np.array([[0.0, 0.0, 1.0], [0.01, 0, 1.0], [0, 0.01, 1.0]])
    observation = RuntimeObservation(
        intrinsics=CameraIntrinsics(50, 50, 4, 4, width=8, height=8),
        mask_probability=probability,
        depth_m=np.ones((8, 8)),
        scene_points_camera_m=points,
        table_normal_camera=np.array([0, 0, 1]),
        gravity_camera=np.array([0, 0, 1]),
        grounding_condition="oracle_gt_mask",
        mask_source_sha256="0" * 64,
        depth_source_sha256="1" * 64,
        scene_points_source_sha256="2" * 64,
    )
    frame = extract_candidate_features([candidate], observation)
    assert len(frame) == 1


def test_target_points_are_derived_from_the_selected_condition() -> None:
    probability = np.zeros((8, 8), dtype=float)
    probability[4, 4] = 1.0
    depth = np.ones((8, 8), dtype=float)
    observation = RuntimeObservation(
        intrinsics=CameraIntrinsics(50, 50, 4, 4, width=8, height=8),
        mask_probability=probability,
        depth_m=depth,
        scene_points_camera_m=np.array([[0.0, 0.0, 1.0]]),
        table_normal_camera=np.array([0, 0, 1]),
        gravity_camera=np.array([0, 0, 1]),
        grounding_condition="oracle_gt_mask",
        mask_source_sha256="0" * 64,
        depth_source_sha256="1" * 64,
        scene_points_source_sha256="2" * 64,
    )
    assert np.array_equal(observation.target_points_camera_m(), [[0.0, 0.0, 1.0]])
