"""Orchestration tests use tiny fixtures/mocks, never formal experiment rows."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from graspnet6d.evaluator import CandidateEvaluation
from graspnet6d.io import atomic_json, atomic_jsonl, sha256_file
from graspnet6d.splits import SceneSplit
from graspnet6d.stages import (
    GEOMETRY_SCHEMA,
    LABEL_BUNDLE_SCHEMA,
    PARITY_SCHEMA,
    StageBatchError,
    StageInputError,
    assemble_labeled_feature_rows,
    load_evaluator_geometry_contract,
    load_evaluator_parity_gate,
    load_target_language_jsonl,
    run_official_label_stage,
    run_oracle_mask_stage,
    run_evaluate_hook,
    run_train_ranker_hook,
    run_tsdf_stage,
    run_vgn_candidate_stage,
)
from graspnet6d.tsdf import TSDFBuildResult
from graspnet6d.vgn import VGNCandidate


def _fixture_manifests(tmp_path: Path, *, missing_label: bool = False) -> tuple[Path, Path, dict[str, Any]]:
    inputs = tmp_path / "real_input_fixture"
    inputs.mkdir()
    label = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint16)
    depth = np.array([[900, 1000, 1100], [1200, 1300, 1400]], dtype=np.uint16)
    label_path = inputs / "label.png"
    depth_path = inputs / "depth.png"
    if not missing_label:
        Image.fromarray(label).save(label_path)
    Image.fromarray(depth).save(depth_path)
    intrinsic = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]])
    intrinsics_path = inputs / "camK.npy"
    camera_pose_path = inputs / "camera_poses.npy"
    table_path = inputs / "cam0_wrt_table.npy"
    meta_path = inputs / "meta.mat"
    np.save(intrinsics_path, intrinsic, allow_pickle=False)
    np.save(camera_pose_path, np.eye(4, dtype=np.float64)[None], allow_pickle=False)
    np.save(table_path, np.eye(4, dtype=np.float64), allow_pickle=False)
    savemat(
        meta_path,
        {
            "factor_depth": np.array([[1000.0]]),
            "intrinsic_matrix": intrinsic,
        },
    )
    target = {
        "group_id": "scene_0000_kinect_0000_obj_000",
        "split": "train",
        "scene_id": "scene_0000",
        "camera": "kinect",
        "frame_id": 0,
        "target_object_id": 0,
        "target_instance_label": 1,
        "depth_path": str(depth_path),
        "instance_label_path": str(label_path),
        "meta_path": str(meta_path),
        "intrinsics_path": str(intrinsics_path),
        "camera_pose_path": str(camera_pose_path),
        "table_transform_path": str(table_path),
    }
    language = {
        "group_id": target["group_id"],
        "query": "the cracker box",
        "is_unique": True,
        "resolver_result": [0],
        "provenance": "derived_test_fixture",
    }
    target_path = tmp_path / "targets.jsonl"
    language_path = tmp_path / "language.jsonl"
    atomic_jsonl(target_path, [target])
    atomic_jsonl(language_path, [language])
    return target_path, language_path, target


def _geometry_gate(tmp_path: Path, *, height_m: float = 0.02) -> Path:
    artifact = tmp_path / "geometry_validation.json"
    atomic_json(artifact, {"status": "PASSED", "fixture_only": True})
    contract = tmp_path / "geometry_contract.json"
    atomic_json(
        contract,
        {
            "schema_version": GEOMETRY_SCHEMA,
            "validated": True,
            "validation_artifact": artifact.name,
            "R_vgn_gripper_to_graspnet_gripper": np.eye(3).tolist(),
            "height_m": height_m,
            "depth_m": 0.04,
        },
    )
    return contract


def _parity_gate(tmp_path: Path) -> Path:
    artifact = tmp_path / "evaluator_parity.json"
    atomic_json(artifact, {"status": "PASSED", "fixture_only": True})
    gate = tmp_path / "parity_gate.json"
    atomic_json(
        gate,
        {
            "schema_version": PARITY_SCHEMA,
            "validated": True,
            "artifact_path": artifact.name,
        },
    )
    return gate


def _fake_tsdf_builder(call_log: list[np.ndarray]):
    def build(
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        *,
        T_camera_to_table: np.ndarray,
        depth_scale: float,
        source_view_id: int,
    ) -> TSDFBuildResult:
        # A non-target pixel remains non-zero: the stage passed complete depth,
        # not depth multiplied by the oracle mask.
        assert depth[1, 2] == 1400
        assert not mask[1, 2]
        assert depth_scale == 1000.0
        assert np.array_equal(T_camera_to_table, np.eye(4))
        call_log.append(depth.copy())
        return TSDFBuildResult(
            tsdf=np.full((1, 40, 40, 40), 0.25, dtype=np.float32),
            voxel_size_m=0.0075,
            physical_size_m=0.30,
            workspace_origin_camera_m=np.array([-0.15, -0.15, 0.85]),
            T_local_to_camera=np.eye(4),
            T_local_to_table=np.eye(4),
            depth_scale=depth_scale,
            valid_voxel_fraction=1.0,
            source_view_ids=(source_view_id,),
            full_scene_depth_integrated=True,
        )

    return build


def _prepare_tsdf(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    target_path, language_path, target = _fixture_manifests(tmp_path)
    output_root = tmp_path / "outputs"
    run_oracle_mask_stage(target_path, language_path, output_root)
    run_tsdf_stage(
        target_path,
        language_path,
        output_root,
        builder=_fake_tsdf_builder([]),
    )
    return target_path, language_path, output_root, target


def test_manifest_oracle_and_tsdf_resume_are_group_idempotent_and_atomic(tmp_path: Path) -> None:
    target_path, language_path, target = _fixture_manifests(tmp_path)
    groups = load_target_language_jsonl(target_path, language_path)
    assert [group.group_id for group in groups] == [target["group_id"]]
    output_root = tmp_path / "outputs"

    first_masks = run_oracle_mask_stage(target_path, language_path, output_root)
    second_masks = run_oracle_mask_stage(
        target_path, language_path, output_root, resume=True
    )
    assert first_masks.completed_groups == 1
    assert second_masks.completed_groups == 0
    assert second_masks.resumed_groups == 1
    with np.load(first_masks.output_paths[0], allow_pickle=False) as archive:
        assert archive["mask"].tolist() == [[0, 1, 1], [0, 1, 0]]

    calls: list[np.ndarray] = []
    first_tsdf = run_tsdf_stage(
        target_path,
        language_path,
        output_root,
        builder=_fake_tsdf_builder(calls),
    )
    second_tsdf = run_tsdf_stage(
        target_path,
        language_path,
        output_root,
        resume=True,
        builder=_fake_tsdf_builder(calls),
    )
    assert len(calls) == 1
    assert first_tsdf.completed_groups == 1
    assert second_tsdf.resumed_groups == 1
    assert not list(output_root.rglob("*.tmp"))
    assert not [path for path in output_root.rglob(".*") if path.is_file()]


def test_predicted_mask_centres_condition_specific_tsdf_and_binds_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, language_path, target = _fixture_manifests(tmp_path)
    output_root = tmp_path / "outputs"
    mask_root = tmp_path / "predicted"
    commit_root = mask_root / "fixture"
    commit_root.mkdir(parents=True)
    probability_path = commit_root / "probability.npz"
    mask_path = commit_root / "mask.png"
    sidecar_path = commit_root / "mask.json"
    np.savez(probability_path, probability=np.ones((2, 3), dtype=np.float32))
    Image.fromarray(np.array([[0, 255, 255], [0, 255, 0]], dtype=np.uint8)).save(
        mask_path
    )
    atomic_json(sidecar_path, {"input_fingerprint": "a" * 64})
    binary = np.array([[False, True, True], [False, True, False]])

    import graspnet6d.formal_inputs as formal_inputs

    monkeypatch.setattr(
        formal_inputs,
        "predicted_mask_paths",
        lambda *_: (probability_path, mask_path, sidecar_path),
    )
    monkeypatch.setattr(
        formal_inputs,
        "load_committed_predicted_mask",
        lambda *_args, **_kwargs: SimpleNamespace(
            binary_mask=binary,
            sidecar_path=sidecar_path,
            probability_path=probability_path,
            mask_path=mask_path,
            sidecar={"input_fingerprint": "a" * 64},
        ),
    )
    calls: list[np.ndarray] = []
    summary = run_tsdf_stage(
        target_path,
        language_path,
        output_root,
        grounding_condition="hifics_zero_shot_mask",
        mask_output_root=mask_root,
        builder=_fake_tsdf_builder(calls),
    )

    assert len(calls) == 1
    output = Path(summary.output_paths[0])
    assert output.parent.name == "hifics_zero_shot_mask"
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["group_id"] == target["group_id"]
    assert sidecar["grounding_condition"] == "hifics_zero_shot_mask"
    assert sidecar["grounding_mask_input_fingerprint"] == "a" * 64
    assert sidecar["grounding_mask_commit_sha256"] == sha256_file(sidecar_path)
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["grounding_condition"]) == "hifics_zero_shot_mask"


def test_missing_real_input_records_structured_group_error_without_placeholder(tmp_path: Path) -> None:
    target_path, language_path, target = _fixture_manifests(tmp_path, missing_label=True)
    output_root = tmp_path / "outputs"
    with pytest.raises(StageBatchError) as caught:
        run_oracle_mask_stage(target_path, language_path, output_root)
    assert [failure.group_id for failure in caught.value.failures] == [target["group_id"]]
    error_path = Path(caught.value.failures[0].error_path)
    error = json.loads(error_path.read_text(encoding="utf-8"))
    assert error["stage"] == "oracle_masks"
    assert error["group_id"] == target["group_id"]
    assert error["error_type"] == "StageInputError"
    assert not list((output_root / "oracle_masks").glob("*.npz"))


def test_frozen_vgn_bundle_runs_once_preserves_raw_frames_and_resumes(tmp_path: Path) -> None:
    target_path, language_path, output_root, target = _prepare_tsdf(tmp_path)
    geometry = _geometry_gate(tmp_path)
    checkpoint = tmp_path / "vgn_checkpoint.pth"
    checkpoint.write_bytes(b"unit-test-checkpoint-not-formal")
    calls = {"load": 0, "infer": 0, "extract": 0}

    def load_model(**_: Any) -> object:
        calls["load"] += 1
        return object()

    def infer(tsdf: np.ndarray, model: object, *, device: str) -> object:
        assert tsdf.shape == (1, 40, 40, 40)
        assert model is not None and device == "cpu"
        calls["infer"] += 1
        return object()

    def extract(
        tsdf: np.ndarray,
        raw: object,
        *,
        group_id: str,
        config: Any,
        T_local_to_camera: np.ndarray,
        T_local_to_table: np.ndarray,
    ) -> list[VGNCandidate]:
        assert raw is not None and config.frozen_top_k >= 2
        assert np.array_equal(T_local_to_camera, np.eye(4))
        calls["extract"] += 1
        return [
            VGNCandidate(
                candidate_id="candidate-a",
                group_id=group_id,
                native_rank=1,
                native_score=0.9,
                translation_local_m=np.array([0.10, 0.11, 0.12]),
                rotation_local_vgn=np.eye(3),
                width_m=0.05,
                voxel_index=(1, 2, 3),
                translation_camera_m=np.array([0.10, 0.11, 0.12]),
                rotation_camera_vgn=np.eye(3),
                translation_table_m=np.array([0.10, 0.11, 0.12]),
                rotation_table_vgn=np.eye(3),
            ),
            VGNCandidate(
                candidate_id="candidate-b",
                group_id=group_id,
                native_rank=3,
                native_score=0.8,
                translation_local_m=np.array([0.20, 0.21, 0.22]),
                rotation_local_vgn=np.eye(3),
                width_m=0.04,
                voxel_index=(4, 5, 6),
                translation_camera_m=np.array([0.20, 0.21, 0.22]),
                rotation_camera_vgn=np.eye(3),
                translation_table_m=np.array([0.20, 0.21, 0.22]),
                rotation_table_vgn=np.eye(3),
            ),
        ]

    first = run_vgn_candidate_stage(
        target_path,
        language_path,
        output_root,
        geometry_contract_path=geometry,
        checkpoint=checkpoint,
        evidence_policy="test",
        model_loader=load_model,
        inference=infer,
        extractor=extract,
    )
    first_hash = sha256_file(Path(first.output_paths[0]))
    second = run_vgn_candidate_stage(
        target_path,
        language_path,
        output_root,
        geometry_contract_path=geometry,
        checkpoint=checkpoint,
        resume=True,
        evidence_policy="test",
        model_loader=load_model,
        inference=infer,
        extractor=extract,
    )
    assert calls == {"load": 1, "infer": 1, "extract": 1}
    assert second.resumed_groups == 1
    assert sha256_file(Path(first.output_paths[0])) == first_hash
    bundle = json.loads(Path(first.output_paths[0]).read_text(encoding="utf-8"))
    assert bundle["inference_calls_for_group"] == 1
    assert bundle["candidate_count"] == 2
    assert [row["candidate_id"] for row in bundle["raw_vgn_candidates"]] == [
        "candidate-a",
        "candidate-b",
    ]
    assert bundle["raw_vgn_candidates"][0]["rotation_local_vgn"] == np.eye(3).tolist()
    assert bundle["raw_vgn_pool_fingerprint"]
    assert bundle["candidate_pool_fingerprint"]
    assert np.asarray(bundle["graspnet_rows"]).shape == (2, 17)
    assert {record["height_m"] for record in bundle["candidate_records"]} == {0.02}
    assert {record["depth_m"] for record in bundle["candidate_records"]} == {0.04}
    assert bundle["group_id"] == target["group_id"]


def test_vgn_geometry_gate_fails_closed_and_does_not_overwrite_bundle(tmp_path: Path) -> None:
    target_path, language_path, output_root, target = _prepare_tsdf(tmp_path)
    invalid_geometry = _geometry_gate(tmp_path, height_m=0.0)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(StageBatchError) as caught:
        run_vgn_candidate_stage(
            target_path,
            language_path,
            output_root,
            geometry_contract_path=invalid_geometry,
            checkpoint=checkpoint,
            evidence_policy="test",
            model_loader=lambda **_: pytest.fail("model must not load before geometry gate"),
        )
    assert caught.value.failures[0].group_id == target["group_id"]
    assert "positive" in caught.value.failures[0].message
    assert not list((output_root / "vgn_candidates").glob("*.json"))


def test_formal_evidence_policy_rejects_explicit_test_fixtures(tmp_path: Path) -> None:
    geometry = _geometry_gate(tmp_path)
    parity = _parity_gate(tmp_path)
    with pytest.raises(StageInputError, match="formal evidence schema"):
        load_evaluator_geometry_contract(geometry)
    with pytest.raises(StageInputError, match="formal evidence schema"):
        load_evaluator_parity_gate(parity)
    # Tests can opt in only when the evidence itself is marked fixture_only.
    assert load_evaluator_geometry_contract(geometry, evidence_policy="test")[1][
        "evidence_policy"
    ] == "test"
    assert load_evaluator_parity_gate(parity, evidence_policy="test")[1][
        "evidence_policy"
    ] == "test"


def test_official_label_hook_preserves_frozen_membership_and_resumes(tmp_path: Path) -> None:
    target_path, language_path, output_root, target = _prepare_tsdf(tmp_path)
    geometry = _geometry_gate(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")

    def candidates(*_: Any, group_id: str, **__: Any) -> list[VGNCandidate]:
        return [
            VGNCandidate(
                candidate_id="candidate-only",
                group_id=group_id,
                native_rank=1,
                native_score=0.9,
                translation_local_m=np.array([0.1, 0.1, 0.1]),
                rotation_local_vgn=np.eye(3),
                width_m=0.05,
                voxel_index=(1, 1, 1),
                translation_camera_m=np.array([0.1, 0.1, 0.1]),
                rotation_camera_vgn=np.eye(3),
                translation_table_m=np.array([0.1, 0.1, 0.1]),
                rotation_table_vgn=np.eye(3),
            )
        ]

    run_vgn_candidate_stage(
        target_path,
        language_path,
        output_root,
        geometry_contract_path=geometry,
        checkpoint=checkpoint,
        evidence_policy="test",
        model_loader=lambda **_: object(),
        inference=lambda *_, **__: object(),
        extractor=candidates,
    )
    parity = _parity_gate(tmp_path)
    dataset_root = tmp_path / "official_dataset_fixture"
    dataset_root.mkdir()
    evaluator_source = tmp_path / "official_source.fixture"
    evaluator_source.write_text("fixture", encoding="utf-8")
    calls = {"load": 0, "evaluate": 0}

    def load_inputs(*_: Any, **__: Any) -> dict[str, Any]:
        calls["load"] += 1
        return {
            "models_object_m": [np.zeros((1, 3))],
            "dexnet_models": [object()],
            "poses_object_to_camera": [np.eye(4)],
            "object_ids": [0],
            "dexnet_config": {"metrics": {"force_closure": {}}},
            "table_points_camera_m": np.zeros((1, 3)),
            "source_hashes": {str(evaluator_source): sha256_file(evaluator_source)},
            "dexnet_source_kind": "fixture_no_pickle",
        }

    def evaluate(rows: np.ndarray, **kwargs: Any) -> list[CandidateEvaluation]:
        calls["evaluate"] += 1
        assert len(rows) == 1
        assert kwargs["validation_probe"] is False
        return [
            CandidateEvaluation(
                candidate_index=0,
                associated_instance_index=0,
                associated_object_id=0,
                target_object_id=0,
                correct_target=True,
                collision=False,
                empty_grasp=False,
                friction_score=0.4,
                valid_geometry=True,
                relevance=5,
            )
        ]

    first = run_official_label_stage(
        target_path,
        language_path,
        output_root,
        dataset_root=dataset_root,
        parity_gate_path=parity,
        evidence_policy="test",
        input_loader=load_inputs,
        evaluator=evaluate,
    )
    second = run_official_label_stage(
        target_path,
        language_path,
        output_root,
        dataset_root=dataset_root,
        parity_gate_path=parity,
        resume=True,
        evidence_policy="test",
        input_loader=load_inputs,
        evaluator=evaluate,
    )
    assert calls == {"load": 1, "evaluate": 1}
    assert second.resumed_groups == 1
    labels = json.loads(Path(first.output_paths[0]).read_text(encoding="utf-8"))
    assert labels["schema_version"] == LABEL_BUNDLE_SCHEMA
    assert labels["candidate_ids"] == ["candidate-only"]
    assert [row["candidate_id"] for row in labels["labels"]] == ["candidate-only"]
    assert labels["labels"][0]["relevance"] == 5
    assert labels["evaluator_operation"] == "per_candidate_low_level_no_eval_grasp_no_nms_no_topk"
    assert labels["group_id"] == target["group_id"]


def test_strict_feature_label_join_refuses_missing_candidate_rows(tmp_path: Path) -> None:
    features = tmp_path / "features.jsonl"
    labels = tmp_path / "labels.json"
    output = tmp_path / "joined.jsonl"
    atomic_jsonl(
        features,
        [{"group_id": "g", "candidate_id": "a", "native_score": 0.9}],
    )
    atomic_json(
        labels,
        {
            "schema_version": LABEL_BUNDLE_SCHEMA,
            "group_id": "g",
            "candidate_count": 2,
            "labels": [
                {"candidate_id": "a", "relevance": 1},
                {"candidate_id": "b", "relevance": 0},
            ],
        },
    )
    with pytest.raises(StageInputError, match="universes differ"):
        assemble_labeled_feature_rows(features, [labels], output)
    assert not output.exists()


def test_train_hook_rejects_test_rows_and_cross_split_candidate_overlap(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    atomic_json(
        split_path,
        SceneSplit(
            train=("scene_train",),
            validation=("scene_validation",),
            test=("scene_test",),
            source_profile="unit fixture",
        ).to_dict(),
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    base_train = {
        "group_id": "train-group",
        "candidate_id": "train-candidate",
        "scene_id": "scene_train",
        "split": "train",
        "relevance": 1,
        "native_score": 0.9,
    }
    base_validation = {
        "group_id": "validation-group",
        "candidate_id": "validation-candidate",
        "scene_id": "scene_validation",
        "split": "validation",
        "relevance": 1,
        "native_score": 0.8,
    }

    atomic_jsonl(train_path, [{**base_train, "split": "test", "scene_id": "scene_test"}])
    atomic_jsonl(validation_path, [base_validation])
    with pytest.raises(StageInputError, match="only split='train'"):
        run_train_ranker_hook(
            train_path,
            validation_path,
            tmp_path / "test-leak-output",
            split_manifest_path=split_path,
            feature_columns=["native_score"],
            config={"seed": 20260815},
        )

    atomic_jsonl(train_path, [base_train])
    atomic_jsonl(
        validation_path,
        [{**base_validation, "candidate_id": base_train["candidate_id"]}],
    )
    with pytest.raises(StageInputError, match="candidates overlap"):
        run_train_ranker_hook(
            train_path,
            validation_path,
            tmp_path / "candidate-overlap-output",
            split_manifest_path=split_path,
            feature_columns=["native_score"],
            config={"seed": 20260815},
        )


def test_official_label_schema_assembles_to_parquet_and_recomputes_metrics(
    tmp_path: Path,
) -> None:
    group_id = "scene_test_kinect_0000_obj_001"
    candidate_id = f"{group_id}:candidate-0001"
    features_path = tmp_path / "features.jsonl"
    atomic_jsonl(
        features_path,
        [
            {
                "group_id": group_id,
                "candidate_id": candidate_id,
                "scene_id": "scene_test",
                "split": "test",
                "native_rank": 1,
                "native_score": 0.9,
            }
        ],
    )
    evaluation = CandidateEvaluation(
        candidate_index=0,
        associated_instance_index=0,
        associated_object_id=1,
        target_object_id=1,
        correct_target=True,
        collision=False,
        empty_grasp=False,
        friction_score=0.4,
        valid_geometry=True,
        relevance=5,
    ).to_record()
    labels_path = tmp_path / "labels.json"
    atomic_json(
        labels_path,
        {
            "schema_version": LABEL_BUNDLE_SCHEMA,
            "group_id": group_id,
            "candidate_count": 1,
            "labels": [{"candidate_id": candidate_id, **evaluation}],
        },
    )
    joined_path = tmp_path / "joined.parquet"
    assemble_labeled_feature_rows(features_path, [labels_path], joined_path)
    joined = __import__("pandas").read_parquet(joined_path)
    assert joined.loc[0, "pose_valid"]
    assert joined.loc[0, "friction_required"] == pytest.approx(0.4)

    universe_path = tmp_path / "groups.jsonl"
    atomic_jsonl(
        universe_path,
        [{"group_id": group_id, "scene_id": "scene_test"}],
    )
    output_root = tmp_path / "evaluation-output"
    first = run_evaluate_hook(
        joined_path,
        universe_path,
        output_root,
        score_column="native_score",
        max_k=2,
    )
    second = run_evaluate_hook(
        joined_path,
        universe_path,
        output_root,
        score_column="native_score",
        max_k=2,
        resume=True,
    )
    assert first == second
    assert first["metrics"]["group_count"] == 1
    assert first["metrics"]["target_p_at_1_mu_0.4"] == pytest.approx(1.0)
    assert first["metrics"]["target_p_at_1_mu_0.2"] == pytest.approx(0.0)
