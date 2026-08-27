"""Contract fixtures only; no row emitted here is formal experiment evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from scipy.io import savemat

import graspnet6d.formal_inputs as formal_inputs
from graspnet6d.contracts import Candidate6D, candidate_pool_fingerprint
from graspnet6d.feature_extraction import extract_candidate_features
from graspnet6d.grounding import (
    GroundingPrediction,
    HiFiModelBundle,
    validate_adaptation_splits,
)
from graspnet6d.io import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    canonical_sha256,
    sha256_file,
)
from graspnet6d.stages import (
    StageBatchError,
    StageInputError,
    TSDF_SCHEMA,
    VGN_BUNDLE_SCHEMA,
    run_oracle_mask_stage,
)


def _manifests(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    source = tmp_path / "inputs"
    source.mkdir()
    height, width = 8, 8
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[2:6, 2:6] = (200, 40, 20)
    depth = np.full((height, width), 1000, dtype=np.uint16)
    label = np.zeros((height, width), dtype=np.uint16)
    label[2:6, 2:6] = 1
    rgb_path = source / "rgb.png"
    depth_path = source / "depth.png"
    label_path = source / "label.png"
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(depth).save(depth_path)
    Image.fromarray(label).save(label_path)

    intrinsics = np.array(
        [[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsics_path = source / "camK.npy"
    poses_path = source / "camera_poses.npy"
    table_path = source / "cam0_wrt_table.npy"
    meta_path = source / "meta.mat"
    np.save(intrinsics_path, intrinsics, allow_pickle=False)
    np.save(poses_path, np.eye(4, dtype=np.float64)[None], allow_pickle=False)
    np.save(table_path, np.eye(4, dtype=np.float64), allow_pickle=False)
    savemat(
        meta_path,
        {"factor_depth": np.array([[1000.0]]), "intrinsic_matrix": intrinsics},
    )
    target = {
        "group_id": "scene_0000_kinect_0000_obj_000",
        "split": "train",
        "scene_id": "scene_0000",
        "camera": "kinect",
        "frame_id": 0,
        "target_object_id": 0,
        "target_instance_label": 1,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "instance_label_path": str(label_path),
        "meta_path": str(meta_path),
        "intrinsics_path": str(intrinsics_path),
        "camera_pose_path": str(poses_path),
        "table_transform_path": str(table_path),
    }
    language = {
        "group_id": target["group_id"],
        "query": "the red box",
        "template_family": "unit_test_fixture",
        "is_unique": True,
        "resolver_result": [0],
        "provenance": "derived_test_fixture",
    }
    target_path = tmp_path / "targets.jsonl"
    language_path = tmp_path / "language.jsonl"
    atomic_jsonl(target_path, [target])
    atomic_jsonl(language_path, [language])
    return target_path, language_path, target


def _fake_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HiFiModelBundle:
    checkpoint = tmp_path / "fake_hifi.pth"
    clip_weight = tmp_path / "fake_clip.pt"
    checkpoint.write_bytes(b"unit-test-hifi")
    clip_weight.write_bytes(b"unit-test-clip")
    checkpoint_sha = sha256_file(checkpoint)
    clip_sha = sha256_file(clip_weight)
    monkeypatch.setattr(
        formal_inputs, "EXPECTED_HIFI_CHECKPOINT_SHA256", checkpoint_sha
    )
    monkeypatch.setattr(formal_inputs, "EXPECTED_CLIP_WEIGHT_SHA256", clip_sha)
    model = torch.nn.Linear(1, 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return HiFiModelBundle(
        model=model,
        device=torch.device("cpu"),
        mode="zero_shot",
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        clip_weight_path=clip_weight,
        clip_weight_sha256=clip_sha,
        checkpoint_metadata=MappingProxyType({"fixture_only": True}),
        adaptation_parameter_names=(),
    )


def _fake_predictor(call_count: list[int]):
    def predict(
        bundle: HiFiModelBundle,
        rgb: Path,
        query: str,
        *,
        foreground_threshold: float,
    ) -> GroundingPrediction:
        call_count.append(1)
        with Image.open(rgb) as image:
            height, width = image.height, image.width
        native = np.full((height, width), 0.1, dtype=np.float32)
        native[2:6, 2:6] = 0.9
        return GroundingPrediction(
            probability_352=np.full((352, 352), 0.25, dtype=np.float32),
            native_probability=native,
            native_mask=native >= foreground_threshold,
            native_height=height,
            native_width=width,
            foreground_threshold=foreground_threshold,
            query=query,
            device=str(bundle.device),
        )

    return predict


def _run_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_path: Path,
    language_path: Path,
    *,
    output_root: Path | None = None,
    calls: list[int] | None = None,
):
    bundle = _fake_bundle(tmp_path, monkeypatch)
    counter = calls if calls is not None else []

    def loader(**_: Any) -> HiFiModelBundle:
        return bundle

    root = output_root or (tmp_path / "run")
    return formal_inputs.run_predicted_mask_stage(
        target_path,
        language_path,
        root,
        model_loader=loader,
        predictor=_fake_predictor(counter),
    )


def _candidate(group_id: str, index: int) -> Candidate6D:
    return Candidate6D(
        candidate_id=f"{group_id}:candidate-{index}",
        group_id=group_id,
        native_rank=index,
        native_score=0.9 - 0.1 * index,
        translation_local_m=(0.15, 0.15, 0.15),
        rotation_local=np.eye(3),
        translation_camera_m=(0.005 * index, 0.0, 1.0),
        rotation_camera=np.eye(3),
        translation_table_m=(0.005 * index, 0.0, 1.0),
        rotation_table=np.eye(3),
        width_m=0.06,
        height_m=0.02,
        depth_m=0.04,
        voxel_index=(20 + index, 20, 20),
        conversion_provenance={
            "mapping_name": "unit_test_identity",
            "mapping_status": "fixture_only",
            "source": "deterministic_test",
        },
    )


def _candidate_bundle(
    output_root: Path,
    group_id: str,
    count: int = 2,
    *,
    condition: str = "hifics_zero_shot_mask",
) -> Path:
    candidates = [_candidate(group_id, index) for index in range(count)]
    evaluator_rows = [
        [
            candidate.native_score,
            candidate.width_m,
            candidate.height_m,
            candidate.depth_m,
            *np.asarray(candidate.rotation_camera).reshape(-1),
            *np.asarray(candidate.translation_camera_m),
            -1.0,
        ]
        for candidate in candidates
    ]
    if condition == "oracle_gt_mask":
        _, mask_sidecar = formal_inputs.oracle_mask_paths(output_root, group_id)
    else:
        _, _, mask_sidecar = formal_inputs.predicted_mask_paths(
            output_root, condition, group_id
        )
    mask_commit = json.loads(mask_sidecar.read_text(encoding="utf-8"))
    mask_input_fingerprint = mask_commit["input_fingerprint"]
    mask_commit_sha = sha256_file(mask_sidecar)
    slug = formal_inputs.group_artifact_slug(group_id)
    tsdf_path = output_root / "target_tsdf" / condition / f"{slug}.npz"
    atomic_npz(tsdf_path, fixture=np.asarray([1], dtype=np.uint8))
    atomic_json(
        tsdf_path.with_suffix(".json"),
        {
            "schema_version": TSDF_SCHEMA,
            "group_id": group_id,
            "grounding_condition": condition,
            "grounding_mask_input_fingerprint": mask_input_fingerprint,
            "grounding_mask_commit_sha256": mask_commit_sha,
            "output_sha256": sha256_file(tsdf_path),
        },
    )
    checkpoint_path = output_root / "fixture_vgn_checkpoint.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"fixture-vgn-checkpoint")
    payload: dict[str, Any] = {
        "schema_version": VGN_BUNDLE_SCHEMA,
        "group_id": group_id,
        "input_fingerprint": "f" * 64,
        "candidate_count": count,
        "raw_vgn_candidates": [
            {"candidate_id": candidate.candidate_id} for candidate in candidates
        ],
        "candidate_records": [candidate.to_dict() for candidate in candidates],
        "candidate_pool_fingerprint": candidate_pool_fingerprint(candidates),
        "graspnet_rows": evaluator_rows,
        "geometry_contract": {"fixture_only": True},
        "grounding_condition": condition,
        "grounding_mask_input_fingerprint": mask_input_fingerprint,
        "grounding_mask_commit_sha256": mask_commit_sha,
        "tsdf_path": str(tsdf_path),
        "tsdf_sha256": sha256_file(tsdf_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    path = (
        output_root
        / "vgn_candidates"
        / condition
        / f"{formal_inputs.group_artifact_slug(group_id)}.json"
    )
    atomic_json(path, payload)
    return path


def test_predicted_mask_stage_commits_resumes_and_rejects_stale_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    calls: list[int] = []
    first = _run_masks(tmp_path, monkeypatch, target_path, language_path, calls=calls)
    assert first.completed_groups == 1
    assert first.resumed_groups == 0
    assert len(calls) == 1

    bundle = _fake_bundle(tmp_path, monkeypatch)
    second = formal_inputs.run_predicted_mask_stage(
        target_path,
        language_path,
        tmp_path / "run",
        resume=True,
        model_loader=lambda **_: bundle,
        predictor=_fake_predictor(calls),
    )
    assert second.completed_groups == 0
    assert second.resumed_groups == 1
    assert len(calls) == 1

    changed = np.full((8, 8, 3), 17, dtype=np.uint8)
    Image.fromarray(changed).save(target["rgb_path"])
    with pytest.raises(StageBatchError) as caught:
        formal_inputs.run_predicted_mask_stage(
            target_path,
            language_path,
            tmp_path / "run",
            resume=True,
            model_loader=lambda **_: bundle,
            predictor=_fake_predictor(calls),
        )
    assert "stale predicted mask input fingerprint" in caught.value.failures[0].message
    error_payload = json.loads(
        Path(caught.value.failures[0].error_path).read_text(encoding="utf-8")
    )
    assert error_payload["group_id"] == target["group_id"]
    assert "Traceback" in error_payload["traceback"]
    assert len(calls) == 1


def test_predicted_loader_rejects_actual_oracle_cache(
    tmp_path: Path,
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    run_oracle_mask_stage(target_path, language_path, tmp_path / "oracle")
    oracle_sidecar = next((tmp_path / "oracle" / "oracle_masks").glob("*.json"))
    with pytest.raises(StageInputError, match="reject non-HiFi mask caches"):
        formal_inputs.load_committed_predicted_mask(
            oracle_sidecar,
            expected_group_id=target["group_id"],
            expected_condition="hifics_zero_shot_mask",
        )


def test_formal_features_preserve_ids_and_resume_without_reextracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    root = tmp_path / "run"
    _run_masks(
        tmp_path,
        monkeypatch,
        target_path,
        language_path,
        output_root=root,
    )
    candidates = [_candidate(target["group_id"], index) for index in range(2)]
    _candidate_bundle(root, target["group_id"], count=2)
    extractor_calls: list[int] = []

    def extractor(values, observation):
        extractor_calls.append(1)
        return extract_candidate_features(values, observation)

    first = formal_inputs.run_formal_feature_stage(
        target_path,
        language_path,
        root,
        condition="hifics_zero_shot_mask",
        extractor=extractor,
    )
    assert first.completed_groups == 1
    assert len(extractor_calls) == 1
    sidecar = Path(first.output_paths[0])
    frame = formal_inputs.load_committed_formal_feature_table(
        sidecar,
        expected_group_id=target["group_id"],
        expected_condition="hifics_zero_shot_mask",
        expected_candidate_ids=[candidate.candidate_id for candidate in candidates],
    )
    assert frame[
        ["group_id", "candidate_id", "scene_id", "split", "condition"]
    ].to_dict("records") == [
        {
            "group_id": target["group_id"],
            "candidate_id": candidate.candidate_id,
            "scene_id": target["scene_id"],
            "split": "train",
            "condition": "hifics_zero_shot_mask",
        }
        for candidate in candidates
    ]

    second = formal_inputs.run_formal_feature_stage(
        target_path,
        language_path,
        root,
        condition="hifics_zero_shot_mask",
        extractor=extractor,
        resume=True,
    )
    assert second.completed_groups == 0
    assert second.resumed_groups == 1
    assert len(extractor_calls) == 1


def test_formal_feature_stage_rejects_extractor_membership_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    source_root = tmp_path / "source"
    _run_masks(
        tmp_path,
        monkeypatch,
        target_path,
        language_path,
        output_root=source_root,
    )
    _candidate_bundle(source_root, target["group_id"], count=2)

    def dropping_extractor(values, observation) -> pd.DataFrame:
        return extract_candidate_features(values, observation).iloc[:-1].copy()

    with pytest.raises(StageBatchError) as caught:
        formal_inputs.run_formal_feature_stage(
            target_path,
            language_path,
            tmp_path / "bad_features",
            condition="hifics_zero_shot_mask",
            mask_output_root=source_root,
            candidate_output_root=source_root,
            extractor=dropping_extractor,
        )
    failure = caught.value.failures[0]
    assert "changed candidate membership" in failure.message
    error_payload = json.loads(Path(failure.error_path).read_text(encoding="utf-8"))
    assert error_payload["group_id"] == target["group_id"]
    assert "Traceback" in error_payload["traceback"]


def test_predicted_features_reject_oracle_centred_candidate_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    root = tmp_path / "condition_mismatch"
    _run_masks(
        tmp_path,
        monkeypatch,
        target_path,
        language_path,
        output_root=root,
    )
    candidate_path = _candidate_bundle(root, target["group_id"], count=1)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["grounding_condition"] = "oracle_gt_mask"
    payload.pop("bundle_fingerprint")
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    atomic_json(candidate_path, payload)
    with pytest.raises(StageBatchError) as caught:
        formal_inputs.run_formal_feature_stage(
            target_path,
            language_path,
            root,
            condition="hifics_zero_shot_mask",
        )
    assert "grounding lineage mismatch" in caught.value.failures[0].message


def test_empty_candidate_pool_commits_schema_bearing_zero_row_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    root = tmp_path / "empty_pool"
    _run_masks(
        tmp_path,
        monkeypatch,
        target_path,
        language_path,
        output_root=root,
    )
    _candidate_bundle(root, target["group_id"], count=0)
    summary = formal_inputs.run_formal_feature_stage(
        target_path,
        language_path,
        root,
        condition="hifics_zero_shot_mask",
    )
    frame = formal_inputs.load_committed_formal_feature_table(
        summary.output_paths[0],
        expected_group_id=target["group_id"],
        expected_condition="hifics_zero_shot_mask",
        expected_candidate_ids=[],
    )
    assert frame.empty
    assert list(frame.columns[:5]) == [
        "group_id",
        "candidate_id",
        "scene_id",
        "split",
        "condition",
    ]
    commit = json.loads(Path(summary.output_paths[0]).read_text(encoding="utf-8"))
    assert commit["candidate_count"] == 0
    assert commit["candidate_ids"] == []


def test_oracle_condition_uses_verified_official_mask_only_in_named_track(
    tmp_path: Path,
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    root = tmp_path / "oracle_features"
    run_oracle_mask_stage(target_path, language_path, root)
    _candidate_bundle(
        root,
        target["group_id"],
        count=1,
        condition="oracle_gt_mask",
    )
    summary = formal_inputs.run_formal_feature_stage(
        target_path,
        language_path,
        root,
        condition="oracle_gt_mask",
    )
    frame = formal_inputs.load_committed_formal_feature_table(
        summary.output_paths[0],
        expected_group_id=target["group_id"],
        expected_condition="oracle_gt_mask",
        expected_candidate_ids=[_candidate(target["group_id"], 0).candidate_id],
    )
    assert frame["condition"].tolist() == ["oracle_gt_mask"]
    assert "target_object_id" not in frame.columns
    commit = json.loads(Path(summary.output_paths[0]).read_text(encoding="utf-8"))
    assert commit["ground_truth_inputs_consumed"] == [
        "official_instance_label_pixels_for_explicit_oracle_gt_mask_only"
    ]


def _adaptation_evidence(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    train_rgb = tmp_path / "adapt_train_rgb.png"
    train_label = tmp_path / "adapt_train_label.png"
    val_rgb = tmp_path / "adapt_val_rgb.png"
    val_label = tmp_path / "adapt_val_label.png"
    labels = np.zeros((4, 5), dtype=np.uint16)
    labels[1:3, 2:4] = 3
    Image.fromarray(np.full((4, 5, 3), 20, dtype=np.uint8)).save(train_rgb)
    Image.fromarray(labels).save(train_label)
    Image.fromarray(np.full((4, 5, 3), 40, dtype=np.uint8)).save(val_rgb)
    Image.fromarray(labels).save(val_label)
    train_rows = [
        {
            "sample_id": "train-a",
            "scene_id": "scene_0000",
            "split": "train",
            "query": "the train fixture object",
            "rgb_path": str(train_rgb),
            "rgb_sha256": sha256_file(train_rgb),
            "instance_label_path": str(train_label),
            "instance_label_sha256": sha256_file(train_label),
            "target_instance_label": 3,
        }
    ]
    validation_rows = [
        {
            "sample_id": "val-a",
            "scene_id": "scene_0100",
            "split": "val",
            "query": "the validation fixture object",
            "rgb_path": str(val_rgb),
            "rgb_sha256": sha256_file(val_rgb),
            "instance_label_path": str(val_label),
            "instance_label_sha256": sha256_file(val_label),
            "target_instance_label": 3,
        }
    ]
    train_path = tmp_path / "adapt_train.jsonl"
    validation_path = tmp_path / "adapt_val.jsonl"
    atomic_jsonl(train_path, train_rows)
    atomic_jsonl(validation_path, validation_rows)
    split_contract = dict(validate_adaptation_splits(train_rows, validation_rows))
    checkpoint_path = tmp_path / "adapted.pth"
    torch.save(
        {
            "format": "graspnet6d_hifi_decoder_adaptation_v1",
            "base_checkpoint_sha256": formal_inputs.EXPECTED_HIFI_CHECKPOINT_SHA256,
            "clip_weight_sha256": formal_inputs.EXPECTED_CLIP_WEIGHT_SHA256,
            "trainable_state": {"decoder.fixture": torch.ones(1)},
            "metadata": {
                "best_epoch": 2,
                "best_validation_mean_iou": 0.75,
                "selection_metric": "validation_mean_iou",
                "selection_split": "val",
                "test_rows_consumed": 0,
                "split_contract": split_contract,
            },
        },
        checkpoint_path,
    )
    evidence: dict[str, Any] = {
        "schema_version": formal_inputs.ADAPTATION_EVIDENCE_SCHEMA,
        "adaptation_checkpoint_path": checkpoint_path.name,
        "adaptation_checkpoint_sha256": sha256_file(checkpoint_path),
        "base_checkpoint_sha256": formal_inputs.EXPECTED_HIFI_CHECKPOINT_SHA256,
        "clip_weight_sha256": formal_inputs.EXPECTED_CLIP_WEIGHT_SHA256,
        "train_manifest_path": train_path.name,
        "train_manifest_sha256": sha256_file(train_path),
        "validation_manifest_path": validation_path.name,
        "validation_manifest_sha256": sha256_file(validation_path),
        "selection_metric": "validation_mean_iou",
        "selection_split": "val",
        "best_validation_mean_iou": 0.75,
        "test_rows_consumed": 0,
        "input_splits": ["train", "val"],
        "split_contract": split_contract,
    }
    evidence["evidence_fingerprint"] = canonical_sha256(evidence)
    evidence_path = tmp_path / "adaptation_evidence.json"
    atomic_json(evidence_path, evidence)
    return evidence_path, evidence


def test_adaptation_evidence_is_hash_bound_and_scene_disjoint(tmp_path: Path) -> None:
    evidence_path, evidence = _adaptation_evidence(tmp_path)
    validated = formal_inputs.validate_adaptation_evidence(evidence_path)
    assert validated.best_validation_mean_iou == pytest.approx(0.75)
    assert validated.split_contract["scene_overlap"] == 0
    assert validated.split_contract["test_rows_consumed"] == 0

    validation_path = tmp_path / "adapt_val.jsonl"
    validation_record = json.loads(validation_path.read_text(encoding="utf-8").strip())
    validation_record["scene_id"] = "scene_0000"
    atomic_jsonl(
        validation_path,
        [validation_record],
    )
    evidence["validation_manifest_sha256"] = sha256_file(validation_path)
    evidence.pop("evidence_fingerprint")
    evidence["evidence_fingerprint"] = canonical_sha256(evidence)
    atomic_json(evidence_path, evidence)
    with pytest.raises(StageInputError, match="scene_overlap"):
        formal_inputs.validate_adaptation_evidence(evidence_path)


def test_adaptation_evidence_rehashes_rgb_and_instance_sources(tmp_path: Path) -> None:
    evidence_path, _ = _adaptation_evidence(tmp_path)
    train_row = json.loads(
        (tmp_path / "adapt_train.jsonl").read_text(encoding="utf-8").strip()
    )
    Image.fromarray(np.full((4, 5, 3), 99, dtype=np.uint8)).save(train_row["rgb_path"])
    with pytest.raises(StageInputError, match="stale rgb_path"):
        formal_inputs.validate_adaptation_evidence(evidence_path)


def test_adapted_mask_stage_requires_evidence_before_loading_a_model(
    tmp_path: Path,
) -> None:
    target_path, language_path, _ = _manifests(tmp_path)
    with pytest.raises(StageBatchError) as caught:
        formal_inputs.run_predicted_mask_stage(
            target_path,
            language_path,
            tmp_path / "run",
            condition="hifics_adapted_mask",
        )
    assert "require a versioned adaptation evidence" in caught.value.failures[0].message
