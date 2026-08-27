"""Regression tests for honest zero-candidate grounding terminal outcomes.

All inputs are tiny local fixtures.  These tests verify orchestration contracts;
they do not create formal experiment evidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat

import graspnet6d.analysis_inputs as analysis_inputs
import graspnet6d.formal_inputs as formal_inputs
from graspnet6d.features import default_feature_schema
from graspnet6d.grounding import GroundingPrediction, HiFiModelBundle
from graspnet6d.io import atomic_json, atomic_jsonl, canonical_sha256, sha256_file
from graspnet6d.stages import (
    GEOMETRY_SCHEMA,
    GROUNDING_TERMINAL_SCHEMA,
    LABEL_BUNDLE_SCHEMA,
    PARITY_SCHEMA,
    StageBatchError,
    VGN_BUNDLE_SCHEMA,
    grounding_terminal_path,
    run_official_label_stage,
    run_tsdf_stage,
    run_vgn_candidate_stage,
)


CONDITION = "hifics_zero_shot_mask"
SHA256 = re.compile(r"[0-9a-f]{64}")


def _manifests(
    tmp_path: Path, *, zero_depth_in_foreground: bool = False
) -> tuple[Path, Path, dict[str, Any]]:
    source = tmp_path / "inputs"
    source.mkdir()
    height = width = 8
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[2:6, 2:6] = (200, 40, 20)
    depth = np.full((height, width), 1000, dtype=np.uint16)
    if zero_depth_in_foreground:
        depth[2:6, 2:6] = 0
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
    camera_poses_path = source / "camera_poses.npy"
    table_path = source / "cam0_wrt_table.npy"
    meta_path = source / "meta.mat"
    np.save(intrinsics_path, intrinsics, allow_pickle=False)
    np.save(camera_poses_path, np.eye(4, dtype=np.float64)[None], allow_pickle=False)
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
        "camera_pose_path": str(camera_poses_path),
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


def _fake_hifi_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> HiFiModelBundle:
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


def _run_predicted_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_path: Path,
    language_path: Path,
    output_root: Path,
    *,
    empty: bool,
) -> None:
    bundle = _fake_hifi_bundle(tmp_path, monkeypatch)

    def predictor(
        _bundle: HiFiModelBundle,
        rgb_path: Path,
        query: str,
        *,
        foreground_threshold: float,
    ) -> GroundingPrediction:
        with Image.open(rgb_path) as image:
            height, width = image.height, image.width
        probability = np.zeros((height, width), dtype=np.float32)
        if not empty:
            probability[2:6, 2:6] = 0.9
        return GroundingPrediction(
            probability_352=np.zeros((352, 352), dtype=np.float32),
            native_probability=probability,
            native_mask=probability >= foreground_threshold,
            native_height=height,
            native_width=width,
            foreground_threshold=foreground_threshold,
            query=query,
            device=str(bundle.device),
        )

    formal_inputs.run_predicted_mask_stage(
        target_path,
        language_path,
        output_root,
        model_loader=lambda **_: bundle,
        predictor=predictor,
    )


def _geometry_gate(tmp_path: Path) -> Path:
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
            "height_m": 0.02,
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


@pytest.mark.parametrize(
    ("empty_mask", "zero_depth_in_foreground", "expected_reason_fragment"),
    [
        (True, False, "empty"),
        (False, True, "depth"),
    ],
)
def test_predicted_grounding_terminal_short_circuits_all_downstream_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_mask: bool,
    zero_depth_in_foreground: bool,
    expected_reason_fragment: str,
) -> None:
    target_path, language_path, target = _manifests(
        tmp_path, zero_depth_in_foreground=zero_depth_in_foreground
    )
    root = tmp_path / "run"
    _run_predicted_mask(
        tmp_path,
        monkeypatch,
        target_path,
        language_path,
        root,
        empty=empty_mask,
    )

    tsdf = run_tsdf_stage(
        target_path,
        language_path,
        root,
        grounding_condition=CONDITION,
        builder=lambda *_args, **_kwargs: pytest.fail(
            "TSDF builder must not run for a grounding terminal observation"
        ),
    )
    terminal_path = grounding_terminal_path(root, target["group_id"], CONDITION)
    assert Path(tsdf.output_paths[0]) == terminal_path
    assert terminal_path.is_file()
    assert not list((root / "target_tsdf" / CONDITION).glob("*.npz"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["schema_version"] == GROUNDING_TERMINAL_SCHEMA
    assert terminal["group_id"] == target["group_id"]
    assert terminal["grounding_condition"] == CONDITION
    assert expected_reason_fragment in str(terminal["reason"]).lower()
    assert SHA256.fullmatch(str(terminal["input_fingerprint"]))
    assert terminal["tsdf_constructed"] is False
    assert terminal["vgn_inference_calls_for_group"] == 0
    assert terminal["candidate_count"] == 0
    assert terminal["candidate_ids"] == []
    assert terminal["source_hashes"]
    for source, digest in terminal["source_hashes"].items():
        assert Path(source).is_file()
        assert sha256_file(Path(source)) == digest

    checkpoint = tmp_path / "vgn_checkpoint.pth"
    checkpoint.write_bytes(b"fixture-vgn-checkpoint")
    calls = {"load": 0, "infer": 0, "extract": 0}

    def forbidden(name: str):
        def invoke(*_args: Any, **_kwargs: Any) -> Any:
            calls[name] += 1
            pytest.fail(f"{name} must not run for a grounding terminal observation")

        return invoke

    geometry_path = _geometry_gate(tmp_path)
    candidate_summary = run_vgn_candidate_stage(
        target_path,
        language_path,
        root,
        geometry_contract_path=geometry_path,
        checkpoint=checkpoint,
        grounding_condition=CONDITION,
        evidence_policy="test",
        model_loader=forbidden("load"),
        inference=forbidden("infer"),
        extractor=forbidden("extract"),
    )
    assert calls == {"load": 0, "infer": 0, "extract": 0}
    candidate_path = Path(candidate_summary.output_paths[0])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["schema_version"] == VGN_BUNDLE_SCHEMA
    assert candidate["generation_status"] == "skipped_grounding_failure"
    assert candidate["grounding_terminal_path"] == str(terminal_path)
    assert candidate["grounding_terminal_sha256"] == sha256_file(terminal_path)
    assert candidate["tsdf_path"] is None
    assert candidate["tsdf_sha256"] is None
    assert candidate["inference_calls_for_group"] == 0
    assert candidate["candidate_count"] == 0
    assert candidate["raw_vgn_candidates"] == []
    assert candidate["candidate_records"] == []
    assert candidate["graspnet_rows"] == []
    resumed_candidates = run_vgn_candidate_stage(
        target_path,
        language_path,
        root,
        geometry_contract_path=geometry_path,
        checkpoint=checkpoint,
        grounding_condition=CONDITION,
        evidence_policy="test",
        resume=True,
        model_loader=forbidden("load"),
        inference=forbidden("infer"),
        extractor=forbidden("extract"),
    )
    assert resumed_candidates.resumed_groups == 1
    assert calls == {"load": 0, "infer": 0, "extract": 0}

    dataset_root = tmp_path / "official_dataset_fixture"
    dataset_root.mkdir()
    label_calls = {"load": 0, "evaluate": 0}

    def forbidden_label(name: str):
        def invoke(*_args: Any, **_kwargs: Any) -> Any:
            label_calls[name] += 1
            pytest.fail(f"{name} must not run for a zero-candidate pool")

        return invoke

    parity_path = _parity_gate(tmp_path)
    label_summary = run_official_label_stage(
        target_path,
        language_path,
        root,
        dataset_root=dataset_root,
        parity_gate_path=parity_path,
        grounding_condition=CONDITION,
        evidence_policy="test",
        input_loader=forbidden_label("load"),
        evaluator=forbidden_label("evaluate"),
    )
    assert label_calls == {"load": 0, "evaluate": 0}
    labels = json.loads(Path(label_summary.output_paths[0]).read_text(encoding="utf-8"))
    assert labels["schema_version"] == LABEL_BUNDLE_SCHEMA
    assert labels["candidate_count"] == 0
    assert labels["candidate_ids"] == []
    assert labels["labels"] == []
    assert labels["official_source_hashes"] == {}
    assert labels["evaluator_calls_for_group"] == 0
    assert labels["evaluator_operation"] == "not_called_empty_frozen_pool"
    assert labels["grounding_terminal_path"] == str(terminal_path)
    assert labels["grounding_terminal_sha256"] == sha256_file(terminal_path)
    assert labels["parity_gate"]
    resumed_labels = run_official_label_stage(
        target_path,
        language_path,
        root,
        dataset_root=dataset_root,
        parity_gate_path=parity_path,
        grounding_condition=CONDITION,
        evidence_policy="test",
        resume=True,
        input_loader=forbidden_label("load"),
        evaluator=forbidden_label("evaluate"),
    )
    assert resumed_labels.resumed_groups == 1
    assert label_calls == {"load": 0, "evaluate": 0}

    monkeypatch.setattr(
        formal_inputs,
        "_depth_observation",
        lambda *_args, **_kwargs: pytest.fail(
            "RuntimeObservation must not be constructed for a zero-candidate pool"
        ),
    )
    feature_summary = formal_inputs.run_formal_feature_stage(
        target_path,
        language_path,
        root,
        condition=CONDITION,
        extractor=lambda *_args, **_kwargs: pytest.fail(
            "feature extractor must not run for a zero-candidate pool"
        ),
    )
    feature_sidecar = Path(feature_summary.output_paths[0])
    frame = formal_inputs.load_committed_formal_feature_table(
        feature_sidecar,
        expected_group_id=target["group_id"],
        expected_condition=CONDITION,
        expected_candidate_ids=[],
    )
    assert frame.empty
    assert list(frame.columns) == [
        "group_id",
        "candidate_id",
        "scene_id",
        "split",
        "condition",
        *(spec.name for spec in default_feature_schema()),
    ]

    # Assembly keeps this group in its group universe even though it contributes
    # no candidate rows.  Expensive formal geometry/parity evidence validation is
    # independently covered elsewhere; here the fixture gates are deliberately
    # bypassed so this test can focus on terminal provenance preservation.
    monkeypatch.setattr(
        analysis_inputs, "_validate_geometry_evidence", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        analysis_inputs, "_validate_parity_evidence", lambda *_args, **_kwargs: None
    )
    assembled = analysis_inputs._assemble_group(
        group_id=target["group_id"],
        target=target,
        condition=CONDITION,
        target_manifest=target_path.resolve(),
        target_manifest_sha256=sha256_file(target_path),
        language_manifest=language_path.resolve(),
        language_manifest_sha256=sha256_file(language_path),
        feature_directory=root / "candidate_features" / CONDITION,
        label_directory=root / "official_labels" / CONDITION,
        candidate_directory=root / "vgn_candidates" / CONDITION,
        feature_names=[spec.name for spec in default_feature_schema()],
    )
    assert assembled.rows == ()
    assert assembled.provenance["candidate_count"] == 0
    assert assembled.provenance["generation_status"] == "skipped_grounding_failure"
    assert assembled.provenance["grounding_failure_reason"] == terminal["reason"]
    assert assembled.provenance["grounding_terminal_path"] == str(terminal_path)
    assert assembled.provenance["grounding_terminal_sha256"] == sha256_file(
        terminal_path
    )

    label_path = Path(label_summary.output_paths[0])
    corruptions = (
        {
            "evaluator_operation": (
                "per_candidate_low_level_no_eval_grasp_no_nms_no_topk"
            )
        },
        {"official_source_hashes": {str(parity_path): sha256_file(parity_path)}},
        {"grounding_terminal_sha256": "0" * 64},
    )
    for changes in corruptions:
        corrupted = dict(labels)
        corrupted.pop("bundle_fingerprint")
        corrupted.update(changes)
        corrupted["bundle_fingerprint"] = canonical_sha256(corrupted)
        atomic_json(label_path, corrupted)
        with pytest.raises(
            analysis_inputs.AnalysisInputAssemblyError,
            match="empty-pool labels",
        ):
            analysis_inputs._assemble_group(
                group_id=target["group_id"],
                target=target,
                condition=CONDITION,
                target_manifest=target_path.resolve(),
                target_manifest_sha256=sha256_file(target_path),
                language_manifest=language_path.resolve(),
                language_manifest_sha256=sha256_file(language_path),
                feature_directory=root / "candidate_features" / CONDITION,
                label_directory=root / "official_labels" / CONDITION,
                candidate_directory=root / "vgn_candidates" / CONDITION,
                feature_names=[spec.name for spec in default_feature_schema()],
            )
    atomic_json(label_path, labels)

    resumed_tsdf = run_tsdf_stage(
        target_path,
        language_path,
        root,
        grounding_condition=CONDITION,
        resume=True,
        builder=lambda *_args, **_kwargs: pytest.fail(
            "resuming a terminal must not run the TSDF builder"
        ),
    )
    assert resumed_tsdf.resumed_groups == 1


def test_grounding_terminal_resume_rejects_stale_depth_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path, language_path, target = _manifests(
        tmp_path, zero_depth_in_foreground=True
    )
    root = tmp_path / "run"
    _run_predicted_mask(
        tmp_path,
        monkeypatch,
        target_path,
        language_path,
        root,
        empty=False,
    )
    run_tsdf_stage(
        target_path,
        language_path,
        root,
        grounding_condition=CONDITION,
        builder=lambda *_args, **_kwargs: pytest.fail("builder must not run"),
    )
    terminal_path = grounding_terminal_path(root, target["group_id"], CONDITION)
    assert terminal_path.is_file()

    depth_path = Path(target["depth_path"])
    changed = np.full((8, 8), 1000, dtype=np.uint16)
    Image.fromarray(changed).save(depth_path)
    with pytest.raises(StageBatchError) as caught:
        run_tsdf_stage(
            target_path,
            language_path,
            root,
            grounding_condition=CONDITION,
            resume=True,
            builder=lambda *_args, **_kwargs: pytest.fail(
                "stale terminal must fail before TSDF construction"
            ),
        )
    assert "terminal" in caught.value.failures[0].message.lower()


@pytest.mark.parametrize("damage", ["missing_depth", "malformed_depth", "stale_mask"])
def test_malformed_or_missing_sources_do_not_become_grounding_terminals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    target_path, language_path, target = _manifests(tmp_path)
    root = tmp_path / "run"
    _run_predicted_mask(
        tmp_path,
        monkeypatch,
        target_path,
        language_path,
        root,
        empty=True,
    )
    if damage == "missing_depth":
        Path(target["depth_path"]).unlink()
    elif damage == "malformed_depth":
        Image.fromarray(np.zeros((7, 8), dtype=np.uint16)).save(target["depth_path"])
    else:
        _, mask_path, _ = formal_inputs.predicted_mask_paths(
            root, CONDITION, target["group_id"]
        )
        Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(mask_path)

    with pytest.raises(StageBatchError):
        run_tsdf_stage(
            target_path,
            language_path,
            root,
            grounding_condition=CONDITION,
            builder=lambda *_args, **_kwargs: pytest.fail(
                "malformed/missing/stale inputs must fail before TSDF construction"
            ),
        )
    assert not grounding_terminal_path(root, target["group_id"], CONDITION).exists()
