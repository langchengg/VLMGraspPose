from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import graspnet6d.experiment_analysis as analysis
from graspnet6d.formal_inputs import _load_candidate_bundle
from graspnet6d.io import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    canonical_sha256,
    sha256_file,
)
from graspnet6d.stages import (
    GEOMETRY_SCHEMA,
    TSDF_SCHEMA,
    StageBatchError,
    StageInputError,
    _slug,
    run_vgn_candidate_stage,
)
from graspnet6d.vgn import (
    ExtractionConfig,
    VGNExtractionSnapshot,
    VGNCandidate,
    pose_nms,
    validate_extraction_snapshot,
)


def _candidate(group_id: str, rank: int) -> VGNCandidate:
    if rank == 2:
        position = np.zeros(3, dtype=np.float64)
    elif rank == 1:
        position = np.zeros(3, dtype=np.float64)
    else:
        cell = rank - 2
        position = 0.03 * np.asarray(
            [cell % 5, (cell // 5) % 5, (cell // 25) % 4], dtype=np.float64
        )
    voxel = rank - 1
    return VGNCandidate(
        candidate_id=f"{group_id}:candidate-{rank:03d}",
        group_id=group_id,
        native_rank=rank,
        native_score=1.0 - rank / 1000.0,
        translation_local_m=position,
        rotation_local_vgn=np.eye(3),
        width_m=0.05,
        voxel_index=(voxel // 1600, (voxel // 40) % 40, voxel % 40),
        translation_camera_m=position.copy(),
        rotation_camera_vgn=np.eye(3),
        translation_table_m=position.copy(),
        rotation_table_vgn=np.eye(3),
    )


def _stage_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    group_id = "scene_0000_kinect_0000_obj_000"
    target = {
        "group_id": group_id,
        "split": "train",
        "scene_id": "scene_0000",
        "camera": "kinect",
        "frame_id": 0,
        "target_object_id": 0,
        "target_instance_label": 1,
        "depth_path": "/unused/depth.png",
        "instance_label_path": "/unused/label.png",
        "meta_path": "/unused/meta.mat",
        "intrinsics_path": "/unused/camK.npy",
        "camera_pose_path": "/unused/camera_poses.npy",
        "table_transform_path": "/unused/table.npy",
    }
    language = {
        "group_id": group_id,
        "query": "the unique object",
        "is_unique": True,
        "resolver_result": [0],
    }
    targets = tmp_path / "targets.jsonl"
    languages = tmp_path / "language.jsonl"
    atomic_jsonl(targets, [target])
    atomic_jsonl(languages, [language])
    root = tmp_path / "run"
    tsdf = root / "target_tsdf" / "oracle_gt_mask" / f"{_slug(group_id)}.npz"
    atomic_npz(
        tsdf,
        tsdf=np.zeros((1, 40, 40, 40), dtype=np.float32),
        T_local_to_camera=np.eye(4),
        T_local_to_table=np.eye(4),
        full_scene_depth_integrated=np.asarray(True),
        grounding_condition=np.asarray("oracle_gt_mask"),
        grounding_mask_input_fingerprint=np.asarray("a" * 64),
        grounding_mask_commit_sha256=np.asarray("b" * 64),
    )
    atomic_json(
        tsdf.with_suffix(".json"),
        {
            "schema_version": TSDF_SCHEMA,
            "group_id": group_id,
            "grounding_condition": "oracle_gt_mask",
            "grounding_mask_input_fingerprint": "a" * 64,
            "grounding_mask_commit_sha256": "b" * 64,
            "output_sha256": sha256_file(tsdf),
        },
    )
    artifact = tmp_path / "geometry_validation.json"
    atomic_json(artifact, {"status": "PASSED", "fixture_only": True})
    geometry = tmp_path / "geometry_contract.json"
    atomic_json(
        geometry,
        {
            "schema_version": GEOMETRY_SCHEMA,
            "validated": True,
            "validation_artifact": artifact.name,
            "R_vgn_gripper_to_graspnet_gripper": np.eye(3).tolist(),
            "height_m": 0.02,
            "depth_m": 0.04,
        },
    )
    checkpoint = tmp_path / "vgn.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    return targets, languages, root, geometry, group_id


def test_same_inference_snapshot_drives_all_a7_memberships_and_tamper_fails(
    tmp_path: Path,
) -> None:
    targets, languages, root, geometry, group_id = _stage_inputs(tmp_path)
    config = ExtractionConfig()
    pre_nms = tuple(_candidate(group_id, rank) for rank in range(1, 101))
    snapshot = validate_extraction_snapshot(
        VGNExtractionSnapshot(
            pre_nms,
            tuple(pose_nms(pre_nms, config)[: config.frozen_top_k]),
        ),
        group_id=group_id,
        config=config,
    )
    calls = {"load": 0, "inference": 0, "extract": 0}

    def load_model(**_: Any) -> object:
        calls["load"] += 1
        return object()

    def infer(*_: Any, **__: Any) -> object:
        calls["inference"] += 1
        return object()

    def extract(*_: Any, **__: Any) -> VGNExtractionSnapshot:
        calls["extract"] += 1
        return snapshot

    first = run_vgn_candidate_stage(
        targets,
        languages,
        root,
        geometry_contract_path=geometry,
        checkpoint=tmp_path / "vgn.pth",
        evidence_policy="test",
        model_loader=load_model,
        inference=infer,
        extractor=extract,
    )
    bundle_path = Path(first.output_paths[0])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert calls == {"load": 1, "inference": 1, "extract": 1}
    assert bundle["pre_nms_snapshot_status"] == "complete_same_inference"
    assert bundle["pre_nms_candidate_count"] == 100
    assert bundle["candidate_count"] == 50
    assert {
        key: value["candidate_count"]
        for key, value in bundle["a7_top_k_membership"].items()
    } == {"20": 19, "50": 49, "100": 50}

    valid_bundle = dict(bundle)
    partial = dict(bundle)
    partial.pop("pre_nms_pool_fingerprint")
    partial.pop("bundle_fingerprint")
    partial["bundle_fingerprint"] = canonical_sha256(partial)
    atomic_json(bundle_path, partial)
    with pytest.raises(StageInputError, match="partial A7 contract"):
        _load_candidate_bundle(bundle_path, group_id)
    missing = dict(valid_bundle)
    for field in (
        "pre_nms_snapshot_status",
        "pre_nms_candidate_count",
        "pre_nms_vgn_candidates",
        "pre_nms_pool_fingerprint",
        "a7_top_k_membership",
        "a7_top_k_membership_fingerprint",
    ):
        missing.pop(field)
    missing.pop("bundle_fingerprint")
    missing["bundle_fingerprint"] = canonical_sha256(missing)
    atomic_json(bundle_path, missing)
    with pytest.raises(StageInputError, match="mandatory A7"):
        _load_candidate_bundle(bundle_path, group_id)
    atomic_json(bundle_path, valid_bundle)

    resumed = run_vgn_candidate_stage(
        targets,
        languages,
        root,
        geometry_contract_path=geometry,
        checkpoint=tmp_path / "vgn.pth",
        evidence_policy="test",
        resume=True,
        model_loader=load_model,
        inference=infer,
        extractor=extract,
    )
    assert resumed.resumed_groups == 1
    assert calls == {"load": 1, "inference": 1, "extract": 1}

    bundle["a7_top_k_membership"]["20"] = {
        "candidate_count": 0,
        "candidate_ids": [],
        "candidate_ids_sha256": canonical_sha256([]),
    }
    bundle["a7_top_k_membership_fingerprint"] = canonical_sha256(
        bundle["a7_top_k_membership"]
    )
    bundle.pop("bundle_fingerprint")
    bundle["bundle_fingerprint"] = canonical_sha256(bundle)
    atomic_json(bundle_path, bundle)
    with pytest.raises(StageBatchError, match="vgn_candidates"):
        run_vgn_candidate_stage(
            targets,
            languages,
            root,
            geometry_contract_path=geometry,
            checkpoint=tmp_path / "vgn.pth",
            evidence_policy="test",
            resume=True,
            model_loader=lambda **_: pytest.fail("tamper must fail before model load"),
            inference=lambda *_args, **_kwargs: pytest.fail("must not re-infer"),
            extractor=lambda *_args, **_kwargs: pytest.fail("must not re-extract"),
        )


def test_formal_a7_executes_k20_k50_k100_from_pre_nms_rank(monkeypatch: Any) -> None:
    ranks = (1, 25, 75)
    rows = pd.DataFrame(
        [
            {
                "group_id": "test-group",
                "scene_id": "test-scene",
                "candidate_id": f"candidate-{rank}",
                "native_rank": rank,
                "pre_nms_native_rank": rank,
                "native_score": 1.0 - rank / 100.0,
                "raw_rerank_score": 1.0 - rank / 100.0,
                "collision": False,
                "pose_valid": True,
                "friction_required": 0.4,
                "target_object_id": 1,
                "associated_object_id": 1,
                "relevance": 5,
            }
            for rank in ranks
        ]
    )
    universe = pd.DataFrame([{"group_id": "test-group", "scene_id": "test-scene"}])
    fit = analysis._VariantFit(
        variant="all",
        feature_columns=("native_score",),
        selected_config={},
        selected_config_sha256="a" * 64,
        validation_trials=[],
        imputer_artifact={},
        validation_predictions=rows.copy(),
        test_predictions=rows.copy(),
        models=(),
    )
    monkeypatch.setattr(
        analysis, "select_feature_columns", lambda *_: ("native_score",)
    )
    monkeypatch.setattr(analysis, "_fit_variant", lambda *_args, **_kwargs: fit)
    ablations, _ = analysis._run_ablations(
        schema=object(),
        train=rows,
        validation=rows,
        test=rows,
        universes={"train": universe, "validation": universe, "test": universe},
        all_feature_fit=fit,
        base_metrics={"B0_NATIVE": {"metric": 1.0}, "R0_RAW": {"metric": 1.0}},
        gated_rows=rows.assign(gated_score=rows["raw_rerank_score"]),
        gate_artifact={"status": "GO"},
        scope=analysis.FORMAL_SCOPE,
        config=analysis.AnalysisConfig(),
    )
    a7 = ablations.loc[ablations["ablation"].eq("A7")]
    assert a7["status"].eq("EXECUTED").all()
    assert a7.groupby("setting")["actual_candidate_count"].first().to_dict() == {
        "top_k_20": 1,
        "top_k_50": 2,
        "top_k_100": 3,
    }
