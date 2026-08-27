from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from graspnet6d.analysis_inputs import (
    AnalysisInputAssemblyError,
    assemble_analysis_inputs,
    assemble_combined_analysis_inputs,
)
from graspnet6d.contracts import Candidate6D, candidate_pool_fingerprint
from graspnet6d.evaluator import CandidateEvaluation
from graspnet6d.experiment_analysis import FORMAL_SCOPE, load_analysis_input_manifest
from graspnet6d.features import (
    default_feature_schema,
    feature_schema_sha256,
    load_feature_schema,
)
from graspnet6d.formal_inputs import (
    FORMAL_FEATURE_SCHEMA,
    group_artifact_slug,
)
from graspnet6d.io import atomic_json, atomic_jsonl, canonical_sha256, sha256_file
from graspnet6d.stages import (
    GEOMETRY_EVIDENCE_SCHEMA,
    GEOMETRY_SCHEMA,
    LABEL_BUNDLE_SCHEMA,
    PARITY_EVIDENCE_SCHEMA,
    TSDF_SCHEMA,
    VGN_BUNDLE_SCHEMA,
    _convert_frozen_candidates,
    load_evaluator_geometry_contract,
)
from graspnet6d.vgn import vgn_candidate_from_record


CONDITION = "hifics_zero_shot_mask"


def _text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _candidate(group_id: str, rank: int) -> Candidate6D:
    offset = 0.01 * rank
    return Candidate6D(
        candidate_id=f"{group_id}:candidate-{rank:03d}",
        group_id=group_id,
        native_rank=rank,
        native_score=1.0 - 0.1 * rank,
        translation_local_m=(offset, 0.0, 0.15),
        rotation_local=np.eye(3),
        translation_camera_m=(offset, 0.0, 0.8),
        rotation_camera=np.eye(3),
        translation_table_m=(offset, 0.0, 0.2),
        rotation_table=np.eye(3),
        width_m=0.06,
        height_m=0.02,
        depth_m=0.04,
        voxel_index=(20 + rank, 20, 20),
        conversion_provenance={
            "mapping_name": "formal-test-contract",
            "mapping_status": "validated_real_data_artifact",
            "source": "content-addressed-test-input",
        },
    )


def _geometry_evidence(root: Path) -> dict[str, Any]:
    evidence_root = root / "geometry_evidence"
    artifact = evidence_root / "artifact.json"
    contract = evidence_root / "contract.json"
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
    binding_keys = (
        "data_manifest_sha256",
        "group_manifest_sha256",
        "tsdf_config_sha256",
        "extraction_config_sha256",
        "upstream_versions_sha256",
    )
    binding_paths = {
        key: _text(evidence_root / f"{key}.txt", f"{key}\n") for key in binding_keys
    }
    samples = evidence_root / "samples.csv"
    samples.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "group_id": "geometry-validation-group",
                "reprojection_error_px": 0.0,
                "camera_table_round_trip_error_m": 0.0,
                "local_table_round_trip_error_m": 0.0,
                "rotation_orthogonality_error": 0.0,
                "rotation_determinant": 1.0,
                "candidate_center_inside_workspace": True,
                "projected_center_reasonable": True,
                "approach_visual_audit_passed": True,
            }
        ]
    ).to_csv(samples, index=False)
    figures = [
        _text(evidence_root / f"figure-{index:02d}.png", f"figure-{index}\n")
        for index in range(20)
    ]
    atomic_json(
        artifact,
        {
            "schema_version": GEOMETRY_EVIDENCE_SCHEMA,
            "scope": "formal_real_data",
            "fixture_only": False,
            "status": "PASSED",
            "validated_group_ids": ["geometry-validation-group"],
            "bindings": {key: sha256_file(path) for key, path in binding_paths.items()},
            "binding_paths": {key: str(path) for key, path in binding_paths.items()},
            "R_vgn_gripper_to_graspnet_gripper": np.eye(3).tolist(),
            "height_m": 0.02,
            "depth_m": 0.04,
            "sample_metrics_path": str(samples),
            "sample_metrics_sha256": sha256_file(samples),
            "audit_figure_paths": [str(path) for path in figures],
            "audit_figure_sha256": {str(path): sha256_file(path) for path in figures},
        },
    )
    return load_evaluator_geometry_contract(contract, evidence_policy="formal")[1]


def _parity_evidence(root: Path) -> dict[str, Any]:
    evidence_root = root / "parity_evidence"
    gate = _text(evidence_root / "gate.json", "parity-gate\n")
    artifact = _text(evidence_root / "artifact.json", "parity-evidence\n")
    binding = _text(evidence_root / "binding.txt", "binding\n")
    comparison = _text(
        evidence_root / "comparison.csv", "candidate_id,agreement\nprobe,1\n"
    )
    report = _text(evidence_root / "report.md", "# Parity passed\n")
    return {
        "gate_path": str(gate),
        "gate_sha256": sha256_file(gate),
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "evidence_policy": "formal",
        "verified_evidence": {
            "evidence_schema": PARITY_EVIDENCE_SCHEMA,
            "bindings": {"parity_source_sha256": sha256_file(binding)},
            "binding_paths": {"parity_source_sha256": str(binding)},
            "comparison_csv_path": str(comparison),
            "comparison_csv_sha256": sha256_file(comparison),
            "report_path": str(report),
            "report_sha256": sha256_file(report),
        },
    }


def _target(group_id: str, partition: str, scene_id: str) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "split": partition,
        "scene_id": scene_id,
        "camera": "kinect",
        "frame_id": 0,
        "target_object_id": 1,
        "target_instance_label": 2,
        "depth_path": f"/official/{group_id}/depth.png",
        "instance_label_path": f"/official/{group_id}/label.png",
        "meta_path": f"/official/{group_id}/meta.mat",
        "intrinsics_path": f"/official/{group_id}/camK.npy",
        "camera_pose_path": f"/official/{group_id}/camera_poses.npy",
        "table_transform_path": f"/official/{group_id}/table.npy",
    }


def _language(group_id: str) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "query": "the uniquely resolved object",
        "resolver_result": [1],
        "is_unique": True,
        "provenance": "derived",
    }


def _write_candidate_bundle(
    root: Path,
    *,
    group_id: str,
    candidates: list[Candidate6D],
    geometry_evidence: dict[str, Any],
    condition: str = CONDITION,
) -> Path:
    slug = group_artifact_slug(group_id)
    tsdf = _text(root / "target_tsdf" / condition / f"{slug}.npz", "tsdf\n")
    mask_input = "a" * 64
    mask_commit = "b" * 64
    atomic_json(
        tsdf.with_suffix(".json"),
        {
            "schema_version": TSDF_SCHEMA,
            "group_id": group_id,
            "grounding_condition": condition,
            "grounding_mask_input_fingerprint": mask_input,
            "grounding_mask_commit_sha256": mask_commit,
            "output_sha256": sha256_file(tsdf),
        },
    )
    checkpoint = _text(root / "sources" / f"{slug}-vgn.pth", "checkpoint\n")
    raw_records = [
        {
            "candidate_id": candidate.candidate_id,
            "group_id": candidate.group_id,
            "native_rank": candidate.native_rank,
            "native_score": candidate.native_score,
            "translation_local_m": list(candidate.translation_local_m),
            "rotation_local_vgn": [list(row) for row in candidate.rotation_local],
            "translation_camera_m": list(candidate.translation_camera_m),
            "rotation_camera_vgn": [list(row) for row in candidate.rotation_camera],
            "translation_table_m": list(candidate.translation_table_m),
            "rotation_table_vgn": [list(row) for row in candidate.rotation_table],
            "width_m": candidate.width_m,
            "voxel_index": list(candidate.voxel_index),
            "gripper_frame": "vgn_(+Z_approach,+Y_closing)",
        }
        for candidate in candidates
    ]
    raw_candidates = tuple(
        vgn_candidate_from_record(record, expected_group_id=group_id)
        for record in raw_records
    )
    geometry, checked_evidence = load_evaluator_geometry_contract(
        geometry_evidence["contract_path"],
        evidence_policy=geometry_evidence["evidence_policy"],
    )
    assert canonical_sha256(checked_evidence) == canonical_sha256(geometry_evidence)
    converted_candidates, rows = _convert_frozen_candidates(
        raw_candidates, geometry, checked_evidence
    )
    a7_membership = {
        str(top_k): {
            "candidate_count": len(candidates),
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "candidate_ids_sha256": canonical_sha256(
                [candidate.candidate_id for candidate in candidates]
            ),
        }
        for top_k in (20, 50, 100)
    }
    payload: dict[str, Any] = {
        "schema_version": VGN_BUNDLE_SCHEMA,
        "group_id": group_id,
        "input_fingerprint": canonical_sha256({"group_id": group_id}),
        "candidate_count": len(candidates),
        "raw_vgn_candidates": raw_records,
        "raw_vgn_pool_fingerprint": canonical_sha256(raw_records),
        "pre_nms_snapshot_status": "complete_same_inference",
        "pre_nms_candidate_count": len(raw_records),
        "pre_nms_vgn_candidates": raw_records,
        "pre_nms_pool_fingerprint": canonical_sha256(raw_records),
        "a7_top_k_membership": a7_membership,
        "a7_top_k_membership_fingerprint": canonical_sha256(a7_membership),
        "candidate_records": [
            candidate.to_dict() for candidate in converted_candidates
        ],
        "candidate_pool_fingerprint": candidate_pool_fingerprint(converted_candidates),
        "graspnet_rows": rows,
        "geometry_contract": geometry_evidence,
        "grounding_condition": condition,
        "grounding_mask_input_fingerprint": mask_input,
        "grounding_mask_commit_sha256": mask_commit,
        "tsdf_path": str(tsdf),
        "tsdf_sha256": sha256_file(tsdf),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "generation_status": "completed_vgn_inference",
        "inference_calls_for_group": 1,
        "extraction_config": {
            "pre_nms_max_candidates": 100,
            "frozen_top_k": 50,
            "translation_threshold_m": 0.015,
            "rotation_threshold_deg": 15.0,
            "width_threshold_m": 0.01,
        },
    }
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    path = root / "vgn_candidates" / condition / f"{slug}.json"
    atomic_json(path, payload)
    return path


def _write_feature_commit(
    root: Path,
    *,
    target_manifest: Path,
    language_manifest: Path,
    target: dict[str, Any],
    candidates: list[Candidate6D],
    candidate_bundle: Path,
    condition: str = CONDITION,
) -> Path:
    group_id = target["group_id"]
    slug = group_artifact_slug(group_id)
    directory = root / "candidate_features" / condition
    path = directory / f"{slug}.parquet"
    identifiers = ("group_id", "candidate_id", "scene_id", "split", "condition")
    feature_names = tuple(spec.name for spec in default_feature_schema())
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        record: dict[str, Any] = {
            "group_id": group_id,
            "candidate_id": candidate.candidate_id,
            "scene_id": target["scene_id"],
            "split": target["split"],
            "condition": condition,
        }
        record.update({name: 0.25 for name in feature_names})
        record["native_rank"] = candidate.native_rank
        record["native_score"] = candidate.native_score
        records.append(record)
    frame = pd.DataFrame(records, columns=(*identifiers, *feature_names))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    candidate_payload = json.loads(candidate_bundle.read_text(encoding="utf-8"))
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    contract = {
        "schema": FORMAL_FEATURE_SCHEMA,
        "group_id": group_id,
        "scene_id": target["scene_id"],
        "split": target["split"],
        "condition": condition,
        "mask_input_fingerprint": "c" * 64,
        "mask_commit_fingerprint": "d" * 64,
        "candidate_pool_fingerprint": candidate_payload["candidate_pool_fingerprint"],
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "feature_schema_sha256": feature_schema_sha256(default_feature_schema()),
        "scene_provenance": {"source": "test-bound-source"},
        "source_hashes": {
            str(target_manifest): sha256_file(target_manifest),
            str(language_manifest): sha256_file(language_manifest),
            str(candidate_bundle): sha256_file(candidate_bundle),
        },
        "ground_truth_inputs_consumed": [],
        "official_evaluator_outputs_consumed": [],
    }
    payload: dict[str, Any] = {
        "schema_version": FORMAL_FEATURE_SCHEMA,
        "group_id": group_id,
        "scene_id": target["scene_id"],
        "split": target["split"],
        "condition": condition,
        "input_fingerprint": canonical_sha256(contract),
        "input_contract": contract,
        "feature_file": path.name,
        "feature_sha256": sha256_file(path),
        "feature_schema_sha256": feature_schema_sha256(default_feature_schema()),
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids,
        "candidate_pool_fingerprint": candidate_payload["candidate_pool_fingerprint"],
        "ground_truth_inputs_consumed": [],
        "official_evaluator_outputs_consumed": [],
    }
    payload["commit_fingerprint"] = canonical_sha256(payload)
    sidecar = directory / f"{slug}.json"
    atomic_json(sidecar, payload)
    return sidecar


def _write_label_bundle(
    root: Path,
    *,
    target: dict[str, Any],
    candidates: list[Candidate6D],
    candidate_bundle: Path,
    parity_evidence: dict[str, Any],
    condition: str = CONDITION,
) -> Path:
    group_id = target["group_id"]
    evaluator_source = _text(
        root / "sources" / f"{group_artifact_slug(group_id)}-evaluator.txt",
        "official-evaluator-source\n",
    )
    labels = []
    for index, candidate in enumerate(candidates):
        evaluation = CandidateEvaluation(
            candidate_index=index,
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
        labels.append({"candidate_id": candidate.candidate_id, **evaluation})
    candidate_payload = json.loads(candidate_bundle.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": LABEL_BUNDLE_SCHEMA,
        "group_id": group_id,
        "target_object_id": 1,
        "candidate_bundle_path": str(candidate_bundle),
        "candidate_bundle_sha256": sha256_file(candidate_bundle),
        "candidate_pool_fingerprint": candidate_payload["candidate_pool_fingerprint"],
        "grounding_condition": condition,
        "candidate_count": len(candidates),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "labels": labels,
        "parity_gate": parity_evidence,
    }
    if candidates:
        payload.update(
            {
                "official_source_hashes": {
                    str(evaluator_source): sha256_file(evaluator_source)
                },
                "dexnet_source_kind": "official_textured_obj_and_sdf_no_pickle",
                "evaluator_operation": "per_candidate_low_level_no_eval_grasp_no_nms_no_topk",
                "label_generation_status": "completed_official_low_level_evaluation",
                "evaluator_calls_for_group": 1,
            }
        )
    else:
        payload.update(
            {
                "official_source_hashes": {},
                "dexnet_source_kind": "not_loaded_empty_frozen_pool",
                "evaluator_operation": "not_called_empty_frozen_pool",
                "label_generation_status": "skipped_empty_pool",
                "evaluator_calls_for_group": 0,
                "empty_pool_reason": "vgn_no_candidates",
            }
        )
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    path = (
        root / "official_labels" / condition / f"{group_artifact_slug(group_id)}.json"
    )
    atomic_json(path, payload)
    return path


def _world(
    tmp_path: Path,
    *,
    leak_scene_across_splits: bool = False,
    conditions: tuple[str, ...] = (CONDITION,),
) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    root = tmp_path / "run"
    targets = [
        _target("train-group", "train", "scene-train"),
        _target(
            "validation-group",
            "validation",
            "scene-train" if leak_scene_across_splits else "scene-validation",
        ),
        _target("test-group", "test", "scene-test"),
        _target("test-empty-group", "test", "scene-test-empty"),
    ]
    target_manifest = tmp_path / "target_groups.jsonl"
    language_manifest = tmp_path / "language_queries.jsonl"
    atomic_jsonl(target_manifest, targets)
    atomic_jsonl(
        language_manifest, [_language(target["group_id"]) for target in targets]
    )
    geometry = _geometry_evidence(tmp_path)
    parity = _parity_evidence(tmp_path)
    for condition in conditions:
        for target in targets:
            candidates = (
                []
                if target["group_id"] == "test-empty-group"
                else [_candidate(target["group_id"], 1)]
            )
            candidate_bundle = _write_candidate_bundle(
                root,
                group_id=target["group_id"],
                candidates=candidates,
                geometry_evidence=geometry,
                condition=condition,
            )
            _write_feature_commit(
                root,
                target_manifest=target_manifest,
                language_manifest=language_manifest,
                target=target,
                candidates=candidates,
                candidate_bundle=candidate_bundle,
                condition=condition,
            )
            _write_label_bundle(
                root,
                target=target,
                candidates=candidates,
                candidate_bundle=candidate_bundle,
                parity_evidence=parity,
                condition=condition,
            )
    return root, target_manifest, language_manifest, targets


def _assemble(
    tmp_path: Path,
    root: Path,
    target_manifest: Path,
    language_manifest: Path,
):
    return assemble_analysis_inputs(
        target_manifest,
        language_manifest,
        root,
        tmp_path / "outputs",
        condition=CONDITION,
        run_id="real-data-run-001",
    )


def test_assemble_analysis_inputs_preserves_frozen_pool_and_empty_universe(
    tmp_path: Path,
) -> None:
    root, target_manifest, language_manifest, _ = _world(tmp_path)
    result = _assemble(tmp_path, root, target_manifest, language_manifest)

    assert result.condition == CONDITION
    assert result.group_count == 4
    assert result.candidate_count == 3
    assert result.empty_group_count == 1
    assert not result.resumed
    assert result.output_dir.parent.name == CONDITION
    assert set(result.partition_rows) == {"train", "validation", "test"}
    loaded = load_analysis_input_manifest(result.manifest_path)
    assert loaded.status == "COMPLETE"
    assert loaded.scope == FORMAL_SCOPE
    assert not loaded.fixture_only

    test_rows = pd.read_parquet(result.partition_rows["test"])
    test_universe = pd.read_parquet(result.partition_group_universes["test"])
    assert test_rows["group_id"].tolist() == ["test-group"]
    assert set(test_universe["group_id"]) == {"test-group", "test-empty-group"}
    assert (
        test_rows.loc[0, "geometry_sha256"]
        == _candidate("test-group", 1).geometry_sha256
    )
    assert test_rows.loc[0, "relevance"] == 5
    assert bool(test_rows.loc[0, "target_match"])
    assert test_rows.loc[0, "grounding_condition"] == CONDITION
    feature_names = [
        spec.name for spec in load_feature_schema(loaded.feature_schema.path)
    ]
    assert feature_names == [spec.name for spec in default_feature_schema()]
    assert not {
        "collision",
        "pose_valid",
        "friction_required",
        "relevance",
        "target_object_id",
    }.intersection(feature_names)

    resumed = _assemble(tmp_path, root, target_manifest, language_manifest)
    assert resumed.resumed
    assert resumed.manifest_sha256 == result.manifest_sha256
    assert resumed.partition_rows == result.partition_rows


def test_assembler_refuses_missing_or_stale_group_commits(tmp_path: Path) -> None:
    root, target_manifest, language_manifest, targets = _world(tmp_path)
    missing_label = (
        root
        / "official_labels"
        / CONDITION
        / f"{group_artifact_slug(targets[-1]['group_id'])}.json"
    )
    missing_label.unlink()
    with pytest.raises(AnalysisInputAssemblyError, match="universe differs"):
        _assemble(tmp_path, root, target_manifest, language_manifest)
    assert not (tmp_path / "outputs" / "analysis_inputs" / CONDITION).exists()

    # Restore the world, then alter a committed Parquet after its sidecar hash.
    root, target_manifest, language_manifest, targets = _world(tmp_path / "stale")
    stale_table = (
        root
        / "candidate_features"
        / CONDITION
        / f"{group_artifact_slug(targets[0]['group_id'])}.parquet"
    )
    stale_table.write_bytes(b"not-the-committed-parquet")
    with pytest.raises(
        AnalysisInputAssemblyError, match="invalid formal feature commit"
    ):
        _assemble(tmp_path / "stale", root, target_manifest, language_manifest)


def test_assembler_refuses_cross_condition_labels_and_nonformal_evidence(
    tmp_path: Path,
) -> None:
    root, target_manifest, language_manifest, targets = _world(tmp_path)
    label_path = (
        root
        / "official_labels"
        / CONDITION
        / f"{group_artifact_slug(targets[0]['group_id'])}.json"
    )
    label = json.loads(label_path.read_text(encoding="utf-8"))
    label["grounding_condition"] = "oracle_gt_mask"
    label.pop("bundle_fingerprint")
    label["bundle_fingerprint"] = canonical_sha256(label)
    atomic_json(label_path, label)
    with pytest.raises(AnalysisInputAssemblyError, match="another condition"):
        _assemble(tmp_path, root, target_manifest, language_manifest)

    root, target_manifest, language_manifest, targets = _world(tmp_path / "fixture")
    candidate_path = (
        root
        / "vgn_candidates"
        / CONDITION
        / f"{group_artifact_slug(targets[0]['group_id'])}.json"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["geometry_contract"]["evidence_policy"] = "test"
    candidate.pop("bundle_fingerprint")
    candidate["bundle_fingerprint"] = canonical_sha256(candidate)
    atomic_json(candidate_path, candidate)
    with pytest.raises(
        AnalysisInputAssemblyError,
        match="not formal_real_data|test evidence policy requires fixture_only",
    ):
        _assemble(tmp_path / "fixture", root, target_manifest, language_manifest)


def test_assembler_refuses_scene_split_leakage(tmp_path: Path) -> None:
    root, target_manifest, language_manifest, _ = _world(
        tmp_path, leak_scene_across_splits=True
    )
    with pytest.raises(AnalysisInputAssemblyError, match="scene split leakage"):
        _assemble(tmp_path, root, target_manifest, language_manifest)


def test_assembler_refuses_changed_label_candidate_order(tmp_path: Path) -> None:
    root, target_manifest, language_manifest, targets = _world(tmp_path)
    label_path = (
        root
        / "official_labels"
        / CONDITION
        / f"{group_artifact_slug(targets[0]['group_id'])}.json"
    )
    label = json.loads(label_path.read_text(encoding="utf-8"))
    label["candidate_ids"] = ["wrong-candidate-id"]
    label.pop("bundle_fingerprint")
    label["bundle_fingerprint"] = canonical_sha256(label)
    atomic_json(label_path, label)
    with pytest.raises(AnalysisInputAssemblyError, match="order/membership"):
        _assemble(tmp_path, root, target_manifest, language_manifest)


def test_combined_assembler_namespaces_all_three_grounding_arms(
    tmp_path: Path,
) -> None:
    conditions = (
        "oracle_gt_mask",
        "hifics_zero_shot_mask",
        "hifics_adapted_mask",
    )
    root, target_manifest, language_manifest, _ = _world(
        tmp_path, conditions=conditions
    )
    result = assemble_combined_analysis_inputs(
        target_manifest,
        language_manifest,
        root,
        tmp_path / "outputs",
        run_id="real-data-run-001",
        conditions=tuple(reversed(conditions)),
    )

    assert result.condition == "combined_grounding_conditions"
    assert result.conditions == conditions
    assert result.group_count == 12
    assert result.candidate_count == 9
    assert result.empty_group_count == 3
    test_rows = pd.read_parquet(result.partition_rows["test"])
    test_universe = pd.read_parquet(result.partition_group_universes["test"])
    assert set(test_rows["grounding_condition"]) == set(conditions)
    assert set(test_universe["grounding_condition"]) == set(conditions)
    assert test_rows["candidate_id"].is_unique
    assert test_universe["group_id"].is_unique
    assert all(
        str(row["candidate_id"]).startswith(f"{row['grounding_condition']}::")
        for _, row in test_rows.iterrows()
    )
    assert all(
        str(row["group_id"]).startswith(f"{row['grounding_condition']}::")
        for _, row in test_universe.iterrows()
    )

    resumed = assemble_combined_analysis_inputs(
        target_manifest,
        language_manifest,
        root,
        tmp_path / "outputs",
        run_id="real-data-run-001",
    )
    assert resumed.resumed
    assert resumed.manifest_sha256 == result.manifest_sha256
