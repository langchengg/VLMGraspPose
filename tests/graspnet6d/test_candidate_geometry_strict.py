"""Tamper tests for the frozen raw-VGN to evaluator conversion boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from graspnet6d.contracts import Candidate6D, candidate_pool_fingerprint
from graspnet6d.formal_inputs import _load_candidate_bundle as load_formal_bundle
from graspnet6d.io import (
    atomic_json,
    atomic_npz,
    canonical_sha256,
    sha256_file,
)
from graspnet6d.stages import (
    GEOMETRY_SCHEMA,
    TSDF_SCHEMA,
    VGN_BUNDLE_SCHEMA,
    StageInputError,
    _a7_pre_nms_payload,
    _convert_frozen_candidates,
    load_evaluator_geometry_contract,
    load_frozen_candidate_bundle,
)
from graspnet6d.vgn import ExtractionConfig, VGNCandidate


GROUP_ID = "scene_0000_kinect_0000_obj_000"
CONDITION = "oracle_gt_mask"


def _raw_candidate(rank: int) -> VGNCandidate:
    translation = np.asarray([0.03 * rank, 0.01, 0.40], dtype=np.float64)
    return VGNCandidate(
        candidate_id=f"candidate-{rank}",
        group_id=GROUP_ID,
        native_rank=rank,
        native_score=1.0 - 0.1 * rank,
        translation_local_m=translation.copy(),
        rotation_local_vgn=np.eye(3),
        width_m=0.04 + 0.005 * rank,
        voxel_index=(rank, rank + 1, rank + 2),
        translation_camera_m=translation.copy(),
        rotation_camera_vgn=np.eye(3),
        translation_table_m=translation.copy(),
        rotation_table_vgn=np.eye(3),
    )


def _write_valid_bundle(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "geometry_validation.json"
    atomic_json(artifact, {"status": "PASSED", "fixture_only": True})
    contract_path = tmp_path / "geometry_contract.json"
    atomic_json(
        contract_path,
        {
            "schema_version": GEOMETRY_SCHEMA,
            "validated": True,
            "validation_artifact": artifact.name,
            "R_vgn_gripper_to_graspnet_gripper": np.eye(3).tolist(),
            "height_m": 0.02,
            "depth_m": 0.04,
        },
    )
    geometry, evidence = load_evaluator_geometry_contract(
        contract_path, evidence_policy="test"
    )
    raw_candidates = (_raw_candidate(1), _raw_candidate(2))
    converted, rows = _convert_frozen_candidates(raw_candidates, geometry, evidence)

    mask_fingerprint = "a" * 64
    mask_commit = "b" * 64
    tsdf_path = tmp_path / "target_tsdf" / CONDITION / "group.npz"
    atomic_npz(
        tsdf_path,
        tsdf=np.zeros((1, 40, 40, 40), dtype=np.float32),
        T_local_to_camera=np.eye(4),
        T_local_to_table=np.eye(4),
        full_scene_depth_integrated=np.asarray(True),
        grounding_condition=np.asarray(CONDITION),
        grounding_mask_input_fingerprint=np.asarray(mask_fingerprint),
        grounding_mask_commit_sha256=np.asarray(mask_commit),
    )
    atomic_json(
        tsdf_path.with_suffix(".json"),
        {
            "schema_version": TSDF_SCHEMA,
            "group_id": GROUP_ID,
            "grounding_condition": CONDITION,
            "grounding_mask_input_fingerprint": mask_fingerprint,
            "grounding_mask_commit_sha256": mask_commit,
            "output_sha256": sha256_file(tsdf_path),
        },
    )
    checkpoint = tmp_path / "vgn.pth"
    checkpoint.write_bytes(b"strict-conversion-test-checkpoint")
    config = ExtractionConfig()
    raw_records = [candidate.to_record() for candidate in raw_candidates]
    payload: dict[str, Any] = {
        "schema_version": VGN_BUNDLE_SCHEMA,
        "group_id": GROUP_ID,
        "input_fingerprint": "c" * 64,
        "tsdf_path": str(tsdf_path),
        "tsdf_sha256": sha256_file(tsdf_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": "cpu",
        "grounding_condition": CONDITION,
        "grounding_mask_input_fingerprint": mask_fingerprint,
        "grounding_mask_commit_sha256": mask_commit,
        "extraction_config": {
            "pre_nms_max_candidates": config.pre_nms_max_candidates,
            "frozen_top_k": config.frozen_top_k,
            "translation_threshold_m": config.translation_threshold_m,
            "rotation_threshold_deg": config.rotation_threshold_deg,
            "width_threshold_m": config.width_threshold_m,
        },
        "generation_status": "completed_vgn_inference",
        "inference_calls_for_group": 1,
        "candidate_count": len(raw_candidates),
        "raw_vgn_candidates": raw_records,
        "raw_vgn_pool_fingerprint": canonical_sha256(raw_records),
        "candidate_records": [candidate.to_dict() for candidate in converted],
        "candidate_pool_fingerprint": candidate_pool_fingerprint(converted),
        "graspnet_rows": rows,
        "geometry_contract": evidence,
        **_a7_pre_nms_payload(
            (),
            raw_candidates,
            config=config,
            status="test_injected_post_nms_only",
        ),
    }
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    bundle_path = tmp_path / "candidates.json"
    atomic_json(bundle_path, payload)
    return bundle_path, contract_path


def _recommit_converted_tamper(path: Path, kind: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload["candidate_records"][1]
    row = payload["graspnet_rows"][1]
    if kind == "rotation":
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ).tolist()
        for field in ("rotation_local", "rotation_camera", "rotation_table"):
            record[field] = rotation
        row[4:13] = np.asarray(rotation).reshape(-1).tolist()
    elif kind == "translation":
        translation = [0.5, -0.2, 0.7]
        for field in (
            "translation_local_m",
            "translation_camera_m",
            "translation_table_m",
        ):
            record[field] = translation
        row[13:16] = translation
    elif kind == "width":
        record["width_m"] = 0.073
        row[1] = 0.073
    elif kind == "height_depth":
        record["height_m"] = 0.031
        record["depth_m"] = 0.061
        row[2] = 0.031
        row[3] = 0.061
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(kind)
    decoded = [Candidate6D.from_dict(value) for value in payload["candidate_records"]]
    payload["candidate_pool_fingerprint"] = candidate_pool_fingerprint(decoded)
    payload.pop("bundle_fingerprint")
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    atomic_json(path, payload)


def test_strict_loaders_accept_source_derived_geometry(tmp_path: Path) -> None:
    path, _ = _write_valid_bundle(tmp_path)
    stage_payload = load_frozen_candidate_bundle(path, group_id=GROUP_ID)
    formal_candidates, _, _, _ = load_formal_bundle(path, GROUP_ID)
    assert stage_payload["candidate_count"] == 2
    assert len(formal_candidates) == 2


@pytest.mark.parametrize("kind", ["rotation", "translation", "width", "height_depth"])
def test_strict_loaders_reject_consistently_rehashed_converted_geometry(
    tmp_path: Path, kind: str
) -> None:
    path, _ = _write_valid_bundle(tmp_path)
    _recommit_converted_tamper(path, kind)

    with pytest.raises(StageInputError, match="raw VGN|bound geometry"):
        load_frozen_candidate_bundle(path, group_id=GROUP_ID)
    with pytest.raises(StageInputError, match="raw VGN|bound geometry"):
        load_formal_bundle(path, GROUP_ID)


def test_strict_loaders_rehash_geometry_contract_source(tmp_path: Path) -> None:
    path, contract_path = _write_valid_bundle(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["height_m"] = 0.021
    atomic_json(contract_path, contract)

    with pytest.raises(StageInputError, match="contract/evidence"):
        load_frozen_candidate_bundle(path, group_id=GROUP_ID)
    with pytest.raises(StageInputError, match="contract/evidence"):
        load_formal_bundle(path, GROUP_ID)


def test_formal_policy_rejects_explicit_fixture_incompleteness(tmp_path: Path) -> None:
    path, _ = _write_valid_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["geometry_contract"] = {
        "fixture_only": True,
        "evidence_policy": "formal",
    }
    payload.pop("bundle_fingerprint")
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    atomic_json(path, payload)

    with pytest.raises(StageInputError, match="fixture|incomplete"):
        load_formal_bundle(path, GROUP_ID)
