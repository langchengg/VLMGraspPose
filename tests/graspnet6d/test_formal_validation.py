from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from graspnet6d.contracts import Candidate6D, candidate_pool_fingerprint
from graspnet6d.formal_validation import (
    AI_ASSISTED_GROUP_CHECKS,
    AI_ASSISTED_REVIEW_DISCLAIMER,
    AI_ASSISTED_REVIEW_KIND,
    RAW_VGN_VALIDATION_SCHEMA,
    VISUAL_REVIEW_SCHEMA,
    FormalValidationError,
    _candidate_metrics,
    run_evaluator_parity_validation,
    run_geometry_validation,
)
from graspnet6d.geometry import CameraIntrinsics
from graspnet6d.io import atomic_json, atomic_jsonl, atomic_npz, canonical_sha256, sha256_file
from graspnet6d.stages import (
    TSDF_SCHEMA,
    VGN_BUNDLE_SCHEMA,
    load_evaluator_geometry_contract,
    load_evaluator_parity_gate,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _geometry_fixture(tmp_path: Path) -> dict[str, Any]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[..., 1] = 100
    depth = np.full((20, 20), 1000, dtype=np.uint16)
    label = np.zeros((20, 20), dtype=np.uint16)
    label[7:14, 7:14] = 1
    rgb_path = inputs / "rgb.png"
    depth_path = inputs / "depth.png"
    label_path = inputs / "label.png"
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(depth).save(depth_path)
    Image.fromarray(label).save(label_path)
    intrinsic = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    intrinsics_path = inputs / "camK.npy"
    np.save(intrinsics_path, intrinsic, allow_pickle=False)
    meta_path = inputs / "meta.mat"
    savemat(meta_path, {"factor_depth": np.array([[1000.0]])})
    group_id = "fixture-group"
    target = {
        "group_id": group_id,
        "target_object_id": 0,
        "target_instance_label": 1,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "instance_label_path": str(label_path),
        "meta_path": str(meta_path),
        "intrinsics_path": str(intrinsics_path),
        "provenance": "explicit_test_fixture",
    }
    manifest = tmp_path / "target_groups.jsonl"
    atomic_jsonl(manifest, [target])
    local_to_camera = np.eye(4)
    local_to_camera[:3, 3] = [-0.15, -0.15, 0.85]
    tsdf = tmp_path / "target_tsdf.npz"
    atomic_npz(
        tsdf,
        tsdf=np.zeros((1, 40, 40, 40), dtype=np.float32),
        T_local_to_camera=local_to_camera,
        T_local_to_table=local_to_camera,
        physical_size=np.float64(0.30),
        full_scene_depth_integrated=np.bool_(True),
    )
    atomic_json(
        tsdf.with_suffix(".json"),
        {
            "schema_version": TSDF_SCHEMA,
            "group_id": group_id,
            "output_sha256": sha256_file(tsdf),
        },
    )
    candidate = {
        "candidate_id": "candidate-1",
        "group_id": group_id,
        "native_rank": 1,
        "native_score": 0.9,
        "translation_local_m": [0.15, 0.15, 0.15],
        "rotation_local_vgn": np.eye(3).tolist(),
        "translation_camera_m": [0.0, 0.0, 1.0],
        "rotation_camera_vgn": np.eye(3).tolist(),
        "translation_table_m": [0.0, 0.0, 1.0],
        "rotation_table_vgn": np.eye(3).tolist(),
        "width_m": 0.05,
        "voxel_index": [20, 20, 20],
    }
    candidates = tmp_path / "raw_candidates.json"
    payload = {
        "schema_version": RAW_VGN_VALIDATION_SCHEMA,
        "group_id": group_id,
        "candidate_count": 1,
        "inference_calls_for_group": 1,
        "tsdf_sha256": sha256_file(tsdf),
        "fixture_only": True,
        "raw_vgn_candidates": [candidate],
    }
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    atomic_json(candidates, payload)
    binding_paths = {
        "data_manifest_sha256": _write(tmp_path / "data_manifest.json", "{}\n"),
        "group_manifest_sha256": manifest,
        "tsdf_config_sha256": _write(tmp_path / "tsdf.yaml", "resolution: 40\n"),
        "extraction_config_sha256": _write(tmp_path / "extract.yaml", "top_k: 50\n"),
        "upstream_versions_sha256": _write(tmp_path / "upstream.lock", "vgn: abc\n"),
    }
    return {
        "group_id": group_id,
        "manifest": manifest,
        "tsdf": tsdf,
        "candidates": candidates,
        "bindings": binding_paths,
    }


def _geometry_review(output: Path, group_id: str, *, bad_hash: bool = False) -> Path:
    render = json.loads((output / "geometry_render_manifest.json").read_text(encoding="utf-8"))
    hashes = dict(render["figure_sha256"])
    if bad_hash:
        hashes[next(iter(hashes))] = "0" * 64
    review = output / "ai_assisted_visual_review.json"
    atomic_json(
        review,
        {
            "schema_version": VISUAL_REVIEW_SCHEMA,
            "scope": "test_fixture",
            "fixture_only": True,
            "status": "PASSED",
            "review_kind": AI_ASSISTED_REVIEW_KIND,
            "reviewer_system": "unit-test-visual-reviewer",
            "independent_human_review": False,
            "disclaimer": AI_ASSISTED_REVIEW_DISCLAIMER,
            "reviewed_group_ids": [group_id],
            "checks_by_group": {
                group_id: {name: True for name in AI_ASSISTED_GROUP_CHECKS}
            },
            "figure_sha256": hashes,
        },
    )
    return review


def test_geometry_runner_requires_hashed_ai_assisted_review_and_emits_gate(
    tmp_path: Path,
) -> None:
    fixture = _geometry_fixture(tmp_path)
    output = tmp_path / "geometry"
    kwargs = {
        "R_vgn_gripper_to_graspnet_gripper": np.eye(3),
        "height_m": 0.02,
        "depth_m": 0.04,
        "binding_paths": fixture["bindings"],
        "scope": "test_fixture",
        "fixture_only": True,
        "minimum_figures": 1,
    }
    with pytest.raises(FormalValidationError, match="no AI-assisted visual review"):
        run_geometry_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: fixture["tsdf"]},
            {fixture["group_id"]: fixture["candidates"]},
            output,
            **kwargs,
        )
    figures = list((output / "figures").glob("*.png"))
    assert len(figures) == 1
    review = _geometry_review(output, fixture["group_id"])

    result = run_geometry_validation(
        fixture["manifest"],
        [fixture["group_id"]],
        {fixture["group_id"]: fixture["tsdf"]},
        {fixture["group_id"]: fixture["candidates"]},
        output,
        visual_review_path=review,
        **kwargs,
    )

    evidence = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    assert evidence["status"] == "PASSED"
    assert evidence["fixture_only"] is True
    assert evidence["visual_review_kind"] == AI_ASSISTED_REVIEW_KIND
    assert evidence["independent_human_review"] is False
    assert evidence["visual_review_disclaimer"] == AI_ASSISTED_REVIEW_DISCLAIMER
    assert set(evidence["bindings"]) == set(fixture["bindings"])
    assert len(evidence["audit_figure_paths"]) == 1
    frame = __import__("pandas").read_csv(result.metrics_csv_path)
    assert frame["approach_visual_audit_passed"].all()
    assert frame["depth_reprojection_median_error_px"].max() < 1.0
    assert frame["depth_reprojection_p95_error_px"].max() < 2.0
    assert frame["depth_unit_valid"].all()
    assert frame["width_m_valid"].all()
    assert frame["table_frame_not_inverted"].all()
    assert evidence["recomputed_summary"]["depth_m_min"] == pytest.approx(1.0)
    assert evidence["recomputed_summary"]["depth_m_max"] == pytest.approx(1.0)
    assert evidence["recomputed_summary"]["all_widths_valid_metres"] is True
    assert evidence["recomputed_summary"]["full_pool_candidate_count"] == 1
    assert evidence["recomputed_summary"]["projected_center_in_image_count"] == 1
    assert (
        evidence["recomputed_summary"][
            "projected_center_near_target_bbox_diagnostic_count"
        ]
        == 1
    )
    assert (
        evidence["recomputed_summary"][
            "projection_image_bounds_gate_required_count"
        ]
        == 0
    )
    assert evidence["recomputed_summary"]["table_frame_not_inverted"] is True
    contract, loader_evidence = load_evaluator_geometry_contract(
        result.contract_path, evidence_policy="test"
    )
    assert contract.validated
    assert loader_evidence["verified_evidence"]["fixture_only"] is True


def test_geometry_runner_rejects_independent_human_claim_in_ai_review(
    tmp_path: Path,
) -> None:
    fixture = _geometry_fixture(tmp_path)
    output = tmp_path / "geometry"
    kwargs = {
        "R_vgn_gripper_to_graspnet_gripper": np.eye(3),
        "height_m": 0.02,
        "depth_m": 0.04,
        "binding_paths": fixture["bindings"],
        "scope": "test_fixture",
        "fixture_only": True,
        "minimum_figures": 1,
    }
    with pytest.raises(FormalValidationError, match="no AI-assisted visual review"):
        run_geometry_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: fixture["tsdf"]},
            {fixture["group_id"]: fixture["candidates"]},
            output,
            **kwargs,
        )
    review = _geometry_review(output, fixture["group_id"])
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["independent_human_review"] = True
    atomic_json(review, payload)
    with pytest.raises(FormalValidationError, match="independent_human_review:false"):
        run_geometry_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: fixture["tsdf"]},
            {fixture["group_id"]: fixture["candidates"]},
            output,
            visual_review_path=review,
            **kwargs,
        )


def test_candidate_projection_gates_only_explicit_valid_visible_target() -> None:
    cache = {
        "T_local_to_camera": np.eye(4),
        "T_local_to_table": np.eye(4),
        "physical_size": np.float64(0.3),
    }
    intrinsics = CameraIntrinsics(
        100.0, 100.0, 10.0, 10.0, width=20, height=20
    )
    target_mask = np.zeros((20, 20), dtype=bool)
    target_mask[8:12, 8:12] = True
    candidate = {
        "candidate_id": "off-frame-distractor",
        "native_rank": 1,
        "translation_local_m": [0.1, 0.1, 0.01],
        "translation_camera_m": [0.1, 0.1, 0.01],
        "translation_table_m": [0.1, 0.1, 0.01],
        "rotation_local_vgn": np.eye(3),
        "rotation_camera_vgn": np.eye(3),
        "rotation_table_vgn": np.eye(3),
        "width_m": 0.05,
    }
    kwargs = {
        "group_id": "group",
        "cache": cache,
        "intrinsics": intrinsics,
        "target_mask": target_mask,
        "conversion": np.eye(3),
        "depth_metrics": {},
    }

    diagnostic_rows, render_rows = _candidate_metrics(
        candidates=[candidate], **kwargs
    )
    assert len(render_rows) == 1
    assert diagnostic_rows[0]["projected_center_in_image"] is False
    assert diagnostic_rows[0]["projected_center_near_target_bbox"] is False
    assert diagnostic_rows[0]["projection_image_bounds_gate_required"] is False
    assert diagnostic_rows[0]["projected_center_reasonable"] is True

    valid_visible = dict(candidate)
    valid_visible["official_evaluator_valid_visible_target"] = True
    gated_rows, _ = _candidate_metrics(candidates=[valid_visible], **kwargs)
    assert gated_rows[0]["projection_image_bounds_gate_required"] is True
    assert gated_rows[0]["projected_center_reasonable"] is False


def test_candidate_projection_rejects_malformed_evaluator_marker() -> None:
    cache = {
        "T_local_to_camera": np.eye(4),
        "T_local_to_table": np.eye(4),
        "physical_size": np.float64(0.3),
    }
    intrinsics = CameraIntrinsics(
        100.0, 100.0, 10.0, 10.0, width=20, height=20
    )
    candidate = {
        "candidate_id": "candidate",
        "native_rank": 1,
        "translation_local_m": [0.1, 0.1, 0.1],
        "translation_camera_m": [0.1, 0.1, 0.1],
        "translation_table_m": [0.1, 0.1, 0.1],
        "rotation_local_vgn": np.eye(3),
        "rotation_camera_vgn": np.eye(3),
        "rotation_table_vgn": np.eye(3),
        "width_m": 0.05,
        "official_evaluator_valid_visible_target": "true",
    }
    with pytest.raises(FormalValidationError, match="malformed.*marker"):
        _candidate_metrics(
            group_id="group",
            candidates=[candidate],
            cache=cache,
            intrinsics=intrinsics,
            target_mask=np.ones((20, 20), dtype=bool),
            conversion=np.eye(3),
            depth_metrics={},
        )


def test_geometry_runner_rejects_review_of_different_image_hash(tmp_path: Path) -> None:
    fixture = _geometry_fixture(tmp_path)
    output = tmp_path / "geometry"
    base = {
        "R_vgn_gripper_to_graspnet_gripper": np.eye(3),
        "height_m": 0.02,
        "depth_m": 0.04,
        "binding_paths": fixture["bindings"],
        "scope": "test_fixture",
        "fixture_only": True,
        "minimum_figures": 1,
    }
    with pytest.raises(FormalValidationError):
        run_geometry_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: fixture["tsdf"]},
            {fixture["group_id"]: fixture["candidates"]},
            output,
            **base,
        )
    review = _geometry_review(output, fixture["group_id"], bad_hash=True)
    with pytest.raises(FormalValidationError, match="figure hashes differ"):
        run_geometry_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: fixture["tsdf"]},
            {fixture["group_id"]: fixture["candidates"]},
            output,
            visual_review_path=review,
            **base,
        )
    assert not (output / "geometry_validation_evidence.json").exists()


def test_geometry_runner_rejects_injected_renderer_for_formal_scope(tmp_path: Path) -> None:
    with pytest.raises(FormalValidationError, match="injection.*only for test fixtures"):
        run_geometry_validation(
            tmp_path / "missing.jsonl",
            ["group"],
            {},
            {},
            tmp_path / "output",
            R_vgn_gripper_to_graspnet_gripper=np.eye(3),
            height_m=0.02,
            depth_m=0.04,
            binding_paths={},
            renderer=lambda *_args, **_kwargs: None,
        )


def test_formal_geometry_minimum_cannot_be_lowered_below_twenty(
    tmp_path: Path,
) -> None:
    with pytest.raises(FormalValidationError, match="at least 20 real groups"):
        run_geometry_validation(
            tmp_path / "missing.jsonl",
            ["group"],
            {},
            {},
            tmp_path / "output",
            R_vgn_gripper_to_graspnet_gripper=np.eye(3),
            height_m=0.02,
            depth_m=0.04,
            binding_paths={},
            minimum_figures=19,
        )


def _parity_fixture(tmp_path: Path) -> dict[str, Any]:
    group_id = "fixture-parity-group"
    manifest = tmp_path / "target_groups.jsonl"
    atomic_jsonl(
        manifest,
        [{"group_id": group_id, "target_object_id": 0, "provenance": "test_fixture"}],
    )
    candidates = [
        Candidate6D(
            candidate_id=f"candidate-{index}",
            group_id=group_id,
            native_rank=index + 1,
            native_score=0.9 - index * 0.1,
            translation_local_m=[0.1 + index * 0.01, 0.1, 0.1],
            rotation_local=np.eye(3),
            translation_camera_m=[0.0 + index * 0.01, 0.0, 1.0],
            rotation_camera=np.eye(3),
            translation_table_m=[0.0 + index * 0.01, 0.0, 1.0],
            rotation_table=np.eye(3),
            width_m=0.05,
            height_m=0.02,
            depth_m=0.04,
            voxel_index=[index + 1, 2, 3],
            conversion_provenance={
                "mapping_name": "fixture",
                "mapping_status": "fixture",
                "source": "explicit_test_fixture",
            },
        )
        for index in range(2)
    ]
    rows = []
    for candidate in candidates:
        row = np.empty(17, dtype=np.float64)
        row[0] = candidate.native_score
        row[1:4] = [candidate.width_m, candidate.height_m, candidate.depth_m]
        row[4:13] = np.asarray(candidate.rotation_camera).reshape(-1)
        row[13:16] = np.asarray(candidate.translation_camera_m)
        row[16] = -1
        rows.append(row.tolist())
    bundle = tmp_path / "candidate_bundle.json"
    payload = {
        "schema_version": VGN_BUNDLE_SCHEMA,
        "group_id": group_id,
        "candidate_count": len(candidates),
        "inference_calls_for_group": 1,
        "raw_vgn_candidates": [
            {"candidate_id": candidate.candidate_id} for candidate in candidates
        ],
        "candidate_records": [candidate.to_dict() for candidate in candidates],
        "candidate_pool_fingerprint": candidate_pool_fingerprint(candidates),
        "graspnet_rows": rows,
        "fixture_only": True,
        "grounding_condition": "oracle_gt_mask",
    }
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    atomic_json(bundle, payload)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    evaluator_source = _write(tmp_path / "official_sources.txt", "fixture official inputs\n")

    def loader(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
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

    def equal_evaluator(rows: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "candidate_index": index,
                "associated_object_id": 0,
                "collision": False,
                "friction_score": 0.4,
                "valid_geometry": True,
            }
            for index in range(len(rows))
        ]

    return {
        "group_id": group_id,
        "manifest": manifest,
        "bundle": bundle,
        "dataset_root": dataset_root,
        "loader": loader,
        "evaluator": equal_evaluator,
    }


def test_parity_runner_compares_every_candidate_and_emits_strict_gate(tmp_path: Path) -> None:
    fixture = _parity_fixture(tmp_path)
    result = run_evaluator_parity_validation(
        fixture["manifest"],
        [fixture["group_id"]],
        {fixture["group_id"]: fixture["bundle"]},
        tmp_path / "parity",
        dataset_root=fixture["dataset_root"],
        scope="test_fixture",
        fixture_only=True,
        input_loader=fixture["loader"],
        official_reference=fixture["evaluator"],
        adapter_evaluator=fixture["evaluator"],
    )

    comparison = __import__("pandas").read_csv(result.comparison_csv_path)
    assert comparison["candidate_id"].tolist() == ["candidate-0", "candidate-1"]
    evidence = json.loads(Path(result.evidence_path).read_text(encoding="utf-8"))
    assert evidence["candidate_count"] == 2
    assert evidence["grounding_condition"] == "oracle_gt_mask"
    assert evidence["recomputed_summary"] == {
        "association_mismatches": 0,
        "binary_validity_mismatches": 0,
        "collision_mismatches": 0,
        "friction_max_abs_error": 0.0,
    }
    gate, loader_evidence = load_evaluator_parity_gate(
        result.gate_path, evidence_policy="test"
    )
    assert gate.validated
    assert loader_evidence["verified_evidence"]["fixture_only"] is True


def test_parity_runner_rejects_non_oracle_candidate_bundle(tmp_path: Path) -> None:
    fixture = _parity_fixture(tmp_path)
    bundle = Path(fixture["bundle"])
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["grounding_condition"] = "hifics_zero_shot_mask"
    payload.pop("bundle_fingerprint")
    payload["bundle_fingerprint"] = canonical_sha256(payload)
    atomic_json(bundle, payload)

    with pytest.raises(FormalValidationError, match="requires.*oracle_gt_mask"):
        run_evaluator_parity_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: bundle},
            tmp_path / "parity",
            dataset_root=fixture["dataset_root"],
            scope="test_fixture",
            fixture_only=True,
            input_loader=fixture["loader"],
            official_reference=fixture["evaluator"],
            adapter_evaluator=fixture["evaluator"],
        )


def test_parity_runner_writes_failed_evidence_and_withholds_gate_on_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _parity_fixture(tmp_path)

    def mismatching_adapter(rows: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        values = fixture["evaluator"](rows, **kwargs)
        values[1]["friction_score"] = 0.400001
        return values

    output = tmp_path / "parity"
    with pytest.raises(FormalValidationError, match="parity failed"):
        run_evaluator_parity_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: fixture["bundle"]},
            output,
            dataset_root=fixture["dataset_root"],
            scope="test_fixture",
            fixture_only=True,
            input_loader=fixture["loader"],
            official_reference=fixture["evaluator"],
            adapter_evaluator=mismatching_adapter,
        )
    evidence = json.loads((output / "evaluator_parity_evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "FAILED"
    assert evidence["recomputed_summary"]["friction_max_abs_error"] > 1e-9
    assert not (output / "evaluator_parity_gate.json").exists()


def test_parity_runner_rejects_reference_pruning_candidate_membership(tmp_path: Path) -> None:
    fixture = _parity_fixture(tmp_path)

    def pruning_reference(rows: np.ndarray, **kwargs: Any) -> list[dict[str, Any]]:
        return fixture["evaluator"](rows, **kwargs)[:1]

    with pytest.raises(FormalValidationError, match="changed candidate membership"):
        run_evaluator_parity_validation(
            fixture["manifest"],
            [fixture["group_id"]],
            {fixture["group_id"]: fixture["bundle"]},
            tmp_path / "parity",
            dataset_root=fixture["dataset_root"],
            scope="test_fixture",
            fixture_only=True,
            input_loader=fixture["loader"],
            official_reference=pruning_reference,
            adapter_evaluator=fixture["evaluator"],
        )


def test_parity_runner_rejects_callback_injection_for_formal_scope(tmp_path: Path) -> None:
    with pytest.raises(FormalValidationError, match="injection.*only for test fixtures"):
        run_evaluator_parity_validation(
            tmp_path / "missing.jsonl",
            ["group"],
            {},
            tmp_path / "output",
            dataset_root=tmp_path,
            official_reference=lambda *_args, **_kwargs: [],
        )
