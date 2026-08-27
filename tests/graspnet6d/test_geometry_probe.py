from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from graspnet6d.formal_validation import (
    RAW_VGN_VALIDATION_SCHEMA,
    _load_raw_candidates,
)
from graspnet6d.geometry_probe import (
    RAW_VGN_EMPTY_OBSERVATION_SCHEMA,
    discover_geometry_probe_group_ids,
    promote_geometry_probe_bundles,
    run_geometry_probe_stage,
    select_nonempty_geometry_probe_paths,
)
from graspnet6d.io import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    canonical_sha256,
    sha256_file,
)
from graspnet6d.stages import (
    GEOMETRY_SCHEMA,
    StageBatchError,
    StageInputError,
    TSDF_SCHEMA,
    VGN_BUNDLE_SCHEMA,
    _slug,
    run_vgn_candidate_stage,
)
from graspnet6d.vgn import ExtractionConfig, VGNCandidate


CONDITION = "oracle_gt_mask"
MASK_INPUT = "a" * 64
MASK_COMMIT = "b" * 64


def _inputs(tmp_path: Path, count: int = 2) -> tuple[Path, Path, Path, Path, list[str]]:
    root = tmp_path / "outputs"
    targets: list[dict[str, Any]] = []
    languages: list[dict[str, Any]] = []
    group_ids: list[str] = []
    for index in range(count):
        group_id = f"scene_000{index}_kinect_0000_obj_000"
        group_ids.append(group_id)
        targets.append(
            {
                "group_id": group_id,
                "split": "train",
                "scene_id": f"scene_000{index}",
                "camera": "kinect",
                "frame_id": 0,
                "target_object_id": 0,
                "target_instance_label": 1,
                "depth_path": str(tmp_path / f"depth-{index}.png"),
                "instance_label_path": str(tmp_path / f"label-{index}.png"),
                "meta_path": str(tmp_path / f"meta-{index}.mat"),
                "intrinsics_path": str(tmp_path / f"intrinsics-{index}.npy"),
                "camera_pose_path": str(tmp_path / f"pose-{index}.npy"),
                "table_transform_path": str(tmp_path / f"table-{index}.npy"),
            }
        )
        languages.append(
            {
                "group_id": group_id,
                "query": "the target object",
                "is_unique": True,
                "resolver_result": [0],
            }
        )
        tsdf_path = root / "target_tsdf" / CONDITION / f"{_slug(group_id)}.npz"
        local_to_camera = np.eye(4, dtype=np.float64)
        local_to_camera[:3, 3] = [float(index), 0.0, 1.0]
        local_to_table = np.eye(4, dtype=np.float64)
        local_to_table[:3, 3] = [0.0, float(index), 0.5]
        atomic_npz(
            tsdf_path,
            tsdf=np.full((1, 40, 40, 40), 0.1 + index, dtype=np.float32),
            T_local_to_camera=local_to_camera,
            T_local_to_table=local_to_table,
            physical_size=np.float64(0.30),
            full_scene_depth_integrated=np.bool_(True),
            grounding_condition=np.asarray(CONDITION),
            grounding_mask_input_fingerprint=np.asarray(MASK_INPUT),
            grounding_mask_commit_sha256=np.asarray(MASK_COMMIT),
        )
        atomic_json(
            tsdf_path.with_suffix(".json"),
            {
                "schema_version": TSDF_SCHEMA,
                "group_id": group_id,
                "grounding_condition": CONDITION,
                "grounding_mask_input_fingerprint": MASK_INPUT,
                "grounding_mask_commit_sha256": MASK_COMMIT,
                "output_sha256": sha256_file(tsdf_path),
            },
        )
    target_path = tmp_path / "targets.jsonl"
    language_path = tmp_path / "language.jsonl"
    atomic_jsonl(target_path, targets)
    atomic_jsonl(language_path, languages)
    checkpoint = tmp_path / "vgn_conv.pth"
    checkpoint.write_bytes(b"injected-test-checkpoint")
    return target_path, language_path, root, checkpoint, group_ids


def _candidates(
    *,
    group_id: str,
    T_local_to_camera: np.ndarray,
    T_local_to_table: np.ndarray,
) -> list[VGNCandidate]:
    values: list[VGNCandidate] = []
    for rank, coordinate, score in ((1, 0.05, 0.9), (3, 0.20, 0.7)):
        local = np.array([coordinate, coordinate + 0.01, coordinate + 0.02])
        rotation = np.eye(3)
        values.append(
            VGNCandidate(
                candidate_id=f"{group_id}-candidate-{rank}",
                group_id=group_id,
                native_rank=rank,
                native_score=score,
                translation_local_m=local,
                rotation_local_vgn=rotation,
                width_m=0.04 + rank * 0.001,
                voxel_index=(rank, rank + 1, rank + 2),
                translation_camera_m=(
                    T_local_to_camera[:3, :3] @ local + T_local_to_camera[:3, 3]
                ),
                rotation_camera_vgn=T_local_to_camera[:3, :3] @ rotation,
                translation_table_m=(
                    T_local_to_table[:3, :3] @ local + T_local_to_table[:3, 3]
                ),
                rotation_table_vgn=T_local_to_table[:3, :3] @ rotation,
            )
        )
    return values


def _geometry_gate(tmp_path: Path, *, height_m: float = 0.02) -> Path:
    artifact = tmp_path / "geometry-validation.json"
    atomic_json(artifact, {"status": "PASSED", "fixture_only": True})
    contract = tmp_path / "geometry-contract.json"
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


def test_probe_runs_once_per_group_loads_once_preserves_order_and_resumes(
    tmp_path: Path,
) -> None:
    target, language, root, checkpoint, group_ids = _inputs(tmp_path)
    calls = {"load": 0, "infer": 0, "extract": 0}
    seen_transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    config = ExtractionConfig(
        pre_nms_max_candidates=12,
        frozen_top_k=5,
        translation_threshold_m=0.01,
        rotation_threshold_deg=10.0,
        width_threshold_m=0.005,
    )

    def load_model(**kwargs: Any) -> object:
        assert kwargs == {"checkpoint": checkpoint.resolve(), "device": "cpu"}
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
        config: ExtractionConfig,
        T_local_to_camera: np.ndarray,
        T_local_to_table: np.ndarray,
    ) -> list[VGNCandidate]:
        assert raw is not None and tsdf.shape == (1, 40, 40, 40)
        assert config == config_expected
        calls["extract"] += 1
        seen_transforms[group_id] = (
            T_local_to_camera.copy(),
            T_local_to_table.copy(),
        )
        return _candidates(
            group_id=group_id,
            T_local_to_camera=T_local_to_camera,
            T_local_to_table=T_local_to_table,
        )

    config_expected = config
    first = run_geometry_probe_stage(
        target,
        language,
        root,
        checkpoint=checkpoint,
        config=config,
        model_loader=load_model,
        inference=infer,
        extractor=extract,
    )
    hashes = {path: sha256_file(path) for path in first.output_paths}
    second = run_geometry_probe_stage(
        target,
        language,
        root,
        checkpoint=checkpoint,
        config=config,
        resume=True,
        model_loader=load_model,
        inference=infer,
        extractor=extract,
    )

    assert calls == {"load": 1, "infer": 2, "extract": 2}
    assert first.completed_groups == 2 and first.resumed_groups == 0
    assert second.completed_groups == 0 and second.resumed_groups == 2
    assert {path: sha256_file(path) for path in first.output_paths} == hashes
    assert set(seen_transforms) == set(group_ids)
    assert not list(root.rglob("*.tmp"))
    for output_string in first.output_paths:
        output = Path(output_string)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["schema_version"] == RAW_VGN_VALIDATION_SCHEMA
        assert payload["grounding_condition"] == CONDITION
        assert payload["grounding_mask_input_fingerprint"] == MASK_INPUT
        assert payload["grounding_mask_commit_sha256"] == MASK_COMMIT
        assert payload["checkpoint_sha256"] == sha256_file(checkpoint)
        assert payload["tsdf_sha256"] == sha256_file(payload["tsdf_path"])
        assert payload["device"] == "cpu"
        assert payload["extraction_config"] == {
            "pre_nms_max_candidates": 12,
            "frozen_top_k": 5,
            "translation_threshold_m": 0.01,
            "rotation_threshold_deg": 10.0,
            "width_threshold_m": 0.005,
        }
        assert payload["inference_calls_for_group"] == 1
        assert payload["candidate_count"] == 2
        assert [row["native_rank"] for row in payload["raw_vgn_candidates"]] == [
            1,
            3,
        ]
        assert payload["raw_vgn_pool_fingerprint"] == canonical_sha256(
            payload["raw_vgn_candidates"]
        )
        checked = dict(payload)
        fingerprint = checked.pop("bundle_fingerprint")
        assert fingerprint == canonical_sha256(checked)
        assert len(payload["input_fingerprint"]) == 64
        loaded, raw = _load_raw_candidates(
            output,
            payload["group_id"],
            Path(payload["tsdf_path"]),
            fixture_only=False,
        )
        assert loaded["bundle_fingerprint"] == fingerprint
        assert [record["candidate_id"] for record in raw] == [
            record["candidate_id"] for record in payload["raw_vgn_candidates"]
        ]

    with pytest.raises(StageInputError, match="at least 20"):
        select_nonempty_geometry_probe_paths(root, group_ids)
    selected = select_nonempty_geometry_probe_paths(
        root, reversed(group_ids), minimum_groups=2
    )
    assert list(selected) == list(reversed(group_ids))
    assert all(path.is_file() for path in selected.values())


def test_probe_records_hash_bound_empty_observation_and_resumes_without_inference(
    tmp_path: Path,
) -> None:
    target, language, root, checkpoint, group_ids = _inputs(tmp_path, count=1)
    calls = {"load": 0, "infer": 0}

    def load_model(**_: Any) -> object:
        calls["load"] += 1
        return object()

    def infer(*_: Any, **__: Any) -> object:
        calls["infer"] += 1
        return object()

    first = run_geometry_probe_stage(
        target,
        language,
        root,
        checkpoint=checkpoint,
        model_loader=load_model,
        inference=infer,
        extractor=lambda *_args, **_kwargs: [],
    )

    assert calls == {"load": 1, "infer": 1}
    assert first.completed_groups == 1
    marker_path = Path(first.output_paths[0])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["schema_version"] == RAW_VGN_EMPTY_OBSERVATION_SCHEMA
    assert marker["group_id"] == group_ids[0]
    assert marker["inference_calls_for_group"] == 1
    assert marker["candidate_count"] == 0
    assert marker["raw_vgn_candidates"] == []
    assert marker["raw_vgn_pool_fingerprint"] == canonical_sha256([])
    checked = dict(marker)
    assert checked.pop("bundle_fingerprint") == canonical_sha256(checked)
    assert discover_geometry_probe_group_ids(
        root, grounding_condition=CONDITION
    ) == tuple(group_ids)
    with pytest.raises(StageInputError, match="empty candidate pool"):
        select_nonempty_geometry_probe_paths(root, group_ids, minimum_groups=1)

    marker_hash = sha256_file(marker_path)
    second = run_geometry_probe_stage(
        target,
        language,
        root,
        checkpoint=checkpoint,
        resume=True,
        model_loader=lambda **_: pytest.fail("empty observation must resume"),
        inference=lambda *_args, **_kwargs: pytest.fail(
            "empty observation must prevent a second inference"
        ),
        extractor=lambda *_args, **_kwargs: pytest.fail(
            "empty observation must prevent a second extraction"
        ),
    )
    assert second.completed_groups == 0
    assert second.resumed_groups == 1
    assert sha256_file(second.output_paths[0]) == marker_hash

    marker["grounding_mask_commit_sha256"] = "c" * 64
    atomic_json(marker_path, marker)
    with pytest.raises(StageBatchError) as caught:
        run_geometry_probe_stage(
            target,
            language,
            root,
            checkpoint=checkpoint,
            resume=True,
            model_loader=lambda **_: pytest.fail("tampered marker must fail closed"),
            inference=lambda *_args, **_kwargs: pytest.fail(
                "tampered marker must fail before inference"
            ),
            extractor=lambda *_args, **_kwargs: pytest.fail(
                "tampered marker must fail before extraction"
            ),
        )
    assert "bundle fingerprint mismatch" in caught.value.failures[0].message
    assert not list(
        (root / "geometry_probe" / "raw_vgn_candidates" / CONDITION).glob("*.json")
    )


def test_probe_rejects_stale_or_tampered_resume_without_new_inference(
    tmp_path: Path,
) -> None:
    target, language, root, checkpoint, group_ids = _inputs(tmp_path, count=1)

    def extract(
        _tsdf: np.ndarray,
        _raw: object,
        *,
        group_id: str,
        T_local_to_camera: np.ndarray,
        T_local_to_table: np.ndarray,
        **_: Any,
    ) -> list[VGNCandidate]:
        return _candidates(
            group_id=group_id,
            T_local_to_camera=T_local_to_camera,
            T_local_to_table=T_local_to_table,
        )

    first = run_geometry_probe_stage(
        target,
        language,
        root,
        checkpoint=checkpoint,
        model_loader=lambda **_: object(),
        inference=lambda *_args, **_kwargs: object(),
        extractor=extract,
    )
    output = Path(first.output_paths[0])
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["candidate_count"] = 999
    atomic_json(output, payload)
    infer_calls = 0

    def should_not_infer(*_: Any, **__: Any) -> object:
        nonlocal infer_calls
        infer_calls += 1
        return object()

    with pytest.raises(StageBatchError) as caught:
        run_geometry_probe_stage(
            target,
            language,
            root,
            checkpoint=checkpoint,
            resume=True,
            model_loader=lambda **_: pytest.fail(
                "stale bundle must fail before loading"
            ),
            inference=should_not_infer,
            extractor=extract,
        )
    assert infer_calls == 0
    assert caught.value.failures[0].group_id == group_ids[0]
    assert "candidate_count" in caught.value.failures[0].message


def test_promotion_reuses_exact_raw_pool_and_normal_stage_resumes_without_inference(
    tmp_path: Path,
) -> None:
    target, language, root, checkpoint, group_ids = _inputs(tmp_path)
    geometry = _geometry_gate(tmp_path)
    config = ExtractionConfig(
        pre_nms_max_candidates=12,
        frozen_top_k=5,
        translation_threshold_m=0.01,
        rotation_threshold_deg=10.0,
        width_threshold_m=0.005,
    )
    calls = {"load": 0, "infer": 0, "extract": 0}

    def load_model(**_: Any) -> object:
        calls["load"] += 1
        return object()

    def infer(*_: Any, **__: Any) -> object:
        calls["infer"] += 1
        return object()

    def extract(
        _tsdf: np.ndarray,
        _raw: object,
        *,
        group_id: str,
        T_local_to_camera: np.ndarray,
        T_local_to_table: np.ndarray,
        **_: Any,
    ) -> list[VGNCandidate]:
        calls["extract"] += 1
        if group_id == group_ids[1]:
            return []
        return _candidates(
            group_id=group_id,
            T_local_to_camera=T_local_to_camera,
            T_local_to_table=T_local_to_table,
        )

    raw_summary = run_geometry_probe_stage(
        target,
        language,
        root,
        checkpoint=checkpoint,
        device="cpu",
        grounding_condition=CONDITION,
        config=config,
        model_loader=load_model,
        inference=infer,
        extractor=extract,
    )
    raw_ids = {
        payload["group_id"]: [
            row["candidate_id"] for row in payload["raw_vgn_candidates"]
        ]
        for payload in (
            json.loads(Path(path).read_text(encoding="utf-8"))
            for path in raw_summary.output_paths
        )
    }
    assert discover_geometry_probe_group_ids(
        root, grounding_condition=CONDITION
    ) == tuple(group_ids)
    assert set(
        select_nonempty_geometry_probe_paths(root, [group_ids[0]], minimum_groups=1)
    ) == {group_ids[0]}
    with pytest.raises(StageInputError, match="empty candidate pool"):
        select_nonempty_geometry_probe_paths(root, [group_ids[1]], minimum_groups=1)

    promoted = promote_geometry_probe_bundles(
        target,
        language,
        root,
        geometry_contract_path=geometry,
        selected_group_ids=group_ids,
        checkpoint=checkpoint,
        device="cpu",
        grounding_condition=CONDITION,
        config=config,
        evidence_policy="test",
    )

    assert calls == {"load": 1, "infer": 2, "extract": 2}
    assert promoted.completed_groups == 2
    assert promoted.resumed_groups == 0
    for path_string in promoted.output_paths:
        payload = json.loads(Path(path_string).read_text(encoding="utf-8"))
        assert payload["schema_version"] == VGN_BUNDLE_SCHEMA
        assert [
            row["candidate_id"] for row in payload["raw_vgn_candidates"]
        ] == raw_ids[payload["group_id"]]
        assert [row["candidate_id"] for row in payload["candidate_records"]] == raw_ids[
            payload["group_id"]
        ]
        expected_ranks = [] if payload["group_id"] == group_ids[1] else [1, 3]
        assert [
            row["native_rank"] for row in payload["raw_vgn_candidates"]
        ] == expected_ranks
        assert payload["candidate_count"] == len(expected_ranks)
        if payload["group_id"] == group_ids[1]:
            assert payload["candidate_pool_fingerprint"] == canonical_sha256([])
            assert payload["graspnet_rows"] == []

    resumed = run_vgn_candidate_stage(
        target,
        language,
        root,
        geometry_contract_path=geometry,
        checkpoint=checkpoint,
        device="cpu",
        grounding_condition=CONDITION,
        config=config,
        resume=True,
        evidence_policy="test",
        model_loader=lambda **_: pytest.fail(
            "promotion must avoid a second model load"
        ),
        inference=lambda *_args, **_kwargs: pytest.fail(
            "promotion must avoid a second VGN inference"
        ),
        extractor=lambda *_args, **_kwargs: pytest.fail(
            "promotion must avoid a second candidate extraction"
        ),
    )
    assert resumed.completed_groups == 0
    assert resumed.resumed_groups == 2
    assert calls == {"load": 1, "infer": 2, "extract": 2}


def test_promotion_refuses_stale_geometry_without_overwriting(
    tmp_path: Path,
) -> None:
    target, language, root, checkpoint, group_ids = _inputs(tmp_path, count=1)
    geometry = _geometry_gate(tmp_path)
    config = ExtractionConfig(
        pre_nms_max_candidates=12,
        frozen_top_k=5,
        translation_threshold_m=0.01,
        rotation_threshold_deg=10.0,
        width_threshold_m=0.005,
    )

    def extract(
        _tsdf: np.ndarray,
        _raw: object,
        *,
        group_id: str,
        T_local_to_camera: np.ndarray,
        T_local_to_table: np.ndarray,
        **_: Any,
    ) -> list[VGNCandidate]:
        return _candidates(
            group_id=group_id,
            T_local_to_camera=T_local_to_camera,
            T_local_to_table=T_local_to_table,
        )

    run_geometry_probe_stage(
        target,
        language,
        root,
        checkpoint=checkpoint,
        config=config,
        model_loader=lambda **_: object(),
        inference=lambda *_args, **_kwargs: object(),
        extractor=extract,
    )
    first = promote_geometry_probe_bundles(
        target,
        language,
        root,
        geometry_contract_path=geometry,
        selected_group_ids=group_ids,
        checkpoint=checkpoint,
        device="cpu",
        grounding_condition=CONDITION,
        config=config,
        evidence_policy="test",
    )
    output = Path(first.output_paths[0])
    original_hash = sha256_file(output)
    atomic_json(
        geometry,
        {
            "schema_version": GEOMETRY_SCHEMA,
            "validated": True,
            "validation_artifact": "geometry-validation.json",
            "R_vgn_gripper_to_graspnet_gripper": np.eye(3).tolist(),
            "height_m": 0.03,
            "depth_m": 0.04,
        },
    )

    with pytest.raises(StageBatchError) as caught:
        promote_geometry_probe_bundles(
            target,
            language,
            root,
            geometry_contract_path=geometry,
            selected_group_ids=group_ids,
            checkpoint=checkpoint,
            device="cpu",
            grounding_condition=CONDITION,
            config=config,
            evidence_policy="test",
        )
    assert "input fingerprint mismatch" in caught.value.failures[0].message
    assert sha256_file(output) == original_hash
