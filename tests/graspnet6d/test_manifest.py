"""Deterministic unit fixtures only; these are not experiment samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from graspnet6d.io import atomic_jsonl, sha256_file
from graspnet6d.manifest import (
    SelectionConfig,
    build_language_audit,
    inspect_frame_targets,
)


def _fixture(root: Path) -> None:
    base = root / "scenes/scene_0000/kinect"
    for name in ("rgb", "depth", "label", "meta"):
        (base / name).mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.zeros((8, 10), dtype=np.uint16)
    label = np.zeros((8, 10), dtype=np.uint16)
    label[2:6, 2:5] = 6  # official object 5 / banana label is object+1
    label[2:6, 6:9] = 8
    depth[label > 0] = 1000
    Image.fromarray(rgb).save(base / "rgb/0000.png")
    Image.fromarray(depth).save(base / "depth/0000.png")
    Image.fromarray(label).save(base / "label/0000.png")
    intrinsic = np.array([[100.0, 0, 5.0], [0, 100.0, 4.0], [0, 0, 1.0]])
    np.save(base / "camK.npy", intrinsic)
    poses = np.repeat(np.eye(4)[None], 256, axis=0)
    np.save(base / "camera_poses.npy", poses)
    np.save(base / "cam0_wrt_table.npy", np.eye(4))
    savemat(
        base / "meta/0000.mat",
        {
            "cls_indexes": np.array([[6], [8]], dtype=np.int32),
            "poses": np.zeros((3, 4, 2)),
            "intrinsic_matrix": intrinsic,
            "factor_depth": np.array([[1000.0]]),
        },
    )


def test_frame_target_selection_is_real_file_driven_and_deterministic(tmp_path: Path) -> None:
    _fixture(tmp_path)
    catalog = {index: f"object {index}" for index in range(88)}
    config = SelectionConfig(
        frames_per_scene=1, max_targets_per_frame=1, min_mask_pixels=4,
        min_valid_depth_fraction=0.7, seed=20260815,
    )
    first = inspect_frame_targets(
        tmp_path, scene_id="scene_0000", camera="kinect", frame_id=0,
        split="train", catalog=catalog, config=config,
    )
    second = inspect_frame_targets(
        tmp_path, scene_id="scene_0000", camera="kinect", frame_id=0,
        split="train", catalog=catalog, config=config,
    )
    assert [item.group_id for item in first[0]] == [item.group_id for item in second[0]]
    assert len(first[0]) == 1
    assert any(item.reason == "deterministic_max_targets_per_frame_cap" for item in first[1])
    assert set(first[2]) == {6, 8}


def test_border_target_is_recorded_not_silently_skipped(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "scenes/scene_0000/kinect/label/0000.png"
    label = np.asarray(Image.open(path)).copy()
    label[0, 0] = 6
    Image.fromarray(label).save(path)
    accepted, excluded, _ = inspect_frame_targets(
        tmp_path, scene_id="scene_0000", camera="kinect", frame_id=0,
        split="train", catalog={index: str(index) for index in range(88)},
        config=SelectionConfig(frames_per_scene=1, max_targets_per_frame=3, min_mask_pixels=4),
    )
    assert all(item.target_instance_label != 6 for item in accepted)
    assert any(item.instance_label == 6 and "mask_touches_image_border" in item.reason for item in excluded)


def test_language_audit_renders_real_highlight_and_binds_manifests(tmp_path: Path) -> None:
    _fixture(tmp_path)
    base = tmp_path / "scenes/scene_0000/kinect"
    targets = tmp_path / "target_groups.jsonl"
    queries = tmp_path / "language_queries.jsonl"
    group_id = "scene_0000_kinect_0000_obj_005"
    atomic_jsonl(
        targets,
        [
            {
                "group_id": group_id,
                "target_object_id": 5,
                "target_instance_label": 6,
                "rgb_path": str(base / "rgb/0000.png"),
                "instance_label_path": str(base / "label/0000.png"),
            }
        ],
    )
    atomic_jsonl(
        queries,
        [
            {
                "group_id": group_id,
                "query": "Pick the banana.",
                "template_family": "catalog_name",
                "resolver_result": [5],
                "is_unique": True,
                "provenance": "derived",
            }
        ],
    )
    output = tmp_path / "audit"
    result = build_language_audit(
        targets,
        queries,
        output,
        sample_count=1,
        required_minimum=1,
    )
    assert result["audited_group_count"] == 1
    assert Path(result["html_path"]).is_file()
    assert "not official GraspNet language" in Path(result["html_path"]).read_text(
        encoding="utf-8"
    )
    manifest_path = Path(result["audit_manifest_path"])
    assert result["audit_manifest_sha256"] == sha256_file(manifest_path)
    rendered = next((output / "figures/language_audit").glob("*.png"))
    assert np.asarray(Image.open(rendered))[3, 3].tolist() != [0, 0, 0]
