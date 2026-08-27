from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from graspnet6d.dataset import (
    DatasetValidationError,
    OFFICIAL_ARCHIVES,
    check_disk_budget,
    discover_scene_ids,
    estimate_storage,
    validate_graspnet_structure,
)


def _minimal_scene(root: Path) -> None:
    scene = root / "scenes" / "scene_0003"
    camera = scene / "kinect"
    scene.mkdir(parents=True)
    camera.mkdir(parents=True)
    (scene / "object_id_list.txt").write_text("7\n", encoding="utf-8")
    np.save(scene / "rs_wrt_kn.npy", np.eye(4))
    np.save(camera / "camK.npy", np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 2.0], [0.0, 0.0, 1.0]]))
    np.save(camera / "camera_poses.npy", np.eye(4)[None])
    np.save(camera / "cam0_wrt_table.npy", np.eye(4))
    for directory in ("rgb", "depth", "label", "meta", "annotations"):
        (camera / directory).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(camera / "rgb/0000.png")
    Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(camera / "depth/0000.png")
    Image.fromarray(np.full((4, 4), 8, dtype=np.uint8)).save(camera / "label/0000.png")
    savemat(
        camera / "meta/0000.mat",
        {
            "cls_indexes": np.array([[8]]),
            "poses": np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[..., None],
            "intrinsic_matrix": np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 2.0], [0.0, 0.0, 1.0]]),
            "factor_depth": np.array([[1000.0]]),
        },
    )
    (camera / "annotations/0000.xml").write_text(
        "<scene><obj><obj_id>7</obj_id><obj_name>fixture</obj_name>"
        "<obj_path>fixture</obj_path><pos_in_world>0 0 1</pos_in_world>"
        "<ori_in_world>1 0 0 0</ori_in_world></obj></scene>\n",
        encoding="utf-8",
    )
    model = root / "models" / "007"
    model.mkdir(parents=True)
    (model / "nontextured.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n0 0 0\n",
        encoding="utf-8",
    )
    (model / "textured.obj").write_text("v 0 0 0\n", encoding="utf-8")
    (model / "textured.sdf").write_text("1 1 1\n", encoding="utf-8")
    grasp = root / "grasp_label" / "007_labels.npz"
    grasp.parent.mkdir(parents=True)
    np.savez(grasp, points=np.zeros((1, 3)), offsets=np.zeros((1, 1)), scores=np.zeros((1, 1)))
    collision = root / "collision_label" / "scene_0003" / "collision_labels.npz"
    collision.parent.mkdir(parents=True)
    np.savez(collision, np.zeros((1,), dtype=bool))


def test_published_paper_lite_archive_total_includes_train_3() -> None:
    estimate = estimate_storage("paper-lite", extracted_multiplier=1.0, cache_and_artifact_bytes=0)
    published = sum(
        OFFICIAL_ARCHIVES[key].published_size_bytes
        for key in (
            "train_4",
            "train_3",
            "grasp_label",
            "collision_label",
            "models",
        )
    )
    assert estimate.compressed_bytes == published == 33_904_237_276
    assert estimate.total_required_bytes > estimate.compressed_bytes


def test_disk_budget_preserves_twenty_percent(tmp_path: Path) -> None:
    estimate = estimate_storage("paper-lite")
    budget = check_disk_budget(tmp_path, estimate, safety_fraction=0.20)
    assert budget.safety_reserve_bytes >= int(0.20 * budget.filesystem_total_bytes)
    assert budget.allowed == (budget.shortfall_bytes == 0)


def test_strict_graspnet_structure_validator(tmp_path: Path) -> None:
    _minimal_scene(tmp_path)
    assert discover_scene_ids(tmp_path) == [3]
    report = validate_graspnet_structure(
        tmp_path, camera="kinect", scene_ids=[3], frame_ids=[0], strict=True
    )
    assert report.valid
    assert report.object_ids == [7]
    assert not report.missing


def test_missing_frame_is_never_silently_skipped(tmp_path: Path) -> None:
    _minimal_scene(tmp_path)
    missing = tmp_path / "scenes" / "scene_0003" / "kinect" / "depth" / "0000.png"
    missing.unlink()
    with pytest.raises(DatasetValidationError) as raised:
        validate_graspnet_structure(
            tmp_path, camera="kinect", scene_ids=[3], frame_ids=[0], strict=True
        )
    assert str(missing) in raised.value.report.missing
