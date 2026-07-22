"""Dependency-light tests for resumable full Dex-Net candidate generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.grasping.dexnet_run_reliability as reliability
from src.grasping.camera_geometry import T_CAMERA_GRASP_FIXED_APPROACH_KEY
from src.grasping.dexnet_run_reliability import (
    FAILED,
    SUCCESS_EMPTY,
    SUCCESS_NONEMPTY,
    SUCCESS_REQUIRED_FILES,
    ValidationResult,
    atomic_commit_sample,
    atomic_write_json,
    canonical_json_hash,
    decide_sample_action,
    has_identity_mismatch,
    make_staging_directory,
    select_manifest_indices,
    select_sample_ids,
    sha256_file,
    validate_sample_output,
    write_completion_marker,
)
from src.grasping.grasp_serialization import save_candidate_bundle, save_candidates_json


SAMPLE_ID = "q0000000_test"
CONFIG = {"generation": {"num_grasp_samples": 256, "top_k": 30, "seed": 42}}
CONFIG_HASH = canonical_json_hash(CONFIG)
CONFIG_FILE_HASH = "a" * 64
RUNTIME = {
    "version": "1.3.0",
    "release": "v1.3.0",
    "commit": "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21",
    "sampler_class": "AntipodalDepthImageGraspSampler",
}


def candidate() -> dict:
    return {
        "candidate_id": "g0000",
        "sample_id": SAMPLE_ID,
        "query": "test object",
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "center_uv": [10.0, 12.0],
        "center_depth_m": 1.0,
        "center_camera_xyz_m": [0.0, 0.0, 1.0],
        "angle_rad": 0.2,
        "width_m": 0.05,
        "width_px": 20.0,
        "endpoints_uv": [[0.0, 12.0], [20.0, 12.0]],
        "grasp_axis_mask_support": 1.0,
        "centre_boundary_distance_px": 5.0,
        "centre_inside_mask": True,
        "rejection_reason": None,
        "camera_frame": "ocid_camera_optical",
        "seed": 42,
        T_CAMERA_GRASP_FIXED_APPROACH_KEY: np.eye(4).tolist(),
    }


def build_success(root: Path, *, empty: bool = False) -> Path:
    root.mkdir(parents=True)
    candidates = [] if empty else [candidate()]
    count = len(candidates)
    metadata = {
        "schema_version": 1,
        "sample_id": SAMPLE_ID,
        "question_index": 0,
        "scene_id": "scene/image.png",
        "query": "test object",
        "representation": "planar_parallel_jaw_4dof",
        "approach_constraint": "fixed_camera_optical_axis",
        "camera_frame": "ocid_camera_optical",
        "pose": {
            "name": T_CAMERA_GRASP_FIXED_APPROACH_KEY,
            "is_freely_predicted_6dof": False,
        },
        "counts": {
            "requested": 256,
            "raw": count,
            "mask_validated": count,
            "post_nms": count,
            "top_k": count,
            "scored": 0,
        },
        "timing_ms": {"generation": 1.0, "scoring": 0.0, "total": 1.0},
        "mask_area_px": 16,
        "valid_target_depth_px": 16,
        "rejection_summary": {},
        "seed": 42,
        "config": CONFIG,
        "gqcnn_runtime": RUNTIME,
        "failure_reason": "official_sampler_returned_no_candidates" if empty else None,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name in (
        "raw_candidates.json",
        "mask_validated_candidates.json",
        "filtered_candidates.json",
        "topk_candidates.json",
    ):
        save_candidates_json(root / name, candidates)
    save_candidate_bundle(
        candidates,
        json_path=root / "candidates.json",
        npz_path=root / "candidates.npz",
        csv_path=root / "candidates.csv",
        metadata=metadata,
    )
    (root / "rejection_summary.json").write_text("{}\n", encoding="utf-8")
    (root / "camera.intr").write_text("frame ocid_camera_optical\n", encoding="utf-8")
    np.save(root / "depth_m.npy", np.ones((4, 4), dtype=np.float32), allow_pickle=False)
    Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(
        root / "hifics_mask_processed.png"
    )
    summary = {
        "sample_id": SAMPLE_ID,
        "query": "test object",
        "mask_area_px": 16,
        "valid_target_depth_px": 16,
        "requested_candidate_count": 256,
        "raw_candidate_count": count,
        "mask_validated_count": count,
        "post_nms_count": count,
        "scored_candidate_count": 0,
        "best_gqcnn_q": "",
        "median_gqcnn_q": "",
        "generation_time_ms": 1.0,
        "scoring_time_ms": 0.0,
        "total_time_ms": 1.0,
        "failure_reason": metadata["failure_reason"] or "",
        "status": SUCCESS_EMPTY if empty else SUCCESS_NONEMPTY,
        "question_index": 0,
        "scene_id": "scene/image.png",
    }
    write_completion_marker(
        root,
        sample_id=SAMPLE_ID,
        question_index=0,
        configuration_hash=CONFIG_HASH,
        config_file_sha256=CONFIG_FILE_HASH,
        seed=42,
        sampler_runtime=RUNTIME,
        counts=metadata["counts"],
        status=summary["status"],
        required_files=SUCCESS_REQUIRED_FILES,
        summary_row=summary,
        failure_reason=metadata["failure_reason"],
    )
    return root


def validate(path: Path):
    return validate_sample_output(
        path,
        expected_sample_id=SAMPLE_ID,
        expected_configuration_hash=CONFIG_HASH,
        expected_config_file_sha256=CONFIG_FILE_HASH,
        expected_seed=42,
        verify_hashes=True,
    )


def test_shards_cover_without_overlap_and_final_shard_is_uneven() -> None:
    shards = [select_manifest_indices(17, shard_index=index, num_shards=4) for index in range(4)]
    assert sorted(value for shard in shards for value in shard) == list(range(17))
    assert sum(map(len, shards)) == len(set(value for shard in shards for value in shard))
    assert [len(shard) for shard in shards] == [5, 4, 4, 4]
    assert shards == [select_manifest_indices(17, shard_index=index, num_shards=4) for index in range(4)]


def test_half_open_range_and_selection_order() -> None:
    ids = [f"s{index}" for index in range(10)]
    assert select_sample_ids(ids, start_index=2, end_index=8, shard_index=0, num_shards=3) == ["s3", "s6"]
    with pytest.raises(ValueError, match="half-open"):
        select_sample_ids(ids, start_index=8, end_index=2)
    with pytest.raises(ValueError, match="supplied together"):
        select_sample_ids(ids, shard_index=0)


def test_resume_overwrite_retry_and_incomplete_decisions() -> None:
    valid = ValidationResult(SAMPLE_ID, True, status=SUCCESS_NONEMPTY)
    failed = ValidationResult(SAMPLE_ID, True, status=FAILED)
    invalid = ValidationResult(SAMPLE_ID, False, errors=["missing file"])
    assert decide_sample_action(output_exists=False, validation=None, resume=False, overwrite_existing=False, retry_failures=False) == "process"
    assert decide_sample_action(output_exists=True, validation=valid, resume=True, overwrite_existing=False, retry_failures=False) == "skip"
    assert decide_sample_action(output_exists=True, validation=invalid, resume=True, overwrite_existing=False, retry_failures=False) == "process"
    assert decide_sample_action(output_exists=True, validation=valid, resume=False, overwrite_existing=True, retry_failures=False) == "process"
    assert decide_sample_action(output_exists=True, validation=failed, resume=True, overwrite_existing=False, retry_failures=False) == "skip"
    assert decide_sample_action(output_exists=True, validation=failed, resume=True, overwrite_existing=False, retry_failures=True) == "process"
    with pytest.raises(FileExistsError):
        decide_sample_action(output_exists=True, validation=valid, resume=False, overwrite_existing=False, retry_failures=False)


@pytest.mark.parametrize("empty,expected", [(False, SUCCESS_NONEMPTY), (True, SUCCESS_EMPTY)])
def test_success_marker_and_zero_candidate_serialization(tmp_path: Path, empty: bool, expected: str) -> None:
    result = validate(build_success(tmp_path / SAMPLE_ID, empty=empty))
    assert result.valid, result.errors
    assert result.status == expected
    with np.load(tmp_path / SAMPLE_ID / "candidates.npz", allow_pickle=False) as arrays:
        count = 0 if empty else 1
        assert arrays["center_uv"].shape == (count, 2)
        assert arrays[T_CAMERA_GRASP_FIXED_APPROACH_KEY].shape == (count, 4, 4)


def test_missing_and_corrupt_npz_are_detected(tmp_path: Path) -> None:
    root = build_success(tmp_path / SAMPLE_ID)
    (root / "candidates.npz").unlink()
    result = validate(root)
    assert not result.valid
    assert any("missing" in error for error in result.errors)

    root = build_success(tmp_path / "second")
    (root / "candidates.npz").write_bytes(b"not an npz")
    result = validate(root)
    assert not result.valid
    assert any("hash mismatch" in error or "unreadable" in error for error in result.errors)


def test_configuration_seed_and_required_hash_mismatch(tmp_path: Path) -> None:
    root = build_success(tmp_path / SAMPLE_ID)
    wrong_config = validate_sample_output(
        root,
        expected_sample_id=SAMPLE_ID,
        expected_configuration_hash="b" * 64,
        expected_config_file_sha256=CONFIG_FILE_HASH,
        expected_seed=42,
    )
    assert not wrong_config.valid
    assert any("configuration hash mismatch" in error for error in wrong_config.errors)
    wrong_seed = validate_sample_output(
        root,
        expected_sample_id=SAMPLE_ID,
        expected_configuration_hash=CONFIG_HASH,
        expected_config_file_sha256=CONFIG_FILE_HASH,
        expected_seed=7,
    )
    assert not wrong_seed.valid
    assert any("seed mismatch" in error for error in wrong_seed.errors)
    with (root / "candidates.csv").open("a", encoding="utf-8") as stream:
        stream.write("corrupt\n")
    assert any("hash mismatch" in error for error in validate(root).errors)


def test_marker_required_files_counts_sampler_and_visual_policy_are_cross_checked(
    tmp_path: Path,
) -> None:
    root = build_success(tmp_path / SAMPLE_ID)
    marker_path = root / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text())
    marker["required_files"].remove("depth_m.npy")
    marker["required_file_hashes"].pop("depth_m.npy")
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert any("omits required files" in error for error in validate(root).errors)

    root = build_success(tmp_path / "count-mismatch")
    marker_path = root / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text())
    marker["candidate_counts"]["post_nms"] = 99
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert any("candidate count post_nms mismatch" in error for error in validate(root).errors)

    root = build_success(tmp_path / "runtime-mismatch")
    result = validate_sample_output(
        root,
        expected_sample_id=SAMPLE_ID,
        expected_configuration_hash=CONFIG_HASH,
        expected_config_file_sha256=CONFIG_FILE_HASH,
        expected_seed=42,
        expected_sampler_runtime={**RUNTIME, "commit": "wrong"},
        expect_visualizations=True,
    )
    assert not result.valid
    assert has_identity_mismatch(result)
    assert any("sampler commit mismatch" in error for error in result.errors)
    assert any("visualization policy requires" in error for error in result.errors)


def test_npz_values_must_match_candidates_json_even_with_updated_hash(tmp_path: Path) -> None:
    root = build_success(tmp_path / SAMPLE_ID)
    npz_path = root / "candidates.npz"
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    arrays["center_depth_m"][0] = 9.0
    np.savez(npz_path, **arrays)
    marker_path = root / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text())
    marker["required_file_hashes"]["candidates.npz"] = sha256_file(npz_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    result = validate(root)
    assert not result.valid
    assert any("values differ from candidates.json" in error for error in result.errors)


def test_atomic_sample_commit_hides_temporary_and_replaces_only_after_ready(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    final = output / SAMPLE_ID
    final.mkdir()
    (final / "old.txt").write_text("old", encoding="utf-8")
    staging = make_staging_directory(output, SAMPLE_ID)
    (staging / "new.txt").write_text("new", encoding="utf-8")
    assert (final / "old.txt").is_file()
    assert staging.name.startswith(f".{SAMPLE_ID}.tmp.")
    atomic_commit_sample(staging, final, output)
    assert not (final / "old.txt").exists()
    assert (final / "new.txt").read_text() == "new"
    assert not list(output.glob(f".{SAMPLE_ID}.backup.*"))


def test_post_publish_fsync_failure_does_not_turn_commit_into_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    final = output / SAMPLE_ID
    staging = make_staging_directory(output, SAMPLE_ID)
    (staging / "new.txt").write_text("new", encoding="utf-8")
    original = reliability.fsync_directory

    def fail_only_for_output(path: Path) -> None:
        if Path(path) == output.resolve():
            raise OSError("simulated directory fsync failure")
        original(path)

    monkeypatch.setattr(reliability, "fsync_directory", fail_only_for_output)
    atomic_commit_sample(staging, final, output)
    assert (final / "new.txt").read_text() == "new"


def test_progress_style_atomic_json_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    atomic_write_json(path, {"remaining": 10})
    atomic_write_json(path, {"remaining": 9})
    assert json.loads(path.read_text()) == {"remaining": 9}
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_marker_is_valid_terminal_but_explicit(tmp_path: Path) -> None:
    root = tmp_path / SAMPLE_ID
    root.mkdir()
    (root / "failure.json").write_text('{"failure_reason":"boom"}\n', encoding="utf-8")
    summary = {"sample_id": SAMPLE_ID, "status": FAILED, "failure_reason": "boom"}
    write_completion_marker(
        root,
        sample_id=SAMPLE_ID,
        question_index=0,
        configuration_hash=CONFIG_HASH,
        config_file_sha256=CONFIG_FILE_HASH,
        seed=42,
        sampler_runtime=RUNTIME,
        counts={},
        status=FAILED,
        required_files=("failure.json",),
        summary_row=summary,
        failure_reason="boom",
    )
    result = validate(root)
    assert result.valid
    assert result.status == FAILED
    assert sum(status in {SUCCESS_NONEMPTY, SUCCESS_EMPTY, FAILED} for status in [result.status]) == 1
