from pathlib import Path

import numpy as np
import pytest

from graspnet6d.device import DeviceBenchmark, choose_formal_device, resolve_device
from graspnet6d.vgn import (
    DEFAULT_CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    EvaluatorGeometryContract,
    ExtractionConfig,
    UnvalidatedEvaluatorGeometryError,
    VGNCandidate,
    candidate_to_graspnet_row,
    checkpoint_sha256,
    deterministic_candidates_from_processed,
    load_frozen_vgn,
    pose_nms,
    run_vgn,
    validate_vgn_input,
)


def test_vgn_input_contract() -> None:
    value = validate_vgn_input(np.zeros((1, 40, 40, 40), dtype=np.float64))
    assert value.shape == (1, 40, 40, 40)
    assert value.dtype == np.float32
    with pytest.raises(ValueError):
        validate_vgn_input(np.zeros((40, 40, 40), dtype=np.float32))


def test_vgn_output_finite_with_real_frozen_checkpoint() -> None:
    pytest.importorskip("torch")
    if not Path(DEFAULT_CHECKPOINT).is_file():
        pytest.skip("official pretrained VGN checkpoint is not installed")
    assert checkpoint_sha256() == EXPECTED_CHECKPOINT_SHA256
    model = load_frozen_vgn(device="cpu")
    output = run_vgn(np.zeros((1, 40, 40, 40), dtype=np.float32), model, device="cpu")
    assert output.quality.shape == (40, 40, 40)
    assert output.rotation_xyzw.shape == (4, 40, 40, 40)
    assert output.width_voxels.shape == (40, 40, 40)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not model.training
    assert np.all(np.isfinite(output.quality))
    assert np.all(np.isfinite(output.rotation_xyzw))
    assert np.all(np.isfinite(output.width_voxels))


def test_native_candidate_order_is_score_then_voxel_not_upstream_permutation() -> None:
    quality = np.zeros((40, 40, 40), dtype=np.float32)
    quality[3, 4, 5] = 0.95
    quality[20, 21, 22] = 0.97
    quality[30, 31, 32] = 0.95
    rotation = np.zeros((4, 40, 40, 40), dtype=np.float32)
    rotation[3] = 1.0
    width = np.full((40, 40, 40), 5.0, dtype=np.float32)
    candidates = deterministic_candidates_from_processed(
        quality, rotation, width, group_id="scene/frame/target"
    )
    assert [item.voxel_index for item in candidates] == [
        (20, 21, 22),
        (3, 4, 5),
        (30, 31, 32),
    ]
    assert [item.native_rank for item in candidates] == [1, 2, 3]


def _candidate(identifier: str, rank: int, x: float, width: float = 0.05) -> VGNCandidate:
    return VGNCandidate(
        candidate_id=identifier,
        group_id="g",
        native_rank=rank,
        native_score=1.0 - rank / 100,
        translation_local_m=np.array([x, 0.0, 0.0]),
        rotation_local_vgn=np.eye(3),
        width_m=width,
        voxel_index=(rank, 0, 0),
    )


def test_pose_nms_deterministic() -> None:
    config = ExtractionConfig(
        translation_threshold_m=0.015,
        rotation_threshold_deg=15.0,
        width_threshold_m=0.010,
    )
    values = [_candidate("third", 3, 0.03), _candidate("second", 2, 0.01), _candidate("first", 1, 0.0)]
    expected = ["first", "third"]
    assert [item.candidate_id for item in pose_nms(values, config)] == expected
    assert [item.candidate_id for item in pose_nms(list(reversed(values)), config)] == expected


def test_vgn_to_graspnet_conversion_requires_real_validation_artifact() -> None:
    candidate = _candidate("candidate", 1, 0.0)
    candidate = VGNCandidate(
        **{
            **candidate.__dict__,
            "translation_camera_m": np.zeros(3),
            "rotation_camera_vgn": np.eye(3),
        }
    )
    blocked = EvaluatorGeometryContract(False, "", np.eye(3), 0.02, 0.04)
    with pytest.raises(UnvalidatedEvaluatorGeometryError):
        candidate_to_graspnet_row(candidate, blocked)


def test_cpu_mps_parity_or_explicit_cpu_decision() -> None:
    assert str(resolve_device("auto")) == "cpu"
    cpu = DeviceBenchmark("cpu", True, True, 1, 1.0, 0.0, 0.0, 1.0, 1.0, None, False)
    unavailable_mps = DeviceBenchmark(
        "mps", False, False, 1, None, None, None, None, None, None, False, "unavailable"
    )
    decision = choose_formal_device(cpu, unavailable_mps)
    assert decision.device == "cpu"
    assert "unavailable" in decision.reason.lower()
