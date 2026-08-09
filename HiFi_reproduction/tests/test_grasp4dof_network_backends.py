"""Synthetic contracts and real-checkpoint smokes for 4-DoF network adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from skimage.filters import gaussian
from torch import nn

from src.grasping.backends import (
    BackendSample,
    GGCNN2Backend,
    GGCNN2Config,
    GRConvNetBackend,
    GRConvNetConfig,
    GroundTruthAccessError,
)
from src.grasping.backends.base import prepare_conditioned_input
from src.grasping.backends.network_utils import (
    GGCNN2_CHECKPOINT_SHA256,
    GRCONVNET_CHECKPOINT_SHA256,
    load_ggcnn2_state_dict,
    load_trusted_grconvnet_full_pickle,
    official_gaussian_post_process,
    rescore_candidates_with_mask_support,
    select_device,
    sha256_file,
)
from src.grasping.backends.vendor_models import OfficialGGCNN2
from src.grasping.common import CropTransform, Grasp4DoF, GraspPrediction


class SyntheticFourMapModel(nn.Module):
    """A deterministic four-map model that also records channel order."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.pos_output = nn.Conv2d(1, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(1, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(1, 1, kernel_size=1)
        self.width_output = nn.Conv2d(1, 1, kernel_size=1)
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor):
        self.last_input = value.detach().cpu()
        shape = (value.shape[0], 1, value.shape[2], value.shape[3])
        quality = torch.zeros(shape, dtype=value.dtype, device=value.device)
        quality[:, :, value.shape[2] // 2, value.shape[3] // 2] = 1.0
        cosine = torch.ones_like(quality)
        sine = torch.zeros_like(quality)
        width = torch.full_like(quality, 0.1)
        return quality, cosine, sine, width


def _sample(
    *,
    sample_id: str = "synthetic",
    probability_value: float = 1.0,
    empty_mask: bool = False,
    invalid_target_depth: bool = False,
) -> BackendSample:
    height, width = 80, 120
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    rgb = np.stack(
        (
            np.broadcast_to(columns, (height, width)),
            np.broadcast_to(rows, (height, width)),
            np.broadcast_to(columns + rows, (height, width)),
        ),
        axis=-1,
    ).astype(np.uint8)
    depth = (1.0 + rows * 0.001 + columns * 0.002).astype(np.float32)
    mask = np.zeros((height, width), dtype=bool)
    if not empty_mask:
        mask[20:61, 40:81] = True
    if invalid_target_depth:
        depth[mask] = 0.0
    probability = mask.astype(np.float32) * np.float32(probability_value)
    return BackendSample(
        sample_id=sample_id,
        rgb=rgb,
        depth_m=depth,
        predicted_mask=mask,
        probability_map=probability,
    )


def _config(config_type, **overrides):
    values = dict(
        device="cpu",
        input_size=32,
        minimum_crop_side_px=32,
        quality_threshold=0.001,
        min_peak_distance_px=3,
        max_raw_candidates=20,
    )
    values.update(overrides)
    return config_type(**values)


@pytest.mark.parametrize(
    ("backend_type", "config_type", "input_channels"),
    (
        (GRConvNetBackend, GRConvNetConfig, 4),
        (GGCNN2Backend, GGCNN2Config, 1),
    ),
)
@pytest.mark.parametrize("variant", ("hard_mask", "dilated_crop"))
def test_synthetic_predict_restores_native_coordinates_and_complete_pool(
    backend_type, config_type, input_channels: int, variant: str
) -> None:
    model = SyntheticFourMapModel(input_channels)
    backend = backend_type(
        _config(config_type, conditioning_variant=variant), model=model
    )

    prediction = backend.predict(_sample())

    assert isinstance(prediction, GraspPrediction)
    assert prediction.top1 is not None
    assert prediction.top1.center_x == pytest.approx(60.0, abs=2.0)
    assert prediction.top1.center_y == pytest.approx(40.0, abs=2.0)
    assert 1 <= len(prediction.top5) <= 5
    assert len(prediction.candidates) == prediction.nms_candidate_count
    assert prediction.top5 == prediction.candidates[:5]
    assert prediction.top1.metadata["center_mask_support"] > 0.9
    assert "jaw_mask_support" in prediction.top1.metadata
    assert prediction.metadata["mask_source"] == "predicted"


def test_grconvnet_input_order_is_depth_then_rgb() -> None:
    model = SyntheticFourMapModel(4)
    config = _config(GRConvNetConfig)
    sample = _sample()
    expected = prepare_conditioned_input(sample, config)
    backend = GRConvNetBackend(config, model=model)

    backend.predict(sample)

    assert model.last_input is not None
    assert model.last_input.shape[1] == 4
    assert model.last_input.numpy()[0, 0] == pytest.approx(expected.depth_chw[0])
    assert model.last_input.numpy()[0, 1:] == pytest.approx(expected.rgb_chw)


def test_soft_probability_gate_can_produce_traceable_no_grasp() -> None:
    backend = GGCNN2Backend(
        _config(GGCNN2Config), model=SyntheticFourMapModel(1)
    )

    prediction = backend.predict(_sample(probability_value=0.0))

    assert prediction.top1 is None
    assert prediction.candidates == ()
    assert prediction.empty_reason == "no_candidate_generated"
    assert prediction.metadata["failure_stage"] == "no_candidate_generated"


def test_center_and_jaw_support_rescoring_samples_predicted_width_endpoints() -> None:
    support = np.ones((21, 21), dtype=np.float32)
    support[10, 15] = 0.25
    quality = np.full((21, 21), 0.8, dtype=np.float32)
    transform = CropTransform(0, 0, 21, 21, 21, 21, 21, 21)
    candidate = Grasp4DoF(
        center_x=10,
        center_y=10,
        angle_deg=0,
        width_px=10,
        height_px=20,
        score=0.8,
        candidate_id="jaw-test",
        metadata={"source_row": 10, "source_column": 10},
    )

    rescored = rescore_candidates_with_mask_support(
        [candidate],
        network_quality_map=quality,
        probability_gate=support,
        transform=transform,
        center_gate_exponent=1.0,
        jaw_gate_exponent=2.0,
    )[0]

    assert rescored.metadata["jaw_endpoint_support"] == pytest.approx([1.0, 0.25])
    assert rescored.metadata["jaw_mask_support"] == pytest.approx(0.5)
    assert rescored.metadata["center_mask_support"] == pytest.approx(1.0)
    assert rescored.score == pytest.approx(0.8 * 0.5**2)


@pytest.mark.parametrize(
    ("backend_type", "config_type", "channels"),
    (
        (GRConvNetBackend, GRConvNetConfig, 4),
        (GGCNN2Backend, GGCNN2Config, 1),
    ),
)
def test_empty_mask_and_invalid_target_depth_are_traceable(
    backend_type, config_type, channels: int
) -> None:
    backend = backend_type(
        _config(config_type), model=SyntheticFourMapModel(channels)
    )

    empty = backend.predict(_sample(empty_mask=True))
    invalid = backend.predict(_sample(invalid_target_depth=True))

    assert empty.empty_reason == "empty_mask"
    assert invalid.empty_reason == "invalid_target_depth"
    assert empty.candidates == invalid.candidates == ()


def test_main_protocol_rejects_gt_but_explicit_oracle_config_accepts_it() -> None:
    base = _sample()
    oracle_sample = BackendSample(
        sample_id="oracle",
        rgb=base.rgb,
        depth_m=base.depth_m,
        predicted_mask=None,
        probability_map=None,
        mask_source="gt_mask_oracle",
        oracle_mask=base.predicted_mask,
    )
    main = GGCNN2Backend(
        _config(GGCNN2Config, allow_oracle=False),
        model=SyntheticFourMapModel(1),
    )
    oracle = GGCNN2Backend(
        _config(GGCNN2Config, allow_oracle=True),
        model=SyntheticFourMapModel(1),
    )

    with pytest.raises(GroundTruthAccessError, match="forbidden"):
        main.predict(oracle_sample)
    prediction = oracle.predict(oracle_sample)
    assert prediction.conditioning_variant.endswith("_gt_mask_oracle")
    assert prediction.metadata["mask_source"] == "gt_mask_oracle"


def test_grconvnet_hash_is_checked_before_unsafe_pickle_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "untrusted.pkl"
    checkpoint.write_bytes(b"not an official pickle")
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before hash verification")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint_sha256_mismatch"):
        load_trusted_grconvnet_full_pickle(
            checkpoint, expected_sha256="0" * 64, device="cpu"
        )
    assert not called


def test_ggcnn2_state_dict_loading_is_strict(tmp_path: Path) -> None:
    state = OfficialGGCNN2().state_dict()
    state.pop("width_output.bias")
    checkpoint = tmp_path / "incomplete.pt"
    torch.save(state, checkpoint)

    with pytest.raises(RuntimeError, match="Missing key"):
        load_ggcnn2_state_dict(
            checkpoint, expected_sha256=sha256_file(checkpoint), device="cpu"
        )


def test_postprocessing_matches_official_gaussian_angle_and_width_semantics() -> None:
    q = torch.zeros((1, 1, 9, 9), dtype=torch.float32)
    q[0, 0, 4, 4] = 1.0
    angle = np.zeros((9, 9), dtype=np.float32)
    angle[:, 5:] = np.deg2rad(40.0)
    cos = torch.from_numpy(np.cos(2 * angle))[None, None]
    sin = torch.from_numpy(np.sin(2 * angle))[None, None]
    width = torch.full_like(q, 0.2)

    quality, cos_out, sin_out, width_out = official_gaussian_post_process(
        (q, cos, sin, width), expected_spatial_shape=(9, 9)
    )
    expected_angle = gaussian(angle, 2.0, preserve_range=True)

    assert quality == pytest.approx(gaussian(q.numpy()[0, 0], 2.0, preserve_range=True))
    assert cos_out == pytest.approx(np.cos(2 * expected_angle))
    assert sin_out == pytest.approx(np.sin(2 * expected_angle))
    assert width_out == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("backend_type", "config_type", "expected_sha"),
    (
        (GRConvNetBackend, GRConvNetConfig, GRCONVNET_CHECKPOINT_SHA256),
        (GGCNN2Backend, GGCNN2Config, GGCNN2_CHECKPOINT_SHA256),
    ),
)
def test_real_official_weight_cpu_forward_smoke(
    backend_type, config_type, expected_sha: str
) -> None:
    backend = backend_type(
        _config(config_type, device="cpu", quality_threshold=1e9)
    )

    prediction = backend.predict(_sample(sample_id=f"cpu-{backend.backend_name}"))

    assert prediction.device == "cpu"
    assert prediction.empty_reason == "no_candidate_generated"
    assert prediction.metadata["checkpoint_sha256"] == expected_sha
    assert prediction.metadata["quality_map"]["shape"] == [32, 32]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable on this host"
)
@pytest.mark.parametrize(
    ("backend_type", "config_type"),
    (
        (GRConvNetBackend, GRConvNetConfig),
        (GGCNN2Backend, GGCNN2Config),
    ),
)
def test_real_official_weight_mps_forward_smoke(backend_type, config_type) -> None:
    backend = backend_type(
        _config(config_type, device="mps", quality_threshold=1e9)
    )

    prediction = backend.predict(_sample(sample_id=f"mps-{backend.backend_name}"))

    assert prediction.device == "mps"
    assert prediction.empty_reason == "no_candidate_generated"


def test_device_policy_never_selects_cuda() -> None:
    assert select_device("cpu").type == "cpu"
    assert select_device("auto").type in {"cpu", "mps"}
    with pytest.raises(ValueError, match="only cpu, mps"):
        select_device("cuda")
