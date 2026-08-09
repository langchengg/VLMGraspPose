"""Safe checkpoint loading and official dense-map post-processing."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.filters import gaussian
from torch import nn

from ..common import CropTransform, Grasp4DoF
from .vendor_models import OfficialGGCNN2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRCONVNET_VENDOR_ROOT = PROJECT_ROOT / "third_party_src/grconvnet"
GRCONVNET_CHECKPOINT = (
    GRCONVNET_VENDOR_ROOT
    / "trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93"
)
GRCONVNET_CHECKPOINT_SHA256 = (
    "adfb2cbbb8df2708a732e12ddc4db114f3ec399ffb5d403ca75c5b5b9e769171"
)
GGCNN2_CHECKPOINT = (
    PROJECT_ROOT
    / "third_party_src/checkpoints/ggcnn2/ggcnn2_weights_cornell/"
    "epoch_50_cornell_statedict.pt"
)
GGCNN2_CHECKPOINT_SHA256 = (
    "865d538a51d427f7ee84defc99e093bdf51eeb0627c302068037e52188c11d1c"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_checkpoint_hash(path: str | Path, expected_sha256: str) -> str:
    checkpoint = Path(path)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty checkpoint: {checkpoint}")
    observed = sha256_file(checkpoint)
    if observed != str(expected_sha256).lower():
        raise ValueError(
            f"checkpoint_sha256_mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def select_device(requested: str | torch.device = "auto") -> torch.device:
    name = str(requested)
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name not in ("cpu", "mps"):
        raise ValueError("only cpu, mps, or auto devices are supported")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps_requested_but_unavailable")
    return torch.device(name)


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _path_belongs_to(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@contextlib.contextmanager
def controlled_vendor_sys_path(vendor_root: str | Path) -> Iterator[Path]:
    """Temporarily expose one audited vendor root for legacy pickle imports."""

    root = Path(vendor_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing vendor source root: {root}")
    existing = sys.modules.get("inference")
    existing_path = getattr(existing, "__file__", None) if existing else None
    if existing is not None and existing_path and not _path_belongs_to(existing_path, root):
        raise RuntimeError("foreign `inference` module already loaded; refusing pickle load")
    original = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        yield root
    finally:
        sys.path[:] = original


def model_input_channels(model: nn.Module) -> int:
    first = next((module for module in model.modules() if isinstance(module, nn.Conv2d)), None)
    if first is None:
        raise ValueError("model_has_no_conv2d_input")
    return int(first.in_channels)


def validate_model_layout(model: nn.Module, *, expected_input_channels: int) -> None:
    observed = model_input_channels(model)
    if observed != expected_input_channels:
        raise ValueError(
            f"input_channel_mismatch: expected {expected_input_channels}, observed {observed}"
        )
    for name in ("pos_output", "cos_output", "sin_output", "width_output"):
        head = getattr(model, name, None)
        if not isinstance(head, nn.Conv2d) or head.out_channels != 1:
            raise ValueError(f"output_channel_mismatch:{name}")


def freeze_for_inference(model: nn.Module, device: torch.device) -> nn.Module:
    model.requires_grad_(False)
    model.eval()
    return model.to(device=device, dtype=torch.float32)


def load_trusted_grconvnet_full_pickle(
    checkpoint_path: str | Path = GRCONVNET_CHECKPOINT,
    *,
    expected_sha256: str = GRCONVNET_CHECKPOINT_SHA256,
    vendor_root: str | Path = GRCONVNET_VENDOR_ROOT,
    device: str | torch.device = "cpu",
    expected_input_channels: int = 4,
) -> tuple[nn.Module, str]:
    """Load the audited official full pickle only after exact hash verification."""

    checkpoint = Path(checkpoint_path).resolve()
    observed_sha = require_checkpoint_hash(checkpoint, expected_sha256)
    with controlled_vendor_sys_path(vendor_root) as root:
        model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(model, nn.Module):
            raise TypeError("GR-ConvNet checkpoint did not contain an nn.Module")
        class_path = Path(inspect.getfile(type(model))).resolve()
        if not _path_belongs_to(class_path, root):
            raise ValueError("GR-ConvNet class was not loaded from the pinned vendor root")
    validate_model_layout(model, expected_input_channels=expected_input_channels)
    return freeze_for_inference(model, select_device(device)), observed_sha


def load_ggcnn2_state_dict(
    checkpoint_path: str | Path = GGCNN2_CHECKPOINT,
    *,
    expected_sha256: str = GGCNN2_CHECKPOINT_SHA256,
    device: str | torch.device = "cpu",
) -> tuple[OfficialGGCNN2, str]:
    """Strictly load the official one-channel Cornell GG-CNN2 state_dict."""

    checkpoint = Path(checkpoint_path).resolve()
    observed_sha = require_checkpoint_hash(checkpoint, expected_sha256)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("GG-CNN2 checkpoint is not a non-empty state_dict")
    model = OfficialGGCNN2(input_channels=1)
    model.load_state_dict(payload, strict=True)
    validate_model_layout(model, expected_input_channels=1)
    return freeze_for_inference(model, select_device(device)), observed_sha


def validate_four_outputs(
    outputs: Any, *, expected_spatial_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, Sequence) or len(outputs) != 4:
        raise ValueError("model_output_must_contain_four_maps")
    tensors = tuple(outputs)
    expected = (1, 1, int(expected_spatial_shape[0]), int(expected_spatial_shape[1]))
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
            shape = None if not isinstance(tensor, torch.Tensor) else tuple(tensor.shape)
            raise ValueError(f"invalid_model_output_shape:{shape}, expected:{expected}")
    return tensors  # type: ignore[return-value]


def official_gaussian_post_process(
    outputs: Any, *, expected_spatial_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preserve upstream sigma/angle/width semantics, then return decoder maps."""

    q_tensor, cos_tensor, sin_tensor, width_tensor = validate_four_outputs(
        outputs, expected_spatial_shape=expected_spatial_shape
    )
    q_map = q_tensor.detach().float().cpu().numpy()[0, 0]
    cos_map = cos_tensor.detach().float().cpu().numpy()[0, 0]
    sin_map = sin_tensor.detach().float().cpu().numpy()[0, 0]
    width_map = width_tensor.detach().float().cpu().numpy()[0, 0] * 150.0
    angle_radians = np.arctan2(sin_map, cos_map) * 0.5

    quality = gaussian(q_map, 2.0, preserve_range=True).astype(np.float32)
    angle = gaussian(angle_radians, 2.0, preserve_range=True).astype(np.float32)
    width = gaussian(width_map, 1.0, preserve_range=True).astype(np.float32)
    cos_2theta = np.cos(2.0 * angle).astype(np.float32)
    sin_2theta = np.sin(2.0 * angle).astype(np.float32)
    return quality, cos_2theta, sin_2theta, width


def gated_quality_map(
    quality_map: np.ndarray,
    *,
    probability_gate: np.ndarray,
    valid_depth_gate: np.ndarray,
    center_gate_exponent: float = 1.0,
) -> np.ndarray:
    quality = np.asarray(quality_map, dtype=np.float32)
    probability = np.asarray(probability_gate, dtype=np.float32)
    valid = np.asarray(valid_depth_gate, dtype=bool)
    if quality.shape != probability.shape or quality.shape != valid.shape:
        raise ValueError("output_gate_shape_mismatch")
    exponent = float(center_gate_exponent)
    if not np.isfinite(exponent) or exponent < 0.0:
        raise ValueError("center_gate_exponent must be finite and non-negative")
    support = np.clip(probability, 0.0, 1.0)
    gate = np.power(support, exponent) * valid.astype(np.float32)
    return (quality * gate).astype(np.float32, copy=False)


def _bilinear_sample(value: np.ndarray, x: float, y: float) -> float:
    array = np.asarray(value, dtype=np.float32)
    height, width = array.shape
    if not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        return 0.0
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    dx, dy = float(x - x0), float(y - y0)
    return float(
        array[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + array[y0, x1] * dx * (1.0 - dy)
        + array[y1, x0] * (1.0 - dx) * dy
        + array[y1, x1] * dx * dy
    )


def rescore_candidates_with_mask_support(
    candidates: Sequence[Grasp4DoF],
    *,
    network_quality_map: np.ndarray,
    probability_gate: np.ndarray,
    transform: CropTransform,
    center_gate_exponent: float,
    jaw_gate_exponent: float,
) -> list[Grasp4DoF]:
    """Record center/jaw support and apply validation-selectable jaw gating.

    Jaw support is the geometric mean of the two endpoint probabilities at
    half the predicted model-space opening width.  Thus one unsupported jaw
    endpoint strongly penalizes a candidate when ``jaw_gate_exponent > 0``.
    """

    quality = np.asarray(network_quality_map, dtype=np.float32)
    support = np.asarray(probability_gate, dtype=np.float32)
    if quality.ndim != 2 or quality.shape != support.shape:
        raise ValueError("candidate_support_map_shape_mismatch")
    if any(
        not np.isfinite(float(value)) or float(value) < 0.0
        for value in (center_gate_exponent, jaw_gate_exponent)
    ):
        raise ValueError("gate exponents must be finite and non-negative")

    rescored: list[Grasp4DoF] = []
    for candidate in candidates:
        source_row = float(candidate.metadata["source_row"])
        source_column = float(candidate.metadata["source_column"])
        center_support = _bilinear_sample(support, source_column, source_row)
        network_quality = _bilinear_sample(quality, source_column, source_row)
        _, _, model_angle, model_width = transform.native_to_model_pose(
            candidate.center_x,
            candidate.center_y,
            candidate.angle_deg,
            candidate.width_px,
        )
        radians = np.deg2rad(model_angle)
        offset_x = 0.5 * model_width * float(np.cos(radians))
        offset_y = 0.5 * model_width * float(np.sin(radians))
        first_support = _bilinear_sample(
            support, source_column - offset_x, source_row - offset_y
        )
        second_support = _bilinear_sample(
            support, source_column + offset_x, source_row + offset_y
        )
        jaw_support = float(np.sqrt(max(first_support * second_support, 0.0)))
        score = network_quality
        score *= center_support ** float(center_gate_exponent)
        score *= jaw_support ** float(jaw_gate_exponent)
        metadata = {
            **candidate.metadata,
            "network_quality": network_quality,
            "center_mask_support": center_support,
            "jaw_mask_support": jaw_support,
            "jaw_endpoint_support": [first_support, second_support],
        }
        rescored.append(
            Grasp4DoF(
                center_x=candidate.center_x,
                center_y=candidate.center_y,
                angle_deg=candidate.angle_deg,
                width_px=candidate.width_px,
                height_px=candidate.height_px,
                score=score,
                candidate_id=candidate.candidate_id,
                metadata=metadata,
            )
        )
    return rescored


def map_metadata(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    return {
        "shape": [int(size) for size in array.shape],
        "finite_fraction": float(finite.size / array.size) if array.size else 0.0,
        "minimum": None if finite.size == 0 else float(finite.min()),
        "maximum": None if finite.size == 0 else float(finite.max()),
    }
