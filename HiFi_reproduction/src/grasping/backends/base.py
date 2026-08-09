"""GT-safe data contract and shared configuration for 4-DoF backends."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from ..common import GraspPrediction, NMSConfig

from .conditioning import ConditionedInput, ConditioningVariant, condition_rgbd


MaskSource = Literal["predicted", "gt_mask_oracle"]


class GroundTruthAccessError(ValueError):
    """Raised when the deployment protocol is asked to consume a GT mask."""


class BackendInputError(ValueError):
    """A traceable per-sample input failure rather than a model failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


@dataclass(frozen=True, slots=True)
class BackendSample:
    """Numpy-only inference sample with an explicit mask provenance."""

    sample_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    predicted_mask: np.ndarray | None
    probability_map: np.ndarray | None
    mask_source: MaskSource = "predicted"
    oracle_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must have the same H, W as rgb")
        for name in ("predicted_mask", "oracle_mask"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != depth.shape:
                raise ValueError(f"{name} must have the same H, W as depth_m")
        probability = self.probability_map
        if probability is not None and np.asarray(probability).ndim != 2:
            raise ValueError("probability_map must be 2D")
        if self.mask_source not in ("predicted", "gt_mask_oracle"):
            raise ValueError(f"unsupported mask_source: {self.mask_source}")
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class NetworkBackendConfig:
    """Validation-selected inference settings shared by both networks."""

    conditioning_variant: ConditioningVariant = "hard_mask"
    input_size: int = 300
    dilation_fraction: float = 0.15
    minimum_crop_side_px: int = 64
    minimum_mask_area_px: int = 1
    quality_threshold: float = 0.2
    min_peak_distance_px: int = 20
    max_raw_candidates: int = 100
    fixed_height_px: float = 20.0
    center_gate_exponent: float = 1.0
    jaw_gate_exponent: float = 0.0
    allow_oracle: bool = False
    device: str = "auto"
    nms: NMSConfig = field(default_factory=NMSConfig)

    def __post_init__(self) -> None:
        if self.conditioning_variant not in ("hard_mask", "dilated_crop"):
            raise ValueError("unsupported conditioning_variant")
        if min(
            self.input_size,
            self.minimum_crop_side_px,
            self.minimum_mask_area_px,
            self.max_raw_candidates,
        ) <= 0:
            raise ValueError("size and count settings must be positive")
        if self.min_peak_distance_px < 0:
            raise ValueError("min_peak_distance_px must be non-negative")
        if not 0.0 <= float(self.dilation_fraction) <= 1.0:
            raise ValueError("dilation_fraction must be in [0, 1]")
        if not math.isfinite(float(self.quality_threshold)):
            raise ValueError("quality_threshold must be finite")
        if not math.isfinite(float(self.fixed_height_px)) or self.fixed_height_px <= 0:
            raise ValueError("fixed_height_px must be finite and positive")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in (self.center_gate_exponent, self.jaw_gate_exponent)
        ):
            raise ValueError("gate exponents must be finite and non-negative")


@runtime_checkable
class GraspBackend(Protocol):
    """Uniform backend interface used by experiment runners."""

    def predict(self, sample: BackendSample) -> GraspPrediction:
        ...


def conditioning_label(sample: BackendSample, variant: str) -> str:
    if sample.mask_source == "predicted":
        return variant
    return f"{variant}_gt_mask_oracle"


def prepare_conditioned_input(
    sample: BackendSample, config: NetworkBackendConfig
) -> ConditionedInput:
    """Resolve mask provenance, rejecting GT unless an oracle run is explicit."""

    if sample.mask_source == "predicted":
        if sample.predicted_mask is None:
            raise BackendInputError("missing_predicted_mask")
        if sample.probability_map is None:
            raise BackendInputError("missing_probability_map")
        mask = np.asarray(sample.predicted_mask).astype(bool)
        probability = np.asarray(sample.probability_map, dtype=np.float32)
    else:
        if not config.allow_oracle:
            raise GroundTruthAccessError(
                "gt_mask_oracle is forbidden by the main predicted-mask protocol"
            )
        if sample.oracle_mask is None:
            raise BackendInputError("missing_oracle_mask")
        mask = np.asarray(sample.oracle_mask).astype(bool)
        probability = mask.astype(np.float32)

    area = int(np.count_nonzero(mask))
    if area == 0:
        raise BackendInputError("empty_mask")
    if area < config.minimum_mask_area_px:
        raise BackendInputError("mask_too_small")
    try:
        return condition_rgbd(
            rgb=sample.rgb,
            depth_m=sample.depth_m,
            binary_mask=mask,
            probability=probability,
            variant=config.conditioning_variant,
            output_size=config.input_size,
            dilation_fraction=config.dilation_fraction,
            minimum_side_px=config.minimum_crop_side_px,
        )
    except ValueError as error:
        reason = str(error)
        if reason in {
            "empty_mask",
            "invalid_depth",
            "invalid_target_depth",
        }:
            raise BackendInputError(reason) from error
        raise BackendInputError(f"invalid_conditioning_input:{reason}") from error
