"""Official pinned GR-ConvNet RGB-D inference adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..common import (
    GraspPrediction,
    decode_quality_maps,
    non_maximum_suppression,
)

from .base import (
    BackendInputError,
    BackendSample,
    NetworkBackendConfig,
    conditioning_label,
    prepare_conditioned_input,
)
from .network_utils import (
    GRCONVNET_CHECKPOINT,
    GRCONVNET_CHECKPOINT_SHA256,
    freeze_for_inference,
    gated_quality_map,
    load_trusted_grconvnet_full_pickle,
    map_metadata,
    official_gaussian_post_process,
    rescore_candidates_with_mask_support,
    select_device,
    synchronize_device,
    validate_model_layout,
)


@dataclass(frozen=True, slots=True)
class GRConvNetConfig(NetworkBackendConfig):
    checkpoint_path: Path = GRCONVNET_CHECKPOINT
    checkpoint_sha256: str = GRCONVNET_CHECKPOINT_SHA256
    input_channels: int = 4

    def __post_init__(self) -> None:
        NetworkBackendConfig.__post_init__(self)
        if self.input_channels not in (1, 4):
            raise ValueError("GR-ConvNet input_channels must be 1 (depth) or 4 (depth+RGB)")


class GRConvNetBackend:
    """Frozen 4-channel official model behind the uniform predict contract."""

    backend_name = "grconvnet"

    def __init__(
        self,
        config: GRConvNetConfig = GRConvNetConfig(),
        *,
        model: nn.Module | None = None,
    ) -> None:
        self.config = config
        self.device = select_device(config.device)
        if model is None:
            self.model, self.checkpoint_sha256 = load_trusted_grconvnet_full_pickle(
                config.checkpoint_path,
                expected_sha256=config.checkpoint_sha256,
                device=self.device,
                expected_input_channels=config.input_channels,
            )
            self.checkpoint_path = str(Path(config.checkpoint_path).resolve())
        else:
            validate_model_layout(model, expected_input_channels=config.input_channels)
            self.model = freeze_for_inference(model, self.device)
            self.checkpoint_sha256 = "injected_unverified"
            self.checkpoint_path = "<injected>"

    def _empty(
        self,
        sample: BackendSample,
        reason: str,
        runtime_seconds: float,
        **metadata: object,
    ) -> GraspPrediction:
        return GraspPrediction(
            sample_id=sample.sample_id,
            backend=self.backend_name,
            conditioning_variant=conditioning_label(
                sample, self.config.conditioning_variant
            ),
            raw_candidate_count=0,
            nms_candidate_count=0,
            candidates=(),
            top1=None,
            top5=(),
            empty_reason=reason,
            runtime_seconds=runtime_seconds,
            device=self.device.type,
            metadata={
                "mask_source": sample.mask_source,
                "failure_stage": reason,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                **metadata,
            },
        )

    def predict(self, sample: BackendSample) -> GraspPrediction:
        started = time.perf_counter()
        try:
            conditioned = prepare_conditioned_input(sample, self.config)
        except BackendInputError as error:
            return self._empty(sample, error.reason, time.perf_counter() - started)

        # Pinned GraspDataset order is [depth, R, G, B], not RGB-D.
        model_input = (
            conditioned.depth_chw
            if self.config.input_channels == 1
            else np.concatenate((conditioned.depth_chw, conditioned.rgb_chw), axis=0)
        )
        if model_input.shape != (
            self.config.input_channels,
            self.config.input_size,
            self.config.input_size,
        ):
            return self._empty(
                sample,
                "invalid_model_input_shape",
                time.perf_counter() - started,
                observed_shape=list(model_input.shape),
            )
        tensor = torch.from_numpy(np.ascontiguousarray(model_input)).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        try:
            with torch.inference_mode():
                outputs = self.model(tensor)
            synchronize_device(self.device)
            network_quality, cos_map, sin_map, width = official_gaussian_post_process(
                outputs,
                expected_spatial_shape=(self.config.input_size, self.config.input_size),
            )
            quality = gated_quality_map(
                network_quality,
                probability_gate=conditioned.gate_map,
                valid_depth_gate=conditioned.valid_depth_map,
                center_gate_exponent=self.config.center_gate_exponent,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            return self._empty(
                sample,
                "model_forward_failure",
                time.perf_counter() - started,
                error=f"{type(error).__name__}:{error}",
            )

        raw = decode_quality_maps(
            quality,
            cos_map,
            sin_map,
            width,
            sample_id=sample.sample_id,
            backend=self.backend_name,
            transform=conditioned.transform,
            quality_threshold=self.config.quality_threshold,
            min_peak_distance_px=self.config.min_peak_distance_px,
            max_peaks=self.config.max_raw_candidates,
            width_scale=1.0,
            fixed_height_px=self.config.fixed_height_px,
        )
        raw = [
            candidate
            for candidate in rescore_candidates_with_mask_support(
                raw,
                network_quality_map=network_quality,
                probability_gate=conditioned.gate_map,
                transform=conditioned.transform,
                center_gate_exponent=self.config.center_gate_exponent,
                jaw_gate_exponent=self.config.jaw_gate_exponent,
            )
            if candidate.score > self.config.quality_threshold
        ]
        kept = non_maximum_suppression(raw, self.config.nms)
        runtime = time.perf_counter() - started
        if not kept:
            return self._empty(
                sample,
                "no_candidate_generated",
                runtime,
                quality_map=map_metadata(quality),
                angle_map=map_metadata(np.arctan2(sin_map, cos_map) * 0.5),
                width_map=map_metadata(width),
            )
        top5 = tuple(kept[:5])
        prediction = GraspPrediction(
            sample_id=sample.sample_id,
            backend=self.backend_name,
            conditioning_variant=conditioning_label(
                sample, self.config.conditioning_variant
            ),
            raw_candidate_count=len(raw),
            nms_candidate_count=len(kept),
            candidates=tuple(kept),
            top1=top5[0],
            top5=top5,
            runtime_seconds=runtime,
            device=self.device.type,
            metadata={},
        )
        return replace(
            prediction,
            metadata={
                "mask_source": sample.mask_source,
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "input_channels": self.config.input_channels,
                "quality_map": map_metadata(quality),
                "angle_map": map_metadata(np.arctan2(sin_map, cos_map) * 0.5),
                "width_map": map_metadata(width),
                "mask_area_px": conditioned.mask_area_px,
                "valid_depth_fraction": conditioned.valid_depth_fraction,
            },
        )
