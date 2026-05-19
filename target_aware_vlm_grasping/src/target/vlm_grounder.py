from __future__ import annotations

import numpy as np

from target.base_grounder import BaseTargetGrounder
from target.grounding_backends import build_vlm_backend
from utils.data_types import DatasetSample, TargetRegion


class VLMTargetGrounder(BaseTargetGrounder):
    """Language + RGB target grounding with optional VLM backends."""

    def __init__(
        self,
        backend_name: str = "florence2",
        backend_config: dict | None = None,
        fallback_to_oracle: bool = False,
    ):
        self.backend_name = backend_name
        self.backend_config = backend_config or {}
        self.fallback_to_oracle = fallback_to_oracle
        self._backend = None

    def predict(self, sample: DatasetSample, rgb_image: np.ndarray) -> TargetRegion:
        try:
            if self._backend is None:
                self._backend = build_vlm_backend(self.backend_name, self.backend_config)
            target = self._backend.ground(rgb_image, sample.command, target_id=sample.target_id)
            target.metadata["vlm_predicted_label"] = target.label
            if sample.target_label:
                target.label = sample.target_label
            target.command = sample.command
            target.target_source = "vlm"
            target.metadata["target_bbox_pred"] = target.bbox
            target.metadata["target_bbox_gt"] = sample.target_bbox_gt
            target.metadata["target_mask_source"] = "vlm_bbox"
            return target
        except Exception as exc:
            if not self.fallback_to_oracle:
                raise RuntimeError(
                    f"VLM target mode failed with backend '{self.backend_name}': {type(exc).__name__}: {exc}. "
                    "Use --target-source oracle or install/cache the selected VLM backend."
                ) from exc
            from target.oracle_grounder import OracleTargetGrounder
            target = OracleTargetGrounder().predict(sample, rgb_image)
            target.metadata["vlm_error"] = f"{type(exc).__name__}: {exc}"
            target.metadata["target_source_requested"] = "vlm"
            return target
