"""Shared, inference-only data contracts for planar parallel-jaw grasps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Grasp4DoF:
    """One image-plane grasp, where x is column and y is row."""

    center_x: float
    center_y: float
    angle_deg: float
    width_px: float
    score: float
    candidate_id: str = ""
    height_px: float = 20.0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("center_x", "center_y", "angle_deg", "score"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in ("width_px", "height_px"):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "candidate_id", str(self.candidate_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self, *, rank: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "angle_deg": self.angle_deg,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "score": self.score,
        }
        if rank is not None:
            result["rank"] = int(rank)
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True, slots=True)
class GraspPrediction:
    """Serializable Top-1/Top-5 result for one sample and one backend."""

    sample_id: str
    backend: str
    conditioning_variant: str
    raw_candidate_count: int
    nms_candidate_count: int
    top1: Grasp4DoF | None
    top5: tuple[Grasp4DoF, ...]
    candidates: tuple[Grasp4DoF, ...] = ()
    empty_reason: str | None = None
    runtime_seconds: float = 0.0
    device: str = "cpu"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "conditioning_variant", str(self.conditioning_variant))
        object.__setattr__(self, "top5", tuple(self.top5))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.raw_candidate_count < 0 or self.nms_candidate_count < 0:
            raise ValueError("candidate counts must be non-negative")
        if self.nms_candidate_count > self.raw_candidate_count:
            raise ValueError("NMS candidate count cannot exceed raw candidate count")
        if len(self.top5) > 5:
            raise ValueError("top5 cannot contain more than five candidates")
        if self.candidates:
            if len(self.candidates) != self.nms_candidate_count:
                raise ValueError("candidates must contain the complete NMS candidate pool")
            if self.top5 != self.candidates[:5]:
                raise ValueError("top5 must be the first five complete candidates")
        if self.top1 is None and self.top5:
            raise ValueError("top1 cannot be empty when top5 is non-empty")
        if self.top1 is not None and (not self.top5 or self.top5[0] != self.top1):
            raise ValueError("top1 must equal the first top5 candidate")
        if not self.top5 and not self.empty_reason:
            raise ValueError("an empty prediction must record empty_reason")
        runtime = _finite("runtime_seconds", self.runtime_seconds)
        if runtime < 0.0:
            raise ValueError("runtime_seconds must be non-negative")
        object.__setattr__(self, "runtime_seconds", runtime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "backend": self.backend,
            "conditioning_variant": self.conditioning_variant,
            "raw_candidate_count": self.raw_candidate_count,
            "nms_candidate_count": self.nms_candidate_count,
            "top1": None if self.top1 is None else self.top1.to_dict(rank=1),
            "top5": [item.to_dict(rank=index) for index, item in enumerate(self.top5, 1)],
            "candidates": [
                item.to_dict(rank=index)
                for index, item in enumerate(self.candidates, 1)
            ],
            "empty_reason": self.empty_reason,
            "runtime_seconds": self.runtime_seconds,
            "device": self.device,
            "metadata": dict(self.metadata),
        }
