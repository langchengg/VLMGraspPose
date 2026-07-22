"""Abstract, safety-first boundary for physical grasp execution.

This repository intentionally ships no hardware implementation.  A concrete
executor must be supplied by a separately reviewed robot integration and must
explicitly report whether physical motion occurred.  Offline inference and
simulation are never accepted as evidence of a real-robot execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


class RobotSafetyError(RuntimeError):
    """Raised when a request cannot pass the physical execution safety gate."""


@dataclass(frozen=True)
class ExecutionRequest:
    """Immutable grasp command proposed to a robot executor."""

    trial_id: str
    sample_id: str
    instruction: str
    T_base_grasp: np.ndarray
    gripper_width_m: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id.strip() or not self.sample_id.strip():
            raise ValueError("trial_id and sample_id must be non-empty")
        transform = np.asarray(self.T_base_grasp, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("T_base_grasp must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("T_base_grasp must be a homogeneous transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("T_base_grasp rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("T_base_grasp rotation must be right handed")
        width = float(self.gripper_width_m)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("gripper_width_m must be finite and positive")
        object.__setattr__(self, "T_base_grasp", transform.copy())
        object.__setattr__(self, "gripper_width_m", width)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome returned by an executor without implying physical success."""

    trial_id: str
    executor_mode: str
    status: str
    physical_execution_attempted: bool
    physical_success: bool | None
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.physical_execution_attempted and self.physical_success is not None:
            raise ValueError("physical_success must be null when no physical execution occurred")
        object.__setattr__(self, "details", dict(self.details))

    def to_record(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "executor_mode": self.executor_mode,
            "status": self.status,
            "physical_execution_attempted": self.physical_execution_attempted,
            "physical_success": self.physical_success,
            "message": self.message,
            "details": dict(self.details),
        }


class RobotExecutor(ABC):
    """Interface that keeps planning separate from reviewed robot control."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return a stable executor mode such as ``dry_run`` or ``hardware``."""

    @property
    def hardware_enabled(self) -> bool:
        """Whether this executor is authorized to command physical hardware."""

        return False

    @abstractmethod
    def preflight(self) -> Mapping[str, Any]:
        """Return structured safety and capability checks."""

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Evaluate or execute one immutable request."""

