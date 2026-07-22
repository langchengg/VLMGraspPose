"""Safety-gated interfaces for optional physical robot evaluation."""

from .dry_run_executor import DryRunExecutor
from .robot_executor import (
    ExecutionRequest,
    ExecutionResult,
    RobotExecutor,
    RobotSafetyError,
)

__all__ = [
    "DryRunExecutor",
    "ExecutionRequest",
    "ExecutionResult",
    "RobotExecutor",
    "RobotSafetyError",
]
