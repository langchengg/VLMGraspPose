"""Non-moving executor used by default for real-robot trial preparation."""

from __future__ import annotations

from typing import Any, Mapping

from .robot_executor import ExecutionRequest, ExecutionResult, RobotExecutor


class DryRunExecutor(RobotExecutor):
    """Validate requests while guaranteeing that no hardware command is sent."""

    @property
    def mode(self) -> str:
        return "dry_run"

    def preflight(self) -> Mapping[str, Any]:
        return {
            "status": "ok",
            "executor_mode": self.mode,
            "hardware_enabled": False,
            "physical_motion_possible": False,
            "checks": {
                "ros": False,
                "moveit": False,
                "robot_driver": False,
                "base_camera_transform": False,
                "workspace_collision_model": False,
            },
            "message": "Dry-run validation only; no physical motion can occur.",
        }

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        details = {
            "sample_id": request.sample_id,
            "instruction": request.instruction,
            "T_base_grasp": request.T_base_grasp.tolist(),
            "gripper_width_m": request.gripper_width_m,
            "metadata": dict(request.metadata),
        }
        return ExecutionResult(
            trial_id=request.trial_id,
            executor_mode=self.mode,
            status="dry_run_validated",
            physical_execution_attempted=False,
            physical_success=None,
            message="Request validated; no robot command was issued.",
            details=details,
        )

