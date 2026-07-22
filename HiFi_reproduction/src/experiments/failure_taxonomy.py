"""Explicit state semantics for offline VGN experiments.

The taxonomy distinguishes scientific terminal outcomes from data-quality
failures and retryable infrastructure failures.  This prevents a resume loop
from silently retrying deterministic input failures forever.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatusDisposition:
    """How a per-sample status participates in scheduling and metrics."""

    category: str
    terminal: bool
    retryable: bool
    candidate_outcome: bool = False


_SCIENTIFIC_OUTCOMES = {
    "ok",
    "no_official_grasp",
    "no_target_grasp",
}

_DETERMINISTIC_INPUT_FAILURES = {
    "ambiguous_depth_unit",
    "candidate_decode_error",
    "depth_shape_error",
    "duplicate_manifest_sample",
    "empty_mask",
    "empty_pred_mask",
    "gt_oracle_ambiguous",
    "gt_oracle_unavailable",
    "insufficient_valid_depth",
    "insufficient_masked_depth",
    "intrinsics_fit_failed",
    "invalid_intrinsics",
    "manifest_sample_not_ready",
    "manifest_schema_error",
    "mask_shape_error",
    "mask_depth_shape_error",
    "mask_too_large",
    "mask_too_small",
    "missing_camera_intrinsics",
    "missing_depth",
    "missing_metadata",
    "missing_pcd",
    "missing_pred_mask",
    "missing_rgb",
    "missing_sample_file",
    "oracle_mask_forbidden",
    "projection_error",
    "rgb_depth_shape_error",
    "source_checksum_mismatch",
    "source_path_outside_ocid_root",
    "support_plane_failed",
    "task_frame_invalid",
    "tsdf_empty",
}

_RETRYABLE_FAILURES = {
    "database_busy",
    "io_error",
    "out_of_memory",
    "processing_error",
    "render_error",
    "vgn_checkpoint_error",
    "vgn_inference_error",
    "vgn_inference_failed",
    "write_error",
    "worker_lost",
}

_OPERATIONAL = {"pending", "running"}


def classify_status(status: str) -> StatusDisposition:
    """Return the scheduling/aggregation semantics of ``status``.

    Unknown non-empty statuses are conservatively retryable rather than being
    counted as completed scientific outcomes.  Callers should still persist
    the original status so unexpected failures remain auditable.
    """

    normalized = str(status).strip().lower()
    if not normalized:
        raise ValueError("status must be a non-empty string")
    if normalized in _SCIENTIFIC_OUTCOMES:
        return StatusDisposition(
            category="scientific_outcome",
            terminal=True,
            retryable=False,
            candidate_outcome=True,
        )
    if normalized in _DETERMINISTIC_INPUT_FAILURES:
        return StatusDisposition(
            category="deterministic_input_failure",
            terminal=True,
            retryable=False,
        )
    if normalized in _RETRYABLE_FAILURES:
        return StatusDisposition(
            category="retryable_infrastructure_failure",
            # The current attempt has a reportable terminal outcome.  The
            # independent retryable bit permits an explicit requeue without
            # silently excluding this row from the current-run denominator.
            terminal=True,
            retryable=True,
        )
    if normalized in _OPERATIONAL:
        return StatusDisposition(
            category="operational",
            terminal=False,
            retryable=False,
        )
    return StatusDisposition(
        category="unexpected_failure",
        terminal=False,
        retryable=True,
    )


def is_terminal(status: str) -> bool:
    """Whether the status needs no further scheduling work."""

    return classify_status(status).terminal


def is_retryable(status: str) -> bool:
    """Whether a failed sample may safely return to the pending queue."""

    return classify_status(status).retryable


def is_candidate_outcome(status: str) -> bool:
    """Whether VGN candidate generation was actually reached."""

    return classify_status(status).candidate_outcome


def known_statuses() -> frozenset[str]:
    """All explicitly classified statuses (useful for validation/reporting)."""

    return frozenset(
        _SCIENTIFIC_OUTCOMES
        | _DETERMINISTIC_INPUT_FAILURES
        | _RETRYABLE_FAILURES
        | _OPERATIONAL
    )
