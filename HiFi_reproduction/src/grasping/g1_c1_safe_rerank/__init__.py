"""Frozen-candidate G1/C1 safe re-ranking research pipeline.

The package never generates or refines grasp geometry.  It consumes the
audited G1/C1 post-NMS candidates, learns ranking/calibration models from the
development split, and protects the original backend Top-1 with a conservative
switch gate.
"""

from .contracts import (
    BACKENDS,
    FORBIDDEN_INFERENCE_TOKENS,
    candidate_identity_sha256,
    validate_frozen_candidates,
)

__all__ = [
    "BACKENDS",
    "FORBIDDEN_INFERENCE_TOKENS",
    "candidate_identity_sha256",
    "validate_frozen_candidates",
]
