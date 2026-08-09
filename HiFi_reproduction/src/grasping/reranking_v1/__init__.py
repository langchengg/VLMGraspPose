"""Leakage-safe candidate re-ranking utilities.

This namespace is intentionally separate from the frozen HiFi-CS, Dex-Net,
and GQ-CNN outputs.  Modules may read those artifacts, but never modify them.
"""

from .identity import (
    CANDIDATE_POSE_FIELDS,
    assert_candidate_identity_invariant,
    candidate_identity_sha256,
    sha256_file,
    stable_sample_id,
)
from .labels import (
    CandidateLabel,
    evaluate_candidate_label,
    is_positive_pair,
    periodic_angle_difference_deg,
)

__all__ = [
    "CANDIDATE_POSE_FIELDS",
    "CandidateLabel",
    "assert_candidate_identity_invariant",
    "candidate_identity_sha256",
    "evaluate_candidate_label",
    "is_positive_pair",
    "periodic_angle_difference_deg",
    "sha256_file",
    "stable_sample_id",
]
