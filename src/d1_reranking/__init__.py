"""Locked retrospective D1 (Dex-Net/GQ-CNN) reranking extension."""

from .candidates import (
    build_canonical_candidate_split,
    canonicalise_candidate_rows,
    verify_canonical_candidate_frame,
)
from .contracts import D1_ROUTE, POOL_LIMITS, RunState

__all__ = [
    "D1_ROUTE",
    "POOL_LIMITS",
    "RunState",
    "build_canonical_candidate_split",
    "canonicalise_candidate_rows",
    "verify_canonical_candidate_frame",
]
