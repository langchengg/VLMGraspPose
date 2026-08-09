"""Frozen route, feature, and candidate contracts."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Iterable


ROUTES = ("CROG", "G1", "C1")


class EvidenceTrack(StrEnum):
    NATIVE = "T1_native"
    MATCHED_COMMON = "T2_matched_common"
    TRI_BACKEND = "T3_tri_backend"
    ROUTE_RICH = "T4_route_rich_secondary"
    CROSS_ROUTE = "T5_cross_route"


CANDIDATE_IDENTITY_COLUMNS = (
    "sample_id",
    "route",
    "candidate_id",
    "native_rank",
    "native_score",
    "cx_px",
    "cy_px",
    "theta_deg",
    "width_px",
    "height_px",
)


_FORBIDDEN_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^gt_",
        r"ground_truth",
        r"candidate_success",
        r"jacquard[_-]?margin",
        r"pool[_-]?solvable",
        r"native[_-]?correct",
        r"challenger[_-]?correct",
        r"selected[_-]?correct",
        r"first[_-]?positive",
        r"recovered",
        r"harmful",
        r"best_same_gt",
        r"best_rectangle",
        r"best_angle",
        r"matched_gt",
        r"oracle",
        r"j_at_",
        r"target_object",
        r"category_label",
        r"object[_-]?category",
        r"scene_graph",
        r"^sample_id$",
        r"^scene_id$",
        r"^candidate_id$",
        r"^source_.*_path$",
        r".*_path$",
    )
)


def forbidden_model_columns(columns: Iterable[str]) -> list[str]:
    """Return identifiers, paths, labels, and GT-derived columns in a feature list."""

    return sorted(
        {
            str(column)
            for column in columns
            if any(pattern.search(str(column)) for pattern in _FORBIDDEN_PATTERNS)
        }
    )


def assert_model_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(str(column) for column in columns)
    forbidden = forbidden_model_columns(selected)
    if forbidden:
        raise ValueError(f"forbidden model feature columns: {forbidden}")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate model feature columns")
    return selected
