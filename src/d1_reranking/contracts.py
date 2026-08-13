"""Frozen protocol constants for the D1 retrospective extension."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re

import pyarrow.parquet as pq


D1_ROUTE = "D1"
RECTANGLE_HEIGHT_PX = 20.0
EXPECTED_UNIFIED_FINAL_LOCK_SHA256 = (
    "4b52eac6494e59a0f902792b794c3cca02824bf7a36bed4699b569986383f793"
)
EXPECTED_EVALUATOR_SHA256 = (
    "f5155590b0b8d9f0748ad463688edfe6ca595d8ab7239ef5e29a5c595aeef301"
)

POOL_LIMITS: dict[str, int | None] = {
    "top5": 5,
    "top10": 10,
    "allnms": None,
}

CANDIDATE_ID_COLUMNS = ("sample_id", "candidate_id")
CANDIDATE_GEOMETRY_COLUMNS = (
    "cx_px",
    "cy_px",
    "center_depth_m",
    "theta_deg",
    "width_px",
    "height_px",
)
CANDIDATE_REQUIRED_COLUMNS = (
    "sample_id",
    "sample_index",
    "frame_id",
    "scene_id",
    "route",
    "split",
    "candidate_id",
    "native_rank",
    "native_score",
    *CANDIDATE_GEOMETRY_COLUMNS,
    "source_angle_rad",
    "width_m",
    "endpoints_uv_json",
    "center_camera_xyz_m_json",
    "pose_matrix_json",
    "candidate_identity_sha256",
    "candidate_geometry_sha256",
)

FORBIDDEN_CANDIDATE_OUTPUT_COLUMNS = (
    "candidate_success",
    "candidate_positive",
    "best_gt_id",
    "candidate_gt_iou",
    "candidate_gt_angle_error_deg",
    "jacquard_margin",
    "matched_gt_index",
    "pool_has_positive",
    "first_positive_rank",
)

_FORBIDDEN_TEST_SCHEMA_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"candidate[_-]?(success|positive)",
        r"(^|[_-])gt($|[_-])",
        r"ground[_-]?truth",
        r"(^|[_-])j[_-]?at[_-]?\d+($|[_-])",
        r"jacquard[_-]?margin",
        r"(native|challenger|selected)[_-]?correct",
        r"best[_-]?(rectangle|angle|same[_-]?gt)",
        r"matched[_-]?gt",
        r"first[_-]?positive",
        r"pool[_-]?solvable",
        r"oracle",
        r"recovered",
        r"harmful",
        r"diagnostic[_-]?(iou|angle|gt)",
        r"target[_-]?object",
        r"object[_-]?category",
        r"category[_-]?label",
        r"scene[_-]?graph",
    )
)


def forbidden_test_schema_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    """Return every supervision-bearing column from a prelock Test schema."""

    return tuple(
        sorted(
            {
                str(column)
                for column in columns
                if any(
                    pattern.search(str(column))
                    for pattern in _FORBIDDEN_TEST_SCHEMA_PATTERNS
                )
            }
        )
    )


def assert_label_free_parquet_schema(path: str | Path, *, name: str) -> tuple[str, ...]:
    """Inspect only Parquet metadata and reject supervision before row access."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"D1 {name} must be a regular Parquet file: {source}")
    columns = tuple(map(str, pq.ParquetFile(source).schema.names))
    forbidden = forbidden_test_schema_columns(columns)
    if forbidden:
        raise PermissionError(f"D1 {name} contains forbidden Test columns: {forbidden}")
    return columns


class RunState(StrEnum):
    AUDIT = "AUDIT"
    CANDIDATES_FROZEN = "CANDIDATES_FROZEN"
    FEATURES_READY = "FEATURES_READY"
    TRAIN_OOF = "TRAIN_OOF"
    VALIDATION_SCREEN = "VALIDATION_SCREEN"
    PRELOCK_LABEL_FREE = "PRELOCK_LABEL_FREE"
    FORMAL_LOCKED = "FORMAL_LOCKED"
    FORMAL_EXECUTED = "FORMAL_EXECUTED"
    POSTFORMAL = "POSTFORMAL"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


RUN_DIRECTORIES = (
    "00_audit",
    "01_manifests",
    "02_candidates",
    "03_features",
    "04_splits",
    "05_calibration",
    "06_oof",
    "07_validation",
    "08_lock",
    "09_formal_test",
    "10_statistics",
    "11_k_sensitivity",
    "12_feature_ablation",
    "13_four_route_extension",
    "14_failure_analysis",
    "15_figures",
    "16_reports",
    "17_independent_recompute",
    "18_postformal_diagnostics",
    "configs",
    "checkpoints",
    "predictions",
    "tables",
    "logs",
)
