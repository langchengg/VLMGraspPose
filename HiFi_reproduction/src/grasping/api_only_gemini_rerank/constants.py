"""Frozen identities and schemas for the API-only experiment."""

from __future__ import annotations


EXACT_MODEL_IDS = (
    "gemini-robotics-er-2-preview",
    "gemini-3.6-flash",
)
PROTOCOLS = (
    "P1_API_DIRECT_FULL_LIST",
    "P2_API_BASELINE_AWARE",
    "P3_API_BASELINE_AWARE_CONFIDENCE",
    "P4_API_SELF_CONSISTENT",
    "P5_API_CROSS_MODEL_CONSENSUS",
)
EVIDENCE_VARIANTS = (
    "E0_RGB_ONLY",
    "E1_RGB_MASK",
    "E2_RGBD_GEOMETRY_SCORE_BLIND",
    "E3_RGBD_GEOMETRY_SCORE_AWARE",
)
SEED = 20260805
SOURCE_INFERENCE_COLUMNS = (
    "method",
    "sample_id",
    "scene_id",
    "candidate_id",
    "rank",
    "center_x",
    "center_y",
    "angle_deg",
    "width_px",
    "height_px",
    "score",
    "candidate_metadata_json",
)
SOURCE_LABEL_COLUMNS = (
    "sample_id",
    "candidate_id",
    "candidate_success",
    "best_rectangle_iou",
    "best_angle_difference_deg",
)
CANDIDATE_COLUMNS = (
    "backend",
    "split",
    "sample_id",
    "scene_id",
    "candidate_id",
    "original_rank",
    "original_score",
    "center_x",
    "center_y",
    "angle_deg",
    "width_px",
    "height_px",
    "center_mask_support",
    "jaw_mask_support",
    "candidate_geometry_sha256",
    "candidate_set_sha256",
)
FORBIDDEN_PAYLOAD_TOKENS = (
    "candidate_success",
    "best_rectangle_iou",
    "best_angle_difference",
    "best_gt_index",
    "gt_grasp",
    "gt_mask",
    "target_object_id",
    "j_at_1",
    "j_at_5",
    "oracle",
    "baseline_correct",
    "recoverable",
    "unrecoverable",
    "recovered",
    "harmful",
    "first_valid_rank",
    "pairwise_json",
)
