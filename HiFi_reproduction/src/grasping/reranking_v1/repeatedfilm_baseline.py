"""Independent retained repeated-FiLM baseline and Oracle audit.

This module deliberately reads candidate geometry from the frozen raw/NMS
tables and recomputes labels from the official OCID-VLG annotations.  It does
not trust the GT-derived columns in the retained GQ-CNN or evaluation tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import ujson
import cv2
from ruamel.yaml import YAML
from skimage.draw import polygon as draw_polygon

from src.grasping.geometric_ranker import (
    make_candidate_evaluation_rectangle,
    make_ocid_vlg_evaluation_rectangles,
)

from .identity import candidate_identity_sha256, sha256_file, stable_sample_id
from .artifact_contract import identity_payload


SAFE_SCORE_COLUMNS = (
    "pipeline",
    "sample_index",
    "sample_id",
    "question_index",
    "scene_id",
    "candidate_id",
    "gqcnn_rank",
    "gqcnn_q_value",
    "source_candidate_index",
    "center_u_px",
    "center_v_px",
    "center_depth_m",
    "angle_rad",
    "angle_deg_contact_span",
    "width_m",
    "contact_span_width_px",
    "configured_width_px",
    "endpoints_uv_json",
    "contact_points_uv_json",
    "contact_normals_json",
    "center_camera_xyz_m_json",
    "pose_matrix_json",
    "candidate_seed",
)

FORBIDDEN_SCORE_COLUMNS = {
    "candidate_success",
    "best_gt_id",
    "best_gt_index",
    "rectangle_iou",
    "angle_difference_deg",
    "iou_ok",
    "angle_ok",
    "joint_success",
    "failure_mode",
    "evaluator_version",
}
EVALUATION_SHAPE = (480, 640)


def _periodic_angle_difference_deg(
    first_deg: float, second_deg: float
) -> float:
    return abs(
        (
            (float(first_deg) - float(second_deg) + 90.0) % 180.0
        )
        - 90.0
    )


def _rectangle_pixels(
    grasp: Sequence[float],
    shape: tuple[int, int] = EVALUATION_SHAPE,
) -> np.ndarray:
    """Rasterize with the frozen corrected_geometric_v2 x/y convention."""

    x, y, width_px, height_px, angle_deg = map(float, grasp[:5])
    vertices = np.asarray(
        cv2.boxPoints(
            ((x, y), (width_px, height_px), -angle_deg)
        ),
        dtype=np.intp,
    )
    rows, columns = draw_polygon(
        vertices[:, 1], vertices[:, 0], shape=shape
    )
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(
        rows.astype(np.int64) * int(shape[1])
        + columns.astype(np.int64)
    )


def _pixel_iou(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 and second.size == 0:
        return 0.0
    intersection = np.intersect1d(
        first, second, assume_unique=True
    ).size
    union = int(first.size + second.size - intersection)
    return float(intersection / union) if union else 0.0


def _raster_gt_geometry(
    grasp_rectangles: Sequence[Sequence[Sequence[float]]],
    evaluation_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    geometries = make_ocid_vlg_evaluation_rectangles(
        grasp_rectangles, evaluation_config
    )
    output = []
    for gt_index, item in enumerate(geometries):
        grasp = [
            float(item["center_uv"][0]),
            float(item["center_uv"][1]),
            float(item["width_px"]),
            float(item["height_px"]),
            float(math.degrees(item["angle_rad"])),
        ]
        encoded = json.dumps(
            grasp, separators=(",", ":"), allow_nan=False
        )
        output.append(
            {
                "gt_index": gt_index,
                "gt_id": "gt_"
                + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
                "grasp": grasp,
                "pixels": _rectangle_pixels(grasp),
            }
        )
    return output


def _atomic_json(path: Path, payload: Any, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(
    path: Path, frame: pd.DataFrame, tmp_root: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    frame.to_parquet(
        temporary,
        index=False,
        compression="zstd",
        engine="pyarrow",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    temporary = tmp_root / f"{path.name}.{uuid.uuid4().hex}.tmp"
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": float(numerator / denominator) if denominator else None,
    }


def _load_samples(
    manifest_path: Path,
    annotations_path: Path,
    evaluation_config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("test manifest must be a non-empty list")
    annotation_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotation_rows = annotation_payload.get("data")
    if not isinstance(annotation_rows, list):
        raise ValueError("annotation file has no data list")
    annotation_index = {
        int(row["question_index"]): row for row in annotation_rows
    }
    if len(annotation_index) != len(annotation_rows):
        raise ValueError("annotation question_index values are not unique")
    samples: dict[str, dict[str, Any]] = {}
    gt_geometry: dict[str, list[dict[str, Any]]] = {}
    for sample_index, row in enumerate(manifest):
        question_index = int(row["question_index"])
        sample_id = stable_sample_id(str(row["scene_id"]), question_index)
        annotation = annotation_index.get(question_index)
        if (
            annotation is None
            or annotation["image_filename"] != row["scene_id"]
            or annotation["question"] != row["text"]
        ):
            raise ValueError(
                f"manifest/annotation identity mismatch: {question_index}"
            )
        grasps = annotation.get("grasps")
        if not isinstance(grasps, list) or not grasps:
            raise ValueError(f"sample has no GT rectangles: {sample_id}")
        samples[sample_id] = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "question_index": question_index,
            "scene_id": str(row["scene_id"]),
            "query": str(row["text"]),
            "split": "test",
            "raw_candidate_count": 0,
            "mask_validated_candidate_count": 0,
            "nms_candidate_count": 0,
            "scored_candidate_count": 0,
            "empty_mask": False,
            "raw_oracle": False,
            "mask_validated_oracle": False,
            "nms_oracle": False,
            "q_top1": False,
            "q_top5": False,
            "q_top10": False,
            "first_correct_rank": None,
            "reciprocal_rank": 0.0,
        }
        gt_geometry[sample_id] = _raster_gt_geometry(
            grasps, evaluation_config
        )
    if len(samples) != len(manifest):
        raise ValueError("stable sample IDs are not unique")
    return samples, gt_geometry


def _candidate_record(
    *,
    sample_id: str,
    candidate_id: str,
    center_u_px: float,
    center_v_px: float,
    candidate_json: str,
) -> dict[str, Any]:
    payload = ujson.loads(candidate_json)
    return {
        **payload,
        "sample_id": sample_id,
        "candidate_id": candidate_id,
        "center_uv": [float(center_u_px), float(center_v_px)],
        "center_u_px": float(center_u_px),
        "center_v_px": float(center_v_px),
    }


def _label_against_geometry(
    record: Mapping[str, Any],
    ground_truth: Sequence[Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply frozen corrected_geometric_v2 with cached GT raster pixels."""

    predicted_geometry = make_candidate_evaluation_rectangle(
        record, evaluation_config
    )
    predicted = [
        float(predicted_geometry["center_uv"][0]),
        float(predicted_geometry["center_uv"][1]),
        float(predicted_geometry["width_px"]),
        float(predicted_geometry["height_px"]),
        float(math.degrees(predicted_geometry["angle_rad"])),
    ]
    predicted_pixels = _rectangle_pixels(predicted)
    angle_threshold = float(evaluation_config["angle_threshold_deg"])
    iou_threshold = float(evaluation_config["iou_threshold"])
    comparisons: list[dict[str, Any]] = []
    exact_count = 0
    for target in ground_truth:
        target_grasp = target["grasp"]
        angle_error = _periodic_angle_difference_deg(
            predicted[4], float(target_grasp[4])
        )
        iou = _pixel_iou(predicted_pixels, target["pixels"])
        angle_ok = angle_error <= angle_threshold
        strict = bool(angle_ok and iou > iou_threshold)
        legacy = bool(angle_ok and iou >= iou_threshold)
        exact_count += int(
            angle_ok
            and math.isclose(
                iou, iou_threshold, rel_tol=0.0, abs_tol=1e-12
            )
        )
        comparisons.append(
            {
                "gt_id": str(target["gt_id"]),
                "gt_index": int(target["gt_index"]),
                "angle_error": angle_error,
                "iou": iou,
                "angle_ok": angle_ok,
                "strict": strict,
                "legacy": legacy,
            }
        )
    successes = [row for row in comparisons if row["strict"]]
    best = min(
        successes or comparisons,
        key=lambda item: (
            -float(item["iou"]),
            float(item["angle_error"]),
            str(item["gt_id"]),
        ),
    )
    return {
        "candidate_positive": any(row["strict"] for row in comparisons),
        "best_gt_id": str(best["gt_id"]),
        "best_gt_index": int(best["gt_index"]),
        "candidate_gt_iou": float(best["iou"]),
        "candidate_gt_angle_error_deg": float(best["angle_error"]),
        "maximum_rectangle_iou_with_angle_gate": float(
            max(
                (
                    row["iou"] if row["angle_ok"] else 0.0
                    for row in comparisons
                ),
                default=0.0,
            )
        ),
        "legacy_geq_positive": any(
            row["legacy"] for row in comparisons
        ),
        "exact_iou_threshold_pair_count": int(exact_count),
    }


def _identity_from_score_row(row: Mapping[str, Any]) -> str:
    record = {
        "sample_id": str(row["sample_id"]),
        "candidate_id": str(row["candidate_id"]),
        "center_uv": [
            float(row["center_u_px"]),
            float(row["center_v_px"]),
        ],
        "center_depth_m": float(row["center_depth_m"]),
        "center_camera_xyz_m": json.loads(
            str(row["center_camera_xyz_m_json"])
        ),
        "angle_rad": float(row["angle_rad"]),
        "width_m": float(row["width_m"]),
        "width_px": float(row["configured_width_px"]),
        "endpoints_uv": json.loads(str(row["endpoints_uv_json"])),
        "T_camera_grasp_fixed_approach": json.loads(
            str(row["pose_matrix_json"])
        ),
    }
    return candidate_identity_sha256(record)


def _funnel_category(row: Mapping[str, Any]) -> str:
    # A sample with no frozen post-NMS candidate is a valid empty deployment
    # outcome regardless of whether emptiness originated at the mask, sampling,
    # mask filter, or NMS boundary.
    if int(row["nms_candidate_count"]) == 0:
        return "valid_empty"
    if not bool(row["raw_oracle"]):
        return "generation_limited"
    if not bool(row["mask_validated_oracle"]):
        return "mask_filter_loss"
    if not bool(row["nms_oracle"]):
        return "nms_loss"
    if bool(row["q_top1"]):
        return "already_correct"
    if bool(row["q_top5"]):
        return "ranking_loss_top5"
    return "ranking_loss_beyond_top5"


def audit_repeatedfilm_baseline(
    *,
    raw_candidates: Path,
    nms_candidates: Path,
    scores: Path,
    mask_metadata: Path,
    test_manifest: Path,
    annotations: Path,
    evaluation_config_path: Path,
    output_root: Path,
    tmp_root: Path,
    expected_samples: int = 7675,
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Recompute every stage Oracle and q baseline from frozen source data."""

    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"baseline output is not empty: {output_root}")
    config = YAML(typ="safe").load(
        evaluation_config_path.read_text(encoding="utf-8")
    )
    samples, gt_geometry = _load_samples(
        test_manifest, annotations, config
    )
    if len(samples) != expected_samples:
        raise ValueError(
            f"sample count {len(samples)} != expected {expected_samples}"
        )

    mask_frame = pd.read_csv(
        mask_metadata,
        usecols=[
            "sample_id",
            "empty_mask",
            "mask_area_352_px",
            "mask_area_native_px",
        ],
    )
    if len(mask_frame) != expected_samples or mask_frame["sample_id"].duplicated().any():
        raise ValueError("mask metadata sample identity/count mismatch")
    if set(mask_frame["sample_id"]) != set(samples):
        raise ValueError("mask metadata sample set mismatch")
    for row in mask_frame.to_dict("records"):
        sample = samples[str(row["sample_id"])]
        sample["empty_mask"] = bool(row["empty_mask"])
        sample["mask_area_352_px"] = int(row["mask_area_352_px"])
        sample["mask_area_native_px"] = int(row["mask_area_native_px"])

    raw_parquet = pq.ParquetFile(raw_candidates)
    raw_rows = 0
    valid_rows = 0
    previous_sample_index = -1
    current_sample_id: str | None = None
    current_ids: set[str] = set()
    for batch in raw_parquet.iter_batches(
        batch_size=25_000,
        columns=[
            "pipeline",
            "stage",
            "sample_index",
            "sample_id",
            "candidate_id",
            "center_u_px",
            "center_v_px",
            "valid",
            "rejection_reason",
            "candidate_json",
        ],
    ):
        columns = batch.to_pydict()
        for offset in range(batch.num_rows):
            pipeline = str(columns["pipeline"][offset])
            stage = str(columns["stage"][offset])
            sample_index = int(columns["sample_index"][offset])
            sample_id = str(columns["sample_id"][offset])
            candidate_id = str(columns["candidate_id"][offset])
            if pipeline != "hierfilm" or stage != "raw":
                raise ValueError("raw Parquet contains a foreign lineage/stage")
            if sample_id not in samples:
                raise ValueError(f"unknown raw sample: {sample_id}")
            if sample_index != int(samples[sample_id]["sample_index"]):
                raise ValueError(f"raw sample index mismatch: {sample_id}")
            if sample_index < previous_sample_index:
                raise ValueError("raw Parquet is not sample-grouped")
            if sample_id != current_sample_id:
                current_sample_id = sample_id
                current_ids = set()
            if candidate_id in current_ids:
                raise ValueError(
                    f"duplicate raw candidate: {sample_id}/{candidate_id}"
                )
            current_ids.add(candidate_id)
            previous_sample_index = sample_index
            valid = bool(columns["valid"][offset])
            rejection = columns["rejection_reason"][offset]
            if valid != (rejection in (None, "")):
                raise ValueError(
                    f"raw valid/rejection mismatch: {sample_id}/{candidate_id}"
                )
            sample = samples[sample_id]
            sample["raw_candidate_count"] += 1
            if valid:
                sample["mask_validated_candidate_count"] += 1
            needs_label = (
                not bool(sample["raw_oracle"])
                or (valid and not bool(sample["mask_validated_oracle"]))
            )
            if needs_label:
                record = _candidate_record(
                    sample_id=sample_id,
                    candidate_id=candidate_id,
                    center_u_px=float(columns["center_u_px"][offset]),
                    center_v_px=float(columns["center_v_px"][offset]),
                    candidate_json=str(columns["candidate_json"][offset]),
                )
                label = _label_against_geometry(
                    record, gt_geometry[sample_id], config
                )
                sample["raw_oracle"] = bool(
                    sample["raw_oracle"]
                    or label["candidate_positive"]
                )
            else:
                label = {"candidate_positive": False}
            if valid:
                sample["mask_validated_oracle"] = bool(
                    sample["mask_validated_oracle"]
                    or label["candidate_positive"]
                )
                valid_rows += 1
            raw_rows += 1
            if raw_rows % progress_every == 0:
                print(
                    f"raw audit rows={raw_rows:,} valid={valid_rows:,}",
                    flush=True,
                )

    nms_parquet = pq.ParquetFile(nms_candidates)
    nms_rows = 0
    nms_identities: dict[tuple[str, str], str] = {}
    label_rows: list[dict[str, Any]] = []
    previous_sample_index = -1
    current_sample_id = None
    current_ids = set()
    for batch in nms_parquet.iter_batches(
        batch_size=25_000,
        columns=[
            "pipeline",
            "stage",
            "sample_index",
            "sample_id",
            "candidate_id",
            "center_u_px",
            "center_v_px",
            "candidate_json",
        ],
    ):
        columns = batch.to_pydict()
        for offset in range(batch.num_rows):
            pipeline = str(columns["pipeline"][offset])
            stage = str(columns["stage"][offset])
            sample_index = int(columns["sample_index"][offset])
            sample_id = str(columns["sample_id"][offset])
            candidate_id = str(columns["candidate_id"][offset])
            if pipeline != "hierfilm" or stage != "nms":
                raise ValueError("NMS Parquet contains a foreign lineage/stage")
            if sample_id not in samples:
                raise ValueError(f"unknown NMS sample: {sample_id}")
            if sample_index != int(samples[sample_id]["sample_index"]):
                raise ValueError(f"NMS sample index mismatch: {sample_id}")
            if sample_index < previous_sample_index:
                raise ValueError("NMS Parquet is not sample-grouped")
            if sample_id != current_sample_id:
                current_sample_id = sample_id
                current_ids = set()
            if candidate_id in current_ids:
                raise ValueError(
                    f"duplicate NMS candidate: {sample_id}/{candidate_id}"
                )
            current_ids.add(candidate_id)
            previous_sample_index = sample_index
            record = _candidate_record(
                sample_id=sample_id,
                candidate_id=candidate_id,
                center_u_px=float(columns["center_u_px"][offset]),
                center_v_px=float(columns["center_v_px"][offset]),
                candidate_json=str(columns["candidate_json"][offset]),
            )
            label = _label_against_geometry(
                record, gt_geometry[sample_id], config
            )
            identity = candidate_identity_sha256(record)
            key = (sample_id, candidate_id)
            nms_identities[key] = identity
            samples[sample_id]["nms_candidate_count"] += 1
            samples[sample_id]["nms_oracle"] = bool(
                samples[sample_id]["nms_oracle"]
                or label["candidate_positive"]
            )
            label_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": samples[sample_id]["scene_id"],
                    "split": "test",
                    "candidate_id": candidate_id,
                    "candidate_identity_sha256": identity,
                    **label,
                }
            )
            nms_rows += 1
            if nms_rows % progress_every == 0:
                print(f"NMS audit rows={nms_rows:,}", flush=True)
    if len(nms_identities) != nms_rows:
        raise ValueError("duplicate NMS candidate keys")

    source_schema = set(pq.ParquetFile(scores).schema_arrow.names)
    missing_safe = set(SAFE_SCORE_COLUMNS) - source_schema
    if missing_safe:
        raise ValueError(f"score Parquet missing safe columns: {missing_safe}")
    score_frame = pq.read_table(
        scores, columns=list(SAFE_SCORE_COLUMNS)
    ).to_pandas()
    if (
        len(score_frame) != nms_rows
        or score_frame[["sample_id", "candidate_id"]].duplicated().any()
        or not np.all(np.isfinite(score_frame["gqcnn_q_value"]))
    ):
        raise ValueError("score count/key/finite-q audit failed")
    if set(score_frame["pipeline"].astype(str)) != {"hierfilm"}:
        raise ValueError("score Parquet contains a foreign lineage")
    score_keys = set(
        zip(
            score_frame["sample_id"].astype(str),
            score_frame["candidate_id"].astype(str),
        )
    )
    if score_keys != set(nms_identities):
        raise ValueError("NMS and score candidate key sets differ")
    score_frame["candidate_identity_sha256"] = [
        _identity_from_score_row(row)
        for row in score_frame.to_dict("records")
    ]
    for row in score_frame[
        ["sample_id", "candidate_id", "candidate_identity_sha256"]
    ].to_dict("records"):
        key = (str(row["sample_id"]), str(row["candidate_id"]))
        if str(row["candidate_identity_sha256"]) != nms_identities[key]:
            raise ValueError(f"NMS/score geometry mismatch: {key}")

    expected_ranks = (
        score_frame.sort_values(
            ["sample_id", "gqcnn_q_value", "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("sample_id", sort=False)
        .cumcount()
        .add(1)
    )
    if not np.array_equal(
        score_frame.loc[expected_ranks.index, "gqcnn_rank"].to_numpy(
            dtype=np.int64
        ),
        expected_ranks.to_numpy(dtype=np.int64),
    ):
        raise ValueError("stored GQ-CNN rank disagrees with q/candidate_id order")

    label_frame = pd.DataFrame(label_rows)
    ranked = score_frame.merge(
        label_frame[
            [
                "sample_id",
                "candidate_id",
                "candidate_positive",
                "candidate_identity_sha256",
            ]
        ],
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_label"),
    )
    if ranked["candidate_positive"].isna().any():
        raise ValueError("candidate label join is incomplete")
    if not (
        ranked["candidate_identity_sha256"]
        == ranked["candidate_identity_sha256_label"]
    ).all():
        raise ValueError("label/score identity hashes disagree")
    ranked = ranked.sort_values(
        ["sample_id", "gqcnn_rank", "candidate_id"],
        kind="mergesort",
    )
    for sample_id, group in ranked.groupby("sample_id", sort=False):
        positive = group["candidate_positive"].to_numpy(dtype=bool)
        ranks = group["gqcnn_rank"].to_numpy(dtype=np.int64)
        if not np.array_equal(ranks, np.arange(1, len(group) + 1)):
            raise ValueError(f"non-contiguous score ranks: {sample_id}")
        correct_ranks = ranks[positive]
        first = (
            None if correct_ranks.size == 0 else int(correct_ranks.min())
        )
        sample = samples[str(sample_id)]
        sample["scored_candidate_count"] = len(group)
        sample["q_top1"] = bool(first is not None and first <= 1)
        sample["q_top5"] = bool(first is not None and first <= 5)
        sample["q_top10"] = bool(first is not None and first <= 10)
        sample["first_correct_rank"] = first
        sample["reciprocal_rank"] = (
            0.0 if first is None else float(1.0 / first)
        )
    for sample in samples.values():
        if sample["nms_candidate_count"] != sample["scored_candidate_count"]:
            raise ValueError(
                f"NMS/score per-sample mismatch: {sample['sample_id']}"
            )
        sample["funnel_category"] = _funnel_category(sample)

    per_sample = pd.DataFrame(samples.values()).sort_values(
        "sample_index", kind="mergesort"
    )
    label_frame = label_frame.sort_values(
        ["sample_id", "candidate_id"], kind="mergesort"
    )
    deployment = score_frame.copy()
    deployment["query"] = deployment["sample_id"].map(
        {key: value["query"] for key, value in samples.items()}
    )
    deployment = deployment.sort_values(
        ["sample_index", "gqcnn_rank", "candidate_id"],
        kind="mergesort",
    )
    if FORBIDDEN_SCORE_COLUMNS & set(deployment.columns):
        raise AssertionError("deployment projection contains GT columns")
    candidate_universe_sha = hashlib.sha256(
        "".join(
            f"{row.sample_id}\t{row.candidate_id}\t"
            f"{row.candidate_identity_sha256}\n"
            for row in deployment[
                [
                    "sample_id",
                    "candidate_id",
                    "candidate_identity_sha256",
                ]
            ].itertuples(index=False)
        ).encode("utf-8")
    ).hexdigest()

    totals = {
        "samples": int(len(per_sample)),
        "scenes": int(per_sample["scene_id"].nunique()),
        "raw_candidates": int(per_sample["raw_candidate_count"].sum()),
        "mask_validated_candidates": int(
            per_sample["mask_validated_candidate_count"].sum()
        ),
        "nms_candidates": int(per_sample["nms_candidate_count"].sum()),
        "scored_candidates": int(
            per_sample["scored_candidate_count"].sum()
        ),
        "empty_masks": int(per_sample["empty_mask"].sum()),
        "empty_nms_samples": int(
            (per_sample["nms_candidate_count"] == 0).sum()
        ),
        "finite_q_values": int(
            np.isfinite(deployment["gqcnn_q_value"]).sum()
        ),
        "execution_failures": 0,
    }
    metric_columns = {
        "raw_oracle": "raw_oracle",
        "mask_validated_oracle": "mask_validated_oracle",
        "nms_oracle": "nms_oracle",
        "j_at_1": "q_top1",
        "recall_at_5": "q_top5",
        "recall_at_10": "q_top10",
        "j_at_any": "nms_oracle",
    }
    metrics = {
        name: _ratio(int(per_sample[column].sum()), expected_samples)
        for name, column in metric_columns.items()
    }
    metrics["mrr"] = {
        "denominator": expected_samples,
        "value": float(per_sample["reciprocal_rank"].mean()),
    }
    valid_ranks = per_sample["first_correct_rank"].dropna().astype(float)
    metrics["first_correct_rank"] = {
        "count": int(len(valid_ranks)),
        "mean": float(valid_ranks.mean()),
        "median": float(valid_ranks.median()),
        "maximum": float(valid_ranks.max()),
    }
    funnel = Counter(per_sample["funnel_category"].astype(str))
    if sum(funnel.values()) != expected_samples:
        raise AssertionError("funnel is not exhaustive")
    summary = {
        "schema_version": 1,
        **identity_payload(),
        "lineage": "hierarchical_repeated_film_only",
        "single_film_used": False,
        "metric_label": (
            "offline 2D rectangle consistency; not physical grasp success"
        ),
        "predicate": {
            "same_ground_truth_must_jointly_pass": True,
            "rectangle_iou": "> 0.25",
            "parallel_jaw_periodic_angle_difference": "<= 30 degrees",
            "coordinate_convention": "(x,y)=(u,v); raster row=y,column=x",
        },
        "denominator_policy": "all 7,675 official unique test queries",
        "sources": {
            "raw_candidates": {
                "path": str(raw_candidates),
                "sha256": sha256_file(raw_candidates),
            },
            "nms_candidates": {
                "path": str(nms_candidates),
                "sha256": sha256_file(nms_candidates),
            },
            "scores_read_with_safe_projection_only": {
                "path": str(scores),
                "sha256": sha256_file(scores),
            },
            "test_manifest": {
                "path": str(test_manifest),
                "sha256": sha256_file(test_manifest),
            },
            "annotations": {
                "path": str(annotations),
                "sha256": sha256_file(annotations),
            },
            "evaluation_config": {
                "path": str(evaluation_config_path),
                "sha256": sha256_file(evaluation_config_path),
            },
        },
        "counts": totals,
        "metrics": metrics,
        "funnel": dict(sorted(funnel.items())),
        "candidate_universe_sha256": candidate_universe_sha,
        "gt_leakage_control": {
            "retained_score_gt_columns_read": False,
            "labels_recomputed_from_official_annotations": True,
            "deployment_projection_forbidden_column_intersection": [],
            "labels_physically_separate": True,
        },
    }
    expected_anchor = {
        "raw_candidates": 1466046,
        "mask_validated_candidates": 1449011,
        "nms_candidates": 187077,
        "scored_candidates": 187077,
        "empty_masks": 1,
        "empty_nms_samples": 108,
        "j_at_1": 2934,
        "recall_at_5": 4655,
        "recall_at_10": 5201,
        "j_at_any": 5799,
    }
    observed_anchor = {
        **{
            key: totals[key]
            for key in (
                "raw_candidates",
                "mask_validated_candidates",
                "nms_candidates",
                "scored_candidates",
                "empty_masks",
                "empty_nms_samples",
            )
        },
        **{
            key: metrics[key]["numerator"]
            for key in (
                "j_at_1",
                "recall_at_5",
                "recall_at_10",
                "j_at_any",
            )
        },
    }
    summary["retained_anchor_check"] = {
        "expected": expected_anchor,
        "observed": observed_anchor,
        "all_equal": observed_anchor == expected_anchor,
    }
    if observed_anchor != expected_anchor:
        raise AssertionError(
            f"independent baseline disagrees with retained anchors: "
            f"{observed_anchor}"
        )

    deployment_path = output_root / "test_deployment_candidates.parquet"
    labels_path = output_root / "test_candidate_labels_gt_only.parquet"
    sample_path = output_root / "test_per_sample_baseline.parquet"
    funnel_path = output_root / "test_funnel_counts.csv"
    summary_path = output_root / "repeatedfilm_baseline_oracle.json"
    _atomic_parquet(deployment_path, deployment, tmp_root)
    _atomic_parquet(labels_path, label_frame, tmp_root)
    _atomic_parquet(sample_path, per_sample, tmp_root)
    _atomic_csv(
        funnel_path,
        pd.DataFrame(
            [
                {"funnel_category": key, "count": value}
                for key, value in sorted(funnel.items())
            ]
        ),
        tmp_root,
    )
    summary["outputs"] = {
        "deployment_candidates": {
            "path": str(deployment_path),
            "sha256": sha256_file(deployment_path),
            "rows": int(len(deployment)),
        },
        "gt_only_labels": {
            "path": str(labels_path),
            "sha256": sha256_file(labels_path),
            "rows": int(len(label_frame)),
        },
        "per_sample": {
            "path": str(sample_path),
            "sha256": sha256_file(sample_path),
            "rows": int(len(per_sample)),
        },
        "funnel_counts": {
            "path": str(funnel_path),
            "sha256": sha256_file(funnel_path),
        },
    }
    _atomic_json(summary_path, summary, tmp_root)
    return summary
