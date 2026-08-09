"""Independent frozen-test baseline and Oracle-funnel audit."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from src.grasping.geometric_ranker import load_frozen_candidates

from .identity import CANDIDATE_POSE_FIELDS, sha256_file
from .labels import evaluate_candidate_label


EXPECTED_BASELINE = {
    "total_samples": 7675,
    "independent_scene_frames": 325,
    "raw_candidates": 1567552,
    "mask_validated_candidates": 1551137,
    "nms_candidates": 206538,
    "nonempty_samples": 7620,
    "valid_empty_samples": 55,
    "finite_q_values": 206538,
    "execution_failures": 0,
    "historical_geq_top1_nonempty_numerator": 2835,
    "historical_geq_top5_nonempty_numerator": 4787,
    "strict_gt_top1_nonempty_numerator": 2833,
    "strict_gt_top5_nonempty_numerator": 4787,
    "strict_gt_nms_oracle_nonempty_numerator": 6114,
    "nonempty_all_negative_samples": 1506,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("candidate JSON does not contain a candidate list")
    return [dict(row) for row in rows]


def _annotation_index(path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("annotation file has no data list")
    index = {int(row["question_index"]): dict(row) for row in rows}
    if len(index) != len(rows):
        raise ValueError("annotation question_index values are not unique")
    return index


def _ranked_records(
    source_dir: Path,
    scored_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    source, _, _ = load_frozen_candidates(
        source_dir / "candidates.npz", source_dir / "candidates.json"
    )
    with np.load(
        scored_dir / "gqcnn_scored_candidates.npz", allow_pickle=False
    ) as archive:
        scored = {name: np.asarray(archive[name]) for name in archive.files}
    ids = [str(row["candidate_id"]) for row in source]
    scored_ids = [str(value) for value in scored["candidate_id"].tolist()]
    if ids != scored_ids:
        raise ValueError(f"{source_dir.name}: scored NPZ source order changed")
    for field in CANDIDATE_POSE_FIELDS:
        expected = np.asarray(scored[field])
        if field == "center_uv":
            actual = np.asarray(
                [[row["center_u_px"], row["center_v_px"]] for row in source],
                dtype=expected.dtype,
            )
        elif field == "endpoints_uv":
            actual = np.asarray(
                [[row["endpoint_1_uv"], row["endpoint_2_uv"]] for row in source],
                dtype=expected.dtype,
            )
        else:
            actual = np.asarray([row[field] for row in source], dtype=expected.dtype)
        if not np.array_equal(actual, expected, equal_nan=True):
            raise ValueError(f"{source_dir.name}: scored pose mismatch for {field}")
    q_values = np.asarray(scored["gqcnn_q_value"], dtype=np.float64)
    stored_ranks = np.asarray(scored["gqcnn_rank"], dtype=np.int64)
    if q_values.shape != (len(source),) or not np.all(np.isfinite(q_values)):
        raise ValueError(f"{source_dir.name}: q-values are missing or non-finite")
    order = np.asarray(
        sorted(range(len(source)), key=lambda index: (-q_values[index], ids[index])),
        dtype=np.int64,
    )
    reconstructed = np.empty(len(source), dtype=np.int64)
    reconstructed[order] = np.arange(1, len(source) + 1, dtype=np.int64)
    if not np.array_equal(stored_ranks, reconstructed):
        raise ValueError(f"{source_dir.name}: stored GQ-CNN ranks disagree")
    ranked = []
    for index in order:
        row = dict(source[int(index)])
        row["gqcnn_q_value"] = float(q_values[index])
        row["gqcnn_rank"] = int(reconstructed[index])
        ranked.append(row)
    return source, ranked, q_values, stored_ranks


def _stage_labels(
    rows: Sequence[Mapping[str, Any]],
    grasps: Sequence[Sequence[Sequence[float]]],
    config: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    labels = []
    for row in rows:
        label = evaluate_candidate_label(row, grasps, config)
        labels.append(
            {
                "candidate_id": str(row["candidate_id"]),
                **label.to_dict(),
            }
        )
    return any(row["candidate_positive"] for row in labels), labels


def classify_funnel_stage(
    *,
    valid_empty: bool,
    raw_oracle: bool,
    mask_validated_oracle: bool,
    nms_oracle: bool,
    gqcnn_top1: bool,
    gqcnn_top5: bool,
) -> str:
    """Assign exactly one preregistered failure/success stage."""

    if valid_empty:
        return "valid_empty"
    if gqcnn_top1:
        return "already_correct"
    if gqcnn_top5:
        return "ranking_loss_top5"
    if nms_oracle:
        return "ranking_loss_beyond_top5"
    if mask_validated_oracle:
        return "nms_loss"
    if raw_oracle:
        return "mask_filter_loss"
    return "generation_limited"


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "decimal": None if denominator == 0 else float(numerator / denominator),
        "percentage": None
        if denominator == 0
        else float(100.0 * numerator / denominator),
    }


def audit_frozen_baseline(
    *,
    candidate_root: Path,
    scored_root: Path,
    bundle_manifest: Path,
    annotation_file: Path,
    evaluation_config: Path,
    progress_every: int = 100,
    limit: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Audit all stages and return summary, per-sample rows, and NMS labels."""

    config = YAML(typ="safe").load(evaluation_config.read_text(encoding="utf-8"))
    manifest = [
        json.loads(line)
        for line in bundle_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        manifest = manifest[: int(limit)]
    annotations = _annotation_index(annotation_file)
    sample_rows: list[dict[str, Any]] = []
    candidate_labels: list[dict[str, Any]] = []
    totals = Counter()
    legacy_difference_candidates = 0
    exact_boundary_pairs = 0
    reciprocal_rank_sum = 0.0
    legacy_reciprocal_rank_sum = 0.0
    first_valid_ranks: list[int] = []
    legacy_first_valid_ranks: list[int] = []
    execution_failures: list[dict[str, str]] = []

    for position, manifest_row in enumerate(manifest, start=1):
        sample_id = str(manifest_row["sample_id"])
        source_dir = candidate_root / sample_id
        sample_scored_dir = scored_root / sample_id
        try:
            metadata = _read_json(source_dir / "metadata.json")
            if (
                metadata["sample_id"] != sample_id
                or metadata["scene_id"] != manifest_row["scene_id"]
                or metadata["query"] != manifest_row["query"]
            ):
                raise ValueError(f"{sample_id}: manifest/metadata join mismatch")
            annotation = annotations[int(metadata["question_index"])]
            if (
                annotation["image_filename"] != metadata["scene_id"]
                or annotation["question"] != metadata["query"]
            ):
                raise ValueError(f"{sample_id}: annotation join mismatch")
            grasps = annotation.get("grasps")
            if not isinstance(grasps, list) or not grasps:
                raise ValueError(f"{sample_id}: no GT grasp rectangles")
            counts = metadata["counts"]
            raw_rows = _candidate_rows(_read_json(source_dir / "raw_candidates.json"))
            mask_rows = _candidate_rows(
                _read_json(source_dir / "mask_validated_candidates.json")
            )
            nms_rows = _candidate_rows(_read_json(source_dir / "candidates.json"))
            if (
                len(raw_rows) != int(counts["raw"])
                or len(mask_rows) != int(counts["mask_validated"])
                or len(nms_rows) != int(counts["post_nms"])
            ):
                raise ValueError(f"{sample_id}: candidate stage counts disagree")

            raw_oracle, _ = _stage_labels(raw_rows, grasps, config)
            mask_oracle, _ = _stage_labels(mask_rows, grasps, config)
            nms_oracle, nms_labels = _stage_labels(nms_rows, grasps, config)
            label_by_id = {row["candidate_id"]: row for row in nms_labels}
            if len(label_by_id) != len(nms_rows):
                raise ValueError(f"{sample_id}: duplicate NMS candidate IDs")
            valid_empty = len(nms_rows) == 0
            ranked: list[dict[str, Any]] = []
            if valid_empty:
                scoring_metadata = _read_json(
                    sample_scored_dir / "scoring_metadata.json"
                )
                if scoring_metadata["scoring_status"] != "skipped_valid_empty":
                    raise ValueError(f"{sample_id}: invalid empty scoring status")
            else:
                _, ranked, _, _ = _ranked_records(source_dir, sample_scored_dir)
                if {row["candidate_id"] for row in ranked} != set(label_by_id):
                    raise ValueError(f"{sample_id}: label/ranking candidate IDs differ")

            positives = [
                bool(label_by_id[row["candidate_id"]]["candidate_positive"])
                for row in ranked
            ]
            legacy_positives = [
                bool(label_by_id[row["candidate_id"]]["legacy_geq_positive"])
                for row in ranked
            ]
            first_rank = next(
                (rank for rank, positive in enumerate(positives, start=1) if positive),
                None,
            )
            legacy_first_rank = next(
                (
                    rank
                    for rank, positive in enumerate(legacy_positives, start=1)
                    if positive
                ),
                None,
            )
            top1 = bool(positives[:1] and any(positives[:1]))
            top5 = bool(any(positives[:5]))
            top10 = bool(any(positives[:10]))
            legacy_top1 = bool(legacy_positives[:1] and any(legacy_positives[:1]))
            legacy_top5 = bool(any(legacy_positives[:5]))
            legacy_top10 = bool(any(legacy_positives[:10]))
            reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
            legacy_reciprocal_rank = (
                0.0 if legacy_first_rank is None else 1.0 / legacy_first_rank
            )
            reciprocal_rank_sum += reciprocal_rank
            legacy_reciprocal_rank_sum += legacy_reciprocal_rank
            if first_rank is not None:
                first_valid_ranks.append(first_rank)
            if legacy_first_rank is not None:
                legacy_first_valid_ranks.append(legacy_first_rank)

            for row in ranked:
                candidate_id = str(row["candidate_id"])
                label = label_by_id[candidate_id]
                legacy_difference_candidates += int(
                    label["candidate_positive"] != label["legacy_geq_positive"]
                )
                exact_boundary_pairs += int(label["exact_iou_threshold_pair_count"])
                candidate_labels.append(
                    {
                        "sample_id": sample_id,
                        "scene_id": str(metadata["scene_id"]),
                        "candidate_id": candidate_id,
                        "split": "test",
                        "original_gqcnn_rank": int(row["gqcnn_rank"]),
                        "q_raw": float(row["gqcnn_q_value"]),
                        **label,
                    }
                )

            totals.update(
                {
                    "total_samples": 1,
                    "independent_scene_frames": 0,
                    "raw_candidates": len(raw_rows),
                    "mask_validated_candidates": len(mask_rows),
                    "nms_candidates": len(nms_rows),
                    "nonempty_samples": int(not valid_empty),
                    "valid_empty_samples": int(valid_empty),
                    "finite_q_values": len(ranked),
                    "raw_oracle": int(raw_oracle),
                    "mask_validated_oracle": int(mask_oracle),
                    "nms_oracle": int(nms_oracle),
                    "gqcnn_top1": int(top1),
                    "gqcnn_top5": int(top5),
                    "gqcnn_top10": int(top10),
                    "gqcnn_any": int(first_rank is not None),
                    "legacy_geq_gqcnn_top1": int(legacy_top1),
                    "legacy_geq_gqcnn_top5": int(legacy_top5),
                    "legacy_geq_gqcnn_top10": int(legacy_top10),
                    "legacy_geq_gqcnn_any": int(legacy_first_rank is not None),
                    "nonempty_all_negative_samples": int(
                        not valid_empty and first_rank is None
                    ),
                }
            )
            stage = classify_funnel_stage(
                valid_empty=valid_empty,
                raw_oracle=raw_oracle,
                mask_validated_oracle=mask_oracle,
                nms_oracle=nms_oracle,
                gqcnn_top1=top1,
                gqcnn_top5=top5,
            )
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": str(metadata["scene_id"]),
                    "question_index": int(metadata["question_index"]),
                    "query": str(metadata["query"]),
                    "split": "test",
                    "raw_candidate_count": len(raw_rows),
                    "mask_validated_candidate_count": len(mask_rows),
                    "nms_candidate_count": len(nms_rows),
                    "valid_empty": valid_empty,
                    "raw_oracle": raw_oracle,
                    "mask_validated_oracle": mask_oracle,
                    "nms_oracle": nms_oracle,
                    "gqcnn_top1": top1,
                    "gqcnn_top5": top5,
                    "gqcnn_top10": top10,
                    "gqcnn_any": first_rank is not None,
                    "first_valid_rank": first_rank,
                    "reciprocal_rank": reciprocal_rank,
                    "legacy_geq_gqcnn_top1": legacy_top1,
                    "legacy_geq_gqcnn_top5": legacy_top5,
                    "legacy_geq_gqcnn_top10": legacy_top10,
                    "legacy_geq_first_valid_rank": legacy_first_rank,
                    "legacy_geq_reciprocal_rank": legacy_reciprocal_rank,
                    "funnel_category": stage,
                }
            )
        except Exception as error:
            execution_failures.append(
                {"sample_id": sample_id, "error": f"{type(error).__name__}: {error}"}
            )
        if progress_every > 0 and (
            position % progress_every == 0 or position == len(manifest)
        ):
            print(
                f"audit {position}/{len(manifest)} "
                f"failures={len(execution_failures)} "
                f"nms_candidates={totals['nms_candidates']}",
                flush=True,
            )

    totals["independent_scene_frames"] = len(
        {str(row["scene_id"]) for row in sample_rows}
    )
    totals["execution_failures"] = len(execution_failures)
    nonempty = int(totals["nonempty_samples"])
    all_samples = int(totals["total_samples"])
    positive_samples = len(first_valid_ranks)
    category_counts = Counter(row["funnel_category"] for row in sample_rows)
    summary = {
        "schema_version": 1,
        "title": "Independent strict-threshold frozen baseline audit",
        "is_full_frozen_test": limit is None,
        "predicate": {
            "angle": "periodic parallel-jaw angle difference <= 30 degrees",
            "iou": "rectangle IoU > 0.25",
            "same_gt_rectangle_required": True,
            "coordinate_convention": "(x,y)=(u,v); never (row,column)",
        },
        "inputs": {
            "candidate_root": str(candidate_root.resolve()),
            "scored_root": str(scored_root.resolve()),
            "bundle_manifest": str(bundle_manifest.resolve()),
            "bundle_manifest_sha256": sha256_file(bundle_manifest),
            "annotation_file": str(annotation_file.resolve()),
            "annotation_file_sha256": sha256_file(annotation_file),
            "evaluation_config": str(evaluation_config.resolve()),
            "evaluation_config_sha256": sha256_file(evaluation_config),
        },
        "counts": dict(totals),
        "metrics": {
            "strict_gt_iou_0_25": {
                "conditional_nonempty": {
                    "top1": _ratio(int(totals["gqcnn_top1"]), nonempty),
                    "top5": _ratio(int(totals["gqcnn_top5"]), nonempty),
                    "top10": _ratio(int(totals["gqcnn_top10"]), nonempty),
                    "full_nms_oracle": _ratio(
                        int(totals["nms_oracle"]), nonempty
                    ),
                    "mrr_zero_for_no_positive": None
                    if nonempty == 0
                    else float(reciprocal_rank_sum / nonempty),
                },
                "all_samples": {
                    "top1": _ratio(int(totals["gqcnn_top1"]), all_samples),
                    "top5": _ratio(int(totals["gqcnn_top5"]), all_samples),
                    "top10": _ratio(int(totals["gqcnn_top10"]), all_samples),
                    "full_nms_oracle": _ratio(
                        int(totals["nms_oracle"]), all_samples
                    ),
                    "mrr_zero_for_no_positive": None
                    if all_samples == 0
                    else float(reciprocal_rank_sum / all_samples),
                },
                "positive_samples_only": {
                    "count": positive_samples,
                    "mean_first_valid_rank": None
                    if not first_valid_ranks
                    else float(np.mean(first_valid_ranks)),
                    "median_first_valid_rank": None
                    if not first_valid_ranks
                    else float(np.median(first_valid_ranks)),
                    "mrr": None
                    if positive_samples == 0
                    else float(reciprocal_rank_sum / positive_samples),
                },
            },
            "historical_geq_iou_0_25_reproduction": {
                "conditional_nonempty": {
                    "top1": _ratio(
                        int(totals["legacy_geq_gqcnn_top1"]), nonempty
                    ),
                    "top5": _ratio(
                        int(totals["legacy_geq_gqcnn_top5"]), nonempty
                    ),
                    "top10": _ratio(
                        int(totals["legacy_geq_gqcnn_top10"]), nonempty
                    ),
                    "mrr_zero_for_no_positive": None
                    if nonempty == 0
                    else float(legacy_reciprocal_rank_sum / nonempty),
                },
                "all_samples": {
                    "top1": _ratio(
                        int(totals["legacy_geq_gqcnn_top1"]), all_samples
                    ),
                    "top5": _ratio(
                        int(totals["legacy_geq_gqcnn_top5"]), all_samples
                    ),
                    "top10": _ratio(
                        int(totals["legacy_geq_gqcnn_top10"]), all_samples
                    ),
                    "mrr_zero_for_no_positive": None
                    if all_samples == 0
                    else float(legacy_reciprocal_rank_sum / all_samples),
                },
                "positive_samples_only": {
                    "count": len(legacy_first_valid_ranks),
                    "mean_first_valid_rank": None
                    if not legacy_first_valid_ranks
                    else float(np.mean(legacy_first_valid_ranks)),
                    "median_first_valid_rank": None
                    if not legacy_first_valid_ranks
                    else float(np.median(legacy_first_valid_ranks)),
                    "mrr": None
                    if not legacy_first_valid_ranks
                    else float(
                        legacy_reciprocal_rank_sum / len(legacy_first_valid_ranks)
                    ),
                },
            },
        },
        "oracle_funnel": {
            name: _ratio(int(totals[name]), all_samples)
            for name in (
                "raw_oracle",
                "mask_validated_oracle",
                "nms_oracle",
                "gqcnn_top10",
                "gqcnn_top5",
                "gqcnn_top1",
                "gqcnn_any",
            )
        },
        "mutually_exclusive_funnel_categories": {
            "counts": dict(sorted(category_counts.items())),
            "sum": int(sum(category_counts.values())),
        },
        "threshold_boundary_audit": {
            "exact_angle_eligible_iou_0_25_pairs": exact_boundary_pairs,
            "candidates_differing_between_strict_gt_and_legacy_geq": (
                legacy_difference_candidates
            ),
        },
        "execution_failures": execution_failures,
    }
    return summary, pd.DataFrame(sample_rows), pd.DataFrame(candidate_labels)


def assert_expected_baseline(summary: Mapping[str, Any]) -> None:
    """Fail closed unless every preregistered hard-gate count reproduces."""

    counts = summary["counts"]
    observed = {
        "total_samples": int(counts["total_samples"]),
        "independent_scene_frames": int(counts["independent_scene_frames"]),
        "raw_candidates": int(counts["raw_candidates"]),
        "mask_validated_candidates": int(counts["mask_validated_candidates"]),
        "nms_candidates": int(counts["nms_candidates"]),
        "nonempty_samples": int(counts["nonempty_samples"]),
        "valid_empty_samples": int(counts["valid_empty_samples"]),
        "finite_q_values": int(counts["finite_q_values"]),
        "execution_failures": int(counts["execution_failures"]),
        "historical_geq_top1_nonempty_numerator": int(
            counts["legacy_geq_gqcnn_top1"]
        ),
        "historical_geq_top5_nonempty_numerator": int(
            counts["legacy_geq_gqcnn_top5"]
        ),
        "strict_gt_top1_nonempty_numerator": int(counts["gqcnn_top1"]),
        "strict_gt_top5_nonempty_numerator": int(counts["gqcnn_top5"]),
        "strict_gt_nms_oracle_nonempty_numerator": int(counts["nms_oracle"]),
        "nonempty_all_negative_samples": int(
            counts["nonempty_all_negative_samples"]
        ),
    }
    differences = {
        key: {"expected": expected, "observed": observed[key]}
        for key, expected in EXPECTED_BASELINE.items()
        if observed[key] != expected
    }
    if differences:
        raise AssertionError(f"frozen baseline hard gate failed: {differences}")
    if summary["mutually_exclusive_funnel_categories"]["sum"] != 7675:
        raise AssertionError("mutually exclusive funnel categories do not sum to 7675")
    for convention in (
        "strict_gt_iou_0_25",
        "historical_geq_iou_0_25_reproduction",
    ):
        for scope in ("conditional_nonempty", "all_samples"):
            for name in ("top1", "top5"):
                value = summary["metrics"][convention][scope][name]["decimal"]
                if value is None or not math.isfinite(float(value)):
                    raise AssertionError(
                        f"{convention}/{scope}/{name} is not finite"
                    )
