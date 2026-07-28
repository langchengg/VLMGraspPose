#!/usr/bin/env python3
"""Independently recompute, compare, visualize, and close the modular experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import binomtest, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SAMPLES = 7675
ARCHIVED_HISTORICAL = {
    "scope": "archived historical; fixed-seed-42 and continuous-polygon evaluator",
    "scored": 7620,
    "valid_empty": 55,
    "top1_correct": 2835,
    "top1_wrong_but_later_correct_all_pool": 3279,
    "top1_wrong_later_rank_2_to_5": 1952,
    "top1_wrong_later_rank_after_5": 1327,
    "no_correct_candidate": 1506,
    "per_sample_sha256": (
        "45d3c5fb32c92116a839f980311dd2102aa924fc3c44d88ecbef07ef2a445838"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--official-annotations", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def file_manifest(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def combine_mask_metadata(run_dir: Path) -> pd.DataFrame:
    frames = []
    for pipeline in ("old_singlefilm", "hierfilm"):
        path = run_dir / "masks" / pipeline / "per_sample_mask_metadata.csv"
        frame = pd.read_csv(path)
        if len(frame) != EXPECTED_SAMPLES:
            raise ValueError(f"{pipeline} mask metadata count mismatch")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output = run_dir / "masks" / "per_sample_mask_metadata.csv"
    combined.to_csv(output, index=False)
    return combined


CANDIDATE_SCHEMA = pa.schema(
    [
        ("pipeline", pa.string()),
        ("stage", pa.string()),
        ("sample_index", pa.int32()),
        ("sample_id", pa.string()),
        ("question_index", pa.int32()),
        ("scene_id", pa.string()),
        ("candidate_id", pa.string()),
        ("sampler_rank", pa.int32()),
        ("candidate_seed", pa.int64()),
        ("center_u_px", pa.float64()),
        ("center_v_px", pa.float64()),
        ("center_depth_m", pa.float64()),
        ("angle_rad", pa.float64()),
        ("width_m", pa.float64()),
        ("width_px", pa.float64()),
        ("valid", pa.bool_()),
        ("rejection_reason", pa.string()),
        ("candidate_json", pa.string()),
    ]
)


def candidate_row(
    pipeline: str,
    stage: str,
    sample_index: int,
    question_index: int,
    scene_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "stage": stage,
        "sample_index": int(sample_index),
        "sample_id": str(candidate["sample_id"]),
        "question_index": int(question_index),
        "scene_id": scene_id,
        "candidate_id": str(candidate["candidate_id"]),
        "sampler_rank": int(candidate["sampler_rank"]),
        "candidate_seed": int(candidate["seed"]),
        "center_u_px": float(candidate["center_u_px"]),
        "center_v_px": float(candidate["center_v_px"]),
        "center_depth_m": float(candidate["center_depth_m"]),
        "angle_rad": float(candidate["angle_rad"]),
        "width_m": float(candidate["width_m"]),
        "width_px": float(candidate["width_px"]),
        "valid": candidate.get("rejection_reason") is None,
        "rejection_reason": candidate.get("rejection_reason"),
        "candidate_json": json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), allow_nan=True
        ),
    }


def write_candidate_parquets(run_dir: Path) -> dict[str, Any]:
    outputs = {
        "raw": run_dir / "candidates" / "dexnet_raw_candidates.parquet",
        "nms": run_dir / "candidates" / "dexnet_nms_candidates.parquet",
    }
    writers = {
        key: pq.ParquetWriter(
            path, CANDIDATE_SCHEMA, compression="zstd", use_dictionary=True
        )
        for key, path in outputs.items()
    }
    batches: dict[str, list[dict[str, Any]]] = {"raw": [], "nms": []}
    counts = Counter()
    try:
        for pipeline in ("old_singlefilm", "hierfilm"):
            root = run_dir / "candidates" / pipeline
            summary = pd.read_csv(root / "summary.csv")
            if len(summary) != EXPECTED_SAMPLES:
                raise ValueError(f"{pipeline} candidate summary count mismatch")
            for sample_index, summary_row in summary.iterrows():
                sample_id = str(summary_row["sample_id"])
                directory = root / sample_id
                metadata = load_json(directory / "metadata.json")
                sources = {
                    "raw": load_json(directory / "raw_candidates.json"),
                    "nms": load_json(directory / "candidates.json")["candidates"],
                }
                for stage, records in sources.items():
                    for candidate in records:
                        batches[stage].append(
                            candidate_row(
                                pipeline,
                                stage,
                                int(sample_index),
                                int(metadata["question_index"]),
                                str(metadata["scene_id"]),
                                candidate,
                            )
                        )
                    counts[f"{pipeline}_{stage}"] += len(records)
                    if len(batches[stage]) >= 5000:
                        writers[stage].write_table(
                            pa.Table.from_pylist(
                                batches[stage], schema=CANDIDATE_SCHEMA
                            )
                        )
                        batches[stage].clear()
        for stage in batches:
            if batches[stage]:
                writers[stage].write_table(
                    pa.Table.from_pylist(batches[stage], schema=CANDIDATE_SCHEMA)
                )
                batches[stage].clear()
    finally:
        for writer in writers.values():
            writer.close()
    return {
        "counts": dict(counts),
        "files": {
            stage: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
            for stage, path in outputs.items()
        },
    }


def combine_scores(run_dir: Path) -> dict[str, Any]:
    paths = [
        run_dir
        / "evaluation"
        / f"{pipeline}_gqcnn_per_candidate.parquet"
        for pipeline in ("old_singlefilm", "hierfilm")
    ]
    output = run_dir / "scores" / "gqcnn_per_candidate.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    try:
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=50000):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(
                        output,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("no scored candidate rows")
    table = pq.read_table(
        output, columns=["gqcnn_q_value", "gqcnn_rank", "candidate_success"]
    )
    q_values = table["gqcnn_q_value"].to_numpy()
    if not np.all(np.isfinite(q_values)):
        raise RuntimeError("combined score parquet contains non-finite Q")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "rows": rows,
        "finite_q_values": int(np.isfinite(q_values).sum()),
    }


def load_paired(run_dir: Path) -> pd.DataFrame:
    frames = {}
    for pipeline in ("old_singlefilm", "hierfilm"):
        path = (
            run_dir
            / "evaluation"
            / f"{pipeline}_per_sample_pipeline_metrics.csv"
        )
        frame = pd.read_csv(path)
        if len(frame) != EXPECTED_SAMPLES or frame["sample_id"].nunique() != EXPECTED_SAMPLES:
            raise ValueError(f"{pipeline} per-sample metrics identity mismatch")
        frames[pipeline] = frame
    identity = ["sample_index", "sample_id", "question_index", "scene_id", "query"]
    old = frames["old_singlefilm"].rename(
        columns={
            column: f"old_{column}"
            for column in frames["old_singlefilm"].columns
            if column not in identity
        }
    )
    new = frames["hierfilm"].rename(
        columns={
            column: f"new_{column}"
            for column in frames["hierfilm"].columns
            if column not in identity
        }
    )
    paired = old.merge(new, on=identity, how="inner", validate="one_to_one")
    if len(paired) != EXPECTED_SAMPLES:
        raise ValueError("paired merge count mismatch")
    paired["mask_iou_delta"] = paired["new_mask_iou"] - paired["old_mask_iou"]
    paired["mask_area_delta_px"] = (
        paired["new_mask_area_native_px"] - paired["old_mask_area_native_px"]
    )
    output = run_dir / "evaluation" / "per_sample_pipeline_metrics.csv"
    paired.to_csv(output, index=False)
    return paired


def cluster_bootstrap(
    paired: pd.DataFrame,
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped = (
        paired.assign(
            delta=paired[f"new_{metric}"].astype(int)
            - paired[f"old_{metric}"].astype(int)
        )
        .groupby("scene_id", sort=True)
        .agg(delta=("delta", "sum"), count=("sample_id", "size"))
    )
    deltas = grouped["delta"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    cluster_count = len(grouped)
    values = np.empty(replicates, dtype=np.float64)
    chunk = 500
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        draws = rng.integers(0, cluster_count, size=(size, cluster_count))
        values[start : start + size] = (
            deltas[draws].sum(axis=1) / counts[draws].sum(axis=1)
        )
    return {
        "cluster_key": "scene_id/RGB frame",
        "clusters": cluster_count,
        "replicates": replicates,
        "seed": seed,
        "estimate": float(
            paired[f"new_{metric}"].mean() - paired[f"old_{metric}"].mean()
        ),
        "ci_95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def paired_metric(
    paired: pd.DataFrame,
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    old = paired[f"old_{metric}"].astype(bool)
    new = paired[f"new_{metric}"].astype(bool)
    both_correct = int((old & new).sum())
    recovered = int((~old & new).sum())
    harmful = int((old & ~new).sum())
    both_wrong = int((~old & ~new).sum())
    changed = recovered + harmful
    exact_p = 1.0 if not changed else float(binomtest(recovered, changed, 0.5).pvalue)
    return {
        "both_correct": both_correct,
        "recovered": recovered,
        "harmful": harmful,
        "both_wrong": both_wrong,
        "net": recovered - harmful,
        "outcome_changing_precision": (
            None if not changed else recovered / changed
        ),
        "old_numerator": int(old.sum()),
        "new_numerator": int(new.sum()),
        "denominator": len(paired),
        "absolute_change": float(new.mean() - old.mean()),
        "absolute_percentage_point_change": float(
            100.0 * (new.mean() - old.mean())
        ),
        "mcnemar_exact_two_sided_p": exact_p,
        "cluster_bootstrap": cluster_bootstrap(
            paired, metric, replicates=replicates, seed=seed
        ),
    }


def finite_distribution(series: pd.Series) -> dict[str, Any]:
    values = series.astype(float).to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p5": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p5": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def finite_spearman(left: pd.Series, right: pd.Series) -> dict[str, Any]:
    x = left.astype(float).to_numpy()
    y = right.astype(float).to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"n": int(x.size), "rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {
        "n": int(x.size),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def recompute_pipeline(frame: pd.DataFrame) -> dict[str, Any]:
    terminal = Counter(frame["terminal_status"])
    first = frame["first_correct_rank"].dropna().astype(int)
    return {
        "samples": len(frame),
        "terminal_counts": {
            "scored": int(terminal["scored"]),
            "valid_empty_mask": int(terminal["valid_empty_mask"]),
            "valid_empty_candidates": int(terminal["valid_empty_candidates"]),
            "failed": int(terminal["failed"]),
        },
        "top1_numerator": int(frame["top1_correct"].astype(bool).sum()),
        "top5_numerator": int(frame["top5_correct"].astype(bool).sum()),
        "oracle_all_numerator": int(frame["oracle_all"].astype(bool).sum()),
        "mrr": float(frame["reciprocal_rank"].sum() / len(frame)),
        "first_correct_rank_mean": None if first.empty else float(first.mean()),
        "first_correct_rank_median": None if first.empty else float(first.median()),
        "technical_failures": int(frame["technical_failure"].astype(bool).sum()),
        "raw_candidates": int(frame["raw_candidate_count"].sum()),
        "nms_candidates": int(frame["nms_candidate_count"].sum()),
        "scored_candidates": int(frame["scored_candidate_count"].sum()),
    }


def evaluate_verification(
    run_dir: Path,
    paired: pd.DataFrame,
    candidate_info: dict[str, Any],
    score_info: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "independent_source": "paired per-sample CSV plus Parquet metadata",
        "pipelines": {},
        "all_checks_passed": True,
        "checks": [],
    }
    for prefix, pipeline in (("old", "old_singlefilm"), ("new", "hierfilm")):
        columns = {
            column[len(prefix) + 1 :]: paired[column]
            for column in paired
            if column.startswith(prefix + "_")
        }
        frame = pd.DataFrame(columns)
        recomputed = recompute_pipeline(frame)
        metrics = load_json(
            run_dir / "evaluation" / f"{pipeline}_pipeline_metrics.json"
        )
        expected = {
            "samples": metrics["primary_denominator"],
            "terminal_counts": metrics["terminal_counts"],
            "top1_numerator": metrics["top1"]["numerator"],
            "top5_numerator": metrics["top5"]["numerator"],
            "oracle_all_numerator": metrics["oracle_all"]["numerator"],
            "mrr": metrics["mrr"]["value"],
            "technical_failures": metrics["technical_failure_count"],
        }
        for key, value in expected.items():
            observed = recomputed[key]
            if isinstance(value, float):
                passed = math.isclose(observed, value, rel_tol=0.0, abs_tol=1e-15)
            else:
                passed = observed == value
            result["checks"].append(
                {
                    "pipeline": pipeline,
                    "field": key,
                    "observed": observed,
                    "expected": value,
                    "passed": passed,
                }
            )
            result["all_checks_passed"] &= passed
        result["pipelines"][pipeline] = recomputed
    expected_score_rows = sum(
        item["scored_candidates"] for item in result["pipelines"].values()
    )
    score_pass = score_info["rows"] == expected_score_rows
    result["checks"].append(
        {
            "field": "combined_score_parquet_rows",
            "observed": score_info["rows"],
            "expected": expected_score_rows,
            "passed": score_pass,
        }
    )
    result["all_checks_passed"] &= score_pass
    for pipeline in ("old_singlefilm", "hierfilm"):
        expected_nms = result["pipelines"][pipeline]["nms_candidates"]
        observed_nms = candidate_info["counts"][f"{pipeline}_nms"]
        passed = expected_nms == observed_nms
        result["checks"].append(
            {
                "pipeline": pipeline,
                "field": "nms_parquet_rows",
                "observed": observed_nms,
                "expected": expected_nms,
                "passed": passed,
            }
        )
        result["all_checks_passed"] &= passed
    input_ids = set(pd.read_csv(run_dir / "input_manifest.csv")["sample_id"])
    paired_ids = set(paired["sample_id"])
    sample_identity_pass = (
        len(input_ids) == EXPECTED_SAMPLES
        and len(paired_ids) == EXPECTED_SAMPLES
        and input_ids == paired_ids
    )
    result["checks"].append(
        {
            "field": "frozen_manifest_sample_identity",
            "observed": {
                "input_unique": len(input_ids),
                "paired_unique": len(paired_ids),
                "symmetric_difference": len(input_ids ^ paired_ids),
            },
            "expected": {
                "input_unique": EXPECTED_SAMPLES,
                "paired_unique": EXPECTED_SAMPLES,
                "symmetric_difference": 0,
            },
            "passed": sample_identity_pass,
        }
    )
    result["all_checks_passed"] &= sample_identity_pass
    ordering_pass = bool(
        (
            paired["old_top1_correct"].astype(int)
            <= paired["old_top5_correct"].astype(int)
        ).all()
        and (
            paired["old_top5_correct"].astype(int)
            <= paired["old_oracle_all"].astype(int)
        ).all()
        and (
            paired["new_top1_correct"].astype(int)
            <= paired["new_top5_correct"].astype(int)
        ).all()
        and (
            paired["new_top5_correct"].astype(int)
            <= paired["new_oracle_all"].astype(int)
        ).all()
    )
    result["checks"].append(
        {
            "field": "top1_le_top5_le_oracle_per_sample",
            "observed": ordering_pass,
            "expected": True,
            "passed": ordering_pass,
        }
    )
    result["all_checks_passed"] &= ordering_pass
    score_frame = pq.read_table(
        run_dir / "scores" / "gqcnn_per_candidate.parquet",
        columns=[
            "pipeline",
            "sample_id",
            "candidate_id",
            "gqcnn_rank",
            "gqcnn_q_value",
        ],
    ).to_pandas()
    nms_frame = pq.read_table(
        run_dir / "candidates" / "dexnet_nms_candidates.parquet",
        columns=["pipeline", "sample_id", "candidate_id"],
    ).to_pandas()
    key = ["pipeline", "sample_id", "candidate_id"]
    duplicate_scores = int(score_frame.duplicated(key).sum())
    duplicate_nms = int(nms_frame.duplicated(key).sum())
    finite_q = bool(np.isfinite(score_frame["gqcnn_q_value"].to_numpy()).all())
    grouped_ranks = score_frame.groupby(["pipeline", "sample_id"])[
        "gqcnn_rank"
    ].agg(["size", "min", "max", "nunique"])
    continuous_ranks = bool(
        (
            (grouped_ranks["min"] == 1)
            & (grouped_ranks["max"] == grouped_ranks["size"])
            & (grouped_ranks["nunique"] == grouped_ranks["size"])
        ).all()
    )
    foreign_key_missing = int(
        score_frame.merge(
            nms_frame.drop_duplicates(key),
            on=key,
            how="left",
            indicator=True,
        )["_merge"]
        .eq("left_only")
        .sum()
    )
    relational_pass = (
        duplicate_scores == 0
        and duplicate_nms == 0
        and finite_q
        and continuous_ranks
        and foreign_key_missing == 0
    )
    result["checks"].append(
        {
            "field": "candidate_score_relational_integrity",
            "observed": {
                "duplicate_score_keys": duplicate_scores,
                "duplicate_nms_keys": duplicate_nms,
                "all_q_finite": finite_q,
                "continuous_one_based_ranks": continuous_ranks,
                "score_rows_missing_nms_parent": foreign_key_missing,
            },
            "expected": {
                "duplicate_score_keys": 0,
                "duplicate_nms_keys": 0,
                "all_q_finite": True,
                "continuous_one_based_ranks": True,
                "score_rows_missing_nms_parent": 0,
            },
            "passed": relational_pass,
        }
    )
    result["all_checks_passed"] &= relational_pass
    if not result["all_checks_passed"]:
        raise RuntimeError("independent metric verification failed")
    return result


def grasp_polygon(candidate: dict[str, Any]) -> np.ndarray:
    contacts = np.asarray(candidate["contact_points_uv"], dtype=np.float64)
    center = np.asarray(
        [candidate["center_u_px"], candidate["center_v_px"]], dtype=np.float64
    )
    axis = contacts[1] - contacts[0]
    width = float(np.linalg.norm(axis))
    axis /= width
    normal = np.asarray([-axis[1], axis[0]])
    height = 20.0
    return np.stack(
        [
            center + sx * width * 0.5 * axis + sy * height * 0.5 * normal
            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1))
        ]
    )


def draw_polygons(
    image: np.ndarray,
    polygons: list[np.ndarray],
    color: tuple[int, int, int],
    width: int = 3,
) -> np.ndarray:
    output = image.copy()
    for polygon in polygons:
        cv2.polylines(
            output,
            [np.rint(polygon).astype(np.int32)],
            isClosed=True,
            color=color,
            thickness=width,
            lineType=cv2.LINE_AA,
        )
    return output


def load_top1(run_dir: Path, pipeline: str, sample_id: str) -> dict[str, Any] | None:
    path = (
        run_dir
        / "scores"
        / pipeline
        / sample_id
        / "gqcnn_scored_candidates.json"
    )
    if not path.is_file():
        return None
    candidates = load_json(path)["candidates"]
    return None if not candidates else candidates[0]


def overlay_mask(
    rgb: np.ndarray,
    gt: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    output = rgb.astype(np.float32)
    output[gt] = 0.55 * output[gt] + 0.45 * np.asarray([0, 255, 80])
    output[prediction] = (
        0.55 * output[prediction] + 0.45 * np.asarray([255, 60, 40])
    )
    return np.clip(output, 0, 255).astype(np.uint8)


def select_qualitative(paired: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    old_top1 = paired.old_top1_correct.astype(bool)
    new_top1 = paired.new_top1_correct.astype(bool)
    old_top5 = paired.old_top5_correct.astype(bool)
    new_top5 = paired.new_top5_correct.astype(bool)
    old_oracle = paired.old_oracle_all.astype(bool)
    new_oracle = paired.new_oracle_all.astype(bool)
    categories = [
        ("successful_top1", new_top1),
        (
            "predicted_mask_empty",
            paired.old_failure_category.eq("predicted_mask_empty")
            | paired.new_failure_category.eq("predicted_mask_empty"),
        ),
        (
            "empty_candidates",
            paired.old_failure_category.eq("mask_nonempty_no_candidate")
            | paired.new_failure_category.eq("mask_nonempty_no_candidate"),
        ),
        (
            "no_correct_candidate",
            paired.old_failure_category.eq("candidate_pool_no_correct")
            | paired.new_failure_category.eq("candidate_pool_no_correct"),
        ),
        (
            "ranking_failure",
            paired.old_failure_category.eq("ranking_failure")
            | paired.new_failure_category.eq("ranking_failure"),
        ),
        ("top1_recovered", (~old_top1) & new_top1),
        ("top1_harmful", old_top1 & (~new_top1)),
        ("top5_recovered", (~old_top5) & new_top5),
        ("top5_harmful", old_top5 & (~new_top5)),
        ("oracle_recovered", (~old_oracle) & new_oracle),
        ("oracle_harmful", old_oracle & (~new_oracle)),
        (
            "mask_improved_grasp_worse",
            (paired.mask_iou_delta > 0.05)
            & old_top1
            & (~new_top1),
        ),
        (
            "mask_worse_grasp_better",
            (paired.mask_iou_delta < -0.05)
            & (~old_top1)
            & new_top1,
        ),
        (
            "candidate_coverage_changed",
            paired.old_nms_candidate_count != paired.new_nms_candidate_count,
        ),
    ]
    selected: list[tuple[str, pd.Series]] = []
    seen: set[str] = set()
    for category_index, (category, mask) in enumerate(categories):
        subset = paired.loc[mask].copy()
        if subset.empty:
            continue
        subset = subset.sample(
            n=min(3, len(subset)),
            random_state=20260728 + category_index,
        )
        for _, row in subset.sort_values("sample_id").iterrows():
            if row.sample_id in seen:
                continue
            seen.add(row.sample_id)
            selected.append((category, row))
    if len(selected) < 25:
        remaining = paired.loc[~paired.sample_id.isin(seen)].sample(
            frac=1.0, random_state=20260728
        )
        for _, row in remaining.iterrows():
            selected.append(("fixed_seed_random_fill", row))
            seen.add(row.sample_id)
            if len(selected) >= 25:
                break
    return selected[:32]


def depth_valid_visual(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.quantile(depth[valid], [0.02, 0.98])
        if high <= low:
            high = low + 1.0
        normalized[valid] = np.clip(
            255.0 * (depth[valid] - low) / (high - low), 0, 255
        ).astype(np.uint8)
    colored = cv2.cvtColor(
        cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS),
        cv2.COLOR_BGR2RGB,
    )
    colored[~valid] = np.asarray([90, 0, 90], dtype=np.uint8)
    cv2.putText(
        colored,
        "DEPTH (purple=invalid)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return colored


def draw_ranked_candidates(
    image: np.ndarray,
    candidates: list[dict[str, Any]],
    evaluation_by_id: dict[str, dict[str, Any]],
    *,
    panel_name: str,
) -> np.ndarray:
    output = image.copy()
    ordered = sorted(
        candidates, key=lambda item: int(item["gqcnn_rank"]), reverse=True
    )
    for candidate in ordered:
        rank = int(candidate["gqcnn_rank"])
        evaluation = evaluation_by_id.get(str(candidate["candidate_id"]), {})
        if rank == 1:
            color, thickness = (255, 0, 255), 5
        elif rank <= 5:
            color = (40, 235, 40) if evaluation.get("candidate_success") else (255, 170, 0)
            thickness = 3
        else:
            color, thickness = (135, 175, 255), 1
        output = draw_polygons(
            output, [grasp_polygon(candidate)], color, width=thickness
        )
        if rank <= 5:
            center = (
                int(round(float(candidate["center_u_px"]))),
                int(round(float(candidate["center_v_px"]))),
            )
            cv2.putText(
                output,
                f"r{rank}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        output,
        panel_name,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def top5_summary(
    candidates: list[dict[str, Any]],
    evaluation_by_id: dict[str, dict[str, Any]],
) -> str:
    values = []
    for candidate in sorted(candidates, key=lambda item: int(item["gqcnn_rank"]))[:5]:
        evaluation = evaluation_by_id.get(str(candidate["candidate_id"]), {})
        values.append(
            f"r{int(candidate['gqcnn_rank'])}:"
            f"q={float(candidate['gqcnn_q_value']):.5g},"
            f"ok={bool(evaluation.get('candidate_success', False))}"
        )
    return " | ".join(values) if values else "no candidates"


def build_qualitative(
    run_dir: Path,
    paired: pd.DataFrame,
    annotations_path: Path,
) -> dict[str, Any]:
    audit_root = run_dir / "artifacts" / "qualitative_audit"
    cases_root = audit_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    annotations = load_json(annotations_path)["data"]
    by_question = {int(row["question_index"]): row for row in annotations}
    records = []
    selected = select_qualitative(paired)
    selected_ids = [str(row.sample_id) for _, row in selected]
    evaluation_maps: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for pipeline in ("old_singlefilm", "hierfilm"):
        table = pq.read_table(
            run_dir
            / "evaluation"
            / f"{pipeline}_gqcnn_per_candidate.parquet",
            filters=[("sample_id", "in", selected_ids)],
            columns=[
                "sample_id",
                "candidate_id",
                "candidate_success",
                "rectangle_iou",
                "angle_difference_deg",
            ],
        )
        pipeline_map: dict[str, dict[str, dict[str, Any]]] = {}
        for evaluation_row in table.to_pylist():
            pipeline_map.setdefault(evaluation_row["sample_id"], {})[
                evaluation_row["candidate_id"]
            ] = evaluation_row
        evaluation_maps[pipeline] = pipeline_map
    for index, (category, row) in enumerate(selected):
        sample_id = row.sample_id
        template = (
            run_dir
            / "masks"
            / "hierfilm"
            / "bundles"
            / sample_id
        )
        rgb = np.asarray(Image.open(template / "color.png").convert("RGB"))
        annotation = by_question[int(row.question_index)]
        _, image_name = row.scene_id.split(",", 1)
        source_mask = (
            Path(load_json(template / "metadata.json")["source_rgb"]).parent.parent
            / "seg_mask_instances_combi"
            / image_name
        )
        instance = np.asarray(Image.open(source_mask))
        gt = instance == int(annotation["answer"])
        old_mask = (
            np.asarray(
                Image.open(
                    run_dir
                    / "masks"
                    / "old_singlefilm"
                    / "predictions"
                    / sample_id
                    / "predicted_mask_original_resolution.png"
                )
            )
            > 0
        )
        new_mask = (
            np.asarray(
                Image.open(
                    run_dir
                    / "masks"
                    / "hierfilm"
                    / "predictions"
                    / sample_id
                    / "predicted_mask_original_resolution.png"
                )
            )
            > 0
        )
        gt_polygons = [np.asarray(value, dtype=np.float64) for value in annotation["grasps"]]
        old_payload_path = (
            run_dir
            / "scores"
            / "old_singlefilm"
            / sample_id
            / "gqcnn_scored_candidates.json"
        )
        new_payload_path = (
            run_dir
            / "scores"
            / "hierfilm"
            / sample_id
            / "gqcnn_scored_candidates.json"
        )
        old_candidates = (
            load_json(old_payload_path)["candidates"]
            if old_payload_path.is_file()
            else []
        )
        new_candidates = (
            load_json(new_payload_path)["candidates"]
            if new_payload_path.is_file()
            else []
        )
        old_evaluations = evaluation_maps["old_singlefilm"].get(sample_id, {})
        new_evaluations = evaluation_maps["hierfilm"].get(sample_id, {})
        gt_panel = draw_polygons(
            overlay_mask(rgb, gt, np.zeros_like(gt)),
            gt_polygons,
            (0, 235, 80),
            width=3,
        )
        cv2.putText(
            gt_panel,
            "RGB + GT MASK/GRASPS",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        depth_panel = depth_valid_visual(
            np.asarray(Image.open(template / "depth.png"), dtype=np.float32)
        )
        old_panel = draw_ranked_candidates(
            overlay_mask(rgb, gt, old_mask),
            old_candidates,
            old_evaluations,
            panel_name="OLD: all candidates + Top5",
        )
        new_panel = draw_ranked_candidates(
            overlay_mask(rgb, gt, new_mask),
            new_candidates,
            new_evaluations,
            panel_name="NEW: all candidates + Top5",
        )
        panels = np.concatenate(
            (gt_panel, depth_panel, old_panel, new_panel), axis=1
        )
        canvas = Image.new(
            "RGB", (panels.shape[1], panels.shape[0] + 150), "white"
        )
        canvas.paste(Image.fromarray(panels), (0, 150))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{index:02d} {category} | {sample_id} | {row.query}\n"
            f"OLD {row.old_failure_category}, maskIoU={row.old_mask_iou:.3f}: "
            f"{top5_summary(old_candidates, old_evaluations)}\n"
            f"NEW {row.new_failure_category}, maskIoU={row.new_mask_iou:.3f}: "
            f"{top5_summary(new_candidates, new_evaluations)}"
        )
        draw.multiline_text((12, 10), title, fill="black", font=ImageFont.load_default())
        output = cases_root / f"{index:02d}_{category}_{sample_id}.png"
        canvas.save(output)
        why = (
            f"Category={category}; mask IoU delta={row.mask_iou_delta:+.4f}; "
            f"old/new Top1={bool(row.old_top1_correct)}/{bool(row.new_top1_correct)}, "
            f"Top5={bool(row.old_top5_correct)}/{bool(row.new_top5_correct)}, "
            f"Oracle={bool(row.old_oracle_all)}/{bool(row.new_oracle_all)}. "
            "Judgment uses frozen corrected rectangle IoU>0.25 and periodic "
            "angle difference<=30° against the same GT rectangle."
        )
        records.append(
            {
                "index": index,
                "category": category,
                "sample_id": sample_id,
                "question_index": int(row.question_index),
                "scene_id": row.scene_id,
                "query": row.query,
                "old_mask_iou": float(row.old_mask_iou),
                "new_mask_iou": float(row.new_mask_iou),
                "old_top1_correct": bool(row.old_top1_correct),
                "new_top1_correct": bool(row.new_top1_correct),
                "old_top5_correct": bool(row.old_top5_correct),
                "new_top5_correct": bool(row.new_top5_correct),
                "old_oracle_all": bool(row.old_oracle_all),
                "new_oracle_all": bool(row.new_oracle_all),
                "old_failure_category": row.old_failure_category,
                "new_failure_category": row.new_failure_category,
                "old_top5_summary": top5_summary(
                    old_candidates, old_evaluations
                ),
                "new_top5_summary": top5_summary(
                    new_candidates, new_evaluations
                ),
                "why": why,
                "image": str(output),
                "image_sha256": sha256_file(output),
            }
        )
    pd.DataFrame(records).to_csv(audit_root / "qualitative_cases.csv", index=False)
    markdown = [
        "# Qualitative audit",
        "",
        "Green rectangles are GT annotations. Mask overlays use GT green and "
        "prediction red. All post-NMS candidates are light blue; ranks 2–5 are "
        "green when correct and orange when wrong; Top-1 is thick magenta. "
        "The second panel visualizes valid depth (purple means invalid).",
        "",
    ]
    for record in records:
        markdown.extend(
            [
                f"## {record['index']:02d} {record['category']} — {record['sample_id']}",
                "",
                record["why"],
                "",
                f"![case]({record['image']})",
                "",
            ]
        )
    write_text(audit_root / "qualitative_audit.md", "\n".join(markdown))
    widths = 1920
    thumbs = []
    for record in records:
        image = Image.open(record["image"]).convert("RGB")
        image.thumbnail((widths, 570))
        thumbs.append(image.copy())
        image.close()
    sheet = Image.new("RGB", (widths, sum(image.height for image in thumbs)), "white")
    y = 0
    for image in thumbs:
        sheet.paste(image, (0, y))
        y += image.height
    contact = audit_root / "contact_sheet.png"
    sheet.save(contact)
    return {
        "cases": len(records),
        "categories": dict(Counter(record["category"] for record in records)),
        "contact_sheet": str(contact),
        "contact_sheet_sha256": sha256_file(contact),
    }


def make_reports(
    run_dir: Path,
    paired: pd.DataFrame,
    comparison: dict[str, Any],
    verification: dict[str, Any],
    qualitative: dict[str, Any],
) -> None:
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = []
    for pipeline in ("old_singlefilm", "hierfilm"):
        metrics = load_json(
            run_dir / "evaluation" / f"{pipeline}_pipeline_metrics.json"
        )
        rows.append(
            {
                "pipeline": pipeline,
                "denominator": metrics["primary_denominator"],
                "scored": metrics["terminal_counts"]["scored"],
                "valid_empty_mask": metrics["terminal_counts"]["valid_empty_mask"],
                "valid_empty_candidates": metrics["terminal_counts"][
                    "valid_empty_candidates"
                ],
                "failed": metrics["terminal_counts"]["failed"],
                "top1_numerator": metrics["top1"]["numerator"],
                "top1": metrics["top1"]["value"],
                "top5_numerator": metrics["top5"]["numerator"],
                "top5": metrics["top5"]["value"],
                "oracle_numerator": metrics["oracle_all"]["numerator"],
                "oracle_all": metrics["oracle_all"]["value"],
                "mrr": metrics["mrr"]["value"],
                "nms_candidates": metrics["per_candidate_rows"],
            }
        )
    pd.DataFrame(rows).to_csv(
        reports / "hierfilm_vs_singlefilm_pipeline.csv", index=False
    )
    flow_rows = []
    for row in rows:
        metrics = load_json(
            run_dir
            / "evaluation"
            / f"{row['pipeline']}_pipeline_metrics.json"
        )
        flow_rows.append({"pipeline": row["pipeline"], **metrics["failure_flow"]})
    pd.DataFrame(flow_rows).to_csv(
        reports / "failure_flow_comparison.csv", index=False
    )
    runtime_lines = ["# Runtime and coverage", ""]
    for pipeline in ("old_singlefilm", "hierfilm"):
        mask = load_json(run_dir / "masks" / pipeline / "MASKS_COMPLETED.json")
        candidate = load_json(run_dir / "candidates" / pipeline / "progress.json")
        score = load_json(run_dir / "scores" / pipeline / "progress.json")
        evaluation = load_json(
            run_dir / "evaluation" / f"{pipeline}_pipeline_metrics.json"
        )
        runtime_lines.extend(
            [
                f"## {pipeline}",
                "",
                f"- Mask stage: {mask['elapsed_seconds']:.3f} s",
                "- Candidate terminal samples: "
                f"{int(candidate['success_nonempty']) + int(candidate['success_empty'])}",
                f"- Candidate wall time: {candidate['elapsed_seconds']:.3f} s",
                "- Candidate throughput: "
                f"{EXPECTED_SAMPLES / float(candidate['elapsed_seconds']):.3f} samples/s",
                f"- GQ-CNN wall time: {score['elapsed_seconds']:.3f} s",
                "- GQ-CNN throughput: "
                f"{score['throughput_candidates_per_second']:.3f} candidates/s",
                f"- Corrected evaluation: {evaluation['total_evaluation_seconds']:.3f} s",
                "",
            ]
        )
    write_text(reports / "runtime_and_coverage.md", "\n".join(runtime_lines))
    old = rows[0]
    new = rows[1]
    old_eval = load_json(
        run_dir / "evaluation" / "old_singlefilm_pipeline_metrics.json"
    )
    new_eval = load_json(
        run_dir / "evaluation" / "hierfilm_pipeline_metrics.json"
    )
    old_mask = load_json(
        run_dir / "masks" / "old_singlefilm" / "MASKS_COMPLETED.json"
    )["metrics"]
    new_mask = load_json(
        run_dir / "masks" / "hierfilm" / "MASKS_COMPLETED.json"
    )["metrics"]
    top1 = comparison["paired"]["top1_correct"]
    mask_rows = []
    for label, metrics in (
        ("Old single-FiLM", old_mask),
        ("Five-stage repeated FiLM", new_mask),
    ):
        mask_rows.append(
            f"| {label} | {100*metrics['mean_iou']:.4f}% | "
            f"{100*metrics['median_iou']:.4f}% | "
            f"{metrics['p_at_50_numerator']}/{metrics['p_at_50_denominator']} "
            f"({100*metrics['p_at_50']:.4f}%) | "
            f"{metrics['p_at_60_numerator']}/{metrics['p_at_60_denominator']} "
            f"({100*metrics['p_at_60']:.4f}%) | "
            f"{metrics['p_at_70_numerator']}/{metrics['p_at_70_denominator']} "
            f"({100*metrics['p_at_70']:.4f}%) | "
            f"{metrics['p_at_80_numerator']}/{metrics['p_at_80_denominator']} "
            f"({100*metrics['p_at_80']:.4f}%) | "
            f"{metrics['p_at_90_numerator']}/{metrics['p_at_90_denominator']} "
            f"({100*metrics['p_at_90']:.4f}%) | {metrics['empty_masks']} |"
        )
    paired_rows = []
    for label, metric in (
        ("Top-1", "top1_correct"),
        ("Top-5", "top5_correct"),
        ("Oracle@All", "oracle_all"),
    ):
        value = comparison["paired"][metric]
        paired_rows.append(
            f"| {label} | {value['both_correct']} | {value['recovered']} | "
            f"{value['harmful']} | {value['both_wrong']} | {value['net']} | "
            f"{value['outcome_changing_precision']} | "
            f"{value['absolute_percentage_point_change']:+.4f} | "
            f"{value['mcnemar_exact_two_sided_p']:.8g} | "
            f"{value['cluster_bootstrap']['ci_95']} |"
        )
    flow_rows_md = []
    for label, metrics in (
        ("Old single-FiLM", old_eval),
        ("Five-stage repeated FiLM", new_eval),
    ):
        flow = metrics["failure_flow"]
        flow_rows_md.append(
            f"| {label} | {flow['predicted_mask_empty']} | "
            f"{flow['mask_nonempty_no_candidate']} | "
            f"{flow['candidate_pool_no_correct']} | "
            f"{flow['ranking_failure']} | {flow['top1_correct']} | "
            f"{flow['technical_failure']} |"
        )
    final = [
        "# Hierarchical repeated-FiLM → Dex-Net → GQ-CNN final report",
        "",
        "This is an offline modular pipeline comparison. Correctness means "
        "2D consistency with OCID-VLG rectangles, not physical grasp success.",
        "",
        "## Frozen identities",
        "",
        f"- Hierarchical checkpoint SHA-256: `{load_json(run_dir / 'run_manifest.json')['inputs']['hierfilm_checkpoint']['sha256']}`",
        f"- Official unique test manifest SHA-256: `{load_json(run_dir / 'run_manifest.json')['inputs']['test_manifest']['sha256']}`",
        "- GQ-CNN: official GQCNN-2.1 neural scoring on frozen post-NMS candidates; "
        "this is not the full Dex-Net CEM policy.",
        "",
        "## Visual grounding",
        "",
        "| Pipeline | mIoU | median IoU | P@50 | P@60 | P@70 | P@80 | P@90 | Empty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *mask_rows,
        "",
        "## Primary all-7,675 results",
        "",
        "| Pipeline | Top-1 | Top-5 | Oracle@All | MRR |",
        "|---|---:|---:|---:|---:|",
        f"| Old single-FiLM matched downstream | {old['top1_numerator']}/{old['denominator']} ({100*old['top1']:.3f}%) | {old['top5_numerator']}/{old['denominator']} ({100*old['top5']:.3f}%) | {old['oracle_numerator']}/{old['denominator']} ({100*old['oracle_all']:.3f}%) | {old['mrr']:.6f} |",
        f"| Five-stage repeated FiLM | {new['top1_numerator']}/{new['denominator']} ({100*new['top1']:.3f}%) | {new['top5_numerator']}/{new['denominator']} ({100*new['top5']:.3f}%) | {new['oracle_numerator']}/{new['denominator']} ({100*new['oracle_all']:.3f}%) | {new['mrr']:.6f} |",
        "",
        "Conditional scored-only results use only samples with at least one "
        "post-NMS candidate:",
        "",
        f"- Old: Top-1 {old_eval['scored_only']['top1_numerator']}/{old_eval['scored_only']['denominator']} ({100*old_eval['scored_only']['top1']:.3f}%), "
        f"Top-5 {old_eval['scored_only']['top5_numerator']}/{old_eval['scored_only']['denominator']} ({100*old_eval['scored_only']['top5']:.3f}%), "
        f"Oracle@All {old_eval['scored_only']['oracle_all_numerator']}/{old_eval['scored_only']['denominator']} ({100*old_eval['scored_only']['oracle_all']:.3f}%).",
        f"- New: Top-1 {new_eval['scored_only']['top1_numerator']}/{new_eval['scored_only']['denominator']} ({100*new_eval['scored_only']['top1']:.3f}%), "
        f"Top-5 {new_eval['scored_only']['top5_numerator']}/{new_eval['scored_only']['denominator']} ({100*new_eval['scored_only']['top5']:.3f}%), "
        f"Oracle@All {new_eval['scored_only']['oracle_all_numerator']}/{new_eval['scored_only']['denominator']} ({100*new_eval['scored_only']['oracle_all']:.3f}%).",
        "",
        "## Paired comparison",
        "",
        "| Metric | Both correct | Recovered | Harmful | Both wrong | Net | Outcome-changing precision | Δ pp | McNemar p | Scene bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *paired_rows,
        "",
        "## Mutually exclusive failure flow",
        "",
        "| Pipeline | Empty mask | Non-empty/no candidate | Pool has no correct | Ranking failure | Top-1 correct | Technical failure |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *flow_rows_md,
        "",
        "## Candidate and ranking diagnostics",
        "",
        f"- Old raw candidates: {verification['pipelines']['old_singlefilm']['raw_candidates']}; "
        f"post-NMS/scored: {verification['pipelines']['old_singlefilm']['nms_candidates']}.",
        f"- New raw candidates: {verification['pipelines']['hierfilm']['raw_candidates']}; "
        f"post-NMS/scored: {verification['pipelines']['hierfilm']['nms_candidates']}.",
        f"- Old raw per-sample min/P5/median/mean/P95/max: "
        f"{old_eval['raw_candidate_count']['min']}/"
        f"{old_eval['raw_candidate_count']['p5']}/"
        f"{old_eval['raw_candidate_count']['median']}/"
        f"{old_eval['raw_candidate_count']['mean']}/"
        f"{old_eval['raw_candidate_count']['p95']}/"
        f"{old_eval['raw_candidate_count']['max']}; post-NMS: "
        f"{old_eval['nms_candidate_count']['min']}/"
        f"{old_eval['nms_candidate_count']['p5']}/"
        f"{old_eval['nms_candidate_count']['median']}/"
        f"{old_eval['nms_candidate_count']['mean']}/"
        f"{old_eval['nms_candidate_count']['p95']}/"
        f"{old_eval['nms_candidate_count']['max']}.",
        f"- New raw per-sample min/P5/median/mean/P95/max: "
        f"{new_eval['raw_candidate_count']['min']}/"
        f"{new_eval['raw_candidate_count']['p5']}/"
        f"{new_eval['raw_candidate_count']['median']}/"
        f"{new_eval['raw_candidate_count']['mean']}/"
        f"{new_eval['raw_candidate_count']['p95']}/"
        f"{new_eval['raw_candidate_count']['max']}; post-NMS: "
        f"{new_eval['nms_candidate_count']['min']}/"
        f"{new_eval['nms_candidate_count']['p5']}/"
        f"{new_eval['nms_candidate_count']['median']}/"
        f"{new_eval['nms_candidate_count']['mean']}/"
        f"{new_eval['nms_candidate_count']['p95']}/"
        f"{new_eval['nms_candidate_count']['max']}.",
        f"- Old later-correct ranks 2–5 / after 5: "
        f"{old_eval['top1_wrong_but_later_correct']['rank_2_to_5']} / "
        f"{old_eval['top1_wrong_but_later_correct']['rank_after_5']}.",
        f"- New later-correct ranks 2–5 / after 5: "
        f"{new_eval['top1_wrong_but_later_correct']['rank_2_to_5']} / "
        f"{new_eval['top1_wrong_but_later_correct']['rank_after_5']}.",
        f"- Mask-area distributions and valid-depth/candidate correlations are "
        f"frozen in `evaluation/paired_comparison.json`; new valid-depth versus "
        f"post-NMS count Spearman rho is "
        f"{comparison['mask']['new_valid_depth_vs_nms_candidates_spearman']['rho']}.",
        "",
        "## Provenance qualification",
        "",
        "The old model output was recoverable and was rerun under the same current "
        "nearest-mask mapping, sample-derived Dex-Net seed, GQCNN-2.1 scorer, "
        "tie-break, corrected evaluator, and all-sample denominator. The model "
        "training optimizer and checkpoint-selection conditions differ, so this "
        "is not a pure causal estimate of repeated FiLM.",
        "",
        f"Independent verification passed: `{verification['all_checks_passed']}`. "
        f"Qualitative audit cases: {qualitative['cases']}.",
        "",
    ]
    write_text(reports / "hierfilm_dexnet_gqcnn_final_report.md", "\n".join(final))


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    run_dir = args.run_dir.expanduser().resolve()
    annotations = args.official_annotations.expanduser().resolve()
    if (run_dir / "COMPLETED").exists():
        raise FileExistsError("formal experiment is already completed")
    combined_masks = combine_mask_metadata(run_dir)
    candidate_info = write_candidate_parquets(run_dir)
    score_info = combine_scores(run_dir)
    paired = load_paired(run_dir)
    comparison = {
        "schema_version": 1,
        "description": (
            "Old single-FiLM predicted mask versus five-stage repeated-FiLM "
            "Standard predicted mask under the frozen current downstream protocol."
        ),
        "causal_claim_allowed": False,
        "causal_limitation": (
            "Optimizer and validation-checkpoint-selection conditions differ "
            "between the two segmenters."
        ),
        "archived_historical": ARCHIVED_HISTORICAL,
        "old_baseline_status": (
            "recomputed matched downstream from recoverable old model-resolution "
            "binary masks; not a matched training control"
        ),
        "paired": {
            metric: paired_metric(
                paired,
                metric,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + index,
            )
            for index, metric in enumerate(
                ("top1_correct", "top5_correct", "oracle_all")
            )
        },
        "mask": {
            "old_mean_iou": float(paired.old_mask_iou.mean()),
            "new_mean_iou": float(paired.new_mask_iou.mean()),
            "mean_iou_delta": float(
                paired.new_mask_iou.mean() - paired.old_mask_iou.mean()
            ),
            "old_empty": int((paired.old_terminal_status == "valid_empty_mask").sum()),
            "new_empty": int((paired.new_terminal_status == "valid_empty_mask").sum()),
            "old_area_native_px": finite_distribution(
                paired.old_mask_area_native_px
            ),
            "new_area_native_px": finite_distribution(
                paired.new_mask_area_native_px
            ),
            "old_valid_depth_fraction": finite_distribution(
                paired.old_valid_depth_fraction
            ),
            "new_valid_depth_fraction": finite_distribution(
                paired.new_valid_depth_fraction
            ),
            "old_valid_depth_vs_nms_candidates_spearman": finite_spearman(
                paired.old_valid_depth_fraction,
                paired.old_nms_candidate_count,
            ),
            "new_valid_depth_vs_nms_candidates_spearman": finite_spearman(
                paired.new_valid_depth_fraction,
                paired.new_nms_candidate_count,
            ),
        },
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    comparison_path = run_dir / "evaluation" / "paired_comparison.json"
    write_json(comparison_path, comparison)
    verification = evaluate_verification(
        run_dir, paired, candidate_info, score_info
    )
    verification.update(
        {
            "candidate_parquets": candidate_info,
            "score_parquet": score_info,
            "combined_mask_rows": len(combined_masks),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    verification_path = (
        run_dir / "evaluation" / "independent_metric_verification.json"
    )
    write_json(verification_path, verification)
    old_metrics = load_json(
        run_dir / "evaluation" / "old_singlefilm_pipeline_metrics.json"
    )
    old_metrics["baseline_provenance_status"] = comparison["old_baseline_status"]
    old_metrics["archived_historical"] = ARCHIVED_HISTORICAL
    write_json(
        run_dir / "evaluation" / "old_singlefilm_metrics_recomputed.json",
        old_metrics,
    )
    shutil.copy2(
        run_dir / "evaluation" / "hierfilm_pipeline_metrics.json",
        run_dir / "evaluation" / "hierfilm_pipeline_metrics.json.copy-verification",
    )
    # The canonical file already has the required name. The byte-for-byte copy
    # provides an explicit pre-report snapshot without changing it.
    qualitative = build_qualitative(run_dir, paired, annotations)
    make_reports(
        run_dir, paired, comparison, verification, qualitative
    )
    required = [
        run_dir / "frozen_protocol.yaml",
        run_dir / "run_manifest.json",
        run_dir / "environment.json",
        run_dir / "input_manifest.csv",
        run_dir / "sample_status.jsonl",
        run_dir / "masks" / "per_sample_mask_metadata.csv",
        run_dir / "candidates" / "dexnet_raw_candidates.parquet",
        run_dir / "candidates" / "dexnet_nms_candidates.parquet",
        run_dir / "scores" / "gqcnn_per_candidate.parquet",
        run_dir / "evaluation" / "per_sample_pipeline_metrics.csv",
        run_dir / "evaluation" / "hierfilm_pipeline_metrics.json",
        run_dir / "evaluation" / "old_singlefilm_metrics_recomputed.json",
        run_dir / "evaluation" / "paired_comparison.json",
        run_dir / "evaluation" / "independent_metric_verification.json",
        run_dir / "reports" / "hierfilm_dexnet_gqcnn_final_report.md",
        run_dir / "reports" / "hierfilm_vs_singlefilm_pipeline.csv",
        run_dir / "reports" / "failure_flow_comparison.csv",
        run_dir / "reports" / "runtime_and_coverage.md",
        run_dir / "artifacts" / "qualitative_audit" / "contact_sheet.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required outputs missing: {missing}")
    for pipeline in ("old_singlefilm", "hierfilm"):
        metrics = load_json(
            run_dir / "evaluation" / f"{pipeline}_pipeline_metrics.json"
        )
        if metrics["technical_failure_count"] != 0:
            raise RuntimeError(f"{pipeline} has technical failures")
        if sum(metrics["terminal_counts"].values()) != EXPECTED_SAMPLES:
            raise RuntimeError(f"{pipeline} terminal accounting mismatch")
    final_manifest_path = run_dir / "final_output_manifest.json"
    write_json(
        final_manifest_path,
        {
            "schema_version": 1,
            "files": file_manifest(run_dir, required),
            "candidate_parquets": candidate_info,
            "score_parquet": score_info,
            "qualitative": qualitative,
            "finalization_seconds": time.perf_counter() - started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    completion = {
        "status": "SUCCESS",
        "samples_per_pipeline": EXPECTED_SAMPLES,
        "technical_failures": 0,
        "independent_verification_passed": verification["all_checks_passed"],
        "final_output_manifest": str(final_manifest_path),
        "final_output_manifest_sha256": sha256_file(final_manifest_path),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "COMPLETED", completion)
    print(json.dumps(completion, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
