"""Independent authoritative recomputation of frozen G1/C1 baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.grasping.common.evaluator import evaluate_ocid_predictions
from src.grasping.common.types import Grasp4DoF

from .audit import source_candidate_path, source_sample_path
from .constants import SOURCE_INFERENCE_COLUMNS
from .io import atomic_json, atomic_parquet, sha256_file


def _pool(frame: pd.DataFrame) -> list[Grasp4DoF]:
    selected = frame.loc[frame["rank"].astype(int) <= 5].sort_values(
        ["rank", "candidate_id"], kind="mergesort"
    )
    return [
        Grasp4DoF(
            center_x=float(row.center_x),
            center_y=float(row.center_y),
            angle_deg=float(row.angle_deg),
            width_px=float(row.width_px),
            height_px=float(row.height_px),
            score=float(row.score),
            candidate_id=str(row.candidate_id),
        )
        for row in selected.itertuples(index=False)
    ]


def _gt_rectangles(value: Any, *, sample_id: str) -> list[list[list[float]]]:
    rectangles: list[list[list[float]]] = []
    for rectangle in value:
        points = [np.asarray(point, dtype=np.float64).reshape(-1) for point in rectangle]
        if len(points) != 4 or any(point.shape != (2,) for point in points):
            raise ValueError(f"{sample_id}: malformed frozen GT rectangle")
        array = np.stack(points)
        if not np.isfinite(array).all():
            raise ValueError(f"{sample_id}: non-finite frozen GT rectangle")
        rectangles.append(array.tolist())
    if not rectangles:
        raise ValueError(f"{sample_id}: no frozen GT rectangles")
    return rectangles


def recompute_backend(source_run: Path, split: str, backend: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_name = "validation" if split == "validation" else "test"
    deployment = pd.read_parquet(
        source_run / "manifests" / f"{manifest_name}_samples.parquet",
        columns=["sample_id", "scene_id"],
    )
    labels = pd.read_parquet(
        source_run / "manifests" / f"{manifest_name}_labels.parquet",
        columns=["sample_id", "scene_id", "gt_grasp_rectangles"],
    )
    if deployment["sample_id"].astype(str).tolist() != labels["sample_id"].astype(str).tolist():
        raise RuntimeError(f"{split}: deployment/label ordering mismatch")
    candidates = pd.read_parquet(
        source_candidate_path(source_run, split, backend),
        columns=list(SOURCE_INFERENCE_COLUMNS),
    )
    by_sample = {str(sample_id): group for sample_id, group in candidates.groupby("sample_id", sort=False)}
    saved = pd.read_parquet(
        source_sample_path(source_run, split, backend),
        columns=["sample_id", "j_at_1", "j_at_5", "top1_candidate_id"],
    ).set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for label in labels.itertuples(index=False):
        sample_id = str(label.sample_id)
        source_group = by_sample.get(sample_id, candidates.iloc[:0])
        pool = _pool(source_group)
        outcome = evaluate_ocid_predictions(
            pool, _gt_rectangles(label.gt_grasp_rectangles, sample_id=sample_id)
        )
        candidate_correct = {
            item.candidate_id: bool(item.candidate_success) for item in outcome.candidates
        }
        top1_id = None if not pool else pool[0].candidate_id
        recoverable = bool((not outcome.j_at_1) and outcome.j_at_5)
        unrecoverable = bool(not outcome.j_at_5)
        row = {
            "backend": backend,
            "split": split,
            "sample_id": sample_id,
            "scene_id": str(label.scene_id),
            "candidate_count": len(pool),
            "top1_candidate_id": top1_id,
            "top1_correct": bool(outcome.j_at_1),
            "top5_any_correct": bool(outcome.j_at_5),
            "recoverable_error": recoverable,
            "unrecoverable_error": unrecoverable,
            "candidate_correctness_json": json.dumps(candidate_correct, sort_keys=True, separators=(",", ":")),
        }
        saved_row = saved.loc[sample_id]
        if bool(saved_row["j_at_1"]) != row["top1_correct"] or bool(saved_row["j_at_5"]) != row["top5_any_correct"]:
            raise RuntimeError(f"{backend}/{split}/{sample_id}: independent evaluator mismatch")
        saved_top1 = None if pd.isna(saved_row["top1_candidate_id"]) else str(saved_row["top1_candidate_id"])
        if saved_top1 != top1_id:
            raise RuntimeError(f"{backend}/{split}/{sample_id}: Top-1 identity mismatch")
        rows.append(row)
    frame = pd.DataFrame(rows)
    total = len(frame)
    metrics = {
        "backend": backend,
        "split": split,
        "sample_count": total,
        "source_candidate_rows": int(len(candidates)),
        "frozen_top5_candidate_rows": int((candidates["rank"].astype(int) <= 5).sum()),
        "non_empty_count": int((frame["candidate_count"] > 0).sum()),
        "non_empty_rate": float((frame["candidate_count"] > 0).mean()),
        "top1_correct_count": int(frame["top1_correct"].sum()),
        "top5_any_correct_count": int(frame["top5_any_correct"].sum()),
        "j_at_1": float(frame["top1_correct"].mean()),
        "j_at_5": float(frame["top5_any_correct"].mean()),
        "recoverable_error_count": int(frame["recoverable_error"].sum()),
        "top5_no_correct_count": int(frame["unrecoverable_error"].sum()),
        "exact_match_to_saved_source": True,
        "evaluator_contract": "rasterized IoU > 0.25 AND 180-degree periodic angle error <= 30 degrees against the same GT grasp",
        "metric_claim": "OCID-VLG offline 2D grasp-rectangle consistency",
    }
    return frame, metrics


def run_baseline_recompute(source_run: Path, run_dir: Path) -> dict[str, Any]:
    parts: list[pd.DataFrame] = []
    results: dict[str, Any] = {}
    for split in ("validation", "test"):
        for backend in ("G1", "C1"):
            frame, metrics = recompute_backend(source_run, split, backend)
            parts.append(frame)
            results[f"{backend}_{split}"] = metrics
    per_sample = pd.concat(parts, ignore_index=True)
    path = run_dir / "baseline_per_sample.parquet"
    atomic_parquet(path, per_sample)
    results["baseline_per_sample_sha256"] = sha256_file(path)
    results["all_exact_match"] = True
    results["formal_pristine_warning"] = (
        "The locked source experiment already published test outcomes, and this required audit independently recomputes them. "
        "The new API experiment is therefore not a pristine first look at the test set; test labels remain prohibited from API selection and requests."
    )
    atomic_json(run_dir / "BASELINE_RECOMPUTE.json", results)
    lines = [
        "# Independent Baseline Recompute",
        "",
        "The authoritative corrected evaluator was executed from candidate geometry and GT rectangles. Saved candidate_success fields were not used as evaluator input.",
        "",
        "| Split | Backend | N | Candidate rows (source / Top-5) | Non-empty | J@1 | J@5 | Recoverable | No correct Top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for backend in ("G1", "C1"):
            item = results[f"{backend}_{split}"]
            lines.append(
                f"| {split} | {backend} | {item['sample_count']} | {item['source_candidate_rows']} / {item['frozen_top5_candidate_rows']} | "
                f"{item['non_empty_count']} | {item['j_at_1']:.6f} | {item['j_at_5']:.6f} | "
                f"{item['recoverable_error_count']} | {item['top5_no_correct_count']} |"
            )
    lines.extend([
        "",
        "Evaluator: rasterized rectangle IoU > 0.25 and parallel-jaw 180° periodic angle error ≤ 30° must be satisfied by the same GT grasp.",
        "",
        "These are offline 2D grasp-rectangle consistency results, not physical grasp success rates.",
    ])
    (run_dir / "BASELINE_RECOMPUTE.md").write_text("\n".join(lines) + "\n")
    return results
