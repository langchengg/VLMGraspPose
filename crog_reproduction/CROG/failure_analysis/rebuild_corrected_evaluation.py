#!/usr/bin/env python3
"""Rebuild CROG test metrics from the immutable full-test candidate cache.

This command performs no model inference and never writes to the frozen input
directory.  It uses the versioned canonical evaluator for every grasp label and
the original-resolution instance masks for segmentation metrics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import torch
from scipy.stats import binomtest

from failure_analysis.failure_utils import rle_to_mask
from failure_analysis.reranking.rankers import q_only_matches_q_rank, rank_candidates
from utils.grasp_metrics import (
    CORRECTED_EVALUATOR_VERSION,
    LEGACY_EVALUATOR_VERSION,
    binary_mask_iou,
    evaluate_candidate,
    validate_binary_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FROZEN_ROOT = (
    REPO_ROOT / "failure_analysis" / "reranking_outputs" / "full_test_17749_v1"
)
DEFAULT_EXPRESSIONS = REPO_ROOT.parent / "OCID-VLG" / "refer" / "multiple" / "test_expressions.json"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "exp"
    / "OCID-VLG_multiple_mac"
    / "CROG_mac_mps_official_params_50epoch_bs8"
    / "best_jindex_model.pth"
)
EXPECTED_SAMPLE_COUNT = 17_749
EXPECTED_CANDIDATES_PER_SAMPLE = 5
MASK_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
XY_ONLY_EVALUATOR_VERSION = "xy_only_geometric_sensitivity"
GRASP_VERSIONS = (
    LEGACY_EVALUATOR_VERSION,
    XY_ONLY_EVALUATOR_VERSION,
    CORRECTED_EVALUATOR_VERSION,
)
RANKERS = (
    "legacy",
    "q_only",
    "q_mask",
    "rule_2d_equal",
    "q_mask_width_angle",
    "q_mask_width_depth",
    "rule_fixed_v1",
)
HOLM_FAMILY = (
    "q_mask",
    "rule_2d_equal",
    "q_mask_width_angle",
    "q_mask_width_depth",
    "rule_fixed_v1",
)
CASE_ANCHORS = {293, 7011, 12328, 11246, 15383, 17490}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--expressions", type=Path, default=DEFAULT_EXPRESSIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args(argv)


class RunLogger:
    def __init__(self, path):
        self.path = Path(path)

    def log(self, message):
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_output(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout


def _refer_type(expression):
    filename = expression["template_filename"]
    if filename == "name.json":
        return "name"
    if filename == "location.json":
        return "location"
    if filename == "attribute.json":
        return "attribute"
    if filename != "relation.json":
        raise ValueError(f"unknown template filename: {filename}")
    pure_types = {"scene", "ground", "filter_category", "unique", "relate", "return"}
    program_types = {item["type"] for item in expression["program"]}
    return "pure_relation" if program_types.issubset(pure_types) else "mixed_relation"


def load_expressions(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = {}
    for row in payload["data"]:
        sample_id = int(row["question_index"])
        if sample_id in records:
            raise ValueError(f"duplicate expression question_index: {sample_id}")
        records[sample_id] = {
            "refer_type": _refer_type(row),
            "image_filename": row["image_filename"],
            "question": row["question"],
            "target": row["target"],
            "template_filename": row["template_filename"],
            "question_family_index": int(row["question_family_index"]),
            "program_signature": ">".join(item["type"] for item in row["program"]),
        }
    counts = Counter(row["refer_type"] for row in records.values())
    expected = {
        "name": 5809,
        "location": 2672,
        "attribute": 781,
        "pure_relation": 5769,
        "mixed_relation": 2718,
    }
    if len(records) != EXPECTED_SAMPLE_COUNT or dict(counts) != expected:
        raise AssertionError(f"expression partition mismatch: n={len(records)}, counts={dict(counts)}")
    return records, dict(counts)


def _candidate_identity(candidate):
    fields = (
        "candidate_id",
        "candidate_checksum",
        "legacy_rank",
        "q_rank",
        "row",
        "col",
        "cx",
        "cy",
        "angle_rad",
        "angle_deg",
        "width_px",
        "height_px",
        "polygon",
        "q_raw",
        "legacy_grasp",
    )
    return {field: candidate[field] for field in fields}


def _validate_candidate_join(sample_id, feature_candidates, prediction_candidates, pool_digest):
    if len(feature_candidates) != EXPECTED_CANDIDATES_PER_SAMPLE:
        raise AssertionError(f"sample {sample_id}: expected five feature candidates")
    if len(prediction_candidates) != EXPECTED_CANDIDATES_PER_SAMPLE:
        raise AssertionError(f"sample {sample_id}: expected five prediction candidates")
    if not q_only_matches_q_rank(feature_candidates):
        raise AssertionError(f"sample {sample_id}: q-only order differs from q_rank")
    ids = []
    for feature, prediction in zip(feature_candidates, prediction_candidates):
        left = _candidate_identity(feature)
        right = _candidate_identity(prediction)
        if left != right:
            raise AssertionError(f"sample {sample_id}: frozen candidate geometry mismatch")
        required = np.asarray(
            [feature["cx"], feature["cy"], feature["angle_rad"], feature["angle_deg"],
             feature["width_px"], feature["height_px"], feature["q_raw"]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(required)):
            raise AssertionError(f"sample {sample_id}: candidate contains NaN/Inf")
        if int(feature["row"]) != int(round(float(feature["cy"]))) or int(feature["col"]) != int(
            round(float(feature["cx"]))
        ):
            raise AssertionError(f"sample {sample_id}: row/col and x/y mismatch")
        ids.append(str(feature["candidate_id"]))
        pool_digest.update(canonical_json({"sample_id": sample_id, **left}).encode("utf-8"))
        pool_digest.update(b"\n")
    if len(set(ids)) != EXPECTED_CANDIDATES_PER_SAMPLE:
        raise AssertionError(f"sample {sample_id}: duplicate candidate_id within sample")


def _raw_target_mask(mask_path, object_id, mask_file_cache):
    key = str(mask_path)
    if key not in mask_file_cache:
        mask = cv2.imread(key, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"unable to load raw GT mask: {key}")
        if mask.ndim == 3:
            if not np.array_equal(mask[..., 0], mask[..., 1]) or not np.array_equal(
                mask[..., 0], mask[..., 2]
            ):
                raise ValueError(f"non-identical channels in instance mask: {key}")
            mask = mask[..., 0]
        if mask.shape != (480, 640):
            raise ValueError(f"unexpected raw GT mask shape {mask.shape}: {key}")
        mask_file_cache[key] = mask
    target = mask_file_cache[key] == int(object_id)
    if not target.any():
        raise ValueError(f"object_id {object_id} absent from {key}")
    return validate_binary_mask(target)


def _transition(before, after):
    if before and after:
        return "success_to_success"
    if not before and after:
        return "failure_to_success"
    if before and not after:
        return "success_to_failure"
    return "failure_to_failure"


def _first_valid(validities):
    return next((index + 1 for index, value in enumerate(validities) if value), None)


def _position_bin(bbox):
    center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    if center_x < 640.0 / 3.0:
        return "left"
    if center_x < 2.0 * 640.0 / 3.0:
        return "center"
    return "right"


def _sequence_key(frame_key):
    return str(frame_key).split(",", 1)[0]


def _failure_primary(top1_success, oracle_success, mask_iou, failure_mode):
    if top1_success:
        return "success"
    if oracle_success:
        return "ranking_failure"
    if mask_iou <= 0.50:
        return "segmentation_grounding_failure"
    return {
        "geometry_iou_failure": "candidate_pool_geometry_iou_failure",
        "angle_failure": "candidate_pool_angle_failure",
        "joint_mismatch": "candidate_pool_joint_mismatch",
        "both_failure": "candidate_pool_both_failure",
        "no_gt": "technical_no_gt",
        None: "technical_unknown",
    }[failure_mode]


def _update_breakdown(storage, dimension, value, mask_iou, j1, jany):
    bucket = storage[(dimension, str(value))]
    bucket["n"] += 1
    bucket["mask_iou_sum"] += float(mask_iou)
    bucket["j1"] += int(j1)
    bucket["jany"] += int(jany)
    for threshold in MASK_THRESHOLDS:
        bucket[f"pr{int(threshold * 100)}"] += int(mask_iou > threshold)


def _bootstrap_delta(method, baseline, clusters, *, seed, iterations):
    cluster_to_values = defaultdict(list)
    for method_value, base_value, cluster in zip(method, baseline, clusters):
        cluster_to_values[str(cluster)].append(int(method_value) - int(base_value))
    keys = sorted(cluster_to_values)
    sums = np.asarray([sum(cluster_to_values[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(cluster_to_values[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sampled = rng.integers(0, len(keys), size=len(keys))
        deltas[index] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "low_pp": float(np.percentile(deltas, 2.5) * 100.0),
        "high_pp": float(np.percentile(deltas, 97.5) * 100.0),
        "cluster_count": len(keys),
    }


def _holm_adjust(raw_pvalues):
    ordered = sorted(raw_pvalues, key=lambda name: (raw_pvalues[name], name))
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (m - index) * float(raw_pvalues[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(evaluator, family, metric, numerator, denominator, value):
    return {
        "evaluator": evaluator,
        "metric_family": family,
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "percentage": float(value) * 100.0,
    }


def _plot_outputs(
    output,
    aggregate_rows,
    failure_counts,
    reranking_rows,
    right_rows,
    legacy_masks,
    corrected_masks,
):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    grasp = [row for row in aggregate_rows if row["metric_family"] == "grasp" and row["metric"] in {"J@1", "J@Any"}]
    labels = []
    j1 = []
    jany = []
    for evaluator in (LEGACY_EVALUATOR_VERSION, XY_ONLY_EVALUATOR_VERSION, CORRECTED_EVALUATOR_VERSION):
        labels.append(evaluator.replace("_geometric", "\ngeometric"))
        j1.append(next(row["percentage"] for row in grasp if row["evaluator"] == evaluator and row["metric"] == "J@1"))
        jany.append(next(row["percentage"] for row in grasp if row["evaluator"] == evaluator and row["metric"] == "J@Any"))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, j1, 0.36, label="J@1")
    ax.bar(x + 0.18, jany, 0.36, label="J@Any / Oracle@5")
    ax.set_ylabel("Success (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(75, 100)
    ax.legend()
    ax.set_title("Frozen CROG candidates under versioned evaluators")
    fig.tight_layout()
    fig.savefig(figures / "evaluator_j_metrics.png", dpi=180)
    plt.close(fig)

    corrected_j = [
        next(row["percentage"] for row in aggregate_rows if row["evaluator"] == CORRECTED_EVALUATOR_VERSION and row["metric"] == f"J@{k}")
        for k in range(1, 6)
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, 6), corrected_j, marker="o")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Top K")
    ax.set_ylabel("Cumulative success (%)")
    ax.set_title("Corrected cumulative J@K")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "corrected_j_at_k.png", dpi=180)
    plt.close(fig)

    thresholds = [50, 60, 70, 80, 90]
    legacy_pr = [
        next(row["percentage"] for row in aggregate_rows if row["evaluator"] == LEGACY_EVALUATOR_VERSION and row["metric"] == f"Pr@{threshold}")
        for threshold in thresholds
    ]
    corrected_pr = [
        next(row["percentage"] for row in aggregate_rows if row["evaluator"] == CORRECTED_EVALUATOR_VERSION and row["metric"] == f"Pr@{threshold}")
        for threshold in thresholds
    ]
    x = np.arange(len(thresholds))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - 0.18, legacy_pr, 0.36, label="legacy transformed GT")
    ax.bar(x + 0.18, corrected_pr, 0.36, label="corrected raw binary GT")
    ax.set_xticks(x, [f"Pr@{value}" for value in thresholds])
    ax.set_ylabel("Expressions passing threshold (%)")
    ax.set_title("Segmentation metrics before and after raw-GT correction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "segmentation_thresholds.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(legacy_masks, bins=bins, alpha=0.55, label="legacy transformed GT")
    ax.hist(corrected_masks, bins=bins, alpha=0.55, label="corrected raw binary GT")
    ax.set_xlabel("Mask IoU")
    ax.set_ylabel("Samples")
    ax.set_title("Per-sample mask IoU distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "segmentation_iou_distribution.png", dpi=180)
    plt.close(fig)

    failure_labels = [key for key in sorted(failure_counts) if key != "success"]
    fig, ax = plt.subplots(figsize=(10, 5))
    values = [failure_counts[key] for key in failure_labels]
    ax.barh(failure_labels, values)
    ax.invert_yaxis()
    ax.set_xlabel("Samples")
    ax.set_title("Mutually exclusive corrected Top-1 failure taxonomy")
    fig.tight_layout()
    fig.savefig(figures / "failure_taxonomy.png", dpi=180)
    plt.close(fig)

    methods = [row["method"] for row in reranking_rows if row["status"] == "direct_corrected_reevaluation"]
    deltas = [row["delta_pp"] for row in reranking_rows if row["status"] == "direct_corrected_reevaluation"]
    lows = [row["frame_bootstrap_low_pp"] for row in reranking_rows if row["status"] == "direct_corrected_reevaluation"]
    highs = [row["frame_bootstrap_high_pp"] for row in reranking_rows if row["status"] == "direct_corrected_reevaluation"]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(deltas, y, xerr=[np.asarray(deltas) - np.asarray(lows), np.asarray(highs) - np.asarray(deltas)], fmt="o")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, methods)
    ax.set_xlabel("Corrected J@1 delta vs q-only (percentage points)")
    ax.set_title("Paired frame-cluster bootstrap 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(figures / "reranking_delta_ci.png", dpi=180)
    plt.close(fig)

    flip_rows = [
        row for row in reranking_rows
        if row["status"] == "direct_corrected_reevaluation"
        and row["method"] not in {"legacy", "q_only"}
    ]
    x = np.arange(len(flip_rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, [row["recovered"] for row in flip_rows], 0.36, label="Recovered")
    ax.bar(x + 0.18, [row["harmful"] for row in flip_rows], 0.36, label="Harmful")
    ax.set_xticks(x, [row["method"] for row in flip_rows], rotation=20, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Corrected re-ranking flips")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "reranking_recovered_harmful.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([row["group_value"] for row in right_rows], [row["j1_percentage"] for row in right_rows])
    ax.set_ylabel("Corrected J@1 (%)")
    ax.set_title("Corrected J@1 by horizontal target position")
    fig.tight_layout()
    fig.savefig(figures / "x_position_j1.png", dpi=180)
    plt.close(fig)


def _draw_rectangles(image, grasps, color, thickness=2):
    output = image.copy()
    for grasp in grasps:
        if len(grasp) < 5:
            continue
        x, y, width, height, angle = [float(value) for value in grasp[:5]]
        box = cv2.boxPoints(((x, y), (width, height), -angle)).astype(np.intp)
        cv2.polylines(output, [box], True, color, thickness)
    return output


def _make_case_outputs(output, case_records, selection_manifest):
    root = output / "failure_cases"
    success_dir = root / "success"
    failure_dir = root / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    rendered = {"success": [], "failure": []}
    for sample_id in sorted(case_records):
        row = case_records[sample_id]
        image_bgr = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(row["image_path"])
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        overlay = _draw_rectangles(image, row["gt_grasps"], (40, 100, 255), 2)
        overlay = _draw_rectangles(overlay, [row["candidate_grasp"]], (255, 50, 50), 3)
        kind = "success" if row["j1_success"] else "failure"
        directory = success_dir if kind == "success" else failure_dir
        path = directory / f"sample_{sample_id}_{row['primary_category']}.png"
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        axes[0].imshow(overlay)
        axes[0].set_title("GT grasps (blue), predicted Top-1 (red)")
        axes[1].imshow(row["raw_gt_mask"], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Raw binary GT mask")
        axes[2].imshow(row["predicted_mask"], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"Predicted mask; IoU={row['mask_iou']:.3f}")
        for axis in axes:
            axis.axis("off")
        best = row["best_gt"]
        diagnostics = (
            f"sample={sample_id} | {row['primary_category']} | "
            f"rectangle IoU={best['rectangle_iou']:.3f}, "
            f"d180={best['angle_difference_deg']:.1f}°, "
            f"center={best['center_distance_px']:.1f}px, width Δ={best['width_difference_px']:.1f}px"
        )
        fig.suptitle(row["language"] + "\n" + diagnostics, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        rendered[kind].append(path)
        selection_manifest.append({
            "sample_id": sample_id,
            "case_type": kind,
            "primary_category": row["primary_category"],
            "selection_rule": row["selection_rule"],
            "language": row["language"],
            "figure": str(path.relative_to(output)),
        })

    for kind, paths in rendered.items():
        if not paths:
            continue
        thumbs = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            width = 480
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            thumbs.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
        columns = 2
        rows = math.ceil(len(thumbs) / columns)
        thumb_height = max(image.shape[0] for image in thumbs)
        sheet = np.full((rows * thumb_height, columns * 480, 3), 255, dtype=np.uint8)
        for index, image in enumerate(thumbs):
            y = (index // columns) * thumb_height
            x = (index % columns) * 480
            sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
        cv2.imwrite(str(root / f"{kind}_contact_sheet.jpg"), sheet)


def _metric_inventory_text(output):
    return f"""# CROG corrected-evaluation metric inventory

This inventory maps legacy outputs to the regenerated artifacts in `{output}`.

| Legacy artifact / producer | Affected content | Root cause sensitivity | Corrected output | Status |
|---|---|---|---|---|
| `test_mac.log`, `console.log`, old audit/report Markdown | mean mask IoU, Pr@50–90, J@1, J@5/J@Any | transformed GT mask; x/y raster; legacy angle | `aggregate_metrics.csv`, `report.md`, `report.html` | regenerated for frozen full test |
| `failure_analysis/results/per_sample_diagnostics.csv` | per-sample mask/grasp labels and errors | all evaluator defects; nearest-GT explanation | `per_sample_metrics.csv.gz`, `evaluation_diff.csv` | regenerated |
| `failure_summary.md/.csv`, old taxonomy docs | failure counts/categories | legacy labels and center-nearest matching | `failure_summary.json`, `failures.csv` | regenerated |
| old failure/success/ranking PNGs and contact sheets | titles, captions, examples | legacy labels | `failure_cases/`, `figures/` | deterministically regenerated; old files preserved |
| `full_test_17749_v1/labels.jsonl` | candidate validity | legacy evaluator | `corrected_labels.jsonl.gz`, `candidate_gt_pairs.csv.gz` | regenerated; legacy file read only for hashing, never as labels |
| `eval_*/summary.json`, `per_sample.jsonl`, `case_index.json` | J@1, flips, McNemar, bootstrap | legacy candidate labels | `reranking_summary.csv`, `reranking_per_sample.csv.gz` | regenerated for fixed prediction-only rankers |
| `docs/CROG_RERANKING_IMPLEMENTATION.md` hand-maintained tables | all full-test reranking claims | legacy labels | `report.md`, `report.html` | replaced by generated evidence |
| `refer_types.json` | refer-type group memberships | five missing first records | `breakdown_metrics.csv` | rebuilt from raw expressions |
| `utils/dataset.py` polygon calls for training targets | training grasp target rasterization | not the evaluator path | none | inspected and excluded; frozen candidates unchanged |
| historical validation/Mac porting tables | validation/debug metrics | legacy evaluator but no matching frozen validation cache | none | preserved, explicitly out of full-test scope |
| manually curated `curated_current` images | orphaned/manual figures | no reproducible generator | none | preserved, not reused |

New coverage not present in the legacy reports: cumulative J@1–5, rank-independent
success, first-valid distribution, full migration tables, candidate × GT pairwise
records, raw-mask flips, object/refer/program/x-position/frame/sequence breakdowns,
Holm-adjusted McNemar tests, paired frame bootstrap, sequence-scene sensitivity,
and a self-contained HTML report.
"""


def _build_artifact(output, generated_at, aggregate_rows, reranking_rows, failure_counts):
    corrected = {
        row["metric"]: row for row in aggregate_rows
        if row["evaluator"] == CORRECTED_EVALUATOR_VERSION
    }
    # Execute the exact report-shaping queries in SQLite so every structured
    # report block can expose real, reproducible query provenance.
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE aggregate_metrics "
        "(evaluator TEXT, metric TEXT, numerator REAL, denominator REAL, value REAL, percentage REAL)"
    )
    connection.executemany(
        "INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?)",
        [
            (row["evaluator"], row["metric"], row["numerator"], row["denominator"], row["value"], row["percentage"])
            for row in aggregate_rows
        ],
    )
    connection.execute(
        "CREATE TABLE reranking_summary "
        "(method TEXT, status TEXT, j1_percentage REAL, delta_pp REAL, recovered REAL, harmful REAL, holm_p REAL)"
    )
    connection.executemany(
        "INSERT INTO reranking_summary VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (row["method"], row["status"], row.get("j1_percentage"), row.get("delta_pp"), row.get("recovered"), row.get("harmful"), row.get("holm_adjusted_pvalue"))
            for row in reranking_rows
        ],
    )
    connection.execute("CREATE TABLE failure_summary (category TEXT, count INTEGER)")
    connection.executemany(
        "INSERT INTO failure_summary VALUES (?, ?)",
        [(category, count) for category, count in sorted(failure_counts.items()) if category != "success"],
    )
    headline_sql = (
        "SELECT MAX(CASE WHEN metric='J@1' THEN value END) AS j1_rate, "
        "MAX(CASE WHEN metric='J@Any' THEN value END) AS oracle_rate, "
        "MAX(CASE WHEN metric='mean_mask_IoU' THEN value END) AS mask_iou, "
        "MAX(CASE WHEN metric='Pr@50' THEN numerator END) AS pr50_count, "
        "MAX(CASE WHEN metric='Pr@50' THEN value END) AS pr50_rate, "
        "(SELECT SUM(count) FROM failure_summary) AS top1_failure_count, "
        "(SELECT count FROM failure_summary WHERE category='ranking_failure') AS ranking_failure_count, "
        "(SELECT SUM(count) FROM failure_summary WHERE category!='ranking_failure') AS candidate_pool_failure_count, "
        "(SELECT delta_pp FROM reranking_summary WHERE method='q_mask') AS q_mask_delta_pp "
        "FROM aggregate_metrics WHERE evaluator='corrected_geometric_v2'"
    )
    evaluator_sql = (
        "SELECT evaluator, CASE evaluator "
        "WHEN 'legacy_official_impl_v1' THEN 'legacy' "
        "WHEN 'xy_only_geometric_sensitivity' THEN 'x/y-only' "
        "ELSE 'corrected' END AS evaluator_label, "
        "MAX(CASE WHEN metric='J@1' THEN percentage END) AS j1, "
        "MAX(CASE WHEN metric='J@Any' THEN percentage END) AS jany "
        "FROM aggregate_metrics WHERE evaluator IN "
        "('legacy_official_impl_v1','xy_only_geometric_sensitivity','corrected_geometric_v2') "
        "GROUP BY evaluator ORDER BY evaluator"
    )
    reranking_sql = (
        "SELECT method, status, j1_percentage AS j1, delta_pp, recovered, harmful, holm_p "
        "FROM reranking_summary ORDER BY delta_pp DESC"
    )
    failure_sql = "SELECT category, count FROM failure_summary ORDER BY count DESC, category"
    headline_dataset = [dict(row) for row in connection.execute(headline_sql)]
    evaluator_rows = [dict(row) for row in connection.execute(evaluator_sql)]
    rerank_dataset = [dict(row) for row in connection.execute(reranking_sql)]
    failure_dataset = [dict(row) for row in connection.execute(failure_sql)]
    connection.close()
    manifest_sources = [
        {"id": "headline_source", "label": "Corrected full-test summary", "path": "run_manifest.json"},
        {"id": "evaluator_source", "label": "Versioned evaluator comparison", "path": "aggregate_metrics.csv"},
        {"id": "reranking_source", "label": "Corrected re-ranking summary", "path": "reranking_summary.csv"},
        {"id": "failure_source", "label": "Corrected failure summary", "path": "failure_summary.json"},
    ]
    canonical_sources = [
        {"id": "headline_source", "query": {"engine": "sqlite", "sql": headline_sql, "description": "Joins corrected aggregate, failure, and reranking summaries for the report headline."}},
        {"id": "evaluator_source", "query": {"engine": "sqlite", "sql": evaluator_sql, "description": "Pivots J@1 and J@Any for the three versioned grasp evaluators."}},
        {"id": "reranking_source", "query": {"engine": "sqlite", "sql": reranking_sql, "description": "Loads all directly re-evaluated and excluded reranking rows."}},
        {"id": "failure_source", "query": {"engine": "sqlite", "sql": failure_sql, "description": "Loads mutually exclusive corrected failure categories."}},
    ]
    cards = [
        {
            "id": "j1_card",
            "description": "Top-ranked grasp has one same-GT rectangle-and-angle match.",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [{"label": "Corrected J@1", "field": "j1_rate", "format": "percent"}],
        },
        {
            "id": "oracle_card",
            "description": "At least one frozen Top-5 candidate is valid.",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [{"label": "Oracle@5", "field": "oracle_rate", "format": "percent"}],
        },
        {
            "id": "mask_card",
            "description": "Mean IoU against original-resolution binary target masks.",
            "dataset": "headline",
            "sourceId": "headline_source",
            "metrics": [{"label": "Mean mask IoU", "field": "mask_iou", "format": "percent"}],
        },
    ]
    charts = [
        {
            "id": "evaluator_chart",
            "title": "J@1 by evaluator implementation",
            "subtitle": "The frozen checkpoint and candidate pool are unchanged.",
            "type": "bar",
            "dataset": "evaluators",
            "sourceId": "evaluator_source",
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "evaluator_label", "type": "nominal", "label": "Evaluator"},
                "y": {"field": "j1", "type": "quantitative", "label": "J@1 (%)"},
                "tooltip": [
                    {"field": "evaluator", "type": "nominal", "label": "Version"},
                    {"field": "jany", "type": "quantitative", "label": "J@Any (%)"},
                ],
            },
        },
        {
            "id": "failure_chart",
            "title": "Corrected Top-1 failures by primary category",
            "subtitle": "Categories are mutually exclusive; center and width remain auxiliary diagnostics.",
            "type": "bar",
            "dataset": "failures",
            "sourceId": "failure_source",
            "encodings": {
                "x": {"field": "category", "type": "nominal", "label": "Primary category"},
                "y": {"field": "count", "type": "quantitative", "label": "Samples"},
            },
        },
    ]
    tables = [
        {
            "id": "reranking_table",
            "title": "Corrected re-ranking comparison",
            "subtitle": "Fixed prediction-only rankers are re-evaluated; legacy-label-trained or missing artifacts are excluded.",
            "dataset": "reranking",
            "sourceId": "reranking_source",
            "defaultSort": {"field": "delta_pp", "direction": "desc"},
            "columns": [
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "j1", "label": "J@1 (%)", "format": "number"},
                {"field": "delta_pp", "label": "Delta (pp)", "format": "number"},
                {"field": "recovered", "label": "Recovered", "format": "number"},
                {"field": "harmful", "label": "Harmful", "format": "number"},
                {"field": "holm_p", "label": "Holm p", "format": "number"},
            ],
        }
    ]
    j1_count = int(corrected["J@1"]["numerator"])
    oracle_count = int(corrected["J@Any"]["numerator"])
    pr50_count = int(corrected["Pr@50"]["numerator"])
    ranking_failure_count = int(failure_counts.get("ranking_failure", 0))
    candidate_pool_failure_count = EXPECTED_SAMPLE_COUNT - oracle_count
    q_mask_delta = next(row["delta_pp"] for row in reranking_rows if row["method"] == "q_mask")
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "CROG corrected geometric evaluation",
            "description": "Full 17,749-sample offline evaluation of a frozen Mac/MPS reproduction checkpoint.",
            "generatedAt": generated_at,
            "cards": [],
            "charts": [charts[0]],
            "tables": [],
            "sources": manifest_sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# CROG corrected geometric evaluation\n\nFrozen Mac/MPS reproduction checkpoint; offline geometric metrics, not robot execution success."},
                {"id": "answer", "type": "markdown", "sourceId": "headline_source", "body": f"Corrected **J@1 {j1_count:,}**, **Oracle@5 {oracle_count:,}**, **Pr@50 {pr50_count:,}** / {EXPECTED_SAMPLE_COUNT:,}. Failures: {ranking_failure_count:,} ranking + {candidate_pool_failure_count:,} pool. Best fixed reranker: Q+Mask {q_mask_delta:+.2f} pp. Full evidence: `report.md`."},
                {"id": "evaluator_chart_block", "type": "chart", "chartId": "evaluator_chart"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_dataset,
                "evaluators": evaluator_rows,
                "failures": failure_dataset,
                "reranking": rerank_dataset,
            },
        },
        "sources": canonical_sources,
    }
    write_json(output / "artifact.json", artifact)


def rebuild(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = RunLogger(output / "run.log")
    start = datetime.now(timezone.utc)
    log.log("Starting corrected full-test rebuild; no model inference will be run")
    if not (output / "RUNNING").exists():
        raise FileNotFoundError(f"RUNNING marker is required: {output / 'RUNNING'}")

    frozen_root = args.frozen_root.resolve()
    features_path = frozen_root / "features.jsonl"
    predictions_path = frozen_root / "predictions.jsonl"
    legacy_labels_path = frozen_root / "labels.jsonl"
    for path in (features_path, predictions_path, legacy_labels_path, args.expressions, args.checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    commit = git_output("rev-parse", "HEAD").strip()
    status = git_output("status", "--short")
    diff = git_output("diff", "--binary")
    (output / "git_status.txt").write_text(status, encoding="utf-8")
    (output / "git_diff.patch").write_text(diff, encoding="utf-8")
    input_paths = [features_path, predictions_path, legacy_labels_path, args.expressions.resolve(), args.checkpoint.resolve()]
    input_hashes = {}
    for path in input_paths:
        log.log(f"Hashing input {path}")
        input_hashes[str(path)] = sha256_file(path)
    with (output / "input_manifest.sha256").open("w", encoding="utf-8") as handle:
        for path in input_paths:
            handle.write(f"{input_hashes[str(path)]}  {path}\n")

    run_config = {
        "evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "legacy_evaluator_version": LEGACY_EVALUATOR_VERSION,
        "xy_only_evaluator_version": XY_ONLY_EVALUATOR_VERSION,
        "image_shape": [480, 640],
        "rectangle_iou_threshold": 0.25,
        "angle_threshold_deg": 30.0,
        "predicted_mask_threshold": 0.35,
        "segmentation_thresholds": list(MASK_THRESHOLDS),
        "bootstrap": {
            "seed": args.seed,
            "iterations": args.bootstrap_iterations,
            "primary_cluster_key": "scene_id (RGB-D frame)",
            "sensitivity_cluster_key": "scene_id prefix before comma (capture sequence)",
        },
        "candidate_pool": str(features_path),
        "candidate_pool_policy": "Jul-16 full_test_17749_v1 is canonical; no regeneration/filtering",
        "recomputed_rankers": list(RANKERS),
        "excluded_rankers": {
            "mlp_smoke_train200": "trained on legacy labels using only 200 train expressions / 3 frames",
            "rule_val_tuned": "no validation-tuned weight artifact exists",
        },
        "command": " ".join(sys.argv),
    }
    write_json(output / "run_config.json", run_config)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "scikit_image": skimage.__version__,
        "scipy": scipy.__version__,
    }
    (output / "environment.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    write_json(output / "metric_definitions.json", {
        "evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "rectangle_iou": "Raster IoU on the 480x640 canvas using cv2 [x,y] vertices and skimage polygon(row=y,col=x).",
        "angle_difference": "abs(((prediction-ground_truth+90) mod 180)-90) degrees.",
        "candidate_success": "Exists one GT with rectangle IoU >0.25 and periodic angle difference <=30 degrees.",
        "J@K": "Fraction of samples with at least one successful candidate among ranks 1..K.",
        "Oracle@5": "J@5 on the frozen five-candidate set; invariant to reordering.",
        "mask_IoU": "Binary IoU of predicted mask (frozen sigmoid/inverse/threshold>0.35 result) and raw instance_mask==obj_id.",
        "Pr@T": "Fraction of samples with mask IoU strictly greater than T.",
        "recovered": "q-only failure and reranker success on corrected labels.",
        "harmful": "q-only success and reranker failure on corrected labels.",
        "mcnemar": "Exact two-sided binomial test on recovered versus harmful discordant pairs.",
        "bootstrap": "Paired cluster bootstrap of method_success - q_only_success.",
    })
    (output / "metric_inventory.md").write_text(_metric_inventory_text(output), encoding="utf-8")

    expressions, refer_counts = load_expressions(args.expressions)
    log.log(f"Loaded complete raw expression partition: {refer_counts}")

    per_sample_fields = [
        "sample_id", "frame_key", "sequence_key", "refer_type", "target_name", "x_position",
        "language_instruction", "legacy_mask_iou", "corrected_mask_iou", "legacy_pr50", "corrected_pr50",
        "legacy_j1", "xy_only_j1", "corrected_j1", "legacy_jany", "xy_only_jany", "corrected_jany",
        "corrected_j2", "corrected_j3", "corrected_j4", "corrected_j5", "first_valid_rank",
        "rank_1_valid", "rank_2_valid", "rank_3_valid", "rank_4_valid", "rank_5_valid",
        "ranking_recoverable", "candidate_pool_failure", "primary_failure", "top1_failure_mode",
        "best_gt_id", "best_gt_index", "best_rectangle_iou", "best_periodic_angle_deg",
        "best_center_distance_px", "best_width_difference_px", "predicted_mask_area", "raw_gt_mask_area",
    ]
    pair_fields = [
        "sample_id", "candidate_id", "candidate_checksum", "candidate_rank", "gt_index", "gt_id",
        "rectangle_iou", "periodic_angle_difference_deg", "iou_ok", "angle_ok", "joint_success",
        "center_distance_px", "width_difference_px", "evaluator_version",
    ]
    diff_fields = [
        "sample_id", "legacy_j1", "corrected_j1", "legacy_jany", "corrected_jany",
        "legacy_best_gt_id", "corrected_best_gt_id", "legacy_rectangle_iou", "corrected_rectangle_iou",
        "periodic_angle_difference_deg", "legacy_mask_iou", "corrected_mask_iou", "j1_flip", "jany_flip",
        "mask_pr50_flip", "flip_reason",
    ]
    rerank_sample_fields = [
        "sample_id", "method", "baseline_top1_candidate_id", "method_top1_candidate_id",
        "baseline_success", "method_success", "oracle_success", "flip_type", "candidate_set_mismatch",
        "frame_key", "sequence_key", "ordered_candidate_ids", "ordered_scores",
    ]
    failure_fields = [
        "sample_id", "primary_failure", "ranking_failure", "candidate_pool_failure",
        "segmentation_grounding_failure", "geometry_iou_failure", "angle_failure", "joint_mismatch",
        "both_failure", "mask_iou", "best_rectangle_iou", "best_angle_difference_deg",
        "center_distance_px", "width_difference_px", "refer_type", "target_name", "frame_key", "language_instruction",
    ]

    version_validities = {version: [] for version in GRASP_VERSIONS}
    legacy_mask_ious = []
    corrected_mask_ious = []
    frame_keys = []
    sequence_keys = []
    rerank_success = {name: [] for name in RANKERS}
    corrected_rank_success = np.zeros(5, dtype=np.int64)
    breakdowns = defaultdict(lambda: defaultdict(float))
    migrations = {metric: Counter() for metric in ("J@1", "J@Any", "Pr@50")}
    failure_counts = Counter()
    auxiliary_failure_counts = Counter()
    first_valid_counts = Counter()
    case_records = {}
    category_case_counts = Counter()
    corrected_labels_path = output / "corrected_labels.jsonl.gz"
    pool_digest = hashlib.sha256()
    mask_file_cache = {}
    seen_ids = set()
    technical_failures = []
    sample_count = 0
    candidate_count = 0

    log.log("Streaming 17,749 samples and recomputing all candidate × GT labels")
    with (
        features_path.open(encoding="utf-8") as feature_handle,
        predictions_path.open(encoding="utf-8") as prediction_handle,
        gzip.open(output / "per_sample_metrics.csv.gz", "wt", encoding="utf-8", newline="") as sample_handle,
        gzip.open(output / "candidate_gt_pairs.csv.gz", "wt", encoding="utf-8", newline="") as pair_handle,
        gzip.open(output / "reranking_per_sample.csv.gz", "wt", encoding="utf-8", newline="") as rerank_handle,
        gzip.open(corrected_labels_path, "wt", encoding="utf-8") as label_handle,
        (output / "evaluation_diff.csv").open("w", encoding="utf-8", newline="") as diff_handle,
        (output / "label_flips.csv").open("w", encoding="utf-8", newline="") as flip_handle,
        (output / "failures.csv").open("w", encoding="utf-8", newline="") as failure_handle,
    ):
        sample_writer = csv.DictWriter(sample_handle, fieldnames=per_sample_fields)
        pair_writer = csv.DictWriter(pair_handle, fieldnames=pair_fields)
        rerank_writer = csv.DictWriter(rerank_handle, fieldnames=rerank_sample_fields)
        diff_writer = csv.DictWriter(diff_handle, fieldnames=diff_fields)
        flip_writer = csv.DictWriter(flip_handle, fieldnames=diff_fields)
        failure_writer = csv.DictWriter(failure_handle, fieldnames=failure_fields)
        for writer in (sample_writer, pair_writer, rerank_writer, diff_writer, flip_writer, failure_writer):
            writer.writeheader()

        while True:
            feature_line = feature_handle.readline()
            prediction_line = prediction_handle.readline()
            if not feature_line and not prediction_line:
                break
            if bool(feature_line) != bool(prediction_line):
                raise AssertionError("features/predictions line counts differ")
            feature = json.loads(feature_line)
            prediction = json.loads(prediction_line)
            sample_id = int(feature["sample_id"])
            if sample_id != int(prediction["sample_id"]):
                raise AssertionError(f"line join mismatch: {sample_id} vs {prediction['sample_id']}")
            if sample_id in seen_ids:
                raise AssertionError(f"duplicate sample_id: {sample_id}")
            seen_ids.add(sample_id)
            if sample_id not in expressions:
                raise AssertionError(f"missing raw expression for {sample_id}")
            expression = expressions[sample_id]
            if str(prediction["scene_id"]) != str(expression["image_filename"]):
                raise AssertionError(f"sample {sample_id}: scene mismatch with raw expression")
            if str(prediction["language_instruction"]) != str(expression["question"]):
                raise AssertionError(f"sample {sample_id}: language mismatch with raw expression")
            candidates = feature["candidates"]
            _validate_candidate_join(sample_id, candidates, prediction["candidates"], pool_digest)
            candidate_count += len(candidates)
            gt_grasps = prediction["gt_grasps"]
            gt_array = np.asarray(gt_grasps, dtype=np.float64)
            if gt_array.ndim != 2 or gt_array.shape[1] < 5 or not np.all(np.isfinite(gt_array)):
                raise AssertionError(f"sample {sample_id}: invalid GT grasps")

            predicted_mask = rle_to_mask(feature["predicted_mask_rle"])
            predicted_mask = validate_binary_mask(predicted_mask)
            if predicted_mask.shape != (480, 640):
                raise AssertionError(f"sample {sample_id}: predicted mask shape mismatch")
            if int(predicted_mask.sum()) != int(feature["predicted_mask_area"]):
                raise AssertionError(f"sample {sample_id}: predicted RLE area mismatch")
            raw_gt = _raw_target_mask(prediction["mask_path"], prediction["obj_id"], mask_file_cache)
            corrected_mask_iou = binary_mask_iou(predicted_mask, raw_gt)
            legacy_mask_iou = float(prediction["mask_iou"])
            if not math.isfinite(legacy_mask_iou):
                raise AssertionError(f"sample {sample_id}: non-finite legacy mask IoU")
            legacy_mask_ious.append(legacy_mask_iou)
            corrected_mask_ious.append(corrected_mask_iou)

            evaluations = {
                version: [
                    evaluate_candidate(candidate, gt_grasps, evaluator_version=version)
                    for candidate in candidates
                ]
                for version in GRASP_VERSIONS
            }
            validity = {
                version: [item["candidate_success"] for item in items]
                for version, items in evaluations.items()
            }
            for version in GRASP_VERSIONS:
                version_validities[version].append(validity[version])
            corrected = evaluations[CORRECTED_EVALUATOR_VERSION]
            corrected_valid = validity[CORRECTED_EVALUATOR_VERSION]
            for rank, value in enumerate(corrected_valid):
                corrected_rank_success[rank] += int(value)
            first_valid = _first_valid(corrected_valid)
            first_valid_counts[str(first_valid) if first_valid is not None else "none"] += 1
            j_at_k = [any(corrected_valid[:k]) for k in range(1, 6)]
            legacy_j1 = validity[LEGACY_EVALUATOR_VERSION][0]
            legacy_jany = any(validity[LEGACY_EVALUATOR_VERSION])
            xy_j1 = validity[XY_ONLY_EVALUATOR_VERSION][0]
            xy_jany = any(validity[XY_ONLY_EVALUATOR_VERSION])
            corrected_j1 = corrected_valid[0]
            corrected_jany = any(corrected_valid)
            legacy_pr50 = legacy_mask_iou > 0.50
            corrected_pr50 = corrected_mask_iou > 0.50
            migrations["J@1"][_transition(legacy_j1, corrected_j1)] += 1
            migrations["J@Any"][_transition(legacy_jany, corrected_jany)] += 1
            migrations["Pr@50"][_transition(legacy_pr50, corrected_pr50)] += 1

            frame_key = str(feature["scene_id"])
            sequence_key = _sequence_key(frame_key)
            frame_keys.append(frame_key)
            sequence_keys.append(sequence_key)
            x_position = _position_bin(prediction["bbox_xyxy"])
            refer_type = expression["refer_type"]
            primary = _failure_primary(
                corrected_j1,
                corrected_jany,
                corrected_mask_iou,
                corrected[0]["failure_mode"],
            )
            failure_counts[primary] += 1
            auxiliary = {
                "ranking_failure": (not corrected_j1 and corrected_jany),
                "candidate_pool_failure": not corrected_jany,
                "segmentation_grounding_failure": corrected_mask_iou <= 0.50,
                "geometry_iou_failure": corrected[0]["failure_mode"] == "geometry_iou_failure",
                "angle_failure": corrected[0]["failure_mode"] == "angle_failure",
                "joint_mismatch": corrected[0]["failure_mode"] == "joint_mismatch",
                "both_failure": corrected[0]["failure_mode"] == "both_failure",
            }
            for name, value in auxiliary.items():
                auxiliary_failure_counts[name] += int(value)

            for dimension, value in (
                ("refer_type", refer_type),
                ("target_name", prediction["target_name"]),
                ("x_position", x_position),
                ("frame", frame_key),
                ("sequence_scene", sequence_key),
                ("program_signature", expression["program_signature"]),
                ("question_family", expression["question_family_index"]),
            ):
                _update_breakdown(breakdowns, dimension, value, corrected_mask_iou, corrected_j1, corrected_jany)

            corrected_label = {
                "sample_id": sample_id,
                "evaluator_version": CORRECTED_EVALUATOR_VERSION,
                "candidate_labels": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_checksum": candidate["candidate_checksum"],
                        "candidate_valid": evaluation["candidate_success"],
                        "failure_mode": evaluation["failure_mode"],
                        "best_gt": evaluation["best_gt"],
                        "pairwise": evaluation["pairwise"],
                    }
                    for candidate, evaluation in zip(candidates, corrected)
                ],
            }
            label_handle.write(canonical_json(corrected_label) + "\n")
            for rank, (candidate, evaluation) in enumerate(zip(candidates, corrected), start=1):
                for pair in evaluation["pairwise"]:
                    pair_writer.writerow({
                        "sample_id": sample_id,
                        "candidate_id": candidate["candidate_id"],
                        "candidate_checksum": candidate["candidate_checksum"],
                        "candidate_rank": rank,
                        "gt_index": pair["gt_index"],
                        "gt_id": pair["gt_id"],
                        "rectangle_iou": pair["rectangle_iou"],
                        "periodic_angle_difference_deg": pair["angle_difference_deg"],
                        "iou_ok": pair["iou_ok"],
                        "angle_ok": pair["angle_ok"],
                        "joint_success": pair["joint_success"],
                        "center_distance_px": pair["center_distance_px"],
                        "width_difference_px": pair["width_difference_px"],
                        "evaluator_version": CORRECTED_EVALUATOR_VERSION,
                    })

            corrected_best = corrected[0]["best_gt"]
            legacy_best = evaluations[LEGACY_EVALUATOR_VERSION][0]["best_gt"]
            sample_writer.writerow({
                "sample_id": sample_id,
                "frame_key": frame_key,
                "sequence_key": sequence_key,
                "refer_type": refer_type,
                "target_name": prediction["target_name"],
                "x_position": x_position,
                "language_instruction": prediction["language_instruction"],
                "legacy_mask_iou": legacy_mask_iou,
                "corrected_mask_iou": corrected_mask_iou,
                "legacy_pr50": legacy_pr50,
                "corrected_pr50": corrected_pr50,
                "legacy_j1": legacy_j1,
                "xy_only_j1": xy_j1,
                "corrected_j1": corrected_j1,
                "legacy_jany": legacy_jany,
                "xy_only_jany": xy_jany,
                "corrected_jany": corrected_jany,
                "corrected_j2": j_at_k[1],
                "corrected_j3": j_at_k[2],
                "corrected_j4": j_at_k[3],
                "corrected_j5": j_at_k[4],
                "first_valid_rank": first_valid,
                **{f"rank_{index + 1}_valid": value for index, value in enumerate(corrected_valid)},
                "ranking_recoverable": auxiliary["ranking_failure"],
                "candidate_pool_failure": auxiliary["candidate_pool_failure"],
                "primary_failure": primary,
                "top1_failure_mode": corrected[0]["failure_mode"],
                "best_gt_id": corrected_best["gt_id"],
                "best_gt_index": corrected_best["gt_index"],
                "best_rectangle_iou": corrected_best["rectangle_iou"],
                "best_periodic_angle_deg": corrected_best["angle_difference_deg"],
                "best_center_distance_px": corrected_best["center_distance_px"],
                "best_width_difference_px": corrected_best["width_difference_px"],
                "predicted_mask_area": int(predicted_mask.sum()),
                "raw_gt_mask_area": int(raw_gt.sum()),
            })
            flip_reasons = []
            if legacy_j1 != corrected_j1:
                flip_reasons.append("J@1_geometry_or_angle")
            if legacy_jany != corrected_jany:
                flip_reasons.append("J@Any_geometry_or_angle")
            if legacy_pr50 != corrected_pr50:
                flip_reasons.append("Pr@50_raw_GT_mask")
            diff_row = {
                "sample_id": sample_id,
                "legacy_j1": legacy_j1,
                "corrected_j1": corrected_j1,
                "legacy_jany": legacy_jany,
                "corrected_jany": corrected_jany,
                "legacy_best_gt_id": legacy_best["gt_id"],
                "corrected_best_gt_id": corrected_best["gt_id"],
                "legacy_rectangle_iou": legacy_best["rectangle_iou"],
                "corrected_rectangle_iou": corrected_best["rectangle_iou"],
                "periodic_angle_difference_deg": corrected_best["angle_difference_deg"],
                "legacy_mask_iou": legacy_mask_iou,
                "corrected_mask_iou": corrected_mask_iou,
                "j1_flip": _transition(legacy_j1, corrected_j1),
                "jany_flip": _transition(legacy_jany, corrected_jany),
                "mask_pr50_flip": _transition(legacy_pr50, corrected_pr50),
                "flip_reason": ";".join(flip_reasons) or "unchanged",
            }
            diff_writer.writerow(diff_row)
            if flip_reasons:
                flip_writer.writerow(diff_row)

            baseline_ranked = rank_candidates(candidates, "q_only")
            baseline_top = baseline_ranked[0]["candidate_id"]
            valid_by_id = {
                candidate["candidate_id"]: value for candidate, value in zip(candidates, corrected_valid)
            }
            baseline_success = bool(valid_by_id[baseline_top])
            if baseline_success != corrected_j1:
                raise AssertionError(f"sample {sample_id}: corrected baseline does not equal q-only Top-1")
            original_ids = {candidate["candidate_id"] for candidate in candidates}
            for method in RANKERS:
                ranked = rank_candidates(candidates, method)
                ranked_ids = [candidate["candidate_id"] for candidate in ranked]
                candidate_set_mismatch = set(ranked_ids) != original_ids
                if candidate_set_mismatch:
                    raise AssertionError(f"sample {sample_id}: {method} changed candidate set")
                method_success = bool(valid_by_id[ranked_ids[0]])
                rerank_success[method].append(method_success)
                if baseline_success and method_success:
                    flip_type = "both_success"
                elif not baseline_success and method_success:
                    flip_type = "recovered"
                elif baseline_success and not method_success:
                    flip_type = "harmful"
                else:
                    flip_type = "both_fail"
                rerank_writer.writerow({
                    "sample_id": sample_id,
                    "method": method,
                    "baseline_top1_candidate_id": baseline_top,
                    "method_top1_candidate_id": ranked_ids[0],
                    "baseline_success": baseline_success,
                    "method_success": method_success,
                    "oracle_success": corrected_jany,
                    "flip_type": flip_type,
                    "candidate_set_mismatch": False,
                    "frame_key": frame_key,
                    "sequence_key": sequence_key,
                    "ordered_candidate_ids": canonical_json(ranked_ids),
                    "ordered_scores": canonical_json([candidate["rerank_score"] for candidate in ranked]),
                })

            if not corrected_j1:
                failure_writer.writerow({
                    "sample_id": sample_id,
                    "primary_failure": primary,
                    **auxiliary,
                    "mask_iou": corrected_mask_iou,
                    "best_rectangle_iou": corrected_best["rectangle_iou"],
                    "best_angle_difference_deg": corrected_best["angle_difference_deg"],
                    "center_distance_px": corrected_best["center_distance_px"],
                    "width_difference_px": corrected_best["width_difference_px"],
                    "refer_type": refer_type,
                    "target_name": prediction["target_name"],
                    "frame_key": frame_key,
                    "language_instruction": prediction["language_instruction"],
                })

            select = False
            selection_rule = None
            if sample_id in CASE_ANCHORS:
                select = True
                selection_rule = "required_golden_or_boundary_anchor"
            elif corrected_j1 and category_case_counts["success"] < 10:
                select = True
                selection_rule = "lowest_sample_id_among_corrected_successes"
                category_case_counts["success"] += 1
            elif not corrected_j1 and category_case_counts[primary] < 4:
                select = True
                selection_rule = f"lowest_sample_id_in_{primary}"
                category_case_counts[primary] += 1
            if select:
                case_records[sample_id] = {
                    "image_path": prediction["image_path"],
                    "language": prediction["language_instruction"],
                    "gt_grasps": gt_grasps,
                    "candidate_grasp": candidates[0]["legacy_grasp"],
                    "raw_gt_mask": raw_gt.copy(),
                    "predicted_mask": predicted_mask.copy(),
                    "mask_iou": corrected_mask_iou,
                    "best_gt": corrected_best,
                    "j1_success": corrected_j1,
                    "primary_category": primary,
                    "selection_rule": selection_rule,
                }

            sample_count += 1
            if sample_count % 1000 == 0:
                log.log(f"Recomputed {sample_count}/{EXPECTED_SAMPLE_COUNT} samples")

    expected_ids = set(range(EXPECTED_SAMPLE_COUNT))
    missing_ids = sorted(expected_ids - seen_ids)
    extra_ids = sorted(seen_ids - expected_ids)
    if sample_count != EXPECTED_SAMPLE_COUNT or missing_ids or extra_ids:
        raise AssertionError(
            f"incomplete population: n={sample_count}, missing={missing_ids[:20]}, extra={extra_ids[:20]}"
        )
    if candidate_count != EXPECTED_SAMPLE_COUNT * EXPECTED_CANDIDATES_PER_SAMPLE:
        raise AssertionError(f"candidate count mismatch: {candidate_count}")
    log.log("Full population evaluated with zero missing, duplicate, non-finite, or technical failures")

    validity_arrays = {version: np.asarray(values, dtype=bool) for version, values in version_validities.items()}
    legacy_masks = np.asarray(legacy_mask_ious, dtype=np.float64)
    corrected_masks = np.asarray(corrected_mask_ious, dtype=np.float64)
    aggregate_rows = []
    for version, mask_source in (
        (LEGACY_EVALUATOR_VERSION, legacy_masks),
        (XY_ONLY_EVALUATOR_VERSION, legacy_masks),
        (CORRECTED_EVALUATOR_VERSION, corrected_masks),
    ):
        values = validity_arrays[version]
        for k in range(1, 6):
            successes = np.any(values[:, :k], axis=1)
            aggregate_rows.append(_metric_row(version, "grasp", f"J@{k}", int(successes.sum()), sample_count, float(successes.mean())))
        oracle = np.any(values, axis=1)
        aggregate_rows.append(_metric_row(version, "grasp", "J@Any", int(oracle.sum()), sample_count, float(oracle.mean())))
        aggregate_rows.append(_metric_row(version, "grasp", "Oracle@5", int(oracle.sum()), sample_count, float(oracle.mean())))
        aggregate_rows.append(_metric_row(version, "segmentation", "mean_mask_IoU", float(mask_source.sum()), sample_count, float(mask_source.mean())))
        for threshold in MASK_THRESHOLDS:
            success = mask_source > threshold
            aggregate_rows.append(_metric_row(version, "segmentation", f"Pr@{int(threshold * 100)}", int(success.sum()), sample_count, float(success.mean())))
    _write_csv(
        output / "aggregate_metrics.csv",
        aggregate_rows,
        ["evaluator", "metric_family", "metric", "numerator", "denominator", "value", "percentage"],
    )

    corrected_values = validity_arrays[CORRECTED_EVALUATOR_VERSION]
    cumulative = [int(np.any(corrected_values[:, :k], axis=1).sum()) for k in range(1, 6)]
    if cumulative != sorted(cumulative):
        raise AssertionError(f"J@K is not monotonic: {cumulative}")
    if cumulative[-1] != int(np.any(corrected_values, axis=1).sum()):
        raise AssertionError("J@5 != J@Any")
    ranking_recoverable_count = cumulative[-1] - cumulative[0]
    if ranking_recoverable_count != auxiliary_failure_counts["ranking_failure"]:
        raise AssertionError("Oracle gap does not equal ranking-recoverable count")
    pr_counts = [int((corrected_masks > threshold).sum()) for threshold in MASK_THRESHOLDS]
    if pr_counts != sorted(pr_counts, reverse=True):
        raise AssertionError(f"Pr thresholds not monotonic: {pr_counts}")

    # Cross-check immutable frozen population against the read-only diagnostic anchors.
    anchors = {
        LEGACY_EVALUATOR_VERSION: (83.2047, 90.8727),
        XY_ONLY_EVALUATOR_VERSION: (89.97, 94.51),
        CORRECTED_EVALUATOR_VERSION: (89.24, 94.35),
    }
    for version, (expected_j1, expected_jany) in anchors.items():
        actual_j1 = validity_arrays[version][:, 0].mean() * 100.0
        actual_jany = np.any(validity_arrays[version], axis=1).mean() * 100.0
        if abs(actual_j1 - expected_j1) > 0.02 or abs(actual_jany - expected_jany) > 0.02:
            raise AssertionError(
                f"sanity anchor mismatch for {version}: J1={actual_j1}, JAny={actual_jany}"
            )
    # The external diagnostic anchor is reported to ten decimal places, so its
    # comparison tolerance must be wider than the omitted trailing precision.
    if abs(corrected_masks.mean() - 0.8018632345) > 1e-8:
        raise AssertionError(f"raw mask mean IoU mismatch: {corrected_masks.mean()}")
    if int((corrected_masks > 0.5).sum()) != 16843:
        raise AssertionError("raw-mask Pr@50 numerator mismatch")

    migration_rows = []
    for metric, counts in migrations.items():
        if sum(counts.values()) != sample_count:
            raise AssertionError(f"migration table incomplete: {metric}")
        migration_rows.append({"metric": metric, **{name: counts[name] for name in (
            "success_to_success", "failure_to_success", "success_to_failure", "failure_to_failure"
        )}, "denominator": sample_count})
    _write_csv(
        output / "legacy_vs_corrected.csv",
        migration_rows,
        ["metric", "success_to_success", "failure_to_success", "success_to_failure", "failure_to_failure", "denominator"],
    )

    breakdown_rows = []
    for (dimension, value), bucket in sorted(breakdowns.items()):
        n = int(bucket["n"])
        row = {
            "dimension": dimension,
            "group_value": value,
            "n": n,
            "mean_mask_iou": bucket["mask_iou_sum"] / n,
            "j1_numerator": int(bucket["j1"]),
            "j1_percentage": 100.0 * bucket["j1"] / n,
            "jany_numerator": int(bucket["jany"]),
            "jany_percentage": 100.0 * bucket["jany"] / n,
        }
        for threshold in MASK_THRESHOLDS:
            key = f"pr{int(threshold * 100)}"
            row[f"{key}_numerator"] = int(bucket[key])
            row[f"{key}_percentage"] = 100.0 * bucket[key] / n
        breakdown_rows.append(row)
    breakdown_fields = list(breakdown_rows[0])
    _write_csv(output / "breakdown_metrics.csv", breakdown_rows, breakdown_fields)

    baseline = np.asarray(rerank_success["q_only"], dtype=bool)
    oracle = np.any(corrected_values, axis=1)
    baseline_count = int(baseline.sum())
    oracle_count = int(oracle.sum())
    raw_pvalues = {}
    direct_rows = []
    for method in RANKERS:
        values = np.asarray(rerank_success[method], dtype=bool)
        both_success = int(np.logical_and(baseline, values).sum())
        recovered = int(np.logical_and(~baseline, values).sum())
        harmful = int(np.logical_and(baseline, ~values).sum())
        both_fail = int(np.logical_and(~baseline, ~values).sum())
        if both_success + recovered + harmful + both_fail != sample_count:
            raise AssertionError(f"reranking four-cell table incomplete: {method}")
        method_count = int(values.sum())
        delta_pp = 100.0 * (recovered - harmful) / sample_count
        if not math.isclose(delta_pp, 100.0 * (method_count - baseline_count) / sample_count, abs_tol=1e-12):
            raise AssertionError(f"reranking delta identity failed: {method}")
        discordant = recovered + harmful
        pvalue = float(binomtest(recovered, discordant, p=0.5).pvalue) if discordant else 1.0
        frame_ci = _bootstrap_delta(values, baseline, frame_keys, seed=args.seed, iterations=args.bootstrap_iterations)
        scene_ci = _bootstrap_delta(values, baseline, sequence_keys, seed=args.seed, iterations=args.bootstrap_iterations)
        row = {
            "method": method,
            "status": "direct_corrected_reevaluation",
            "n": sample_count,
            "success_numerator": method_count,
            "j1_percentage": 100.0 * method_count / sample_count,
            "delta_pp": delta_pp,
            "both_success": both_success,
            "recovered": recovered,
            "harmful": harmful,
            "both_fail": both_fail,
            "net_recovered": recovered - harmful,
            "recoverable_baseline_failures": ranking_recoverable_count,
            "recovery_rate": recovered / ranking_recoverable_count if ranking_recoverable_count else None,
            "oracle_at_5_numerator": oracle_count,
            "oracle_at_5_percentage": 100.0 * oracle_count / sample_count,
            "mcnemar_exact_two_sided_pvalue": pvalue,
            "holm_adjusted_pvalue": None,
            "holm_family": ";".join(HOLM_FAMILY),
            "frame_bootstrap_low_pp": frame_ci["low_pp"],
            "frame_bootstrap_high_pp": frame_ci["high_pp"],
            "frame_cluster_key": "scene_id (RGB-D frame)",
            "frame_cluster_count": frame_ci["cluster_count"],
            "scene_bootstrap_low_pp": scene_ci["low_pp"],
            "scene_bootstrap_high_pp": scene_ci["high_pp"],
            "scene_cluster_key": "scene_id prefix before comma (capture sequence)",
            "scene_cluster_count": scene_ci["cluster_count"],
            "bootstrap_seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "candidate_set_mismatch_count": 0,
        }
        direct_rows.append(row)
        if method in HOLM_FAMILY:
            raw_pvalues[method] = pvalue
    adjusted = _holm_adjust(raw_pvalues)
    for row in direct_rows:
        if row["method"] in adjusted:
            row["holm_adjusted_pvalue"] = adjusted[row["method"]]
    excluded_rows = []
    for method, reason in (
        ("mlp_smoke_train200", "excluded: trained on legacy labels using only 200 train expressions / 3 frames"),
        ("rule_val_tuned", "excluded: no validation-tuned weight artifact exists"),
    ):
        excluded_rows.append({
            **{field: None for field in direct_rows[0]},
            "method": method,
            "status": reason,
            "n": sample_count,
            "holm_family": ";".join(HOLM_FAMILY),
            "bootstrap_seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "candidate_set_mismatch_count": 0,
        })
    reranking_rows = direct_rows + excluded_rows
    _write_csv(output / "reranking_summary.csv", reranking_rows, list(direct_rows[0]))

    failure_summary = {
        "evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "sample_count": sample_count,
        "top1_success_count": int(corrected_values[:, 0].sum()),
        "top1_failure_count": int((~corrected_values[:, 0]).sum()),
        "ranking_failure_count": ranking_recoverable_count,
        "candidate_pool_failure_count": int((~oracle).sum()),
        "technical_failure_count": len(technical_failures),
        "primary_taxonomy": dict(sorted(failure_counts.items())),
        "auxiliary_diagnostics": dict(sorted(auxiliary_failure_counts.items())),
        "first_valid_rank_distribution": dict(sorted(first_valid_counts.items())),
        "rank_independent_success": {
            str(index + 1): {
                "numerator": int(value),
                "denominator": sample_count,
                "percentage": float(value) * 100.0 / sample_count,
            }
            for index, value in enumerate(corrected_rank_success)
        },
    }
    write_json(output / "failure_summary.json", failure_summary)

    right_rows = [row for row in breakdown_rows if row["dimension"] == "x_position"]
    _plot_outputs(
        output,
        aggregate_rows,
        failure_counts,
        reranking_rows,
        right_rows,
        legacy_masks,
        corrected_masks,
    )
    selection_manifest = []
    _make_case_outputs(output, case_records, selection_manifest)
    _write_csv(
        output / "failure_cases" / "selection_manifest.csv",
        selection_manifest,
        ["sample_id", "case_type", "primary_category", "selection_rule", "language", "figure"],
    )

    corrected_j1_count = int(corrected_values[:, 0].sum())
    corrected_oracle_count = int(oracle.sum())
    legacy_j1_count = int(validity_arrays[LEGACY_EVALUATOR_VERSION][:, 0].sum())
    xy_j1_count = int(validity_arrays[XY_ONLY_EVALUATOR_VERSION][:, 0].sum())
    failure_only = {key: value for key, value in sorted(failure_counts.items()) if key != "success"}
    report_lines = [
        "# CROG corrected geometric evaluation",
        "",
        "## Summary",
        "",
        f"This run evaluates the **CROG Mac/MPS reproduction checkpoint under the corrected geometric evaluator** on all {sample_count:,} frozen test predictions. It performs no new inference and does not alter the checkpoint or candidate pool.",
        "",
        f"Corrected J@1 is **{corrected_j1_count:,}/{sample_count:,} ({100*corrected_j1_count/sample_count:.4f}%)**; Oracle@5/J@Any is **{corrected_oracle_count:,}/{sample_count:,} ({100*corrected_oracle_count/sample_count:.4f}%)**. Raw-GT mean mask IoU is **{corrected_masks.mean():.6f}**, and Pr@50 is **{int((corrected_masks>0.5).sum()):,}/{sample_count:,} ({100*(corrected_masks>0.5).mean():.4f}%)**.",
        "",
        "## Evaluator migration",
        "",
        "| Evaluation view | Grasp evaluator | Mask GT | J@1 | J@Any | mean mask IoU | Pr@50 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    evaluation_views = (
        ("legacy", LEGACY_EVALUATOR_VERSION, "transformed", legacy_masks),
        ("x/y only", XY_ONLY_EVALUATOR_VERSION, "transformed", legacy_masks),
        ("x/y + periodic angle", CORRECTED_EVALUATOR_VERSION, "transformed", legacy_masks),
        ("full corrected", CORRECTED_EVALUATOR_VERSION, "raw binary", corrected_masks),
    )
    for view, version, mask_source, mask_values in evaluation_views:
        values = validity_arrays[version]
        j1_n = int(values[:, 0].sum())
        any_n = int(np.any(values, axis=1).sum())
        pr50_n = int((mask_values > 0.50).sum())
        report_lines.append(
            f"| {view} | `{version}` | {mask_source} | "
            f"{j1_n:,}/{sample_count:,} ({100*j1_n/sample_count:.4f}%) | "
            f"{any_n:,}/{sample_count:,} ({100*any_n/sample_count:.4f}%) | "
            f"{mask_values.mean():.6f} | {pr50_n:,}/{sample_count:,} ({100*pr50_n/sample_count:.4f}%) |"
        )
    report_lines.extend([
        "",
        "The first row is a **legacy official-implementation-compatible metric**, not a claim that the implementation was geometrically correct. The corrected rows are reproduction results, not a paper-reported result.",
        "",
        "## Corrected cumulative grasp metrics",
        "",
        "| Metric | Numerator | Denominator | Percentage |",
        "|---|---:|---:|---:|",
    ])
    for k, count in enumerate(cumulative, start=1):
        report_lines.append(f"| J@{k} | {count:,} | {sample_count:,} | {100*count/sample_count:.4f}% |")
    report_lines.extend([
        f"| J@Any / Oracle@5 | {corrected_oracle_count:,} | {sample_count:,} | {100*corrected_oracle_count/sample_count:.4f}% |",
        "",
        f"Top-1 failures: **{sample_count-corrected_j1_count:,}**. Ranking-recoverable failures: **{ranking_recoverable_count:,}**. Candidate-pool failures: **{sample_count-corrected_oracle_count:,}**.",
        "",
        "## Corrected segmentation metrics",
        "",
        "| Metric | Numerator | Denominator | Percentage/value |",
        "|---|---:|---:|---:|",
        f"| mean mask IoU | {corrected_masks.sum():.6f} | {sample_count:,} | {corrected_masks.mean():.6f} |",
    ])
    for threshold in MASK_THRESHOLDS:
        count = int((corrected_masks > threshold).sum())
        report_lines.append(f"| Pr@{int(threshold*100)} | {count:,} | {sample_count:,} | {100*count/sample_count:.4f}% |")
    report_lines.extend([
        "",
        "## Re-ranking",
        "",
        "| Method | Status | Corrected J@1 | Δpp | Recovered | Harmful | McNemar p | Holm p | Frame bootstrap 95% CI (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in reranking_rows:
        if row["success_numerator"] is None:
            report_lines.append(f"| {row['method']} | {row['status']} | — | — | — | — | — | — | — |")
        else:
            holm = "—" if row["holm_adjusted_pvalue"] is None else f"{row['holm_adjusted_pvalue']:.4g}"
            report_lines.append(
                f"| {row['method']} | direct corrected re-evaluation | {row['j1_percentage']:.4f}% | {row['delta_pp']:+.4f} | {row['recovered']} | {row['harmful']} | {row['mcnemar_exact_two_sided_pvalue']:.4g} | {holm} | [{row['frame_bootstrap_low_pp']:+.3f}, {row['frame_bootstrap_high_pp']:+.3f}] |"
            )
    report_lines.extend([
        "",
        f"Bootstrap configuration: seed {args.seed}, {args.bootstrap_iterations:,} paired iterations, 344 RGB-D frame clusters; sequence-scene sensitivity uses 115 capture-sequence clusters. Holm correction covers {', '.join(HOLM_FAMILY)}.",
        "",
        "## Failure taxonomy",
        "",
        "| Primary category | Count | Share of all samples |",
        "|---|---:|---:|",
    ])
    for category, count in failure_only.items():
        report_lines.append(f"| {category} | {count:,} | {100*count/sample_count:.3f}% |")
    report_lines.extend([
        "",
        "Primary categories are mutually exclusive. Center distance and width difference are auxiliary diagnostics only; low rectangle IoU is not treated as proof of a width failure.",
        "",
        "## Reproducibility and limitations",
        "",
        f"Input integrity: {sample_count:,} unique IDs, zero missing/duplicate IDs, {candidate_count:,} candidates, exactly five per sample, zero candidate-set mismatches, zero non-finite required geometry values, and zero technical failures. Canonical candidate-pool SHA-256 is `{pool_digest.hexdigest()}`.",
        "",
        "The Jul-16 `full_test_17749_v1` cache is the sole frozen candidate source. The old MLP is excluded because it learned legacy labels on a 200-expression/3-frame smoke subset. `rule_val_tuned` is excluded because no validation weight artifact exists. Dense quality maps were not cached, so this run cannot and does not change peak generation.",
        "",
        "This is an offline geometric metric. It is not an official CROG score, a strict official training reproduction, or real-robot grasp success.",
    ])
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    generated_at = datetime.now(timezone.utc).isoformat()
    _build_artifact(output, generated_at, aggregate_rows, reranking_rows, failure_counts)
    manifest = {
        "status": "analysis_complete_html_and_final_QA_pending",
        "started_at": start.isoformat(),
        "analysis_completed_at": generated_at,
        "git_commit": commit,
        "git_status_at_start": status.splitlines(),
        "git_diff_path": str(output / "git_diff.patch"),
        "input_hashes": input_hashes,
        "checkpoint_sha256": input_hashes[str(args.checkpoint.resolve())],
        "prediction_sha256": input_hashes[str(predictions_path)],
        "feature_cache_sha256": input_hashes[str(features_path)],
        "candidate_pool_sha256": pool_digest.hexdigest(),
        "evaluator_version": CORRECTED_EVALUATOR_VERSION,
        "legacy_evaluator_version": LEGACY_EVALUATOR_VERSION,
        "sample_count": sample_count,
        "unique_sample_count": len(seen_ids),
        "missing_sample_count": len(missing_ids),
        "duplicate_sample_count": sample_count - len(seen_ids),
        "candidate_count": candidate_count,
        "candidates_per_sample": EXPECTED_CANDIDATES_PER_SAMPLE,
        "technical_failure_count": len(technical_failures),
        "frame_cluster_count": len(set(frame_keys)),
        "sequence_cluster_count": len(set(sequence_keys)),
        "mask_file_count": len(mask_file_cache),
        "candidate_set_mismatch_count": 0,
        "environment": environment,
        "run_config": run_config,
        "old_candidate_caveat": "Jul-16 cache differs from Jul-9 legacy predictions by tiny float geometry for samples 17744-17748; Jul-16 is canonical.",
    }
    write_json(output / "run_manifest.json", manifest)
    log.log(f"Analysis artifacts complete in {output}; report.html packaging and final QA remain")


def main(argv=None):
    args = parse_args(argv)
    try:
        rebuild(args)
    except Exception:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "INCOMPLETE").write_text(
            "Corrected evaluation rebuild failed. See run.log and traceback.log.\n",
            encoding="utf-8",
        )
        (output / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
