#!/usr/bin/env python3
"""Consolidate native predictions, evaluate, analyse, and render formal outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.fair_crog_hifics_g1_c1_no_rerank.geometry import (
    ANGLE_THRESHOLD_DEG,
    IOU_THRESHOLD,
    CanonicalGrasp,
    candidate_row_to_grasp,
    corners,
    evaluate_ranked,
    rectangle_mask,
)


METHODS = ("CROG", "G1", "C1")
ORACLES = ("G1-ORACLE", "C1-ORACLE")
DISPLAY = {"CROG": "CROG-native", "G1": "HiFi-CS→G1", "C1": "HiFi-CS→C1"}
COLORS = {"CROG": "#0072B2", "G1": "#D55E00", "C1": "#009E73"}
BOOTSTRAP_SEED = 20260806
BOOTSTRAP_REPLICATES = 10000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rle_decode(record: dict[str, Any]) -> np.ndarray:
    size = tuple(map(int, record["size"]))
    counts = list(map(int, record["counts"]))
    values = np.empty(sum(counts), dtype=np.uint8)
    cursor, value = 0, int(record.get("start_value", 0))
    for count in counts:
        values[cursor : cursor + count] = value
        cursor += count
        value = 1 - value
    if cursor != int(np.prod(size)):
        raise ValueError("invalid CROG RLE length")
    return values.reshape(size).astype(bool)


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    union = np.count_nonzero(first | second)
    return 1.0 if union == 0 else float(np.count_nonzero(first & second) / union)


def mask_from_instance(path: str, instance_id: int) -> np.ndarray:
    return np.asarray(Image.open(path)) == int(instance_id)


def _candidate_frame(run: Path, variant: str, method: str) -> pd.DataFrame:
    frame = pd.read_parquet(run / "02_predictions/native_work" / variant / "candidates.parquet")
    frame["method"] = method
    frame["variant"] = variant
    return frame


def extract_and_validate(run: Path, crog_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(run / "01_manifest/paired_manifest.parquet")
    expected = set(manifest["query_id"].astype(int))
    checkpoint_registry = pd.read_csv(run / "00_audit/checkpoint_registry.csv")
    crog_sha = str(checkpoint_registry.loc[checkpoint_registry.method == "CROG-native", "sha256"].iloc[0])
    crog_rows, mask_records = [], []
    with crog_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            query_id = int(record["sample_index"])
            if query_id not in expected:
                continue
            sid = str(manifest.loc[manifest.query_id == query_id, "sample_id"].iloc[0])
            candidates = sorted(record["candidates"], key=lambda item: int(item["q_rank"]))
            if len(candidates) != 5 or [item["q_rank"] for item in candidates] != list(range(5)):
                raise ValueError(f"CROG native K/rank mismatch: {sid}")
            scores = [float(item["q_raw"]) for item in candidates]
            if any(scores[index] < scores[index + 1] for index in range(4)):
                raise ValueError(f"CROG q-score order mismatch: {sid}")
            provenance = record.get("provenance", {})
            for rank, candidate in enumerate(candidates, 1):
                crog_rows.append(
                    {
                        "sample_id": sid,
                        "method": "CROG",
                        "variant": "crog_native",
                        "candidate_id": str(candidate["candidate_id"]),
                        "native_rank": rank,
                        "native_score": float(candidate["q_raw"]),
                        "cx_px": float(candidate["cx"]),
                        "cy_px": float(candidate["cy"]),
                        "theta_deg": float((float(candidate["angle_deg"]) + 90) % 180 - 90),
                        "jaw_width_px": float(candidate["width_px"]),
                        "rectangle_height_px": float(candidate["height_px"]),
                        "source_row": int(candidate["row"]),
                        "source_column": int(candidate["col"]),
                        "status": "ok",
                        "failure_reason": None,
                        "transform_json": json.dumps(provenance, sort_keys=True),
                        "checkpoint_sha256": crog_sha,
                        "selected_config_sha256": sha256_file(
                            ROOT / "crog_reproduction/CROG/config/OCID-VLG/crog_multiple_r50.yaml"
                        ),
                        "native_decoder_config_sha256": hashlib.sha256(
                            json.dumps(provenance, sort_keys=True).encode()
                        ).hexdigest(),
                    }
                )
            mask_records.append(
                {
                    "sample_id": sid,
                    "query_id": query_id,
                    "crog_mask_rle_json": json.dumps(record["predicted_mask_rle"], separators=(",", ":")),
                    "crog_mask_area": int(record["predicted_mask_area"]),
                    "scene_instance_count": int(record.get("scene_instance_count", 0)),
                }
            )
    crog = pd.DataFrame(crog_rows)
    masks = pd.DataFrame(mask_records)
    if crog.sample_id.nunique() != len(manifest) or len(crog) != 5 * len(manifest):
        raise RuntimeError("CROG paired extraction incomplete")
    outputs = {
        "g1": _candidate_frame(run, "g1", "G1"),
        "c1": _candidate_frame(run, "c1", "C1"),
        "g1_gtmask_oracle": _candidate_frame(run, "g1_gtmask_oracle", "G1-ORACLE"),
        "c1_gtmask_oracle": _candidate_frame(run, "c1_gtmask_oracle", "C1-ORACLE"),
    }
    for name, frame in outputs.items():
        sample = pd.read_parquet(run / "02_predictions/native_work" / name / "per_sample.parquet")
        if len(sample) != len(manifest) or sample.sample_id.nunique() != len(manifest):
            raise RuntimeError(f"{name} sample coverage mismatch")
        if int(sample.status.eq("technical_failure").sum()) != 0:
            raise RuntimeError(f"{name} contains technical failures")
        for _, group in frame.groupby("sample_id", sort=False):
            ordered = group.sort_values("native_rank")
            if ordered.native_rank.tolist() != list(range(1, len(ordered) + 1)):
                raise RuntimeError(f"{name} rank gaps")
            if np.any(np.diff(ordered.native_score.to_numpy(float)) > 1e-7):
                raise RuntimeError(f"{name} native-score order violation")
    # The required named files are immutable copies of native candidate rows.
    crog.to_parquet(run / "02_predictions/crog_native_predictions.parquet", index=False)
    outputs["g1"].to_parquet(run / "02_predictions/g1_native_predictions.parquet", index=False)
    outputs["c1"].to_parquet(run / "02_predictions/c1_native_predictions.parquet", index=False)
    outputs["g1_gtmask_oracle"].to_parquet(run / "02_predictions/g1_gtmask_oracle_predictions.parquet", index=False)
    outputs["c1_gtmask_oracle"].to_parquet(run / "02_predictions/c1_gtmask_oracle_predictions.parquet", index=False)
    all_candidates = pd.concat([crog, *outputs.values()], ignore_index=True)
    return all_candidates, masks


def compute_masks(run: Path, crog_masks: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_parquet(run / "01_manifest/paired_manifest.parquet")
    by_sid = crog_masks.set_index("sample_id")
    rows = []
    for index, row in enumerate(manifest.itertuples(index=False), 1):
        gt = mask_from_instance(row.gt_mask_path, row.target_instance_id)
        hifi = np.asarray(Image.open(row.predicted_hifics_mask_path)) > 0
        if hifi.shape != gt.shape:
            raise RuntimeError(f"HiFi/native GT mask shape mismatch: {row.sample_id}")
        crog_record = by_sid.loc[row.sample_id]
        crog = rle_decode(json.loads(crog_record.crog_mask_rle_json))
        if crog.shape != gt.shape:
            raise RuntimeError(f"CROG/native GT mask shape mismatch: {row.sample_id}")
        rows.append(
            {
                "sample_id": row.sample_id,
                "hifics_mask_path": row.predicted_hifics_mask_path,
                "hifics_mask_sha256": row.predicted_hifics_mask_sha256,
                "hifics_probability_path": row.predicted_hifics_probability_path,
                "hifics_probability_sha256": row.predicted_hifics_probability_sha256,
                "gt_mask_area_px": int(np.count_nonzero(gt)),
                "hifics_mask_area_px": int(np.count_nonzero(hifi)),
                "crog_mask_area_px": int(np.count_nonzero(crog)),
                "hifics_mask_iou": binary_iou(hifi, gt),
                "crog_mask_iou": binary_iou(crog, gt),
                "hifics_empty": not np.any(hifi),
                "crog_empty": not np.any(crog),
            }
        )
        if index % 500 == 0:
            print(f"mask metrics {index}/{len(manifest)}", flush=True)
    result = pd.DataFrame(rows)
    if result.hifics_mask_sha256.isna().any() or result.sample_id.nunique() != len(manifest):
        raise RuntimeError("HiFi shared-mask identity failure")
    result.to_parquet(run / "02_predictions/hifics_masks.parquet", index=False)
    result.to_csv(run / "04_metrics/per_sample_mask_metrics.csv", index=False)
    return result


def evaluate_all(run: Path, all_candidates: pd.DataFrame, masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(run / "01_manifest/paired_manifest.parquet").set_index("sample_id")
    mask_by_sid = masks.set_index("sample_id")
    grouped = {(method, sid): group.sort_values("native_rank") for (method, sid), group in all_candidates.groupby(["method", "sample_id"], sort=False)}
    candidate_updates, sample_rows = [], []
    methods = METHODS + ORACLES
    for sample_index, (sid, meta) in enumerate(manifest.iterrows(), 1):
        gt_corners = json.loads(meta.gt_grasp_list_json)
        gt_mask = mask_from_instance(meta.gt_mask_path, int(meta.target_instance_id))
        ys, xs = np.nonzero(gt_mask)
        bbox_diag = math.hypot(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)) if xs.size else float("nan")
        for method in methods:
            frame = grouped.get((method, sid))
            candidate_records = [] if frame is None else frame.to_dict("records")
            predictions = [candidate_row_to_grasp(item) for item in candidate_records]
            result = evaluate_ranked(predictions, gt_corners)
            for record, evaluated in zip(candidate_records, result["evaluated"], strict=True):
                diagnostic = evaluated["diagnostic_match"]
                candidate_updates.append(
                    {
                        **record,
                        "candidate_success": bool(evaluated["success"]),
                        "diagnostic_gt_index": None if diagnostic is None else int(diagnostic["gt_index"]),
                        "diagnostic_iou": None if diagnostic is None else float(diagnostic["iou"]),
                        "diagnostic_angle_error_deg": None if diagnostic is None else float(diagnostic["angle_error_deg"]),
                    }
                )
            row: dict[str, Any] = {
                "sample_id": sid,
                "method": method,
                "candidate_count": int(result["candidate_count"]),
                "no_output": int(result["candidate_count"]) == 0,
                "status": "no_output" if int(result["candidate_count"]) == 0 else "ok",
                "failure_reason": "no_candidate" if int(result["candidate_count"]) == 0 else None,
                **{f"j_at_{k}": bool(result[f"j_at_{k}"]) for k in range(1, 6)},
                "oracle_all": bool(result["oracle_all"]),
                "first_success_rank": result["first_success_rank"],
            }
            if predictions:
                top = predictions[0]
                diagnostic = result["top1"]["diagnostic_match"]
                matched_gt = result["ground_truth"][int(diagnostic["gt_index"])]
                pred_mask = rectangle_mask(top)
                row.update(
                    {
                        "top1_candidate_id": candidate_records[0]["candidate_id"],
                        "top1_native_score": top.native_score,
                        "top1_cx_px": top.cx_px,
                        "top1_cy_px": top.cy_px,
                        "top1_theta_deg": top.theta_deg,
                        "top1_jaw_width_px": top.jaw_width_px,
                        "top1_rectangle_height_px": top.rectangle_height_px,
                        "matched_gt_index": int(diagnostic["gt_index"]),
                        "rotated_iou": float(diagnostic["iou"]),
                        "angle_error_deg": float(diagnostic["angle_error_deg"]),
                        "normalized_center_error": math.hypot(top.cx_px - matched_gt.cx_px, top.cy_px - matched_gt.cy_px) / bbox_diag,
                        "relative_width_error": abs(top.jaw_width_px - matched_gt.jaw_width_px) / max(matched_gt.jaw_width_px, 1e-9),
                        "center_in_gt_mask": bool(
                            0 <= round(top.cy_px) < gt_mask.shape[0]
                            and 0 <= round(top.cx_px) < gt_mask.shape[1]
                            and gt_mask[int(round(top.cy_px)), int(round(top.cx_px))]
                        ),
                        "rectangle_coverage_by_gt_mask": float(np.count_nonzero(pred_mask & gt_mask) / max(np.count_nonzero(pred_mask), 1)),
                    }
                )
            else:
                for key in (
                    "top1_native_score", "top1_cx_px", "top1_cy_px", "top1_theta_deg",
                    "top1_jaw_width_px", "top1_rectangle_height_px", "matched_gt_index",
                    "rotated_iou", "angle_error_deg", "normalized_center_error",
                    "relative_width_error", "center_in_gt_mask", "rectangle_coverage_by_gt_mask",
                ):
                    row[key] = None
                row["top1_candidate_id"] = None
            mask = mask_by_sid.loc[sid]
            row["mask_iou"] = float(mask.crog_mask_iou if method == "CROG" else mask.hifics_mask_iou)
            row["predicted_mask_empty"] = bool(mask.crog_empty if method == "CROG" else mask.hifics_empty)
            sample_rows.append(row)
        if sample_index % 250 == 0:
            print(f"canonical evaluation {sample_index}/{len(manifest)}", flush=True)
    candidates = pd.DataFrame(candidate_updates)
    samples = pd.DataFrame(sample_rows)
    candidates.to_parquet(run / "03_canonical/canonical_candidates.parquet", index=False)
    top1_columns = [column for column in samples.columns if column.startswith("top1_")] + ["sample_id", "method", "status", "failure_reason"]
    samples[top1_columns].to_parquet(run / "03_canonical/canonical_top1.parquet", index=False)
    samples.to_parquet(run / "04_metrics/per_sample_metrics.parquet", index=False)
    samples.to_csv(run / "04_metrics/per_sample_metrics.csv", index=False)
    return candidates, samples


def bootstrap_binary(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values.astype(float), minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(float)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    batch = 500
    for start in range(0, BOOTSTRAP_REPLICATES, batch):
        stop = min(start + batch, BOOTSTRAP_REPLICATES)
        draws = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        estimates[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return estimates


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues), dtype=float)
    running = 0.0
    total = len(pvalues)
    for position, index in enumerate(order):
        running = max(running, (total - position) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def summarize_and_stats(run: Path, samples: pd.DataFrame, masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(run / "01_manifest/paired_manifest.parquet")
    main = samples[samples.method.isin(METHODS)].merge(
        manifest[["sample_id", "scene_id", "frame_id", "scene_family", "expression_type", "object_category", "gt_grasp_count", "scene_instance_count", "target_bbox_width", "target_bbox_height"]],
        on="sample_id", validate="many_to_one",
    )
    mask_summary = {}
    for method, column, empty in (("CROG", "crog_mask_iou", "crog_empty"), ("G1", "hifics_mask_iou", "hifics_empty"), ("C1", "hifics_mask_iou", "hifics_empty")):
        values = masks[column].to_numpy(float)
        mask_summary[method] = {
            "mask_miou": float(values.mean()),
            **{f"p_at_{threshold}": float(np.mean(values > threshold / 100)) for threshold in (50, 60, 70, 80, 90)},
            "empty_mask_rate": float(masks[empty].mean()),
        }
    summary_rows = []
    bootstrap: dict[str, Any] = {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "methods": {}, "pairs": {}}
    scene_rng = np.random.default_rng(BOOTSTRAP_SEED)
    frame_rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for method in METHODS:
        frame = main[main.method == method].sort_values("sample_id")
        values = frame.j_at_1.to_numpy(bool)
        scene_dist = bootstrap_binary(values, frame.scene_family.to_numpy(str), scene_rng)
        frame_dist = bootstrap_binary(values, frame.frame_id.to_numpy(str), frame_rng)
        bootstrap["methods"][method] = {"scene_ci": interval(scene_dist), "frame_ci": interval(frame_dist)}
        summary_rows.append(
            {
                "Method": DISPLAY[method],
                "N": len(frame),
                **mask_summary[method],
                **{f"J@{k}": float(frame[f"j_at_{k}"].mean()) for k in range(1, 6)},
                "Oracle@All (diagnostic)": float(frame.oracle_all.mean()),
                "No-output": float(frame.no_output.mean()),
                "J@1 numerator": int(frame.j_at_1.sum()),
                "J@1 scene CI low": interval(scene_dist)[0],
                "J@1 scene CI high": interval(scene_dist)[1],
                "Median latency": np.nan,
                "P95 latency": np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    pivot = main.pivot(index="sample_id", columns="method", values="j_at_1").loc[:, METHODS].astype(bool)
    cochran = cochrans_q(pivot.to_numpy(int))
    comparisons = [("CROG", "G1"), ("CROG", "C1"), ("G1", "C1")]
    stat_rows, raw_p = [], []
    manifest_indexed = manifest.set_index("sample_id").loc[pivot.index]
    for number, (a, b) in enumerate(comparisons):
        av, bv = pivot[a].to_numpy(bool), pivot[b].to_numpy(bool)
        a_wrong_b_correct = int(np.sum(~av & bv))
        a_correct_b_wrong = int(np.sum(av & ~bv))
        discordant = a_wrong_b_correct + a_correct_b_wrong
        p = 1.0 if discordant == 0 else float(binomtest(min(a_wrong_b_correct, a_correct_b_wrong), discordant, 0.5, alternative="two-sided").pvalue)
        raw_p.append(p)
        delta = bv.astype(float) - av.astype(float)
        scene_dist = bootstrap_binary(delta, manifest_indexed.scene_family.to_numpy(str), np.random.default_rng(BOOTSTRAP_SEED + 10 + number))
        frame_dist = bootstrap_binary(delta, manifest_indexed.frame_id.to_numpy(str), np.random.default_rng(BOOTSTRAP_SEED + 20 + number))
        bootstrap["pairs"][f"{b}-{a}"] = {"scene_ci": interval(scene_dist), "frame_ci": interval(frame_dist)}
        stat_rows.append(
            {
                "Comparison": f"{DISPLAY[a]} vs {DISPLAY[b]}",
                "Method A J@1": float(av.mean()),
                "Method B J@1": float(bv.mean()),
                "Delta B-A pp": float(100 * delta.mean()),
                "A wrong / B correct": a_wrong_b_correct,
                "A correct / B wrong": a_correct_b_wrong,
                "McNemar exact p": p,
                "Scene-bootstrap 95% CI pp": f"[{100*interval(scene_dist)[0]:.3f}, {100*interval(scene_dist)[1]:.3f}]",
                "Frame-bootstrap sensitivity CI pp": f"[{100*interval(frame_dist)[0]:.3f}, {100*interval(frame_dist)[1]:.3f}]",
            }
        )
    adjusted = holm(raw_p)
    for row, value in zip(stat_rows, adjusted, strict=True):
        row["Holm-adjusted p"] = value
    stats = pd.DataFrame(stat_rows)
    tests = {
        "cochran_q": {"statistic": float(cochran.statistic), "pvalue": float(cochran.pvalue), "N": len(pivot)},
        "pairwise": stat_rows,
        "cluster_definition": {
            "primary": "scene_family (OCID capture sequence; 111 clusters)",
            "sensitivity": "frame_id (unique RGB frame; 325 clusters)",
            "split_limitation": "Official train/validation/test reuse capture sequences but not RGB frames.",
        },
    }
    (run / "05_statistics/statistical_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    (run / "05_statistics/bootstrap_intervals.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    summary.to_csv(run / "04_metrics/main_results.csv", index=False)
    stats.to_csv(run / "05_statistics/paired_statistics.csv", index=False)
    return summary, stats, tests


def continuous_summary(run: Path, samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        frame = samples[samples.method == method]
        iou = frame.rotated_iou.dropna().to_numpy(float)
        angle = frame.angle_error_deg.dropna().to_numpy(float)
        centre = frame.normalized_center_error.dropna().to_numpy(float)
        width = frame.relative_width_error.dropna().to_numpy(float)
        rows.append(
            {
                "Method": DISPLAY[method],
                "Median rotated IoU": float(np.median(iou)),
                "IQR rotated IoU": float(np.quantile(iou, 0.75) - np.quantile(iou, 0.25)),
                "Median angle error": float(np.median(angle)),
                "IQR angle error": float(np.quantile(angle, 0.75) - np.quantile(angle, 0.25)),
                "Median normalized centre error": float(np.median(centre)),
                "Median relative width error": float(np.median(width)),
                "Centre-in-GT-mask rate": float(frame.center_in_gt_mask.fillna(False).mean()),
                "Rectangle target coverage": float(frame.rectangle_coverage_by_gt_mask.mean()),
                "High-IoU/angle-fail count": int(((frame.rotated_iou > IOU_THRESHOLD) & (frame.angle_error_deg > ANGLE_THRESHOLD_DEG)).sum()),
                "Angle-pass/IoU-fail count": int(((frame.angle_error_deg <= ANGLE_THRESHOLD_DEG) & (frame.rotated_iou <= IOU_THRESHOLD)).sum()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(run / "04_metrics/continuous_geometry_summary.csv", index=False)
    return result


def failure_analysis(run: Path, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    taxonomy_rows = []
    for method in ("G1", "C1"):
        predicted = samples[samples.method == method].set_index("sample_id")
        oracle = samples[samples.method == f"{method}-ORACLE"].set_index("sample_id")
        rows = []
        for sid, pred in predicted.iterrows():
            gt = oracle.loc[sid]
            if bool(pred.j_at_1):
                category = "top1_success"
            elif str(pred.status) == "technical_failure" or str(gt.status) == "technical_failure":
                category = "technical_or_data_failure"
            elif bool(pred.oracle_all) and bool(pred.j_at_5):
                category = "native_selection_within_top5"
            elif bool(pred.oracle_all) and not bool(pred.j_at_5):
                category = "correct_candidate_below_top5"
            elif not bool(pred.oracle_all) and bool(gt.j_at_1):
                category = "grounding_limited"
            elif not bool(pred.oracle_all) and not bool(gt.j_at_1) and bool(gt.oracle_all):
                category = "grounding_plus_selection"
            else:
                category = "grasper_or_candidate_generation_limited"
            rows.append(
                {
                    "sample_id": sid,
                    "method": method,
                    "category": category,
                    "predicted_mask_empty": bool(pred.predicted_mask_empty),
                    "no_candidate_predicted_mask": bool(pred.no_output),
                    "no_candidate_gt_mask": bool(gt.no_output),
                    "invalid_depth": "invalid_depth" in str(pred.failure_reason),
                    "invalid_geometry": "geometry" in str(pred.failure_reason),
                    "target_mask_iou_bin": pd.cut([pred.mask_iou], [-1, .25, .5, .7, .9, 1], labels=["[0,.25)", "[.25,.5)", "[.5,.7)", "[.7,.9)", "[.9,1]"])[0],
                }
            )
        detail = pd.DataFrame(rows)
        if len(detail) != len(predicted) or detail.category.isna().any():
            raise RuntimeError(f"{method} taxonomy conservation failure")
        detail.to_csv(run / f"06_failure_analysis/{method.lower()}_failure_taxonomy.csv", index=False)
        counts = detail.category.value_counts()
        taxonomy_rows.append({"Method": method, **{key: int(counts.get(key, 0)) for key in [
            "top1_success", "native_selection_within_top5", "correct_candidate_below_top5",
            "grounding_limited", "grounding_plus_selection", "grasper_or_candidate_generation_limited",
            "technical_or_data_failure",
        ]}})
    taxonomy = pd.DataFrame(taxonomy_rows)
    main = samples[samples.method.isin(METHODS)].pivot(index="sample_id", columns="method", values="j_at_1").astype(bool)
    labels = {
        (True, True, True): "all_correct",
        (True, False, False): "CROG_only",
        (False, True, False): "G1_only",
        (False, False, True): "C1_only",
        (True, True, False): "CROG_G1",
        (True, False, True): "CROG_C1",
        (False, True, True): "G1_C1",
        (False, False, False): "all_wrong",
    }
    outcomes = pd.DataFrame(
        [{"sample_id": sid, "outcome": labels[tuple(bool(row[item]) for item in METHODS)]} for sid, row in main.iterrows()]
    )
    table = outcomes.outcome.value_counts().reindex(labels.values(), fill_value=0).rename_axis("Outcome combination").reset_index(name="Count")
    table["Percentage"] = 100 * table.Count / len(outcomes)
    outcomes.to_csv(run / "06_failure_analysis/per_sample_outcome.csv", index=False)
    table.to_csv(run / "06_failure_analysis/outcome_combinations.csv", index=False)
    taxonomy.to_csv(run / "06_failure_analysis/modular_failure_taxonomy_summary.csv", index=False)
    return taxonomy, table


def stratified(run: Path, samples: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_parquet(run / "01_manifest/paired_manifest.parquet")
    base = manifest.merge(masks[["sample_id", "gt_mask_area_px", "hifics_mask_iou", "crog_mask_iou"]], on="sample_id")
    g1_work = pd.read_parquet(run / "02_predictions/native_work/g1/per_sample.parquet")[["sample_id", "valid_depth_fraction"]]
    base = base.merge(g1_work, on="sample_id")
    base["mask_area_quartile"] = pd.qcut(base.gt_mask_area_px, 4, labels=["Q1-small", "Q2", "Q3", "Q4-large"])
    base["depth_valid_quartile"] = pd.qcut(base.valid_depth_fraction.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    base["bbox_aspect_bin"] = pd.cut(base.target_bbox_width / base.target_bbox_height.clip(lower=1), [0, .67, 1.5, np.inf], labels=["tall", "balanced", "wide"])
    base["gt_grasp_count_bin"] = pd.cut(base.gt_grasp_count, [0, 2, 5, 10, np.inf], labels=["1-2", "3-5", "6-10", "11+"])
    base["clutter_bin"] = pd.cut(base.scene_instance_count, [0, 5, 10, np.inf], labels=["low", "medium", "high"])
    long = samples[samples.method.isin(METHODS)].merge(base, on="sample_id", validate="many_to_one")
    long["method_mask_iou"] = np.where(long.method == "CROG", long.crog_mask_iou, long.hifics_mask_iou)
    long["mask_iou_bin"] = pd.cut(long.method_mask_iou, [-1e-12, .25, .5, .7, .9, 1.0000001], right=False, labels=["[0,.25)", "[.25,.5)", "[.5,.7)", "[.7,.9)", "[.9,1]"])
    groupings = [
        "expression_type", "mask_area_quartile", "object_category", "mask_iou_bin",
        "gt_grasp_count_bin", "depth_valid_quartile", "clutter_bin", "scene_family", "bbox_aspect_bin",
    ]
    rows = []
    for grouping in groupings:
        for value, group in long.groupby(grouping, observed=True):
            record: dict[str, Any] = {"Stratum type": grouping, "Stratum": str(value), "N": int(group.sample_id.nunique())}
            rates = {}
            for method in METHODS:
                m = group[group.method == method]
                n, successes = len(m), int(m.j_at_1.sum())
                record[f"{method} N"] = n
                low, high = proportion_confint(successes, n, method="wilson") if n else (np.nan, np.nan)
                record[f"{method} J@1"] = successes / n if n else np.nan
                record[f"{method} Oracle@5"] = float(m.j_at_5.mean()) if n else np.nan
                record[f"{method} no-output"] = float(m.no_output.mean()) if n else np.nan
                record[f"{method} 95% CI"] = f"[{low:.4f}, {high:.4f}]" if n else "NA"
                rates[method] = successes / n if n else -1
            if grouping == "mask_iou_bin":
                record["Best method"] = "not paired: method-specific bin membership"
                record["Max-min delta pp"] = np.nan
            else:
                record["Best method"] = max(rates, key=rates.get)
                record["Max-min delta pp"] = 100 * (max(rates.values()) - min(rates.values()))
            rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(run / "05_statistics/stratified_results.csv", index=False)
    return result


def write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(markdown_table(frame) + "\n")
    path.with_suffix(".tex").write_text(frame.to_latex(index=False, float_format="%.6f"))


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, float):
            value = f"{value:.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns = [cell(item) for item in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(cell(item) for item in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def formal_tables(run: Path, summary: pd.DataFrame, continuous: pd.DataFrame, stats: pd.DataFrame, strata: pd.DataFrame, taxonomy: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    registry = pd.read_csv(run / "00_audit/checkpoint_registry.csv").set_index("method")
    table1 = pd.DataFrame([
        {"Method": "CROG-native", "Architecture type": "end-to-end", "Input modalities": "RGB+text", "Grounding model": "CROG", "Grasp model": "CROG", "Checkpoint SHA-256": registry.loc["CROG-native", "sha256"], "Training split": "OCID-VLG multiple train; validation selected", "Native Top-1 rule": "highest q local peak", "Candidate count": 5, "Device": "MPS"},
        {"Method": "HiFi-CS→G1", "Architecture type": "modular", "Input modalities": "RGB+text→mask; RGB+depth+mask crop→grasp", "Grounding model": "HiFi-CS repeated-FiLM", "Grasp model": "fine-tuned GR-ConvNet", "Checkpoint SHA-256": registry.loc["G1 GR-ConvNet", "sha256"], "Training split": "OCID-VLG unique train; validation selected", "Native Top-1 rule": "highest official q peak", "Candidate count": "≤100", "Device": "MPS"},
        {"Method": "HiFi-CS→C1", "Architecture type": "modular", "Input modalities": "RGB+text→mask; depth+mask crop→grasp", "Grounding model": "HiFi-CS repeated-FiLM", "Grasp model": "fine-tuned GG-CNN2", "Checkpoint SHA-256": registry.loc["C1 GG-CNN2", "sha256"], "Training split": "OCID-VLG unique train; validation selected", "Native Top-1 rule": "highest official q peak", "Candidate count": "≤100", "Device": "MPS"},
    ])
    expression = strata[strata["Stratum type"] == "expression_type"].copy()
    for number, table in enumerate((table1, summary, continuous, stats, expression, taxonomy, outcomes), 1):
        write_table(table, run / "09_reports" / f"Table_{number}")


def save_figure(fig: plt.Figure, run: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(run / "07_figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(run / "07_figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(run: Path, samples: pd.DataFrame, summary: pd.DataFrame, taxonomy: pd.DataFrame, outcomes: pd.DataFrame, strata: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 120})
    main = samples[samples.method.isin(METHODS)]
    # 1 forest
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for index, method in enumerate(METHODS):
        row = summary.iloc[index]
        value, low, high = row["J@1"], row["J@1 scene CI low"], row["J@1 scene CI high"]
        ax.errorbar(value, index, xerr=[[value-low], [high-value]], fmt="o", color=COLORS[method], capsize=3, label=DISPLAY[method])
    ax.set_yticks(range(3), [DISPLAY[item] for item in METHODS]); ax.set_xlabel("J@1 (scene-cluster 95% CI)"); ax.grid(axis="x", alpha=.25)
    save_figure(fig, run, "01_j1_forest")
    # 2 J@K
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for method in METHODS:
        frame = main[main.method == method]
        ax.plot(range(1, 6), [frame[f"j_at_{k}"].mean() for k in range(1,6)], marker="o", color=COLORS[method], label=DISPLAY[method])
    ax.set(xticks=range(1,6), xlabel="Native K", ylabel="J@K", ylim=(0,1)); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, run, "02_j_at_k")
    # 3 mask bins
    fig, ax = plt.subplots(figsize=(6.2, 3.3)); bins=[0,.25,.5,.7,.9,1.000001]
    for method in METHODS:
        frame=main[main.method==method].copy(); frame["bin"]=pd.cut(frame.mask_iou,bins,right=False)
        rates=frame.groupby("bin",observed=False).j_at_1.mean(); ax.plot(range(len(rates)),rates,marker="o",color=COLORS[method],label=DISPLAY[method])
    ax.set_xticks(range(5),["[0,.25)","[.25,.5)","[.5,.7)","[.7,.9)","[.9,1]"]); ax.set(ylabel="J@1",xlabel="Method-specific mask IoU bin",ylim=(0,1)); ax.grid(alpha=.25); ax.legend()
    save_figure(fig, run, "03_mask_iou_bin_j1")
    # 4 expression
    expression=strata[strata["Stratum type"]=="expression_type"].set_index("Stratum"); categories=expression.index.tolist(); x=np.arange(len(categories)); width=.24
    fig,ax=plt.subplots(figsize=(7,3.4))
    for offset,method in enumerate(METHODS): ax.bar(x+(offset-1)*width,expression[f"{method} J@1"],width,color=COLORS[method],label=DISPLAY[method])
    ax.set_xticks(x,categories); ax.set(ylabel="J@1",ylim=(0,1)); ax.legend(ncol=3); ax.grid(axis="y",alpha=.2)
    save_figure(fig,run,"04_expression_types")
    # 5 failure stack
    categories=[c for c in taxonomy.columns if c!="Method"]
    fig,ax=plt.subplots(figsize=(7,3.7)); bottom=np.zeros(2)
    palette=plt.get_cmap("tab20").colors
    for i,category in enumerate(categories): values=taxonomy[category].to_numpy(); ax.bar(taxonomy.Method,values,bottom=bottom,label=category,color=palette[i]); bottom+=values
    ax.set_ylabel("Samples"); ax.legend(fontsize=6,ncol=2,loc="upper center",bbox_to_anchor=(.5,-.15))
    save_figure(fig,run,"05_modular_failure_decomposition")
    # 6 IoU-angle (deterministic downsample per method)
    fig,ax=plt.subplots(figsize=(5.4,4.2))
    for method in METHODS:
        frame=main[(main.method==method)&main.rotated_iou.notna()].copy(); frame["key"]=frame.sample_id.map(lambda x:hashlib.sha256(x.encode()).hexdigest()); frame=frame.sort_values("key").head(min(1600,len(frame)))
        ax.scatter(frame.rotated_iou,frame.angle_error_deg,s=5,alpha=.25,color=COLORS[method],label=DISPLAY[method],rasterized=True)
    ax.axvline(.25,color="black",ls="--",lw=1); ax.axhline(30,color="black",ls="--",lw=1); ax.set(xlabel="Rotated raster IoU",ylabel="Periodic angle error (deg)",ylim=(0,90)); ax.legend(markerscale=2)
    save_figure(fig,run,"06_iou_vs_angle")
    # 7 disagreements
    pivot=main.pivot(index="sample_id",columns="method",values="j_at_1").astype(int); matrix=np.zeros((3,3))
    for i,a in enumerate(METHODS):
        for j,b in enumerate(METHODS): matrix[i,j]=np.mean(pivot[a]!=pivot[b])
    fig,ax=plt.subplots(figsize=(4.2,3.5)); image=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=max(matrix.max(),.01));
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center")
    ax.set_xticks(range(3),METHODS);ax.set_yticks(range(3),METHODS);fig.colorbar(image,ax=ax,label="Disagreement rate")
    save_figure(fig,run,"07_pairwise_disagreement")
    # 8 latency placeholder is replaced by latency benchmark
    fig,ax=plt.subplots(figsize=(5.4,2.8)); ax.text(.5,.5,"Latency benchmark pending\n(common MPS, batch=1, 20+200)",ha="center",va="center");ax.axis("off")
    save_figure(fig,run,"08_latency_distribution")
    # 9 UpSet-style bars
    fig,ax=plt.subplots(figsize=(6.4,3.4)); ax.bar(outcomes["Outcome combination"],outcomes.Count,color="#666666");ax.tick_params(axis="x",rotation=40);ax.set_ylabel("Samples")
    save_figure(fig,run,"09_three_method_upset")
    pd.DataFrame(matrix,index=METHODS,columns=METHODS).to_csv(run/"06_failure_analysis/pairwise_disagreement_matrix.csv")


def reports(run: Path, summary: pd.DataFrame, stats: pd.DataFrame, tests: dict[str, Any], continuous: pd.DataFrame, taxonomy: pd.DataFrame, outcomes: pd.DataFrame, samples: pd.DataFrame) -> None:
    geometry = f"""# Geometry contract

The frozen evaluator is `{run / 'config/canonical_evaluator.py'}`. Its SHA-256 is `{sha256_file(run / 'config/canonical_evaluator.py')}`.

- Original 640×480 coordinates; x is column and y is row.
- Angles are normalized to [-90°,90°), with modulo-180° error.
- A prediction succeeds only when one same GT has rasterized rotated IoU >0.25 and angle error ≤30°.
- GT uses centre=(corner0+corner2)/2, jaw axis corner3-corner0, jaw width clipped to 100 px, rectangle height 20 px.
- CROG predictions retain height 20 px. G1/C1 retain the upstream predicted jaw width and height=width/2. The evaluator never rewrites prediction geometry.
- Primary IoU is corrected clipped raster IoU. OpenCV and Shapely continuous polygon IoU are independent validation implementations, not the main verdict.
"""
    (run / "09_reports/GEOMETRY_CONTRACT.md").write_text(geometry)
    q = tests["cochran_q"]
    report = ["# Fair paired CROG / HiFi-CS→G1 / HiFi-CS→C1 comparison", "", "## Scope", "", "These are three concrete systems evaluated on the same OCID-VLG samples in their native input configurations. The result is not a causal claim about end-to-end versus modular architecture. J@1 measures agreement with annotated planar grasps, not physical robot success.", "", "## Main results", "", markdown_table(summary), "", f"Cochran Q={q['statistic']:.6f}, p={q['pvalue']:.6g}.", "", markdown_table(stats), "", "## Continuous geometry", "", markdown_table(continuous), "", "## Modular counterfactual failure evidence", "", markdown_table(taxonomy), "", "GT-mask rows are offline diagnostics, not deployable methods and not in the leaderboard. Categories are counterfactual evidence, not strict causal attribution.", "", "## Complementarity", "", markdown_table(outcomes), "", "## Limitations", "", "- Official OCID-VLG splits reuse capture-sequence families but not exact RGB frames; this run verifies zero exact scene-record/RGB/RGBD overlap and reports the sequence-family limitation.", "- Input modalities, pretrained models, training procedures, and architectures are not isolated as one causal variable.", "- The offline rectangle criterion does not measure collision-free or physical grasp success.", "- Native candidate-pool sizes differ; Oracle@All is candidate-generation diagnostic only.", "- CROG mask/grasp association is descriptive; no GT-mask intervention is available for CROG."]
    (run / "09_reports/FAIR_COMPARISON_FINAL_REPORT.md").write_text("\n".join(report)+"\n")
    crog = samples[samples.method=="CROG"]
    assoc = f"""# CROG associative failure analysis

- Low mask IoU (<0.50) and J@1 failure: {int(((crog.mask_iou<.5)&~crog.j_at_1).sum())}
- High mask IoU (≥0.90) but J@1 failure: {int(((crog.mask_iou>=.9)&~crog.j_at_1).sum())}
- Low mask IoU (<0.50) but J@1 success: {int(((crog.mask_iou<.5)&crog.j_at_1).sum())}
- IoU passes but angle fails: {int(((crog.rotated_iou>.25)&(crog.angle_error_deg>30)).sum())}
- Angle passes but IoU fails: {int(((crog.angle_error_deg<=30)&(crog.rotated_iou<=.25)).sum())}
- No output: {int(crog.no_output.sum())}

These are associations or patterns consistent with failure modes; they do not establish that mask quality caused grasp failure.
"""
    (run / "06_failure_analysis/crog_failure_analysis.md").write_text(assoc)
    (run / "06_failure_analysis/modular_failure_decomposition.md").write_text("# Modular failure decomposition\n\n"+markdown_table(taxonomy)+"\n\nCounts plus Top-1 success equal paired N for each method. GT-mask comparisons are offline counterfactual evidence only.\n")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-dir",type=Path,required=True);parser.add_argument("--crog-predictions",type=Path,required=True);parser.add_argument("--force-canonical",action="store_true");args=parser.parse_args()
    run=args.run_dir.resolve()
    if not args.force_canonical and (run / "03_canonical/canonical_candidates.parquet").exists() and (run / "04_metrics/per_sample_metrics.parquet").exists() and (run / "02_predictions/hifics_masks.parquet").exists():
        canonical=pd.read_parquet(run / "03_canonical/canonical_candidates.parquet")
        samples=pd.read_parquet(run / "04_metrics/per_sample_metrics.parquet")
        masks=pd.read_parquet(run / "02_predictions/hifics_masks.parquet")
        print("reusing completed canonical evaluation", flush=True)
    else:
        candidates,crog_masks=extract_and_validate(run,args.crog_predictions.resolve())
        masks=compute_masks(run,crog_masks)
        canonical,samples=evaluate_all(run,candidates,masks)
    summary,stats,tests=summarize_and_stats(run,samples,masks)
    continuous=continuous_summary(run,samples)
    taxonomy,outcomes=failure_analysis(run,samples)
    strata=stratified(run,samples,masks)
    formal_tables(run,summary,continuous,stats,strata,taxonomy,outcomes)
    figures(run,samples,summary,taxonomy,outcomes,strata)
    reports(run,summary,stats,tests,continuous,taxonomy,outcomes,samples)
    print(summary.to_string(index=False)); print(stats.to_string(index=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
