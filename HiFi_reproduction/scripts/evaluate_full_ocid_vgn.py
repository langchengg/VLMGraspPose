#!/usr/bin/env python3
"""Export denominator-explicit metrics from a full-run SQLite database."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.experiments.bootstrap import cluster_bootstrap_interval
from src.experiments.experiment_store import ExperimentStore
from src.experiments.failure_taxonomy import classify_status, is_candidate_outcome, is_terminal
from src.experiments.metrics import export_metrics, wilson_interval
from src.grasping.vgn_pipeline import atomic_write_csv, atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--cluster-key", default="scene_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def _open_rows(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = output / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    database = Path(config.get("database", output / "experiment.sqlite3"))
    if not database.is_file():
        raise FileNotFoundError(f"missing experiment database: {database}")
    with ExperimentStore(database, str(config["run_id"])) as store:
        rows = store.sample_rows()
    return config, rows


def _rate(numerator: int, denominator: int, name: str) -> dict[str, Any]:
    value = wilson_interval(numerator, denominator)
    value["name"] = name
    value["percentage"] = None if value["estimate"] is None else 100.0 * value["estimate"]
    return value


def _finite(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _quantiles(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray(
        [value for row in rows if (value := _finite(row, key)) is not None], dtype=np.float64
    )
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _bin(value: Any, edges: Sequence[float], labels: Sequence[str]) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(numeric):
        return "unknown"
    return labels[int(np.digitize([numeric], edges, right=False)[0])]


def _decorate_groups(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        area = _finite(row, "mask_area")
        row["mask_area_fraction"] = area / (640 * 480) if area is not None else None
        row["mask_area_bin"] = _bin(
            row["mask_area_fraction"], [0.005, 0.02, 0.08, 0.25],
            ["<0.5%", "0.5-2%", "2-8%", "8-25%", ">=25%"],
        )
        row["valid_depth_points_bin"] = _bin(
            row.get("valid_target_depth_points"), [100, 500, 2_000, 10_000],
            ["<100", "100-499", "500-1999", "2000-9999", ">=10000"],
        )
        row["intrinsics_fit_error_bin"] = _bin(
            row.get("fit_rmse_px"), [0.2, 0.4, 0.8, 1.5],
            ["<0.2px", "0.2-0.4px", "0.4-0.8px", "0.8-1.5px", ">=1.5px"],
        )
        row["support_plane_rmse_bin"] = _bin(
            row.get("support_plane_residual"), [0.002, 0.004, 0.008, 0.016],
            ["<2mm", "2-4mm", "4-8mm", "8-16mm", ">=16mm"],
        )
        row["mask_iou_bin"] = _bin(
            row.get("pred_mask_iou", row.get("mask_iou")), [0.5, 0.6, 0.7, 0.8, 0.9],
            ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", ">=0.9"],
        )
        row["official_candidate_count_bin"] = _bin(
            row.get("official_candidate_count"), [1, 3, 6, 11],
            ["0", "1-2", "3-5", "6-10", ">=11"],
        )
        row.setdefault("query_type", "unknown")
        row.setdefault("target_category", "unknown")


def _group_rows(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    results = []
    for value, members in sorted(grouped.items()):
        evaluable = [row for row in members if is_candidate_outcome(str(row["status"]))]
        official = sum((_finite(row, "official_candidate_count") or 0) > 0 for row in evaluable)
        target = sum((_finite(row, "target_candidate_count") or 0) > 0 for row in evaluable)
        results.append(
            {
                "group_field": key,
                "group_value": value,
                "manifest_count": len(members),
                "vgn_evaluable_count": len(evaluable),
                "official_candidate_positive": official,
                "target_candidate_positive": target,
                "official_candidate_coverage": official / len(evaluable) if evaluable else None,
                "target_candidate_coverage": target / len(evaluable) if evaluable else None,
                "conditional_target_candidate_coverage": target / official if official else None,
            }
        )
    return results


def _paired_oracle(
    predicted: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    *,
    output: Path,
    replicates: int,
    seed: int,
    cluster_key: str = "scene_id",
) -> dict[str, Any]:
    pred = {str(row["sample_id"]): row for row in predicted}
    gt = {str(row["sample_id"]): row for row in oracle}
    if set(pred) != set(gt):
        return {
            "gt_oracle_available": False,
            "reason": "predicted and oracle stores do not contain identical sample IDs",
            "predicted_count": len(pred),
            "oracle_count": len(gt),
        }
    paired: list[dict[str, Any]] = []
    for sample_id in sorted(pred, key=lambda value: int(pred[value]["dataset_index"])):
        left, right = pred[sample_id], gt[sample_id]
        pred_official_count = _finite(left, "official_candidate_count") or 0.0
        oracle_official_count = _finite(right, "official_candidate_count") or 0.0
        pred_target_count = _finite(left, "target_candidate_count") or 0.0
        oracle_target_count = _finite(right, "target_candidate_count") or 0.0
        paired.append(
            {
                "sample_id": sample_id,
                "scene_id": left["scene_id"],
                "pred_status": left["status"],
                "oracle_status": right["status"],
                "pred_official_positive": int(pred_official_count > 0),
                "oracle_official_positive": int(oracle_official_count > 0),
                "pred_target_positive": int(pred_target_count > 0),
                "oracle_target_positive": int(oracle_target_count > 0),
                "pred_no_official_grasp": int(left["status"] == "no_official_grasp"),
                "oracle_no_official_grasp": int(right["status"] == "no_official_grasp"),
                "pred_no_target_grasp": int(left["status"] == "no_target_grasp"),
                "oracle_no_target_grasp": int(right["status"] == "no_target_grasp"),
                "pred_target_given_official": (
                    int(pred_target_count > 0) if pred_official_count > 0 else None
                ),
                "oracle_target_given_official": (
                    int(oracle_target_count > 0) if oracle_official_count > 0 else None
                ),
                "pred_candidate_count_retention": (
                    pred_target_count / pred_official_count
                    if pred_official_count > 0
                    else None
                ),
                "oracle_candidate_count_retention": (
                    oracle_target_count / oracle_official_count
                    if oracle_official_count > 0
                    else None
                ),
                "pred_top1_vgn_quality": _finite(left, "top1_vgn_quality"),
                "oracle_top1_vgn_quality": _finite(right, "top1_vgn_quality"),
                "pred_processing_time_total": _finite(left, "processing_time_total"),
                "oracle_processing_time_total": _finite(right, "processing_time_total"),
            }
        )
    atomic_write_csv(
        output / "pred_vs_gt_oracle.csv",
        paired,
        tuple(paired[0]) if paired else ("sample_id",),
    )
    deltas: dict[str, Any] = {
        "gt_oracle_available": True,
        "paired_sample_count": len(paired),
        "paired_cluster_key": cluster_key,
        "delta_definition": "gt_oracle minus predicted",
    }

    def marginal_summary(key: str) -> dict[str, Any]:
        values = [float(value) for row in paired if (value := row.get(key)) is not None]
        return {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
        }

    for name, before, after in (
        ("official_candidate_coverage", "pred_official_positive", "oracle_official_positive"),
        ("target_candidate_coverage", "pred_target_positive", "oracle_target_positive"),
        ("no_official_grasp_rate", "pred_no_official_grasp", "oracle_no_official_grasp"),
        ("no_target_grasp_rate", "pred_no_target_grasp", "oracle_no_target_grasp"),
        (
            "target_given_official_availability",
            "pred_target_given_official",
            "oracle_target_given_official",
        ),
        (
            "candidate_count_retention_ratio",
            "pred_candidate_count_retention",
            "oracle_candidate_count_retention",
        ),
        ("top1_vgn_quality", "pred_top1_vgn_quality", "oracle_top1_vgn_quality"),
        (
            "processing_time_total",
            "pred_processing_time_total",
            "oracle_processing_time_total",
        ),
    ):
        delta_rows = []
        for row in paired:
            before_value, after_value = row.get(before), row.get(after)
            if before_value is None or after_value is None:
                continue
            delta_rows.append(
                {
                    **row,
                    "paired_delta": float(after_value) - float(before_value),
                }
            )
        deltas[name] = {
            "paired_finite_count": len(delta_rows),
            "predicted": (
                float(np.mean([float(row[before]) for row in delta_rows]))
                if delta_rows
                else None
            ),
            "gt_oracle": (
                float(np.mean([float(row[after]) for row in delta_rows]))
                if delta_rows
                else None
            ),
            "predicted_marginal": marginal_summary(before),
            "gt_oracle_marginal": marginal_summary(after),
            "paired_scene_cluster_delta": cluster_bootstrap_interval(
                delta_rows,
                "paired_delta",
                replicates=replicates,
                seed=seed,
                cluster_key=cluster_key,
            )
            if delta_rows
            else None,
        }
    atomic_write_json(output / "oracle_delta.json", deltas)
    return deltas


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    config, rows = _open_rows(output)
    manifest_count = int(config["expected_manifest_count"])
    if len(rows) != manifest_count:
        raise RuntimeError(f"database has {len(rows)} rows; expected {manifest_count}")
    terminal_count = sum(is_terminal(str(row["status"])) for row in rows)
    if terminal_count != manifest_count and not args.allow_partial:
        raise RuntimeError(
            f"only {terminal_count}/{manifest_count} rows have a terminal outcome; use --allow-partial for a shard report"
        )
    _decorate_groups(rows)
    metrics_dir = output / "metrics"
    exported = export_metrics(
        rows,
        metrics_dir,
        manifest_count=manifest_count,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        require_parquet=True,
    )
    aggregate = dict(exported["aggregate"])
    evaluable = [row for row in rows if is_candidate_outcome(str(row["status"]))]
    official = [row for row in evaluable if (_finite(row, "official_candidate_count") or 0) > 0]
    target = [row for row in evaluable if (_finite(row, "target_candidate_count") or 0) > 0]
    technical = [
        row for row in rows if classify_status(str(row["status"])).category.endswith("failure")
    ]
    aggregate.update(
        experiment_scope="OCID-VLG language-guided single-view offline candidate coverage",
        physical_grasp_success_not_measured=True,
        counts={
            "N_manifest_total": manifest_count,
            "N_input_files_complete": sum(str(row["status"]) not in {"missing_sample_file", "missing_rgb", "missing_depth", "missing_pred_mask"} for row in rows),
            "N_geometry_valid": len(evaluable),
            "N_vgn_evaluable": len(evaluable),
            "N_official_candidate_positive": len(official),
            "N_target_candidate_positive": len(target),
            "N_technical_failures": len(technical),
            "N_model_outcomes": len(evaluable),
        },
        coverage={
            "official_candidate_coverage": _rate(len(official), len(evaluable), "official_candidate_coverage"),
            "target_candidate_coverage": _rate(len(target), len(evaluable), "target_candidate_coverage"),
            "conditional_target_candidate_coverage": _rate(len(target), len(official), "conditional_target_candidate_coverage"),
        },
        requested_distributions={
            key: _quantiles(rows, key)
            for key in (
                "fit_rmse_px",
                "fit_p95_px",
                "valid_target_depth_points",
                "mask_area_fraction",
                "official_candidate_count",
                "target_candidate_count",
                "top1_vgn_quality",
                "top1_width_m",
                "processing_time_total",
            )
        },
    )
    gt_rows = [row for row in rows if isinstance(row.get("top1_inside_gt_target_mask"), bool)]
    gt_positive = sum(bool(row["top1_inside_gt_target_mask"]) for row in gt_rows)
    aggregate["target_consistency"] = {
        "top1_inside_gt_target_mask_rate": _rate(
            gt_positive, len(gt_rows), "top1_inside_gt_target_mask_rate"
        ),
        "top1_nearest_gt_target_point_distance_m": _quantiles(
            rows, "top1_nearest_gt_target_point_distance_m"
        ),
        "top1_projected_depth_error_m": _quantiles(rows, "top1_projected_depth_error_m"),
        "not_physical_grasp_success": True,
        "not_6dof_ground_truth_accuracy": True,
    }
    atomic_write_json(metrics_dir / "aggregate.json", aggregate)
    aggregate_table = []
    for section in ("counts", "coverage"):
        for key, value in aggregate[section].items():
            if isinstance(value, Mapping):
                aggregate_table.append({"section": section, "metric": key, **value})
            else:
                aggregate_table.append({"section": section, "metric": key, "value": value})
    atomic_write_csv(
        metrics_dir / "aggregate.csv",
        aggregate_table,
        ("section", "metric", "value", "numerator", "denominator", "estimate", "percentage", "ci_lower", "ci_upper", "confidence", "method", "name"),
    )
    failure_counts = Counter(str(row["status"]) for row in rows if row["status"] != "ok")
    atomic_write_csv(
        metrics_dir / "failure_counts.csv",
        [
            {
                "status": status,
                "category": classify_status(status).category,
                "count": count,
                "unconditional_rate": count / manifest_count,
            }
            for status, count in sorted(failure_counts.items())
        ],
        ("status", "category", "count", "unconditional_rate"),
    )
    atomic_write_csv(
        metrics_dir / "runtime.csv",
        rows,
        (
            "sample_id", "scene_id", "processing_time_depth", "processing_time_tsdf",
            "processing_time_vgn", "processing_time_render", "processing_time_total", "status",
        ),
    )
    confidence_rows = [
        {"metric": key, **value}
        for key, value in aggregate["coverage"].items()
    ]
    for key, value in aggregate.get("scene_cluster_bootstrap", {}).items():
        confidence_rows.append({"metric": key, **value})
    atomic_write_csv(
        metrics_dir / "confidence_intervals.csv",
        confidence_rows,
        tuple(sorted({key for row in confidence_rows for key in row})),
    )
    dimensions = (
        "query_type", "target_category", "scene_id", "view", "mask_area_bin",
        "valid_depth_points_bin", "intrinsics_fit_error_bin", "support_plane_rmse_bin",
        "mask_iou_bin", "official_candidate_count_bin",
    )
    all_groups = []
    for dimension in dimensions:
        grouped = _group_rows(rows, dimension)
        all_groups.extend(grouped)
        atomic_write_csv(
            metrics_dir / f"by_{dimension}.csv",
            grouped,
            tuple(grouped[0]) if grouped else ("group_field", "group_value"),
        )
    atomic_write_csv(
        metrics_dir / "grouped_results.csv",
        all_groups,
        tuple(all_groups[0]) if all_groups else ("group_field", "group_value"),
    )
    oracle = {"gt_oracle_available": False, "reason": "oracle output not provided"}
    if args.oracle_output is not None:
        _, oracle_rows = _open_rows(args.oracle_output.expanduser().resolve())
        oracle = _paired_oracle(
            rows,
            oracle_rows,
            output=metrics_dir,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
            cluster_key=args.cluster_key,
        )
    else:
        atomic_write_json(metrics_dir / "oracle_delta.json", oracle)
    return {"aggregate": aggregate, "oracle": oracle, "metrics_dir": str(metrics_dir)}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = evaluate(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
