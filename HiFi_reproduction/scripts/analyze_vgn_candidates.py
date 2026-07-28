"""Analyze frozen predicted-mask and GT-regenerated official VGN pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import ANALYSIS_SCHEMA_VERSION
from src.analysis.analysis_report import (
    atomic_json,
    build_figures,
    build_report,
    build_tables,
)
from src.analysis.candidate_loader import EXPECTED_SAMPLE_COUNT, load_analysis_tables
from src.analysis.diagnostic_features import (
    DiagnosticThresholds,
    add_secondary_diagnostic_features,
    iter_no_official_stage_diagnostics,
)
from src.analysis.failure_taxonomy import assign_failure_taxonomy
from src.analysis.gt_candidate_labels import label_candidate_pools
from src.analysis.human_audit import build_human_audit
from src.analysis.multiplicity import (
    PRIMARY_ROTATION_THRESHOLD_DEG,
    PRIMARY_TRANSLATION_THRESHOLD_M,
    PRIMARY_WIDTH_THRESHOLD_M,
    build_multiplicity_table,
)
from src.analysis.oracle_bounds import build_oracle_analysis
from src.analysis.reranking_opportunity import (
    build_opportunity_table,
    first_positive_recall,
    opportunity_funnel,
    sample_cluster_bootstrap_quality_auc,
)


LOGGER = logging.getLogger("vgn_candidate_analysis")

MANIFEST_FIELD_MAPPING = {
    "sample_id": "sample_id",
    "dataset_index": "sample_index",
    "instruction": "query",
    "scene_id": "scene_id",
    "bundle_dir": "output_dir",
    "rgb_path": "bundle metadata source_rgb",
    "depth_path": "bundle metadata source_depth",
    "mask_path": "bundle target_mask.png",
    "intrinsics": "bundle/saved workspace intrinsics",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Candidate multiplicity, target-consistency oracle, and failure analysis."
    )
    parser.add_argument("--pred-output", type=Path, required=True)
    parser.add_argument("--gt-oracle-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ocid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-gt-dilation-px", type=int, default=3, choices=(3,))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--cluster-key", default="scene_id", choices=("scene_id",))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnose-no-official", action="store_true")
    parser.add_argument("--skip-no-official-diagnostics", action="store_true")
    parser.add_argument("--skip-union-oracle", action="store_true")
    parser.add_argument("--build-human-audit", action="store_true")
    parser.add_argument("--audit-per-class", type=int, default=30)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--vgn-root", type=Path, default=Path("third_party/vgn"))
    parser.add_argument(
        "--vgn-weights", type=Path, default=Path("third_party/vgn/data/models/vgn_conv.pth")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _portable_frame(frame: pd.DataFrame, *, stringify_nested: bool) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include="object"):
        non_null = result[column].dropna()
        if non_null.empty:
            continue
        if any(isinstance(value, (dict, list, tuple, set, np.ndarray)) for value in non_null.head(20)):
            if stringify_nested:
                result[column] = result[column].map(
                    lambda value: None
                    if value is None
                    else json.dumps(value, default=_jsonable, sort_keys=True, ensure_ascii=False)
                )
            else:
                # Arrow accepts consistent lists but heterogeneous nested
                # structs are safer as explicit JSON with a documented schema.
                if any(isinstance(value, (dict, set)) for value in non_null.head(20)):
                    result[column] = result[column].map(
                        lambda value: None
                        if value is None
                        else json.dumps(value, default=_jsonable, sort_keys=True, ensure_ascii=False)
                    )
    return result


def _rotation_matrix_xyzw(quaternion: Any) -> list[list[float]]:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _candidate_export_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    result["rotation_camera_3x3"] = result.quaternion_camera_xyzw.map(
        _rotation_matrix_xyzw
    )
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        _portable_frame(frame, stringify_nested=False).to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _portable_frame(frame, stringify_nested=True).to_csv(temporary, index=False)
    temporary.replace(path)


def _stage_diagnostics(
    samples: pd.DataFrame,
    *,
    output: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    path = output / "data" / "no_official_stage_diagnostics.csv"
    existing = pd.read_csv(path) if args.resume and path.is_file() else pd.DataFrame()
    completed = set(existing.get("sample_id", pd.Series(dtype=str)).astype(str))
    requested = samples.loc[
        samples.pred_status.eq("no_official_grasp")
        & ~samples.sample_id.astype(str).isin(completed)
    ]
    if len(requested):
        if not args.vgn_weights.is_file():
            raise FileNotFoundError(
                f"pinned VGN checkpoint required for stage diagnostics: {args.vgn_weights}"
            )
        rows: list[dict[str, Any]] = existing.to_dict(orient="records")
        for index, diagnostic in enumerate(
            iter_no_official_stage_diagnostics(
                requested,
                vgn_weights=args.vgn_weights,
                vgn_root=args.vgn_root,
                device=args.device,
            ),
            start=1,
        ):
            rows.append(diagnostic.to_dict())
            if index % 25 == 0:
                _atomic_csv(pd.DataFrame(rows), path)
                LOGGER.info("no-official stage diagnostics: %d/%d new samples", index, len(requested))
        existing = pd.DataFrame(rows)
        _atomic_csv(existing, path)
    if len(existing) != int(samples.pred_status.eq("no_official_grasp").sum()):
        raise RuntimeError(
            "no-official stage diagnostics are incomplete: "
            f"{len(existing)} != {int(samples.pred_status.eq('no_official_grasp').sum())}"
        )
    return existing


def _expand_stage_flags(samples: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    if stages.empty:
        samples = samples.copy()
        samples["stage_diagnostics_available"] = False
        return samples
    stage = stages.copy()
    flags = (
        "S_no_quality_above_threshold",
        "S_removed_by_surface_filter",
        "S_removed_by_width_filter",
        "S_removed_by_nms",
        "S_unknown_candidate_generation_failure",
    )
    for flag in flags:
        stage[flag] = stage.candidate_generation_secondary_flag.eq(flag)
    stage["S_workspace_boundary_risk"] = False
    stage["stage_diagnostics_available"] = True
    columns = [
        "sample_id",
        "raw_quality_max",
        "count_raw_quality_above_0_90",
        "count_after_gaussian",
        "count_after_surface_filter",
        "count_after_width_filter",
        "count_after_threshold",
        "count_after_3d_local_maximum",
        "first_zero_stage",
        "candidate_generation_secondary_flag",
        "tsdf_nonzero_count",
        "tsdf_nonzero_fraction",
        "S_empty_or_sparse_tsdf",
        "stage_diagnostics_available",
        *flags,
        "S_workspace_boundary_risk",
    ]
    result = samples.merge(stage[columns], on="sample_id", how="left", validate="one_to_one")
    result["stage_diagnostics_available"] = result.stage_diagnostics_available.fillna(False)
    for flag in (*flags, "S_empty_or_sparse_tsdf", "S_workspace_boundary_risk"):
        result[flag] = result[flag].fillna(False).astype(bool)
    boundary = result.get("S_target_near_workspace_boundary", False)
    result["S_workspace_boundary_risk"] = (
        result["S_workspace_boundary_risk"] | pd.Series(boundary, index=result.index).fillna(False)
    )
    return result


def _recommendation(samples: pd.DataFrame) -> dict[str, Any]:
    filter_count = int(samples.opportunity_class.eq("filter_recoverable").sum())
    post_count = int(samples.opportunity_class.eq("post_filter_ranking_recoverable").sum())
    generation_count = int(samples.opportunity_class.eq("generation_limited").sum())
    no_official = int(samples.opportunity_class.eq("no_official_candidate").sum())
    if filter_count > post_count:
        insertion = (
            "Apply any future re-ranker to all official candidates before the hard mask filter; "
            "replace the hard decision with a soft target-consistency feature."
        )
    else:
        insertion = (
            "The larger recoverable subset is within the filtered pool; a post-filter study is "
            "supported, while pre-filter results must still be reported."
        )
    failures = len(samples) - int(samples.hard_filter_top1_is_gt_positive.sum())
    generation_dominates = generation_count + no_official > failures / 2
    rankable = int((samples.n_distinct_pose_modes >= 2).sum())
    recoverable = filter_count + post_count
    pairs = int(samples.n_positive_negative_candidate_pairs.sum())
    training_sufficiency = (
        "Candidate multiplicity and pair counts are numerically sufficient for a future grouped "
        "train/validation study, but the present task does not train or validate an MLP/GNN."
        if rankable >= 1_000 and recoverable >= 200 and pairs >= 1_000
        else "The recoverable/rankable subset is too small to justify a learned re-ranker."
    )
    return {
        "reranker_insertion": insertion,
        "generation_change_required": generation_dominates,
        "generation_limited_plus_no_official": generation_count + no_official,
        "current_target_inconsistent_or_missing_selection": failures,
        "samples_with_two_or_more_distinct_modes": rankable,
        "recoverable_errors": recoverable,
        "positive_negative_candidate_pairs": pairs,
        "training_sufficiency": training_sufficiency,
    }


def _quantile_bins(series: pd.Series, name: str) -> tuple[pd.Series, list[float]]:
    finite = pd.to_numeric(series, errors="coerce")
    edges = np.unique(finite.dropna().quantile([0, .25, .50, .75, 1]).to_numpy(float))
    if len(edges) < 2:
        return pd.Series("unavailable", index=series.index), edges.tolist()
    edges[-1] = np.nextafter(edges[-1], np.inf)
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    return pd.cut(finite, edges, labels=labels, include_lowest=True, right=False), edges.tolist()


def _write_grouped_opportunity(samples: pd.DataFrame, output: Path) -> dict[str, list[float]]:
    working = samples.copy()
    quantiles: dict[str, list[float]] = {}
    for source, target in (
        ("gt_mask_area_px", "target_mask_area_bin"),
        ("valid_target_depth_points", "target_valid_depth_bin"),
        ("support_plane_residual", "support_plane_rmse_bin"),
        ("fit_rmse_px", "intrinsics_fit_rmse_bin"),
    ):
        working[target], quantiles[target] = _quantile_bins(working[source], target)
    working["candidate_multiplicity_bin"] = working["official_candidate_count_bin"]
    dimensions = (
        "query_type",
        "mask_iou_bin",
        "target_mask_area_bin",
        "target_valid_depth_bin",
        "scene_id",
        "target_category",
        "candidate_multiplicity_bin",
        "support_plane_rmse_bin",
        "intrinsics_fit_rmse_bin",
    )
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        if dimension == "mask_iou_bin":
            working[dimension] = pd.cut(
                working.mask_iou,
                [0, .25, .50, .70, .90, 1.0000001],
                labels=["[0,.25)", "[.25,.50)", "[.50,.70)", "[.70,.90)", "[.90,1]"],
                include_lowest=True,
                right=False,
            )
        for (value, opportunity), group in working.groupby(
            [dimension, "opportunity_class"], observed=True, dropna=False
        ):
            denominator = int((working[dimension] == value).sum())
            rows.append(
                {
                    "dimension": dimension,
                    "group": str(value),
                    "opportunity_class": opportunity,
                    "numerator": len(group),
                    "denominator": denominator,
                    "percentage": 100.0 * len(group) / denominator if denominator else None,
                }
            )
    _atomic_csv(pd.DataFrame(rows), output / "tables" / "opportunity_grouped.csv")
    pose_rows = []
    for column in sorted(
        name for name in working if name == "n_distinct_pose_modes" or name.startswith("n_distinct_modes_")
    ):
        pose_rows.append(
            {
                "setting": column,
                "samples_with_two_or_more_modes": int((working[column] >= 2).sum()),
                "denominator": len(working),
                "percentage": 100.0 * (working[column] >= 2).mean(),
                "total_modes": int(working[column].sum()),
            }
        )
    _atomic_csv(pd.DataFrame(pose_rows), output / "tables" / "pose_mode_sensitivity.csv")
    distribution_rows = []
    contexts = {
        "official_all_samples": (working, "n_official_candidates"),
        "pred_filter_all_samples": (working, "n_pred_filtered_candidates"),
        "official_given_official_exists": (
            working.loc[working.n_official_candidates > 0],
            "n_official_candidates",
        ),
        "pred_filter_given_pred_filter_exists": (
            working.loc[working.n_pred_filtered_candidates > 0],
            "n_pred_filtered_candidates",
        ),
        "gt_positive_given_any": (
            working.loc[working.n_gt_positive_primary > 0],
            "n_gt_positive_primary",
        ),
    }
    order = ("0", "1", "2", "3", "4", "5–9", "10–19", "20–49", "50+")
    for context, (frame, column) in contexts.items():
        counts = frame[column].map(
            lambda value: (
                str(int(value))
                if value <= 4
                else "5–9"
                if value <= 9
                else "10–19"
                if value <= 19
                else "20–49"
                if value <= 49
                else "50+"
            )
        ).value_counts()
        distribution_rows.extend(
            {
                "context": context,
                "count_bin": label,
                "sample_count": int(counts.get(label, 0)),
                "denominator": len(frame),
                "percentage": 100.0 * counts.get(label, 0) / len(frame) if len(frame) else None,
            }
            for label in order
        )
    _atomic_csv(
        pd.DataFrame(distribution_rows),
        output / "tables" / "multiplicity_distributions.csv",
    )
    return quantiles


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output.expanduser().resolve()
    completion = output / "analysis_complete.json"
    if completion.is_file() and args.resume and not args.force:
        return json.loads(completion.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "analysis.log"
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path
        for handler in logging.getLogger().handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
    LOGGER.info("manifest field adapter: %s", json.dumps(MANIFEST_FIELD_MAPPING, sort_keys=True))
    loaded = load_analysis_tables(args.pred_output, args.gt_oracle_output, args.manifest)
    LOGGER.info(
        "integrity load complete: %d samples, %d predicted candidates, %d regenerated candidates",
        len(loaded.samples),
        len(loaded.predicted_candidates),
        len(loaded.gt_regenerated_candidates),
    )
    samples = loaded.samples
    predicted = loaded.predicted_candidates
    regenerated = loaded.gt_regenerated_candidates
    if args.sample_id:
        selected = set(args.sample_id)
        missing = selected - set(samples.sample_id)
        if missing:
            raise ValueError(f"unknown --sample-id values: {sorted(missing)}")
        samples = samples.loc[samples.sample_id.isin(selected)].copy()
        predicted = predicted.loc[predicted.sample_id.isin(selected)].copy()
        regenerated = regenerated.loc[regenerated.sample_id.isin(selected)].copy()
    LOGGER.info(
        "computing shared GT geometry labels for %d predicted and %d regenerated candidates",
        len(predicted),
        len(regenerated),
    )
    predicted, regenerated = label_candidate_pools(samples, predicted, regenerated)
    LOGGER.info("GT geometry labeling complete")
    LOGGER.info("computing raw and distinct pose multiplicity")
    samples = build_multiplicity_table(samples, predicted)
    LOGGER.info("assigning P0-P6 taxonomy and secondary diagnostics")
    samples = assign_failure_taxonomy(samples, predicted)
    thresholds = DiagnosticThresholds()
    samples = add_secondary_diagnostic_features(samples, predicted, thresholds=thresholds)
    samples = build_opportunity_table(samples, predicted)
    LOGGER.info("computing same-pool, regenerated-pool, and union oracle bounds")
    oracle = build_oracle_analysis(
        samples,
        predicted,
        regenerated,
        include_union=not args.skip_union_oracle,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    samples = oracle.samples
    stages = pd.DataFrame()
    diagnose = args.diagnose_no_official and not args.skip_no_official_diagnostics
    if diagnose:
        LOGGER.info("starting no-official stage diagnostics")
        stages = _stage_diagnostics(samples, output=output, args=args)
        stage_counts = (
            stages.groupby(
                ["first_zero_stage", "candidate_generation_secondary_flag"],
                dropna=False,
            )
            .size()
            .reset_index(name="sample_count")
        )
        stage_counts["denominator"] = len(stages)
        stage_counts["percentage"] = 100.0 * stage_counts.sample_count / len(stages)
        _atomic_csv(stage_counts, output / "tables" / "no_official_stage_diagnostics.csv")
    samples = _expand_stage_flags(samples, stages)
    flag_columns = sorted(column for column in samples if column.startswith("S_"))
    samples["secondary_flags"] = samples.apply(
        lambda row: ";".join(column for column in flag_columns if bool(row.get(column, False))),
        axis=1,
    )
    data = output / "data"
    _atomic_parquet(samples, data / "per_sample.parquet")
    _atomic_parquet(_candidate_export_frame(predicted), data / "per_candidate.parquet")
    _atomic_parquet(
        _candidate_export_frame(regenerated),
        data / "per_candidate_gt_regenerated.parquet",
    )
    if not oracle.union_candidates.empty:
        _atomic_parquet(oracle.union_candidates, data / "per_candidate_union_diagnostic.parquet")
    _atomic_csv(samples, data / "per_sample.csv")
    LOGGER.info("core tables persisted; building statistical tables and figures")
    _atomic_csv(opportunity_funnel(samples), output / "tables" / "opportunity_funnel.csv")
    config = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "primary_gt_label": "projected official grasp-pose origin inside GT mask dilated 3 px",
        "primary_gt_dilation_px": args.primary_gt_dilation_px,
        "score_source": "official_vgn_processed_quality",
        "custom_reranking": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "cluster_key": args.cluster_key,
        "pose_mode_thresholds": {
            "translation_m": PRIMARY_TRANSLATION_THRESHOLD_M,
            "rotation_deg": PRIMARY_ROTATION_THRESHOLD_DEG,
            "width_m": PRIMARY_WIDTH_THRESHOLD_M,
            "clustering": "deterministic transitive single-link/union-find in camera frame",
        },
        "diagnostic_thresholds": thresholds.to_dict(),
        "official_postprocessing": {
            "quality_threshold": 0.90,
            "gaussian_sigma": 1.0,
            "maximum_filter_size": 4,
            "min_width_voxels": 1.33,
            "max_width_voxels": 9.33,
        },
        "workspace_size_m": 0.30,
        "voxel_size_m": 0.0075,
        "depth_unit": "mm (explicit frozen-run provenance), converted once with scale 1000",
        "intrinsics_source": sorted(samples.intrinsics_source.unique().tolist()),
        "no_official_diagnostics_executed": diagnose,
        "union_oracle_executed": not args.skip_union_oracle,
    }
    config["empirical_grouping_quantile_edges"] = _write_grouped_opportunity(samples, output)
    atomic_json(output / "analysis_config.json", config)
    tables = build_tables(
        samples,
        predicted,
        output,
        oracle_table=oracle.oracle_upper_bounds,
        sensitivity_table=oracle.oracle_sensitivity,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    quality = sample_cluster_bootstrap_quality_auc(
        predicted,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    recall = first_positive_recall(samples)
    _atomic_csv(recall, output / "tables" / "recall_at_k.csv")
    figures = build_figures(samples, predicted, tables, output) if args.render else []
    if args.build_human_audit:
        audit = build_human_audit(
            samples,
            predicted,
            output / "human_audit",
            per_class=args.audit_per_class,
            seed=args.seed,
            render=args.render,
        )
    else:
        audit = pd.DataFrame()
    recommendation = _recommendation(samples)
    main_oracles = oracle.oracle_upper_bounds.loc[
        oracle.oracle_upper_bounds.denominator_scope.eq("all_samples")
    ].set_index("metric")
    headline = {
        "sample_count": len(samples),
        "candidate_count": len(predicted),
        "gt_regenerated_candidate_count": len(regenerated),
        "pre_filter_rankable": int(samples.pre_filter_rankable.sum()),
        "post_filter_rankable": int(samples.post_filter_rankable.sum()),
        "baseline_target_consistent": int(samples.hard_filter_top1_is_gt_positive.sum()),
        "opportunity_counts": samples.opportunity_class.value_counts().sort_index().to_dict(),
        "primary_failure_counts": samples.primary_failure_class.value_counts().sort_index().to_dict(),
        "recall_at_k": recall.set_index("metric")[["numerator", "denominator", "percentage"]].to_dict(orient="index"),
        "oracle_all_sample_rows": main_oracles[["numerator", "denominator", "percentage", "absolute_gain_percentage_points"]].to_dict(orient="index"),
        "quality_target_identity_diagnostic": quality,
        "no_official_stage_diagnostics_count": len(stages),
        "no_official_stage_flag_counts": (
            stages.candidate_generation_secondary_flag.value_counts(dropna=False).to_dict()
            if not stages.empty
            else {}
        ),
        "human_audit_status": "manual_audit_pending",
        "human_audit_sample_count": len(audit),
    }
    manifest = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "predicted_output": str(args.pred_output.resolve()),
        "gt_regenerated_output": str(args.gt_oracle_output.resolve()),
        "manifest_field_mapping": MANIFEST_FIELD_MAPPING,
        "integrity": dict(loaded.integrity),
        "row_counts": {
            "per_sample": len(samples),
            "per_candidate": len(predicted),
            "per_candidate_gt_regenerated": len(regenerated),
            "union_diagnostic": len(oracle.union_candidates),
        },
        "official_vgn": {
            "repository_url": "https://github.com/ethz-asl/vgn",
            "branch": "corl2020",
            "commit": "d7af0622433f52ae88ebe81533f12b46b33e951a",
            "checkpoint": str(args.vgn_weights.resolve()),
            "checkpoint_sha256": _sha256(args.vgn_weights),
        },
        "data_integrity_status": "complete" if len(samples) == EXPECTED_SAMPLE_COUNT else "explicit_subset",
    }
    atomic_json(data / "analysis_manifest.json", manifest)
    executive = {
        "integrity": dict(loaded.integrity),
        "headline_metrics": headline,
        "method_design_recommendation": recommendation,
        "limitations": [
            "single-view TSDF adaptation",
            "no 6-DoF ground truth in OCID-VLG",
            "no robot execution validation",
            "2-D projected-origin target consistency is not physical grasp correctness",
            "GT-regenerated pool changes task-frame geometry and is not a same-pool re-ranking bound",
        ],
        "human_audit_status": "manual_audit_pending",
    }
    atomic_json(output / "report" / "executive_summary.json", executive)
    build_report(output, integrity=loaded.integrity, executive=executive)
    completion_payload = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "headline_metrics": headline,
        "report": str(output / "report" / "report.html"),
        "figure_count": len(figures),
    }
    atomic_json(completion, completion_payload)
    return completion_payload


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.force and args.resume:
        LOGGER.warning("--force overrides --resume completion shortcut")
    try:
        result = run(args)
    except Exception:
        LOGGER.exception("candidate analysis failed")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
