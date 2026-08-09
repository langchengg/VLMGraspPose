#!/usr/bin/env python3
"""Evaluate locked post-hoc benchmark masks and all shared-bank baselines."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.conservative_mask_gate import (  # noqa: E402
    add_gate_evidence,
    proposed_alternatives,
)
from src.segmentation.proposal_evaluation import simple_baseline_decisions  # noqa: E402
from src.segmentation.proposal_statistics import (  # noqa: E402
    clustered_bootstrap_values,
    holm_adjust,
    paired_transitions,
    wilson_interval,
)
from src.segmentation.proposal_types import load_candidate_masks_npz  # noqa: E402
from src.segmentation.query_semantics import parse_query  # noqa: E402
from src.segmentation.selective_sam3_vg.evaluation import (  # noqa: E402
    load_frozen_ground_truth_manifest,
    load_ground_truth_mask,
)
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_binary_mask,
    load_compact_manifest,
    resize_binary_mask,
    sha256_file,
)
from src.segmentation.selective_sam3_vg.metrics import (  # noqa: E402
    THRESHOLDS,
    binary_mask_metrics,
    summarize_ious,
)


PAPER = {
    "mean_iou": 0.8826,
    "p_at_50": 0.9268,
    "p_at_60": 0.9213,
    "p_at_70": 0.9153,
    "p_at_80": 0.8969,
    "p_at_90": 0.8321,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _aggregate(frame: pd.DataFrame) -> dict:
    summary = summarize_ious(frame["iou"].to_numpy(float))
    for column in (
        "dice",
        "mask_precision",
        "mask_recall",
        "boundary_fscore",
        "false_positive_area_px",
        "false_negative_area_px",
        "connected_component_count",
    ):
        summary[f"mean_{column}"] = float(frame[column].mean())
    summary["low_precision_count_lt_0_5"] = int(
        np.count_nonzero(frame["mask_precision"].to_numpy() < 0.5)
    )
    summary["low_recall_count_lt_0_5"] = int(
        np.count_nonzero(frame["mask_recall"].to_numpy() < 0.5)
    )
    return summary


def _candidate_maps(decisions: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {
        method: dict(zip(group["sample_id"].astype(str), group["candidate_id"].astype(str)))
        for method, group in decisions.groupby("method", sort=False)
    }


def _sample_row_group(
    parquet: pq.ParquetFile, group: int, sample_id: str
) -> pd.DataFrame:
    frame = parquet.read_row_group(group).to_pandas()
    identities = set(frame["sample_id"].astype(str))
    if identities != {sample_id}:
        raise ValueError(
            f"candidate-label row-group drift at {group}: {identities} != {sample_id}"
        )
    return frame


def _unique_shared_runtime_seconds(
    runtimes: list[dict],
    compact_rows: list,
    family: str,
) -> float:
    """Count frame-shared text/automatic inference once, not once per query."""

    seen: set[str] = set()
    total = 0.0
    for runtime, row in zip(runtimes, compact_rows, strict=True):
        value = runtime.get(family, {})
        key = str(
            value.get("cache_key")
            or f"legacy:{family}:{row.raw['source_rgb_sha256']}"
        )
        if key in seen:
            continue
        seen.add(key)
        total += float(value.get("shared_runtime_seconds", 0.0))
    return total


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    experiment = args.experiment_root.expanduser().resolve()
    output_lock = experiment / "locked_benchmark_masks/LOCKED_BEFORE_GT_EVALUATION"
    if not output_lock.is_file():
        raise RuntimeError("final GT evaluation is forbidden before immutable output lock")
    lock_payload = json.loads(output_lock.read_text(encoding="utf-8"))
    manifest_path = experiment / "locked_benchmark_masks/immutable_output_manifest.jsonl"
    if sha256_file(manifest_path) != lock_payload["immutable_output_manifest_sha256"]:
        raise RuntimeError("immutable benchmark output manifest checksum mismatch")
    compact = load_compact_manifest(
        PROJECT_ROOT
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/test/manifest.jsonl",
        expected_split="test",
        expected_count=7675,
    )
    gt_rows = load_frozen_ground_truth_manifest(
        PROJECT_ROOT / "artifacts/data_audit/frozen_manifests/ocidvlg_unique_test.json",
        hifics_root=PROJECT_ROOT / "hifics",
        expected_count=7675,
    )
    gt_by_prefix = {f"q{int(row['question_index']):07d}_": row for row in gt_rows}
    stage1_label_file = pq.ParquetFile(
        experiment / "oracle_stage1_test/candidate_iou.parquet"
    )
    stage2_label_file = pq.ParquetFile(
        experiment / "oracle_stage2_test/candidate_iou.parquet"
    )
    if (
        stage1_label_file.num_row_groups != len(compact)
        or stage2_label_file.num_row_groups != len(compact)
    ):
        raise ValueError("test candidate-label row groups do not match frozen manifest")
    baseline_parts: list[pd.DataFrame] = []
    stage1_learned: dict[str, str] = {}
    stage2_proposed_frames = []
    locked_sources: dict[str, str] = {}
    locked_evidence: dict[str, dict] = {}
    eligible_stage1_counts: dict[str, int] = {}
    stage1_oracle_map: dict[str, str] = {}
    stage2_oracle_map: dict[str, str] = {}
    stage1_oracle_iou: dict[str, float] = {}
    stage2_oracle_iou: dict[str, float] = {}
    stage1_oracle_source: dict[str, str] = {}
    final_gate_method = json.loads(
        (PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1/final_selector/gate.json").read_text()
    )["selected_method"]
    if str(final_gate_method).startswith("HIFI_FALLBACK"):
        final_gate_method = "F1_hgb_classifier"
    for group, row in enumerate(compact):
        stage1_labels = _sample_row_group(
            stage1_label_file, group, row.sample_id
        )
        stage2_labels = _sample_row_group(
            stage2_label_file, group, row.sample_id
        )
        features = pd.read_parquet(
            experiment / "features/test_samples" / f"{row.sample_id}.parquet"
        )
        stage1_data = features.merge(
            stage1_labels[
                [
                    "sample_id",
                    "candidate_id",
                    "candidate_iou",
                    "continuous_iou",
                    "y70",
                    "y80",
                    "y90",
                ]
            ],
            on=["sample_id", "candidate_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(stage1_data) != len(features):
            raise ValueError(f"test Stage-1 feature/label join drift: {row.sample_id}")
        decisions, _ = simple_baseline_decisions(stage1_data)
        baseline_parts.append(decisions)
        eligible_stage1_counts[row.sample_id] = int(
            features["eligible_final"].sum()
        )
        stage1_best = (
            stage1_labels[stage1_labels["eligible_final"]]
            .sort_values(
                ["candidate_iou", "candidate_id"],
                ascending=[False, True],
                kind="stable",
            )
            .iloc[0]
        )
        stage2_best = (
            stage2_labels[stage2_labels["eligible_final"]]
            .sort_values(
                ["candidate_iou", "candidate_id"],
                ascending=[False, True],
                kind="stable",
            )
            .iloc[0]
        )
        stage1_oracle_map[row.sample_id] = str(stage1_best["candidate_id"])
        stage2_oracle_map[row.sample_id] = str(stage2_best["candidate_id"])
        stage1_oracle_iou[row.sample_id] = float(stage1_best["candidate_iou"])
        stage2_oracle_iou[row.sample_id] = float(stage2_best["candidate_iou"])
        stage1_oracle_source[row.sample_id] = str(stage1_best["source_family"])
        root = experiment / "locked_benchmark_masks" / row.sample_id
        stage1_learned[row.sample_id] = str(
            json.loads((root / "stage1_decision.json").read_text())["candidate_id"]
        )
        scores = pd.read_parquet(root / "stage2_scores.parquet")
        scores = scores.merge(
            stage2_labels[["sample_id", "candidate_id", "candidate_iou"]],
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
        scores = add_gate_evidence(scores)
        gate = json.loads((root / "gate_decision.json").read_text())
        stage2_proposed_frames.append(
            proposed_alternatives(scores, str(final_gate_method))
        )
        locked_sources[row.sample_id] = str(gate["source_family"])
        selected_evidence = scores[
            scores["candidate_id"].astype(str) == str(gate["candidate_id"])
        ]
        if len(selected_evidence) != 1:
            raise RuntimeError(f"locked candidate evidence mismatch: {row.sample_id}")
        locked_evidence[row.sample_id] = selected_evidence.iloc[0].to_dict()
    baseline_decisions = pd.concat(baseline_parts, ignore_index=True)
    candidate_maps = _candidate_maps(baseline_decisions)
    stage2_proposed = pd.concat(stage2_proposed_frames, ignore_index=True)
    stage2_proposed_map = dict(
        zip(stage2_proposed["sample_id"].astype(str), stage2_proposed["candidate_id"].astype(str))
    )
    method_candidate_maps = {
        "SAM3 full-query highest score": candidate_maps["B2_full_query_highest_sam"],
        "SAM3 target-category highest score": candidate_maps["B3_target_category_highest_sam"],
        "SAM3 target-attribute highest score": candidate_maps["B4_target_attribute_highest_sam"],
        "Maximum HiFi overlap": candidate_maps["B6_max_hifi_overlap"],
        "Deterministic relation-aware": candidate_maps["B9_relation_aware_rules"],
        "Learned Stage-1 selector": stage1_learned,
        "Stage-1 + Stage-2 selector (pre-gate)": stage2_proposed_map,
        "Stage-1 GT oracle": stage1_oracle_map,
        "Stage-2 GT oracle": stage2_oracle_map,
    }
    details: list[dict] = []
    selective_root = PROJECT_ROOT / "outputs/selective_sam3_vg/formal_test_masks"
    for number, row in enumerate(compact, start=1):
        prefix = row.sample_id.split("_", 1)[0] + "_"
        gt = load_ground_truth_mask(gt_by_prefix[prefix])
        stage1_masks = load_candidate_masks_npz(
            experiment / "proposals" / row.sample_id / "candidate_masks.npz"
        )
        stage2_masks = load_candidate_masks_npz(
            experiment / "stage2" / row.sample_id / "candidate_masks.npz"
        )
        locked_root = experiment / "locked_benchmark_masks" / row.sample_id
        masks: dict[str, np.ndarray] = {
            "Local HiFi-CS baseline": load_binary_mask(
                row.native_mask_path, expected_shape=(480, 640)
            ),
            "Existing selective SAM3": load_binary_mask(
                selective_root / row.sample_id / "final_mask.png", expected_shape=(480, 640)
            ),
            "Locked conservative method": load_binary_mask(
                locked_root / "final_mask.png", expected_shape=(480, 640)
            ),
        }
        for method, mapping in method_candidate_maps.items():
            candidate_id = mapping[row.sample_id]
            bank = stage2_masks if method in {
                "Stage-1 + Stage-2 selector (pre-gate)", "Stage-2 GT oracle"
            } else stage1_masks
            masks[method] = bank[candidate_id]
        semantics = parse_query(row.query)
        evidence = locked_evidence[row.sample_id]
        eligible_stage1_count = eligible_stage1_counts[row.sample_id]
        target_area = int(np.count_nonzero(gt))
        baseline_native = resize_binary_mask(masks["Local HiFi-CS baseline"], gt.shape)
        baseline_area = int(np.count_nonzero(baseline_native))
        baseline_iou = binary_mask_metrics(baseline_native, gt)["evaluator_iou_float32"]
        for method, native in masks.items():
            prediction = resize_binary_mask(native, gt.shape)
            metric = binary_mask_metrics(prediction, gt)
            metric["iou"] = metric["evaluator_iou_float32"]
            details.append(
                {
                    "sample_id": row.sample_id,
                    "sample_index": row.sample_index,
                    "frame_id": row.scene_id,
                    "scene_group": row.scene_id.split(",", 1)[0],
                    "query": row.query,
                    "query_type": semantics.query_type,
                    "relation_type": semantics.pairwise_relation,
                    "absolute_location_type": semantics.absolute_location,
                    "target_category": semantics.target_category,
                    "parser_confidence": semantics.parser_confidence,
                    "target_area_px": target_area,
                    "target_area_fraction": target_area / float(gt.size),
                    "baseline_iou": baseline_iou,
                    "baseline_area_ratio": baseline_area / max(target_area, 1),
                    "selected_source": (
                        locked_sources[row.sample_id]
                        if method == "Locked conservative method"
                        else None
                    ),
                    "selection_stage": (
                        "hifi"
                        if locked_sources[row.sample_id] == "STAGE2_HIFI_FALLBACK"
                        else "stage1"
                        if locked_sources[row.sample_id] == "STAGE2_SELECTED_STAGE1"
                        else "stage2"
                    ),
                    "switch_action": (
                        "keep_hifi"
                        if locked_sources[row.sample_id] == "STAGE2_HIFI_FALLBACK"
                        else "switch"
                    ),
                    "proposal_bank_size": eligible_stage1_count,
                    "stage1_oracle_source": stage1_oracle_source[row.sample_id],
                    "same_category_alternative_count": evidence.get(
                        "same_category_alternative_count", np.nan
                    ),
                    "selector_p90_probability": evidence.get(
                        "m1_p90_calibrated", np.nan
                    ),
                    "selector_p90_margin": evidence.get(
                        "predicted_p90_margin_over_hifi", np.nan
                    ),
                    "selector_iou_margin": evidence.get(
                        "predicted_iou_margin_over_hifi", np.nan
                    ),
                    "relation_features_valid": evidence.get(
                        "relation_features_valid", False
                    ),
                    "relation_consistency": evidence.get(
                        "relation_max_consistency", np.nan
                    ),
                    "requested_colour_match_fraction": evidence.get(
                        "requested_colour_match_fraction", np.nan
                    ),
                    "depth_reliability": evidence.get("depth_reliability", np.nan),
                    "source_consensus_count": evidence.get(
                        "source_consensus_count", np.nan
                    ),
                    "method": method,
                    **metric,
                }
            )
        if number % 100 == 0:
            print(f"locked evaluation: {number}/7675", flush=True)
    detail = pd.DataFrame(details)
    report = experiment / "report"
    report.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(report / "per_sample_method_metrics.parquet", index=False)
    metrics = {
        method: _aggregate(frame)
        for method, frame in detail.groupby("method", sort=False)
    }
    metrics["HiFi-CS paper numeric reference"] = {
        **PAPER,
        "protocol_equivalence_proven": False,
    }
    baseline_frame = detail[detail["method"] == "Local HiFi-CS baseline"].sort_values("sample_id")
    locked_frame = detail[detail["method"] == "Locked conservative method"].sort_values("sample_id")
    transitions = {}
    p_values = []
    for threshold in THRESHOLDS:
        key = f"p_at_{int(threshold * 100)}"
        transition = paired_transitions(
            baseline_frame["iou"].to_numpy(), locked_frame["iou"].to_numpy(), threshold
        )
        transitions[key] = transition
        p_values.append(float(transition["mcnemar_exact_p"]))
    adjusted = holm_adjust(p_values)
    for key, value in zip(transitions, adjusted, strict=True):
        transitions[key]["mcnemar_holm_adjusted_p"] = value
    proposed_source_by_sample = dict(
        zip(
            stage2_proposed["sample_id"].astype(str),
            stage2_proposed["source_family"].astype(str),
            strict=True,
        )
    )
    fallback_sources = {"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"}
    triggered = np.asarray(
        [
            proposed_source_by_sample[sample_id] not in fallback_sources
            for sample_id in baseline_frame["sample_id"].astype(str)
        ],
        dtype=bool,
    )
    accepted = np.asarray(
        [
            locked_sources[sample_id] not in fallback_sources
            for sample_id in baseline_frame["sample_id"].astype(str)
        ],
        dtype=bool,
    )
    baseline_values = baseline_frame["iou"].to_numpy(float)
    locked_values = locked_frame["iou"].to_numpy(float)
    p90_transition = transitions["p_at_90"]
    outcome_changes = p90_transition["recovered"] + p90_transition["harmed"]
    accepted_count = int(np.count_nonzero(accepted))
    gate_statistics = {
        "triggered_samples": int(np.count_nonzero(triggered)),
        "triggered_fraction": float(np.mean(triggered)),
        "accepted_switch_samples": accepted_count,
        "accepted_switch_fraction": float(np.mean(accepted)),
        "accepted_switch_improvement_precision": (
            float(np.mean(locked_values[accepted] > baseline_values[accepted]))
            if accepted_count
            else None
        ),
        "accepted_switch_strict_p90_precision": (
            float(np.mean(locked_values[accepted] > 0.90))
            if accepted_count
            else None
        ),
        "p90_recovered": int(p90_transition["recovered"]),
        "p90_harmed": int(p90_transition["harmed"]),
        "p90_net": int(p90_transition["net"]),
        "p90_outcome_changing_precision": (
            float(p90_transition["recovered"] / outcome_changes)
            if outcome_changes
            else None
        ),
        "definitions": {
            "triggered": "pre-gate selector proposes a non-HiFi candidate",
            "accepted_switch_improvement_precision": (
                "fraction of accepted switches with IoU strictly above local HiFi"
            ),
            "p90_outcome_changing_precision": "recovered / (recovered + harmed)",
        },
    }
    uncertainty = {}
    for method, frame in (
        ("Local HiFi-CS baseline", baseline_frame),
        ("Locked conservative method", locked_frame),
    ):
        uncertainty[method] = {}
        for threshold in THRESHOLDS:
            percent = int(threshold * 100)
            numerator = int(np.count_nonzero(frame["iou"].to_numpy() > threshold))
            uncertainty[method][f"p_at_{percent}"] = {
                "wilson_95": wilson_interval(numerator, len(frame)),
                "frame_clustered_bootstrap_95": clustered_bootstrap_values(
                    frame["iou"].to_numpy(),
                    frame["frame_id"].to_numpy(),
                    threshold=threshold,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed,
                )[:2],
                "scene_clustered_bootstrap_95": clustered_bootstrap_values(
                    frame["iou"].to_numpy(),
                    frame["scene_group"].to_numpy(),
                    threshold=threshold,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed,
                )[:2],
            }
        for metric_name, threshold in (("mean_iou", None),):
            uncertainty[method].setdefault(metric_name, {})
            uncertainty[method][metric_name].update(
                {
                    "frame_clustered_bootstrap_95": clustered_bootstrap_values(
                        frame["iou"].to_numpy(),
                        frame["frame_id"].to_numpy(),
                        threshold=threshold,
                        replicates=args.bootstrap_replicates,
                        seed=args.seed,
                    )[:2],
                    "scene_clustered_bootstrap_95": clustered_bootstrap_values(
                        frame["iou"].to_numpy(),
                        frame["scene_group"].to_numpy(),
                        threshold=threshold,
                        replicates=args.bootstrap_replicates,
                        seed=args.seed,
                    )[:2],
                }
            )
    paired_uncertainty = {}
    paired_metrics = [("mean_iou", None)] + [
        (f"p_at_{int(threshold * 100)}", threshold) for threshold in THRESHOLDS
    ]
    for metric_name, threshold in paired_metrics:
        paired_uncertainty[metric_name] = {}
        for group_column in ("frame_id", "scene_group"):
            paired_uncertainty[metric_name][f"{group_column}_bootstrap_95"] = clustered_bootstrap_values(
                locked_frame["iou"].to_numpy(),
                locked_frame[group_column].to_numpy(),
                threshold=threshold,
                baseline_values=baseline_frame["iou"].to_numpy(),
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            )[:2]
    gap = {}
    for key in ("mean_iou", "p_at_50", "p_at_60", "p_at_70", "p_at_80", "p_at_90"):
        denominator = PAPER[key] - metrics["Local HiFi-CS baseline"][key]
        gap[key] = (
            (metrics["Locked conservative method"][key] - metrics["Local HiFi-CS baseline"][key])
            / denominator
            if denominator != 0.0
            else None
        )
    grouped = []
    locked_only = locked_frame.copy()
    locked_only["target_size_bin"] = pd.cut(
        locked_only["target_area_fraction"],
        bins=[-np.inf, 0.01, 0.05, np.inf],
        labels=["small", "medium", "large"],
    ).astype(str)
    locked_only["baseline_iou_bin"] = pd.cut(
        locked_only["baseline_iou"],
        bins=[-np.inf, 0.25, 0.50, 0.70, 0.80, 0.90, np.inf],
        labels=["<=0.25", "0.25-0.50", "0.50-0.70", "0.70-0.80", "0.80-0.90", ">0.90"],
    ).astype(str)
    locked_only["same_category_proposals_bin"] = pd.cut(
        pd.to_numeric(locked_only["same_category_alternative_count"], errors="coerce"),
        bins=[-np.inf, 0, 1, np.inf],
        labels=["0", "1", "2+"],
    ).astype(str)
    locked_only["clutter_proxy_bin"] = pd.cut(
        locked_only["proposal_bank_size"],
        bins=[-np.inf, 250, 500, np.inf],
        labels=["low_candidate_count", "medium_candidate_count", "high_candidate_count"],
    ).astype(str)
    locked_only["segmentation_error_type"] = np.select(
        [
            locked_only["baseline_area_ratio"] < 0.85,
            locked_only["baseline_area_ratio"] > 1.15,
        ],
        ["under_segmentation", "over_segmentation"],
        default="area_matched",
    )
    locked_only["component_count_bin"] = pd.cut(
        locked_only["connected_component_count"],
        bins=[-np.inf, 1, 2, np.inf],
        labels=["0-1", "2", "3+"],
    ).astype(str)
    locked_only["parser_confidence_bin"] = pd.cut(
        locked_only["parser_confidence"],
        bins=[-np.inf, 0.5, 0.9, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    for column in (
        "query_type",
        "target_category",
        "relation_type",
        "absolute_location_type",
        "target_size_bin",
        "baseline_iou_bin",
        "same_category_proposals_bin",
        "clutter_proxy_bin",
        "segmentation_error_type",
        "component_count_bin",
        "selected_source",
        "selection_stage",
        "switch_action",
        "parser_confidence_bin",
    ):
        for value, frame in locked_only.groupby(column, dropna=False, sort=False):
            grouped.append(
                {"grouping": column, "value": str(value), **summarize_ious(frame["iou"])}
            )
    pd.DataFrame(grouped).to_csv(report / "grouped_results.csv", index=False)
    failures = []
    stage1_best = stage1_oracle_iou
    stage2_best = stage2_oracle_iou
    method_iou = detail.pivot(index="sample_id", columns="method", values="iou")
    for row in locked_frame[locked_frame["iou"] <= 0.90].itertuples():
        stage1_iou = float(stage1_best[row.sample_id])
        stage2_iou = float(stage2_best[row.sample_id])
        evidence = locked_evidence[row.sample_id]
        if stage1_iou <= 0.25 and stage2_iou <= 0.25:
            category = "no_target_proposal_generated"
            failure_stage = "proposal_generation"
        elif stage1_iou <= 0.90 and stage2_iou <= 0.90:
            category = "target_exists_but_no_candidate_reaches_iou_0.90"
            failure_stage = "proposal_generation"
        elif pd.notna(row.relation_type) and row.parser_confidence < 0.90:
            category = "relation_parser_failure"
            failure_stage = "selector"
        elif pd.notna(row.relation_type) and not bool(
            evidence.get("relation_features_valid", False)
        ):
            category = "reference_object_proposal_failure"
            failure_stage = "selector"
        elif pd.notna(row.absolute_location_type):
            category = "absolute_location_rule_failure"
            failure_stage = "selector"
        elif np.isfinite(float(evidence.get("requested_colour_match_fraction", np.nan))) and float(
            evidence.get("requested_colour_match_fraction", np.nan)
        ) < 0.15:
            category = "attribute_or_colour_mismatch"
            failure_stage = "selector"
        elif (
            float(method_iou.loc[row.sample_id, "Learned Stage-1 selector"]) > 0.90
            and float(
                method_iou.loc[
                    row.sample_id, "Stage-1 + Stage-2 selector (pre-gate)"
                ]
            )
            <= 0.90
        ):
            category = "p90_candidate_exists_but_selector_chooses_another"
            failure_stage = "second_refinement_or_final_selector"
        elif stage1_iou > 0.90 or stage2_iou > 0.90:
            category = "p90_candidate_exists_but_selector_chooses_another"
            failure_stage = "selector"
        elif row.false_positive_area_px > 1.5 * row.false_negative_area_px:
            category = "neighbouring_object_leakage"
            failure_stage = "mask_quality"
        elif row.mask_recall < row.mask_precision - 0.10:
            category = "under_segmentation"
            failure_stage = "mask_quality"
        elif row.mask_precision < row.mask_recall - 0.10:
            category = "over_segmentation"
            failure_stage = "mask_quality"
        elif row.target_area_fraction < 0.01:
            category = "small_or_thin_target_boundary_sensitivity"
            failure_stage = "mask_quality"
        elif float(evidence.get("depth_reliability", 1.0)) < 0.50:
            category = "depth_inconsistency"
            failure_stage = "mask_quality"
        else:
            category = "unknown"
            failure_stage = "unknown"
        failures.append(
            {
                "sample_id": row.sample_id,
                "category": category,
                "failure_stage": failure_stage,
                "final_iou": row.iou,
                "stage1_oracle_iou": float(stage1_best[row.sample_id]),
                "stage2_oracle_iou": float(stage2_best[row.sample_id]),
                "selected_source": row.selected_source,
            }
        )
    pd.DataFrame(failures).to_csv(report / "failure_taxonomy.csv", index=False)
    harmful_by_source = (
        locked_only.assign(
            recovered=(locked_only["baseline_iou"] <= 0.90) & (locked_only["iou"] > 0.90),
            harmed=(locked_only["baseline_iou"] > 0.90) & (locked_only["iou"] <= 0.90),
        )
        .groupby("selected_source", dropna=False)
        .agg(
            selected_samples=("sample_id", "count"),
            recovered_p90=("recovered", "sum"),
            harmed_p90=("harmed", "sum"),
            mean_iou=("iou", "mean"),
        )
        .reset_index()
    )
    harmful_by_source["net_p90"] = (
        harmful_by_source["recovered_p90"] - harmful_by_source["harmed_p90"]
    )
    harmful_by_source.to_csv(report / "selected_source_transitions.csv", index=False)
    proposal_runtime = []
    stage2_runtime = []
    final_runtime = []
    for row in compact:
        proposal_runtime.append(
            json.loads((experiment / "proposals" / row.sample_id / "runtime.json").read_text())
        )
        stage2_runtime.append(
            json.loads((experiment / "stage2" / row.sample_id / "runtime.json").read_text())
        )
        final_runtime.append(
            json.loads(
                (experiment / "locked_benchmark_masks" / row.sample_id / "timing.json").read_text()
            )
        )
    proposal_manifest_path = experiment / "proposal_generation_test_manifest.parquet"
    proposal_manifest_seconds = (
        float(pd.read_parquet(proposal_manifest_path)["runtime_seconds"].sum())
        if proposal_manifest_path.is_file()
        else float(
            sum(value.get("sample_seconds_before_write", 0.0) for value in proposal_runtime)
        )
    )
    shared_text_seconds = _unique_shared_runtime_seconds(
        proposal_runtime, compact, "text"
    )
    shared_automatic_seconds = _unique_shared_runtime_seconds(
        proposal_runtime, compact, "automatic"
    )
    model_load_seconds = float(
        max(
            (value.get("model_load_seconds_run_level", 0.0) for value in proposal_runtime),
            default=0.0,
        )
    )
    efficiency = {
        "stage1_sample_seconds_before_write_sum": float(
            sum(value.get("sample_seconds_before_write", 0.0) for value in proposal_runtime)
        ),
        "stage1_per_sample_end_to_end_seconds_sum": proposal_manifest_seconds,
        "stage1_shared_text_inference_seconds_sum": shared_text_seconds,
        "stage1_shared_automatic_inference_seconds_sum": shared_automatic_seconds,
        "stage1_model_load_seconds": model_load_seconds,
        "stage1_total_measured_seconds": float(
            proposal_manifest_seconds
            + shared_text_seconds
            + shared_automatic_seconds
            + model_load_seconds
        ),
        "stage2_seconds_before_write_sum": float(
            sum(value.get("runtime_seconds_before_write", 0.0) for value in stage2_runtime)
        ),
        "final_selection_seconds_sum": float(
            sum(value.get("final_selection_seconds", 0.0) for value in final_runtime)
        ),
        "peak_rss_bytes": int(
            max((value.get("peak_rss_bytes_so_far", 0) for value in proposal_runtime), default=0)
        ),
        "experiment_disk_bytes": int(
            sum(path.stat().st_size for path in experiment.rglob("*") if path.is_file())
        ),
    }
    payload = {
        "status": "COMPLETED_AFTER_OUTPUT_LOCK",
        "evaluation_started_after_output_lock": True,
        "posthoc_benchmark": True,
        "metrics": metrics,
        "threshold_transitions": transitions,
        "gate_statistics": gate_statistics,
        "uncertainty": uncertainty,
        "paired_uncertainty": paired_uncertainty,
        "fraction_of_paper_gap_closed": gap,
        "runtime_seconds": time.perf_counter() - started,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "efficiency": efficiency,
    }
    (report / "formal_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
