#!/usr/bin/env python3
"""Run validation-only proposal, feature, reasoning, and selector ablations.

All learned rows are exact-frame GroupKFold out-of-fold estimates.  Proposal
oracle rows use ground truth only as explicitly marked diagnostic upper bounds.
The frozen proposal bank is never regenerated or changed by this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.proposal_evaluation import simple_baseline_decisions  # noqa: E402
from src.segmentation.selector_dataset import (  # noqa: E402
    LABEL_COLUMNS,
    CandidateDatasetPair,
    infer_dataset_split,
    iter_joined_candidate_samples,
)
from src.segmentation.selector_out_of_core import (  # noqa: E402
    grouped_oof_predictions_out_of_core,
)
from src.segmentation.selective_sam3_vg.metrics import summarize_ious  # noqa: E402


EXPERIMENT = PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1"
ARTIFACTS = PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1"
IDENTITY_AND_LABEL_COLUMNS = {
    "sample_id",
    "candidate_id",
    "split",
    "scene_id",
    "frame_id",
    "rgb_sha256",
    "query",
    "source_family",
    "source_variant",
    "eligible_final",
    "candidate_iou",
    "continuous_iou",
    "y70",
    "y80",
    "y90",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-features",
        type=Path,
        default=EXPERIMENT / "features/candidate_features_validation.parquet",
    )
    parser.add_argument(
        "--stage1-labels",
        type=Path,
        default=EXPERIMENT / "oracle_stage1/candidate_iou.parquet",
    )
    parser.add_argument(
        "--stage2-features",
        type=Path,
        default=EXPERIMENT / "features_stage2/candidate_features_validation.parquet",
    )
    parser.add_argument(
        "--stage2-labels",
        type=Path,
        default=EXPERIMENT / "oracle_stage2/candidate_iou.parquet",
    )
    parser.add_argument(
        "--selector-config",
        type=Path,
        default=PROJECT_ROOT / "configs/sam3_proposal_bank_p90_v1/selector_training.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT / "report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int)
    return parser.parse_args()


def _metric_fields(values: pd.Series | np.ndarray) -> dict[str, Any]:
    metrics = summarize_ious(np.asarray(values, dtype=np.float64))
    return {
        key: metrics[key]
        for key in (
            "mean_iou",
            "median_iou",
            "p_at_50",
            "p_at_60",
            "p_at_70",
            "p_at_80",
            "p_at_90",
            "p_at_50_numerator",
            "p_at_60_numerator",
            "p_at_70_numerator",
            "p_at_80_numerator",
            "p_at_90_numerator",
            "p_at_90_denominator",
        )
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _learned_row(
    pair: CandidateDatasetPair,
    *,
    family: str,
    variant: str,
    folds: int,
    seed: int,
    decision_root: Path,
    scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    output_path = decision_root / (
        f".{_safe_name(family)}__{_safe_name(variant)}__full_oof.parquet"
    )
    decisions, artifacts = grouped_oof_predictions_out_of_core(
        [pair],
        output_path=output_path,
        folds=folds,
        seed=seed,
        maximum_training_candidates_per_sample=64,
    )
    method = "M3_two_head_ensemble"
    selected = decisions[decisions["method"] == method].copy()
    selected.to_parquet(
        decision_root / f"{_safe_name(family)}__{_safe_name(variant)}.parquet",
        index=False,
    )
    row = {
        "ablation_family": family,
        "variant": variant,
        "estimate_type": "GROUPED_OOF_GT_FREE_SELECTOR",
        "scope": scope,
        "selector": method,
        "candidate_rows": int(
            artifacts["training_subset_audit"]["eligible_candidates_by_split"][
                pair.split
            ]
        ),
        "samples": int(selected["sample_id"].nunique()),
        "groups": int(selected["frame_id"].nunique()),
        "seconds": float(time.perf_counter() - started),
        **_metric_fields(selected["candidate_iou"]),
    }
    audit = {
        "ablation_family": family,
        "variant": variant,
        "folds": artifacts["fold_audit"],
        "feature_names": artifacts["encoder"].feature_names,
        "feature_count": len(artifacts["encoder"].feature_names),
        "training_subset_audit": artifacts["training_subset_audit"],
    }
    output_path.unlink(missing_ok=True)
    return row, audit


def _stream_oracle_rows(
    pair: CandidateDatasetPair,
    *,
    family: str,
    variants: tuple[str, ...],
    scope: str,
) -> tuple[list[dict[str, Any]], int]:
    values = {variant: [] for variant in variants}
    candidate_rows = {variant: 0 for variant in variants}
    groups: set[str] = set()
    samples = 0
    all_eligible_rows = 0
    for sample in iter_joined_candidate_samples([pair]):
        masks = _proposal_masks(sample)
        groups.add(str(sample.iloc[0]["frame_id"]))
        samples += 1
        all_eligible_rows += int(sample["eligible_final"].sum())
        for variant in variants:
            keep = np.asarray(masks.get(variant, np.ones(len(sample))), dtype=bool)
            subset = sample[keep & sample["eligible_final"].to_numpy(bool)]
            if subset.empty:
                raise ValueError(f"oracle ablation removed fallback: {variant}")
            selected = subset.sort_values(
                ["candidate_iou", "candidate_id"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]
            values[variant].append(float(selected["candidate_iou"]))
            candidate_rows[variant] += len(subset)
    rows = [
        {
            "ablation_family": family,
            "variant": variant,
            "estimate_type": "GT_ORACLE_NON_DEPLOYABLE",
            "scope": scope,
            "candidate_rows": candidate_rows[variant],
            "samples": samples,
            "groups": len(groups),
            **_metric_fields(np.asarray(values[variant], dtype=np.float64)),
        }
        for variant in variants
    ]
    return rows, all_eligible_rows


def _feature_columns(data: pd.DataFrame, predicates: list[Callable[[str], bool]]) -> list[str]:
    return [
        column
        for column in data.columns
        if column in IDENTITY_AND_LABEL_COLUMNS or any(predicate(column) for predicate in predicates)
    ]


def _prefix(*values: str) -> Callable[[str], bool]:
    return lambda column: column.startswith(values)


def _exact(*values: str) -> Callable[[str], bool]:
    allowed = set(values)
    return lambda column: column in allowed


def _feature_variants(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sam = [
        _prefix("sam_", "presence_", "mask_quality_", "mask_threshold", "instance_threshold", "source_rank"),
        _exact("source_family", "source_variant", "eligible_final"),
    ]
    overlap = sam + [
        _prefix("hifi_candidate_", "hifi_coarse_", "candidate_hifi_area_ratio", "boundary_displacement"),
    ]
    probability = overlap + [
        _prefix("hifi_probability_", "high_confidence_core_coverage", "low_probability_expansion"),
    ]
    clip = probability + [_prefix("clip_")]
    attributes = clip + [
        _prefix("requested_colour_", "hsv_", "lab_", "inside_outside_rgb_contrast"),
        _exact("has_colour"),
    ]
    absolute = attributes + [
        _prefix("centroid_", "left_to_right_", "right_to_left_", "near_to_far_", "far_to_near_", "leftmost_", "rightmost_", "closest_", "furthest_", "scene_centre_"),
        _exact("has_absolute_location", "absolute_location_type"),
    ]
    relation = absolute + [
        _prefix("relation_"),
        _exact("has_relation", "relation_type"),
    ]
    depth = relation + [
        _prefix("depth_", "valid_depth_", "median_depth", "robust_depth_", "foreground_background_depth_", "major_depth_", "point_cloud_", "reference_centroid_3d_"),
    ]
    candidate_set = depth + [
        _prefix("candidate_area_rank", "candidate_sam_score_rank", "candidate_hifi_overlap_rank", "candidate_depth_consistency_rank", "candidate_spatial_relation_rank", "same_category_alternative_count", "source_consensus_count"),
    ]
    all_columns = list(data.columns)
    definitions: list[tuple[str, list[Callable[[str], bool]] | None]] = [
        ("SAM scores only", sam),
        ("HiFi overlap only", overlap),
        ("+ HiFi probability features", probability),
        ("+ CLIP semantics", clip),
        ("+ attributes", attributes),
        ("+ absolute position", absolute),
        ("+ pairwise relations", relation),
        ("+ depth", depth),
        ("+ candidate-set relations", candidate_set),
        ("all features", None),
    ]
    result = {}
    for name, predicates in definitions:
        columns = all_columns if predicates is None else _feature_columns(data, predicates)
        result[name] = data[columns].copy()
    return result


def _reasoning_variants(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parse_columns = {
        column
        for column in data.columns
        if column in {
            "query_type",
            "parser_confidence",
            "has_relation",
            "has_absolute_location",
            "has_colour",
            "target_category",
            "absolute_location_type",
            "relation_type",
        }
        or column.startswith(
            (
                "requested_colour_",
                "left_to_right_",
                "right_to_left_",
                "near_to_far_",
                "far_to_near_",
                "leftmost_",
                "rightmost_",
                "closest_",
                "furthest_",
                "relation_",
            )
        )
        or column in {"clip_target_category_similarity", "clip_target_attribute_similarity"}
    }
    base = [column for column in data.columns if column not in parse_columns]
    category = {"target_category", "clip_target_category_similarity"}
    attributes = category | {
        column
        for column in data.columns
        if column == "has_colour"
        or column.startswith(("requested_colour_", "hsv_", "lab_"))
        or column == "clip_target_attribute_similarity"
    }
    absolute = category | {
        column
        for column in data.columns
        if column in {"has_absolute_location", "absolute_location_type"}
        or column.startswith(
            (
                "left_to_right_",
                "right_to_left_",
                "near_to_far_",
                "far_to_near_",
                "leftmost_",
                "rightmost_",
                "closest_",
                "furthest_",
            )
        )
    }
    pairwise = category | {
        column
        for column in data.columns
        if column in {"has_relation", "relation_type"} or column.startswith("relation_")
    }
    variants = {
        "no query parse": set(),
        "category only": category,
        "category + attributes": attributes,
        "category + absolute location": absolute,
        "category + pairwise relation": pairwise,
        "complete parser": parse_columns,
    }
    available = set(data.columns)
    return {
        name: data[[*base, *sorted((columns & available) - set(base))]].copy()
        for name, columns in variants.items()
    }


def _proposal_masks(data: pd.DataFrame) -> dict[str, np.ndarray]:
    source = data["source_family"].astype(str)
    threshold = pd.to_numeric(data.get("mask_threshold"), errors="coerce")
    return {
        "all Stage-1 proposals": np.ones(len(data), dtype=bool),
        "no text proposals": ~source.str.startswith("TEXT_").to_numpy(),
        "no automatic proposals": (source != "AUTOMATIC").to_numpy(),
        "no HiFi box/point proposals": (source != "VISUAL_HIFI").to_numpy(),
        "no component proposals": (source != "COMPONENT").to_numpy(),
        "no threshold variants": (
            (source != "HIFI_THRESHOLD")
            & (threshold.isna() | np.isclose(threshold.fillna(0.5), 0.50))
        ).to_numpy(),
    }


def _source_variant_pairs(
    pair: CandidateDatasetPair,
) -> dict[str, CandidateDatasetPair]:
    return {
        "all Stage-1 proposals": pair,
        "no text proposals": replace(pair, excluded_source_prefixes=("TEXT_",)),
        "no automatic proposals": replace(
            pair, excluded_source_families=("AUTOMATIC",)
        ),
        "no HiFi box/point proposals": replace(
            pair, excluded_source_families=("VISUAL_HIFI",)
        ),
        "no component proposals": replace(
            pair, excluded_source_families=("COMPONENT",)
        ),
        "no threshold variants": replace(
            pair, remove_noncanonical_threshold_variants=True
        ),
    }


def _feature_variant_pairs(
    pair: CandidateDatasetPair,
    builder: Callable[[pd.DataFrame], dict[str, pd.DataFrame]],
) -> dict[str, CandidateDatasetPair]:
    sample = next(iter(iter_joined_candidate_samples([pair])))
    label_columns = set(LABEL_COLUMNS)
    return {
        name: replace(
            pair,
            feature_columns=tuple(
                column for column in frame.columns if column not in label_columns
            ),
        )
        for name, frame in builder(sample).items()
    }


def _selector_rows(
    stage1_pair: CandidateDatasetPair,
    stage2_pair: CandidateDatasetPair,
    *,
    stage1_candidate_rows: int,
    stage2_candidate_rows: int,
) -> list[dict[str, Any]]:
    baseline_parts = []
    for sample in iter_joined_candidate_samples([stage1_pair]):
        decisions, _ = simple_baseline_decisions(sample)
        baseline_parts.append(decisions)
    baseline_decisions = pd.concat(baseline_parts, ignore_index=True)
    stage1_decisions = pd.read_parquet(ARTIFACTS / "stage1_selector/oof_selected_candidates.parquet")
    stage2_decisions = pd.read_parquet(ARTIFACTS / "final_selector/oof_gate_decisions.parquet")
    final_metrics = json.loads(
        (ARTIFACTS / "final_selector/validation_metrics.json").read_text(encoding="utf-8")
    )
    selected_gate = str(final_metrics["selected_method"])
    mappings = {
        "highest score": (baseline_decisions, "B5_bank_highest_sam"),
        "overlap maximum": (baseline_decisions, "B6_max_hifi_overlap"),
        "rule-based": (baseline_decisions, "B9_relation_aware_rules"),
        "classifier": (stage1_decisions, "M1_hgb_classifier"),
        "classifier + regressor": (stage1_decisions, "M3_two_head_ensemble"),
        "classifier + regressor + conservative gate": (stage2_decisions, selected_gate),
    }
    rows = []
    for variant, (frame, method) in mappings.items():
        chosen = frame[frame["method"] == method]
        if chosen.empty:
            if variant.endswith("conservative gate") and selected_gate.startswith("HIFI_FALLBACK"):
                fallback_parts = []
                for sample in iter_joined_candidate_samples([stage2_pair]):
                    fallback = sample[
                        sample["source_family"].isin(
                            {"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"}
                        )
                    ]
                    if len(fallback) != 1:
                        raise ValueError("Stage-2 selector ablation requires one fallback")
                    fallback_parts.append(fallback)
                chosen = pd.concat(fallback_parts, ignore_index=True)
            else:
                raise ValueError(f"missing selector ablation decisions for {method}")
        rows.append(
            {
                "ablation_family": "selector",
                "variant": variant,
                "estimate_type": "GROUPED_OOF_GT_FREE_SELECTOR",
                "scope": "validation grouped OOF; frozen candidate bank",
                "selector": method,
                "candidate_rows": int(
                    stage2_candidate_rows if "gate" in variant else stage1_candidate_rows
                ),
                "samples": int(chosen["sample_id"].nunique()),
                "groups": int(chosen["frame_id"].nunique()),
                **_metric_fields(chosen["candidate_iou"]),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.selector_config.read_text(encoding="utf-8"))
    folds = int(args.folds or config["folds"])
    seed = int(args.seed)
    stage1_pair = CandidateDatasetPair(
        args.stage1_features.expanduser().resolve(),
        args.stage1_labels.expanduser().resolve(),
        infer_dataset_split(args.stage1_features.expanduser().resolve()),
    )
    stage2_pair = CandidateDatasetPair(
        args.stage2_features.expanduser().resolve(),
        args.stage2_labels.expanduser().resolve(),
        infer_dataset_split(args.stage2_features.expanduser().resolve()),
    )
    if stage1_pair.split != "val" or stage2_pair.split != "val":
        raise ValueError("formal ablations are validation-only")
    output_root = args.output_root.expanduser().resolve()
    decision_root = output_root / "ablations/grouped_oof_decisions"
    decision_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    source_variants = tuple(_source_variant_pairs(stage1_pair))
    oracle_rows, stage1_candidate_rows = _stream_oracle_rows(
        stage1_pair,
        family="proposal source",
        variants=source_variants,
        scope="Stage-1 canonical-source oracle; GT diagnostic only",
    )
    rows.extend(oracle_rows)
    for variant, variant_pair in _source_variant_pairs(stage1_pair).items():
        learned, audit = _learned_row(
            variant_pair,
            family="proposal source",
            variant=variant,
            folds=folds,
            seed=seed,
            decision_root=decision_root,
            scope="Stage-1 grouped OOF selector; frozen surviving-candidate features",
        )
        rows.append(learned)
        audits.append(audit)

    stage1_no_refinement, _ = _stream_oracle_rows(
        stage1_pair,
        family="proposal source",
        variants=("no Stage-2 refinement",),
        scope="Stage-1 oracle; GT diagnostic only",
    )
    rows.extend(stage1_no_refinement)
    stage2_refinement, stage2_candidate_rows = _stream_oracle_rows(
        stage2_pair,
        family="proposal source",
        variants=("with Stage-2 refinement",),
        scope="Stage-2 oracle; GT diagnostic only",
    )
    rows.extend(stage2_refinement)

    for variant, variant_pair in _feature_variant_pairs(
        stage1_pair, _feature_variants
    ).items():
        learned, audit = _learned_row(
            variant_pair,
            family="feature",
            variant=variant,
            folds=folds,
            seed=seed,
            decision_root=decision_root,
            scope="Stage-1 grouped OOF; cumulative inference-feature families",
        )
        rows.append(learned)
        audits.append(audit)

    for variant, variant_pair in _feature_variant_pairs(
        stage1_pair, _reasoning_variants
    ).items():
        learned, audit = _learned_row(
            variant_pair,
            family="query reasoning",
            variant=variant,
            folds=folds,
            seed=seed,
            decision_root=decision_root,
            scope="Stage-1 grouped OOF feature ablation on one frozen proposal bank",
        )
        rows.append(learned)
        audits.append(audit)

    rows.extend(
        _selector_rows(
            stage1_pair,
            stage2_pair,
            stage1_candidate_rows=stage1_candidate_rows,
            stage2_candidate_rows=stage2_candidate_rows,
        )
    )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "ablation_results.csv", index=False)
    result[result["ablation_family"] == "proposal source"].to_csv(
        output_root / "proposal_source_ablations.csv", index=False
    )
    result[result["ablation_family"] == "feature"].to_csv(
        output_root / "feature_ablations.csv", index=False
    )
    result[result["ablation_family"] == "query reasoning"].to_csv(
        output_root / "query_reasoning_ablations.csv", index=False
    )
    result[result["ablation_family"] == "selector"].to_csv(
        output_root / "selector_ablations.csv", index=False
    )
    audit_payload = {
        "split": "validation",
        "group_unit": "exact RGB frame",
        "folds": folds,
        "seed": seed,
        "strict_threshold": "candidate_iou > threshold",
        "test_rows_used": 0,
        "proposal_source_scope_note": (
            "Source removal uses each deduplicated candidate's canonical source family; "
            "oracle and selector effects are reported separately."
        ),
        "query_reasoning_scope_note": (
            "Reasoning ablations remove selector features but keep the frozen proposal bank; "
            "they do not claim prompt-generation ablation."
        ),
        "runs": audits,
    }
    (output_root / "ablation_audit.json").write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rows": len(result),
                "output": str(output_root / "ablation_results.csv"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
