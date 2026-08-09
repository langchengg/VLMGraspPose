#!/usr/bin/env python3
"""Compute the GT-only Stage-2 refinement oracle on development data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.proposal_oracle import candidate_ious  # noqa: E402
from src.segmentation.selective_sam3_vg.evaluation import (  # noqa: E402
    load_frozen_ground_truth_manifest,
    load_ground_truth_mask,
)
from src.segmentation.selective_sam3_vg.metrics import summarize_ious  # noqa: E402
from src.segmentation.selective_sam3_vg.io import load_compact_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = args.experiment_root.expanduser().resolve()
    counts = {"train": 26295, "val": 3778, "test": 7675}
    if args.split == "test":
        marker = experiment / "locked_benchmark_masks/LOCKED_BEFORE_GT_EVALUATION"
        if not marker.is_file():
            raise RuntimeError("test oracle is forbidden before output lock")
    gt = load_frozen_ground_truth_manifest(
        PROJECT_ROOT / f"artifacts/data_audit/frozen_manifests/ocidvlg_unique_{args.split}.json",
        hifics_root=PROJECT_ROOT / "hifics",
        expected_count=counts[args.split],
    )
    gt_by_prefix = {f"q{int(row['question_index']):07d}_": row for row in gt}
    compact = load_compact_manifest(
        PROJECT_ROOT
        / f"runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/{args.split}/manifest.jsonl",
        expected_split=args.split,
        expected_count=counts[args.split],
    )
    directories = [experiment / "stage2" / row.sample_id for row in compact]
    missing = [
        directory.name
        for directory in directories
        if not (directory / "terminal_status.json").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Stage-2 oracle requires all {counts[args.split]} {args.split} "
            f"bundles; missing {len(missing)}"
        )
    output_names = {
        "train": "oracle_stage2_train",
        "val": "oracle_stage2",
        "test": "oracle_stage2_test",
    }
    output = experiment / output_names[args.split]
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_iou.parquet"
    temporary_candidate_path = candidate_path.with_name(
        f".{candidate_path.name}.tmp"
    )
    temporary_candidate_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    stage1_roots = {
        "train": "oracle_stage1_train",
        "val": "oracle_stage1",
        "test": "oracle_stage1_test",
    }
    stage1_root = stage1_roots[args.split]
    stage1 = pd.read_parquet(
        experiment / stage1_root / "per_sample_oracle.parquet"
    ).set_index("sample_id")
    rows = []
    for number, directory in enumerate(directories, start=1):
        prefix = directory.name.split("_", 1)[0] + "_"
        labels = candidate_ious(
            directory, load_ground_truth_mask(gt_by_prefix[prefix])
        )
        table = pa.Table.from_pandas(labels, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_candidate_path,
                table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        writer.write_table(table)
        sample_id = directory.name
        best = labels.loc[labels["candidate_iou"].idxmax()]
        selected = labels[labels["source_family"] == "STAGE2_SELECTED_STAGE1"]
        selected_iou = float(selected["candidate_iou"].max()) if len(selected) else 0.0
        refinements = labels[
            ~labels["source_family"].isin(
                {"STAGE2_HIFI_FALLBACK", "STAGE2_SELECTED_STAGE1"}
            )
        ]
        best_refinement = (
            float(refinements["candidate_iou"].max()) if len(refinements) else 0.0
        )
        stage1_oracle = float(stage1.loc[sample_id, "best_candidate_iou"])
        rows.append(
            {
                "sample_id": sample_id,
                "stage1_oracle_iou": stage1_oracle,
                "selected_stage1_iou": selected_iou,
                "best_refinement_iou": best_refinement,
                "stage2_oracle_iou": float(best["candidate_iou"]),
                "stage2_best_candidate_id": str(best["candidate_id"]),
                "stage2_best_source_family": str(best["source_family"]),
                "newly_crosses_p90_after_stage2": bool(
                    stage1_oracle <= 0.90 and float(best["candidate_iou"]) > 0.90
                ),
                "all_refinements_harm_selected_stage1": bool(
                    best_refinement < selected_iou
                ),
            }
        )
        if number % 25 == 0:
            print(f"Stage-2 oracle: {number}/{len(directories)}", flush=True)
    if writer is None:
        raise RuntimeError("Stage-2 oracle did not process any samples")
    writer.close()
    temporary_candidate_path.replace(candidate_path)
    per_sample = pd.DataFrame(rows)
    stage1_metrics = summarize_ious(per_sample["stage1_oracle_iou"])
    stage2_metrics = summarize_ious(per_sample["stage2_oracle_iou"])
    source = (
        per_sample.groupby("stage2_best_source_family")
        .agg(samples=("sample_id", "count"), mean_iou=("stage2_oracle_iou", "mean"))
        .reset_index()
    )
    summary = {
        "samples": len(per_sample),
        "stage1_oracle": stage1_metrics,
        "stage2_oracle": stage2_metrics,
        "new_p90_successes_after_stage2": int(per_sample["newly_crosses_p90_after_stage2"].sum()),
        "all_refinements_harm_stage1_selection_count": int(
            per_sample["all_refinements_harm_selected_stage1"].sum()
        ),
        "non_deployable_gt_oracle": True,
    }
    per_sample.to_parquet(output / "per_sample_oracle.parquet", index=False)
    source.to_csv(output / "refinement_source_contributions.csv", index=False)
    per_sample[
        ["sample_id", "stage1_oracle_iou", "stage2_oracle_iou"]
    ].to_csv(output / "stage1_vs_stage2_oracle.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
