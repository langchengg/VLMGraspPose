#!/usr/bin/env python3
"""Compute a GT-only Stage-1 proposal oracle after proposal generation."""

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

from src.segmentation.proposal_oracle import (  # noqa: E402
    OracleAccumulator,
    candidate_ious,
)
from src.segmentation.selective_sam3_vg.evaluation import (  # noqa: E402
    load_frozen_ground_truth_manifest,
    load_ground_truth_mask,
)
from src.segmentation.selective_sam3_vg.io import load_compact_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument(
        "--proposal-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/proposals",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen = PROJECT_ROOT / f"artifacts/data_audit/frozen_manifests/ocidvlg_unique_{args.split}.json"
    counts = {"train": 26295, "val": 3778, "test": 7675}
    if args.split == "test":
        marker = (
            PROJECT_ROOT
            / "outputs/sam3_proposal_bank_p90_v1/locked_benchmark_masks/LOCKED_BEFORE_GT_EVALUATION"
        )
        if not marker.is_file():
            raise RuntimeError("test oracle is forbidden before output lock")
    ground_truth = load_frozen_ground_truth_manifest(
        frozen, hifics_root=PROJECT_ROOT / "hifics", expected_count=counts[args.split]
    )
    gt_by_id = {
        f"q{int(row['question_index']):07d}_": row for row in ground_truth
    }
    if args.sample_manifest:
        manifest = pd.read_parquet(args.sample_manifest.expanduser().resolve())
        sample_ids = list(manifest["sample_id"].astype(str))
    else:
        compact = load_compact_manifest(
            PROJECT_ROOT
            / f"runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/{args.split}/manifest.jsonl",
            expected_split=args.split,
            expected_count=counts[args.split],
        )
        sample_ids = [row.sample_id for row in compact]
        missing = [
            sample_id
            for sample_id in sample_ids
            if not (args.proposal_root / sample_id / "terminal_status.json").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Stage-1 oracle requires all {counts[args.split]} {args.split} "
                f"proposal bundles; missing {len(missing)}"
            )
    output_names = {
        "train": "oracle_stage1_train",
        "val": "oracle_stage1",
        "test": "oracle_stage1_test",
    }
    output_root = (
        args.output_root
        if args.output_root is not None
        else PROJECT_ROOT
        / "outputs/sam3_proposal_bank_p90_v1"
        / output_names[args.split]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_path = output_root / "candidate_iou.parquet"
    temporary_candidate_path = candidate_path.with_name(
        f".{candidate_path.name}.tmp"
    )
    temporary_candidate_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    accumulator = OracleAccumulator()
    for index, sample_id in enumerate(sample_ids, start=1):
        prefix = sample_id.split("_", 1)[0] + "_"
        gt_row = gt_by_id.get(prefix)
        if gt_row is None:
            raise ValueError(f"no {args.split} GT identity for {sample_id}")
        labels = candidate_ious(
            args.proposal_root / sample_id, load_ground_truth_mask(gt_row)
        )
        accumulator.add(labels)
        table = pa.Table.from_pandas(labels, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                temporary_candidate_path,
                table.schema,
                compression="snappy",
                use_dictionary=True,
            )
        writer.write_table(table)
        if index % 25 == 0:
            print(f"oracle: {index}/{len(sample_ids)}", flush=True)
    if writer is None:
        raise RuntimeError("oracle did not process any samples")
    writer.close()
    temporary_candidate_path.replace(candidate_path)
    summary, per_sample, contributions = accumulator.finalize()
    per_sample.to_parquet(output_root / "per_sample_oracle.parquet", index=False)
    contributions.to_csv(output_root / "source_contributions.csv", index=False)
    contributions.to_csv(output_root / "oracle_growth_curve.csv", index=False)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    failures = per_sample[~per_sample["has_p90_candidate"]]
    links = "\n".join(
        f'<li><a href="../proposals/{row.sample_id}/proposal_grid.png">{row.sample_id}</a> '
        f'oracle={row.best_candidate_iou:.4f}</li>'
        for row in failures.itertuples()
    )
    (output_root / "no_p90_candidate_cases.html").write_text(
        f"<!doctype html><meta charset='utf-8'><h1>No strict P@90 candidate</h1><ul>{links}</ul>",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
