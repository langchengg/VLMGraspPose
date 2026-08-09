#!/usr/bin/env python3
"""Merge interleaved Stage-1 oracle shards without loading all labels in memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.proposal_oracle import OracleAccumulator  # noqa: E402
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_oracle_outputs(
    *,
    accumulator: OracleAccumulator,
    output_root: Path,
    proposal_root: Path,
) -> dict[str, object]:
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
        f'<li><a href="{proposal_root / row.sample_id / "proposal_grid.png"}">'
        f"{row.sample_id}</a> oracle={row.best_candidate_iou:.4f}</li>"
        for row in failures.itertuples()
    )
    (output_root / "no_p90_candidate_cases.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<h1>No strict P@90 candidate</h1><ul>{links}</ul>",
        encoding="utf-8",
    )
    return summary


def merge_oracle_shards(
    *,
    shard_roots: Sequence[Path],
    expected_sample_ids: Sequence[str],
    output_root: Path,
    proposal_root: Path,
) -> dict[str, object]:
    """Interleave modulo shards and publish the canonical oracle artifacts."""
    roots = [Path(root).expanduser().resolve() for root in shard_roots]
    if len(roots) < 2:
        raise ValueError("oracle merge requires at least two shard roots")
    expected = [str(sample_id) for sample_id in expected_sample_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected oracle sample IDs are not unique")
    files = [root / "candidate_iou.parquet" for root in roots]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing oracle shard candidate tables: {missing}")
    readers = [pq.ParquetFile(path) for path in files]
    expected_row_groups = [
        len(expected[index :: len(roots)]) for index in range(len(roots))
    ]
    observed_row_groups = [reader.num_row_groups for reader in readers]
    if observed_row_groups != expected_row_groups:
        raise ValueError(
            "oracle shard row-group counts do not match modulo partition: "
            f"observed={observed_row_groups}, expected={expected_row_groups}"
        )

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_path = output_root / "candidate_iou.parquet"
    temporary_path = candidate_path.with_name(f".{candidate_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    accumulator = OracleAccumulator()
    candidate_rows = 0
    try:
        for global_index, expected_sample_id in enumerate(expected):
            shard_index = global_index % len(readers)
            row_group_index = global_index // len(readers)
            table = readers[shard_index].read_row_group(row_group_index)
            sample_ids = set(map(str, table["sample_id"].to_pylist()))
            if sample_ids != {expected_sample_id}:
                raise ValueError(
                    "oracle shard order/identity mismatch at global index "
                    f"{global_index}: observed={sorted(sample_ids)}, "
                    f"expected={expected_sample_id}"
                )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            accumulator.add(table.to_pandas())
            candidate_rows += table.num_rows
            if (global_index + 1) % 1000 == 0:
                print(
                    f"oracle merge: {global_index + 1}/{len(expected)}",
                    flush=True,
                )
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    if writer is None:
        raise RuntimeError("oracle shard merge did not process any samples")
    writer.close()
    temporary_path.replace(candidate_path)
    summary = _write_oracle_outputs(
        accumulator=accumulator,
        output_root=output_root,
        proposal_root=proposal_root,
    )
    receipt = {
        "status": "COMPLETE",
        "sample_count": len(expected),
        "candidate_row_count": candidate_rows,
        "shard_count": len(roots),
        "shard_candidate_iou_sha256": {
            str(path): _sha256_file(path) for path in files
        },
        "candidate_iou_sha256": _sha256_file(candidate_path),
    }
    (output_root / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "receipt": receipt}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--proposal-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1/proposals",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = {"train": 26295, "val": 3778, "test": 7675}
    compact = load_compact_manifest(
        PROJECT_ROOT
        / "runs/modular_reranking_repeatedfilm_v1_20260729_203147"
        / f"compact_inputs/{args.split}/manifest.jsonl",
        expected_split=args.split,
        expected_count=counts[args.split],
    )
    result = merge_oracle_shards(
        shard_roots=args.shard_root,
        expected_sample_ids=[row.sample_id for row in compact],
        output_root=args.output_root,
        proposal_root=args.proposal_root.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
