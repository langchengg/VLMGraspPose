from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.merge_proposal_oracle_shards import merge_oracle_shards
from src.segmentation.proposal_oracle import summarize_oracle


def _labels(sample_id: str, best_iou: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "candidate_id": f"{sample_id}_hifi",
                "source_family": "HIFI_ORIGINAL",
                "source_variant": "canonical",
                "eligible_final": True,
                "candidate_iou": 0.60,
                "y70": False,
                "y80": False,
                "y90": False,
                "continuous_iou": 0.60,
            },
            {
                "sample_id": sample_id,
                "candidate_id": f"{sample_id}_text",
                "source_family": "TEXT_FULL_QUERY",
                "source_variant": "full",
                "eligible_final": True,
                "candidate_iou": best_iou,
                "y70": best_iou > 0.70,
                "y80": best_iou > 0.80,
                "y90": best_iou > 0.90,
                "continuous_iou": best_iou,
            },
        ]
    )


def _write_shard(path: Path, frames: list[pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True)
    writer = None
    try:
        for frame in frames:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def test_merge_oracle_shards_restores_original_order_and_summary(
    tmp_path: Path,
) -> None:
    sample_ids = [f"q{index:03d}" for index in range(5)]
    frames = [_labels(sample_id, 0.75 + index * 0.05) for index, sample_id in enumerate(sample_ids)]
    shard_roots = [tmp_path / "shard0", tmp_path / "shard1"]
    _write_shard(
        shard_roots[0] / "candidate_iou.parquet",
        frames[0::2],
    )
    _write_shard(
        shard_roots[1] / "candidate_iou.parquet",
        frames[1::2],
    )
    output = tmp_path / "merged"

    result = merge_oracle_shards(
        shard_roots=shard_roots,
        expected_sample_ids=sample_ids,
        output_root=output,
        proposal_root=tmp_path / "proposals",
    )

    merged = pq.read_table(output / "candidate_iou.parquet").to_pandas()
    expected = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(merged, expected)
    expected_summary, expected_per_sample, _ = summarize_oracle(expected)
    assert result["summary"] == expected_summary
    actual_per_sample = pd.read_parquet(output / "per_sample_oracle.parquet")
    pd.testing.assert_frame_equal(actual_per_sample, expected_per_sample)
    assert result["receipt"]["sample_count"] == len(sample_ids)
    assert result["receipt"]["candidate_row_count"] == len(expected)


def test_merge_oracle_shards_rejects_identity_drift(tmp_path: Path) -> None:
    shard_roots = [tmp_path / "shard0", tmp_path / "shard1"]
    _write_shard(
        shard_roots[0] / "candidate_iou.parquet",
        [_labels("unexpected", 0.95)],
    )
    _write_shard(
        shard_roots[1] / "candidate_iou.parquet",
        [_labels("q001", 0.95)],
    )

    try:
        merge_oracle_shards(
            shard_roots=shard_roots,
            expected_sample_ids=["q000", "q001"],
            output_root=tmp_path / "merged",
            proposal_root=tmp_path / "proposals",
        )
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("identity drift was not rejected")
