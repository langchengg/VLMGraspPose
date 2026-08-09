from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from tools.modular_reranking.adapt_retained_test_candidate_tables import (
    main as adapt_main,
)


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    schema = pa.schema(
        [
            ("stage", pa.string()),
            ("sample_id", pa.string()),
            ("candidate_id", pa.string()),
            ("valid", pa.bool_()),
            ("candidate_json", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
    )


def test_retained_projection_builds_three_gt_free_stage_tables(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "raw.parquet"
    nms = tmp_path / "nms.parquet"
    _write(
        raw,
        [
            {
                "stage": "raw",
                "sample_id": "s1",
                "candidate_id": "c1",
                "valid": True,
                "candidate_json": "{}",
            },
            {
                "stage": "raw",
                "sample_id": "s1",
                "candidate_id": "c2",
                "valid": False,
                "candidate_json": "{}",
            },
            {
                "stage": "raw",
                "sample_id": "s2",
                "candidate_id": "c1",
                "valid": True,
                "candidate_json": "{}",
            },
        ],
    )
    _write(
        nms,
        [
            {
                "stage": "nms",
                "sample_id": "s1",
                "candidate_id": "c1",
                "valid": True,
                "candidate_json": "{}",
            }
        ],
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / ".RUN_ACTIVE").write_text("test\n", encoding="utf-8")
    output = run / "compact_inputs" / "test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapt_retained_test_candidate_tables.py",
            "--raw-candidates",
            str(raw),
            "--nms-candidates",
            str(nms),
            "--output-root",
            str(output),
            "--tmp-root",
            str(run / "tmp"),
            "--expected-raw",
            "3",
            "--expected-mask-valid",
            "2",
            "--expected-nms",
            "1",
        ],
    )
    assert adapt_main() == 0
    assert os.stat(raw).st_ino == os.stat(output / "raw_candidates.parquet").st_ino
    assert os.stat(nms).st_ino == os.stat(output / "nms_candidates.parquet").st_ino
    mask = pq.read_table(output / "mask_validated_candidates.parquet")
    assert mask.num_rows == 2
    assert set(mask.column("stage").to_pylist()) == {"mask_validated"}
    assert all(mask.column("valid").to_pylist())
    manifest = json.loads(
        (output / "candidate_tables.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["candidate_stage_subset_invariant"] is True
    assert manifest["candidate_primary_keys_unique"] is True
    assert manifest["single_film_used"] is False
