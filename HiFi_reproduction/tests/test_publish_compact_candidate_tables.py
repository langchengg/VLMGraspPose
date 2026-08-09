from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.modular_reranking.publish_compact_candidate_tables import (
    STAGE_FILES,
    main,
    sha256_file,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    (run / ".RUN_ACTIVE").write_text("\n", encoding="utf-8")
    temporary = run / "tmp"
    source = temporary / "train" / "candidates" / "merged"
    source.mkdir(parents=True)
    artifacts = {}
    counts = {"execution_failures": 0}
    for stage, filename in STAGE_FILES.items():
        path = source / filename
        table = pa.table(
            {
                "sample_id": ["sample-1", "sample-1"],
                "candidate_id": ["g0000", "g0001"],
                "center_u_px": [10.0, 20.0],
            }
        )
        pq.write_table(table, path, compression="zstd")
        artifacts[stage] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": 2,
            "primary_key": ["sample_id", "candidate_id"],
        }
    (source / "run_config.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "split": "train",
                "counts": counts,
                "candidate_stage_artifacts": artifacts,
                "protocol_identity_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return run, temporary, source


def test_publish_uses_verified_hardlinks_and_persistent_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, temporary, source = _fixture(tmp_path)
    output = run / "candidate_tables" / "train"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_compact_candidate_tables.py",
            "--split",
            "train",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--tmp-root",
            str(temporary),
        ],
    )
    assert main() == 0
    manifest = json.loads(
        (output / "candidate_tables.manifest.json").read_text()
    )
    assert manifest["gt_free"] is True
    assert manifest["storage"]["data_blocks_duplicated_by_publication"] is False
    for stage, filename in STAGE_FILES.items():
        source_stat = (source / filename).stat()
        output_stat = (output / filename).stat()
        assert (source_stat.st_dev, source_stat.st_ino) == (
            output_stat.st_dev,
            output_stat.st_ino,
        )
        assert manifest["artifacts"][stage]["path"] == str(
            (output / filename).resolve()
        )


def test_publish_rejects_changed_source_and_tmp_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, temporary, source = _fixture(tmp_path)
    (source / "raw_candidates.parquet").write_bytes(b"changed")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_compact_candidate_tables.py",
            "--split",
            "train",
            "--source-root",
            str(source),
            "--output-root",
            str(run / "candidate_tables" / "train"),
            "--tmp-root",
            str(temporary),
        ],
    )
    with pytest.raises(Exception, match="raw"):
        main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_compact_candidate_tables.py",
            "--split",
            "train",
            "--source-root",
            str(source),
            "--output-root",
            str(temporary / "published"),
            "--tmp-root",
            str(temporary),
        ],
    )
    with pytest.raises(ValueError, match="outside tmp-root"):
        main()
