from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.unified_reranking.pipeline_status import audit_pipeline_readiness
from src.unified_reranking.test_access_guard import append_access_log


def test_pipeline_label_state_tracks_formal_execution_lifecycle(tmp_path: Path) -> None:
    preformal = audit_pipeline_readiness(tmp_path, process_rows=[])
    assert preformal["candidate_test_labels_read"] is False
    assert preformal["candidate_test_label_access_state"] == "PRELOCK_LABEL_FREE"
    assert preformal["candidate_test_label_read_event_count"] == 0

    execution = tmp_path / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    execution.parent.mkdir(parents=True)
    execution.write_text(
        json.dumps({"status": "COMPLETE", "execution_count": 1}),
        encoding="utf-8",
    )
    completed = audit_pipeline_readiness(tmp_path, process_rows=[])
    assert completed["candidate_test_labels_read"] is True
    assert completed["candidate_test_label_access_state"] == "FORMAL_TEST_COMPLETE"


def test_access_log_append_is_process_safe_and_lossless(tmp_path: Path) -> None:
    count = 40
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: append_access_log(
                    tmp_path, {"event": "synthetic_concurrent", "index": index}
                ),
                range(count),
            )
        )
    path = tmp_path / "09_formal_test" / "test_access.log"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == count
    assert {row["index"] for row in rows} == set(range(count))
