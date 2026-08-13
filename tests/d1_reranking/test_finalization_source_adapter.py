from __future__ import annotations

from pathlib import Path

from d1_reranking.finalization_source_adapter import access_output_supersessions
from unified_reranking.hashing import sha256_file


def test_access_output_accepts_one_later_preclaim_same_stage(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("new", encoding="utf-8")
    events = [
        {
            "event": "prelock_label_free_test_stage",
            "stage": "gate",
            "route": "D1",
            "output_manifest": str(output),
            "output_manifest_sha256": "a" * 64,
            "candidate_labels_opened_as_table": False,
        },
        {
            "event": "prelock_label_free_test_stage",
            "stage": "gate",
            "route": "D1",
            "output_manifest": str(output),
            "output_manifest_sha256": sha256_file(output),
            "candidate_labels_opened_as_table": False,
        },
        {"event": "d1_formal_test_exclusive_claim_created"},
    ]
    observed = access_output_supersessions(events)
    assert len(observed) == 1
    assert observed[0]["stale_event_index"] == 0
    assert observed[0]["superseding_event_index"] == 1


def test_access_output_rejects_postclaim_or_other_stage_successor(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"
    output.write_text("new", encoding="utf-8")
    events = [
        {
            "event": "prelock_label_free_test_stage",
            "stage": "gate",
            "route": "D1",
            "output_manifest": str(output),
            "output_manifest_sha256": "a" * 64,
            "candidate_labels_opened_as_table": False,
        },
        {"event": "d1_formal_test_exclusive_claim_created"},
        {
            "event": "prelock_label_free_test_stage",
            "stage": "other",
            "route": "D1",
            "output_manifest": str(output),
            "output_manifest_sha256": sha256_file(output),
            "candidate_labels_opened_as_table": False,
        },
    ]
    try:
        access_output_supersessions(events)
    except RuntimeError as error:
        assert "lacks one preclaim supersession" in str(error)
    else:
        raise AssertionError("postclaim/other-stage supersession was accepted")
