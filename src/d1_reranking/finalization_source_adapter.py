"""Audited finalization adapter for append-only access-event supersession."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping, Sequence

from unified_reranking.hashing import sha256_file
from unified_reranking.hashing import canonical_sha256


def access_output_supersessions(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact preclaim output records superseded by later same-stage events."""

    claim_indices = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "d1_formal_test_exclusive_claim_created"
    ]
    if len(claim_indices) != 1:
        raise RuntimeError("D1 access log must contain exactly one formal claim")
    claim_index = claim_indices[0]
    result: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        output = event.get("output_manifest")
        recorded = event.get("output_manifest_sha256")
        if not isinstance(output, str) or not output:
            continue
        path = Path(output).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"D1 access output is absent/not regular: {path}")
        current = sha256_file(path)
        if recorded == current:
            continue
        successors = [
            (later_index, later)
            for later_index, later in enumerate(events[index + 1 : claim_index], index + 1)
            if later.get("event") == event.get("event")
            and later.get("stage") == event.get("stage")
            and later.get("route") == event.get("route")
            and later.get("output_manifest") == output
            and later.get("output_manifest_sha256") == current
            and later.get("candidate_labels_opened_as_table") is False
        ]
        if len(successors) != 1:
            raise RuntimeError(
                f"D1 stale access output lacks one preclaim supersession at event {index}"
            )
        result.append(
            {
                "stale_event_index": index,
                "superseding_event_index": successors[0][0],
                "event": event.get("event"),
                "stage": event.get("stage"),
                "route": event.get("route"),
                "path": str(path),
                "stale_sha256": recorded,
                "current_sha256": current,
            }
        )
    return result


def verify_access_event_outputs_with_supersession(
    events: Sequence[Mapping[str, Any]],
) -> None:
    """Verify current outputs while preserving proven append-only predecessors."""

    from d1_reranking import postformal as locked_postformal

    superseded = {
        item["stale_event_index"] for item in access_output_supersessions(events)
    }
    for index, event in enumerate(events):
        output = event.get("output_manifest")
        output_sha = event.get("output_manifest_sha256")
        if isinstance(output, str) and output:
            current = locked_postformal._record(output)
            if index not in superseded and output_sha != current["sha256"]:
                raise RuntimeError(
                    f"D1 access event output manifest hash differs at event {index}"
                )
        elif isinstance(output, Mapping):
            locked_postformal._verified_path(
                output, name=f"D1 access output manifest {index}"
            )
        for field in ("output_manifests", "outputs"):
            records = event.get(field)
            if not isinstance(records, Mapping):
                continue
            for name, record in records.items():
                if isinstance(record, Mapping):
                    locked_postformal._verified_path(
                        record, name=f"D1 access {field} {name} at event {index}"
                    )


def access_log_check_with_plan_history(
    root: Path, sample_count: int, formal: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay locked access checks while admitting verified preclaim plan history."""

    from d1_reranking import postformal as locked_postformal

    base = locked_postformal._access_log_check_original_for_adapter(
        root, sample_count, formal
    )
    path = root / "09_formal_test/test_access.log"
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prelock_path = locked_postformal._verified_path(
        formal["plan"]["components"]["prelock_readiness"],
        name="D1 adapted access prelock readiness",
    )
    prelock = locked_postformal._load_content(
        prelock_path,
        name="D1 adapted access prelock readiness",
        statuses=("PASS",),
    )
    prefix_bytes = int(prelock["sources"]["access_log"]["bytes"])
    payload = path.read_bytes()
    prefix_count = len(
        [line for line in payload[:prefix_bytes].decode("utf-8").splitlines() if line]
    )
    suffix = events[prefix_count:]
    names = [str(event.get("event", "")) for event in suffix]
    expected_tail = [
        "d1_formal_test_exclusive_claim_created",
        "d1_raw_test_ground_truth_read_once",
        "d1_formal_test_execution_finalized",
        "d1_independent_recompute_raw_test_ground_truth_read",
        "d1_postformal_visual_ground_truth_read",
    ]
    hash_events = []
    while names and names[0] == "d1_prelock_raw_test_ground_truth_hash_only":
        hash_events.append(suffix[len(hash_events)])
        names.pop(0)
    if not hash_events or names != expected_tail:
        raise RuntimeError("D1 adapted postclaim access-event order differs")
    raw_sha = formal["plan"]["raw_test_ground_truth"]["sha256"]
    current_plan_sha = sha256_file(Path(str(formal["plan_path"])))
    if any(event.get("raw_test_ground_truth_sha256") != raw_sha for event in hash_events):
        raise RuntimeError("D1 access plan history raw-ground-truth binding differs")
    if hash_events[-1].get("formal_plan_sha256") != current_plan_sha:
        raise RuntimeError("D1 access plan history does not end at current plan")
    history = root / "configs/d1_formal_evaluation_plan_history"
    archived = {
        sha256_file(candidate)
        for candidate in history.glob("*.json")
        if candidate.is_file() and not candidate.is_symlink()
    }
    prior = [str(event.get("formal_plan_sha256", "")) for event in hash_events[:-1]]
    if len(prior) != len(set(prior)) or set(prior) != archived:
        raise RuntimeError("D1 access plan history differs from archived plans")
    primary = [e for e in events if e.get("event") == expected_tail[1]]
    independent = [e for e in events if e.get("event") == expected_tail[3]]
    visual = [e for e in events if e.get("event") == expected_tail[4]]
    exact_rows = all(
        len(group) == 1 and int(group[0].get("row_count", -1)) == sample_count
        for group in (primary, independent, visual)
    )
    visual_order = bool(
        visual
        and visual[0].get("opened_after_complete_formal_claim") is True
        and visual[0].get("opened_after_independent_recompute_pass") is True
    )
    passed = (
        exact_rows
        and visual_order
        and base["forbidden_events"] == 0
        and base["duplicate_event_ids"] is False
        and base["timestamps_ordered"] is True
        and base["prelock_stage_inventory_matches"] is True
        and base["unknown_read_like_events"] == 0
    )
    result = dict(base)
    result.update(
        {
            "status": "PASS" if passed else "FAIL",
            "postclaim_event_inventory_matches": True,
            "archived_formal_plan_events": len(prior),
            "current_formal_plan_event": 1,
        }
    )
    return result


def reuse_source_immutability_after(
    root: Path, formal: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and reuse a fully materialized immutable-source AFTER audit."""

    from d1_reranking import postformal as locked_postformal

    path = root / "00_audit/SOURCE_RUN_IMMUTABILITY_AFTER.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if value.get("status") != "PASS" or recorded != canonical_sha256(unsigned):
        raise RuntimeError("D1 source immutability AFTER self hash differs")
    closure_record = value.get("source_closure")
    current_closure = formal.get("plan", {}).get("sources", {}).get("source_closure")
    if not isinstance(closure_record, Mapping) or not isinstance(
        current_closure, Mapping
    ):
        raise RuntimeError("D1 source immutability closure record is absent")
    for key in ("path", "sha256", "bytes"):
        if closure_record.get(key) != current_closure.get(key):
            raise RuntimeError("D1 source immutability/P14 closure differs")
    locked_postformal._verified_path(
        closure_record, name="D1 reused source immutability closure"
    )
    locked_postformal._verified_path(
        value.get("before"), name="D1 reused source immutability BEFORE"
    )
    records = value.get("source_records")
    if not isinstance(records, Mapping) or len(records) != int(
        value.get("closure_source_count", -1)
    ):
        raise RuntimeError("D1 reused source inventory count differs")
    for name, record in records.items():
        resolved = locked_postformal._verified_path(
            record, name=f"D1 reused source {name}"
        )
        if str(resolved) != str(Path(str(name)).expanduser().resolve()):
            raise RuntimeError("D1 reused source inventory key differs")
    return {"status": "PASS", "artifact": locked_postformal._record(path)}


__all__ = [
    "access_log_check_with_plan_history",
    "access_output_supersessions",
    "reuse_source_immutability_after",
    "verify_access_event_outputs_with_supersession",
]
