from __future__ import annotations

import ast
import builtins
import json
import socket
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import failure_analysis.vlm_safe_rerank.replay as replay_module
from failure_analysis.vlm_safe_rerank.replay import (
    MODEL_IDS,
    build_cohort_records,
    ledger_summary,
    replay_direct_decisions,
    replay_existing_direct,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _candidate(index: int, q_raw: float) -> dict:
    return {
        "candidate_id": f"candidate_{index}",
        "q_raw": q_raw,
        "features": {
            "mask_consistency": {"value": 0.5 + index / 10, "reliability": 1.0},
            "depth_mad_m": {"value": 0.01 + index / 100, "reliability": 0.8},
        },
        "diagnostics": {"depth_available": True},
    }


def _source_fixture(tmp_path: Path, count: int = 4) -> dict[str, Path]:
    paths = {
        "features": tmp_path / "features.jsonl",
        "legacy": tmp_path / "legacy.jsonl",
        "corrected": tmp_path / "corrected.jsonl",
        "split": tmp_path / "split.json",
        "annotations": tmp_path / "annotations.json",
    }
    features: list[dict] = []
    legacy: list[dict] = []
    corrected: list[dict] = []
    split_rows: list[dict] = []
    annotations: list[dict] = []
    # candidate_2 is deliberately first by q but not first in the source list.
    source_candidates = [
        _candidate(4, 0.2),
        _candidate(0, 0.8),
        _candidate(2, 1.0),
        _candidate(1, 0.6),
        _candidate(3, 0.4),
    ]
    for index in range(count):
        sample_id = f"multiple:train:{index:08d}"
        features.append(
            {
                "sample_id": index,
                "sample_index": index,
                "split": "train",
                "candidates": source_candidates,
            }
        )
        if index == 0:
            correct = {"candidate_2"}  # protected q-only success
        elif index == 1:
            correct = {"candidate_1"}  # recoverable q-only error
        elif index == 2:
            correct = set()  # unrecoverable q-only error
        else:
            correct = {"candidate_2", "candidate_1"}
        label = {
            "sample_id": sample_id,
            "candidate_labels": [
                {
                    "candidate_id": f"candidate_{candidate_index}",
                    "candidate_correct": f"candidate_{candidate_index}" in correct,
                }
                for candidate_index in range(5)
            ],
        }
        legacy.append(label)
        corrected.append(json.loads(json.dumps(label)))
        split_rows.append(
            {
                "sample_id": sample_id,
                "development_partition": "train",
                "official_split": "train",
                "source_sample_id": index,
                "frame_id": f"frame-{index}",
                "scene_id": f"scene-{index}",
                "sequence_id": f"sequence-{index // 2}",
            }
        )
        annotations.append(
            {
                "question_index": index,
                "template_filename": "name.json",
                "program": [{"type": "scene"}],
                "question": f"question {index}",
            }
        )
    _write_jsonl(paths["features"], features)
    _write_jsonl(paths["legacy"], legacy)
    _write_jsonl(paths["corrected"], corrected)
    _write_json(paths["split"], {"rows": split_rows})
    _write_json(paths["annotations"], {"data": annotations})
    return paths


def _build_records(paths: dict[str, Path], partitions: set[str] | None = None):
    return build_cohort_records(
        features_path=paths["features"],
        legacy_labels_path=paths["legacy"],
        corrected_labels_path=paths["corrected"],
        split_manifest_path=paths["split"],
        annotation_paths={"train": paths["annotations"]},
        partitions=partitions,
    )


def _decision(
    sample_index: int,
    *,
    model_id: str = "gemini-3.6-flash",
    selected: str = "candidate_1",
    lifecycle: str = "SUCCEEDED",
    json_valid: bool = True,
    schema_valid: bool = True,
    abstain: bool = False,
    permanent: bool = False,
) -> dict:
    return {
        "sample_id": f"multiple:train:{sample_index:08d}",
        "model_id": model_id,
        "request_hash": f"hash-{model_id}-{sample_index}",
        "selected_candidate_id": selected,
        "lifecycle_status": lifecycle,
        "json_parse_valid": json_valid,
        "schema_valid": schema_valid,
        "abstain": abstain,
        "permanent_api_failure": permanent,
        "cache_hit": False,
        "api_attempted": True,
        "latency_seconds": 1.0 + sample_index,
        "estimated_charge_usd": 0.01,
    }


def _write_decisions(path: Path, decisions: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index, decision in enumerate(decisions):
        _write_json(path / f"{index:02d}.json", decision)


def _cache(path: Path, *, duplicate_success: bool = False) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE request_attempts (
          attempt_id INTEGER PRIMARY KEY,
          request_hash TEXT,
          status TEXT,
          estimated_charge_usd REAL
        );
        CREATE TABLE responses (
          request_hash TEXT PRIMARY KEY,
          valid INTEGER,
          estimated_charge_usd REAL
        );
        """
    )
    connection.execute(
        "INSERT INTO request_attempts VALUES (1, 'request-a', 'SUCCEEDED', 0.01)"
    )
    if duplicate_success:
        connection.execute(
            "INSERT INTO request_attempts VALUES (2, 'request-a', 'SUCCEEDED', 0.02)"
        )
    connection.execute("INSERT INTO responses VALUES ('request-a', 1, 0.01)")
    connection.commit()
    connection.close()


def test_cohort_records_use_frozen_q_and_dual_track_labels(tmp_path: Path) -> None:
    paths = _source_fixture(tmp_path, count=3)
    records, summary = _build_records(paths)
    by_sample = {row["sample_id"]: row for row in records}

    assert all(row["q_only_candidate_id"] == "candidate_2" for row in records)
    assert all(row["candidate_1_id"] == "candidate_2" for row in records)
    assert by_sample["multiple:train:00000000"]["legacy_cohort"] == "protected_correct"
    assert by_sample["multiple:train:00000001"]["legacy_cohort"] == "recoverable_error"
    assert by_sample["multiple:train:00000002"]["legacy_cohort"] == "unrecoverable_error"
    assert [row["corrected_cohort"] for row in records] == [
        "protected_correct",
        "recoverable_error",
        "unrecoverable_error",
    ]
    assert summary["partitions"]["train"]["legacy"]["cohorts"] == {
        "protected_correct": 1,
        "recoverable_error": 1,
        "unrecoverable_error": 1,
    }
    assert summary["partitions"]["train"]["query_type_distribution"] == {"name": 3}


def test_replay_uses_full_fallback_denominator_and_retains_permanent_failure(
    tmp_path: Path,
) -> None:
    paths = _source_fixture(tmp_path / "sources")
    records, _ = _build_records(paths)
    decision_dir = tmp_path / "decisions"
    _write_decisions(
        decision_dir,
        [
            _decision(0),
            _decision(
                1,
                lifecycle="PERMANENT_FAILED",
                json_valid=False,
                schema_valid=False,
                permanent=True,
            ),
            _decision(2, lifecycle="ABSTAIN", abstain=True),
            _decision(
                3,
                lifecycle="TECHNICAL_FALLBACK",
                json_valid=False,
                schema_valid=False,
            ),
        ],
    )

    outcomes, summary = replay_direct_decisions(
        decision_dir=decision_dir, cohort_records=records
    )

    model = summary["models"]["gemini-3.6-flash"]
    assert model["decisions"] == model["legacy"]["total"] == 4
    assert model["fallback"] == 3
    assert model["schema_valid"] == 2
    assert model["terminal_failures"] == 1
    assert sum(row["fallback_to_q_only"] for row in outcomes) == 3
    permanent = next(row for row in outcomes if row["permanent_api_failure"])
    assert permanent["status"] == "PERMANENT_FAILED"
    assert permanent["selected_candidate_id"] == permanent["q_only_candidate_id"]
    assert permanent["switch"] is False


def test_candidate_and_q_inputs_are_read_only_and_outside_top5_is_rejected(
    tmp_path: Path,
) -> None:
    paths = _source_fixture(tmp_path / "sources", count=1)
    before = {name: path.read_bytes() for name, path in paths.items()}
    records, _ = _build_records(paths)
    records_before = json.loads(json.dumps(records))
    valid_dir = tmp_path / "valid"
    _write_decisions(valid_dir, [_decision(0, selected="candidate_1")])

    outcomes, _ = replay_direct_decisions(
        decision_dir=valid_dir, cohort_records=records
    )
    assert outcomes[0]["q_only_candidate_id"] == "candidate_2"
    assert outcomes[0]["selected_candidate_id"] == "candidate_1"
    assert records == records_before
    assert all(path.read_bytes() == before[name] for name, path in paths.items())

    invalid_dir = tmp_path / "invalid"
    _write_decisions(invalid_dir, [_decision(0, selected="candidate_outside")])
    with pytest.raises(ValueError, match="outside frozen Top-5"):
        replay_direct_decisions(decision_dir=invalid_dir, cohort_records=records)


def test_split_manifest_controls_partition_and_evaluator_identity(tmp_path: Path) -> None:
    paths = _source_fixture(tmp_path / "partitioned", count=2)
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    split["rows"][1]["development_partition"] = "calibration"
    _write_json(paths["split"], split)
    records, _ = _build_records(paths, partitions={"calibration"})
    assert [row["sample_id"] for row in records] == ["multiple:train:00000001"]
    assert records[0]["partition"] == "calibration"
    assert records[0]["official_split"] == "train"

    mismatched = _source_fixture(tmp_path / "mismatched", count=1)
    corrected = [json.loads(line) for line in mismatched["corrected"].read_text().splitlines()]
    corrected[0]["sample_id"] = "multiple:train:99999999"
    _write_jsonl(mismatched["corrected"], corrected)
    with pytest.raises(ValueError, match="feature/evaluator identity mismatch"):
        _build_records(mismatched)

    absent = _source_fixture(tmp_path / "absent", count=1)
    split = json.loads(absent["split"].read_text(encoding="utf-8"))
    split["rows"][0]["sample_id"] = "multiple:train:99999999"
    _write_json(absent["split"], split)
    with pytest.raises(ValueError, match="absent from frozen split manifest"):
        _build_records(absent)


def test_replay_is_statically_and_dynamically_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = ast.parse(Path(replay_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name.startswith(("google", "requests", "httpx", "urllib", "aiohttp"))
        or name.endswith(".api")
        for name in imported
    )

    paths = _source_fixture(tmp_path / "sources", count=1)
    existing = tmp_path / "existing"
    _write_decisions(existing / "pilot" / "decisions", [_decision(0)])
    _cache(existing / "gemini_cache.sqlite")
    source_bytes = {name: path.read_bytes() for name, path in paths.items()}

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("google") or name.endswith(".api") or ".api." in name:
            raise AssertionError(f"API import attempted: {name}")
        return real_import(name, *args, **kwargs)

    def forbidden_socket(*args, **kwargs):
        raise AssertionError("network socket attempted during offline replay")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "socket", forbidden_socket)
    output = tmp_path / "output"
    result = replay_existing_direct(
        output_dir=output,
        existing_run_root=existing,
        features_path=paths["features"],
        legacy_labels_path=paths["legacy"],
        corrected_labels_path=paths["corrected"],
        split_manifest_path=paths["split"],
        annotation_paths={"train": paths["annotations"]},
    )

    assert result["mode"] == "offline_replay_no_api"
    assert result["direct"]["decision_records"] == 1
    assert pq.read_table(output / "per_sample_cohorts.parquet").num_rows == 1
    assert pq.read_table(output / "direct_decision_outcomes.parquet").num_rows == 1
    assert all(path.read_bytes() == source_bytes[name] for name, path in paths.items())


def test_common_valid_agreement_excludes_permanent_failure(tmp_path: Path) -> None:
    paths = _source_fixture(tmp_path / "sources", count=2)
    records, _ = _build_records(paths)
    decision_dir = tmp_path / "decisions"
    _write_decisions(
        decision_dir,
        [
            _decision(0, model_id=MODEL_IDS[0], selected="candidate_1"),
            _decision(0, model_id=MODEL_IDS[1], selected="candidate_1"),
            _decision(1, model_id=MODEL_IDS[0], selected="candidate_1"),
            _decision(
                1,
                model_id=MODEL_IDS[1],
                selected="candidate_2",
                lifecycle="PERMANENT_FAILED",
                json_valid=False,
                schema_valid=False,
                permanent=True,
            ),
        ],
    )

    _, summary = replay_direct_decisions(
        decision_dir=decision_dir, cohort_records=records
    )
    assert summary["decision_records"] == 4
    assert summary["models"][MODEL_IDS[1]]["terminal_failures"] == 1
    assert summary["paired_common_valid"] == 1
    assert summary["paired_selected_id_agreement"] == 1
    assert summary["paired_selected_id_agreement_rate"] == 1.0


def test_ledger_summary_detects_duplicate_successful_request_hash(tmp_path: Path) -> None:
    cache = tmp_path / "cache.sqlite"
    _cache(cache, duplicate_success=True)
    before = cache.read_bytes()
    result = ledger_summary(cache)

    assert result["attempts"] == 2
    assert result["distinct_attempt_request_hashes"] == 1
    assert result["attempt_status"] == {"SUCCEEDED": 2}
    assert result["duplicate_successful_request_hashes"] == 1
    assert cache.read_bytes() == before
