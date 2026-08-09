from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from failure_analysis.gemini_crog_evidence_v1 import MODEL_IDS
from failure_analysis.gemini_crog_evidence_v1.planner import (
    CacheContract,
    CohortSizes,
    LogicalRequestKey,
    MAX_PLANNED_REQUEST_SLOTS,
    P1_PROTOCOL,
    PlanAssumptions,
    annotation_query_type,
    build_full_run_manifests,
    build_full_run_plan,
    load_annotation_query_types,
    load_exact_p1_cache_keys,
    request_slot_contract,
    with_content_digest,
)


def _annotation(sample_id: int, kind: str) -> dict:
    templates = {
        "name": ("name.json", [{"type": "scene"}]),
        "attribute": ("attribute.json", [{"type": "filter_color"}]),
        "location": ("location.json", [{"type": "filter_location"}]),
        "relation": (
            "relation.json",
            [{"type": "scene"}, {"type": "relate"}, {"type": "return"}],
        ),
        "mixed": (
            "relation.json",
            [{"type": "scene"}, {"type": "filter_color"}, {"type": "relate"}],
        ),
    }
    filename, program = templates[kind]
    return {
        "question_index": sample_id,
        "template_filename": filename,
        "program": program,
        "question": f"question {sample_id}",
        "target": f"secret-target-{sample_id}",
        "answer": f"secret-answer-{sample_id}",
    }


def _write_annotations(path: Path, count: int) -> None:
    query_types = ("name", "attribute", "relation", "location", "mixed")
    path.write_text(
        json.dumps(
            {
                "data": [
                    _annotation(index, query_types[index % len(query_types)])
                    for index in range(count)
                ]
            }
        ),
        encoding="utf-8",
    )


def _split_row(
    official_split: str,
    partition: str,
    local_id: int,
) -> dict:
    stable_split = official_split
    return {
        "sample_id": f"multiple:{stable_split}:{local_id:08d}",
        "source_sample_id": local_id,
        "official_split": official_split,
        "development_partition": partition,
        "frame_id": f"scene-{local_id % 7}/frame-{local_id // 2}.png",
        "scene_id": f"scene-{local_id % 7}/frame-{local_id // 2}.png",
        "sequence_id": f"group-{local_id % 5}",
    }


@pytest.fixture()
def synthetic_sources(tmp_path: Path):
    sizes = CohortSizes(
        pilot=5,
        stability=3,
        ablation=8,
        calibration=4,
        validation=3,
        formal_test=4,
    )
    rows = [
        *[_split_row("train", "train", index) for index in range(12)],
        *[_split_row("train", "calibration", index) for index in range(12, 16)],
        *[_split_row("val", "validation", index) for index in range(3)],
        *[_split_row("test", "test", index) for index in range(4)],
    ]
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    annotations = {
        "train": tmp_path / "train_expressions.json",
        "val": tmp_path / "val_expressions.json",
        "test": tmp_path / "test_expressions.json",
    }
    _write_annotations(annotations["train"], 16)
    _write_annotations(annotations["val"], 3)
    _write_annotations(annotations["test"], 4)
    strata_names = (
        "q_only_correct",
        "q_only_wrong_recoverable",
        "top5_all_wrong",
        "high_q_margin",
        "low_q_margin",
        "small_target",
        "large_target",
        "heavy_overlap",
        "low_predicted_mask_confidence",
        "high_predicted_mask_confidence",
    )
    strata = {
        f"multiple:train:{index:08d}": [strata_names[index % len(strata_names)]]
        for index in range(12)
    }
    smoke = tmp_path / "selected_local_ids.json"
    smoke.write_text(json.dumps([0, 1]), encoding="utf-8")
    return sizes, split, annotations, strata, smoke


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_annotation_query_type_uses_metadata_but_does_not_return_program(tmp_path: Path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    _annotation(0, "name"),
                    _annotation(1, "attribute"),
                    _annotation(2, "relation"),
                    _annotation(3, "location"),
                    _annotation(4, "mixed"),
                ]
            }
        ),
        encoding="utf-8",
    )
    result = load_annotation_query_types(path)
    assert result == {
        0: "name",
        1: "attribute",
        2: "relation",
        3: "location",
        4: "mixed",
    }
    assert all(isinstance(value, str) for value in result.values())
    assert annotation_query_type(_annotation(9, "mixed")) == "mixed"


def test_manifests_are_deterministic_immutable_and_keep_smoke_ids(
    tmp_path: Path,
    synthetic_sources,
):
    sizes, split, annotations, strata, smoke = synthetic_sources
    run_root = tmp_path / "run"
    first = build_full_run_manifests(
        run_root=run_root,
        split_manifest_path=split,
        annotation_paths=annotations,
        sizes=sizes,
        smoke_ids_path=smoke,
        development_strata=strata,
    )
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_root.glob("*_manifest.json")
    }
    second = build_full_run_manifests(
        run_root=run_root,
        split_manifest_path=split,
        annotation_paths=annotations,
        sizes=sizes,
        smoke_ids_path=smoke,
        development_strata=strata,
    )
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in run_root.glob("*_manifest.json")
    }
    assert first == second
    assert before == after
    assert set(first) == {
        "pilot",
        "stability",
        "ablation",
        "calibration",
        "validation",
        "formal_test",
    }
    pilot_ids = {row["sample_id"] for row in first["pilot"]["rows"]}
    ablation_ids = {row["sample_id"] for row in first["ablation"]["rows"]}
    stability_ids = {row["sample_id"] for row in first["stability"]["rows"]}
    assert {"multiple:train:00000000", "multiple:train:00000001"} <= pilot_ids
    assert pilot_ids <= ablation_ids
    assert stability_ids <= pilot_ids
    assert first["pilot"]["sample_count"] == sizes.pilot
    assert first["calibration"]["sample_count"] == sizes.calibration
    assert first["validation"]["sample_count"] == sizes.validation
    assert first["formal_test"]["sample_count"] == sizes.formal_test
    assert sum(first["pilot"]["query_type_distribution"].values()) == sizes.pilot
    assert sum(first["pilot"]["scene_distribution"].values()) == sizes.pilot
    assert sum(first["pilot"]["group_distribution"].values()) == sizes.pilot

    forbidden_keys = {
        "program",
        "answer",
        "target",
        "target_idx",
        "box",
        "grasps",
        "objid",
        "candidate_correct",
        "candidate_iou",
        "oracle",
        "recovered",
        "harmful",
    }
    assert forbidden_keys.isdisjoint(set(_all_keys(first)))
    for manifest in first.values():
        for row in manifest["rows"]:
            assert set(row["evaluation_only"]) == {"query_type", "strata"}

    with pytest.raises(FileExistsError, match="immutable artifact already differs"):
        build_full_run_manifests(
            run_root=run_root,
            split_manifest_path=split,
            annotation_paths=annotations,
            sizes=sizes,
            seed=99,
            smoke_ids_path=smoke,
            development_strata=strata,
        )


def _plan_manifest(name: str, sample_ids: list[str], partition: str = "train") -> dict:
    rows = [
        {
            "sample_id": sample_id,
            "scene_id": f"scene-{index % 3}",
            "group_id": f"group-{index % 2}",
            "evaluation_only": {"query_type": "name", "strata": []},
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    return with_content_digest(
        {
            "cohort": name,
            "partition": partition,
            "sample_count": len(rows),
            "rows": rows,
            "scene_distribution": {"scene": len(rows)},
            "group_distribution": {"group": len(rows)},
            "query_type_distribution": {
                "name": len(rows),
                "attribute": 0,
                "relation": 0,
                "location": 0,
                "mixed": 0,
            },
        }
    )


def _small_plan_manifests() -> dict[str, dict]:
    pilot = [f"multiple:train:{index:08d}" for index in range(5)]
    return {
        "pilot": _plan_manifest("pilot", pilot),
        "stability": _plan_manifest("stability", pilot[:3]),
        "ablation": _plan_manifest(
            "ablation", pilot + [f"multiple:train:{index:08d}" for index in range(5, 8)]
        ),
        "calibration": _plan_manifest(
            "calibration",
            [f"multiple:train:{index:08d}" for index in range(20, 24)],
            "calibration",
        ),
        "validation": _plan_manifest(
            "validation",
            [f"multiple:val:{index:08d}" for index in range(3)],
            "validation",
        ),
        "formal_test": _plan_manifest(
            "formal_test",
            [f"multiple:test:{index:08d}" for index in range(4)],
            "test",
        ),
    }


def test_default_request_contract_is_exact_and_capped():
    counts = request_slot_contract()
    assert counts == {
        "pilot": 200,
        "stability": 120,
        "ablation": 4000,
        "calibration": 19580,
        "validation": 17338,
        "formal_test": 35498,
        "total": 76736,
    }
    assert counts["total"] == MAX_PLANNED_REQUEST_SLOTS


def test_plan_counts_cache_reuse_cost_tokens_and_cap(tmp_path: Path):
    manifests = _small_plan_manifests()
    exact_cache_key = LogicalRequestKey(
        "inference",
        P1_PROTOCOL,
        MODEL_IDS[0],
        "multiple:train:00000000",
        0,
    ).encode()
    assumptions = PlanAssumptions(
        concurrency=2,
        retry_max=5,
        er2_cost_cap_per_request_usd=0.25,
        max_spend_usd=1000.0,
    )
    plan = build_full_run_plan(
        run_root=tmp_path / "run",
        manifests=manifests,
        assumptions=assumptions,
        existing_cache_keys={exact_cache_key},
        max_request_slots=114,
    )
    totals = plan["totals"]
    assert totals["planned_request_slots"] == 114
    assert totals["planned_unique_requests"] == 104
    assert totals["cross_phase_cache_hits"] == 10
    assert totals["existing_cache_hits"] == 1
    assert totals["cache_hits"] == 11
    assert totals["new_requests"] == 103
    assert totals["expected_retries_upper_bound"] == 515
    assert totals["expected_token_usage"][MODEL_IDS[0]]["requests"] == 51
    assert totals["expected_token_usage"][MODEL_IDS[1]]["requests"] == 52
    assert totals["expected_cost"]["er2_conservative_reserve_usd"] == 12.75
    assert totals["expected_cost"]["flash_estimated_usd"] > 0
    assert plan["status"] == "ready"
    assert plan["phases"][2]["phase"] == "ablation"
    assert plan["phases"][2]["cross_phase_cache_hits"] == 10
    assert plan["cache_contract"]["stability_namespace_isolated"] is True

    with pytest.raises(AssertionError, match="exceed cap"):
        build_full_run_plan(
            run_root=tmp_path / "over-cap",
            manifests=manifests,
            assumptions=assumptions,
            max_request_slots=113,
        )


def test_plan_blocks_missing_budget_or_er2_cap(tmp_path: Path):
    plan = build_full_run_plan(
        run_root=tmp_path / "run",
        manifests=_small_plan_manifests(),
        assumptions=PlanAssumptions(),
        max_request_slots=114,
    )
    assert plan["status"] == "blocked"
    assert plan["blockers"] == [
        "missing_GEMINI_MAX_SPEND_USD",
        "missing_GEMINI_ER2_COST_CAP_PER_REQUEST_USD",
    ]


def test_exact_cache_projection_requires_valid_4096_contract(tmp_path: Path):
    cache = tmp_path / "cache.sqlite"
    connection = sqlite3.connect(cache)
    connection.execute(
        """
        CREATE TABLE responses (
          sample_id TEXT, model_id TEXT, valid INTEGER,
          generation_config_json TEXT, prompt_hash TEXT, schema_hash TEXT,
          renderer_hash TEXT, evidence_schema_hash TEXT
        )
        """
    )
    rows = [
        ("multiple:train:00000000", MODEL_IDS[0], 1, 4096, "p"),
        ("multiple:train:00000001", MODEL_IDS[1], 1, 3072, "p"),
        ("multiple:train:00000002", MODEL_IDS[1], 0, 4096, "p"),
        ("multiple:train:00000003", MODEL_IDS[1], 1, 4096, "wrong"),
    ]
    for sample_id, model_id, valid, tokens, prompt in rows:
        connection.execute(
            "INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)",
            (
                sample_id,
                model_id,
                valid,
                json.dumps({"max_output_tokens": tokens, "image_resolution": "high"}),
                prompt,
                "s",
                "r",
                "e",
            ),
        )
    connection.commit()
    connection.close()
    keys = load_exact_p1_cache_keys(
        [cache],
        contract=CacheContract(
            prompt_hash="p",
            schema_hash="s",
            renderer_hash="r",
            evidence_schema_hash="e",
        ),
    )
    assert keys == {
        LogicalRequestKey(
            "inference", P1_PROTOCOL, MODEL_IDS[0], "multiple:train:00000000", 0
        ).encode()
    }

