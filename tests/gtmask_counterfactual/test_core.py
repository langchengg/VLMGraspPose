from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.audit import (  # noqa: E402
    bootstrap_run,
    transition_pipeline_status,
    verify_source_final_lock,
)
from gtmask_counterfactual.baseline import (  # noqa: E402
    assert_no_raw_test_supervision,
    replay_baselines,
)
from gtmask_counterfactual.contracts import BaselineTarget, RunState  # noqa: E402
from gtmask_counterfactual.io import (  # noqa: E402
    artifact_record,
    canonical_sha256,
    sha256_file,
)
from gtmask_counterfactual.protocol import (  # noqa: E402
    claim_bulk_execution,
    create_protocol_lock,
    inline_binding,
    load_execution_authority,
    verify_protocol_lock,
)


def _source_lock(root: Path, *, unified_style: bool = False) -> str:
    artifact = root / "artifacts" / "frozen.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frozen")
    record = {
        "relative_path": str(artifact.relative_to(root)),
        **artifact_record(artifact),
    }
    inventory = [record]
    lock: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "inventory": inventory,
        "inventory_count": 1,
        (
            "inventory_content_sha256" if unified_style else "inventory_sha256"
        ): canonical_sha256(inventory),
    }
    lock["self_sha256"] = canonical_sha256(lock)
    lock_path = root / "FINAL_RUN_LOCK.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    return sha256_file(lock_path)


def _verification(root: Path) -> dict[str, object]:
    lock_path = root / "FINAL_RUN_LOCK.json"
    if not lock_path.exists():
        _source_lock(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(
                {"status": "COMPLETE", "formal_test_execution_count": 1},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    verified = verify_source_final_lock(
        root,
        expected_file_sha256=sha256_file(lock_path),
        full_inventory_rehash=True,
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "full_inventory_byte_rehash": True,
        "sources": {"synthetic": verified},
    }


def test_source_lock_file_self_inventory_and_full_byte_rehash(tmp_path: Path) -> None:
    expected = _source_lock(tmp_path, unified_style=True)
    report = verify_source_final_lock(
        tmp_path,
        expected_file_sha256=expected,
        full_inventory_rehash=True,
    )
    assert report["inventory_files_rehashed"] == 1
    assert report["scientific_rows_opened"] == 0
    (tmp_path / "artifacts/frozen.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="byte record differs"):
        verify_source_final_lock(
            tmp_path,
            expected_file_sha256=expected,
            full_inventory_rehash=True,
        )


def test_source_lock_rehash_is_explicit_and_mandatory(tmp_path: Path) -> None:
    expected = _source_lock(tmp_path)
    with pytest.raises(PermissionError, match="mandatory"):
        verify_source_final_lock(
            tmp_path,
            expected_file_sha256=expected,
            full_inventory_rehash=False,
        )


def test_bootstrap_creates_independent_status_and_sample_ledger(tmp_path: Path) -> None:
    run = tmp_path / "run"
    status = bootstrap_run(run, source_verification=_verification(tmp_path))
    assert status["status"] == RunState.P0_AUDIT.value
    assert status["counterfactual_execution_count"] == 0
    with sqlite3.connect(run / "run_ledger.sqlite") as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(counterfactual_jobs)")
        }
    assert {"stage", "route", "branch", "sample_id", "artifact_sha256"} <= columns


def _unified_outcomes() -> pd.DataFrame:
    rows = []
    for route in ("g1", "c1"):
        for system, selected in (
            (f"{route}_native", [True, False, False]),
            (f"{route}_gated_primary", [True, True, False]),
        ):
            for sample_id, correct, rank, count in zip(
                ("s1", "s2", "s3"),
                selected,
                (1.0, None, None),
                (5, 5, 0),
                strict=True,
            ):
                rows.append(
                    {
                        "system_name": system,
                        "sample_id": sample_id,
                        "selected_correct": correct,
                        "first_positive_rank": rank,
                        "candidate_count": count,
                    }
                )
    return pd.DataFrame(rows)


def _d1_outcomes() -> pd.DataFrame:
    rows = []
    systems = {
        "d1_top5_r0": ([True, False, False], [1.0, 2.0, None]),
        "d1_top5_r7_gated": ([True, True, False], [1.0, 2.0, None]),
        "d1_top10_locked": ([True, True, False], [1.0, 2.0, None]),
        "d1_allnms_locked": ([True, True, False], [1.0, 6.0, None]),
    }
    for system, (selected, ranks) in systems.items():
        for sample_id, correct, rank in zip(
            ("s1", "s2", "s3"), selected, ranks, strict=True
        ):
            rows.append(
                {
                    "system_name": system,
                    "sample_id": sample_id,
                    "selected_correct": correct,
                    "first_positive_rank": rank,
                    "candidate_count": 0 if sample_id == "s3" else 5,
                    "no_output": sample_id == "s3",
                }
            )
    return pd.DataFrame(rows)


def _full_pool_outcomes() -> pd.DataFrame:
    rows = []
    for route in ("g1", "c1"):
        for (
            sample_id,
            candidate_count_all,
            candidate_count_top5,
            full,
            top5,
            native,
            gated,
        ) in zip(
            ("s1", "s2", "s3"),
            (5, 6, 0),
            (5, 5, 0),
            (True, True, False),
            (True, False, False),
            (True, False, False),
            (True, True, False),
            strict=True,
        ):
            rows.append(
                {
                    "route": route,
                    "sample_id": sample_id,
                    "candidate_count_all": candidate_count_all,
                    "candidate_count_top5": candidate_count_top5,
                    "full_pool_positive": full,
                    "top5_positive": top5,
                    "first_positive_rank": 1 if top5 else 6 if full else None,
                    "native_correct": native,
                    "gated_correct": gated,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_replay_exactly_recomputes_integers_without_raw_gt() -> None:
    targets = {
        route: BaselineTarget(
            sample_count=3,
            native_correct=1,
            oracle_top5=1,
            oracle_all=2,
            no_output=1,
            no_positive_full_pool=0,
            positive_only_below_top5=1,
            final_correct=2,
        )
        for route in ("g1", "c1")
    }
    targets["d1"] = BaselineTarget(
        sample_count=3,
        native_correct=1,
        oracle_top5=2,
        oracle_top10=2,
        oracle_all=2,
        no_output=1,
        no_positive_full_pool=1,
        positive_only_below_top5=0,
        final_correct=2,
        top10_selected_correct=2,
        all_selected_correct=2,
        no_positive_includes_no_output=True,
    )
    result = replay_baselines(
        _unified_outcomes(),
        _d1_outcomes(),
        unified_full_pool=_full_pool_outcomes(),
        targets=targets,
    )
    assert result["status"] == "PASS"
    assert result["raw_test_ground_truth_rows_read"] == 0


def test_g1_c1_oracle_all_requires_locked_full_pool_summary() -> None:
    with pytest.raises(ValueError, match="Oracle@All"):
        replay_baselines(_unified_outcomes(), _d1_outcomes(), targets={})


def test_baseline_replay_rejects_raw_gt_columns() -> None:
    with pytest.raises(PermissionError, match="raw Test"):
        assert_no_raw_test_supervision(("sample_id", "gt_grasp_rectangles"))


def _bindings(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "bound.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    record = artifact_record(source)
    sample_manifest = tmp_path / "counterfactual_manifest.parquet"
    sample_manifest.write_bytes(b"synthetic-sample-manifest")
    registry = tmp_path / "gt_mask_registry.parquet"
    registry.write_bytes(b"synthetic-gt-mask-registry")
    mapping_qa = tmp_path / "FINAL_P2_MAPPING_PIXEL_QA.json"
    mapping_qa.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "stage": "P2_GT_MAPPING_PASS",
                "pixel_qa_status": "P2_MAPPING_QA_PASS",
                "mapping_qa_gt_mask_rows_read": 3,
                "candidate_generation_gt_mask_rows_read": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    routes = _declaration()["routes"]
    return {
        "source_locks": {"synthetic": record},
        "source_code": {"module": record},
        "configs": {"route": record},
        "baseline_replay": record,
        "sample_manifest": artifact_record(sample_manifest),
        "gt_grasp_source": artifact_record(sample_manifest),
        "gt_mask_registry": artifact_record(registry),
        "mapping_qa": artifact_record(mapping_qa),
        "route_contracts": inline_binding(routes),
        "resize_rules": inline_binding({"binary": "nearest"}),
        "evaluator": record,
        "taxonomy": inline_binding({"classes": ["T0", "T1"]}),
        "statistics": inline_binding({"bootstrap_seed": 20260813}),
        "case_selection": inline_binding({"rule": "synthetic"}),
    }


def _declaration() -> dict[str, object]:
    return {
        "branch": "gt_oracle",
        "gt_candidate_generation_authorized": True,
        "bulk_execution_max_count": 1,
        "mapping_qa_gt_mask_rows_read_before_lock": 3,
        "candidate_generation_gt_mask_rows_read_before_lock": 0,
        "routes": {
            "g1": {"allowed_gt_branches": ["gt_oracle"]},
            "c1": {"allowed_gt_branches": ["gt_oracle"]},
            "d1": {
                "allowed_gt_branches": ["gt_oracle"],
                "case": "B",
                "mask_affects_raw_sampling": True,
                "raw_candidate_regeneration_required": True,
                "filter_only_primary_allowed": False,
            },
        },
    }


def _advance_to_mapping(run: Path) -> None:
    transition_pipeline_status(
        run,
        RunState.P1_BASELINE_REPLAY_PASS,
        first_incomplete_stage=RunState.P2_GT_MAPPING_PASS.value,
    )
    transition_pipeline_status(
        run,
        RunState.P2_GT_MAPPING_PASS,
        first_incomplete_stage=RunState.P3_PROTOCOL_LOCKED.value,
    )


def test_protocol_lock_binds_closure_and_bulk_claim_is_exactly_once(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    bootstrap_run(run, source_verification=_verification(tmp_path))
    _advance_to_mapping(run)
    lock = create_protocol_lock(
        run,
        bindings=_bindings(tmp_path),
        declaration=_declaration(),
        test_only_allow_synthetic_contract=True,
    )
    verified = verify_protocol_lock(run)
    assert verified["counterfactual_execution_count"] == 0
    assert verified["mapping_qa_gt_mask_rows_read_before_lock"] == 3
    assert verified["candidate_generation_gt_mask_rows_read_before_lock"] == 0
    assert "raw_test_ground_truth_rows_read_before_lock" not in verified
    authority = load_execution_authority(lock)
    assert authority["gt_candidate_generation_authorized"] is True
    assert set(authority["routes"]) == {"g1", "c1", "d1"}
    assert authority["routes"]["d1"]["case"] == "B"
    assert authority["routes"]["d1"]["filter_only_primary_allowed"] is False
    with pytest.raises(FileExistsError):
        create_protocol_lock(
            run,
            bindings=_bindings(tmp_path),
            declaration=_declaration(),
            test_only_allow_synthetic_contract=True,
        )
    claim = claim_bulk_execution(run)
    assert json.loads(claim.read_text(encoding="utf-8"))["execution_count"] == 1
    with pytest.raises(PermissionError, match="already non-zero"):
        claim_bulk_execution(run)
    assert lock.is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("gt_candidate_generation_authorized", False, "authorize"),
        ("bulk_execution_max_count", 2, "maximum"),
        ("candidate_generation_gt_mask_rows_read_before_lock", 1, "before lock"),
    ),
)
def test_protocol_rejects_unsafe_execution_declaration(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    run = tmp_path / "run"
    bootstrap_run(run, source_verification=_verification(tmp_path))
    _advance_to_mapping(run)
    declaration = _declaration()
    declaration[field] = value
    with pytest.raises((PermissionError, ValueError), match=message):
        create_protocol_lock(
            run,
            bindings=_bindings(tmp_path),
            declaration=declaration,
            test_only_allow_synthetic_contract=True,
        )


def test_protocol_requires_all_routes_and_d1_case_b(tmp_path: Path) -> None:
    run = tmp_path / "run"
    bootstrap_run(run, source_verification=_verification(tmp_path))
    _advance_to_mapping(run)
    declaration = _declaration()
    routes = dict(declaration["routes"])
    routes.pop("c1")
    declaration["routes"] = routes
    with pytest.raises(ValueError, match="exactly g1"):
        create_protocol_lock(
            run,
            bindings=_bindings(tmp_path),
            declaration=declaration,
            test_only_allow_synthetic_contract=True,
        )
    declaration = _declaration()
    routes = dict(declaration["routes"])
    routes["d1"] = {**routes["d1"], "filter_only_primary_allowed": True}
    declaration["routes"] = routes
    with pytest.raises(ValueError, match="Case B"):
        create_protocol_lock(
            run,
            bindings=_bindings(tmp_path),
            declaration=declaration,
            test_only_allow_synthetic_contract=True,
        )


def test_protocol_requires_matching_final_p2_pixel_qa(tmp_path: Path) -> None:
    run = tmp_path / "run"
    bootstrap_run(run, source_verification=_verification(tmp_path))
    _advance_to_mapping(run)
    declaration = _declaration()
    declaration["mapping_qa_gt_mask_rows_read_before_lock"] = 7675
    with pytest.raises(ValueError, match="mapping pixel-QA"):
        create_protocol_lock(
            run,
            bindings=_bindings(tmp_path),
            declaration=declaration,
            test_only_allow_synthetic_contract=True,
        )
