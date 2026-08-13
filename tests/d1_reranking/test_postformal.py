from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from d1_reranking import (
    formal,
    postformal,
    postformal_evidence,
    postformal_sources,
    validation_evidence,
)
from tools.d1_reranking.independent_recompute import independent_recompute
from tools.d1_reranking import build_postformal as build_postformal_cli
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log


_FORMAL_TEST_PATH = Path(__file__).with_name("test_formal.py")
_SPEC = importlib.util.spec_from_file_location(
    "_d1_test_formal_fixture_postformal", _FORMAL_TEST_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE_MODULE)
_synthetic_run = _FIXTURE_MODULE._synthetic_run

_VALIDATION_TEST_PATH = Path(__file__).with_name("test_validation_evidence.py")
_VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "_d1_test_validation_evidence_fixture_postformal", _VALIDATION_TEST_PATH
)
assert _VALIDATION_SPEC is not None and _VALIDATION_SPEC.loader is not None
_VALIDATION_FIXTURE_MODULE = importlib.util.module_from_spec(_VALIDATION_SPEC)
_VALIDATION_SPEC.loader.exec_module(_VALIDATION_FIXTURE_MODULE)
_validation_fixture = _VALIDATION_FIXTURE_MODULE._fixture


def _write_content(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def stable_repository_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formal,
        "_repository_state",
        lambda: {
            "git_commit": "a" * 40,
            "dirty": False,
            "status_sha256": "b" * 64,
            "tracked_diff_sha256": "c" * 64,
        },
    )


def _evidence_sources(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], Path, dict[str, Path]]:
    _validation_fixture(root, monkeypatch)
    validation = validation_evidence.assemble_validation_evidence_tables(root)
    tables = {
        name: Path(record["path"])
        for name, record in validation["artifacts"]["tables"].items()
    }
    covariates = root / postformal_sources.COVARIATES_RELATIVE_PATH
    covariates.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": ["sample-0", "sample-1"],
            "target_size": ["small", "large"],
            "relation_query": [True, False],
            "clutter": [False, True],
            "depth_missing": [False, False],
            "predicted_mask_confidence": [0.7, 0.8],
            "native_mask_support": [0.4, 0.5],
            "selected_mask_support": [0.6, 0.5],
        }
    ).to_parquet(covariates, index=False)
    runtime: dict[str, Path] = {}
    for component in postformal_evidence.RUNTIME_COMPONENTS:
        telemetry: dict[str, object] = {
            "measurement_semantics": (
                "feature_extraction_not_artifact_loading"
                if component == "feature_extraction"
                else "wall_clock_per_sample"
            ),
            "latency_ms_per_sample": 1.0,
            "peak_memory_mb": 2.0,
        }
        if component == "ranker_inference":
            telemetry.update({"parameter_count": 10, "model_bytes": 80})
        path = _write_content(
            root / postformal_sources.OUTPUT_ROOT / "runtime" / f"{component}.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "component": component,
                "candidate_test_labels_read": False,
                "telemetry": telemetry,
            },
        )
        runtime[component] = path
    contribution = _write_content(
        root / postformal_sources.CONTRIBUTIONS_RELATIVE_PATH,
        {
            "schema_version": 1,
            "status": "NOT_APPLICABLE",
            "selected_method": "R3",
            "reason": "selected ranker R3 is not LightGBM R5",
            "candidate_test_labels_read": False,
        },
    )
    sources = {"run_manifest": formal._record(root / "manifest.json")}
    source_manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "sample_count": 2,
        "covariate_schema": list(postformal_sources.COVARIATE_COLUMNS),
        "fixed_source_paths": postformal_sources.FIXED_INPUTS,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": {
            "sample_covariates": formal._record(covariates),
            "runtime_manifests": {
                name: formal._record(path) for name, path in runtime.items()
            },
            "ranker_contributions": formal._record(contribution),
        },
    }
    _write_content(
        root / postformal_sources.SOURCE_MANIFEST_RELATIVE_PATH,
        source_manifest,
    )
    return tables, covariates, runtime


def _bind_evidence(
    root: Path,
    plan_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    closure_path = Path(plan["sources"]["source_closure"]["path"])
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        postformal_evidence,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )
    monkeypatch.setattr(
        postformal,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )
    source_records = {
        name: record
        for name, record in {
            "opaque_ground_truth": closure["canonical_inputs"]["test"][
                "opaque_ground_truth"
            ],
            "opaque_visual_ground_truth": closure["canonical_inputs"]["test"][
                "opaque_visual_ground_truth"
            ],
        }.items()
    }
    source_records["source_child"] = closure["verified_artifacts"][0]
    (root / "00_audit").mkdir(parents=True, exist_ok=True)
    (root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_BEFORE.json").write_text(
        json.dumps(
            {"status": "PASS", "source_records": source_records}, sort_keys=True
        ),
        encoding="utf-8",
    )
    append_access_log(
        root,
        {
            "event_id": "synthetic-source-closure-hash-only",
            "event": "d1_test_ground_truth_hash_only_source_closure",
            "candidate_labels_opened_as_table": False,
            "candidate_test_labels_read": False,
        },
    )
    _evidence_sources(root, monkeypatch)
    postformal_evidence.assemble_postformal_evidence(
        root,
        q_saturation_threshold=0.85,
        mask_quality_threshold=0.5,
    )
    evidence_binding = postformal_evidence.validate_postformal_evidence_prelock(
        root,
        source_closure=closure,
    )
    prelock_path = Path(plan["components"]["prelock_readiness"]["path"])
    prelock = json.loads(prelock_path.read_text(encoding="utf-8"))
    prelock.pop("content_sha256", None)
    prelock["sources"]["postformal_evidence"] = evidence_binding
    prelock["sources"]["access_log"] = formal._record(
        root / "09_formal_test" / "test_access.log"
    )
    _write_content(prelock_path, prelock)
    plan.pop("content_sha256", None)
    plan["components"]["prelock_readiness"] = formal._record(prelock_path)
    plan["components"]["postformal_evidence"] = formal._record(
        root / postformal_evidence.EVIDENCE_RELATIVE_PATH
    )
    plan["sources"]["components"] = plan["components"]
    plan["sources"]["source_closure"] = formal._record(closure_path)
    plan["source_signature_sha256"] = canonical_sha256(plan["sources"])
    _write_content(plan_path, plan)
    formal._append_hash_only_event(
        root,
        ground_truth_record=plan["raw_test_ground_truth"],
        plan_record=formal._record(plan_path),
    )


def _ledger_row(root: Path, stage: str, substage: str, artifact: Path) -> None:
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage=stage,
        substage=substage,
        route="D1",
        evidence_track="synthetic",
        pool="synthetic",
        method="synthetic_contract_test",
        command=f"synthetic {stage} {substage}",
    ) as state:
        state["artifact_path"] = str(artifact.resolve())
        state["artifact_sha256"] = sha256_file(artifact)


def _completed_postformal(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    plan = _synthetic_run(root)
    _bind_evidence(root, plan, monkeypatch)
    formal.create_formal_lock(root, evaluation_plan_path=plan)
    formal.run_formal_test_once(root)
    independent_recompute(root)
    for stage, substage, artifact in (
        ("P14", "d1_formal_evaluation_plan", plan),
        ("P14", "d1_formal_test_lock", root / "08_lock" / formal.LOCK_NAME),
        (
            "P14",
            "d1_exactly_once_formal_test",
            root / "09_formal_test" / formal.EXECUTION_NAME,
        ),
        (
            "P17",
            "d1_independent_recompute",
            root / "17_independent_recompute" / "recomputed_metrics.json",
        ),
    ):
        _ledger_row(root, stage, substage, artifact)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P15",
        substage="d1_postformal_artifacts",
        route="D1",
        evidence_track="synthetic",
        pool="synthetic",
        method="synthetic_contract_test",
        command="synthetic P15 d1_postformal_artifacts",
    ) as state:
        result = postformal.build_postformal_artifacts(root)
        artifact = root / postformal.POSTFORMAL_MANIFEST_RELATIVE_PATH
        state["artifact_path"] = str(artifact.resolve())
        state["artifact_sha256"] = sha256_file(artifact)
    return result


def test_postformal_evidence_to_lock_boards_statistics_and_final_lock(
    tmp_path: Path,
    stable_repository_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stable_repository_state
    result = _completed_postformal(tmp_path, monkeypatch)
    assert result["status"] == "COMPLETE"
    statistics = pd.read_csv(tmp_path / "tables" / "d1_formal_statistics.csv")
    assert len(statistics) == len(formal.PRIMARY_COMPARISONS)
    assert statistics["mcnemar_exact_p_display"].astype(str).str.len().gt(0).all()
    runtime = pd.read_csv(tmp_path / "tables" / "d1_runtime_complexity.csv")
    assert set(runtime["component"]) == set(postformal_evidence.RUNTIME_COMPONENTS)
    board = next((tmp_path / "14_case_selection" / "boards").rglob("*_case_board.png"))
    assert board.is_file()
    qa = json.loads(
        (tmp_path / "14_case_selection" / "CASE_BOARD_QA.json").read_text(
            encoding="utf-8"
        )
    )
    assert qa["status"] == "PASS"
    assert qa["black_image_fallback_used"] is False
    assert qa["required_panel_inventory"] == list("ABCDEFGHIJKL")
    assert qa["canonical_evaluator_gt_conversion"] is True
    assert all(row["required_panels"] == list("ABCDEFGHIJKL") for row in qa["boards"])
    assert all(row["language_prompt"] for row in qa["boards"])
    assert all(
        {
            "visual_grounding",
            "candidate_generation_count",
            "native_gq_rank",
            "learned_reranker",
            "gate_utility",
            "earliest_issue",
            "lightgbm_contribution_diff",
        }
        == set(row["module_diagnosis"])
        for row in qa["boards"]
    )
    assert any(
        candidate["display_score"] is None
        for row in qa["boards"]
        for candidate in row["allnms_candidates"]
        if candidate["native_rank"] > 5
    )

    with pytest.raises(RuntimeError, match="synthetic transient"):
        with ledger_stage(
            tmp_path / "run_ledger.sqlite",
            stage="PFINAL",
            substage="d1_finalization_preflight",
            route="D1",
            evidence_track="synthetic",
            pool="synthetic",
            method="synthetic_transient_failure",
            command="synthetic PFINAL transient",
        ):
            raise RuntimeError("synthetic transient")
    _ledger_row(
        tmp_path,
        "PFINAL",
        "d1_finalization_preflight",
        tmp_path / postformal.POSTFORMAL_MANIFEST_RELATIVE_PATH,
    )
    authority = Path(
        json.loads(
            (tmp_path / formal.FORMAL_PLAN_RELATIVE_PATH).read_text(encoding="utf-8")
        )["components"]["source_run_authority"]["path"]
    )
    monkeypatch.setattr(
        postformal,
        "EXPECTED_UNIFIED_FINAL_LOCK_SHA256",
        sha256_file(authority),
    )
    finalized = postformal.finalize_d1_run(tmp_path)
    assert finalized["status"] == "COMPLETE"
    final_lock = postformal.verify_final_lock(tmp_path)
    assert final_lock["integrity_checks"]["ledger_commands"]["failed_count"] == 1
    assert (
        final_lock["integrity_checks"]["ledger_commands"]["superseded_failed_count"]
        == 1
    )

    added = tmp_path / "unexpected_after_lock.txt"
    added.write_text("tamper", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fresh exact inventory"):
        postformal.verify_final_lock(tmp_path)


def test_postformal_rejects_metric_corruption(
    tmp_path: Path,
    stable_repository_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stable_repository_state
    plan = _synthetic_run(tmp_path)
    _bind_evidence(tmp_path, plan, monkeypatch)
    formal.create_formal_lock(tmp_path, evaluation_plan_path=plan)
    formal.run_formal_test_once(tmp_path)
    independent_recompute(tmp_path)
    metrics_path = tmp_path / "09_formal_test" / "formal_test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["systems"]["d1_top5_r0"]["rank_metrics"][0]["mrr_at_k"] = 0.25
    metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="record differs"):
        postformal.build_postformal_artifacts(tmp_path)


def test_evidence_writer_rejects_missing_runtime_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _synthetic_run(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    closure_path = Path(payload["sources"]["source_closure"]["path"])
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        postformal_evidence,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )
    _evidence_sources(tmp_path, monkeypatch)
    source_path = tmp_path / postformal_sources.SOURCE_MANIFEST_RELATIVE_PATH
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.pop("content_sha256")
    source["artifacts"]["runtime_manifests"].pop("gate_inference")
    _write_content(source_path, source)
    with pytest.raises(RuntimeError, match="runtime evidence component inventory"):
        postformal_evidence.assemble_postformal_evidence(
            tmp_path,
            q_saturation_threshold=0.85,
            mask_quality_threshold=0.5,
        )


@pytest.mark.parametrize(
    "forbidden_column",
    ("three_route_correct", "oracle", "mask_iou", "renamed_success"),
)
def test_evidence_writer_rejects_renamed_outcome_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_column: str,
) -> None:
    plan = _synthetic_run(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    closure_path = Path(payload["sources"]["source_closure"]["path"])
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        postformal_evidence,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )
    _, covariates_path, _ = _evidence_sources(tmp_path, monkeypatch)
    covariates = pd.read_parquet(covariates_path)
    covariates[forbidden_column] = 0.5
    covariates.to_parquet(covariates_path, index=False)
    source_path = tmp_path / postformal_sources.SOURCE_MANIFEST_RELATIVE_PATH
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.pop("content_sha256")
    source["artifacts"]["sample_covariates"] = formal._record(covariates_path)
    _write_content(source_path, source)
    with pytest.raises(PermissionError, match="outcome/GT-derived"):
        postformal_evidence.assemble_postformal_evidence(
            tmp_path,
            q_saturation_threshold=0.85,
            mask_quality_threshold=0.5,
        )


def test_evidence_writer_hashes_but_does_not_open_visual_gt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _synthetic_run(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    closure_path = Path(payload["sources"]["source_closure"]["path"])
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    visual_path = Path(
        closure["canonical_inputs"]["test"]["opaque_visual_ground_truth"]["path"]
    ).resolve()
    monkeypatch.setattr(
        postformal_evidence,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )
    original = postformal_evidence.pd.read_parquet

    def guarded_read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if Path(path).resolve() == visual_path:
            raise AssertionError("opaque visual GT rows opened prelock")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(postformal_evidence.pd, "read_parquet", guarded_read)
    _evidence_sources(tmp_path, monkeypatch)
    result = postformal_evidence.assemble_postformal_evidence(
        tmp_path,
        q_saturation_threshold=0.85,
        mask_quality_threshold=0.5,
    )
    assert result["opaque_visual_ground_truth_opened_as_table"] is False


def test_postformal_cli_guard_precedes_ledger_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _synthetic_run(tmp_path)
    monkeypatch.setattr(
        build_postformal_cli,
        "parse_args",
        lambda: type("Args", (), {"run_dir": tmp_path})(),
    )
    ledger = tmp_path / "run_ledger.sqlite"
    assert not ledger.exists()
    with pytest.raises((FileNotFoundError, RuntimeError, PermissionError)):
        build_postformal_cli.main()
    assert not ledger.exists()


def test_ledger_check_rejects_current_unsuperseded_failure(
    tmp_path: Path,
    stable_repository_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stable_repository_state
    _completed_postformal(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="current failure"):
        with ledger_stage(
            tmp_path / "run_ledger.sqlite",
            stage="PFINAL",
            substage="d1_finalization_preflight",
            route="D1",
            evidence_track="synthetic",
            pool="synthetic",
            method="synthetic_current_failure",
            command="synthetic PFINAL failed",
        ):
            raise RuntimeError("current failure")
    check = postformal._ledger_check(tmp_path)
    assert check["status"] == "FAIL"
    assert check["unresolved_failed"]


def test_source_immutability_rejects_child_mutation(
    tmp_path: Path,
    stable_repository_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del stable_repository_state
    _completed_postformal(tmp_path, monkeypatch)
    child = tmp_path / "source_child.bin"
    child.write_bytes(b"mutated after formal completion")
    with pytest.raises(RuntimeError, match="record differs"):
        postformal._source_immutability_after(
            tmp_path, postformal._load_formal(tmp_path)
        )


@pytest.mark.parametrize("mutation", ["missing_required", "statistics_corruption"])
def test_finalizer_fails_closed_on_missing_or_corrupt_artifact(
    tmp_path: Path,
    stable_repository_state: None,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    del stable_repository_state
    _completed_postformal(tmp_path, monkeypatch)
    if mutation == "missing_required":
        (tmp_path / "tables" / "d1_runtime_complexity.csv").unlink()
    else:
        cochran = tmp_path / "10_statistics" / "cochran_q.json"
        value = json.loads(cochran.read_text(encoding="utf-8"))
        value["statistic"] = 999.0
        cochran.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    _ledger_row(
        tmp_path,
        "PFINAL",
        "d1_finalization_preflight",
        tmp_path / postformal.POSTFORMAL_MANIFEST_RELATIVE_PATH,
    )
    authority = Path(
        json.loads(
            (tmp_path / formal.FORMAL_PLAN_RELATIVE_PATH).read_text(encoding="utf-8")
        )["components"]["source_run_authority"]["path"]
    )
    monkeypatch.setattr(
        postformal,
        "EXPECTED_UNIFIED_FINAL_LOCK_SHA256",
        sha256_file(authority),
    )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        postformal.finalize_d1_run(tmp_path)
    assert not (tmp_path / "COMPLETE").exists()
    assert not (tmp_path / postformal.FINAL_LOCK_NAME).exists()
    assert (tmp_path / "FINALIZATION_FAILED.json").is_file()
