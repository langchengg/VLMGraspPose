"""D1 P15 postformal analysis and fail-closed final-run lock assembly."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.hashing import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.ledger import render_ledger_commands
from unified_reranking.test_access_guard import append_access_log

from .contracts import EXPECTED_UNIFIED_FINAL_LOCK_SHA256
from .execution import load_content_manifest
from .formal import (
    EXECUTION_NAME,
    FORMAL_SYSTEMS,
    LOCK_NAME,
    REFERENCE_SYSTEMS,
    verify_formal_lock,
)
from .run import transition_pipeline_status
from .postformal_evidence import (
    EVIDENCE_RELATIVE_PATH,
    EVIDENCE_TABLE_NAMES,
    RUNTIME_COMPONENTS,
)
from .postformal_sources import load_postformal_sources
from .provenance import load_source_closure


POSTFORMAL_MANIFEST_RELATIVE_PATH = "16_reports/D1_POSTFORMAL_MANIFEST.json"
FINAL_LOCK_NAME = "FINAL_RUN_LOCK.json"
FINAL_LOCK_DIGEST_NAME = "FINAL_RUN_LOCK.sha256"
FAILURE_CATEGORIES = (
    "E0",
    "E1",
    "E2",
    "E2b",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
    "E10",
)
TABLE_NAMES = (
    "d1_provenance_comparison.csv",
    "d1_r0_r7_validation.csv",
    "d1_formal_test_primary.csv",
    "d1_k_sensitivity.csv",
    "d1_evidence_tracks.csv",
    "d1_feature_ablation.csv",
    "d1_gate_transitions.csv",
    "d1_failure_decomposition.csv",
    "four_route_router.csv",
    "top20_union.csv",
    "complete_four_route_comparison.csv",
)
ADDITIONAL_TABLE_NAMES = (
    "d1_q_objective_mismatch.csv",
    "d1_q_bins_success.csv",
    "d1_stratified_results.csv",
    "d1_case_selection.csv",
    "d1_formal_statistics.csv",
    "d1_runtime_complexity.csv",
)
FIGURE_NAMES = (
    "d1_native_reranked_oracle.pdf",
    "d1_r0_r7_validation.pdf",
    "d1_headroom_by_k.pdf",
    "d1_first_positive_rank.pdf",
    "d1_recovered_harmful.pdf",
    "d1_q_saturation.pdf",
    "d1_feature_importance.pdf",
    "d1_gate_risk_coverage.pdf",
    "four_route_router_union.pdf",
    "d1_failure_funnel.pdf",
)
REPORT_NAMES = (
    "FINAL_D1_RERANKING_REPORT_EN.md",
    "FINAL_D1_RERANKING_SUMMARY_ZH.md",
    "D1_EXPERIMENT_CONCLUSION.json",
    "THESIS_READY_D1_METHODS.tex",
    "THESIS_READY_D1_RESULTS_DISCUSSION.tex",
    "THESIS_READY_TABLES.tex",
    "THESIS_READY_FIGURE_CAPTIONS.md",
    "D1_LIMITATIONS_AND_CLAIMS.md",
    "README_REPRODUCE.md",
)
CASE_CATEGORIES = (
    "d1_recovered",
    "d1_harmful",
    "gate_prevented_harmful",
    "gate_missed_recoverable",
    "positive_rank_2_5",
    "positive_rank_6_10",
    "no_positive_full_pool",
    "correct_grounding_ranking_failure",
    "low_quality_grounding_association",
    "q_saturation_failure",
    "four_route_router_uniquely_recovered_by_d1",
    "top20_union_uniquely_selected_d1",
)


def assert_writable_postformal(run_dir: str | Path) -> None:
    """Fail before ledger mutation unless P14/P17 are complete and P15 is writable."""

    root = Path(run_dir).expanduser().resolve()
    forbidden = (
        root / POSTFORMAL_MANIFEST_RELATIVE_PATH,
        root / FINAL_LOCK_NAME,
        root / FINAL_LOCK_DIGEST_NAME,
        root / "COMPLETE",
        root / "FINALIZATION_FAILED.json",
    )
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise PermissionError(
            f"D1 postformal output is immutable/already present: {present}"
        )
    verify_formal_lock(root)
    execution = _load_content(
        root / "09_formal_test" / EXECUTION_NAME,
        name="D1 postformal lifecycle formal execution",
        statuses=("COMPLETE",),
    )
    independent = _load_content(
        root / "17_independent_recompute" / "recomputed_metrics.json",
        name="D1 postformal lifecycle independent recompute",
        statuses=("PASS",),
    )
    pipeline = _read_json(root / "pipeline_status.json", name="D1 pipeline status")
    if (
        execution.get("execution_count") != 1
        or independent.get("formal_execution_count") != 1
        or independent.get("formal_metrics_match") is not True
        or pipeline.get("status") != "FORMAL_EXECUTED"
        or pipeline.get("formal_test_executed") is not True
        or pipeline.get("test_candidate_labels_read") is not True
        or int(pipeline.get("formal_test_execution_count", -1)) != 1
    ):
        raise PermissionError("D1 postformal lifecycle is not P14 COMPLETE + P17 PASS")


def assert_writable_finalization(run_dir: str | Path) -> None:
    """Reject terminal/replayed finalization before its preflight ledger row."""

    root = Path(run_dir).expanduser().resolve()
    forbidden = (
        root / FINAL_LOCK_NAME,
        root / FINAL_LOCK_DIGEST_NAME,
        root / "COMPLETE",
        root / "FINALIZATION_FAILED.json",
    )
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise PermissionError(
            f"D1 finalization is terminal/already consumed: {present}"
        )
    postformal_path = root / POSTFORMAL_MANIFEST_RELATIVE_PATH
    _load_content(
        postformal_path,
        name="D1 finalization lifecycle postformal manifest",
        statuses=("COMPLETE",),
    )
    pipeline = _read_json(root / "pipeline_status.json", name="D1 pipeline status")
    if (
        pipeline.get("status") != "POSTFORMAL"
        or pipeline.get("formal_test_executed") is not True
        or pipeline.get("test_candidate_labels_read") is not True
        or int(pipeline.get("formal_test_execution_count", -1)) != 1
    ):
        raise PermissionError("D1 finalization lifecycle is not POSTFORMAL")


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 postformal artifact is not a regular file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _same_record(observed: object, expected: Mapping[str, Any], *, name: str) -> None:
    if not isinstance(observed, Mapping):
        raise RuntimeError(f"{name} record is absent")
    if observed.get("path") != expected.get("path") or observed.get(
        "sha256"
    ) != expected.get("sha256"):
        raise RuntimeError(f"{name} record differs")
    if "bytes" in observed and observed.get("bytes") != expected.get("bytes"):
        raise RuntimeError(f"{name} byte count differs")


def _verified_path(record: object, *, name: str) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{name} record is absent")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    _same_record(record, _record(path), name=name)
    return path


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} is not a JSON object")
    return value


def _load_content(
    path: str | Path, *, name: str, statuses: tuple[str, ...]
) -> dict[str, Any]:
    return load_content_manifest(path, name=name, statuses=statuses)


def _record_in_inventory(
    record: Mapping[str, Any], inventory: Mapping[str, Any]
) -> bool:
    return any(
        isinstance(value, Mapping)
        and value.get("path") == record.get("path")
        and value.get("sha256") == record.get("sha256")
        for value in inventory.values()
    )


def _artifact_records(value: object) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            records.append(dict(value))
        for child in value.values():
            records.extend(_artifact_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_artifact_records(child))
    return records


def _source_immutability_after(root: Path, formal: Mapping[str, Any]) -> dict[str, Any]:
    """Freshly re-hash the semantic P1 closure and persist the final source audit."""

    before_path = root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_BEFORE.json"
    before = _read_json(before_path, name="D1 source immutability BEFORE")
    before_sources = before.get("source_records")
    if before.get("status") != "PASS" or not isinstance(before_sources, Mapping):
        raise RuntimeError("D1 source immutability BEFORE contract differs")
    before_records = _artifact_records(before_sources)
    if not before_records:
        raise RuntimeError("D1 source immutability BEFORE source inventory is empty")
    for index, record in enumerate(before_records):
        _verified_path(record, name=f"D1 source immutability BEFORE record {index}")

    closure_path, closure = load_source_closure(root)
    bound_record = formal.get("plan", {}).get("sources", {}).get("source_closure")
    _same_record(
        bound_record, _record(closure_path), name="D1 P14-bound source closure"
    )
    closure_records = _artifact_records(closure)
    if not closure_records:
        raise RuntimeError("D1 semantic source closure artifact inventory is empty")
    current_by_path: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(closure_records):
        path = _verified_path(record, name=f"D1 semantic source record {index}")
        current_by_path[str(path)] = _record(path)
    missing_before = [
        record
        for record in before_records
        if str(Path(str(record["path"])).expanduser().resolve()) not in current_by_path
    ]
    if missing_before:
        raise RuntimeError(
            "D1 BEFORE sources are not contained in the semantic closure"
        )
    for record in before_records:
        current = current_by_path[str(Path(str(record["path"])).expanduser().resolve())]
        _same_record(record, current, name="D1 BEFORE/closure source")

    after_path = root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_AFTER.json"
    if after_path.exists():
        raise FileExistsError(
            f"D1 source immutability AFTER already exists: {after_path}"
        )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "comparison": "BEFORE == P14-bound semantic closure == freshly re-hashed AFTER",
        "before": _record(before_path),
        "source_closure": _record(closure_path),
        "closure_id": closure.get("closure_id"),
        "before_source_count": len(before_records),
        "closure_source_count": len(current_by_path),
        "source_records": dict(sorted(current_by_path.items())),
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(after_path, result)
    return {"status": "PASS", "artifact": _record(after_path)}


def _load_independent(root: Path) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    path = root / "17_independent_recompute" / "recomputed_metrics.json"
    value = _load_content(path, name="D1 independent recompute", statuses=("PASS",))
    self_unsigned = dict(value)
    self_unsigned.pop("content_sha256", None)
    recorded_self = self_unsigned.pop("self_sha256", None)
    if recorded_self != canonical_sha256(self_unsigned):
        raise RuntimeError("D1 independent recompute self hash differs")
    per_sample_path = _verified_path(
        value.get("artifacts", {}).get("independent_per_sample"),
        name="D1 independent per-sample input",
    )
    per_sample = pd.read_parquet(per_sample_path)
    if (
        value.get("formal_execution_count") != 1
        or value.get("formal_metrics_match") is not True
        or value.get("candidate_evaluator_integrity") != "PASS"
        or set(value.get("system_names", [])) != set(FORMAL_SYSTEMS)
        or set(per_sample["system_name"].astype(str)) != set(FORMAL_SYSTEMS)
    ):
        raise RuntimeError("D1 independent recompute contract differs")
    return path, value, per_sample


def _load_formal(root: Path) -> dict[str, Any]:
    lock = verify_formal_lock(root)
    execution = _load_content(
        root / "09_formal_test" / EXECUTION_NAME,
        name="D1 formal execution",
        statuses=("COMPLETE",),
    )
    manifest = _load_content(
        root / "09_formal_test" / "formal_test_manifest.json",
        name="D1 formal manifest",
        statuses=("COMPLETE",),
    )
    plan_path = _verified_path(lock.get("evaluation_plan"), name="D1 formal plan")
    plan = _load_content(plan_path, name="D1 formal plan", statuses=("LOCK_READY",))
    if (
        execution.get("execution_count") != 1
        or manifest.get("formal_test_execution_count") != 1
        or set(manifest.get("system_names", [])) != set(FORMAL_SYSTEMS)
        or set(lock.get("system_names", [])) != set(FORMAL_SYSTEMS)
    ):
        raise RuntimeError("D1 formal completion/system contract differs")
    bundle_path = _verified_path(
        manifest.get("artifacts", {}).get("candidate_score_decision_bundle"),
        name="D1 formal bundle",
    )
    outcomes_path = _verified_path(
        manifest.get("artifacts", {}).get("candidate_outcomes"),
        name="D1 formal candidate outcomes",
    )
    metrics_path = _verified_path(
        manifest.get("artifacts", {}).get("metrics"), name="D1 formal metrics"
    )
    bundle = pd.read_parquet(bundle_path)
    outcomes = pd.read_parquet(outcomes_path)
    metrics = _read_json(metrics_path, name="D1 formal metrics")
    if set(bundle["system_name"].astype(str)) != set(FORMAL_SYSTEMS):
        raise RuntimeError("D1 formal bundle system coverage differs")
    return {
        "lock": lock,
        "execution": execution,
        "manifest": manifest,
        "plan_path": plan_path,
        "plan": plan,
        "bundle_path": bundle_path,
        "bundle": bundle,
        "outcomes_path": outcomes_path,
        "outcomes": outcomes,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def _load_evaluator(record: Mapping[str, Any]) -> Any:
    path = _verified_path(record, name="D1 canonical evaluator")
    name = f"_d1_postformal_evaluator_{record['sha256'][:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load D1 canonical evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for primitive in (
        "gt_from_corners",
        "corners",
        "IOU_THRESHOLD",
        "ANGLE_THRESHOLD_DEG",
    ):
        if not hasattr(module, primitive):
            raise RuntimeError(
                f"D1 canonical evaluator misses visual primitive: {primitive}"
            )
    return module


def _load_evaluator_thresholds(record: Mapping[str, Any]) -> tuple[float, float]:
    module = _load_evaluator(record)
    return float(module.IOU_THRESHOLD), float(module.ANGLE_THRESHOLD_DEG)


def _load_evidence(
    root: Path, lock: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    path = root / EVIDENCE_RELATIVE_PATH
    evidence = _load_content(path, name="D1 postformal evidence", statuses=("LOCKED",))
    if (
        evidence.get("declared_before_formal_lock") is not True
        or evidence.get("declared_before_prelock") is not True
        or evidence.get("candidate_test_labels_read") is not False
        or evidence.get("opaque_visual_ground_truth_opened_as_table") is not False
        or not isinstance(evidence.get("q_saturation_threshold"), (float, int))
        or not isinstance(evidence.get("mask_quality_threshold"), (float, int))
    ):
        raise RuntimeError("D1 postformal evidence declaration differs")
    evidence_sources = evidence.get("sources")
    if not isinstance(evidence_sources, Mapping) or evidence.get(
        "source_signature_sha256"
    ) != canonical_sha256(evidence_sources):
        raise RuntimeError("D1 postformal evidence source signature differs")
    inventory = lock.get("inventory")
    if not isinstance(inventory, Mapping) or not _record_in_inventory(
        _record(path), inventory
    ):
        raise RuntimeError("D1 postformal evidence declaration was not formal-locked")
    table_records = evidence.get("tables")
    if not isinstance(table_records, Mapping) or set(table_records) != set(
        EVIDENCE_TABLE_NAMES
    ):
        raise RuntimeError("D1 postformal evidence table inventory differs")
    for name, record in table_records.items():
        source = _verified_path(record, name=f"D1 evidence table {name}")
        if not _record_in_inventory(_record(source), inventory):
            raise RuntimeError(f"D1 evidence table was not formal-locked: {name}")
        frame = pd.read_csv(source)
        if frame.empty:
            raise RuntimeError(f"D1 evidence table is empty: {name}")
    covariates_path = _verified_path(
        evidence.get("sample_covariates"), name="D1 sample covariates"
    )
    if not _record_in_inventory(_record(covariates_path), inventory):
        raise RuntimeError("D1 sample covariates were not formal-locked")
    covariates = pd.read_parquet(covariates_path)
    required = {
        "sample_id",
        "target_size",
        "relation_query",
        "clutter",
        "depth_missing",
        "predicted_mask_confidence",
        "native_mask_support",
        "selected_mask_support",
    }
    if (
        required.difference(covariates.columns)
        or covariates["sample_id"].astype(str).duplicated().any()
    ):
        raise RuntimeError("D1 sample covariate schema/identity differs")
    runtime_records = evidence.get("runtime_manifests")
    if not isinstance(runtime_records, Mapping) or set(runtime_records) != set(
        RUNTIME_COMPONENTS
    ):
        raise RuntimeError("D1 runtime evidence inventory differs")
    for component, record in runtime_records.items():
        source = _verified_path(record, name=f"D1 runtime source {component}")
        if not _record_in_inventory(_record(source), inventory):
            raise RuntimeError(f"D1 runtime source was not formal-locked: {component}")
    visual_record = evidence.get("opaque_visual_ground_truth")
    visual_path = _verified_path(visual_record, name="D1 opaque visual ground truth")
    if not _record_in_inventory(_record(visual_path), inventory):
        raise RuntimeError("D1 opaque visual ground truth was not formal-locked")
    closure_path = _verified_path(
        evidence_sources.get("source_closure"),
        name="D1 postformal source closure",
    )
    closure = _load_content(closure_path, name="D1 source closure", statuses=("PASS",))
    expected_visual = (
        closure.get("canonical_inputs", {})
        .get("test", {})
        .get("opaque_visual_ground_truth")
    )
    _same_record(
        expected_visual,
        _record(visual_path),
        name="D1 source-closure visual ground truth",
    )
    return evidence, covariates


def _decision_table(bundle: pd.DataFrame, system: str) -> pd.DataFrame:
    rows = bundle.loc[bundle["system_name"].eq(system)].copy()
    decisions = rows.groupby("sample_id", as_index=False).agg(
        selected_source_route=("selected_source_route", "first"),
        selected_candidate_id=("selected_candidate_id", "first"),
        selected_correct=("selected_correct", "max"),
        no_output=("no_output", "max"),
    )
    return decisions


def classify_failure_funnel(
    formal: Mapping[str, Any], covariates: pd.DataFrame
) -> pd.DataFrame:
    bundle = formal["bundle"]
    outcomes = formal["outcomes"].copy()
    outcomes["sample_id"] = outcomes["sample_id"].astype(str)
    positive = outcomes.loc[
        outcomes["source_route"].eq("D1") & outcomes["candidate_success"].astype(bool)
    ]
    first_positive = positive.groupby("sample_id")["native_rank"].min()
    candidate_count = (
        outcomes.loc[outcomes["source_route"].eq("D1")].groupby("sample_id").size()
    )
    native_q = outcomes.loc[
        outcomes["source_route"].eq("D1") & outcomes["native_rank"].eq(1)
    ].set_index("sample_id")["native_score"]
    native = _decision_table(bundle, "d1_top5_r0")
    ungated = _decision_table(bundle, "d1_top5_r7_ungated")
    gated = _decision_table(bundle, "d1_top5_r7_gated")
    allnms = _decision_table(bundle, "d1_allnms_locked")
    result = (
        native.rename(
            columns={
                "selected_source_route": "native_route",
                "selected_candidate_id": "native_candidate_id",
                "selected_correct": "native_correct",
                "no_output": "native_no_output",
            }
        )
        .merge(
            ungated.rename(
                columns={
                    "selected_source_route": "ungated_route",
                    "selected_candidate_id": "ungated_candidate_id",
                    "selected_correct": "ungated_correct",
                    "no_output": "ungated_no_output",
                }
            ),
            on="sample_id",
            validate="one_to_one",
        )
        .merge(
            gated.rename(
                columns={
                    "selected_source_route": "gated_route",
                    "selected_candidate_id": "gated_candidate_id",
                    "selected_correct": "gated_correct",
                    "no_output": "gated_no_output",
                }
            ),
            on="sample_id",
            validate="one_to_one",
        )
        .merge(
            allnms[["sample_id", "no_output"]].rename(
                columns={"no_output": "allnms_no_output"}
            ),
            on="sample_id",
            validate="one_to_one",
        )
    )
    result["first_positive_native_rank"] = result["sample_id"].map(first_positive)
    result["candidate_count"] = (
        result["sample_id"].map(candidate_count).fillna(0).astype(int)
    )
    result["native_q"] = result["sample_id"].map(native_q)
    result = result.merge(covariates, on="sample_id", validate="one_to_one")
    result["gated_switched"] = (
        result["gated_route"].astype(str)
        + "\0"
        + result["gated_candidate_id"].astype(str)
    ) != (
        result["native_route"].astype(str)
        + "\0"
        + result["native_candidate_id"].astype(str)
    )

    def category(row: Any) -> str:
        rank = (
            float(row.first_positive_native_rank)
            if pd.notna(row.first_positive_native_rank)
            else np.inf
        )
        if bool(row.allnms_no_output) or bool(row.gated_no_output):
            return "E0"
        if not np.isfinite(rank):
            return "E1"
        if rank > 10:
            return "E2"
        if rank > 5:
            return "E2b"
        if (
            bool(row.native_correct)
            and not bool(row.ungated_correct)
            and bool(row.gated_correct)
        ):
            return "E10"
        if (
            not bool(row.native_correct)
            and bool(row.ungated_correct)
            and not bool(row.gated_correct)
        ):
            return "E9"
        if not bool(row.native_correct) and bool(row.gated_correct):
            return "E5"
        if bool(row.native_correct) and not bool(row.gated_correct):
            return "E6"
        if (
            bool(row.gated_switched)
            and not bool(row.native_correct)
            and not bool(row.gated_correct)
        ):
            return "E7"
        if (
            bool(row.gated_switched)
            and bool(row.native_correct)
            and bool(row.gated_correct)
        ):
            return "E8"
        if not bool(row.native_correct):
            return "E3"
        return "E4"

    result["failure_category"] = [
        category(row) for row in result.itertuples(index=False)
    ]
    if result["sample_id"].duplicated().any() or not set(
        result["failure_category"]
    ).issubset(FAILURE_CATEGORIES):
        raise RuntimeError("D1 failure funnel is not mutually exclusive/exhaustive")
    return result


def _objective_mismatch(
    formal: Mapping[str, Any], funnel: pd.DataFrame, evidence: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = (
        formal["outcomes"].loc[lambda frame: frame["source_route"].eq("D1")].copy()
    )
    threshold = float(evidence["q_saturation_threshold"])
    bins = [-np.inf, 0.25, 0.5, 0.75, threshold, np.inf]
    outcomes["q_bin"] = pd.cut(
        outcomes["native_score"], bins=bins, include_lowest=True
    ).astype(str)
    q_table = (
        outcomes.groupby("q_bin", observed=False)
        .agg(
            candidates=("candidate_id", "size"),
            successes=("candidate_success", "sum"),
            mean_q=("native_score", "mean"),
        )
        .reset_index()
    )
    q_table["success_rate"] = q_table["successes"] / q_table["candidates"].replace(
        0, np.nan
    )
    iou_threshold, angle_threshold = _load_evaluator_thresholds(
        formal["plan"]["components"]["canonical_evaluator"]
    )
    native_outcomes = outcomes.loc[outcomes["native_rank"].eq(1)].merge(
        funnel[["sample_id", "failure_category"]], on="sample_id", validate="one_to_one"
    )
    native_outcomes["failure_mode"] = np.select(
        [
            native_outcomes["candidate_success"].astype(bool),
            native_outcomes["best_same_gt_iou"].ge(iou_threshold)
            & native_outcomes["best_same_gt_angle_error_deg"].gt(angle_threshold),
            native_outcomes["best_same_gt_iou"].lt(iou_threshold)
            & native_outcomes["best_same_gt_angle_error_deg"].le(angle_threshold),
        ],
        ["pass", "iou_pass_angle_fail", "angle_pass_iou_fail"],
        default="both_fail",
    )
    native_outcomes["q_saturated"] = native_outcomes["native_score"].ge(threshold)
    summary = (
        native_outcomes.groupby(["failure_category", "failure_mode", "q_saturated"])
        .size()
        .rename("samples")
        .reset_index()
    )
    return q_table, summary


def _case_selection(
    formal: Mapping[str, Any], funnel: pd.DataFrame, evidence: Mapping[str, Any]
) -> pd.DataFrame:
    router = _decision_table(formal["bundle"], "four_route_crog_default_router").rename(
        columns={
            "selected_source_route": "router_route",
            "selected_correct": "router_correct",
        }
    )
    union = _decision_table(formal["bundle"], "top20_union").rename(
        columns={
            "selected_source_route": "union_route",
            "selected_correct": "union_correct",
        }
    )
    work = funnel.merge(
        router[["sample_id", "router_route", "router_correct"]], on="sample_id"
    ).merge(union[["sample_id", "union_route", "union_correct"]], on="sample_id")
    q_threshold = float(evidence["q_saturation_threshold"])
    mask_threshold = float(evidence["mask_quality_threshold"])
    selectors = {
        "d1_recovered": work["failure_category"].eq("E5"),
        "d1_harmful": work["failure_category"].eq("E6"),
        "gate_prevented_harmful": work["failure_category"].eq("E10"),
        "gate_missed_recoverable": work["failure_category"].eq("E9"),
        "positive_rank_2_5": work["first_positive_native_rank"].between(2, 5),
        "positive_rank_6_10": work["first_positive_native_rank"].between(6, 10),
        "no_positive_full_pool": work["failure_category"].eq("E1"),
        "correct_grounding_ranking_failure": work["mask_quality"].ge(mask_threshold)
        & work["failure_category"].eq("E3"),
        "low_quality_grounding_association": work["mask_quality"].lt(mask_threshold),
        "q_saturation_failure": work["native_correct"].eq(False)
        & work["native_q"].ge(q_threshold),
        "four_route_router_uniquely_recovered_by_d1": work["router_route"].eq("D1")
        & work["router_correct"].astype(bool)
        & ~work["three_route_correct"].astype(bool),
        "top20_union_uniquely_selected_d1": work["union_route"].eq("D1")
        & work["union_correct"].astype(bool)
        & ~work["router_correct"].astype(bool),
    }
    rows = []
    for category in CASE_CATEGORIES:
        candidates = sorted(work.loc[selectors[category], "sample_id"].astype(str))
        for index, sample_id in enumerate(candidates[:5], 1):
            rows.append(
                {
                    "case_category": category,
                    "selection_rank": index,
                    "sample_id": sample_id,
                    "selection_rule": "lexicographic_sample_id_over_complete_population",
                }
            )
    return pd.DataFrame(
        rows,
        columns=["case_category", "selection_rank", "sample_id", "selection_rule"],
    )


def _asset_path(
    row: Mapping[str, Any],
    *,
    path_columns: Sequence[str],
    hash_columns: Sequence[str],
    name: str,
) -> tuple[Path, str]:
    raw_path = next(
        (
            row[column]
            for column in path_columns
            if column in row and pd.notna(row[column]) and str(row[column]).strip()
        ),
        None,
    )
    if raw_path is None:
        raise RuntimeError(f"D1 selected case misses {name} path")
    path = Path(str(raw_path)).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"D1 selected case {name} is missing/not regular: {path}")
    observed = sha256_file(path)
    expected = next(
        (
            str(row[column])
            for column in hash_columns
            if column in row and pd.notna(row[column]) and str(row[column]).strip()
        ),
        "",
    )
    if expected and expected != observed:
        raise RuntimeError(f"D1 selected case {name} asset hash differs: {path}")
    return path, observed


def _rectangle_corners(row: Mapping[str, Any]) -> np.ndarray:
    values = [
        float(row[field])
        for field in ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")
    ]
    cx, cy, theta, width, height = values
    if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
        raise RuntimeError("D1 case candidate geometry is non-finite/non-positive")
    angle = np.deg2rad(-theta)
    axis = np.asarray([np.cos(angle), np.sin(angle)])
    normal = np.asarray([-axis[1], axis[0]])
    center = np.asarray([cx, cy])
    return np.asarray(
        [
            center - axis * width / 2 - normal * height / 2,
            center + axis * width / 2 - normal * height / 2,
            center + axis * width / 2 + normal * height / 2,
            center - axis * width / 2 + normal * height / 2,
        ]
    )


def _ground_truth_rectangles(value: object, evaluator: Any) -> list[np.ndarray]:
    try:
        decoded = json.loads(str(value)) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "D1 selected case ground-truth grasp JSON is invalid"
        ) from error
    if not isinstance(decoded, list):
        raise RuntimeError("D1 selected case ground-truth grasps are not a list")
    rectangles = []
    for item in decoded:
        if isinstance(item, Mapping):
            item = item.get("corners", item.get("points"))
        converted = evaluator.gt_from_corners(item)
        array = np.asarray(evaluator.corners(converted), dtype=float)
        if array.shape != (4, 2) or not np.isfinite(array).all():
            raise RuntimeError(
                "D1 selected case ground-truth rectangle geometry differs"
            )
        rectangles.append(array)
    return rectangles


def _rectangle_diagnostics(
    candidate: Mapping[str, Any], gt: np.ndarray
) -> dict[str, float]:
    candidate_corners = _rectangle_corners(candidate)
    candidate_center = candidate_corners.mean(axis=0)
    gt_center = gt.mean(axis=0)
    candidate_edges = np.linalg.norm(
        np.roll(candidate_corners, -1, axis=0) - candidate_corners, axis=1
    )
    gt_edges = np.linalg.norm(np.roll(gt, -1, axis=0) - gt, axis=1)
    candidate_angle = math.degrees(
        math.atan2(
            candidate_corners[1, 1] - candidate_corners[0, 1],
            candidate_corners[1, 0] - candidate_corners[0, 0],
        )
    )
    gt_angle = math.degrees(math.atan2(gt[1, 1] - gt[0, 1], gt[1, 0] - gt[0, 0]))
    return {
        "center_error_px": float(np.linalg.norm(candidate_center - gt_center)),
        "width_error_px": abs(float(candidate_edges.max() - gt_edges.max())),
        "periodic_angle_error_deg": abs(
            ((candidate_angle - gt_angle + 90.0) % 180.0) - 90.0
        ),
    }


def _load_visual_ground_truth(
    root: Path,
    *,
    evidence: Mapping[str, Any],
    formal: Mapping[str, Any],
    denominator: set[str],
) -> pd.DataFrame:
    if (
        formal["execution"].get("status") != "COMPLETE"
        or formal["execution"].get("execution_count") != 1
    ):
        raise PermissionError("D1 visual Test rows require the completed formal claim")
    path = _verified_path(
        evidence.get("opaque_visual_ground_truth"),
        name="D1 opaque visual ground truth",
    )
    visual = pd.read_parquet(path)
    required = {"sample_id", "gt_mask_path", "gt_grasp_list_json"}
    if required.difference(visual.columns):
        raise RuntimeError("D1 visual ground-truth schema differs")
    if not any(column in visual for column in ("rgb_path", "source_rgb_path")):
        raise RuntimeError("D1 visual ground truth misses RGB paths")
    if not any(
        column in visual
        for column in ("predicted_mask_path", "predicted_hifics_mask_path")
    ):
        raise RuntimeError("D1 visual ground truth misses predicted-mask paths")
    visual["sample_id"] = visual["sample_id"].astype(str)
    if (
        visual["sample_id"].duplicated().any()
        or set(visual["sample_id"]) != denominator
    ):
        raise RuntimeError("D1 visual ground-truth denominator differs")
    append_access_log(
        root,
        {
            "event_id": "d1-postformal-visual-ground-truth-read-v1",
            "event": "d1_postformal_visual_ground_truth_read",
            "opened_after_complete_formal_claim": True,
            "opened_after_independent_recompute_pass": True,
            "formal_execution_count": 1,
            "opaque_visual_ground_truth_path": str(path),
            "opaque_visual_ground_truth_sha256": sha256_file(path),
            "row_count": len(visual),
            "selection_feedback_used": False,
        },
    )
    return visual.set_index("sample_id", drop=False)


def _mask_array(path: Path, *, target_instance_id: object, name: str) -> np.ndarray:
    from PIL import Image

    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2 or not np.isfinite(array.astype(float)).all():
        raise RuntimeError(f"D1 {name} mask is not a finite 2D array: {path}")
    if name == "GT" and pd.notna(target_instance_id):
        return array == int(target_instance_id)
    if name == "GT" and np.unique(array).size > 2:
        raise RuntimeError(
            "D1 multi-instance GT mask requires a locked target_instance_id"
        )
    return array > 0


def _derive_postformal_covariates(
    *,
    visual: pd.DataFrame,
    formal: Mapping[str, Any],
    label_free_covariates: pd.DataFrame,
) -> pd.DataFrame:
    mask_rows = []
    for sample_id, source in visual.iterrows():
        values = source.to_dict()
        gt_path, _ = _asset_path(
            values,
            path_columns=("gt_mask_path",),
            hash_columns=("gt_mask_sha256",),
            name="GT mask",
        )
        predicted_path, _ = _asset_path(
            values,
            path_columns=("predicted_mask_path", "predicted_hifics_mask_path"),
            hash_columns=("predicted_mask_sha256", "predicted_hifics_mask_sha256"),
            name="predicted mask",
        )
        target_instance = values.get("target_instance_id", pd.NA)
        gt = _mask_array(
            gt_path,
            target_instance_id=target_instance,
            name="GT",
        )
        predicted = _mask_array(
            predicted_path,
            target_instance_id=pd.NA,
            name="predicted",
        )
        if gt.shape != predicted.shape:
            raise RuntimeError(f"D1 predicted/GT mask shape differs: {sample_id}")
        intersection = int(np.logical_and(gt, predicted).sum())
        union = int(np.logical_or(gt, predicted).sum())
        mask_rows.append(
            {
                "sample_id": str(sample_id),
                "mask_quality": 1.0 if union == 0 else intersection / union,
            }
        )
    reference = _decision_table(
        formal["bundle"], "three_route_crog_default_router_reference"
    )[["sample_id", "selected_correct"]].rename(
        columns={"selected_correct": "three_route_correct"}
    )
    result = (
        label_free_covariates.merge(
            pd.DataFrame(mask_rows), on="sample_id", validate="one_to_one"
        )
        .merge(reference, on="sample_id", validate="one_to_one")
        .copy()
    )
    if result["mask_quality"].isna().any():
        raise RuntimeError("D1 postformal mask-quality derivation is incomplete")
    return result


def _allnms_candidates(formal: Mapping[str, Any]) -> pd.DataFrame:
    outcomes = (
        formal["outcomes"]
        .loc[lambda frame: frame["source_route"].astype(str).eq("D1")]
        .copy()
    )
    allnms_path = _verified_path(
        formal["plan"]["components"]["d1_allnms_candidates"],
        name="D1 AllNMS candidates",
    )
    allnms = pd.read_parquet(allnms_path)
    if "route" in allnms and "source_route" not in allnms:
        allnms = allnms.rename(columns={"route": "source_route"})
    keys = ["source_route", "sample_id", "candidate_id"]
    allnms[keys] = allnms[keys].astype(str)
    outcomes[keys] = outcomes[keys].astype(str)
    expected = set(map(tuple, allnms[keys].itertuples(index=False, name=None)))
    observed = set(map(tuple, outcomes[keys].itertuples(index=False, name=None)))
    if expected != observed or outcomes.duplicated(keys).any():
        raise RuntimeError("D1 formal outcomes are not the exact AllNMS universe")
    geometry = allnms[keys + ["candidate_geometry_sha256"]].merge(
        outcomes,
        on=keys,
        validate="one_to_one",
        suffixes=("_allnms", ""),
    )
    if (
        not geometry["candidate_geometry_sha256_allnms"]
        .eq(geometry["candidate_geometry_sha256"])
        .all()
    ):
        raise RuntimeError("D1 case AllNMS geometry hash differs")
    return outcomes


def _render_case_boards(
    root: Path,
    *,
    formal: Mapping[str, Any],
    evidence: Mapping[str, Any],
    funnel: pd.DataFrame,
    cases: pd.DataFrame,
    denominator: set[str],
    visual: pd.DataFrame,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image

    plt.rcParams["svg.hashsalt"] = "d1-postformal-case-boards-v1"

    if set(visual.index.astype(str)) != denominator:
        raise RuntimeError("D1 case-board visual denominator differs")
    evaluator = _load_evaluator(formal["plan"]["components"]["canonical_evaluator"])
    iou_threshold = float(evaluator.IOU_THRESHOLD)
    angle_threshold = float(evaluator.ANGLE_THRESHOLD_DEG)
    feature_manifest_path = _verified_path(
        formal["plan"]["components"]["d1_top5_feature_manifest"],
        name="D1 case-board Top5 feature manifest",
    )
    feature_manifest = _load_content(
        feature_manifest_path,
        name="D1 case-board Top5 feature manifest",
        statuses=("COMPLETE",),
    )
    feature_path = _verified_path(
        feature_manifest.get("artifacts", {}).get("candidate_features"),
        name="D1 case-board Top5 features",
    )
    feature_diagnostics = pd.read_parquet(feature_path)
    feature_columns = {
        "sample_id",
        "candidate_id",
        "calibrated_native_probability",
    }
    if feature_columns.difference(feature_diagnostics.columns):
        raise RuntimeError("D1 case-board calibrated-q feature schema differs")
    feature_diagnostics[["sample_id", "candidate_id"]] = feature_diagnostics[
        ["sample_id", "candidate_id"]
    ].astype(str)
    gate_manifest_path = _verified_path(
        formal["plan"]["components"]["d1_primary_gate_manifest"],
        name="D1 case-board gate manifest",
    )
    gate_manifest = _load_content(
        gate_manifest_path, name="D1 case-board gate manifest", statuses=("COMPLETE",)
    )
    gate_path = _verified_path(
        gate_manifest.get("artifacts", {}).get("decisions"),
        name="D1 case-board gate diagnostics",
    )
    gate_diagnostics = pd.read_parquet(gate_path)
    if {"sample_id", "utility"}.difference(gate_diagnostics.columns):
        raise RuntimeError("D1 case-board gate utility schema differs")
    gate_diagnostics["sample_id"] = gate_diagnostics["sample_id"].astype(str)
    if gate_diagnostics["sample_id"].duplicated().any():
        raise RuntimeError("D1 case-board gate utility denominator is duplicated")
    ranker_manifest_path = _verified_path(
        formal["plan"]["components"]["d1_primary_ranker_manifest"],
        name="D1 case-board ranker manifest",
    )
    ranker_manifest = _load_content(
        ranker_manifest_path,
        name="D1 case-board ranker manifest",
        statuses=("COMPLETE",),
    )
    selected_method = str(ranker_manifest.get("selected_method", ""))
    if not selected_method:
        raise RuntimeError("D1 case-board selected ranker method is absent")
    contribution_status = f"NOT_APPLICABLE_{selected_method}"
    if selected_method.upper() in {"R5", "LAMBDAMART"}:
        _source_path, postformal_sources = load_postformal_sources(root)
        contribution_declaration_path = _verified_path(
            postformal_sources.get("artifacts", {}).get("ranker_contributions"),
            name="D1 fixed ranker-contribution declaration",
        )
        contribution_declaration = _load_content(
            contribution_declaration_path,
            name="D1 fixed ranker-contribution declaration",
            statuses=("COMPLETE",),
        )
        contribution_record = contribution_declaration.get(
            "candidate_contributions"
        )
        if (
            contribution_declaration.get("selected_method") != selected_method
            or not isinstance(contribution_record, Mapping)
        ):
            raise RuntimeError(
                "D1 R5 case board requires locked LightGBM candidate contributions"
            )
        _verified_path(contribution_record, name="D1 LightGBM candidate contributions")
        contribution_status = "LOCKED_LIGHTGBM_CONTRIBUTION_DIFF_AVAILABLE"
    candidates = _allnms_candidates(formal)
    allnms_bundle = formal["bundle"].loc[
        lambda frame: (
            frame["system_name"].eq("d1_allnms_locked")
            & frame["row_kind"].eq("candidate")
        )
    ][["sample_id", "candidate_id", "formal_score"]]
    candidates = candidates.merge(
        allnms_bundle,
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    decisions = {}
    for system in ("d1_top5_r0", "d1_top5_r7_ungated", "d1_top5_r7_gated"):
        decision = _decision_table(formal["bundle"], system).set_index("sample_id")
        decisions[system] = decision
    funnel_lookup = funnel.set_index("sample_id")
    output = root / "14_case_selection" / "boards"
    output.mkdir(parents=True, exist_ok=True)
    artifact_records: dict[str, Any] = {}
    qa_rows: list[dict[str, Any]] = []

    def draw_rectangles(
        axis: Any,
        rows: pd.DataFrame,
        *,
        selected: Mapping[str, str],
        external_labels: bool = False,
    ) -> None:
        ordered = rows.sort_values("native_rank", kind="mergesort").to_dict("records")
        for index, candidate in enumerate(ordered):
            candidate_id = str(candidate["candidate_id"])
            rank = int(candidate["native_rank"])
            color = "#56B4E9" if rank <= 5 else "#E69F00" if rank <= 10 else "#999999"
            width = 0.8
            zorder = 2
            linestyle = "-" if rank <= 5 else "--" if rank <= 10 else ":"
            if candidate_id in selected.values():
                role = next(
                    key for key, value in selected.items() if value == candidate_id
                )
                color = {
                    "native": "#00BFC4",
                    "ungated": "#E69F00",
                    "gated": "#CC79A7",
                }[role]
                linestyle = "--" if role == "ungated" else "-"
                width = 3.0
                zorder = 4
            corners = _rectangle_corners(candidate)
            axis.add_patch(
                Polygon(
                    corners,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=width,
                    linestyle=linestyle,
                    zorder=zorder,
                )
            )
            if external_labels:
                height = float(rows["cy_px"].max() + rows["height_px"].max())
                label_y = (index + 1) * max(height, 1.0) / (len(ordered) + 1)
                anchor = corners.mean(axis=0)
                axis.annotate(
                    f"{candidate_id} · r{rank}",
                    xy=(float(anchor[0]), float(anchor[1])),
                    xytext=(1.02, label_y / max(height, 1.0)),
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
                    fontsize=9,
                    color=color,
                    va="center",
                    clip_on=False,
                )

    for category in CASE_CATEGORIES:
        category_dir = output / category
        category_dir.mkdir(parents=True, exist_ok=True)
        selected_cases = cases.loc[cases["case_category"].eq(category)].sort_values(
            "selection_rank", kind="mergesort"
        )
        if selected_cases.empty:
            no_case = category_dir / "NO_CASE.md"
            atomic_text(
                no_case,
                f"# No case\n\nThe locked formal population for `{category}` is zero; no case was fabricated.\n",
            )
            artifact_records[category] = {
                "population_zero": True,
                "no_case": _record(no_case),
                "boards": [],
            }
            continue
        board_paths: list[Path] = []
        board_records = []
        for selected_case in selected_cases.itertuples(index=False):
            sample_id = str(selected_case.sample_id)
            source = visual.loc[sample_id].to_dict()
            rgb_path, rgb_sha = _asset_path(
                source,
                path_columns=("rgb_path", "source_rgb_path"),
                hash_columns=("image_sha256", "rgb_sha256", "source_rgb_sha256"),
                name="RGB",
            )
            gt_path, gt_sha = _asset_path(
                source,
                path_columns=("gt_mask_path",),
                hash_columns=("gt_mask_sha256",),
                name="GT mask",
            )
            predicted_path, predicted_sha = _asset_path(
                source,
                path_columns=("predicted_mask_path", "predicted_hifics_mask_path"),
                hash_columns=(
                    "predicted_mask_sha256",
                    "predicted_hifics_mask_sha256",
                ),
                name="predicted mask",
            )
            probability_path, probability_sha = _asset_path(
                source,
                path_columns=(
                    "hifics_probability_path",
                    "predicted_probability_path",
                    "probability_map_path",
                ),
                hash_columns=(
                    "hifics_probability_sha256",
                    "predicted_probability_sha256",
                    "probability_map_sha256",
                ),
                name="HiFi probability map",
            )
            depth_path, depth_sha = _asset_path(
                source,
                path_columns=("aligned_depth_path", "depth_path"),
                hash_columns=("aligned_depth_sha256", "depth_sha256"),
                name="aligned depth",
            )
            prompt = next(
                (
                    str(source[column]).strip()
                    for column in (
                        "language_prompt",
                        "referring_expression",
                        "query",
                        "text",
                    )
                    if column in source
                    and pd.notna(source[column])
                    and str(source[column]).strip()
                ),
                "",
            )
            if not prompt:
                raise RuntimeError(
                    f"D1 selected case language prompt is absent: {sample_id}"
                )
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            gt_mask = np.asarray(Image.open(gt_path))
            predicted_mask = np.asarray(Image.open(predicted_path))
            probability = (
                np.load(probability_path)
                if probability_path.suffix.lower() == ".npy"
                else np.asarray(Image.open(probability_path))
            )
            depth = (
                np.load(depth_path)
                if depth_path.suffix.lower() == ".npy"
                else np.asarray(Image.open(depth_path))
            )
            if probability.ndim == 3:
                probability = probability[..., 0]
            if depth.ndim == 3:
                depth = depth[..., 0]
            if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.any():
                raise RuntimeError(f"D1 selected RGB is empty/invalid: {sample_id}")
            if (
                gt_mask.shape[:2] != rgb.shape[:2]
                or predicted_mask.shape[:2] != rgb.shape[:2]
                or probability.shape[:2] != rgb.shape[:2]
                or depth.shape[:2] != rgb.shape[:2]
                or probability.ndim != 2
                or depth.ndim != 2
                or not np.isfinite(probability.astype(float)).all()
                or not np.isfinite(depth.astype(float)).all()
            ):
                raise RuntimeError(f"D1 selected case asset shape differs: {sample_id}")
            gt_rectangles = _ground_truth_rectangles(
                source["gt_grasp_list_json"], evaluator
            )
            sample_candidates = candidates.loc[
                candidates["sample_id"].eq(sample_id)
            ].copy()
            if sample_candidates.empty:
                raise RuntimeError(
                    f"D1 selected case has no AllNMS candidates: {sample_id}"
                )
            selected_ids = {
                "native": str(
                    decisions["d1_top5_r0"].loc[sample_id, "selected_candidate_id"]
                ),
                "ungated": str(
                    decisions["d1_top5_r7_ungated"].loc[
                        sample_id, "selected_candidate_id"
                    ]
                ),
                "gated": str(
                    decisions["d1_top5_r7_gated"].loc[
                        sample_id, "selected_candidate_id"
                    ]
                ),
            }
            nonempty_selected = {value for value in selected_ids.values() if value}
            if nonempty_selected.difference(
                set(sample_candidates["candidate_id"].astype(str))
            ):
                raise RuntimeError(
                    f"D1 selected case identity is outside AllNMS: {sample_id}"
                )
            funnel_row = funnel_lookup.loc[sample_id]
            sample_features = feature_diagnostics.loc[
                feature_diagnostics["sample_id"].eq(sample_id)
            ].set_index("candidate_id")
            utility_rows = gate_diagnostics.loc[
                gate_diagnostics["sample_id"].eq(sample_id), "utility"
            ]
            if len(utility_rows) != 1:
                raise RuntimeError(
                    f"D1 selected case gate utility is absent: {sample_id}"
                )
            gate_utility = float(utility_rows.iloc[0])
            role_systems = {
                "native": "d1_top5_r0",
                "ungated": "d1_top5_r7_ungated",
                "gated": "d1_top5_r7_gated",
            }
            metric_rows: list[dict[str, Any]] = []
            for role, candidate_id in selected_ids.items():
                if not candidate_id:
                    metric_rows.append({"role": role, "candidate_id": "NO_OUTPUT"})
                    continue
                candidate = sample_candidates.loc[
                    sample_candidates["candidate_id"].astype(str).eq(candidate_id)
                ].iloc[0]
                bundle_score = formal["bundle"].loc[
                    formal["bundle"]["system_name"].eq(role_systems[role])
                    & formal["bundle"]["row_kind"].eq("candidate")
                    & formal["bundle"]["sample_id"].astype(str).eq(sample_id)
                    & formal["bundle"]["candidate_id"].astype(str).eq(candidate_id),
                    "formal_score",
                ]
                if len(bundle_score) != 1 or candidate_id not in sample_features.index:
                    raise RuntimeError(
                        f"D1 selected case score/calibrated q is absent: {sample_id}/{candidate_id}"
                    )
                matched_index = (
                    int(candidate["matched_gt_index"])
                    if pd.notna(candidate["matched_gt_index"])
                    else -1
                )
                geometry_diagnostics = {
                    "center_error_px": math.nan,
                    "width_error_px": math.nan,
                    "periodic_angle_error_deg": float(
                        candidate["best_same_gt_angle_error_deg"]
                    ),
                }
                if 0 <= matched_index < len(gt_rectangles):
                    geometry_diagnostics = _rectangle_diagnostics(
                        candidate, gt_rectangles[matched_index]
                    )
                passed = bool(candidate["candidate_success"])
                metric_rows.append(
                    {
                        "role": role,
                        "candidate_id": candidate_id,
                        "native_rank": int(candidate["native_rank"]),
                        "gqcnn_q": float(candidate["native_score"]),
                        "calibrated_q": float(
                            sample_features.loc[
                                candidate_id, "calibrated_native_probability"
                            ]
                        ),
                        "rerank_score": float(bundle_score.iloc[0]),
                        "gate_utility": gate_utility,
                        "iou": float(candidate["best_same_gt_iou"]),
                        **geometry_diagnostics,
                        "matched_gt_index": matched_index,
                        "pass_reason": (
                            f"PASS: IoU>={iou_threshold:g} and angle<{angle_threshold:g}°"
                            if passed
                            else f"FAIL: IoU/angle threshold not jointly met ({iou_threshold:g}, {angle_threshold:g}°)"
                        ),
                    }
                )
            gated_metric = next(row for row in metric_rows if row["role"] == "gated")
            matched_gt_index = int(gated_metric.get("matched_gt_index", -1))
            earliest_issue = {
                "E0": "Candidate generation / no D1 output",
                "E1": "Candidate generation / no positive in AllNMS",
                "E2": "Native GQ rank / positive below Top10",
                "E2b": "Native GQ rank / positive at ranks 6–10",
                "E3": "Native GQ rank / Top5 ordering",
                "E4": "No failure: native selection correct",
                "E5": "Native GQ rank recovered by learned reranker",
                "E6": "Learned reranker harmful switch",
                "E7": "Learned reranker failed to recover",
                "E8": "No failure: gate accepted recovery",
                "E9": "Gate missed recoverable switch",
                "E10": "Gate prevented harmful switch",
            }.get(str(funnel_row.failure_category), "Unclassified")

            fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=160)
            for axis in axes.flat:
                axis.set_xticks([])
                axis.set_yticks([])
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title("A · RGB + language prompt", fontsize=12)
            axes[0, 0].set_xlabel(prompt, fontsize=10, wrap=True)
            axes[0, 1].imshow(gt_mask, cmap="gray")
            axes[0, 1].set_title("B · Frozen GT target mask", fontsize=12)
            axes[0, 2].imshow(probability, cmap="viridis")
            axes[0, 2].contour(
                predicted_mask > 0, levels=[0.5], colors=["white"], linewidths=1.3
            )
            axes[0, 2].set_title("C · HiFi probability + binary contour", fontsize=12)
            axes[0, 3].imshow(depth, cmap="magma")
            axes[0, 3].set_title("D · Aligned depth / crop frame", fontsize=12)
            axes[1, 0].imshow(rgb)
            draw_rectangles(
                axes[1, 0], sample_candidates, selected={}, external_labels=True
            )
            axes[1, 0].set_title("E · Full AllNMS D1 candidate pool", fontsize=12)
            axes[1, 1].imshow(rgb)
            draw_rectangles(
                axes[1, 1], sample_candidates, selected={}, external_labels=True
            )
            axes[1, 1].set_title(
                "F · Native ranks: Top5 solid / Top10 dashed", fontsize=12
            )
            for column, (role, title) in enumerate(
                (
                    ("native", "G · Native selection · thick cyan"),
                    ("ungated", "H · Ungated reranker · orange dashed"),
                ),
                2,
            ):
                axes[1, column].imshow(rgb)
                draw_rectangles(
                    axes[1, column],
                    sample_candidates,
                    selected={role: selected_ids[role]},
                )
                axes[1, column].set_title(title, fontsize=12)
            axes[2, 0].imshow(rgb)
            draw_rectangles(
                axes[2, 0], sample_candidates, selected={"gated": selected_ids["gated"]}
            )
            axes[2, 0].set_title("I · Gated selection · thick magenta", fontsize=12)
            axes[2, 1].imshow(rgb)
            for rectangle in gt_rectangles:
                axes[2, 1].add_patch(
                    Polygon(
                        rectangle,
                        closed=True,
                        fill=False,
                        edgecolor="#0072B2",
                        linewidth=1.7,
                        linestyle="--",
                    )
                )
            axes[2, 1].set_title("J · Evaluator-converted all GT grasps", fontsize=12)
            axes[2, 2].imshow(rgb)
            if selected_ids["gated"]:
                draw_rectangles(
                    axes[2, 2],
                    sample_candidates,
                    selected={"gated": selected_ids["gated"]},
                )
            if 0 <= matched_gt_index < len(gt_rectangles):
                axes[2, 2].add_patch(
                    Polygon(
                        gt_rectangles[matched_gt_index],
                        closed=True,
                        fill=False,
                        edgecolor="#0072B2",
                        linewidth=3.2,
                    )
                )
            axes[2, 2].set_title("K · Selected candidate + matched GT", fontsize=12)
            axes[2, 3].axis("off")
            metric_lines = []
            for row in metric_rows:
                if row["candidate_id"] == "NO_OUTPUT":
                    metric_lines.append(f"{row['role']}: NO_OUTPUT")
                    continue
                metric_lines.extend(
                    (
                        f"{row['role']}: {row['candidate_id']} · rank {row['native_rank']}",
                        f" q={row['gqcnn_q']:.4f} · calibrated={row['calibrated_q']:.4f} · rerank={row['rerank_score']:.4f}",
                        f" utility={row['gate_utility']:.4f} · IoU={row['iou']:.4f} · angle={row['periodic_angle_error_deg']:.2f}°",
                        f" center={row['center_error_px']:.2f}px · width={row['width_error_px']:.2f}px · {row['pass_reason']}",
                    )
                )
            axes[2, 3].text(
                0.0,
                1.0,
                "\n".join(
                    [
                        "L · Metrics and module diagnosis",
                        f"Category/sample: {category} / {sample_id}",
                        f"Failure funnel: {funnel_row.failure_category}",
                        *metric_lines,
                        "",
                        f"Visual grounding: mask IoU={float(funnel_row.mask_quality):.4f}",
                        f"Candidate generation: AllNMS n={len(sample_candidates)}",
                        f"Native GQ rank: first positive={funnel_row.first_positive_native_rank}",
                        f"Learned reranker: {selected_method}",
                        f"Gate: utility={gate_utility:.4f}",
                        f"Earliest issue: {earliest_issue}",
                        f"LightGBM contribution diff: {contribution_status}",
                        "Validation ablation/global importance: locked evidence retained",
                    ]
                ),
                va="top",
                ha="left",
                fontsize=10,
                wrap=True,
            )
            fig.subplots_adjust(
                left=0.03, right=0.97, top=0.96, bottom=0.04, wspace=0.30, hspace=0.22
            )
            safe_sample = canonical_sha256(sample_id)[:16]
            stem = category_dir / (
                f"{int(selected_case.selection_rank):02d}_{safe_sample}_case_board"
            )
            png = stem.with_suffix(".png")
            svg = stem.with_suffix(".svg")
            fig.savefig(png, dpi=160)
            fig.savefig(svg, metadata={"Date": None})
            plt.close(fig)
            with Image.open(png) as rendered:
                width_px, height_px = rendered.size
            if width_px <= 0 or height_px <= 0:
                raise RuntimeError("D1 case board has invalid dimensions")
            geometry_hashes = {
                role: (
                    str(
                        sample_candidates.loc[
                            sample_candidates["candidate_id"]
                            .astype(str)
                            .eq(candidate_id),
                            "candidate_geometry_sha256",
                        ].iloc[0]
                    )
                    if candidate_id
                    else ""
                )
                for role, candidate_id in selected_ids.items()
            }
            candidate_qa = []
            for candidate in sample_candidates.sort_values("native_rank").itertuples(
                index=False
            ):
                candidate_qa.append(
                    {
                        "candidate_id": str(candidate.candidate_id),
                        "native_rank": int(candidate.native_rank),
                        "candidate_geometry_sha256": str(
                            candidate.candidate_geometry_sha256
                        ),
                        "display_score": (
                            float(candidate.formal_score)
                            if int(candidate.native_rank) <= 5
                            and pd.notna(candidate.formal_score)
                            else None
                        ),
                        "display_score_na_reason": (
                            "below_top5_by_predeclared_visual_policy"
                            if int(candidate.native_rank) > 5
                            else ""
                        ),
                    }
                )
            qa_rows.append(
                {
                    "case_category": category,
                    "sample_id": sample_id,
                    "png_width": width_px,
                    "png_height": height_px,
                    "rgb_sha256": rgb_sha,
                    "gt_mask_sha256": gt_sha,
                    "predicted_mask_sha256": predicted_sha,
                    "probability_map_sha256": probability_sha,
                    "aligned_depth_sha256": depth_sha,
                    "language_prompt": prompt,
                    "selected_candidate_ids": selected_ids,
                    "selected_geometry_sha256": geometry_hashes,
                    "matched_gt_index": matched_gt_index,
                    "evaluator_conversion_used": True,
                    "required_panels": list("ABCDEFGHIJKL"),
                    "required_metric_fields": [
                        "candidate_id",
                        "native_rank",
                        "gqcnn_q",
                        "calibrated_q",
                        "rerank_score",
                        "gate_utility",
                        "iou",
                        "periodic_angle_error_deg",
                        "center_error_px",
                        "width_error_px",
                        "pass_reason",
                    ],
                    "metric_rows": metric_rows,
                    "module_diagnosis": {
                        "visual_grounding": float(funnel_row.mask_quality),
                        "candidate_generation_count": len(sample_candidates),
                        "native_gq_rank": funnel_row.first_positive_native_rank,
                        "learned_reranker": selected_method,
                        "gate_utility": gate_utility,
                        "earliest_issue": earliest_issue,
                        "lightgbm_contribution_diff": contribution_status,
                    },
                    "allnms_candidates": candidate_qa,
                }
            )
            board_paths.append(png)
            board_records.append({"png": _record(png), "svg": _record(svg)})
        contact_fig, contact_axes = plt.subplots(
            len(board_paths), 1, figsize=(12, 7 * len(board_paths)), dpi=100
        )
        axes_list = np.atleast_1d(contact_axes)
        for axis, board_path in zip(axes_list, board_paths, strict=True):
            axis.imshow(np.asarray(Image.open(board_path).convert("RGB")))
            axis.axis("off")
        contact_fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.01)
        contact_png = category_dir / "CONTACT_SHEET.png"
        contact_svg = category_dir / "CONTACT_SHEET.svg"
        contact_fig.savefig(contact_png, dpi=100)
        contact_fig.savefig(contact_svg, metadata={"Date": None})
        plt.close(contact_fig)
        artifact_records[category] = {
            "population_zero": False,
            "boards": board_records,
            "contact_sheet_png": _record(contact_png),
            "contact_sheet_svg": _record(contact_svg),
        }
    qa_path = root / "14_case_selection" / "CASE_BOARD_QA.json"
    qa: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "visual_ground_truth": evidence["opaque_visual_ground_truth"],
        "all_d1_candidates_source": _record(
            _verified_path(
                formal["plan"]["components"]["d1_allnms_candidates"],
                name="D1 AllNMS candidates",
            )
        ),
        "boards": qa_rows,
        "required_panel_inventory": list("ABCDEFGHIJKL"),
        "required_metric_inventory": [
            "candidate_id",
            "native_rank",
            "gqcnn_q",
            "calibrated_q",
            "rerank_score",
            "gate_utility",
            "iou",
            "periodic_angle_error_deg",
            "center_error_px",
            "width_error_px",
            "pass_reason",
        ],
        "canonical_evaluator_gt_conversion": True,
        "asset_shape_fallback_used": False,
        "black_image_fallback_used": False,
    }
    qa["content_sha256"] = canonical_sha256(qa)
    atomic_json(qa_path, qa)
    return {"categories": artifact_records, "qa": _record(qa_path)}


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _formal_table(formal: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    metrics = formal["metrics"]["systems"]
    for name in sorted(FORMAL_SYSTEMS):
        value = metrics[name]
        rank_rows = value.get("rank_metrics")
        inventory = value.get("rank_metric_inventory")
        if (
            not isinstance(rank_rows, list)
            or not rank_rows
            or inventory != ["j_at_k_numerator", "j_at_k", "mrr_at_k", "ndcg_at_k"]
        ):
            raise RuntimeError(f"D1 formal rank metrics are incomplete: {name}")
        by_k = {int(row["k"]): row for row in rank_rows}
        max_k = int(value.get("max_k", 0))
        if sorted(by_k) != list(range(1, max_k + 1)):
            raise RuntimeError(f"D1 formal rank metric K inventory differs: {name}")

        def at(requested: int, field: str) -> float:
            # K beyond a variable-size pool is identical to the last available
            # rank; the pool oracle remains separately reported.
            return float(by_k[min(requested, max_k)][field])

        final = by_k[max_k]
        rows.append(
            {
                "system": name,
                "system_role": (
                    "REFERENCE_DIAGNOSTIC"
                    if name in REFERENCE_SYSTEMS
                    else "FORMAL_OUTPUT"
                ),
                "sample_count": value["sample_count"],
                "j_at_1": value["j_at_1"],
                "j_at_5": at(5, "j_at_k"),
                "j_at_10": at(10, "j_at_k"),
                "j_at_pool_max": float(final["j_at_k"]),
                "mrr_at_pool_max": float(final["mrr_at_k"]),
                "ndcg_at_pool_max": float(final["ndcg_at_k"]),
                "oracle": value["oracle"],
                "selected_correct_numerator": value["selected_correct_numerator"],
                "oracle_numerator": value["oracle_numerator"],
                "no_output_samples": value["no_output_samples"],
                "max_k": max_k,
                "first_positive_rank_distribution_json": json.dumps(
                    value["first_positive_rank_distribution"], sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def _stratified(funnel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata = {
        "small_target": funnel["target_size"].astype(str).eq("small"),
        "relation_query": funnel["relation_query"].astype(bool),
        "clutter": funnel["clutter"].astype(bool),
        "depth_missing": funnel["depth_missing"].astype(bool),
    }
    for name, mask in strata.items():
        selected = funnel.loc[mask]
        rows.append(
            {
                "stratum": name,
                "samples": len(selected),
                "native_j_at_1": selected["native_correct"].mean()
                if len(selected)
                else np.nan,
                "gated_j_at_1": selected["gated_correct"].mean()
                if len(selected)
                else np.nan,
                "mean_mask_quality": selected["mask_quality"].mean()
                if len(selected)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _exact_mcnemar_p(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = min(b, c)
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    p_value = min(1.0, numerator / (1 << discordant))
    return max(float(p_value), float(np.nextafter(0.0, 1.0)))


def _gammaincc(shape: float, value: float) -> float:
    if value < 0 or shape <= 0:
        raise ValueError("invalid incomplete-gamma arguments")
    if value == 0:
        return 1.0
    epsilon = 3e-14
    if value < shape + 1.0:
        term = total = 1.0 / shape
        current = shape
        for _ in range(10_000):
            current += 1.0
            term *= value / current
            total += term
            if abs(term) < abs(total) * epsilon:
                break
        lower = total * math.exp(-value + shape * math.log(value) - math.lgamma(shape))
        return max(0.0, min(1.0, 1.0 - lower))
    tiny = 1e-300
    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(b, tiny)
    total = d
    for index in range(1, 10_000):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        total *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return max(
        float(np.nextafter(0.0, 1.0)),
        min(
            1.0, total * math.exp(-value + shape * math.log(value) - math.lgamma(shape))
        ),
    )


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    difference: np.ndarray,
    cluster_column: str,
    iterations: int,
    seed: int,
) -> np.ndarray:
    clusters = frame[cluster_column].fillna("").astype(str)
    clusters = clusters.where(clusters.ne(""), frame["sample_id"].astype(str))
    grouped = pd.DataFrame({"cluster": clusters, "difference": difference}).groupby(
        "cluster"
    )
    sums = grouped["difference"].sum().to_numpy(float)
    counts = grouped.size().to_numpy(float)
    if len(sums) == 0:
        raise RuntimeError("D1 clustered bootstrap has no clusters")
    random = np.random.default_rng(seed)
    replicates = np.empty(iterations, dtype=np.float64)
    batch_size = 256
    for start in range(0, iterations, batch_size):
        width = min(batch_size, iterations - start)
        sampled = random.integers(0, len(sums), size=(width, len(sums)))
        replicates[start : start + width] = sums[sampled].sum(axis=1) / counts[
            sampled
        ].sum(axis=1)
    return replicates


def _holm_adjust(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for position, index in enumerate(order):
        running = max(running, min(1.0, (total - position) * values[index]))
        adjusted[index] = running
    return adjusted


def _p_display(value: float) -> str:
    if value < 1e-6:
        return f"<{max(value, float(np.nextafter(0.0, 1.0))):.3e}"
    return f"{value:.6f}"


def _formal_statistics(
    root: Path,
    *,
    formal: Mapping[str, Any],
    independent: Mapping[str, Any],
    per_sample: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config_path = _verified_path(
        formal["plan"]["components"]["statistics_config"],
        name="D1 statistics config",
    )
    config = _load_content(
        config_path, name="D1 statistics config", statuses=("LOCKED",)
    )
    if (
        int(config.get("bootstrap_iterations", -1)) != 10_000
        or config.get("sample_unit") != "sample"
        or config.get("scene_clustered_bootstrap") is not True
        or config.get("frame_cluster_sensitivity") is not True
        or config.get("paired_test") != "exact_mcnemar"
        or config.get("multiple_comparison_correction") != "holm"
    ):
        raise RuntimeError("D1 postformal statistics config differs")
    _same_record(
        independent.get("sources", {}).get("statistics_config"),
        _record(config_path),
        name="D1 P17 statistics config",
    )
    wide = per_sample.pivot(
        index=["sample_id", "scene_id", "frame_id"],
        columns="system_name",
        values="selected_correct",
    ).reset_index()
    if set(FORMAL_SYSTEMS).difference(wide.columns):
        raise RuntimeError("D1 statistical input misses formal systems")
    comparisons = config.get("primary_comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise RuntimeError("D1 statistical comparisons are absent")
    iterations = int(config["bootstrap_iterations"])
    seed = int(config["bootstrap_seed"])
    rows = []
    replicate_rows = []
    raw_p = []
    for comparison_index, comparison in enumerate(comparisons):
        if not isinstance(comparison, list) or len(comparison) != 2:
            raise RuntimeError("D1 statistical comparison contract differs")
        challenger, reference = map(str, comparison)
        challenger_values = wide[challenger].astype(bool).to_numpy()
        reference_values = wide[reference].astype(bool).to_numpy()
        recovered = int((challenger_values & ~reference_values).sum())
        harmful = int((~challenger_values & reference_values).sum())
        p_value = _exact_mcnemar_p(recovered, harmful)
        raw_p.append(p_value)
        difference = challenger_values.astype(float) - reference_values.astype(float)
        scene = _cluster_bootstrap(
            wide,
            difference=difference,
            cluster_column="scene_id",
            iterations=iterations,
            seed=seed + comparison_index,
        )
        frame = _cluster_bootstrap(
            wide,
            difference=difference,
            cluster_column="frame_id",
            iterations=iterations,
            seed=seed + 10_000 + comparison_index,
        )
        rows.append(
            {
                "challenger": challenger,
                "reference": reference,
                "samples": len(wide),
                "recovered": recovered,
                "harmful": harmful,
                "net": recovered - harmful,
                "delta_j_at_1": float(difference.mean()),
                "mcnemar_exact_p": p_value,
                "mcnemar_exact_p_display": _p_display(p_value),
                "scene_bootstrap_ci_low": float(np.quantile(scene, 0.025)),
                "scene_bootstrap_ci_high": float(np.quantile(scene, 0.975)),
                "frame_bootstrap_ci_low": float(np.quantile(frame, 0.025)),
                "frame_bootstrap_ci_high": float(np.quantile(frame, 0.975)),
            }
        )
        replicate_rows.extend(
            {
                "challenger": challenger,
                "reference": reference,
                "replicate": index,
                "scene_delta": scene[index],
                "frame_delta": frame[index],
            }
            for index in range(iterations)
        )
    adjusted = _holm_adjust(raw_p)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = value
        row["holm_adjusted_p_display"] = _p_display(value)
    comparisons_frame = pd.DataFrame(rows)
    binary = wide[list(sorted(FORMAL_SYSTEMS))].astype(int).to_numpy()
    system_sums = binary.sum(axis=0)
    row_sums = binary.sum(axis=1)
    systems_count = binary.shape[1]
    denominator = systems_count * row_sums.sum() - np.square(row_sums).sum()
    if denominator <= 0:
        cochran_q = 0.0
        cochran_p = 1.0
        applicable = False
    else:
        cochran_q = float(
            (systems_count - 1)
            * (systems_count * np.square(system_sums).sum() - row_sums.sum() ** 2)
            / denominator
        )
        cochran_p = _gammaincc((systems_count - 1) / 2.0, cochran_q / 2.0)
        applicable = True
    output = root / "10_statistics"
    output.mkdir(parents=True, exist_ok=True)
    comparisons_path = output / "paired_comparisons.csv"
    replicates_path = output / "bootstrap_replicates.parquet"
    cochran_path = output / "cochran_q.json"
    _atomic_csv(comparisons_path, comparisons_frame)
    pd.DataFrame(replicate_rows).to_parquet(
        replicates_path, index=False, compression="zstd"
    )
    cochran = {
        "schema_version": 1,
        "status": "COMPLETE",
        "applicable": applicable,
        "systems": sorted(FORMAL_SYSTEMS),
        "sample_count": len(wide),
        "statistic": cochran_q,
        "degrees_of_freedom": systems_count - 1,
        "p_value": cochran_p,
        "p_value_display": _p_display(cochran_p),
    }
    cochran["content_sha256"] = canonical_sha256(cochran)
    atomic_json(cochran_path, cochran)
    sources = {
        "independent_per_sample": independent["artifacts"]["independent_per_sample"],
        "statistics_config": _record(config_path),
        "formal_bundle": _record(formal["bundle_path"]),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "sample_unit": "sample",
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": {
            "paired_comparisons": _record(comparisons_path),
            "bootstrap_replicates": _record(replicates_path),
            "cochran_q": _record(cochran_path),
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(output / "STATISTICS_MANIFEST.json", manifest)
    return manifest, comparisons_frame


def _runtime_complexity(evidence: Mapping[str, Any]) -> pd.DataFrame:
    records = evidence.get("runtime_manifests")
    if not isinstance(records, Mapping) or set(records) != set(RUNTIME_COMPONENTS):
        raise RuntimeError("D1 runtime source inventory differs")
    rows = []
    for component in RUNTIME_COMPONENTS:
        path = _verified_path(records[component], name=f"D1 runtime {component}")
        manifest = _load_content(
            path,
            name=f"D1 runtime {component}",
            statuses=("COMPLETE", "NOT_AVAILABLE"),
        )
        telemetry = manifest.get("telemetry", manifest)
        if not isinstance(telemetry, Mapping):
            raise RuntimeError(f"D1 runtime telemetry is not a mapping: {component}")
        if component == "feature_extraction" and manifest.get("status") == "COMPLETE":
            if telemetry.get("measurement_semantics") != (
                "feature_extraction_not_artifact_loading"
            ):
                raise RuntimeError(
                    "D1 feature runtime cannot substitute artifact-load latency"
                )
        scope = (
            "dexnet_candidate_generation"
            if component == "dexnet_candidate_generation"
            else "incremental_d1_reranking"
        )
        missing = []

        def value(field: str, *, required: bool) -> float | int | None:
            observed = telemetry.get(field)
            if observed is None:
                if required:
                    missing.append(field)
                return None
            if not isinstance(observed, (int, float)) or not math.isfinite(
                float(observed)
            ):
                raise RuntimeError(
                    f"D1 runtime {component}/{field} is not finite numeric"
                )
            if float(observed) < 0:
                raise RuntimeError(f"D1 runtime {component}/{field} is negative")
            return observed

        unavailable = str(manifest.get("unavailable_reason", "")).strip()
        complete = manifest.get("status") == "COMPLETE"
        latency = value("latency_ms_per_sample", required=complete)
        peak = value("peak_memory_mb", required=complete)
        parameters = value(
            "parameter_count", required=component == "ranker_inference" and complete
        )
        model_bytes = value(
            "model_bytes", required=component == "ranker_inference" and complete
        )
        reason = unavailable or (
            "missing_fields:" + ",".join(sorted(missing)) if missing else ""
        )
        rows.append(
            {
                "component": component,
                "cost_scope": scope,
                "availability_status": (
                    "AVAILABLE" if complete and not missing else "NOT_AVAILABLE"
                ),
                "latency_ms_per_sample": latency,
                "peak_memory_mb": peak,
                "parameter_count": parameters,
                "model_bytes": model_bytes,
                "measurement_semantics": telemetry.get("measurement_semantics", ""),
                "na_reason": reason,
                "source_manifest_path": str(path),
                "source_manifest_sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    if (
        frame.loc[
            frame["component"].eq("dexnet_candidate_generation"), "cost_scope"
        ].iloc[0]
        == frame.loc[frame["component"].eq("ranker_inference"), "cost_scope"].iloc[0]
    ):
        raise RuntimeError("D1 candidate-generation and reranking costs were conflated")
    return frame


def _plot_all(
    output: Path,
    formal_table: pd.DataFrame,
    funnel: pd.DataFrame,
    q_table: pd.DataFrame,
    evidence_tables: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "savefig.bbox": "tight",
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    output.mkdir(parents=True, exist_ok=True)

    def save(
        name: str, labels: Sequence[str], values: Sequence[float], ylabel: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.75, 2.8))
        x = np.arange(len(labels))
        ax.bar(
            x,
            values,
            color=[colors[index % len(colors)] for index in range(len(labels))],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        fig.savefig(
            output / name,
            metadata={"CreationDate": None, "ModDate": None},
        )
        plt.close(fig)

    formal_labels = formal_table["system"].str.replace("d1_", "", regex=False).tolist()
    save(FIGURE_NAMES[0], formal_labels, formal_table["j_at_1"], "J@1")
    validation = evidence_tables["d1_r0_r7_validation.csv"]
    numeric_validation = validation.select_dtypes(include=["number"])
    save(
        FIGURE_NAMES[1],
        [str(value) for value in validation.iloc[:, 0]],
        numeric_validation.iloc[:, -1].astype(float),
        str(numeric_validation.columns[-1]),
    )
    k_rows = formal_table.loc[
        formal_table["system"].isin(
            ["d1_top5_r0", "d1_top10_locked", "d1_allnms_locked"]
        )
    ]
    save(
        FIGURE_NAMES[2],
        k_rows["system"],
        k_rows["oracle"] - k_rows["j_at_1"],
        "Headroom",
    )
    ranks = funnel["first_positive_native_rank"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.hist(ranks, bins=min(10, max(1, len(ranks))), color=colors[0])
    ax.set_xlabel("First positive native rank")
    ax.set_ylabel("Samples")
    fig.savefig(
        output / FIGURE_NAMES[3],
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    counts = funnel["failure_category"].value_counts()
    save(
        FIGURE_NAMES[4],
        ["Recovered", "Harmful"],
        [counts.get("E5", 0), counts.get("E6", 0)],
        "Samples",
    )
    save(
        FIGURE_NAMES[5],
        q_table["q_bin"],
        q_table["success_rate"].fillna(0),
        "Candidate success rate",
    )
    feature = evidence_tables["d1_feature_ablation.csv"]
    feature_numeric = feature.select_dtypes(include=["number"])
    save(
        FIGURE_NAMES[6],
        feature.iloc[:, 0].astype(str),
        feature_numeric.iloc[:, -1],
        str(feature_numeric.columns[-1]),
    )
    gate = evidence_tables["d1_gate_transitions.csv"]
    gate_numeric = gate.select_dtypes(include=["number"])
    save(
        FIGURE_NAMES[7],
        gate.iloc[:, 0].astype(str),
        gate_numeric.iloc[:, -1],
        str(gate_numeric.columns[-1]),
    )
    route = formal_table.loc[
        formal_table["system"].isin(["four_route_crog_default_router", "top20_union"])
    ]
    save(FIGURE_NAMES[8], route["system"], route["j_at_1"], "J@1")
    ordered = [category for category in FAILURE_CATEGORIES if category in counts]
    save(
        FIGURE_NAMES[9], ordered, [counts[category] for category in ordered], "Samples"
    )
    return {name: _record(output / name) for name in FIGURE_NAMES}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join("NA" if pd.isna(value) else str(value) for value in row)
            + " |"
        )
    return "\n".join(lines)


def _reports(
    root: Path,
    formal_table: pd.DataFrame,
    funnel: pd.DataFrame,
    q_table: pd.DataFrame,
    statistics_table: pd.DataFrame,
    runtime_table: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    output = root / "16_reports"
    output.mkdir(parents=True, exist_ok=True)
    metrics = formal_table.set_index("system")
    native = metrics.loc["d1_top5_r0"]
    gated = metrics.loc["d1_top5_r7_gated"]
    top10 = metrics.loc["d1_top10_locked"]
    allnms = metrics.loc["d1_allnms_locked"]
    counts = funnel["failure_category"].value_counts()
    sentence = (
        f"Under the same {int(native.sample_count):,}-sample paired manifest and same-GT 4-DoF evaluator, "
        f"D1 native GQ-CNN ranking achieved J@1 = {native.j_at_1:.6f}, with Oracle@5 = "
        f"{native.oracle:.6f}, Oracle@10 = {top10.oracle:.6f}, and Oracle@All = {allnms.oracle:.6f}. "
        f"The validation-selected gated reranker changed J@1 to {gated.j_at_1:.6f} on the locked Test, "
        f"with {int(counts.get('E5', 0))} recovered and {int(counts.get('E6', 0))} harmful samples."
    )
    available_runtime = runtime_table.loc[
        runtime_table["availability_status"].eq("AVAILABLE")
    ]
    runtime_sentence = (
        "No complete runtime telemetry was available; each NA is accompanied by its "
        "prelocked reason."
        if available_runtime.empty
        else "Runtime is reported from exact prelocked telemetry; Dex-Net candidate generation "
        "is separated from incremental feature/ranker/gate cost."
    )
    en = (
        "# Final D1 reranking report\n\n"
        + sentence
        + "\n\n## Formal systems\n\n"
        + _markdown_table(formal_table)
        + "\n\n## Formal paired statistics\n\n"
        + _markdown_table(statistics_table)
        + "\n\n## Runtime and complexity\n\n"
        + _markdown_table(runtime_table)
        + "\n\n"
        + runtime_sentence
        + "\n\nThis is a locked retrospective D1 extension. Route-wise Top5 is primary; "
        "Top10, AllNMS, router, and union are secondary. Associations are not causal evidence.\n"
    )
    zh = (
        "# D1 重排序最终摘要\n\n"
        f"本实验是锁定后的回顾性 D1 扩展，共 {int(native.sample_count):,} 个配对样本。"
        f"原生 J@1 为 {native.j_at_1:.6f}，门控重排序后为 {gated.j_at_1:.6f}。"
        "Top5 为主分析；Top10、AllNMS、四路线 router 与 Top20 union 为次要分析。"
        "候选生成成本与增量特征、ranker、gate 成本分别报告；缺失遥测均说明原因。"
        "离线旋转矩形正确性不等同于真实物理抓取成功率。\n"
    )
    atomic_text(output / REPORT_NAMES[0], en)
    atomic_text(output / REPORT_NAMES[1], zh)
    conclusion = {
        "schema_version": 1,
        "status": "COMPLETE",
        "retrospective_extension": True,
        "formal_test_execution_count": 1,
        "summary_sentence": sentence,
        "native_j_at_1": native.j_at_1,
        "gated_j_at_1": gated.j_at_1,
        "oracle_at_5": native.oracle,
        "oracle_at_10": top10.oracle,
        "oracle_all": allnms.oracle,
        "recovered": int(counts.get("E5", 0)),
        "harmful": int(counts.get("E6", 0)),
        "claim_boundary": "offline target-specific 2D rectangle correctness; not physical grasp success",
    }
    conclusion["content_sha256"] = canonical_sha256(conclusion)
    atomic_json(output / REPORT_NAMES[2], conclusion)
    atomic_text(
        output / REPORT_NAMES[3],
        "\\section{D1 reranking method}\nD1 uses frozen Snapshot-A candidates, label-free features, "
        "a Validation-selected reranker, and an expected-gain gate.\\par\n",
    )
    atomic_text(
        output / REPORT_NAMES[4],
        "\\section{D1 results and discussion}\n"
        + sentence.replace("%", "\\%")
        + "\\par\n",
    )
    latex_table_rows = "\n".join(
        "{} & {:.4f} & {:.4f} \\\\".format(
            str(row.system).replace("_", "\\_"), row.j_at_1, row.oracle
        )
        for row in formal_table.itertuples(index=False)
    )
    atomic_text(
        output / REPORT_NAMES[5],
        "\\begin{table}[t]\\centering\\caption{D1 formal results.}"
        "\\begin{tabular}{lrr}System & J@1 & Oracle \\\\ \\hline\n"
        + latex_table_rows
        + "\n\\end{tabular}\\end{table}\n",
    )
    atomic_text(
        output / REPORT_NAMES[6],
        "# Thesis-ready D1 figure captions\n\n"
        + "\n".join(
            f"- `{name}`: Data-driven locked D1 postformal analysis."
            for name in FIGURE_NAMES
        ),
    )
    atomic_text(
        output / REPORT_NAMES[7],
        "# D1 limitations and claim boundaries\n\n"
        "- D1 is a retrospective locked extension, not a pristine joint blind experiment.\n"
        "- GQ-CNN q estimates local depth robustness; the evaluator measures target-specific 2D rectangle agreement.\n"
        "- Observed mask/score associations are not causal.\n"
        "- Offline rectangle correctness does not establish physical grasp success.\n",
    )
    atomic_text(
        output / REPORT_NAMES[8],
        "# Reproduce D1 postformal artifacts\n\n"
        "Run the P15 builder only after P14 COMPLETE and P17 PASS, then run the fail-closed finalizer. "
        "Neither command performs model selection. The builder opens only the separately locked visual "
        "ground-truth table after P14/P17, solely for deterministic case boards, and logs that access.\n",
    )
    return {name: _record(output / name) for name in REPORT_NAMES}


def build_postformal_artifacts(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    if any((root / name).exists() for name in (FINAL_LOCK_NAME, "COMPLETE")):
        raise PermissionError("D1 postformal builder refuses writes after final lock")
    destination = root / POSTFORMAL_MANIFEST_RELATIVE_PATH
    if destination.exists():
        raise FileExistsError(f"D1 postformal manifest already exists: {destination}")
    formal = _load_formal(root)
    independent_path, independent, independent_per_sample = _load_independent(root)
    if independent.get("sample_count") != formal["manifest"].get("sample_count"):
        raise RuntimeError("D1 P14/P17 sample count differs")
    evidence, label_free_covariates = _load_evidence(root, formal["lock"])
    denominator = set(independent_per_sample["sample_id"].astype(str))
    if set(label_free_covariates["sample_id"].astype(str)) != denominator:
        raise RuntimeError("D1 postformal covariate denominator differs")
    visual = _load_visual_ground_truth(
        root,
        evidence=evidence,
        formal=formal,
        denominator=denominator,
    )
    covariates = _derive_postformal_covariates(
        visual=visual,
        formal=formal,
        label_free_covariates=label_free_covariates,
    )
    funnel = classify_failure_funnel(formal, covariates)
    q_table, mismatch = _objective_mismatch(formal, funnel, evidence)
    cases = _case_selection(formal, funnel, evidence)
    formal_table = _formal_table(formal)
    statistics_manifest, statistics_table = _formal_statistics(
        root,
        formal=formal,
        independent=independent,
        per_sample=independent_per_sample,
    )
    runtime_table = _runtime_complexity(evidence)
    evidence_tables = {
        name: pd.read_csv(_verified_path(evidence["tables"][name], name=name))
        for name in EVIDENCE_TABLE_NAMES
    }
    tables = root / "tables"
    table_frames: dict[str, pd.DataFrame] = {
        **evidence_tables,
        "d1_formal_test_primary.csv": formal_table,
        "d1_failure_decomposition.csv": funnel,
    }
    for name in TABLE_NAMES:
        _atomic_csv(tables / name, table_frames[name])
    _atomic_csv(tables / "d1_q_objective_mismatch.csv", mismatch)
    _atomic_csv(tables / "d1_q_bins_success.csv", q_table)
    _atomic_csv(tables / "d1_stratified_results.csv", _stratified(funnel))
    _atomic_csv(tables / "d1_case_selection.csv", cases)
    _atomic_csv(tables / "d1_formal_statistics.csv", statistics_table)
    _atomic_csv(tables / "d1_runtime_complexity.csv", runtime_table)
    case_dir = root / "14_case_selection"
    case_dir.mkdir(parents=True, exist_ok=True)
    case_boards = _render_case_boards(
        root,
        formal=formal,
        evidence=evidence,
        funnel=funnel,
        cases=cases,
        denominator=denominator,
        visual=visual,
    )
    case_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "manual_selection": False,
        "population_samples": len(funnel),
        "categories": {
            category: {
                "eligible": int((cases["case_category"] == category).sum())
                if len(cases)
                else 0,
                "selected_sample_ids": cases.loc[
                    cases["case_category"].eq(category), "sample_id"
                ].tolist()
                if len(cases)
                else [],
            }
            for category in CASE_CATEGORIES
        },
        "selection_table": _record(tables / "d1_case_selection.csv"),
        "boards": case_boards,
    }
    case_manifest["content_sha256"] = canonical_sha256(case_manifest)
    atomic_json(case_dir / "CASE_SELECTION_MANIFEST.json", case_manifest)
    figures = _plot_all(
        root / "15_figures", formal_table, funnel, q_table, evidence_tables
    )
    reports = _reports(
        root,
        formal_table,
        funnel,
        q_table,
        statistics_table,
        runtime_table,
    )
    sources = {
        "formal_lock": _record(root / "08_lock" / LOCK_NAME),
        "formal_execution": _record(root / "09_formal_test" / EXECUTION_NAME),
        "formal_manifest": _record(
            root / "09_formal_test" / "formal_test_manifest.json"
        ),
        "formal_bundle": _record(formal["bundle_path"]),
        "formal_outcomes": _record(formal["outcomes_path"]),
        "formal_metrics": _record(formal["metrics_path"]),
        "independent_recompute": _record(independent_path),
        "independent_per_sample": independent["artifacts"]["independent_per_sample"],
        "postformal_evidence": _record(root / EVIDENCE_RELATIVE_PATH),
        "opaque_visual_ground_truth": evidence["opaque_visual_ground_truth"],
        "runtime_manifests": evidence["runtime_manifests"],
        "statistics_config": formal["plan"]["components"]["statistics_config"],
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_test_execution_count": 1,
        "independent_recompute_status": "PASS",
        "failure_categories": list(FAILURE_CATEGORIES),
        "failure_category_counts": funnel["failure_category"].value_counts().to_dict(),
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": {
            "tables": {name: _record(tables / name) for name in TABLE_NAMES},
            "additional_tables": {
                name: _record(tables / name) for name in ADDITIONAL_TABLE_NAMES
            },
            "figures": figures,
            "reports": reports,
            "case_selection_manifest": _record(
                case_dir / "CASE_SELECTION_MANIFEST.json"
            ),
            "case_boards": case_boards,
            "statistics_manifest": {
                **_record(root / "10_statistics" / "STATISTICS_MANIFEST.json"),
                "artifacts": statistics_manifest["artifacts"],
            },
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    transition_pipeline_status(
        root,
        status="POSTFORMAL",
        first_incomplete_stage="PFINAL_FINALIZATION",
        formal_test_executed=True,
        test_candidate_labels_read=True,
        formal_test_execution_count=1,
    )
    return result


def _verify_access_event_outputs(events: Sequence[Mapping[str, Any]]) -> None:
    for index, event in enumerate(events):
        output = event.get("output_manifest")
        output_sha = event.get("output_manifest_sha256")
        if isinstance(output, str) and output:
            current = _record(output)
            if output_sha != current["sha256"]:
                raise RuntimeError(
                    f"D1 access event output manifest hash differs at event {index}"
                )
        elif isinstance(output, Mapping):
            _verified_path(output, name=f"D1 access output manifest {index}")
        for field in ("output_manifests", "outputs"):
            records = event.get(field)
            if not isinstance(records, Mapping):
                continue
            for name, record in records.items():
                if isinstance(record, Mapping):
                    _verified_path(
                        record, name=f"D1 access {field} {name} at event {index}"
                    )


def _access_log_check(
    root: Path, sample_count: int, formal: Mapping[str, Any]
) -> dict[str, Any]:
    path = root / "09_formal_test" / "test_access.log"
    payload = path.read_bytes()
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"D1 access event {line_number} is not an object")
        events.append(value)
    event_ids = [
        str(event["event_id"])
        for event in events
        if isinstance(event.get("event_id"), str) and event.get("event_id")
    ]
    duplicate_ids = len(event_ids) != len(set(event_ids))
    timestamps = pd.to_datetime(
        [event.get("timestamp_utc") for event in events], utc=True, errors="coerce"
    )
    timestamps_ordered = bool(
        len(timestamps) == len(events)
        and not timestamps.isna().any()
        and all(left <= right for left, right in zip(timestamps[:-1], timestamps[1:]))
    )
    prelock_path = _verified_path(
        formal["plan"]["components"]["prelock_readiness"],
        name="D1 access-audit prelock readiness",
    )
    prelock = _load_content(
        prelock_path, name="D1 access-audit prelock readiness", statuses=("PASS",)
    )
    prelock_record = prelock.get("sources", {}).get("access_log")
    if not isinstance(prelock_record, Mapping):
        raise RuntimeError("D1 prelock access-log binding is absent")
    if Path(str(prelock_record.get("path", ""))).resolve() != path.resolve():
        raise RuntimeError("D1 prelock access-log path differs")
    prelock_bytes = int(prelock_record.get("bytes", -1))
    if prelock_bytes < 1 or prelock_bytes > len(payload):
        raise RuntimeError("D1 prelock access-log byte boundary differs")
    prefix = payload[:prelock_bytes]
    if not prefix.endswith(b"\n") or hashlib.sha256(
        prefix
    ).hexdigest() != prelock_record.get("sha256"):
        raise RuntimeError("D1 prelock access-log immutable prefix differs")
    prefix_events = [
        json.loads(line) for line in prefix.decode("utf-8").splitlines() if line
    ]
    declared_stages = set(
        map(str, prelock.get("checks", {}).get("label_free_test_stage_inventory", []))
    )
    prefix_stages = {
        str(event.get("stage"))
        for event in prefix_events
        if event.get("event") == "prelock_label_free_test_stage"
    }
    stage_inventory_matches = not declared_stages or declared_stages == prefix_stages
    suffix_events = events[len(prefix_events) :]
    expected_suffix_names = (
        "d1_prelock_raw_test_ground_truth_hash_only",
        "d1_formal_test_exclusive_claim_created",
        "d1_raw_test_ground_truth_read_once",
        "d1_formal_test_execution_finalized",
        "d1_independent_recompute_raw_test_ground_truth_read",
        "d1_postformal_visual_ground_truth_read",
    )
    suffix_names = tuple(str(event.get("event", "")) for event in suffix_events)
    suffix_exact = suffix_names == expected_suffix_names
    _verify_access_event_outputs(events)
    primary = [
        event
        for event in events
        if event.get("event") == "d1_raw_test_ground_truth_read_once"
    ]
    independent = [
        event
        for event in events
        if event.get("event") == "d1_independent_recompute_raw_test_ground_truth_read"
    ]
    visual = [
        event
        for event in events
        if event.get("event") == "d1_postformal_visual_ground_truth_read"
    ]
    forbidden = [
        event
        for event in events
        if event.get("candidate_test_labels_read") is True
        or event.get("candidate_labels_opened_as_table") is True
        or event.get("selection_feedback_used") is True
    ]
    allowed_read_like = {
        "d1_test_ground_truth_hash_only_source_closure",
        "d1_test_visual_ground_truth_hash_only_source_closure",
        *expected_suffix_names,
    }
    unknown_read_like = [
        event
        for event in events
        if any(
            token in str(event.get("event", "")).lower()
            for token in ("ground_truth", "label_access", "label_read")
        )
        and event.get("event") not in allowed_read_like
    ]
    raw_record = formal["plan"].get("raw_test_ground_truth", {})
    plan_record = _record(formal["plan_path"])
    hash_only_binding = bool(
        suffix_events
        and suffix_events[0].get("raw_test_ground_truth_sha256")
        == raw_record.get("sha256")
        and suffix_events[0].get("formal_plan_sha256") == plan_record["sha256"]
    )
    passed = (
        len(primary) == 1
        and len(independent) == 1
        and len(visual) == 1
        and int(primary[0].get("row_count", -1)) == sample_count
        and int(independent[0].get("row_count", -1)) == sample_count
        and int(visual[0].get("row_count", -1)) == sample_count
        and visual[0].get("opened_after_complete_formal_claim") is True
        and visual[0].get("opened_after_independent_recompute_pass") is True
        and not duplicate_ids
        and timestamps_ordered
        and stage_inventory_matches
        and suffix_exact
        and hash_only_binding
        and not forbidden
        and not unknown_read_like
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "primary_raw_gt_reads": len(primary),
        "independent_raw_gt_reads": len(independent),
        "postformal_visual_gt_reads": len(visual),
        "forbidden_events": len(forbidden),
        "duplicate_event_ids": duplicate_ids,
        "timestamps_ordered": timestamps_ordered,
        "prelock_stage_inventory_matches": stage_inventory_matches,
        "postclaim_event_inventory_matches": suffix_exact,
        "unknown_read_like_events": len(unknown_read_like),
    }


def _ledger_check(root: Path) -> dict[str, Any]:
    ledger = root / "run_ledger.sqlite"
    if not ledger.is_file() or ledger.is_symlink():
        return {"status": "FAIL", "reason": "missing ledger"}
    with sqlite3.connect(f"file:{ledger.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT id, stage, substage, status, command, artifact_path, artifact_sha256 "
            "FROM stages ORDER BY id"
        ).fetchall()
    commands = root / "commands.log"
    expected = render_ledger_commands(ledger)
    required = {
        ("P14", "d1_formal_evaluation_plan"),
        ("P14", "d1_formal_test_lock"),
        ("P14", "d1_exactly_once_formal_test"),
        ("P17", "d1_independent_recompute"),
        ("P15", "d1_postformal_artifacts"),
        ("PFINAL", "d1_finalization_preflight"),
    }
    normalized = [
        {
            "id": int(row[0]),
            "stage": str(row[1]),
            "substage": str(row[2]).split("__cmd_", 1)[0],
            "status": str(row[3]),
            "command": str(row[4]),
            "artifact_path": str(row[5]),
            "artifact_sha256": str(row[6]),
        }
        for row in rows
    ]
    observed = {
        (row["stage"], row["substage"])
        for row in normalized
        if row["status"] == "COMPLETE"
    }
    lineage: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in normalized:
        lineage.setdefault((row["stage"], row["substage"]), []).append(row)
    unresolved_failed = []
    superseded_failed = 0
    for key, group in lineage.items():
        complete_ids = [row["id"] for row in group if row["status"] == "COMPLETE"]
        for row in group:
            if row["status"] != "FAILED":
                continue
            if any(identifier > row["id"] for identifier in complete_ids):
                superseded_failed += 1
            else:
                unresolved_failed.append(
                    {"stage": key[0], "substage": key[1], "id": row["id"]}
                )
    invalid_required_artifacts = []
    for stage, substage in required:
        candidates = [
            row
            for row in normalized
            if row["stage"] == stage
            and row["substage"] == substage
            and row["status"] == "COMPLETE"
        ]
        valid = False
        for row in candidates:
            artifact = Path(row["artifact_path"]).expanduser().resolve()
            if (
                row["command"].strip()
                and artifact.is_file()
                and not artifact.is_symlink()
                and sha256_file(artifact) == row["artifact_sha256"]
            ):
                valid = True
                break
        if not valid:
            invalid_required_artifacts.append((stage, substage))
    running = [
        {"stage": row["stage"], "substage": row["substage"], "id": row["id"]}
        for row in normalized
        if row["status"] == "RUNNING"
    ]
    passed = (
        bool(rows)
        and not running
        and not unresolved_failed
        and all(row["command"].strip() for row in normalized)
        and required.issubset(observed)
        and not invalid_required_artifacts
        and commands.is_file()
        and commands.read_text(encoding="utf-8") == expected
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "stage_rows": len(rows),
        "missing_required_stages": sorted(required - observed),
        "invalid_required_artifacts": sorted(invalid_required_artifacts),
        "failed_count": sum(row["status"] == "FAILED" for row in normalized),
        "superseded_failed_count": superseded_failed,
        "unresolved_failed": unresolved_failed,
        "running": running,
    }


def _inventory(root: Path, *, excluded: Iterable[Path]) -> list[dict[str, Any]]:
    excluded_set = {path.resolve() for path in excluded}
    rows = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() in excluded_set
            or path.name == ".DS_Store"
            or path.name == ".commands.log.lock"
            or path.name.endswith(("-wal", "-shm"))
            or (path.name.startswith(".") and path.name.endswith(".tmp"))
        ):
            continue
        rows.append(
            {
                "relative_path": str(path.resolve().relative_to(root)),
                **_record(path),
            }
        )
    return rows


def _required_paths(root: Path) -> list[Path]:
    return [
        root / POSTFORMAL_MANIFEST_RELATIVE_PATH,
        root / "14_case_selection" / "CASE_SELECTION_MANIFEST.json",
        root / "17_independent_recompute" / "recomputed_metrics.json",
        root / "17_independent_recompute" / "independent_per_sample.parquet",
        root / EVIDENCE_RELATIVE_PATH,
        *[root / "tables" / name for name in TABLE_NAMES],
        *[root / "tables" / name for name in ADDITIONAL_TABLE_NAMES],
        *[root / "15_figures" / name for name in FIGURE_NAMES],
        *[root / "16_reports" / name for name in REPORT_NAMES],
        root / "10_statistics" / "STATISTICS_MANIFEST.json",
        root / "10_statistics" / "paired_comparisons.csv",
        root / "10_statistics" / "bootstrap_replicates.parquet",
        root / "10_statistics" / "cochran_q.json",
        root / "14_case_selection" / "CASE_BOARD_QA.json",
        root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_AFTER.json",
    ]


def finalize_d1_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    lock_path = root / FINAL_LOCK_NAME
    digest_path = root / FINAL_LOCK_DIGEST_NAME
    complete_path = root / "COMPLETE"
    failed_path = root / "FINALIZATION_FAILED.json"
    if any(
        path.exists() for path in (lock_path, digest_path, complete_path, failed_path)
    ):
        raise FileExistsError("D1 terminal finalization state already exists")
    checks: dict[str, Any] = {}
    try:
        formal = _load_formal(root)
        independent_path, independent, _ = _load_independent(root)
        postformal = _load_content(
            root / POSTFORMAL_MANIFEST_RELATIVE_PATH,
            name="D1 postformal manifest",
            statuses=("COMPLETE",),
        )
        verify_artifact_records_recursive(
            postformal,
            name="D1 postformal final artifact inventory",
            require_at_least_one=True,
        )
        verify_artifact_records_recursive(
            independent,
            name="D1 independent recompute final artifact inventory",
            require_at_least_one=True,
        )
        checks["formal"] = {"status": "PASS"}
        checks["independent"] = {
            "status": "PASS" if independent.get("status") == "PASS" else "FAIL"
        }
        checks["access_log"] = _access_log_check(
            root, int(independent["sample_count"]), formal
        )
        checks["ledger_commands"] = _ledger_check(root)
        checks["source_immutability"] = _source_immutability_after(root, formal)
        missing = [str(path) for path in _required_paths(root) if not path.is_file()]
        checks["required_files"] = {
            "status": "PASS" if not missing else "FAIL",
            "missing": missing,
        }
        source_closure_record = formal["plan"].get("sources", {}).get("source_closure")
        source_closure_path = _verified_path(
            source_closure_record, name="D1 source closure"
        )
        checks["source_closure"] = {
            "status": "PASS",
            "record": _record(source_closure_path),
        }
        authority_path = _verified_path(
            formal["plan"]["components"]["source_run_authority"],
            name="three-route authority",
        )
        authority_sha = sha256_file(authority_path)
        checks["three_route_lock"] = {
            "status": "PASS"
            if authority_sha == EXPECTED_UNIFIED_FINAL_LOCK_SHA256
            else "FAIL",
            "observed_sha256": authority_sha,
            "expected_sha256": EXPECTED_UNIFIED_FINAL_LOCK_SHA256,
        }
        checks["formal_count"] = {
            "status": "PASS"
            if formal["execution"].get("execution_count") == 1
            and formal["manifest"].get("formal_test_execution_count") == 1
            else "FAIL"
        }
        checks["postformal_sources"] = {
            "status": "PASS"
            if postformal.get("source_signature_sha256")
            == canonical_sha256(postformal.get("sources"))
            else "FAIL"
        }
        pipeline_path = root / "pipeline_status.json"
        pipeline = _read_json(pipeline_path, name="D1 pipeline status")
        checks["pipeline_status"] = {
            "status": (
                "PASS"
                if pipeline.get("status") == "POSTFORMAL"
                and pipeline.get("formal_test_executed") is True
                and pipeline.get("test_candidate_labels_read") is True
                and int(pipeline.get("formal_test_execution_count", -1)) == 1
                else "FAIL"
            ),
            "record": _record(pipeline_path),
        }
        statistics = _load_content(
            root / "10_statistics" / "STATISTICS_MANIFEST.json",
            name="D1 statistics manifest",
            statuses=("COMPLETE",),
        )
        verify_artifact_records_recursive(
            statistics,
            name="D1 statistics artifact inventory",
            require_at_least_one=True,
        )
        checks["statistics"] = {
            "status": (
                "PASS"
                if statistics.get("bootstrap_iterations") == 10_000
                and statistics.get("source_signature_sha256")
                == canonical_sha256(statistics.get("sources"))
                else "FAIL"
            )
        }
        case_manifest = _load_content(
            root / "14_case_selection" / "CASE_SELECTION_MANIFEST.json",
            name="D1 case selection manifest",
            statuses=("COMPLETE",),
        )
        verify_artifact_records_recursive(
            case_manifest,
            name="D1 case-board inventory",
            require_at_least_one=True,
        )
        categories = case_manifest.get("categories")
        boards = case_manifest.get("boards", {}).get("categories")
        checks["case_boards"] = {
            "status": (
                "PASS"
                if isinstance(categories, Mapping)
                and set(categories) == set(CASE_CATEGORIES)
                and isinstance(boards, Mapping)
                and set(boards) == set(CASE_CATEGORIES)
                else "FAIL"
            )
        }
        if not all(value.get("status") == "PASS" for value in checks.values()):
            raise RuntimeError("D1 final readiness checks failed")
        excluded = [lock_path, digest_path, complete_path, failed_path]
        transition_pipeline_status(
            root,
            status="COMPLETE",
            first_incomplete_stage=None,
            formal_test_executed=True,
            test_candidate_labels_read=True,
            formal_test_execution_count=1,
        )
        root_manifest_path = root / "manifest.json"
        root_manifest = _read_json(root_manifest_path, name="D1 root manifest")
        root_manifest.update(
            {
                "status": "COMPLETE",
                "test_label_state": "FORMAL_TEST_COMPLETE",
                "formal_test_execution_count": 1,
            }
        )
        atomic_json(root_manifest_path, root_manifest)
        inventory = _inventory(root, excluded=excluded)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE",
            "formal_test_execution_count": 1,
            "independent_recompute": _record(independent_path),
            "postformal_manifest": _record(root / POSTFORMAL_MANIFEST_RELATIVE_PATH),
            "integrity_checks": checks,
            "inventory": inventory,
            "inventory_count": len(inventory),
            "inventory_sha256": canonical_sha256(inventory),
            "excluded_self_referential_files": [
                FINAL_LOCK_NAME,
                FINAL_LOCK_DIGEST_NAME,
                "COMPLETE",
                "FINALIZATION_FAILED.json",
            ],
            "inventory_exclusion_policy": [
                ".DS_Store",
                ".commands.log.lock",
                "sqlite_sidecars_-wal_-shm",
                "dot_prefixed_*.tmp",
            ],
        }
        payload["self_sha256"] = canonical_sha256(payload)
        _exclusive_json(lock_path, payload)
        lock_sha = sha256_file(lock_path)
        _exclusive_text(digest_path, lock_sha + "\n")
        _exclusive_text(
            complete_path, f"COMPLETE\n{FINAL_LOCK_NAME} sha256={lock_sha}\n"
        )
        verify_final_lock(root)
        return {
            "status": "COMPLETE",
            "final_lock_sha256": lock_sha,
            "inventory_count": len(inventory),
        }
    except Exception as error:
        root_manifest_path = root / "manifest.json"
        if not lock_path.exists() and root_manifest_path.is_file():
            try:
                transition_pipeline_status(
                    root,
                    status="FAILED",
                    first_incomplete_stage="PFINAL_FAILED",
                    formal_test_executed=True,
                    test_candidate_labels_read=True,
                    formal_test_execution_count=1,
                )
            except Exception:
                pass
            root_manifest = _read_json(root_manifest_path, name="D1 root manifest")
            root_manifest.update(
                {
                    "status": "FINALIZATION_FAILED",
                    "test_label_state": "FORMAL_TEST_COMPLETE",
                    "formal_test_execution_count": 1,
                }
            )
            atomic_json(root_manifest_path, root_manifest)
        failure = {
            "schema_version": 1,
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "checks": checks,
            "complete_written": complete_path.exists(),
            "final_lock_written": lock_path.exists(),
        }
        failure["content_sha256"] = canonical_sha256(failure)
        atomic_json(failed_path, failure)
        raise


def _exclusive_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _exclusive_text(path, json.dumps(dict(value), sort_keys=True, indent=2) + "\n")


def verify_final_lock(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    lock_path = root / FINAL_LOCK_NAME
    digest_path = root / FINAL_LOCK_DIGEST_NAME
    complete_path = root / "COMPLETE"
    lock_sha = sha256_file(lock_path)
    if digest_path.read_text(encoding="ascii").strip() != lock_sha:
        raise RuntimeError("D1 final-lock detached digest differs")
    lock = _read_json(lock_path, name="D1 final lock")
    unsigned = dict(lock)
    recorded = unsigned.pop("self_sha256", None)
    inventory = lock.get("inventory")
    if (
        lock.get("status") != "COMPLETE"
        or recorded != canonical_sha256(unsigned)
        or not isinstance(inventory, list)
        or lock.get("inventory_count") != len(inventory)
        or lock.get("inventory_sha256") != canonical_sha256(inventory)
    ):
        raise RuntimeError("D1 final-lock self/inventory contract differs")
    for row in inventory:
        if not isinstance(row, Mapping):
            raise RuntimeError("D1 final-lock inventory record is invalid")
        _same_record(row, _record(str(row.get("path", ""))), name="D1 final inventory")
    excluded = [
        lock_path,
        digest_path,
        complete_path,
        root / "FINALIZATION_FAILED.json",
    ]
    fresh_inventory = _inventory(root, excluded=excluded)
    if fresh_inventory != inventory:
        raise RuntimeError("D1 final-lock fresh exact inventory differs")
    expected_marker = f"COMPLETE\n{FINAL_LOCK_NAME} sha256={lock_sha}\n"
    if complete_path.read_text(encoding="utf-8") != expected_marker:
        raise RuntimeError("D1 COMPLETE marker binding differs")
    return lock
