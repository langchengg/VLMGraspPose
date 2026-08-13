"""Deterministic post-formal analysis, reporting, and integrity utilities.

This module is intentionally downstream of the exactly-once formal-Test
transaction.  It never trains, selects a method, changes a candidate, or
modifies a formal decision.  All reported numbers are derived from persisted
machine-readable artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import verify_artifact_records_recursive
from .contracts import forbidden_model_columns
from .hashing import canonical_sha256, sha256_file
from .lock import verify_formal_test_lock
from .test_access_guard import append_access_log


ROUTES = ("crog", "g1", "c1")
EXPECTED_SOURCE_RUN_HASH_NAMES = {
    "paired_manifest",
    "crog_candidates",
    "g1_candidates",
    "c1_candidates",
    "canonical_candidates_labels_locked",
    "main_results",
    "evaluator",
    "decoder",
    "source_hashes",
    "g1_selected",
    "c1_selected",
    "train_samples",
    "validation_samples",
    "test_samples",
}
ROUTE_GALLERY_QUOTAS = {
    "recovered": 12,
    "harmful": 12,
    "unchanged_wrong_top5_solvable": 12,
    "no_positive_pool": 8,
    "gate_prevented_harmful": 8,
    "gate_missed_recoverable": 8,
}
CROSS_ROUTE_GALLERY_QUOTAS = {
    "router_success": 10,
    "router_harmful_or_failure": 10,
    "modular_only_correct_while_crog_wrong": 10,
    "crog_correct_while_both_modular_wrong": 10,
}
FAILURE_DEFINITIONS = {
    "E0": "technical/no-output",
    "E1": "no positive in full candidate pool",
    "E2": "positive exists only below Top-5",
    "E3": "positive in Top-5, native Top-1 wrong",
    "E4": "native Top-1 correct",
    "E5": "final gated reranker recovered",
    "E6": "final gated reranker harmful",
    "E7": "wrong-to-wrong switch",
    "E8": "correct-to-correct switch",
    "E9": "gate abstained on an ungated-recoverable sample",
    "E10": "gate prevented an ungated harmful switch",
}
FIGURE_NAMES = (
    "oracle_at_k_curves",
    "j_at_1_delta_forest",
    "recovered_vs_harmful",
    "headroom_recovery_at_5",
    "loss_comparison",
    "encoder_comparison",
    "cumulative_feature_ablation",
    "leave_one_family_out_ablation",
    "risk_coverage",
    "calibration_reliability",
    "switch_rate_vs_net_gain",
    "tri_backend_complementarity_matrix",
    "eight_way_outcome_intersection",
    "runtime_vs_gain",
    "parameter_count_vs_gain",
    "attribution_bridge",
)
TABLE_NAMES = (
    "fair_baseline_oracle.csv",
    "evidence_track_comparison.csv",
    "feature_ablation.csv",
    "loss_comparison.csv",
    "encoder_comparison.csv",
    "gate_comparison.csv",
    "tri_backend_consensus.csv",
    "cross_route_router.csv",
    "union_pool.csv",
    "attribution_bridge.csv",
    "formal_test_primary.csv",
    "formal_test_all_predeclared.csv",
    "statistical_tests.csv",
    "runtime_complexity.csv",
    "failure_taxonomy.csv",
)
REPORT_NAMES = (
    "FINAL_SUMMARY_ZH.md",
    "FINAL_REPORT_EN.md",
    "METHODS_UNIFIED_RERANKING_EN.md",
    "RESULTS_UNIFIED_RERANKING_EN.md",
    "DISCUSSION_UNIFIED_RERANKING_EN.md",
    "LIMITATIONS_EN.md",
    "BASELINE_GAP_ATTRIBUTION.md",
    "MATERIAL_PASSPORT.md",
    "EXPERIMENT_CONCLUSION.json",
    "THESIS_READY_TABLES.tex",
    "THESIS_READY_FIGURES.md",
)


@dataclass(frozen=True)
class FormalBundle:
    run_dir: Path
    manifest: Mapping[str, Any]
    plan: Mapping[str, Any]
    systems: Mapping[str, Mapping[str, Any]]
    metrics: Mapping[str, Mapping[str, Any]]
    statistics: Mapping[str, Any]
    sample_manifest: pd.DataFrame
    per_sample: pd.DataFrame
    rankings: pd.DataFrame
    per_candidate_scores: pd.DataFrame
    bridge_per_candidate_scores: pd.DataFrame
    labels: pd.DataFrame
    candidate_pools: Mapping[str, pd.DataFrame]


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _verified_path(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or path.is_symlink():
        raise PermissionError(f"missing regular post-formal input: {label}")
    expected = record.get("sha256")
    if expected is not None and sha256_file(path) != expected:
        raise PermissionError(f"post-formal input hash mismatch: {label}")
    return path


_FORMAL_ARTIFACT_BINDINGS = {
    "metrics": ("formal_test_metrics", "formal_test_metrics.json"),
    "statistics": ("formal_test_statistics", "formal_test_statistics.json"),
    "per_sample": ("formal_test_per_sample", "formal_test_per_sample.parquet"),
    "realized_rankings": (
        "formal_test_realized_rankings",
        "formal_test_realized_rankings.parquet",
    ),
    "outcomes_wide": (
        "formal_test_outcomes_wide",
        "formal_test_outcomes_wide.parquet",
    ),
    "per_candidate_scores": ("per_candidate_scores", "per_candidate_scores.parquet"),
    "bridge_per_candidate_scores": (
        "bridge_per_candidate_scores",
        "bridge_per_candidate_scores.parquet",
    ),
    "per_sample_decisions": ("per_sample_decisions", "per_sample_decisions.parquet"),
}


def _exact_current_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _load_execution_bound_formal_manifest(
    root: Path, lock: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    """Load the one canonical formal manifest and prove the closed artifact set."""

    execution_path = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    execution = read_json(execution_path)
    if execution.get("status") != "COMPLETE" or execution.get("execution_count") != 1:
        raise PermissionError(
            "post-formal reporting requires one completed formal Test"
        )
    lock_path = root / "08_lock" / "FORMAL_TEST_LOCK.json"
    if execution.get("formal_lock_file_sha256") != sha256_file(
        lock_path
    ) or execution.get("formal_lock_self_sha256") != lock.get("self_sha256"):
        raise PermissionError("formal execution does not bind the current formal lock")
    execution_artifacts = execution.get("artifacts")
    expected_execution_keys = {
        "formal_test_manifest",
        *(value[0] for value in _FORMAL_ARTIFACT_BINDINGS.values()),
    }
    if not isinstance(execution_artifacts, Mapping) or set(execution_artifacts) != set(
        expected_execution_keys
    ):
        raise PermissionError("formal execution artifact inventory is not exact")
    manifest_path = root / "09_formal_test" / "formal_test_manifest.json"
    expected_manifest_record = _exact_current_record(manifest_path)
    if dict(execution_artifacts["formal_test_manifest"]) != expected_manifest_record:
        raise PermissionError(
            "formal execution redirects or does not hash-bind the canonical manifest"
        )
    formal = read_json(manifest_path)
    unsigned = dict(formal)
    recorded_content_sha256 = unsigned.pop("content_sha256", None)
    if recorded_content_sha256 != canonical_sha256(unsigned):
        raise PermissionError("formal-Test manifest content hash mismatch")
    if (
        formal.get("status") != "COMPLETE"
        or formal.get("formal_test_execution_count") != 1
    ):
        raise PermissionError("formal-Test result manifest is incomplete")
    formal_artifacts = formal.get("artifacts")
    if not isinstance(formal_artifacts, Mapping) or set(formal_artifacts) != set(
        _FORMAL_ARTIFACT_BINDINGS
    ):
        raise PermissionError("formal-Test manifest artifact inventory is not exact")
    verified: dict[str, Path] = {}
    for manifest_key, (execution_key, filename) in _FORMAL_ARTIFACT_BINDINGS.items():
        canonical_path = root / "09_formal_test" / filename
        expected_record = _exact_current_record(canonical_path)
        if (
            dict(formal_artifacts[manifest_key]) != expected_record
            or dict(execution_artifacts[execution_key]) != expected_record
        ):
            raise PermissionError(
                f"formal artifact record mismatch or redirect: {manifest_key}"
            )
        verified[manifest_key] = canonical_path.resolve()
    return execution, formal, verified


def load_formal_bundle(run_dir: Path) -> FormalBundle:
    """Load and hash-check the completed formal transaction."""

    root = run_dir.resolve()
    lock = verify_formal_test_lock(root)
    _execution, formal, artifacts = _load_execution_bound_formal_manifest(root, lock)
    formal_lock = dict(formal.get("formal_lock", {}))
    lock_path = root / "08_lock" / "FORMAL_TEST_LOCK.json"
    if (
        Path(str(formal_lock.get("path", ""))).resolve() != lock_path.resolve()
        or formal_lock.get("file_sha256") != sha256_file(lock_path)
        or formal_lock.get("self_sha256") != lock.get("self_sha256")
    ):
        raise PermissionError("formal result does not bind the verified formal lock")
    plan_path = _verified_path(formal["evaluation_plan"], "evaluation_plan")
    sample_path = _verified_path(formal["sample_manifest"], "sample_manifest")
    plan = read_json(plan_path)
    systems = {str(row["name"]): dict(row) for row in plan["systems"]}
    metrics = read_json(artifacts["metrics"])["systems"]
    statistics = read_json(artifacts["statistics"])
    if set(metrics) != set(systems):
        raise ValueError("formal metric and system inventories differ")
    sample_manifest = pd.read_parquet(sample_path)
    per_sample = pd.read_parquet(artifacts["per_sample"])
    rankings = pd.read_parquet(artifacts["realized_rankings"])
    per_candidate_scores = pd.read_parquet(artifacts["per_candidate_scores"])
    bridge_per_candidate_scores = pd.read_parquet(
        artifacts["bridge_per_candidate_scores"]
    )
    required_scores = {
        "system_name",
        "system_kind",
        "route",
        "sample_id",
        "candidate_id",
        "rank",
        "formal_score",
        "candidate_success",
    }
    missing_scores = sorted(required_scores.difference(per_candidate_scores.columns))
    if missing_scores:
        raise ValueError(
            f"formal per-candidate score/label bundle misses columns: {missing_scores}"
        )
    required_bridge_scores = {
        "route",
        "candidate_pool_contract",
        "sample_id",
        "candidate_id",
        "source_candidate_id",
        "raw_candidate_id",
        "native_rank",
        "candidate_geometry_sha256",
        "fair_native_selector_score",
        "historical_selector_score",
        "candidate_success",
        "evaluator_sha256",
    }
    missing_bridge_scores = sorted(
        required_bridge_scores.difference(bridge_per_candidate_scores.columns)
    )
    if missing_bridge_scores:
        raise ValueError(
            "formal bridge per-candidate bundle misses columns: "
            f"{missing_bridge_scores}"
        )

    label_manifest_path = Path(str(plan["candidate_label_manifest"])).resolve()
    locked_label_manifest = dict(lock.get("locked_files", {})).get(
        "formal_candidate_label_manifest", {}
    )
    if label_manifest_path != Path(
        str(locked_label_manifest.get("path", ""))
    ).resolve() or sha256_file(label_manifest_path) != locked_label_manifest.get(
        "sha256"
    ):
        raise PermissionError(
            "candidate-label manifest is not bound by the formal lock"
        )
    label_manifest = read_json(label_manifest_path)
    labels_path = Path(str(label_manifest["candidate_labels_path"])).resolve()
    if sha256_file(labels_path) != label_manifest["candidate_labels_sha256"]:
        raise PermissionError("post-formal candidate-label hash mismatch")
    candidate_pools: dict[str, pd.DataFrame] = {}
    for route in ROUTES:
        record = label_manifest["candidate_pools"][route]
        path = _verified_path(record, f"candidate_pool/{route}")
        pool = pd.read_parquet(path)
        required = {
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_geometry_sha256",
        }
        if not required.issubset(pool.columns):
            raise ValueError(f"{route} All pool lacks canonical identity columns")
        pool = pool.copy()
        pool["sample_id"] = pool["sample_id"].astype(str)
        pool["candidate_id"] = pool["candidate_id"].astype(str)
        candidate_pools[route] = pool
    raw_labels = pd.read_parquet(labels_path)
    append_access_log(
        root,
        {
            "event": "postclaim_postformal_candidate_labels_read",
            "purpose": "P12-P14 deterministic failure analysis and galleries",
            "source_path": str(labels_path.resolve()),
            "source_sha256": sha256_file(labels_path),
            "rows": int(len(raw_labels)),
            "formal_test_execution_count": 1,
        },
    )
    labels = normalize_candidate_labels(
        raw_labels,
        label_manifest=label_manifest,
        sample_ids=set(sample_manifest["sample_id"].astype(str)),
        candidate_pools=candidate_pools,
    )
    return FormalBundle(
        run_dir=root,
        manifest=formal,
        plan=plan,
        systems=systems,
        metrics=metrics,
        statistics=statistics,
        sample_manifest=sample_manifest,
        per_sample=per_sample,
        rankings=rankings,
        per_candidate_scores=per_candidate_scores,
        bridge_per_candidate_scores=bridge_per_candidate_scores,
        labels=labels,
        candidate_pools=candidate_pools,
    )


def normalize_candidate_labels(
    frame: pd.DataFrame,
    *,
    label_manifest: Mapping[str, Any],
    sample_ids: set[str],
    candidate_pools: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Apply the exact route/variant normalization locked for formal Test.

    The fair canonical source uses ``method`` as the route and contains other
    variants.  Synthetic and already-normalized sources may use ``route``.
    The normalization declaration, not a guessed physical schema, is the
    source of truth.
    """

    normalization = dict(label_manifest.get("normalization", {}))
    route_column = str(normalization.get("route_column", "route"))
    variant_column = str(normalization.get("variant_column", "variant"))
    include_variants = normalization.get("include_variants")
    work = frame.copy()
    if include_variants is not None:
        if variant_column not in work.columns:
            raise ValueError("candidate labels miss the locked variant column")
        work = work.loc[
            work[variant_column].astype(str).isin(map(str, include_variants))
        ].copy()
    required = {route_column, "sample_id", "candidate_id", "candidate_success"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"candidate labels miss locked columns: {missing}")
    labels = work.copy()
    if route_column != "route" and "route" in labels.columns:
        labels = labels.drop(columns="route")
    labels = labels.rename(columns={route_column: "route"})
    labels["route"] = labels["route"].astype(str).str.lower()
    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    numeric = pd.to_numeric(labels["candidate_success"], errors="coerce")
    if (
        numeric.isna().any()
        or not numeric.isin([0, 1]).all()
        or not set(labels["route"]).issubset(ROUTES)
        or not set(labels["sample_id"]).issubset(sample_ids)
        or labels.duplicated(["route", "sample_id", "candidate_id"]).any()
    ):
        raise ValueError(
            "normalized candidate labels violate identity/binary contracts"
        )
    labels["candidate_success"] = numeric.astype(bool)
    observed = set(
        map(tuple, labels[["route", "sample_id", "candidate_id"]].to_numpy())
    )
    expected = {
        (route, str(row.sample_id), str(row.candidate_id))
        for route, pool in candidate_pools.items()
        for row in pool[["sample_id", "candidate_id"]].itertuples(index=False)
    }
    if observed != expected:
        raise ValueError(
            "normalized labels do not exactly cover locked All pools: "
            f"missing={len(expected.difference(observed))}, "
            f"extra={len(observed.difference(expected))}"
        )
    return labels


def _route_system(bundle: FormalBundle, route: str, kind: str) -> str:
    names = [
        name
        for name, system in bundle.systems.items()
        if str(system.get("route", "")).lower() == route and system.get("kind") == kind
    ]
    if len(names) != 1:
        raise ValueError(f"expected one {route}/{kind} formal system, found {names}")
    return names[0]


def _decision_frame(bundle: FormalBundle, system_name: str) -> pd.DataFrame:
    columns = {
        "sample_id",
        "selected_route",
        "selected_candidate_id",
        "selected_correct",
    }
    frame = bundle.per_sample.loc[
        bundle.per_sample["system_name"].astype(str).eq(system_name)
    ].copy()
    if not columns.issubset(frame.columns) or len(frame) != len(bundle.sample_manifest):
        raise ValueError(f"incomplete formal per-sample decisions: {system_name}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate formal per-sample decisions: {system_name}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["selected_candidate_id"] = (
        frame["selected_candidate_id"].fillna("").astype(str)
    )
    frame["selected_correct"] = frame["selected_correct"].astype(bool)
    return frame


def classify_failures(bundle: FormalBundle) -> pd.DataFrame:
    """Return one route/sample row with exhaustive E0--E10 predicates.

    E0--E4 describe candidate/baseline state, while E5--E10 describe final and
    counterfactual switch outcomes.  Because those meanings overlap, all
    predicates are retained and ``primary_category`` supplies one deterministic
    leaf for aggregation.
    """

    denominator = bundle.sample_manifest[["sample_id", "scene_id", "frame_id"]].copy()
    denominator["sample_id"] = denominator["sample_id"].astype(str)
    rows: list[pd.DataFrame] = []
    for route in ROUTES:
        native_name = _route_system(bundle, route, "native")
        ungated_name = _route_system(bundle, route, "ungated")
        gated_name = _route_system(bundle, route, "gated")
        native = _decision_frame(bundle, native_name)
        ungated = _decision_frame(bundle, ungated_name)
        gated = _decision_frame(bundle, gated_name)
        base = denominator.merge(
            native[["sample_id", "selected_candidate_id", "selected_correct"]].rename(
                columns={
                    "selected_candidate_id": "native_candidate_id",
                    "selected_correct": "native_correct",
                }
            ),
            on="sample_id",
            validate="one_to_one",
        )
        base = base.merge(
            ungated[["sample_id", "selected_candidate_id", "selected_correct"]].rename(
                columns={
                    "selected_candidate_id": "ungated_candidate_id",
                    "selected_correct": "ungated_correct",
                }
            ),
            on="sample_id",
            validate="one_to_one",
        ).merge(
            gated[["sample_id", "selected_candidate_id", "selected_correct"]].rename(
                columns={
                    "selected_candidate_id": "gated_candidate_id",
                    "selected_correct": "gated_correct",
                }
            ),
            on="sample_id",
            validate="one_to_one",
        )
        labels = bundle.labels.loc[bundle.labels["route"].eq(route)]
        full_positive = labels.groupby("sample_id")["candidate_success"].any()
        full_count = bundle.candidate_pools[route].groupby("sample_id").size()
        native_ranking = bundle.rankings.loc[
            bundle.rankings["system_name"].astype(str).eq(native_name),
            ["sample_id", "candidate_id", "rank"],
        ].merge(
            labels[["sample_id", "candidate_id", "candidate_success"]],
            on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        if native_ranking["candidate_success"].isna().any():
            raise ValueError(
                f"{route} native ranking references an unlabeled candidate"
            )
        top5_positive = native_ranking.groupby("sample_id")["candidate_success"].any()
        top5_count = native_ranking.groupby("sample_id").size()
        positive_ranks = (
            native_ranking.loc[native_ranking["candidate_success"]]
            .groupby("sample_id")["rank"]
            .min()
        )
        base["route"] = route
        base["candidate_count_all"] = (
            base["sample_id"].map(full_count).fillna(0).astype(int)
        )
        base["candidate_count_top5"] = (
            base["sample_id"].map(top5_count).fillna(0).astype(int)
        )
        base["full_pool_positive"] = (
            base["sample_id"].map(full_positive).fillna(False).astype(bool)
        )
        base["top5_positive"] = (
            base["sample_id"].map(top5_positive).fillna(False).astype(bool)
        )
        base["first_positive_rank"] = base["sample_id"].map(positive_ranks)
        native_changed_ungated = base["native_candidate_id"].ne(
            base["ungated_candidate_id"]
        )
        native_changed_gated = base["native_candidate_id"].ne(
            base["gated_candidate_id"]
        )
        base["E0"] = base["candidate_count_all"].eq(0)
        base["E1"] = base["candidate_count_all"].gt(0) & ~base["full_pool_positive"]
        base["E2"] = base["full_pool_positive"] & ~base["top5_positive"]
        base["E3"] = base["top5_positive"] & ~base["native_correct"]
        base["E4"] = base["native_correct"]
        base["E5"] = ~base["native_correct"] & base["gated_correct"]
        base["E6"] = base["native_correct"] & ~base["gated_correct"]
        base["E7"] = (
            ~base["native_correct"] & ~base["gated_correct"] & native_changed_gated
        )
        base["E8"] = (
            base["native_correct"] & base["gated_correct"] & native_changed_gated
        )
        base["E9"] = (
            ~base["native_correct"] & base["ungated_correct"] & ~base["gated_correct"]
        )
        base["E10"] = (
            base["native_correct"] & ~base["ungated_correct"] & base["gated_correct"]
        )
        base["ungated_recovered"] = ~base["native_correct"] & base["ungated_correct"]
        base["ungated_harmful"] = base["native_correct"] & ~base["ungated_correct"]
        base["gated_recovered"] = base["E5"]
        base["gated_harmful"] = base["E6"]
        base["ungated_switched"] = native_changed_ungated
        base["gated_switched"] = native_changed_gated
        priority = ("E0", "E1", "E2", "E9", "E10", "E5", "E6", "E7", "E8", "E3", "E4")
        predicate = base[list(priority)].to_numpy(bool)
        if not predicate.any(axis=1).all():
            raise AssertionError(f"{route} failure taxonomy left samples unclassified")
        first = predicate.argmax(axis=1)
        base["primary_category"] = [priority[index] for index in first]
        base["primary_definition"] = base["primary_category"].map(FAILURE_DEFINITIONS)
        rows.append(base)
    output = pd.concat(rows, ignore_index=True)
    if len(output) != len(bundle.sample_manifest) * len(ROUTES):
        raise AssertionError(
            "failure taxonomy did not preserve route/sample denominator"
        )
    return output


def gallery_category_members(
    taxonomy: pd.DataFrame, bundle: FormalBundle
) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for route in ROUTES:
        frame = taxonomy.loc[taxonomy["route"].eq(route)].copy()
        masks = {
            "recovered": frame["gated_recovered"],
            "harmful": frame["gated_harmful"],
            "unchanged_wrong_top5_solvable": (
                ~frame["gated_correct"]
                & frame["top5_positive"]
                & ~frame["gated_switched"]
            ),
            "no_positive_pool": ~frame["full_pool_positive"],
            "gate_prevented_harmful": frame["E10"],
            "gate_missed_recoverable": frame["E9"],
        }
        for category, mask in masks.items():
            result[(route, category)] = (
                frame.loc[mask, "sample_id"].astype(str).tolist()
            )

    final_by_route = {
        route: taxonomy.loc[taxonomy["route"].eq(route)].set_index("sample_id")[
            "gated_correct"
        ]
        for route in ROUTES
    }
    router_names = [
        name
        for name, system in bundle.systems.items()
        if system.get("kind") == "router"
    ]
    if router_names:
        router = _decision_frame(bundle, router_names[0]).set_index("sample_id")
        index = final_by_route["crog"].index
        crog = final_by_route["crog"].reindex(index).astype(bool)
        g1 = final_by_route["g1"].reindex(index).astype(bool)
        c1 = final_by_route["c1"].reindex(index).astype(bool)
        routed = router["selected_correct"].reindex(index).astype(bool)
        cross = {
            "router_success": ~crog & routed,
            "router_harmful_or_failure": ~routed,
            "modular_only_correct_while_crog_wrong": ~crog & (g1 | c1),
            "crog_correct_while_both_modular_wrong": crog & ~g1 & ~c1,
        }
        for category, mask in cross.items():
            result[("cross_route", category)] = index[mask].astype(str).tolist()
    else:
        for category in CROSS_ROUTE_GALLERY_QUOTAS:
            result[("cross_route", category)] = []
    return result


def deterministic_case_selection(
    sample_ids: Iterable[object], category: str, quota: int
) -> list[str]:
    """Select cases by SHA256(sample_id + category), never presentation appeal."""

    unique = sorted(set(map(str, sample_ids)))
    ranked = sorted(
        unique,
        key=lambda sample_id: (
            hashlib.sha256((sample_id + str(category)).encode("utf-8")).hexdigest(),
            sample_id,
        ),
    )
    return ranked[: int(quota)]


def _comparison_row(
    bundle: FormalBundle, name: str, system: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = dict(bundle.metrics[name])
    comparison = dict(bundle.statistics.get("comparisons", {}).get(name, {}))
    selection = dict(comparison.get("selection_comparison", {}))
    scene = dict(comparison.get("scene_bootstrap", {}))
    frame = dict(comparison.get("frame_bootstrap_sensitivity", {}))
    mcnemar = dict(comparison.get("mcnemar_conventional_supportive", {}))
    ci = scene.get("ci95", [None, None])
    frame_ci = frame.get("ci95", [None, None])
    row = {
        "system_name": name,
        "system_kind": system.get("kind"),
        "route": system.get("route"),
        "reference": comparison.get("reference"),
        "hypothesis_family": comparison.get("hypothesis_family"),
        "sample_count": metrics.get("sample_count"),
        **{f"j_at_{k}": metrics.get(f"j_at_{k}") for k in range(1, 16)},
        "oracle_at_5": metrics.get("oracle_at_5"),
        "oracle_at_15": metrics.get("oracle_at_15"),
        "oracle_all": metrics.get("oracle_all"),
        "mrr_at_5": metrics.get("mrr_at_5"),
        "mrr_at_15": metrics.get("mrr_at_15"),
        "ndcg_at_1": metrics.get("ndcg_at_1"),
        "ndcg_at_5": metrics.get("ndcg_at_5"),
        "ndcg_at_15": metrics.get("ndcg_at_15"),
        "delta_j_at_1": selection.get("delta_j_at_1"),
        "delta_percentage_points": None
        if selection.get("delta_j_at_1") is None
        else 100.0 * float(selection["delta_j_at_1"]),
        "recovered": selection.get("recovered"),
        "harmful": selection.get("harmful"),
        "net": selection.get("net"),
        "switch_count": selection.get("switch_count"),
        "switch_rate": selection.get("switch_rate"),
        "outcome_changing_precision": selection.get("outcome_changing_precision"),
        "headroom_recovery_at_5": selection.get("headroom_recovery_at_5"),
        "headroom_recovery_at_15": selection.get(
            "headroom_recovery_at_15", metrics.get("headroom_recovery_at_15")
        ),
        "scene_ci95_lower": ci[0] if len(ci) == 2 else None,
        "scene_ci95_upper": ci[1] if len(ci) == 2 else None,
        "frame_ci95_lower": frame_ci[0] if len(frame_ci) == 2 else None,
        "frame_ci95_upper": frame_ci[1] if len(frame_ci) == 2 else None,
        "mcnemar_p": mcnemar.get("pvalue"),
        "holm_p": comparison.get("holm_adjusted_mcnemar_pvalue"),
        "discordant": mcnemar.get("discordant"),
    }
    # Preserve any future formal metric rather than narrowing the schema in
    # the reporting layer (notably union J@6--15 and @15 summaries).
    for key, value in metrics.items():
        row.setdefault(str(key), value)
    return row


def formal_results_table(bundle: FormalBundle) -> pd.DataFrame:
    rows = [
        _comparison_row(bundle, name, system) for name, system in bundle.systems.items()
    ]
    return pd.DataFrame(rows)


def deployment_decision(
    row: Mapping[str, Any],
    *,
    independent_pass: bool,
    integrity_pass: bool,
    validation_gate_decision: str | None = None,
) -> tuple[str, str]:
    """Apply the predeclared GO/CAUTION/NO-GO rule without post-Test tuning."""

    delta = row.get("delta_j_at_1")
    lower = row.get("scene_ci95_lower")
    holm = row.get("holm_p")
    if not independent_pass or not integrity_pass:
        return "NO-GO", "independent recompute or integrity gate failed"
    if validation_gate_decision is not None and str(
        validation_gate_decision
    ).upper().replace("-", "_") in {"NO_GO", "NO_GO_NATIVE", "NO_GO_CROG"}:
        return "NO-GO", "predeclared Validation gate found no positive lower bound"
    if delta is None:
        return "NO-GO", "no paired formal comparison is available"
    if float(delta) <= 0:
        return "NO-GO", "formal net J@1 gain is non-positive"
    if (
        lower is not None
        and float(lower) > 0
        and holm is not None
        and float(holm) < 0.05
    ):
        return (
            "GO",
            "positive gain, positive scene-cluster lower bound, and Holm p < 0.05",
        )
    return (
        "CAUTION",
        "mean gain is positive but confirmatory uncertainty/significance is incomplete",
    )


def taxonomy_summary(taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route, group in taxonomy.groupby("route", sort=True):
        denominator = len(group)
        for code, definition in FAILURE_DEFINITIONS.items():
            count = int(group[code].sum())
            rows.append(
                {
                    "route": route,
                    "category": code,
                    "definition": definition,
                    "count": count,
                    "denominator": denominator,
                    "fraction": count / denominator if denominator else None,
                    "predicate_is_overlapping": code not in {"E0", "E1", "E2"},
                }
            )
        for code, count in (
            group["primary_category"].value_counts().sort_index().items()
        ):
            rows.append(
                {
                    "route": route,
                    "category": f"PRIMARY_{code}",
                    "definition": f"exclusive priority leaf: {FAILURE_DEFINITIONS[code]}",
                    "count": int(count),
                    "denominator": denominator,
                    "fraction": int(count) / denominator if denominator else None,
                    "predicate_is_overlapping": False,
                }
            )
    return pd.DataFrame(rows)


def _current_source_hash_checks(run_dir: Path) -> tuple[bool, list[dict[str, Any]]]:
    path = run_dir / "00_audit" / "source_run_hashes.json"
    if not path.is_file():
        return False, [{"name": "source_run_hashes", "status": "MISSING"}]
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        return False, [{"name": "source_run_hashes", "status": "MALFORMED"}]
    rows: list[dict[str, Any]] = []
    observed_names = set(map(str, payload))
    for name in sorted(EXPECTED_SOURCE_RUN_HASH_NAMES | observed_names):
        record = payload.get(name)
        if (
            not isinstance(record, Mapping)
            or "path" not in record
            or "sha256" not in record
        ):
            rows.append(
                {
                    "name": name,
                    "status": "MISSING" if name not in payload else "MALFORMED",
                }
            )
            continue
        source = Path(str(record["path"])).resolve()
        status = (
            "PASS"
            if source.is_file()
            and not source.is_symlink()
            and sha256_file(source) == record["sha256"]
            and (
                "bytes" not in record
                or isinstance(record.get("bytes"), int)
                and source.stat().st_size == record["bytes"]
            )
            else "FAIL"
        )
        rows.append({"name": name, "status": status, "path": str(source)})
    exact_inventory = observed_names == EXPECTED_SOURCE_RUN_HASH_NAMES
    return exact_inventory and all(row["status"] == "PASS" for row in rows), rows


def _feature_leakage_check(run_dir: Path) -> tuple[bool, list[str]]:
    violations: list[str] = []
    prelock_path = run_dir / "08_lock" / "prelock_assembly_manifest.json"
    try:
        prelock = read_json(prelock_path)
        if prelock.get("status") != "COMPLETE" or prelock.get("stage") != "P11_PRELOCK":
            raise RuntimeError("P11 prelock assembly is not COMPLETE")
        sources = prelock.get("sources")
        if not isinstance(sources, Mapping):
            raise RuntimeError("P11 source inventory is malformed")
        expected_names = {f"{route}_feature_manifest_record" for route in ROUTES}
        union_expected = "union_test_features" in sources
        observed_names = {
            str(name)
            for name in sources
            if str(name).endswith("_feature_manifest_record")
            or str(name) == "union_test_features"
        }
        expected_inventory = expected_names | (
            {"union_test_features"} if union_expected else set()
        )
        if observed_names != expected_inventory:
            raise RuntimeError(
                "P11 consumed feature-manifest inventory is not exact: "
                f"{sorted(observed_names)}"
            )

        queue: list[tuple[str, Mapping[str, Any]]] = [
            (name, sources[name]) for name in sorted(expected_inventory)
        ]
        seen: set[Path] = set()
        while queue:
            name, record = queue.pop(0)
            if not isinstance(record, Mapping):
                raise RuntimeError(f"{name} is not an artifact record")
            verified = verify_artifact_records_recursive(
                record,
                name=f"consumed feature manifest {name}",
                require_at_least_one=True,
            )
            path = Path(verified[0]["path"])
            if path in seen:
                continue
            seen.add(path)
            if path.suffix.lower() != ".json":
                raise RuntimeError(f"consumed feature manifest is not JSON: {path}")
            payload = read_json(path)
            columns_value = payload.get("model_feature_columns")
            if columns_value is not None:
                if not isinstance(columns_value, list) or not columns_value:
                    violations.append(f"{path}:model_feature_columns malformed")
                else:
                    columns = list(map(str, columns_value))
                    for column in forbidden_model_columns(columns):
                        violations.append(f"{path}:{column}")
                    schema = payload.get(
                        "model_feature_schema_sha256",
                        payload.get("feature_schema_sha256"),
                    )
                    if schema != canonical_sha256(columns):
                        violations.append(f"{path}:feature schema hash mismatch")
            if payload.get("labels_physically_separate") is False:
                violations.append(f"{path}:labels_physically_separate=false")
            dependency_records = 0

            def verify_dependency_records(node: Any, location: str) -> None:
                nonlocal dependency_records
                if isinstance(node, Mapping):
                    if "path" in node or "sha256" in node:
                        dependency_records += 1
                        raw_path = Path(str(node.get("path", ""))).resolve()
                        digest = node.get("sha256")
                        if raw_path.suffix.lower() == ".py":
                            if (
                                not raw_path.is_file()
                                or raw_path.is_symlink()
                                or not isinstance(digest, str)
                                or len(digest) != 64
                                or any(
                                    character not in "0123456789abcdef"
                                    for character in digest.lower()
                                )
                            ):
                                raise RuntimeError(
                                    f"{location} has a malformed source-code record"
                                )
                            return
                        verify_artifact_records_recursive(
                            node,
                            name=location,
                            require_at_least_one=True,
                        )
                        return
                    for key, value in node.items():
                        verify_dependency_records(value, f"{location}.{key}")
                elif isinstance(node, Sequence) and not isinstance(
                    node, (str, bytes, bytearray)
                ):
                    for index, value in enumerate(node):
                        verify_dependency_records(value, f"{location}[{index}]")

            verify_dependency_records(
                {
                    "sources": payload.get("sources"),
                    "artifacts": payload.get("artifacts"),
                    "artifact": payload.get("artifact"),
                },
                f"consumed feature dependency {path}",
            )
            if dependency_records == 0:
                raise RuntimeError(
                    f"consumed feature manifest has no dependencies: {path}"
                )

            def collect_json_records(node: Any, location: str) -> None:
                if isinstance(node, Mapping):
                    if "path" in node or "sha256" in node:
                        raw_path = node.get("path")
                        if (
                            isinstance(raw_path, str)
                            and Path(raw_path).suffix.lower() == ".json"
                        ):
                            queue.append((location, node))
                        return
                    for key, value in node.items():
                        collect_json_records(value, f"{location}.{key}")
                elif isinstance(node, Sequence) and not isinstance(
                    node, (str, bytes, bytearray)
                ):
                    for index, value in enumerate(node):
                        collect_json_records(value, f"{location}[{index}]")

            collect_json_records(
                {
                    "sources": payload.get("sources"),
                    "artifacts": payload.get("artifacts"),
                    "artifact": payload.get("artifact"),
                },
                str(path),
            )
        if not seen:
            violations.append("no consumed feature manifests")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        violations.append(str(error))
    return not violations, violations


def _native_score_source_check(bundle: FormalBundle) -> tuple[bool, list[str]]:
    """Compare canonical Test native scores/order with the audited source rows."""

    errors: list[str] = []
    try:
        audited = read_json(bundle.run_dir / "00_audit" / "source_run_hashes.json")
        if not isinstance(audited, Mapping):
            raise RuntimeError("audited source inventory is malformed")
        source_names = {
            "crog": "crog_candidates",
            "g1": "g1_candidates",
            "c1": "c1_candidates",
        }
        for route, source_name in source_names.items():
            record = audited.get(source_name)
            if not isinstance(record, Mapping):
                errors.append(f"{route}: audited candidate source missing")
                continue
            verified = verify_artifact_records_recursive(
                record,
                name=f"audited {route} candidate source",
                require_at_least_one=True,
            )
            source = pd.read_parquet(Path(verified[0]["path"]))
            aliases = {
                "q_raw": "native_score",
                "original_rank": "native_rank",
            }
            source = source.rename(
                columns={
                    old: new
                    for old, new in aliases.items()
                    if old in source and new not in source
                }
            )
            required = ("sample_id", "candidate_id", "native_rank", "native_score")
            if not set(required).issubset(source.columns):
                errors.append(f"{route}: audited source misses native identity/score")
                continue
            current = bundle.candidate_pools[route]
            if not set(required).issubset(current.columns):
                errors.append(f"{route}: canonical pool misses native identity/score")
                continue
            sample_ids = set(bundle.sample_manifest["sample_id"].astype(str))
            source = source.loc[source["sample_id"].astype(str).isin(sample_ids)].copy()
            left = current[list(required)].copy()
            right = source[list(required)].copy()
            for frame in (left, right):
                frame["sample_id"] = frame["sample_id"].astype(str)
                frame["candidate_id"] = frame["candidate_id"].astype(str)
                frame["native_rank"] = pd.to_numeric(
                    frame["native_rank"], errors="coerce"
                )
                frame["native_score"] = pd.to_numeric(
                    frame["native_score"], errors="coerce"
                )
            order = ["sample_id", "native_rank", "candidate_id"]
            left = left.sort_values(order, kind="mergesort").reset_index(drop=True)
            right = right.sort_values(order, kind="mergesort").reset_index(drop=True)
            identity_equal = left[["sample_id", "candidate_id"]].equals(
                right[["sample_id", "candidate_id"]]
            )
            if not identity_equal or not _exact_numeric_values_equal(
                left["native_rank"], right["native_rank"]
            ):
                errors.append(f"{route}: audited native candidate order differs")
                continue
            left_scores = left["native_score"].to_numpy(dtype=np.float64)
            right_scores = right["native_score"].to_numpy(dtype=np.float64)
            if (
                not np.isfinite(left_scores).all()
                or not np.isfinite(right_scores).all()
                or not np.array_equal(left_scores, right_scores)
            ):
                errors.append(f"{route}: audited native scores differ")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        errors.append(str(error))
    return not errors, errors


def _exact_numeric_values_equal(left: pd.Series, right: pd.Series) -> bool:
    """Compare numeric values exactly without treating storage dtype as semantics."""

    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    return bool(
        len(left_numeric) == len(right_numeric)
        and left_numeric.notna().all()
        and right_numeric.notna().all()
        and np.array_equal(
            left_numeric.to_numpy(dtype=np.float64),
            right_numeric.to_numpy(dtype=np.float64),
        )
    )


def shared_hifi_mask_check(run_dir: Path) -> tuple[bool, list[str]]:
    """Prove G1/C1 T2 Test features consumed identical, current HiFi masks."""

    paired_path = run_dir / "01_manifests" / "paired_test.parquet"
    if not paired_path.is_file():
        return False, ["paired Test manifest missing"]
    paired_sha = sha256_file(paired_path)
    paired = pd.read_parquet(paired_path)
    required_paired = {"sample_id", "predicted_mask_path", "predicted_mask_sha256"}
    if not required_paired.issubset(paired.columns):
        return False, ["paired Test manifest misses predicted mask identity columns"]
    paired_context = paired[list(required_paired)].copy()
    paired_context["sample_id"] = paired_context["sample_id"].astype(str)
    paired_context["predicted_mask_path"] = paired_context[
        "predicted_mask_path"
    ].astype(str)
    paired_context["predicted_mask_sha256"] = paired_context[
        "predicted_mask_sha256"
    ].astype(str)
    if paired_context["sample_id"].duplicated().any():
        return False, ["paired Test manifest has duplicate sample IDs"]
    errors: list[str] = []
    contexts: dict[str, pd.DataFrame] = {}
    for route in ("g1", "c1"):
        common_manifest_path = (
            run_dir
            / "03_features"
            / "common"
            / f"{route}_test"
            / "feature_manifest.json"
        )
        track_manifest_path = (
            run_dir
            / "03_features"
            / "tracks"
            / "T2_matched_common"
            / f"{route}_test"
            / "feature_manifest.json"
        )
        if not common_manifest_path.is_file() or not track_manifest_path.is_file():
            errors.append(f"{route}: common/T2 Test feature manifest missing")
            continue
        common = read_json(common_manifest_path)
        track = read_json(track_manifest_path)
        if (
            common.get("status") != "COMPLETE"
            or common.get("split") != "test"
            or str(common.get("route", "")).lower() != route
            or common.get("paired_manifest_sha256") != paired_sha
        ):
            errors.append(f"{route}: common feature manifest does not bind paired Test")
            continue
        artifacts = dict(common.get("artifacts", {}))
        feature_record = dict(artifacts.get("candidate_features", {}))
        context_record = dict(artifacts.get("sample_context", {}))
        try:
            feature_path = _verified_path(
                feature_record, f"{route}/common candidate features"
            )
            context_path = _verified_path(
                context_record, f"{route}/common sample context"
            )
        except (PermissionError, ValueError) as error:
            errors.append(str(error))
            continue
        track_sources = {
            (
                str(Path(str(record.get("path", ""))).resolve()),
                str(record.get("sha256", "")),
            )
            for record in track.get("sources", [])
            if isinstance(record, Mapping)
        }
        if (str(feature_path), feature_record.get("sha256")) not in track_sources:
            errors.append(
                f"{route}: T2 track does not consume hash-bound common features"
            )
            continue
        context = pd.read_parquet(context_path)
        if not required_paired.issubset(context.columns):
            errors.append(f"{route}: common sample context misses mask identities")
            continue
        context = context[list(required_paired)].copy()
        for column in context:
            context[column] = context[column].astype(str)
        context = context.sort_values("sample_id", kind="mergesort").reset_index(
            drop=True
        )
        if context["sample_id"].duplicated().any() or len(context) != len(
            paired_context
        ):
            errors.append(
                f"{route}: common sample context does not preserve denominator"
            )
            continue
        expected = paired_context.sort_values(
            "sample_id", kind="mergesort"
        ).reset_index(drop=True)
        if not context.equals(expected):
            errors.append(
                f"{route}: common sample mask identities differ from paired Test"
            )
            continue
        contexts[route] = context
    if set(contexts) == {"g1", "c1"} and not contexts["g1"].equals(contexts["c1"]):
        errors.append("G1/C1 common feature contexts contain divergent HiFi masks")
    if not errors and contexts:
        unique = contexts["g1"][
            ["predicted_mask_path", "predicted_mask_sha256"]
        ].drop_duplicates()
        for row in unique.itertuples(index=False):
            path = Path(str(row.predicted_mask_path))
            if not path.is_file() or path.is_symlink():
                errors.append(f"shared HiFi mask missing/non-regular: {path}")
            elif sha256_file(path) != str(row.predicted_mask_sha256):
                errors.append(f"shared HiFi mask byte hash mismatch: {path}")
    return not errors and set(contexts) == {"g1", "c1"}, errors


def _access_manifest_specifications(run_dir: Path) -> list[dict[str, Any]]:
    """Return the exact completed label-free Test manifest inventory."""

    specs: list[dict[str, Any]] = []

    def add(
        event: str,
        path: Path,
        *,
        stage: str | None = None,
        route: str | None = None,
        application_id: str | None = None,
    ) -> None:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise RuntimeError(f"required label-free Test manifest missing: {resolved}")
        specs.append(
            {
                "event": event,
                "stage": stage,
                "route": route,
                "application_id": application_id,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
            }
        )

    for route in ROUTES:
        add(
            "prelock_label_free_test_stage",
            run_dir
            / "03_features"
            / "common"
            / f"{route}_test"
            / "feature_manifest.json",
            stage=f"common_features_{route}_test_top5_formal",
        )
        add(
            "prelock_label_free_test_stage",
            run_dir / "03_features" / "rgb" / f"{route}_test" / "feature_manifest.json",
            stage=f"rgb_features_{route}_test",
        )
        for track in ("T1_native", "T2_matched_common"):
            add(
                "prelock_label_free_test_stage",
                run_dir
                / "03_features"
                / "tracks"
                / track
                / f"{route}_test"
                / "feature_manifest.json",
                stage=f"feature_track_{track}_{route}_test",
            )
        add(
            "prelock_label_free_test_calibration",
            run_dir / "05_calibration" / f"{route}_test_application_manifest.json",
            route=route,
        )
        add(
            "prelock_label_free_test_stage",
            run_dir / "08_lock" / "label_free_test_gates" / route / "manifest.json",
            stage="gate_test_application",
            route=route,
        )
    for route in ("g1", "c1"):
        add(
            "prelock_label_free_test_stage",
            run_dir
            / "03_features"
            / "backend_maps"
            / f"{route}_test"
            / "feature_manifest.json",
            stage=f"backend_maps_{route}_test",
        )
    add(
        "prelock_label_free_test_stage",
        run_dir / "03_features" / "tri_backend_dense" / "test" / "manifest.json",
        stage="tri_backend_dense_test",
    )
    add(
        "prelock_label_free_test_stage",
        run_dir / "03_features" / "consensus" / "test_manifest.json",
        stage="tri_backend_consensus_test",
    )

    prelock_path = run_dir / "08_lock" / "prelock_assembly_manifest.json"
    prelock = read_json(prelock_path)
    sources = prelock.get("sources")
    if prelock.get("status") != "COMPLETE" or not isinstance(sources, Mapping):
        raise RuntimeError("P11 source inventory is unavailable for access audit")
    for route in ROUTES:
        ranker_record = sources.get(f"{route}_ranker_manifest_record")
        if not isinstance(ranker_record, Mapping):
            raise RuntimeError(f"P11 misses {route} selected Test ranker manifest")
        ranker_path = _verified_path(ranker_record, f"{route} Test ranker manifest")
        ranker = read_json(ranker_path)
        applications = dict(ranker.get("sources", {})).get("applications")
        if not isinstance(applications, Mapping) or not applications:
            raise RuntimeError(f"{route} selected Test ranker applications are empty")
        for seed, record in sorted(applications.items()):
            application_path = _verified_path(
                record, f"{route} Test matrix application seed {seed}"
            )
            application = read_json(application_path)
            add(
                "prelock_label_free_test_stage",
                application_path,
                stage="matrix_cell_test_application",
                application_id=str(application.get("application_id", "")),
            )
    add(
        "prelock_label_free_test_stage",
        run_dir / "08_lock" / "label_free_test_rankers" / "manifest.json",
        stage="selected_ranker_test_ensemble_application",
    )
    add(
        "prelock_label_free_test_stage",
        run_dir / "08_lock" / "route_router_inputs" / "manifest.json",
        stage="router_test_input_preparation",
    )
    add(
        "prelock_label_free_test_stage",
        run_dir / "08_lock" / "route_router_test" / "manifest.json",
        stage="route_router_test_application",
    )

    union_names = {
        "union_test_features",
        "union_ranker_test",
        "union_ranker_selection",
    }
    present_union = union_names.intersection(sources)
    if present_union and present_union != union_names:
        raise RuntimeError("P11 union source inventory is partial")
    if present_union:
        add(
            "prelock_label_free_test_stage",
            _verified_path(sources["union_test_features"], "union Test features"),
            stage="union_test_feature_preparation",
        )
        add(
            "prelock_label_free_test_stage",
            _verified_path(sources["union_ranker_test"], "union Test application"),
            stage="union_ranker_test_application",
        )
    bridge_record = sources.get("test_bridge_manifest")
    bridge_path = _verified_path(bridge_record, "label-free Test bridge manifest")
    add("label_free_test_bridge_input", bridge_path)
    return specs


def _access_log_check(run_dir: Path) -> tuple[bool, list[str]]:
    path = run_dir / "09_formal_test" / "test_access.log"
    if not path.is_file():
        return False, ["test_access.log missing"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("access-log row is not an object")
            records.append(dict(value))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, [f"access-log-unreadable:{error}"]
    events = [str(value.get("event", "")) for value in records]
    try:
        claim_index = events.index("formal_test_exclusive_claim_created")
        read_index = events.index("candidate_test_labels_read_once")
    except ValueError:
        return False, events
    allowed_unbound_preclaim = {
        "prelock_audit",
        "prelock_candidate_test_label_hash_only",
        "prelock_historical_test_ground_truth_hash_only",
        "candidate_test_label_access_denied",
    }
    try:
        expected_specs = _access_manifest_specifications(run_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        return False, [*events, f"expected-access-inventory:{error}"]

    def event_matches(record: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
        if record.get("event") != spec["event"]:
            return False
        for key in ("stage", "route", "application_id"):
            expected = spec.get(key)
            if (
                expected is not None
                and str(record.get(key, "")).lower() != str(expected).lower()
            ):
                return False
        if spec["event"] == "prelock_label_free_test_calibration":
            return (
                record.get("candidate_labels_loaded") is False
                and record.get("allowed")
                == ["candidate_geometry", "native_score"]
            )
        return (
            Path(str(record.get("output_manifest", ""))).resolve()
            == Path(str(spec["path"]))
            and record.get("output_manifest_sha256") == spec["sha256"]
            and sha256_file(Path(str(spec["path"]))) == spec["sha256"]
        )

    bound_preclaim = records[:claim_index]
    for spec in expected_specs:
        matches = [record for record in bound_preclaim if event_matches(record, spec)]
        if not matches:
            errors.append(
                f"missing-bound-preclaim-stage:{spec.get('stage') or spec['event']}"
            )
    expected_preclaim_events = {str(spec["event"]) for spec in expected_specs}
    for record in bound_preclaim:
        event = str(record.get("event", ""))
        if event in allowed_unbound_preclaim:
            continue
        if event not in expected_preclaim_events:
            errors.append(f"unexpected-or-unbound-preclaim-event:{event}")
        if (
            record.get("candidate_labels_opened_as_table") is True
            or record.get("candidate_labels_loaded") is True
            or record.get("candidate_test_labels_read") is True
            or record.get("historical_test_ground_truth_rows_read") is True
        ):
            errors.append(f"preclaim-label-row-access:{event}")

    required_once = {
        "formal_test_exclusive_claim_created",
        "candidate_test_label_access_authorized",
        "candidate_test_labels_read_once",
        "formal_bridge_test_ground_truth_read_once",
        "formal_test_execution_finalized",
        "postclaim_independent_candidate_labels_read",
        "postclaim_independent_bridge_ground_truth_read",
    }
    for event in sorted(required_once):
        indexes = [index for index, value in enumerate(events) if value == event]
        if len(indexes) != 1:
            errors.append(f"event-count:{event}:{len(indexes)}")
    postformal_indexes = [
        index
        for index, value in enumerate(events)
        if value == "postclaim_postformal_candidate_labels_read"
    ]
    if not postformal_indexes:
        errors.append("event-count:postclaim_postformal_candidate_labels_read:0")
    order = [
        "formal_test_exclusive_claim_created",
        "candidate_test_label_access_authorized",
        "candidate_test_labels_read_once",
        "formal_bridge_test_ground_truth_read_once",
        "formal_test_execution_finalized",
        "postclaim_independent_candidate_labels_read",
        "postclaim_independent_bridge_ground_truth_read",
    ]
    if all(events.count(event) == 1 for event in order):
        positions = [events.index(event) for event in order]
        if positions != sorted(positions):
            errors.append("formal/postclaim access event order is invalid")
        if postformal_indexes and min(postformal_indexes) < positions[-1]:
            errors.append("postformal label reads precede independent recompute")

    try:
        lock = verify_formal_test_lock(run_dir)
        execution, formal, _ = _load_execution_bound_formal_manifest(run_dir, lock)
        plan = read_json(
            Path(str(lock["locked_files"]["formal_evaluation_plan"]["path"]))
        )
        label_manifest = read_json(Path(str(plan["candidate_label_manifest"])))
        label_path = Path(str(label_manifest["candidate_labels_path"])).resolve()
        label_sha = str(label_manifest["candidate_labels_sha256"])
        bridge = plan["test_bridge_contract"]["historical_ground_truth"]
        bridge_path = Path(str(bridge["path"])).resolve()
        bridge_sha = str(bridge["sha256"])
        manifest_path = run_dir / "09_formal_test" / "formal_test_manifest.json"
        for event in (
            "candidate_test_labels_read_once",
            "postclaim_independent_candidate_labels_read",
            "postclaim_postformal_candidate_labels_read",
        ):
            matches = [record for record in records if record.get("event") == event]
            for record in matches:
                raw_path = record.get("path", record.get("source_path"))
                raw_sha = record.get("sha256", record.get("source_sha256"))
                raw_rows = record.get("row_count", record.get("rows"))
                if (
                    Path(str(raw_path)).resolve() != label_path
                    or raw_sha != label_sha
                    or not isinstance(raw_rows, int)
                    or raw_rows <= 0
                ):
                    errors.append(f"label-read-binding:{event}")
        for event in (
            "formal_bridge_test_ground_truth_read_once",
            "postclaim_independent_bridge_ground_truth_read",
        ):
            matches = [record for record in records if record.get("event") == event]
            if len(matches) == 1:
                record = matches[0]
                raw_path = record.get("path", record.get("source_path"))
                raw_sha = record.get("sha256", record.get("source_sha256"))
                raw_rows = record.get("row_count", record.get("rows"))
                if (
                    Path(str(raw_path)).resolve() != bridge_path
                    or raw_sha != bridge_sha
                    or not isinstance(raw_rows, int)
                    or raw_rows <= 0
                ):
                    errors.append(f"bridge-read-binding:{event}")
        finalized = [
            record
            for record in records
            if record.get("event") == "formal_test_execution_finalized"
        ]
        if len(finalized) == 1 and (
            finalized[0].get("execution_count") != 1
            or finalized[0].get("manifest_sha256") != sha256_file(manifest_path)
            or execution.get("artifacts", {}).get("formal_test_manifest")
            != {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            }
            or formal.get("content_sha256") is None
        ):
            errors.append("formal-finalization-binding")
    except (
        OSError,
        ValueError,
        RuntimeError,
        PermissionError,
        json.JSONDecodeError,
    ) as error:
        errors.append(f"formal-access-binding:{error}")
    return read_index > claim_index and not errors, [*events, *errors]


def _independent_recompute_integrity(root: Path) -> tuple[bool, dict[str, Any]]:
    path = root / "15_independent_recompute" / "INDEPENDENT_RECOMPUTE.json"
    details: dict[str, Any] = {"path": str(path.resolve()), "errors": []}
    errors: list[str] = details["errors"]
    try:
        current_lock = verify_formal_test_lock(root)
        _load_execution_bound_formal_manifest(root, current_lock)
        independent = read_json(path)
        if not isinstance(independent, dict):
            raise ValueError("independent recompute is not a JSON object")
        unsigned = dict(independent)
        recorded = unsigned.pop("content_sha256", None)
        if recorded != canonical_sha256(unsigned):
            errors.append("content_hash_mismatch")
        if independent.get("status") != "PASS":
            errors.append("status_not_pass")
        if independent.get("formal_test_execution_count") != 1:
            errors.append("formal_execution_count_not_one")
        for field in (
            "metric_checks",
            "comparison_checks",
            "sample_count",
            "system_count",
        ):
            value = independent.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"nonpositive_{field}")
        sources = independent.get("sources")
        artifacts = independent.get("artifacts")
        required_sources = {
            "formal_lock",
            "execution",
            "formal_manifest",
            "labels",
            "label_manifest",
            "sample_manifest",
            "evaluation_plan",
            "candidate_pools",
        }
        if not isinstance(sources, Mapping) or not required_sources.issubset(sources):
            errors.append("source_inventory_incomplete")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "metrics",
            "per_sample",
            "report",
        }:
            errors.append("artifact_inventory_not_exact")
        if isinstance(sources, Mapping) and isinstance(artifacts, Mapping):
            try:
                verified = verify_artifact_records_recursive(
                    {"sources": sources, "artifacts": artifacts},
                    name="independent recompute",
                    require_at_least_one=True,
                )
                details["verified_artifact_records"] = len(verified)
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f"recursive_artifact_verification:{error}")
            current_records = {
                "formal_lock": _exact_current_record(
                    root / "08_lock" / "FORMAL_TEST_LOCK.json"
                ),
                "execution": _exact_current_record(
                    root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
                ),
                "formal_manifest": _exact_current_record(
                    root / "09_formal_test" / "formal_test_manifest.json"
                ),
            }
            for name, expected in current_records.items():
                if (
                    not isinstance(sources.get(name), Mapping)
                    or dict(sources[name]) != expected
                ):
                    errors.append(f"current_binding_mismatch:{name}")
            execution = read_json(
                root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
            )
            execution_manifest = dict(execution.get("artifacts", {})).get(
                "formal_test_manifest"
            )
            if execution_manifest != current_records["formal_manifest"]:
                errors.append("execution_formal_manifest_binding_mismatch")
        bridge = independent.get("test_bridge_checks")
        if not isinstance(bridge, Mapping):
            errors.append("test_bridge_checks_missing")
        else:
            if (
                int(bridge.get("candidate_rows", 0)) <= 0
                or int(bridge.get("pool_cells", 0)) != 4
            ):
                errors.append("test_bridge_checks_nonpositive")
            for field in (
                "membership_geometry_verified",
                "selector_formulas_verified",
                "candidate_labels_recomputed",
                "no_output_denominators_verified",
            ):
                if bridge.get(field) is not True:
                    errors.append(f"test_bridge_check_failed:{field}")
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        PermissionError,
        json.JSONDecodeError,
    ) as error:
        errors.append(f"unreadable:{error}")
    return not errors, details


def core_integrity_checks(
    bundle: FormalBundle, taxonomy: pd.DataFrame
) -> dict[str, Any]:
    """Recheck the core completion conditions from persisted artifacts."""

    root = bundle.run_dir
    checks: dict[str, dict[str, Any]] = {}
    source_ok, source_rows = _current_source_hash_checks(root)
    checks["source_runs_immutable"] = {
        "status": "PASS" if source_ok else "FAIL",
        "details": source_rows,
    }

    candidate_errors: list[str] = []
    top5_ok = True
    for route in ROUTES:
        native_name = _route_system(bundle, route, "native")
        native = bundle.rankings.loc[
            bundle.rankings["system_name"].astype(str).eq(native_name)
        ].copy()
        pool = bundle.candidate_pools[route]
        top5_pool = pool.loc[pd.to_numeric(pool["native_rank"], errors="coerce").le(5)]
        expected = set(map(tuple, top5_pool[["sample_id", "candidate_id"]].to_numpy()))
        for kind in ("native", "ungated", "gated"):
            name = _route_system(bundle, route, kind)
            observed_frame = bundle.rankings.loc[
                bundle.rankings["system_name"].astype(str).eq(name)
            ]
            observed = set(
                map(tuple, observed_frame[["sample_id", "candidate_id"]].to_numpy())
            )
            if observed != expected:
                candidate_errors.append(
                    f"{name}:candidate membership differs from Top-5 pool"
                )
        geometry = native.merge(
            top5_pool[
                [
                    "sample_id",
                    "candidate_id",
                    "candidate_geometry_sha256",
                    "native_rank",
                ]
            ],
            on=["sample_id", "candidate_id"],
            suffixes=("_ranking", "_pool"),
            how="outer",
            indicator=True,
        )
        if not geometry["_merge"].eq("both").all() or not geometry[
            "candidate_geometry_sha256_ranking"
        ].astype(str).equals(geometry["candidate_geometry_sha256_pool"].astype(str)):
            candidate_errors.append(f"{route}:geometry binding differs")
        if not _exact_numeric_values_equal(
            geometry["frozen_native_rank"], geometry["native_rank"]
        ):
            candidate_errors.append(f"{route}:frozen native ranks differ")
        native_j5 = bundle.metrics[native_name].get("j_at_5")
        for kind in ("ungated", "gated"):
            if (
                bundle.metrics[_route_system(bundle, route, kind)].get("j_at_5")
                != native_j5
            ):
                top5_ok = False
    checks["candidate_membership_geometry_invariance"] = {
        "status": "PASS" if not candidate_errors else "FAIL",
        "details": candidate_errors,
    }
    native_score_ok, native_score_errors = _native_score_source_check(bundle)
    checks["native_score_preserved"] = {
        "status": "PASS" if native_score_ok else "FAIL",
        "details": native_score_errors
        or "canonical native candidate order/scores exactly match audited sources",
    }
    checks["top5_oracle_invariance"] = {
        "status": "PASS" if top5_ok else "FAIL",
        "details": "native/ungated/gated J@5 compared route-wise",
    }

    shared_mask_ok, shared_mask_errors = shared_hifi_mask_check(root)
    checks["g1_c1_shared_hifi_masks"] = {
        "status": "PASS" if shared_mask_ok else "FAIL",
        "details": shared_mask_errors
        or "G1/C1 T2 contexts match paired Test and current HiFi mask bytes",
    }

    leakage_ok, leakage = _feature_leakage_check(root)
    checks["no_gt_feature_leakage"] = {
        "status": "PASS" if leakage_ok else "FAIL",
        "details": leakage,
    }
    split_path = root / "04_splits" / "split_leakage_audit.json"
    split = read_json(split_path) if split_path.is_file() else {}
    split_ok = (
        split.get("status") == "PASS"
        and int(split.get("scene_cross_fold_overlap", -1)) == 0
        and int(split.get("frame_cross_fold_overlap", -1)) == 0
    )
    checks["no_scene_frame_split_leakage"] = {
        "status": "PASS" if split_ok else "FAIL",
        "details": split,
    }
    access_ok, access_events = _access_log_check(root)
    checks["no_candidate_test_access_before_lock"] = {
        "status": "PASS" if access_ok else "FAIL",
        "details": access_events,
    }
    execution = read_json(root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json")
    count_ok = (
        execution.get("status") == "COMPLETE" and execution.get("execution_count") == 1
    )
    checks["formal_test_only_once"] = {
        "status": "PASS" if count_ok else "FAIL",
        "details": {
            "status": execution.get("status"),
            "execution_count": execution.get("execution_count"),
        },
    }
    independent_ok, independent_details = _independent_recompute_integrity(root)
    checks["independent_recompute"] = {
        "status": "PASS" if independent_ok else "FAIL",
        "details": independent_details,
    }
    taxonomy_ok = (
        len(taxonomy) == len(bundle.sample_manifest) * 3
        and not taxonomy[["route", "sample_id"]].duplicated().any()
    )
    checks["failure_taxonomy_denominator"] = {
        "status": "PASS" if taxonomy_ok else "FAIL",
        "details": {"rows": len(taxonomy), "expected": len(bundle.sample_manifest) * 3},
    }
    passed = all(value["status"] == "PASS" for value in checks.values())
    return {"status": "PASS" if passed else "FAIL", "checks": checks}


def hash_inventory(
    run_dir: Path,
    *,
    excluded: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    """Inventory immutable files relevant to the final scientific result."""

    root = run_dir.resolve()
    excluded_resolved = {path.resolve() for path in excluded}
    prefixes = (
        "00_audit",
        "01_manifests",
        "02_candidates",
        "03_features",
        "04_splits",
        "05_calibration",
        "05_models",
        "06_oof",
        "07_validation",
        "08_lock",
        "09_formal_test",
        "10_statistics",
        "11_attribution_bridge",
        "12_figures",
        "13_failure_galleries",
        "14_reports",
        "15_independent_recompute",
        "configs",
        "checkpoints",
        "predictions",
        "tables",
    )
    rows: list[dict[str, Any]] = []
    root_control_files = (
        "manifest.json",
        "README_REPRODUCE.md",
        "run_ledger.sqlite",
        "commands.log",
        "environment.txt",
        "git_state.txt",
    )
    for name in root_control_files:
        path = root / name
        resolved = path.resolve()
        if (
            path.is_file()
            and not path.is_symlink()
            and resolved not in excluded_resolved
        ):
            rows.append(
                {
                    "path": str(resolved),
                    "relative_path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for prefix in prefixes:
        directory = root / prefix
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            resolved = path.resolve()
            if path.is_symlink() or not path.is_file() or resolved in excluded_resolved:
                continue
            rows.append(
                {
                    "path": str(resolved),
                    "relative_path": str(resolved.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def verify_inventory(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    relative_paths: set[str] = set()
    for row in rows:
        path_value = row.get("path")
        relative = row.get("relative_path")
        expected_bytes = row.get("bytes")
        if not isinstance(path_value, str) or not isinstance(relative, str):
            errors.append("malformed-inventory-record")
            continue
        if relative in relative_paths:
            errors.append(f"duplicate-relative-path:{relative}")
        relative_paths.add(relative)
        path = Path(path_value)
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing/non-regular:{path}")
        elif (
            not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes
        ):
            errors.append(f"size-mismatch:{path}")
        elif sha256_file(path) != row.get("sha256"):
            errors.append(f"hash-mismatch:{path}")
    return not errors, errors


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return "_No machine-readable rows were available._"
    display = frame.loc[:, available].copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "NA" if pd.isna(value) else f"{float(value):.6f}"
        )

    def cell(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r\n", "<br>")
            .replace("\r", "<br>")
            .replace("\n", "<br>")
        )

    headers = [cell(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def safe_number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "CROSS_ROUTE_GALLERY_QUOTAS",
    "FAILURE_DEFINITIONS",
    "FIGURE_NAMES",
    "FormalBundle",
    "REPORT_NAMES",
    "ROUTES",
    "ROUTE_GALLERY_QUOTAS",
    "TABLE_NAMES",
    "atomic_csv",
    "atomic_json",
    "atomic_parquet",
    "atomic_text",
    "classify_failures",
    "core_integrity_checks",
    "deployment_decision",
    "deterministic_case_selection",
    "formal_results_table",
    "gallery_category_members",
    "hash_inventory",
    "latex_escape",
    "load_formal_bundle",
    "markdown_table",
    "normalize_candidate_labels",
    "read_json",
    "safe_number",
    "shared_hifi_mask_check",
    "taxonomy_summary",
    "verify_inventory",
]
