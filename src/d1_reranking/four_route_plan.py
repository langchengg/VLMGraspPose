"""Immutable P12 source plan and Validation-only artifact contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file

from .contracts import (
    EXPECTED_UNIFIED_FINAL_LOCK_SHA256,
    assert_label_free_parquet_schema,
)
from .execution import load_content_manifest
from .four_route import (
    ALTERNATIVE_ROUTES,
    DEFAULT_ROUTE_TIE_BREAK,
    FOUR_ROUTES,
    MAX_UNION_CANDIDATES,
    FourRouteRouterSelection,
    validation_router_union_summary,
)
from .io import atomic_csv, atomic_parquet
from .run import assert_writable_prelock


THREE_ROUTE_SOURCE_NAMES = (
    "train_oof_predictions",
    "validation_gated_top1",
    "validation_gated_top5",
    "test_gated_top1",
    "test_gated_top5",
    "validation_router_predictions",
    "test_router_predictions",
    "validation_top15_union_predictions",
    "test_top15_union_predictions",
)
D1_SOURCE_NAMES = (
    "train_oof_r7_predictions",
    "validation_r7_predictions",
    "test_ranker_application",
    "test_gate_application",
    "validation_top5",
    "test_top5",
    "train_t3_manifest",
    "validation_t3_manifest",
    "test_t3_manifest",
)
PLAN_RELATIVE_PATH = Path("configs/d1_four_route_extension_plan.json")
VALIDATION_RELATIVE_DIR = Path("13_four_route_extension")
PRODUCER_SPEC_STATUS = "VALIDATION_PRODUCER_DECLARED"
PRODUCER_JOB_STAGES = (
    "router_validation",
    "top20_union_validation",
    "t4_train",
    "t4_validation",
    "validation_summary",
)


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"P12 source must be a regular file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_completed_lock(
    root: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock_path = root / "FINAL_RUN_LOCK.json"
    lock_sha256 = sha256_file(lock_path)
    if lock_sha256 != expected_sha256:
        raise RuntimeError("completed three-route final-lock file hash differs")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("status") != "COMPLETE":
        raise RuntimeError("three-route final lock is not COMPLETE")
    unsigned = dict(lock)
    observed_self = unsigned.pop("self_sha256", None)
    if observed_self != canonical_sha256(unsigned):
        raise RuntimeError("three-route final-lock self hash differs")
    inventory = lock.get("inventory")
    if (
        not isinstance(inventory, list)
        or lock.get("inventory_count") != len(inventory)
        or lock.get("inventory_content_sha256") != canonical_sha256(inventory)
    ):
        raise RuntimeError("three-route final-lock inventory contract differs")
    by_path: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(inventory):
        if not isinstance(raw_record, dict):
            raise RuntimeError(f"three-route inventory row {index} is invalid")
        path = str(Path(str(raw_record.get("path", ""))).resolve())
        if not path or path in by_path:
            raise RuntimeError("three-route inventory paths are absent or duplicated")
        by_path[path] = raw_record
    marker = root / "COMPLETE"
    marker_text = marker.read_text(encoding="utf-8")
    if "COMPLETE" not in marker_text or f"sha256={lock_sha256}" not in marker_text:
        raise RuntimeError("three-route COMPLETE marker does not bind the final lock")
    return lock, by_path


def _normalized_paths(
    values: Mapping[str, str | Path], *, expected_names: tuple[str, ...], name: str
) -> dict[str, Path]:
    if set(values) != set(expected_names):
        missing = sorted(set(expected_names).difference(values))
        extra = sorted(set(values).difference(expected_names))
        raise ValueError(f"{name} names differ; missing={missing}, extra={extra}")
    return {key: Path(values[key]).expanduser().resolve() for key in expected_names}


def _assert_application_contract(path: Path, *, name: str) -> dict[str, Any]:
    value = load_content_manifest(path, name=name, statuses=("COMPLETE",))
    if value.get("candidate_test_labels_read") is not False:
        raise PermissionError(
            f"{name} does not declare candidate_test_labels_read=false"
        )
    if value.get("selection_used_test_metrics") is True:
        raise PermissionError(f"{name} declares Test-based selection")
    verify_artifact_records_recursive(
        {"sources": value.get("sources"), "artifacts": value.get("artifacts")},
        name=name,
        require_at_least_one=True,
    )
    return value


def _spec_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"P12 producer spec path is absent: {name}")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _string_columns(value: Any, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(column, str) and column for column in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"P12 producer feature columns differ: {name}")
    return list(value)


def _build_validation_producer_contract(
    spec_path: Path,
    *,
    d1_root: Path,
    three_root: Path,
    frozen_inventory: Mapping[str, Mapping[str, Any]],
    forbidden_test_paths: set[Path],
) -> dict[str, Any]:
    """Expand the human-authored producer spec into exact current file records."""

    resolved_spec = spec_path.expanduser().resolve()
    if not _inside(resolved_spec, d1_root):
        raise PermissionError("P12 producer spec must be inside the D1 run")
    spec = load_content_manifest(
        resolved_spec,
        name="P12 Validation producer spec",
        statuses=(PRODUCER_SPEC_STATUS,),
    )
    if (
        spec.get("candidate_test_labels_read") is not False
        or spec.get("selection_used_test_metrics") is not False
    ):
        raise PermissionError("P12 Validation producer spec may not reference Test")
    allowed_keys = {
        "schema_version",
        "status",
        "router",
        "union",
        "t4",
        "candidate_test_labels_read",
        "selection_used_test_metrics",
        "content_sha256",
    }
    if set(spec) != allowed_keys:
        raise ValueError("P12 Validation producer spec top-level keys differ")
    base = resolved_spec.parent
    router = spec.get("router")
    union = spec.get("union")
    t4 = spec.get("t4")
    if (
        not isinstance(router, Mapping)
        or not isinstance(union, Mapping)
        or not isinstance(t4, Mapping)
    ):
        raise ValueError("P12 Validation producer spec sections differ")
    if set(router) != {"train_oof", "validation", "feature_columns"}:
        raise ValueError("P12 router producer spec keys differ")
    if set(union) != {
        "train_top20",
        "validation_top20",
        "folds",
        "train_denominator",
        "validation_denominator",
        "feature_columns",
    }:
        raise ValueError("P12 union producer spec keys differ")
    if set(t4) != {"train", "validation"}:
        raise PermissionError(
            "P12 Validation producer T4 splits must be Train/Validation only"
        )

    def d1_record(value: Any, *, name: str) -> dict[str, Any]:
        path = _spec_path(value, base=base, name=name)
        if path in forbidden_test_paths or not _inside(path, d1_root):
            raise PermissionError(
                f"P12 Validation producer input is not D1 development-only: {name}"
            )
        return _record(path)

    def frozen_record(value: Any, *, name: str) -> dict[str, Any]:
        path = _spec_path(value, base=base, name=name)
        if path in forbidden_test_paths or not _inside(path, three_root):
            raise PermissionError(
                f"P12 peer input is outside the completed run: {name}"
            )
        current = _record(path)
        frozen = frozen_inventory.get(str(path))
        if not isinstance(frozen, Mapping) or any(
            frozen.get(key) != current[key] for key in ("path", "sha256", "bytes")
        ):
            raise RuntimeError(f"P12 peer input is not frozen by final lock: {name}")
        return current

    route_columns = router["feature_columns"]
    if not isinstance(route_columns, Mapping) or set(route_columns) != set(
        ALTERNATIVE_ROUTES
    ):
        raise ValueError("P12 router producer feature routes differ")
    router_contract = {
        "train_oof": d1_record(router["train_oof"], name="router.train_oof"),
        "validation": d1_record(router["validation"], name="router.validation"),
        "feature_columns": {
            route: _string_columns(route_columns[route], name=f"router.{route}")
            for route in ALTERNATIVE_ROUTES
        },
    }
    union_contract = {
        key: d1_record(union[key], name=f"union.{key}")
        for key in (
            "train_top20",
            "validation_top20",
            "folds",
            "train_denominator",
            "validation_denominator",
        )
    }
    union_contract["feature_columns"] = _string_columns(
        union["feature_columns"], name="union"
    )
    t4_contract: dict[str, Any] = {}
    for split in ("train", "validation"):
        raw = t4[split]
        if not isinstance(raw, Mapping) or set(raw) != {
            "d1_top5",
            "t3_manifest",
            "peer_top5",
        }:
            raise ValueError(f"P12 {split} T4 producer spec keys differ")
        peers = raw["peer_top5"]
        if not isinstance(peers, Mapping) or set(peers) != {"CROG", "G1", "C1"}:
            raise ValueError(f"P12 {split} T4 peer routes differ")
        t4_contract[split] = {
            "d1_top5": d1_record(raw["d1_top5"], name=f"t4.{split}.d1_top5"),
            "t3_manifest": d1_record(
                raw["t3_manifest"], name=f"t4.{split}.t3_manifest"
            ),
            "peer_top5": {
                route: frozen_record(peers[route], name=f"t4.{split}.{route}")
                for route in ("CROG", "G1", "C1")
            },
        }
    configurations = [
        {"stage": stage, "device": "cpu", "threads": 1} for stage in PRODUCER_JOB_STAGES
    ]
    jobs = [
        {
            "job_id": canonical_sha256(configuration)[:16],
            "configuration": configuration,
        }
        for configuration in configurations
    ]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "status": "READY",
        "spec": _record(resolved_spec),
        "router": router_contract,
        "union": union_contract,
        "t4": t4_contract,
        "jobs": jobs,
        "job_count": len(jobs),
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "device": "cpu",
        "max_parallel": 1,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    contract["content_sha256"] = canonical_sha256(contract)
    return contract


def build_four_route_plan(
    *,
    d1_run_dir: str | Path,
    completed_three_route_run: str | Path,
    three_route_sources: Mapping[str, str | Path],
    d1_sources: Mapping[str, str | Path],
    code_paths: tuple[Path, ...],
    producer_spec_path: str | Path | None = None,
    expected_final_lock_sha256: str = EXPECTED_UNIFIED_FINAL_LOCK_SHA256,
) -> dict[str, Any]:
    """Build a non-executable, current-hash-bound four-route extension plan."""

    d1_root = Path(d1_run_dir).expanduser().resolve()
    three_root = Path(completed_three_route_run).expanduser().resolve()
    if d1_root == three_root or _inside(d1_root, three_root):
        raise PermissionError("P12 outputs may not be written inside the completed run")
    _lock, inventory = _load_completed_lock(
        three_root, expected_sha256=str(expected_final_lock_sha256)
    )
    three_paths = _normalized_paths(
        three_route_sources,
        expected_names=THREE_ROUTE_SOURCE_NAMES,
        name="three-route P12 sources",
    )
    d1_paths = _normalized_paths(
        d1_sources,
        expected_names=D1_SOURCE_NAMES,
        name="D1 P12 sources",
    )
    three_records: dict[str, Any] = {}
    for name, path in three_paths.items():
        if not _inside(path, three_root):
            raise PermissionError(
                f"three-route source is outside completed run: {name}"
            )
        current = _record(path)
        frozen = inventory.get(str(path))
        if not isinstance(frozen, dict) or any(
            frozen.get(key) != current[key] for key in ("path", "sha256", "bytes")
        ):
            raise RuntimeError(
                f"three-route source is not frozen by final lock: {name}"
            )
        three_records[name] = current
    d1_records: dict[str, Any] = {}
    for name, path in d1_paths.items():
        if not _inside(path, d1_root):
            raise PermissionError(f"D1 source is outside the D1 run: {name}")
        d1_records[name] = _record(path)
    forbidden_test_paths = {
        path
        for name, path in {**three_paths, **d1_paths}.items()
        if "test" in name.lower()
    }
    producer_contract = (
        {"status": "NOT_CONFIGURED", "candidate_test_labels_read": False}
        if producer_spec_path is None
        else _build_validation_producer_contract(
            Path(producer_spec_path),
            d1_root=d1_root,
            three_root=three_root,
            frozen_inventory=inventory,
            forbidden_test_paths=forbidden_test_paths,
        )
    )
    if producer_contract.get("status") == "READY":
        expected_t4_bindings = {
            ("train", "t3_manifest"): d1_records["train_t3_manifest"],
            ("validation", "t3_manifest"): d1_records["validation_t3_manifest"],
            ("validation", "d1_top5"): d1_records["validation_top5"],
        }
        for (split, key), expected in expected_t4_bindings.items():
            if producer_contract["t4"][split][key] != expected:
                raise RuntimeError(
                    f"P12 producer {split} T4 {key} is not the canonical D1 source"
                )
    _assert_application_contract(
        d1_paths["test_ranker_application"], name="D1 Test ranker application"
    )
    _assert_application_contract(
        d1_paths["test_gate_application"], name="D1 Test gate application"
    )
    for split in ("train", "validation", "test"):
        t3_path = d1_paths[f"{split}_t3_manifest"]
        t3 = load_content_manifest(
            t3_path, name=f"D1 {split} T3", statuses=("COMPLETE",)
        )
        configuration = t3.get("configuration", {})
        columns = tuple(map(str, t3.get("model_feature_columns", ())))
        if (
            configuration.get("split") != split
            or configuration.get("pool") != "top5"
            or configuration.get("track") != "T3_route_rich"
            or t3.get("candidate_test_labels_read") is not False
            or not columns
            or t3.get("model_feature_schema_sha256") != canonical_sha256(columns)
        ):
            raise RuntimeError(f"D1 {split} T3 source contract differs")
        verify_artifact_records_recursive(
            t3.get("artifacts"),
            name=f"D1 {split} T3 artifacts",
            require_at_least_one=True,
        )
        if split == "test":
            feature_record = t3.get("artifacts", {}).get("candidate_features", {})
            assert_label_free_parquet_schema(
                Path(str(feature_record.get("path", ""))),
                name="P12 Test T3 features",
            )
    assert_label_free_parquet_schema(d1_paths["test_top5"], name="P12 Test Top5")
    resolved_code = sorted(
        {Path(path).expanduser().resolve() for path in code_paths}, key=str
    )
    if not resolved_code:
        raise ValueError("P12 plan requires implementation source paths")
    sources = {
        "completed_three_route_final_lock": _record(three_root / "FINAL_RUN_LOCK.json"),
        "completed_three_route_marker": _record(three_root / "COMPLETE"),
        "completed_three_route_outputs": three_records,
        "d1_outputs": d1_records,
        "code": [_record(path) for path in resolved_code],
        "validation_producer": producer_contract,
    }
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "PLANNED",
        "stage": "P12_FOUR_ROUTE_EXTENSION",
        "execution_authorized": False,
        "formal_test_lock_created": False,
        "validation_producer_ready": producer_contract.get("status") == "READY",
        "completed_three_route_run_mode": "STRICT_READ_ONLY",
        "development_selection_split": "Validation",
        "router_fit_prediction_source": "paired_train_oof",
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "router_contract": {
            "default_route": "CROG",
            "alternative_routes": list(ALTERNATIVE_ROUTES),
            "tie_break": list(DEFAULT_ROUTE_TIE_BREAK),
            "decision_rule": "recover_probability-lambda_router*harm_probability",
            "no_go_fallback": "CROG",
            "test_application": "locked Validation operating point; label-free evidence only",
            "grid": {
                "lambda_router": [1.0, 2.0, 4.0],
                "utility_thresholds": [0.0, 0.05, 0.1],
                "margin_thresholds": [0.0, 0.05, 0.1],
                "reliability_thresholds": [0.5, 0.75],
                "stability_thresholds": [0.5, 0.75],
                "trial_count": 108,
            },
        },
        "top20_union_contract": {
            "routes": list(FOUR_ROUTES),
            "source_membership": "exact Top5 from each route",
            "maximum_candidates": MAX_UNION_CANDIDATES,
            "route_qualified_candidate_ids": True,
            "cross_route_deduplication": "NONE",
            "duplicate_raw_candidate_ids_across_routes": "RETAIN",
            "deterministic_order": "route native rank then CROG/G1/C1/D1",
            "rankers": ["LambdaMART", "DeepSets_if_existing_stable"],
            "negative_result_rule": "retain when Validation Oracle@20 <= Oracle@15",
            "encoders": ["lambdamart", "deepsets"],
            "seeds": [42, 123, 2026],
            "outer_folds": 5,
            "cell_count": 36,
        },
        "t4_contract": {
            "route": "D1",
            "pool": "top5",
            "track": "T4_four_route_consensus",
            "base_track": "T3_route_rich",
            "column_order": "exact T3 model columns then stable-sorted t4 consensus",
            "candidate_join": "exact one-to-one candidate identity and geometry",
            "splits": ["train", "validation", "test"],
            "peer_routes": ["CROG", "G1", "C1"],
            "coordinates": "original_image_pixels",
            "supervision_features": False,
            "test_application": "label-free schema checked before row access",
        },
        "prelock_ready_outputs": [
            "13_four_route_extension/router/router_selection_manifest.json",
            "13_four_route_extension/union/union_plan.json",
            "13_four_route_extension/union/selected_union_ranker.json",
            "13_four_route_extension/validation_router_decisions.parquet",
            "13_four_route_extension/validation_top20_union.parquet",
            "13_four_route_extension/validation_router_union.csv",
            "13_four_route_extension/validation_router_union_manifest.json",
            "03_features/train/top5/T4_four_route_consensus/manifest.json",
            "03_features/validation/top5/T4_four_route_consensus/manifest.json",
            "03_features/test/top5/T4_four_route_consensus/manifest.json",
            "08_lock/formal_inputs/four_route_crog_default_router/manifest.json",
            "08_lock/formal_inputs/top20_union/manifest.json",
        ],
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    plan["content_sha256"] = canonical_sha256(plan)
    return plan


def write_four_route_plan(
    *,
    d1_run_dir: str | Path,
    completed_three_route_run: str | Path,
    three_route_sources: Mapping[str, str | Path],
    d1_sources: Mapping[str, str | Path],
    code_paths: tuple[Path, ...],
    producer_spec_path: str | Path | None = None,
    resume: bool,
    expected_final_lock_sha256: str = EXPECTED_UNIFIED_FINAL_LOCK_SHA256,
) -> dict[str, Any]:
    root = Path(d1_run_dir).expanduser().resolve()
    assert_writable_prelock(root)
    plan = build_four_route_plan(
        d1_run_dir=root,
        completed_three_route_run=completed_three_route_run,
        three_route_sources=three_route_sources,
        d1_sources=d1_sources,
        code_paths=code_paths,
        producer_spec_path=producer_spec_path,
        expected_final_lock_sha256=expected_final_lock_sha256,
    )
    destination = root / PLAN_RELATIVE_PATH
    if destination.exists():
        existing = load_content_manifest(
            destination, name="D1 four-route plan", statuses=("PLANNED",)
        )
        if resume and existing == plan:
            return existing
        raise RuntimeError("immutable D1 four-route plan exists and differs")
    atomic_json(destination, plan)
    return plan


def _frame_sha256(frame: pd.DataFrame) -> str:
    return canonical_sha256(
        {
            "columns": list(map(str, frame.columns)),
            "records": frame.to_dict("records"),
        }
    )


def _verify_current_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"P12 source record is absent: {name}")
        current = _record(str(record.get("path", "")))
        if any(
            current.get(key) != record.get(key) for key in ("path", "sha256", "bytes")
        ):
            raise RuntimeError(f"P12 source record drift: {name}")


def write_validation_router_union_artifacts(
    run_dir: str | Path,
    *,
    samples: pd.DataFrame,
    top20_union: pd.DataFrame,
    router_selection: FourRouteRouterSelection | Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
    resume: bool,
) -> dict[str, Any]:
    """Write only Validation semantics needed by P13 readiness."""

    root = Path(run_dir).expanduser().resolve()
    assert_writable_prelock(root)
    if any(
        str(column).lower().startswith("test_")
        or "candidate_test" in str(column).lower()
        for frame in (samples, top20_union)
        for column in frame.columns
    ):
        raise PermissionError("P12 Validation inputs expose Test-derived columns")
    if "prediction_source" in samples and set(
        samples["prediction_source"].astype(str)
    ) != {"validation"}:
        raise RuntimeError("P12 sample predictions are not Validation-only")
    selection = (
        router_selection.artifact()
        if isinstance(router_selection, FourRouteRouterSelection)
        else dict(router_selection)
    )
    if selection.get("status") not in {"GO", "NO_GO_CROG"}:
        raise ValueError("P12 router selection status is invalid")
    if selection.get("candidate_test_labels_read") is not False:
        raise PermissionError("P12 router selection lacks Test-label isolation")
    decisions = samples["four_route_decision"].astype(str).str.upper()
    if selection.get("status") == "NO_GO_CROG" and set(decisions) != {"CROG"}:
        raise RuntimeError("P12 NO_GO_CROG selection contains route switches")
    summary = validation_router_union_summary(samples, top20_union)
    if any(route not in set(summary["route"].astype(str)) for route in FOUR_ROUTES):
        raise RuntimeError("P12 Validation summary does not cover all four routes")
    sources = {name: _record(path) for name, path in sorted(source_paths.items())}
    if not sources:
        raise ValueError("P12 Validation artifacts require current source bindings")
    input_content = {
        "samples_sha256": _frame_sha256(samples),
        "top20_union_sha256": _frame_sha256(top20_union),
        "router_selection_sha256": canonical_sha256(selection),
        "summary_sha256": _frame_sha256(summary),
    }
    configuration = {
        "stage": "P12_FOUR_ROUTE_EXTENSION",
        "development_split": "Validation",
        "routes": list(FOUR_ROUTES),
        "default_route": "CROG",
        "top20_maximum_candidates": MAX_UNION_CANDIDATES,
        "cross_route_deduplication": "NONE",
        "route_qualified_candidate_ids": True,
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "formal_test_lock_created": False,
    }
    signature = canonical_sha256(
        {
            "configuration": configuration,
            "sources": sources,
            "input_content": input_content,
            "router_selection": selection,
        }
    )
    output = root / VALIDATION_RELATIVE_DIR
    manifest_path = output / "validation_router_union_manifest.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path,
            name="D1 P12 Validation router/union",
            statuses=("COMPLETE",),
        )
        if (
            resume
            and existing.get("signature_sha256") == signature
            and existing.get("sources") == sources
            and existing.get("input_content") == input_content
        ):
            _verify_current_records(existing["sources"])
            verify_artifact_records_recursive(
                existing.get("artifacts"),
                name="D1 P12 Validation artifacts",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("immutable D1 P12 Validation artifacts exist and differ")
    planned_paths = (
        output / "validation_router_decisions.parquet",
        output / "validation_top20_union.parquet",
        output / "validation_router_union.csv",
    )
    if any(path.exists() for path in planned_paths):
        raise RuntimeError("partial D1 P12 Validation artifacts already exist")
    decision_path = atomic_parquet(
        samples, output / "validation_router_decisions.parquet"
    )
    union_path = atomic_parquet(top20_union, output / "validation_top20_union.parquet")
    table_path = atomic_csv(summary, output / "validation_router_union.csv")
    artifacts = {
        "router_decisions": _record(decision_path),
        "top20_union": _record(union_path),
        "prelock_table": _record(table_path),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": configuration,
        "router_selection": selection,
        "input_content": input_content,
        "sources": sources,
        "artifacts": artifacts,
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "formal_test_lock_created": False,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = [
    "D1_SOURCE_NAMES",
    "PLAN_RELATIVE_PATH",
    "PRODUCER_JOB_STAGES",
    "PRODUCER_SPEC_STATUS",
    "THREE_ROUTE_SOURCE_NAMES",
    "VALIDATION_RELATIVE_DIR",
    "build_four_route_plan",
    "write_four_route_plan",
    "write_validation_router_union_artifacts",
]
