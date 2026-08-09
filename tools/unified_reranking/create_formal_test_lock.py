"""Create the immutable, content-verified unified formal-Test lock."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.lock import create_formal_test_lock, verify_formal_test_lock
from unified_reranking.prelock_validation import (
    validate_application_signature,
    validate_encoder_execution,
    validate_feature_ablation_manifest,
    validate_feature_extraction_benchmark,
    validate_gate_application_outputs,
    validate_gate_selection,
    validate_matrix_phase_execution,
    validate_router_application_outputs,
    validate_router_selection,
    validate_screen_selection,
    validate_scalar_selection,
    validate_union_application_outputs,
    validate_union_headroom,
    validate_union_selection,
)
from unified_reranking.test_bridge import validate_label_free_test_bridge_manifest


REQUIRED_LOCK_FILES = {
    "selected_methods": "selected_methods.json",
    "selected_features": "selected_features.json",
    "selected_hyperparameters": "selected_hyperparameters.json",
    "calibration_manifest": "calibration_manifest.json",
    "gate_thresholds": "gate_thresholds.json",
    "router_thresholds": "router_thresholds.json",
    "candidate_manifests": "candidate_manifests.json",
    "fold_assignments_sha256": "fold_assignments_sha256.txt",
    "evaluator_sha256": "evaluator_sha256.txt",
    "code_sha256": "code_sha256.txt",
    "primary_method_declaration": "PRIMARY_METHOD_DECLARATION.md",
}
ROUTES = ("crog", "g1", "c1")
BASE_SYSTEM_KINDS = {"native", "ungated", "gated", "router"}
SYSTEM_KINDS = {*BASE_SYSTEM_KINDS, "union"}


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _regular_file(path: str | Path, name: str) -> Path:
    result = Path(path).resolve()
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"{name} is not a regular file: {result}")
    return result


def _validate_hash_text(path: Path, name: str) -> None:
    value = path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must contain exactly one lowercase SHA-256")


def validate_formal_evaluation_plan(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate a label-free evaluation plan and resolve every locked input."""

    plan = _read_json(path)
    if not isinstance(plan, dict) or int(plan.get("schema_version", -1)) != 1:
        raise ValueError("formal evaluation plan schema_version must be 1")
    if plan.get("candidate_test_labels_read") is not False:
        raise ValueError("formal evaluation plan must declare candidate_test_labels_read=false")
    sample_manifest = _regular_file(plan.get("sample_manifest", ""), "sample_manifest")
    label_manifest_path = _regular_file(
        plan.get("candidate_label_manifest", ""), "candidate_label_manifest"
    )
    label_manifest = _read_json(label_manifest_path)
    if (
        not isinstance(label_manifest, dict)
        or int(label_manifest.get("schema_version", -1)) != 1
        or str(label_manifest.get("split", "")).lower() != "test"
        or not str(label_manifest.get("provenance", ""))
        or re.fullmatch(r"[0-9a-f]{64}", str(label_manifest.get("candidate_labels_sha256", "")))
        is None
        or re.fullmatch(r"[0-9a-f]{64}", str(label_manifest.get("evaluator_sha256", "")))
        is None
    ):
        raise ValueError("candidate label manifest has an invalid provenance/hash contract")
    label_path = Path(str(label_manifest.get("candidate_labels_path", ""))).resolve()
    if label_path.is_symlink() or not label_path.is_file():
        raise ValueError("predeclared candidate-level Test label path is not a regular file")
    normalization = label_manifest.get("normalization", {})
    if not isinstance(normalization, dict):
        raise ValueError("candidate label normalization must be a JSON object")
    if str(normalization.get("route_column", "route")) not in {"route", "method"}:
        raise ValueError("candidate label route_column must be route or method")
    variant_values = normalization.get("include_variants")
    if variant_values is not None and (
        not isinstance(variant_values, list)
        or not variant_values
        or any(not str(value) for value in variant_values)
    ):
        raise ValueError("candidate label include_variants must be a non-empty string list")
    pool_records = label_manifest.get("candidate_pools")
    if not isinstance(pool_records, dict) or set(pool_records) != set(ROUTES):
        raise ValueError("candidate label manifest must declare all three route candidate pools")
    systems = plan.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError("formal evaluation plan systems must be a non-empty list")
    names: set[str] = set()
    kinds: set[str] = set()
    route_kinds: dict[str, set[str]] = {route: set() for route in ROUTES}
    locked: dict[str, Path] = {
        "formal_sample_manifest": sample_manifest,
        "formal_candidate_label_manifest": label_manifest_path,
    }
    provenance = plan.get("bound_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("formal plan must contain bound_provenance")
    for name in (
        "fold_assignments",
        "evaluator",
        "code_manifest",
        "primary_selection",
        "screen_latest_execution",
        "screen_selection_manifest",
        "screen_finalists",
        "selected_latest_execution",
        "encoder_loss_selection",
        "encoder_latest_execution",
        "feature_ablation_manifest",
        "feature_extraction_benchmark",
    ):
        record = provenance.get(name)
        if (
            not isinstance(record, dict)
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
            is None
        ):
            raise ValueError(f"formal bound_provenance has an invalid {name} record")
        source_path = _regular_file(record.get("path", ""), f"provenance.{name}")
        if sha256_file(source_path) != record["sha256"]:
            raise ValueError(f"formal bound_provenance hash mismatch: {name}")
        locked[f"formal_provenance_{name}"] = source_path
    code_manifest = _read_json(locked["formal_provenance_code_manifest"])
    if provenance.get("code_bundle_sha256") != code_manifest.get("bundle_sha256"):
        raise ValueError("formal provenance code bundle hash differs from code manifest")
    run_dir = path.resolve().parents[1]
    validate_matrix_phase_execution(run_dir, "screen")
    validate_screen_selection(locked["formal_provenance_screen_selection_manifest"])
    validate_matrix_phase_execution(
        run_dir,
        "selected",
        expected_selection=provenance["screen_finalists"],
    )
    validate_matrix_phase_execution(
        run_dir,
        "encoder",
        expected_selection=provenance["encoder_loss_selection"],
    )
    validate_scalar_selection(locked["formal_provenance_primary_selection"])
    validate_encoder_execution(
        locked["formal_provenance_encoder_latest_execution"],
        expected_selection=provenance["encoder_loss_selection"],
    )
    validate_feature_ablation_manifest(
        locked["formal_provenance_feature_ablation_manifest"],
        expected_selection=provenance["primary_selection"],
    )
    validate_feature_extraction_benchmark(
        locked["formal_provenance_feature_extraction_benchmark"]
    )
    bridge_records = provenance.get("attribution_bridges")
    if not isinstance(bridge_records, dict) or set(bridge_records) != {
        "g1_train",
        "g1_validation",
        "c1_train",
        "c1_validation",
    }:
        raise ValueError("formal provenance must bind all four development bridge manifests")
    for name, record in sorted(bridge_records.items()):
        if (
            not isinstance(record, dict)
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
            is None
        ):
            raise ValueError(f"formal attribution-bridge record is invalid: {name}")
        bridge_path = _regular_file(record.get("path", ""), f"bridge.{name}")
        if sha256_file(bridge_path) != record["sha256"]:
            raise ValueError(f"formal attribution-bridge hash mismatch: {name}")
        locked[f"formal_provenance_bridge_{name}"] = bridge_path
    test_bridge_contract = plan.get("test_bridge_contract")
    if not isinstance(test_bridge_contract, dict):
        raise ValueError("formal plan must declare the label-free Test bridge contract")
    bridge_manifest_record = test_bridge_contract.get("manifest")
    if not isinstance(bridge_manifest_record, dict):
        raise ValueError("formal Test bridge contract has no manifest record")
    bridge_manifest_path = _regular_file(
        bridge_manifest_record.get("path", ""), "formal Test bridge manifest"
    )
    if sha256_file(bridge_manifest_path) != bridge_manifest_record.get("sha256"):
        raise ValueError("formal Test bridge manifest hash mismatch")
    if provenance.get("test_bridge_manifest") != bridge_manifest_record:
        raise ValueError("formal Test bridge manifest differs from bound provenance")
    bridge_manifest, _bridge_bundle, bridge_paths = (
        validate_label_free_test_bridge_manifest(
            bridge_manifest_path,
            expected_denominator={
                "path": str(sample_manifest),
                "sha256": sha256_file(sample_manifest),
            },
            expected_evaluator=provenance["evaluator"],
        )
    )
    for name in ("candidate_bundle", "historical_ground_truth", "evaluator", "denominator"):
        declared = (
            bridge_manifest["artifacts"]["candidate_bundle"]
            if name == "candidate_bundle"
            else bridge_manifest["sources"][name]
        )
        if test_bridge_contract.get(name) != declared:
            raise ValueError(f"formal Test bridge {name} binding mismatch")
    if Path(bridge_manifest["sources"]["historical_ground_truth"]["path"]).resolve() == label_path:
        raise ValueError("primary candidate labels and historical bridge GT must be distinct")
    locked["formal_test_bridge_manifest"] = bridge_manifest_path
    locked["formal_test_bridge_candidate_bundle"] = bridge_paths["candidate_bundle"]
    locked["formal_test_bridge_ground_truth"] = bridge_paths[
        "source_historical_ground_truth"
    ]
    for name in sorted(bridge_manifest["sources"]):
        locked[f"formal_test_bridge_source_{name}"] = bridge_paths[f"source_{name}"]
    for route in ROUTES:
        record = pool_records[route]
        if not isinstance(record, dict) or re.fullmatch(
            r"[0-9a-f]{64}", str(record.get("sha256", ""))
        ) is None:
            raise ValueError(f"invalid candidate-pool record for {route}")
        pool_path = _regular_file(record.get("path", ""), f"{route}.candidate_pool")
        if sha256_file(pool_path) != record["sha256"]:
            raise ValueError(f"predeclared candidate-pool hash mismatch: {route}")
        locked[f"formal_{route}_all_candidate_pool"] = pool_path
    for index, raw in enumerate(systems):
        if not isinstance(raw, dict):
            raise ValueError("every formal system must be a JSON object")
        name = str(raw.get("name", ""))
        kind = str(raw.get("kind", ""))
        route = raw.get("route")
        if not name or name in names:
            raise ValueError("formal system names must be unique and non-empty")
        if kind not in SYSTEM_KINDS:
            raise ValueError(f"invalid formal system kind for {name}: {kind}")
        if kind in {"router", "union"}:
            if route not in {None, "cross_route"}:
                raise ValueError(f"{kind} route must be null or cross_route")
        else:
            route = str(route).lower()
            if route not in ROUTES:
                raise ValueError(f"invalid route for formal system {name}: {route}")
            route_kinds[route].add(kind)
        decisions = _regular_file(raw.get("decisions_path", ""), f"{name}.decisions")
        locked[f"formal_system_{index:02d}_{name}_decisions"] = decisions
        ranking_raw = raw.get("ranking_path")
        base = raw.get("base_ranking_system")
        if kind in {"native", "ungated", "union"} and not ranking_raw:
            raise ValueError(f"{name} requires a prelocked ranking_path")
        if kind == "gated" and bool(ranking_raw) == bool(base):
            raise ValueError(
                f"{name} must provide exactly one of ranking_path or base_ranking_system"
            )
        if ranking_raw:
            locked[f"formal_system_{index:02d}_{name}_ranking"] = _regular_file(
                ranking_raw, f"{name}.ranking"
            )
        if kind == "union":
            for record_name in ("selection_manifest", "application_manifest"):
                record = raw.get(record_name)
                if (
                    not isinstance(record, dict)
                    or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
                    is None
                ):
                    raise ValueError(f"{name} requires a hash-bound {record_name}")
                record_path = _regular_file(record.get("path", ""), f"{name}.{record_name}")
                if sha256_file(record_path) != record["sha256"]:
                    raise ValueError(f"{name} {record_name} hash mismatch")
                payload = _read_json(record_path)
                if record_name == "selection_manifest" and (
                    payload.get("status") != "VALIDATION_LOCKED"
                    or payload.get("test_access") != "NONE"
                ):
                    raise ValueError(f"{name} union selection is not Validation-locked")
                if record_name == "application_manifest" and (
                    payload.get("status") != "COMPLETE"
                    or payload.get("candidate_test_labels_read") is not False
                    or payload.get("test_access") != "LABEL_FREE_INFERENCE_ONLY"
                ):
                    raise ValueError(f"{name} union application is not label-free and complete")
                if record_name == "selection_manifest":
                    validate_union_selection(record_path)
                else:
                    validate_application_signature(payload, kind="union")
                    selection_path = _regular_file(
                        raw.get("selection_manifest", {}).get("path", ""),
                        f"{name}.selection_manifest",
                    )
                    validate_union_application_outputs(
                        record_path,
                        _read_json(selection_path),
                    )
                locked[f"formal_system_{index:02d}_{name}_{record_name}"] = record_path
        names.add(name)
        kinds.add(kind)
    if not BASE_SYSTEM_KINDS.issubset(kinds) or not kinds.issubset(SYSTEM_KINDS):
        raise ValueError(f"formal plan has incomplete/invalid system kinds: {sorted(kinds)}")
    for route, observed in route_kinds.items():
        if observed != {"native", "ungated", "gated"}:
            raise ValueError(
                f"formal plan route {route} must contain native/ungated/gated; "
                f"observed {sorted(observed)}"
            )
    by_name = {str(raw["name"]): raw for raw in systems}
    for route in ROUTES:
        for kind in ("native", "ungated", "gated"):
            count = sum(
                str(raw.get("route", "")).lower() == route and raw["kind"] == kind
                for raw in systems
            )
            if count != 1:
                raise ValueError(f"formal plan requires exactly one {route}/{kind} system")
    if sum(raw["kind"] == "router" for raw in systems) != 1:
        raise ValueError("formal plan requires exactly one router system")
    union_contract = plan.get("union_contract")
    if not isinstance(union_contract, dict):
        raise ValueError("formal plan must predeclare a union_contract JSON object")
    union_decision = union_contract.get("decision")
    if union_decision not in {"UNION_HEADROOM_AVAILABLE", "NO_UNION_HEADROOM"}:
        raise ValueError("formal plan union_contract has an invalid decision")
    headroom_record = union_contract.get("headroom_manifest")
    if (
        not isinstance(headroom_record, dict)
        or re.fullmatch(r"[0-9a-f]{64}", str(headroom_record.get("sha256", "")))
        is None
    ):
        raise ValueError("formal union_contract requires a hash-bound headroom_manifest")
    headroom_path = _regular_file(
        headroom_record.get("path", ""), "union headroom manifest"
    )
    if sha256_file(headroom_path) != headroom_record["sha256"]:
        raise ValueError("union headroom manifest hash mismatch")
    if provenance.get("union_headroom") != headroom_record:
        raise ValueError("formal provenance and union contract headroom records differ")
    headroom = _read_json(headroom_path)
    if (
        headroom.get("status") != "COMPLETE"
        or headroom.get("decision") != union_decision
        or headroom.get("test_access") != "NONE"
    ):
        raise ValueError("formal union decision differs from its Validation headroom manifest")
    validate_union_headroom(run_dir, headroom_path)
    locked["formal_union_headroom_manifest"] = headroom_path
    union_count = sum(raw["kind"] == "union" for raw in systems)
    if union_decision == "UNION_HEADROOM_AVAILABLE" and union_count != 1:
        raise ValueError("positive union headroom requires exactly one formal union system")
    if union_decision == "NO_UNION_HEADROOM" and union_count != 0:
        raise ValueError("NO_UNION_HEADROOM forbids a formal union ranker")
    if union_count > 1 or (union_count and union_decision != "UNION_HEADROOM_AVAILABLE"):
        raise ValueError("formal union system lacks a positive locked headroom decision")
    for raw in systems:
        reference = raw.get("native_reference")
        if raw["kind"] != "native" and (not reference or str(reference) not in names):
            raise ValueError(f"{raw['name']} requires a valid native_reference")
        if raw["kind"] != "native" and not str(
            raw.get("hypothesis_family", "")
        ).strip():
            raise ValueError(f"{raw['name']} requires a predeclared hypothesis_family")
        if reference:
            referenced = by_name[str(reference)]
            if raw["kind"] == "router":
                if referenced["kind"] != "gated" or str(referenced.get("route", "")).lower() != "crog":
                    raise ValueError("router native_reference must be the locked CROG gated system")
            elif raw["kind"] == "union":
                if referenced["kind"] != "gated" or str(referenced.get("route", "")).lower() != "crog":
                    raise ValueError("union native_reference must be the locked CROG gated system")
            elif referenced["kind"] != "native" or str(
                referenced.get("route", "")
            ).lower() != str(raw.get("route", "")).lower():
                raise ValueError(f"{raw['name']} native_reference is not its route native system")
        base = raw.get("base_ranking_system")
        if base and str(base) not in names:
            raise ValueError(f"{raw['name']} has an unknown base_ranking_system")
        if base:
            base_system = by_name[str(base)]
            if base_system["kind"] in {"router", "union"} or str(base_system.get("route", "")).lower() != str(
                raw.get("route", "")
            ).lower():
                raise ValueError(f"{raw['name']} base ranking is not from the same route")
    return plan, locked


def _read_samples(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"sample_id", "scene_id", "frame_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"formal sample manifest misses columns: {missing}")
    output = frame[list(required)].copy()
    output["sample_id"] = output["sample_id"].astype(str)
    if output.empty or output["sample_id"].eq("").any() or output["sample_id"].duplicated().any():
        raise ValueError("formal sample manifest has invalid sample IDs")
    return output


def _read_pool(path: Path, route: str, sample_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_geometry_sha256",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{route} frozen candidate pool misses columns: {missing}")
    output = frame[list(required)].copy()
    output["sample_id"] = output["sample_id"].astype(str)
    output["candidate_id"] = output["candidate_id"].astype(str)
    output["candidate_geometry_sha256"] = output["candidate_geometry_sha256"].astype(str)
    ranks = pd.to_numeric(output["native_rank"], errors="coerce")
    if (
        ranks.isna().any()
        or not np.equal(ranks, np.floor(ranks)).all()
        or (ranks < 1).any()
        or output["candidate_id"].eq("").any()
        or output["candidate_geometry_sha256"].eq("").any()
        or not set(output["sample_id"]).issubset(sample_ids)
        or output.duplicated(["sample_id", "candidate_id"]).any()
        or output.duplicated(["sample_id", "native_rank"]).any()
    ):
        raise ValueError(f"{route} frozen candidate pool violates identity/rank contracts")
    output["native_rank"] = ranks.astype(int)
    for _sample_id, group in output.groupby("sample_id", sort=False):
        ordered = sorted(group["native_rank"].tolist())
        if ordered != list(range(1, len(ordered) + 1)):
            raise ValueError(f"{route} native ranks are not contiguous")
    return output


def _read_decisions(
    path: Path,
    samples: pd.DataFrame,
    name: str,
    *,
    router: bool = False,
    union: bool = False,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"sample_id", "selected_candidate_id"}
    if router:
        required.add("selected_route")
    if union:
        required.update(
            {"source_route", "source_candidate_id", "candidate_geometry_sha256"}
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} decisions miss columns: {missing}")
    output = samples[["sample_id"]].merge(
        frame[list(required)], on="sample_id", how="left", validate="one_to_one"
    )
    if len(frame) != len(samples) or output["selected_candidate_id"].isna().any():
        raise ValueError(f"{name} decisions do not exactly cover the sample universe")
    output["selected_candidate_id"] = output["selected_candidate_id"].astype(str)
    if router:
        output["selected_route"] = output["selected_route"].astype(str).str.lower()
        if not set(output["selected_route"]).issubset(ROUTES):
            raise ValueError("router decisions contain an unknown route")
    if union:
        output["source_route"] = output["source_route"].astype(str).str.lower()
        output["source_candidate_id"] = output["source_candidate_id"].astype(str)
        output["candidate_geometry_sha256"] = output[
            "candidate_geometry_sha256"
        ].astype(str)
        if not set(output["source_route"]).issubset(ROUTES):
            raise ValueError("union decisions contain an unknown source route")
        qualified = (
            output["source_route"].str.upper()
            + ":"
            + output["source_candidate_id"]
        )
        if not output["selected_candidate_id"].equals(qualified):
            raise ValueError("union decision route-qualified candidate ID mismatch")
    return output


def _read_bound_ranking(
    system: Mapping[str, Any], pool: pd.DataFrame, samples: pd.DataFrame
) -> pd.DataFrame:
    frame = pd.read_parquet(Path(str(system["ranking_path"])))
    rank_column = str(system.get("rank_column", "rank"))
    required = {
        "sample_id",
        "candidate_id",
        rank_column,
        "candidate_geometry_sha256",
        "frozen_native_rank",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{system['name']} ranking misses frozen binding columns: {missing}")
    ranking = frame[list(required)].rename(columns={rank_column: "rank"}).copy()
    ranking["sample_id"] = ranking["sample_id"].astype(str)
    ranking["candidate_id"] = ranking["candidate_id"].astype(str)
    ranking["candidate_geometry_sha256"] = ranking["candidate_geometry_sha256"].astype(str)
    for column in ("rank", "frozen_native_rank"):
        values = pd.to_numeric(ranking[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{system['name']} ranking has invalid {column}")
        ranking[column] = values.astype(int)
    top5 = pool.loc[pool["native_rank"].le(5)].rename(
        columns={"native_rank": "locked_native_rank"}
    )
    keys = ["sample_id", "candidate_id"]
    if set(map(tuple, ranking[keys].to_numpy())) != set(map(tuple, top5[keys].to_numpy())):
        raise ValueError(f"{system['name']} ranking membership differs from frozen Top-5")
    checked = ranking.merge(
        top5[keys + ["candidate_geometry_sha256", "locked_native_rank"]],
        on=keys,
        validate="one_to_one",
        suffixes=("", "_locked"),
    )
    if not checked["candidate_geometry_sha256"].equals(
        checked["candidate_geometry_sha256_locked"]
    ) or not checked["frozen_native_rank"].equals(checked["locked_native_rank"]):
        raise ValueError(f"{system['name']} ranking geometry/native-rank binding mismatch")
    if ranking.duplicated(["sample_id", "rank"]).any() or (ranking["rank"] < 1).any() or (
        ranking["rank"] > 5
    ).any():
        raise ValueError(f"{system['name']} ranking is not a valid Top-5 permutation")
    if system["kind"] == "native" and not ranking["rank"].equals(
        ranking["frozen_native_rank"]
    ):
        raise ValueError(f"{system['name']} native ordering differs from frozen native_rank")
    return ranking


def _read_bound_union_ranking(
    system: Mapping[str, Any],
    pools: Mapping[str, pd.DataFrame],
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Bind a route-qualified Top-15 ranking to the three frozen Top-5 pools."""

    frame = pd.read_parquet(Path(str(system["ranking_path"])))
    rank_column = str(system.get("rank_column", "rank"))
    required = {
        "sample_id",
        "candidate_id",
        rank_column,
        "source_route",
        "source_candidate_id",
        "candidate_geometry_sha256",
        "frozen_native_rank",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{system['name']} union ranking misses binding columns: {missing}")
    ranking = frame[list(required)].rename(columns={rank_column: "rank"}).copy()
    for column in (
        "sample_id",
        "candidate_id",
        "source_candidate_id",
        "candidate_geometry_sha256",
    ):
        ranking[column] = ranking[column].astype(str)
    ranking["source_route"] = ranking["source_route"].astype(str).str.lower()
    if not set(ranking["source_route"]).issubset(ROUTES):
        raise ValueError(f"{system['name']} union ranking contains an unknown route")
    qualified = (
        ranking["source_route"].str.upper()
        + ":"
        + ranking["source_candidate_id"]
    )
    if not ranking["candidate_id"].equals(qualified):
        raise ValueError(f"{system['name']} route-qualified candidate ID mismatch")
    for column in ("rank", "frozen_native_rank"):
        values = pd.to_numeric(ranking[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{system['name']} union ranking has invalid {column}")
        ranking[column] = values.astype(int)
    expected_parts: list[pd.DataFrame] = []
    for route, pool in pools.items():
        part = pool.loc[
            pool["native_rank"].le(5),
            ["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"],
        ].copy()
        part["source_route"] = route
        part["source_candidate_id"] = part["candidate_id"].astype(str)
        part["candidate_id"] = route.upper() + ":" + part["source_candidate_id"]
        part = part.rename(columns={"native_rank": "locked_native_rank"})
        expected_parts.append(part)
    expected = pd.concat(expected_parts, ignore_index=True)
    keys = ["sample_id", "candidate_id"]
    if set(map(tuple, ranking[keys].to_numpy())) != set(
        map(tuple, expected[keys].to_numpy())
    ):
        raise ValueError(
            f"{system['name']} membership differs from the union of frozen route Top-5 pools"
        )
    checked = ranking.merge(
        expected[
            keys
            + [
                "source_route",
                "source_candidate_id",
                "candidate_geometry_sha256",
                "locked_native_rank",
            ]
        ],
        on=keys,
        validate="one_to_one",
        suffixes=("", "_locked"),
    )
    for column in ("source_route", "source_candidate_id", "candidate_geometry_sha256"):
        if not checked[column].equals(checked[f"{column}_locked"]):
            raise ValueError(f"{system['name']} union {column} binding mismatch")
    if not checked["frozen_native_rank"].equals(checked["locked_native_rank"]):
        raise ValueError(f"{system['name']} union native-rank binding mismatch")
    if (
        ranking.duplicated(["sample_id", "rank"]).any()
        or ranking.duplicated(["sample_id", "candidate_id"]).any()
        or (ranking["rank"] < 1).any()
        or (ranking["rank"] > 15).any()
        or not set(ranking["sample_id"]).issubset(set(samples["sample_id"]))
    ):
        raise ValueError(f"{system['name']} union ranking is not a valid Top-15 ranking")
    counts = expected.groupby("sample_id", sort=False).size().to_dict()
    for sample_id, group in ranking.groupby("sample_id", sort=False):
        observed = sorted(group["rank"].tolist())
        if observed != list(range(1, int(counts[str(sample_id)]) + 1)):
            raise ValueError(f"{system['name']} union ranks are not contiguous")
    return ranking


def _validate_selection_contracts(
    plan: Mapping[str, Any], candidate_manifest_path: Path, evaluator_hash_path: Path
) -> None:
    samples = _read_samples(Path(str(plan["sample_manifest"])))
    sample_ids = set(samples["sample_id"])
    label_manifest = _read_json(Path(str(plan["candidate_label_manifest"])))
    evaluator_hash = evaluator_hash_path.read_text(encoding="utf-8").strip()
    if label_manifest["evaluator_sha256"] != evaluator_hash:
        raise ValueError("candidate label manifest evaluator hash does not match P11 evaluator hash")
    candidate_manifest = _read_json(candidate_manifest_path)
    records = candidate_manifest.get("candidate_pools")
    if not isinstance(records, dict) or set(records) != set(ROUTES):
        raise ValueError("P11 candidate_manifests.json must bind all route candidate pools")
    pools: dict[str, pd.DataFrame] = {}
    for route in ROUTES:
        declared = label_manifest["candidate_pools"][route]
        p11 = records[route]
        if (
            Path(str(declared["path"])).resolve() != Path(str(p11.get("path", ""))).resolve()
            or declared["sha256"] != p11.get("sha256")
        ):
            raise ValueError(f"candidate label/P11 candidate-pool manifests disagree: {route}")
        pools[route] = _read_pool(Path(str(declared["path"])), route, sample_ids)
    systems = {str(system["name"]): system for system in plan["systems"]}
    decisions = {
        name: _read_decisions(
            Path(str(system["decisions_path"])),
            samples,
            name,
            router=system["kind"] == "router",
            union=system["kind"] == "union",
        )
        for name, system in systems.items()
    }
    rankings: dict[str, pd.DataFrame] = {}
    for name, system in systems.items():
        if system.get("ranking_path"):
            if system["kind"] == "union":
                rankings[name] = _read_bound_union_ranking(system, pools, samples)
            else:
                rankings[name] = _read_bound_ranking(
                    system, pools[str(system["route"]).lower()], samples
                )
    top1 = {
        name: ranking.loc[ranking["rank"].eq(1)].set_index("sample_id")["candidate_id"]
        for name, ranking in rankings.items()
    }
    gated_by_route: dict[str, pd.Series] = {}
    for route in ROUTES:
        native_name = next(
            name for name, system in systems.items() if system.get("route") == route and system["kind"] == "native"
        )
        ungated_name = next(
            name for name, system in systems.items() if system.get("route") == route and system["kind"] == "ungated"
        )
        gated_name = next(
            name for name, system in systems.items() if system.get("route") == route and system["kind"] == "gated"
        )
        for name in (native_name, ungated_name):
            expected = samples["sample_id"].map(top1[name]).fillna("").astype(str)
            if not decisions[name]["selected_candidate_id"].equals(expected):
                raise ValueError(f"{name} decisions disagree with locked ranking Top-1")
        native_top = samples["sample_id"].map(top1[native_name]).fillna("").astype(str)
        ungated_top = samples["sample_id"].map(top1[ungated_name]).fillna("").astype(str)
        selected = decisions[gated_name]["selected_candidate_id"]
        if not ((selected == native_top) | (selected == ungated_top)).all():
            raise ValueError(f"{gated_name} selects outside native/ungated Top-1 choices")
        gated_by_route[route] = selected.set_axis(samples["sample_id"])
    router_name = next(name for name, system in systems.items() if system["kind"] == "router")
    router = decisions[router_name]
    expected_router = np.asarray(
        [
            gated_by_route[str(route)].loc[sample_id]
            for sample_id, route in zip(samples["sample_id"], router["selected_route"])
        ],
        dtype=object,
    )
    if not np.array_equal(router["selected_candidate_id"].to_numpy(object), expected_router):
        raise ValueError("router selection does not equal the selected route's locked gated selection")
    union_names = [name for name, system in systems.items() if system["kind"] == "union"]
    if union_names:
        union_name = union_names[0]
        union = decisions[union_name]
        expected = samples["sample_id"].map(top1[union_name]).fillna("").astype(str)
        if not union["selected_candidate_id"].equals(expected):
            raise ValueError("union decisions disagree with locked ranking Top-1")
        top = rankings[union_name].loc[rankings[union_name]["rank"].eq(1)].set_index(
            "sample_id"
        )
        for column in (
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ):
            observed = samples["sample_id"].map(top[column]).astype(str)
            if not union[column].equals(observed):
                raise ValueError(f"union decision {column} differs from ranking Top-1")


def _locked_snapshot(path: Path, name: str) -> dict[str, Any]:
    value = _read_json(path)
    if (
        not isinstance(value, dict)
        or int(value.get("schema_version", -1)) != 1
        or value.get("status") != "LOCKED"
    ):
        raise ValueError(f"{name} must be a schema_version=1 LOCKED snapshot")
    return value


def _validate_p11_snapshot_schemas(
    locked: Mapping[str, Path], plan: Mapping[str, Any]
) -> None:
    methods = _locked_snapshot(locked["selected_methods"], "selected_methods")
    features = _locked_snapshot(locked["selected_features"], "selected_features")
    hyperparameters = _locked_snapshot(
        locked["selected_hyperparameters"], "selected_hyperparameters"
    )
    calibration = _locked_snapshot(locked["calibration_manifest"], "calibration_manifest")
    gates = _locked_snapshot(locked["gate_thresholds"], "gate_thresholds")
    router = _locked_snapshot(locked["router_thresholds"], "router_thresholds")
    candidates = _locked_snapshot(locked["candidate_manifests"], "candidate_manifests")
    provenance = dict(plan["bound_provenance"])
    if (
        locked["fold_assignments_sha256"].read_text(encoding="utf-8").strip()
        != provenance["fold_assignments"]["sha256"]
        or locked["evaluator_sha256"].read_text(encoding="utf-8").strip()
        != provenance["evaluator"]["sha256"]
        or locked["code_sha256"].read_text(encoding="utf-8").strip()
        != provenance["code_bundle_sha256"]
    ):
        raise ValueError("P11 hash text files differ from formal bound provenance")
    for name, snapshot in (
        ("selected_methods", methods),
        ("selected_features", features),
        ("selected_hyperparameters", hyperparameters),
        ("calibration_manifest", calibration),
        ("gate_thresholds", gates),
    ):
        routes = snapshot.get("routes")
        if not isinstance(routes, dict) or set(routes) != set(ROUTES):
            raise ValueError(f"{name} must contain exactly crog/g1/c1 route records")
    systems = {str(system["name"]): system for system in plan["systems"]}
    for route in ROUTES:
        method = methods["routes"][route]
        if not isinstance(method, dict):
            raise ValueError(f"selected_methods route record is invalid: {route}")
        expected_names = {
            kind: next(
                name
                for name, system in systems.items()
                if system.get("route") == route and system["kind"] == kind
            )
            for kind in ("native", "ungated", "gated")
        }
        for kind, expected in expected_names.items():
            if method.get(f"{kind}_system") != expected:
                raise ValueError(f"selected_methods {route}/{kind} does not match formal plan")
        if (
            not str(method.get("primary_evidence_track", ""))
            or not str(method.get("encoder", ""))
            or not str(method.get("loss", ""))
            or list(method.get("seeds", [])) != [42, 123, 2026]
        ):
            raise ValueError(f"selected_methods route contract is incomplete: {route}")
        ranker_record = method.get("label_free_test_application")
        if not isinstance(ranker_record, dict):
            raise ValueError(f"selected_methods lacks Test ranker application: {route}")
        ranker_path = _regular_file(
            ranker_record.get("path", ""), f"{route} Test ranker application"
        )
        if sha256_file(ranker_path) != ranker_record.get("sha256"):
            raise ValueError(f"{route} Test ranker application hash mismatch")
        validate_application_signature(_read_json(ranker_path), kind="ranker")
        route_features = features["routes"][route]
        columns = route_features.get("feature_columns") if isinstance(route_features, dict) else None
        if not isinstance(columns, list) or not columns or any(not str(column) for column in columns):
            raise ValueError(f"selected_features route contract is incomplete: {route}")
        route_hyperparameters = hyperparameters["routes"][route]
        if not isinstance(route_hyperparameters, dict) or not route_hyperparameters:
            raise ValueError(f"selected_hyperparameters route contract is empty: {route}")
        route_calibration = calibration["routes"][route]
        if not isinstance(route_calibration, dict) or re.fullmatch(
            r"[0-9a-f]{64}", str(route_calibration.get("manifest_sha256", ""))
        ) is None:
            raise ValueError(f"calibration_manifest route hash is invalid: {route}")
        route_gate = gates["routes"][route]
        if not isinstance(route_gate, dict) or route_gate.get("decision") not in {
            "GO",
            "NO_GO_NATIVE",
        }:
            raise ValueError(f"gate_thresholds route decision is invalid: {route}")
        if (route_gate["decision"] == "GO") != isinstance(
            route_gate.get("operating_point"), dict
        ):
            raise ValueError(f"gate_thresholds operating point disagrees with decision: {route}")
        gate_selection_record = route_gate.get("selection_manifest")
        if not isinstance(gate_selection_record, dict):
            raise ValueError(f"gate_thresholds lacks gate selection: {route}")
        gate_selection_path = _regular_file(
            gate_selection_record.get("path", ""), f"{route} gate selection"
        )
        if sha256_file(gate_selection_path) != gate_selection_record.get("sha256"):
            raise ValueError(f"{route} gate selection hash mismatch")
        gate_selection = validate_gate_selection(gate_selection_path)
        if (
            gate_selection.get("decision") != route_gate.get("decision")
            or gate_selection.get("selection", {}).get("selected_operating_point")
            != route_gate.get("operating_point")
        ):
            raise ValueError(f"{route} gate snapshot differs from recomputed selection")
        gate_record = route_gate.get("label_free_test_application")
        if not isinstance(gate_record, dict):
            raise ValueError(f"gate_thresholds lacks Test application: {route}")
        gate_path = _regular_file(
            gate_record.get("path", ""), f"{route} Test gate application"
        )
        if sha256_file(gate_path) != gate_record.get("sha256"):
            raise ValueError(f"{route} Test gate application hash mismatch")
        gate_application = _read_json(gate_path)
        if gate_application.get("selected_operating_point") != route_gate.get(
            "operating_point"
        ):
            raise ValueError(f"{route} Test gate operating point binding mismatch")
        validate_application_signature(gate_application, kind="gate")
        validate_gate_application_outputs(gate_path, gate_selection)
    router_system = next(name for name, system in systems.items() if system["kind"] == "router")
    if (
        str(router.get("default_route", "")).lower() != "crog"
        or list(router.get("tie_break", [])) != ["G1", "C1"]
        or router.get("system_name") != router_system
        or router.get("decision") not in {"GO", "NO_GO_CROG"}
        or (router["decision"] == "GO")
        != isinstance(router.get("operating_point"), dict)
    ):
        raise ValueError("router_thresholds contract is incomplete or differs from plan")
    router_record = router.get("label_free_test_application")
    if not isinstance(router_record, dict):
        raise ValueError("router_thresholds lacks its Test application")
    router_path = _regular_file(
        router_record.get("path", ""), "Test route-router application"
    )
    if sha256_file(router_path) != router_record.get("sha256"):
        raise ValueError("Test route-router application hash mismatch")
    router_application = _read_json(router_path)
    if router_application.get("configuration", {}).get(
        "selected_operating_point"
    ) != router.get("operating_point"):
        raise ValueError("Test route-router operating point binding mismatch")
    validate_application_signature(router_application, kind="router")
    router_selection_record = router.get("selection_manifest")
    if not isinstance(router_selection_record, dict):
        raise ValueError("router_thresholds lacks its Validation selection")
    router_selection_path = _regular_file(
        router_selection_record.get("path", ""), "route-router selection"
    )
    if sha256_file(router_selection_path) != router_selection_record.get("sha256"):
        raise ValueError("route-router selection hash mismatch")
    recomputed_router = validate_router_selection(router_selection_path)
    validate_router_application_outputs(router_path, recomputed_router)
    if (
        recomputed_router.get("decision") != router.get("decision")
        or recomputed_router.get("selection", {}).get("selected_operating_point")
        != router.get("operating_point")
    ):
        raise ValueError("router snapshot differs from recomputed Validation selection")
    records = candidates.get("candidate_pools")
    if not isinstance(records, dict) or set(records) != set(ROUTES):
        raise ValueError("candidate_manifests must contain all route candidate pools")
    union_systems = [
        (name, system) for name, system in systems.items() if system["kind"] == "union"
    ]
    union_decision = dict(plan.get("union_contract") or {}).get("decision")
    if union_decision == "UNION_HEADROOM_AVAILABLE":
        if len(union_systems) != 1:
            raise ValueError("positive union contract requires one selected union system")
        union_name, _ = union_systems[0]
        method = methods.get("union")
        feature = features.get("union")
        hyperparameter = hyperparameters.get("union")
        if (
            not isinstance(method, dict)
            or method.get("system_name") != union_name
            or not str(method.get("encoder", ""))
            or list(method.get("seeds", [])) != [42, 123, 2026]
        ):
            raise ValueError("selected_methods union contract is incomplete")
        if (
            not isinstance(feature, dict)
            or not isinstance(feature.get("feature_columns"), list)
            or not feature["feature_columns"]
        ):
            raise ValueError("selected_features union contract is incomplete")
        if not isinstance(hyperparameter, dict) or not hyperparameter:
            raise ValueError("selected_hyperparameters union contract is empty")
    elif union_decision == "NO_UNION_HEADROOM":
        if union_systems or any(
            snapshot.get("union") is not None
            for snapshot in (methods, features, hyperparameters)
        ):
            raise ValueError("NO_UNION_HEADROOM forbids selected union snapshots")


def create_unified_formal_lock(
    *,
    run_dir: Path,
    evaluation_plan_path: Path,
    extra_locked_files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Lock all P11 selections and label-free formal-evaluation inputs."""

    root = run_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if int(manifest.get("formal_test_execution_count", -1)) != 0:
        raise PermissionError("formal Test execution count must be zero before locking")
    if manifest.get("test_label_state") not in {
        "VALIDATION_SELECTION_COMPLETE",
        "FORMAL_TEST_READY",
    }:
        raise PermissionError("run manifest is not in a pre-Test lockable state")
    lock_dir = root / "08_lock"
    locked: dict[str, Path] = {}
    for name, filename in REQUIRED_LOCK_FILES.items():
        path = _regular_file(lock_dir / filename, name)
        if filename.endswith("_sha256.txt"):
            _validate_hash_text(path, name)
        if filename == "PRIMARY_METHOD_DECLARATION.md" and not path.read_text(
            encoding="utf-8"
        ).strip():
            raise ValueError("PRIMARY_METHOD_DECLARATION.md is empty")
        locked[name] = path
    plan_path = _regular_file(evaluation_plan_path, "formal_evaluation_plan")
    plan, plan_locked = validate_formal_evaluation_plan(plan_path)
    _validate_p11_snapshot_schemas(locked, plan)
    _validate_selection_contracts(
        plan,
        locked["candidate_manifests"],
        locked["evaluator_sha256"],
    )
    locked["formal_evaluation_plan"] = plan_path
    locked.update(plan_locked)
    for name, raw_path in sorted(dict(extra_locked_files or {}).items()):
        normalized = str(name)
        if not normalized or normalized in locked:
            raise ValueError(f"invalid or duplicate extra locked-file name: {normalized}")
        locked[normalized] = _regular_file(raw_path, normalized)
    declaration = {
        "status": "PRIMARY_METHODS_AND_POLICIES_FROZEN",
        "evaluation_plan_schema_version": 1,
        "evaluation_plan_sha256": sha256_file(plan_path),
        "evaluation_plan_canonical_sha256": canonical_sha256(plan),
        "system_names": [str(system["name"]) for system in plan["systems"]],
        "system_kinds": [str(system["kind"]) for system in plan["systems"]],
        "routes": list(ROUTES),
        "candidate_test_labels_read": False,
        "required_p11_artifacts": list(REQUIRED_LOCK_FILES),
    }
    lock_path = create_formal_test_lock(
        root, declaration=declaration, locked_files=locked
    )
    verified = verify_formal_test_lock(root)
    if verified["self_sha256"] != _read_json(lock_path)["self_sha256"]:
        raise RuntimeError("formal Test lock verification changed its self hash")
    return verified


def execute_create_unified_formal_lock(
    *,
    run_dir: Path,
    evaluation_plan_path: Path,
    extra_locked_files: Mapping[str, str | Path] | None = None,
    command: str = "",
) -> dict[str, Any]:
    if (run_dir.resolve() / "08_lock" / "FORMAL_TEST_LOCK.json").exists() or (
        run_dir.resolve() / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    ).exists():
        raise FileExistsError("formal Test lock/execution already exists")
    with ledger_stage(
        run_dir.resolve() / "run_ledger.sqlite",
        stage="P11",
        substage="formal_test_lock",
        method="content_hash_verified_lock",
        command=command,
    ) as state:
        result = create_unified_formal_lock(
            run_dir=run_dir,
            evaluation_plan_path=evaluation_plan_path,
            extra_locked_files=extra_locked_files,
        )
        path = run_dir.resolve() / "08_lock" / "FORMAL_TEST_LOCK.json"
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evaluation-plan", required=True, type=Path)
    parser.add_argument(
        "--extra-locked-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    return parser.parse_args()


def _parse_extra(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--extra-locked-file must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"duplicate/empty extra locked-file name: {name}")
        result[name] = Path(path).resolve()
    return result


def main() -> int:
    args = parse_args()
    execute_create_unified_formal_lock(
        run_dir=args.run_dir.resolve(),
        evaluation_plan_path=args.evaluation_plan.resolve(),
        extra_locked_files=_parse_extra(args.extra_locked_file),
        command=" ".join(map(str, sys.argv)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_LOCK_FILES",
    "create_unified_formal_lock",
    "execute_create_unified_formal_lock",
    "validate_formal_evaluation_plan",
]
