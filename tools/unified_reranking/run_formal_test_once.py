"""Execute the hash-locked unified formal Test exactly once."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.evaluator_adapter import (
    evaluate_candidate_rows_with_frozen_evaluator,
)
from unified_reranking.ledger import ledger_stage
from unified_reranking.lock import (
    claim_formal_test_execution,
    finalize_formal_test_execution,
    verify_formal_test_lock,
)
from unified_reranking.metrics import compare_selections, evaluate_order_only
from unified_reranking.statistics import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    paired_system_statistics,
    three_system_paired_tests,
)
from unified_reranking.test_access_guard import append_access_log, assert_test_labels_unlocked
from unified_reranking.test_bridge import (
    BRIDGE_COLUMNS,
    validate_label_free_test_bridge_manifest,
)


ROUTES = ("crog", "g1", "c1")
CLAIM_SENTINEL = "FORMAL_TEST_CLAIM.sentinel"


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _claim_exclusively(run_dir: Path) -> Path:
    destination = run_dir / "09_formal_test" / CLAIM_SENTINEL
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "purpose": "exclusive exactly-once formal-Test claim",
        },
        sort_keys=True,
    ).encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise PermissionError("formal Test exclusive claim already exists") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _claim_before_label_access(run_dir: Path) -> dict[str, Any]:
    sentinel = _claim_exclusively(run_dir)
    try:
        claim = claim_formal_test_execution(run_dir)
    except Exception:
        execution = run_dir / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
        if not execution.exists():
            sentinel.unlink(missing_ok=True)
        raise
    append_access_log(
        run_dir,
        {
            "event": "formal_test_exclusive_claim_created",
            "claim_sentinel": str(sentinel.resolve()),
            "execution_count": claim["execution_count"],
        },
    )
    return claim


def _read_candidate_labels_once(path: Path) -> tuple[pd.DataFrame, str]:
    """Open label-file bytes once; hash and parse the same in-memory payload."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular candidate-level Test label file: {path}")
    with path.open("rb") as stream:
        payload = stream.read()
    digest = hashlib.sha256(payload).hexdigest()
    frame = pd.read_parquet(io.BytesIO(payload))
    return frame, digest


def _read_bridge_ground_truth_once(path: Path) -> tuple[pd.DataFrame, str]:
    """Hash and parse the locked historical GT from one byte-stream open."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular historical Test ground truth: {path}")
    with path.open("rb") as stream:
        payload = stream.read()
    digest = hashlib.sha256(payload).hexdigest()
    frame = pd.read_parquet(io.BytesIO(payload))
    return frame, digest


def _locked_plan(lock: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    record = dict(lock.get("locked_files", {})).get("formal_evaluation_plan")
    if not isinstance(record, dict):
        raise PermissionError("formal lock does not contain formal_evaluation_plan")
    path = Path(str(record.get("path", "")))
    if sha256_file(path) != record.get("sha256"):
        raise PermissionError("formal evaluation plan hash mismatch")
    return path, _read_json(path)


def _sample_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"sample_id", "scene_id", "frame_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"formal sample manifest misses columns: {missing}")
    output = frame[["sample_id", "scene_id", "frame_id"]].copy()
    for column in output:
        if output[column].isna().any() or output[column].astype(str).eq("").any():
            raise ValueError(f"formal sample manifest has empty {column}")
        output[column] = output[column].astype(str)
    if output["sample_id"].duplicated().any() or output.empty:
        raise ValueError("formal sample manifest sample IDs must be unique and non-empty")
    return output


def _validate_labels(
    frame: pd.DataFrame,
    sample_ids: set[str],
    candidate_pools: Mapping[str, pd.DataFrame],
    label_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    normalization = dict(label_manifest.get("normalization", {}))
    route_column = str(normalization.get("route_column", "route"))
    variant_column = str(normalization.get("variant_column", "variant"))
    include_variants = normalization.get("include_variants")
    if include_variants is not None:
        if variant_column not in frame.columns:
            raise ValueError("candidate-level Test labels miss declared variant column")
        frame = frame.loc[
            frame[variant_column].astype(str).isin(map(str, include_variants))
        ].copy()
    required = {route_column, "sample_id", "candidate_id", "candidate_success"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate-level Test labels miss columns: {missing}")
    labels = frame[list(required)].rename(columns={route_column: "route"}).copy()
    labels["route"] = labels["route"].astype(str).str.lower()
    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    if not set(labels["route"]).issubset(ROUTES) or not set(labels["sample_id"]).issubset(
        sample_ids
    ):
        raise ValueError("candidate labels contain an unknown route or sample")
    numeric = pd.to_numeric(labels["candidate_success"], errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise ValueError("candidate_success must be binary")
    labels["candidate_success"] = numeric.astype(bool)
    if labels.duplicated(["route", "sample_id", "candidate_id"]).any():
        raise ValueError("candidate labels contain duplicate route/sample/candidate keys")
    observed = set(map(tuple, labels[["route", "sample_id", "candidate_id"]].to_numpy()))
    expected = {
        (route, str(row.sample_id), str(row.candidate_id))
        for route, pool in candidate_pools.items()
        for row in pool[["sample_id", "candidate_id"]].itertuples(index=False)
    }
    if observed != expected:
        missing = len(expected.difference(observed))
        extra = len(observed.difference(expected))
        raise ValueError(
            f"candidate labels do not exactly cover locked All pools: missing={missing}, extra={extra}"
        )
    return labels


def _decisions(
    system: Mapping[str, Any],
    sample_manifest: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_parquet(Path(str(system["decisions_path"])))
    required = {"sample_id", "selected_candidate_id"}
    if system["kind"] == "router":
        required.add("selected_route")
    if system["kind"] == "union":
        required.update(
            {"source_route", "source_candidate_id", "candidate_geometry_sha256"}
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{system['name']} decisions miss columns: {missing}")
    columns = list(required)
    output = sample_manifest[["sample_id"]].merge(
        frame[columns], on="sample_id", how="left", validate="one_to_one"
    )
    if len(frame) != len(sample_manifest) or output["selected_candidate_id"].isna().any():
        raise ValueError(f"{system['name']} decisions do not exactly cover samples")
    output["selected_candidate_id"] = output["selected_candidate_id"].astype(str)
    if system["kind"] == "router":
        output["selected_route"] = output["selected_route"].astype(str).str.lower()
        if not set(output["selected_route"]).issubset(ROUTES):
            raise ValueError("router selected an unknown route")
    elif system["kind"] == "union":
        output["source_route"] = output["source_route"].astype(str).str.lower()
        output["source_candidate_id"] = output["source_candidate_id"].astype(str)
        output["candidate_geometry_sha256"] = output[
            "candidate_geometry_sha256"
        ].astype(str)
        if not set(output["source_route"]).issubset(ROUTES):
            raise ValueError("union selected an unknown source route")
        qualified = (
            output["source_route"].str.upper()
            + ":"
            + output["source_candidate_id"]
        )
        if not output["selected_candidate_id"].equals(qualified):
            raise ValueError("union selected candidate is not route-qualified")
        output["selected_route"] = output["source_route"]
    else:
        output["selected_route"] = str(system["route"]).lower()
    output["label_candidate_id"] = (
        output["source_candidate_id"]
        if system["kind"] == "union"
        else output["selected_candidate_id"]
    )
    joined = output.merge(
        labels.rename(
            columns={
                "route": "selected_route",
                "candidate_id": "label_candidate_id",
                "candidate_success": "selected_correct",
            }
        ),
        on=["selected_route", "sample_id", "label_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    nonempty = joined["selected_candidate_id"].ne("")
    if joined.loc[nonempty, "selected_correct"].isna().any():
        raise ValueError(f"{system['name']} selected a candidate absent from locked labels")
    joined["selected_correct"] = joined["selected_correct"].fillna(False).astype(bool)
    joined["candidate_count"] = nonempty.astype(int)
    columns = [
        "sample_id",
        "selected_route",
        "selected_candidate_id",
        "selected_correct",
        "candidate_count",
    ]
    if system["kind"] == "union":
        columns.extend(
            ["source_route", "source_candidate_id", "candidate_geometry_sha256"]
        )
    return joined[columns]


def _explicit_ranking(system: Mapping[str, Any], sample_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(Path(str(system["ranking_path"])))
    rank_column = str(system.get("rank_column", "rank"))
    required = {
        "sample_id",
        "candidate_id",
        rank_column,
        "candidate_geometry_sha256",
        "frozen_native_rank",
    }
    if system["kind"] == "union":
        required.update({"source_route", "source_candidate_id"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{system['name']} ranking misses columns: {missing}")
    output = frame[list(required)].rename(
        columns={rank_column: "rank"}
    )
    output["sample_id"] = output["sample_id"].astype(str)
    output["candidate_id"] = output["candidate_id"].astype(str)
    output["candidate_geometry_sha256"] = output["candidate_geometry_sha256"].astype(str)
    if system["kind"] == "union":
        output["source_route"] = output["source_route"].astype(str).str.lower()
        output["source_candidate_id"] = output["source_candidate_id"].astype(str)
        if not set(output["source_route"]).issubset(ROUTES):
            raise ValueError(f"{system['name']} ranking contains an unknown source route")
        qualified = (
            output["source_route"].str.upper()
            + ":"
            + output["source_candidate_id"]
        )
        if not output["candidate_id"].equals(qualified):
            raise ValueError(f"{system['name']} ranking candidate ID is not route-qualified")
    maximum_rank = 15 if system["kind"] == "union" else 5
    rank = pd.to_numeric(output["rank"], errors="coerce")
    if (
        rank.isna().any()
        or not np.equal(rank, np.floor(rank)).all()
        or (rank < 1).any()
        or (rank > maximum_rank).any()
        or not set(output["sample_id"]).issubset(sample_ids)
    ):
        raise ValueError(f"{system['name']} ranking has invalid Top-{maximum_rank} ranks")
    output["rank"] = rank.astype(int)
    if output.duplicated(["sample_id", "candidate_id"]).any() or output.duplicated(
        ["sample_id", "rank"]
    ).any():
        raise ValueError(f"{system['name']} ranking has duplicate identities/ranks")
    return output


def _locked_label_contract(
    lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    record = dict(lock.get("locked_files", {})).get("formal_candidate_label_manifest")
    if not isinstance(record, dict):
        raise PermissionError("formal lock has no candidate-label provenance manifest")
    manifest_path = Path(str(record.get("path", "")))
    if manifest_path.resolve() != Path(str(plan["candidate_label_manifest"])).resolve():
        raise PermissionError("evaluation plan and lock candidate-label manifests differ")
    if sha256_file(manifest_path) != record.get("sha256"):
        raise PermissionError("candidate-label provenance manifest hash mismatch")
    manifest = _read_json(manifest_path)
    pools: dict[str, pd.DataFrame] = {}
    for route in ROUTES:
        pool_record = dict(lock["locked_files"]).get(
            f"formal_{route}_all_candidate_pool"
        )
        declared = manifest["candidate_pools"][route]
        if not isinstance(pool_record, dict):
            raise PermissionError(f"formal lock misses {route} All candidate pool")
        path = Path(str(pool_record.get("path", "")))
        if (
            path.resolve() != Path(str(declared["path"])).resolve()
            or pool_record.get("sha256") != declared["sha256"]
            or sha256_file(path) != declared["sha256"]
        ):
            raise PermissionError(f"locked candidate-pool contract mismatch: {route}")
        pool = pd.read_parquet(path)
        required = {"sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"}
        missing = sorted(required.difference(pool.columns))
        if missing:
            raise ValueError(f"locked {route} All pool misses columns: {missing}")
        pools[route] = pool[list(required)].copy()
        pools[route]["sample_id"] = pools[route]["sample_id"].astype(str)
        pools[route]["candidate_id"] = pools[route]["candidate_id"].astype(str)
    return manifest, pools


def _locked_test_bridge_contract(
    lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Path]]:
    contract = plan.get("test_bridge_contract")
    if not isinstance(contract, dict):
        raise PermissionError("formal plan has no Test bridge contract")
    locked = dict(lock.get("locked_files", {}))
    manifest_record = locked.get("formal_test_bridge_manifest")
    if not isinstance(manifest_record, dict) or manifest_record != contract.get("manifest"):
        raise PermissionError("formal lock and plan Test bridge manifests differ")
    manifest_path = Path(str(manifest_record.get("path", ""))).resolve()
    if sha256_file(manifest_path) != manifest_record.get("sha256"):
        raise PermissionError("locked Test bridge manifest hash mismatch")
    manifest, bundle, paths = validate_label_free_test_bridge_manifest(
        manifest_path,
        expected_denominator=contract.get("denominator"),
        expected_evaluator=contract.get("evaluator"),
    )
    required_locked = {
        "formal_test_bridge_candidate_bundle": manifest["artifacts"]["candidate_bundle"],
        "formal_test_bridge_ground_truth": manifest["sources"]["historical_ground_truth"],
    }
    for name, record in required_locked.items():
        if locked.get(name) != record:
            raise PermissionError(f"formal lock Test bridge binding mismatch: {name}")
    for name, record in manifest["sources"].items():
        if locked.get(f"formal_test_bridge_source_{name}") != record:
            raise PermissionError(f"formal lock Test bridge source mismatch: {name}")
    return manifest, bundle, paths


def _evaluate_locked_test_bridge(
    *,
    root: Path,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    sample_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest, bundle, paths = _locked_test_bridge_contract(lock, plan)
    ground_truth_path = paths["source_historical_ground_truth"]
    ground_truth, ground_truth_sha256 = _read_bridge_ground_truth_once(
        ground_truth_path
    )
    if ground_truth_sha256 != manifest["sources"]["historical_ground_truth"]["sha256"]:
        raise PermissionError("historical Test ground-truth hash differs from formal lock")
    required = {"sample_id", "gt_grasp_rectangles"}
    missing = sorted(required.difference(ground_truth.columns))
    if missing:
        raise ValueError(f"historical Test ground truth misses columns: {missing}")
    ground_truth["sample_id"] = ground_truth["sample_id"].astype(str)
    expected_samples = set(sample_manifest["sample_id"].astype(str))
    if (
        ground_truth["sample_id"].duplicated().any()
        or set(ground_truth["sample_id"]) != expected_samples
    ):
        raise ValueError("historical Test ground truth does not exactly cover denominator")
    append_access_log(
        root,
        {
            "event": "formal_bridge_test_ground_truth_read_once",
            "path": str(ground_truth_path),
            "sha256": ground_truth_sha256,
            "row_count": int(len(ground_truth)),
            "execution_count": 1,
        },
    )
    evaluator_record = manifest["sources"]["evaluator"]
    evaluated = evaluate_candidate_rows_with_frozen_evaluator(
        bundle,
        ground_truth,
        Path(str(evaluator_record["path"])),
        str(evaluator_record["sha256"]),
    )
    output = evaluated[[*BRIDGE_COLUMNS, "candidate_success"]].copy()
    if (
        len(output) != len(bundle)
        or output.duplicated(
            ["route", "candidate_pool_contract", "sample_id", "candidate_id"]
        ).any()
        or not output["candidate_success"].isin([True, False]).all()
        or not output[list(BRIDGE_COLUMNS)].equals(bundle[list(BRIDGE_COLUMNS)])
    ):
        raise RuntimeError("formal Test bridge output changed locked membership/geometry")
    return output, {
        "manifest": {
            "path": str(paths["manifest"]),
            "sha256": sha256_file(paths["manifest"]),
        },
        "candidate_bundle": manifest["artifacts"]["candidate_bundle"],
        "historical_ground_truth": {
            "path": str(ground_truth_path),
            "sha256": ground_truth_sha256,
            "opened_once_after_claim": True,
            "row_count": int(len(ground_truth)),
        },
        "evaluator": evaluator_record,
    }


def _move_selected_first(base: pd.DataFrame, selected: pd.DataFrame, name: str) -> pd.DataFrame:
    selected_ids = selected.set_index("sample_id")["selected_candidate_id"].to_dict()
    rows: list[pd.DataFrame] = []
    for sample_id, group in base.groupby("sample_id", sort=False):
        candidate_id = str(selected_ids.get(str(sample_id), ""))
        ordered = group.sort_values("rank", kind="mergesort").copy()
        if candidate_id:
            if candidate_id not in set(ordered["candidate_id"]):
                raise ValueError(f"{name} selected candidate is absent from its base ranking")
            ordered["_selected"] = ordered["candidate_id"].eq(candidate_id)
            ordered = ordered.sort_values(
                ["_selected", "rank"], ascending=[False, True], kind="mergesort"
            ).drop(columns="_selected")
            ordered["rank"] = np.arange(1, len(ordered) + 1)
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True) if rows else base.copy()


def _evaluate_ranking(
    ranking: pd.DataFrame,
    system: Mapping[str, Any],
    labels: pd.DataFrame,
    sample_ids: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if system["kind"] == "union":
        evaluation = ranking.merge(
            labels.rename(
                columns={
                    "route": "source_route",
                    "candidate_id": "source_candidate_id",
                }
            )[["source_route", "sample_id", "source_candidate_id", "candidate_success"]],
            on=["source_route", "sample_id", "source_candidate_id"],
            how="left",
            validate="one_to_one",
        )
        max_k = 15
    else:
        route = str(system["route"]).lower()
        evaluation = ranking.merge(
            labels.loc[
                labels["route"].eq(route),
                ["sample_id", "candidate_id", "candidate_success"],
            ],
            on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
        max_k = 5
    if evaluation["candidate_success"].isna().any():
        raise ValueError("ranking contains candidates absent from candidate-level labels")
    evaluation["native_rank"] = evaluation["rank"]
    evaluation["formal_score"] = -evaluation["rank"].astype(float)
    metrics, per_sample = evaluate_order_only(
        sample_ids, evaluation, score_column="formal_score", max_k=max_k
    )
    if system["kind"] == "union":
        # The metrics helper uses legacy `*_at_5` names for its terminal-K
        # summaries; expose the semantically exact Top-15 aliases in the formal result.
        metrics["oracle_at_15"] = metrics.pop("oracle_at_5")
        metrics["mrr_at_15"] = metrics.pop("mrr_at_5")
    return metrics, per_sample, evaluation


def _assert_same_candidate_universe(
    native: pd.DataFrame, challenger: pd.DataFrame, system_name: str
) -> None:
    left = native.groupby("sample_id")["candidate_id"].agg(lambda values: tuple(sorted(values)))
    right = challenger.groupby("sample_id")["candidate_id"].agg(lambda values: tuple(sorted(values)))
    all_ids = left.index.union(right.index)
    if not left.reindex(all_ids).fillna("").equals(right.reindex(all_ids).fillna("")):
        raise RuntimeError(f"{system_name} changed the locked Top-5 candidate universe")


def _canonical_selection_keys(frame: pd.DataFrame) -> pd.Series:
    """Return route+raw-candidate keys without double-qualifying union IDs."""

    route = frame["selected_route"].astype(str).str.lower()
    selected = frame["selected_candidate_id"].fillna("").astype(str)
    source = (
        frame["source_candidate_id"].fillna("").astype(str)
        if "source_candidate_id" in frame
        else pd.Series("", index=frame.index, dtype=str)
    )
    keys = []
    for route_value, selected_value, source_value in zip(
        route, selected, source, strict=True
    ):
        prefix = route_value.upper() + ":"
        raw = source_value or (
            selected_value[len(prefix) :]
            if selected_value.startswith(prefix)
            else selected_value
        )
        keys.append(f"{route_value}::{raw}")
    return pd.Series(keys, index=frame.index, dtype=str)


def _system_metrics_and_statistics(
    plan: Mapping[str, Any],
    sample_manifest: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    sample_ids = sample_manifest["sample_id"].tolist()
    sample_set = set(sample_ids)
    systems = {str(value["name"]): dict(value) for value in plan["systems"]}
    decisions = {
        name: _decisions(system, sample_manifest, labels)
        for name, system in systems.items()
    }
    rankings: dict[str, pd.DataFrame] = {}
    for name, system in systems.items():
        if system.get("ranking_path"):
            rankings[name] = _explicit_ranking(system, sample_set)
    unresolved = {
        name for name, system in systems.items() if system["kind"] == "gated" and name not in rankings
    }
    while unresolved:
        progress = False
        for name in tuple(unresolved):
            base_name = str(systems[name]["base_ranking_system"])
            if base_name in rankings:
                rankings[name] = _move_selected_first(rankings[base_name], decisions[name], name)
                unresolved.remove(name)
                progress = True
        if not progress:
            raise ValueError("gated base-ranking dependencies contain a cycle")

    metrics: dict[str, Any] = {}
    per_sample_parts: list[pd.DataFrame] = []
    realized_rankings: list[pd.DataFrame] = []
    scored_candidate_parts: list[pd.DataFrame] = []
    evaluated_rankings: dict[str, pd.DataFrame] = {}
    for name, system in systems.items():
        selection = decisions[name].copy()
        if name in rankings:
            system_metrics, ranked_sample, evaluation = _evaluate_ranking(
                rankings[name], system, labels, sample_ids
            )
            top = ranked_sample.set_index("sample_id")
            expected = selection.set_index("sample_id")
            if not top["selected_candidate_id"].fillna("").astype(str).equals(
                expected["selected_candidate_id"].fillna("").astype(str)
            ):
                raise RuntimeError(f"{name} decision and ranking Top-1 disagree")
            selection = selection.drop(columns="candidate_count").merge(
                ranked_sample.drop(
                    columns=["selected_candidate_id", "selected_correct"]
                ),
                on="sample_id",
                validate="one_to_one",
            )
            metrics[name] = dict(system_metrics)
            realized = rankings[name].copy()
            realized["system_name"] = name
            realized["route"] = (
                realized["source_route"]
                if system["kind"] == "union"
                else str(system["route"]).lower()
            )
            realized_rankings.append(realized)
            evaluated_rankings[name] = evaluation
            scored = evaluation.copy()
            scored["system_name"] = name
            scored["system_kind"] = system["kind"]
            scored["route"] = (
                scored["source_route"]
                if system["kind"] == "union"
                else str(system["route"]).lower()
            )
            scored_candidate_parts.append(scored)
        else:
            numerator = int(selection["selected_correct"].sum())
            metrics[name] = {
                "sample_count": len(selection),
                "j_at_1_numerator": numerator,
                "j_at_1": numerator / len(selection),
                "j_at_2": None,
                "j_at_3": None,
                "j_at_4": None,
                "j_at_5": None,
                "oracle_at_5": None,
                "mrr_at_5": None,
                "ndcg_at_1": numerator / len(selection),
                "ndcg_at_5": None,
            }
        route = system.get("route")
        if route in ROUTES:
            all_route = labels.loc[labels["route"].eq(route)]
            oracle_by_sample = all_route.groupby("sample_id")["candidate_success"].any()
            oracle_all_numerator = int(
                sample_manifest["sample_id"].map(oracle_by_sample).fillna(False).sum()
            )
            metrics[name]["oracle_all_numerator"] = oracle_all_numerator
            metrics[name]["oracle_all"] = oracle_all_numerator / len(sample_manifest)
            metrics[name]["ndcg_at_1"] = metrics[name]["j_at_1"]
        elif system["kind"] == "union":
            oracle_by_sample = labels.groupby("sample_id")["candidate_success"].any()
            oracle_all_numerator = int(
                sample_manifest["sample_id"].map(oracle_by_sample).fillna(False).sum()
            )
            metrics[name]["oracle_all_numerator"] = oracle_all_numerator
            metrics[name]["oracle_all"] = oracle_all_numerator / len(sample_manifest)
            metrics[name]["ndcg_at_1"] = metrics[name]["j_at_1"]
        selection["system_name"] = name
        selection["system_kind"] = system["kind"]
        per_sample_parts.append(selection)

    for route in ROUTES:
        route_systems = {
            system["kind"]: name
            for name, system in systems.items()
            if system.get("route") == route
        }
        native_name = route_systems["native"]
        native_ranking = evaluated_rankings[native_name]
        for kind in ("ungated", "gated"):
            name = route_systems[kind]
            _assert_same_candidate_universe(native_ranking, evaluated_rankings[name], name)
            if metrics[name]["j_at_5_numerator"] != metrics[native_name]["j_at_5_numerator"]:
                raise RuntimeError(f"{name} changed J@5 under an order-only contract")

    per_sample = pd.concat(per_sample_parts, ignore_index=True).merge(
        sample_manifest,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    realized = (
        pd.concat(realized_rankings, ignore_index=True)
        if realized_rankings
        else pd.DataFrame(columns=["system_name", "route", "sample_id", "candidate_id", "rank"])
    )
    selection_by_name = {
        name: part.sort_values("sample_id")
        for name, part in per_sample.groupby("system_name", sort=False)
    }
    statistics: dict[str, Any] = {"comparisons": {}}
    family_members: dict[str, list[str]] = {}
    for name, system in systems.items():
        if system["kind"] == "native":
            continue
        reference_name = str(system["native_reference"])
        reference = selection_by_name[reference_name]
        challenger = selection_by_name[name]
        oracle = (
            metrics[name].get("oracle_at_15")
            if system["kind"] == "union"
            else metrics[reference_name].get("oracle_at_5")
        )
        reference_for_comparison = reference.copy()
        challenger_for_comparison = challenger.copy()
        for frame in (reference_for_comparison, challenger_for_comparison):
            frame["selected_candidate_id"] = _canonical_selection_keys(frame)
        comparison = compare_selections(
            reference_for_comparison,
            challenger_for_comparison,
            oracle_at_5=float(oracle if oracle is not None else reference["selected_correct"].mean()),
        )
        if system["kind"] == "union":
            comparison["headroom_recovery_at_15"] = comparison.pop(
                "headroom_recovery_at_5"
            )
        paired = paired_system_statistics(
            reference["selected_correct"],
            challenger["selected_correct"],
            scene_ids=sample_manifest.sort_values("sample_id")["scene_id"],
            frame_ids=sample_manifest.sort_values("sample_id")["frame_id"],
        )
        statistics["comparisons"][name] = {
            "reference": reference_name,
            "hypothesis_family": str(system["hypothesis_family"]),
            "selection_comparison": comparison,
            **paired,
        }
        family_members.setdefault(str(system["hypothesis_family"]), []).append(name)
    for family, names in sorted(family_members.items()):
        pvalues = [
            float(
                statistics["comparisons"][name]["mcnemar_conventional_supportive"][
                    "pvalue"
                ]
            )
            for name in names
        ]
        adjusted = multipletests(pvalues, method="holm")[1]
        for name, value in zip(names, adjusted):
            statistics["comparisons"][name]["holm_adjusted_mcnemar_pvalue"] = float(value)
    statistics["hypothesis_families"] = {
        family: {
            "members": names,
            "correction": "Holm-Bonferroni within predeclared family",
            "comparison_count": len(names),
        }
        for family, names in sorted(family_members.items())
    }
    final_gated = {
        route.upper(): selection_by_name[
            next(
                name
                for name, system in systems.items()
                if system.get("route") == route and system["kind"] == "gated"
            )
        ]["selected_correct"]
        for route in ROUTES
    }
    statistics["three_route_final_systems"] = three_system_paired_tests(final_gated)
    ordered_gated = [
        selection_by_name[
            next(
                name
                for name, system in systems.items()
                if system.get("route") == route and system["kind"] == "gated"
            )
        ]
        .sort_values("sample_id")["selected_correct"]
        .to_numpy(bool)
        for route in ROUTES
    ]
    intersection_code = (
        ordered_gated[0].astype(int) * 4
        + ordered_gated[1].astype(int) * 2
        + ordered_gated[2].astype(int)
    )
    statistics["three_route_outcome_intersections"] = {
        format(index, "03b"): int((intersection_code == index).sum())
        for index in range(8)
    }
    statistics["bootstrap_contract"] = {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "primary_cluster": "scene_id",
        "sensitivity_cluster": "frame_id",
    }
    outcomes_wide = sample_manifest.copy()
    for name, selection in selection_by_name.items():
        safe_name = "".join(character if character.isalnum() else "_" for character in name)
        columns = selection[
            ["sample_id", "selected_route", "selected_candidate_id", "selected_correct"]
        ].rename(
            columns={
                "selected_route": f"{safe_name}__selected_route",
                "selected_candidate_id": f"{safe_name}__selected_candidate_id",
                "selected_correct": f"{safe_name}__selected_correct",
            }
        )
        outcomes_wide = outcomes_wide.merge(
            columns, on="sample_id", how="left", validate="one_to_one"
        )
    per_candidate_scores = (
        pd.concat(scored_candidate_parts, ignore_index=True)
        if scored_candidate_parts
        else pd.DataFrame(
            columns=[
                "system_name",
                "system_kind",
                "route",
                "sample_id",
                "candidate_id",
                "rank",
                "formal_score",
                "candidate_success",
            ]
        )
    )
    return metrics, statistics, per_sample, realized, outcomes_wide, per_candidate_scores


def run_formal_test_once(
    *,
    run_dir: Path,
    candidate_test_labels_path: Path | None = None,
) -> dict[str, Any]:
    """Claim, read the candidate-level labels once, evaluate, and finalize."""

    root = run_dir.resolve()
    _claim_before_label_access(root)
    try:
        assert_test_labels_unlocked(root)
        lock = verify_formal_test_lock(root)
        plan_path, plan = _locked_plan(lock)
        label_contract, candidate_pools = _locked_label_contract(lock, plan)
        declared_label_path = Path(str(label_contract["candidate_labels_path"])).resolve()
        if (
            candidate_test_labels_path is not None
            and candidate_test_labels_path.resolve() != declared_label_path
        ):
            raise PermissionError("CLI candidate-label path differs from the predeclared locked path")
        label_path = declared_label_path
        sample_path = Path(str(plan["sample_manifest"])).resolve()
        sample_manifest = _sample_manifest(sample_path)
        labels_raw, labels_sha256 = _read_candidate_labels_once(label_path)
        if labels_sha256 != label_contract["candidate_labels_sha256"]:
            raise PermissionError("candidate-level Test label hash differs from predeclared lock")
        append_access_log(
            root,
            {
                "event": "candidate_test_labels_read_once",
                "path": str(label_path),
                "sha256": labels_sha256,
                "row_count": int(len(labels_raw)),
            },
        )
        labels = _validate_labels(
            labels_raw,
            set(sample_manifest["sample_id"]),
            candidate_pools,
            label_contract,
        )
        (
            metrics,
            statistics,
            per_sample,
            realized_rankings,
            outcomes_wide,
            per_candidate_scores,
        ) = (
            _system_metrics_and_statistics(plan, sample_manifest, labels)
        )
        # The secondary bridge is intentionally evaluated only after every
        # primary system has been frozen and evaluated inside this claim.
        bridge_per_candidate_scores, bridge_sources = _evaluate_locked_test_bridge(
            root=root,
            lock=lock,
            plan=plan,
            sample_manifest=sample_manifest,
        )
        output = root / "09_formal_test"
        metrics_path = output / "formal_test_metrics.json"
        statistics_path = output / "formal_test_statistics.json"
        per_sample_path = output / "formal_test_per_sample.parquet"
        rankings_path = output / "formal_test_realized_rankings.parquet"
        outcomes_path = output / "formal_test_outcomes_wide.parquet"
        candidate_scores_path = output / "per_candidate_scores.parquet"
        bridge_candidate_scores_path = output / "bridge_per_candidate_scores.parquet"
        decisions_alias_path = output / "per_sample_decisions.parquet"
        manifest_path = output / "formal_test_manifest.json"
        atomic_json(metrics_path, {"systems": metrics})
        atomic_json(statistics_path, statistics)
        _atomic_parquet(per_sample_path, per_sample)
        _atomic_parquet(rankings_path, realized_rankings)
        _atomic_parquet(outcomes_path, outcomes_wide)
        _atomic_parquet(candidate_scores_path, per_candidate_scores)
        _atomic_parquet(bridge_candidate_scores_path, bridge_per_candidate_scores)
        _atomic_parquet(decisions_alias_path, per_sample)
        artifacts = {
            "metrics": {"path": str(metrics_path.resolve()), "sha256": sha256_file(metrics_path)},
            "statistics": {"path": str(statistics_path.resolve()), "sha256": sha256_file(statistics_path)},
            "per_sample": {"path": str(per_sample_path.resolve()), "sha256": sha256_file(per_sample_path)},
            "realized_rankings": {"path": str(rankings_path.resolve()), "sha256": sha256_file(rankings_path)},
            "outcomes_wide": {"path": str(outcomes_path.resolve()), "sha256": sha256_file(outcomes_path)},
            "per_candidate_scores": {
                "path": str(candidate_scores_path.resolve()),
                "sha256": sha256_file(candidate_scores_path),
            },
            "bridge_per_candidate_scores": {
                "path": str(bridge_candidate_scores_path.resolve()),
                "sha256": sha256_file(bridge_candidate_scores_path),
            },
            "per_sample_decisions": {
                "path": str(decisions_alias_path.resolve()),
                "sha256": sha256_file(decisions_alias_path),
            },
        }
        manifest: dict[str, Any] = {
            "status": "COMPLETE",
            "schema_version": 1,
            "benchmark_description": "locked retrospective paired test benchmark",
            "formal_test_execution_count": 1,
            "formal_lock": {
                "path": str((root / "08_lock" / "FORMAL_TEST_LOCK.json").resolve()),
                "file_sha256": sha256_file(root / "08_lock" / "FORMAL_TEST_LOCK.json"),
                "self_sha256": lock["self_sha256"],
            },
            "evaluation_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "candidate_test_labels": {
                "path": str(label_path),
                "sha256": labels_sha256,
                "opened_once_after_claim": True,
                "row_count": int(len(labels)),
                "provenance": label_contract["provenance"],
                "evaluator_sha256": label_contract["evaluator_sha256"],
            },
            "sample_manifest": {
                "path": str(sample_path),
                "sha256": sha256_file(sample_path),
                "sample_count": int(len(sample_manifest)),
            },
            "test_bridge": {
                **bridge_sources,
                "analysis_role": "SECONDARY_POSTLOCK_NO_SELECTION",
                "primary_systems_evaluated_before_ground_truth_open": True,
                "historical_candidate_labels_recomputed": True,
                "historical_candidate_labels_reused": False,
                "angles_negated": False,
            },
            "artifacts": artifacts,
            "independent_recompute_contract": {
                "join_keys": ["route", "sample_id", "candidate_id"],
                "union_label_join_keys": [
                    "source_route",
                    "sample_id",
                    "source_candidate_id",
                ],
                "union_candidate_identity": "<SOURCE_ROUTE>:<source_candidate_id>",
                "denominator_source": "sample_manifest",
                "realized_rankings_persisted": True,
                "wide_native_ungated_gated_router_union_outcomes_persisted": True,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
        }
        manifest["content_sha256"] = canonical_sha256(manifest)
        atomic_json(manifest_path, manifest)
        append_access_log(
            root,
            {
                "event": "bridge_per_candidate_scores_persisted",
                "path": str(bridge_candidate_scores_path.resolve()),
                "sha256": sha256_file(bridge_candidate_scores_path),
                "candidate_rows": int(len(bridge_per_candidate_scores)),
                "feeds_selection": False,
            },
        )
        finalize_formal_test_execution(
            root,
            {
                "formal_test_manifest": manifest_path,
                "formal_test_metrics": metrics_path,
                "formal_test_statistics": statistics_path,
                "formal_test_per_sample": per_sample_path,
                "formal_test_realized_rankings": rankings_path,
                "formal_test_outcomes_wide": outcomes_path,
                "per_candidate_scores": candidate_scores_path,
                "bridge_per_candidate_scores": bridge_candidate_scores_path,
                "per_sample_decisions": decisions_alias_path,
            },
        )
        append_access_log(
            root,
            {
                "event": "formal_test_execution_finalized",
                "execution_count": 1,
                "manifest_sha256": sha256_file(manifest_path),
            },
        )
        return manifest
    except Exception as error:
        append_access_log(
            root,
            {
                "event": "formal_test_execution_failed_after_claim",
                "error": f"{type(error).__name__}: {error}",
                "execution_consumed": True,
            },
        )
        raise


def execute_formal_test_once(
    *,
    run_dir: Path,
    candidate_test_labels_path: Path | None = None,
    command: str = "",
) -> dict[str, Any]:
    execution = run_dir.resolve() / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    if execution.exists():
        raise PermissionError("formal Test has already been claimed/executed")
    with ledger_stage(
        run_dir.resolve() / "run_ledger.sqlite",
        stage="P12",
        substage="exactly_once_formal_test",
        method="prelocked_paired_evaluation",
        command=command,
    ) as state:
        result = run_formal_test_once(
            run_dir=run_dir, candidate_test_labels_path=candidate_test_labels_path
        )
        path = run_dir.resolve() / "09_formal_test" / "formal_test_manifest.json"
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--candidate-test-labels",
        type=Path,
        help="optional assertion; must equal the predeclared locked label path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute_formal_test_once(
        run_dir=args.run_dir.resolve(),
        candidate_test_labels_path=(
            None
            if args.candidate_test_labels is None
            else args.candidate_test_labels.resolve()
        ),
        command=" ".join(map(str, sys.argv)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["execute_formal_test_once", "run_formal_test_once"]
