"""Independently recompute every locked formal-Test result from persisted IDs.

This module intentionally imports no project evaluator, model, ranker, gate,
router, statistics, or training code.  It trusts only files transitively bound
by ``FORMAL_TEST_LOCK.json`` and the hash-bound formal transaction outputs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROUTES = ("crog", "g1", "c1")
FORMAL_ARTIFACTS = {
    "metrics": "formal_test_metrics",
    "statistics": "formal_test_statistics",
    "per_sample": "formal_test_per_sample",
    "realized_rankings": "formal_test_realized_rankings",
    "outcomes_wide": "formal_test_outcomes_wide",
    "per_candidate_scores": "per_candidate_scores",
    "bridge_per_candidate_scores": "bridge_per_candidate_scores",
    "per_sample_decisions": "per_sample_decisions",
}
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260808
BRIDGE_COLUMNS = (
    "system_name",
    "system_kind",
    "route",
    "candidate_pool_contract",
    "sample_id",
    "candidate_id",
    "source_candidate_id",
    "raw_candidate_id",
    "native_rank",
    "frozen_native_rank",
    "rank",
    "formal_score",
    "native_score",
    "p_center",
    "raw_network_quality",
    "original_score",
    "fair_native_selector_score",
    "historical_selector_score",
    "cx_px",
    "cy_px",
    "theta_deg",
    "width_px",
    "height_px",
    "candidate_geometry_sha256",
    "source_candidate_identity_sha256",
    "evaluator_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _append_access_log(root: Path, event: Mapping[str, Any]) -> None:
    destination = root / "09_formal_test" / "test_access.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **dict(event),
    }
    with destination.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _verified(record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping):
        raise PermissionError(f"independent recompute has no bound record: {label}")
    path = Path(str(record.get("path", ""))).resolve()
    if path.is_symlink() or not path.is_file() or _sha256(path) != record.get("sha256"):
        raise PermissionError(f"independent recompute source drift: {label}")
    return path


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"independent recompute input is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    normalized_left = {
        "path": str(Path(str(left.get("path", ""))).resolve()),
        "sha256": left.get("sha256"),
    }
    normalized_right = {
        "path": str(Path(str(right.get("path", ""))).resolve()),
        "sha256": right.get("sha256"),
    }
    if normalized_left != normalized_right:
        raise PermissionError(f"independent recompute binding mismatch: {label}")


def _formal_lock(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "08_lock" / "FORMAL_TEST_LOCK.json"
    payload = _read_json(path)
    if payload.get("status") != "LOCKED" or payload.get("schema_version") != 1:
        raise PermissionError("formal Test lock is not LOCKED schema_version=1")
    unsigned = dict(payload)
    recorded = unsigned.pop("self_sha256", None)
    if not isinstance(recorded, str) or recorded != _canonical_sha256(unsigned):
        raise PermissionError("formal Test lock self-hash mismatch")
    records = payload.get("locked_files")
    if not isinstance(records, dict):
        raise PermissionError("formal Test lock has no locked-file inventory")
    for name, record in records.items():
        _verified(record, f"lock/{name}")
    return path, payload


def _exact_mcnemar(recovered: int, harmful: int) -> float:
    discordant = int(recovered) + int(harmful)
    if discordant == 0:
        return 1.0
    tail = min(int(recovered), int(harmful))
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _holm(pvalues: Sequence[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: (pvalues[index], index))
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    count = len(pvalues)
    for position, index in enumerate(order):
        running = max(running, (count - position) * float(pvalues[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _pool(path: Path, route: str, sample_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"locked {route} All pool misses columns: {missing}")
    output = frame[list(required)].copy()
    for column in ("sample_id", "candidate_id", "candidate_geometry_sha256"):
        output[column] = output[column].astype(str)
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
        raise ValueError(f"locked {route} All pool violates identity/rank contracts")
    output["native_rank"] = ranks.astype(int)
    for sample_id, group in output.groupby("sample_id", sort=False):
        if sorted(group["native_rank"].tolist()) != list(range(1, len(group) + 1)):
            raise ValueError(f"locked {route}/{sample_id} All ranks are not contiguous")
    return output


def _labels(
    frame: pd.DataFrame,
    sample_ids: set[str],
    label_manifest: Mapping[str, Any],
    pools: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    normalization = dict(label_manifest.get("normalization", {}))
    route_column = str(normalization.get("route_column", "route"))
    variant_column = str(normalization.get("variant_column", "variant"))
    include_variants = normalization.get("include_variants")
    if include_variants is not None:
        if variant_column not in frame.columns:
            raise ValueError("independent labels miss declared variant column")
        frame = frame.loc[
            frame[variant_column].astype(str).isin(map(str, include_variants))
        ].copy()
    required = {route_column, "sample_id", "candidate_id", "candidate_success"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"independent labels miss columns: {missing}")
    output = frame[list(required)].rename(columns={route_column: "route"}).copy()
    output["route"] = output["route"].astype(str).str.lower()
    output["sample_id"] = output["sample_id"].astype(str)
    output["candidate_id"] = output["candidate_id"].astype(str)
    numeric = pd.to_numeric(output["candidate_success"], errors="coerce")
    if (
        numeric.isna().any()
        or not numeric.isin([0, 1]).all()
        or output.duplicated(["route", "sample_id", "candidate_id"]).any()
        or not set(output["route"]).issubset(ROUTES)
        or not set(output["sample_id"]).issubset(sample_ids)
    ):
        raise ValueError("independent labels violate binary/identity contracts")
    output["candidate_success"] = numeric.astype(bool)
    observed = set(
        map(tuple, output[["route", "sample_id", "candidate_id"]].to_numpy())
    )
    expected = {
        (route, str(row.sample_id), str(row.candidate_id))
        for route, pool in pools.items()
        for row in pool[["sample_id", "candidate_id"]].itertuples(index=False)
    }
    if observed != expected:
        raise ValueError(
            "independent normalized labels do not exactly cover locked All pools: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    return output


def _pool_geometry(pools: Mapping[str, pd.DataFrame]) -> dict[tuple[str, str, str], str]:
    return {
        (route, str(row.sample_id), str(row.candidate_id)): str(
            row.candidate_geometry_sha256
        )
        for route, pool in pools.items()
        for row in pool[
            ["sample_id", "candidate_id", "candidate_geometry_sha256"]
        ].itertuples(index=False)
    }


def _selection_outcomes(
    selections: pd.DataFrame,
    labels: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    system: Mapping[str, Any],
    geometry: Mapping[tuple[str, str, str], str],
) -> pd.DataFrame:
    name = str(system["name"])
    base_required = {
        "sample_id",
        "selected_route",
        "selected_candidate_id",
        "selected_correct",
    }
    union_required = {"source_route", "source_candidate_id", "candidate_geometry_sha256"}
    required = base_required | (union_required if system["kind"] == "union" else set())
    missing = sorted(required.difference(selections.columns))
    if missing:
        raise ValueError(f"{name} saved selections miss columns: {missing}")
    selected = selections[list(required)].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected["selected_route"] = selected["selected_route"].astype(str).str.lower()
    selected["selected_candidate_id"] = selected["selected_candidate_id"].fillna("").astype(str)
    reported_correct = pd.to_numeric(selected["selected_correct"], errors="coerce")
    if (
        selected["sample_id"].duplicated().any()
        or set(selected["sample_id"]) != set(sample_manifest["sample_id"])
        or not set(selected["selected_route"]).issubset(ROUTES)
        or reported_correct.isna().any()
        or not reported_correct.isin([0, 1]).all()
    ):
        raise ValueError(f"{name} saved selections violate denominator/outcome identities")
    selected["reported_correct"] = reported_correct.astype(bool)
    selected = selected.drop(columns="selected_correct")
    if system["kind"] == "union":
        selected["source_route"] = selected["source_route"].astype(str).str.lower()
        selected["source_candidate_id"] = selected["source_candidate_id"].astype(str)
        selected["candidate_geometry_sha256"] = selected[
            "candidate_geometry_sha256"
        ].astype(str)
        qualified = (
            selected["source_route"].str.upper()
            + ":"
            + selected["source_candidate_id"]
        )
        if (
            not selected["selected_candidate_id"].equals(qualified)
            or not selected["selected_route"].equals(selected["source_route"])
        ):
            raise ValueError(f"{name} union selection qualifier/source-route mismatch")
        selected["label_candidate_id"] = selected["source_candidate_id"]
    else:
        selected["label_candidate_id"] = selected["selected_candidate_id"]
    output = sample_manifest.merge(selected, on="sample_id", how="left", validate="one_to_one")
    if output[["selected_route", "selected_candidate_id"]].isna().any().any():
        raise ValueError(f"{name} does not preserve the full denominator")
    nonempty = output["selected_candidate_id"].ne("")
    for row in output.loc[nonempty].itertuples(index=False):
        key = (str(row.selected_route), str(row.sample_id), str(row.label_candidate_id))
        if key not in geometry:
            raise ValueError(f"{name} selects outside locked candidate pools")
        if system["kind"] == "union" and str(row.candidate_geometry_sha256) != geometry[key]:
            raise ValueError(f"{name} union selection geometry mismatch")
    output = output.merge(
        labels.rename(
            columns={
                "route": "selected_route",
                "candidate_id": "label_candidate_id",
                "candidate_success": "independent_correct",
            }
        ),
        on=["selected_route", "sample_id", "label_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if output.loc[nonempty, "independent_correct"].isna().any():
        raise ValueError(f"{name} selects a candidate outside independent labels")
    output["independent_correct"] = output["independent_correct"].fillna(False).astype(bool)
    if not output["reported_correct"].equals(output["independent_correct"]):
        raise AssertionError(f"{name} persisted selected_correct differs from labels")
    output["system_name"] = name
    return output


def _bound_ranking(
    ranking: pd.DataFrame,
    system: Mapping[str, Any],
    pools: Mapping[str, pd.DataFrame],
    sample_ids: set[str],
) -> tuple[pd.DataFrame, int]:
    name = str(system["name"])
    required = {
        "sample_id",
        "candidate_id",
        "rank",
        "candidate_geometry_sha256",
        "frozen_native_rank",
    }
    if system["kind"] == "union":
        required.update({"source_route", "source_candidate_id"})
    missing = sorted(required.difference(ranking.columns))
    if missing:
        raise ValueError(f"{name} independent realized ranking misses columns: {missing}")
    work = ranking[list(required)].copy()
    for column in ("sample_id", "candidate_id", "candidate_geometry_sha256"):
        work[column] = work[column].astype(str)
    for column in ("rank", "frozen_native_rank"):
        values = pd.to_numeric(work[column], errors="coerce")
        if values.isna().any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name} realized ranking has invalid {column}")
        work[column] = values.astype(int)
    maximum = 15 if system["kind"] == "union" else 5
    if (
        not set(work["sample_id"]).issubset(sample_ids)
        or work["rank"].lt(1).any()
        or work["rank"].gt(maximum).any()
        or work.duplicated(["sample_id", "candidate_id"]).any()
        or work.duplicated(["sample_id", "rank"]).any()
    ):
        raise ValueError(f"{name} realized ranking has invalid Top-{maximum} ranks/identities")
    expected_parts: list[pd.DataFrame] = []
    if system["kind"] == "union":
        work["source_route"] = work["source_route"].astype(str).str.lower()
        work["source_candidate_id"] = work["source_candidate_id"].astype(str)
        qualified = work["source_route"].str.upper() + ":" + work["source_candidate_id"]
        if not set(work["source_route"]).issubset(ROUTES) or not work[
            "candidate_id"
        ].equals(qualified):
            raise ValueError(f"{name} union ranking qualifier/source identity mismatch")
        for route, pool in pools.items():
            part = pool.loc[
                pool["native_rank"].le(5),
                ["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"],
            ].copy()
            part["source_route"] = route
            part["source_candidate_id"] = part["candidate_id"].astype(str)
            part["candidate_id"] = route.upper() + ":" + part["source_candidate_id"]
            expected_parts.append(part)
        expected = pd.concat(expected_parts, ignore_index=True).rename(
            columns={"native_rank": "locked_native_rank"}
        )
        binding_columns = ["source_route", "source_candidate_id"]
    else:
        route = str(system["route"]).lower()
        expected = pools[route].loc[
            pools[route]["native_rank"].le(5),
            ["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"],
        ].rename(columns={"native_rank": "locked_native_rank"})
        binding_columns = []
    keys = ["sample_id", "candidate_id"]
    if set(map(tuple, work[keys].to_numpy())) != set(map(tuple, expected[keys].to_numpy())):
        raise ValueError(f"{name} realized ranking membership differs from locked pool")
    checked = work.merge(
        expected[keys + binding_columns + ["candidate_geometry_sha256", "locked_native_rank"]],
        on=keys,
        validate="one_to_one",
        suffixes=("", "_locked"),
    )
    for column in binding_columns + ["candidate_geometry_sha256"]:
        if not checked[column].equals(checked[f"{column}_locked"]):
                if column == "candidate_geometry_sha256":
                    raise ValueError(f"{name} realized ranking geometry binding mismatch")
                raise ValueError(f"{name} realized ranking {column} binding mismatch")
    if not checked["frozen_native_rank"].equals(checked["locked_native_rank"]):
        raise ValueError(f"{name} realized ranking native-rank binding mismatch")
    expected_counts = expected.groupby("sample_id", sort=False).size().to_dict()
    for sample_id, group in work.groupby("sample_id", sort=False):
        if sorted(group["rank"].tolist()) != list(
            range(1, int(expected_counts[str(sample_id)]) + 1)
        ):
            raise ValueError(f"{name} realized ranks are not contiguous")
    if system["kind"] == "native" and not work["rank"].equals(
        work["frozen_native_rank"]
    ):
        raise ValueError(f"{name} native ranking differs from frozen native ranks")
    return work, maximum


def _ranking_metrics(
    ranking: pd.DataFrame,
    system: Mapping[str, Any],
    labels: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    maximum: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if system["kind"] == "union":
        joined = ranking.merge(
            labels.rename(
                columns={"route": "source_route", "candidate_id": "source_candidate_id"}
            )[["source_route", "sample_id", "source_candidate_id", "candidate_success"]],
            on=["source_route", "sample_id", "source_candidate_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        route = str(system["route"]).lower()
        joined = ranking.merge(
            labels.loc[
                labels["route"].eq(route),
                ["sample_id", "candidate_id", "candidate_success"],
            ],
            on=["sample_id", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    if joined["candidate_success"].isna().any():
        raise ValueError(f"{system['name']} ranking references an unlabeled candidate")
    joined["candidate_success"] = joined["candidate_success"].astype(bool)
    per_sample = sample_manifest[["sample_id"]].copy()
    first = joined.loc[joined["candidate_success"]].groupby("sample_id")["rank"].min()
    per_sample["first_positive_rank"] = per_sample["sample_id"].map(first)
    count = len(per_sample)
    metrics: dict[str, Any] = {"sample_count": count}
    for k in range(1, maximum + 1):
        value = per_sample["first_positive_rank"].le(k).fillna(False)
        per_sample[f"j_at_{k}"] = value
        metrics[f"j_at_{k}_numerator"] = int(value.sum())
        metrics[f"j_at_{k}"] = float(value.mean())
    reciprocal = (1.0 / per_sample["first_positive_rank"].astype(float)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    per_sample["reciprocal_rank"] = reciprocal
    ndcg_by_sample: dict[str, float] = {}
    for sample_id, group in joined.groupby("sample_id", sort=False):
        gains = (
            group.sort_values("rank", kind="mergesort")["candidate_success"]
            .to_numpy(bool)
            .astype(float)
        )
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2, dtype=float))
        relevant = int(gains.sum())
        idcg = float(
            (1.0 / np.log2(np.arange(2, min(relevant, maximum) + 2, dtype=float))).sum()
        )
        ndcg_by_sample[str(sample_id)] = (
            0.0 if idcg == 0.0 else float((gains * discounts).sum()) / idcg
        )
    ndcg = per_sample["sample_id"].map(ndcg_by_sample).fillna(0.0)
    oracle_key = "oracle_at_15" if maximum == 15 else "oracle_at_5"
    mrr_key = "mrr_at_15" if maximum == 15 else "mrr_at_5"
    metrics[oracle_key] = metrics[f"j_at_{maximum}"]
    metrics[mrr_key] = float(reciprocal.mean())
    metrics["ndcg_at_1"] = metrics["j_at_1"]
    metrics[f"ndcg_at_{maximum}"] = float(ndcg.mean())
    route_labels = labels if system["kind"] == "union" else labels.loc[
        labels["route"].eq(str(system["route"]).lower())
    ]
    all_positive = route_labels.groupby("sample_id")["candidate_success"].any()
    oracle_all = sample_manifest["sample_id"].map(all_positive).fillna(False)
    metrics["oracle_all_numerator"] = int(oracle_all.sum())
    metrics["oracle_all"] = float(oracle_all.mean())
    return metrics, per_sample


def _unranked_metrics(outcome: pd.DataFrame) -> dict[str, Any]:
    numerator = int(outcome["independent_correct"].sum())
    count = len(outcome)
    return {
        "sample_count": count,
        "j_at_1_numerator": numerator,
        "j_at_1": numerator / count,
        "j_at_2": None,
        "j_at_3": None,
        "j_at_4": None,
        "j_at_5": None,
        "oracle_at_5": None,
        "mrr_at_5": None,
        "ndcg_at_1": numerator / count,
        "ndcg_at_5": None,
    }


def _assert_equal(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise AssertionError(f"independent inventory mismatch for {label}")
        for key in expected:
            _assert_equal(observed[key], expected[key], f"{label}/{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, (list, tuple)) or len(observed) != len(expected):
            raise AssertionError(f"independent sequence mismatch for {label}")
        for index, value in enumerate(expected):
            _assert_equal(observed[index], value, f"{label}/{index}")
        return
    if expected is None or isinstance(expected, (str, bool, np.bool_)):
        if observed != expected:
            raise AssertionError(f"independent mismatch for {label}: {observed} != {expected}")
        return
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, (bool, np.bool_)):
        if isinstance(observed, bool) or int(observed) != int(expected):
            raise AssertionError(f"independent mismatch for {label}: {observed} != {expected}")
        return
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-14):
        raise AssertionError(f"independent mismatch for {label}: {observed} != {expected}")


def _selection_comparison(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    oracle: float,
    *,
    union: bool,
) -> dict[str, Any]:
    before = reference["independent_correct"].to_numpy(bool)
    after = challenger["independent_correct"].to_numpy(bool)
    recovered = int((~before & after).sum())
    harmful = int((before & ~after).sum())
    changed = _canonical_selection_keys(reference).to_numpy() != _canonical_selection_keys(
        challenger
    ).to_numpy()
    native_j = float(before.mean())
    challenger_j = float(after.mean())
    headroom = float(oracle) - native_j
    changing = recovered + harmful
    key = "headroom_recovery_at_15" if union else "headroom_recovery_at_5"
    return {
        "sample_count": len(reference),
        "native_j_at_1": native_j,
        "challenger_j_at_1": challenger_j,
        "delta_j_at_1": challenger_j - native_j,
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "switch_count": int(changed.sum()),
        "switch_rate": float(changed.mean()),
        "outcome_changing_precision": None if changing == 0 else recovered / changing,
        key: None if headroom <= 0 else (challenger_j - native_j) / headroom,
    }


def _canonical_selection_keys(frame: pd.DataFrame) -> pd.Series:
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


def _mcnemar(reference: pd.DataFrame, challenger: pd.DataFrame) -> dict[str, Any]:
    before = reference["independent_correct"].to_numpy(bool)
    after = challenger["independent_correct"].to_numpy(bool)
    recovered = int((~before & after).sum())
    harmful = int((before & ~after).sum())
    pvalue = _exact_mcnemar(recovered, harmful)
    return {
        "sample_count": len(reference),
        "both_wrong": int((~before & ~after).sum()),
        "recovered": recovered,
        "harmful": harmful,
        "both_correct": int((before & after).sum()),
        "discordant": recovered + harmful,
        "net_recovered": recovered - harmful,
        "effect": float(after.mean() - before.mean()),
        "pvalue": pvalue,
        "raw_p": pvalue,
        "method": "exact two-sided binomial McNemar",
    }


def _cluster_bootstrap(
    reference: pd.DataFrame,
    challenger: pd.DataFrame,
    clusters: Sequence[Any],
) -> dict[str, Any]:
    before = reference["independent_correct"].to_numpy(bool)
    after = challenger["independent_correct"].to_numpy(bool)
    raw_clusters = np.asarray(clusters, dtype=object)
    cluster_values, inverse = np.unique(raw_clusters.astype(str), return_inverse=True)
    difference = after.astype(float) - before.astype(float)
    cluster_sums = np.bincount(inverse, weights=difference, minlength=len(cluster_values))
    cluster_counts = np.bincount(inverse, minlength=len(cluster_values))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for start in range(0, BOOTSTRAP_ITERATIONS, 1_000):
        stop = min(start + 1_000, BOOTSTRAP_ITERATIONS)
        draws = rng.integers(
            0, len(cluster_values), size=(stop - start, len(cluster_values))
        )
        bootstrap[start:stop] = (
            cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
        )
    ci = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    return {
        "point_estimate": float(difference.mean()),
        "ci": ci,
        "ci95": ci,
        "confidence": 0.95,
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "sample_count": len(reference),
        "cluster_count": len(cluster_values),
        "resampling_unit": "cluster",
    }


def _three_route_statistics(
    outcomes: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    names = tuple(outcomes)
    matrix = np.column_stack(
        [outcomes[name]["independent_correct"].to_numpy(bool).astype(int) for name in names]
    )
    identical = bool(np.all(matrix == matrix[:, [0]]))
    column_sums = matrix.sum(axis=0).astype(float)
    row_sums = matrix.sum(axis=1).astype(float)
    denominator = len(names) * float(column_sums.sum()) - float((row_sums**2).sum())
    statistic = (
        0.0
        if identical or denominator == 0.0
        else (len(names) - 1)
        * (len(names) * float((column_sums**2).sum()) - float(column_sums.sum() ** 2))
        / denominator
    )
    pairs: list[dict[str, Any]] = []
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            first, second = names[first_index], names[second_index]
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    **_mcnemar(outcomes[first], outcomes[second]),
                }
            )
    adjusted = _holm([float(row["pvalue"]) for row in pairs])
    for row, value in zip(pairs, adjusted):
        row["holm_adjusted_pvalue"] = value
    return {
        "cochran_q": {
            "statistic": statistic,
            "pvalue": 1.0 if identical else math.exp(-statistic / 2.0),
            "df": 2,
            "degenerate_identical_systems": identical,
            "method_note": "conventional/supportive because sample rows are scene-clustered",
        },
        "pairwise_mcnemar_holm": pairs,
    }


def _bridge_geometry_sha256(row: Any) -> str:
    geometry = [
        float(row.cx_px),
        float(row.cy_px),
        float(row.theta_deg),
        float(row.width_px),
        float(row.height_px),
    ]
    if str(row.candidate_pool_contract) == "fair_gaussian":
        value = [
            str(row.route).upper(),
            str(row.sample_id),
            str(row.source_candidate_id),
            int(row.native_rank),
            *geometry,
        ]
    else:
        value = [
            str(row.route).upper(),
            "historical_nms",
            str(row.sample_id),
            str(row.source_candidate_id),
            int(row.native_rank),
            *geometry,
        ]
    return _canonical_sha256(value)


def _independent_bridge_successes(
    bundle: pd.DataFrame,
    ground_truth: pd.DataFrame,
    evaluator_path: Path,
    evaluator_sha256: str,
) -> np.ndarray:
    if _sha256(evaluator_path) != evaluator_sha256:
        raise PermissionError("independent bridge evaluator hash mismatch")
    module_name = f"_independent_bridge_evaluator_{evaluator_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load locked bridge evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    labels = ground_truth.set_index("sample_id")["gt_grasp_rectangles"].to_dict()
    successes: list[bool] = []
    for row in bundle.itertuples(index=False):
        candidate = module.CanonicalGrasp(
            cx_px=float(row.cx_px),
            cy_px=float(row.cy_px),
            theta_deg=float(row.theta_deg),
            jaw_width_px=float(row.width_px),
            rectangle_height_px=float(row.height_px),
            native_score=float(row.native_score),
            native_rank=int(row.native_rank),
            source_method=str(row.route),
            sample_id=str(row.sample_id),
        )
        rectangles = (
            np.stack([np.asarray(point, dtype=np.float64) for point in rectangle])
            for rectangle in labels[str(row.sample_id)]
        )
        converted = tuple(module.gt_from_corners(rectangle) for rectangle in rectangles)
        successes.append(bool(module.evaluate_candidate(candidate, converted)["success"]))
    return np.asarray(successes, dtype=bool)


def _bridge_pool_checks(bundle: pd.DataFrame, denominator: list[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for route in ("g1", "c1"):
        for pool in ("fair_gaussian", "historical_nms"):
            part = bundle.loc[
                bundle["route"].eq(route)
                & bundle["candidate_pool_contract"].eq(pool)
            ]
            bearing = set(part["sample_id"].astype(str))
            checks[f"{route}/{pool}"] = {
                "candidate_rows": int(len(part)),
                "candidate_bearing_samples": int(len(bearing)),
                "no_output_samples": int(len(denominator) - len(bearing)),
                "denominator_sample_count": int(len(denominator)),
                "qualified_identity_sha256": _canonical_sha256(
                    part[
                        [
                            "sample_id",
                            "candidate_id",
                            "source_candidate_id",
                            "native_rank",
                            "candidate_geometry_sha256",
                            "fair_native_selector_score",
                            "historical_selector_score",
                        ]
                    ].values.tolist()
                ),
            }
    return checks


def _verify_bridge_authority(manifest: Mapping[str, Any]) -> None:
    sources = manifest.get("sources", {})
    required = {
        "historical_run_sha_manifest",
        "historical_run_lock_sha256",
        "historical_formal_lock",
        "historical_pool_inventory",
        "historical_ground_truth",
        "historical_source_manifest",
        "historical_experiment_lock_marker",
        "historical_finalization",
        "g1_historical_candidates",
        "c1_historical_candidates",
        "g1_historical_allnms",
        "c1_historical_allnms",
    }
    if not isinstance(sources, Mapping) or not required.issubset(sources):
        raise PermissionError("independent Test bridge authority inventory is incomplete")
    run_sha_path = _verified(sources["historical_run_sha_manifest"], "historical run SHA manifest")
    run_lock_path = _verified(sources["historical_run_lock_sha256"], "historical run SHA lock")
    if run_lock_path.read_text(encoding="utf-8").strip() != _sha256(run_sha_path):
        raise PermissionError("independent historical run SHA lock mismatch")
    records: dict[str, str] = {}
    previous: str | None = None
    for raw in run_sha_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        if match is None:
            raise PermissionError("independent historical run SHA manifest is malformed")
        relative = match.group(2)
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or relative in records
            or (previous is not None and relative <= previous)
        ):
            raise PermissionError("independent historical run SHA paths are unsafe/unsorted")
        records[relative] = match.group(1)
        previous = relative
    run_root = run_sha_path.parent.resolve()
    for name in (
        "historical_formal_lock",
        "historical_pool_inventory",
        "g1_historical_candidates",
        "c1_historical_candidates",
        "g1_historical_allnms",
        "c1_historical_allnms",
    ):
        path = Path(str(sources[name]["path"])).resolve()
        try:
            relative = path.relative_to(run_root).as_posix()
        except ValueError as error:
            raise PermissionError("independent historical authority path escaped run") from error
        if records.get(relative) != sources[name].get("sha256"):
            raise PermissionError(f"independent historical authority mismatch: {name}")

    formal_lock = _read_json(Path(str(sources["historical_formal_lock"]["path"])))
    modular_root = Path(str(sources["historical_source_manifest"]["path"])).parent.parent.resolve()
    if (
        formal_lock.get("status") != "LOCKED"
        or Path(str(formal_lock.get("base_run", ""))).resolve() != modular_root
    ):
        raise PermissionError("independent historical formal-lock authority mismatch")
    inventory_sha = sources["historical_pool_inventory"]["sha256"]
    if not any(
        record.get("sha256") == inventory_sha
        for record in formal_lock.get("audit_artifacts", [])
    ):
        raise PermissionError("independent historical formal lock omits pool inventory")
    ground_truth_sha = sources["historical_ground_truth"]["sha256"]
    if not any(
        record.get("sha256") == ground_truth_sha
        for record in formal_lock.get("source_label_artifacts", [])
    ):
        raise PermissionError("independent historical formal lock omits bridge GT")
    for route in ("g1", "c1"):
        digest = sources[f"{route}_historical_allnms"]["sha256"]
        if not any(
            record.get("sha256") == digest
            for record in formal_lock.get("candidate_artifacts", [])
        ):
            raise PermissionError(f"independent historical formal lock omits {route} AllNMS")

    experiment_lock = _read_json(Path(str(sources["historical_source_manifest"]["path"])))
    unsigned = dict(experiment_lock)
    content_sha256 = unsigned.pop("manifest_content_sha256", None)
    marker = _read_json(Path(str(sources["historical_experiment_lock_marker"]["path"])))
    finalization = _read_json(Path(str(sources["historical_finalization"]["path"])))
    if (
        experiment_lock.get("lock_status") != "LOCKED"
        or experiment_lock.get("effective") is not True
        or content_sha256 != _canonical_sha256(unsigned)
        or marker.get("lock_status") != "LOCKED"
        or marker.get("manifest_content_sha256") != content_sha256
        or finalization.get("status") != "COMPLETE"
        or finalization.get("experiment_lock_sha256") != content_sha256
    ):
        raise PermissionError("independent modular experiment-lock authority mismatch")


def _verify_formal_bridge(
    *,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    formal_scores_path: Path,
    sample_manifest: pd.DataFrame,
    access_log_root: Path,
) -> dict[str, Any]:
    locked = dict(lock.get("locked_files", {}))
    contract = plan.get("test_bridge_contract")
    if not isinstance(contract, Mapping):
        raise PermissionError("independent recompute has no locked Test bridge contract")
    manifest_record = locked.get("formal_test_bridge_manifest")
    _same_record(manifest_record, contract.get("manifest"), "Test bridge manifest")
    manifest_path = _verified(manifest_record, "Test bridge manifest")
    manifest = _read_json(manifest_path)
    unsigned = dict(manifest)
    recorded = unsigned.pop("content_sha256", None)
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("candidate_test_labels_read") is not False
        or manifest.get("historical_test_ground_truth_rows_read") is not False
        or recorded != _canonical_sha256(unsigned)
    ):
        raise PermissionError("independent Test bridge manifest integrity/state mismatch")
    bundle_record = manifest.get("artifacts", {}).get("candidate_bundle")
    _same_record(
        locked.get("formal_test_bridge_candidate_bundle"),
        bundle_record,
        "Test bridge candidate bundle",
    )
    _same_record(contract.get("candidate_bundle"), bundle_record, "plan Test bridge bundle")
    bundle_path = _verified(bundle_record, "Test bridge candidate bundle")
    for name, record in manifest.get("sources", {}).items():
        _same_record(
            locked.get(f"formal_test_bridge_source_{name}"),
            record,
            f"Test bridge source {name}",
        )
        _verified(record, f"Test bridge source {name}")
    _verify_bridge_authority(manifest)
    for name in ("historical_ground_truth", "evaluator", "denominator"):
        _same_record(contract.get(name), manifest["sources"][name], f"plan bridge {name}")
    bundle = pd.read_parquet(bundle_path)
    formal = pd.read_parquet(formal_scores_path)
    if set(bundle.columns) != set(BRIDGE_COLUMNS) or set(formal.columns) != {
        *BRIDGE_COLUMNS,
        "candidate_success",
    }:
        raise ValueError("independent Test bridge schema mismatch")
    bundle = bundle[list(BRIDGE_COLUMNS)].reset_index(drop=True)
    formal = formal[[*BRIDGE_COLUMNS, "candidate_success"]].reset_index(drop=True)
    if not formal[list(BRIDGE_COLUMNS)].equals(bundle):
        raise AssertionError("formal Test bridge membership/geometry differs from lock")
    if any(
        token in str(column).lower()
        for column in bundle.columns
        for token in (
            "candidate_success",
            "ground_truth",
            "gt_grasp",
            "matched_gt",
            "jacquard",
            "angle_error",
            "iou",
        )
    ):
        raise PermissionError("locked preclaim Test bridge contains historical labels")
    if set(bundle["route"].astype(str)) != {"g1", "c1"} or set(
        bundle["candidate_pool_contract"].astype(str)
    ) != {"fair_gaussian", "historical_nms"}:
        raise ValueError("independent Test bridge route/pool inventory mismatch")
    qualified = (
        bundle["route"].str.upper()
        + "::"
        + bundle["candidate_pool_contract"].astype(str)
        + "::"
        + bundle["source_candidate_id"].astype(str)
    )
    if (
        not bundle["candidate_id"].astype(str).equals(qualified)
        or not bundle["raw_candidate_id"].astype(str).equals(
            bundle["source_candidate_id"].astype(str)
        )
        or bundle.duplicated(
            ["route", "candidate_pool_contract", "sample_id", "candidate_id"]
        ).any()
    ):
        raise ValueError("independent Test bridge qualified identity mismatch")
    numeric_columns = [
        "native_rank",
        "fair_native_selector_score",
        "historical_selector_score",
        "native_score",
        "p_center",
        "raw_network_quality",
        "original_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ]
    numeric = bundle[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("independent Test bridge contains non-finite values")
    fair = bundle["candidate_pool_contract"].eq("fair_gaussian")
    historical = ~fair
    if (
        not np.array_equal(
            bundle.loc[fair, "fair_native_selector_score"].to_numpy(float),
            bundle.loc[fair, "native_score"].to_numpy(float),
        )
        or not np.allclose(
            bundle.loc[fair, "historical_selector_score"].to_numpy(float),
            bundle.loc[fair, "native_score"].to_numpy(float)
            * np.clip(bundle.loc[fair, "p_center"].to_numpy(float), 0.0, 1.0),
            rtol=0.0,
            atol=0.0,
        )
        or not np.array_equal(
            bundle.loc[historical, "fair_native_selector_score"].to_numpy(float),
            bundle.loc[historical, "raw_network_quality"].to_numpy(float),
        )
        or not np.allclose(
            bundle.loc[historical, "historical_selector_score"].to_numpy(float),
            bundle.loc[historical, "raw_network_quality"].to_numpy(float)
            * np.clip(
                bundle.loc[historical, "p_center"].to_numpy(float), 0.0, 1.0
            ),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise AssertionError("independent Test bridge selector formula mismatch")
    geometry = [_bridge_geometry_sha256(row) for row in bundle.itertuples(index=False)]
    if bundle["candidate_geometry_sha256"].astype(str).tolist() != geometry:
        raise AssertionError("independent Test bridge geometry hash mismatch")
    denominator = sample_manifest["sample_id"].astype(str).tolist()
    if not set(bundle["sample_id"].astype(str)).issubset(denominator):
        raise ValueError("independent Test bridge includes a non-denominator sample")
    if manifest.get("pool_checks") != _bridge_pool_checks(bundle, denominator):
        raise AssertionError("independent Test bridge exact coverage/no-output mismatch")
    ground_truth_path = _verified(
        manifest["sources"]["historical_ground_truth"], "Test bridge ground truth"
    )
    ground_truth = pd.read_parquet(ground_truth_path)
    _append_access_log(
        access_log_root,
        {
            "event": "postclaim_independent_bridge_ground_truth_read",
            "purpose": "P15 independent bridge label recomputation",
            "source_path": str(ground_truth_path.resolve()),
            "source_sha256": _sha256(ground_truth_path),
            "rows": int(len(ground_truth)),
            "formal_test_execution_count": 1,
        },
    )
    if (
        not {"sample_id", "gt_grasp_rectangles"}.issubset(ground_truth.columns)
        or ground_truth["sample_id"].astype(str).duplicated().any()
        or set(ground_truth["sample_id"].astype(str)) != set(denominator)
    ):
        raise ValueError("independent Test bridge GT denominator mismatch")
    ground_truth = ground_truth.copy()
    ground_truth["sample_id"] = ground_truth["sample_id"].astype(str)
    evaluator_record = manifest["sources"]["evaluator"]
    expected_success = _independent_bridge_successes(
        bundle,
        ground_truth,
        Path(str(evaluator_record["path"])),
        str(evaluator_record["sha256"]),
    )
    observed_success = pd.to_numeric(formal["candidate_success"], errors="coerce")
    if (
        observed_success.isna().any()
        or not observed_success.isin([0, 1]).all()
        or not np.array_equal(observed_success.to_numpy(bool), expected_success)
    ):
        raise AssertionError("independent Test bridge candidate labels mismatch")
    return {
        "candidate_rows": int(len(bundle)),
        "pool_cells": 4,
        "membership_geometry_verified": True,
        "selector_formulas_verified": True,
        "candidate_labels_recomputed": True,
        "no_output_denominators_verified": True,
    }


def run(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = run_dir.resolve()
    lock_path, lock = _formal_lock(root)
    locked = lock["locked_files"]
    execution_path = root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    execution = _read_json(execution_path)
    if execution.get("status") != "COMPLETE" or execution.get("execution_count") != 1:
        raise PermissionError("independent recompute requires one completed formal-Test transaction")
    if (
        execution.get("formal_lock_file_sha256") != _sha256(lock_path)
        or execution.get("formal_lock_self_sha256") != lock["self_sha256"]
    ):
        raise PermissionError("formal execution is not bound to the verified formal lock")
    for name, record in execution.get("artifacts", {}).items():
        _verified(record, f"execution/{name}")
    manifest_path = root / "09_formal_test" / "formal_test_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "COMPLETE" or manifest.get("formal_test_execution_count") != 1:
        raise PermissionError("formal-Test result manifest is incomplete")
    if (
        manifest.get("formal_lock", {}).get("file_sha256") != _sha256(lock_path)
        or manifest.get("formal_lock", {}).get("self_sha256") != lock["self_sha256"]
    ):
        raise PermissionError("formal result manifest is not bound to the verified lock")
    plan_record = locked.get("formal_evaluation_plan")
    label_manifest_record = locked.get("formal_candidate_label_manifest")
    sample_record = locked.get("formal_sample_manifest")
    plan_path = _verified(plan_record, "lock/evaluation_plan")
    label_manifest_path = _verified(label_manifest_record, "lock/candidate_label_manifest")
    sample_path = _verified(sample_record, "lock/sample_manifest")
    _same_record(manifest["evaluation_plan"], plan_record, "manifest/evaluation_plan")
    _same_record(manifest["sample_manifest"], sample_record, "manifest/sample_manifest")
    plan = _read_json(plan_path)
    if Path(str(plan.get("candidate_label_manifest", ""))).resolve() != label_manifest_path:
        raise PermissionError("formal plan candidate-label manifest differs from lock")
    label_manifest = _read_json(label_manifest_path)
    label_record = {
        "path": label_manifest.get("candidate_labels_path"),
        "sha256": label_manifest.get("candidate_labels_sha256"),
    }
    _same_record(manifest["candidate_test_labels"], label_record, "manifest/candidate_labels")
    label_path = _verified(label_record, "candidate_labels")
    sample_manifest = pd.read_parquet(sample_path)[
        ["sample_id", "scene_id", "frame_id"]
    ].copy()
    for column in sample_manifest:
        sample_manifest[column] = sample_manifest[column].astype(str)
    if sample_manifest.empty or sample_manifest["sample_id"].duplicated().any():
        raise ValueError("independent denominator is empty or duplicated")
    sample_ids = set(sample_manifest["sample_id"])
    pools: dict[str, pd.DataFrame] = {}
    for route in ROUTES:
        record = locked.get(f"formal_{route}_all_candidate_pool")
        declared = label_manifest.get("candidate_pools", {}).get(route)
        _same_record(record, declared, f"{route}/All pool")
        pools[route] = _pool(_verified(record, f"{route}/All pool"), route, sample_ids)
    raw_labels = pd.read_parquet(label_path)
    _append_access_log(
        root,
        {
            "event": "postclaim_independent_candidate_labels_read",
            "purpose": "P15 independent metric/statistic recomputation",
            "source_path": str(label_path.resolve()),
            "source_sha256": _sha256(label_path),
            "rows": int(len(raw_labels)),
            "formal_test_execution_count": 1,
        },
    )
    labels = _labels(raw_labels, sample_ids, label_manifest, pools)
    if set(manifest.get("artifacts", {})) != set(FORMAL_ARTIFACTS):
        raise PermissionError("formal artifact inventory is missing or unknown")
    artifact_paths: dict[str, Path] = {}
    for manifest_name, execution_name in FORMAL_ARTIFACTS.items():
        record = manifest["artifacts"][manifest_name]
        _same_record(record, execution["artifacts"][execution_name], f"formal/{manifest_name}")
        artifact_paths[manifest_name] = _verified(record, f"formal/{manifest_name}")
    bridge_checks = _verify_formal_bridge(
        lock=lock,
        plan=plan,
        formal_scores_path=artifact_paths["bridge_per_candidate_scores"],
        sample_manifest=sample_manifest,
        access_log_root=root,
    )
    metric_payload = _read_json(artifact_paths["metrics"])
    if not isinstance(metric_payload, dict) or set(metric_payload) != {"systems"}:
        raise AssertionError("independent top-level metric inventory mismatch")
    reported_metrics = metric_payload["systems"]
    reported_statistics = _read_json(artifact_paths["statistics"])
    formal_per_sample = pd.read_parquet(artifact_paths["per_sample"])
    alias_decisions = pd.read_parquet(artifact_paths["per_sample_decisions"])
    if not formal_per_sample.equals(alias_decisions):
        raise AssertionError("per_sample_decisions alias differs from formal per-sample artifact")
    realized = pd.read_parquet(artifact_paths["realized_rankings"])
    declared_systems = plan.get("systems")
    if not isinstance(declared_systems, list):
        raise ValueError("locked formal plan has no system list")
    systems = {str(system["name"]): dict(system) for system in declared_systems}
    if len(systems) != len(declared_systems):
        raise ValueError("locked formal plan contains duplicate system names")
    if set(reported_metrics) != set(systems):
        raise AssertionError("independent system inventory differs from formal metrics")
    if "system_name" not in formal_per_sample or set(
        formal_per_sample["system_name"].astype(str)
    ) != set(systems):
        raise AssertionError("independent system inventory differs from formal decisions")
    expected_ranked = {
        name for name, system in systems.items() if system["kind"] != "router"
    }
    if "system_name" not in realized or set(realized["system_name"].astype(str)) != expected_ranked:
        raise AssertionError("independent system inventory differs from formal rankings")
    geometry = _pool_geometry(pools)
    recomputed_metrics: dict[str, Any] = {}
    independent_outcomes: dict[str, pd.DataFrame] = {}
    per_sample_parts: list[pd.DataFrame] = []
    metric_checks = 0
    for name, system in systems.items():
        saved = formal_per_sample.loc[
            formal_per_sample["system_name"].astype(str).eq(name)
        ].copy()
        outcome = _selection_outcomes(saved, labels, sample_manifest, system, geometry)
        independent_outcomes[name] = outcome.sort_values("sample_id").reset_index(drop=True)
        per_sample_parts.append(outcome)
        ranking = realized.loc[realized["system_name"].astype(str).eq(name)].copy()
        if not ranking.empty:
            bound, maximum = _bound_ranking(ranking, system, pools, sample_ids)
            metrics, _ranked_sample = _ranking_metrics(
                bound, system, labels, sample_manifest, maximum
            )
            top = bound.loc[bound["rank"].eq(1)].set_index("sample_id")["candidate_id"].astype(str)
            selected = outcome.set_index("sample_id")["selected_candidate_id"].astype(str)
            if not top.reindex(selected.index, fill_value="").equals(selected):
                raise AssertionError(f"independent ranking Top-1 differs from {name} selection")
        else:
            if system["kind"] != "router":
                raise ValueError(f"non-router formal system has no realized ranking: {name}")
            metrics = _unranked_metrics(outcome)
        recomputed_metrics[name] = metrics
        if set(reported_metrics[name]) != set(metrics):
            raise AssertionError(f"independent metric inventory mismatch for {name}")
        _assert_equal(reported_metrics[name], metrics, f"metrics/{name}")
        metric_checks += len(metrics)
    expected_comparison_names = {
        name for name, system in systems.items() if system["kind"] != "native"
    }
    if set(reported_statistics) != {
        "comparisons",
        "hypothesis_families",
        "three_route_final_systems",
        "three_route_outcome_intersections",
        "bootstrap_contract",
    } or set(reported_statistics["comparisons"]) != expected_comparison_names:
        raise AssertionError("independent formal statistics inventory mismatch")
    comparisons: dict[str, Any] = {}
    family_members: dict[str, list[str]] = {}
    comparison_checks = 0
    sample_order = sample_manifest.sort_values("sample_id")
    for name, system in systems.items():
        if system["kind"] == "native":
            continue
        reference_name = str(system["native_reference"])
        reference = independent_outcomes[reference_name]
        challenger = independent_outcomes[name]
        union = system["kind"] == "union"
        oracle = (
            recomputed_metrics[name]["oracle_at_15"]
            if union
            else recomputed_metrics[reference_name].get("oracle_at_5")
        )
        if oracle is None:
            oracle = float(reference["independent_correct"].mean())
        mcnemar = _mcnemar(reference, challenger)
        record = {
            "reference": reference_name,
            "hypothesis_family": str(system["hypothesis_family"]),
            "selection_comparison": _selection_comparison(
                reference, challenger, float(oracle), union=union
            ),
            "primary_uncertainty": "scene_clustered_bootstrap",
            "scene_bootstrap": _cluster_bootstrap(
                reference, challenger, sample_order["scene_id"].to_numpy()
            ),
            "frame_bootstrap_sensitivity": _cluster_bootstrap(
                reference, challenger, sample_order["frame_id"].to_numpy()
            ),
            "mcnemar_conventional_supportive": mcnemar,
            "dependence_disclosure": (
                "Sample-level McNemar treats paired rows as independent; repeated language "
                "queries share visual scenes/frames, so it is supportive rather than the "
                "primary uncertainty analysis."
            ),
        }
        comparisons[name] = record
        family_members.setdefault(str(system["hypothesis_family"]), []).append(name)
    for family, names in sorted(family_members.items()):
        adjusted = _holm(
            [float(comparisons[name]["mcnemar_conventional_supportive"]["pvalue"]) for name in names]
        )
        for name, value in zip(names, adjusted):
            comparisons[name]["holm_adjusted_mcnemar_pvalue"] = value
    for name, record in comparisons.items():
        _assert_equal(reported_statistics["comparisons"][name], record, f"statistics/{name}")
        comparison_checks += 1
    expected_families = {
        family: {
            "members": names,
            "correction": "Holm-Bonferroni within predeclared family",
            "comparison_count": len(names),
        }
        for family, names in sorted(family_members.items())
    }
    _assert_equal(
        reported_statistics["hypothesis_families"], expected_families, "hypothesis_families"
    )
    gated = {
        route.upper(): independent_outcomes[
            next(
                name
                for name, system in systems.items()
                if system.get("route") == route and system["kind"] == "gated"
            )
        ]
        for route in ROUTES
    }
    three_route = _three_route_statistics(gated)
    _assert_equal(
        reported_statistics["three_route_final_systems"],
        three_route,
        "three_route_final_systems",
    )
    ordered = [gated[route]["independent_correct"].to_numpy(bool) for route in ("CROG", "G1", "C1")]
    code = ordered[0].astype(int) * 4 + ordered[1].astype(int) * 2 + ordered[2].astype(int)
    intersections = {
        format(index, "03b"): int((code == index).sum()) for index in range(8)
    }
    if sum(intersections.values()) != len(sample_manifest):
        raise AssertionError("independent three-route intersections lose denominator rows")
    _assert_equal(
        reported_statistics["three_route_outcome_intersections"],
        intersections,
        "three_route_outcome_intersections",
    )
    _assert_equal(
        reported_statistics["bootstrap_contract"],
        {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "primary_cluster": "scene_id",
            "sensitivity_cluster": "frame_id",
        },
        "bootstrap_contract",
    )
    output = (output_dir or (root / "15_independent_recompute")).resolve()
    metrics_path = output / "independent_metrics.json"
    sample_output = output / "independent_per_sample.parquet"
    report_path = output / "INDEPENDENT_RECOMPUTE.md"
    _atomic_json(
        metrics_path,
        {
            "systems": recomputed_metrics,
            "comparisons": comparisons,
            "three_route_outcome_intersections": intersections,
            "test_bridge_checks": bridge_checks,
        },
    )
    _atomic_parquet(sample_output, pd.concat(per_sample_parts, ignore_index=True))
    report = (
        "# Independent formal-Test recompute\n\n"
        "Status: **PASS**. This implementation used only FORMAL_TEST_LOCK-bound "
        "identifiers, frozen All pools, normalized candidate labels, and persisted formal "
        "outputs. No training, model, ranker, gate, router, or project statistics module "
        "was imported; the hash-locked frozen evaluator was loaded only to independently "
        "recompute the secondary bridge labels.\n\n"
        f"- Samples: {len(sample_manifest)}\n"
        f"- Systems: {len(systems)}\n"
        f"- Metric fields checked: {metric_checks}\n"
        f"- Complete statistical comparisons checked: {comparison_checks}\n"
        f"- Independent metrics SHA-256: `{_sha256(metrics_path)}`\n"
        f"- Independent per-sample SHA-256: `{_sha256(sample_output)}`\n"
        f"- FORMAL_TEST_LOCK SHA-256: `{_sha256(lock_path)}`\n"
    )
    _atomic_text(report_path, report)
    result = {
        "status": "PASS",
        "implementation": "standalone_lock_bound_id_join_without_project_training_or_ranker_imports",
        "formal_test_execution_count": 1,
        "metric_checks": metric_checks,
        "comparison_checks": comparison_checks,
        "sample_count": len(sample_manifest),
        "system_count": len(systems),
        "test_bridge_checks": bridge_checks,
        "sources": {
            "formal_lock": _record(lock_path),
            "execution": _record(execution_path),
            "formal_manifest": _record(manifest_path),
            "labels": _record(label_path),
            "label_manifest": _record(label_manifest_path),
            "sample_manifest": _record(sample_path),
            "evaluation_plan": _record(plan_path),
            "candidate_pools": {
                route: _record(Path(str(label_manifest["candidate_pools"][route]["path"])))
                for route in ROUTES
            },
        },
        "artifacts": {
            "metrics": _record(metrics_path),
            "per_sample": _record(sample_output),
            "report": _record(report_path),
        },
    }
    result["content_sha256"] = _canonical_sha256(result)
    _atomic_json(output / "INDEPENDENT_RECOMPUTE.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    repository = Path(__file__).resolve().parents[2]
    source = repository / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from unified_reranking.hashing import sha256_file
    from unified_reranking.ledger import ledger_stage

    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P15",
        substage="independent_formal_recompute",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(root, args.output_dir)
        artifact = Path(
            args.output_dir or root / "15_independent_recompute"
        ) / "INDEPENDENT_RECOMPUTE.json"
        state["artifact_path"] = str(artifact.resolve())
        state["artifact_sha256"] = sha256_file(artifact)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_exact_mcnemar", "run"]
