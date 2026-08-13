"""Independently replay the locked D1 formal evaluation after Test execution.

This module deliberately imports no D1 training, ranking, gate-selection, or
formal-transaction implementation.  Its only non-stdlib dependencies are
NumPy, pandas, and the byte-frozen evaluator loaded from the formal lock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import pandas as pd


LOCK = "08_lock/FORMAL_TEST_LOCK.json"
LOCK_DIGEST = "08_lock/FORMAL_TEST_LOCK.sha256"
EXECUTION = "09_formal_test/FORMAL_TEST_EXECUTION.json"
FORMAL_MANIFEST = "09_formal_test/formal_test_manifest.json"
FORMAL_BUNDLE = "09_formal_test/formal_candidate_score_decision_bundle.parquet"
FORMAL_OUTCOMES = "09_formal_test/formal_candidate_outcomes.parquet"
OUTPUT = "17_independent_recompute"
SYSTEMS = {
    "d1_top5_r0",
    "d1_top5_r7_ungated",
    "d1_top5_r7_gated",
    "d1_top10_locked",
    "d1_allnms_locked",
    "four_route_crog_default_router",
    "top20_union",
    "three_route_crog_default_router_reference",
    "top15_union_reference",
}
ID_COLUMNS = ["source_route", "sample_id", "candidate_id"]
VALUE_COLUMNS = [
    "candidate_geometry_sha256",
    "native_rank",
    "native_score",
    "cx_px",
    "cy_px",
    "theta_deg",
    "width_px",
    "height_px",
]
UNIVERSE_COLUMNS = [*ID_COLUMNS, *VALUE_COLUMNS]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, final_path: Path | None = None) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"independent recompute artifact is not regular: {source}")
    return {
        "path": str((final_path or source).resolve()),
        "sha256": _sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _verified_path(record: object, *, name: str) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{name} record is absent")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    observed = _record(path)
    if (
        record.get("path") != observed["path"]
        or record.get("sha256") != observed["sha256"]
        or ("bytes" in record and record.get("bytes") != observed["bytes"])
    ):
        raise RuntimeError(f"{name} record or bytes differ")
    return path


def _load_json(path: Path, *, name: str, content_hash: bool = False) -> dict[str, Any]:
    source = path.expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} is not a JSON object")
    if content_hash:
        unsigned = dict(value)
        recorded = unsigned.pop("content_sha256", None)
        if recorded != _canonical_sha256(unsigned):
            raise RuntimeError(f"{name} content hash differs")
    return value


def _verify_lock(root: Path) -> dict[str, Any]:
    lock_path = root / LOCK
    digest = (root / LOCK_DIGEST).read_text(encoding="ascii").strip()
    if digest != _sha256_file(lock_path):
        raise RuntimeError("independent recompute formal-lock detached hash differs")
    lock = _load_json(lock_path, name="formal lock")
    unsigned = dict(lock)
    recorded = unsigned.pop("self_sha256", None)
    if lock.get("status") != "LOCKED" or recorded != _canonical_sha256(unsigned):
        raise RuntimeError("independent recompute formal-lock self hash differs")
    inventory = lock.get("inventory")
    if (
        not isinstance(inventory, Mapping)
        or lock.get("inventory_count") != len(inventory)
        or lock.get("inventory_sha256") != _canonical_sha256(inventory)
    ):
        raise RuntimeError("independent recompute formal-lock inventory differs")
    for name, artifact in inventory.items():
        _verified_path(artifact, name=f"formal inventory {name}")
    return lock


def _load_evaluator(path: Path, expected_sha256: str) -> ModuleType:
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError("independent recompute evaluator bytes differ")
    name = f"_d1_independent_evaluator_{observed[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _record_present(value: Any, expected: Mapping[str, Any]) -> bool:
    if isinstance(value, Mapping):
        if value.get("path") == expected.get("path") and value.get(
            "sha256"
        ) == expected.get("sha256"):
            return True
        return any(_record_present(child, expected) for child in value.values())
    if isinstance(value, list):
        return any(_record_present(child, expected) for child in value)
    return False


def _label_candidate(module: ModuleType, row: Any) -> dict[str, Any]:
    candidate = module.CanonicalGrasp(
        cx_px=float(row.cx_px),
        cy_px=float(row.cy_px),
        theta_deg=float(row.theta_deg),
        jaw_width_px=float(row.width_px),
        rectangle_height_px=float(row.height_px),
        native_score=float(row.native_score),
        native_rank=int(row.native_rank),
        source_method=str(row.source_route),
        sample_id=str(row.sample_id),
    )
    ground_truth = tuple(
        module.gt_from_corners(
            np.stack([np.asarray(point, dtype=np.float64) for point in rectangle])
        )
        for rectangle in row.gt_grasp_rectangles
    )
    evaluated = module.evaluate_candidate(candidate, ground_truth)
    pairwise = evaluated["pairwise"]
    margins = [
        float(
            np.clip(
                min(
                    (float(pair["iou"]) - module.IOU_THRESHOLD) / module.IOU_THRESHOLD,
                    (module.ANGLE_THRESHOLD_DEG - float(pair["angle_error_deg"]))
                    / module.ANGLE_THRESHOLD_DEG,
                ),
                -1.0,
                1.0,
            )
        )
        for pair in pairwise
    ]
    if pairwise:
        index = max(
            range(len(pairwise)),
            key=lambda item: (
                margins[item],
                float(pairwise[item]["iou"]),
                -float(pairwise[item]["angle_error_deg"]),
                -int(pairwise[item]["gt_index"]),
            ),
        )
        match = pairwise[index]
        matched_index: int | None = int(match["gt_index"])
        iou = float(match["iou"])
        angle = float(match["angle_error_deg"])
        margin = margins[index]
    else:
        matched_index, iou, angle, margin = None, math.nan, math.nan, -1.0
    return {
        "candidate_success": bool(evaluated["success"]),
        "best_same_gt_iou": iou,
        "best_same_gt_angle_error_deg": angle,
        "matched_gt_index": matched_index,
        "jacquard_margin": margin,
    }


def _exact_universe(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> None:
    keys = ID_COLUMNS
    left = (
        left.loc[:, UNIVERSE_COLUMNS]
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        right.loc[:, UNIVERSE_COLUMNS]
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(left) != len(right):
        raise RuntimeError(f"{name} universe row count differs")
    for column in [*ID_COLUMNS, "candidate_geometry_sha256"]:
        if not left[column].astype(str).equals(right[column].astype(str)):
            raise RuntimeError(f"{name} universe {column} differs")
    numeric = [
        column for column in VALUE_COLUMNS if column != "candidate_geometry_sha256"
    ]
    if not np.array_equal(
        left[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float),
        right[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float),
        equal_nan=False,
    ):
        raise RuntimeError(f"{name} universe geometry/rank/score differs")


def _load_systems(
    plan: Mapping[str, Any], denominator: list[str]
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    entries = plan.get("systems")
    if not isinstance(entries, list):
        raise RuntimeError("independent recompute systems are absent")
    by_name = {
        str(value.get("name")): value for value in entries if isinstance(value, Mapping)
    }
    if set(by_name) != SYSTEMS or len(by_name) != len(entries):
        raise RuntimeError(
            "independent recompute requires exactly seven declared systems"
        )
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    universes: list[pd.DataFrame] = []
    for name, system in by_name.items():
        manifest_path = _verified_path(system.get("manifest"), name=f"{name} manifest")
        manifest = _load_json(manifest_path, name=f"{name} manifest", content_hash=True)
        if (
            manifest.get("status") not in {"COMPLETE", "LOCKED", "PASS"}
            or manifest.get("candidate_test_labels_read") is not False
        ):
            raise RuntimeError(f"{name} manifest status/Test-isolation drift")
        universe_path = _verified_path(
            system.get("candidate_universe"), name=f"{name} universe"
        )
        score_path = _verified_path(
            system.get("candidate_scores"), name=f"{name} scores"
        )
        decision_path = _verified_path(
            system.get("per_sample_decisions"), name=f"{name} decisions"
        )
        for key in (
            "candidate_universe",
            "candidate_scores",
            "per_sample_decisions",
        ):
            record = system.get(key)
            if not isinstance(record, Mapping) or not _record_present(manifest, record):
                raise RuntimeError(f"{name} manifest does not bind exact {key}")
        universe = pd.read_parquet(universe_path, columns=UNIVERSE_COLUMNS)
        universe[[*ID_COLUMNS, "candidate_geometry_sha256"]] = universe[
            [*ID_COLUMNS, "candidate_geometry_sha256"]
        ].astype(str)
        if universe.empty or universe.duplicated(ID_COLUMNS).any():
            raise RuntimeError(f"{name} universe is empty or duplicate")
        routes = set(map(str, system.get("routes", [])))
        if set(universe["source_route"]) - routes:
            raise RuntimeError(f"{name} source-route drift")
        numeric = universe[
            [
                column
                for column in VALUE_COLUMNS
                if column != "candidate_geometry_sha256"
            ]
        ].apply(pd.to_numeric, errors="coerce")
        if (
            not np.isfinite(numeric.to_numpy(float)).all()
            or (numeric[["width_px", "height_px"]].to_numpy(float) <= 0).any()
            or not universe["candidate_geometry_sha256"]
            .str.fullmatch(r"[0-9a-f]{64}")
            .all()
        ):
            raise RuntimeError(f"{name} geometry drift")
        scores = pd.read_parquet(score_path)
        score_keys = scores[ID_COLUMNS].astype(str)
        if score_keys.duplicated().any():
            raise RuntimeError(f"{name} duplicate score identity")
        if set(map(tuple, score_keys.to_numpy())) != set(
            map(tuple, universe[ID_COLUMNS].to_numpy())
        ) or len(scores) != len(universe):
            raise RuntimeError(f"{name} missing/extra candidate scores")
        rank = pd.to_numeric(scores["rank"], errors="coerce")
        if (
            not np.isfinite(rank).all()
            or (rank <= 0).any()
            or not np.equal(rank, np.floor(rank)).all()
            or scores.assign(_rank=rank)
            .duplicated(["source_route", "sample_id", "_rank"])
            .any()
        ):
            raise RuntimeError(f"{name} rerank drift")
        native_rank = pd.to_numeric(universe["native_rank"], errors="coerce")
        if name == "four_route_crog_default_router" and (
            universe.duplicated(["sample_id", "source_route"]).any()
            or not native_rank.eq(1).all()
            or not rank.eq(1).all()
        ):
            raise RuntimeError("four-route router source-route/rank drift")
        if name == "top20_union" and (
            universe.groupby("sample_id").size().gt(20).any()
            or scores.assign(_rank=rank).duplicated(["sample_id", "_rank"]).any()
            or scores.groupby("sample_id").size().gt(20).any()
        ):
            raise RuntimeError("route-qualified Top20 universe/rank drift")
        decisions = pd.read_parquet(decision_path)
        decisions[["sample_id", "selected_source_route", "selected_candidate_id"]] = (
            decisions[["sample_id", "selected_source_route", "selected_candidate_id"]]
            .fillna("")
            .astype(str)
        )
        if (
            decisions["sample_id"].duplicated().any()
            or set(decisions["sample_id"]) != set(denominator)
            or len(decisions) != len(denominator)
        ):
            raise RuntimeError(f"{name} decision denominator drift")
        allowed = set(
            map(
                tuple,
                scores[["sample_id", "source_route", "candidate_id"]]
                .astype(str)
                .to_numpy(),
            )
        )
        for row in decisions.itertuples(index=False):
            identity = (
                str(row.sample_id),
                str(row.selected_source_route),
                str(row.selected_candidate_id),
            )
            if bool(row.selected_source_route) != bool(row.selected_candidate_id):
                raise RuntimeError(f"{name} partial selected identity")
            if row.selected_candidate_id and identity not in allowed:
                raise RuntimeError(f"{name} selected identity outside universe")
        loaded[name] = {"universe": universe, "scores": scores, "decisions": decisions}
        universes.append(universe)
    combined = pd.concat(universes, ignore_index=True)
    for column in VALUE_COLUMNS:
        if (
            combined.groupby(ID_COLUMNS, dropna=False)[column]
            .nunique(dropna=False)
            .gt(1)
            .any()
        ):
            raise RuntimeError(f"cross-system candidate {column} drift")
    return loaded, combined.drop_duplicates(ID_COLUMNS).reset_index(drop=True)


def _verify_d1_pools(
    plan: Mapping[str, Any], systems: Mapping[str, Mapping[str, pd.DataFrame]]
) -> None:
    components = plan["components"]
    bindings = {
        "d1_top5_r0": "d1_top5_candidates",
        "d1_top5_r7_ungated": "d1_top5_candidates",
        "d1_top5_r7_gated": "d1_top5_candidates",
        "d1_top10_locked": "d1_top10_candidates",
        "d1_allnms_locked": "d1_allnms_candidates",
    }
    for system_name, component_name in bindings.items():
        pool_path = _verified_path(components[component_name], name=component_name)
        columns = [
            "route",
            "sample_id",
            "candidate_id",
            *VALUE_COLUMNS,
        ]
        pool = pd.read_parquet(pool_path, columns=columns).rename(
            columns={"route": "source_route"}
        )
        _exact_universe(systems[system_name]["universe"], pool, name=system_name)


def _append_postclaim_event(root: Path, value: Mapping[str, Any]) -> None:
    path = root / "09_formal_test" / "test_access.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **dict(value)}
    payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _system_recompute(
    *,
    name: str,
    data: Mapping[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    denominator: pd.DataFrame,
    baseline_decisions: pd.DataFrame,
    baseline_correct: Mapping[str, bool],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = data["scores"].copy()
    scores[ID_COLUMNS] = scores[ID_COLUMNS].astype(str)
    joined = scores.merge(outcomes, on=ID_COLUMNS, how="left", validate="one_to_one")
    if joined["candidate_success"].isna().any():
        raise RuntimeError(f"{name} outcome/score label drift")
    decisions = data["decisions"].copy()
    selected = decisions.loc[decisions["selected_candidate_id"].ne("")].merge(
        joined,
        left_on=["sample_id", "selected_source_route", "selected_candidate_id"],
        right_on=["sample_id", "source_route", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if selected["candidate_success"].isna().any():
        raise RuntimeError(f"{name} selected label drift")
    selected_correct = {
        str(row.sample_id): bool(row.candidate_success)
        for row in selected.itertuples(index=False)
    }
    decision_identity = decisions.set_index("sample_id")[
        ["selected_source_route", "selected_candidate_id"]
    ]
    joined["is_selected"] = [
        (
            str(row.source_route),
            str(row.candidate_id),
        )
        == tuple(map(str, decision_identity.loc[str(row.sample_id)]))
        for row in joined.itertuples(index=False)
    ]
    joined["effective_rank"] = 0
    for sample_id, indices in joined.groupby("sample_id", sort=False).groups.items():
        if not str(decision_identity.loc[sample_id, "selected_candidate_id"]):
            continue
        ordered = sorted(
            indices,
            key=lambda index: (
                0 if bool(joined.at[index, "is_selected"]) else 1,
                int(joined.at[index, "rank"]),
                str(joined.at[index, "source_route"]),
                str(joined.at[index, "candidate_id"]),
            ),
        )
        for effective_rank, index in enumerate(ordered, 1):
            joined.at[index, "effective_rank"] = effective_rank
    oracle_positive = joined.loc[joined["candidate_success"].astype(bool)]
    positive = joined.loc[
        joined["candidate_success"].astype(bool) & joined["effective_rank"].gt(0)
    ]
    first_positive = positive.groupby("sample_id")["effective_rank"].min().to_dict()
    oracle_samples = set(map(str, oracle_positive["sample_id"]))
    baseline_identity = baseline_decisions.set_index("sample_id")[
        ["selected_source_route", "selected_candidate_id"]
    ]
    rows = []
    recovered = harmful = switched = 0
    for sample in denominator.itertuples(index=False):
        sample_id = str(sample.sample_id)
        correct = bool(selected_correct.get(sample_id, False))
        baseline = bool(baseline_correct.get(sample_id, False))
        identity = tuple(map(str, decision_identity.loc[sample_id]))
        baseline_id = tuple(map(str, baseline_identity.loc[sample_id]))
        recovered += int(correct and not baseline)
        harmful += int(baseline and not correct)
        switched += int(identity != baseline_id)
        rows.append(
            {
                "system_name": name,
                "sample_id": sample_id,
                "scene_id": str(getattr(sample, "scene_id", "")),
                "frame_id": str(getattr(sample, "frame_id", "")),
                "selected_source_route": identity[0],
                "selected_candidate_id": identity[1],
                "selected_correct": correct,
                "no_output": identity[1] == "",
                "oracle": sample_id in oracle_samples,
                "first_positive_rank": first_positive.get(sample_id, np.nan),
                "reciprocal_rank": (
                    1.0 / float(first_positive[sample_id])
                    if sample_id in first_positive
                    else 0.0
                ),
                "recovered_vs_d1_r0": correct and not baseline,
                "harmful_vs_d1_r0": baseline and not correct,
                "switched_vs_d1_r0": identity != baseline_id,
            }
        )
    per_sample = pd.DataFrame(rows)
    count = len(denominator)
    numerator = int(per_sample["selected_correct"].sum())
    oracle = int(per_sample["oracle"].sum())
    base_j = sum(baseline_correct.values()) / count
    j_at_1 = numerator / count
    oracle_rate = oracle / count
    headroom_denominator = oracle_rate - base_j
    max_k = int(joined["effective_rank"].max()) if len(joined) else 0
    first_distribution: dict[str, int] = {}
    for sample_id in denominator["sample_id"].astype(str):
        key = (
            str(int(first_positive[sample_id]))
            if sample_id in first_positive
            else "no_positive"
        )
        first_distribution[key] = first_distribution.get(key, 0) + 1
    rank_metrics = []
    for k in range(1, max_k + 1):
        j_numerator = sum(float(rank) <= k for rank in first_positive.values())
        mrr = (
            sum(
                1.0 / float(rank) if float(rank) <= k else 0.0
                for rank in first_positive.values()
            )
            / count
        )
        ndcg_total = 0.0
        for sample_id in denominator["sample_id"].astype(str):
            sample_rows = joined.loc[
                joined["sample_id"].eq(sample_id)
                & joined["effective_rank"].between(1, k)
            ].sort_values("effective_rank", kind="mergesort")
            relevance = sample_rows["candidate_success"].astype(float).to_numpy()
            dcg = sum(
                value / np.log2(index + 2.0) for index, value in enumerate(relevance)
            )
            positives = int(
                joined.loc[
                    joined["sample_id"].eq(sample_id) & joined["effective_rank"].gt(0),
                    "candidate_success",
                ].sum()
            )
            ideal = sum(
                1.0 / np.log2(index + 2.0) for index in range(min(k, positives))
            )
            ndcg_total += 0.0 if ideal == 0.0 else dcg / ideal
        rank_metrics.append(
            {
                "k": k,
                "j_at_k_numerator": int(j_numerator),
                "j_at_k": j_numerator / count,
                "mrr_at_k": mrr,
                "ndcg_at_k": ndcg_total / count,
            }
        )
    metrics = {
        "sample_count": count,
        "selected_correct_numerator": numerator,
        "j_at_1": j_at_1,
        "oracle_numerator": oracle,
        "oracle": oracle_rate,
        "mrr": float(per_sample["reciprocal_rank"].mean()),
        "headroom_recovery": (
            None
            if abs(headroom_denominator) <= 1e-15
            else (j_at_1 - base_j) / headroom_denominator
        ),
        "recovered": recovered,
        "harmful": harmful,
        "net": recovered - harmful,
        "switch_count": switched,
        "switch_rate": switched / count,
        "outcome_changing_precision": (
            None if recovered + harmful == 0 else recovered / (recovered + harmful)
        ),
        "no_output_samples": int(per_sample["no_output"].sum()),
        "max_k": max_k,
        "rank_metric_inventory": [
            "j_at_k_numerator",
            "j_at_k",
            "mrr_at_k",
            "ndcg_at_k",
        ],
        "rank_metrics": rank_metrics,
        "first_positive_rank_distribution": first_distribution,
    }
    return per_sample, metrics


def independent_recompute(
    run_dir: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    destination = root / OUTPUT
    if destination.exists():
        metrics = _load_json(
            destination / "recomputed_metrics.json",
            name="independent recompute metrics",
            content_hash=True,
        )
        self_unsigned = dict(metrics)
        self_unsigned.pop("content_sha256", None)
        recorded_self = self_unsigned.pop("self_sha256", None)
        if recorded_self != _canonical_sha256(self_unsigned):
            raise RuntimeError("independent recompute self hash differs")
        if resume and metrics.get("status") == "PASS":
            return metrics
        raise FileExistsError(
            f"independent recompute output already exists: {destination}"
        )

    lock = _verify_lock(root)
    execution = _load_json(root / EXECUTION, name="formal execution", content_hash=True)
    if (
        execution.get("status") != "COMPLETE"
        or execution.get("execution_count") != 1
        or execution.get("execution_consumed") is True
    ):
        raise RuntimeError(
            "independent recompute requires one COMPLETE formal execution"
        )
    execution_lock = _verified_path(
        execution.get("formal_lock"), name="execution formal lock"
    )
    if execution_lock != (root / LOCK).resolve():
        raise RuntimeError("independent recompute execution/lock binding differs")
    plan_path = _verified_path(lock.get("evaluation_plan"), name="formal plan")
    plan = _load_json(plan_path, name="formal plan", content_hash=True)
    if (
        plan.get("status") != "LOCK_READY"
        or lock.get("evaluation_plan_content_sha256") != plan.get("content_sha256")
        or set(str(value.get("name")) for value in plan.get("systems", [])) != SYSTEMS
    ):
        raise RuntimeError("independent recompute evaluation plan differs")
    formal_manifest = _load_json(
        root / FORMAL_MANIFEST, name="formal Test manifest", content_hash=True
    )
    if (
        formal_manifest.get("status") != "COMPLETE"
        or formal_manifest.get("formal_test_execution_count") != 1
    ):
        raise RuntimeError("independent recompute formal manifest differs")
    execution_artifacts = execution.get("artifacts")
    if not isinstance(execution_artifacts, Mapping):
        raise RuntimeError(
            "independent recompute execution artifact inventory is absent"
        )
    execution_manifest_path = _verified_path(
        execution_artifacts.get("formal_test_manifest"),
        name="execution formal manifest",
    )
    if execution_manifest_path != (root / FORMAL_MANIFEST).resolve():
        raise RuntimeError(
            "independent recompute execution/formal-manifest binding differs"
        )

    paired_path = _verified_path(
        plan["components"]["paired_test_manifest"], name="paired Test denominator"
    )
    available = pd.read_parquet(paired_path).columns
    denominator_columns = [
        column
        for column in ("sample_id", "scene_id", "frame_id")
        if column in available
    ]
    denominator = pd.read_parquet(paired_path, columns=denominator_columns)
    denominator["sample_id"] = denominator["sample_id"].astype(str)
    if denominator.empty or denominator["sample_id"].duplicated().any():
        raise RuntimeError("independent recompute paired denominator differs")
    denominator_ids = denominator["sample_id"].tolist()
    systems, universe = _load_systems(plan, denominator_ids)
    _verify_d1_pools(plan, systems)

    ground_truth_path = _verified_path(
        plan.get("raw_test_ground_truth"), name="raw Test ground truth"
    )
    formal_ground_truth = formal_manifest.get("raw_test_ground_truth")
    if (
        not isinstance(formal_ground_truth, Mapping)
        or formal_ground_truth.get("opened_once_after_claim") is not True
        or formal_ground_truth.get("path") != plan["raw_test_ground_truth"].get("path")
        or formal_ground_truth.get("sha256")
        != plan["raw_test_ground_truth"].get("sha256")
    ):
        raise RuntimeError("independent recompute formal raw-GT binding differs")
    ground_truth_bytes = ground_truth_path.read_bytes()
    if (
        hashlib.sha256(ground_truth_bytes).hexdigest()
        != plan["raw_test_ground_truth"]["sha256"]
    ):
        raise RuntimeError("independent recompute raw Test ground-truth bytes differ")
    import io

    ground_truth = pd.read_parquet(io.BytesIO(ground_truth_bytes))
    if (
        not {"sample_id", "gt_grasp_rectangles"}.issubset(ground_truth.columns)
        or ground_truth["sample_id"].astype(str).duplicated().any()
        or set(ground_truth["sample_id"].astype(str)) != set(denominator_ids)
    ):
        raise RuntimeError(
            "independent recompute raw Test ground-truth denominator differs"
        )
    evaluator_path = _verified_path(
        plan["components"]["canonical_evaluator"], name="canonical evaluator"
    )
    if plan.get("ground_truth_evaluator_sha256") != _sha256_file(evaluator_path):
        raise RuntimeError("independent recompute evaluator binding differs")
    module = _load_evaluator(evaluator_path, _sha256_file(evaluator_path))
    joined = universe.merge(
        ground_truth[["sample_id", "gt_grasp_rectangles"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if joined["gt_grasp_rectangles"].isna().any():
        raise RuntimeError("independent recompute candidate missing raw ground truth")
    evaluated = pd.DataFrame(
        [_label_candidate(module, row) for row in joined.itertuples(index=False)]
    )
    outcomes = pd.concat(
        [universe.reset_index(drop=True), evaluated.reset_index(drop=True)], axis=1
    )
    formal_outcomes_path = _verified_path(
        formal_manifest["artifacts"]["candidate_outcomes"],
        name="formal candidate outcomes",
    )
    formal_outcomes = pd.read_parquet(formal_outcomes_path)
    outcome_columns = [
        *ID_COLUMNS,
        "candidate_success",
        "best_same_gt_iou",
        "best_same_gt_angle_error_deg",
        "matched_gt_index",
        "jacquard_margin",
    ]
    left = (
        outcomes[outcome_columns]
        .sort_values(ID_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        formal_outcomes[outcome_columns]
        .sort_values(ID_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(left) != len(right) or not left[ID_COLUMNS].astype(str).equals(
        right[ID_COLUMNS].astype(str)
    ):
        raise RuntimeError("independent recompute formal outcome membership drift")
    if (
        not left["candidate_success"]
        .astype(bool)
        .equals(right["candidate_success"].astype(bool))
    ):
        raise RuntimeError("independent recompute candidate label drift")
    for column in [
        "best_same_gt_iou",
        "best_same_gt_angle_error_deg",
        "jacquard_margin",
    ]:
        if not np.allclose(
            pd.to_numeric(left[column], errors="coerce"),
            pd.to_numeric(right[column], errors="coerce"),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(f"independent recompute evaluator {column} drift")

    baseline_data = systems["d1_top5_r0"]
    baseline_selected = (
        baseline_data["decisions"]
        .loc[baseline_data["decisions"]["selected_candidate_id"].ne("")]
        .merge(
            outcomes,
            left_on=["sample_id", "selected_source_route", "selected_candidate_id"],
            right_on=["sample_id", "source_route", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    )
    baseline_correct = {sample_id: False for sample_id in denominator_ids}
    baseline_correct.update(
        {
            str(row.sample_id): bool(row.candidate_success)
            for row in baseline_selected.itertuples(index=False)
        }
    )
    per_sample_parts = []
    metrics = {}
    for name in sorted(SYSTEMS):
        per_sample, system_metrics = _system_recompute(
            name=name,
            data=systems[name],
            outcomes=outcomes,
            denominator=denominator,
            baseline_decisions=baseline_data["decisions"],
            baseline_correct=baseline_correct,
        )
        per_sample_parts.append(per_sample)
        metrics[name] = system_metrics
    independent_per_sample = pd.concat(per_sample_parts, ignore_index=True)
    formal_bundle_path = _verified_path(
        formal_manifest["artifacts"]["candidate_score_decision_bundle"],
        name="formal candidate-score/decision bundle",
    )
    if formal_bundle_path != (root / FORMAL_BUNDLE).resolve():
        raise RuntimeError("independent recompute formal bundle path drift")
    formal_bundle = pd.read_parquet(formal_bundle_path)
    if set(formal_bundle["system_name"].astype(str)) != SYSTEMS:
        raise RuntimeError("independent recompute formal bundle system drift")
    formal_per_sample = formal_bundle.groupby(
        ["system_name", "sample_id"], as_index=False
    ).agg(selected_correct=("selected_correct", "max"), no_output=("no_output", "max"))
    formal_metrics_path = _verified_path(
        formal_manifest["artifacts"]["metrics"], name="formal metrics"
    )
    formal_metrics = _load_json(formal_metrics_path, name="formal metrics")
    if formal_metrics.get("status") != "COMPLETE":
        raise RuntimeError("independent recompute formal metrics status drift")
    for name, values in metrics.items():
        observed_rows = (
            independent_per_sample.loc[
                independent_per_sample["system_name"].eq(name),
                ["sample_id", "selected_correct", "no_output"],
            ]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        formal_rows = (
            formal_per_sample.loc[
                formal_per_sample["system_name"].eq(name),
                ["sample_id", "selected_correct", "no_output"],
            ]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        if (
            len(observed_rows) != len(formal_rows)
            or not observed_rows["sample_id"]
            .astype(str)
            .equals(formal_rows["sample_id"].astype(str))
            or not observed_rows[["selected_correct", "no_output"]]
            .astype(bool)
            .equals(formal_rows[["selected_correct", "no_output"]].astype(bool))
        ):
            raise RuntimeError(f"independent recompute formal per-sample drift: {name}")
        observed = int(observed_rows["selected_correct"].sum())
        if observed != values["selected_correct_numerator"]:
            raise RuntimeError(
                f"independent recompute internal selected count drift: {name}"
            )
        locked_metrics = formal_metrics.get("systems", {}).get(name)
        if not isinstance(locked_metrics, Mapping):
            raise RuntimeError(f"independent recompute formal metrics miss {name}")
        for field in (
            "sample_count",
            "selected_correct_numerator",
            "oracle_numerator",
            "no_output_samples",
            "max_k",
            "rank_metric_inventory",
            "first_positive_rank_distribution",
        ):
            if locked_metrics.get(field) != values[field]:
                raise RuntimeError(
                    f"independent recompute formal metric {name}/{field} drift"
                )
        for field in ("j_at_1", "oracle"):
            if not math.isclose(
                float(locked_metrics.get(field, math.nan)),
                float(values[field]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError(
                    f"independent recompute formal metric {name}/{field} drift"
                )
        locked_rank_metrics = locked_metrics.get("rank_metrics")
        if not isinstance(locked_rank_metrics, list) or len(locked_rank_metrics) != len(
            values["rank_metrics"]
        ):
            raise RuntimeError(
                f"independent recompute formal rank-metric inventory drift: {name}"
            )
        for locked_row, recomputed_row in zip(
            locked_rank_metrics, values["rank_metrics"], strict=True
        ):
            if (
                not isinstance(locked_row, Mapping)
                or locked_row.get("k") != recomputed_row["k"]
            ):
                raise RuntimeError(
                    f"independent recompute formal rank key drift: {name}"
                )
            if locked_row.get("j_at_k_numerator") != recomputed_row["j_at_k_numerator"]:
                raise RuntimeError(
                    f"independent recompute formal J@K numerator drift: {name}"
                )
            for field in ("j_at_k", "mrr_at_k", "ndcg_at_k"):
                if not math.isclose(
                    float(locked_row.get(field, math.nan)),
                    float(recomputed_row[field]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise RuntimeError(
                        f"independent recompute formal rank metric {name}/{field} drift"
                    )

    _append_postclaim_event(
        root,
        {
            "event_id": "d1-independent-recompute-raw-test-ground-truth-read-v1",
            "event": "d1_independent_recompute_raw_test_ground_truth_read",
            "opened_after_complete_formal_claim": True,
            "formal_execution_count": 1,
            "raw_test_ground_truth_path": str(ground_truth_path),
            "raw_test_ground_truth_sha256": _sha256_file(ground_truth_path),
            "row_count": len(ground_truth),
            "selection_feedback_used": False,
        },
    )

    temporary = root / f".{OUTPUT}.{os.getpid()}.tmp"
    if temporary.exists():
        raise FileExistsError(
            f"independent recompute staging directory already exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    per_sample_path = temporary / "independent_per_sample.parquet"
    independent_per_sample.to_parquet(per_sample_path, index=False, compression="zstd")
    source_records = {
        "formal_lock": _record(root / LOCK),
        "formal_execution": _record(root / EXECUTION),
        "formal_evaluation_plan": _record(plan_path),
        "formal_manifest": _record(root / FORMAL_MANIFEST),
        "formal_bundle": _record(root / FORMAL_BUNDLE),
        "formal_candidate_outcomes": _record(formal_outcomes_path),
        "raw_test_ground_truth": _record(ground_truth_path),
        "canonical_evaluator": _record(evaluator_path),
        "statistics_config": _record(
            _verified_path(
                plan["components"]["statistics_config"], name="statistics config"
            )
        ),
    }
    source_signature = _canonical_sha256(source_records)
    report = (
        "# D1 independent recompute\n\n"
        "- Status: **PASS**\n"
        f"- Formal execution count: `1`\n"
        f"- Samples: `{len(denominator)}`\n"
        f"- Systems: `{len(SYSTEMS)}`\n"
        f"- Route-qualified candidate outcomes: `{len(outcomes)}`\n"
        f"- Source signature: `{source_signature}`\n"
        "- Candidate membership, geometry, route, rank, raw-GT bytes, evaluator, "
        "selected correctness, and aggregate inputs independently matched.\n"
        "- No model, K, feature, threshold, gate, or router selection was changed.\n"
    )
    report_path = temporary / "INDEPENDENT_RECOMPUTE.md"
    report_path.write_text(report, encoding="utf-8")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "formal_execution_count": 1,
        "system_names": sorted(SYSTEMS),
        "sample_count": len(denominator),
        "candidate_outcome_count": len(outcomes),
        "metrics": metrics,
        "formal_metrics_match": True,
        "candidate_evaluator_integrity": "PASS",
        "source_signature_sha256": source_signature,
        "sources": source_records,
        "artifacts": {
            "independent_per_sample": _record(
                per_sample_path,
                final_path=destination / "independent_per_sample.parquet",
            ),
            "report": _record(
                report_path,
                final_path=destination / "INDEPENDENT_RECOMPUTE.md",
            ),
        },
    }
    payload["artifact_inventory_sha256"] = _canonical_sha256(payload["artifacts"])
    payload["self_sha256"] = _canonical_sha256(payload)
    payload["content_sha256"] = _canonical_sha256(payload)
    metrics_path = temporary / "recomputed_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _record_ledger(root: Path, *, command: str, status: str, error: str = "") -> None:
    path = root / "run_ledger.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT NOT NULL,
                substage TEXT NOT NULL DEFAULT '', route TEXT NOT NULL DEFAULT '',
                evidence_track TEXT NOT NULL DEFAULT '', pool TEXT NOT NULL DEFAULT '',
                method TEXT NOT NULL DEFAULT '', feature_set TEXT NOT NULL DEFAULT '',
                loss TEXT NOT NULL DEFAULT '', encoder TEXT NOT NULL DEFAULT '',
                seed INTEGER NOT NULL DEFAULT -1, status TEXT NOT NULL,
                start_time TEXT NOT NULL, end_time TEXT, command TEXT NOT NULL DEFAULT '',
                return_code INTEGER, artifact_path TEXT NOT NULL DEFAULT '',
                artifact_sha256 TEXT NOT NULL DEFAULT '', error_summary TEXT NOT NULL DEFAULT '',
                UNIQUE(stage, substage, route, evidence_track, pool, method,
                       feature_set, loss, encoder, seed)
            )
            """
        )
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        output = root / OUTPUT / "recomputed_metrics.json"
        connection.execute(
            """
            INSERT INTO stages (
                stage, substage, route, evidence_track, pool, method, seed,
                status, start_time, end_time, command, return_code,
                artifact_path, artifact_sha256, error_summary
            ) VALUES ('P17', 'd1_independent_recompute', 'D1',
                      'postclaim_independent', 'all_locked_systems',
                      'canonical_evaluator_replay', -1, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(stage, substage, route, evidence_track, pool, method,
                        feature_set, loss, encoder, seed)
            DO UPDATE SET status=excluded.status, end_time=excluded.end_time,
                          command=excluded.command, return_code=excluded.return_code,
                          artifact_path=excluded.artifact_path,
                          artifact_sha256=excluded.artifact_sha256,
                          error_summary=excluded.error_summary
            """,
            (
                status,
                now,
                now,
                command,
                0 if status == "COMPLETE" else 1,
                str(output.resolve()) if output.is_file() else "",
                _sha256_file(output) if output.is_file() else "",
                error,
            ),
        )
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, stage, substage, route, evidence_track, pool, method,
                   feature_set, loss, encoder, seed, status, start_time,
                   end_time, command, return_code, artifact_path,
                   artifact_sha256, error_summary
            FROM stages
            ORDER BY stage, substage, route, evidence_track, pool, method,
                     feature_set, loss, encoder, seed, id
            """
        ).fetchall()
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    temporary = root / f".commands.log.{os.getpid()}.tmp"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, root / "commands.log")


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    command = " ".join(map(str, sys.argv))
    try:
        independent_recompute(root, resume=args.resume)
    except Exception as error:
        _record_ledger(
            root,
            command=command,
            status="FAILED",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    _record_ledger(root, command=command, status="COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
