"""Normalize frozen three-route references for the D1 formal transaction."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.artifacts import verified_artifact_path
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.test_access_guard import append_access_log

from .contracts import assert_label_free_parquet_schema
from .execution import load_content_manifest


REFERENCE_NAMES = (
    "three_route_crog_default_router_reference",
    "top15_union_reference",
)
ID_COLUMNS = ["source_route", "sample_id", "candidate_id"]
UNIVERSE_COLUMNS = [
    *ID_COLUMNS,
    "candidate_geometry_sha256",
    "native_rank",
    "native_score",
    "cx_px",
    "cy_px",
    "theta_deg",
    "width_px",
    "height_px",
]


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 reference source is not a regular file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return path


def _decision_frame(path: Path, denominator: list[str], *, name: str) -> pd.DataFrame:
    assert_label_free_parquet_schema(path, name=name)
    frame = pd.read_parquet(path)
    aliases = {
        "selected_route": "selected_source_route",
        "route": "selected_source_route",
        "source_route": "selected_source_route",
        "candidate_id": "selected_candidate_id",
    }
    for source, target in aliases.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    required = ["sample_id", "selected_source_route", "selected_candidate_id"]
    if set(required).difference(frame.columns):
        raise RuntimeError(f"{name} decision schema differs")
    result = frame[required].copy()
    result[required] = result[required].fillna("").astype(str)
    result["selected_source_route"] = result["selected_source_route"].str.upper()
    if result["sample_id"].duplicated().any() or set(result["sample_id"]) != set(
        denominator
    ):
        raise RuntimeError(f"{name} decision denominator differs")
    if set(result["selected_source_route"]).difference({"", "CROG", "G1", "C1"}):
        raise RuntimeError(f"{name} decision contains a non-reference route")
    return result.set_index("sample_id").loc[denominator].reset_index()


def _load_normalized_manifest(
    path: Path, *, name: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = load_content_manifest(path, name=name, statuses=("COMPLETE",))
    if manifest.get("candidate_test_labels_read") is not False:
        raise PermissionError(f"{name} is not label-free")
    universe_path = verified_artifact_path(
        manifest.get("artifacts", {}).get("candidate_universe"), name=f"{name} universe"
    )
    universe = pd.read_parquet(universe_path)
    if set(UNIVERSE_COLUMNS).difference(universe.columns):
        raise RuntimeError(f"{name} universe schema differs")
    universe[ID_COLUMNS] = universe[ID_COLUMNS].astype(str)
    return manifest, universe[UNIVERSE_COLUMNS].copy()


def _assert_frozen_by_source_lock(root: Path, paths: list[Path]) -> dict[str, Any]:
    plan = load_content_manifest(
        root / "configs/d1_four_route_extension_plan.json",
        name="D1 four-route source plan",
        statuses=("PLANNED",),
    )
    lock_path = verified_artifact_path(
        plan["sources"]["completed_three_route_final_lock"],
        name="completed three-route final lock",
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inventory = lock.get("inventory")
    if not isinstance(inventory, list):
        raise RuntimeError("completed three-route final inventory is absent")
    by_path = {
        str(Path(str(row.get("path", ""))).resolve()): row
        for row in inventory
        if isinstance(row, Mapping)
    }
    for path in paths:
        current = _record(path)
        frozen = by_path.get(current["path"])
        if not isinstance(frozen, Mapping) or any(
            frozen.get(field) != current[field] for field in ("path", "sha256", "bytes")
        ):
            raise RuntimeError(
                f"D1 reference source is not frozen by the completed run: {path}"
            )
    return {
        "source_plan": _record(root / "configs/d1_four_route_extension_plan.json"),
        "source_lock": _record(lock_path),
    }


def assemble_reference_systems(
    run_dir: str | Path,
    *,
    three_route_decisions_path: str | Path,
    top15_decisions_path: str | Path,
    top15_ranking_path: str | Path,
    resume: bool = False,
) -> dict[str, dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    if any(
        (root / relative).exists()
        for relative in (
            "08_lock/FORMAL_TEST_LOCK.json",
            "09_formal_test/FORMAL_TEST_EXECUTION.json",
            "FINAL_RUN_LOCK.json",
        )
    ):
        raise PermissionError("D1 reference systems must be normalized before P14 lock")
    three_decisions_path = Path(three_route_decisions_path).expanduser().resolve()
    top15_decision_path = Path(top15_decisions_path).expanduser().resolve()
    top15_score_path = Path(top15_ranking_path).expanduser().resolve()
    frozen = _assert_frozen_by_source_lock(
        root, [three_decisions_path, top15_decision_path, top15_score_path]
    )
    denominator_path = root / "01_manifests/d1_paired_manifest.parquet"
    assert_label_free_parquet_schema(denominator_path, name="D1 reference denominator")
    denominator = (
        pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
        .astype(str)
        .tolist()
    )
    if not denominator or len(set(denominator)) != len(denominator):
        raise RuntimeError("D1 reference denominator is empty/duplicated")
    four_manifest, four_universe = _load_normalized_manifest(
        root / "08_lock/formal_inputs/four_route_crog_default_router/manifest.json",
        name="D1 four-route normalized input",
    )
    top20_manifest, top20_universe = _load_normalized_manifest(
        root / "08_lock/formal_inputs/top20_union/manifest.json",
        name="D1 Top20 normalized input",
    )
    three_decisions = _decision_frame(
        three_decisions_path, denominator, name="three-route frozen Test decisions"
    )
    three_universe = four_universe.loc[
        four_universe["source_route"].isin(["CROG", "G1", "C1"])
    ].copy()
    selected = three_decisions.loc[three_decisions["selected_candidate_id"].ne("")]
    realized_three = selected.merge(
        three_universe,
        left_on=["sample_id", "selected_source_route", "selected_candidate_id"],
        right_on=["sample_id", "source_route", "candidate_id"],
        how="left",
        indicator=True,
    )
    if not realized_three["_merge"].eq("both").all():
        raise RuntimeError("frozen three-route selection is outside its exact universe")
    three_scores = three_universe[ID_COLUMNS].copy()
    selected_keys = set(
        map(
            tuple,
            selected[
                ["selected_source_route", "sample_id", "selected_candidate_id"]
            ].itertuples(index=False, name=None),
        )
    )
    three_scores["score"] = [
        1.0 if tuple(row) in selected_keys else 0.0
        for row in three_scores[ID_COLUMNS].itertuples(index=False, name=None)
    ]
    three_scores["rank"] = 1

    top15_universe = top20_universe.loc[top20_universe["source_route"].ne("D1")].copy()
    assert_label_free_parquet_schema(top15_score_path, name="frozen Top15 ranking")
    raw_scores = pd.read_parquet(top15_score_path)
    if "route" in raw_scores and "source_route" not in raw_scores:
        raw_scores = raw_scores.rename(columns={"route": "source_route"})
    if "ensemble_score" in raw_scores and "score" not in raw_scores:
        raw_scores = raw_scores.rename(columns={"ensemble_score": "score"})
    required_scores = {*ID_COLUMNS, "score"}
    if required_scores.difference(raw_scores.columns):
        raise RuntimeError("frozen Top15 ranking schema differs")
    raw_scores[ID_COLUMNS] = raw_scores[ID_COLUMNS].astype(str)
    top15_scores = top15_universe[ID_COLUMNS].merge(
        raw_scores[[*ID_COLUMNS, "score"]], on=ID_COLUMNS, validate="one_to_one"
    )
    top15_scores["rank"] = (
        top15_scores.sort_values(
            ["sample_id", "score", "source_route", "candidate_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .groupby("sample_id", sort=False)
        .cumcount()
        .add(1)
        .reindex(top15_scores.index)
    )
    top15_decisions = _decision_frame(
        top15_decision_path, denominator, name="Top15 frozen Test decisions"
    )
    selected_top15 = top15_decisions.loc[
        top15_decisions["selected_candidate_id"].ne("")
    ]
    realized = selected_top15.merge(
        top15_universe,
        left_on=["sample_id", "selected_source_route", "selected_candidate_id"],
        right_on=["sample_id", "source_route", "candidate_id"],
        how="left",
        indicator=True,
    )
    if not realized["_merge"].eq("both").all():
        raise RuntimeError(
            "frozen Top15 selection is outside the exact non-D1 universe"
        )

    outputs = {
        "three_route_crog_default_router_reference": (
            three_universe,
            three_scores,
            three_decisions,
            [three_decisions_path],
        ),
        "top15_union_reference": (
            top15_universe,
            top15_scores,
            top15_decisions,
            [top15_decision_path, top15_score_path],
        ),
    }
    results = {}
    for name, (universe, scores, decisions, legacy_paths) in outputs.items():
        output = root / "08_lock/formal_inputs" / name
        marker = output / "manifest.json"
        sources = {
            **frozen,
            "legacy_locked_outputs": [_record(path) for path in legacy_paths],
            "denominator": _record(denominator_path),
            "four_route_input": _record(
                root
                / "08_lock/formal_inputs/four_route_crog_default_router/manifest.json"
            ),
            "top20_input": _record(
                root / "08_lock/formal_inputs/top20_union/manifest.json"
            ),
        }
        configuration = {
            "system": name,
            "role": "READ_ONLY_REFERENCE_DIAGNOSTIC",
            "routes": ["CROG", "G1", "C1"],
            "candidate_test_labels_read": False,
            "selection_used_test_metrics": False,
        }
        signature = canonical_sha256(
            {"configuration": configuration, "sources": sources}
        )
        if marker.exists():
            existing = load_content_manifest(marker, name=name, statuses=("COMPLETE",))
            if resume and existing.get("source_signature_sha256") == signature:
                results[name] = existing
                continue
            raise RuntimeError(f"D1 reference system exists and differs: {name}")
        artifacts = {
            "candidate_universe": _record(
                _atomic_parquet(universe, output / "candidate_universe.parquet")
            ),
            "candidate_scores": _record(
                _atomic_parquet(scores, output / "candidate_scores.parquet")
            ),
            "per_sample_decisions": _record(
                _atomic_parquet(decisions, output / "per_sample_decisions.parquet")
            ),
        }
        value: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE",
            "configuration": configuration,
            "source_signature_sha256": signature,
            "sources": sources,
            "artifacts": artifacts,
            "candidate_test_labels_read": False,
            "selection_used_test_metrics": False,
        }
        value["content_sha256"] = canonical_sha256(value)
        atomic_json(marker, value)
        results[name] = value
    output_manifests = {
        name: _record(root / "08_lock/formal_inputs" / name / "manifest.json")
        for name in REFERENCE_NAMES
    }
    legacy_sources = {
        "three_route_decisions": _record(three_decisions_path),
        "top15_decisions": _record(top15_decision_path),
        "top15_ranking": _record(top15_score_path),
    }
    event_identity = {
        "stage": "d1_read_only_reference_systems",
        "source_lock": frozen["source_lock"],
        "legacy_sources": legacy_sources,
        "output_manifests": output_manifests,
    }
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "event_id": canonical_sha256(event_identity)[:24],
            **event_identity,
            "purpose": (
                "normalize frozen three-route router and Top15 union decisions/ranks "
                "as read-only reference systems without Test outcomes"
            ),
            "candidate_labels_opened_as_table": False,
            "candidate_test_labels_read": False,
            "selection_used_test_metrics": False,
            "resumed": bool(resume),
        },
    )
    return results
