"""Exact three-route predicted-mask replay closure for lifecycle P1."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import verify_artifact_records_recursive

from .audit import transition_pipeline_status
from .contracts import RunState
from .d1_predicted_replay import (
    REQUIRED_COMPARISONS,
    validate_d1_predicted_replay_manifest,
)
from .io import artifact_record, atomic_json, canonical_sha256


EXPECTED_SAMPLES = 7_675
EXPECTED_NO_OUTPUT = {"g1": 41, "c1": 13, "d1": 108}
EXPECTED_D1_CANDIDATES = 187_077
D1_COMPARISONS = REQUIRED_COMPARISONS


def _load_self_hashed(path: Path, *, name: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"{name} must be a regular file: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash differs")
    verify_artifact_records_recursive(value, name=name, require_at_least_one=True)
    return value


def _validate_dense_route(value: Mapping[str, Any], *, route: str) -> None:
    differences = value.get("maximum_absolute_differences")
    if (
        value.get("status") != "PASS"
        or value.get("route") != route
        or value.get("sample_count") != EXPECTED_SAMPLES
        or value.get("no_output_count") != EXPECTED_NO_OUTPUT[route]
        or value.get("serializer_atol") != 0.0
        or value.get("raw_test_ground_truth_rows_read", 0) != 0
        or not isinstance(differences, Mapping)
        or not differences
        or any(float(number) != 0.0 for number in differences.values())
    ):
        raise RuntimeError(f"{route} predicted replay is not an exact frozen replay")


def _validate_d1(value: Mapping[str, Any], *, path: Path) -> None:
    comparisons = value.get("comparisons")
    if (
        value.get("status") != "PASS"
        or value.get("route") != "d1"
        or value.get("branch") != "predicted"
        or value.get("sample_count") != EXPECTED_SAMPLES
        or value.get("candidate_count") != EXPECTED_D1_CANDIDATES
        or value.get("no_output_count") != EXPECTED_NO_OUTPUT["d1"]
        or value.get("raw_test_ground_truth_rows_read", 0) != 0
        or not isinstance(comparisons, Mapping)
        or set(comparisons) != D1_COMPARISONS
        or any(comparisons[name] is not True for name in D1_COMPARISONS)
    ):
        raise RuntimeError("D1 predicted replay is not an exact frozen replay")
    replayed = validate_d1_predicted_replay_manifest(path)
    if replayed != value:
        raise RuntimeError("D1 predicted replay semantic replay differs")


def close_predicted_replay(
    run_dir: str | Path,
    *,
    derived_reconciliation: str | Path,
    route_manifests: Mapping[str, str | Path],
) -> Path:
    """Bind derived metrics and exact G1/C1/D1 execution replays, then enter P1."""

    root = Path(run_dir).expanduser().resolve()
    payload = validate_predicted_replay_inputs(
        root,
        derived_reconciliation=derived_reconciliation,
        route_manifests=route_manifests,
    )
    destination = root / "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json"
    if destination.exists():
        existing = _load_self_hashed(destination, name="baseline replay closure")
        if existing != payload:
            raise RuntimeError("existing baseline replay closure differs")
    else:
        atomic_json(destination, payload)
    transition_pipeline_status(
        root,
        RunState.P1_BASELINE_REPLAY_PASS,
        first_incomplete_stage=RunState.P2_GT_MAPPING_PASS.value,
    )
    return destination


def validate_predicted_replay_inputs(
    run_dir: str | Path,
    *,
    derived_reconciliation: str | Path,
    route_manifests: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Rebuild the canonical P1 closure payload without changing lifecycle state."""

    root = Path(run_dir).expanduser().resolve()
    if set(route_manifests) != {"g1", "c1", "d1"}:
        raise ValueError("predicted replay closure requires exactly g1/c1/d1")
    derived_path = Path(derived_reconciliation).expanduser().resolve()
    if derived_path.parent != root / "04_predicted_replay":
        raise RuntimeError("derived baseline reconciliation path is not canonical")
    for route, path in route_manifests.items():
        if Path(path).expanduser().resolve() != (
            root / f"04_predicted_replay/{route}/manifest.json"
        ):
            raise RuntimeError(f"{route} predicted replay path is not canonical")
    derived = _load_self_hashed(derived_path, name="derived baseline reconciliation")
    if (
        derived.get("status") != "PASS"
        or derived.get("raw_test_ground_truth_rows_read") != 0
        or set(derived.get("routes", {})) != {"g1", "c1", "d1"}
    ):
        raise RuntimeError("derived baseline reconciliation did not PASS")
    loaded = {
        route: _load_self_hashed(Path(path), name=f"{route} predicted replay")
        for route, path in sorted(route_manifests.items())
    }
    _validate_dense_route(loaded["g1"], route="g1")
    _validate_dense_route(loaded["c1"], route="c1")
    _validate_d1(loaded["d1"], path=Path(route_manifests["d1"]))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "sample_count": EXPECTED_SAMPLES,
        "derived_reconciliation": artifact_record(derived_path),
        "route_replays": {
            route: artifact_record(Path(path))
            for route, path in sorted(route_manifests.items())
        },
        "all_three_predicted_pipelines_exact": True,
        "raw_test_ground_truth_rows_read": 0,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def validate_predicted_replay_closure(path: str | Path) -> dict[str, Any]:
    """Revalidate one canonical closure and all three live route replays."""

    source = Path(path).expanduser().resolve()
    root = source.parents[1]
    if source != root / "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json":
        raise RuntimeError("baseline replay closure path is not canonical")
    value = _load_self_hashed(source, name="baseline replay closure")
    derived = value.get("derived_reconciliation")
    routes = value.get("route_replays")
    if not isinstance(derived, Mapping) or not isinstance(routes, Mapping):
        raise RuntimeError("baseline replay closure bindings are malformed")
    rebuilt = validate_predicted_replay_inputs(
        root,
        derived_reconciliation=Path(str(derived.get("path", ""))),
        route_manifests={
            route: Path(str(record.get("path", "")))
            for route, record in routes.items()
            if isinstance(record, Mapping)
        },
    )
    if rebuilt != value:
        raise RuntimeError("baseline replay closure semantic replay differs")
    return value


__all__ = [
    "D1_COMPARISONS",
    "close_predicted_replay",
    "validate_predicted_replay_closure",
    "validate_predicted_replay_inputs",
]
