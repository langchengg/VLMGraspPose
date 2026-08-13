"""D1 P14 formal lock and exactly-once formal-Test transaction.

Prelock code may only hash the authoritative raw Test ground-truth bytes from
the immutable Snapshot-A source closure.  After an O_EXCL claim and the run
counter transition, ``run_formal_test_once`` opens those bytes once and applies
the byte-frozen canonical evaluator to the exact locked candidate universes.
No historical or corrected D1 candidate-label table is accepted.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.test_access_guard import append_access_log
from unified_reranking.evaluator_adapter import (
    evaluate_candidate_rows_with_frozen_evaluator,
)

from .contracts import assert_label_free_parquet_schema
from .run import transition_pipeline_status
from .execution import load_content_manifest
from .primary_source_adapter import (
    load_primary_plan_with_source_adapter,
    verify_records_with_primary_source_adapter,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_NAME = "FORMAL_TEST_LOCK.json"
LOCK_DIGEST_NAME = "FORMAL_TEST_LOCK.sha256"
CLAIM_SENTINEL_NAME = "FORMAL_TEST_CLAIM.sentinel"
EXECUTION_NAME = "FORMAL_TEST_EXECUTION.json"
FORMAL_BUNDLE_NAME = "formal_candidate_score_decision_bundle.parquet"
CANDIDATE_OUTCOMES_NAME = "formal_candidate_outcomes.parquet"
FORMAL_PLAN_RELATIVE_PATH = "configs/d1_formal_evaluation_plan.json"
FORMAL_INPUT_ROOT = "08_lock/formal_inputs"

UNIVERSE_ID_COLUMNS = ("source_route", "sample_id", "candidate_id")
UNIVERSE_VALUE_COLUMNS = (
    "candidate_geometry_sha256",
    "native_rank",
    "native_score",
    "cx_px",
    "cy_px",
    "theta_deg",
    "width_px",
    "height_px",
)
UNIVERSE_COLUMNS = (*UNIVERSE_ID_COLUMNS, *UNIVERSE_VALUE_COLUMNS)

REQUIRED_COMPONENTS = {
    "prelock_readiness",
    "source_run_authority",
    "paired_test_manifest",
    "canonical_evaluator",
    "statistics_config",
    "postformal_evidence",
    "d1_test_candidate_manifest",
    "d1_top5_candidates",
    "d1_top10_candidates",
    "d1_allnms_candidates",
    "d1_top5_feature_manifest",
    "d1_top10_feature_manifest",
    "d1_allnms_feature_manifest",
    "d1_primary_ranker_manifest",
    "d1_primary_gate_manifest",
    "d1_k_selection_manifest",
    "d1_top10_ranker_manifest",
    "d1_top10_gate_manifest",
    "d1_allnms_ranker_manifest",
    "d1_allnms_gate_manifest",
    "four_route_router_manifest",
    "top20_union_manifest",
}

REQUIRED_SYSTEMS: dict[str, dict[str, Any]] = {
    "d1_top5_r0": {
        "kind": "R0",
        "pool": "top5",
        "routes": ["D1"],
        "pool_component": "d1_top5_candidates",
        "inputs": {"d1_test_candidate_manifest", "d1_top5_candidates"},
    },
    "d1_top5_r7_ungated": {
        "kind": "R7_UNGATED",
        "pool": "top5",
        "routes": ["D1"],
        "pool_component": "d1_top5_candidates",
        "inputs": {
            "d1_test_candidate_manifest",
            "d1_top5_candidates",
            "d1_top5_feature_manifest",
            "d1_primary_ranker_manifest",
        },
    },
    "d1_top5_r7_gated": {
        "kind": "R7_GATED",
        "pool": "top5",
        "routes": ["D1"],
        "pool_component": "d1_top5_candidates",
        "inputs": {
            "d1_test_candidate_manifest",
            "d1_top5_candidates",
            "d1_top5_feature_manifest",
            "d1_primary_ranker_manifest",
            "d1_primary_gate_manifest",
        },
    },
    "d1_top10_locked": {
        "kind": "K_LOCKED",
        "pool": "top10",
        "routes": ["D1"],
        "pool_component": "d1_top10_candidates",
        "inputs": {
            "d1_test_candidate_manifest",
            "d1_top10_candidates",
            "d1_top10_feature_manifest",
            "d1_k_selection_manifest",
            "d1_top10_ranker_manifest",
            "d1_top10_gate_manifest",
        },
    },
    "d1_allnms_locked": {
        "kind": "K_LOCKED",
        "pool": "allnms",
        "routes": ["D1"],
        "pool_component": "d1_allnms_candidates",
        "inputs": {
            "d1_test_candidate_manifest",
            "d1_allnms_candidates",
            "d1_allnms_feature_manifest",
            "d1_k_selection_manifest",
            "d1_allnms_ranker_manifest",
            "d1_allnms_gate_manifest",
        },
    },
    "four_route_crog_default_router": {
        "kind": "FOUR_ROUTE_ROUTER",
        "pool": "four_route_top1",
        "routes": ["CROG", "G1", "C1", "D1"],
        "pool_component": None,
        "inputs": {"four_route_router_manifest"},
    },
    "top20_union": {
        "kind": "TOP20_UNION",
        "pool": "top20",
        "routes": ["CROG", "G1", "C1", "D1"],
        "pool_component": None,
        "inputs": {"top20_union_manifest"},
    },
}

REFERENCE_SYSTEMS: dict[str, dict[str, Any]] = {
    "three_route_crog_default_router_reference": {
        "kind": "READ_ONLY_REFERENCE",
        "pool": "three_route_top1",
        "routes": ["CROG", "G1", "C1"],
        "pool_component": None,
        "inputs": {"four_route_router_manifest"},
    },
    "top15_union_reference": {
        "kind": "READ_ONLY_REFERENCE",
        "pool": "top15",
        "routes": ["CROG", "G1", "C1"],
        "pool_component": None,
        "inputs": {"top20_union_manifest"},
    },
}
FORMAL_SYSTEMS: dict[str, dict[str, Any]] = {
    **REQUIRED_SYSTEMS,
    **REFERENCE_SYSTEMS,
}

FIXED_COMPONENT_PATHS = {
    "prelock_readiness": "08_lock/PRELOCK_READINESS.json",
    "paired_test_manifest": "01_manifests/d1_paired_manifest.parquet",
    "canonical_evaluator": "configs/canonical_evaluator.py",
    "statistics_config": "configs/d1_statistics.json",
    "postformal_evidence": "configs/d1_postformal_evidence.json",
    "d1_test_candidate_manifest": "02_candidates/test_manifest.json",
    "d1_top5_feature_manifest": "03_features/test/top5/T2_matched_common/manifest.json",
    "d1_top10_feature_manifest": "03_features/test/top10/T2_matched_common/manifest.json",
    "d1_allnms_feature_manifest": "03_features/test/allnms/T2_matched_common/manifest.json",
    "d1_primary_ranker_manifest": "08_lock/label_free_test_rankers/d1/manifest.json",
    "d1_primary_gate_manifest": "08_lock/label_free_test_gates/d1/manifest.json",
    "d1_k_selection_manifest": "11_k_sensitivity/selection_manifest.json",
    "d1_top10_ranker_manifest": "08_lock/label_free_test_rankers/d1_top10/manifest.json",
    "d1_top10_gate_manifest": "08_lock/label_free_test_gates/d1_top10/manifest.json",
    "d1_allnms_ranker_manifest": "08_lock/label_free_test_rankers/d1_allnms/manifest.json",
    "d1_allnms_gate_manifest": "08_lock/label_free_test_gates/d1_allnms/manifest.json",
    "four_route_router_manifest": f"{FORMAL_INPUT_ROOT}/four_route_crog_default_router/manifest.json",
    "top20_union_manifest": f"{FORMAL_INPUT_ROOT}/top20_union/manifest.json",
}

STATISTICS_CONFIG_RELATIVE_PATH = "configs/d1_statistics.json"
PRIMARY_COMPARISONS = (
    ("d1_top5_r7_gated", "d1_top5_r0"),
    ("d1_top5_r7_ungated", "d1_top5_r0"),
    ("d1_top10_locked", "d1_top5_r0"),
    ("d1_allnms_locked", "d1_top5_r0"),
    (
        "four_route_crog_default_router",
        "three_route_crog_default_router_reference",
    ),
    ("top20_union", "top15_union_reference"),
    ("top20_union", "four_route_crog_default_router"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file(path: str | Path, *, name: str) -> Path:
    result = Path(path).expanduser().resolve()
    if result.is_symlink() or not result.is_file():
        raise RuntimeError(f"{name} is not a regular file: {result}")
    return result


def _record(path: str | Path) -> dict[str, Any]:
    source = _regular_file(path, name="D1 formal artifact")
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


def _primary_source_adapter(root: Path) -> dict[str, Any]:
    _plan_path, _plan, _record_value, adapter = load_primary_plan_with_source_adapter(
        root
    )
    return adapter


def _verify_formal_records(
    root: Path,
    value: Any,
    *,
    name: str,
    require_at_least_one: bool,
) -> None:
    datasets_path = str((REPO_ROOT / "src/unified_reranking/datasets.py").resolve())
    current_datasets = _record(datasets_path)

    def contains_stale_datasets(node: Any) -> bool:
        if isinstance(node, Mapping):
            return (
                node.get("path") == datasets_path
                and isinstance(node.get("sha256"), str)
                and node.get("sha256") != current_datasets["sha256"]
            ) or any(contains_stale_datasets(child) for child in node.values())
        if isinstance(node, list):
            return any(contains_stale_datasets(child) for child in node)
        return False

    if contains_stale_datasets(value):
        adapter = _primary_source_adapter(root)
        verify_records_with_primary_source_adapter(value, adapter=adapter, name=name)
        return
    verify_artifact_records_recursive(
        value, name=name, require_at_least_one=require_at_least_one
    )


def _normalize_formal_inventory(
    root: Path, inventory: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Replace only the audited frozen datasets record with current bytes.

    Parent manifests retaining the frozen record are themselves locked.  The
    adapter proves that record's restricted semantic compatibility; the flat
    inventory must nevertheless contain hashes that ``verify_formal_lock`` can
    check against current filesystem bytes.
    """

    datasets_path = str((REPO_ROOT / "src/unified_reranking/datasets.py").resolve())
    current_datasets = _record(datasets_path)
    has_stale = any(
        record.get("path") == datasets_path
        and record.get("sha256") != current_datasets["sha256"]
        for record in inventory.values()
    )
    frozen: Mapping[str, Any] | None = None
    if has_stale:
        frozen = _primary_source_adapter(root)["frozen_source"]
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw_record in inventory.items():
        record = dict(raw_record)
        if (
            record.get("path") == datasets_path
            and record.get("sha256") != current_datasets["sha256"]
        ):
            if frozen is None or record.get("sha256") != frozen["sha256"]:
                raise RuntimeError(f"D1 formal inventory {name} datasets hash differs")
            record = current_datasets
        normalized[str(name)] = record
    return normalized


def _read_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = _regular_file(path, name=name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} is not a JSON object")
    return value


def _load_content(
    path: str | Path, *, name: str, statuses: tuple[str, ...]
) -> dict[str, Any]:
    return load_content_manifest(path, name=name, statuses=statuses)


def _exclusive_write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as error:
        raise FileExistsError(
            f"exclusive D1 formal artifact already exists: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # A created claim/lock is intentionally not removed by this helper.  Its
        # existence is fail-closed evidence of a consumed exclusive attempt.
        raise


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), sort_keys=True, indent=2) + "\n").encode("utf-8")
    _exclusive_write(path, payload)


def _content_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _repository_state() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
            )
        return result.stdout

    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD")
    return {
        "git_commit": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


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


def _walk_records(root: Path, value: Any, *, prefix: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    datasets_path = str((REPO_ROOT / "src/unified_reranking/datasets.py").resolve())
    current_datasets = _record(datasets_path)

    def visit(child: Any, location: str) -> None:
        if isinstance(child, Mapping):
            if isinstance(child.get("path"), str) and isinstance(
                child.get("sha256"), str
            ):
                try:
                    records[location] = _record(str(child["path"]))
                except RuntimeError as error:
                    raise RuntimeError(
                        f"D1 formal inventory artifact invalid at {location}"
                    ) from error
                if (
                    child.get("path") == datasets_path
                    and child.get("sha256") != current_datasets["sha256"]
                ):
                    adapter = _primary_source_adapter(root)
                    frozen = adapter["frozen_source"]
                    if child.get("sha256") != frozen["sha256"]:
                        raise RuntimeError(
                            f"D1 formal inventory {location} datasets hash differs"
                        )
                else:
                    _same_record(
                        child,
                        records[location],
                        name=f"D1 formal inventory {location}",
                    )
                return
            for key, grandchild in sorted(child.items(), key=lambda item: str(item[0])):
                visit(grandchild, f"{location}/{key}")
        elif isinstance(child, list):
            for index, grandchild in enumerate(child):
                visit(grandchild, f"{location}/{index}")

    visit(value, prefix)
    return records


def _manifest_label_free(value: Mapping[str, Any], *, name: str) -> None:
    if value.get("candidate_test_labels_read") is not False:
        raise RuntimeError(f"{name} does not declare candidate_test_labels_read=false")


def _validate_statistics_config(path: Path) -> dict[str, Any]:
    value = _read_json(path, name="D1 statistics config")
    unsigned = dict(value)
    content_sha256 = unsigned.pop("content_sha256", None)
    if content_sha256 != canonical_sha256(unsigned):
        raise RuntimeError("D1 statistics config content hash differs")
    self_unsigned = dict(unsigned)
    self_sha256 = self_unsigned.pop("self_sha256", None)
    if self_sha256 != canonical_sha256(self_unsigned):
        raise RuntimeError("D1 statistics config self hash differs")
    if (
        value.get("status") != "LOCKED"
        or value.get("immutable") is not True
        or value.get("candidate_test_labels_read") is not False
        or value.get("sample_unit") != "sample"
        or value.get("scene_clustered_bootstrap") is not True
        or int(value.get("bootstrap_iterations", -1)) != 10_000
        or not isinstance(value.get("bootstrap_seed"), int)
        or value.get("multiple_comparison_correction") != "holm"
        or value.get("paired_test") != "exact_mcnemar"
        or value.get("frame_cluster_sensitivity") is not True
        or value.get("formal_systems") != sorted(FORMAL_SYSTEMS)
        or value.get("primary_comparisons")
        != [list(comparison) for comparison in PRIMARY_COMPARISONS]
    ):
        raise RuntimeError(
            "D1 statistics config differs from the frozen sample-level contract"
        )
    return value


def assemble_statistics_config(
    run_dir: str | Path,
    *,
    bootstrap_seed: int,
    resume: bool,
) -> dict[str, Any]:
    """Write the predeclared P14 statistics contract before formal locking."""

    root = Path(run_dir).expanduser().resolve()
    if any(
        (root / relative).exists()
        for relative in (
            f"08_lock/{LOCK_NAME}",
            f"09_formal_test/{CLAIM_SENTINEL_NAME}",
            f"09_formal_test/{EXECUTION_NAME}",
        )
    ):
        raise PermissionError("D1 statistics config cannot change after lock/claim")
    if bootstrap_seed < 0:
        raise ValueError("D1 statistics bootstrap seed must be non-negative")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED",
        "immutable": True,
        "candidate_test_labels_read": False,
        "sample_unit": "sample",
        "paired_test": "exact_mcnemar",
        "scene_clustered_bootstrap": True,
        "bootstrap_iterations": 10_000,
        "bootstrap_seed": int(bootstrap_seed),
        "frame_cluster_sensitivity": True,
        "multiple_comparison_correction": "holm",
        "formal_systems": sorted(FORMAL_SYSTEMS),
        "primary_comparisons": [list(value) for value in PRIMARY_COMPARISONS],
        "selection_feedback_allowed": False,
    }
    value["self_sha256"] = canonical_sha256(value)
    value["content_sha256"] = canonical_sha256(value)
    destination = root / STATISTICS_CONFIG_RELATIVE_PATH
    if destination.exists():
        existing = _validate_statistics_config(destination)
        if not resume or existing != value:
            raise RuntimeError("D1 statistics config exists and differs")
        return existing
    atomic_json(destination, value)
    return _validate_statistics_config(destination)


def _validate_source_authority(path: Path) -> dict[str, Any]:
    value = _read_json(path, name="D1 completed source-run authority")
    if (
        value.get("status") != "COMPLETE"
        or int(value.get("formal_test_execution_count", -1)) != 1
    ):
        raise RuntimeError("D1 source-run authority is not a completed formal run")
    recorded = value.get("self_sha256")
    if recorded is not None:
        unsigned = dict(value)
        unsigned.pop("self_sha256", None)
        if recorded != canonical_sha256(unsigned):
            raise RuntimeError("D1 source-run authority self hash differs")
    return value


def _candidate_universe_from_pool(path: Path, *, name: str) -> pd.DataFrame:
    assert_label_free_parquet_schema(path, name=name)
    required = [
        "route",
        "sample_id",
        "candidate_id",
        "candidate_geometry_sha256",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ]
    frame = pd.read_parquet(path, columns=required).rename(
        columns={"route": "source_route"}
    )
    frame[list(UNIVERSE_ID_COLUMNS)] = frame[list(UNIVERSE_ID_COLUMNS)].astype(str)
    if frame.duplicated(list(UNIVERSE_ID_COLUMNS)).any():
        raise RuntimeError(f"{name} has duplicate candidate keys")
    return frame.loc[:, list(UNIVERSE_COLUMNS)]


def _same_universe_rows(
    observed: pd.DataFrame, expected: pd.DataFrame, *, name: str
) -> None:
    ordered = list(UNIVERSE_ID_COLUMNS)
    left = (
        observed.loc[:, list(UNIVERSE_COLUMNS)]
        .sort_values(ordered, kind="mergesort")
        .reset_index(drop=True)
    )
    right = (
        expected.loc[:, list(UNIVERSE_COLUMNS)]
        .sort_values(ordered, kind="mergesort")
        .reset_index(drop=True)
    )
    if len(left) != len(right):
        raise RuntimeError(f"{name} row count differs")
    for column in (*UNIVERSE_ID_COLUMNS, "candidate_geometry_sha256"):
        if not left[column].astype(str).equals(right[column].astype(str)):
            raise RuntimeError(f"{name} {column} differs")
    numeric = [
        column
        for column in UNIVERSE_VALUE_COLUMNS
        if column != "candidate_geometry_sha256"
    ]
    left_values = (
        left.loc[:, numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    )
    right_values = (
        right.loc[:, numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    )
    if not np.array_equal(left_values, right_values, equal_nan=False):
        raise RuntimeError(f"{name} rank/score/geometry differs")


def _validate_prelock_readiness(root: Path, path: Path) -> dict[str, Any]:
    value = _load_content(path, name="D1 prelock readiness", statuses=("PASS",))
    if (
        value.get("candidate_test_labels_read") is not False
        or value.get("formal_test_execution_count") != 0
        or value.get("formal_test_lock_created") is not False
    ):
        raise RuntimeError("D1 prelock readiness is not label-free/preformal")
    sources = value.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("D1 prelock readiness sources are absent")
    access_path = (root / "09_formal_test" / "test_access.log").resolve()
    declared_prefix_bytes: dict[str, int] = {}

    def collect_access_prefixes(node: Any) -> None:
        if isinstance(node, Mapping):
            if {"path", "sha256", "bytes"}.issubset(node):
                try:
                    record_path = Path(str(node["path"])).resolve()
                    byte_count = int(node["bytes"])
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
                else:
                    if record_path == access_path and byte_count >= 0:
                        digest = str(node["sha256"])
                        previous = declared_prefix_bytes.setdefault(digest, byte_count)
                        if previous != byte_count:
                            raise RuntimeError(
                                "D1 prelock access-log prefix byte declarations differ"
                            )
            for child in node.values():
                collect_access_prefixes(child)
        elif isinstance(node, list):
            for child in node:
                collect_access_prefixes(child)

    collect_access_prefixes(
        {"sources": sources, "artifacts": value.get("artifacts")}
    )

    def immutable_closure(node: Any, location: str) -> Any:
        """Remove append-only access-log leaves after proving prefix identity."""

        if isinstance(node, Mapping):
            if {"path", "sha256"}.issubset(node):
                try:
                    record_path = Path(str(node["path"])).resolve()
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    raise RuntimeError(f"{location} artifact path is invalid") from error
                if record_path == access_path:
                    expected_sha = str(node["sha256"])
                    if "bytes" in node:
                        try:
                            expected_bytes = int(node["bytes"])
                        except (TypeError, ValueError) as error:
                            raise RuntimeError(
                                f"{location} access-log byte count is invalid"
                            ) from error
                    elif expected_sha in declared_prefix_bytes:
                        expected_bytes = declared_prefix_bytes[expected_sha]
                    else:
                        raise RuntimeError(
                            f"{location} abbreviated access-log record has no "
                            "matching byte-count declaration"
                        )
                    current = access_path.read_bytes()
                    if expected_bytes < 0 or len(current) < expected_bytes:
                        raise RuntimeError(
                            f"{location} access log no longer contains its locked prefix"
                        )
                    observed_sha = hashlib.sha256(current[:expected_bytes]).hexdigest()
                    if observed_sha != expected_sha:
                        raise RuntimeError(
                            f"{location} access-log locked prefix SHA-256 mismatch"
                        )
                    return None
            return {
                key: child
                for key, value_child in node.items()
                if (child := immutable_closure(value_child, f"{location}.{key}"))
                is not None
            }
        if isinstance(node, list):
            return [
                child
                for index, value_child in enumerate(node)
                if (child := immutable_closure(value_child, f"{location}[{index}]"))
                is not None
            ]
        return node

    immutable = immutable_closure(
        {"sources": sources, "artifacts": value.get("artifacts")},
        "D1 prelock readiness",
    )
    verify_artifact_records_recursive(
        immutable,
        name="D1 prelock readiness immutable closure",
        require_at_least_one=True,
    )
    if access_path.exists():
        for line_number, line in enumerate(
            access_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"D1 Test access log line {line_number} is invalid"
                ) from error
            if (
                not isinstance(event, Mapping)
                or event.get("candidate_labels_opened_as_table") is True
                or event.get("candidate_test_labels_read") is True
                or event.get("raw_test_ground_truth_opened_as_table") is True
                or int(event.get("raw_test_ground_truth_rows_read", 0) or 0) != 0
                or event.get("event") == "d1_raw_test_ground_truth_read_once"
                or event.get("event") == "candidate_test_label_access_authorized"
                or int(event.get("candidate_label_rows_read", 0) or 0) != 0
            ):
                raise PermissionError(
                    "D1 preformal access log contains label-row access"
                )
    return value


def _source_authority_from_prelock(prelock: Mapping[str, Any]) -> Path:
    sources = prelock.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("D1 prelock sources are absent")
    direct = sources.get("source_run_authority")
    if isinstance(direct, Mapping):
        return verified_artifact_path(direct, name="D1 source-run authority")
    closure_record = sources.get("source_closure")
    if not isinstance(closure_record, Mapping):
        raise RuntimeError("D1 prelock has no source closure/authority binding")
    closure_path = verified_artifact_path(closure_record, name="D1 source closure")
    closure = _read_json(closure_path, name="D1 source closure")
    records = closure.get("verified_artifacts")
    if not isinstance(records, list):
        raise RuntimeError("D1 source closure evidence inventory is absent")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("name") == "unified_final_lock"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "D1 source closure has no unique completed unified authority"
        )
    return verified_artifact_path(matches[0], name="D1 completed unified authority")


def _source_closure_from_prelock(
    prelock: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    sources = prelock.get("sources")
    if not isinstance(sources, Mapping):
        raise RuntimeError("D1 prelock sources are absent")
    closure_record = sources.get("source_closure")
    if not isinstance(closure_record, Mapping):
        raise RuntimeError("D1 prelock has no immutable source-closure binding")
    closure_path = verified_artifact_path(
        closure_record, name="D1 immutable source closure"
    )
    closure = _load_content(
        closure_path,
        name="D1 immutable source closure",
        statuses=("PASS", "COMPLETE"),
    )
    return closure_path, closure


def _raw_test_ground_truth_from_closure(
    closure: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    canonical_inputs = closure.get("canonical_inputs")
    test_inputs = (
        canonical_inputs.get("test") if isinstance(canonical_inputs, Mapping) else None
    )
    record = (
        test_inputs.get("opaque_ground_truth")
        if isinstance(test_inputs, Mapping)
        else None
    )
    if not isinstance(record, Mapping):
        raise RuntimeError(
            "D1 source closure has no authoritative opaque Test ground-truth record"
        )
    path = verified_artifact_path(record, name="D1 raw Test ground truth")
    normalized = _record(path)
    _same_record(record, normalized, name="D1 raw Test ground truth")
    return path, normalized


def _append_hash_only_event(
    root: Path,
    *,
    ground_truth_record: Mapping[str, Any],
    plan_record: Mapping[str, Any],
) -> None:
    access_path = root / "09_formal_test" / "test_access.log"
    existing: list[dict[str, Any]] = []
    if access_path.exists():
        for line in access_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    existing.append(value)
    if any(
        event.get("event") == "d1_prelock_raw_test_ground_truth_hash_only"
        and event.get("raw_test_ground_truth_sha256")
        == ground_truth_record.get("sha256")
        and event.get("formal_plan_sha256") == plan_record.get("sha256")
        for event in existing
    ):
        return
    append_access_log(
        root,
        {
            "event": "d1_prelock_raw_test_ground_truth_hash_only",
            "raw_test_ground_truth_path": ground_truth_record.get("path"),
            "raw_test_ground_truth_sha256": ground_truth_record.get("sha256"),
            "formal_plan_path": plan_record.get("path"),
            "formal_plan_sha256": plan_record.get("sha256"),
            "raw_test_ground_truth_opened_as_table": False,
            "candidate_labels_opened_as_table": False,
            "allowed_access": "opaque_byte_hash_only",
        },
    )


def assemble_formal_evaluation_plan(
    run_dir: str | Path,
    *,
    resume: bool,
    tool_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Build the fixed P14 plan from run-local label-free producer outputs."""

    root = Path(run_dir).expanduser().resolve()
    if any(
        (root / relative).exists()
        for relative in (
            f"08_lock/{LOCK_NAME}",
            f"09_formal_test/{CLAIM_SENTINEL_NAME}",
            f"09_formal_test/{EXECUTION_NAME}",
        )
    ):
        raise PermissionError("D1 formal plan cannot change after lock/claim")
    prelock_path = root / FIXED_COMPONENT_PATHS["prelock_readiness"]
    prelock = _validate_prelock_readiness(root, prelock_path)
    source_closure_path, source_closure = _source_closure_from_prelock(prelock)
    ground_truth_path, ground_truth_record = _raw_test_ground_truth_from_closure(
        source_closure
    )
    source_authority = _source_authority_from_prelock(prelock)
    _validate_source_authority(source_authority)
    candidate_manifest_path = root / FIXED_COMPONENT_PATHS["d1_test_candidate_manifest"]
    candidate_manifest = _load_content(
        candidate_manifest_path,
        name="D1 Test candidate manifest",
        statuses=("COMPLETE",),
    )
    _manifest_label_free(candidate_manifest, name="D1 Test candidate manifest")
    candidate_artifacts = candidate_manifest.get("artifacts")
    if not isinstance(candidate_artifacts, Mapping):
        raise RuntimeError("D1 Test candidate artifacts are absent")

    components: dict[str, dict[str, Any]] = {
        "source_run_authority": _record(source_authority),
        "d1_top5_candidates": _record(
            verified_artifact_path(candidate_artifacts.get("top5", {}), name="D1 Top5")
        ),
        "d1_top10_candidates": _record(
            verified_artifact_path(
                candidate_artifacts.get("top10", {}), name="D1 Top10"
            )
        ),
        "d1_allnms_candidates": _record(
            verified_artifact_path(
                candidate_artifacts.get("allnms", {}), name="D1 AllNMS"
            )
        ),
    }
    for name, relative in FIXED_COMPONENT_PATHS.items():
        components[name] = _record(root / relative)
    if set(components) != REQUIRED_COMPONENTS:
        raise RuntimeError("D1 fixed formal component producer inventory is incomplete")

    systems: list[dict[str, Any]] = []
    for name, expected in FORMAL_SYSTEMS.items():
        manifest_path = root / FORMAL_INPUT_ROOT / name / "manifest.json"
        manifest = _load_content(
            manifest_path,
            name=f"D1 normalized formal input {name}",
            statuses=("COMPLETE",),
        )
        _manifest_label_free(manifest, name=f"D1 normalized formal input {name}")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError(
                f"D1 normalized formal input {name} artifacts are absent"
            )
        universe_path = verified_artifact_path(
            artifacts.get("candidate_universe", {}), name=f"{name} candidate universe"
        )
        score_path = verified_artifact_path(
            artifacts.get("candidate_scores", {}), name=f"{name} candidate scores"
        )
        decision_path = verified_artifact_path(
            artifacts.get("per_sample_decisions", {}), name=f"{name} decisions"
        )
        systems.append(
            {
                "name": name,
                "role": "PRIMARY" if name.startswith("d1_top5") else "SECONDARY",
                "kind": expected["kind"],
                "pool": expected["pool"],
                "routes": expected["routes"],
                "pool_component": expected["pool_component"],
                "feeds_selection": False,
                "locked_inputs": sorted(expected["inputs"]),
                "manifest": _record(manifest_path),
                "candidate_universe": _record(universe_path),
                "candidate_scores": _record(score_path),
                "per_sample_decisions": _record(decision_path),
            }
        )
    sources = {
        "components": components,
        "source_closure": _record(source_closure_path),
        "raw_test_ground_truth": ground_truth_record,
        "formal_input_manifests": {
            system["name"]: system["manifest"] for system in systems
        },
        "tools": [_record(path) for path in tool_paths],
    }
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCK_READY",
        "route": "D1",
        "candidate_test_labels_read": False,
        "raw_test_ground_truth_opened": False,
        "selection_feedback_allowed": False,
        "raw_test_ground_truth": _record(ground_truth_path),
        "ground_truth_evaluator_sha256": components["canonical_evaluator"]["sha256"],
        "components": components,
        "systems": systems,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
    }
    plan["content_sha256"] = canonical_sha256(plan)
    destination = root / FORMAL_PLAN_RELATIVE_PATH
    if destination.exists():
        existing = _load_content(
            destination, name="D1 formal evaluation plan", statuses=("LOCK_READY",)
        )
        if not resume:
            raise RuntimeError("D1 formal evaluation plan exists and differs")
        if existing != plan:
            validation_path = destination.with_name(
                f".{destination.name}.{os.getpid()}.validation"
            )
            try:
                atomic_json(validation_path, plan)
                validate_formal_evaluation_plan(root, validation_path)
            finally:
                validation_path.unlink(missing_ok=True)
            history = destination.parent / "d1_formal_evaluation_plan_history"
            archive = history / f"{existing['content_sha256']}.json"
            if archive.exists():
                archived = _load_content(
                    archive,
                    name="archived D1 formal evaluation plan",
                    statuses=("LOCK_READY",),
                )
                if archived != existing:
                    raise RuntimeError(
                        "D1 formal evaluation plan history content collision"
                    )
            else:
                atomic_json(archive, existing)
            atomic_json(destination, plan)
        validate_formal_evaluation_plan(root, destination)
        _append_hash_only_event(
            root,
            ground_truth_record=ground_truth_record,
            plan_record=_record(destination),
        )
        return plan
    validation_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.validation"
    )
    try:
        atomic_json(validation_path, plan)
        # Run the exact lock consumer before the immutable O_EXCL publication.
        validate_formal_evaluation_plan(root, validation_path)
    finally:
        validation_path.unlink(missing_ok=True)
    _exclusive_json(destination, plan)
    _append_hash_only_event(
        root,
        ground_truth_record=ground_truth_record,
        plan_record=_record(destination),
    )
    return plan


def _validate_system_artifacts(
    *,
    root: Path,
    system: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
    components: Mapping[str, Mapping[str, Any]],
    denominator_ids: list[str],
    candidate_keys: Mapping[str, pd.DataFrame],
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    name = str(system.get("name", ""))
    if not name:
        raise RuntimeError("D1 formal system name is absent")
    if system.get("feeds_selection") is not False:
        raise RuntimeError(f"D1 formal system {name} may not feed model selection")
    if expected is not None:
        for key in ("kind", "pool", "routes", "pool_component"):
            if system.get(key) != expected[key]:
                raise RuntimeError(f"D1 formal system {name} {key} differs")
        locked_inputs = system.get("locked_inputs")
        if not isinstance(locked_inputs, list) or not expected["inputs"].issubset(
            set(map(str, locked_inputs))
        ):
            raise RuntimeError(f"D1 formal system {name} locked inputs are incomplete")
    elif system.get("role") != "EXPLORATORY" or not name.startswith("exploratory_"):
        raise RuntimeError(f"undeclared non-exploratory D1 formal system: {name}")

    manifest_path = verified_artifact_path(
        system.get("manifest", {}), name=f"{name} manifest"
    )
    manifest = _load_content(
        manifest_path,
        name=f"D1 formal system {name} manifest",
        statuses=("COMPLETE", "LOCKED", "PASS"),
    )
    _manifest_label_free(manifest, name=f"D1 formal system {name} manifest")
    _verify_formal_records(
        root,
        manifest,
        name=f"D1 formal system {name}",
        require_at_least_one=True,
    )
    score_record = system.get("candidate_scores")
    decision_record = system.get("per_sample_decisions")
    universe_record = system.get("candidate_universe")
    if (
        not isinstance(score_record, Mapping)
        or not isinstance(decision_record, Mapping)
        or not isinstance(universe_record, Mapping)
    ):
        raise RuntimeError(
            f"D1 formal system {name} universe/score/decision records are absent"
        )
    score_path = verified_artifact_path(score_record, name=f"{name} candidate scores")
    decision_path = verified_artifact_path(decision_record, name=f"{name} decisions")
    universe_path = verified_artifact_path(
        universe_record, name=f"{name} candidate universe"
    )
    if (
        not _record_present(manifest, score_record)
        or not _record_present(manifest, decision_record)
        or not _record_present(manifest, universe_record)
    ):
        raise RuntimeError(
            f"D1 formal system {name} manifest does not bind universe/scores/decisions"
        )
    for path, artifact_name in (
        (universe_path, f"Test {name} candidate universe"),
        (score_path, f"Test {name} candidate scores"),
        (decision_path, f"Test {name} decisions"),
    ):
        assert_label_free_parquet_schema(path, name=artifact_name)
    universe = pd.read_parquet(universe_path, columns=list(UNIVERSE_COLUMNS))
    universe[[*UNIVERSE_ID_COLUMNS, "candidate_geometry_sha256"]] = universe[
        [*UNIVERSE_ID_COLUMNS, "candidate_geometry_sha256"]
    ].astype(str)
    universe_ranks = pd.to_numeric(universe["native_rank"], errors="coerce")
    universe_numeric = universe.loc[
        :,
        [
            "native_score",
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
        ],
    ].apply(pd.to_numeric, errors="coerce")
    if (
        universe.empty
        or universe.duplicated(list(UNIVERSE_ID_COLUMNS)).any()
        or set(universe["source_route"]).difference(map(str, system.get("routes", [])))
        or set(universe["sample_id"]).difference(denominator_ids)
        or not universe["candidate_geometry_sha256"]
        .str.fullmatch(r"[0-9a-f]{64}")
        .all()
        or not np.isfinite(universe_ranks).all()
        or not np.isfinite(universe_numeric.to_numpy(float)).all()
        or (universe_numeric[["width_px", "height_px"]].to_numpy(float) <= 0).any()
        or (universe_ranks <= 0).any()
        or not np.equal(universe_ranks, np.floor(universe_ranks)).all()
        or universe.assign(_rank=universe_ranks)
        .duplicated(["source_route", "sample_id", "_rank"])
        .any()
    ):
        raise RuntimeError(f"D1 formal system {name} candidate universe differs")
    scores = pd.read_parquet(
        score_path,
        columns=["source_route", "sample_id", "candidate_id", "score", "rank"],
    )
    required_routes = set(map(str, system.get("routes", [])))
    numeric_ranks = pd.to_numeric(scores["rank"], errors="coerce")
    if (
        scores.empty
        or scores[["source_route", "sample_id", "candidate_id"]]
        .astype(str)
        .duplicated()
        .any()
        or not set(scores["source_route"].astype(str)).issubset(required_routes)
        or not np.isfinite(pd.to_numeric(scores["score"], errors="coerce")).all()
        or not np.isfinite(numeric_ranks).all()
        or (numeric_ranks <= 0).any()
        or not np.equal(numeric_ranks, np.floor(numeric_ranks)).all()
        or scores.assign(_rank=numeric_ranks)
        .duplicated(["source_route", "sample_id", "_rank"])
        .any()
        or set(scores["sample_id"].astype(str)).difference(denominator_ids)
    ):
        raise RuntimeError(
            f"D1 formal system {name} candidate scores violate label-free contract"
        )
    if (
        len(required_routes) > 1
        and set(scores["source_route"].astype(str)) != required_routes
    ):
        raise RuntimeError(
            f"D1 formal system {name} does not cover every declared route"
        )
    if name == "four_route_crog_default_router" and (
        universe.duplicated(["sample_id", "source_route"]).any()
        or not numeric_ranks.eq(1).all()
    ):
        raise RuntimeError(
            "D1 four-route router universe is not one gated candidate per route"
        )
    if name == "top20_union" and (
        universe.groupby("sample_id").size().gt(20).any()
        or scores.assign(_rank=numeric_ranks).duplicated(["sample_id", "_rank"]).any()
        or scores.groupby("sample_id").size().gt(20).any()
    ):
        raise RuntimeError("D1 Top20 union universe/ranking exceeds its exact budget")
    universe_keys = set(
        map(
            tuple,
            universe[["source_route", "sample_id", "candidate_id"]]
            .astype(str)
            .to_numpy(),
        )
    )
    score_universe_keys = set(
        map(
            tuple,
            scores[["source_route", "sample_id", "candidate_id"]]
            .astype(str)
            .to_numpy(),
        )
    )
    if score_universe_keys != universe_keys or len(scores) != len(universe):
        raise RuntimeError(
            f"D1 formal system {name} scores do not exactly cover universe"
        )
    pool_component = system.get("pool_component")
    if isinstance(pool_component, str):
        expected_universe = candidate_keys[pool_component]
        observed = set(
            map(
                tuple,
                scores[["source_route", "sample_id", "candidate_id"]]
                .astype(str)
                .to_numpy(),
            )
        )
        expected_keys = expected_universe.loc[:, list(UNIVERSE_ID_COLUMNS)]
        required = set(map(tuple, expected_keys.astype(str).to_numpy()))
        universe_observed = set(
            map(
                tuple,
                universe[["source_route", "sample_id", "candidate_id"]]
                .astype(str)
                .to_numpy(),
            )
        )
        if (
            observed != required
            or universe_observed != required
            or len(scores) != len(expected_keys)
            or len(universe) != len(expected_keys)
        ):
            raise RuntimeError(f"D1 formal system {name} candidate membership differs")
        _same_universe_rows(
            universe,
            expected_universe,
            name=f"D1 formal system {name} exact pool universe",
        )
    decisions = pd.read_parquet(
        decision_path,
        columns=["sample_id", "selected_source_route", "selected_candidate_id"],
    )
    decisions[["sample_id", "selected_source_route", "selected_candidate_id"]] = (
        decisions[["sample_id", "selected_source_route", "selected_candidate_id"]]
        .fillna("")
        .astype(str)
    )
    if (
        decisions["sample_id"].duplicated().any()
        or set(decisions["sample_id"]) != set(denominator_ids)
        or len(decisions) != len(denominator_ids)
    ):
        raise RuntimeError(
            f"D1 formal system {name} decisions do not preserve denominator"
        )
    score_keys = set(
        map(
            tuple,
            scores[["sample_id", "source_route", "candidate_id"]]
            .astype(str)
            .to_numpy(),
        )
    )
    for row in decisions.itertuples(index=False):
        selected = (
            str(row.sample_id),
            str(row.selected_source_route),
            str(row.selected_candidate_id),
        )
        if row.selected_candidate_id and selected not in score_keys:
            raise RuntimeError(
                f"D1 formal system {name} selects a candidate outside scores"
            )
        if bool(row.selected_candidate_id) != bool(row.selected_source_route):
            raise RuntimeError(f"D1 formal system {name} has partial selected identity")
    return manifest_path, universe_path, score_path, decision_path, manifest


def validate_formal_evaluation_plan(
    run_dir: str | Path, plan_path: str | Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate all label-free lock inputs and return a flat immutable inventory."""

    root = Path(run_dir).expanduser().resolve()
    plan_source = _regular_file(plan_path, name="D1 formal evaluation plan")
    plan = _load_content(
        plan_source, name="D1 formal evaluation plan", statuses=("LOCK_READY",)
    )
    _manifest_label_free(plan, name="D1 formal evaluation plan")
    if (
        plan.get("selection_feedback_allowed") is not False
        or plan.get("raw_test_ground_truth_opened") is not False
        or "candidate_labels" in plan
        or "candidate_label_evaluator_sha256" in plan
    ):
        raise RuntimeError("D1 formal evaluation plan must forbid selection feedback")
    components = plan.get("components")
    if not isinstance(components, Mapping) or set(components) != REQUIRED_COMPONENTS:
        missing = sorted(REQUIRED_COMPONENTS.difference(components or {}))
        extra = sorted(set(components or {}).difference(REQUIRED_COMPONENTS))
        raise RuntimeError(
            f"D1 formal component inventory differs: missing={missing} extra={extra}"
        )
    normalized_components: dict[str, dict[str, Any]] = {}
    component_paths: dict[str, Path] = {}
    for name, observed in components.items():
        if not isinstance(observed, Mapping):
            raise RuntimeError(f"D1 formal component {name} record is invalid")
        path = verified_artifact_path(observed, name=f"D1 formal component {name}")
        expected_record = _record(path)
        _same_record(observed, expected_record, name=f"D1 formal component {name}")
        normalized_components[str(name)] = expected_record
        component_paths[str(name)] = path

    prelock_path = component_paths["prelock_readiness"]
    if prelock_path != (root / "08_lock" / "PRELOCK_READINESS.json").resolve():
        raise RuntimeError(
            "D1 formal plan does not bind the run-local prelock readiness"
        )
    _validate_prelock_readiness(root, prelock_path)
    _validate_source_authority(component_paths["source_run_authority"])
    _validate_statistics_config(component_paths["statistics_config"])

    candidate_manifest = _load_content(
        component_paths["d1_test_candidate_manifest"],
        name="D1 Test candidate manifest",
        statuses=("COMPLETE",),
    )
    _manifest_label_free(candidate_manifest, name="D1 Test candidate manifest")
    verify_artifact_records_recursive(
        candidate_manifest, name="D1 Test candidate manifest", require_at_least_one=True
    )
    candidate_keys: dict[str, pd.DataFrame] = {}
    for pool in ("top5", "top10", "allnms"):
        component_name = f"d1_{pool}_candidates"
        _same_record(
            candidate_manifest.get("artifacts", {}).get(pool),  # type: ignore[union-attr]
            normalized_components[component_name],
            name=f"D1 Test {pool} candidate component",
        )
        candidate_keys[component_name] = _candidate_universe_from_pool(
            component_paths[component_name], name=f"Test D1 {pool} candidates"
        )

    for name in REQUIRED_COMPONENTS.difference(
        {
            "prelock_readiness",
            "source_run_authority",
            "paired_test_manifest",
            "canonical_evaluator",
            "statistics_config",
            "d1_test_candidate_manifest",
            "d1_top5_candidates",
            "d1_top10_candidates",
            "d1_allnms_candidates",
        }
    ):
        value = _read_json(component_paths[name], name=f"D1 formal component {name}")
        if value.get("status") not in {"COMPLETE", "LOCKED", "PASS"}:
            raise RuntimeError(f"D1 formal component {name} is incomplete")
        _manifest_label_free(value, name=f"D1 formal component {name}")
        _verify_formal_records(
            root,
            value,
            name=f"D1 formal component {name}",
            require_at_least_one=True,
        )

    paired_path = component_paths["paired_test_manifest"]
    assert_label_free_parquet_schema(paired_path, name="Test paired denominator")
    paired = pd.read_parquet(paired_path, columns=["sample_id"])
    denominator_ids = paired["sample_id"].astype(str).tolist()
    if not denominator_ids or len(set(denominator_ids)) != len(denominator_ids):
        raise RuntimeError("D1 formal paired Test denominator is empty/duplicated")

    prelock = _validate_prelock_readiness(root, prelock_path)
    closure_path, closure = _source_closure_from_prelock(prelock)
    ground_truth_path, ground_truth_record = _raw_test_ground_truth_from_closure(
        closure
    )
    _same_record(
        plan.get("raw_test_ground_truth"),
        ground_truth_record,
        name="D1 raw Test ground truth",
    )
    if (
        plan.get("ground_truth_evaluator_sha256")
        != normalized_components["canonical_evaluator"]["sha256"]
    ):
        raise RuntimeError("D1 raw-GT evaluator differs from canonical evaluator")
    sources = plan.get("sources")
    if not isinstance(sources, Mapping) or plan.get(
        "source_signature_sha256"
    ) != canonical_sha256(sources):
        raise RuntimeError("D1 formal plan source signature differs")
    _same_record(
        sources.get("source_closure"),
        _record(closure_path),
        name="D1 formal plan source closure",
    )
    _same_record(
        sources.get("raw_test_ground_truth"),
        ground_truth_record,
        name="D1 formal plan source raw Test ground truth",
    )

    systems = plan.get("systems")
    if not isinstance(systems, list):
        raise RuntimeError("D1 formal plan systems are absent")
    by_name: dict[str, Mapping[str, Any]] = {}
    system_paths: dict[str, tuple[Path, Path, Path, Path, dict[str, Any]]] = {}
    for system in systems:
        if not isinstance(system, Mapping):
            raise RuntimeError("D1 formal plan system entry is invalid")
        name = str(system.get("name", ""))
        if not name or name in by_name:
            raise RuntimeError(f"D1 formal system name is empty/duplicated: {name}")
        by_name[name] = system
    if set(FORMAL_SYSTEMS) != set(by_name):
        raise RuntimeError(
            "D1 formal plan system inventory differs; "
            f"missing={sorted(set(FORMAL_SYSTEMS) - set(by_name))}, "
            f"extra={sorted(set(by_name) - set(FORMAL_SYSTEMS))}"
        )
    for name, system in by_name.items():
        system_paths[name] = _validate_system_artifacts(
            root=root,
            system=system,
            expected=FORMAL_SYSTEMS.get(name),
            components=normalized_components,
            denominator_ids=denominator_ids,
            candidate_keys=candidate_keys,
        )
    _locked_candidate_universe(by_name.values())

    inventory: dict[str, dict[str, Any]] = {
        "formal_evaluation_plan": _record(plan_source),
        "source_closure": _record(closure_path),
        "raw_test_ground_truth": _record(ground_truth_path),
        **{
            f"component/{name}": record
            for name, record in normalized_components.items()
        },
    }
    # The prelock access log is intentionally not imported transitively: it is
    # append-only transaction evidence.  PRELOCK_READINESS itself remains locked.
    for name, path in component_paths.items():
        if name == "prelock_readiness" or path.suffix.lower() != ".json":
            continue
        value = _read_json(path, name=f"D1 formal inventory component {name}")
        inventory.update(
            _walk_records(root, value, prefix=f"component_transitive/{name}")
        )
    for name, system in by_name.items():
        manifest_path, universe_path, score_path, decision_path, manifest = (
            system_paths[name]
        )
        inventory[f"system/{name}/manifest"] = _record(manifest_path)
        inventory[f"system/{name}/candidate_universe"] = _record(universe_path)
        inventory[f"system/{name}/candidate_scores"] = _record(score_path)
        inventory[f"system/{name}/per_sample_decisions"] = _record(decision_path)
        inventory.update(
            _walk_records(root, manifest, prefix=f"system_transitive/{name}")
        )
    inventory = _normalize_formal_inventory(root, inventory)
    return plan, dict(sorted(inventory.items()))


def create_formal_lock(
    run_dir: str | Path,
    *,
    evaluation_plan_path: str | Path,
    code_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Create the immutable D1 lock and detached file digest via O_EXCL."""

    root = Path(run_dir).expanduser().resolve()
    lock_path = root / "08_lock" / LOCK_NAME
    digest_path = root / "08_lock" / LOCK_DIGEST_NAME
    claim_path = root / "09_formal_test" / CLAIM_SENTINEL_NAME
    execution_path = root / "09_formal_test" / EXECUTION_NAME
    present = [
        str(path)
        for path in (lock_path, digest_path, claim_path, execution_path)
        if path.exists()
    ]
    if present:
        raise FileExistsError(
            f"D1 formal lock/claim/execution already exists: {present}"
        )
    manifest_path = _regular_file(root / "manifest.json", name="D1 run manifest")
    manifest = _read_json(manifest_path, name="D1 run manifest")
    if int(manifest.get("formal_test_execution_count", -1)) != 0:
        raise PermissionError(
            "D1 formal Test execution count must be zero before locking"
        )
    plan, inventory = validate_formal_evaluation_plan(root, evaluation_plan_path)
    for index, path in enumerate(code_paths):
        inventory[f"lock_code/{index}"] = _record(path)
    inventory = dict(sorted(inventory.items()))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED",
        "created_at_utc": _now(),
        "route": "D1",
        "formal_test_max_execution_count": 1,
        "candidate_test_labels_read": False,
        "selection_feedback_allowed": False,
        "evaluation_plan": _record(evaluation_plan_path),
        "evaluation_plan_content_sha256": plan["content_sha256"],
        "system_names": [str(system["name"]) for system in plan["systems"]],
        "repository_state": _repository_state(),
        "inventory": inventory,
        "inventory_count": len(inventory),
        "inventory_sha256": canonical_sha256(inventory),
    }
    payload["self_sha256"] = canonical_sha256(payload)
    _exclusive_json(lock_path, payload)
    lock_file_sha = sha256_file(lock_path)
    _exclusive_write(digest_path, f"{lock_file_sha}\n".encode("ascii"))
    verified = verify_formal_lock(root)
    if verified["self_sha256"] != payload["self_sha256"]:
        raise RuntimeError("D1 formal lock self verification changed")
    transition_pipeline_status(
        root,
        status="FORMAL_LOCKED",
        first_incomplete_stage="P14_FORMAL_EXECUTION",
        formal_test_executed=False,
        test_candidate_labels_read=False,
        formal_test_execution_count=0,
    )
    return verified


def verify_formal_lock(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    lock_path = _regular_file(root / "08_lock" / LOCK_NAME, name="D1 formal lock")
    digest_path = _regular_file(
        root / "08_lock" / LOCK_DIGEST_NAME, name="D1 formal lock digest"
    )
    declared_digest = digest_path.read_text(encoding="ascii").strip()
    if declared_digest != sha256_file(lock_path):
        raise PermissionError("D1 formal lock detached digest differs")
    payload = _read_json(lock_path, name="D1 formal lock")
    recorded_self = payload.get("self_sha256")
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    if payload.get("status") != "LOCKED" or recorded_self != canonical_sha256(unsigned):
        raise PermissionError("D1 formal lock status/self hash differs")
    inventory = payload.get("inventory")
    if (
        not isinstance(inventory, Mapping)
        or payload.get("inventory_count") != len(inventory)
        or payload.get("inventory_sha256") != canonical_sha256(inventory)
    ):
        raise PermissionError("D1 formal lock inventory hash/count differs")
    for name, observed in inventory.items():
        if not isinstance(observed, Mapping):
            raise PermissionError(f"D1 formal lock inventory record invalid: {name}")
        try:
            _same_record(
                observed, _record(str(observed.get("path", ""))), name=str(name)
            )
        except RuntimeError as error:
            raise PermissionError(f"D1 formal locked artifact drift: {name}") from error
    return payload


def _write_run_manifest(root: Path, value: Mapping[str, Any]) -> None:
    atomic_json(root / "manifest.json", dict(value))


def _execution_payload(root: Path, *, status: str, **extra: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "execution_count": 1,
        "formal_lock": _record(root / "08_lock" / LOCK_NAME),
        "claim_sentinel": _record(root / "09_formal_test" / CLAIM_SENTINEL_NAME),
        **extra,
    }
    return _content_payload(value)


def claim_formal_execution(run_dir: str | Path) -> dict[str, Any]:
    """Consume the one D1 formal claim before any label-table row may be opened."""

    root = Path(run_dir).expanduser().resolve()
    lock = verify_formal_lock(root)
    sentinel = root / "09_formal_test" / CLAIM_SENTINEL_NAME
    execution_path = root / "09_formal_test" / EXECUTION_NAME
    if execution_path.exists():
        raise PermissionError("D1 formal Test execution has already been claimed")
    manifest_path = _regular_file(root / "manifest.json", name="D1 run manifest")
    manifest = _read_json(manifest_path, name="D1 run manifest")
    if int(manifest.get("formal_test_execution_count", -1)) != 0:
        raise PermissionError("D1 formal Test execution count is already nonzero")
    sentinel_payload = {
        "schema_version": 1,
        "claimed_at_utc": _now(),
        "pid": os.getpid(),
        "purpose": "D1 exclusive exactly-once formal-Test claim",
        "formal_lock_self_sha256": lock["self_sha256"],
    }
    _exclusive_json(sentinel, sentinel_payload)
    claim = _execution_payload(
        root,
        status="RUNNING",
        claimed_at_utc=sentinel_payload["claimed_at_utc"],
        formal_lock_self_sha256=lock["self_sha256"],
        label_rows_opened=0,
    )
    try:
        _exclusive_json(execution_path, claim)
        manifest["formal_test_execution_count"] = 1
        manifest["status"] = "FORMAL_TEST_EXECUTION_CLAIMED"
        manifest["test_label_state"] = "FORMAL_TEST_EXECUTION_CLAIMED"
        _write_run_manifest(root, manifest)
        # Validate the complete state transition before granting row access.
        assert_formal_label_access_authorized(root)
    except Exception as error:
        failed = {key: child for key, child in claim.items() if key != "content_sha256"}
        failed.update(
            {
                "status": "FAILED",
                "failed_at_utc": _now(),
                "execution_consumed": True,
                "label_rows_opened": 0,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        # Once the sentinel exists the claim is permanently consumed.  Persist
        # that fact even when the counter transition itself was the failure.
        atomic_json(execution_path, _content_payload(failed))
        try:
            current_manifest = _read_json(
                manifest_path, name="D1 run manifest after failed claim"
            )
            current_manifest["formal_test_execution_count"] = 1
            current_manifest["status"] = "FORMAL_TEST_EXECUTION_FAILED"
            current_manifest["test_label_state"] = "FORMAL_TEST_EXECUTION_FAILED"
            _write_run_manifest(root, current_manifest)
        finally:
            append_access_log(
                root,
                {
                    "event": "d1_formal_claim_transition_failed",
                    "execution_count": 1,
                    "execution_consumed": True,
                    "candidate_labels_opened_as_table": False,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise PermissionError(
            "D1 formal claim transition failed and was consumed"
        ) from error
    append_access_log(
        root,
        {
            "event": "d1_formal_test_exclusive_claim_created",
            "execution_count": 1,
            "claim_sentinel": str(sentinel.resolve()),
            "candidate_labels_opened_as_table": False,
        },
    )
    return claim


def _load_execution(root: Path) -> dict[str, Any]:
    value = _read_json(
        root / "09_formal_test" / EXECUTION_NAME, name="D1 formal execution"
    )
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise PermissionError("D1 formal execution content hash differs")
    return value


def assert_formal_label_access_authorized(run_dir: str | Path) -> None:
    root = Path(run_dir).expanduser().resolve()
    lock = verify_formal_lock(root)
    try:
        sentinel = _read_json(
            root / "09_formal_test" / CLAIM_SENTINEL_NAME,
            name="D1 formal exclusive claim",
        )
        execution = _load_execution(root)
        manifest = _read_json(root / "manifest.json", name="D1 run manifest")
    except (RuntimeError, FileNotFoundError) as error:
        raise PermissionError(
            "D1 formal label access requires a complete exclusive claim"
        ) from error
    if (
        sentinel.get("formal_lock_self_sha256") != lock["self_sha256"]
        or execution.get("status") != "RUNNING"
        or execution.get("execution_count") != 1
        or execution.get("formal_lock_self_sha256") != lock["self_sha256"]
        or execution.get("label_rows_opened") != 0
        or int(manifest.get("formal_test_execution_count", -1)) != 1
        or manifest.get("test_label_state") != "FORMAL_TEST_EXECUTION_CLAIMED"
    ):
        raise PermissionError("D1 formal claim state does not authorize label access")


def _read_raw_test_ground_truth_once(
    root: Path, path: Path, expected_sha256: str
) -> pd.DataFrame:
    assert_formal_label_access_authorized(root)
    source = _regular_file(path, name="D1 raw Test ground truth")
    with source.open("rb") as stream:
        payload = stream.read()
    observed_sha = hashlib.sha256(payload).hexdigest()
    if observed_sha != expected_sha256:
        raise PermissionError(
            "D1 raw Test ground-truth bytes differ from the formal lock"
        )
    frame = pd.read_parquet(io.BytesIO(payload))
    _set_execution_state(
        root,
        status="EVALUATING",
        raw_ground_truth_opened_at_utc=_now(),
        raw_ground_truth_rows_opened=len(frame),
        label_rows_opened=len(frame),
        raw_test_ground_truth_sha256=observed_sha,
    )
    append_access_log(
        root,
        {
            "event": "d1_raw_test_ground_truth_read_once",
            "path": str(source),
            "sha256": observed_sha,
            "row_count": len(frame),
            "opened_after_claim": True,
        },
    )
    return frame


def _validate_raw_test_ground_truth(
    ground_truth: pd.DataFrame, denominator_ids: list[str]
) -> pd.DataFrame:
    columns = ["sample_id", "gt_grasp_rectangles"]
    missing = sorted(set(columns).difference(ground_truth.columns))
    if missing:
        raise RuntimeError(f"D1 formal raw Test ground truth misses columns: {missing}")
    result = ground_truth.loc[:, columns].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    if (
        result["sample_id"].eq("").any()
        or result["sample_id"].duplicated().any()
        or set(result["sample_id"]) != set(denominator_ids)
        or len(result) != len(denominator_ids)
    ):
        raise RuntimeError("D1 formal raw Test ground-truth denominator differs")
    return result


def _locked_candidate_universe(systems: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for system in systems:
        name = str(system["name"])
        path = verified_artifact_path(
            system["candidate_universe"], name=f"{name} candidate universe"
        )
        frame = pd.read_parquet(path, columns=list(UNIVERSE_COLUMNS))
        frame[[*UNIVERSE_ID_COLUMNS, "candidate_geometry_sha256"]] = frame[
            [*UNIVERSE_ID_COLUMNS, "candidate_geometry_sha256"]
        ].astype(str)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    keys = list(UNIVERSE_ID_COLUMNS)
    for column in UNIVERSE_VALUE_COLUMNS:
        if (
            combined.groupby(keys, dropna=False)[column]
            .nunique(dropna=False)
            .gt(1)
            .any()
        ):
            raise RuntimeError(
                f"D1 locked systems disagree on candidate-universe field {column}"
            )
    unique = combined.drop_duplicates(keys, keep="first").reset_index(drop=True)
    return unique.loc[:, list(UNIVERSE_COLUMNS)]


def _evaluate_locked_candidate_universe(
    *,
    universe: pd.DataFrame,
    ground_truth: pd.DataFrame,
    evaluator_path: Path,
    evaluator_sha256: str,
) -> pd.DataFrame:
    candidates = universe.rename(columns={"source_route": "route"}).copy()
    outcomes = evaluate_candidate_rows_with_frozen_evaluator(
        candidates,
        ground_truth,
        evaluator_path,
        evaluator_sha256,
    ).rename(columns={"route": "source_route"})
    if (
        len(outcomes) != len(universe)
        or outcomes.duplicated(list(UNIVERSE_ID_COLUMNS)).any()
        or not outcomes["candidate_success"].isin([True, False]).all()
    ):
        raise RuntimeError("D1 canonical evaluator candidate outcomes differ")
    return outcomes


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return path


def _evaluate_system(
    system: Mapping[str, Any], outcomes: pd.DataFrame, denominator_ids: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    name = str(system["name"])
    score_path = verified_artifact_path(
        system["candidate_scores"], name=f"{name} scores"
    )
    decision_path = verified_artifact_path(
        system["per_sample_decisions"], name=f"{name} decisions"
    )
    scores = (
        pd.read_parquet(score_path)
        .loc[:, ["source_route", "sample_id", "candidate_id", "score", "rank"]]
        .copy()
    )
    scores[["source_route", "sample_id", "candidate_id"]] = scores[
        ["source_route", "sample_id", "candidate_id"]
    ].astype(str)
    decisions = (
        pd.read_parquet(decision_path)
        .loc[:, ["sample_id", "selected_source_route", "selected_candidate_id"]]
        .copy()
    )
    decisions[["sample_id", "selected_source_route", "selected_candidate_id"]] = (
        decisions[["sample_id", "selected_source_route", "selected_candidate_id"]]
        .fillna("")
        .astype(str)
    )
    joined = scores.merge(
        outcomes.loc[
            :,
            [
                *UNIVERSE_ID_COLUMNS,
                "candidate_success",
                "best_same_gt_iou",
                "best_same_gt_angle_error_deg",
                "matched_gt_index",
                "jacquard_margin",
            ],
        ],
        on=["source_route", "sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["candidate_success"].isna().any():
        raise RuntimeError(
            f"D1 formal system {name} has candidates without canonical outcomes"
        )
    decision_map = decisions.set_index("sample_id")[
        ["selected_source_route", "selected_candidate_id"]
    ]
    joined = joined.join(decision_map, on="sample_id", validate="many_to_one")
    joined["is_selected"] = joined["source_route"].eq(
        joined["selected_source_route"]
    ) & joined["candidate_id"].eq(joined["selected_candidate_id"])
    joined["effective_rank"] = 0
    for sample_id, indices in joined.groupby("sample_id", sort=False).groups.items():
        if not str(decision_map.loc[sample_id, "selected_candidate_id"]):
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
    selected_counts = joined.groupby("sample_id")["is_selected"].sum()
    if (selected_counts > 1).any():
        raise RuntimeError(f"D1 formal system {name} selects more than one candidate")
    selected_success = {
        str(row.sample_id): bool(row.candidate_success)
        for row in joined.loc[joined["is_selected"]].itertuples(index=False)
    }
    declared_selected = decisions["selected_candidate_id"].ne("")
    if set(decisions.loc[declared_selected, "sample_id"]).difference(selected_success):
        raise RuntimeError(
            f"D1 formal system {name} selected identity was not realized"
        )
    joined.insert(0, "row_kind", "candidate")
    joined.insert(0, "system_pool", str(system["pool"]))
    joined.insert(0, "system_kind", str(system["kind"]))
    joined.insert(0, "system_name", name)
    joined = joined.rename(columns={"score": "formal_score"})
    joined["selected_correct"] = joined["sample_id"].map(selected_success).eq(True)
    joined["no_output"] = False
    no_output_ids = (
        decisions.loc[decisions["selected_candidate_id"].eq(""), "sample_id"]
        .astype(str)
        .tolist()
    )
    no_output = pd.DataFrame(
        {
            "system_name": name,
            "system_kind": str(system["kind"]),
            "system_pool": str(system["pool"]),
            "row_kind": "decision_no_output",
            "source_route": "",
            "sample_id": no_output_ids,
            "candidate_id": "",
            "formal_score": np.nan,
            "rank": 0,
            "effective_rank": 0,
            "candidate_success": False,
            "best_same_gt_iou": np.nan,
            "best_same_gt_angle_error_deg": np.nan,
            "matched_gt_index": pd.NA,
            "jacquard_margin": np.nan,
            "selected_source_route": "",
            "selected_candidate_id": "",
            "is_selected": False,
            "selected_correct": False,
            "no_output": True,
        }
    )
    bundle = pd.concat([joined, no_output], ignore_index=True)
    oracle_samples = set(
        joined.loc[joined["candidate_success"].astype(bool), "sample_id"]
    )
    numerator = sum(
        bool(selected_success.get(sample_id, False)) for sample_id in denominator_ids
    )
    max_k = int(joined["effective_rank"].max()) if len(joined) else 0
    first_positive_by_sample: dict[str, int] = {}
    for sample_id, rows in joined.loc[joined["effective_rank"].gt(0)].groupby(
        "sample_id", sort=False
    ):
        positives = rows.loc[rows["candidate_success"].astype(bool), "effective_rank"]
        if len(positives):
            first_positive_by_sample[str(sample_id)] = int(positives.min())
    first_positive_distribution: dict[str, int] = {}
    for sample_id in denominator_ids:
        key = str(first_positive_by_sample.get(sample_id, "no_positive"))
        first_positive_distribution[key] = first_positive_distribution.get(key, 0) + 1
    rank_metrics = []
    for k in range(1, max_k + 1):
        j_numerator = sum(rank <= k for rank in first_positive_by_sample.values())
        reciprocal = sum(
            1.0 / rank if rank <= k else 0.0
            for rank in first_positive_by_sample.values()
        )
        ndcg_total = 0.0
        for sample_id in denominator_ids:
            sample_rows = joined.loc[
                joined["sample_id"].eq(sample_id)
                & joined["effective_rank"].between(1, k)
            ].sort_values("effective_rank", kind="mergesort")
            relevance = sample_rows["candidate_success"].astype(float).to_numpy()
            dcg = float(
                sum(
                    value / np.log2(index + 2.0)
                    for index, value in enumerate(relevance)
                )
            )
            positives = int(
                joined.loc[
                    joined["sample_id"].eq(sample_id) & joined["effective_rank"].gt(0),
                    "candidate_success",
                ].sum()
            )
            ideal = float(
                sum(1.0 / np.log2(index + 2.0) for index in range(min(k, positives)))
            )
            ndcg_total += 0.0 if ideal == 0.0 else dcg / ideal
        rank_metrics.append(
            {
                "k": k,
                "j_at_k_numerator": int(j_numerator),
                "j_at_k": j_numerator / len(denominator_ids),
                "mrr_at_k": reciprocal / len(denominator_ids),
                "ndcg_at_k": ndcg_total / len(denominator_ids),
            }
        )
    metrics = {
        "sample_count": len(denominator_ids),
        "selected_correct_numerator": numerator,
        "j_at_1": numerator / len(denominator_ids),
        "oracle_numerator": len(oracle_samples),
        "oracle": len(oracle_samples) / len(denominator_ids),
        "no_output_samples": len(no_output_ids),
        "max_k": max_k,
        "rank_metric_inventory": [
            "j_at_k_numerator",
            "j_at_k",
            "mrr_at_k",
            "ndcg_at_k",
        ],
        "rank_metrics": rank_metrics,
        "first_positive_rank_distribution": first_positive_distribution,
        "selection_feedback_used": False,
    }
    if metrics["j_at_1"] > metrics["oracle"] + 1e-12:
        raise RuntimeError(f"D1 formal system {name} J@1 exceeds oracle")
    return bundle, metrics


def _set_execution_state(root: Path, *, status: str, **extra: Any) -> dict[str, Any]:
    execution = _load_execution(root)
    value = {key: child for key, child in execution.items() if key != "content_sha256"}
    value.update({"status": status, **extra})
    payload = _content_payload(value)
    atomic_json(root / "09_formal_test" / EXECUTION_NAME, payload)
    return payload


def _run_claimed_formal_test(run_dir: str | Path) -> dict[str, Any]:
    """Evaluate after the caller has consumed and validated the exclusive claim."""
    root = Path(run_dir).expanduser().resolve()
    try:
        lock = verify_formal_lock(root)
        plan_path = verified_artifact_path(
            lock["evaluation_plan"], name="D1 formal plan"
        )
        plan = _load_content(plan_path, name="D1 formal plan", statuses=("LOCK_READY",))
        components = plan["components"]
        paired_path = verified_artifact_path(
            components["paired_test_manifest"], name="D1 paired Test manifest"
        )
        denominator_ids = (
            pd.read_parquet(paired_path, columns=["sample_id"])["sample_id"]
            .astype(str)
            .tolist()
        )
        ground_truth_record = plan["raw_test_ground_truth"]
        ground_truth_path = verified_artifact_path(
            ground_truth_record, name="D1 raw Test ground truth"
        )
        ground_truth = _validate_raw_test_ground_truth(
            _read_raw_test_ground_truth_once(
                root,
                ground_truth_path,
                str(ground_truth_record["sha256"]),
            ),
            denominator_ids,
        )
        candidate_universe = _locked_candidate_universe(plan["systems"])
        evaluator_record = components["canonical_evaluator"]
        evaluator_path = verified_artifact_path(
            evaluator_record, name="D1 canonical evaluator"
        )
        outcomes = _evaluate_locked_candidate_universe(
            universe=candidate_universe,
            ground_truth=ground_truth,
            evaluator_path=evaluator_path,
            evaluator_sha256=str(evaluator_record["sha256"]),
        )
        bundles: list[pd.DataFrame] = []
        metrics: dict[str, Any] = {}
        for system in plan["systems"]:
            bundle, system_metrics = _evaluate_system(system, outcomes, denominator_ids)
            bundles.append(bundle)
            metrics[str(system["name"])] = system_metrics
        combined = pd.concat(bundles, ignore_index=True)
        expected_systems = {str(system["name"]) for system in plan["systems"]}
        if set(combined["system_name"]) != expected_systems:
            raise RuntimeError("D1 formal bundle system inventory differs")
        output = root / "09_formal_test"
        bundle_path = _atomic_parquet(output / FORMAL_BUNDLE_NAME, combined)
        outcomes_path = _atomic_parquet(output / CANDIDATE_OUTCOMES_NAME, outcomes)
        metrics_path = output / "formal_test_metrics.json"
        atomic_json(
            metrics_path,
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "formal_test_execution_count": 1,
                "systems": metrics,
                "selection_feedback_used": False,
            },
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE",
            "formal_test_execution_count": 1,
            "formal_lock": _record(root / "08_lock" / LOCK_NAME),
            "evaluation_plan": _record(plan_path),
            "raw_test_ground_truth": {
                **_record(ground_truth_path),
                "opened_once_after_claim": True,
                "row_count": len(ground_truth),
                "canonical_evaluator": _record(evaluator_path),
            },
            "candidate_outcome_count": len(outcomes),
            "sample_count": len(denominator_ids),
            "system_names": sorted(expected_systems),
            "selection_feedback_used": False,
            "selection_changes_allowed": False,
            "artifacts": {
                "candidate_score_decision_bundle": _record(bundle_path),
                "candidate_outcomes": _record(outcomes_path),
                "metrics": _record(metrics_path),
            },
        }
        result["content_sha256"] = canonical_sha256(result)
        manifest_path = output / "formal_test_manifest.json"
        atomic_json(manifest_path, result)
        _set_execution_state(
            root,
            status="COMPLETE",
            completed_at_utc=_now(),
            label_rows_opened=len(ground_truth),
            raw_ground_truth_rows_opened=len(ground_truth),
            artifacts={
                "formal_test_manifest": _record(manifest_path),
                "candidate_score_decision_bundle": _record(bundle_path),
                "candidate_outcomes": _record(outcomes_path),
                "metrics": _record(metrics_path),
            },
            selection_feedback_used=False,
        )
        run_manifest = _read_json(root / "manifest.json", name="D1 run manifest")
        run_manifest["status"] = "FORMAL_TEST_COMPLETE"
        run_manifest["test_label_state"] = "FORMAL_TEST_COMPLETE"
        _write_run_manifest(root, run_manifest)
        append_access_log(
            root,
            {
                "event": "d1_formal_test_execution_finalized",
                "execution_count": 1,
                "bundle": _record(bundle_path),
                "selection_feedback_used": False,
            },
        )
        transition_pipeline_status(
            root,
            status="FORMAL_EXECUTED",
            first_incomplete_stage="P17_INDEPENDENT_RECOMPUTE",
            formal_test_executed=True,
            test_candidate_labels_read=True,
            formal_test_execution_count=1,
        )
        return result
    except Exception as error:
        try:
            _set_execution_state(
                root,
                status="FAILED",
                failed_at_utc=_now(),
                label_rows_opened="unknown_after_claim",
                error=f"{type(error).__name__}: {error}",
                execution_consumed=True,
            )
        finally:
            append_access_log(
                root,
                {
                    "event": "d1_formal_test_execution_failed_after_claim",
                    "execution_count": 1,
                    "execution_consumed": True,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise


def run_formal_test_once(
    run_dir: str | Path,
    *,
    transaction_wrapper: Callable[[Callable[[], dict[str, Any]]], dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Claim first, then optionally wrap the already-claimed evaluation."""

    root = Path(run_dir).expanduser().resolve()
    claim_formal_execution(root)
    try:

        def evaluate() -> dict[str, Any]:
            return _run_claimed_formal_test(root)

        return (
            evaluate() if transaction_wrapper is None else transaction_wrapper(evaluate)
        )
    except Exception as error:
        try:
            execution = _load_execution(root)
            if execution.get("status") == "RUNNING":
                _set_execution_state(
                    root,
                    status="FAILED",
                    failed_at_utc=_now(),
                    label_rows_opened=0,
                    error=f"{type(error).__name__}: {error}",
                    execution_consumed=True,
                )
                append_access_log(
                    root,
                    {
                        "event": "d1_postclaim_transaction_wrapper_failed",
                        "execution_count": 1,
                        "execution_consumed": True,
                        "candidate_labels_opened_as_table": False,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
        except Exception:
            # The exclusive sentinel/execution file still prevents replay even
            # if a second filesystem fault prevents the FAILED annotation.
            pass
        raise
