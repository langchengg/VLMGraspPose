"""Hash-bound post-execution analysis for the GT-mask counterfactual.

This module is downstream-only.  It never invokes a candidate generator,
ranker, gate, or model-selection routine.  In particular, no ground-truth
frame is opened until the canonical protocol lock and exactly-once execution
claim have both been verified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .audit import transition_pipeline_status
from .candidate_matching import match_candidate_pools
from .contracts import RunState
from .independent import independent_recompute_from_frames
from .io import (
    artifact_record,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    atomic_text,
    canonical_sha256,
    exclusive_json,
    sha256_file,
)
from .metrics import (
    compare_branch_outcomes,
    compute_branch_metrics,
    evaluate_candidate_rows,
    summarize_sample_outcomes,
)
from .protocol import (
    EXECUTION_RELATIVE_PATH,
    LOCK_RELATIVE_PATH,
    complete_bulk_execution,
    load_execution_authority,
    verify_protocol_lock,
)
from .reporting import TABLE_CONTRACTS, load_bound_tables, write_table_bundle
from .statistics import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    apply_holm_family,
    paired_metric_statistics,
)
from .taxonomy import (
    NATIVE_CLASSES,
    POST_R7_CLASSES,
    add_secondary_flags,
    classify_native_taxonomy,
    classify_post_r7_taxonomy,
    taxonomy_counts,
    taxonomy_definitions,
)
from .visual_assets import validate_visual_asset_registry


ROUTES = ("G1", "C1", "D1")
BRANCHES = ("predicted", "gt_oracle")
EXPECTED_SAMPLE_COUNT = 7_675
INPUT_MANIFEST_RELATIVE_PATH = Path("07_candidate_tables/POSTPROCESS_INPUTS.json")
OUTPUT_MANIFEST_RELATIVE_PATH = Path("08_metrics/POSTPROCESS_MANIFEST.json")
ROUTE_STATUS_RELATIVE_PATH = Path("08_metrics/ROUTE_STATUS.json")
SCIENTIFIC_ROLE = "post-formal oracle stage-replacement diagnostic"
FINAL_OUTCOMES_AUTHORITY_RELATIVE_PATH = Path(
    "04_predicted_replay/FINAL_OUTCOMES_AUTHORITY.json"
)
D1_DEPTH_DECISIONS_RELATIVE_PATH = Path(
    "04_predicted_replay/d1_locked_depth_selector_decisions.parquet"
)

_REQUIRED_COVARIATES = (
    "sample_id",
    "query_type",
    "predicted_mask_iou",
    "target_area_fraction",
    "mask_component_count",
    "mask_boundary_complexity",
    "valid_depth_ratio",
    "scene_family",
    "frame_family",
)
_FORBIDDEN_RAW = re.compile(
    r"(^|_)(gt_mask|ground_truth_mask|candidate_success|correctness_label|"
    r"oracle_label|soft_target|training_target)($|_)",
    flags=re.IGNORECASE,
)


class PostprocessContractError(RuntimeError):
    """Saved execution artifacts differ from the locked analysis contract."""


def _assert_run_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    if not root.name.startswith("fair_gtmask_counterfactual_g1_c1_d1_"):
        raise PermissionError("postprocess output is outside the isolated GT-mask run")
    if root.parent.name != "runs":
        raise PermissionError("counterfactual run must be a direct child of runs/")
    return root


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise PostprocessContractError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostprocessContractError(f"cannot parse {label}: {source}") from error
    if not isinstance(value, dict):
        raise PostprocessContractError(f"{label} must contain one JSON object")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise PostprocessContractError(f"{label} content hash differs")


def _safe_record(
    record: Mapping[str, Any],
    *,
    root: Path,
    label: str,
    allowed_prefixes: Sequence[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise PostprocessContractError(f"{label} is not an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{label} escapes the counterfactual run") from error
    relative_text = relative.as_posix()
    if allowed_prefixes is not None:
        if not any(
            relative_text == prefix.rstrip("/")
            or relative_text.startswith(prefix.rstrip("/") + "/")
            for prefix in allowed_prefixes
        ):
            raise PermissionError(
                f"{label} is outside its exact upstream namespace: {relative_text}"
            )
    elif not relative.parts or relative.parts[0] in {
        "08_metrics",
        "09_failure_taxonomy",
        "10_statistics",
        "11_stratified_analysis",
        "12_case_selection",
        "13_figures",
        "14_galleries",
        "15_reports",
        "16_independent_recompute",
        "tables",
    }:
        raise PermissionError(f"{label} is in a downstream/feedback namespace")
    if path.is_symlink() or not path.is_file():
        raise PostprocessContractError(f"{label} is absent or unsafe: {path}")
    observed = artifact_record(path)
    if (
        record.get("sha256") != observed["sha256"]
        or int(record.get("bytes", -1)) != observed["bytes"]
    ):
        raise PostprocessContractError(f"{label} hash/byte record differs")
    return path, observed


def _collect_records(
    value: Any, *, prefix: str = "root"
) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            if not {"path", "sha256"}.issubset(value):
                raise PostprocessContractError(
                    f"incomplete artifact record at {prefix}"
                )
            records.append((prefix, dict(value)))
        elif value.get("kind") != "inline":
            for key, child in value.items():
                records.extend(_collect_records(child, prefix=f"{prefix}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            records.extend(_collect_records(child, prefix=f"{prefix}[{index}]"))
    return records


def write_postprocess_input_manifest(
    run_dir: str | Path,
    *,
    artifacts: Mapping[str, Any],
    protocol_lock: str | Path,
    sample_count: int = EXPECTED_SAMPLE_COUNT,
) -> Path:
    """Freeze canonical saved-frame inputs without opening scientific rows."""

    root = _assert_run_dir(run_dir)
    required = {
        "sample_manifest",
        "ground_truth",
        "sample_covariates",
        "final_outcomes",
        "final_outcomes_authority",
        "baseline_replay",
        "visual_assets",
        "candidates",
        "per_sample",
        "route_manifests",
    }
    missing = sorted(required.difference(artifacts))
    if missing:
        raise ValueError(f"postprocess input artifacts are incomplete: {missing}")
    candidates_value = artifacts.get("candidates")
    if not isinstance(candidates_value, Mapping):
        raise ValueError("postprocess candidates declaration must be an object")
    candidate_keys = set(candidates_value)
    available_routes = tuple(
        route
        for route in ROUTES
        if {f"{route}|{branch}" for branch in BRANCHES}.issubset(candidate_keys)
    )
    if available_routes not in {ROUTES, ("G1", "C1")}:
        raise ValueError("available routes must be G1/C1 or the complete G1/C1/D1 set")
    expected_keys = {
        f"{route}|{branch}" for route in available_routes for branch in BRANCHES
    }
    for group in ("candidates", "per_sample", "route_manifests"):
        value = artifacts[group]
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError(f"{group} route/branch records differ from availability")
    if available_routes == ("G1", "C1") and "d1_blocker" not in artifacts:
        raise ValueError("a D1-omitted input manifest requires a hash-bound blocker")
    if available_routes == ROUTES and "d1_blocker" in artifacts:
        raise ValueError(
            "a complete three-route manifest cannot also declare a D1 blocker"
        )
    lock_path = Path(protocol_lock).expanduser().resolve()
    if lock_path != root / LOCK_RELATIVE_PATH:
        raise PermissionError("postprocess protocol lock is outside the run")
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED",
        "scientific_role": SCIENTIFIC_ROLE,
        "sample_count": int(sample_count),
        "routes": list(ROUTES),
        "available_routes": list(available_routes),
        "branches": list(BRANCHES),
        "training_or_selection_feedback_allowed": False,
        "protocol_lock": artifact_record(lock_path),
        "artifacts": dict(artifacts),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    destination = root / INPUT_MANIFEST_RELATIVE_PATH
    if destination.exists():
        existing = _json_object(destination, label="postprocess input manifest")
        if existing != payload:
            raise FileExistsError("existing postprocess input manifest differs")
        return destination
    return exclusive_json(destination, payload)


def write_route_frame_manifest(
    run_dir: str | Path,
    *,
    route: str,
    branch: str,
    candidates: Mapping[str, Any],
    per_sample: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    sample_count: int = EXPECTED_SAMPLE_COUNT,
) -> Path:
    """Close one normalized candidate/per-sample frame pair by exact hashes."""

    root = _assert_run_dir(run_dir)
    route_name, branch_name = str(route).upper(), str(branch).lower()
    if route_name not in ROUTES or branch_name not in BRANCHES:
        raise ValueError("route frame identity is unsupported")
    expected_root = root / "07_candidate_tables/raw" / route_name.lower() / branch_name
    candidate_path, candidate_record = _safe_record(
        candidates,
        root=root,
        label="route candidates",
        allowed_prefixes=(str(expected_root.relative_to(root)),),
    )
    sample_path, sample_record = _safe_record(
        per_sample,
        root=root,
        label="route per-sample",
        allowed_prefixes=(str(expected_root.relative_to(root)),),
    )
    if (
        candidate_path.name not in {"candidates.parquet", "per_candidate.parquet"}
        or sample_path.name != "per_sample.parquet"
    ):
        raise PostprocessContractError("normalized route frame basenames differ")
    _assert_no_raw_supervision(candidate_path, label="route candidates")
    _assert_no_raw_supervision(sample_path, label="route per-sample")
    candidate_frame = _read_parquet_columns(
        candidate_path,
        columns=["sample_id", "route", "branch", "candidate_id", "native_rank"],
        label="route candidates",
    )
    sample_frame = _read_parquet_columns(
        sample_path,
        columns=["sample_id", "route", "branch", "candidate_count", "no_output"],
        label="route per-sample",
    )
    for name, frame in (("candidate", candidate_frame), ("per-sample", sample_frame)):
        if (
            not frame["route"].astype(str).str.upper().eq(route_name).all()
            or not frame["branch"].astype(str).str.lower().eq(branch_name).all()
        ):
            raise PostprocessContractError(f"{name} route/branch identity differs")
    sample_frame["sample_id"] = sample_frame["sample_id"].astype(str)
    candidate_frame["sample_id"] = candidate_frame["sample_id"].astype(str)
    if (
        len(sample_frame) != int(sample_count)
        or sample_frame["sample_id"].duplicated().any()
        or candidate_frame.duplicated(["sample_id", "candidate_id"]).any()
    ):
        raise PostprocessContractError("normalized route frame identity/count differs")
    counts = candidate_frame.groupby("sample_id").size().reindex(
        sample_frame["sample_id"], fill_value=0
    )
    if not np.array_equal(
        counts.to_numpy(int),
        pd.to_numeric(sample_frame["candidate_count"], errors="raise").to_numpy(int),
    ) or not np.array_equal(
        counts.eq(0).to_numpy(), sample_frame["no_output"].astype(bool).to_numpy()
    ):
        raise PostprocessContractError("normalized route sample counts/no-output differ")
    if branch_name == "predicted":
        expected_keys = {"baseline_replay"}
    else:
        expected_keys = {"protocol_lock", "execution_claim"}
    if set(source_contract) != expected_keys:
        raise PostprocessContractError("route frame source-contract keys differ")
    for label, record in source_contract.items():
        _verify_external_record(record, label=f"route source contract {label}")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "route": route_name,
        "branch": branch_name,
        "sample_count": int(sample_count),
        "candidate_count": len(candidate_frame),
        "candidates": candidate_record,
        "per_sample": sample_record,
        "source_contract": dict(source_contract),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    destination = expected_root / "manifest.json"
    if destination.exists():
        existing = _json_object(destination, label="route frame manifest")
        if existing != payload:
            raise FileExistsError("existing route frame manifest differs")
        return destination
    return exclusive_json(destination, payload)


def _load_input_manifest(
    root: Path, input_manifest: str | Path | None
) -> tuple[Path, dict[str, Any]]:
    path = (
        root / INPUT_MANIFEST_RELATIVE_PATH
        if input_manifest is None
        else Path(input_manifest).expanduser().resolve()
    )
    if path != root / INPUT_MANIFEST_RELATIVE_PATH:
        raise PermissionError("postprocess input manifest must use its canonical path")
    value = _json_object(path, label="postprocess input manifest")
    _verify_self_hash(value, label="postprocess input manifest")
    if (
        value.get("status") != "LOCKED"
        or value.get("scientific_role") != SCIENTIFIC_ROLE
        or value.get("training_or_selection_feedback_allowed") is not False
        or value.get("routes") != list(ROUTES)
        or value.get("branches") != list(BRANCHES)
    ):
        raise PostprocessContractError("postprocess input declaration differs")
    available = tuple(value.get("available_routes", ()))
    if available not in {ROUTES, ("G1", "C1")}:
        raise PostprocessContractError(
            "postprocess available-route declaration differs"
        )
    return path, value


def _verify_execution_authority(
    root: Path,
    *,
    protocol_lock: str | Path,
    input_value: Mapping[str, Any],
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """This guard must run before any frame or GT registry is opened."""

    lock_path = Path(protocol_lock).expanduser().resolve()
    if lock_path != root / LOCK_RELATIVE_PATH:
        raise PermissionError("analysis protocol lock is outside the canonical run")
    declared_lock = input_value.get("protocol_lock")
    if not isinstance(declared_lock, Mapping) or declared_lock != artifact_record(
        lock_path
    ):
        raise PostprocessContractError("input manifest protocol binding differs")
    authority = load_execution_authority(lock_path)
    artifacts = input_value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PostprocessContractError("postprocess artifacts declaration is malformed")
    if artifacts.get("sample_manifest") != authority.get("sample_manifest"):
        raise PermissionError(
            "analysis sample manifest differs from protocol authority"
        )
    if artifacts.get("ground_truth") != authority.get("sample_manifest"):
        raise PermissionError(
            "GT grasp frame must be the protocol-bound counterfactual manifest"
        )
    claim = _json_object(root / EXECUTION_RELATIVE_PATH, label="execution claim")
    pipeline = _json_object(root / "pipeline_status.json", label="pipeline status")
    available = tuple(input_value.get("available_routes", ()))
    allowed_status = {
        RunState.P6_D1_COUNTERFACTUAL_COMPLETE.value,
        RunState.P7_TAXONOMY_COMPLETE.value,
        RunState.P8_STATISTICS_COMPLETE.value,
    }
    if available == ("G1", "C1"):
        if "d1_blocker" not in input_value.get("artifacts", {}):
            raise PermissionError("P5 partial analysis lacks a D1 blocker")
        allowed_status = {RunState.P5_C1_COUNTERFACTUAL_COMPLETE.value}
    elif not resume:
        allowed_status = {RunState.P6_D1_COUNTERFACTUAL_COMPLETE.value}
    if (
        claim.get("status") != "RUNNING"
        or int(claim.get("execution_count", -1)) != 1
        or claim.get("protocol_lock_file_sha256") != sha256_file(lock_path)
        or int(pipeline.get("counterfactual_execution_count", -1)) != 1
        or pipeline.get("status") not in allowed_status
    ):
        raise PermissionError(
            "postprocess requires completed route execution authority"
        )
    return authority, pipeline


def _verify_inputs_after_authority(
    root: Path, input_value: Mapping[str, Any], *, expected_sample_count: int
) -> dict[str, Path]:
    if int(input_value.get("sample_count", -1)) != int(expected_sample_count):
        raise PostprocessContractError(
            "input denominator differs from the locked analysis"
        )
    artifacts = input_value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PostprocessContractError("postprocess artifacts declaration is malformed")
    available = tuple(input_value.get("available_routes", ()))
    paths: dict[str, Path] = {}
    top_level = {
        "sample_manifest": ("02_sample_manifest",),
        "ground_truth": ("02_sample_manifest",),
        "sample_covariates": ("02_sample_manifest", "03_gt_mask_registry"),
        "final_outcomes": ("04_predicted_replay",),
        "final_outcomes_authority": ("04_predicted_replay",),
        "baseline_replay": ("04_predicted_replay",),
        "visual_assets": ("04_predicted_replay",),
    }
    for key, prefixes in top_level.items():
        path, _ = _safe_record(
            artifacts[key],
            root=root,
            label=f"artifacts.{key}",
            allowed_prefixes=prefixes,
        )
        paths[f"artifacts.{key}"] = path
    validate_visual_asset_registry(
        root,
        artifacts["visual_assets"],
        expected_count=expected_sample_count,
    )
    for route in available:
        for branch in BRANCHES:
            key = f"{route}|{branch}"
            prefixes = (
                ("04_predicted_replay", "07_candidate_tables/raw")
                if branch == "predicted"
                else ("06_gtmask_predictions", "07_candidate_tables/raw")
            )
            for group in ("candidates", "per_sample", "route_manifests"):
                path, _ = _safe_record(
                    artifacts[group][key],
                    root=root,
                    label=f"artifacts.{group}.{key}",
                    allowed_prefixes=prefixes,
                )
                if route.lower() not in path.parts or branch not in path.parts:
                    raise PermissionError(
                        f"artifacts.{group}.{key} path identity differs"
                    )
                paths[f"artifacts.{group}.{key}"] = path
    if "d1_blocker" in artifacts:
        path, _ = _safe_record(
            artifacts["d1_blocker"],
            root=root,
            label="artifacts.d1_blocker",
            allowed_prefixes=("00_audit/machine_blockers",),
        )
        paths["artifacts.d1_blocker"] = path
    if "frozen_selector_transfer" in artifacts:
        path, _ = _safe_record(
            artifacts["frozen_selector_transfer"],
            root=root,
            label="artifacts.frozen_selector_transfer",
            allowed_prefixes=("07_candidate_tables/frozen_selector_transfer",),
        )
        paths["artifacts.frozen_selector_transfer"] = path
    return paths


def _schema(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except (ImportError, OSError, ValueError) as error:
        raise PostprocessContractError(
            f"cannot inspect Parquet schema: {path}"
        ) from error


def _read_parquet_columns(
    path: Path, *, columns: Sequence[str], label: str
) -> pd.DataFrame:
    names = set(_schema(path))
    missing = sorted(set(columns).difference(names))
    if missing:
        raise PostprocessContractError(f"{label} misses columns: {missing}")
    try:
        return pd.read_parquet(path, columns=list(columns))
    except (OSError, ValueError, TypeError) as error:
        raise PostprocessContractError(f"cannot read {label}: {path}") from error


def _assert_no_raw_supervision(path: Path, *, label: str) -> None:
    forbidden = sorted(
        column for column in _schema(path) if _FORBIDDEN_RAW.search(column)
    )
    if forbidden:
        raise PermissionError(
            f"{label} contains forbidden raw supervision: {forbidden}"
        )


def _normalize_route_branch(
    frame: pd.DataFrame, *, route: str, branch: str, label: str
) -> pd.DataFrame:
    result = frame.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    observed_route = result["route"].astype(str).str.upper()
    observed_branch = result["branch"].astype(str).str.lower()
    if not observed_route.eq(route).all() or not observed_branch.eq(branch).all():
        raise PostprocessContractError(f"{label} route/branch identity differs")
    result["route"] = route
    result["branch"] = branch
    return result


def _load_candidate_frames(
    artifacts: Mapping[str, Any],
    root: Path,
    denominator: set[str],
    *,
    routes: Sequence[str],
) -> tuple[
    pd.DataFrame, dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], set[str]]
]:
    candidates: list[pd.DataFrame] = []
    saved_samples: dict[tuple[str, str], pd.DataFrame] = {}
    technical: dict[tuple[str, str], set[str]] = {}
    for route in routes:
        for branch in BRANCHES:
            key = f"{route}|{branch}"
            prefixes = (
                ("04_predicted_replay", "07_candidate_tables/raw")
                if branch == "predicted"
                else ("06_gtmask_predictions", "07_candidate_tables/raw")
            )
            candidate_path, _ = _safe_record(
                artifacts["candidates"][key],
                root=root,
                label=f"candidates.{key}",
                allowed_prefixes=prefixes,
            )
            sample_path, _ = _safe_record(
                artifacts["per_sample"][key],
                root=root,
                label=f"per_sample.{key}",
                allowed_prefixes=prefixes,
            )
            route_manifest_path, _ = _safe_record(
                artifacts["route_manifests"][key],
                root=root,
                label=f"route_manifests.{key}",
                allowed_prefixes=prefixes,
            )
            route_manifest = _json_object(
                route_manifest_path, label=f"route manifest {key}"
            )
            _verify_self_hash(route_manifest, label=f"route manifest {key}")
            if (
                candidate_path.name
                not in {"candidates.parquet", "per_candidate.parquet"}
                or sample_path.name != "per_sample.parquet"
                or route_manifest_path.name != "manifest.json"
            ):
                raise PostprocessContractError(
                    f"canonical route artifact basename differs: {key}"
                )
            expected_provenance = (
                {"baseline_replay": dict(artifacts["baseline_replay"])}
                if branch == "predicted"
                else {
                    "protocol_lock": artifact_record(root / LOCK_RELATIVE_PATH),
                    "execution_claim": artifact_record(root / EXECUTION_RELATIVE_PATH),
                }
            )
            if (
                route_manifest.get("status") != "COMPLETE"
                or str(route_manifest.get("route", "")).upper() != route
                or str(route_manifest.get("branch", "")).lower() != branch
                or int(route_manifest.get("sample_count", -1)) != len(denominator)
                or route_manifest.get("candidates")
                != dict(artifacts["candidates"][key])
                or route_manifest.get("per_sample")
                != dict(artifacts["per_sample"][key])
                or route_manifest.get("source_contract") != expected_provenance
            ):
                raise PostprocessContractError(
                    f"canonical route manifest differs: {key}"
                )
            _assert_no_raw_supervision(candidate_path, label=f"candidates.{key}")
            _assert_no_raw_supervision(sample_path, label=f"per_sample.{key}")
            candidate_columns = [
                "sample_id",
                "route",
                "branch",
                "candidate_id",
                "native_rank",
                "cx_px",
                "cy_px",
                "theta_deg",
            ]
            candidate_schema = set(_schema(candidate_path))
            if "native_score" in candidate_schema:
                candidate_columns.append("native_score")
            width = "jaw_width_px" if "jaw_width_px" in candidate_schema else "width_px"
            height = (
                "rectangle_height_px"
                if "rectangle_height_px" in candidate_schema
                else "height_px"
            )
            candidate_columns.extend([width, height])
            frame = _read_parquet_columns(
                candidate_path, columns=candidate_columns, label=f"candidates.{key}"
            )
            frame = _normalize_route_branch(
                frame, route=route, branch=branch, label=f"candidates.{key}"
            )
            unknown = sorted(set(frame["sample_id"]).difference(denominator))
            if unknown:
                raise PostprocessContractError(
                    f"candidates.{key} escapes denominator: {unknown[:5]}"
                )
            candidates.append(frame)
            if int(route_manifest.get("candidate_count", -1)) != len(frame):
                raise PostprocessContractError(
                    f"canonical route manifest candidate count differs: {key}"
                )

            sample_columns = [
                "sample_id",
                "route",
                "branch",
                "candidate_count",
                "no_output",
            ]
            sample_schema = set(_schema(sample_path))
            for optional in ("technical_failure", "status"):
                if optional in sample_schema:
                    sample_columns.append(optional)
            sample = _read_parquet_columns(
                sample_path, columns=sample_columns, label=f"per_sample.{key}"
            )
            sample = _normalize_route_branch(
                sample, route=route, branch=branch, label=f"per_sample.{key}"
            )
            if (
                sample["sample_id"].duplicated().any()
                or set(sample["sample_id"]) != denominator
            ):
                raise PostprocessContractError(f"per_sample.{key} lost the denominator")
            counts = (
                frame.groupby("sample_id")
                .size()
                .reindex(sample["sample_id"], fill_value=0)
            )
            recorded = pd.to_numeric(sample["candidate_count"], errors="coerce")
            if recorded.isna().any() or not np.array_equal(
                recorded.to_numpy(dtype=int), counts.to_numpy(dtype=int)
            ):
                raise PostprocessContractError(
                    f"per_sample.{key} candidate counts differ"
                )
            no_output = sample["no_output"].astype(bool).to_numpy()
            if not np.array_equal(no_output, counts.eq(0).to_numpy()):
                raise PostprocessContractError(
                    f"per_sample.{key} no-output flags differ"
                )
            failed_ids: set[str] = set()
            if "technical_failure" in sample:
                failed_ids.update(
                    sample.loc[sample["technical_failure"].astype(bool), "sample_id"]
                )
            if "status" in sample:
                allowed = {"PASS", "COMPLETE", "NO_OUTPUT", "TECHNICAL_FAILURE"}
                status = sample["status"].astype(str).str.upper()
                if not set(status).issubset(allowed):
                    raise PostprocessContractError(
                        f"per_sample.{key} status is unknown"
                    )
                failed_ids.update(
                    sample.loc[status.eq("TECHNICAL_FAILURE"), "sample_id"]
                )
            saved_samples[(route, branch)] = sample
            technical[(route, branch)] = failed_ids
    combined = pd.concat(candidates, ignore_index=True)
    if combined.duplicated(["sample_id", "route", "branch", "candidate_id"]).any():
        raise PostprocessContractError("candidate identities are duplicated")
    return combined, saved_samples, technical


def _normalize_gt(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    values: list[Any] = []

    def plain(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return [plain(item) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    for value in result["gt_grasp_rectangles"]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise PostprocessContractError("GT grasp JSON is malformed") from error
        value = plain(value)
        if not isinstance(value, (list, tuple)):
            raise PostprocessContractError("GT grasp rectangles are not a sequence")
        values.append(value)
    result["gt_grasp_rectangles"] = values
    return result


def _evaluator_record(lock: Mapping[str, Any]) -> dict[str, Any]:
    bindings = lock.get("bindings")
    evaluator = bindings.get("evaluator") if isinstance(bindings, Mapping) else None
    records = _collect_records(evaluator, prefix="bindings.evaluator")
    if len(records) != 1:
        raise PostprocessContractError("protocol must bind exactly one evaluator")
    return records[0][1]


def _validate_binary(frame: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise PostprocessContractError(f"{label}.{column} must be finite binary")
    return values.astype(bool)


def _load_analysis_frames(
    root: Path,
    input_value: Mapping[str, Any],
    *,
    expected_sample_count: int,
) -> dict[str, Any]:
    artifacts = input_value["artifacts"]
    routes = tuple(input_value.get("available_routes", ()))
    manifest_path, _ = _safe_record(
        artifacts["sample_manifest"],
        root=root,
        label="sample_manifest",
        allowed_prefixes=("02_sample_manifest",),
    )
    manifest = _read_parquet_columns(
        manifest_path,
        columns=["sample_id", "scene_id", "frame_id"],
        label="sample manifest",
    )
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    if (
        len(manifest) != expected_sample_count
        or manifest["sample_id"].eq("").any()
        or manifest["sample_id"].duplicated().any()
    ):
        raise PostprocessContractError("sample manifest denominator differs")
    denominator = set(manifest["sample_id"])

    covariate_path, _ = _safe_record(
        artifacts["sample_covariates"],
        root=root,
        label="sample_covariates",
        allowed_prefixes=("02_sample_manifest", "03_gt_mask_registry"),
    )
    covariates = _read_parquet_columns(
        covariate_path, columns=_REQUIRED_COVARIATES, label="sample covariates"
    )
    covariates["sample_id"] = covariates["sample_id"].astype(str)
    if (
        covariates["sample_id"].duplicated().any()
        or set(covariates["sample_id"]) != denominator
    ):
        raise PostprocessContractError("sample covariates lost the denominator")
    for column in (
        "query_type",
        "scene_family",
        "frame_family",
    ):
        if (
            covariates[column].isna().any()
            or covariates[column].astype(str).str.strip().eq("").any()
        ):
            raise PostprocessContractError(f"sample covariate {column} is missing")
    for column in (
        "predicted_mask_iou",
        "target_area_fraction",
        "mask_component_count",
        "mask_boundary_complexity",
        "valid_depth_ratio",
    ):
        values = pd.to_numeric(covariates[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise PostprocessContractError(f"sample covariate {column} is non-finite")

    gt_path, _ = _safe_record(
        artifacts["ground_truth"],
        root=root,
        label="ground_truth",
        allowed_prefixes=("02_sample_manifest",),
    )
    gt_schema = set(_schema(gt_path))
    grasp_column = (
        "gt_grasp_rectangles"
        if "gt_grasp_rectangles" in gt_schema
        else "gt_grasp_list_json"
    )
    ground_truth = _read_parquet_columns(
        gt_path,
        columns=["sample_id", grasp_column],
        label="ground-truth grasp frame",
    ).rename(columns={grasp_column: "gt_grasp_rectangles"})
    ground_truth = _normalize_gt(ground_truth)
    if (
        ground_truth["sample_id"].duplicated().any()
        or set(ground_truth["sample_id"]) != denominator
    ):
        raise PostprocessContractError("ground-truth grasp frame lost the denominator")

    candidates, saved_samples, technical = _load_candidate_frames(
        artifacts, root, denominator, routes=routes
    )
    final_path, _ = _safe_record(
        artifacts["final_outcomes"],
        root=root,
        label="final_outcomes",
        allowed_prefixes=("04_predicted_replay",),
    )
    final = _read_parquet_columns(
        final_path,
        columns=["sample_id", "route", "final_correct"],
        label="frozen final outcomes",
    )
    final["sample_id"] = final["sample_id"].astype(str)
    final["route"] = final["route"].astype(str).str.upper()
    final["final_correct"] = _validate_binary(
        final, "final_correct", label="final outcomes"
    )
    if (
        final.duplicated(["sample_id", "route"]).any()
        or len(final) != expected_sample_count * len(routes)
        or set(final["sample_id"]) != denominator
        or set(final["route"]) != set(routes)
    ):
        raise PostprocessContractError("frozen final outcomes are not route-aligned")
    return {
        "manifest": manifest,
        "covariates": covariates,
        "ground_truth": ground_truth,
        "candidates": candidates,
        "saved_samples": saved_samples,
        "technical": technical,
        "final": final,
    }


def _registry_technical_and_table(
    authority: Mapping[str, Any], *, denominator: set[str]
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    record = authority.get("gt_mask_registry")
    if not isinstance(record, Mapping):
        raise PostprocessContractError("authority lacks the GT registry")
    path = Path(str(record["path"])).expanduser().resolve()
    names = set(_schema(path))
    required = {
        "sample_id",
        "original_gt_mask_sha256",
        "original_height",
        "original_width",
        "mapping_status",
        "pixel_qa_status",
        "annotation_suspect",
    }
    if not required.issubset(names):
        raise PostprocessContractError(
            f"GT registry schema differs: {sorted(required.difference(names))}"
        )
    registry = _read_parquet_columns(
        path, columns=sorted(required), label="locked GT registry"
    )
    registry["sample_id"] = registry["sample_id"].astype(str)
    if (
        registry["sample_id"].duplicated().any()
        or set(registry["sample_id"]) != denominator
    ):
        raise PostprocessContractError("GT registry lost the denominator")
    pass_mask = registry["mapping_status"].astype(str).eq("PASS") & registry[
        "pixel_qa_status"
    ].astype(str).eq("P2_MAPPING_QA_PASS")
    technical = set(registry.loc[~pass_mask, "sample_id"])
    rows = []
    for route in ROUTES:
        for row in registry.itertuples(index=False):
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "route": route,
                    "gt_mask_sha256": (
                        ""
                        if pd.isna(row.original_gt_mask_sha256)
                        else str(row.original_gt_mask_sha256)
                    ),
                    "rgb_shape": f"{int(row.original_height)}x{int(row.original_width)}",
                    "mask_shape": f"{int(row.original_height)}x{int(row.original_width)}",
                    "status": "PASS"
                    if row.sample_id not in technical
                    else "UNRESOLVED",
                }
            )
    return technical, pd.DataFrame(rows), registry


def _build_outcomes(
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    technical_by_branch: Mapping[tuple[str, str], set[str]],
    registry_technical: set[str],
    routes: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, Any]] = {}
    failures_by_route: dict[str, set[str]] = {}
    for route in routes:
        failed = set(registry_technical)
        for branch in BRANCHES:
            failed.update(technical_by_branch[(route, branch)])
            failed.update(
                labels.loc[
                    labels["route"].eq(route)
                    & labels["branch"].eq(branch)
                    & ~labels["evaluator_valid"].astype(bool),
                    "sample_id",
                ].astype(str)
            )
        failures_by_route[route] = failed
    for route in routes:
        ks = (5, 10) if route == "D1" else (5,)
        for branch in BRANCHES:
            outcomes, summary = compute_branch_metrics(
                labels,
                manifest[["sample_id"]],
                route=route,
                branch=branch,
                k_values=ks,
            )
            failed = failures_by_route[route]
            outcomes["technical_failure"] = outcomes["sample_id"].isin(failed)
            technical_mask = outcomes["technical_failure"]
            outcomes.loc[technical_mask, "native_correct"] = False
            outcomes.loc[technical_mask, "first_positive_rank"] = pd.NA
            outcomes.loc[technical_mask, "positive_candidate_count"] = 0
            outcomes.loc[technical_mask, "oracle_all"] = False
            outcomes.loc[technical_mask, "reciprocal_rank"] = 0.0
            for column in outcomes.columns:
                if column.startswith(("j_at_", "oracle_at_")):
                    outcomes.loc[technical_mask, column] = False
            frames.append(outcomes)
            del summary
            metrics[f"{route}|{branch}"] = summarize_sample_outcomes(outcomes)
    return pd.concat(frames, ignore_index=True), metrics


def _paired_route_frame(
    outcomes: pd.DataFrame,
    *,
    route: str,
    covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred = outcomes.loc[
        outcomes["route"].eq(route) & outcomes["branch"].eq("predicted")
    ].copy()
    gt = outcomes.loc[
        outcomes["route"].eq(route) & outcomes["branch"].eq("gt_oracle")
    ].copy()
    _, summary = compare_branch_outcomes(pred, gt)
    if route != "D1":
        pred["oracle_at_10"] = False
        gt["oracle_at_10"] = False
    elif pred["oracle_at_10"].isna().any() or gt["oracle_at_10"].isna().any():
        raise PostprocessContractError("D1 Top-10 outcomes are missing")
    rename_pred = {
        "native_correct": "pred_native_correct",
        "oracle_at_5": "pred_top5_positive",
        "oracle_at_10": "pred_top10_positive",
        "oracle_all": "pred_all_positive",
        "candidate_count": "pred_candidate_count",
        "positive_candidate_count": "pred_positive_candidate_count",
        "first_positive_rank": "pred_first_positive_rank",
        "no_output": "pred_no_output",
        "technical_failure": "pred_technical_failure",
    }
    rename_gt = {key: key.replace("pred_", "gt_") for key in rename_pred.values()}
    rename_gt = dict(zip(rename_pred, rename_gt.values(), strict=True))
    pred_columns = ["sample_id", *rename_pred]
    gt_columns = ["sample_id", *rename_gt]
    paired = (
        pred[pred_columns]
        .rename(columns=rename_pred)
        .merge(
            gt[gt_columns].rename(columns=rename_gt),
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
    )
    paired.insert(1, "route", route)
    paired["technical_failure"] = paired[
        ["pred_technical_failure", "gt_technical_failure"]
    ].any(axis=1)
    paired = paired.merge(covariates, on="sample_id", validate="one_to_one")
    paired["predicted_mask_empty"] = False
    paired["gt_mask_empty_or_invalid"] = paired["technical_failure"]
    paired["annotation_suspect"] = (
        paired["annotation_suspect"].astype(bool)
        if "annotation_suspect" in paired
        else False
    )
    return paired, summary


def _source_reconciliation(lock: Mapping[str, Any]) -> pd.DataFrame:
    bindings = lock.get("bindings")
    source_locks = (
        bindings.get("source_locks") if isinstance(bindings, Mapping) else None
    )
    records = _collect_records(source_locks, prefix="source_locks")
    if not records:
        raise PostprocessContractError("protocol source-lock binding is empty")
    return pd.DataFrame(
        [
            {
                "source_name": label,
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
                "bytes": int(
                    record.get("bytes", Path(str(record["path"])).stat().st_size)
                ),
                "status": "PASS",
            }
            for label, record in records
        ]
    )


def _verify_external_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise PostprocessContractError(f"{label} is absent or unsafe")
    if sha256_file(path) != record.get("sha256"):
        raise PostprocessContractError(f"{label} hash differs")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise PostprocessContractError(f"{label} byte count differs")
    return path


def _locked_source_inventory(lock: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    bindings = lock.get("bindings")
    source_locks = (
        bindings.get("source_locks") if isinstance(bindings, Mapping) else None
    )
    pending = [
        record for _, record in _collect_records(source_locks, prefix="source_locks")
    ]
    seen: set[tuple[str, str]] = set()
    inventory: set[tuple[str, str, int]] = set()
    while pending:
        record = pending.pop()
        path = _verify_external_record(record, label="protocol-bound source lock")
        identity = (str(path), str(record["sha256"]))
        if identity in seen:
            continue
        seen.add(identity)
        if path.suffix.lower() != ".json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping):
            continue
        rows = value.get("inventory")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping) or not {
                    "path",
                    "sha256",
                    "bytes",
                }.issubset(row):
                    raise PostprocessContractError(
                        f"source-lock inventory row {index} is malformed"
                    )
                source = _verify_external_record(
                    row, label="source-lock inventory artifact"
                )
                inventory.add((str(source), str(row["sha256"]), int(row["bytes"])))
        for label, child in _collect_records(value, prefix="source_lock_payload"):
            if label.endswith((".final_lock", ".source_lock_verification")):
                pending.append(child)
    if not inventory:
        raise PostprocessContractError(
            "protocol source locks expose no immutable inventory"
        )
    return inventory


def _d1_depth_selector_frame(source: Path) -> pd.DataFrame:
    names = set(_schema(source))
    required = {
        "sample_id",
        "system_name",
        "selected_candidate_id",
        "selected_correct",
        "no_output",
    }
    if not required.issubset(names):
        raise PostprocessContractError(
            "D1 formal source lacks locked depth-selector columns"
        )
    columns = sorted(
        required
        | {
            column
            for column in ("formal_score", "is_selected", "row_kind")
            if column in names
        }
    )
    raw = _read_parquet_columns(
        source, columns=columns, label="D1 locked depth selectors"
    )
    frames: list[pd.DataFrame] = []
    for pool, system in (
        ("top5", "d1_top5_r7_gated"),
        ("top10", "d1_top10_locked"),
        ("allnms", "d1_allnms_locked"),
    ):
        subset = raw.loc[raw["system_name"].astype(str).eq(system)].copy()
        selected_mask = (
            subset["is_selected"].astype(bool)
            if "is_selected" in subset
            else pd.Series(True, index=subset.index)
        )
        no_output_mask = subset["no_output"].astype(bool)
        subset = subset.loc[selected_mask | no_output_mask].copy()
        subset["sample_id"] = subset["sample_id"].astype(str)
        if subset["sample_id"].duplicated().any():
            raise PostprocessContractError(
                f"D1 {pool} selector has duplicate per-sample decisions"
            )
        subset["selected_candidate_id"] = subset["selected_candidate_id"].fillna("").astype(str)
        subset["selected_correct"] = _validate_binary(
            subset, "selected_correct", label=f"D1 {pool} selector"
        )
        subset["no_output"] = _validate_binary(
            subset, "no_output", label=f"D1 {pool} selector"
        )
        if subset["no_output"].ne(subset["selected_candidate_id"].eq("")).any():
            raise PostprocessContractError(
                f"D1 {pool} no-output and selected candidate differ"
            )
        subset["pool"] = pool
        subset["selector"] = system
        subset["selector_score"] = (
            pd.to_numeric(subset["formal_score"], errors="coerce")
            if "formal_score" in subset
            else np.nan
        )
        frames.append(
            subset[
                [
                    "sample_id",
                    "pool",
                    "selector",
                    "selected_candidate_id",
                    "selector_score",
                    "selected_correct",
                    "no_output",
                ]
            ]
        )
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["pool", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    denominator = set(result.loc[result["pool"].eq("top5"), "sample_id"])
    if (
        len(result) != 3 * len(denominator)
        or any(
            set(result.loc[result["pool"].eq(pool), "sample_id"]) != denominator
            for pool in ("top5", "top10", "allnms")
        )
    ):
        raise PostprocessContractError("D1 depth-selector denominator differs")
    return result


def _validate_final_outcomes_authority(
    *,
    root: Path,
    lock: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    routes: Sequence[str],
) -> dict[str, dict[str, Any]]:
    path, _ = _safe_record(
        artifacts["final_outcomes_authority"],
        root=root,
        label="final_outcomes_authority",
        allowed_prefixes=("04_predicted_replay",),
    )
    value = _json_object(path, label="final outcomes authority")
    _verify_self_hash(value, label="final outcomes authority")
    sources = value.get("source_formal_artifacts")
    selectors = value.get("frozen_selector_contracts")
    expected_systems = {
        "G1": "g1_gated_primary",
        "C1": "c1_gated_primary",
        "D1": "d1_top5_r7_gated",
    }
    if (
        value.get("status") != "LOCKED"
        or value.get("scientific_role") != SCIENTIFIC_ROLE
        or value.get("raw_test_ground_truth_rows_read") != 0
        or value.get("training_or_selection_feedback_allowed") is not False
        or value.get("final_outcomes") != dict(artifacts["final_outcomes"])
        or not isinstance(sources, Mapping)
        or not sources
        or not isinstance(selectors, Mapping)
        or set(map(str.upper, selectors)) != set(routes)
    ):
        raise PostprocessContractError("final outcomes authority contract differs")
    locked_inventory = _locked_source_inventory(lock)
    declared_sources: set[tuple[str, str, int]] = set()
    for label, record in _collect_records(sources, prefix="final_sources"):
        source = _verify_external_record(record, label=label)
        identity = (str(source), str(record["sha256"]), int(record.get("bytes", -1)))
        if identity not in locked_inventory:
            raise PermissionError(
                f"final outcome source is absent from protocol source inventory: {label}"
            )
        declared_sources.add(identity)
    normalized: dict[str, dict[str, Any]] = {}
    for route, contract in selectors.items():
        route_name = str(route).upper()
        if not isinstance(contract, Mapping):
            raise PostprocessContractError(f"{route_name} selector contract is malformed")
        record = contract.get("source_artifact")
        system = contract.get("system")
        if not isinstance(record, Mapping) or system != expected_systems[route_name]:
            raise PostprocessContractError(
                f"{route_name} frozen selector identity differs"
            )
        source = _verify_external_record(record, label=f"{route_name} formal outcomes")
        identity = (str(source), str(record["sha256"]), int(record.get("bytes", -1)))
        if identity not in locked_inventory or identity not in declared_sources:
            raise PermissionError(
                f"{route_name} selector source is not doubly bound to the source inventory"
            )
        normalized[route_name] = {
            "source_artifact": dict(record),
            "system": str(system),
        }
    if "D1" in set(routes):
        depth_record = value.get("d1_locked_depth_selector_decisions")
        if not isinstance(depth_record, Mapping):
            raise PostprocessContractError(
                "final outcomes authority lacks D1 depth selectors"
            )
        depth_path, _ = _safe_record(
            depth_record,
            root=root,
            label="d1_locked_depth_selector_decisions",
            allowed_prefixes=("04_predicted_replay",),
        )
        if depth_path != root / D1_DEPTH_DECISIONS_RELATIVE_PATH:
            raise PermissionError("D1 depth selector decisions path is noncanonical")
        depth = _read_parquet_columns(
            depth_path,
            columns=[
                "sample_id",
                "pool",
                "selector",
                "selected_candidate_id",
                "selector_score",
                "selected_correct",
                "no_output",
            ],
            label="D1 depth selector decisions",
        )
        if (
            set(depth["pool"].astype(str)) != {"top5", "top10", "allnms"}
            or depth.duplicated(["sample_id", "pool"]).any()
            or depth["no_output"].astype(bool).ne(
                depth["selected_candidate_id"].fillna("").astype(str).eq("")
            ).any()
        ):
            raise PostprocessContractError("D1 depth selector decisions differ")
        expected_depth = _d1_depth_selector_frame(
            Path(str(normalized["D1"]["source_artifact"]["path"]))
            .expanduser()
            .resolve()
        )
        try:
            pd.testing.assert_frame_equal(
                depth.sort_values(["pool", "sample_id"], kind="mergesort").reset_index(drop=True),
                expected_depth,
                check_dtype=False,
            )
        except AssertionError as error:
            raise PostprocessContractError(
                "D1 depth selector decisions differ from frozen formal source"
            ) from error
        normalized["D1"]["depth_selector_decisions"] = dict(depth_record)
    return normalized


def _assert_final_outcomes_exact(
    final: pd.DataFrame,
    *,
    selector_contracts: Mapping[str, Mapping[str, Any]],
    denominator: set[str],
) -> None:
    """Recompute frozen final bits from source formal tables, never trust a copy."""

    for route, contract in selector_contracts.items():
        source = Path(str(contract["source_artifact"]["path"])).expanduser().resolve()
        names = set(_schema(source))
        system_column = "system_name" if "system_name" in names else "system"
        required = {"sample_id", system_column, "selected_correct"}
        if not required.issubset(names):
            raise PostprocessContractError(
                f"{route} formal source misses {sorted(required.difference(names))}"
            )
        raw = _read_parquet_columns(
            source,
            columns=sorted(required | ({"is_selected", "no_output"} & names)),
            label=f"{route} source formal outcomes",
        )
        selected = raw.loc[
            raw[system_column].astype(str).eq(str(contract["system"]))
        ].copy()
        selected["sample_id"] = selected["sample_id"].astype(str)
        if selected["sample_id"].duplicated().any() and "is_selected" in selected:
            keep = selected["is_selected"].astype(bool)
            if "no_output" in selected:
                keep |= selected["no_output"].astype(bool)
            selected = selected.loc[keep].copy()
        if (
            selected["sample_id"].duplicated().any()
            or set(selected["sample_id"]) != denominator
        ):
            raise PostprocessContractError(
                f"{route} source selector does not exactly cover the denominator"
            )
        selected["final_correct"] = _validate_binary(
            selected, "selected_correct", label=f"{route} source selector"
        )
        observed = final.loc[final["route"].eq(route), ["sample_id", "final_correct"]]
        compared = observed.merge(
            selected[["sample_id", "final_correct"]],
            on="sample_id",
            how="outer",
            validate="one_to_one",
            suffixes=("_saved", "_source"),
        )
        if not compared["final_correct_saved"].eq(
            compared["final_correct_source"]
        ).all():
            raise PostprocessContractError(
                f"{route} saved final outcomes differ from frozen source selector"
            )


def _validate_d1_blocker(root: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    record = artifacts.get("d1_blocker")
    if not isinstance(record, Mapping):
        raise PostprocessContractError("partial analysis lacks a D1 blocker record")
    path, _ = _safe_record(
        record,
        root=root,
        label="d1_blocker",
        allowed_prefixes=("00_audit/machine_blockers",),
    )
    value = _json_object(path, label="D1 blocker")
    if "content_sha256" in value:
        _verify_self_hash(value, label="D1 blocker")
    if (
        value.get("status") != "UNRECOVERABLE_BLOCKER"
        or value.get("blocker_class") != "IRRECOVERABLE_FROZEN_SOURCE_EVIDENCE"
        or not value.get("missing_evidence")
        or not value.get("search_paths")
        or not value.get("stack_trace")
        or not value.get("resume_command")
        or value.get("filter_only_primary_allowed") is not False
        or value.get("raw_candidate_regeneration_required") is not True
    ):
        raise PostprocessContractError("D1 blocker could permit a fabricated primary")
    return value


def write_final_outcomes_authority(
    run_dir: str | Path,
    *,
    protocol_lock: str | Path,
    final_outcomes: Mapping[str, Any],
    selector_sources: Mapping[str, Mapping[str, Any]],
    routes: Sequence[str] = ROUTES,
) -> Path:
    """Bind and independently verify the copied frozen-final outcome bits."""

    root = _assert_run_dir(run_dir)
    lock_path = Path(protocol_lock).expanduser().resolve()
    if lock_path != root / LOCK_RELATIVE_PATH:
        raise PermissionError("final-outcomes authority uses a noncanonical protocol")
    lock = verify_protocol_lock(root)
    route_names = tuple(str(route).upper() for route in routes)
    if route_names not in {ROUTES, ("G1", "C1")} or set(selector_sources) != set(
        route_names
    ):
        raise ValueError("frozen selector source routes differ")
    final_path, final_record = _safe_record(
        final_outcomes,
        root=root,
        label="final_outcomes",
        allowed_prefixes=("04_predicted_replay",),
    )
    expected_systems = {
        "G1": "g1_gated_primary",
        "C1": "c1_gated_primary",
        "D1": "d1_top5_r7_gated",
    }
    selectors = {
        route: {
            "source_artifact": dict(selector_sources[route]),
            "system": expected_systems[route],
        }
        for route in route_names
    }
    d1_depth_record: dict[str, Any] | None = None
    if "D1" in route_names:
        d1_source = Path(str(selector_sources["D1"]["path"])).expanduser().resolve()
        depth = _d1_depth_selector_frame(d1_source)
        depth_path = atomic_parquet(
            depth,
            root / D1_DEPTH_DECISIONS_RELATIVE_PATH,
        )
        d1_depth_record = artifact_record(depth_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED",
        "scientific_role": SCIENTIFIC_ROLE,
        "raw_test_ground_truth_rows_read": 0,
        "training_or_selection_feedback_allowed": False,
        "final_outcomes": final_record,
        "source_formal_artifacts": {
            route: dict(record) for route, record in selector_sources.items()
        },
        "frozen_selector_contracts": selectors,
        "protocol_lock": artifact_record(lock_path),
    }
    if d1_depth_record is not None:
        payload["d1_locked_depth_selector_decisions"] = d1_depth_record
    payload["content_sha256"] = canonical_sha256(payload)
    destination = root / FINAL_OUTCOMES_AUTHORITY_RELATIVE_PATH
    if destination.exists():
        existing = _json_object(destination, label="final outcomes authority")
        if existing != payload:
            raise FileExistsError("existing final outcomes authority differs")
    else:
        exclusive_json(destination, payload)
    contracts = _validate_final_outcomes_authority(
        root=root,
        lock=lock,
        artifacts={
            "final_outcomes": final_record,
            "final_outcomes_authority": artifact_record(destination),
        },
        routes=route_names,
    )
    names = set(_schema(final_path))
    required = {"sample_id", "route", "final_correct"}
    if not required.issubset(names):
        raise PostprocessContractError("saved final-outcomes schema differs")
    final = _read_parquet_columns(
        final_path, columns=sorted(required), label="saved final outcomes"
    )
    final["sample_id"] = final["sample_id"].astype(str)
    final["route"] = final["route"].astype(str).str.upper()
    final["final_correct"] = _validate_binary(
        final, "final_correct", label="saved final outcomes"
    )
    denominator = set(final["sample_id"])
    if set(final["route"]) != set(route_names):
        raise PostprocessContractError("saved final outcome routes differ")
    _assert_final_outcomes_exact(
        final, selector_contracts=contracts, denominator=denominator
    )
    return destination


def _load_baseline_attestation(
    record: Mapping[str, Any],
    *,
    root: Path,
    branch_metrics: Mapping[str, Mapping[str, Any]],
    n: int,
    routes_to_check: Sequence[str],
) -> pd.DataFrame:
    path, _ = _safe_record(
        record,
        root=root,
        label="baseline_replay",
        allowed_prefixes=("04_predicted_replay",),
    )
    if path != root / "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json":
        raise PostprocessContractError(
            "baseline replay must be the canonical P1 closure"
        )
    closure = _json_object(path, label="baseline replay closure")
    _verify_self_hash(closure, label="baseline replay closure")
    if (
        closure.get("status") != "PASS"
        or closure.get("sample_count") != n
        or closure.get("all_three_predicted_pipelines_exact") is not True
        or closure.get("raw_test_ground_truth_rows_read") != 0
    ):
        raise PostprocessContractError(
            "predicted replay closure is not exact/label-free"
        )
    derived_record = closure.get("derived_reconciliation")
    route_records = closure.get("route_replays")
    if not isinstance(derived_record, Mapping) or not isinstance(
        route_records, Mapping
    ):
        raise PostprocessContractError(
            "predicted replay closure bindings are malformed"
        )
    if set(route_records) != {"g1", "c1", "d1"}:
        raise PostprocessContractError("predicted replay closure lacks a route")
    derived_path, _ = _safe_record(
        derived_record,
        root=root,
        label="derived_reconciliation",
        allowed_prefixes=("04_predicted_replay",),
    )
    derived = _json_object(derived_path, label="derived baseline reconciliation")
    _verify_self_hash(derived, label="derived baseline reconciliation")
    if (
        derived.get("status") != "PASS"
        or derived.get("raw_test_ground_truth_rows_read") != 0
    ):
        raise PostprocessContractError("derived baseline reconciliation did not PASS")
    routes = derived.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != {"g1", "c1", "d1"}:
        raise PostprocessContractError("predicted replay route attestation differs")
    for route, record_value in route_records.items():
        replay_path, _ = _safe_record(
            record_value,
            root=root,
            label=f"predicted replay {route}",
            allowed_prefixes=(f"04_predicted_replay/{route}", "04_predicted_replay"),
        )
        replay = _json_object(replay_path, label=f"predicted replay {route}")
        _verify_self_hash(replay, label=f"predicted replay {route}")
        if (
            replay.get("status") != "PASS"
            or replay.get("route") != route
            or replay.get("sample_count") != n
            or replay.get("raw_test_ground_truth_rows_read", 0) != 0
        ):
            raise PostprocessContractError(f"{route} predicted replay closure differs")
        if route in {"g1", "c1"}:
            differences = replay.get("maximum_absolute_differences")
            if (
                replay.get("serializer_atol") != 0.0
                or not isinstance(differences, Mapping)
                or not differences
                or any(float(value) != 0.0 for value in differences.values())
            ):
                raise PostprocessContractError(f"{route} replay is not exact")
        else:
            comparisons = replay.get("comparisons")
            if (
                not isinstance(comparisons, Mapping)
                or not comparisons
                or not all(value is True for value in comparisons.values())
            ):
                raise PostprocessContractError("D1 replay comparisons did not all PASS")
    rows = []
    for route in routes_to_check:
        observed = branch_metrics[f"{route}|predicted"]
        expected = routes[route.lower()]
        comparisons = {
            "N": (observed["N"], n),
            "native_correct": (
                observed["native_j_at_1_numerator"],
                expected.get("native_correct"),
            ),
            "oracle_top5": (
                observed["oracle_at_5_numerator"],
                expected.get("oracle_top5"),
            ),
            "oracle_all": (
                observed["oracle_all_numerator"],
                expected.get("oracle_all"),
            ),
            "no_output": (observed["no_output"], expected.get("no_output")),
        }
        if route == "D1":
            comparisons["oracle_top10"] = (
                observed["oracle_at_10_numerator"],
                expected.get("oracle_top10"),
            )
        bad = {key: pair for key, pair in comparisons.items() if pair[0] != pair[1]}
        if bad:
            raise PostprocessContractError(f"{route} predicted replay differs: {bad}")
        rows.append(
            {
                "route": route,
                "N": n,
                "native_correct": observed["native_j_at_1_numerator"],
                "oracle_all": observed["oracle_all_numerator"],
                "no_output": observed["no_output"],
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows)


def _taxonomy_tables(
    paired_by_route: Mapping[str, pd.DataFrame],
    final: pd.DataFrame,
    *,
    routes: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_frames: list[pd.DataFrame] = []
    post_frames: list[pd.DataFrame] = []
    for route in routes:
        native = classify_native_taxonomy(paired_by_route[route])
        native = add_secondary_flags(native)
        route_final = final.loc[
            final["route"].eq(route), ["sample_id", "final_correct"]
        ]
        post_input = native.merge(route_final, on="sample_id", validate="one_to_one")
        post = classify_post_r7_taxonomy(post_input)
        taxonomy_counts(native, column="native_taxonomy")
        taxonomy_counts(post, column="post_r7_taxonomy")
        native_frames.append(native)
        post_frames.append(post)
    return pd.concat(native_frames, ignore_index=True), pd.concat(
        post_frames, ignore_index=True
    )


def _candidate_mechanism_analysis(
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    routes: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match branch-local pools and publish only observable mechanisms.

    Final NMS-pool membership, geometry, evaluator labels, and rank movement are
    directly observable.  Crop, dense-peak, raw-sampling, and target-filter
    origins are reported as unknown unless an exact source-equivalence key is
    present; the diagnostic never guesses an upstream causal stage.
    """

    required = {
        "sample_id",
        "route",
        "branch",
        "candidate_id",
        "native_rank",
        "candidate_success",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise PostprocessContractError(
            f"candidate mechanism input misses columns: {missing}"
        )
    sample_ids = manifest["sample_id"].astype(str).tolist()
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise PostprocessContractError(
            "candidate mechanism denominator must be unique and non-empty"
        )
    work = labels.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["route"] = work["route"].astype(str).str.upper()
    work["branch"] = work["branch"].astype(str).str.lower()
    work["candidate_id"] = work["candidate_id"].astype(str)
    relation_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for route in routes:
        route_frame = work.loc[work["route"].eq(route)]
        for sample_id in sample_ids:
            sample = route_frame.loc[route_frame["sample_id"].eq(sample_id)]
            predicted = sample.loc[sample["branch"].eq("predicted")].copy()
            gt = sample.loc[sample["branch"].eq("gt_oracle")].copy()
            pred_records = predicted.to_dict("records")
            gt_records = gt.to_dict("records")
            matches = match_candidate_pools(pred_records, gt_records)
            pred_by_id = {
                str(row["candidate_id"]): row for row in pred_records
            }
            gt_by_id = {str(row["candidate_id"]): row for row in gt_records}
            pred_positive_ranks = sorted(
                int(row["native_rank"])
                for row in pred_records
                if bool(row["candidate_success"])
            )
            gt_positive_ranks = sorted(
                int(row["native_rank"])
                for row in gt_records
                if bool(row["candidate_success"])
            )
            if not matches:
                matches = [
                    {
                        "status": "empty_both_pool",
                        "predicted_candidate_id": None,
                        "gt_candidate_id": None,
                        "match_basis": "none",
                    }
                ]
            introduced_positive = 0
            removed_positive = 0
            matched_count = 0
            pred_only_count = 0
            gt_only_count = 0
            for match in matches:
                status = str(match["status"])
                predicted_id = match.get("predicted_candidate_id")
                gt_id = match.get("gt_candidate_id")
                pred_row = pred_by_id.get(str(predicted_id)) if predicted_id else None
                gt_row = gt_by_id.get(str(gt_id)) if gt_id else None
                pred_success = bool(pred_row["candidate_success"]) if pred_row else False
                gt_success = bool(gt_row["candidate_success"]) if gt_row else False
                pred_rank = int(pred_row["native_rank"]) if pred_row else None
                gt_rank = int(gt_row["native_rank"]) if gt_row else None
                if status == "matched_pred_gt_candidate":
                    matched_count += 1
                    if gt_success and not pred_success:
                        mechanism = "matched_geometry_became_positive"
                        introduced_positive += 1
                    elif pred_success and not gt_success:
                        mechanism = "matched_geometry_lost_positive"
                        removed_positive += 1
                    elif pred_success and gt_success and gt_rank < pred_rank:
                        mechanism = "matched_positive_rank_improved"
                    elif pred_success and gt_success and gt_rank > pred_rank:
                        mechanism = "matched_positive_rank_worsened"
                    else:
                        mechanism = "matched_no_positive_change"
                    upstream = (
                        "same_source_candidate"
                        if match.get("match_basis") == "source_nms_equivalence"
                        else "geometry_equivalent_source_stage_unknown"
                    )
                elif status == "gt_only_candidate":
                    gt_only_count += 1
                    mechanism = (
                        "gt_only_positive_candidate"
                        if gt_success
                        else "gt_only_negative_candidate"
                    )
                    upstream = "source_stage_unknown"
                    introduced_positive += int(gt_success)
                elif status == "pred_only_candidate":
                    pred_only_count += 1
                    mechanism = (
                        "pred_only_positive_removed"
                        if pred_success
                        else "pred_only_negative_removed"
                    )
                    upstream = "source_stage_unknown"
                    removed_positive += int(pred_success)
                elif status == "empty_both_pool":
                    mechanism = "both_pools_empty"
                    upstream = "no_candidate_evidence"
                else:  # pragma: no cover - guarded by matching primitive
                    raise PostprocessContractError(
                        f"unknown candidate match status: {status}"
                    )
                relation_rows.append(
                    {
                        "sample_id": sample_id,
                        "route": route,
                        "match_status": status,
                        "predicted_candidate_id": predicted_id,
                        "gt_candidate_id": gt_id,
                        "match_basis": str(match.get("match_basis", "none")),
                        "periodic_angle_difference_deg": match.get(
                            "periodic_angle_difference_deg"
                        ),
                        "center_distance_px": match.get("center_distance_px"),
                        "width_difference_px": match.get("width_difference_px"),
                        "rotated_rectangle_iou": match.get(
                            "rotated_rectangle_iou"
                        ),
                        "predicted_native_rank": pred_rank,
                        "gt_native_rank": gt_rank,
                        "rank_delta_gt_minus_pred": (
                            None
                            if pred_rank is None or gt_rank is None
                            else gt_rank - pred_rank
                        ),
                        "predicted_candidate_success": pred_success,
                        "gt_candidate_success": gt_success,
                        "observable_mechanism": mechanism,
                        "upstream_attribution": upstream,
                        "evidence_scope": "final_nms_pool_only",
                    }
                )
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "route": route,
                    "predicted_candidate_count": len(pred_records),
                    "gt_candidate_count": len(gt_records),
                    "nms_candidate_count_delta": len(gt_records) - len(pred_records),
                    "predicted_first_positive_rank": (
                        pred_positive_ranks[0] if pred_positive_ranks else None
                    ),
                    "gt_first_positive_rank": (
                        gt_positive_ranks[0] if gt_positive_ranks else None
                    ),
                    "first_positive_rank_delta": (
                        gt_positive_ranks[0] - pred_positive_ranks[0]
                        if pred_positive_ranks and gt_positive_ranks
                        else None
                    ),
                    "matched_candidate_count": matched_count,
                    "pred_only_candidate_count": pred_only_count,
                    "gt_only_candidate_count": gt_only_count,
                    "gt_introduced_positive_count": introduced_positive,
                    "gt_removed_positive_count": removed_positive,
                    "crop_change_status": "UNKNOWN_SOURCE_ARTIFACT_ABSENT",
                    "raw_candidate_count_change_status": (
                        "UNKNOWN_SOURCE_ARTIFACT_ABSENT"
                    ),
                    "dense_peak_or_filter_attribution": (
                        "UNKNOWN_UNLESS_SOURCE_EQUIVALENCE_KEY"
                    ),
                }
            )

    relations = pd.DataFrame(relation_rows)
    samples = pd.DataFrame(sample_rows)
    expected_sample_rows = len(sample_ids) * len(routes)
    if len(samples) != expected_sample_rows or samples.duplicated(
        ["sample_id", "route"]
    ).any():
        raise PostprocessContractError("candidate mechanism sample coverage differs")
    for route in routes:
        route_labels = work.loc[work["route"].eq(route)]
        route_relations = relations.loc[relations["route"].eq(route)]
        matched = route_relations["match_status"].eq("matched_pred_gt_candidate")
        pred_references = int(matched.sum()) + int(
            route_relations["match_status"].eq("pred_only_candidate").sum()
        )
        gt_references = int(matched.sum()) + int(
            route_relations["match_status"].eq("gt_only_candidate").sum()
        )
        if (
            pred_references
            != int(route_labels["branch"].eq("predicted").sum())
            or gt_references
            != int(route_labels["branch"].eq("gt_oracle").sum())
        ):
            raise PostprocessContractError(
                f"{route} candidate mechanism inventory is not exhaustive"
            )
    summary = (
        relations.assign(
            positive_transition=relations[
                "observable_mechanism"
            ].isin(
                {
                    "matched_geometry_became_positive",
                    "matched_geometry_lost_positive",
                    "gt_only_positive_candidate",
                    "pred_only_positive_removed",
                }
            )
        )
        .groupby(
            ["route", "observable_mechanism", "upstream_attribution"],
            sort=True,
            dropna=False,
        )
        .agg(
            candidate_relation_count=("sample_id", "size"),
            affected_sample_count=("sample_id", "nunique"),
            positive_transition_count=("positive_transition", "sum"),
        )
        .reset_index()
    )
    summary["evidence_scope"] = "observable_final_nms_pool_transition_only"
    return relations, samples, summary


def _d1_depth_analysis(
    *,
    root: Path,
    artifacts: Mapping[str, Any],
    native: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Replay all three frozen D1 depth selectors into comparable evidence.

    The primary post-R7 taxonomy remains the preregistered Top5 selector.  This
    secondary table applies the same residual taxonomy to the locked Top5,
    Top10, and AllNMS decisions without selecting or refitting anything.
    """

    d1_native = native.loc[native["route"].eq("D1")].copy()
    if d1_native.empty:
        return None
    authority_record = artifacts.get("final_outcomes_authority")
    if not isinstance(authority_record, Mapping):
        raise PostprocessContractError(
            "D1 depth analysis lacks final-outcomes authority"
        )
    authority_path, _ = _safe_record(
        authority_record,
        root=root,
        label="final_outcomes_authority",
        allowed_prefixes=("04_predicted_replay",),
    )
    authority = _json_object(authority_path, label="final outcomes authority")
    _verify_self_hash(authority, label="final outcomes authority")
    depth_record = authority.get("d1_locked_depth_selector_decisions")
    if not isinstance(depth_record, Mapping):
        raise PostprocessContractError(
            "D1 depth analysis lacks locked selector decisions"
        )
    depth_path, _ = _safe_record(
        depth_record,
        root=root,
        label="d1_locked_depth_selector_decisions",
        allowed_prefixes=("04_predicted_replay",),
    )
    if depth_path != root / D1_DEPTH_DECISIONS_RELATIVE_PATH:
        raise PermissionError("D1 depth analysis uses a noncanonical decision table")
    depth = _read_parquet_columns(
        depth_path,
        columns=[
            "sample_id",
            "pool",
            "selector",
            "selected_candidate_id",
            "selector_score",
            "selected_correct",
            "no_output",
        ],
        label="D1 locked depth selector decisions",
    )
    d1_native["sample_id"] = d1_native["sample_id"].astype(str)
    denominator = set(d1_native["sample_id"])
    if d1_native["sample_id"].duplicated().any():
        raise PostprocessContractError("D1 native taxonomy denominator is duplicated")

    metric_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    per_sample: list[pd.DataFrame] = []
    for pool, selector in (
        ("top5", "d1_top5_r7_gated"),
        ("top10", "d1_top10_locked"),
        ("allnms", "d1_allnms_locked"),
    ):
        selected = depth.loc[depth["pool"].astype(str).eq(pool)].copy()
        selected["sample_id"] = selected["sample_id"].astype(str)
        if (
            len(selected) != len(denominator)
            or selected["sample_id"].duplicated().any()
            or set(selected["sample_id"]) != denominator
            or not selected["selector"].astype(str).eq(selector).all()
        ):
            raise PostprocessContractError(
                f"D1 {pool} locked-selector denominator differs"
            )
        selected["selected_correct"] = _validate_binary(
            selected, "selected_correct", label=f"D1 {pool} locked selector"
        )
        selected["no_output"] = _validate_binary(
            selected, "no_output", label=f"D1 {pool} locked selector"
        )
        post_input = d1_native.drop(
            columns=["final_correct", "post_r7_taxonomy", "is_residual_failure"],
            errors="ignore",
        ).merge(
            selected[["sample_id", "selected_correct"]].rename(
                columns={"selected_correct": "final_correct"}
            ),
            on="sample_id",
            validate="one_to_one",
        )
        classified = classify_post_r7_taxonomy(post_input)
        classified["pool"] = pool
        classified["selector"] = selector
        per_sample.append(classified)
        counts = taxonomy_counts(classified, column="post_r7_taxonomy")
        residual = int(classified["is_residual_failure"].sum())
        taxonomy_rows.extend(
            {
                "pool": pool,
                "selector": selector,
                "taxonomy": label,
                "count": count,
                "N": len(selected),
                "residual_failures": residual,
            }
            for label, count in counts.items()
        )
        correct = int(selected["selected_correct"].sum())
        metric_rows.append(
            {
                "pool": pool,
                "selector": selector,
                "N": len(selected),
                "selected_correct": correct,
                "no_output": int(selected["no_output"].sum()),
                "j_at_1": correct / len(selected),
                "secondary_locked_sensitivity": True,
            }
        )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(per_sample, ignore_index=True),
        pd.DataFrame(taxonomy_rows),
    )


def _statistics_inputs(native: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    result = native.merge(
        manifest[["sample_id", "scene_id", "frame_id"]],
        on="sample_id",
        validate="many_to_one",
    )
    result["pred_native_j_at_1"] = result["pred_native_correct"]
    result["gt_native_j_at_1"] = result["gt_native_correct"]
    result["pred_oracle_at_5"] = result["pred_top5_positive"]
    result["gt_oracle_at_5"] = result["gt_top5_positive"]
    result["pred_oracle_at_10"] = result["pred_top10_positive"]
    result["gt_oracle_at_10"] = result["gt_top10_positive"]
    result["pred_oracle_all"] = result["pred_all_positive"]
    result["gt_oracle_all"] = result["gt_all_positive"]
    return (
        result[
            [
                "sample_id",
                "route",
                "scene_id",
                "frame_id",
                "pred_native_j_at_1",
                "gt_native_j_at_1",
                "pred_oracle_at_5",
                "gt_oracle_at_5",
                "pred_oracle_at_10",
                "gt_oracle_at_10",
                "pred_oracle_all",
                "gt_oracle_all",
            ]
        ]
        .sort_values(["route", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _statistical_tests(
    statistical_inputs: pd.DataFrame,
    *,
    iterations: int,
    routes: Sequence[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    for route in routes:
        frame = statistical_inputs.loc[statistical_inputs["route"].eq(route)]
        metrics = [
            ("native_j_at_1", "pred_native_j_at_1", "gt_native_j_at_1"),
            ("oracle_at_5", "pred_oracle_at_5", "gt_oracle_at_5"),
            ("oracle_all", "pred_oracle_all", "gt_oracle_all"),
        ]
        if route == "D1":
            metrics.insert(2, ("oracle_at_10", "pred_oracle_at_10", "gt_oracle_at_10"))
        for metric, reference, counterfactual in metrics:
            value = paired_metric_statistics(
                frame,
                reference_column=reference,
                counterfactual_column=counterfactual,
                iterations=iterations,
                seed=BOOTSTRAP_SEED,
            )
            comparisons.append({"route": route, "metric": metric, **value})
    adjusted = apply_holm_family(comparisons)
    rows = []
    for value in adjusted:
        ci = value["scene_cluster_bootstrap"]["ci"]
        frame_ci = value["frame_cluster_bootstrap_sensitivity"]["ci"]
        rows.append(
            {
                "route": value["route"],
                "metric": value["metric"],
                "N": value["N"],
                "delta": value["delta"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "raw_p": value["raw_p"],
                "holm_p": value["holm_adjusted_p"],
                "reference_numerator": value["reference_numerator"],
                "counterfactual_numerator": value["counterfactual_numerator"],
                "reference_rate": value["reference_rate"],
                "counterfactual_rate": value["counterfactual_rate"],
                "b_reference_only": value["b_reference_only"],
                "c_counterfactual_only": value["c_counterfactual_only"],
                "scene_cluster_count": value["scene_cluster_bootstrap"][
                    "cluster_count"
                ],
                "frame_ci_low": frame_ci[0],
                "frame_ci_high": frame_ci[1],
                "frame_cluster_count": value[
                    "frame_cluster_bootstrap_sensitivity"
                ]["cluster_count"],
                "bootstrap_iterations": value["scene_cluster_bootstrap"][
                    "iterations"
                ],
                "bootstrap_seed": value["scene_cluster_bootstrap"]["seed"],
                "inference_scope": value["inference_scope"],
            }
        )
    return pd.DataFrame(rows), adjusted


def _quartile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.nunique() <= 1:
        return pd.Series("all", index=values.index, dtype=object)
    ranked = numeric.rank(method="first")
    return pd.qcut(
        ranked,
        4,
        labels=("Q1", "Q2", "Q3", "Q4"),
    ).astype(str)


def _first_rank_bin(value: Any) -> str:
    if pd.isna(value):
        return "none"
    rank = int(value)
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    return ">10"


def _failure_modes(labels: pd.DataFrame, *, route: str) -> dict[str, str]:
    selected = labels.loc[labels["route"].eq(route) & labels["branch"].eq("predicted")]
    result: dict[str, str] = {}
    for sample_id, group in selected.groupby("sample_id", sort=False):
        if group["candidate_success"].astype(bool).any():
            result[str(sample_id)] = "has_positive"
            continue
        iou = pd.to_numeric(group["best_same_gt_iou"], errors="coerce")
        angle = pd.to_numeric(group["best_same_gt_angle_error_deg"], errors="coerce")
        iou_only = (iou > 0.25) & (angle > 30.0)
        angle_only = (angle <= 30.0) & (iou <= 0.25)
        if iou_only.any():
            result[str(sample_id)] = "iou_pass_angle_fail"
        elif angle_only.any():
            result[str(sample_id)] = "angle_pass_iou_fail"
        else:
            result[str(sample_id)] = "both_fail"
    return result


def _stratified_table(
    native: pd.DataFrame, labels: pd.DataFrame, *, routes: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route in routes:
        frame = native.loc[native["route"].eq(route)].copy()
        frame["predicted_mask_iou_bin"] = pd.cut(
            pd.to_numeric(frame["predicted_mask_iou"]),
            [-math.inf, 0.25, 0.50, 0.70, 0.90, math.inf],
            right=False,
            labels=(
                "[0,0.25)",
                "[0.25,0.50)",
                "[0.50,0.70)",
                "[0.70,0.90)",
                "[0.90,1.00]",
            ),
        ).astype(str)
        frame["target_area_quartile"] = _quartile(frame["target_area_fraction"])
        frame["boundary_complexity_quartile"] = _quartile(
            frame["mask_boundary_complexity"]
        )
        frame["valid_depth_ratio_quartile"] = _quartile(frame["valid_depth_ratio"])
        frame["mask_component_count_bin"] = pd.to_numeric(
            frame["mask_component_count"], errors="raise"
        ).map(lambda value: "1" if value == 1 else "2" if value == 2 else "3+")
        frame["candidate_count_bin"] = pd.cut(
            pd.to_numeric(frame["pred_candidate_count"]),
            [-1, 0, 5, 10, 30, math.inf],
            labels=("0", "1-5", "6-10", "11-30", "31+"),
        ).astype(str)
        frame["first_positive_rank_bin"] = frame["pred_first_positive_rank"].map(
            _first_rank_bin
        )
        modes = _failure_modes(labels, route=route)
        frame["predicted_candidate_failure_mode"] = (
            frame["sample_id"].map(modes).fillna("no_output")
        )
        stratum_columns = (
            "query_type",
            "predicted_mask_iou_bin",
            "target_area_quartile",
            "mask_component_count_bin",
            "boundary_complexity_quartile",
            "candidate_count_bin",
            "valid_depth_ratio_quartile",
            "scene_family",
            "frame_family",
            "first_positive_rank_bin",
            "predicted_candidate_failure_mode",
        )
        for column in stratum_columns:
            for value, group in frame.groupby(column, dropna=False, sort=True):
                pred = group["pred_all_positive"].astype(bool)
                gt = group["gt_all_positive"].astype(bool)
                rows.append(
                    {
                        "route": route,
                        "stratum_name": column,
                        "stratum_value": str(value),
                        "N": len(group),
                        "recovered": int((~pred & gt).sum()),
                        "harmful": int((pred & ~gt).sum()),
                        "delta": float(gt.mean() - pred.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _tables(
    *,
    lock: Mapping[str, Any],
    mapping_table: pd.DataFrame,
    baseline_table: pd.DataFrame,
    branch_metrics: Mapping[str, Mapping[str, Any]],
    paired_summaries: Mapping[str, Mapping[str, Any]],
    native: pd.DataFrame,
    post: pd.DataFrame,
    statistics: pd.DataFrame,
    stratified: pd.DataFrame,
    artifacts: Mapping[str, Any],
    root: Path,
    routes: Sequence[str],
    candidate_mechanism_summary: pd.DataFrame,
    d1_depth_metrics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    branch_rows = []
    for route in routes:
        for branch in BRANCHES:
            value = branch_metrics[f"{route}|{branch}"]
            row = {
                "route": route,
                "branch": branch,
                "N": value["N"],
                "native_correct": value["native_j_at_1_numerator"],
                "oracle_at_5": value["oracle_at_5_numerator"],
                "oracle_all": value["oracle_all_numerator"],
                "no_output": value["no_output"],
                "no_output_rate": value["no_output_rate"],
                "candidate_count_mean": value["candidate_count_mean"],
                "candidate_count_median": value["candidate_count_median"],
                "candidate_count_p95": value["candidate_count_p95"],
                "native_j_at_1": value["native_j_at_1"],
                "j_at_5": value["j_at_5"],
                "oracle_at_5_rate": value["oracle_at_5"],
                "oracle_all_rate": value["oracle_all"],
                "mrr": value["mrr"],
                "first_positive_rank_distribution_json": json.dumps(
                    value["first_positive_rank_distribution"], sort_keys=True
                ),
                "positive_candidates_per_sample_mean": value[
                    "positive_candidates_per_sample_mean"
                ],
                "positive_candidates_per_sample_median": value[
                    "positive_candidates_per_sample_median"
                ],
                "positive_candidates_per_sample_p95": value[
                    "positive_candidates_per_sample_p95"
                ],
                "positive_candidates_per_sample_distribution_json": json.dumps(
                    value["positive_candidates_per_sample_distribution"], sort_keys=True
                ),
            }
            if route == "D1":
                row["oracle_at_10"] = value["oracle_at_10_numerator"]
                row["j_at_10"] = value["j_at_10"]
                row["oracle_at_10_rate"] = value["oracle_at_10"]
            else:
                row["oracle_at_10"] = pd.NA
                row["j_at_10"] = pd.NA
                row["oracle_at_10_rate"] = pd.NA
            branch_rows.append(row)
    branch_table = pd.DataFrame(branch_rows)

    paired_rows = []
    native_count_rows = []
    post_count_rows = []
    transition_rows = []
    rank_rows = []
    sensitivity_rows = []
    for route in routes:
        route_native = native.loc[native["route"].eq(route)]
        route_post = post.loc[post["route"].eq(route)]
        n = len(route_native)
        native_counts = taxonomy_counts(route_native, column="native_taxonomy")
        post_counts = taxonomy_counts(route_post, column="post_r7_taxonomy")
        residual_n = sum(post_counts.values())
        pred_all = route_native["pred_all_positive"].astype(bool)
        gt_all = route_native["gt_all_positive"].astype(bool)
        residual_grounding = sum(post_counts[label] for label in POST_R7_CLASSES[2:5])
        paired_rows.append(
            {
                "route": route,
                "N": n,
                "pred_oracle_all": int(pred_all.sum()),
                "gt_oracle_all": int(gt_all.sum()),
                "delta_oracle_all": float(gt_all.mean() - pred_all.mean()),
                "pred_no_positive": int((~pred_all).sum()),
                "gt_no_positive": int((~gt_all).sum()),
                "grounding_recovered": int((~pred_all & gt_all).sum()),
                "grounding_plus_selection": native_counts[NATIVE_CLASSES[5]],
                "generator_limited_under_gt": native_counts[NATIVE_CLASSES[7]],
                "gt_regression": int((pred_all & ~gt_all).sum()),
                "both_positive": int((pred_all & gt_all).sum()),
                "neither_positive": int((~pred_all & ~gt_all).sum()),
                "grounding_candidate_recovery_denominator": paired_summaries[route][
                    "grounding_candidate_recovery_denominator"
                ],
                "grounding_candidate_recovery_rate": paired_summaries[route][
                    "grounding_candidate_recovery_rate"
                ],
                "oracle_grounding_ceiling": paired_summaries[route][
                    "oracle_grounding_ceiling"
                ],
                "residual_generator_failure_count": paired_summaries[route][
                    "residual_generator_failure_count"
                ],
                "residual_generator_failure_rate": paired_summaries[route][
                    "residual_generator_failure_rate"
                ],
                "candidate_count_delta_mean": paired_summaries[route][
                    "candidate_count_delta_mean"
                ],
                "candidate_count_delta_median": paired_summaries[route][
                    "candidate_count_delta_median"
                ],
                "first_positive_rank_delta_mean_both_positive": paired_summaries[
                    route
                ]["first_positive_rank_delta_mean_both_positive"],
                "delta_native_j_at_1": paired_summaries[route][
                    "delta_native_j_at_1"
                ],
                "delta_oracle_at_5": paired_summaries[route]["delta_oracle_at_5"],
                "delta_oracle_at_10": (
                    paired_summaries[route].get("delta_oracle_at_10")
                    if route == "D1"
                    else pd.NA
                ),
                "post_r7_residual_grounding_fraction": (
                    0.0 if residual_n == 0 else residual_grounding / residual_n
                ),
                "post_r7_residual_generator_fraction": (
                    0.0
                    if residual_n == 0
                    else post_counts[POST_R7_CLASSES[5]] / residual_n
                ),
            }
        )
        native_count_rows.extend(
            {"route": route, "taxonomy": label, "count": count, "N": n}
            for label, count in native_counts.items()
        )
        post_count_rows.extend(
            {"route": route, "taxonomy": label, "count": count, "N": n}
            for label, count in post_counts.items()
        )
        transitions = {
            "all_oracle:pred_negative_to_gt_positive": ~pred_all & gt_all,
            "all_oracle:pred_positive_to_gt_negative": pred_all & ~gt_all,
            "all_oracle:both_positive": pred_all & gt_all,
            "all_oracle:neither_positive": ~pred_all & ~gt_all,
            "candidate_count:increased": route_native["gt_candidate_count"]
            > route_native["pred_candidate_count"],
            "candidate_count:decreased": route_native["gt_candidate_count"]
            < route_native["pred_candidate_count"],
            "candidate_count:unchanged": route_native["gt_candidate_count"]
            == route_native["pred_candidate_count"],
        }
        transition_rows.extend(
            {
                "route": route,
                "transition_family": label.split(":", 1)[0],
                "transition": label,
                "count": int(mask.sum()),
                "N": n,
            }
            for label, mask in transitions.items()
        )
        ranks = (
            route_native.assign(
                pred_rank=route_native["pred_first_positive_rank"].map(
                    lambda value: "none" if pd.isna(value) else str(int(value))
                ),
                gt_rank=route_native["gt_first_positive_rank"].map(
                    lambda value: "none" if pd.isna(value) else str(int(value))
                ),
            )
            .groupby(["pred_rank", "gt_rank"], dropna=False)
            .size()
        )
        rank_rows.extend(
            {
                "route": route,
                "pred_first_positive_rank": pred_rank,
                "gt_first_positive_rank": gt_rank,
                "count": int(count),
            }
            for (pred_rank, gt_rank), count in ranks.items()
        )
        suspect = route_native["annotation_suspect"].astype(bool)
        for status, selected in (
            ("included", pd.Series(True, index=route_native.index)),
            ("excluded", ~suspect),
        ):
            if not selected.any():
                raise PostprocessContractError(
                    f"all {route} samples are annotation-suspect"
                )
            subset = route_native.loc[selected]
            sensitivity_rows.append(
                {
                    "route": route,
                    "suspect_status": status,
                    "N": len(subset),
                    "delta_oracle_all": float(
                        subset["gt_all_positive"].astype(bool).mean()
                        - subset["pred_all_positive"].astype(bool).mean()
                    ),
                }
            )

    selector_record = artifacts.get("frozen_selector_transfer")
    if selector_record is None:
        selector = pd.DataFrame(
            [
                {
                    "route": route,
                    "branch": "gt_oracle",
                    "N": branch_metrics[f"{route}|gt_oracle"]["N"],
                    "selector": "NOT_RUN",
                    "correct": pd.NA,
                    "oracle_all": branch_metrics[f"{route}|gt_oracle"][
                        "oracle_all_numerator"
                    ],
                    "secondary_only": 1,
                    "status": "NOT_RUN_OPTIONAL_SECONDARY",
                }
                for route in routes
            ]
        )
    else:
        path, _ = _safe_record(
            selector_record, root=root, label="frozen_selector_transfer"
        )
        selector = pd.read_parquet(path)
        if set(selector["route"].astype(str).str.upper()) != set(ROUTES):
            raise PostprocessContractError(
                "frozen selector transfer route coverage differs"
            )

    if d1_depth_metrics is not None:
        expected_columns = {
            "pool",
            "selector",
            "N",
            "selected_correct",
            "no_output",
            "j_at_1",
            "secondary_locked_sensitivity",
        }
        if set(d1_depth_metrics.columns) != expected_columns:
            raise PostprocessContractError("D1 depth-selector metrics schema differs")
        d1_rows = d1_depth_metrics.assign(
            route="D1",
            branch="predicted",
            correct=d1_depth_metrics["selected_correct"],
            oracle_all=branch_metrics["D1|predicted"]["oracle_all_numerator"],
            secondary_only=1,
            status="LOCKED_DEPTH_SENSITIVITY",
        )
        selector = pd.concat(
            [
                selector,
                d1_rows[
                    [
                        "route",
                        "branch",
                        "N",
                        "selector",
                        "correct",
                        "oracle_all",
                        "secondary_only",
                        "status",
                        "pool",
                        "no_output",
                        "j_at_1",
                    ]
                ],
            ],
            ignore_index=True,
            sort=False,
        )

    tables = {
        "source_reconciliation.csv": _source_reconciliation(lock),
        "gt_mask_mapping_audit.csv": mapping_table,
        "predicted_replay_metrics.csv": baseline_table,
        "branch_metrics.csv": branch_table,
        "pred_vs_gt_paired_metrics.csv": pd.DataFrame(paired_rows),
        "native_failure_taxonomy.csv": pd.DataFrame(native_count_rows),
        "post_r7_bottleneck_taxonomy.csv": pd.DataFrame(post_count_rows),
        "candidate_pool_transitions.csv": pd.DataFrame(transition_rows),
        "candidate_mechanism_summary.csv": candidate_mechanism_summary,
        "first_positive_rank_transitions.csv": pd.DataFrame(rank_rows),
        "stratified_results.csv": stratified,
        "statistical_tests.csv": statistics,
        "annotation_suspect_sensitivity.csv": pd.DataFrame(sensitivity_rows),
        "frozen_selector_transfer.csv": selector,
    }
    if set(tables) != set(TABLE_CONTRACTS):
        raise RuntimeError(
            "internal table bundle does not contain the complete table contract"
        )
    return tables


def _independent_check(
    *,
    manifest: pd.DataFrame,
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    labels: pd.DataFrame,
    outcomes: pd.DataFrame,
    branch_metrics: Mapping[str, Mapping[str, Any]],
    native: pd.DataFrame,
    post: pd.DataFrame,
    statistical_inputs: pd.DataFrame,
    final: pd.DataFrame,
    routes: Sequence[str],
) -> dict[str, Any]:
    checked_routes = []
    for route in routes:
        route_native = native.loc[native["route"].eq(route)].copy()
        independent_manifest = manifest.copy()
        technical = set(
            route_native.loc[
                route_native["technical_failure"].astype(bool), "sample_id"
            ]
        )
        independent_manifest["technical_failure"] = independent_manifest[
            "sample_id"
        ].isin(technical)
        result = independent_recompute_from_frames(
            independent_manifest,
            candidates.loc[candidates["route"].eq(route)],
            ground_truth,
            k_by_route={route: (5, 10) if route == "D1" else (5,)},
            final_outcomes=final.loc[final["route"].eq(route)],
            taxonomy_definitions=taxonomy_definitions(),
            saved_candidate_labels=labels.loc[labels["route"].eq(route)],
            saved_sample_outcomes=outcomes.loc[outcomes["route"].eq(route)],
            saved_branch_metrics={
                key: value
                for key, value in branch_metrics.items()
                if key.startswith(f"{route}|")
            },
            saved_native_taxonomy=route_native,
            saved_post_r7_taxonomy=post.loc[post["route"].eq(route)],
            saved_statistical_inputs=statistical_inputs.loc[
                statistical_inputs["route"].eq(route)
            ],
        )
        required_checks = {
            "candidate_labels",
            "sample_outcomes",
            "branch_metrics",
            "native_taxonomy",
            "post_r7_taxonomy",
            "statistical_inputs",
        }
        if result.get("status") != "PASS" or not all(
            result["exact_checks"].get(name) is True for name in required_checks
        ):
            raise PostprocessContractError(
                f"independent exact recompute failed for {route}"
            )
        checked_routes.append(route)
    return {
        "status": "PASS",
        "routes": checked_routes,
        "same_gt_geometry_recomputed": True,
        "per_sample_exact_match": True,
        "metrics_exact_match": True,
        "taxonomy_exact_match": True,
        "paired_inputs_exact_match": True,
        "report_or_ranker_modules_imported": False,
    }


def _verify_resume(
    root: Path,
    *,
    input_record: Mapping[str, Any],
    protocol_record: Mapping[str, Any],
    iterations: int,
    routes: Sequence[str],
) -> Path:
    route_path = root / ROUTE_STATUS_RELATIVE_PATH
    value = _json_object(route_path, label="route status")
    _verify_self_hash(value, label="route status")
    expected_routes = {
        route: ("COMPLETE" if route in routes else "UNRECOVERABLE_BLOCKER")
        for route in ROUTES
    }
    expected_status = "COMPLETE" if tuple(routes) == ROUTES else "PARTIAL"
    if (
        value.get("status") != expected_status
        or value.get("routes") != expected_routes
        or value.get("postprocess_inputs") != dict(input_record)
        or value.get("protocol_lock") != dict(protocol_record)
        or value.get("bootstrap_iterations") != int(iterations)
    ):
        raise PostprocessContractError("resume route-status identity differs")
    records = _collect_records(value.get("artifacts"), prefix="route_status.artifacts")
    if not records:
        raise PostprocessContractError("resume route status has no output artifacts")
    for label, record in records:
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PermissionError(f"resume artifact escapes run: {label}") from error
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != record.get("sha256")
        ):
            raise PostprocessContractError(f"resume artifact hash differs: {label}")
        if "bytes" in record and path.stat().st_size != int(record["bytes"]):
            raise PostprocessContractError(
                f"resume artifact byte count differs: {label}"
            )
    load_bound_tables(root)
    return route_path


def _close_postprocess_lifecycle(
    root: Path,
    *,
    route_status: Path,
    routes: Sequence[str],
) -> None:
    """Idempotently close execution and the full-run P7/P8 lifecycle.

    The route-status file is published before its completion sidecar.  Keeping
    this closure separate lets ``--resume`` repair a crash at that exact
    boundary without reopening any scientific frame.
    """

    complete_bulk_execution(root, route_status_manifest=route_status)
    if tuple(routes) != ROUTES:
        return
    pipeline = _json_object(root / "pipeline_status.json", label="pipeline status")
    observed = str(pipeline.get("status", ""))
    if observed == RunState.P6_D1_COUNTERFACTUAL_COMPLETE.value:
        transition_pipeline_status(
            root,
            RunState.P7_TAXONOMY_COMPLETE,
            first_incomplete_stage=RunState.P8_STATISTICS_COMPLETE.value,
        )
        observed = RunState.P7_TAXONOMY_COMPLETE.value
    if observed == RunState.P7_TAXONOMY_COMPLETE.value:
        transition_pipeline_status(
            root,
            RunState.P8_STATISTICS_COMPLETE,
            first_incomplete_stage=RunState.P9_GALLERIES_COMPLETE.value,
        )


def run_postprocess(
    run_dir: str | Path,
    *,
    protocol_lock: str | Path,
    input_manifest: str | Path | None = None,
    resume: bool = False,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> Path:
    """Evaluate and analyse six saved route/branch frames without feedback."""

    root = _assert_run_dir(run_dir)
    input_path, input_value = _load_input_manifest(root, input_manifest)
    routes = tuple(input_value["available_routes"])
    authority, pipeline = _verify_execution_authority(
        root,
        protocol_lock=protocol_lock,
        input_value=input_value,
        resume=resume,
    )
    # No frame, registry row, or GT grasp has been opened above this line.
    _verify_inputs_after_authority(
        root, input_value, expected_sample_count=expected_sample_count
    )
    input_record = artifact_record(input_path)
    protocol_record = artifact_record(protocol_lock)
    route_status = root / ROUTE_STATUS_RELATIVE_PATH
    if route_status.exists():
        if not resume:
            raise FileExistsError("postprocess output exists; pass --resume")
        verified = _verify_resume(
            root,
            input_record=input_record,
            protocol_record=protocol_record,
            iterations=bootstrap_iterations,
            routes=routes,
        )
        _close_postprocess_lifecycle(
            root, route_status=verified, routes=routes
        )
        return verified

    lock = verify_protocol_lock(root)
    selector_contracts = _validate_final_outcomes_authority(
        root=root,
        lock=lock,
        artifacts=input_value["artifacts"],
        routes=routes,
    )
    d1_blocker = None
    if routes != ROUTES:
        d1_blocker = _validate_d1_blocker(root, input_value["artifacts"])
    frames = _load_analysis_frames(
        root, input_value, expected_sample_count=expected_sample_count
    )
    denominator = set(frames["manifest"]["sample_id"])
    _assert_final_outcomes_exact(
        frames["final"],
        selector_contracts=selector_contracts,
        denominator=denominator,
    )
    registry_technical, mapping_table, registry_frame = _registry_technical_and_table(
        authority, denominator=denominator
    )
    if "annotation_suspect" not in registry_frame:
        raise PostprocessContractError("GT registry lacks annotation-suspect status")
    annotation = registry_frame[["sample_id", "annotation_suspect"]].copy()
    annotation["annotation_suspect"] = annotation["annotation_suspect"].astype(bool)
    frames["covariates"] = frames["covariates"].drop(
        columns=["annotation_suspect"], errors="ignore"
    ).merge(annotation, on="sample_id", how="left", validate="one_to_one")
    evaluator = _evaluator_record(lock)
    evaluator_path = Path(str(evaluator["path"])).expanduser().resolve()
    labels = evaluate_candidate_rows(
        frames["candidates"],
        frames["ground_truth"],
        evaluator_path=evaluator_path,
        evaluator_sha256=str(evaluator["sha256"]),
        on_invalid="mark",
    )
    outcomes, branch_metrics = _build_outcomes(
        labels,
        frames["manifest"],
        technical_by_branch=frames["technical"],
        registry_technical=registry_technical,
        routes=routes,
    )
    candidate_matches, candidate_mechanism_samples, candidate_mechanism_summary = (
        _candidate_mechanism_analysis(
            labels,
            frames["manifest"],
            routes=routes,
        )
    )
    paired_by_route: dict[str, pd.DataFrame] = {}
    paired_summaries: dict[str, dict[str, Any]] = {}
    for route in routes:
        paired, summary = _paired_route_frame(
            outcomes, route=route, covariates=frames["covariates"]
        )
        paired_by_route[route] = paired
        paired_summaries[route] = summary
    native, post = _taxonomy_tables(paired_by_route, frames["final"], routes=routes)
    statistical_inputs = _statistics_inputs(native, frames["manifest"])
    statistical_table, statistical_details = _statistical_tests(
        statistical_inputs, iterations=bootstrap_iterations, routes=routes
    )
    stratified = _stratified_table(native, labels, routes=routes)
    baseline_table = _load_baseline_attestation(
        input_value["artifacts"]["baseline_replay"],
        root=root,
        branch_metrics=branch_metrics,
        n=expected_sample_count,
        routes_to_check=routes,
    )
    independent = _independent_check(
        manifest=frames["manifest"],
        candidates=frames["candidates"],
        ground_truth=frames["ground_truth"],
        labels=labels,
        outcomes=outcomes,
        branch_metrics=branch_metrics,
        native=native,
        post=post,
        statistical_inputs=statistical_inputs,
        final=frames["final"],
        routes=routes,
    )

    output_paths = {
        "candidate_labels": atomic_parquet(
            labels, root / "07_candidate_tables/per_candidate_labels.parquet"
        ),
        "sample_outcomes": atomic_parquet(
            outcomes, root / "08_metrics/per_sample_outcomes.parquet"
        ),
        "paired_outcomes": atomic_parquet(
            pd.concat(paired_by_route.values(), ignore_index=True),
            root / "08_metrics/pred_vs_gt_per_sample.parquet",
        ),
        "native_taxonomy": atomic_parquet(
            native,
            root / "09_failure_taxonomy/native_failure_taxonomy_per_sample.parquet",
        ),
        "post_r7_taxonomy": atomic_parquet(
            post, root / "09_failure_taxonomy/post_r7_bottleneck_per_sample.parquet"
        ),
        "statistical_inputs": atomic_parquet(
            statistical_inputs, root / "10_statistics/statistical_inputs.parquet"
        ),
        "stratified_inputs": atomic_parquet(
            stratified, root / "11_stratified_analysis/stratified_results.parquet"
        ),
        "candidate_matches": atomic_parquet(
            candidate_matches,
            root / "07_candidate_tables/cross_branch_candidate_matches.parquet",
        ),
        "candidate_mechanism_per_sample": atomic_parquet(
            candidate_mechanism_samples,
            root / "08_metrics/candidate_pool_mechanism_per_sample.parquet",
        ),
    }
    mechanism_contract: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scientific_role": "observable candidate-pool mechanism evidence",
        "final_nms_pool_matching": "COMPLETE",
        "candidate_count_change": "OBSERVED_FINAL_NMS_POOL",
        "first_positive_rank_change": "OBSERVED_FINAL_NMS_POOL",
        "positive_candidate_introduction_or_removal": (
            "OBSERVED_BY_LOCKED_GEOMETRY_MATCHING"
        ),
        "crop_change": "UNKNOWN_SOURCE_ARTIFACT_ABSENT",
        "raw_candidate_count_change": "UNKNOWN_SOURCE_ARTIFACT_ABSENT",
        "dense_peak_vs_gate_attribution": (
            "UNKNOWN_UNLESS_SOURCE_EQUIVALENCE_KEY"
        ),
        "d1_filter_vs_new_sampling_attribution": (
            "UNKNOWN_UNLESS_SOURCE_EQUIVALENCE_KEY"
        ),
        "causal_claim_supported": False,
        "candidate_matches": artifact_record(output_paths["candidate_matches"]),
        "per_sample": artifact_record(output_paths["candidate_mechanism_per_sample"]),
    }
    mechanism_contract["content_sha256"] = canonical_sha256(mechanism_contract)
    output_paths["candidate_mechanism_contract"] = atomic_json(
        root / "08_metrics/CANDIDATE_MECHANISM_EVIDENCE.json",
        mechanism_contract,
    )
    d1_depth = _d1_depth_analysis(
        root=root,
        artifacts=input_value["artifacts"],
        native=native,
    )
    d1_depth_metrics: pd.DataFrame | None = None
    if d1_depth is not None:
        d1_depth_metrics, d1_depth_per_sample, d1_depth_taxonomy = d1_depth
        output_paths["d1_locked_depth_selector_metrics"] = atomic_csv(
            d1_depth_metrics,
            root / "08_metrics/d1_locked_depth_selector_metrics.csv",
        )
        output_paths["d1_locked_depth_bottleneck_per_sample"] = atomic_parquet(
            d1_depth_per_sample,
            root
            / "09_failure_taxonomy/d1_locked_depth_bottleneck_per_sample.parquet",
        )
        output_paths["d1_locked_depth_bottleneck_summary"] = atomic_csv(
            d1_depth_taxonomy,
            root / "09_failure_taxonomy/d1_locked_depth_bottleneck_summary.csv",
        )
    statistics_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "inference_scope": "descriptive counterfactual inference",
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "comparisons": statistical_details,
    }
    statistics_payload["content_sha256"] = canonical_sha256(statistics_payload)
    output_paths["statistics"] = atomic_json(
        root / "10_statistics/statistics.json", statistics_payload
    )
    independent["source_candidate_geometry"] = artifact_record(
        output_paths["candidate_labels"]
    )
    independent["content_sha256"] = canonical_sha256(independent)
    output_paths["independent_recompute"] = atomic_json(
        root / "16_independent_recompute/recomputed_metrics.json", independent
    )
    output_paths["independent_report"] = atomic_text(
        root / "16_independent_recompute/INDEPENDENT_RECOMPUTE.md",
        "# Independent recompute\n\nStatus: PASS. Saved geometry was independently "
        "raster-evaluated and exactly matched labels, per-sample outcomes, metrics, "
        "T/R taxonomies, and paired statistic inputs.\n",
    )

    tables = _tables(
        lock=lock,
        mapping_table=mapping_table,
        baseline_table=baseline_table,
        branch_metrics=branch_metrics,
        paired_summaries=paired_summaries,
        native=native,
        post=post,
        statistics=statistical_table,
        stratified=stratified,
        artifacts=input_value["artifacts"],
        root=root,
        routes=routes,
        candidate_mechanism_summary=candidate_mechanism_summary,
        d1_depth_metrics=d1_depth_metrics,
    )
    table_manifest = write_table_bundle(
        root,
        tables,
        source_bindings={
            "postprocess_inputs": input_record,
            "protocol_lock": protocol_record,
            "candidate_labels": artifact_record(output_paths["candidate_labels"]),
            "native_taxonomy": artifact_record(output_paths["native_taxonomy"]),
        },
    )
    output_paths["table_bundle"] = table_manifest
    output_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE" if routes == ROUTES else "PARTIAL",
        "scientific_role": SCIENTIFIC_ROLE,
        "N": expected_sample_count,
        "routes": {
            route: ("COMPLETE" if route in routes else "UNRECOVERABLE_BLOCKER")
            for route in ROUTES
        },
        "branches": list(BRANCHES),
        "same_gt_evaluator": evaluator,
        "strict_iou": ">0.25",
        "periodic_angle": "<=30deg",
        "no_output_in_denominator": True,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "training_or_selection_feedback_allowed": False,
        "postprocess_inputs": input_record,
        "protocol_lock": protocol_record,
        "artifacts": {
            label: artifact_record(path) for label, path in output_paths.items()
        },
    }
    if d1_blocker is not None:
        output_manifest["d1_blocker"] = dict(input_value["artifacts"]["d1_blocker"])
    output_manifest["content_sha256"] = canonical_sha256(output_manifest)
    manifest_path = atomic_json(root / OUTPUT_MANIFEST_RELATIVE_PATH, output_manifest)
    output_manifest_record = artifact_record(manifest_path)
    route_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE" if routes == ROUTES else "PARTIAL",
        "routes": {
            route: ("COMPLETE" if route in routes else "UNRECOVERABLE_BLOCKER")
            for route in ROUTES
        },
        "postprocess_inputs": input_record,
        "protocol_lock": protocol_record,
        "bootstrap_iterations": int(bootstrap_iterations),
        "artifacts": {
            **output_manifest["artifacts"],
            "postprocess_manifest": output_manifest_record,
        },
    }
    route_payload["content_sha256"] = canonical_sha256(route_payload)
    atomic_json(route_status, route_payload)
    del pipeline
    _close_postprocess_lifecycle(root, route_status=route_status, routes=routes)
    return route_status


__all__ = [
    "EXPECTED_SAMPLE_COUNT",
    "INPUT_MANIFEST_RELATIVE_PATH",
    "PostprocessContractError",
    "ROUTES",
    "run_postprocess",
    "write_final_outcomes_authority",
    "write_postprocess_input_manifest",
    "write_route_frame_manifest",
]
