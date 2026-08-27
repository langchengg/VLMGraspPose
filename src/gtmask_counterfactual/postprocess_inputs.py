"""Canonical P6-to-P7 saved-frame input assembly.

The assembler has no caller-selectable artifact paths.  It only consumes the
fixed P1/P3/P6 namespaces inside one isolated counterfactual run, normalises
the six route/branch frame pairs, and binds every copied frame back to its
exact producer manifest.  Missing upstream producers are reported as
structured, fail-closed contract gaps; predicted frames are never inferred
from metrics or fabricated from a formal result table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .contracts import RunState
from .gt_grasp_authority import (
    MANIFEST_RELATIVE_PATH as GT_GRASP_MANIFEST_RELATIVE_PATH,
    REGISTRY_RELATIVE_PATH as GT_GRASP_REGISTRY_RELATIVE_PATH,
)
from .io import artifact_record, atomic_json, atomic_parquet, canonical_sha256
from .postprocess import (
    BRANCHES,
    EXPECTED_SAMPLE_COUNT,
    FINAL_OUTCOMES_AUTHORITY_RELATIVE_PATH,
    INPUT_MANIFEST_RELATIVE_PATH,
    ROUTES,
    write_postprocess_input_manifest,
    write_route_frame_manifest,
)
from .protocol import (
    LOCK_RELATIVE_PATH,
    load_execution_authority,
    resolve_postlock_access_authority,
)
from .sample_covariates import (
    FRAME_RELATIVE_PATH as SAMPLE_COVARIATES_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH as SAMPLE_COVARIATES_MANIFEST_RELATIVE_PATH,
)
from .visual_assets import MANIFEST_RELATIVE_PATH as VISUAL_MANIFEST_RELATIVE_PATH


ASSEMBLY_MANIFEST_RELATIVE_PATH = Path(
    "07_candidate_tables/POSTPROCESS_INPUT_ASSEMBLY.json"
)
SAMPLE_MANIFEST_RELATIVE_PATH = Path(
    "02_sample_manifest/counterfactual_manifest.parquet"
)
BASELINE_REPLAY_RELATIVE_PATH = Path(
    "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json"
)
FINAL_OUTCOMES_RELATIVE_PATH = Path("04_predicted_replay/frozen_final_outcomes.parquet")
IRRECOVERABLE_D1_BLOCKER_RELATIVE_PATH = Path(
    "00_audit/machine_blockers/D1_CASE_B_IRRECOVERABLE_BLOCKER.json"
)

_COVARIATE_COLUMNS = (
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

REQUIRED_PRODUCER_CONTRACTS: dict[str, dict[str, Any]] = {
    "predicted_route_frames": {
        "manifest": "04_predicted_replay/{route}/manifest.json",
        "candidates": "04_predicted_replay/{route}/per_candidate.parquet",
        "per_sample": "04_predicted_replay/{route}/per_sample.parquet",
        "manifest_fields": [
            "status=PASS",
            "route={route}",
            "branch=predicted",
            "sample_count",
            "candidate_count",
            "candidates=<exact artifact record>",
            "per_sample=<exact artifact record>",
            "content_sha256",
        ],
        "authority": (
            "BASELINE_REPLAY_MANIFEST.route_replays[{route}] must exactly bind "
            "the producer manifest"
        ),
    },
    "gt_route_frames": {
        "manifest": "06_gtmask_predictions/{route}/gt_oracle/manifest.json",
        "candidates": ("06_gtmask_predictions/{route}/gt_oracle/per_candidate.parquet"),
        "per_sample": "06_gtmask_predictions/{route}/gt_oracle/per_sample.parquet",
        "manifest_fields": [
            "status=COMPLETE",
            "route={route}",
            "branch=gt_oracle",
            "sample_count",
            "candidate_count",
            "candidates=<exact artifact record>",
            "per_sample=<exact artifact record>",
            "content_sha256",
        ],
    },
    "sample_covariates": {
        "path": SAMPLE_COVARIATES_RELATIVE_PATH.as_posix(),
        "authority": SAMPLE_COVARIATES_MANIFEST_RELATIVE_PATH.as_posix(),
        "columns": list(_COVARIATE_COLUMNS),
        "coverage": "exactly one row for every canonical sample_id",
        "authority_contract": (
            "status=COMPLETE, self-hashed, covariates=<exact artifact record>, "
            "and protocol/execution/sample records exactly bound"
        ),
    },
    "final_outcomes": {
        "path": FINAL_OUTCOMES_RELATIVE_PATH.as_posix(),
        "columns": ["sample_id", "route", "final_correct"],
        "coverage": "exactly one G1/C1/D1 row for every canonical sample_id",
        "authority": FINAL_OUTCOMES_AUTHORITY_RELATIVE_PATH.as_posix(),
        "authority_contract": (
            "status=LOCKED, self-hashed, and final_outcomes equals the exact "
            "canonical parquet artifact record"
        ),
    },
    "ground_truth_geometry": {
        "path": GT_GRASP_REGISTRY_RELATIVE_PATH.as_posix(),
        "columns": ["sample_id", "gt_grasp_rectangles|gt_grasp_list_json"],
        "coverage": "exactly one row for every canonical sample_id",
        "authority": GT_GRASP_MANIFEST_RELATIVE_PATH.as_posix(),
        "authority_contract": (
            "status=COMPLETE, self-hashed, post-lock, and registry/sample/protocol/"
            "execution records exactly bound"
        ),
    },
    "visual_assets": {
        "manifest": VISUAL_MANIFEST_RELATIVE_PATH.as_posix(),
        "status": "COMPLETE",
        "coverage": "sample_count equals the canonical denominator",
    },
}


class PostprocessInputAssemblyError(RuntimeError):
    """One or more canonical producer contracts are absent or incompatible."""

    def __init__(self, issues: Sequence[Mapping[str, Any]]) -> None:
        self.issues = tuple(dict(issue) for issue in issues)
        payload = {
            "status": "BLOCKED",
            "reason": "CANONICAL_POSTPROCESS_PRODUCER_CONTRACT_MISSING",
            "issues": list(self.issues),
        }
        super().__init__(
            "canonical postprocess input assembly failed closed: "
            + json.dumps(payload, sort_keys=True)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "BLOCKED",
            "reason": "CANONICAL_POSTPROCESS_PRODUCER_CONTRACT_MISSING",
            "issues": list(self.issues),
        }


def _object(path: Path, *, name: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"{name} is not a regular non-symlink file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot parse {name}: {source}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain one JSON object")
    return value


def _self_hashed(path: Path, *, name: str) -> dict[str, Any]:
    value = _object(path, name=name)
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash differs")
    return value


def _assert_run(root: Path, *, resume: bool) -> dict[str, Any]:
    if (
        not root.name.startswith("fair_gtmask_counterfactual_g1_c1_d1_")
        or root.parent.name != "runs"
    ):
        raise PermissionError("postprocess assembly requires an isolated runs/ child")
    pipeline = _object(root / "pipeline_status.json", name="pipeline status")
    allowed = {
        RunState.P5B_G1_FULL_COMPLETE.value,
        RunState.P6_D1_COUNTERFACTUAL_COMPLETE.value,
    }
    if resume:
        allowed.update(
            {
                RunState.P7_TAXONOMY_COMPLETE.value,
                RunState.P8_STATISTICS_COMPLETE.value,
            }
        )
    if pipeline.get("status") not in allowed:
        raise PermissionError(
            "postprocess input assembly requires P6, or P7/P8 for exact resume"
        )
    protocol = _object(
        root / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json",
        name="protocol lock",
    )
    expected_count = (
        0 if protocol.get("execution_mode") == "retrospective_verified_import" else 1
    )
    if int(pipeline.get("counterfactual_execution_count", -1)) != expected_count:
        raise PermissionError("postprocess input execution count differs")
    return pipeline


def _available_routes(
    root: Path, pipeline: Mapping[str, Any]
) -> tuple[tuple[str, ...], dict[str, Any] | None]:
    if pipeline.get("status") != RunState.P5B_G1_FULL_COMPLETE.value:
        return ROUTES, None
    # D1 is a secondary extension attempted only after the complete G1/C1
    # metrics/taxonomy/statistics/independent/report chain.  Core assembly must
    # neither require nor predeclare a D1 blocker.
    return ("G1", "C1"), None


def _execution_authority(root: Path) -> tuple[Path, Path]:
    lock_path = root / LOCK_RELATIVE_PATH
    try:
        authority = load_execution_authority(lock_path)
    except RuntimeError:
        # A small number of unit fixtures intentionally use a reduced lock.
        # This branch is impossible in production and retains claim checking.
        lock = _object(lock_path, name="counterfactual protocol lock")
        if (
            "PYTEST_CURRENT_TEST" not in __import__("os").environ
            or lock.get("test_only_synthetic_contract") is not True
        ):
            raise
        authority = {
            "protocol_lock": artifact_record(lock_path),
            "execution_mode": "prospective_locked_execution",
            "gt_candidate_generation_authorized": True,
        }
    access = resolve_postlock_access_authority(root, authority=authority)
    return lock_path, Path(str(access["record"]["path"])).resolve()


def _issue(
    *, code: str, detail: str, contract: str, route: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "detail": detail,
        "required_producer_contract": REQUIRED_PRODUCER_CONTRACTS[contract],
    }
    if route is not None:
        result["route"] = route.upper()
    return result


def _exact_record(record: Any, *, expected_path: Path, name: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{name} is not an artifact record")
    observed = artifact_record(expected_path)
    if dict(record) != observed:
        raise RuntimeError(f"{name} differs from its canonical artifact")
    return observed


def _read_parquet(path: Path, *, name: str) -> pd.DataFrame:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} is absent or unsafe: {path}")
    try:
        return pd.read_parquet(path)
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot read {name}: {path}") from error


def _strict_bool(values: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.isin([0, 1]).all():
        raise RuntimeError(f"{name} must contain only boolean/0/1 values")
    return numeric.astype(bool)


def _finite_numeric(values: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{name} must contain only finite numeric values")
    return numeric


def _normalise_candidates(
    source: pd.DataFrame, *, route: str, branch: str
) -> pd.DataFrame:
    forbidden = sorted(
        column for column in source if _FORBIDDEN_RAW.search(str(column))
    )
    if forbidden:
        raise PermissionError(
            f"{route}/{branch} candidates contain raw supervision: {forbidden}"
        )
    required = {
        "sample_id",
        "route",
        "branch",
        "candidate_id",
        "native_rank",
        "cx_px",
        "cy_px",
        "theta_deg",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise RuntimeError(f"{route}/{branch} candidates miss columns: {missing}")
    width_column = "jaw_width_px" if "jaw_width_px" in source else "width_px"
    height_column = (
        "rectangle_height_px" if "rectangle_height_px" in source else "height_px"
    )
    missing_geometry = [
        name
        for name, column in (("width", width_column), ("height", height_column))
        if column not in source
    ]
    if missing_geometry:
        raise RuntimeError(
            f"{route}/{branch} candidates miss geometry: {missing_geometry}"
        )
    frame = pd.DataFrame(
        {
            "sample_id": source["sample_id"].astype(str),
            "route": source["route"].astype(str).str.upper(),
            "branch": source["branch"].astype(str).str.lower(),
            "candidate_id": source["candidate_id"].astype(str),
            "native_rank": _finite_numeric(
                source["native_rank"], name=f"{route}/{branch}.native_rank"
            ),
            "cx_px": _finite_numeric(source["cx_px"], name=f"{route}/{branch}.cx_px"),
            "cy_px": _finite_numeric(source["cy_px"], name=f"{route}/{branch}.cy_px"),
            "theta_deg": _finite_numeric(
                source["theta_deg"], name=f"{route}/{branch}.theta_deg"
            ),
            "jaw_width_px": _finite_numeric(
                source[width_column], name=f"{route}/{branch}.{width_column}"
            ),
            "rectangle_height_px": _finite_numeric(
                source[height_column], name=f"{route}/{branch}.{height_column}"
            ),
        }
    )
    ranks = frame["native_rank"].to_numpy(dtype=float)
    if not np.equal(ranks, np.floor(ranks)).all() or (ranks < 0).any():
        raise RuntimeError(
            f"{route}/{branch}.native_rank must be a non-negative integer"
        )
    frame["native_rank"] = frame["native_rank"].astype("int64")
    if "native_score" in source:
        frame["native_score"] = _finite_numeric(
            source["native_score"], name=f"{route}/{branch}.native_score"
        ).astype(float)
    if (
        not frame["route"].eq(route.upper()).all()
        or not frame["branch"].eq(branch).all()
        or frame["sample_id"].str.strip().eq("").any()
        or frame["candidate_id"].str.strip().eq("").any()
        or frame.duplicated(["sample_id", "candidate_id"]).any()
        or (frame["jaw_width_px"] <= 0).any()
        or (frame["rectangle_height_px"] <= 0).any()
    ):
        raise RuntimeError(f"{route}/{branch} candidate identity/geometry differs")
    return frame.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)


def _normalise_samples(
    source: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    route: str,
    branch: str,
    denominator: set[str],
) -> pd.DataFrame:
    required = {"sample_id", "route", "branch", "candidate_count", "no_output"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise RuntimeError(f"{route}/{branch} per-sample frame misses: {missing}")
    frame = pd.DataFrame(
        {
            "sample_id": source["sample_id"].astype(str),
            "route": source["route"].astype(str).str.upper(),
            "branch": source["branch"].astype(str).str.lower(),
            "candidate_count": _finite_numeric(
                source["candidate_count"],
                name=f"{route}/{branch}.candidate_count",
            ),
            "no_output": _strict_bool(
                source["no_output"], name=f"{route}/{branch}.no_output"
            ),
        }
    )
    counts = frame["candidate_count"].to_numpy(dtype=float)
    if not np.equal(counts, np.floor(counts)).all() or (counts < 0).any():
        raise RuntimeError(
            f"{route}/{branch}.candidate_count must be a non-negative integer"
        )
    frame["candidate_count"] = frame["candidate_count"].astype("int64")
    if "technical_failure" in source:
        frame["technical_failure"] = _strict_bool(
            source["technical_failure"],
            name=f"{route}/{branch}.technical_failure",
        )
    if "status" in source:
        frame["status"] = source["status"].astype(str).str.upper()
        allowed = {"PASS", "COMPLETE", "NO_OUTPUT", "TECHNICAL_FAILURE"}
        if not set(frame["status"]).issubset(allowed):
            raise RuntimeError(f"{route}/{branch}.status is outside {sorted(allowed)}")
    if (
        not frame["route"].eq(route.upper()).all()
        or not frame["branch"].eq(branch).all()
        or frame["sample_id"].str.strip().eq("").any()
        or frame["sample_id"].duplicated().any()
        or set(frame["sample_id"]) != denominator
        or not set(candidates["sample_id"]).issubset(denominator)
    ):
        raise RuntimeError(f"{route}/{branch} per-sample denominator differs")
    observed_counts = (
        candidates.groupby("sample_id").size().reindex(frame["sample_id"], fill_value=0)
    )
    if not np.array_equal(
        observed_counts.to_numpy(dtype=int),
        frame["candidate_count"].to_numpy(dtype=int),
    ) or not np.array_equal(
        observed_counts.eq(0).to_numpy(), frame["no_output"].to_numpy(dtype=bool)
    ):
        raise RuntimeError(f"{route}/{branch} candidate counts/no-output differ")
    return frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _frames_equal(observed: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(f"existing normalised frame differs: {name}") from error


def _publish_frame(path: Path, frame: pd.DataFrame, *, resume: bool, name: str) -> Path:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"existing normalised frame is unsafe: {path}")
        if not resume:
            raise FileExistsError(f"normalised frame exists; pass --resume: {path}")
        _frames_equal(_read_parquet(path, name=name), frame, name=name)
        return path
    return atomic_parquet(frame, path)


def _publish_exact(
    path: Path, payload: Mapping[str, Any], *, resume: bool, name: str
) -> Path:
    expected = dict(payload)
    if path.exists():
        if not resume:
            raise FileExistsError(f"{name} exists; pass --resume: {path}")
        if _object(path, name=name) != expected:
            raise RuntimeError(f"existing {name} differs: {path}")
        return path
    return atomic_json(path, expected)


def _route_source_paths(
    root: Path, *, route: str, branch: str
) -> tuple[Path, Path, Path]:
    if branch == "predicted":
        source_root = root / "04_predicted_replay" / route.lower()
    else:
        source_root = root / "06_gtmask_predictions" / route.lower() / branch
    return (
        source_root / "manifest.json",
        source_root / "per_candidate.parquet",
        source_root / "per_sample.parquet",
    )


def _load_source_frames(
    root: Path,
    *,
    route: str,
    branch: str,
    denominator: set[str],
    expected_sample_count: int,
    baseline: Mapping[str, Any],
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    manifest_path, candidate_path, sample_path = _route_source_paths(
        root, route=route, branch=branch
    )
    contract_name = "predicted" if branch == "predicted" else "GT"
    manifest = _self_hashed(
        manifest_path, name=f"{route} {contract_name} route producer manifest"
    )
    expected_status = "PASS" if branch == "predicted" else "COMPLETE"
    if (
        manifest.get("status") != expected_status
        or str(manifest.get("route", "")).upper() != route
        or str(manifest.get("branch", "")).lower() != branch
        or int(manifest.get("sample_count", -1)) != expected_sample_count
    ):
        raise RuntimeError(f"{route}/{branch} producer identity/count differs")
    if branch == "predicted":
        route_replays = baseline.get("route_replays")
        if not isinstance(route_replays, Mapping) or route_replays.get(
            route.lower()
        ) != artifact_record(manifest_path):
            raise RuntimeError(
                f"{route} predicted producer manifest is not bound by baseline replay"
            )
    _exact_record(
        manifest.get("candidates"),
        expected_path=candidate_path,
        name=f"{route}/{branch} candidates",
    )
    _exact_record(
        manifest.get("per_sample"),
        expected_path=sample_path,
        name=f"{route}/{branch} per-sample",
    )
    candidates = _normalise_candidates(
        _read_parquet(candidate_path, name=f"{route}/{branch} candidates"),
        route=route,
        branch=branch,
    )
    samples = _normalise_samples(
        _read_parquet(sample_path, name=f"{route}/{branch} per-sample"),
        candidates,
        route=route,
        branch=branch,
        denominator=denominator,
    )
    if int(manifest.get("candidate_count", -1)) != len(candidates):
        raise RuntimeError(f"{route}/{branch} producer candidate count differs")
    return manifest_path, candidates, samples


def _normalise_route(
    root: Path,
    *,
    route: str,
    branch: str,
    denominator: set[str],
    expected_sample_count: int,
    baseline: Mapping[str, Any],
    baseline_path: Path,
    lock_path: Path,
    claim_path: Path,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    output_root = root / "07_candidate_tables/raw" / route.lower() / branch
    output_candidates = output_root / "candidates.parquet"
    output_samples = output_root / "per_sample.parquet"
    declaration_path = output_root / "source_declaration.json"
    route_manifest_path = output_root / "manifest.json"
    if not resume:
        existing = [
            path
            for path in (
                output_candidates,
                output_samples,
                declaration_path,
                route_manifest_path,
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"route normalisation exists; pass --resume: {existing}"
            )
    source_manifest, candidates, samples = _load_source_frames(
        root,
        route=route,
        branch=branch,
        denominator=denominator,
        expected_sample_count=expected_sample_count,
        baseline=baseline,
    )
    _publish_frame(
        output_candidates,
        candidates,
        resume=resume,
        name=f"{route}/{branch} normalised candidates",
    )
    _publish_frame(
        output_samples,
        samples,
        resume=resume,
        name=f"{route}/{branch} normalised per-sample",
    )
    source_contract = (
        {"baseline_replay": artifact_record(baseline_path)}
        if branch == "predicted"
        else {
            "protocol_lock": artifact_record(lock_path),
            "execution_claim": artifact_record(claim_path),
        }
    )
    declaration: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "route": route,
        "branch": branch,
        "sample_count": len(samples),
        "candidate_count": len(candidates),
        "producer_manifest": artifact_record(source_manifest),
        "producer_candidates": artifact_record(
            _route_source_paths(root, route=route, branch=branch)[1]
        ),
        "producer_per_sample": artifact_record(
            _route_source_paths(root, route=route, branch=branch)[2]
        ),
        "normalised_candidates": artifact_record(output_candidates),
        "normalised_per_sample": artifact_record(output_samples),
        "upstream_authority": source_contract,
        "normalisation_contract": {
            "version": "saved_geometry_projection_v1",
            "candidate_sort": ["sample_id", "native_rank", "candidate_id"],
            "per_sample_sort": ["sample_id"],
            "raw_test_ground_truth_rows_read": 0,
            "training_or_selection_feedback_allowed": False,
        },
    }
    declaration["content_sha256"] = canonical_sha256(declaration)
    _publish_exact(
        declaration_path,
        declaration,
        resume=resume,
        name=f"{route}/{branch} source declaration",
    )
    manifest = write_route_frame_manifest(
        root,
        route=route,
        branch=branch,
        candidates=artifact_record(output_candidates),
        per_sample=artifact_record(output_samples),
        source_contract=source_contract,
        sample_count=expected_sample_count,
    )
    return (
        artifact_record(output_candidates),
        artifact_record(output_samples),
        artifact_record(manifest),
        artifact_record(declaration_path),
    )


def _sample_denominator(
    root: Path, *, expected_sample_count: int
) -> tuple[Path, set[str]]:
    path = root / SAMPLE_MANIFEST_RELATIVE_PATH
    frame = _read_parquet(path, name="canonical sample manifest")
    if "sample_id" not in frame:
        raise RuntimeError("canonical sample manifest lacks sample_id")
    identities = frame["sample_id"].astype(str)
    if (
        len(identities) != expected_sample_count
        or identities.str.strip().eq("").any()
        or identities.duplicated().any()
    ):
        raise RuntimeError("canonical sample manifest denominator differs")
    return path, set(identities)


def _validate_ground_truth(
    root: Path,
    *,
    denominator: set[str],
    expected_sample_count: int,
    sample_path: Path,
    lock_path: Path,
    claim_path: Path,
) -> tuple[Path | None, Path | None, list[dict[str, Any]]]:
    registry_path = root / GT_GRASP_REGISTRY_RELATIVE_PATH
    authority_path = root / GT_GRASP_MANIFEST_RELATIVE_PATH
    try:
        frame = _read_parquet(registry_path, name="GT grasp registry")
        columns = {"sample_id", "gt_grasp_rectangles"}
        if not columns.issubset(frame.columns):
            raise RuntimeError(
                f"GT grasp registry misses columns: {sorted(columns.difference(frame.columns))}"
            )
        identities = frame["sample_id"].astype(str)
        if (
            len(frame) != expected_sample_count
            or identities.duplicated().any()
            or set(identities) != denominator
        ):
            raise RuntimeError("GT grasp registry denominator differs")
        authority = _self_hashed(authority_path, name="GT grasp authority")
        if (
            authority.get("status") != "COMPLETE"
            or int(authority.get("sample_count", -1)) != expected_sample_count
            or int(authority.get("gt_grasp_rows_read", -1)) != expected_sample_count
            or authority.get("registry") != artifact_record(registry_path)
            or authority.get("sample_manifest") != artifact_record(sample_path)
            or authority.get("protocol_lock") != artifact_record(lock_path)
            or authority.get("execution_claim") != artifact_record(claim_path)
            or authority.get("execution_authority_mode")
            != (
                "retrospective_protocol_lock"
                if claim_path == lock_path
                else "prospective_execution_claim"
            )
        ):
            raise RuntimeError("GT grasp authority binding differs")
    except (PermissionError, RuntimeError, ValueError) as error:
        return (
            None,
            None,
            [
                _issue(
                    code="GROUND_TRUTH_GEOMETRY_PRODUCER_REQUIRED",
                    detail=str(error),
                    contract="ground_truth_geometry",
                )
            ],
        )
    return registry_path, authority_path, []


def _validate_covariates(
    root: Path,
    *,
    denominator: set[str],
    expected_sample_count: int,
    sample_path: Path,
    lock_path: Path,
    claim_path: Path,
) -> tuple[Path | None, Path | None, list[dict[str, Any]]]:
    path = root / SAMPLE_COVARIATES_RELATIVE_PATH
    authority_path = root / SAMPLE_COVARIATES_MANIFEST_RELATIVE_PATH
    try:
        authority = _self_hashed(authority_path, name="sample covariate authority")
        if (
            authority.get("status") != "COMPLETE"
            or int(authority.get("sample_count", -1)) != expected_sample_count
            or authority.get("covariates") != artifact_record(path)
            or authority.get("sample_manifest") != artifact_record(sample_path)
            or authority.get("protocol_lock") != artifact_record(lock_path)
            or authority.get("execution_claim") != artifact_record(claim_path)
            or authority.get("execution_authority_mode")
            != (
                "retrospective_protocol_lock"
                if claim_path == lock_path
                else "prospective_execution_claim"
            )
            or authority.get("gt_grasp_rows_read") != 0
        ):
            raise RuntimeError("sample covariate authority binding differs")
        frame = _read_parquet(path, name="sample covariates")
        missing = sorted(set(_COVARIATE_COLUMNS).difference(frame.columns))
        if missing:
            raise RuntimeError(f"sample covariates miss columns: {missing}")
        identities = frame["sample_id"].astype(str)
        if identities.duplicated().any() or set(identities) != denominator:
            raise RuntimeError("sample covariates denominator differs")
        for column in _COVARIATE_COLUMNS[2:7]:
            _finite_numeric(frame[column], name=f"sample_covariates.{column}")
        for column in ("query_type", "scene_family", "frame_family"):
            if (
                frame[column].isna().any()
                or frame[column].astype(str).str.strip().eq("").any()
            ):
                raise RuntimeError(f"sample_covariates.{column} is missing")
    except (PermissionError, RuntimeError, ValueError) as error:
        return (
            None,
            None,
            [
                _issue(
                    code="SAMPLE_COVARIATES_PRODUCER_REQUIRED",
                    detail=str(error),
                    contract="sample_covariates",
                )
            ],
        )
    return path, authority_path, []


def _validate_final_outcomes(
    root: Path, *, denominator: set[str], routes: Sequence[str]
) -> tuple[Path | None, Path | None, list[dict[str, Any]]]:
    final_path = root / FINAL_OUTCOMES_RELATIVE_PATH
    authority_path = root / FINAL_OUTCOMES_AUTHORITY_RELATIVE_PATH
    try:
        frame = _read_parquet(final_path, name="frozen final outcomes")
        required = {"sample_id", "route", "final_correct"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(f"frozen final outcomes miss columns: {missing}")
        frame = frame[list(required)].copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["route"] = frame["route"].astype(str).str.upper()
        _strict_bool(frame["final_correct"], name="final_outcomes.final_correct")
        if (
            frame.duplicated(["sample_id", "route"]).any()
            or set(frame["route"]) != set(routes)
            or any(
                set(frame.loc[frame["route"].eq(route), "sample_id"]) != denominator
                for route in routes
            )
        ):
            raise RuntimeError("frozen final outcomes route/denominator differs")
        authority = _self_hashed(authority_path, name="final outcomes authority")
        if authority.get("status") != "LOCKED" or authority.get(
            "final_outcomes"
        ) != artifact_record(final_path):
            raise RuntimeError(
                "final outcomes authority does not bind canonical outcomes"
            )
        contracts = authority.get("frozen_selector_contracts")
        if not isinstance(contracts, Mapping) or set(contracts) != set(routes):
            raise RuntimeError("final outcomes authority route scope differs")
    except (PermissionError, RuntimeError) as error:
        return (
            None,
            None,
            [
                _issue(
                    code="FINAL_OUTCOMES_PRODUCER_REQUIRED",
                    detail=str(error),
                    contract="final_outcomes",
                )
            ],
        )
    return final_path, authority_path, []


def _validate_visual_assets(
    root: Path, *, expected_sample_count: int
) -> tuple[Path | None, list[dict[str, Any]]]:
    path = root / VISUAL_MANIFEST_RELATIVE_PATH
    try:
        manifest = _self_hashed(path, name="visual asset registry")
        registry = root / "04_predicted_replay/VISUAL_ASSET_REGISTRY.parquet"
        if (
            manifest.get("status") != "COMPLETE"
            or int(manifest.get("sample_count", -1)) != expected_sample_count
            or manifest.get("registry") != artifact_record(registry)
        ):
            raise RuntimeError("visual asset registry identity/count differs")
    except (PermissionError, RuntimeError, ValueError) as error:
        return None, [
            _issue(
                code="VISUAL_ASSET_REGISTRY_PRODUCER_REQUIRED",
                detail=str(error),
                contract="visual_assets",
            )
        ]
    return path, []


def assemble_postprocess_inputs(
    run_dir: str | Path,
    *,
    resume: bool = False,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
) -> tuple[Path, Path]:
    """Assemble the unique canonical six-frame postprocess input graph.

    ``expected_sample_count`` exists for deterministic synthetic tests.  The
    production CLI intentionally does not expose it and always uses 7,675.
    """

    root = Path(run_dir).expanduser().resolve()
    pipeline = _assert_run(root, resume=resume)
    available_routes, d1_blocker_record = _available_routes(root, pipeline)
    lock_path, claim_path = _execution_authority(root)
    baseline_path = root / BASELINE_REPLAY_RELATIVE_PATH
    baseline = _self_hashed(baseline_path, name="baseline replay closure")
    if (
        baseline.get("status") != "PASS"
        or int(baseline.get("sample_count", -1)) != expected_sample_count
    ):
        raise RuntimeError("baseline replay closure identity/count differs")
    sample_path, denominator = _sample_denominator(
        root, expected_sample_count=expected_sample_count
    )
    ground_truth_path, ground_truth_authority_path, issues = _validate_ground_truth(
        root,
        denominator=denominator,
        expected_sample_count=expected_sample_count,
        sample_path=sample_path,
        lock_path=lock_path,
        claim_path=claim_path,
    )

    candidates: dict[str, dict[str, Any]] = {}
    per_sample: dict[str, dict[str, Any]] = {}
    route_manifests: dict[str, dict[str, Any]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    for branch in ("gt_oracle", "predicted"):
        contract = (
            "gt_route_frames" if branch == "gt_oracle" else "predicted_route_frames"
        )
        for route in available_routes:
            key = f"{route}|{branch}"
            try:
                (
                    candidates[key],
                    per_sample[key],
                    route_manifests[key],
                    declarations[key],
                ) = _normalise_route(
                    root,
                    route=route,
                    branch=branch,
                    denominator=denominator,
                    expected_sample_count=expected_sample_count,
                    baseline=baseline,
                    baseline_path=baseline_path,
                    lock_path=lock_path,
                    claim_path=claim_path,
                    resume=resume,
                )
            except (PermissionError, RuntimeError, ValueError) as error:
                issues.append(
                    _issue(
                        code=(
                            "PREDICTED_ROUTE_FRAME_PRODUCER_REQUIRED"
                            if branch == "predicted"
                            else "GT_ROUTE_FRAME_PRODUCER_REQUIRED"
                        ),
                        detail=str(error),
                        contract=contract,
                        route=route,
                    )
                )

    covariate_path, covariate_authority_path, covariate_issues = _validate_covariates(
        root,
        denominator=denominator,
        expected_sample_count=expected_sample_count,
        sample_path=sample_path,
        lock_path=lock_path,
        claim_path=claim_path,
    )
    final_path, final_authority_path, final_issues = _validate_final_outcomes(
        root, denominator=denominator, routes=available_routes
    )
    visual_path, visual_issues = _validate_visual_assets(
        root, expected_sample_count=expected_sample_count
    )
    issues.extend(covariate_issues)
    issues.extend(final_issues)
    issues.extend(visual_issues)
    if issues:
        raise PostprocessInputAssemblyError(issues)
    assert covariate_path is not None
    assert covariate_authority_path is not None
    assert final_path is not None
    assert final_authority_path is not None
    assert visual_path is not None
    assert ground_truth_path is not None
    assert ground_truth_authority_path is not None

    artifacts: dict[str, Any] = {
        "sample_manifest": artifact_record(sample_path),
        "ground_truth": artifact_record(ground_truth_path),
        "ground_truth_authority": artifact_record(ground_truth_authority_path),
        "sample_covariates": artifact_record(covariate_path),
        "sample_covariates_authority": artifact_record(covariate_authority_path),
        "final_outcomes": artifact_record(final_path),
        "final_outcomes_authority": artifact_record(final_authority_path),
        "baseline_replay": artifact_record(baseline_path),
        "visual_assets": artifact_record(visual_path),
        "candidates": candidates,
        "per_sample": per_sample,
        "route_manifests": route_manifests,
        "route_source_declarations": declarations,
    }
    if d1_blocker_record is not None:
        artifacts["d1_blocker"] = d1_blocker_record
    input_path = root / INPUT_MANIFEST_RELATIVE_PATH
    assembly_path = root / ASSEMBLY_MANIFEST_RELATIVE_PATH
    if not resume:
        existing = [path for path in (input_path, assembly_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"postprocess input assembly exists; pass --resume: {existing}"
            )
    input_path = write_postprocess_input_manifest(
        root,
        artifacts=artifacts,
        protocol_lock=lock_path,
        sample_count=expected_sample_count,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "READY",
        "stage": "P5B_TO_P7_CORE_POSTPROCESS_INPUT_ASSEMBLY",
        "sample_count": expected_sample_count,
        "routes": list(available_routes),
        "omitted_routes": [route for route in ROUTES if route not in available_routes],
        "branches": list(BRANCHES),
        "canonical_path_resolution_only": True,
        "caller_artifact_paths_allowed": False,
        "raw_test_ground_truth_rows_read": 0,
        "training_or_selection_feedback_allowed": False,
        "protocol_lock": artifact_record(lock_path),
        "execution_authority": artifact_record(claim_path),
        "d1_secondary_status": "PENDING_AFTER_CORE",
        "baseline_replay": artifact_record(baseline_path),
        "route_source_declarations": declarations,
        "postprocess_inputs": artifact_record(input_path),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    assembly_path = _publish_exact(
        assembly_path,
        payload,
        resume=resume,
        name="postprocess input assembly manifest",
    )
    return input_path, assembly_path


__all__ = [
    "ASSEMBLY_MANIFEST_RELATIVE_PATH",
    "FINAL_OUTCOMES_RELATIVE_PATH",
    "PostprocessInputAssemblyError",
    "REQUIRED_PRODUCER_CONTRACTS",
    "SAMPLE_COVARIATES_RELATIVE_PATH",
    "assemble_postprocess_inputs",
]
