"""Hash-bound label-free inputs for the post-lock Test attribution bridge.

The pre-lock builder in this module never parses ground-truth rows.  It freezes
only candidate geometry, selector inputs, the shared denominator, and opaque
source hashes.  Candidate labels are attached later by the already-claimed
formal-Test transaction.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .hashing import atomic_json, canonical_sha256, sha256_file
from .candidates import regenerate_candidate_contract_hashes


BRIDGE_ROUTES = ("g1", "c1")
POOL_CONTRACTS = ("fair_gaussian", "historical_nms")
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
FORBIDDEN_LABEL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"candidate[_-]?success",
        r"ground[_-]?truth",
        r"(?:^|_)gt[_-]?grasp",
        r"jacquard",
        r"matched[_-]?gt",
        r"angle[_-]?error",
        # IoU between candidates is valid label-free geometry evidence.  Only
        # the evaluator/GT-derived IoU field names are forbidden here.
        r"best[_-]?(?:same[_-]?gt|rectangle)[_-]?iou",
        r"(?:ground[_-]?truth|matched[_-]?gt)[_-]?iou",
    )
)
FORMULAS = {
    "fair_gaussian/fair_native_selector": "native_score",
    "fair_gaussian/historical_selector": "native_score*clip(p_center,0,1)",
    "historical_nms/fair_native_selector": "raw_network_quality",
    "historical_nms/historical_selector": "original_score",
    "historical_nms/original_score_check": "raw_network_quality*clip(stored_center_mask_support,0,1)",
    "historical_jaw_exponent": 0,
}


def _regular(path: str | Path, label: str) -> Path:
    result = Path(path).resolve()
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"{label} is not a regular file: {result}")
    return result


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(_regular(path, "JSON source").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    previous_path: str | None = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        if match is None:
            raise ValueError(f"invalid historical run SHA manifest line {line_number}")
        relative = match.group(2)
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or relative in records
            or (previous_path is not None and relative <= previous_path)
        ):
            raise ValueError(
                "historical run SHA manifest contains an unsafe, duplicate, or unsorted path"
            )
        records[relative] = match.group(1)
        previous_path = relative
    return records


def _historical_run_authority(
    *,
    run_sha_manifest_path: Path,
    run_lock_sha256_path: Path,
    formal_lock_path: Path,
    inventory_path: Path,
    historical_candidates: Mapping[str, Path],
    historical_ground_truth_path: Path,
    modular_run_root: Path,
) -> dict[str, dict[str, str]]:
    run_sha_manifest_path = _regular(run_sha_manifest_path, "historical RUN_SHA256_MANIFEST")
    run_lock_sha256_path = _regular(run_lock_sha256_path, "historical RUN_LOCK_SHA256")
    run_root = run_sha_manifest_path.parent.resolve()
    if run_lock_sha256_path != run_root / "RUN_LOCK_SHA256.txt":
        raise ValueError("historical RUN_LOCK_SHA256 path is outside the frozen run root")
    locked_digest = run_lock_sha256_path.read_text(encoding="utf-8").strip()
    if (
        re.fullmatch(r"[0-9a-f]{64}", locked_digest) is None
        or locked_digest != sha256_file(run_sha_manifest_path)
    ):
        raise ValueError("historical RUN_LOCK_SHA256 does not bind RUN_SHA256_MANIFEST")
    records = _sha_manifest(run_sha_manifest_path)

    required_paths: dict[str, Path] = {
        "historical_pool_inventory": inventory_path.resolve(),
        "historical_formal_lock": formal_lock_path.resolve(),
    }
    for route in BRIDGE_ROUTES:
        required_paths[f"{route}_historical_candidates"] = historical_candidates[route].resolve()
        required_paths[f"{route}_historical_allnms"] = (
            run_root / "data" / f"frozen_{route}_test_allnms_candidates.parquet"
        ).resolve()
    authority: dict[str, dict[str, str]] = {
        "historical_run_sha_manifest": _record(run_sha_manifest_path),
        "historical_run_lock_sha256": _record(run_lock_sha256_path),
    }
    for name, candidate_path in required_paths.items():
        candidate_path = _regular(candidate_path, name)
        try:
            relative = candidate_path.relative_to(run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"{name} is outside the frozen historical run") from error
        expected = records.get(relative)
        if expected is None or sha256_file(candidate_path) != expected:
            raise ValueError(f"historical RUN_SHA256_MANIFEST binding mismatch: {relative}")
        authority[name] = {"path": str(candidate_path), "sha256": expected}

    formal_lock = _read_json(formal_lock_path)
    if (
        formal_lock.get("status") != "LOCKED"
        or Path(str(formal_lock.get("base_run", ""))).resolve()
        != modular_run_root.resolve()
    ):
        raise ValueError("historical formal Test lock is not LOCKED")
    inventory_record = next(
        (
            value
            for value in formal_lock.get("audit_artifacts", [])
            if Path(str(value.get("path", ""))).resolve() == inventory_path.resolve()
        ),
        None,
    )
    if not isinstance(inventory_record, dict) or inventory_record.get("sha256") != sha256_file(
        inventory_path
    ):
        raise ValueError("historical formal lock does not bind frozen-pool inventory")
    label_record = next(
        (
            value
            for value in formal_lock.get("source_label_artifacts", [])
            if Path(str(value.get("path", ""))).resolve()
            == historical_ground_truth_path.resolve()
        ),
        None,
    )
    if not isinstance(label_record, dict) or label_record.get("sha256") != sha256_file(
        historical_ground_truth_path
    ):
        raise ValueError("historical formal lock does not bind modular Test ground truth")
    for route in BRIDGE_ROUTES:
        allnms = authority[f"{route}_historical_allnms"]
        record = next(
            (
                value
                for value in formal_lock.get("candidate_artifacts", [])
                if value.get("sha256") == allnms["sha256"]
            ),
            None,
        )
        if not isinstance(record, dict) or record.get("sha256") != allnms["sha256"]:
            raise ValueError(f"historical formal lock does not bind {route} AllNMS pool")
    return authority


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _verify_record(record: Mapping[str, Any], label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} has no artifact record")
    path = _regular(str(record.get("path", "")), label)
    digest = str(record.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None or sha256_file(path) != digest:
        raise ValueError(f"{label} hash mismatch")
    return path


def _qualified(route: str, pool: str, raw_id: pd.Series) -> pd.Series:
    return route.upper() + "::" + pool + "::" + raw_id.astype(str)


def _assert_label_free_columns(frame: pd.DataFrame, label: str) -> None:
    for column in frame.columns:
        if any(pattern.search(str(column)) for pattern in FORBIDDEN_LABEL_PATTERNS):
            raise PermissionError(
                f"{label} contains forbidden historical-label field: {column}"
            )


def _geometry_hash(
    route: str,
    pool: str,
    sample_id: object,
    raw_candidate_id: object,
    native_rank: int,
    geometry: list[float],
) -> str:
    return canonical_sha256(
        [
            route.upper(),
            pool,
            str(sample_id),
            str(raw_candidate_id),
            int(native_rank),
            *map(float, geometry),
        ]
    )


def _validate_rank_and_geometry(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    work = frame.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["source_candidate_id"] = work["source_candidate_id"].astype(str)
    rank = pd.to_numeric(work["native_rank"], errors="coerce")
    geometry_columns = ["cx_px", "cy_px", "theta_deg", "width_px", "height_px"]
    geometry = work[geometry_columns].apply(pd.to_numeric, errors="coerce")
    if (
        work.empty
        or work[["sample_id", "source_candidate_id"]].eq("").any().any()
        or work.duplicated(["sample_id", "source_candidate_id"]).any()
        or rank.isna().any()
        or not np.equal(rank, np.floor(rank)).all()
        or rank.lt(1).any()
        or rank.gt(5).any()
        or not np.isfinite(geometry.to_numpy(float)).all()
        or geometry[["width_px", "height_px"]].le(0).any().any()
    ):
        raise ValueError(f"{label} violates Top-5 identity/rank/geometry contracts")
    work["native_rank"] = rank.astype(int)
    work[geometry_columns] = geometry
    for sample_id, group in work.groupby("sample_id", sort=False):
        ranks = sorted(group["native_rank"].tolist())
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"{label}/{sample_id} ranks are not contiguous")
    return work


def _fair_rows(
    route: str,
    candidate_path: Path,
    feature_path: Path,
    evaluator_sha256: str,
) -> pd.DataFrame:
    candidates = pd.read_parquet(candidate_path)
    _assert_label_free_columns(candidates, f"{route} fair Test candidates")
    candidate_columns = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
        "candidate_geometry_sha256",
    }
    missing = sorted(candidate_columns.difference(candidates.columns))
    if missing:
        raise ValueError(f"{route} fair Test Top-5 misses columns: {missing}")
    work = candidates[list(candidate_columns)].rename(
        columns={"candidate_id": "source_candidate_id"}
    )
    work = _validate_rank_and_geometry(work, f"{route} fair Test Top-5")
    features = pd.read_parquet(feature_path)
    _assert_label_free_columns(features, f"{route} fair Test features")
    feature_columns = {
        "sample_id",
        "candidate_id",
        "native_score_raw",
        "native_rank",
        "p_center",
    }
    missing = sorted(feature_columns.difference(features.columns))
    if missing:
        raise ValueError(f"{route} fair Test bridge features miss columns: {missing}")
    feature = features[list(feature_columns)].rename(
        columns={
            "candidate_id": "source_candidate_id",
            "native_rank": "feature_native_rank",
        }
    )
    if feature.duplicated(["sample_id", "source_candidate_id"]).any():
        raise ValueError(f"{route} fair Test bridge features contain duplicate identities")
    joined = work.merge(
        feature,
        on=["sample_id", "source_candidate_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"{route} fair Test bridge features do not exactly cover Top-5")
    joined = joined.drop(columns="_merge")
    feature_rank = pd.to_numeric(joined["feature_native_rank"], errors="coerce")
    native_score_raw = pd.to_numeric(joined["native_score_raw"], errors="coerce")
    p_center = pd.to_numeric(joined["p_center"], errors="coerce")
    if (
        native_score_raw.isna().any()
        or p_center.isna().any()
        or not np.isfinite(native_score_raw.to_numpy(float)).all()
        or not np.isfinite(p_center.to_numpy(float)).all()
    ):
        raise ValueError(f"{route} fair Test bridge selector inputs must be finite")
    if (
        feature_rank.isna().any()
        or not np.equal(feature_rank, joined["native_rank"]).all()
        or not np.allclose(
            native_score_raw.to_numpy(float),
            joined["native_score"].to_numpy(float),
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError(f"{route} fair Test bridge feature/native binding mismatch")
    transferred = joined["native_score"].to_numpy(float) * np.clip(
        p_center.to_numpy(float), 0.0, 1.0
    )
    expected_geometry = [
        canonical_sha256(
            [
                route.upper(),
                row.sample_id,
                row.source_candidate_id,
                int(row.native_rank),
                float(row.cx_px),
                float(row.cy_px),
                float(row.theta_deg),
                float(row.width_px),
                float(row.height_px),
            ]
        )
        for row in joined.itertuples(index=False)
    ]
    if joined["candidate_geometry_sha256"].astype(str).tolist() != expected_geometry:
        raise ValueError(f"{route} fair Test candidate geometry hash mismatch")
    joined["route"] = route
    joined["candidate_pool_contract"] = "fair_gaussian"
    joined["raw_candidate_id"] = joined["source_candidate_id"]
    joined["candidate_id"] = _qualified(
        route, "fair_gaussian", joined["source_candidate_id"]
    )
    joined["source_candidate_identity_sha256"] = joined[
        "candidate_geometry_sha256"
    ].astype(str)
    joined["p_center"] = p_center.to_numpy(float)
    joined["raw_network_quality"] = joined["native_score"].to_numpy(float)
    joined["original_score"] = transferred
    joined["fair_native_selector_score"] = joined["native_score"].to_numpy(float)
    joined["historical_selector_score"] = transferred
    joined["system_name"] = f"bridge::{route}::fair_gaussian"
    joined["system_kind"] = "bridge"
    joined["evaluator_sha256"] = evaluator_sha256
    return joined


def _historical_rows(
    route: str,
    candidate_path: Path,
    evaluator_sha256: str,
) -> pd.DataFrame:
    source = pd.read_parquet(candidate_path)
    _assert_label_free_columns(source, f"{route} historical Test candidates")
    required = {
        "sample_id",
        "stable_candidate_id",
        "original_rank",
        "original_score",
        "raw_network_quality",
        "stored_center_mask_support",
        "center_x",
        "center_y",
        "angle_deg",
        "width_px",
        "height_px",
        "candidate_identity_sha256",
        "split",
        "backend",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"{route} historical Test Top-5 misses columns: {missing}")
    if set(source["split"].astype(str).str.lower()) != {"test"} or set(
        source["backend"].astype(str).str.lower()
    ) != {route}:
        raise ValueError(f"{route} historical Test source split/backend mismatch")
    work = source[list(required)].rename(
        columns={
            "stable_candidate_id": "source_candidate_id",
            "original_rank": "native_rank",
            "center_x": "cx_px",
            "center_y": "cy_px",
            "angle_deg": "theta_deg",
            "candidate_identity_sha256": "source_candidate_identity_sha256",
        }
    )
    work = _validate_rank_and_geometry(work, f"{route} historical Test Top-5")
    quality = pd.to_numeric(work["raw_network_quality"], errors="coerce")
    support = pd.to_numeric(work["stored_center_mask_support"], errors="coerce")
    original = pd.to_numeric(work["original_score"], errors="coerce")
    if (
        quality.isna().any()
        or support.isna().any()
        or original.isna().any()
        or not np.isfinite(
            np.column_stack([quality.to_numpy(float), support.to_numpy(float), original.to_numpy(float)])
        ).all()
    ):
        raise ValueError(f"{route} historical selector inputs must be finite")
    reconstructed = quality.to_numpy(float) * np.clip(
        support.to_numpy(float), 0.0, 1.0
    )
    if not np.allclose(original.to_numpy(float), reconstructed, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{route} historical original_score violates raw_network_quality*clip(center_support)"
        )
    work["route"] = route
    work["candidate_pool_contract"] = "historical_nms"
    work["raw_candidate_id"] = work["source_candidate_id"]
    work["candidate_id"] = _qualified(
        route, "historical_nms", work["source_candidate_id"]
    )
    work["candidate_geometry_sha256"] = [
        _geometry_hash(
            route,
            "historical_nms",
            row.sample_id,
            row.source_candidate_id,
            int(row.native_rank),
            [row.cx_px, row.cy_px, row.theta_deg, row.width_px, row.height_px],
        )
        for row in work.itertuples(index=False)
    ]
    work["native_score"] = quality.to_numpy(float)
    work["p_center"] = support.to_numpy(float)
    work["fair_native_selector_score"] = quality.to_numpy(float)
    work["historical_selector_score"] = original.to_numpy(float)
    work["system_name"] = f"bridge::{route}::historical_nms"
    work["system_kind"] = "bridge"
    work["evaluator_sha256"] = evaluator_sha256
    return work


def _finalize_bundle(frame: pd.DataFrame, denominator: set[str]) -> pd.DataFrame:
    work = frame.copy()
    for column in ("native_rank", "frozen_native_rank", "rank"):
        if column not in work:
            work[column] = work["native_rank"].astype(int)
    work["formal_score"] = work["fair_native_selector_score"].astype(float)
    if not set(work["sample_id"].astype(str)).issubset(denominator):
        raise ValueError("bridge candidates contain samples outside the Test denominator")
    if work.duplicated(
        ["route", "candidate_pool_contract", "sample_id", "candidate_id"]
    ).any():
        raise ValueError("bridge bundle contains duplicate qualified identities")
    if set(work["route"].astype(str)) != set(BRIDGE_ROUTES) or set(
        work["candidate_pool_contract"].astype(str)
    ) != set(POOL_CONTRACTS):
        raise ValueError("bridge bundle route/pool inventory mismatch")
    qualified = (
        work["route"].astype(str).str.upper()
        + "::"
        + work["candidate_pool_contract"].astype(str)
        + "::"
        + work["source_candidate_id"].astype(str)
    )
    if not work["candidate_id"].astype(str).equals(qualified) or not work[
        "raw_candidate_id"
    ].astype(str).equals(work["source_candidate_id"].astype(str)):
        raise ValueError("bridge bundle candidate IDs are not route/pool qualified")
    if (
        work["candidate_geometry_sha256"].astype(str).eq("").any()
        or work["source_candidate_identity_sha256"].astype(str).eq("").any()
        or work["evaluator_sha256"].astype(str).eq("").any()
    ):
        raise ValueError("bridge bundle hash provenance is empty")
    for pattern in FORBIDDEN_LABEL_PATTERNS:
        if any(pattern.search(str(column)) for column in work.columns):
            raise PermissionError(
                "label-free bridge bundle contains forbidden postclaim field: "
                f"{pattern.pattern}"
            )
    numeric = [
        "native_rank",
        "fair_native_selector_score",
        "historical_selector_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ]
    if not np.isfinite(work[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
        raise ValueError("bridge bundle contains non-finite ranks/scores/geometry")
    return work[list(BRIDGE_COLUMNS)].sort_values(
        ["route", "candidate_pool_contract", "sample_id", "native_rank", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _pool_checks(bundle: pd.DataFrame, denominator: list[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for route in BRIDGE_ROUTES:
        for pool in POOL_CONTRACTS:
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
                "qualified_identity_sha256": canonical_sha256(
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


def _declared_ground_truth(
    source_manifest_path: Path,
    ground_truth_path: Path,
    marker_path: Path,
    finalization_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    source_manifest = _read_json(source_manifest_path)
    marker = _read_json(marker_path)
    finalization = _read_json(finalization_path)
    source_root = source_manifest_path.parent.parent.resolve()
    expected_manifest_path = (
        source_root / str(marker.get("lock_relative_path", ""))
    ).resolve()
    unsigned = dict(source_manifest)
    content_sha256 = unsigned.pop("manifest_content_sha256", None)
    if (
        marker_path.resolve() != source_root / ".EXPERIMENT_LOCKED"
        or finalization_path.resolve() != source_root / "FINALIZATION_COMPLETE.json"
        or expected_manifest_path != source_manifest_path.resolve()
        or marker.get("lock_status") != "LOCKED"
        or source_manifest.get("lock_status") != "LOCKED"
        or source_manifest.get("effective") is not True
        or content_sha256 != canonical_sha256(unsigned)
        or marker.get("manifest_content_sha256") != content_sha256
        or source_manifest.get("run_id") != marker.get("run_id")
        or finalization.get("status") != "COMPLETE"
        or finalization.get("experiment_lock_sha256") != content_sha256
    ):
        raise ValueError("historical ground-truth experiment lock authority mismatch")
    record = dict(source_manifest.get("artifacts", {})).get("test_labels")
    if not isinstance(record, dict):
        raise ValueError("historical source manifest does not declare artifacts.test_labels")
    declared = Path(str(record.get("path", "")))
    if not declared.is_absolute():
        declared = source_manifest_path.parent.parent / declared
    declared = declared.resolve()
    if declared != ground_truth_path.resolve() or str(record.get("sha256", "")) != sha256_file(
        ground_truth_path
    ):
        raise ValueError("opaque historical Test ground truth differs from source manifest")
    return (
        {"path": str(declared), "sha256": str(record["sha256"])},
        {
            "historical_experiment_lock_marker": _record(marker_path),
            "historical_finalization": _record(finalization_path),
        },
    )


def build_label_free_test_bridge(
    *,
    run_dir: Path,
    historical_candidates: Mapping[str, Path],
    historical_ground_truth_path: Path,
    historical_source_manifest_path: Path,
    historical_inventory_path: Path,
    evaluator_path: Path,
    historical_run_sha_manifest_path: Path | None = None,
    historical_run_lock_sha256_path: Path | None = None,
    historical_formal_lock_path: Path | None = None,
    historical_experiment_lock_marker_path: Path | None = None,
    historical_finalization_path: Path | None = None,
    denominator_path: Path | None = None,
    fair_candidates: Mapping[str, Path] | None = None,
    fair_features: Mapping[str, Path] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the immutable label-free Test bridge input and source manifest."""

    root = run_dir.resolve()
    if (root / "08_lock" / "FORMAL_TEST_LOCK.json").exists() or (
        root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    ).exists():
        raise PermissionError("Test bridge inputs cannot be changed after formal lock/claim")
    denominator_path = _regular(
        denominator_path or root / "01_manifests" / "paired_test.parquet",
        "bridge Test denominator",
    )
    evaluator_path = _regular(evaluator_path, "frozen fair evaluator")
    historical_ground_truth_path = _regular(
        historical_ground_truth_path, "opaque historical Test ground truth"
    )
    historical_source_manifest_path = _regular(
        historical_source_manifest_path, "historical Test source manifest"
    )
    historical_inventory_path = _regular(
        historical_inventory_path, "historical frozen-pool inventory"
    )
    historical_root = historical_inventory_path.parent.parent.resolve()
    historical_run_sha_manifest_path = _regular(
        historical_run_sha_manifest_path or historical_root / "RUN_SHA256_MANIFEST.txt",
        "historical RUN_SHA256_MANIFEST",
    )
    historical_run_lock_sha256_path = _regular(
        historical_run_lock_sha256_path or historical_root / "RUN_LOCK_SHA256.txt",
        "historical RUN_LOCK_SHA256",
    )
    historical_formal_lock_path = _regular(
        historical_formal_lock_path or historical_root / "08_lock/FORMAL_TEST_LOCK.json",
        "historical formal Test lock",
    )
    modular_root = historical_source_manifest_path.parent.parent.resolve()
    historical_experiment_lock_marker_path = _regular(
        historical_experiment_lock_marker_path or modular_root / ".EXPERIMENT_LOCKED",
        "historical experiment-lock marker",
    )
    historical_finalization_path = _regular(
        historical_finalization_path or modular_root / "FINALIZATION_COMPLETE.json",
        "historical finalization marker",
    )
    historical_inventory = _read_json(historical_inventory_path)
    denominator_frame = pd.read_parquet(denominator_path, columns=["sample_id"])
    denominator = denominator_frame["sample_id"].astype(str).tolist()
    if not denominator or len(set(denominator)) != len(denominator) or any(
        not value for value in denominator
    ):
        raise ValueError("bridge Test denominator must contain unique non-empty samples")
    evaluator_sha256 = sha256_file(evaluator_path)
    fair_candidates = dict(fair_candidates or {})
    fair_features = dict(fair_features or {})
    historical_candidates = dict(historical_candidates)
    if set(historical_candidates) != set(BRIDGE_ROUTES):
        raise ValueError("historical bridge candidates must provide exactly g1 and c1")
    rows: list[pd.DataFrame] = []
    ground_truth_record, modular_authority = _declared_ground_truth(
        historical_source_manifest_path,
        historical_ground_truth_path,
        historical_experiment_lock_marker_path,
        historical_finalization_path,
    )
    historical_authority = _historical_run_authority(
        run_sha_manifest_path=historical_run_sha_manifest_path,
        run_lock_sha256_path=historical_run_lock_sha256_path,
        formal_lock_path=historical_formal_lock_path,
        inventory_path=historical_inventory_path,
        historical_candidates=historical_candidates,
        historical_ground_truth_path=historical_ground_truth_path,
        modular_run_root=modular_root,
    )
    sources: dict[str, dict[str, str]] = {
        "denominator": _record(denominator_path),
        "evaluator": _record(evaluator_path),
        "historical_ground_truth": ground_truth_record,
        "historical_source_manifest": _record(historical_source_manifest_path),
        "historical_pool_inventory": _record(historical_inventory_path),
        **historical_authority,
        **modular_authority,
    }
    for route in BRIDGE_ROUTES:
        fair_path = _regular(
            fair_candidates.get(route)
            or root / "02_candidates" / f"{route}_test_top5.parquet",
            f"{route} fair Test Top-5",
        )
        feature_path = _regular(
            fair_features.get(route)
            or root
            / "03_features"
            / "common"
            / f"{route}_test"
            / "candidate_features.parquet",
            f"{route} fair Test bridge features",
        )
        historical_path = _regular(
            historical_candidates[route], f"{route} historical Test Top-5"
        )
        inventory_record = historical_inventory.get(f"{route.upper()}_test")
        if not isinstance(inventory_record, dict) or (
            Path(str(inventory_record.get("top5_path", ""))).resolve()
            != historical_path
            or inventory_record.get("top5_artifact_sha256")
            != sha256_file(historical_path)
        ):
            raise ValueError(
                f"{route} historical Test Top-5 differs from frozen-pool inventory"
            )
        sources[f"{route}_fair_candidates"] = _record(fair_path)
        sources[f"{route}_fair_features"] = _record(feature_path)
        sources[f"{route}_historical_candidates"] = _record(historical_path)
        rows.extend(
            [
                _fair_rows(route, fair_path, feature_path, evaluator_sha256),
                _historical_rows(route, historical_path, evaluator_sha256),
            ]
        )
    bundle = _finalize_bundle(pd.concat(rows, ignore_index=True), set(denominator))
    regenerate_candidate_contract_hashes(root)
    destination = (
        output_dir.resolve()
        if output_dir is not None
        else root / "11_attribution_bridge" / "test_bridge_input"
    )
    bundle_path = destination / "bridge_label_free_candidates.parquet"
    manifest_path = destination / "manifest.json"
    _atomic_parquet(bundle_path, bundle)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "split": "test",
        "analysis_role": "SECONDARY_POSTLOCK_NO_SELECTION",
        "candidate_test_labels_read": False,
        "historical_test_ground_truth_rows_read": False,
        "historical_test_ground_truth_access": "HASH_ONLY_PRELOCK",
        "candidate_contract_inventory_regenerated": True,
        "candidate_contract_inventory_locked": False,
        "candidate_identity": "<ROUTE>::<POOL_CONTRACT>::<source_candidate_id>",
        "membership_keys": [
            "route",
            "candidate_pool_contract",
            "sample_id",
            "candidate_id",
        ],
        "formulas": FORMULAS,
        "denominator": {
            "sample_count": len(denominator),
            "sample_id_sha256": canonical_sha256(sorted(denominator)),
        },
        "pool_checks": _pool_checks(bundle, denominator),
        "sources": sources,
        "artifacts": {"candidate_bundle": _record(bundle_path)},
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def validate_label_free_test_bridge_manifest(
    manifest_path: Path,
    *,
    expected_denominator: Mapping[str, Any] | None = None,
    expected_evaluator: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Path]]:
    """Verify the complete pre-lock bridge contract without opening GT rows."""

    path = _regular(manifest_path, "label-free Test bridge manifest")
    manifest = _read_json(path)
    recorded = manifest.get("content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "COMPLETE"
        or manifest.get("candidate_test_labels_read") is not False
        or manifest.get("historical_test_ground_truth_rows_read") is not False
        or recorded != canonical_sha256(unsigned)
    ):
        raise ValueError("label-free Test bridge manifest integrity/state mismatch")
    expected_source_names = {
        "denominator",
        "evaluator",
        "historical_ground_truth",
        "historical_source_manifest",
        "historical_pool_inventory",
        "historical_run_sha_manifest",
        "historical_run_lock_sha256",
        "historical_formal_lock",
        "historical_experiment_lock_marker",
        "historical_finalization",
        "g1_historical_allnms",
        "c1_historical_allnms",
        *{
            f"{route}_{suffix}"
            for route in BRIDGE_ROUTES
            for suffix in (
                "fair_candidates",
                "fair_features",
                "historical_candidates",
            )
        },
    }
    if set(manifest.get("sources", {})) != expected_source_names:
        raise ValueError("label-free Test bridge source inventory mismatch")
    resolved = {
        f"source_{name}": _verify_record(record, f"bridge source {name}")
        for name, record in manifest["sources"].items()
    }
    historical_authority = _historical_run_authority(
        run_sha_manifest_path=resolved["source_historical_run_sha_manifest"],
        run_lock_sha256_path=resolved["source_historical_run_lock_sha256"],
        formal_lock_path=resolved["source_historical_formal_lock"],
        inventory_path=resolved["source_historical_pool_inventory"],
        historical_candidates={
            route: resolved[f"source_{route}_historical_candidates"]
            for route in BRIDGE_ROUTES
        },
        historical_ground_truth_path=resolved["source_historical_ground_truth"],
        modular_run_root=resolved["source_historical_source_manifest"].parent.parent,
    )
    ground_truth_record, modular_authority = _declared_ground_truth(
        resolved["source_historical_source_manifest"],
        resolved["source_historical_ground_truth"],
        resolved["source_historical_experiment_lock_marker"],
        resolved["source_historical_finalization"],
    )
    expected_authority = {
        **historical_authority,
        **modular_authority,
        "historical_ground_truth": ground_truth_record,
    }
    for name, record in expected_authority.items():
        if manifest["sources"].get(name) != record:
            raise ValueError(f"label-free Test bridge authority binding mismatch: {name}")
    if expected_denominator is not None:
        expected = {
            "path": str(Path(str(expected_denominator.get("path", ""))).resolve()),
            "sha256": expected_denominator.get("sha256"),
        }
        if manifest["sources"]["denominator"] != expected:
            raise ValueError("label-free Test bridge denominator binding mismatch")
    if expected_evaluator is not None:
        expected = {
            "path": str(Path(str(expected_evaluator.get("path", ""))).resolve()),
            "sha256": expected_evaluator.get("sha256"),
        }
        if manifest["sources"]["evaluator"] != expected:
            raise ValueError("label-free Test bridge evaluator binding mismatch")
    bundle_path = _verify_record(
        manifest.get("artifacts", {}).get("candidate_bundle", {}),
        "label-free Test bridge candidate bundle",
    )
    resolved["candidate_bundle"] = bundle_path
    bundle = pd.read_parquet(bundle_path)
    if set(bundle.columns) != set(BRIDGE_COLUMNS):
        raise ValueError("label-free Test bridge candidate schema mismatch")
    denominator_path = resolved["source_denominator"]
    denominator = pd.read_parquet(denominator_path, columns=["sample_id"])[
        "sample_id"
    ].astype(str).tolist()
    validated = _finalize_bundle(bundle, set(denominator))
    if not validated.equals(bundle.reset_index(drop=True)):
        raise ValueError("label-free Test bridge candidate ordering/content mismatch")
    if manifest.get("denominator") != {
        "sample_count": len(denominator),
        "sample_id_sha256": canonical_sha256(sorted(denominator)),
    }:
        raise ValueError("label-free Test bridge denominator summary mismatch")
    if manifest.get("formulas") != FORMULAS:
        raise ValueError("label-free Test bridge formula declaration mismatch")
    for route in BRIDGE_ROUTES:
        fair = bundle.loc[
            bundle["route"].eq(route)
            & bundle["candidate_pool_contract"].eq("fair_gaussian")
        ]
        historical = bundle.loc[
            bundle["route"].eq(route)
            & bundle["candidate_pool_contract"].eq("historical_nms")
        ]
        fair_transferred = fair["native_score"].to_numpy(float) * np.clip(
            fair["p_center"].to_numpy(float), 0.0, 1.0
        )
        historical_transferred = historical["raw_network_quality"].to_numpy(float) * np.clip(
            historical["p_center"].to_numpy(float), 0.0, 1.0
        )
        if (
            not np.array_equal(
                fair["fair_native_selector_score"].to_numpy(float),
                fair["native_score"].to_numpy(float),
            )
            or not np.allclose(
                fair["historical_selector_score"].to_numpy(float),
                fair_transferred,
                rtol=0.0,
                atol=0.0,
            )
            or not np.array_equal(
                historical["fair_native_selector_score"].to_numpy(float),
                historical["raw_network_quality"].to_numpy(float),
            )
            or not np.allclose(
                historical["historical_selector_score"].to_numpy(float),
                historical_transferred,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("label-free Test bridge selector formula mismatch")
    expected_geometry = []
    for row in bundle.itertuples(index=False):
        geometry = [row.cx_px, row.cy_px, row.theta_deg, row.width_px, row.height_px]
        if row.candidate_pool_contract == "fair_gaussian":
            expected_geometry.append(
                canonical_sha256(
                    [
                        str(row.route).upper(),
                        str(row.sample_id),
                        str(row.source_candidate_id),
                        int(row.native_rank),
                        *map(float, geometry),
                    ]
                )
            )
        else:
            expected_geometry.append(
                _geometry_hash(
                    str(row.route),
                    str(row.candidate_pool_contract),
                    row.sample_id,
                    row.source_candidate_id,
                    int(row.native_rank),
                    list(map(float, geometry)),
                )
            )
    if bundle["candidate_geometry_sha256"].astype(str).tolist() != expected_geometry:
        raise ValueError("label-free Test bridge geometry binding mismatch")
    if manifest.get("pool_checks") != _pool_checks(bundle, denominator):
        raise ValueError("label-free Test bridge coverage/no-output/formula summary mismatch")
    resolved["manifest"] = path
    return manifest, bundle, resolved


__all__ = [
    "BRIDGE_COLUMNS",
    "BRIDGE_ROUTES",
    "POOL_CONTRACTS",
    "build_label_free_test_bridge",
    "validate_label_free_test_bridge_manifest",
]
