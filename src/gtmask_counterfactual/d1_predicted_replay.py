"""Independent, label-free semantic closure for the frozen D1 replay."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.artifacts import verify_artifact_records_recursive

from .d1_adapter import (
    D1AdapterError,
    _index_rows,
    _load_json,
    _load_jsonl,
    _safe_id,
    _scored_marker,
    candidate_inventory_from_summary,
)
from .d1_source_view import verify_d1_source_view
from .io import artifact_record, atomic_json, canonical_sha256, sha256_file


EXPECTED_SAMPLES = 7_675
EXPECTED_CANDIDATES = 187_077
EXPECTED_NO_OUTPUT = 108
REQUIRED_COMPARISONS = {
    "raw_candidate_stage",
    "mask_validated_candidate_stage",
    "post_nms_candidate_stage",
    "no_output_identity",
    "sample_candidate_counts",
    "candidate_identity_or_stable_geometry",
    "native_scores",
    "top1",
    "top5",
    "top10",
    "allnms",
    "oracle_metrics",
}
_CANDIDATE_STAGE_FILES = {
    "raw": "raw_candidates.json",
    "mask_validated": "mask_validated_candidates.json",
    "filtered": "filtered_candidates.json",
    "post_nms": "candidates.json",
}
_SUMMARY_IDENTITY_FIELDS = (
    "sample_id",
    "query",
    "requested_candidate_count",
    "raw_candidate_count",
    "mask_validated_count",
    "post_nms_count",
    "failure_reason",
    "status",
    "question_index",
    "scene_id",
)
_MARKER_IDENTITY_FIELDS = (
    "sample_id",
    "configuration_hash",
    "config_file_sha256",
    "seed",
    "sampler_version",
    "sampler_release",
    "sampler_commit",
    "sampler_class",
    "status",
    "candidate_counts",
)
_IDENTITY_COLUMNS = (
    "sample_id",
    "candidate_id",
    "source_candidate_index",
    "native_rank",
)
_FLOAT_COLUMNS = (
    "native_score",
    "cx_px",
    "cy_px",
    "center_depth_m",
    "theta_deg",
    "width_px",
    "height_px",
)


class D1PredictedReplayError(RuntimeError):
    """Raised when a regenerated D1 predicted branch differs from authority."""


def _regular(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser().resolve(strict=False)
    if Path(path).expanduser().is_symlink() or not source.is_file():
        raise D1PredictedReplayError(f"{label} must be a regular file: {source}")
    return source


def _self_hashed(path: str | Path, *, label: str) -> dict[str, Any]:
    source = _regular(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D1PredictedReplayError(f"cannot parse {label}: {source}") from error
    if not isinstance(value, dict):
        raise D1PredictedReplayError(f"{label} must contain an object")
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256", None)
    if recorded != canonical_sha256(unsigned):
        raise D1PredictedReplayError(f"{label} content hash differs")
    verify_artifact_records_recursive(value, name=label, require_at_least_one=True)
    return value


def _locked_inventory_record(root: Path, path: Path) -> dict[str, Any]:
    lock_path = _regular(root / "FINAL_RUN_LOCK.json", label="D1 final lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    inventory = lock.get("inventory") if isinstance(lock, Mapping) else None
    if not isinstance(inventory, list):
        raise D1PredictedReplayError("D1 final lock lacks an inventory")
    source = _regular(path, label="D1 frozen artifact")
    try:
        relative = str(source.relative_to(root))
    except ValueError as error:
        raise D1PredictedReplayError("D1 frozen artifact is outside its run") from error
    matches = [
        item
        for item in inventory
        if isinstance(item, Mapping) and item.get("relative_path") == relative
    ]
    if len(matches) != 1:
        raise D1PredictedReplayError(
            f"D1 frozen artifact is not uniquely locked: {relative}"
        )
    record = matches[0]
    observed = artifact_record(source)
    if any(record.get(key) != observed[key] for key in ("path", "sha256", "bytes")):
        raise D1PredictedReplayError(f"D1 frozen artifact changed: {relative}")
    return observed


def _root_records(candidate_root: Path, scored_root: Path) -> dict[str, Any]:
    paths = {
        "candidate_summary": candidate_root / "summary.csv",
        "candidate_run_config": candidate_root / "run_config.json",
        "scorer_run_config": scored_root / "run_config.json",
        "scorer_progress": scored_root / "progress.json",
        "scorer_summary": scored_root / "summary.csv",
        "scorer_manifest": scored_root / "scoring_manifest.jsonl",
        "scorer_statistics": scored_root / "run_statistics.json",
    }
    return {
        name: artifact_record(_regular(path, label=name))
        for name, path in paths.items()
    }


def _candidate_stage_value(path: Path, *, stage: str) -> list[dict[str, Any]]:
    source = _regular(path, label=f"D1 {stage} candidate stage")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D1PredictedReplayError(
            f"cannot parse D1 {stage} candidate stage: {source}"
        ) from error
    if stage == "post_nms":
        records = value.get("candidates")
    else:
        # The frozen helper annotates JSON list payloads as dictionaries only in
        # its type hints; json.load correctly returns the actual list here.
        records = value
    if not isinstance(records, list) or not all(
        isinstance(item, Mapping) for item in records
    ):
        raise D1PredictedReplayError(f"D1 {stage} candidate stage is malformed")
    return [dict(item) for item in records]


def _validated_candidate_marker(
    sample_root: Path, *, sample_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_path = _regular(
        sample_root / "_SUCCESS.json", label=f"D1 candidate marker {sample_id}"
    )
    marker = _load_json(marker_path, label=f"D1 candidate marker {sample_id}")
    if marker.get("sample_id") != sample_id:
        raise D1PredictedReplayError(
            f"D1 candidate marker sample identity differs: {sample_id}"
        )
    required = marker.get("required_files")
    hashes = marker.get("required_file_hashes")
    if (
        not isinstance(required, list)
        or not isinstance(hashes, Mapping)
        or set(required) != set(hashes)
        or not set(_CANDIDATE_STAGE_FILES.values()).issubset(required)
    ):
        raise D1PredictedReplayError(
            f"D1 candidate marker inventory is malformed: {sample_id}"
        )
    records: dict[str, Any] = {}
    for name in sorted(required):
        path = _regular(sample_root / name, label=f"D1 candidate artifact {name}")
        if hashes.get(name) != sha256_file(path):
            raise D1PredictedReplayError(
                f"D1 candidate marker artifact hash differs: {sample_id}/{name}"
            )
        if name in _CANDIDATE_STAGE_FILES.values():
            records[name] = artifact_record(path)
    return marker, {"marker": artifact_record(marker_path), "stages": records}


def _candidate_stage_replay(
    candidate_root: Path,
    frozen_root: Path,
    *,
    expected_sample_ids: set[str],
    expected_no_output: int,
) -> dict[str, Any]:
    """Exact-replay raw, mask-valid and NMS candidates for every sample."""

    actual_summary_path = _regular(
        candidate_root / "summary.csv", label="D1 replay candidate summary"
    )
    frozen_summary_path = _regular(
        frozen_root / "summary.csv", label="D1 frozen candidate summary"
    )
    with actual_summary_path.open(encoding="utf-8", newline="") as stream:
        actual_rows = list(csv.DictReader(stream))
    with frozen_summary_path.open(encoding="utf-8", newline="") as stream:
        frozen_rows = list(csv.DictReader(stream))
    actual_ids = [_safe_id(row.get("sample_id")) for row in actual_rows]
    frozen_ids = [_safe_id(row.get("sample_id")) for row in frozen_rows]
    if (
        len(actual_ids) != len(set(actual_ids))
        or len(frozen_ids) != len(set(frozen_ids))
        or set(actual_ids) != expected_sample_ids
        or actual_ids != frozen_ids
    ):
        raise D1PredictedReplayError(
            "D1 raw-stage sample universe or canonical order differs"
        )

    sample_evidence: dict[str, Any] = {}
    no_output_ids: list[str] = []
    for actual_row, frozen_row in zip(actual_rows, frozen_rows, strict=True):
        sample_id = _safe_id(actual_row.get("sample_id"))
        for field in _SUMMARY_IDENTITY_FIELDS:
            if str(actual_row.get(field, "")) != str(frozen_row.get(field, "")):
                raise D1PredictedReplayError(
                    f"D1 candidate summary differs: {sample_id}/{field}"
                )
        try:
            counts = {
                "raw": int(actual_row["raw_candidate_count"]),
                "mask_validated": int(actual_row["mask_validated_count"]),
                "post_nms": int(actual_row["post_nms_count"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise D1PredictedReplayError(
                f"D1 candidate stage counts are malformed: {sample_id}"
            ) from error
        if not (0 <= counts["post_nms"] <= counts["mask_validated"] <= counts["raw"]):
            raise D1PredictedReplayError(
                f"D1 candidate stage count ordering differs: {sample_id}"
            )
        actual_dir = candidate_root / sample_id
        frozen_dir = frozen_root / sample_id
        actual_marker, actual_records = _validated_candidate_marker(
            actual_dir, sample_id=sample_id
        )
        frozen_marker, frozen_records = _validated_candidate_marker(
            frozen_dir, sample_id=sample_id
        )
        for field in _MARKER_IDENTITY_FIELDS:
            if actual_marker.get(field) != frozen_marker.get(field):
                raise D1PredictedReplayError(
                    f"D1 candidate marker differs: {sample_id}/{field}"
                )

        semantic_hashes: dict[str, str] = {}
        expected_lengths = {
            "raw": counts["raw"],
            "mask_validated": counts["mask_validated"],
            "filtered": counts["post_nms"],
            "post_nms": counts["post_nms"],
        }
        for stage, name in _CANDIDATE_STAGE_FILES.items():
            actual_value = _candidate_stage_value(actual_dir / name, stage=stage)
            frozen_value = _candidate_stage_value(frozen_dir / name, stage=stage)
            if len(actual_value) != expected_lengths[stage]:
                raise D1PredictedReplayError(
                    f"D1 {stage} candidate count differs: {sample_id}"
                )
            actual_hash = canonical_sha256(actual_value)
            frozen_hash = canonical_sha256(frozen_value)
            if actual_hash != frozen_hash:
                raise D1PredictedReplayError(
                    f"D1 {stage} candidate semantics differ: {sample_id}"
                )
            semantic_hashes[stage] = actual_hash
        if semantic_hashes["filtered"] != semantic_hashes["post_nms"]:
            raise D1PredictedReplayError(
                f"D1 filtered/NMS ordering differs: {sample_id}"
            )
        if counts["post_nms"] == 0:
            no_output_ids.append(sample_id)
        sample_evidence[sample_id] = {
            "counts": counts,
            "semantic_sha256": semantic_hashes,
            "replay": actual_records,
            "frozen": frozen_records,
        }
    no_output_ids.sort()
    if len(no_output_ids) != expected_no_output:
        raise D1PredictedReplayError("D1 no-output identity count differs")
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "sample_count": len(actual_ids),
        "raw_candidate_count": sum(
            int(row["raw_candidate_count"]) for row in actual_rows
        ),
        "mask_validated_candidate_count": sum(
            int(row["mask_validated_count"]) for row in actual_rows
        ),
        "post_nms_candidate_count": sum(
            int(row["post_nms_count"]) for row in actual_rows
        ),
        "no_output_count": len(no_output_ids),
        "no_output_sample_ids_sha256": canonical_sha256(no_output_ids),
        "replay_summary": artifact_record(actual_summary_path),
        "frozen_summary": artifact_record(frozen_summary_path),
        "samples": sample_evidence,
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def _scored_frame(
    candidate_root: Path,
    scored_root: Path,
    *,
    expected_sample_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    candidate_root = candidate_root.expanduser().resolve(strict=False)
    scored_root = scored_root.expanduser().resolve(strict=False)
    inventory = candidate_inventory_from_summary(
        candidate_root, expected_sample_ids=expected_sample_ids
    )
    summary_path = _regular(candidate_root / "summary.csv", label="D1 summary")
    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8", newline="")))
    counts = {
        _safe_id(row.get("sample_id")): int(row["post_nms_count"])
        for row in summary_rows
    }
    progress = _load_json(scored_root / "progress.json", label="D1 scorer progress")
    statistics = _load_json(
        scored_root / "run_statistics.json", label="D1 scorer statistics"
    )
    expected_progress = {
        "total_samples": inventory.samples,
        "terminal_samples": inventory.samples,
        "completed_nonempty_samples": inventory.nonempty,
        "skipped_empty_samples": inventory.empty,
        "failed_samples": 0,
        "scored_candidates": inventory.candidates,
        "remaining_candidates": 0,
    }
    if any(int(progress.get(key, -1)) != value for key, value in expected_progress.items()):
        raise D1PredictedReplayError("D1 replay scorer progress differs")
    expected_statistics = {
        "total_samples": inventory.samples,
        "terminal_samples": inventory.samples,
        "scored_nonempty_samples": inventory.nonempty,
        "skipped_valid_empty_samples": inventory.empty,
        "failed_samples": 0,
        "corrupt_committed_samples": 0,
        "expected_candidates": inventory.candidates,
        "scored_candidates": inventory.candidates,
        "finite_q_values": inventory.candidates,
        "invalid_q_values": 0,
    }
    if any(
        int(statistics.get(key, -1)) != value
        for key, value in expected_statistics.items()
    ):
        raise D1PredictedReplayError("D1 replay scorer statistics differ")
    root_rows = _index_rows(
        _load_jsonl(scored_root / "scoring_manifest.jsonl", label="D1 scorer manifest"),
        label="D1 scorer manifest",
    )
    if set(root_rows) != expected_sample_ids:
        raise D1PredictedReplayError("D1 replay scorer sample universe differs")

    rows: list[dict[str, Any]] = []
    sample_sources: dict[str, Any] = {}
    for sample_id in sorted(expected_sample_ids):
        marker, candidates = _scored_marker(
            scored_root / sample_id,
            sample_id=sample_id,
            expected_candidate_count=counts[sample_id],
        )
        root_row = root_rows[sample_id]
        compared = (
            "scoring_status",
            "source_candidate_count",
            "gqcnn_scored_count",
            "top1_candidate_id",
            "source_candidate_sha256",
            "model_config_hash",
        )
        if any(str(root_row.get(key)) != str(marker.get(key)) for key in compared):
            raise D1PredictedReplayError(
                f"D1 replay root/sample provenance differs: {sample_id}"
            )
        for item in candidates:
            try:
                angle = item.get("angle_deg")
                row = {
                    "sample_id": sample_id,
                    "candidate_id": str(item["candidate_id"]),
                    "source_candidate_index": int(item["source_candidate_index"]),
                    "native_rank": int(item["gqcnn_rank"]),
                    "native_score": float(item["gqcnn_q_value"]),
                    "cx_px": float(item.get("center_u_px", item.get("centre_u_px"))),
                    "cy_px": float(item.get("center_v_px", item.get("centre_v_px"))),
                    "center_depth_m": float(item["center_depth_m"]),
                    "theta_deg": (
                        float(angle)
                        if angle is not None
                        else math.degrees(float(item["angle_rad"]))
                    ),
                    "width_px": float(item["width_px"]),
                    "height_px": float(
                        item.get("height_px", item.get("rectangle_height_px", 20.0))
                    ),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise D1PredictedReplayError(
                    f"D1 replay candidate geometry is malformed: {sample_id}"
                ) from error
            if not all(math.isfinite(row[column]) for column in _FLOAT_COLUMNS):
                raise D1PredictedReplayError(
                    f"D1 replay candidate has non-finite values: {sample_id}"
                )
            rows.append(row)
        marker_path = scored_root / sample_id / "_SCORING_COMPLETE.json"
        sample_sources[sample_id] = {
            "marker": artifact_record(marker_path),
            "payload": (
                None
                if not candidates
                else artifact_record(
                    scored_root / sample_id / "gqcnn_scored_candidates.json"
                )
            ),
            "candidate_count": len(candidates),
        }
    frame = pd.DataFrame(rows, columns=[*_IDENTITY_COLUMNS, *_FLOAT_COLUMNS])
    return frame, sample_sources, inventory.empty


def _assert_exact_frame(
    actual: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    label: str,
) -> None:
    columns = [*_IDENTITY_COLUMNS, *_FLOAT_COLUMNS]
    missing = sorted(set(columns).difference(frozen.columns))
    if missing:
        raise D1PredictedReplayError(f"{label} authority lacks columns: {missing}")
    left = actual.loc[:, columns].sort_values(
        list(_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    right = frozen.loc[:, columns].sort_values(
        list(_IDENTITY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    if left.shape != right.shape:
        raise D1PredictedReplayError(
            f"{label} shape differs: replay={left.shape} frozen={right.shape}"
        )
    for column in _IDENTITY_COLUMNS:
        if not left[column].astype(str).equals(right[column].astype(str)):
            raise D1PredictedReplayError(f"{label} identity differs: {column}")
    for column in _FLOAT_COLUMNS:
        observed = left[column].to_numpy(dtype=np.float64)
        expected = right[column].to_numpy(dtype=np.float64)
        if not np.array_equal(observed, expected, equal_nan=True):
            raise D1PredictedReplayError(f"{label} numeric values differ: {column}")


def _pool(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    return frame if maximum is None else frame.loc[frame["native_rank"] <= maximum]


def _load_authorities(d1_run: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    root = d1_run.expanduser().resolve()
    manifest_path = root / "02_candidates/test_manifest.json"
    manifest = _self_hashed(manifest_path, label="D1 Test candidate manifest")
    _locked_inventory_record(root, manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise D1PredictedReplayError("D1 candidate manifest artifacts are malformed")
    expected = {
        "allnms": "d1_allnms_candidates.parquet",
        "top10": "d1_top10_candidates.parquet",
        "top5": "d1_top5_candidates.parquet",
    }
    frames: dict[str, pd.DataFrame] = {}
    for pool, basename in expected.items():
        record = artifacts.get(pool)
        if not isinstance(record, Mapping):
            raise D1PredictedReplayError(f"D1 candidate manifest lacks {pool}")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if path.name != basename:
            raise D1PredictedReplayError(f"D1 {pool} authority path differs")
        locked = _locked_inventory_record(root, path)
        if any(record.get(key) != locked[key] for key in ("path", "sha256", "bytes")):
            raise D1PredictedReplayError(f"D1 {pool} authority record differs")
        frames[pool] = pd.read_parquet(path, columns=[*_IDENTITY_COLUMNS, *_FLOAT_COLUMNS])
    return manifest, frames


def _denominator_ids(manifest: Mapping[str, Any], *, d1_run: Path) -> set[str]:
    del d1_run
    configuration = manifest.get("configuration")
    paired = configuration.get("paired_manifest") if isinstance(configuration, Mapping) else None
    if not isinstance(paired, Mapping):
        raise D1PredictedReplayError("D1 candidate manifest lacks paired denominator")
    path = Path(str(paired.get("path", ""))).expanduser().resolve()
    record = artifact_record(_regular(path, label="D1 paired denominator"))
    if any(paired.get(key) != record[key] for key in ("path", "sha256", "bytes")):
        raise D1PredictedReplayError("D1 paired denominator record differs")
    frame = pd.read_parquet(path, columns=["sample_id"])
    identifiers = frame["sample_id"].astype(str)
    if identifiers.size != identifiers.nunique() or identifiers.eq("").any():
        raise D1PredictedReplayError("D1 paired denominator identities differ")
    return set(identifiers)


def _derived_oracle(path: Path) -> dict[str, int]:
    value = _self_hashed(path, label="derived baseline reconciliation")
    routes = value.get("routes")
    route = routes.get("d1") if isinstance(routes, Mapping) else None
    if value.get("status") != "PASS" or not isinstance(route, Mapping):
        raise D1PredictedReplayError("derived D1 baseline did not PASS")
    required = ("oracle_top5", "oracle_top10", "oracle_all")
    try:
        result = {key: int(route[key]) for key in required}
    except (KeyError, TypeError, ValueError) as error:
        raise D1PredictedReplayError("derived D1 oracle metrics are malformed") from error
    return result


def build_d1_predicted_replay_manifest(
    *,
    run_dir: Path,
    d1_run: Path,
    candidate_root: Path,
    frozen_candidate_root: Path,
    scored_root: Path,
    derived_reconciliation: Path,
    source_view_manifest: Path,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_candidates: int = EXPECTED_CANDIDATES,
    expected_no_output: int = EXPECTED_NO_OUTPUT,
) -> Path:
    """Reconstruct and compare the D1 replay without reading Test GT rows."""

    root = run_dir.expanduser().resolve()
    candidate_root = candidate_root.expanduser().resolve(strict=False)
    frozen_candidate_root = frozen_candidate_root.expanduser().resolve(strict=False)
    scored_root = scored_root.expanduser().resolve(strict=False)
    for label, path in (("candidate root", candidate_root), ("scored root", scored_root)):
        if root not in path.parents:
            raise D1PredictedReplayError(f"D1 replay {label} must be inside the new run")
    if frozen_candidate_root.is_symlink() or not frozen_candidate_root.is_dir():
        raise D1PredictedReplayError(
            "D1 frozen candidate root must be a regular directory"
        )
    candidate_manifest, authorities = _load_authorities(d1_run)
    source_view = verify_d1_source_view(source_view_manifest)
    sample_ids = _denominator_ids(candidate_manifest, d1_run=d1_run.expanduser().resolve())
    if len(sample_ids) != expected_samples:
        raise D1PredictedReplayError("D1 replay denominator differs")
    stage_replay = _candidate_stage_replay(
        candidate_root,
        frozen_candidate_root,
        expected_sample_ids=sample_ids,
        expected_no_output=expected_no_output,
    )
    if int(stage_replay["post_nms_candidate_count"]) != expected_candidates:
        raise D1PredictedReplayError("D1 replay NMS candidate count differs")
    try:
        actual, sample_sources, no_output = _scored_frame(
            candidate_root, scored_root, expected_sample_ids=sample_ids
        )
    except D1AdapterError as error:
        raise D1PredictedReplayError(str(error)) from error
    if actual.shape[0] != expected_candidates or no_output != expected_no_output:
        raise D1PredictedReplayError("D1 replay candidate/no-output counts differ")
    _assert_exact_frame(actual, authorities["allnms"], label="D1 AllNMS")
    _assert_exact_frame(_pool(actual, 10), authorities["top10"], label="D1 Top10")
    _assert_exact_frame(_pool(actual, 5), authorities["top5"], label="D1 Top5")
    top1 = _pool(actual, 1)
    frozen_top1 = _pool(authorities["allnms"], 1)
    _assert_exact_frame(top1, frozen_top1, label="D1 Top1")
    oracle = _derived_oracle(derived_reconciliation)
    source_inventory: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "sample_count": len(sample_sources),
        "candidate_count": int(actual.shape[0]),
        "samples": sample_sources,
    }
    source_inventory["content_sha256"] = canonical_sha256(source_inventory)
    output = root / "04_predicted_replay/d1"
    stage_inventory_path = output / "candidate_stage_replay.json"
    if stage_inventory_path.exists():
        existing = _self_hashed(
            stage_inventory_path, label="D1 candidate stage replay"
        )
        if existing != stage_replay:
            raise D1PredictedReplayError("existing D1 candidate stage replay differs")
    else:
        atomic_json(stage_inventory_path, stage_replay)
    inventory_path = output / "scored_source_inventory.json"
    if inventory_path.exists():
        existing = _self_hashed(inventory_path, label="D1 replay source inventory")
        if existing != source_inventory:
            raise D1PredictedReplayError("existing D1 replay source inventory differs")
    else:
        atomic_json(inventory_path, source_inventory)
    payload: dict[str, Any] = {
        "schema_version": 3,
        "status": "PASS",
        "route": "d1",
        "branch": "predicted",
        "sample_count": expected_samples,
        "candidate_count": expected_candidates,
        "no_output_count": expected_no_output,
        "no_output_sample_ids_sha256": stage_replay[
            "no_output_sample_ids_sha256"
        ],
        "raw_candidate_count": stage_replay["raw_candidate_count"],
        "mask_validated_candidate_count": stage_replay[
            "mask_validated_candidate_count"
        ],
        "serializer_atol": 0.0,
        "raw_test_ground_truth_rows_read": 0,
        "comparisons": {name: True for name in sorted(REQUIRED_COMPARISONS)},
        "oracle_metrics": oracle,
        "derived_reconciliation": artifact_record(derived_reconciliation),
        "source_view": artifact_record(source_view_manifest),
        "source_view_identity_sha256": source_view["source_identity_sha256"],
        "d1_final_lock": artifact_record(d1_run / "FINAL_RUN_LOCK.json"),
        "frozen_candidate_manifest": artifact_record(
            d1_run / "02_candidates/test_manifest.json"
        ),
        "frozen_pools": {
            pool: artifact_record(
                Path(str(candidate_manifest["artifacts"][pool]["path"]))
            )
            for pool in ("top5", "top10", "allnms")
        },
        "frozen_candidate_run": {
            "run_manifest": artifact_record(
                frozen_candidate_root.parents[1] / "run_manifest.json"
            ),
            "final_output_manifest": artifact_record(
                frozen_candidate_root.parents[1] / "final_output_manifest.json"
            ),
            "candidate_summary": artifact_record(
                frozen_candidate_root / "summary.csv"
            ),
        },
        "replay_roots": _root_records(candidate_root, scored_root),
        "candidate_stage_replay": artifact_record(stage_inventory_path),
        "scored_source_inventory": artifact_record(inventory_path),
    }
    payload["content_sha256"] = canonical_sha256(payload)
    destination = output / "manifest.json"
    if destination.exists():
        existing = _self_hashed(destination, label="D1 predicted replay manifest")
        if existing != payload:
            raise D1PredictedReplayError("existing D1 predicted replay manifest differs")
    else:
        atomic_json(destination, payload)
    return destination


def validate_d1_predicted_replay_manifest(
    path: str | Path,
    *,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_candidates: int = EXPECTED_CANDIDATES,
    expected_no_output: int = EXPECTED_NO_OUTPUT,
) -> dict[str, Any]:
    """Replay the semantic comparison from a saved D1 manifest."""

    source = _regular(path, label="D1 predicted replay manifest")
    value = _self_hashed(source, label="D1 predicted replay manifest")
    if source.name != "manifest.json" or source.parent.name != "d1":
        raise D1PredictedReplayError("D1 predicted replay manifest path differs")
    roots = value.get("replay_roots")
    lock_record = value.get("d1_final_lock")
    candidate_manifest_record = value.get("frozen_candidate_manifest")
    derived_record = value.get("derived_reconciliation")
    source_view_record = value.get("source_view")
    frozen_candidate_run = value.get("frozen_candidate_run")
    if not all(
        isinstance(item, Mapping)
        for item in (
            roots,
            lock_record,
            candidate_manifest_record,
            derived_record,
            source_view_record,
            frozen_candidate_run,
        )
    ):
        raise D1PredictedReplayError("D1 predicted replay bindings are malformed")
    d1_run = Path(str(lock_record["path"])).expanduser().resolve().parent
    candidate_root = Path(str(roots["candidate_summary"]["path"])).resolve().parent
    frozen_candidate_root = Path(
        str(frozen_candidate_run["candidate_summary"]["path"])
    ).resolve().parent
    scored_root = Path(str(roots["scorer_progress"]["path"])).resolve().parent
    rebuilt = build_d1_predicted_replay_manifest(
        run_dir=source.parents[2],
        d1_run=d1_run,
        candidate_root=candidate_root,
        frozen_candidate_root=frozen_candidate_root,
        scored_root=scored_root,
        derived_reconciliation=Path(str(derived_record["path"])),
        source_view_manifest=Path(str(source_view_record["path"])),
        expected_samples=expected_samples,
        expected_candidates=expected_candidates,
        expected_no_output=expected_no_output,
    )
    observed = _self_hashed(rebuilt, label="rebuilt D1 predicted replay manifest")
    if observed != value:
        raise D1PredictedReplayError("D1 predicted replay semantic replay differs")
    return value


__all__ = [
    "D1PredictedReplayError",
    "REQUIRED_COMPARISONS",
    "build_d1_predicted_replay_manifest",
    "validate_d1_predicted_replay_manifest",
]
