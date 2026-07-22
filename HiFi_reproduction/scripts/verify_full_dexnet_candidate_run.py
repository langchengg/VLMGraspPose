#!/usr/bin/env python3
"""Verify and aggregate a full manifest-backed Dex-Net candidate run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "gqcnn-official"))

from autolab_core import YamlConfig  # noqa: E402

from src.grasping.dexnet_run_reliability import (  # noqa: E402
    FAILED,
    SUCCESS_EMPTY,
    SUCCESS_NONEMPTY,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
    validate_sample_output,
)


SUMMARY_FIELDS = (
    "sample_id",
    "query",
    "mask_area_px",
    "valid_target_depth_px",
    "requested_candidate_count",
    "raw_candidate_count",
    "mask_validated_count",
    "post_nms_count",
    "scored_candidate_count",
    "best_gqcnn_q",
    "median_gqcnn_q",
    "generation_time_ms",
    "scoring_time_ms",
    "total_time_ms",
    "failure_reason",
    "status",
    "question_index",
    "scene_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--no-write-reports", action="store_true")
    return parser.parse_args()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolved_config(path: Path, seed: int, num_candidates: int | None, top_k: int | None) -> dict[str, Any]:
    config = _plain(dict(YamlConfig(str(path))))
    generation = dict(config["generation"])
    generation["seed"] = int(seed)
    if num_candidates is not None:
        generation["num_grasp_samples"] = int(num_candidates)
    if top_k is not None:
        generation["top_k"] = int(top_k)
    config["generation"] = generation
    return config


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) and row.get("sample_id") for row in rows):
        raise ValueError("expected manifest must contain object rows with sample_id")
    return rows


def numeric(values: Iterable[Any]) -> np.ndarray:
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            parsed.append(number)
    return np.asarray(parsed, dtype=np.float64)


def safe_int(value: Any, default: int = 0) -> int:
    """Parse an integer-like report value without crashing on failed rows."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def describe(values: Iterable[Any], *, percentiles: bool = True) -> dict[str, Any]:
    array = numeric(values)
    if not array.size:
        return {"count": 0}
    result: dict[str, Any] = {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array)),
    }
    if percentiles:
        result["percentiles"] = {
            str(q): float(np.percentile(array, q))
            for q in (0, 1, 5, 25, 50, 75, 90, 95, 99, 100)
        }
    return result


def _group_counts(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return {
        name: {
            "samples": len(members),
            "success_nonempty": sum(row.get("status") == SUCCESS_NONEMPTY for row in members),
            "success_empty": sum(row.get("status") == SUCCESS_EMPTY for row in members),
            "failed": sum(row.get("status") == FAILED for row in members),
            "post_nms": describe((row.get("post_nms_count") for row in members), percentiles=False),
        }
        for name, members in sorted(groups.items())
    }


def _quartile_groups(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    usable = [row for row in rows if numeric([row.get(field)]).size]
    if not usable:
        return {}
    values = np.asarray([float(row[field]) for row in usable], dtype=np.float64)
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        quartile = int(np.searchsorted(edges, float(row[field]), side="right")) + 1
        groups[f"Q{quartile}"] .append(row)
    return {
        "edges": [float(value) for value in edges],
        "groups": {
            name: {
                "samples": len(members),
                "success_nonempty": sum(row.get("status") == SUCCESS_NONEMPTY for row in members),
                "success_empty": sum(row.get("status") == SUCCESS_EMPTY for row in members),
                "failed": sum(row.get("status") == FAILED for row in members),
                "post_nms": describe((row.get("post_nms_count") for row in members), percentiles=False),
            }
            for name, members in sorted(groups.items())
        },
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _history(root: Path) -> list[dict[str, Any]]:
    path = root / "attempt_history.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"event": "corrupt_history_line", "line": line})
    return rows


def _command_option(command: list[Any], option: str, default: Any) -> Any:
    values = [str(value) for value in command]
    try:
        return values[values.index(option) + 1]
    except (ValueError, IndexError):
        return default


def _run_validation_context(root: Path) -> tuple[dict[str, Any] | None, str, int]:
    path = root / "run_config.json"
    if not path.is_file():
        return None, "all", 100
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload.get("gqcnn_runtime")
    execution = payload.get("execution", {})
    command = payload.get("initial_command_line", [])
    policy = str(
        execution.get(
            "visualize_policy",
            _command_option(command, "--visualize-policy", "all"),
        )
    )
    every = safe_int(
        execution.get(
            "visualize_every",
            _command_option(command, "--visualize-every", 100),
        ),
        100,
    )
    return runtime if isinstance(runtime, dict) else None, policy, max(1, every)


def _expect_visualizations(
    sample_dir: Path, *, policy: str, selected_position: int, every: int
) -> bool | None:
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "sampled":
        return selected_position % every == 0
    if policy == "failures":
        try:
            marker = json.loads((sample_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
            return marker.get("status") in {SUCCESS_EMPTY, FAILED}
        except Exception:
            return None
    return None


def aggregate_statistics(
    root: Path,
    expected_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    missing: list[str],
    corrupt: Mapping[str, list[str]],
) -> dict[str, Any]:
    nonempty = [row for row in summary_rows if row.get("status") == SUCCESS_NONEMPTY]
    empty = [row for row in summary_rows if row.get("status") == SUCCESS_EMPTY]
    failed = [row for row in summary_rows if row.get("status") == FAILED]
    completed = nonempty + empty
    requested_top_k = max((safe_int(row.get("post_nms_count")) for row in completed), default=0)
    run_config_path = root / "run_config.json"
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        requested_top_k = int(run_config.get("identity", {}).get("top_k", requested_top_k))
    history = _history(root)
    completions = []
    for row in summary_rows:
        marker = root / str(row["sample_id"]) / "_SUCCESS.json"
        if marker.is_file():
            try:
                completions.append(json.loads(marker.read_text())["completion_timestamp"])
            except Exception:
                pass
    failure_reasons = Counter(str(row.get("failure_reason") or "unspecified") for row in failed)
    empty_reasons = Counter(str(row.get("failure_reason") or "unspecified") for row in empty)
    post_counts = [safe_int(row.get("post_nms_count")) for row in completed]
    distribution = Counter(post_counts)
    expected = len(expected_rows)
    execution_success = len(completed)
    invocation_starts = [
        str(row["timestamp"])
        for row in history
        if row.get("event") == "invocation_started" and row.get("timestamp")
    ]
    invocation_finishes = [
        str(row["timestamp"])
        for row in history
        if row.get("event") == "invocation_finished" and row.get("timestamp")
    ]
    return {
        "total_expected_samples": expected,
        "total_executed": len(summary_rows),
        "total_nonempty_successes": len(nonempty),
        "total_empty_successes": len(empty),
        "total_failures": len(failed),
        "total_missing": len(missing),
        "total_corrupt": len(corrupt),
        "execution_success_rate": execution_success / expected if expected else None,
        "nonempty_candidate_rate": len(nonempty) / expected if expected else None,
        "raw_candidate_count": describe(row.get("raw_candidate_count") for row in completed),
        "mask_valid_candidate_count": describe(row.get("mask_validated_count") for row in completed),
        "post_nms_candidate_count": describe(post_counts),
        "generation_runtime_ms": describe(row.get("generation_time_ms") for row in completed),
        "total_generation_runtime_seconds": float(
            np.sum(numeric(row.get("generation_time_ms") for row in completed)) / 1000.0
        ),
        "mask_area_px": describe(row.get("mask_area_px") for row in completed),
        "valid_target_depth_px": describe(row.get("valid_target_depth_px") for row in completed),
        "candidate_count_distribution": {str(key): value for key, value in sorted(distribution.items())},
        "failure_reason_distribution": dict(sorted(failure_reasons.items())),
        "empty_candidate_reason_distribution": dict(sorted(empty_reasons.items())),
        "samples_reaching_requested_256_raw_candidates": sum(
            safe_int(row.get("raw_candidate_count"), -1) == 256 for row in completed
        ),
        "samples_with_fewer_than_top_k_candidates": sum(value < requested_top_k for value in post_counts),
        "requested_top_k": requested_top_k,
        "disk_usage_bytes": _directory_size(root),
        "start_timestamp": min(invocation_starts) if invocation_starts else None,
        "end_timestamp": max(invocation_finishes) if invocation_finishes else None,
        "sample_completion_start_timestamp": min(completions) if completions else None,
        "sample_completion_end_timestamp": max(completions) if completions else None,
        "resume_events": sum(
            row.get("event") == "invocation_started" and row.get("resume") is True
            for row in history
        ),
        "invocation_count": sum(row.get("event") == "invocation_started" for row in history),
        "by_scene_or_image": _group_counts(summary_rows, "scene_id"),
        "by_mask_area_quartile": _quartile_groups(completed, "mask_area_px"),
        "by_valid_depth_quartile": _quartile_groups(completed, "valid_target_depth_px"),
        "target_object_category": None,
        "target_object_category_note": "No canonical object-category label is present in the frozen manifest; none was inferred from query text.",
        "query_type": None,
        "query_type_note": "No canonical query-type label is present in the frozen manifest; none was inferred.",
    }


def main() -> int:
    args = parse_args()
    root = args.candidate_root.expanduser().resolve()
    manifest_path = args.expected_manifest.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    expected_rows = load_manifest(manifest_path)
    ids = [str(row["sample_id"]) for row in expected_rows]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    expected_set = set(ids)
    config = resolved_config(config_path, args.seed, args.num_candidates, args.top_k)
    configuration_hash = canonical_json_hash(config)
    config_file_hash = sha256_file(config_path)
    expected_runtime, visualization_policy, visualization_every = _run_validation_context(root)

    unexpected = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in expected_set
    ) if root.is_dir() else []
    missing: list[str] = []
    corrupt: dict[str, list[str]] = {}
    config_mismatches: list[str] = []
    seed_mismatches: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    empty_records: list[dict[str, Any]] = []

    for selected_position, sample_id in enumerate(ids):
        sample_dir = root / sample_id
        if not sample_dir.is_dir():
            missing.append(sample_id)
            continue
        validation = validate_sample_output(
            sample_dir,
            expected_sample_id=sample_id,
            expected_configuration_hash=configuration_hash,
            expected_config_file_sha256=config_file_hash,
            expected_seed=args.seed,
            expected_sampler_runtime=expected_runtime,
            expect_visualizations=_expect_visualizations(
                sample_dir,
                policy=visualization_policy,
                selected_position=selected_position,
                every=visualization_every,
            ),
            verify_hashes=True,
        )
        if not validation.valid:
            corrupt[sample_id] = validation.errors
            if any("configuration" in error or "config-file" in error for error in validation.errors):
                config_mismatches.append(sample_id)
            if any("seed mismatch" in error for error in validation.errors):
                seed_mismatches.append(sample_id)
            continue
        if validation.summary_row is None:
            corrupt[sample_id] = ["valid terminal marker has no summary row"]
            continue
        row = dict(validation.summary_row)
        row["status"] = validation.status
        summary_rows.append(row)
        if validation.status == FAILED:
            failure_payload: dict[str, Any] | None = None
            failure_path = sample_dir / "failure.json"
            if failure_path.is_file():
                try:
                    loaded_failure = json.loads(failure_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_failure, dict):
                        failure_payload = loaded_failure
                except Exception as error:
                    failure_payload = {
                        "unavailable_reason": f"failure.json unreadable: {type(error).__name__}: {error}"
                    }
            failed_records.append(
                {
                    "sample_id": sample_id,
                    "status": FAILED,
                    "failure_reason": row.get("failure_reason") or "unspecified",
                    "summary_row": row,
                    "failure": failure_payload,
                    "completion_marker": validation.marker,
                }
            )
        elif validation.status == SUCCESS_EMPTY:
            empty_records.append(row)

    counts = Counter(str(row.get("status")) for row in summary_rows)
    accounting_identity = (
        counts[SUCCESS_NONEMPTY]
        + counts[SUCCESS_EMPTY]
        + counts[FAILED]
        + len(missing)
        + len(corrupt)
    )
    statistics = aggregate_statistics(
        root,
        expected_rows,
        summary_rows,
        missing=missing,
        corrupt=corrupt,
    )
    report = {
        "expected_samples": len(ids),
        "complete_nonempty_samples": counts[SUCCESS_NONEMPTY],
        "complete_empty_samples": counts[SUCCESS_EMPTY],
        "failed_samples": counts[FAILED],
        "missing_samples": len(missing),
        "corrupt_samples": len(corrupt),
        "duplicate_sample_ids": duplicate_ids,
        "unexpected_output_directories": unexpected,
        "configuration_hash_mismatches": config_mismatches,
        "seed_mismatches": seed_mismatches,
        "total_raw_candidates": sum(safe_int(row.get("raw_candidate_count")) for row in summary_rows),
        "total_mask_valid_candidates": sum(
            safe_int(row.get("mask_validated_count")) for row in summary_rows
        ),
        "total_post_nms_candidates": sum(
            safe_int(row.get("post_nms_count")) for row in summary_rows
        ),
        "accounting_identity": accounting_identity,
        "accounting_identity_expected": len(ids),
        "missing_ids": missing,
        "corrupt_details": corrupt,
        "configuration_hash": configuration_hash,
        "config_file_sha256": config_file_hash,
        "expected_manifest_sha256": sha256_file(manifest_path),
        "expected_sampler_runtime": expected_runtime,
        "visualization_policy": visualization_policy,
        "visualization_every": visualization_every,
    }
    if not args.no_write_reports and root.is_dir():
        manifest_index = {sample_id: index for index, sample_id in enumerate(ids)}
        ordered = sorted(summary_rows, key=lambda row: manifest_index[str(row["sample_id"])])
        atomic_write_csv(root / "summary.csv", ordered, SUMMARY_FIELDS)
        atomic_write_json(root / "summary.json", {"verification": report, "samples": ordered})
        atomic_write_json(root / "failures.json", failed_records)
        atomic_write_json(root / "empty_candidates.json", empty_records)
        atomic_write_json(root / "run_statistics.json", statistics)
        atomic_write_json(root / "verification_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    failure = bool(
        missing
        or corrupt
        or duplicate_ids
        or unexpected
        or config_mismatches
        or seed_mismatches
        or accounting_identity != len(ids)
        or (counts[FAILED] and not args.allow_failures)
    )
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
