#!/usr/bin/env python3
"""Verify complete GQCNN-2.1 scoring of immutable frozen candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.gqcnn_full_scoring import (  # noqa: E402
    GQCNN_COMMIT,
    SCORED_NONEMPTY,
    SCORING_FAILED,
    SKIPPED_VALID_EMPTY,
    atomic_write_json,
    assert_disjoint_roots,
    load_source_manifest,
    load_source_sample,
    model_file_manifest,
    select_entries,
    sha256_file,
    validate_scored_output,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expect-total-samples", type=int)
    parser.add_argument("--expect-scored-nonempty", type=int)
    parser.add_argument("--expect-valid-empty", type=int)
    parser.add_argument("--expect-candidates", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_root, scored_root = assert_disjoint_roots(
        args.candidate_root, args.scored_root
    )
    model_dir = args.model_dir.expanduser().resolve()
    source_manifest = (
        args.source_manifest.expanduser().resolve()
        if args.source_manifest is not None
        else scored_root / "source_candidate_manifest.jsonl"
    )
    run_config = json.loads((scored_root / "run_config.json").read_text(encoding="utf-8"))
    model_info = run_config["model"]
    model_files, current_model_manifest_hash = model_file_manifest(model_dir)
    del model_files
    global_errors: list[str] = []
    manifest_hash = sha256_file(source_manifest)
    recorded_manifest_hash = run_config.get("source_identity", {}).get(
        "source_manifest_sha256"
    )
    if recorded_manifest_hash != manifest_hash:
        global_errors.append("source manifest/run config hash mismatch")
    if model_info.get("model_name") != "GQCNN-2.1":
        global_errors.append("model name is not GQCNN-2.1")
    if model_info.get("model_commit") != GQCNN_COMMIT:
        global_errors.append("GQ-CNN commit mismatch")
    if not model_info.get("docker_image_id"):
        global_errors.append("Docker image ID missing")
    if sha256_file(model_dir / "config.json") != model_info["model_config_hash"]:
        global_errors.append("model config hash mismatch")
    if current_model_manifest_hash != model_info["model_file_manifest_hash"]:
        global_errors.append("model file manifest hash mismatch")
    entries = load_source_manifest(source_manifest)
    selected = select_entries(
        entries,
        sample_ids=args.sample_id,
        sample_limit=args.sample_limit,
        start_index=args.start_index,
        end_index=args.end_index,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    selected_ids = {entry["sample_id"] for entry in selected}
    expected_nonempty = sum(int(entry["candidate_count"]) > 0 for entry in selected)
    expected_empty = sum(int(entry["candidate_count"]) == 0 for entry in selected)
    expected_candidates = sum(int(entry["candidate_count"]) for entry in selected)
    expectations = (
        ("total samples", len(selected), args.expect_total_samples),
        ("scored nonempty", expected_nonempty, args.expect_scored_nonempty),
        ("valid empty", expected_empty, args.expect_valid_empty),
        ("candidates", expected_candidates, args.expect_candidates),
    )
    for label, actual, expected in expectations:
        if expected is not None and actual != expected:
            global_errors.append(
                "source selection %s expected %s, got %s" % (label, expected, actual)
            )

    scored_nonempty = 0
    skipped_empty = 0
    failed = 0
    missing: list[str] = []
    corrupt: dict[str, list[str]] = {}
    source_mismatches: dict[str, str] = {}
    finite_q_values = 0
    scored_candidates = 0
    duplicate_candidate_ids: list[str] = []
    for index, entry in enumerate(selected, start=1):
        sample_id = entry["sample_id"]
        source_dir = candidate_root / sample_id
        output_dir = scored_root / sample_id
        try:
            source = load_source_sample(
                source_dir, expected_entry=entry, verify_hashes=True
            )
            if len(source["candidate_ids"]) != len(set(source["candidate_ids"])):
                duplicate_candidate_ids.append(sample_id)
        except Exception as error:  # source failure is fatal and separately visible
            source_mismatches[sample_id] = "%s: %s" % (type(error).__name__, error)
            continue
        if not output_dir.is_dir():
            missing.append(sample_id)
            continue
        valid, status, marker, errors = validate_scored_output(
            output_dir,
            source_dir,
            entry,
            model_info,
            args.seed,
            verify_hashes=True,
        )
        if not valid:
            corrupt[sample_id] = errors
            continue
        if status == SCORED_NONEMPTY:
            scored_nonempty += 1
            count = int(marker["gqcnn_scored_count"])
            scored_candidates += count
            with np.load(
                output_dir / "gqcnn_scored_candidates.npz", allow_pickle=False
            ) as archive:
                values = np.asarray(archive["gqcnn_q_value"], dtype=np.float64)
            finite_q_values += int(np.isfinite(values).sum())
        elif status == SKIPPED_VALID_EMPTY:
            skipped_empty += 1
        elif status == SCORING_FAILED:
            failed += 1
        if index % 500 == 0:
            print(
                json.dumps(
                    {
                        "status": "VERIFY_PROGRESS",
                        "checked_samples": index,
                        "selected_samples": len(selected),
                        "scored_candidates": scored_candidates,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    actual_dirs = {
        path.name
        for path in scored_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    canonical_ids = {entry["sample_id"] for entry in entries}
    unexpected = sorted(actual_dirs - canonical_ids)
    report = {
        "schema_version": 1,
        "verified_at": utc_timestamp(),
        "candidate_root": str(candidate_root),
        "scored_root": str(scored_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": manifest_hash,
        "model_name": model_info["model_name"],
        "model_commit": model_info["model_commit"],
        "model_config_hash": model_info["model_config_hash"],
        "model_file_manifest_hash": model_info["model_file_manifest_hash"],
        "expected_total_samples": len(selected),
        "expected_scored_nonempty_samples": expected_nonempty,
        "expected_skipped_valid_empty_samples": expected_empty,
        "expected_frozen_candidates": expected_candidates,
        "terminal_samples": scored_nonempty + skipped_empty + failed,
        "scored_nonempty_samples": scored_nonempty,
        "skipped_valid_empty_samples": skipped_empty,
        "failed_samples": failed,
        "missing_samples": len(missing),
        "missing_sample_ids": missing,
        "corrupt_samples": len(corrupt),
        "corrupt_details": corrupt,
        "source_mismatch_samples": len(source_mismatches),
        "source_mismatch_details": source_mismatches,
        "scored_candidates": scored_candidates,
        "finite_q_values": finite_q_values,
        "invalid_q_values": scored_candidates - finite_q_values,
        "duplicate_candidate_id_samples": duplicate_candidate_ids,
        "unexpected_output_directories": unexpected,
        "global_errors": global_errors,
        "pose_integrity": "checked elementwise by scored-output validator",
        "ranking_integrity": "checked raw full-precision q descending, exact-ID ties, one-based ranks",
    }
    report["accounting_identity"] = scored_nonempty + skipped_empty + failed
    report["clean"] = bool(
        not global_errors
        and report["accounting_identity"] == len(selected)
        and scored_nonempty == expected_nonempty
        and skipped_empty == expected_empty
        and failed == 0
        and not missing
        and not corrupt
        and not source_mismatches
        and scored_candidates == expected_candidates
        and finite_q_values == expected_candidates
        and not duplicate_candidate_ids
        and not unexpected
    )
    atomic_write_json(scored_root / "verification_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["clean"] else 2


if __name__ == "__main__":
    sys.exit(main())
