from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .security import (
    EVALUATION_ONLY_FIELDS,
    FORBIDDEN_INFERENCE_FIELDS,
    INFERENCE_INPUT_ALLOWLIST,
)


EXPECTED_BASELINE = {
    "sample_count": 17749,
    "candidate_count": 88745,
    "legacy_q_only_success_count": 14768,
    "legacy_oracle_success_count": 16129,
    "corrected_q_only_success_count": 15840,
    "corrected_oracle_success_count": 16746,
}

EXPECTED_PARTITIONS = {
    "train": 53431,
    "calibration": 9790,
    "validation": 8669,
    "test": 17749,
}

PREEXISTING_V2_TREE_SHA256 = "2c0791359d252af73987d180ae1852ac86d3200ebfe2ac93d74a18ad4581481c"
EXPECTED_TEST_FEATURES_SHA256 = "04f467d669a4cfb2ccafca61e9dee18ebaa18c093133798533a2761221f652aa"
EXPECTED_CANDIDATE_IDENTITY_STREAM_SHA256 = (
    "34e773b06119619bd45771f8e000c317ecc768d51a132ef6dc7b40e5f3faa36c"
)


def _jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc


def stable_sample_id(split: str, local_id: int | str) -> str:
    normalized = "val" if str(split).lower() in {"val", "validation"} else str(split).lower()
    return f"multiple:{normalized}:{int(local_id):08d}"


def audit_frozen_test_baseline(
    *,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    corrected_labels_path: str | Path,
) -> dict[str, Any]:
    sample_count = candidate_count = five_count = q_order_mismatches = 0
    identity = hashlib.sha256()
    legacy_top1 = legacy_oracle = corrected_top1 = corrected_oracle = 0
    exact_ties: list[dict[str, Any]] = []
    for feature, legacy, corrected in zip(
        _jsonl(features_path), _jsonl(legacy_labels_path), _jsonl(corrected_labels_path), strict=True
    ):
        sample_count += 1
        expected_id = stable_sample_id(feature["split"], feature["sample_id"])
        if expected_id != legacy["sample_id"] or expected_id != corrected["sample_id"]:
            raise AssertionError(f"feature/dual-label identity mismatch at {expected_id}")
        candidates = feature["candidates"]
        candidate_count += len(candidates)
        five_count += int(len(candidates) == 5)
        if len(candidates) != 5:
            raise AssertionError(f"{expected_id} has {len(candidates)} candidates")
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item["q_raw"]), str(item["candidate_id"])),
        )
        q_order_mismatches += int(
            [item["candidate_id"] for item in ordered]
            != [item["candidate_id"] for item in candidates]
        )
        grouped = Counter(float(item["q_raw"]) for item in candidates)
        for value, count in grouped.items():
            if count > 1:
                exact_ties.append({"sample_id": expected_id, "q_raw": value, "count": count})
        legacy_by_id = {item["candidate_id"]: item for item in legacy["candidate_labels"]}
        corrected_by_id = {item["candidate_id"]: item for item in corrected["candidate_labels"]}
        top_id = str(ordered[0]["candidate_id"])
        legacy_top1 += int(legacy_by_id[top_id]["candidate_correct"])
        corrected_top1 += int(corrected_by_id[top_id]["candidate_correct"])
        legacy_oracle += int(any(item["candidate_correct"] for item in legacy["candidate_labels"]))
        corrected_oracle += int(any(item["candidate_correct"] for item in corrected["candidate_labels"]))
        for rank, candidate in enumerate(ordered):
            payload = {
                "sample_id": expected_id,
                "candidate_id": str(candidate["candidate_id"]),
                "original_rank": int(rank),
                "q_raw": float(candidate["q_raw"]),
                "candidate_checksum": str(candidate["candidate_checksum"]),
            }
            identity.update(
                (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            )
    observed = {
        "sample_count": sample_count,
        "candidate_count": candidate_count,
        "five_candidates_per_sample_count": five_count,
        "q_order_mismatch_count": q_order_mismatches,
        "legacy_q_only_success_count": legacy_top1,
        "legacy_oracle_success_count": legacy_oracle,
        "corrected_q_only_success_count": corrected_top1,
        "corrected_oracle_success_count": corrected_oracle,
        "legacy_q_only_j1": legacy_top1 / sample_count,
        "legacy_oracle_at_5": legacy_oracle / sample_count,
        "corrected_q_only_j1": corrected_top1 / sample_count,
        "corrected_oracle_at_5": corrected_oracle / sample_count,
        "candidate_identity_stream_sha256": identity.hexdigest(),
        "exact_q_ties": exact_ties,
    }
    for name, expected in EXPECTED_BASELINE.items():
        if observed[name] != expected:
            raise AssertionError(f"baseline mismatch for {name}: {observed[name]} != {expected}")
    if five_count != sample_count or q_order_mismatches:
        raise AssertionError("frozen candidate count/order audit failed")
    if observed["candidate_identity_stream_sha256"] != EXPECTED_CANDIDATE_IDENTITY_STREAM_SHA256:
        raise AssertionError("frozen candidate identity differs from the preregistered baseline")
    return observed


def audit_split_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["rows"]
    counts = Counter(str(item["development_partition"]) for item in rows)
    for name, expected in EXPECTED_PARTITIONS.items():
        if counts[name] != expected:
            raise AssertionError(f"partition mismatch for {name}: {counts[name]} != {expected}")
    if not payload["audit"]["required_zero_overlap_passed"]:
        raise AssertionError("split manifest reports forbidden overlap")
    overlap = payload["audit"]["pairwise_overlap"]
    forbidden_overlap = {
        pair: {name: count for name, count in fields.items() if count}
        for pair, fields in overlap.items()
        if any(fields.values())
    }
    if forbidden_overlap:
        raise AssertionError(f"partition leakage: {forbidden_overlap}")
    return {
        "partition_counts": dict(counts),
        "pairwise_overlap": overlap,
        "required_zero_overlap_passed": True,
        "split_manifest_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def v2_tree_digest(path: str | Path) -> str:
    """Match the immutable V2 audit's `shasum | shasum` tree digest."""
    root = Path(path)
    # `shasum` includes the path text in every line.  The frozen digest was
    # created from the repository-relative V2 path, so an absolute spelling of
    # the identical tree must be normalised before hashing.  Without this,
    # `cli verify-v2` reports a false identity change solely because V2_ROOT is
    # absolute.
    if root.is_absolute():
        try:
            root = root.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            pass
    command = (
        f"find {json.dumps(str(root))} -type f -print0 | "
        "LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256"
    )
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout.split()[0]


def write_phase0_artifacts(
    *,
    output_dir: str | Path,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    corrected_labels_path: str | Path,
    split_manifest_path: str | Path,
    v2_root: str | Path,
    verify_v2_tree: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    baseline = audit_frozen_test_baseline(
        features_path=features_path,
        legacy_labels_path=legacy_labels_path,
        corrected_labels_path=corrected_labels_path,
    )
    split = audit_split_manifest(split_manifest_path)
    (output / "inference_input_allowlist.json").write_text(
        json.dumps(INFERENCE_INPUT_ALLOWLIST, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "forbidden_inference_fields.json").write_text(
        json.dumps(list(FORBIDDEN_INFERENCE_FIELDS), indent=2) + "\n", encoding="utf-8"
    )
    (output / "evaluation_only_fields.json").write_text(
        json.dumps(list(EVALUATION_ONLY_FIELDS), indent=2) + "\n", encoding="utf-8"
    )
    v2_before = PREEXISTING_V2_TREE_SHA256
    if verify_v2_tree:
        v2_before = v2_tree_digest(v2_root)
        if v2_before != PREEXISTING_V2_TREE_SHA256:
            raise AssertionError("pre-existing V2 tree hash changed before experiment")
    result = {
        "status": "phase0_passed",
        "baseline": baseline,
        "split_audit": split,
        "v2_tree_sha256_before": v2_before,
        "v2_tree_hash_verified_live": bool(verify_v2_tree),
        "formal_test_allowed": False,
        "reason_formal_test_blocked": "no Gemini budget/validation lock at Phase 0",
    }
    (output / "phase0_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    material_passport = {
        "kind": "material_passport",
        "sources": {
            "frozen_test_candidates": str(Path(features_path).resolve()),
            "legacy_evaluation_labels": str(Path(legacy_labels_path).resolve()),
            "corrected_evaluation_labels": str(Path(corrected_labels_path).resolve()),
            "split_manifest": str(Path(split_manifest_path).resolve()),
        },
        "inference_inputs": list(INFERENCE_INPUT_ALLOWLIST["raw_inputs"])
        + list(INFERENCE_INPUT_ALLOWLIST["crog_predictions"])
        + list(INFERENCE_INPUT_ALLOWLIST["derived_without_ground_truth"]),
        "evaluation_only": list(EVALUATION_ONLY_FIELDS),
        "candidate_generation": "frozen CROG Top-5 only",
        "formal_test_disclosure": "post-hoc benchmark extension on a previously used test split",
    }
    (output / "material_passport.json").write_text(
        json.dumps(material_passport, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "gemini_api_key": "SET" if os.environ.get("GEMINI_API_KEY") else "UNSET",
        "gemini_max_spend_usd": "SET" if os.environ.get("GEMINI_MAX_SPEND_USD") else "UNSET",
        "gemini_max_concurrency": os.environ.get("GEMINI_MAX_CONCURRENCY", "2"),
    }
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
