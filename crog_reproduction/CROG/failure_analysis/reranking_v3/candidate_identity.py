from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import atomic_write_json, read_jsonl, sha256_file, stable_sample_id


IDENTITY_FIELDS = (
    "candidate_id",
    "candidate_checksum",
    "q_rank",
    "q_raw",
    "cx",
    "cy",
    "col",
    "row",
    "angle_deg",
    "angle_rad",
    "width_px",
    "height_px",
    "polygon",
    "legacy_grasp",
)


def _stable_feature_id(record: dict[str, Any]) -> str:
    return stable_sample_id(record["split"], record["sample_id"])


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(candidate.get(name) for name in IDENTITY_FIELDS)


def _stream_feature_index(path: str | Path) -> dict[str, tuple[tuple[Any, ...], ...]]:
    result: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for record in read_jsonl(path):
        sample_id = _stable_feature_id(record)
        if sample_id in result:
            raise ValueError(f"duplicate sample ID in frozen features: {sample_id}")
        candidates = record["candidates"]
        if len(candidates) != 5:
            raise ValueError(f"{sample_id} has {len(candidates)} candidates, expected 5")
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate ID within {sample_id}")
        if [int(candidate["q_rank"]) for candidate in candidates] != list(range(5)):
            raise ValueError(f"q rank changed for {sample_id}")
        q = [float(candidate["q_raw"]) for candidate in candidates]
        if any(q[index] < q[index + 1] for index in range(4)):
            raise ValueError(f"q ordering is not monotone for {sample_id}")
        result[sample_id] = tuple(_candidate_signature(candidate) for candidate in candidates)
    return result


def verify_candidate_identity(
    *,
    v1_features: str | Path,
    prediction_paths: Iterable[str | Path],
    enhanced_index_paths: Iterable[str | Path] = (),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    source = _stream_feature_index(v1_features)
    candidate_pairs = sum(len(value) for value in source.values())
    prediction_checks = []
    for prediction_path in prediction_paths:
        seen = set()
        mismatches = 0
        for record in read_jsonl(prediction_path):
            raw_id = str(record["sample_id"])
            sample_id = (
                raw_id
                if raw_id.startswith("multiple:")
                else stable_sample_id(record["split"], raw_id)
            )
            if sample_id not in source or sample_id in seen:
                mismatches += 1
                continue
            seen.add(sample_id)
            if "candidate_order" in record:
                expected = {signature[0] for signature in source[sample_id]}
                observed = list(map(str, record["candidate_order"]))
                if len(observed) != 5 or set(observed) != expected:
                    mismatches += 1
            elif "candidates" in record:
                observed_signatures = tuple(
                    _candidate_signature(candidate)
                    for candidate in record["candidates"]
                )
                if observed_signatures != source[sample_id]:
                    mismatches += 1
            else:
                mismatches += 1
        missing = len(set(source) - seen)
        prediction_checks.append(
            {
                "path": str(Path(prediction_path).resolve()),
                "sha256": sha256_file(prediction_path),
                "rows": len(seen),
                "missing_samples": missing,
                "mismatches": mismatches,
                "passed": missing == 0 and mismatches == 0,
            }
        )
    enhanced_checks = []
    for index_path in enhanced_index_paths:
        seen = set()
        mismatches = 0
        for record in read_jsonl(index_path):
            sample_id = str(record["sample_id"])
            seen.add(sample_id)
            expected_signatures = source.get(sample_id, ())
            expected = [signature[0] for signature in expected_signatures]
            expected_checksums = [signature[1] for signature in expected_signatures]
            if expected and (
                list(map(str, record.get("candidate_ids", []))) != expected
                or list(map(str, record.get("candidate_checksums", [])))
                != expected_checksums
            ):
                mismatches += 1
        enhanced_checks.append(
            {
                "path": str(Path(index_path).resolve()),
                "sha256": sha256_file(index_path),
                "rows": len(seen),
                "mismatches": mismatches,
                "passed": mismatches == 0,
            }
        )
    result = {
        "v1_features": str(Path(v1_features).resolve()),
        "v1_features_sha256": sha256_file(v1_features),
        "unique_samples": len(source),
        "unique_candidate_pairs": candidate_pairs,
        "candidates_per_sample": 5,
        "identity_fields": list(IDENTITY_FIELDS),
        "prediction_checks": prediction_checks,
        "enhanced_checks": enhanced_checks,
        "passed": all(item["passed"] for item in prediction_checks + enhanced_checks),
    }
    if not result["passed"]:
        raise AssertionError("candidate identity failed; formal training/test is prohibited")
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result
