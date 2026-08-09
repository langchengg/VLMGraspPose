#!/usr/bin/env python3
"""Compare two genuinely independent, cache-cold local-VLM repeat runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=20)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    left = _rows(run_a)
    right = _rows(run_b)
    if len(left) != args.expected_samples or len(right) != args.expected_samples:
        raise ValueError("repeat runs do not have the expected sample count")
    if any(row.get("cache_hit") for row in [*left, *right]):
        raise ValueError("independent repeat comparison forbids cache hits")
    if any(
        not isinstance(row.get("request_hash"), str)
        or re.fullmatch(r"[0-9a-f]{64}", row["request_hash"]) is None
        for row in [*left, *right]
    ):
        raise ValueError(
            "independent repeat comparison requires a 64-hex request hash "
            "for every row"
        )
    left_ids = [str(row["sample_id"]) for row in left]
    right_ids = [str(row["sample_id"]) for row in right]
    if left_ids != right_ids or len(left_ids) != len(set(left_ids)):
        raise ValueError("repeat runs have different or duplicate ordered sample IDs")
    rows = []
    for first, second in zip(left, right):
        first_ranking = [item["candidate_id"] for item in first["ranking"]]
        second_ranking = [item["candidate_id"] for item in second["ranking"]]
        first_scores = [float(item["score"]) for item in first["ranking"]]
        second_scores = [float(item["score"]) for item in second["ranking"]]
        rows.append(
            {
                "sample_id": first["sample_id"],
                "request_hash_equal": first["request_hash"] == second["request_hash"],
                "selected_equal": (
                    first["selected_candidate_id"] == second["selected_candidate_id"]
                ),
                "ranking_ids_equal": first_ranking == second_ranking,
                "parsed_json_equal": (
                    first.get("parsed_model_response")
                    == second.get("parsed_model_response")
                ),
                "fallback_equal": (
                    first.get("fallback") == second.get("fallback")
                    and first.get("fallback_reason") == second.get("fallback_reason")
                    and first.get("abstain") == second.get("abstain")
                ),
                "confidence_abs_difference": abs(
                    float(first["confidence"]) - float(second["confidence"])
                ),
                "score_max_abs_difference": (
                    max(
                        abs(a - b)
                        for a, b in zip(first_scores, second_scores)
                    )
                    if len(first_scores) == len(second_scores)
                    else None
                ),
            }
        )
    count = len(rows)
    summary = {
        "schema_version": 1,
        **identity_payload(),
        "samples": count,
        "both_runs_cache_cold": True,
        "request_hash_agreement_rate": sum(
            row["request_hash_equal"] for row in rows
        )
        / count,
        "selected_candidate_agreement_rate": sum(
            row["selected_equal"] for row in rows
        )
        / count,
        "ranking_id_agreement_rate": sum(
            row["ranking_ids_equal"] for row in rows
        )
        / count,
        "exact_parsed_json_agreement_rate": sum(
            row["parsed_json_equal"] for row in rows
        )
        / count,
        "fallback_agreement_rate": sum(row["fallback_equal"] for row in rows)
        / count,
        "confidence_max_abs_difference": max(
            row["confidence_abs_difference"] for row in rows
        ),
        "score_max_abs_difference": max(
            (
                row["score_max_abs_difference"]
                for row in rows
                if row["score_max_abs_difference"] is not None
            ),
            default=None,
        ),
        "inputs": {
            "run_a": str(run_a),
            "run_a_sha256": _sha256(run_a),
            "run_b": str(run_b),
            "run_b_sha256": _sha256(run_b),
        },
        "per_sample": rows,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
