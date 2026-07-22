#!/usr/bin/env python3
"""Evaluate stored GQ-CNN q-value ranking without sampling or inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.geometric_ranker import save_strict_json, sha256_file  # noqa: E402
from src.grasping.gqcnn_ranking_evaluation import (  # noqa: E402
    BASELINE_NAME,
    METRIC_NAME,
    PER_SAMPLE_FIELDS,
    SUMMARY_FIELDS,
    EvaluationDataError,
    aggregate_metrics,
    classify_failures,
    evaluate_sample,
    load_evaluation_config,
    save_failure_visualizations,
    save_invalid_diagnostic,
    summary_csv_row,
    write_csv,
)


PROTECTED_NAMES = (
    "candidates.npz",
    "candidates.json",
    "gqcnn_scored_candidates.npz",
    "gqcnn_scored_candidates.json",
    "gqcnn_scored_candidates.csv",
    "geometrically_ranked_candidates.npz",
    "geometrically_ranked_candidates.json",
    "geometrically_ranked_candidates.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        required=True,
        help="OCID-VLG test_expressions.json or a root containing refer/unique/test_expressions.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--expect-geometric-top1", type=int)
    parser.add_argument("--expect-geometric-top5", type=int)
    return parser.parse_args()


def resolve_annotation_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    candidates = [
        path / "test_expressions.json",
        path / "refer" / "unique" / "test_expressions.json",
    ]
    matches = [item for item in candidates if item.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one OCID-VLG test_expressions.json below {path}, found {matches}"
        )
    return matches[0]


def sample_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"candidate root does not exist: {root}")
    samples = sorted(path for path in root.iterdir() if path.is_dir())
    samples = [path for path in samples if (path / "metadata.json").is_file()]
    if not samples:
        raise FileNotFoundError(f"no sample directories below {root}")
    return samples


def protected_manifest(samples: list[Path], root: Path) -> dict[str, str]:
    manifest = {}
    for sample in samples:
        for name in PROTECTED_NAMES:
            path = sample / name
            if path.is_file():
                manifest[str(path.relative_to(root))] = sha256_file(path)
    return manifest


def assert_shared_config(config: dict[str, Any], path: Path) -> None:
    """Fail if the new shared file drifts from the existing geometric config."""

    geometric_path = REPO_ROOT / "configs" / "dexnet_geometric_ranker.yaml"
    geometric = YAML(typ="safe").load(geometric_path.read_text(encoding="utf-8"))["evaluation"]
    keys = (
        "predicted_rectangle_height_px",
        "ground_truth_rectangle_height_px",
        "ground_truth_width_clip_px",
        "angle_threshold_deg",
        "iou_threshold",
    )
    for key in keys:
        if float(config[key]) != float(geometric[key]):
            raise ValueError(
                f"{path} {key}={config[key]} differs from existing geometric value {geometric[key]}"
            )
    if str(config["metric_name"]) != str(geometric["label"]):
        raise ValueError("shared metric label differs from geometric evaluation label")


def invalid_row(sample_dir: Path, error: EvaluationDataError) -> dict[str, Any]:
    sample_id = sample_dir.name
    query = ""
    metadata_path = sample_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_id = str(metadata.get("sample_id", sample_id))
        query = str(metadata.get("query", ""))
    except Exception:
        pass
    row = {field: None for field in PER_SAMPLE_FIELDS}
    row.update(
        {
            "sample_id": sample_id,
            "query": query,
            "candidate_count": 0,
            "finite_q_count": 0,
            "top1_consistent": False,
            "top5_consistent": False,
            "candidate_generation_success": False,
            "failure_type": error.category,
            "failure_reason": str(error),
            "failure_categories": [error.category],
            "data_valid": False,
        }
    )
    return row


def geometric_row(outcome: dict[str, Any]) -> dict[str, Any]:
    source = outcome["geometric_reference_metrics"]
    categories = classify_failures(
        candidate_count=int(source["candidate_count"]),
        top1_consistent=bool(source["top1_consistent"]),
        top5_consistent=bool(source["top5_consistent"]),
        first_valid_rank=source["first_valid_rank"],
    )
    return {
        **source,
        "data_valid": True,
        "failure_categories": categories,
        "failure_type": "|".join(categories),
    }


def main() -> int:
    args = parse_args()
    candidate_root = args.candidate_root.expanduser().resolve()
    annotation_file = resolve_annotation_file(args.annotation_root)
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.evaluation_config.expanduser().resolve()
    config = load_evaluation_config(config_path)
    assert_shared_config(config, config_path)
    samples = sample_dirs(candidate_root)
    before = protected_manifest(samples, candidate_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes = []
    per_sample = []
    invalid = []
    for sample_dir in samples:
        try:
            outcome = evaluate_sample(sample_dir, annotation_file, config)
            outcome["evaluation_config"] = config
            outcome["metrics"]["data_valid"] = True
            outcomes.append(outcome)
            per_sample.append(outcome["metrics"])
            print(
                json.dumps(
                    {
                        "sample_id": outcome["metrics"]["sample_id"],
                        "top1_consistent": outcome["metrics"]["top1_consistent"],
                        "top5_consistent": outcome["metrics"]["top5_consistent"],
                        "first_valid_rank": outcome["metrics"]["first_valid_rank"],
                        "failure_type": outcome["metrics"]["failure_type"],
                    },
                    sort_keys=True,
                )
            )
        except EvaluationDataError as error:
            row = invalid_row(sample_dir, error)
            invalid.append(row)
            per_sample.append(row)
            save_invalid_diagnostic(sample_dir, output_dir, row)
            print(json.dumps(row, sort_keys=True), file=sys.stderr)

    after = protected_manifest(samples, candidate_root)
    if before != after:
        raise RuntimeError("protected candidate, score, or ranking files changed during evaluation")

    gq_aggregate = aggregate_metrics(per_sample, method=BASELINE_NAME)
    geometric_rows = [geometric_row(outcome) for outcome in outcomes]
    geometric_aggregate = aggregate_metrics(
        geometric_rows,
        method="Transparent target-aware geometric re-ranking",
    )
    if args.expect_geometric_top1 is not None:
        actual = geometric_aggregate["top1_consistency"]["numerator"]
        if actual != args.expect_geometric_top1:
            raise RuntimeError(
                f"geometric Top-1 regression: expected {args.expect_geometric_top1}, got {actual}"
            )
    if args.expect_geometric_top5 is not None:
        actual = geometric_aggregate["top5_recall"]["numerator"]
        if actual != args.expect_geometric_top5:
            raise RuntimeError(
                f"geometric Top-5 regression: expected {args.expect_geometric_top5}, got {actual}"
            )

    for outcome in outcomes:
        save_failure_visualizations(outcome, output_dir)

    ranked_payload = {
        "metadata": {
            "schema_version": 1,
            "baseline": BASELINE_NAME,
            "ranking": "raw stored gqcnn_q_value descending; exact ties by candidate_id ascending",
            "q_value_normalized_before_sort": False,
            "rank_indexing": "one_based",
            "candidate_source": "exact frozen post-NMS candidates; no resampling or pose alteration",
            "metric": METRIC_NAME,
            "is_physical_grasp_success": False,
            "candidate_root": str(candidate_root),
            "annotation_file": str(annotation_file),
            "evaluation_config": str(config_path),
        },
        "samples": [
            {
                "sample_id": outcome["metrics"]["sample_id"],
                "query": outcome["metrics"]["query"],
                "source_files": outcome["source_hashes"],
                "storage_order_note": outcome["storage_order"],
                "stored_rank_matches_exact_q_reconstruction": outcome[
                    "stored_rank_matches_exact_q_reconstruction"
                ],
                "stored_rank_mismatch_count": outcome["stored_rank_mismatch_count"],
                "stored_rank_mismatches": outcome["stored_rank_mismatches"],
                "candidates": outcome["ranked"],
            }
            for outcome in outcomes
        ],
        "invalid_samples": invalid,
    }
    save_strict_json(output_dir / "gqcnn_ranked_candidates.json", ranked_payload)
    write_csv(output_dir / "per_sample_metrics.csv", per_sample, PER_SAMPLE_FIELDS)
    write_csv(
        output_dir / "summary.csv",
        [summary_csv_row(gq_aggregate), summary_csv_row(geometric_aggregate)],
        SUMMARY_FIELDS,
    )

    failure_cases = [row for row in per_sample if row.get("failure_type") != "none"]
    success_cases = [row for row in per_sample if row.get("top1_consistent") is True]
    save_strict_json(
        output_dir / "failure_cases.json",
        {
            "baseline": BASELINE_NAME,
            "failure_case_count": len(failure_cases),
            "cases": failure_cases,
        },
    )
    save_strict_json(
        output_dir / "success_cases.json",
        {
            "baseline": BASELINE_NAME,
            "top1_success_case_count": len(success_cases),
            "cases": success_cases,
        },
    )

    summary = {
        "schema_version": 1,
        "title": "Preliminary ten-sample comparison",
        "baseline": BASELINE_NAME,
        "baseline_scope_warning": (
            "Fixed post-NMS candidates scored by official GQCNN-2.1; this is not the complete "
            "original Dex-Net CEM policy."
        ),
        "metric": METRIC_NAME,
        "is_physical_grasp_success": False,
        "ranking": {
            "primary_key": "stored raw gqcnn_q_value descending",
            "tie_breaker": "candidate_id ascending",
            "q_value_normalized": False,
            "rank_indexing": "one_based",
        },
        "evaluation_config": config,
        "inputs": {
            "candidate_root": str(candidate_root),
            "annotation_file": str(annotation_file),
            "annotation_file_sha256": sha256_file(annotation_file),
            "evaluation_config_file": str(config_path),
            "evaluation_config_sha256": sha256_file(config_path),
            "evaluated_sample_ids": [row["sample_id"] for row in per_sample],
            "protected_source_file_sha256": before,
            "protected_sources_unchanged": True,
        },
        "audit": {
            "sample_count": len(samples),
            "total_candidate_count": sum(int(row.get("candidate_count") or 0) for row in per_sample),
            "total_finite_q_count": sum(int(row.get("finite_q_count") or 0) for row in per_sample),
            "all_stored_ranks_match_exact_q_reconstruction": all(
                outcome["stored_rank_matches_exact_q_reconstruction"] for outcome in outcomes
            ),
            "stored_rank_mismatch_count": sum(
                outcome["stored_rank_mismatch_count"] for outcome in outcomes
            ),
            "stored_rank_mismatch_sample_ids": [
                outcome["metrics"]["sample_id"]
                for outcome in outcomes
                if outcome["stored_rank_mismatch_count"]
            ],
            "stored_rank_note": (
                "The evaluator ignores stored rank labels and deterministically reconstructs ranks "
                "from raw finite q-values; mismatches are retained as an audit finding."
            ),
        },
        "gqcnn_q_value_ranking": gq_aggregate,
        "geometric_re_ranking_reference": geometric_aggregate,
        "per_sample_metrics": per_sample,
    }
    save_strict_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "DONE" if not invalid else "PARTIAL",
                "samples": len(samples),
                "invalid_samples": len(invalid),
                "top1": gq_aggregate["top1_consistency"],
                "top5": gq_aggregate["top5_recall"],
                "mean_valid_grasp_rank": gq_aggregate["first_valid_rank"]["mean"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0 if not invalid else 2


if __name__ == "__main__":
    raise SystemExit(main())
