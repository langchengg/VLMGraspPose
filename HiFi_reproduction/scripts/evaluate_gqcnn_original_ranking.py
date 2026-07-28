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
    aggregate_metric_conventions,
    classify_failures,
    evaluate_sample,
    load_evaluation_config,
    load_ocid_vlg_annotation_index,
    load_skipped_valid_empty_metrics,
    save_failure_visualizations,
    save_invalid_diagnostic,
    summary_csv_row,
    write_csv,
)


PROTECTED_NAMES = (
    "candidates.npz",
    "candidates.json",
    "metadata.json",
    "_SUCCESS.json",
    "gqcnn_scored_candidates.npz",
    "gqcnn_scored_candidates.json",
    "gqcnn_scored_candidates.csv",
    "geometrically_ranked_candidates.npz",
    "geometrically_ranked_candidates.json",
    "geometrically_ranked_candidates.csv",
    "scoring_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root",
        "--source-root",
        dest="candidate_root",
        type=Path,
        required=True,
        help="Immutable frozen-candidate root (--source-root is an alias)",
    )
    parser.add_argument(
        "--scored-root",
        type=Path,
        help="Independent per-sample scored root; defaults to candidate root for legacy outputs",
    )
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
    geometric = parser.add_mutually_exclusive_group()
    geometric.add_argument(
        "--geometric-reference",
        dest="geometric_reference",
        action="store_true",
        help="Require and evaluate the source geometric ranking",
    )
    geometric.add_argument(
        "--no-geometric-reference",
        dest="geometric_reference",
        action="store_false",
        help="Do not load a geometric reference ranking",
    )
    parser.set_defaults(geometric_reference=None)
    parser.add_argument(
        "--visualizations",
        choices=("auto", "none", "failures"),
        default="auto",
        help="auto preserves legacy ten-sample behavior and disables full-root rendering",
    )
    parser.add_argument("--expect-total-samples", type=int)
    parser.add_argument("--expect-nonempty-samples", type=int)
    parser.add_argument("--expect-valid-empty-samples", type=int)
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


def protected_manifest(
    samples: list[Path], root: Path, *, key_prefix: str = ""
) -> dict[str, str]:
    manifest = {}
    for sample in samples:
        for name in PROTECTED_NAMES:
            path = sample / name
            if path.is_file():
                key = str(path.relative_to(root))
                manifest[f"{key_prefix}{key}"] = sha256_file(path)
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


def invalid_row(source_sample_dir: Path, error: EvaluationDataError) -> dict[str, Any]:
    sample_id = source_sample_dir.name
    query = ""
    candidate_count = 0
    valid_empty = False
    metadata_path = source_sample_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_id = str(metadata.get("sample_id", sample_id))
        query = str(metadata.get("query", ""))
        counts = metadata.get("counts", {})
        candidate_count = int(counts.get("post_nms", 0)) if isinstance(counts, dict) else 0
        valid_empty = candidate_count == 0
    except Exception:
        pass
    row = {field: None for field in PER_SAMPLE_FIELDS}
    row.update(
        {
            "sample_id": sample_id,
            "query": query,
            "scoring_status": "invalid_evaluation_input",
            "valid_empty": valid_empty,
            "candidate_count": candidate_count,
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
    scored_root = (
        candidate_root
        if args.scored_root is None
        else args.scored_root.expanduser().resolve()
    )
    annotation_file = resolve_annotation_file(args.annotation_root)
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.evaluation_config.expanduser().resolve()
    config = load_evaluation_config(config_path)
    assert_shared_config(config, config_path)
    samples = sample_dirs(candidate_root)
    if output_dir == candidate_root or candidate_root in output_dir.parents:
        raise ValueError("output-dir must not be inside the immutable candidate root")
    if output_dir == scored_root or scored_root in output_dir.parents:
        raise ValueError("output-dir must not be inside the immutable scored root")
    if not scored_root.is_dir():
        raise FileNotFoundError(f"scored root does not exist: {scored_root}")
    colocated = scored_root == candidate_root
    scored_samples = [scored_root / sample.name for sample in samples]
    before = protected_manifest(samples, candidate_root, key_prefix="source/")
    if not colocated:
        before.update(
            protected_manifest(scored_samples, scored_root, key_prefix="scored/")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_index = load_ocid_vlg_annotation_index(annotation_file)
    include_geometric_reference = args.geometric_reference
    if include_geometric_reference is None and not colocated:
        include_geometric_reference = False
    render_failures = args.visualizations == "failures" or (
        args.visualizations == "auto" and colocated
    )

    per_sample = []
    invalid = []
    geometric_rows = []
    legacy_ranked_samples = []
    rank_mismatch_count = 0
    rank_mismatch_sample_ids = []
    jsonl_path = output_dir / "per_sample_metrics.jsonl"
    jsonl_temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    jsonl_stream = jsonl_temporary.open("w", encoding="utf-8", newline="\n")
    for sample_dir in samples:
        scored_sample_dir = scored_root / sample_dir.name
        try:
            source_metadata = json.loads(
                (sample_dir / "metadata.json").read_text(encoding="utf-8")
            )
            counts = source_metadata.get("counts", {})
            source_candidate_count = int(counts.get("post_nms", -1))
            if source_candidate_count == 0:
                row = load_skipped_valid_empty_metrics(
                    sample_dir, scored_sample_dir, config
                )
                per_sample.append(row)
                jsonl_stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "sample_id": row["sample_id"],
                            "scoring_status": row["scoring_status"],
                            "top1_consistent": False,
                            "top5_consistent": False,
                        },
                        sort_keys=True,
                    )
                )
                continue
            if source_candidate_count < 0:
                raise EvaluationDataError(
                    "mapping_or_geometry_error", "source post_nms count is missing"
                )
            outcome = evaluate_sample(
                sample_dir,
                annotation_file,
                config,
                scored_sample_dir=scored_sample_dir,
                annotation_index=annotation_index,
                include_geometric_reference=include_geometric_reference,
            )
            outcome["evaluation_config"] = config
            outcome["metrics"]["data_valid"] = True
            per_sample.append(outcome["metrics"])
            if outcome["geometric_reference_metrics"] is not None:
                geometric_rows.append(geometric_row(outcome))
            rank_mismatch_count += int(outcome["stored_rank_mismatch_count"])
            if outcome["stored_rank_mismatch_count"]:
                rank_mismatch_sample_ids.append(outcome["metrics"]["sample_id"])
            if render_failures:
                save_failure_visualizations(outcome, output_dir)
            if colocated:
                legacy_ranked_samples.append(
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
                )
            compact = {
                **outcome["metrics"],
                "source_files": outcome["source_hashes"],
                "stored_rank_matches_exact_q_reconstruction": outcome[
                    "stored_rank_matches_exact_q_reconstruction"
                ],
                "stored_rank_mismatch_count": outcome["stored_rank_mismatch_count"],
            }
            jsonl_stream.write(
                json.dumps(compact, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )
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
            jsonl_stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )
            print(json.dumps(row, sort_keys=True), file=sys.stderr)
    jsonl_stream.close()
    jsonl_temporary.replace(jsonl_path)

    after = protected_manifest(samples, candidate_root, key_prefix="source/")
    if not colocated:
        after.update(
            protected_manifest(scored_samples, scored_root, key_prefix="scored/")
        )
    if before != after:
        raise RuntimeError("protected candidate, score, or ranking files changed during evaluation")

    conventions = aggregate_metric_conventions(per_sample, method=BASELINE_NAME)
    gq_aggregate = conventions["conditional_nonempty"]
    geometric_aggregate = None
    if geometric_rows:
        geometric_aggregate = aggregate_metrics(
            geometric_rows,
            method="Transparent target-aware geometric re-ranking",
        )
    if args.expect_geometric_top1 is not None:
        if geometric_aggregate is None:
            raise RuntimeError("geometric Top-1 expectation requires a geometric reference")
        actual = geometric_aggregate["top1_consistency"]["numerator"]
        if actual != args.expect_geometric_top1:
            raise RuntimeError(
                f"geometric Top-1 regression: expected {args.expect_geometric_top1}, got {actual}"
            )
    if args.expect_geometric_top5 is not None:
        if geometric_aggregate is None:
            raise RuntimeError("geometric Top-5 expectation requires a geometric reference")
        actual = geometric_aggregate["top5_recall"]["numerator"]
        if actual != args.expect_geometric_top5:
            raise RuntimeError(
                f"geometric Top-5 regression: expected {args.expect_geometric_top5}, got {actual}"
            )

    observed_total = len(per_sample)
    observed_nonempty = sum(not row.get("valid_empty", False) for row in per_sample)
    observed_valid_empty = conventions["denominators"]["skipped_valid_empty"]
    for label, expected, actual in (
        ("total samples", args.expect_total_samples, observed_total),
        ("non-empty samples", args.expect_nonempty_samples, observed_nonempty),
        ("skipped valid empty samples", args.expect_valid_empty_samples, observed_valid_empty),
    ):
        if expected is not None and actual != expected:
            raise RuntimeError(f"expected {expected} {label}, got {actual}")

    if colocated:
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
                "scored_root": str(scored_root),
                "annotation_file": str(annotation_file),
                "evaluation_config": str(config_path),
            },
            "samples": legacy_ranked_samples,
            "invalid_samples": invalid,
        }
        save_strict_json(output_dir / "gqcnn_ranked_candidates.json", ranked_payload)
    write_csv(output_dir / "per_sample_metrics.csv", per_sample, PER_SAMPLE_FIELDS)
    summary_rows = (
        [aggregate_metrics(per_sample, method=BASELINE_NAME)]
        if colocated and not any(row.get("valid_empty", False) for row in per_sample)
        else [
            conventions["conditional_nonempty"],
            conventions["end_to_end_all_samples"],
        ]
    )
    if geometric_aggregate is not None:
        summary_rows.append(geometric_aggregate)
    write_csv(
        output_dir / "summary.csv",
        [summary_csv_row(item) for item in summary_rows],
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
        "title": "Preliminary ten-sample comparison" if colocated else "Full-dataset ranking evaluation",
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
            "scored_root": str(scored_root),
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
            "valid_empty_sample_count": observed_valid_empty,
            "all_stored_ranks_match_exact_q_reconstruction": rank_mismatch_count == 0,
            "stored_rank_mismatch_count": rank_mismatch_count,
            "stored_rank_mismatch_sample_ids": rank_mismatch_sample_ids,
            "stored_rank_note": (
                "The evaluator ignores stored rank labels and deterministically reconstructs ranks "
                "from raw finite q-values; mismatches are retained as an audit finding."
            ),
        },
        "gqcnn_q_value_ranking": gq_aggregate,
        "metric_conventions": conventions,
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
