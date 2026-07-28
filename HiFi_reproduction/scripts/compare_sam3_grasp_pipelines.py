#!/usr/bin/env python3
"""Create a paired four-pipeline comparison using the frozen 2D predicate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_serialization import save_strict_json  # noqa: E402


BASELINE = REPO_ROOT / "outputs" / "gqcnn_original_ranking_evaluation" / "summary.json"
SAM3 = REPO_ROOT / "outputs" / "gqcnn_sam3_ranking_evaluation" / "summary.json"
MASK = REPO_ROOT / "outputs" / "sam3_mask_evaluation" / "summary.json"
MASK_PER_SAMPLE = REPO_ROOT / "outputs" / "sam3_mask_evaluation" / "per_sample_metrics.csv"
OUTPUT = REPO_ROOT / "outputs" / "sam3_grasp_comparison"


def _runtime_stats(candidate_root: Path, ranking_method: str) -> dict[str, float]:
    with (candidate_root / "summary.csv").open(encoding="utf-8", newline="") as stream:
        generation = list(csv.DictReader(stream))
    if not generation:
        raise ValueError(f"candidate summary is empty: {candidate_root}")
    raw = [float(row["raw_candidate_count"]) for row in generation if row["raw_candidate_count"]]
    post_nms = [float(row["post_nms_count"]) for row in generation if row["post_nms_count"]]
    generation_ms = [float(row["generation_time_ms"]) for row in generation if row["generation_time_ms"]]
    summary_name = "gqcnn_scoring_summary.csv" if ranking_method == "gqcnn" else "geometric_ranking_summary.csv"
    runtime_field = "official_quality_function_time_ms" if ranking_method == "gqcnn" else "ranking_time_ms"
    with (candidate_root / summary_name).open(encoding="utf-8", newline="") as stream:
        ranking = list(csv.DictReader(stream))
    ranking_ms = [float(row[runtime_field]) for row in ranking if row[runtime_field]]
    return {
        "mean_raw_candidate_count": float(np.mean(raw)),
        "mean_post_nms_candidate_count": float(np.mean(post_nms)),
        "mean_generation_runtime_ms": float(np.mean(generation_ms)),
        "mean_ranking_runtime_ms": float(np.mean(ranking_ms)),
        "mean_total_runtime_ms": float(np.mean(generation_ms) + np.mean(ranking_ms)),
    }


def _pipeline_row(
    name: str,
    method: dict,
    mask_miou: float,
    candidate_root: Path,
    ranking_method: str,
) -> dict:
    return {
        "pipeline": name,
        "mask_mean_iou": mask_miou,
        "evaluable_samples": method["evaluable_samples"],
        "top1_numerator": method["top1_consistency"]["numerator"],
        "top1_denominator": method["top1_consistency"]["denominator"],
        "top5_numerator": method["top5_recall"]["numerator"],
        "top5_denominator": method["top5_recall"]["denominator"],
        "mean_first_valid_rank": method["first_valid_rank"]["mean"],
        "median_first_valid_rank": method["first_valid_rank"]["median"],
        "population_std_first_valid_rank": method["first_valid_rank"]["population_standard_deviation"],
        "candidate_generation_failures": method["failures"]["candidate_generation_failures"],
        "top1_ranking_failures": method["failures"]["top1_failures"],
        "top5_ranking_failures": method["failures"]["top5_ranking_failures"],
        **_runtime_stats(candidate_root, ranking_method),
    }


def _paired(old_rows: list[dict], new_rows: list[dict], mask_rows: dict[str, dict], method: str) -> list[dict]:
    old = {row["sample_id"]: row for row in old_rows}
    new = {row["sample_id"]: row for row in new_rows}
    if set(old) != set(new) or set(old) != set(mask_rows):
        raise ValueError("paired comparison sample IDs differ")
    output: list[dict] = []
    for sample_id in sorted(old):
        before, after, mask = old[sample_id], new[sample_id], mask_rows[sample_id]
        old_top1 = bool(before["top1_consistent"])
        new_top1 = bool(after["top1_consistent"])
        old_rank, new_rank = before["first_valid_rank"], after["first_valid_rank"]
        output.append(
            {
                "sample_id": sample_id,
                "ranking_method": method,
                "mask_delta_iou": float(mask["delta_iou"]),
                "mask_classification": mask["classification"],
                "top1_before": old_top1,
                "top1_after": new_top1,
                "top1_transition": "improved" if new_top1 and not old_top1 else "degraded" if old_top1 and not new_top1 else "unchanged",
                "first_valid_rank_before": old_rank,
                "first_valid_rank_after": new_rank,
                "first_valid_rank_transition": (
                    "unavailable"
                    if old_rank is None or new_rank is None
                    else "improved"
                    if int(new_rank) < int(old_rank)
                    else "degraded"
                    if int(new_rank) > int(old_rank)
                    else "unchanged"
                ),
                "mask_iou_improved_grasp_unchanged": mask["classification"] == "improved" and old_top1 == new_top1,
                "mask_iou_improved_grasp_improved": mask["classification"] == "improved" and new_top1 and not old_top1,
                "mask_iou_degraded_grasp_improved": mask["classification"] == "degraded" and new_top1 and not old_top1,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE)
    parser.add_argument("--sam3-summary", type=Path, default=SAM3)
    parser.add_argument("--mask-summary", type=Path, default=MASK)
    parser.add_argument("--mask-per-sample", type=Path, default=MASK_PER_SAMPLE)
    parser.add_argument(
        "--baseline-candidates",
        type=Path,
        default=REPO_ROOT / "outputs" / "dexnet_candidates_ten_samples",
    )
    parser.add_argument(
        "--sam3-candidates",
        type=Path,
        default=REPO_ROOT / "outputs" / "dexnet_candidates_sam3_ten_samples",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite grasp comparison: {output}")
    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    sam3 = json.loads(args.sam3_summary.read_text(encoding="utf-8"))
    mask = json.loads(args.mask_summary.read_text(encoding="utf-8"))
    baseline_candidates = args.baseline_candidates.expanduser().resolve()
    sam3_candidates = args.sam3_candidates.expanduser().resolve()
    rows = [
        _pipeline_row("HiFi + Dex-Net + GQ-CNN", baseline["gqcnn_q_value_ranking"], mask["coarse_mean_iou"], baseline_candidates, "gqcnn"),
        _pipeline_row("HiFi + Dex-Net + geometric", baseline["geometric_re_ranking_reference"], mask["coarse_mean_iou"], baseline_candidates, "geometric"),
        _pipeline_row("HiFi + SAM 3 + Dex-Net + GQ-CNN", sam3["gqcnn_q_value_ranking"], mask["refined_mean_iou"], sam3_candidates, "gqcnn"),
        _pipeline_row("HiFi + SAM 3 + Dex-Net + geometric", sam3["geometric_re_ranking_reference"], mask["refined_mean_iou"], sam3_candidates, "geometric"),
    ]
    with args.mask_per_sample.open(encoding="utf-8", newline="") as stream:
        mask_rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    paired = _paired(baseline["per_sample_metrics"], sam3["per_sample_metrics"], mask_rows, "gqcnn")
    # The evaluator carries only GQ-CNN per-sample rows; geometric transitions are reconstructed
    # from the ranker summaries to preserve exactly the same frozen predicate.
    def geometric_rows(root: Path) -> list[dict]:
        with root.open(encoding="utf-8", newline="") as stream:
            return [
                {
                    "sample_id": row["sample_id"],
                    "top1_consistent": bool(int(row["top1_2d_rectangle_accuracy"])),
                    "first_valid_rank": int(row["first_2d_matching_rank"]) if row["first_2d_matching_rank"] else None,
                }
                for row in csv.DictReader(stream)
            ]
    paired.extend(
        _paired(
            geometric_rows(baseline_candidates / "geometric_ranking_summary.csv"),
            geometric_rows(sam3_candidates / "geometric_ranking_summary.csv"),
            mask_rows,
            "geometric",
        )
    )
    temporary = output.with_name(output.name + ".incomplete")
    temporary.mkdir(parents=True)
    try:
        with (temporary / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (temporary / "per_sample_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        deltas = np.asarray([float(row["mask_delta_iou"]) for row in paired if row["ranking_method"] == "gqcnn"])
        top1_changes = np.asarray(
            [int(row["top1_after"]) - int(row["top1_before"]) for row in paired if row["ranking_method"] == "gqcnn"]
        )
        correlation = None
        if len(deltas) > 1 and np.std(deltas) > 0 and np.std(top1_changes) > 0:
            correlation = float(np.corrcoef(deltas, top1_changes)[0, 1])
        correlation_row = {"metric": "pearson_mask_delta_iou_vs_gqcnn_top1_change", "value": correlation}
        with (temporary / "mask_vs_grasp_correlation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(correlation_row))
            writer.writeheader()
            writer.writerow(correlation_row)
        transitions = {
            "top1_improved": [row["sample_id"] for row in paired if row["top1_transition"] == "improved"],
            "top1_unchanged": [row["sample_id"] for row in paired if row["top1_transition"] == "unchanged"],
            "top1_degraded": [row["sample_id"] for row in paired if row["top1_transition"] == "degraded"],
            "first_valid_rank_improved": [row["sample_id"] for row in paired if row["first_valid_rank_transition"] == "improved"],
            "first_valid_rank_degraded": [row["sample_id"] for row in paired if row["first_valid_rank_transition"] == "degraded"],
        }
        save_strict_json(temporary / "success_transitions.json", transitions)
        save_strict_json(
            temporary / "failure_transitions.json",
            {"top1_degraded": transitions["top1_degraded"], "first_valid_rank_degraded": transitions["first_valid_rank_degraded"]},
        )
        (temporary / "qualitative_cases").mkdir()
        save_strict_json(
            temporary / "summary.json",
            {
                "schema_version": 1,
                "metric": "2D consistency with OCID-VLG planar grasp annotations",
                "is_physical_grasp_success": False,
                "pipelines": rows,
                "paired_transition_counts": {key: len(value) for key, value in transitions.items()},
                "mask_top1_pearson": correlation,
            },
        )
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": "COMPARED", "pipelines": 4, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
