#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from failure_utils import (
    ensure_dir,
    format_metric,
    read_jsonl,
    save_case_figure,
    save_contact_sheet,
    write_csv,
)


DEFAULT_INPUT = "failure_analysis/predictions/test_predictions.jsonl"
RESULTS_DIR = Path("failure_analysis/results")
FIGURES_DIR = Path("failure_analysis/figures")

LANGUAGE_KEYWORDS = {
    "left", "right", "front", "behind", "back", "rear", "middle", "small",
    "large", "big", "tiny", "red", "blue", "green", "yellow", "white",
    "black", "orange", "purple", "near", "next", "between", "beside",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse exported CROG failure cases.")
    parser.add_argument("--predictions", default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--max-figures-per-group", type=int, default=25)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def has_language_ambiguity(row):
    text = str(row.get("language_instruction", "")).lower()
    return any(keyword in text.split() or keyword in text for keyword in LANGUAGE_KEYWORDS)


def is_cluttered(row):
    count = row.get("scene_instance_count")
    bbox_area = row.get("bbox_area")
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 0
    try:
        bbox_area = int(bbox_area)
    except (TypeError, ValueError):
        bbox_area = 0
    return count >= 6 or (bbox_area > 0 and bbox_area < 2500)


def classify(row):
    mask_iou = safe_float(row.get("mask_iou"))
    center_error = safe_float(row.get("grasp_center_error"))
    angle_error = safe_float(row.get("grasp_angle_error"))
    width_error = safe_float(row.get("grasp_width_error"))
    j1 = bool(row.get("j1_success"))
    jany = bool(row.get("jany_success"))
    pr50 = bool(row.get("pr50_success"))
    top1 = row.get("predicted_grasps_top1") or []
    secondary = []

    ambiguous = has_language_ambiguity(row)
    cluttered = is_cluttered(row)
    if ambiguous:
        secondary.append("language_ambiguity_failure")
    if cluttered:
        secondary.append("clutter_occlusion_failure")

    if pr50 and j1:
        return "success", "", "mask IoU is above 0.5 and top-1 grasp matches a labelled grasp"

    if not top1:
        primary = "localization_failure" if mask_iou >= 0.5 else "grounding_failure"
        return primary, join_secondary(secondary), "no predicted grasp peak was detected"

    if (not j1) and jany:
        return "top1_ranking_failure", join_secondary(secondary), "top-1 grasp failed but a top-5 candidate passed Jaccard"

    if mask_iou < 0.5:
        return "grounding_failure", join_secondary(secondary), "predicted target mask has IoU below 0.5"

    if not bool(row.get("predicted_center_in_gt_mask")) or (not_nan(center_error) and center_error > 35.0):
        return "localization_failure", join_secondary(secondary), "predicted grasp centre is far from labelled target grasps or outside target mask"

    if not_nan(angle_error) and angle_error > 30.0:
        return "orientation_failure", join_secondary(secondary), "grasp centre is plausible but angle error exceeds 30 degrees"

    if not_nan(width_error) and width_error > 25.0:
        return "width_failure", join_secondary(secondary), "grasp centre and angle are plausible but width differs substantially"

    if mask_iou >= 0.7 and not j1:
        return "dataset_or_metric_edge_case", join_secondary(secondary), "mask is strong but grasp metric still rejects the prediction"

    if cluttered:
        return "clutter_occlusion_failure", join_secondary([item for item in secondary if item != "clutter_occlusion_failure"]), "scene has many labelled instances or a small target bbox"

    if ambiguous:
        return "language_ambiguity_failure", join_secondary([item for item in secondary if item != "language_ambiguity_failure"]), "instruction contains spatial, colour, or attribute cues and the prediction failed"

    return "dataset_or_metric_edge_case", join_secondary(secondary), "failure does not cleanly separate under available diagnostics"


def join_secondary(items):
    clean = []
    for item in items:
        if item and item not in clean:
            clean.append(item)
    return ";".join(clean)


def safe_float(value):
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def not_nan(value):
    return value is not None and not math.isnan(float(value))


def diagnostics_rows(rows):
    diagnostics = []
    for row in rows:
        primary, secondary, reason = classify(row)
        result = dict(row)
        result["failure_category_primary"] = primary
        result["failure_category_secondary"] = secondary
        result["short_reason"] = reason
        result["is_interesting_case"] = bool(
            primary in {
                "top1_ranking_failure",
                "language_ambiguity_failure",
                "clutter_occlusion_failure",
                "dataset_or_metric_edge_case",
            }
            or (not bool(row.get("j1_success")) and bool(row.get("jany_success")))
        )
        diagnostics.append(result)
    return diagnostics


def select_success_cases(rows):
    candidates = [row for row in rows if row["failure_category_primary"] == "success"]
    candidates.sort(key=lambda row: (safe_float(row.get("mask_iou")), safe_float(row.get("predicted_confidence"))), reverse=True)
    selected = diverse_by_target(candidates, 12)
    return selected[:12]


def select_failure_cases(rows):
    failures = [row for row in rows if row["failure_category_primary"] != "success"]
    by_category = defaultdict(list)
    for row in failures:
        by_category[row["failure_category_primary"]].append(row)
    selected = []
    for category in [
        "grounding_failure",
        "localization_failure",
        "orientation_failure",
        "width_failure",
        "language_ambiguity_failure",
        "clutter_occlusion_failure",
        "top1_ranking_failure",
        "dataset_or_metric_edge_case",
    ]:
        category_rows = by_category.get(category, [])
        category_rows.sort(key=lambda row: (row.get("is_interesting_case"), -safe_float(row.get("mask_iou"))), reverse=True)
        selected.extend(category_rows[:3])
    seen = {row["sample_id"] for row in selected}
    remainder = [row for row in failures if row["sample_id"] not in seen]
    remainder.sort(key=lambda row: (row.get("is_interesting_case"), -safe_float(row.get("mask_iou"))), reverse=True)
    selected.extend(remainder[: max(0, 25 - len(selected))])
    return selected[:25]


def select_ranking_cases(rows):
    candidates = [
        row for row in rows
        if row["failure_category_primary"] == "top1_ranking_failure"
        or (not bool(row.get("j1_success")) and bool(row.get("jany_success")))
    ]
    candidates.sort(key=lambda row: (safe_float(row.get("mask_iou")), safe_float(row.get("predicted_confidence"))), reverse=True)
    return candidates[:10]


def diverse_by_target(candidates, limit):
    selected = []
    used_targets = set()
    for row in candidates:
        target = row.get("target_name")
        if target in used_targets:
            continue
        selected.append(row)
        used_targets.add(target)
        if len(selected) >= limit:
            return selected
    seen = {row["sample_id"] for row in selected}
    for row in candidates:
        if row["sample_id"] in seen:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def selected_fields():
    return [
        "sample_id",
        "image_path",
        "language_instruction",
        "target_name",
        "split",
        "mask_iou",
        "pr50_success",
        "j1_success",
        "jany_success",
        "grasp_center_error",
        "grasp_angle_error",
        "grasp_width_error",
        "predicted_confidence",
        "failure_category_primary",
        "failure_category_secondary",
        "short_reason",
        "is_interesting_case",
    ]


def write_missing_diagnostics(rows, path):
    missing = Counter()
    fields = [
        "grasp_center_error",
        "grasp_angle_error",
        "grasp_width_error",
        "predicted_confidence",
    ]
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                missing[field] += 1
    lines = [
        "# Missing Diagnostics",
        "",
        "The analysis does not invent unavailable metrics. Missing values are left blank in CSV exports.",
        "",
        "| field | missing rows | reason |",
        "|---|---:|---|",
    ]
    for field in fields:
        reason = "usually no top-1 grasp peak was detected" if field != "predicted_confidence" else "top-1 confidence is undefined when no grasp peak is detected"
        lines.append(f"| {field} | {missing[field]} | {reason} |")
    lines.extend([
        "",
        "Additional limitations:",
        "- No explicit CROG candidate-ranking scores are exported by the original evaluation loop; top-k here is reconstructed from grasp-quality-map peaks.",
        "- Language ambiguity and clutter/occlusion labels are heuristic diagnostics from the prompt text, object count, and target bbox area.",
        "- Width and angle errors are measured against the nearest labelled grasp centre, not against unlabelled physically valid grasps.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(rows, success_cases, failure_cases, ranking_cases, results_dir):
    total = len(rows)
    failures = [row for row in rows if row["failure_category_primary"] != "success"]
    successes = [row for row in rows if row["failure_category_primary"] == "success"]
    category_counts = Counter(row["failure_category_primary"] for row in failures)
    summary_rows = []
    for category, count in sorted(category_counts.items()):
        summary_rows.append({
            "failure_category": category,
            "count": count,
            "percentage_of_failures": count / len(failures) * 100.0 if failures else 0.0,
            "percentage_of_all_samples": count / total * 100.0 if total else 0.0,
        })
    write_csv(results_dir / "failure_summary.csv", summary_rows, [
        "failure_category", "count", "percentage_of_failures", "percentage_of_all_samples",
    ])

    avg_success_iou = mean(row.get("mask_iou") for row in successes)
    avg_failure_iou = mean(row.get("mask_iou") for row in failures)
    j1_count = sum(1 for row in rows if row.get("j1_success"))
    jany_count = sum(1 for row in rows if row.get("jany_success"))
    j_gap_count = sum(1 for row in rows if (not row.get("j1_success")) and row.get("jany_success"))

    lines = [
        "# Failure Summary",
        "",
        f"- Samples analysed: {total}",
        f"- Success samples: {len(successes)} ({pct(len(successes), total)})",
        f"- Failure samples: {len(failures)} ({pct(len(failures), total)})",
        f"- Selected success examples: {len(success_cases)}",
        f"- Selected failure examples: {len(failure_cases)}",
        f"- Selected ranking-motivation examples: {len(ranking_cases)}",
        f"- Average IoU for successes: {format_metric(avg_success_iou)}",
        f"- Average IoU for failures: {format_metric(avg_failure_iou)}",
        f"- J@1 success rate: {pct(j1_count, total)}",
        f"- J@Any success rate: {pct(jany_count, total)}",
        f"- J@1 fails but J@Any succeeds: {j_gap_count} ({pct(j_gap_count, total)})",
        "",
        "## Failure Category Counts",
        "",
        "| category | count | % of failures | % of all samples |",
        "|---|---:|---:|---:|",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['failure_category']} | {item['count']} | "
            f"{item['percentage_of_failures']:.2f}% | {item['percentage_of_all_samples']:.2f}% |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The J@1/J@Any gap is the clearest quantitative signal for ranking-related errors: those cases already contain at least one acceptable top-k grasp but the selected top-1 grasp is not the accepted one.",
    ])
    (results_dir / "failure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values):
    clean = [safe_float(value) for value in values]
    clean = [value for value in clean if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def pct(count, total):
    if not total:
        return "0.00%"
    return f"{count / total * 100.0:.2f}%"


def report_figure_path(row, figures_dir, group, report_path):
    """Return a report-relative path for a generated qualitative case figure."""
    category = row.get("failure_category_primary", "case")
    filename = f"sample_{row['sample_id']}_{category}.png"
    figure_path = Path(figures_dir) / group / filename
    return Path(os.path.relpath(figure_path, start=Path(report_path).parent)).as_posix()


def append_visual_case(lines, row, figure_path):
    """Append one self-contained qualitative example to a Markdown report."""
    category = row.get("failure_category_primary", "case")
    lines.extend([
        f"#### Sample {row['sample_id']} — {category}",
        "",
        f"- Language: `{row.get('language_instruction', '')}`",
        f"- Target: `{row.get('target_name', '')}`",
        f"- Mask IoU: {format_metric(row.get('mask_iou'))}",
        f"- J@1: {bool(row.get('j1_success'))}; J@Any: {bool(row.get('jany_success'))}",
        f"- Diagnostic: {row.get('short_reason', '')}",
        "",
        f"![Sample {row['sample_id']} {category}]({figure_path})",
        "",
    ])


def select_one_per_category(rows, categories):
    """Select the first available case for each requested diagnostic category."""
    selected = []
    for category in categories:
        match = next(
            (row for row in rows if row.get("failure_category_primary") == category),
            None,
        )
        if match is not None:
            selected.append(match)
    return selected


def save_plots(rows, figures_dir):
    ensure_dir(figures_dir)
    failures = [row for row in rows if row["failure_category_primary"] != "success"]
    counts = Counter(row["failure_category_primary"] for row in failures)
    if counts:
        labels, values = zip(*sorted(counts.items(), key=lambda item: item[1], reverse=True))
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color="#4078c0")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("count")
        plt.title("Failure Category Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "failure_category_bar.png", dpi=150)
        plt.close()

    ious = [safe_float(row.get("mask_iou")) for row in rows]
    ious = [value for value in ious if not math.isnan(value)]
    if ious:
        plt.figure(figsize=(8, 5))
        plt.hist(ious, bins=20, color="#4c9f70", edgecolor="white")
        plt.xlabel("mask IoU")
        plt.ylabel("samples")
        plt.title("Mask IoU Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "iou_distribution.png", dpi=150)
        plt.close()

    total = len(rows)
    j1 = sum(1 for row in rows if row.get("j1_success"))
    jany = sum(1 for row in rows if row.get("jany_success"))
    gap = sum(1 for row in rows if (not row.get("j1_success")) and row.get("jany_success"))
    plt.figure(figsize=(7, 5))
    plt.bar(["J@1", "J@Any", "Gap"], [j1, jany, gap], color=["#3465a4", "#73a839", "#c17d11"])
    plt.ylim(0, max(total, 1))
    plt.ylabel("samples")
    plt.title("J@1 vs J@Any")
    for idx, value in enumerate([j1, jany, gap]):
        plt.text(idx, value, f"{pct(value, total)}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(figures_dir / "j1_vs_jany_summary.png", dpi=150)
    plt.close()


def generate_case_figures(groups, figures_dir, max_per_group):
    generated = {}
    for group_name, rows in groups.items():
        paths = []
        group_dir = figures_dir / group_name
        ensure_dir(group_dir)
        for stale_path in group_dir.glob("*.png"):
            stale_path.unlink()
        for row in rows[:max_per_group]:
            category = row.get("failure_category_primary", "case")
            filename = f"sample_{row['sample_id']}_{category}.png"
            output_path = group_dir / filename
            save_case_figure(row, output_path)
            row["figure_path"] = str(output_path)
            paths.append(output_path)
        contact_name = {
            "success": "contact_sheet_success.png",
            "failure": "contact_sheet_failure.png",
            "ranking_motivation": "contact_sheet_ranking_motivation.png",
        }[group_name]
        save_contact_sheet(paths, figures_dir / contact_name)
        generated[group_name] = paths
    return generated


def write_reranking_motivation(rows, ranking_cases, path):
    total = len(rows)
    j_gap = sum(1 for row in rows if (not row.get("j1_success")) and row.get("jany_success"))
    counts = Counter(row["failure_category_primary"] for row in rows if row["failure_category_primary"] != "success")
    lines = [
        "# Re-ranking Motivation from Failure Cases",
        "",
        "This analysis is diagnostic, not an improvement claim. It uses the reproduced CROG outputs to identify where a semantic-geometric re-ranking module is most defensible.",
        "",
        "## What Better Training Alone May Not Solve",
        "",
        "- Dataset or metric edge cases can remain ambiguous even when the predicted grasp is visually plausible.",
        "- Top-1 ranking failures are selection errors: the exported top-k heatmap peaks include an accepted grasp, but the first selected grasp fails.",
        "- Clutter and occlusion can require explicit clearance and collision-sensitive features rather than only stronger mask supervision.",
        "",
        "## Ranking-Related Evidence",
        "",
        f"- Samples analysed: {total}",
        f"- J@1 fails while J@Any succeeds: {j_gap} ({pct(j_gap, total)})",
        f"- Selected ranking-motivation cases: {len(ranking_cases)}",
        "",
        "## Score-Term Mapping",
        "",
        "| failure type | relevance to re-ranking | possible score terms |",
        "|---|---|---|",
        "| grounding_failure | May help only when the correct region remains in the candidate set. | M(g_i), Sem(g_i) |",
        "| localization_failure | Candidate centres can be favoured by target-mask or target-cloud proximity. | M(g_i), distance-to-mask, Q(g_i) |",
        "| orientation_failure | Needs candidate-level orientation quality, not just mask quality. | geometric orientation features, Q(g_i) |",
        "| width_failure | Needs compatibility between predicted gripper opening and object extent. | width compatibility, detector confidence |",
        "| clutter_occlusion_failure | Clearance and collision checks are directly relevant. | Clear(g_i), Coll(g_i) |",
        "| top1_ranking_failure | Direct target for re-ranking when a correct top-k grasp exists. | Q(g_i), M(g_i), Sem(g_i), Clear(g_i), Coll(g_i) |",
        "| dataset_or_metric_edge_case | Should be treated as a limitation rather than promised as solvable. | analysis flag, not an optimisation target |",
        "",
        "## Observed Failure Counts",
        "",
    ]
    for category, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {category}: {count}")
    lines.extend([
        "",
        "## Conclusion",
        "",
        "The strongest motivation for semantic-geometric re-ranking is the subset where J@Any succeeds but J@1 fails. In those samples, the model has produced at least one acceptable grasp candidate, so changing the final selection criterion could plausibly improve the selected grasp without changing the CROG architecture.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(rows, success_cases, failure_cases, ranking_cases, results_dir, figures_dir, path):
    meta_path = Path("failure_analysis/predictions/test_predictions.meta.json")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    total = len(rows)
    failures = [row for row in rows if row["failure_category_primary"] != "success"]
    j_gap = sum(1 for row in rows if (not row.get("j1_success")) and row.get("jany_success"))
    counts = Counter(row["failure_category_primary"] for row in failures)

    lines = [
        "# CROG Failure Case Analysis",
        "",
        "## 1. Purpose",
        "",
        "After reproducing the end-to-end CROG baseline, this failure analysis inspects per-sample predictions to understand where the model succeeds, where it fails, and which failures support a semantic-geometric re-ranking module.",
        "",
        "## 2. Reproduction Setting",
        "",
        f"- Checkpoint: `{meta.get('checkpoint', 'unknown')}`",
        f"- Config: `{meta.get('config', 'unknown')}`",
        f"- Split: `{meta.get('split', 'unknown')}`",
        f"- Dataset: OCID-VLG `{meta.get('version', 'unknown')}`",
        f"- Device used for export: `{meta.get('device', 'unknown')}`",
        f"- Batch size used for export: `{meta.get('batch_size', 'unknown')}`",
        f"- Samples analysed: {total}",
        "- Existing aggregate test metrics: IoU 79.02, Pr@50 95.51, Pr@60 93.26, Pr@70 85.51, Pr@80 63.56, Pr@90 16.53, J@1 83.20, J@Any 90.87.",
        "",
        "## 3. Success Examples",
        "",
    ]
    for row in success_cases[:8]:
        lines.append(f"- sample {row['sample_id']}: `{row['language_instruction']}`; IoU {format_metric(row.get('mask_iou'))}; target `{row.get('target_name')}`.")
    lines.extend([
        "",
        "### 3.1 Visual Success Gallery",
        "",
        "Each figure shows, from left to right: the RGB image, labelled ground-truth mask and grasps, CROG prediction, and their overlay. Blue rectangles are labelled ground-truth grasps, green is the original top-1 prediction, and yellow rectangles are the remaining top-5 predictions.",
        "",
    ])
    for row in success_cases[:5]:
        append_visual_case(
            lines,
            row,
            report_figure_path(row, figures_dir, "success", path),
        )
    lines.extend([
        "",
        "## 4. Failure Taxonomy",
        "",
        "The taxonomy separates grounding, localization, orientation, width, language ambiguity, clutter/occlusion, top-1 ranking, and dataset/metric edge cases. Full definitions are in `01_failure_taxonomy.md`.",
        "",
        "## 5. Failure Distribution",
        "",
        "| category | count | % of failures |",
        "|---|---:|---:|",
    ])
    for category, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {category} | {count} | {pct(count, len(failures))} |")
    lines.extend([
        "",
        "## 6. Representative Failure Cases",
        "",
    ])
    for row in failure_cases[:16]:
        figure = report_figure_path(row, figures_dir, "failure", path)
        fig_ref = f" Figure: `{figure}`." if figure else ""
        lines.append(
            f"- sample {row['sample_id']} ({row['failure_category_primary']}): "
            f"`{row['language_instruction']}`; IoU {format_metric(row.get('mask_iou'))}; "
            f"J@1={bool(row.get('j1_success'))}, J@Any={bool(row.get('jany_success'))}. "
            f"{row.get('short_reason')}.{fig_ref}"
        )
    gallery_failures = select_one_per_category(
        failure_cases,
        [
            "grounding_failure",
            "localization_failure",
            "orientation_failure",
            "width_failure",
            "clutter_occlusion_failure",
            "top1_ranking_failure",
            "dataset_or_metric_edge_case",
        ],
    )
    lines.extend([
        "",
        "### 6.1 Visual Failure Gallery",
        "",
        "The gallery selects one example from each available primary failure category. These are diagnostic heuristic categories, not manually verified causal labels.",
        "",
    ])
    for row in gallery_failures:
        append_visual_case(
            lines,
            row,
            report_figure_path(row, figures_dir, "failure", path),
        )
    lines.extend([
        "",
        "## 7. J@1 vs J@Any Analysis",
        "",
        f"J@1 fails but J@Any succeeds in {j_gap} samples ({pct(j_gap, total)}). These are the most direct evidence for final-selection errors because at least one acceptable top-k grasp exists, but the selected top-1 grasp is not accepted by the metric.",
        "",
        "## 8. Implications for Semantic-Geometric Re-ranking",
        "",
        "A re-ranker is most justified for top1_ranking_failure and some localization/clutter cases. It is less likely to recover samples where grounding is completely wrong, because all downstream candidate scoring depends on the correct target region being represented in the candidate set.",
        "",
        "Q(g_i) can preserve grasp-quality evidence, M(g_i) can favour target-mask overlap, Sem(g_i) can encode language-target consistency, Clear(g_i) can penalise cluttered approaches, and Coll(g_i) can reduce grasps that collide with neighbouring objects.",
        "",
        "## 9. Limitations",
        "",
        "- Top-k candidates are reconstructed from the predicted grasp quality map, not from a separately saved CROG candidate-ranking module.",
        "- Language ambiguity and clutter labels are heuristic.",
        "- Some visually reasonable grasps may be rejected because only labelled grasps count.",
        "- If grounding is completely wrong, final-stage re-ranking may not recover the sample.",
        "",
        "## 10. Next Steps",
        "",
        "- Reproduce the modular baseline with explicit candidate outputs.",
        "- Extract candidate-level features for Q, M, Sem, Clear, and Coll.",
        "- Implement a rule-based re-ranker.",
        "- Compare original ranking, semantic-only, geometry-only, and semantic-geometric ranking.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_slide_outline(rows, success_cases, failure_cases, ranking_cases, path):
    total = len(rows)
    j_gap = sum(1 for row in rows if (not row.get("j1_success")) and row.get("jany_success"))
    lines = [
        "# Meeting Slide Outline",
        "",
        "## Slide 1: End-to-End Baseline Reproduction",
        "- Test split: OCID-VLG multiple.",
        "- Reproduced aggregate metrics: IoU 79.02, J@1 83.20, J@Any 90.87.",
        "- Mac/MPS run used batch size 8 for training and batch size 24 for evaluation.",
        "",
        "## Slide 2: Qualitative Success Examples",
    ]
    for row in success_cases[:4]:
        lines.append(f"- sample {row['sample_id']}: {row['target_name']}, IoU {format_metric(row.get('mask_iou'))}.")
    lines.extend([
        "",
        "## Slide 3: Failure Taxonomy",
        "- Grounding, localization, orientation, width.",
        "- Language ambiguity, clutter/occlusion.",
        "- Top-1 ranking and dataset/metric edge cases.",
        "",
        "## Slide 4: Representative Failure Cases",
    ])
    for row in failure_cases[:4]:
        lines.append(f"- {row['failure_category_primary']}: sample {row['sample_id']}, {row.get('short_reason')}.")
    lines.extend([
        "",
        "## Slide 5: J@1 vs J@Any Gap",
        f"- Samples analysed: {total}.",
        f"- J@1 fails but J@Any succeeds: {j_gap}.",
        "- This indicates cases where the correct grasp appears in top-k but final selection is wrong.",
        "",
        "## Slide 6: Motivation for Re-ranking",
        "- Original final selection is driven by the predicted quality map peak.",
        "- Proposed scoring can combine Q, M, Sem, Clear, and Coll.",
        "- Ranking cases provide the cleanest motivation because they do not require model retraining to create a correct candidate.",
        "",
        "## Slide 7: Next Week Plan",
        "- Run modular baseline with candidate export.",
        "- Extract candidate-level semantic and geometric features.",
        "- Build rule-based re-ranking prototype.",
        "- Compare original, semantic-only, geometry-only, and semantic-geometric ranking.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cli = parse_args()
    predictions_path = Path(cli.predictions)
    results_dir = Path(cli.results_dir)
    figures_dir = Path(cli.figures_dir)
    ensure_dir(results_dir)
    ensure_dir(figures_dir)

    rows = diagnostics_rows(read_jsonl(predictions_path))
    success_cases = select_success_cases(rows)
    failure_cases = select_failure_cases(rows)
    ranking_cases = select_ranking_cases(rows)

    write_csv(results_dir / "per_sample_diagnostics.csv", rows, selected_fields())
    write_csv(results_dir / "success_cases.csv", success_cases, selected_fields())
    write_csv(results_dir / "failure_cases.csv", failure_cases, selected_fields())
    write_csv(results_dir / "ranking_motivation_cases.csv", ranking_cases, selected_fields())
    write_missing_diagnostics(rows, results_dir / "missing_diagnostics.md")
    write_summary(rows, success_cases, failure_cases, ranking_cases, results_dir)
    save_plots(rows, figures_dir)

    if not cli.skip_figures:
        generate_case_figures(
            {
                "success": success_cases,
                "failure": failure_cases,
                "ranking_motivation": ranking_cases,
            },
            figures_dir,
            cli.max_figures_per_group,
        )
        # Save CSVs again after figure paths have been attached.
        write_csv(results_dir / "success_cases.csv", success_cases, selected_fields() + ["figure_path"])
        write_csv(results_dir / "failure_cases.csv", failure_cases, selected_fields() + ["figure_path"])
        write_csv(results_dir / "ranking_motivation_cases.csv", ranking_cases, selected_fields() + ["figure_path"])

    write_reranking_motivation(rows, ranking_cases, "failure_analysis/02_reranking_motivation.md")
    write_report(rows, success_cases, failure_cases, ranking_cases, results_dir, figures_dir, "failure_analysis/CROG_failure_case_analysis_report.md")
    write_slide_outline(rows, success_cases, failure_cases, ranking_cases, "failure_analysis/meeting_slides_outline.md")

    print(f"samples={len(rows)}")
    print(f"success_cases={len(success_cases)}")
    print(f"failure_cases={len(failure_cases)}")
    print(f"ranking_motivation_cases={len(ranking_cases)}")


if __name__ == "__main__":
    main()
