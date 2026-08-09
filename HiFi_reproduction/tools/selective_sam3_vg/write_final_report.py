#!/usr/bin/env python3
"""Assemble the complete selective-SAM3 scientific experiment report."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _pct(value: float) -> str:
    return f"{100 * float(value):.4f}"


def _metric_row(name: str, source: dict, *, oracle: bool = False) -> str:
    label = name + (" **(non-deployable)**" if oracle else "")
    return "| " + " | ".join(
        [label, _pct(source["mean_iou"])]
        + [_pct(source[f"p_at_{percent}"]) for percent in (50, 60, 70, 80, 90)]
    ) + " |"


def main() -> None:
    report_root = ROOT / "outputs/selective_sam3_vg/report"
    formal = json.loads((report_root / "formal_metrics.json").read_text())
    oracle = json.loads((ROOT / "outputs/selective_sam3_vg/oracle/ORACLE_GT_SELECTED_summary.json").read_text())
    gap = json.loads((ROOT / "outputs/selective_sam3_vg/diagnostics/paper_gap_requirements.json").read_text())
    split = json.loads((ROOT / "outputs/selective_sam3_vg/splits/split_audit.json").read_text())
    validation_oracle = json.loads((ROOT / "outputs/selective_sam3_vg/validation_full_analysis/analysis_summary.json").read_text())
    trigger = json.loads((ROOT / "artifacts/selective_sam3_vg/locked_trigger/calibration.json").read_text())
    selector = json.loads((ROOT / "artifacts/selective_sam3_vg/locked_selector/validation_results.json").read_text())
    lock = json.loads((ROOT / "artifacts/selective_sam3_vg/formal_lock.json").read_text())
    output_lock = json.loads((ROOT / "artifacts/selective_sam3_vg/formal_output_lock.json").read_text())
    grouped = json.loads((report_root / "grouped/grouped_failure_summary.json").read_text())
    canonical = json.loads((ROOT / "artifacts/selective_sam3_vg/canonical_export_decision.json").read_text())
    tests = json.loads((report_root / "test_run_summary.json").read_text())
    config = yaml.safe_load((ROOT / "configs/selective_sam3_vg_LOCKED.yaml").read_text())
    confidence = pd.read_csv(report_root / "confidence_intervals.csv")
    mcnemar = pd.read_csv(report_root / "mcnemar_tests.csv")
    prompt_comparison = pd.read_csv(ROOT / "outputs/selective_sam3_vg/validation_pilot/final_analysis/prompt_comparison.csv")
    query_groups = pd.read_csv(report_root / "grouped/results_by_query_type.csv")
    size_groups = pd.read_csv(report_root / "grouped/results_by_target_area_bin.csv")
    formal_per_sample = pd.read_parquet(report_root / "formal_per_sample_metrics.parquet")
    timing_values = []
    for sample_id in formal_per_sample["sample_id"]:
        timing = json.loads((ROOT / "outputs/selective_sam3_vg/formal_test_masks" / sample_id / "timing.json").read_text())
        timing_values.append(float(timing["total_sample_seconds"]))
    final_resume_contract = json.loads(
        (ROOT / "outputs/selective_sam3_vg/formal_test_masks/inference_contract.json").read_text()
    )
    peak_memory = []
    for path in (ROOT / "outputs/selective_sam3_vg/formal_test_masks").glob("*/memory.json"):
        peak_memory.append(json.loads(path.read_text())["peak_rss_bytes"])
    diff_stat = subprocess.run(
        ["git", "diff", "--stat", "--", "HiFi_reproduction"],
        cwd=ROOT.parent, capture_output=True, text=True, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--", "HiFi_reproduction"],
        cwd=ROOT.parent, capture_output=True, text=True, check=False,
    ).stdout.strip()

    baseline, hybrid, paper = formal["baseline"], formal["locked_selective_sam3"], formal["paper_numeric_reference"]
    lines = [
        "# Selective SAM 3 Visual-Grounding Results",
        "",
        "## Executive result",
        "",
        "This experiment evaluates a locked, GT-free-at-inference selective SAM 3 boundary-refinement layer on top of the frozen repeated-FiLM HiFi-CS reproduction. HiFi-CS retains object-identity responsibility; SAM 3 is invoked only by the locked pre-SAM trigger, and every replacement must pass the locked conservative selector. Test GT was loaded only after all 7,675 final masks were checksummed and locked.",
        "",
        "| Method | mIoU | P@50 | P@60 | P@70 | P@80 | P@90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _metric_row("HiFi-CS paper reference", paper),
        _metric_row("Local repeated-FiLM baseline", baseline),
        "| Unconditional SAM 3 | N/A | N/A | N/A | N/A | N/A | N/A |",
        _metric_row("Locked selective SAM 3", hybrid),
        _metric_row("GT-selected SAM oracle", oracle["ORACLE_GT_SELECTED"], oracle=True),
        "",
        "The paper row is a **paper numeric reference**, not an exact local-protocol reproduction. The paper describes a 70/30 protocol, whereas the authoritative local unique split contains 26,295 train / 3,778 validation / 7,675 test expressions and is scene-disjoint. No full-protocol unconditional-SAM result exists; the earlier ten-sample run used an incompatible earlier input contract and is therefore not promoted into this table.",
        "",
        "## 1. Authoritative baseline source and exact reproduction",
        "",
        f"- Frozen repeated-FiLM checkpoint: `{ROOT / 'runs/hifics_ocidvlg_hierfilm_20260727_214615/checkpoints/best.pth'}`",
        "- Frozen checkpoint SHA-256: `b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601`.",
        f"- Locked trigger artifact SHA-256: `{config['trigger']['artifact_sha256']}`.",
        f"- Authoritative prediction root: `{ROOT / 'runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528/masks/hierfilm'}`",
        f"- Test manifest: `{ROOT / 'artifacts/data_audit/frozen_manifests/ocidvlg_unique_test.json'}`",
        f"- Exact denominator: {baseline['samples']:,}.",
        f"- Exact recomputed mIoU: `{baseline['mean_iou']:.16f}`; P@50 `{baseline['p_at_50_numerator']}/{baseline['p_at_50_denominator']}`; P@90 `{baseline['p_at_90_numerator']}/{baseline['p_at_90_denominator']}`.",
        "- Every original probability and mask checksum was verified before development and again in the integrity suite.",
        "",
        "## 2. Protocol and split provenance",
        "",
        f"The local split audit status is `{split['status']}`. Scene counts are train={split['scene_counts']['train']}, validation={split['scene_counts']['val']}, test={split['scene_counts']['test']}; every pair has zero scene overlap and zero processed-RGB overlap. Prompt selection, trigger training, selector training, and all thresholds were completed on validation before the formal lock timestamp `{lock['locked_timestamp_utc']}`. The final-mask output lock timestamp is `{output_lock['locked_timestamp_utc']}`.",
        "",
        "## 3. Required recovery to equal the paper numeric reference",
        "",
        f"The mIoU gap requires a total IoU sum gain of `{gap['required_total_iou_gain']:.6f}` across 7,675 test expressions.",
        "",
        "| Threshold | Baseline successes | Required successes | Additional successes required |",
        "|---|---:|---:|---:|",
    ]
    for percent in (50, 60, 70, 80, 90):
        item = gap["thresholds"][f"p_at_{percent}"]
        lines.append(f"| P@{percent} | {item['baseline_successes']}/7675 | {item['required_successes']}/7675 | {item['required_additional_successes']} |")
    lines += [
        "",
        "## 4. Prompt pilot and validation oracle",
        "",
        f"The deterministic pilot contains 300 expressions spanning all 165 validation scenes. It compared P0–P4 at 5% box expansion, then tested 0/3/5/8% expansion for the best box family. The locked prompt is **{config['prompt']['family']}**, expansion **{100*config['prompt']['box_expansion_fraction']:.0f}%**, output threshold 0.5.",
        "",
        "Top validation-pilot configurations (GT-selected diagnostic only):",
        "",
        "| Rank | Prompt | Expansion | Oracle mIoU | Oracle P@90 | Any positive SAM gain |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in prompt_comparison.head(8).itertuples(index=False):
        lines.append(
            f"| {row.validation_rank} | {row.prompt_family} | {100*row.box_expansion_fraction:.0f}% | "
            f"{100*row.hybrid_oracle_mean_iou:.3f} | {100*row.hybrid_oracle_p_at_90:.3f} | {100*row.fraction_any_positive_sam_gain:.2f}% |"
        )
    selected_validation = validation_oracle["selected_metrics"]
    lines += [
        "",
        f"On the complete 3,778-sample validation split, the GT-selected hybrid ceiling for the locked prompt is mIoU `{100*selected_validation['hybrid_oracle_mean_iou']:.4f}` and P@90 `{100*selected_validation['hybrid_oracle_p_at_90']:.4f}`. This is an oracle diagnostic, not the deployable selector result.",
        "",
        "## 5. Locked trigger and selector",
        "",
        f"- Trigger model: `{config['trigger']['model_name']}` with threshold `{config['trigger']['threshold']:.8f}`; grouped OOF PR-AUC `{trigger['selected']['pr_auc_oof']:.6f}`, precision `{trigger['selected']['calibration']['precision']:.6f}`, recall `{trigger['selected']['calibration']['recall']:.6f}`, and triggered fraction `{trigger['selected']['calibration']['triggered_fraction']:.6f}`.",
        f"- Selector: `{config['selector']['selector_type']}` with acceptance margin `{config['selector']['acceptance_margin']:.8f}` and a hard conservative gate retaining `coarse_0` as candidate zero.",
        f"- Validation selector result: accepted {selector['selected']['accepted_count']} candidates, of which {selector['selected']['accepted_improvement_count']} improved and {selector['selected']['accepted_harmful_count']} degraded validation IoU.",
        f"- Formal config SHA-256: `{lock['config_sha256']}`; `LOCKED_BEFORE_TEST=true`.",
        "",
        "## 6. Formal test result and threshold transitions",
        "",
        f"SAM was invoked for `{formal['selective_counts']['trigger_count']}` samples ({100*formal['selective_counts']['sam_invocation_rate']:.3f}%) and accepted for `{formal['selective_counts']['sam_acceptance_count']}` ({100*formal['selective_counts']['sam_acceptance_rate']:.3f}%); `{formal['selective_counts']['hifi_fallback_count']}` retained HiFi. Accepted replacements improved {formal['selective_counts']['accepted_sam_improvement_count']} and degraded {formal['selective_counts']['accepted_sam_degradation_count']} samples.",
        "",
        "| Threshold | Baseline successes | Hybrid successes | Recovered | Harmed | Net |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in formal["threshold_transitions"]:
        percent = int(round(100 * item["threshold"]))
        lines.append(f"| P@{percent} | {item['baseline_successes']}/7675 | {item['hybrid_successes']}/7675 | {item['recovered']} | {item['harmed']} | {item['net']} |")
    lines += [
        "",
        f"Absolute mIoU change: `{hybrid['mean_iou'] - baseline['mean_iou']:+.8f}`; relative change: `{100*(hybrid['mean_iou']/baseline['mean_iou']-1):+.4f}%`; paper mIoU gap closed: `{100*formal['fraction_of_paper_gap_closed']['mean_iou']:.3f}%`.",
        "",
        "Fraction of paper gap closed by threshold: " + ", ".join(
            f"P@{p} `{100*formal['fraction_of_paper_gap_closed'][f'p_at_{p}']:.2f}%`" for p in (50, 60, 70, 80, 90)
        ) + ".",
        "",
        "## 7. Statistical analysis",
        "",
        "Point estimates use all 7,675 expressions. Wilson intervals are sample-level binomial intervals; bootstrap intervals resample the 325 scenes as clusters. All 10,000 bootstrap replicates and paired differences use seed 42.",
        "",
        "| Metric | Estimate | Wilson 95% CI | Scene-bootstrap 95% CI | Paired difference 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in confidence[confidence["method"] == "locked_selective_sam3"].itertuples(index=False):
        wilson = "—" if pd.isna(getattr(row, "wilson_ci_low", np.nan)) else f"[{row.wilson_ci_low:.5f}, {row.wilson_ci_high:.5f}]"
        paired = "—" if pd.isna(getattr(row, "paired_difference_ci_low", np.nan)) else f"[{row.paired_difference_ci_low:.5f}, {row.paired_difference_ci_high:.5f}]"
        lines.append(f"| {row.metric} | {row.point_estimate:.6f} | {wilson} | [{row.scene_clustered_bootstrap_ci_low:.5f}, {row.scene_clustered_bootstrap_ci_high:.5f}] | {paired} |")
    lines += [
        "",
        "| Threshold | Recovered | Harmed | Net | Outcome-changing precision | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in mcnemar.itertuples(index=False):
        lines.append(f"| P@{int(100*row.threshold)} | {row.recovered} | {row.harmed} | {row.net_change} | {100*row.outcome_changing_precision:.2f}% | {row.exact_mcnemar_p_value:.4g} |")
    lines += [
        "",
        "## 8. Grouped failure analysis",
        "",
        "Query-type results:",
        "",
        "| Query type | N | Baseline mIoU | Hybrid mIoU | Delta | Trigger rate | Acceptance rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in query_groups.itertuples(index=False):
        lines.append(f"| {row.group} | {row.samples} | {_pct(row.baseline_mean_iou)} | {_pct(row.hybrid_mean_iou)} | {100*row.mean_delta_iou:+.3f} | {100*row.trigger_rate:.2f}% | {100*row.sam_acceptance_rate:.2f}% |")
    lines += [
        "",
        "Mask-size results:",
        "",
        "| Size | N | Baseline mIoU | Hybrid mIoU | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in size_groups.itertuples(index=False):
        lines.append(f"| {row.group} | {row.samples} | {_pct(row.baseline_mean_iou)} | {_pct(row.hybrid_mean_iou)} | {100*row.mean_delta_iou:+.3f} |")
    lines += [
        "",
        f"Clutter level was not reported because no authoritative clutter field exists in the frozen local manifest. Accepted harmful replacements: {grouped['accepted_harmful_samples']}. These grouped findings are descriptive and do not establish causal effects.",
        "",
        "## 9. GT-selected test oracle (non-deployable)",
        "",
        "**Diagnostic upper bound; uses test ground truth; not deployable.**",
        "",
        f"The oracle cohort contains {oracle['cohort_samples']} samples with 0.60 ≤ baseline IoU < 0.90. `{oracle['samples_with_no_positive_sam_gain']}` cannot be improved by the configured SAM hypotheses. The global GT-selected ceiling is mIoU `{100*oracle['ORACLE_GT_SELECTED']['mean_iou']:.4f}` and P@90 `{100*oracle['ORACLE_GT_SELECTED']['p_at_90']:.4f}`. Its theoretically recoverable fraction of the paper mIoU gap is `{100*oracle['fraction_of_paper_gap_theoretically_recoverable']['mean_iou']:.3f}%`.",
        "",
        "Accordingly, whether the current prompt/checkpoint can numerically reach the paper reference is determined by the oracle row above; it is never substituted for the locked formal result.",
        "",
        "## 10. Runtime, memory, and outputs",
        "",
        f"- Formal accumulated per-sample processing time across resumable invocations: `{np.sum(timing_values):.2f}` s; mean `{np.mean(timing_values):.4f}` s; median `{np.median(timing_values):.4f}` s. The final resume segment contract recorded `{final_resume_contract['wall_seconds']:.2f}` s.",
        f"- Peak recorded process RSS across SAM invocations: `{max(peak_memory, default=0) / (1024**3):.3f}` GiB.",
        f"- Model: official `facebook/sam3` revision `{config['model']['revision']}`, CPU only, float32, batch size 1, persistent load, `local_files_only=true`.",
        f"- Formal masks: `{ROOT / 'outputs/selective_sam3_vg/formal_test_masks'}`",
        f"- Formal metrics/statistics: `{report_root}`",
        f"- Figures: `{report_root / 'figures'}` (PNG/PDF/SVG).",
        f"- Galleries: `{report_root / 'galleries'}`.",
        f"- Canonical export decision: `{canonical['status']}`; Dex-Net/GQ-CNN/VGN were not run.",
        "",
        "## 11. Tests and integrity checks",
        "",
        f"- Focused selective-SAM3 tests: `{tests['focused']}`.",
        f"- Complete existing suite: `{tests['complete_suite']}`.",
        f"- Static leakage guard: `{tests['leakage_guard']}`.",
        f"- Final terminal samples: `{output_lock['sample_count']}/7675`.",
        "- The original mask remains candidate `coarse_0`; prompt generation and formal selection are deterministic; all final masks are binary 480×640 with finite float32 probability maps.",
        "",
        "## 12. External resources and reuse decision",
        "",
        "- [Official Hugging Face Transformers SAM 3 documentation](https://huggingface.co/docs/transformers/model_doc/sam3): API and multimask/Tracker behavior source of truth.",
        "- [Official facebook/sam3 model card](https://huggingface.co/facebook/sam3): checkpoint identity and use constraints.",
        "- [Official HiFi-CS repository](https://github.com/vineet2104/hifics) and [paper v2](https://arxiv.org/abs/2409.10419): architecture/protocol and paper numeric references.",
        "- No external implementation code was copied. The HiFi-CS repository did not expose a clear top-level license during the audit, so it was used only as a behavioral reference. Existing local SAM 3 CPU adapters and serialization utilities were reused.",
        "",
        "## 13. Files changed and git diff",
        "",
        "The repository was already dirty with unrelated Dex-Net/GQ-CNN/VGN and report changes. The raw scoped status below is included for audit completeness; it is not a claim that every listed path was changed by this selective-SAM3 task.",
        "",
        "```text",
        status or "(clean within scoped path)",
        "```",
        "",
        "`git diff --stat`:",
        "",
        "```text",
        diff_stat or "(no tracked diff stat; new experiment files may be untracked)",
        "```",
        "",
        "## 14. Remaining limitations",
        "",
        "- Local and paper protocols are not proven identical; paper numbers remain numeric references.",
        "- SAM 3 was not fine-tuned, and the current oracle is limited to the locked visual-prompt family and multimask outputs.",
        "- Trigger and selector training uses one official validation split with grouped OOF estimates; a second independent development split was unavailable.",
        "- Grouped analyses are observational, and the local manifest lacks an authoritative clutter label.",
        "- No grasping stage was executed; downstream effects must be tested only from the canonical root if its safety gate passed.",
        "",
    ]
    output = ROOT / "docs/SELECTIVE_SAM3_VISUAL_GROUNDING_RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "report": str(output), "lines": len(lines)}, indent=2))


if __name__ == "__main__":
    main()
