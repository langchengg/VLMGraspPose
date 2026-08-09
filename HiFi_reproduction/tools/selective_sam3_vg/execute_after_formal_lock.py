#!/usr/bin/env python3
"""Resume the experiment after the immutable pre-test method lock exists."""

from __future__ import annotations

import json

import yaml

from execute_remaining_experiment import ANALYSIS_PYTHON, LOG_ROOT, ROOT, SAM_PYTHON, _run


def main() -> None:
    formal_lock = ROOT / "artifacts/selective_sam3_vg/formal_lock.json"
    locked_config_path = ROOT / "configs/selective_sam3_vg_LOCKED.yaml"
    if not formal_lock.is_file() or not locked_config_path.is_file():
        raise RuntimeError("the immutable pre-test method lock is missing")
    config = yaml.safe_load(locked_config_path.read_text(encoding="utf-8"))
    family = str(config["prompt"]["family"])
    expansion = float(config["prompt"]["box_expansion_fraction"])

    _run("14_formal_inference", [str(SAM_PYTHON), "tools/selective_sam3_vg/run_formal_inference.py"])
    if not (ROOT / "artifacts/selective_sam3_vg/formal_output_lock.json").exists():
        _run("15_lock_formal_outputs", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/lock_formal_outputs.py"])
    _run("16_evaluate_formal", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/evaluate_formal.py"])
    _run("17_statistics", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/statistical_analysis.py"])
    _run("18_export_oracle_cohort", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/export_oracle_test_cohort.py"])

    oracle_root = ROOT / "outputs/selective_sam3_vg/oracle"
    oracle_manifest = oracle_root / "ORACLE_GT_SELECTED_inference_manifest.jsonl"
    oracle_output = oracle_root / "ORACLE_GT_SELECTED_hypotheses"
    _run(
        "18_reuse_formal_oracle_hypotheses",
        [
            str(ANALYSIS_PYTHON),
            "tools/selective_sam3_vg/reuse_formal_for_oracle.py",
            "--manifest",
            str(oracle_manifest),
            "--family",
            family,
            "--box-expansion",
            str(expansion),
            "--output-root",
            str(oracle_output),
        ],
    )
    _run(
        "18_oracle_sam_inference",
        [
            str(SAM_PYTHON),
            "tools/selective_sam3_vg/run_pilot_inference.py",
            "--manifest",
            str(oracle_manifest),
            "--output-root",
            str(oracle_output),
            "--families",
            family,
            "--box-expansion",
            str(expansion),
            "--oracle-diagnostic",
        ],
    )
    oracle_count = sum(1 for line in oracle_manifest.read_text().splitlines() if line)
    oracle_analysis = oracle_root / "ORACLE_GT_SELECTED_analysis"
    _run(
        "18_analyze_oracle_candidates",
        [
            str(ANALYSIS_PYTHON),
            "tools/selective_sam3_vg/analyze_pilot.py",
            "--stratification-manifest",
            str(oracle_root / "ORACLE_GT_SELECTED_cohort_stratification.parquet"),
            "--input-root",
            str(oracle_output),
            "--output-root",
            str(oracle_analysis),
            "--expected-configurations",
            str(oracle_count),
            "--diagnostic-label",
            "Diagnostic upper bound; uses test ground truth; not deployable.",
        ],
    )
    _run("18_oracle_global_ceiling", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/analyze_oracle_test.py"])
    _run("19_grouped_failure_analysis", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/grouped_failure_analysis.py"])
    _run("19_galleries", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/generate_galleries.py"])
    _run("19_canonical_export", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/export_canonical_masks.py"])
    _run("19_figures", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/generate_figures.py"])

    focused = _run(
        "20_focused_tests",
        [str(ANALYSIS_PYTHON), "-m", "pytest", "-q", "tests/test_selective_sam3_vg.py"],
        allow_failure=True,
    )
    leakage = _run(
        "20_leakage_guard",
        [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/leakage_guard.py"],
        allow_failure=True,
    )
    complete = _run(
        "20_complete_test_suite",
        [str(ANALYSIS_PYTHON), "-m", "pytest", "-q"],
        allow_failure=True,
    )
    report_root = ROOT / "outputs/selective_sam3_vg/report"
    report_root.mkdir(parents=True, exist_ok=True)
    test_summary = {
        "focused": "PASS" if focused == 0 else f"FAIL (exit {focused})",
        "leakage_guard": "PASS" if leakage == 0 else f"FAIL (exit {leakage})",
        "complete_suite": "PASS" if complete == 0 else f"FAIL (exit {complete})",
        "logs": str(LOG_ROOT),
    }
    (report_root / "test_run_summary.json").write_text(
        json.dumps(test_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _run("21_write_report", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/write_final_report.py"])
    print(json.dumps({"status": "COMPLETED", "locked_prompt_family": family, "box_expansion": expansion}, indent=2))


if __name__ == "__main__":
    main()
