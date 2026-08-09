#!/usr/bin/env python3
"""Run every remaining experiment phase in the mandated scientific order."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PYTHON = Path("/opt/anaconda3/bin/python")
SAM_PYTHON = ROOT / ".venv-sam3-cpu/bin/python"
LOG_ROOT = ROOT / "outputs/selective_sam3_vg/execution_logs"


def _run(name: str, command: list[str], *, allow_failure: bool = False) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    print(f"START {name}: {' '.join(command)}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    print(f"END {name}: exit={result.returncode} seconds={time.time()-started:.2f} log={log_path}", flush=True)
    if result.returncode and not allow_failure:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"{name} failed with exit {result.returncode}:\n{tail}")
    return result.returncode


def main() -> None:
    initial = ROOT / "outputs/selective_sam3_vg/validation_pilot/initial_prompt_comparison"
    statuses = list(initial.glob("*/*/status.json"))
    if len(statuses) != 1500:
        raise RuntimeError(f"initial pilot is incomplete: {len(statuses)}/1500")
    initial_analysis = ROOT / "outputs/selective_sam3_vg/validation_pilot/initial_analysis"
    _run(
        "05_analyze_initial_pilot",
        [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/analyze_pilot.py", "--input-root", str(initial), "--output-root", str(initial_analysis), "--expected-configurations", "1500"],
    )
    comparison = pd.read_csv(initial_analysis / "prompt_comparison.csv")
    best_box = comparison[comparison["prompt_family"].isin(["P1", "P2", "P3", "P4"])].sort_values("validation_rank").iloc[0]
    family = str(best_box["prompt_family"])
    expansion_root = ROOT / "outputs/selective_sam3_vg/validation_pilot/box_expansion_comparison"
    for expansion in (0.00, 0.03, 0.08):
        destination = expansion_root / f"exp_{int(expansion*100):03d}"
        _run(
            f"05_expand_{int(expansion*100):03d}",
            [
                str(SAM_PYTHON), "tools/selective_sam3_vg/run_pilot_inference.py",
                "--output-root", str(destination), "--families", family,
                "--box-expansion", str(expansion),
            ],
        )
    final_pilot = ROOT / "outputs/selective_sam3_vg/validation_pilot/final_analysis"
    _run(
        "06_analyze_final_pilot",
        [
            str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/analyze_pilot.py",
            "--input-root", str(initial), str(expansion_root),
            "--output-root", str(final_pilot), "--expected-configurations", "2400",
        ],
    )
    locked_prompt = json.loads((final_pilot / "analysis_summary.json").read_text())
    family = str(locked_prompt["selected_prompt_family"])
    expansion = float(locked_prompt["selected_box_expansion_fraction"])
    validation_root = ROOT / "outputs/selective_sam3_vg/validation_full"
    _run(
        "07_full_validation_sam",
        [
            str(SAM_PYTHON), "tools/selective_sam3_vg/run_pilot_inference.py",
            "--manifest", str(ROOT / "outputs/selective_sam3_vg/splits/validation_inference_manifest.jsonl"),
            "--output-root", str(validation_root), "--families", family,
            "--box-expansion", str(expansion),
        ],
    )
    validation_analysis = ROOT / "outputs/selective_sam3_vg/validation_full_analysis"
    _run(
        "08_full_validation_oracle",
        [
            str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/analyze_pilot.py",
            "--stratification-manifest", str(ROOT / "outputs/selective_sam3_vg/diagnostics/baseline_validation_per_sample_metrics.parquet"),
            "--input-root", str(validation_root), "--output-root", str(validation_analysis),
            "--expected-configurations", "3778",
        ],
    )
    _run(
        "10_11_fit_trigger_selector",
        [
            str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/fit_trigger_selector.py",
            "--candidate-evaluation", str(validation_analysis / "pilot_candidate_evaluation.parquet"),
            "--validation-output-root", str(validation_root),
        ],
    )
    if not (ROOT / "artifacts/selective_sam3_vg/formal_lock.json").exists():
        _run(
            "13_lock_formal_method",
            [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/lock_formal_method.py", "--prompt-analysis", str(final_pilot / "analysis_summary.json")],
        )
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
            str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/reuse_formal_for_oracle.py",
            "--manifest", str(oracle_manifest), "--family", family,
            "--box-expansion", str(expansion), "--output-root", str(oracle_output),
        ],
    )
    _run(
        "18_oracle_sam_inference",
        [
            str(SAM_PYTHON), "tools/selective_sam3_vg/run_pilot_inference.py",
            "--manifest", str(oracle_manifest), "--output-root", str(oracle_output),
            "--families", family, "--box-expansion", str(expansion), "--oracle-diagnostic",
        ],
    )
    oracle_count = sum(1 for line in oracle_manifest.read_text().splitlines() if line)
    oracle_analysis = oracle_root / "ORACLE_GT_SELECTED_analysis"
    _run(
        "18_analyze_oracle_candidates",
        [
            str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/analyze_pilot.py",
            "--stratification-manifest", str(oracle_root / "ORACLE_GT_SELECTED_cohort_stratification.parquet"),
            "--input-root", str(oracle_output), "--output-root", str(oracle_analysis),
            "--expected-configurations", str(oracle_count),
            "--diagnostic-label", "Diagnostic upper bound; uses test ground truth; not deployable.",
        ],
    )
    _run("18_oracle_global_ceiling", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/analyze_oracle_test.py"])
    _run("19_grouped_failure_analysis", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/grouped_failure_analysis.py"])
    _run("19_galleries", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/generate_galleries.py"])
    _run("19_canonical_export", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/export_canonical_masks.py"])
    _run("19_figures", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/generate_figures.py"])
    focused = _run("20_focused_tests", [str(ANALYSIS_PYTHON), "-m", "pytest", "-q", "tests/test_selective_sam3_vg.py"], allow_failure=True)
    leakage = _run("20_leakage_guard", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/leakage_guard.py"], allow_failure=True)
    complete = _run("20_complete_test_suite", [str(ANALYSIS_PYTHON), "-m", "pytest", "-q"], allow_failure=True)
    test_summary = {
        "focused": "PASS" if focused == 0 else f"FAIL (exit {focused})",
        "leakage_guard": "PASS" if leakage == 0 else f"FAIL (exit {leakage})",
        "complete_suite": "PASS" if complete == 0 else f"FAIL (exit {complete})",
        "logs": str(LOG_ROOT),
    }
    report_root = ROOT / "outputs/selective_sam3_vg/report"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "test_run_summary.json").write_text(json.dumps(test_summary, indent=2, sort_keys=True) + "\n")
    _run("21_write_report", [str(ANALYSIS_PYTHON), "tools/selective_sam3_vg/write_final_report.py"])
    print(json.dumps({"status": "COMPLETED", "locked_prompt_family": family, "box_expansion": expansion}, indent=2))


if __name__ == "__main__":
    main()
