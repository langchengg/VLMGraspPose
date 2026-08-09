from __future__ import annotations

import csv
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


METHODS = (
    "crog_q_only",
    "gemini_robotics_er2_crog_evidence_direct",
    "gemini_robotics_er2_crog_evidence_safe",
    "gemini_3_6_flash_crog_evidence_direct",
    "gemini_3_6_flash_crog_evidence_safe",
    "gemini_dual_consensus_safe",
    "locked_gemini_primary",
)


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_blocked_results_bundle(
    *,
    run_root: str | Path,
    phase0_path: str | Path,
    evidence_summary_path: str | Path,
    smoke_status_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_root)
    phase0, evidence, smoke = map(_read, (phase0_path, evidence_summary_path, smoke_status_path))
    baseline = phase0["baseline"]
    method_rows = []
    for method in METHODS:
        row = {
            "method": method,
            "status": "complete" if method == "crog_q_only" else "not_run_missing_api_key",
            "legacy_j1": baseline["legacy_q_only_j1"] if method == "crog_q_only" else None,
            "legacy_delta_pp": 0.0 if method == "crog_q_only" else None,
            "legacy_recovered": 0 if method == "crog_q_only" else None,
            "legacy_harmful": 0 if method == "crog_q_only" else None,
            "legacy_net": 0 if method == "crog_q_only" else None,
            "corrected_j1": baseline["corrected_q_only_j1"] if method == "crog_q_only" else None,
            "corrected_delta_pp": 0.0 if method == "crog_q_only" else None,
            "corrected_recovered": 0 if method == "crog_q_only" else None,
            "corrected_harmful": 0 if method == "crog_q_only" else None,
            "corrected_net": 0 if method == "crog_q_only" else None,
            "switch_coverage": 0.0 if method == "crog_q_only" else None,
            "outcome_precision": None,
            "holm_p": None,
            "scene_bootstrap_ci": None,
        }
        method_rows.append(row)
    with (root / "per_method_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)

    sample_table = pq.read_table(Path(evidence_summary_path).parent / "sample_evidence.parquet")
    sample_records = []
    for row in sample_table.to_pylist():
        sample_records.append(
            {
                "sample_id": row["sample_id"],
                "partition": "development_smoke",
                "q_only_candidate_id": row["original_q_top1_candidate_id"],
                "er2_candidate_id": None,
                "flash_candidate_id": None,
                "safe_candidate_id": row["original_q_top1_candidate_id"],
                "status": "not_run_missing_api_key",
            }
        )
    pq.write_table(pa.Table.from_pylist(sample_records), root / "per_sample_predictions.parquet", compression="zstd")
    empty_scores_schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("model_id", pa.string()),
            ("candidate_id", pa.string()),
            ("display_id", pa.string()),
            ("overall_score", pa.float64()),
            ("reason_codes_json", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist([], schema=empty_scores_schema), root / "per_candidate_gemini_scores.parquet")
    candidate_source = Path(evidence_summary_path).parent / "candidate_evidence.parquet"
    target = root / "per_candidate_evidence.parquet"
    if target.resolve() != candidate_source.resolve():
        target.write_bytes(candidate_source.read_bytes())
    for source, destination in (
        (Path(evidence_summary_path).parent / "request_manifest.parquet", root / "request_manifest.parquet"),
        (Path(evidence_summary_path).parent / "candidate_mapping.parquet", root / "candidate_mapping.parquet"),
        (Path(smoke_status_path).parent / "per_model_decisions.parquet", root / "per_model_decisions.parquet"),
    ):
        if source.resolve() != destination.resolve():
            destination.write_bytes(source.read_bytes())

    _write(root / "statistical_tests.json", {"status": "not_run", "reason": "no Gemini decisions", "planned": ["exact McNemar", "Holm"]})
    _write(root / "bootstrap_intervals.json", {"status": "not_run", "reason": "no Gemini decisions", "draws": 10000, "clusters": ["frame", "scene"]})
    with (root / "threshold_sweeps.csv").open("w", encoding="utf-8") as handle:
        handle.write("model,confidence_threshold,margin_threshold,overall_threshold,recovered,harmful,net,status\n")
    _write(root / "model_agreement.json", {"status": "not_run", "reason": "both model smokes blocked"})
    _write(root / "stability_results.json", {"status": "not_run", "planned_samples": 20, "planned_repeats": 3})
    _write(
        root / "api_runtime_metrics.json",
        {
            "status": smoke["status"],
            "prepared_requests": smoke["prepared_request_count"],
            "live_api_requests": smoke["live_api_request_count"],
            "cache_hits": sum(bool(item["cache_hit"]) for item in smoke["decisions"]),
            "latency_seconds": [],
        },
    )
    with (root / "api_token_usage.csv").open("w", encoding="utf-8") as handle:
        handle.write("sample_id,model_id,input_tokens,output_tokens,thought_tokens,total_tokens\n")
    flash_estimated_request = (2620 * 1.50 + 512 * 7.50) / 1_000_000
    _write(
        root / "cost_estimate.json",
        {
            "status": "partial_estimate_only",
            "gemini_3_6_flash": {
                "standard_input_usd_per_million": 1.50,
                "standard_output_including_thought_usd_per_million": 7.50,
                "illustrative_tokens_per_request": {"input": 2620, "output_including_thought": 512},
                "illustrative_cost_per_request_usd": flash_estimated_request,
                "validation_8669_usd": flash_estimated_request * 8669,
                "formal_test_17749_usd": flash_estimated_request * 17749,
            },
            "gemini_robotics_er_2_preview": {
                "status": "unknown",
                "reason": "official ER2 pricing was not published in the audited public pricing page",
            },
            "batch": {
                "interactions_supported": False,
                "note": "Gemini Batch is currently a generateContent transport, not Batch Interactions",
            },
            "budget_guard_upper_bound_per_request_usd": 0.10,
        },
    )
    _write(
        root / "actual_cost_estimate.json",
        {"live_api_requests": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "actual_estimated_cost_usd": 0.0},
    )
    fallback_counts = {}
    for item in smoke["decisions"]:
        reason = item["fallback_reason"] or "none"
        fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
    _write(
        root / "fallback_summary.json",
        {
            "technical_fallback_count": len(smoke["decisions"]),
            "model_abstain_count": 0,
            "by_reason": fallback_counts,
        },
    )
    _write(
        root / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "google_genai": "2.16.0",
            "gemini_api_key": "SET" if os.environ.get("GEMINI_API_KEY") else "UNSET",
            "gemini_max_spend_usd": "SET" if os.environ.get("GEMINI_MAX_SPEND_USD") else "UNSET",
            "gemini_max_concurrency": os.environ.get("GEMINI_MAX_CONCURRENCY", "2"),
        },
    )
    (root / "commands.log").write_text(
        "\n".join(
            (
                "python -m failure_analysis.gemini_crog_evidence_v1.cli phase0 --output <run>/phase0",
                "python -m failure_analysis.gemini_crog_evidence_v1.cli select-smoke --output <run>/smoke_selection --count 10",
                "python -m failure_analysis.gemini_crog_evidence_v1.cli export-evidence --split train --selected-local-ids <ids> --output <run>/smoke_evidence_clipped --device mps --keep-dense-maps",
                "python -m failure_analysis.gemini_crog_evidence_v1.cli run-smoke --request-manifest <manifest> --output <run>/smoke_api",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    bundle = {
        "status": "offline_complete_api_blocked",
        "phase0": phase0,
        "evidence_export": evidence,
        "api_smoke": {
            "status": smoke["status"],
            "prepared_request_count": smoke["prepared_request_count"],
            "live_api_request_count": smoke["live_api_request_count"],
            "valid_count": smoke["valid_count"],
        },
        "baseline_methods": method_rows,
        "validation": {"status": "not_run", "reason": "missing API key and budget"},
        "formal_test": {"status": "not_run", "reason": "no validation-selected primary and no experiment lock"},
        "frozen_manifest": {"status": "not_created", "reason": "locking before validation is forbidden"},
    }
    _write(root / "results_bundle.json", bundle)
    _write(root / "lock_status.json", bundle["frozen_manifest"])
    return bundle
