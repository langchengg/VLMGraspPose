"""Development-only deterministic policy sweep; no learned/local candidate score."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .decisions import baseline_aware, confidence_accept, direct, self_consistent
from .io import atomic_json, atomic_parquet, sha256_file, utc_now
from .metrics import outcome_metrics
from .stages import assert_stage_result_coverage


THRESHOLDS = (50, 60, 70, 80, 90, 95)


def _response(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None or row.get("status") != "SUCCEEDED":
        return None
    return {
        "selected_internal_candidate_id": row.get("selected_candidate_id"),
        "decision": row.get("decision"), "switch_confidence": row.get("switch_confidence"),
        "evidence_reliability": row.get("evidence_reliability"),
    }


def _evaluate(
    baseline: pd.DataFrame, responses: Mapping[str, Mapping[str, Any]], *,
    protocol: str, threshold: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = []
    for item in baseline.to_dict(orient="records"):
        sample_id, original = str(item["sample_id"]), item.get("top1_candidate_id")
        response = _response(responses.get(sample_id))
        if original is None:
            final = None
        elif protocol == "P1_API_DIRECT_FULL_LIST":
            final = direct(response, str(original))
        elif threshold is None:
            final = baseline_aware(response, str(original))
        else:
            final = confidence_accept(response, str(original), threshold)
        correctness = json.loads(str(item["candidate_correctness_json"]))
        output.append({
            "scene_id": str(item["scene_id"]), "sample_id": sample_id,
            "baseline_candidate_id": original, "final_candidate_id": final,
            "baseline_correct": bool(item["top1_correct"]),
            "final_correct": bool(correctness.get(str(final), False)) if final is not None else False,
            "recoverable_error": bool(item["recoverable_error"]),
        })
    return outcome_metrics(output), output


def _evaluate_self_consistent(
    baseline: pd.DataFrame, first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]], threshold: int,
) -> dict[str, Any]:
    rows = []
    for item in baseline.to_dict(orient="records"):
        sample_id, original = str(item["sample_id"]), item.get("top1_candidate_id")
        final = None if original is None else self_consistent(
            _response(first.get(sample_id)), _response(second.get(sample_id)), str(original), threshold,
        )
        correctness = json.loads(str(item["candidate_correctness_json"]))
        rows.append({
            "scene_id": str(item["scene_id"]), "sample_id": sample_id,
            "baseline_candidate_id": original, "final_candidate_id": final,
            "baseline_correct": bool(item["top1_correct"]),
            "final_correct": bool(correctness.get(str(final), False)) if final is not None else False,
            "recoverable_error": bool(item["recoverable_error"]),
        })
    return outcome_metrics(rows)


def build_policy_confirmation_manifest(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    base = pd.read_parquet(run / "stage_results/policy_selection.parquet")
    baseline = pd.read_parquet(run / "baseline_per_sample.parquet")[["backend", "sample_id", "top1_candidate_id"]]
    selected = base.loc[
        base["protocol"].eq("P2_API_BASELINE_AWARE")
        & base["status"].eq("SUCCEEDED")
        & base["decision"].eq("SELECT_CANDIDATE")
        & base["evidence_reliability"].eq("HIGH")
        & pd.to_numeric(base["switch_confidence"], errors="coerce").ge(min(THRESHOLDS))
    ].merge(baseline, on=["backend", "sample_id"], how="left", validate="many_to_one")
    selected = selected.loc[selected["selected_candidate_id"].astype(str) != selected["top1_candidate_id"].astype(str)].copy()
    selected["replicate_id"] = 2
    selected["perturbation"] = "p4_display_panel_colour_permutation"
    keep = ["stage", "backend", "sample_id", "scene_id", "candidate_count",
            "model_id", "protocol", "evidence_variant", "perturbation", "replicate_id"]
    path = run / "request_manifests/policy_confirmation.parquet"
    atomic_parquet(path, selected[keep])
    payload = {"rows": len(selected), "sha256": sha256_file(path), "created_at_utc": utc_now(),
               "rule": "P2 successful HIGH-reliability non-baseline switch at confidence >= 50"}
    atomic_json(run / "request_manifests/policy_confirmation.json", payload)
    return payload


def sweep_policy_selection(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    evidence_selection = json.loads((run / "EVIDENCE_SELECTION.json").read_text())
    diagnostic = json.loads((run / "DIAGNOSTIC_REPORT.json").read_text())
    policy_coverage = assert_stage_result_coverage(run, "policy_selection")
    results_path = run / "stage_results/policy_selection.parquet"
    if not results_path.is_file():
        raise FileNotFoundError("policy provider results are absent")
    results = pd.read_parquet(results_path)
    if len(results) == 0 or not results["status"].isin(["SUCCEEDED", "PERMANENT_FAILED", "SCHEMA_FAILED", "TECHNICAL_FALLBACK"]).all():
        raise RuntimeError("policy request manifest is not terminal")
    cohorts = pd.read_parquet(run / "STAGE_COHORTS.parquet")
    baseline_all = pd.read_parquet(run / "baseline_per_sample.parquet")
    rows = []
    selections: dict[str, Any] = {}
    for backend in ("G1", "C1"):
        sample_ids = set(cohorts.loc[(cohorts["stage"] == "policy_selection") & (cohorts["backend"] == backend), "sample_id"].astype(str))
        baseline = baseline_all.loc[(baseline_all["backend"] == backend) & (baseline_all["split"] == "validation") & baseline_all["sample_id"].astype(str).isin(sample_ids)]
        recoverable_prevalence = float(baseline["recoverable_error"].mean())
        for model in sorted(results.loc[results["backend"] == backend, "model_id"].unique()):
            model_rows = results.loc[(results["backend"] == backend) & (results["model_id"] == model)]
            stability = diagnostic["per_backend_model_stability"].get(f"{backend}:{model}", {})
            stability_pass = bool(stability.get("passes_preregistered_minimum", False))
            methods = [("P1_API_DIRECT_FULL_LIST", None), ("P2_API_BASELINE_AWARE", None)] + [("P3_API_BASELINE_AWARE_CONFIDENCE", threshold) for threshold in THRESHOLDS]
            candidates = []
            for protocol, threshold in methods:
                source_protocol = "P2_API_BASELINE_AWARE" if protocol.startswith("P3") else protocol
                source_rows = model_rows.loc[model_rows["protocol"] == source_protocol]
                source_latency = pd.to_numeric(
                    source_rows.get("latency_seconds", pd.Series(index=source_rows.index, dtype=float)), errors="coerce"
                ).dropna()
                source_cost = pd.to_numeric(
                    source_rows.get("estimated_cost_usd", pd.Series(index=source_rows.index, dtype=float)), errors="coerce"
                ).fillna(0)
                indexed = {str(row["sample_id"]): row for row in source_rows.to_dict(orient="records")}
                metrics, _ = _evaluate(baseline, indexed, protocol=protocol if not protocol.startswith("P3") else "P2_API_BASELINE_AWARE", threshold=threshold)
                value = {"backend": backend, "model_id": model, "protocol": protocol,
                         "threshold": threshold, "recoverable_prevalence": recoverable_prevalence,
                         "selection_api_requests": len(source_rows),
                         "selection_p95_latency_seconds": None if source_latency.empty else float(source_latency.quantile(.95)),
                         "selection_estimated_cost_usd": float(source_cost.sum()),
                         "stability_pass": stability_pass,
                         "top1_stability": stability.get("top1_stability"),
                         "preregistered_stability_minimum": stability.get("preregistered_minimum", 0.90),
                         **metrics}
                value["eligible"] = bool(
                    metrics["recovered"] > metrics["harmful"]
                    and metrics["final_j_at_1"] >= metrics["baseline_j_at_1"]
                    and metrics["harm_rate"] <= 0.01
                    and metrics["outcome_changing_precision"] is not None
                    and metrics["outcome_changing_precision"] >= 0.67
                    and metrics["switch_rate"] <= max(0.01, 1.5*recoverable_prevalence)
                    and stability_pass
                )
                rows.append(value); candidates.append(value)
            confirmation_path = run / "stage_results/policy_confirmation.parquet"
            if confirmation_path.is_file():
                confirmation = pd.read_parquet(confirmation_path)
                confirmation = confirmation.loc[(confirmation["backend"] == backend) & (confirmation["model_id"] == model)]
                first = {str(row["sample_id"]): row for row in model_rows.loc[model_rows["protocol"] == "P2_API_BASELINE_AWARE"].to_dict(orient="records")}
                second = {str(row["sample_id"]): row for row in confirmation.to_dict(orient="records")}
                for threshold in THRESHOLDS:
                    metrics = _evaluate_self_consistent(baseline, first, second, threshold)
                    first_rows = model_rows.loc[model_rows["protocol"] == "P2_API_BASELINE_AWARE"]
                    latency = pd.concat([
                        pd.to_numeric(first_rows.get("latency_seconds", pd.Series(index=first_rows.index, dtype=float)), errors="coerce"),
                        pd.to_numeric(confirmation.get("latency_seconds", pd.Series(index=confirmation.index, dtype=float)), errors="coerce"),
                    ], ignore_index=True).dropna()
                    value = {"backend": backend, "model_id": model, "protocol": "P4_API_SELF_CONSISTENT",
                             "threshold": threshold, "recoverable_prevalence": recoverable_prevalence,
                             "selection_api_requests": len(first) + len(confirmation),
                             "selection_p95_latency_seconds": None if latency.empty else float(latency.quantile(.95)),
                             "selection_estimated_cost_usd": float(
                                 pd.to_numeric(first_rows.get("estimated_cost_usd", pd.Series(index=first_rows.index, dtype=float)), errors="coerce").fillna(0).sum()
                                 + pd.to_numeric(confirmation.get("estimated_cost_usd", pd.Series(index=confirmation.index, dtype=float)), errors="coerce").fillna(0).sum()
                             ),
                             "stability_pass": stability_pass,
                             "top1_stability": stability.get("top1_stability"),
                             "preregistered_stability_minimum": stability.get("preregistered_minimum", 0.90),
                             **metrics}
                    value["eligible"] = bool(
                        metrics["recovered"] > metrics["harmful"]
                        and metrics["final_j_at_1"] >= metrics["baseline_j_at_1"]
                        and metrics["harm_rate"] <= 0.01
                        and metrics["outcome_changing_precision"] is not None
                        and metrics["outcome_changing_precision"] >= 0.67
                        and metrics["switch_rate"] <= max(0.01, 1.5*recoverable_prevalence)
                        and stability_pass
                    )
                    rows.append(value); candidates.append(value)
            eligible = [row for row in candidates if row["eligible"]]
            key = f"{backend}:{model}"
            if not eligible:
                reasons = []
                if not stability_pass:
                    reasons.append("STABILITY_BELOW_PREREGISTERED_MINIMUM")
                if not any(
                    row["recovered"] > row["harmful"]
                    and row["final_j_at_1"] >= row["baseline_j_at_1"]
                    and row["harm_rate"] <= 0.01
                    and row["outcome_changing_precision"] is not None
                    and row["outcome_changing_precision"] >= 0.67
                    and row["switch_rate"] <= max(0.01, 1.5*recoverable_prevalence)
                    for row in candidates
                ):
                    reasons.append("NO_POLICY_MET_EFFECT_AND_HARM_GATES")
                selections[key] = {
                    "status": "NO_ELIGIBLE_DEVELOPMENT_POLICY",
                    "protocol": "P0_ORIGINAL_BACKEND_SCORE",
                    "ineligibility_reasons": reasons,
                    "top1_stability": stability.get("top1_stability"),
                    "preregistered_stability_minimum": stability.get("preregistered_minimum", 0.90),
                }
            else:
                eligible.sort(key=lambda row: (
                    row["harmful"], -row["net"], row["switch_rate"],
                    row["selection_api_requests"], float("inf") if row["selection_p95_latency_seconds"] is None else row["selection_p95_latency_seconds"],
                    row["selection_estimated_cost_usd"], row["protocol"],
                    -1 if row["threshold"] is None else int(row["threshold"]),
                ))
                selections[key] = {**eligible[0], "status": "PRESELECTED_REQUIRES_P4_CONFIRMATION_AND_UNTOUCHED_VALIDATION",
                                   "evidence_variant": str(evidence_selection[key]["evidence_variant"])}
    pd.DataFrame(rows).to_csv(run / "policy_threshold_sweep.csv", index=False)
    confirmation_coverage = None
    if (run / "request_manifests/policy_confirmation.parquet").is_file():
        confirmation_coverage = assert_stage_result_coverage(run, "policy_confirmation")
    payload = {"thresholds": list(THRESHOLDS), "selections": selections,
               "selection_data": "policy_selection only", "untouched_validation_used": False,
               "request_coverage": {
                   "policy_selection": policy_coverage,
                   "policy_confirmation": confirmation_coverage,
               }}
    atomic_json(run / "POLICY_PRESELECTION.json", payload)
    return payload
