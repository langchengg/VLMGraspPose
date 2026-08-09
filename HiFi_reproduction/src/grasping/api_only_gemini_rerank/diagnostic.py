"""Balanced evidence ablation and display/order/colour stability diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io import atomic_json
from .metrics import outcome_metrics
from .stages import assert_stage_result_coverage


def evaluate_diagnostic(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    coverage = assert_stage_result_coverage(run, "diagnostic")
    results = pd.read_parquet(run / "stage_results/diagnostic.parquet")
    baseline = pd.read_parquet(run / "baseline_per_sample.parquet")
    metric_rows = []
    selected: dict[str, Any] = {}
    for (backend, model, variant), group in results.loc[results["replicate_id"].eq(1)].groupby(["backend", "model_id", "evidence_variant"]):
        sample_ids = set(group["sample_id"].astype(str))
        source = baseline.loc[(baseline["backend"] == backend) & baseline["sample_id"].astype(str).isin(sample_ids)]
        by_sample = {str(row["sample_id"]): row for row in group.to_dict(orient="records")}
        rows = []
        for item in source.to_dict(orient="records"):
            response = by_sample.get(str(item["sample_id"]))
            original = item.get("top1_candidate_id")
            final = original if response is None or response.get("status") != "SUCCEEDED" else response.get("selected_candidate_id")
            correctness = json.loads(str(item["candidate_correctness_json"]))
            rows.append({
                "scene_id": item["scene_id"], "baseline_candidate_id": original,
                "final_candidate_id": final, "baseline_correct": item["top1_correct"],
                "final_correct": bool(correctness.get(str(final), False)) if final is not None else False,
                "recoverable_error": item["recoverable_error"],
            })
        metrics = outcome_metrics(rows)
        metric_rows.append({"backend": backend, "model_id": model, "evidence_variant": variant, **metrics})
    metrics_frame = pd.DataFrame(metric_rows)
    stability_rows = []
    perturbed = results.loc[results["evidence_variant"].eq("E3_RGBD_GEOMETRY_SCORE_AWARE")]
    for (backend, model, sample_id), group in perturbed.groupby(["backend", "model_id", "sample_id"]):
        if group["replicate_id"].nunique() < 3:
            continue
        terminal = group.sort_values("replicate_id")
        selected_ids = terminal["selected_candidate_id"].astype(str).tolist()
        decisions = terminal["decision"].astype(str).tolist()
        stability_rows.append({
            "backend": backend, "model_id": model, "sample_id": sample_id,
            "top1_stable": len(set(selected_ids)) == 1,
            "decision_stable": len(set(decisions)) == 1,
            "selected_ids": json.dumps(selected_ids),
        })
    stability = pd.DataFrame(stability_rows)
    if len(stability):
        stability.to_csv(run / "diagnostic_stability.csv", index=False)
    for (backend, model), group in metrics_frame.groupby(["backend", "model_id"]):
        ordered = group.sort_values(["net", "harmful", "switch_rate"], ascending=[False, True, True])
        best = ordered.iloc[0].to_dict()
        key = f"{backend}:{model}"
        selected[key] = {
            "evidence_variant": best["evidence_variant"],
            "balanced_diagnostic_only": True,
            "net": int(best["net"]), "harmful": int(best["harmful"]),
            "selection_rule": "max balanced diagnostic net; then fewer harmful; then lower switch rate",
        }
    metrics_frame.to_csv(run / "diagnostic_evidence_ablation.csv", index=False)
    per_model_stability = {}
    if len(stability):
        for (backend, model), group in stability.groupby(["backend", "model_id"]):
            per_model_stability[f"{backend}:{model}"] = {
                "samples": len(group), "top1_stability": float(group["top1_stable"].mean()),
                "decision_stability": float(group["decision_stable"].mean()),
                "passes_preregistered_minimum": float(group["top1_stable"].mean()) >= 0.90,
                "preregistered_minimum": 0.90,
            }
    payload = {
        "metrics": metric_rows, "stability_sample_count": len(stability),
        "top1_stability": None if not len(stability) else float(stability["top1_stable"].mean()),
        "decision_stability": None if not len(stability) else float(stability["decision_stable"].mean()),
        "per_backend_model_stability": per_model_stability,
        "selected_evidence": selected,
        "warning": "balanced diagnostic metrics are not natural-distribution J@1 estimates",
        "request_coverage": coverage,
    }
    atomic_json(run / "DIAGNOSTIC_REPORT.json", payload)
    atomic_json(run / "EVIDENCE_SELECTION.json", selected)
    return payload
