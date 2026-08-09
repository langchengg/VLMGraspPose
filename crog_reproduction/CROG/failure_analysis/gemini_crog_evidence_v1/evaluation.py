from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .security import assert_candidate_identity
from .statistics import clustered_bootstrap_delta, exact_mcnemar_pvalue, holm_adjust


def _jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield value


def stable_candidate_id(sample_id: str, candidate_id: str) -> str:
    return f"{sample_id}/{candidate_id}"


def _raw_candidate_id(sample_id: str, selected_stable_candidate_id: str) -> str:
    prefix = f"{sample_id}/"
    if not str(selected_stable_candidate_id).startswith(prefix):
        raise ValueError("selected stable candidate ID does not belong to the sample")
    value = str(selected_stable_candidate_id)[len(prefix) :]
    if not value or "/" in value:
        raise ValueError("invalid selected stable candidate ID")
    return value


def load_evaluation_index(
    *,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    corrected_labels_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Join frozen candidates to both evaluator tracks without request data."""

    index: dict[str, dict[str, Any]] = {}
    for feature, legacy, corrected in zip(
        _jsonl(features_path),
        _jsonl(legacy_labels_path),
        _jsonl(corrected_labels_path),
        strict=True,
    ):
        sample_id = str(legacy["sample_id"])
        if sample_id != str(corrected["sample_id"]):
            raise ValueError("Legacy/Corrected sample identity mismatch")
        candidates = feature["candidates"]
        assert_candidate_identity(candidates)
        candidate_ids = [str(item["candidate_id"]) for item in candidates]
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item["q_raw"]), str(item["candidate_id"])),
        )
        q_only_id = str(ordered[0]["candidate_id"])
        legacy_by_id = {
            str(item["candidate_id"]): bool(item["candidate_correct"])
            for item in legacy["candidate_labels"]
        }
        corrected_by_id = {
            str(item["candidate_id"]): bool(item["candidate_correct"])
            for item in corrected["candidate_labels"]
        }
        if set(candidate_ids) != set(legacy_by_id) or set(candidate_ids) != set(corrected_by_id):
            raise ValueError(f"evaluation label candidate mismatch for {sample_id}")
        rank_by_id = {str(item["candidate_id"]): rank + 1 for rank, item in enumerate(ordered)}
        index[sample_id] = {
            "sample_id": sample_id,
            "frame_id": str(feature.get("frame_id", feature.get("scene_id", legacy.get("frame_id", sample_id)))),
            "scene_id": str(feature.get("scene_id", legacy.get("scene_id", sample_id))),
            "candidate_ids": candidate_ids,
            "q_only_candidate_id": q_only_id,
            "rank_by_candidate_id": rank_by_id,
            "legacy_by_candidate_id": legacy_by_id,
            "corrected_by_candidate_id": corrected_by_id,
            "legacy_oracle": any(legacy_by_id.values()),
            "corrected_oracle": any(corrected_by_id.values()),
        }
    return index


def evaluate_saved_predictions(
    *,
    evaluation_index: dict[str, dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate only saved frozen IDs and return outcomes plus method metrics."""

    outcomes: list[dict[str, Any]] = []
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        method = str(prediction["method"])
        key = (sample_id, method)
        if key in seen:
            raise ValueError(f"duplicate prediction for {sample_id}/{method}")
        seen.add(key)
        truth = evaluation_index.get(sample_id)
        if truth is None:
            raise ValueError(f"prediction sample is absent from evaluator index: {sample_id}")
        selected_id = _raw_candidate_id(sample_id, str(prediction["selected_stable_candidate_id"]))
        if selected_id not in truth["candidate_ids"]:
            raise ValueError("prediction selected an ID outside the frozen Top-5")
        q_id = truth["q_only_candidate_id"]
        status = str(prediction.get("status", "valid"))
        switched = selected_id != q_id
        technical_fallback = bool(prediction.get("technical_fallback", status == "technical_fallback"))
        permanent_api_failure = bool(
            prediction.get("permanent_api_failure", status == "permanent_failed")
        )
        abstain = bool(prediction.get("abstain", status == "abstain"))
        active_model_decision = not (technical_fallback or permanent_api_failure or abstain)
        row = {
            "sample_id": sample_id,
            "frame_id": truth["frame_id"],
            "scene_id": truth["scene_id"],
            "method": method,
            "selected_stable_candidate_id": stable_candidate_id(sample_id, selected_id),
            "q_only_stable_candidate_id": stable_candidate_id(sample_id, q_id),
            "status": status,
            "valid_response": bool(prediction.get("valid_response", status in {"valid", "abstain"})),
            "technical_fallback": technical_fallback,
            "permanent_api_failure": permanent_api_failure,
            "abstain": abstain,
            "fallback_to_q_only": not active_model_decision,
            "keep": active_model_decision and not switched,
            "switch": active_model_decision and switched,
            "selected_original_rank": int(truth["rank_by_candidate_id"][selected_id]),
            "selected_score_margin": prediction.get("selected_score_margin"),
            "legacy_q_only_correct": bool(truth["legacy_by_candidate_id"][q_id]),
            "legacy_selected_correct": bool(truth["legacy_by_candidate_id"][selected_id]),
            "legacy_oracle": bool(truth["legacy_oracle"]),
            "corrected_q_only_correct": bool(truth["corrected_by_candidate_id"][q_id]),
            "corrected_selected_correct": bool(truth["corrected_by_candidate_id"][selected_id]),
            "corrected_oracle": bool(truth["corrected_oracle"]),
        }
        outcomes.append(row)
        methods[method].append(row)

    metrics: list[dict[str, Any]] = []
    for method, rows in sorted(methods.items()):
        metric: dict[str, Any] = {
            "method": method,
            "total": len(rows),
            "valid_response_count": sum(bool(row["valid_response"]) for row in rows),
            "technical_fallback": sum(bool(row["technical_fallback"]) for row in rows),
            "permanent_api_failure": sum(bool(row["permanent_api_failure"]) for row in rows),
            "abstain": sum(bool(row["abstain"]) for row in rows),
            "fallback_to_q_only": sum(bool(row["fallback_to_q_only"]) for row in rows),
            "keep": sum(bool(row["keep"]) for row in rows),
            "switch": sum(bool(row["switch"]) for row in rows),
            "mean_original_rank_of_selected_candidate": float(
                np.mean([row["selected_original_rank"] for row in rows])
            ),
        }
        metric["q_copy_rate"] = metric["keep"] / len(rows)
        margins = [float(row["selected_score_margin"]) for row in rows if row["selected_score_margin"] is not None]
        metric["mean_selected_score_margin"] = float(np.mean(margins)) if margins else None
        for track in ("legacy", "corrected"):
            q = np.asarray([row[f"{track}_q_only_correct"] for row in rows], dtype=bool)
            selected = np.asarray([row[f"{track}_selected_correct"] for row in rows], dtype=bool)
            oracle = np.asarray([row[f"{track}_oracle"] for row in rows], dtype=bool)
            recovered = int((~q & selected).sum())
            harmful = int((q & ~selected).sum())
            net = recovered - harmful
            switches = int(sum(bool(row["switch"]) for row in rows))
            headroom = int((oracle & ~q).sum())
            metric.update(
                {
                    f"{track}_j1": float(selected.mean()),
                    f"{track}_delta_pp": 100.0 * net / len(rows),
                    f"{track}_recovered": recovered,
                    f"{track}_harmful": harmful,
                    f"{track}_net": net,
                    f"{track}_q_only_j1": float(q.mean()),
                    f"{track}_oracle_at_5": float(oracle.mean()),
                    f"{track}_recoverable_headroom_utilization": recovered / headroom if headroom else None,
                    f"{track}_outcome_changing_precision": recovered / (recovered + harmful)
                    if recovered + harmful
                    else None,
                    f"{track}_harmful_rate": harmful / len(rows),
                    f"{track}_switch_coverage": switches / len(rows),
                }
            )
        metric["switch_coverage"] = metric["legacy_switch_coverage"]
        metric["outcome_precision"] = metric["legacy_outcome_changing_precision"]
        metrics.append(metric)
    return outcomes, metrics


def paired_statistical_tests(
    *,
    outcomes: Iterable[dict[str, Any]],
    draws: int = 10_000,
    seed: int = 47,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_method[str(row["method"])].append(row)
    raw_p: dict[str, float] = {}
    bootstrap: dict[str, Any] = {}
    excluded_methods = {"crog_q_only", "locked_gemini_primary"}
    for method, rows in sorted(by_method.items()):
        if method in excluded_methods:
            continue
        rows = sorted(rows, key=lambda item: item["sample_id"])
        for track in ("legacy", "corrected"):
            baseline = [bool(row[f"{track}_q_only_correct"]) for row in rows]
            selected = [bool(row[f"{track}_selected_correct"]) for row in rows]
            name = f"{method}:{track}"
            raw_p[name] = exact_mcnemar_pvalue(baseline, selected)
            bootstrap[name] = {
                "frame": clustered_bootstrap_delta(
                    baseline=baseline,
                    method=selected,
                    groups=[str(row["frame_id"]) for row in rows],
                    draws=draws,
                    seed=seed,
                ),
                "scene": clustered_bootstrap_delta(
                    baseline=baseline,
                    method=selected,
                    groups=[str(row["scene_id"]) for row in rows],
                    draws=draws,
                    seed=seed,
                ),
            }
    adjusted = holm_adjust(raw_p)
    tests = {
        "test": "exact_mcnemar_two_sided",
        "multiple_comparison": "Holm",
        "family": "unique_nonbaseline_preregistered_methods",
        "excluded_duplicate_or_baseline_methods": sorted(excluded_methods & set(by_method)),
        "raw_p": raw_p,
        "holm_adjusted_p": adjusted,
    }
    return tests, bootstrap


def select_validation_primary(
    *,
    metrics: Iterable[dict[str, Any]],
    bootstrap: dict[str, Any],
    estimated_cost_by_method: dict[str, float],
    minimum_response_coverage: float = 0.98,
    harmful_rate_limit: float = 0.01,
    maximum_negative_scene_ci_pp: float = 1.0,
) -> dict[str, Any]:
    eligible: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    diagnostics = []
    for row in metrics:
        method = str(row["method"])
        if method == "crog_q_only":
            continue
        total = int(row["total"])
        coverage = float(row["valid_response_count"]) / total if total else 0.0
        scene = bootstrap.get(f"{method}:legacy", {}).get("scene", {})
        scene_low = float(scene.get("ci_low_pp", -math.inf))
        reasons = []
        if int(row["legacy_net"]) <= 0:
            reasons.append("legacy_net_not_positive")
        if float(row["legacy_harmful_rate"]) > harmful_rate_limit:
            reasons.append("legacy_harmful_cap_exceeded")
        if coverage < minimum_response_coverage:
            reasons.append("response_coverage_below_minimum")
        if scene_low < -float(maximum_negative_scene_ci_pp):
            reasons.append("scene_bootstrap_negative_risk")
        diagnostics.append({"method": method, "eligible": not reasons, "reasons": reasons})
        if reasons:
            continue
        precision = row["legacy_outcome_changing_precision"]
        corrected_net = int(row["corrected_net"])
        scene_width = float(scene.get("ci_high_pp", math.inf)) - scene_low
        cost = float(estimated_cost_by_method.get(method, math.inf))
        flash_safe_preference = 1.0 if method == "gemini_3_6_flash_crog_evidence_safe" else 0.0
        score = (
            float(row["legacy_net"]),
            float(precision if precision is not None else 0.0),
            float(corrected_net),
            -scene_width,
            -cost,
            flash_safe_preference,
        )
        eligible.append((score, row))
    if not eligible:
        return {
            "locked_primary": "crog_q_only",
            "status": "no_eligible_gemini_method",
            "diagnostics": diagnostics,
            "selection_rule_version": "validation-primary-v1",
        }
    selected = max(eligible, key=lambda item: item[0])[1]
    return {
        "locked_primary": str(selected["method"]),
        "status": "selected",
        "diagnostics": diagnostics,
        "selection_rule_version": "validation-primary-v1",
    }
