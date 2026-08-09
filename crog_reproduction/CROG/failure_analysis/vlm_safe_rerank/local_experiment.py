from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import binomtest
from failure_analysis.gemini_crog_evidence_v1.planner import load_annotation_query_types

from .calibration import (
    DualRiskCalibration,
    LogisticModel,
    choose_query_threshold,
    cross_fit_dual_risk,
    fit_query_gate,
    threshold_sweep,
)
from .dataset import load_feature_index, load_label_index
from .features import ordered_candidates, select_challengers, stored_local_feature_vector
from .policy import full_denominator_metrics
from .manifest import compute_candidate_q_identity
from .runner import (
    DEFAULT_COHORTS,
    TRAIN_CORRECTED,
    TRAIN_FEATURES,
    VALIDATION_CORRECTED,
    VALIDATION_FEATURES,
    VALIDATION_LEGACY,
    V2_ROOT,
    _atomic_json,
    _parquet,
)


SPLIT_MANIFEST = V2_ROOT / "split_manifest.json"
BENEFIT_THRESHOLD_GRID = np.asarray([
    .01, .02, .03, .04, .05, .075, .10, .15, .20, .30, .40,
    .50, .55, .60, .65, .70, .75, .80, .85, .90, .95,
])


def _cohort_calibration() -> list[dict[str, Any]]:
    rows = pq.read_table(DEFAULT_COHORTS).to_pylist()
    return [row for row in rows if row["partition"] == "calibration"]


def _split_index() -> dict[str, dict[str, Any]]:
    payload = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload["rows"]}


def _reliability(feature: Mapping[str, Any], candidate_id: str) -> tuple[float, bool]:
    candidate = {str(row["candidate_id"]): row for row in feature["candidates"]}[candidate_id]
    values = []
    for name in ("mask_consistency", "depth_mad_m", "width_compatibility", "clearance", "collision_proxy"):
        row = candidate.get("features", {}).get(name, {})
        values.append(float(row.get("reliability", 0.0) or 0.0))
    hard_valid = bool(
        0 <= float(candidate["cx"]) < 640 and 0 <= float(candidate["cy"]) < 480
        and float(candidate["width_px"]) > 0 and float(candidate["height_px"]) > 0
    )
    return float(np.mean(values)), hard_valid


def _build_pair_matrix(
    sample_rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, bool]],
) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    records: list[dict[str, Any]] = []; vectors = []; feature_names: list[str] | None = None
    for sample in sample_rows:
        sample_id = str(sample["sample_id"]); feature = features[sample_id]
        ordered = ordered_candidates(feature); baseline_id = str(ordered[0]["candidate_id"])
        for challenger_id in select_challengers(feature, maximum=2):
            names, vector = stored_local_feature_vector(feature, challenger_id)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise AssertionError("local feature schema changed between samples")
            reliability, hard_valid = _reliability(feature, challenger_id)
            records.append({
                "sample_id": sample_id, "challenger_id": challenger_id,
                "baseline_correct": bool(labels[sample_id][baseline_id]),
                "challenger_correct": bool(labels[sample_id][challenger_id]),
                "group_id": str(sample["group_id"]),
                "query_type": str(sample.get("query_type", "unknown")),
                "evidence_reliable": reliability >= 0.5,
                "evidence_reliability": reliability,
                "challenger_hard_valid": hard_valid,
                "confirmation_stable": True,
                "hard_veto": False,
            })
            vectors.append(vector)
    return records, np.vstack(vectors), list(feature_names or [])


def _bundle_json(bundle: DualRiskCalibration) -> dict[str, Any]:
    return {
        "benefit_model": asdict(bundle.benefit_model), "harm_model": asdict(bundle.harm_model),
        "benefit_platt": asdict(bundle.benefit_platt), "harm_platt": asdict(bundle.harm_platt),
        "n_splits": bundle.n_splits, "seed": bundle.seed,
        "calibration_partition": bundle.calibration_partition,
    }


def _model(value: Mapping[str, Any]) -> LogisticModel:
    return LogisticModel(**value)


def _bundle(value: Mapping[str, Any]) -> DualRiskCalibration:
    return DualRiskCalibration(
        _model(value["benefit_model"]), _model(value["harm_model"]),
        _model(value["benefit_platt"]), _model(value["harm_platt"]),
        int(value["n_splits"]), int(value["seed"]), str(value["calibration_partition"]),
    )


def calibrate_local_only(run_dir: str | Path, *, seed: int = 20260803) -> dict[str, Any]:
    root = Path(run_dir); output = root / "calibration"
    samples = _cohort_calibration(); ids = {str(row["sample_id"]) for row in samples}
    features = load_feature_index(TRAIN_FEATURES, ids); labels = load_label_index(TRAIN_CORRECTED, ids)
    records, matrix, names = _build_pair_matrix(samples, features, labels)
    groups = [str(row["group_id"]) for row in records]
    beneficial = np.asarray([not row["baseline_correct"] and row["challenger_correct"] for row in records], dtype=float)
    harmful = np.asarray([row["baseline_correct"] and not row["challenger_correct"] for row in records], dtype=float)
    bundle, p_benefit, p_harm, folds = cross_fit_dual_risk(
        matrix, beneficial, harmful, groups, feature_names=names, n_splits=5, seed=seed,
    )
    # Query gate sees one deterministic vector per sample, never labels at inference.
    first_indices = np.arange(0, len(records), 2)
    query_matrix = matrix[first_indices]
    query_names = names
    query_labels = np.asarray([str(row["corrected_cohort"]) == "recoverable_error" for row in samples], dtype=float)
    query_groups = [str(row["group_id"]) for row in samples]
    query_model, query_oof, query_folds = fit_query_gate(
        query_matrix, query_labels, query_groups, feature_names=query_names, n_splits=5, seed=seed,
    )
    query_selection = choose_query_threshold(query_oof, query_labels, minimum_recall=0.90)
    sample_query = {str(row["sample_id"]): float(score) for row, score in zip(samples, query_oof, strict=True)}
    for index, row in enumerate(records):
        row["p_benefit"] = float(p_benefit[index]); row["p_harm"] = float(p_harm[index])
        row["oof_fold"] = int(folds[index]); row["query_score"] = sample_query[row["sample_id"]]
        row["query_fold"] = int(query_folds[index // 2])
    sweep, selected = threshold_sweep(
        records,
        tau_grid=BENEFIT_THRESHOLD_GRID,
        eta_grid=np.round(np.arange(0.0, 0.251, 0.025), 3),
        query_grid=(float(query_selection["threshold"]),),
    )
    state = "eligible" if selected is not None else "no_beneficial_switch"
    thresholds = selected or {
        "query_threshold": float(query_selection["threshold"]), "tau": 1.0, "eta": 0.0,
        "recovered": 0, "harmful": 0, "net": 0, "switches": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    _parquet(output / "local_only_oof_pairs.parquet", records)
    _parquet(output / "local_only_threshold_sweep.parquet", sweep)
    model_payload = {
        "schema_version": "1.0.0", "method": "P6_local_only_safe_gate",
        "dual_risk": _bundle_json(bundle), "query_model": asdict(query_model),
        "feature_names": names, "thresholds": thresholds,
        "query_selection": query_selection, "threshold_state": state,
        "calibration_samples": len(samples), "calibration_pairs": len(records),
    }
    _atomic_json(output / "local_only_model.json", model_payload)
    summary = {
        "method": "P6_local_only_safe_gate", "samples": len(samples), "pairs": len(records),
        "beneficial_pairs": int(beneficial.sum()), "harmful_pairs": int(harmful.sum()),
        "query_gate": query_selection, "threshold_state": state, "selected_thresholds": thresholds,
    }
    _atomic_json(output / "CALIBRATION_RESULTS.json", summary)
    return summary


def _validation_samples() -> list[dict[str, Any]]:
    split = _split_index(); rows = []
    annotations = load_annotation_query_types(
        V2_ROOT.parents[3] / "OCID-VLG/refer/multiple/val_expressions.json"
    )
    for feature in load_feature_index(VALIDATION_FEATURES).values():
        sample_id = f"multiple:val:{int(feature.get('sample_id', feature.get('sample_index'))):08d}"
        split_row = split[sample_id]
        rows.append({
            "sample_id": sample_id, "group_id": str(split_row["sequence_id"]),
            "frame_id": str(split_row["frame_id"]), "corrected_cohort": "unknown",
            "query_type": str(annotations[int(split_row["source_sample_id"])]),
        })
    return rows


def _bootstrap_delta(rows: Sequence[Mapping[str, Any]], *, draws: int = 10000, seed: int = 20260803) -> dict[str, float]:
    # Exact protected fallback: when every saved decision equals q-only, every
    # possible clustered resample has delta zero.  Materialising 10,000 copies
    # of an 8,669-row denominator would be wasteful and cannot change the CI.
    if all(
        bool(row["selected_correct"]) == bool(row["baseline_correct"])
        for row in rows
    ):
        return {"lower": 0.0, "median": 0.0, "upper": 0.0, "draws": draws}
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    names = sorted(groups); rng = np.random.default_rng(seed); deltas = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(names, size=len(names), replace=True)
        chosen = [row for name in sampled for row in groups[str(name)]]
        deltas[draw] = np.mean([float(row["selected_correct"]) - float(row["baseline_correct"]) for row in chosen])
    return {"lower": float(np.quantile(deltas, .025)), "median": float(np.quantile(deltas, .5)), "upper": float(np.quantile(deltas, .975)), "draws": draws}


def validate_local_only(run_dir: str | Path, *, seed: int = 20260803) -> dict[str, Any]:
    root = Path(run_dir); output = root / "validation"
    model_payload = json.loads((root / "calibration/local_only_model.json").read_text(encoding="utf-8"))
    dual = _bundle(model_payload["dual_risk"]); query_model = _model(model_payload["query_model"])
    thresholds = model_payload["thresholds"]
    samples = _validation_samples(); ids = {str(row["sample_id"]) for row in samples}
    features = load_feature_index(VALIDATION_FEATURES, ids); labels = load_label_index(VALIDATION_CORRECTED, ids)
    legacy_labels = load_label_index(VALIDATION_LEGACY, ids)
    for sample in samples:
        sample_id = str(sample["sample_id"])
        baseline_id = str(ordered_candidates(features[sample_id])[0]["candidate_id"])
        baseline_correct = bool(labels[sample_id][baseline_id])
        sample["corrected_cohort"] = (
            "protected_correct" if baseline_correct else
            "recoverable_error" if any(labels[sample_id].values()) else
            "unrecoverable_error"
        )
    records, matrix, names = _build_pair_matrix(samples, features, labels)
    if names != list(model_payload["feature_names"]):
        raise AssertionError("validation feature schema differs from calibration")
    p_benefit, p_harm = dual.predict(matrix)
    query_scores = query_model.predict_proba(matrix[np.arange(0, len(records), 2)])
    sample_query = {str(row["sample_id"]): float(score) for row, score in zip(samples, query_scores, strict=True)}
    for index, row in enumerate(records):
        row["p_benefit"] = float(p_benefit[index]); row["p_harm"] = float(p_harm[index]); row["query_score"] = sample_query[row["sample_id"]]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_sample.setdefault(row["sample_id"], []).append(row)
    decisions = []
    for sample in samples:
        sample_id = str(sample["sample_id"]); pairs = by_sample[sample_id]
        eligible = [row for row in pairs if row["query_score"] >= float(thresholds["query_threshold"])
                    and row["p_benefit"] >= float(thresholds["tau"]) and row["p_harm"] <= float(thresholds["eta"])
                    and row["evidence_reliable"] and row["challenger_hard_valid"]]
        chosen = sorted(eligible, key=lambda row: (-row["p_benefit"], row["p_harm"], row["challenger_id"]))[0] if eligible else None
        baseline_correct = bool(pairs[0]["baseline_correct"])
        selected_correct = bool(chosen["challenger_correct"]) if chosen else baseline_correct
        baseline_id = str(ordered_candidates(features[sample_id])[0]["candidate_id"])
        selected_id = baseline_id if chosen is None else str(chosen["challenger_id"])
        legacy_baseline_correct = bool(legacy_labels[sample_id][baseline_id])
        legacy_selected_correct = bool(legacy_labels[sample_id][selected_id])
        decisions.append({
            "sample_id": sample_id, "group_id": sample["group_id"], "frame_id": sample["frame_id"],
            "query_type": sample["query_type"], "cohort": sample["corrected_cohort"],
            "baseline_candidate_id": baseline_id, "selected_candidate_id": selected_id,
            "baseline_correct": baseline_correct, "selected_correct": selected_correct,
            "legacy_baseline_correct": legacy_baseline_correct,
            "legacy_selected_correct": legacy_selected_correct,
            "switched": chosen is not None, "selected_challenger_id": None if chosen is None else chosen["challenger_id"],
            "p_benefit": None if chosen is None else chosen["p_benefit"], "p_harm": None if chosen is None else chosen["p_harm"],
            "query_score": sample_query[sample_id],
        })
    metrics = full_denominator_metrics(decisions, baseline_key="baseline_correct", selected_key="selected_correct")
    legacy_metrics = full_denominator_metrics(
        decisions, baseline_key="legacy_baseline_correct", selected_key="legacy_selected_correct"
    )
    corrected_recoverable = sum(row["cohort"] == "recoverable_error" for row in decisions)
    metrics["oracle_at_5_successes"] = int(metrics["baseline_successes"]) + corrected_recoverable
    metrics["oracle_at_5"] = metrics["oracle_at_5_successes"] / len(decisions)
    metrics["recovery_recall"] = metrics["recovered"] / corrected_recoverable if corrected_recoverable else 0.0
    legacy_recoverable = 0
    for sample_id in ids:
        baseline_id = str(ordered_candidates(features[sample_id])[0]["candidate_id"])
        if not legacy_labels[sample_id][baseline_id] and any(legacy_labels[sample_id].values()):
            legacy_recoverable += 1
    legacy_metrics["oracle_at_5_successes"] = int(legacy_metrics["baseline_successes"]) + legacy_recoverable
    legacy_metrics["oracle_at_5"] = legacy_metrics["oracle_at_5_successes"] / len(decisions)
    legacy_metrics["recovery_recall"] = legacy_metrics["recovered"] / legacy_recoverable if legacy_recoverable else 0.0
    ci = _bootstrap_delta(decisions, draws=10000, seed=seed)
    mcnemar = binomtest(int(metrics["recovered"]), int(metrics["recovered"] + metrics["harmful"]), p=.5, alternative="two-sided").pvalue if metrics["recovered"] + metrics["harmful"] else 1.0
    precision = float(metrics["outcome_changing_precision"]); harm_rate = float(metrics["harm_rate"])
    go = bool(
        metrics["final_j1"] >= metrics["baseline_j1"] and metrics["recovered"] > metrics["harmful"]
        and metrics["net"] > 0 and precision >= .67 and harm_rate <= .01 and ci["lower"] >= 0.0
    )
    status = "GO" if go else ("INCONCLUSIVE" if metrics["net"] > 0 and ci["lower"] < 0 else "NO_GO")
    stratified = {}
    for field in ("query_type", "cohort"):
        stratified[field] = {}
        for value in sorted({str(row[field]) for row in decisions}):
            subset = [row for row in decisions if str(row[field]) == value]
            stratified[field][value] = full_denominator_metrics(
                subset, baseline_key="baseline_correct", selected_key="selected_correct"
            )
    output.mkdir(parents=True, exist_ok=True)
    _parquet(output / "local_only_pair_scores.parquet", records)
    _parquet(output / "local_only_decisions.parquet", decisions)
    result = {
        "schema_version": "1.0.0", "validation_status": status,
        "primary_method": "P6_local_only_safe_gate" if go else "q_only",
        "expected_denominator": len(decisions), "corrected": metrics,
        "legacy": legacy_metrics,
        "stratified_corrected": stratified,
        "scene_sequence_bootstrap_delta_j1": ci, "exact_mcnemar_p": float(mcnemar),
        "thresholds_from_calibration": thresholds, "no_validation_tuning": True,
        "query_gate_call_count": sum(row["query_score"] >= float(thresholds["query_threshold"]) for row in decisions),
        "query_gate_call_rate": sum(row["query_score"] >= float(thresholds["query_threshold"]) for row in decisions) / len(decisions),
    }
    validation_identity = compute_candidate_q_identity(VALIDATION_FEATURES)
    result.update({
        "candidate_identity_sha256": validation_identity["candidate_identity_sha256"],
        "q_value_sha256": validation_identity["q_value_sha256"],
        "combined_identity_sha256": validation_identity["combined_identity_sha256"],
    })
    _atomic_json(output / "VALIDATION_RESULTS.json", result)
    return result
