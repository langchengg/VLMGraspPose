from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq

from .calibration import cross_fit_dual_risk, threshold_sweep
from .critic_features import critic_feature_vector
from .critic_evaluation import STABILITY_CONTRACT
from .dataset import cohort_rows, load_feature_index
from .features import build_pair_evidence, stored_local_feature_vector
from .runner import (
    DEFAULT_COHORTS,
    TRAIN_FEATURES,
    _atomic_json,
    _parquet,
    assert_phase_inference_manifest_frozen,
)
from .local_experiment import BENEFIT_THRESHOLD_GRID
from .manifest import file_identity


MODELS = ("gemini-robotics-er-2-preview", "gemini-3.6-flash")
METHOD_MODELS = {
    "P5_er2_safe": (MODELS[0],),
    "P5_flash_safe": (MODELS[1],),
    "P5_joint_safe": MODELS,
}
CALIBRATION_COHORT_COUNTS = {
    "protected_correct": 9148,
    "recoverable_error": 359,
    "unrecoverable_error": 283,
}
API_CALIBRATION_PHASE = "api_calibration_natural_v2"
API_CALIBRATION_CONFIRMATION_PHASE = "api_calibration_natural_v2_confirmation"
QUERY_GATE_QUANTILES = np.asarray([
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95,
])


def _response_index(
    path: Path, *, expected_variant: str | None
) -> dict[tuple[str, str, str], Mapping[str, Any] | None]:
    result = {}
    for row in pq.read_table(path).to_pylist():
        variant = str(row.get("variant", ""))
        if expected_variant is not None and variant != expected_variant:
            raise ValueError(
                f"calibration response variant must be {expected_variant!r}, got {variant!r}"
            )
        key = (
            str(row["sample_id"]), str(row["challenger_candidate_id"]),
            str(row["model_id"]),
        )
        if key in result:
            raise ValueError("duplicate model/pair response would overwrite calibration input")
        result[key] = row.get("parsed")
    return result


def _assert_response_coverage(
    response_index: Mapping[tuple[str, str, str], Any],
    pair_rows: list[Mapping[str, Any]],
    models: tuple[str, ...] | list[str],
    *,
    context: str,
) -> None:
    expected = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]), str(model))
        for row in pair_rows
        for model in models
    }
    actual = set(response_index)
    if actual != expected:
        raise ValueError(
            f"{context} response coverage mismatch: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )


def _support_and_confirmation(
    original: Mapping[str, Mapping[str, Any] | None],
    confirmation: Mapping[str, Mapping[str, Any] | None],
    models: tuple[str, ...],
) -> tuple[bool, bool, bool]:
    supporters = []
    hard_veto = False
    for model in models:
        output = original.get(model); repeated = confirmation.get(model)
        for candidate_output in (output, repeated):
            if candidate_output is None:
                continue
            reasons = set(candidate_output.get("reason_codes", []))
            hard_veto = hard_veto or "CHALLENGER_OFF_TARGET" in reasons or (
                "COLLISION_RISK" in reasons
                and float(candidate_output.get("challenger_collision_risk", 0))
                > float(candidate_output.get("baseline_collision_risk", 0))
            )
        if output is not None and output.get("decision") == "PREFER_CHALLENGER" and bool(output.get("evidence_reliable")):
            supporters.append(bool(repeated and repeated.get("decision") == "PREFER_CHALLENGER" and repeated.get("evidence_reliable")))
    critic_support = bool(supporters)
    confirmation_stable = critic_support and all(supporters)
    return critic_support, confirmation_stable, hard_veto


def calibrate_api_safe_gate(
    run_dir: str | Path,
    *,
    source_phase: str = API_CALIBRATION_PHASE,
    confirmation_phase: str = API_CALIBRATION_CONFIRMATION_PHASE,
    seed: int = 20260803,
) -> dict[str, Any]:
    root = Path(run_dir); source = root / source_phase; confirmation_dir = root / confirmation_phase
    if source_phase != API_CALIBRATION_PHASE:
        raise ValueError("API safe-gate fitting is restricted to the calibration partition")
    prepare = json.loads((source / "PREPARE_SUMMARY.json").read_text(encoding="utf-8"))
    if prepare.get("partition") != "calibration":
        raise ValueError("API safe-gate source is not the frozen calibration partition")
    if prepare.get("all_challengers") is not False:
        raise ValueError("natural calibration must use the frozen maximum-two challenger preselector")
    stability_path = root / "diagnostic_expanded/DIAGNOSTIC_RESULTS.json"
    if not stability_path.is_file():
        raise ValueError("completed diagnostic perturbation stability is required before calibration")
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    if stability.get("stability_contract") != STABILITY_CONTRACT:
        raise ValueError("diagnostic stability contract differs from preregistration")
    assert_phase_inference_manifest_frozen(source)
    evaluation = pq.read_table(source / "evaluation_manifest.parquet").to_pylist()
    original_index = _response_index(
        source / "pairwise_responses.parquet", expected_variant="original"
    )
    _assert_response_coverage(
        original_index, evaluation, list(MODELS), context="calibration original"
    )
    confirmation_path = confirmation_dir / "pairwise_responses.parquet"
    has_potential_switch = any(
        output is not None and output.get("decision") == "PREFER_CHALLENGER"
        for output in original_index.values()
    )
    if has_potential_switch and not confirmation_path.exists():
        raise ValueError("calibration confirmation responses are required for potential switches")
    confirmation_index = {}
    if confirmation_path.exists():
        assert_phase_inference_manifest_frozen(confirmation_dir)
        confirmation_manifest = json.loads(
            (confirmation_dir / "inference_manifest.json").read_text(encoding="utf-8")
        )
        confirmation_index = _response_index(
            confirmation_path, expected_variant="panel_swap"
        )
        _assert_response_coverage(
            confirmation_index,
            list(confirmation_manifest["rows"]),
            list(MODELS),
            context="calibration confirmation",
        )
    sample_ids = {str(row["sample_id"]) for row in evaluation}
    features = load_feature_index(TRAIN_FEATURES, sample_ids)
    local_oof_path = root / "calibration/local_only_oof_pairs.parquet"
    local_model_path = root / "calibration/local_only_model.json"
    local_oof_rows = pq.read_table(local_oof_path).to_pylist()
    all_query_scores_by_sample: dict[str, float] = {}
    for row in local_oof_rows:
        sample_id = str(row["sample_id"])
        score = float(row["query_score"])
        previous = all_query_scores_by_sample.setdefault(sample_id, score)
        if not np.isclose(previous, score, rtol=0.0, atol=1e-15):
            raise AssertionError("OOF query score differs across a sample's pairs")
    query_scores_by_sample = {
        sample_id: all_query_scores_by_sample[sample_id]
        for sample_id in sample_ids
        if sample_id in all_query_scores_by_sample
    }
    if set(query_scores_by_sample) != sample_ids:
        raise ValueError("calibration subset is not exactly covered by frozen OOF query scores")
    natural_calibration_rows = [
        row for row in cohort_rows(DEFAULT_COHORTS)
        if str(row["partition"]) == "calibration"
    ]
    natural_sample_ids = {str(row["sample_id"]) for row in natural_calibration_rows}
    if set(all_query_scores_by_sample) != natural_sample_ids:
        raise ValueError("full calibration is not exactly covered by frozen OOF query scores")
    local_model = json.loads(local_model_path.read_text(encoding="utf-8"))
    selected_local_query_threshold = float(
        local_model["query_selection"]["threshold"]
    )
    local_vectors = []; hard_validity = []; local_names = None
    for row in evaluation:
        feature = features[str(row["sample_id"])]
        challenger_id = str(row["challenger_candidate_id"])
        names, values = stored_local_feature_vector(feature, challenger_id)
        local_names = names if local_names is None else local_names
        if names != local_names:
            raise AssertionError("local feature schema drift")
        local_vectors.append(values)
        hard_validity.append(
            bool(build_pair_evidence(feature, challenger_id, load_depth=False)["challenger"]["hard_valid"])
        )
    local_matrix = np.vstack(local_vectors)
    sampled_sample_strata = {
        str(row["sample_id"]): (str(row["cohort"]), str(row["query_type"]))
        for row in evaluation
    }
    sampled_total = len(sampled_sample_strata)
    if sampled_total != 150:
        raise ValueError("calibration post-stratification sample count is invalid")
    natural_cohorts = Counter(
        str(row["corrected_cohort"]) for row in natural_calibration_rows
    )
    natural_query_types = Counter(
        str(row["query_type"]) for row in natural_calibration_rows
    )
    sampled_cohorts = Counter(value[0] for value in sampled_sample_strata.values())
    sampled_query_types = Counter(value[1] for value in sampled_sample_strata.values())
    if set(natural_cohorts) != set(sampled_cohorts) or set(natural_query_types) != set(sampled_query_types):
        raise ValueError("balanced calibration omits a natural weighting margin")
    natural_joint = Counter(
        (str(row["corrected_cohort"]), str(row["query_type"]))
        for row in natural_calibration_rows
    )
    sampled_joint = Counter(sampled_sample_strata.values())
    unsupported_natural_joint = sorted(set(natural_joint) - set(sampled_joint))
    sampling_contract_passed = not unsupported_natural_joint

    # The frozen balanced cohort was not sampled in exact proportion to either
    # the natural correctness cohorts or the five query types.  Iterative
    # proportional fitting matches both natural marginal distributions without
    # pretending that an unsampled cohort/query cross-cell was observed.
    sampled_ids = sorted(sampled_sample_strata)
    sample_weights = np.ones(len(sampled_ids), dtype=float)
    target_margins = (
        (0, {key: value / len(natural_calibration_rows) for key, value in natural_cohorts.items()}),
        (1, {key: value / len(natural_calibration_rows) for key, value in natural_query_types.items()}),
    )
    raking_tolerance = 1e-12
    raking_iterations = 0
    raking_converged = False
    weighting_contract = "iterative_proportional_fitting"
    if sampling_contract_passed:
        for iteration in range(1, 1001):
            for field_index, targets in target_margins:
                total_weight = float(sample_weights.sum())
                for value, target in targets.items():
                    mask = np.asarray([
                        sampled_sample_strata[sample_id][field_index] == value
                        for sample_id in sampled_ids
                    ])
                    current = float(sample_weights[mask].sum() / total_weight)
                    if current <= 0.0:
                        raise ValueError("calibration raking margin has no sampled support")
                    sample_weights[mask] *= target / current
            sample_weights *= len(sample_weights) / sample_weights.sum()
            maximum_error = max(
                abs(
                    float(sample_weights[np.asarray([
                        sampled_sample_strata[sample_id][field_index] == value
                        for sample_id in sampled_ids
                    ])].sum() / sample_weights.sum()) - target
                )
                for field_index, targets in target_margins
                for value, target in targets.items()
            )
            raking_iterations = iteration
            if maximum_error <= raking_tolerance:
                raking_converged = True
                break
        if not raking_converged:
            raise ValueError("calibration raking did not converge")
    else:
        # Missing joint cells make the two requested margins mathematically
        # incompatible for this frozen cohort.  The experiment is therefore
        # ineligible by construction.  Cohort-only weights are retained solely
        # to emit an auditable exploratory sweep; they can never unlock P5.
        weighting_contract = "cohort_only_exploratory_sampling_ineligible"
        sample_weights = np.asarray([
            (natural_cohorts[sampled_sample_strata[sample_id][0]] / len(natural_calibration_rows))
            / (sampled_cohorts[sampled_sample_strata[sample_id][0]] / sampled_total)
            for sample_id in sampled_ids
        ], dtype=float)
        sample_weights *= len(sample_weights) / sample_weights.sum()
    sample_weight_by_id = dict(zip(sampled_ids, sample_weights, strict=True))
    weights = np.asarray([
        sample_weight_by_id[str(row["sample_id"])] for row in evaluation
    ])
    sample_contract: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(evaluation):
        sample_id = str(row["sample_id"])
        value = {
            "query_score": query_scores_by_sample[sample_id],
            "recoverable": str(row["cohort"]) == "recoverable_error",
            "sample_weight": float(weights[index]),
        }
        previous = sample_contract.setdefault(sample_id, value)
        if previous != value:
            raise AssertionError("sample-level query contract differs across challenger pairs")
    query_score_values = np.asarray(
        list(all_query_scores_by_sample.values()), dtype=float
    )
    candidate_query_grid = np.unique(np.r_[
        0.0,
        selected_local_query_threshold,
        np.quantile(query_score_values, QUERY_GATE_QUANTILES),
    ])
    query_grid_stats: list[dict[str, float]] = []
    for threshold in candidate_query_grid:
        called = sum(score >= threshold for score in all_query_scores_by_sample.values())
        recoverable_ids = {
            str(row["sample_id"])
            for row in natural_calibration_rows
            if str(row["corrected_cohort"]) == "recoverable_error"
        }
        recall = (
            sum(all_query_scores_by_sample[sample_id] >= threshold for sample_id in recoverable_ids)
            / len(recoverable_ids)
            if recoverable_ids else 1.0
        )
        query_grid_stats.append({
            "threshold": float(threshold),
            "recoverable_recall": float(recall),
            "natural_call_rate": float(called / len(all_query_scores_by_sample)),
        })
    query_grid = np.asarray([
        row["threshold"] for row in query_grid_stats
        if row["recoverable_recall"] >= 0.90
    ], dtype=float)
    if query_grid.size == 0:
        raise ValueError("pre-registered query grid has no threshold with 90% recoverable recall")
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "methods": {},
        "source_phase": source_phase,
        "query_gate_contract": {
            "minimum_calibration_recoverable_recall": 0.90,
            "recall_denominator": len(recoverable_ids),
            "call_rate_denominator": len(all_query_scores_by_sample),
            "post_stratification": weighting_contract,
            "post_stratification_margins": ["corrected_cohort", "query_type"],
            "raking_tolerance": raking_tolerance,
            "raking_iterations": raking_iterations,
            "raking_converged": raking_converged,
            "exploratory_weights_cannot_authorize_p5": not sampling_contract_passed,
            "natural_cohort_counts": dict(sorted(natural_cohorts.items())),
            "sampled_cohort_counts": dict(sorted(sampled_cohorts.items())),
            "natural_query_type_counts": dict(sorted(natural_query_types.items())),
            "sampled_query_type_counts": dict(sorted(sampled_query_types.items())),
            "natural_joint_counts": {
                f"{cohort}|{query_type}": count
                for (cohort, query_type), count in sorted(natural_joint.items())
            },
            "sampled_joint_counts": {
                f"{cohort}|{query_type}": count
                for (cohort, query_type), count in sorted(sampled_joint.items())
            },
            "unsupported_natural_joint_strata": [
                f"{cohort}|{query_type}"
                for cohort, query_type in unsupported_natural_joint
            ],
            "sampling_contract_passed": sampling_contract_passed,
            "sampling_contract_rule": (
                "every natural corrected_cohort x query_type cell must have sampled support"
            ),
            "candidate_grid_stats": query_grid_stats,
            "eligible_query_grid": query_grid.tolist(),
            "local_only_oof_pairs_sha256": hashlib.sha256(local_oof_path.read_bytes()).hexdigest(),
            "local_only_model_sha256": hashlib.sha256(local_model_path.read_bytes()).hexdigest(),
        },
        "diagnostic_stability": {
            "path": str(stability_path.resolve()),
            "sha256": hashlib.sha256(stability_path.read_bytes()).hexdigest(),
            "contract": STABILITY_CONTRACT,
        },
    }
    all_pair_rows = []
    for method, models in METHOD_MODELS.items():
        critic_vectors = []; confirmation_critic_vectors = []; critic_names = None; rows = []
        for index, label in enumerate(evaluation):
            key_base = (str(label["sample_id"]), str(label["challenger_candidate_id"]))
            original = {model: original_index.get((*key_base, model)) for model in MODELS}
            repeated = {model: confirmation_index.get((*key_base, model)) for model in MODELS}
            selected_outputs = {model: original[model] if model in models else None for model in MODELS}
            names, values = critic_feature_vector(selected_outputs)
            critic_names = names if critic_names is None else critic_names
            if names != critic_names:
                raise AssertionError("critic feature schema drift")
            critic_vectors.append(values)
            repeated_outputs = {
                model: repeated[model] if model in models else None for model in MODELS
            }
            repeated_names, repeated_values = critic_feature_vector(repeated_outputs)
            if repeated_names != names:
                raise AssertionError("confirmation critic feature schema drift")
            confirmation_critic_vectors.append(repeated_values)
            support, stable, hard_veto = _support_and_confirmation(original, repeated, models)
            row = {
                "method": method, "sample_id": str(label["sample_id"]),
                "challenger_id": str(label["challenger_candidate_id"]),
                "baseline_correct": bool(label["corrected_baseline_correct"]),
                "challenger_correct": bool(label["corrected_challenger_correct"]),
                "group_id": str(label["bootstrap_sequence_id"]), "cohort": str(label["cohort"]),
                "critic_support": support, "confirmation_stable": stable,
                "evidence_reliable": support,
                "challenger_hard_valid": bool(hard_validity[index]),
                "hard_veto": hard_veto,
                "query_score": query_scores_by_sample[str(label["sample_id"])],
                "sample_weight": float(weights[index]),
            }
            rows.append(row)
        matrix = np.hstack((local_matrix, np.vstack(critic_vectors)))
        confirmation_matrix = np.hstack(
            (local_matrix, np.vstack(confirmation_critic_vectors))
        )
        feature_names = list(local_names or []) + list(critic_names or [])
        beneficial = np.asarray([not row["baseline_correct"] and row["challenger_correct"] for row in rows], dtype=float)
        harmful = np.asarray([row["baseline_correct"] and not row["challenger_correct"] for row in rows], dtype=float)
        bundle, p_b, p_h, folds, confirmation_p_b, confirmation_p_h = cross_fit_dual_risk(
            matrix, beneficial, harmful, [row["group_id"] for row in rows],
            feature_names=feature_names, sample_weight=weights, n_splits=5, seed=seed,
            secondary_values=confirmation_matrix,
        )
        for index, row in enumerate(rows):
            row["p_benefit"] = float(p_b[index]); row["p_harm"] = float(p_h[index]); row["oof_fold"] = int(folds[index])
            row["confirmation_p_benefit"] = float(confirmation_p_b[index])
            row["confirmation_p_harm"] = float(confirmation_p_h[index])
        sweep, selected = threshold_sweep(
            rows, tau_grid=BENEFIT_THRESHOLD_GRID,
            eta_grid=np.round(np.arange(0, .251, .025), 3), query_grid=query_grid,
        )
        stability_models = [
            stability["models"][model] for model in models
        ]
        stability_passed = all(
            bool(model_result.get("stability_passed"))
            for model_result in stability_models
        )
        state = (
            "eligible" if selected is not None and stability_passed and sampling_contract_passed
            else "sampling_ineligible" if not sampling_contract_passed
            else "stability_ineligible" if selected is not None
            else "no_beneficial_switch"
        )
        result["methods"][method] = {
            "threshold_state": state,
            "selected_thresholds": selected or {"tau": 1.0, "eta": 0.0, "recovered": 0, "harmful": 0, "net": 0, "switches": 0},
            "feature_names": feature_names,
            "dual_risk": {
                "benefit_model": asdict(bundle.benefit_model), "harm_model": asdict(bundle.harm_model),
                "benefit_platt": asdict(bundle.benefit_platt), "harm_platt": asdict(bundle.harm_platt),
                "n_splits": bundle.n_splits, "seed": bundle.seed, "calibration_partition": bundle.calibration_partition,
            },
            "original_response_coverage": sum(any(original_index.get((str(row["sample_id"]), str(row["challenger_candidate_id"]), model)) for model in models) for row in evaluation) / len(evaluation),
            "confirmation_stable_pairs": sum(row["confirmation_stable"] for row in rows),
            "threshold_sweep": sweep,
            "query_grid": query_grid.tolist(),
            "stability_passed": stability_passed,
            "sampling_contract_passed": sampling_contract_passed,
        }
        all_pair_rows.extend(rows)
    output = root / "calibration"
    _parquet(output / "api_safe_gate_oof_pairs.parquet", all_pair_rows)
    # Keep the large sweep machine-readable without embedding it in the compact model JSON.
    sweep_rows = []
    for method, payload in result["methods"].items():
        sweep_rows.extend({"method": method, **row} for row in payload.pop("threshold_sweep"))
    _parquet(output / "api_safe_gate_threshold_sweep.parquet", sweep_rows)
    calibration_evidence_paths = {
        "source_inference_sidecar": source / "INFERENCE_MANIFEST_IDENTITY.json",
        "source_inference_manifest": source / "inference_manifest.json",
        "source_sample_manifest": source / "sample_manifest.json",
        "source_evaluation_manifest": source / "evaluation_manifest.parquet",
        "source_original_responses": source / "pairwise_responses.parquet",
        "confirmation_inference_sidecar": confirmation_dir / "INFERENCE_MANIFEST_IDENTITY.json",
        "confirmation_inference_manifest": confirmation_dir / "inference_manifest.json",
        "confirmation_responses": confirmation_dir / "pairwise_responses.parquet",
        "local_query_oof": local_oof_path,
        "local_query_model": local_model_path,
        "train_features": TRAIN_FEATURES,
        "cohort_source": DEFAULT_COHORTS,
        "diagnostic_stability": stability_path,
        "calibration_oof_pairs": output / "api_safe_gate_oof_pairs.parquet",
        "calibration_threshold_sweep": output / "api_safe_gate_threshold_sweep.parquet",
        "evaluator_source": Path(__file__),
    }
    result["calibration_evidence_bindings"] = {
        name: file_identity(path)
        for name, path in calibration_evidence_paths.items()
        if path.is_file()
    }
    _atomic_json(output / "API_SAFE_GATE_CALIBRATION.json", result)
    return result
