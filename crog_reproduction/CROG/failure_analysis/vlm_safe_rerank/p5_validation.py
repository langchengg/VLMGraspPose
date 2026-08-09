from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

from .api_calibration import (
    METHOD_MODELS,
    MODELS,
    _assert_response_coverage,
    _response_index,
    _support_and_confirmation,
)
from .critic_features import critic_feature_vector
from .dataset import load_feature_index, load_label_index
from .features import build_pair_evidence, select_challengers, stored_local_feature_vector
from .local_experiment import (
    SPLIT_MANIFEST,
    _bootstrap_delta,
    _bundle,
    _model,
    _validation_samples,
)
from .manifest import compute_candidate_q_identity, file_identity
from .policy import full_denominator_metrics
from .runner import (
    VALIDATION_CORRECTED,
    VALIDATION_FEATURES,
    VALIDATION_LEGACY,
    _atomic_json,
    _exclusive_json,
    _parquet,
    assert_phase_inference_manifest_frozen,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_p5_dependencies(root: Path, phase: Path) -> dict[str, Any]:
    identity_path = phase / "INFERENCE_MANIFEST_IDENTITY.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    files = identity.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("P5 identity sidecar does not bind all frozen dependencies")
    required_dependencies = {
        "inference_manifest",
        "sample_query_gate",
        "api_safe_gate_calibration",
        "local_query_model",
        "local_query_oof",
        "validation_features",
        "diagnostic_stability",
    }
    if set(files) != required_dependencies:
        raise RuntimeError("P5 identity sidecar dependency set changed")
    for name, payload in files.items():
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"invalid P5 dependency identity for {name}")
        path = Path(str(payload.get("path", "")))
        if not path.is_file() or _file_sha256(path) != payload.get("sha256"):
            raise RuntimeError(f"P5 frozen dependency changed: {name}")
    manifest = json.loads((phase / "inference_manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads(
        (root / "calibration/API_SAFE_GATE_CALIBRATION.json").read_text(encoding="utf-8")
    )
    eligible = sorted(
        method for method, value in calibration["methods"].items()
        if value["threshold_state"] == "eligible"
    )
    needed = sorted({model for method in eligible for model in METHOD_MODELS[method]})
    if manifest.get("eligible_methods") != eligible or manifest.get("needed_models") != needed:
        raise RuntimeError("P5 frozen eligibility/model plan differs from calibration")
    query_contract = calibration.get("query_gate_contract", {})
    if (
        query_contract.get("local_only_model_sha256")
        != files["local_query_model"].get("sha256")
        or query_contract.get("local_only_oof_pairs_sha256")
        != files["local_query_oof"].get("sha256")
    ):
        raise RuntimeError("P5 query model/OOF lineage differs from API calibration")
    stability_contract = calibration.get("diagnostic_stability", {})
    if stability_contract.get("sha256") != files["diagnostic_stability"].get("sha256"):
        raise RuntimeError("P5 diagnostic stability lineage differs from API calibration")
    return manifest


def prepare_p5_validation_inference(run_dir: str | Path) -> dict[str, Any]:
    """Inference-only query gating over untouched validation; never opens labels."""

    root = Path(run_dir)
    phase = root / "p5_validation"
    if phase.exists():
        raise FileExistsError("P5 validation manifest is immutable once created")
    calibration_path = root / "calibration/API_SAFE_GATE_CALIBRATION.json"
    local_path = root / "calibration/local_only_model.json"
    local_oof_path = root / "calibration/local_only_oof_pairs.parquet"
    stability_path = root / "diagnostic_expanded/DIAGNOSTIC_RESULTS.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    eligible_methods = [
        method for method, value in calibration["methods"].items()
        if value["threshold_state"] == "eligible"
    ]
    eligible_methods.sort()
    needed_models = sorted({
        model for method in eligible_methods for model in METHOD_MODELS[method]
    })
    local = json.loads(local_path.read_text(encoding="utf-8"))
    query_model = _model(local["query_model"])
    method_query_thresholds = {
        method: float(calibration["methods"][method]["selected_thresholds"]["query_threshold"])
        for method in eligible_methods
    }
    query_threshold = (
        min(method_query_thresholds.values())
        if method_query_thresholds else 1.0
    )
    samples = _validation_samples()
    sample_ids = {str(row["sample_id"]) for row in samples}
    features = load_feature_index(VALIDATION_FEATURES, sample_ids)
    inference_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        challengers = select_challengers(features[sample_id], maximum=2)
        names, vector = stored_local_feature_vector(features[sample_id], challengers[0])
        if names != list(query_model.feature_names):
            raise AssertionError("P5 query-gate feature schema drift")
        query_score = float(query_model.predict_proba(vector)[0])
        called = bool(eligible_methods and query_score >= query_threshold)
        sample_rows.append({
            "sample_id": sample_id,
            "group_id": sample["group_id"],
            "frame_id": sample["frame_id"],
            "query_type": sample["query_type"],
            "query_score": query_score,
            "query_gate_called": called,
        })
        if called:
            for challenger_id in challengers:
                inference_rows.append({
                    "sample_id": sample_id,
                    "feature_source_sample_id": int(
                        features[sample_id].get("sample_id", features[sample_id].get("sample_index"))
                    ),
                    "baseline_candidate_id": "candidate_0",
                    "challenger_candidate_id": challenger_id,
                    "official_split": "val",
                    "query_score": query_score,
                })
    phase.mkdir(parents=True, exist_ok=False)
    inference_path = phase / "inference_manifest.json"
    _exclusive_json(inference_path, {
        "schema_version": "1.0.0",
        "phase": "p5_validation",
        "feature_file": str(VALIDATION_FEATURES.resolve()),
        "eligible_methods": eligible_methods,
        "needed_models": needed_models,
        "query_call_threshold": query_threshold,
        "method_query_thresholds": method_query_thresholds,
        "rows": inference_rows,
    })
    sample_gate_path = phase / "sample_query_gate.parquet"
    _parquet(sample_gate_path, sample_rows)
    _exclusive_json(phase / "INFERENCE_MANIFEST_IDENTITY.json", {
        "sha256": _file_sha256(inference_path),
        "binding": "exact P5 inference, query gate, API calibration, and local query model bytes",
        "files": {
            "inference_manifest": {
                "path": str(inference_path.resolve()),
                "sha256": _file_sha256(inference_path),
            },
            "sample_query_gate": {
                "path": str(sample_gate_path.resolve()),
                "sha256": _file_sha256(sample_gate_path),
            },
            "api_safe_gate_calibration": {
                "path": str(calibration_path.resolve()),
                "sha256": _file_sha256(calibration_path),
            },
            "local_query_model": {
                "path": str(local_path.resolve()),
                "sha256": _file_sha256(local_path),
            },
            "local_query_oof": {
                "path": str(local_oof_path.resolve()),
                "sha256": _file_sha256(local_oof_path),
            },
            "validation_features": {
                "path": str(VALIDATION_FEATURES.resolve()),
                "sha256": _file_sha256(VALIDATION_FEATURES),
            },
            "diagnostic_stability": {
                "path": str(stability_path.resolve()),
                "sha256": _file_sha256(stability_path),
            },
        },
    })
    summary = {
        "validation_samples": len(samples),
        "query_gate_called_samples": sum(row["query_gate_called"] for row in sample_rows),
        "query_gate_call_rate": sum(row["query_gate_called"] for row in sample_rows) / len(samples),
        "pair_requests_per_model": len(inference_rows),
        "eligible_methods": eligible_methods,
        "needed_models": needed_models,
        "query_call_threshold": query_threshold,
        "method_query_thresholds": method_query_thresholds,
        "labels_opened": False,
    }
    _atomic_json(phase / "PREPARE_SUMMARY.json", summary)
    return summary


def evaluate_p5_validation(run_dir: str | Path, *, seed: int = 20260803) -> dict[str, Any]:
    """Evaluator-only frozen P5 policy over the full 8,669 denominator."""

    root = Path(run_dir)
    phase = root / "p5_validation"
    manifest = _verify_p5_dependencies(root, phase)
    calibration = json.loads(
        (root / "calibration/API_SAFE_GATE_CALIBRATION.json").read_text(encoding="utf-8")
    )
    samples = _validation_samples()
    sample_ids = {str(row["sample_id"]) for row in samples}
    features = load_feature_index(VALIDATION_FEATURES, sample_ids)
    corrected = load_label_index(VALIDATION_CORRECTED, sample_ids)
    legacy = load_label_index(VALIDATION_LEGACY, sample_ids)
    query_payload = pq.read_table(phase / "sample_query_gate.parquet").to_pylist()
    query_rows = {str(row["sample_id"]): row for row in query_payload}
    if len(query_rows) != len(query_payload) or set(query_rows) != sample_ids:
        raise RuntimeError("P5 query-gate rows do not exactly cover untouched validation")
    original_path = phase / "pairwise_responses.parquet"
    original = (
        _response_index(original_path, expected_variant="original")
        if original_path.exists() else {}
    )
    _assert_response_coverage(
        original,
        list(manifest["rows"]),
        list(manifest["needed_models"]),
        context="P5 validation original",
    )
    confirmation_path = root / "p5_validation_confirmation/pairwise_responses.parquet"
    has_potential_switch = any(
        output is not None and output.get("decision") == "PREFER_CHALLENGER"
        for output in original.values()
    )
    if has_potential_switch and not confirmation_path.exists():
        raise RuntimeError("P5 validation confirmation is required for potential switches")
    confirmation = {}
    if confirmation_path.exists():
        confirmation_dir = confirmation_path.parent
        assert_phase_inference_manifest_frozen(confirmation_dir)
        confirmation_manifest = json.loads(
            (confirmation_dir / "inference_manifest.json").read_text(encoding="utf-8")
        )
        confirmation = _response_index(
            confirmation_path, expected_variant="panel_swap"
        )
        _assert_response_coverage(
            confirmation,
            list(confirmation_manifest["rows"]),
            list(manifest["needed_models"]),
            context="P5 validation confirmation",
        )
    method_pair_rows: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHOD_MODELS
    }
    decisions: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHOD_MODELS
    }
    bundles = {
        method: _bundle(payload["dual_risk"])
        for method, payload in calibration["methods"].items()
    }
    for sample in samples:
        sample_id = str(sample["sample_id"])
        feature = features[sample_id]
        challenger_ids = select_challengers(feature, maximum=2)
        hard_valid_by_challenger = {
            challenger_id: bool(
                build_pair_evidence(feature, challenger_id, load_depth=False)["challenger"]["hard_valid"]
            )
            for challenger_id in challenger_ids
        }
        baseline_id = "candidate_0"
        query_score = float(query_rows[sample_id]["query_score"])
        for method, models in METHOD_MODELS.items():
            method_payload = calibration["methods"][method]
            threshold_state = method_payload["threshold_state"]
            thresholds = method_payload["selected_thresholds"]
            bundle = bundles[method]
            feature_names = list(method_payload["feature_names"])
            candidates: list[dict[str, Any]] = []
            for challenger_id in challenger_ids:
                local_names, local_values = stored_local_feature_vector(feature, challenger_id)
                key_base = (sample_id, challenger_id)
                original_outputs = {
                    model: original.get((*key_base, model)) for model in MODELS
                }
                repeated_outputs = {
                    model: confirmation.get((*key_base, model)) for model in MODELS
                }
                selected_outputs = {
                    model: original_outputs[model] if model in models else None
                    for model in MODELS
                }
                critic_names, critic_values = critic_feature_vector(selected_outputs)
                names = list(local_names) + list(critic_names)
                if names != feature_names:
                    raise AssertionError("P5 validation feature schema differs from calibration")
                matrix = np.r_[local_values, critic_values]
                p_benefit, p_harm = bundle.predict(matrix)
                repeated_selected_outputs = {
                    model: repeated_outputs[model] if model in models else None
                    for model in MODELS
                }
                repeated_names, repeated_values = critic_feature_vector(
                    repeated_selected_outputs
                )
                if repeated_names != critic_names:
                    raise AssertionError("P5 confirmation critic feature schema differs")
                repeated_matrix = np.r_[local_values, repeated_values]
                confirmation_p_benefit, confirmation_p_harm = bundle.predict(
                    repeated_matrix
                )
                support, stable, hard_veto = _support_and_confirmation(
                    original_outputs, repeated_outputs, models
                )
                hard_valid = hard_valid_by_challenger[challenger_id]
                row = {
                    "method": method,
                    "sample_id": sample_id,
                    "challenger_id": challenger_id,
                    "p_benefit": float(p_benefit[0]),
                    "p_harm": float(p_harm[0]),
                    "confirmation_p_benefit": float(confirmation_p_benefit[0]),
                    "confirmation_p_harm": float(confirmation_p_harm[0]),
                    "query_score": query_score,
                    "critic_support": support,
                    "confirmation_stable": stable,
                    "challenger_hard_valid": hard_valid,
                    "hard_veto": hard_veto,
                    "api_called": bool(query_rows[sample_id]["query_gate_called"]),
                }
                row["eligible"] = bool(
                    threshold_state == "eligible"
                    and row["api_called"]
                    and query_score >= float(thresholds["query_threshold"])
                    and support and stable and hard_valid and not hard_veto
                    and row["p_benefit"] >= float(thresholds["tau"])
                    and row["p_harm"] <= float(thresholds["eta"])
                    and row["confirmation_p_benefit"] >= float(thresholds["tau"])
                    and row["confirmation_p_harm"] <= float(thresholds["eta"])
                )
                method_pair_rows[method].append(row)
                if row["eligible"]:
                    candidates.append(row)
            chosen = sorted(
                candidates,
                key=lambda row: (-row["p_benefit"], row["p_harm"], row["challenger_id"]),
            )[0] if candidates else None
            selected_id = baseline_id if chosen is None else str(chosen["challenger_id"])
            decisions[method].append({
                "method": method,
                "sample_id": sample_id,
                "group_id": sample["group_id"],
                "frame_id": sample["frame_id"],
                "query_type": sample["query_type"],
                "baseline_candidate_id": baseline_id,
                "selected_candidate_id": selected_id,
                "baseline_correct": bool(corrected[sample_id][baseline_id]),
                "selected_correct": bool(corrected[sample_id][selected_id]),
                "legacy_baseline_correct": bool(legacy[sample_id][baseline_id]),
                "legacy_selected_correct": bool(legacy[sample_id][selected_id]),
                "switched": selected_id != baseline_id,
                "query_gate_called": bool(query_rows[sample_id]["query_gate_called"]),
                "p_benefit": None if chosen is None else chosen["p_benefit"],
                "p_harm": None if chosen is None else chosen["p_harm"],
            })
    results: dict[str, Any] = {}
    for method, rows in decisions.items():
        corrected_metrics = full_denominator_metrics(
            rows, baseline_key="baseline_correct", selected_key="selected_correct"
        )
        legacy_metrics = full_denominator_metrics(
            rows, baseline_key="legacy_baseline_correct", selected_key="legacy_selected_correct"
        )
        ci = _bootstrap_delta(rows, draws=10000, seed=seed)
        discordant = corrected_metrics["recovered"] + corrected_metrics["harmful"]
        mcnemar = (
            float(binomtest(corrected_metrics["recovered"], discordant, p=.5).pvalue)
            if discordant else 1.0
        )
        stability_passed = bool(calibration["methods"][method].get("stability_passed"))
        hard_constraints_passed = bool(
            corrected_metrics["net"] > 0
            and corrected_metrics["recovered"] > corrected_metrics["harmful"]
            and corrected_metrics["final_j1"] >= corrected_metrics["baseline_j1"]
            and corrected_metrics["outcome_changing_precision"] >= .67
            and corrected_metrics["harm_rate"] <= .01
            and stability_passed
        )
        go = bool(hard_constraints_passed and ci["lower"] >= 0.0)
        inconclusive = bool(
            hard_constraints_passed and ci["lower"] < 0.0 and ci["upper"] > 0.0
        )
        results[method] = {
            "validation_status": "GO" if go else "INCONCLUSIVE" if inconclusive else "NO_GO",
            "corrected": corrected_metrics,
            "legacy": legacy_metrics,
            "scene_sequence_bootstrap_delta_j1": ci,
            "exact_mcnemar_p": mcnemar,
            "thresholds_from_calibration": calibration["methods"][method]["selected_thresholds"],
            "stability_contract_passed": stability_passed,
            "hard_constraints_passed": hard_constraints_passed,
        }
    eligible = [
        (method, payload) for method, payload in results.items()
        if payload["validation_status"] == "GO"
    ]
    if eligible:
        primary = sorted(
            eligible,
            key=lambda item: (
                -item[1]["corrected"]["net"],
                -item[1]["corrected"]["outcome_changing_precision"],
                item[1]["corrected"]["switch_rate"],
                0 if item[0] == "P5_flash_safe" else 1,
            ),
        )[0][0]
        status = "GO"
    else:
        primary = "q_only"
        status = (
            "INCONCLUSIVE"
            if any(payload["validation_status"] == "INCONCLUSIVE" for payload in results.values())
            else "NO_GO"
        )
    all_decisions = [row for rows in decisions.values() for row in rows]
    all_pairs = [row for rows in method_pair_rows.values() for row in rows]
    _parquet(phase / "p5_pair_scores.parquet", all_pairs)
    _parquet(phase / "p5_decisions.parquet", all_decisions)
    identity = compute_candidate_q_identity(VALIDATION_FEATURES)
    evidence_paths = {
        "p5_dependency_sidecar": phase / "INFERENCE_MANIFEST_IDENTITY.json",
        "api_calibration": root / "calibration/API_SAFE_GATE_CALIBRATION.json",
        "diagnostic_stability": root / "diagnostic_expanded/DIAGNOSTIC_RESULTS.json",
        "inference_manifest": phase / "inference_manifest.json",
        "sample_query_gate": phase / "sample_query_gate.parquet",
        "original_responses": original_path,
        "confirmation_inference_sidecar": confirmation_path.parent / "INFERENCE_MANIFEST_IDENTITY.json",
        "confirmation_inference_manifest": confirmation_path.parent / "inference_manifest.json",
        "confirmation_responses": confirmation_path,
        "pair_scores": phase / "p5_pair_scores.parquet",
        "decisions": phase / "p5_decisions.parquet",
        "validation_features": VALIDATION_FEATURES,
        "validation_split_manifest": SPLIT_MANIFEST,
        "corrected_labels": VALIDATION_CORRECTED,
        "legacy_labels": VALIDATION_LEGACY,
        "evaluator_source": Path(__file__),
    }
    evidence_bindings = {
        name: file_identity(path)
        for name, path in evidence_paths.items()
        if Path(path).is_file()
    }
    result = {
        "schema_version": "1.0.0",
        "validation_status": status,
        "primary_method": primary,
        "expected_denominator": len(samples),
        "methods": results,
        "no_validation_tuning": True,
        "statistical_seed": seed,
        "bootstrap_draws": 10000,
        "candidate_identity_sha256": identity["candidate_identity_sha256"],
        "q_value_sha256": identity["q_value_sha256"],
        "combined_identity_sha256": identity["combined_identity_sha256"],
        "stability_contract_passed": bool(
            primary != "q_only"
            and results[primary]["stability_contract_passed"]
        ),
        # A GO decision is not lockable merely by copying its booleans.  These
        # identities bind the actual calibration, responses, labels, evaluator,
        # and saved per-pair/per-sample evidence that produced the decision.
        "p5_evidence_bindings": evidence_bindings,
    }
    _atomic_json(phase / "P5_VALIDATION_RESULTS.json", result)
    return result


__all__ = ["evaluate_p5_validation", "prepare_p5_validation_inference"]
