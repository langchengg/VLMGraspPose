"""Protocol, validation, and formal-run gates for safe VLM reranking.

The gate is intentionally local and deterministic.  Passing it never performs
an API call; it only proves that immutable artifacts still match the identities
that were validated and that both independent formal-run opt-ins are present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import (
    LOCKED_MANIFEST_KIND,
    PROTOCOL_LOCK_KIND,
    ManifestError,
    compute_candidate_q_identity,
    file_identity,
    load_json_object,
    sha256_file,
    verify_data_manifest,
    verify_file_identity,
    verify_inference_manifest,
    verify_layer,
    write_layer,
)


FORMAL_CLAIM_KIND = "vlm_safe_rerank_formal_run_claim"
FORMAL_ENVIRONMENT_VARIABLE = "ALLOW_FORMAL_API_RUN"
SUPPORTED_FORMAL_PRIMARY_METHODS = frozenset({
    "P5_er2_safe",
    "P5_flash_safe",
    "P5_joint_safe",
})


class FormalRunDenied(ManifestError):
    """Raised when either formal-run gate or any frozen binding is invalid."""


def _nonempty(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ManifestError(f"{field} must be non-empty")
    return text


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ManifestError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ManifestError(f"{field} must be a positive integer")
    return parsed


def _sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ManifestError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _normalise_method(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _is_q_only(value: Any) -> bool:
    normalised = _normalise_method(value)
    return "qonly" in normalised or normalised in {
        "q",
    }


def _string_list(value: Sequence[str] | None, field: str) -> list[str]:
    if value is None or isinstance(value, (str, bytes)):
        raise ManifestError(f"{field} must be a non-empty sequence")
    result = [_nonempty(item, field) for item in value]
    if not result or len(result) != len(set(result)):
        raise ManifestError(f"{field} must contain unique non-empty methods")
    return result


def _identity_hashes(data_manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    identity = data_manifest.get("formal_candidate_q_identity")
    if not isinstance(identity, Mapping):
        raise ManifestError("DATA_MANIFEST has no formal candidate/q identity")
    return (
        _sha256(identity.get("candidate_identity_sha256"), "candidate_identity_sha256"),
        _sha256(identity.get("q_value_sha256"), "q_value_sha256"),
        _sha256(identity.get("combined_identity_sha256"), "combined_identity_sha256"),
    )


def _assert_same_file(expected: Mapping[str, Any], actual_path: str | Path, field: str) -> None:
    actual = file_identity(actual_path)
    for key in ("path", "sha256", "size_bytes"):
        if expected.get(key) != actual.get(key):
            raise ManifestError(f"{field} identity drift in {key}")


def write_protocol_lock(
    path: str | Path,
    *,
    run_id: str,
    data_manifest: str | Path,
    inference_manifest: str | Path,
    expected_denominator: int,
    protocol: Mapping[str, Any],
    eligible_primary_methods: Sequence[str] | None = None,
    primary_method: str | None = None,
) -> dict[str, Any]:
    """Create PROTOCOL_LOCK exactly once after checking all data bindings."""

    denominator = _positive_int(expected_denominator, "expected_denominator")
    data = verify_data_manifest(data_manifest)
    inference = verify_inference_manifest(inference_manifest)
    if data.get("expected_denominator") != denominator:
        raise ManifestError("DATA_MANIFEST expected denominator mismatch")
    if inference.get("expected_denominator") != denominator:
        raise ManifestError("inference manifest expected denominator mismatch")
    data_inference = data.get("inference_manifest")
    if not isinstance(data_inference, Mapping):
        raise ManifestError("DATA_MANIFEST inference binding is invalid")
    _assert_same_file(data_inference, inference_manifest, "inference manifest")
    if data.get("inference_manifest_content_sha256") != inference.get("content_sha256"):
        raise ManifestError("inference manifest semantic hash mismatch")
    if not isinstance(protocol, Mapping) or not protocol:
        raise ManifestError("protocol must be a non-empty mapping")

    if eligible_primary_methods is None:
        if primary_method is None:
            raise ManifestError("eligible_primary_methods or primary_method is required")
        eligible = [_nonempty(primary_method, "primary_method")]
    else:
        eligible = _string_list(eligible_primary_methods, "eligible_primary_methods")
        if primary_method is not None and primary_method not in eligible:
            raise ManifestError("primary_method is not eligible under the protocol")

    candidate_sha, q_sha, combined_sha = _identity_hashes(data)
    payload: dict[str, Any] = {
        "run_id": _nonempty(run_id, "run_id"),
        "expected_denominator": denominator,
        "data_manifest": file_identity(data_manifest),
        "data_manifest_content_sha256": data["content_sha256"],
        "inference_manifest": file_identity(inference_manifest),
        "inference_manifest_content_sha256": inference["content_sha256"],
        "candidate_identity_sha256": candidate_sha,
        "q_value_sha256": q_sha,
        "combined_identity_sha256": combined_sha,
        "eligible_primary_methods": eligible,
        "preselected_primary_method": primary_method,
        "protocol": dict(protocol),
    }
    return write_layer(path, PROTOCOL_LOCK_KIND, payload)


def verify_protocol_lock(path: str | Path) -> dict[str, Any]:
    lock = verify_layer(path, expected_kind=PROTOCOL_LOCK_KIND)
    denominator = _positive_int(lock.get("expected_denominator"), "expected_denominator")
    data_identity = lock.get("data_manifest")
    inference_identity = lock.get("inference_manifest")
    if not isinstance(data_identity, Mapping) or not isinstance(inference_identity, Mapping):
        raise ManifestError("PROTOCOL_LOCK artifact binding is invalid")
    data = verify_data_manifest(str(data_identity.get("path", "")))
    inference = verify_inference_manifest(str(inference_identity.get("path", "")))
    if data.get("content_sha256") != lock.get("data_manifest_content_sha256"):
        raise ManifestError("DATA_MANIFEST semantic hash drift")
    if inference.get("content_sha256") != lock.get("inference_manifest_content_sha256"):
        raise ManifestError("inference manifest semantic hash drift")
    if data.get("expected_denominator") != denominator:
        raise ManifestError("DATA_MANIFEST denominator drift")
    if inference.get("expected_denominator") != denominator:
        raise ManifestError("inference manifest denominator drift")

    candidate_sha, q_sha, combined_sha = _identity_hashes(data)
    expected_hashes = {
        "candidate_identity_sha256": candidate_sha,
        "q_value_sha256": q_sha,
        "combined_identity_sha256": combined_sha,
    }
    for field, expected in expected_hashes.items():
        if lock.get(field) != expected:
            raise ManifestError(f"PROTOCOL_LOCK {field} binding drift")

    eligible = _string_list(lock.get("eligible_primary_methods"), "eligible_primary_methods")
    preselected = lock.get("preselected_primary_method")
    if preselected is not None and preselected not in eligible:
        raise ManifestError("preselected primary method is no longer eligible")
    return lock


def _validation_binding(
    validation_result: str | Path,
    *,
    primary_method: str | None,
    validation_status: str | None,
) -> tuple[dict[str, Any], str, str]:
    validation = load_json_object(validation_result)
    artifact_status = validation.get("validation_status")
    if artifact_status != "GO":
        raise FormalRunDenied("validation_status must be exactly 'GO'")
    if validation_status is not None and validation_status != artifact_status:
        raise FormalRunDenied("explicit validation_status does not match validation artifact")
    artifact_primary = _nonempty(validation.get("primary_method"), "validation primary_method")
    if primary_method is not None and primary_method != artifact_primary:
        raise FormalRunDenied("explicit primary method does not match validation artifact")
    if _is_q_only(artifact_primary):
        raise FormalRunDenied("q-only primary is not permitted to authorise a formal API run")
    if artifact_primary.lower().startswith("p5"):
        _verify_p5_validation_evidence(validation, artifact_primary)
    return validation, artifact_status, artifact_primary


def _verify_p5_validation_evidence(
    validation: Mapping[str, Any], primary_method: str
) -> None:
    """Recompute the immutable evidence chain behind a lockable P5 GO.

    A copied ``stability_contract_passed=true`` is not evidence.  A formal lock
    must be anchored to the exact calibration, diagnostic perturbations,
    responses, labels, evaluator source, and saved P5 decisions that generated
    the GO result.
    """

    if validation.get("stability_contract_passed") is not True:
        raise FormalRunDenied(
            "P5 primary requires a passed frozen perturbation stability contract"
        )
    bindings = validation.get("p5_evidence_bindings")
    required = {
        "p5_dependency_sidecar",
        "api_calibration",
        "diagnostic_stability",
        "inference_manifest",
        "sample_query_gate",
        "original_responses",
        "confirmation_inference_sidecar",
        "confirmation_inference_manifest",
        "confirmation_responses",
        "pair_scores",
        "decisions",
        "validation_features",
        "validation_split_manifest",
        "corrected_labels",
        "legacy_labels",
        "evaluator_source",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != required:
        raise FormalRunDenied("P5 GO has no complete frozen evidence binding set")
    try:
        for identity in bindings.values():
            if not isinstance(identity, Mapping):
                raise ManifestError("invalid P5 evidence file identity")
            verify_file_identity(identity)
    except ManifestError as exc:
        raise FormalRunDenied(f"P5 evidence binding drift: {exc}") from exc

    calibration = load_json_object(str(bindings["api_calibration"]["path"]))
    _verify_calibration_evidence_bindings(calibration)
    methods = calibration.get("methods")
    if not isinstance(methods, Mapping) or primary_method not in methods:
        raise FormalRunDenied("P5 primary is absent from the bound calibration")
    method = methods[primary_method]
    if not isinstance(method, Mapping):
        raise FormalRunDenied("P5 primary calibration payload is invalid")
    if method.get("threshold_state") != "eligible" or method.get("stability_passed") is not True:
        raise FormalRunDenied("P5 primary was not stability-eligible in calibration")

    validation_methods = validation.get("methods")
    if not isinstance(validation_methods, Mapping):
        raise FormalRunDenied("P5 validation has no per-method evidence")
    validation_method = validation_methods.get(primary_method)
    if not isinstance(validation_method, Mapping):
        raise FormalRunDenied("P5 primary is absent from per-method validation evidence")
    if validation_method.get("validation_status") != "GO":
        raise FormalRunDenied("P5 primary did not individually pass validation")
    if validation_method.get("thresholds_from_calibration") != method.get("selected_thresholds"):
        raise FormalRunDenied("P5 validation thresholds differ from frozen calibration")
    if (
        validation_method.get("stability_contract_passed") is not True
        or validation_method.get("hard_constraints_passed") is not True
    ):
        raise FormalRunDenied("P5 primary validation constraints did not pass")

    from .critic_evaluation import STABILITY_CONTRACT

    diagnostic_binding = bindings["diagnostic_stability"]
    calibration_stability = calibration.get("diagnostic_stability")
    if not isinstance(calibration_stability, Mapping):
        raise FormalRunDenied("P5 calibration has no diagnostic stability binding")
    if (
        calibration_stability.get("sha256") != diagnostic_binding.get("sha256")
        or calibration_stability.get("contract") != STABILITY_CONTRACT
    ):
        raise FormalRunDenied("P5 calibration stability lineage/contract drifted")
    diagnostic = load_json_object(str(diagnostic_binding["path"]))
    if diagnostic.get("stability_contract") != STABILITY_CONTRACT:
        raise FormalRunDenied("P5 diagnostic stability contract drifted")
    diagnostic_models = diagnostic.get("models")
    if not isinstance(diagnostic_models, Mapping):
        raise FormalRunDenied("P5 diagnostic stability model evidence is missing")

    from .api_calibration import METHOD_MODELS

    expected_models = METHOD_MODELS.get(primary_method)
    if not expected_models:
        raise FormalRunDenied("P5 primary has no registered model contract")
    _verify_diagnostic_stability_evidence(diagnostic, expected_models)
    if any(
        not isinstance(diagnostic_models.get(model), Mapping)
        or diagnostic_models[model].get("stability_passed") is not True
        for model in expected_models
    ):
        raise FormalRunDenied("P5 primary models did not pass bound perturbation stability")

    sidecar = load_json_object(str(bindings["p5_dependency_sidecar"]["path"]))
    sidecar_files = sidecar.get("files")
    expected_sidecar_files = {
        "inference_manifest",
        "sample_query_gate",
        "api_safe_gate_calibration",
        "local_query_model",
        "local_query_oof",
        "validation_features",
        "diagnostic_stability",
    }
    if not isinstance(sidecar_files, Mapping) or set(sidecar_files) != expected_sidecar_files:
        raise FormalRunDenied("P5 dependency sidecar is incomplete")
    for name, payload in sidecar_files.items():
        if not isinstance(payload, Mapping):
            raise FormalRunDenied(f"P5 sidecar identity is invalid: {name}")
        path = Path(str(payload.get("path", "")))
        if not path.is_file() or sha256_file(path) != payload.get("sha256"):
            raise FormalRunDenied(f"P5 sidecar dependency drifted: {name}")
    if (
        sidecar_files["api_safe_gate_calibration"].get("sha256")
        != bindings["api_calibration"].get("sha256")
        or sidecar_files["diagnostic_stability"].get("sha256")
        != diagnostic_binding.get("sha256")
        or sidecar_files["inference_manifest"].get("sha256")
        != bindings["inference_manifest"].get("sha256")
        or sidecar_files["sample_query_gate"].get("sha256")
        != bindings["sample_query_gate"].get("sha256")
        or sidecar_files["validation_features"].get("sha256")
        != bindings["validation_features"].get("sha256")
    ):
        raise FormalRunDenied("P5 evaluator and dependency-sidecar evidence disagree")
    confirmation_sidecar = load_json_object(
        str(bindings["confirmation_inference_sidecar"]["path"])
    )
    if (
        confirmation_sidecar.get("sha256")
        != bindings["confirmation_inference_manifest"].get("sha256")
    ):
        raise FormalRunDenied("P5 confirmation inference binding drifted")

    # Recompute the primary's saved denominator metrics and clustered CI from
    # the bound decisions, independent of the claimed top-level GO booleans.
    import numpy as np
    import pyarrow.parquet as pq

    from .api_calibration import MODELS, _response_index, _support_and_confirmation
    from .critic_features import critic_feature_vector
    from .dataset import load_feature_index, load_label_index
    from .features import (
        build_pair_evidence,
        ordered_candidates,
        select_challengers,
        stored_local_feature_vector,
    )
    from .local_experiment import _bootstrap_delta, _bundle, _model
    from .policy import full_denominator_metrics

    decision_rows = [
        row
        for row in pq.read_table(str(bindings["decisions"]["path"])).to_pylist()
        if str(row.get("method")) == primary_method
    ]
    denominator = _positive_int(validation.get("expected_denominator"), "validation expected_denominator")
    sample_ids = [str(row.get("sample_id")) for row in decision_rows]
    if len(decision_rows) != denominator or len(set(sample_ids)) != denominator:
        raise FormalRunDenied("P5 primary decisions do not exactly cover validation")
    sample_id_set = set(sample_ids)
    features = load_feature_index(
        str(bindings["validation_features"]["path"]), sample_id_set
    )
    corrected_labels = load_label_index(
        str(bindings["corrected_labels"]["path"]), sample_id_set
    )
    legacy_labels = load_label_index(
        str(bindings["legacy_labels"]["path"]), sample_id_set
    )
    split_payload = load_json_object(str(bindings["validation_split_manifest"]["path"]))
    split_rows = {
        str(row.get("sample_id")): row
        for row in split_payload.get("rows", [])
        if isinstance(row, Mapping) and str(row.get("sample_id")) in sample_id_set
    }
    if set(split_rows) != sample_id_set:
        raise FormalRunDenied("P5 bound split manifest does not cover validation decisions")
    pair_rows = [
        row
        for row in pq.read_table(str(bindings["pair_scores"]["path"])).to_pylist()
        if str(row.get("method")) == primary_method
    ]
    pairs_by_sample: dict[str, list[Mapping[str, Any]]] = {}
    pair_keys: set[tuple[str, str]] = set()
    for row in pair_rows:
        key = (str(row.get("sample_id")), str(row.get("challenger_id")))
        if key in pair_keys:
            raise FormalRunDenied("P5 primary pair scores are duplicated")
        pair_keys.add(key)
        pairs_by_sample.setdefault(key[0], []).append(row)
    if set(pairs_by_sample) != sample_id_set:
        raise FormalRunDenied("P5 primary pair scores do not cover validation")

    query_rows_payload = pq.read_table(
        str(bindings["sample_query_gate"]["path"])
    ).to_pylist()
    query_rows = {str(row["sample_id"]): row for row in query_rows_payload}
    if len(query_rows) != len(query_rows_payload) or set(query_rows) != sample_id_set:
        raise FormalRunDenied("P5 bound query gate does not exactly cover validation")
    original_outputs = _response_index(
        Path(str(bindings["original_responses"]["path"])),
        expected_variant="original",
    )
    confirmation_outputs = _response_index(
        Path(str(bindings["confirmation_responses"]["path"])),
        expected_variant="panel_swap",
    )
    method_models = METHOD_MODELS[primary_method]
    risk_bundle = _bundle(method["dual_risk"])
    thresholds = method["selected_thresholds"]
    feature_names = list(method["feature_names"])
    local_query_payload = load_json_object(
        str(sidecar_files["local_query_model"]["path"])
    )
    local_query_model = _model(local_query_payload["query_model"])
    eligible_methods = sorted(
        name
        for name, payload in methods.items()
        if isinstance(payload, Mapping) and payload.get("threshold_state") == "eligible"
    )
    global_query_threshold = min(
        float(methods[name]["selected_thresholds"]["query_threshold"])
        for name in eligible_methods
    )
    p5_inference = load_json_object(str(bindings["inference_manifest"]["path"]))
    if (
        p5_inference.get("eligible_methods") != eligible_methods
        or p5_inference.get("query_call_threshold") != global_query_threshold
    ):
        raise FormalRunDenied("P5 query/inference plan differs from frozen calibration")
    expected_inference_keys: set[tuple[str, str]] = set()

    trusted_rows: list[dict[str, Any]] = []
    for decision in decision_rows:
        sample_id = str(decision["sample_id"])
        feature = features[sample_id]
        candidates = ordered_candidates(feature)
        candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
        baseline_id = str(candidates[0]["candidate_id"])
        expected_challengers = set(select_challengers(feature, maximum=2))
        sample_pairs = pairs_by_sample[sample_id]
        if {str(row["challenger_id"]) for row in sample_pairs} != expected_challengers:
            raise FormalRunDenied("P5 pair scores differ from frozen challengers")
        query_names, query_values = stored_local_feature_vector(
            feature, select_challengers(feature, maximum=2)[0]
        )
        if list(query_names) != list(local_query_model.feature_names):
            raise FormalRunDenied("P5 query feature schema drifted")
        recomputed_query_score = float(local_query_model.predict_proba(query_values)[0])
        query_score = float(query_rows[sample_id]["query_score"])
        api_called = bool(query_rows[sample_id]["query_gate_called"])
        recomputed_api_called = bool(
            eligible_methods and recomputed_query_score >= global_query_threshold
        )
        if abs(query_score - recomputed_query_score) > 1e-12 or api_called != recomputed_api_called:
            raise FormalRunDenied("P5 query gate does not recompute from frozen local model")
        if api_called:
            expected_inference_keys.update(
                (sample_id, challenger_id) for challenger_id in expected_challengers
            )
        for pair_score in sample_pairs:
            challenger_id = str(pair_score["challenger_id"])
            local_names, local_values = stored_local_feature_vector(feature, challenger_id)
            key = (sample_id, challenger_id)
            original = {
                model_name: original_outputs.get((*key, model_name))
                for model_name in MODELS
            }
            confirmation = {
                model_name: confirmation_outputs.get((*key, model_name))
                for model_name in MODELS
            }
            selected_original = {
                model_name: original[model_name] if model_name in method_models else None
                for model_name in MODELS
            }
            critic_names, critic_values = critic_feature_vector(selected_original)
            if list(local_names) + list(critic_names) != feature_names:
                raise FormalRunDenied("P5 pair-score feature schema drifted")
            p_benefit, p_harm = risk_bundle.predict(
                np.r_[local_values, critic_values]
            )
            selected_confirmation = {
                model_name: confirmation[model_name] if model_name in method_models else None
                for model_name in MODELS
            }
            repeated_names, repeated_values = critic_feature_vector(selected_confirmation)
            if repeated_names != critic_names:
                raise FormalRunDenied("P5 confirmation feature schema drifted")
            confirmation_p_benefit, confirmation_p_harm = risk_bundle.predict(
                np.r_[local_values, repeated_values]
            )
            support, stable, hard_veto = _support_and_confirmation(
                original, confirmation, method_models
            )
            hard_valid = bool(
                build_pair_evidence(feature, challenger_id, load_depth=False)["challenger"]["hard_valid"]
            )
            expected_eligible = bool(
                method.get("threshold_state") == "eligible"
                and api_called
                and query_score >= float(thresholds["query_threshold"])
                and support
                and stable
                and hard_valid
                and not hard_veto
                and float(p_benefit[0]) >= float(thresholds["tau"])
                and float(p_harm[0]) <= float(thresholds["eta"])
                and float(confirmation_p_benefit[0]) >= float(thresholds["tau"])
                and float(confirmation_p_harm[0]) <= float(thresholds["eta"])
            )
            expected_fields: dict[str, Any] = {
                "query_score": query_score,
                "critic_support": support,
                "confirmation_stable": stable,
                "challenger_hard_valid": hard_valid,
                "hard_veto": hard_veto,
                "api_called": api_called,
                "eligible": expected_eligible,
                "p_benefit": float(p_benefit[0]),
                "p_harm": float(p_harm[0]),
                "confirmation_p_benefit": float(confirmation_p_benefit[0]),
                "confirmation_p_harm": float(confirmation_p_harm[0]),
            }
            for field, expected in expected_fields.items():
                actual = pair_score.get(field)
                if isinstance(expected, float):
                    if actual is None or abs(float(actual) - expected) > 1e-12:
                        raise FormalRunDenied(f"P5 pair score does not recompute: {field}")
                elif actual != expected:
                    raise FormalRunDenied(f"P5 pair score does not recompute: {field}")
        eligible_pairs = [row for row in sample_pairs if bool(row.get("eligible"))]
        chosen = sorted(
            eligible_pairs,
            key=lambda row: (
                -float(row["p_benefit"]),
                float(row["p_harm"]),
                str(row["challenger_id"]),
            ),
        )[0] if eligible_pairs else None
        selected_id = baseline_id if chosen is None else str(chosen["challenger_id"])
        if selected_id not in candidate_ids:
            raise FormalRunDenied("P5 selected candidate is not frozen Top-K evidence")
        if (
            str(decision.get("baseline_candidate_id")) != baseline_id
            or str(decision.get("selected_candidate_id")) != selected_id
            or bool(decision.get("switched")) != (selected_id != baseline_id)
        ):
            raise FormalRunDenied("P5 decision does not recompute from bound pair scores")
        corrected = corrected_labels[sample_id]
        legacy = legacy_labels[sample_id]
        split_row = split_rows[sample_id]
        trusted = dict(decision)
        trusted.update({
            "group_id": str(split_row["sequence_id"]),
            "frame_id": str(split_row["frame_id"]),
            "baseline_correct": bool(corrected[baseline_id]),
            "selected_correct": bool(corrected[selected_id]),
            "legacy_baseline_correct": bool(legacy[baseline_id]),
            "legacy_selected_correct": bool(legacy[selected_id]),
        })
        for field in (
            "group_id",
            "frame_id",
            "baseline_correct",
            "selected_correct",
            "legacy_baseline_correct",
            "legacy_selected_correct",
        ):
            if decision.get(field) != trusted[field]:
                raise FormalRunDenied(f"P5 saved decision disagrees with bound truth: {field}")
        trusted_rows.append(trusted)

    inference_rows = p5_inference.get("rows")
    if not isinstance(inference_rows, list):
        raise FormalRunDenied("P5 inference rows are invalid")
    actual_inference_keys = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]))
        for row in inference_rows
    }
    if len(actual_inference_keys) != len(inference_rows) or actual_inference_keys != expected_inference_keys:
        raise FormalRunDenied("P5 inference requests do not match the recomputed query gate")

    validation_identity = compute_candidate_q_identity(
        str(bindings["validation_features"]["path"])
    )
    for field in (
        "candidate_identity_sha256",
        "q_value_sha256",
        "combined_identity_sha256",
    ):
        if validation.get(field) != validation_identity[field]:
            raise FormalRunDenied(f"P5 validation {field} does not recompute")
    corrected = full_denominator_metrics(
        trusted_rows,
        baseline_key="baseline_correct",
        selected_key="selected_correct",
    )
    legacy = full_denominator_metrics(
        trusted_rows,
        baseline_key="legacy_baseline_correct",
        selected_key="legacy_selected_correct",
    )
    if corrected != validation_method.get("corrected") or legacy != validation_method.get("legacy"):
        raise FormalRunDenied("P5 validation metrics do not recompute from bound decisions")
    seed = validation.get("statistical_seed")
    draws = validation.get("bootstrap_draws")
    if seed != 20260803 or draws != 10000:
        raise FormalRunDenied("P5 validation statistical contract drifted")
    interval = _bootstrap_delta(trusted_rows, draws=draws, seed=seed)
    if interval != validation_method.get("scene_sequence_bootstrap_delta_j1"):
        raise FormalRunDenied("P5 validation clustered interval does not recompute")
    hard_constraints = bool(
        corrected["net"] > 0
        and corrected["recovered"] > corrected["harmful"]
        and corrected["final_j1"] >= corrected["baseline_j1"]
        and corrected["outcome_changing_precision"] >= .67
        and corrected["harm_rate"] <= .01
        and validation_method.get("stability_contract_passed") is True
    )
    if not hard_constraints or interval["lower"] < 0.0:
        raise FormalRunDenied("P5 primary fails recomputed hard constraints or CI gate")


def _verify_calibration_evidence_bindings(calibration: Mapping[str, Any]) -> None:
    bindings = calibration.get("calibration_evidence_bindings")
    required = {
        "source_inference_sidecar",
        "source_inference_manifest",
        "source_sample_manifest",
        "source_evaluation_manifest",
        "source_original_responses",
        "confirmation_inference_sidecar",
        "confirmation_inference_manifest",
        "confirmation_responses",
        "local_query_oof",
        "local_query_model",
        "train_features",
        "cohort_source",
        "diagnostic_stability",
        "calibration_oof_pairs",
        "calibration_threshold_sweep",
        "evaluator_source",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != required:
        raise FormalRunDenied("P5 calibration has no complete frozen evidence binding set")
    try:
        for identity in bindings.values():
            if not isinstance(identity, Mapping):
                raise ManifestError("invalid calibration evidence file identity")
            verify_file_identity(identity)
    except ManifestError as exc:
        raise FormalRunDenied(f"P5 calibration evidence drift: {exc}") from exc
    for prefix in ("source", "confirmation"):
        sidecar = load_json_object(str(bindings[f"{prefix}_inference_sidecar"]["path"]))
        if sidecar.get("sha256") != bindings[f"{prefix}_inference_manifest"].get("sha256"):
            raise FormalRunDenied(f"P5 calibration {prefix} inference binding drifted")
    source_manifest = load_json_object(str(bindings["source_inference_manifest"]["path"]))
    if Path(str(source_manifest.get("feature_file", ""))).resolve() != Path(
        str(bindings["train_features"]["path"])
    ).resolve():
        raise FormalRunDenied("P5 calibration inference uses an unbound feature source")
    diagnostic = calibration.get("diagnostic_stability")
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("sha256") != bindings["diagnostic_stability"].get("sha256")
    ):
        raise FormalRunDenied("P5 calibration diagnostic binding disagrees with provenance")
    query_contract = calibration.get("query_gate_contract")
    if not isinstance(query_contract, Mapping):
        raise FormalRunDenied("P5 calibration query contract is missing")
    if (
        query_contract.get("local_only_oof_pairs_sha256")
        != bindings["local_query_oof"].get("sha256")
        or query_contract.get("local_only_model_sha256")
        != bindings["local_query_model"].get("sha256")
    ):
        raise FormalRunDenied("P5 calibration query lineage disagrees with provenance")
    if query_contract.get("sampling_contract_passed") is not True:
        raise FormalRunDenied("P5 calibration cohort/query sampling contract did not pass")

    import pyarrow.parquet as pq

    sample_manifest = load_json_object(str(bindings["source_sample_manifest"]["path"]))
    samples = sample_manifest.get("samples")
    if not isinstance(samples, list):
        raise FormalRunDenied("P5 calibration sample manifest is invalid")
    sample_ids = [str(row.get("sample_id")) for row in samples if isinstance(row, Mapping)]
    if len(sample_ids) != 150 or len(set(sample_ids)) != 150:
        raise FormalRunDenied("P5 calibration does not bind 150 unique samples")
    natural_rows = [
        row
        for row in pq.read_table(str(bindings["cohort_source"]["path"])).to_pylist()
        if str(row.get("partition")) == "calibration"
    ]
    natural_joint = {
        (str(row["corrected_cohort"]), str(row["query_type"]))
        for row in natural_rows
    }
    sampled_joint = {
        (str(row["corrected_cohort"]), str(row["query_type"]))
        for row in samples
        if isinstance(row, Mapping)
    }
    unsupported_joint = sorted(natural_joint - sampled_joint)
    if unsupported_joint:
        raise FormalRunDenied(
            "P5 calibration omits natural cohort/query strata and cannot authorise formal"
        )
    inference_rows = source_manifest.get("rows")
    if not isinstance(inference_rows, list):
        raise FormalRunDenied("P5 calibration inference rows are invalid")
    evaluation_rows = pq.read_table(
        str(bindings["source_evaluation_manifest"]["path"])
    ).to_pylist()
    inference_keys = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]))
        for row in inference_rows
    }
    evaluation_keys = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]))
        for row in evaluation_rows
    }
    if (
        len(inference_rows) != 300
        or len(inference_keys) != 300
        or evaluation_keys != inference_keys
        or {sample_id for sample_id, _ in inference_keys} != set(sample_ids)
    ):
        raise FormalRunDenied("P5 calibration sample/pair coverage drifted")
    from .api_calibration import MODELS, _response_index

    originals = _response_index(
        Path(str(bindings["source_original_responses"]["path"])),
        expected_variant="original",
    )
    expected_originals = {
        (*key, model) for key in inference_keys for model in MODELS
    }
    if set(originals) != expected_originals:
        raise FormalRunDenied("P5 calibration original response coverage drifted")
    confirmation_manifest = load_json_object(
        str(bindings["confirmation_inference_manifest"]["path"])
    )
    confirmation_rows = confirmation_manifest.get("rows")
    if not isinstance(confirmation_rows, list):
        raise FormalRunDenied("P5 calibration confirmation manifest is invalid")
    confirmation_keys = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]))
        for row in confirmation_rows
    }
    confirmations = _response_index(
        Path(str(bindings["confirmation_responses"]["path"])),
        expected_variant="panel_swap",
    )
    expected_confirmations = {
        (*key, model) for key in confirmation_keys for model in MODELS
    }
    if set(confirmations) != expected_confirmations:
        raise FormalRunDenied("P5 calibration confirmation response coverage drifted")
    oof_rows = pq.read_table(str(bindings["calibration_oof_pairs"]["path"])).to_pylist()
    expected_oof = {
        (method_name, *key)
        for method_name in calibration.get("methods", {})
        for key in inference_keys
    }
    actual_oof = {
        (str(row["method"]), str(row["sample_id"]), str(row["challenger_id"]))
        for row in oof_rows
    }
    if len(oof_rows) != len(actual_oof) or actual_oof != expected_oof:
        raise FormalRunDenied("P5 calibration OOF evidence coverage drifted")
    sweep_rows = pq.read_table(
        str(bindings["calibration_threshold_sweep"]["path"])
    ).to_pylist()
    sweep_by_method: dict[str, list[Mapping[str, Any]]] = {}
    for row in sweep_rows:
        sweep_by_method.setdefault(str(row["method"]), []).append(row)
    methods = calibration.get("methods")
    if not isinstance(methods, Mapping) or set(sweep_by_method) != set(methods):
        raise FormalRunDenied("P5 calibration threshold-sweep coverage drifted")
    for method_name, payload in methods.items():
        if not isinstance(payload, Mapping):
            raise FormalRunDenied("P5 calibration method payload is invalid")
        selected = payload.get("selected_thresholds")
        if payload.get("threshold_state") == "eligible":
            if not isinstance(selected, Mapping) or not any(
                all(row.get(field) == selected.get(field) for field in ("query_threshold", "tau", "eta"))
                for row in sweep_by_method[str(method_name)]
            ):
                raise FormalRunDenied("P5 selected threshold is absent from the frozen sweep")


def _verify_diagnostic_stability_evidence(
    diagnostic: Mapping[str, Any], required_models: Sequence[str]
) -> None:
    bindings = diagnostic.get("diagnostic_evidence_bindings")
    required = {
        "source_inference_sidecar",
        "source_inference_manifest",
        "source_sample_manifest",
        "source_evaluation_manifest",
        "source_responses",
        "source_features",
        "pairwise_outcomes",
        "sample_outcomes",
        "evaluator_source",
        "perturbation_inference_sidecar",
        "perturbation_inference_manifest",
        "perturbation_responses",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != required:
        raise FormalRunDenied("P5 diagnostic has no complete raw evidence binding set")
    try:
        for identity in bindings.values():
            if not isinstance(identity, Mapping):
                raise ManifestError("invalid diagnostic evidence file identity")
            verify_file_identity(identity)
    except ManifestError as exc:
        raise FormalRunDenied(f"P5 diagnostic raw evidence drift: {exc}") from exc
    for prefix in ("source", "perturbation"):
        sidecar = load_json_object(str(bindings[f"{prefix}_inference_sidecar"]["path"]))
        if sidecar.get("sha256") != bindings[f"{prefix}_inference_manifest"].get("sha256"):
            raise FormalRunDenied(f"P5 diagnostic {prefix} inference binding drifted")

    import pyarrow.parquet as pq

    from .api import MODEL_IDS
    from .critic_evaluation import STABILITY_CONTRACT
    from .critic_features import p3_hard_rule

    evaluation = pq.read_table(str(bindings["source_evaluation_manifest"]["path"])).to_pylist()
    original = pq.read_table(str(bindings["source_responses"]["path"])).to_pylist()
    perturbation = pq.read_table(str(bindings["perturbation_responses"]["path"])).to_pylist()
    expected_pairs = {
        (str(row["sample_id"]), str(row["challenger_candidate_id"]))
        for row in evaluation
    }
    expected_original = {
        (*pair, model) for pair in expected_pairs for model in MODEL_IDS
    }
    original_index = {
        (
            str(row["sample_id"]),
            str(row["challenger_candidate_id"]),
            str(row["model_id"]),
        ): row
        for row in original
    }
    if len(original_index) != len(original) or set(original_index) != expected_original:
        raise FormalRunDenied("P5 diagnostic original response coverage drifted")
    variants = set(STABILITY_CONTRACT["required_variants"])
    expected_perturbation = {
        (*key, variant) for key in expected_original for variant in variants
    }
    perturbation_index = {
        (
            str(row["sample_id"]),
            str(row["challenger_candidate_id"]),
            str(row["model_id"]),
            str(row["variant"]),
        ): row
        for row in perturbation
    }
    if (
        len(perturbation_index) != len(perturbation)
        or set(perturbation_index) != expected_perturbation
    ):
        raise FormalRunDenied("P5 diagnostic perturbation response coverage drifted")

    claimed_models = diagnostic.get("models")
    if not isinstance(claimed_models, Mapping):
        raise FormalRunDenied("P5 diagnostic model summaries are missing")
    for model in required_models:
        recomputed: dict[str, dict[str, Any]] = {}
        for variant in sorted(variants):
            paired = [
                (
                    original_index[(*pair, model)],
                    perturbation_index[(*pair, model, variant)],
                )
                for pair in sorted(expected_pairs)
            ]
            valid = [
                (first, second)
                for first, second in paired
                if first.get("status") == "SUCCEEDED"
                and first.get("parsed") is not None
                and second.get("status") == "SUCCEEDED"
                and second.get("parsed") is not None
            ]
            hard_consistency = (
                sum(
                    p3_hard_rule(first.get("parsed"))
                    == p3_hard_rule(second.get("parsed"))
                    for first, second in valid
                ) / len(valid)
                if valid else None
            )
            direction_flip = (
                sum(
                    bool(first.get("parsed") and first["parsed"].get("decision") == "PREFER_CHALLENGER")
                    != bool(second.get("parsed") and second["parsed"].get("decision") == "PREFER_CHALLENGER")
                    for first, second in valid
                ) / len(valid)
                if valid else None
            )
            critic_consistency = None if direction_flip is None else 1.0 - direction_flip
            recomputed[variant] = {
                "pairs": len(paired),
                "valid_pairs": len(valid),
                "valid_pair_coverage": len(valid) / len(paired) if paired else 0.0,
                "critic_choice_consistency": critic_consistency,
                "hard_rule_consistency": hard_consistency,
                "direction_flip_rate": direction_flip,
            }
        claimed = claimed_models.get(model)
        if not isinstance(claimed, Mapping) or claimed.get("stability") != recomputed:
            raise FormalRunDenied("P5 diagnostic stability does not recompute from raw responses")
        passed = all(
            row["hard_rule_consistency"] is not None
            and row["valid_pair_coverage"] >= STABILITY_CONTRACT["minimum_valid_pair_coverage"]
            and row["hard_rule_consistency"] >= STABILITY_CONTRACT["minimum_hard_rule_consistency"]
            and row["direction_flip_rate"] is not None
            and row["direction_flip_rate"] <= STABILITY_CONTRACT["maximum_direction_flip_rate"]
            for row in recomputed.values()
        )
        if not passed or claimed.get("stability_passed") is not True:
            raise FormalRunDenied("P5 primary model fails recomputed perturbation stability")


def write_locked_manifest(
    path: str | Path,
    *,
    run_id: str,
    protocol_lock: str | Path,
    validation_result: str | Path,
    inference_manifest: str | Path,
    expected_denominator: int,
    primary_method: str | None = None,
    validation_status: str | None = None,
) -> dict[str, Any]:
    """Create LOCKED_MANIFEST only for an eligible, validated non-q primary."""

    denominator = _positive_int(expected_denominator, "expected_denominator")
    protocol = verify_protocol_lock(protocol_lock)
    inference = verify_inference_manifest(inference_manifest)
    validation, status, primary = _validation_binding(
        validation_result,
        primary_method=primary_method,
        validation_status=validation_status,
    )
    if protocol.get("expected_denominator") != denominator:
        raise ManifestError("PROTOCOL_LOCK expected denominator mismatch")
    if inference.get("expected_denominator") != denominator:
        raise ManifestError("inference manifest expected denominator mismatch")
    validation_denominator = _positive_int(
        validation.get("expected_denominator"), "validation expected_denominator"
    )
    if primary not in protocol.get("eligible_primary_methods", []):
        raise FormalRunDenied("validation primary is not eligible under PROTOCOL_LOCK")
    preselected = protocol.get("preselected_primary_method")
    if preselected is not None and preselected != primary:
        raise FormalRunDenied("validation primary differs from preselected primary")

    protocol_inference = protocol.get("inference_manifest")
    if not isinstance(protocol_inference, Mapping):
        raise ManifestError("PROTOCOL_LOCK inference binding is invalid")
    _assert_same_file(protocol_inference, inference_manifest, "inference manifest")
    if protocol.get("inference_manifest_content_sha256") != inference.get("content_sha256"):
        raise ManifestError("inference manifest semantic hash drift")

    validation_identity: dict[str, str] = {}
    for field in (
        "candidate_identity_sha256",
        "q_value_sha256",
        "combined_identity_sha256",
    ):
        validation_identity[field] = _sha256(
            validation.get(field), f"validation {field}"
        )

    payload: dict[str, Any] = {
        "run_id": _nonempty(run_id, "run_id"),
        "expected_denominator": denominator,
        "validation_expected_denominator": validation_denominator,
        "validation_status": status,
        "primary_method": primary,
        "protocol_lock": file_identity(protocol_lock),
        "protocol_lock_content_sha256": protocol["content_sha256"],
        "validation_result": file_identity(validation_result),
        "inference_manifest": file_identity(inference_manifest),
        "inference_manifest_content_sha256": inference["content_sha256"],
        "candidate_identity_sha256": protocol["candidate_identity_sha256"],
        "q_value_sha256": protocol["q_value_sha256"],
        "combined_identity_sha256": protocol["combined_identity_sha256"],
        "validation_candidate_identity_sha256": validation_identity[
            "candidate_identity_sha256"
        ],
        "validation_q_value_sha256": validation_identity["q_value_sha256"],
        "validation_combined_identity_sha256": validation_identity[
            "combined_identity_sha256"
        ],
    }
    return write_layer(path, LOCKED_MANIFEST_KIND, payload)


def verify_locked_manifest(path: str | Path) -> dict[str, Any]:
    locked = verify_layer(path, expected_kind=LOCKED_MANIFEST_KIND)
    if locked.get("validation_status") != "GO":
        raise FormalRunDenied("LOCKED_MANIFEST validation_status must be exactly 'GO'")
    primary = _nonempty(locked.get("primary_method"), "primary_method")
    if _is_q_only(primary):
        raise FormalRunDenied("LOCKED_MANIFEST q-only primary cannot authorise formal API use")
    denominator = _positive_int(locked.get("expected_denominator"), "expected_denominator")

    protocol_identity = locked.get("protocol_lock")
    validation_identity = locked.get("validation_result")
    inference_identity = locked.get("inference_manifest")
    if not all(
        isinstance(item, Mapping)
        for item in (protocol_identity, validation_identity, inference_identity)
    ):
        raise ManifestError("LOCKED_MANIFEST has an invalid artifact binding")

    protocol = verify_protocol_lock(str(protocol_identity.get("path", "")))
    inference = verify_inference_manifest(str(inference_identity.get("path", "")))
    validation = load_json_object(str(validation_identity.get("path", "")))
    if protocol.get("content_sha256") != locked.get("protocol_lock_content_sha256"):
        raise ManifestError("PROTOCOL_LOCK semantic hash drift")
    if inference.get("content_sha256") != locked.get("inference_manifest_content_sha256"):
        raise ManifestError("inference manifest semantic hash drift")
    if protocol.get("expected_denominator") != denominator:
        raise ManifestError("PROTOCOL_LOCK denominator drift")
    if inference.get("expected_denominator") != denominator:
        raise ManifestError("inference manifest denominator drift")

    if validation.get("validation_status") != "GO":
        raise FormalRunDenied("validation artifact no longer has validation_status='GO'")
    if validation.get("primary_method") != primary:
        raise FormalRunDenied("validation primary/LOCKED_MANIFEST primary mismatch")
    if primary.lower().startswith("p5"):
        _verify_p5_validation_evidence(validation, primary)
    validation_denominator = _positive_int(
        validation.get("expected_denominator"), "validation expected_denominator"
    )
    if validation_denominator != locked.get("validation_expected_denominator"):
        raise FormalRunDenied("validation denominator/LOCKED_MANIFEST binding mismatch")
    if primary not in protocol.get("eligible_primary_methods", []):
        raise FormalRunDenied("LOCKED_MANIFEST primary is not protocol-eligible")
    preselected = protocol.get("preselected_primary_method")
    if preselected is not None and preselected != primary:
        raise FormalRunDenied("LOCKED_MANIFEST primary differs from preselected primary")

    for field in (
        "candidate_identity_sha256",
        "q_value_sha256",
        "combined_identity_sha256",
    ):
        if locked.get(field) != protocol.get(field):
            raise ManifestError(f"LOCKED_MANIFEST {field} differs from PROTOCOL_LOCK")
        validation_field = f"validation_{field}"
        if validation.get(field) != locked.get(validation_field):
            raise FormalRunDenied(
                f"validation {field} differs from LOCKED_MANIFEST validation binding"
            )
    return locked


def assert_formal_run_allowed(
    locked_manifest: str | Path,
    *,
    cli_allow_formal: bool,
    environ: Mapping[str, str] | None = None,
    expected_denominator: int | None = None,
    primary_method: str | None = None,
) -> dict[str, Any]:
    """Enforce both explicit opt-ins and re-verify the entire lock chain."""

    if cli_allow_formal is not True:
        raise FormalRunDenied("formal API run requires an explicit CLI allow value")
    environment = os.environ if environ is None else environ
    if environment.get(FORMAL_ENVIRONMENT_VARIABLE) != "1":
        raise FormalRunDenied(
            f"formal API run requires {FORMAL_ENVIRONMENT_VARIABLE} to equal exactly '1'"
        )

    before = sha256_file(locked_manifest)
    locked = verify_locked_manifest(locked_manifest)
    if sha256_file(locked_manifest) != before:
        raise ManifestError("LOCKED_MANIFEST changed while it was being verified")
    if expected_denominator is not None:
        denominator = _positive_int(expected_denominator, "expected_denominator")
        if locked.get("expected_denominator") != denominator:
            raise FormalRunDenied("requested expected denominator differs from lock")
    if primary_method is not None and locked.get("primary_method") != primary_method:
        raise FormalRunDenied("requested primary method differs from lock")
    if locked.get("primary_method") not in SUPPORTED_FORMAL_PRIMARY_METHODS:
        raise FormalRunDenied(
            "formal API execution supports only an evidence-recomputed P5 primary"
        )
    return locked


def claim_formal_run_once(
    claim_path: str | Path,
    locked_manifest: str | Path,
    *,
    cli_allow_formal: bool,
    environ: Mapping[str, str] | None = None,
    run_id: str | None = None,
    expected_denominator: int | None = None,
    primary_method: str | None = None,
) -> dict[str, Any]:
    """Create an exclusive formal-run claim after both gates have passed."""

    locked = assert_formal_run_allowed(
        locked_manifest,
        cli_allow_formal=cli_allow_formal,
        environ=environ,
        expected_denominator=expected_denominator,
        primary_method=primary_method,
    )
    payload = {
        "run_id": _nonempty(run_id or locked.get("run_id"), "run_id"),
        "locked_manifest": file_identity(locked_manifest),
        "locked_manifest_content_sha256": locked["content_sha256"],
        "expected_denominator": locked["expected_denominator"],
        "validation_status": locked["validation_status"],
        "primary_method": locked["primary_method"],
        "inference_manifest": locked["inference_manifest"],
        "inference_manifest_content_sha256": locked[
            "inference_manifest_content_sha256"
        ],
        "candidate_identity_sha256": locked["candidate_identity_sha256"],
        "q_value_sha256": locked["q_value_sha256"],
        "combined_identity_sha256": locked["combined_identity_sha256"],
        "formal_environment_gate": FORMAL_ENVIRONMENT_VARIABLE,
        "formal_environment_gate_value": "1",
        "explicit_cli_allow": True,
    }
    return write_layer(claim_path, FORMAL_CLAIM_KIND, payload)


# Compatibility-friendly aliases with equally strict semantics.
create_protocol_lock = write_protocol_lock
create_locked_manifest = write_locked_manifest
assert_formal_allowed = assert_formal_run_allowed
claim_formal_test_once = claim_formal_run_once


__all__ = [
    "FORMAL_CLAIM_KIND",
    "FORMAL_ENVIRONMENT_VARIABLE",
    "FormalRunDenied",
    "assert_formal_allowed",
    "assert_formal_run_allowed",
    "claim_formal_run_once",
    "claim_formal_test_once",
    "create_locked_manifest",
    "create_protocol_lock",
    "verify_locked_manifest",
    "verify_protocol_lock",
    "write_locked_manifest",
    "write_protocol_lock",
]
