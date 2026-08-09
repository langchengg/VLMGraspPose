"""Fail-closed identity contract for repeated-FiLM reranking artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .identity import sha256_file
from .method_namespace import (
    FULL_NMS_BASELINE,
    FULL_NMS_INTERNAL_METHODS,
    GQCNN_TOP5_INTERNAL_ALIASES,
    GQCNN_TOP5_BASELINE,
    LOCAL_VLM_RAW_METHODS,
    LOCAL_VLM_SAFE_SWITCH_METHOD,
    PUBLIC_METHOD_NAMES,
)


EXPERIMENT_NAME: Final[str] = "modular_reranking_repeatedfilm_v1"
PUBLIC_METHOD_NAMESPACE_VERSION: Final[int] = 1
PUBLIC_METHOD_PREFIX: Final[str] = "repeatedfilm_"
CONFIG_METHOD_NAMESPACE: Final[str] = "repeatedfilm_only"
VISUAL_GROUNDING_LINEAGE: Final[str] = "hierarchical_repeated_film"
REPEATEDFILM_SOURCE_CHECKPOINT_SHA256: Final[str] = (
    "b19a649326384ba4524295cd100b22e54cb9ea615174229fc310fbd6bc898601"
)
REPEATEDFILM_SOURCE_SNAPSHOT_SHA256: Final[str] = (
    "4e05df9f909e966e4531d9c5e36d6582e077d61400ff2d574c4114ccc2584f84"
)
LOCAL_VLM_MAXIMUM_MEMORY_MIB: Final[float] = 20_000.0
LOCAL_VLM_MAXIMUM_P95_SECONDS: Final[float] = 180.0
LOCAL_VLM_MAX_OUTPUT_TOKENS: Final[int] = 768
LOCAL_VLM_GEOMETRY_RISK_COLUMN: Final[str] = "collision_proxy_total"
LOCAL_VLM_GEOMETRY_RISK_THRESHOLD: Final[float] = 0.5
LOCAL_VLM_GEOMETRY_SEMANTICS: Final[str] = (
    "visible_surface_proxy_only_not_complete_collision_guarantee"
)
SCALER_METADATA_SCHEMA_VERSION: Final[int] = 1
SCALER_METADATA_ARTIFACT_KIND: Final[str] = (
    "final_deployment_scaler_metadata"
)
SCALER_FIT_SCOPE: Final[str] = "train/development candidates only"
SCALER_MODEL_ARTIFACT_KEYS: Final[dict[str, str]] = {
    "regularized_linear_ranker": "model_regularized_linear_ranker",
    "pairwise_ranker": "model_pairwise_ranker",
    "multi_positive_listwise_ranker": (
        "model_multi_positive_listwise_ranker"
    ),
    "residual_mlp": "model_residual_mlp",
    "set_aware_residual": "model_set_aware_residual",
}
PUBLIC_METHOD_SET: Final[frozenset[str]] = frozenset(
    {
        *PUBLIC_METHOD_NAMES.values(),
        *LOCAL_VLM_RAW_METHODS,
        LOCAL_VLM_SAFE_SWITCH_METHOD,
    }
)
PROTOCOL_BASELINES: Final[dict[str, str]] = {
    "full_nms": FULL_NMS_BASELINE,
    "gqcnn_top5": GQCNN_TOP5_BASELINE,
}
CONFIG_PROTOCOL_METHODS: Final[dict[str, tuple[str, ...]]] = {
    "full_nms": tuple(
        PUBLIC_METHOD_NAMES[method] for method in FULL_NMS_INTERNAL_METHODS
    ),
    "gqcnn_top5": (
        *(
            PUBLIC_METHOD_NAMES[alias]
            for alias in GQCNN_TOP5_INTERNAL_ALIASES.values()
        ),
        *sorted(LOCAL_VLM_RAW_METHODS),
        LOCAL_VLM_SAFE_SWITCH_METHOD,
    ),
}


def local_vlm_preregistration_payload(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact preregistered VLM resource and geometry contract."""

    local_vlm = value.get("local_vlm")
    if not isinstance(local_vlm, Mapping):
        raise ValueError("config must preregister local_vlm")
    safe_switch = local_vlm.get("safe_switch")
    if not isinstance(safe_switch, Mapping):
        raise ValueError("config local_vlm must preregister safe_switch geometry")
    observed = {
        "maximum_memory_mib": local_vlm.get("maximum_memory_mib"),
        "maximum_p95_seconds": local_vlm.get("maximum_p95_seconds"),
        "max_output_tokens": local_vlm.get("max_output_tokens"),
        "geometry_risk_column": safe_switch.get("geometry_risk_column"),
        "geometry_risk_threshold": safe_switch.get(
            "geometry_risk_threshold"
        ),
        "geometry_semantics": safe_switch.get("geometry_semantics"),
    }
    expected = {
        "maximum_memory_mib": LOCAL_VLM_MAXIMUM_MEMORY_MIB,
        "maximum_p95_seconds": LOCAL_VLM_MAXIMUM_P95_SECONDS,
        "max_output_tokens": LOCAL_VLM_MAX_OUTPUT_TOKENS,
        "geometry_risk_column": LOCAL_VLM_GEOMETRY_RISK_COLUMN,
        "geometry_risk_threshold": LOCAL_VLM_GEOMETRY_RISK_THRESHOLD,
        "geometry_semantics": LOCAL_VLM_GEOMETRY_SEMANTICS,
    }
    if observed != expected:
        raise ValueError(
            "config local_vlm resource/generation/safe-switch geometry "
            f"contract must exactly equal {expected}"
        )
    return dict(expected)


def normalize_scaler_payload(
    value: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Validate and normalize one final deployment scaler semantically."""

    required = {
        "feature_columns",
        "mean",
        "scale",
        "source_splits",
        "fit_scope",
    }
    if set(value) != required:
        raise ValueError(
            f"{context} scaler fields must exactly equal {sorted(required)}"
        )
    feature_columns = value["feature_columns"]
    mean = value["mean"]
    scale = value["scale"]
    source_splits = value["source_splits"]
    for name, item in (
        ("feature_columns", feature_columns),
        ("mean", mean),
        ("scale", scale),
        ("source_splits", source_splits),
    ):
        if not isinstance(item, Sequence) or isinstance(
            item, (str, bytes, bytearray)
        ):
            raise ValueError(f"{context} scaler {name} must be a sequence")
    if (
        not feature_columns
        or any(
            not isinstance(column, str) or not column
            for column in feature_columns
        )
        or len(feature_columns) != len(set(feature_columns))
    ):
        raise ValueError(
            f"{context} scaler feature_columns must be non-empty and unique"
        )
    if (
        len(mean) != len(feature_columns)
        or len(scale) != len(feature_columns)
        or any(isinstance(item, bool) for item in (*mean, *scale))
    ):
        raise ValueError(
            f"{context} scaler statistics must match the feature order"
        )
    try:
        normalized_mean = [float(item) for item in mean]
        normalized_scale = [float(item) for item in scale]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context} scaler statistics must be numeric"
        ) from error
    if (
        any(not math.isfinite(item) for item in normalized_mean)
        or any(
            not math.isfinite(item) or item <= 0.0
            for item in normalized_scale
        )
    ):
        raise ValueError(
            f"{context} scaler statistics must be finite with positive scale"
        )
    normalized_splits = list(source_splits)
    if (
        not normalized_splits
        or any(
            not isinstance(split, str)
            or split not in {"train", "development"}
            for split in normalized_splits
        )
        or normalized_splits != sorted(set(normalized_splits))
    ):
        raise ValueError(
            f"{context} scaler source_splits must be sorted development splits"
        )
    if value["fit_scope"] != SCALER_FIT_SCOPE:
        raise ValueError(
            f"{context} scaler fit_scope must equal {SCALER_FIT_SCOPE!r}"
        )
    return {
        "feature_columns": list(feature_columns),
        "mean": normalized_mean,
        "scale": normalized_scale,
        "source_splits": normalized_splits,
        "fit_scope": SCALER_FIT_SCOPE,
    }


def scaler_payload_sha256(
    value: Mapping[str, Any], *, context: str
) -> str:
    """Hash a scaler after semantic normalization."""

    normalized = normalize_scaler_payload(value, context=context)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_final_scaler_metadata(
    model_artifacts: Mapping[str, Mapping[str, Any]],
    model_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Aggregate the identical scalers embedded in all deployment models."""

    expected_methods = tuple(SCALER_MODEL_ARTIFACT_KEYS)
    if set(model_artifacts) != set(expected_methods) or set(
        model_paths
    ) != set(expected_methods):
        raise ValueError(
            "final scaler metadata requires exact deployment-model coverage"
        )
    normalized_by_method: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for method in expected_methods:
        artifact = model_artifacts[method]
        if artifact.get("method") != method:
            raise ValueError(
                f"deployment model artifact method mismatch: {method}"
            )
        scaler = artifact.get("scaler")
        if not isinstance(scaler, Mapping):
            raise ValueError(
                f"deployment model {method} omits its fitted scaler"
            )
        normalized_by_method[method] = normalize_scaler_payload(
            scaler, context=f"deployment model {method}"
        )
        path = model_paths[method].expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                f"deployment model artifact is missing: {path}"
            )
        paths[method] = path
    common = normalized_by_method[expected_methods[0]]
    if any(
        normalized_by_method[method] != common
        for method in expected_methods[1:]
    ):
        raise ValueError(
            "final deployment models do not share one identical scaler"
        )
    scaler_sha256 = scaler_payload_sha256(
        common, context="final deployment scaler"
    )
    return {
        "schema_version": SCALER_METADATA_SCHEMA_VERSION,
        **identity_payload(),
        "artifact_kind": SCALER_METADATA_ARTIFACT_KIND,
        "model_methods": list(expected_methods),
        "model_count": len(expected_methods),
        "feature_columns": list(common["feature_columns"]),
        "scaler": common,
        "scaler_sha256": scaler_sha256,
        "all_model_scalers_identical": True,
        "models": {
            method: {
                "artifact_key": SCALER_MODEL_ARTIFACT_KEYS[method],
                "artifact_path": str(paths[method]),
                "artifact_sha256": sha256_file(paths[method]),
                "scaler_sha256": scaler_sha256,
            }
            for method in expected_methods
        },
    }


def validate_scaler_metadata_structure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate canonical scaler metadata before checking model files."""

    validate_artifact_identity(value, context="scaler metadata")
    expected_methods = tuple(SCALER_MODEL_ARTIFACT_KEYS)
    models = value.get("models")
    scaler = value.get("scaler")
    if (
        value.get("schema_version") != SCALER_METADATA_SCHEMA_VERSION
        or value.get("artifact_kind") != SCALER_METADATA_ARTIFACT_KIND
        or value.get("model_methods") != list(expected_methods)
        or value.get("model_count") != len(expected_methods)
        or value.get("all_model_scalers_identical") is not True
        or not isinstance(models, Mapping)
        or set(models) != set(expected_methods)
        or not isinstance(scaler, Mapping)
    ):
        raise ValueError(
            "scaler metadata is not the final deployment scaler artifact"
        )
    normalized = normalize_scaler_payload(
        scaler, context="scaler metadata"
    )
    scaler_sha256 = scaler_payload_sha256(
        normalized, context="scaler metadata"
    )
    if (
        value.get("feature_columns") != normalized["feature_columns"]
        or value.get("scaler_sha256") != scaler_sha256
    ):
        raise ValueError(
            "scaler metadata feature order or semantic hash is invalid"
        )
    required_model_fields = {
        "artifact_key",
        "artifact_path",
        "artifact_sha256",
        "scaler_sha256",
    }
    for method, artifact_key in SCALER_MODEL_ARTIFACT_KEYS.items():
        record = models[method]
        if (
            not isinstance(record, Mapping)
            or set(record) != required_model_fields
            or record.get("artifact_key") != artifact_key
            or record.get("scaler_sha256") != scaler_sha256
        ):
            raise ValueError(
                f"scaler metadata model coverage is invalid for {method}"
            )
    return normalized


def identity_payload(
    *,
    source_checkpoint_sha256: str | None = None,
    source_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a fresh, JSON-serializable copy of the public artifact identity."""

    checkpoint_sha256 = (
        REPEATEDFILM_SOURCE_CHECKPOINT_SHA256
        if source_checkpoint_sha256 is None
        else source_checkpoint_sha256
    )
    snapshot_sha256 = (
        REPEATEDFILM_SOURCE_SNAPSHOT_SHA256
        if source_snapshot_sha256 is None
        else source_snapshot_sha256
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "public_method_namespace_version": PUBLIC_METHOD_NAMESPACE_VERSION,
        "baseline_name": FULL_NMS_BASELINE,
        "protocol_baselines": dict(PROTOCOL_BASELINES),
        "visual_grounding_lineage": VISUAL_GROUNDING_LINEAGE,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_snapshot_sha256": snapshot_sha256,
    }


def validate_public_methods(
    methods: Sequence[str], *, context: str, allow_empty: bool = False
) -> list[str]:
    """Reject internal/bare implementation keys in a public machine artifact."""

    normalized = list(map(str, methods))
    if not normalized and not allow_empty:
        raise ValueError(f"{context} requires at least one public method")
    invalid = sorted(
        {
            method
            for method in normalized
            if (
                not method.startswith(PUBLIC_METHOD_PREFIX)
                or method not in PUBLIC_METHOD_SET
            )
        }
    )
    if invalid:
        raise ValueError(
            f"{context} contains bare/internal method names: {invalid}"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{context} contains duplicate public methods")
    return normalized


def validate_artifact_identity(
    value: Mapping[str, Any], *, context: str
) -> None:
    """Require the exact repeated-FiLM experiment, baseline, and protocols."""

    expected = identity_payload()
    for key in (
        "experiment",
        "public_method_namespace_version",
        "baseline_name",
    ):
        if value.get(key) != expected[key]:
            raise ValueError(
                f"{context} has invalid or missing {key}: "
                f"expected {expected[key]!r}"
            )
    protocol_baselines = value.get("protocol_baselines")
    if not isinstance(protocol_baselines, Mapping) or dict(
        protocol_baselines
    ) != expected["protocol_baselines"]:
        raise ValueError(
            f"{context} protocol_baselines must exactly equal "
            f"{expected['protocol_baselines']}"
        )
    if value.get("visual_grounding_lineage") != VISUAL_GROUNDING_LINEAGE:
        raise ValueError(
            f"{context} must use {VISUAL_GROUNDING_LINEAGE}"
        )
    approved_source = {
        "source_checkpoint_sha256": REPEATEDFILM_SOURCE_CHECKPOINT_SHA256,
        "source_snapshot_sha256": REPEATEDFILM_SOURCE_SNAPSHOT_SHA256,
    }
    for key, approved_digest in approved_source.items():
        digest = str(value.get(key, ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{context} has invalid or missing {key}")
        if digest != approved_digest:
            raise ValueError(
                f"{context} {key} differs from the approved repeated-FiLM source"
            )


def validate_matching_artifact_identity(
    value: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Require two artifacts to carry the same complete source identity."""

    validate_artifact_identity(value, context=context)
    validate_artifact_identity(
        reference, context=f"{context} reference"
    )
    keys = tuple(identity_payload())
    if any(value.get(key) != reference.get(key) for key in keys):
        raise ValueError(
            f"{context} belongs to another repeated-FiLM source"
        )


def validate_vlm_summary_runtime_binding(
    summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    results_path: Path,
    expected_split: str,
    expected_formal_mode: bool,
    expected_sample_count: int,
    expected_eligible_count: int,
    expected_max_output_tokens: int,
    context: str,
) -> Path:
    """Bind a VLM summary and runtime record to one immutable local run."""

    validate_artifact_identity(summary, context=f"{context} summary")
    validate_artifact_identity(runtime, context=f"{context} runtime")
    validate_matching_artifact_identity(
        runtime,
        summary,
        context=f"{context} summary/runtime",
    )
    results = results_path.expanduser().resolve()
    input_path = Path(str(summary.get("input_jsonl", ""))).resolve()
    results_sha256 = sha256_file(results) if results.is_file() else None
    input_sha256 = sha256_file(input_path) if input_path.is_file() else None
    sample_count = int(expected_sample_count)
    eligible_count = int(expected_eligible_count)
    max_output_tokens = int(expected_max_output_tokens)
    if (
        not results.is_file()
        or not input_path.is_file()
        or Path(str(summary.get("results_jsonl", ""))).resolve()
        != results
        or summary.get("results_jsonl_sha256") != results_sha256
        or summary.get("input_jsonl_sha256") != input_sha256
        or Path(str(runtime.get("results_jsonl", ""))).resolve()
        != results
        or runtime.get("results_jsonl_sha256") != results_sha256
        or Path(str(runtime.get("input_jsonl", ""))).resolve()
        != input_path
        or runtime.get("input_jsonl_sha256") != input_sha256
        or summary.get("input_split") != expected_split
        or runtime.get("input_split") != expected_split
        or summary.get("formal_mode") is not expected_formal_mode
        or runtime.get("formal_mode") is not expected_formal_mode
        or not str(summary.get("model_name", ""))
        or not str(summary.get("model_digest", ""))
        or runtime.get("model_name") != summary.get("model_name")
        or runtime.get("model_digest") != summary.get("model_digest")
        or len(
            str(summary.get("stable_session_contract_sha256", ""))
        )
        != 64
        or runtime.get("stable_session_contract_sha256")
        != summary.get("stable_session_contract_sha256")
        or int(summary.get("sample_count", -1)) != sample_count
        or int(runtime.get("sample_count", -1)) != sample_count
        or int(summary.get("eligible_sample_count", -1))
        != eligible_count
        or int(runtime.get("eligible_sample_count", -1))
        != eligible_count
        or int(summary.get("empty_skipped_count", -1))
        != sample_count - eligible_count
        or int(runtime.get("empty_skipped_count", -1))
        != sample_count - eligible_count
        or max_output_tokens <= 0
        or int(summary.get("max_output_tokens", -1))
        != max_output_tokens
        or int(runtime.get("max_output_tokens", -1))
        != max_output_tokens
        or runtime.get("local_only_runtime_passed") is not True
        or not isinstance(
            runtime.get("remote_established_connection_events"), list
        )
        or int(
            runtime.get(
                "remote_established_connection_event_count", -1
            )
        )
        != len(runtime["remote_established_connection_events"])
        or runtime["remote_established_connection_events"]
        or int(runtime.get("fresh_http_call_count", -1))
        + int(runtime.get("cache_hit_count", -1))
        != eligible_count
    ):
        raise ValueError(f"{context} summary/runtime binding is invalid")
    return input_path


def validate_config_identity(value: Mapping[str, Any]) -> None:
    """Validate the preregistered repeated-FiLM-only config.

    Public method names are useful for presentation but cannot establish model
    lineage on their own. The config therefore has to preregister the exact
    namespace, hierarchical repeated-FiLM source, and fail-closed legacy flags.
    The experiment lock performs the cryptographic/checkpoint half.
    """

    if value.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(
            "config must be the modular_reranking_repeatedfilm_v1 experiment"
        )
    if value.get("schema_version") != 2:
        raise ValueError("config schema_version must equal 2")
    if (
        value.get("public_method_namespace_version")
        != PUBLIC_METHOD_NAMESPACE_VERSION
        or value.get("method_namespace") != CONFIG_METHOD_NAMESPACE
    ):
        raise ValueError(
            "config must preregister the repeatedfilm-only public namespace"
        )
    if value.get("baseline_name") != FULL_NMS_BASELINE:
        raise ValueError(
            "config must preregister baseline_name="
            f"{FULL_NMS_BASELINE}"
        )
    lineage = value.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("config must preregister repeated-FiLM lineage")
    expected_lineage_flags = {
        "visual_grounding": VISUAL_GROUNDING_LINEAGE,
        "legacy_singlefilm_allowed": False,
        "legacy_singlefilm_metrics_comparable": False,
        "recover_or_regenerate_singlefilm": False,
        "mix_lineages": False,
    }
    if {
        key: lineage.get(key) for key in expected_lineage_flags
    } != expected_lineage_flags:
        raise ValueError(
            "config lineage must be hierarchical repeated-FiLM only with "
            "single-FiLM recovery/reuse/mixing disabled"
        )
    for key in ("source_run", "source_checkpoint"):
        if not str(lineage.get(key, "")).strip():
            raise ValueError(f"config lineage must provide {key}")
    approved_source = {
        "source_checkpoint_sha256": REPEATEDFILM_SOURCE_CHECKPOINT_SHA256,
        "source_snapshot_sha256": REPEATEDFILM_SOURCE_SNAPSHOT_SHA256,
    }
    for key, approved_digest in approved_source.items():
        digest = str(lineage.get(key, ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"config lineage {key} must be a lowercase SHA-256")
        if digest != approved_digest:
            raise ValueError(
                f"config lineage {key} differs from the approved "
                "repeated-FiLM source"
            )
    protocols = value.get("protocols")
    if not isinstance(protocols, Mapping):
        raise ValueError("config must preregister public protocol baselines")
    observed = {
        protocol: (
            protocols.get(protocol, {}).get("baseline")
            if isinstance(protocols.get(protocol), Mapping)
            else None
        )
        for protocol in PROTOCOL_BASELINES
    }
    if observed != PROTOCOL_BASELINES:
        raise ValueError(
            "config protocol baselines must exactly equal "
            f"{PROTOCOL_BASELINES}"
        )
    if set(map(str, protocols)) != set(PROTOCOL_BASELINES):
        raise ValueError("config must define exactly the registered protocols")
    for protocol, expected_methods in CONFIG_PROTOCOL_METHODS.items():
        item = protocols.get(protocol)
        if not isinstance(item, Mapping):
            raise ValueError(f"config protocol {protocol} must be a mapping")
        methods = validate_public_methods(
            item.get("methods", ()),
            context=f"config protocol {protocol} methods",
        )
        if tuple(methods) != expected_methods:
            raise ValueError(
                f"config protocol {protocol} methods must exactly equal the "
                "registered repeated-FiLM namespace"
            )
        if methods[0] != PROTOCOL_BASELINES[protocol]:
            raise ValueError(
                f"config protocol {protocol} must list its baseline first"
            )

    hifi = value.get("hifi_inference")
    if not isinstance(hifi, Mapping) or {
        "architecture": hifi.get("architecture"),
        "selected_visual_layers": hifi.get("selected_visual_layers"),
        "decoder_order": hifi.get("decoder_order"),
        "independent_film_stages": hifi.get("independent_film_stages"),
    } != {
        "architecture": "models.hifics.HierarchicalCLIPDensePredT",
        "selected_visual_layers": [1, 3, 5, 7, 9],
        "decoder_order": [9, 7, 5, 3, 1],
        "independent_film_stages": 5,
    }:
        raise ValueError(
            "config hifi_inference must use five independent hierarchical "
            "repeated-FiLM stages"
        )
    retained = value.get("retained_test")
    if not isinstance(retained, Mapping) or any(
        retained.get(key) is not False
        for key in (
            "regenerate_hifi",
            "regenerate_dexnet",
            "regenerate_gqcnn",
        )
    ):
        raise ValueError(
            "config retained_test must keep all retained pipeline regeneration "
            "disabled"
        )
    safe_switch = value.get("safe_switch")
    if (
        not isinstance(safe_switch, Mapping)
        or safe_switch.get("fallback") != FULL_NMS_BASELINE
    ):
        raise ValueError("config safe-switch fallback must be the full-NMS baseline")
    local_vlm = value.get("local_vlm")
    if (
        not isinstance(local_vlm, Mapping)
        or local_vlm.get("fallback") != GQCNN_TOP5_BASELINE
        or local_vlm.get("cloud_disabled") is not True
        or tuple(local_vlm.get("variants", ()))
        != ("visual", "visual_metadata")
    ):
        raise ValueError(
            "config local VLM must be local-only and fall back to the "
            "repeated-FiLM GQ-CNN Top-5 baseline"
        )
    local_vlm_preregistration_payload(value)
    formal = value.get("formal_test")
    if (
        not isinstance(formal, Mapping)
        or formal.get("configuration_lock_required") is not True
        or formal.get("invocation_limit") != 1
    ):
        raise ValueError(
            "config formal_test must require one immutable locked invocation"
        )


__all__ = [
    "CONFIG_METHOD_NAMESPACE",
    "CONFIG_PROTOCOL_METHODS",
    "EXPERIMENT_NAME",
    "LOCAL_VLM_GEOMETRY_RISK_COLUMN",
    "LOCAL_VLM_GEOMETRY_RISK_THRESHOLD",
    "LOCAL_VLM_GEOMETRY_SEMANTICS",
    "LOCAL_VLM_MAXIMUM_MEMORY_MIB",
    "LOCAL_VLM_MAXIMUM_P95_SECONDS",
    "LOCAL_VLM_MAX_OUTPUT_TOKENS",
    "PROTOCOL_BASELINES",
    "PUBLIC_METHOD_SET",
    "PUBLIC_METHOD_NAMESPACE_VERSION",
    "SCALER_METADATA_ARTIFACT_KIND",
    "SCALER_METADATA_SCHEMA_VERSION",
    "SCALER_MODEL_ARTIFACT_KEYS",
    "VISUAL_GROUNDING_LINEAGE",
    "build_final_scaler_metadata",
    "identity_payload",
    "local_vlm_preregistration_payload",
    "normalize_scaler_payload",
    "scaler_payload_sha256",
    "validate_artifact_identity",
    "validate_config_identity",
    "validate_public_methods",
    "validate_scaler_metadata_structure",
]
