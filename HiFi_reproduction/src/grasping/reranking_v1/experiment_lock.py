"""Immutable experiment-lock construction and verification.

The lock records every decision and artifact needed before the formal test is
consumed.  It does not run inference.  A separate one-time guard records the
first formal-test start while allowing an interrupted run to resume against the
same lock and output directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from .artifact_contract import (
    SCALER_MODEL_ARTIFACT_KEYS,
    identity_payload,
    local_vlm_preregistration_payload,
    normalize_scaler_payload,
    scaler_payload_sha256,
    validate_artifact_identity,
    validate_config_identity,
    validate_matching_artifact_identity,
    validate_scaler_metadata_structure,
)
from .method_namespace import (
    LOCAL_VLM_SAFE_SWITCH_METHOD,
    expected_formal_method_protocols,
)

from src.grasping.reranking_v1.method_namespace import FULL_NMS_BASELINE
from src.grasping.reranking_v1.local_vlm import (
    stable_session_contract_sha256,
)
from src.grasping.reranking_v1.vlm_visualization import (
    canonical_recipe_sha256,
    validate_vlm_visualization_recipe,
)

LOCK_SCHEMA_VERSION = 2
REQUIRED_SELECTION_KEYS = (
    "primary_method",
    "selected_feature_list",
    "safe_switch_threshold",
    "vlm_backend",
    "vlm_model_digest",
    "prompt_hash",
    "json_schema_hash",
    "seeds",
    "evaluation_definition",
    "expected_test_sample_count",
)
REQUIRED_ARTIFACT_KEYS = (
    "config",
    "repeatedfilm_source_manifest",
    "feature_schema",
    "feature_allowlist",
    "train_manifest",
    "reranker_training_manifest",
    "validation_manifest",
    "validation_evaluation_bundle",
    "test_manifest",
    "split_audit",
    "evaluation_config",
    "inference_bundle",
    "safe_switch_selection",
    "primary_selection",
    "rule_selection",
    "model_regularized_linear_ranker",
    "model_pairwise_ranker",
    "model_multi_positive_listwise_ranker",
    "model_residual_mlp",
    "model_set_aware_residual",
    "model_safe_switch_gate",
    "scaler_metadata",
    "runtime_benchmark",
    "local_ollama_audit",
    "vlm_prompt",
    "vlm_json_schema",
    "vlm_test_input_manifest_visual",
    "vlm_test_input_manifest_visual_metadata",
    "vlm_visual_manifest_visual",
    "vlm_visual_manifest_visual_metadata",
    "vlm_safe_switch_selection",
    "vlm_validation_selection",
    "test_per_candidate",
    "test_sample_universe",
    "strict_test_labels",
)
PUBLIC_IDENTITY_ARTIFACT_KEYS = (
    "reranker_training_manifest",
    "validation_evaluation_bundle",
    "inference_bundle",
    "safe_switch_selection",
    "primary_selection",
    "rule_selection",
    "runtime_benchmark",
    "scaler_metadata",
    "vlm_test_input_manifest_visual",
    "vlm_test_input_manifest_visual_metadata",
    "vlm_visual_manifest_visual",
    "vlm_visual_manifest_visual_metadata",
    "vlm_safe_switch_selection",
    "vlm_validation_selection",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_identity(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    validate_artifact_identity(value, context=context)
    return {key: value[key] for key in identity_payload()}


def _validate_same_locked_identity(
    value: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if _locked_identity(value, context=context) != _locked_identity(
        lock, context="formal experiment lock"
    ):
        raise ValueError(f"{context} belongs to another repeated-FiLM source")


def selected_vlm_contract(lock: Mapping[str, Any]) -> dict[str, str]:
    """Return and revalidate the validation-selected formal VLM contract.

    Formal code must not infer the selected variant from a command-line flag or
    from a test-time result manifest.  The immutable lock is the source of
    truth, and it binds the decision to the exact validation-selection
    artifact that produced it.
    """

    validate_artifact_identity(
        lock, context="selected-VLM experiment lock"
    )
    raw = lock.get("selected_vlm")
    if not isinstance(raw, Mapping):
        raise ValueError("experiment lock omits selected_vlm")
    required = {
        "source_method",
        "source_variant",
        "protocol",
        "model_digest",
        "stable_session_contract_sha256",
        "validation_selection_path",
        "validation_selection_sha256",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"selected_vlm contract missing fields: {missing}")
    source_method = str(raw["source_method"])
    source_variant = str(raw["source_variant"])
    protocol = str(raw["protocol"])
    model_digest = str(raw["model_digest"])
    stable_contract_sha256 = str(
        raw["stable_session_contract_sha256"]
    )
    if (
        not source_method
        or source_variant not in {"visual", "visual_metadata"}
        or not protocol
        or not model_digest
        or len(stable_contract_sha256) != 64
    ):
        raise ValueError("selected_vlm method/variant/protocol/digest is invalid")
    expected_source_method = (
        "repeatedfilm_local_vlm_visual_metadata"
        if source_variant == "visual_metadata"
        else "repeatedfilm_local_vlm_visual"
    )
    if (
        source_method != expected_source_method
        or protocol != "gqcnn_top5"
    ):
        raise ValueError(
            "selected_vlm must use the repeated-FiLM public method namespace"
        )

    artifact = lock.get("artifacts", {}).get("vlm_validation_selection", {})
    selection_path = Path(str(raw["validation_selection_path"])).resolve()
    selection_hash = str(raw["validation_selection_sha256"])
    if (
        Path(str(artifact.get("path", ""))).resolve() != selection_path
        or artifact.get("sha256") != selection_hash
        or not selection_path.is_file()
        or sha256_file(selection_path) != selection_hash
    ):
        raise ValueError(
            "selected_vlm validation selection differs from locked artifact"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        selection, context="selected-VLM validation selection"
    )
    validate_matching_artifact_identity(
        selection,
        lock,
        context="selected-VLM validation selection",
    )
    if (
        selection.get("selection_split") != "validation"
        or selection.get("selected_method") != source_method
        or selection.get("selected_variant") != source_variant
        or selection.get("selected_model_digest") != model_digest
        or selection.get("selected_stable_session_contract_sha256")
        != stable_contract_sha256
        or lock.get("vlm_model_digest") != model_digest
    ):
        raise ValueError(
            "selected_vlm differs from the locked validation selection"
        )
    formal_entries = lock.get("evaluation_definition", {}).get(
        "formal_method_protocols", []
    )
    formal_pairs = [
        (str(item.get("protocol", "")), str(item.get("method", "")))
        for item in formal_entries
        if isinstance(item, Mapping)
    ]
    if [
        pair for pair in formal_pairs if pair[1] == source_method
    ] != [(protocol, source_method)]:
        raise ValueError(
            "selected_vlm does not have exactly one locked formal method entry"
        )
    safe_method = str(
        lock.get("ranking_parameters", {})
        .get("vlm_safe_switch", {})
        .get("method", "")
    )
    if (
        safe_method != LOCAL_VLM_SAFE_SWITCH_METHOD
        or safe_method == source_method
        or [pair for pair in formal_pairs if pair[1] == safe_method]
        != [(protocol, safe_method)]
    ):
        raise ValueError(
            "selected VLM safe-switch formal method binding is invalid"
        )
    validation_methods = {
        str(item.get("method"))
        for item in selection.get("candidates", [])
        if isinstance(item, Mapping) and item.get("method")
    }
    if any(
        method in validation_methods - {source_method}
        for _protocol, method in formal_pairs
    ):
        raise ValueError(
            "formal method set contains an unselected validation VLM variant"
        )
    expected_pairs = expected_formal_method_protocols(source_method)
    if tuple(formal_pairs) != expected_pairs:
        raise ValueError(
            "formal method/protocol set differs from the exact repeated-FiLM "
            "tabular + validation-selected VLM contract"
        )
    return {
        "source_method": source_method,
        "source_variant": source_variant,
        "protocol": protocol,
        "model_digest": model_digest,
        "stable_session_contract_sha256": stable_contract_sha256,
        "validation_selection_path": str(selection_path),
        "validation_selection_sha256": selection_hash,
    }


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_files(paths: Sequence[str], *, repo_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        if path.is_dir():
            resolved.extend(
                item.resolve()
                for item in sorted(path.rglob("*"))
                if item.is_file() and "__pycache__" not in item.parts
            )
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"lock source path missing: {path}")
    unique = sorted(set(resolved), key=str)
    if not unique:
        raise ValueError("lock must include at least one source file")
    return unique


def _path_hashes(
    mapping: Mapping[str, str], *, repo_root: Path
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, raw in sorted(mapping.items()):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"locked artifact missing ({name}): {path}")
        result[str(name)] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object: {path}")
    return value


def _read_deployment_model_scaler(
    path: Path, *, method: str
) -> dict[str, Any]:
    """Safely extract and normalize one scaler from a locked model artifact."""

    if method == "regularized_linear_ranker":
        artifact: Any = _read_json_mapping(
            path, context=f"deployment model {method}"
        )
    else:
        try:
            import torch

            payload = torch.load(
                path, map_location="cpu", weights_only=True
            )
        except Exception as error:
            raise ValueError(
                f"deployment model {method} could not be safely inspected"
            ) from error
        artifact = (
            payload.get("artifact")
            if isinstance(payload, Mapping)
            else None
        )
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("method") != method
        or not isinstance(artifact.get("scaler"), Mapping)
    ):
        raise ValueError(
            f"deployment model {method} omits its exact fitted scaler"
        )
    return normalize_scaler_payload(
        artifact["scaler"], context=f"deployment model {method}"
    )


def _validate_final_scaler_metadata(
    artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Bind canonical scaler metadata to every locked deployment checkpoint."""

    metadata_path = Path(str(artifacts["scaler_metadata"]["path"])).resolve()
    metadata = _read_json_mapping(
        metadata_path, context="scaler metadata"
    )
    common = validate_scaler_metadata_structure(metadata)
    common_sha256 = scaler_payload_sha256(
        common, context="scaler metadata"
    )
    models = metadata["models"]
    for method, artifact_key in SCALER_MODEL_ARTIFACT_KEYS.items():
        locked = artifacts[artifact_key]
        path = Path(str(locked["path"])).resolve()
        record = models[method]
        if (
            Path(str(record["artifact_path"])).resolve() != path
            or record["artifact_sha256"] != locked["sha256"]
            or record["artifact_sha256"] != sha256_file(path)
        ):
            raise ValueError(
                f"scaler metadata model artifact binding is invalid for {method}"
            )
        observed = _read_deployment_model_scaler(path, method=method)
        observed_sha256 = scaler_payload_sha256(
            observed, context=f"deployment model {method}"
        )
        if (
            observed != common
            or observed_sha256 != common_sha256
            or record["scaler_sha256"] != observed_sha256
        ):
            raise ValueError(
                f"deployment model scaler differs from metadata: {method}"
            )

    training_path = Path(
        str(artifacts["reranker_training_manifest"]["path"])
    ).resolve()
    training = _read_json_mapping(
        training_path, context="reranker training manifest"
    )
    if (
        Path(str(training.get("scaler_metadata", ""))).resolve()
        != metadata_path
        or training.get("scaler_metadata_sha256")
        != artifacts["scaler_metadata"]["sha256"]
        or training.get("scaler_sha256") != common_sha256
    ):
        raise ValueError(
            "reranker training manifest does not bind final scaler metadata"
        )

    bundle_path = Path(str(artifacts["inference_bundle"]["path"])).resolve()
    bundle = _read_json_mapping(
        bundle_path, context="inference bundle"
    )
    raw_bundle_scaler = Path(str(bundle.get("scaler_metadata", "")))
    if not raw_bundle_scaler.is_absolute():
        raw_bundle_scaler = bundle_path.parent / raw_bundle_scaler
    if (
        raw_bundle_scaler.resolve() != metadata_path
        or bundle.get("scaler_metadata_sha256")
        != artifacts["scaler_metadata"]["sha256"]
    ):
        raise ValueError(
            "inference bundle does not bind final scaler metadata"
        )
    return metadata


def _validate_repeatedfilm_source_manifest(
    source: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> None:
    """Cryptographically bind the lock to the audited repeated-FiLM source.

    The manifest is evidence, not merely a human-readable lineage label.  This
    validator checks its live checkpoint, repeated FiLM parameter structure,
    frozen split files, source-run protection marker, and agreement with the
    preregistered config.
    """

    lineage = config["lineage"]
    hifi = config["hifi_inference"]
    splits = config["splits"]
    required_exact = {
        "schema_version": 1,
        "audit_status": "COMPLETE_AND_REUSABLE",
        "visual_grounding_variant": "hierarchical_repeated_film",
        "single_film_allowed": False,
        "single_film_artifacts_intentionally_deleted": True,
        "legacy_singlefilm_baseline_reused": False,
        "checkpoint_format": "hifics_hierfilm_trainable_only_v1",
        "checkpoint_strict_load_success": True,
        "checkpoint_missing_keys": [],
        "checkpoint_unexpected_keys": [],
        "architecture_class": "models.hifics.HierarchicalCLIPDensePredT",
        "selected_visual_layers": [1, 3, 5, 7, 9],
        "selected_visual_layers_decoder_order": [9, 7, 5, 3, 1],
        "film_injection_count": 5,
        "projection_count": 5,
        "decoder_count": 5,
        "legacy_singlefilm_attributes_present": False,
        "dataset_protocol": "official OCID-VLG unique train/val/test",
        "split_overlap_audit_passed": True,
    }
    drift = sorted(
        key for key, expected in required_exact.items()
        if source.get(key) != expected
    )
    if drift:
        raise ValueError(
            "repeated-FiLM source manifest failed its audited contract: "
            f"{drift}"
        )
    if not str(source.get("film_parameter_sharing", "")).startswith(
        "independent "
    ):
        raise ValueError(
            "repeated-FiLM source manifest does not prove independent FiLM stages"
        )
    for key in (
        "checkpoint_sha256",
        "checkpoint_trainable_state_sha256",
        "config_run_raw_sha256",
        "config_frozen_raw_sha256",
        "config_canonical_sha256",
        "source_snapshot_sha256",
        "clip_checkpoint_sha256",
    ):
        digest = str(source.get(key, ""))
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"repeated-FiLM source manifest has invalid {key}"
            )

    source_run = Path(str(source.get("source_run_path", ""))).resolve()
    checkpoint = Path(str(source.get("checkpoint_path", ""))).resolve()
    approved_identity = identity_payload()
    protection = Path(
        str(source.get("source_run_protection_marker", ""))
    ).resolve()
    if (
        source_run != Path(str(lineage["source_run"])).resolve()
        or source.get("source_run_id") != source_run.name
        or checkpoint != Path(str(lineage["source_checkpoint"])).resolve()
        or checkpoint.parent.parent != source_run
        or protection != (source_run / ".DO_NOT_PRUNE").resolve()
        or not protection.is_file()
        or not checkpoint.is_file()
        or int(source.get("checkpoint_bytes", -1))
        != checkpoint.stat().st_size
        or source.get("checkpoint_sha256") != sha256_file(checkpoint)
        or source.get("checkpoint_sha256")
        != lineage["source_checkpoint_sha256"]
        or source.get("checkpoint_sha256")
        != approved_identity["source_checkpoint_sha256"]
        or source.get("source_snapshot_sha256")
        != lineage["source_snapshot_sha256"]
        or source.get("source_snapshot_sha256")
        != approved_identity["source_snapshot_sha256"]
    ):
        raise ValueError(
            "repeated-FiLM source run/checkpoint differs from live config evidence"
        )
    if {
        "architecture": hifi.get("architecture"),
        "selected_visual_layers": hifi.get("selected_visual_layers"),
        "decoder_order": hifi.get("decoder_order"),
        "independent_film_stages": hifi.get("independent_film_stages"),
    } != {
        "architecture": source["architecture_class"],
        "selected_visual_layers": source["selected_visual_layers"],
        "decoder_order": source["selected_visual_layers_decoder_order"],
        "independent_film_stages": source["film_injection_count"],
    }:
        raise ValueError(
            "repeated-FiLM source architecture disagrees with registered config"
        )

    run_config_path = source_run / "config.yaml"
    frozen_config_path = (
        source_run
        / "source_snapshot"
        / "hifics_ocidvlg_hierfilm_controlled.yaml"
    )
    config_digest_path = source_run / "config_sha256.txt"
    snapshot_hashes_path = source_run / "source_snapshot_sha256.json"
    snapshot_digest_path = (
        source_run / "source_snapshot_aggregate_sha256.txt"
    )
    if not all(
        path.is_file()
        for path in (
            run_config_path,
            frozen_config_path,
            config_digest_path,
            snapshot_hashes_path,
            snapshot_digest_path,
        )
    ):
        raise ValueError(
            "repeated-FiLM source run omits frozen config/source evidence"
        )
    run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    frozen_config = yaml.safe_load(
        frozen_config_path.read_text(encoding="utf-8")
    )
    canonical_config_sha256 = str(source["config_canonical_sha256"])
    if (
        not isinstance(run_config, Mapping)
        or not isinstance(frozen_config, Mapping)
        or run_config != frozen_config
        or sha256_file(run_config_path) != source["config_run_raw_sha256"]
        or sha256_file(frozen_config_path)
        != source["config_frozen_raw_sha256"]
        or canonical_json_sha256(run_config) != canonical_config_sha256
        or canonical_json_sha256(frozen_config)
        != canonical_config_sha256
        or config_digest_path.read_text(encoding="utf-8").strip()
        != canonical_config_sha256
    ):
        raise ValueError(
            "repeated-FiLM source config differs from its audited evidence"
        )
    snapshot_hashes = _read_json_mapping(
        snapshot_hashes_path,
        context="repeated-FiLM source snapshot hash manifest",
    )
    expected_snapshot_files = {
        "hifics.py",
        "dataloader.py",
        "train_hierfilm.py",
        "verify_hierarchical_film.py",
        "hifics_ocidvlg_hierfilm_controlled.yaml",
    }
    if set(snapshot_hashes) != expected_snapshot_files:
        raise ValueError(
            "repeated-FiLM source snapshot file set is incomplete"
        )
    snapshot_root = source_run / "source_snapshot"
    live_snapshot_entries = {
        path.name
        for path in snapshot_root.iterdir()
        if path.name != "__pycache__"
    }
    if (
        live_snapshot_entries != expected_snapshot_files
        or any(
            (snapshot_root / name).is_symlink()
            for name in expected_snapshot_files
        )
    ):
        raise ValueError(
            "repeated-FiLM source snapshot contains unregistered files"
        )
    for name, expected_hash in snapshot_hashes.items():
        snapshot_file = snapshot_root / name
        if (
            not snapshot_file.is_file()
            or sha256_file(snapshot_file) != expected_hash
        ):
            raise ValueError(
                f"repeated-FiLM source snapshot file changed: {name}"
            )
    source_snapshot_sha256 = str(source["source_snapshot_sha256"])
    if (
        canonical_json_sha256(snapshot_hashes)
        != source_snapshot_sha256
        or snapshot_digest_path.read_text(encoding="utf-8").strip()
        != source_snapshot_sha256
    ):
        raise ValueError(
            "repeated-FiLM source snapshot aggregate is invalid"
        )

    expected_loaded = int(source.get("checkpoint_expected_keys", -1))
    observed_loaded = int(source.get("checkpoint_loaded_keys", -2))
    if expected_loaded <= 0 or observed_loaded != expected_loaded:
        raise ValueError("repeated-FiLM checkpoint strict-load accounting is invalid")
    try:
        import torch

        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError(
            "repeated-FiLM checkpoint could not be safely inspected"
        ) from error
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("repeated-FiLM checkpoint payload must be a mapping")
    trainable_state = checkpoint_payload.get("trainable_state")
    metadata = checkpoint_payload.get("metadata")
    if not isinstance(trainable_state, Mapping) or not isinstance(
        metadata, Mapping
    ):
        raise ValueError(
            "repeated-FiLM checkpoint omits trainable_state or metadata"
        )
    film_keys = {
        str(key)
        for key in trainable_state
        if str(key).startswith("film_stages.")
    }
    expected_film_keys = {
        f"film_stages.{stage}.{branch}.{parameter}"
        for stage in range(5)
        for branch in ("alpha", "beta")
        for parameter in ("weight", "bias")
    }
    block_shapes = {
        "self_attn.in_proj_weight": (192, 64),
        "self_attn.in_proj_bias": (192,),
        "self_attn.out_proj.weight": (64, 64),
        "self_attn.out_proj.bias": (64,),
        "linear1.weight": (2048, 64),
        "linear1.bias": (2048,),
        "linear2.weight": (64, 2048),
        "linear2.bias": (64,),
        "norm1.weight": (64,),
        "norm1.bias": (64,),
        "norm2.weight": (64,),
        "norm2.bias": (64,),
    }
    expected_state_shapes: dict[str, tuple[int, ...]] = {
        "trans_conv.weight": (64, 1, 16, 16),
        "trans_conv.bias": (1,),
        **{
            f"reduces.{stage}.weight": (64, 768)
            for stage in range(5)
        },
        **{
            f"reduces.{stage}.bias": (64,)
            for stage in range(5)
        },
        **{
            f"blocks.{stage}.{suffix}": shape
            for stage in range(5)
            for suffix, shape in block_shapes.items()
        },
        **{
            f"film_stages.{stage}.{branch}.weight": (64, 512)
            for stage in range(5)
            for branch in ("alpha", "beta")
        },
        **{
            f"film_stages.{stage}.{branch}.bias": (64,)
            for stage in range(5)
            for branch in ("alpha", "beta")
        },
    }
    suspicious_legacy_film_keys = {
        str(key)
        for key in trainable_state
        if "film" in str(key).lower() and str(key) not in expected_film_keys
    }
    trainable_digest = hashlib.sha256()
    try:
        for name in sorted(trainable_state):
            tensor = trainable_state[name].detach().cpu().contiguous()
            trainable_digest.update(str(name).encode())
            trainable_digest.update(str(tensor.dtype).encode())
            trainable_digest.update(str(tuple(tensor.shape)).encode())
            trainable_digest.update(tensor.numpy().tobytes())
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "repeated-FiLM checkpoint trainable state contains invalid tensors"
        ) from error
    observed_trainable_state_sha256 = trainable_digest.hexdigest()
    state_schema_valid = all(
        isinstance(trainable_state.get(name), torch.Tensor)
        and trainable_state[name].dtype == torch.float32
        and tuple(trainable_state[name].shape) == expected_shape
        for name, expected_shape in expected_state_shapes.items()
    )
    weight_sources = metadata.get("weight_sources")
    clip_weight_source = (
        weight_sources.get("clip")
        if isinstance(weight_sources, Mapping)
        else None
    )
    decoder_weight_source = (
        weight_sources.get("decoder_film_head")
        if isinstance(weight_sources, Mapping)
        else None
    )
    if (
        checkpoint_payload.get("format") != source["checkpoint_format"]
        or len(trainable_state) != expected_loaded
        or set(map(str, trainable_state)) != set(expected_state_shapes)
        or not state_schema_valid
        or film_keys != expected_film_keys
        or suspicious_legacy_film_keys
        or observed_trainable_state_sha256
        != source.get("checkpoint_trainable_state_sha256")
        or int(metadata.get("global_step", -1))
        != int(source.get("checkpoint_step", -2))
        or int(metadata.get("epoch", -1))
        != int(source.get("checkpoint_epoch", -2))
        or float(metadata.get("best_validation_iou", -1.0))
        != float(source.get("best_validation_mean_iou", -2.0))
        or metadata.get("source_snapshot_sha256")
        != source["source_snapshot_sha256"]
        or metadata.get("config_sha256")
        != source.get("config_canonical_sha256")
        or metadata.get("current_trainable_state_sha256")
        != source.get("checkpoint_trainable_state_sha256")
        or not isinstance(clip_weight_source, Mapping)
        or Path(str(clip_weight_source.get("cache_path", ""))).resolve()
        != Path(str(source.get("clip_checkpoint_path", ""))).resolve()
        or clip_weight_source.get("cache_sha256")
        != source.get("clip_checkpoint_sha256")
        or clip_weight_source.get("frozen") is not True
        or not isinstance(decoder_weight_source, Mapping)
        or decoder_weight_source.get("checkpoint_loaded") is not False
        or not str(decoder_weight_source.get("kind", "")).startswith(
            "fresh "
        )
    ):
        raise ValueError(
            "live checkpoint does not prove the audited five-stage "
            "hierarchical repeated-FiLM source"
        )

    source_splits = source.get("split_manifests")
    metadata_splits = metadata.get("manifest_sha256")
    if not isinstance(source_splits, Mapping) or not isinstance(
        metadata_splits, Mapping
    ):
        raise ValueError("repeated-FiLM source manifest omits frozen split evidence")
    if (
        source.get("manifest_set_sha256") != splits.get("manifest_set_sha256")
        or source.get("dataset_protocol") != splits.get("protocol")
    ):
        raise ValueError("repeated-FiLM source split protocol disagrees with config")
    for split in ("train", "val", "test"):
        source_item = source_splits.get(split)
        config_item = splits.get(split)
        if not isinstance(source_item, Mapping) or not isinstance(
            config_item, Mapping
        ):
            raise ValueError(f"repeated-FiLM source omits {split} split evidence")
        path = Path(str(source_item.get("path", ""))).resolve()
        if (
            path != Path(str(config_item.get("manifest", ""))).resolve()
            or not path.is_file()
            or source_item.get("sha256") != sha256_file(path)
            or source_item.get("sha256") != config_item.get("manifest_sha256")
            or source_item.get("sha256") != metadata_splits.get(split)
            or int(source_item.get("records", -1))
            != int(config_item.get("count", -2))
        ):
            raise ValueError(
                f"repeated-FiLM source {split} manifest disagrees with live config"
            )

    clip_path = Path(str(source.get("clip_checkpoint_path", ""))).resolve()
    if (
        not clip_path.is_file()
        or source.get("clip_checkpoint_sha256") != sha256_file(clip_path)
    ):
        raise ValueError("repeated-FiLM frozen CLIP checkpoint evidence changed")


def _validate_locked_public_artifact_identities(
    lock: Mapping[str, Any],
) -> None:
    """Revalidate nested public identities even when a lock is rehashed."""

    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("experiment lock artifacts must be a mapping")
    missing = sorted(set(REQUIRED_ARTIFACT_KEYS) - set(artifacts))
    if missing:
        raise ValueError(f"experiment lock omits required artifacts: {missing}")
    config_path = Path(str(artifacts["config"]["path"])).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("locked config must be a mapping")
    validate_config_identity(config)
    local_vlm_contract = local_vlm_preregistration_payload(config)

    def require_config_binding(
        owner: Mapping[str, Any], *, context: str
    ) -> None:
        if (
            Path(str(owner.get("config", ""))).resolve() != config_path
            or owner.get("config_sha256")
            != artifacts["config"]["sha256"]
        ):
            raise ValueError(
                f"{context} is not bound to the locked preregistered config"
            )

    source_path = Path(
        str(artifacts["repeatedfilm_source_manifest"]["path"])
    ).resolve()
    source = _read_json_mapping(
        source_path, context="repeated-FiLM source manifest"
    )
    _validate_repeatedfilm_source_manifest(source, config=config)
    expected_source_identity = {
        "visual_grounding_lineage": config["lineage"][
            "visual_grounding"
        ],
        "source_checkpoint_sha256": config["lineage"][
            "source_checkpoint_sha256"
        ],
        "source_snapshot_sha256": config["lineage"][
            "source_snapshot_sha256"
        ],
    }

    for name in PUBLIC_IDENTITY_ARTIFACT_KEYS:
        path = Path(str(artifacts[name]["path"])).resolve()
        value = _read_json_mapping(path, context=name)
        validate_artifact_identity(value, context=name.replace("_", " "))
        if any(
            value.get(key) != expected
            for key, expected in expected_source_identity.items()
        ):
            raise ValueError(
                f"{name.replace('_', ' ')} belongs to another "
                "repeated-FiLM source"
            )

    _validate_final_scaler_metadata(artifacts)

    def bound_path(
        owner: Mapping[str, Any],
        path_key: str,
        *,
        context: str,
        hash_key: str | None = None,
    ) -> Path:
        expected_hash_key = hash_key or f"{path_key}_sha256"
        path = Path(str(owner.get(path_key, ""))).resolve()
        if (
            not path.is_file()
            or owner.get(expected_hash_key) != sha256_file(path)
        ):
            raise ValueError(f"{context} changed or is unbound")
        return path

    def bound_identity(
        owner: Mapping[str, Any],
        path_key: str,
        *,
        context: str,
    ) -> dict[str, Any]:
        path = bound_path(owner, path_key, context=context)
        value = _read_json_mapping(path, context=context)
        validate_artifact_identity(value, context=context)
        if any(
            value.get(key) != expected
            for key, expected in expected_source_identity.items()
        ):
            raise ValueError(
                f"{context} belongs to another repeated-FiLM source"
            )
        return value

    primary = _read_json_mapping(
        Path(str(artifacts["primary_selection"]["path"])).resolve(),
        context="primary selection",
    )
    primary_inputs = primary.get("inputs")
    if not isinstance(primary_inputs, Mapping):
        raise ValueError("primary selection omits bound inputs")
    bound_identity(
        primary_inputs,
        "eligibility_evidence",
        context="tabular eligibility evidence",
    )

    vlm_safe = _read_json_mapping(
        Path(
            str(artifacts["vlm_safe_switch_selection"]["path"])
        ).resolve(),
        context="VLM safe-switch selection",
    )
    vlm_safe_inputs = vlm_safe.get("inputs")
    if not isinstance(vlm_safe_inputs, Mapping):
        raise ValueError("VLM safe-switch selection omits bound inputs")
    require_config_binding(
        vlm_safe, context="VLM safe-switch selection"
    )
    if {
        "geometry_risk_column": vlm_safe.get("geometry_risk_column"),
        "geometry_risk_threshold": vlm_safe.get(
            "geometry_risk_threshold"
        ),
        "geometry_semantics": vlm_safe.get("geometry_semantics"),
    } != {
        key: local_vlm_contract[key]
        for key in (
            "geometry_risk_column",
            "geometry_risk_threshold",
            "geometry_semantics",
        )
    }:
        raise ValueError(
            "VLM safe-switch geometry differs from preregistered config"
        )
    bound_path(
        vlm_safe_inputs,
        "vlm_results",
        context="VLM safe-switch validation results",
    )
    bound_identity(
        vlm_safe_inputs,
        "vlm_summary",
        context="VLM safe-switch validation summary",
    )
    bound_identity(
        vlm_safe_inputs,
        "vlm_runtime_metrics",
        context="VLM safe-switch validation runtime",
    )

    vlm_validation = _read_json_mapping(
        Path(str(artifacts["vlm_validation_selection"]["path"])).resolve(),
        context="VLM validation selection",
    )
    vlm_validation_inputs = vlm_validation.get("inputs")
    if not isinstance(vlm_validation_inputs, Mapping):
        raise ValueError("VLM validation selection omits bound inputs")
    require_config_binding(
        vlm_validation, context="VLM validation selection"
    )
    if {
        "maximum_memory_mib": vlm_validation.get(
            "maximum_memory_mib"
        ),
        "maximum_p95_seconds": vlm_validation.get(
            "maximum_p95_seconds"
        ),
    } != {
        key: local_vlm_contract[key]
        for key in ("maximum_memory_mib", "maximum_p95_seconds")
    }:
        raise ValueError(
            "VLM validation resource limits differ from preregistered config"
        )
    evaluation_bundle = bound_identity(
        vlm_validation_inputs,
        "evaluation_bundle",
        context="VLM validation evaluation bundle",
    )
    if (
        Path(
            str(vlm_validation_inputs["evaluation_bundle"])
        ).resolve()
        != Path(
            str(artifacts["validation_evaluation_bundle"]["path"])
        ).resolve()
        or evaluation_bundle
        != _read_json_mapping(
            Path(
                str(artifacts["validation_evaluation_bundle"]["path"])
            ).resolve(),
            context="locked validation evaluation bundle",
        )
    ):
        raise ValueError(
            "VLM validation evaluation bundle differs from locked artifact"
        )
    model_audit = bound_identity(
        vlm_validation_inputs,
        "model_candidate_audit",
        context="VLM model candidate audit",
    )
    require_config_binding(
        model_audit, context="VLM model candidate audit"
    )
    if (
        model_audit.get("selection_split") != "validation"
        or {
            "maximum_memory_mib": model_audit.get(
                "maximum_memory_mib"
            ),
            "maximum_p95_seconds": model_audit.get(
                "maximum_p95_seconds"
            ),
        }
        != {
            key: local_vlm_contract[key]
            for key in ("maximum_memory_mib", "maximum_p95_seconds")
        }
    ):
        raise ValueError(
            "VLM model candidate audit resource limits differ from "
            "preregistered config"
        )
    if (
        vlm_validation.get("model_candidates")
        != model_audit.get("model_candidates")
    ):
        raise ValueError(
            "VLM validation selection differs from model candidate audit"
        )
    source_artifacts = model_audit.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise ValueError("VLM model candidate audit omits source artifacts")
    audit_role_suffixes = {
        "local_audit",
        "pilot_summary",
        "pilot_runtime",
        "repeat_a",
        "repeat_b",
        "repeat_a_runtime",
        "repeat_b_runtime",
        "repeat_comparison",
    }
    model_candidates = model_audit.get("model_candidates")
    if not isinstance(model_candidates, list) or not model_candidates:
        raise ValueError("VLM model candidate audit omits candidates")
    size_classes = {
        str(candidate.get("size_class", ""))
        for candidate in model_candidates
        if isinstance(candidate, Mapping)
    }
    if not size_classes or "" in size_classes:
        raise ValueError("VLM model candidate audit has invalid size classes")
    allowed_source_roles = {
        f"{size_class}_{suffix}"
        for size_class in size_classes
        for suffix in audit_role_suffixes
    }
    identity_source_suffixes = {
        "pilot_summary",
        "pilot_runtime",
        "repeat_a_runtime",
        "repeat_b_runtime",
        "repeat_comparison",
    }
    observed_source_roles: set[str] = set()
    for item in source_artifacts:
        if not isinstance(item, Mapping):
            raise ValueError(
                "VLM model candidate audit has an invalid source artifact"
            )
        role = str(item.get("role", ""))
        if (
            role not in allowed_source_roles
            or role in observed_source_roles
        ):
            raise ValueError(
                "VLM model candidate audit has unknown or duplicate roles"
            )
        observed_source_roles.add(role)
        path = bound_path(
            item,
            "path",
            context=f"VLM model candidate audit source {role}",
            hash_key="sha256",
        )
        suffix = next(
            suffix
            for suffix in sorted(
                audit_role_suffixes, key=len, reverse=True
            )
            if role.endswith(f"_{suffix}")
        )
        if suffix in identity_source_suffixes:
            value = _read_json_mapping(
                path,
                context=f"VLM model candidate audit source {role}",
            )
            validate_artifact_identity(
                value,
                context=f"VLM model candidate audit source {role}",
            )
            if any(
                value.get(key) != expected
                for key, expected in expected_source_identity.items()
            ):
                raise ValueError(
                    f"VLM model candidate audit source {role} belongs "
                    "to another repeated-FiLM source"
                )
    for candidate in model_candidates:
        if (
            isinstance(candidate, Mapping)
            and candidate.get("eligible") is True
        ):
            required_roles = {
                f"{candidate['size_class']}_{suffix}"
                for suffix in audit_role_suffixes
            }
            if not required_roles <= observed_source_roles:
                raise ValueError(
                    "eligible VLM model candidate omits source evidence"
                )

    candidates = vlm_validation.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("VLM validation selection omits candidate runs")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError("VLM validation candidate must be a mapping")
        context = f"VLM validation candidate {index}"
        bound_path(candidate, "results", context=f"{context} results")
        bound_identity(
            candidate,
            "summary",
            context=f"{context} summary",
        )
        bound_identity(
            candidate,
            "runtime_metrics",
            context=f"{context} runtime",
        )


def _validate_vlm_input_variants(
    artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant, expected_metadata in (
        ("visual", False),
        ("visual_metadata", True),
    ):
        summary_path = Path(
            artifacts[f"vlm_test_input_manifest_{variant}"]["path"]
        )
        aggregate_path = Path(
            artifacts[f"vlm_visual_manifest_{variant}"]["path"]
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        input_path = Path(str(summary.get("output_jsonl", ""))).resolve()
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "COMPLETED"
            or summary.get("gt_fields_included") is not False
            or bool(summary.get("include_metadata")) != expected_metadata
            or not input_path.is_file()
            or summary.get("output_jsonl_sha256") != sha256_file(input_path)
            or Path(str(summary.get("aggregate_visual_manifest", ""))).resolve()
            != aggregate_path.resolve()
            or summary.get("aggregate_visual_manifest_sha256")
            != sha256_file(aggregate_path)
            or aggregate.get("gt_free") is not True
            or int(aggregate.get("sample_count", -1))
            != int(summary.get("samples", -2))
        ):
            raise ValueError(f"locked {variant} VLM input manifest is invalid")
        formal_recipe_mode = summary.get("input_mode") == "formal_test"
        if formal_recipe_mode and (
            summary.get("input_split") != "test"
            or summary.get("visual_storage") != "on_demand_recipe"
            or summary.get("ordinary_pngs_persisted") is not False
            or aggregate.get("schema_version") != 2
            or aggregate.get("manifest_kind")
            != "vlm_visualization_recipes"
            or aggregate.get("storage_mode") != "on_demand_recipe"
            or aggregate.get("input_mode") != "formal_test"
            or aggregate.get("input_split") != "test"
            or int(aggregate.get("nonempty_sample_count", -1))
            + int(aggregate.get("empty_sample_count", -1))
            != int(aggregate.get("sample_count", -2))
        ):
            raise ValueError(
                f"locked {variant} formal VLM inputs are not on-demand recipes"
            )
        aggregate_items = {
            str(item.get("sample_id", "")): item
            for item in aggregate.get("visuals", [])
            if isinstance(item, Mapping)
        }
        if len(aggregate_items) != len(aggregate.get("visuals", [])):
            raise ValueError(
                f"locked {variant} VLM aggregate has duplicate samples"
            )
        empty_ids = set(map(str, aggregate.get("empty_sample_ids", [])))
        rows = _read_jsonl_objects(input_path)
        if len(rows) != int(summary["samples"]):
            raise ValueError(f"locked {variant} VLM input row count changed")
        for row in rows:
            candidate_ids = list(map(str, row.get("candidate_ids", [])))
            metadata = row.get("candidate_metadata")
            if (
                row.get("gt_fields_included") is not False
                or bool(row.get("include_metadata")) != expected_metadata
                or not isinstance(metadata, dict)
                or (
                    expected_metadata
                    and candidate_ids
                    and set(map(str, metadata)) != set(candidate_ids)
                )
                or (not expected_metadata and metadata)
            ):
                raise ValueError(f"locked {variant} VLM row semantics are invalid")
            if formal_recipe_mode and candidate_ids:
                recipe = row.get("visualization_recipe")
                if (
                    row.get("visualization_storage_mode")
                    != "on_demand_recipe"
                    or not isinstance(recipe, Mapping)
                    or row.get("visualization_recipe_sha256")
                    != canonical_recipe_sha256(recipe)
                ):
                    raise ValueError(
                        f"locked {variant} VLM recipe binding is invalid"
                    )
                validated = validate_vlm_visualization_recipe(
                    recipe, verify_sources=True
                )
                item = aggregate_items.get(str(row["sample_id"]))
                if (
                    validated["sample_id"] != str(row["sample_id"])
                    or validated["candidate_ids"] != candidate_ids
                    or item is None
                    or item.get("storage_mode") != "on_demand_recipe"
                    or item.get("candidate_ids") != candidate_ids
                    or item.get("candidate_records_sha256")
                    != validated["candidate_records_sha256"]
                    or item.get("visualization_recipe_sha256")
                    != row["visualization_recipe_sha256"]
                    or any(
                        key in row
                        for key in (
                            "full_scene_overlay_path",
                            "candidate_contact_sheet_path",
                            "visualization_manifest_path",
                        )
                    )
                ):
                    raise ValueError(
                        f"locked {variant} VLM recipe provenance is invalid"
                    )
            elif formal_recipe_mode and (
                str(row["sample_id"]) not in empty_ids
                or str(row["sample_id"]) in aggregate_items
            ):
                raise ValueError(
                    f"locked {variant} valid-empty recipe accounting is invalid"
                )
        if formal_recipe_mode and (
            set(aggregate_items) | empty_ids
            != {str(row["sample_id"]) for row in rows}
            or int(aggregate.get("nonempty_sample_count", -1))
            != len(aggregate_items)
            or int(aggregate.get("empty_sample_count", -1))
            != len(empty_ids)
        ):
            raise ValueError(
                f"locked {variant} recipe sample universe is invalid"
            )
        rows_by_variant[variant] = rows
    ignored = {
        "aggregate_visual_manifest_path",
        "candidate_metadata",
        "include_metadata",
    }
    visual_semantic = [
        {key: value for key, value in row.items() if key not in ignored}
        for row in rows_by_variant["visual"]
    ]
    metadata_semantic = [
        {key: value for key, value in row.items() if key not in ignored}
        for row in rows_by_variant["visual_metadata"]
    ]
    if visual_semantic != metadata_semantic:
        raise ValueError(
            "locked visual and visual-metadata test inputs differ beyond metadata"
        )


def build_lock_manifest(spec: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Build a lock manifest without writing any file."""

    root = repo_root.expanduser().resolve()
    missing = [key for key in REQUIRED_SELECTION_KEYS if key not in spec]
    if missing:
        raise ValueError(f"lock specification missing fields: {missing}")
    if int(spec["expected_test_sample_count"]) <= 0:
        raise ValueError("expected_test_sample_count must be positive")
    threshold = float(spec["safe_switch_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("safe_switch_threshold must be in [0, 1]")
    feature_list = list(map(str, spec["selected_feature_list"]))
    if not feature_list or len(feature_list) != len(set(feature_list)):
        raise ValueError("selected_feature_list must be non-empty and unique")
    source_paths = list(map(str, spec.get("source_paths", ())))
    source_files = _resolve_files(source_paths, repo_root=root)
    required_source_files = {
        path.resolve()
        for relative in (
            "src/grasping/reranking_v1",
            "tools/modular_reranking",
        )
        for path in (root / relative).rglob("*.py")
        if path.is_file()
    }
    missing_source_files = sorted(required_source_files - set(source_files), key=str)
    if missing_source_files:
        raise ValueError(
            "source_paths must cover all reranking implementation files; "
            f"missing={list(map(str, missing_source_files[:5]))}"
        )
    artifact_paths = spec.get("artifact_paths")
    if not isinstance(artifact_paths, Mapping) or not artifact_paths:
        raise ValueError("artifact_paths must be a non-empty mapping")
    missing_artifacts = sorted(set(REQUIRED_ARTIFACT_KEYS) - set(artifact_paths))
    if missing_artifacts:
        raise ValueError(
            f"lock specification missing required artifacts: {missing_artifacts}"
        )
    ranking_parameters = spec.get("ranking_parameters")
    if not isinstance(ranking_parameters, Mapping):
        raise ValueError("ranking_parameters must be a mapping")
    required_ranking = {
        "safe_switch_force_no_switch",
        "tabular_inference_device",
        "vlm_safe_switch",
        "vlm_top_k",
        "vlm_max_output_tokens",
        "vlm_temperature",
        "vlm_stream",
        "vlm_think",
        "mmr_enabled",
        "mmr_lambda",
    }
    if missing_ranking := sorted(required_ranking - set(ranking_parameters)):
        raise ValueError(
            f"ranking_parameters missing locked decisions: {missing_ranking}"
        )
    artifacts = _path_hashes(
        {str(key): str(value) for key, value in artifact_paths.items()},
        repo_root=root,
    )
    _validate_locked_public_artifact_identities({"artifacts": artifacts})
    _validate_vlm_input_variants(artifacts)
    if str(spec["prompt_hash"]) != artifacts["vlm_prompt"]["sha256"]:
        raise ValueError("prompt_hash must equal the locked VLM prompt artifact hash")
    if str(spec["json_schema_hash"]) != artifacts["vlm_json_schema"]["sha256"]:
        raise ValueError(
            "json_schema_hash must equal the locked VLM JSON schema artifact hash"
        )
    local_audit = json.loads(
        Path(artifacts["local_ollama_audit"]["path"]).read_text(encoding="utf-8")
    )
    audit_digest = "sha256:" + str(
        local_audit.get("model", {}).get("manifest_sha256")
    )
    audit_stable_contract_sha256 = stable_session_contract_sha256(
        local_audit
    )
    if str(spec["vlm_model_digest"]) != audit_digest:
        raise ValueError(
            "vlm_model_digest must equal the locked local Ollama audit digest"
        )
    if (
        local_audit.get("stable_session_contract_sha256")
        != audit_stable_contract_sha256
    ):
        raise ValueError("locked local Ollama stable contract is invalid")
    primary_selection = json.loads(
        Path(artifacts["primary_selection"]["path"]).read_text(encoding="utf-8")
    )
    validate_artifact_identity(
        primary_selection, context="primary selection"
    )
    if not str(spec["primary_method"]).startswith("repeatedfilm_"):
        raise ValueError(
            "primary_method must use the repeated-FiLM public namespace"
        )
    eligible_primary = (
        primary_selection.get("selection_split") != "validation"
        or primary_selection.get("primary_method") != str(spec["primary_method"])
    )
    if eligible_primary:
        raise ValueError(
            "primary_method must equal the validation primary selection"
        )
    primary_inputs = primary_selection.get("inputs")
    if not isinstance(primary_inputs, Mapping) or not primary_inputs:
        raise ValueError("primary selection must bind its validation input artifacts")
    for key, expected_hash in primary_inputs.items():
        if not str(key).endswith("_sha256"):
            continue
        path_key = str(key)[: -len("_sha256")]
        raw_path = primary_inputs.get(path_key)
        if raw_path is None or sha256_file(Path(str(raw_path))) != expected_hash:
            raise ValueError(f"primary validation input changed: {path_key}")
    candidate_gate_pass = all(
        primary_selection.get("selected_metrics", {}).get(gate) is True
        for gate in (
            "positive_net_gain",
            "bootstrap_positive_trend",
            "harm_cap_passed",
            "leakage_free",
            "feature_allowlist_passed",
            "candidate_identity_invariant",
            "runtime_acceptable",
            "eligible",
        )
    )
    baseline_fallback = (
        primary_selection.get("primary_method") == FULL_NMS_BASELINE
        and primary_selection.get("baseline_fallback") is True
        and primary_selection.get("selection_reason")
        == (
            "no_candidate_reranker_eligible_"
            "repeatedfilm_gqcnn_q_only_fallback"
        )
    )
    if not candidate_gate_pass and not baseline_fallback:
        raise ValueError("primary selection did not pass every preregistered gate")
    evidence_path = Path(str(primary_inputs["eligibility_evidence"]))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        evidence, context="tabular eligibility evidence"
    )
    role_to_artifact = {
        "split_audit": "split_audit",
        "feature_allowlist": "feature_allowlist",
        "training_manifest": "reranker_training_manifest",
        "runtime_benchmark": "runtime_benchmark",
    }
    for role, artifact_name in role_to_artifact.items():
        source = evidence.get("source_artifacts", {}).get(role, {})
        locked = artifacts[artifact_name]
        if (
            Path(str(source.get("path", ""))).resolve()
            != Path(locked["path"]).resolve()
            or source.get("sha256") != locked["sha256"]
        ):
            raise ValueError(f"primary evidence {role} disagrees with experiment lock")
    reranker_training = json.loads(
        Path(artifacts["reranker_training_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    runtime_benchmark = json.loads(
        Path(artifacts["runtime_benchmark"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    validate_artifact_identity(
        reranker_training, context="reranker training manifest"
    )
    validate_artifact_identity(
        runtime_benchmark, context="runtime benchmark"
    )
    runtime_inputs = runtime_benchmark.get("inputs")
    expected_runtime_inputs = {
        "per_candidate": Path(
            str(reranker_training.get("validation_per_candidate", ""))
        ).resolve(),
        "per_candidate_sha256": str(
            reranker_training.get("validation_sha256", "")
        ),
        "sample_universe": Path(
            str(reranker_training.get("validation_per_sample", ""))
        ).resolve(),
        "sample_universe_sha256": str(
            reranker_training.get("validation_per_sample_sha256", "")
        ),
        "training_manifest": Path(
            artifacts["reranker_training_manifest"]["path"]
        ).resolve(),
        "training_manifest_sha256": artifacts[
            "reranker_training_manifest"
        ]["sha256"],
        "inference_bundle": Path(artifacts["inference_bundle"]["path"]).resolve(),
        "inference_bundle_sha256": artifacts["inference_bundle"]["sha256"],
    }
    if not isinstance(runtime_inputs, Mapping):
        raise ValueError("runtime benchmark omits locked validation inputs")
    for role in (
        "per_candidate",
        "sample_universe",
        "training_manifest",
        "inference_bundle",
    ):
        actual_path = Path(str(runtime_inputs.get(role, ""))).resolve()
        actual_hash = runtime_inputs.get(f"{role}_sha256")
        if (
            actual_path != expected_runtime_inputs[role]
            or actual_hash != expected_runtime_inputs[f"{role}_sha256"]
            or not actual_path.is_file()
            or sha256_file(actual_path) != actual_hash
        ):
            raise ValueError(f"runtime benchmark input disagrees with lock: {role}")
    if (
        Path(str(primary_inputs.get("evaluation_bundle", ""))).resolve()
        != Path(artifacts["validation_evaluation_bundle"]["path"]).resolve()
        or primary_inputs.get("evaluation_bundle_sha256")
        != artifacts["validation_evaluation_bundle"]["sha256"]
    ):
        raise ValueError("primary evaluation bundle disagrees with experiment lock")
    validation_evaluation_bundle = json.loads(
        Path(artifacts["validation_evaluation_bundle"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    validate_artifact_identity(
        validation_evaluation_bundle,
        context="validation evaluation bundle",
    )
    inference_bundle = json.loads(
        Path(artifacts["inference_bundle"]["path"]).read_text(encoding="utf-8")
    )
    validate_artifact_identity(
        inference_bundle, context="formal inference bundle"
    )
    training_primary_methods = list(
        map(
            str,
            reranker_training.get("primary_candidate_methods", []),
        )
    )
    if (
        list(map(str, inference_bundle.get("feature_columns", []))) != feature_list
        or inference_bundle.get("candidate_pool_modified") is not False
        or inference_bundle.get("reload_parity_verified") is not True
        or not training_primary_methods
        or any(
            not method.startswith("repeatedfilm_")
            for method in training_primary_methods
        )
        or list(
            map(
                str,
                inference_bundle.get("primary_candidate_methods", []),
            )
        )
        != training_primary_methods
        or set(map(str, runtime_benchmark.get("methods", {})))
        != set(training_primary_methods)
    ):
        raise ValueError("selected feature list disagrees with inference bundle")
    allowlist = json.loads(
        Path(artifacts["feature_allowlist"]["path"]).read_text(encoding="utf-8")
    )
    if (
        allowlist.get("ground_truth_allowed") is not False
        or not set(feature_list) <= set(map(str, allowlist.get("features", [])))
    ):
        raise ValueError("selected feature list is absent from the GT-free allowlist")
    scaler_metadata = _read_json_mapping(
        Path(artifacts["scaler_metadata"]["path"]),
        context="scaler metadata",
    )
    if list(map(str, scaler_metadata.get("feature_columns", []))) != feature_list:
        raise ValueError(
            "selected feature list disagrees with final deployment scaler"
        )
    safe_selection = json.loads(
        Path(artifacts["safe_switch_selection"]["path"]).read_text(encoding="utf-8")
    )
    if (
        safe_selection.get("selection_split") != "validation"
        or float(safe_selection.get("threshold", -1.0)) != threshold
        or bool(safe_selection.get("force_no_switch"))
        != bool(ranking_parameters["safe_switch_force_no_switch"])
    ):
        raise ValueError("safe-switch decision disagrees with validation selection")
    if (
        Path(
            str(reranker_training.get("safe_switch_selection_path", ""))
        ).resolve()
        != Path(artifacts["safe_switch_selection"]["path"]).resolve()
        or reranker_training.get("safe_switch_selection_sha256")
        != artifacts["safe_switch_selection"]["sha256"]
        or reranker_training.get("safe_switch_selection") != safe_selection
        or Path(
            str(safe_selection.get("validation_per_candidate", ""))
        ).resolve()
        != Path(
            str(reranker_training.get("validation_per_candidate", ""))
        ).resolve()
        or safe_selection.get("validation_per_candidate_sha256")
        != reranker_training.get("validation_sha256")
    ):
        raise ValueError(
            "safe-switch selection disagrees with locked training manifest"
        )
    config = yaml.safe_load(
        Path(artifacts["config"]["path"]).read_text(encoding="utf-8")
    )
    validate_config_identity(config)
    registered_vlm_contract = local_vlm_preregistration_payload(config)
    registered_harm_limit = float(
        config["safe_switch"]["harmful_rate_limit_all_samples"]
    )
    registered_runtime_limit = float(
        config["primary_selection"][
            "maximum_inference_seconds_per_sample"
        ]
    )
    registered_inference_device = str(
        config["primary_selection"]["inference_device"]
    )
    if (
        int(config["learned_rankers"]["seed"]) != int(spec["seeds"].get("model", -1))
        or int(config["evaluation"]["bootstrap_seed"])
        != int(spec["seeds"].get("bootstrap", -1))
        or int(config["evaluation"]["bootstrap_draws"])
        != int(spec["evaluation_definition"].get("bootstrap_replicates", -1))
        or int(config["local_vlm"]["seed"])
        != int(spec["seeds"].get("vlm", -1))
        or float(config["local_vlm"]["temperature"])
        != float(ranking_parameters["vlm_temperature"])
        or bool(config["local_vlm"]["stream"])
        != bool(ranking_parameters["vlm_stream"])
        or bool(config["local_vlm"]["thinking"])
        != bool(ranking_parameters["vlm_think"])
        or int(config["local_vlm"]["top_k_pool"])
        != int(ranking_parameters["vlm_top_k"])
        or int(ranking_parameters["vlm_max_output_tokens"])
        != int(registered_vlm_contract["max_output_tokens"])
    ):
        raise ValueError("registered config seeds/runtime settings disagree with lock")
    if (
        float(safe_selection.get("harmful_rate_limit", -1.0))
        != registered_harm_limit
        or safe_selection.get("harmful_rate_denominator")
        != "all_validation_samples"
        or float(primary_selection.get("harmful_rate_limit_all_samples", -1.0))
        != registered_harm_limit
        or registered_runtime_limit <= 0.0
        or float(
            primary_selection.get(
                "maximum_inference_seconds_per_sample", -1.0
            )
        )
        != registered_runtime_limit
        or float(
            evidence.get("maximum_inference_seconds_per_sample", -1.0)
        )
        != registered_runtime_limit
        or str(evidence.get("formal_inference_device", ""))
        != registered_inference_device
        or str(primary_selection.get("formal_inference_device", ""))
        != registered_inference_device
        or str(ranking_parameters["tabular_inference_device"])
        != registered_inference_device
        or runtime_benchmark.get("device_requested")
        != registered_inference_device
    ):
        raise ValueError(
            "validation selections disagree with registered harm/runtime gates"
        )
    rule_selection = json.loads(
        Path(artifacts["rule_selection"]["path"]).read_text(encoding="utf-8")
    )
    if (
        Path(str(reranker_training.get("rule_selection", ""))).resolve()
        != Path(artifacts["rule_selection"]["path"]).resolve()
        or reranker_training.get("rule_selection_sha256")
        != artifacts["rule_selection"]["sha256"]
        or Path(
            str(rule_selection.get("validation_per_candidate", ""))
        ).resolve()
        != Path(
            str(reranker_training.get("validation_per_candidate", ""))
        ).resolve()
        or rule_selection.get("validation_per_candidate_sha256")
        != reranker_training.get("validation_sha256")
    ):
        raise ValueError("rule selection disagrees with locked training manifest")
    rule_alpha = float(rule_selection.get("selected", {}).get("alpha", -1.0))
    bundle_rule = inference_bundle.get("rule_methods", {}).get(
        "q_softmask_rule", {}
    )
    if (
        rule_selection.get("selection_split") != "validation"
        or rule_selection.get("method") != "q_softmask_rule"
        or float(bundle_rule.get("alpha", -2.0)) != rule_alpha
        or abs(float(bundle_rule.get("beta", -2.0)) - (1.0 - rule_alpha))
        > 1e-12
    ):
        raise ValueError("rule selection disagrees with trained inference bundle")
    for path_key, hash_key in (
        ("validation_per_candidate", "validation_per_candidate_sha256"),
        ("sweep", "sweep_sha256"),
    ):
        path = Path(str(rule_selection.get(path_key, "")))
        if (
            not path.is_file()
            or sha256_file(path) != rule_selection.get(hash_key)
        ):
            raise ValueError(f"rule selection input changed: {path_key}")
    vlm_safe_selection = json.loads(
        Path(artifacts["vlm_safe_switch_selection"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    expected_vlm_switch = {
        "method": vlm_safe_selection["method"],
        "threshold_kind": vlm_safe_selection["threshold_kind"],
        "threshold": vlm_safe_selection.get("threshold"),
        "geometry_risk_column": vlm_safe_selection["geometry_risk_column"],
        "geometry_risk_threshold": vlm_safe_selection[
            "geometry_risk_threshold"
        ],
        "geometry_semantics": vlm_safe_selection[
            "geometry_semantics"
        ],
    }
    if (
        vlm_safe_selection.get("selection_split") != "validation"
        or vlm_safe_selection.get("harmful_rate_denominator")
        != "all_validation_samples"
        or float(vlm_safe_selection.get("harmful_rate_limit", -1.0))
        != registered_harm_limit
        or vlm_safe_selection.get("vlm_runtime_local_only_passed") is not True
        or vlm_safe_selection.get("vlm_model_digest")
        != str(spec["vlm_model_digest"])
        or vlm_safe_selection.get(
            "stable_session_contract_sha256"
        )
        != audit_stable_contract_sha256
        or dict(ranking_parameters["vlm_safe_switch"]) != expected_vlm_switch
        or {
            key: expected_vlm_switch[key]
            for key in (
                "geometry_risk_column",
                "geometry_risk_threshold",
                "geometry_semantics",
            )
        }
        != {
            key: registered_vlm_contract[key]
            for key in (
                "geometry_risk_column",
                "geometry_risk_threshold",
                "geometry_semantics",
            )
        }
    ):
        raise ValueError("VLM safe-switch decision disagrees with validation selection")
    for path_key, expected_hash in vlm_safe_selection.get("inputs", {}).items():
        if not str(path_key).endswith("_sha256"):
            continue
        raw_path = vlm_safe_selection["inputs"].get(
            str(path_key)[: -len("_sha256")]
        )
        if raw_path is None or sha256_file(Path(str(raw_path))) != expected_hash:
            raise ValueError("VLM safe-switch validation input changed")
    vlm_validation = json.loads(
        Path(artifacts["vlm_validation_selection"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        vlm_validation.get("selection_split") != "validation"
        or vlm_validation.get("selected_variant")
        not in {"visual", "visual_metadata"}
        or not str(vlm_validation.get("selected_method", ""))
        or vlm_validation.get("selected_model_digest")
        != str(spec["vlm_model_digest"])
        or vlm_validation.get(
            "selected_stable_session_contract_sha256"
        )
        != audit_stable_contract_sha256
        or vlm_validation.get("selected_method")
        != vlm_safe_selection.get("source_method")
        or vlm_validation.get("selected_variant")
        != vlm_safe_selection.get("source_variant")
    ):
        raise ValueError("VLM model/variant differs from validation selection")
    selected_candidates = [
        item
        for item in vlm_validation.get("candidates", [])
        if item.get("method") == vlm_validation.get("selected_method")
        and item.get("variant") == vlm_validation.get("selected_variant")
        and item.get("model_digest")
        == vlm_validation.get("selected_model_digest")
    ]
    if (
        len(selected_candidates) != 1
        or selected_candidates[0].get("eligible") is not True
        or selected_candidates[0].get("resource_pass") is not True
    ):
        raise ValueError("selected VLM run is absent from eligible validation candidates")
    selected_candidate = selected_candidates[0]
    safe_to_selected_fields = {
        "vlm_results": "results",
        "vlm_summary": "summary",
        "vlm_runtime_metrics": "runtime_metrics",
    }
    safe_inputs = vlm_safe_selection.get("inputs", {})
    for safe_name, selected_name in safe_to_selected_fields.items():
        if (
            Path(str(safe_inputs.get(safe_name, ""))).resolve()
            != Path(str(selected_candidate.get(selected_name, ""))).resolve()
            or safe_inputs.get(f"{safe_name}_sha256")
            != selected_candidate.get(f"{selected_name}_sha256")
        ):
            raise ValueError(
                "VLM safe-switch evidence differs from selected validation run"
            )
    for path_key, expected_hash in vlm_validation.get("inputs", {}).items():
        if not str(path_key).endswith("_sha256"):
            continue
        raw_path = vlm_validation["inputs"].get(
            str(path_key)[: -len("_sha256")]
        )
        if raw_path is None or sha256_file(Path(str(raw_path))) != expected_hash:
            raise ValueError("VLM model/variant validation input changed")
    if (
        Path(str(vlm_validation["inputs"]["evaluation_bundle"])).resolve()
        != Path(artifacts["validation_evaluation_bundle"]["path"]).resolve()
    ):
        raise ValueError("VLM selection evaluation bundle disagrees with lock")
    evaluation_config = yaml.safe_load(
        Path(artifacts["evaluation_config"]["path"]).read_text(encoding="utf-8")
    )
    evaluation_definition = spec["evaluation_definition"]
    expected_evaluation = {
        "iou_operator": ">",
        "iou_threshold": float(evaluation_config["iou_threshold"]),
        "angle_operator": "<=",
        "angle_threshold_deg": float(evaluation_config["angle_threshold_deg"]),
        "top_k": int(evaluation_config["top_k"]),
    }
    if any(evaluation_definition.get(key) != value for key, value in expected_evaluation.items()):
        raise ValueError("evaluation definition disagrees with locked evaluation config")
    formal_methods = evaluation_definition.get("formal_method_protocols")
    if not isinstance(formal_methods, list) or not formal_methods:
        raise ValueError(
            "evaluation_definition must lock formal_method_protocols"
        )
    formal_pairs: list[tuple[str, str]] = []
    for item in formal_methods:
        if (
            not isinstance(item, Mapping)
            or not str(item.get("protocol", ""))
            or not str(item.get("method", ""))
        ):
            raise ValueError("formal_method_protocols contains an invalid entry")
        formal_pairs.append((str(item["protocol"]), str(item["method"])))
    if any(
        not method.startswith("repeatedfilm_")
        for _protocol, method in formal_pairs
    ):
        raise ValueError(
            "formal methods must use the repeated-FiLM public namespace"
        )
    if len(formal_pairs) != len(set(formal_pairs)):
        raise ValueError("formal_method_protocols contains duplicate entries")
    selected_method = str(vlm_validation["selected_method"])
    selected_variant = str(vlm_validation["selected_variant"])
    selected_pairs = [
        pair for pair in formal_pairs if pair[1] == selected_method
    ]
    if len(selected_pairs) != 1:
        raise ValueError(
            "formal method set must contain the selected raw VLM exactly once"
        )
    selected_protocol = selected_pairs[0][0]
    safe_method = str(vlm_safe_selection["method"])
    if not safe_method or safe_method == selected_method:
        raise ValueError(
            "selected raw VLM and VLM safe-switch methods must be distinct"
        )
    safe_pairs = [pair for pair in formal_pairs if pair[1] == safe_method]
    if safe_pairs != [(selected_protocol, safe_method)]:
        raise ValueError(
            "formal method set must contain the selected VLM safe-switch output "
            "exactly once under the selected protocol"
        )
    validation_vlm_methods = {
        str(item.get("method"))
        for item in vlm_validation.get("candidates", [])
        if item.get("method")
    }
    unselected_vlm_methods = validation_vlm_methods - {selected_method}
    leaked_unselected = sorted(
        {
            method
            for _protocol, method in formal_pairs
            if method in unselected_vlm_methods
        }
    )
    if leaked_unselected:
        raise ValueError(
            "formal method set contains unselected validation VLM variants: "
            f"{leaked_unselected}"
        )
    expected_formal_pairs = expected_formal_method_protocols(selected_method)
    if tuple(formal_pairs) != expected_formal_pairs:
        raise ValueError(
            "formal method/protocol set differs from the exact repeated-FiLM "
            "tabular + validation-selected VLM contract"
        )
    selected_vlm = {
        "source_method": selected_method,
        "source_variant": selected_variant,
        "protocol": selected_protocol,
        "model_digest": str(vlm_validation["selected_model_digest"]),
        "stable_session_contract_sha256": (
            audit_stable_contract_sha256
        ),
        "validation_selection_path": artifacts[
            "vlm_validation_selection"
        ]["path"],
        "validation_selection_sha256": artifacts[
            "vlm_validation_selection"
        ]["sha256"],
    }
    universe_path = Path(artifacts["test_sample_universe"]["path"])
    if universe_path.suffix.lower() in {".parquet", ".pq"}:
        universe = pd.read_parquet(universe_path)
    else:
        universe = pd.read_csv(universe_path)
    if (
        "sample_id" not in universe
        or universe["sample_id"].astype(str).duplicated().any()
        or len(universe) != int(spec["expected_test_sample_count"])
    ):
        raise ValueError("expected test count disagrees with locked sample universe")
    if int(ranking_parameters["vlm_top_k"]) != 5:
        raise ValueError("formal VLM candidate pool must remain the frozen Top-5")
    if (
        int(ranking_parameters["vlm_max_output_tokens"])
        != int(registered_vlm_contract["max_output_tokens"])
        or float(ranking_parameters["vlm_temperature"]) != 0.0
        or bool(ranking_parameters["vlm_stream"])
        or bool(ranking_parameters["vlm_think"])
    ):
        raise ValueError("formal VLM generation settings violate deterministic mode")
    if "vlm" not in spec["seeds"]:
        raise ValueError("seeds must lock the VLM seed")
    if bool(ranking_parameters["mmr_enabled"]):
        mmr_lambda = ranking_parameters["mmr_lambda"]
        if mmr_lambda is None or not 0.0 <= float(mmr_lambda) <= 1.0:
            raise ValueError("enabled MMR requires mmr_lambda in [0,1]")
    elif ranking_parameters["mmr_lambda"] is not None:
        raise ValueError("disabled MMR must lock mmr_lambda=null")

    source_hashes = {
        str(path): sha256_file(path)
        for path in source_files
    }
    diff = _git(root, "diff", "--binary", "--", ".")
    staged_diff = _git(root, "diff", "--binary", "--cached", "--", ".")
    manifest = {
        **identity_payload(
            source_checkpoint_sha256=config["lineage"][
                "source_checkpoint_sha256"
            ],
            source_snapshot_sha256=config["lineage"][
                "source_snapshot_sha256"
            ],
        ),
        "schema_version": LOCK_SCHEMA_VERSION,
        "lock_kind": "modular_reranking_v1_pre_formal_test",
        "repo_root": str(root),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "git_staged_diff_sha256": hashlib.sha256(
            staged_diff.encode("utf-8")
        ).hexdigest(),
        "source_code_hashes": source_hashes,
        "artifacts": artifacts,
        "config_hash": artifacts.get("config", {}).get("sha256"),
        "feature_schema_hash": artifacts.get("feature_schema", {}).get("sha256"),
        "train_manifest_hash": artifacts.get("train_manifest", {}).get("sha256"),
        "validation_manifest_hash": artifacts.get(
            "validation_manifest", {}
        ).get("sha256"),
        "test_manifest_hash": artifacts.get("test_manifest", {}).get("sha256"),
        "model_checkpoint_hashes": {
            name: value["sha256"]
            for name, value in artifacts.items()
            if name.startswith("model_") or name.startswith("scaler_")
        },
        "selected_feature_list": feature_list,
        "primary_method": str(spec["primary_method"]),
        "vlm_backend": str(spec["vlm_backend"]),
        "vlm_model_digest": str(spec["vlm_model_digest"]),
        "selected_vlm": selected_vlm,
        "prompt_hash": str(spec["prompt_hash"]),
        "json_schema_hash": str(spec["json_schema_hash"]),
        "safe_switch_threshold": threshold,
        "ranking_parameters": dict(ranking_parameters),
        "seeds": dict(spec["seeds"]),
        "evaluation_definition": dict(spec["evaluation_definition"]),
        "expected_test_sample_count": int(spec["expected_test_sample_count"]),
        "validation_selection": dict(spec.get("validation_selection", {})),
        "formal_test_consumed": False,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def write_lock_exclusive(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a valid lock once; overwrite is intentionally impossible."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def verify_lock(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    expected = payload.pop("manifest_content_sha256", None)
    actual = canonical_json_sha256(payload)
    if expected != actual:
        raise ValueError("experiment lock content hash mismatch")
    payload["manifest_content_sha256"] = expected
    if payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported experiment lock schema: {payload.get('schema_version')}"
        )
    validate_artifact_identity(payload, context="verified experiment lock")
    if payload.get("formal_test_consumed") is not False:
        raise ValueError("pre-test lock must record formal_test_consumed=false")
    for item in payload["artifacts"].values():
        artifact = Path(item["path"])
        if sha256_file(artifact) != item["sha256"]:
            raise ValueError(f"locked artifact changed: {artifact}")
    for raw, expected_hash in payload["source_code_hashes"].items():
        source = Path(raw)
        if sha256_file(source) != expected_hash:
            raise ValueError(f"locked source changed: {source}")
    _validate_locked_public_artifact_identities(payload)
    _validate_vlm_input_variants(payload["artifacts"])
    selected_vlm_contract(payload)
    return payload


def verify_completed_formal_stage(
    lock_path: Path,
    *,
    stage: str,
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify one formal manifest against its guarded start/completion ledger."""

    normalized_stage = str(stage).upper()
    if normalized_stage != "TEST" and not re.fullmatch(
        r"[A-Z][A-Z0-9_]{1,63}", normalized_stage
    ):
        raise ValueError("formal stage must be TEST or a short identifier")
    lock = verify_lock(lock_path)
    resolved_lock = lock_path.expanduser().resolve()
    start_name = (
        f"{resolved_lock.name}.FORMAL_TEST_STARTED.json"
        if normalized_stage == "TEST"
        else f"{resolved_lock.name}.FORMAL_{normalized_stage}_STARTED.json"
    )
    completion_name = (
        f"{resolved_lock.name}.FORMAL_TEST_COMPLETED.json"
        if normalized_stage == "TEST"
        else f"{resolved_lock.name}.FORMAL_{normalized_stage}_COMPLETED.json"
    )
    start_path = resolved_lock.with_name(start_name)
    completion_path = resolved_lock.with_name(completion_name)
    if not start_path.is_file():
        raise FileNotFoundError(
            f"formal {normalized_stage} start ledger is missing"
        )
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"formal {normalized_stage} completion ledger is missing"
        )
    start = _read_json_mapping(
        start_path, context=f"formal {normalized_stage} start ledger"
    )
    completion = _read_json_mapping(
        completion_path,
        context=f"formal {normalized_stage} completion ledger",
    )
    _validate_same_locked_identity(
        start,
        lock,
        context=f"formal {normalized_stage} start ledger",
    )
    _validate_same_locked_identity(
        completion,
        lock,
        context=f"formal {normalized_stage} completion ledger",
    )
    manifest = manifest_path.expanduser().resolve()
    guarded_root = Path(str(start.get("output_root", ""))).resolve()
    expected_manifest = (
        guarded_root / "inference_manifest.json"
        if normalized_stage == "TEST"
        else manifest
    )
    if (
        start.get("lock_path") != str(resolved_lock)
        or start.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
        or int(start.get("expected_test_sample_count", -1))
        != int(lock["expected_test_sample_count"])
        or manifest.parent != guarded_root
        or manifest != expected_manifest
        or not manifest.is_file()
    ):
        raise ValueError(
            f"formal {normalized_stage} manifest is outside its guarded output"
        )
    completed_manifest = _read_json_mapping(
        manifest,
        context=f"formal {normalized_stage} completed manifest",
    )
    _validate_same_locked_identity(
        completed_manifest,
        lock,
        context=f"formal {normalized_stage} completed manifest",
    )
    if (
        completed_manifest.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
        or completion.get("lock_path") != str(resolved_lock)
        or completion.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
        or completion.get("stage") != normalized_stage
        or Path(str(completion.get("start_ledger", ""))).resolve()
        != start_path
        or completion.get("start_ledger_sha256") != sha256_file(start_path)
        or Path(str(completion.get("manifest_path", ""))).resolve()
        != manifest
        or completion.get("manifest_sha256") != sha256_file(manifest)
    ):
        raise ValueError(
            f"formal {normalized_stage} completion ledger is invalid"
        )
    return {
        "lock": lock,
        "start": start,
        "completion": completion,
        "manifest": completed_manifest,
    }


def consume_formal_test_once(
    lock_path: Path,
    *,
    output_root: Path,
    invocation: Sequence[str],
) -> dict[str, Any]:
    """Atomically mark the only formal-test decision run.

    Reusing the exact same lock and output root returns the existing marker so
    an interrupted run can resume.  A different invocation is rejected.
    """

    lock = verify_lock(lock_path)
    resolved_lock = lock_path.expanduser().resolve()
    root = output_root.expanduser().resolve()
    # The ledger is global to the immutable lock, not scoped to an arbitrary
    # output directory. Otherwise changing --output-root could consume the
    # same formal test repeatedly.
    marker = resolved_lock.with_name(
        f"{resolved_lock.name}.FORMAL_TEST_STARTED.json"
    )
    payload = {
        **_locked_identity(lock, context="formal experiment lock"),
        "lock_path": str(resolved_lock),
        "lock_content_sha256": lock["manifest_content_sha256"],
        "expected_test_sample_count": lock["expected_test_sample_count"],
        "output_root": str(root),
        "invocation": list(map(str, invocation)),
    }
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                "formal test was already started with a different locked invocation"
            )
        root.mkdir(parents=True, exist_ok=True)
        return existing
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "a new formal test requires an empty dedicated output directory"
        )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    root.mkdir(parents=True, exist_ok=True)
    return payload


def consume_formal_stage_once(
    lock_path: Path,
    *,
    stage: str,
    output_root: Path,
    invocation: Sequence[str],
    allowed_preexisting_files: Iterable[Path] = (),
) -> dict[str, Any]:
    """Atomically bind one post-reranker formal stage to one locked invocation.

    A caller may allow a small, explicitly enumerated set of immutable
    preflight files below ``output_root``.  This is used for a fresh local
    runtime audit that must be written before the guarded VLM process starts.
    Directories are never implicitly allowed, and every existing file must be
    named exactly.
    """

    normalized_stage = str(stage).upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", normalized_stage):
        raise ValueError("formal stage must be a short alphanumeric identifier")
    lock = verify_lock(lock_path)
    resolved_lock = lock_path.expanduser().resolve()
    prerequisite = resolved_lock.with_name(
        f"{resolved_lock.name}.FORMAL_TEST_STARTED.json"
    )
    if not prerequisite.is_file():
        raise FileNotFoundError(
            "formal reranker stage must start before downstream formal stages"
        )
    prerequisite_payload = json.loads(prerequisite.read_text(encoding="utf-8"))
    _validate_same_locked_identity(
        prerequisite_payload,
        lock,
        context="formal reranker start ledger",
    )
    if (
        prerequisite_payload.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
    ):
        raise ValueError("formal reranker ledger belongs to another lock")
    prerequisite_completion = resolved_lock.with_name(
        f"{resolved_lock.name}.FORMAL_TEST_COMPLETED.json"
    )
    if not prerequisite_completion.is_file():
        raise FileNotFoundError(
            "formal reranker inference must complete before downstream stages"
        )
    completed_test = json.loads(
        prerequisite_completion.read_text(encoding="utf-8")
    )
    _validate_same_locked_identity(
        completed_test,
        lock,
        context="formal reranker completion ledger",
    )
    if (
        completed_test.get("lock_content_sha256")
        != lock["manifest_content_sha256"]
        or completed_test.get("start_ledger_sha256")
        != sha256_file(prerequisite)
        or sha256_file(Path(str(completed_test.get("manifest_path", ""))))
        != completed_test.get("manifest_sha256")
    ):
        raise ValueError("formal reranker completion ledger is invalid")
    root = output_root.expanduser().resolve()
    marker = resolved_lock.with_name(
        f"{resolved_lock.name}.FORMAL_{normalized_stage}_STARTED.json"
    )
    payload = {
        **_locked_identity(lock, context="formal experiment lock"),
        "lock_path": str(resolved_lock),
        "lock_content_sha256": lock["manifest_content_sha256"],
        "expected_test_sample_count": lock["expected_test_sample_count"],
        "formal_test_ledger": str(prerequisite),
        "output_root": str(root),
        "stage": normalized_stage,
        "invocation": list(map(str, invocation)),
    }
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"formal stage {normalized_stage} was already started "
                "with a different locked invocation"
            )
        root.mkdir(parents=True, exist_ok=True)
        return existing
    allowed = {
        path.expanduser().resolve() for path in allowed_preexisting_files
    }
    if any(path == root or root not in path.parents for path in allowed):
        raise ValueError(
            "allowed formal-stage preflight files must be below output_root"
        )
    existing_files: set[Path] = set()
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise FileExistsError(
                    "a new formal stage rejects pre-existing symlinks"
                )
            if path.is_file():
                existing_files.add(path.resolve())
            elif not path.is_dir():
                raise FileExistsError(
                    "a new formal stage rejects special pre-existing paths"
                )
    if existing_files != allowed or any(not path.is_file() for path in allowed):
        raise FileExistsError(
            f"a new formal stage {normalized_stage} requires an empty "
            "dedicated output directory apart from explicitly bound "
            "preflight files"
        )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    root.mkdir(parents=True, exist_ok=True)
    return payload


def _complete_formal_once(
    lock_path: Path,
    *,
    stage: str,
    manifest_path: Path,
) -> dict[str, Any]:
    """Atomically bind a completed formal manifest to its start ledger."""

    normalized_stage = str(stage).upper()
    if normalized_stage != "TEST" and not re.fullmatch(
        r"[A-Z][A-Z0-9_]{1,63}", normalized_stage
    ):
        raise ValueError("formal stage must be a short alphanumeric identifier")
    lock = verify_lock(lock_path)
    resolved_lock = lock_path.expanduser().resolve()
    start_name = (
        f"{resolved_lock.name}.FORMAL_TEST_STARTED.json"
        if normalized_stage == "TEST"
        else f"{resolved_lock.name}.FORMAL_{normalized_stage}_STARTED.json"
    )
    start_ledger = resolved_lock.with_name(start_name)
    if not start_ledger.is_file():
        raise FileNotFoundError("formal completion requires its start ledger")
    started = json.loads(start_ledger.read_text(encoding="utf-8"))
    _validate_same_locked_identity(
        started,
        lock,
        context=f"formal {normalized_stage} start ledger",
    )
    manifest = manifest_path.expanduser().resolve()
    expected_root = Path(str(started.get("output_root", ""))).resolve()
    if (
        started.get("lock_content_sha256") != lock["manifest_content_sha256"]
        or manifest.parent != expected_root
        or not manifest.is_file()
    ):
        raise ValueError("formal completion manifest is outside its guarded output")
    completed_manifest = _read_json_mapping(
        manifest,
        context=f"formal {normalized_stage} completion manifest",
    )
    _validate_same_locked_identity(
        completed_manifest,
        lock,
        context=f"formal {normalized_stage} completion manifest",
    )
    completed_name = (
        f"{resolved_lock.name}.FORMAL_TEST_COMPLETED.json"
        if normalized_stage == "TEST"
        else f"{resolved_lock.name}.FORMAL_{normalized_stage}_COMPLETED.json"
    )
    completed_ledger = resolved_lock.with_name(completed_name)
    payload = {
        **_locked_identity(lock, context="formal experiment lock"),
        "lock_path": str(resolved_lock),
        "lock_content_sha256": lock["manifest_content_sha256"],
        "stage": normalized_stage,
        "start_ledger": str(start_ledger),
        "start_ledger_sha256": sha256_file(start_ledger),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    if completed_ledger.exists():
        existing = json.loads(completed_ledger.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"formal stage {normalized_stage} completion changed"
            )
        return existing
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        completed_ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        completed_ledger.unlink(missing_ok=True)
        raise
    return payload


def complete_formal_test_once(
    lock_path: Path, *, manifest_path: Path
) -> dict[str, Any]:
    return _complete_formal_once(
        lock_path, stage="TEST", manifest_path=manifest_path
    )


def complete_formal_stage_once(
    lock_path: Path, *, stage: str, manifest_path: Path
) -> dict[str, Any]:
    return _complete_formal_once(
        lock_path, stage=stage, manifest_path=manifest_path
    )
