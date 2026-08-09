from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import src.grasping.reranking_v1.artifact_contract as artifact_contract
from src.grasping.reranking_v1.artifact_contract import (
    CONFIG_PROTOCOL_METHODS,
    LOCAL_VLM_GEOMETRY_SEMANTICS,
    SCALER_MODEL_ARTIFACT_KEYS,
    build_final_scaler_metadata,
    identity_payload,
)
from src.grasping.reranking_v1.experiment_lock import (
    build_lock_manifest,
    canonical_json_sha256,
    complete_formal_stage_once,
    complete_formal_test_once,
    consume_formal_stage_once,
    consume_formal_test_once,
    sha256_file,
    verify_completed_formal_stage,
    verify_lock,
    write_lock_exclusive,
)
from src.grasping.reranking_v1.local_vlm import (
    SESSION_AUDIT_POLICY_VERSION,
    stable_session_contract,
    stable_session_contract_sha256,
)
from src.grasping.reranking_v1.method_namespace import (
    expected_formal_method_protocols,
)


def _tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _spec(
    tmp_path: Path,
    source: Path,
    artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    from src.grasping.reranking_v1.experiment_lock import (
        REQUIRED_ARTIFACT_KEYS,
        sha256_file,
    )

    source_run = tmp_path / "hifics_hierfilm_source"
    checkpoint = source_run / "checkpoints" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    (source_run / ".DO_NOT_PRUNE").write_text("protected\n")
    source_snapshot = source_run / "source_snapshot"
    source_snapshot.mkdir()
    training_config = {
        "architecture": "models.hifics.HierarchicalCLIPDensePredT",
        "selected_visual_layers": [1, 3, 5, 7, 9],
        "film_injection_count": 5,
    }
    run_config = source_run / "config.yaml"
    frozen_config = (
        source_snapshot / "hifics_ocidvlg_hierfilm_controlled.yaml"
    )
    run_config.write_text(
        json.dumps(training_config, sort_keys=True) + "\n"
    )
    frozen_config.write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n"
    )
    for name in (
        "hifics.py",
        "dataloader.py",
        "train_hierfilm.py",
        "verify_hierarchical_film.py",
    ):
        (source_snapshot / name).write_text(f"# frozen {name}\n")
    source_snapshot_hashes = {
        path.name: sha256_file(path)
        for path in sorted(source_snapshot.iterdir())
        if path.is_file()
    }
    source_snapshot_sha256 = canonical_json_sha256(
        source_snapshot_hashes
    )
    (source_run / "source_snapshot_sha256.json").write_text(
        json.dumps(source_snapshot_hashes, indent=2, sort_keys=True) + "\n"
    )
    (source_run / "source_snapshot_aggregate_sha256.txt").write_text(
        source_snapshot_sha256 + "\n"
    )
    config_canonical_sha256 = canonical_json_sha256(training_config)
    (source_run / "config_sha256.txt").write_text(
        config_canonical_sha256 + "\n"
    )
    split_paths: dict[str, Path] = {}
    split_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}_frozen_manifest.json"
        path.write_text(json.dumps([{"sample_id": f"{split}-1"}]) + "\n")
        split_paths[split] = path
        split_hashes[split] = sha256_file(path)
    clip_checkpoint = tmp_path / "ViT-B-16.pt"
    clip_checkpoint.write_bytes(b"frozen clip checkpoint")
    trainable_state = {
        f"film_stages.{stage}.{branch}.weight": torch.zeros(64, 512)
        for stage in range(5)
        for branch in ("alpha", "beta")
    }
    trainable_state.update(
        {
            f"film_stages.{stage}.{branch}.bias": torch.zeros(64)
            for stage in range(5)
            for branch in ("alpha", "beta")
        }
    )
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
    trainable_state.update(
        {
            "trans_conv.weight": torch.zeros(64, 1, 16, 16),
            "trans_conv.bias": torch.zeros(1),
            **{
                f"reduces.{stage}.weight": torch.zeros(64, 768)
                for stage in range(5)
            },
            **{
                f"reduces.{stage}.bias": torch.zeros(64)
                for stage in range(5)
            },
            **{
                f"blocks.{stage}.{suffix}": torch.zeros(shape)
                for stage in range(5)
                for suffix, shape in block_shapes.items()
            },
        }
    )
    trainable_state_sha256 = _tensor_state_sha256(trainable_state)
    torch.save(
        {
            "format": "hifics_hierfilm_trainable_only_v1",
            "trainable_state": trainable_state,
            "metadata": {
                "global_step": 19728,
                "epoch": 12,
                "best_validation_iou": 0.82,
                "source_snapshot_sha256": source_snapshot_sha256,
                "config_sha256": config_canonical_sha256,
                "current_trainable_state_sha256": trainable_state_sha256,
                "manifest_sha256": split_hashes,
                "weight_sources": {
                    "clip": {
                        "cache_path": str(clip_checkpoint),
                        "cache_sha256": sha256_file(clip_checkpoint),
                        "frozen": True,
                    },
                    "decoder_film_head": {
                        "kind": "fresh deterministic initialization",
                        "checkpoint_loaded": False,
                    },
                },
            },
        },
        checkpoint,
    )
    checkpoint_sha256 = sha256_file(checkpoint)
    monkeypatch.setattr(
        artifact_contract,
        "REPEATEDFILM_SOURCE_CHECKPOINT_SHA256",
        checkpoint_sha256,
    )
    monkeypatch.setattr(
        artifact_contract,
        "REPEATEDFILM_SOURCE_SNAPSHOT_SHA256",
        source_snapshot_sha256,
    )

    def source_identity() -> dict:
        return identity_payload(
            source_checkpoint_sha256=checkpoint_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
        )

    config = {
        "schema_version": 2,
        "experiment": "modular_reranking_repeatedfilm_v1",
        "public_method_namespace_version": 1,
        "method_namespace": "repeatedfilm_only",
        "baseline_name": "repeatedfilm_gqcnn_q_only",
        "lineage": {
            "visual_grounding": "hierarchical_repeated_film",
            "source_run": str(source_run),
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "legacy_singlefilm_allowed": False,
            "legacy_singlefilm_metrics_comparable": False,
            "recover_or_regenerate_singlefilm": False,
            "mix_lineages": False,
        },
        "splits": {
            "protocol": "official OCID-VLG unique train/val/test",
            "manifest_set_sha256": "1" * 64,
            **{
                split: {
                    "manifest": str(path),
                    "manifest_sha256": split_hashes[split],
                    "count": 1,
                }
                for split, path in split_paths.items()
            },
        },
        "hifi_inference": {
            "architecture": "models.hifics.HierarchicalCLIPDensePredT",
            "selected_visual_layers": [1, 3, 5, 7, 9],
            "decoder_order": [9, 7, 5, 3, 1],
            "independent_film_stages": 5,
        },
        "retained_test": {
            "regenerate_hifi": False,
            "regenerate_dexnet": False,
            "regenerate_gqcnn": False,
        },
        "protocols": {
            "full_nms": {
                "baseline": "repeatedfilm_gqcnn_q_only",
                "methods": list(CONFIG_PROTOCOL_METHODS["full_nms"]),
            },
            "gqcnn_top5": {
                "baseline": "repeatedfilm_gqcnn_q_top5",
                "methods": list(CONFIG_PROTOCOL_METHODS["gqcnn_top5"]),
            },
        },
        "safe_switch": {
            "harmful_rate_limit_all_samples": 0.01,
            "fallback": "repeatedfilm_gqcnn_q_only",
        },
        "primary_selection": {
            "maximum_inference_seconds_per_sample": 0.5,
            "inference_device": "auto",
        },
        "learned_rankers": {"seed": 42},
        "evaluation": {"bootstrap_seed": 42, "bootstrap_draws": 10000},
        "local_vlm": {
            "seed": 20260729,
            "temperature": 0.0,
            "stream": False,
            "thinking": False,
            "top_k_pool": 5,
            "maximum_memory_mib": 20_000.0,
            "maximum_p95_seconds": 180.0,
            "max_output_tokens": 768,
            "safe_switch": {
                "geometry_risk_column": "collision_proxy_total",
                "geometry_risk_threshold": 0.5,
                "geometry_semantics": LOCAL_VLM_GEOMETRY_SEMANTICS,
            },
            "cloud_disabled": True,
            "fallback": "repeatedfilm_gqcnn_q_top5",
            "variants": ["visual", "visual_metadata"],
        },
        "formal_test": {
            "configuration_lock_required": True,
            "invocation_limit": 1,
        },
    }
    artifact.write_text(json.dumps(config) + "\n", encoding="utf-8")
    repeatedfilm_source_manifest = tmp_path / "repeatedfilm_source_manifest.json"
    repeatedfilm_source_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit_status": "COMPLETE_AND_REUSABLE",
                "visual_grounding_variant": "hierarchical_repeated_film",
                "single_film_allowed": False,
                "single_film_artifacts_intentionally_deleted": True,
                "legacy_singlefilm_baseline_reused": False,
                "source_run_id": source_run.name,
                "source_run_path": str(source_run),
                "source_run_protection_marker": str(
                    source_run / ".DO_NOT_PRUNE"
                ),
                "checkpoint_path": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_format": "hifics_hierfilm_trainable_only_v1",
                "checkpoint_step": 19728,
                "checkpoint_epoch": 12,
                "best_validation_mean_iou": 0.82,
                "checkpoint_trainable_state_sha256": trainable_state_sha256,
                "checkpoint_strict_load_success": True,
                "checkpoint_expected_keys": 92,
                "checkpoint_loaded_keys": 92,
                "checkpoint_missing_keys": [],
                "checkpoint_unexpected_keys": [],
                "config_run_raw_sha256": sha256_file(run_config),
                "config_frozen_raw_sha256": sha256_file(frozen_config),
                "config_canonical_sha256": config_canonical_sha256,
                "source_snapshot_sha256": source_snapshot_sha256,
                "architecture_class": (
                    "models.hifics.HierarchicalCLIPDensePredT"
                ),
                "selected_visual_layers": [1, 3, 5, 7, 9],
                "selected_visual_layers_decoder_order": [9, 7, 5, 3, 1],
                "film_injection_count": 5,
                "projection_count": 5,
                "decoder_count": 5,
                "film_parameter_sharing": (
                    "independent alpha/beta linear pair per level"
                ),
                "legacy_singlefilm_attributes_present": False,
                "dataset_protocol": (
                    "official OCID-VLG unique train/val/test"
                ),
                "manifest_set_sha256": "1" * 64,
                "split_manifests": {
                    split: {
                        "path": str(path),
                        "records": 1,
                        "sha256": split_hashes[split],
                    }
                    for split, path in split_paths.items()
                },
                "split_overlap_audit_passed": True,
                "clip_checkpoint_path": str(clip_checkpoint),
                "clip_checkpoint_sha256": sha256_file(clip_checkpoint),
            }
        )
        + "\n"
    )
    prompt, schema, audit = (
        tmp_path / "prompt.txt",
        tmp_path / "schema.json",
        tmp_path / "audit.json",
    )
    prompt.write_text("prompt\n")
    schema.write_text("{}\n")
    audit_payload = {
        "session_audit_policy_version": SESSION_AUDIT_POLICY_VERSION,
        "local_only": True,
        "remote_api_used": False,
        "machine": {"architecture": "arm64"},
        "ollama": {
            "cli_version": "0.13.5",
            "api_version": {"version": "0.13.5"},
            "executable_path": "/Applications/Ollama.app/ollama",
            "executable_sha256": "a" * 64,
            "endpoint": "http://127.0.0.1:11434",
            "server_config_path": "/tmp/server.json",
            "server_config_sha256": "b" * 64,
            "server_config": {"disable_ollama_cloud": True},
            "remote_established_connections": [],
        },
        "model": {
            "exact_name": "qwen3-vl:4b-instruct-q4_K_M",
            "manifest_sha256": "model",
            "model_layer_digest": "sha256:" + "c" * 64,
            "model_layer_bytes": 100,
            "quantization": "Q4_K_M",
            "layers": [
                {
                    "digest": "sha256:" + "c" * 64,
                    "mediaType": "application/vnd.ollama.image.model",
                    "size": 100,
                    "local_size": 100,
                    "local_sha256": "c" * 64,
                }
            ],
        },
    }
    audit_payload["stable_session_contract"] = stable_session_contract(
        audit_payload
    )
    stable_hash = stable_session_contract_sha256(audit_payload)
    audit_payload["stable_session_contract_sha256"] = stable_hash
    audit.write_text(json.dumps(audit_payload) + "\n")
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps({"ground_truth_allowed": False, "features": ["q_raw"]}) + "\n"
    )
    scaler_payload = {
        "feature_columns": ["q_raw"],
        "mean": [0.25],
        "scale": [0.5],
        "source_splits": ["development"],
        "fit_scope": "train/development candidates only",
    }
    model_paths: dict[str, Path] = {}
    model_artifacts: dict[str, dict] = {}
    for method in SCALER_MODEL_ARTIFACT_KEYS:
        model_artifact = {
            "method": method,
            "scaler": scaler_payload,
        }
        model_artifacts[method] = model_artifact
        suffix = (
            ".json"
            if method == "regularized_linear_ranker"
            else ".pt"
        )
        model_path = tmp_path / f"{method}{suffix}"
        if suffix == ".json":
            model_path.write_text(json.dumps(model_artifact) + "\n")
        else:
            torch.save(
                {"artifact": model_artifact, "state_dict": {}},
                model_path,
            )
        model_paths[method] = model_path
    scaler_metadata_path = tmp_path / "scaler_metadata.json"
    scaler_metadata = build_final_scaler_metadata(
        model_artifacts, model_paths
    )
    scaler_metadata_path.write_text(
        json.dumps(scaler_metadata, indent=2, sort_keys=True) + "\n"
    )
    inference_bundle = tmp_path / "inference_bundle.json"
    inference_bundle.write_text(
        json.dumps(
            {
                **source_identity(),
                "feature_columns": ["q_raw"],
                "candidate_pool_modified": False,
                "reload_parity_verified": True,
                "primary_candidate_methods": [
                    "repeatedfilm_residual_mlp_safe_switch"
                ],
                "rule_methods": {
                    "q_softmask_rule": {"alpha": 0.8, "beta": 0.2}
                },
                "scaler_metadata": scaler_metadata_path.name,
                "scaler_metadata_sha256": sha256_file(
                    scaler_metadata_path
                ),
            }
        )
        + "\n"
    )
    evaluation_config = tmp_path / "evaluation.yaml"
    evaluation_config.write_text(
        "iou_threshold: 0.25\nangle_threshold_deg: 30.0\ntop_k: 5\n"
    )
    universe = tmp_path / "test_universe.csv"
    universe.write_text("sample_id,scene_id\ns1,a\ns2,b\n")
    validation_candidates = tmp_path / "rule_validation.parquet"
    validation_candidates.write_bytes(b"validation")
    safe = tmp_path / "safe.json"
    safe_payload = {
        **source_identity(),
        "selection_split": "validation",
        "threshold": 0.9,
        "force_no_switch": True,
        "harmful_rate_limit": 0.01,
        "harmful_rate_denominator": "all_validation_samples",
        "validation_per_candidate": str(validation_candidates),
        "validation_per_candidate_sha256": sha256_file(
            validation_candidates
        ),
    }
    safe.write_text(json.dumps(safe_payload) + "\n")
    rule_validation = validation_candidates
    rule_sweep = tmp_path / "rule_sweep.csv"
    rule_sweep.write_text("alpha,beta\n0.8,0.2\n")
    rule_selection = tmp_path / "rule_selection.json"
    rule_selection.write_text(
        json.dumps(
            {
                **source_identity(),
                "selection_split": "validation",
                "method": "q_softmask_rule",
                "selected": {"alpha": 0.8, "beta": 0.2},
                "validation_per_candidate": str(rule_validation),
                "validation_per_candidate_sha256": sha256_file(rule_validation),
                "sweep": str(rule_sweep),
                "sweep_sha256": sha256_file(rule_sweep),
            }
        )
        + "\n"
    )
    split_audit = tmp_path / "split_audit.json"
    split_audit.write_text(
        json.dumps({"required_intersections_all_zero": True}) + "\n"
    )
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps(
            {
                **source_identity(),
                "candidate_pool_modified": False,
                "primary_candidate_methods": [
                    "repeatedfilm_residual_mlp_safe_switch"
                ],
                "validation_per_candidate": str(validation_candidates),
                "validation_sha256": sha256_file(validation_candidates),
                "validation_per_sample": str(universe),
                "validation_per_sample_sha256": sha256_file(universe),
                "rule_selection": str(rule_selection),
                "rule_selection_sha256": sha256_file(rule_selection),
                "safe_switch_selection": safe_payload,
                "safe_switch_selection_path": str(safe),
                "safe_switch_selection_sha256": sha256_file(safe),
                "scaler_metadata": str(scaler_metadata_path),
                "scaler_metadata_sha256": sha256_file(
                    scaler_metadata_path
                ),
                "scaler_sha256": scaler_metadata["scaler_sha256"],
            }
        )
        + "\n"
    )
    train_split_manifest = tmp_path / "train_split_manifest.json"
    train_split_manifest.write_text('{"split":"train"}\n')
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                **source_identity(),
                "sample_count": 2,
                "methods": {
                    "repeatedfilm_residual_mlp_safe_switch": {
                        "inference_seconds_per_sample": 0.1,
                        "protocol": "full_nms",
                        "measurement": "conservative_complete_suite_upper_bound",
                    }
                },
                "device_requested": "auto",
                "inputs": {
                    "per_candidate": str(validation_candidates),
                    "per_candidate_sha256": sha256_file(
                        validation_candidates
                    ),
                    "sample_universe": str(universe),
                    "sample_universe_sha256": sha256_file(universe),
                    "training_manifest": str(training),
                    "training_manifest_sha256": sha256_file(training),
                    "inference_bundle": str(inference_bundle),
                    "inference_bundle_sha256": sha256_file(inference_bundle),
                },
            }
        )
        + "\n"
    )
    validation_bundle = tmp_path / "validation_bundle.json"
    validation_bundle.write_text(
        json.dumps(
            {
                **source_identity(),
                "report_recomputation_verified": True,
            }
        )
        + "\n"
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                **source_identity(),
                "maximum_inference_seconds_per_sample": 0.5,
                "formal_inference_device": "auto",
                "source_artifacts": {
                    role: {
                        "path": str(path),
                        "sha256": sha256_file(path),
                    }
                    for role, path in {
                        "split_audit": split_audit,
                        "feature_allowlist": allowlist,
                        "training_manifest": training,
                        "runtime_benchmark": runtime,
                    }.items()
                }
            }
        )
        + "\n"
    )
    primary = tmp_path / "primary.json"
    primary_inputs = {
        "evaluation_bundle": str(validation_bundle),
        "evaluation_bundle_sha256": sha256_file(validation_bundle),
        "sample_universe": str(universe),
        "sample_universe_sha256": sha256_file(universe),
        "eligibility_evidence": str(evidence),
        "eligibility_evidence_sha256": sha256_file(evidence),
    }
    primary.write_text(
        json.dumps(
            {
                **source_identity(),
                "selection_split": "validation",
                "primary_method": "repeatedfilm_residual_mlp_safe_switch",
                "baseline_fallback": False,
                "harmful_rate_limit_all_samples": 0.01,
                "maximum_inference_seconds_per_sample": 0.5,
                "formal_inference_device": "auto",
                "inputs": primary_inputs,
                "selected_metrics": {
                    gate: True
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
                },
            }
        )
        + "\n"
    )
    artifact_paths = {
        name: str(artifact) for name in REQUIRED_ARTIFACT_KEYS
    }
    vlm_results = tmp_path / "vlm_results.jsonl"
    vlm_results.write_text("{}\n")
    vlm_summary = tmp_path / "vlm_summary.json"
    vlm_summary.write_text(json.dumps(source_identity()) + "\n")
    vlm_runtime = tmp_path / "vlm_runtime.json"
    vlm_runtime.write_text(json.dumps(source_identity()) + "\n")
    vlm_safe = tmp_path / "vlm_safe.json"
    vlm_safe.write_text(
        json.dumps(
            {
                **source_identity(),
                "selection_split": "validation",
                "method": "repeatedfilm_local_vlm_safe_switch",
                "source_method": "repeatedfilm_local_vlm_visual",
                "source_variant": "visual",
                "threshold_kind": "never_switch",
                "threshold": None,
                "geometry_risk_column": "collision_proxy_total",
                "geometry_risk_threshold": 0.5,
                "geometry_semantics": LOCAL_VLM_GEOMETRY_SEMANTICS,
                "config": str(artifact.resolve()),
                "config_sha256": sha256_file(artifact),
                "harmful_rate_limit": 0.01,
                "harmful_rate_denominator": "all_validation_samples",
                "vlm_model_digest": "sha256:model",
                "stable_session_contract_sha256": stable_hash,
                "vlm_runtime_local_only_passed": True,
                "inputs": {
                    "vlm_results": str(vlm_results),
                    "vlm_results_sha256": sha256_file(vlm_results),
                    "vlm_summary": str(vlm_summary),
                    "vlm_summary_sha256": sha256_file(vlm_summary),
                    "vlm_runtime_metrics": str(vlm_runtime),
                    "vlm_runtime_metrics_sha256": sha256_file(vlm_runtime),
                },
            }
        )
        + "\n"
    )
    repeat_comparison = tmp_path / "repeat_comparison.json"
    repeat_comparison.write_text(json.dumps(source_identity()) + "\n")
    model_candidates = [{"size_class": "4B", "eligible": True}]
    model_audit_sources = [
        {
            "role": role,
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for role, path in (
            ("4B_local_audit", audit),
            ("4B_pilot_summary", vlm_summary),
            ("4B_pilot_runtime", vlm_runtime),
            ("4B_repeat_a", vlm_results),
            ("4B_repeat_b", vlm_results),
            ("4B_repeat_a_runtime", vlm_runtime),
            ("4B_repeat_b_runtime", vlm_runtime),
            ("4B_repeat_comparison", repeat_comparison),
        )
    ]
    model_audit = tmp_path / "model_candidates.json"
    model_audit.write_text(
        json.dumps(
            {
                **source_identity(),
                "selection_split": "validation",
                "config": str(artifact.resolve()),
                "config_sha256": sha256_file(artifact),
                "maximum_memory_mib": 20_000.0,
                "maximum_p95_seconds": 180.0,
                "model_candidates": model_candidates,
                "source_artifacts": model_audit_sources,
            }
        )
        + "\n"
    )
    vlm_validation = tmp_path / "vlm_validation.json"
    vlm_validation.write_text(
        json.dumps(
            {
                **source_identity(),
                "selection_split": "validation",
                "config": str(artifact.resolve()),
                "config_sha256": sha256_file(artifact),
                "maximum_memory_mib": 20_000.0,
                "maximum_p95_seconds": 180.0,
                "selected_model_digest": "sha256:model",
                "selected_stable_session_contract_sha256": stable_hash,
                "selected_method": "repeatedfilm_local_vlm_visual",
                "selected_variant": "visual",
                "model_candidates": model_candidates,
                "candidates": [
                    {
                        "method": "repeatedfilm_local_vlm_visual",
                        "variant": "visual",
                        "model_digest": "sha256:model",
                        "stable_session_contract_sha256": stable_hash,
                        "eligible": True,
                        "resource_pass": True,
                        "results": str(vlm_results),
                        "results_sha256": sha256_file(vlm_results),
                        "summary": str(vlm_summary),
                        "summary_sha256": sha256_file(vlm_summary),
                        "runtime_metrics": str(vlm_runtime),
                        "runtime_metrics_sha256": sha256_file(vlm_runtime),
                    }
                ],
                "inputs": {
                    "evaluation_bundle": str(validation_bundle),
                    "evaluation_bundle_sha256": sha256_file(validation_bundle),
                    "model_candidate_audit": str(model_audit),
                    "model_candidate_audit_sha256": sha256_file(model_audit),
                },
            }
        )
        + "\n"
    )
    vlm_input_artifacts: dict[str, Path] = {}
    for variant, include_metadata in (
        ("visual", False),
        ("visual_metadata", True),
    ):
        input_path = tmp_path / f"test_inputs_{variant}.jsonl"
        input_rows = [
            {
                "sample_id": sample_id,
                "candidate_ids": ["g0"],
                "candidate_metadata": (
                    {"g0": {"q_percentile": 0.5}}
                    if include_metadata
                    else {}
                ),
                "include_metadata": include_metadata,
                "gt_fields_included": False,
            }
            for sample_id in ("s1", "s2")
        ]
        input_path.write_text(
            "".join(json.dumps(row) + "\n" for row in input_rows)
        )
        aggregate_path = tmp_path / f"aggregate_{variant}.json"
        aggregate_path.write_text(
            json.dumps(
                {
                    "gt_free": True,
                    **source_identity(),
                    "sample_count": 2,
                    "nonempty_sample_count": 2,
                    "empty_sample_count": 0,
                    "visuals": [],
                }
            )
            + "\n"
        )
        summary_path = tmp_path / f"input_summary_{variant}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    **source_identity(),
                    "samples": 2,
                    "gt_fields_included": False,
                    "include_metadata": include_metadata,
                    "output_jsonl": str(input_path),
                    "output_jsonl_sha256": sha256_file(input_path),
                    "aggregate_visual_manifest": str(aggregate_path),
                    "aggregate_visual_manifest_sha256": sha256_file(
                        aggregate_path
                    ),
                }
            )
            + "\n"
        )
        vlm_input_artifacts[
            f"vlm_test_input_manifest_{variant}"
        ] = summary_path
        vlm_input_artifacts[
            f"vlm_visual_manifest_{variant}"
        ] = aggregate_path
    artifact_paths.update(
        {
            "vlm_prompt": str(prompt),
            "vlm_json_schema": str(schema),
            "repeatedfilm_source_manifest": str(
                repeatedfilm_source_manifest
            ),
            "local_ollama_audit": str(audit),
            "primary_selection": str(primary),
            "feature_allowlist": str(allowlist),
            "inference_bundle": str(inference_bundle),
            "safe_switch_selection": str(safe),
            "evaluation_config": str(evaluation_config),
            "test_sample_universe": str(universe),
            "split_audit": str(split_audit),
            "train_manifest": str(train_split_manifest),
            "reranker_training_manifest": str(training),
            "runtime_benchmark": str(runtime),
            "validation_evaluation_bundle": str(validation_bundle),
            "rule_selection": str(rule_selection),
            "scaler_metadata": str(scaler_metadata_path),
            "vlm_safe_switch_selection": str(vlm_safe),
            "vlm_validation_selection": str(vlm_validation),
            **{
                artifact_key: str(model_paths[method])
                for method, artifact_key in (
                    SCALER_MODEL_ARTIFACT_KEYS.items()
                )
            },
            **{
                name: str(path)
                for name, path in vlm_input_artifacts.items()
            },
        }
    )
    return {
        "primary_method": "repeatedfilm_residual_mlp_safe_switch",
        "selected_feature_list": ["q_raw"],
        "safe_switch_threshold": 0.9,
        "vlm_backend": "ollama",
        "vlm_model_digest": "sha256:model",
        "prompt_hash": sha256_file(prompt),
        "json_schema_hash": sha256_file(schema),
        "seeds": {"model": 42, "vlm": 20260729, "bootstrap": 42},
        "evaluation_definition": {
            "iou_operator": ">",
            "iou_threshold": 0.25,
            "angle_operator": "<=",
            "angle_threshold_deg": 30.0,
            "top_k": 5,
            "bootstrap_replicates": 10_000,
            "formal_method_protocols": [
                {"protocol": protocol, "method": method}
                for protocol, method in expected_formal_method_protocols(
                    "repeatedfilm_local_vlm_visual"
                )
            ],
        },
        "expected_test_sample_count": 2,
        "source_paths": [str(source)],
        "ranking_parameters": {
            "safe_switch_force_no_switch": True,
            "tabular_inference_device": "auto",
            "vlm_safe_switch": {
                "method": "repeatedfilm_local_vlm_safe_switch",
                "threshold_kind": "never_switch",
                "threshold": None,
                "geometry_risk_column": "collision_proxy_total",
                "geometry_risk_threshold": 0.5,
                "geometry_semantics": LOCAL_VLM_GEOMETRY_SEMANTICS,
            },
            "vlm_top_k": 5,
            "vlm_max_output_tokens": 768,
            "vlm_temperature": 0.0,
            "vlm_stream": False,
            "vlm_think": False,
            "mmr_enabled": False,
            "mmr_lambda": None,
        },
        "artifact_paths": artifact_paths,
    }


def test_lock_is_exclusive_and_detects_artifact_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    manifest = build_lock_manifest(spec, repo_root=tmp_path)
    config = json.loads(
        Path(spec["artifact_paths"]["config"]).read_text(encoding="utf-8")
    )
    expected_identity = identity_payload(
        source_checkpoint_sha256=config["lineage"][
            "source_checkpoint_sha256"
        ],
        source_snapshot_sha256=config["lineage"][
            "source_snapshot_sha256"
        ],
    )
    assert {
        key: manifest[key] for key in expected_identity
    } == expected_identity
    assert (
        manifest["selected_vlm"]["source_method"]
        == "repeatedfilm_local_vlm_visual"
    )
    assert manifest["selected_vlm"]["source_variant"] == "visual"
    assert (
        manifest["selected_vlm"]["validation_selection_sha256"]
        == manifest["artifacts"]["vlm_validation_selection"]["sha256"]
    )
    lock = tmp_path / "frozen_experiment_manifest.json"
    write_lock_exclusive(lock, manifest)
    assert (
        verify_lock(lock)["primary_method"]
        == "repeatedfilm_residual_mlp_safe_switch"
    )
    with pytest.raises(FileExistsError):
        write_lock_exclusive(lock, manifest)
    artifact.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="locked artifact changed"):
        verify_lock(lock)


def test_lock_rejects_cross_lineage_primary_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    primary_path = Path(spec["artifact_paths"]["primary_selection"])
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary["baseline_name"] = "q_only"
    primary_path.write_text(json.dumps(primary) + "\n")
    with pytest.raises(ValueError, match="primary selection.*baseline_name"):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_verify_lock_rejects_rehashed_nested_cross_lineage_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    manifest = build_lock_manifest(spec, repo_root=tmp_path)
    lock = tmp_path / "lock.json"
    write_lock_exclusive(lock, manifest)

    training_path = Path(
        spec["artifact_paths"]["reranker_training_manifest"]
    )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["baseline_name"] = "q_only"
    training_path.write_text(json.dumps(training) + "\n")

    rehashed_lock = json.loads(lock.read_text(encoding="utf-8"))
    rehashed_lock["artifacts"]["reranker_training_manifest"][
        "sha256"
    ] = sha256_file(training_path)
    rehashed_lock.pop("manifest_content_sha256")
    rehashed_lock["manifest_content_sha256"] = canonical_json_sha256(
        rehashed_lock
    )
    lock.chmod(0o644)
    lock.write_text(json.dumps(rehashed_lock) + "\n")

    with pytest.raises(
        ValueError,
        match="reranker training manifest.*baseline_name",
    ):
        verify_lock(lock)


def test_verify_lock_rejects_rehashed_deep_eligibility_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    manifest = build_lock_manifest(spec, repo_root=tmp_path)
    lock = tmp_path / "lock.json"
    write_lock_exclusive(lock, manifest)

    primary_path = Path(spec["artifact_paths"]["primary_selection"])
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    evidence_path = Path(primary["inputs"]["eligibility_evidence"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["baseline_name"] = "q_only"
    evidence_path.write_text(json.dumps(evidence) + "\n")
    primary["inputs"]["eligibility_evidence_sha256"] = sha256_file(
        evidence_path
    )
    primary_path.write_text(json.dumps(primary) + "\n")

    rehashed_lock = json.loads(lock.read_text(encoding="utf-8"))
    rehashed_lock["artifacts"]["primary_selection"][
        "sha256"
    ] = sha256_file(primary_path)
    rehashed_lock.pop("manifest_content_sha256")
    rehashed_lock["manifest_content_sha256"] = canonical_json_sha256(
        rehashed_lock
    )
    lock.chmod(0o644)
    lock.write_text(json.dumps(rehashed_lock) + "\n")

    with pytest.raises(
        ValueError,
        match="tabular eligibility evidence.*baseline_name",
    ):
        verify_lock(lock)


def test_verify_lock_rejects_rehashed_vlm_model_audit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    manifest = build_lock_manifest(spec, repo_root=tmp_path)
    lock = tmp_path / "lock.json"
    write_lock_exclusive(lock, manifest)

    selection_path = Path(
        spec["artifact_paths"]["vlm_validation_selection"]
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit_path = Path(selection["inputs"]["model_candidate_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["baseline_name"] = "q_only"
    audit_path.write_text(json.dumps(audit) + "\n")
    selection["inputs"]["model_candidate_audit_sha256"] = sha256_file(
        audit_path
    )
    selection_path.write_text(json.dumps(selection) + "\n")

    rehashed_lock = json.loads(lock.read_text(encoding="utf-8"))
    selection_sha256 = sha256_file(selection_path)
    rehashed_lock["artifacts"]["vlm_validation_selection"][
        "sha256"
    ] = selection_sha256
    rehashed_lock["selected_vlm"][
        "validation_selection_sha256"
    ] = selection_sha256
    rehashed_lock.pop("manifest_content_sha256")
    rehashed_lock["manifest_content_sha256"] = canonical_json_sha256(
        rehashed_lock
    )
    lock.chmod(0o644)
    lock.write_text(json.dumps(rehashed_lock) + "\n")

    with pytest.raises(
        ValueError,
        match="VLM model candidate audit.*baseline_name",
    ):
        verify_lock(lock)


def test_lock_rejects_public_artifact_from_another_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    training_path = Path(
        spec["artifact_paths"]["reranker_training_manifest"]
    )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["source_snapshot_sha256"] = "0" * 64
    training_path.write_text(json.dumps(training) + "\n")

    with pytest.raises(
        ValueError,
        match=(
            "reranker training manifest source_snapshot_sha256 differs "
            "from the approved"
        ),
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_self_consistent_unapproved_checkpoint_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    source_manifest_path = Path(
        spec["artifact_paths"]["repeatedfilm_source_manifest"]
    )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint_path = Path(source_manifest["checkpoint_path"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    checkpoint["trainable_state"]["film_stages.0.alpha.bias"][0] = 1.0
    trainable_state_sha256 = _tensor_state_sha256(
        checkpoint["trainable_state"]
    )
    checkpoint["metadata"][
        "current_trainable_state_sha256"
    ] = trainable_state_sha256
    torch.save(checkpoint, checkpoint_path)
    replacement_sha256 = sha256_file(checkpoint_path)
    source_manifest.update(
        {
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": replacement_sha256,
            "checkpoint_trainable_state_sha256": trainable_state_sha256,
        }
    )
    source_manifest_path.write_text(json.dumps(source_manifest) + "\n")
    config_path = Path(spec["artifact_paths"]["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["lineage"]["source_checkpoint_sha256"] = replacement_sha256
    config_path.write_text(json.dumps(config) + "\n")

    with pytest.raises(
        ValueError,
        match="differs from the approved repeated-FiLM source",
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_checkpoint_missing_a_repeated_film_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    source_manifest_path = Path(
        spec["artifact_paths"]["repeatedfilm_source_manifest"]
    )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint_path = Path(source_manifest["checkpoint_path"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    checkpoint["trainable_state"].pop("film_stages.4.beta.bias")
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    source_manifest.update(
        {
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_expected_keys": 91,
            "checkpoint_loaded_keys": 91,
        }
    )
    source_manifest_path.write_text(json.dumps(source_manifest) + "\n")
    config_path = Path(spec["artifact_paths"]["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["lineage"]["source_checkpoint_sha256"] = checkpoint_sha256
    config_path.write_text(json.dumps(config) + "\n")
    monkeypatch.setattr(
        artifact_contract,
        "REPEATEDFILM_SOURCE_CHECKPOINT_SHA256",
        checkpoint_sha256,
    )

    with pytest.raises(
        ValueError,
        match="live checkpoint does not prove.*repeated-FiLM source",
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_checkpoint_with_invalid_film_tensor_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    source_manifest_path = Path(
        spec["artifact_paths"]["repeatedfilm_source_manifest"]
    )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint_path = Path(source_manifest["checkpoint_path"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    checkpoint["trainable_state"]["film_stages.4.beta.bias"] = (
        torch.zeros(1)
    )
    trainable_state_sha256 = _tensor_state_sha256(
        checkpoint["trainable_state"]
    )
    checkpoint["metadata"][
        "current_trainable_state_sha256"
    ] = trainable_state_sha256
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    source_manifest.update(
        {
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_trainable_state_sha256": (
                trainable_state_sha256
            ),
        }
    )
    source_manifest_path.write_text(json.dumps(source_manifest) + "\n")
    config_path = Path(spec["artifact_paths"]["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["lineage"]["source_checkpoint_sha256"] = checkpoint_sha256
    config_path.write_text(json.dumps(config) + "\n")
    monkeypatch.setattr(
        artifact_contract,
        "REPEATEDFILM_SOURCE_CHECKPOINT_SHA256",
        checkpoint_sha256,
    )

    with pytest.raises(
        ValueError,
        match="live checkpoint does not prove.*repeated-FiLM source",
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_unregistered_source_snapshot_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    source_manifest = json.loads(
        Path(
            spec["artifact_paths"]["repeatedfilm_source_manifest"]
        ).read_text(encoding="utf-8")
    )
    source_run = Path(source_manifest["source_run_path"])
    (
        source_run / "source_snapshot" / "legacy_singlefilm.py"
    ).write_text("# forbidden unregistered lineage\n")

    with pytest.raises(
        ValueError,
        match="source snapshot contains unregistered files",
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_config_that_allows_lineage_mixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    config_path = Path(spec["artifact_paths"]["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["lineage"]["mix_lineages"] = True
    config_path.write_text(json.dumps(config) + "\n")

    with pytest.raises(
        ValueError,
        match="hierarchical repeated-FiLM only",
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


@pytest.mark.parametrize(
    "payload_kind",
    ["arbitrary_json", "train_feature_statistics"],
)
def test_lock_rejects_non_deployment_scaler_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_kind: str,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    scaler_path = Path(spec["artifact_paths"]["scaler_metadata"])
    payload = (
        {"mean": [0.0], "scale": [1.0]}
        if payload_kind == "arbitrary_json"
        else {
            **identity_payload(),
            "schema_version": 1,
            "artifact_kind": "feature_statistics_train_only",
            "feature_columns": ["q_raw"],
            "train_mean": [0.0],
            "train_scale": [1.0],
        }
    )
    scaler_path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="scaler metadata"):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_incomplete_scaler_model_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    scaler_path = Path(spec["artifact_paths"]["scaler_metadata"])
    payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    payload["models"].pop("set_aware_residual")
    scaler_path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(
        ValueError, match="final deployment scaler artifact"
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_model_scaler_that_differs_from_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    method = "pairwise_ranker"
    model_path = Path(
        spec["artifact_paths"][
            SCALER_MODEL_ARTIFACT_KEYS[method]
        ]
    )
    model = torch.load(
        model_path, map_location="cpu", weights_only=True
    )
    model["artifact"]["scaler"]["mean"] = [9.0]
    torch.save(model, model_path)
    scaler_path = Path(spec["artifact_paths"]["scaler_metadata"])
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler["models"][method]["artifact_sha256"] = sha256_file(
        model_path
    )
    scaler_path.write_text(json.dumps(scaler) + "\n")
    with pytest.raises(
        ValueError, match="deployment model scaler differs"
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("config_max_tokens", "resource/generation/safe-switch"),
        ("lock_max_tokens", "registered config seeds/runtime settings"),
        ("config_geometry", "resource/generation/safe-switch"),
        ("selection_resources", "resource limits differ"),
    ],
)
def test_lock_rejects_unregistered_vlm_contract_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    if mutation in {"config_max_tokens", "config_geometry"}:
        config_path = Path(spec["artifact_paths"]["config"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if mutation == "config_max_tokens":
            config["local_vlm"]["max_output_tokens"] = 512
        else:
            config["local_vlm"]["safe_switch"][
                "geometry_risk_threshold"
            ] = 0.4
        config_path.write_text(json.dumps(config) + "\n")
    elif mutation == "lock_max_tokens":
        spec["ranking_parameters"]["vlm_max_output_tokens"] = 512
    else:
        selection_path = Path(
            spec["artifact_paths"]["vlm_validation_selection"]
        )
        selection = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
        selection["maximum_memory_mib"] = 19_000.0
        selection_path.write_text(json.dumps(selection) + "\n")
    with pytest.raises(ValueError, match=message):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_formal_method_set_with_unselected_vlm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    raw_vlm = next(
        item
        for item in spec["evaluation_definition"][
            "formal_method_protocols"
        ]
        if item["method"] == "repeatedfilm_local_vlm_visual"
    )
    raw_vlm["method"] = "repeatedfilm_local_vlm_visual_metadata"
    with pytest.raises(
        ValueError, match="selected raw VLM exactly once"
    ):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_lock_rejects_incomplete_tabular_formal_method_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    methods = spec["evaluation_definition"]["formal_method_protocols"]
    methods.pop(0)
    with pytest.raises(ValueError, match="exact repeated-FiLM"):
        build_lock_manifest(spec, repo_root=tmp_path)


@pytest.mark.parametrize("binding", ["safe_switch", "rule_selection"])
def test_lock_rejects_training_selection_cross_run_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    spec = _spec(tmp_path, source, artifact, monkeypatch)
    training_path = Path(
        spec["artifact_paths"]["reranker_training_manifest"]
    )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if binding == "safe_switch":
        training["safe_switch_selection"]["threshold"] = 0.1
    else:
        training["rule_selection_sha256"] = "0" * 64
    training_path.write_text(json.dumps(training) + "\n")
    with pytest.raises(ValueError, match="primary evidence training_manifest"):
        build_lock_manifest(spec, repo_root=tmp_path)


def test_formal_test_guard_allows_only_identical_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    artifact = tmp_path / "config.json"
    artifact.write_text("{}\n")
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock._git",
        lambda _root, *args: "abc123" if "rev-parse" in args else "",
    )
    manifest = build_lock_manifest(
        _spec(tmp_path, source, artifact, monkeypatch),
        repo_root=tmp_path,
    )
    formal_identity = {
        key: manifest[key] for key in identity_payload()
    }
    lock = tmp_path / "lock.json"
    write_lock_exclusive(lock, manifest)
    output = tmp_path / "formal"
    first = consume_formal_test_once(lock, output_root=output, invocation=["run", "x"])
    second = consume_formal_test_once(lock, output_root=output, invocation=["run", "x"])
    assert first == second
    with pytest.raises(FileExistsError, match="different locked invocation"):
        consume_formal_test_once(
            lock, output_root=output, invocation=["run", "different"]
        )
    with pytest.raises(FileExistsError, match="different locked invocation"):
        consume_formal_test_once(
            lock,
            output_root=tmp_path / "another-formal-output",
            invocation=["run", "x"],
        )
    inference_manifest = output / "inference_manifest.json"
    inference_manifest.write_text(
        json.dumps(
            {
                **formal_identity,
                "lock_content_sha256": manifest[
                    "manifest_content_sha256"
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(
        FileNotFoundError, match="completion ledger is missing"
    ):
        verify_completed_formal_stage(
            lock, stage="TEST", manifest_path=inference_manifest
        )
    completion = complete_formal_test_once(
        lock, manifest_path=inference_manifest
    )
    assert completion == complete_formal_test_once(
        lock, manifest_path=inference_manifest
    )
    verified = verify_completed_formal_stage(
        lock, stage="TEST", manifest_path=inference_manifest
    )
    assert verified["manifest"]["lock_content_sha256"] == manifest[
        "manifest_content_sha256"
    ]
    escaped_manifest = tmp_path / "inference_manifest.json"
    escaped_manifest.write_text(inference_manifest.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="outside its guarded output"):
        verify_completed_formal_stage(
            lock, stage="TEST", manifest_path=escaped_manifest
        )
    stage_output = tmp_path / "formal-vlm"
    first_stage = consume_formal_stage_once(
        lock,
        stage="VLM",
        output_root=stage_output,
        invocation=["vlm", "locked"],
    )
    assert first_stage == consume_formal_stage_once(
        lock,
        stage="VLM",
        output_root=stage_output,
        invocation=["vlm", "locked"],
    )
    with pytest.raises(FileExistsError, match="formal stage VLM"):
        consume_formal_stage_once(
            lock,
            stage="VLM",
            output_root=tmp_path / "other-vlm",
            invocation=["vlm", "locked"],
        )
    stage_manifest = stage_output / "formal_vlm_manifest.json"
    stage_manifest.write_text(json.dumps(formal_identity) + "\n")
    stage_completion = complete_formal_stage_once(
        lock, stage="VLM", manifest_path=stage_manifest
    )
    assert stage_completion == complete_formal_stage_once(
        lock, stage="VLM", manifest_path=stage_manifest
    )
    stage_manifest.write_text(
        json.dumps({**formal_identity, "tampered": True}) + "\n"
    )
    with pytest.raises(FileExistsError, match="completion changed"):
        complete_formal_stage_once(
            lock, stage="VLM", manifest_path=stage_manifest
        )


def test_formal_guards_reject_preseeded_output_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text("{}\n")
    verified = {
        **identity_payload(),
        "manifest_content_sha256": "lock-hash",
        "expected_test_sample_count": 1,
    }
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock.verify_lock",
        lambda _path: verified,
    )

    formal_output = tmp_path / "formal"
    formal_output.mkdir()
    (formal_output / "inference_manifest.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="empty dedicated output"):
        consume_formal_test_once(
            lock, output_root=formal_output, invocation=["formal"]
        )
    assert not lock.with_name(
        f"{lock.name}.FORMAL_TEST_STARTED.json"
    ).exists()

    clean_formal = tmp_path / "clean-formal"
    consume_formal_test_once(
        lock, output_root=clean_formal, invocation=["formal"]
    )
    clean_manifest = clean_formal / "inference_manifest.json"
    clean_manifest.write_text(json.dumps(identity_payload()) + "\n")
    complete_formal_test_once(lock, manifest_path=clean_manifest)
    stage_output = tmp_path / "visual"
    stage_output.mkdir()
    (stage_output / "formal_vlm_manifest.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="empty dedicated"):
        consume_formal_stage_once(
            lock,
            stage="VLM_VISUAL",
            output_root=stage_output,
            invocation=["vlm"],
        )


def test_formal_stage_allows_only_explicit_preflight_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text("{}\n")
    verified = {
        **identity_payload(),
        "manifest_content_sha256": "lock-hash",
        "expected_test_sample_count": 1,
    }
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock.verify_lock",
        lambda _path: verified,
    )
    formal = tmp_path / "formal"
    consume_formal_test_once(
        lock, output_root=formal, invocation=["formal"]
    )
    inference_manifest = formal / "inference_manifest.json"
    inference_manifest.write_text(json.dumps(identity_payload()) + "\n")
    complete_formal_test_once(lock, manifest_path=inference_manifest)

    output = tmp_path / "formal-vlm"
    attempts = output / "attempts"
    attempts.mkdir(parents=True)
    audit = attempts / "session_audit_0001.json"
    audit.write_text('{"local_only":true}\n')
    consume_formal_stage_once(
        lock,
        stage="VLM_VISUAL",
        output_root=output,
        invocation=["vlm"],
        allowed_preexisting_files=(audit,),
    )

    other_output = tmp_path / "other-formal-vlm"
    other_attempts = other_output / "attempts"
    other_attempts.mkdir(parents=True)
    allowed = other_attempts / "session_audit_0001.json"
    allowed.write_text('{"local_only":true}\n')
    (other_output / "unexpected.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="preflight files"):
        consume_formal_stage_once(
            lock,
            stage="VLM_METADATA",
            output_root=other_output,
            invocation=["vlm-metadata"],
            allowed_preexisting_files=(allowed,),
        )
