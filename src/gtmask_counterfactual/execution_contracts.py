"""Typed immutable route contracts for the counterfactual protocol lock."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from .io import artifact_record, canonical_sha256, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_G1_C1_SOURCE = (
    REPOSITORY_ROOT
    / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500"
)
FROZEN_D1_SOURCE = (
    REPOSITORY_ROOT
    / "HiFi_reproduction/runs/modular_hierfilm_standard_dexnet_gqcnn_20260728_094528"
)
NATIVE_INFERENCE = (
    REPOSITORY_ROOT / "experiments/fair_crog_hifics_g1_c1_no_rerank/native_inference.py"
)
FROZEN_D1_CANDIDATE_SCRIPT = (
    FROZEN_D1_SOURCE / "source_snapshot/scripts/run_hifics_dexnet_candidates.py"
)
FROZEN_D1_SCORER_SCRIPT = (
    FROZEN_D1_SOURCE / "source_snapshot/scripts/run_full_gqcnn_scoring.py"
)
FROZEN_D1_CONFIG = (
    FROZEN_D1_SOURCE
    / "source_snapshot/configs/dexnet_candidates_formal_no_refinement.yaml"
)
FROZEN_NATIVE_INFERENCE_SHA256 = (
    "5bbae71830db8264494b20651ff6c62fd87210011614b159c48f11bfd4586544"
)
FROZEN_D1_CANDIDATE_SCRIPT_SHA256 = (
    "86c84e1ff1b7674dc4f089c21f6bd6fb67cc7925f3be16b14ce5638eb6da7623"
)
FROZEN_D1_SCORER_SCRIPT_SHA256 = (
    "f8b875cf91a97eb0884f836a5b0303da31deffb01606d4bd75b794b8c36fc02b"
)
FROZEN_D1_CONFIG_SHA256 = (
    "96b0f85bd053cb59c28566ce55b5b4b1d12c9db89240370d35d6c24d083e40fb"
)
CANONICAL_EVALUATOR = (
    REPOSITORY_ROOT
    / "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py"
)
CANONICAL_EVALUATOR_SHA256 = (
    "f5155590b0b8d9f0748ad463688edfe6ca595d8ab7239ef5e29a5c595aeef301"
)


def _selected_route_contract(
    route: str, *, adapter_manifest: Path | None = None
) -> dict[str, Any]:
    selected_path = FROZEN_G1_C1_SOURCE / f"selected_configs/{route.upper()}.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(selected["finetuned_checkpoint"])).expanduser().resolve()
    checkpoint_sha = str(selected["finetuned_checkpoint_sha256"])
    result: dict[str, Any] = {
        "allowed_gt_branches": ["gt_oracle", "gt_shape_only"],
        "source_run": str(FROZEN_G1_C1_SOURCE.resolve()),
        "native_inference": artifact_record(NATIVE_INFERENCE),
        "test_samples": artifact_record(
            FROZEN_G1_C1_SOURCE / "manifests/test_samples.parquet"
        ),
        "oracle_label_manifest": artifact_record(
            FROZEN_G1_C1_SOURCE / "manifests/test_labels.parquet"
        ),
        "selected_config": artifact_record(selected_path),
        "checkpoint": artifact_record(checkpoint),
        "native_manifest_fields": {
            "status": "COMPLETE",
            "variant": f"{route}_gtmask_oracle",
            "sample_count": 7_675,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "selected_config": str(selected_path.resolve()),
            "selected_config_sha256": sha256_file(selected_path),
            "raw_maps_saved": False,
            "source_samples_sha256": sha256_file(
                FROZEN_G1_C1_SOURCE / "manifests/test_samples.parquet"
            ),
            "source_labels_sha256": sha256_file(
                FROZEN_G1_C1_SOURCE / "manifests/test_labels.parquet"
            ),
        },
        "candidate_budget": 100,
        "quality_threshold": 0.2,
        "minimum_peak_distance_px": 20,
        "post_peak_nms": False,
        "mask_intervention_source": "prepared_352_binary_pil_nearest_to_native",
        "prepared_to_original_roundtrip_required": True,
    }
    if adapter_manifest is not None:
        from .g1_c1_adapter import verify_g1_c1_source_adapter

        adapter = verify_g1_c1_source_adapter(adapter_manifest)
        execution_config = adapter["selected_configs"][route]
        result.update(
            {
                "execution_source_adapter": artifact_record(adapter_manifest),
                "execution_source_root": str(adapter_manifest.parent.resolve()),
                "execution_test_samples": dict(adapter["test_samples"]),
                "execution_label_projection": dict(
                    adapter["test_labels_projection"]
                ),
                "execution_selected_config": dict(execution_config),
                "evaluable_sample_count": int(adapter["evaluable_sample_count"]),
                "technical_complement_count": int(
                    adapter["unresolved_sample_count"]
                ),
                "ordered_evaluable_ids_sha256": adapter[
                    "ordered_evaluable_ids_sha256"
                ],
                "sorted_unresolved_ids_sha256": adapter[
                    "sorted_unresolved_ids_sha256"
                ],
                "native_manifest_fields": {
                    **result["native_manifest_fields"],
                    "sample_count": int(adapter["evaluable_sample_count"]),
                    "selected_config": str(
                        Path(str(execution_config["path"])).resolve()
                    ),
                    "selected_config_sha256": execution_config["sha256"],
                    "source_samples_sha256": adapter["test_samples"]["sha256"],
                    "source_labels_sha256": adapter["test_labels_projection"][
                        "sha256"
                    ],
                },
            }
        )
    return result


def canonical_route_contracts(
    adapter_manifest: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the only production route contract from current frozen bytes."""

    return {
        "g1": _selected_route_contract("g1", adapter_manifest=adapter_manifest),
        "c1": _selected_route_contract("c1", adapter_manifest=adapter_manifest),
        "d1": {
            "allowed_gt_branches": ["gt_oracle", "gt_shape_only"],
            "case": "B",
            "mask_affects_raw_sampling": True,
            "raw_candidate_regeneration_required": True,
            "filter_only_primary_allowed": False,
            "candidate_runner": artifact_record(FROZEN_D1_CANDIDATE_SCRIPT),
            "scorer": artifact_record(FROZEN_D1_SCORER_SCRIPT),
            "execution_config": artifact_record(FROZEN_D1_CONFIG),
            "raw_candidate_budget": 256,
            "nms_candidate_budget": 30,
            "native_pools": [5, 10, "allnms"],
        },
    }


def canonical_semantic_contracts() -> dict[str, dict[str, Any]]:
    """Return preregistered semantics that may not be caller-selected."""

    from .taxonomy import taxonomy_definitions

    return {
        "resize_rules": {
            "primary_branch": "gt_oracle",
            "allowed_changed_variable": "target_mask_or_probability_support_only",
            "non_mask_configuration_must_remain_frozen": True,
            "candidate_budget_parameters_must_equal_route_contract": True,
            "checkpoint_and_decoder_must_equal_route_contract": True,
            "training_allowed": False,
            "ranker_retraining_allowed": False,
            "gate_or_selector_retuning_allowed": False,
            "gt_feedback_to_training_or_selection_allowed": False,
            "binary_support": "binary_gt_target_mask",
            "g1_c1_native_input": "prepared_352_binary_pil_nearest_to_native",
            "d1_native_input": "original_resolution_binary_gt_segmask",
            "soft_support": "deterministic_binary_float32",
            "probability_range": [0.0, 1.0],
            "test_tuned_blur": False,
            "test_tuned_dilation": False,
            "secondary_gt_shape_only": "predicted_probability_times_resized_gt_support",
        },
        "evaluator_contract": {
            "implementation_sha256": CANONICAL_EVALUATOR_SHA256,
            "image_shape": [480, 640],
            "iou_comparator": "strict_greater_than",
            "iou_threshold": 0.25,
            "periodic_angle_degrees": 180.0,
            "angle_comparator": "less_than_or_equal",
            "angle_threshold_degrees": 30.0,
            "same_gt_conjunction": True,
            "x_axis": "column",
            "y_axis": "row",
            "no_output_in_denominator": True,
            "gt_height_px": 20.0,
            "gt_width_clip_px": 100.0,
        },
        "taxonomy": dict(taxonomy_definitions()),
        "statistics": {
            "inference_label": "descriptive counterfactual inference",
            "paired_test": "exact_mcnemar",
            "scene_cluster_bootstrap_iterations": 10_000,
            "frame_cluster_bootstrap_sensitivity_iterations": 10_000,
            "seed": 20260813,
            "multiple_comparison_correction": "holm",
        },
        "case_selection": {
            "eligible_table_first": True,
            "mechanism_purity_filter": True,
            "presentation_quality_eligibility": True,
            "selection": "cluster_medoid_then_farthest_first",
            "tie_break": "sha256(sample_id+route+category)",
            "quota_per_route_category": 2,
            "manual_qa_exact_selected_coverage": True,
            "predicate_version": "gtmask_case_predicates_v1",
            "technical_rows_ineligible": True,
            "category_predicates": {
                "clear_grounding_limited": "native_taxonomy == T4_grounding_limited",
                "grounding_plus_selection": "native_taxonomy in {T5_grounding_plus_selection_within_top5,T6_grounding_plus_deep_ranking}",
                "generator_limited_under_gt": "native_taxonomy == T7_grasper_or_candidate_generation_limited",
                "predicted_mask_ranking_limited": "pred_all_positive and (not pred_native_correct or not final_correct)",
                "gt_mask_regression": "GT_mask_regression or gt_first_positive_rank > pred_first_positive_rank",
                "no_output_recovered_by_gt_mask": "pred_no_output and not gt_no_output",
                "no_change_success": "pred_native_correct and gt_native_correct and cross_branch_native_geometry_match",
                "borderline_annotation_sensitive": "annotation_suspect",
            },
            "feature_vector_schema": [
                "pred_candidate_count",
                "gt_candidate_count",
                "pred_positive_candidate_count",
                "gt_positive_candidate_count",
                "pred_first_positive_rank_missing_minus_one",
                "gt_first_positive_rank_missing_minus_one",
                "predicted_mask_iou",
                "target_area_fraction",
                "mask_component_count",
                "mask_boundary_complexity",
                "valid_depth_ratio",
            ],
            "feature_vector_missing_policy": "first-rank missing=-1; all other fields must be finite",
            "presentation_predicates": [
                "rgb/depth/gt-mask/pred-mask records present and hash-valid",
                "predicted probability is hash-valid or explicitly NOT_AVAILABLE",
                "all image coordinate frames equal rgb_native",
                "non-empty language prompt",
            ],
        },
    }


def canonical_source_code_inventory() -> dict[str, dict[str, Any]]:
    """Hash every scientific module and CLI that can affect the new run."""

    paths = sorted((REPOSITORY_ROOT / "src/gtmask_counterfactual").glob("*.py"))
    paths += sorted((REPOSITORY_ROOT / "tools/gtmask_counterfactual").glob("*.py"))
    return {
        str(path.relative_to(REPOSITORY_ROOT)): artifact_record(path) for path in paths
    }


def canonical_config_inventory() -> dict[str, dict[str, Any]]:
    paths = (
        FROZEN_G1_C1_SOURCE / "selected_configs/G1.json",
        FROZEN_G1_C1_SOURCE / "selected_configs/C1.json",
        FROZEN_D1_CONFIG,
    )
    return {
        str(path.relative_to(REPOSITORY_ROOT)): artifact_record(path) for path in paths
    }


def _inline_payload(value: Any, *, name: str) -> Any:
    if not isinstance(value, Mapping) or value.get("kind") != "inline":
        raise ValueError(f"production protocol requires inline binding: {name}")
    payload = value.get("value")
    if value.get("sha256") != canonical_sha256(payload):
        raise ValueError(f"production protocol inline hash differs: {name}")
    return payload


def validate_scientific_bindings(bindings: Mapping[str, Any]) -> None:
    """Replay the fixed protocol semantics and current code/config inventory."""

    expected_semantics = canonical_semantic_contracts()
    for name in ("resize_rules", "taxonomy", "statistics", "case_selection"):
        if _inline_payload(bindings.get(name), name=name) != expected_semantics[name]:
            raise ValueError(f"production protocol semantic binding differs: {name}")
    evaluator = bindings.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("production evaluator binding is malformed")
    _artifact(
        evaluator.get("implementation"),
        name="evaluator.implementation",
        expected=CANONICAL_EVALUATOR,
        expected_sha=CANONICAL_EVALUATOR_SHA256,
    )
    if (
        _inline_payload(evaluator.get("contract"), name="evaluator.contract")
        != expected_semantics["evaluator_contract"]
    ):
        raise ValueError("production evaluator semantic contract differs")
    if bindings.get("source_code") != canonical_source_code_inventory():
        raise ValueError("production scientific source-code inventory differs")
    if bindings.get("configs") != canonical_config_inventory():
        raise ValueError("production frozen configuration inventory differs")


def _artifact(value: Any, *, name: str, expected: Path, expected_sha: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"route contract lacks artifact: {name}")
    path = Path(str(value.get("path", ""))).expanduser().resolve()
    if path != expected.resolve() or value.get("sha256") != expected_sha:
        raise ValueError(f"route contract artifact differs: {name}")
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"route contract current bytes differ: {name}")


def validate_route_contracts(routes: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject caller-authored route contracts that differ from frozen sources."""

    adapter_record = routes.get("g1", {}).get("execution_source_adapter")
    if not isinstance(adapter_record, Mapping) or (
        routes.get("c1", {}).get("execution_source_adapter") != adapter_record
    ):
        raise ValueError("G1/C1 routes lack one shared execution source adapter")
    adapter_path = Path(str(adapter_record.get("path", ""))).expanduser().resolve()
    from .g1_c1_adapter import verify_g1_c1_source_adapter

    adapter = verify_g1_c1_source_adapter(adapter_path)
    if dict(adapter_record) != artifact_record(adapter_path):
        raise ValueError("G1/C1 execution source adapter record differs")
    for route in ("g1", "c1"):
        value = routes[route]
        if (
            Path(str(value.get("source_run", ""))).expanduser().resolve()
            != FROZEN_G1_C1_SOURCE.resolve()
        ):
            raise ValueError(f"{route} source run differs from the frozen source")
        _artifact(
            value.get("native_inference"),
            name=f"{route}.native_inference",
            expected=NATIVE_INFERENCE,
            expected_sha=FROZEN_NATIVE_INFERENCE_SHA256,
        )
        for label, relative in (
            ("test_samples", Path("manifests/test_samples.parquet")),
            ("oracle_label_manifest", Path("manifests/test_labels.parquet")),
            ("selected_config", Path(f"selected_configs/{route.upper()}.json")),
        ):
            expected = FROZEN_G1_C1_SOURCE / relative
            _artifact(
                value.get(label),
                name=f"{route}.{label}",
                expected=expected,
                expected_sha=sha256_file(expected),
            )
        selected_path = FROZEN_G1_C1_SOURCE / f"selected_configs/{route.upper()}.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        checkpoint = (
            Path(str(selected.get("finetuned_checkpoint", ""))).expanduser().resolve()
        )
        checkpoint_sha = str(selected.get("finetuned_checkpoint_sha256", ""))
        try:
            checkpoint.relative_to(FROZEN_G1_C1_SOURCE.resolve())
        except ValueError as error:
            raise ValueError(f"{route} checkpoint escapes the frozen source") from error
        _artifact(
            value.get("checkpoint"),
            name=f"{route}.checkpoint",
            expected=checkpoint,
            expected_sha=checkpoint_sha,
        )
        execution_config = adapter["selected_configs"][route]
        expected_manifest_fields = {
            "status": "COMPLETE",
            "variant": f"{route}_gtmask_oracle",
            "sample_count": int(adapter["evaluable_sample_count"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "selected_config": str(Path(str(execution_config["path"])).resolve()),
            "selected_config_sha256": execution_config["sha256"],
            "raw_maps_saved": False,
            "source_samples_sha256": adapter["test_samples"]["sha256"],
            "source_labels_sha256": adapter["test_labels_projection"]["sha256"],
        }
        if value.get("native_manifest_fields") != expected_manifest_fields:
            raise ValueError(f"{route} native manifest contract differs")
        expected_dynamic = {
            "execution_source_adapter": artifact_record(adapter_path),
            "execution_source_root": str(adapter_path.parent.resolve()),
            "execution_test_samples": dict(adapter["test_samples"]),
            "execution_label_projection": dict(adapter["test_labels_projection"]),
            "execution_selected_config": dict(execution_config),
            "evaluable_sample_count": int(adapter["evaluable_sample_count"]),
            "technical_complement_count": int(adapter["unresolved_sample_count"]),
            "ordered_evaluable_ids_sha256": adapter["ordered_evaluable_ids_sha256"],
            "sorted_unresolved_ids_sha256": adapter["sorted_unresolved_ids_sha256"],
        }
        if any(value.get(key) != expected for key, expected in expected_dynamic.items()):
            raise ValueError(f"{route} execution partition contract differs")
        if (
            value.get("candidate_budget") != 100
            or value.get("quality_threshold") != 0.2
            or value.get("minimum_peak_distance_px") != 20
            or value.get("post_peak_nms") is not False
            or value.get("mask_intervention_source")
            != "prepared_352_binary_pil_nearest_to_native"
            or value.get("prepared_to_original_roundtrip_required") is not True
        ):
            raise ValueError(f"{route} frozen decoder/intervention contract differs")
    d1 = routes["d1"]
    for label, expected, digest in (
        (
            "candidate_runner",
            FROZEN_D1_CANDIDATE_SCRIPT,
            FROZEN_D1_CANDIDATE_SCRIPT_SHA256,
        ),
        ("scorer", FROZEN_D1_SCORER_SCRIPT, FROZEN_D1_SCORER_SCRIPT_SHA256),
        ("execution_config", FROZEN_D1_CONFIG, FROZEN_D1_CONFIG_SHA256),
    ):
        _artifact(
            d1.get(label), name=f"d1.{label}", expected=expected, expected_sha=digest
        )
    if (
        d1.get("raw_candidate_budget") != 256
        or d1.get("nms_candidate_budget") != 30
        or d1.get("native_pools") != [5, 10, "allnms"]
    ):
        raise ValueError("D1 frozen budget/pool contract differs")

    if dict(routes) != canonical_route_contracts(adapter_path):
        raise ValueError("route contracts differ from the canonical frozen contract")


__all__ = [
    "CANONICAL_EVALUATOR",
    "CANONICAL_EVALUATOR_SHA256",
    "canonical_config_inventory",
    "canonical_route_contracts",
    "canonical_semantic_contracts",
    "canonical_source_code_inventory",
    "validate_route_contracts",
    "validate_scientific_bindings",
]
