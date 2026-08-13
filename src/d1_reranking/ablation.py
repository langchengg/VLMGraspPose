"""Immutable P10 evidence-track and feature-family ablation contracts.

This module plans only Train/Validation work.  It freezes the already-selected
primary algorithm and numerical hyperparameters; changing evidence never opens
a new tuning dimension.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.hashing import atomic_json, canonical_sha256
from unified_reranking.training import FORMAL_SEEDS

from .execution import artifact_record, load_content_manifest
from .k_sensitivity import _selected_primary_contract, selected_training_spec


ABLATION_SCHEMA_VERSION = 1
ABLATION_PLAN_RELATIVE = Path("configs/d1_ablation_plan_v1.json")
ABLATION_EXECUTION_POINTER_RELATIVE = Path("configs/d1_ablation_execution.json")
ABLATION_EXECUTION_REGISTRY_RELATIVE = Path("configs/ablation_executions")
ABLATION_OUTER_FOLDS = 5
EVIDENCE_TRACKS = (
    "T1_native_available",
    "T2_matched_common",
    "T3_route_rich",
    "T4_four_route_consensus",
)
TRAINED_EVIDENCE_TRACKS = tuple(
    track for track in EVIDENCE_TRACKS if track != "T2_matched_common"
)
FEATURE_FAMILIES = (
    "q_native_calibration",
    "soft_target_support",
    "width_shape",
    "depth",
    "contacts",
    "clearance",
    "relations",
    "reliability",
    "gq_crop_latent",
    "four_route_consensus",
)
REPORTING_METHODS = ("R0", "selected_ungated", "R7")
RUNNER_MODULE = "tools.d1_reranking.run_ablation_cell"


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _track_manifest_path(root: Path, split: str, track: str) -> Path:
    return root / "03_features" / split / "top5" / track / "manifest.json"


def _load_track_manifest(
    path: Path, *, split: str, track: str
) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest = load_content_manifest(
        path, name=f"D1 P10 {split}/{track}", statuses=("COMPLETE",)
    )
    configuration = _mapping(
        manifest.get("configuration"), name=f"D1 P10 {split}/{track} configuration"
    )
    if (
        configuration.get("route") not in {None, "D1"}
        or configuration.get("split") != split
        or configuration.get("pool") != "top5"
        or configuration.get("track") != track
        or manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 P10 {split}/{track} semantics differ")
    if track == "T1_native_available":
        schema_path = verified_artifact_path(
            manifest.get("feature_schema", {}), name=f"D1 P10 {split}/{track} schema"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        columns = assert_model_feature_columns(
            tuple(map(str, schema.get("model_columns", ())))
        )
        feature_record = manifest.get("artifact", {})
        observed_schema_sha256 = schema.get("model_schema_sha256")
    else:
        columns = assert_model_feature_columns(
            tuple(map(str, manifest.get("model_feature_columns", ())))
        )
        feature_record = manifest.get("artifacts", {}).get("candidate_features", {})
        observed_schema_sha256 = manifest.get(
            "feature_schema_sha256", manifest.get("model_feature_schema_sha256")
        )
    verified_artifact_path(
        feature_record, name=f"D1 P10 {split}/{track} candidate features"
    )
    if observed_schema_sha256 not in {
        None,
        canonical_sha256(columns),
    }:
        raise RuntimeError(f"D1 P10 {split}/{track} feature schema hash differs")
    verify_artifact_records_recursive(
        manifest, name=f"D1 P10 {split}/{track} closure", require_at_least_one=True
    )
    return manifest, columns


def track_feature_artifact(
    manifest: Mapping[str, Any], *, track: str, name: str
) -> Path:
    """Resolve the canonical feature table across legacy T1 and unified tracks."""

    record = (
        manifest.get("artifact", {})
        if track == "T1_native_available"
        else manifest.get("artifacts", {}).get("candidate_features", {})
    )
    return verified_artifact_path(record, name=name)


def _family_for_column(column: str) -> str:
    name = column.lower()
    if name.startswith("t4_") or any(
        token in name
        for token in (
            "four_route",
            "route_agreement",
            "route_disagreement",
            "route_support",
            "mutual_nearest",
        )
    ):
        return "four_route_consensus"
    if any(
        token in name
        for token in (
            "latent",
            "embedding",
            "penultimate",
            "aligned_crop",
            "gqcnn_crop",
        )
    ):
        return "gq_crop_latent"
    if any(
        token in name
        for token in (
            "reliability",
            "stability",
            "missing",
            "valid_fraction",
            "_valid",
            "touches_border",
            "fragmented",
            "component_count",
            "retention_rate",
            "feature_missing_fraction",
            "perturbed_score",
            "overflow",
        )
    ):
        return "reliability"
    if any(
        token in name
        for token in (
            "relation",
            "nearest_candidate",
            "higher_score_distance",
            "nearby_candidates",
            "cluster",
            "uniqueness",
            "iou_with_other",
            "overlapping_candidates",
            "delta_x",
            "delta_y",
            "rank_difference",
            "candidate_count",
            "neighbour_q",
            "local_q_contrast",
        )
    ):
        return "relations"
    if any(
        token in name
        for token in (
            "clearance",
            "sweep_",
            "obstacle",
            "intrusion",
            "clutter",
            "corridor_conflict",
            "border_clearance",
            "background_intrusion",
            "invalid_depth_in_sweep",
            "occupancy",
            "collision_proxy",
        )
    ):
        return "clearance"
    if any(
        token in name
        for token in (
            "left_contact",
            "right_contact",
            "contacts",
            "jaw_probability",
            "normal_",
            "normaldot",
            "antipodal",
            "friction",
            "force_closure",
            "camera_xyz",
            "candidate_camera",
            "contact_symmetry",
        )
    ):
        return "contacts"
    if "depth" in name or any(
        token in name
        for token in (
            "plane_residual",
            "local_mad",
            "local_range",
            "estimated_local_object_thickness",
            "z_center",
        )
    ):
        return "depth"
    if any(
        token in name
        for token in (
            "soft_target",
            "target_support",
            "p_center",
            "probability",
            "mask_support",
            "mask_geometry",
            "probability_support",
            "center_support",
            "rect_support",
            "inside_mask",
            "binary_coverage",
            "mask_boundary",
            "mask_span",
            "background_fraction_inside_rectangle",
            "local_probability_entropy",
            "mask_entropy",
            "p_axis",
            "p_contact",
        )
    ):
        return "soft_target_support"
    if any(
        token in name
        for token in (
            "width",
            "height",
            "shape",
            "theta",
            "angle",
            "rectangle",
            "bbox",
            "jaw",
            "cx_px",
            "cy_px",
            "center_x",
            "center_y",
        )
    ):
        return "width_shape"
    if (
        name
        in {
            "native_rank",
            "original_gqcnn_rank",
            "native_score",
            "native_score_raw",
            "calibrated_native_probability",
            "base_logit",
            "gqcnn_q",
            "q_raw",
        }
        or name.startswith("gqcnn_q_")
        or name.startswith("native_q_")
        or name.startswith("calibrated_q_")
        or any(
            token in name
            for token in (
                "q_gap",
                "q_log",
                "q_percentile",
                "q_rank",
                "score_percentile",
                "score_zscore",
                "pool_score",
                "rank_percentile",
                "delta_to_",
                "top1_top2_margin",
                "is_native_top1",
            )
        )
    ):
        return "q_native_calibration"
    raise RuntimeError(f"D1 P10 feature family is not predeclared for column: {column}")


def feature_family_registry(columns: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return the deterministic v1 family membership, retaining empty families."""

    validated = assert_model_feature_columns(tuple(map(str, columns)))
    grouped: dict[str, list[str]] = {family: [] for family in FEATURE_FAMILIES}
    for column in validated:
        grouped[_family_for_column(column)].append(column)
    return {
        family: {
            "family_index": index,
            "columns": grouped[family],
            "status": "AVAILABLE" if grouped[family] else "NOT_AVAILABLE",
            "reason": None
            if grouped[family]
            else "no hash-bound T4 model columns belong to this predeclared family",
        }
        for index, family in enumerate(FEATURE_FAMILIES)
    }


def feature_ablation_variants(
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build cumulative and leave-one-family-out variants without fabrication."""

    if tuple(registry) != FEATURE_FAMILIES:
        raise RuntimeError("D1 P10 feature-family registry order differs")
    all_columns = [
        str(column)
        for family in FEATURE_FAMILIES
        for column in registry[family].get("columns", ())
    ]
    variants: list[dict[str, Any]] = []
    cumulative: list[str] = []
    for index, family in enumerate(FEATURE_FAMILIES):
        columns = list(map(str, registry[family].get("columns", ())))
        cumulative.extend(columns)
        available = bool(columns) and bool(cumulative)
        variants.append(
            {
                "variant_id": f"cumulative_{index:02d}_{family}",
                "kind": "cumulative",
                "target_family": family,
                "included_families": list(FEATURE_FAMILIES[: index + 1]),
                "excluded_families": list(FEATURE_FAMILIES[index + 1 :]),
                "feature_columns": list(cumulative) if available else [],
                "status": "AVAILABLE" if available else "NOT_AVAILABLE",
                "reason": None
                if available
                else f"family {family} has no available columns",
            }
        )
    for family in FEATURE_FAMILIES:
        removed = list(map(str, registry[family].get("columns", ())))
        available = bool(removed) and len(all_columns) > len(removed)
        variants.append(
            {
                "variant_id": f"leave_one_out_{family}",
                "kind": "leave_one_family_out",
                "target_family": family,
                "included_families": [
                    candidate for candidate in FEATURE_FAMILIES if candidate != family
                ],
                "excluded_families": [family],
                "feature_columns": [
                    column for column in all_columns if column not in set(removed)
                ]
                if available
                else [],
                "status": "AVAILABLE" if available else "NOT_AVAILABLE",
                "reason": None
                if available
                else (
                    f"family {family} is unavailable"
                    if not removed
                    else "removing the only available family would create an empty model"
                ),
            }
        )
    if len(variants) != 2 * len(FEATURE_FAMILIES):
        raise AssertionError("D1 P10 feature-ablation variant count differs")
    return variants


def _job(
    *,
    analysis: str,
    variant_id: str,
    track: str,
    columns: Sequence[str],
    selected_primary: Mapping[str, Any],
    seed: int,
    mode_fold: str | int,
) -> dict[str, Any]:
    mode = "validation" if mode_fold == "validation" else "oof"
    held_fold = None if mode == "validation" else int(mode_fold)
    configuration = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "route": "D1",
        "analysis": analysis,
        "variant_id": variant_id,
        "pool": "top5",
        "track": track,
        "method": selected_primary["method"],
        "selected_primary_trial_id": selected_primary["trial_id"],
        "selected_primary_configuration_sha256": selected_primary[
            "configuration_sha256"
        ],
        "training_spec": selected_training_spec(
            _mapping(
                selected_primary["configuration"],
                name="D1 P10 selected primary configuration",
            ),
            method=str(selected_primary["method"]),
        ),
        "feature_columns": list(map(str, columns)),
        "feature_schema_sha256": canonical_sha256(list(map(str, columns))),
        "seed": int(seed),
        "mode": mode,
        "held_fold": held_fold,
        "max_candidates": 5,
        "fold_local_preprocessing": True,
        "fold_local_calibration": True,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    job_id = canonical_sha256(configuration)[:16]
    base = (
        "07_validation/evidence_ablation/cells"
        if analysis == "evidence_track"
        else "12_feature_ablation/cells"
    )
    return {
        "job_id": job_id,
        "worker_argv": ["-m", RUNNER_MODULE, "--job-id", job_id, "--resume"],
        "output_manifest": f"{base}/{job_id}/manifest.json",
        "configuration": configuration,
    }


def ablation_jobs(
    *,
    selected_primary: Mapping[str, Any],
    track_columns: Mapping[str, Sequence[str]],
    variants: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for track in TRAINED_EVIDENCE_TRACKS:
        columns = tuple(map(str, track_columns[track]))
        if not columns:
            continue
        for seed in FORMAL_SEEDS:
            for mode_fold in ("validation", 0, 1, 2, 3, 4):
                jobs.append(
                    _job(
                        analysis="evidence_track",
                        variant_id=track,
                        track=track,
                        columns=columns,
                        selected_primary=selected_primary,
                        seed=int(seed),
                        mode_fold=mode_fold,
                    )
                )
    for variant in variants:
        if variant.get("status") != "AVAILABLE":
            continue
        for seed in FORMAL_SEEDS:
            for mode_fold in ("validation", 0, 1, 2, 3, 4):
                jobs.append(
                    _job(
                        analysis="feature_family",
                        variant_id=str(variant["variant_id"]),
                        track="T4_four_route_consensus",
                        columns=tuple(map(str, variant["feature_columns"])),
                        selected_primary=selected_primary,
                        seed=int(seed),
                        mode_fold=mode_fold,
                    )
                )
    identifiers = [str(job["job_id"]) for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("D1 P10 ablation job identifiers collide")
    return jobs


def build_ablation_plan(
    run_dir: str | Path, *, tool_paths: Sequence[Path]
) -> dict[str, Any]:
    """Build the versioned, Train/Validation-only P10 plan."""

    root = Path(run_dir).expanduser().resolve()
    selected_primary, selected_records = _selected_primary_contract(root)
    track_records: dict[str, dict[str, dict[str, str]]] = {}
    track_columns: dict[str, tuple[str, ...]] = {}
    for track in EVIDENCE_TRACKS:
        track_records[track] = {}
        split_columns: dict[str, tuple[str, ...]] = {}
        for split in ("train", "validation"):
            path = _track_manifest_path(root, split, track)
            _manifest, columns = _load_track_manifest(path, split=split, track=track)
            track_records[track][split] = artifact_record(path)
            split_columns[split] = columns
        if split_columns["train"] != split_columns["validation"]:
            raise RuntimeError(f"D1 P10 {track} Train/Validation schemas differ")
        track_columns[track] = split_columns["train"]
    registry = feature_family_registry(track_columns["T4_four_route_consensus"])
    variants = feature_ablation_variants(registry)
    jobs = ablation_jobs(
        selected_primary=selected_primary,
        track_columns=track_columns,
        variants=variants,
    )
    source_paths = tuple(
        sorted({Path(path).expanduser().resolve() for path in tool_paths}, key=str)
    )
    if not source_paths:
        raise ValueError("D1 P10 ablation plan requires code paths")
    sources: dict[str, Any] = {
        **selected_records,
        "primary_gate": artifact_record(
            root / "07_validation/gate/d1/gate_selection.json"
        ),
        "gate_grid": artifact_record(root / "configs/d1_gate_grid.json"),
        "r0_r1_selection": artifact_record(root / "07_validation/r0_r1_selection.json"),
        "track_manifests": track_records,
        "candidate_manifests": {
            split: artifact_record(root / f"02_candidates/{split}/manifest.json")
            for split in ("train", "validation")
        },
        "development_label_manifests": {
            split: artifact_record(
                root / f"03_features/{split}/top5/labels/manifest.json"
            )
            for split in ("train", "validation")
        },
        "calibration_manifest": artifact_record(
            root / "05_calibration/top5/calibration_manifest.json"
        ),
        "fold_assignments": artifact_record(
            root / "04_splits/fold_assignments.parquet"
        ),
        "denominators": {
            split: artifact_record(
                root
                / "01_manifests"
                / (
                    "d1_paired_train.parquet"
                    if split == "train"
                    else "d1_paired_validation.parquet"
                )
            )
            for split in ("train", "validation")
        },
        "code": [artifact_record(path) for path in source_paths],
    }
    plan: dict[str, Any] = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "status": "PLANNED",
        "route": "D1",
        "pool": "top5",
        "analysis": "evidence_track_and_feature_family_ablation",
        "development_splits": ["train", "validation"],
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "selected_primary": selected_primary,
        "retuning_permitted": False,
        "reporting_methods": list(REPORTING_METHODS),
        "formal_seeds": list(FORMAL_SEEDS),
        "outer_folds": ABLATION_OUTER_FOLDS,
        "evidence_tracks": {
            track: {
                "status": "REUSED_PRIMARY_EXACT"
                if track == "T2_matched_common"
                else ("AVAILABLE" if track_columns[track] else "NOT_AVAILABLE"),
                "feature_columns": list(track_columns[track]),
                "feature_schema_sha256": canonical_sha256(track_columns[track]),
                "cell_count": 0
                if track == "T2_matched_common" or not track_columns[track]
                else 18,
            }
            for track in EVIDENCE_TRACKS
        },
        "feature_family_registry_version": "d1_p10_feature_families_v1",
        "feature_family_registry": registry,
        "feature_ablation_variants": variants,
        "gate_contract": {
            "per_track_refit": True,
            "operating_point_count": 108,
            "fit_source": "track-specific Train OOF ranker outputs",
            "selection_source": "track-specific Validation ranker outputs",
            "t2_reuse": "exact existing primary R7 gate",
            "test_metrics_used": False,
        },
        "execution_contract": {
            "runner_module": RUNNER_MODULE,
            "authorization_tool": "tools.d1_reranking.authorize_ablation_execution",
            "orchestrator_tool": "tools.d1_reranking.run_ablation_matrix",
            "direct_cli_execution_permitted": False,
            "execution_manifest": str(ABLATION_EXECUTION_POINTER_RELATIVE),
            "execution_state_transitions": ["ACTIVE", "COMPLETE", "FAILED"],
            "max_parallel": 1,
            "device": "cpu",
            "fresh_resource_gate_required_on_start_and_resume": True,
            "global_heavy_resource_lease_required": True,
        },
        "jobs": jobs,
        "job_count": len(jobs),
        "job_universe_sha256": canonical_sha256([job["configuration"] for job in jobs]),
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
    }
    plan["content_sha256"] = canonical_sha256(plan)
    return validate_ablation_plan(plan)


def validate_ablation_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    plan = {str(key): child for key, child in value.items()}
    unsigned = dict(plan)
    observed_content = unsigned.pop("content_sha256", None)
    if observed_content != canonical_sha256(unsigned):
        raise RuntimeError("D1 P10 ablation plan content hash mismatch")
    if (
        plan.get("schema_version") != ABLATION_SCHEMA_VERSION
        or plan.get("status") != "PLANNED"
        or plan.get("route") != "D1"
        or plan.get("pool") != "top5"
        or plan.get("development_splits") != ["train", "validation"]
        or plan.get("candidate_test_labels_read") is not False
        or plan.get("test_inputs_referenced") is not False
        or plan.get("retuning_permitted") is not False
        or tuple(plan.get("formal_seeds", ())) != tuple(FORMAL_SEEDS)
        or plan.get("outer_folds") != ABLATION_OUTER_FOLDS
        or plan.get("reporting_methods") != list(REPORTING_METHODS)
    ):
        raise RuntimeError("D1 P10 ablation plan header differs")
    selected = _mapping(plan.get("selected_primary"), name="D1 P10 selected primary")
    if selected.get("configuration_sha256") != canonical_sha256(
        selected.get("configuration")
    ):
        raise RuntimeError("D1 P10 selected primary binding differs")
    tracks = _mapping(plan.get("evidence_tracks"), name="D1 P10 evidence tracks")
    if tuple(tracks) != EVIDENCE_TRACKS:
        raise RuntimeError("D1 P10 evidence-track universe differs")
    track_columns = {
        track: tuple(map(str, _mapping(tracks[track], name=track)["feature_columns"]))
        for track in EVIDENCE_TRACKS
    }
    if tracks["T2_matched_common"].get("status") != "REUSED_PRIMARY_EXACT":
        raise RuntimeError("D1 P10 T2 does not exact-reuse primary")
    expected_registry = feature_family_registry(
        track_columns["T4_four_route_consensus"]
    )
    if (
        plan.get("feature_family_registry_version") != "d1_p10_feature_families_v1"
        or plan.get("feature_family_registry") != expected_registry
    ):
        raise RuntimeError("D1 P10 feature-family registry differs")
    expected_variants = feature_ablation_variants(expected_registry)
    if plan.get("feature_ablation_variants") != expected_variants:
        raise RuntimeError("D1 P10 feature-ablation variants differ")
    expected_jobs = ablation_jobs(
        selected_primary=selected,
        track_columns=track_columns,
        variants=expected_variants,
    )
    if (
        plan.get("jobs") != expected_jobs
        or plan.get("job_count") != len(expected_jobs)
        or plan.get("job_universe_sha256")
        != canonical_sha256([job["configuration"] for job in expected_jobs])
        or plan.get("job_ids_sha256")
        != canonical_sha256([job["job_id"] for job in expected_jobs])
    ):
        raise RuntimeError("D1 P10 exact job universe differs")
    execution = _mapping(
        plan.get("execution_contract"), name="D1 P10 execution contract"
    )
    if (
        execution.get("runner_module") != RUNNER_MODULE
        or execution.get("direct_cli_execution_permitted") is not False
        or execution.get("execution_manifest")
        != str(ABLATION_EXECUTION_POINTER_RELATIVE)
        or execution.get("execution_state_transitions")
        != ["ACTIVE", "COMPLETE", "FAILED"]
        or execution.get("max_parallel") != 1
        or execution.get("device") != "cpu"
        or execution.get("fresh_resource_gate_required_on_start_and_resume") is not True
        or execution.get("global_heavy_resource_lease_required") is not True
    ):
        raise RuntimeError("D1 P10 execution boundary differs")
    sources = _mapping(plan.get("sources"), name="D1 P10 sources")
    if plan.get("source_signature_sha256") != canonical_sha256(sources):
        raise RuntimeError("D1 P10 source signature differs")
    return plan


def load_ablation_plan(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    plan = load_content_manifest(
        source, name="D1 P10 ablation plan", statuses=("PLANNED",)
    )
    validated = validate_ablation_plan(plan)
    verify_artifact_records_recursive(
        validated["sources"], name="D1 P10 plan sources", require_at_least_one=True
    )
    # Re-open every feature manifest and re-derive its exact schema.
    records = _mapping(validated["sources"]["track_manifests"], name="track manifests")
    tracks = _mapping(validated["evidence_tracks"], name="tracks")
    for track in EVIDENCE_TRACKS:
        split_records = _mapping(records[track], name=f"{track} records")
        schemas = []
        for split in ("train", "validation"):
            manifest_path = verified_artifact_path(
                _mapping(split_records[split], name=f"{track}/{split}"),
                name=f"D1 P10 {track}/{split}",
            )
            _manifest, columns = _load_track_manifest(
                manifest_path, split=split, track=track
            )
            schemas.append(columns)
        if (
            schemas[0] != schemas[1]
            or list(schemas[0]) != tracks[track]["feature_columns"]
        ):
            raise RuntimeError(f"D1 P10 {track} current schema differs from plan")
    return validated


def write_ablation_plan(
    path: str | Path,
    *,
    run_dir: str | Path,
    tool_paths: Sequence[Path],
    resume: bool,
) -> dict[str, Any]:
    destination = Path(path).expanduser().resolve()
    proposed = build_ablation_plan(run_dir, tool_paths=tool_paths)
    if destination.exists():
        existing = load_ablation_plan(destination)
        if resume and existing == proposed:
            return existing
        raise RuntimeError("D1 P10 ablation plan exists and differs")
    atomic_json(destination, proposed)
    return proposed


__all__ = [
    "ABLATION_EXECUTION_POINTER_RELATIVE",
    "ABLATION_EXECUTION_REGISTRY_RELATIVE",
    "ABLATION_OUTER_FOLDS",
    "ABLATION_PLAN_RELATIVE",
    "EVIDENCE_TRACKS",
    "FEATURE_FAMILIES",
    "REPORTING_METHODS",
    "RUNNER_MODULE",
    "track_feature_artifact",
    "TRAINED_EVIDENCE_TRACKS",
    "ablation_jobs",
    "build_ablation_plan",
    "feature_ablation_variants",
    "feature_family_registry",
    "load_ablation_plan",
    "validate_ablation_plan",
    "write_ablation_plan",
]
