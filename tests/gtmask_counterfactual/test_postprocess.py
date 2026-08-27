from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import numpy as np

import gtmask_counterfactual.postprocess as postprocess_module
from gtmask_counterfactual.acceptance import accept_gallery
from gtmask_counterfactual.audit import bootstrap_run, transition_pipeline_status
from gtmask_counterfactual.contracts import RunState
from gtmask_counterfactual.independent import canonical_corners
from gtmask_counterfactual.io import (
    artifact_record,
    atomic_json,
    atomic_parquet,
    canonical_sha256,
)
from gtmask_counterfactual.gt_grasp_authority import materialize_gt_grasp_authority
from gtmask_counterfactual.postprocess import (
    PostprocessContractError,
    run_postprocess,
    write_final_outcomes_authority,
    write_postprocess_input_manifest,
    write_route_frame_manifest,
)
from gtmask_counterfactual.galleries import write_gallery_manifest
from gtmask_counterfactual.protocol import (
    claim_bulk_execution,
    create_protocol_lock,
    inline_binding,
)
from gtmask_counterfactual.reporting import TABLE_CONTRACTS, load_bound_tables
from gtmask_counterfactual.taxonomy import NATIVE_CLASSES
from gtmask_counterfactual.visual_assets import write_visual_asset_registry


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = (
    ROOT / "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py"
)


def _rect(*, cx: float = 100.0) -> dict[str, float]:
    return {
        "cx_px": cx,
        "cy_px": 100.0,
        "theta_deg": 0.0,
        "width_px": 80.0,
        "height_px": 20.0,
    }


def _candidate(
    sample_id: str,
    route: str,
    branch: str,
    rank: int,
    *,
    positive: bool,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "route": route,
        "branch": branch,
        "candidate_id": f"{route}-{branch}-{sample_id}-{rank}",
        "native_rank": rank,
        "native_score": 1.0 / rank,
        **_rect(cx=100.0 if positive else 400.0 + rank),
    }


def _pool(route: str, branch: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(sample: str, positive_rank: int | None, count: int) -> None:
        rows.extend(
            _candidate(
                sample,
                route,
                branch,
                rank,
                positive=positive_rank == rank,
            )
            for rank in range(1, count + 1)
        )

    if branch == "predicted":
        add("s0", 1, 1)
        add("s1", 2, 2)
        add("s2", 6, 6)
        add("s6", None, 1)
        add("s7", None, 1)
    else:
        add("s0", 1, 1)
        add("s1", 2, 2)
        add("s2", 6, 6)
        add("s3", 1, 1)
        add("s4", 2, 2)
        add("s5", 6, 6)
        add("s6", None, 1)
        add("s7", None, 1)
    frame = pd.DataFrame(rows)
    if route == "D1":
        # The real D1 producer persists both canonical and source aliases.
        frame["jaw_width_px"] = frame["width_px"]
        frame["rectangle_height_px"] = frame["height_px"]
    return frame


def _per_sample(candidates: pd.DataFrame, route: str, branch: str) -> pd.DataFrame:
    counts = candidates.groupby("sample_id").size().to_dict()
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "route": route,
                "branch": branch,
                "candidate_count": int(counts.get(sample_id, 0)),
                "no_output": sample_id not in counts,
                "technical_failure": sample_id == "s7",
                "status": "TECHNICAL_FAILURE" if sample_id == "s7" else "COMPLETE",
            }
            for sample_id in (f"s{index}" for index in range(8))
        ]
    )


def _content_json(path: Path, value: dict[str, Any]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    return atomic_json(path, payload)


def test_nested_source_lock_discovery_ignores_logical_path_fields() -> None:
    final_lock = {
        "path": "/immutable/FINAL_RUN_LOCK.json",
        "sha256": "a" * 64,
        "bytes": 123,
    }
    payload = {
        "sources": {"unified": {"final_lock": final_lock}},
        "integrity_checks": {
            "deterministic_galleries_audited": {
                "details": {
                    "summaries": [
                        {
                            "path": "qualitative/recovery/example",
                            "status": "PASS",
                        }
                    ]
                }
            }
        },
    }

    observed = postprocess_module._collect_named_artifact_records(
        payload,
        names=frozenset({"final_lock", "source_lock_verification"}),
        prefix="source_lock_payload",
    )

    assert observed == [("source_lock_payload.sources.unified.final_lock", final_lock)]


def test_nested_source_lock_discovery_rejects_incomplete_named_record() -> None:
    with pytest.raises(PostprocessContractError, match="incomplete named artifact"):
        postprocess_module._collect_named_artifact_records(
            {"sources": {"unified": {"final_lock": {"path": "/missing/hash"}}}},
            names=frozenset({"final_lock", "source_lock_verification"}),
        )


def test_artifact_collection_honors_inline_discriminator_first() -> None:
    inline = {
        "kind": "inline",
        "sha256": "a" * 64,
        "value": {"path": "logical.evaluator.call.path"},
    }

    assert postprocess_module._collect_records(
        {"contract": inline}, prefix="bindings.evaluator"
    ) == []

    with pytest.raises(PostprocessContractError, match="incomplete artifact record"):
        postprocess_module._collect_records(
            {"implementation": {"path": "/missing/hash"}},
            prefix="bindings.evaluator",
        )


def _build_run(
    tmp_path: Path,
    *,
    claim: bool,
    include_d1: bool = True,
    tamper_final_bits: bool = False,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    run = tmp_path / "runs" / "fair_gtmask_counterfactual_g1_c1_d1_20260813T000000Z"
    sample_ids = [f"s{index}" for index in range(8)]
    formal_source = tmp_path / "formal_outcomes.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "system_name": system,
                "selected_candidate_id": (
                    ""
                    if sample_id == "s7"
                    else f"D1-predicted-{sample_id}-1"
                    if system.startswith("d1_")
                    else f"{system.split('_')[0].upper()}-predicted-{sample_id}-1"
                ),
                "formal_score": 0.25,
                "is_selected": sample_id != "s7",
                "row_kind": "decision_no_output" if sample_id == "s7" else "candidate",
                "no_output": sample_id == "s7",
                "selected_correct": sample_id == "s0",
            }
            for system in (
                "g1_gated_primary",
                "c1_gated_primary",
                "d1_top5_r7_gated",
                "d1_top10_locked",
                "d1_allnms_locked",
            )
            for sample_id in sample_ids
        ]
    ).to_parquet(formal_source, index=False)
    asset_dir = tmp_path / "visual_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_paths: dict[str, Path] = {}
    for name, value in (
        ("rgb", np.zeros((8, 8, 3), dtype=np.uint8)),
        ("depth", np.ones((8, 8), dtype=np.float32)),
        ("predicted_mask", np.zeros((8, 8), dtype=np.uint8)),
        ("predicted_probability", np.zeros((8, 8), dtype=np.float32)),
        ("gt_mask", np.ones((8, 8), dtype=np.uint8)),
    ):
        path = asset_dir / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        asset_paths[name] = path
    visual_source = tmp_path / "visual_source.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "source_rgb_path": str(asset_paths["rgb"]),
                "source_rgb_sha256": artifact_record(asset_paths["rgb"])["sha256"],
                "source_depth_path": str(asset_paths["depth"]),
                "source_depth_sha256": artifact_record(asset_paths["depth"])["sha256"],
                "predicted_mask_path": str(asset_paths["predicted_mask"]),
                "predicted_mask_sha256": artifact_record(asset_paths["predicted_mask"])["sha256"],
                "predicted_probability_path": str(asset_paths["predicted_probability"]),
                "predicted_probability_sha256": artifact_record(asset_paths["predicted_probability"])["sha256"],
                "intrinsics_path": "",
                "intrinsics_sha256": "",
            }
            for sample_id in sample_ids
        ]
    ).to_parquet(visual_source, index=False)
    source = tmp_path / "FINAL_RUN_LOCK.json"
    atomic_json(
        source,
        {
            "status": "COMPLETE",
            "inventory": [artifact_record(formal_source), artifact_record(visual_source)],
        },
    )
    bootstrap_run(
        run,
        source_verification={
            "status": "PASS",
            "full_inventory_byte_rehash": True,
            "sources": {},
        },
    )
    transition_pipeline_status(
        run,
        RunState.P1_BASELINE_REPLAY_PASS,
        first_incomplete_stage=RunState.P2_GT_MAPPING_PASS.value,
    )

    sample_manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "scene_id": [f"scene-{index // 2}" for index in range(8)],
            "frame_id": [f"frame-{index}" for index in range(8)],
            "language_prompt": [f"pick sample {index}" for index in range(8)],
            "gt_grasp_rectangles": [
                [canonical_corners(_rect()).tolist()] for _ in sample_ids
            ],
        }
    )
    sample_path = atomic_parquet(
        sample_manifest, run / "02_sample_manifest/counterfactual_manifest.parquet"
    )
    covariates = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "query_type": ["name", "attribute", "relation", "location"] * 2,
            "predicted_mask_iou": [0.1, 0.3, 0.55, 0.75, 0.95, 0.2, 0.6, 0.8],
            "target_area_fraction": [0.01 * (index + 1) for index in range(8)],
            "mask_component_count": [1, 1, 2, 1, 3, 1, 2, 1],
            "mask_boundary_complexity": [0.1 * (index + 1) for index in range(8)],
            "valid_depth_ratio": [0.5 + 0.05 * index for index in range(8)],
            "scene_family": ["table", "floor"] * 4,
            "frame_family": [f"family-{index // 2}" for index in range(8)],
        }
    )
    covariate_path = atomic_parquet(
        covariates, run / "03_gt_mask_registry/sample_covariates.parquet"
    )
    gt_path = sample_path
    registry = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "original_gt_mask_sha256": artifact_record(asset_paths["gt_mask"])["sha256"],
                "original_gt_mask_path": str(asset_paths["gt_mask"]),
                "original_height": 8,
                "original_width": 8,
                "mapping_status": "PASS",
                "pixel_qa_status": "P2_MAPPING_QA_PASS",
                "annotation_suspect": sample_id == "s7",
            }
            for sample_id in sample_ids
        ]
    )
    registry_path = atomic_parquet(
        registry, run / "03_gt_mask_registry/gt_mask_registry.parquet"
    )
    mapping_qa = atomic_json(
        run / "03_gt_mask_registry/GT_MASK_MAPPING_AUDIT.json",
        {
            "status": "PASS",
            "stage": "P2_GT_MAPPING_PASS",
            "pixel_qa_status": "P2_MAPPING_QA_PASS",
            "mapping_qa_gt_mask_rows_read": 8,
            "candidate_generation_gt_mask_rows_read": 0,
        },
    )
    transition_pipeline_status(
        run,
        RunState.P2_GT_MAPPING_PASS,
        first_incomplete_stage=RunState.P3_PROTOCOL_LOCKED.value,
    )
    record = artifact_record(source)
    code_record = artifact_record(formal_source)
    routes = {
        "g1": {"allowed_gt_branches": ["gt_oracle"]},
        "c1": {"allowed_gt_branches": ["gt_oracle"]},
        "d1": {
            "allowed_gt_branches": ["gt_oracle"],
            "case": "B",
            "mask_affects_raw_sampling": True,
            "raw_candidate_regeneration_required": True,
            "filter_only_primary_allowed": False,
        },
    }
    bindings = {
        "source_locks": {"synthetic_source": record},
        "source_code": {"synthetic_source": code_record},
        "configs": {"synthetic_source": code_record},
        "baseline_replay": record,
        "sample_manifest": artifact_record(sample_path),
        "gt_grasp_source": artifact_record(sample_path),
        "gt_mask_registry": artifact_record(registry_path),
        "mapping_qa": artifact_record(mapping_qa),
        "route_contracts": inline_binding(routes),
        "resize_rules": inline_binding({"binary": "PIL nearest"}),
        "evaluator": artifact_record(EVALUATOR),
        "taxonomy": inline_binding({"classes": list(NATIVE_CLASSES)}),
        "statistics": inline_binding({"iterations": 10_000, "seed": 20260813}),
        "case_selection": inline_binding({"rule": "synthetic"}),
    }
    lock = create_protocol_lock(
        run,
        bindings=bindings,
        declaration={
            "gt_candidate_generation_authorized": True,
            "bulk_execution_max_count": 1,
            "mapping_qa_gt_mask_rows_read_before_lock": 8,
            "candidate_generation_gt_mask_rows_read_before_lock": 0,
            "routes": routes,
        },
        test_only_allow_synthetic_contract=True,
    )
    if claim:
        claim_bulk_execution(run)
        gt_authority_path = materialize_gt_grasp_authority(
            run, protocol_lock=lock, expected_count=8
        )
        gt_authority = json.loads(gt_authority_path.read_text(encoding="utf-8"))
        gt_path = Path(gt_authority["registry"]["path"])
        covariate_authority_path = _content_json(
            run / "03_gt_mask_registry/SAMPLE_COVARIATES_AUTHORITY.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "sample_count": 8,
                "protocol_lock": artifact_record(lock),
                "execution_authority_mode": "prospective_execution_claim",
                "execution_claim": artifact_record(
                    run / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
                ),
                "sample_manifest": artifact_record(sample_path),
                "gt_mask_registry": artifact_record(registry_path),
                "visual_assets": {"synthetic": True},
                "resource_gate": {"synthetic": True},
                "covariates": artifact_record(covariate_path),
                "gt_mask_rows_read": 8,
                "gt_grasp_rows_read": 0,
            },
        )
    else:
        # The no-authority test must fail before this placeholder can be opened.
        gt_authority_path = sample_path
        covariate_authority_path = sample_path

    visual_manifest = write_visual_asset_registry(
        run,
        protocol_lock=lock,
        unified_samples=artifact_record(visual_source),
        d1_samples=artifact_record(visual_source),
        expected_count=8,
    )

    execution_routes = ("G1", "C1", "D1") if include_d1 else ("G1", "C1")
    final = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "route": route,
                "final_correct": sample_id in ({"s0", "s1"} if tamper_final_bits else {"s0"}),
            }
            for route in execution_routes
            for sample_id in sample_ids
        ]
    )
    final_path = atomic_parquet(
        final, run / "04_predicted_replay/frozen_final_outcomes.parquet"
    )
    derived_path = _content_json(
        run / "04_predicted_replay/baseline_replay.json",
        {
            "status": "PASS",
            "raw_test_ground_truth_rows_read": 0,
            "routes": {
                route: {
                    "N": 8,
                    "native_correct": 1,
                    "oracle_top5": 2,
                    "oracle_top10": 3,
                    "oracle_all": 3,
                    "no_output": 3,
                }
                for route in ("g1", "c1", "d1")
            },
        },
    )
    route_replays: dict[str, dict[str, Any]] = {}
    for route in ("g1", "c1", "d1"):
        value: dict[str, Any] = {
            "status": "PASS",
            "route": route,
            "sample_count": 8,
            "no_output_count": 3,
            "raw_test_ground_truth_rows_read": 0,
        }
        if route in {"g1", "c1"}:
            value.update(
                serializer_atol=0.0,
                maximum_absolute_differences={"native_score": 0.0},
            )
        else:
            value.update(
                branch="predicted",
                candidate_count=11,
                comparisons={"allnms": True},
            )
        replay_path = _content_json(
            run / "04_predicted_replay" / route / "manifest.json", value
        )
        route_replays[route] = artifact_record(replay_path)
    baseline_path = _content_json(
        run / "04_predicted_replay/BASELINE_REPLAY_MANIFEST.json",
        {
            "status": "PASS",
            "sample_count": 8,
            "derived_reconciliation": artifact_record(derived_path),
            "route_replays": route_replays,
            "all_three_predicted_pipelines_exact": True,
            "raw_test_ground_truth_rows_read": 0,
        },
    )
    candidate_paths: dict[str, dict[str, Any]] = {}
    sample_paths: dict[str, dict[str, Any]] = {}
    route_manifests: dict[str, dict[str, Any]] = {}
    raw_paths: dict[str, Path] = {}
    for route in execution_routes:
        for branch in ("predicted", "gt_oracle"):
            key = f"{route}|{branch}"
            candidates = _pool(route, branch)
            candidate_path = atomic_parquet(
                candidates,
                run
                / "07_candidate_tables/raw"
                / route.lower()
                / branch
                / "candidates.parquet",
            )
            per_sample_path = atomic_parquet(
                _per_sample(candidates, route, branch),
                run
                / "07_candidate_tables/raw"
                / route.lower()
                / branch
                / "per_sample.parquet",
            )
            candidate_paths[key] = artifact_record(candidate_path)
            sample_paths[key] = artifact_record(per_sample_path)
            source_contract = (
                {"baseline_replay": artifact_record(baseline_path)}
                if branch == "predicted"
                else {
                    "protocol_lock": artifact_record(lock),
                    "execution_claim": (
                        artifact_record(
                            run / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
                        )
                        if claim
                        else {
                            "path": str(
                                run
                                / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
                            ),
                            "sha256": "0" * 64,
                            "bytes": 0,
                        }
                    ),
                }
            )
            if claim or branch == "predicted":
                route_manifest_path = write_route_frame_manifest(
                    run,
                    route=route,
                    branch=branch,
                    candidates=candidate_paths[key],
                    per_sample=sample_paths[key],
                    source_contract=source_contract,
                    sample_count=8,
                )
            else:
                route_manifest_path = _content_json(
                    run
                    / "07_candidate_tables/raw"
                    / route.lower()
                    / branch
                    / "manifest.json",
                    {
                        "status": "COMPLETE",
                        "route": route,
                        "branch": branch,
                        "sample_count": 8,
                        "candidate_count": len(candidates),
                        "candidates": candidate_paths[key],
                        "per_sample": sample_paths[key],
                        "source_contract": source_contract,
                    },
                )
            route_manifests[key] = artifact_record(route_manifest_path)
            raw_paths[key] = candidate_path
    final_authority_path = write_final_outcomes_authority(
        run,
        protocol_lock=lock,
        final_outcomes=artifact_record(final_path),
        selector_sources={
            route: artifact_record(formal_source) for route in execution_routes
        },
        routes=execution_routes,
    )
    d1_blocker_path = None
    if not include_d1:
        d1_blocker_path = _content_json(
            run / "00_audit/machine_blockers/D1_CASE_B_UNRECOVERABLE_BLOCKER.json",
            {
                "status": "UNRECOVERABLE_BLOCKER",
                "blocker_class": "IRRECOVERABLE_FROZEN_SOURCE_EVIDENCE",
                "missing_evidence": "frozen scorer source is irretrievably absent",
                "search_paths": ["/synthetic/source/archive"],
                "stack_trace": "RuntimeError: frozen scorer source unavailable",
                "resume_command": "python -m tools.gtmask_counterfactual.run_route --route d1 --resume",
                "raw_candidate_regeneration_required": True,
                "filter_only_primary_allowed": False,
                "execution_attempted": False,
            },
        )
    input_artifacts = {
        "sample_manifest": artifact_record(sample_path),
        "ground_truth": artifact_record(gt_path),
        "ground_truth_authority": artifact_record(gt_authority_path),
        "sample_covariates": artifact_record(covariate_path),
        "sample_covariates_authority": artifact_record(covariate_authority_path),
        "final_outcomes": artifact_record(final_path),
        "final_outcomes_authority": artifact_record(final_authority_path),
        "baseline_replay": artifact_record(baseline_path),
        "visual_assets": artifact_record(visual_manifest),
        "candidates": candidate_paths,
        "per_sample": sample_paths,
        "route_manifests": route_manifests,
    }
    if d1_blocker_path is not None:
        input_artifacts["d1_blocker"] = artifact_record(d1_blocker_path)
    input_manifest = write_postprocess_input_manifest(
        run,
        artifacts=input_artifacts,
        protocol_lock=lock,
        sample_count=8,
    )
    if claim:
        transitions = [
            (
                RunState.P4_C1_PILOT_PASS,
                RunState.P5_C1_FULL_COMPLETE,
            ),
            (
                RunState.P5_C1_FULL_COMPLETE,
                RunState.P5B_G1_FULL_COMPLETE,
            ),
            (
                RunState.P5B_G1_FULL_COMPLETE,
                RunState.P6_D1_COUNTERFACTUAL_COMPLETE,
            ),
        ]
        if include_d1:
            transitions.extend(
                [
                    (
                        RunState.P6_D1_COUNTERFACTUAL_COMPLETE,
                        RunState.P7_TAXONOMY_COMPLETE,
                    ),
                ]
            )
        for state, next_state in transitions:
            transition_pipeline_status(
                run, state, first_incomplete_stage=next_state.value
            )
    return run, lock, input_manifest, raw_paths


def test_postprocess_synthetic_end_to_end_resume_and_tamper(tmp_path: Path) -> None:
    run, lock, input_manifest, raw_paths = _build_run(tmp_path, claim=True)
    status = run_postprocess(
        run,
        protocol_lock=lock,
        input_manifest=input_manifest,
        expected_sample_count=8,
        bootstrap_iterations=32,
    )
    route_status = json.loads(status.read_text(encoding="utf-8"))
    assert route_status["routes"] == {route: "COMPLETE" for route in ("G1", "C1", "D1")}
    tables, manifest = load_bound_tables(run)
    assert set(tables) == set(TABLE_CONTRACTS)
    assert manifest["table_count"] == len(TABLE_CONTRACTS)
    native = tables["native_failure_taxonomy.csv"]
    for route in ("G1", "C1", "D1"):
        counts = native.loc[native["route"].eq(route)].set_index("taxonomy")["count"]
        assert counts.tolist() == [1] * 8
    branch = tables["branch_metrics.csv"].set_index(["route", "branch"])
    assert int(branch.loc[("G1", "predicted"), "no_output"]) == 3
    assert {
        "candidate_count_mean",
        "candidate_count_median",
        "candidate_count_p95",
        "native_j_at_1",
        "j_at_5",
        "mrr",
        "first_positive_rank_distribution_json",
        "positive_candidates_per_sample_mean",
    }.issubset(branch.columns)
    paired = tables["pred_vs_gt_paired_metrics.csv"]
    assert {
        "delta_native_j_at_1",
        "delta_oracle_at_5",
        "delta_oracle_at_10",
        "oracle_grounding_ceiling",
        "residual_generator_failure_rate",
        "candidate_count_delta_mean",
        "first_positive_rank_delta_mean_both_positive",
    }.issubset(paired.columns)
    depth_selector = tables["frozen_selector_transfer.csv"]
    depth_selector = depth_selector.loc[
        depth_selector["route"].eq("D1")
        & depth_selector["branch"].eq("predicted")
    ]
    assert set(depth_selector["selector"]) == {
        "d1_top5_r7_gated",
        "d1_top10_locked",
        "d1_allnms_locked",
    }
    assert set(depth_selector["pool"]) == {"top5", "top10", "allnms"}
    depth_metrics = pd.read_csv(
        run / "08_metrics/d1_locked_depth_selector_metrics.csv"
    )
    assert set(depth_metrics["pool"]) == {"top5", "top10", "allnms"}
    depth_taxonomy = pd.read_csv(
        run / "09_failure_taxonomy/d1_locked_depth_bottleneck_summary.csv"
    )
    assert len(depth_taxonomy) == 3 * 6
    assert set(depth_taxonomy["selector"]) == {
        "d1_top5_r7_gated",
        "d1_top10_locked",
        "d1_allnms_locked",
    }
    statistical_csv = tables["statistical_tests.csv"]
    assert {
        "frame_ci_low",
        "frame_ci_high",
        "b_reference_only",
        "c_counterfactual_only",
        "bootstrap_iterations",
        "bootstrap_seed",
    }.issubset(statistical_csv.columns)
    transitions = tables["candidate_pool_transitions.csv"]
    for (_, _), group in transitions.groupby(["route", "transition_family"]):
        assert int(group["count"].sum()) == 8
    mechanisms = tables["candidate_mechanism_summary.csv"]
    assert set(mechanisms["route"]) == {"G1", "C1", "D1"}
    assert mechanisms["evidence_scope"].eq(
        "observable_final_nms_pool_transition_only"
    ).all()
    per_sample_mechanisms = pd.read_parquet(
        run / "08_metrics/candidate_pool_mechanism_per_sample.parquet"
    )
    assert len(per_sample_mechanisms) == 3 * 8
    assert per_sample_mechanisms["crop_change_status"].eq(
        "UNKNOWN_SOURCE_ARTIFACT_ABSENT"
    ).all()
    matches = pd.read_parquet(
        run / "07_candidate_tables/cross_branch_candidate_matches.parquet"
    )
    assert set(matches["match_status"]).issubset(
        {
            "matched_pred_gt_candidate",
            "pred_only_candidate",
            "gt_only_candidate",
            "empty_both_pool",
        }
    )
    mechanism_contract = json.loads(
        (run / "08_metrics/CANDIDATE_MECHANISM_EVIDENCE.json").read_text()
    )
    assert mechanism_contract["causal_claim_supported"] is False
    assert mechanism_contract["raw_candidate_count_change"] == (
        "UNKNOWN_SOURCE_ARTIFACT_ABSENT"
    )
    statistics = json.loads(
        (run / "10_statistics/statistics.json").read_text(encoding="utf-8")
    )
    assert statistics["bootstrap_iterations"] == 32
    assert len(statistics["comparisons"]) == 10
    pipeline = json.loads((run / "pipeline_status.json").read_text(encoding="utf-8"))
    assert pipeline["status"] == "P8_STATISTICS_COMPLETE"
    assert (
        run_postprocess(
            run,
            protocol_lock=lock,
            input_manifest=input_manifest,
            resume=True,
            expected_sample_count=8,
            bootstrap_iterations=32,
        )
        == status
    )

    tampered = pd.read_parquet(raw_paths["G1|predicted"])
    tampered.loc[0, "native_score"] = 123.0
    tampered.to_parquet(raw_paths["G1|predicted"], index=False)
    with pytest.raises(PostprocessContractError, match="hash/byte record differs"):
        run_postprocess(
            run,
            protocol_lock=lock,
            input_manifest=input_manifest,
            resume=True,
            expected_sample_count=8,
            bootstrap_iterations=32,
        )


def test_resume_repairs_crash_between_route_status_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, lock, input_manifest, _ = _build_run(tmp_path, claim=True)
    original_complete = postprocess_module.complete_bulk_execution

    def interrupted_completion(*args: Any, **kwargs: Any) -> Path:
        del args, kwargs
        raise RuntimeError("synthetic crash after route-status publication")

    monkeypatch.setattr(
        postprocess_module, "complete_bulk_execution", interrupted_completion
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        run_postprocess(
            run,
            protocol_lock=lock,
            input_manifest=input_manifest,
            expected_sample_count=8,
            bootstrap_iterations=16,
        )
    route_status = run / "08_metrics/ROUTE_STATUS.json"
    completion = run / "01_protocol_lock/COUNTERFACTUAL_EXECUTION_COMPLETE.json"
    assert route_status.is_file()
    assert not completion.exists()

    monkeypatch.setattr(
        postprocess_module, "complete_bulk_execution", original_complete
    )
    assert (
        run_postprocess(
            run,
            protocol_lock=lock,
            input_manifest=input_manifest,
            resume=True,
            expected_sample_count=8,
            bootstrap_iterations=16,
        )
        == route_status
    )
    assert completion.is_file()
    pipeline = json.loads((run / "pipeline_status.json").read_text(encoding="utf-8"))
    assert pipeline["status"] == "P8_STATISTICS_COMPLETE"


def test_legacy_caller_authored_gallery_cannot_advance_lifecycle(
    tmp_path: Path,
) -> None:
    run, lock, input_manifest, _ = _build_run(tmp_path, claim=True)
    run_postprocess(
        run,
        protocol_lock=lock,
        input_manifest=input_manifest,
        expected_sample_count=8,
        bootstrap_iterations=16,
    )
    rows = []
    boards = []
    manual = []
    for route in ("G1", "C1", "D1"):
        for sample_id in ("s0", "s1"):
            row = {
                "sample_id": sample_id,
                "route": route,
                "category": "clear_grounding_limited",
                "mechanism_pure": True,
                "presentation_eligible": True,
                "feature_vector_json": json.dumps([0.0 if sample_id == "s0" else 1.0]),
                "asset_bundle_sha256": canonical_sha256(
                    {"sample_id": sample_id, "route": route}
                ),
            }
            rows.append(row)
            formats = {}
            for suffix in ("png", "svg"):
                path = run / "14_galleries" / f"{route}_{sample_id}.{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"board")
                formats[suffix] = artifact_record(path)
            boards.append(
                {
                    "status": "AUTO_QA_PASS",
                    "sample_id": sample_id,
                    "route": route,
                    "category": "clear_grounding_limited",
                    **formats,
                }
            )
            manual.append(
                {
                    "sample_id": sample_id,
                    "route": route,
                    "category": "clear_grounding_limited",
                    "status": "PASS",
                }
            )
    eligible = pd.DataFrame(rows)
    gallery = write_gallery_manifest(
        run,
        eligible=eligible,
        selected=eligible,
        board_qa=boards,
        postprocess_manifest=artifact_record(
            run / "08_metrics/POSTPROCESS_MANIFEST.json"
        ),
        manual_qa_status="PASS",
        manual_qa_rows=manual,
    )
    assert json.loads(gallery.read_text(encoding="utf-8"))["status"] == "COMPLETE"
    with pytest.raises(RuntimeError, match="gallery build manifest"):
        accept_gallery(run)
    pipeline = json.loads((run / "pipeline_status.json").read_text(encoding="utf-8"))
    assert pipeline["status"] == "P8_STATISTICS_COMPLETE"

def test_no_scientific_frame_is_opened_before_execution_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, lock, input_manifest, _ = _build_run(tmp_path, claim=False)
    opened: list[str] = []

    def forbidden_read(*args: Any, **kwargs: Any) -> pd.DataFrame:
        del args, kwargs
        opened.append("frame")
        raise AssertionError("scientific frame opened before execution authority")

    monkeypatch.setattr(
        "gtmask_counterfactual.postprocess._read_parquet_columns", forbidden_read
    )
    with pytest.raises((PermissionError, PostprocessContractError)):
        run_postprocess(
            run,
            protocol_lock=lock,
            input_manifest=input_manifest,
            expected_sample_count=8,
            bootstrap_iterations=8,
        )
    assert opened == []


def test_frozen_final_authority_rejects_arbitrary_copied_bits(tmp_path: Path) -> None:
    with pytest.raises(PostprocessContractError, match="differ from frozen source"):
        _build_run(tmp_path, claim=True, tamper_final_bits=True)


def test_core_postprocess_is_complete_without_fabricating_d1(
    tmp_path: Path,
) -> None:
    run, lock, input_manifest, _ = _build_run(
        tmp_path, claim=True, include_d1=False
    )
    route_path = run_postprocess(
        run,
        protocol_lock=lock,
        input_manifest=input_manifest,
        expected_sample_count=8,
        bootstrap_iterations=16,
    )
    route = json.loads(route_path.read_text(encoding="utf-8"))
    assert route["status"] == "COMPLETE"
    assert route["routes"] == {"G1": "COMPLETE", "C1": "COMPLETE"}
    assert route["d1_secondary_status"] == "PENDING_AFTER_CORE"
    pipeline = json.loads((run / "pipeline_status.json").read_text(encoding="utf-8"))
    assert pipeline["status"] == "P8_STATISTICS_COMPLETE"
    tables, _ = load_bound_tables(run)
    assert set(tables["branch_metrics.csv"]["route"]) == {"G1", "C1"}
