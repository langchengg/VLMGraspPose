from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gtmask_counterfactual.acceptance import accept_gallery
from gtmask_counterfactual.audit import bootstrap_run, transition_pipeline_status
from gtmask_counterfactual.contracts import RunState
from gtmask_counterfactual.gallery_pipeline import (
    GalleryContractError,
    accept_gallery_manual_qa,
    manual_qa_signature,
    prepare_gallery,
    verify_gallery_build,
)
from gtmask_counterfactual.independent import canonical_corners
from gtmask_counterfactual.io import (
    artifact_record,
    atomic_json,
    atomic_parquet,
    canonical_sha256,
)
from gtmask_counterfactual.postprocess import write_final_outcomes_authority
from gtmask_counterfactual.protocol import (
    claim_bulk_execution,
    create_protocol_lock,
    inline_binding,
)
from gtmask_counterfactual.taxonomy import NATIVE_CLASSES
from gtmask_counterfactual.visual_assets import write_visual_asset_registry


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "runs/fair_unified_reranking_20260809_103012/configs/canonical_evaluator.py"
CATEGORIES = (
    "clear",
    "grounding_selection",
    "generator",
    "ranking",
    "regression",
    "no_output",
    "no_change",
    "annotation",
)


def _content_json(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    return atomic_json(path, payload)


def _candidate(
    sample_id: str,
    route: str,
    branch: str,
    candidate_id: str,
    rank: int,
    *,
    success: bool,
    cx: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "route": route,
        "branch": branch,
        "candidate_id": candidate_id,
        "native_rank": rank,
        "native_score": 1.0 / rank,
        "cx_px": cx,
        "cy_px": 4.0,
        "theta_deg": 0.0,
        "width_px": 3.0,
        "height_px": 2.0,
        "candidate_success": success,
        "matched_gt_index": 0,
        "best_same_gt_iou": 0.6 if success else 0.1,
        "best_same_gt_angle_error_deg": 5.0 if success else 40.0,
    }


def _scenario(
    sample_id: str, route: str, category: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    pred: list[dict[str, object]] = []
    gt: list[dict[str, object]] = []

    def add(
        target: list[dict[str, object]],
        branch: str,
        candidate_id: str,
        rank: int,
        success: bool,
        cx: float,
    ) -> None:
        target.append(
            _candidate(
                sample_id,
                route,
                branch,
                candidate_id,
                rank,
                success=success,
                cx=cx,
            )
        )

    flags: dict[str, object]
    if category == "clear":
        add(pred, "predicted", "p1", 1, False, 6.0)
        add(gt, "gt_oracle", "g1", 1, True, 4.0)
        flags = dict(tax=NATIVE_CLASSES[4], pred=False, pred_native=False, gt=True, gt_native=True)
    elif category == "grounding_selection":
        add(pred, "predicted", "p1", 1, False, 6.0)
        add(gt, "gt_oracle", "g1", 1, False, 6.0)
        add(gt, "gt_oracle", "g2", 2, True, 4.0)
        flags = dict(tax=NATIVE_CLASSES[5], pred=False, pred_native=False, gt=True, gt_native=False)
    elif category == "generator":
        add(pred, "predicted", "p1", 1, False, 6.0)
        add(gt, "gt_oracle", "g1", 1, False, 6.0)
        flags = dict(tax=NATIVE_CLASSES[7], pred=False, pred_native=False, gt=False, gt_native=False)
    elif category == "ranking":
        add(pred, "predicted", "p1", 1, False, 6.0)
        add(pred, "predicted", "p2", 2, True, 4.0)
        add(gt, "gt_oracle", "g1", 1, True, 4.0)
        flags = dict(tax=NATIVE_CLASSES[2], pred=True, pred_native=False, gt=True, gt_native=True)
    elif category == "regression":
        add(pred, "predicted", "p1", 1, True, 4.0)
        add(gt, "gt_oracle", "g1", 1, False, 6.0)
        flags = dict(tax=NATIVE_CLASSES[1], pred=True, pred_native=True, gt=False, gt_native=False)
    elif category == "no_output":
        add(gt, "gt_oracle", "g1", 1, True, 4.0)
        flags = dict(tax=NATIVE_CLASSES[4], pred=False, pred_native=False, gt=True, gt_native=True)
    elif category == "no_change":
        add(pred, "predicted", "p1", 1, True, 4.0)
        add(gt, "gt_oracle", "g1", 1, True, 4.0)
        flags = dict(tax=NATIVE_CLASSES[1], pred=True, pred_native=True, gt=True, gt_native=True)
    else:
        add(pred, "predicted", "p1", 1, True, 3.0)
        add(gt, "gt_oracle", "g1", 1, True, 7.0)
        flags = dict(tax=NATIVE_CLASSES[1], pred=True, pred_native=True, gt=True, gt_native=True)
    pred_first = next((int(row["native_rank"]) for row in pred if row["candidate_success"]), None)
    gt_first = next((int(row["native_rank"]) for row in gt if row["candidate_success"]), None)
    final_correct = bool(flags["pred_native"])
    row = {
        "sample_id": sample_id,
        "route": route,
        "technical_failure": False,
        "native_taxonomy": flags["tax"],
        "final_correct": final_correct,
        "pred_all_positive": flags["pred"],
        "pred_native_correct": flags["pred_native"],
        "pred_top5_positive": flags["pred"],
        "pred_top10_positive": flags["pred"],
        "gt_all_positive": flags["gt"],
        "gt_native_correct": flags["gt_native"],
        "gt_top5_positive": flags["gt"],
        "gt_top10_positive": flags["gt"],
        "GT_mask_regression": category == "regression",
        "pred_no_output": not pred,
        "gt_no_output": not gt,
        "pred_candidate_count": len(pred),
        "gt_candidate_count": len(gt),
        "pred_positive_candidate_count": sum(bool(value["candidate_success"]) for value in pred),
        "gt_positive_candidate_count": sum(bool(value["candidate_success"]) for value in gt),
        "pred_first_positive_rank": pred_first,
        "gt_first_positive_rank": gt_first,
        "predicted_mask_iou": 0.1 + 0.01 * int(sample_id[1:]),
        "target_area_fraction": 0.05,
        "mask_component_count": 1,
        "mask_boundary_complexity": 0.2,
        "valid_depth_ratio": 0.9,
        "annotation_suspect": category == "annotation",
    }
    return pred, gt, row


def _build_gallery_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs/fair_gtmask_counterfactual_g1_c1_d1_20260813T010000Z"
    sample_ids = [f"s{index:02d}" for index in range(16)]
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    arrays = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth": np.ones((8, 8), dtype=np.float32),
        "pred_mask": np.zeros((8, 8), dtype=np.uint8),
        "probability": np.zeros((8, 8), dtype=np.float32),
        "gt_mask": np.ones((8, 8), dtype=np.uint8),
    }
    paths: dict[str, Path] = {}
    for name, array in arrays.items():
        path = asset_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = path
    visual_source = tmp_path / "visual.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "source_rgb_path": str(paths["rgb"]),
                "source_rgb_sha256": artifact_record(paths["rgb"])["sha256"],
                "source_depth_path": str(paths["depth"]),
                "source_depth_sha256": artifact_record(paths["depth"])["sha256"],
                "predicted_mask_path": str(paths["pred_mask"]),
                "predicted_mask_sha256": artifact_record(paths["pred_mask"])["sha256"],
                "predicted_probability_path": str(paths["probability"]),
                "predicted_probability_sha256": artifact_record(paths["probability"])["sha256"],
                "intrinsics_path": "",
                "intrinsics_sha256": "",
            }
            for sample_id in sample_ids
        ]
    ).to_parquet(visual_source, index=False)
    formal_rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    post_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    systems = {"G1": "g1_gated_primary", "C1": "c1_gated_primary", "D1": "d1_top5_r7_gated"}
    for route in systems:
        for index, sample_id in enumerate(sample_ids):
            category = CATEGORIES[index // 2]
            pred, gt, post = _scenario(sample_id, route, category)
            labels.extend(pred)
            labels.extend(gt)
            post_rows.append(post)
            selected = "" if not pred else str(pred[0]["candidate_id"])
            formal_rows.append(
                {
                    "sample_id": sample_id,
                    "system_name": systems[route],
                    "selected_candidate_id": selected,
                    "selected_route": route,
                    "selected_source_route": route if selected else "",
                    "selected_correct": bool(post["final_correct"]),
                    "formal_score": np.nan if not selected else 0.25,
                    "is_selected": bool(selected),
                    "row_kind": "candidate" if selected else "decision_no_output",
                    "no_output": not bool(selected),
                }
            )
            if route == "D1":
                if index == 0 and selected:
                    # Real D1 formal bundles contain multiple candidate rows.
                    # This forces selector normalization to retain later
                    # decision_no_output rows while removing unselected
                    # candidate rows.
                    formal_rows.append(
                        {
                            "sample_id": sample_id,
                            "system_name": systems[route],
                            "selected_candidate_id": f"{sample_id}-unused",
                            "selected_route": route,
                            "selected_source_route": route,
                            "selected_correct": False,
                            "formal_score": 0.1,
                            "is_selected": False,
                            "row_kind": "candidate",
                            "no_output": False,
                        }
                    )
                for depth_system in ("d1_top10_locked", "d1_allnms_locked"):
                    formal_rows.append(
                        {
                            "sample_id": sample_id,
                            "system_name": depth_system,
                            "selected_candidate_id": selected,
                            "selected_source_route": "D1" if selected else "",
                            "selected_correct": bool(post["final_correct"]),
                            "formal_score": np.nan if not selected else 0.2,
                            "is_selected": bool(selected),
                            "row_kind": "candidate" if selected else "decision_no_output",
                            "no_output": not bool(selected),
                        }
                    )
            final_rows.append(
                {"sample_id": sample_id, "route": route, "final_correct": bool(post["final_correct"])}
            )
    formal_source = tmp_path / "formal.parquet"
    pd.DataFrame(formal_rows).to_parquet(formal_source, index=False)
    source_lock = tmp_path / "FINAL_RUN_LOCK.json"
    atomic_json(
        source_lock,
        {"status": "COMPLETE", "inventory": [artifact_record(formal_source), artifact_record(visual_source)]},
    )
    bootstrap_run(
        run,
        source_verification={"status": "PASS", "full_inventory_byte_rehash": True, "sources": {}},
    )
    transition_pipeline_status(run, RunState.P1_BASELINE_REPLAY_PASS, first_incomplete_stage=RunState.P2_GT_MAPPING_PASS.value)
    sample = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "scene_id": ["scene"] * 16,
            "frame_id": sample_ids,
            "language_prompt": [f"pick {sample_id}" for sample_id in sample_ids],
            "gt_grasp_rectangles": [
                [canonical_corners({"cx_px": 4.0, "cy_px": 4.0, "theta_deg": 0.0, "width_px": 3.0, "height_px": 2.0}).tolist()]
                for _ in sample_ids
            ],
        }
    )
    sample_path = atomic_parquet(sample, run / "02_sample_manifest/counterfactual_manifest.parquet")
    registry_path = atomic_parquet(
        pd.DataFrame(
            [
                {
                    "sample_id": sample_id,
                    "original_gt_mask_path": str(paths["gt_mask"]),
                    "original_gt_mask_sha256": artifact_record(paths["gt_mask"])["sha256"],
                    "original_height": 8,
                    "original_width": 8,
                    "mapping_status": "PASS",
                    "pixel_qa_status": "P2_MAPPING_QA_PASS",
                    "annotation_suspect": CATEGORIES[index // 2] == "annotation",
                }
                for index, sample_id in enumerate(sample_ids)
            ]
        ),
        run / "03_gt_mask_registry/gt_mask_registry.parquet",
    )
    mapping_qa = atomic_json(
        run / "03_gt_mask_registry/GT_MASK_MAPPING_AUDIT.json",
        {"status": "PASS", "stage": "P2_GT_MAPPING_PASS", "pixel_qa_status": "P2_MAPPING_QA_PASS", "mapping_qa_gt_mask_rows_read": 16, "candidate_generation_gt_mask_rows_read": 0},
    )
    transition_pipeline_status(run, RunState.P2_GT_MAPPING_PASS, first_incomplete_stage=RunState.P3_PROTOCOL_LOCKED.value)
    source_record = artifact_record(source_lock)
    code_record = artifact_record(formal_source)
    route_contracts = {
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
    lock = create_protocol_lock(
        run,
        bindings={
            "source_locks": {"synthetic": source_record},
            "source_code": {"synthetic": code_record},
            "configs": {"synthetic": code_record},
            "baseline_replay": source_record,
            "sample_manifest": artifact_record(sample_path),
            "gt_mask_registry": artifact_record(registry_path),
            "mapping_qa": artifact_record(mapping_qa),
            "route_contracts": inline_binding(route_contracts),
            "resize_rules": inline_binding({"binary": "nearest"}),
            "evaluator": artifact_record(EVALUATOR),
            "taxonomy": inline_binding({"classes": list(NATIVE_CLASSES)}),
            "statistics": inline_binding({"iterations": 10_000, "seed": 20260813}),
            "case_selection": inline_binding({"rule": "synthetic"}),
        },
        declaration={"gt_candidate_generation_authorized": True, "bulk_execution_max_count": 1, "mapping_qa_gt_mask_rows_read_before_lock": 16, "candidate_generation_gt_mask_rows_read_before_lock": 0, "routes": route_contracts},
        test_only_allow_synthetic_contract=True,
    )
    claim_bulk_execution(run)
    visual_manifest = write_visual_asset_registry(
        run,
        protocol_lock=lock,
        unified_samples=artifact_record(visual_source),
        d1_samples=artifact_record(visual_source),
        expected_count=16,
    )
    final_path = atomic_parquet(pd.DataFrame(final_rows), run / "04_predicted_replay/frozen_final_outcomes.parquet")
    authority = write_final_outcomes_authority(
        run,
        protocol_lock=lock,
        final_outcomes=artifact_record(final_path),
        selector_sources={route: artifact_record(formal_source) for route in systems},
    )
    inputs_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "LOCKED",
        "sample_count": 16,
        "available_routes": list(systems),
        "protocol_lock": artifact_record(lock),
        "artifacts": {
            "sample_manifest": artifact_record(sample_path),
            "ground_truth": artifact_record(sample_path),
            "visual_assets": artifact_record(visual_manifest),
            "final_outcomes": artifact_record(final_path),
            "final_outcomes_authority": artifact_record(authority),
        },
    }
    inputs_path = _content_json(run / "07_candidate_tables/POSTPROCESS_INPUTS.json", inputs_payload)
    labels_path = atomic_parquet(pd.DataFrame(labels), run / "07_candidate_tables/per_candidate_labels.parquet")
    post_path = atomic_parquet(pd.DataFrame(post_rows), run / "09_failure_taxonomy/post_r7_bottleneck_per_sample.parquet")
    output_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "routes": {route: "COMPLETE" for route in systems},
        "postprocess_inputs": artifact_record(inputs_path),
        "artifacts": {"candidate_labels": artifact_record(labels_path), "post_r7_taxonomy": artifact_record(post_path)},
    }
    _content_json(run / "08_metrics/POSTPROCESS_MANIFEST.json", output_payload)
    transitions = (
        (RunState.P4_G1_COUNTERFACTUAL_COMPLETE, RunState.P5_C1_COUNTERFACTUAL_COMPLETE),
        (RunState.P5_C1_COUNTERFACTUAL_COMPLETE, RunState.P6_D1_COUNTERFACTUAL_COMPLETE),
        (RunState.P6_D1_COUNTERFACTUAL_COMPLETE, RunState.P7_TAXONOMY_COMPLETE),
        (RunState.P7_TAXONOMY_COMPLETE, RunState.P8_STATISTICS_COMPLETE),
        (RunState.P8_STATISTICS_COMPLETE, RunState.P9_GALLERIES_COMPLETE),
    )
    for state, next_state in transitions:
        transition_pipeline_status(run, state, first_incomplete_stage=next_state.value)
    return run


def _signed_qa(run: Path, tmp_path: Path) -> Path:
    build = json.loads((run / "14_galleries/GALLERY_BUILD_MANIFEST.json").read_text())
    qa = pd.read_csv(run / "14_galleries/MANUAL_QA_TEMPLATE.csv", keep_default_na=False)
    qa["status"] = "PASS"
    qa["reviewer"] = "synthetic-reviewer"
    qa["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    qa["notes"] = "synthetic visual inspection passed"
    qa["signature_sha256"] = [
        manual_qa_signature(str(build["content_sha256"]), row)
        for row in qa.to_dict(orient="records")
    ]
    path = tmp_path / "manual.csv"
    qa.to_csv(path, index=False)
    return path


def test_gallery_prepare_accept_and_reject_selection_tamper(tmp_path: Path) -> None:
    run = _build_gallery_run(tmp_path)
    build_path = prepare_gallery(
        run,
        expected_sample_count=16,
        test_only_allow_synthetic_without_gate=True,
    )
    build = json.loads(build_path.read_text())
    assert build["status"] == "PENDING_MANUAL_QA"
    assert build["selected_count"] == 48
    assert not (run / "14_galleries/GALLERY_MANIFEST.json").exists()
    assert verify_gallery_build(run, expected_sample_count=16) == build

    qa_path = _signed_qa(run, tmp_path)
    final = accept_gallery_manual_qa(
        run, manual_qa_csv=qa_path, expected_sample_count=16
    )
    assert json.loads(final.read_text())["status"] == "COMPLETE"
    assert accept_gallery(run, expected_sample_count=16).is_file()
    assert json.loads((run / "pipeline_status.json").read_text())["status"] == (
        RunState.P9_GALLERIES_COMPLETE.value
    )

    selected_path = Path(str(build["selected"]["path"]))
    selected = pd.read_parquet(selected_path)
    selected.loc[0, "sample_id"] = "injected"
    selected.to_parquet(selected_path, index=False)
    with pytest.raises(GalleryContractError, match="artifact differs|canonical recompute"):
        verify_gallery_build(run, expected_sample_count=16)


def test_manual_qa_requires_exact_signed_coverage(tmp_path: Path) -> None:
    run = _build_gallery_run(tmp_path)
    prepare_gallery(
        run,
        expected_sample_count=16,
        test_only_allow_synthetic_without_gate=True,
    )
    qa_path = _signed_qa(run, tmp_path)
    qa = pd.read_csv(qa_path)
    qa.iloc[:-1].to_csv(qa_path, index=False)
    with pytest.raises(GalleryContractError, match="exactly cover"):
        accept_gallery_manual_qa(
            run, manual_qa_csv=qa_path, expected_sample_count=16
        )


def test_gallery_rejects_visual_asset_byte_tamper(tmp_path: Path) -> None:
    run = _build_gallery_run(tmp_path)
    prepare_gallery(
        run,
        expected_sample_count=16,
        test_only_allow_synthetic_without_gate=True,
    )
    registry = pd.read_parquet(
        run / "04_predicted_replay/VISUAL_ASSET_REGISTRY.parquet"
    )
    probability = Path(str(registry.iloc[0]["predicted_probability_path"]))
    probability.write_bytes(probability.read_bytes() + b"tamper")
    with pytest.raises(GalleryContractError, match="predicted probability asset hash differs"):
        verify_gallery_build(run, expected_sample_count=16)


def test_acceptance_rejects_legacy_caller_authored_gallery(tmp_path: Path) -> None:
    run = _build_gallery_run(tmp_path)
    with pytest.raises(GalleryContractError, match="gallery build manifest"):
        accept_gallery(run)
