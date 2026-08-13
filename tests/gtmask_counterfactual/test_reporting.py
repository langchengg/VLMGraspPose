from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import gtmask_counterfactual.finalize as finalize_module
from gtmask_counterfactual.figures import FIGURE_SPECS, render_all_figures
from gtmask_counterfactual.audit import initialize_counterfactual_ledger
from gtmask_counterfactual.finalize import assert_counterfactual_run, finalize_run
from gtmask_counterfactual.galleries import (
    BOARD_PANELS,
    build_eligible_table,
    deterministic_medoid_selection,
    render_case_board,
)
from gtmask_counterfactual.io import (
    artifact_record,
    atomic_json,
    canonical_sha256,
)
from gtmask_counterfactual.reporting import (
    REPORT_NAMES,
    TABLE_CONTRACTS,
    load_bound_tables,
    write_reports,
    write_table_bundle,
)


def _run(tmp_path: Path) -> Path:
    root = tmp_path / "runs" / "fair_gtmask_counterfactual_g1_c1_d1_20260813T000000Z"
    root.mkdir(parents=True)
    initialize_counterfactual_ledger(root / "run_ledger.sqlite")
    return root


def _tables(*, include_d1: bool = True) -> dict[str, pd.DataFrame]:
    routes = ["G1", "C1"] + (["D1"] if include_d1 else [])
    tables: dict[str, pd.DataFrame] = {
        "source_reconciliation.csv": pd.DataFrame(
            [
                {
                    "source_name": "source",
                    "path": "/tmp/source",
                    "sha256": "a" * 64,
                    "bytes": 1,
                    "status": "PASS",
                }
            ]
        ),
        "gt_mask_mapping_audit.csv": pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "route": "G1",
                    "gt_mask_sha256": "b" * 64,
                    "rgb_shape": "8x8",
                    "mask_shape": "8x8",
                    "status": "PASS",
                }
            ]
        ),
        "predicted_replay_metrics.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "N": 10,
                    "native_correct": 3,
                    "oracle_all": 6,
                    "no_output": 0,
                    "status": "PASS",
                }
                for route in routes
            ]
        ),
        "branch_metrics.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "branch": branch,
                    "N": 10,
                    "native_correct": 3 + index,
                    "oracle_at_5": 5 + index,
                    "oracle_at_10": 6 + index,
                    "oracle_all": 7 + index,
                    "no_output": 0,
                }
                for route in routes
                for index, branch in enumerate(("predicted", "gt_oracle"))
            ]
        ),
        "pred_vs_gt_paired_metrics.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "N": 10,
                    "pred_oracle_all": 6,
                    "gt_oracle_all": 7,
                    "delta_oracle_all": 0.1,
                    "pred_no_positive": 4,
                    "gt_no_positive": 3,
                    "grounding_recovered": 2,
                    "grounding_plus_selection": 0,
                    "generator_limited_under_gt": 3,
                    "gt_regression": 1,
                    "post_r7_residual_grounding_fraction": 3 / 7,
                    "post_r7_residual_generator_fraction": 2 / 7,
                }
                for route in routes
            ]
        ),
        "native_failure_taxonomy.csv": pd.DataFrame(
            [
                {"route": route, "taxonomy": taxonomy, "count": count, "N": 10}
                for route in routes
                for taxonomy, count in (
                    ("T1_deployed_native_success", 3),
                    ("T4_grounding_limited", 4),
                    ("T7_grasper_or_candidate_generation_limited", 3),
                )
            ]
        ),
        "post_r7_bottleneck_taxonomy.csv": pd.DataFrame(
            [
                {"route": route, "taxonomy": taxonomy, "count": count, "N": 10}
                for route in routes
                for taxonomy, count in (
                    ("R1_reranker_limited", 2),
                    ("R2_residual_grounding_limited", 3),
                    ("R5_residual_generator_limited_under_GT", 2),
                )
            ]
        ),
        "candidate_pool_transitions.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "transition_family": transition_family,
                    "transition": transition,
                    "count": count,
                    "N": 10,
                }
                for route in routes
                for transition_family, transitions in (
                    (
                        "all_oracle",
                        (
                            ("none_to_positive", 2),
                            ("positive_to_positive", 7),
                            ("positive_to_none", 1),
                        ),
                    ),
                    (
                        "candidate_count",
                        (
                            ("increased", 4),
                            ("unchanged", 5),
                            ("decreased", 1),
                        ),
                    ),
                )
                for transition, count in transitions
            ]
        ),
        "candidate_mechanism_summary.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "observable_mechanism": "gt_only_positive_candidate",
                    "upstream_attribution": "source_stage_unknown",
                    "candidate_relation_count": 2,
                    "affected_sample_count": 2,
                    "positive_transition_count": 2,
                    "evidence_scope": (
                        "observable_final_nms_pool_transition_only"
                    ),
                }
                for route in routes
            ]
        ),
        "first_positive_rank_transitions.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "pred_first_positive_rank": pred,
                    "gt_first_positive_rank": gt,
                    "count": count,
                }
                for route in routes
                for pred, gt, count in (("none", "1", 2), ("1", "1", 3), ("3", "2", 5))
            ]
        ),
        "stratified_results.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "stratum_name": name,
                    "stratum_value": value,
                    "N": 5,
                    "recovered": 2,
                    "harmful": 1,
                    "delta": 0.2,
                }
                for route in routes
                for name, value in (("mask_iou_bin", "low"), ("query_type", "spatial"))
            ]
        ),
        "statistical_tests.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "metric": "oracle_all",
                    "N": 10,
                    "delta": 0.1,
                    "ci_low": 0.02,
                    "ci_high": 0.18,
                    "raw_p": 0.1,
                    "holm_p": 0.3,
                }
                for route in routes
            ]
        ),
        "annotation_suspect_sensitivity.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "suspect_status": "included",
                    "N": 10,
                    "delta_oracle_all": 0.1,
                }
                for route in routes
            ]
        ),
        "frozen_selector_transfer.csv": pd.DataFrame(
            [
                {
                    "route": route,
                    "branch": "gt_oracle",
                    "N": 10,
                    "selector": "R7",
                    "correct": 5,
                    "oracle_all": 7,
                    "secondary_only": 1,
                }
                for route in routes
            ]
        ),
    }
    return tables


def _bound_tables(root: Path, *, include_d1: bool = True) -> Path:
    source = root / "00_audit" / "synthetic_source.json"
    atomic_json(source, {"synthetic": True})
    return write_table_bundle(
        root,
        _tables(include_d1=include_d1),
        source_bindings={"synthetic": artifact_record(source)},
    )


def _candidate(candidate_id: str, cx: float, native_rank: int = 1) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "native_rank": native_rank,
        "cx_px": cx,
        "cy_px": 4.0,
        "width_px": 3.0,
        "height_px": 1.5,
        "theta_deg": 10.0,
    }


def test_table_bundle_figures_and_reports_are_hash_bound(tmp_path: Path) -> None:
    root = _run(tmp_path)
    table_manifest = _bound_tables(root)
    tables, manifest = load_bound_tables(root)
    assert set(tables) == set(TABLE_CONTRACTS)
    assert manifest["table_count"] == len(TABLE_CONTRACTS)
    figure_manifest = render_all_figures(root, table_manifest)
    figure_value = json.loads(figure_manifest.read_text())
    assert set(figure_value["figures"]) == set(FIGURE_SPECS)
    assert all(
        set(value) == {"pdf", "svg", "png"}
        for value in figure_value["figures"].values()
    )
    report_manifest = write_reports(root, table_manifest)
    reports = json.loads(report_manifest.read_text())["reports"]
    assert set(reports) == set(REPORT_NAMES)
    assert "not proof" in (root / "15_reports" / REPORT_NAMES[0]).read_text().lower()
    table_path = Path(manifest["tables"]["branch_metrics.csv"]["path"])
    table_path.write_text(table_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        load_bound_tables(root)


def test_blocked_d1_omits_curve_and_reports_partial_scope(tmp_path: Path) -> None:
    root = _run(tmp_path)
    table_manifest = _bound_tables(root, include_d1=False)
    blocker = {
        "missing_evidence": "frozen scorer dependency",
        "search_paths": ["/repo"],
        "stack_trace": "RuntimeError: unavailable",
        "resume_command": "python -m tools.gtmask_counterfactual.run_route --route d1 --resume",
    }
    figures = json.loads(
        render_all_figures(
            root, table_manifest, allow_missing_d1_primary=True
        ).read_text()
    )
    assert figures["status"] == "PARTIAL"
    assert len(figures["figures"]) == 11
    assert not any(name.startswith("12_") for name in figures["figures"])
    reports = json.loads(
        write_reports(root, table_manifest, d1_blocker=blocker).read_text()
    )
    assert reports["status"] == "PARTIAL"
    report = (root / "15_reports" / REPORT_NAMES[0]).read_text()
    assert "D1 primary was not fabricated" in report


def test_deterministic_medoid_is_order_invariant_and_sha_breaks_tie() -> None:
    rows = []
    for sample, feature in (("a", [0.0]), ("b", [1.0]), ("c", [2.0])):
        rows.append(
            {
                "sample_id": sample,
                "route": "G1",
                "category": "clear_grounding_limited",
                "mechanism_pure": True,
                "presentation_eligible": True,
                "feature_vector_json": json.dumps(feature),
                "asset_bundle_sha256": hashlib.sha256(sample.encode()).hexdigest(),
            }
        )
    frame = pd.DataFrame(rows)
    first, _ = deterministic_medoid_selection(frame)
    second, _ = deterministic_medoid_selection(frame.sample(frac=1, random_state=3))
    assert first.iloc[0]["sample_id"] == "b"
    assert first.iloc[0]["sample_id"] == second.iloc[0]["sample_id"]
    assert len(build_eligible_table(frame)) == 3


def test_case_board_identity_and_all_panels(tmp_path: Path) -> None:
    shape = (8, 8)
    pred_all = [_candidate("p1", 3.0), _candidate("p2", 5.0, native_rank=2)]
    gt_all = [_candidate("g1", 4.0)]
    asset_hash = canonical_sha256(
        {
            "sample_id": "s1",
            "route": "G1",
            "pred_ids": ["p1", "p2"],
            "gt_ids": ["g1"],
            "shape": [8, 8],
        }
    )
    case = {
        "sample_id": "s1",
        "route": "G1",
        "category": "clear_grounding_limited",
        "language_prompt": "pick the block",
        "candidate_count_pred": 2,
        "candidate_count_gt": 1,
        "positive_count_pred": 0,
        "positive_count_gt": 1,
        "first_positive_rank_pred": "none",
        "first_positive_rank_gt": 1,
        "native_candidate_id": "p1",
        "r7_candidate_id": "p2",
        "gt_candidate_id": "g1",
        "native_q": 0.2,
        "rerank_score": 0.3,
        "rotated_iou": 0.5,
        "angle_error_deg": 5.0,
        "pass_fail": "PASS",
        "earliest_observable_issue": "visual grounding",
        "asset_bundle_sha256": asset_hash,
    }
    assets = {
        "rgb": np.zeros((*shape, 3), dtype=np.uint8),
        "gt_mask": np.ones(shape),
        "pred_probability": np.zeros(shape),
        "pred_mask": np.zeros(shape),
        "depth": np.ones(shape),
        "pred_all_candidates": pred_all,
        "pred_top_candidates": pred_all,
        "gt_all_candidates": gt_all,
        "gt_top_candidates": gt_all,
        "gt_grasps": gt_all,
        "qa_evidence": {
            "same_crop_pass": True,
            "same_gt_metrics_recompute_pass": True,
            "no_clipped_text_pass": True,
            "no_overlay_obscures_rectangles_pass": True,
            "route_branch_labels_pass": True,
        },
    }
    qa = render_case_board(
        case=case,
        assets=assets,
        output_png=tmp_path / "board.png",
        output_svg=tmp_path / "board.svg",
    )
    assert qa["required_panels"] == list(BOARD_PANELS)
    bad = dict(case)
    bad["r7_candidate_id"] = "absent"
    with pytest.raises(ValueError, match="absent"):
        render_case_board(
            case=bad,
            assets=assets,
            output_png=tmp_path / "bad.png",
            output_svg=tmp_path / "bad.svg",
        )


def _self_hashed(path: Path, value: dict[str, Any]) -> None:
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def _source_audit(root: Path) -> None:
    source = root / "00_audit" / "source.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    value = {
        "status": "PASS",
        "sources": {"source": artifact_record(source)},
        "formal_test_counts": {"unified": 1, "d1": 1},
        "inventory_rehashes": {
            "source": {
                "inventory_count": 1,
                "inventory_sha256": artifact_record(source)["sha256"],
            }
        },
    }
    _self_hashed(
        root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_BEFORE.json", dict(value)
    )
    _self_hashed(
        root / "00_audit" / "SOURCE_RUN_IMMUTABILITY_AFTER.json", dict(value)
    )


def _terminal_prerequisites(
    root: Path, *, d1_status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source_audit(root)
    _self_hashed(
        root / "04_predicted_replay" / "BASELINE_REPLAY_MANIFEST.json",
        {"status": "PASS", "sample_count": 7675},
    )
    _self_hashed(
        root / "03_gt_mask_registry" / "GT_MASK_MAPPING_AUDIT.json",
        {"status": "PASS", "sample_count": 7675},
    )
    protocol = {"status": "LOCKED", "counterfactual_execution_count": 0}
    protocol["self_sha256"] = canonical_sha256(protocol)
    protocol_path = root / "01_protocol_lock" / "COUNTERFACTUAL_PROTOCOL_LOCK.json"
    atomic_json(protocol_path, protocol)
    # Protocol construction is covered by test_core.  This finalizer fixture
    # isolates terminal graph validation from the much larger P1/P2 fixture.
    monkeypatch.setattr(finalize_module, "verify_protocol_lock", lambda _root: protocol)
    atomic_json(
        root / "01_protocol_lock" / "COUNTERFACTUAL_EXECUTION.json",
        {
            "status": "RUNNING",
            "execution_count": 1,
            "protocol_lock_file_sha256": artifact_record(protocol_path)["sha256"],
        },
    )
    atomic_json(
        root / "pipeline_status.json",
        {
            "status": (
                "P5_C1_COUNTERFACTUAL_COMPLETE"
                if d1_status == "UNRECOVERABLE_BLOCKER"
                else "P10_INDEPENDENT_RECOMPUTE_PASS"
            ),
            "counterfactual_execution_count": 1,
        },
    )
    atomic_json(
        root / "manifest.json",
        {
            "counterfactual_execution_count": 1,
            "formal_test_execution_count": 0,
            "source_formal_test_execution_modified": False,
        },
    )
    postprocess_path = root / "08_metrics" / "POSTPROCESS_MANIFEST.json"
    _self_hashed(postprocess_path, {"status": "COMPLETE"})
    route_status_path = root / "08_metrics" / "ROUTE_STATUS.json"
    _self_hashed(
        route_status_path,
        {
            "status": "PARTIAL" if d1_status == "UNRECOVERABLE_BLOCKER" else "COMPLETE",
            "routes": {"G1": "COMPLETE", "C1": "COMPLETE", "D1": d1_status},
            "protocol_lock": artifact_record(protocol_path),
            "artifacts": {"postprocess_manifest": artifact_record(postprocess_path)},
        },
    )
    claim_path = root / "01_protocol_lock" / "COUNTERFACTUAL_EXECUTION.json"
    completion = {
        "status": "COMPLETE",
        "execution_count": 1,
        "execution_claim": artifact_record(claim_path),
        "protocol_lock": artifact_record(protocol_path),
        "route_status": artifact_record(route_status_path),
        "postprocess_manifest": artifact_record(postprocess_path),
    }
    _self_hashed(
        root / "01_protocol_lock" / "COUNTERFACTUAL_EXECUTION_COMPLETE.json",
        completion,
    )
    _bound_tables(root, include_d1=d1_status != "UNRECOVERABLE_BLOCKER")
    table_manifest = json.loads(
        (root / "08_metrics" / "TABLE_BUNDLE_MANIFEST.json").read_text()
    )
    figures: dict[str, Any] = {}
    figure_count = 11 if d1_status == "UNRECOVERABLE_BLOCKER" else 12
    for index in range(1, figure_count + 1):
        formats = {}
        for suffix in ("pdf", "svg", "png"):
            path = root / "13_figures" / f"{index:02d}_synthetic.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"figure")
            formats[suffix] = artifact_record(path)
        figures[f"{index:02d}_synthetic"] = formats
    figure_manifest = {
        "status": "PARTIAL" if d1_status == "UNRECOVERABLE_BLOCKER" else "COMPLETE",
        "palette": "Okabe-Ito",
        "figures": figures,
    }
    _self_hashed(root / "13_figures" / "FIGURES_MANIFEST.json", figure_manifest)
    reports = {}
    for name in REPORT_NAMES:
        path = root / "15_reports" / name
        if name.endswith(".json"):
            atomic_json(path, {"synthetic": True})
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic", encoding="utf-8")
        reports[name] = artifact_record(path)
    report_manifest = {
        "status": "PARTIAL" if d1_status == "UNRECOVERABLE_BLOCKER" else "COMPLETE",
        "reports": reports,
        "table_bundle_content_sha256": table_manifest["content_sha256"],
    }
    _self_hashed(root / "15_reports" / "REPORTS_MANIFEST.json", report_manifest)
    eligible = root / "14_galleries" / "eligible_cases.csv"
    selected = root / "14_galleries" / "selected_cases.csv"
    eligible.parent.mkdir(parents=True, exist_ok=True)
    eligible.write_text("sample_id\ns1\n")
    selected.write_text("sample_id\ns1\n")
    board_records = {}
    for suffix in ("png", "svg"):
        path = root / "14_galleries" / f"board.{suffix}"
        path.write_bytes(b"board")
        board_records[suffix] = artifact_record(path)
    gallery = {
        "status": "COMPLETE",
        "manual_qa_status": "PASS",
        "manual_qa_coverage_pass": True,
        "selection_rule": "mechanism-purity + presentation eligibility + cluster medoid + SHA256 tie",
        "postprocess_manifest": artifact_record(postprocess_path),
        "eligible": artifact_record(eligible),
        "selected": artifact_record(selected),
        "boards": [{"status": "AUTO_QA_PASS", **board_records}],
    }
    gallery_path = root / "14_galleries" / "GALLERY_MANIFEST.json"
    _self_hashed(gallery_path, gallery)
    monkeypatch.setattr(
        finalize_module,
        "verify_complete_gallery",
        lambda _root: json.loads(gallery_path.read_text(encoding="utf-8")),
    )
    gallery_acceptance_path = (
        root / "12_case_selection" / "P9_GALLERY_ACCEPTANCE.json"
    )
    _self_hashed(
        gallery_acceptance_path,
        {
            "status": "PASS",
            "gallery_manifest": artifact_record(gallery_path),
            "postprocess_manifest": artifact_record(postprocess_path),
        },
    )
    candidate_geometry = root / "16_independent_recompute" / "candidate_geometry.parquet"
    candidate_geometry.parent.mkdir(parents=True, exist_ok=True)
    candidate_geometry.write_bytes(b"geometry")
    _self_hashed(
        root / "16_independent_recompute" / "recomputed_metrics.json",
        {
            "status": "PASS",
            "per_sample_exact_match": True,
            "metrics_exact_match": True,
            "taxonomy_exact_match": True,
            "paired_inputs_exact_match": True,
            "source_candidate_geometry": artifact_record(candidate_geometry),
        },
    )
    _self_hashed(
        root / "16_independent_recompute" / "INDEPENDENT_VALIDATION.json",
        {
            "status": "PASS",
            "process_role": "standalone saved-frame independent recompute",
            "forbidden_modules_imported": False,
            "postprocess_manifest": artifact_record(postprocess_path),
            "gallery_acceptance": artifact_record(gallery_acceptance_path),
            "source_candidate_geometry": artifact_record(candidate_geometry),
            "per_sample_exact_match": True,
            "metrics_exact_match": True,
            "taxonomy_exact_match": True,
            "paired_inputs_exact_match": True,
        },
    )


def test_d1_blocker_is_partial_and_never_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _run(tmp_path)
    _terminal_prerequisites(
        root, d1_status="UNRECOVERABLE_BLOCKER", monkeypatch=monkeypatch
    )
    blocker = {
        "status": "UNRECOVERABLE_BLOCKER",
        "blocker_class": "IRRECOVERABLE_FROZEN_SOURCE_EVIDENCE",
        "raw_candidate_regeneration_required": True,
        "missing_evidence": "frozen scorer dependency",
        "search_paths": ["/repo"],
        "stack_trace": "RuntimeError: unavailable",
        "resume_command": "python -m tools.gtmask_counterfactual.run_route --route d1 --resume",
        "filter_only_sensitivity_used_as_primary": False,
    }
    result = finalize_run(root, d1_blocker=blocker)
    assert result["status"] == "PARTIAL"
    assert (root / "PARTIAL").is_file()
    assert not (root / "COMPLETE").exists()


def test_full_three_route_run_is_only_path_to_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _run(tmp_path)
    _terminal_prerequisites(root, d1_status="COMPLETE", monkeypatch=monkeypatch)
    result = finalize_run(root)
    assert result["status"] == "COMPLETE"
    assert (root / "COMPLETE").is_file()
    assert not (root / "PARTIAL").exists()


def test_terminal_sidecars_are_repaired_after_post_lock_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _run(tmp_path)
    _terminal_prerequisites(root, d1_status="COMPLETE", monkeypatch=monkeypatch)
    original = finalize_module.exclusive_text

    def fail_digest(path: str | Path, value: str) -> Path:
        if Path(path).name == finalize_module.FINAL_LOCK_DIGEST_NAME:
            raise OSError("synthetic crash after final lock publication")
        return original(path, value)

    monkeypatch.setattr(finalize_module, "exclusive_text", fail_digest)
    with pytest.raises(OSError, match="synthetic crash"):
        finalize_run(root)
    assert (root / finalize_module.FINAL_LOCK_NAME).is_file()
    assert not (root / finalize_module.FINAL_LOCK_DIGEST_NAME).exists()

    monkeypatch.setattr(finalize_module, "exclusive_text", original)
    repaired = finalize_run(root)
    assert repaired["status"] == "COMPLETE"
    assert repaired["repaired"] is True
    assert (root / "COMPLETE").is_file()


def test_cli_namespace_rejects_formal_source_runs(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "fair_unified_reranking_20260809_103012"
    source.mkdir(parents=True)
    with pytest.raises(PermissionError, match="restricted"):
        assert_counterfactual_run(source)
