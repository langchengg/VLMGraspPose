from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from reranking.matrix_reporting import (
    ReportingContractError,
    compute_reporting_statistics,
    load_query_outcomes,
    load_reporting_inputs,
    multi_seed_summary,
)
from reranking.report import (
    CONSISTENCY_EN,
    CONSISTENCY_ZH,
    REPORT_NAMES,
    check_required_artifacts,
    run_reporting_stage,
)
from reranking.visualize import (
    FIGURE_NAMES,
    _inference_pool_edgecolor,
    build_galleries,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _outcomes(experiment_id: str, *, post: bool = False) -> list[dict]:
    # Includes every required gallery class; rank>5 is an additional label.
    values = [
        ("empty", "s0", 0, 0, 0, 0, 0, 0),
        ("no_pos", "s0", 0, 0, 0, 3, 0, 0),
        ("recovered", "s1", 0, 1, 1, 5, 2, 2),
        ("harmful", "s1", 1, 0, 1, 5, 2, 1),
        ("unchanged", "s2", 1, 1, 1, 5, 2, 1),
        ("bothwrong", "s2", 0, 0, 1, 5, 1, 3),
        ("rank_gt5", "s3", 0, 1, 1, 8, 1, 7),
        ("neutral", "s3", 0, 0, 1, 5, 1, 2),
    ]
    rows = []
    for index, (sample, scene, before, after, oracle, count, positives, first_rank) in enumerate(values):
        if post and sample == "harmful":
            after = 1
        rows.append(
            {
                "experiment_id": experiment_id,
                "sample_id": sample,
                "scene_id": scene,
                "frame_id": f"f{index // 2}",
                "baseline_correct": before,
                "selected_correct": after,
                "oracle": oracle,
                "baseline_candidate_id": "a",
                "selected_candidate_id": "a" if before == after else "b",
                "candidate_count": count,
                "positive_count": positives,
                "first_positive_rank": first_rank,
                "confidence": 0.15 + 0.1 * index,
                "q_drop": 0.01 * index,
            }
        )
    return rows


def _stage(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    predictions = stage / "predictions"
    predictions.mkdir(parents=True)
    for experiment_id, post in (("locked_primary", False), ("post_lock_better", True)):
        path = predictions / f"{experiment_id}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in _outcomes(experiment_id, post=post)),
            encoding="utf-8",
        )
    validation = []
    for seed, score in ((42, 0.70), (123, 0.72), (2026, 0.71)):
        validation.append(
            {
                "experiment_id": f"locked_primary_seed{seed}",
                "experiment_family": "locked_primary",
                "method": "gnn_ranknet",
                "configuration": "full_features",
                "route": "modular",
                "pool": "top5",
                "seed": seed,
                "j_at_1": score,
                "delta_j_at_1": score - 0.68,
                "runtime_ms": 4.5 + seed % 3,
                "category": "reference",
            }
        )
    validation.append(
        {
            "experiment_id": "without_mask",
            "experiment_family": "without_mask",
            "method": "gnn_ranknet",
            "configuration": "without_mask",
            "route": "modular",
            "pool": "top5",
            "seed": 42,
            "j_at_1": 0.69,
            "delta_j_at_1": 0.01,
            "runtime_ms": 4.0,
            "category": "feature_ablation",
        }
    )
    test = [
        {
            "experiment_id": "post_lock_better",
            "method": "post_lock_better",
            "j_at_1": 0.99,
            "prediction_path": "predictions/post_lock_better.jsonl",
            "status": "post_lock_comparison",
        },
        {
            "experiment_id": "locked_primary",
            "method": "locked_primary",
            "j_at_1": 0.75,
            "prediction_path": "predictions/locked_primary.jsonl",
            "status": "locked_primary",
        },
    ]
    _write_json(stage / "metrics/validation_registry.json", {"records": validation})
    _write_json(stage / "metrics/test_registry.json", {"records": test})
    _write_json(
        stage / "manifests/PRIMARY_METHOD_LOCK.json",
        {
            "primary_experiment_id": "locked_primary",
            "baseline_name": "q_only",
            "locked_before_test": True,
            "test_used_for_selection": False,
        },
    )
    return stage


def test_lock_selects_primary_without_using_better_test_row(tmp_path: Path) -> None:
    inputs = load_reporting_inputs(_stage(tmp_path))
    assert inputs.primary_experiment_id == "locked_primary"
    assert inputs.test_registry.iloc[0]["experiment_id"] == "post_lock_better"
    statistics = compute_reporting_statistics(inputs, bootstrap_iterations=100, seed=7)
    assert statistics["test_used_for_selection"] is False
    assert statistics["primary_summary"]["recovered"] == 2
    assert statistics["primary_summary"]["harmful"] == 1
    assert statistics["primary_summary"]["net_recovered"] == 1
    assert len(statistics["holm_rows"]) == 2


def test_exact_statistics_use_scene_clusters_and_holm(tmp_path: Path) -> None:
    inputs = load_reporting_inputs(_stage(tmp_path))
    result = compute_reporting_statistics(inputs, bootstrap_iterations=111, seed=9)
    primary_bootstrap = next(
        row for row in result["bootstrap_rows"] if row["experiment_id"] == "locked_primary"
    )
    assert primary_bootstrap["unit"] == "scene"
    assert primary_bootstrap["iterations"] == 111
    assert primary_bootstrap["cluster_count"] == 4
    primary_mcnemar = next(
        row for row in result["mcnemar_rows"] if row["experiment_id"] == "locked_primary"
    )
    assert primary_mcnemar["scipy_statsmodels_cross_check"] is True
    for raw, adjusted in zip(result["mcnemar_rows"], result["holm_rows"], strict=True):
        assert adjusted["holm_adjusted_p"] >= raw["raw_p"]


def test_multi_seed_summary_uses_validation_rows() -> None:
    frame = pd.DataFrame(
        {
            "method": ["m", "m", "m"],
            "seed": [1, 2, 3],
            "j_at_1": [0.5, 0.7, 0.6],
        }
    )
    summary = multi_seed_summary(frame)
    assert summary.iloc[0]["seed_count"] == 3
    assert summary.iloc[0]["j_at_1_mean"] == pytest.approx(0.6)
    assert summary.iloc[0]["j_at_1_min"] == 0.5


def test_query_outcomes_reject_duplicate_ids_and_oracle_violations(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    rows = _outcomes("x")
    duplicate.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n", encoding="utf-8")
    with pytest.raises(ReportingContractError, match="duplicate"):
        load_query_outcomes(duplicate, expected_experiment_id="x")
    invalid = tmp_path / "invalid.jsonl"
    row = rows[0] | {"selected_correct": 1, "oracle": 0}
    invalid.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ReportingContractError, match="exceeds"):
        load_query_outcomes(invalid, expected_experiment_id="x")


def test_reporting_stage_writes_required_artifacts_and_no_success_marker(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    output = tmp_path / "reporting"
    manifest = run_reporting_stage(stage, output)
    assert manifest["primary_experiment_id"] == "locked_primary"
    assert manifest["test_used_for_primary_selection"] is False
    assert manifest["bootstrap_iterations"] == 10_000
    assert manifest["run_success_marker_written"] is False
    assert not list(output.rglob("_SUCCESS*"))
    assert check_required_artifacts(output)["missing"] == []
    assert (output / "reporting_manifest.json").is_file()


def test_all_required_png_pdf_reports_and_tables_are_nonempty(tmp_path: Path) -> None:
    output = tmp_path / "reporting"
    run_reporting_stage(_stage(tmp_path), output)
    for name in FIGURE_NAMES:
        for suffix in ("png", "pdf"):
            path = output / "figures" / f"{name}.{suffix}"
            assert path.stat().st_size > 100
    for name in REPORT_NAMES:
        assert (output / "reports" / name).stat().st_size > 100
    tables = (output / "reports/DISSERTATION_TABLES.md").read_text(encoding="utf-8")
    assert tables.count("## Table ") == 10


def test_reports_contain_2d_consistency_scope_and_file_derived_counts(tmp_path: Path) -> None:
    output = tmp_path / "reporting"
    run_reporting_stage(_stage(tmp_path), output)
    zh = (output / "reports/FINAL_REPORT_ZH.md").read_text(encoding="utf-8")
    en = (output / "reports/FINAL_REPORT_EN.md").read_text(encoding="utf-8")
    limitations = (output / "reports/LIMITATIONS.md").read_text(encoding="utf-8")
    assert CONSISTENCY_ZH in zh
    assert CONSISTENCY_EN in en
    assert CONSISTENCY_EN in limitations
    assert "Exact McNemar recovered/harmful：2/1" in zh
    assert "test registry 未用于选择" in zh


def test_gallery_lists_real_shortfalls_without_fabricating_images(tmp_path: Path) -> None:
    outcomes = pd.DataFrame(_outcomes("locked_primary"))
    gallery = build_galleries(tmp_path / "gallery", outcomes, quota_per_category=25)
    assert gallery["categories"]["recovered"]["eligible"] == 2
    assert gallery["categories"]["recovered"]["materialized"] == 0
    assert gallery["categories"]["recovered"]["shortfall"] == 23
    assert gallery["categories"]["rank>5"]["eligible"] == 1
    html = (tmp_path / "gallery/index.html").read_text(encoding="utf-8")
    assert "material unavailable" in html
    assert not list((tmp_path / "gallery").rglob("*.png"))


def test_gallery_copies_only_an_existing_declared_asset(tmp_path: Path) -> None:
    asset = tmp_path / "real.png"
    asset.write_bytes(b"real image material")
    outcomes = pd.DataFrame(_outcomes("locked_primary"))
    outcomes.loc[outcomes["sample_id"].eq("recovered"), "image_path"] = str(asset)
    gallery = build_galleries(tmp_path / "gallery", outcomes, quota_per_category=3)
    assert gallery["categories"]["recovered"]["materialized"] == 1
    copied = list((tmp_path / "gallery/recovered").glob("*.png"))
    assert len(copied) == 1
    assert copied[0].read_bytes() == asset.read_bytes()


def test_gallery_treats_namespaced_sample_id_as_identity_not_path(tmp_path: Path) -> None:
    asset = tmp_path / "real.png"
    asset.write_bytes(b"real image material")
    outcomes = pd.DataFrame(_outcomes("locked_primary"))
    selected = outcomes["sample_id"].eq("recovered")
    outcomes.loc[selected, "sample_id"] = "crog_full/recovered"
    outcomes.loc[selected, "image_path"] = str(asset)
    build_galleries(tmp_path / "gallery", outcomes, quota_per_category=3)
    copied = list((tmp_path / "gallery/recovered").glob("*.png"))
    assert len(copied) == 1
    assert copied[0].parent == tmp_path / "gallery/recovered"
    assert "crog_full" not in copied[0].name


def test_gallery_renders_audited_six_panel_case_from_frozen_pool(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (64, 48), (80, 100, 120)).save(rgb)
    Image.new("I;16", (64, 48), 1000).save(depth)
    Image.new("L", (64, 48), 255).save(mask)
    row = _outcomes("locked_primary")[2]
    row.update(
        {
            "dataset": "synthetic_full",
            "image_path": str(rgb),
            "depth_path": str(depth),
            "predicted_mask_path": str(mask),
            "language_instruction": "grasp the synthetic object",
            "gate_id": "G1",
            "switch_applied": True,
            "failure_stage": "F6",
            "gt_grasps_json": json.dumps([[[20, 18], [42, 18], [42, 30], [20, 30]]]),
            "feature_delta_json": json.dumps(
                [{"feature": "p_center", "baseline": 0.2, "selected": 0.8, "delta": 0.6}]
            ),
            "candidate_pool_json": json.dumps(
                [
                    {
                        "candidate_id": "a",
                        "q": 0.9,
                        "original_rank": 1,
                        "rerank_score": 0.1,
                        "rerank_rank": 2,
                        "correct": False,
                        "x_px": 22,
                        "y_px": 24,
                        "angle_rad": 0.0,
                        "width_px": 22,
                        "height_px": 10,
                    },
                    {
                        "candidate_id": "b",
                        "q": 0.7,
                        "original_rank": 2,
                        "rerank_score": 0.8,
                        "rerank_rank": 1,
                        "correct": True,
                        "x_px": 34,
                        "y_px": 24,
                        "angle_rad": 0.2,
                        "width_px": 20,
                        "height_px": 10,
                    },
                ]
            ),
        }
    )
    result = build_galleries(
        tmp_path / "gallery", pd.DataFrame([row]), quota_per_category=1
    )
    assert result["categories"]["recovered"]["materialized"] == 1
    rendered = list((tmp_path / "gallery/recovered/synthetic_full").glob("*.png"))
    assert len(rendered) == 1
    with Image.open(rendered[0]) as image:
        assert image.width > 1000
        assert image.height > 600


def test_inference_candidate_pool_style_does_not_encode_gt_correctness() -> None:
    assert _inference_pool_edgecolor({"correct": True}) == _inference_pool_edgecolor(
        {"correct": False}
    )


def test_missing_lock_or_registry_is_rejected_before_output(tmp_path: Path) -> None:
    stage = tmp_path / "empty"
    stage.mkdir()
    output = tmp_path / "output"
    with pytest.raises(ReportingContractError, match="primary lock"):
        run_reporting_stage(stage, output)
    assert not output.exists()


def test_reporting_output_is_immutable_and_success_marker_is_forbidden(tmp_path: Path) -> None:
    output = tmp_path / "reporting"
    stage = _stage(tmp_path)
    run_reporting_stage(stage, output)
    with pytest.raises(FileExistsError, match="must be new"):
        run_reporting_stage(stage, output)
    (output / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReportingContractError, match="forbidden"):
        check_required_artifacts(output)
