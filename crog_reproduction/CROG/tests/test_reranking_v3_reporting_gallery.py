from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from failure_analysis.reranking_v3 import galleries
from failure_analysis.reranking_v3.galleries import (
    GALLERY_GROUPS,
    build_v3_galleries,
    build_v3_galleries_strict,
    classify_gallery_groups,
    render_case,
)
from failure_analysis.reranking_v3.plotting import build_statistical_figures, plot_ablation
from failure_analysis.reranking_v3.reporting import (
    CLAIM_NEGATIVE,
    CLAIM_NOT_SIGNIFICANT,
    CLAIM_Q_ONLY,
    CLAIM_RELIABLE_V2,
    MACHINE_RESULT_FILES,
    build_machine_results,
    build_report_artifacts,
    derive_conclusion,
)
from failure_analysis.reranking_v3.report_validation import FORMAL_METHODS, FORMAL_TRACKS
from failure_analysis.reranking_v3.schema import artifact_identity


def _conclusion(**overrides):
    value = {
        "corrected_delta_vs_v2_pp": 0.2,
        "corrected_delta_vs_q_pp": 1.0,
        "frame_bootstrap_ci": [0.01, 0.4],
        "scene_bootstrap_ci": [0.02, 0.5],
        "mcnemar_holm_p": 0.04,
        "statistically_reliable_vs_q": True,
        "recovered_vs_v2": 7,
        "harmful_vs_v2": 3,
    }
    value.update(overrides)
    return value


def _machine_inputs():
    return {
        "validation_rows": [{"method": "q_only", "j_at_1": 0.5}, {"method": "v3_primary", "j_at_1": 0.6, "extra": "kept"}],
        "lockcheck_rows": [],
        "test_rows": [],
        "pairwise_rows": [],
        "calibration_rows": [],
        "feature_ablation_rows": [],
        "subgroup_rows": [],
        "feature_provenance": {"features": []},
        "conclusion_payload": _conclusion(),
        "commands": ["python -m synthetic"],
        "environment": {"python": "synthetic"},
        "tests": {"passed": 1, "failed": 0},
    }


def _formal_rows():
    scores = {
        "corrected_scientific": {
            "q_only": 0.500,
            "v2_locked_primary": 0.506,
            "v3_full_head_scalar_gate": 0.507,
            "v3_fcer_native": 0.507,
            "v3_fcer_rgbd": 0.506,
            "v3_locked_primary": 0.510,
        },
        "legacy_official_compatibility": {
            "q_only": 0.490,
            "v2_locked_primary": 0.505,
            "v3_full_head_scalar_gate": 0.506,
            "v3_fcer_native": 0.504,
            "v3_fcer_rgbd": 0.503,
            "v3_locked_primary": 0.507,
        },
    }
    rows = []
    for track in FORMAL_TRACKS:
        for method in FORMAL_METHODS:
            row = {"track": track, "method": method, "j_at_1": scores[track][method], "sample_count": 1000}
            if method == "v3_locked_primary":
                row |= {
                    "delta_vs_q": scores[track][method] - scores[track]["q_only"],
                    "delta_vs_v2": scores[track][method] - scores[track]["v2_locked_primary"],
                    "headroom_recovered_vs_v2": 0.25,
                }
            rows.append(row)
    return rows


def _pairwise_rows():
    return [
        {"track": "corrected_scientific", "method": "v3_locked_primary", "reference": "q_only", "effect": 0.010, "recovered": 15, "harmful": 5, "raw_p": 0.01, "holm_adjusted_p": 0.02, "frame_ci_lower": 0.002, "frame_ci_upper": 0.018, "scene_ci_lower": 0.001, "scene_ci_upper": 0.019, "sample_count": 1000},
        {"track": "corrected_scientific", "method": "v3_locked_primary", "reference": "v2_locked_primary", "effect": 0.004, "recovered": 7, "harmful": 3, "raw_p": 0.02, "holm_adjusted_p": 0.04, "frame_ci_lower": 0.001, "frame_ci_upper": 0.008, "scene_ci_lower": 0.0005, "scene_ci_upper": 0.009, "sample_count": 1000},
        {"track": "legacy_official_compatibility", "method": "v3_locked_primary", "reference": "q_only", "effect": 0.017, "recovered": 20, "harmful": 3, "raw_p": 0.01, "holm_adjusted_p": 0.02, "frame_ci_lower": 0.005, "frame_ci_upper": 0.025, "scene_ci_lower": 0.004, "scene_ci_upper": 0.026, "sample_count": 1000},
        {"track": "legacy_official_compatibility", "method": "v3_locked_primary", "reference": "v2_locked_primary", "effect": 0.002, "recovered": 7, "harmful": 5, "raw_p": 0.3, "holm_adjusted_p": 0.5, "frame_ci_lower": -0.003, "frame_ci_upper": 0.007, "scene_ci_lower": -0.004, "scene_ci_upper": 0.008, "sample_count": 1000},
    ]


def _strict_machine_inputs():
    rows = _formal_rows()
    return {
        "validation_rows": [{"method": "v3_locked_primary", "j_at_1": 0.51}],
        "lockcheck_rows": rows,
        "test_rows": [dict(row) for row in rows],
        "pairwise_rows": _pairwise_rows(),
        "calibration_rows": [{"method": "v3_locked_primary", "ece": 0.1}],
        "feature_ablation_rows": [{"feature_group": "G10", "delta_j_at_1": 0.002}],
        "subgroup_rows": [{"subgroup": "synthetic", "sample_count": 1000}],
        "feature_provenance": {"features": ["synthetic"]},
        "conclusion_payload": {"corrected_delta_vs_v2_pp": 0.4},
        "commands": ["python -m synthetic"],
        "environment": {"python": "synthetic"},
        "tests": {"passed": 1, "failed": 0},
    }


def _report_evidence(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text('{"synthetic":true}\n')
    independent_result = tmp_path / "independent-result.json"
    independent_result.write_text('{"status":"passed"}\n')
    completion = tmp_path / "independent-complete.json"
    completion.write_text(json.dumps({
        "kind": "independent_evaluate_run_complete",
        "status": "complete",
        "stage": "independent_evaluate",
        "result_artifacts": [artifact_identity(independent_result)],
    }))
    completion_identity = artifact_identity(completion)
    gallery = tmp_path / "gallery.json"
    gallery.write_text(json.dumps({
        "kind": "v3_failure_galleries",
        "status": "complete",
        "independent_evaluation_completion": completion_identity,
        "inputs": {"synthetic": artifact_identity(source)},
    }))
    evidence = {
        "independent_evaluation_completion": completion,
        "gallery": gallery,
    }
    for name in ("diagnostic", "efficiency", "subgroup"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"kind": name, "status": "complete"}))
        evidence[name] = path
    return evidence, completion


def test_conclusion_rule_has_all_four_predeclared_outcomes() -> None:
    reliable = derive_conclusion(_conclusion())
    assert reliable["final_claim"] == CLAIM_RELIABLE_V2
    assert reliable["conclusion_category"] == "A_reliable_beyond_v2"
    assert reliable["net_vs_v2"] == 4

    q_only = derive_conclusion(_conclusion(scene_bootstrap_ci=[-0.01, 0.5]))
    assert q_only["final_claim"] == CLAIM_Q_ONLY
    assert q_only["statistically_reliable_vs_v2"] is False

    nonsignificant = derive_conclusion(
        _conclusion(corrected_delta_vs_q_pp=0.0, corrected_delta_vs_v2_pp=0.0, frame_bootstrap_ci=[-0.1, 0.1])
    )
    assert nonsignificant["final_claim"] == CLAIM_NOT_SIGNIFICANT

    negative = derive_conclusion(_conclusion(corrected_delta_vs_v2_pp=-0.01, recovered_vs_v2=2, harmful_vs_v2=3))
    assert negative["final_claim"] == CLAIM_NEGATIVE
    assert negative["net_vs_v2"] == -1


@pytest.mark.parametrize(
    "updates",
    [
        {"frame_bootstrap_ci": [0.2]},
        {"scene_bootstrap_ci": [0.3, 0.2]},
        {"mcnemar_holm_p": 1.2},
        {"corrected_delta_vs_v2_pp": float("nan")},
        {"harmful_vs_v2": -1},
    ],
)
def test_conclusion_rejects_invalid_statistical_inputs(updates) -> None:
    with pytest.raises(ValueError):
        derive_conclusion(_conclusion(**updates))


def test_machine_results_are_complete_immutable_and_keep_union_schema(tmp_path: Path) -> None:
    output = tmp_path / "machine"
    result = build_machine_results(output, **_machine_inputs())
    assert result["status"] == "complete"
    assert {path.name for path in output.iterdir()} == {*MACHINE_RESULT_FILES, "machine_results_manifest.json"}
    with (output / "results_validation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["extra"] == "" and rows[1]["extra"] == "kept"
    assert (output / "results_lockcheck.csv").read_bytes() == b""
    conclusion = json.loads((output / "conclusion.json").read_text())
    assert conclusion["statistically_reliable_vs_v2"] is True
    with pytest.raises(FileExistsError):
        build_machine_results(output, **_machine_inputs())


def test_machine_results_validate_before_writing(tmp_path: Path) -> None:
    inputs = _machine_inputs()
    inputs["validation_rows"] = [{"value": float("nan")}]
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        build_machine_results(output, **inputs)
    assert not output.exists()


def test_report_backend_writes_required_report_without_label_io(tmp_path: Path) -> None:
    output = tmp_path / "report"
    evidence, _ = _report_evidence(tmp_path)
    result = build_report_artifacts(
        output,
        audit={"conclusion": "synthetic audit"},
        selection={"selected": {"configuration": "synthetic_v3"}, "results": []},
        lockcheck={},
        formal={},
        machine_inputs=_strict_machine_inputs(),
        evidence_artifacts=evidence,
    )
    assert result["labels_read"] is False
    markdown = (output / "RESULTS.md").read_text()
    assert "## V2 audit conclusion" in markdown
    assert "## Final scientific conclusion" in markdown
    assert result["conclusion_source"] == "derived_and_cross_checked_from_machine_tables"
    assert set(result["evidence_artifacts"]) == {"independent_evaluation_completion", "diagnostic", "efficiency", "subgroup", "gallery"}
    conclusion = json.loads((output / "conclusion.json").read_text())
    assert conclusion["corrected_delta_vs_v2_pp"] == pytest.approx(0.4)


@pytest.mark.parametrize("table", ["validation_rows", "lockcheck_rows", "test_rows", "pairwise_rows", "calibration_rows", "feature_ablation_rows", "subgroup_rows"])
def test_report_fails_closed_on_empty_machine_table(tmp_path: Path, table: str) -> None:
    machine = _strict_machine_inputs()
    machine[table] = []
    evidence, _ = _report_evidence(tmp_path)
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="non-empty|six formal|lacks"):
        build_report_artifacts(output, audit={}, selection={}, lockcheck={}, formal={}, machine_inputs=machine, evidence_artifacts=evidence)
    assert not output.exists()


def test_report_rejects_machine_contradiction_and_unbound_gallery(tmp_path: Path) -> None:
    machine = _strict_machine_inputs()
    machine["conclusion_payload"]["corrected_delta_vs_v2_pp"] = 9.0
    evidence, _ = _report_evidence(tmp_path)
    with pytest.raises(ValueError, match="contradicts"):
        build_report_artifacts(tmp_path / "bad-conclusion", audit={}, selection={}, lockcheck={}, formal={}, machine_inputs=machine, evidence_artifacts=evidence)
    gallery = Path(evidence["gallery"])
    value = json.loads(gallery.read_text())
    value["independent_evaluation_completion"]["sha256"] = "0" * 64
    gallery.write_text(json.dumps(value))
    machine = _strict_machine_inputs()
    with pytest.raises(ValueError, match="not bound"):
        build_report_artifacts(tmp_path / "bad-gallery", audit={}, selection={}, lockcheck={}, formal={}, machine_inputs=machine, evidence_artifacts=evidence)
    assert not (tmp_path / "bad-conclusion").exists()
    assert not (tmp_path / "bad-gallery").exists()


def test_report_rejects_contradictory_pairwise_machine_table_before_writing(tmp_path: Path) -> None:
    machine = _strict_machine_inputs()
    machine["pairwise_rows"][1]["effect"] = 0.005
    evidence, _ = _report_evidence(tmp_path)
    output = tmp_path / "bad-pairwise"
    with pytest.raises(ValueError, match="contradicts"):
        build_report_artifacts(output, audit={}, selection={}, lockcheck={}, formal={}, machine_inputs=machine, evidence_artifacts=evidence)
    assert not output.exists()


def test_statistical_plotting_writes_300dpi_png_and_pdf_and_handles_empty_groups(tmp_path: Path) -> None:
    empty_paths = plot_ablation([], tmp_path / "empty_ablation")
    assert {Path(value).suffix for value in empty_paths} == {".png", ".pdf"}
    from PIL import Image

    with Image.open(tmp_path / "empty_ablation.png") as image:
        assert image.info["dpi"][0] == pytest.approx(300, abs=1)
    manifest = build_statistical_figures(
        tmp_path / "figures",
        result_rows=[
            {"method": "q_only", "j_at_1": 0.5, "oracle_at_5": 0.8},
            {"method": "v2_locked_primary", "j_at_1": 0.6, "oracle_at_5": 0.8},
            {"method": "v3_primary", "j_at_1": 0.65, "oracle_at_5": 0.8},
        ],
        pairwise_rows=[],
        reliability={"q_only": [], "v2_locked_primary": [], "v3_primary": []},
        risk_coverage={},
        ablation_rows=[],
        outcome_rows=[],
        feature_distribution_rows=[],
        gate_uncertainty={},
        switch_precision_coverage=[],
    )
    assert manifest["palette"] == "Okabe-Ito"
    assert manifest["png_dpi"] == 300
    assert len(manifest["figures"]) == 9
    assert all(Path(item["path"]).exists() for outputs in manifest["figures"].values() for item in outputs)


def test_ten_gallery_categories_are_declared_and_classified() -> None:
    names = [name for name, _ in GALLERY_GROUPS]
    assert len(names) == 10 and len(set(names)) == 10
    observed = set()
    cases = [
        dict(q_correct=False, v2_correct=False, v3_correct=True, oracle=True, legacy_v3_correct=False, corrected_v3_correct=True, mask_iou=0.8),
        dict(q_correct=False, v2_correct=True, v3_correct=False, oracle=True, legacy_v3_correct=False, corrected_v3_correct=False, mask_iou=0.8),
        dict(q_correct=True, v2_correct=False, v3_correct=True, oracle=True, legacy_v3_correct=True, corrected_v3_correct=True, mask_iou=0.8),
        dict(q_correct=False, v2_correct=False, v3_correct=False, oracle=False, legacy_v3_correct=False, corrected_v3_correct=False, mask_iou=0.2),
        dict(q_correct=True, v2_correct=True, v3_correct=False, oracle=True, legacy_v3_correct=True, corrected_v3_correct=False, mask_iou=0.8),
    ]
    for case in cases:
        observed.update(classify_gallery_groups(**case))
    assert observed == set(names)


def _candidates():
    result = []
    for index in range(5):
        x = 25 + index * 12
        result.append(
            {
                "candidate_id": f"c{index}",
                "polygon": [[x - 8, 35], [x + 8, 35], [x + 8, 45], [x - 8, 45]],
                "q_raw": 0.9 - index * 0.1,
                "cx": x,
                "cy": 40,
                "width_px": 16,
                "height_px": 10,
                "angle_deg": 0,
                "features": {"mask_support": 0.5, "angle_confidence": 0.6},
            }
        )
    return result


def test_render_case_aligns_vectors_and_marks_evaluation_only(monkeypatch, tmp_path: Path) -> None:
    image = np.full((80, 100, 3), 220, dtype=np.uint8)
    monkeypatch.setattr(galleries, "load_rgb", lambda _: image.copy())
    monkeypatch.setattr(galleries, "load_gt_mask", lambda *_: np.zeros((80, 100), dtype=bool))
    monkeypatch.setattr(galleries, "draw_grasps", lambda value, *_args, **_kwargs: value)
    candidates = _candidates()
    labels = [{"candidate_id": f"c{index}", "candidate_correct": index == 2} for index in range(5)]
    sample = SimpleNamespace(
        sample_id="multiple:val:00000001",
        feature={
            "image_path": "unused.png",
            "candidates": candidates,
            "predicted_mask_rle": {"size": [80, 100], "start_value": 0, "counts": [8000]},
            "language_instruction": "grasp the synthetic object",
        },
        label={"candidate_labels": labels},
    )
    v2 = {"candidate_order": ["c1", "c0", "c2", "c3", "c4"]}
    v3 = {
        "candidate_order": ["c2", "c1", "c0", "c3", "c4"],
        "candidate_probability_ids": ["c4", "c3", "c2", "c1", "c0"],
        "candidate_scores": [0.4, 0.3, 0.9, 0.2, 0.1],
        "gate_gains": [0.04, 0.03, 0.09, 0.02, 0.01],
        "uncertainty": [0.4, 0.3, 0.2, 0.1, 0.0],
        "candidate_evidence": [{"top_token_index": index} for index in range(4, -1, -1)],
    }
    output = tmp_path / "case.png"
    record = render_case(
        sample=sample,
        legacy_label={"candidate_labels": labels},
        raw_prediction={"mask_path": "unused-mask", "obj_id": 1, "gt_grasps": []},
        v2=v2,
        v3=v3,
        group="01_v3_recovered_from_v2",
        output_path=output,
    )
    assert output.exists() and output.stat().st_size > 0
    assert output.with_suffix(".pdf").exists()
    from PIL import Image

    with Image.open(output) as image:
        assert image.info["dpi"][0] == pytest.approx(300, abs=1)
    assert record["gt_usage"] == "evaluation_panel_only"
    assert record["v3_candidate_id"] == "c2"
    assert record["evidence_coverage"]["candidate_scores"]["status"] == "complete"
    assert record["evidence_coverage"]["depth_evidence"]["status"] == "n/a"


def test_gallery_builder_emits_all_empty_group_sections(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(galleries, "load_joined", lambda *_: [])
    paths = {}
    for name in ("features", "corrected", "legacy", "raw", "v2", "v3"):
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("")
    output = tmp_path / "gallery"
    _, completion = _report_evidence(tmp_path)
    result = build_v3_galleries_strict(
        features_path=paths["features"],
        corrected_labels_path=paths["corrected"],
        legacy_labels_path=paths["legacy"],
        raw_predictions_path=paths["raw"],
        v2_predictions_path=paths["v2"],
        v3_predictions_path=paths["v3"],
        output_dir=output,
        per_group=5,
        independent_evaluation_completion_path=completion,
    )
    assert result["case_count"] == 0
    assert result["labels_read"] is True
    assert result["formal_evidence_ready"] is True
    assert result["independent_evaluation_completion"] == artifact_identity(completion)
    assert result["input_hashes"] == {name: identity["sha256"] for name, identity in result["inputs"].items()}
    assert result["evidence_coverage"]["depth_evidence"] == {
        "available": 0,
        "total": 0,
        "coverage": None,
        "status": "n/a",
    }
    assert all((output / name).is_dir() for name, _ in GALLERY_GROUPS)
    html = (output / "index.html").read_text()
    assert html.count("No samples met this category") == 10
