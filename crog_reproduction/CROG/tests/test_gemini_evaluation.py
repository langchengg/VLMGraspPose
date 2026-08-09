from __future__ import annotations

import json

from failure_analysis.gemini_crog_evidence_v1.evaluation import (
    evaluate_saved_predictions,
    load_evaluation_index,
    paired_statistical_tests,
    select_validation_primary,
    stable_candidate_id,
)


def _candidate(index: int):
    return {
        "candidate_id": f"candidate_{index}",
        "candidate_checksum": f"checksum_{index}",
        "cx": 10.0 + index,
        "cy": 20.0 + index,
        "row": 20 + index,
        "col": 10 + index,
        "angle_deg": float(index),
        "width_px": 20.0 + index,
        "height_px": 20.0,
        "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "q_raw": 1.0 - 0.1 * index,
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _index(tmp_path):
    features = []
    legacy = []
    corrected = []
    for sample_index in range(2):
        sample_id = f"multiple:test:{sample_index:08d}"
        features.append({"scene_id": f"scene-{sample_index}", "candidates": [_candidate(i) for i in range(5)]})
        legacy.append(
            {
                "sample_id": sample_id,
                "candidate_labels": [
                    {"candidate_id": f"candidate_{i}", "candidate_correct": i == (1 if sample_index == 0 else 0)}
                    for i in range(5)
                ],
            }
        )
        corrected.append(
            {
                "sample_id": sample_id,
                "candidate_labels": [
                    {"candidate_id": f"candidate_{i}", "candidate_correct": i in ({0, 1} if sample_index == 0 else {0})}
                    for i in range(5)
                ],
            }
        )
    paths = [tmp_path / name for name in ("features.jsonl", "legacy.jsonl", "corrected.jsonl")]
    for path, rows in zip(paths, (features, legacy, corrected), strict=True):
        _write_jsonl(path, rows)
    return load_evaluation_index(
        features_path=paths[0], legacy_labels_path=paths[1], corrected_labels_path=paths[2]
    )


def test_dual_track_evaluation_uses_only_saved_ids(tmp_path):
    index = _index(tmp_path)
    predictions = []
    for sample_index in range(2):
        sample_id = f"multiple:test:{sample_index:08d}"
        selected = "candidate_1" if sample_index == 0 else "candidate_0"
        predictions.append(
            {
                "sample_id": sample_id,
                "method": "candidate",
                "selected_stable_candidate_id": stable_candidate_id(sample_id, selected),
                "valid_response": True,
                "selected_score_margin": 0.2,
            }
        )
    outcomes, metrics = evaluate_saved_predictions(evaluation_index=index, predictions=predictions)
    assert len(outcomes) == 2
    assert metrics[0]["legacy_recovered"] == 1
    assert metrics[0]["legacy_harmful"] == 0
    assert metrics[0]["corrected_recovered"] == 0


def test_statistics_and_primary_selection_are_deterministic(tmp_path):
    index = _index(tmp_path)
    predictions = []
    for sample_index in range(2):
        sample_id = f"multiple:test:{sample_index:08d}"
        predictions.append(
            {
                "sample_id": sample_id,
                "method": "gemini_3_6_flash_crog_evidence_safe",
                "selected_stable_candidate_id": stable_candidate_id(
                    sample_id, "candidate_1" if sample_index == 0 else "candidate_0"
                ),
                "valid_response": True,
            }
        )
    outcomes, metrics = evaluate_saved_predictions(evaluation_index=index, predictions=predictions)
    tests, bootstrap = paired_statistical_tests(outcomes=outcomes, draws=100, seed=47)
    assert tests["test"] == "exact_mcnemar_two_sided"
    selected = select_validation_primary(
        metrics=metrics,
        bootstrap=bootstrap,
        estimated_cost_by_method={"gemini_3_6_flash_crog_evidence_safe": 1.0},
        harmful_rate_limit=0.01,
        maximum_negative_scene_ci_pp=100.0,
    )
    assert selected["locked_primary"] == "gemini_3_6_flash_crog_evidence_safe"
