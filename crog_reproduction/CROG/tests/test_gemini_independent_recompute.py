from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from failure_analysis.gemini_crog_evidence_v1.independent_recompute_gemini_results import (
    recompute,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sample_id = "multiple:test:00000000"
    candidates = [
        {"candidate_id": str(index), "q_raw": score}
        for index, score in enumerate((0.7, 0.9, 0.4, 0.2, 0.1))
    ]
    features = tmp_path / "features.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    corrected = tmp_path / "corrected.jsonl"
    predictions = tmp_path / "predictions.parquet"
    _write_jsonl(features, [{"split": "test", "sample_id": 0, "candidates": candidates}])
    _write_jsonl(
        legacy,
        [
            {
                "sample_id": sample_id,
                "candidate_labels": [
                    {"candidate_id": str(index), "candidate_correct": index == 0}
                    for index in range(5)
                ],
            }
        ],
    )
    _write_jsonl(
        corrected,
        [
            {
                "sample_id": sample_id,
                "candidate_labels": [
                    {"candidate_id": str(index), "candidate_correct": index in {0, 1}}
                    for index in range(5)
                ],
            }
        ],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": sample_id,
                    "method": "crog_q_only",
                    "selected_stable_candidate_id": f"{sample_id}/1",
                },
                {
                    "sample_id": sample_id,
                    "method": "candidate_zero",
                    "selected_stable_candidate_id": f"{sample_id}/0",
                },
            ]
        ),
        predictions,
    )
    return predictions, features, legacy, corrected


def test_independent_recompute_uses_q_raw_top1_and_both_tracks(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = recompute(
        predictions_path=paths[0],
        features_path=paths[1],
        legacy_labels_path=paths[2],
        corrected_labels_path=paths[3],
    )

    changed = result["aggregate"]["candidate_zero"]
    assert changed["legacy_recovered"] == 1
    assert changed["legacy_harmful"] == 0
    assert changed["corrected_net"] == 0
    assert result["aggregate"]["crog_q_only"]["legacy_j1"] == 0.0


def test_independent_recompute_rejects_candidate_outside_frozen_top5(
    tmp_path: Path,
) -> None:
    predictions, features, legacy, corrected = _fixture(tmp_path)
    table = pq.read_table(predictions).to_pylist()
    table[0]["selected_stable_candidate_id"] = "multiple:test:00000000/99"
    pq.write_table(pa.Table.from_pylist(table), predictions)

    with pytest.raises(ValueError, match="outside frozen Top-5"):
        recompute(
            predictions_path=predictions,
            features_path=features,
            legacy_labels_path=legacy,
            corrected_labels_path=corrected,
        )


def test_independent_recompute_rejects_duplicate_predictions(tmp_path: Path) -> None:
    predictions, features, legacy, corrected = _fixture(tmp_path)
    rows = pq.read_table(predictions).to_pylist()
    pq.write_table(pa.Table.from_pylist(rows + [rows[0]]), predictions)

    with pytest.raises(ValueError, match="duplicate saved prediction"):
        recompute(
            predictions_path=predictions,
            features_path=features,
            legacy_labels_path=legacy,
            corrected_labels_path=corrected,
        )


def test_independent_recompute_uses_raw_gt_and_verified_evaluators(tmp_path: Path) -> None:
    sample_id = "multiple:test:00000000"
    candidates = []
    for index in range(5):
        grasp = [100.0 + index * 60.0, 100.0, 40.0, 20.0, 0.0]
        candidates.append(
            {
                "candidate_id": str(index),
                "q_raw": 1.0 - index / 10.0,
                "cx": grasp[0],
                "cy": grasp[1],
                "width_px": grasp[2],
                "height_px": grasp[3],
                "angle_deg": grasp[4],
                "legacy_grasp": grasp,
            }
        )
    features = tmp_path / "features.jsonl"
    raw = tmp_path / "raw_predictions.jsonl"
    predictions = tmp_path / "predictions.parquet"
    _write_jsonl(features, [{"split": "test", "sample_id": 0, "candidates": candidates}])
    _write_jsonl(
        raw,
        [{"split": "test", "sample_id": 0, "gt_grasps": [candidates[1]["legacy_grasp"]]}],
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": sample_id,
                    "method": "select_gt",
                    "selected_stable_candidate_id": f"{sample_id}/1",
                }
            ]
        ),
        predictions,
    )

    result = recompute(
        predictions_path=predictions,
        features_path=features,
        raw_predictions_path=raw,
    )

    assert result["ground_truth_source_kind"] == "raw_gt_grasps_recomputed_with_verified_evaluators"
    assert result["aggregate"]["select_gt"]["legacy_recovered"] == 1
    assert result["aggregate"]["select_gt"]["corrected_recovered"] == 1
    assert result["aggregate"]["select_gt"]["legacy_oracle_at_5"] == 1.0
    assert result["input_sha256"]["raw_predictions_with_gt"]
