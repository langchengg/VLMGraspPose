"""Independent formal recomputation from geometry-only candidate artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.grasping.common import Grasp4DoF, GraspPrediction
from src.grasping.common.results import (
    aggregate_method_metrics,
    evaluate_prediction_records,
)
from tools.grasp4dof.independent_recompute import (
    main as independent_main,
    run_independent_recompute,
)


def _corners(x: float, y: float, width: float = 40.0) -> list[list[float]]:
    return [
        [x - width / 2, y - 10],
        [x - width / 2, y + 10],
        [x + width / 2, y + 10],
        [x + width / 2, y - 10],
    ]


def _candidate(
    candidate_id: str, rank: int, *, correct: bool, score: float
) -> Grasp4DoF:
    return Grasp4DoF(
        center_x=100.0 if correct else 250.0 + rank,
        center_y=100.0 if correct else 200.0,
        angle_deg=0.0,
        width_px=40.0,
        height_px=20.0,
        score=score,
        candidate_id=candidate_id,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "formal-run"
    manifests = run / "manifests"
    manifests.mkdir(parents=True)
    labels = [
        {
            "sample_id": sample_id,
            "scene_id": scene_id,
            "gt_grasp_rectangles": [_corners(100, 100)],
        }
        for sample_id, scene_id in (
            ("sample-correct", "scene-a"),
            ("sample-empty", "scene-a"),
            ("sample-fewer5", "scene-b"),
        )
    ]
    pd.DataFrame(labels).to_parquet(manifests / "test_labels.parquet", index=False)

    method = tmp_path / "method-G1"
    method.mkdir()
    pools = {
        "sample-correct": [_candidate("correct-1", 1, correct=True, score=0.9)],
        "sample-empty": [],
        "sample-fewer5": [
            _candidate("wrong-1", 1, correct=False, score=0.9),
            _candidate("correct-2", 2, correct=True, score=0.8),
            _candidate("wrong-3", 3, correct=False, score=0.7),
        ],
    }
    sample_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for label in labels:
        sample_id = label["sample_id"]
        pool = pools[sample_id]
        top5 = tuple(pool[:5])
        prediction = GraspPrediction(
            sample_id=sample_id,
            backend="synthetic",
            conditioning_variant="hard_mask",
            raw_candidate_count=len(pool),
            nms_candidate_count=len(pool),
            top1=top5[0] if top5 else None,
            top5=top5,
            candidates=tuple(pool),
            empty_reason=None if pool else "no_candidate_generated",
            runtime_seconds=0.01,
        )
        sample_row, rows = evaluate_prediction_records(
            method="G1", prediction=prediction, label=label
        )
        # These stored correctness fields are deliberately wrong.  The
        # independent audit must ignore them and trust geometry only.
        for row in rows:
            row["candidate_success"] = not bool(row["candidate_success"])
            row["pairwise_json"] = '[{"tampered":true}]'
        sample_rows.append(sample_row)
        candidate_rows.extend(rows)
    pd.DataFrame(candidate_rows).to_parquet(
        method / "per_candidate_predictions.parquet", index=False
    )
    pd.DataFrame(sample_rows).to_parquet(
        method / "per_sample_predictions.parquet", index=False
    )
    metrics = aggregate_method_metrics(sample_rows)
    (method / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True), encoding="utf-8"
    )
    return run, method


def test_geometry_only_recompute_matches_with_empty_and_fewer_than_five(
    tmp_path: Path,
) -> None:
    run, method = _fixture(tmp_path)

    result = run_independent_recompute(
        run_dir=run, method_dirs=[("G1", method)]
    )

    assert result["status"] == "EXACT_MATCH"
    assert result["candidate_correctness_fields_trusted"] is False
    method_result = result["methods"][0]
    assert method_result["empty_prediction_count"] == 1
    assert method_result["fewer_than_five_count"] == 3
    assert method_result["metrics"]["j_at_1"] == pytest.approx(1 / 3)
    assert method_result["metrics"]["j_at_5"] == pytest.approx(2 / 3)
    assert method_result["metrics"]["mrr"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("artifact", "column", "value", "message"),
    (
        ("sample", "j_at_1", False, "saved j_at_1 mismatch"),
        ("sample", "top5_candidate_ids_json", '["drifted"]', "top5_candidate_ids"),
        ("candidate", "rank", 4, "candidate ranks"),
        ("candidate", "candidate_id", "correct-2", "duplicate candidate_id"),
        ("metrics", "j_at_1", 0.0, "aggregate j_at_1 mismatch"),
    ),
)
def test_recompute_rejects_tampering(
    tmp_path: Path, artifact: str, column: str, value: object, message: str
) -> None:
    run, method = _fixture(tmp_path)
    if artifact == "metrics":
        path = method / "metrics.json"
        payload = json.loads(path.read_text())
        payload[column] = value
        path.write_text(json.dumps(payload), encoding="utf-8")
        frame = None
        index = -1
    elif artifact == "sample":
        path = method / "per_sample_predictions.parquet"
        frame = pd.read_parquet(path)
        index = int(frame.index[frame["sample_id"] == "sample-correct"][0])
    else:
        path = method / "per_candidate_predictions.parquet"
        frame = pd.read_parquet(path)
        index = int(frame.index[frame["candidate_id"] == "wrong-3"][0])
    if frame is not None:
        frame.loc[index, column] = value
        frame.to_parquet(path, index=False)

    with pytest.raises((ValueError, RuntimeError), match=message):
        run_independent_recompute(run_dir=run, method_dirs=[("G1", method)])


def test_missing_and_duplicate_sample_rows_fail(tmp_path: Path) -> None:
    run, method = _fixture(tmp_path)
    path = method / "per_sample_predictions.parquet"
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["sample_id"] != "sample-empty"]
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="sample coverage mismatch"):
        run_independent_recompute(run_dir=run, method_dirs=[("G1", method)])

    run, method = _fixture(tmp_path / "duplicate")
    path = method / "per_sample_predictions.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        run_independent_recompute(run_dir=run, method_dirs=[("G1", method)])


def test_cli_supports_multiple_methods_and_refuses_overwrite(tmp_path: Path) -> None:
    run, first = _fixture(tmp_path / "first")
    _, second = _fixture(tmp_path / "second")

    assert (
        independent_main(
            [
                "--run-dir",
                str(run),
                "--method-dir",
                f"G1={first}",
                "--method-dir",
                f"C1={second}",
            ]
        )
        == 0
    )
    output = run / "independent_recompute_results.json"
    saved = json.loads(output.read_text())
    assert saved["method_count"] == 2
    assert all(item["status"] == "EXACT_MATCH" for item in saved["methods"])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        independent_main(
            [
                "--run-dir",
                str(run),
                "--method-dir",
                f"G1={first}",
            ]
        )
