from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.grasping.geometric_ranker import (
    evaluate_planar_annotation_consistency,
    load_ocid_vlg_annotation_record,
    save_strict_json,
)
from src.grasping.gqcnn_ranking_evaluation import (
    BASELINE_NAME,
    PER_SAMPLE_FIELDS,
    EvaluationDataError,
    aggregate_metrics,
    classify_failures,
    evaluate_sample,
    load_evaluation_config,
    load_verified_scored_candidates,
    rank_stored_q_values,
    save_invalid_diagnostic,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "dexnet_grasp_consistency.yaml"
ANNOTATIONS = ROOT.parent / "crog_reproduction" / "OCID-VLG" / "refer" / "unique" / "test_expressions.json"
TEN_ROOT = ROOT / "outputs" / "dexnet_candidates_ten_samples"
ONE_SAMPLE = TEN_ROOT / "q0000000_b32eb3299dcd3ae9"


def _config() -> dict:
    return load_evaluation_config(CONFIG_PATH)


def _candidate(candidate_id: str, q: float, center=(32.0, 24.0), angle=0.0) -> dict:
    axis = np.array([math.cos(angle), math.sin(angle)])
    contacts = [np.asarray(center) - 10.0 * axis, np.asarray(center) + 10.0 * axis]
    return {
        "candidate_id": candidate_id,
        "gqcnn_q_value": q,
        "center_uv": list(center),
        "contact_points_uv": [item.tolist() for item in contacts],
    }


def _gt(center=(32.0, 24.0), width=20.0, height=20.0, angle=0.0):
    center = np.asarray(center, dtype=float)
    axis = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-axis[1], axis[0]])
    return np.stack(
        [
            center - width * axis / 2 - height * normal / 2,
            center - width * axis / 2 + height * normal / 2,
            center + width * axis / 2 + height * normal / 2,
            center + width * axis / 2 - height * normal / 2,
        ]
    ).tolist()


def _evaluate(records, grasps):
    ranked = rank_stored_q_values(records)
    result = evaluate_planar_annotation_consistency(
        ranked,
        grasps,
        _config(),
        rank_field="gqcnn_rank",
    )
    return ranked, result


def test_descending_q_order_and_deterministic_id_tie_break():
    ranked = rank_stored_q_values(
        [_candidate("g0002", 0.5), _candidate("g0003", 0.9), _candidate("g0001", 0.5)]
    )
    assert [item["candidate_id"] for item in ranked] == ["g0003", "g0001", "g0002"]
    assert [item["gqcnn_rank"] for item in ranked] == [1, 2, 3]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_q_values_are_rejected(bad):
    with pytest.raises(EvaluationDataError, match="non-finite") as caught:
        rank_stored_q_values([_candidate("g0001", bad)])
    assert caught.value.category == "invalid_gqcnn_scores"


def test_shared_evaluator_rejects_zero_based_or_noncontiguous_rank():
    record = _candidate("g0001", 1.0)
    record["gqcnn_rank"] = 0
    with pytest.raises(ValueError, match="one-based and contiguous"):
        evaluate_planar_annotation_consistency(
            [record], [_gt()], _config(), rank_field="gqcnn_rank"
        )


def test_top1_and_top5_with_fewer_than_five_candidates():
    ranked, result = _evaluate(
        [_candidate("bad", 0.9, center=(4, 4)), _candidate("good", 0.8)],
        [_gt()],
    )
    assert len(ranked) == 2
    assert result["top1_rectangle_accuracy"] is False
    assert result["topk_recall"] is True
    assert result["first_matching_rank"] == 2


def test_no_valid_candidate_has_null_first_rank():
    _, result = _evaluate(
        [_candidate("a", 0.9, center=(2, 2)), _candidate("b", 0.8, center=(60, 40))],
        [_gt()],
    )
    assert result["top1_rectangle_accuracy"] is False
    assert result["topk_recall"] is False
    assert result["first_matching_rank"] is None


def test_multiple_valid_candidates_use_first_one_based_rank():
    _, result = _evaluate(
        [_candidate("bad", 0.9, center=(2, 2)), _candidate("valid2", 0.8), _candidate("valid3", 0.7)],
        [_gt()],
    )
    assert result["first_matching_rank"] == 2
    assert sum(item["rectangle_match"] for item in result["per_candidate"]) == 2


def test_angle_symmetry_is_modulo_pi():
    _, result = _evaluate([_candidate("reverse", 1.0, angle=math.pi)], [_gt(angle=0.0)])
    item = result["per_candidate"][0]
    assert item["minimum_angle_difference_deg_modulo_pi"] == pytest.approx(0.0, abs=1e-10)
    assert item["rectangle_match"] is True


def test_multiple_ground_truth_annotations_match_any_one():
    _, result = _evaluate(
        [_candidate("target", 1.0)],
        [_gt(center=(200, 200)), _gt(center=(32, 24))],
    )
    assert result["annotation_count"] == 2
    assert result["per_candidate"][0]["rectangle_match"] is True


def test_failure_taxonomy_separates_ranking_and_generation():
    assert classify_failures(
        candidate_count=8,
        top1_consistent=False,
        top5_consistent=True,
        first_valid_rank=3,
    ) == ["top1_ranking_failure"]
    assert classify_failures(
        candidate_count=20,
        top1_consistent=False,
        top5_consistent=False,
        first_valid_rank=16,
    ) == ["top1_ranking_failure", "top5_ranking_failure"]
    assert classify_failures(
        candidate_count=20,
        top1_consistent=False,
        top5_consistent=False,
        first_valid_rank=None,
    ) == ["candidate_generation_failure"]
    assert classify_failures(
        candidate_count=0,
        top1_consistent=False,
        top5_consistent=False,
        first_valid_rank=None,
    ) == ["no_candidates"]


def _copy_scored_fixture(tmp_path: Path) -> Path:
    sample = tmp_path / ONE_SAMPLE.name
    sample.mkdir()
    for name in (
        "metadata.json",
        "candidates.json",
        "candidates.npz",
        "gqcnn_scored_candidates.json",
        "gqcnn_scored_candidates.npz",
        "gqcnn_scored_candidates.csv",
    ):
        shutil.copy2(ONE_SAMPLE / name, sample / name)
    return sample


def _rewrite_npz(path: Path, mutate):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    mutate(arrays)
    np.savez(path, **arrays)


def test_candidate_id_mismatch_is_rejected(tmp_path: Path):
    sample = _copy_scored_fixture(tmp_path)
    _rewrite_npz(
        sample / "gqcnn_scored_candidates.npz",
        lambda arrays: arrays["candidate_id"].__setitem__(0, "wrong_id"),
    )
    with pytest.raises(EvaluationDataError) as caught:
        load_verified_scored_candidates(sample)
    assert caught.value.category == "mapping_or_geometry_error"


def test_candidate_pose_mismatch_is_rejected(tmp_path: Path):
    sample = _copy_scored_fixture(tmp_path)
    _rewrite_npz(
        sample / "gqcnn_scored_candidates.npz",
        lambda arrays: arrays["center_uv"].__setitem__((0, 0), arrays["center_uv"][0, 0] + 1),
    )
    with pytest.raises(EvaluationDataError) as caught:
        load_verified_scored_candidates(sample)
    assert caught.value.category == "mapping_or_geometry_error"


def test_strict_json_and_csv_serialization(tmp_path: Path):
    json_path = save_strict_json(tmp_path / "strict.json", {"rank": None, "q": float("nan")})
    assert json.loads(json_path.read_text()) == {"rank": None, "q": None}
    row = {field: None for field in PER_SAMPLE_FIELDS}
    row.update({"sample_id": "sample", "top1_consistent": False, "first_valid_rank": None})
    csv_path = write_csv(tmp_path / "metrics.csv", [row], PER_SAMPLE_FIELDS)
    saved = next(csv.DictReader(csv_path.open()))
    assert saved["sample_id"] == "sample"
    assert saved["top1_consistent"] == "0"
    assert saved["first_valid_rank"] == ""


def test_invalid_sample_gets_diagnostic_not_accuracy_visualization(tmp_path: Path):
    sample_dir = tmp_path / "bad_sample"
    sample_dir.mkdir()
    row = {
        "sample_id": "bad_sample",
        "failure_type": "annotation_unavailable",
        "failure_reason": "no usable annotations",
    }
    paths = save_invalid_diagnostic(sample_dir, tmp_path / "output", row)
    assert {path.name for path in paths} == {"diagnostic.png", "failure_details.json"}
    details = json.loads(
        (tmp_path / "output/failures/bad_sample/failure_details.json").read_text()
    )
    assert details["accuracy_visualization_generated"] is False
    assert details["error"]["failure_type"] == "annotation_unavailable"


def test_question_index_lookup_does_not_assume_list_position():
    record = load_ocid_vlg_annotation_record(ANNOTATIONS, 21)
    assert int(record["question_index"]) == 21
    assert record["question"] == "Grasp the Vichy shampoo"


def test_real_output_integrity_and_geometric_regression():
    config = _config()
    outcomes = [
        evaluate_sample(path, ANNOTATIONS, config)
        for path in sorted(TEN_ROOT.iterdir())
        if path.is_dir() and (path / "metadata.json").is_file()
    ]
    assert sum(item["finite_q_count"] for item in outcomes) == 241
    rank_mismatch_samples = {
        item["sample_metadata"]["sample_id"]
        for item in outcomes
        if not item["stored_rank_matches_exact_q_reconstruction"]
    }
    assert rank_mismatch_samples == {"q0000003_c9f21176e1f0d767"}
    assert sum(item["stored_rank_mismatch_count"] for item in outcomes) == 24

    geometric_rows = []
    for item in outcomes:
        row = dict(item["geometric_reference_metrics"])
        row["data_valid"] = True
        row["failure_categories"] = classify_failures(
            candidate_count=row["candidate_count"],
            top1_consistent=row["top1_consistent"],
            top5_consistent=row["top5_consistent"],
            first_valid_rank=row["first_valid_rank"],
        )
        row["failure_type"] = "|".join(row["failure_categories"])
        geometric_rows.append(row)
    geometric = aggregate_metrics(geometric_rows, method="geometric")
    assert geometric["top1_consistency"]["numerator"] == 8
    assert geometric["top1_consistency"]["denominator"] == 10
    assert geometric["top5_recall"]["numerator"] == 10
    assert geometric["top5_recall"]["denominator"] == 10
    assert geometric["first_valid_rank"]["mean"] == pytest.approx(1.2)

    gq_rows = []
    for item in outcomes:
        row = dict(item["metrics"])
        row["data_valid"] = True
        gq_rows.append(row)
    gq = aggregate_metrics(gq_rows, method=BASELINE_NAME)
    assert gq["top1_consistency"]["numerator"] == 8
    assert gq["top5_recall"]["numerator"] == 9
    assert gq["first_valid_rank"]["mean"] == pytest.approx(2.7)
    assert gq["first_valid_rank"]["per_sample"]["q0000002_65b99b4d1aaf2b7b"] == 16
