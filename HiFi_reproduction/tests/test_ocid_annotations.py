from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.experiments.ocid_annotations import (
    GTOracleMappingError,
    QUERY_TYPE_RULE_PAYLOAD,
    annotate_expression,
    annotate_top1_with_gt,
    build_gt_oracle_sample,
    derive_query_type,
    evaluate_predicted_mask,
    expression_for_sample,
    iou_precision_indicators,
    load_expression_index,
    load_gt_mask,
    mask_iou,
    nearest_gt_point_distance_m,
    projected_uv_inside_mask,
    resolve_gt_mask_reference,
    resolve_gt_oracle_mapping,
    target_category_from_name,
)
from src.grasping.vgn_pipeline import ManifestSample, load_sample_arrays, sha256_file


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_image(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value).save(path)


@pytest.fixture()
def annotated_sample(tmp_path: Path) -> ManifestSample:
    sample_id = "q0000007_fixture"
    predictions = tmp_path / "predictions" / sample_id
    bundle = tmp_path / "bundle" / sample_id
    source = tmp_path / "ocid" / "scene"
    instance = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 0, 0, 3],
        ],
        dtype=np.uint8,
    )
    gt = (instance == 2).astype(np.uint8) * 255
    predicted = gt.copy()
    predicted[1, 2] = 0
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.full((4, 5), 1000, dtype=np.uint16)

    instance_path = source / "seg_mask_instances_combi" / "image.png"
    rgb_path = source / "rgb" / "image.png"
    depth_path = source / "depth" / "image.png"
    predicted_path = bundle / "target_mask.png"
    metadata_path = predictions / "sample_metadata.json"
    gt_path = predictions / "ground_truth_mask_original_resolution.png"
    intrinsics_path = bundle / "intrinsics.json"
    _write_image(instance_path, instance)
    _write_image(rgb_path, rgb)
    _write_image(depth_path, depth)
    _write_image(predicted_path, predicted)
    _write_image(gt_path, gt)
    _write_json(
        metadata_path,
        {
            "stable_sample_id": sample_id,
            "question_index": 7,
            "scene_id": "scene,image.png",
            "query": "the red cereal box",
            "answer_instance_value": 2,
            "target_name": "cereal_box_3",
            "source_rgb_path": str(rgb_path),
            "source_instance_mask_path": str(instance_path),
        },
    )
    _write_json(
        intrinsics_path,
        {"width": 5, "height": 4, "fx": 10, "fy": 10, "cx": 2, "cy": 1.5},
    )
    bundle_metadata_path = bundle / "metadata.json"
    _write_json(
        bundle_metadata_path,
        {"prediction_sample_metadata": str(metadata_path)},
    )
    return ManifestSample(
        sample_id=sample_id,
        dataset_index=7,
        scene_id="scene,image.png",
        instruction="the red cereal box",
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=predicted_path,
        bundle_dir=bundle,
        metadata_path=bundle_metadata_path,
        intrinsics_path=intrinsics_path,
        view="top",
        row={"question_index": 7},
        metadata={
            "prediction_sample_metadata": str(metadata_path),
            "prediction_mask_sha256": sha256_file(predicted_path),
        },
    )


def test_gt_oracle_mapping_is_unique(annotated_sample: ManifestSample) -> None:
    reference = resolve_gt_mask_reference(annotated_sample)
    assert resolve_gt_oracle_mapping(annotated_sample) == reference.exported_gt_mask_path
    assert reference.answer_instance_value == 2
    assert reference.target_name == "cereal_box_3"
    assert reference.target_category == "cereal_box"
    assert (reference.width, reference.height) == (5, 4)
    expected = np.asarray(Image.open(reference.source_instance_mask_path)) == 2
    np.testing.assert_array_equal(load_gt_mask(reference), expected)


def test_gt_oracle_refuses_ambiguous_mapping(
    annotated_sample: ManifestSample,
) -> None:
    gt_path = resolve_gt_oracle_mapping(annotated_sample)
    conflicting = np.zeros((4, 5), dtype=np.uint8)
    conflicting[3, 4] = 255
    _write_image(gt_path, conflicting)
    with pytest.raises(GTOracleMappingError) as error:
        resolve_gt_oracle_mapping(annotated_sample)
    assert error.value.status == "gt_oracle_ambiguous"
    assert "does not exactly equal" in str(error.value)


def test_gt_oracle_missing_metadata_is_unavailable(
    annotated_sample: ManifestSample,
) -> None:
    sample = ManifestSample(
        **{
            **annotated_sample.__dict__,
            "metadata": {},
        }
    )
    with pytest.raises(GTOracleMappingError) as error:
        resolve_gt_oracle_mapping(sample)
    assert error.value.status == "gt_oracle_unavailable"


def test_build_gt_oracle_sample_changes_only_mask_provenance(
    annotated_sample: ManifestSample,
) -> None:
    oracle = build_gt_oracle_sample(annotated_sample)
    assert oracle.mask_path.name == "ground_truth_mask_original_resolution.png"
    assert oracle.sample_id == annotated_sample.sample_id
    assert oracle.rgb_path == annotated_sample.rgb_path
    assert oracle.depth_path == annotated_sample.depth_path
    assert oracle.metadata["mask_source"] == "ground_truth_mask_oracle"
    _, _, loaded = load_sample_arrays(oracle)
    np.testing.assert_array_equal(loaded != 0, load_gt_mask(annotated_sample))


def test_mask_iou_and_p_at_thresholds(annotated_sample: ManifestSample) -> None:
    result = evaluate_predicted_mask(annotated_sample)
    assert result["mask_iou"] == pytest.approx(3 / 4)
    assert result["pred_mask_area_px"] == 3
    assert result["gt_mask_area_px"] == 4
    assert {result[f"p_at_{n}"] for n in (50, 60, 70)} == {1}
    assert {result[f"p_at_{n}"] for n in (80, 90)} == {0}
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
    assert iou_precision_indicators(0.9)["p_at_90"] == 1


@pytest.mark.parametrize(
    ("operators", "expected"),
    [
        (["scene", "filter_category", "unique", "return"], "name"),
        (
            ["scene", "filter_category", "filter_color", "unique", "return"],
            "attribute",
        ),
        (["scene", "ground", "relate", "filter_category", "return"], "relation"),
        (["scene", "filter_category", "locate", "return"], "location"),
        (
            [
                "scene",
                "filter_category",
                "filter_color",
                "relate",
                "filter_category",
                "return",
            ],
            "mixed",
        ),
        (["scene", "future_operator", "return"], "unknown"),
    ],
)
def test_symbolic_query_type_rules(operators: list[str], expected: str) -> None:
    result = derive_query_type([{"type": operator} for operator in operators])
    assert result.query_type == expected
    assert result.rule_version == QUERY_TYPE_RULE_PAYLOAD["version"]


def test_query_type_does_not_use_natural_language() -> None:
    expression = {
        "question": "left red object behind a box",
        "target": "flashlight_1",
        "template_filename": "location.json",
        "program": [
            {"type": "scene"},
            {"type": "filter_category"},
            {"type": "unique"},
            {"type": "return"},
        ],
    }
    annotated = annotate_expression(expression)
    assert annotated["query_type"] == "name"
    assert annotated["target_category"] == "flashlight"
    assert "query_type_rule_payload" in annotated


def test_target_category_requires_dataset_instance_suffix() -> None:
    assert target_category_from_name("instant_noodles_2") == "instant_noodles"
    assert target_category_from_name("flashlight") is None
    assert target_category_from_name(None) is None


def test_expression_index_requires_unique_alignment(
    tmp_path: Path, annotated_sample: ManifestSample
) -> None:
    path = tmp_path / "test_expressions.json"
    record = {
        "question_index": 7,
        "image_filename": annotated_sample.scene_id,
        "question": annotated_sample.instruction,
        "target": "cereal_box_3",
        "program": [{"type": "scene"}, {"type": "ground"}, {"type": "return"}],
    }
    _write_json(path, {"info": {"split": "test"}, "data": [record]})
    index = load_expression_index(path)
    assert expression_for_sample(annotated_sample, index) == record
    record["question"] = "different"
    with pytest.raises(ValueError, match="question mismatch"):
        expression_for_sample(annotated_sample, {7: record})


def test_projection_and_nearest_gt_point_helpers() -> None:
    mask = np.zeros((4, 5), dtype=bool)
    mask[2, 3] = True
    assert projected_uv_inside_mask([3.1, 1.8], mask)
    assert not projected_uv_inside_mask([-0.1, 2], mask)
    assert nearest_gt_point_distance_m(
        [0, 0, 1], np.array([[0, 0, 2], [0, 0, 1.25]])
    ) == pytest.approx(0.25)


def test_annotate_top1_with_gt_metric_depth(
    annotated_sample: ManifestSample,
) -> None:
    payload = {
        "candidate": {
            "position_camera_m": [-0.1, 0.05, 1.0],
            "projected_uv": [1.0, 2.0],
        }
    }
    result = annotate_top1_with_gt(
        annotated_sample, payload, depth_unit="mm", depth_scale=1000
    )
    assert result["target_consistency_metric"] is True
    assert result["top1_inside_gt_target_mask"] is True
    assert result["top1_nearest_gt_target_point_distance_m"] == pytest.approx(0.0)
    assert result["top1_projected_depth_error_m"] == pytest.approx(0.0)
    assert result["gt_target_valid_depth_points"] == 4


def test_annotate_top1_refuses_implicit_depth_unit(
    annotated_sample: ManifestSample,
) -> None:
    payload = {
        "position_camera_m": [0.0, 0.0, 1.0],
        "projected_uv": [2.0, 2.0],
    }
    with pytest.raises(ValueError, match="explicitly"):
        annotate_top1_with_gt(annotated_sample, payload, depth_unit="auto")
