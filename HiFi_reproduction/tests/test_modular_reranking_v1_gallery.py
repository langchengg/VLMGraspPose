from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.grasping.reranking_v1.gallery import (
    GT_COLOR,
    GalleryError,
    generate_failure_gallery,
)


def _write_fixture(root: Path) -> dict[str, Path]:
    candidate_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    hifi_rows: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    candidate_root = root / "candidates"
    image_root = root / "images"
    mask_root = root / "masks"
    image_root.mkdir()
    mask_root.mkdir()
    categories = {
        "s-recovered": (False, True, None),
        "s-harmful": (True, False, None),
        "s-both": (False, False, None),
        "s-fallback": (
            False,
            False,
            {
                "fallback": True,
                "abstain": True,
                "fallback_reason": "abstain",
                "selected_candidate_id": "g0",
                "ranking": [],
            },
        ),
        "s-disagree": (
            False,
            True,
            {
                "fallback": False,
                "abstain": False,
                "selected_candidate_id": "g0",
                "ranking": [
                    {"candidate_id": "g0", "score": 0.9},
                    {"candidate_id": "g1", "score": 0.8},
                ],
            },
        ),
    }
    vlm_rows: list[dict[str, object]] = []
    for index, (sample_id, (old_ok, new_ok, vlm)) in enumerate(categories.items()):
        scene_id = f"scene-{index}.png"
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(160, dtype=np.uint8)
        rgb[..., 1] = 72 + index * 15
        rgb_path = image_root / f"{sample_id}.png"
        Image.fromarray(rgb).save(rgb_path)
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[32:94, 38:128] = 255
        mask_path = mask_root / f"{sample_id}.png"
        Image.fromarray(mask).save(mask_path)
        sample_dir = candidate_root / sample_id
        sample_dir.mkdir(parents=True)
        geometries = []
        for candidate_index, candidate_id in enumerate(("g0", "g1", "g2")):
            positive = (
                old_ok
                if candidate_id == "g0"
                else new_ok
                if candidate_id == "g1"
                else True
            )
            rank = candidate_index + 1
            candidate_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "candidate_id": candidate_id,
                    "original_gqcnn_rank": rank,
                    "q_raw": 0.9 - candidate_index * 0.2,
                    "candidate_positive": positive,
                    "candidate_gt_iou": 0.42 if positive else 0.12,
                    "candidate_gt_angle_error_deg": 10.0 if positive else 42.0,
                    "best_gt_id": 0,
                }
            )
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "reranker_method": "learned",
                    "reranker_rank": (2, 1, 3)[candidate_index],
                    "reranker_score": (0.2, 0.8, 0.1)[candidate_index],
                }
            )
            geometries.append(
                {
                    "candidate_id": candidate_id,
                    "center_u_px": 55 + candidate_index * 23,
                    "center_v_px": 62 + candidate_index * 5,
                    "angle_rad": candidate_index * 0.3,
                    "width_px": 38,
                }
            )
        (sample_dir / "candidates.json").write_text(
            json.dumps({"candidates": geometries}), encoding="utf-8"
        )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "candidate_count": 3,
            }
        )
        hifi_rows.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "question_index": index,
                "query": f"grasp object number {index}",
                "source_rgb_path": str(rgb_path),
                "native_mask_path": str(mask_path),
            }
        )
        annotations.append(
            {
                "question_index": index,
                "image_filename": scene_id,
                "question": f"grasp object number {index}",
                "grasps": [[[47, 45], [47, 76], [102, 76], [102, 45]]],
            }
        )
        if vlm is not None:
            vlm_rows.append({"sample_id": sample_id, **vlm})
    per_candidate = root / "per_candidate.parquet"
    per_sample = root / "per_sample.parquet"
    predictions = root / "predictions.parquet"
    hifi_manifest = root / "hifi.jsonl"
    annotation_file = root / "annotations.json"
    vlm_results = root / "vlm.jsonl"
    config = root / "metric.yaml"
    pd.DataFrame(candidate_rows).to_parquet(per_candidate, index=False)
    pd.DataFrame(sample_rows).to_parquet(per_sample, index=False)
    pd.DataFrame(prediction_rows).to_parquet(predictions, index=False)
    hifi_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in hifi_rows), encoding="utf-8"
    )
    annotation_file.write_text(
        json.dumps({"data": annotations}), encoding="utf-8"
    )
    vlm_results.write_text(
        "".join(json.dumps(row) + "\n" for row in vlm_rows), encoding="utf-8"
    )
    config.write_text(
        "predicted_rectangle_height_px: 20.0\n"
        "ground_truth_rectangle_height_px: 20.0\n"
        "ground_truth_width_clip_px: 100.0\n"
        "angle_threshold_deg: 30.0\n"
        "iou_threshold: 0.25\n"
        "top_k: 5\n",
        encoding="utf-8",
    )
    return {
        "per_candidate": per_candidate,
        "per_sample": per_sample,
        "predictions": predictions,
        "hifi_manifest": hifi_manifest,
        "candidate_root": candidate_root,
        "annotations": annotation_file,
        "vlm_results": vlm_results,
        "config": config,
    }


def test_gallery_renders_all_categories_and_isolates_gt(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    output = tmp_path / "gallery"
    result = generate_failure_gallery(
        per_candidate_path=paths["per_candidate"],
        per_sample_path=paths["per_sample"],
        predictions_path=paths["predictions"],
        method="learned",
        hifi_manifest_path=paths["hifi_manifest"],
        candidate_root=paths["candidate_root"],
        annotation_file=paths["annotations"],
        evaluation_config_path=paths["config"],
        output_dir=output,
        vlm_results_path=paths["vlm_results"],
        quotas={
            "recovered": 1,
            "harmful": 1,
            "both_wrong": 1,
            "vlm_fallback": 1,
            "vlm_learned_disagreement": 1,
        },
    )
    assert result["rendered_case_count"] == 5
    assert not result["shortfalls"]
    assert (output / "index.html").is_file()
    manifest = [
        json.loads(line)
        for line in (output / "gallery_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["category"] for row in manifest} == {
        "recovered",
        "harmful",
        "both_wrong",
        "vlm_fallback",
        "vlm_learned_disagreement",
    }
    for row in manifest:
        image_path = output / row["image_path"]
        case_path = output / row["case_json_path"]
        assert image_path.is_file() and case_path.is_file()
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        inference_text = json.dumps(payload["inference"], sort_keys=True)
        assert payload["inference"]["gt_fields_included"] is False
        assert "candidate_positive" not in inference_text
        assert "candidate_gt_iou" not in inference_text
        assert "ground_truth" not in inference_text
        image = np.asarray(Image.open(image_path).convert("RGB"))
        exact_gt = np.all(image == np.asarray(GT_COLOR), axis=2)
        # GT cyan is present only in the right-most lower evaluation panel.
        assert exact_gt[:, : 2 * 496].sum() == 0
        assert exact_gt[:, 2 * 496 :].sum() > 0
    markup = (output / "index.html").read_text(encoding="utf-8")
    assert "EVALUATION ONLY" in markup
    assert 'data-filter="recovered"' in markup


def test_gallery_fails_closed_on_quota_shortfall(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    try:
        generate_failure_gallery(
            per_candidate_path=paths["per_candidate"],
            per_sample_path=paths["per_sample"],
            predictions_path=paths["predictions"],
            method="learned",
            hifi_manifest_path=paths["hifi_manifest"],
            candidate_root=paths["candidate_root"],
            annotation_file=paths["annotations"],
            evaluation_config_path=paths["config"],
            output_dir=tmp_path / "gallery",
            quotas={
                "recovered": 3,
                "harmful": 0,
                "both_wrong": 0,
                "vlm_fallback": 0,
                "vlm_learned_disagreement": 0,
            },
        )
    except GalleryError as error:
        assert "quota shortfall" in str(error)
    else:
        raise AssertionError("strict gallery generation accepted a quota shortfall")
