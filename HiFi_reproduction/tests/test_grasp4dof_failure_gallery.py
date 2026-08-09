"""Formal test-only failure taxonomy and selected-gallery regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.grasping.common import Grasp4DoF, GraspPrediction
from src.grasping.common.results import evaluate_prediction_records
from tools.grasp4dof.build_failure_gallery import (
    DenseMapExtractor,
    _heatmap_panel,
    build_failure_gallery,
    main as gallery_main,
    select_gallery_cases,
)


def test_dense_map_rendering_is_finite_and_rejects_non_network_config() -> None:
    panel = _heatmap_panel(
        np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8),
        title="QUALITY",
    )
    assert panel.size == (420, 315)
    with pytest.raises(ValueError, match="only for G0/G1/C0/C1"):
        DenseMapExtractor({"A0": Path("unused.json")})


def _corners(x: float = 80.0, y: float = 60.0, width: float = 40.0) -> list[list[float]]:
    return [
        [x - width / 2, y - 10],
        [x - width / 2, y + 10],
        [x + width / 2, y + 10],
        [x + width / 2, y - 10],
    ]


def _candidate(
    candidate_id: str,
    *,
    score: float,
    center_x: float = 80.0,
    center_y: float = 60.0,
    angle_deg: float = 0.0,
    width_px: float = 40.0,
) -> Grasp4DoF:
    return Grasp4DoF(
        center_x=center_x,
        center_y=center_y,
        angle_deg=angle_deg,
        width_px=width_px,
        height_px=20.0,
        score=score,
        candidate_id=candidate_id,
    )


def _write_image_inputs(root: Path, sample_id: str, *, wrong_mask: bool) -> dict[str, str]:
    sample_root = root / "images" / sample_id
    sample_root.mkdir(parents=True)
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(160, dtype=np.uint8)[None, :]
    rgb[..., 1] = np.arange(120, dtype=np.uint8)[:, None]
    depth = np.full((120, 160), 900, dtype=np.uint16)
    gt = np.zeros((120, 160), dtype=np.uint8)
    gt[42:78, 55:105] = 255
    predicted = np.zeros_like(gt)
    if wrong_mask:
        predicted[3:18, 3:18] = 255
    else:
        predicted[43:77, 57:103] = 255
    paths = {
        "source_rgb_path": sample_root / "rgb.png",
        "source_depth_path": sample_root / "depth.png",
        "predicted_mask_path": sample_root / "predicted.png",
        "prepared_gt_mask_path": sample_root / "gt.png",
    }
    Image.fromarray(rgb).save(paths["source_rgb_path"])
    Image.fromarray(depth).save(paths["source_depth_path"])
    Image.fromarray(predicted).save(paths["predicted_mask_path"])
    Image.fromarray(gt).save(paths["prepared_gt_mask_path"])
    return {key: str(value) for key, value in paths.items()}


def _pools(method: str) -> dict[str, list[Grasp4DoF]]:
    wrong = lambda sample, rank, score: _candidate(  # noqa: E731
        f"{method}-{sample}-{rank}",
        score=score,
        center_x=135.0,
        center_y=100.0,
    )
    if method == "G1":
        success = [_candidate("G1-success-1", score=0.9)]
        ranking = [wrong("ranking", 1, 0.9), _candidate("G1-ranking-2", score=0.8)]
    else:
        # Reversed outcomes exercise deterministic disagreement-first cross selection.
        success = [wrong("success", 1, 0.9)]
        ranking = [_candidate("C1-ranking-1", score=0.9)]
    return {
        "success": success,
        "ranking": ranking,
        "no-candidate": [],
        "wrong-mask": [],
        "angle": [_candidate(f"{method}-angle-1", score=0.9, angle_deg=31.0)],
        "width": [_candidate(f"{method}-width-1", score=0.9, width_px=8.0)],
    }


def _fixture(tmp_path: Path) -> tuple[Path, list[tuple[str, Path]]]:
    run = tmp_path / "formal-run"
    manifests = run / "manifests"
    manifests.mkdir(parents=True)
    sample_ids = ("success", "ranking", "no-candidate", "wrong-mask", "angle", "width")
    sample_rows: list[dict] = []
    label_rows: list[dict] = []
    for index, sample_id in enumerate(sample_ids):
        paths = _write_image_inputs(
            tmp_path, sample_id, wrong_mask=sample_id == "wrong-mask"
        )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "scene_id": f"scene-{index // 2}",
                "split": "test",
                "language": f"pick the object for {sample_id}",
                "source_rgb_path": paths["source_rgb_path"],
                "source_depth_path": paths["source_depth_path"],
                "predicted_mask_path": paths["predicted_mask_path"],
            }
        )
        label_rows.append(
            {
                "sample_id": sample_id,
                "scene_id": f"scene-{index // 2}",
                "split": "test",
                "prepared_gt_mask_path": paths["prepared_gt_mask_path"],
                "gt_grasp_rectangles": [_corners()],
            }
        )
    pd.DataFrame(sample_rows).to_parquet(
        manifests / "test_samples.parquet", index=False
    )
    pd.DataFrame(label_rows).to_parquet(
        manifests / "test_labels.parquet", index=False
    )

    methods: list[tuple[str, Path]] = []
    labels_by_id = {row["sample_id"]: row for row in label_rows}
    for method in ("G1", "C1"):
        method_dir = tmp_path / f"method-{method}"
        method_dir.mkdir()
        predictions: list[dict] = []
        candidates: list[dict] = []
        for sample_id, pool in _pools(method).items():
            prediction = GraspPrediction(
                sample_id=sample_id,
                backend="synthetic",
                conditioning_variant="hard_mask",
                raw_candidate_count=len(pool),
                nms_candidate_count=len(pool),
                top1=pool[0] if pool else None,
                top5=tuple(pool[:5]),
                candidates=tuple(pool),
                empty_reason=None if pool else "no_candidate_generated",
                runtime_seconds=0.001,
                metadata={"quality_map": {"maximum": 0.9}},
            )
            sample_row, candidate_rows = evaluate_prediction_records(
                method=method,
                prediction=prediction,
                label=labels_by_id[sample_id],
            )
            predictions.append(sample_row)
            candidates.extend(candidate_rows)
        pd.DataFrame(predictions).to_parquet(
            method_dir / "per_sample_predictions.parquet", index=False
        )
        pd.DataFrame(candidates).to_parquet(
            method_dir / "per_candidate_predictions.parquet", index=False
        )
        methods.append((method, method_dir))
    return run, methods


def test_builds_failure_table_and_underfilled_selected_gallery(tmp_path: Path) -> None:
    run, methods = _fixture(tmp_path)

    result = build_failure_gallery(run_dir=run, method_dirs=methods)

    assert result["status"] == "COMPLETE"
    assert result["analysis_scope"] == "formal_test_only"
    assert result["configuration_selection_performed"] is False
    analysis = pd.read_parquet(run / "per_sample_failure_stage.parquet")
    assert len(analysis) == 12
    assert set(analysis["dense_maps_status"]) == {"unavailable"}
    assert not analysis["configuration_selection_performed"].any()
    g1 = analysis.loc[analysis["method"] == "G1"].set_index("sample_id")
    assert g1.loc["success", "failure_stage"] == "successful"
    assert g1.loc["ranking", "failure_stage"] == "ranking_failure"
    assert bool(g1.loc["ranking", "candidate_pool_oracle"])
    assert g1.loc["no-candidate", "failure_stage"] == "no_candidate_generated"
    assert bool(g1.loc["no-candidate", "no_candidate_flag"])
    assert not bool(g1.loc["no-candidate", "candidate_pool_has_no_positive_flag"])
    assert g1.loc["wrong-mask", "failure_stage"] == "grounding_wrong_target"
    assert bool(g1.loc["wrong-mask", "wrong_mask_flag"])
    assert bool(g1.loc["wrong-mask", "no_candidate_flag"])
    assert g1.loc["angle", "failure_stage"] == "angle_failure"
    assert bool(g1.loc["angle", "candidate_pool_has_no_positive_flag"])
    assert g1.loc["width", "failure_stage"] == "width_failure"

    gallery = run / "gallery"
    saved = json.loads((gallery / "selection_manifest.json").read_text())
    assert saved["selection"]["counts"]["G1"]["success"] == {
        "requested": 20,
        "available": 1,
        "actual": 1,
    }
    assert saved["selection"]["counts"]["G1"]["no_candidate"] == {
        "requested": 15,
        "available": 2,
        "actual": 2,
    }
    assert saved["selection"]["cross_method_count"] == {
        "requested": 30,
        "available": 6,
        "actual": 6,
    }
    pngs = list((gallery / "images").glob("*.png"))
    assert len(pngs) == (
        saved["rendered_method_image_count"]
        + saved["rendered_cross_method_image_count"]
    )
    index = (gallery / "index.html").read_text(encoding="utf-8")
    assert "Post-hoc test-only analysis" in index
    assert "Dense maps" in index and "unavailable" in index
    assert "data:image/png;base64," in index
    assert "http://" not in index and "https://" not in index


def test_selection_is_deterministic_and_disagreement_first(tmp_path: Path) -> None:
    run, methods = _fixture(tmp_path)
    build_failure_gallery(run_dir=run, method_dirs=methods)
    analysis = pd.read_parquet(run / "per_sample_failure_stage.parquet")

    first = select_gallery_cases(analysis)
    shuffled = analysis.sample(frac=1.0, random_state=919).reset_index(drop=True)
    second = select_gallery_cases(shuffled)

    assert first == second
    cross = first["cross_method"]
    assert cross[0]["outcome_disagreement"] is True
    assert {cross[0]["sample_id"], cross[1]["sample_id"]} == {
        "success",
        "ranking",
    }


def test_cli_refuses_overwrite_and_non_test_manifest(tmp_path: Path) -> None:
    run, methods = _fixture(tmp_path / "valid")
    args = ["--run-dir", str(run)]
    for name, path in methods:
        args.extend(["--method-dir", f"{name}={path}"])
    assert gallery_main(args) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gallery_main(args)

    run, methods = _fixture(tmp_path / "invalid")
    manifest_path = run / "manifests" / "test_samples.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest.loc[0, "split"] = "validation"
    manifest.to_parquet(manifest_path, index=False)
    with pytest.raises(ValueError, match="not test-only"):
        build_failure_gallery(run_dir=run, method_dirs=methods)


def test_missing_formal_sample_fails_before_rendering(tmp_path: Path) -> None:
    run, methods = _fixture(tmp_path)
    method_path = methods[0][1] / "per_sample_predictions.parquet"
    predictions = pd.read_parquet(method_path)
    predictions = predictions.loc[predictions["sample_id"] != "width"]
    predictions.to_parquet(method_path, index=False)

    with pytest.raises(ValueError, match="sample coverage mismatch"):
        build_failure_gallery(run_dir=run, method_dirs=methods)
    assert not (run / "gallery").exists()
    assert not (run / "per_sample_failure_stage.parquet").exists()
