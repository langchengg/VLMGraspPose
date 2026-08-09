from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from src.segmentation.sam3_cpu_serialization import atomic_output_directory
from src.segmentation.selective_sam3_vg.features import pre_sam_features
from src.segmentation.selective_sam3_vg.io import sha256_file
from src.segmentation.selective_sam3_vg.models import select_candidate, trigger_probability
from src.segmentation.selective_sam3_vg.prompts import build_selective_prompt


ROOT = Path(__file__).resolve().parents[1]


def test_valid_empty_prediction_has_finite_pre_sam_features() -> None:
    mask = np.zeros((12, 16), dtype=bool)
    probability = np.zeros(mask.shape, dtype=np.float32)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    depth = np.zeros(mask.shape, dtype=np.uint16)
    features = pre_sam_features(mask, probability, rgb, depth, "the object")
    assert features["coarse_area_fraction"] == 0.0
    assert features["bbox_area_fraction"] == 0.0
    assert features["query_type"] == "name"
    assert all(
        np.isfinite(value)
        for value in features.values()
        if isinstance(value, (int, float))
    )


def test_authoritative_repeated_film_baseline_reproduces_exactly() -> None:
    summary = json.loads(
        (ROOT / "outputs/selective_sam3_vg/diagnostics/baseline_diagnostics_summary.json").read_text()
    )["formal_test"]
    assert summary["samples"] == 7675
    assert summary["mean_iou"] == pytest.approx(0.8074137115168442, abs=1e-15)
    assert summary["p_at_50_numerator"] == 6997
    assert summary["p_at_60_numerator"] == 6818
    assert summary["p_at_70_numerator"] == 6475
    assert summary["p_at_80_numerator"] == 5655
    assert summary["p_at_90_numerator"] == 3363


def test_scene_and_rgb_splits_are_disjoint() -> None:
    audit = json.loads((ROOT / "outputs/selective_sam3_vg/splits/split_audit.json").read_text())
    for pair in ("train_val", "train_test", "val_test"):
        assert audit["overlaps"][pair]["scene_overlap"] == 0
        assert audit["overlaps"][pair]["processed_rgb_path_overlap"] == 0


def test_prompt_generation_is_deterministic_and_prediction_only() -> None:
    config = yaml.safe_load((ROOT / "configs/selective_sam3_vg_validation.yaml").read_text())["prompt"]
    probability = np.zeros((80, 100), dtype=np.float32)
    probability[20:55, 25:65] = 0.91
    probability[30:35, 70:75] = 0.72
    mask = probability >= 0.5
    depth = np.full(mask.shape, 1000, dtype=np.uint16)
    first = build_selective_prompt("P4", probability, mask, depth, box_expansion_fraction=0.50, config=config)
    second = build_selective_prompt("P4", probability, mask, depth, box_expansion_fraction=0.50, config=config)
    assert first.metadata == second.metadata
    assert np.array_equal(first.visual_prompt.cleaned_mask, second.visual_prompt.cleaned_mask)
    core_threshold = first.metadata["core_probability_threshold"]
    core = first.visual_prompt.cleaned_mask & (probability >= core_threshold)
    assert all(core[y, x] for x, y in first.visual_prompt.positive_points_xy)
    assert all(not core[y, x] for x, y in first.visual_prompt.negative_points_xy)
    assert first.metadata["uses_ground_truth"] is False


def test_original_mask_is_always_candidate_and_safe_fallback() -> None:
    candidate = {
        "candidate_id": "coarse_0", "sam_quality": 0.0, "coarse_sam_iou": 1.0,
        "hifi_probability_mass_recall": 1.0, "positive_point_inclusion_ratio": 1.0,
        "largest_component_ratio": 1.0, "rgb_edge_alignment": 0.5,
        "depth_consistency_with_positive_region": 1.0, "low_hifi_probability_expansion_fraction": 0.0,
        "fragmentation_penalty": 0.0, "sam_to_coarse_area_ratio": 1.0,
        "prompt_box_support": 1.0,
    }
    artifact = {
        "selector_type": "deterministic_rule", "acceptance_margin": 0.0,
        "conservative_gate": {
            "minimum_positive_point_inclusion": 1.0, "minimum_probability_mass_recall": 0.75,
            "minimum_area_ratio": 0.4, "maximum_area_ratio": 2.0,
            "maximum_low_probability_expansion": 0.75, "maximum_fragmentation_penalty": 0.5,
            "minimum_prompt_box_support": 0.95,
        },
    }
    result = select_candidate(artifact, [candidate])
    assert result["selected_candidate_id"] == "coarse_0"
    assert result["selected_source"] == "hifics"
    assert "no_sam_candidate" in result["fallback_reason"]


def test_atomic_output_refuses_overwrite_and_cleans_failure(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    with pytest.raises(RuntimeError):
        with atomic_output_directory(destination) as temporary:
            (temporary / "partial").write_text("x")
            raise RuntimeError("injected")
    assert not destination.exists()
    with atomic_output_directory(destination) as temporary:
        (temporary / "status.json").write_text("{}")
    with pytest.raises(FileExistsError):
        with atomic_output_directory(destination):
            pass


def test_static_inference_leakage_guard_passes() -> None:
    result = subprocess.run(
        ["python3", str(ROOT / "tools/selective_sam3_vg/leakage_guard.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_baseline_files_remain_byte_identical() -> None:
    manifest = pd.read_parquet(ROOT / "outputs/selective_sam3_vg/baseline_manifest.parquet")
    # Full-file verification is intentional: the experiment must not mutate a
    # single authoritative prediction artifact.
    for row in manifest.itertuples(index=False):
        assert sha256_file(row.probability_path) == row.probability_sha256
        assert sha256_file(row.coarse_mask_path) == row.coarse_mask_sha256


def test_locked_models_are_deterministic_when_available() -> None:
    trigger_path = ROOT / "artifacts/selective_sam3_vg/locked_trigger/model.pkl"
    selector_path = ROOT / "artifacts/selective_sam3_vg/locked_selector/model.pkl"
    if not trigger_path.is_file() or not selector_path.is_file():
        pytest.skip("validation training has not finished")
    with trigger_path.open("rb") as stream:
        trigger = pickle.load(stream)
    sample = next(
        (ROOT / "outputs/selective_sam3_vg/validation_full").glob("*/*/pre_sam_features.json")
    )
    features = json.loads(sample.read_text())
    assert trigger_probability(trigger, features) == trigger_probability(trigger, features)


def test_all_formal_outputs_are_terminal_and_locked_when_available() -> None:
    lock_path = ROOT / "artifacts/selective_sam3_vg/formal_output_lock.json"
    if not lock_path.is_file():
        pytest.skip("formal run has not finished")
    lock = json.loads(lock_path.read_text())
    manifest = pd.read_parquet(lock["output_manifest_path"])
    assert len(manifest) == 7675
    assert manifest["sample_id"].nunique() == 7675
    assert sha256_file(lock["output_manifest_path"]) == lock["output_manifest_sha256"]
    for row in manifest.itertuples(index=False):
        mask = np.asarray(Image.open(row.final_mask_path).convert("L"))
        assert mask.shape == (480, 640)
        assert set(np.unique(mask)).issubset({0, 255})
        probability = np.load(row.final_probability_path, allow_pickle=False)
        assert probability.shape == (480, 640)
        assert probability.dtype == np.float32
        assert np.isfinite(probability).all()


def test_evaluator_requires_output_lock_before_gt_access() -> None:
    source = (ROOT / "tools/selective_sam3_vg/evaluate_formal.py").read_text()
    assert source.index("formal_output_lock.json") < source.index("baseline_manifest.parquet")
    assert "LOCKED_BEFORE_GT_EVALUATION" in source
