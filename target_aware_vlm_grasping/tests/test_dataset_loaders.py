from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.ocid_vlg_loader import OCIDGraspIndexBuilder, OCIDVLGIndexBuilder


def test_dataset_builders_return_language_conditioned_samples() -> None:
    vlg_root = ROOT / "data" / "OCID-VLG"
    grasp_root = ROOT / "data" / "OCID-Grasp"
    vlg = OCIDVLGIndexBuilder(vlg_root, ROOT / "outputs" / "test_dataset_loaders").build(max_samples=1)
    grasp = OCIDGraspIndexBuilder(grasp_root, ROOT / "outputs" / "test_dataset_loaders").build(max_samples=1)
    assert vlg[0].command
    assert vlg[0].target_bbox_gt
    assert grasp[0].command.startswith("pick the ")
    assert grasp[0].target_bbox_gt
