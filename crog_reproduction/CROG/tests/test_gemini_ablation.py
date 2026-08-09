from __future__ import annotations

import numpy as np

from failure_analysis.gemini_crog_evidence_v1.ablation import (
    metadata_for_ablation,
    render_ablation_board,
)
from failure_analysis.gemini_crog_evidence_v1.renderer import deterministic_candidate_mapping


def _fixture():
    candidates = []
    evidence = []
    for index in range(5):
        candidate_id = f"candidate_{index}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "cx": 12.0 + 10 * index,
                "cy": 32.0,
                "angle_deg": -30.0 + 15 * index,
                "width_px": 10.0 + index,
                "polygon": [[8 + 10 * index, 28], [16 + 10 * index, 28], [16 + 10 * index, 36], [8 + 10 * index, 36]],
            }
        )
        evidence.append(
            {
                "candidate_id": candidate_id,
                "original_q_rank": index,
                "q_probability_at_center": 0.9 - 0.1 * index,
                "mask_probability_at_center": 0.7,
                "rectangle_mask_probability_mean": 0.6,
                "min_jaw_mask_support": 0.5,
                "signed_distance_to_predicted_mask_boundary": 20.0,
                "angle_deg_periodic_180": -30.0 + 15 * index,
                "angle_vector_magnitude": 0.8,
                "angle_consistency_score": 0.7,
                "width_px": 10.0 + index,
                "width_consistency_score": 0.75,
                "candidate_uniqueness": 0.6,
            }
        )
    mapping = deterministic_candidate_mapping([item["candidate_id"] for item in candidates], "sample")
    return candidates, evidence, mapping


def test_ablation_metadata_removes_registered_modalities():
    candidates, evidence, mapping = _fixture()
    common = {
        "referring_expression_block": "<referring_expression>red cup</referring_expression>",
        "candidates": candidates,
        "candidate_evidence": evidence,
        "mapping": mapping,
        "original_q_top1_candidate_id": "candidate_0",
    }
    a0 = metadata_for_ablation(protocol="A0", **common)
    assert "q_probability:" not in a0 and "mask_center_probability:" not in a0
    a1 = metadata_for_ablation(protocol="A1", **common)
    assert "q_probability:" in a1 and "mask_center_probability:" not in a1
    a3 = metadata_for_ablation(protocol="A3", **common)
    assert "q_probability:" not in a3 and "mask_center_probability:" in a3


def test_a3_board_has_no_q_panel(tmp_path):
    candidates, evidence, mapping = _fixture()
    shape = (64, 64)
    manifest = render_ablation_board(
        protocol="A3",
        rgb=np.zeros((*shape, 3), dtype=np.uint8),
        candidates=candidates,
        candidate_evidence=evidence,
        mapping=mapping,
        sample_id="sample",
        output_path=tmp_path / "a3.png",
        mask_probability=np.full(shape, 0.5, dtype=np.float32),
        sin_2theta=np.zeros(shape, dtype=np.float32),
        cos_2theta=np.ones(shape, dtype=np.float32),
        width_probability=np.full(shape, 0.25, dtype=np.float32),
    )
    assert (tmp_path / "a3.png").is_file()
    assert manifest["q_map_included"] is False
    assert "predicted_q" not in manifest["panels"]
