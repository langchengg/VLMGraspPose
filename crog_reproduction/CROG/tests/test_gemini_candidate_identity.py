from __future__ import annotations

import copy

import pytest

from failure_analysis.gemini_crog_evidence_v1.renderer import deterministic_candidate_mapping
from failure_analysis.gemini_crog_evidence_v1.security import (
    assert_candidate_identity,
    assert_display_mapping,
    assert_no_gt_values_in_provider_text,
)


def _candidates():
    return [
        {
            "candidate_id": f"candidate_{index}",
            "candidate_checksum": f"checksum_{index}",
            "cx": 20.0 + index,
            "cy": 30.0 + index,
            "row": 30 + index,
            "col": 20 + index,
            "angle_deg": -40.0 + index,
            "width_px": 25.0 + index,
            "height_px": 20.0,
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "q_raw": 0.9 - 0.1 * index,
        }
        for index in range(5)
    ]


def test_candidate_identity_detects_any_geometry_or_q_change():
    frozen = _candidates()
    assert_candidate_identity(frozen, frozen_candidates=copy.deepcopy(frozen))
    for field in ("cx", "angle_deg", "width_px", "q_raw"):
        changed = copy.deepcopy(frozen)
        changed[2][field] += 1e-9
        with pytest.raises(ValueError, match="identity changed"):
            assert_candidate_identity(changed, frozen_candidates=frozen)
    changed = copy.deepcopy(frozen)
    changed[0]["polygon"][0][0] += 1e-9
    with pytest.raises(ValueError, match="identity changed"):
        assert_candidate_identity(changed, frozen_candidates=frozen)


def test_display_mapping_is_exact_bijection():
    candidates = _candidates()
    ids = [item["candidate_id"] for item in candidates]
    mapping = deterministic_candidate_mapping(ids, "multiple:test:00000000")
    assert_display_mapping(mapping, candidate_ids=ids)
    broken = copy.deepcopy(mapping)
    broken["display_to_candidate"]["E"] = broken["display_to_candidate"]["A"]
    with pytest.raises(ValueError):
        assert_display_mapping(broken, candidate_ids=ids)


def test_provider_text_gt_value_audit_rejects_identity_and_grasp_tuple():
    assert_no_gt_values_in_provider_text(
        "A has predicted width 25.0",
        correct_candidate_ids={"candidate_2"},
        gt_grasps=[[1.0, 2.0, 3.0, 4.0, 5.0]],
    )
    with pytest.raises(ValueError, match="GT-only candidate"):
        assert_no_gt_values_in_provider_text(
            "selected candidate_2",
            correct_candidate_ids={"candidate_2"},
            gt_grasps=[],
        )
    with pytest.raises(ValueError, match="GT grasp-coordinate"):
        assert_no_gt_values_in_provider_text(
            "1.000000,2.000000,3.000000,4.000000",
            correct_candidate_ids=set(),
            gt_grasps=[[1.0, 2.0, 3.0, 4.0, 5.0]],
        )
