from __future__ import annotations

import itertools
from copy import deepcopy

import numpy as np
import pytest

from failure_analysis.reranking_v3.aligned_crops import CROP_CHANNELS
from failure_analysis.reranking_v3.candidate_relations import (
    G10_FEATURE_DIM,
    G10_FEATURE_NAMES,
    extract_candidate_relation_features,
)


HEAD_FEATURE_NAMES = (
    "g0_q_raw",
    "g1_q_probability_center",
    "g2_mask_axis_soft",
)


def _candidate(
    candidate_id: str,
    *,
    q_rank: int,
    q: float,
    cx: float,
    cy: float,
    width: float,
    height: float = 20.0,
    angle: float = 0.0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_checksum": f"checksum-{candidate_id}",
        "q_rank": q_rank,
        "q_raw": q,
        "cx": cx,
        "cy": cy,
        "width_px": width,
        "height_px": height,
        "angle_deg": angle,
    }


def _five_candidates() -> list[dict[str, object]]:
    # Deliberately not in q-rank order: array position must have no semantics.
    return [
        _candidate("c2", q_rank=2, q=0.71, cx=160, cy=110, width=42, angle=18),
        _candidate("c0", q_rank=0, q=0.94, cx=120, cy=100, width=55, angle=-1),
        _candidate("c4", q_rank=4, q=0.45, cx=410, cy=320, width=28, angle=83),
        _candidate("c1", q_rank=1, q=0.82, cx=124, cy=101, width=52, angle=179),
        _candidate("c3", q_rank=3, q=0.60, cx=250, cy=205, width=67, angle=41),
    ]


def _head_evidence() -> np.ndarray:
    return np.asarray(
        [
            [0.71, 0.76, 0.61],
            [0.94, 0.91, 0.86],
            [0.45, 0.38, 0.29],
            [0.82, 0.87, 0.81],
            [0.60, 0.58, 0.43],
        ],
        dtype=np.float64,
    )


def _lookup(values: np.ndarray, name: str) -> np.ndarray:
    return values[:, G10_FEATURE_NAMES.index(name)]


def _crops(quality: list[float], mask: list[float], size: int = 9) -> np.ndarray:
    result = np.zeros((len(quality), len(CROP_CHANNELS), size, size), dtype=np.float32)
    quality_index = CROP_CHANNELS.index("quality_probability")
    mask_index = CROP_CHANNELS.index("mask_probability")
    for index, (quality_value, mask_value) in enumerate(zip(quality, mask, strict=True)):
        result[index, quality_index] = quality_value
        result[index, mask_index] = mask_value
    return result


def test_feature_contract_is_stable_finite_and_label_free():
    candidates = _five_candidates()
    names, values = extract_candidate_relation_features(
        candidates,
        image_shape=(480, 640),
        head_features=_head_evidence(),
        head_feature_names=HEAD_FEATURE_NAMES,
    )
    assert names == G10_FEATURE_NAMES
    assert len(names) == G10_FEATURE_DIM == 86
    assert len(names) == len(set(names))
    assert values.shape == (5, G10_FEATURE_DIM)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert not any(token in name for name in names for token in ("ground_truth", "label", "success", "iou"))
    assert _lookup(values, "g10_candidate_q_dominance_margin")[1] == pytest.approx(0.12)
    assert _lookup(values, "g10_candidate_quality_dominance_margin")[1] == pytest.approx(0.04)
    assert _lookup(values, "g10_candidate_mask_support_dominance_margin")[1] == pytest.approx(0.05)
    assert _lookup(values, "g10_candidate_q_peer_win_fraction")[1] == 1.0
    assert _lookup(values, "g10_candidate_head_disagreement")[1] > 0.0
    for name in (
        "g10_set_center_diversity",
        "g10_set_axial_angle_diversity",
        "g10_set_width_diversity",
        "g10_set_q_disagreement",
    ):
        assert np.all(_lookup(values, name) > 0.0)

    contaminated = deepcopy(candidates)
    for candidate in contaminated:
        candidate.update({"ground_truth": [1, 2, 3], "label": 1, "success": True})
    _, contaminated_values = extract_candidate_relation_features(
        contaminated,
        image_shape=(480, 640),
        head_features=_head_evidence(),
        head_feature_names=HEAD_FEATURE_NAMES,
    )
    np.testing.assert_array_equal(contaminated_values, values)


def test_all_120_candidate_permutations_are_equivariant():
    candidates = _five_candidates()
    head = _head_evidence()
    _, baseline = extract_candidate_relation_features(
        candidates,
        image_shape=(480, 640),
        head_features=head,
        head_feature_names=HEAD_FEATURE_NAMES,
    )
    top1 = _lookup(baseline, "g10_candidate_is_q_top1")
    assert top1.tolist() == [0, 1, 0, 0, 0]
    assert _lookup(baseline, "g10_candidate_q_rank").tolist() == [2, 0, 4, 1, 3]

    for permutation in itertools.permutations(range(5)):
        order = np.asarray(permutation, dtype=np.int64)
        _, actual = extract_candidate_relation_features(
            [candidates[index] for index in order],
            image_shape=(480, 640),
            head_features=head[order],
            head_feature_names=HEAD_FEATURE_NAMES,
        )
        np.testing.assert_allclose(actual, baseline[order], rtol=0.0, atol=2e-7)


def test_candidate_mask_ignores_padded_rows_and_all_false_is_zero():
    candidates = _five_candidates()
    head = _head_evidence()
    mask = np.asarray([True, True, True, False, False])
    _, baseline = extract_candidate_relation_features(
        candidates,
        image_shape=(480, 640),
        head_features=head,
        head_feature_names=HEAD_FEATURE_NAMES,
        candidate_mask=mask,
    )
    assert np.count_nonzero(baseline[~mask]) == 0
    assert np.all(_lookup(baseline, "g10_set_valid_fraction")[mask] == pytest.approx(3 / 5))

    changed = deepcopy(candidates)
    changed[3] = {}
    changed[4] = _candidate("ignored", q_rank=999, q=1e20, cx=1e20, cy=-1e20, width=1e20)
    changed_head = head.copy()
    changed_head[~mask] = 1e20
    _, changed_values = extract_candidate_relation_features(
        changed,
        image_shape=(480, 640),
        head_features=changed_head,
        head_feature_names=HEAD_FEATURE_NAMES,
        candidate_mask=mask,
    )
    np.testing.assert_array_equal(changed_values, baseline)

    names, empty = extract_candidate_relation_features(
        [{}, {}],
        image_shape=(480, 640),
        candidate_mask=[False, False],
    )
    assert names == G10_FEATURE_NAMES
    assert empty.shape == (2, G10_FEATURE_DIM)
    assert np.count_nonzero(empty) == 0


def test_head_evidence_precedes_crop_and_nonfinite_head_falls_back_per_candidate():
    candidates = [
        _candidate("top", q_rank=0, q=0.2, cx=10, cy=10, width=20),
        _candidate("other", q_rank=1, q=0.3, cx=20, cy=10, width=20),
    ]
    head = np.asarray([[0.2, 0.9, 0.8], [0.3, np.nan, np.nan]], dtype=np.float64)
    crops = _crops([0.1, 0.6], [0.2, 0.7])
    _, values = extract_candidate_relation_features(
        candidates,
        image_shape=(100, 100),
        head_features=head,
        head_feature_names=HEAD_FEATURE_NAMES,
        crops=crops,
    )
    assert _lookup(values, "g10_candidate_quality_evidence_available").tolist() == [1, 1]
    assert _lookup(values, "g10_candidate_mask_evidence_available").tolist() == [1, 1]
    assert _lookup(values, "g10_relative_top1_quality_delta")[1] == pytest.approx(-0.3)
    assert _lookup(values, "g10_relative_top1_mask_support_delta")[1] == pytest.approx(-0.1)

    _, crop_only = extract_candidate_relation_features(
        candidates,
        image_shape=(100, 100),
        crops=crops,
    )
    assert _lookup(crop_only, "g10_relative_top1_quality_delta")[1] == pytest.approx(0.5)
    assert _lookup(crop_only, "g10_relative_top1_mask_support_delta")[1] == pytest.approx(0.5)


def test_axial_overlap_entropy_and_degenerate_numeric_boundaries():
    identical = [
        _candidate("a", q_rank=0, q=0.0, cx=50, cy=50, width=40, angle=179),
        _candidate("b", q_rank=1, q=0.0, cx=50, cy=50, width=40, angle=-1),
    ]
    head = np.zeros((2, len(HEAD_FEATURE_NAMES)), dtype=np.float64)
    _, values = extract_candidate_relation_features(
        identical,
        image_shape=(100, 100),
        head_features=head,
        head_feature_names=HEAD_FEATURE_NAMES,
    )
    for name in (
        "g10_pair_center_distance_fraction_max",
        "g10_pair_axial_angle_difference_fraction_max",
        "g10_pair_width_difference_fraction_max",
    ):
        np.testing.assert_allclose(_lookup(values, name), 0.0, atol=1e-7)
    for name in (
        "g10_pair_rectangle_overlap_mean",
        "g10_pair_axis_overlap_mean",
        "g10_pair_contact_overlap_mean",
        "g10_pair_same_local_peak_proxy_mean",
        "g10_set_q_entropy",
        "g10_set_quality_entropy",
        "g10_set_mask_support_entropy",
    ):
        np.testing.assert_allclose(_lookup(values, name), 1.0, atol=1e-6)

    singleton = [_candidate("zero", q_rank=0, q=0, cx=0, cy=0, width=0, height=0)]
    _, boundary = extract_candidate_relation_features(singleton, image_shape=(1, 1))
    assert boundary.shape == (1, G10_FEATURE_DIM)
    assert np.isfinite(boundary).all()
    assert _lookup(boundary, "g10_set_q_entropy")[0] == 0
    assert _lookup(boundary, "g10_relative_top1_rectangle_overlap")[0] == 0
    assert _lookup(boundary, "g10_pair_rectangle_overlap_max")[0] == 0


def test_nonoverlap_and_input_validation_boundaries():
    candidates = [
        _candidate("a", q_rank=0, q=0.5, cx=0, cy=0, width=10),
        _candidate("b", q_rank=1, q=0.5, cx=100, cy=100, width=10),
    ]
    _, values = extract_candidate_relation_features(candidates, image_shape=(100, 100))
    for name in (
        "g10_pair_rectangle_overlap_max",
        "g10_pair_axis_overlap_max",
        "g10_pair_contact_overlap_max",
    ):
        np.testing.assert_array_equal(_lookup(values, name), 0.0)
    assert np.all((_lookup(values, "g10_pair_same_local_peak_proxy_mean") >= 0.0))
    assert np.all((_lookup(values, "g10_pair_same_local_peak_proxy_mean") <= 1.0))

    bad = deepcopy(candidates)
    bad[0]["cx"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_candidate_relation_features(bad, image_shape=(100, 100))
    bad = deepcopy(candidates)
    bad[0]["width_px"] = -1
    with pytest.raises(ValueError, match="negative"):
        extract_candidate_relation_features(bad, image_shape=(100, 100))
    with pytest.raises(ValueError, match="candidate_mask"):
        extract_candidate_relation_features(candidates, image_shape=(100, 100), candidate_mask=[1, 2])
    with pytest.raises(ValueError, match="unique explicit q_rank"):
        duplicate_rank = [dict(candidates[0]), {**candidates[1], "q_rank": 0}]
        extract_candidate_relation_features(duplicate_rank, image_shape=(100, 100))
