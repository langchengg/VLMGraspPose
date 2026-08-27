"""Unit-only fixtures for the frozen 6-DoF candidate contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from graspnet6d.contracts import (
    Candidate6D,
    assert_frozen_candidate_pool,
    candidate_pool_fingerprint,
    load_candidate_cache,
    pose_nms,
    save_candidate_cache,
    stable_candidate_id,
)
from graspnet6d.geometry import vgn_to_graspnet_provenance
from graspnet6d.io import atomic_json, atomic_jsonl, canonical_sha256, sha256_file


def _candidate(
    rank: int,
    *,
    score: float,
    x_m: float,
    width_m: float = 0.04,
) -> Candidate6D:
    rotation = np.eye(3)
    translation = np.array([x_m, 0.10, 0.20])
    candidate_id = stable_candidate_id(
        "unit_group",
        translation_local_m=translation,
        rotation_local=rotation,
        width_m=width_m,
        height_m=0.02,
        depth_m=0.02,
        voxel_index=(rank, 2, 3),
    )
    return Candidate6D(
        candidate_id=candidate_id,
        group_id="unit_group",
        native_rank=rank,
        native_score=score,
        translation_local_m=translation,
        rotation_local=rotation,
        translation_camera_m=translation + np.array([0.0, 0.0, 0.30]),
        rotation_camera=rotation,
        translation_table_m=translation + np.array([0.40, 0.0, 0.0]),
        rotation_table=rotation,
        width_m=width_m,
        height_m=0.02,
        depth_m=0.02,
        voxel_index=(rank, 2, 3),
        conversion_provenance=vgn_to_graspnet_provenance(
            source="deterministic unit fixture; not an experiment result"
        ),
    )


def test_candidate_validation_and_stable_hashes() -> None:
    candidate = _candidate(0, score=0.90, x_m=0.01)
    rebuilt = Candidate6D.from_dict(candidate.to_dict())

    assert rebuilt == candidate
    assert rebuilt.geometry_sha256 == candidate.geometry_sha256
    assert rebuilt.record_sha256 == candidate.record_sha256
    assert len(candidate.geometry_sha256) == 64
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("native_score", float("nan")),
        ("width_m", 0.0),
        ("height_m", float("inf")),
        ("depth_m", -0.01),
        ("rotation_local", np.diag([1.0, 1.0, -1.0])),
    ],
)
def test_candidate_rejects_invalid_numeric_contract(field: str, bad_value: object) -> None:
    candidate = _candidate(0, score=0.90, x_m=0.01)
    with pytest.raises(ValueError):
        replace(candidate, **{field: bad_value})


def test_candidate_requires_explicit_conversion_provenance() -> None:
    candidate = _candidate(0, score=0.90, x_m=0.01)
    with pytest.raises(ValueError, match="missing fields"):
        replace(candidate, conversion_provenance={"source": "unit fixture"})


def test_candidate_cache_npz_json_round_trip(tmp_path) -> None:
    native = [
        _candidate(0, score=0.90, x_m=0.01),
        _candidate(1, score=0.80, x_m=0.04),
    ]
    npz_path, json_path = save_candidate_cache(
        tmp_path / "candidates.npz",
        native,
        metadata={"purpose": "unit test only"},
    )

    loaded, metadata = load_candidate_cache(npz_path)

    assert loaded == native
    assert metadata == {"purpose": "unit test only"}
    assert json_path.is_file()
    assert len(sha256_file(npz_path)) == 64
    assert candidate_pool_fingerprint(loaded) == candidate_pool_fingerprint(native)


def test_atomic_json_and_jsonl_are_complete(tmp_path) -> None:
    json_path = atomic_json(tmp_path / "nested" / "value.json", {"z": 1, "a": 2})
    jsonl_path = atomic_jsonl(
        tmp_path / "nested" / "rows.jsonl", [{"row": 1}, {"row": 2}]
    )

    assert json_path.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "z": 1\n}\n'
    assert jsonl_path.read_text(encoding="utf-8") == '{"row": 1}\n{"row": 2}\n'
    assert not list((tmp_path / "nested").glob("*.tmp"))


def test_pose_nms_is_deterministic_and_width_rule_is_explicit() -> None:
    best = _candidate(0, score=0.90, x_m=0.010, width_m=0.040)
    close_different_width = _candidate(1, score=0.80, x_m=0.011, width_m=0.060)
    far = _candidate(2, score=0.70, x_m=0.050, width_m=0.040)

    official_style = pose_nms(
        [far, close_different_width, best],
        translation_threshold_m=0.015,
        rotation_threshold_deg=15.0,
        width_threshold_m=None,
        top_k=50,
    )
    experiment_style = pose_nms(
        [far, close_different_width, best],
        translation_threshold_m=0.015,
        rotation_threshold_deg=15.0,
        width_threshold_m=0.010,
        top_k=50,
    )

    assert [item.candidate_id for item in official_style] == [
        best.candidate_id,
        far.candidate_id,
    ]
    assert [item.candidate_id for item in experiment_style] == [
        best.candidate_id,
        close_different_width.candidate_id,
        far.candidate_id,
    ]


def test_frozen_pool_accepts_only_reordering() -> None:
    first = _candidate(0, score=0.90, x_m=0.01)
    second = _candidate(1, score=0.80, x_m=0.04)

    assert_frozen_candidate_pool([first, second], [second, first])

    with pytest.raises(ValueError, match="membership changed"):
        assert_frozen_candidate_pool([first, second], [first])
    with pytest.raises(ValueError, match="geometry changed"):
        assert_frozen_candidate_pool(
            [first, second],
            [replace(first, translation_table_m=(0.99, 0.10, 0.20)), second],
        )
