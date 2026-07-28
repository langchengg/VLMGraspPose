from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.grasping.gqcnn_full_scoring import (
    MARKER_NAME,
    SCORED_NONEMPTY,
    SKIPPED_VALID_EMPTY,
    ScoringValidationError,
    _rank_indices,
    atomic_commit_sample,
    assert_disjoint_roots,
    build_source_manifest,
    make_staging_directory,
    cleanup_stale_staging,
    deterministic_save_npz,
    score_and_write_sample,
    select_entries,
    sha256_file,
    source_manifest_entry,
    load_source_sample,
    validate_scored_output,
    write_empty_sample,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_ROOT = ROOT / "outputs" / "dexnet_candidates_full_hifics"
NONEMPTY_ID = "q0000000_b32eb3299dcd3ae9"
EMPTY_ID = "q0002096_ba63e6cf5bdc38a5"
SOURCE_FILES = (
    "_SUCCESS.json",
    "metadata.json",
    "candidates.json",
    "candidates.npz",
    "camera.intr",
    "depth_m.npy",
    "hifics_mask_processed.png",
)


def model_info():
    return {
        "model_name": "GQCNN-2.1",
        "model_commit": "499a609fe9dfb074bdfb6c4e6e33667ea50f4c21",
        "model_config_hash": "config-hash",
        "model_file_manifest_hash": "model-hash",
        "docker_image": "test/image",
        "docker_image_id": "sha256:test",
    }


def copy_source(tmp_path: Path, sample_id: str) -> Path:
    destination = tmp_path / "source" / sample_id
    destination.mkdir(parents=True)
    for name in SOURCE_FILES:
        shutil.copy2(FULL_ROOT / sample_id / name, destination / name)
    return destination


def entry_for(source_dir: Path, index: int = 0) -> dict:
    return source_manifest_entry(load_source_sample(source_dir), index)


def fake_state_builder(sample_dir, arrays, records, inpaint_rescale_factor):
    del sample_dir, inpaint_rescale_factor
    return object(), list(range(len(records))), {"frame": "ocid_camera_optical"}


class FakeQuality:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def __call__(self, state, grasps, params=None):
        del state, params
        assert len(grasps) == len(self.values)
        return self.values.copy()


def test_raw_full_precision_sort_and_candidate_id_tie_break():
    values = np.asarray([0.5, 0.5, np.nextafter(0.5, 1.0)])
    order = _rank_indices(values, ["g0002", "g0001", "g0003"])
    assert order == [2, 1, 0]


def test_source_manifest_gate_and_valid_empty(tmp_path: Path):
    source_root = tmp_path / "source"
    nonempty = copy_source(tmp_path, NONEMPTY_ID)
    empty = copy_source(tmp_path, EMPTY_ID)
    rows = [
        {
            "sample_id": NONEMPTY_ID,
            "post_nms_count": "26",
            "status": "success_nonempty",
        },
        {
            "sample_id": EMPTY_ID,
            "post_nms_count": "0",
            "status": "success_empty",
        },
    ]
    import csv

    with (source_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "scored" / "source_candidate_manifest.jsonl"
    entries, identity = build_source_manifest(
        source_root,
        manifest,
        expected_samples=2,
        expected_nonempty=1,
        expected_empty=1,
        expected_candidates=26,
    )
    assert identity["candidate_count"] == 26
    assert [item["sample_id"] for item in entries] == [NONEMPTY_ID, EMPTY_ID]
    assert len(entries[0]["candidate_ids"]) == 26
    assert entries[1]["empty_reason"]

    output = tmp_path / "scored"
    staging = make_staging_directory(output, EMPTY_ID)
    write_empty_sample(staging, entries[1], model_info(), seed=42)
    final = output / EMPTY_ID
    atomic_commit_sample(staging, final, output)
    valid, status, marker, errors = validate_scored_output(
        final, empty, entries[1], model_info(), 42, verify_hashes=True
    )
    assert (valid, status, errors) == (True, SKIPPED_VALID_EMPTY, [])
    assert marker["top1_candidate_id"] is None
    assert not (final / "gqcnn_scored_candidates.npz").exists()


def test_score_preserves_source_pose_and_builds_exact_ranking(tmp_path: Path):
    source_dir = copy_source(tmp_path, NONEMPTY_ID)
    entry = entry_for(source_dir)
    before = sha256_file(source_dir / "candidates.npz")
    values = np.linspace(0.0, 0.9, entry["candidate_count"], dtype=np.float64)
    values[0] = 0.5
    values[1] = 0.5
    output = tmp_path / "scored"
    staging = make_staging_directory(output, NONEMPTY_ID)
    marker = score_and_write_sample(
        staging,
        source_dir,
        entry,
        FakeQuality(values),
        fake_state_builder,
        model_info(),
        seed=42,
    )
    assert marker["scoring_status"] == SCORED_NONEMPTY
    final = output / NONEMPTY_ID
    atomic_commit_sample(staging, final, output)
    valid, status, _, errors = validate_scored_output(
        final, source_dir, entry, model_info(), 42, verify_hashes=True
    )
    assert (valid, status, errors) == (True, SCORED_NONEMPTY, [])
    assert sha256_file(source_dir / "candidates.npz") == before

    with np.load(source_dir / "candidates.npz", allow_pickle=False) as source, np.load(
        final / "gqcnn_scored_candidates.npz", allow_pickle=False
    ) as scored:
        for name in source.files:
            if name == "gqcnn_q_value":
                continue
            assert source[name].dtype == scored[name].dtype
            assert np.array_equal(source[name], scored[name], equal_nan=True)
        assert scored["gqcnn_q_value"].dtype == np.float64
        assert scored["gqcnn_rank"].dtype == np.int32

    payload = json.loads((final / "gqcnn_scored_candidates.json").read_text())
    ranked = payload["candidates"]
    expected = sorted(
        range(len(values)),
        key=lambda index: (-values[index], entry["candidate_ids"][index]),
    )
    assert [item["source_candidate_index"] for item in ranked] == expected
    assert [item["gqcnn_rank"] for item in ranked] == list(range(1, len(values) + 1))
    assert all(np.isfinite(item["gqcnn_q_value"]) for item in ranked)
    assert len(json.loads((final / "gqcnn_top5.json").read_text())["candidates"]) == 5


def test_nonfinite_score_rejected_before_commit(tmp_path: Path):
    source_dir = copy_source(tmp_path, NONEMPTY_ID)
    entry = entry_for(source_dir)
    values = np.ones(entry["candidate_count"], dtype=np.float64)
    values[3] = np.nan
    staging = make_staging_directory(tmp_path / "scored", NONEMPTY_ID)
    with pytest.raises(ScoringValidationError, match="finite"):
        score_and_write_sample(
            staging,
            source_dir,
            entry,
            FakeQuality(values),
            fake_state_builder,
            model_info(),
            seed=42,
        )


def test_corrupt_rank_and_model_hash_are_rejected(tmp_path: Path):
    source_dir = copy_source(tmp_path, NONEMPTY_ID)
    entry = entry_for(source_dir)
    values = np.linspace(0.0, 1.0, entry["candidate_count"])
    output = tmp_path / "scored"
    staging = make_staging_directory(output, NONEMPTY_ID)
    score_and_write_sample(
        staging,
        source_dir,
        entry,
        FakeQuality(values),
        fake_state_builder,
        model_info(),
        seed=42,
    )
    final = output / NONEMPTY_ID
    atomic_commit_sample(staging, final, output)
    marker_path = final / MARKER_NAME
    marker = json.loads(marker_path.read_text())
    marker["model_config_hash"] = "wrong"
    marker_path.write_text(json.dumps(marker))
    valid, _, _, errors = validate_scored_output(
        final, source_dir, entry, model_info(), 42, verify_hashes=False
    )
    assert not valid
    assert any("model_config_hash mismatch" in value for value in errors)


def test_range_and_sharding_have_exact_union():
    entries = [{"sample_id": "s%d" % value} for value in range(17)]
    shards = [
        select_entries(entries, num_shards=3, shard_index=index)
        for index in range(3)
    ]
    ids = [item["sample_id"] for shard in shards for item in shard]
    assert sorted(ids) == sorted(item["sample_id"] for item in entries)
    assert len(ids) == len(set(ids))
    assert [item["sample_id"] for item in select_entries(entries, start_index=3, end_index=7)] == [
        "s3", "s4", "s5", "s6"
    ]


def test_source_and_output_roots_must_be_disjoint(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        assert_disjoint_roots(source, source)
    with pytest.raises(ValueError, match="disjoint"):
        assert_disjoint_roots(source, source / "scored")
    with pytest.raises(ValueError, match="disjoint"):
        assert_disjoint_roots(source, tmp_path)
    assert_disjoint_roots(source, tmp_path / "scored")


def test_stale_staging_for_selected_sample_is_removed(tmp_path: Path):
    output = tmp_path / "scored"
    selected = make_staging_directory(output, "sample_a")
    other = make_staging_directory(output, "sample_b")
    removed = cleanup_stale_staging(output, ["sample_a"])
    assert removed == [selected.name]
    assert not selected.exists()
    assert other.exists()


def test_depth_and_mask_are_part_of_source_identity(tmp_path: Path):
    source_dir = copy_source(tmp_path, NONEMPTY_ID)
    entry = entry_for(source_dir)
    assert entry["source_hashes"]["depth_m_sha256"]
    assert entry["source_hashes"]["processed_mask_sha256"]
    depth = source_dir / "depth_m.npy"
    depth.write_bytes(depth.read_bytes() + b"corrupt")
    with pytest.raises(ScoringValidationError, match="depth_m.npy"):
        load_source_sample(source_dir, verify_hashes=True)


def _committed_fake_score(tmp_path: Path) -> tuple[Path, Path, dict]:
    source_dir = copy_source(tmp_path, NONEMPTY_ID)
    entry = entry_for(source_dir)
    output = tmp_path / "scored"
    staging = make_staging_directory(output, NONEMPTY_ID)
    values = np.linspace(0.0, 1.0, entry["candidate_count"], dtype=np.float64)
    score_and_write_sample(
        staging,
        source_dir,
        entry,
        FakeQuality(values),
        fake_state_builder,
        model_info(),
        seed=42,
    )
    final = output / NONEMPTY_ID
    atomic_commit_sample(staging, final, output)
    return source_dir, final, entry


def _refresh_required_hash(final: Path, name: str) -> None:
    marker_path = final / MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["required_file_hashes"][name] = sha256_file(final / name)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("missing_q", "NPZ keys mismatch"),
        ("pose_change", "center_uv values differ"),
        ("wrong_rank", "ranks do not match"),
    ],
)
def test_verifier_rejects_missing_q_pose_change_and_wrong_rank(
    tmp_path: Path, mutation: str, expected_error: str
):
    source_dir, final, entry = _committed_fake_score(tmp_path)
    npz_path = final / "gqcnn_scored_candidates.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if mutation == "missing_q":
        arrays.pop("gqcnn_q_value")
    elif mutation == "pose_change":
        arrays["center_uv"][0, 0] += 1
    else:
        arrays["gqcnn_rank"] = arrays["gqcnn_rank"][::-1].copy()
    deterministic_save_npz(npz_path, arrays)
    _refresh_required_hash(final, "gqcnn_scored_candidates.npz")
    valid, _, _, errors = validate_scored_output(
        final, source_dir, entry, model_info(), 42, verify_hashes=True
    )
    assert not valid
    assert any(expected_error in error for error in errors)


def test_verifier_rejects_semantically_wrong_csv_even_with_fresh_hash(tmp_path: Path):
    import csv

    source_dir, final, entry = _committed_fake_score(tmp_path)
    csv_path = final / "gqcnn_scored_candidates.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    rows[0]["gqcnn_q_value"] = "0.123456789"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_required_hash(final, "gqcnn_scored_candidates.csv")
    valid, _, _, errors = validate_scored_output(
        final, source_dir, entry, model_info(), 42, verify_hashes=True
    )
    assert not valid
    assert "CSV candidate values mismatch" in errors
