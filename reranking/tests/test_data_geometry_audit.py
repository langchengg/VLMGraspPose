from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from reranking import geometry
from reranking.audit_runs import (
    ProvenanceAuditError,
    audit_run,
    manifest_file_inventory,
    write_audit_bundle,
)
from reranking.data_contracts import (
    CANONICAL_PATHS,
    HIFI_CANONICAL_LABELS,
    KNOWN_HASHES,
    REPO_ROOT,
    assert_candidate_keys_and_source_order,
    candidate_key_sequence,
    file_inventory,
    streaming_sha256,
)


def test_canonical_paths_and_hash_contract_are_repository_relative() -> None:
    assert REPO_ROOT.name == "VLMGraspPose"
    assert CANONICAL_PATHS["hifi_label_evaluator"] == HIFI_CANONICAL_LABELS
    assert HIFI_CANONICAL_LABELS.is_file()
    relative = HIFI_CANONICAL_LABELS.relative_to(REPO_ROOT).as_posix()
    assert len(KNOWN_HASHES[relative]) == 64


def test_streaming_sha256_matches_standard_digest(tmp_path: Path) -> None:
    payload = bytes(range(251)) * 37
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)
    assert streaming_sha256(source, chunk_size=17) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="chunk_size"):
        streaming_sha256(source, chunk_size=0)


def test_file_inventory_is_sorted_and_does_not_follow_symlinks(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "a.txt")
    inventory = file_inventory(tmp_path, hash_files=True)
    assert [row["relative_path"] for row in inventory] == ["a.txt", "b.txt", "link"]
    link = inventory[-1]
    assert link["type"] == "symlink"
    assert link["sha256"] is None
    with pytest.raises(ValueError, match="symlink"):
        streaming_sha256(tmp_path / "link")


def test_candidate_keys_reject_duplicates_and_missing_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        candidate_key_sequence(
            [
                {"sample_id": "q", "candidate_id": "a"},
                {"sample_id": "q", "candidate_id": "a"},
            ]
        )
    with pytest.raises(ValueError, match="lacks"):
        candidate_key_sequence([{"sample_id": "q"}])


def test_candidate_source_order_is_strict_within_query() -> None:
    reference = [
        {"sample_id": "q1", "candidate_id": "a"},
        {"sample_id": "q1", "candidate_id": "b"},
        {"sample_id": "q2", "candidate_id": "c"},
        {"sample_id": "q2", "candidate_id": "d"},
    ]
    query_blocks_reordered = reference[2:] + reference[:2]
    result = assert_candidate_keys_and_source_order(reference, query_blocks_reordered)
    assert result["source_order_equal_within_query"] is True
    assert result["global_order_equal"] is False
    with pytest.raises(ValueError, match="global query/source order"):
        assert_candidate_keys_and_source_order(
            reference, query_blocks_reordered, require_query_order=True
        )
    within_query_reordered = [reference[1], reference[0], *reference[2:]]
    with pytest.raises(ValueError, match="source order changed"):
        assert_candidate_keys_and_source_order(reference, within_query_reordered)
    interleaved = [reference[0], reference[2], reference[1], reference[3]]
    with pytest.raises(ValueError, match="non-contiguous"):
        assert_candidate_keys_and_source_order(reference, interleaved)


def test_candidate_key_set_mismatch_is_rejected() -> None:
    reference = [{"sample_id": "q", "candidate_id": "a"}]
    observed = [{"sample_id": "q", "candidate_id": "b"}]
    with pytest.raises(ValueError, match="candidate key set changed"):
        assert_candidate_keys_and_source_order(reference, observed)


def _write_manifest_run(root: Path, *, correct_hash: bool = True) -> Path:
    root.mkdir()
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"canonical bytes")
    digest = streaming_sha256(artifact)
    manifest = {
        "artifacts": [
            {
                "path": "artifact.bin",
                "sha256": digest if correct_hash else "0" * 64,
                "bytes": artifact.stat().st_size,
            }
        ]
    }
    manifest_path = root / "final_output_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_manifest_inventory_checks_paths_sizes_and_hashes(tmp_path: Path) -> None:
    manifest = _write_manifest_run(tmp_path / "run")
    result = manifest_file_inventory(manifest)
    assert result["unique_file_count"] == 1
    assert result["all_present"] is True
    assert result["sizes_match"] is True
    assert result["hashes_match"] is True
    assert result["passed"] is True


def test_manifest_inventory_detects_hash_mismatch_and_missing_file(tmp_path: Path) -> None:
    manifest = _write_manifest_run(tmp_path / "bad", correct_hash=False)
    mismatched = manifest_file_inventory(manifest)
    assert mismatched["hashes_match"] is False
    assert mismatched["passed"] is False
    (manifest.parent / "artifact.bin").unlink()
    missing = manifest_file_inventory(manifest)
    assert missing["all_present"] is False
    assert missing["passed"] is False


def test_manifest_and_manifest_artifact_symlinks_are_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest_run(tmp_path / "real")
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(manifest)
    with pytest.raises(ProvenanceAuditError, match="not a regular file"):
        manifest_file_inventory(manifest_link)

    artifact = manifest.parent / "artifact.bin"
    target = manifest.parent / "target.bin"
    artifact.rename(target)
    artifact.symlink_to(target)
    result = manifest_file_inventory(manifest)
    assert result["inventory"][0]["symlink"] is True
    assert result["passed"] is False


def test_audit_run_is_read_only_and_requires_completion(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_manifest_run(run)
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    incomplete = audit_run(
        run,
        manifest_names=("final_output_manifest.json",),
        completion_markers=("COMPLETED",),
    )
    assert incomplete["passed"] is False
    (run / "COMPLETED").write_text("", encoding="utf-8")
    complete = audit_run(
        run,
        manifest_names=("final_output_manifest.json",),
        completion_markers=("COMPLETED",),
    )
    assert complete["passed"] is True
    after = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    assert {key: after[key] for key in before} == before


def test_audit_bundle_only_writes_three_files_to_new_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "immutable.txt"
    source_file.write_text("unchanged", encoding="utf-8")
    report = {
        "passed": True,
        "inventory": [
            {
                "path": str(source_file),
                "exists": True,
                "size_bytes": source_file.stat().st_size,
                "sha256": streaming_sha256(source_file),
            }
        ],
    }
    output = tmp_path / "new-audit"
    written = write_audit_bundle(report, output, source_roots=(source,))
    assert set(path.name for path in output.iterdir()) == {
        "audit.json",
        "audit.md",
        "inventory.tsv",
    }
    assert Path(written["json"]).is_file()
    assert source_file.read_text(encoding="utf-8") == "unchanged"
    with pytest.raises(FileExistsError):
        write_audit_bundle(report, output, source_roots=(source,))
    with pytest.raises(ProvenanceAuditError, match="overlaps"):
        write_audit_bundle(report, source / "nested-output", source_roots=(source,))
    with pytest.raises(FileNotFoundError, match="parent directory"):
        write_audit_bundle(report, tmp_path / "missing-parent" / "audit")


def test_geometry_module_delegates_to_canonical_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert geometry.canonical_evaluator_path() == HIFI_CANONICAL_LABELS
    sentinel = object()
    monkeypatch.setattr(geometry._canonical, "is_positive_pair", lambda *a, **k: sentinel)
    assert geometry.is_positive_pair(0.9, 0.0) is sentinel


@pytest.mark.parametrize(
    ("iou", "angle", "expected"),
    [
        (0.25, 0.0, False),
        (math.nextafter(0.25, 1.0), 0.0, True),
        (math.nextafter(0.25, 0.0), 0.0, False),
        (1.0, 30.0, True),
        (1.0, math.nextafter(30.0, 31.0), False),
        (float("nan"), 0.0, False),
    ],
)
def test_strict_iou_and_inclusive_angle_boundaries(
    iou: float, angle: float, expected: bool
) -> None:
    assert geometry.is_positive_pair(iou, angle) is expected


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (0.0, math.pi, 0.0),
        (0.0, 2.0 * math.pi, 0.0),
        (math.radians(179.0), math.radians(1.0), 2.0),
        (math.radians(-89.0), math.radians(89.0), 2.0),
        (0.0, math.pi / 2.0, 90.0),
        (math.radians(30.0), math.radians(-150.0), 0.0),
    ],
)
def test_parallel_jaw_angle_is_180_degree_periodic(
    first: float, second: float, expected: float
) -> None:
    assert geometry.periodic_angle_difference_deg(first, second) == pytest.approx(expected)


def test_raster_coordinates_are_row_y_column_x() -> None:
    width = 30
    pixels = geometry.raster_pixels(
        center_uv=[20.0, 5.0],
        width_px=4.0,
        height_px=2.0,
        angle_rad=0.0,
        shape=(20, width),
    )
    rows, columns = pixels // width, pixels % width
    assert np.median(rows) == pytest.approx(5.0)
    assert np.median(columns) == pytest.approx(20.0)


def test_raster_x_and_y_shifts_affect_only_the_corresponding_axis() -> None:
    width = 40

    def center(pixels: np.ndarray) -> tuple[float, float]:
        return float(np.median(pixels // width)), float(np.median(pixels % width))

    base = geometry.raster_pixels(
        center_uv=[20.0, 10.0], width_px=4.0, height_px=2.0, angle_rad=0.0, shape=(30, width)
    )
    x_shift = geometry.raster_pixels(
        center_uv=[23.0, 10.0], width_px=4.0, height_px=2.0, angle_rad=0.0, shape=(30, width)
    )
    y_shift = geometry.raster_pixels(
        center_uv=[20.0, 13.0], width_px=4.0, height_px=2.0, angle_rad=0.0, shape=(30, width)
    )
    assert center(x_shift) == pytest.approx((center(base)[0], center(base)[1] + 3))
    assert center(y_shift) == pytest.approx((center(base)[0] + 3, center(base)[1]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center_uv": [5.0, 5.0], "width_px": 0.0, "height_px": 2.0, "angle_rad": 0.0},
        {"center_uv": [5.0, 5.0], "width_px": 2.0, "height_px": 0.0, "angle_rad": 0.0},
        {"center_uv": [math.nan, 5.0], "width_px": 2.0, "height_px": 2.0, "angle_rad": 0.0},
        {"center_uv": [5.0], "width_px": 2.0, "height_px": 2.0, "angle_rad": 0.0},
        {"center_uv": [5.0, 5.0], "width_px": 2.0, "height_px": 2.0, "angle_rad": math.inf},
    ],
)
def test_degenerate_or_nonfinite_raster_rectangles_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        geometry.raster_pixels(**kwargs)


def test_degenerate_polygon_and_empty_rasters_have_zero_iou() -> None:
    degenerate = np.zeros((4, 2), dtype=np.float32)
    square = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    assert geometry.polygon_iou(degenerate, square) == 0.0
    empty = np.asarray([], dtype=np.int64)
    assert geometry.raster_pixel_iou(empty, empty) == 0.0


def _candidate() -> dict:
    return {
        "candidate_id": "g0000",
        "center_uv": [50.0, 50.0],
        "center_depth_m": 1.0,
        "center_camera_xyz_m": [0.0, 0.0, 1.0],
        "angle_rad": 0.0,
        "width_m": 0.05,
        "width_px": 20.0,
        "contact_points_uv": [[40.0, 50.0], [60.0, 50.0]],
        "endpoints_uv": [[40.0, 50.0], [60.0, 50.0]],
    }


def test_equal_gt_tie_uses_stable_first_gt_index() -> None:
    rectangle = [[40.0, 45.0], [40.0, 55.0], [60.0, 55.0], [60.0, 45.0]]
    label = geometry.evaluate_candidate_label(
        _candidate(),
        [rectangle, rectangle],
        {
            "iou_threshold": 0.25,
            "angle_threshold_deg": 30.0,
            "predicted_rectangle_height_px": 10.0,
            "ground_truth_rectangle_height_px": 10.0,
            "ground_truth_width_clip_px": 100.0,
        },
    )
    assert label.candidate_positive is True
    assert label.best_gt_id == 0
