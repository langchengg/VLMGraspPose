from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gtmask_counterfactual.io import canonical_sha256, sha256_file
from gtmask_counterfactual.manifest import build_counterfactual_manifest
from gtmask_counterfactual.mapping import (
    build_prelock_registry,
    mapping_pixel_qa,
)
from gtmask_counterfactual.mapping_pipeline import (
    run_mapping_pixel_qa_bulk,
    select_mapping_qa_cases,
    validate_manual_mapping_qa,
)
from tools.gtmask_counterfactual.build_counterfactual_manifest import (
    _canonical_denominator,
)


def _digest() -> str:
    return "a" * 64


def _manifest_row(sample_id: str) -> dict[str, object]:
    root = Path("/synthetic") / sample_id
    return {
        "sample_id": sample_id,
        "scene_id": "scene-a",
        "frame_id": "frame-a",
        "query_id": sample_id,
        "query_type": "name",
        "target_instance_id": 9,
        "gt_grasp_target_instance_id": 9,
        "language_prompt": f"pick {sample_id}",
        "rgb_path": str(root / "rgb.png"),
        "rgb_sha256": _digest(),
        "rgb_height": 480,
        "rgb_width": 640,
        "depth_path": str(root / "depth.png"),
        "depth_sha256": _digest(),
        "depth_height": 480,
        "depth_width": 640,
        "intrinsics_path": str(root / "intrinsics.json"),
        "intrinsics_sha256": _digest(),
        "prepared_gt_mask_path": str(root / "prepared.png"),
        "prepared_gt_mask_sha256": _digest(),
        "source_instance_mask_path": str(root / "instances.png"),
        "source_instance_mask_sha256": _digest(),
        "gt_grasp_set_path": str(root / "grasps.parquet"),
        "gt_grasp_set_sha256": _digest(),
    }


def test_manifest_preserves_unresolved_denominator_and_uses_no_row_join() -> None:
    rows = [_manifest_row("s1"), _manifest_row("s2")]
    route_sources = {}
    for route in ("g1", "c1", "d1"):
        route_sources[route] = [
            {**rows[0], "source_identity": f"{route}:s1"},
            # s2 is deliberately absent from C1 only.
            *([] if route == "c1" else [{**rows[1], "source_identity": f"{route}:s2"}]),
        ]
    result = build_counterfactual_manifest(rows, route_sources, expected_count=2)
    assert len(result.rows) == 2
    assert result.audit["counterfactual_evaluable_count"] == 1
    assert result.audit["unresolved_count"] == 1
    assert result.unresolved[0]["sample_id"] == "s2"
    assert result.audit["row_number_join_used"] is False


def test_canonical_denominator_reconciles_path_and_digest_frame_ids(
    tmp_path: Path,
) -> None:
    sample_id = "s1"
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    frame_digest = "f" * 64
    visual_path = tmp_path / "visual.parquet"
    labels_path = tmp_path / "labels.parquet"
    paired_path = tmp_path / "paired.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "scene_id": "scene-a",
                "frame_id": str(rgb_path),
                "query_id": 7,
                "target_instance_id": 9,
                "expression": "pick it",
                "expression_type": "name",
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "gt_mask_path": str(tmp_path / "instances.png"),
                "gt_grasp_list_json": "[]",
                "image_width": 640,
                "image_height": 480,
                "image_sha256": "1" * 64,
                "depth_sha256": "2" * 64,
                "rgbd_pair_sha256": frame_digest,
            }
        ]
    ).to_parquet(visual_path, index=False)
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "scene_id": "scene-a",
                "question_index": 7,
                "target_object_id": 9,
                "prepared_gt_mask_path": str(tmp_path / "prepared.png"),
                "prepared_gt_mask_sha256": "3" * 64,
                "official_annotations_path": str(tmp_path / "labels.json"),
                "official_annotations_sha256": "4" * 64,
            }
        ]
    ).to_parquet(labels_path, index=False)
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "scene_id": "scene-a",
                "frame_id": frame_digest,
                "question_index": 7,
                "source_rgb_path": str(rgb_path),
                "source_rgb_sha256": "1" * 64,
                "source_depth_path": str(depth_path),
                "source_depth_sha256": "2" * 64,
                "rgbd_pair_sha256": frame_digest,
                "language": "pick it",
                "intrinsics_path": str(tmp_path / "intrinsics.json"),
                "intrinsics_sha256": "5" * 64,
            }
        ]
    ).to_parquet(paired_path, index=False)

    rows = _canonical_denominator(
        visual_path, labels_path=labels_path, paired_path=paired_path
    )

    assert rows[0]["frame_id"] == frame_digest
    assert rows[0]["rgb_path"] == str(rgb_path.resolve())


def test_manual_mapping_qa_is_pending_then_requires_exact_signed_coverage() -> None:
    selected = [{"sample_id": "s1"}, {"sample_id": "s2"}]
    assert validate_manual_mapping_qa(selected, None)["status"] == "PENDING_MANUAL_QA"
    complete = []
    for sample_id in ("s1", "s2"):
        row = {
            "sample_id": sample_id,
            "review_status": "ANNOTATION_SUSPECT" if sample_id == "s2" else "PASS",
            "reviewer": "human@example.invalid",
            "reviewed_at_utc": "2026-08-13T12:00:00Z",
            "review_notes": "synthetic review",
        }
        row["review_signature"] = canonical_sha256(row)
        complete.append(row)
    result = validate_manual_mapping_qa(selected, complete)
    assert result["status"] == "PASS"
    assert result["annotation_suspect_sample_ids"] == ["s2"]
    assert validate_manual_mapping_qa(selected, complete[:1])["status"] == "FAIL"


def test_stratified_selection_is_deterministic_and_shared_across_routes() -> None:
    rows = []
    for index in range(32):
        rows.append(
            {
                "sample_id": f"s{index:03d}",
                "mapping_status": "PASS",
                "pixel_qa_status": "P2_MAPPING_QA_PASS",
                "query_type": "name" if index % 2 else "relation",
                "foreground_fraction": (index + 1) / 1000,
            }
        )
    first, audit = select_mapping_qa_cases(
        rows,
        minimum_total=16,
        minimum_per_query_type=4,
        minimum_per_quartile=4,
    )
    second, _ = select_mapping_qa_cases(
        rows,
        minimum_total=16,
        minimum_per_query_type=4,
        minimum_per_quartile=4,
    )
    assert audit["status"] == "PASS"
    assert audit["same_cases_shared_across_routes"] is True
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]


def test_bulk_pixel_qa_remains_pending_without_manual_csv(tmp_path: Path) -> None:
    sample_id = "s1"
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.zeros((480, 640), dtype=np.uint16)
    instances = np.zeros((480, 640), dtype=np.uint8)
    instances[40:140, 80:180] = 9
    target = instances == 9
    paths = {
        "rgb": tmp_path / "rgb.png",
        "depth": tmp_path / "depth.png",
        "instance": tmp_path / "instance.png",
        "prepared": tmp_path / "prepared.png",
        "intrinsics": tmp_path / "intrinsics.json",
        "grasps": tmp_path / "grasps.parquet",
    }
    Image.fromarray(rgb).save(paths["rgb"])
    Image.fromarray(depth).save(paths["depth"])
    Image.fromarray(instances).save(paths["instance"])
    Image.fromarray(target.astype(np.uint8) * 255).resize(
        (352, 352), Image.Resampling.NEAREST
    ).save(paths["prepared"])
    paths["intrinsics"].write_text("{}", encoding="utf-8")
    paths["grasps"].write_bytes(b"hash-bound identity authority only")
    manifest = {
        **_manifest_row(sample_id),
        "rgb_path": str(paths["rgb"]),
        "rgb_sha256": sha256_file(paths["rgb"]),
        "depth_path": str(paths["depth"]),
        "depth_sha256": sha256_file(paths["depth"]),
        "intrinsics_path": str(paths["intrinsics"]),
        "intrinsics_sha256": sha256_file(paths["intrinsics"]),
        "prepared_gt_mask_path": str(paths["prepared"]),
        "prepared_gt_mask_sha256": sha256_file(paths["prepared"]),
        "source_instance_mask_path": str(paths["instance"]),
        "source_instance_mask_sha256": None,
        "gt_grasp_set_path": str(paths["grasps"]),
        "gt_grasp_set_sha256": sha256_file(paths["grasps"]),
    }
    authority = {
        "sample_id": sample_id,
        "scene_id": "scene-a",
        "question_index": sample_id,
        "target_object_id": 9,
        "prepared_gt_mask_path": str(paths["prepared"]),
        "prepared_gt_mask_sha256": sha256_file(paths["prepared"]),
        "source_instance_mask_path": str(paths["instance"]),
        "source_instance_mask_sha256": None,
    }
    registry = build_prelock_registry(
        [{"sample_id": sample_id, "scene_id": "scene-a", "question_index": sample_id}],
        [authority],
        expected_count=1,
    )
    result = run_mapping_pixel_qa_bulk(
        [manifest],
        registry,
        run_dir=tmp_path / "run",
        expected_count=1,
        minimum_contact_cases=1,
        minimum_per_query_type=1,
        minimum_per_quartile=0,
    )
    assert result["status"] == "PENDING_MANUAL_QA"
    assert result["pixel_qa_status"] == "P2_MAPPING_QA_PENDING_MANUAL"
    assert result["candidate_generation_gt_mask_rows_read"] == 0
    assert result["qa_evidence"]["observed_resize_inverse_round_trip_min_iou"] >= 0.95
    registry = pd.read_parquet(result["outputs"]["gt_mask_registry"]["path"])
    assert registry.iloc[0]["source_instance_mask_hash_status"] == (
        "COMPUTED_UNDER_P2_AUTHORITY"
    )


def test_small_exact_forward_mask_is_not_rejected_by_lossy_inverse(
    tmp_path: Path,
) -> None:
    sample_id = "small-target"
    instances = np.zeros((480, 640), dtype=np.uint8)
    instances[101:108, 203:210] = 9
    expected = instances == 9
    instance_path = tmp_path / "instances.png"
    prepared_path = tmp_path / "prepared.png"
    Image.fromarray(instances).save(instance_path)
    Image.fromarray(expected.astype(np.uint8) * 255).resize(
        (352, 352), Image.Resampling.NEAREST
    ).save(prepared_path)

    result = mapping_pixel_qa(
        {
            "sample_id": sample_id,
            "mapping_status": "PATH_HASH_INSTANCE_AUTHORITY_MAPPED",
            "prepared_gt_mask_path": str(prepared_path),
            "prepared_gt_mask_sha256": sha256_file(prepared_path),
            "source_instance_mask_path": str(instance_path),
            "source_instance_mask_sha256": sha256_file(instance_path),
            "rgb_height": 480,
            "rgb_width": 640,
            "target_instance_id": 9,
            "gt_grasp_target_instance_id": 9,
            "gt_grasp_set_sha256": "b" * 64,
        },
        access_authority={
            "status": "AUTHORIZED",
            "stage": "P2_GT_MAPPING_PASS",
            "purpose": "mapping_and_annotation_pixel_qa_only",
            "candidate_generation_allowed": False,
        },
        derived_output_dir=tmp_path / "derived",
    )

    assert result["mapping_status"] == "PASS"
    assert result["resize_inverse_round_trip_iou"] < 0.95
    assert result["resize_inverse_round_trip_below_reference"] is True
