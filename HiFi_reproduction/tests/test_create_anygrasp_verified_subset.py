import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tools.create_anygrasp_verified_subset import create_verified_subset
except ModuleNotFoundError:
    create_verified_subset = None


ARTIFACTS = (
    "color.png",
    "depth.png",
    "target_mask.png",
    "target_probability.npy",
    "language.txt",
    "intrinsics.json",
    "metadata.json",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pcd(path, depth, K):
    height, width = depth.shape
    fx, fy, cx, cy = K
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth.astype(np.float32) / 1000.0
    records = np.empty(
        height * width,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("rgba", "<u4"),
                ("label", "<u4"),
            ]
        ),
    )
    records["x"] = (((xx - cx) * z / fx).ravel())
    records["y"] = (((yy - cy) * z / fy).ravel())
    records["z"] = z.ravel()
    records["rgba"] = np.uint32(0xFF507828)
    records["label"] = 0
    header = (
        "VERSION 0.7\nFIELDS x y z rgba label\nSIZE 4 4 4 4 4\n"
        "TYPE F F F U U\nCOUNT 1 1 1 1 1\n"
        f"WIDTH {width}\nHEIGHT {height}\nPOINTS {width * height}\nDATA binary\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + records.tobytes())


def _fixture(tmp_path, *, empty_mask_sample=None, sparse_depth_sample=None):
    run_dir = tmp_path / "run"
    source_root = tmp_path / "OCID-VLG"
    predictions = run_dir / "predictions"
    anygrasp = run_dir / "anygrasp_input_predicted_mask"
    metrics_path = run_dir / "evaluation" / "per_sample_metrics.csv"
    frozen_path = run_dir / "ocid_vlg_test.json"
    prediction_manifest_path = predictions / "export_manifest.json"
    anygrasp_manifest_path = anygrasp / "manifest.jsonl"
    metrics_path.parent.mkdir(parents=True)
    predictions.mkdir()
    anygrasp.mkdir()
    K = (50.0, 55.0, 3.5, 2.5)
    height, width = 6, 8
    frozen = []
    original = []
    prediction_rows = []
    anygrasp_rows = []
    metric_rows = []
    for index in range(30):
        stable_id = f"q{index:07d}_fixture{index:02d}"
        category = f"category_{index}"
        query = (
            f"Pick the {category} left of the other object"
            if index >= 20
            else f"Pick the {category}"
        )
        sequence = f"ARID20/table/top/seq{index + 1:02d}"
        image_name = f"frame_{index:02d}.png"
        scene_id = f"{sequence},{image_name}"
        frozen.append(
            {
                "num": index,
                "question_index": index,
                "scene_id": scene_id,
                "text": query,
            }
        )
        original.append(
            {
                "question_index": index,
                "image_filename": scene_id,
                "question": query,
                "target": f"{category}_1",
                "answer": 1,
            }
        )
        rgb = np.full((height, width, 3), [40, 120, 80], dtype=np.uint8)
        depth = np.full((height, width), 1000, dtype=np.uint16)
        if index == sparse_depth_sample:
            depth[1, 2] = 0
        instance_map = np.zeros((height, width), dtype=np.uint8)
        clutter = min(1 + index // 3, width)
        for instance in range(1, clutter + 1):
            instance_map[:, instance - 1] = instance
        source_rgb = source_root / sequence / "rgb" / image_name
        source_depth = source_root / sequence / "depth" / image_name
        source_instance = source_root / sequence / "seg_mask_instances_combi" / image_name
        source_pcd = source_root / sequence / "pcd" / Path(image_name).with_suffix(".pcd")
        for path in (source_rgb, source_depth, source_instance):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(source_rgb)
        Image.fromarray(depth).save(source_depth)
        Image.fromarray(instance_map).save(source_instance)
        _write_pcd(source_pcd, depth, K)

        prediction_dir = predictions / stable_id
        prediction_dir.mkdir()
        prediction_metadata = {
            "stable_sample_id": stable_id,
            "sample_index": index,
            "question_index": index,
            "scene_id": scene_id,
            "query": query,
        }
        (prediction_dir / "sample_metadata.json").write_text(
            json.dumps(prediction_metadata)
        )
        prediction_rows.append(
            {
                "stable_sample_id": stable_id,
                "sample_index": index,
                "question_index": index,
                "scene_id": scene_id,
                "query": query,
                "export_directory": f"predictions/{stable_id}",
            }
        )

        bundle = anygrasp / stable_id
        bundle.mkdir()
        Image.fromarray(rgb).save(bundle / "color.png")
        Image.fromarray(depth).save(bundle / "depth.png")
        target_mask = np.zeros((height, width), dtype=np.uint8)
        if index != empty_mask_sample:
            target_mask[1:5, 2:6] = 255
        Image.fromarray(target_mask).save(bundle / "target_mask.png")
        np.save(
            bundle / "target_probability.npy",
            np.full((height, width), 0.75, dtype=np.float32),
        )
        (bundle / "language.txt").write_text(query)
        intrinsics = {
            "source": "derived_from_organized_pcd",
            "factory_calibration": False,
            "fx": K[0],
            "fy": K[1],
            "cx": K[2],
            "cy": K[3],
            "width": width,
            "height": height,
            "depth_scale": 1000.0,
            "depth_unit": "millimetres",
            "pcd_coordinate_unit": "metres",
            "fit_rmse_px": 0.0,
            "fit_p95_px": 0.0,
            "depth_pcd_abs_p95_mm": 0.0,
            "depth_scale_verified": True,
        }
        (bundle / "intrinsics.json").write_text(json.dumps(intrinsics))
        (bundle / "metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": stable_id,
                    "question_index": index,
                    "scene_id": scene_id,
                    "query": query,
                    "ready_for_anygrasp": True,
                    "oracle_artifacts_exported": False,
                    "anygrasp_inference_ran": False,
                }
            )
        )
        (bundle / "checksums.sha256").write_text(
            "".join(f"{_sha256(bundle / name)}  {name}\n" for name in ARTIFACTS)
        )
        anygrasp_rows.append(
            {
                "sample_id": stable_id,
                "question_index": index,
                "scene_id": scene_id,
                "query": query,
                "ready": True,
                "ready_for_anygrasp": True,
                "blockers": [],
                "fit_rmse_px": 0.0,
                "fit_p95_px": 0.0,
                "depth_pcd_abs_p95_mm": 0.0,
                "output_dir": str(bundle),
            }
        )
        metric_rows.append(
            {
                "sample_index": index,
                "sample_id": index,
                "text": query,
                "rgb_path": f"processed/{index}.png",
                "mask_path": f"processed/{index}.png",
                "iou": index / 29,
                "inference_seconds": 0.01,
            }
        )

    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    frozen_path.write_text(json.dumps(frozen))
    prediction_manifest_path.write_text(json.dumps({"samples": prediction_rows}))
    anygrasp_manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in anygrasp_rows)
    )
    original_path = source_root / "refer" / "unique" / "test_expressions.json"
    original_path.parent.mkdir(parents=True)
    original_path.write_text(json.dumps({"info": {}, "data": original}))
    return {
        "run_dir": run_dir,
        "source_root": source_root,
        "metrics": metrics_path,
        "frozen": frozen_path,
        "prediction_manifest": prediction_manifest_path,
        "anygrasp_manifest": anygrasp_manifest_path,
        "original": original_path,
    }


def test_selects_exact_verified_groups_and_materializes_without_symlinks(tmp_path):
    assert create_verified_subset is not None, "verified subset tool is not implemented"
    fixture = _fixture(tmp_path)
    report = tmp_path / "reports/anygrasp_verified_subset_report.md"

    result = create_verified_subset(**fixture, report_path=report)

    assert result["status"] == "DONE"
    assert result["selected"] == 20
    assert result["group_counts"] == {
        "highest_iou": 5,
        "nearest_median": 5,
        "lowest_iou": 5,
        "diverse_clutter_spatial": 5,
    }
    rows = result["rows"]
    assert len({row["sample_id"] for row in rows}) == 20
    assert [row["sample_index"] for row in rows if row["selection_group"] == "highest_iou"] == [29, 28, 27, 26, 25]
    assert [row["sample_index"] for row in rows if row["selection_group"] == "lowest_iou"] == [0, 1, 2, 3, 4]
    diverse = [row for row in rows if row["selection_group"] == "diverse_clutter_spatial"]
    assert all(row["spatial_language"] for row in diverse)
    assert len({row["scene_id"] for row in diverse}) == 5
    assert len({row["target_category"] for row in diverse}) == 5
    assert all(row["target_point_count"] > 0 for row in rows)
    assert all(0 < row["target_valid_point_fraction"] <= 1 for row in rows)
    output = fixture["run_dir"] / "anygrasp_verified_subset"
    assert (output / "selection_manifest.csv").is_file()
    assert (output / "selection_manifest.jsonl").is_file()
    assert (output / "index.html").is_file()
    for row in rows:
        bundle = output / row["sample_id"]
        assert bundle.is_dir()
        assert not any(path.is_symlink() for path in bundle.iterdir())
    report_text = report.read_text()
    assert "Target visual correspondence requires human inspection" in report_text
    assert "Factory calibration and camera/robot extrinsics are unavailable" in report_text
    assert "Full-DoF AnyGrasp generation remains blocked" in report_text


def test_validation_only_performs_selection_without_writing_outputs(tmp_path):
    assert create_verified_subset is not None, "verified subset tool is not implemented"
    fixture = _fixture(tmp_path)
    report = tmp_path / "reports/anygrasp_verified_subset_report.md"

    result = create_verified_subset(
        **fixture, report_path=report, validation_only=True
    )

    assert result["status"] == "DONE"
    assert result["selected"] == 20
    assert not (fixture["run_dir"] / "anygrasp_verified_subset").exists()
    assert not report.exists()


def test_blocks_materialization_when_a_selected_target_mask_is_empty(tmp_path):
    assert create_verified_subset is not None, "verified subset tool is not implemented"
    fixture = _fixture(tmp_path, empty_mask_sample=29)
    report = tmp_path / "reports/anygrasp_verified_subset_report.md"

    result = create_verified_subset(**fixture, report_path=report)

    assert result["status"] == "BLOCKED"
    assert result["selected"] == 20
    failed = next(row for row in result["rows"] if row["sample_index"] == 29)
    assert "target_mask_empty" in failed["blockers"]
    assert not (fixture["run_dir"] / "anygrasp_verified_subset").exists()


def test_sparse_invalid_depth_inside_mask_is_reported_but_not_blocked(tmp_path):
    assert create_verified_subset is not None, "verified subset tool is not implemented"
    fixture = _fixture(tmp_path, sparse_depth_sample=29)

    result = create_verified_subset(
        **fixture,
        report_path=tmp_path / "reports/anygrasp_verified_subset_report.md",
        validation_only=True,
    )

    assert result["status"] == "DONE"
    sample = next(row for row in result["rows"] if row["sample_index"] == 29)
    assert sample["target_point_count"] == 15
    assert sample["target_mask_pixel_count"] == 16
    assert sample["target_valid_point_fraction"] == 15 / 16
    assert sample["blockers"] == []


def test_blocks_when_bundle_metadata_query_disagrees_with_frozen_manifest(tmp_path):
    assert create_verified_subset is not None, "verified subset tool is not implemented"
    fixture = _fixture(tmp_path)
    sample_id = "q0000029_fixture29"
    metadata_path = (
        fixture["run_dir"]
        / "anygrasp_input_predicted_mask"
        / sample_id
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    metadata["query"] = "wrong query"
    metadata_path.write_text(json.dumps(metadata))

    result = create_verified_subset(
        **fixture,
        report_path=tmp_path / "reports/anygrasp_verified_subset_report.md",
        validation_only=True,
    )

    assert result["status"] == "BLOCKED"
    failed = next(row for row in result["rows"] if row["sample_id"] == sample_id)
    assert "bundle_metadata_query_mismatch" in failed["blockers"]
