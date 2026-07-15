import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

try:
    from tools.export_anygrasp_inputs import export_run, stable_sample_id
except (ImportError, ModuleNotFoundError):
    export_run = stable_sample_id = None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_binary_pcd(path, depth_mm, K, *, invalidate_target=False):
    height, width = depth_mm.shape
    fx, fy, cx, cy = K
    yy, xx = np.indices((height, width), dtype=np.float32)
    z = depth_mm.astype(np.float32) / 1000.0
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    if invalidate_target:
        x[2:8, 4:12] = np.nan
        y[2:8, 4:12] = np.nan
        z[2:8, 4:12] = np.nan
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
    records["x"] = x.ravel()
    records["y"] = y.ravel()
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


def _prediction_row(sample_index, sample_id, question_index, scene_id, query):
    return {
        "sample_index": sample_index,
        "evaluation_sample_id": str(sample_index + 3),
        "stable_sample_id": sample_id,
        "question_index": question_index,
        "scene_id": scene_id,
        "query": query,
        "iou": 0.75,
        "directory": f"predictions/{sample_id}",
    }


def _fixture(
    tmp_path,
    *,
    probability_dtype=np.float32,
    probability_value=0.75,
    mask_value=255,
    mask_nonempty=True,
    invalidate_target=False,
):
    source_root = tmp_path / "OCID-VLG"
    run_dir = tmp_path / "run"
    sequence = "ARID10/floor/top/non-fruits/seq09"
    image_name = "frame.png"
    scene_id = f"{sequence},{image_name}"
    query = "Pick the green object"
    question_index = 42
    sample_id = stable_sample_id(scene_id, question_index) if stable_sample_id else "missing"
    height, width = 12, 16
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = 40
    rgb[..., 1] = 120
    rgb[..., 2] = 80
    depth = np.full((height, width), 1000, dtype=np.uint16)
    rgb_path = source_root / sequence / "rgb" / image_name
    depth_path = source_root / sequence / "depth" / image_name
    pcd_path = source_root / sequence / "pcd" / "frame.pcd"
    rgb_path.parent.mkdir(parents=True)
    depth_path.parent.mkdir(parents=True)
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(depth).save(depth_path)
    expected_K = (100.0, 110.0, 7.5, 5.5)
    _write_binary_pcd(
        pcd_path, depth, expected_K, invalidate_target=invalidate_target
    )

    original_manifest = source_root / "refer" / "unique" / "test_expressions.json"
    original_manifest.parent.mkdir(parents=True)
    original_manifest.write_text(
        json.dumps(
            {
                "info": {"split": "test", "version": "unique"},
                "data": [
                    {
                        "question_index": question_index,
                        "image_filename": scene_id,
                        "question": query,
                        "answer": 7,
                        "target": "fixture_1",
                    }
                ],
            }
        )
    )
    run_dir.mkdir()
    run_manifest = run_dir / "ocid_vlg_test.json"
    run_manifest.write_text(
        json.dumps(
            [
                {
                    "num": 3,
                    "question_index": question_index,
                    "scene_id": scene_id,
                    "text": query,
                }
            ]
        )
    )
    metrics = run_dir / "evaluation" / "per_sample_metrics.csv"
    metrics.parent.mkdir()
    metrics.write_text(
        "sample_index,sample_id,text,iou,inference_seconds\n"
        f"0,3,{query},0.75,0.01\n"
    )
    checkpoint = run_dir / "checkpoints" / "final.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic checkpoint")

    prediction_dir = run_dir / "predictions" / sample_id
    prediction_dir.mkdir(parents=True)
    probability = np.full(
        (height, width), probability_value, dtype=probability_dtype
    )
    np.save(prediction_dir / "predicted_probability_original_resolution.npy", probability)
    mask = np.zeros((height, width), dtype=np.uint8)
    if mask_nonempty:
        mask[2:8, 4:12] = mask_value
    Image.fromarray(mask).save(
        prediction_dir / "predicted_mask_original_resolution.png"
    )
    prediction_metadata = {
        "sample_index": 0,
        "evaluation_sample_id": "3",
        "stable_sample_id": sample_id,
        "question_index": question_index,
        "scene_id": scene_id,
        "query": query,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "frozen_test_manifest": str(run_manifest.resolve()),
        "frozen_test_manifest_sha256": _sha256(run_manifest),
        "source_expression_file": str(original_manifest.resolve()),
        "source_expression_sha256": _sha256(original_manifest),
    }
    (prediction_dir / "sample_metadata.json").write_text(
        json.dumps(prediction_metadata)
    )
    prediction_manifest = run_dir / "predictions" / "export_manifest.json"
    manifest_payload = {
        "number_of_samples": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "repo_commit": "deadbeef",
        "checkpoint_repo_commit": "cafebabe",
        "frozen_test_manifest": str(run_manifest.resolve()),
        "frozen_test_manifest_sha256": _sha256(run_manifest),
        "metrics_csv": str(metrics.resolve()),
        "metrics_csv_sha256": _sha256(metrics),
        "source_root": str(source_root.resolve()),
        "source_expression_file": str(original_manifest.resolve()),
        "source_expression_sha256": _sha256(original_manifest),
        "samples": [
            _prediction_row(0, sample_id, question_index, scene_id, query)
        ],
    }
    prediction_manifest.write_text(json.dumps(manifest_payload))
    return {
        "source_root": source_root,
        "run_dir": run_dir,
        "original_manifest": original_manifest,
        "run_manifest": run_manifest,
        "prediction_manifest": prediction_manifest,
        "prediction_dir": prediction_dir,
        "sample_id": sample_id,
        "scene_id": scene_id,
        "query": query,
        "question_index": question_index,
        "expected_K": expected_K,
        "checkpoint": checkpoint,
        "metrics": metrics,
        "rgb": rgb_path,
        "depth": depth_path,
        "pcd": pcd_path,
    }


def _export(fixture, tmp_path, **overrides):
    arguments = {
        "run_dir": fixture["run_dir"],
        "source_root": fixture["source_root"],
        "original_manifest_path": fixture["original_manifest"],
        "sample_ids": [fixture["sample_id"]],
        "expected_count": 1,
        "expected_size": (16, 12),
        "report_path": tmp_path / "reports/anygrasp_export_summary.md",
    }
    arguments.update(overrides)
    return export_run(**arguments)


def test_stable_sample_id_matches_production_digest_contract():
    assert stable_sample_id is not None
    scene_id = "ARID20/table/top/seq01,frame.png"
    question_index = 42
    digest = hashlib.sha256(f"{scene_id}\t{question_index}".encode()).hexdigest()[:16]

    assert stable_sample_id(scene_id, question_index) == f"q0000042_{digest}"


def test_exports_isolated_copies_and_complete_provenance(tmp_path):
    assert export_run is not None
    fixture = _fixture(tmp_path)
    source_depth_before = fixture["depth"].read_bytes()

    result = _export(fixture, tmp_path)

    assert result["status"] == "DONE"
    sample_dir = (
        fixture["run_dir"]
        / "anygrasp_input_predicted_mask"
        / fixture["sample_id"]
    )
    assert {path.name for path in sample_dir.iterdir()} == {
        "color.png",
        "depth.png",
        "target_mask.png",
        "target_probability.npy",
        "language.txt",
        "intrinsics.json",
        "metadata.json",
        "checksums.sha256",
    }
    assert np.asarray(Image.open(sample_dir / "depth.png")).dtype == np.uint16
    assert fixture["depth"].read_bytes() == source_depth_before
    assert os.stat(fixture["depth"]).st_ino != os.stat(sample_dir / "depth.png").st_ino
    assert os.stat(fixture["prediction_dir"] / "predicted_probability_original_resolution.npy").st_ino != os.stat(sample_dir / "target_probability.npy").st_ino
    metadata = json.loads((sample_dir / "metadata.json").read_text())
    assert metadata["ready_for_anygrasp"] is True
    assert set(metadata["materialization"].values()) == {"copy"}
    assert metadata["checkpoint_sha256"] == _sha256(fixture["checkpoint"])
    assert metadata["frozen_test_manifest_sha256"] == _sha256(fixture["run_manifest"])
    assert metadata["evaluation_metrics_csv_sha256"] == _sha256(fixture["metrics"])
    assert metadata["prediction_export_manifest_sha256"] == _sha256(fixture["prediction_manifest"])
    assert metadata["anygrasp_exporter_source_sha256"] == _sha256(
        Path(__file__).parents[1] / "tools/export_anygrasp_inputs.py"
    )
    assert metadata["source_rgb_sha256"] == _sha256(fixture["rgb"])
    assert metadata["source_depth_sha256"] == _sha256(fixture["depth"])
    assert metadata["source_pcd_sha256"] == _sha256(fixture["pcd"])
    assert not any("ground_truth" in path.name or "oracle" in path.name for path in sample_dir.iterdir())


def test_missing_or_nonbijective_prediction_manifest_fails_closed(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["prediction_manifest"].unlink()
    with pytest.raises(FileNotFoundError, match="Prediction export manifest"):
        _export(fixture, tmp_path, validation_only=True)

    fixture = _fixture(tmp_path / "duplicate")
    payload = json.loads(fixture["prediction_manifest"].read_text())
    payload["number_of_samples"] = 2
    payload["samples"].append(dict(payload["samples"][0]))
    fixture["prediction_manifest"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate stable sample IDs"):
        _export(fixture, tmp_path, expected_count=2, validation_only=True)


def test_stable_id_and_prediction_metadata_mismatches_are_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["prediction_manifest"].read_text())
    payload["samples"][0]["stable_sample_id"] = "q0000042_wrongdigest"
    fixture["prediction_manifest"].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stable_sample_id mismatch"):
        _export(fixture, tmp_path, validation_only=True)

    fixture = _fixture(tmp_path / "metadata")
    metadata_path = fixture["prediction_dir"] / "sample_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        _export(fixture, tmp_path, validation_only=True)


@pytest.mark.parametrize(
    ("fixture_kwargs", "blocker"),
    [
        ({"probability_dtype": np.float64}, "probability_dtype_not_float32:float64"),
        ({"probability_value": 1.1}, "probability_out_of_range"),
        ({"mask_value": 1}, "predicted_mask_values_invalid"),
        ({"mask_nonempty": False}, "predicted_mask_empty"),
        ({"invalidate_target": True}, "target_has_no_valid_depth_pcd_point"),
    ],
)
def test_exact_readiness_gates_block_invalid_inputs(tmp_path, fixture_kwargs, blocker):
    fixture = _fixture(tmp_path, **fixture_kwargs)

    result = _export(fixture, tmp_path, validation_only=True)

    assert result["status"] == "BLOCKED"
    assert blocker in result["rows"][0]["blockers"]
    assert not (fixture["run_dir"] / "anygrasp_input_predicted_mask").exists()


def test_production_size_gate_is_configurable_for_synthetic_tests(tmp_path):
    fixture = _fixture(tmp_path)

    result = _export(
        fixture, tmp_path, expected_size=(640, 480), validation_only=True
    )

    assert result["status"] == "BLOCKED"
    assert "original_size_mismatch:16x12:expected:640x480" in result["rows"][0]["blockers"]


def test_geometry_fit_is_cached_for_two_expressions_from_same_scene(tmp_path):
    fixture = _fixture(tmp_path)
    original_payload = json.loads(fixture["original_manifest"].read_text())
    original_payload["data"].append(
        {
            **original_payload["data"][0],
            "question_index": 43,
            "question": "Pick the object on the left",
        }
    )
    fixture["original_manifest"].write_text(json.dumps(original_payload))
    run_records = json.loads(fixture["run_manifest"].read_text())
    run_records.append(
        {
            **run_records[0],
            "num": 4,
            "question_index": 43,
            "text": "Pick the object on the left",
        }
    )
    fixture["run_manifest"].write_text(json.dumps(run_records))
    second_id = stable_sample_id(fixture["scene_id"], 43)
    second_dir = fixture["run_dir"] / "predictions" / second_id
    second_dir.mkdir()
    np.save(
        second_dir / "predicted_probability_original_resolution.npy",
        np.full((12, 16), 0.8, dtype=np.float32),
    )
    second_mask = np.zeros((12, 16), dtype=np.uint8)
    second_mask[1:5, 2:7] = 255
    Image.fromarray(second_mask).save(
        second_dir / "predicted_mask_original_resolution.png"
    )
    prediction_payload = json.loads(fixture["prediction_manifest"].read_text())
    prediction_payload["number_of_samples"] = 2
    prediction_payload["frozen_test_manifest_sha256"] = _sha256(fixture["run_manifest"])
    prediction_payload["source_expression_sha256"] = _sha256(fixture["original_manifest"])
    prediction_payload["samples"].append(
        _prediction_row(
            1,
            second_id,
            43,
            fixture["scene_id"],
            "Pick the object on the left",
        )
    )
    fixture["prediction_manifest"].write_text(json.dumps(prediction_payload))
    for sample_dir, index, sid, qid, query in (
        (fixture["prediction_dir"], 0, fixture["sample_id"], 42, fixture["query"]),
        (second_dir, 1, second_id, 43, "Pick the object on the left"),
    ):
        metadata = json.loads((fixture["prediction_dir"] / "sample_metadata.json").read_text())
        metadata.update(
            {
                "sample_index": index,
                "evaluation_sample_id": str(index + 3),
                "stable_sample_id": sid,
                "question_index": qid,
                "query": query,
                "frozen_test_manifest_sha256": _sha256(fixture["run_manifest"]),
                "source_expression_sha256": _sha256(fixture["original_manifest"]),
            }
        )
        (sample_dir / "sample_metadata.json").write_text(json.dumps(metadata))

    result = _export(
        fixture,
        tmp_path,
        sample_ids=[fixture["sample_id"], second_id],
        expected_count=2,
        validation_only=True,
    )

    assert result["status"] == "DONE"
    assert result["geometry_scenes_fitted"] == 1
    assert result["geometry_cache_hits"] == 1


def test_atomic_output_refuses_stale_directory_and_force_replaces_it(tmp_path):
    fixture = _fixture(tmp_path)
    _export(fixture, tmp_path)
    output = fixture["run_dir"] / "anygrasp_input_predicted_mask"
    stale = output / "stale.txt"
    stale.write_text("old")

    with pytest.raises(FileExistsError, match="already exists"):
        _export(fixture, tmp_path)

    result = _export(fixture, tmp_path, force=True)

    assert result["status"] == "DONE"
    assert not stale.exists()
    assert (output / "manifest.jsonl").is_file()
    assert not list(output.parent.glob(".anygrasp_input_predicted_mask.*"))


def test_validation_only_is_write_free(tmp_path):
    fixture = _fixture(tmp_path)
    result = _export(fixture, tmp_path, validation_only=True)

    assert result["status"] == "DONE"
    assert not (fixture["run_dir"] / "anygrasp_input_predicted_mask").exists()
    assert not (tmp_path / "reports/anygrasp_export_summary.md").exists()
