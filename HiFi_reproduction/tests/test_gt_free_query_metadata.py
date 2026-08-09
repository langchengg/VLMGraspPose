from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.grasping.grasp_serialization import save_candidate_bundle
from src.grasping.reranking_v1.identity import sha256_file, stable_sample_id
from tools.modular_reranking.extract_candidate_features import (
    main as extract_features_main,
)
from tools.modular_reranking.project_gt_free_query_metadata import (
    FORBIDDEN_QUERY_METADATA_FIELDS,
    QUERY_METADATA_FIELDS,
    load_query_metadata_bundle,
    load_verified_prediction_bundle,
    main as project_query_metadata_main,
    validate_tmp_scope,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    run = tmp_path / "active_run"
    run.mkdir()
    (run / ".RUN_ACTIVE").touch()
    tmp_root = run / "tmp"
    tmp_root.mkdir()
    scene_id = "ARID10/table/top/seq01,result.png"
    question_index = 7
    query = "Grasp the red mug"
    sample_id = stable_sample_id(scene_id, question_index)

    frozen_manifest = tmp_path / "ocidvlg_unique_val.json"
    _write_json(
        frozen_manifest,
        [
            {
                "num": 0,
                "question_index": question_index,
                "scene_id": scene_id,
                "text": query,
                # Deliberately present in the upstream dataset manifest. The
                # projection must never copy it into a query row.
                "mask_path": "/ground/truth/mask.png",
            }
        ],
    )
    probability_path = run / "compact_inputs" / "probability.npz"
    probability_path.parent.mkdir(parents=True)
    np.savez_compressed(
        probability_path, probability=np.full((2, 2), 0.75, np.float32)
    )
    prediction_root = run / "compact_inputs" / "val"
    prediction_row = {
        "schema_version": 1,
        "split": "val",
        "sample_index": 0,
        "sample_id": sample_id,
        "question_index": question_index,
        "scene_id": scene_id,
        "query": query,
        "manifest_path": str(frozen_manifest.resolve()),
        "manifest_sha256": sha256_file(frozen_manifest),
        "probability_path": str(probability_path.resolve()),
        "probability_sha256": sha256_file(probability_path),
        "gt_artifacts_exported": False,
        "ready": True,
    }
    manifest_path = prediction_root / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(prediction_row, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(prediction_root / "rows" / f"{sample_id}.json", prediction_row)
    _write_json(
        prediction_root / "summary.json",
        {
            "status": "COMPLETED",
            "split": "val",
            "samples": 1,
            "output_manifest": str(manifest_path.resolve()),
            "output_manifest_sha256": sha256_file(manifest_path),
            "manifest_sha256": sha256_file(frozen_manifest),
        },
    )

    annotations = tmp_path / "val_expressions.json"
    _write_json(
        annotations,
        {
            "info": {"split": "val", "version": "unique"},
            "data": [
                {
                    "split": "val",
                    "image_filename": scene_id,
                    "question": query,
                    "question_index": question_index,
                    "program": [
                        {"type": "scene", "inputs": []},
                        {
                            "type": "filter_category",
                            "inputs": [0],
                            "side_inputs": ["mug"],
                        },
                        {"type": "unique", "inputs": [1]},
                        {"type": "return", "inputs": [2]},
                    ],
                    "grasps": [[[1, 2], [3, 4], [5, 6], [7, 8]]],
                    "box": [1, 2, 3, 4],
                    "answer": 9,
                    "target": "mug_1",
                    "concept_map": {"<Y>": "mug"},
                }
            ],
        },
    )
    return {
        "run": run,
        "tmp_root": tmp_root,
        "sample_id": sample_id,
        "scene_id": scene_id,
        "query": query,
        "annotations": annotations,
        "prediction_root": prediction_root,
        "probability_path": probability_path,
    }


def _project(
    fixture: dict[str, Path | str],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    output = Path(fixture["run"]) / "manifests" / "query_metadata_val"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project_gt_free_query_metadata.py",
            "--annotations",
            str(fixture["annotations"]),
            "--prediction-root",
            str(fixture["prediction_root"]),
            "--output-root",
            str(output),
            "--tmp-root",
            str(fixture["tmp_root"]),
            "--split",
            "val",
        ],
    )
    assert project_query_metadata_main() == 0
    return output


def test_projection_emits_only_five_gt_free_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = _project(fixture, monkeypatch)
    row = json.loads((output / "query_metadata.jsonl").read_text())
    assert tuple(sorted(row)) == tuple(sorted(QUERY_METADATA_FIELDS))
    assert not (set(row) & set(FORBIDDEN_QUERY_METADATA_FIELDS))
    assert row["query_type"] == "name"
    manifest = json.loads(
        (output / "query_metadata_manifest.json").read_text()
    )
    assert manifest["ground_truth_allowed"] is False
    assert manifest["exact_schema_required"] is True
    assert manifest["forbidden_fields"] == list(FORBIDDEN_QUERY_METADATA_FIELDS)
    assert manifest["source_annotations_may_contain_ground_truth"] is True
    assert not list(Path(fixture["tmp_root"]).rglob("*.tmp"))


@pytest.mark.parametrize("leaked_field", ["grasps", "target", "candidate_labels"])
def test_query_loader_rejects_any_extra_gt_field_even_with_updated_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaked_field: str,
) -> None:
    fixture = _fixture(tmp_path)
    output = _project(fixture, monkeypatch)
    query_path = output / "query_metadata.jsonl"
    row = json.loads(query_path.read_text())
    row[leaked_field] = []
    query_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest_path = output / "query_metadata_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["query_metadata_sha256"] = sha256_file(query_path)
    _write_json(manifest_path, manifest)
    prediction = load_verified_prediction_bundle(
        fixture["prediction_root"], split="val"
    )
    with pytest.raises(ValueError, match="exact GT-free schema"):
        load_query_metadata_bundle(
            query_path, prediction_bundle=prediction, split="val"
        )


def test_query_loader_rejects_gt_fields_embedded_in_projection_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = _project(fixture, monkeypatch)
    manifest_path = output / "query_metadata_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["candidate_labels"] = []
    _write_json(manifest_path, manifest)
    prediction = load_verified_prediction_bundle(
        fixture["prediction_root"], split="val"
    )
    with pytest.raises(ValueError, match="exact safe schema"):
        load_query_metadata_bundle(
            output / "query_metadata.jsonl",
            prediction_bundle=prediction,
            split="val",
        )


def test_feature_extractor_uses_projection_after_expressions_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    query_root = _project(fixture, monkeypatch)
    Path(fixture["annotations"]).unlink()

    sample_id = str(fixture["sample_id"])
    candidate_dir = Path(fixture["run"]) / "candidates" / sample_id
    candidate_dir.mkdir(parents=True)
    save_candidate_bundle(
        [],
        json_path=candidate_dir / "candidates.json",
        npz_path=candidate_dir / "candidates.npz",
        csv_path=candidate_dir / "candidates.csv",
        metadata={
            "sample_id": sample_id,
            "question_index": 7,
            "scene_id": str(fixture["scene_id"]),
            "query": str(fixture["query"]),
        },
    )
    np.save(candidate_dir / "depth_m.npy", np.ones((20, 20), np.float32))
    Image.fromarray(np.full((20, 20), 255, np.uint8)).save(
        candidate_dir / "hifics_mask_processed.png"
    )
    _write_json(
        candidate_dir / "camera.intr",
        {
            "frame": "camera",
            "fx": 100.0,
            "fy": 100.0,
            "cx": 9.5,
            "cy": 9.5,
            "skew": 0.0,
            "height": 20,
            "width": 20,
        },
    )
    scored = Path(fixture["run"]) / "scores" / sample_id
    marker = {
        "sample_id": sample_id,
        "scoring_status": "skipped_valid_empty",
        "source_candidate_count": 0,
        "gqcnn_scored_count": 0,
    }
    _write_json(scored / "_SCORING_COMPLETE.json", marker)
    _write_json(scored / "scoring_metadata.json", marker)

    output = Path(fixture["run"]) / "features" / "val"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_candidate_features.py",
            "--candidate-root",
            str(Path(fixture["run"]) / "candidates"),
            "--scored-root",
            str(Path(fixture["run"]) / "scores"),
            "--prediction-root",
            str(fixture["prediction_root"]),
            "--query-metadata",
            str(query_root / "query_metadata.jsonl"),
            "--output-root",
            str(output),
            "--tmp-root",
            str(fixture["tmp_root"]),
            "--split",
            "val",
        ],
    )
    assert extract_features_main() == 0
    samples = pd.read_parquet(output / "per_sample.parquet")
    assert samples.loc[0, "query_type"] == "name"
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["official_expressions_opened_by_feature_extractor"] is False
    assert Path(manifest["annotations_path"]).name == (
        "query_metadata_manifest.json"
    )
    assert (
        manifest["annotations_sha256"]
        == manifest["query_metadata_manifest_sha256"]
    )
    assert (
        manifest["prediction_manifest_sha256"]
        == load_verified_prediction_bundle(
            fixture["prediction_root"], split="val"
        ).manifest_sha256
    )
    assert not list(Path(fixture["tmp_root"]).rglob("*.tmp"))


def test_prediction_manifest_rejects_embedded_candidate_labels(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prediction_root = Path(fixture["prediction_root"])
    manifest_path = prediction_root / "manifest.jsonl"
    row = json.loads(manifest_path.read_text())
    row["candidate_labels"] = []
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _write_json(prediction_root / "rows" / f"{fixture['sample_id']}.json", row)
    summary = json.loads((prediction_root / "summary.json").read_text())
    summary["output_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(prediction_root / "summary.json", summary)
    with pytest.raises(ValueError, match="forbidden GT fields"):
        load_verified_prediction_bundle(prediction_root, split="val")


def test_atomic_tmp_scope_rejects_a_different_active_run(tmp_path: Path) -> None:
    output_run = tmp_path / "output_run"
    temporary_run = tmp_path / "temporary_run"
    output_run.mkdir()
    temporary_run.mkdir()
    (output_run / ".RUN_ACTIVE").touch()
    (temporary_run / ".RUN_ACTIVE").touch()
    with pytest.raises(ValueError, match="different active runs"):
        validate_tmp_scope(
            output_run / "features",
            temporary_run / "tmp",
        )
