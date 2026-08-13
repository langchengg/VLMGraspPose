from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from d1_reranking.execution import artifact_record
from d1_reranking.plan import primary_plan
from d1_reranking.prelock import _replay_primary_selection
from tools.d1_reranking.select_primary_ranker import run as select_primary
from unified_reranking.hashing import atomic_json, canonical_sha256


def _content(path: Path, value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)


def _development_labels(root: Path, split: str, sample_ids: list[str]) -> None:
    label_dir = root / "03_features" / split / "top5" / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    labels = label_dir / "candidate_labels.parquet"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "candidate_id": ["candidate"] * len(sample_ids),
            "candidate_success": [1] * len(sample_ids),
        }
    ).to_parquet(labels, index=False)
    _content(
        label_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifact": artifact_record(labels),
        },
    )
    denominator = root / "01_manifests" / f"d1_paired_{split}.parquet"
    denominator.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sample_id": sample_ids, "scene_id": [f"scene-{item}" for item in sample_ids]}
    ).to_parquet(denominator, index=False)


def _build_primary_selection(
    root: Path,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, Path],
    dict[str, pd.DataFrame],
    dict[str, object],
]:
    _development_labels(root, "train", [f"train-{index}" for index in range(5)])
    _development_labels(root, "validation", ["validation-0"])
    plan_path = root / "configs" / "d1_primary_matrix_plan.json"
    plan = primary_plan(tool_paths=(Path(__file__),))
    atomic_json(plan_path, plan)
    outputs: dict[str, dict[str, str]] = {}
    cell_paths: dict[str, Path] = {}
    cell_predictions: dict[str, pd.DataFrame] = {}
    for job in plan["jobs"]:
        job_id = str(job["job_id"])
        configuration = dict(job["configuration"])
        sample_id = (
            "validation-0"
            if configuration["mode"] == "validation"
            else f"train-{configuration['held_fold']}"
        )
        predictions = pd.DataFrame(
            {
                "sample_id": [sample_id],
                "candidate_id": ["candidate"],
                "native_rank": [1],
                "candidate_identity_sha256": [f"identity-{sample_id}"],
                "candidate_geometry_sha256": [f"geometry-{sample_id}"],
                "score": [float(configuration["seed"]) / 10_000.0],
            }
        )
        cell_dir = root / "synthetic_cells" / job_id
        prediction_path = cell_dir / "predictions.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(prediction_path, index=False)
        cell_manifest_path = cell_dir / "manifest.json"
        cell_configuration = {
            **configuration,
            "planned_job_id": job_id,
            "planned_configuration": configuration,
        }
        _content(
            cell_manifest_path,
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "configuration": cell_configuration,
                "sources": {"test_contract": artifact_record(Path(__file__))},
                "artifacts": {"predictions": artifact_record(prediction_path)},
                "candidate_test_labels_read": False,
            },
        )
        outputs[job_id] = artifact_record(cell_manifest_path)
        cell_paths[job_id] = cell_manifest_path.resolve()
        cell_predictions[job_id] = predictions
    execution_path = root / "07_validation" / "primary_matrix_execution.json"
    _content(
        execution_path,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "plan": artifact_record(plan_path),
            "expected_jobs": 360,
            "completed_jobs": 360,
            "outputs": outputs,
        },
    )
    selection = select_primary(root, resume=False)
    planned = {str(job["job_id"]): dict(job["configuration"]) for job in plan["jobs"]}
    return planned, cell_paths, cell_predictions, selection


def test_prelock_replays_all_primary_cells_and_rejects_reauthored_winner(
    tmp_path: Path,
) -> None:
    planned, cell_paths, predictions, selection = _build_primary_selection(tmp_path)
    plan_path = tmp_path / "configs" / "d1_primary_matrix_plan.json"
    execution_path = tmp_path / "07_validation" / "primary_matrix_execution.json"
    _replay_primary_selection(
        tmp_path,
        planned=planned,
        cell_paths=cell_paths,
        cell_predictions=predictions,
        selection=selection,
        plan_path=plan_path,
        execution_path=execution_path,
    )
    tampered = json.loads(json.dumps(selection))
    tampered["selected_method"] = "R6"
    with pytest.raises(RuntimeError, match="winner differs from semantic replay"):
        _replay_primary_selection(
            tmp_path,
            planned=planned,
            cell_paths=cell_paths,
            cell_predictions=predictions,
            selection=tampered,
            plan_path=plan_path,
            execution_path=execution_path,
        )
