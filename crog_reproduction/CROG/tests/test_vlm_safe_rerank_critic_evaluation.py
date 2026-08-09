from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from failure_analysis.vlm_safe_rerank.critic_evaluation import evaluate_diagnostic_phase
from failure_analysis.vlm_safe_rerank.renderer import PerturbationVariant
from failure_analysis.vlm_safe_rerank.runner import prepare_perturbation_phase


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_prepare_perturbation_phase_copies_only_frozen_inference(tmp_path: Path) -> None:
    source = tmp_path / "diagnostic_expanded"
    manifest = source / "inference_manifest.json"
    _json(manifest, {"feature_file": "/frozen/features.jsonl", "rows": [{"sample_id": "s"}]})
    _json(
        source / "INFERENCE_MANIFEST_IDENTITY.json",
        {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
    )
    summary = prepare_perturbation_phase(
        run_dir=tmp_path,
        source_phase="diagnostic_expanded",
        destination_phase="diagnostic_perturbations",
    )
    destination = tmp_path / "diagnostic_perturbations"
    assert summary["pairs"] == 1
    assert set(summary["planned_variants"]) == {
        variant.value for variant in PerturbationVariant
        if variant is not PerturbationVariant.ORIGINAL
    }
    assert not (destination / "evaluation_manifest.parquet").exists()


def _diagnostic_fixture(tmp_path: Path, *, omit_last_variant: bool = False) -> None:
    original = tmp_path / "diagnostic_expanded"
    perturbation = tmp_path / "diagnostic_perturbations"
    features = tmp_path / "features.jsonl"
    features.write_text("{}\n", encoding="utf-8")
    source_manifest = original / "inference_manifest.json"
    _json(
        source_manifest,
        {
            "feature_file": str(features.resolve()),
            "rows": [{"sample_id": "s", "challenger_candidate_id": "candidate_1"}],
        },
    )
    _json(
        original / "INFERENCE_MANIFEST_IDENTITY.json",
        {"sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest()},
    )
    _json(original / "sample_manifest.json", {"samples": [{"sample_id": "s"}]})
    perturbation_manifest = perturbation / "inference_manifest.json"
    _json(
        perturbation_manifest,
        {
            "feature_file": str(features.resolve()),
            "rows": [{"sample_id": "s", "challenger_candidate_id": "candidate_1"}],
        },
    )
    _json(
        perturbation / "INFERENCE_MANIFEST_IDENTITY.json",
        {"sha256": hashlib.sha256(perturbation_manifest.read_bytes()).hexdigest()},
    )
    base = {
        "sample_id": "s",
        "challenger_candidate_id": "candidate_1",
        "status": "SUCCEEDED",
        "parsed": None,
    }
    models = ["gemini-robotics-er-2-preview", "gemini-3.6-flash"]
    _parquet(
        original / "pairwise_responses.parquet",
        [{**base, "model_id": model, "variant": "original"} for model in models],
    )
    _parquet(
        original / "evaluation_manifest.parquet",
        [{
            "sample_id": "s",
            "challenger_candidate_id": "candidate_1",
            "corrected_baseline_correct": True,
            "corrected_challenger_correct": False,
            "cohort": "protected_correct",
            "query_type": "name",
        }],
    )
    variants = [
        variant.value for variant in PerturbationVariant
        if variant is not PerturbationVariant.ORIGINAL
    ]
    if omit_last_variant:
        variants.pop()
    _parquet(
        perturbation / "pairwise_responses.parquet",
        [
            {**base, "model_id": model, "variant": variant}
            for model in models
            for variant in variants
        ],
    )


def test_diagnostic_evaluator_requires_complete_perturbation_matrix(tmp_path: Path) -> None:
    _diagnostic_fixture(tmp_path)
    result = evaluate_diagnostic_phase(
        tmp_path,
        "diagnostic_expanded",
        perturbation_phase="diagnostic_perturbations",
    )
    stability = result["models"]["gemini-robotics-er-2-preview"]["stability"]
    assert len(stability) == 5
    assert all(item["pairs"] == 1 for item in stability.values())


def test_diagnostic_evaluator_rejects_missing_perturbation(tmp_path: Path) -> None:
    _diagnostic_fixture(tmp_path, omit_last_variant=True)
    with pytest.raises(ValueError, match="coverage"):
        evaluate_diagnostic_phase(
            tmp_path,
            "diagnostic_expanded",
            perturbation_phase="diagnostic_perturbations",
        )
