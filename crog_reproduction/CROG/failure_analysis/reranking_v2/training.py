from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .datasets import load_joined
from .labels import validate_inference_label_join
from .metrics import evaluate_rankings, risk_coverage_curve
from .models.pairwise_gate import (
    predict_gate,
    select_with_gate,
    train_gate,
    tune_gate_policy,
)
from .protocol import split_ids
from .schema import atomic_write_json, atomic_write_jsonl, read_jsonl


def _atomic_torch_save(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def rankings_from_gate(
    samples,
    probabilities,
    policy,
):
    rankings, selections, confidence = {}, {}, {}
    for sample in samples:
        selection = select_with_gate(
            sample,
            probabilities[sample.sample_id],
            harm_cost=policy["harm_cost"],
            threshold=policy["threshold"],
        )
        original = [
            candidate["candidate_id"] for candidate in sample.feature["candidates"]
        ]
        selected = selection["selected_candidate_id"]
        rankings[sample.sample_id] = [selected] + [
            value for value in original if value != selected
        ]
        selections[sample.sample_id] = selection
        confidence[sample.sample_id] = max(0.0, float(selection["best_gain"]))
    return rankings, selections, confidence


def q_candidate_probabilities(samples) -> dict[str, np.ndarray]:
    return {
        sample.sample_id: np.asarray(
            [candidate["q_raw"] for candidate in sample.feature["candidates"]],
            dtype=np.float64,
        )
        for sample in samples
    }


def _replace_labels(samples, corrected_labels_path: str | Path):
    labels = {
        (str(record["split"]), int(record["source_sample_id"])): record
        for record in read_jsonl(corrected_labels_path)
    }
    replaced = []
    for sample in samples:
        key = (
            str(sample.feature["split"]),
            int(sample.feature["sample_id"]),
        )
        corrected = labels[key]
        validate_inference_label_join(sample.feature, corrected)
        replaced.append(type(sample)(feature=sample.feature, label=corrected))
    return replaced


def train_scalar_gate_experiment(
    *,
    train_features_path: str | Path,
    train_legacy_labels_path: str | Path,
    validation_features_path: str | Path,
    validation_legacy_labels_path: str | Path,
    validation_corrected_labels_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 23,
    device: str = "auto",
    epochs: int = 40,
    resume: bool = False,
    bootstrap_iterations_selection: int = 2000,
    bootstrap_iterations_report: int = 10_000,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=resume)
    train = load_joined(
        train_features_path,
        train_legacy_labels_path,
        allowed_sample_ids=split_ids(split_manifest_path, "train"),
    )
    validation = load_joined(
        validation_features_path, validation_legacy_labels_path
    )
    ensemble_seeds = (int(seed), int(seed) + 6, int(seed) + 12)
    model_paths = []
    seed_probabilities = []
    for ensemble_seed in ensemble_seeds:
        model_path = output_dir / f"scalar_gate_seed{ensemble_seed}.pt"
        artifact = (
            torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )
            if resume and model_path.exists()
            else train_gate(
                train,
                validation,
                seed=ensemble_seed,
                device=device,
                epochs=epochs,
            )
        )
        if not model_path.exists():
            _atomic_torch_save(artifact, model_path)
        model_paths.append(model_path)
        seed_probabilities.append(
            predict_gate(artifact, validation, device=device)
        )
    probabilities = {
        sample.sample_id: np.mean(
            [
                values[sample.sample_id]
                for values in seed_probabilities
            ],
            axis=0,
        )
        for sample in validation
    }
    tuning = tune_gate_policy(
        validation,
        probabilities,
        bootstrap_iterations=bootstrap_iterations_selection,
        seed=seed + 100,
    )
    policy = tuning["selected"]
    rankings, selections, confidence = rankings_from_gate(
        validation, probabilities, policy
    )
    legacy_summary, legacy_rows = evaluate_rankings(
        validation,
        rankings,
        candidate_probabilities=q_candidate_probabilities(validation),
        bootstrap_iterations=bootstrap_iterations_report,
        bootstrap_seed=seed + 200,
    )
    corrected_validation = _replace_labels(
        validation, validation_corrected_labels_path
    )
    corrected_summary, _ = evaluate_rankings(
        corrected_validation,
        rankings,
        candidate_probabilities=q_candidate_probabilities(corrected_validation),
        bootstrap_iterations=bootstrap_iterations_report,
        bootstrap_seed=seed + 200,
    )
    prediction_path = output_dir / "validation_predictions.jsonl"
    atomic_write_jsonl(
        prediction_path,
        (
            {
                "sample_id": sample.sample_id,
                "candidate_order": rankings[sample.sample_id],
                "selection": selections[sample.sample_id],
                "pair_probabilities_r_h_n": probabilities[
                    sample.sample_id
                ].tolist(),
                "candidate_correctness_probabilities": [
                    float(candidate["q_raw"])
                    for candidate in sample.feature["candidates"]
                ],
            }
            for sample in validation
        ),
    )
    result = {
        "method": "scalar_only_pairwise_expected_gain_gate",
        "seed": int(seed),
        "ensemble_seeds": list(ensemble_seeds),
        "policy": policy,
        "tuning": tuning,
        "legacy_official": legacy_summary,
        "corrected": corrected_summary,
        "risk_coverage": risk_coverage_curve(legacy_rows, confidence),
        "model_paths": [str(path.resolve()) for path in model_paths],
        "prediction_path": str(prediction_path.resolve()),
        "train_samples": len(train),
        "validation_samples": len(validation),
    }
    atomic_write_json(output_dir / "validation_summary.json", result)
    return result
