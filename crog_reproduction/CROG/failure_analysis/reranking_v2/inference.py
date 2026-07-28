from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import apply_temperature
from .artifacts import atomic_savez_compressed
from .datasets import FrozenInferenceSample
from .models.latent_residual import predict_latent_residual_arrays
from .models.pairwise_gate import predict_gate, select_with_gate
from .models.rgbd_critic import predict_critic_arrays
from .models.setrank import predict_setrank_arrays
from .oof import _full_tokens, _sigmoid
from .schema import (
    atomic_write_json,
    atomic_write_jsonl,
    sha256_file,
)


def load_local_model_artifact(path: str | Path) -> dict[str, Any]:
    """Load an experiment-produced local artifact, never an external file."""
    return torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )


def _stable_score_order(
    samples: list[FrozenInferenceSample], scores: np.ndarray
) -> dict[str, list[str]]:
    result = {}
    for index, sample in enumerate(samples):
        order = sorted(
            range(5),
            key=lambda candidate: (
                -float(scores[index, candidate]),
                candidate,
                str(sample.feature["candidates"][candidate]["candidate_id"]),
            ),
        )
        result[sample.sample_id] = [
            str(sample.feature["candidates"][candidate]["candidate_id"])
            for candidate in order
        ]
    return result


def write_rankings(
    path: str | Path,
    samples: list[FrozenInferenceSample],
    rankings: dict[str, list[str]],
    *,
    probabilities: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    records = []
    for index, sample in enumerate(samples):
        record = {
            "sample_id": sample.sample_id,
            "candidate_order": rankings[sample.sample_id],
        }
        if probabilities is not None:
            record["candidate_correctness_probabilities"] = (
                np.asarray(probabilities[index], dtype=float).tolist()
            )
        if metadata:
            record["inference_metadata"] = metadata
        records.append(record)
    atomic_write_jsonl(path, records)
    return path


def export_npz_rankings(
    *,
    samples: list[FrozenInferenceSample],
    npz_path: str | Path,
    output_path: str | Path,
    method: str,
    score_key: str = "scores",
    probability_key: str | None = None,
) -> dict[str, Any]:
    """Convert a label-free validation score artifact into ranking JSONL."""
    with np.load(npz_path) as payload:
        if score_key not in payload:
            raise KeyError(f"missing score array {score_key!r} in {npz_path}")
        scores = np.asarray(payload[score_key], dtype=np.float64)
        if scores.shape == (len(samples) * 5,):
            scores = scores.reshape(len(samples), 5)
        if scores.shape != (len(samples), 5):
            raise ValueError(
                f"score array must be [samples,5], observed {scores.shape}"
            )
        if "sample_ids" in payload:
            observed_ids = list(map(str, payload["sample_ids"]))
            expected_ids = [sample.sample_id for sample in samples]
            if observed_ids != expected_ids:
                raise ValueError("score artifact/sample order mismatch")
        probabilities = (
            _sigmoid(scores)
            if probability_key is None
            else np.asarray(payload[probability_key], dtype=np.float64)
        )
        if probabilities.shape == (len(samples) * 5,):
            probabilities = probabilities.reshape(len(samples), 5)
        if probabilities.shape != scores.shape:
            raise ValueError(
                "probability array must have the same [samples,5] shape"
            )
    path = write_rankings(
        output_path,
        samples,
        _stable_score_order(samples, scores),
        probabilities=probabilities,
        metadata={
            "method": str(method),
            "source_npz": str(Path(npz_path).resolve()),
            "score_key": str(score_key),
            "probability_key": probability_key,
        },
    )
    result = {
        "method": str(method),
        "sample_count": len(samples),
        "prediction_path": str(path.resolve()),
        "prediction_sha256": sha256_file(path),
    }
    atomic_write_json(path.with_suffix(".summary.json"), result)
    return result


def predict_base_ensemble(
    *,
    arrays: dict[str, np.ndarray],
    critic_models: list[str | Path],
    latent_models: list[str | Path],
    device: str = "auto",
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    if len(critic_models) < 3 or len(latent_models) < 3:
        raise ValueError("primary ensemble requires at least three declared seeds")
    sample_count = len(arrays["sample_ids"])
    flat_crops = arrays["crops"].reshape(
        sample_count * 5, *arrays["crops"].shape[2:]
    )
    critic_scores, critic_embeddings = [], []
    for path in critic_models:
        artifact = load_local_model_artifact(path)
        scores, embeddings = predict_critic_arrays(
            artifact, flat_crops, device=device
        )
        critic_scores.append(scores.reshape(sample_count, 5))
        critic_embeddings.append(
            # OOF embeddings are intentionally cached as float16. Quantize at
            # the same boundary during inference so SetRank receives the exact
            # representation used for training and validation selection.
            embeddings.reshape(sample_count, 5, -1).astype(np.float16)
        )
    latent_scores, latent_residuals = [], []
    for path in latent_models:
        artifact = load_local_model_artifact(path)
        scores, residuals = predict_latent_residual_arrays(
            artifact,
            arrays["latent_post"],
            arrays["q"],
            alpha=alpha,
            device=device,
        )
        latent_scores.append(scores)
        latent_residuals.append(residuals)
    return {
        "critic_scores": np.stack(critic_scores),
        "critic_embeddings": np.stack(critic_embeddings),
        "latent_scores": np.stack(latent_scores),
        "latent_residuals": np.stack(latent_residuals),
    }


def predict_setrank_ensemble(
    *,
    arrays: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    setrank_models: list[str | Path],
    device: str = "auto",
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    if len(setrank_models) < 3:
        raise ValueError("primary SetRank ensemble requires at least three seeds")
    tokens = _full_tokens(
        arrays,
        critic_scores=base["critic_scores"].mean(axis=0),
        critic_embeddings=base["critic_embeddings"].mean(axis=0),
        latent_scores=base["latent_scores"].mean(axis=0),
        latent_residuals=base["latent_residuals"].mean(axis=0),
    )
    scores, probabilities, residuals = [], [], []
    for path in setrank_models:
        artifact = load_local_model_artifact(path)
        seed_scores, seed_probabilities, seed_residuals = (
            predict_setrank_arrays(
                artifact,
                tokens,
                arrays["q"],
                alpha=alpha,
                device=device,
            )
        )
        scores.append(seed_scores)
        probabilities.append(seed_probabilities)
        residuals.append(seed_residuals)
    return {
        "tokens": tokens,
        "scores": np.stack(scores),
        "probabilities": np.stack(probabilities),
        "residuals": np.stack(residuals),
    }


def _gate_extra(
    base: dict[str, np.ndarray], setrank: dict[str, np.ndarray]
) -> np.ndarray:
    return np.stack(
        (
            base["critic_scores"].mean(axis=0),
            base["critic_scores"].std(axis=0),
            _sigmoid(base["critic_scores"]).mean(axis=0),
            base["latent_scores"].mean(axis=0),
            base["latent_scores"].std(axis=0),
            base["latent_residuals"].mean(axis=0),
            setrank["scores"].mean(axis=0),
            setrank["scores"].std(axis=0),
            setrank["probabilities"].mean(axis=0),
        ),
        axis=-1,
    ).astype(np.float32)


def predict_primary_ensemble(
    *,
    samples: list[FrozenInferenceSample],
    arrays: dict[str, np.ndarray],
    critic_models: list[str | Path],
    latent_models: list[str | Path],
    setrank_models: list[str | Path],
    gate_models: list[str | Path],
    policy: dict[str, Any],
    output_dir: str | Path,
    device: str = "auto",
    alpha: float = 1.0,
    uncertainty_kappa: float = 1.0,
    required_consensus: int = 2,
    candidate_probability_temperature: float = 1.0,
    stability_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run locked inference; only original candidate IDs can be emitted."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if len(samples) != len(arrays["sample_ids"]):
        raise ValueError("sample/enhanced feature count mismatch")
    if list(map(str, arrays["sample_ids"])) != [
        sample.sample_id for sample in samples
    ]:
        raise ValueError("sample/enhanced feature order mismatch")
    base = predict_base_ensemble(
        arrays=arrays,
        critic_models=critic_models,
        latent_models=latent_models,
        device=device,
        alpha=alpha,
    )
    setrank = predict_setrank_ensemble(
        arrays=arrays,
        base=base,
        setrank_models=setrank_models,
        device=device,
        alpha=alpha,
    )
    component_paths = export_component_rankings(
        samples=samples,
        base=base,
        setrank=setrank,
        output_dir=output_dir / "components",
    )
    extras = _gate_extra(base, setrank)
    extra_lookup = {
        sample.sample_id: extras[index]
        for index, sample in enumerate(samples)
    }
    gate_predictions = []
    for path in gate_models:
        artifact = load_local_model_artifact(path)
        gate_predictions.append(
            predict_gate(
                artifact,
                samples,
                extra_candidate_features=extra_lookup,
                device=device,
            )
        )
    if len(gate_predictions) < 3:
        raise ValueError("primary gate ensemble requires at least three seeds")
    stability = None
    if stability_path is not None:
        with np.load(stability_path) as payload:
            stability_ids = list(map(str, payload["sample_ids"]))
            if stability_ids != [sample.sample_id for sample in samples]:
                raise ValueError("stability/sample cohort mismatch")
            stability = np.asarray(payload["stable_scores"], dtype=np.float64)
            if stability.shape != (len(samples), 5):
                raise ValueError("stable_scores must be [samples,5]")
    candidate_probabilities = apply_temperature(
        setrank["probabilities"].mean(axis=0),
        candidate_probability_temperature,
    )
    mean_setrank_scores = setrank["scores"].mean(axis=0)
    records = []
    rankings = {}
    full_gate_rankings = {}
    fallback_counts: dict[str, int] = {}
    for sample_index, sample in enumerate(samples):
        per_seed = [
            prediction[sample.sample_id]
            for prediction in gate_predictions
        ]
        mean_probability = np.mean(per_seed, axis=0)
        proposed = select_with_gate(
            sample,
            mean_probability,
            harm_cost=policy["harm_cost"],
            threshold=policy["threshold"],
        )
        proposed_index = int(proposed["selected_index"])
        seed_indices = [
            int(
                select_with_gate(
                    sample,
                    probability,
                    harm_cost=policy["harm_cost"],
                    threshold=policy["threshold"],
                )["selected_index"]
            )
            for probability in per_seed
        ]
        consensus = seed_indices.count(proposed_index)
        reason = None
        if proposed_index:
            challenger = proposed_index - 1
            seed_gains = np.asarray(
                [
                    value[challenger, 0]
                    - float(policy["harm_cost"]) * value[challenger, 1]
                    for value in per_seed
                ],
                dtype=np.float64,
            )
            lower = float(
                seed_gains.mean()
                - float(uncertainty_kappa) * seed_gains.std()
            )
            if lower <= float(policy["threshold"]):
                reason = "gain_lower_bound"
            elif consensus < int(required_consensus):
                reason = "ensemble_disagreement"
            elif (
                stability is not None
                and stability[sample_index, proposed_index]
                <= stability[sample_index, 0]
            ):
                reason = "perturbation_instability"
        else:
            lower = float(proposed["best_gain"])
            reason = "gate_threshold"
        selected_index = 0 if reason is not None else proposed_index
        if reason is not None:
            fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
        setrank_order = sorted(
            range(5),
            key=lambda index: (
                -float(mean_setrank_scores[sample_index, index]),
                index,
            ),
        )
        full_gate_order = [proposed_index] + [
            index for index in setrank_order if index != proposed_index
        ]
        full_gate_rankings[sample.sample_id] = [
            str(sample.feature["candidates"][index]["candidate_id"])
            for index in full_gate_order
        ]
        order = [selected_index] + [
            index for index in setrank_order if index != selected_index
        ]
        candidate_ids = [
            str(sample.feature["candidates"][index]["candidate_id"])
            for index in order
        ]
        rankings[sample.sample_id] = candidate_ids
        records.append(
            {
                "sample_id": sample.sample_id,
                "candidate_order": candidate_ids,
                "candidate_correctness_probabilities": (
                    candidate_probabilities[sample_index].astype(float).tolist()
                ),
                "selection": {
                    "selected_index": selected_index,
                    "selected_candidate_id": candidate_ids[0],
                    "proposed_index": proposed_index,
                    "switched": selected_index != 0,
                    "fallback_reason": reason,
                    "gain_lower_bound": lower,
                    "ensemble_consensus": consensus,
                    "seed_selected_indices": seed_indices,
                },
            }
        )
    prediction_path = output_dir / "predictions.jsonl"
    atomic_write_jsonl(prediction_path, records)
    full_gate_path = output_dir / "components" / "full_feature_gate.jsonl"
    write_rankings(
        full_gate_path,
        samples,
        full_gate_rankings,
        probabilities=candidate_probabilities,
        metadata={
            "method": "full_feature_gate",
            "harm_cost": float(policy["harm_cost"]),
            "threshold": float(policy["threshold"]),
        },
    )
    component_paths["full_feature_gate"] = str(full_gate_path.resolve())
    diagnostic_path = output_dir / "ensemble_diagnostics.npz"
    atomic_savez_compressed(
        diagnostic_path,
        sample_ids=arrays["sample_ids"],
        critic_scores=base["critic_scores"],
        latent_scores=base["latent_scores"],
        setrank_scores=setrank["scores"],
        setrank_probabilities=setrank["probabilities"],
    )
    result = {
        "kind": "primary_v2_predictions",
        "sample_count": len(samples),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "component_prediction_paths": component_paths,
        "diagnostic_path": str(diagnostic_path.resolve()),
        "policy": policy,
        "alpha": float(alpha),
        "uncertainty_kappa": float(uncertainty_kappa),
        "required_consensus": int(required_consensus),
        "candidate_probability_temperature": float(
            candidate_probability_temperature
        ),
        "stability_path": (
            None
            if stability_path is None
            else str(Path(stability_path).resolve())
        ),
        "fallback_counts": fallback_counts,
        "model_sha256": {
            "critic": [sha256_file(path) for path in critic_models],
            "latent": [sha256_file(path) for path in latent_models],
            "setrank": [sha256_file(path) for path in setrank_models],
            "gate": [sha256_file(path) for path in gate_models],
        },
    }
    atomic_write_json(output_dir / "summary.json", result)
    return result


def predict_scalar_gate_ensemble(
    *,
    samples: list[FrozenInferenceSample],
    gate_models: list[str | Path],
    policy: dict[str, Any],
    output_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Run a label-free scalar-only gate ensemble on frozen candidates."""
    if not gate_models:
        raise ValueError("at least one scalar gate model is required")
    predictions = [
        predict_gate(
            load_local_model_artifact(path),
            samples,
            device=device,
        )
        for path in gate_models
    ]
    rankings: dict[str, list[str]] = {}
    switch_count = 0
    for sample in samples:
        mean_probability = np.mean(
            [prediction[sample.sample_id] for prediction in predictions],
            axis=0,
        )
        selected = select_with_gate(
            sample,
            mean_probability,
            harm_cost=float(policy["harm_cost"]),
            threshold=float(policy["threshold"]),
        )
        selected_index = int(selected["selected_index"])
        switch_count += int(selected_index != 0)
        order = [selected_index] + [
            index for index in range(5) if index != selected_index
        ]
        rankings[sample.sample_id] = [
            str(sample.feature["candidates"][index]["candidate_id"])
            for index in order
        ]
    prediction_path = write_rankings(
        output_path,
        samples,
        rankings,
        probabilities=np.asarray(
            [
                [
                    float(candidate["q_raw"])
                    for candidate in sample.feature["candidates"]
                ]
                for sample in samples
            ],
            dtype=np.float64,
        ),
        metadata={
            "method": "scalar_gate",
            "seed_ensemble": len(gate_models),
            "harm_cost": float(policy["harm_cost"]),
            "threshold": float(policy["threshold"]),
        },
    )
    return {
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "switch_count": switch_count,
        "model_sha256": [sha256_file(path) for path in gate_models],
        "policy": {
            "harm_cost": float(policy["harm_cost"]),
            "threshold": float(policy["threshold"]),
        },
    }


def export_component_rankings(
    *,
    samples: list[FrozenInferenceSample],
    base: dict[str, np.ndarray],
    setrank: dict[str, np.ndarray],
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    definitions = {
        "rgbd_critic": (
            base["critic_scores"].mean(axis=0),
            _sigmoid(base["critic_scores"]).mean(axis=0),
            int(base["critic_scores"].shape[0]),
        ),
        "latent_roi_residual": (
            base["latent_scores"].mean(axis=0),
            _sigmoid(base["latent_scores"]).mean(axis=0),
            int(base["latent_scores"].shape[0]),
        ),
        "residual_setrank": (
            setrank["scores"].mean(axis=0),
            setrank["probabilities"].mean(axis=0),
            int(setrank["scores"].shape[0]),
        ),
    }
    result = {}
    for name, (scores, probabilities, seed_count) in definitions.items():
        path = output_dir / f"{name}.jsonl"
        write_rankings(
            path,
            samples,
            _stable_score_order(samples, scores),
            probabilities=probabilities,
            metadata={"method": name, "seed_ensemble": seed_count},
        )
        result[name] = str(path.resolve())
    atomic_write_json(output_dir / "summary.json", result)
    return result
