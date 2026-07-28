from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_savez_compressed
from .datasets import JoinedSample
from .enhanced_data import flatten_candidate_arrays
from .models.latent_residual import (
    predict_latent_residual_arrays,
    train_latent_residual_arrays,
)
from .models.pairwise_gate import predict_gate, train_gate, tune_gate_policy
from .models.rgbd_critic import (
    predict_critic_arrays,
    train_critic_arrays,
)
from .models.setrank import (
    predict_setrank_arrays,
    train_setrank_arrays,
)
from .protocol import fold_lookup
from .schema import atomic_write_json, atomic_write_jsonl, sha256_file
from .training import _atomic_torch_save, rankings_from_gate


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def _fold_ids(samples: list[JoinedSample], split_manifest: str | Path):
    lookup = fold_lookup(split_manifest)
    result = np.asarray([lookup[sample.sample_id] for sample in samples], dtype=np.int64)
    if set(result.tolist()) != {0, 1, 2}:
        raise ValueError("expected exactly three non-empty OOF folds")
    return result


def _write_score_rankings(
    path: Path,
    samples: list[JoinedSample],
    scores: np.ndarray,
    probabilities: np.ndarray | None,
    *,
    method: str,
) -> Path:
    if path.exists():
        raise FileExistsError(path)
    scores = np.asarray(scores)
    if scores.shape != (len(samples), 5):
        raise ValueError(f"{method} validation scores must be [samples,5]")
    records = []
    for sample_index, sample in enumerate(samples):
        order = sorted(
            range(5),
            key=lambda candidate: (
                -float(scores[sample_index, candidate]),
                candidate,
                str(
                    sample.feature["candidates"][candidate][
                        "candidate_id"
                    ]
                ),
            ),
        )
        record = {
            "sample_id": sample.sample_id,
            "candidate_order": [
                str(sample.feature["candidates"][index]["candidate_id"])
                for index in order
            ],
            "inference_metadata": {"method": method},
        }
        if probabilities is not None:
            record["candidate_correctness_probabilities"] = (
                np.asarray(
                    probabilities[sample_index], dtype=float
                ).tolist()
            )
        records.append(record)
    atomic_write_jsonl(path, records)
    return path


def validate_oof_provenance_records(
    records: list[dict[str, Any]],
    *,
    checkpoint_field: str,
    expected_checkpoints: int,
) -> None:
    if not records:
        raise AssertionError("OOF provenance is empty")
    for record in records:
        heldout = int(record["heldout_fold"])
        checkpoints = record[checkpoint_field]
        if len(checkpoints) != int(expected_checkpoints):
            raise AssertionError("OOF checkpoint count is incomplete")
        for checkpoint in checkpoints:
            if heldout in set(map(int, checkpoint["fit_folds"])):
                raise AssertionError(
                    f"OOF checkpoint saw its held-out fold for {record['sample_id']}"
                )


def train_oof_base_models(
    *,
    train_samples: list[JoinedSample],
    validation_samples: list[JoinedSample],
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    split_manifest: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, int, int] = (31, 37, 43),
    device: str = "auto",
    critic_epochs: int = 10,
    latent_epochs: int = 12,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=resume)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=resume)
    fold_ids = _fold_ids(train_samples, split_manifest)
    train_count = len(train_samples)
    validation_count = len(validation_samples)
    oof_critic_scores = np.empty((len(seeds), train_count, 5), dtype=np.float32)
    oof_critic_embeddings = np.empty(
        (len(seeds), train_count, 5, 64), dtype=np.float16
    )
    oof_latent_scores = np.empty((len(seeds), train_count, 5), dtype=np.float32)
    oof_latent_residuals = np.empty_like(oof_latent_scores)
    provenance = {
        sample.sample_id: {
            "sample_id": sample.sample_id,
            "heldout_fold": int(fold_ids[index]),
            "base_checkpoints": [],
        }
        for index, sample in enumerate(train_samples)
    }
    validation_flat = flatten_candidate_arrays(validation_arrays)
    train_flat = flatten_candidate_arrays(train_arrays)
    for fold in range(3):
        fit = np.flatnonzero(fold_ids != fold)
        heldout = np.flatnonzero(fold_ids == fold)
        fit_candidates = (
            fit[:, None] * 5 + np.arange(5, dtype=np.int64)[None, :]
        ).reshape(-1)
        heldout_candidates = (
            heldout[:, None] * 5
            + np.arange(5, dtype=np.int64)[None, :]
        ).reshape(-1)
        if set(train_samples[index].sequence_id for index in fit) & set(
            train_samples[index].sequence_id for index in heldout
        ):
            raise AssertionError("OOF fold leaked a capture sequence")
        for seed_index, seed in enumerate(seeds):
            critic_path = model_dir / f"critic_fold{fold}_seed{seed}.pt"
            critic = (
                torch.load(
                    critic_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if resume and critic_path.exists()
                else train_critic_arrays(
                    train_flat["crops"],
                    train_flat["labels"],
                    train_flat["sample_index"],
                    train_flat["q"],
                    validation_flat["crops"],
                    validation_flat["labels"],
                    validation_flat["sample_index"],
                    validation_flat["q"],
                    channels=tuple(
                        range(train_arrays["crops"].shape[2])
                    ),
                    seed=seed + 100 * fold,
                    device=device,
                    epochs=critic_epochs,
                    patience=3,
                    train_candidate_indices=fit_candidates,
                )
            )
            if not critic_path.exists():
                _atomic_torch_save(critic, critic_path)
            critic_scores, critic_embeddings = predict_critic_arrays(
                critic,
                train_flat["crops"],
                device=device,
                candidate_indices=heldout_candidates,
            )
            oof_critic_scores[seed_index, heldout] = critic_scores.reshape(-1, 5)
            oof_critic_embeddings[seed_index, heldout] = critic_embeddings.reshape(
                -1, 5, 64
            )

            latent_path = model_dir / f"latent_fold{fold}_seed{seed}.pt"
            latent = (
                torch.load(
                    latent_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if resume and latent_path.exists()
                else train_latent_residual_arrays(
                    train_arrays["latent_post"][fit],
                    train_arrays["q"][fit],
                    train_arrays["labels"][fit],
                    validation_arrays["latent_post"],
                    validation_arrays["q"],
                    validation_arrays["labels"],
                    seed=seed + 200 * fold,
                    device=device,
                    epochs=latent_epochs,
                    patience=3,
                )
            )
            if not latent_path.exists():
                _atomic_torch_save(latent, latent_path)
            latent_scores, latent_residuals = predict_latent_residual_arrays(
                latent,
                train_arrays["latent_post"][heldout],
                train_arrays["q"][heldout],
                alpha=1.0,
                device=device,
            )
            oof_latent_scores[seed_index, heldout] = latent_scores
            oof_latent_residuals[seed_index, heldout] = latent_residuals
            for index in heldout:
                provenance[train_samples[index].sample_id][
                    "base_checkpoints"
                ].append(
                    {
                        "seed": int(seed),
                        "critic": str(critic_path.resolve()),
                        "critic_sha256": sha256_file(critic_path),
                        "latent": str(latent_path.resolve()),
                        "latent_sha256": sha256_file(latent_path),
                        "fit_folds": [
                            value for value in range(3) if value != fold
                        ],
                    }
                )
    validate_oof_provenance_records(
        list(provenance.values()),
        checkpoint_field="base_checkpoints",
        expected_checkpoints=len(seeds),
    )

    validation_critic_scores = []
    validation_critic_embeddings = []
    validation_latent_scores = []
    validation_latent_residuals = []
    final_model_paths = []
    final_critic_paths = []
    final_latent_paths = []
    for seed in seeds:
        critic_path = model_dir / f"critic_final_seed{seed}.pt"
        critic = (
            torch.load(
                critic_path,
                map_location="cpu",
                weights_only=False,
            )
            if resume and critic_path.exists()
            else train_critic_arrays(
                train_flat["crops"],
                train_flat["labels"],
                train_flat["sample_index"],
                train_flat["q"],
                validation_flat["crops"],
                validation_flat["labels"],
                validation_flat["sample_index"],
                validation_flat["q"],
                channels=tuple(range(train_arrays["crops"].shape[2])),
                seed=seed,
                device=device,
                epochs=critic_epochs,
                patience=3,
            )
        )
        if not critic_path.exists():
            _atomic_torch_save(critic, critic_path)
        scores, embeddings = predict_critic_arrays(
            critic, validation_flat["crops"], device=device
        )
        validation_critic_scores.append(scores.reshape(validation_count, 5))
        validation_critic_embeddings.append(
            embeddings.reshape(validation_count, 5, 64)
        )
        latent_path = model_dir / f"latent_final_seed{seed}.pt"
        latent = (
            torch.load(
                latent_path,
                map_location="cpu",
                weights_only=False,
            )
            if resume and latent_path.exists()
            else train_latent_residual_arrays(
                train_arrays["latent_post"],
                train_arrays["q"],
                train_arrays["labels"],
                validation_arrays["latent_post"],
                validation_arrays["q"],
                validation_arrays["labels"],
                seed=seed,
                device=device,
                epochs=latent_epochs,
                patience=3,
            )
        )
        if not latent_path.exists():
            _atomic_torch_save(latent, latent_path)
        latent_scores, latent_residuals = predict_latent_residual_arrays(
            latent,
            validation_arrays["latent_post"],
            validation_arrays["q"],
            alpha=1.0,
            device=device,
        )
        validation_latent_scores.append(latent_scores)
        validation_latent_residuals.append(latent_residuals)
        final_model_paths.extend((critic_path, latent_path))
        final_critic_paths.append(critic_path)
        final_latent_paths.append(latent_path)
    validation_critic_scores = np.stack(validation_critic_scores)
    validation_critic_embeddings = np.stack(validation_critic_embeddings)
    validation_latent_scores = np.stack(validation_latent_scores)
    validation_latent_residuals = np.stack(validation_latent_residuals)
    component_dir = output_dir / "validation_components"
    component_dir.mkdir(exist_ok=resume)
    critic_prediction_path = component_dir / "rgbd_critic.jsonl"
    if not (resume and critic_prediction_path.exists()):
        _write_score_rankings(
            critic_prediction_path,
            validation_samples,
            validation_critic_scores.mean(axis=0),
            _sigmoid(validation_critic_scores).mean(axis=0),
            method="rgbd_critic",
        )
    latent_prediction_path = component_dir / "latent_roi_residual.jsonl"
    if not (resume and latent_prediction_path.exists()):
        _write_score_rankings(
            latent_prediction_path,
            validation_samples,
            validation_latent_scores.mean(axis=0),
            _sigmoid(validation_latent_scores).mean(axis=0),
            method="latent_roi_residual",
        )
    oof_path = output_dir / "oof_base_predictions.npz"
    atomic_savez_compressed(
        oof_path,
        sample_ids=train_arrays["sample_ids"],
        fold_ids=fold_ids,
        critic_scores=oof_critic_scores,
        critic_embeddings=oof_critic_embeddings,
        latent_scores=oof_latent_scores,
        latent_residuals=oof_latent_residuals,
    )
    validation_path = output_dir / "validation_base_predictions.npz"
    atomic_savez_compressed(
        validation_path,
        sample_ids=validation_arrays["sample_ids"],
        critic_scores=validation_critic_scores,
        critic_embeddings=validation_critic_embeddings.astype(np.float16),
        latent_scores=validation_latent_scores,
        latent_residuals=validation_latent_residuals,
    )
    provenance_path = output_dir / "oof_provenance.jsonl"
    if not (resume and provenance_path.exists()):
        atomic_write_jsonl(
            provenance_path,
            (provenance[sample.sample_id] for sample in train_samples),
        )
    result = {
        "seeds": list(seeds),
        "folds": 3,
        "train_samples": train_count,
        "validation_samples": validation_count,
        "oof_predictions": str(oof_path.resolve()),
        "validation_predictions": str(validation_path.resolve()),
        "oof_provenance": str(provenance_path.resolve()),
        "final_models": [str(path.resolve()) for path in final_model_paths],
        "final_critic_models": [
            str(path.resolve()) for path in final_critic_paths
        ],
        "final_latent_models": [
            str(path.resolve()) for path in final_latent_paths
        ],
        "validation_component_predictions": {
            "rgbd_critic": str(critic_prediction_path.resolve()),
            "latent_roi_residual": str(latent_prediction_path.resolve()),
        },
    }
    atomic_write_json(output_dir / "summary.json", result)
    return result


def _full_tokens(
    arrays: dict[str, np.ndarray],
    *,
    critic_scores: np.ndarray,
    critic_embeddings: np.ndarray,
    latent_scores: np.ndarray,
    latent_residuals: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(arrays["scalar"], dtype=np.float32),
            np.asarray(critic_scores, dtype=np.float32)[..., None],
            _sigmoid(critic_scores).astype(np.float32)[..., None],
            np.asarray(critic_embeddings, dtype=np.float32),
            np.asarray(latent_scores, dtype=np.float32)[..., None],
            np.asarray(latent_residuals, dtype=np.float32)[..., None],
        ),
        axis=-1,
    )


def build_token_ablation_artifacts(
    *,
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    base_oof_path: str | Path,
    base_validation_path: str | Path,
    output_dir: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Build leakage-safe SetRank token ablations from OOF base predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=resume)
    with np.load(base_oof_path) as payload:
        if list(map(str, payload["sample_ids"])) != list(
            map(str, train_arrays["sample_ids"])
        ):
            raise ValueError("OOF base/train sample order mismatch")
        train_base = {
            "critic_scores": payload["critic_scores"].mean(axis=0),
            "critic_embeddings": payload["critic_embeddings"].mean(axis=0),
            "latent_scores": payload["latent_scores"].mean(axis=0),
            "latent_residuals": payload["latent_residuals"].mean(axis=0),
        }
    with np.load(base_validation_path) as payload:
        if list(map(str, payload["sample_ids"])) != list(
            map(str, validation_arrays["sample_ids"])
        ):
            raise ValueError("base validation/sample order mismatch")
        validation_base = {
            "critic_scores": payload["critic_scores"].mean(axis=0),
            "critic_embeddings": payload["critic_embeddings"].mean(axis=0),
            "latent_scores": payload["latent_scores"].mean(axis=0),
            "latent_residuals": payload["latent_residuals"].mean(axis=0),
        }

    def variants(arrays, base):
        scalar = np.asarray(arrays["scalar"], dtype=np.float32)
        critic = np.concatenate(
            (
                scalar,
                np.asarray(base["critic_scores"], dtype=np.float32)[..., None],
                _sigmoid(base["critic_scores"]).astype(np.float32)[..., None],
                np.asarray(base["critic_embeddings"], dtype=np.float32),
            ),
            axis=-1,
        )
        latent = np.concatenate(
            (
                scalar,
                np.asarray(base["latent_scores"], dtype=np.float32)[..., None],
                np.asarray(base["latent_residuals"], dtype=np.float32)[..., None],
            ),
            axis=-1,
        )
        return {
            "scalar": scalar,
            "scalar_critic": critic,
            "scalar_latent": latent,
            "scalar_critic_latent": _full_tokens(
                arrays,
                critic_scores=base["critic_scores"],
                critic_embeddings=base["critic_embeddings"],
                latent_scores=base["latent_scores"],
                latent_residuals=base["latent_residuals"],
            ),
        }

    train_variants = variants(train_arrays, train_base)
    validation_variants = variants(validation_arrays, validation_base)
    outputs = {}
    for name in train_variants:
        path = output_dir / f"{name}.npz"
        atomic_savez_compressed(
            path,
            train_tokens=train_variants[name],
            validation_tokens=validation_variants[name],
            train_sample_ids=train_arrays["sample_ids"],
            validation_sample_ids=validation_arrays["sample_ids"],
        )
        outputs[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "token_dim": int(train_variants[name].shape[-1]),
        }
    result = {
        "kind": "setrank_token_ablations",
        "train_samples": int(len(train_arrays["sample_ids"])),
        "validation_samples": int(len(validation_arrays["sample_ids"])),
        "outputs": outputs,
    }
    atomic_write_json(output_dir / "summary.json", result)
    return result


def train_oof_setrank_and_gate(
    *,
    train_samples: list[JoinedSample],
    validation_samples: list[JoinedSample],
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    base_oof_path: str | Path,
    base_validation_path: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, int, int] = (31, 37, 43),
    device: str = "auto",
    setrank_epochs: int = 12,
    gate_epochs: int = 25,
    resume: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=resume)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=resume)
    expected_fold_ids = _fold_ids(train_samples, split_manifest)
    with np.load(base_oof_path) as payload:
        if list(map(str, payload["sample_ids"])) != list(
            map(str, train_arrays["sample_ids"])
        ):
            raise ValueError("base OOF/train sample order mismatch")
        oof_critic_scores = payload["critic_scores"]
        oof_critic_embeddings = payload["critic_embeddings"]
        oof_latent_scores = payload["latent_scores"]
        oof_latent_residuals = payload["latent_residuals"]
        fold_ids = payload["fold_ids"]
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("base OOF fold assignment differs from split manifest")
    with np.load(base_validation_path) as payload:
        if list(map(str, payload["sample_ids"])) != list(
            map(str, validation_arrays["sample_ids"])
        ):
            raise ValueError("base validation/sample order mismatch")
        validation_critic_scores = payload["critic_scores"]
        validation_critic_embeddings = payload["critic_embeddings"]
        validation_latent_scores = payload["latent_scores"]
        validation_latent_residuals = payload["latent_residuals"]
    expected_train_score_shape = (len(seeds), len(train_samples), 5)
    expected_validation_score_shape = (
        len(seeds),
        len(validation_samples),
        5,
    )
    for name, values in {
        "oof_critic_scores": oof_critic_scores,
        "oof_latent_scores": oof_latent_scores,
        "oof_latent_residuals": oof_latent_residuals,
    }.items():
        if values.shape != expected_train_score_shape:
            raise ValueError(f"{name} shape mismatch: {values.shape}")
    for name, values in {
        "validation_critic_scores": validation_critic_scores,
        "validation_latent_scores": validation_latent_scores,
        "validation_latent_residuals": validation_latent_residuals,
    }.items():
        if values.shape != expected_validation_score_shape:
            raise ValueError(f"{name} shape mismatch: {values.shape}")
    # Use seed-ensemble means as the stable token features.
    train_tokens = _full_tokens(
        train_arrays,
        critic_scores=oof_critic_scores.mean(axis=0),
        critic_embeddings=oof_critic_embeddings.mean(axis=0),
        latent_scores=oof_latent_scores.mean(axis=0),
        latent_residuals=oof_latent_residuals.mean(axis=0),
    )
    validation_tokens = _full_tokens(
        validation_arrays,
        critic_scores=validation_critic_scores.mean(axis=0),
        critic_embeddings=validation_critic_embeddings.mean(axis=0),
        latent_scores=validation_latent_scores.mean(axis=0),
        latent_residuals=validation_latent_residuals.mean(axis=0),
    )
    train_count = len(train_samples)
    validation_count = len(validation_samples)
    oof_setrank_scores = np.empty((len(seeds), train_count, 5), dtype=np.float32)
    oof_setrank_probabilities = np.empty_like(oof_setrank_scores)
    setrank_provenance = {
        sample.sample_id: {
            "sample_id": sample.sample_id,
            "heldout_fold": int(fold_ids[index]),
            "setrank_checkpoints": [],
        }
        for index, sample in enumerate(train_samples)
    }
    for fold in range(3):
        fit = np.flatnonzero(fold_ids != fold)
        heldout = np.flatnonzero(fold_ids == fold)
        for seed_index, seed in enumerate(seeds):
            path = model_dir / f"setrank_fold{fold}_seed{seed}.pt"
            artifact = (
                torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                )
                if resume and path.exists()
                else train_setrank_arrays(
                    train_tokens[fit],
                    train_arrays["q"][fit],
                    train_arrays["labels"][fit],
                    validation_tokens,
                    validation_arrays["q"],
                    validation_arrays["labels"],
                    seed=seed + 300 * fold,
                    device=device,
                    epochs=setrank_epochs,
                    patience=3,
                )
            )
            if not path.exists():
                _atomic_torch_save(artifact, path)
            scores, probabilities, _ = predict_setrank_arrays(
                artifact,
                train_tokens[heldout],
                train_arrays["q"][heldout],
                device=device,
            )
            oof_setrank_scores[seed_index, heldout] = scores
            oof_setrank_probabilities[seed_index, heldout] = probabilities
            for index in heldout:
                setrank_provenance[train_samples[index].sample_id][
                    "setrank_checkpoints"
                ].append(
                    {
                        "seed": seed,
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "fit_folds": [
                            value for value in range(3) if value != fold
                        ],
                    }
                )
    validation_setrank_scores = []
    validation_setrank_probabilities = []
    final_setrank_paths = []
    for seed in seeds:
        path = model_dir / f"setrank_final_seed{seed}.pt"
        artifact = (
            torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            if resume and path.exists()
            else train_setrank_arrays(
                train_tokens,
                train_arrays["q"],
                train_arrays["labels"],
                validation_tokens,
                validation_arrays["q"],
                validation_arrays["labels"],
                seed=seed,
                device=device,
                epochs=setrank_epochs,
                patience=3,
            )
        )
        if not path.exists():
            _atomic_torch_save(artifact, path)
        scores, probabilities, _ = predict_setrank_arrays(
            artifact, validation_tokens, validation_arrays["q"], device=device
        )
        validation_setrank_scores.append(scores)
        validation_setrank_probabilities.append(probabilities)
        final_setrank_paths.append(path)
    validation_setrank_scores = np.stack(validation_setrank_scores)
    validation_setrank_probabilities = np.stack(
        validation_setrank_probabilities
    )
    component_dir = output_dir / "validation_components"
    component_dir.mkdir(exist_ok=resume)
    setrank_prediction_path = component_dir / "residual_setrank.jsonl"
    if not (resume and setrank_prediction_path.exists()):
        _write_score_rankings(
            setrank_prediction_path,
            validation_samples,
            validation_setrank_scores.mean(axis=0),
            validation_setrank_probabilities.mean(axis=0),
            method="residual_setrank",
        )
    oof_path = output_dir / "oof_setrank_predictions.npz"
    atomic_savez_compressed(
        oof_path,
        sample_ids=train_arrays["sample_ids"],
        scores=oof_setrank_scores,
        probabilities=oof_setrank_probabilities,
    )
    validation_path = output_dir / "validation_setrank_predictions.npz"
    atomic_savez_compressed(
        validation_path,
        sample_ids=validation_arrays["sample_ids"],
        scores=validation_setrank_scores,
        probabilities=validation_setrank_probabilities,
    )
    provenance_path = output_dir / "oof_setrank_provenance.jsonl"
    if not (resume and provenance_path.exists()):
        atomic_write_jsonl(
            provenance_path,
            (
                setrank_provenance[sample.sample_id]
                for sample in train_samples
            ),
        )
    validate_oof_provenance_records(
        list(setrank_provenance.values()),
        checkpoint_field="setrank_checkpoints",
        expected_checkpoints=len(seeds),
    )

    train_extra = np.stack(
        (
            oof_critic_scores.mean(axis=0),
            oof_critic_scores.std(axis=0),
            _sigmoid(oof_critic_scores).mean(axis=0),
            oof_latent_scores.mean(axis=0),
            oof_latent_scores.std(axis=0),
            oof_latent_residuals.mean(axis=0),
            oof_setrank_scores.mean(axis=0),
            oof_setrank_scores.std(axis=0),
            oof_setrank_probabilities.mean(axis=0),
        ),
        axis=-1,
    )
    validation_extra = np.stack(
        (
            validation_critic_scores.mean(axis=0),
            validation_critic_scores.std(axis=0),
            _sigmoid(validation_critic_scores).mean(axis=0),
            validation_latent_scores.mean(axis=0),
            validation_latent_scores.std(axis=0),
            validation_latent_residuals.mean(axis=0),
            validation_setrank_scores.mean(axis=0),
            validation_setrank_scores.std(axis=0),
            validation_setrank_probabilities.mean(axis=0),
        ),
        axis=-1,
    )
    train_extra_lookup = {
        sample.sample_id: train_extra[index]
        for index, sample in enumerate(train_samples)
    }
    validation_extra_lookup = {
        sample.sample_id: validation_extra[index]
        for index, sample in enumerate(validation_samples)
    }
    gate_probabilities = []
    final_gate_paths = []
    for seed in seeds:
        path = model_dir / f"gate_final_seed{seed}.pt"
        gate = (
            torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            if resume and path.exists()
            else train_gate(
                train_samples,
                validation_samples,
                train_extra=train_extra_lookup,
                validation_extra=validation_extra_lookup,
                seed=seed,
                device=device,
                epochs=gate_epochs,
                patience=4,
            )
        )
        if not path.exists():
            _atomic_torch_save(gate, path)
        gate_probabilities.append(
            predict_gate(
                gate,
                validation_samples,
                extra_candidate_features=validation_extra_lookup,
                device=device,
            )
        )
        final_gate_paths.append(path)
    mean_probabilities = {}
    for sample in validation_samples:
        mean_probabilities[sample.sample_id] = np.mean(
            [values[sample.sample_id] for values in gate_probabilities], axis=0
        )
    tuning = tune_gate_policy(
        validation_samples,
        mean_probabilities,
        bootstrap_iterations=2000,
        seed=seeds[0] + 999,
    )
    rankings, selections, confidence = rankings_from_gate(
        validation_samples, mean_probabilities, tuning["selected"]
    )
    # Add seed agreement without changing the validation-selected policy.
    for sample in validation_samples:
        seed_selected = []
        for values in gate_probabilities:
            seed_rankings, _, _ = rankings_from_gate(
                [sample],
                {sample.sample_id: values[sample.sample_id]},
                tuning["selected"],
            )
            seed_selected.append(seed_rankings[sample.sample_id][0])
        selected = rankings[sample.sample_id][0]
        consensus = seed_selected.count(selected)
        selections[sample.sample_id]["ensemble_consensus"] = consensus
        selections[sample.sample_id]["seed_selected_candidate_ids"] = seed_selected
        if consensus < 2:
            original = sample.feature["candidates"][0]["candidate_id"]
            rankings[sample.sample_id] = [original] + [
                candidate["candidate_id"]
                for candidate in sample.feature["candidates"][1:]
            ]
            selections[sample.sample_id]["switched"] = False
            selections[sample.sample_id]["uncertainty_fallback"] = "q_only"
    prediction_path = output_dir / "validation_primary_predictions.jsonl"
    mean_validation_correctness = (
        validation_setrank_probabilities.mean(axis=0)
    )
    if not (resume and prediction_path.exists()):
        atomic_write_jsonl(
            prediction_path,
            (
                {
                    "sample_id": sample.sample_id,
                    "candidate_order": rankings[sample.sample_id],
                    "selection": selections[sample.sample_id],
                    "mean_gate_probabilities": mean_probabilities[
                        sample.sample_id
                    ].tolist(),
                    "candidate_correctness_probabilities": (
                        mean_validation_correctness[
                            validation_index
                        ].tolist()
                    ),
                }
                for validation_index, sample in enumerate(
                    validation_samples
                )
            ),
        )
    full_gate_prediction_path = component_dir / "full_feature_gate.jsonl"
    if not (resume and full_gate_prediction_path.exists()):
        atomic_write_jsonl(
            full_gate_prediction_path,
            (
                {
                    "sample_id": sample.sample_id,
                    "candidate_order": rankings[sample.sample_id],
                    "candidate_correctness_probabilities": (
                        mean_validation_correctness[
                            validation_index
                        ].tolist()
                    ),
                    "inference_metadata": {
                        "method": "full_feature_gate",
                        "ensemble_consensus_required": 2,
                    },
                }
                for validation_index, sample in enumerate(
                    validation_samples
                )
            ),
        )
    result = {
        "seeds": list(seeds),
        "oof_setrank_predictions": str(oof_path.resolve()),
        "validation_setrank_predictions": str(validation_path.resolve()),
        "oof_setrank_provenance": str(provenance_path.resolve()),
        "final_setrank_models": [
            str(path.resolve()) for path in final_setrank_paths
        ],
        "final_gate_models": [str(path.resolve()) for path in final_gate_paths],
        "gate_tuning": tuning,
        "validation_primary_predictions": str(prediction_path.resolve()),
        "validation_component_predictions": {
            "residual_setrank": str(setrank_prediction_path.resolve()),
            "full_feature_gate": str(
                full_gate_prediction_path.resolve()
            ),
        },
        "candidate_order_by_sample": rankings,
    }
    atomic_write_json(output_dir / "summary.json", result)
    return result
