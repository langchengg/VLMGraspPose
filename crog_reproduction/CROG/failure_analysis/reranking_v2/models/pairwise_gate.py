from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import (
    JoinedSample,
    apply_standardizer,
    fit_standardizer,
    gate_outcome_labels,
    gate_pair_matrix,
)
from .common import (
    batches,
    clone_state_dict,
    early_stopping_update,
    resolve_device,
    seed_everything,
)


class PairwiseGainGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(int(hidden_dim), int(hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim // 2), 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _stack_pairs(
    samples: list[JoinedSample],
    *,
    extra_candidate_features: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    values, targets, locations = [], [], []
    for sample_index, sample in enumerate(samples):
        pairs = gate_pair_matrix(sample.feature)
        if extra_candidate_features is not None:
            extra = np.asarray(extra_candidate_features[sample.sample_id], dtype=np.float32)
            if extra.shape[0] != 5:
                raise ValueError("gate extra features must have K=5 rows")
            top = extra[0]
            pair_extra = np.stack(
                [
                    np.concatenate((top, extra[index], extra[index] - top))
                    for index in range(1, 5)
                ]
            )
            pairs = np.concatenate((pairs, pair_extra), axis=1)
        labels = gate_outcome_labels(sample.label)
        values.append(pairs)
        targets.append(labels)
        locations.extend((sample_index, challenger) for challenger in range(1, 5))
    return (
        np.concatenate(values).astype(np.float32),
        np.concatenate(targets).astype(np.int64),
        locations,
    )


def _stack_inference_pairs(
    samples,
    *,
    extra_candidate_features: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    values, locations = [], []
    for sample_index, sample in enumerate(samples):
        pairs = gate_pair_matrix(sample.feature)
        if extra_candidate_features is not None:
            extra = np.asarray(
                extra_candidate_features[sample.sample_id],
                dtype=np.float32,
            )
            if extra.shape[0] != 5:
                raise ValueError("gate extra features must have K=5 rows")
            top = extra[0]
            pair_extra = np.stack(
                [
                    np.concatenate(
                        (top, extra[index], extra[index] - top)
                    )
                    for index in range(1, 5)
                ]
            )
            pairs = np.concatenate((pairs, pair_extra), axis=1)
        values.append(pairs)
        locations.extend(
            (sample_index, challenger) for challenger in range(1, 5)
        )
    return np.concatenate(values).astype(np.float32), locations


def _weighted_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=weights)


def train_gate(
    train_samples: list[JoinedSample],
    validation_samples: list[JoinedSample],
    *,
    train_extra: dict[str, np.ndarray] | None = None,
    validation_extra: dict[str, np.ndarray] | None = None,
    seed: int = 17,
    device: str = "auto",
    epochs: int = 40,
    patience: int = 6,
    learning_rate: float = 1e-3,
    batch_size: int = 2048,
) -> dict[str, Any]:
    seed_everything(seed)
    train_x, train_y, _ = _stack_pairs(
        train_samples, extra_candidate_features=train_extra
    )
    validation_x, validation_y, _ = _stack_pairs(
        validation_samples, extra_candidate_features=validation_extra
    )
    standardizer = fit_standardizer([train_x])
    train_x = apply_standardizer(train_x, standardizer)
    validation_x = apply_standardizer(validation_x, standardizer)
    counts = np.bincount(train_y, minlength=3).astype(np.float64)
    class_weights = counts.sum() / np.maximum(3.0 * counts, 1.0)
    torch_device = resolve_device(device)
    model = PairwiseGainGate(train_x.shape[1]).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-5
    )
    weights = torch.tensor(class_weights, dtype=torch.float32, device=torch_device)
    val_x_tensor = torch.from_numpy(validation_x).to(torch_device)
    val_y_tensor = torch.from_numpy(validation_y).to(torch_device)
    best_value, best_state, stale = math.inf, None, 0
    history = []
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for indices in batches(
            len(train_x), batch_size, shuffle=True, seed=seed + epoch
        ):
            x = torch.from_numpy(train_x[indices]).to(torch_device)
            y = torch.from_numpy(train_y[indices]).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            loss = _weighted_cross_entropy(model(x), y, weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                _weighted_cross_entropy(
                    model(val_x_tensor), val_y_tensor, weights
                ).cpu()
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        best_value, best_state, stale = early_stopping_update(
            validation_loss,
            best_value=best_value,
            best_state=best_state,
            model=model,
            stale=stale,
        )
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("gate training produced no finite checkpoint")
    model.load_state_dict(best_state)
    return {
        "kind": "pairwise_expected_gain_gate",
        "model_state_dict": clone_state_dict(model),
        "input_dim": int(train_x.shape[1]),
        "hidden_dim": 64,
        "standardizer": standardizer,
        "class_counts": counts.astype(int),
        "class_weights": class_weights.astype(np.float32),
        "history": history,
        "seed": int(seed),
        "label_order": ("R", "H", "N"),
    }


def predict_gate(
    artifact: dict[str, Any],
    samples: list[JoinedSample],
    *,
    extra_candidate_features: dict[str, np.ndarray] | None = None,
    device: str = "auto",
) -> dict[str, np.ndarray]:
    values, locations = _stack_inference_pairs(
        samples, extra_candidate_features=extra_candidate_features
    )
    values = apply_standardizer(values, artifact["standardizer"])
    torch_device = resolve_device(device)
    model = PairwiseGainGate(
        int(artifact["input_dim"]), int(artifact.get("hidden_dim", 64))
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.to(torch_device).eval()
    probabilities = []
    with torch.no_grad():
        for indices in batches(len(values), 8192, shuffle=False, seed=0):
            logits = model(torch.from_numpy(values[indices]).to(torch_device))
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    by_sample = {
        sample.sample_id: np.zeros((4, 3), dtype=np.float32)
        for sample in samples
    }
    for probability, (sample_index, challenger) in zip(
        probabilities, locations, strict=True
    ):
        by_sample[samples[sample_index].sample_id][challenger - 1] = probability
    return by_sample


def select_with_gate(
    sample: JoinedSample,
    probabilities: np.ndarray,
    *,
    harm_cost: float,
    threshold: float,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (4, 3):
        raise ValueError("gate probabilities must be [4,3] for R/H/N")
    gain = probabilities[:, 0] - float(harm_cost) * probabilities[:, 1]
    best_gain = float(np.max(gain))
    tied = np.flatnonzero(np.isclose(gain, best_gain, atol=1e-12, rtol=0.0))
    # Higher original q-rank means the smaller candidate index.
    challenger_index = int(tied[0]) + 1
    switch = bool(best_gain > float(threshold))
    selected_index = challenger_index if switch else 0
    return {
        "selected_index": selected_index,
        "selected_candidate_id": sample.feature["candidates"][selected_index][
            "candidate_id"
        ],
        "switched": switch,
        "best_gain": best_gain,
        "challenger_index": challenger_index,
        "challenger_probabilities": probabilities[challenger_index - 1].tolist(),
        "all_gains": gain.tolist(),
    }


def _cluster_bootstrap_lower(
    deltas: np.ndarray,
    clusters: list[str],
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[str(cluster)].append(index)
    keys = sorted(grouped)
    cluster_sums = np.asarray(
        [np.asarray(deltas[grouped[key]], dtype=np.float64).sum() for key in keys]
    )
    cluster_counts = np.asarray(
        [len(grouped[key]) for key in keys], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    for start in range(0, int(iterations), 1024):
        stop = min(int(iterations), start + 1024)
        sampled = rng.integers(
            0,
            len(keys),
            size=(stop - start, len(keys)),
        )
        draws[start:stop] = (
            cluster_sums[sampled].sum(axis=1)
            / cluster_counts[sampled].sum(axis=1)
        )
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def tune_gate_policy(
    validation_samples: list[JoinedSample],
    probabilities: dict[str, np.ndarray],
    *,
    harm_costs: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0),
    thresholds: tuple[float, ...] = (
        0.0,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
    ),
    bootstrap_iterations: int = 2000,
    seed: int = 1703,
) -> dict[str, Any]:
    rows = []
    for harm_cost in harm_costs:
        for threshold in thresholds:
            deltas, switched = [], []
            for sample in validation_samples:
                selection = select_with_gate(
                    sample,
                    probabilities[sample.sample_id],
                    harm_cost=harm_cost,
                    threshold=threshold,
                )
                labels = {
                    item["candidate_id"]: bool(item["candidate_correct"])
                    for item in sample.label["candidate_labels"]
                }
                original = labels[sample.feature["candidates"][0]["candidate_id"]]
                selected = labels[selection["selected_candidate_id"]]
                deltas.append(int(selected) - int(original))
                switched.append(selection["switched"])
            deltas_array = np.asarray(deltas, dtype=np.float64)
            lower, upper = _cluster_bootstrap_lower(
                deltas_array,
                [sample.frame_id for sample in validation_samples],
                seed=seed,
                iterations=bootstrap_iterations,
            )
            recovered = int(np.sum(deltas_array == 1))
            harmful = int(np.sum(deltas_array == -1))
            rows.append(
                {
                    "harm_cost": float(harm_cost),
                    "threshold": float(threshold),
                    "delta": float(deltas_array.mean()),
                    "ci95": [lower, upper],
                    "recovered": recovered,
                    "harmful": harmful,
                    "switch_coverage": float(np.mean(switched)),
                }
            )
    # Primary objective: lower confidence bound, then net gain, then less harm,
    # then lower coverage and finally deterministic numeric hyperparameters.
    best = max(
        rows,
        key=lambda row: (
            row["ci95"][0],
            row["delta"],
            -row["harmful"],
            -row["switch_coverage"],
            -row["harm_cost"],
            -row["threshold"],
        ),
    )
    if best["ci95"][0] <= 0.0:
        best = {
            "harm_cost": float(best["harm_cost"]),
            # Gains are bounded above by 1.0, so 2.0 is a finite,
            # standards-compliant JSON sentinel that can never switch.
            "threshold": 2.0,
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "recovered": 0,
            "harmful": 0,
            "switch_coverage": 0.0,
            "fallback_reason": (
                "validation cluster-bootstrap lower bound was not positive; "
                "policy degenerates to q-only"
            ),
        }
    return {"selected": best, "ablation": rows}
