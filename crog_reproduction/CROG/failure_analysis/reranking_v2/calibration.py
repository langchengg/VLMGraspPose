from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import expected_calibration_error
from .schema import atomic_write_json, read_jsonl


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7
    )
    return np.log(values) - np.log1p(-values)


def apply_temperature(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    if not np.isfinite(temperature) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = probability_logit(probabilities) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def _nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    labels = np.asarray(labels, dtype=np.float64)
    return float(
        -np.mean(
            labels * np.log(values)
            + (1.0 - labels) * np.log(1.0 - values)
        )
    )


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probabilities.shape != labels.shape or not probabilities.size:
        raise ValueError("calibration probabilities/labels must align")
    if not np.isfinite(probabilities).all() or not np.isfinite(labels).all():
        raise ValueError("calibration inputs contain NaN or Inf")
    result = minimize_scalar(
        lambda log_temperature: _nll(
            apply_temperature(probabilities, np.exp(log_temperature)),
            labels,
        ),
        bounds=(np.log(0.05), np.log(10.0)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    temperature = float(np.exp(result.x))
    calibrated = apply_temperature(probabilities, temperature)
    before_ece, _ = expected_calibration_error(probabilities, labels)
    after_ece, _ = expected_calibration_error(calibrated, labels)
    return {
        "temperature": temperature,
        "sample_count": int(probabilities.size),
        "nll_before": _nll(probabilities, labels),
        "nll_after": _nll(calibrated, labels),
        "brier_before": float(np.mean((probabilities - labels) ** 2)),
        "brier_after": float(np.mean((calibrated - labels) ** 2)),
        "ece_before": before_ece,
        "ece_after": after_ece,
        "ranking_changed": False,
    }


def fit_temperature_from_artifacts(
    *,
    predictions_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Join held-out calibration labels only after label-free inference."""
    predictions = {
        str(record["sample_id"]): record
        for record in read_jsonl(predictions_path)
    }
    labels = {
        str(record["sample_id"]): record
        for record in read_jsonl(labels_path)
    }
    missing_labels = set(predictions) - set(labels)
    if missing_labels:
        raise ValueError(
            "calibration labels are missing prediction IDs: "
            f"{sorted(missing_labels)[:5]}"
        )
    probability_rows = []
    label_rows = []
    for sample_id in sorted(predictions):
        values = predictions[sample_id].get(
            "candidate_correctness_probabilities"
        )
        if values is None or len(values) != 5:
            raise ValueError(
                f"missing candidate probabilities for {sample_id}"
            )
        candidate_labels = labels[sample_id]["candidate_labels"]
        if len(candidate_labels) != 5:
            raise ValueError(f"invalid candidate labels for {sample_id}")
        probability_rows.append(values)
        label_rows.append(
            [
                float(candidate["candidate_correct"])
                for candidate in candidate_labels
            ]
        )
    result = fit_temperature(
        np.asarray(probability_rows),
        np.asarray(label_rows),
    )
    result.update(
        {
            "kind": "candidate_probability_temperature_calibration",
            "expression_count": len(probability_rows),
            "predictions_path": str(Path(predictions_path).resolve()),
            "labels_path": str(Path(labels_path).resolve()),
        }
    )
    atomic_write_json(output_path, result)
    return result
