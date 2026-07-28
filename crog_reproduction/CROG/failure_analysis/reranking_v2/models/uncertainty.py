from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..aligned_crops import CROP_CHANNELS, build_aligned_crop


PERTURBATIONS = {
    "center_px": (-4, -2, 0, 2, 4),
    "angle_deg": (-10, -5, 0, 5, 10),
    "width_fraction": (-0.10, -0.05, 0.0, 0.05, 0.10),
}


def perturbation_geometries(
    candidate: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return internal scoring geometries; none are output candidates."""
    result = [({"kind": "base", "value": 0.0}, dict(candidate))]
    for delta in PERTURBATIONS["center_px"]:
        if not delta:
            continue
        for axis in ("x", "y"):
            geometry = dict(candidate)
            geometry["cx"] = float(candidate["cx"]) + (
                float(delta) if axis == "x" else 0.0
            )
            geometry["cy"] = float(candidate["cy"]) + (
                float(delta) if axis == "y" else 0.0
            )
            geometry["col"] = int(round(geometry["cx"]))
            geometry["row"] = int(round(geometry["cy"]))
            result.append(
                (
                    {"kind": f"center_{axis}", "value": float(delta)},
                    geometry,
                )
            )
    for delta in PERTURBATIONS["angle_deg"]:
        if not delta:
            continue
        geometry = dict(candidate)
        angle = (
            float(candidate["angle_deg"]) + float(delta) + 90.0
        ) % 180.0 - 90.0
        geometry["angle_deg"] = angle
        geometry["angle_rad"] = float(np.deg2rad(angle))
        result.append(({"kind": "angle", "value": float(delta)}, geometry))
    for fraction in PERTURBATIONS["width_fraction"]:
        if not fraction:
            continue
        geometry = dict(candidate)
        geometry["width_px"] = max(
            1.0, float(candidate["width_px"]) * (1.0 + float(fraction))
        )
        result.append(
            ({"kind": "width_fraction", "value": float(fraction)}, geometry)
        )
    return result


def score_candidate_perturbations(
    candidate: dict[str, Any],
    *,
    scorer: Callable[[np.ndarray], np.ndarray],
    crop_inputs: dict[str, Any],
    output_size: int = 32,
    kappa: float = 1.0,
) -> dict[str, Any]:
    """Batch-score fixed perturbations and retain only compact statistics."""
    definitions = perturbation_geometries(candidate)
    crops = []
    valid = []
    for _, geometry in definitions:
        try:
            crop, metadata = build_aligned_crop(
                geometry, output_size=output_size, **crop_inputs
            )
            crops.append(crop)
            valid.append(bool(np.isfinite(crop).all()))
        except (ValueError, FloatingPointError):
            crops.append(
                np.full(
                    (
                        len(CROP_CHANNELS),
                        int(output_size),
                        int(output_size),
                    ),
                    np.nan,
                    dtype=np.float32,
                )
            )
            valid.append(False)
    scores = np.full(len(crops), np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        batch = np.stack([crops[index] for index in valid_indices])
        observed = np.asarray(scorer(batch), dtype=np.float64).reshape(-1)
        if observed.shape != (len(valid_indices),):
            raise ValueError("perturbation scorer must return one score per crop")
        scores[valid_indices] = observed
    stats = stability_statistics(scores, kappa=kappa)
    return {
        "perturbation_count": len(definitions),
        "definitions": [definition for definition, _ in definitions],
        "statistics": stats,
    }


def stability_statistics(scores: np.ndarray, kappa: float) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return {
            "mean": 0.0,
            "standard_deviation": 0.0,
            "minimum": 0.0,
            "valid_fraction": 0.0,
            "stability_penalty": 0.0,
            "stable_score": 0.0,
        }
    values = values[valid]
    mean = float(values.mean())
    standard_deviation = float(values.std())
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": float(values.min()),
        "valid_fraction": float(valid.mean()),
        "stability_penalty": float(kappa) * standard_deviation,
        "stable_score": mean - float(kappa) * standard_deviation,
    }


def ensemble_statistics(seed_scores: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(seed_scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("ensemble scores must be [seeds,candidates]")
    winners = np.argmax(values, axis=1)
    vote_counts = np.bincount(winners, minlength=values.shape[1])
    return {
        "mean": values.mean(axis=0),
        "standard_deviation": values.std(axis=0),
        "vote_counts": vote_counts,
        "consensus": int(vote_counts.max()),
        "winner": int(np.flatnonzero(vote_counts == vote_counts.max())[0]),
    }


def conservative_uncertainty_switch(
    original_index: int,
    proposed_index: int,
    *,
    gain_lower_bound: float,
    threshold: float,
    consensus: int,
    required_consensus: int,
) -> int:
    if (
        float(gain_lower_bound) > float(threshold)
        and int(consensus) >= int(required_consensus)
    ):
        return int(proposed_index)
    return int(original_index)
