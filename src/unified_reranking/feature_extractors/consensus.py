"""Symmetric tri-backend nearest-candidate consensus features."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from .common import periodic_angle_error


BACKENDS = ("crog", "g1", "c1")


def _prepared(frame: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"consensus candidate table missing columns: {missing}")
    probability_columns = {"sample_id", "candidate_id", "calibrated_native_probability"}
    missing = sorted(probability_columns.difference(calibration.columns))
    if missing:
        raise ValueError(f"consensus calibration table missing columns: {missing}")
    result = frame[list(required)].merge(
        calibration[list(probability_columns)],
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if result["calibrated_native_probability"].isna().any():
        raise ValueError("calibration does not cover consensus candidates")
    return result


def tri_backend_consensus_features(
    *,
    anchor_route: str,
    candidates: Mapping[str, pd.DataFrame],
    calibrations: Mapping[str, pd.DataFrame],
    image_diagonal_px: float = 800.0,
    nearby_threshold_px: float = 20.0,
) -> pd.DataFrame:
    """Build one identical cross-backend schema for every anchor route."""

    if anchor_route not in BACKENDS:
        raise ValueError(f"unknown anchor route: {anchor_route}")
    if set(candidates) != set(BACKENDS) or set(calibrations) != set(BACKENDS):
        raise ValueError("candidates and calibrations must provide crog/g1/c1")
    if image_diagonal_px <= 0 or nearby_threshold_px <= 0:
        raise ValueError("distance scales must be positive")
    prepared = {
        route: _prepared(candidates[route], calibrations[route]) for route in BACKENDS
    }
    grouped = {
        route: {str(key): value.copy() for key, value in frame.groupby("sample_id", sort=False)}
        for route, frame in prepared.items()
    }
    rows: list[dict[str, object]] = []
    for anchor in prepared[anchor_route].itertuples(index=False):
        anchor_xy = np.asarray([float(anchor.cx_px), float(anchor.cy_px)])
        evidence: list[float] = []
        distances: list[float] = []
        agreements: list[float] = []
        output: dict[str, object] = {
            "sample_id": str(anchor.sample_id),
            "candidate_id": str(anchor.candidate_id),
        }
        for backend in BACKENDS:
            group = grouped[backend].get(str(anchor.sample_id))
            prefix = f"consensus_{backend}"
            if group is None or group.empty:
                output.update(
                    {
                        f"{prefix}_calibrated_probability": 0.0,
                        f"{prefix}_nearest_peak_distance_px": image_diagonal_px,
                        f"{prefix}_nearest_peak_distance_normalized": 1.0,
                        f"{prefix}_nearest_peak_rank": 0.0,
                        f"{prefix}_absolute_angle_difference_deg": 90.0,
                        f"{prefix}_cos_2_angle_difference": -1.0,
                        f"{prefix}_sin_2_angle_difference": 0.0,
                        f"{prefix}_log_width_ratio": 0.0,
                        f"{prefix}_nearby_peak": 0.0,
                        f"{prefix}_mutual_nearest": 0.0,
                        f"{prefix}_missing": 1.0,
                    }
                )
                continue
            coordinates = group[["cx_px", "cy_px"]].to_numpy(float)
            distance = np.linalg.norm(coordinates - anchor_xy[None, :], axis=1)
            index = int(np.argmin(distance))
            nearest = group.iloc[index]
            distance_px = float(distance[index])
            angle_difference = periodic_angle_error(float(anchor.theta_deg), float(nearest["theta_deg"]))
            directed_angle_difference = float(anchor.theta_deg) - float(nearest["theta_deg"])
            probability = float(nearest["calibrated_native_probability"])
            width_ratio = math.log(
                max(float(anchor.width_px), 1e-6) / max(float(nearest["width_px"]), 1e-6)
            )
            # Mutual nearest is evaluated geometrically against all anchors of
            # the current route and therefore remains independent of row order.
            anchor_group = grouped[anchor_route][str(anchor.sample_id)]
            anchor_coordinates = anchor_group[["cx_px", "cy_px"]].to_numpy(float)
            reverse_distance = np.linalg.norm(
                anchor_coordinates
                - nearest[["cx_px", "cy_px"]].to_numpy(float)[None, :],
                axis=1,
            )
            reverse_index = int(np.argmin(reverse_distance))
            mutual = str(anchor_group.iloc[reverse_index]["candidate_id"]) == str(anchor.candidate_id)
            output.update(
                {
                    f"{prefix}_calibrated_probability": probability,
                    f"{prefix}_nearest_peak_distance_px": distance_px,
                    f"{prefix}_nearest_peak_distance_normalized": min(distance_px / image_diagonal_px, 1.0),
                    f"{prefix}_nearest_peak_rank": float(nearest["native_rank"]),
                    f"{prefix}_absolute_angle_difference_deg": angle_difference,
                    f"{prefix}_cos_2_angle_difference": math.cos(math.radians(2.0 * angle_difference)),
                    # In physical image axes the backend-minus-anchor change is
                    # (-backend theta) - (-anchor theta) = anchor - backend.
                    f"{prefix}_sin_2_angle_difference": math.sin(
                        math.radians(2.0 * directed_angle_difference)
                    ),
                    f"{prefix}_log_width_ratio": width_ratio,
                    f"{prefix}_nearby_peak": float(distance_px <= nearby_threshold_px),
                    f"{prefix}_mutual_nearest": float(mutual),
                    f"{prefix}_missing": 0.0,
                }
            )
            evidence.append(probability)
            distances.append(min(distance_px / nearby_threshold_px, 1.0))
            agreements.append(max(0.0, math.cos(math.radians(2.0 * angle_difference))))
        output["number_of_backends_with_nearby_peak"] = sum(
            float(output[f"consensus_{backend}_nearby_peak"]) for backend in BACKENDS
        )
        if evidence:
            evidence_array = np.asarray(evidence)
            output["backend_support_mean"] = float(evidence_array.mean())
            output["backend_support_min"] = float(evidence_array.min())
            output["backend_support_variance"] = float(evidence_array.var())
            output["consensus_score"] = float(
                np.mean(evidence_array * (1.0 - np.asarray(distances)) * np.asarray(agreements))
            )
            output["disagreement_score"] = float(
                evidence_array.std() + np.mean(distances) + (1.0 - np.mean(agreements))
            )
        else:
            output.update(
                {
                    "backend_support_mean": 0.0,
                    "backend_support_min": 0.0,
                    "backend_support_variance": 0.0,
                    "consensus_score": 0.0,
                    "disagreement_score": 3.0,
                }
            )
        rows.append(output)
    return pd.DataFrame(rows)
