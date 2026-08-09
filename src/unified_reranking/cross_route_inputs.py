"""Leakage-safe primitives for gated cross-route router inputs.

The functions in this module are deliberately split-agnostic and perform no
filesystem access.  Development supervision is accepted only by the explicit
outer-fold gate helper; feature construction consumes label-free selected-pose
evidence.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .gate import (
    ConservativeTransitionModel,
    GateEvidence,
    GateOperatingPoint,
    OOFTransitionData,
    gate_switch_mask,
)


ROUTES = ("crog", "g1", "c1")
ALTERNATIVES = ("g1", "c1")
ROUTE_TIE_BREAK = ("G1", "C1")

_SUPERVISION_TOKENS = (
    "correct",
    "success",
    "ground_truth",
    "matched_gt",
    "jacquard",
    "recovered",
    "harmful",
    "first_positive",
    "object_category",
    "sample_id",
    "scene_id",
    "candidate_id",
)


def router_feature_columns(route: str) -> tuple[str, ...]:
    """Return the fixed route-router feature schema for one alternative."""

    route = str(route).lower()
    if route not in ALTERNATIVES:
        raise ValueError("router features are defined only for g1 and c1")
    prefix = f"{route}_"
    names = (
        "selected_calibrated_probability",
        "calibrated_probability_delta_from_crog",
        "gate_probability_recover",
        "gate_probability_harm",
        "gate_utility",
        "route_specific_margin",
        "selected_reliability",
        "reliability_delta_from_crog",
        "selected_stability",
        "selected_mask_reliability",
        "candidate_exists_feature",
        "no_output",
        "agreement_with_crog_center",
        "agreement_with_crog_angle",
        "agreement_with_crog_width",
        "agreement_with_crog_geometry",
        "agreement_with_peer_geometry",
        "tri_backend_nearby_count",
        "tri_backend_support_mean",
        "tri_backend_support_min",
        "tri_backend_support_variance",
        "tri_backend_consensus",
    )
    result = tuple(prefix + name for name in names)
    assert_router_feature_names(result)
    return result


def assert_router_feature_names(names: Sequence[str]) -> tuple[str, ...]:
    """Reject identity and supervision-bearing route-model features."""

    values = tuple(map(str, names))
    if not values or len(values) != len(set(values)):
        raise ValueError("router feature names must be non-empty and unique")
    forbidden = [
        name for name in values if any(token in name.lower() for token in _SUPERVISION_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"router features contain identity/supervision: {forbidden}")
    return values


def gate_evidence_from_frame(frame: pd.DataFrame) -> GateEvidence:
    """Build the immutable conservative-gate evidence bundle."""

    return GateEvidence(
        score_margin=frame["score_margin"].to_numpy(),
        challenger_reliability=frame["challenger_reliability"].to_numpy(),
        perturbation_stability=frame["perturbation_stability"].to_numpy(),
        seed_challenger_votes=frame["seed_challenger_votes"].to_numpy(),
        candidate_id_unchanged=frame["candidate_id_unchanged"].to_numpy(),
        geometry_hash_unchanged=frame["geometry_hash_unchanged"].to_numpy(),
        challenger_exists=frame["challenger_exists"].to_numpy(),
    )


def apply_gate_probabilities(
    frame: pd.DataFrame,
    probability_recover: Any,
    probability_harm: Any,
    operating_point: GateOperatingPoint | None,
) -> pd.DataFrame:
    """Apply a locked gate point without changing candidate identities."""

    recover = np.asarray(probability_recover, dtype=np.float64)
    harm = np.asarray(probability_harm, dtype=np.float64)
    if recover.shape != (len(frame),) or harm.shape != (len(frame),):
        raise ValueError("gate probability vectors must match input rows")
    if not np.isfinite(recover).all() or not np.isfinite(harm).all():
        raise ValueError("gate probabilities must be finite")
    if operating_point is None:
        switches = np.zeros(len(frame), dtype=bool)
        # NO-GO is represented by the separate switch bit.  A finite neutral
        # utility keeps downstream feature tables well-defined.
        utility = np.zeros(len(frame), dtype=np.float64)
    else:
        switches = gate_switch_mask(
            recover, harm, gate_evidence_from_frame(frame), operating_point
        )
        utility = recover - float(operating_point.lambda_harm) * harm
    native_ids = frame["native_candidate_id"].fillna("").astype(str).to_numpy()
    challenger_ids = frame["challenger_candidate_id"].fillna("").astype(str).to_numpy()
    output = frame.copy()
    output["gate_probability_recover"] = recover
    output["gate_probability_harm"] = harm
    output["gate_utility"] = utility
    output["gate_switch"] = switches
    output["gated_candidate_id"] = np.where(switches, challenger_ids, native_ids)
    if {"native_correct", "challenger_correct"}.issubset(output.columns):
        native = output["native_correct"].fillna(False).astype(bool).to_numpy()
        challenger = output["challenger_correct"].fillna(False).astype(bool).to_numpy()
        output["gated_correct"] = np.where(switches, challenger, native)
    return output


def cross_fitted_gate_probabilities(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
    """Outer-fold gate predictions for leakage-safe route-router training.

    Ranker predictions in ``frame`` are already OOF.  This second outer loop is
    necessary because predicting on the same rows used to fit the gate would
    leak gate outcomes into the downstream route router.
    """

    names = tuple(map(str, feature_names))
    required = {
        "sample_id",
        "scene_id",
        "oof_fold",
        "prediction_source",
        "native_correct",
        "challenger_correct",
        *names,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"gate OOF input misses columns: {missing}")
    provenance = set(frame["prediction_source"].dropna().astype(str))
    if provenance != {"train_oof"} or frame["prediction_source"].isna().any():
        raise ValueError("outer gate predictions require train_oof provenance")
    folds = frame["oof_fold"].astype(str).to_numpy()
    if len(np.unique(folds)) < 3:
        raise ValueError("outer gate cross-fitting requires at least three folds")
    recover = np.full(len(frame), np.nan, dtype=np.float64)
    harm = np.full(len(frame), np.nan, dtype=np.float64)
    audits: list[dict[str, object]] = []
    for outer_index, fold in enumerate(dict.fromkeys(folds.tolist())):
        held_out = folds == fold
        training = ~held_out
        model = ConservativeTransitionModel(seed=int(seed) + 10_000 + outer_index).fit(
            OOFTransitionData(
                features=frame.loc[training, names].to_numpy(float),
                feature_names=names,
                native_correct=frame.loc[training, "native_correct"].to_numpy(),
                challenger_correct=frame.loc[training, "challenger_correct"].to_numpy(),
                scene_ids=frame.loc[training, "scene_id"].to_numpy(),
                oof_fold_ids=frame.loc[training, "oof_fold"].to_numpy(),
                prediction_source="train_oof",
            )
        )
        fold_recover, fold_harm = model.predict_probabilities(
            frame.loc[held_out, names].to_numpy(float)
        )
        recover[held_out] = fold_recover
        harm[held_out] = fold_harm
        audits.append(
            {
                "held_out_fold": str(fold),
                "training_rows": int(training.sum()),
                "held_out_rows": int(held_out.sum()),
                "training_fold_count": int(len(np.unique(folds[training]))),
                "model": model.artifact(),
            }
        )
    if not np.isfinite(recover).all() or not np.isfinite(harm).all():
        raise RuntimeError("outer gate cross-fitting did not cover every row")
    return recover, harm, tuple(audits)


def gate_operating_point_from_manifest(manifest: Mapping[str, Any]) -> GateOperatingPoint | None:
    """Decode the Validation-locked point, preserving explicit NO-GO."""

    decision = str(manifest.get("decision", ""))
    selection = dict(manifest.get("selection", {}))
    value = selection.get("selected_operating_point")
    if decision == "NO_GO_NATIVE":
        if value is not None:
            raise ValueError("NO_GO_NATIVE gate unexpectedly contains an operating point")
        return None
    if decision != "GO" or not isinstance(value, Mapping):
        raise ValueError("gate manifest has no valid Validation-locked decision")
    return GateOperatingPoint(**dict(value))


def geometry_agreement(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    """Deterministic, scale-aware agreement for two selected rectangles."""

    required = ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")
    if any(key not in left or key not in right for key in required):
        raise ValueError("geometry agreement requires complete rectangle fields")
    lhs = np.asarray([float(left[key]) for key in required], dtype=np.float64)
    rhs = np.asarray([float(right[key]) for key in required], dtype=np.float64)
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        return {"center": 0.0, "angle": 0.0, "width": 0.0, "geometry": 0.0}
    scale = max(0.5 * (lhs[3] + rhs[3]), 1.0)
    center = float(np.exp(-np.linalg.norm(lhs[:2] - rhs[:2]) / scale))
    delta = abs((lhs[2] - rhs[2]) % 180.0)
    angle_error = min(delta, 180.0 - delta)
    angle = float(max(0.0, np.cos(np.deg2rad(2.0 * angle_error))))
    width = float(np.exp(-abs(np.log(max(lhs[3], 1e-6) / max(rhs[3], 1e-6)))))
    geometry = float((center * angle * width) ** (1.0 / 3.0))
    return {"center": center, "angle": angle, "width": width, "geometry": geometry}


def add_router_features(gated_by_route: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Pair three gated route tables and add only observable router features."""

    if set(gated_by_route) != set(ROUTES):
        raise ValueError("gated_by_route must contain crog/g1/c1")
    prepared: dict[str, pd.DataFrame] = {}
    keys = ["sample_id", "scene_id", "prediction_source"]
    for route in ROUTES:
        frame = gated_by_route[route].copy()
        if frame.empty or frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"{route} gated decisions must have unique rows")
        if "oof_fold" in frame:
            keys_with_fold = [*keys, "oof_fold"]
        else:
            keys_with_fold = keys
        rename = {
            column: f"{route}_{column}"
            for column in frame.columns
            if column not in keys_with_fold
        }
        prepared[route] = frame.rename(columns=rename)
    output = prepared["crog"]
    for route in ALTERNATIVES:
        join_keys = [*keys]
        if "oof_fold" in output.columns and "oof_fold" in prepared[route].columns:
            join_keys.append("oof_fold")
        output = output.merge(prepared[route], on=join_keys, validate="one_to_one")
    if not (len(output) == len(prepared["crog"]) == len(prepared["g1"]) == len(prepared["c1"])):
        raise RuntimeError("route tables do not preserve the paired denominator")

    geometry_fields = ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")
    feature_rows: list[dict[str, float]] = []
    for row in output.to_dict("records"):
        exists = {
            route: bool(row.get(f"{route}_selected_candidate_exists", False))
            for route in ROUTES
        }
        geometry = {
            route: {field: row.get(f"{route}_selected_{field}", np.nan) for field in geometry_fields}
            for route in ROUTES
        }
        pair: dict[tuple[str, str], dict[str, float]] = {}
        for left in ROUTES:
            for right in ROUTES:
                if left == right:
                    pair[(left, right)] = {
                        "center": float(exists[left]),
                        "angle": float(exists[left]),
                        "width": float(exists[left]),
                        "geometry": float(exists[left]),
                    }
                elif exists[left] and exists[right]:
                    pair[(left, right)] = geometry_agreement(geometry[left], geometry[right])
                else:
                    pair[(left, right)] = {
                        "center": 0.0,
                        "angle": 0.0,
                        "width": 0.0,
                        "geometry": 0.0,
                    }
        record: dict[str, float] = {}
        crog_probability = float(row.get("crog_selected_calibrated_probability", 0.0))
        crog_reliability = float(row.get("crog_selected_reliability", 0.0))
        for route in ALTERNATIVES:
            peer = "c1" if route == "g1" else "g1"
            prefix = f"{route}_"
            probability = float(row.get(f"{route}_selected_calibrated_probability", 0.0))
            reliability = float(row.get(f"{route}_selected_reliability", 0.0))
            support = np.asarray(
                [
                    float(row.get(f"{backend}_selected_calibrated_probability", 0.0))
                    * pair[(route, backend)]["geometry"]
                    for backend in ROUTES
                    if exists[backend]
                ],
                dtype=np.float64,
            )
            if support.size == 0:
                support = np.zeros(1, dtype=np.float64)
            record.update(
                {
                    prefix + "selected_calibrated_probability": probability,
                    prefix + "calibrated_probability_delta_from_crog": probability - crog_probability,
                    prefix + "gate_probability_recover": float(row.get(f"{route}_gate_probability_recover", 0.0)),
                    prefix + "gate_probability_harm": float(row.get(f"{route}_gate_probability_harm", 0.0)),
                    prefix + "gate_utility": float(row.get(f"{route}_gate_utility", 0.0)),
                    prefix + "route_specific_margin": float(row.get(f"{route}_route_specific_margin", 0.0)),
                    prefix + "selected_reliability": reliability,
                    prefix + "reliability_delta_from_crog": reliability - crog_reliability,
                    prefix + "selected_stability": float(row.get(f"{route}_selected_stability", 0.0)),
                    prefix + "selected_mask_reliability": float(row.get(f"{route}_selected_mask_reliability", 0.0)),
                    prefix + "candidate_exists_feature": float(exists[route]),
                    prefix + "no_output": float(not exists[route]),
                    prefix + "agreement_with_crog_center": pair[(route, "crog")]["center"],
                    prefix + "agreement_with_crog_angle": pair[(route, "crog")]["angle"],
                    prefix + "agreement_with_crog_width": pair[(route, "crog")]["width"],
                    prefix + "agreement_with_crog_geometry": pair[(route, "crog")]["geometry"],
                    prefix + "agreement_with_peer_geometry": pair[(route, peer)]["geometry"],
                    prefix + "tri_backend_nearby_count": float(
                        sum(pair[(route, backend)]["center"] >= np.exp(-20.0 / max(float(row.get(f"{route}_selected_width_px", 1.0)), 1.0)) for backend in ROUTES)
                    ),
                    prefix + "tri_backend_support_mean": float(support.mean()),
                    prefix + "tri_backend_support_min": float(support.min()),
                    prefix + "tri_backend_support_variance": float(support.var()),
                    prefix + "tri_backend_consensus": float(support.mean()),
                }
            )
        feature_rows.append(record)
    features = pd.DataFrame(feature_rows, index=output.index)
    for route in ALTERNATIVES:
        assert_router_feature_names(router_feature_columns(route))
    if not np.isfinite(features.to_numpy(float)).all():
        raise RuntimeError("route-router features are not finite")
    repeated = sorted(set(features.columns).intersection(output.columns))
    for column in repeated:
        declared = pd.to_numeric(output[column], errors="coerce").to_numpy(float)
        recomputed = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
        if not np.allclose(
            declared,
            recomputed,
            rtol=0.0,
            atol=1e-12,
            equal_nan=False,
        ):
            raise RuntimeError(
                f"route-router observable feature differs from paired input: {column}"
            )
    features = features.drop(columns=repeated)
    return pd.concat([output.reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def operating_point_audit(point: GateOperatingPoint | None) -> dict[str, object] | None:
    return None if point is None else asdict(point)


__all__ = [
    "ALTERNATIVES",
    "ROUTES",
    "ROUTE_TIE_BREAK",
    "add_router_features",
    "apply_gate_probabilities",
    "assert_router_feature_names",
    "cross_fitted_gate_probabilities",
    "gate_evidence_from_frame",
    "gate_operating_point_from_manifest",
    "geometry_agreement",
    "operating_point_audit",
    "router_feature_columns",
]
