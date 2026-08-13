"""Label-free four-route router and route-qualified Top-20 primitives.

The completed CROG/G1/C1 experiment remains an input to this module.  This
module has no filesystem writes and does not import a Test evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from unified_reranking.gate import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    ConservativeTransitionModel,
    OOFTransitionData,
    _binary,
    _identifiers,
    _probability,
)
from unified_reranking.hashing import canonical_sha256
from unified_reranking.route_router import RouterEvidence, RouterOperatingPoint
from unified_reranking.statistics import cluster_bootstrap_difference

from .contracts import forbidden_test_schema_columns


FOUR_ROUTES = ("CROG", "G1", "C1", "D1")
ALTERNATIVE_ROUTES = ("G1", "C1", "D1")
DEFAULT_ROUTE_TIE_BREAK = ALTERNATIVE_ROUTES
MAX_ROUTE_CANDIDATES = 5
MAX_UNION_CANDIDATES = len(FOUR_ROUTES) * MAX_ROUTE_CANDIDATES


class FourRouteCROGDefaultTransitionRouter:
    """One OOF-trained recover/harm model per non-CROG route."""

    def __init__(self, *, seed: int = 20260808) -> None:
        self.seed = int(seed)
        self.models_: dict[str, ConservativeTransitionModel] = {}

    def fit(
        self, data_by_route: Mapping[str, OOFTransitionData]
    ) -> "FourRouteCROGDefaultTransitionRouter":
        if set(data_by_route) != set(ALTERNATIVE_ROUTES):
            raise ValueError("router OOF data must contain exactly G1, C1, and D1")
        validated = {
            route: data_by_route[route].validated() for route in ALTERNATIVE_ROUTES
        }
        reference = validated[ALTERNATIVE_ROUTES[0]]
        for route in ALTERNATIVE_ROUTES[1:]:
            current = validated[route]
            if not np.array_equal(reference[1], current[1]):
                raise ValueError(f"{route} and G1 have different CROG outcomes")
            if not np.array_equal(reference[3], current[3]) or not np.array_equal(
                reference[4], current[4]
            ):
                raise ValueError(f"{route} and G1 are not scene/fold aligned")
        for index, route in enumerate(ALTERNATIVE_ROUTES):
            data = data_by_route[route]
            names = tuple(name.lower() for name in data.feature_names)
            if not any(
                "candidate_exists" in name or "no_output" in name for name in names
            ):
                raise ValueError(
                    f"{route} router features require candidate-existence/no-output"
                )
            self.models_[route] = ConservativeTransitionModel(
                seed=self.seed + 1_000 * index
            ).fit(data)
        return self

    def predict_probabilities(
        self, features_by_route: Mapping[str, Any]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if set(self.models_) != set(ALTERNATIVE_ROUTES):
            raise RuntimeError("four-route transition models are not fitted")
        if set(features_by_route) != set(ALTERNATIVE_ROUTES):
            raise ValueError("features must contain exactly G1, C1, and D1")
        return {
            route: self.models_[route].predict_probabilities(features_by_route[route])
            for route in ALTERNATIVE_ROUTES
        }

    def artifact(self) -> dict[str, Any]:
        if set(self.models_) != set(ALTERNATIVE_ROUTES):
            raise RuntimeError("four-route transition models are not fitted")
        return {
            "kind": "four_route_crog_default_expected_gain_router",
            "default_route": "CROG",
            "alternative_routes": list(ALTERNATIVE_ROUTES),
            "tie_break": list(DEFAULT_ROUTE_TIE_BREAK),
            "prediction_source": "paired_train_oof",
            "seed": self.seed,
            "transition_models": {
                route: self.models_[route].artifact() for route in ALTERNATIVE_ROUTES
            },
        }


def _validate_route_inputs(
    probabilities: Mapping[str, tuple[Any, Any]],
    evidence_by_route: Mapping[str, RouterEvidence],
) -> tuple[
    int,
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    expected = set(ALTERNATIVE_ROUTES)
    if set(probabilities) != expected or set(evidence_by_route) != expected:
        raise ValueError("router inputs must contain exactly G1, C1, and D1")
    first = np.asarray(probabilities[ALTERNATIVE_ROUTES[0]][0])
    if first.ndim != 1 or len(first) == 0:
        raise ValueError("route probabilities must be non-empty vectors")
    length = len(first)
    clean_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    clean_evidence: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for route in ALTERNATIVE_ROUTES:
        recover, harm = probabilities[route]
        clean_probabilities[route] = (
            _probability(recover, f"{route}_probability_recover", length=length),
            _probability(harm, f"{route}_probability_harm", length=length),
        )
        clean_evidence[route] = evidence_by_route[route].validated(length)
    return length, clean_probabilities, clean_evidence


def four_route_decisions(
    probabilities: Mapping[str, tuple[Any, Any]],
    evidence_by_route: Mapping[str, RouterEvidence],
    operating_point: RouterOperatingPoint,
    *,
    tie_break: Sequence[str] = DEFAULT_ROUTE_TIE_BREAK,
) -> np.ndarray:
    """Apply the CROG-default router using only label-free evidence."""

    normalized_tie_break = tuple(map(str, tie_break))
    if len(normalized_tie_break) != len(ALTERNATIVE_ROUTES) or set(
        normalized_tie_break
    ) != set(ALTERNATIVE_ROUTES):
        raise ValueError("tie_break must contain G1, C1, and D1 exactly once")
    length, clean_probability, clean_evidence = _validate_route_inputs(
        probabilities, evidence_by_route
    )
    utility = {
        route: clean_probability[route][0]
        - operating_point.lambda_router * clean_probability[route][1]
        for route in ALTERNATIVE_ROUTES
    }
    eligible: dict[str, np.ndarray] = {}
    for route in ALTERNATIVE_ROUTES:
        margin, reliability, stability, exists = clean_evidence[route]
        eligible[route] = (
            (utility[route] > operating_point.utility_threshold)
            & (margin > operating_point.margin_threshold)
            & (reliability > operating_point.reliability_threshold)
            & (stability >= operating_point.stability_threshold)
            & exists
        )
    decisions = np.full(length, "CROG", dtype=object)
    best_utility = np.full(length, -np.inf, dtype=np.float64)
    for route in normalized_tie_break:
        choose = eligible[route] & (utility[route] > best_utility)
        decisions[choose] = route
        best_utility[choose] = utility[route][choose]
    return decisions


@dataclass(frozen=True)
class FourRouteRouterTrial:
    operating_point: RouterOperatingPoint
    bootstrap_lower_bound: float
    mean_delta: float
    recovered: int
    harmful: int
    switch_count: int
    switch_rate: float
    route_switches: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FourRouteRouterSelection:
    status: Literal["GO", "NO_GO_CROG"]
    selected_operating_point: RouterOperatingPoint | None
    trials: tuple[FourRouteRouterTrial, ...]
    tie_break: tuple[str, ...] = DEFAULT_ROUTE_TIE_BREAK
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS
    bootstrap_seed: int = BOOTSTRAP_SEED

    def artifact(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "default_route": "CROG",
            "alternative_routes": list(ALTERNATIVE_ROUTES),
            "tie_break": list(self.tie_break),
            "selected_operating_point": (
                None
                if self.selected_operating_point is None
                else asdict(self.selected_operating_point)
            ),
            "trials": [
                {
                    **asdict(trial),
                    "operating_point": asdict(trial.operating_point),
                    "route_switches": dict(trial.route_switches),
                }
                for trial in self.trials
            ],
            "selection_order": [
                "scene_bootstrap_95pct_lower_bound",
                "mean_delta",
                "fewer_harmful",
                "lower_switch_rate",
            ],
            "no_go_rule": "CROG when every lower bound <= 0",
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "candidate_test_labels_read": False,
        }


def select_four_route_operating_point(
    probabilities: Mapping[str, tuple[Any, Any]],
    evidence_by_route: Mapping[str, RouterEvidence],
    route_correct: Mapping[str, Any],
    scene_ids: Any,
    operating_points: Sequence[RouterOperatingPoint],
    *,
    tie_break: Sequence[str] = DEFAULT_ROUTE_TIE_BREAK,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> FourRouteRouterSelection:
    """Validation-select an operating point or retain CROG as explicit NO-GO."""

    points = tuple(operating_points)
    if not points:
        raise ValueError("at least one Validation operating point is required")
    length, _, _ = _validate_route_inputs(probabilities, evidence_by_route)
    if set(route_correct) != set(FOUR_ROUTES):
        raise ValueError("route_correct must contain CROG, G1, C1, and D1")
    correct = {
        route: _binary(route_correct[route], f"{route}_correct", length=length)
        for route in FOUR_ROUTES
    }
    scenes = _identifiers(scene_ids, "scene_ids", length=length)
    normalized_tie_break = tuple(map(str, tie_break))
    trials: list[FourRouteRouterTrial] = []
    indexes = np.arange(length)
    for point in points:
        decisions = four_route_decisions(
            probabilities,
            evidence_by_route,
            point,
            tie_break=normalized_tie_break,
        )
        selected = np.asarray(
            [correct[str(route)][index] for index, route in zip(indexes, decisions)],
            dtype=bool,
        )
        crog = correct["CROG"]
        bootstrap = cluster_bootstrap_difference(
            crog,
            selected,
            scenes,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed),
        )
        switched = decisions != "CROG"
        trials.append(
            FourRouteRouterTrial(
                operating_point=point,
                bootstrap_lower_bound=float(bootstrap["ci95"][0]),
                mean_delta=float(selected.mean() - crog.mean()),
                recovered=int((~crog & selected).sum()),
                harmful=int((crog & ~selected).sum()),
                switch_count=int(switched.sum()),
                switch_rate=float(switched.mean()),
                route_switches=tuple(
                    (route, int((decisions == route).sum()))
                    for route in ALTERNATIVE_ROUTES
                ),
            )
        )
    best = max(
        trials,
        key=lambda trial: (
            trial.bootstrap_lower_bound,
            trial.mean_delta,
            -trial.harmful,
            -trial.switch_rate,
        ),
    )
    selected = best.operating_point if best.bootstrap_lower_bound > 0.0 else None
    return FourRouteRouterSelection(
        status="GO" if selected is not None else "NO_GO_CROG",
        selected_operating_point=selected,
        trials=tuple(trials),
        tie_break=normalized_tie_break,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )


def _denominator_ids(denominator: Sequence[Any] | pd.DataFrame) -> tuple[str, ...]:
    values = (
        denominator["sample_id"].tolist()
        if isinstance(denominator, pd.DataFrame)
        else list(denominator)
    )
    result = tuple(str(value) for value in values)
    if (
        not result
        or any(not value for value in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("union denominator must contain unique non-empty sample IDs")
    return result


def build_top20_union(
    route_top5: Mapping[str, pd.DataFrame],
    denominator: Sequence[Any] | pd.DataFrame,
    *,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Concatenate four exact Top-5 memberships without cross-route deduplication."""

    normalized = {str(route).upper(): frame for route, frame in route_top5.items()}
    if set(normalized) != set(FOUR_ROUTES):
        raise ValueError("route_top5 must contain CROG, G1, C1, and D1")
    split_name = str(split).lower()
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    denominator_ids = _denominator_ids(denominator)
    denominator_set = set(denominator_ids)
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
    }
    frames: list[pd.DataFrame] = []
    route_audit: dict[str, Any] = {}
    for route in FOUR_ROUTES:
        frame = normalized[route].copy()
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{route} Top5 misses columns: {missing}")
        if split_name == "test":
            forbidden = forbidden_test_schema_columns(tuple(map(str, frame.columns)))
            if forbidden:
                raise PermissionError(
                    f"{route} Test Top5 contains supervision columns: {forbidden}"
                )
        keys = ["sample_id", "candidate_id"]
        if frame[keys].isna().any().any():
            raise ValueError(f"{route} Top5 contains null candidate keys")
        frame[keys] = frame[keys].astype(str)
        if frame.duplicated(keys).any():
            raise ValueError(f"{route} Top5 contains duplicate raw candidate IDs")
        if set(frame["sample_id"]).difference(denominator_set):
            raise ValueError(f"{route} Top5 contains samples outside the denominator")
        if "route" in frame.columns and set(
            frame["route"].astype(str).str.upper()
        ) not in (
            set(),
            {route},
        ):
            raise ValueError(f"{route} Top5 route column differs")
        numeric_rank = pd.to_numeric(frame["native_rank"], errors="coerce")
        if (
            numeric_rank.isna().any()
            or not np.equal(numeric_rank, np.floor(numeric_rank)).all()
        ):
            raise ValueError(f"{route} Top5 native ranks must be finite integers")
        frame["native_rank"] = numeric_rank.astype(int)
        if ((frame["native_rank"] < 1) | (frame["native_rank"] > 5)).any():
            raise ValueError(f"{route} Top5 native ranks must lie in 1..5")
        for sample_id, group in frame.groupby("sample_id", sort=False):
            ranks = sorted(group["native_rank"].tolist())
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError(f"{route}/{sample_id} Top5 ranks are not contiguous")
        for column in ("candidate_identity_sha256", "candidate_geometry_sha256"):
            if frame[column].isna().any() or frame[column].astype(str).eq("").any():
                raise ValueError(f"{route} Top5 has missing {column}")
        source_ids = frame["candidate_id"].astype(str)
        frame["source_route"] = route
        frame["source_candidate_id"] = source_ids
        frame["route_native_rank"] = frame["native_rank"].astype(int)
        frame["candidate_id"] = route + ":" + source_ids
        frame["route_member_sha256"] = [
            canonical_sha256(
                {
                    "sample_id": sample_id,
                    "source_route": route,
                    "source_candidate_id": candidate_id,
                    "candidate_identity_sha256": identity,
                    "candidate_geometry_sha256": geometry,
                }
            )
            for sample_id, candidate_id, identity, geometry in zip(
                frame["sample_id"],
                source_ids,
                frame["candidate_identity_sha256"].astype(str),
                frame["candidate_geometry_sha256"].astype(str),
            )
        ]
        for encoded_route in FOUR_ROUTES:
            frame[f"union_route_{encoded_route.lower()}"] = float(
                encoded_route == route
            )
        frames.append(frame)
        route_audit[route] = {
            "candidate_rows": int(len(frame)),
            "candidate_bearing_samples": int(frame["sample_id"].nunique()),
            "membership_sha256": canonical_sha256(
                sorted(
                    frame[["sample_id", "candidate_id", "route_member_sha256"]].to_dict(
                        "records"
                    ),
                    key=lambda row: (row["sample_id"], row["candidate_id"]),
                )
            ),
        }
    output = pd.concat(frames, ignore_index=True, sort=False)
    order = {sample_id: index for index, sample_id in enumerate(denominator_ids)}
    route_order = {route: index for index, route in enumerate(FOUR_ROUTES)}
    output["_sample_order"] = output["sample_id"].map(order)
    output["_route_order"] = output["source_route"].map(route_order)
    output = output.sort_values(
        ["_sample_order", "route_native_rank", "_route_order"], kind="mergesort"
    ).reset_index(drop=True)
    output["native_rank"] = output.groupby("sample_id", sort=False).cumcount() + 1
    output = output.drop(columns=["_sample_order", "_route_order"])
    keys = ["sample_id", "candidate_id"]
    if output.duplicated(keys).any():
        raise RuntimeError("route-qualified union identities are not unique")
    counts = (
        output.groupby("sample_id", sort=False)
        .size()
        .reindex(denominator_ids, fill_value=0)
    )
    if int(counts.max()) > MAX_UNION_CANDIDATES:
        raise RuntimeError("four-route union exceeds the Top20 contract")
    expected_rows = sum(len(normalized[route]) for route in FOUR_ROUTES)
    if len(output) != expected_rows:
        raise RuntimeError("four-route union does not exactly preserve Top5 membership")
    raw_duplicate_rows = int(
        output.duplicated(["sample_id", "source_candidate_id"], keep=False).sum()
    )
    geometry_duplicate_rows = int(
        output.duplicated(["sample_id", "candidate_geometry_sha256"], keep=False).sum()
    )
    audit = {
        "split": split_name,
        "routes": list(FOUR_ROUTES),
        "route_tie_order": list(FOUR_ROUTES),
        "source_pool": "exact_route_top5_membership",
        "primary_union_deduplication": "NONE",
        "candidate_identity": "route_qualified_raw_candidate_id",
        "maximum_candidates_per_sample": int(counts.max()),
        "contract_maximum_candidates": MAX_UNION_CANDIDATES,
        "full_top20_samples": int((counts == MAX_UNION_CANDIDATES).sum()),
        "candidate_rows": int(len(output)),
        "expected_membership_rows": int(expected_rows),
        "raw_id_duplicate_rows_across_routes_retained": raw_duplicate_rows,
        "geometry_duplicate_rows_across_routes_retained": geometry_duplicate_rows,
        "route_sources": route_audit,
        "membership_sha256": canonical_sha256(
            output[
                [
                    "sample_id",
                    "candidate_id",
                    "source_route",
                    "source_candidate_id",
                    "route_native_rank",
                    "route_member_sha256",
                ]
            ].to_dict("records")
        ),
        "candidate_test_labels_read": False if split_name == "test" else None,
    }
    return output, audit


def validate_top20_union_frame(
    union: pd.DataFrame, denominator: Sequence[Any] | pd.DataFrame
) -> dict[str, Any]:
    """Replay the route-qualified membership contract on a materialized union."""

    denominator_ids = _denominator_ids(denominator)
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "source_route",
        "source_candidate_id",
        "route_native_rank",
        "candidate_identity_sha256",
        "candidate_geometry_sha256",
        "route_member_sha256",
        *{f"union_route_{route.lower()}" for route in FOUR_ROUTES},
    }
    missing = sorted(required.difference(union.columns))
    if missing:
        raise ValueError(f"Top20 union contract misses columns: {missing}")
    work = union.copy()
    for column in ("sample_id", "candidate_id", "source_route", "source_candidate_id"):
        if work[column].isna().any():
            raise ValueError(f"Top20 union contains null {column}")
        work[column] = work[column].astype(str)
    work["source_route"] = work["source_route"].str.upper()
    if not set(work["source_route"]).issubset(FOUR_ROUTES):
        raise ValueError("Top20 union contains an unknown source route")
    if set(work["sample_id"]).difference(denominator_ids):
        raise ValueError("Top20 union contains samples outside the denominator")
    if work.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError("Top20 union contains duplicate route-qualified identities")
    expected_ids = work["source_route"] + ":" + work["source_candidate_id"]
    if not work["candidate_id"].equals(expected_ids):
        raise ValueError("Top20 union candidate IDs are not route-qualified raw IDs")
    for rank_column, maximum in (
        ("route_native_rank", MAX_ROUTE_CANDIDATES),
        ("native_rank", MAX_UNION_CANDIDATES),
    ):
        numeric = pd.to_numeric(work[rank_column], errors="coerce")
        if (
            numeric.isna().any()
            or not np.equal(numeric, np.floor(numeric)).all()
            or ((numeric < 1) | (numeric > maximum)).any()
        ):
            raise ValueError(f"Top20 union {rank_column} contract differs")
        work[rank_column] = numeric.astype(int)
    for (sample_id, route), group in work.groupby(
        ["sample_id", "source_route"], sort=False
    ):
        ranks = sorted(group["route_native_rank"].tolist())
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"Top20 union {sample_id}/{route} route ranks differ")
    for sample_id, group in work.groupby("sample_id", sort=False):
        ranks = sorted(group["native_rank"].tolist())
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"Top20 union {sample_id} union ranks differ")
    expected_member_hash = [
        canonical_sha256(
            {
                "sample_id": sample_id,
                "source_route": route,
                "source_candidate_id": candidate_id,
                "candidate_identity_sha256": identity,
                "candidate_geometry_sha256": geometry,
            }
        )
        for sample_id, route, candidate_id, identity, geometry in zip(
            work["sample_id"],
            work["source_route"],
            work["source_candidate_id"],
            work["candidate_identity_sha256"].astype(str),
            work["candidate_geometry_sha256"].astype(str),
        )
    ]
    if work["route_member_sha256"].astype(str).tolist() != expected_member_hash:
        raise ValueError("Top20 union route-member identity hash differs")
    one_hot = work[[f"union_route_{route.lower()}" for route in FOUR_ROUTES]].apply(
        pd.to_numeric, errors="coerce"
    )
    expected_one_hot = np.asarray(
        [
            [float(source_route == route) for route in FOUR_ROUTES]
            for source_route in work["source_route"]
        ]
    )
    if one_hot.isna().any().any() or not np.array_equal(
        one_hot.to_numpy(float), expected_one_hot
    ):
        raise ValueError("Top20 union route one-hot identity differs")
    counts = work.groupby("sample_id").size().reindex(denominator_ids, fill_value=0)
    if int(counts.max()) > MAX_UNION_CANDIDATES:
        raise ValueError("Top20 union exceeds 20 candidates")
    return {
        "candidate_rows": int(len(work)),
        "maximum_candidates_per_sample": int(counts.max()),
        "full_top20_samples": int((counts == MAX_UNION_CANDIDATES).sum()),
        "membership_sha256": canonical_sha256(
            work[
                [
                    "sample_id",
                    "candidate_id",
                    "source_route",
                    "source_candidate_id",
                    "route_native_rank",
                    "route_member_sha256",
                ]
            ].to_dict("records")
        ),
    }


def validation_router_union_summary(
    samples: pd.DataFrame, union: pd.DataFrame
) -> pd.DataFrame:
    """Build the P12 Validation-only long-form readiness table."""

    required_samples = {
        "sample_id",
        "scene_id",
        "crog_correct",
        "g1_correct",
        "c1_correct",
        "d1_correct",
        "three_route_router_correct",
        "four_route_decision",
        "existing_top15_oracle",
    }
    missing = sorted(required_samples.difference(samples.columns))
    if missing:
        raise ValueError(f"four-route Validation samples miss columns: {missing}")
    if samples.empty or samples["sample_id"].astype(str).duplicated().any():
        raise ValueError("four-route Validation samples must be non-empty and unique")
    work = samples.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    validate_top20_union_frame(union, work[["sample_id"]])
    decisions = work["four_route_decision"].astype(str).str.upper().to_numpy()
    if not set(decisions).issubset(FOUR_ROUTES):
        raise ValueError("four-route Validation decisions contain an unknown route")
    correct = {
        route: _binary(
            work[f"{route.lower()}_correct"],
            f"{route}_correct",
            length=len(work),
        )
        for route in FOUR_ROUTES
    }
    three_route = _binary(
        work["three_route_router_correct"],
        "three_route_router_correct",
        length=len(work),
    )
    oracle15 = _binary(
        work["existing_top15_oracle"],
        "existing_top15_oracle",
        length=len(work),
    )
    indexes = np.arange(len(work))
    selected = np.asarray(
        [correct[route][index] for index, route in zip(indexes, decisions)], dtype=bool
    )
    required_union = {"sample_id", "source_route", "candidate_success"}
    missing_union = sorted(required_union.difference(union.columns))
    if missing_union:
        raise ValueError(f"Top20 Validation union misses columns: {missing_union}")
    union_work = union.copy()
    union_work["sample_id"] = union_work["sample_id"].astype(str)
    union_work["source_route"] = union_work["source_route"].astype(str).str.upper()
    if not set(union_work["source_route"]).issubset(FOUR_ROUTES):
        raise ValueError("Top20 Validation union contains an unknown source route")
    if set(union_work["sample_id"]).difference(set(work["sample_id"])):
        raise ValueError("Top20 Validation union contains a foreign sample")
    union_success = _binary(
        union_work["candidate_success"],
        "candidate_success",
        length=len(union_work),
    )
    union_work["candidate_success"] = union_success
    oracle20_series = union_work.groupby("sample_id")["candidate_success"].any()
    oracle20 = work["sample_id"].map(oracle20_series).fillna(False).to_numpy(bool)
    rows: list[dict[str, Any]] = []

    def add(section: str, system: str, route: str, metric: str, value: Any) -> None:
        rows.append(
            {
                "section": section,
                "system": system,
                "route": route,
                "metric": metric,
                "value": float(value),
                "sample_count": int(len(work)),
                "development_split": "Validation",
            }
        )

    for route in FOUR_ROUTES:
        add("route", f"gated_{route}", route, "j_at_1", correct[route].mean())
    add("router", "three_route_router", "CROG/G1/C1", "j_at_1", three_route.mean())
    add("router", "four_route_router", "CROG/G1/C1/D1", "j_at_1", selected.mean())
    add(
        "router",
        "four_route_router",
        "CROG/G1/C1/D1",
        "delta_vs_gated_crog",
        selected.mean() - correct["CROG"].mean(),
    )
    add(
        "router",
        "four_route_router",
        "CROG/G1/C1/D1",
        "delta_vs_three_route_router",
        selected.mean() - three_route.mean(),
    )
    for route in FOUR_ROUTES:
        add(
            "router_switch",
            "four_route_router",
            route,
            "destination_count",
            int((decisions == route).sum()),
        )
    d1_selected = decisions == "D1"
    add(
        "router_d1",
        "four_route_router",
        "D1",
        "unique_recovered_count",
        int(
            (
                d1_selected
                & correct["D1"]
                & ~correct["CROG"]
                & ~correct["G1"]
                & ~correct["C1"]
            ).sum()
        ),
    )
    add(
        "router_d1",
        "four_route_router",
        "D1",
        "harmful_vs_crog_count",
        int((d1_selected & correct["CROG"] & ~correct["D1"]).sum()),
    )
    add(
        "router_d1",
        "four_route_router",
        "D1",
        "duplicates_g1_c1_error_count",
        int((d1_selected & ~correct["D1"] & ~correct["G1"] & ~correct["C1"]).sum()),
    )
    add("union", "existing_top15_union", "CROG/G1/C1", "oracle_at_15", oracle15.mean())
    add(
        "union",
        "four_route_top20_union",
        "CROG/G1/C1/D1",
        "oracle_at_20",
        oracle20.mean(),
    )
    add(
        "union",
        "four_route_top20_union",
        "D1",
        "oracle_delta_vs_top15",
        oracle20.mean() - oracle15.mean(),
    )
    positive = union_work.loc[union_work["candidate_success"]]
    route_has_positive = {
        route: set(positive.loc[positive["source_route"] == route, "sample_id"])
        for route in FOUR_ROUTES
    }
    for route in FOUR_ROUTES:
        others = set().union(
            *(route_has_positive[other] for other in FOUR_ROUTES if other != route)
        )
        add(
            "union_source",
            "four_route_top20_union",
            route,
            "positive_candidate_count",
            int((positive["source_route"] == route).sum()),
        )
        add(
            "union_complementarity",
            "four_route_top20_union",
            route,
            "unique_positive_sample_count",
            len(route_has_positive[route].difference(others)),
        )
    return pd.DataFrame(rows)


__all__ = [
    "ALTERNATIVE_ROUTES",
    "DEFAULT_ROUTE_TIE_BREAK",
    "FOUR_ROUTES",
    "FourRouteCROGDefaultTransitionRouter",
    "FourRouteRouterSelection",
    "FourRouteRouterTrial",
    "MAX_UNION_CANDIDATES",
    "build_top20_union",
    "four_route_decisions",
    "select_four_route_operating_point",
    "validate_top20_union_frame",
    "validation_router_union_summary",
]
