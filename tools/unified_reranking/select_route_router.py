"""Fit and lock the paired Train-OOF CROG-default Validation route router."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.unified_reranking.select_gate import (
    _artifact_record,
    _atomic_parquet,
    _atomic_pickle,
    _load_json,
    _read_frame,
    _require_columns,
)
from unified_reranking.gate import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, OOFTransitionData
from unified_reranking.cross_route_inputs import (
    assert_router_feature_names,
    router_feature_columns,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.route_router import (
    CROGDefaultTransitionRouter,
    DEFAULT_ROUTE_TIE_BREAK,
    RouterEvidence,
    RouterOperatingPoint,
    route_decisions,
    route_utilities,
    select_router_operating_point,
)


@dataclass(frozen=True)
class RouterColumns:
    sample_id: str = "sample_id"
    scene_id: str = "scene_id"
    provenance: str = "prediction_source"
    oof_fold: str = "oof_fold"
    crog_correct: str = "crog_correct"
    g1_correct: str = "g1_correct"
    c1_correct: str = "c1_correct"
    crog_candidate_id: str = "crog_candidate_id"
    g1_candidate_id: str = "g1_candidate_id"
    c1_candidate_id: str = "c1_candidate_id"
    g1_margin: str = "g1_margin"
    c1_margin: str = "c1_margin"
    g1_reliability: str = "g1_reliability"
    c1_reliability: str = "c1_reliability"
    g1_stability: str = "g1_stability"
    c1_stability: str = "c1_stability"
    g1_candidate_exists: str = "g1_candidate_exists"
    c1_candidate_exists: str = "c1_candidate_exists"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RouterColumns":
        if values is None:
            return cls()
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"unknown router column mappings: {unknown}")
        normalized = {str(key): str(value) for key, value in values.items()}
        if any(not value for value in normalized.values()):
            raise ValueError("router column names must be non-empty")
        return cls(**normalized)


def load_router_grid(path: Path) -> tuple[RouterOperatingPoint, ...]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("router grid must be a JSON object")
    if "operating_points" in payload:
        rows = payload["operating_points"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("operating_points must be a non-empty list")
        points = tuple(RouterOperatingPoint(**dict(row)) for row in rows)
    else:
        keys = {
            "lambda_router": "lambda_router",
            "utility_threshold": "utility_thresholds",
            "margin_threshold": "margin_thresholds",
            "reliability_threshold": "reliability_thresholds",
            "stability_threshold": "stability_thresholds",
        }
        missing = sorted(value for value in keys.values() if value not in payload)
        if missing:
            raise ValueError(f"router Cartesian grid misses fields: {missing}")
        values = {name: tuple(payload[source]) for name, source in keys.items()}
        if any(not value for value in values.values()):
            raise ValueError("router Cartesian grid dimensions must be non-empty")
        points = tuple(
            RouterOperatingPoint(lam, utility, margin, reliability, stability)
            for lam in values["lambda_router"]
            for utility in values["utility_threshold"]
            for margin in values["margin_threshold"]
            for reliability in values["reliability_threshold"]
            for stability in values["stability_threshold"]
        )
    if len({canonical_sha256(asdict(point)) for point in points}) != len(points):
        raise ValueError("router grid contains duplicate operating points")
    return points


def _validate_identity(frame: pd.DataFrame, columns: RouterColumns, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} is empty")
    identifiers = frame[columns.sample_id]
    if (
        identifiers.isna().any()
        or identifiers.astype(str).eq("").any()
        or identifiers.astype(str).duplicated().any()
    ):
        raise ValueError(f"{name} must have unique non-empty sample IDs")


def _resume_manifest(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    previous = _load_json(marker)
    if previous.get("status") != "COMPLETE" or previous.get("signature_sha256") != signature:
        raise RuntimeError("immutable router output exists with a different signature")
    unsigned = dict(previous)
    recorded_content = unsigned.pop("content_sha256", None)
    if recorded_content != canonical_sha256(unsigned):
        raise RuntimeError("resumable router manifest content hash is invalid")
    for name, record in dict(previous.get("artifacts", {})).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"resumable router artifact failed hash check: {name}")
    return previous


def _trial_frame(result: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                **asdict(trial.operating_point),
                "bootstrap_lower_bound": trial.bootstrap_lower_bound,
                "mean_delta": trial.mean_delta,
                "recovered": trial.recovered,
                "harmful": trial.harmful,
                "switch_count": trial.switch_count,
                "switch_rate": trial.switch_rate,
                "g1_switches": trial.g1_switches,
                "c1_switches": trial.c1_switches,
            }
            for trial in result.trials
        ]
    )


def run_route_router_selection(
    *,
    oof_path: Path,
    validation_path: Path,
    grid_path: Path,
    output_dir: Path,
    g1_feature_columns: Sequence[str],
    c1_feature_columns: Sequence[str],
    columns: RouterColumns = RouterColumns(),
    model_seed: int = BOOTSTRAP_SEED,
    input_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Fit paired CROG->G1/C1 transition models and lock the Validation router."""

    feature_columns = {
        "G1": tuple(map(str, g1_feature_columns)),
        "C1": tuple(map(str, c1_feature_columns)),
    }
    for route, names in feature_columns.items():
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{route} feature columns must be non-empty and unique")
        assert_router_feature_names(names)
        expected = router_feature_columns(route)
        if names != expected:
            raise ValueError(
                f"{route} feature columns must exactly match the fixed producer schema; "
                f"expected {list(expected)}, observed {list(names)}"
            )
    points = load_router_grid(grid_path)
    configuration = {
        "default_route": "CROG",
        "tie_break": list(DEFAULT_ROUTE_TIE_BREAK),
        "feature_columns": {route: list(names) for route, names in feature_columns.items()},
        "columns": asdict(columns),
        "model_seed": int(model_seed),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "operating_points": [asdict(point) for point in points],
    }
    sources = {
        "paired_train_oof": _artifact_record(oof_path),
        "paired_validation": _artifact_record(validation_path),
        "predeclared_grid": _artifact_record(grid_path),
        "selection_code": [
            _artifact_record(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "tools" / "unified_reranking" / "select_gate.py",
                SRC / "unified_reranking" / "route_router.py",
                SRC / "unified_reranking" / "gate.py",
                SRC / "unified_reranking" / "cross_route_inputs.py",
            )
        ],
    }
    inferred_input_manifest = oof_path.resolve().parent / "manifest.json"
    input_manifest_path = (
        inferred_input_manifest
        if input_manifest_path is None and inferred_input_manifest.is_file()
        else input_manifest_path
    )
    if input_manifest_path is not None:
        sources["input_manifest"] = _artifact_record(input_manifest_path)
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output_dir = output_dir.resolve()
    marker = output_dir / "route_router_selection.json"
    resumed = _resume_manifest(marker, signature)
    if resumed is not None:
        return resumed

    oof = _read_frame(
        oof_path, provenance_column=columns.provenance, expected="train_oof"
    )
    validation = _read_frame(
        validation_path, provenance_column=columns.provenance, expected="validation"
    )
    shared = [
        columns.sample_id,
        columns.scene_id,
        columns.crog_correct,
        columns.g1_correct,
        columns.c1_correct,
    ]
    _require_columns(
        oof,
        [*shared, columns.oof_fold, *feature_columns["G1"], *feature_columns["C1"]],
        "paired Train OOF",
    )
    _require_columns(
        validation,
        [
            *shared,
            *feature_columns["G1"],
            *feature_columns["C1"],
            columns.crog_candidate_id,
            columns.g1_candidate_id,
            columns.c1_candidate_id,
            columns.g1_margin,
            columns.c1_margin,
            columns.g1_reliability,
            columns.c1_reliability,
            columns.g1_stability,
            columns.c1_stability,
            columns.g1_candidate_exists,
            columns.c1_candidate_exists,
        ],
        "paired Validation",
    )
    _validate_identity(oof, columns, "paired Train OOF")
    _validate_identity(validation, columns, "paired Validation")
    def transition_data(route: str, correct_column: str) -> OOFTransitionData:
        return OOFTransitionData(
            features=oof.loc[:, feature_columns[route]].to_numpy(float),
            feature_names=feature_columns[route],
            native_correct=oof[columns.crog_correct].to_numpy(),
            challenger_correct=oof[correct_column].to_numpy(),
            scene_ids=oof[columns.scene_id].to_numpy(),
            oof_fold_ids=oof[columns.oof_fold].to_numpy(),
            prediction_source="train_oof",
        )

    router = CROGDefaultTransitionRouter(seed=int(model_seed)).fit(
        transition_data("G1", columns.g1_correct),
        transition_data("C1", columns.c1_correct),
    )
    probabilities = router.predict_probabilities(
        {
            route: validation.loc[:, names].to_numpy(float)
            for route, names in feature_columns.items()
        }
    )
    evidence = {
        "G1": RouterEvidence(
            route_margin=validation[columns.g1_margin].to_numpy(),
            reliability=validation[columns.g1_reliability].to_numpy(),
            perturbation_stability=validation[columns.g1_stability].to_numpy(),
            candidate_exists=validation[columns.g1_candidate_exists].to_numpy(),
        ),
        "C1": RouterEvidence(
            route_margin=validation[columns.c1_margin].to_numpy(),
            reliability=validation[columns.c1_reliability].to_numpy(),
            perturbation_stability=validation[columns.c1_stability].to_numpy(),
            candidate_exists=validation[columns.c1_candidate_exists].to_numpy(),
        ),
    }
    correct = {
        "CROG": validation[columns.crog_correct].to_numpy(),
        "G1": validation[columns.g1_correct].to_numpy(),
        "C1": validation[columns.c1_correct].to_numpy(),
    }
    result = select_router_operating_point(
        probabilities,
        evidence,
        correct,
        validation[columns.scene_id].to_numpy(),
        points,
        tie_break=DEFAULT_ROUTE_TIE_BREAK,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if result.selected_operating_point is None:
        selected_routes = np.full(len(validation), "CROG", dtype=object)
        utilities = {
            route: np.full(len(validation), np.nan) for route in DEFAULT_ROUTE_TIE_BREAK
        }
    else:
        selected_routes = route_decisions(
            probabilities,
            evidence,
            result.selected_operating_point,
            tie_break=DEFAULT_ROUTE_TIE_BREAK,
        )
        utilities = route_utilities(
            probabilities,
            lambda_router=result.selected_operating_point.lambda_router,
        )
    candidate_ids = {
        "CROG": validation[columns.crog_candidate_id].fillna("").astype(str).to_numpy(),
        "G1": validation[columns.g1_candidate_id].fillna("").astype(str).to_numpy(),
        "C1": validation[columns.c1_candidate_id].fillna("").astype(str).to_numpy(),
    }
    correct_bool = {route: np.asarray(values, dtype=bool) for route, values in correct.items()}
    indexes = np.arange(len(validation))
    selected_candidate = np.asarray(
        [candidate_ids[str(route)][index] for index, route in zip(indexes, selected_routes)],
        dtype=object,
    )
    selected_correct = np.asarray(
        [correct_bool[str(route)][index] for index, route in zip(indexes, selected_routes)],
        dtype=bool,
    )
    decisions = pd.DataFrame(
        {
            "sample_id": validation[columns.sample_id].astype(str),
            "scene_id": validation[columns.scene_id].astype(str),
            "g1_probability_recover": probabilities["G1"][0],
            "g1_probability_harm": probabilities["G1"][1],
            "g1_utility": utilities["G1"],
            "c1_probability_recover": probabilities["C1"][0],
            "c1_probability_harm": probabilities["C1"][1],
            "c1_utility": utilities["C1"],
            "selected_route": selected_routes,
            "switched_from_crog": selected_routes != "CROG",
            "crog_candidate_id": candidate_ids["CROG"],
            "g1_candidate_id": candidate_ids["G1"],
            "c1_candidate_id": candidate_ids["C1"],
            "selected_candidate_id": selected_candidate,
            "crog_correct": correct_bool["CROG"],
            "g1_correct": correct_bool["G1"],
            "c1_correct": correct_bool["C1"],
            "selected_correct": selected_correct,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "route_transition_models.pkl"
    decisions_path = output_dir / "route_router_validation_decisions.parquet"
    trials_path = output_dir / "route_router_validation_trials.parquet"
    _atomic_pickle(model_path, router)
    _atomic_parquet(decisions_path, decisions)
    _atomic_parquet(trials_path, _trial_frame(result))
    artifacts = {
        "transition_models": _artifact_record(model_path),
        "validation_decisions": _artifact_record(decisions_path),
        "validation_trials": _artifact_record(trials_path),
    }
    manifest: dict[str, Any] = {
        "status": "COMPLETE",
        "decision": result.status,
        "signature_sha256": signature,
        "configuration": configuration,
        "sources": sources,
        "transition_models": router.artifact(),
        "selection": result.artifact(),
        "validation_sample_count": int(len(validation)),
        "artifacts": artifacts,
        "test_access": "NONE",
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(marker, manifest)
    return manifest


def execute_route_router_selection(
    *, run_dir: Path, command: str = "", **kwargs: Any
) -> dict[str, Any]:
    with ledger_stage(
        run_dir.resolve() / "run_ledger.sqlite",
        stage="P9",
        substage="crog_default_route_router",
        route="cross_route",
        method="calibrated_expected_gain_router",
        seed=int(kwargs.get("model_seed", BOOTSTRAP_SEED)),
        command=command,
    ) as state:
        manifest = run_route_router_selection(**kwargs)
        marker = Path(kwargs["output_dir"]).resolve() / "route_router_selection.json"
        state["artifact_path"] = str(marker)
        state["artifact_sha256"] = sha256_file(marker)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--g1-feature-columns", required=True, nargs="+")
    parser.add_argument("--c1-feature-columns", required=True, nargs="+")
    parser.add_argument("--columns-json", type=Path)
    parser.add_argument("--model-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns_payload = None if args.columns_json is None else _load_json(args.columns_json)
    columns = RouterColumns.from_mapping(columns_payload)
    output = args.output_dir or (args.run_dir / "08_lock" / "route_router")
    execute_route_router_selection(
        run_dir=args.run_dir,
        command=" ".join(map(str, sys.argv)),
        oof_path=args.oof.resolve(),
        validation_path=args.validation.resolve(),
        grid_path=args.grid.resolve(),
        output_dir=output.resolve(),
        g1_feature_columns=args.g1_feature_columns,
        c1_feature_columns=args.c1_feature_columns,
        columns=columns,
        model_seed=args.model_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RouterColumns",
    "execute_route_router_selection",
    "load_router_grid",
    "run_route_router_selection",
]
