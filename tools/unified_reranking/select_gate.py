"""Fit an OOF-only conservative gate and lock its Validation operating point."""

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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.gate import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    ConservativeTransitionModel,
    GateEvidence,
    GateOperatingPoint,
    OOFTransitionData,
    SAFE_GATE_FEATURE_COLUMNS,
    gate_switch_mask,
    select_gate_operating_point,
)
from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage


@dataclass(frozen=True)
class GateColumns:
    sample_id: str = "sample_id"
    scene_id: str = "scene_id"
    provenance: str = "prediction_source"
    oof_fold: str = "oof_fold"
    native_correct: str = "native_correct"
    challenger_correct: str = "challenger_correct"
    native_candidate_id: str = "native_candidate_id"
    challenger_candidate_id: str = "challenger_candidate_id"
    score_margin: str = "score_margin"
    challenger_reliability: str = "challenger_reliability"
    perturbation_stability: str = "perturbation_stability"
    seed_challenger_votes: str = "seed_challenger_votes"
    candidate_id_unchanged: str = "candidate_id_unchanged"
    geometry_hash_unchanged: str = "geometry_hash_unchanged"
    challenger_exists: str = "challenger_exists"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "GateColumns":
        if values is None:
            return cls()
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"unknown gate column mappings: {unknown}")
        normalized = {str(key): str(value) for key, value in values.items()}
        if any(not value for value in normalized.values()):
            raise ValueError("gate column names must be non-empty")
        return cls(**normalized)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate_grid(path: Path) -> tuple[GateOperatingPoint, ...]:
    """Load either an explicit operating-point list or a Cartesian grid."""

    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("gate grid must be a JSON object")
    if "operating_points" in payload:
        rows = payload["operating_points"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("operating_points must be a non-empty list")
        points = tuple(GateOperatingPoint(**dict(row)) for row in rows)
    else:
        keys = {
            "lambda_harm": "lambda_harm",
            "utility_threshold": "utility_thresholds",
            "score_margin_threshold": "score_margin_thresholds",
            "reliability_threshold": "reliability_thresholds",
            "stability_threshold": "stability_thresholds",
        }
        missing = sorted(value for value in keys.values() if value not in payload)
        if missing:
            raise ValueError(f"gate Cartesian grid misses fields: {missing}")
        values = {name: tuple(payload[source]) for name, source in keys.items()}
        if any(not value for value in values.values()):
            raise ValueError("gate Cartesian grid dimensions must be non-empty")
        points = tuple(
            GateOperatingPoint(lam, utility, margin, reliability, stability)
            for lam in values["lambda_harm"]
            for utility in values["utility_threshold"]
            for margin in values["score_margin_threshold"]
            for reliability in values["reliability_threshold"]
            for stability in values["stability_threshold"]
        )
    if len({canonical_sha256(asdict(point)) for point in points}) != len(points):
        raise ValueError("gate grid contains duplicate operating points")
    return points


def _read_frame(path: Path, *, provenance_column: str, expected: str) -> pd.DataFrame:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular parquet input: {path}")
    frame = pd.read_parquet(path)
    if provenance_column not in frame:
        raise ValueError(f"input misses required provenance column {provenance_column!r}")
    provenance = set(frame[provenance_column].dropna().astype(str).str.lower())
    if provenance != {expected} or frame[provenance_column].isna().any():
        raise ValueError(
            f"input provenance must be exactly {expected!r}; observed {sorted(provenance)}"
        )
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} misses columns: {missing}")


def _validate_identity(frame: pd.DataFrame, columns: GateColumns, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} is empty")
    ids = frame[columns.sample_id]
    if ids.isna().any() or ids.astype(str).eq("").any() or ids.astype(str).duplicated().any():
        raise ValueError(f"{name} must have unique non-empty sample IDs")


def _artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _resume_manifest(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    previous = _load_json(marker)
    if previous.get("status") != "COMPLETE" or previous.get("signature_sha256") != signature:
        raise RuntimeError("immutable gate output exists with a different signature")
    unsigned = dict(previous)
    recorded_content = unsigned.pop("content_sha256", None)
    if recorded_content != canonical_sha256(unsigned):
        raise RuntimeError("resumable gate manifest content hash is invalid")
    for name, record in dict(previous.get("artifacts", {})).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"resumable gate artifact failed hash check: {name}")
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
            }
            for trial in result.trials
        ]
    )


def run_gate_selection(
    *,
    oof_path: Path,
    validation_path: Path,
    grid_path: Path,
    output_dir: Path,
    route: str,
    feature_columns: Sequence[str],
    columns: GateColumns = GateColumns(),
    model_seed: int = BOOTSTRAP_SEED,
    input_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Fit, validate, and immutably persist one route gate."""

    route = str(route).lower()
    if route not in {"crog", "g1", "c1"}:
        raise ValueError("route must be crog, g1, or c1")
    features = tuple(map(str, feature_columns))
    if not features or len(features) != len(set(features)):
        raise ValueError("feature_columns must be non-empty and unique")
    assert_model_feature_columns(features)
    if features != SAFE_GATE_FEATURE_COLUMNS:
        raise ValueError(
            "gate feature columns must exactly match the fixed producer schema; "
            f"expected {list(SAFE_GATE_FEATURE_COLUMNS)}, observed {list(features)}"
        )
    points = load_gate_grid(grid_path)
    configuration = {
        "route": route,
        "feature_columns": list(features),
        "columns": asdict(columns),
        "model_seed": int(model_seed),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "operating_points": [asdict(point) for point in points],
    }
    sources = {
        "train_oof": _artifact_record(oof_path),
        "validation": _artifact_record(validation_path),
        "predeclared_grid": _artifact_record(grid_path),
        "selection_code": [
            _artifact_record(path)
            for path in (
                Path(__file__).resolve(),
                SRC / "unified_reranking" / "gate.py",
                SRC / "unified_reranking" / "contracts.py",
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
    marker = output_dir / "gate_selection.json"
    resumed = _resume_manifest(marker, signature)
    if resumed is not None:
        return resumed

    oof = _read_frame(
        oof_path, provenance_column=columns.provenance, expected="train_oof"
    )
    validation = _read_frame(
        validation_path, provenance_column=columns.provenance, expected="validation"
    )
    common = [
        columns.sample_id,
        columns.scene_id,
        columns.native_correct,
        columns.challenger_correct,
        *features,
    ]
    _require_columns(oof, [*common, columns.oof_fold], "Train OOF")
    validation_required = [
        *common,
        columns.native_candidate_id,
        columns.challenger_candidate_id,
        columns.score_margin,
        columns.challenger_reliability,
        columns.perturbation_stability,
        columns.seed_challenger_votes,
        columns.candidate_id_unchanged,
        columns.geometry_hash_unchanged,
        columns.challenger_exists,
    ]
    _require_columns(validation, validation_required, "Validation")
    _validate_identity(oof, columns, "Train OOF")
    _validate_identity(validation, columns, "Validation")
    model = ConservativeTransitionModel(seed=int(model_seed)).fit(
        OOFTransitionData(
            features=oof.loc[:, features].to_numpy(float),
            feature_names=features,
            native_correct=oof[columns.native_correct].to_numpy(),
            challenger_correct=oof[columns.challenger_correct].to_numpy(),
            scene_ids=oof[columns.scene_id].to_numpy(),
            oof_fold_ids=oof[columns.oof_fold].to_numpy(),
            prediction_source="train_oof",
        )
    )
    probability_recover, probability_harm = model.predict_probabilities(
        validation.loc[:, features].to_numpy(float)
    )
    evidence = GateEvidence(
        score_margin=validation[columns.score_margin].to_numpy(),
        challenger_reliability=validation[columns.challenger_reliability].to_numpy(),
        perturbation_stability=validation[columns.perturbation_stability].to_numpy(),
        seed_challenger_votes=validation[columns.seed_challenger_votes].to_numpy(),
        candidate_id_unchanged=validation[columns.candidate_id_unchanged].to_numpy(),
        geometry_hash_unchanged=validation[columns.geometry_hash_unchanged].to_numpy(),
        challenger_exists=validation[columns.challenger_exists].to_numpy(),
    )
    result = select_gate_operating_point(
        probability_recover,
        probability_harm,
        evidence,
        validation[columns.native_correct].to_numpy(),
        validation[columns.challenger_correct].to_numpy(),
        validation[columns.scene_id].to_numpy(),
        points,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if result.selected_operating_point is None:
        switches = np.zeros(len(validation), dtype=bool)
        utility = np.full(len(validation), np.nan)
    else:
        point = result.selected_operating_point
        switches = gate_switch_mask(
            probability_recover, probability_harm, evidence, point
        )
        utility = probability_recover - point.lambda_harm * probability_harm
    native_correct = validation[columns.native_correct].astype(bool).to_numpy()
    challenger_correct = validation[columns.challenger_correct].astype(bool).to_numpy()
    native_ids = validation[columns.native_candidate_id].fillna("").astype(str).to_numpy()
    challenger_ids = (
        validation[columns.challenger_candidate_id].fillna("").astype(str).to_numpy()
    )
    decisions = pd.DataFrame(
        {
            "sample_id": validation[columns.sample_id].astype(str),
            "scene_id": validation[columns.scene_id].astype(str),
            "probability_recover": probability_recover,
            "probability_harm": probability_harm,
            "utility": utility,
            "switch": switches,
            "native_candidate_id": native_ids,
            "challenger_candidate_id": challenger_ids,
            "selected_candidate_id": np.where(switches, challenger_ids, native_ids),
            "native_correct": native_correct,
            "challenger_correct": challenger_correct,
            "selected_correct": np.where(switches, challenger_correct, native_correct),
            "transition": np.select(
                [~native_correct & challenger_correct, native_correct & ~challenger_correct],
                ["recovered_if_switched", "harmful_if_switched"],
                default="outcome_unchanged",
            ),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gate_transition_model.pkl"
    decisions_path = output_dir / "gate_validation_decisions.parquet"
    trials_path = output_dir / "gate_validation_trials.parquet"
    _atomic_pickle(model_path, model)
    _atomic_parquet(decisions_path, decisions)
    _atomic_parquet(trials_path, _trial_frame(result))
    artifacts = {
        "transition_model": _artifact_record(model_path),
        "validation_decisions": _artifact_record(decisions_path),
        "validation_trials": _artifact_record(trials_path),
    }
    manifest: dict[str, Any] = {
        "status": "COMPLETE",
        "decision": result.status,
        "signature_sha256": signature,
        "configuration": configuration,
        "sources": sources,
        "transition_model": model.artifact(),
        "selection": result.artifact(),
        "validation_sample_count": int(len(validation)),
        "artifacts": artifacts,
        "test_access": "NONE",
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(marker, manifest)
    return manifest


def execute_gate_selection(
    *,
    run_dir: Path,
    command: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    route = str(kwargs["route"]).lower()
    with ledger_stage(
        run_dir.resolve() / "run_ledger.sqlite",
        stage="P8",
        substage=f"conservative_gate_{route}",
        route=route,
        method="calibrated_expected_gain_gate",
        seed=int(kwargs.get("model_seed", BOOTSTRAP_SEED)),
        command=command,
    ) as state:
        manifest = run_gate_selection(**kwargs)
        marker = Path(kwargs["output_dir"]).resolve() / "gate_selection.json"
        state["artifact_path"] = str(marker)
        state["artifact_sha256"] = sha256_file(marker)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    parser.add_argument("--oof", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-columns", required=True, nargs="+")
    parser.add_argument("--columns-json", type=Path)
    parser.add_argument("--model-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns_payload = None if args.columns_json is None else _load_json(args.columns_json)
    columns = GateColumns.from_mapping(columns_payload)
    output = args.output_dir or (
        args.run_dir / "08_lock" / "gates" / str(args.route).lower()
    )
    execute_gate_selection(
        run_dir=args.run_dir,
        command=" ".join(map(str, sys.argv)),
        oof_path=args.oof.resolve(),
        validation_path=args.validation.resolve(),
        grid_path=args.grid.resolve(),
        output_dir=output.resolve(),
        route=args.route,
        feature_columns=args.feature_columns,
        columns=columns,
        model_seed=args.model_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GateColumns",
    "execute_gate_selection",
    "load_gate_grid",
    "run_gate_selection",
]
