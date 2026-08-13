"""Independent semantic replay for a locked D1 expected-gain gate.

The validator deliberately consumes only hash-bound development artifacts.  It
must complete before a caller opens any Test manifest or table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
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
from unified_reranking.hashing import canonical_sha256, sha256_file

from .execution import load_content_manifest


@dataclass(frozen=True)
class ValidatedGateSelection:
    """Development-only state independently replayed from a locked gate."""

    model: ConservativeTransitionModel
    selected_operating_point: GateOperatingPoint | None
    train_oof_path: Path
    validation_path: Path
    grid_path: Path
    transition_model_path: Path


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} is absent or is not a mapping")
    return value


def _evidence(frame: pd.DataFrame) -> GateEvidence:
    required = {
        "score_margin",
        "challenger_reliability",
        "perturbation_stability",
        "seed_challenger_votes",
        "candidate_id_unchanged",
        "geometry_hash_unchanged",
        "challenger_exists",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            f"D1 gate Validation inputs miss evidence columns: {missing}"
        )
    return GateEvidence(
        score_margin=frame["score_margin"].to_numpy(),
        challenger_reliability=frame["challenger_reliability"].to_numpy(),
        perturbation_stability=frame["perturbation_stability"].to_numpy(),
        seed_challenger_votes=frame["seed_challenger_votes"].to_numpy(),
        candidate_id_unchanged=frame["candidate_id_unchanged"].to_numpy(),
        geometry_hash_unchanged=frame["geometry_hash_unchanged"].to_numpy(),
        challenger_exists=frame["challenger_exists"].to_numpy(),
    )


def _grid_points(grid: Mapping[str, Any]) -> tuple[GateOperatingPoint, ...]:
    if (
        grid.get("minimum_seed_votes") != 2
        or grid.get("bootstrap_iterations") != BOOTSTRAP_ITERATIONS
        or grid.get("bootstrap_seed") != BOOTSTRAP_SEED
    ):
        raise RuntimeError("D1 gate grid bootstrap/consensus contract differs")
    try:
        points = tuple(
            GateOperatingPoint(lam, utility, margin, reliability, stability)
            for lam in grid["lambda_harm"]
            for utility in grid["utility_thresholds"]
            for margin in grid["score_margin_thresholds"]
            for reliability in grid["reliability_thresholds"]
            for stability in grid["stability_thresholds"]
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError("D1 gate grid is incomplete") from error
    identities = {canonical_sha256(asdict(point)) for point in points}
    if len(points) != 108 or len(identities) != 108:
        raise RuntimeError("D1 gate grid must contain 108 unique operating points")
    return points


def _required_development_columns(
    frame: pd.DataFrame, *, prediction_source: str, train: bool
) -> None:
    required = {
        *SAFE_GATE_FEATURE_COLUMNS,
        "native_correct",
        "challenger_correct",
        "scene_id",
        "prediction_source",
    }
    if train:
        required.add("oof_fold")
    else:
        required.update(
            {
                "sample_id",
                "native_candidate_id",
                "challenger_candidate_id",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(
            f"D1 gate {prediction_source} inputs miss columns: {missing}"
        )
    observed_sources = set(frame["prediction_source"].astype(str).unique().tolist())
    if observed_sources != {prediction_source}:
        raise RuntimeError(
            f"D1 gate {prediction_source} inputs have prediction-source drift"
        )


def _validation_decisions(
    validation: pd.DataFrame,
    probability_recover: np.ndarray,
    probability_harm: np.ndarray,
    selected_point: GateOperatingPoint | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if selected_point is None:
        switches = np.zeros(len(validation), dtype=bool)
        utility = probability_recover - probability_harm
    else:
        switches = gate_switch_mask(
            probability_recover,
            probability_harm,
            _evidence(validation),
            selected_point,
        )
        utility = probability_recover - selected_point.lambda_harm * probability_harm
    native = validation["native_correct"].astype(bool).to_numpy()
    challenger = validation["challenger_correct"].astype(bool).to_numpy()
    selected_correct = np.where(switches, challenger, native)
    native_ids = validation["native_candidate_id"].fillna("").astype(str).to_numpy()
    challenger_ids = (
        validation["challenger_candidate_id"].fillna("").astype(str).to_numpy()
    )
    decisions = pd.DataFrame(
        {
            "sample_id": validation["sample_id"].astype(str),
            "scene_id": validation["scene_id"].astype(str),
            "probability_recover": probability_recover,
            "probability_harm": probability_harm,
            "utility": utility,
            "switch": switches,
            "native_candidate_id": native_ids,
            "challenger_candidate_id": challenger_ids,
            "selected_candidate_id": np.where(switches, challenger_ids, native_ids),
            "native_correct": native,
            "challenger_correct": challenger,
            "selected_correct": selected_correct,
        }
    )
    return decisions, switches, native, challenger


def _validation_metrics(
    switches: np.ndarray, native: np.ndarray, challenger: np.ndarray
) -> dict[str, float | int | None]:
    selected_correct = np.where(switches, challenger, native)
    recovered = int((~native & challenger).sum())
    harmful = int((native & ~challenger).sum())
    selected_recovered = int((~native & challenger & switches).sum())
    selected_harmful = int((native & ~challenger & switches).sum())
    return {
        "sample_count": len(native),
        "native_j_at_1": float(native.mean()),
        "ungated_j_at_1": float(challenger.mean()),
        "gated_j_at_1": float(selected_correct.mean()),
        "gated_delta_vs_native": float(selected_correct.mean() - native.mean()),
        "gated_delta_vs_ungated": float(selected_correct.mean() - challenger.mean()),
        "ungated_recovered": recovered,
        "ungated_harmful": harmful,
        "gate_prevented_harmful": int((native & ~challenger & ~switches).sum()),
        "gate_missed_recoverable": int((~native & challenger & ~switches).sum()),
        "switch_count": int(switches.sum()),
        "switch_rate": float(switches.mean()),
        "outcome_changing_precision": (
            None
            if selected_recovered + selected_harmful == 0
            else selected_recovered / (selected_recovered + selected_harmful)
        ),
    }


def _assert_frame_exact(
    observed_path: Path, expected: pd.DataFrame, *, name: str
) -> None:
    observed = pd.read_parquet(observed_path)
    try:
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as error:
        raise RuntimeError(f"{name} differs from semantic replay") from error


def validate_gate_selection_semantics(
    gate: Mapping[str, Any],
) -> ValidatedGateSelection:
    """Recompute a locked D1 gate solely from hash-bound development inputs."""

    if gate.get("status") != "COMPLETE":
        raise RuntimeError("D1 gate selection is not COMPLETE")
    if gate.get("candidate_test_labels_read") is not False:
        raise RuntimeError("D1 gate selection violates Test-label isolation")
    configuration = _mapping(gate.get("configuration"), name="D1 gate configuration")
    configured_features = tuple(configuration.get("feature_columns", ()))
    if configured_features != tuple(SAFE_GATE_FEATURE_COLUMNS):
        raise RuntimeError("D1 gate feature schema differs")
    if (
        configuration.get("route") != "D1"
        or configuration.get("method") != "R7"
        or configuration.get("model_seed") != BOOTSTRAP_SEED
        or configuration.get("operating_point_count") != 108
        or configuration.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 gate configuration differs from the locked protocol")
    sources = _mapping(gate.get("sources"), name="D1 gate sources")
    expected_signature = canonical_sha256(
        {"configuration": dict(configuration), "sources": dict(sources)}
    )
    if gate.get("source_signature_sha256") != expected_signature:
        raise RuntimeError("D1 gate source signature differs")
    verify_artifact_records_recursive(
        sources,
        name="D1 gate development sources",
        require_at_least_one=True,
    )

    train_oof_path = verified_artifact_path(
        _mapping(sources.get("train_oof"), name="D1 gate Train OOF record"),
        name="D1 gate Train OOF inputs",
    )
    validation_path = verified_artifact_path(
        _mapping(sources.get("validation"), name="D1 gate Validation record"),
        name="D1 gate Validation inputs",
    )
    grid_path = verified_artifact_path(
        _mapping(sources.get("grid"), name="D1 gate grid record"),
        name="D1 locked gate grid",
    )
    input_manifest_path = verified_artifact_path(
        _mapping(sources.get("input_manifest"), name="D1 gate input-manifest record"),
        name="D1 gate input manifest",
    )
    input_manifest = load_content_manifest(
        input_manifest_path, name="D1 gate inputs", statuses=("COMPLETE",)
    )
    input_configuration = _mapping(
        input_manifest.get("configuration"), name="D1 gate input configuration"
    )
    if (
        tuple(input_configuration.get("feature_columns", ()))
        != tuple(SAFE_GATE_FEATURE_COLUMNS)
        or input_configuration.get("train_prediction_source") != "train_oof"
        or input_configuration.get("validation_prediction_source") != "validation"
        or input_configuration.get("candidate_test_labels_read") is not False
        or input_manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError("D1 gate input-manifest configuration differs")
    input_artifacts = _mapping(
        input_manifest.get("artifacts"), name="D1 gate input artifacts"
    )
    if input_artifacts.get("train_oof") != sources.get(
        "train_oof"
    ) or input_artifacts.get("validation") != sources.get("validation"):
        raise RuntimeError(
            "D1 gate Train OOF/Validation sources do not bind the input manifest"
        )
    verify_artifact_records_recursive(
        {
            "sources": input_manifest.get("sources"),
            "artifacts": input_artifacts,
        },
        name="D1 gate input manifest",
        require_at_least_one=True,
    )
    grid = load_content_manifest(grid_path, name="D1 gate grid", statuses=("PLANNED",))
    verify_artifact_records_recursive(
        grid.get("sources"),
        name="D1 gate grid sources",
        require_at_least_one=True,
    )
    points = _grid_points(grid)

    train = pd.read_parquet(train_oof_path)
    validation = pd.read_parquet(validation_path)
    _required_development_columns(train, prediction_source="train_oof", train=True)
    _required_development_columns(
        validation, prediction_source="validation", train=False
    )
    model = ConservativeTransitionModel(seed=BOOTSTRAP_SEED).fit(
        OOFTransitionData(
            features=train.loc[:, configured_features].to_numpy(float),
            feature_names=configured_features,
            native_correct=train["native_correct"].to_numpy(),
            challenger_correct=train["challenger_correct"].to_numpy(),
            scene_ids=train["scene_id"].to_numpy(),
            oof_fold_ids=train["oof_fold"].to_numpy(),
            prediction_source="train_oof",
        )
    )
    if model.artifact() != gate.get("transition_model"):
        raise RuntimeError("D1 rebuilt gate transition model differs from its manifest")

    artifacts = _mapping(gate.get("artifacts"), name="D1 gate artifacts")
    transition_model_path = verified_artifact_path(
        _mapping(
            artifacts.get("transition_model"), name="D1 gate transition-model record"
        ),
        name="D1 gate transition model",
    )
    rebuilt_model_sha = hashlib.sha256(
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()
    if rebuilt_model_sha != sha256_file(transition_model_path):
        raise RuntimeError(
            "D1 gate transition-model artifact differs from semantic replay"
        )

    probability_recover, probability_harm = model.predict_probabilities(
        validation.loc[:, configured_features].to_numpy(float)
    )
    selection = select_gate_operating_point(
        probability_recover,
        probability_harm,
        _evidence(validation),
        validation["native_correct"].to_numpy(),
        validation["challenger_correct"].to_numpy(),
        validation["scene_id"].to_numpy(),
        points,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if selection.artifact() != gate.get("selection"):
        raise RuntimeError(
            "D1 gate bootstrap trials/selection differ from semantic replay"
        )
    if selection.status != gate.get("decision"):
        raise RuntimeError("D1 gate decision differs from semantic replay")

    decisions, switches, native, challenger = _validation_decisions(
        validation,
        probability_recover,
        probability_harm,
        selection.selected_operating_point,
    )
    trials = pd.DataFrame(
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
            for trial in selection.trials
        ]
    )
    metrics = _validation_metrics(switches, native, challenger)
    if metrics != gate.get("validation_metrics"):
        raise RuntimeError("D1 gate Validation metrics differ from semantic replay")

    decision_path = verified_artifact_path(
        _mapping(
            artifacts.get("validation_decisions"),
            name="D1 gate Validation-decision record",
        ),
        name="D1 gate Validation decisions",
    )
    trial_path = verified_artifact_path(
        _mapping(
            artifacts.get("validation_trials"),
            name="D1 gate Validation-trial record",
        ),
        name="D1 gate Validation trials",
    )
    metrics_path = verified_artifact_path(
        _mapping(
            artifacts.get("gate_operating_point_table"),
            name="D1 gate Validation-metrics record",
        ),
        name="D1 gate Validation metrics",
    )
    _assert_frame_exact(decision_path, decisions, name="D1 gate Validation decisions")
    _assert_frame_exact(trial_path, trials, name="D1 gate Validation trials")
    expected_metrics_csv = pd.DataFrame([metrics]).to_csv(index=False)
    if metrics_path.read_text(encoding="utf-8") != expected_metrics_csv:
        raise RuntimeError(
            "D1 gate Validation metrics artifact differs from semantic replay"
        )

    return ValidatedGateSelection(
        model=model,
        selected_operating_point=selection.selected_operating_point,
        train_oof_path=train_oof_path,
        validation_path=validation_path,
        grid_path=grid_path,
        transition_model_path=transition_model_path,
    )


__all__ = ["ValidatedGateSelection", "validate_gate_selection_semantics"]
