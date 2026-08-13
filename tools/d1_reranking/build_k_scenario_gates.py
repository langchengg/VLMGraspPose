"""Select scenario-specific R7 gates for Top10-T2 and AllNMS-T2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.gate_inputs import build_gate_input_frame  # noqa: E402
from d1_reranking.gate_validation import validate_gate_selection_semantics  # noqa: E402
from d1_reranking.io import atomic_parquet, atomic_pickle  # noqa: E402
from d1_reranking.k_replay import validate_k_selection  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.gate import (  # noqa: E402
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
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


SCENARIOS = {
    "top10_t2": {"pool": "top10", "track": "T2_matched_common"},
    "allnms_t2": {"pool": "allnms", "track": "T2_matched_common"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def _assert_frame(observed: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    except AssertionError as error:
        raise RuntimeError(f"{name} differs from semantic replay") from error


def _candidate_manifest_path(root: Path, split: str) -> Path:
    return root / "02_candidates" / split / "manifest.json"


def _paired_path(root: Path, split: str) -> Path:
    return root / "01_manifests" / f"d1_paired_{split}.parquet"


def _native_decisions(
    paired: pd.DataFrame, candidates: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    top = candidates.loc[
        candidates["native_rank"].eq(1), ["sample_id", "candidate_id"]
    ].merge(
        labels[["sample_id", "candidate_id", "candidate_success"]],
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if top["candidate_success"].isna().any():
        raise RuntimeError("D1 K native decisions lack development labels")
    top = top.rename(
        columns={
            "candidate_id": "selected_candidate_id",
            "candidate_success": "selected_correct",
        }
    )
    result = paired[["sample_id"]].merge(
        top, on="sample_id", how="left", validate="one_to_one"
    )
    result["selected_correct"] = result["selected_correct"].fillna(False).astype(bool)
    return result


def _development_inputs(
    root: Path,
    *,
    split: str,
    scenario_id: str,
    definition: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pool = str(definition["pool"])
    track = str(definition["track"])
    candidate_manifest_path = _candidate_manifest_path(root, split)
    candidate_manifest = load_content_manifest(
        candidate_manifest_path,
        name=f"D1 K {split} candidate manifest",
        statuses=("COMPLETE",),
    )
    candidate_path = verified_artifact_path(
        _mapping(
            _mapping(
                candidate_manifest.get("artifacts"), name="K candidate artifacts"
            ).get(pool),
            name=f"D1 K {split}/{pool} candidates",
        ),
        name=f"D1 K {split}/{pool} candidates",
    )
    feature_manifest_path = root / f"03_features/{split}/{pool}/{track}/manifest.json"
    feature_manifest = load_content_manifest(
        feature_manifest_path,
        name=f"D1 K {split}/{pool}/{track} features",
        statuses=("COMPLETE",),
    )
    feature_path = verified_artifact_path(
        _mapping(
            _mapping(feature_manifest.get("artifacts"), name="K feature artifacts").get(
                "candidate_features"
            ),
            name="D1 K feature table",
        ),
        name=f"D1 K {split}/{pool} feature table",
    )
    label_manifest_path = root / f"03_features/{split}/{pool}/labels/manifest.json"
    label_manifest = load_content_manifest(
        label_manifest_path,
        name=f"D1 K {split}/{pool} labels",
        statuses=("COMPLETE",),
    )
    label_path = verified_artifact_path(
        _mapping(label_manifest.get("artifact"), name="D1 K labels"),
        name=f"D1 K {split}/{pool} labels",
    )
    paired_path = _paired_path(root, split)
    paired = pd.read_parquet(paired_path, columns=["sample_id", "scene_id"])
    candidates = pd.read_parquet(candidate_path)
    labels = pd.read_parquet(label_path)
    native_predictions = candidates.loc[
        :,
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    ]
    native_decisions = _native_decisions(paired, candidates, labels)
    prediction_key = "oof_predictions" if split == "train" else "validation_predictions"
    decision_key = "oof_decisions" if split == "train" else "validation_decisions"
    artifacts = _mapping(scenario.get("artifacts"), name=f"K {scenario_id} artifacts")
    challenger_predictions_path = verified_artifact_path(
        _mapping(artifacts.get(prediction_key), name=f"K {scenario_id} predictions"),
        name=f"D1 K {scenario_id} {split} predictions",
    )
    challenger_decisions_path = verified_artifact_path(
        _mapping(artifacts.get(decision_key), name=f"K {scenario_id} decisions"),
        name=f"D1 K {scenario_id} {split} decisions",
    )
    folds_path = root / "04_splits/fold_assignments.parquet"
    frame = build_gate_input_frame(
        paired=paired,
        native_predictions=native_predictions,
        native_decisions=native_decisions,
        challenger_predictions=pd.read_parquet(challenger_predictions_path),
        challenger_decisions=pd.read_parquet(challenger_decisions_path),
        candidate_features=pd.read_parquet(feature_path),
        candidates=candidates,
        prediction_source="train_oof" if split == "train" else "validation",
        folds=pd.read_parquet(folds_path) if split == "train" else None,
    )
    return frame, {
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "feature_manifest": artifact_record(feature_manifest_path),
        "features": artifact_record(feature_path),
        "label_manifest": artifact_record(label_manifest_path),
        "labels": artifact_record(label_path),
        "paired": artifact_record(paired_path),
        "folds": artifact_record(folds_path) if split == "train" else None,
        "challenger_predictions": artifact_record(challenger_predictions_path),
        "challenger_decisions": artifact_record(challenger_decisions_path),
    }


def _evidence(frame: pd.DataFrame) -> GateEvidence:
    return GateEvidence(
        score_margin=frame["score_margin"].to_numpy(float),
        challenger_reliability=frame["challenger_reliability"].to_numpy(float),
        perturbation_stability=frame["perturbation_stability"].to_numpy(float),
        seed_challenger_votes=frame["seed_challenger_votes"].to_numpy(float),
        candidate_id_unchanged=frame["candidate_id_unchanged"].to_numpy(bool),
        geometry_hash_unchanged=frame["geometry_hash_unchanged"].to_numpy(bool),
        challenger_exists=frame["challenger_exists"].to_numpy(bool),
    )


def _grid(value: Mapping[str, Any]) -> tuple[GateOperatingPoint, ...]:
    if (
        value.get("minimum_seed_votes") != 2
        or value.get("bootstrap_iterations") != BOOTSTRAP_ITERATIONS
        or value.get("bootstrap_seed") != BOOTSTRAP_SEED
    ):
        raise RuntimeError("D1 K gate grid contract differs")
    points = tuple(
        GateOperatingPoint(lam, utility, margin, reliability, stability)
        for lam in value["lambda_harm"]
        for utility in value["utility_thresholds"]
        for margin in value["score_margin_thresholds"]
        for reliability in value["reliability_thresholds"]
        for stability in value["stability_thresholds"]
    )
    if len(points) != 108:
        raise RuntimeError("D1 K gate requires exactly 108 operating points")
    return points


def _fit_gate(
    root: Path,
    *,
    scenario_id: str,
    definition: Mapping[str, Any],
    selection_record: Mapping[str, Any],
    scenario: Mapping[str, Any],
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = root / "11_k_sensitivity/gates" / scenario_id
    input_manifest_path = output / "inputs/manifest.json"
    train, train_sources = _development_inputs(
        root,
        split="train",
        scenario_id=scenario_id,
        definition=definition,
        scenario=scenario,
    )
    validation, validation_sources = _development_inputs(
        root,
        split="validation",
        scenario_id=scenario_id,
        definition=definition,
        scenario=scenario,
    )
    input_configuration = {
        "schema_version": 1,
        "route": "D1",
        "scenario_id": scenario_id,
        "pool": definition["pool"],
        "track": definition["track"],
        "feature_columns": list(SAFE_GATE_FEATURE_COLUMNS),
        "train_prediction_source": "train_oof",
        "validation_prediction_source": "validation",
        "candidate_test_labels_read": False,
    }
    input_sources = {
        "k_selection": dict(selection_record),
        "train": train_sources,
        "validation": validation_sources,
        "builder": artifact_record(ROOT / "src/d1_reranking/gate_inputs.py"),
        "tool": artifact_record(Path(__file__)),
    }
    input_signature = canonical_sha256(
        {"configuration": input_configuration, "sources": input_sources}
    )
    if input_manifest_path.exists():
        inputs = load_content_manifest(
            input_manifest_path,
            name=f"D1 K {scenario_id} gate inputs",
            statuses=("COMPLETE",),
        )
        if (
            not resume
            or inputs.get("source_signature_sha256") != input_signature
            or inputs.get("configuration") != input_configuration
            or inputs.get("sources") != input_sources
            or inputs.get("train_sample_count") != len(train)
            or inputs.get("validation_sample_count") != len(validation)
            or inputs.get("candidate_test_labels_read") is not False
        ):
            raise RuntimeError(f"D1 K {scenario_id} gate inputs differ")
        verify_artifact_records_recursive(
            inputs,
            name=f"D1 K {scenario_id} gate input resume",
            require_at_least_one=True,
        )
        input_artifacts = _mapping(
            inputs.get("artifacts"), name=f"D1 K {scenario_id} input artifacts"
        )
        _assert_frame(
            pd.read_parquet(
                verified_artifact_path(
                    _mapping(input_artifacts.get("train_oof"), name="K gate Train"),
                    name=f"D1 K {scenario_id} gate Train",
                )
            ),
            train,
            name=f"D1 K {scenario_id} gate Train inputs",
        )
        _assert_frame(
            pd.read_parquet(
                verified_artifact_path(
                    _mapping(
                        input_artifacts.get("validation"), name="K gate Validation"
                    ),
                    name=f"D1 K {scenario_id} gate Validation",
                )
            ),
            validation,
            name=f"D1 K {scenario_id} gate Validation inputs",
        )
    else:
        input_artifacts = {
            "train_oof": artifact_record(
                atomic_parquet(train, output / "inputs/train_oof.parquet")
            ),
            "validation": artifact_record(
                atomic_parquet(validation, output / "inputs/validation.parquet")
            ),
        }
        inputs = {
            "schema_version": 1,
            "status": "COMPLETE",
            "source_signature_sha256": input_signature,
            "configuration": input_configuration,
            "train_sample_count": len(train),
            "validation_sample_count": len(validation),
            "candidate_test_labels_read": False,
            "sources": input_sources,
            "artifacts": input_artifacts,
        }
        inputs["content_sha256"] = canonical_sha256(inputs)
        atomic_json(input_manifest_path, inputs)

    grid_path = root / "configs/d1_gate_grid.json"
    grid = load_content_manifest(grid_path, name="D1 gate grid", statuses=("PLANNED",))
    train_path = verified_artifact_path(
        _mapping(inputs["artifacts"]["train_oof"], name="K gate Train"),
        name=f"D1 K {scenario_id} gate Train",
    )
    validation_path = verified_artifact_path(
        _mapping(inputs["artifacts"]["validation"], name="K gate Validation"),
        name=f"D1 K {scenario_id} gate Validation",
    )
    gate_sources = {
        "input_manifest": artifact_record(input_manifest_path),
        "train_oof": artifact_record(train_path),
        "validation": artifact_record(validation_path),
        "grid": artifact_record(grid_path),
        "gate_primitive": artifact_record(ROOT / "src/unified_reranking/gate.py"),
        "tool": artifact_record(Path(__file__)),
    }
    gate_configuration = {
        "schema_version": 1,
        "route": "D1",
        "scenario_id": scenario_id,
        "pool": definition["pool"],
        "track": definition["track"],
        "method": "R7",
        "feature_columns": list(SAFE_GATE_FEATURE_COLUMNS),
        "model_seed": BOOTSTRAP_SEED,
        "operating_point_count": 108,
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256(
        {"configuration": gate_configuration, "sources": gate_sources}
    )
    manifest_path = output / "gate_selection.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name=f"D1 K {scenario_id} gate", statuses=("COMPLETE",)
        )
        if not resume or existing.get("source_signature_sha256") != signature:
            raise RuntimeError(f"D1 K {scenario_id} gate differs")
        if (
            existing.get("configuration") != gate_configuration
            or existing.get("sources") != gate_sources
            or existing.get("candidate_test_labels_read") is not False
        ):
            raise RuntimeError(f"D1 K {scenario_id} gate resume contract differs")
        validate_gate_selection_semantics(existing)
        return existing, inputs

    train_frame = pd.read_parquet(train_path)
    validation_frame = pd.read_parquet(validation_path)
    features = tuple(SAFE_GATE_FEATURE_COLUMNS)
    model = ConservativeTransitionModel(seed=BOOTSTRAP_SEED).fit(
        OOFTransitionData(
            features=train_frame.loc[:, features].to_numpy(float),
            feature_names=features,
            native_correct=train_frame["native_correct"].to_numpy(),
            challenger_correct=train_frame["challenger_correct"].to_numpy(),
            scene_ids=train_frame["scene_id"].to_numpy(),
            oof_fold_ids=train_frame["oof_fold"].to_numpy(),
            prediction_source="train_oof",
        )
    )
    probability_recover, probability_harm = model.predict_probabilities(
        validation_frame.loc[:, features].to_numpy(float)
    )
    selection = select_gate_operating_point(
        probability_recover,
        probability_harm,
        _evidence(validation_frame),
        validation_frame["native_correct"].to_numpy(),
        validation_frame["challenger_correct"].to_numpy(),
        validation_frame["scene_id"].to_numpy(),
        _grid(grid),
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if selection.selected_operating_point is None:
        switches = np.zeros(len(validation_frame), dtype=bool)
        utility = probability_recover - probability_harm
    else:
        switches = gate_switch_mask(
            probability_recover,
            probability_harm,
            _evidence(validation_frame),
            selection.selected_operating_point,
        )
        utility = (
            probability_recover
            - selection.selected_operating_point.lambda_harm * probability_harm
        )
    native = validation_frame["native_correct"].astype(bool).to_numpy()
    challenger = validation_frame["challenger_correct"].astype(bool).to_numpy()
    selected_correct = np.where(switches, challenger, native)
    native_ids = (
        validation_frame["native_candidate_id"].fillna("").astype(str).to_numpy()
    )
    challenger_ids = (
        validation_frame["challenger_candidate_id"].fillna("").astype(str).to_numpy()
    )
    decisions = pd.DataFrame(
        {
            "sample_id": validation_frame["sample_id"].astype(str),
            "scene_id": validation_frame["scene_id"].astype(str),
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
    selected_recovered = int((~native & challenger & switches).sum())
    selected_harmful = int((native & ~challenger & switches).sum())
    metrics = {
        "sample_count": len(validation_frame),
        "native_j_at_1": float(native.mean()),
        "ungated_j_at_1": float(challenger.mean()),
        "gated_j_at_1": float(selected_correct.mean()),
        "gated_delta_vs_native": float(selected_correct.mean() - native.mean()),
        "gated_delta_vs_ungated": float(selected_correct.mean() - challenger.mean()),
        "ungated_recovered": int((~native & challenger).sum()),
        "ungated_harmful": int((native & ~challenger).sum()),
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
    metrics_path = output / "gate_operating_point.csv"
    atomic_text(metrics_path, pd.DataFrame([metrics]).to_csv(index=False))
    gate_artifacts = {
        "transition_model": artifact_record(
            atomic_pickle(model, output / "transition_model.pkl")
        ),
        "validation_decisions": artifact_record(
            atomic_parquet(decisions, output / "validation_decisions.parquet")
        ),
        "validation_trials": artifact_record(
            atomic_parquet(trials, output / "validation_trials.parquet")
        ),
        "gate_operating_point_table": artifact_record(metrics_path),
    }
    gate = {
        "schema_version": 1,
        "status": "COMPLETE",
        "decision": selection.status,
        "source_signature_sha256": signature,
        "configuration": gate_configuration,
        "transition_model": model.artifact(),
        "selection": selection.artifact(),
        "validation_metrics": metrics,
        "candidate_test_labels_read": False,
        "sources": gate_sources,
        "artifacts": gate_artifacts,
    }
    gate["content_sha256"] = canonical_sha256(gate)
    atomic_json(manifest_path, gate)
    validate_gate_selection_semantics(gate)
    return gate, inputs


def run(run_dir: Path, *, resume: bool) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    validate_k_selection(root)
    selection_path = root / "11_k_sensitivity/selection_manifest.json"
    selection = load_content_manifest(
        selection_path, name="D1 K selection", statuses=("COMPLETE",)
    )
    scenarios = _mapping(selection.get("scenarios"), name="D1 K scenarios")
    gate_records: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for scenario_id, definition in SCENARIOS.items():
        scenario = _mapping(scenarios.get(scenario_id), name=f"D1 K {scenario_id}")
        observed_definition = _mapping(
            scenario.get("definition"), name=f"D1 K {scenario_id} definition"
        )
        if (
            observed_definition.get("scenario_id") != scenario_id
            or observed_definition.get("pool") != definition["pool"]
            or observed_definition.get("track") != definition["track"]
            or not isinstance(observed_definition.get("max_candidates"), int)
            or int(observed_definition["max_candidates"]) <= 0
            or (
                scenario_id == "top10_t2"
                and observed_definition.get("max_candidates") != 10
            )
        ):
            raise RuntimeError(f"D1 K {scenario_id} definition differs")
        gate, inputs = _fit_gate(
            root,
            scenario_id=scenario_id,
            definition=scenario["definition"],
            selection_record=artifact_record(selection_path),
            scenario=scenario,
            resume=resume,
        )
        gate_records[scenario_id] = {
            "inputs": artifact_record(
                root / f"11_k_sensitivity/gates/{scenario_id}/inputs/manifest.json"
            ),
            "selection": artifact_record(
                root / f"11_k_sensitivity/gates/{scenario_id}/gate_selection.json"
            ),
        }
        metrics = gate["validation_metrics"]
        for method, metric_key, role in (
            ("R0", "native_j_at_1", "native_baseline"),
            (
                str(selection["scenarios"][scenario_id]["method"]),
                "ungated_j_at_1",
                "ungated",
            ),
            ("R7", "gated_j_at_1", "expected_gain_gate"),
        ):
            comparison_rows.append(
                {
                    "scenario_id": scenario_id,
                    "pool": definition["pool"],
                    "track": definition["track"],
                    "method": method,
                    "role": role,
                    "validation_j_at_1": metrics[metric_key],
                    "gate_decision": gate["decision"]
                    if method == "R7"
                    else "NOT_APPLICABLE",
                }
            )
        verify_artifact_records_recursive(
            inputs, name=f"D1 K {scenario_id} inputs", require_at_least_one=True
        )
    comparison_path = root / "11_k_sensitivity/k_r0_ungated_r7_comparison.csv"
    expected_comparison = pd.DataFrame(comparison_rows)
    sources = {
        "k_selection": artifact_record(selection_path),
        "gates": gate_records,
        "grid": artifact_record(root / "configs/d1_gate_grid.json"),
        "tool": artifact_record(Path(__file__)),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "analysis": "K_scenario_R0_ungated_R7",
        "candidate_test_labels_read": False,
        "allnms_t3_sensitivity_only": True,
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
    }
    path = root / "11_k_sensitivity/scenario_gate_selection.json"
    if path.exists():
        existing = load_content_manifest(
            path, name="D1 K scenario gates", statuses=("COMPLETE",)
        )
        if (
            not resume
            or existing.get("sources") != sources
            or existing.get("source_signature_sha256") != canonical_sha256(sources)
            or existing.get("candidate_test_labels_read") is not False
            or existing.get("allnms_t3_sensitivity_only") is not True
        ):
            raise RuntimeError("D1 K scenario gate aggregate differs")
        verify_artifact_records_recursive(
            existing,
            name="D1 K scenario gate aggregate resume",
            require_at_least_one=True,
        )
        observed_path = verified_artifact_path(
            _mapping(
                _mapping(
                    existing.get("artifacts"), name="D1 K aggregate artifacts"
                ).get("comparison_table"),
                name="D1 K comparison table",
            ),
            name="D1 K comparison table",
        )
        _assert_frame(
            pd.read_csv(observed_path),
            expected_comparison,
            name="D1 K R0/ungated/R7 comparison",
        )
        return existing
    atomic_text(comparison_path, expected_comparison.to_csv(index=False))
    result["artifacts"] = {"comparison_table": artifact_record(comparison_path)}
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(path, result)
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    output = root / "11_k_sensitivity/scenario_gate_selection.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P11",
        substage="d1_k_scenario_r7_selection",
        route="D1",
        pool="top10_allnms",
        evidence_track="T2_matched_common",
        method="R0_R7",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(output)
        state["artifact_sha256"] = sha256_file(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
