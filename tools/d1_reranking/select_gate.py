"""Fit the D1 OOF transition model and lock its Validation R7 operating point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.io import atomic_parquet, atomic_pickle  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _grid(value: dict[str, object]) -> tuple[GateOperatingPoint, ...]:
    if (
        value.get("minimum_seed_votes") != 2
        or value.get("bootstrap_iterations") != BOOTSTRAP_ITERATIONS
        or value.get("bootstrap_seed") != BOOTSTRAP_SEED
    ):
        raise RuntimeError("D1 gate grid bootstrap/consensus contract differs")
    points = tuple(
        GateOperatingPoint(lam, utility, margin, reliability, stability)
        for lam in value["lambda_harm"]  # type: ignore[union-attr]
        for utility in value["utility_thresholds"]  # type: ignore[union-attr]
        for margin in value["score_margin_thresholds"]  # type: ignore[union-attr]
        for reliability in value["reliability_thresholds"]  # type: ignore[union-attr]
        for stability in value["stability_thresholds"]  # type: ignore[union-attr]
    )
    if len(points) != 108:
        raise RuntimeError("D1 gate grid must contain exactly 108 operating points")
    return points


def _evidence(frame: pd.DataFrame) -> GateEvidence:
    return GateEvidence(
        score_margin=frame["score_margin"].to_numpy(),
        challenger_reliability=frame["challenger_reliability"].to_numpy(),
        perturbation_stability=frame["perturbation_stability"].to_numpy(),
        seed_challenger_votes=frame["seed_challenger_votes"].to_numpy(),
        candidate_id_unchanged=frame["candidate_id_unchanged"].to_numpy(),
        geometry_hash_unchanged=frame["geometry_hash_unchanged"].to_numpy(),
        challenger_exists=frame["challenger_exists"].to_numpy(),
    )


def run(run_dir: Path, *, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    input_manifest_path = (
        root / "07_validation" / "gate_inputs" / "d1" / "manifest.json"
    )
    grid_path = root / "configs" / "d1_gate_grid.json"
    inputs = load_content_manifest(
        input_manifest_path, name="D1 gate inputs", statuses=("COMPLETE",)
    )
    grid = load_content_manifest(grid_path, name="D1 gate grid", statuses=("PLANNED",))
    if inputs.get("configuration", {}).get("feature_columns") != list(
        SAFE_GATE_FEATURE_COLUMNS
    ):
        raise RuntimeError("D1 gate inputs do not expose the exact safe schema")
    verify_artifact_records_recursive(
        {"inputs": inputs, "grid_sources": grid.get("sources")},
        name="D1 gate selection sources",
        require_at_least_one=True,
    )
    train_path = verified_artifact_path(
        inputs.get("artifacts", {}).get("train_oof", {}), name="D1 gate Train OOF"
    )
    validation_path = verified_artifact_path(
        inputs.get("artifacts", {}).get("validation", {}), name="D1 gate Validation"
    )
    sources = {
        "input_manifest": artifact_record(input_manifest_path),
        "train_oof": artifact_record(train_path),
        "validation": artifact_record(validation_path),
        "grid": artifact_record(grid_path),
        "gate_primitive": artifact_record(ROOT / "src/unified_reranking/gate.py"),
        "tool": artifact_record(Path(__file__)),
    }
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "method": "R7",
        "feature_columns": list(SAFE_GATE_FEATURE_COLUMNS),
        "model_seed": BOOTSTRAP_SEED,
        "operating_point_count": 108,
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = root / "07_validation" / "gate" / "d1"
    manifest_path = output / "gate_selection.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name="D1 gate selection", statuses=("COMPLETE",)
        )
        if resume and existing.get("source_signature_sha256") == signature:
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name="D1 gate selection resume",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("D1 gate selection differs or is corrupt")
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(validation_path)
    features = tuple(SAFE_GATE_FEATURE_COLUMNS)
    model = ConservativeTransitionModel(seed=BOOTSTRAP_SEED).fit(
        OOFTransitionData(
            features=train.loc[:, features].to_numpy(float),
            feature_names=features,
            native_correct=train["native_correct"].to_numpy(),
            challenger_correct=train["challenger_correct"].to_numpy(),
            scene_ids=train["scene_id"].to_numpy(),
            oof_fold_ids=train["oof_fold"].to_numpy(),
            prediction_source="train_oof",
        )
    )
    probability_recover, probability_harm = model.predict_probabilities(
        validation.loc[:, features].to_numpy(float)
    )
    points = _grid(grid)
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
    if selection.selected_operating_point is None:
        switches = np.zeros(len(validation), dtype=bool)
        utility = probability_recover - probability_harm
    else:
        switches = gate_switch_mask(
            probability_recover,
            probability_harm,
            _evidence(validation),
            selection.selected_operating_point,
        )
        utility = (
            probability_recover
            - selection.selected_operating_point.lambda_harm * probability_harm
        )
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
    recovered = int((~native & challenger).sum())
    harmful = int((native & ~challenger).sum())
    metrics = {
        "sample_count": len(validation),
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
            if int((~native & challenger & switches).sum())
            + int((native & ~challenger & switches).sum())
            == 0
            else int((~native & challenger & switches).sum())
            / (
                int((~native & challenger & switches).sum())
                + int((native & ~challenger & switches).sum())
            )
        ),
    }
    artifacts = {
        "transition_model": artifact_record(
            atomic_pickle(model, output / "transition_model.pkl")
        ),
        "validation_decisions": artifact_record(
            atomic_parquet(decisions, output / "validation_decisions.parquet")
        ),
        "validation_trials": artifact_record(
            atomic_parquet(trials, output / "validation_trials.parquet")
        ),
    }
    metrics_path = root / "07_validation" / "tables" / "gate_operating_point.csv"
    atomic_text(metrics_path, pd.DataFrame([metrics]).to_csv(index=False))
    artifacts["gate_operating_point_table"] = artifact_record(metrics_path)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "decision": selection.status,
        "source_signature_sha256": signature,
        "configuration": configuration,
        "transition_model": model.artifact(),
        "selection": selection.artifact(),
        "validation_metrics": metrics,
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    path = root / "07_validation" / "gate" / "d1" / "gate_selection.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P11",
        substage="d1_select_expected_gain_gate",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R7",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
