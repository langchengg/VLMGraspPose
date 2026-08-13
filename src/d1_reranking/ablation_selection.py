"""P10 aggregation, track-specific R7 refits, and table production."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_reranking.artifacts import verified_artifact_path
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
from unified_reranking.hashing import atomic_json, canonical_sha256
from unified_reranking.metrics import evaluate_order_only
from unified_reranking.training import FORMAL_SEEDS

from .ablation import (
    EVIDENCE_TRACKS,
    load_ablation_plan,
    track_feature_artifact,
)
from .execution import artifact_record, load_content_manifest
from .gate_inputs import build_gate_input_frame
from .io import atomic_csv, atomic_parquet, atomic_pickle
from .selection import ensemble_seed_scores


SELECTION_RELATIVE = Path("12_feature_ablation/ablation_selection_manifest.json")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return {str(key): child for key, child in value.items()}


def gate_grid_points(grid: Mapping[str, Any]) -> tuple[GateOperatingPoint, ...]:
    configuration = _mapping(grid.get("configuration", grid), name="D1 P10 gate grid")
    points = tuple(
        GateOperatingPoint(*values)
        for values in itertools.product(
            configuration["lambda_harm"],
            configuration["utility_thresholds"],
            configuration["score_margin_thresholds"],
            configuration["reliability_thresholds"],
            configuration["stability_thresholds"],
        )
    )
    if (
        len(points) != 108
        or configuration.get("minimum_seed_votes") != 2
        or int(configuration.get("bootstrap_iterations", -1)) != BOOTSTRAP_ITERATIONS
        or int(configuration.get("bootstrap_seed", -1)) != BOOTSTRAP_SEED
    ):
        raise RuntimeError("D1 P10 gate grid differs from the 108-point contract")
    return points


def _trial_frame(selection: Any) -> pd.DataFrame:
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
            for trial in selection.trials
        ]
    )


def _label_frame(plan: Mapping[str, Any], split: str) -> pd.DataFrame:
    manifest_path = verified_artifact_path(
        plan["sources"]["development_label_manifests"][split],
        name=f"D1 P10 {split} labels",
    )
    manifest = load_content_manifest(
        manifest_path, name=f"D1 P10 {split} labels", statuses=("COMPLETE",)
    )
    return pd.read_parquet(
        verified_artifact_path(manifest["artifact"], name=f"D1 P10 {split} labels")
    )


def _denominator(plan: Mapping[str, Any], split: str) -> pd.DataFrame:
    path = verified_artifact_path(
        plan["sources"]["denominators"][split], name=f"D1 P10 {split} denominator"
    )
    columns = ["sample_id", "scene_id"]
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        frame = pd.read_parquet(path, columns=["sample_id"])
        frame["scene_id"] = frame["sample_id"].astype(str)
        return frame


def _gate_materials(
    plan: Mapping[str, Any], split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fixed matched-common inference evidence used by every R7 refit."""

    track = "T2_matched_common"
    feature_manifest_path = verified_artifact_path(
        plan["sources"]["track_manifests"][track][split],
        name=f"D1 P10 {split} gate feature manifest",
    )
    feature_manifest = load_content_manifest(
        feature_manifest_path,
        name=f"D1 P10 {split} gate features",
        statuses=("COMPLETE",),
    )
    candidate_manifest_path = verified_artifact_path(
        plan["sources"]["candidate_manifests"][split],
        name=f"D1 P10 {split} candidate manifest",
    )
    candidate_manifest = load_content_manifest(
        candidate_manifest_path,
        name=f"D1 P10 {split} candidates",
        statuses=("COMPLETE",),
    )
    features = pd.read_parquet(
        track_feature_artifact(
            feature_manifest,
            track=track,
            name=f"D1 P10 {split} fixed gate features",
        )
    )
    candidates = pd.read_parquet(
        verified_artifact_path(
            candidate_manifest["artifacts"]["top5"],
            name=f"D1 P10 {split} Top5 candidates",
        )
    )
    return features, candidates


def _cell_predictions(cell: Mapping[str, Any]) -> pd.DataFrame:
    return pd.read_parquet(
        verified_artifact_path(
            cell["artifacts"]["predictions"], name="D1 P10 predictions"
        )
    )


def aggregate_variant(
    *,
    cells: Sequence[Mapping[str, Any]],
    labels: pd.DataFrame,
    denominator: Sequence[str],
    split: str,
) -> dict[str, Any]:
    """Rebuild one three-seed OOF or Validation score ensemble."""

    mode = "oof" if split == "train_oof" else "validation"
    expected = len(FORMAL_SEEDS) * (5 if mode == "oof" else 1)
    selected = [cell for cell in cells if cell["configuration"]["mode"] == mode]
    if len(selected) != expected:
        raise RuntimeError(f"D1 P10 {split} cell inventory differs")
    seed_frames: dict[int, pd.DataFrame] = {}
    seed_metrics: dict[int, dict[str, Any]] = {}
    for seed in FORMAL_SEEDS:
        parts = [
            _cell_predictions(cell)
            for cell in selected
            if int(cell["configuration"]["seed"]) == int(seed)
        ]
        frame = pd.concat(parts, ignore_index=True)
        if frame.duplicated(["sample_id", "candidate_id"]).any():
            raise RuntimeError("D1 P10 per-seed candidate predictions overlap")
        evaluation = frame.merge(
            labels[["sample_id", "candidate_id", "candidate_success"]],
            on=["sample_id", "candidate_id"],
            validate="one_to_one",
        )
        if len(evaluation) != len(frame) or len(evaluation) != len(labels):
            raise RuntimeError("D1 P10 per-seed prediction/label membership differs")
        metrics, _decisions = evaluate_order_only(
            denominator, evaluation, score_column="score", max_k=5
        )
        seed_frames[int(seed)] = frame
        seed_metrics[int(seed)] = metrics
    ensemble = ensemble_seed_scores(seed_frames).rename(
        columns={"ensemble_score": "score"}
    )
    evaluation = ensemble.merge(
        labels[["sample_id", "candidate_id", "candidate_success"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    if len(evaluation) != len(ensemble) or len(evaluation) != len(labels):
        raise RuntimeError("D1 P10 ensemble/label membership differs")
    metrics, decisions = evaluate_order_only(
        denominator, evaluation, score_column="score", max_k=5
    )
    return {
        "predictions": ensemble,
        "decisions": decisions,
        "metrics": metrics,
        "seed_metrics": seed_metrics,
        "seed_predictions": seed_frames,
    }


def feature_ablation_rows(
    plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate per-seed Validation means/stds and deltas from full T4."""

    variants = list(plan["feature_ablation_variants"])
    available = [
        variant
        for variant in variants
        if variant["status"] == "AVAILABLE" and variant["kind"] == "cumulative"
    ]
    if not available:
        raise RuntimeError("D1 P10 has no available cumulative T4 reference")
    full = max(
        available,
        key=lambda row: (len(row["feature_columns"]), variants.index(row)),
    )
    by_variant: dict[str, dict[int, Mapping[str, Any]]] = {}
    for result in results.values():
        configuration = result["configuration"]
        if (
            configuration["analysis"] == "feature_family"
            and configuration["mode"] == "validation"
        ):
            by_variant.setdefault(str(configuration["variant_id"]), {})[
                int(configuration["seed"])
            ] = result["metrics"]
    full_seed = by_variant.get(str(full["variant_id"]), {})
    if set(full_seed) != set(FORMAL_SEEDS):
        raise RuntimeError("D1 P10 full-T4 seed metrics are incomplete")
    rows: list[dict[str, Any]] = []
    for variant in variants:
        base = {
            "variant_id": variant["variant_id"],
            "kind": variant["kind"],
            "target_family": variant["target_family"],
            "status": variant["status"],
            "feature_count": len(variant["feature_columns"]),
            "reference_variant_id": full["variant_id"],
            "reason": variant["reason"],
        }
        if variant["status"] != "AVAILABLE":
            rows.append(
                {
                    **base,
                    "validation_j_at_1_mean": None,
                    "validation_j_at_1_std": None,
                    "delta_vs_full_t4_mean": None,
                    "delta_vs_full_t4_std": None,
                    "per_seed_json": None,
                }
            )
            continue
        seed_metrics = by_variant.get(str(variant["variant_id"]), {})
        if set(seed_metrics) != set(FORMAL_SEEDS):
            raise RuntimeError(f"D1 P10 {variant['variant_id']} seed metrics differ")
        values = np.asarray(
            [float(seed_metrics[int(seed)]["j_at_1"]) for seed in FORMAL_SEEDS]
        )
        full_values = np.asarray(
            [float(full_seed[int(seed)]["j_at_1"]) for seed in FORMAL_SEEDS]
        )
        deltas = values - full_values
        rows.append(
            {
                **base,
                "validation_j_at_1_mean": float(values.mean()),
                "validation_j_at_1_std": float(values.std(ddof=0)),
                "delta_vs_full_t4_mean": float(deltas.mean()),
                "delta_vs_full_t4_std": float(deltas.std(ddof=0)),
                "per_seed_json": json.dumps(
                    {
                        str(seed): {
                            "j_at_1": float(seed_metrics[int(seed)]["j_at_1"]),
                            "delta_vs_full_t4": float(deltas[index]),
                        }
                        for index, seed in enumerate(FORMAL_SEEDS)
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def _gate_inputs(
    *,
    paired: pd.DataFrame,
    native_predictions: pd.DataFrame,
    native_decisions: pd.DataFrame,
    challenger: Mapping[str, Any],
    candidate_features: pd.DataFrame,
    candidates: pd.DataFrame,
    prediction_source: str,
    folds: pd.DataFrame | None,
) -> pd.DataFrame:
    ensemble = challenger["predictions"].rename(columns={"score": "ensemble_score"})
    return build_gate_input_frame(
        paired=paired,
        native_predictions=native_predictions,
        native_decisions=native_decisions,
        challenger_predictions=ensemble,
        challenger_decisions=challenger["decisions"],
        candidate_features=candidate_features,
        candidates=candidates,
        prediction_source=prediction_source,
        folds=folds,
    )


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


def compute_track_gate(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    points: Sequence[GateOperatingPoint],
) -> dict[str, Any]:
    features = tuple(SAFE_GATE_FEATURE_COLUMNS)
    model = ConservativeTransitionModel(seed=BOOTSTRAP_SEED).fit(
        OOFTransitionData(
            features=train.loc[:, features].to_numpy(float),
            feature_names=features,
            native_correct=train["native_correct"].to_numpy(bool),
            challenger_correct=train["challenger_correct"].to_numpy(bool),
            scene_ids=train["scene_id"].astype(str).to_numpy(),
            oof_fold_ids=train["oof_fold"].to_numpy(int),
            prediction_source="train_oof",
        )
    )
    recover, harm = model.predict_probabilities(
        validation.loc[:, features].to_numpy(float)
    )
    selection = select_gate_operating_point(
        recover,
        harm,
        _evidence(validation),
        validation["native_correct"].to_numpy(bool),
        validation["challenger_correct"].to_numpy(bool),
        validation["scene_id"].astype(str).to_numpy(),
        points,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    if selection.selected_operating_point is None:
        switches = np.zeros(len(validation), dtype=bool)
        utility = recover - harm
    else:
        switches = gate_switch_mask(
            recover, harm, _evidence(validation), selection.selected_operating_point
        )
        utility = recover - selection.selected_operating_point.lambda_harm * harm
    native = validation["native_correct"].to_numpy(bool)
    challenger_correct = validation["challenger_correct"].to_numpy(bool)
    selected_correct = np.where(switches, challenger_correct, native)
    decisions = pd.DataFrame(
        {
            "sample_id": validation["sample_id"].astype(str),
            "scene_id": validation["scene_id"].astype(str),
            "probability_recover": recover,
            "probability_harm": harm,
            "utility": utility,
            "switch": switches,
            "native_candidate_id": validation["native_candidate_id"].astype(str),
            "challenger_candidate_id": validation["challenger_candidate_id"].astype(
                str
            ),
            "selected_candidate_id": np.where(
                switches,
                validation["challenger_candidate_id"].astype(str),
                validation["native_candidate_id"].astype(str),
            ),
            "native_correct": native,
            "challenger_correct": challenger_correct,
            "selected_correct": selected_correct,
        }
    )
    metrics = {
        "sample_count": len(validation),
        "native_j_at_1": float(native.mean()),
        "ungated_j_at_1": float(challenger_correct.mean()),
        "gated_j_at_1": float(selected_correct.mean()),
        "gated_delta_vs_native": float(selected_correct.mean() - native.mean()),
        "gated_delta_vs_ungated": float(
            selected_correct.mean() - challenger_correct.mean()
        ),
        "switch_count": int(switches.sum()),
        "switch_rate": float(switches.mean()),
    }
    return {
        "model": model,
        "selection_object": selection,
        "selection": selection.artifact(),
        "decision": selection.status,
        "decisions": decisions,
        "trials": _trial_frame(selection),
        "metrics": metrics,
    }


def fit_track_gate(
    *,
    track: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    points: Sequence[GateOperatingPoint],
    output_dir: Path,
    sources: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = tuple(SAFE_GATE_FEATURE_COLUMNS)
    replay = compute_track_gate(train=train, validation=validation, points=points)
    model = replay["model"]
    selection = replay["selection_object"]
    decisions = replay["decisions"]
    metrics = replay["metrics"]
    artifacts = {
        "train_gate_inputs": artifact_record(
            atomic_parquet(train, output_dir / "train_oof_gate_inputs.parquet")
        ),
        "validation_gate_inputs": artifact_record(
            atomic_parquet(validation, output_dir / "validation_gate_inputs.parquet")
        ),
        "transition_model": artifact_record(
            atomic_pickle(model, output_dir / "transition_model.pkl")
        ),
        "validation_decisions": artifact_record(
            atomic_parquet(decisions, output_dir / "validation_decisions.parquet")
        ),
        "validation_trials": artifact_record(
            atomic_parquet(replay["trials"], output_dir / "validation_trials.parquet")
        ),
    }
    configuration = {
        "route": "D1",
        "pool": "top5",
        "track": track,
        "method": "R7",
        "feature_columns": list(features),
        "operating_point_count": 108,
        "model_seed": BOOTSTRAP_SEED,
        "candidate_test_labels_read": False,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": configuration,
        "source_signature_sha256": canonical_sha256(
            {"configuration": configuration, "sources": sources}
        ),
        "selection": selection.artifact(),
        "decision": selection.status,
        "transition_model": model.artifact(),
        "metrics": metrics,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "sources": dict(sources),
        "artifacts": artifacts,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(output_dir / "gate_selection.json", manifest)
    return manifest, decisions


def evidence_rows(
    *,
    plan: Mapping[str, Any],
    result_values: Mapping[str, Mapping[str, Any]],
    run_dir: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r0_path = verified_artifact_path(plan["sources"]["r0_r1_selection"], name="D1 R0")
    r0 = load_content_manifest(r0_path, name="D1 R0", statuses=("COMPLETE",))
    primary_path = verified_artifact_path(
        plan["sources"]["selected_primary"], name="primary"
    )
    primary = load_content_manifest(
        primary_path, name="D1 primary", statuses=("COMPLETE",)
    )
    primary_gate_path = verified_artifact_path(
        plan["sources"]["primary_gate"], name="primary gate"
    )
    primary_gate = load_content_manifest(
        primary_gate_path, name="D1 primary gate", statuses=("COMPLETE",)
    )
    rows: list[dict[str, Any]] = []
    track_artifacts: dict[str, Any] = {
        "T2_matched_common": {
            "status": "REUSED_PRIMARY_EXACT",
            "primary": artifact_record(primary_path),
            "gate": artifact_record(primary_gate_path),
        }
    }
    t2_metrics = {
        "R0": r0["r0_validation_metrics"],
        "selected_ungated": primary["validation_metrics"],
        "R7": {
            "j_at_1": primary_gate["metrics"]["gated_j_at_1"],
            "sample_count": primary_gate["metrics"]["sample_count"],
        },
    }
    for method in ("R0", "selected_ungated", "R7"):
        metric = t2_metrics[method]
        rows.append(
            {
                "track": "T2_matched_common",
                "method": method,
                "status": "REUSED_PRIMARY_EXACT",
                "sample_count": metric["sample_count"],
                "j_at_1": metric["j_at_1"],
                "delta_vs_r0": float(metric["j_at_1"] - t2_metrics["R0"]["j_at_1"]),
                "gate_decision": primary_gate["decision"] if method == "R7" else None,
            }
        )
    labels = {split: _label_frame(plan, split) for split in ("train", "validation")}
    paired = {split: _denominator(plan, split) for split in ("train", "validation")}
    gate_materials = {
        split: _gate_materials(plan, split) for split in ("train", "validation")
    }
    folds_path = verified_artifact_path(
        plan["sources"]["fold_assignments"], name="folds"
    )
    folds = pd.read_parquet(folds_path, columns=["sample_id", "fold"])
    grid_path = verified_artifact_path(plan["sources"]["gate_grid"], name="gate grid")
    grid = load_content_manifest(grid_path, name="gate grid", statuses=("PLANNED",))
    points = gate_grid_points(grid)
    native = {
        "train_predictions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_train_predictions"], name="R0 Train"
            )
        ),
        "train_decisions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_train_decisions"], name="R0 Train decisions"
            )
        ),
        "validation_predictions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_validation_predictions"], name="R0 Validation"
            )
        ),
        "validation_decisions": pd.read_parquet(
            verified_artifact_path(
                r0["artifacts"]["r0_validation_decisions"],
                name="R0 Validation decisions",
            )
        ),
    }
    for track in EVIDENCE_TRACKS:
        if track == "T2_matched_common":
            continue
        track_cells = [
            value
            for value in result_values.values()
            if value["configuration"]["analysis"] == "evidence_track"
            and value["configuration"]["variant_id"] == track
        ]
        if plan["evidence_tracks"][track]["status"] != "AVAILABLE":
            for method in ("R0", "selected_ungated", "R7"):
                rows.append(
                    {
                        "track": track,
                        "method": method,
                        "status": "NOT_AVAILABLE",
                        "sample_count": None,
                        "j_at_1": None,
                        "delta_vs_r0": None,
                        "gate_decision": None,
                    }
                )
            track_artifacts[track] = {"status": "NOT_AVAILABLE"}
            continue
        train = aggregate_variant(
            cells=track_cells,
            labels=labels["train"],
            denominator=paired["train"]["sample_id"].astype(str).tolist(),
            split="train_oof",
        )
        validation = aggregate_variant(
            cells=track_cells,
            labels=labels["validation"],
            denominator=paired["validation"]["sample_id"].astype(str).tolist(),
            split="validation",
        )
        ensemble_dir = output_root / "ensembles" / track
        ensemble_artifacts = {}
        for split, value in (("train_oof", train), ("validation", validation)):
            ensemble_artifacts[split] = {
                "predictions": artifact_record(
                    atomic_parquet(
                        value["predictions"],
                        ensemble_dir / split / "predictions.parquet",
                    )
                ),
                "decisions": artifact_record(
                    atomic_parquet(
                        value["decisions"], ensemble_dir / split / "decisions.parquet"
                    )
                ),
                "metrics": value["metrics"],
            }
        gate_train = _gate_inputs(
            paired=paired["train"],
            native_predictions=native["train_predictions"],
            native_decisions=native["train_decisions"],
            challenger=train,
            candidate_features=gate_materials["train"][0],
            candidates=gate_materials["train"][1],
            prediction_source="train_oof",
            folds=folds,
        )
        gate_validation = _gate_inputs(
            paired=paired["validation"],
            native_predictions=native["validation_predictions"],
            native_decisions=native["validation_decisions"],
            challenger=validation,
            candidate_features=gate_materials["validation"][0],
            candidates=gate_materials["validation"][1],
            prediction_source="validation",
            folds=None,
        )
        gate_sources = {
            "plan": artifact_record(run_dir / "configs/d1_ablation_plan_v1.json"),
            "r0": artifact_record(r0_path),
            "gate_grid": artifact_record(grid_path),
            "cells": [
                artifact_record(
                    Path(cell["artifacts"]["model"]["path"]).parent / "manifest.json"
                )
                for cell in track_cells
            ],
        }
        gate, _decisions = fit_track_gate(
            track=track,
            train=gate_train,
            validation=gate_validation,
            points=points,
            output_dir=output_root / "gates" / track,
            sources=gate_sources,
        )
        baseline_j = float(r0["r0_validation_metrics"]["j_at_1"])
        values = {
            "R0": baseline_j,
            "selected_ungated": float(validation["metrics"]["j_at_1"]),
            "R7": float(gate["metrics"]["gated_j_at_1"]),
        }
        for method, score in values.items():
            rows.append(
                {
                    "track": track,
                    "method": method,
                    "status": "COMPLETE",
                    "sample_count": len(paired["validation"]),
                    "j_at_1": score,
                    "delta_vs_r0": score - baseline_j,
                    "gate_decision": gate["decision"] if method == "R7" else None,
                }
            )
        track_artifacts[track] = {
            "status": "COMPLETE",
            "ensembles": ensemble_artifacts,
            "gate": artifact_record(
                output_root / "gates" / track / "gate_selection.json"
            ),
        }
    return rows, track_artifacts


def write_ablation_selection(
    run_dir: str | Path,
    *,
    execution_records: Mapping[str, Any],
    result_values: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    plan_path = root / "configs/d1_ablation_plan_v1.json"
    plan = load_ablation_plan(plan_path)
    output = root / "12_feature_ablation"
    manifest_path = root / SELECTION_RELATIVE
    sources = {
        "plan": artifact_record(plan_path),
        "execution": dict(execution_records),
        "selector": artifact_record(
            Path(__file__).resolve().parents[2]
            / "tools/d1_reranking/select_ablation.py"
        ),
        "cells": {
            job_id: artifact_record(
                root
                / next(
                    str(job["output_manifest"])
                    for job in plan["jobs"]
                    if job["job_id"] == job_id
                )
            )
            for job_id in sorted(result_values)
        },
    }
    signature = canonical_sha256(sources)
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name="D1 P10 selection", statuses=("COMPLETE",)
        )
        if resume and existing.get("source_signature_sha256") == signature:
            return existing
        raise RuntimeError("D1 P10 selection exists and differs")
    feature_rows = feature_ablation_rows(plan, result_values)
    evidence_table_rows, track_artifacts = evidence_rows(
        plan=plan,
        result_values=result_values,
        run_dir=root,
        output_root=root / "07_validation/evidence_ablation",
    )
    evidence_path = atomic_csv(
        pd.DataFrame(evidence_table_rows),
        root / "07_validation/tables/evidence_track_ablation.csv",
    )
    feature_path = atomic_csv(
        pd.DataFrame(feature_rows), output / "feature_ablation.csv"
    )
    artifacts = {
        "evidence_track_table": artifact_record(evidence_path),
        "feature_ablation_table": artifact_record(feature_path),
        "track_artifacts": track_artifacts,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_signature_sha256": signature,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "selected_primary": plan["selected_primary"],
        "retuning_permitted": False,
        "sources": sources,
        "artifacts": artifacts,
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


__all__ = [
    "SELECTION_RELATIVE",
    "aggregate_variant",
    "compute_track_gate",
    "evidence_rows",
    "feature_ablation_rows",
    "fit_track_gate",
    "gate_grid_points",
    "write_ablation_selection",
]
