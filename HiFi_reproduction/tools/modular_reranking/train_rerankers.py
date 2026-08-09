#!/usr/bin/env python3
"""Train, calibrate, and validation-lock leakage-safe candidate rerankers.

The command consumes mutually disjoint development, calibration, and official
validation tables. Development supplies gradients and scene-grouped OOF
predictions; calibration supplies final-deployment early stopping; official
validation locks rule weights, safe-switch thresholds, and final method
selection. It never consumes the formal test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.models import (  # noqa: E402
    DEFAULT_FEATURE_COLUMNS,
    GeometryGateConfig,
    NeuralRankerConfig,
    RegularizedLinearRanker,
    TorchCandidateRanker,
    assert_feature_identity_invariant,
    attach_scores,
    geometry_gated_q_scores,
    geometry_risk,
    q_only_scores,
    q_softmask_rule_scores,
    save_json,
    scene_grouped_oof,
    top1_accuracy,
    validate_candidate_contract,
)
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    build_final_scaler_metadata,
    identity_payload,
    validate_artifact_identity,
    validate_public_methods,
)
from src.grasping.reranking_v1.method_namespace import (  # noqa: E402
    public_method_name,
    public_method_names,
)
from src.grasping.reranking_v1.safe_switch import (  # noqa: E402
    SafeSwitchGate,
    apply_safe_switch,
    build_switch_examples,
    candidate_predictions_from_switch,
    threshold_sweep,
)


LEARNED_METHODS = (
    "regularized_linear_ranker",
    "pairwise_ranker",
    "multi_positive_listwise_ranker",
    "residual_mlp",
    "set_aware_residual",
)
PREDICTION_COLUMNS = (
    "sample_id",
    "scene_id",
    "candidate_id",
    "candidate_identity_sha256",
    "original_gqcnn_rank",
    "candidate_positive",
    "reranker_method",
    "reranker_score",
    "reranker_rank",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_feature_columns(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return tuple(DEFAULT_FEATURE_COLUMNS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get(
            "feature_columns",
            payload.get("inference_feature_allowlist", payload.get("features")),
        )
    else:
        values = None
    if not isinstance(values, list) or not values:
        raise ValueError("feature-columns JSON must contain a non-empty list")
    # Constructors validate the authoritative features.INFERENCE_FEATURE_ALLOWLIST.
    return tuple(map(str, values))


def _validate_input(
    frame: pd.DataFrame, *, allowed_splits: set[str], name: str
) -> None:
    del name
    validate_candidate_contract(
        frame, require_label=True, allowed_splits=allowed_splits
    )
    required = {"q_rank_normalized", "p_axis_mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate frame missing required columns: {missing}")


def _compact_predictions(
    scored: pd.DataFrame,
    *,
    protocol: str,
    oof_fold: bool = False,
    public_names: bool = True,
) -> pd.DataFrame:
    columns = [column for column in PREDICTION_COLUMNS if column in scored.columns]
    if oof_fold and "oof_fold" in scored.columns:
        columns.append("oof_fold")
    result = scored.loc[:, columns].copy()
    if public_names:
        result["reranker_method"] = result["reranker_method"].map(
            public_method_name
        )
    result["protocol"] = protocol
    return result


def _direct_scores(
    frame: pd.DataFrame, *, softmask_alpha: float = 0.8
) -> dict[str, pd.DataFrame]:
    risk, _ = geometry_risk(frame)
    base = frame.copy()
    base["geometry_risk"] = risk
    return {
        "q_only": attach_scores(base, q_only_scores(base), method="q_only"),
        "q_softmask_rule": attach_scores(
            base,
            q_softmask_rule_scores(
                base, alpha=softmask_alpha, beta=1.0 - softmask_alpha
            ),
            method="q_softmask_rule",
        ),
        "geometry_gated_q": attach_scores(
            base,
            geometry_gated_q_scores(base),
            method="geometry_gated_q",
        ),
    }


def _protocol_predictions(
    scored: pd.DataFrame, *, protocol: str, method_alias: str | None = None
) -> pd.DataFrame:
    if protocol == "full_nms":
        pool = scored.copy()
    elif protocol == "gqcnn_top5":
        pool = scored.loc[scored["original_gqcnn_rank"] <= 5].copy()
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    reranked = attach_scores(
        pool,
        pool["reranker_score"].to_numpy(float),
        method=method_alias or str(pool["reranker_method"].iloc[0]),
    )
    return _compact_predictions(reranked, protocol=protocol)


def _fit_model(
    method: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: Sequence[str],
    config: NeuralRankerConfig,
) -> RegularizedLinearRanker | TorchCandidateRanker:
    if method == "regularized_linear_ranker":
        model: RegularizedLinearRanker | TorchCandidateRanker = (
            RegularizedLinearRanker(features, seed=config.seed)
        )
    else:
        model = TorchCandidateRanker(method, features, config=config)
    model.fit(train, validation=validation)
    return model


def _validate_sample_universe(
    candidates: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    name: str,
    allowed_splits: set[str],
) -> int:
    required = {"sample_id", "scene_id", "split", "candidate_count"}
    if missing := sorted(required - set(universe.columns)):
        raise ValueError(f"{name} per-sample table missing columns: {missing}")
    normalized = universe.copy()
    normalized["sample_id"] = normalized["sample_id"].astype(str)
    normalized["scene_id"] = normalized["scene_id"].astype(str)
    if normalized["sample_id"].duplicated().any():
        raise ValueError(f"{name} per-sample table has duplicate sample IDs")
    if set(normalized["split"].astype(str)) - allowed_splits:
        raise ValueError(f"{name} per-sample table contains another split")
    candidate_sample_ids = set(candidates["sample_id"].astype(str))
    expected_nonempty_ids = set(
        normalized.loc[
            normalized["candidate_count"].astype(int) > 0, "sample_id"
        ].astype(str)
    )
    if candidate_sample_ids != expected_nonempty_ids:
        raise ValueError(
            f"{name} candidate samples disagree with the per-sample universe"
        )
    candidate_counts = (
        candidates.assign(sample_id=candidates["sample_id"].astype(str))
        .groupby("sample_id", sort=False)
        .size()
        .rename("observed_candidate_count")
    )
    universe_counts = normalized.set_index("sample_id")[
        "candidate_count"
    ].astype(int)
    if not candidate_counts.astype(int).equals(
        universe_counts.loc[candidate_counts.index].astype(int)
    ):
        raise ValueError(f"{name} candidate counts disagree with universe")
    candidate_scenes = (
        candidates.assign(sample_id=candidates["sample_id"].astype(str))
        .groupby("sample_id", sort=False)["scene_id"]
        .agg(lambda values: set(map(str, values)))
    )
    universe_scenes = normalized.set_index("sample_id")["scene_id"]
    if any(
        scenes != {universe_scenes.loc[sample_id]}
        for sample_id, scenes in candidate_scenes.items()
    ):
        raise ValueError(f"{name} scene IDs disagree with universe")
    return int(len(normalized))


def run(args: argparse.Namespace) -> dict[str, Any]:
    development_path = args.development_per_candidate.resolve()
    calibration_path = args.calibration_per_candidate.resolve()
    validation_path = args.validation_per_candidate.resolve()
    output_root = args.output_root.resolve()
    rule_selection_path = args.rule_selection.expanduser().resolve()
    rule_selection = json.loads(rule_selection_path.read_text(encoding="utf-8"))
    validate_artifact_identity(
        rule_selection, context="rule hyperparameter selection"
    )
    if (
        rule_selection.get("selection_split") != "validation"
        or rule_selection.get("method") != "q_softmask_rule"
        or Path(
            str(rule_selection.get("validation_per_candidate", ""))
        ).resolve()
        != validation_path
        or rule_selection.get("validation_per_candidate_sha256")
        != _sha256(validation_path)
    ):
        raise ValueError(
            "rule selection is not bound to this official validation input"
        )
    sweep_path = Path(str(rule_selection.get("sweep", ""))).resolve()
    if (
        not sweep_path.is_file()
        or rule_selection.get("sweep_sha256") != _sha256(sweep_path)
    ):
        raise ValueError("rule-selection sweep artifact changed")
    softmask_alpha = float(rule_selection["selected"]["alpha"])
    if args.softmask_alpha is not None and float(args.softmask_alpha) != softmask_alpha:
        raise ValueError("--softmask-alpha disagrees with --rule-selection")
    development = pd.read_parquet(development_path)
    calibration = pd.read_parquet(calibration_path)
    validation = pd.read_parquet(validation_path)
    calibration_per_sample_path = args.calibration_per_sample.resolve()
    validation_per_sample_path = args.validation_per_sample.resolve()
    calibration_universe = pd.read_parquet(calibration_per_sample_path)
    validation_universe = pd.read_parquet(validation_per_sample_path)
    calibration_all_sample_count = _validate_sample_universe(
        calibration,
        calibration_universe,
        name="calibration",
        allowed_splits={"calibration"},
    )
    validation_all_sample_count = _validate_sample_universe(
        validation,
        validation_universe,
        name="validation",
        allowed_splits={"val", "validation"},
    )
    harmful_rate_denominator = "all_validation_samples"
    _validate_input(
        development,
        allowed_splits={"train", "development"},
        name="development",
    )
    _validate_input(
        calibration, allowed_splits={"calibration"}, name="calibration"
    )
    _validate_input(
        validation, allowed_splits={"validation", "val"}, name="validation"
    )
    partitions = {
        "development": development,
        "calibration": calibration,
        "validation": validation,
    }
    for left_name, left in partitions.items():
        for right_name, right in partitions.items():
            if left_name >= right_name:
                continue
            for column in (
                "sample_id",
                "scene_id",
                "candidate_identity_sha256",
            ):
                overlap = set(left[column].astype(str)) & set(
                    right[column].astype(str)
                )
                if overlap:
                    raise ValueError(
                        f"{left_name}/{right_name} leakage in {column}: "
                        f"{sorted(overlap)[:5]}"
                    )
    features = _load_feature_columns(args.feature_columns)
    # Force validation before any model or scaler sees data.
    from src.grasping.reranking_v1.features import validate_inference_allowlist

    features = validate_inference_allowlist(features)
    missing_dev = sorted(set(features) - set(development.columns))
    missing_cal = sorted(set(features) - set(calibration.columns))
    missing_val = sorted(set(features) - set(validation.columns))
    if missing_dev or missing_cal or missing_val:
        raise ValueError(
            f"selected feature columns absent: development={missing_dev}, "
            f"calibration={missing_cal}, validation={missing_val}"
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoints").mkdir()
    (output_root / "models").mkdir()

    neural_config = NeuralRankerConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        residual_bound=args.residual_bound,
        q_alpha=args.q_alpha,
        sample_batch_size=args.sample_batch_size,
        hard_negative_limit=args.hard_negative_limit,
        device=args.device,
        seed=args.seed,
    )
    save_json(
        output_root / "hyperparameters.json",
        {
            "seed": args.seed,
            "scene_grouped_oof_folds": args.oof_folds,
            "softmask_rule": {
                "alpha": softmask_alpha,
                "beta": 1.0 - softmask_alpha,
                "selection_split": "validation",
                "selection_path": str(rule_selection_path),
                "selection_sha256": _sha256(rule_selection_path),
            },
            "geometry_gate": asdict(GeometryGateConfig()),
            "neural": asdict(neural_config),
            "safe_switch_harmful_rate_limit": args.harmful_rate_limit,
        },
    )
    save_json(
        output_root / "feature_columns.json",
        {
            "feature_columns": list(features),
            "source": (
                str(args.feature_columns.resolve())
                if args.feature_columns is not None
                else "models.DEFAULT_FEATURE_COLUMNS"
            ),
            "GT_columns_permitted": False,
        },
    )

    # Direct, non-learned comparators.
    validation_scored: dict[str, pd.DataFrame] = _direct_scores(
        validation, softmask_alpha=softmask_alpha
    )
    # Training code uses short implementation keys internally.  Every
    # externally persisted prediction/metric uses the public repeated-FiLM
    # namespace so a downstream consumer cannot silently mix lineages.
    oof_parts_internal: list[pd.DataFrame] = []
    fold_metadata: dict[str, Any] = {}
    fitted: dict[str, RegularizedLinearRanker | TorchCandidateRanker] = {}
    deployment_model_paths: dict[str, Path] = {}

    for method in LEARNED_METHODS:
        def factory(
            method_name: str = method,
        ) -> RegularizedLinearRanker | TorchCandidateRanker:
            if method_name == "regularized_linear_ranker":
                return RegularizedLinearRanker(features, seed=args.seed)
            return TorchCandidateRanker(
                method_name, features, config=neural_config
            )

        oof, folds = scene_grouped_oof(
            development, factory, n_splits=args.oof_folds
        )
        assert_feature_identity_invariant(development, oof)
        oof_parts_internal.append(
            _compact_predictions(
                oof,
                protocol="full_nms",
                oof_fold=True,
                public_names=False,
            )
        )
        fold_metadata[method] = folds

        model = _fit_model(
            method, development, calibration, features, neural_config
        )
        fitted[method] = model
        scored = attach_scores(
            validation, model.predict_scores(validation), method=method
        )
        assert_feature_identity_invariant(validation, scored)
        validation_scored[method] = scored
        save_json(output_root / "models" / f"{method}.json", model.artifact())
        if isinstance(model, TorchCandidateRanker):
            model_path = output_root / "checkpoints" / f"{method}.pt"
            model.save(model_path)
        else:
            model_path = (
                output_root
                / "checkpoints"
                / "regularized_linear_ranker.json"
            )
            model.save(model_path)
            model.coefficients().to_csv(
                output_root / "coefficients.csv", index=False
            )
        deployment_model_paths[method] = model_path

    scaler_metadata_path = output_root / "scaler_metadata.json"
    scaler_metadata = build_final_scaler_metadata(
        {
            method: fitted[method].artifact()
            for method in LEARNED_METHODS
        },
        deployment_model_paths,
    )
    save_json(scaler_metadata_path, scaler_metadata)
    scaler_metadata_sha256 = _sha256(scaler_metadata_path)

    oof_predictions_internal = pd.concat(
        oof_parts_internal, ignore_index=True
    )
    oof_predictions_public = oof_predictions_internal.copy()
    oof_predictions_public["reranker_method"] = (
        oof_predictions_public["reranker_method"].map(public_method_name)
    )
    validate_public_methods(
        sorted(oof_predictions_public["reranker_method"].astype(str).unique()),
        context="persisted OOF predictions",
    )
    oof_predictions_public.to_parquet(
        output_root / "oof_predictions.parquet", index=False
    )
    save_json(
        output_root / "oof_fold_models.json",
        {
            "registry_key_namespace": "internal_implementation_keys",
            "models_by_internal_method": fold_metadata,
        },
    )

    # Save both the full-NMS and frozen-GQ-CNN-Top-5 protocols.  Neural
    # set-aware scores are recomputed on Top-5 because its set context changes.
    validation_parts: list[pd.DataFrame] = []
    protocol_b_aliases = {
        "q_only": "q_top5",
        "residual_mlp": "tabular_residual_top5",
        "set_aware_residual": "setrank_top5",
    }
    for method, scored in validation_scored.items():
        validation_parts.append(
            _protocol_predictions(scored, protocol="full_nms")
        )
        if method not in protocol_b_aliases:
            continue
        if method in fitted:
            top5 = validation.loc[
                validation["original_gqcnn_rank"] <= 5
            ].copy()
            top5_scored = attach_scores(
                top5, fitted[method].predict_scores(top5), method=method
            )
        else:
            top5_scored = scored.loc[
                scored["original_gqcnn_rank"] <= 5
            ].copy()
        validation_parts.append(
            _protocol_predictions(
                top5_scored,
                protocol="gqcnn_top5",
                method_alias=protocol_b_aliases[method],
            )
        )
    validation_predictions = pd.concat(validation_parts, ignore_index=True)
    validation_predictions.to_parquet(
        output_root / "validation_predictions.parquet", index=False
    )

    # Safe switch: train only from the residual model's scene-held-out scores.
    residual_oof = oof_predictions_internal.loc[
        oof_predictions_internal["reranker_method"] == "residual_mlp"
    ].merge(
        development[
            [
                "sample_id",
                "candidate_id",
                "q_raw",
                "p_axis_mean",
            ]
        ],
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    residual_oof["geometry_risk"] = geometry_risk(
        development.set_index(["sample_id", "candidate_id"]).loc[
            pd.MultiIndex.from_frame(
                residual_oof[["sample_id", "candidate_id"]]
            )
        ].reset_index()
    )[0]
    oof_examples = build_switch_examples(residual_oof)
    gate = SafeSwitchGate(seed=args.seed).fit(oof_examples)
    save_json(output_root / "models" / "safe_switch_gate.json", gate.artifact())

    # The gate is learned only from development OOF. The user-registered
    # protocol explicitly locks its switch threshold on official validation;
    # formal test data cannot affect this decision.
    residual_validation = validation_scored["residual_mlp"].copy()
    residual_validation["geometry_risk"] = geometry_risk(
        residual_validation
    )[0]
    validation_examples = build_switch_examples(residual_validation)
    validation_gate_confidence = gate.predict_confidence(
        validation_examples
    )
    sweep, selected = threshold_sweep(
        validation_examples,
        validation_gate_confidence,
        harmful_rate_limit=args.harmful_rate_limit,
        all_sample_count=validation_all_sample_count,
    )
    validation_decisions = apply_safe_switch(
        validation_examples,
        validation_gate_confidence,
        threshold=float(selected["threshold"]),
        force_no_switch=bool(selected["force_no_switch"]),
    )
    if float(selected["harmful_rate_all_samples"]) > (
        args.harmful_rate_limit + 1e-15
    ):
        raise AssertionError(
            "validation-locked safe switch violates harmful-rate limit"
        )
    oof_examples.to_parquet(output_root / "safe_switch_oof_examples.parquet", index=False)
    sweep.to_csv(output_root / "safe_switch_threshold_sweep.csv", index=False)
    validation_decisions.to_parquet(
        output_root / "safe_switch_validation_decisions.parquet", index=False
    )
    locked_safe_selection = {
        **identity_payload(),
        **dict(selected),
        "selection_split": "validation",
        "validation_per_candidate": str(validation_path),
        "validation_per_candidate_sha256": _sha256(validation_path),
    }
    save_json(
        output_root / "safe_switch_selection.json",
        locked_safe_selection,
    )
    safe_scored = candidate_predictions_from_switch(
        residual_validation, validation_decisions
    )
    assert_feature_identity_invariant(validation, safe_scored)
    validation_predictions = pd.concat(
        [
            validation_predictions,
            _compact_predictions(safe_scored, protocol="full_nms"),
        ],
        ignore_index=True,
    )
    validation_predictions.to_parquet(
        output_root / "validation_predictions.parquet", index=False
    )

    metrics: dict[str, Any] = {}
    for (protocol, method), scored in validation_predictions.groupby(
        ["protocol", "reranker_method"], sort=True
    ):
        metric = top1_accuracy(scored)
        metric.update(
            {
                "j_at_1_nonempty": float(metric["accuracy"]),
                "nonempty_samples": int(metric["total"]),
                "j_at_1_all_samples": (
                    int(metric["correct"]) / validation_all_sample_count
                    if validation_all_sample_count
                    else 0.0
                ),
                "all_samples": int(validation_all_sample_count),
            }
        )
        metrics[f"{protocol}/{method}"] = metric
    safe_key = (
        "full_nms/" + public_method_name("residual_mlp_safe_switch")
    )
    validation_recovered = int(
        (
            validation_decisions["switch_applied"]
            & (validation_examples["switch_outcome"] == "beneficial")
        ).sum()
    )
    validation_harmful = int(
        (
            validation_decisions["switch_applied"]
            & (validation_examples["switch_outcome"] == "harmful")
        ).sum()
    )
    validation_switches = int(
        validation_decisions["switch_applied"].sum()
    )
    metrics[safe_key].update(
        {
            "recovered": validation_recovered,
            "harmful": validation_harmful,
            "net_gain": validation_recovered - validation_harmful,
            "decision_precision": (
                validation_recovered
                / (validation_recovered + validation_harmful)
                if validation_recovered + validation_harmful
                else 0.0
            ),
            "coverage": (
                validation_switches / validation_all_sample_count
                if validation_all_sample_count
                else 0.0
            ),
            "harmful_rate_all_samples": (
                validation_harmful / validation_all_sample_count
                if validation_all_sample_count
                else 0.0
            ),
            "harmful_rate_nonempty_samples": (
                validation_harmful / len(validation_examples)
                if len(validation_examples)
                else 0.0
            ),
            "harmful_rate_denominator": "all_validation_samples",
            "all_validation_sample_count": int(validation_all_sample_count),
            "nonempty_validation_sample_count": int(len(validation_examples)),
            "harmful_rate_limit": float(args.harmful_rate_limit),
            "force_no_switch": bool(selected["force_no_switch"]),
            "threshold_selection_split": "validation",
            "locked_threshold": float(selected["threshold"]),
            "validation_locked_harmful_rate_all_samples": float(
                selected["harmful_rate_all_samples"]
            ),
        }
    )
    save_json(output_root / "validation_metrics.json", metrics)
    development_metrics: dict[str, Any] = {}
    for method, scored in _direct_scores(
        development, softmask_alpha=softmask_alpha
    ).items():
        development_metrics[
            f"direct/{public_method_name(method)}"
        ] = top1_accuracy(scored)
    for method, scored in oof_predictions_public.groupby(
        "reranker_method", sort=True
    ):
        development_metrics[
            f"scene_grouped_oof/{method}"
        ] = top1_accuracy(scored)
    save_json(
        output_root / "development_oof_metrics.json", development_metrics
    )

    # Reload every saved model before the artifacts can be considered usable.
    parity: dict[str, Any] = {}
    linear_reloaded = RegularizedLinearRanker.load(
        output_root / "checkpoints" / "regularized_linear_ranker.json"
    )
    expected_linear = fitted["regularized_linear_ranker"].predict_scores(validation)
    actual_linear = linear_reloaded.predict_scores(validation)
    parity["regularized_linear_ranker"] = float(
        np.max(np.abs(expected_linear - actual_linear))
    )
    for method in LEARNED_METHODS:
        if method == "regularized_linear_ranker":
            continue
        reloaded = TorchCandidateRanker.load(
            output_root / "checkpoints" / f"{method}.pt",
            device=str(fitted[method].device),
        )
        expected = fitted[method].predict_scores(validation)
        actual = reloaded.predict_scores(validation)
        parity[method] = float(np.max(np.abs(expected - actual)))
    gate_reloaded = SafeSwitchGate.load(
        output_root / "models" / "safe_switch_gate.json"
    )
    parity["safe_switch_gate"] = float(
        np.max(
            np.abs(
                validation_gate_confidence
                - gate_reloaded.predict_confidence(validation_examples)
            )
        )
    )
    tolerance = 1e-6 if args.device != "cpu" else 1e-10
    if any(value > tolerance for value in parity.values()):
        raise AssertionError(f"model reload parity failed: {parity}")
    save_json(
        output_root / "reload_parity.json",
        {"maximum_absolute_differences": parity, "tolerance": tolerance},
    )

    bundle = {
        "schema_version": 1,
        **identity_payload(),
        "registry_key_namespace": "internal_implementation_keys",
        "internal_registries": {
            "rule_methods": "internal_implementation_keys",
            "learned_models": "internal_implementation_keys",
            "safe_switch_gate": "internal_implementation_key",
        },
        "feature_columns": list(features),
        "rule_methods": {
            "q_only": {},
            "q_softmask_rule": {
                "alpha": softmask_alpha,
                "beta": 1.0 - softmask_alpha,
            },
            "geometry_gated_q": asdict(GeometryGateConfig()),
        },
        "learned_models": {
            "regularized_linear_ranker": "checkpoints/regularized_linear_ranker.json",
            **{
                method: f"checkpoints/{method}.pt"
                for method in LEARNED_METHODS
                if method != "regularized_linear_ranker"
            },
        },
        "scaler_metadata": "scaler_metadata.json",
        "scaler_metadata_sha256": scaler_metadata_sha256,
        "safe_switch_gate": "models/safe_switch_gate.json",
        "safe_switch_selection": "safe_switch_selection.json",
        "candidate_pool_modified": False,
        "reload_parity_verified": True,
        "primary_candidate_methods": public_method_names(
            [
                "q_softmask_rule",
                "geometry_gated_q",
                *LEARNED_METHODS,
                "residual_mlp_safe_switch",
            ]
        ),
        "public_method_names": {
            method: public_method_name(method)
            for method in [
                "q_only",
                "q_softmask_rule",
                "geometry_gated_q",
                *LEARNED_METHODS,
                "residual_mlp_safe_switch",
                "q_top5",
                "tabular_residual_top5",
                "setrank_top5",
            ]
        },
    }
    save_json(output_root / "inference_bundle.json", bundle)

    validate_public_methods(
        list(map(str, bundle["primary_candidate_methods"])),
        context="inference bundle primary_candidate_methods",
    )
    manifest = {
        "schema_version": 1,
        **identity_payload(),
        "development_per_candidate": str(development_path),
        "development_sha256": _sha256(development_path),
        "calibration_per_candidate": str(calibration_path),
        "calibration_sha256": _sha256(calibration_path),
        "calibration_per_sample": str(calibration_per_sample_path),
        "calibration_per_sample_sha256": _sha256(
            calibration_per_sample_path
        ),
        "validation_per_candidate": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "validation_per_sample": (
            str(validation_per_sample_path)
            if validation_per_sample_path is not None
            else None
        ),
        "validation_per_sample_sha256": (
            _sha256(validation_per_sample_path)
            if validation_per_sample_path is not None
            else None
        ),
        "development_candidates": int(len(development)),
        "development_samples": int(development["sample_id"].nunique()),
        "development_scenes": int(development["scene_id"].nunique()),
        "calibration_candidates": int(len(calibration)),
        "calibration_samples": int(calibration["sample_id"].nunique()),
        "calibration_all_samples": int(calibration_all_sample_count),
        "calibration_scenes": int(calibration["scene_id"].nunique()),
        "validation_candidates": int(len(validation)),
        "validation_samples": int(validation["sample_id"].nunique()),
        "validation_all_samples": int(validation_all_sample_count),
        "safe_switch_threshold_denominator": harmful_rate_denominator,
        "validation_scenes": int(validation["scene_id"].nunique()),
        "methods": public_method_names(
            [
                "q_only",
                "q_softmask_rule",
                "geometry_gated_q",
                *LEARNED_METHODS,
                "residual_mlp_safe_switch",
            ]
        ),
        "primary_candidate_methods": public_method_names(
            [
                "q_softmask_rule",
                "geometry_gated_q",
                *LEARNED_METHODS,
                "residual_mlp_safe_switch",
            ]
        ),
        "protocols": ["full_nms", "gqcnn_top5"],
        "candidate_pool_modified": False,
        "formal_test_consumed": False,
        "official_validation_used_for_gradient_training": False,
        "official_validation_used_for_rule_threshold_and_method_selection": True,
        "rule_selection": str(rule_selection_path),
        "rule_selection_sha256": _sha256(rule_selection_path),
        "safe_switch_selection": locked_safe_selection,
        "safe_switch_selection_path": str(
            output_root / "safe_switch_selection.json"
        ),
        "safe_switch_selection_sha256": _sha256(
            output_root / "safe_switch_selection.json"
        ),
        "final_deployment_early_stopping_split": "calibration",
        "oof_fold_stopping": "fold_development_training_loss",
        "inference_bundle": str(output_root / "inference_bundle.json"),
        "scaler_metadata": str(scaler_metadata_path),
        "scaler_metadata_sha256": scaler_metadata_sha256,
        "scaler_sha256": scaler_metadata["scaler_sha256"],
        "reload_parity": parity,
        "resolved_devices": {
            method: (
                str(model.device)
                if isinstance(model, TorchCandidateRanker)
                else "cpu"
            )
            for method, model in fitted.items()
        },
    }
    validate_public_methods(
        list(map(str, manifest["methods"])),
        context="training manifest methods",
    )
    validate_public_methods(
        list(map(str, manifest["primary_candidate_methods"])),
        context="training manifest primary_candidate_methods",
    )
    save_json(output_root / "training_manifest.json", manifest)
    (output_root / "run_command.txt").write_text(
        " ".join(map(shlex.quote, [sys.executable, *sys.argv])) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-per-candidate", type=Path, required=True
    )
    parser.add_argument(
        "--calibration-per-candidate", type=Path, required=True
    )
    parser.add_argument(
        "--calibration-per-sample",
        type=Path,
        required=True,
        help=(
            "Complete held-out calibration universe, including valid-empty "
            "samples; final-deployment early stopping may consume only this set"
        ),
    )
    parser.add_argument("--validation-per-candidate", type=Path, required=True)
    parser.add_argument(
        "--validation-per-sample",
        type=Path,
        required=True,
        help=(
            "Complete validation sample universe, including valid-empty samples; "
            "required for the registered all-sample harmful-rate denominator"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-columns", type=Path)
    parser.add_argument("--rule-selection", type=Path, required=True)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--residual-bound", type=float, default=0.25)
    parser.add_argument("--q-alpha", type=float, default=1.0)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--hard-negative-limit", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--harmful-rate-limit", type=float, default=0.01)
    parser.add_argument("--softmask-alpha", type=float)
    args = parser.parse_args(argv)
    if args.oof_folds < 2:
        parser.error("--oof-folds must be at least 2")
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("--epochs and --patience must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if args.residual_bound <= 0.0:
        parser.error("--residual-bound must be positive")
    if args.sample_batch_size <= 0 or args.hard_negative_limit <= 0:
        parser.error("--sample-batch-size and --hard-negative-limit must be positive")
    if not 0.0 <= args.harmful_rate_limit <= 1.0:
        parser.error("--harmful-rate-limit must be in [0, 1]")
    if args.softmask_alpha is not None and not 0.0 <= args.softmask_alpha <= 1.0:
        parser.error("--softmask-alpha must be in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(parse_args(argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
