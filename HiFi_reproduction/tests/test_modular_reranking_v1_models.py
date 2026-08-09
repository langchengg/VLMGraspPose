from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.grasping.reranking_v1.features import INFERENCE_FEATURE_ALLOWLIST
from src.grasping.reranking_v1.artifact_contract import (
    EXPERIMENT_NAME,
    PROTOCOL_BASELINES,
    PUBLIC_METHOD_NAMESPACE_VERSION,
    SCALER_METADATA_ARTIFACT_KIND,
    SCALER_MODEL_ARTIFACT_KEYS,
    build_final_scaler_metadata,
    identity_payload,
    validate_artifact_identity,
    validate_config_identity,
    validate_public_methods,
)
from src.grasping.reranking_v1.models import (
    GeometryGateConfig,
    NeuralRankerConfig,
    RegularizedLinearRanker,
    TorchCandidateRanker,
    TrainOnlyScaler,
    assert_feature_identity_invariant,
    attach_scores,
    geometry_gated_q_scores,
    multi_positive_listwise_loss,
    q_only_scores,
    q_softmask_rule_scores,
    sample_balanced_pairwise_ranknet_loss,
    scene_grouped_oof,
    validate_candidate_contract,
)
from src.grasping.reranking_v1.method_namespace import (
    expected_formal_method_protocols,
    public_method_name,
    public_method_names,
)
from src.grasping.reranking_v1.safe_switch import (
    SafeSwitchGate,
    apply_safe_switch,
    build_switch_examples,
    build_switch_features,
    threshold_sweep,
    validate_gate_features,
)
from src.grasping.reranking_v1.experiment_lock import (
    canonical_json_sha256,
    sha256_file,
)
from tools.modular_reranking.apply_rerankers import (
    main as apply_main,
    score_candidate_frame,
)
from tools.modular_reranking.train_rerankers import main as train_main
from tools.modular_reranking.benchmark_tabular_inference import (
    main as benchmark_main,
)
from tools.modular_reranking.build_tabular_eligibility_evidence import (
    main as build_eligibility_main,
)
from tools.modular_reranking.select_rule_hyperparameters import (
    main as select_rule_main,
)


def test_repeatedfilm_artifact_contract_rejects_generic_config() -> None:
    with pytest.raises(ValueError, match="repeatedfilm_v1 experiment"):
        validate_config_identity(
            {
                "experiment": "modular_reranking_v1",
                "baseline_name": PROTOCOL_BASELINES["full_nms"],
                "protocols": {
                    name: {"baseline": baseline}
                    for name, baseline in PROTOCOL_BASELINES.items()
                },
            }
        )


def test_repeatedfilm_artifact_contract_rejects_bare_method_name() -> None:
    with pytest.raises(ValueError, match="bare/internal"):
        validate_public_methods(
            ["q_softmask_rule"], context="test public methods"
        )


def test_repeatedfilm_artifact_contract_rejects_missing_baseline() -> None:
    payload = identity_payload()
    payload.pop("baseline_name")
    with pytest.raises(ValueError, match="baseline_name"):
        validate_artifact_identity(payload, context="test artifact")


def test_repeatedfilm_artifact_contract_rejects_protocol_baseline_drift() -> None:
    payload = identity_payload()
    payload["protocol_baselines"]["gqcnn_top5"] = (
        "repeatedfilm_gqcnn_q_only"
    )
    with pytest.raises(ValueError, match="must exactly equal"):
        validate_artifact_identity(payload, context="test artifact")


def _synthetic_frame(
    *,
    split: str = "train",
    scenes: int = 6,
    samples_per_scene: int = 2,
    candidates: int = 4,
    prefix: str = "",
) -> pd.DataFrame:
    rows = []
    sample_number = 0
    for scene_index in range(scenes):
        for local_sample in range(samples_per_scene):
            sample_id = f"{prefix}s{scene_index:02d}_q{local_sample:02d}"
            positive_index = 0 if sample_number % 2 else 1
            for candidate_index in range(candidates):
                positive = candidate_index == positive_index
                # GQ-CNN succeeds on half the samples and chooses candidate 0
                # (the wrong candidate) on the other half.
                q_rank = 1.0 - 0.08 * candidate_index
                row = {
                    "sample_id": sample_id,
                    "scene_id": f"{prefix}scene_{scene_index:02d}",
                    "candidate_id": f"c{candidate_index:03d}",
                    "candidate_identity_sha256": hashlib.sha256(
                        f"{prefix}/{scene_index}/{sample_number}/{candidate_index}".encode()
                    ).hexdigest(),
                    "split": split,
                    "query_type": "name",
                    "original_gqcnn_rank": candidate_index + 1,
                    "candidate_positive": positive,
                }
                for feature in INFERENCE_FEATURE_ALLOWLIST:
                    row[feature] = 0.0
                row.update(
                    {
                        "q_raw": q_rank,
                        "q_log": np.log(max(q_rank, 1e-6)),
                        "q_percentile_within_sample": q_rank,
                        "q_rank_normalized": q_rank,
                        "q_gap_to_top1": q_rank - 1.0,
                        "p_center": 0.95 if positive else 0.05,
                        "p_axis_mean": 0.95 if positive else 0.05,
                        "p_contact_min": 0.9 if positive else 0.1,
                        "grasp_axis_mask_support": 0.9 if positive else 0.1,
                        "width_ratio_to_max_gripper": 0.6,
                        "normalized_width_mismatch": 0.1 if positive else 0.4,
                        "jaw_depth_difference": 0.005 if positive else 0.02,
                        "normal_opposition": 0.8 if positive else 0.0,
                        "contact_symmetry": 0.8 if positive else 0.3,
                        "left_finger_occupancy": 0.05,
                        "right_finger_occupancy": 0.05,
                        "palm_occupancy": 0.05,
                        "approach_corridor_occupancy": 0.05,
                        "approach_clearance": 0.02,
                        "collision_proxy_total": 0.1,
                        "candidate_uniqueness": 0.8 if positive else 0.3,
                        "cluster_q_mean": q_rank - 0.01,
                        "cluster_q_std": 0.02,
                    }
                )
                rows.append(row)
            sample_number += 1
    return pd.DataFrame(rows)


def test_rules_are_deterministic_and_identity_preserving() -> None:
    frame = _synthetic_frame(scenes=1, samples_per_scene=1)
    q = q_only_scores(frame)
    assert np.array_equal(q, frame["q_raw"].to_numpy())
    soft = q_softmask_rule_scores(frame, alpha=0.5, beta=0.5)
    assert soft[1] > soft[0]
    tied = attach_scores(frame, np.ones(len(frame)), method="tie")
    assert tied.sort_values("reranker_rank")["candidate_id"].tolist() == sorted(
        frame["candidate_id"].tolist()
    )
    assert_feature_identity_invariant(frame, tied)
    validate_candidate_contract(
        frame, require_label=True, allowed_splits={"train"}
    )
    broken = frame.copy()
    broken.loc[broken.index[0], "original_gqcnn_rank"] = 2
    with pytest.raises(ValueError, match="rank"):
        validate_candidate_contract(broken)


def test_train_only_scaler_and_allowlist_fail_closed() -> None:
    train = _synthetic_frame(scenes=2)
    scaler = TrainOnlyScaler(("q_raw", "p_axis_mean")).fit(train)
    assert scaler.to_dict()["fit_scope"] == "train/development candidates only"
    validation = train.copy()
    validation["split"] = "validation"
    with pytest.raises(ValueError, match="train/development"):
        TrainOnlyScaler(("q_raw",)).fit(validation)
    with pytest.raises(ValueError, match="forbidden GT"):
        TrainOnlyScaler(("candidate_positive",))


def test_final_scaler_metadata_rejects_different_model_scalers(
    tmp_path: Path,
) -> None:
    base_scaler = {
        "feature_columns": ["q_raw"],
        "mean": [0.0],
        "scale": [1.0],
        "source_splits": ["development"],
        "fit_scope": "train/development candidates only",
    }
    artifacts = {
        method: {
            "method": method,
            "scaler": {
                **base_scaler,
                "mean": (
                    [1.0]
                    if method == "set_aware_residual"
                    else [0.0]
                ),
            },
        }
        for method in SCALER_MODEL_ARTIFACT_KEYS
    }
    paths = {}
    for method in SCALER_MODEL_ARTIFACT_KEYS:
        path = tmp_path / f"{method}.artifact"
        path.write_text("{}\n")
        paths[method] = path
    with pytest.raises(ValueError, match="do not share one identical scaler"):
        build_final_scaler_metadata(artifacts, paths)


def test_geometry_gate_penalizes_severe_visible_surface_risk() -> None:
    frame = _synthetic_frame(scenes=1, samples_per_scene=1)
    safe_score = geometry_gated_q_scores(frame)
    unsafe = frame.copy()
    unsafe.loc[0, "palm_occupancy"] = 0.99
    unsafe_score = geometry_gated_q_scores(
        unsafe, GeometryGateConfig(risk_penalty=3.0)
    )
    assert unsafe_score[0] < safe_score[0] - 3.0


def test_pairwise_loss_is_sample_balanced_and_handles_all_negative() -> None:
    basic = sample_balanced_pairwise_ranknet_loss(
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        ["a", "a", "b", "b"],
    )
    duplicated_a = sample_balanced_pairwise_ranknet_loss(
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]),
        ["a", "a", "a", "b", "b"],
    )
    assert torch.allclose(basic, duplicated_a)
    scores = torch.tensor([0.2, 0.1], requires_grad=True)
    empty = sample_balanced_pairwise_ranknet_loss(
        scores, torch.zeros(2), ["all_negative", "all_negative"]
    )
    assert float(empty) == 0.0
    empty.backward()
    assert scores.grad is not None


def test_multi_positive_listwise_loss_rewards_positive_mass() -> None:
    labels = torch.tensor([1.0, 1.0, 0.0])
    good = multi_positive_listwise_loss(
        torch.tensor([2.0, 1.0, -1.0]), labels, ["s", "s", "s"]
    )
    bad = multi_positive_listwise_loss(
        torch.tensor([-1.0, 0.0, 2.0]), labels, ["s", "s", "s"]
    )
    assert good < bad


def test_linear_pairwise_and_listwise_rankers_fit_synthetic_data() -> None:
    frame = _synthetic_frame()
    linear = RegularizedLinearRanker(
        ("q_rank_normalized", "p_axis_mean"), c=1.0
    ).fit(frame)
    assert len(linear.predict_scores(frame)) == len(frame)
    coefficients = linear.coefficients()
    assert set(coefficients["feature"]) == {
        "q_rank_normalized",
        "p_axis_mean",
    }
    config = NeuralRankerConfig(
        epochs=3, patience=2, learning_rate=0.02, seed=7
    )
    for method in ("pairwise_ranker", "multi_positive_listwise_ranker"):
        model = TorchCandidateRanker(
            method,
            ("q_rank_normalized", "p_axis_mean"),
            config=config,
        ).fit(frame)
        assert np.all(np.isfinite(model.predict_scores(frame)))


def test_residual_mlp_is_bounded_and_deepsets_is_permutation_equivariant() -> None:
    frame = _synthetic_frame(scenes=2)
    config = NeuralRankerConfig(
        epochs=4,
        patience=2,
        learning_rate=0.02,
        residual_bound=0.17,
        dropout=0.0,
        seed=11,
    )
    residual = TorchCandidateRanker(
        "residual_mlp",
        ("q_rank_normalized", "p_axis_mean"),
        config=config,
    ).fit(frame)
    assert np.max(np.abs(residual.residuals(frame))) <= 0.17 + 1e-6

    deepsets = TorchCandidateRanker(
        "set_aware_residual",
        ("q_rank_normalized", "p_axis_mean"),
        config=config,
    ).fit(frame)
    first = deepsets.predict_scores(frame)
    permutation = frame.sample(frac=1.0, random_state=5)
    second = pd.Series(
        deepsets.predict_scores(permutation), index=permutation.index
    ).sort_index()
    assert np.allclose(first, second.to_numpy(), atol=1e-6)


def test_scene_grouped_oof_scores_each_candidate_once() -> None:
    frame = _synthetic_frame(scenes=6)

    def factory() -> RegularizedLinearRanker:
        return RegularizedLinearRanker(
            ("q_rank_normalized", "p_axis_mean"), c=1.0
        )

    scored, artifacts = scene_grouped_oof(frame, factory, n_splits=3)
    assert len(scored) == len(frame)
    assert scored["oof_fold"].notna().all()
    assert_feature_identity_invariant(frame, scored)
    for fold in artifacts:
        assert not (
            set(fold["train_scenes"]) & set(fold["held_out_scenes"])
        )


def _switch_scored() -> pd.DataFrame:
    rows = []
    # beneficial, harmful, and neutral samples
    definitions = [
        ("beneficial", False, True, 0.9),
        ("harmful", True, False, 0.8),
        ("neutral", True, True, 0.2),
        ("neutral_wrong", False, False, 0.1),
    ]
    for index, (sample, old_ok, new_ok, confidence) in enumerate(definitions):
        for candidate, old_rank, new_score, correct in (
            ("old", 1, 0.0, old_ok),
            ("new", 2, confidence, new_ok),
        ):
            rows.append(
                {
                    "sample_id": sample,
                    "scene_id": f"scene_{index}",
                    "candidate_id": candidate,
                    "q_raw": 0.8 if candidate == "old" else 0.7,
                    "original_gqcnn_rank": old_rank,
                    "reranker_score": new_score,
                    "p_axis_mean": 0.5 + 0.1 * (candidate == "new"),
                    "geometry_risk": 0.0,
                    "candidate_positive": correct,
                    "oof_fold": index % 2,
                }
            )
    return pd.DataFrame(rows)


def test_safe_switch_training_threshold_and_fallback() -> None:
    examples = build_switch_examples(_switch_scored())
    assert set(examples["switch_outcome"]) == {
        "beneficial",
        "harmful",
        "neutral",
    }
    gate = SafeSwitchGate().fit(examples)
    confidence = gate.predict_confidence(examples)
    sweep, selected = threshold_sweep(
        examples, confidence, harmful_rate_limit=0.25
    )
    assert len(sweep) == 202
    assert 0.0 <= float(selected["threshold"]) <= 1.0
    decisions = apply_safe_switch(
        examples, confidence, threshold=float(selected["threshold"])
    )
    assert set(decisions["selected_candidate_id"]) <= {"old", "new"}

    unsafe = examples.iloc[[0]].copy()
    unsafe["new_geometry_safe"] = False
    fallback = apply_safe_switch(unsafe, [1.0], threshold=0.0)
    assert fallback.iloc[0]["selected_candidate_id"] == "old"
    assert fallback.iloc[0]["fallback_reason"] == "new_geometry_unsafe"
    invalid = apply_safe_switch(examples.iloc[[0]], [np.nan], threshold=0.0)
    assert invalid.iloc[0]["selected_candidate_id"] == "old"
    with pytest.raises(ValueError):
        validate_gate_features(["candidate_positive"])

    constant_examples = examples.copy()
    constant_examples["switch_outcome"] = "neutral"
    constant_gate = SafeSwitchGate().fit(constant_examples)
    assert np.array_equal(
        constant_gate.predict_confidence(constant_examples),
        np.zeros(len(constant_examples)),
    )
    assert constant_gate.artifact()["gate_kind"].startswith("constant")
    label_free = build_switch_features(
        _switch_scored().drop(columns=["candidate_positive"])
    )
    assert "switch_outcome" not in label_free
    exact_one = np.ones(len(examples))
    forced_sweep, forced_selected = threshold_sweep(
        examples.assign(switch_outcome="harmful"),
        exact_one,
        harmful_rate_limit=0.0,
    )
    assert forced_sweep.iloc[-1]["force_no_switch"]
    assert forced_selected["force_no_switch"] is True
    forced = apply_safe_switch(
        examples, exact_one, threshold=1.0, force_no_switch=True
    )
    assert not forced["switch_applied"].any()


def test_training_cli_writes_complete_artifact_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development = _synthetic_frame(scenes=6)
    calibration = _synthetic_frame(
        split="calibration", scenes=3, prefix="calibration_"
    )
    validation = _synthetic_frame(
        split="validation", scenes=3, prefix="validation_"
    )
    development_path = tmp_path / "development.parquet"
    calibration_path = tmp_path / "calibration.parquet"
    validation_path = tmp_path / "validation.parquet"
    calibration_universe_path = tmp_path / "calibration_per_sample.parquet"
    validation_universe_path = tmp_path / "validation_per_sample.parquet"
    output = tmp_path / "trained"
    features_path = tmp_path / "features.json"
    rule_selection_path = tmp_path / "rule_selection.json"
    rule_sweep_path = tmp_path / "rule_sweep.csv"
    features_path.write_text(
        json.dumps({"features": ["q_rank_normalized", "p_axis_mean"]}),
        encoding="utf-8",
    )
    development.to_parquet(development_path, index=False)
    calibration.to_parquet(calibration_path, index=False)
    validation.to_parquet(validation_path, index=False)
    calibration_universe = (
        calibration.groupby(["sample_id", "scene_id", "split"], as_index=False)
        .size()
        .rename(columns={"size": "candidate_count"})
    )
    calibration_universe.to_parquet(
        calibration_universe_path, index=False
    )
    validation_universe = (
        validation.groupby(["sample_id", "scene_id", "split"], as_index=False)
        .size()
        .rename(columns={"size": "candidate_count"})
    )
    validation_universe.to_parquet(validation_universe_path, index=False)
    rule_sweep_path.write_text("alpha,net_gain\n0.8,1\n", encoding="utf-8")
    rule_selection_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "method": "q_softmask_rule",
                "selection_split": "validation",
                "selected": {"alpha": 0.8, "beta": 0.2},
                "validation_per_candidate": str(validation_path.resolve()),
                "validation_per_candidate_sha256": sha256_file(
                    validation_path
                ),
                "sweep": str(rule_sweep_path.resolve()),
                "sweep_sha256": sha256_file(rule_sweep_path),
            }
        ),
        encoding="utf-8",
    )
    result = train_main(
        [
            "--development-per-candidate",
            str(development_path),
            "--calibration-per-candidate",
            str(calibration_path),
            "--calibration-per-sample",
            str(calibration_universe_path),
            "--validation-per-candidate",
            str(validation_path),
            "--validation-per-sample",
            str(validation_universe_path),
            "--output-root",
            str(output),
            "--feature-columns",
            str(features_path),
            "--rule-selection",
            str(rule_selection_path),
            "--oof-folds",
            "3",
            "--epochs",
            "4",
            "--patience",
            "2",
            "--learning-rate",
            "0.02",
            "--dropout",
            "0",
            "--residual-bound",
            "0.5",
            "--device",
            "cpu",
        ]
    )
    assert result == 0
    expected = {
        "hyperparameters.json",
        "feature_columns.json",
        "coefficients.csv",
        "oof_predictions.parquet",
        "oof_fold_models.json",
        "validation_predictions.parquet",
        "safe_switch_oof_examples.parquet",
        "safe_switch_threshold_sweep.csv",
        "safe_switch_validation_decisions.parquet",
        "safe_switch_selection.json",
        "validation_metrics.json",
        "development_oof_metrics.json",
        "reload_parity.json",
        "inference_bundle.json",
        "scaler_metadata.json",
        "training_manifest.json",
        "run_command.txt",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    manifest = json.loads(
        (output / "training_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["formal_test_consumed"] is False
    assert manifest["candidate_pool_modified"] is False
    assert manifest["experiment"] == EXPERIMENT_NAME
    assert (
        manifest["public_method_namespace_version"]
        == PUBLIC_METHOD_NAMESPACE_VERSION
    )
    assert manifest["protocol_baselines"] == PROTOCOL_BASELINES
    assert manifest["baseline_name"] == PROTOCOL_BASELINES["full_nms"]
    inference_bundle = json.loads(
        (output / "inference_bundle.json").read_text()
    )
    scaler_metadata_path = output / "scaler_metadata.json"
    scaler_metadata = json.loads(scaler_metadata_path.read_text())
    assert (
        scaler_metadata["artifact_kind"]
        == SCALER_METADATA_ARTIFACT_KIND
    )
    assert scaler_metadata["feature_columns"] == [
        "q_rank_normalized",
        "p_axis_mean",
    ]
    assert set(scaler_metadata["models"]) == set(
        SCALER_MODEL_ARTIFACT_KEYS
    )
    assert {
        row["scaler_sha256"]
        for row in scaler_metadata["models"].values()
    } == {scaler_metadata["scaler_sha256"]}
    assert manifest["scaler_metadata"] == str(scaler_metadata_path)
    assert manifest["scaler_metadata_sha256"] == sha256_file(
        scaler_metadata_path
    )
    assert manifest["scaler_sha256"] == scaler_metadata["scaler_sha256"]
    assert inference_bundle["scaler_metadata"] == "scaler_metadata.json"
    assert inference_bundle["scaler_metadata_sha256"] == sha256_file(
        scaler_metadata_path
    )
    for field in (
        "experiment",
        "public_method_namespace_version",
        "baseline_name",
        "protocol_baselines",
    ):
        assert inference_bundle[field] == manifest[field]
    oof_methods = set(
        pd.read_parquet(output / "oof_predictions.parquet")[
            "reranker_method"
        ].astype(str)
    )
    assert oof_methods
    assert all(method.startswith("repeatedfilm_") for method in oof_methods)
    development_metrics = json.loads(
        (output / "development_oof_metrics.json").read_text()
    )
    assert all(
        key.split("/", 1)[1].startswith("repeatedfilm_")
        for key in development_metrics
    )
    assert set(path.name for path in (output / "checkpoints").iterdir()) == {
        "regularized_linear_ranker.json",
        "pairwise_ranker.pt",
        "multi_positive_listwise_ranker.pt",
        "residual_mlp.pt",
        "set_aware_residual.pt",
    }
    metrics = json.loads(
        (output / "validation_metrics.json").read_text(encoding="utf-8")
    )
    safe_key = (
        "full_nms/" + public_method_name("residual_mlp_safe_switch")
    )
    assert safe_key in metrics
    assert metrics[safe_key][
        "threshold_selection_split"
    ] == "validation"
    assert metrics[
        "full_nms/" + public_method_name("q_only")
    ]["all_samples"] == len(
        validation_universe
    )
    selection = json.loads(
        (output / "safe_switch_selection.json").read_text(encoding="utf-8")
    )
    assert selection == manifest["safe_switch_selection"]
    assert selection["harmful_rate_denominator"] == "all_validation_samples"

    benchmark_root = tmp_path / "benchmark"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_tabular_inference.py",
            "--per-candidate",
            str(validation_path),
            "--sample-universe",
            str(validation_universe_path),
            "--training-root",
            str(output),
            "--output-root",
            str(benchmark_root),
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert benchmark_main() == 0
    benchmark_path = benchmark_root / "runtime_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    assert benchmark["sample_count"] == len(validation_universe)
    assert benchmark["candidate_identity_invariant"] is True
    assert benchmark["protocol_baselines"] == PROTOCOL_BASELINES
    assert benchmark["baseline_name"] == manifest["baseline_name"]

    split_audit_path = tmp_path / "split_audit.json"
    split_audit_path.write_text(
        json.dumps({"required_intersections_all_zero": True})
    )
    allowlist_path = tmp_path / "allowlist.json"
    allowlist_path.write_text(
        json.dumps(
            {
                "ground_truth_allowed": False,
                "features": ["q_rank_normalized", "p_axis_mean"],
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_payload = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "modular_reranking_repeatedfilm_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    config_payload["primary_selection"][
        "maximum_inference_seconds_per_sample"
    ] = 0.1
    config_payload["primary_selection"]["inference_device"] = "cpu"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )
    evidence_path = tmp_path / "eligibility.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tabular_eligibility_evidence.py",
            "--config",
            str(config_path),
            "--split-audit",
            str(split_audit_path),
            "--feature-allowlist",
            str(allowlist_path),
            "--training-manifest",
            str(output / "training_manifest.json"),
            "--runtime-benchmark",
            str(benchmark_path),
            "--output",
            str(evidence_path),
        ],
    )
    assert build_eligibility_main() == 0
    evidence = json.loads(evidence_path.read_text())
    assert evidence["maximum_inference_seconds_per_sample"] == 0.1
    assert evidence["formal_inference_device"] == "cpu"
    assert evidence["protocol_baselines"] == PROTOCOL_BASELINES
    assert evidence["baseline_name"] == manifest["baseline_name"]
    assert set(evidence["candidate_methods"]) == set(
        manifest["primary_candidate_methods"]
    )

    predictions, decisions, audit = score_candidate_frame(
        validation.drop(columns=["candidate_positive"]),
        training_root=output,
        device="cpu",
    )
    assert set(predictions["reranker_method"]) == set(
        public_method_names(
            [
                "q_only",
                "q_softmask_rule",
                "geometry_gated_q",
                "regularized_linear_ranker",
                "pairwise_ranker",
                "multi_positive_listwise_ranker",
                "residual_mlp",
                "set_aware_residual",
                "residual_mlp_safe_switch",
                "q_top5",
                "tabular_residual_top5",
                "setrank_top5",
            ]
        )
    )
    assert "candidate_positive" not in predictions
    assert "switch_outcome" not in decisions
    assert audit["candidate_identity_invariant"] is True

    test = validation.drop(columns=["candidate_positive"]).copy()
    test["split"] = "test"
    test_path = tmp_path / "test.parquet"
    universe_path = tmp_path / "test_universe.parquet"
    test.to_parquet(test_path, index=False)
    test[["sample_id", "scene_id"]].drop_duplicates().to_parquet(
        universe_path, index=False
    )
    bundle = json.loads((output / "inference_bundle.json").read_text())
    locked_paths = [
        test_path,
        universe_path,
        output / "inference_bundle.json",
        output / bundle["safe_switch_gate"],
        output / bundle["safe_switch_selection"],
        *[output / value for value in bundle["learned_models"].values()],
    ]
    vlm_validation_selection = tmp_path / "vlm_validation_selection.json"
    vlm_validation_selection.write_text(
        json.dumps(
            {
                **identity_payload(),
                "selection_split": "validation",
                "selected_method": "repeatedfilm_local_vlm_visual",
                "selected_variant": "visual",
                "selected_model_digest": "test-vlm-digest",
                "selected_stable_session_contract_sha256": "e" * 64,
                "candidates": [
                    {
                        "method": "repeatedfilm_local_vlm_visual",
                        "variant": "visual",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    lock_payload = {
        "schema_version": 2,
        **identity_payload(),
        "lock_kind": "modular_reranking_v1_pre_formal_test",
        "artifacts": {
            f"artifact_{index}": {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
            }
                for index, path in enumerate(locked_paths)
        }
        | {
            "vlm_validation_selection": {
                "path": str(vlm_validation_selection.resolve()),
                "sha256": sha256_file(vlm_validation_selection),
            }
        },
        "source_code_hashes": {},
        "selected_feature_list": bundle["feature_columns"],
        "safe_switch_threshold": json.loads(
            (output / "safe_switch_selection.json").read_text()
        )["threshold"],
            "ranking_parameters": {
                "safe_switch_force_no_switch": json.loads(
                    (output / "safe_switch_selection.json").read_text()
                )["force_no_switch"],
                "tabular_inference_device": "cpu",
                "vlm_safe_switch": {
                    "method": "repeatedfilm_local_vlm_safe_switch"
                },
            },
        "vlm_model_digest": "test-vlm-digest",
        "selected_vlm": {
            "source_method": "repeatedfilm_local_vlm_visual",
            "source_variant": "visual",
            "protocol": "gqcnn_top5",
            "model_digest": "test-vlm-digest",
            "stable_session_contract_sha256": "e" * 64,
            "validation_selection_path": str(
                vlm_validation_selection.resolve()
            ),
            "validation_selection_sha256": sha256_file(
                vlm_validation_selection
            ),
        },
        "evaluation_definition": {
            "formal_method_protocols": [
                {"protocol": protocol, "method": method}
                for protocol, method in expected_formal_method_protocols(
                    "repeatedfilm_local_vlm_visual"
                )
            ]
        },
        "expected_test_sample_count": test["sample_id"].nunique(),
        "formal_test_consumed": False,
    }
    lock_payload["manifest_content_sha256"] = canonical_json_sha256(
        lock_payload
    )
    lock_path = tmp_path / "frozen_experiment_manifest.json"
    lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")
    monkeypatch.setattr(
        "tools.modular_reranking.apply_rerankers.verify_lock",
        lambda _path: lock_payload,
    )
    monkeypatch.setattr(
        "src.grasping.reranking_v1.experiment_lock.verify_lock",
        lambda _path: lock_payload,
    )
    formal_output = tmp_path / "formal"
    arguments = [
        "--per-candidate",
        str(test_path),
        "--sample-universe",
        str(universe_path),
        "--training-root",
        str(output),
        "--experiment-lock",
        str(lock_path),
        "--output-root",
        str(formal_output),
        "--device",
        "cpu",
    ]
    assert apply_main(arguments) == 0
    assert apply_main(arguments) == 0
    formal_predictions = pd.read_parquet(
        formal_output / "reranker_predictions.parquet"
    )
    assert "candidate_positive" not in formal_predictions
    manifest_path = formal_output / "inference_manifest.json"
    tampered_manifest = json.loads(manifest_path.read_text())
    tampered_manifest["per_candidate"] = str(tmp_path / "different.parquet")
    manifest_path.write_text(json.dumps(tampered_manifest))
    with pytest.raises(ValueError, match="different inputs"):
        apply_main(arguments)


def test_rule_selection_is_validation_only_and_atomically_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "protected_run"
    run.mkdir()
    (run / ".RUN_ACTIVE").write_text("test\n", encoding="utf-8")
    validation = _synthetic_frame(
        split="validation", scenes=2, prefix="rule_validation_"
    )
    validation_path = run / "validation.parquet"
    validation.to_parquet(validation_path, index=False)
    output = run / "validation" / "rule_selection"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_rule_hyperparameters.py",
            "--validation-per-candidate",
            str(validation_path),
            "--output-root",
            str(output),
            "--tmp-root",
            str(run / "tmp"),
            "--grid-step",
            "0.5",
        ],
    )
    assert select_rule_main() == 0
    selection = json.loads(
        (output / "selection.json").read_text(encoding="utf-8")
    )
    assert selection["selection_split"] == "validation"
    assert Path(selection["validation_per_candidate"]) == validation_path
    assert selection["validation_per_candidate_sha256"] == sha256_file(
        validation_path
    )
    assert Path(selection["sweep"]) == output / "sweep.csv"
    assert selection["sweep_sha256"] == sha256_file(output / "sweep.csv")
