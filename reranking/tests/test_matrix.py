from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
import torch

from reranking import matrix as matrix_module
from reranking.matrix import (
    FORMAL_SEEDS,
    DatasetArtifact,
    ExperimentSpec,
    MatrixError,
    _build_nested_neural_split,
    _criterion,
    _make_neural,
    _specs,
    _test_manifest,
    run_matrix_stage,
    select_primary_configuration,
)
from reranking.reporting_bridge import run_reporting_bridge
from reranking.leakage_audit import verify_leakage_audit_bundle
from reranking.splits import build_stratified_group_folds
from reranking.models.neural import SharedMLPScorer


def _candidate_tables(
    prefix: str, query_count: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    for query_index in range(query_count):
        query_id = f"{prefix}-q{query_index:02d}"
        queries.append(
            {
                "sample_id": query_id,
                "scene_id": f"scene-{query_index:02d}",
                "frame_id": f"frame-{query_index:02d}",
            }
        )
        positive_index = query_index % 3
        no_positive = query_index in {5, 11}
        for candidate_index in range(3):
            correct = (candidate_index == positive_index) and not no_positive
            # q-only deliberately prefers candidate 0.  The fixed R1 signal is
            # strong enough to recover non-zero positives without fitting.
            q_raw = 0.50 - 0.005 * candidate_index
            features.append(
                {
                    "sample_id": query_id,
                    "scene_id": f"scene-{query_index:02d}",
                    "frame_id": f"frame-{query_index:02d}",
                    "candidate_id": f"c{candidate_index}",
                    "route": "toy",
                    "pool_type": "full_post_filter",
                    "q_raw": q_raw,
                    "original_rank": candidate_index + 1,
                    "x_px": float(10 * candidate_index + query_index),
                    "y_px": float(5 * candidate_index + query_index),
                    "angle_rad": float(candidate_index) * 0.2,
                    "width_m": 0.03 + 0.005 * candidate_index,
                    "width_px": 18.0 + 3.0 * candidate_index,
                    "mask_support": 1.0 if correct else 0.0,
                    "width_compatibility": 0.8 if correct else 0.2,
                    "depth_missing": float(candidate_index == 2),
                    "feature_reliability": 0.9,
                }
            )
            labels.append(
                {
                    "sample_id": query_id,
                    "candidate_id": f"c{candidate_index}",
                    "candidate_correct": int(correct),
                }
            )
    queries.append(
        {
            "sample_id": f"{prefix}-empty",
            "scene_id": f"{prefix}-empty-scene",
            "frame_id": f"{prefix}-empty-frame",
        }
    )
    return pd.DataFrame(features), pd.DataFrame(labels), pd.DataFrame(queries)


def test_batched_neural_prediction_matches_query_at_a_time_reference() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q0", "q0", "q1", "q2", "q2", "q2"],
            "candidate_id": ["a", "b", "a", "a", "b", "c"],
            "q_raw": [0.7, 0.2, 0.4, 0.8, 0.5, 0.1],
            "label": [1, 0, 0, 1, 0, 0],
        },
        index=[17, 3, 91, 22, 8, 44],
    )
    transformed = pd.DataFrame(
        {
            "feature_a": [0.2, -0.4, 0.1, 0.7, -0.3, 0.5],
            "feature_b": [1.0, 0.0, -1.0, 0.2, 0.4, -0.8],
            "q_platt_train_fold": [0.65, 0.25, 0.45, 0.75, 0.55, 0.15],
        },
        index=frame.index,
    ).iloc[::-1]
    torch.manual_seed(9)
    model = SharedMLPScorer(
        transformed.shape[1],
        hidden_dims=(8, 4),
        dropout=0.0,
        mode="direct",
    )

    expected_rows: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for _, indices in frame.groupby("query_id", sort=False).groups.items():
            positions = list(indices)
            scores = model(
                torch.tensor(transformed.loc[positions].to_numpy(np.float32)),
                baseline_scores=torch.tensor(
                    transformed.loc[positions, "q_platt_train_fold"].to_numpy(
                        np.float32
                    )
                ),
            ).numpy()
            expected_rows.append(
                frame.loc[positions, ["query_id", "candidate_id"]].assign(
                    score=scores.astype(np.float64)
                )
            )
    expected = pd.concat(expected_rows, ignore_index=True)

    actual = matrix_module._predict_neural(
        model,
        frame,
        transformed,
        torch.device("cpu"),
        query_batch_size=2,
    )
    pd.testing.assert_frame_equal(
        actual[["query_id", "candidate_id"]],
        expected[["query_id", "candidate_id"]],
    )
    np.testing.assert_allclose(actual["score"], expected["score"], rtol=1e-6, atol=1e-7)


def test_vectorized_examples_preserve_loc_alignment_and_group_order() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q0"],
            "label": [1, 0, 0],
            "q_raw": [0.8, 0.4, 0.2],
        },
        index=[20, 7, 12],
    )
    transformed = pd.DataFrame(
        {
            "feature": [2.0, 0.7, 1.2],
            "q_platt_train_fold": [0.82, 0.42, 0.22],
        },
        index=frame.index,
    ).iloc[[2, 0, 1]]
    raw_edges = pd.DataFrame(
        np.arange(30, dtype=np.float32).reshape(3, 10),
        index=frame.index,
    ).iloc[[1, 2, 0]]

    examples = matrix_module._examples(
        frame,
        transformed,
        raw_edge_inputs=raw_edges,
    )

    assert [example.query_id for example in examples] == ["q0", "q1"]
    np.testing.assert_allclose(examples[0].features.numpy(), [[2.0, 0.82], [1.2, 0.22]])
    np.testing.assert_array_equal(examples[0].labels.numpy(), [1.0, 0.0])
    np.testing.assert_allclose(examples[0].baseline_scores.numpy(), [0.82, 0.22])
    np.testing.assert_array_equal(
        examples[0].raw_edge_inputs.numpy(),
        raw_edges.loc[[20, 12]].to_numpy(np.float32),
    )


def test_vectorized_examples_fail_closed_on_missing_or_duplicate_indices() -> None:
    frame = pd.DataFrame(
        {"query_id": ["q0", "q1"], "label": [1, 0], "q_raw": [0.8, 0.2]},
        index=[4, 9],
    )
    missing = pd.DataFrame({"feature": [1.0], "q_platt_train_fold": [0.8]}, index=[4])
    duplicate = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "q_platt_train_fold": [0.8, 0.2, 0.3],
        },
        index=[4, 9, 9],
    )

    with pytest.raises(MatrixError, match="cover every candidate frame index"):
        matrix_module._examples(frame, missing)
    with pytest.raises(MatrixError, match="indices are not one-to-one"):
        matrix_module._examples(frame, duplicate)


def _initialize_output(root: Path) -> Path:
    output = root / "matrix-run"
    for directory in (
        "configs",
        "features",
        "data",
        "predictions",
        "metrics",
        "manifests",
        "checkpoints",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    (output / "configs" / "run_config.json").write_text(
        json.dumps(
            {
                "matrix_profile": "unit_test",
                "folds": 2,
                "seeds": [42],
                "device": "cpu",
                "matrix_epochs": 1,
                "matrix_patience": 1,
                "matrix_query_batch_size": 8,
            }
        ),
        encoding="utf-8",
    )
    development_features, development_labels, development_queries = _candidate_tables(
        "dev"
    )
    development_features.to_parquet(
        output / "features" / "candidates_toy_full_post_filter_development.parquet",
        index=False,
    )
    development_labels.to_parquet(
        output / "data" / "labels_toy_full_post_filter_development.parquet",
        index=False,
    )
    development_queries.to_parquet(
        output / "data" / "queries_toy_full_post_filter_development.parquet",
        index=False,
    )
    # These files deliberately cannot be parsed. Train and validation must
    # succeed without opening either held-out artifact; the fixture replaces
    # them before lock-primary freezes the label-free candidate identity.
    (output / "features" / "candidates_toy_full_post_filter_test.parquet").write_bytes(
        b"TEST FEATURES MUST NOT BE READ BEFORE LOCK"
    )
    (output / "data" / "labels_toy_full_post_filter_test.parquet").write_bytes(
        b"TEST LABELS MUST NOT BE READ BEFORE LOCK"
    )
    return output


def test_spec_hyperparameters_drive_loss_and_set_transformer() -> None:
    spec = ExperimentSpec(
        "custom",
        "R9",
        "custom",
        "set_transformer",
        "listwise",
        True,
        True,
        "set_transformer",
        {
            "hidden_dim": 32,
            "heads": 2,
            "blocks": 2,
            "dropout": 0.2,
            "residual_scale": 0.25,
            "bce_weight": 0.5,
            "listwise_weight": 1.0,
            "residual_weight": 0.01,
            "temperature": 2.0,
        },
    )
    criterion = _criterion(spec)
    model = _make_neural(spec, 7, "full_post_filter")
    assert criterion.bce_weight == pytest.approx(0.5)
    assert criterion.listwise_weight == pytest.approx(1.0)
    assert criterion.residual_weight == pytest.approx(0.01)
    assert criterion.temperature == pytest.approx(2.0)
    assert model.hidden_dim == 32
    assert len(model.blocks) == 2
    assert model.residual_scale == pytest.approx(0.25)


def test_neural_experiment_identity_includes_effective_training_protocol(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features.parquet"
    labels = tmp_path / "labels.parquet"
    features.write_bytes(b"frozen features")
    labels.write_bytes(b"frozen labels")
    artifact = DatasetArtifact(
        "toy_frozen_top5",
        "toy",
        "frozen_top5",
        "development",
        features,
        labels,
    )
    spec = next(
        item
        for item in matrix_module._specs("formal")
        if item.key == "r4_mlp_direct_bce_h64x32_d0p0"
    )
    common = {"matrix_profile": "formal"}
    cpu_protocol = matrix_module._neural_training_protocol(
        spec, {**common, "device": "cpu"}
    )
    mps_protocol = matrix_module._neural_training_protocol(
        spec, {**common, "device": "mps"}
    )

    cpu_identity = matrix_module._experiment_identity_sha256(
        artifact,
        spec,
        42,
        0,
        ("feature",),
        np.array([0, 1]),
        np.array([2]),
        cpu_protocol,
    )
    mps_identity = matrix_module._experiment_identity_sha256(
        artifact,
        spec,
        42,
        0,
        ("feature",),
        np.array([0, 1]),
        np.array([2]),
        mps_protocol,
    )

    assert cpu_protocol["effective_device"] == "cpu"
    assert cpu_protocol["train_query_batch_size"] == 128
    assert mps_protocol["requested_device"] == "mps"
    assert cpu_identity != mps_identity


def test_non_neural_experiment_identity_keeps_pre_protocol_payload(
    tmp_path: Path,
) -> None:
    features = tmp_path / "features.parquet"
    labels = tmp_path / "labels.parquet"
    features.write_bytes(b"features")
    labels.write_bytes(b"labels")
    artifact = DatasetArtifact(
        "toy",
        "toy",
        "top5",
        "development",
        features,
        labels,
    )
    spec = ExperimentSpec(
        "linear",
        "R2",
        "linear",
        "logistic",
        "bce",
        False,
        True,
        "linear",
        {},
    )
    train_indices = np.array([0, 2], dtype=np.int64)
    validation_indices = np.array([1], dtype=np.int64)
    payload = {
        "dataset": artifact.key,
        "route": artifact.route,
        "pool": artifact.pool,
        "split": artifact.split,
        "features_path": str(features.resolve()),
        "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
        "labels_path": str(labels.resolve()),
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
        "spec": spec.as_dict(),
        "seed": 42,
        "fold": 0,
        "feature_columns": ["feature"],
        "train_indices_sha256": hashlib.sha256(train_indices.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(
            validation_indices.tobytes()
        ).hexdigest(),
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    actual = matrix_module._experiment_identity_sha256(
        artifact,
        spec,
        42,
        0,
        ("feature",),
        train_indices,
        validation_indices,
    )

    assert actual == expected


def test_primary_selection_reaches_preregistered_75_percent_tier() -> None:
    baseline = {
        "method": "r0_q_baseline",
        "gate": "G0",
        "j_at_1": 0.50,
        "outcome_changing_precision": None,
    }
    learned = {
        "method": "r4_mlp_residual_bce",
        "gate": "G2",
        "j_at_1": 0.60,
        "outcome_changing_precision": 0.78,
    }

    assert select_primary_configuration([baseline, learned]) is learned


def test_primary_selection_prefers_80_percent_tier_boundary() -> None:
    baseline = {
        "method": "r0_q_baseline",
        "gate": "G0",
        "j_at_1": 0.50,
        "outcome_changing_precision": None,
    }
    preferred = {
        "method": "r2_logistic",
        "gate": "G2",
        "j_at_1": 0.55,
        "outcome_changing_precision": 0.80,
    }
    secondary = {
        "method": "r4_mlp_residual_bce",
        "gate": "G2",
        "j_at_1": 0.60,
        "outcome_changing_precision": 0.79,
    }

    assert select_primary_configuration([baseline, secondary, preferred]) is preferred


def test_primary_selection_falls_back_to_r0_below_75_percent() -> None:
    baseline = {
        "method": "r0_q_baseline",
        "gate": "G0",
        "j_at_1": 0.50,
        "outcome_changing_precision": None,
    }
    learned = {
        "method": "r4_mlp_residual_bce",
        "gate": "G2",
        "j_at_1": 0.60,
        "outcome_changing_precision": 0.749999,
    }

    assert select_primary_configuration([baseline, learned]) is baseline


def test_complete_test_manifest_hashes_its_exact_artifact_set(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest-output"
    for directory in ("manifests/experiments", "metrics", "predictions"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    prediction = output / "predictions" / "held-out.parquet"
    pd.DataFrame([{"query_id": "q", "candidate_id": "c", "score": 0.5}]).to_parquet(
        prediction, index=False
    )
    dataset = DatasetArtifact(
        key="toy",
        route="toy",
        pool="frozen_top5",
        split="test",
        features_path=output / "unused-features.parquet",
        labels_path=output / "unused-labels.parquet",
    )
    manifest_path = _test_manifest(
        output,
        stage="test-primary",
        dataset=dataset,
        method="r0_q_baseline",
        gate="G0",
        prediction=prediction,
        metrics={"j_at_1": 0.5},
        designation="LOCKED_PRIMARY_TEST_BASELINE",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(manifest["artifact_sha256"])
    assert manifest["artifact_sha256"][str(prediction.resolve())] == (
        hashlib.sha256(prediction.read_bytes()).hexdigest()
    )


def test_nested_neural_split_is_deterministic_grouped_and_outer_disjoint() -> None:
    features, labels, _ = _candidate_tables("nested", query_count=12)
    candidates = features.merge(
        labels,
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    ).rename(
        columns={
            "sample_id": "query_id",
            "candidate_correct": "label",
        }
    )
    outer_train = candidates.loc[
        candidates["query_id"].isin([f"nested-q{index:02d}" for index in range(8)])
    ].copy()
    outer_validation = candidates.loc[
        ~candidates["query_id"].isin(outer_train["query_id"])
    ].copy()
    config = {
        "matrix_profile": "unit_test",
        "matrix_inner_folds": 2,
        "matrix_inner_split_random_state": 1701,
    }
    first = _build_nested_neural_split(
        outer_train,
        outer_validation,
        outer_fold=1,
        config=config,
    )
    second = _build_nested_neural_split(
        outer_train,
        outer_validation,
        outer_fold=1,
        config=config,
    )
    assert first.train_indices.tolist() == second.train_indices.tolist()
    assert first.early_stop_indices.tolist() == second.early_stop_indices.tolist()
    audit = first.audit
    assert audit["split_scope"] == "outer_training_fold_only"
    assert audit["inner_partitions_exactly_cover_outer_train"] is True
    assert audit["all_inner_outer_overlaps_empty"] is True
    assert audit["outer_validation_usage"] == "single_oof_inference_and_metrics_only"
    assert (
        audit["inner_train"]["candidate_count"]
        + audit["inner_early_stop"]["candidate_count"]
        == audit["outer_train"]["candidate_count"]
    )
    assert all(
        count == 0
        for comparison in audit["intersections"].values()
        for count in comparison.values()
    )
    for partition in (
        "outer_train",
        "inner_train",
        "inner_early_stop",
        "outer_validation",
    ):
        for identity in ("query_ids", "group_hashes", "candidate_keys"):
            assert len(audit[partition][f"{identity}_sha256"]) == 64


def test_neural_early_stopping_loader_never_sees_outer_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _initialize_output(tmp_path)
    config_path = output / "configs" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["matrix_methods"] = ["r4_mlp_residual_bce"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    observed: dict[int, tuple[set[str], set[str]]] = {}
    real_fit = matrix_module.fit_neural_ranker

    def recording_fit(model, train_batches, validation_batches, **kwargs):
        metadata = kwargs["metadata"]
        train_query_ids = {
            train_batches.dataset[index].query_id
            for index in range(len(train_batches.dataset))
        }
        early_stop_query_ids = {
            validation_batches.dataset[index].query_id
            for index in range(len(validation_batches.dataset))
        }
        observed[int(metadata["fold"])] = (
            train_query_ids,
            early_stop_query_ids,
        )
        return real_fit(model, train_batches, validation_batches, **kwargs)

    monkeypatch.setattr(matrix_module, "fit_neural_ranker", recording_fit)
    run_matrix_stage("train", output)
    assignments = pd.read_parquet(
        output / "data" / "split_query_assignments_toy_full_post_filter.parquet"
    )
    assignments = assignments.loc[assignments["candidate_count"].gt(0)]
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "manifests" / "experiments").glob("*.json"))
        if "r4_mlp_residual_bce" in path.name
    ]
    assert len(manifests) == 2
    assert set(observed) == {0, 1}
    for manifest in manifests:
        fold = int(manifest["fold"])
        inner_train, inner_early_stop = observed[fold]
        outer_validation = set(
            assignments.loc[assignments["fold"].eq(fold), "query_id"].astype(str)
        )
        outer_train = set(assignments["query_id"].astype(str)) - outer_validation
        predicted = set(
            pd.read_parquet(manifest["prediction_path"])["query_id"].astype(str)
        )
        assert not inner_train & inner_early_stop
        assert inner_train | inner_early_stop == outer_train
        assert not (inner_train | inner_early_stop) & outer_validation
        assert predicted == outer_validation
        assert manifest["model_fit_scope"] == "nested_inner_train_only"
        assert manifest["early_stopping_scope"] == "nested_inner_early_stop_only"
        assert manifest["nested_validation"]["all_inner_outer_overlaps_empty"] is True
        assert manifest["score_calibration"]["baseline_fit_scope"] == (
            "nested_inner_train_only"
        )
        assert manifest["score_calibration"]["prediction_fit_scope"] == (
            "development_train_fold_only"
        )


def test_train_validate_write_concrete_fit_rows_without_reading_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _initialize_output(tmp_path)
    config_path = output / "configs" / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["matrix_methods"] = ["r2_logistic"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    real_read_parquet = matrix_module.pd.read_parquet
    opened: list[Path] = []

    def guarded_read(path, *args, **kwargs):
        resolved = Path(path).resolve()
        opened.append(resolved)
        if resolved.name.endswith("_test.parquet"):
            raise AssertionError(f"pre-lock stage opened held-out test: {resolved}")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(matrix_module.pd, "read_parquet", guarded_read)
    train_outputs = run_matrix_stage("train", output)
    run_matrix_stage("validate", output)

    assert not any(path.name.endswith("_test.parquet") for path in opened)
    assert not (output / "audit" / "LEAKAGE_AUDIT.md").exists()
    evidence_root = (
        output
        / "audit"
        / "leakage"
        / "development_fit_evidence"
        / "toy_full_post_filter"
    )
    evidence_paths = sorted(evidence_root.glob("outer_fold_*.parquet"))
    assert len(evidence_paths) == 2
    assert all(str(path.resolve()) in train_outputs for path in evidence_paths)
    for fold, path in enumerate(evidence_paths):
        evidence = real_read_parquet(path)
        assert set(evidence["outer_fold"]) == {fold}
        assert not evidence["source_validation_fold"].eq(fold).any()
        assert {
            "query_id",
            "group_identity_sha256",
            "candidate_key_sha256",
        } <= set(evidence)


def _write_modular_source_manifest(
    path: Path,
    queries: pd.DataFrame,
    *,
    split: str,
    media_root: Path,
    reused_media: tuple[Path, Path] | None = None,
) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    first_media: tuple[Path, Path] | None = None
    for index, query in enumerate(queries.itertuples(index=False)):
        if index == 0 and reused_media is not None:
            rgb_path, depth_path = reused_media
        else:
            rgb_path = media_root / f"{split}-{index}-rgb.bin"
            depth_path = media_root / f"{split}-{index}-depth.bin"
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_path.write_bytes(f"{split}-rgb-{index}".encode())
            depth_path.write_bytes(f"{split}-depth-{index}".encode())
        first_media = first_media or (rgb_path, depth_path)
        rows.append(
            {
                "sample_id": str(query.sample_id),
                "scene_id": str(query.scene_id),
                "split": split,
                "source_rgb_path": str(rgb_path),
                "source_rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                "source_depth_path": str(depth_path),
                "source_depth_sha256": hashlib.sha256(
                    depth_path.read_bytes()
                ).hexdigest(),
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert first_media is not None
    return first_media


def _formal_modular_leakage_fixture(
    tmp_path: Path, *, overlap: bool = False
) -> tuple[
    Path,
    dict[str, object],
    DatasetArtifact,
    DatasetArtifact,
]:
    output = tmp_path / "formal-leakage"
    for directory in ("features", "data", "audit"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    dev_features, dev_labels, dev_queries = _candidate_tables("formal-dev", 12)
    test_features, _, test_queries = _candidate_tables("formal-test", 4)
    for frame in (dev_features, dev_queries):
        frame["scene_id"] = "dev-" + frame["scene_id"].astype(str)
        frame["frame_id"] = frame["scene_id"]
    for frame in (test_features, test_queries):
        frame["scene_id"] = "test-" + frame["scene_id"].astype(str)
        frame["frame_id"] = frame["scene_id"]
    dev_feature_path = (
        output / "features" / "candidates_modular_full_post_filter_development.parquet"
    )
    test_feature_path = (
        output / "features" / "candidates_modular_full_post_filter_test.parquet"
    )
    dev_features.to_parquet(dev_feature_path, index=False)
    test_features.to_parquet(test_feature_path, index=False)
    poison_dev_labels = output / "data" / "development-labels-must-not-open.parquet"
    poison_test_labels = output / "data" / "test-labels-must-not-open.parquet"
    poison_dev_labels.write_bytes(b"DO NOT OPEN DEVELOPMENT LABELS IN LOCK AUDIT")
    poison_test_labels.write_bytes(b"DO NOT OPEN TEST LABELS IN LOCK AUDIT")
    development = DatasetArtifact(
        "modular_full_post_filter",
        "modular",
        "full_post_filter",
        "development",
        dev_feature_path,
        poison_dev_labels,
    )
    test = DatasetArtifact(
        "modular_full_post_filter",
        "modular",
        "full_post_filter",
        "test",
        test_feature_path,
        poison_test_labels,
    )
    joined = dev_features.merge(
        dev_labels.rename(columns={"candidate_correct": "label"}),
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    ).rename(columns={"sample_id": "query_id"})
    split_plan = build_stratified_group_folds(joined, n_splits=2, random_state=42)
    split_plan.candidate_assignments.to_parquet(
        output
        / "data"
        / "split_candidate_assignments_modular_full_post_filter.parquet",
        index=False,
    )
    matrix_module._write_outer_fit_evidence(output, development, joined, split_plan)
    media_root = tmp_path / "media"
    dev_manifest = tmp_path / "source" / "development.jsonl"
    test_manifest = tmp_path / "source" / "test.jsonl"
    dev_media = _write_modular_source_manifest(
        dev_manifest,
        dev_queries,
        split="train",
        media_root=media_root,
    )
    if overlap:
        test_queries.loc[0, ["scene_id", "frame_id"]] = dev_queries.loc[
            0, ["scene_id", "frame_id"]
        ].to_numpy()
        test_features.loc[
            test_features["sample_id"].eq(test_queries.loc[0, "sample_id"]),
            ["scene_id", "frame_id"],
        ] = test_queries.loc[0, ["scene_id", "frame_id"]].to_numpy()
        test_features.to_parquet(test_feature_path, index=False)
    _write_modular_source_manifest(
        test_manifest,
        test_queries,
        split="test",
        media_root=media_root,
        reused_media=dev_media if overlap else None,
    )
    config: dict[str, object] = {
        "matrix_profile": "formal",
        "folds": 2,
        "seeds": [42],
        "modular_source_identity_manifests": {
            "development": [str(dev_manifest)],
            "test": [str(test_manifest)],
        },
    }
    return output, config, development, test


def test_formal_lock_leakage_bundle_uses_label_free_source_and_outer_fit_rows(
    tmp_path: Path,
) -> None:
    output, config, development, test = _formal_modular_leakage_fixture(tmp_path)

    artifacts, record = matrix_module._write_formal_leakage_bundle(
        output, config, [development], [test]
    )

    assert record["status"] == "PASS"
    assert record["test_labels_opened"] is False
    assert len(record["development_fit_evidence"]) == 2
    assert record["source_evidence"]["unique_media_files_verified_once"] is True
    assert output / "audit" / "LEAKAGE_AUDIT.md" in artifacts
    verified = verify_leakage_audit_bundle(output / "audit" / "leakage")
    assert verified.passed
    assert verified.summary["identity_overlap_counts"] == {
        "query_id": 0,
        "scene_id": 0,
        "frame_identity": 0,
        "rgb_sha256": 0,
        "depth_sha256": 0,
    }
    assert not development.labels_path.read_bytes().startswith(b"PAR1")
    assert not test.labels_path.read_bytes().startswith(b"PAR1")


def test_formal_lock_leakage_bundle_rejects_tampered_fit_membership(
    tmp_path: Path,
) -> None:
    output, config, development, test = _formal_modular_leakage_fixture(tmp_path)
    evidence_path = (
        output
        / "audit"
        / "leakage"
        / "development_fit_evidence"
        / development.key
        / "outer_fold_0.parquet"
    )
    evidence = pd.read_parquet(evidence_path)
    evidence.loc[0, "candidate_key_sha256"] = "0" * 64
    evidence.to_parquet(evidence_path, index=False)

    with pytest.raises(MatrixError, match="membership changed|receipt mismatch"):
        matrix_module._write_formal_leakage_bundle(
            output, config, [development], [test]
        )


def test_formal_lock_leakage_bundle_fails_on_source_entity_overlap(
    tmp_path: Path,
) -> None:
    output, config, development, test = _formal_modular_leakage_fixture(
        tmp_path, overlap=True
    )

    with pytest.raises(MatrixError, match="leakage audit failed"):
        matrix_module._write_formal_leakage_bundle(
            output, config, [development], [test]
        )
    report = (output / "audit" / "LEAKAGE_AUDIT.md").read_text(encoding="utf-8")
    assert "**Overall result: FAIL**" in report
    assert "| `scene_id` | 1 | FAIL |" in report
    assert "| `rgb_sha256` | 1 | FAIL |" in report


def test_crog_formal_identity_is_derived_from_built_candidate_media(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "crog-rgb.bin"
    depth = tmp_path / "crog-depth.bin"
    rgb.write_bytes(b"crog-rgb")
    depth.write_bytes(b"crog-depth")
    feature_path = tmp_path / "crog-development.parquet"
    pd.DataFrame(
        {
            "sample_id": ["crog:train:00000001", "crog:train:00000001"],
            "candidate_id": ["a", "b"],
            "scene_id": ["scene", "scene"],
            "frame_id": ["frame", "frame"],
            "image_path": [str(rgb), str(rgb)],
            "depth_path": [str(depth), str(depth)],
        }
    ).to_parquet(feature_path, index=False)
    poison_labels = tmp_path / "labels-must-not-open.parquet"
    poison_labels.write_bytes(b"NOT PARQUET")
    artifact = DatasetArtifact(
        "crog_frozen_top5",
        "crog",
        "frozen_top5",
        "development",
        feature_path,
        poison_labels,
    )
    cache: dict[Path, str] = {}

    identities = matrix_module._crog_identity_rows(artifact, cache)

    assert len(identities) == 1
    assert identities.loc[0, "query_id"] == "crog:train:00000001"
    assert identities.loc[0, "rgb_sha256"] == hashlib.sha256(b"crog-rgb").hexdigest()
    assert (
        identities.loc[0, "depth_sha256"] == hashlib.sha256(b"crog-depth").hexdigest()
    )
    assert len(cache) == 2
    assert poison_labels.read_bytes() == b"NOT PARQUET"


@pytest.fixture(scope="module")
def completed_matrix(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = _initialize_output(tmp_path_factory.mktemp("matrix"))
    train_outputs = run_matrix_stage("train", output)
    assert train_outputs
    validate_outputs = run_matrix_stage("validate", output)
    assert validate_outputs

    # The lock freezes the label-free held-out feature pool and query universe
    # before prediction. Train/validate above still proved that the deliberately
    # invalid placeholders were never opened.
    test_features, test_labels, test_queries = _candidate_tables("test", query_count=6)
    test_features.to_parquet(
        output / "features" / "candidates_toy_full_post_filter_test.parquet",
        index=False,
    )
    test_labels.to_parquet(
        output / "data" / "labels_toy_full_post_filter_test.parquet",
        index=False,
    )
    test_queries.to_parquet(
        output / "data" / "queries_toy_full_post_filter_test.parquet",
        index=False,
    )
    lock_outputs = run_matrix_stage("lock-primary", output)
    assert lock_outputs == [
        str((output / "manifests" / "PRIMARY_METHOD_LOCK.json").resolve())
    ]
    run_matrix_stage("test-primary", output)
    run_matrix_stage("test-post-lock", output)
    return output


def test_train_registry_covers_matrix_and_is_individually_resumable(
    completed_matrix: Path,
) -> None:
    registry_path = completed_matrix / "metrics" / "experiment_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["experiments"]
    methods = {row["method"] for row in registry if row.get("stage") == "train"}
    assert {
        "r0_q_baseline",
        "r1_fixed_rule",
        "r2_logistic",
        "r2_linear_residual",
        "r2_linear_ranknet",
        "r3_random_forest",
        "r3_histgb_fallback",
        "r3_lambdamart",
        "r4_mlp_direct_bce",
        "r4_mlp_residual_bce",
        "r5_mlp_residual_ranknet",
        "r6_mlp_residual_listwise",
        "r7_deepsets_residual",
        "r8_candidate_gnn_residual",
        "r9_set_transformer_residual",
    }.issubset(methods)
    lambda_rows = [
        row
        for row in registry
        if row.get("method") == "r3_lambdamart" and row.get("stage") == "train"
    ]
    assert len(lambda_rows) == 2
    assert {row["status"] for row in lambda_rows} == {"COMPLETE"}
    assert {row["spec"]["hyperparameters"]["objective"] for row in lambda_rows} == {
        "rank:ndcg"
    }
    assert len({row["experiment_id"] for row in registry}) == len(registry)

    shared = {
        row["method"]: row["spec"]
        for row in registry
        if row.get("method")
        in {
            "r4_mlp_residual_bce",
            "r5_mlp_residual_ranknet",
            "r6_mlp_residual_listwise",
        }
        and row.get("status") == "COMPLETE"
        and row.get("stage") == "train"
    }
    assert len(shared) == 3
    assert {value["scorer_family"] for value in shared.values()} == {"shared_mlp"}
    assert (
        len(
            {
                json.dumps(value["hyperparameters"], sort_keys=True)
                for value in shared.values()
            }
        )
        == 1
    )
    neural_rows = [
        row
        for row in registry
        if row.get("status") == "COMPLETE"
        and row.get("spec", {}).get("backend")
        in {"mlp", "deepsets", "gnn", "set_transformer"}
    ]
    assert any(
        row["score_calibration"]["neural_preparation_cache_hit"] for row in neural_rows
    )
    calibrator_hashes_by_fold: dict[int, set[str]] = {}
    for row in neural_rows:
        bundle = json.loads(Path(row["bundle_path"]).read_text(encoding="utf-8"))
        calibrator_hashes_by_fold.setdefault(int(row["fold"]), set()).add(
            bundle["score_calibrator_sha256"]
        )
    assert all(len(hashes) == 1 for hashes in calibrator_hashes_by_fold.values())

    prediction = next(
        Path(row["prediction_path"])
        for row in registry
        if row.get("status") == "COMPLETE" and row.get("stage") == "train"
    )
    modified = prediction.stat().st_mtime_ns
    time.sleep(0.01)
    run_matrix_stage("train", completed_matrix)
    assert prediction.stat().st_mtime_ns == modified


def test_formal_grid_contains_true_lambdamart_and_exact_unique_coverage() -> None:
    specs = _specs("formal")
    assert len(specs) == 104
    assert len({spec.key for spec in specs}) == len(specs)
    lambdamart = next(spec for spec in specs if spec.key == "r3_lambdamart")
    assert lambdamart.backend == "lambdamart"
    assert lambdamart.hyperparameters["objective"] == "rank:ndcg"
    assert lambdamart.hyperparameters["lambda_mart"] is True
    loss_keys = {spec.key for spec in specs if spec.rung == "A_LOSS"}
    assert loss_keys == {
        "loss_mlp_ranknet_pure",
        "loss_mlp_bce_ranknet",
        "loss_mlp_ranknet_regularized",
        "loss_mlp_listwise_pure",
        "loss_mlp_listwise_bce",
    }
    spatial = next(spec for spec in specs if spec.key == "r8_gnn_spatial_edges")
    spatial_angle = next(
        spec for spec in specs if spec.key == "r8_gnn_spatial_angle_edges"
    )
    spatial_angle_overlap = next(
        spec for spec in specs if spec.key == "r8_gnn_spatial_angle_overlap_edges"
    )
    assert spatial.hyperparameters["edge_feature_indices"] == [0, 1, 2]
    assert spatial_angle.hyperparameters["edge_feature_indices"] == [0, 1, 2, 3, 4]
    assert spatial_angle_overlap.hyperparameters["edge_feature_indices"] == [
        0,
        1,
        2,
        3,
        4,
        7,
        8,
    ]


def test_validation_uses_oof_seeds_and_all_gates(completed_matrix: Path) -> None:
    payload = json.loads(
        (completed_matrix / "metrics" / "validation_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["fit_scope"] == "development_grouped_oof_only"
    assert payload["gates_evaluated"] == ["G0", "G1", "G2", "G3"]
    assert payload["test_rows_read"] is False
    assert payload["primaries"]
    results = pd.read_parquet(
        completed_matrix / "metrics" / "all_validation_results.parquet"
    )
    ensemble_gates = set(results.loc[results["level"].eq("ensemble"), "gate"])
    assert {"G0", "G1", "G2", "G3"}.issubset(ensemble_gates)
    assert (results["query_count"] == 13).any()  # includes the valid-empty query


def test_lock_precedes_test_and_post_lock_cannot_reselect(
    completed_matrix: Path,
) -> None:
    lock_path = completed_matrix / "manifests" / "PRIMARY_METHOD_LOCK.json"
    before = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "LOCKED_BEFORE_TEST_PRIMARY"
    assert lock["test_inputs_materialized_before_lock"] is True
    assert lock["test_inputs_used_for_primary_selection"] is False
    assert lock["test_predictions_generated_before_lock"] is False
    assert lock["test_prediction_artifacts_at_lock"] == []
    assert lock["primaries"][0]["folds"] == 2
    assert lock["primaries"][0]["seeds"] == [42]
    assert lock["test_feature_identity_frozen_before_prediction"] is True
    assert lock["test_labels_opened_at_lock"] is False
    assert lock["leakage_audit"] == {
        "status": "SKIPPED_UNIT_TEST_PROFILE",
        "reason": "synthetic unit-test datasets have no canonical source-media contract",
        "development_fit_evidence_written": True,
        "test_labels_opened": False,
    }
    assert len(lock["held_out_test_inputs"]) == 1
    held_out = lock["held_out_test_inputs"][0]
    assert held_out["candidate_count"] == 18
    assert held_out["query_universe_count"] == 7
    assert len(held_out["candidate_pool_identity_sha256"]) == 64
    training_lock = lock["primaries"][0]["training_artifact_lock"]
    assert training_lock["manifest_count"] == len(training_lock["manifests"])
    assert training_lock["manifests"]

    primary_predictions = sorted(
        (completed_matrix / "predictions").glob("test_primary__*.parquet")
    )
    assert len(primary_predictions) == 2
    primary_summary = json.loads(
        (completed_matrix / "metrics" / "primary_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert {row["designation"] for row in primary_summary["results"]} == {
        "LOCKED_PRIMARY_TEST_BASELINE",
        "LOCKED_PRIMARY_TEST",
    }
    post = json.loads(
        (completed_matrix / "metrics" / "post_lock_test_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert post["designation"] == "POST_LOCK_COMPARATIVE_ONLY"
    assert post["primary_reselection_permitted"] is False
    assert post["results"]
    assert all(
        row["eligible_for_primary_reselection"] is False for row in post["results"]
    )
    assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == before


def test_test_stages_reject_locked_training_or_held_out_input_tampering(
    completed_matrix: Path,
) -> None:
    lock = json.loads(
        (completed_matrix / "manifests/PRIMARY_METHOD_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    artifact = Path(
        lock["primaries"][0]["training_artifact_lock"]["manifests"][0]["artifacts"][0][
            "path"
        ]
    )
    original_artifact = artifact.read_bytes()
    try:
        artifact.write_bytes(b"tampered selected training artifact\n")
        with pytest.raises(MatrixError, match="locked artifact hash mismatch"):
            run_matrix_stage("test-primary", completed_matrix)
    finally:
        artifact.write_bytes(original_artifact)

    feature_path = Path(lock["held_out_test_inputs"][0]["features"]["path"])
    original_features = feature_path.read_bytes()
    try:
        feature_path.write_bytes(b"tampered held-out pool\n")
        with pytest.raises(MatrixError, match="locked artifact hash mismatch"):
            run_matrix_stage("test-post-lock", completed_matrix)
    finally:
        feature_path.write_bytes(original_features)


def test_formal_profile_rejects_nonformal_fold_seed_protocol(tmp_path: Path) -> None:
    output = tmp_path / "formal-invalid"
    (output / "configs").mkdir(parents=True)
    (output / "configs" / "run_config.json").write_text(
        json.dumps({"matrix_profile": "formal", "folds": 2, "seeds": [42]}),
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="5 grouped folds"):
        run_matrix_stage("train", output)
    assert FORMAL_SEEDS == (42, 123, 2026)

    (output / "configs" / "run_config.json").write_text(
        json.dumps(
            {
                "matrix_profile": "formal",
                "folds": 5,
                "seeds": list(FORMAL_SEEDS),
                "matrix_methods": ["r0_q_baseline"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MatrixError, match="complete frozen matrix"):
        run_matrix_stage("train", output)


def test_atomic_manifest_resume_does_not_require_eager_registry_rebuild(
    tmp_path: Path,
) -> None:
    (tmp_path / "manifests" / "experiments").mkdir(parents=True)
    (tmp_path / "metrics").mkdir()
    artifact = tmp_path / "prediction.parquet"
    artifact.write_bytes(b"immutable prediction")
    artifact_path = str(artifact.resolve())
    experiment_id = "dataset=toy__method=rule__seed=42__fold=0"
    identity = "a" * 64
    manifest = matrix_module._write_manifest(
        tmp_path,
        experiment_id,
        {
            "status": "COMPLETE",
            "experiment_identity_sha256": identity,
            "artifacts": [artifact_path],
            "artifact_sha256": {
                artifact_path: hashlib.sha256(artifact.read_bytes()).hexdigest()
            },
        },
    )

    assert manifest.is_file()
    assert not (tmp_path / "metrics" / "experiment_registry.json").exists()
    assert (
        matrix_module._completed_manifest(
            tmp_path,
            experiment_id,
            expected_identity_sha256=identity,
        )
        is not None
    )

    registry_json, registry_parquet = matrix_module._write_registry(tmp_path)
    assert registry_json.is_file()
    assert registry_parquet.is_file()
    rows = json.loads(registry_json.read_text(encoding="utf-8"))["experiments"]
    assert [row["experiment_id"] for row in rows] == [experiment_id]


def test_id_join_failure_is_not_recorded_as_success(tmp_path: Path) -> None:
    output = _initialize_output(tmp_path)
    labels_path = output / "data" / "labels_toy_full_post_filter_development.parquet"
    labels = pd.read_parquet(labels_path).iloc[:-1]
    labels.to_parquet(labels_path, index=False)
    with pytest.raises(MatrixError, match="key sets differ"):
        run_matrix_stage("train", output)
    registry = output / "metrics" / "experiment_registry.json"
    if registry.exists():
        assert not any(
            row.get("status") == "COMPLETE"
            for row in json.loads(registry.read_text(encoding="utf-8"))["experiments"]
        )


def test_reporting_bridge_materializes_query_contract_and_formal_outputs(
    completed_matrix: Path,
) -> None:
    statistics = run_reporting_bridge("statistics", completed_matrix)
    assert statistics
    visualize = run_reporting_bridge("visualize", completed_matrix)
    assert visualize
    reports = run_reporting_bridge("report", completed_matrix)
    assert reports
    assert (completed_matrix / "statistics" / "mcnemar_results.csv").is_file()
    assert (completed_matrix / "figures" / "switch_analysis.png").is_file()
    assert (completed_matrix / "galleries" / "index.html").is_file()
    assert (completed_matrix / "reports" / "FINAL_REPORT_ZH.md").is_file()
    assert (completed_matrix / "checksums.sha256").is_file()
    inputs = sorted(
        (completed_matrix / "reporting_inputs").glob("*/predictions/*.parquet")
    )
    assert inputs
    outcome = pd.read_parquet(inputs[0])
    assert {
        "sample_id",
        "baseline_correct",
        "selected_correct",
        "oracle",
        "candidate_count",
        "positive_count",
        "first_positive_rank",
        "candidate_pool_json",
        "feature_delta_json",
        "gate_id",
        "switch_applied",
        "failure_stage",
    } <= set(outcome.columns)
    assert len(outcome) == 7  # six non-empty test queries plus the frozen empty query
