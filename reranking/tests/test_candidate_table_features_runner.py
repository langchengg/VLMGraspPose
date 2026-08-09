from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from reranking.build_candidate_table import (
    CandidateTableError,
    build_crog_candidate_tables,
    build_modular_candidate_tables,
)
from reranking.data_contracts import REPO_ROOT
from reranking.extract_features import (
    FeatureAuditError,
    audit_feature_table,
    select_crog_native_feature_columns,
)
from reranking.publish_modular_development import (
    PublishError,
    publish_modular_development,
)
from reranking.run_experiment_matrix import (
    STAGES,
    StageError,
    _modular_baseline,
    _initialize_run,
    _safe_output_path,
    _stage_completed,
    _stage_paths,
    _run_stage,
    _write_resumable_audit_bundle,
    _stream_crog_baseline,
    _verify_modular_development_source,
    _write_run_success,
)


def _runner_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "routes": ["crog", "modular"],
        "pools": ["top5", "full"],
        "folds": 5,
        "seeds": [42, 123, 2026],
        "device": "cpu",
        "neural_query_batch_size": 128,
        "bootstrap_iterations": 10_000,
        "no_hash_audit": False,
        "stage": "all",
        "resume": False,
        "force_rerun_specific_experiment": [],
        "amend_prelock_device": False,
        "amend_prelock_training": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _initialize_runner_config(root: Path) -> None:
    _initialize_run(root, _runner_args())


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_resumed_provenance_audit_preserves_peer_evidence(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    peer = audit_dir / "TEST_CLIP_FEATURE_AUDIT.md"
    peer.write_text("peer evidence\n", encoding="utf-8")

    written = _write_resumable_audit_bundle(
        {"schema_version": 1, "passed": True},
        audit_dir,
        resume=True,
        source_roots=(),
    )

    assert peer.read_text(encoding="utf-8") == "peer evidence\n"
    assert Path(written["json"]) == audit_dir / "audit.json"
    assert (audit_dir / "audit.json").is_file()
    assert (audit_dir / "audit.md").is_file()
    assert (audit_dir / "inventory.tsv").is_file()


def _publisher_sources(
    feature_path: Path, candidate_path: Path
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    features = pd.read_parquet(feature_path)
    candidates = pd.read_parquet(candidate_path)
    observed = (
        features.groupby("sample_id", sort=True)
        .size()
        .rename("candidate_count")
        .reset_index()
    )
    evidence = feature_path.parent / "per_sample.parquet"
    pd.concat(
        [
            observed,
            pd.DataFrame([{"sample_id": "__zero_val", "candidate_count": 0}]),
        ],
        ignore_index=True,
    ).to_parquet(evidence, index=False)
    feature_digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (feature_path.parent / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "candidate_count": len(features),
                "sample_count": len(observed) + 1,
                "per_candidate_sha256": feature_digest,
                "per_sample_sha256": evidence_digest,
            }
        ),
        encoding="utf-8",
    )
    candidate_digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    (candidate_path.parent / "run_config.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "candidate_stage_artifacts": {
                    "nms": {
                        "path": str(candidate_path.resolve()),
                        "rows": len(candidates),
                        "sha256": candidate_digest,
                    }
                },
                "counts": {"samples": len(observed) + 1},
            }
        ),
        encoding="utf-8",
    )
    train_universe = feature_path.parent / "train_universe.jsonl"
    scenes = (
        features.groupby("sample_id", sort=True)["scene_id"].first()
        if "scene_id" in features
        else pd.Series("scene", index=observed["sample_id"])
    )
    _write_jsonl(
        train_universe,
        [
            {
                "sample_id": str(sample_id),
                "split": "train",
                "scene_id": str(scenes.loc[sample_id]),
            }
            for sample_id in observed["sample_id"]
        ],
    )
    val_universe = feature_path.parent / "val_universe.jsonl"
    _write_jsonl(
        val_universe,
        [{"sample_id": "__zero_val", "split": "val", "scene_id": "scene-val"}],
    )
    return (train_universe, val_universe), (evidence,)


def _crog_candidate(
    candidate_id: str,
    rank: int,
    q: float,
    *,
    checksum: str | None = None,
    extra_features: dict[str, object] | None = None,
) -> dict[str, object]:
    features: dict[str, object] = {
        "soft_coverage": {"value": 0.4 + rank / 10, "reliability": 0.9},
        "optional_depth": {
            "value": None,
            "reliability": 0.0,
            "missing_reason": "not_available",
        },
    }
    if extra_features:
        features.update(extra_features)
    return {
        "candidate_id": candidate_id,
        "candidate_checksum": checksum or (str(rank + 1) * 64),
        "q_rank": rank,
        "q_raw": q,
        "cx": 10.0 + rank,
        "cy": 20.0 + rank,
        "angle_rad": 0.1 * rank,
        "width_px": 30.0,
        "height_px": 12.0,
        "features": features,
        "diagnostics": {"center_depth_m": None, "angle_concentration": 0.5},
    }


def _crog_sample(candidates: list[dict[str, object]]) -> dict[str, object]:
    return {"sample_id": "sample-1", "scene_id": "scene-1", "candidates": candidates}


def _crog_labels(
    candidates: list[dict[str, object]],
    *,
    reverse: bool = False,
) -> dict[str, object]:
    labels = [
        {
            "candidate_id": candidate["candidate_id"],
            "candidate_checksum": candidate["candidate_checksum"],
            "candidate_correct": index == 0,
            "best_gt": {
                "rectangle_iou": 0.8 if index == 0 else 0.1,
                "angle_difference_deg": 5.0 if index == 0 else 60.0,
            },
        }
        for index, candidate in enumerate(candidates)
    ]
    if reverse:
        labels.reverse()
    return {"sample_id": "sample-1", "candidate_labels": labels}


@pytest.mark.parametrize("mismatch", ["sample_id", "candidate_id", "checksum"])
def test_crog_rejects_feature_label_identity_and_checksum_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    candidates = [_crog_candidate("c0", 0, 0.8), _crog_candidate("c1", 1, 0.4)]
    feature_sample = _crog_sample(candidates)
    label_sample = _crog_labels(candidates)
    if mismatch == "sample_id":
        label_sample["sample_id"] = "other-sample"
    elif mismatch == "candidate_id":
        label_sample["candidate_labels"][0]["candidate_id"] = "other-candidate"
    else:
        label_sample["candidate_labels"][0]["candidate_checksum"] = "f" * 64
    features_jsonl = tmp_path / "features.jsonl"
    labels_jsonl = tmp_path / "labels.jsonl"
    _write_jsonl(features_jsonl, [feature_sample])
    _write_jsonl(labels_jsonl, [label_sample])

    with pytest.raises(CandidateTableError, match="mismatch|missing"):
        build_crog_candidate_tables(
            features_jsonl,
            tmp_path / "features.parquet",
            labels_jsonl=labels_jsonl,
            output_labels=tmp_path / "labels.parquet",
        )


def test_crog_nested_features_labels_and_q_ties_remain_separate(tmp_path: Path) -> None:
    candidates = [_crog_candidate("c0", 0, 0.8), _crog_candidate("c1", 1, 0.8)]
    features_jsonl = tmp_path / "features.jsonl"
    labels_jsonl = tmp_path / "labels.jsonl"
    _write_jsonl(features_jsonl, [_crog_sample(candidates)])
    _write_jsonl(labels_jsonl, [_crog_labels(candidates)])
    result = build_crog_candidate_tables(
        features_jsonl,
        tmp_path / "features.parquet",
        labels_jsonl=labels_jsonl,
        output_labels=tmp_path / "labels.parquet",
    )

    features = pd.read_parquet(result.features_path)
    labels = pd.read_parquet(result.labels_path)
    assert set(
        features[["sample_id", "candidate_id"]].itertuples(index=False, name=None)
    ) == set(labels[["sample_id", "candidate_id"]].itertuples(index=False, name=None))
    assert "candidate_correct" not in features.columns
    assert "best_iou_same_gt" not in features.columns
    assert {"candidate_correct", "best_iou_same_gt"} <= set(labels.columns)
    assert features["optional_depth_missing"].tolist() == [1, 1]
    assert features["optional_depth_reliability"].tolist() == [0.0, 0.0]
    assert features["q_percentile_within_query"].nunique() == 1
    assert features["top1_margin"].eq(0.0).all()
    assert features["score_prominence"].eq(0.0).all()
    q_columns = [
        "q_clipped",
        "q_log",
        "q_logit",
        "q_zscore_within_query",
        "q_percentile_within_query",
        "score_entropy",
        "score_concentration",
    ]
    assert np.isfinite(features[q_columns].to_numpy(np.float64)).all()
    audit = audit_feature_table(features, labels)
    assert audit["forbidden_column_scanner_passed"] is True


def test_crog_uses_nested_joint_success_and_rejects_conflicting_labels(
    tmp_path: Path,
) -> None:
    candidates = [_crog_candidate("c0", 0, 0.8), _crog_candidate("c1", 1, 0.4)]
    label_sample = _crog_labels(candidates)
    for label in label_sample["candidate_labels"]:
        correct = bool(label.pop("candidate_correct"))
        label["best_gt"]["joint_success"] = correct

    features_jsonl = tmp_path / "features.jsonl"
    labels_jsonl = tmp_path / "labels.jsonl"
    _write_jsonl(features_jsonl, [_crog_sample(candidates)])
    _write_jsonl(labels_jsonl, [label_sample])
    result = build_crog_candidate_tables(
        features_jsonl,
        tmp_path / "features.parquet",
        labels_jsonl=labels_jsonl,
        output_labels=tmp_path / "labels.parquet",
    )
    labels = pd.read_parquet(result.labels_path)
    assert labels["candidate_correct"].tolist() == [True, False]
    assert labels["pool_has_positive"].tolist() == [True, True]
    assert labels["baseline_top1_correct"].tolist() == [True, True]

    label_sample["candidate_labels"][0]["candidate_correct"] = False
    _write_jsonl(labels_jsonl, [label_sample])
    with pytest.raises(CandidateTableError, match="conflicting CROG correctness"):
        build_crog_candidate_tables(
            features_jsonl,
            tmp_path / "conflict_features.parquet",
            labels_jsonl=labels_jsonl,
            output_labels=tmp_path / "conflict_labels.parquet",
        )

    label_sample["candidate_labels"][0].pop("candidate_correct")
    label_sample["candidate_labels"][0]["best_gt"].pop("joint_success")
    _write_jsonl(labels_jsonl, [label_sample])
    with pytest.raises(CandidateTableError, match="missing CROG correctness"):
        build_crog_candidate_tables(
            features_jsonl,
            tmp_path / "missing_features.parquet",
            labels_jsonl=labels_jsonl,
            output_labels=tmp_path / "missing_labels.parquet",
        )


def test_crog_native_schema_excludes_depth_collision_and_contact_proxies() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["q"],
            "candidate_id": ["c"],
            "q_raw": [0.5],
            "soft_coverage": [0.8],
            "contact_probability_min": [0.7],
            "contact_depth_difference_m": [0.01],
            "z_m": [0.4],
            "border_clearance_px": [10.0],
            "collision_proxy": [0.2],
            "safety": [0.9],
            "nearest_obstacle_distance_px": [4.0],
        }
    )
    selected = set(select_crog_native_feature_columns(frame))
    assert {"q_raw", "soft_coverage", "contact_probability_min"} <= selected
    assert (
        not {
            "contact_depth_difference_m",
            "z_m",
            "border_clearance_px",
            "collision_proxy",
            "safety",
            "nearest_obstacle_distance_px",
        }
        & selected
    )


def test_crog_corrected_labels_join_source_id_and_emit_stable_id(
    tmp_path: Path,
) -> None:
    candidates = [_crog_candidate("c0", 0, 0.8)]
    feature_sample = _crog_sample(candidates)
    feature_sample["sample_id"] = 17
    label_sample = _crog_labels(candidates)
    label_sample.update(
        {
            "sample_id": "multiple:test:00000017",
            "source_sample_id": 17,
        }
    )
    features_jsonl = tmp_path / "features.jsonl"
    labels_jsonl = tmp_path / "labels.jsonl"
    _write_jsonl(features_jsonl, [feature_sample])
    _write_jsonl(labels_jsonl, [label_sample])
    result = build_crog_candidate_tables(
        features_jsonl,
        tmp_path / "features.parquet",
        labels_jsonl=labels_jsonl,
        output_labels=tmp_path / "labels.parquet",
    )
    features = pd.read_parquet(result.features_path)
    labels = pd.read_parquet(result.labels_path)
    assert features["sample_id"].tolist() == ["multiple:test:00000017"]
    assert labels["sample_id"].tolist() == ["multiple:test:00000017"]


def test_crog_split_local_integer_ids_are_namespaced(tmp_path: Path) -> None:
    candidate = _crog_candidate("c0", 0, 0.8)
    feature_sample = _crog_sample([candidate])
    feature_sample.update({"sample_id": 0, "split": "train"})
    label_sample = _crog_labels([candidate])
    label_sample["sample_id"] = 0
    features_jsonl = tmp_path / "features.jsonl"
    labels_jsonl = tmp_path / "labels.jsonl"
    _write_jsonl(features_jsonl, [feature_sample])
    _write_jsonl(labels_jsonl, [label_sample])
    result = build_crog_candidate_tables(
        features_jsonl,
        tmp_path / "features.parquet",
        labels_jsonl=labels_jsonl,
        output_labels=tmp_path / "labels.parquet",
    )
    assert pd.read_parquet(result.features_path)["sample_id"].tolist() == [
        "crog:train:00000000"
    ]


def test_crog_rejects_forbidden_label_name_nested_as_candidate_feature(
    tmp_path: Path,
) -> None:
    candidate = _crog_candidate(
        "c0",
        0,
        0.8,
        extra_features={"is_correct": {"value": 1.0, "reliability": 1.0}},
    )
    features_jsonl = tmp_path / "features.jsonl"
    labels_jsonl = tmp_path / "labels.jsonl"
    _write_jsonl(features_jsonl, [_crog_sample([candidate])])
    _write_jsonl(labels_jsonl, [_crog_labels([candidate])])
    with pytest.raises(CandidateTableError, match="forbidden|label"):
        build_crog_candidate_tables(
            features_jsonl,
            tmp_path / "features.parquet",
            labels_jsonl=labels_jsonl,
            output_labels=tmp_path / "labels.parquet",
        )


def test_crog_top1_label_fallback_is_id_ranked_not_label_list_order(
    tmp_path: Path,
) -> None:
    candidates = [_crog_candidate("top", 0, 0.9), _crog_candidate("second", 1, 0.2)]
    features_jsonl = tmp_path / "features.jsonl"
    labels_jsonl = tmp_path / "labels.jsonl"
    _write_jsonl(features_jsonl, [_crog_sample(candidates)])
    # No original_top1_correct field: fallback must resolve the rank-0 candidate
    # by ID, not trust candidate_labels array position.
    _write_jsonl(labels_jsonl, [_crog_labels(candidates, reverse=True)])
    result = build_crog_candidate_tables(
        features_jsonl,
        tmp_path / "features.parquet",
        labels_jsonl=labels_jsonl,
        output_labels=tmp_path / "labels.parquet",
    )
    labels = pd.read_parquet(result.labels_path)
    assert labels["baseline_top1_correct"].eq(True).all()


def test_feature_audit_rejects_labels_embedded_in_feature_table() -> None:
    features = pd.DataFrame(
        {
            "sample_id": ["q", "q"],
            "candidate_id": ["a", "b"],
            "q_raw": [0.8, 0.2],
            "candidate_correct": [1, 0],
        }
    )
    labels = features[["sample_id", "candidate_id", "candidate_correct"]].copy()
    # Silently excluding this column from the model schema is insufficient:
    # the promised feature/label storage separation has already been violated.
    with pytest.raises(FeatureAuditError, match="label|forbidden"):
        audit_feature_table(features, labels)


def test_feature_audit_forbidden_scanner_rejects_correctness_alias() -> None:
    features = pd.DataFrame(
        {
            "sample_id": ["q", "q"],
            "candidate_id": ["a", "b"],
            "q_raw": [0.8, 0.2],
            "prediction_correctness": [1.0, 0.0],
        }
    )
    labels = pd.DataFrame(
        {
            "sample_id": ["q", "q"],
            "candidate_id": ["a", "b"],
            "candidate_correct": [1, 0],
        }
    )
    with pytest.raises(FeatureAuditError, match="forbidden"):
        audit_feature_table(features, labels)


def _modular_source() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_index in range(2):
        for rank in range(1, 7):
            rows.append(
                {
                    "sample_id": f"q{query_index}",
                    "scene_id": f"scene{query_index}",
                    "candidate_id": f"c{rank}",
                    "candidate_identity_sha256": f"{query_index + 1:x}" * 63
                    + f"{rank:x}",
                    "gqcnn_q_value": 0.8 if rank <= 2 else 1.0 / (rank + 1),
                    "gqcnn_rank": rank,
                    "center_u_px": 10.0 + rank,
                    "center_v_px": 20.0,
                    "center_depth_m": 0.7,
                    "angle_rad": 0.1,
                    "width_m": 0.04,
                    "width_px": 30.0,
                    "candidate_success": rank == 2,
                    "rectangle_iou": 0.8 if rank == 2 else 0.1,
                    "angle_difference_deg": 5.0 if rank == 2 else 60.0,
                    "target_probability_mean": 0.9 if rank == 2 else 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_modular_full_and_top5_preserve_exact_pool_and_separate_labels(
    tmp_path: Path,
) -> None:
    source = _modular_source()
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)
    for pool_type, expected in (
        ("full_post_filter", source),
        ("frozen_top5", source.loc[source["gqcnn_rank"] <= 5]),
    ):
        stem = "full" if pool_type == "full_post_filter" else "top5"
        result = build_modular_candidate_tables(
            source_path,
            tmp_path / f"{stem}-features.parquet",
            tmp_path / f"{stem}-labels.parquet",
            pool_type=pool_type,
        )
        features = pd.read_parquet(result.features_path)
        labels = pd.read_parquet(result.labels_path)
        expected_keys = set(
            expected[["sample_id", "candidate_id"]].itertuples(index=False, name=None)
        )
        assert (
            set(
                features[["sample_id", "candidate_id"]].itertuples(
                    index=False, name=None
                )
            )
            == expected_keys
        )
        assert (
            set(
                labels[["sample_id", "candidate_id"]].itertuples(index=False, name=None)
            )
            == expected_keys
        )
        assert not {
            "candidate_success",
            "candidate_correct",
            "rectangle_iou",
            "angle_difference_deg",
            "pool_has_positive",
            "baseline_top1_correct",
        } & set(features.columns)
        assert {
            "candidate_correct",
            "pool_has_positive",
            "baseline_top1_correct",
        } <= set(labels.columns)
        assert np.isfinite(features.select_dtypes(include=[np.number]).to_numpy()).all()
        audit_feature_table(features, labels)


def test_modular_query_universe_preserves_empty_queries(tmp_path: Path) -> None:
    source = _modular_source()
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)
    universe = pd.DataFrame(
        {
            "sample_id": ["q0", "q1", "empty"],
            "scene_id": ["scene0", "scene1", "scene-empty"],
        }
    )
    result = build_modular_candidate_tables(
        source_path,
        tmp_path / "features.parquet",
        tmp_path / "labels.parquet",
        query_universe=universe,
    )
    queries = pd.read_parquet(result.query_universe_path)
    empty = queries.set_index("sample_id").loc["empty"]
    assert result.query_count == 3
    assert result.empty_query_count == 1
    assert int(empty["candidate_count"]) == 0
    assert bool(empty["empty_query"]) is True


def test_modular_rich_test_features_join_only_by_exact_candidate_identity(
    tmp_path: Path,
) -> None:
    source = _modular_source()
    source_path = tmp_path / "canonical.parquet"
    source.to_parquet(source_path, index=False)
    rich = source.rename(
        columns={
            "gqcnn_q_value": "q_raw",
            "gqcnn_rank": "original_rank",
            "center_u_px": "x_px",
            "center_v_px": "y_px",
            "center_depth_m": "z_m",
        }
    ).drop(
        columns=[
            "candidate_success",
            "rectangle_iou",
            "angle_difference_deg",
        ]
    )
    rich["mask_support"] = np.linspace(0.1, 0.9, len(rich))
    rich_path = tmp_path / "rich.parquet"
    rich.sample(frac=1.0, random_state=3).to_parquet(rich_path, index=False)
    result = build_modular_candidate_tables(
        source_path,
        tmp_path / "features.parquet",
        tmp_path / "labels.parquet",
        inference_features=rich_path,
    )
    features = pd.read_parquet(result.features_path)
    labels = pd.read_parquet(result.labels_path)
    assert "mask_support" in features
    assert "candidate_correct" not in features
    assert result.inference_features_sha256 is not None
    assert len(features) == len(labels) == len(source)

    tampered = rich.copy()
    tampered.loc[0, "q_raw"] += 0.1
    tampered_path = tmp_path / "tampered.parquet"
    tampered.to_parquet(tampered_path, index=False)
    with pytest.raises(CandidateTableError, match="q_raw"):
        build_modular_candidate_tables(
            source_path,
            tmp_path / "bad-features.parquet",
            tmp_path / "bad-labels.parquet",
            inference_features=tampered_path,
        )


def test_modular_rich_test_features_accept_canonical_configured_width_alias(
    tmp_path: Path,
) -> None:
    source = _modular_source().rename(columns={"width_px": "configured_width_px"})
    source_path = tmp_path / "canonical.parquet"
    source.to_parquet(source_path, index=False)
    rich = source.rename(
        columns={
            "gqcnn_q_value": "q_raw",
            "gqcnn_rank": "original_rank",
            "center_u_px": "x_px",
            "center_v_px": "y_px",
            "center_depth_m": "z_m",
            "configured_width_px": "width_px",
        }
    ).drop(
        columns=[
            "candidate_success",
            "rectangle_iou",
            "angle_difference_deg",
            "x_px",
            "y_px",
            "z_m",
            "angle_rad",
        ]
    )
    rich_path = tmp_path / "rich.parquet"
    rich.to_parquet(rich_path, index=False)

    result = build_modular_candidate_tables(
        source_path,
        tmp_path / "features.parquet",
        tmp_path / "labels.parquet",
        inference_features=rich_path,
    )
    features = pd.read_parquet(result.features_path)
    assert features["width_px"].tolist() == rich["width_px"].tolist()
    assert features["x_px"].tolist() == source["center_u_px"].tolist()
    assert features["angle_rad"].tolist() == source["angle_rad"].tolist()

    rich.loc[0, "width_px"] += 1.0
    rich.to_parquet(rich_path, index=False)
    with pytest.raises(CandidateTableError, match="width_px"):
        build_modular_candidate_tables(
            source_path,
            tmp_path / "bad-features.parquet",
            tmp_path / "bad-labels.parquet",
            inference_features=rich_path,
        )


def test_modular_publisher_restores_geometry_by_exact_candidate_id(
    tmp_path: Path,
) -> None:
    features = pd.DataFrame(
        {
            "sample_id": ["q", "q"],
            "candidate_id": ["b", "a"],
            "scene_id": ["s", "s"],
            "q_raw": [0.2, 0.8],
            "original_gqcnn_rank": [2, 1],
            "candidate_positive": [0, 1],
            "candidate_gt_iou": [0.1, 0.5],
            "candidate_gt_angle_error_deg": [40.0, 10.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "sample_id": ["q", "q"],
            "candidate_id": ["a", "b"],
            "center_u_px": [10.0, 20.0],
            "center_v_px": [30.0, 40.0],
            "center_depth_m": [0.5, 0.6],
            "angle_rad": [0.1, 0.2],
            "width_m": [0.03, 0.04],
            "width_px": [30.0, 40.0],
        }
    )
    feature_path = tmp_path / "features.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    output_path = tmp_path / "published.parquet"
    features.to_parquet(feature_path, index=False)
    candidates.to_parquet(candidate_path, index=False)
    universes, count_evidence = _publisher_sources(feature_path, candidate_path)
    manifest = publish_modular_development(
        [feature_path],
        [candidate_path],
        output_path,
        query_universe_paths=universes,
        candidate_count_evidence_paths=count_evidence,
    )
    receipt_path = _verify_modular_development_source(
        tmp_path / "formal-run",
        output_path,
        query_universe_paths=universes,
        candidate_count_evidence_paths=count_evidence,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "VERIFIED_BEFORE_FORMAL_CONSUMPTION"
    assert receipt["manifest_payload_sha256"] == manifest["manifest_payload_sha256"]
    assert receipt["independent_reconstruction_exact"] is True
    published = pd.read_parquet(output_path)
    assert published["candidate_id"].tolist() == ["b", "a"]
    assert published["x_px"].tolist() == [20.0, 10.0]
    assert manifest["labels_or_scores_modified"] is False
    built = build_modular_candidate_tables(
        output_path, tmp_path / "model_features.parquet", tmp_path / "labels.parquet"
    )
    assert built.feature_rows == 2
    model_features = pd.read_parquet(built.features_path)
    assert not {
        "candidate_positive",
        "best_gt_id",
        "candidate_gt_iou",
        "candidate_gt_angle_error",
        "candidate_gt_angle_error_deg",
        "maximum_rectangle_iou_with_angle_gate",
        "legacy_geq_positive",
        "exact_iou_threshold_pair_count",
    } & set(model_features.columns)

    published.loc[0, "q_raw"] = 99.0
    published.to_parquet(output_path, index=False)
    with pytest.raises(StageError, match="failed independent verification"):
        _verify_modular_development_source(
            tmp_path / "tampered-run",
            output_path,
            query_universe_paths=universes,
            candidate_count_evidence_paths=count_evidence,
        )


def test_modular_publisher_rejects_candidate_key_mismatch(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    pd.DataFrame({"sample_id": ["q"], "candidate_id": ["a"]}).to_parquet(
        feature_path, index=False
    )
    pd.DataFrame(
        {
            "sample_id": ["q"],
            "candidate_id": ["b"],
            "center_u_px": [1.0],
            "center_v_px": [1.0],
            "center_depth_m": [1.0],
            "angle_rad": [0.0],
            "width_m": [0.1],
            "width_px": [1.0],
        }
    ).to_parquet(candidate_path, index=False)
    universes, count_evidence = _publisher_sources(feature_path, candidate_path)
    with pytest.raises(PublishError, match="key sets differ"):
        publish_modular_development(
            [feature_path],
            [candidate_path],
            tmp_path / "output.parquet",
            query_universe_paths=universes,
            candidate_count_evidence_paths=count_evidence,
        )


def test_modular_publisher_accepts_only_float32_round_trip_geometry(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "features.parquet"
    candidate_path = tmp_path / "candidates.parquet"
    output_path = tmp_path / "output.parquet"
    canonical_width = 0.05
    float32_width = float(np.float32(canonical_width))
    pd.DataFrame(
        {
            "sample_id": ["q"],
            "candidate_id": ["a"],
            "x_px": [1.0],
            "y_px": [2.0],
            "z_m": [0.5],
            "angle_rad": [0.1],
            "width_m": [float32_width],
            "width_px": [20.0],
        }
    ).to_parquet(feature_path, index=False)
    candidate = pd.DataFrame(
        {
            "sample_id": ["q"],
            "candidate_id": ["a"],
            "center_u_px": [1.0],
            "center_v_px": [2.0],
            "center_depth_m": [0.5],
            "angle_rad": [0.1],
            "width_m": [canonical_width],
            "width_px": [20.0],
        }
    )
    candidate.to_parquet(candidate_path, index=False)
    universes, count_evidence = _publisher_sources(feature_path, candidate_path)
    manifest = publish_modular_development(
        [feature_path],
        [candidate_path],
        output_path,
        query_universe_paths=universes,
        candidate_count_evidence_paths=count_evidence,
    )
    assert manifest["existing_geometry_equality"].startswith("exact")

    candidate.loc[0, "width_m"] = 0.0501
    candidate.to_parquet(candidate_path, index=False)
    universes, count_evidence = _publisher_sources(feature_path, candidate_path)
    with pytest.raises(PublishError, match="width_m"):
        publish_modular_development(
            [feature_path],
            [candidate_path],
            tmp_path / "bad.parquet",
            query_universe_paths=universes,
            candidate_count_evidence_paths=count_evidence,
        )


def test_output_path_safety_refuses_runs_root_and_paths_outside_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(StageError, match="output"):
        _safe_output_path(REPO_ROOT / "runs", resume=True)
    with pytest.raises(StageError, match="below"):
        _safe_output_path(tmp_path / "outside-runs", resume=False)


def test_resume_marker_requires_stage_identity_and_nonempty_validated_outputs(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("content", encoding="utf-8")
    _, marker = _stage_paths(tmp_path, "train")
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "stage": "different-stage",
                "status": "SUCCESS",
                "outputs": [str(artifact)],
            }
        ),
        encoding="utf-8",
    )
    assert _stage_completed(tmp_path, "train") is False

    marker.write_text(
        json.dumps({"stage": "train", "status": "SUCCESS", "outputs": []}),
        encoding="utf-8",
    )
    assert _stage_completed(tmp_path, "train") is False


def test_stage_receipt_marks_undeclared_immutable_output_mutation_stale(
    tmp_path: Path,
) -> None:
    _initialize_runner_config(tmp_path)
    artifact = tmp_path / "cumulative_registry.json"
    artifact.write_text('{"stage": "train"}\n', encoding="utf-8")
    _run_stage(
        tmp_path,
        "train",
        lambda: [artifact],
        resume=False,
        force=set(),
    )
    assert _stage_completed(tmp_path, "train") is True
    receipt = json.loads(
        (
            tmp_path / "logs" / "stages" / "train" / "outputs_at_completion.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["declared_outputs"][0]["path"] == str(artifact.resolve())

    artifact.write_text('{"stage": "validate"}\n', encoding="utf-8")
    assert _stage_completed(tmp_path, "train") is False


def test_cumulative_output_requires_exact_successor_receipt(
    tmp_path: Path,
) -> None:
    _initialize_runner_config(tmp_path)
    registry = tmp_path / "metrics" / "experiment_registry.json"
    registry.write_text('{"stage": "train"}\n', encoding="utf-8")
    _run_stage(
        tmp_path,
        "train",
        lambda: [registry],
        resume=False,
        force=set(),
    )
    registry.write_text('{"stage": "test-post-lock"}\n', encoding="utf-8")
    assert _stage_completed(tmp_path, "train") is False

    _run_stage(
        tmp_path,
        "test-post-lock",
        lambda: [registry],
        resume=False,
        force=set(),
    )
    assert _stage_completed(tmp_path, "test-post-lock") is True
    assert _stage_completed(tmp_path, "train") is True

    registry.write_text('{"stage": "tampered"}\n', encoding="utf-8")
    assert _stage_completed(tmp_path, "test-post-lock") is False
    assert _stage_completed(tmp_path, "train") is False


def test_stage_completed_rejects_forged_mutation_policy(
    tmp_path: Path,
) -> None:
    _initialize_runner_config(tmp_path)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("content\n", encoding="utf-8")
    _run_stage(
        tmp_path,
        "train",
        lambda: [artifact],
        resume=False,
        force=set(),
    )
    receipt_path = tmp_path / "logs/stages/train/outputs_at_completion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    descriptor = receipt["declared_outputs"][0]
    descriptor["mutation_policy"] = "superseded_by_stage"
    descriptor["superseded_by_stage"] = "test-post-lock"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    marker_path = tmp_path / "logs/stages/train/_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"] = [
        {
            "path": str(receipt_path.resolve()),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "size_bytes": receipt_path.stat().st_size,
        }
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert _stage_completed(tmp_path, "train") is False


def test_report_checksum_output_uses_immutable_stage_snapshot(
    tmp_path: Path,
) -> None:
    _initialize_runner_config(tmp_path)
    checksums = tmp_path / "checksums.sha256"
    checksums.write_text("a" * 64 + "  report.txt\n", encoding="utf-8")
    stage_bytes = checksums.read_bytes()
    _run_stage(
        tmp_path,
        "report",
        lambda: [checksums],
        resume=False,
        force=set(),
    )
    receipt = json.loads(
        (tmp_path / "logs/stages/report/outputs_at_completion.json").read_text(
            encoding="utf-8"
        )
    )
    descriptor = receipt["declared_outputs"][0]
    snapshot = Path(descriptor["snapshot_path"])
    assert descriptor["mutation_policy"] == "immutable_snapshot"
    assert snapshot.read_bytes() == stage_bytes
    assert descriptor["snapshot_sha256"] == hashlib.sha256(stage_bytes).hexdigest()

    checksums.write_text("b" * 64 + "  final.txt\n", encoding="utf-8")
    assert _stage_completed(tmp_path, "report") is True

    snapshot.write_bytes(b"tampered snapshot\n")
    assert _stage_completed(tmp_path, "report") is False


def test_resume_freezes_semantic_configuration_but_allows_execution_controls(
    tmp_path: Path,
) -> None:
    initial = _runner_args(stage="audit-only", resume=False)
    _initialize_run(tmp_path, initial)
    config = tmp_path / "configs" / "run_config.json"
    before = config.read_bytes()
    payload = json.loads(before)

    _initialize_run(
        tmp_path,
        _runner_args(
            stage="train",
            resume=True,
            force_rerun_specific_experiment=["some-experiment"],
        ),
    )
    assert config.read_bytes() == before
    assert payload["semantic_config_sha256"]
    assert "stage" not in payload["semantic_config"]
    assert "resume" not in payload["semantic_config"]


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("routes", ["crog"]),
        ("pools", ["top5"]),
        ("folds", 4),
        ("seeds", [42]),
        ("device", "mps"),
        ("neural_query_batch_size", 64),
        ("bootstrap_iterations", 999),
        ("no_hash_audit", True),
    ],
)
def test_resume_rejects_every_semantic_configuration_change(
    tmp_path: Path, field: str, changed: object
) -> None:
    _initialize_runner_config(tmp_path)
    with pytest.raises(StageError, match="semantic configuration mismatch"):
        _initialize_run(tmp_path, _runner_args(resume=True, **{field: changed}))


def _write_device_benchmark_evidence(root: Path) -> Path:
    path = root / "audit/neural_device_benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "recommended_device": "cpu",
                "benchmarks": [
                    {
                        "backend": backend,
                        "cpu_seconds": 1.0,
                        "mps_seconds": 2.0,
                    }
                    for backend in ("mlp", "deepsets", "gnn", "set_transformer")
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prelock_device_amendment_is_evidence_backed_and_auditable(
    tmp_path: Path,
) -> None:
    initial = _runner_args(device="mps")
    _initialize_run(tmp_path, initial)
    config_path = tmp_path / "configs/run_config.json"
    old = json.loads(config_path.read_text(encoding="utf-8"))
    _write_device_benchmark_evidence(tmp_path)

    _initialize_run(
        tmp_path,
        _runner_args(device="cpu", resume=True, amend_prelock_device=True),
    )

    amended = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(
        (tmp_path / "audit/prelock_device_amendment.json").read_text(encoding="utf-8")
    )
    archive = Path(audit["old_config_snapshot"])
    assert amended["device"] == "cpu"
    assert amended["semantic_config"]["device"] == "cpu"
    assert amended["semantic_config_sha256"] != old["semantic_config_sha256"]
    assert audit["status"] == "APPLIED_BEFORE_PRIMARY_LOCK"
    assert audit["old_device"] == "mps"
    assert audit["new_device"] == "cpu"
    assert json.loads(archive.read_text(encoding="utf-8")) == old

    # The amendment is one-time: ordinary same-device resume is now valid.
    _initialize_run(tmp_path, _runner_args(device="cpu", resume=True))


def test_prelock_device_amendment_requires_evidence_and_no_primary_lock(
    tmp_path: Path,
) -> None:
    _initialize_run(tmp_path, _runner_args(device="mps"))
    with pytest.raises(StageError, match="requires audit/neural_device_benchmark"):
        _initialize_run(
            tmp_path,
            _runner_args(device="cpu", resume=True, amend_prelock_device=True),
        )

    _write_device_benchmark_evidence(tmp_path)
    lock = tmp_path / "manifests/PRIMARY_METHOD_LOCK.json"
    lock.write_text("{}", encoding="utf-8")
    with pytest.raises(StageError, match="forbidden after primary locking"):
        _initialize_run(
            tmp_path,
            _runner_args(device="cpu", resume=True, amend_prelock_device=True),
        )


def test_prelock_batch_amendment_is_evidence_backed(tmp_path: Path) -> None:
    # Simulate the run schema immediately before query-batch identity was
    # introduced: every previously recorded semantic field remains frozen.
    _initialize_run(tmp_path, _runner_args())
    config_path = tmp_path / "configs/run_config.json"
    old = json.loads(config_path.read_text(encoding="utf-8"))
    old["semantic_config"].pop("neural_query_batch_size")
    old.pop("neural_query_batch_size")
    old["semantic_config_sha256"] = hashlib.sha256(
        json.dumps(
            old["semantic_config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    config_path.write_text(json.dumps(old), encoding="utf-8")
    evidence = tmp_path / "audit/neural_batch_benchmark.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "recommended_query_batch_size": 128,
                "test_labels_opened": False,
                "primary_lock_present_during_benchmark": False,
            }
        ),
        encoding="utf-8",
    )

    _initialize_run(
        tmp_path,
        _runner_args(resume=True, amend_prelock_training=True),
    )

    amended = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(
        (tmp_path / "audit/prelock_training_amendment.json").read_text(encoding="utf-8")
    )
    assert amended["semantic_config"]["neural_query_batch_size"] == 128
    assert audit["old_value"] is None
    assert audit["new_value"] == 128
    assert Path(audit["old_config_snapshot"]).is_file()


def test_stage_receipt_is_bound_to_frozen_configuration_identity(
    tmp_path: Path,
) -> None:
    _initialize_runner_config(tmp_path)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("content\n", encoding="utf-8")
    _run_stage(
        tmp_path,
        "train",
        lambda: [artifact],
        resume=False,
        force=set(),
    )
    marker = json.loads(
        (tmp_path / "logs/stages/train/_SUCCESS.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (tmp_path / "logs/stages/train/outputs_at_completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["semantic_config_sha256"] == receipt["semantic_config_sha256"]

    config = tmp_path / "configs/run_config.json"
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload["semantic_config"]["bootstrap_iterations"] = 1
    config_payload["semantic_config_sha256"] = hashlib.sha256(
        json.dumps(
            config_payload["semantic_config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    assert _stage_completed(tmp_path, "train") is False


def test_resume_rejects_unhashed_top_level_matrix_override(tmp_path: Path) -> None:
    _initialize_runner_config(tmp_path)
    config = tmp_path / "configs/run_config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["matrix_epochs"] = 1
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StageError, match="unfrozen semantic overrides"):
        _initialize_run(tmp_path, _runner_args(resume=True))


def test_run_success_refuses_forged_empty_stage_markers(tmp_path: Path) -> None:
    for stage in STAGES:
        _, marker = _stage_paths(tmp_path, stage)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"stage": stage, "status": "SUCCESS", "outputs": []}),
            encoding="utf-8",
        )
    args = SimpleNamespace(folds=5, seeds=[42, 123, 2026])
    with pytest.raises(StageError, match="success forbidden|incomplete"):
        _write_run_success(tmp_path, args)
    assert not (tmp_path / "_SUCCESS.json").exists()


def test_crog_baseline_evaluator_rejects_duplicate_label_candidate_ids(
    tmp_path: Path,
) -> None:
    candidate = _crog_candidate("c0", 0, 0.9)
    features_path = tmp_path / "features.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    _write_jsonl(features_path, [_crog_sample([candidate])])
    duplicate_labels = {
        "sample_id": "sample-1",
        "candidate_labels": [
            {
                "candidate_id": "c0",
                "candidate_checksum": candidate["candidate_checksum"],
                "candidate_correct": False,
            },
            {
                "candidate_id": "c0",
                "candidate_checksum": candidate["candidate_checksum"],
                "candidate_correct": True,
            },
        ],
    }
    _write_jsonl(labels_path, [duplicate_labels])
    with pytest.raises(StageError, match="duplicate"):
        _stream_crog_baseline(features_path, labels_path)


def test_modular_baseline_evaluator_rejects_conflicting_label_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reranking.run_experiment_matrix as runner

    candidates = _modular_source().iloc[:2].copy()
    candidates["joint_success"] = ~candidates["candidate_success"]
    evaluated_path = tmp_path / "evaluated.parquet"
    candidates.to_parquet(evaluated_path, index=False)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"sample_id": ["q0"], "scene_id": ["scene0"]}).to_csv(
        manifest_path, index=False
    )
    monkeypatch.setattr(runner, "MODULAR_EVALUATED", evaluated_path)
    monkeypatch.setattr(runner, "MODULAR_INPUT_MANIFEST", manifest_path)
    with pytest.raises(StageError, match="conflicting|ambiguous|label"):
        _modular_baseline()
