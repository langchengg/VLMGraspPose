from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from reranking.leakage_audit import (
    audit_development_test_identities,
    build_identity_rows,
)
from reranking.publish_modular_development import publish_modular_development
from reranking.completion_audit import (
    ABLATION_FILES,
    CUMULATIVE_STAGE_OUTPUT_POLICIES,
    FORMAL_STAGES,
    GALLERY_CATEGORIES,
    IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES,
    REPORT_FILES,
    STATISTICS_FILES,
    DatasetExpectation,
    _canonical_frame_sha256,
    _canonical_mapping_sha256,
    _reranking_source_tree_sha256,
    audit_background_activity,
    audit_run_completion,
)


METHODS = ("r0_q_baseline", "r4_mlp_residual_bce")
SEEDS = (42, 123, 2026)
FOLDS = tuple(range(5))
DATASETS = (
    DatasetExpectation("crog_frozen_top5", "crog", "frozen_top5"),
    DatasetExpectation("modular_frozen_top5", "modular", "frozen_top5"),
    DatasetExpectation("modular_full_post_filter", "modular", "full_post_filter"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(root: Path, name: str, payload: bytes = b"artifact\n") -> Path:
    path = root / "artifacts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_r10_evidence(
    root: Path, *, eligible: bool = True
) -> tuple[Path, Path, dict[str, object], list[dict[str, object]]]:
    method_id = "v3_fcer_native"
    run_root = root / "legacy" / "v3-frozen"
    run_root.mkdir(parents=True, exist_ok=True)
    checks = {
        name: {"passed": eligible, "evidence": [], "issues": []}
        for name in (
            "manifest",
            "complete_oof",
            "checkpoint",
            "candidate_hash",
            "evaluator_hash",
            "independent_recomputation",
        )
    }
    method = {
        "method_id": method_id,
        "run_id": "v3-frozen",
        "run_root": str(run_root),
        "eligible": eligible,
        "checks": checks,
        "gaps": [] if eligible else [{"code": "complete_oof_missing"}],
    }
    eligible_methods = (
        [
            {
                "method_id": method_id,
                "run_id": "v3-frozen",
                "run_root": str(run_root),
            }
        ]
        if eligible
        else []
    )
    discovery = _write_json(
        root / "audit" / "r10_existing_run_discovery.json",
        {
            "schema_version": 1,
            "kind": "existing_reranker_r10_discovery",
            "roots": [str(run_root)],
            "scan": {"documents": 1, "errors": []},
            "methods": [method],
            "eligible_methods": eligible_methods,
            "ineligible_methods": [] if eligible else [method_id],
        },
    )
    output = root / "metrics" / "r10_existing" / "crog_frozen_top5"
    output.mkdir(parents=True, exist_ok=True)
    report_discovery = _write_json(output / "discovery.json", {"source": str(discovery)})
    exclusions_payload = (
        []
        if eligible
        else [
            {
                "run_id": "v3-frozen",
                "run_root": str(run_root),
                "method_id": method_id,
                "stage": "eligibility",
                "eligibility_checks": checks,
                "gaps": [{"code": "complete_oof_missing"}],
            }
        ]
    )
    exclusions = _write_json(output / "exclusions.json", exclusions_payload)
    reference = output / "reference_predictions.jsonl"
    reference.write_text('{"query_id":"q","candidate_id":"candidate_0","score":1}\n', encoding="utf-8")
    comparisons: list[dict[str, object]] = []
    if eligible:
        method_root = output / "methods" / "v3_fcer_native"
        predictions = _artifact(root, "r10_predictions.jsonl")
        per_query = _artifact(root, "r10_per_query.jsonl")
        comparison_file = _artifact(root, "r10_comparison.json")
        provenance = _artifact(root, "r10_provenance.json")
        evaluator = _artifact(root, "r10_evaluator.py")
        comparisons = [
            {
                "method_id": method_id,
                "run_id": "v3-frozen",
                "run_root": str(run_root),
                "comparison": {
                    "reference_metrics": {"j_at_1": 0.8, "j_at_1_count": 8},
                    "challenger_metrics": {
                        "j_at_1": 0.9,
                        "j_at_1_count": 9,
                        "query_count": 10,
                    },
                    "switch_metrics": {"query_count": 10, "recovered": 1, "harmful": 0},
                },
                "comparison_evidence": {
                    "exact_candidate_join": True,
                    "independent_recomputed": True,
                    "candidate_pool_identity_sha256": "a" * 64,
                    "evaluator": _descriptor(evaluator),
                    "artifacts": {
                        "predictions": _descriptor(predictions),
                        "per_query": _descriptor(per_query),
                        "comparison": _descriptor(comparison_file),
                        "provenance": _descriptor(provenance),
                    },
                },
            }
        ]
        method_root.mkdir(parents=True, exist_ok=True)
    report = _write_json(
        output / "comparison_report.json",
        {
            "schema_version": 1,
            "kind": "crog_existing_r10_comparison",
            "dataset": "CROG",
            "scope": "test",
            "status": "complete" if eligible else "complete_no_eligible",
            "eligible_count": int(eligible),
            "comparison_count": int(eligible),
            "excluded_count": len(exclusions_payload),
            "eligible_methods": eligible_methods,
            "comparisons": comparisons,
            "exclusions": exclusions_payload,
            "candidate_pool_identity_sha256": "a" * 64,
            "artifacts": {
                "discovery": _descriptor(report_discovery),
                "exclusions": _descriptor(exclusions),
                "reference_predictions": _descriptor(reference),
            },
        },
    )
    summary: dict[str, object] = {
        "dataset": "crog_frozen_top5",
        "status": "complete" if eligible else "complete_no_eligible",
        "eligible_count": int(eligible),
        "comparison_count": int(eligible),
        "excluded_count": len(exclusions_payload),
        "report_path": str(report.resolve()),
        "report_sha256": _sha256(report),
        "primary_reselection_permitted": False,
        "trained_by_current_matrix": False,
    }
    rows = (
        [
            {
                "dataset": "crog_frozen_top5",
                "route": "crog",
                "rung": "R10",
                "method": "r10_v3_fcer_native_synthetic",
                "source_method": method_id,
                "source_run_id": "v3-frozen",
                "source_run_root": str(run_root),
                "gate": "FROZEN_EXISTING",
                "designation": "POST_LOCK_R10_EXISTING_COMPARISON",
                "eligible_for_primary_reselection": False,
                "trained_by_current_matrix": False,
                "r10_comparison_report": str(report.resolve()),
                "j_at_1": 0.9,
            }
        ]
        if eligible
        else []
    )
    return discovery, report, summary, rows


def _stage_output_descriptor(
    root: Path,
    stage: str,
    path: Path,
    *,
    captured_payload: bytes | None = None,
) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    snapshot_relative = IMMUTABLE_STAGE_OUTPUT_SNAPSHOT_POLICIES.get(
        (stage, relative)
    )
    policy = CUMULATIVE_STAGE_OUTPUT_POLICIES.get(relative)
    mutable = bool(
        policy is not None and stage in policy["mutable_receipt_stages"]
    )
    payload = path.read_bytes() if captured_payload is None else captured_payload
    descriptor: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256_at_stage_completion": hashlib.sha256(payload).hexdigest(),
        "size_bytes_at_stage_completion": len(payload),
        "mutation_policy": (
            "immutable_snapshot"
            if snapshot_relative is not None
            else "superseded_by_stage"
            if mutable
            else "immutable"
        ),
        "superseded_by_stage": (
            str(policy["successor_stage"]) if mutable and policy is not None else None
        ),
    }
    if snapshot_relative is not None:
        snapshot = root / snapshot_relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(payload)
        descriptor.update(
            {
                "snapshot_path": str(snapshot.resolve()),
                "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                "snapshot_size_bytes": len(payload),
            }
        )
    return descriptor


def _write_stage_receipt(
    root: Path,
    stage: str,
    config_identity: str,
    declared_outputs: list[dict[str, object]],
) -> None:
    receipt = _write_json(
        root / "logs" / "stages" / stage / "outputs_at_completion.json",
        {
            "schema_version": 1,
            "stage": stage,
            "status": "CAPTURED_AT_STAGE_COMPLETION",
            "semantic_config_sha256": config_identity,
            "declared_outputs": declared_outputs,
            "downstream_mutation_policy": (
                "immutable unless exact stage/path policy names a successor receipt"
            ),
        },
    )
    _write_json(
        root / "logs" / "stages" / stage / "_SUCCESS.json",
        {
            "stage": stage,
            "status": "SUCCESS",
            "semantic_config_sha256": config_identity,
            "outputs": [_descriptor(receipt)],
        },
    )


def _training_inventory(root: Path, dataset: str, method: str) -> dict[str, object]:
    rows = []
    for manifest_path in sorted((root / "manifests/experiments").glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "COMPLETE"
            or manifest.get("stage") != "train"
            or manifest.get("dataset") != dataset
            or manifest.get("method") != method
        ):
            continue
        bundle_path = Path(manifest["bundle_path"])
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_id": manifest["experiment_id"],
                "experiment_identity_sha256": manifest[
                    "experiment_identity_sha256"
                ],
                "seed": manifest["seed"],
                "fold": manifest["fold"],
                "manifest": _descriptor(manifest_path),
                "artifacts": [
                    _descriptor(Path(path)) for path in manifest["artifacts"]
                ],
                "bundle": _descriptor(bundle_path),
                "bundle_artifacts": {
                    "checkpoint": _descriptor(Path(bundle["checkpoint"])),
                    "score_calibrator": _descriptor(
                        Path(bundle["score_calibrator"])
                    ),
                    "prediction_calibrator": _descriptor(
                        Path(bundle["prediction_calibrator"])
                    ),
                },
                "preprocessor_storage": "embedded_in_bundle",
                "preprocessor_sha256": _canonical_mapping_sha256(
                    bundle["preprocessor"]
                ),
            }
        )
    coordinates = [
        {"seed": seed, "fold": fold} for seed in SEEDS for fold in FOLDS
    ]
    return {
        "coverage_kind": "complete_seed_fold_cartesian_product",
        "expected_coordinates": coordinates,
        "manifest_count": len(rows),
        "manifests": rows,
    }


def _held_out_inventory(root: Path, dataset: DatasetExpectation) -> dict[str, object]:
    features_path = root / f"features/candidates_{dataset.key}_test.parquet"
    query_path = root / f"data/queries_{dataset.key}_test.parquet"
    features = pd.read_parquet(features_path)
    universe = pd.read_parquet(query_path)
    pool_columns = [
        "sample_id",
        "candidate_id",
        "x_px",
        "y_px",
        "angle_rad",
        "width_px",
        "q_raw",
    ]
    query_columns = ["sample_id", "scene_id", "frame_id"]
    schema = {
        "columns": [
            {"name": str(column), "dtype": str(features[column].dtype)}
            for column in features.columns
        ]
    }
    return {
        "dataset": dataset.key,
        "route": dataset.route,
        "pool": dataset.pool,
        "features": _descriptor(features_path),
        "feature_schema": schema,
        "feature_schema_sha256": _canonical_mapping_sha256(schema),
        "candidate_count": len(features),
        "candidate_identity_columns": pool_columns,
        "candidate_pool_identity_sha256": _canonical_frame_sha256(
            features.sort_values(["sample_id", "candidate_id"]), pool_columns
        ),
        "query_universe": _descriptor(query_path),
        "query_universe_count": len(universe),
        "query_universe_identity_columns": query_columns,
        "query_universe_identity_sha256": _canonical_frame_sha256(
            universe.sort_values("sample_id"), query_columns
        ),
        "labels_opened_at_lock": False,
    }


def _experiment(
    root: Path,
    name: str,
    *,
    dataset: DatasetExpectation,
    method: str,
    stage: str,
    status: str = "COMPLETE",
    seed: int = -1,
    fold: int = -1,
    learned: bool = False,
    gate: str | None = None,
    designation: str | None = None,
    completed_at: str = "2026-08-03T10:01:00+00:00",
) -> Path:
    payload: dict[str, object] = {
        "experiment_id": name,
        "status": status,
        "stage": stage,
        "dataset": dataset.key,
        "route": dataset.route,
        "pool": dataset.pool,
        "method": method,
        "seed": seed,
        "fold": fold,
        "spec": {"key": method, "learned": learned},
        "completed_at": completed_at,
    }
    if status == "COMPLETE":
        output = _artifact(root, f"{name}.bin")
        artifacts = [output]
        if learned:
            checkpoint = _artifact(root, f"{name}.checkpoint")
            score_calibrator = _artifact(root, f"{name}.q-calibrator")
            prediction_calibrator = _artifact(root, f"{name}.prediction-calibrator")
            bundle = _write_json(
                root / "artifacts" / f"{name}.bundle.json",
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "score_calibrator": str(score_calibrator),
                    "score_calibrator_sha256": _sha256(score_calibrator),
                    "prediction_calibrator": str(prediction_calibrator),
                    "prediction_calibrator_sha256": _sha256(prediction_calibrator),
                    "preprocessor": {
                        "feature_columns": ["q_raw"],
                        "means": [0.5],
                        "scales": [0.2],
                    },
                },
            )
            artifacts.extend(
                [checkpoint, score_calibrator, prediction_calibrator, bundle]
            )
            payload["checkpoint_path"] = str(checkpoint)
            payload["bundle_path"] = str(bundle)
        payload["artifacts"] = [str(path) for path in artifacts]
        payload["artifact_sha256"] = {
            str(path): _sha256(path) for path in artifacts
        }
        payload["experiment_identity_sha256"] = hashlib.sha256(
            name.encode("utf-8")
        ).hexdigest()
    if gate is not None:
        payload["gate"] = gate
    if designation is not None:
        payload["prediction_designation"] = designation
    return _write_json(root / "manifests" / "experiments" / f"{name}.json", payload)


def _write_formal_run(root: Path, *, r10_eligible: bool = True) -> Path:
    root.mkdir()
    semantic_config = {
        "output": str(root.resolve()),
        "folds": 5,
        "seeds": list(SEEDS),
    }
    config_identity = _canonical_mapping_sha256(semantic_config)
    _write_json(
        root / "configs" / "run_config.json",
        {
            "schema_version": 2,
            "semantic_config": semantic_config,
            "semantic_config_sha256": config_identity,
        },
    )
    source = root.parent / "canonical-source.bin"
    source.write_bytes(b"immutable canonical source\n")
    source_sha = _sha256(source)
    (root / "manifests").mkdir()
    (root / "manifests" / "input_checksums.sha256").write_text(
        f"{source_sha}  {source}\n", encoding="utf-8"
    )
    for route in ("crog", "modular"):
        _write_json(
            root / "manifests" / f"canonical_{route}_run.json",
            {
                "route": route,
                "run_path": str(source),
                "candidate_pool": "frozen",
                "candidate_sha256": source_sha,
            },
        )

    _write_json(
        root / "audit" / "baseline_recomputation.json",
        {
            "computed_at": "2026-08-03T09:00:00+00:00",
            "crog": {"query_count": 10, "candidate_total": 50, "q_only_j_at_1": 0.2, "oracle": 0.6},
            "modular": {"query_count": 10, "candidate_total": 100, "q_only_j_at_1": 0.3, "oracle": 0.8},
        },
    )
    development_identity_parts = []
    test_identity_parts = []
    fit_partitions: dict[tuple[str, str, int], list[str]] = {}
    expected_fit_partitions: list[tuple[str, str, int]] = []
    fit_descriptors: list[dict[str, object]] = []
    for dataset in DATASETS:
        development_query = f"{dataset.key}-development-query"
        test_query = f"{dataset.key}-test-query"
        development_identity_parts.append(
            build_identity_rows(
                pd.DataFrame(
                    {
                        "query_id": [development_query],
                        "scene_id": [f"{dataset.key}-development-scene"],
                        "frame_id": [f"{dataset.key}-development-frame"],
                        "source_rgb_sha256": [
                            hashlib.sha256(
                                f"{dataset.key}-development-rgb".encode()
                            ).hexdigest()
                        ],
                        "source_depth_sha256": [
                            hashlib.sha256(
                                f"{dataset.key}-development-depth".encode()
                            ).hexdigest()
                        ],
                    }
                ),
                dataset=dataset.key,
                split="development",
            )
        )
        test_identity_parts.append(
            build_identity_rows(
                pd.DataFrame(
                    {
                        "query_id": [test_query],
                        "scene_id": [f"{dataset.key}-test-scene"],
                        "frame_id": [f"{dataset.key}-test-frame"],
                        "source_rgb_sha256": [
                            hashlib.sha256(
                                f"{dataset.key}-test-rgb".encode()
                            ).hexdigest()
                        ],
                        "source_depth_sha256": [
                            hashlib.sha256(
                                f"{dataset.key}-test-depth".encode()
                            ).hexdigest()
                        ],
                    }
                ),
                dataset=dataset.key,
                split="test",
            )
        )
        for fold in FOLDS:
            key = (dataset.key, "outer_train", fold)
            fit_partitions[key] = [development_query]
            expected_fit_partitions.append(key)
            fit_path = (
                root
                / "audit/leakage/development_fit_evidence"
                / dataset.key
                / f"outer_fold_{fold}.parquet"
            )
            fit_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "query_id": [development_query],
                    "candidate_id": ["candidate-0"],
                }
            ).to_parquet(fit_path, index=False)
            fit_descriptors.append(
                {
                    "dataset": dataset.key,
                    "outer_fold": fold,
                    "path": str(fit_path.resolve()),
                    "sha256": _sha256(fit_path),
                    "rows": 1,
                    "queries": 1,
                    "groups": 1,
                    "candidates": 1,
                }
            )
    leakage_result = audit_development_test_identities(
        pd.concat(development_identity_parts, ignore_index=True),
        pd.concat(test_identity_parts, ignore_index=True),
        fit_partitions=fit_partitions,
        expected_fit_partitions=expected_fit_partitions,
        require_fit_evidence=True,
    )
    leakage_result.write_bundle(root / "audit" / "leakage")
    (root / "audit" / "LEAKAGE_AUDIT.md").write_bytes(
        (root / "audit" / "leakage" / "LEAKAGE_AUDIT.md").read_bytes()
    )
    leakage_bundle_paths = [
        root / "audit" / "leakage" / name
        for name in (
            "identity_rows.jsonl",
            "identity_overlaps.jsonl",
            "fold_fit_identities.jsonl",
            "fold_fit_overlaps.jsonl",
            "fit_resolution_errors.jsonl",
            "summary.json",
            "LEAKAGE_AUDIT.md",
            "artifacts.sha256",
        )
    ] + [root / "audit" / "LEAKAGE_AUDIT.md"]
    leakage_record = {
        "status": "PASS",
        "audit_digest_sha256": leakage_result.summary["audit_digest_sha256"],
        "canonical_markdown_sha256": _sha256(
            root / "audit" / "LEAKAGE_AUDIT.md"
        ),
        "bundle_artifacts": [
            _descriptor(path) for path in leakage_bundle_paths
        ],
        "development_fit_evidence": fit_descriptors,
        "test_labels_opened": False,
    }

    upstream = root / "_fixture_upstream"
    upstream.mkdir()
    publication_features = upstream / "features.parquet"
    publication_candidates = upstream / "candidates.parquet"
    publication_output = upstream / "per_candidate.parquet"
    pd.DataFrame(
        {
            "sample_id": ["development-query"],
            "candidate_id": ["candidate-0"],
            "scene_id": ["development-scene"],
            "q_raw": [0.5],
        }
    ).to_parquet(publication_features, index=False)
    pd.DataFrame(
        {
            "sample_id": ["development-query"],
            "candidate_id": ["candidate-0"],
            "center_u_px": [10.0],
            "center_v_px": [20.0],
            "center_depth_m": [0.8],
            "angle_rad": [0.1],
            "width_m": [0.04],
            "width_px": [30.0],
        }
    ).to_parquet(publication_candidates, index=False)
    publication_per_sample = upstream / "per_sample.parquet"
    pd.DataFrame(
        {
            "sample_id": ["development-query", "development-zero-val"],
            "candidate_count": [1, 0],
        }
    ).to_parquet(publication_per_sample, index=False)
    _write_json(
        upstream / "dataset_manifest.json",
        {
            "status": "COMPLETED",
            "candidate_count": 1,
            "sample_count": 2,
            "per_candidate_sha256": _sha256(publication_features),
            "per_sample_sha256": _sha256(publication_per_sample),
        },
    )
    _write_json(
        upstream / "run_config.json",
        {
            "status": "COMPLETED",
            "candidate_stage_artifacts": {
                "nms": {
                    "path": str(publication_candidates.resolve()),
                    "rows": 1,
                    "sha256": _sha256(publication_candidates),
                }
            },
            "counts": {"samples": 2},
        },
    )
    train_universe = upstream / "train_universe.jsonl"
    train_universe.write_text(
        json.dumps(
            {
                "sample_id": "development-query",
                "split": "train",
                "scene_id": "development-scene",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    val_universe = upstream / "val_universe.jsonl"
    val_universe.write_text(
        json.dumps(
            {
                "sample_id": "development-zero-val",
                "split": "val",
                "scene_id": "development-zero-scene",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    publication = publish_modular_development(
        [publication_features],
        [publication_candidates],
        publication_output,
        query_universe_paths=(train_universe, val_universe),
        candidate_count_evidence_paths=(publication_per_sample,),
    )
    publication_manifest = publication_output.with_name(
        publication_output.name + ".manifest.json"
    )
    _write_json(
        root / "manifests" / "modular_development_publication_verification.json",
        {
            "schema_version": 1,
            "status": "VERIFIED_BEFORE_FORMAL_CONSUMPTION",
            "verified_at": "2026-08-03T09:10:00+00:00",
            "published_features_path": str(publication_output.resolve()),
            "published_features_sha256": _sha256(publication_output),
            "publication_manifest_path": str(publication_manifest.resolve()),
            "publication_manifest_sha256": _sha256(publication_manifest),
            "manifest_payload_sha256": publication[
                "manifest_payload_sha256"
            ],
            "rows": publication["rows"],
            "queries": publication["queries"],
            "independent_reconstruction_exact": True,
        },
    )
    audits = []
    pools = []
    for dataset in DATASETS:
        audits.append(
            {
                "route": dataset.route,
                "pool": dataset.pool,
                "candidate_count": 10,
                "query_count": 5,
                "forbidden_column_scanner_passed": True,
                "feature_schema": {"feature_schema_sha256": "a" * 64},
            }
        )
        pools.append({"route": dataset.route, "pool": dataset.pool, "status": "AVAILABLE"})
        for relative in (
            f"features/candidates_{dataset.key}_development.parquet",
            f"data/labels_{dataset.key}_development.parquet",
            f"data/labels_{dataset.key}_test.parquet",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic table\n")
        test_features = root / f"features/candidates_{dataset.key}_test.parquet"
        pd.DataFrame(
            {
                "sample_id": [f"{dataset.key}-q", f"{dataset.key}-q"],
                "candidate_id": ["c0", "c1"],
                "q_raw": [0.9, 0.4],
                "x_px": [10.0, 20.0],
                "y_px": [12.0, 24.0],
                "angle_rad": [0.0, 0.5],
                "width_px": [20.0, 30.0],
            }
        ).to_parquet(test_features, index=False)
        query_path = root / f"data/queries_{dataset.key}_test.parquet"
        pd.DataFrame(
            {
                "sample_id": [f"{dataset.key}-q", f"{dataset.key}-empty"],
                "scene_id": ["scene", "empty-scene"],
                "frame_id": ["frame", "empty-frame"],
            }
        ).to_parquet(query_path, index=False)
        _write_json(
            root / "features" / f"feature_schema_{dataset.key}.json",
            {"feature_schema_sha256": "a" * 64},
        )
    _write_json(root / "audit" / "feature_audit.json", {"audits": audits})
    _write_json(root / "data" / "candidate_pool_status.json", {"pools": pools})

    evaluator_log = root / "logs" / "evaluator-tests.log"
    evaluator_log.parent.mkdir(parents=True)
    evaluator_log.write_text("136 passed\n", encoding="utf-8")
    _write_json(
        root / "audit" / "evaluator_tests.json",
        {
            "status": "PASS",
            "exit_code": 0,
            "tests_failed": 0,
            "command": "python -m pytest reranking/tests",
            "completed_at": "2026-08-03T09:30:00+00:00",
            "log_path": str(evaluator_log),
        },
    )

    for dataset in DATASETS:
        _experiment(
            root,
            f"train__{dataset.key}__r0",
            dataset=dataset,
            method=METHODS[0],
            stage="train",
        )
        for seed in SEEDS:
            for fold in FOLDS:
                _experiment(
                    root,
                    f"train__{dataset.key}__r4__s{seed}__f{fold}",
                    dataset=dataset,
                    method=METHODS[1],
                    stage="train",
                    seed=seed,
                    fold=fold,
                    learned=True,
                )
        _experiment(
            root,
            f"r11__{dataset.key}",
            dataset=dataset,
            method="r11_vlm_candidate_judge",
            stage="train",
            status="NOT_RUN_CREDENTIAL_OR_BILLING_REQUIRED",
        )

    primaries = []
    for dataset in DATASETS:
        primaries.append(
            {
                "dataset": dataset.key,
                "route": dataset.route,
                "pool": dataset.pool,
                "method": METHODS[1],
                "gate": "G3",
                "feature_schema_sha256": "a" * 64,
                "model_class": "mlp",
                "exact_hyperparameters": {"hidden": [64, 32]},
                "checkpoint_selection_rule": "mean all grouped OOF checkpoints",
                "folds": 5,
                "seeds": list(SEEDS),
                "score_calibration": "train-fold-only",
                "gate_type": "G3",
                "gate_thresholds": {"gain": 0.8},
                "q_floor": 0.1,
                "candidate_manifest_sha256": "b" * 64,
                "selection_rationale": "grouped OOF winner",
                "training_artifact_lock": _training_inventory(
                    root, dataset.key, METHODS[1]
                ),
            }
        )
    lock = _write_json(
        root / "manifests" / "PRIMARY_METHOD_LOCK.json",
        {
            "status": "LOCKED_BEFORE_TEST_PRIMARY",
            "locked_at": "2026-08-03T10:00:00+00:00",
            "test_inputs_materialized_before_lock": True,
            "test_feature_identity_frozen_before_prediction": True,
            "test_labels_opened_at_lock": False,
            "test_inputs_used_for_primary_selection": False,
            "test_predictions_generated_before_lock": False,
            "test_prediction_artifacts_at_lock": [],
            "evaluator_sha256": "c" * 64,
            "git_commit": "d" * 40,
            "git_worktree_dirty": True,
            "reranking_source_tree_sha256": _reranking_source_tree_sha256(),
            "primaries": primaries,
            "held_out_test_inputs": [
                _held_out_inventory(root, dataset) for dataset in DATASETS
            ],
            "leakage_audit": leakage_record,
        },
    )
    for dataset in DATASETS:
        _experiment(
            root,
            f"test-primary__{dataset.key}__baseline",
            dataset=dataset,
            method=METHODS[0],
            stage="test-primary",
            gate="G0",
            designation="LOCKED_PRIMARY_TEST_BASELINE",
        )
        _experiment(
            root,
            f"test-primary__{dataset.key}__locked",
            dataset=dataset,
            method=METHODS[1],
            stage="test-primary",
            gate="G3",
            designation="LOCKED_PRIMARY_TEST",
        )
        _experiment(
            root,
            f"test-post-lock__{dataset.key}",
            dataset=dataset,
            method=METHODS[0],
            stage="test-post-lock",
            gate="G0",
            designation="POST_LOCK_COMPARATIVE_ONLY",
        )
    r10_discovery, r10_report, r10_summary, r10_rows = _write_r10_evidence(
        root, eligible=r10_eligible
    )
    if r10_eligible:
        _experiment(
            root,
            "test-post-lock__crog_frozen_top5__r10_existing",
            dataset=DATASETS[0],
            method="r10_v3_fcer_native_synthetic",
            stage="test-post-lock",
            gate="FROZEN_EXISTING",
            designation="POST_LOCK_R10_EXISTING_COMPARISON",
        )
    ordinary_post_rows = [{"dataset": item.key} for item in DATASETS]
    _write_json(
        root / "metrics" / "post_lock_test_results.json",
        {
            "designation": "POST_LOCK_COMPARATIVE_ONLY",
            "primary_reselection_permitted": False,
            "r10_existing_crog": [r10_summary],
            "results": [*ordinary_post_rows, *r10_rows],
        },
    )
    _write_json(
        root / "metrics" / "all_test_results.json",
        {
            "r10_existing_crog": [r10_summary],
            "results": [*ordinary_post_rows, *r10_rows],
        },
    )
    _write_json(
        root / "metrics" / "primary_summary.json",
        {"lock_path": str(lock), "lock_sha256": _sha256(lock), "results": [{"dataset": item.key} for item in DATASETS]},
    )

    _write_json(
        root / "audit" / "SANITY_AUDIT.json",
        {
            "stage": "post_test_complete",
            "all_mandatory_checks_executed": True,
            "datasets": [
                {
                    "dataset": dataset.key,
                    "independent_test_evaluator": {"status": "PASS", "exact_match": True},
                    "test_fit_scanner": {"passed": True},
                }
                for dataset in DATASETS
            ],
        },
    )
    (root / "audit" / "SANITY_AUDIT.md").write_text(
        "# Final sanity audit\n", encoding="utf-8"
    )

    for relative in (*STATISTICS_FILES, *ABLATION_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("name,value\nsynthetic,1\n", encoding="utf-8")
    for relative in REPORT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Machine-traceable report\n\nSynthetic value: 1\n", encoding="utf-8")

    gallery = root / "galleries"
    gallery.mkdir()
    gallery_rows = ["category,asset_path"]
    gallery_summary = {}
    for category in GALLERY_CATEGORIES:
        image = gallery / f"{category}.png"
        image.write_bytes(b"real image bytes\n")
        gallery_rows.append(f"{category},{image.name}")
        gallery_summary[category] = {"requested": 25, "eligible": 1, "selected": 1, "materialized": 1}
    gallery_summary["failure_stages"] = {
        f"F{index}": {
            "requested": 25,
            "eligible": 0,
            "selected": 0,
            "materialized": 0,
            "evidence_status": "UNAVAILABLE",
        }
        for index in range(11)
    }
    gallery_summary["failure_groups"] = {
        name: {"requested": 25, "eligible": 0, "selected": 0, "materialized": 0}
        for name in ("grounding", "candidate-generation", "ranking")
    }
    (gallery / "index.csv").write_text("\n".join(gallery_rows) + "\n", encoding="utf-8")
    (gallery / "index.html").write_text("<html>gallery</html>\n", encoding="utf-8")
    _write_json(gallery / "gallery_summary.json", gallery_summary)

    bundle_rows = []
    for dataset in DATASETS:
        bundle = root / "reporting_bundles" / dataset.key
        machine = bundle / "metrics.csv"
        machine.parent.mkdir(parents=True)
        machine.write_text("metric,value\nj_at_1,0.5\n", encoding="utf-8")
        manifest = _write_json(
            bundle / "reporting_manifest.json",
            {
                "status": "complete",
                "run_success_marker_written": False,
                "artifacts": {
                    "metrics.csv": {
                        "path": str(machine),
                        "sha256": _sha256(machine),
                        "size_bytes": machine.stat().st_size,
                    }
                },
            },
        )
        bundle_rows.append(
            {"dataset": dataset.key, "path": str(bundle), "manifest_sha256": _sha256(manifest)}
        )
    _write_json(root / "statistics" / "reporting_bundle_index.json", {"bundles": bundle_rows})

    registry_json = _write_json(
        root / "metrics" / "experiment_registry.json",
        {"finalized_by": "test-post-lock"},
    )
    registry_parquet = root / "metrics" / "experiment_registry.parquet"
    registry_parquet.write_bytes(b"finalized by test-post-lock\n")
    primary_summary = root / "metrics" / "primary_summary.json"
    sanity_json = root / "audit" / "SANITY_AUDIT.json"
    sanity_md = root / "audit" / "SANITY_AUDIT.md"
    for stage in FORMAL_STAGES:
        output = _artifact(root, f"stage__{stage}.txt", payload=f"{stage}\n".encode())
        declared = [_stage_output_descriptor(root, stage, output)]
        if stage == "train":
            declared.extend(
                (
                    _stage_output_descriptor(root, stage, r10_discovery),
                    _stage_output_descriptor(
                        root,
                        stage,
                        registry_json,
                        captured_payload=b'{"finalized_by":"train"}\n',
                    ),
                    _stage_output_descriptor(
                        root,
                        stage,
                        registry_parquet,
                        captured_payload=b"finalized by train\n",
                    ),
                )
            )
        elif stage == "validate":
            declared.extend(
                (
                    _stage_output_descriptor(
                        root,
                        stage,
                        sanity_json,
                        captured_payload=b'{"stage":"development"}\n',
                    ),
                    _stage_output_descriptor(
                        root,
                        stage,
                        sanity_md,
                        captured_payload=b"# Development sanity audit\n",
                    ),
                )
            )
        elif stage == "test-primary":
            declared.extend(
                (
                    _stage_output_descriptor(
                        root,
                        stage,
                        registry_json,
                        captured_payload=b'{"finalized_by":"test-primary"}\n',
                    ),
                    _stage_output_descriptor(
                        root,
                        stage,
                        registry_parquet,
                        captured_payload=b"finalized by test-primary\n",
                    ),
                    _stage_output_descriptor(
                        root,
                        stage,
                        primary_summary,
                        captured_payload=b'{"reporting_by_dataset":null}\n',
                    ),
                )
            )
        elif stage == "test-post-lock":
            declared.extend(
                (
                    _stage_output_descriptor(root, stage, registry_json),
                    _stage_output_descriptor(root, stage, registry_parquet),
                    _stage_output_descriptor(root, stage, r10_report),
                )
            )
        elif stage == "statistics":
            declared.append(
                _stage_output_descriptor(root, stage, primary_summary)
            )
        elif stage == "report":
            declared.extend(
                (
                    _stage_output_descriptor(root, stage, sanity_json),
                    _stage_output_descriptor(root, stage, sanity_md),
                    _stage_output_descriptor(
                        root,
                        stage,
                        root / "checksums.sha256",
                        captured_payload=b"report-stage checksum snapshot\n",
                    ),
                )
            )
        _write_stage_receipt(root, stage, config_identity, declared)

    files = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path != root / "checksums.sha256"
    )
    (root / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def formal_run(tmp_path: Path) -> Path:
    return _write_formal_run(tmp_path / "run")


def _background() -> dict[str, object]:
    return audit_background_activity(captured_at="2026-08-03T10:10:00+00:00")


def _audit(root: Path, *, background: dict[str, object] | None = None) -> dict[str, object]:
    return audit_run_completion(
        root,
        METHODS,
        DATASETS,
        SEEDS,
        FOLDS,
        background_evidence=_background() if background is None else background,
    )


def _codes(result: dict[str, object]) -> set[str]:
    return {str(row["code"]) for row in result["blockers"]}  # type: ignore[index]


def test_complete_machine_evidence_passes(formal_run: Path) -> None:
    result = _audit(formal_run)
    assert result["status"] == "PASS"
    assert result["passed"] is True
    assert result["blockers"] == []
    assert result["missing_paths"] == []
    assert all(check["passed"] for check in result["checks"].values())  # type: ignore[union-attr]


def test_modular_development_publication_tamper_blocks_completion(
    formal_run: Path,
) -> None:
    receipt = json.loads(
        (
            formal_run
            / "manifests/modular_development_publication_verification.json"
        ).read_text(encoding="utf-8")
    )
    publication = Path(str(receipt["published_features_path"]))
    publication.write_bytes(publication.read_bytes() + b"tampered")

    result = _audit(formal_run)

    assert result["status"] == "FAIL"
    assert "modular_development_publication_source_changed" in _codes(result)


def test_entity_leakage_bundle_tamper_blocks_completion(
    formal_run: Path,
) -> None:
    identities = formal_run / "audit/leakage/identity_rows.jsonl"
    identities.write_text(
        identities.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )

    result = _audit(formal_run)

    assert result["status"] == "FAIL"
    assert "entity_leakage_bundle_invalid" in _codes(result)


def test_r10_eligible_comparison_artifact_tamper_blocks_completion(
    formal_run: Path,
) -> None:
    report = json.loads(
        (
            formal_run
            / "metrics/r10_existing/crog_frozen_top5/comparison_report.json"
        ).read_text(encoding="utf-8")
    )
    prediction = Path(
        report["comparisons"][0]["comparison_evidence"]["artifacts"][
            "predictions"
        ]["path"]
    )
    prediction.write_bytes(b"tampered R10 predictions\n")

    result = _audit(formal_run)

    assert result["status"] == "FAIL"
    assert {
        "r10_comparison_report_invalid",
        "r10_artifact_integrity_mismatch",
    } <= _codes(result)


def test_r10_eligible_comparison_report_missing_blocks_completion(
    formal_run: Path,
) -> None:
    report = (
        formal_run
        / "metrics/r10_existing/crog_frozen_top5/comparison_report.json"
    )
    report.unlink()

    result = _audit(formal_run)

    assert result["status"] == "FAIL"
    assert "r10_comparison_report_missing" in _codes(result)


def test_r10_post_lock_summary_tamper_blocks_completion(formal_run: Path) -> None:
    post = formal_run / "metrics/post_lock_test_results.json"
    payload = json.loads(post.read_text(encoding="utf-8"))
    payload["r10_existing_crog"][0]["comparison_count"] = 99
    _write_json(post, payload)

    result = _audit(formal_run)

    assert result["status"] == "FAIL"
    assert "r10_summary_mismatch" in _codes(result)


def test_r10_zero_eligible_with_complete_exclusions_passes(tmp_path: Path) -> None:
    run = _write_formal_run(tmp_path / "zero-eligible", r10_eligible=False)

    result = _audit(run)

    assert result["status"] == "PASS"
    assert result["checks"]["r10_existing_crog"]["passed"] is True


def test_r10_zero_eligible_without_exclusion_evidence_blocks_completion(
    tmp_path: Path,
) -> None:
    run = _write_formal_run(tmp_path / "zero-eligible", r10_eligible=False)
    report_path = (
        run / "metrics/r10_existing/crog_frozen_top5/comparison_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["exclusions"] = []
    report["excluded_count"] = 0
    _write_json(report_path, report)

    result = _audit(run)

    assert result["status"] == "FAIL"
    assert {
        "r10_comparison_report_invalid",
        "r10_zero_eligible_evidence_invalid",
    } <= _codes(result)


def test_legacy_stage_marker_without_hash_size_fails_closed(formal_run: Path) -> None:
    marker = formal_run / "logs/stages/train/_SUCCESS.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["outputs"] = [payload["outputs"][0]["path"]]
    _write_json(marker, payload)
    result = _audit(formal_run)
    assert result["status"] == "FAIL"
    assert "stage_receipt_marker_invalid" in _codes(result)


def test_stage_output_tampering_is_detected(formal_run: Path) -> None:
    receipt = json.loads(
        (
            formal_run / "logs/stages/train/outputs_at_completion.json"
        ).read_text(encoding="utf-8")
    )
    immutable = next(
        row
        for row in receipt["declared_outputs"]
        if row["mutation_policy"] == "immutable"
    )
    Path(immutable["path"]).write_bytes(b"tampered\n")
    result = _audit(formal_run)
    assert "stage_declared_output_integrity_mismatch" in _codes(result)
    assert "checksum_mismatch" in _codes(result)


def test_stage_receipt_tampering_is_detected_by_marker(formal_run: Path) -> None:
    receipt = formal_run / "logs/stages/train/outputs_at_completion.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["captured_at"] = "tampered"
    _write_json(receipt, payload)
    result = _audit(formal_run)
    assert "stage_receipt_integrity_mismatch" in _codes(result)


def test_immutable_stage_snapshot_tampering_fails_closed(
    formal_run: Path,
) -> None:
    snapshot = (
        formal_run / "logs/stages/report/snapshots/checksums.sha256"
    )
    snapshot.write_bytes(b"tampered snapshot\n")
    result = _audit(formal_run)
    assert "stage_output_snapshot_integrity_mismatch" in _codes(result)


def test_cumulative_policy_has_exact_five_path_roles() -> None:
    assert CUMULATIVE_STAGE_OUTPUT_POLICIES == {
        "metrics/experiment_registry.json": {
            "mutable_receipt_stages": ("train", "test-primary"),
            "successor_stage": "test-post-lock",
        },
        "metrics/experiment_registry.parquet": {
            "mutable_receipt_stages": ("train", "test-primary"),
            "successor_stage": "test-post-lock",
        },
        "metrics/primary_summary.json": {
            "mutable_receipt_stages": ("test-primary",),
            "successor_stage": "statistics",
        },
        "audit/SANITY_AUDIT.json": {
            "mutable_receipt_stages": ("validate",),
            "successor_stage": "report",
        },
        "audit/SANITY_AUDIT.md": {
            "mutable_receipt_stages": ("validate",),
            "successor_stage": "report",
        },
    }


def test_cumulative_output_requires_declaring_successor_receipt(
    formal_run: Path,
) -> None:
    receipt_path = (
        formal_run / "logs/stages/test-post-lock/outputs_at_completion.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["declared_outputs"] = [
        row
        for row in receipt["declared_outputs"]
        if not str(row["path"]).endswith("experiment_registry.json")
    ]
    _write_json(receipt_path, receipt)
    marker_path = formal_run / "logs/stages/test-post-lock/_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"] = [_descriptor(receipt_path)]
    _write_json(marker_path, marker)
    result = _audit(formal_run)
    assert "stage_declared_output_integrity_mismatch" in _codes(result)


def test_forged_mutation_exception_outside_exact_allowlist_fails(
    formal_run: Path,
) -> None:
    receipt_path = formal_run / "logs/stages/train/outputs_at_completion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    immutable = next(
        row
        for row in receipt["declared_outputs"]
        if row["mutation_policy"] == "immutable"
    )
    immutable["mutation_policy"] = "superseded_by_stage"
    immutable["superseded_by_stage"] = "test-post-lock"
    _write_json(receipt_path, receipt)
    marker_path = formal_run / "logs/stages/train/_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"] = [_descriptor(receipt_path)]
    _write_json(marker_path, marker)
    result = _audit(formal_run)
    assert "stage_output_mutation_policy_invalid" in _codes(result)


def _non_primary_complete_manifest(root: Path) -> tuple[Path, dict[str, object]]:
    for path in sorted((root / "manifests/experiments").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("stage") == "train"
            and manifest.get("method") == "r0_q_baseline"
        ):
            return path, manifest
    raise AssertionError("synthetic formal run lacks the non-primary COMPLETE manifest")


def test_complete_manifest_artifact_tampering_fails_closed(
    formal_run: Path,
) -> None:
    _, manifest = _non_primary_complete_manifest(formal_run)
    Path(manifest["artifacts"][0]).write_bytes(b"tampered manifest artifact\n")
    result = _audit(formal_run)
    assert "manifest_artifact_integrity_mismatch" in _codes(result)


def test_complete_manifest_without_artifact_hash_mapping_fails_closed(
    formal_run: Path,
) -> None:
    path, manifest = _non_primary_complete_manifest(formal_run)
    manifest.pop("artifact_sha256")
    _write_json(path, manifest)
    result = _audit(formal_run)
    assert "complete_manifest_artifact_hashes_missing" in _codes(result)


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_complete_manifest_artifact_set_must_equal_hash_mapping(
    formal_run: Path, mutation: str
) -> None:
    path, manifest = _non_primary_complete_manifest(formal_run)
    hashes = manifest["artifact_sha256"]
    if mutation == "missing":
        hashes.pop(next(iter(hashes)))
    else:
        hashes[str(formal_run / "artifacts/unexpected.bin")] = "a" * 64
    _write_json(path, manifest)
    result = _audit(formal_run)
    assert "complete_manifest_artifact_hash_set_mismatch" in _codes(result)


def test_missing_seed_fold_coordinate_fails_coverage(formal_run: Path) -> None:
    path = (
        formal_run
        / "manifests/experiments"
        / "train__modular_full_post_filter__r4__s2026__f4.json"
    )
    path.unlink()
    result = _audit(formal_run)
    assert "method_coverage_incomplete" in _codes(result)
    assert str(path) in result["missing_paths"]


def test_lock_timestamp_must_precede_primary_test(formal_run: Path) -> None:
    lock = formal_run / "manifests/PRIMARY_METHOD_LOCK.json"
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["locked_at"] = "2026-08-03T11:00:00+00:00"
    _write_json(lock, payload)
    result = _audit(formal_run)
    assert "primary_lock_not_before_test" in _codes(result)


def test_completion_rechecks_locked_primary_training_artifacts(
    formal_run: Path,
) -> None:
    lock = json.loads(
        (formal_run / "manifests/PRIMARY_METHOD_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = Path(
        lock["primaries"][0]["training_artifact_lock"]["manifests"][0][
            "bundle_artifacts"
        ]["checkpoint"]["path"]
    )
    checkpoint.write_bytes(b"tampered checkpoint\n")
    result = _audit(formal_run)
    assert "primary_training_artifact_integrity" in _codes(result)
    assert "primary_training_bundle_artifact_integrity" in _codes(result)


def test_completion_rechecks_held_out_candidate_and_query_identity(
    formal_run: Path,
) -> None:
    lock = json.loads(
        (formal_run / "manifests/PRIMARY_METHOD_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    query_path = Path(lock["held_out_test_inputs"][0]["query_universe"]["path"])
    universe = pd.read_parquet(query_path)
    universe.loc[0, "scene_id"] = "changed-after-lock"
    universe.to_parquet(query_path, index=False)
    result = _audit(formal_run)
    assert "held_out_query_universe_integrity" in _codes(result)


def test_gallery_counts_cannot_claim_unmaterialized_cases(formal_run: Path) -> None:
    gallery = formal_run / "galleries/gallery_summary.json"
    payload = json.loads(gallery.read_text(encoding="utf-8"))
    payload["harmful"]["materialized"] = 0
    _write_json(gallery, payload)
    result = _audit(formal_run)
    assert "gallery_materialization_incomplete" in _codes(result)


def test_gallery_quota_cannot_be_reduced_below_protocol(formal_run: Path) -> None:
    gallery = formal_run / "galleries/gallery_summary.json"
    payload = json.loads(gallery.read_text(encoding="utf-8"))
    payload["recovered"]["requested"] = 24
    payload["recovered"]["selected"] = 1
    _write_json(gallery, payload)
    result = _audit(formal_run)
    assert "gallery_quota_too_small" in _codes(result)


def test_background_evidence_is_injectable_and_fail_closed(formal_run: Path) -> None:
    missing = audit_run_completion(
        formal_run,
        METHODS,
        DATASETS,
        SEEDS,
        FOLDS,
        background_evidence=None,
    )
    assert "background_activity_evidence_missing" in _codes(missing)

    active = audit_background_activity(
        [{"pid": 123, "command": "python train.py"}],
        captured_at="2026-08-03T10:10:00+00:00",
    )
    assert "background_activity_detected" in _codes(_audit(formal_run, background=active))


def test_failed_manifest_blocks_success_even_when_not_formal_method(formal_run: Path) -> None:
    _write_json(
        formal_run / "manifests/experiments/unrelated-failure.json",
        {"experiment_id": "unrelated", "status": "FAILED", "stage": "train"},
    )
    result = _audit(formal_run)
    assert "failed_experiment_manifest" in _codes(result)


def test_missing_evaluator_test_evidence_fails_closed(formal_run: Path) -> None:
    evidence = formal_run / "audit/evaluator_tests.json"
    evidence.unlink()
    result = _audit(formal_run)
    assert "missing_evidence" in _codes(result)
    assert str(evidence) in result["missing_paths"]


def test_every_critical_file_must_be_in_root_checksum_manifest(formal_run: Path) -> None:
    checksums = formal_run / "checksums.sha256"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    checksums.write_text(
        "\n".join(line for line in lines if "reports/FINAL_REPORT_EN.md" not in line) + "\n",
        encoding="utf-8",
    )
    result = _audit(formal_run)
    assert "critical_checksum_missing" in _codes(result)


def test_r11_requires_complete_or_exact_credential_status(formal_run: Path) -> None:
    manifest = formal_run / "manifests/experiments/r11__crog_frozen_top5.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "SKIPPED"
    _write_json(manifest, payload)
    result = _audit(formal_run)
    assert "r11_status_invalid" in _codes(result)


def test_formal_expectations_cannot_omit_required_route_pool() -> None:
    result = audit_run_completion(
        "/does/not/matter",
        METHODS,
        DATASETS[:1],
        SEEDS,
        FOLDS,
        background_evidence=_background(),
    )
    assert result["status"] == "FAIL"
    assert "audit_expectations_invalid" in _codes(result)
