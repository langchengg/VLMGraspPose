from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.unified_reranking.create_formal_test_lock import (
    REQUIRED_LOCK_FILES,
    create_unified_formal_lock,
    execute_create_unified_formal_lock,
)
from tools.unified_reranking.run_formal_test_once import (
    _canonical_selection_keys,
    _validate_labels,
    execute_formal_test_once,
    run_formal_test_once,
)
from unified_reranking.hashing import canonical_sha256, sha256_file
from unified_reranking.lock import verify_formal_test_lock
from unified_reranking.prelock import assemble_prelock_bundle
from unified_reranking.test_bridge import build_label_free_test_bridge


ROUTES = ("crog", "g1", "c1")

PRELOCK_TEST_PATH = Path(__file__).with_name("test_prelock_assembly.py")
PRELOCK_SPEC = importlib.util.spec_from_file_location(
    "synthetic_prelock_fixture", PRELOCK_TEST_PATH
)
assert PRELOCK_SPEC is not None and PRELOCK_SPEC.loader is not None
PRELOCK_FIXTURE = importlib.util.module_from_spec(PRELOCK_SPEC)
PRELOCK_SPEC.loader.exec_module(PRELOCK_FIXTURE)


def test_union_and_native_same_g1_candidate_have_one_canonical_selection_key() -> None:
    native = pd.DataFrame(
        {"selected_route": ["g1"], "selected_candidate_id": ["same"]}
    )
    union = pd.DataFrame(
        {
            "selected_route": ["g1"],
            "selected_candidate_id": ["G1:same"],
            "source_candidate_id": ["same"],
        }
    )
    assert _canonical_selection_keys(native).tolist() == ["g1::same"]
    assert _canonical_selection_keys(union).tolist() == ["g1::same"]


def test_label_normalization_filters_diagnostic_variants_without_key_drift() -> None:
    rows = []
    pools = {}
    for route in ROUTES:
        pools[route] = pd.DataFrame(
            {"sample_id": ["s"], "candidate_id": [f"{route}-native"]}
        )
        rows.append(
            {
                "method": route.upper(),
                "variant": "crog_native" if route == "crog" else route,
                "sample_id": "s",
                "candidate_id": f"{route}-native",
                "candidate_success": True,
            }
        )
        rows.append(
            {
                "method": route.upper(),
                "variant": f"{route}_gtmask_oracle",
                "sample_id": "s",
                "candidate_id": f"{route}-diagnostic",
                "candidate_success": True,
            }
        )
    result = _validate_labels(
        pd.DataFrame(rows),
        {"s"},
        pools,
        {
            "normalization": {
                "route_column": "method",
                "variant_column": "variant",
                "include_variants": ["crog_native", "g1", "c1"],
            }
        },
    )
    assert len(result) == 3
    assert set(result["route"]) == set(ROUTES)


def _atomic_json_for_test(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _build_semantic_p11_evidence(
    root: Path,
    formal_run_dir: Path,
    *,
    positive_union: bool = False,
) -> tuple[Path, dict[str, object]]:
    """Build real synthetic producer evidence without opening Test label rows."""

    evidence_root = root / (
        "positive_p11_evidence" if positive_union else "base_p11_evidence"
    )
    evidence_run = evidence_root / "run"
    evidence_plan_path = evidence_run / "08_lock/formal_evaluation_plan.json"
    if not evidence_plan_path.is_file():
        evidence_run, opaque_labels, evaluator, code = PRELOCK_FIXTURE._build_run(
            evidence_root,
            positive_union=positive_union,
        )
        if positive_union:
            PRELOCK_FIXTURE._enable_positive_union(evidence_run)
        assemble_prelock_bundle(
            evidence_run,
            candidate_test_labels_path=opaque_labels,
            evaluator_path=evaluator,
            code_roots=[code],
        )
    # Union semantic recomputation is intentionally rooted at the run being
    # locked. Copy only development candidates/labels and denominators; none of
    # these paths contains Test labels.
    for relative in (
        "01_manifests/paired_train.parquet",
        "01_manifests/paired_validation.parquet",
        "02_candidates/crog_train_top5.parquet",
        "02_candidates/g1_train_top5.parquet",
        "02_candidates/c1_train_top5.parquet",
        "02_candidates/crog_validation_top5.parquet",
        "02_candidates/g1_validation_top5.parquet",
        "02_candidates/c1_validation_top5.parquet",
        "03_features/candidate_labels_crog_train_top5.parquet",
        "03_features/candidate_labels_g1_train_top5.parquet",
        "03_features/candidate_labels_c1_train_top5.parquet",
        "03_features/candidate_labels_crog_validation_top5.parquet",
        "03_features/candidate_labels_g1_validation_top5.parquet",
        "03_features/candidate_labels_c1_validation_top5.parquet",
    ):
        source = evidence_run / relative
        destination = formal_run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    plan = json.loads(
        evidence_plan_path.read_text(encoding="utf-8")
    )
    return evidence_run, plan


def _build_synthetic_run(
    root: Path, *, positive_union: bool = False
) -> tuple[Path, Path, Path]:
    staging_run = root / "synthetic_run"
    (staging_run / "08_lock").mkdir(parents=True)
    evidence_run, semantic_plan = _build_semantic_p11_evidence(
        root, staging_run, positive_union=positive_union
    )
    # The formal lock semantically replays the matrix phases relative to the
    # run being locked.  Build the synthetic formal bundle in that same P11
    # evidence run so command run-dir bindings and cell source paths remain
    # exact instead of copying/re-authoring lineage artifacts.
    run_dir = evidence_run
    lock_dir = run_dir / "08_lock"
    semantic_methods = json.loads(
        (evidence_run / "08_lock/selected_methods.json").read_text(encoding="utf-8")
    )
    semantic_features = json.loads(
        (evidence_run / "08_lock/selected_features.json").read_text(
            encoding="utf-8"
        )
    )
    semantic_hyperparameters = json.loads(
        (evidence_run / "08_lock/selected_hyperparameters.json").read_text(
            encoding="utf-8"
        )
    )
    semantic_gates = json.loads(
        (evidence_run / "08_lock/gate_thresholds.json").read_text(encoding="utf-8")
    )
    semantic_router = json.loads(
        (evidence_run / "08_lock/router_thresholds.json").read_text(encoding="utf-8")
    )
    _atomic_json_for_test(
        run_dir / "manifest.json",
        {
            "formal_test_execution_count": 0,
            "test_label_state": "VALIDATION_SELECTION_COMPLETE",
        },
    )
    for name, filename in REQUIRED_LOCK_FILES.items():
        path = lock_dir / filename
        if filename.endswith("_sha256.txt"):
            path.write_text("a" * 64 + "\n", encoding="utf-8")
        elif filename.endswith(".md"):
            path.write_text("# Synthetic primary declaration\n\nAll policies frozen.\n", encoding="utf-8")
        else:
            route_records = {route: {} for route in ROUTES}
            if name == "selected_methods":
                route_records = {
                    route: {
                        **semantic_methods["routes"][route],
                        "native_system": f"{route}_native",
                        "ungated_system": f"{route}_ungated",
                        "gated_system": f"{route}_gated",
                    }
                    for route in ROUTES
                }
            elif name == "selected_features":
                route_records = {
                    route: {"feature_columns": ["native_score", "reliability"]}
                    for route in ROUTES
                }
            elif name == "selected_hyperparameters":
                route_records = {route: {"alpha": 0.5} for route in ROUTES}
            elif name == "calibration_manifest":
                route_records = {
                    route: {"manifest_sha256": "b" * 64} for route in ROUTES
                }
            elif name == "gate_thresholds":
                route_records = semantic_gates["routes"]
            payload: dict[str, object] = {
                "schema_version": 1,
                "status": "LOCKED",
                "routes": route_records,
            }
            semantic_source = {
                "selected_methods": semantic_methods,
                "selected_features": semantic_features,
                "selected_hyperparameters": semantic_hyperparameters,
            }.get(name)
            if isinstance(semantic_source, dict) and "union" in semantic_source:
                payload["union"] = semantic_source["union"]
            if name == "router_thresholds":
                payload = semantic_router
            _atomic_json_for_test(path, payload)

    if positive_union:
        samples = pd.read_parquet(
            evidence_run / "01_manifests/paired_test.parquet"
        )[["sample_id", "scene_id", "frame_id"]].copy()
    else:
        count = 8
        samples = pd.DataFrame(
            {
                "sample_id": [f"sample-{index}" for index in range(count)],
                "scene_id": [f"scene-{index}" for index in range(count)],
                "frame_id": [f"frame-{index // 2}" for index in range(count)],
            }
        )
    count = len(samples)
    sample_path = lock_dir / "synthetic_paired_test.parquet"
    samples.to_parquet(sample_path, index=False)
    fold_path = lock_dir / "synthetic_fold_assignments.parquet"
    pd.DataFrame(
        {"sample_id": samples["sample_id"], "fold": np.arange(count) % 2}
    ).to_parquet(fold_path, index=False)
    evaluator_path = lock_dir / "synthetic_evaluator.py"
    evaluator_path.write_text(
        """import numpy as np
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0
class CanonicalGrasp:
    def __init__(self, **values):
        self.__dict__.update(values)
def gt_from_corners(value):
    return np.asarray(value, dtype=float)
def evaluate_candidate(candidate, ground_truth):
    pairwise = [
        {"gt_index": index, "iou": 1.0, "angle_error_deg": 0.0}
        for index, _ in enumerate(ground_truth)
    ]
    return {"success": bool(pairwise), "pairwise": pairwise}
""",
        encoding="utf-8",
    )
    code_manifest_path = lock_dir / "code_manifest.json"
    code_bundle_sha256 = "c" * 64
    _atomic_json_for_test(
        code_manifest_path,
        {"schema_version": 1, "bundle_sha256": code_bundle_sha256, "files": []},
    )
    (lock_dir / "fold_assignments_sha256.txt").write_text(
        sha256_file(fold_path) + "\n", encoding="utf-8"
    )
    (lock_dir / "evaluator_sha256.txt").write_text(
        sha256_file(evaluator_path) + "\n", encoding="utf-8"
    )
    (lock_dir / "code_sha256.txt").write_text(
        code_bundle_sha256 + "\n", encoding="utf-8"
    )
    bridge_records: dict[str, dict[str, str]] = {}
    for route in ("g1", "c1"):
        for split in ("train", "validation"):
            bridge_path = lock_dir / f"bridge_{route}_{split}.json"
            _atomic_json_for_test(
                bridge_path,
                {"status": "COMPLETE", "route": route, "split": split},
            )
            bridge_records[f"{route}_{split}"] = {
                "path": str(bridge_path),
                "sha256": sha256_file(bridge_path),
            }
    label_rows: list[dict[str, object]] = []
    systems: list[dict[str, object]] = []
    decisions_dir = lock_dir / "formal_label_free_decisions"
    decisions_dir.mkdir()
    route_success = (
        {
            "crog": np.asarray([0, 1, 0], dtype=bool),
            "g1": np.asarray([1, 0, 1], dtype=bool),
            "c1": np.asarray([0, 1, 1], dtype=bool),
        }
        if positive_union
        else {
            "crog": np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool),
            "g1": np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool),
            "c1": np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool),
        }
    )
    pool_records: dict[str, dict[str, str]] = {}
    for route in ROUTES:
        for sample_index, sample_id in enumerate(samples["sample_id"]):
            a_success = bool(route_success[route][sample_index])
            label_rows.extend(
                [
                    {
                        "route": route,
                        "sample_id": sample_id,
                        "candidate_id": "a",
                        "candidate_success": a_success,
                    },
                    {
                        "route": route,
                        "sample_id": sample_id,
                        "candidate_id": "b",
                        "candidate_success": not a_success,
                    },
                ]
            )
        if positive_union:
            candidate_pool = pd.read_parquet(
                evidence_run / f"02_candidates/{route}_test_top5.parquet"
            )[
                [
                    "sample_id",
                    "candidate_id",
                    "native_rank",
                    "candidate_geometry_sha256",
                ]
            ].copy()
        else:
            candidate_pool = pd.DataFrame(
                [
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "native_rank": rank,
                        "candidate_geometry_sha256": f"geometry-{route}-{candidate}",
                    }
                    for sample_id in samples["sample_id"]
                    for candidate, rank in (("a", 1), ("b", 2))
                ]
            )
        native_ranking = candidate_pool.rename(
            columns={"native_rank": "rank"}
        ).copy()
        native_ranking["frozen_native_rank"] = native_ranking["rank"]
        ungated_ranking = candidate_pool.copy()
        ungated_ranking["frozen_native_rank"] = ungated_ranking["native_rank"]
        ungated_ranking["rank"] = 3 - ungated_ranking["native_rank"]
        ungated_ranking = ungated_ranking.drop(columns=["native_rank"])
        pool_path = lock_dir / f"{route}_all_candidates.parquet"
        candidate_pool.to_parquet(pool_path, index=False)
        pool_records[route] = {"path": str(pool_path), "sha256": sha256_file(pool_path)}
        native_decisions = pd.DataFrame(
            {"sample_id": samples["sample_id"], "selected_candidate_id": "a"}
        )
        ungated_decisions = pd.DataFrame(
            {"sample_id": samples["sample_id"], "selected_candidate_id": "b"}
        )
        gated_decisions = ungated_decisions.copy()
        paths = {
            "native_ranking": decisions_dir / f"{route}_native_ranking.parquet",
            "ungated_ranking": decisions_dir / f"{route}_ungated_ranking.parquet",
            "native_decisions": decisions_dir / f"{route}_native_decisions.parquet",
            "ungated_decisions": decisions_dir / f"{route}_ungated_decisions.parquet",
            "gated_decisions": decisions_dir / f"{route}_gated_decisions.parquet",
        }
        native_ranking.to_parquet(paths["native_ranking"], index=False)
        ungated_ranking.to_parquet(paths["ungated_ranking"], index=False)
        native_decisions.to_parquet(paths["native_decisions"], index=False)
        ungated_decisions.to_parquet(paths["ungated_decisions"], index=False)
        gated_decisions.to_parquet(paths["gated_decisions"], index=False)
        native_name = f"{route}_native"
        ungated_name = f"{route}_ungated"
        gated_name = f"{route}_gated"
        systems.extend(
            [
                {
                    "name": native_name,
                    "kind": "native",
                    "route": route,
                    "decisions_path": str(paths["native_decisions"]),
                    "ranking_path": str(paths["native_ranking"]),
                    "rank_column": "rank",
                },
                {
                    "name": ungated_name,
                    "kind": "ungated",
                    "route": route,
                    "native_reference": native_name,
                    "hypothesis_family": "ungated_secondary",
                    "decisions_path": str(paths["ungated_decisions"]),
                    "ranking_path": str(paths["ungated_ranking"]),
                    "rank_column": "rank",
                },
                {
                    "name": gated_name,
                    "kind": "gated",
                    "route": route,
                    "native_reference": native_name,
                    "hypothesis_family": "primary_confirmatory",
                    "decisions_path": str(paths["gated_decisions"]),
                    "base_ranking_system": ungated_name,
                },
            ]
        )
    router = pd.DataFrame(
        {
            "sample_id": samples["sample_id"],
            "selected_route": [
                ("g1", "c1", "crog")[index % 3] for index in range(count)
            ],
            "selected_candidate_id": "b",
        }
    )
    router_path = decisions_dir / "router_decisions.parquet"
    router.to_parquet(router_path, index=False)
    systems.append(
        {
            "name": "crog_default_router",
            "kind": "router",
            "route": "cross_route",
            "native_reference": "crog_gated",
            "hypothesis_family": "primary_confirmatory",
            "decisions_path": str(router_path),
        }
    )
    labels_path = root / "synthetic_candidate_test_labels.parquet"
    pd.DataFrame(label_rows).to_parquet(labels_path, index=False)
    label_manifest_path = lock_dir / "candidate_test_label_manifest.json"
    _atomic_json_for_test(
        label_manifest_path,
        {
            "schema_version": 1,
            "split": "test",
            "provenance": "synthetic_frozen_evaluator_labels",
            "candidate_labels_path": str(labels_path),
            "candidate_labels_sha256": sha256_file(labels_path),
            "evaluator_sha256": sha256_file(evaluator_path),
            "candidate_pools": pool_records,
        },
    )
    _atomic_json_for_test(
        lock_dir / "candidate_manifests.json",
        {
            "schema_version": 1,
            "status": "LOCKED",
            "candidate_pools": pool_records,
        },
    )
    bridge_sources = root / "synthetic_bridge_sources"
    bridge_sources.mkdir()
    historical_root = bridge_sources / "historical"
    modular_root = bridge_sources / "modular"
    (historical_root / "data").mkdir(parents=True)
    (historical_root / "audit").mkdir(parents=True)
    (modular_root / "manifests").mkdir(parents=True)
    historical_paths: dict[str, Path] = {}
    fair_paths: dict[str, Path] = {}
    feature_paths: dict[str, Path] = {}
    for route in ("g1", "c1"):
        fair_rows = []
        historical_rows = []
        for sample_id in samples["sample_id"]:
            for rank, candidate in ((1, "a"), (2, "b")):
                geometry = [float(rank), 2.0, -10.0 + rank, 4.0, 2.0]
                fair_rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_id": candidate,
                        "native_rank": rank,
                        "native_score": 1.0 / rank,
                        "cx_px": geometry[0],
                        "cy_px": geometry[1],
                        "theta_deg": geometry[2],
                        "width_px": geometry[3],
                        "height_px": geometry[4],
                        "candidate_geometry_sha256": canonical_sha256(
                            [
                                route.upper(),
                                sample_id,
                                candidate,
                                rank,
                                *geometry,
                            ]
                        ),
                    }
                )
                quality = 0.9 / rank
                support = 0.8
                historical_rows.append(
                    {
                        "sample_id": sample_id,
                        "stable_candidate_id": candidate,
                        "original_rank": rank,
                        "original_score": quality * support,
                        "raw_network_quality": quality,
                        "stored_center_mask_support": support,
                        "center_x": geometry[0] + 0.25,
                        "center_y": geometry[1],
                        "angle_deg": geometry[2],
                        "width_px": geometry[3],
                        "height_px": geometry[4],
                        "candidate_identity_sha256": canonical_sha256(
                            [route, sample_id, candidate, rank, "historical"]
                        ),
                        "split": "test",
                        "backend": route.upper(),
                    }
                )
        fair = pd.DataFrame(fair_rows)
        fair_path = bridge_sources / f"{route}_fair.parquet"
        fair.to_parquet(fair_path, index=False)
        features = fair[["sample_id", "candidate_id", "native_rank"]].copy()
        features["native_score_raw"] = fair["native_score"]
        features["p_center"] = 0.8
        feature_path = bridge_sources / f"{route}_features.parquet"
        features.to_parquet(feature_path, index=False)
        historical_frame = pd.DataFrame(historical_rows)
        historical_path = (
            historical_root / f"data/frozen_{route}_test_top5_candidates.parquet"
        )
        historical_frame.to_parquet(historical_path, index=False)
        historical_frame.to_parquet(
            historical_root / f"data/frozen_{route}_test_allnms_candidates.parquet",
            index=False,
        )
        fair_paths[route] = fair_path
        feature_paths[route] = feature_path
        historical_paths[route] = historical_path
    bridge_ground_truth = modular_root / "manifests/test_labels.parquet"
    pd.DataFrame(
        {
            "sample_id": samples["sample_id"],
            "gt_grasp_rectangles": [
                [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]]
                for _ in range(count)
            ],
        }
    ).to_parquet(bridge_ground_truth, index=False)
    bridge_source_manifest = modular_root / "manifests/experiment_lock.json"
    experiment_lock = {
        "schema_version": 1,
        "lock_status": "LOCKED",
        "effective": True,
        "run_id": "synthetic-modular",
        "run_dir": str(modular_root.resolve()),
        "lock_relative_path": "manifests/experiment_lock.json",
        "marker_relative_path": ".EXPERIMENT_LOCKED",
        "artifacts": {
            "test_labels": {
                "path": "manifests/test_labels.parquet",
                "sha256": sha256_file(bridge_ground_truth),
            }
        },
    }
    experiment_lock["manifest_content_sha256"] = canonical_sha256(experiment_lock)
    _atomic_json_for_test(bridge_source_manifest, experiment_lock)
    _atomic_json_for_test(
        modular_root / ".EXPERIMENT_LOCKED",
        {
            "schema_version": 1,
            "lock_status": "LOCKED",
            "run_id": experiment_lock["run_id"],
            "lock_relative_path": "manifests/experiment_lock.json",
            "manifest_content_sha256": experiment_lock["manifest_content_sha256"],
        },
    )
    _atomic_json_for_test(
        modular_root / "FINALIZATION_COMPLETE.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "experiment_lock_sha256": experiment_lock["manifest_content_sha256"],
        },
    )
    bridge_inventory_path = historical_root / "audit/frozen_pool_inventory.json"
    _atomic_json_for_test(
        bridge_inventory_path,
        {
            f"{route.upper()}_test": {
                "top5_path": str(historical_paths[route].resolve()),
                "top5_artifact_sha256": sha256_file(historical_paths[route]),
            }
            for route in ("g1", "c1")
        },
    )
    historical_formal_lock = historical_root / "08_lock/FORMAL_TEST_LOCK.json"
    _atomic_json_for_test(
        historical_formal_lock,
        {
            "status": "LOCKED",
            "base_run": str(modular_root.resolve()),
            "source_label_artifacts": [
                {
                    "path": str(bridge_ground_truth.resolve()),
                    "sha256": sha256_file(bridge_ground_truth),
                }
            ],
            "audit_artifacts": [
                {
                    "path": str(bridge_inventory_path.resolve()),
                    "sha256": sha256_file(bridge_inventory_path),
                }
            ],
            "candidate_artifacts": [
                {
                    "path": str(
                        (
                            historical_root
                            / f"data/frozen_{route}_test_allnms_candidates.parquet"
                        ).resolve()
                    ),
                    "sha256": sha256_file(
                        historical_root
                        / f"data/frozen_{route}_test_allnms_candidates.parquet"
                    ),
                }
                for route in ("g1", "c1")
            ],
        },
    )
    authority_paths = [
        bridge_inventory_path,
        historical_formal_lock,
        *[
            historical_root / f"data/frozen_{route}_test_{pool}_candidates.parquet"
            for route in ("g1", "c1")
            for pool in ("top5", "allnms")
        ],
    ]
    run_sha_manifest = historical_root / "RUN_SHA256_MANIFEST.txt"
    run_sha_manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(historical_root).as_posix()}\n"
            for path in sorted(authority_paths)
        ),
        encoding="utf-8",
    )
    (historical_root / "RUN_LOCK_SHA256.txt").write_text(
        sha256_file(run_sha_manifest) + "\n", encoding="utf-8"
    )
    bridge_manifest = build_label_free_test_bridge(
        run_dir=run_dir,
        historical_candidates=historical_paths,
        historical_ground_truth_path=bridge_ground_truth,
        historical_source_manifest_path=bridge_source_manifest,
        historical_inventory_path=bridge_inventory_path,
        evaluator_path=evaluator_path,
        denominator_path=sample_path,
        fair_candidates=fair_paths,
        fair_features=feature_paths,
    )
    bridge_manifest_path = (
        run_dir / "11_attribution_bridge/test_bridge_input/manifest.json"
    )
    bridge_record = {
        "path": str(bridge_manifest_path.resolve()),
        "sha256": sha256_file(bridge_manifest_path),
    }
    plan_path = lock_dir / "formal_evaluation_plan.json"
    semantic_provenance = semantic_plan["bound_provenance"]
    headroom_record = semantic_provenance["union_headroom"]
    _atomic_json_for_test(
        plan_path,
        {
            "schema_version": 1,
            "candidate_test_labels_read": False,
            "sample_manifest": str(sample_path),
            "candidate_label_manifest": str(label_manifest_path),
            "systems": systems,
            "union_contract": semantic_plan["union_contract"],
            "test_bridge_contract": {
                "manifest": bridge_record,
                "candidate_bundle": bridge_manifest["artifacts"]["candidate_bundle"],
                "historical_ground_truth": bridge_manifest["sources"][
                    "historical_ground_truth"
                ],
                "evaluator": bridge_manifest["sources"]["evaluator"],
                "denominator": bridge_manifest["sources"]["denominator"],
                "analysis_role": "SECONDARY_POSTLOCK_NO_SELECTION",
                "historical_test_ground_truth_rows_read_preclaim": False,
            },
            "bound_provenance": {
                "fold_assignments": {
                    "path": str(fold_path),
                    "sha256": sha256_file(fold_path),
                },
                "evaluator": {
                    "path": str(evaluator_path),
                    "sha256": sha256_file(evaluator_path),
                },
                "code_manifest": {
                    "path": str(code_manifest_path),
                    "sha256": sha256_file(code_manifest_path),
                },
                "code_bundle_sha256": code_bundle_sha256,
                "primary_selection": semantic_provenance["primary_selection"],
                "screen_latest_execution": semantic_provenance[
                    "screen_latest_execution"
                ],
                "screen_selection_manifest": semantic_provenance[
                    "screen_selection_manifest"
                ],
                "screen_finalists": semantic_provenance["screen_finalists"],
                "selected_latest_execution": semantic_provenance[
                    "selected_latest_execution"
                ],
                "encoder_loss_selection": semantic_provenance[
                    "encoder_loss_selection"
                ],
                "encoder_latest_execution": semantic_provenance[
                    "encoder_latest_execution"
                ],
                "feature_ablation_manifest": semantic_provenance[
                    "feature_ablation_manifest"
                ],
                "feature_extraction_benchmark": semantic_provenance[
                    "feature_extraction_benchmark"
                ],
                "union_headroom": headroom_record,
                "attribution_bridges": bridge_records,
                "test_bridge_manifest": bridge_record,
            },
        },
    )
    return run_dir, plan_path, labels_path


def _add_positive_union_system(
    run_dir: Path,
    plan_path: Path,
    *,
    replace_union_contract: bool = True,
) -> None:
    """Add a route-qualified union whose source IDs collide across all routes."""

    lock_dir = run_dir / "08_lock"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["union_contract"]["decision"] == "UNION_HEADROOM_AVAILABLE":
        positive_run, positive_plan = run_dir, plan
    else:
        positive_run, positive_plan = _build_semantic_p11_evidence(
            run_dir.parent,
            run_dir.parent / "positive_staging",
            positive_union=True,
        )
    positive_contract = positive_plan["union_contract"]
    selection_record = positive_contract["selection_manifest"]
    application_record = positive_contract["label_free_test_application"]
    application = json.loads(
        Path(application_record["path"]).read_text(encoding="utf-8")
    )
    ranking = pd.read_parquet(application["artifacts"]["predictions"]["path"])
    ranking["source_route"] = ranking["source_route"].astype(str).str.lower()
    ranking = ranking.sort_values(
        ["sample_id", "ensemble_score", "native_rank", "candidate_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    ranking["rank"] = ranking.groupby("sample_id", sort=False).cumcount() + 1
    frozen_parts: list[pd.DataFrame] = []
    for route in ROUTES:
        frozen = pd.read_parquet(
            positive_run / f"02_candidates/{route}_test_top5.parquet"
        )[
            [
                "sample_id",
                "candidate_id",
                "native_rank",
                "candidate_geometry_sha256",
            ]
        ].copy()
        frozen["source_route"] = route
        frozen["source_candidate_id"] = frozen["candidate_id"].astype(str)
        frozen["candidate_id"] = (
            route.upper() + ":" + frozen["source_candidate_id"]
        )
        frozen = frozen.rename(
            columns={
                "native_rank": "frozen_native_rank",
                "candidate_geometry_sha256": "frozen_geometry_sha256",
            }
        )
        frozen_parts.append(frozen)
    frozen_union = pd.concat(frozen_parts, ignore_index=True)
    ranking = ranking.merge(
        frozen_union,
        on=["sample_id", "candidate_id", "source_route", "source_candidate_id"],
        validate="one_to_one",
    )
    assert ranking["candidate_geometry_sha256"].equals(
        ranking["frozen_geometry_sha256"]
    )
    ranking = ranking[
        [
            "sample_id",
            "candidate_id",
            "rank",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
            "frozen_native_rank",
        ]
    ]
    ranking_path = lock_dir / "union_top15_ranking.parquet"
    ranking.to_parquet(ranking_path, index=False)
    decisions = ranking.loc[ranking["rank"].eq(1)].rename(
        columns={"candidate_id": "selected_candidate_id"}
    )[
        [
            "sample_id",
            "selected_candidate_id",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ]
    ]
    decision_path = lock_dir / "union_top15_decisions.parquet"
    decisions.to_parquet(decision_path, index=False)
    if replace_union_contract:
        plan["union_contract"] = positive_contract
        plan["bound_provenance"]["union_headroom"] = dict(
            plan["union_contract"]["headroom_manifest"]
        )
    plan["systems"].append(
        {
            "name": "top15_union_primary",
            "kind": "union",
            "route": "cross_route",
            "native_reference": "crog_gated",
            "hypothesis_family": "secondary_cross_route_union",
            "decisions_path": str(decision_path),
            "ranking_path": str(ranking_path),
            "rank_column": "rank",
            "selection_manifest": selection_record,
            "application_manifest": application_record,
        }
    )
    _atomic_json_for_test(plan_path, plan)
    methods_path = lock_dir / "selected_methods.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    positive_methods = json.loads(
        (positive_run / "08_lock/selected_methods.json").read_text(encoding="utf-8")
    )
    methods["union"] = positive_methods["union"]
    _atomic_json_for_test(methods_path, methods)
    features_path = lock_dir / "selected_features.json"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    positive_features = json.loads(
        (positive_run / "08_lock/selected_features.json").read_text(encoding="utf-8")
    )
    features["union"] = positive_features["union"]
    _atomic_json_for_test(features_path, features)
    hyperparameters_path = lock_dir / "selected_hyperparameters.json"
    hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    positive_hyperparameters = json.loads(
        (positive_run / "08_lock/selected_hyperparameters.json").read_text(
            encoding="utf-8"
        )
    )
    hyperparameters["union"] = positive_hyperparameters["union"]
    _atomic_json_for_test(hyperparameters_path, hyperparameters)


def test_union_top15_route_qualified_labels_and_alias_artifacts(tmp_path: Path) -> None:
    run_dir, plan, _labels = _build_synthetic_run(tmp_path, positive_union=True)
    _add_positive_union_system(run_dir, plan)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    result = run_formal_test_once(run_dir=run_dir)
    union_metrics = json.loads(
        (run_dir / "09_formal_test/formal_test_metrics.json").read_text(encoding="utf-8")
    )["systems"]["top15_union_primary"]
    assert union_metrics["j_at_15"] == 1.0
    assert union_metrics["oracle_at_15"] == 1.0
    assert "oracle_at_5" not in union_metrics
    assert {"per_candidate_scores", "per_sample_decisions"}.issubset(
        result["artifacts"]
    )
    execution = json.loads(
        (run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json").read_text(
            encoding="utf-8"
        )
    )
    assert {"per_candidate_scores", "per_sample_decisions"}.issubset(
        execution["artifacts"]
    )
    scores = pd.read_parquet(run_dir / "09_formal_test/per_candidate_scores.parquet")
    union = scores.loc[scores["system_name"].eq("top15_union_primary")]
    assert union.groupby("sample_id").size().eq(6).all()
    assert union["rank"].between(1, 15).all()
    assert set(union["source_candidate_id"]) == {"a", "b"}
    assert set(union["candidate_id"]) == {
        f"{route.upper()}:{candidate}" for route in ROUTES for candidate in ("a", "b")
    }
    decisions = pd.read_parquet(run_dir / "09_formal_test/per_sample_decisions.parquet")
    selected = decisions.loc[decisions["system_name"].eq("top15_union_primary")]
    assert selected["selected_candidate_id"].eq("CROG:a").all()
    assert selected["source_route"].eq("crog").all()
    assert selected["source_candidate_id"].eq("a").all()
    selected_geometry = selected["sample_id"].map(
        union.loc[union["candidate_id"].eq("CROG:a")].set_index("sample_id")[
            "candidate_geometry_sha256"
        ]
    )
    assert selected["candidate_geometry_sha256"].equals(selected_geometry)


def test_union_lock_rejects_non_top1_application_ranking_swap(
    tmp_path: Path,
) -> None:
    run_dir, plan_path, _labels = _build_synthetic_run(
        tmp_path, positive_union=True
    )
    _add_positive_union_system(run_dir, plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    union_system = next(
        system for system in plan["systems"] if system["kind"] == "union"
    )
    ranking_path = Path(union_system["ranking_path"])
    ranking = pd.read_parquet(ranking_path)
    sample_id = str(ranking.loc[ranking["rank"].eq(2), "sample_id"].iloc[0])
    second = ranking.index[
        ranking["sample_id"].astype(str).eq(sample_id) & ranking["rank"].eq(2)
    ][0]
    third = ranking.index[
        ranking["sample_id"].astype(str).eq(sample_id) & ranking["rank"].eq(3)
    ][0]
    ranking.loc[[second, third], "rank"] = [3, 2]
    ranking.to_parquet(ranking_path, index=False)
    with pytest.raises(
        ValueError,
        match=r"union ranking differs from hash-verified Test application predictions",
    ):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan_path)


def test_no_union_headroom_rejects_a_formal_union_system(tmp_path: Path) -> None:
    run_dir, plan, _labels = _build_synthetic_run(tmp_path)
    _add_positive_union_system(
        run_dir, plan, replace_union_contract=False
    )
    with pytest.raises(ValueError, match="NO_UNION_HEADROOM forbids"):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)


def test_union_lock_rejects_invalid_selection_manifest_hash(tmp_path: Path) -> None:
    run_dir, plan, _labels = _build_synthetic_run(tmp_path, positive_union=True)
    _add_positive_union_system(run_dir, plan)
    value = json.loads(plan.read_text(encoding="utf-8"))
    union = next(system for system in value["systems"] if system["kind"] == "union")
    union["selection_manifest"]["sha256"] = "not-a-sha256"
    _atomic_json_for_test(plan, value)
    with pytest.raises(ValueError, match="hash-bound selection_manifest"):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)


def test_formal_lock_is_complete_immutable_and_hash_verified(tmp_path: Path) -> None:
    run_dir, plan, _labels = _build_synthetic_run(tmp_path)
    result = execute_create_unified_formal_lock(
        run_dir=run_dir,
        evaluation_plan_path=plan,
        command="synthetic-lock",
    )
    assert result["status"] == "LOCKED"
    assert result["formal_test_max_execution_count"] == 1
    assert result["declaration"]["candidate_test_labels_read"] is False
    assert set(REQUIRED_LOCK_FILES).issubset(result["locked_files"])
    assert "formal_evaluation_plan" in result["locked_files"]
    assert {
        "formal_provenance_fold_assignments",
        "formal_provenance_evaluator",
        "formal_provenance_code_manifest",
        "formal_provenance_primary_selection",
        "formal_provenance_screen_latest_execution",
        "formal_provenance_screen_selection_manifest",
        "formal_provenance_screen_finalists",
        "formal_provenance_selected_latest_execution",
        "formal_provenance_encoder_loss_selection",
        "formal_provenance_encoder_latest_execution",
        "formal_provenance_feature_ablation_manifest",
        "formal_provenance_feature_extraction_benchmark",
        "formal_provenance_bridge_g1_train",
        "formal_provenance_bridge_g1_validation",
        "formal_provenance_bridge_c1_train",
        "formal_provenance_bridge_c1_validation",
        "formal_union_headroom_manifest",
    }.issubset(result["locked_files"])
    assert verify_formal_test_lock(run_dir)["self_sha256"] == result["self_sha256"]
    with pytest.raises(FileExistsError):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)


@pytest.mark.parametrize(
    "provenance_name",
    (
        "screen_latest_execution",
        "screen_selection_manifest",
        "screen_finalists",
        "selected_latest_execution",
        "encoder_latest_execution",
        "feature_ablation_manifest",
        "feature_extraction_benchmark",
    ),
)
def test_lock_rejects_post_p11_selection_evidence_mutation(
    tmp_path: Path,
    provenance_name: str,
) -> None:
    run_dir, plan_path, _labels = _build_synthetic_run(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    evidence_path = Path(plan["bound_provenance"][provenance_name]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["post_p11_tamper"] = True
    _atomic_json_for_test(evidence_path, evidence)
    with pytest.raises(
        ValueError,
        match=rf"formal bound_provenance hash mismatch: {provenance_name}",
    ):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan_path)


def test_lock_rejects_prevalidation_state_and_gate_router_contract_violations(
    tmp_path: Path,
) -> None:
    run_dir, plan, _labels = _build_synthetic_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["test_label_state"] = "LOCKED_PREVALIDATION"
    _atomic_json_for_test(manifest_path, manifest)
    with pytest.raises(PermissionError, match="not in a pre-Test lockable state"):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    manifest["test_label_state"] = "VALIDATION_SELECTION_COMPLETE"
    _atomic_json_for_test(manifest_path, manifest)
    methods_path = run_dir / "08_lock/selected_methods.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    methods["routes"]["crog"]["seeds"] = [42]
    _atomic_json_for_test(methods_path, methods)
    with pytest.raises(ValueError, match="route contract is incomplete"):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    methods["routes"]["crog"]["seeds"] = [42, 123, 2026]
    _atomic_json_for_test(methods_path, methods)
    plan_value = json.loads(plan.read_text(encoding="utf-8"))
    ungated = next(
        system for system in plan_value["systems"] if system["name"] == "crog_ungated"
    )
    family = ungated.pop("hypothesis_family")
    _atomic_json_for_test(plan, plan_value)
    with pytest.raises(ValueError, match="predeclared hypothesis_family"):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    ungated["hypothesis_family"] = family
    _atomic_json_for_test(plan, plan_value)
    gated = next(system for system in plan_value["systems"] if system["name"] == "crog_gated")
    frame = pd.read_parquet(gated["decisions_path"])
    frame.loc[0, "selected_candidate_id"] = "outside-pool"
    frame.to_parquet(gated["decisions_path"], index=False)
    with pytest.raises(ValueError, match="outside native/ungated"):
        create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)


def test_lock_rejects_native_geometry_and_router_gated_mismatch(tmp_path: Path) -> None:
    geometry_run, geometry_plan, _ = _build_synthetic_run(tmp_path / "geometry")
    geometry_value = json.loads(geometry_plan.read_text(encoding="utf-8"))
    native = next(system for system in geometry_value["systems"] if system["name"] == "g1_native")
    ranking = pd.read_parquet(native["ranking_path"])
    ranking.loc[0, "candidate_geometry_sha256"] = "tampered-geometry"
    ranking.to_parquet(native["ranking_path"], index=False)
    with pytest.raises(ValueError, match="geometry/native-rank binding"):
        create_unified_formal_lock(
            run_dir=geometry_run, evaluation_plan_path=geometry_plan
        )

    router_run, router_plan, _ = _build_synthetic_run(tmp_path / "router")
    router_value = json.loads(router_plan.read_text(encoding="utf-8"))
    router = next(system for system in router_value["systems"] if system["kind"] == "router")
    decisions = pd.read_parquet(router["decisions_path"])
    decisions.loc[0, "selected_candidate_id"] = "a"
    decisions.to_parquet(router["decisions_path"], index=False)
    with pytest.raises(ValueError, match="locked gated selection"):
        create_unified_formal_lock(run_dir=router_run, evaluation_plan_path=router_plan)


def test_lock_hash_mismatch_refuses_before_execution_or_label_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, plan, labels = _build_synthetic_run(tmp_path)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    system = json.loads(plan.read_text(encoding="utf-8"))["systems"][0]
    changed = pd.read_parquet(system["decisions_path"])
    changed.loc[0, "selected_candidate_id"] = "b"
    changed.to_parquet(system["decisions_path"], index=False)
    import tools.unified_reranking.run_formal_test_once as module

    opened = False

    def forbidden_open(path: Path):
        nonlocal opened
        opened = True
        raise AssertionError("candidate labels must not open")

    monkeypatch.setattr(module, "_read_candidate_labels_once", forbidden_open)
    with pytest.raises(PermissionError, match="hash mismatch"):
        run_formal_test_once(run_dir=run_dir, candidate_test_labels_path=labels)
    assert opened is False
    assert not (run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json").exists()


def test_formal_test_claims_before_single_label_read_and_finalizes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, plan, labels = _build_synthetic_run(tmp_path)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    import tools.unified_reranking.run_formal_test_once as module

    original = module._read_candidate_labels_once
    opens = 0

    def checked_open(path: Path):
        nonlocal opens
        opens += 1
        execution = json.loads(
            (run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert execution["status"] == "RUNNING"
        assert execution["execution_count"] == 1
        assert manifest["formal_test_execution_count"] == 1
        return original(path)

    monkeypatch.setattr(module, "_read_candidate_labels_once", checked_open)
    result = execute_formal_test_once(
        run_dir=run_dir,
        command="synthetic-formal-test",
    )
    assert opens == 1
    assert result["status"] == "COMPLETE"
    assert result["candidate_test_labels"]["opened_once_after_claim"] is True
    assert result["candidate_test_labels"]["sha256"] == sha256_file(labels)
    execution = json.loads(
        (run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["status"] == "COMPLETE"
    assert execution["execution_count"] == 1
    assert "formal_test_outcomes_wide" in execution["artifacts"]
    assert "bridge_per_candidate_scores" in execution["artifacts"]
    bridge_scores = pd.read_parquet(
        run_dir / "09_formal_test/bridge_per_candidate_scores.parquet"
    )
    assert len(bridge_scores) == 8 * 2 * 2 * 2
    assert bridge_scores["candidate_success"].all()
    assert set(bridge_scores["candidate_pool_contract"]) == {
        "fair_gaussian",
        "historical_nms",
    }
    access_events = [
        json.loads(line)["event"]
        for line in (run_dir / "09_formal_test/test_access.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ordered_events = [
        "formal_test_exclusive_claim_created",
        "candidate_test_label_access_authorized",
        "candidate_test_labels_read_once",
        "formal_bridge_test_ground_truth_read_once",
        "bridge_per_candidate_scores_persisted",
        "formal_test_execution_finalized",
    ]
    assert [access_events.index(event) for event in ordered_events] == sorted(
        access_events.index(event) for event in ordered_events
    )
    final_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["formal_test_execution_count"] == 1
    assert final_manifest["test_label_state"] == "FORMAL_TEST_COMPLETE"

    rankings = pd.read_parquet(
        run_dir / "09_formal_test/formal_test_realized_rankings.parquet"
    )
    assert set(rankings["system_name"]) == {
        f"{route}_{kind}" for route in ROUTES for kind in ("native", "ungated", "gated")
    }
    assert rankings.groupby(["system_name", "sample_id"]).size().eq(2).all()
    outcomes = pd.read_parquet(run_dir / "09_formal_test/formal_test_outcomes_wide.parquet")
    assert {"sample_id", "scene_id", "frame_id"}.issubset(outcomes.columns)
    assert "crog_native__selected_correct" in outcomes
    assert "crog_ungated__selected_correct" in outcomes
    assert "crog_gated__selected_correct" in outcomes
    assert "crog_default_router__selected_correct" in outcomes
    per_sample = pd.read_parquet(run_dir / "09_formal_test/formal_test_per_sample.parquet")
    assert {"scene_id", "frame_id", "system_kind", "selected_correct"}.issubset(
        per_sample.columns
    )
    metrics = json.loads(
        (run_dir / "09_formal_test/formal_test_metrics.json").read_text(encoding="utf-8")
    )["systems"]
    for route in ROUTES:
        for kind in ("native", "ungated", "gated"):
            values = metrics[f"{route}_{kind}"]
            assert values["j_at_5"] == 1.0
            assert values["oracle_all"] == 1.0
    statistics = json.loads(
        (run_dir / "09_formal_test/formal_test_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert statistics["hypothesis_families"]["primary_confirmatory"][
        "comparison_count"
    ] == 4
    assert statistics["hypothesis_families"]["ungated_secondary"][
        "comparison_count"
    ] == 3
    assert statistics["comparisons"]["crog_default_router"]["reference"] == "crog_gated"
    access_events = [
        json.loads(line)["event"]
        for line in (run_dir / "09_formal_test/test_access.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert access_events.index("formal_test_exclusive_claim_created") < access_events.index(
        "candidate_test_labels_read_once"
    )
    with pytest.raises(PermissionError, match="already"):
        execute_formal_test_once(
            run_dir=run_dir,
            candidate_test_labels_path=labels,
            command="forbidden-second-test",
        )


def test_failed_postclaim_evaluation_consumes_the_only_execution(tmp_path: Path) -> None:
    run_dir, plan, labels = _build_synthetic_run(tmp_path)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    invalid = pd.read_parquet(labels).drop(columns="candidate_success")
    invalid.to_parquet(labels, index=False)
    with pytest.raises(PermissionError, match="label hash"):
        run_formal_test_once(run_dir=run_dir, candidate_test_labels_path=labels)
    execution = json.loads(
        (run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["status"] == "RUNNING"
    assert execution["execution_count"] == 1
    with pytest.raises(PermissionError, match="exclusive claim"):
        run_formal_test_once(run_dir=run_dir, candidate_test_labels_path=labels)


def test_postclaim_labels_must_exactly_cover_locked_all_pools(tmp_path: Path) -> None:
    run_dir, plan, labels = _build_synthetic_run(tmp_path)
    incomplete = pd.read_parquet(labels).iloc[:-1].copy()
    incomplete.to_parquet(labels, index=False)
    plan_value = json.loads(plan.read_text(encoding="utf-8"))
    label_manifest_path = Path(plan_value["candidate_label_manifest"])
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    label_manifest["candidate_labels_sha256"] = sha256_file(labels)
    _atomic_json_for_test(label_manifest_path, label_manifest)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    with pytest.raises(ValueError, match="exactly cover locked All pools"):
        run_formal_test_once(run_dir=run_dir)
    execution = json.loads(
        (run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["status"] == "RUNNING"
    assert execution["execution_count"] == 1


def test_cli_cannot_redirect_to_an_unlocked_label_file(tmp_path: Path) -> None:
    run_dir, plan, labels = _build_synthetic_run(tmp_path)
    create_unified_formal_lock(run_dir=run_dir, evaluation_plan_path=plan)
    alternate = tmp_path / "alternate_labels.parquet"
    alternate.write_bytes(labels.read_bytes())
    with pytest.raises(PermissionError, match="differs from the predeclared"):
        run_formal_test_once(
            run_dir=run_dir, candidate_test_labels_path=alternate
        )
    execution = json.loads(
        (run_dir / "09_formal_test/FORMAL_TEST_EXECUTION.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["status"] == "RUNNING"
    events = (run_dir / "09_formal_test/test_access.log").read_text(encoding="utf-8")
    assert "candidate_test_labels_read_once" not in events
