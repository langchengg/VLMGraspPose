from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

from d1_reranking.ablation import (
    ABLATION_EXECUTION_POINTER_RELATIVE,
    EVIDENCE_TRACKS,
    FEATURE_FAMILIES,
    REPORTING_METHODS,
    RUNNER_MODULE,
    _load_track_manifest,
    ablation_jobs,
    feature_ablation_variants,
    feature_family_registry,
    validate_ablation_plan,
)
from d1_reranking.ablation_execution import (
    ABLATION_RESOURCE_POLICY_POINTER_RELATIVE,
    ablation_execution_scope,
    ablation_resource_policy,
    create_execution_authority,
    load_ablation_resource_policy,
)
from d1_reranking.ablation_replay import validate_track_gate_artifacts
from d1_reranking.ablation_selection import (
    compute_track_gate,
    feature_ablation_rows,
    gate_grid_points,
)
from d1_reranking.execution import artifact_record
from d1_reranking.io import atomic_parquet, atomic_pickle
from unified_reranking.gate import BOOTSTRAP_ITERATIONS, SAFE_GATE_FEATURE_COLUMNS
from unified_reranking.hashing import atomic_json, canonical_sha256
from unified_reranking.training import FORMAL_SEEDS


ALL_FAMILY_COLUMNS = (
    "native_score",
    "soft_target_support_mean",
    "grasp_width_px",
    "depth_mean",
    "contact_normal_dot",
    "clearance_min",
    "nearest_candidate_distance",
    "overall_reliability",
    "crop_latent_0",
    "t4_route_agreement",
)


def _selected_primary() -> dict[str, object]:
    configuration = {
        "route": "D1",
        "pool": "top5",
        "track": "T2_matched_common",
        "method": "R5",
        "encoder": "lambdamart",
        "loss": "lambdarank",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
    }
    return {
        "method": "R5",
        "trial_id": "trial-r5",
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
        "seed_policy": (
            "fixed seeds 42,123,2026; score ensemble; best-seed selection forbidden"
        ),
    }


def _plan() -> dict[str, object]:
    registry = feature_family_registry(ALL_FAMILY_COLUMNS)
    variants = feature_ablation_variants(registry)
    track_columns = {
        "T1_native_available": ("native_score",),
        "T2_matched_common": ("native_score", "soft_target_support_mean"),
        "T3_route_rich": ("native_score", "depth_mean"),
        "T4_four_route_consensus": ALL_FAMILY_COLUMNS,
    }
    selected = _selected_primary()
    jobs = ablation_jobs(
        selected_primary=selected,
        track_columns=track_columns,
        variants=variants,
    )
    sources: dict[str, object] = {}
    value: dict[str, object] = {
        "schema_version": 1,
        "status": "PLANNED",
        "route": "D1",
        "pool": "top5",
        "analysis": "evidence_track_and_feature_family_ablation",
        "development_splits": ["train", "validation"],
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "selected_primary": selected,
        "retuning_permitted": False,
        "reporting_methods": list(REPORTING_METHODS),
        "formal_seeds": list(FORMAL_SEEDS),
        "outer_folds": 5,
        "evidence_tracks": {
            track: {
                "status": "REUSED_PRIMARY_EXACT"
                if track == "T2_matched_common"
                else "AVAILABLE",
                "feature_columns": list(track_columns[track]),
            }
            for track in EVIDENCE_TRACKS
        },
        "feature_family_registry_version": "d1_p10_feature_families_v1",
        "feature_family_registry": registry,
        "feature_ablation_variants": variants,
        "execution_contract": {
            "runner_module": RUNNER_MODULE,
            "authorization_tool": ("tools.d1_reranking.authorize_ablation_execution"),
            "orchestrator_tool": "tools.d1_reranking.run_ablation_matrix",
            "direct_cli_execution_permitted": False,
            "execution_manifest": str(ABLATION_EXECUTION_POINTER_RELATIVE),
            "execution_state_transitions": ["ACTIVE", "COMPLETE", "FAILED"],
            "max_parallel": 1,
            "device": "cpu",
            "fresh_resource_gate_required_on_start_and_resume": True,
            "global_heavy_resource_lease_required": True,
        },
        "jobs": jobs,
        "job_count": len(jobs),
        "job_universe_sha256": canonical_sha256([job["configuration"] for job in jobs]),
        "job_ids_sha256": canonical_sha256([job["job_id"] for job in jobs]),
        "source_signature_sha256": canonical_sha256(sources),
        "sources": sources,
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def _manifest(path: Path, value: dict[str, object]) -> Path:
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(path, payload)
    return path


def _gate_frame(rows: int, *, validation: bool) -> pd.DataFrame:
    rng = np.random.default_rng(1729 + int(validation))
    frame = pd.DataFrame(
        rng.uniform(0.05, 0.95, size=(rows, len(SAFE_GATE_FEATURE_COLUMNS))),
        columns=SAFE_GATE_FEATURE_COLUMNS,
    )
    index = np.arange(rows)
    frame["sample_id"] = [f"sample-{index_:03d}" for index_ in index]
    frame["scene_id"] = [f"scene-{index_:03d}" for index_ in index]
    frame["native_correct"] = index % 4 < 2
    frame["challenger_correct"] = np.isin(index % 4, [0, 2])
    frame["score_margin"] = frame["ranker_score_margin"]
    frame["challenger_reliability"] = 0.9
    frame["perturbation_stability"] = 1.0
    frame["seed_challenger_votes"] = 3
    frame["candidate_id_unchanged"] = True
    frame["geometry_hash_unchanged"] = True
    frame["challenger_exists"] = True
    frame["native_candidate_id"] = [f"native-{item}" for item in index]
    frame["challenger_candidate_id"] = [f"challenger-{item}" for item in index]
    if not validation:
        frame["oof_fold"] = index % 5
    return frame


def _grid() -> dict[str, object]:
    return {
        "configuration": {
            "lambda_harm": [1.0, 2.0, 4.0],
            "utility_thresholds": [-0.1, 0.0, 0.1],
            "score_margin_thresholds": [0.0, 0.1, 0.2],
            "reliability_thresholds": [0.5, 0.8],
            "stability_thresholds": [0.5, 0.9],
            "minimum_seed_votes": 2,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": 20260808,
        }
    }


def _gate_manifest(
    tmp_path: Path, train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, object], tuple[object, ...]]:
    points = gate_grid_points(_grid())
    replay = compute_track_gate(train=train, validation=validation, points=points)
    artifacts = {
        "train_gate_inputs": artifact_record(
            atomic_parquet(train, tmp_path / "train.parquet")
        ),
        "validation_gate_inputs": artifact_record(
            atomic_parquet(validation, tmp_path / "validation.parquet")
        ),
        "transition_model": artifact_record(
            atomic_pickle(replay["model"], tmp_path / "model.pkl")
        ),
        "validation_decisions": artifact_record(
            atomic_parquet(replay["decisions"], tmp_path / "decisions.parquet")
        ),
        "validation_trials": artifact_record(
            atomic_parquet(replay["trials"], tmp_path / "trials.parquet")
        ),
    }
    gate = {
        "selection": replay["selection"],
        "decision": replay["decision"],
        "transition_model": replay["model"].artifact(),
        "metrics": replay["metrics"],
        "artifacts": artifacts,
    }
    return gate, points


def test_ablation_plan_freezes_exact_cells_and_rejects_rehashed_drift() -> None:
    plan = _plan()
    validated = validate_ablation_plan(plan)

    assert validated["job_count"] == 414
    assert all(job["configuration"]["method"] == "R5" for job in plan["jobs"])
    assert all(
        job["configuration"]["candidate_test_labels_read"] is False
        and job["configuration"]["test_inputs_referenced"] is False
        for job in plan["jobs"]
    )
    assert (
        sum(
            job["configuration"]["analysis"] == "evidence_track" for job in plan["jobs"]
        )
        == 54
    )
    tampered = deepcopy(plan)
    tampered["jobs"] = tampered["jobs"][:-1]
    tampered["job_count"] = len(tampered["jobs"])
    tampered["job_universe_sha256"] = canonical_sha256(
        [job["configuration"] for job in tampered["jobs"]]
    )
    tampered["job_ids_sha256"] = canonical_sha256(
        [job["job_id"] for job in tampered["jobs"]]
    )
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(RuntimeError, match="exact job universe"):
        validate_ablation_plan(tampered)


def test_feature_registry_preserves_empty_families_as_not_available() -> None:
    registry = feature_family_registry(("native_score",))
    variants = feature_ablation_variants(registry)

    assert tuple(registry) == FEATURE_FAMILIES
    assert registry["q_native_calibration"]["status"] == "AVAILABLE"
    assert registry["four_route_consensus"]["status"] == "NOT_AVAILABLE"
    assert len(variants) == 20
    assert all(
        variant["status"] == "NOT_AVAILABLE"
        for variant in variants
        if variant["target_family"] == "four_route_consensus"
    )


def test_widened_t4_route_features_have_predeclared_families() -> None:
    columns = (
        "candidate_count",
        "d1_route_local_neighbour_q_mean",
        "d1_route_collision_proxy_total",
        "d1_route_contact_symmetry",
        "d1_route_estimated_local_object_thickness",
        "d1_route_mask_entropy_local",
        "d1_route_p_axis_mean",
        "d1_route_q_gap_to_top1",
        "pool_score_entropy",
        "rank_percentile",
        "z_center",
    )
    registry = feature_family_registry(columns)
    assigned = {
        column: family
        for family, value in registry.items()
        for column in value["columns"]
    }
    assert set(assigned) == set(columns)
    assert assigned["candidate_count"] == "relations"
    assert assigned["d1_route_collision_proxy_total"] == "clearance"
    assert assigned["d1_route_contact_symmetry"] == "contacts"
    assert assigned["d1_route_mask_entropy_local"] == "soft_target_support"
    assert assigned["d1_route_q_gap_to_top1"] == "q_native_calibration"
    assert assigned["z_center"] == "depth"


def test_t1_legacy_manifest_normalizes_to_the_p10_track_contract(
    tmp_path: Path,
) -> None:
    feature_path = atomic_parquet(
        pd.DataFrame(
            {
                "sample_id": ["s0"],
                "candidate_id": ["c0"],
                "native_score": [0.7],
            }
        ),
        tmp_path / "candidate_features.parquet",
    )
    columns = ["native_score"]
    schema_path = tmp_path / "feature_schema.json"
    atomic_json(
        schema_path,
        {
            "schema_version": 1,
            "track": "T1_native_available",
            "model_columns": columns,
            "model_schema_sha256": canonical_sha256(columns),
            "candidate_test_labels_read": False,
        },
    )
    manifest_path = _manifest(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "configuration": {
                "split": "train",
                "pool": "top5",
                "track": "T1_native_available",
                "candidate_test_labels_read": False,
            },
            "artifact": artifact_record(feature_path),
            "feature_schema": artifact_record(schema_path),
            "candidate_test_labels_read": False,
        },
    )

    _manifest_value, observed = _load_track_manifest(
        manifest_path, split="train", track="T1_native_available"
    )
    assert observed == ("native_score",)


def test_widened_t4_realistic_columns_cover_all_predeclared_families() -> None:
    columns = (
        "native_score_raw",
        "p_center",
        "width_to_grasp_span_ratio",
        "local_depth_median",
        "left_contact_probability_mean",
        "finger_sweep_obstacle_max",
        "nearest_candidate_distance",
        "peak_retention_rate",
        "d1_route_gqcnn_crop_mean",
        "t4_crog_agreement",
    )
    registry = feature_family_registry(columns)

    assert all(registry[family]["status"] == "AVAILABLE" for family in FEATURE_FAMILIES)
    assert [registry[family]["columns"][0] for family in FEATURE_FAMILIES] == list(
        columns
    )


def test_feature_ablation_recomputes_per_seed_deltas() -> None:
    plan = _plan()
    results: dict[str, dict[str, object]] = {}
    variants = plan["feature_ablation_variants"]
    for variant_index, variant in enumerate(variants):
        if variant["status"] != "AVAILABLE":
            continue
        for seed_index, seed in enumerate(FORMAL_SEEDS):
            key = f"{variant['variant_id']}-{seed}"
            results[key] = {
                "configuration": {
                    "analysis": "feature_family",
                    "mode": "validation",
                    "variant_id": variant["variant_id"],
                    "seed": seed,
                },
                "metrics": {"j_at_1": 0.4 + variant_index / 100 + seed_index / 1000},
            }
    rows = feature_ablation_rows(plan, results)
    full = next(
        row for row in rows if row["variant_id"] == "cumulative_09_four_route_consensus"
    )
    assert full["delta_vs_full_t4_mean"] == pytest.approx(0.0)
    tampered = deepcopy(results)
    key = "cumulative_00_q_native_calibration-42"
    tampered[key]["metrics"]["j_at_1"] += 0.2
    tampered_rows = feature_ablation_rows(plan, tampered)
    assert tampered_rows[0]["delta_vs_full_t4_mean"] != rows[0]["delta_vs_full_t4_mean"]


def test_track_gate_replay_rejects_rehashed_trial_model_and_decision(
    tmp_path: Path,
) -> None:
    train = _gate_frame(60, validation=False)
    validation = _gate_frame(32, validation=True)
    gate, points = _gate_manifest(tmp_path, train, validation)

    replay = validate_track_gate_artifacts(
        gate, train=train, validation=validation, points=points
    )
    assert len(replay["trials"]) == 108

    trial_tamper = deepcopy(gate)
    trials_path = Path(trial_tamper["artifacts"]["validation_trials"]["path"])
    trials = pd.read_parquet(trials_path)
    trials.loc[0, "switch_count"] += 1
    atomic_parquet(trials, trials_path)
    trial_tamper["artifacts"]["validation_trials"] = artifact_record(trials_path)
    with pytest.raises(RuntimeError, match="gate trials"):
        validate_track_gate_artifacts(
            trial_tamper, train=train, validation=validation, points=points
        )

    atomic_parquet(replay["trials"], trials_path)
    model_tamper = deepcopy(gate)
    model_path = Path(model_tamper["artifacts"]["transition_model"]["path"])
    model_path.write_bytes(pickle.dumps({"tampered": True}, protocol=5))
    model_tamper["artifacts"]["transition_model"] = artifact_record(model_path)
    with pytest.raises(RuntimeError, match="model bytes"):
        validate_track_gate_artifacts(
            model_tamper, train=train, validation=validation, points=points
        )

    atomic_pickle(replay["model"], model_path)
    decision_tamper = deepcopy(gate)
    decision_tamper["decision"] = (
        "GO" if gate["decision"] == "NO_GO_NATIVE" else "NO_GO_NATIVE"
    )
    with pytest.raises(RuntimeError, match="selection/model/metric"):
        validate_track_gate_artifacts(
            decision_tamper, train=train, validation=validation, points=points
        )


def test_ablation_policy_and_zero_output_recovery_are_exact(tmp_path: Path) -> None:
    root = tmp_path / "run"
    plan = _plan()
    plan_path = root / "configs/d1_ablation_plan_v1.json"
    atomic_json(plan_path, plan)
    policy = ablation_resource_policy(plan_path=plan_path, plan=plan)

    assert policy["scope"] == ablation_execution_scope(plan)
    assert policy["plan"] == artifact_record(plan_path)
    assert policy["source_signature_sha256"] == canonical_sha256(policy["sources"])
    policy_path = _manifest(
        root / "configs/ablation_resource_policies/policy.json",
        {key: value for key, value in policy.items() if key != "content_sha256"},
    )
    pointer: dict[str, object] = {
        "schema_version": 1,
        "status": "LOCKED_POLICY_POINTER",
        "active_policy": artifact_record(policy_path),
        "plan": artifact_record(plan_path),
        "scope": ablation_execution_scope(plan),
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
    }
    pointer["content_sha256"] = canonical_sha256(pointer)
    atomic_json(root / ABLATION_RESOURCE_POLICY_POINTER_RELATIVE, pointer)
    loaded_path, loaded = load_ablation_resource_policy(
        root, plan_path=plan_path, plan=plan
    )
    assert loaded_path == policy_path.resolve()
    assert loaded == policy
    gate_path = _manifest(
        root / "00_audit/resource_gates/gate.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "policy": artifact_record(policy_path),
            "candidate_test_labels_read": False,
        },
    )
    failed_path = _manifest(
        root / "configs/ablation_executions/prior/events/0001_failed.json",
        {
            "schema_version": 1,
            "status": "FAILED",
            "outputs": {},
            "candidate_test_labels_read": False,
        },
    )
    execution_path, execution = create_execution_authority(
        root,
        plan_path=plan_path,
        plan=plan,
        resource_gate_path=gate_path,
        owner="synthetic",
        resume_from=artifact_record(failed_path),
        resume_outputs={},
    )
    assert execution_path.is_file()
    assert execution["resume_outputs"] == {}
    assert execution["sources"]["resource_policy"] == artifact_record(policy_path)
    assert execution["candidate_test_labels_read"] is False
    assert '"split": "test"' not in json.dumps(plan).lower()

    tampered_policy = deepcopy(policy)
    tampered_policy["sources"][0] = artifact_record(plan_path)
    tampered_policy["source_signature_sha256"] = canonical_sha256(
        tampered_policy["sources"]
    )
    tampered_policy["content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered_policy.items()
            if key != "content_sha256"
        }
    )
    atomic_json(policy_path, tampered_policy)
    pointer["active_policy"] = artifact_record(policy_path)
    pointer["content_sha256"] = canonical_sha256(
        {key: value for key, value in pointer.items() if key != "content_sha256"}
    )
    atomic_json(root / ABLATION_RESOURCE_POLICY_POINTER_RELATIVE, pointer)
    with pytest.raises(RuntimeError, match="policy/plan/code binding differs"):
        load_ablation_resource_policy(root, plan_path=plan_path, plan=plan)


def test_ablation_cell_cannot_represent_test_label_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.d1_reranking.run_ablation_cell import _load_split

    def forbidden_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("no parquet reader may run for rejected Test input")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    with pytest.raises(PermissionError, match="cannot represent Test"):
        _load_split(
            {},
            split="test",
            track="T4_four_route_consensus",
            selected_columns=("native_score",),
        )
