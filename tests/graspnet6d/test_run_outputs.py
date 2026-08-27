from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import graspnet6d.run_outputs as run_outputs
from graspnet6d.analysis_inputs import AnalysisInputAssembly
from graspnet6d.experiment_analysis import ANALYSIS_SCHEMA, FORMAL_SCOPE
from graspnet6d.io import canonical_sha256, sha256_file
from graspnet6d.provenance import RUN_MANIFEST_SCHEMA_VERSION
from graspnet6d.run_outputs import (
    RunOutputError,
    publish_candidate_aggregates,
    publish_environment,
    publish_selected_analysis_outputs,
)
from graspnet6d.stages import LABEL_BUNDLE_SCHEMA


CONDITIONS = (
    "oracle_gt_mask",
    "hifics_zero_shot_mask",
    "hifics_adapted_mask",
)


def _json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _formal_run(
    tmp_path: Path,
    *,
    name: str = "formal-paper-run-001",
    status: str = "IN_PROGRESS",
    environment: dict[str, Any] | None = None,
) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    captured = environment or {
        "recorded_at_utc": "2026-01-01T00:00:00+00:00",
        "storage": {"free_bytes": 100},
        "hardware": {"chip_type": "Test CPU", "cpu_count": 8},
        "platform": "test-platform",
        "privacy": "identifiers omitted",
        "python": {"version": "3.11", "executable": "/test/python"},
        "torch": {"version": "2.0", "mps_available": False},
    }
    _json(
        root / "run_manifest.json",
        {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": root.name,
            "profile": "paper-lite",
            "status": status,
            "immutable_identity": {
                "resolved_config": {
                    "formal_results": True,
                    "vgn": {"device": "cpu"},
                }
            },
            "hardware_environment": captured,
            "software": {"python": "3.11", "platform": "test-platform"},
        },
    )
    return root


def _environment_evidence(
    run_dir: Path,
    environment: dict[str, Any],
) -> tuple[Path, Path]:
    audit = run_dir.parent / "audit"
    devices = run_dir.parent
    _json(audit / "hardware_environment.json", environment)
    _json(audit / "system_profile.json", environment)
    (audit / "pip_freeze.txt").write_text("numpy==2.0.0\n", encoding="utf-8")
    (devices / "device_benchmark.csv").write_text(
        "device,available,successful,finite_rate,median_latency_s,p95_latency_s\n"
        "cpu,True,True,1.0,0.1,0.2\n"
        "mps,False,False,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (devices / "device_decision.md").write_text(
        "# Device decision\n\nFormal inference device: **CPU**\n",
        encoding="utf-8",
    )
    return audit, devices


def test_environment_is_hash_strict_but_tolerates_refreshed_audit_time_and_storage(
    tmp_path: Path,
) -> None:
    captured = {
        "recorded_at_utc": "2026-01-01T00:00:00+00:00",
        "storage": {"free_bytes": 100},
        "hardware": {"chip_type": "Test CPU", "cpu_count": 8},
        "platform": "test-platform",
        "privacy": "identifiers omitted",
        "python": {"version": "3.11", "executable": "/test/python"},
        "torch": {"version": "2.0", "mps_available": False},
    }
    run_dir = _formal_run(tmp_path, status="BLOCKED", environment=captured)
    refreshed = {
        **captured,
        "recorded_at_utc": "2026-01-02T00:00:00+00:00",
        "storage": {"free_bytes": 80},
    }
    audit, devices = _environment_evidence(run_dir, refreshed)
    published = publish_environment(run_dir, audit_root=audit, device_root=devices)
    assert published.resumed is False
    assert (run_dir / "environment.txt").is_file()
    text = (run_dir / "environment.txt").read_text(encoding="utf-8")
    assert "Run-captured hardware / system profile (primary)" in text
    assert "Current refreshed audit" in text

    second_refresh = {
        **captured,
        "recorded_at_utc": "2026-01-03T00:00:00+00:00",
        "storage": {"free_bytes": 60},
    }
    _json(audit / "hardware_environment.json", second_refresh)
    _json(audit / "system_profile.json", second_refresh)
    resumed = publish_environment(
        run_dir, audit_root=audit, device_root=devices, resume=True
    )
    assert resumed.resumed is True
    assert resumed.manifest_sha256 == published.manifest_sha256

    (audit / "pip_freeze.txt").write_text("numpy==2.1.0\n", encoding="utf-8")
    with pytest.raises(RunOutputError, match="immutable environment evidence changed"):
        publish_environment(run_dir, audit_root=audit, device_root=devices, resume=True)


def test_environment_rejects_a_different_host_and_fixture_run(tmp_path: Path) -> None:
    run_dir = _formal_run(tmp_path)
    captured = json.loads((run_dir / "run_manifest.json").read_text())[
        "hardware_environment"
    ]
    changed = {**captured, "hardware": {"chip_type": "Other CPU", "cpu_count": 8}}
    audit, devices = _environment_evidence(run_dir, changed)
    with pytest.raises(RunOutputError, match="differs from the run-captured"):
        publish_environment(run_dir, audit_root=audit, device_root=devices)

    fixture = _formal_run(tmp_path, name="fixture-formal-run")
    fixture_environment = json.loads((fixture / "run_manifest.json").read_text())[
        "hardware_environment"
    ]
    fixture_audit, fixture_devices = _environment_evidence(fixture, fixture_environment)
    with pytest.raises(RunOutputError, match="fixture-labelled"):
        publish_environment(
            fixture, audit_root=fixture_audit, device_root=fixture_devices
        )


def _manifest_rows(run_dir: Path) -> None:
    targets = []
    languages = []
    for group_id, scene_id, split in (
        ("group-real", "scene-001", "train"),
        ("group-empty", "scene-002", "test"),
    ):
        targets.append(
            {
                "group_id": group_id,
                "scene_id": scene_id,
                "split": split,
                "camera": "realsense",
                "frame_id": 0,
                "target_object_id": 1,
                "target_instance_label": 1,
                "depth_path": "depth.png",
                "instance_label_path": "label.png",
                "meta_path": "meta.mat",
                "intrinsics_path": "intrinsics.npy",
                "camera_pose_path": "camera.npy",
                "table_transform_path": "align.npy",
            }
        )
        languages.append(
            {
                "group_id": group_id,
                "query": "the test object",
                "is_unique": True,
                "resolver_result": [1],
            }
        )
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "target_groups.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8"
    )
    (manifest_dir / "language_queries.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in languages), encoding="utf-8"
    )


def _canonical_label(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "candidate_index": 0,
        "target_object_id": 1,
        "associated_object_id": 1,
        "target_match": True,
        "correct_target": True,
        "collision": False,
        "empty_grasp": False,
        "pose_valid": True,
        "valid_geometry": True,
        "friction_required": 0.4,
        "friction_score": 0.4,
        "relevance": 5,
        "success_mu_0.2": False,
        "success_mu_0.4": True,
        "success_mu_0.6": True,
        "success_mu_0.8": True,
        "success_mu_1.0": True,
        "success_mu_1.2": True,
    }


def _candidate_fixture(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_condition: str | None = None,
) -> tuple[dict[str, AnalysisInputAssembly], dict[Path, Any], dict[str, Path]]:
    _manifest_rows(run_dir)
    schema = run_dir / "assembled-feature-schema.json"
    schema.write_text('{"schema":"exact-test-schema"}\n', encoding="utf-8")
    assemblies: dict[str, AnalysisInputAssembly] = {}
    loaded_by_path: dict[Path, Any] = {}
    label_paths: dict[str, Path] = {}
    for condition in CONDITIONS:
        assembly_dir = run_dir / "assembled" / condition
        assembly_dir.mkdir(parents=True)
        input_manifest = _json(assembly_dir / "input_manifest.json", {"test": True})
        commits = []
        for group_id, scene_id, split, count in (
            ("group-real", "scene-001", "train", 1),
            ("group-empty", "scene-002", "test", 0),
        ):
            stem = assembly_dir / group_id
            candidate = _json(stem.with_suffix(".candidate.json"), {"pool": group_id})
            official = stem.with_suffix(".official.txt")
            official.write_text("official-source\n", encoding="utf-8")
            candidate_id = f"{condition}-{group_id}-candidate"
            labels = [_canonical_label(candidate_id)] if count else []
            bundle_condition = (
                "oracle_gt_mask"
                if wrong_condition == condition and group_id == "group-real"
                else condition
            )
            label_payload: dict[str, Any] = {
                "schema_version": LABEL_BUNDLE_SCHEMA,
                "group_id": group_id,
                "target_object_id": 1,
                "candidate_bundle_path": str(candidate),
                "candidate_bundle_sha256": sha256_file(candidate),
                "candidate_pool_fingerprint": canonical_sha256(
                    [condition, group_id, "pool"]
                ),
                "grounding_condition": bundle_condition,
                "candidate_count": count,
                "candidate_ids": [candidate_id] if count else [],
                "labels": labels,
                "official_source_hashes": (
                    {str(official): sha256_file(official)} if count else {}
                ),
                "label_generation_status": (
                    "completed_official_low_level_evaluation"
                    if count
                    else "skipped_empty_pool"
                ),
                "evaluator_calls_for_group": 1 if count else 0,
            }
            if not count:
                label_payload["empty_pool_reason"] = "vgn_no_candidates"
            label_payload["bundle_fingerprint"] = canonical_sha256(label_payload)
            label_path = _json(stem.with_suffix(".labels.json"), label_payload)
            label_paths[f"{condition}/{group_id}"] = label_path

            feature_frame = pd.DataFrame(
                (
                    [
                        {
                            "group_id": group_id,
                            "candidate_id": candidate_id,
                            "scene_id": scene_id,
                            "split": split,
                            "condition": condition,
                            "native_rank": 1,
                            "native_score": 0.5,
                        }
                    ]
                    if count
                    else []
                ),
                columns=(
                    "group_id",
                    "candidate_id",
                    "scene_id",
                    "split",
                    "condition",
                    "native_rank",
                    "native_score",
                ),
            )
            feature_path = stem.with_suffix(".features.parquet")
            feature_frame.to_parquet(feature_path, index=False)
            feature_payload = {
                "feature_file": feature_path.name,
                "feature_sha256": sha256_file(feature_path),
            }
            feature_sidecar = _json(stem.with_suffix(".features.json"), feature_payload)
            commits.append(
                {
                    "group_id": group_id,
                    "scene_id": scene_id,
                    "partition": split,
                    "candidate_count": count,
                    "candidate_pool_fingerprint": label_payload[
                        "candidate_pool_fingerprint"
                    ],
                    "candidate_bundle_path": str(candidate),
                    "candidate_bundle_sha256": sha256_file(candidate),
                    "generation_status": "completed_vgn_inference",
                    "feature_commit_path": str(feature_sidecar),
                    "feature_commit_sha256": sha256_file(feature_sidecar),
                    "feature_table_sha256": sha256_file(feature_path),
                    "label_bundle_path": str(label_path),
                    "label_bundle_sha256": sha256_file(label_path),
                }
            )
        provenance = _json(
            assembly_dir / "assembly_provenance.json",
            {
                "status": "COMPLETE",
                "scope": FORMAL_SCOPE,
                "fixture_only": False,
                "condition": condition,
                "run_id": run_dir.name,
                "group_commits": commits,
            },
        )
        assembly = AnalysisInputAssembly(
            manifest_path=input_manifest,
            output_dir=assembly_dir,
            assembly_fingerprint=canonical_sha256(condition),
            manifest_sha256=sha256_file(input_manifest),
            condition=condition,
            conditions=(condition,),
            group_count=2,
            candidate_count=1,
            empty_group_count=1,
            partition_rows={},
            partition_group_universes={},
            resumed=True,
        )
        assemblies[condition] = assembly
        loaded_by_path[input_manifest.resolve()] = SimpleNamespace(
            status="COMPLETE",
            scope=FORMAL_SCOPE,
            fixture_only=False,
            run_id=run_dir.name,
            provenance={"assembly_provenance_sha256": sha256_file(provenance)},
            feature_schema=SimpleNamespace(path=schema, sha256=sha256_file(schema)),
        )
        _json(
            run_dir / "analysis" / condition / "analysis_manifest.json",
            {
                "schema_version": ANALYSIS_SCHEMA,
                "status": "COMPLETE",
                "analysis_scope": FORMAL_SCOPE,
                "fixture_only": False,
                "formal_report_eligible": True,
                "run_id": run_dir.name,
                "input_manifest_path": str(input_manifest),
                "input_manifest_sha256": sha256_file(input_manifest),
            },
        )

    monkeypatch.setattr(
        run_outputs,
        "assemble_analysis_inputs",
        lambda *args, condition, **kwargs: assemblies[condition],
    )
    monkeypatch.setattr(
        run_outputs,
        "load_analysis_input_manifest",
        lambda path: loaded_by_path[Path(path).resolve()],
    )

    def load_features(
        sidecar: Path,
        *,
        expected_group_id: str,
        expected_condition: str,
        expected_candidate_ids: list[str],
        **_: Any,
    ) -> pd.DataFrame:
        payload = json.loads(Path(sidecar).read_text())
        frame = pd.read_parquet(Path(sidecar).parent / payload["feature_file"])
        assert frame["group_id"].astype(str).tolist() == [expected_group_id] * len(
            frame
        )
        assert frame["condition"].astype(str).tolist() == [expected_condition] * len(
            frame
        )
        assert frame["candidate_id"].astype(str).tolist() == expected_candidate_ids
        return frame

    monkeypatch.setattr(
        run_outputs, "load_committed_formal_feature_table", load_features
    )
    return assemblies, loaded_by_path, label_paths


def test_candidate_aggregates_preserve_zero_pool_rows_and_exact_feature_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _formal_run(tmp_path)
    _, _, label_paths = _candidate_fixture(run_dir, monkeypatch)
    result = publish_candidate_aggregates(run_dir)
    assert result.resumed is False
    labels = pd.read_parquet(run_dir / "candidate_labels.parquet")
    features = pd.read_parquet(run_dir / "candidate_features.parquet")
    universe = pd.read_parquet(run_dir / "candidate_group_universe.parquet")
    assert len(labels) == 6
    assert labels["record_kind"].eq("candidate_label").sum() == 3
    empty = labels.loc[labels["record_kind"].eq("empty_pool_group")]
    assert len(empty) == 3
    assert empty["candidate_id"].isna().all()
    assert empty["empty_pool_reason"].eq("vgn_no_candidates").all()
    assert len(features) == 3
    assert len(universe) == 6
    assert (run_dir / "feature_schema.json").read_bytes() == (
        run_dir / "assembled-feature-schema.json"
    ).read_bytes()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["zero_pool_group_rows"] == 3
    assert set(manifest["conditions"]) == set(CONDITIONS)

    resumed = publish_candidate_aggregates(run_dir, resume=True)
    assert resumed.resumed is True
    stale = label_paths["oracle_gt_mask/group-real"]
    stale.write_text(stale.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RunOutputError, match="label bundle is stale"):
        publish_candidate_aggregates(run_dir, resume=True)


def test_candidate_aggregate_rejects_cross_condition_label_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _formal_run(tmp_path)
    _candidate_fixture(run_dir, monkeypatch, wrong_condition="hifics_zero_shot_mask")
    with pytest.raises(RunOutputError, match="disagrees with assembly"):
        publish_candidate_aggregates(run_dir)
    assert not (run_dir / "candidate_aggregate_provenance.json").exists()


def _selected_fixture(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected: str = "hifics_adapted_mask",
) -> Any:
    geometry = "a" * 64
    native = pd.DataFrame(
        [
            {
                "grounding_condition": selected,
                "group_id": "test-group",
                "candidate_id": "candidate-1",
                "geometry_sha256": geometry,
                "native_rank": 1,
                "native_score": 0.5,
            }
        ]
    )
    reranked = native.assign(raw_rerank_score=0.75)
    selected_root = run_dir / "analysis" / selected
    selected_root.mkdir(parents=True)
    native.to_csv(selected_root / "native_predictions.csv", index=False)
    reranked.to_csv(selected_root / "reranked_predictions.csv", index=False)
    text_sources = {
        "paired_outcomes.csv": "analysis_scope,group_id\nformal_real_data,test-group\n",
        "metrics.csv": "analysis_scope,system\nformal_real_data,B0_NATIVE\n",
        "metrics.json": '{"analysis_scope":"formal_real_data"}\n',
        "bootstrap_results.csv": "analysis_scope,iterations\nformal_real_data,10000\n",
        "significance_tests.json": '{"analysis_scope":"formal_real_data"}\n',
        "ablation_results.csv": "analysis_scope,ablation\nformal_real_data,A0\n",
        "failure_taxonomy.csv": "analysis_scope,group_id,category\nformal_real_data,test-group,S0_UNCHANGED_SUCCESS\n",
        "failure_summary.json": '{"analysis_scope":"formal_real_data"}\n',
        "frozen_pool_audit.json": '{"analysis_scope":"formal_real_data","status":"PASS"}\n',
    }
    for name, text in text_sources.items():
        (selected_root / name).write_text(text, encoding="utf-8")

    input_path = _json(selected_root / "input_manifest.json", {"formal": True})
    feature_schema = selected_root / "feature_schema.json"
    feature_schema.write_text("{}\n", encoding="utf-8")
    partition_objects = {}
    for partition in ("train", "validation", "test"):
        rows_path = selected_root / f"{partition}_rows.parquet"
        group_path = selected_root / f"{partition}_groups.parquet"
        native.to_parquet(rows_path, index=False)
        pd.DataFrame(
            [
                {
                    "partition": partition,
                    "scene_id": "scene-1",
                    "group_id": "test-group",
                    "grounding_condition": selected,
                }
            ]
        ).to_parquet(group_path, index=False)
        partition_objects[partition] = SimpleNamespace(
            rows=SimpleNamespace(path=rows_path, sha256=sha256_file(rows_path)),
            group_universe=SimpleNamespace(
                path=group_path, sha256=sha256_file(group_path)
            ),
        )
    loaded_input = SimpleNamespace(
        source_path=input_path,
        manifest_sha256=sha256_file(input_path),
        status="COMPLETE",
        scope=FORMAL_SCOPE,
        fixture_only=False,
        run_id=run_dir.name,
        partitions=partition_objects,
        feature_schema=SimpleNamespace(
            path=feature_schema, sha256=sha256_file(feature_schema)
        ),
    )
    monkeypatch.setattr(
        run_outputs, "load_analysis_input_manifest", lambda _: loaded_input
    )
    outputs = {
        path.name: sha256_file(path)
        for path in selected_root.iterdir()
        if path.name
        in {
            "native_predictions.csv",
            "reranked_predictions.csv",
            *text_sources,
        }
    }
    manifest = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "COMPLETE",
        "analysis_scope": FORMAL_SCOPE,
        "fixture_only": False,
        "run_id": run_dir.name,
        "analysis_fingerprint": canonical_sha256("selected-analysis"),
        "input_manifest_path": str(input_path),
        "input_manifest_sha256": sha256_file(input_path),
        "outputs": outputs,
    }
    _json(selected_root / "analysis_manifest.json", manifest)
    selected_snapshot = SimpleNamespace(
        root=selected_root,
        manifest=manifest,
        native=native,
        reranked=reranked,
        failures=pd.DataFrame([{"group_id": "test-group"}]),
    )
    analyses = {}
    for condition in (*CONDITIONS, "combined_a8"):
        if condition == selected:
            analyses[condition] = selected_snapshot
            continue
        root = run_dir / "analysis" / condition
        root.mkdir(parents=True)
        _json(root / "analysis_manifest.json", {"condition": condition})
        analyses[condition] = SimpleNamespace(root=root)
    return SimpleNamespace(
        run_dir=run_dir,
        selected_condition=selected,
        analyses=analyses,
    )


def test_selected_analysis_projection_is_predicted_only_and_hash_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _formal_run(tmp_path)
    validated = _selected_fixture(run_dir, monkeypatch)
    monkeypatch.setattr(
        run_outputs, "validate_formal_paper_inputs", lambda _: validated
    )
    result = publish_selected_analysis_outputs(run_dir)
    assert result.resumed is False
    native = pd.read_parquet(run_dir / "native_predictions.parquet")
    reranked = pd.read_parquet(run_dir / "reranked_predictions.parquet")
    assert set(native["grounding_condition"]) == {"hifics_adapted_mask"}
    assert set(reranked["grounding_condition"]) == {"hifics_adapted_mask"}
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["oracle_is_separate_counterfactual"] is True
    assert (
        manifest["analysis_arms"]["oracle_gt_mask"]["role"]
        == "oracle_grounding_counterfactual"
    )
    assert manifest["analysis_arms"]["oracle_gt_mask"]["copied_to_top_level"] is False

    resumed = publish_selected_analysis_outputs(run_dir, resume=True)
    assert resumed.resumed is True
    metrics = run_dir / "analysis" / "hifics_adapted_mask" / "metrics.csv"
    metrics.write_text(metrics.read_text() + "extra,row\n", encoding="utf-8")
    with pytest.raises(RunOutputError, match="selected analysis output is stale"):
        publish_selected_analysis_outputs(run_dir, resume=True)


def test_selected_analysis_rejects_oracle_as_primary_and_blocked_formal_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_run = _formal_run(tmp_path, name="formal-oracle-primary-run")
    oracle = _selected_fixture(oracle_run, monkeypatch, selected="oracle_gt_mask")
    monkeypatch.setattr(run_outputs, "validate_formal_paper_inputs", lambda _: oracle)
    with pytest.raises(RunOutputError, match="validation-selected predicted"):
        publish_selected_analysis_outputs(oracle_run)

    blocked = _formal_run(
        tmp_path, name="formal-blocked-analysis-run", status="BLOCKED"
    )
    with pytest.raises(RunOutputError, match="cannot publish formal outputs"):
        publish_selected_analysis_outputs(blocked)
