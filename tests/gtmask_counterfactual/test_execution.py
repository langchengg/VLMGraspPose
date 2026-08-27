from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import gtmask_counterfactual.protocol as protocol_module

from d1_reranking.resource_gate import host_contract, resource_thresholds
from gtmask_counterfactual.candidate_matching import (
    match_candidate_pools,
    periodic_angle_difference,
    stable_candidate_id,
)
from gtmask_counterfactual.execution import (
    ExecutionContractError,
    adapt_native_cumulative_to_sample_shards,
    atomic_sample_json,
    authorize_gt_candidate_generation,
    d1_case_b_preflight,
    probe_frozen_docker_image,
)
from gtmask_counterfactual.mapping import (
    GTMappingError,
    build_prelock_registry,
    join_real_authority_rows,
    mapping_pixel_qa,
)
from gtmask_counterfactual.io import artifact_record as canonical_artifact_record
from gtmask_counterfactual.audit import (
    bootstrap_run,
    initialize_counterfactual_ledger,
    transition_pipeline_status,
    verify_source_final_lock,
)
from gtmask_counterfactual.contracts import RunState
from gtmask_counterfactual.io import artifact_record
from gtmask_counterfactual.protocol import (
    claim_bulk_execution,
    create_protocol_lock,
    inline_binding,
)
from unified_reranking.hashing import canonical_sha256, sha256_file


def _authority(
    sample_id: str, scene_id: str, query_id: int, target: int
) -> dict[str, object]:
    digest = "a" * 64
    root = Path("/synthetic") / sample_id
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "question_index": query_id,
        "target_object_id": target,
        "prepared_gt_mask_path": str(root / "prepared_352.png"),
        "prepared_gt_mask_sha256": digest,
        "source_instance_mask_path": str(root / "instance_480x640.png"),
        "source_instance_mask_sha256": digest,
    }


def _candidate(candidate_id: str, *, x: float, angle: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "cx_px": x,
        "cy_px": 30.0,
        "theta_deg": angle,
        "width_px": 40.0,
        "height_px": 20.0,
    }


def _core_protocol(tmp_path: Path, *, claim: bool) -> tuple[Path, Path]:
    run = tmp_path / "run"
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_artifact = source_root / "artifact.bin"
    source_artifact.write_bytes(b"source")
    inventory = [
        {
            "relative_path": "artifact.bin",
            **artifact_record(source_artifact),
        }
    ]
    source_lock = {
        "status": "COMPLETE",
        "inventory": inventory,
        "inventory_count": 1,
        "inventory_sha256": canonical_sha256(inventory),
    }
    source_lock["self_sha256"] = canonical_sha256(source_lock)
    (source_root / "FINAL_RUN_LOCK.json").write_text(
        json.dumps(source_lock), encoding="utf-8"
    )
    (source_root / "manifest.json").write_text(
        json.dumps({"status": "COMPLETE", "formal_test_execution_count": 1}),
        encoding="utf-8",
    )
    bootstrap_run(
        run,
        source_verification={
            "schema_version": 1,
            "status": "PASS",
            "full_inventory_byte_rehash": True,
            "sources": {
                "synthetic": verify_source_final_lock(
                    source_root,
                    expected_file_sha256=sha256_file(
                        source_root / "FINAL_RUN_LOCK.json"
                    ),
                    full_inventory_rehash=True,
                )
            },
        },
    )
    transition_pipeline_status(
        run,
        RunState.P1_BASELINE_REPLAY_PASS,
        first_incomplete_stage=RunState.P2_GT_MAPPING_PASS.value,
    )
    transition_pipeline_status(
        run,
        RunState.P2_GT_MAPPING_PASS,
        first_incomplete_stage=RunState.P3_PROTOCOL_LOCKED.value,
    )
    source = tmp_path / "bound.py"
    source.write_text("VALUE=1\n", encoding="utf-8")
    sample_manifest = tmp_path / "samples.parquet"
    sample_manifest.write_bytes(b"samples")
    registry = tmp_path / "registry.parquet"
    registry.write_bytes(b"registry")
    mapping_qa = tmp_path / "mapping_qa.json"
    mapping_qa.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stage": "P2_GT_MAPPING_PASS",
                "pixel_qa_status": "P2_MAPPING_QA_PASS",
                "mapping_qa_gt_mask_rows_read": 1,
                "candidate_generation_gt_mask_rows_read": 0,
            }
        ),
        encoding="utf-8",
    )
    routes = {
        "g1": {"allowed_gt_branches": ["gt_oracle"]},
        "c1": {"allowed_gt_branches": ["gt_oracle"]},
        "d1": {
            "allowed_gt_branches": ["gt_oracle"],
            "case": "B",
            "mask_affects_raw_sampling": True,
            "raw_candidate_regeneration_required": True,
            "filter_only_primary_allowed": False,
        },
    }
    record = artifact_record(source)
    bindings = {
        "source_locks": {"synthetic": record},
        "source_code": {"synthetic": record},
        "configs": {"synthetic": record},
        "baseline_replay": record,
        "sample_manifest": artifact_record(sample_manifest),
        "gt_grasp_source": artifact_record(sample_manifest),
        "gt_mask_registry": artifact_record(registry),
        "mapping_qa": artifact_record(mapping_qa),
        "route_contracts": inline_binding(routes),
        "resize_rules": inline_binding({"binary": "PIL nearest"}),
        "evaluator": record,
        "taxonomy": inline_binding({"classes": ["T0"]}),
        "statistics": inline_binding({"seed": 20260813}),
        "case_selection": inline_binding({"rule": "synthetic"}),
    }
    declaration = {
        "gt_candidate_generation_authorized": True,
        "bulk_execution_max_count": 1,
        "mapping_qa_gt_mask_rows_read_before_lock": 1,
        "candidate_generation_gt_mask_rows_read_before_lock": 0,
        "routes": routes,
    }
    lock = create_protocol_lock(
        run,
        bindings=bindings,
        declaration=declaration,
        test_only_allow_synthetic_contract=True,
    )
    if claim:
        claim_bulk_execution(run)
    return lock, registry


def test_retrospective_d1_claim_is_scoped_and_preserves_p10(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    lock_path = run / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json"
    lock_path.parent.mkdir(parents=True)
    lock = {
        "execution_mode": "retrospective_verified_import",
        "d1_candidate_generation_authorized": True,
        "self_sha256": "synthetic-self-hash",
    }
    lock_path.write_text(json.dumps(lock) + "\n", encoding="utf-8")
    pipeline_path = run / "pipeline_status.json"
    manifest_path = run / "manifest.json"
    pipeline_path.write_text(
        json.dumps(
            {
                "status": RunState.P5B_G1_FULL_COMPLETE.value,
                "counterfactual_execution_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps({"counterfactual_execution_count": 0}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(protocol_module, "verify_protocol_lock", lambda root: lock)
    with pytest.raises(PermissionError, match="completed P10 core"):
        claim_bulk_execution(run)

    pipeline_path.write_text(
        json.dumps(
            {
                "status": RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value,
                "counterfactual_execution_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    claim_path = claim_bulk_execution(run)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert claim["scope"] == "d1_secondary"
    assert claim["execution_count"] == 1
    assert pipeline["status"] == RunState.P10_INDEPENDENT_RECOMPUTE_PASS.value
    assert pipeline["counterfactual_execution_count"] == 1


def _gate() -> dict[str, object]:
    finished = datetime.now(timezone.utc)
    start = finished - timedelta(minutes=15)
    windows = []
    for window_index in range(3):
        observations = []
        for sample_index in range(21):
            offset = window_index * 300 + sample_index * 15
            observations.append(
                {
                    "captured_at_utc": (start + timedelta(seconds=offset)).isoformat(),
                    "monotonic_offset_seconds": float(offset),
                    "memory_free_percent": 60.0,
                    "swap_used_bytes": 0,
                    "disk_free_bytes": 200 * 1024**3,
                    "normalized_load_5m": 0.1,
                    "rank1_workers": [],
                    "rank1_claim_paths": [],
                    "d1_heavy_workers": [],
                    "foreign_heavy_processes": [],
                }
            )
        windows.append(
            {
                "index": window_index,
                "monotonic_start_seconds": float(window_index * 300),
                "monotonic_end_seconds": float((window_index + 1) * 300),
                "observations": observations,
            }
        )
    gate = {
        "status": "PASS",
        "gate_type": "gtmask_d1_three_continuous_five_minute_windows_v1",
        "thresholds": resource_thresholds(),
        "host": host_contract(),
        "finished_at_utc": finished.isoformat(),
        "windows": windows,
    }
    gate["content_sha256"] = canonical_sha256(gate)
    return gate


def test_prelock_registry_is_path_hash_only_and_keeps_unresolved() -> None:
    samples = [
        {"sample_id": "s1", "scene_id": "scene-a", "question_index": 1},
        {"sample_id": "s2", "scene_id": "scene-b", "question_index": 2},
    ]
    rows = build_prelock_registry(
        samples,
        [_authority("s1", "scene-a", 1, 7)],
        expected_count=2,
    )
    assert rows[0]["mapping_status"] == "PATH_HASH_INSTANCE_AUTHORITY_MAPPED"
    assert rows[0]["prepared_height"] == 352
    assert rows[0]["original_width"] == 640
    assert rows[0]["bulk_gt_pixels_read"] is False
    assert rows[1]["mapping_status"] == "gt_oracle_unavailable"


def test_prelock_registry_rejects_identity_mismatch() -> None:
    samples = [{"sample_id": "s1", "scene_id": "scene-a", "question_index": 1}]
    with pytest.raises(GTMappingError, match="scene identity differs"):
        build_prelock_registry(
            samples,
            [_authority("s1", "wrong-scene", 1, 7)],
            expected_count=1,
        )


def test_real_two_manifest_schema_uses_instance_map_authority() -> None:
    digest = "b" * 64
    prepared = [
        {
            "sample_id": "s1",
            "scene_id": "scene-a",
            "question_index": 4,
            "target_object_id": 9,
            "prepared_gt_mask_path": "/prepared/s1.png",
            "prepared_gt_mask_sha256": digest,
        }
    ]
    visual = [
        {
            "sample_id": "s1",
            "scene_id": "scene-a",
            "question_index": 4,
            "target_instance_id": 9,
            "gt_grasp_target_instance_id": 9,
            "gt_grasp_set_sha256": "c" * 64,
            "rgb_height": 480,
            "rgb_width": 640,
            "gt_mask_path": "/dataset/instance-map.png",
            "gt_mask_sha256": digest,
        }
    ]
    authority = join_real_authority_rows(prepared, visual)
    assert authority[0]["source_instance_mask_path"].endswith("instance-map.png")
    assert "original_gt_mask_path" not in authority[0]
    registry = build_prelock_registry(
        [{"sample_id": "s1", "scene_id": "scene-a", "question_index": 4}],
        authority,
        expected_count=1,
    )
    assert registry[0]["original_gt_mask_path"] is None
    assert registry[0]["target_instance_id"] == 9


def test_locked_protocol_without_bulk_claim_rejects_generator(tmp_path: Path) -> None:
    protocol, registry = _core_protocol(tmp_path, claim=False)
    with pytest.raises(Exception, match="execution claim"):
        authorize_gt_candidate_generation(
            protocol_lock_path=protocol,
            registry_path=registry,
            route="d1",
            branch="gt_oracle",
        )


def test_p2_pixel_qa_requires_narrow_non_generation_purpose(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="wrong purpose"):
        mapping_pixel_qa(
            {"sample_id": "s1"},
            access_authority={
                "status": "AUTHORIZED",
                "stage": "P2_GT_MAPPING_PASS",
                "purpose": "candidate_generation",
                "candidate_generation_allowed": False,
            },
            derived_output_dir=tmp_path,
        )


def test_p2_pixel_qa_derives_target_binary_not_all_instances(tmp_path: Path) -> None:
    instance = np.zeros((480, 640), dtype=np.uint8)
    instance[20:80, 40:100] = 9
    instance[200:300, 300:450] = 3
    instance_path = tmp_path / "instances.png"
    Image.fromarray(instance).save(instance_path)
    target = instance == 9
    prepared = Image.fromarray(target.astype(np.uint8) * 255).resize(
        (352, 352), resample=Image.Resampling.NEAREST
    )
    prepared_path = tmp_path / "prepared.png"
    prepared.save(prepared_path)
    result = mapping_pixel_qa(
        {
            "sample_id": "s1",
            "mapping_status": "PATH_HASH_INSTANCE_AUTHORITY_MAPPED",
            "target_instance_id": 9,
            "gt_grasp_target_instance_id": 9,
            "gt_grasp_set_sha256": "c" * 64,
            "rgb_height": 480,
            "rgb_width": 640,
            "prepared_gt_mask_path": str(prepared_path),
            "prepared_gt_mask_sha256": sha256_file(prepared_path),
            "source_instance_mask_path": str(instance_path),
            "source_instance_mask_sha256": sha256_file(instance_path),
        },
        access_authority={
            "status": "AUTHORIZED",
            "stage": "P2_GT_MAPPING_PASS",
            "purpose": "mapping_and_annotation_pixel_qa_only",
            "candidate_generation_allowed": False,
        },
        derived_output_dir=tmp_path / "derived",
    )
    derived = np.asarray(Image.open(result["original_gt_mask_path"])) != 0
    assert np.array_equal(derived, target)
    assert not derived[250, 350]
    assert result["foreground_pixels"] == int(target.sum())


def test_gt_candidate_generation_is_rejected_before_protocol_lock(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    registry = tmp_path / "registry.parquet"
    registry.write_bytes(b"path-hash-registry")
    protocol = run / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json"
    with pytest.raises(PermissionError, match="before P4 lock"):
        authorize_gt_candidate_generation(
            protocol_lock_path=protocol,
            registry_path=registry,
            route="d1",
            branch="gt_oracle",
        )


def test_candidate_ids_are_branch_local_and_matching_is_geometry_only() -> None:
    geometry = _candidate("unused", x=20.0, angle=89.0)
    predicted_id = stable_candidate_id(
        sample_id="s1",
        route="d1",
        branch="predicted",
        source_candidate_index=0,
        candidate=geometry,
    )
    gt_id = stable_candidate_id(
        sample_id="s1",
        route="d1",
        branch="gt_oracle",
        source_candidate_index=0,
        candidate=geometry,
    )
    assert predicted_id != gt_id
    assert periodic_angle_difference(89.0, -89.0) == pytest.approx(2.0)
    matches = match_candidate_pools(
        [_candidate(predicted_id, x=20.0, angle=89.0)],
        [_candidate(gt_id, x=22.0, angle=-89.0)],
    )
    assert matches[0]["status"] == "matched_pred_gt_candidate"
    assert matches[0]["match_basis"] == "center_width"


def test_atomic_sample_shard_is_immutable_on_resume(tmp_path: Path) -> None:
    first = atomic_sample_json(
        run_dir=tmp_path,
        route="g1",
        branch="gt_oracle",
        sample_id="sample/unsafe-name",
        payload={"candidate_count": 2},
    )
    second = atomic_sample_json(
        run_dir=tmp_path,
        route="g1",
        branch="gt_oracle",
        sample_id="sample/unsafe-name",
        payload={"candidate_count": 2},
    )
    assert first == second
    assert first.name.endswith(".json")
    assert "unsafe-name" not in first.name
    with pytest.raises(ExecutionContractError, match="differs"):
        atomic_sample_json(
            run_dir=tmp_path,
            route="g1",
            branch="gt_oracle",
            sample_id="sample/unsafe-name",
            payload={"candidate_count": 3},
        )


def test_native_adapter_executes_pass_subset_and_adds_technical_complement(
    tmp_path: Path,
) -> None:
    native = tmp_path / "native"
    native.mkdir()
    pd.DataFrame(
        [
            {"sample_id": "s-pass", "status": "success", "candidate_count": 1},
            {"sample_id": "s-outside", "status": "success", "candidate_count": 1},
        ]
    ).to_parquet(native / "per_sample.parquet", index=False)
    pd.DataFrame(
        [
            {
                "sample_id": "s-pass",
                "candidate_id": "native-0",
                "native_rank": 1,
                "cx_px": 20.0,
                "cy_px": 30.0,
                "theta_deg": 10.0,
                "width_px": 40.0,
                "height_px": 20.0,
            },
            {
                "sample_id": "s-outside",
                "candidate_id": "native-1",
                "native_rank": 1,
                "cx_px": 50.0,
                "cy_px": 60.0,
                "theta_deg": 20.0,
                "width_px": 30.0,
                "height_px": 15.0,
            },
        ]
    ).to_parquet(native / "candidates.parquet", index=False)
    (native / "run_manifest.json").write_text(
        json.dumps({"status": "COMPLETE"}) + "\n", encoding="utf-8"
    )
    source_adapter = tmp_path / "ADAPTER_MANIFEST.json"
    source_adapter.write_text("{}\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    initialize_counterfactual_ledger(run / "run_ledger.sqlite")
    manifest_path = adapt_native_cumulative_to_sample_shards(
        run_dir=run,
        route="g1",
        branch="gt_oracle",
        native_output=native,
        expected_executed_sample_ids={"s-pass"},
        technical_complement_ids={"s-unresolved"},
        source_adapter_manifest=source_adapter,
        allow_source_superset=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = pd.read_parquet(manifest["per_sample"]["path"])
    candidates = pd.read_parquet(manifest["candidates"]["path"])
    assert manifest["per_sample"] == canonical_artifact_record(
        run / "06_gtmask_predictions/g1/gt_oracle/per_sample.parquet"
    )
    assert manifest["per_candidate"] == canonical_artifact_record(
        run / "06_gtmask_predictions/g1/gt_oracle/per_candidate.parquet"
    )
    assert manifest["candidates"] == manifest["per_candidate"]
    assert manifest["sample_count"] == 2
    assert manifest["executed_sample_count"] == 1
    assert manifest["technical_complement_count"] == 1
    assert manifest["source_sample_count"] == 2
    assert manifest["source_subset_import"] is True
    assert set(samples["sample_id"]) == {"s-pass", "s-unresolved"}
    unresolved = samples.loc[samples["sample_id"].eq("s-unresolved")].iloc[0]
    assert unresolved["technical_failure"]
    assert unresolved["no_output"]
    assert unresolved["candidate_count"] == 0
    assert set(candidates["sample_id"]) == {"s-pass"}
    assert candidates["route"].tolist() == ["G1"]
    assert candidates["branch"].tolist() == ["gt_oracle"]


def test_docker_daemon_failure_is_read_only_blocker() -> None:
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")

    result = probe_frozen_docker_image(command_runner=fake_runner)
    assert result["status"] == "BLOCKED"
    assert result["execution_attempted"] is False
    assert len(calls) == 1
    assert calls[0][1:3] == ["image", "inspect"]


def test_d1_case_b_writes_machine_blocker_and_never_executes(tmp_path: Path) -> None:
    protocol, registry = _core_protocol(tmp_path, claim=True)
    calls: list[list[str]] = []

    def daemon_missing(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")

    result = d1_case_b_preflight(
        run_dir=tmp_path / "run",
        registry_path=registry,
        protocol_lock_path=protocol,
        resource_gate=_gate(),
        command_runner=daemon_missing,
    )
    assert result["status"] == "MACHINE_BLOCKED"
    assert result["execution_attempted"] is False
    assert result["filter_only_primary_allowed"] is False
    assert Path(str(result["artifact_path"])).is_file()
    assert len(calls) == 1
