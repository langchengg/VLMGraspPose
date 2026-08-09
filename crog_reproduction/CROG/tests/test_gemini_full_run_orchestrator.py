from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from failure_analysis.gemini_crog_evidence_v1 import MODEL_IDS
from failure_analysis.gemini_crog_evidence_v1.full_run import (
    DatasetContract,
    FullRunConfig,
    FullRunDependencies,
    FullRunOrchestrator,
    HardStopError,
    P1_PROTOCOL,
    REQUEST_PHASES,
)
from failure_analysis.gemini_crog_evidence_v1.planner import with_content_digest


def _candidate(index: int) -> dict:
    x = 10.0 + index * 4.0
    return {
        "candidate_id": f"candidate_{index}",
        "candidate_checksum": f"checksum-{index}",
        "cx": x,
        "cy": 12.0,
        "row": 12.0,
        "col": x,
        "angle_deg": 0.0,
        "width_px": 4.0,
        "height_px": 2.0,
        "polygon": [[x - 2, 11], [x + 2, 11], [x + 2, 13], [x - 2, 13]],
        "q_raw": 1.0 - index / 10.0,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Rank only the supplied frozen candidates.", encoding="utf-8")
    config_path = tmp_path / "crog.yaml"
    config_path.write_text("model: frozen\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    split_manifest = tmp_path / "split.json"
    _write_json(split_manifest, {"rows": []})
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image_path = tmp_path / "rgb.png"
    assert cv2.imwrite(str(image_path), image)

    datasets = {}
    partition_splits = {"train": "train", "calibration": "train", "validation": "val", "test": "test"}
    for partition, split in partition_splits.items():
        folder = tmp_path / partition
        folder.mkdir()
        features = folder / "features.jsonl"
        stable_id = f"multiple:{split}:00000000"
        feature = {
            "split": split,
            "sample_id": 0,
            "scene_id": f"scene-{partition}",
            "image_path": str(image_path),
            "language_instruction": "the object",
            "candidates": [_candidate(index) for index in range(5)],
        }
        features.write_text(json.dumps(feature) + "\n", encoding="utf-8")
        label = {
            "sample_id": stable_id,
            "scene_id": f"scene-{partition}",
            "candidate_labels": [
                {"candidate_id": f"candidate_{index}", "candidate_correct": index == 0}
                for index in range(5)
            ],
        }
        legacy = folder / "legacy.jsonl"
        corrected = folder / "corrected.jsonl"
        legacy.write_text(json.dumps(label) + "\n", encoding="utf-8")
        corrected.write_text(json.dumps(label) + "\n", encoding="utf-8")
        datasets[partition] = DatasetContract(features, legacy, corrected, split)

    manifests = {}
    phase_partition = {
        "pilot": "train",
        "stability": "train",
        "ablation": "train",
        "calibration": "calibration",
        "validation": "validation",
        "formal_test": "test",
    }
    for phase, partition in phase_partition.items():
        split = partition_splits[partition]
        row = {
            "sample_id": f"multiple:{split}:00000000",
            "source_sample_id": 0,
            "frame_id": f"scene-{partition}",
            "scene_id": f"scene-{partition}",
            "group_id": f"group-{partition}",
            "official_split": split,
            "development_partition": partition,
            "evaluation_only": {"query_type": "name", "strata": []},
        }
        manifest = with_content_digest(
            {
                "schema_version": "1.0",
                "kind": "gemini_crog_cohort_manifest",
                "cohort": phase,
                "partition": partition,
                "sample_count": 1,
                "scene_distribution": {f"scene-{partition}": 1},
                "group_distribution": {f"group-{partition}": 1},
                "query_type_distribution": {"name": 1},
                "rows": [row],
            }
        )
        manifests[phase] = manifest
        _write_json(run_root / f"{phase}_manifest.json", manifest)
    slots = {"pilot": 2, "stability": 6, "ablation": 8, "calibration": 2, "validation": 2, "formal_test": 2}
    plan = with_content_digest(
        {
            "schema_version": "1.0",
            "kind": "gemini_crog_full_run_plan",
            "status": "ready",
            "blockers": [],
            "model_ids": list(MODEL_IDS),
            "phase_order": list(REQUEST_PHASES),
            "assumptions": {"max_spend_usd": 1000.0, "er2_cost_cap_per_request_usd": 1.0},
            "totals": {"planned_request_slots": sum(slots.values())},
            "phases": [
                {
                    "phase": phase,
                    "sample_count": 1,
                    "sample_manifest_sha256": manifests[phase]["content_sha256"],
                    "planned_request_slots": slots[phase],
                }
                for phase in REQUEST_PHASES
            ],
        }
    )
    _write_json(run_root / "full_run_plan.json", plan)
    config = FullRunConfig(
        run_root=run_root,
        system_prompt_path=prompt,
        crog_config_path=config_path,
        checkpoint_path=checkpoint,
        v2_root=tmp_path,
        split_manifest_path=split_manifest,
        datasets=datasets,
        evidence_shard_size=1,
        bootstrap_draws=3,
        verify_environment=False,
        verify_repository_contract=False,
        enforce_canonical_counts=False,
    )
    return config


def _fake_export(**kwargs):
    output = Path(kwargs["output_dir"])
    output.mkdir(parents=True)
    features = [json.loads(line) for line in Path(kwargs["frozen_features_path"]).read_text().splitlines() if line]
    by_id = {int(row["sample_id"]): row for row in features}
    requests, samples, evidence_rows = [], [], []
    mapping = {
        "display_to_candidate": {chr(65 + index): f"candidate_{index}" for index in range(5)},
        "candidate_to_display": {f"candidate_{index}": chr(65 + index) for index in range(5)},
    }
    boards = output / "boards"
    boards.mkdir()
    maps = output / "maps"
    maps.mkdir()
    for local_id in kwargs["selected_local_ids"]:
        feature = by_id[int(local_id)]
        split = "val" if feature["split"] == "val" else feature["split"]
        sample_id = f"multiple:{split}:{int(local_id):08d}"
        board = boards / f"sample_{int(local_id):08d}.png"
        board.write_bytes(b"fake-png-board")
        board_sha = hashlib.sha256(board.read_bytes()).hexdigest()
        _write_json(
            board.with_suffix(".layers.json"),
            {
                "sample_id": sample_id,
                "image_sha256": board_sha,
                "evaluation_overlay_included": False,
                "candidate_mapping": mapping,
                "panels": [
                    "rgb",
                    "predicted_m",
                    "predicted_q",
                    "predicted_angle",
                    "predicted_width",
                    "candidate_cards",
                ],
            },
        )
        requests.append(
            {
                "sample_id": sample_id,
                "frame_id": feature["scene_id"],
                "board_path": str(board.resolve()),
                "board_sha256": board_sha,
                "metadata": "<referring_expression>the object</referring_expression>",
                "mapping": mapping,
                "system_prompt_sha256": "test",
                "renderer_version": "test",
                "store": False,
                "background": False,
                "stream": False,
            }
        )
        samples.append(
            {
                "sample_id": sample_id,
                "original_q_top1_candidate_id": "candidate_0",
                "maps_path": None,
            }
        )
        for index in range(5):
            candidate = feature["candidates"][index]
            evidence_rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": f"candidate_{index}",
                    "candidate_checksum": candidate["candidate_checksum"],
                    "stable_candidate_id": f"{sample_id}/candidate_{index}",
                    "display_candidate_id": chr(65 + index),
                    "original_q_rank": index,
                    "center_x_px": candidate["cx"],
                    "center_y_px": candidate["cy"],
                    "angle_deg_periodic_180": candidate["angle_deg"],
                    "width_px": candidate["width_px"],
                    "fixed_height_px": candidate["height_px"],
                    "rectangle_corners": candidate["polygon"],
                    "q_original_value": candidate["q_raw"],
                }
            )
    (output / "request_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8"
    )
    pq.write_table(pa.Table.from_pylist(samples), output / "sample_evidence.parquet")
    pq.write_table(pa.Table.from_pylist(evidence_rows), output / "candidate_evidence.parquet")
    _write_json(output / "evidence_schema.json", {"schema_version": "test"})
    summary = {
        "status": "complete",
        "sample_count": len(samples),
        "candidate_count": len(evidence_rows),
        "checkpoint_sha256": hashlib.sha256(Path(kwargs["checkpoint_path"]).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(kwargs["config_path"]).read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(
            Path(kwargs["system_prompt_path"]).read_bytes()
        ).hexdigest(),
        "no_ground_truth_forward": True,
        "forward_identity_max_difference": 0.0,
    }
    _write_json(output / "export_summary.json", summary)
    return summary


def _materialize_ablation(**kwargs):
    return dict(kwargs["evidence"]["request"])


class FakeExecutor:
    def __init__(self, *, fail_on_call: int | None = None, auth_failure: bool = False):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.auth_failure = auth_failure

    def __call__(self, kwargs):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected process interruption")
        if self.auth_failure:
            return {
                "request_hash": f"request-{self.calls}",
                "status": "fallback",
                "lifecycle_status": "PERMANENT_FAILED",
                "valid": False,
                "abstain": False,
                "fallback_reason": "http_403",
                "cache_hit": False,
            }
        ranking = [
            {
                "candidate_id": chr(65 + index),
                "target_alignment_score": 1.0 - index / 10,
                "mask_support_score": 1.0 - index / 10,
                "quality_evidence_score": 1.0 - index / 10,
                "angle_consistency_score": 1.0 - index / 10,
                "width_consistency_score": 1.0 - index / 10,
                "edge_safety_score": 1.0 - index / 10,
                "overall_score": 1.0 - index / 10,
                "reason_codes": [],
            }
            for index in range(5)
        ]
        return {
            "request_hash": hashlib.sha256(json.dumps(kwargs, default=str, sort_keys=True).encode()).hexdigest(),
            "request_id": f"provider-{self.calls}",
            "status": "success",
            "lifecycle_status": "SUCCEEDED",
            "valid": True,
            "abstain": False,
            "cache_hit": False,
            "latency_seconds": 0.1,
            "usage": {"total_input_tokens": 10, "total_output_tokens": 5, "total_thought_tokens": 2},
            "estimated_charge_usd": 0.001,
            "response_model": kwargs["model_id"],
            "service_tier": "standard",
            "parsed_output": {
                "selected_candidate_id": "A",
                "ranking": ranking,
                "confidence": 0.9,
                "score_margin_top1_top2": 0.2,
                "decision": "keep_original",
                "global_reason_codes": [],
            },
        }


def test_phase_c_d_state_machine_resumes_without_duplicate_requests(tmp_path: Path):
    config = _fixture(tmp_path)
    interrupted = FakeExecutor(fail_on_call=4)
    dependencies = FullRunDependencies(
        export_evidence=_fake_export,
        request_executor=interrupted,
        materialize_ablation=_materialize_ablation,
    )
    with pytest.raises(RuntimeError, match="injected process interruption"):
        FullRunOrchestrator(config, dependencies).run(stop_after="ablation")
    persisted_before = len(list((config.run_root / "pilot" / "decisions").glob("*.json"))) + len(
        list((config.run_root / "stability" / "decisions").glob("*.json"))
    )
    resumed = FakeExecutor()
    result = FullRunOrchestrator(
        config,
        FullRunDependencies(
            export_evidence=_fake_export,
            request_executor=resumed,
            materialize_ablation=_materialize_ablation,
        ),
    ).run(stop_after="ablation")
    assert result["status"] == "stopped_after_ablation"
    assert (config.run_root / "pilot" / "PHASE_COMPLETE.json").is_file()
    assert (config.run_root / "stability" / "PHASE_COMPLETE.json").is_file()
    assert (config.run_root / "ablation" / "PHASE_COMPLETE.json").is_file()
    assert (config.run_root / "development_protocol_lock.json").is_file()
    assert resumed.calls == 16 - persisted_before
    assert len(list((config.run_root / "pilot" / "evidence").glob("shard_*"))) == 1


def test_credential_failure_is_a_hard_stop_before_second_request(tmp_path: Path):
    config = _fixture(tmp_path)
    executor = FakeExecutor(auth_failure=True)
    with pytest.raises(HardStopError, match="credentials_rejected") as captured:
        FullRunOrchestrator(
            config,
            FullRunDependencies(
                export_evidence=_fake_export,
                request_executor=executor,
                materialize_ablation=_materialize_ablation,
            ),
        ).run(stop_after="pilot")
    assert captured.value.phase_status == "blocked_credentials"
    assert executor.calls == 1
    status = json.loads((config.run_root / "phase_status.json").read_text())
    assert status["phases"]["pilot"]["status"] == "blocked_credentials"


def test_tampered_board_is_rejected_before_request_executor(tmp_path: Path):
    config = _fixture(tmp_path)

    def tampering_export(**kwargs):
        result = _fake_export(**kwargs)
        output = Path(kwargs["output_dir"])
        board = next((output / "boards").glob("*.png"))
        board.write_bytes(b"tampered-after-manifest")
        return result

    executor = FakeExecutor()
    with pytest.raises(HardStopError, match="board_sha256_changed"):
        FullRunOrchestrator(
            config,
            FullRunDependencies(
                export_evidence=tampering_export,
                request_executor=executor,
                materialize_ablation=_materialize_ablation,
            ),
        ).run(stop_after="pilot")
    assert executor.calls == 0


def test_manifest_content_hash_tampering_fails_before_export_or_request(tmp_path: Path):
    config = _fixture(tmp_path)
    path = config.run_root / "pilot_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rows"][0]["frame_id"] = "tampered"
    _write_json(path, manifest)
    executor = FakeExecutor()
    exports = []

    def exporter(**kwargs):
        exports.append(kwargs)
        return _fake_export(**kwargs)

    with pytest.raises(HardStopError, match="manifest_hash_mismatch"):
        FullRunOrchestrator(
            config,
            FullRunDependencies(
                export_evidence=exporter,
                request_executor=executor,
                materialize_ablation=_materialize_ablation,
            ),
        ).run(stop_after="pilot")
    assert not exports
    assert executor.calls == 0


def test_plain_p1_preserves_legacy_hash_dimensions_but_stability_is_isolated(tmp_path: Path):
    config = _fixture(tmp_path)
    orchestrator = FullRunOrchestrator(config, FullRunDependencies(request_executor=FakeExecutor()))
    orchestrator._prompt_hash = "p"
    orchestrator._schema_hash = "s"
    orchestrator._renderer_hash = "r"
    orchestrator._evidence_schema_hash = "e"
    request = {
        "sample_id": "multiple:train:00000000",
        "frame_id": "frame",
        "board_path": str(tmp_path / "board.png"),
        "metadata": "safe",
        "mapping": {
            "display_to_candidate": {chr(65 + i): f"candidate_{i}" for i in range(5)},
            "candidate_to_display": {f"candidate_{i}": chr(65 + i) for i in range(5)},
        },
    }
    normal = orchestrator._request_kwargs(
        request=request, model_id=MODEL_IDS[0], protocol=P1_PROTOCOL, replicate_id=0
    )
    stability = orchestrator._request_kwargs(
        request=request, model_id=MODEL_IDS[0], protocol=P1_PROTOCOL, replicate_id=1
    )
    assert normal["protocol_id"] is normal["replicate_id"] is normal["namespace"] is None
    assert stability["protocol_id"] == P1_PROTOCOL
    assert stability["replicate_id"] == 1
    assert stability["namespace"] == "stability"


def test_rotation_required_marker_blocks_before_env_export_or_request(tmp_path: Path):
    config = _fixture(tmp_path)
    (config.run_root / ".API_KEY_ROTATION_REQUIRED").write_text("blocked\n", encoding="utf-8")
    executor = FakeExecutor()
    exports = []

    def exporter(**kwargs):
        exports.append(kwargs)
        return _fake_export(**kwargs)

    with pytest.raises(HardStopError, match="api_key_rotation_required"):
        FullRunOrchestrator(
            config,
            FullRunDependencies(
                export_evidence=exporter,
                request_executor=executor,
                materialize_ablation=_materialize_ablation,
            ),
        ).run(stop_after="pilot")
    assert not exports
    assert executor.calls == 0


def test_formal_board_cleanup_runs_only_after_gallery_and_stays_in_formal_tree(tmp_path: Path):
    config = _fixture(tmp_path)
    config.delete_committed_boards = True
    orchestrator = FullRunOrchestrator(config)
    orchestrator.manifests = {
        "formal_test": json.loads((config.run_root / "formal_test_manifest.json").read_text())
    }
    phase_root = config.run_root / "formal_test"
    source = phase_root / "evidence" / "shard_00000"
    board = source / "boards" / "sample.png"
    board.parent.mkdir(parents=True)
    board.write_bytes(b"verified-board")
    board.with_suffix(".layers.json").write_text("{}\n", encoding="utf-8")
    sample_id = orchestrator.manifests["formal_test"]["rows"][0]["sample_id"]
    (source / "request_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "board_path": str(board),
                "board_sha256": hashlib.sha256(board.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        phase_root / "evidence_index.json",
        {"rows": [{"sample_id": sample_id, "source": str(source)}]},
    )
    _write_json(
        phase_root / "gallery" / "gallery.json",
        {"safety": {"input_boards_byte_identical": True}},
    )
    external = tmp_path / "external.png"
    external.write_bytes(b"never-delete")
    with (source / "request_manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "board_path": str(external),
                    "board_sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
                }
            )
            + "\n"
        )

    result = orchestrator._cleanup_formal_boards_after_gallery()

    assert result["status"] == "complete"
    assert result["deleted_board_count"] == 1
    assert result["retained_external_board_count"] == 1
    assert not board.exists()
    assert not board.with_suffix(".layers.json").exists()
    assert external.read_bytes() == b"never-delete"
    assert orchestrator._cleanup_formal_boards_after_gallery() == result
