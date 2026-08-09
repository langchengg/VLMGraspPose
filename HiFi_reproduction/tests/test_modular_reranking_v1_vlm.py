from __future__ import annotations

import json
import math
import socket
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.grasping.reranking_v1.local_vlm import (
    OllamaLocalVLMBackend,
    SESSION_AUDIT_POLICY_VERSION,
    VLMGenerationOptions,
    VLMRankingRequest,
    audit_request_gt_free,
    stable_session_contract,
    stable_session_contract_sha256,
    validate_effective_result_record,
)
from src.grasping.reranking_v1.vlm_visualization import (
    build_vlm_visualization_recipe,
    build_vlm_visualizations,
    canonical_recipe_sha256,
    render_vlm_visualization_recipe,
)
from tools.modular_reranking.prepare_vlm_inputs import main as prepare_vlm_main
from tools.modular_reranking.run_local_vlm_reranking import (
    RuntimeMonitor,
    _canonical_sha256,
    _formal_context,
    _locked_vlm_input_summary,
    _rank_recipe_record_durably,
    _recipe_binding,
    _safe_remove_on_demand_visual_dir,
    _validate_completed_formal_run,
    _validate_local_audit,
)
from tools.modular_reranking.select_vlm_validation_variant import (
    canonical_sha256,
    main as select_vlm_variant_main,
)
from tools.modular_reranking.select_vlm_safe_switch import (
    main as select_vlm_safe_switch_main,
)
from tools.modular_reranking.select_vlm_repeat_subset import (
    select_evenly_spaced,
)
from tools.modular_reranking.apply_vlm_safe_switch import (
    main as apply_vlm_safe_switch_main,
)
from tools.modular_reranking.audit_vlm_model_candidates import (
    _candidate as audit_vlm_candidate,
)
from tools.modular_reranking.compare_vlm_repeats import (
    main as compare_vlm_repeats_main,
)
from tools.modular_reranking.audit_local_ollama import (
    atomic_json as atomic_local_audit_json,
)
from tools.modular_reranking.evaluate_rerankers import (
    _validate_formal_vlm_summary_runtime,
)
from src.grasping.reranking_v1.artifact_contract import (
    LOCAL_VLM_GEOMETRY_SEMANTICS,
    identity_payload,
    validate_vlm_summary_runtime_binding,
)
from src.grasping.reranking_v1.identity import sha256_file
from src.grasping.reranking_v1.experiment_lock import (
    _validate_vlm_input_variants,
)
from src.grasping.reranking_v1.method_namespace import (
    expected_formal_method_protocols,
)

VLM_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "modular_reranking_repeatedfilm_v1.yaml"
)


def _candidates() -> list[dict]:
    return [
        {
            "candidate_id": "g0000",
            "center_uv": [24.0, 28.0],
            "angle_rad": 0.0,
            "width_px": 20.0,
        },
        {
            "candidate_id": "g0001",
            "center_uv": [42.0, 36.0],
            "angle_rad": math.pi / 4.0,
            "width_px": 18.0,
        },
    ]


def _visuals(tmp_path: Path):
    rgb = np.zeros((64, 80, 3), dtype=np.uint8)
    rgb[..., 1] = np.arange(80, dtype=np.uint8)[None, :]
    depth = np.linspace(0.6, 1.2, 64 * 80, dtype=np.float32).reshape(64, 80)
    depth[0, 0] = 0.0
    mask = np.zeros((64, 80), dtype=bool)
    mask[15:52, 12:66] = True
    return build_vlm_visualizations(
        sample_id="sample-1",
        rgb=rgb,
        depth_m=depth,
        predicted_mask=mask,
        candidates_q_order=_candidates(),
        output_dir=tmp_path / "visuals",
        crop_size=64,
    )


def _request(tmp_path: Path, *, metadata: bool = False) -> VLMRankingRequest:
    visuals = _visuals(tmp_path)
    candidate_metadata = {}
    if metadata:
        candidate_metadata = {
            candidate_id: {
                "q_percentile": 1.0 - 0.5 * index,
                "soft_mask_support": 0.8,
                "width_compatibility": 0.7,
                "jaw_depth_difference": 0.01,
                "clearance_proxy": 0.04,
                "candidate_cluster_size": 2.0,
            }
            for index, candidate_id in enumerate(("g0000", "g0001"))
        }
    return VLMRankingRequest(
        sample_id="sample-1",
        instruction="Pick up the green block",
        candidate_ids=("g0000", "g0001"),
        original_top1_candidate_id="g0000",
        image_paths=(
            visuals.full_scene_overlay_path,
            visuals.candidate_contact_sheet_path,
        ),
        visualization_manifest_path=visuals.manifest_path,
        candidate_metadata=candidate_metadata,
        include_metadata=metadata,
    )


def _recipe_record(
    tmp_path: Path,
) -> tuple[Path, Path, dict]:
    run_root = tmp_path / "protected_run"
    run_root.mkdir()
    (run_root / ".RUN_ACTIVE").write_text("test\n", encoding="utf-8")
    tmp_root = run_root / "tmp"
    tmp_root.mkdir()
    sources = run_root / "compact_inputs" / "val"
    sources.mkdir(parents=True)
    rgb = np.zeros((64, 80, 3), dtype=np.uint8)
    rgb[..., 1] = np.arange(80, dtype=np.uint8)[None, :]
    rgb_path = sources / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)
    depth_mm = np.linspace(
        600, 1200, 64 * 80, dtype=np.uint16
    ).reshape(64, 80)
    depth_path = sources / "depth.png"
    Image.fromarray(depth_mm).save(depth_path)
    mask = np.zeros((64, 80), dtype=np.uint8)
    mask[15:52, 12:66] = 255
    mask_path = sources / "mask.png"
    Image.fromarray(mask).save(mask_path)
    recipe = build_vlm_visualization_recipe(
        sample_id="sample-1",
        rgb_path=rgb_path,
        depth_mm_path=depth_path,
        predicted_mask_path=mask_path,
        candidates_q_order=_candidates(),
        mask_processing={
            "mask_threshold": 0.5,
            "min_component_area_px": 0,
            "retain_largest_component": False,
            "mask_erode_px": 0,
            "mask_dilate_px": 0,
        },
        crop_size=64,
    )
    recipe_sha256 = canonical_recipe_sha256(recipe)
    record = {
        "sample_id": "sample-1",
        "instruction": "Pick up the green block",
        "candidate_ids": ["g0000", "g0001"],
        "original_top1_candidate_id": "g0000",
        "include_metadata": False,
        "candidate_metadata": {},
        "split": "val",
        "gt_fields_included": False,
        "visualization_storage_mode": "on_demand_recipe",
        "visualization_recipe": recipe,
        "visualization_recipe_sha256": recipe_sha256,
    }
    return run_root, tmp_root, record


def _stable_local_audit_fixture() -> dict:
    audit = {
        "session_audit_policy_version": SESSION_AUDIT_POLICY_VERSION,
        "local_only": True,
        "remote_api_used": False,
        "machine": {"architecture": "arm64"},
        "ollama": {
            "cli_version": "0.13.5",
            "api_version": {"version": "0.13.5"},
            "executable_path": "/Applications/Ollama.app/ollama",
            "executable_sha256": "a" * 64,
            "endpoint": "http://127.0.0.1:11434",
            "server_config_path": "/tmp/server.json",
            "server_config_sha256": "b" * 64,
            "server_config": {"disable_ollama_cloud": True},
            "remote_established_connections": [],
        },
        "model": {
            "exact_name": "qwen3-vl:4b-instruct-q4_K_M",
            "manifest_sha256": "c" * 64,
            "model_layer_digest": "sha256:" + "d" * 64,
            "model_layer_bytes": 100,
            "quantization": "Q4_K_M",
            "layers": [
                {
                    "digest": "sha256:" + "d" * 64,
                    "mediaType": "application/vnd.ollama.image.model",
                    "size": 100,
                    "local_size": 100,
                    "local_sha256": "d" * 64,
                }
            ],
        },
    }
    audit["stable_session_contract"] = stable_session_contract(audit)
    audit["stable_session_contract_sha256"] = (
        stable_session_contract_sha256(audit)
    )
    return audit


def test_prepare_vlm_inputs_accepts_real_valid_empty_marker_without_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_id = "empty-sample"
    query = "pick the empty target"
    prediction_manifest = tmp_path / "predictions.jsonl"
    prediction_manifest.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "query": query,
                "split": "val",
                "source_rgb_path": str(tmp_path / "unused.png"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_root = tmp_path / "candidates"
    sample_root = candidate_root / sample_id
    sample_root.mkdir(parents=True)
    (sample_root / "metadata.json").write_text(
        json.dumps({"sample_id": sample_id, "query": query}) + "\n"
    )
    scored = tmp_path / "scored" / sample_id
    scored.mkdir(parents=True)
    marker = {
        "sample_id": sample_id,
        "scoring_status": "skipped_valid_empty",
        "source_candidate_count": 0,
        "gqcnn_scored_count": 0,
    }
    for filename in ("_SCORING_COMPLETE.json", "scoring_metadata.json"):
        (scored / filename).write_text(json.dumps(marker) + "\n")
    output = tmp_path / "vlm"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_vlm_inputs.py",
            "--prediction-manifest",
            str(prediction_manifest),
            "--candidate-root",
            str(candidate_root),
            "--scored-root",
            str(tmp_path / "scored"),
                "--output-root",
                str(output),
                "--input-mode",
                "validation",
            ],
    )
    assert prepare_vlm_main() == 0
    rows = [
        json.loads(line)
        for line in (output / "vlm_inputs.jsonl").read_text().splitlines()
    ]
    assert rows[0]["candidate_ids"] == []
    assert "full_scene_overlay_path" not in rows[0]
    aggregate = json.loads(
        (output / "aggregate_visual_manifest.json").read_text()
    )
    assert aggregate["empty_sample_ids"] == [sample_id]
    assert aggregate["visuals"] == []


def test_prepare_vlm_inputs_defaults_to_compact_recipe_without_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_root, _tmp_root, template = _recipe_record(tmp_path)
    recipe = template["visualization_recipe"]
    sample_id = template["sample_id"]
    prediction_manifest = tmp_path / "predictions.jsonl"
    prediction_row = {
        "sample_id": sample_id,
        "query": template["instruction"],
        "split": "val",
        "source_rgb_path": recipe["inputs"]["rgb"]["path"],
        "source_rgb_sha256": recipe["inputs"]["rgb"]["sha256"],
        "source_depth_path": recipe["inputs"]["depth_mm"]["path"],
        "source_depth_sha256": recipe["inputs"]["depth_mm"]["sha256"],
        "native_mask_path": recipe["inputs"]["predicted_hifi_mask"]["path"],
        "native_mask_sha256": recipe["inputs"]["predicted_hifi_mask"]["sha256"],
    }
    prediction_manifest.write_text(
        json.dumps(prediction_row) + "\n", encoding="utf-8"
    )
    candidate_root = tmp_path / "candidates"
    sample_root = candidate_root / sample_id
    sample_root.mkdir(parents=True)
    (sample_root / "metadata.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "query": template["instruction"],
                "config": {
                    "input": {
                        "mask_threshold": 0.5,
                        "min_component_area_px": 0,
                        "retain_largest_component": False,
                        "mask_erode_px": 0,
                        "mask_dilate_px": 0,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scored = tmp_path / "scored" / sample_id
    scored.mkdir(parents=True)
    scored_candidates = [
        {**candidate, "gqcnn_rank": index}
        for index, candidate in enumerate(_candidates(), start=1)
    ]
    (scored / "gqcnn_scored_candidates.json").write_text(
        json.dumps({"candidates": scored_candidates}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_vlm_inputs.py",
            "--prediction-manifest",
            str(prediction_manifest),
            "--candidate-root",
            str(candidate_root),
            "--scored-root",
            str(tmp_path / "scored"),
            "--output-root",
            str(output),
            "--input-mode",
            "validation",
        ],
    )
    assert prepare_vlm_main() == 0
    assert not list(output.rglob("*.png"))
    row = json.loads((output / "vlm_inputs.jsonl").read_text())
    assert row["visualization_storage_mode"] == "on_demand_recipe"
    assert "full_scene_overlay_path" not in row
    assert (
        row["visualization_recipe_sha256"]
        == canonical_recipe_sha256(row["visualization_recipe"])
    )
    aggregate = json.loads(
        (output / "aggregate_visual_manifest.json").read_text()
    )
    assert aggregate["storage_mode"] == "on_demand_recipe"
    assert aggregate["manifest_kind"] == "vlm_visualization_recipes"
    assert aggregate["visuals"][0]["visualization_recipe_sha256"] == (
        row["visualization_recipe_sha256"]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["ordinary_pngs_persisted"] is False


def test_recipe_render_is_pixel_identical_to_direct_render(
    tmp_path: Path,
) -> None:
    _run_root, _tmp_root, record = _recipe_record(tmp_path)
    recipe = record["visualization_recipe"]
    depth_mm = np.asarray(Image.open(recipe["inputs"]["depth_mm"]["path"]))
    depth_m = depth_mm.astype(np.float32) * np.float32(0.001)
    mask = np.asarray(
        Image.open(recipe["inputs"]["predicted_hifi_mask"]["path"])
    ) >= 0.5
    direct = build_vlm_visualizations(
        sample_id=record["sample_id"],
        rgb=recipe["inputs"]["rgb"]["path"],
        depth_m=depth_m,
        predicted_mask=mask,
        candidates_q_order=recipe["candidate_records"],
        output_dir=tmp_path / "direct",
        crop_size=64,
        source_recipe_sha256=record["visualization_recipe_sha256"],
    )
    on_demand = render_vlm_visualization_recipe(
        recipe, output_dir=tmp_path / "on_demand"
    )
    assert sha256_file(direct.full_scene_overlay_path) == sha256_file(
        on_demand.full_scene_overlay_path
    )
    assert sha256_file(direct.candidate_contact_sheet_path) == sha256_file(
        on_demand.candidate_contact_sheet_path
    )


def test_recipe_source_tamper_is_rejected_before_materialization(
    tmp_path: Path,
) -> None:
    _run_root, tmp_root, record = _recipe_record(tmp_path)
    mask_path = Path(
        record["visualization_recipe"]["inputs"]["predicted_hifi_mask"][
            "path"
        ]
    )
    Image.fromarray(np.zeros((64, 80), dtype=np.uint8)).save(mask_path)
    with pytest.raises(ValueError, match="source changed"):
        _recipe_binding(record, verify_sources=True)
    assert not (tmp_root / "vlm_ondemand_visuals").exists()


def test_formal_lock_binds_recipe_manifests_and_source_hashes(
    tmp_path: Path,
) -> None:
    _run_root, _tmp_root, template = _recipe_record(tmp_path)
    template["split"] = "test"
    recipe = template["visualization_recipe"]
    item = {
        "sample_id": template["sample_id"],
        "candidate_ids": template["candidate_ids"],
        "candidate_records_sha256": recipe["candidate_records_sha256"],
        "visualization_recipe_sha256": (
            template["visualization_recipe_sha256"]
        ),
        "storage_mode": "on_demand_recipe",
    }
    artifacts: dict[str, dict[str, str]] = {}
    for variant, include_metadata in (
        ("visual", False),
        ("visual_metadata", True),
    ):
        row = dict(template)
        row["include_metadata"] = include_metadata
        row["candidate_metadata"] = (
            {candidate_id: {"q_percentile": 0.5}
             for candidate_id in row["candidate_ids"]}
            if include_metadata
            else {}
        )
        input_path = tmp_path / f"{variant}.jsonl"
        input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        aggregate_path = tmp_path / f"{variant}.aggregate.json"
        aggregate_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "manifest_kind": "vlm_visualization_recipes",
                    "storage_mode": "on_demand_recipe",
                    "gt_free": True,
                    "input_mode": "formal_test",
                    "input_split": "test",
                    "sample_count": 1,
                    "nonempty_sample_count": 1,
                    "empty_sample_count": 0,
                    "empty_sample_ids": [],
                    "visuals": [item],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        summary_path = tmp_path / f"{variant}.summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    **identity_payload(),
                    "status": "COMPLETED",
                    "samples": 1,
                    "gt_fields_included": False,
                    "include_metadata": include_metadata,
                    "input_mode": "formal_test",
                    "input_split": "test",
                    "visual_storage": "on_demand_recipe",
                    "ordinary_pngs_persisted": False,
                    "output_jsonl": str(input_path),
                    "output_jsonl_sha256": sha256_file(input_path),
                    "aggregate_visual_manifest": str(aggregate_path),
                    "aggregate_visual_manifest_sha256": sha256_file(
                        aggregate_path
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts[f"vlm_test_input_manifest_{variant}"] = {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        }
        artifacts[f"vlm_visual_manifest_{variant}"] = {
            "path": str(aggregate_path),
            "sha256": sha256_file(aggregate_path),
        }
    _validate_vlm_input_variants(artifacts)

    mask_path = Path(recipe["inputs"]["predicted_hifi_mask"]["path"])
    Image.fromarray(np.zeros((64, 80), dtype=np.uint8)).save(mask_path)
    with pytest.raises(ValueError, match="source changed"):
        _validate_vlm_input_variants(artifacts)


def test_formal_evaluator_revalidates_recipe_without_temporary_pngs(
    tmp_path: Path,
) -> None:
    _run_root, tmp_root, input_row = _recipe_record(tmp_path)
    input_row["split"] = "test"
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text(json.dumps(input_row) + "\n")
    result_row = {
        "sample_id": input_row["sample_id"],
        "eligible_for_vlm": True,
        "input_record_sha256": _canonical_sha256(input_row),
        "visualization_storage_mode": "on_demand_recipe",
        "visualization_recipe_sha256": (
            input_row["visualization_recipe_sha256"]
        ),
        "temporary_visualization_manifest_sha256": "a" * 64,
        "temporary_image_sha256": ["b" * 64, "c" * 64],
        "request_hash": "d" * 64,
    }
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(json.dumps(result_row) + "\n")
    stable_hash = "e" * 64
    vlm = {
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256_file(input_path),
        "results_jsonl": str(results_path),
        "results_jsonl_sha256": sha256_file(results_path),
        "sample_count": 1,
        "eligible_sample_count": 1,
        "empty_skipped_count": 0,
        "model_name": "model",
        "model_digest": "sha256:model",
        "stable_session_contract_sha256": stable_hash,
    }
    common = {
        **vlm,
        "input_split": "test",
        "formal_mode": True,
        "visualization_storage_mode": "on_demand_recipe",
        "on_demand_visual_sample_count": 1,
        "ordinary_visual_pngs_retained": 0,
    }
    summary = dict(common)
    runtime = {
        **common,
        "fresh_http_call_count": 1,
        "cache_hit_count": 0,
    }
    assert not (tmp_root / "vlm_ondemand_visuals").exists()
    _validate_formal_vlm_summary_runtime(
        vlm=vlm,
        summary=summary,
        runtime=runtime,
        results_path=results_path,
    )
    result_row["visualization_recipe_sha256"] = "f" * 64
    results_path.write_text(json.dumps(result_row) + "\n")
    with pytest.raises(ValueError, match="temporary-visual provenance"):
        _validate_formal_vlm_summary_runtime(
            vlm=vlm,
            summary=summary,
            runtime=runtime,
            results_path=results_path,
        )


def test_vlm_runtime_binding_rejects_hidden_remote_connection_events(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    results_path = tmp_path / "results.jsonl"
    input_path.write_text(
        json.dumps({"sample_id": "s1", "candidate_ids": ["g0000"]})
        + "\n"
    )
    results_path.write_text(json.dumps({"sample_id": "s1"}) + "\n")
    common = {
        **identity_payload(),
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256_file(input_path),
        "results_jsonl": str(results_path),
        "results_jsonl_sha256": sha256_file(results_path),
        "input_split": "validation",
        "formal_mode": False,
        "model_name": "local-model",
        "model_digest": "sha256:" + "a" * 64,
        "stable_session_contract_sha256": "b" * 64,
        "sample_count": 1,
        "eligible_sample_count": 1,
        "empty_skipped_count": 0,
    }
    summary = dict(common)
    runtime = {
        **common,
        "local_only_runtime_passed": True,
        "remote_established_connection_event_count": 0,
        "remote_established_connection_events": [
            {"remote_address": "198.51.100.1:443"}
        ],
        "fresh_http_call_count": 1,
        "cache_hit_count": 0,
    }
    with pytest.raises(
        ValueError, match="summary/runtime binding is invalid"
    ):
        validate_vlm_summary_runtime_binding(
            summary,
            runtime,
            results_path=results_path,
            expected_split="validation",
            expected_formal_mode=False,
            expected_sample_count=1,
            expected_eligible_count=1,
            context="validation probe",
        )


def test_persisted_vlm_result_is_semantically_revalidated() -> None:
    result = {
        "selected_candidate_id": "g0001",
        "ranking": [
            {"candidate_id": "g0000", "score": 1.0, "reason_codes": ["uncertain"]},
            {"candidate_id": "g0001", "score": 0.5, "reason_codes": ["uncertain"]},
        ],
        "confidence": 0.8,
        "abstain": False,
        "switch_from_original_top1": True,
        "fallback": False,
        "parsed_model_response": {},
    }
    with pytest.raises(ValueError, match="first ranked"):
        validate_effective_result_record(
            result,
            candidate_ids=["g0000", "g0001"],
            original_top1_candidate_id="g0000",
        )


def test_formal_input_summary_binds_actual_jsonl_not_summary_path(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text('{"sample_id":"s1"}\n')
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "status": "COMPLETED",
                "gt_fields_included": False,
                "include_metadata": False,
                "output_jsonl": str(input_path),
                "output_jsonl_sha256": sha256_file(input_path),
            }
        )
    )
    lock = {
        **identity_payload(),
        "artifacts": {
            "vlm_test_input_manifest_visual": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            }
        }
    }
    resolved = _locked_vlm_input_summary(
        lock,
        artifact_name="vlm_test_input_manifest_visual",
        input_path=input_path.resolve(),
        expected_metadata=False,
    )
    assert Path(resolved["output_jsonl"]) == input_path
    with pytest.raises(ValueError, match="variant input"):
        _locked_vlm_input_summary(
            lock,
            artifact_name="vlm_test_input_manifest_visual",
            input_path=summary_path.resolve(),
            expected_metadata=False,
        )


def test_formal_vlm_run_rejects_unselected_variant_before_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_selection = tmp_path / "validation_selection.json"
    validation_selection.write_text(
        json.dumps(
            {
                **identity_payload(),
                "selection_split": "validation",
                "selected_method": "repeatedfilm_local_vlm_visual",
                "selected_variant": "visual",
                "selected_model_digest": "sha256:model",
                "selected_stable_session_contract_sha256": "e" * 64,
            }
        )
    )
    lock = {
        **identity_payload(),
        "vlm_model_digest": "sha256:model",
        "selected_vlm": {
            "source_method": "repeatedfilm_local_vlm_visual",
            "source_variant": "visual",
            "protocol": "gqcnn_top5",
            "model_digest": "sha256:model",
            "stable_session_contract_sha256": "e" * 64,
            "validation_selection_path": str(validation_selection),
            "validation_selection_sha256": sha256_file(
                validation_selection
            ),
        },
        "artifacts": {
            "config": {
                "path": str(VLM_CONFIG_PATH),
                "sha256": sha256_file(VLM_CONFIG_PATH),
            },
            "vlm_validation_selection": {
                "path": str(validation_selection),
                "sha256": sha256_file(validation_selection),
            }
        },
        "ranking_parameters": {
            "vlm_safe_switch": {
                "method": "repeatedfilm_local_vlm_safe_switch"
            }
        },
        "evaluation_definition": {
            "formal_method_protocols": [
                {"protocol": protocol, "method": method}
                for protocol, method in expected_formal_method_protocols(
                    "repeatedfilm_local_vlm_visual"
                )
            ]
        },
    }
    monkeypatch.setattr(
        "tools.modular_reranking.run_local_vlm_reranking.verify_lock",
        lambda _path: lock,
    )
    lock_path = tmp_path / "lock.json"
    args = SimpleNamespace(
        experiment_lock=lock_path,
        formal_inference_manifest=tmp_path / "inference.json",
        formal_variant="visual_metadata",
        limit=None,
        max_output_tokens=768,
    )
    with pytest.raises(ValueError, match="validation-selected variant"):
        _formal_context(
            args,
            records=[],
            input_path=tmp_path / "inputs.jsonl",
            locked_local_audit_path=tmp_path / "audit.json",
            session_local_audit_path=tmp_path / "session.json",
            session_local_audit={},
            output_dir=tmp_path / "formal",
        )
    args.formal_variant = "visual"
    args.max_output_tokens = 512
    with pytest.raises(ValueError, match="max output tokens"):
        _formal_context(
            args,
            records=[],
            input_path=tmp_path / "inputs.jsonl",
            locked_local_audit_path=tmp_path / "audit.json",
            session_local_audit_path=tmp_path / "session.json",
            session_local_audit={},
            output_dir=tmp_path / "formal",
        )
    assert not lock_path.with_name(
        f"{lock_path.name}.FORMAL_VLM_METADATA_STARTED.json"
    ).exists()


def test_validation_variant_selector_uses_recomputed_net_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = {
        **identity_payload(),
        "report_recomputation_verified": True,
        "candidate_pool_modified": False,
        "sample_count_all": 2,
        "per_method_metrics": [
            {
                "protocol": "gqcnn_top5",
                "method": "repeatedfilm_local_vlm_visual",
                "net_count": 1,
                "j_at_1_all": 0.5,
            },
            {
                "protocol": "gqcnn_top5",
                "method": "repeatedfilm_local_vlm_visual_metadata",
                "net_count": 2,
                "j_at_1_all": 1.0,
            },
        ],
        "provenance": {"sources": []},
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    audit_path = tmp_path / "models.json"
    audit_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "selection_split": "validation",
                "config": str(VLM_CONFIG_PATH),
                "config_sha256": sha256_file(VLM_CONFIG_PATH),
                "maximum_memory_mib": 20_000.0,
                "maximum_p95_seconds": 180.0,
                "source_artifacts": [],
                "model_candidates": [
                    {
                        "size_class": "4B",
                        "model_digest": "digest-4b",
                        "stable_session_contract_sha256": "e" * 64,
                        "eligible": True,
                    },
                    {
                        "size_class": "8B",
                        "model_digest": "digest-8b",
                        "stable_session_contract_sha256": "f" * 64,
                        "eligible": False,
                    },
                ],
            }
        )
    )
    run_specs = []
    for method in (
        "repeatedfilm_local_vlm_visual",
        "repeatedfilm_local_vlm_visual_metadata",
    ):
        root = tmp_path / method
        root.mkdir()
        input_path = root / "inputs.jsonl"
        include_metadata = (
            method == "repeatedfilm_local_vlm_visual_metadata"
        )
        input_rows = [
            {
                "sample_id": f"s{index}",
                "candidate_ids": ["g0"],
                "candidate_metadata": (
                    {"g0": {"q_percentile": 0.5}}
                    if include_metadata
                    else {}
                ),
                "include_metadata": include_metadata,
                "gt_fields_included": False,
            }
            for index in range(2)
        ]
        input_path.write_text(
            "".join(json.dumps(row) + "\n" for row in input_rows)
        )
        results = root / "results.jsonl"
        results.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "input_record_sha256": canonical_sha256(row),
                    }
                )
                + "\n"
                for row in input_rows
            )
        )
        summary = root / "summary.json"
        summary.write_text(
            json.dumps(
                {
                        **identity_payload(),
                        "results_jsonl": str(results.resolve()),
                        "results_jsonl_sha256": sha256_file(results),
                        "sample_count": 2,
                        "eligible_sample_count": 2,
                        "empty_skipped_count": 0,
                        "model_name": "qwen",
                        "model_digest": "digest-4b",
                        "stable_session_contract_sha256": "e" * 64,
                        "memory_peak_mib": 4096.0,
                        "input_jsonl": str(input_path.resolve()),
                        "input_jsonl_sha256": sha256_file(input_path),
                        "input_split": "validation",
                        "formal_mode": False,
                }
            )
        )
        runtime = root / "runtime.json"
        runtime.write_text(
            json.dumps(
                {
                    **identity_payload(),
                    "local_only_runtime_passed": True,
                    "remote_established_connection_event_count": 0,
                    "remote_established_connection_events": [],
                        "fresh_latency_p95_seconds": 2.0,
                        "sample_count": 2,
                        "eligible_sample_count": 2,
                        "empty_skipped_count": 0,
                        "fresh_http_call_count": 2,
                        "cache_hit_count": 0,
                        "model_name": "qwen",
                        "model_digest": "digest-4b",
                        "stable_session_contract_sha256": "e" * 64,
                        "input_jsonl": str(input_path.resolve()),
                        "input_jsonl_sha256": sha256_file(input_path),
                        "results_jsonl": str(results.resolve()),
                        "results_jsonl_sha256": sha256_file(results),
                        "input_split": "validation",
                        "formal_mode": False,
                }
            )
        )
        bundle["provenance"]["sources"].extend(
            [
                {
                    "role": "vlm_prediction_and_runtime",
                    "method": method,
                    "protocol": "gqcnn_top5",
                    "path": str(results.resolve()),
                    "sha256": sha256_file(results),
                },
                {
                    "role": "vlm_runtime_summary",
                    "method": method,
                    "protocol": "gqcnn_top5",
                    "path": str(summary.resolve()),
                    "sha256": sha256_file(summary),
                },
            ]
        )
        run_specs.append(f"{method}={results},{summary},{runtime}")
    bundle_path.write_text(json.dumps(bundle))
    output = tmp_path / "selection"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_vlm_validation_variant.py",
            "--config",
            str(VLM_CONFIG_PATH),
            "--evaluation-bundle",
            str(bundle_path),
            "--variant-run",
            run_specs[0],
            "--variant-run",
            run_specs[1],
            "--model-candidate-audit",
            str(audit_path),
            "--maximum-memory-mib",
            "20000",
            "--maximum-p95-seconds",
            "180",
            "--output-root",
            str(output),
        ],
    )
    assert select_vlm_variant_main() == 0
    selection = json.loads((output / "selection.json").read_text())
    assert (
        selection["selected_method"]
        == "repeatedfilm_local_vlm_visual_metadata"
    )
    assert selection["candidates"][1]["net_gain"] == 2
    memory_index = sys.argv.index("--maximum-memory-mib") + 1
    sys.argv[memory_index] = "19999"
    with pytest.raises(ValueError, match="preregistered config"):
        select_vlm_variant_main()
    sys.argv[memory_index] = "20000"
    tampered_runtime_path = Path(run_specs[0].rsplit(",", 1)[1])
    tampered_runtime = json.loads(tampered_runtime_path.read_text())
    tampered_runtime["results_jsonl_sha256"] = "0" * 64
    tampered_runtime_path.write_text(json.dumps(tampered_runtime))
    with pytest.raises(ValueError, match="summary/runtime binding"):
        select_vlm_variant_main()


@pytest.mark.parametrize(
    ("variant", "stage", "source_method"),
    [
        (
            "visual",
            "VLM_VISUAL",
            "repeatedfilm_local_vlm_visual",
        ),
        (
            "visual_metadata",
            "VLM_METADATA",
            "repeatedfilm_local_vlm_visual_metadata",
        ),
    ],
)
def test_formal_vlm_safe_apply_binds_selected_variant_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    stage: str,
    source_method: str,
) -> None:
    candidate_path = tmp_path / "test_candidates.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "scene_id": "scene",
                "split": "test",
                "candidate_id": "g0000",
                "candidate_identity_sha256": "identity",
                "original_gqcnn_rank": 1,
                "collision_proxy_total": 0.1,
            }
        ]
    ).to_parquet(candidate_path, index=False)
    universe_path = tmp_path / "test_universe.parquet"
    pd.DataFrame(
        [{"sample_id": "s1", "scene_id": "scene", "split": "test"}]
    ).to_parquet(universe_path, index=False)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "method": "repeatedfilm_local_vlm_safe_switch",
                "source_method": source_method,
                "source_variant": variant,
                "selection_split": "validation",
                "threshold_kind": "never_switch",
                "threshold": None,
                "geometry_risk_column": "collision_proxy_total",
                "geometry_risk_threshold": 0.5,
                "geometry_semantics": LOCAL_VLM_GEOMETRY_SEMANTICS,
                "config": str(VLM_CONFIG_PATH),
                "config_sha256": sha256_file(VLM_CONFIG_PATH),
                "vlm_model_digest": "sha256:model",
            }
        )
    )
    validation_selection_path = tmp_path / "validation_selection.json"
    validation_selection_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "selection_split": "validation",
                "selected_method": source_method,
                "selected_variant": variant,
                "selected_model_digest": "sha256:model",
                "selected_stable_session_contract_sha256": "e" * 64,
            }
        )
    )
    results_path = tmp_path / "results.jsonl"
    effective = {
        "selected_candidate_id": "g0000",
        "ranking": [
            {
                "candidate_id": "g0000",
                "score": 1.0,
                "reason_codes": ["uncertain"],
            }
        ],
        "confidence": 0.8,
        "abstain": False,
        "switch_from_original_top1": False,
    }
    results_path.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                **effective,
                "fallback": False,
                "parsed_model_response": effective,
                "eligible_for_vlm": True,
            }
        )
        + "\n"
    )
    lock_path = tmp_path / "lock.json"
    lock_path.write_text("{}\n")
    inference_root = tmp_path / "formal_inference"
    inference_root.mkdir()
    prediction_path = inference_root / "predictions.parquet"
    pd.DataFrame([{"sample_id": "s1"}]).to_parquet(
        prediction_path, index=False
    )
    inference_manifest_path = inference_root / "inference_manifest.json"
    inference_manifest_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "lock_content_sha256": "lock-hash",
                "predictions": str(prediction_path),
                "predictions_sha256": sha256_file(prediction_path),
            }
        )
    )
    lock_path.with_name(
        f"{lock_path.name}.FORMAL_TEST_STARTED.json"
    ).write_text(
        json.dumps(
            {
                **identity_payload(),
                "output_root": str(inference_root),
            }
        )
    )
    vlm_root = tmp_path / f"formal_{variant}"
    vlm_root.mkdir()
    vlm_manifest_path = vlm_root / "formal_vlm_manifest.json"
    vlm_manifest_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "completed": True,
                "stage": stage,
                "variant": variant,
                "source_method": source_method,
                "validation_selection_path": str(
                    validation_selection_path
                ),
                "validation_selection_sha256": sha256_file(
                    validation_selection_path
                ),
                "lock_content_sha256": "lock-hash",
                "formal_inference_manifest": str(inference_manifest_path),
                "formal_inference_manifest_sha256": sha256_file(
                    inference_manifest_path
                ),
                "model_digest": "sha256:model",
                "results_jsonl": str(results_path),
                "results_jsonl_sha256": sha256_file(results_path),
                "sample_count": 1,
            }
        )
    )
    vlm_stage_ledger = lock_path.with_name(
        f"{lock_path.name}.FORMAL_{stage}_STARTED.json"
    )
    vlm_stage_ledger.write_text(
        json.dumps(
            {
                **identity_payload(),
                "output_root": str(vlm_root),
            }
        )
    )
    lock_path.with_name(
        f"{lock_path.name}.FORMAL_{stage}_COMPLETED.json"
    ).write_text(
        json.dumps(
            {
                **identity_payload(),
                "lock_content_sha256": "lock-hash",
                "stage": stage,
                "start_ledger": str(vlm_stage_ledger),
                "start_ledger_sha256": sha256_file(vlm_stage_ledger),
                "manifest_path": str(vlm_manifest_path),
                "manifest_sha256": sha256_file(vlm_manifest_path),
            }
        )
    )
    lock = {
        **identity_payload(),
        "manifest_content_sha256": "lock-hash",
        "expected_test_sample_count": 1,
        "vlm_model_digest": "sha256:model",
        "selected_vlm": {
            "source_method": source_method,
            "source_variant": variant,
            "protocol": "gqcnn_top5",
            "model_digest": "sha256:model",
            "stable_session_contract_sha256": "e" * 64,
            "validation_selection_path": str(validation_selection_path),
            "validation_selection_sha256": sha256_file(
                validation_selection_path
            ),
        },
        "ranking_parameters": {
            "vlm_safe_switch": {
                "method": "repeatedfilm_local_vlm_safe_switch",
                "threshold_kind": "never_switch",
                "threshold": None,
                "geometry_risk_column": "collision_proxy_total",
                "geometry_risk_threshold": 0.5,
                "geometry_semantics": LOCAL_VLM_GEOMETRY_SEMANTICS,
            }
        },
        "evaluation_definition": {
            "formal_method_protocols": [
                {"protocol": protocol, "method": method}
                for protocol, method in expected_formal_method_protocols(
                    source_method
                )
            ]
        },
        "artifacts": {
            "config": {
                "path": str(VLM_CONFIG_PATH),
                "sha256": sha256_file(VLM_CONFIG_PATH),
            },
            "test_per_candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
            },
            "test_sample_universe": {
                "path": str(universe_path),
                "sha256": sha256_file(universe_path),
            },
            "vlm_safe_switch_selection": {
                "path": str(selection_path),
                "sha256": sha256_file(selection_path),
            },
            "vlm_validation_selection": {
                "path": str(validation_selection_path),
                "sha256": sha256_file(validation_selection_path),
            },
        },
    }
    monkeypatch.setattr(
        "tools.modular_reranking.apply_vlm_safe_switch.verify_lock",
        lambda _path: lock,
    )
    monkeypatch.setattr(
        "tools.modular_reranking.apply_vlm_safe_switch.consume_formal_stage_once",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "tools.modular_reranking.apply_vlm_safe_switch.complete_formal_stage_once",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "tools.modular_reranking.apply_vlm_safe_switch.validate_candidate_contract",
        lambda *_args, **_kwargs: None,
    )
    output = tmp_path / f"apply_{variant}"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_vlm_safe_switch.py",
            "--per-candidate",
            str(candidate_path),
            "--vlm-results",
            str(results_path),
            "--formal-vlm-manifest",
            str(vlm_manifest_path),
            "--sample-universe",
            str(universe_path),
            "--selection",
            str(selection_path),
            "--experiment-lock",
            str(lock_path),
            "--formal-inference-manifest",
            str(inference_manifest_path),
            "--output-root",
            str(output),
        ],
    )
    assert apply_vlm_safe_switch_main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["formal_vlm_stage"] == stage
    assert manifest["source_variant"] == variant
    assert manifest["source_method"] == source_method


def test_vlm_safe_selector_prefers_never_switch_over_zero_net_harm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_rows = []
    result_rows = []
    for sample_id, positives in (
        ("recovered", (False, True)),
        ("harmed", (True, False)),
    ):
        candidate_ids = [f"{sample_id}_g0", f"{sample_id}_g1"]
        for rank, (candidate_id, positive) in enumerate(
            zip(candidate_ids, positives, strict=True), start=1
        ):
            candidate_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_id": sample_id,
                    "split": "val",
                    "candidate_id": candidate_id,
                    "original_gqcnn_rank": rank,
                    "candidate_positive": positive,
                    "collision_proxy_total": 0.1,
                }
            )
        effective = {
            "selected_candidate_id": candidate_ids[1],
            "ranking": [
                {
                    "candidate_id": candidate_ids[1],
                    "score": 0.9,
                    "reason_codes": ["target_support"],
                },
                {
                    "candidate_id": candidate_ids[0],
                    "score": 0.2,
                    "reason_codes": ["uncertain"],
                },
            ],
            "confidence": 0.9,
            "abstain": False,
            "switch_from_original_top1": True,
        }
        result_rows.append(
            {
                "sample_id": sample_id,
                **effective,
                "fallback": False,
                "parsed_model_response": effective,
                "eligible_for_vlm": True,
                "http_call_performed": True,
            }
        )
    candidate_path = tmp_path / "validation_candidates.parquet"
    pd.DataFrame(candidate_rows).to_parquet(candidate_path, index=False)
    sample_path = tmp_path / "validation_samples.parquet"
    pd.DataFrame(
        [
            {"sample_id": sample_id, "scene_id": sample_id, "split": "val"}
            for sample_id in ("recovered", "harmed")
        ]
    ).to_parquet(sample_path, index=False)
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in result_rows)
    )
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "candidate_ids": [
                        f"{sample_id}_g0",
                        f"{sample_id}_g1",
                    ],
                }
            )
            + "\n"
            for sample_id in ("recovered", "harmed")
        )
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "results_jsonl": str(results_path.resolve()),
                "results_jsonl_sha256": sha256_file(results_path),
                "input_jsonl": str(input_path.resolve()),
                "input_jsonl_sha256": sha256_file(input_path),
                "input_split": "validation",
                "formal_mode": False,
                "sample_count": 2,
                "eligible_sample_count": 2,
                "empty_skipped_count": 0,
                "model_name": "model",
                "model_digest": "sha256:model",
                "stable_session_contract_sha256": "e" * 64,
                "ollama_version": "1",
            }
        )
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "sample_count": 2,
                "eligible_sample_count": 2,
                "empty_skipped_count": 0,
                "local_only_runtime_passed": True,
                "remote_established_connection_event_count": 0,
                "remote_established_connection_events": [],
                "fresh_http_call_count": 2,
                "cache_hit_count": 0,
                "input_jsonl": str(input_path.resolve()),
                "input_jsonl_sha256": sha256_file(input_path),
                "results_jsonl": str(results_path.resolve()),
                "results_jsonl_sha256": sha256_file(results_path),
                "input_split": "validation",
                "formal_mode": False,
                "model_name": "model",
                "model_digest": "sha256:model",
                "stable_session_contract_sha256": "e" * 64,
            }
        )
    )
    output = tmp_path / "safe"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_vlm_safe_switch.py",
            "--config",
            str(VLM_CONFIG_PATH),
            "--per-candidate",
            str(candidate_path),
            "--per-sample",
            str(sample_path),
            "--vlm-results",
            str(results_path),
            "--vlm-summary",
            str(summary_path),
            "--vlm-runtime-metrics",
            str(runtime_path),
            "--method",
            "repeatedfilm_local_vlm_visual",
            "--geometry-risk-column",
            "collision_proxy_total",
            "--geometry-risk-threshold",
            "0.5",
            "--harmful-rate-limit",
            "1.0",
            "--output-root",
            str(output),
        ],
    )
    assert select_vlm_safe_switch_main() == 0
    selection = json.loads((output / "selection.json").read_text())
    assert selection["threshold_kind"] == "never_switch"
    assert selection["harmful"] == 0
    assert selection["net_gain"] == 0
    geometry_index = sys.argv.index("--geometry-risk-threshold") + 1
    sys.argv[geometry_index] = "0.4"
    with pytest.raises(ValueError, match="preregistered config"):
        select_vlm_safe_switch_main()


def test_local_audit_is_bound_to_live_listener_and_server_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "server.json"
    config.write_text('{"disable_ollama_cloud": true}\n')
    executable = tmp_path / "ollama"
    executable.write_bytes(b"test-ollama")
    audit_path = tmp_path / "audit.json"
    audit = {
        "session_audit_policy_version": 1,
        "local_only": True,
        "remote_api_used": False,
        "machine": {"architecture": "arm64"},
        "model": {
            "exact_name": "model",
            "manifest_sha256": "digest",
            "model_layer_digest": "sha256:layer",
            "model_layer_bytes": 4,
            "quantization": "Q4",
            "layers": [
                {
                    "digest": "sha256:layer",
                    "mediaType": "application/vnd.ollama.image.model",
                    "size": 4,
                    "local_size": 4,
                    "local_sha256": "layer",
                }
            ],
        },
        "ollama": {
            "cli_version": "ollama 1.2.3",
            "api_version": {"version": "1.2.3"},
            "executable_path": str(executable),
            "executable_sha256": sha256_file(executable),
            "endpoint": "http://127.0.0.1:11434",
            "server_config": {"disable_ollama_cloud": True},
            "server_config_path": str(config),
            "server_config_sha256": sha256_file(config),
            "remote_established_connections": [],
            "listener_pid": 123,
            "listener_process_started": "Mon Jan  1 00:00:00 2024",
        },
    }
    audit["stable_session_contract"] = stable_session_contract(audit)
    audit["stable_session_contract_sha256"] = (
        stable_session_contract_sha256(audit)
    )
    audit_path.write_text(json.dumps(audit))
    args = SimpleNamespace(
        local_audit=audit_path,
        model_name="model",
        model_digest="sha256:digest",
        ollama_version="1.2.3",
        endpoint="http://127.0.0.1:11434/api/chat",
    )

    def fake_run(command, **_kwargs):
        if "lsof" in command[0]:
            return SimpleNamespace(
                stdout=(
                    "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
                    "ollama 123 user 4u IPv4 0 0t0 TCP "
                    "127.0.0.1:11434 (LISTEN)\n"
                )
            )
        return SimpleNamespace(stdout="Mon Jan  1 00:00:00 2024\n")

    monkeypatch.setattr(
        "tools.modular_reranking.run_local_vlm_reranking.subprocess.run",
        fake_run,
    )
    assert _validate_local_audit(args)[0]["ollama"]["listener_pid"] == 123
    audit["ollama"]["listener_pid"] = 999
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="listener PID"):
        _validate_local_audit(args)
    args.endpoint = "http://127.0.0.1:9999/api/chat"
    with pytest.raises(ValueError, match="client_endpoint"):
        _validate_local_audit(args)


def test_runtime_monitor_fails_closed_on_sampling_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = RuntimeMonitor(123, interval_seconds=0.01)

    def fail_snapshot():
        raise OSError("ps unavailable")

    monkeypatch.setattr(monitor, "_snapshot", fail_snapshot)
    monitor.start()
    monitor._thread.join(timeout=1.0)
    with pytest.raises(RuntimeError, match="monitor failed"):
        monitor.stop()


def _valid_model_content(*, abstain: bool = False) -> dict:
    value = {
        "selected_candidate_id": "g0001",
        "ranking": [
            {
                "candidate_id": "g0001",
                "score": 0.9,
                "reason_codes": ["target_support", "jaw_contact"],
            },
            {
                "candidate_id": "g0000",
                "score": 0.3,
                "reason_codes": ["uncertain"],
            },
        ],
        "confidence": 0.82,
        "abstain": abstain,
        "switch_from_original_top1": True,
    }
    return {
        "model": "qwen3-vl:4b-instruct-q4_K_M",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps(value)},
        "prompt_eval_count": 101,
        "eval_count": 32,
        "total_duration": 123456,
    }


def _backend(tmp_path: Path, transport):
    return OllamaLocalVLMBackend(
        model_name="qwen3-vl:4b-instruct-q4_K_M",
        model_digest="sha256:test-model",
        backend_version="0.24.0",
        cache_dir=tmp_path / "cache",
        timeout_seconds=1.0,
        transport=transport,
    )


def test_visualizations_are_gt_free_candidate_complete_and_comparable(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    audit = audit_request_gt_free(request)
    manifest = json.loads(
        request.visualization_manifest_path.read_text(encoding="utf-8")
    )
    assert audit["gt_free"]
    assert manifest["candidate_ids"] == ["g0000", "g0001"]
    assert manifest["depth_normalization"]["same_scale_for_all_candidates"]
    assert manifest["original_top1_neutral_marker"] == "g0000"
    assert "identity colors only" in manifest["candidate_color_semantics"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in request.image_paths)


def test_ollama_payload_is_deterministic_local_structured_and_cached(
    tmp_path: Path,
) -> None:
    calls: list[dict] = []

    def transport(endpoint, payload, timeout):
        calls.append({"endpoint": endpoint, "payload": payload, "timeout": timeout})
        return _valid_model_content()

    backend = _backend(tmp_path, transport)
    request = _request(tmp_path, metadata=True)
    first = backend.rank_candidates(request)
    second = backend.rank_candidates(request)
    assert first.selected_candidate_id == "g0001"
    assert not first.fallback
    assert not first.cache_hit
    assert second.cache_hit
    assert second.request_hash == first.request_hash
    assert len(calls) == 1
    payload = calls[0]["payload"]
    assert calls[0]["endpoint"] == "http://127.0.0.1:11434/api/chat"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0.0,
        "seed": 20260728,
        "num_predict": 768,
    }
    assert payload["format"]["additionalProperties"] is False
    assert len(payload["messages"]) == 2
    assert len(payload["messages"][1]["images"]) == 2
    prompt = payload["messages"][1]["content"].lower()
    assert "candidate_positive" not in prompt
    assert "ground_truth" not in prompt
    assert "dimensionless visible-surface score" in prompt


def test_on_demand_recipe_is_deleted_only_after_result_and_cache_are_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root, tmp_root, record = _recipe_record(tmp_path)
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        return _valid_model_content()

    backend = _backend(tmp_path, transport)
    original_cleanup = _safe_remove_on_demand_visual_dir
    observed_durable_before_cleanup: list[bool] = []
    current_results = run_root / "vlm_results_1.jsonl"

    def checking_cleanup(render_dir: Path, *, tmp_root: Path) -> None:
        with sqlite3.connect(backend.cache_database_path) as connection:
            cache_rows = connection.execute(
                "SELECT COUNT(*) FROM responses"
            ).fetchone()[0]
        durable_rows = [
            json.loads(line)
            for line in current_results.read_text().splitlines()
            if line.strip()
        ]
        observed_durable_before_cleanup.append(
            cache_rows == 1
            and len(durable_rows) == 1
            and all(path.is_file() for path in render_dir.iterdir())
        )
        original_cleanup(render_dir, tmp_root=tmp_root)

    monkeypatch.setattr(
        "tools.modular_reranking.run_local_vlm_reranking."
        "_safe_remove_on_demand_visual_dir",
        checking_cleanup,
    )
    first_rows: list[dict] = []
    first = _rank_recipe_record_durably(
        record=record,
        backend=backend,
        tmp_root=tmp_root,
        results_path=current_results,
        ordered_results=first_rows,
    )
    assert first["cache_hit"] is False
    assert calls == 1
    assert observed_durable_before_cleanup == [True]
    assert not (tmp_root / "vlm_ondemand_visuals").exists()

    current_results = run_root / "vlm_results_2.jsonl"
    second_rows: list[dict] = []
    second = _rank_recipe_record_durably(
        record=record,
        backend=backend,
        tmp_root=tmp_root,
        results_path=current_results,
        ordered_results=second_rows,
    )
    assert second["cache_hit"] is True
    assert second["request_hash"] == first["request_hash"]
    assert calls == 1
    assert observed_durable_before_cleanup == [True, True]
    assert not (tmp_root / "vlm_ondemand_visuals").exists()
    assert first["visualization_recipe_sha256"] == (
        record["visualization_recipe_sha256"]
    )
    assert len(first["temporary_image_sha256"]) == 2


def test_on_demand_visuals_are_cleaned_when_backend_raises(
    tmp_path: Path,
) -> None:
    _run_root, tmp_root, record = _recipe_record(tmp_path)

    class FailingBackend:
        def rank_candidates(self, _request):
            raise RuntimeError("synthetic backend failure")

    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        _rank_recipe_record_durably(
            record=record,
            backend=FailingBackend(),  # type: ignore[arg-type]
            tmp_root=tmp_root,
            results_path=tmp_path / "results.jsonl",
            ordered_results=[],
        )
    assert not (tmp_root / "vlm_ondemand_visuals").exists()


def test_on_demand_visuals_are_cleaned_if_result_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_root, tmp_root, record = _recipe_record(tmp_path)
    backend = _backend(tmp_path, lambda *_: _valid_model_content())
    rows: list[dict] = []

    def fail_append(*_args, **_kwargs):
        raise OSError("synthetic result fsync failure")

    monkeypatch.setattr(
        "tools.modular_reranking.run_local_vlm_reranking."
        "_append_jsonl_durable",
        fail_append,
    )
    with pytest.raises(OSError, match="synthetic result fsync failure"):
        _rank_recipe_record_durably(
            record=record,
            backend=backend,
            tmp_root=tmp_root,
            results_path=tmp_path / "results.jsonl",
            ordered_results=rows,
        )
    assert rows == []
    assert not (tmp_root / "vlm_ondemand_visuals").exists()
    with sqlite3.connect(backend.cache_database_path) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            == 1
        )


def test_cleanup_refuses_any_directory_outside_protected_tmp(
    tmp_path: Path,
) -> None:
    run_root, tmp_root, _record = _recipe_record(tmp_path)
    outside = run_root / "selected_gallery" / "sample-1"
    outside.mkdir(parents=True)
    retained = outside / "selected.png"
    retained.write_bytes(b"retain me")
    with pytest.raises(ValueError, match="refusing to delete"):
        _safe_remove_on_demand_visual_dir(
            outside, tmp_root=tmp_root
        )
    assert retained.read_bytes() == b"retain me"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda value: value["ranking"].__setitem__(
                0,
                {
                    "candidate_id": "invented",
                    "score": 0.9,
                    "reason_codes": ["target_support"],
                },
            ),
            "invalid_candidate_id",
        ),
        (lambda value: value["ranking"].pop(), "missing_candidate"),
        (
            lambda value: value["ranking"][0].__setitem__("score", float("nan")),
            "non_finite_score",
        ),
        (
            lambda value: value["ranking"][0].__setitem__(
                "candidate_id", "g0000"
            ),
            "duplicate_candidate_id",
        ),
    ],
)
def test_invalid_structured_outputs_fallback_to_q_only(
    tmp_path: Path, mutate, reason: str
) -> None:
    raw = _valid_model_content()
    content = json.loads(raw["message"]["content"])
    mutate(content)
    raw["message"]["content"] = json.dumps(content)
    result = _backend(tmp_path, lambda *_: raw).rank_candidates(_request(tmp_path))
    assert result.fallback
    assert result.fallback_reason == reason
    assert result.selected_candidate_id == "g0000"
    assert [item.candidate_id for item in result.ranking] == ["g0000", "g0001"]
    assert result.parser_error


@pytest.mark.parametrize(
    ("transport", "reason"),
    [
        (
            lambda *_: {
                **_valid_model_content(),
                "message": {"content": "{bad json"},
            },
            "parser_failure",
        ),
        (
            lambda *_: (_ for _ in ()).throw(ValueError("bad Ollama envelope")),
            "parser_failure",
        ),
        (
            lambda *_: (_ for _ in ()).throw(socket.timeout("slow local model")),
            "timeout",
        ),
    ],
)
def test_parser_and_timeout_fallbacks_are_recorded(
    tmp_path: Path, transport, reason: str
) -> None:
    result = _backend(tmp_path, transport).rank_candidates(_request(tmp_path))
    assert result.fallback
    assert result.fallback_reason == reason
    assert result.selected_candidate_id == "g0000"
    assert result.parser_error


def test_abstention_falls_back_to_original_q_top1(tmp_path: Path) -> None:
    result = _backend(
        tmp_path, lambda *_: _valid_model_content(abstain=True)
    ).rank_candidates(_request(tmp_path))
    assert result.fallback and result.abstain
    assert result.fallback_reason == "abstain"
    assert result.selected_candidate_id == "g0000"


def test_gt_derived_metadata_is_rejected_before_transport(tmp_path: Path) -> None:
    request = _request(tmp_path, metadata=True)
    poisoned = dict(request.candidate_metadata)
    poisoned["g0000"] = dict(poisoned["g0000"], candidate_positive=True)
    poisoned_request = VLMRankingRequest(
        sample_id=request.sample_id,
        instruction=request.instruction,
        candidate_ids=request.candidate_ids,
        original_top1_candidate_id=request.original_top1_candidate_id,
        image_paths=request.image_paths,
        visualization_manifest_path=request.visualization_manifest_path,
        candidate_metadata=poisoned,
        include_metadata=True,
    )
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        return _valid_model_content()

    with pytest.raises(ValueError, match="forbidden GT-derived"):
        _backend(tmp_path, transport).rank_candidates(poisoned_request)
    assert calls == 0


def test_candidate_ids_and_endpoint_are_strict() -> None:
    with pytest.raises(ValueError, match="q-only order"):
        VLMRankingRequest(
            sample_id="s",
            instruction="pick it",
            candidate_ids=("g0", "g1"),
            original_top1_candidate_id="g1",
            image_paths=(Path("a"), Path("b")),
            visualization_manifest_path=Path("manifest.json"),
        )
    with pytest.raises(ValueError, match="loopback"):
        OllamaLocalVLMBackend(
            model_name="model",
            model_digest="digest",
            backend_version="version",
            cache_dir=Path("cache"),
            endpoint="https://example.com/api/chat",
        )
    with pytest.raises(ValueError, match="temperature=0"):
        VLMGenerationOptions(temperature=0.1)


def test_cache_key_binds_sample_identity(tmp_path: Path) -> None:
    backend = _backend(tmp_path, lambda *_: _valid_model_content())
    first = _request(tmp_path)
    second = VLMRankingRequest(
        sample_id="different-sample",
        instruction=first.instruction,
        candidate_ids=first.candidate_ids,
        original_top1_candidate_id=first.original_top1_candidate_id,
        image_paths=first.image_paths,
        visualization_manifest_path=first.visualization_manifest_path,
    )
    # A same-ID sidecar from another sample must fail before cache lookup.
    with pytest.raises(ValueError, match="sample_id"):
        backend._payload_and_hash(second)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("model", "wrong-model", "model_identity_mismatch"),
        ("done", False, "incomplete_backend_response"),
        ("done_reason", "length", "incomplete_backend_response"),
    ],
)
def test_backend_identity_and_completion_are_checked(
    tmp_path: Path, field: str, value, reason: str
) -> None:
    raw = _valid_model_content()
    raw[field] = value
    response = _backend(tmp_path, lambda *_: raw).rank_candidates(
        _request(tmp_path)
    )
    assert response.fallback_reason == reason
    assert response.selected_candidate_id == "g0000"


def test_tampered_cache_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path, lambda *_: _valid_model_content())
    request = _request(tmp_path)
    response = backend.rank_candidates(request)
    with sqlite3.connect(backend.cache_database_path) as connection:
        encoded = connection.execute(
            "SELECT response_json FROM responses WHERE request_hash = ?",
            (response.request_hash,),
        ).fetchone()[0]
    cached = json.loads(bytes(encoded))
    cached["ranking"][0]["candidate_id"] = "invented"
    with sqlite3.connect(backend.cache_database_path) as connection:
        connection.execute(
            "UPDATE responses SET response_json = ? WHERE request_hash = ?",
            (json.dumps(cached).encode("utf-8"), response.request_hash),
        )
    with pytest.raises(ValueError, match="payload hash mismatch"):
        backend.rank_candidates(request)


def test_vlm_model_candidate_audit_requires_complete_validation_evidence(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    local_audit = tmp_path / "local.json"
    pilot_summary = tmp_path / "summary.json"
    pilot_runtime = tmp_path / "runtime.json"
    pilot_input = tmp_path / "pilot_inputs.jsonl"
    pilot_results = tmp_path / "pilot_results.jsonl"
    repeat_a = tmp_path / "repeat_a.jsonl"
    repeat_b = tmp_path / "repeat_b.jsonl"
    repeat_a_runtime = tmp_path / "repeat_a_runtime.json"
    repeat_b_runtime = tmp_path / "repeat_b_runtime.json"
    repeat = tmp_path / "repeat.json"
    pilot_input.write_text(
        "".join(
            json.dumps({"sample_id": f"p{index}"}) + "\n"
            for index in range(100)
        )
    )
    pilot_results.write_text(
        "".join(
            json.dumps({"sample_id": f"p{index}"}) + "\n"
            for index in range(100)
        )
    )
    repeat_rows = "".join(
        json.dumps({"sample_id": f"r{index}"}) + "\n"
        for index in range(20)
    )
    repeat_a.write_text(repeat_rows)
    repeat_b.write_text(repeat_rows)
    audit_payload = {
        "session_audit_policy_version": SESSION_AUDIT_POLICY_VERSION,
        "local_only": True,
        "remote_api_used": False,
        "machine": {"architecture": "arm64"},
        "ollama": {
            "cli_version": "0.13.5",
            "api_version": {"version": "0.13.5"},
            "executable_path": "/Applications/Ollama.app/ollama",
            "executable_sha256": "b" * 64,
            "endpoint": "http://127.0.0.1:11434",
            "server_config_path": "/tmp/server.json",
            "server_config_sha256": "c" * 64,
            "server_config": {"disable_ollama_cloud": True},
            "remote_established_connections": [],
        },
        "model": {
            "exact_name": "qwen3-vl:4b-instruct-q4_K_M",
            "manifest_sha256": digest,
            "model_layer_digest": "sha256:" + "d" * 64,
            "model_layer_bytes": 100,
            "quantization": "Q4_K_M",
            "layers": [
                {
                    "digest": "sha256:" + "d" * 64,
                    "mediaType": "application/vnd.ollama.image.model",
                    "size": 100,
                    "local_size": 100,
                    "local_sha256": "d" * 64,
                }
            ],
        },
    }
    audit_payload["stable_session_contract"] = stable_session_contract(
        audit_payload
    )
    stable_hash = stable_session_contract_sha256(audit_payload)
    audit_payload["stable_session_contract_sha256"] = stable_hash
    local_audit.write_text(
        json.dumps(audit_payload)
    )
    identity = {
        **identity_payload(),
        "model_name": "qwen3-vl:4b-instruct-q4_K_M",
        "model_digest": f"sha256:{digest}",
        "stable_session_contract_sha256": stable_hash,
        "input_split": "validation",
        "formal_mode": False,
    }
    pilot_summary.write_text(
        json.dumps(
            {
                **identity,
                "input_jsonl": str(pilot_input.resolve()),
                "input_jsonl_sha256": sha256_file(pilot_input),
                "results_jsonl": str(pilot_results.resolve()),
                "results_jsonl_sha256": sha256_file(pilot_results),
                "sample_count": 100,
                "eligible_sample_count": 94,
                "newly_processed": 94,
                "cache_hits": 0,
                "memory_peak_mib": 4096.0,
            }
        )
    )
    pilot_runtime.write_text(
        json.dumps(
            {
                **identity,
                "input_jsonl": str(pilot_input.resolve()),
                "input_jsonl_sha256": sha256_file(pilot_input),
                "results_jsonl": str(pilot_results.resolve()),
                "results_jsonl_sha256": sha256_file(pilot_results),
                "sample_count": 100,
                "eligible_sample_count": 94,
                "local_only_runtime_passed": True,
                "remote_established_connection_event_count": 0,
                "remote_established_connection_events": [],
                "fresh_latency_p95_seconds": 40.0,
                "valid_structured_response_rate": 0.95,
                "fallback_rate": 0.05,
            }
        )
    )
    repeat_runtime_payload = {
        **identity,
        "sample_count": 20,
        "eligible_sample_count": 20,
        "fresh_http_call_count": 20,
        "cache_hit_count": 0,
        "local_only_runtime_passed": True,
        "remote_established_connection_event_count": 0,
        "remote_established_connection_events": [],
    }
    repeat_a_runtime.write_text(
        json.dumps(
            {
                **repeat_runtime_payload,
                "results_jsonl": str(repeat_a.resolve()),
                "results_jsonl_sha256": sha256_file(repeat_a),
            }
        )
    )
    repeat_b_runtime.write_text(
        json.dumps(
            {
                **repeat_runtime_payload,
                "results_jsonl": str(repeat_b.resolve()),
                "results_jsonl_sha256": sha256_file(repeat_b),
            }
        )
    )
    repeat.write_text(
        json.dumps(
            {
                **identity_payload(),
                "samples": 20,
                "both_runs_cache_cold": True,
                "request_hash_agreement_rate": 1.0,
                "selected_candidate_agreement_rate": 0.9,
                "ranking_id_agreement_rate": 0.95,
                "exact_parsed_json_agreement_rate": 0.9,
                "fallback_agreement_rate": 1.0,
                "inputs": {
                    "run_a": str(repeat_a.resolve()),
                    "run_a_sha256": sha256_file(repeat_a),
                    "run_b": str(repeat_b.resolve()),
                    "run_b_sha256": sha256_file(repeat_b),
                },
            }
        )
    )
    four_b, sources = audit_vlm_candidate(
        size_class="4B",
        local_audit_path=local_audit,
        pilot_summary_path=pilot_summary,
        pilot_runtime_path=pilot_runtime,
        repeat_a_path=repeat_a,
        repeat_b_path=repeat_b,
        repeat_a_runtime_path=repeat_a_runtime,
        repeat_b_runtime_path=repeat_b_runtime,
        repeat_comparison_path=repeat,
        installed_names={"qwen3-vl:4b-instruct-q4_K_M"},
        maximum_memory_mib=20_000.0,
        maximum_p95_seconds=180.0,
        minimum_repeat_agreement=0.90,
        minimum_valid_structured_rate=0.90,
        maximum_fallback_rate=0.10,
    )
    assert four_b["eligible"] is True
    assert four_b["stable_contract_valid"] is True
    assert four_b["model_digest"] == f"sha256:{digest}"
    assert len(sources) == 8

    eight_b, sources = audit_vlm_candidate(
        size_class="8B",
        local_audit_path=None,
        pilot_summary_path=None,
        pilot_runtime_path=None,
        repeat_a_path=None,
        repeat_b_path=None,
        repeat_a_runtime_path=None,
        repeat_b_runtime_path=None,
        repeat_comparison_path=None,
        installed_names={"qwen3-vl:4b-instruct-q4_K_M"},
        maximum_memory_mib=20_000.0,
        maximum_p95_seconds=180.0,
        minimum_repeat_agreement=0.90,
        minimum_valid_structured_rate=0.90,
        maximum_fallback_rate=0.10,
    )
    assert eight_b["eligible"] is False
    assert eight_b["installed"] is False
    assert sources == []


def test_vlm_repeat_subset_is_nonempty_unique_and_evenly_spaced() -> None:
    rows = [
        {"sample_id": f"s{index}", "candidate_ids": ([] if index in {2, 7} else ["g0"])}
        for index in range(12)
    ]
    selected = select_evenly_spaced(rows, 5)
    assert selected == [
        {"sample_id": "s1"},
        {"sample_id": "s4"},
        {"sample_id": "s6"},
        {"sample_id": "s9"},
        {"sample_id": "s11"},
    ]


def test_vlm_repeat_comparison_rejects_missing_request_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_a = tmp_path / "a.jsonl"
    run_b = tmp_path / "b.jsonl"
    row = {"sample_id": "s1", "cache_hit": False, "request_hash": None}
    run_a.write_text(json.dumps(row) + "\n")
    run_b.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_vlm_repeats.py",
            "--run-a",
            str(run_a),
            "--run-b",
            str(run_b),
            "--expected-samples",
            "1",
            "--output",
            str(tmp_path / "comparison.json"),
        ],
    )
    with pytest.raises(ValueError, match="64-hex request hash"):
        compare_vlm_repeats_main()


def test_local_session_audit_publication_is_immutable(
    tmp_path: Path,
) -> None:
    output = tmp_path / "attempts" / "session_audit_0001.json"
    temporary = tmp_path / "tmp"
    atomic_local_audit_json(output, {"session": 1}, tmp_root=temporary)
    original_hash = sha256_file(output)
    with pytest.raises(FileExistsError):
        atomic_local_audit_json(
            output, {"session": 2}, tmp_root=temporary
        )
    assert sha256_file(output) == original_hash


def test_completed_formal_vlm_resume_is_strict_and_stage_local(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal_vlm"
    output.mkdir()
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text('{"sample_id":"s1","candidate_ids":[]}\n')
    inference_path = tmp_path / "inference_manifest.json"
    inference_path.write_text('{"completed":true}\n')
    lock_path = tmp_path / "lock.json"
    lock_path.write_text('{"locked":true}\n')
    validation_selection_path = tmp_path / "validation_selection.json"
    validation_selection_path.write_text('{"selection_split":"validation"}\n')
    records = [{"sample_id": "s1", "candidate_ids": []}]
    results_path = output / "vlm_ranking_results.jsonl"
    result = {
        "sample_id": "s1",
        "selected_candidate_id": None,
        "ranking": [],
        "confidence": 0.0,
        "abstain": False,
        "switch_from_original_top1": False,
        "eligible_for_vlm": False,
        "skip_reason": "valid_empty_no_vlm_call",
        "input_record_sha256": _canonical_sha256(records[0]),
    }
    results_path.write_text(json.dumps(result) + "\n")
    session_audit = _stable_local_audit_fixture()
    stable_hash = session_audit["stable_session_contract_sha256"]
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "input_jsonl": str(input_path),
                "input_jsonl_sha256": sha256_file(input_path),
                "results_jsonl": str(results_path),
                "results_jsonl_sha256": sha256_file(results_path),
                "input_split": "test",
                "formal_mode": True,
                "stable_session_contract_sha256": stable_hash,
                "sample_count": 1,
                "eligible_sample_count": 0,
                "empty_skipped_count": 1,
                "model_name": "model",
                "model_digest": "sha256:model",
            }
        )
    )
    attempts = output / "attempts"
    attempts.mkdir()
    session_path = attempts / "session_audit_0001.json"
    session_path.write_text(json.dumps(session_audit) + "\n")
    attempt_path = attempts / "attempt_0001.json"
    attempt_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "attempt_number": 1,
                "previous_attempt_sha256": None,
                "lock_content_sha256": "lock-content",
                "stage": "VLM_VISUAL",
                "stable_session_contract_sha256": stable_hash,
                "session_local_audit": str(session_path),
                "session_local_audit_sha256": sha256_file(session_path),
                "prefix_result_row_count": 0,
                "prefix_results_sha256": None,
            }
        )
    )
    attempt_artifacts = [
        {"path": str(attempt_path), "sha256": sha256_file(attempt_path)}
    ]
    runtime_path = output / "vlm_runtime_metrics.json"
    runtime_path.write_text(
        json.dumps(
            {
                **identity_payload(),
                "sample_count": 1,
                "eligible_sample_count": 0,
                "empty_skipped_count": 1,
                "input_jsonl": str(input_path),
                "input_jsonl_sha256": sha256_file(input_path),
                "results_jsonl": str(results_path),
                "results_jsonl_sha256": sha256_file(results_path),
                "input_split": "test",
                "formal_mode": True,
                "model_name": "model",
                "model_digest": "sha256:model",
                "stable_session_contract_sha256": stable_hash,
                "fresh_http_call_count": 0,
                "cache_hit_count": 0,
                "local_only_runtime_passed": True,
                "formal_attempt_ledgers": attempt_artifacts,
                "formal_attempt_chain_tip_sha256": sha256_file(
                    attempt_path
                ),
            }
        )
    )
    monitor_path = output / "runtime_monitor.jsonl"
    monitor_path.write_text("")
    context = {
        "lock": {
            **identity_payload(),
            "manifest_content_sha256": "lock-content",
        },
        "lock_path": lock_path,
        "formal_inference_manifest": inference_path,
        "formal_inference_manifest_sha256": sha256_file(inference_path),
        "input_jsonl": input_path,
        "input_jsonl_sha256": sha256_file(input_path),
        "locked_local_audit_sha256": "audit-hash",
        "stable_session_contract_sha256": stable_hash,
        "aggregate_visual_manifest_sha256": "visual-hash",
        "formal_variant": "visual",
        "source_method": "repeatedfilm_local_vlm_visual",
        "validation_selection_path": str(validation_selection_path),
        "validation_selection_sha256": sha256_file(
            validation_selection_path
        ),
        "formal_stage": "VLM_VISUAL",
        "model_name": "model",
        "model_digest": "sha256:model",
        "expected_sample_count": 1,
        "expected_eligible_count": 0,
        "expected_empty_count": 1,
    }
    manifest = {
        **identity_payload(),
        "completed": True,
        "lock_content_sha256": "lock-content",
        "lock_path": str(lock_path),
        "formal_inference_manifest": str(inference_path),
        "formal_inference_manifest_sha256": sha256_file(inference_path),
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256_file(input_path),
        "locked_local_audit_sha256": "audit-hash",
        "stable_session_contract_sha256": stable_hash,
        "formal_attempt_ledgers": attempt_artifacts,
        "formal_attempt_chain_tip_sha256": sha256_file(attempt_path),
        "aggregate_visual_manifest_sha256": "visual-hash",
        "variant": "visual",
        "source_method": "repeatedfilm_local_vlm_visual",
        "validation_selection_path": str(validation_selection_path),
        "validation_selection_sha256": sha256_file(
            validation_selection_path
        ),
        "stage": "VLM_VISUAL",
        "model_name": "model",
        "model_digest": "sha256:model",
        "sample_count": 1,
        "eligible_sample_count": 0,
        "empty_skipped_count": 1,
        "results_jsonl": str(results_path),
        "results_jsonl_sha256": sha256_file(results_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "runtime_metrics": str(runtime_path),
        "runtime_metrics_sha256": sha256_file(runtime_path),
        "runtime_monitor": str(monitor_path),
        "runtime_monitor_sha256": sha256_file(monitor_path),
    }
    manifest_path = output / "formal_vlm_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert (
        _validate_completed_formal_run(output, context, records)["completed"]
        is True
    )

    manifest["completed"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FileExistsError, match="completed"):
        _validate_completed_formal_run(output, context, records)

    manifest["completed"] = True
    manifest["results_jsonl"] = str(tmp_path / "outside.jsonl")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escaped stage root"):
        _validate_completed_formal_run(output, context, records)
