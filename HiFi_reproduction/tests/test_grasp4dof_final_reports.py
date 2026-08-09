"""Evidence-bound final report generation tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from src.grasping.common.statistics import (
    DEFAULT_PAIR_SPECS,
    analyze_paired_methods,
    load_aligned_predictions,
)
from tools.grasp4dof.generate_final_reports import (
    REPORT_NAMES,
    SCIENTIFIC_SCOPE_SENTENCE,
    canonical_json_sha256,
    generate_final_reports,
    main as reports_main,
    sha256_file,
)
from tools.grasp4dof.run_formal_inference import validate_r0_reference_evidence
from tools.grasp4dof.recompute_reference import (
    RECOMPUTE_CONTRACT,
    _legacy_outcome_comparison,
)
from tools.grasp4dof.finalize_formal_results import _validated_report_hashes


SYNTHETIC_COUNT = 100
METHOD_VALUES = {
    "R0": 0.0,
    "G0": 0.54,
    "G1": 0.55,
    "C0": 0.59,
    "C1": 0.58,
    "A0": 0.50,
    "G0-O": 0.64,
    "G1-O": 0.67,
    "C0-O": 0.66,
    "C1-O": 0.68,
    "A0-O": 0.61,
}


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _metric_row(method: str, j1: float, *, latency: float) -> dict[str, object]:
    if method == "R0":
        return {
            "method_id": method,
            "method": method,
            "sample_count": SYNTHETIC_COUNT,
            "j_at_1": 0.0,
            "j_at_5": 0.0,
            "recall_at_5": 0.0,
            "candidate_pool_oracle": 0.0,
            "mrr": 0.0,
            "mean_first_valid_rank": None,
            "median_first_valid_rank": None,
            "non_empty_rate": 0.0,
            "no_grasp_rate": 1.0,
            "non_empty_j_at_1": 0.0,
            "non_empty_j_at_5": 0.0,
            "mean_raw_candidates": 0.0,
            "mean_nms_candidates": 0.0,
            "p50_latency_seconds": latency,
            "p95_latency_seconds": latency,
        }
    no_grasp = 0.02 if method == "A0" else 0.04 + latency
    non_empty = 1.0 - no_grasp
    j5 = min(1.0, j1 + 0.10)
    candidate_oracle = min(1.0, j1 + 0.15)
    return {
        "method_id": method,
        "method": method,
        "sample_count": SYNTHETIC_COUNT,
        "j_at_1": j1,
        "j_at_5": j5,
        "recall_at_5": j5,
        "candidate_pool_oracle": candidate_oracle,
        "mrr": min(1.0, j1 + 0.05),
        "mean_first_valid_rank": 1.0,
        "median_first_valid_rank": 1.0,
        "non_empty_rate": non_empty,
        "no_grasp_rate": no_grasp,
        "non_empty_j_at_1": j1 / non_empty,
        "non_empty_j_at_5": j5 / non_empty,
        "mean_raw_candidates": 5.0 * non_empty,
        "mean_nms_candidates": 5.0 * non_empty,
        "p50_latency_seconds": latency,
        "p95_latency_seconds": latency,
    }


def _build_lock(run: Path, *, primary: str) -> dict[str, object]:
    artifacts = {}
    for logical_name, relative in (
        ("test_samples", "manifests/test_samples.parquet"),
        ("test_labels", "manifests/test_labels.parquet"),
        ("formal_input_preflight", "audit/formal_input_reference_preflight.json"),
    ):
        path = run / relative
        artifacts[logical_name] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    selected = json.loads((run / "selected_configs.json").read_text())
    for method_id, record in selected.items():
        path = Path(record["path"])
        artifacts[f"config_{method_id}"] = {
            "path": path.relative_to(run).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    preflight = json.loads(
        (run / "audit/formal_input_reference_preflight.json").read_text()
    )
    lock: dict[str, object] = {
        "schema_version": 1,
        "lock_status": "LOCKED",
        "effective": True,
        "run_id": "synthetic-final-report-run",
        "run_dir": str(run),
        "repository": {"root": str(run), "git_commit": "synthetic"},
        "artifacts": artifacts,
        "source_hashes": {},
        "lineage": {
            "reference": {
                "manifest_artifact": "formal_input_preflight",
                "reference_run": preflight["r0"]["reference_run"],
                "direct_inputs": {
                    name: row["sha256"]
                    for name, row in preflight["r0"]["direct_inputs"].items()
                },
            }
        },
        "protocol": {
            "primary_method": primary,
            "expected_test_sample_count": SYNTHETIC_COUNT,
            "fixed_grasp_height_px": 20.0,
        },
    }
    lock["manifest_content_sha256"] = canonical_json_sha256(lock)
    return lock


def _fixture(tmp_path: Path) -> Path:
    run = (tmp_path / "formal-run").resolve()
    run.mkdir(parents=True)

    checkpoint = run / "protected" / "repeatedfilm.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"five-stage-repeated-film")
    repeated_config = run / "protected" / "repeatedfilm.yaml"
    repeated_config.write_text("film_stages: 5\n", encoding="utf-8")
    candidate_source = run / "protected/evaluation.bin"
    pd.DataFrame(
        columns=[
            "sample_id",
            "sample_index",
            "gqcnn_rank",
            "candidate_id",
            "center_u_px",
            "center_v_px",
            "angle_rad",
            "configured_width_px",
            "gqcnn_q_value",
            "source_candidate_index",
            "candidate_seed",
        ]
    ).to_parquet(candidate_source, index=False)
    sample_source = run / "protected/scores.bin"
    pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "sample_index": index,
                "mask_inference_seconds": 0.12,
                "candidate_generation_time_ms": 0.0,
                "gqcnn_total_time_ms": 0.0,
                "raw_candidate_count": 0,
                "failure_category": "synthetic_empty",
                "top1_correct": False,
                "top5_correct": False,
                "oracle_all": False,
            }
            for index in range(SYNTHETIC_COUNT)
        ]
    ).to_csv(sample_source, index=False)
    reference_files = {
        "evaluation": {
            "path": str(candidate_source),
            "sha256": sha256_file(candidate_source),
        },
        "scores": {"path": str(sample_source), "sha256": sha256_file(sample_source)},
    }
    direct_inputs = {
        "per_candidate": {
            **reference_files["evaluation"],
            "bytes": Path(reference_files["evaluation"]["path"]).stat().st_size,
        },
        "per_sample": {
            **reference_files["scores"],
            "bytes": Path(reference_files["scores"]["path"]).stat().st_size,
        },
    }

    source = {
        "visual_grounding_variant": "hierarchical_repeated_film",
        "single_film_allowed": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(repeated_config),
        "config_sha256": sha256_file(repeated_config),
        "checkpoint_format": "hifics_hierfilm_trainable_only_v1",
        "checkpoint_strict_load_success": True,
        "checkpoint_loaded_keys": 92,
        "checkpoint_expected_keys": 92,
        "film_injection_count": 5,
        "architecture_signature": "five independent FiLM stages",
    }
    _json(
        run / "audit/source_inventory.json",
        {"status": "PHASE_0_PASS", "source": source},
    )
    split_rows = {
        split: {
            "samples": count,
            "scenes": count,
            "gt_grasp_rectangles": count * 2,
            "predicted_mask_coverage": count,
            "frozen_manifest_sha256": (str(index) * 64)[:64],
        }
        for index, (split, count) in enumerate(
            (("train", 5), ("validation", 4), ("test", SYNTHETIC_COUNT)), start=1
        )
    }
    overlap = {
        pair: {
            "sample_id": 0,
            "scene_id": 0,
            "rgb_sha256": 0,
            "depth_sha256": 0,
            "rgbd_pair_sha256": 0,
        }
        for pair in ("train__val", "train__test", "val__test")
    }
    _json(
        run / "audit/dataset_split_audit.json",
        {"all_checks_passed": True, "splits": split_rows, "pairwise_overlap": overlap},
    )
    _json(
        run / "audit/reference_baseline_inventory.json",
        {
            "reference_run_reusable": True,
            "method_name": "repeatedfilm_dexnet_gqcnn_reference",
            "sample_count": SYNTHETIC_COUNT,
            "protected_key_files_before": reference_files,
        },
    )

    vendor_root = run / "protected" / "vendors"
    vendor_root.mkdir()
    gr_source = vendor_root / "gr.py"
    gr_checkpoint = vendor_root / "gr.pth"
    gr_license = vendor_root / "gr.LICENSE"
    gg_source = vendor_root / "gg.py"
    gg_checkpoint = vendor_root / "gg.pt"
    gg_license = vendor_root / "gg.LICENSE"
    for path, content in (
        (gr_source, b"gr source"),
        (gr_checkpoint, b"gr checkpoint"),
        (gr_license, b"gr license"),
        (gg_source, b"gg source"),
        (gg_checkpoint, b"gg checkpoint"),
        (gg_license, b"gg license"),
    ):
        path.write_bytes(content)
    gr = {
        "status": "PASS",
        "repository": "https://github.com/skumra/robotic-grasping.git",
        "pinned_commit": "bdd49367f8619be94123fb3187c2f8ad5100ef46",
        "license": {
            "spdx": "BSD-3-Clause",
            "path": str(gr_license),
            "sha256": sha256_file(gr_license),
        },
        "source_files": {
            "model": {"path": str(gr_source), "sha256": sha256_file(gr_source)}
        },
        "checkpoints": [
            {
                "checkpoint_id": "jacquard_rgbd",
                "path": str(gr_checkpoint),
                "sha256": sha256_file(gr_checkpoint),
            }
        ],
        "architecture": {"family": "GR-ConvNet configurable variant 3"},
        "provenance_disclosures": [{"id": "gr-scale", "severity": "must_disclose"}],
    }
    gg = {
        "status": "PASS",
        "repository": "https://github.com/dougsm/ggcnn.git",
        "pinned_commit": "0c50aa7600e8a30d44c5c85cebd6e3394a81f30e",
        "license": {
            "spdx": "BSD-3-Clause",
            "path": str(gg_license),
            "sha256": sha256_file(gg_license),
        },
        "source_files": {
            "model": {"path": str(gg_source), "sha256": sha256_file(gg_source)}
        },
        "checkpoints": [
            {
                "checkpoint_id": "state_dict",
                "path": str(gg_checkpoint),
                "sha256": sha256_file(gg_checkpoint),
            }
        ],
        "architecture": {"variant": "post-2020-07 bilinear-upsample GG-CNN2"},
        "provenance_disclosures": [
            {"id": "gg-release-source", "severity": "must_disclose"}
        ],
    }
    _json(run / "third_party/grconvnet_source_manifest.json", gr)
    _json(run / "third_party/ggcnn2_source_manifest.json", gg)
    smoke_entries = [
        {
            "model": model,
            "device": device,
            "dtype": "float32",
            "input_shape": [1, 4 if "GR" in model else 1, 32, 32],
            "output_shapes": [[1, 1, 32, 32]] * 4,
            "status": "PASS",
        }
        for model in ("GR-ConvNet Jacquard RGB-D", "GG-CNN2 Cornell depth")
        for device in ("cpu", "mps")
    ]
    _json(
        run / "audit/basic_device_smoke.json",
        {"status": "PASS", "entries": smoke_entries},
    )
    _json(
        run / "environment.json",
        {
            "chip": "Synthetic Apple Silicon",
            "machine": "arm64",
            "macos_version": "test",
            "pytorch": "test",
            "mps_built": True,
            "mps_available": True,
            "cuda_available": False,
            "environment_setup_anomaly": None,
        },
    )
    (run / "package_lock.txt").write_text("python==test\n", encoding="utf-8")

    configs = {}
    for method in ("G0", "G1", "C0", "C1", "A0"):
        path = run / "selected_config_files" / f"{method}.json"
        _json(
            path,
            {
                "conditioning_variant": "dilated_crop"
                if method != "A0"
                else "analytic_mask_depth",
                "device": "mps" if method != "A0" else "cpu",
                "input_size": 224
                if method.startswith("G")
                else 300
                if method.startswith("C")
                else "native",
            },
        )
        configs[method] = {"path": str(path), "sha256": sha256_file(path)}
    _json(run / "selected_configs.json", configs)

    validation = pd.DataFrame(
        [
            _metric_row(method, value, latency=0.03 + index * 0.01)
            for index, (method, value) in enumerate(
                (("G0", 0.50), ("G1", 0.56), ("C0", 0.53), ("C1", 0.57), ("A0", 0.55))
            )
        ]
    )
    validation["gt_mask_oracle_j_at_1"] = validation["j_at_1"] + 0.10
    validation["predicted_mask_oracle_gap_recovery"] = (
        validation["j_at_1"] / validation["gt_mask_oracle_j_at_1"]
    )
    validation.to_csv(run / "validation_results.csv", index=False)
    selection = {
        "selection_split": "validation",
        "test_metrics_read": False,
        "primary_method_id": "C1",
        "rate_tolerance": 0.01,
        "selection_rule": ["j_at_1", "j_at_5", "lower_no_grasp"],
        "trace": [{"selected": "C1"}],
    }
    _json(run / "primary_validation_selection.json", selection)

    latencies = {"R0": 0.12, "G0": 0.08, "G1": 0.09, "C0": 0.05, "C1": 0.06, "A0": 0.01}
    formal_rows = [
        _metric_row(method, METHOD_VALUES[method], latency=latencies[method])
        for method in ("R0", "G0", "G1", "C0", "C1", "A0")
    ]
    for method in ("G0-O", "G1-O", "C0-O", "C1-O", "A0-O"):
        formal_rows.append(_metric_row(method, METHOD_VALUES[method], latency=0.07))
    primary_row = dict(next(row for row in formal_rows if row["method_id"] == "C1"))
    primary_row["method_id"] = "locked_primary_4dof_backend"
    primary_row["source_method_id"] = "C1"
    formal_rows.append(primary_row)
    formal = pd.DataFrame(formal_rows)
    formal.to_csv(run / "formal_test_results.csv", index=False)
    formal.to_csv(run / "per_method_metrics.csv", index=False)
    formal.iloc[:-1].to_csv(run / "common_subset_comparison.csv", index=False)

    oracle_rows = []
    for method in ("G0", "G1", "C0", "C1", "A0"):
        predicted = METHOD_VALUES[method]
        gt = METHOD_VALUES[f"{method}-O"]
        oracle_rows.append(
            {
                "method_id": method,
                "pred_mask_j_at_1": predicted,
                "gt_mask_j_at_1": gt,
                "j_at_1_gap": gt - predicted,
                "pred_mask_oracle": predicted + 0.15,
                "gt_mask_oracle": min(1.0, gt + 0.15),
                "candidate_oracle_gap": gt - predicted,
            }
        )
    pd.DataFrame(oracle_rows).to_csv(run / "oracle_results.csv", index=False)

    per_sample_rows = []
    for method in METHOD_VALUES:
        metric = formal.loc[formal["method_id"] == method].iloc[0]
        for index in range(SYNTHETIC_COUNT):
            non_empty = index < round(float(metric.non_empty_rate) * SYNTHETIC_COUNT)
            pool_positive = index < round(
                float(metric.candidate_pool_oracle) * SYNTHETIC_COUNT
            )
            per_sample_rows.append(
                {
                    "method": method,
                    "method_id": method,
                    "method_name": (
                        "repeatedfilm_dexnet_gqcnn_reference"
                        if method == "R0"
                        else method
                    ),
                    "sample_id": f"sample-{index}",
                    "scene_id": f"scene-{index // 5}",
                    "conditioning": (
                        "repeatedfilm_predicted_mask"
                        if method == "R0"
                        else "synthetic_conditioning"
                    ),
                    "device": (
                        "retained_reference_macos_cpu"
                        if method == "R0"
                        else "synthetic_device"
                    ),
                    "j_at_1": index < round(float(metric.j_at_1) * SYNTHETIC_COUNT),
                    "j_at_5": index < round(float(metric.j_at_5) * SYNTHETIC_COUNT),
                    "candidate_pool_oracle": pool_positive,
                    "first_valid_rank": 1 if pool_positive else None,
                    "reciprocal_rank": float(metric.mrr),
                    "non_empty": non_empty,
                    "raw_candidate_count": 5 if non_empty else 0,
                    "nms_candidate_count": 5 if non_empty else 0,
                    "empty_reason": None if non_empty else "synthetic_empty",
                    "latency_seconds": float(metric.p50_latency_seconds),
                    "top1_candidate_id": None,
                    "top1_json": None,
                    "top1_rectangle_iou": None,
                    "top1_angle_difference_deg": None,
                    "top5_candidate_ids_json": "[]",
                    "top5_json": "[]",
                    "prediction_metadata_json": (
                        json.dumps(
                            {"source_reference_run": str(run / "protected")},
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if method == "R0"
                        else "{}"
                    ),
                }
            )
    per_sample_frame = pd.DataFrame(per_sample_rows)
    per_sample_frame = per_sample_frame.loc[
        :,
        [
            "method_id",
            "method",
            *[c for c in per_sample_frame if c not in {"method_id", "method"}],
        ],
    ]
    per_sample_frame.to_parquet(run / "per_sample_predictions.parquet", index=False)
    root_candidates = pd.DataFrame(
        [
            {
                "method_id": method,
                "method": method,
                "method_name": method,
                "sample_id": "sample-0",
                "candidate_id": f"{method}-c0",
                "rank": 1,
            }
            for method in METHOD_VALUES
            if method != "R0"
        ]
    )
    root_candidates.to_parquet(run / "per_candidate_predictions.parquet", index=False)
    pd.DataFrame([{"method_id": "G1", "epoch": 1, "validation_loss": 0.5}]).to_csv(
        run / "training_curves.csv", index=False
    )

    statistical, intervals = analyze_paired_methods(
        load_aligned_predictions(run / "per_sample_predictions.parquet"),
        DEFAULT_PAIR_SPECS,
        bootstrap_draws=10_000,
        seed=20260803,
    )
    pair_records = [asdict(pair) for pair in DEFAULT_PAIR_SPECS]
    provenance = {
        "predictions_path": str(run / "per_sample_predictions.parquet"),
        "predictions_sha256": sha256_file(run / "per_sample_predictions.parquet"),
        "pair_specifications": pair_records,
        "pair_specifications_sha256": canonical_json_sha256(pair_records),
        "pairs_source_path": None,
        "pairs_source_sha256": None,
    }
    statistical["provenance"] = provenance
    intervals["provenance"] = provenance
    _json(run / "statistical_tests.json", statistical)
    _json(run / "bootstrap_intervals.json", intervals)
    subgroup_dimensions = (
        "query_type",
        "target_area_group",
        "predicted_mask_iou_group",
        "predicted_mask_confidence_group",
        "candidate_count_group",
        "depth_validity_group",
        "object_width_group",
        "scene_clutter_group",
        "no_grasp_reason_group",
        "first_valid_rank_group",
    )
    pd.DataFrame(
        [
            {
                "method_id": "C1",
                "dimension": dimension,
                "group": "synthetic",
                "sample_count": SYNTHETIC_COUNT,
                "j_at_1": 0.58,
                "j_at_5": 0.68,
                "candidate_pool_oracle": 0.73,
                "non_empty_rate": 0.94,
            }
            for dimension in subgroup_dimensions
        ]
    ).to_csv(run / "subgroup_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "method_id": "C1",
                "sample_id": f"sample-{index}",
                "query_type": "name",
                "predicted_mask_iou": 0.5,
            }
            for index in range(SYNTHETIC_COUNT)
        ]
    ).to_parquet(run / "subgroup_sample_features.parquet", index=False)
    _json(
        run / "subgroup_analysis_manifest.json",
        {
            "analysis_scope": "formal_test_only",
            "configuration_selection_performed": False,
            "gt_derived_features_offline_analysis_only": True,
        },
    )

    failure_rows = []
    stages = ("successful", "no_candidate_generated", "ranking_failure")
    for method in ("R0", "G0", "G1", "C0", "C1", "A0"):
        for index in range(SYNTHETIC_COUNT):
            stage = stages[index % len(stages)]
            failure_rows.append(
                {
                    "method": method,
                    "sample_id": f"sample-{index}",
                    "failure_stage": stage,
                    "analysis_scope": "formal_test_only",
                    "configuration_selection_performed": False,
                }
            )
    pd.DataFrame(failure_rows).to_parquet(
        run / "per_sample_failure_stage.parquet", index=False
    )
    selection_counts = {
        method: {
            "success": {"requested": 20, "available": 1, "actual": 1},
            "ranking_failure": {"requested": 20, "available": 1, "actual": 1},
            "no_candidate": {"requested": 15, "available": 1, "actual": 1},
            "wrong_mask": {"requested": 15, "available": 1, "actual": 1},
            "angle_failure": {"requested": 10, "available": 1, "actual": 1},
            "width_failure": {"requested": 10, "available": 1, "actual": 1},
        }
        for method in ("R0", "G0", "G1", "C0", "C1", "A0")
    }
    _json(
        run / "gallery/selection_manifest.json",
        {
            "analysis_scope": "formal_test_only",
            "configuration_selection_performed": False,
            "selection": {
                "counts": selection_counts,
                "cross_method_count": {"requested": 30, "available": 3, "actual": 3},
            },
            "rendered_method_image_count": 36,
            "rendered_cross_method_image_count": 3,
            "dense_maps_status": "unavailable",
            "dense_maps_reason": "no persisted dense-map arrays",
        },
    )
    (run / "gallery/index.html").write_text(
        "<!doctype html><title>gallery</title>", encoding="utf-8"
    )

    test_labels = run / "manifests/test_labels.parquet"
    test_samples = run / "manifests/test_samples.parquet"
    test_labels.parent.mkdir()
    test_manifest = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(SYNTHETIC_COUNT)],
            "scene_id": [f"scene-{index // 5}" for index in range(SYNTHETIC_COUNT)],
        }
    )
    test_manifest.to_parquet(test_samples, index=False)
    test_manifest.assign(
        gt_grasp_rectangles=[[] for _ in range(SYNTHETIC_COUNT)]
    ).to_parquet(test_labels, index=False)
    method_relatives = {
        **{
            method: f"formal_test/{method}"
            for method in ("R0", "G0", "G1", "C0", "C1", "A0")
        },
        **{
            method: f"oracle/{method}"
            for method in ("G0-O", "G1-O", "C0-O", "C1-O", "A0-O")
        },
    }
    independent_methods = []
    for method, relative in method_relatives.items():
        directory = run / relative
        directory.mkdir(parents=True)
        method_samples = (
            per_sample_frame.loc[per_sample_frame["method_id"] == method]
            .drop(columns=["method_id", "method"])
            .rename(columns={"method_name": "method"})
        )
        method_samples.to_parquet(
            directory / "per_sample_predictions.parquet", index=False
        )
        root_candidates.loc[root_candidates["method_id"] == method].drop(
            columns=["method_id", "method"]
        ).rename(columns={"method_name": "method"}).to_parquet(
            directory / "per_candidate_predictions.parquet", index=False
        )
        metric = formal.loc[formal["method_id"] == method].iloc[0].to_dict()
        _json(directory / "metrics.json", metric)
        independent_methods.append(
            {
                "method": method,
                "status": "EXACT_MATCH",
                "method_dir": str(directory),
                "sample_count": SYNTHETIC_COUNT,
                "per_candidate_predictions": str(
                    directory / "per_candidate_predictions.parquet"
                ),
                "per_candidate_predictions_sha256": sha256_file(
                    directory / "per_candidate_predictions.parquet"
                ),
                "per_sample_predictions": str(
                    directory / "per_sample_predictions.parquet"
                ),
                "per_sample_predictions_sha256": sha256_file(
                    directory / "per_sample_predictions.parquet"
                ),
                "saved_metrics": str(directory / "metrics.json"),
                "saved_metrics_sha256": sha256_file(directory / "metrics.json"),
            }
        )
    _json(
        run / "independent_recompute_results.json",
        {
            "status": "EXACT_MATCH",
            "independent_recompute": True,
            "configuration_selection_read": False,
            "candidate_correctness_fields_trusted": False,
            "method_count": len(independent_methods),
            "sample_count": SYNTHETIC_COUNT,
            "frozen_test_labels": str(test_labels),
            "frozen_test_labels_sha256": sha256_file(test_labels),
            "methods": independent_methods,
        },
    )
    _json(
        run / "runtime_metrics.json",
        {
            method: {
                "throughput_samples_per_second": 1.0 / latency,
                "p50_backend_latency_seconds": latency,
                "p95_backend_latency_seconds": latency,
                "sample_count": SYNTHETIC_COUNT,
            }
            for method, latency in latencies.items()
        },
    )
    _json(
        run / "memory_metrics.json",
        {
            method: {"peak_rss_bytes": 1000, "peak_mps_allocated_bytes": 500}
            for method in latencies
        },
    )
    (run / "commands.log").write_text("formal command log\n", encoding="utf-8")
    _json(
        run / "audit/formal_input_reference_preflight.json",
        {
            "status": "PASS",
            "loader_contract": "row_declared_sha256_verified_fail_closed",
            "r0": {
                "direct_inputs": direct_inputs,
                "reference_run": str(run / "protected"),
                "sample_count": SYNTHETIC_COUNT,
            },
        },
    )
    _json(
        run / "storage_budget.json",
        {"schema_version": 1, "budget_bytes": 12 * 1024**3},
    )
    pd.DataFrame(
        [
            {
                "stage": "formal_complete_before_reports",
                "run_bytes": 1000,
                "budget_bytes": 12 * 1024**3,
                "within_budget": True,
            }
        ]
    ).to_csv(run / "storage_usage_by_stage.csv", index=False)

    lock = _build_lock(run, primary="C1")
    _json(run / "manifests/experiment_lock.json", lock)
    (run / "frozen_4dof_backends_experiment_manifest.json").write_bytes(
        (run / "manifests/experiment_lock.json").read_bytes()
    )
    _json(
        run / ".EXPERIMENT_LOCKED",
        {
            "lock_status": "LOCKED",
            "run_id": lock["run_id"],
            "manifest_content_sha256": lock["manifest_content_sha256"],
        },
    )
    _json(
        run / "audit/protected_source_final_verification.json",
        {
            "status": "PASS",
            "all_protected_sources_unchanged": True,
            "experiment_lock_sha256": lock["manifest_content_sha256"],
            "checks": {
                "repeatedfilm_checkpoint": {
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                    "status": "UNCHANGED",
                },
                "repeatedfilm_config": {
                    "path": str(repeated_config),
                    "sha256": sha256_file(repeated_config),
                    "status": "UNCHANGED",
                },
                "reference": {
                    name: {**record, "status": "UNCHANGED"}
                    for name, record in reference_files.items()
                },
                "reference_direct_inputs": {
                    name: {**record, "status": "UNCHANGED"}
                    for name, record in direct_inputs.items()
                },
                "vendors": {
                    name: {
                        "repository": manifest["repository"],
                        "commit": manifest["pinned_commit"],
                        "clean": True,
                        "files": [
                            {
                                "path": row["path"],
                                "sha256": row["sha256"],
                                "status": "UNCHANGED",
                            }
                            for row in (
                                *manifest["source_files"].values(),
                                *manifest["checkpoints"],
                                manifest["license"],
                            )
                        ],
                    }
                    for name, manifest in (("grconvnet", gr), ("ggcnn2", gg))
                },
            },
        },
    )
    bundle_sources = {}
    for method, relative in method_relatives.items():
        directory = run / relative
        base_method = method.removesuffix("-O")
        oracle_method = method.endswith("-O")
        run_config = {
            "schema_version": 2,
            "method_id": base_method,
            "split": "test",
            "oracle": oracle_method,
            "sample_count": SYNTHETIC_COUNT,
            "experiment_lock_sha256": lock["manifest_content_sha256"],
            "samples_manifest_sha256": lock["artifacts"]["test_samples"]["sha256"],
            "labels_manifest_sha256": lock["artifacts"]["test_labels"]["sha256"],
            "per_sample_sha256": sha256_file(
                directory / "per_sample_predictions.parquet"
            ),
            "per_candidate_sha256": sha256_file(
                directory / "per_candidate_predictions.parquet"
            ),
        }
        if method != "R0":
            run_config["config_sha256"] = lock["artifacts"][f"config_{base_method}"][
                "sha256"
            ]
        else:
            run_config.update(
                {
                    "reference_manifest": str(
                        run / "audit/formal_input_reference_preflight.json"
                    ),
                    "reference_manifest_sha256": lock["artifacts"][
                        "formal_input_preflight"
                    ]["sha256"],
                    "source_candidate_sha256": direct_inputs["per_candidate"]["sha256"],
                    "source_sample_sha256": direct_inputs["per_sample"]["sha256"],
                }
            )
        _json(directory / "run_config.json", run_config)
        _json(directory / "runtime_metrics.json", {"sample_count": SYNTHETIC_COUNT})
        _json(directory / "memory_metrics.json", {"peak_rss_bytes": 1000})
        artifact_names = {
            "metrics.json",
            "runtime_metrics.json",
            "memory_metrics.json",
            "run_config.json",
            "per_sample_predictions.parquet",
            "per_candidate_predictions.parquet",
        }
        if method == "R0":
            r0_metrics = json.loads((directory / "metrics.json").read_text())
            _json(
                directory / "independent_reference_recompute.json",
                {
                    "status": "FORMAL_RECOMPUTE_COMPLETE",
                    "sample_count": SYNTHETIC_COUNT,
                    "candidate_count": 0,
                    "saved_output_mismatch_count": 0,
                    "metrics": r0_metrics,
                    "recompute_contract": RECOMPUTE_CONTRACT,
                    "source_candidates_status": "EXACT_REUSE",
                    "formal_metrics_status": "RECOMPUTED_WITH_LOCKED_EVALUATOR",
                    "legacy_success_fields_used_for_formal_metrics": False,
                    "legacy_outcome_comparison": _legacy_outcome_comparison(
                        [
                            (
                                f"sample-{index}",
                                False,
                                False,
                                False,
                                False,
                                False,
                                False,
                            )
                            for index in range(SYNTHETIC_COUNT)
                        ]
                    ),
                    "source_candidate_path": direct_inputs["per_candidate"]["path"],
                    "source_candidate_sha256": direct_inputs["per_candidate"]["sha256"],
                    "source_sample_path": direct_inputs["per_sample"]["path"],
                    "source_sample_sha256": direct_inputs["per_sample"]["sha256"],
                    "output_sample_sha256": sha256_file(
                        directory / "per_sample_predictions.parquet"
                    ),
                    "output_candidate_sha256": sha256_file(
                        directory / "per_candidate_predictions.parquet"
                    ),
                },
            )
            artifact_names.add("independent_reference_recompute.json")
        complete = {
            "schema_version": 2,
            "status": "COMPLETE",
            "method_id": base_method,
            "split": "test",
            "oracle": oracle_method,
            "sample_count": SYNTHETIC_COUNT,
            "experiment_lock_sha256": lock["manifest_content_sha256"],
            "artifacts": {
                name: sha256_file(directory / name) for name in artifact_names
            },
        }
        if method != "R0":
            complete["config_sha256"] = lock["artifacts"][f"config_{base_method}"][
                "sha256"
            ]
        _json(directory / "COMPLETE.json", complete)
        bundle_sources[method] = {
            "directory": str(directory),
            "per_sample_sha256": sha256_file(
                directory / "per_sample_predictions.parquet"
            ),
            "per_candidate_sha256": sha256_file(
                directory / "per_candidate_predictions.parquet"
            ),
            "metrics_sha256": sha256_file(directory / "metrics.json"),
            "run_config_sha256": sha256_file(directory / "run_config.json"),
            "complete_sha256": sha256_file(directory / "COMPLETE.json"),
            "complete_filename": "COMPLETE.json",
        }
    bundle_outputs = {
        name: {"path": str(run / name), "sha256": sha256_file(run / name)}
        for name in (
            "per_sample_predictions.parquet",
            "per_candidate_predictions.parquet",
            "formal_test_results.csv",
            "per_method_metrics.csv",
            "oracle_results.csv",
            "common_subset_comparison.csv",
            "training_curves.csv",
        )
    }
    _json(
        run / "results_bundle.json",
        {
            "status": "COMPLETE",
            "expected_sample_count_per_method": SYNTHETIC_COUNT,
            "method_count": 11,
            "locked_primary_method_id": "C1",
            "sources": bundle_sources,
            "outputs": bundle_outputs,
        },
    )
    return run


def _snapshot(run: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run)): sha256_file(path)
        for path in run.rglob("*")
        if path.is_file() and "reports" not in path.parts
    }


def test_generates_exact_five_reports_without_mutating_evidence(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    before = _snapshot(run)

    result = generate_final_reports(run)

    assert result["status"] == "COMPLETE"
    assert result["configuration_selection_performed"] is False
    assert result["locked_primary_method_id"] == "C1"
    assert set(result["reports"]) == set(REPORT_NAMES)
    assert _snapshot(run) == before
    reports_dir = run / "reports"
    assert {path.name for path in reports_dir.iterdir()} == set(REPORT_NAMES)
    for name in REPORT_NAMES:
        text = (reports_dir / name).read_text(encoding="utf-8")
        assert SCIENTIFIC_SCOPE_SENTENCE in text
        assert "not physical grasp success" in text
        assert "No physical robot experiment was conducted" in text

    implementation = (reports_dir / REPORT_NAMES[0]).read_text()
    assert "https://github.com/skumra/robotic-grasping" in implementation
    assert "https://arxiv.org/abs/1909.04810" in implementation
    assert "https://doi.org/10.1109/IROS45743.2020.9340777" in implementation
    assert "https://github.com/dougsm/ggcnn" in implementation
    assert "https://arxiv.org/abs/1804.05172" in implementation
    assert "https://doi.org/10.15607/RSS.2018.XIV.021" in implementation
    assert "paper implementations were not\ncopied" in implementation

    results = (reports_dir / REPORT_NAMES[3]).read_text()
    assert "## Predicted-mask main result" in results
    assert "## GT-mask oracle" in results
    assert all(
        label in results
        for label in (
            "repeatedfilm_dexnet_gqcnn_reference",
            "repeatedfilm_grconvnet_pretrained_transfer",
            "repeatedfilm_grconvnet_ocidvlg_finetuned",
            "repeatedfilm_ggcnn2_pretrained_transfer",
            "repeatedfilm_ggcnn2_ocidvlg_finetuned",
            "repeatedfilm_mask_depth_analytic",
        )
    )
    assert results.count("**哪种方案") >= 4
    assert all(f"{number}. **" in results for number in range(1, 13))
    assert "C1−C0 ΔJ@1=-0.0100" in results
    assert "No。C1−C0" in results
    assert "锁定 primary C1 相对 R0 的 ΔJ@1=+0.5800" in results
    assert "Negative deltas and\nnon-rejections are retained verbatim" in results

    failure = (reports_dir / REPORT_NAMES[4]).read_text()
    assert "no candidate, candidate-pool failure, and ranking failure" in failure
    assert "candidate_generation" in failure
    assert "Dense maps: `unavailable`" in failure


def test_cli_is_exclusive_and_refuses_overwrite(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    assert reports_main(["--run-dir", str(run)]) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reports_main(["--run-dir", str(run)])


def test_missing_required_artifact_fails_before_any_report(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    (run / "statistical_tests.json").unlink()

    with pytest.raises(FileNotFoundError, match="statistical_tests"):
        generate_final_reports(run)

    assert not (run / "reports").exists()


def test_lock_and_validation_primary_mismatch_fails_closed(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    lock_path = run / "manifests/experiment_lock.json"
    lock = json.loads(lock_path.read_text())
    lock["protocol"]["primary_method"] = "G1"
    lock.pop("manifest_content_sha256")
    lock["manifest_content_sha256"] = canonical_json_sha256(lock)
    _json(lock_path, lock)
    (run / "frozen_4dof_backends_experiment_manifest.json").write_bytes(
        lock_path.read_bytes()
    )
    marker = json.loads((run / ".EXPERIMENT_LOCKED").read_text())
    marker["manifest_content_sha256"] = lock["manifest_content_sha256"]
    _json(run / ".EXPERIMENT_LOCKED", marker)

    with pytest.raises(ValueError, match="lock/validation primary mismatch"):
        generate_final_reports(run)

    assert not (run / "reports").exists()


def test_fabricated_statistics_fail_exact_recomputation_gate(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    path = run / "statistical_tests.json"
    value = json.loads(path.read_text())
    value["tests"][0]["delta_j_at_1_b_minus_a"] = 0.999
    value["tests"][0]["p_value_exact_two_sided"] = 0.0
    _json(path, value)

    with pytest.raises(ValueError, match="do not exactly recompute"):
        generate_final_reports(run)
    assert not (run / "reports").exists()


@pytest.mark.parametrize("artifact", ("bundle", "protected"))
def test_empty_evidence_manifests_fail_closed(tmp_path: Path, artifact: str) -> None:
    run = _fixture(tmp_path)
    if artifact == "bundle":
        path = run / "results_bundle.json"
        value = json.loads(path.read_text())
        value["outputs"] = {}
        message = "bundle output coverage"
    else:
        path = run / "audit/protected_source_final_verification.json"
        value = json.loads(path.read_text())
        value.pop("checks")
        message = "protected-source verification checks"
    _json(path, value)

    with pytest.raises(ValueError, match=message):
        generate_final_reports(run)
    assert not (run / "reports").exists()


def test_protected_decoy_cannot_mask_audited_config_drift(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    source_inventory = json.loads((run / "audit/source_inventory.json").read_text())
    audited_config = Path(source_inventory["source"]["config_path"])
    audited_config.write_text("film_stages: 1\n", encoding="utf-8")
    decoy = run / "protected/decoy.yaml"
    decoy.write_text("film_stages: 5\n", encoding="utf-8")
    protected_path = run / "audit/protected_source_final_verification.json"
    protected = json.loads(protected_path.read_text())
    protected["checks"]["repeatedfilm_config"] = {
        "path": str(decoy),
        "sha256": sha256_file(decoy),
        "status": "UNCHANGED",
    }
    _json(protected_path, protected)

    with pytest.raises(ValueError, match="does not match source audit"):
        generate_final_reports(run)
    assert not (run / "reports").exists()


def test_consolidated_rows_must_equal_formal_source_rows(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    root_path = run / "per_sample_predictions.parquet"
    root = pd.read_parquet(root_path)
    g0 = root.loc[root["method_id"] == "G0"]
    first = g0.index[g0["j_at_1"]][0]
    second = g0.index[~g0["j_at_1"]][0]
    root.loc[[first, second], "j_at_1"] = root.loc[[second, first], "j_at_1"].to_numpy()
    root.to_parquet(root_path, index=False)
    bundle_path = run / "results_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["outputs"]["per_sample_predictions.parquet"]["sha256"] = sha256_file(
        root_path
    )
    _json(bundle_path, bundle)

    with pytest.raises(ValueError, match="exact formal-source concatenation"):
        generate_final_reports(run)
    assert not (run / "reports").exists()


def test_formal_source_config_must_match_locked_config(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    directory = run / "formal_test/G0"
    run_config_path = directory / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["config_sha256"] = "f" * 64
    _json(run_config_path, run_config)
    complete_path = directory / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["config_sha256"] = "f" * 64
    complete["artifacts"]["run_config.json"] = sha256_file(run_config_path)
    _json(complete_path, complete)
    bundle_path = run / "results_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["sources"]["G0"]["run_config_sha256"] = sha256_file(run_config_path)
    bundle["sources"]["G0"]["complete_sha256"] = sha256_file(complete_path)
    _json(bundle_path, bundle)

    with pytest.raises(ValueError, match="selected-config mismatch"):
        generate_final_reports(run)
    assert not (run / "reports").exists()


def test_r0_fabricated_recompute_evidence_is_rejected(tmp_path: Path) -> None:
    run = _fixture(tmp_path)
    directory = run / "formal_test/R0"
    evidence_path = directory / "independent_reference_recompute.json"
    _json(evidence_path, {"status": "FABRICATED"})
    complete_path = directory / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["artifacts"]["independent_reference_recompute.json"] = sha256_file(
        evidence_path
    )
    _json(complete_path, complete)
    bundle_path = run / "results_bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["sources"]["R0"]["complete_sha256"] = sha256_file(complete_path)
    _json(bundle_path, bundle)

    with pytest.raises(ValueError, match="R0 retained-source lineage mismatch"):
        generate_final_reports(run)
    assert not (run / "reports").exists()


def test_r0_self_consistent_hashes_cannot_replace_semantic_recompute(
    tmp_path: Path,
) -> None:
    run = _fixture(tmp_path)
    directory = run / "formal_test/R0"
    sample_path = directory / "per_sample_predictions.parquet"
    samples = pd.read_parquet(sample_path)
    samples.loc[0, "j_at_1"] = True
    samples.to_parquet(sample_path, index=False)
    evidence_path = directory / "independent_reference_recompute.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["output_sample_sha256"] = sha256_file(sample_path)
    _json(evidence_path, evidence)
    lock = json.loads((run / "manifests/experiment_lock.json").read_text())

    with pytest.raises(ValueError, match="saved sample differs from recompute"):
        validate_r0_reference_evidence(
            directory, lock=lock, expected_count=SYNTHETIC_COUNT
        )


def test_complete_looking_fabricated_reports_cannot_be_finalized(
    tmp_path: Path,
) -> None:
    run = _fixture(tmp_path)
    report_dir = run / "reports"
    report_dir.mkdir()
    for name in REPORT_NAMES:
        (report_dir / name).write_text(
            f"# FABRICATED REPORT\n\n{SCIENTIFIC_SCOPE_SENTENCE}\n",
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="content does not match evidence"):
        _validated_report_hashes(run)
