"""Fast acceptance tests for the crash-resumable full VGN experiment.

These tests intentionally use synthetic records only.  They validate the
experiment boundary (manifest completeness, persistence, truthful metric
naming, and physical/simulation separation) without loading Open3D or a VGN
checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.run_vgn_sim_benchmark import build_blocked_aggregate
from scripts.summarize_real_robot_trials import summarize_real_robot_records
from src.experiments.bootstrap import bootstrap_experiment, validate_truthfulness_metadata
from src.experiments.experiment_store import ExperimentStore, ManifestCountMismatch
from src.experiments.failure_taxonomy import classify_status
from src.experiments.full_vgn_runner import (
    FullVGNRunner,
    build_gt_oracle_sample,
    manifest_registration_rows,
    resolve_gt_oracle_mapping,
)
from src.experiments.metrics import aggregate_metrics
from src.experiments.ocid_annotations import GTOracleMappingError
from src.grasping import vgn_pipeline
from src.grasping.vgn_pipeline import ManifestSample
from src.robot import DryRunExecutor, ExecutionRequest


def _metadata() -> dict[str, object]:
    return {
        "repository_url": "https://github.com/ethz-asl/vgn",
        "repository_branch": "corl2020",
        "repository_commit": "d7af0622433f52ae88ebe81533f12b46b33e951a",
        "checkpoint_sha256": (
            "ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b"
        ),
        "tsdf_mode": "single_view_adaptation",
        "score_source": "official_vgn_processed_quality",
        "custom_reranking": False,
        "limitations": [
            "single-view TSDF adaptation",
            "no 6-DoF ground truth in OCID-VLG",
            "no robot execution validation",
        ],
    }


def _manifest(count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample-{index}",
            "dataset_index": index,
            "scene_id": f"scene-{index // 2}",
            "instruction": f"pick target {index}",
            "view": "top",
        }
        for index in range(count)
    ]


def _truthful_result(
    *, official: int = 1, target: int = 1, quality: float | None = 0.95
) -> dict[str, object]:
    return {
        "official_candidate_count": official,
        "target_candidate_count": target,
        "top1_vgn_quality": quality,
        "score_source": "official_vgn_processed_quality",
        "custom_reranking": False,
        "tsdf_mode": "single_view_adaptation",
    }


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_nested_keys(item))
    return keys


def _synthetic_sample(root: Path, index: int = 0) -> ManifestSample:
    sample_root = root / f"input-{index}"
    sample_root.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_root / "rgb.png"
    depth_path = sample_root / "depth.png"
    mask_path = sample_root / "predicted_mask.png"
    Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(rgb_path)
    Image.fromarray(np.full((4, 5), 700, dtype=np.uint16)).save(depth_path)
    predicted = np.zeros((4, 5), dtype=np.uint8)
    predicted[1:3, 1:4] = 255
    Image.fromarray(predicted).save(mask_path)
    return ManifestSample(
        sample_id=f"sample-{index}",
        dataset_index=index,
        scene_id=f"scene-{index // 2}",
        instruction=f"pick target {index}",
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        bundle_dir=sample_root,
        metadata_path=None,
        intrinsics_path=None,
        view="top",
        row={"question_index": index},
        metadata={
            "mask_source": "predicted_mask_original_resolution",
            "width": 5,
            "height": 4,
            "fx": 4.0,
            "fy": 4.0,
            "cx": 2.0,
            "cy": 1.5,
        },
    )


def _runner_args(output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        scene_cache_size=2,
        lease_seconds=30.0,
        mask_source="predicted",
        render_all_2d=False,
        top_k=50,
        selection_policy="highest_vgn_quality",
        target_mask_dilation_px=3,
        mask_cleanup="none",
        depth_unit="mm",
        depth_scale=1000.0,
        depth_min_m=0.05,
        depth_max_m=2.0,
        workspace_size_m=0.30,
        resolution=40,
        table_height_m=0.05,
        allow_camera_aligned_fallback=False,
    )


def _write_synthetic_top1(sample: ManifestSample, *, args: Any, **_: Any) -> dict[str, Any]:
    sample_dir = Path(args.output) / "samples" / sample.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    vgn_pipeline.atomic_write_json(
        sample_dir / "top1.json",
        {
            "sample_id": sample.sample_id,
            "status": "ok",
            "failure_reason": "",
            "selection_policy": "highest_vgn_quality",
            "candidate_count_before_target_filter": 2,
            "candidate_count_after_target_filter": 1,
            "candidate": {
                "vgn_quality": 0.9375,
                "width_m": 0.04,
                "position_task_m": [0.1, 0.1, 0.1],
                "position_camera_m": [0.0, 0.0, 0.7],
                "quaternion_task_xyzw": [0.0, 0.0, 0.0, 1.0],
                "quaternion_camera_xyzw": [0.0, 0.0, 0.0, 1.0],
                "T_camera_grasp": np.eye(4).tolist(),
                "projected_uv": [2.0, 1.5],
                "official_selection_index": 0,
                "score_rank": 1,
                "inside_raw_target_mask": True,
                "inside_dilated_target_mask": True,
            },
        },
    )
    return {
        "sample_id": sample.sample_id,
        "scene_id": sample.scene_id,
        "instruction": sample.instruction,
        "status": "ok",
        "failure_reason": "",
        "official_candidate_count": 2,
        "target_candidate_count": 1,
        "top1_vgn_quality": 0.9375,
        "top1_width_m": 0.04,
        "processing_time_depth": 0.001,
        "processing_time_tsdf": 0.001,
        "processing_time_vgn": 0.001,
    }


def _attach_oracle_annotation(sample: ManifestSample, root: Path) -> ManifestSample:
    annotation_root = root / "prediction-annotation"
    annotation_root.mkdir(parents=True, exist_ok=True)
    source_instances = np.array(
        [[0, 0, 0, 0, 0], [0, 2, 2, 1, 0], [0, 2, 2, 1, 0], [0, 0, 0, 0, 0]],
        dtype=np.uint8,
    )
    instance_path = annotation_root / "instance.png"
    gt_path = annotation_root / "ground_truth_mask_original_resolution.png"
    Image.fromarray(source_instances).save(instance_path)
    Image.fromarray(np.where(source_instances == 2, 255, 0).astype(np.uint8)).save(gt_path)
    metadata_path = annotation_root / "sample_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "stable_sample_id": sample.sample_id,
                "question_index": sample.dataset_index,
                "scene_id": sample.scene_id,
                "query": sample.instruction,
                "answer_instance_value": 2,
                "source_instance_mask_path": str(instance_path),
                "target_name": "mug_3",
            }
        ),
        encoding="utf-8",
    )
    metadata = dict(sample.metadata)
    metadata["prediction_sample_metadata"] = str(metadata_path)
    return ManifestSample(**{**sample.__dict__, "metadata": metadata})


def test_manifest_count_and_no_silent_skip(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    with ExperimentStore(database, "complete-manifest") as store:
        store.initialize_run(_metadata(), 4)
        with pytest.raises(ManifestCountMismatch, match="exactly 4"):
            store.register_samples(_manifest(3), expected_count=4)

        assert store.connection.execute(
            "SELECT COUNT(*) FROM samples WHERE run_id=?", (store.run_id,)
        ).fetchone()[0] == 0
        assert store.register_samples(_manifest(4), expected_count=4) == 4
        assert [row["sample_id"] for row in store.sample_rows()] == [
            "sample-0",
            "sample-1",
            "sample-2",
            "sample-3",
        ]


def test_full_runner_resume_skips_terminal_and_requeues_retryable(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    with ExperimentStore(database, "resume-run") as first:
        bootstrap_experiment(first, _manifest(3), _metadata(), manifest_count=3)
        claim = first.claim_next("worker-before-crash", now=10.0, lease_seconds=5.0)
        assert claim is not None and claim["sample_id"] == "sample-0"
        first.complete_sample(
            "sample-0",
            "ok",
            result=_truthful_result(),
            worker_id="worker-before-crash",
            now=11.0,
        )
        retryable = first.claim_next("worker-before-crash", now=12.0, lease_seconds=5.0)
        assert retryable is not None and retryable["sample_id"] == "sample-1"
        first.fail_sample(
            "sample-1",
            "io_error",
            failure_reason="synthetic interrupted write",
            worker_id="worker-before-crash",
            now=13.0,
        )

    with ExperimentStore(database, "resume-run") as resumed:
        # Re-registering the identical immutable manifest is idempotent.
        bootstrap_experiment(resumed, _manifest(3), _metadata(), manifest_count=3)
        assert resumed.get_sample("sample-0")["state"] == "terminal"
        assert resumed.requeue_retryable() == 1
        next_claim = resumed.claim_next("worker-after-crash", now=20.0)
        assert next_claim is not None
        assert next_claim["sample_id"] == "sample-1"
        assert next_claim["attempts"] == 2


def test_full_runner_resume(tmp_path: Path) -> None:
    samples = [_synthetic_sample(tmp_path, index) for index in range(2)]
    output = tmp_path / "output"
    args = _runner_args(output)
    calls: list[str] = []

    def processor(sample: ManifestSample, **kwargs: Any) -> dict[str, Any]:
        calls.append(sample.sample_id)
        return _write_synthetic_top1(sample, **kwargs)

    database = output / "state.sqlite3"
    with ExperimentStore(database, "synthetic-full-run") as store:
        bootstrap_experiment(
            store,
            manifest_registration_rows(samples),
            _metadata(),
            manifest_count=2,
        )
        runner = FullVGNRunner(
            samples=samples,
            store=store,
            args=args,
            net=object(),
            device="cpu",
            run_metadata=_metadata(),
            processor=processor,
            worker_id="first-worker",
        )
        first = runner.run_pending()
        assert first["processed_count"] == 2
        assert first["resume_skipped_count"] == 0
        assert first["terminal_count"] == 2

    with ExperimentStore(database, "synthetic-full-run") as store:
        runner = FullVGNRunner(
            samples=samples,
            store=store,
            args=args,
            net=object(),
            device="cpu",
            run_metadata=_metadata(),
            processor=processor,
            worker_id="resumed-worker",
        )
        resumed = runner.run_pending()

    # Both immutable rows are completed, while the identical second payload is
    # allowed to reuse the content-addressed result.  Resume invokes neither.
    assert calls == ["sample-0"]
    assert resumed["processed_count"] == 0
    assert resumed["resume_skipped_count"] == 2
    assert resumed["terminal_count"] == 2
    for sample_id in ("sample-0", "sample-1"):
        assert json.loads((output / f"samples/{sample_id}/result.json").read_text())[
            "top1_vgn_quality"
        ] == pytest.approx(0.9375)


def test_atomic_write_never_exposes_partial_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"generation": 1}\n', encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        assert Path(source).parent == tmp_path
        assert Path(target) == destination
        raise OSError("synthetic rename interruption")

    monkeypatch.setattr(vgn_pipeline.os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename interruption"):
        vgn_pipeline.atomic_write_json(destination, {"generation": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"generation": 1}
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_technical_failures_and_model_outcomes_are_separate() -> None:
    assert classify_status("no_official_grasp").category == "scientific_outcome"
    assert classify_status("no_target_grasp").category == "scientific_outcome"
    assert classify_status("support_plane_failed").category == "deterministic_input_failure"
    assert classify_status("vgn_inference_failed").category == (
        "retryable_infrastructure_failure"
    )

    rows = [
        {
            "sample_id": "model-zero",
            "scene_id": "scene-a",
            "status": "no_official_grasp",
            **_truthful_result(official=0, target=0, quality=None),
        },
        {
            "sample_id": "technical",
            "scene_id": "scene-b",
            "status": "support_plane_failed",
            "failure_reason": "synthetic plane failure",
        },
    ]
    aggregate = aggregate_metrics(rows, manifest_count=2, bootstrap_replicates=0)
    assert aggregate["denominators"]["terminal_samples"]["count"] == 2
    assert aggregate["denominators"]["candidate_generation_reached"]["count"] == 1
    assert aggregate["status_counts"] == {
        "no_official_grasp": 1,
        "support_plane_failed": 1,
    }


def test_offline_metrics_never_call_candidate_coverage_success() -> None:
    rows = [
        {
            "sample_id": "sample-0",
            "scene_id": "scene-0",
            "status": "ok",
            **_truthful_result(),
        }
    ]
    aggregate = aggregate_metrics(rows, manifest_count=1, bootstrap_replicates=0)
    keys = {key.lower() for key in _nested_keys(aggregate)}
    assert not any("success" in key for key in keys)
    assert "official_candidate_availability" in keys
    assert "target_candidate_availability" in keys
    assert "manifest_processing_coverage" in keys


def test_all_top1_scores_are_official_vgn_quality() -> None:
    rows = [
        {
            "sample_id": "sample-0",
            "scene_id": "scene-0",
            "status": "ok",
            **_truthful_result(quality=0.975),
        },
        {
            "sample_id": "sample-1",
            "scene_id": "scene-1",
            "status": "no_target_grasp",
            **_truthful_result(official=3, target=0, quality=None),
        },
    ]
    truthfulness = aggregate_metrics(
        rows, manifest_count=2, bootstrap_replicates=0
    )["truthfulness"]
    assert truthfulness["all_scores_from_official_processed_quality"] is True
    assert truthfulness["score_source_values"] == [
        "official_vgn_processed_quality"
    ]

    altered = _metadata()
    altered["score_source"] = "custom_combined_score"
    with pytest.raises(ValueError, match="official_vgn_processed_quality"):
        validate_truthfulness_metadata(altered)


def test_custom_reranking_always_false() -> None:
    rows = [
        {
            "sample_id": "sample-0",
            "scene_id": "scene-0",
            "status": "ok",
            **_truthful_result(),
        }
    ]
    truthfulness = aggregate_metrics(
        rows, manifest_count=1, bootstrap_replicates=0
    )["truthfulness"]
    assert truthfulness["custom_reranking_values"] == [False]
    assert truthfulness["any_custom_reranking"] is False
    assert truthfulness["all_candidate_outcomes_disable_custom_reranking"] is True

    altered = _metadata()
    altered["custom_reranking"] = True
    with pytest.raises(ValueError, match="exactly false"):
        validate_truthfulness_metadata(altered)


def test_gt_oracle_mapping_is_unique(tmp_path: Path) -> None:
    predicted = _synthetic_sample(tmp_path)
    sample = _attach_oracle_annotation(predicted, tmp_path)
    resolved = resolve_gt_oracle_mapping(sample)
    assert resolved == (tmp_path / "prediction-annotation" / "ground_truth_mask_original_resolution.png").resolve()

    oracle_sample = build_gt_oracle_sample(sample)
    assert oracle_sample.sample_id == sample.sample_id
    assert oracle_sample.mask_path == resolved
    assert oracle_sample.metadata["mask_source"] == "ground_truth_mask_oracle"
    assert oracle_sample.metadata["gt_oracle_source_sample_id"] == sample.sample_id
    assert len(str(oracle_sample.metadata["prediction_mask_sha256"])) == 64


def test_gt_oracle_refuses_ambiguous_mapping(tmp_path: Path) -> None:
    predicted = _synthetic_sample(tmp_path)
    sample = _attach_oracle_annotation(predicted, tmp_path)
    # Preserve a plausible non-empty mask while making it disagree with the
    # source instance map.  The adapter must not choose either source by fiat.
    wrong = np.zeros((4, 5), dtype=np.uint8)
    wrong[1:3, 3] = 255
    Image.fromarray(wrong).save(
        tmp_path / "prediction-annotation" / "ground_truth_mask_original_resolution.png"
    )
    with pytest.raises(GTOracleMappingError) as captured:
        resolve_gt_oracle_mapping(sample)
    assert captured.value.status == "gt_oracle_ambiguous"
    assert "does not exactly equal" in str(captured.value)


def test_real_success_null_without_robot_logs_and_dry_run() -> None:
    executor = DryRunExecutor()
    request = ExecutionRequest(
        trial_id="trial-0",
        sample_id="sample-0",
        instruction="pick target",
        T_base_grasp=np.eye(4),
        gripper_width_m=0.04,
    )
    result = executor.execute(request)
    assert result.physical_execution_attempted is False
    assert result.physical_success is None
    summary = summarize_real_robot_records(
        [result.to_record()], planned_trial_count=1
    )
    assert summary["real_robot_grasp_success_rate"] is None
    assert summary["end_to_end_real_success_rate"] is None
    assert summary["reason"] == "no physical robot execution logs"


def test_sim_and_real_metrics_separated() -> None:
    aggregate = build_blocked_aggregate(
        {"blockers": [{"code": "missing_pybullet"}]}, {"seed": 42}
    )
    assert aggregate["metric_scope"] == "pybullet_simulated_physical_execution"
    assert aggregate["simulated_grasp_success_rate"] is None
    assert aggregate["real_robot_metrics_not_computed_here"] is True
    assert "real_robot_grasp_success_rate" not in aggregate


def test_all_manifest_rows_have_terminal_status(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "state.sqlite3", "terminal-run") as store:
        bootstrap_experiment(store, _manifest(2), _metadata(), manifest_count=2)
        for index, status in enumerate(("ok", "missing_camera_intrinsics")):
            sample_id = f"sample-{index}"
            claim = store.claim_next("worker")
            assert claim is not None and claim["sample_id"] == sample_id
            store.complete_sample(
                sample_id,
                status,
                result=_truthful_result() if status == "ok" else {},
                failure_reason="" if status == "ok" else "synthetic missing calibration",
                worker_id="worker",
            )

        assert store.claim_next("worker") is None
        rows = store.sample_rows()
        assert len(rows) == 2
        assert all(row["state"] == "terminal" for row in rows)
        aggregate = aggregate_metrics(rows, manifest_count=2, bootstrap_replicates=0)
        assert aggregate["proportions"]["manifest_processing_coverage"]["numerator"] == 2
        assert aggregate["proportions"]["manifest_processing_coverage"]["denominator"] == 2
