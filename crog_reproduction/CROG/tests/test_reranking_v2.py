import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from failure_analysis.reranking.geometry import geometry_checksum
from failure_analysis.reranking_v2.aligned_crops import (
    CROP_CHANNELS,
    build_aligned_crop,
    candidate_sampling_grid,
)
from failure_analysis.reranking_v2.artifacts import ArtifactRun
from failure_analysis.reranking_v2.calibration import (
    apply_temperature,
    fit_temperature,
)
from failure_analysis.reranking_v2.datasets import gate_outcome_labels
from failure_analysis.reranking_v2.extract import (
    _assert_hook_non_mutating,
    _frozen_export_batch_size,
)
from failure_analysis.reranking_v2.evaluation import (
    load_prediction_rankings,
)
from failure_analysis.reranking_v2.gallery import classify_failure_stage
from failure_analysis.reranking_v2.independent_evaluator import (
    assert_matches_primary,
)
from failure_analysis.reranking_v2.inference import export_npz_rankings
from failure_analysis.reranking_v2.latent_roi import (
    CROGLatentCapture,
    candidate_roi_grid,
    pool_candidate_rois,
)
from failure_analysis.reranking_v2.models.latent_residual import (
    LatentResidualRanker,
    quality_logit,
    residual_scores,
)
from failure_analysis.reranking_v2.models.pairwise_gate import (
    _stack_inference_pairs,
    select_with_gate,
)
from failure_analysis.reranking_v2.models.rgbd_critic import (
    critic_loss,
    grouped_candidate_batches,
)
from failure_analysis.reranking_v2.models.setrank import ResidualSetRank, setrank_loss
from failure_analysis.reranking_v2.models.uncertainty import (
    PERTURBATIONS,
    conservative_uncertainty_switch,
    perturbation_geometries,
    score_candidate_perturbations,
    stability_statistics,
)
from failure_analysis.reranking_v2.models.vlm_reviewer import (
    cached_review,
    map_response_to_candidate_ids,
    parse_vlm_response,
    shuffled_candidate_mapping,
)
from failure_analysis.reranking_v2.oof import validate_oof_provenance_records
from failure_analysis.reranking_v2.protocol import claim_test_once
from failure_analysis.reranking_v2.reporting import build_report_artifact
from failure_analysis.reranking_v2.schema import (
    assert_inference_record_has_no_evaluation_fields,
    assert_model_feature_names,
    stable_sample_id,
)
from failure_analysis.reranking_v2.splits import (
    assign_group_folds,
    audit_partitions,
    grouped_partition,
)


def candidate(cx=320.0, cy=240.0, angle=0.0, width=80.0, identifier="candidate_0"):
    polygon = cv2.boxPoints(((cx, cy), (width, 20.0), -angle)).tolist()
    value = {
        "candidate_id": identifier,
        "legacy_rank": int(identifier.rsplit("_", 1)[-1]),
        "q_rank": int(identifier.rsplit("_", 1)[-1]),
        "row": int(round(cy)),
        "col": int(round(cx)),
        "cx": cx,
        "cy": cy,
        "angle_rad": math.radians(angle),
        "angle_deg": angle,
        "width_px": width,
        "height_px": 20.0,
        "polygon": polygon,
        "legacy_grasp": [cx, cy, width, 20.0, angle],
        "q_raw": 0.9 - 0.05 * int(identifier.rsplit("_", 1)[-1]),
        "features": {},
    }
    value["candidate_checksum"] = geometry_checksum(value)
    return value


def label_record(values):
    return {
        "candidate_labels": [
            {
                "candidate_id": f"candidate_{index}",
                "candidate_correct": bool(value),
            }
            for index, value in enumerate(values)
        ]
    }


def joined_sample(labels):
    class Sample:
        sample_id = "multiple:val:00000001"
        feature = {"candidates": [candidate(identifier=f"candidate_{i}") for i in range(5)]}
        label = label_record(labels)

    return Sample()


def test_stable_ids_and_inference_allowlist_are_split_safe():
    assert stable_sample_id("train", 7) != stable_sample_id("test", 7)
    assert_model_feature_names(["q", "critic_score"])
    for name in ("gt_box", "label", "jany_state", "angle_error_deg", "oracle_rank"):
        with pytest.raises(ValueError):
            assert_model_feature_names([name])
    with pytest.raises(ValueError):
        assert_inference_record_has_no_evaluation_fields(
            {"candidates": [{"features": {"q": 0.8}, "gt_iou": 0.4}]}
        )
    assert_inference_record_has_no_evaluation_fields(
        {"candidates": [{"features": {"q": {"value": 0.8}}}]}
    )


def test_grouped_calibration_and_oof_keep_sequences_intact():
    rows = [
        {
            "sample_id": f"multiple:train:{index:08d}",
            "frame_id": f"frame-{index}",
            "scene_id": f"frame-{index}",
            "sequence_id": f"seq-{index // 2}",
            "rgb_sha256": f"rgb-{index}",
            "depth_sha256": f"depth-{index}",
            "rgbd_content_sha256": f"hash-{index}",
        }
        for index in range(20)
    ]
    partitioned = grouped_partition(
        rows,
        group_key="sequence_id",
        heldout_fraction=0.2,
        seed=1,
        fit_name="train",
        heldout_name="calibration",
    )
    by_sequence = {}
    for row in partitioned:
        by_sequence.setdefault(row["sequence_id"], set()).add(
            row["development_partition"]
        )
    assert all(len(values) == 1 for values in by_sequence.values())
    train = [row for row in partitioned if row["development_partition"] == "train"]
    folded = assign_group_folds(train, group_key="sequence_id", folds=3, seed=2)
    fold_by_sequence = {}
    for row in folded:
        fold_by_sequence.setdefault(row["sequence_id"], set()).add(row["oof_fold"])
    assert all(len(values) == 1 for values in fold_by_sequence.values())


def test_oof_provenance_rejects_checkpoint_that_saw_heldout_fold():
    valid = [
        {
            "sample_id": "multiple:train:00000001",
            "heldout_fold": 1,
            "checkpoints": [
                {"fit_folds": [0, 2]},
                {"fit_folds": [0, 2]},
                {"fit_folds": [0, 2]},
            ],
        }
    ]
    validate_oof_provenance_records(
        valid, checkpoint_field="checkpoints", expected_checkpoints=3
    )
    invalid = [{**valid[0], "checkpoints": [{"fit_folds": [0, 1, 2]}] * 3}]
    with pytest.raises(AssertionError):
        validate_oof_provenance_records(
            invalid, checkpoint_field="checkpoints", expected_checkpoints=3
        )


def test_split_audit_detects_frame_and_hash_overlap():
    base = {
        "sample_id": "a",
        "frame_id": "frame",
        "scene_id": "frame",
        "sequence_id": "seq",
        "rgb_sha256": "rgb",
        "depth_sha256": "depth",
        "rgbd_content_sha256": "hash",
    }
    audit = audit_partitions(
        {"train": [base], "validation": [{**base, "sample_id": "b"}]}
    )
    assert not audit["required_zero_overlap_passed"]
    assert audit["pairwise_overlap"]["train__validation"]["frame_id_overlap"] == 1


def test_candidate_aligned_crop_geometry_axial_symmetry_and_boundaries():
    height, width = 480, 640
    rows, cols = np.indices((height, width))
    rgb = np.stack(
        (cols / width, rows / height, np.zeros_like(rows)), axis=-1
    ).astype(np.float32)
    depth = np.ones((height, width), dtype=np.float32)
    mask = np.ones_like(depth)
    quality = np.float32(cols / width)
    sin_map = np.zeros_like(depth)
    cos_map = np.ones_like(depth)
    width_map = np.full_like(depth, 0.8)
    first = candidate(cx=10, cy=10, angle=25)
    second = candidate(cx=10, cy=10, angle=205)
    first_crop, first_meta = build_aligned_crop(
        first,
        rgb=rgb,
        depth_m=depth,
        mask_probability=mask,
        quality=quality,
        sin_2theta=sin_map,
        cos_2theta=cos_map,
        width_probability=width_map,
        output_size=32,
    )
    second_crop, _ = build_aligned_crop(
        second,
        rgb=rgb,
        depth_m=depth,
        mask_probability=mask,
        quality=quality,
        sin_2theta=sin_map,
        cos_2theta=cos_map,
        width_probability=width_map,
        output_size=32,
    )
    assert first_crop.shape == (len(CROP_CHANNELS), 32, 32)
    assert np.array_equal(first_crop, second_crop)
    assert first_meta["depth_valid_fraction"] < 1.0
    assert first_crop[CROP_CHANNELS.index("left_finger_template")].sum() > 0
    assert first_crop[CROP_CHANNELS.index("contact_template")].sum() > 0


def test_crop_rotation_sign_and_width_change_sampling_extent():
    horizontal_x, horizontal_y = candidate_sampling_grid(candidate(angle=0, width=80), 9)
    vertical_x, vertical_y = candidate_sampling_grid(candidate(angle=90, width=80), 9)
    assert np.ptp(horizontal_x[4]) > np.ptp(horizontal_y[:, 4])
    assert np.ptp(vertical_y[4]) > np.ptp(vertical_x[:, 4])
    wider_x, _ = candidate_sampling_grid(candidate(angle=0, width=120), 9)
    assert np.ptp(wider_x[4]) > np.ptp(horizontal_x[4])


def test_missing_depth_is_neutral_and_finite():
    shape = (40, 60)
    crop, meta = build_aligned_crop(
        candidate(cx=30, cy=20, width=20),
        rgb=np.zeros((*shape, 3), dtype=np.uint8),
        depth_m=None,
        mask_probability=np.zeros(shape),
        quality=np.ones(shape),
        sin_2theta=np.zeros(shape),
        cos_2theta=np.ones(shape),
        width_probability=np.ones(shape),
        output_size=16,
    )
    assert np.isfinite(crop).all()
    assert not meta["depth_available"]
    assert crop[CROP_CHANNELS.index("depth_valid")].sum() == 0


def test_critic_minibatches_keep_all_five_candidates_together():
    sample_index = np.repeat(np.arange(7), 5)
    emitted = list(
        grouped_candidate_batches(
            sample_index,
            12,
            shuffle=True,
            seed=9,
        )
    )
    assert sorted(np.concatenate(emitted).tolist()) == list(
        range(len(sample_index))
    )
    for sample in range(7):
        locations = [
            batch_index
            for batch_index, batch in enumerate(emitted)
            if np.any(sample_index[batch] == sample)
        ]
        assert len(locations) == 1
        batch = emitted[locations[0]]
        assert np.sum(sample_index[batch] == sample) == 5


def test_vectorized_critic_pairwise_loss_is_finite_and_differentiable():
    logits = torch.tensor(
        [0.2, -0.1, 0.4, 0.0, 0.3, -0.3, 0.5, 0.1, -0.2, 0.4],
        requires_grad=True,
    )
    labels = torch.tensor([1, 0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
    samples = torch.tensor([4] * 5 + [9] * 5)
    quality = torch.linspace(0.9, 0.1, 10)
    loss = critic_loss(
        logits,
        labels,
        samples,
        quality,
        positive_weight=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_latent_roi_coordinate_mapping_and_pooling():
    feature = torch.arange(1 * 1 * 4 * 8, dtype=torch.float32).reshape(1, 1, 4, 8)
    grasp = candidate(cx=320, cy=240, angle=0, width=80)
    grid = candidate_roi_grid(
        grasp,
        image_shape=(480, 640),
        feature_shape=(4, 8),
        roi_size=5,
    )
    assert grid.shape == (1, 5, 5, 2)
    pooled = pool_candidate_rois(
        feature, [[grasp]], image_shapes=[(480, 640)], roi_size=5
    )
    assert pooled.shape == (1, 1, 2)
    assert pooled[0, 0, 1] >= pooled[0, 0, 0]


def test_latent_hook_is_compared_against_a_real_unhooked_forward():
    class TinyCROG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.neck = torch.nn.Conv2d(1, 2, 1)
            self.decoder = torch.nn.Conv2d(2, 2, 1)

        def forward(self, value):
            output = self.decoder(self.neck(value))
            return [output], None

    model = TinyCROG().eval()
    inputs = (torch.ones(1, 1, 4, 4),)
    with CROGLatentCapture(model) as capture:
        _assert_hook_non_mutating(model, inputs, capture)
        pre, post = capture.feature_maps()
    assert pre.shape == (1, 2, 4, 4)
    assert post.shape == (1, 2, 4, 4)


def test_residual_alpha_zero_is_exact_q_only():
    torch.manual_seed(2)
    model = LatentResidualRanker(8).eval()
    features = torch.randn(2, 5, 8)
    q = torch.rand(2, 5) * 0.8 + 0.1
    score, residual = model(features, q, alpha=0.0)
    assert torch.equal(score, quality_logit(q))
    recomposed = residual_scores(
        residual.detach().numpy(), q.detach().numpy(), alpha=1.0
    )
    with torch.no_grad():
        direct, _ = model(features, q, alpha=1.0)
    assert np.allclose(recomposed, direct.numpy(), atol=1e-6)


def test_temperature_calibration_does_not_change_ranking():
    probabilities = np.asarray([[0.8, 0.6, 0.2], [0.3, 0.7, 0.4]])
    labels = np.asarray([[1, 0, 0], [0, 1, 1]])
    fitted = fit_temperature(probabilities, labels)
    calibrated = apply_temperature(
        probabilities, fitted["temperature"]
    )
    assert np.array_equal(
        np.argsort(-probabilities, axis=1),
        np.argsort(-calibrated, axis=1),
    )
    assert not fitted["ranking_changed"]


def test_setrank_multiple_positive_no_positive_and_permutation_equivariance():
    torch.manual_seed(4)
    model = ResidualSetRank(6).eval()
    tokens = torch.randn(2, 5, 6)
    q = torch.rand(2, 5) * 0.8 + 0.1
    mask = torch.ones(2, 5, dtype=torch.bool)
    labels = torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    outputs = model(tokens, q, candidate_mask=mask)
    loss = setrank_loss(outputs[0], outputs[1], outputs[2], labels, mask)
    assert torch.isfinite(loss)
    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted = model(tokens[:, permutation], q[:, permutation], candidate_mask=mask)
    assert torch.allclose(outputs[0][:, permutation], permuted[0], atol=1e-6)
    assert torch.allclose(outputs[1][:, permutation], permuted[1], atol=1e-6)


def test_gate_three_class_labels_no_gain_and_tie_break():
    assert gate_outcome_labels(label_record([False, True, False, False, False])).tolist() == [0, 2, 2, 2]
    assert gate_outcome_labels(label_record([True, False, True, True, True])).tolist() == [1, 2, 2, 2]
    sample = joined_sample([False, True, True, False, False])
    probabilities = np.asarray(
        [[0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.1, 0.4, 0.5], [0.1, 0.4, 0.5]]
    )
    held = select_with_gate(sample, probabilities, harm_cost=3, threshold=0.2)
    assert held["selected_index"] == 0
    switched = select_with_gate(sample, probabilities, harm_cost=1, threshold=0.2)
    assert switched["selected_index"] == 1


def test_gate_inference_builds_pairs_without_reading_labels():
    class LabelFreeSample:
        sample_id = "multiple:test:00000001"
        feature = {
            "candidates": [
                candidate(identifier=f"candidate_{index}")
                for index in range(5)
            ]
        }

        @property
        def label(self):
            raise AssertionError("inference attempted to access GT labels")

    values, locations = _stack_inference_pairs([LabelFreeSample()])
    assert values.shape[0] == 4
    assert locations == [(0, 1), (0, 2), (0, 3), (0, 4)]


def test_prediction_probabilities_are_allowed_but_evaluation_fields_are_not(
    tmp_path,
):
    sample = joined_sample([False, True, False, False, False])
    base = {
        "sample_id": sample.sample_id,
        "candidate_order": [
            f"candidate_{index}" for index in range(5)
        ],
        "candidate_correctness_probabilities": [0.4] * 5,
    }
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(json.dumps(base) + "\n", encoding="utf-8")
    rankings, probabilities = load_prediction_rankings(
        prediction, [sample]
    )
    assert rankings[sample.sample_id][0] == "candidate_0"
    assert probabilities[sample.sample_id].shape == (5,)
    leaked = tmp_path / "leaked.jsonl"
    leaked.write_text(
        json.dumps({**base, "candidate_correct": True}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_prediction_rankings(leaked, [sample])


def test_npz_score_export_is_label_free_and_stably_ranked(tmp_path):
    sample = joined_sample([False, True, False, False, False])
    scores_path = tmp_path / "scores.npz"
    np.savez_compressed(
        scores_path,
        sample_ids=np.asarray([sample.sample_id]),
        scores=np.asarray([[0.1, 0.9, 0.2, 0.2, 0.0]], dtype=np.float32),
    )
    output = tmp_path / "predictions.jsonl"
    result = export_npz_rankings(
        samples=[sample],
        npz_path=scores_path,
        output_path=output,
        method="critic_ablation",
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["candidate_order"][0] == "candidate_1"
    assert record["candidate_order"][1:3] == ["candidate_2", "candidate_3"]
    assert "candidate_correctness_probabilities" in record
    assert result["sample_count"] == 1


def test_perturbation_statistics_do_not_create_candidates():
    before = [candidate(identifier=f"candidate_{i}") for i in range(5)]
    checksum = before[0]["candidate_checksum"]
    variants = perturbation_geometries(before[0])
    assert len(variants) == 17
    assert before[0]["candidate_checksum"] == checksum
    assert before[0]["cx"] == 320.0
    stats = stability_statistics([0.8, 0.7, np.nan], kappa=1.0)
    assert stats["valid_fraction"] == pytest.approx(2 / 3)
    assert len(before) == 5
    assert set(PERTURBATIONS) == {"center_px", "angle_deg", "width_fraction"}
    assert conservative_uncertainty_switch(
        0, 2, gain_lower_bound=0.3, threshold=0.2, consensus=2, required_consensus=3
    ) == 0


def test_perturbation_crop_scoring_caches_only_statistics():
    shape = (40, 60)
    inputs = {
        "rgb": np.zeros((*shape, 3), dtype=np.uint8),
        "depth_m": np.ones(shape, dtype=np.float32),
        "mask_probability": np.zeros(shape, dtype=np.float32),
        "quality": np.ones(shape, dtype=np.float32),
        "sin_2theta": np.zeros(shape, dtype=np.float32),
        "cos_2theta": np.ones(shape, dtype=np.float32),
        "width_probability": np.ones(shape, dtype=np.float32),
    }
    result = score_candidate_perturbations(
        candidate(cx=30, cy=20, width=20),
        scorer=lambda crops: crops[:, 6].mean(axis=(1, 2)),
        crop_inputs=inputs,
        output_size=16,
        kappa=1.0,
    )
    assert result["perturbation_count"] == 17
    assert result["statistics"]["valid_fraction"] == 1.0
    assert "crops" not in result
    assert "scores" not in result


def test_vlm_parser_mapping_cache_and_fallback(tmp_path):
    raw = json.dumps(
        {
            "best_candidate": 2,
            "candidates": [
                {
                    "id": index,
                    "target_alignment": 0.5,
                    "contact_quality": 0.5,
                    "width_fit": 0.5,
                    "collision_risk": 0.5,
                    "overall": 0.5,
                }
                for index in range(1, 6)
            ],
        }
    )
    parsed = parse_vlm_response(raw)
    mapping = shuffled_candidate_mapping(
        [f"candidate_{index}" for index in range(5)], sample_id="sample", seed=8
    )
    mapped = map_response_to_candidate_ids(parsed, mapping)
    assert mapped["best_candidate_id"] in mapping["candidate_to_display"]
    blocked = cached_review(
        provider=None,
        image_path=tmp_path / "unused.png",
        prompt="review",
        mapping=mapping,
        cache_path=tmp_path / "cache.json",
        parameters={},
    )
    assert blocked["fallback"] == "q_only"


def test_failure_stage_classification_is_explicit():
    assert (
        classify_failure_stage(
            {"mask_iou": 0.2}, original_correct=False, oracle=True
        )
        == "grounding_failure"
    )
    assert (
        classify_failure_stage(
            {"mask_iou": 0.8}, original_correct=False, oracle=False
        )
        == "candidate_set_failure"
    )
    assert (
        classify_failure_stage(
            {"mask_iou": 0.8}, original_correct=False, oracle=True
        )
        == "ranking_failure"
    )


def test_artifact_resume_is_idempotent(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    output = tmp_path / "artifact"
    kwargs = dict(
        output_dir=output,
        kind="test",
        repo_root=Path(__file__).parents[1],
        config={"x": 1},
        inputs=(source,),
        seed=1,
        device="cpu",
    )
    run = ArtifactRun(**kwargs)
    run.prepare()
    result = output / "result.txt"
    result.write_text("value", encoding="utf-8")
    first = run.complete(outputs=(result,), row_count=1, unique_ids=("id",))
    resumed = ArtifactRun(**kwargs, resume=True)
    second = resumed.prepare()
    assert resumed.is_complete
    assert first["result_sha256"] == second["result_sha256"]
    (output / "SUCCESS").unlink()
    repaired = ArtifactRun(**kwargs, resume=True)
    repaired.prepare()
    assert repaired.is_complete


def test_formal_test_claim_allows_same_lock_resume_but_not_second_run(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text('{"lock": 1}\n', encoding="utf-8")
    run = tmp_path / "formal_test"
    first = claim_test_once(run, lock)
    assert claim_test_once(run, lock, resume=True) == first
    with pytest.raises(FileExistsError):
        claim_test_once(run, lock)
    (run / "TEST_RUN_COMPLETE.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        claim_test_once(run, lock, resume=True)


def test_stability_replay_reads_frozen_export_batch_size(tmp_path):
    features = tmp_path / "features.jsonl"
    features.write_text("", encoding="utf-8")
    assert _frozen_export_batch_size(features) is None
    (tmp_path / "metadata.json").write_text(
        json.dumps({"output_config": {"batch_size": 16}}),
        encoding="utf-8",
    )
    assert _frozen_export_batch_size(features) == 16


def test_primary_base_ensemble_quantizes_critic_embeddings(
    monkeypatch, tmp_path
):
    from failure_analysis.reranking_v2 import inference

    arrays = {
        "sample_ids": np.asarray(["multiple:val:00000000"]),
        "crops": np.zeros((1, 5, 2, 4, 4), dtype=np.float16),
        "latent_post": np.zeros((1, 5, 3), dtype=np.float16),
        "q": np.full((1, 5), 0.5, dtype=np.float32),
    }
    embedding = np.full((5, 64), 0.123456, dtype=np.float32)
    monkeypatch.setattr(
        inference,
        "load_local_model_artifact",
        lambda path: {"path": str(path)},
    )
    monkeypatch.setattr(
        inference,
        "predict_critic_arrays",
        lambda artifact, crops, device: (
            np.zeros(5, dtype=np.float32),
            embedding.copy(),
        ),
    )
    monkeypatch.setattr(
        inference,
        "predict_latent_residual_arrays",
        lambda artifact, latent, q, alpha, device: (
            np.zeros((1, 5), dtype=np.float32),
            np.zeros((1, 5), dtype=np.float32),
        ),
    )

    result = inference.predict_base_ensemble(
        arrays=arrays,
        critic_models=[
            tmp_path / "critic31.pt",
            tmp_path / "critic37.pt",
            tmp_path / "critic43.pt",
        ],
        latent_models=[
            tmp_path / "latent31.pt",
            tmp_path / "latent37.pt",
            tmp_path / "latent43.pt",
        ],
        device="cpu",
    )

    assert result["critic_embeddings"].dtype == np.float16
    np.testing.assert_array_equal(
        result["critic_embeddings"][0, 0, 0],
        embedding[0].astype(np.float16),
    )


def test_report_artifact_has_title_chart_blocks_and_sorted_tables(tmp_path):
    def method(name, delta):
        return {
            "method": name,
            "sample_count": 10,
            "legacy_j1": 0.8 + delta / 100,
            "corrected_j1": 0.82 + delta / 100,
            "delta_j1_pp": delta,
            "oracle_at_5": 0.9,
            "recovered": int(delta > 0),
            "harmful": 0,
            "net_recovered": int(delta > 0),
            "neutral_switch": 0,
            "switch_coverage": 0.1,
            "outcome_changing_switch_precision": 1.0,
            "mrr_at_5": 0.85,
            "ndcg_at_5": 0.86,
            "candidate_brier": 0.2,
            "candidate_nll": 0.5,
            "candidate_ece": 0.1,
            "mcnemar_p_raw": 0.03,
            "mcnemar_p_holm": 0.04,
            "frame_ci_low_pp": 0.1 if delta > 0 else 0,
            "frame_ci_high_pp": 1.0 if delta > 0 else 0,
            "sequence_ci_low_pp": 0.0,
            "sequence_ci_high_pp": 1.2,
        }

    q_only = method("q_only", 0.0)
    primary = method("primary", 0.5)
    result = build_report_artifact(
        {
            "validation": {"methods": [q_only, primary]},
            "test": {"methods": [q_only, primary]},
            "primary_method": "primary",
            "primary_test_detail": {},
            "vlm_status": "blocked",
        },
        tmp_path / "report",
    )
    artifact = json.loads(Path(result["artifact_path"]).read_text())
    assert artifact["manifest"]["blocks"][0]["body"] == (
        "# CROG Re-ranking V2 results"
    )
    assert artifact["manifest"]["charts"]
    assert all(
        table.get("defaultSort")
        for table in artifact["manifest"]["tables"]
    )
    assert artifact["snapshot"]["status"] == "ready"


def test_independent_evaluator_comparison_detects_mismatch():
    primary = {
        "sample_count": 10,
        "q_only_j1": 0.8,
        "legacy_or_corrected_j1": 0.9,
        "delta_j1_percentage_points": 10.0,
        "oracle_at_5": 0.9,
        "recovered": 1,
        "harmful": 0,
        "net_recovered": 1,
        "neutral_switch": 0,
        "switch_coverage": 0.1,
        "mcnemar_exact_two_sided_pvalue": 1.0,
    }
    independent = {
        "sample_count": 10,
        "q_only_j1": 0.8,
        "selected_j1": 0.9,
        "delta_j1_percentage_points": 10.0,
        "oracle_at_5": 0.9,
        "recovered": 1,
        "harmful": 0,
        "net_recovered": 1,
        "neutral_switch": 0,
        "switch_coverage": 0.1,
        "mcnemar_exact_two_sided_pvalue": 1.0,
    }
    assert_matches_primary(primary, independent)
    independent["selected_j1"] = 0.8
    with pytest.raises(AssertionError):
        assert_matches_primary(primary, independent)


def test_v1_frozen_baseline_metadata_still_matches():
    root = Path(__file__).parents[1]
    summary = json.loads(
        (
            root
            / "failure_analysis/reranking_outputs/full_test_17749_v1/eval_legacy/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["sample_count"] == 17749
    assert summary["candidate_count"] == 88745
    assert summary["original_success_count"] == 14768
    assert summary["oracle_success_count"] == 16129
