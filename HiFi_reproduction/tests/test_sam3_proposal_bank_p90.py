from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from scripts.extract_proposal_features import (
    FEATURE_EXTRACTOR_SEMANTIC_SHA256,
    _stream_feature_table,
)
from scripts.generate_sam3_proposal_bank import (
    _rebuild_split_manifest,
    _select_frame_shard,
)
from scripts.generate_stage2_refinements import (
    _rank_candidates,
    _row_group_by_sample,
)
from scripts.evaluate_locked_sam3_p90_masks import _sample_row_group
from src.segmentation.conservative_mask_gate import (
    GateThresholds,
    add_gate_evidence,
    apply_gate,
    proposed_alternatives,
)
from src.segmentation.depth_mask_features import depth_mask_features
from src.segmentation.p90_selector import (
    FeatureEncoder,
    inverse_candidate_weights,
    p90_labels,
    select_scored_candidates,
)
from src.segmentation.proposal_deduplication import deduplicate_candidates
from src.segmentation.proposal_oracle import summarize_oracle
from src.segmentation.proposal_oracle import _packed_candidate_ious
from src.segmentation.proposal_features import (
    _independent_provenance_count,
    extract_candidate_features,
)
from src.segmentation.proposal_statistics import (
    clustered_bootstrap,
    clustered_bootstrap_values,
    holm_adjust,
    paired_transitions,
    wilson_interval,
)
from src.segmentation.proposal_types import (
    ProposalCandidate,
    load_candidate_masks_npz,
    mask_sha256,
    reference_candidate_ids_from_provenance,
)
from src.segmentation.query_semantics import parse_query
from src.segmentation.sam3_embedding_cache import EmbeddingCacheKey, Sam3EmbeddingCache
from src.segmentation.sam3_text_proposals import OfficialSam3TextProposalGenerator
from src.segmentation.sam3_proposal_generator import (
    build_component_prompt_specs,
    build_hifi_candidates,
    build_text_prompt_specs,
    build_visual_prompt_specs,
    write_proposal_bundle,
)
from src.segmentation.second_stage_refiner import build_stage2_prompts
from src.segmentation.selector_dataset import (
    CandidateDatasetPair,
    iter_joined_candidate_samples,
)
from src.segmentation.selector_out_of_core import (
    grouped_oof_predictions_out_of_core,
)
from src.segmentation.selective_sam3_vg.metrics import (
    boundary_fscore,
    boundary_fscores,
    evaluator_iou_float32,
)
from src.segmentation.selective_sam3_vg.io import resize_binary_mask
from src.segmentation.spatial_relation_features import (
    location_ranks,
    pairwise_relation_features,
    relation_score,
)


REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SHA_A = "a" * 64
SHA_B = "b" * 64


def test_feature_stream_unions_sparse_relation_columns(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    output = tmp_path / "combined.parquet"
    pd.DataFrame(
        {"sample_id": ["plain"], "candidate_id": ["c0"], "score": [0.5]}
    ).to_parquet(first, index=False)
    pd.DataFrame(
        {
            "sample_id": ["relational"],
            "candidate_id": ["c1"],
            "score": [0.7],
            "relation_left_score": [0.9],
        }
    ).to_parquet(second, index=False)

    rows, columns, statistics = _stream_feature_table([first, second], output)
    combined = pd.read_parquet(output)

    assert rows == 2
    assert list(combined["sample_id"]) == ["plain", "relational"]
    assert np.isnan(combined.loc[0, "relation_left_score"])
    assert combined.loc[1, "relation_left_score"] == pytest.approx(0.9)
    assert "relation_left_score" in columns
    assert statistics["relation_left_score"]["count"] == 1
    assert statistics["relation_left_score"]["missing"] == 1
    assert FEATURE_EXTRACTOR_SEMANTIC_SHA256 == (
        "575287c08985eefcf0a0a07bc91bc16f52f17a68bd9bde10fe5fc6ef9f2fb3a4"
    )


def test_text_topk_before_resize_is_exactly_equivalent_for_retained_masks():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(42)
    outputs = SimpleNamespace(
        pred_masks=torch.randn(3, 11, 7, 9, generator=generator),
        pred_boxes=torch.randn(3, 11, 4, generator=generator),
    )
    scores = torch.rand(3, 11, generator=generator)
    keep = 4
    output_size = (31, 37)
    order, optimized, selected_boxes = (
        OfficialSam3TextProposalGenerator._select_and_resize_outputs(
            outputs,
            scores,
            maximum_instances_per_prompt=keep,
            output_size=output_size,
        )
    )
    reference_order = torch.argsort(scores, dim=1, descending=True, stable=True)[
        :, :keep
    ]
    reference = torch.nn.functional.interpolate(
        outputs.pred_masks.sigmoid(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    for batch_index in range(scores.shape[0]):
        for rank, raw_index in enumerate(reference_order[batch_index].tolist()):
            assert torch.allclose(
                optimized[batch_index, rank],
                reference[batch_index, raw_index],
                rtol=0.0,
                atol=2e-7,
            )
            assert torch.equal(
                selected_boxes[batch_index, rank], outputs.pred_boxes[batch_index, raw_index]
            )
            for threshold in (0.35, 0.5, 0.65):
                assert torch.equal(
                    optimized[batch_index, rank] > threshold,
                    reference[batch_index, raw_index] > threshold,
                )
    assert torch.equal(order, reference_order)


def test_candidate_geometry_cache_preserves_four_connected_index_semantics():
    from scipy import ndimage

    mask = np.zeros((17, 19), dtype=bool)
    mask[2:6, 3:8] = True
    mask[10:14, 12:17] = True
    mask[6, 8] = True  # diagonal-only contact must remain a separate component
    candidate = ProposalCandidate("sample", "AUTOMATIC", "raw", mask)
    expected_components = int(ndimage.label(mask)[1])
    first = candidate.to_index_record()
    second = candidate.to_index_record()
    assert first == second
    assert first["connected_components"] == expected_components
    assert first["area"] == int(mask.sum())
    assert json.loads(first["box_json"]) == [3.0, 2.0, 16.0, 13.0]
    assert first["mask_sha256"] == mask_sha256(mask)
    assert set(candidate._geometry_cache) == {
        "area",
        "connected_components",
        "mask_box",
        "mask_sha256",
    }


def test_frame_sharding_is_deterministic_and_never_splits_queries_from_one_frame():
    rows = {
        "frame_0": ["q0", "q1"],
        "frame_1": ["q2"],
        "frame_2": ["q3", "q4", "q5"],
        "frame_3": ["q6"],
    }
    shard_0 = _select_frame_shard(rows, 2, 0)
    shard_1 = _select_frame_shard(rows, 2, 1)
    assert list(shard_0) == ["frame_0", "frame_2"]
    assert list(shard_1) == ["frame_1", "frame_3"]
    assert set(shard_0).isdisjoint(shard_1)
    assert {**shard_0, **shard_1} == rows


def test_reference_role_survives_target_reference_mask_deduplication():
    index = pd.DataFrame(
        [
            {
                "candidate_id": "shared_mask",
                "source_family": "TEXT_TARGET_CATEGORY",
            },
            {"candidate_id": "ordinary", "source_family": "AUTOMATIC"},
        ]
    )
    provenance = {
        "shared_mask": [
            {"sample_prompt_id": "T1"},
            {
                "sample_prompt_id": "R1",
                "sample_source_variant": "R1_reference_category",
            },
        ],
        "ordinary": [{"source_family": "AUTOMATIC"}],
    }
    assert reference_candidate_ids_from_provenance(index, provenance) == {
        "shared_mask"
    }
    assert reference_candidate_ids_from_provenance(
        index.set_index("candidate_id"), provenance
    ) == {"shared_mask"}


def test_source_consensus_counts_independent_prompts_not_threshold_duplicates():
    provenance = [
        {"text": "cup", "instance_threshold": 0.1, "mask_threshold": 0.4},
        {"text": "cup", "instance_threshold": 0.3, "mask_threshold": 0.5},
        {"text": "red cup", "instance_threshold": 0.1, "mask_threshold": 0.4},
        {"prompt_id": "V0_expand=0pct"},
        {"prompt_id": "V0_expand=0pct", "mask_threshold": 0.6},
    ]
    assert _independent_provenance_count(provenance) == 3


def test_invariant_feature_cache_is_numerically_identical():
    hifi = np.zeros((32, 32), dtype=bool)
    hifi[8:22, 8:22] = True
    alternative = np.zeros_like(hifi)
    alternative[7:21, 9:23] = True
    candidates = [
        ProposalCandidate("sample", "HIFI_ORIGINAL", "h0", hifi),
        ProposalCandidate("sample", "AUTOMATIC", "auto", alternative),
    ]
    index = pd.DataFrame([candidate.to_index_record() for candidate in candidates])
    masks = {str(candidate.candidate_id): candidate.mask for candidate in candidates}
    kwargs = {
        "hifi_mask": hifi,
        "hifi_probability": np.where(hifi, 0.9, 0.05).astype(np.float32),
        "rgb": np.full((32, 32, 3), 127, dtype=np.uint8),
        "depth_m": np.full((32, 32), 0.8, dtype=np.float32),
        "intrinsics": {"fx": 30.0, "fy": 30.0, "cx": 15.5, "cy": 15.5},
        "semantics": parse_query("Pick the red marker"),
    }
    uncached = extract_candidate_features(index, masks, **kwargs)
    cache: dict[str, dict] = {}
    first = extract_candidate_features(index, masks, invariant_cache=cache, **kwargs)
    second = extract_candidate_features(index, masks, invariant_cache=cache, **kwargs)
    pd.testing.assert_frame_equal(uncached, first)
    pd.testing.assert_frame_equal(first, second)
    assert len(cache) == 2


@pytest.mark.parametrize(
    ("query", "category", "query_type", "relation", "location"),
    [
        ("Pick the marker", "marker", "name", None, None),
        ("Pick the red marker", "marker", "attribute", None, None),
        ("Pick the leftmost red marker", "marker", "mixed", None, "leftmost"),
        ("Pick the marker behind the white bowl", "marker", "relation", "behind", None),
        (
            "Pick the tissues that is on the front left side of the cube keenex box",
            "kleenex",
            "relation",
            "front_left",
            None,
        ),
    ],
)
def test_query_parser_template_families(query, category, query_type, relation, location):
    parsed = parse_query(query)
    assert parsed.target_category == category
    assert parsed.query_type == query_type
    assert parsed.pairwise_relation == relation
    assert parsed.absolute_location == location
    assert parsed.uses_answer_instance is False


def test_query_parser_has_no_answer_input_and_preserves_same_category_reference():
    assert set(inspect.signature(parse_query).parameters) == {"query"}
    parsed = parse_query(
        "Pick the tissues that is on the front left side of the cube keenex box"
    )
    assert parsed.target_instance_phrase is None
    assert parsed.target_category == parsed.reference_category == "kleenex"
    prompts = build_text_prompt_specs(parsed)
    reference = [item for item in prompts if item.prompt_id == "R1"]
    assert len(reference) == 1
    assert reference[0].text == "kleenex"
    assert reference[0].eligible_final is False


def test_hifi_threshold_candidates_use_model_space_threshold_then_nearest_resize():
    probability = np.zeros((352, 352), dtype=np.float32)
    probability[100:200, 100:200] = 0.55
    candidates, original, native_probability = build_hifi_candidates(
        "sample", probability, (480, 640), SHA_A, [0.4, 0.5, 0.6]
    )
    assert len(candidates) == 3
    assert candidates[1].source_family == "HIFI_ORIGINAL"
    assert np.array_equal(candidates[1].mask, original)
    assert native_probability.shape == (480, 640)
    assert candidates[0].area == candidates[1].area > candidates[2].area


def _coarse_example():
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:55, 30:70] = True
    mask[5:10, 5:10] = True
    probability = np.full(mask.shape, 0.05, dtype=np.float32)
    probability[mask] = 0.9
    return mask, probability


def test_visual_and_component_prompts_are_gt_free_and_positive_points_are_inside():
    mask, probability = _coarse_example()
    visual, metadata = build_visual_prompt_specs(
        mask,
        probability,
        {
            "box_expansion_ratios": [0.0, 0.05],
            "positive_point_counts": [1, 3, 5],
            "core_minimum_probability": 0.6,
            "core_probability_quantile": 0.7,
            "point_separation_px": 4.0,
        },
    )
    assert len(visual) == 10
    assert metadata["status"] == "READY"
    for prompt in visual:
        assert all(mask[int(y), int(x)] for x, y in prompt.positive_points_xy)
    components, records = build_component_prompt_specs(
        mask,
        probability,
        {
            "minimum_area_px": 4,
            "minimum_probability_mass_fraction": 0.001,
            "maximum_components": 8,
        },
    )
    assert len(records) == 2
    assert len(components) == 6


def test_deduplication_is_strict_deterministic_and_preserves_provenance():
    mask = np.zeros((32, 32), dtype=bool)
    mask[5:20, 6:22] = True
    a = ProposalCandidate("s", "A", "a", mask, provenance=[{"source": "a"}])
    b = ProposalCandidate("s", "B", "b", mask.copy(), provenance=[{"source": "b"}])
    c_mask = mask.copy()
    c_mask[5, 6] = False
    c = ProposalCandidate("s", "C", "c", c_mask, provenance=[{"source": "c"}])
    first = deduplicate_candidates([a, b, c], iou_threshold=0.995)
    assert len(first) == 1
    assert len(first[0].provenance) == 3
    assert len(first[0].deduplication_parent_ids) == 2
    assert mask_sha256(mask) == mask_sha256(mask.copy())


def test_deduplication_keeps_hifi_original_as_canonical_fallback():
    mask = np.ones((8, 8), dtype=bool)
    threshold = ProposalCandidate("s", "HIFI_THRESHOLD", "H1_0.30", mask)
    original = ProposalCandidate("s", "HIFI_ORIGINAL", "H0", mask.copy())
    result = deduplicate_candidates([threshold, original], iou_threshold=0.995)
    assert len(result) == 1
    assert result[0].source_family == "HIFI_ORIGINAL"
    assert result[0].source_variant == "H0"
    assert len(result[0].provenance) == 2


def test_embedding_cache_key_changes_for_every_contract_field(tmp_path):
    base = EmbeddingCacheKey(SHA_A, REVISION, SHA_B, 1008, "float32", "backend")
    assert base.digest == EmbeddingCacheKey(
        SHA_A, REVISION, SHA_B, 1008, "float32", "backend"
    ).digest
    assert base.digest != EmbeddingCacheKey(
        SHA_A, REVISION, SHA_B, 560, "float32", "backend"
    ).digest
    cache = Sam3EmbeddingCache(tmp_path)
    assert cache.exists(base) is False


def test_tracker_embedding_cache_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    key = EmbeddingCacheKey(SHA_A, REVISION, SHA_B, 1008, "float32", "tracker")
    values = [torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)]
    cache = Sam3EmbeddingCache(tmp_path)
    cache.save_tracker(key, values)
    loaded = cache.load_tracker(key)
    assert cache.exists(key)
    assert len(loaded) == 1
    assert torch.equal(values[0], loaded[0])
    pruning = cache.prune_to_max_bytes(1)
    assert pruning["removed_entries"] == 1
    assert cache.exists(key) is False


def test_depth_location_and_pairwise_relation_features_include_missingness():
    left = np.zeros((20, 30), dtype=bool)
    right = np.zeros_like(left)
    left[5:12, 2:8] = True
    right[6:13, 20:27] = True
    depth = np.full(left.shape, 1.0, dtype=np.float32)
    depth[right] = 1.2
    features = depth_mask_features(left, depth, reference_mask=right)
    assert features["depth_features_valid"] is True
    assert features["depth_features_missing"] is False
    ranks = location_ranks([left, right], [1.0, 1.2], left.shape)
    assert ranks[0]["leftmost_score"] == 1.0
    assert ranks[0]["closest_score"] == 1.0
    pair = pairwise_relation_features(
        left, right, target_depth=1.0, reference_depth=1.2, image_shape=left.shape
    )
    assert relation_score(pair, "left") > relation_score(pair, "right")


def test_oracle_uses_strict_p90_and_excludes_reference_candidates():
    rows = pd.DataFrame(
        [
            {"sample_id": "a", "candidate_id": "h0", "source_family": "HIFI_ORIGINAL", "source_variant": "H0", "eligible_final": True, "candidate_iou": 0.90},
            {"sample_id": "a", "candidate_id": "t", "source_family": "TEXT_TARGET_CATEGORY", "source_variant": "T1", "eligible_final": True, "candidate_iou": 0.91},
            {"sample_id": "b", "candidate_id": "h0", "source_family": "HIFI_ORIGINAL", "source_variant": "H0", "eligible_final": True, "candidate_iou": 0.90},
            {"sample_id": "b", "candidate_id": "r", "source_family": "REFERENCE_TEXT_CATEGORY", "source_variant": "R1", "eligible_final": False, "candidate_iou": 0.99},
        ]
    )
    summary, per_sample, _ = summarize_oracle(rows)
    assert summary["expanded_proposal_bank_oracle_P@90_numerator"] == 1
    assert summary["expanded_proposal_bank_oracle_P@90_denominator"] == 2
    assert per_sample.loc[per_sample.sample_id == "b", "best_candidate_iou"].item() == 0.90


def test_proposal_bundle_is_packed_atomic_and_checksum_verified(tmp_path):
    mask = np.zeros((24, 32), dtype=bool)
    mask[4:18, 7:21] = True
    candidate = ProposalCandidate("s", "HIFI_ORIGINAL", "H0", mask)
    output = tmp_path / "s"
    unique = write_proposal_bundle(
        output,
        Image.new("RGB", (32, 24), "gray"),
        [candidate],
        mask,
        {"uses_ground_truth": False},
        {"seconds": 0.0},
        deduplication_iou=0.995,
    )
    assert len(unique) == 1
    status = json.loads((output / "terminal_status.json").read_text())
    assert status["status"] == "COMPLETE"
    assert "packbits" in status["candidate_mask_storage"]
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["bundle_write_seconds"] >= 0.0
    assert runtime["sample_seconds_total"] == pytest.approx(
        runtime["bundle_write_seconds"]
    )
    restored = load_candidate_masks_npz(output / "candidate_masks.npz")[
        str(candidate.candidate_id)
    ]
    assert np.array_equal(restored, mask)
    target = resize_binary_mask(mask, (13, 17))
    packed_iou = _packed_candidate_ious(output / "candidate_masks.npz", target)
    expected_iou = evaluator_iou_float32(
        resize_binary_mask(mask, target.shape), target
    )
    assert packed_iou[str(candidate.candidate_id)] == expected_iou


def test_parallel_bundle_writes_are_semantically_identical(tmp_path):
    mask = np.zeros((24, 32), dtype=bool)
    mask[4:18, 7:21] = True
    alternative = np.zeros_like(mask)
    alternative[5:19, 8:22] = True

    def write(name: str):
        candidates = [
            ProposalCandidate(name, "HIFI_ORIGINAL", "H0", mask.copy()),
            ProposalCandidate(name, "AUTOMATIC", "auto", alternative.copy()),
        ]
        output = tmp_path / name
        write_proposal_bundle(
            output,
            Image.new("RGB", (32, 24), "gray"),
            candidates,
            mask,
            {"uses_ground_truth": False},
            {"sample_seconds_before_write": 0.0},
            deduplication_iou=0.995,
        )
        return output

    serial = write("serial")
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = [pool.submit(write, f"parallel_{index}") for index in range(2)]
        parallel = [future.result() for future in parallel]
    serial_masks = sorted(
        (value.sum(), value.tobytes())
        for value in load_candidate_masks_npz(serial / "candidate_masks.npz").values()
    )
    for output in parallel:
        status = json.loads((output / "terminal_status.json").read_text())
        assert status["status"] == "COMPLETE"
        actual_masks = sorted(
            (value.sum(), value.tobytes())
            for value in load_candidate_masks_npz(output / "candidate_masks.npz").values()
        )
        assert actual_masks == serial_masks
        assert Image.open(output / "proposal_grid.png").size == Image.open(
            serial / "proposal_grid.png"
        ).size


def test_rebuilt_manifest_preserves_end_to_end_shard_runtime(tmp_path):
    mask = np.ones((8, 10), dtype=bool)
    proposal_root = tmp_path / "proposals"
    write_proposal_bundle(
        proposal_root / "sample",
        Image.new("RGB", (10, 8), "gray"),
        [ProposalCandidate("sample", "HIFI_ORIGINAL", "H0", mask)],
        mask,
        {"uses_ground_truth": False},
        {"sample_seconds_before_write": 1.25},
        deduplication_iou=0.995,
    )
    canonical = tmp_path / "proposal_generation_val_manifest.parquet"
    pd.DataFrame(
        [{"sample_id": "sample", "runtime_seconds": 9.5}]
    ).to_parquet(
        tmp_path / "proposal_generation_val_manifest.shard-000-of-002.parquet",
        index=False,
    )
    rebuilt = _rebuild_split_manifest(
        [
            SimpleNamespace(
                sample_id="sample",
                scene_id="frame",
                raw={"source_rgb_sha256": "abc"},
            )
        ],
        proposal_root,
        canonical,
    )
    assert rebuilt["runtime_seconds"].item() == pytest.approx(9.5)


@pytest.mark.parametrize("schema", [1, 2])
def test_oracle_legacy_mask_archives_match_exact_evaluator(tmp_path, schema):
    masks = {
        "left": np.pad(np.ones((5, 7), dtype=bool), ((2, 3), (1, 4))),
        "right": np.pad(np.ones((4, 5), dtype=bool), ((4, 2), (6, 1))),
    }
    path = tmp_path / f"schema_{schema}.npz"
    if schema == 1:
        np.savez_compressed(path, **masks)
    else:
        np.savez_compressed(
            path,
            __mask_shape__=np.asarray(next(iter(masks.values())).shape, dtype=np.int32),
            __storage_schema__=np.asarray([2], dtype=np.int32),
            **{
                key: np.packbits(value.reshape(-1), bitorder="little")
                for key, value in masks.items()
            },
        )
    target = np.zeros((13, 17), dtype=bool)
    target[3:10, 2:12] = True
    actual = _packed_candidate_ious(path, target)
    expected = {
        key: evaluator_iou_float32(resize_binary_mask(mask, target.shape), target)
        for key, mask in masks.items()
    }
    assert actual == expected


def test_p90_label_is_strict_and_inverse_candidate_weighting_balances_samples():
    labels = p90_labels(np.asarray([0.90, np.nextafter(0.90, 1.0), 0.95]))
    assert labels.tolist() == [0, 1, 1]
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b", "b", "b"],
            "candidate_id": list("abcdef"),
        }
    )
    weights = inverse_candidate_weights(frame, np.asarray([0, 1, 0, 0, 1, 1]))
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(len(weights))


def test_feature_encoder_rejects_gt_columns_from_model_matrix():
    frame = pd.DataFrame(
        {
            "source_family": ["HIFI_ORIGINAL", "AUTOMATIC"],
            "query_type": ["name", "relation"],
            "score": [0.1, np.nan],
            "candidate_iou": [0.2, 0.9],
            "y90": [False, False],
        }
    )
    encoder = FeatureEncoder.fit(frame)
    assert "candidate_iou" not in encoder.numeric_columns
    assert "y90" not in encoder.numeric_columns
    transformed = encoder.transform(frame)
    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()


def test_feature_encoder_applies_training_defaults_to_sparse_sample_schema():
    training = pd.DataFrame(
        {
            "source_family": ["HIFI_ORIGINAL", "AUTOMATIC"],
            "query_type": ["name", "relation"],
            "score": [0.1, 0.9],
            "relation_behind_score": [0.2, 0.8],
        }
    )
    encoder = FeatureEncoder.fit(training)
    sparse_sample = training.drop(
        columns=["query_type", "relation_behind_score"]
    ).iloc[[0]].copy()

    transformed = encoder.transform(sparse_sample)

    assert transformed.shape == (1, len(encoder.feature_names))
    assert np.isfinite(transformed).all()
    behind_index = encoder.numeric_columns.index("relation_behind_score")
    assert transformed[0, behind_index] == pytest.approx(0.5)


def test_clustered_statistics_and_mcnemar_transitions_are_paired():
    frame = pd.DataFrame(
        {"frame_id": ["a", "a", "b", "b"], "value": [0.0, 1.0, 0.5, 0.5]}
    )
    low, high, values = clustered_bootstrap(
        frame,
        cluster_column="frame_id",
        statistic=lambda data: float(data["value"].mean()),
        replicates=100,
        seed=42,
    )
    assert len(values) == 100
    assert 0.0 <= low <= high <= 1.0
    transitions = paired_transitions(
        np.asarray([0.91, 0.10, 0.90]),
        np.asarray([0.80, 0.95, 0.91]),
        0.90,
    )
    assert transitions["recovered"] == 2
    assert transitions["harmed"] == 1
    assert wilson_interval(5, 10)[0] < 0.5 < wilson_interval(5, 10)[1]
    adjusted = holm_adjust([0.01, 0.04, 0.20])
    assert adjusted[0] == pytest.approx(0.03)
    fast_low, fast_high, fast_values = clustered_bootstrap_values(
        frame["value"].to_numpy(),
        frame["frame_id"].to_numpy(),
        replicates=100,
        seed=42,
    )
    assert len(fast_values) == 100
    assert 0.0 <= fast_low <= fast_high <= 1.0


def test_stage2_prompts_and_conservative_gate_keep_hifi_unless_all_evidence_passes():
    selected = np.zeros((32, 32), dtype=bool)
    selected[8:22, 8:22] = True
    competitor = np.zeros_like(selected)
    competitor[10:24, 20:30] = True
    reference = np.zeros_like(selected)
    reference[2:7, 2:7] = True
    depth_m = np.ones(selected.shape, dtype=np.float32)
    depth_m[23:27, 8:14] = 1.10
    hifi_probability = np.ones(selected.shape, dtype=np.float32)
    hifi_probability[8:12, 8:12] = 0.10
    batched_boundary = boundary_fscores([selected, competitor], selected)
    assert batched_boundary[0] == pytest.approx(
        boundary_fscore(selected, selected)
    )
    assert batched_boundary[1] == pytest.approx(
        boundary_fscore(competitor, selected)
    )
    visual, text, metadata = build_stage2_prompts(
        selected,
        [competitor],
        [(20.0, 10.0, 29.0, 23.0)],
        target_text="marker",
        target_attribute_text="red marker",
        reference_masks=[reference],
        reference_boxes=[(2.0, 2.0, 6.0, 6.0)],
        depth_m=depth_m,
        hifi_probability=hifi_probability,
    )
    assert any(prompt.input_mask is not None for prompt in visual)
    assert any(prompt.negative_points_xy for prompt in visual)
    prompt_ids = {prompt.prompt_id for prompt in visual}
    assert "S2C_mask_points_reference_negatives" in prompt_ids
    assert "S2C_mask_points_depth_negatives" in prompt_ids
    assert "S2C_mask_points_low_hifi_negatives" in prompt_ids
    assert [prompt.text for prompt in text] == ["marker", "red marker"]
    assert all(prompt.box_labels == (1, 0, 0) for prompt in text)
    assert metadata["competitor_boxes"] == [(20.0, 10.0, 29.0, 23.0)]
    assert metadata["reference_boxes"] == [(2.0, 2.0, 6.0, 6.0)]
    for key in (
        "reference_negative_points",
        "depth_discontinuity_negative_points",
        "low_hifi_probability_negative_points",
    ):
        assert metadata[key]
        assert all(0 <= x < 32 and 0 <= y < 32 for x, y in metadata[key])
    assert metadata["uses_ground_truth"] is False

    frame = pd.DataFrame(
        [
            {
                "sample_id": "s",
                "candidate_id": "hifi",
                "source_family": "STAGE2_HIFI_FALLBACK",
                "eligible_final": True,
                "candidate_iou": 0.80,
                "m0_p90_calibrated": 0.10,
                "m1_p90_calibrated": 0.10,
                "m2_predicted_iou": 0.80,
                "source_consensus_count": 1,
                "hifi_probability_mass_precision": 0.80,
                "depth_reliability": 1.0,
                "low_probability_expansion_fraction": 0.0,
                "connected_component_count": 1,
                "has_relation": False,
                "appearance_features_valid": True,
                "depth_features_valid": True,
                "deterministic_rule_score": 0.1,
                "hifi_candidate_iou": 1.0,
            },
            {
                "sample_id": "s",
                "candidate_id": "sam",
                "source_family": "STAGE2_TRACKER",
                "eligible_final": True,
                "candidate_iou": 0.95,
                "m0_p90_calibrated": 0.91,
                "m1_p90_calibrated": 0.92,
                "m2_predicted_iou": 0.94,
                "source_consensus_count": 2,
                "hifi_probability_mass_precision": 0.70,
                "depth_reliability": 0.95,
                "low_probability_expansion_fraction": 0.05,
                "connected_component_count": 1,
                "has_relation": False,
                "appearance_features_valid": True,
                "depth_features_valid": True,
                "deterministic_rule_score": 0.8,
                "hifi_candidate_iou": 0.8,
            },
        ]
    )
    evidence = add_gate_evidence(frame)
    proposed = proposed_alternatives(evidence, "F2_classifier_regressor")
    accepted = apply_gate(
        evidence,
        proposed,
        GateThresholds(
            minimum_p90_margin=0.10,
            minimum_iou_margin=0.05,
            minimum_source_consensus=2,
        ),
    )
    assert accepted.iloc[0]["candidate_id"] == "sam"
    kept = apply_gate(
        evidence,
        proposed,
        GateThresholds(minimum_p90_margin=0.90),
    )
    assert kept.iloc[0]["candidate_id"] == "hifi"


def test_final_classifier_and_two_head_selectors_are_distinct_on_probability_ties():
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s",
                "candidate_id": "a_classifier_tie_break",
                "source_family": "STAGE2_TRACKER",
                "eligible_final": True,
                "m1_p90_calibrated": 0.8,
                "m2_predicted_iou": 0.7,
            },
            {
                "sample_id": "s",
                "candidate_id": "z_regressor_tie_break",
                "source_family": "STAGE2_TRACKER",
                "eligible_final": True,
                "m1_p90_calibrated": 0.8,
                "m2_predicted_iou": 0.9,
            },
        ]
    )
    classifier = proposed_alternatives(frame, "F1_hgb_classifier")
    two_head = proposed_alternatives(frame, "F2_classifier_regressor")
    assert classifier.iloc[0]["candidate_id"] == "a_classifier_tie_break"
    assert two_head.iloc[0]["candidate_id"] == "z_regressor_tie_break"
    inference = frame.rename(
        columns={
            "m1_p90_calibrated": "selection_score",
            "m2_predicted_iou": "predicted_iou",
        }
    )
    inference_classifier = select_scored_candidates(inference, "M1_hgb_classifier")
    inference_ensemble = select_scored_candidates(inference, "M3_two_head_ensemble")
    assert inference_classifier.iloc[0]["candidate_id"] == "a_classifier_tie_break"
    assert inference_ensemble.iloc[0]["candidate_id"] == "z_regressor_tie_break"


def test_out_of_core_selector_scores_every_candidate_without_frame_leakage(tmp_path):
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.parquet"
    feature_writer = None
    label_writer = None
    expected_candidates = 0
    try:
        for sample_index in range(6):
            sample_id = f"sample_{sample_index}"
            split = "train"
            ious = (0.80, 0.96, 0.92, 0.40)
            sources = (
                "HIFI_ORIGINAL",
                "AUTOMATIC",
                "TEXT_FULL_QUERY",
                "VISUAL_BOX",
            )
            features = pd.DataFrame(
                {
                    "sample_id": [sample_id] * 4,
                    "candidate_id": [f"{sample_id}_{index}" for index in range(4)],
                    "split": [split] * 4,
                    "frame_id": [f"frame_{sample_index}"] * 4,
                    "scene_id": [f"scene_{sample_index // 2}"] * 4,
                    "source_family": sources,
                    "eligible_final": [True] * 4,
                    "query_type": ["name"] * 4,
                    "target_category": ["cup"] * 4,
                    "source_consensus_count": [1, 3, 2, 1],
                    "hifi_candidate_iou": [1.0, 0.7, 0.6, 0.2],
                    "sam_score": [0.0, 0.9, 0.8, 0.5],
                    "feature_score": [0.1, 0.95, 0.85, 0.2],
                }
            )
            labels = pd.DataFrame(
                {
                    "sample_id": features["sample_id"],
                    "candidate_id": features["candidate_id"],
                    "candidate_iou": ious,
                    "continuous_iou": ious,
                    "y70": [value > 0.70 for value in ious],
                    "y80": [value > 0.80 for value in ious],
                    "y90": [value > 0.90 for value in ious],
                }
            )
            feature_table = pa.Table.from_pandas(features, preserve_index=False)
            label_table = pa.Table.from_pandas(labels, preserve_index=False)
            if feature_writer is None:
                feature_writer = pq.ParquetWriter(feature_path, feature_table.schema)
                label_writer = pq.ParquetWriter(label_path, label_table.schema)
            feature_writer.write_table(feature_table)
            label_writer.write_table(label_table)
            expected_candidates += len(features)
    finally:
        if feature_writer is not None:
            feature_writer.close()
        if label_writer is not None:
            label_writer.close()

    decisions, artifacts = grouped_oof_predictions_out_of_core(
        [CandidateDatasetPair(feature_path, label_path, "train")],
        output_path=tmp_path / "oof.parquet",
        folds=3,
        seed=7,
        maximum_training_candidates_per_sample=16,
    )
    oof = pd.read_parquet(tmp_path / "oof.parquet")
    row_groups = _row_group_by_sample(pq.ParquetFile(tmp_path / "oof.parquet"))
    first_labels = _sample_row_group(
        pq.ParquetFile(label_path), 0, "sample_0"
    )
    assert len(oof) == expected_candidates
    assert len(first_labels) == 4
    assert set(row_groups) == {f"sample_{index}" for index in range(6)}
    assert oof[["sample_id", "candidate_id"]].duplicated().sum() == 0
    assert np.isfinite(
        oof[
            [
                "m0_p90_calibrated",
                "m1_p90_calibrated",
                "m2_predicted_iou",
            ]
        ].to_numpy(float)
    ).all()
    assert len(decisions) == 6 * 4
    assert all(fold["group_overlap"] == 0 for fold in artifacts["fold_audit"])
    assert artifacts["training_subset_audit"]["oof_scored_all_eligible_candidates"]

    projected = CandidateDatasetPair(
        feature_path,
        label_path,
        "train",
        feature_columns=("sam_score",),
        excluded_source_families=("AUTOMATIC",),
    )
    projected_sample = next(iter(iter_joined_candidate_samples([projected])))
    assert "feature_score" not in projected_sample
    assert {
        "sample_id",
        "candidate_id",
        "frame_id",
        "scene_id",
        "source_family",
        "eligible_final",
        "candidate_iou",
    }.issubset(projected_sample.columns)
    assert not projected_sample.loc[
        projected_sample["source_family"] == "AUTOMATIC", "eligible_final"
    ].item()

    tied = pd.DataFrame(
        {
            "candidate_id": ["a", "z"],
            "m1_p90_calibrated": [0.8, 0.8],
            "m2_predicted_iou": [0.7, 0.9],
        }
    )
    classifier = _rank_candidates(
        tied,
        selected_method="M1_hgb_classifier",
        score_column="m1_p90_calibrated",
    )
    ensemble = _rank_candidates(
        tied,
        selected_method="M3_two_head_ensemble",
        score_column="m1_p90_calibrated",
    )
    assert classifier.iloc[0]["candidate_id"] == "a"
    assert ensemble.iloc[0]["candidate_id"] == "z"
