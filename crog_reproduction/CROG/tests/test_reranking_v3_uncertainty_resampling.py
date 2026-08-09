from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from failure_analysis.reranking_v3.aligned_crops import CROP_CHANNELS, geometry_templates
from failure_analysis.reranking_v3.candidate_relations import extract_candidate_relation_features
from failure_analysis.reranking_v3.coordinate_mapping import sample_original_roi
from failure_analysis.reranking_v3.depth_geometry import (
    depth_geometry_features,
    derived_depth_features,
)
from failure_analysis.reranking_v3.derived_features import derived_head_features
from failure_analysis.reranking_v3.output_map_features import extract_head_features
from failure_analysis.reranking_v3.uncertainty import (
    _validate_streamed_sample_ids,
    aggregate_uncertainty,
    perturbation_evidence_contract,
    recompute_perturbed_inputs,
    resample_aligned_crops,
    virtual_candidate_set,
)


def test_streaming_identity_accepts_deterministic_nonlexical_catalog_order() -> None:
    observed = ["sample-c", "sample-a", "sample-b"]
    assert _validate_streamed_sample_ids(observed, set(observed)) == observed


@pytest.mark.parametrize(
    "observed",
    (["sample-a", "sample-a"], ["sample-a", "sample-c"]),
)
def test_streaming_identity_rejects_duplicate_or_wrong_coverage(observed: list[str]) -> None:
    with pytest.raises(AssertionError, match="order/coverage"):
        _validate_streamed_sample_ids(observed, {"sample-a", "sample-b"})


def _candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "candidate_checksum": f"checksum_{index}",
            "q_rank": index,
            "q_raw": 0.9 - 0.1 * index,
            "cx": 80.0 + 20.0 * index,
            "cy": 100.0,
            "width_px": 40.0 + index,
            "height_px": 20.0,
            "angle_deg": -20.0 + 10.0 * index,
        }
        for index in range(5)
    ]


def _crop() -> np.ndarray:
    size = 32
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    vv, uu = np.meshgrid(axis, axis, indexing="ij")
    crop = np.zeros((len(CROP_CHANNELS), size, size), dtype=np.float32)
    crop[CROP_CHANNELS.index("rgb_r")] = (uu + 1.0) / 2.0
    crop[CROP_CHANNELS.index("rgb_g")] = (vv + 1.0) / 2.0
    crop[CROP_CHANNELS.index("rgb_b")] = 0.5
    crop[CROP_CHANNELS.index("relative_depth_m")] = 0.03 * uu + 0.01 * vv
    crop[CROP_CHANNELS.index("depth_valid")] = 1.0
    crop[CROP_CHANNELS.index("mask_raw")] = 2.0 - 3.0 * (uu**2 + vv**2)
    crop[CROP_CHANNELS.index("mask_probability")] = 1.0 / (
        1.0 + np.exp(-crop[CROP_CHANNELS.index("mask_raw")])
    )
    crop[CROP_CHANNELS.index("quality_raw")] = 1.5 - 2.0 * (uu**2 + vv**2)
    crop[CROP_CHANNELS.index("quality_probability")] = 1.0 / (
        1.0 + np.exp(-crop[CROP_CHANNELS.index("quality_raw")])
    )
    crop[CROP_CHANNELS.index("sin_2theta")] = np.sin(0.4 + 0.2 * uu)
    crop[CROP_CHANNELS.index("cos_2theta")] = np.cos(0.4 + 0.2 * uu)
    crop[CROP_CHANNELS.index("width_raw")] = 0.3 + uu
    crop[CROP_CHANNELS.index("width_probability")] = 1.0 / (1.0 + np.exp(-(0.3 + uu)))
    templates = geometry_templates(size, device="cpu", dtype=torch.float32).numpy()
    for index, name in enumerate(
        ("left_finger_template", "right_finger_template", "contact_template", "gripper_template")
    ):
        crop[CROP_CHANNELS.index(name)] = templates[index]
    return crop


def test_center_translation_is_projected_through_candidate_angle() -> None:
    size = 32
    ramp = torch.linspace(-1.0, 1.0, size).view(1, 1, 1, 1, size).expand(1, 1, 1, size, size)
    width = torch.tensor([[40.0]])
    height = torch.tensor([[20.0]])
    angle_zero = resample_aligned_crops(
        ramp, kind="center_x_px", value=4.0,
        candidate_width_px=width, candidate_height_px=height,
        candidate_angle_deg=torch.tensor([[0.0]]),
    )
    angle_ninety = resample_aligned_crops(
        ramp, kind="center_x_px", value=4.0,
        candidate_width_px=width, candidate_height_px=height,
        candidate_angle_deg=torch.tensor([[90.0]]),
    )
    center = size // 2
    assert abs(float(angle_zero[0, 0, 0, center, center])) > 0.1
    assert abs(float(angle_ninety[0, 0, 0, center, center])) < 0.05


def test_v3_crop_resampling_keeps_templates_and_recentres_valid_depth() -> None:
    source = torch.from_numpy(_crop()).view(1, 1, len(CROP_CHANNELS), 32, 32)
    result = resample_aligned_crops(
        source, kind="angle_deg", value=10.0,
        candidate_width_px=torch.tensor([[40.0]]),
        candidate_height_px=torch.tensor([[20.0]]),
        candidate_angle_deg=torch.tensor([[15.0]]),
    )
    assert not torch.equal(result[:, :, CROP_CHANNELS.index("rgb_r")], source[:, :, CROP_CHANNELS.index("rgb_r")])
    for name in ("left_finger_template", "right_finger_template", "contact_template", "gripper_template"):
        index = CROP_CHANNELS.index(name)
        assert torch.equal(result[:, :, index], source[:, :, index])
    valid = result[0, 0, CROP_CHANNELS.index("depth_valid")] > 0.5
    depth = result[0, 0, CROP_CHANNELS.index("relative_depth_m")]
    assert set(torch.unique(result[0, 0, CROP_CHANNELS.index("depth_valid")]).tolist()) <= {0.0, 1.0}
    assert float(depth[valid].median()) == pytest.approx(0.0, abs=1e-7)


def test_cached_affine_matches_direct_virtual_roi_for_linear_source_field() -> None:
    height, width = 360, 480
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32), indexing="ij",
    )
    source = torch.stack((xx / width, yy / height), dim=0).unsqueeze(0)
    candidate = {
        "cx": 240.0, "cy": 180.0, "width_px": 80.0,
        "height_px": 40.0, "angle_deg": 25.0,
    }
    cached = sample_original_roi(source, candidate, output_size=32).unsqueeze(0)
    for kind, value in (
        ("center_x_px", 4.0), ("center_y_px", -4.0),
        ("angle_deg", 10.0), ("width_scale", 1.1),
    ):
        virtual = virtual_candidate_set(
            [{"candidate_id": "c", "candidate_checksum": "x", **candidate}],
            kind=kind, value=value,
        )[0]
        direct = sample_original_roi(source, virtual, output_size=32)
        observed = resample_aligned_crops(
            cached, kind=kind, value=value,
            candidate_width_px=torch.tensor([[candidate["width_px"]]]),
            candidate_height_px=torch.tensor([[candidate["height_px"]]]),
            candidate_angle_deg=torch.tensor([[candidate["angle_deg"]]]),
        )[0]
        # A cached ROI cannot recover newly exposed source pixels at its border;
        # the interior must nevertheless match direct source-grid sampling.
        assert torch.allclose(observed[..., 4:-4, 4:-4], direct[..., 4:-4, 4:-4], atol=2e-5)


def test_virtual_candidate_geometry_never_changes_identity_or_frozen_input() -> None:
    frozen = _candidates()
    original = [dict(value) for value in frozen]
    for kind, value, field in (
        ("center_x_px", 2.0, "cx"),
        ("center_y_px", -4.0, "cy"),
        ("angle_deg", 5.0, "angle_deg"),
        ("width_scale", 1.1, "width_px"),
    ):
        virtual = virtual_candidate_set(frozen, kind=kind, value=value)
        assert [value["candidate_id"] for value in virtual] == [value["candidate_id"] for value in frozen]
        assert [value["candidate_checksum"] for value in virtual] == [value["candidate_checksum"] for value in frozen]
        assert [value[field] for value in virtual] != [value[field] for value in frozen]
    assert frozen == original


def test_evidence_contract_does_not_claim_unrecoverable_pooled_maps() -> None:
    contract = perturbation_evidence_contract({
        "use_depth": True, "use_latent": True, "use_attention": True,
        "use_prior": True, "use_crop": True, "text_mode": "token",
    })
    assert contract["resampling_complete"] is False
    assert any(value.startswith("latent_rois:") for value in contract["held_constant_unavailable"])
    assert any(value.startswith("attention_rois:") for value in contract["held_constant_unavailable"])
    assert any(value.startswith("v2_prior:") for value in contract["held_constant_unavailable"])
    assert any(value.startswith("head_features:") for value in contract["cache_resampled_and_recomputed"])
    assert any(value.startswith("depth_features:") for value in contract["cache_resampled_and_recomputed"])


def test_recompute_perturbed_inputs_updates_head_crop_depth_but_not_q_or_ids() -> None:
    candidates = _candidates()
    crops = np.stack([_crop() for _ in candidates])
    base_names = None
    base_values = []
    derived = []
    for candidate, crop in zip(candidates, crops, strict=True):
        names, values = extract_head_features(crop, candidate, candidates, image_shape=(240, 320))
        base_names = names if base_names is None else base_names
        assert names == base_names
        base_values.append(values)
        derived.append(derived_head_features(crop, candidate, candidates, image_shape=(240, 320)))
    base_array = np.stack(base_values).astype(np.float32)
    relation_names, relations = extract_candidate_relation_features(
        candidates, image_shape=(240, 320), head_features=base_array,
        head_feature_names=base_names,
    )
    head = np.concatenate((base_array, np.stack(derived), relations), axis=-1)[None].astype(np.float32)
    depth = np.stack([
        np.concatenate((depth_geometry_features(crop), derived_depth_features(crop))) for crop in crops
    ])[None].astype(np.float32)
    head_names = [*base_names, *(
        "g0_nearest_higher_peak_distance_fraction",
        "g0_predicted_mask_centroid_distance_fraction",
        "g0_predicted_mask_principal_axis_difference_fraction",
        "g3_rho_roi_min",
        "g3_candidate_contact_angle_difference_fraction",
        "g4_contact_band_width_probability_mean",
    ), *relation_names]
    catalog = SimpleNamespace(
        candidate_records={"dev:1": {"candidates": candidates}},
        base_schema={"head_feature_names": base_names},
    )
    normalizers = {
        "head_features": {
            "median": np.zeros(head.shape[-1], np.float32),
            "mean": np.zeros(head.shape[-1], np.float32),
            "scale": np.ones(head.shape[-1], np.float32),
        },
        "depth_features": {
            "median": np.zeros(depth.shape[-1], np.float32),
            "mean": np.zeros(depth.shape[-1], np.float32),
            "scale": np.ones(depth.shape[-1], np.float32),
        },
    }
    batch = {
        "sample_ids": np.asarray(["dev:1"]),
        "records": [{"sample_id": "dev:1", "original_size": [240, 320], "candidate_ids": [value["candidate_id"] for value in candidates]}],
        "head_features": head,
        "depth_features": depth,
        "crops": crops[None].astype(np.float16),
    }
    inputs = {
        "head": torch.from_numpy(head),
        "depth": torch.from_numpy(depth),
        "crops": torch.from_numpy(crops[None]),
        "q": torch.tensor([[value["q_raw"] for value in candidates]], dtype=torch.float32),
    }
    result, candidate_ids = recompute_perturbed_inputs(
        batch=batch, inputs=inputs, catalog=catalog, normalizers=normalizers,
        head_mask=np.ones(len(head_names), dtype=bool),
        config={"use_crop": True, "use_depth": True}, kind="width_scale", value=1.1,
    )
    assert candidate_ids == [[value["candidate_id"] for value in candidates]]
    assert torch.equal(result["q"], inputs["q"])
    assert not torch.equal(result["head"], inputs["head"])
    assert not torch.equal(result["depth"], inputs["depth"])
    assert not torch.equal(result["crops"], inputs["crops"])


def test_probability_uncertainty_includes_candidate_perturbations() -> None:
    scores = np.zeros((3, 1, 5), dtype=np.float32)
    seed_probability = np.full((3, 1, 5), 0.5, dtype=np.float32)
    perturbation_probability = np.stack(
        (np.full((3, 1, 5), 0.25, np.float32), np.full((3, 1, 5), 0.75, np.float32)),
        axis=1,
    )
    perturbation_scores = np.zeros_like(perturbation_probability)
    result = aggregate_uncertainty(
        scores, perturbation_scores=perturbation_scores,
        seed_probabilities=seed_probability,
        perturbation_probabilities=perturbation_probability,
    )
    assert np.all(result["correctness_probability_variance"] > 0.0)
