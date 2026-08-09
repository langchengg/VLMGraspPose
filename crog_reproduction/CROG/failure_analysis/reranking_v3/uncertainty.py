from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .experiment_config import PERTURBATIONS
from .aligned_crops import CROP_CHANNELS
from .candidate_relations import G10_FEATURE_NAMES, extract_candidate_relation_features
from .depth_geometry import depth_geometry_features, derived_depth_features
from .derived_features import derived_head_features
from .feature_data import apply_normalizer
from .feature_store import FeatureCatalog
from .models.fullchain_ranker import FullChainRanker
from .oof import _normalizers_from_json, model_kwargs, required_array_keys, resolve_device, torch_feature_batch
from .output_map_features import extract_head_features
from .schema import artifact_identity, atomic_write_json, sha256_file


_TEMPLATE_CHANNELS = (
    "left_finger_template", "right_finger_template", "contact_template", "gripper_template",
)
_CONTINUOUS_CROP_CHANNELS = tuple(
    name for name in CROP_CHANNELS if name not in {*_TEMPLATE_CHANNELS, "depth_valid"}
)
_CACHE_RESAMPLE_LIMITATION = (
    "The cache contains already aligned 32x32 ROIs, not the full-resolution RGB/depth/head maps. "
    "Resampling is therefore deterministic and geometry-correct within the cached ROI, but cannot "
    "recover source evidence that lay outside that ROI or undo the first interpolation."
)


def resample_aligned_crops(
    crops: torch.Tensor,
    *,
    kind: str,
    value: float,
    candidate_width_px: torch.Tensor | None = None,
    candidate_height_px: torch.Tensor | None = None,
    candidate_angle_deg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resample cached candidate-local evidence in physical candidate coordinates.

    ``affine_grid`` maps each output (virtually perturbed) ROI location back to
    the frozen input ROI.  Translations are supplied in original-image axes and
    projected into each candidate's opening/perpendicular axes.  For the V3
    17-channel schema, continuous evidence is bilinear, depth validity is
    nearest-neighbour, relative depth is re-centred, and canonical gripper
    templates are deliberately not warped.
    """
    if crops.ndim != 5:
        raise ValueError("aligned crops must have shape [B,K,C,H,W]")
    batch, candidates, channels, height, width = crops.shape
    count = batch * candidates
    theta = torch.zeros((count, 2, 3), dtype=crops.dtype, device=crops.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    widths = None if candidate_width_px is None else candidate_width_px.reshape(-1).to(crops)
    heights = None if candidate_height_px is None else candidate_height_px.reshape(-1).to(crops)
    angles = None if candidate_angle_deg is None else candidate_angle_deg.reshape(-1).to(crops)
    for name, geometry in (("width", widths), ("height", heights), ("angle", angles)):
        if geometry is not None and geometry.numel() != count:
            raise ValueError(f"candidate {name} geometry must have shape [B,K]")
    if kind in {"center_x_px", "center_y_px"}:
        if widths is None:
            raise ValueError("center perturbation requires frozen candidate width")
        # Backwards-compatible defaults are exact for square, zero-angle test
        # crops.  Production inference always supplies all three geometries.
        if heights is None:
            heights = widths
        if angles is None:
            angles = torch.zeros_like(widths)
        radians = torch.deg2rad(angles)
        dx = float(value) if kind == "center_x_px" else 0.0
        dy = float(value) if kind == "center_y_px" else 0.0
        # candidate_original_grid uses half-width=0.75*w and half-height=h.
        theta[:, 0, 2] = (dx * torch.cos(radians) - dy * torch.sin(radians)) / (0.75 * widths).clamp_min(1.0)
        theta[:, 1, 2] = (dx * torch.sin(radians) + dy * torch.cos(radians)) / heights.clamp_min(1.0)
    elif kind == "angle_deg":
        if widths is None or heights is None:
            raise ValueError("angle perturbation requires frozen candidate width and height")
        radians = math.radians(float(value))
        cosine = math.cos(radians)
        sine = math.sin(radians)
        half_width = (0.75 * widths).clamp_min(1.0)
        half_height = heights.clamp_min(1.0)
        # Physical rotation in an anisotropic ROI is not a plain normalized-grid
        # rotation: the cross terms must account for the two physical half axes.
        theta[:, 0, 0] = cosine
        # In image coordinates opening=(cos(theta),-sin(theta)) and
        # perpendicular=(sin(theta),cos(theta)); projecting the perturbed axes
        # back into the frozen axes therefore gives +sin / -sin cross terms.
        theta[:, 0, 1] = sine * half_height / half_width
        theta[:, 1, 0] = -sine * half_width / half_height
        theta[:, 1, 1] = cosine
    elif kind == "width_scale":
        scale = float(value)
        if scale <= 0:
            raise ValueError("width perturbation scale must be positive")
        theta[:, 0, 0] = scale
    else:
        raise ValueError(f"unknown perturbation: {kind}")
    # The frozen ROI producer samples candidate coordinates at linspace(-1,1),
    # whereas affine_grid(align_corners=False) locates output pixel centres at
    # -1+1/N ... 1-1/N.  Linear terms cancel under the two conversions, while
    # translations need this centre-coordinate correction.
    theta[:, 0, 2] *= (width - 1.0) / width
    theta[:, 1, 2] *= (height - 1.0) / height
    flattened = crops.reshape(count, channels, height, width)
    grid = F.affine_grid(theta, flattened.shape, align_corners=False)
    sampled = F.grid_sample(flattened, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    if channels == len(CROP_CHANNELS):
        valid_index = CROP_CHANNELS.index("depth_valid")
        depth_index = CROP_CHANNELS.index("relative_depth_m")
        sampled_valid = F.grid_sample(
            flattened[:, valid_index : valid_index + 1], grid,
            mode="nearest", padding_mode="zeros", align_corners=False,
        )[:, 0]
        sampled[:, valid_index] = sampled_valid
        sampled[:, depth_index] = torch.where(
            sampled_valid > 0.5, sampled[:, depth_index], torch.zeros_like(sampled[:, depth_index])
        )
        # Relative depth was centred in the frozen ROI.  A shifted/rotated ROI
        # needs its own local centre; the unknown absolute offset cancels here.
        for row in range(count):
            valid = sampled_valid[row] > 0.5
            if bool(valid.any()):
                median = sampled[row, depth_index][valid].median()
                sampled[row, depth_index][valid] -= median
        for name in _TEMPLATE_CHANNELS:
            channel = CROP_CHANNELS.index(name)
            sampled[:, channel] = flattened[:, channel]
    return sampled.reshape_as(crops)


def virtual_candidate_set(
    candidates: Sequence[Mapping[str, Any]], *, kind: str, value: float,
) -> list[dict[str, Any]]:
    """Return evidence-only virtual geometries while preserving all identities."""
    result = [deepcopy(dict(candidate)) for candidate in candidates]
    if kind not in {"center_x_px", "center_y_px", "angle_deg", "width_scale"}:
        raise ValueError(f"unknown perturbation: {kind}")
    for candidate in result:
        if kind == "center_x_px":
            candidate["cx"] = float(candidate["cx"]) + float(value)
        elif kind == "center_y_px":
            candidate["cy"] = float(candidate["cy"]) + float(value)
        elif kind == "angle_deg":
            candidate["angle_deg"] = float(candidate["angle_deg"]) + float(value)
        else:
            if float(value) <= 0.0:
                raise ValueError("width perturbation scale must be positive")
            candidate["width_px"] = float(candidate["width_px"]) * float(value)
    frozen_identity = [
        (str(candidate["candidate_id"]), str(candidate.get("candidate_checksum", "")), int(candidate.get("q_rank", -1)))
        for candidate in candidates
    ]
    virtual_identity = [
        (str(candidate["candidate_id"]), str(candidate.get("candidate_checksum", "")), int(candidate.get("q_rank", -1)))
        for candidate in result
    ]
    if virtual_identity != frozen_identity:
        raise AssertionError("uncertainty perturbation changed frozen candidate identity/checksum/order")
    assert_candidate_identity_unchanged(
        [[str(value["candidate_id"]) for value in candidates]],
        [[str(value["candidate_id"]) for value in result]],
    )
    return result


def perturbation_evidence_contract(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Declare which configured evidence can and cannot be perturbed from cache."""
    configured = {} if config is None else dict(config)
    recomputed = [
        "head_features:G0_geometry/G1/G2/G3/G4/G5/derived/G10",
        "crops:RGB/depth/native_raw_and_activated_maps",
    ]
    if configured.get("use_depth", False):
        recomputed.append("depth_features:G9")
    held_constant = []
    if configured.get("use_latent", True) or configured.get("text_mode", "token") == "token":
        held_constant.append("latent_rois:pooled_spatial_support_unavailable")
    if configured.get("use_attention", True):
        held_constant.append("attention_rois:pooled_spatial_support_unavailable")
    if configured.get("use_prior", True):
        held_constant.append("v2_prior:frozen_prior_has_no_resampleable_source_maps")
    return {
        "cache_resampled_and_recomputed": recomputed,
        "cache_resampled_continuous_crop_channels": list(_CONTINUOUS_CROP_CHANNELS),
        "cache_resampled_nearest_crop_channels": ["depth_valid"],
        "canonical_unwarped_crop_channels": list(_TEMPLATE_CHANNELS),
        "held_constant_unavailable": held_constant,
        "geometry_invariant": ["q_anchor", "tokens", "sentence", "dynamic", "candidate_identity"],
        "resampling_complete": not held_constant,
        "cache_resampling_limitation": _CACHE_RESAMPLE_LIMITATION,
        "template_semantics": "canonical gripper templates remain fixed in the perturbed ROI coordinate frame",
        "depth_semantics": "nearest validity mask; bilinear relative depth re-centred on each perturbed ROI",
    }


def recompute_perturbed_inputs(
    *,
    batch: Mapping[str, Any],
    inputs: Mapping[str, torch.Tensor],
    catalog: FeatureCatalog,
    normalizers: Mapping[str, Mapping[str, np.ndarray]],
    head_mask: np.ndarray,
    config: Mapping[str, Any],
    kind: str,
    value: float,
) -> tuple[dict[str, torch.Tensor], list[list[str]]]:
    """Recompute every geometry-sensitive field recoverable from V3 caches."""
    if not catalog.candidate_records:
        raise ValueError("uncertainty recomputation requires frozen candidate feature sources")
    device = inputs["head"].device
    sample_ids = list(map(str, batch["sample_ids"]))
    frozen_sets = [catalog.candidate_records[sample_id]["candidates"] for sample_id in sample_ids]
    virtual_sets = [virtual_candidate_set(values, kind=kind, value=value) for values in frozen_sets]
    widths = torch.as_tensor(
        [[float(value["width_px"]) for value in values] for values in frozen_sets],
        dtype=inputs["head"].dtype, device=device,
    )
    heights = torch.as_tensor(
        [[float(value["height_px"]) for value in values] for values in frozen_sets],
        dtype=inputs["head"].dtype, device=device,
    )
    angles = torch.as_tensor(
        [[float(value["angle_deg"]) for value in values] for values in frozen_sets],
        dtype=inputs["head"].dtype, device=device,
    )
    source_crops = torch.as_tensor(np.asarray(batch["crops"]), dtype=inputs["head"].dtype, device=device)
    perturbed_crops = resample_aligned_crops(
        source_crops, kind=kind, value=value,
        candidate_width_px=widths, candidate_height_px=heights, candidate_angle_deg=angles,
    )
    crops_numpy = perturbed_crops.float().cpu().numpy()
    base_names = list(catalog.base_schema["head_feature_names"])
    raw_head = np.asarray(batch["head_features"], dtype=np.float32)
    head_normalizer = normalizers["head_features"]
    raw_head = (
        raw_head * np.asarray(head_normalizer["scale"], dtype=np.float32)
        + np.asarray(head_normalizer["mean"], dtype=np.float32)
    )
    recomputed_heads = []
    recomputed_depths = []
    candidate_ids: list[list[str]] = []
    for row, (record, frozen, virtual) in enumerate(zip(batch["records"], frozen_sets, virtual_sets, strict=True)):
        base_values = []
        derived_values = []
        for candidate_index, candidate in enumerate(virtual):
            names, values = extract_head_features(
                crops_numpy[row, candidate_index], candidate, virtual,
                image_shape=tuple(map(int, record["original_size"])),
            )
            if names != base_names:
                raise AssertionError("perturbation head feature schema changed")
            # q-prominence is candidate-generator evidence omitted from the
            # minimal candidate cache and is invariant under virtual geometry.
            prominence = base_names.index("g0_peak_prominence")
            values[prominence] = raw_head[row, candidate_index, prominence]
            base_values.append(values)
            derived_values.append(derived_head_features(
                crops_numpy[row, candidate_index], candidate, virtual,
                image_shape=tuple(map(int, record["original_size"])),
            ))
        base_array = np.stack(base_values).astype(np.float32)
        relation_names, relations = extract_candidate_relation_features(
            virtual, image_shape=tuple(map(int, record["original_size"])),
            head_features=base_array, head_feature_names=base_names,
        )
        if relation_names != G10_FEATURE_NAMES:
            raise AssertionError("perturbation G10 feature schema changed")
        recomputed_heads.append(np.concatenate((base_array, np.stack(derived_values), relations), axis=-1))
        if config.get("use_depth", False):
            recomputed_depths.append(np.stack([
                np.concatenate((depth_geometry_features(crop), derived_depth_features(crop)))
                for crop in crops_numpy[row]
            ]))
        candidate_ids.append([str(value["candidate_id"]) for value in frozen])
    head_array = apply_normalizer(np.stack(recomputed_heads), normalizers["head_features"])
    head_array[:, :, ~np.asarray(head_mask, dtype=bool)] = 0.0
    result = dict(inputs)
    result["head"] = torch.from_numpy(head_array).to(device)
    if config.get("use_crop", False):
        result["crops"] = perturbed_crops
    if config.get("use_depth", False):
        depth_array = apply_normalizer(np.stack(recomputed_depths), normalizers["depth_features"])
        result["depth"] = torch.from_numpy(depth_array).to(device)
    return result, candidate_ids


def aggregate_uncertainty(
    seed_scores: np.ndarray,
    *,
    perturbation_scores: np.ndarray | None = None,
    seed_probabilities: np.ndarray | None = None,
    perturbation_probabilities: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    scores = np.asarray(seed_scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[2] != 5:
        raise ValueError("seed scores must be [seeds,samples,5]")
    combined = scores
    if perturbation_scores is not None:
        perturbed = np.asarray(perturbation_scores, dtype=np.float64)
        if perturbed.ndim != 4 or perturbed.shape[0] != scores.shape[0] or perturbed.shape[2:] != scores.shape[1:]:
            raise ValueError("perturbation scores must be [seeds,perturbations,samples,5]")
        combined = np.concatenate((scores[:, None], perturbed), axis=1).reshape(-1, scores.shape[1], 5)
    finite = np.isfinite(combined)
    safe = np.where(finite, combined, np.nan)
    top = np.nanargmax(safe, axis=2)
    votes = np.stack([(top == candidate).sum(axis=0) for candidate in range(5)], axis=1)
    mean = np.nanmean(safe, axis=0)
    mean_top = mean.argmax(axis=1)
    ranking_consistency = (top == mean_top[None, :]).mean(axis=0)
    result = {
        "score_mean": mean.astype(np.float32),
        "score_std": np.nanstd(safe, axis=0).astype(np.float32),
        "score_min": np.nanmin(safe, axis=0).astype(np.float32),
        "score_max": np.nanmax(safe, axis=0).astype(np.float32),
        "valid_fraction": finite.mean(axis=0).astype(np.float32),
        "ranking_consistency": ranking_consistency.astype(np.float32),
        "top1_votes": votes.astype(np.int16),
        "ensemble_disagreement": (1.0 - votes.max(axis=1) / combined.shape[0]).astype(np.float32),
    }
    if seed_probabilities is not None:
        probability = np.asarray(seed_probabilities, dtype=np.float64)
        if probability.shape != scores.shape:
            raise ValueError("seed probabilities must match seed scores")
        if perturbation_probabilities is not None:
            perturbed_probability = np.asarray(perturbation_probabilities, dtype=np.float64)
            if perturbation_scores is None or perturbed_probability.shape != np.asarray(perturbation_scores).shape:
                raise ValueError("perturbation probabilities must match perturbation scores")
            probability = np.concatenate(
                (probability[:, None], perturbed_probability), axis=1,
            ).reshape(-1, scores.shape[1], 5)
        result["correctness_probability_variance"] = probability.var(axis=0).astype(np.float32)
    elif perturbation_probabilities is not None:
        raise ValueError("perturbation probabilities require seed probabilities")
    return result


def perturbation_contract() -> dict[str, Any]:
    return {
        "perturbations": list(PERTURBATIONS),
        "candidate_pool_unchanged": True,
        "candidate_geometry_output_unchanged": True,
        "purpose": "local evidence resampling and uncertainty estimation only",
        "evidence_contract": perturbation_evidence_contract(),
    }


def assert_candidate_identity_unchanged(before: Sequence[Sequence[str]], after: Sequence[Sequence[str]]) -> None:
    if [list(value) for value in before] != [list(value) for value in after]:
        raise AssertionError("uncertainty perturbation changed frozen candidate identity")


def _validate_streamed_sample_ids(
    observed_ids: Sequence[str], expected_ids: set[str],
) -> list[str]:
    """Validate coverage while preserving the feature-store traversal order.

    FeatureCatalog deliberately streams immutable shards in catalog order; that
    order is deterministic but is not required to be lexical sample-ID order.
    All score arrays are accumulated in the same traversal order, so sorting
    only the stored IDs would silently misalign identities and predictions.
    """
    observed = list(map(str, observed_ids))
    if len(observed) != len(expected_ids) or len(set(observed)) != len(observed):
        raise AssertionError("uncertainty streaming order/coverage changed")
    if set(observed) != set(map(str, expected_ids)):
        raise AssertionError("uncertainty streaming order/coverage changed")
    return observed


@torch.no_grad()
def score_perturbation_ensemble_streaming(
    *,
    catalog: FeatureCatalog,
    sample_ids: set[str],
    priors: dict[str, np.ndarray],
    checkpoint_paths: Sequence[str | Path],
    output_path: str | Path,
    device: str = "auto",
    batch_size: int = 32,
) -> dict[str, Any]:
    """Re-score deterministic candidate-local ROI perturbations without changing candidates."""
    if len(checkpoint_paths) != 3:
        raise ValueError("uncertainty contract requires exactly three ensemble checkpoints")
    output = Path(output_path)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    torch_device = resolve_device(device)
    expected_ids = set(map(str, sample_ids))
    models: list[FullChainRanker] = []
    reference_config: dict[str, Any] | None = None
    reference_normalizer_payload: dict[str, Any] | None = None
    reference_head_mask: np.ndarray | None = None
    for checkpoint_path in checkpoint_paths:
        artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if artifact.get("status") != "complete":
            raise ValueError("uncertainty checkpoint is incomplete")
        if reference_config is None:
            reference_config = dict(artifact["config"])
            reference_normalizer_payload = dict(artifact["normalizers"])
            reference_head_mask = np.asarray(artifact["head_mask"], dtype=bool)
        elif dict(artifact["config"]) != reference_config:
            raise ValueError("uncertainty ensemble checkpoints have different model configurations")
        elif dict(artifact["normalizers"]) != reference_normalizer_payload:
            raise ValueError("uncertainty ensemble checkpoints have different feature normalizers")
        elif not np.array_equal(np.asarray(artifact["head_mask"], dtype=bool), reference_head_mask):
            raise ValueError("uncertainty ensemble checkpoints have different head masks")
        model = FullChainRanker(**model_kwargs(artifact["config"]))
        model.load_state_dict(artifact["state_dict"], strict=True)
        models.append(model.to(torch_device).eval())
    if reference_config is None or reference_normalizer_payload is None or reference_head_mask is None:
        raise AssertionError("uncertainty checkpoint loading produced an empty ensemble")
    normalizers = _normalizers_from_json(reference_normalizer_payload)
    all_seed_scores: list[list[np.ndarray]] = [[] for _ in models]
    all_seed_probabilities: list[list[np.ndarray]] = [[] for _ in models]
    all_perturbation_scores: list[list[list[np.ndarray]]] = [
        [[] for _ in PERTURBATIONS] for _ in models
    ]
    all_perturbation_probabilities: list[list[list[np.ndarray]]] = [
        [[] for _ in PERTURBATIONS] for _ in models
    ]
    ordered_ids: list[str] = []
    candidate_ids: list[list[str]] = []
    for batch in catalog.iter_batches(
        allowed_ids=sample_ids, labels=None, priors=priors, normalizers=normalizers,
        batch_size=batch_size, seed=0, shuffle=False,
        array_keys=tuple(dict.fromkeys((*required_array_keys(reference_config), "crops"))),
    ):
        inputs, _ = torch_feature_batch(
            batch, device=torch_device, head_mask=reference_head_mask, config=reference_config,
        )
        for seed_index, model in enumerate(models):
            base = model(**inputs)
            all_seed_scores[seed_index].append(base["scores"].float().cpu().numpy())
            all_seed_probabilities[seed_index].append(
                base["absolute_probability"].float().cpu().numpy()
            )
        frozen_candidate_ids = [
            list(map(str, record["candidate_ids"])) for record in batch["records"]
        ]
        for perturbation_index, perturbation in enumerate(PERTURBATIONS):
            # Geometry-dependent cached evidence is identical across ensemble
            # seeds, so recompute it once and only repeat model inference.
            perturbed_inputs, perturbed_candidate_ids = recompute_perturbed_inputs(
                batch=batch, inputs=inputs, catalog=catalog, normalizers=normalizers,
                head_mask=reference_head_mask, config=reference_config,
                kind=str(perturbation["kind"]), value=float(perturbation["value"]),
            )
            assert_candidate_identity_unchanged(frozen_candidate_ids, perturbed_candidate_ids)
            for seed_index, model in enumerate(models):
                perturbed_output = model(**perturbed_inputs)
                all_perturbation_scores[seed_index][perturbation_index].append(
                    perturbed_output["scores"].float().cpu().numpy()
                )
                all_perturbation_probabilities[seed_index][perturbation_index].append(
                    perturbed_output["absolute_probability"].float().cpu().numpy()
                )
        ordered_ids.extend(map(str, batch["sample_ids"]))
        candidate_ids.extend(frozen_candidate_ids)
    ordered_ids = _validate_streamed_sample_ids(ordered_ids, expected_ids)
    reference_candidate_ids = np.asarray(candidate_ids)
    seed_array = np.stack([np.concatenate(values) for values in all_seed_scores])
    probability_array = np.stack([np.concatenate(values) for values in all_seed_probabilities])
    perturbation_array = np.stack([
        np.stack([np.concatenate(values) for values in seed_values])
        for seed_values in all_perturbation_scores
    ])
    perturbation_probability_array = np.stack([
        np.stack([np.concatenate(values) for values in seed_values])
        for seed_values in all_perturbation_probabilities
    ])
    aggregate = aggregate_uncertainty(
        seed_array, perturbation_scores=perturbation_array,
        seed_probabilities=probability_array,
        perturbation_probabilities=perturbation_probability_array,
    )
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(
            temporary, sample_ids=np.asarray(ordered_ids), candidate_ids=reference_candidate_ids,
            seed_scores=seed_array.astype(np.float32), seed_probabilities=probability_array.astype(np.float32),
            perturbation_scores=perturbation_array.astype(np.float32),
            perturbation_probabilities=perturbation_probability_array.astype(np.float32), **aggregate,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": "3.0.0", "kind": "v3_ensemble_perturbation_uncertainty", "status": "complete",
        "row_count": len(ordered_ids), "unique_sample_count": len(ordered_ids),
        "unique_candidate_count": len(ordered_ids) * 5, "missing_count": 0, "fallback_count": 0,
        "prediction": artifact_identity(output), "checkpoint_sha256": [sha256_file(value) for value in checkpoint_paths],
        "perturbation_contract": perturbation_contract(), "candidate_identity_unchanged": True,
        "candidate_checksums_unchanged": True,
        "evidence_contract": perturbation_evidence_contract(reference_config),
        "resampling_scope": "cached candidate-local crop plus recomputed scalar/depth/set-relation evidence; output geometry and pool remain frozen",
        "valid_fraction_semantics": "fraction of finite seed/perturbation scores; not a claim of full evidence resampling",
        "labels_read": False,
    }
    atomic_write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest
