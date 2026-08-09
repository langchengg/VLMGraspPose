"""Permutation-equivariant candidate-set scorers implemented with PyTorch only.

The models in this module score every candidate in a padded batch.  Inputs use
shape ``[batch, candidates, features]`` (an unbatched ``[candidates, features]``
tensor is also accepted).  ``padding_mask`` follows the PyTorch attention
convention: ``True`` marks a padded candidate that must not affect any valid
candidate.

The implementation intentionally has no PyTorch-Geometric dependency.  The GNN
constructs small per-query graphs and performs mean aggregation with
``Tensor.index_add_``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


ScoreMode = Literal["direct", "residual"]
GraphType = Literal["complete", "knn", "rule", "explicit"]

# The builder below deliberately accepts a separate, unstandardized tensor.
# These fields must use a common spatial unit for x, y, and width; theta is in
# radians, conflict_risk is in [0, 1], and cluster_id is an integer-valued
# identifier (negative means unknown).  Additional trailing fields are ignored.
RAW_EDGE_CANDIDATE_FIELDS = (
    "x",
    "y",
    "theta",
    "width",
    "q",
    "mask_support",
    "depth",
    "clearance",
    "conflict_risk",
    "cluster_id",
)

# Dense relation tensors use layout [batch, source, destination, relation].
# Directional deltas are destination minus source.  IoU, axis overlap,
# same-cluster, and sweep-conflict are symmetric pair attributes.
PAIRWISE_EDGE_RELATION_FIELDS = (
    "delta_x",
    "delta_y",
    "center_distance",
    "sin_2_delta_theta",
    "cos_2_delta_theta",
    "delta_q",
    "delta_width",
    "approx_rectangle_iou",
    "axis_overlap",
    "mask_support_delta",
    "depth_delta",
    "clearance_delta",
    "same_cluster",
    "sweep_conflict",
)

__all__ = [
    "CandidateGNNRanker",
    "CandidateGNNScorer",
    "DeepSetsRanker",
    "DeepSetsScorer",
    "PAIRWISE_EDGE_RELATION_FIELDS",
    "RAW_EDGE_CANDIDATE_FIELDS",
    "SetTransformerRanker",
    "SetTransformerScorer",
    "build_indexed_pairwise_edge_relations",
    "build_pairwise_edge_relations",
    "index_add_mean",
    "parameter_count",
]


def parameter_count(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Return the number of scalar parameters in ``module``.

    Args:
        module: Module to inspect.
        trainable_only: Exclude frozen parameters when true.
    """

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def index_add_mean(messages: Tensor, destination: Tensor, num_nodes: int) -> Tensor:
    """Mean-aggregate edge messages by destination node using ``index_add_``.

    Nodes without incoming edges receive an all-zero aggregate.  ``destination``
    may be on a different device initially and is moved to the message device.
    """

    if messages.ndim != 2:
        raise ValueError("messages must have shape [edges, features]")
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    destination = destination.to(device=messages.device, dtype=torch.long)
    if destination.ndim != 1 or destination.shape[0] != messages.shape[0]:
        raise ValueError("destination must have shape [edges]")
    if destination.numel() and (
        int(destination.min().item()) < 0 or int(destination.max().item()) >= num_nodes
    ):
        raise ValueError("destination contains an out-of-range node index")

    aggregate = messages.new_zeros((num_nodes, messages.shape[-1]))
    counts = messages.new_zeros((num_nodes, 1))
    if destination.numel():
        aggregate.index_add_(0, destination, messages)
        counts.index_add_(
            0,
            destination,
            messages.new_ones((destination.shape[0], 1)),
        )
    return aggregate / counts.clamp_min(1.0)


def build_pairwise_edge_relations(
    raw_candidates: Tensor,
    padding_mask: Tensor | None = None,
    *,
    rectangle_aspect_ratio: float = 0.25,
    eps: float = 1e-6,
) -> Tensor:
    """Build physical pair relations from explicit unstandardized inputs.

    ``raw_candidates`` has shape ``[B, N, F]`` or ``[N, F]`` and its first
    fields follow :data:`RAW_EDGE_CANDIDATE_FIELDS`.  In particular, callers
    must not pass fold-standardized model features: ``x``, ``y``, and ``width``
    use one common physical/image unit and ``theta`` is in radians.

    A grasp rectangle is approximated with major side ``width`` and minor side
    ``rectangle_aspect_ratio * width``.  ``approx_rectangle_iou`` is the IoU of
    the two oriented rectangles' axis-aligned envelopes.  ``axis_overlap``
    instead projects both rectangles onto both candidates' grasp axes and
    averages the two symmetric overlap estimates.  ``sweep_conflict`` is the
    geometric overlap multiplied by the larger explicit per-candidate conflict
    risk.  It is therefore a documented 2.5-D proxy, not collision checking.

    Padded and self-pair relations are exactly zero.  The implementation uses
    PyTorch broadcasting only and is permutation equivariant in the candidate
    dimensions.
    """

    if raw_candidates.ndim not in {2, 3}:
        raise ValueError("raw_candidates must have shape [N, F] or [B, N, F]")
    if not raw_candidates.is_floating_point():
        raise TypeError("raw_candidates must be a floating-point tensor")
    if raw_candidates.shape[-1] < len(RAW_EDGE_CANDIDATE_FIELDS):
        raise ValueError(
            "raw_candidates must provide at least the ordered fields "
            f"{list(RAW_EDGE_CANDIDATE_FIELDS)}"
        )
    if rectangle_aspect_ratio <= 0.0:
        raise ValueError("rectangle_aspect_ratio must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    unbatched = raw_candidates.ndim == 2
    raw = raw_candidates.unsqueeze(0) if unbatched else raw_candidates
    batch_size, candidate_count, _ = raw.shape
    if padding_mask is None:
        padding = torch.zeros(
            (batch_size, candidate_count), dtype=torch.bool, device=raw.device
        )
    else:
        if unbatched and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.shape != (batch_size, candidate_count):
            raise ValueError("padding_mask must match raw candidate dimensions [B, N]")
        padding = padding_mask.to(device=raw.device, dtype=torch.bool)

    raw = raw[..., : len(RAW_EDGE_CANDIDATE_FIELDS)]
    valid = ~padding
    valid_values = raw[valid]
    if valid_values.numel() and not torch.isfinite(valid_values).all():
        raise ValueError("non-padding raw relation inputs must be finite")
    if valid_values.numel():
        if bool((valid_values[:, 3] <= 0.0).any().item()):
            raise ValueError("non-padding candidate widths must be positive")
        conflict_risk = valid_values[:, 8]
        if bool(((conflict_risk < 0.0) | (conflict_risk > 1.0)).any().item()):
            raise ValueError("conflict_risk must be in [0, 1]")
        cluster_id = valid_values[:, 9]
        if bool((cluster_id - cluster_id.round()).abs().gt(1e-5).any().item()):
            raise ValueError("cluster_id must contain integer-valued identifiers")

    # Invalid padded payloads may contain sentinels or NaNs; erase them before
    # any arithmetic so they cannot affect valid pair relations.
    raw = torch.where(padding.unsqueeze(-1), torch.zeros_like(raw), raw)
    x, y, theta, width, q, mask, depth, clearance, risk, cluster = raw.unbind(-1)

    def source(values: Tensor) -> Tensor:
        return values.unsqueeze(2)

    def destination(values: Tensor) -> Tensor:
        return values.unsqueeze(1)

    delta_x = destination(x) - source(x)
    delta_y = destination(y) - source(y)
    distance = torch.sqrt(delta_x.square() + delta_y.square())
    delta_theta = destination(theta) - source(theta)
    delta_q = destination(q) - source(q)
    delta_width = destination(width) - source(width)

    half_major = 0.5 * width
    half_minor = float(rectangle_aspect_ratio) * half_major
    cosine = torch.cos(theta).abs()
    sine = torch.sin(theta).abs()
    half_x = cosine * half_major + sine * half_minor
    half_y = sine * half_major + cosine * half_minor

    intersection_width = (
        torch.minimum(destination(x + half_x), source(x + half_x))
        - torch.maximum(destination(x - half_x), source(x - half_x))
    ).clamp_min(0.0)
    intersection_height = (
        torch.minimum(destination(y + half_y), source(y + half_y))
        - torch.maximum(destination(y - half_y), source(y - half_y))
    ).clamp_min(0.0)
    intersection = intersection_width * intersection_height
    envelope_area = (2.0 * half_x) * (2.0 * half_y)
    union = source(envelope_area) + destination(envelope_area) - intersection
    rectangle_iou = (intersection / union.clamp_min(float(eps))).clamp(0.0, 1.0)

    # Projection support of the destination rectangle on the source axes.
    absolute_cosine_delta = torch.cos(delta_theta).abs()
    absolute_sine_delta = torch.sin(delta_theta).abs()
    destination_on_source_major = (
        destination(half_major) * absolute_cosine_delta
        + destination(half_minor) * absolute_sine_delta
    )
    destination_on_source_minor = (
        destination(half_major) * absolute_sine_delta
        + destination(half_minor) * absolute_cosine_delta
    )
    source_major_distance = (
        delta_x * source(torch.cos(theta)) + delta_y * source(torch.sin(theta))
    ).abs()
    source_minor_distance = (
        -delta_x * source(torch.sin(theta)) + delta_y * source(torch.cos(theta))
    ).abs()
    source_major_extent = source(half_major) + destination_on_source_major
    source_minor_extent = source(half_minor) + destination_on_source_minor
    source_major_overlap = (
        1.0 - source_major_distance / source_major_extent.clamp_min(float(eps))
    ).clamp(0.0, 1.0)
    source_minor_overlap = (
        1.0 - source_minor_distance / source_minor_extent.clamp_min(float(eps))
    ).clamp(0.0, 1.0)
    source_axis_overlap = torch.sqrt(source_major_overlap * source_minor_overlap)

    # Repeat on the destination axes, then average so the scalar is symmetric.
    source_on_destination_major = (
        source(half_major) * absolute_cosine_delta
        + source(half_minor) * absolute_sine_delta
    )
    source_on_destination_minor = (
        source(half_major) * absolute_sine_delta
        + source(half_minor) * absolute_cosine_delta
    )
    destination_major_distance = (
        delta_x * destination(torch.cos(theta))
        + delta_y * destination(torch.sin(theta))
    ).abs()
    destination_minor_distance = (
        -delta_x * destination(torch.sin(theta))
        + delta_y * destination(torch.cos(theta))
    ).abs()
    destination_major_extent = destination(half_major) + source_on_destination_major
    destination_minor_extent = destination(half_minor) + source_on_destination_minor
    destination_major_overlap = (
        1.0
        - destination_major_distance / destination_major_extent.clamp_min(float(eps))
    ).clamp(0.0, 1.0)
    destination_minor_overlap = (
        1.0
        - destination_minor_distance / destination_minor_extent.clamp_min(float(eps))
    ).clamp(0.0, 1.0)
    destination_axis_overlap = torch.sqrt(
        destination_major_overlap * destination_minor_overlap
    )
    axis_overlap = 0.5 * (source_axis_overlap + destination_axis_overlap)

    same_cluster = (
        (source(cluster) >= 0.0)
        & (destination(cluster) >= 0.0)
        & source(cluster).eq(destination(cluster))
    ).to(dtype=raw.dtype)
    pair_conflict_risk = torch.maximum(source(risk), destination(risk))
    sweep_conflict = (
        torch.maximum(rectangle_iou, axis_overlap) * pair_conflict_risk
    ).clamp(0.0, 1.0)

    relations = torch.stack(
        (
            delta_x,
            delta_y,
            distance,
            torch.sin(2.0 * delta_theta),
            torch.cos(2.0 * delta_theta),
            delta_q,
            delta_width,
            rectangle_iou,
            axis_overlap,
            destination(mask) - source(mask),
            destination(depth) - source(depth),
            destination(clearance) - source(clearance),
            same_cluster,
            sweep_conflict,
        ),
        dim=-1,
    )
    diagonal = torch.eye(
        candidate_count, device=raw.device, dtype=torch.bool
    ).unsqueeze(0)
    valid_edges = valid.unsqueeze(2) & valid.unsqueeze(1) & ~diagonal
    relations = relations.masked_fill(~valid_edges.unsqueeze(-1), 0.0)
    return relations.squeeze(0) if unbatched else relations


def build_indexed_pairwise_edge_relations(
    raw_candidates: Tensor,
    edge_batch: Tensor,
    source_index: Tensor,
    destination_index: Tensor,
    padding_mask: Tensor | None = None,
    *,
    relation_indices: Sequence[int] | None = None,
    rectangle_aspect_ratio: float = 0.25,
    eps: float = 1e-6,
) -> Tensor:
    """Build selected physical relations only for the supplied directed edges.

    Edge rows retain the caller's order.  The formulas are the indexed form of
    :func:`build_pairwise_edge_relations`; no ``[B, N, N, R]`` tensor is
    materialized.  Self-edge relations remain exact zeros, matching the dense
    builder even when a graph explicitly includes self edges.
    """

    if raw_candidates.ndim not in {2, 3}:
        raise ValueError("raw_candidates must have shape [N, F] or [B, N, F]")
    if not raw_candidates.is_floating_point():
        raise TypeError("raw_candidates must be a floating-point tensor")
    if raw_candidates.shape[-1] < len(RAW_EDGE_CANDIDATE_FIELDS):
        raise ValueError(
            "raw_candidates must provide at least the ordered fields "
            f"{list(RAW_EDGE_CANDIDATE_FIELDS)}"
        )
    if rectangle_aspect_ratio <= 0.0:
        raise ValueError("rectangle_aspect_ratio must be positive")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    selected = (
        tuple(range(len(PAIRWISE_EDGE_RELATION_FIELDS)))
        if relation_indices is None
        else tuple(relation_indices)
    )
    if any(isinstance(index, bool) or not isinstance(index, int) for index in selected):
        raise TypeError("relation_indices must contain integers")
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= len(PAIRWISE_EDGE_RELATION_FIELDS) for index in selected
    ):
        raise ValueError("relation_indices must be unique in-range field indices")

    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    edge_rows = (edge_batch, source_index, destination_index)
    if any(row.dtype not in integer_dtypes for row in edge_rows):
        raise TypeError("edge index tensors must use an integer dtype")
    if not (
        edge_batch.ndim == source_index.ndim == destination_index.ndim == 1
        and edge_batch.shape == source_index.shape == destination_index.shape
    ):
        raise ValueError("edge index tensors must be one-dimensional and equal length")

    unbatched = raw_candidates.ndim == 2
    raw = raw_candidates.unsqueeze(0) if unbatched else raw_candidates
    batch_size, candidate_count, _ = raw.shape
    if padding_mask is None:
        padding = torch.zeros(
            (batch_size, candidate_count), dtype=torch.bool, device=raw.device
        )
    else:
        if unbatched and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.shape != (batch_size, candidate_count):
            raise ValueError("padding_mask must match raw candidate dimensions [B, N]")
        padding = padding_mask.to(device=raw.device, dtype=torch.bool)

    raw = raw[..., : len(RAW_EDGE_CANDIDATE_FIELDS)]
    valid = ~padding
    valid_values = raw[valid]
    if valid_values.numel() and not torch.isfinite(valid_values).all():
        raise ValueError("non-padding raw relation inputs must be finite")
    if valid_values.numel():
        if bool((valid_values[:, 3] <= 0.0).any().item()):
            raise ValueError("non-padding candidate widths must be positive")
        conflict_risk = valid_values[:, 8]
        if bool(((conflict_risk < 0.0) | (conflict_risk > 1.0)).any().item()):
            raise ValueError("conflict_risk must be in [0, 1]")
        cluster_id = valid_values[:, 9]
        if bool((cluster_id - cluster_id.round()).abs().gt(1e-5).any().item()):
            raise ValueError("cluster_id must contain integer-valued identifiers")

    edge_batch = edge_batch.to(device=raw.device, dtype=torch.long)
    source_index = source_index.to(device=raw.device, dtype=torch.long)
    destination_index = destination_index.to(device=raw.device, dtype=torch.long)
    if edge_batch.numel():
        in_bounds = (
            (edge_batch >= 0)
            & (edge_batch < batch_size)
            & (source_index >= 0)
            & (source_index < candidate_count)
            & (destination_index >= 0)
            & (destination_index < candidate_count)
        )
        if not bool(in_bounds.all().item()):
            raise ValueError("edge index contains an out-of-range coordinate")
        if not bool(
            (valid[edge_batch, source_index] & valid[edge_batch, destination_index])
            .all()
            .item()
        ):
            raise ValueError("indexed edges must reference non-padding candidates")

    # Padded payloads may contain NaN sentinels.  Erase them before gathering so
    # zero-edge batches and future index changes cannot expose invalid values.
    raw = torch.where(padding.unsqueeze(-1), torch.zeros_like(raw), raw)
    source = raw[edge_batch, source_index]
    destination = raw[edge_batch, destination_index]
    if edge_batch.numel() == 0:
        return raw.new_empty((0, len(selected)))
    if not selected:
        return raw.new_empty((edge_batch.numel(), 0))

    sx, sy, stheta, swidth, sq, smask, sdepth, sclearance, srisk, scluster = (
        source.unbind(-1)
    )
    dx, dy, dtheta, dwidth, dq, dmask, ddepth, dclearance, drisk, dcluster = (
        destination.unbind(-1)
    )
    delta_x = dx - sx
    delta_y = dy - sy
    delta_theta = dtheta - stheta

    rectangle_iou: Tensor | None = None
    axis_overlap: Tensor | None = None
    if any(index in {7, 8, 13} for index in selected):
        source_half_major = 0.5 * swidth
        destination_half_major = 0.5 * dwidth
        source_half_minor = float(rectangle_aspect_ratio) * source_half_major
        destination_half_minor = float(rectangle_aspect_ratio) * destination_half_major

        if any(index in {7, 13} for index in selected):
            source_cosine = torch.cos(stheta).abs()
            source_sine = torch.sin(stheta).abs()
            destination_cosine = torch.cos(dtheta).abs()
            destination_sine = torch.sin(dtheta).abs()
            source_half_x = (
                source_cosine * source_half_major + source_sine * source_half_minor
            )
            source_half_y = (
                source_sine * source_half_major + source_cosine * source_half_minor
            )
            destination_half_x = (
                destination_cosine * destination_half_major
                + destination_sine * destination_half_minor
            )
            destination_half_y = (
                destination_sine * destination_half_major
                + destination_cosine * destination_half_minor
            )
            intersection_width = (
                torch.minimum(dx + destination_half_x, sx + source_half_x)
                - torch.maximum(dx - destination_half_x, sx - source_half_x)
            ).clamp_min(0.0)
            intersection_height = (
                torch.minimum(dy + destination_half_y, sy + source_half_y)
                - torch.maximum(dy - destination_half_y, sy - source_half_y)
            ).clamp_min(0.0)
            intersection = intersection_width * intersection_height
            source_area = (2.0 * source_half_x) * (2.0 * source_half_y)
            destination_area = (2.0 * destination_half_x) * (2.0 * destination_half_y)
            union = source_area + destination_area - intersection
            rectangle_iou = (intersection / union.clamp_min(float(eps))).clamp(0.0, 1.0)

        if any(index in {8, 13} for index in selected):
            absolute_cosine_delta = torch.cos(delta_theta).abs()
            absolute_sine_delta = torch.sin(delta_theta).abs()
            destination_on_source_major = (
                destination_half_major * absolute_cosine_delta
                + destination_half_minor * absolute_sine_delta
            )
            destination_on_source_minor = (
                destination_half_major * absolute_sine_delta
                + destination_half_minor * absolute_cosine_delta
            )
            source_major_distance = (
                delta_x * torch.cos(stheta) + delta_y * torch.sin(stheta)
            ).abs()
            source_minor_distance = (
                -delta_x * torch.sin(stheta) + delta_y * torch.cos(stheta)
            ).abs()
            source_major_overlap = (
                1.0
                - source_major_distance
                / (source_half_major + destination_on_source_major).clamp_min(
                    float(eps)
                )
            ).clamp(0.0, 1.0)
            source_minor_overlap = (
                1.0
                - source_minor_distance
                / (source_half_minor + destination_on_source_minor).clamp_min(
                    float(eps)
                )
            ).clamp(0.0, 1.0)
            source_axis_overlap = torch.sqrt(
                source_major_overlap * source_minor_overlap
            )

            source_on_destination_major = (
                source_half_major * absolute_cosine_delta
                + source_half_minor * absolute_sine_delta
            )
            source_on_destination_minor = (
                source_half_major * absolute_sine_delta
                + source_half_minor * absolute_cosine_delta
            )
            destination_major_distance = (
                delta_x * torch.cos(dtheta) + delta_y * torch.sin(dtheta)
            ).abs()
            destination_minor_distance = (
                -delta_x * torch.sin(dtheta) + delta_y * torch.cos(dtheta)
            ).abs()
            destination_major_overlap = (
                1.0
                - destination_major_distance
                / (destination_half_major + source_on_destination_major).clamp_min(
                    float(eps)
                )
            ).clamp(0.0, 1.0)
            destination_minor_overlap = (
                1.0
                - destination_minor_distance
                / (destination_half_minor + source_on_destination_minor).clamp_min(
                    float(eps)
                )
            ).clamp(0.0, 1.0)
            destination_axis_overlap = torch.sqrt(
                destination_major_overlap * destination_minor_overlap
            )
            axis_overlap = 0.5 * (source_axis_overlap + destination_axis_overlap)

    values: dict[int, Tensor] = {}
    if 0 in selected:
        values[0] = delta_x
    if 1 in selected:
        values[1] = delta_y
    if 2 in selected:
        values[2] = torch.sqrt(delta_x.square() + delta_y.square())
    if 3 in selected:
        values[3] = torch.sin(2.0 * delta_theta)
    if 4 in selected:
        values[4] = torch.cos(2.0 * delta_theta)
    if 5 in selected:
        values[5] = dq - sq
    if 6 in selected:
        values[6] = dwidth - swidth
    if 7 in selected:
        assert rectangle_iou is not None
        values[7] = rectangle_iou
    if 8 in selected:
        assert axis_overlap is not None
        values[8] = axis_overlap
    if 9 in selected:
        values[9] = dmask - smask
    if 10 in selected:
        values[10] = ddepth - sdepth
    if 11 in selected:
        values[11] = dclearance - sclearance
    if 12 in selected:
        values[12] = ((scluster >= 0.0) & (dcluster >= 0.0) & scluster.eq(dcluster)).to(
            dtype=raw.dtype
        )
    if 13 in selected:
        assert rectangle_iou is not None and axis_overlap is not None
        values[13] = (
            torch.maximum(rectangle_iou, axis_overlap) * torch.maximum(srisk, drisk)
        ).clamp(0.0, 1.0)

    relations = torch.stack([values[index] for index in selected], dim=-1)
    self_edges = source_index == destination_index
    return relations.masked_fill(self_edges.unsqueeze(-1), 0.0)


def _validate_mode(mode: str) -> ScoreMode:
    if mode not in {"direct", "residual"}:
        raise ValueError("mode must be 'direct' or 'residual'")
    return mode  # type: ignore[return-value]


def _prepare_candidate_inputs(
    features: Tensor,
    baseline_scores: Tensor | None,
    padding_mask: Tensor | None,
) -> tuple[Tensor, Tensor | None, Tensor, bool]:
    if features.ndim not in {2, 3}:
        raise ValueError(
            "features must have shape [candidates, features] or "
            "[batch, candidates, features]"
        )
    if not features.is_floating_point():
        raise TypeError("features must be a floating-point tensor")

    unbatched = features.ndim == 2
    if unbatched:
        features = features.unsqueeze(0)
    batch_size, num_candidates, _ = features.shape

    if padding_mask is None:
        padding = torch.zeros(
            (batch_size, num_candidates),
            dtype=torch.bool,
            device=features.device,
        )
    else:
        if unbatched and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.shape != (batch_size, num_candidates):
            raise ValueError(
                "padding_mask must match the leading feature dimensions "
                f"{(batch_size, num_candidates)}"
            )
        padding = padding_mask.to(device=features.device, dtype=torch.bool)

    baseline: Tensor | None = None
    if baseline_scores is not None:
        if unbatched and baseline_scores.ndim == 1:
            baseline_scores = baseline_scores.unsqueeze(0)
        if baseline_scores.ndim == 3 and baseline_scores.shape[-1] == 1:
            baseline_scores = baseline_scores.squeeze(-1)
        if baseline_scores.shape != (batch_size, num_candidates):
            raise ValueError(
                "baseline_scores must match the leading feature dimensions "
                f"{(batch_size, num_candidates)}"
            )
        baseline = baseline_scores.to(device=features.device, dtype=features.dtype)

    return features, baseline, padding, unbatched


class _ScorerBase(nn.Module):
    """Shared direct/residual output policy."""

    mode: ScoreMode

    def __init__(self, *, mode: ScoreMode, residual_scale: float) -> None:
        super().__init__()
        self.mode = _validate_mode(mode)
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        self.residual_scale = float(residual_scale)

    def _finish_scores(
        self,
        raw_scores: Tensor,
        baseline_scores: Tensor | None,
        padding_mask: Tensor,
        *,
        unbatched: bool,
    ) -> Tensor:
        if self.mode == "residual":
            if baseline_scores is None:
                raise ValueError("baseline_scores are required in residual mode")
            scores = baseline_scores + self.residual_scale * torch.tanh(raw_scores)
            # Padded candidates are exact baseline pass-throughs.
            scores = torch.where(padding_mask, baseline_scores, scores)
        else:
            scores = raw_scores.masked_fill(padding_mask, 0.0)
        return scores.squeeze(0) if unbatched else scores


class DeepSetsScorer(_ScorerBase):
    """Per-candidate DeepSets scorer with masked mean and max context.

    The network is permutation equivariant: each output follows its candidate
    under any candidate permutation, while the set context is formed only with
    symmetric reductions.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        mode: ScoreMode = "residual",
        residual_scale: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(mode=mode, residual_scale=residual_scale)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        baseline_scores: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        features, baseline, padding, unbatched = _prepare_candidate_inputs(
            features, baseline_scores, padding_mask
        )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {features.shape[-1]}"
            )
        if features.shape[1] == 0:
            raw = features.new_zeros(features.shape[:2])
            return self._finish_scores(raw, baseline, padding, unbatched=unbatched)

        valid = ~padding
        hidden = self.phi(features).masked_fill(padding.unsqueeze(-1), 0.0)
        counts = valid.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        mean_context = hidden.sum(dim=1) / counts

        max_input = hidden.masked_fill(padding.unsqueeze(-1), -torch.inf)
        max_context = max_input.amax(dim=1)
        empty_sets = ~valid.any(dim=1)
        max_context = torch.where(
            empty_sets.unsqueeze(-1), torch.zeros_like(max_context), max_context
        )

        context = torch.cat((mean_context, max_context), dim=-1)
        context = context.unsqueeze(1).expand(-1, features.shape[1], -1)
        raw = self.rho(torch.cat((hidden, context), dim=-1)).squeeze(-1)
        return self._finish_scores(raw, baseline, padding, unbatched=unbatched)


class _SetAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, padding_mask: Tensor) -> Tensor:
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + self.attention_dropout(attended))
        update = self.feed_forward(hidden)
        hidden = self.feed_forward_norm(hidden + self.feed_forward_dropout(update))
        return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SetTransformerScorer(_ScorerBase):
    """Small self-attention set scorer without positional encodings."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        mode: ScoreMode = "residual",
        residual_scale: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(mode=mode, residual_scale=residual_scale)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if num_blocks not in {1, 2}:
            raise ValueError("num_blocks must be 1 or 2")
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            _SetAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        baseline_scores: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        features, baseline, padding, unbatched = _prepare_candidate_inputs(
            features, baseline_scores, padding_mask
        )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {features.shape[-1]}"
            )
        batch_size, num_candidates, _ = features.shape
        if num_candidates == 0:
            raw = features.new_zeros((batch_size, 0))
            return self._finish_scores(raw, baseline, padding, unbatched=unbatched)

        # MultiheadAttention produces NaNs when every key in a row is masked.
        # Run only non-empty rows, making empty sets an exact output bypass.
        active = (~padding).any(dim=1)
        raw = features.new_zeros((batch_size, num_candidates))
        if bool(active.any().item()):
            active_features = features[active]
            active_padding = padding[active]
            hidden = self.input_encoder(active_features)
            hidden = hidden.masked_fill(active_padding.unsqueeze(-1), 0.0)
            for block in self.blocks:
                hidden = block(hidden, active_padding)
            active_raw = self.score_head(hidden).squeeze(-1)
            raw[active] = active_raw.masked_fill(active_padding, 0.0)
        return self._finish_scores(raw, baseline, padding, unbatched=unbatched)


@dataclass(frozen=True)
class _GraphEdges:
    source: Tensor
    destination: Tensor
    batch: Tensor
    source_local: Tensor
    destination_local: Tensor
    attributes: Tensor | None = None

    @property
    def count(self) -> int:
        return int(self.source.numel())


class _EdgeMessageBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_hidden_dim: int,
        *,
        use_edge_features: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        message_dim = 2 * hidden_dim
        if use_edge_features:
            message_dim += edge_hidden_dim
        self.message = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        source: Tensor,
        destination: Tensor,
        edge_hidden: Tensor | None,
    ) -> Tensor:
        pieces = (hidden[source], hidden[destination])
        message_input = (
            torch.cat((*pieces, edge_hidden), dim=-1)
            if edge_hidden is not None
            else torch.cat(pieces, dim=-1)
        )
        messages = self.message(message_input)
        aggregate = index_add_mean(messages, destination, hidden.shape[0])
        update = self.update(torch.cat((hidden, aggregate), dim=-1))
        return self.norm(hidden + update)


class CandidateGNNScorer(_ScorerBase):
    """Edge-conditioned candidate graph scorer without PyG.

    Graph choices:

    * ``complete``: directed all-to-all edges, excluding self edges by default.
    * ``knn``: directed edges from each candidate's nearest neighbours.  Pass
      ``coordinates`` to :meth:`forward`, or the first two feature columns are
      used.
    * ``rule``: union of local, angle-compatible, overlap, and sweep-conflict
      edges derived from the explicit 14-field relation tensor.
    * ``explicit``: pass ``edge_index`` with rows ``[source, destination]``.

    For complete/kNN graphs, the legacy edge-feature API accepts dense
    ``[B, N, N, E]`` tensors indexed as
    ``edge_features[batch, source, destination]``.  A kNN graph may instead
    receive ``raw_edge_inputs`` and calculate selected relations only after its
    actual edges are known.  Explicit graphs additionally accept edge-aligned
    ``[E, E_dim]`` or ``[B, E, E_dim]`` attributes.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        edge_dim: int = 0,
        hidden_dim: int = 64,
        edge_hidden_dim: int = 32,
        num_message_passing: int = 2,
        graph_type: GraphType = "complete",
        k: int = 4,
        include_self_edges: bool = False,
        mode: ScoreMode = "residual",
        residual_scale: float = 0.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(mode=mode, residual_scale=residual_scale)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if edge_dim < 0 or edge_hidden_dim <= 0:
            raise ValueError(
                "edge_dim must be non-negative and edge_hidden_dim positive"
            )
        if num_message_passing <= 0:
            raise ValueError("num_message_passing must be positive")
        if graph_type not in {"complete", "knn", "rule", "explicit"}:
            raise ValueError(
                "graph_type must be 'complete', 'knn', 'rule', or 'explicit'"
            )
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = int(input_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.graph_type: GraphType = graph_type
        self.k = int(k)
        self.include_self_edges = bool(include_self_edges)
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if edge_dim:
            self.edge_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(edge_dim, edge_hidden_dim),
                nn.GELU(),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.GELU(),
            )
        else:
            self.edge_encoder = None
        self.message_blocks = nn.ModuleList(
            _EdgeMessageBlock(
                hidden_dim,
                edge_hidden_dim,
                use_edge_features=edge_dim > 0,
                dropout=dropout,
            )
            for _ in range(num_message_passing)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _complete_or_knn_edges(
        self,
        valid: Tensor,
        coordinates: Tensor | None,
        edge_features: Tensor | None,
    ) -> _GraphEdges:
        batch_size, num_candidates = valid.shape
        device = valid.device
        source_parts: list[Tensor] = []
        destination_parts: list[Tensor] = []
        batch_parts: list[Tensor] = []

        for batch_index in range(batch_size):
            local = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            count = local.numel()
            if count == 0:
                continue
            if self.graph_type == "complete":
                source_local = local.repeat_interleave(count)
                destination_local = local.repeat(count)
                if not self.include_self_edges:
                    keep = source_local != destination_local
                    source_local = source_local[keep]
                    destination_local = destination_local[keep]
            elif self.graph_type == "knn":
                if count == 1 and not self.include_self_edges:
                    continue
                assert coordinates is not None
                points = coordinates[batch_index, local]
                squared_distance = (
                    (points[:, None, :] - points[None, :, :]).square().sum(dim=-1)
                )
                if not self.include_self_edges:
                    squared_distance.fill_diagonal_(torch.inf)
                    neighbours = min(self.k, count - 1)
                else:
                    neighbours = min(self.k, count)
                if neighbours == 0:
                    continue
                # Rows are destinations; selected columns are source neighbours.
                neighbour_index = squared_distance.topk(
                    neighbours, dim=1, largest=False, sorted=False
                ).indices
                destination_local = local[:, None].expand(-1, neighbours).reshape(-1)
                source_local = local[neighbour_index.reshape(-1)]
            else:
                if count == 1 and not self.include_self_edges:
                    continue
                if edge_features is None:
                    raise ValueError(
                        "rule graph requires the complete explicit relation tensor"
                    )
                relation = edge_features[batch_index][local[:, None], local[None, :]]
                if relation.shape[-1] < len(PAIRWISE_EDGE_RELATION_FIELDS):
                    raise ValueError(
                        "rule graph requires all documented pairwise relation fields"
                    )
                distance = relation[..., 2].clone()
                if not self.include_self_edges:
                    distance.fill_diagonal_(torch.inf)
                    neighbours = min(self.k, count - 1)
                else:
                    neighbours = min(self.k, count)
                near = torch.zeros((count, count), dtype=torch.bool, device=device)
                if neighbours:
                    # Columns are destinations; rows selected as source nodes.
                    nearest_source = distance.topk(
                        neighbours, dim=0, largest=False, sorted=False
                    ).indices
                    near.scatter_(0, nearest_source, True)
                angle_compatible = relation[..., 4] >= 0.5
                overlaps = (relation[..., 7] >= 0.05) | (relation[..., 8] >= 0.10)
                sweep_conflict = relation[..., 13] >= 0.05
                keep = (near & angle_compatible) | overlaps | sweep_conflict
                if not self.include_self_edges:
                    keep.fill_diagonal_(False)
                source_grid = local[:, None].expand(count, count)
                destination_grid = local[None, :].expand(count, count)
                source_local = source_grid[keep]
                destination_local = destination_grid[keep]

            source_parts.append(source_local)
            destination_parts.append(destination_local)
            batch_parts.append(
                torch.full_like(source_local, batch_index, device=device)
            )

        if not source_parts:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return _GraphEdges(empty, empty, empty, empty, empty)
        source_local = torch.cat(source_parts)
        destination_local = torch.cat(destination_parts)
        edge_batch = torch.cat(batch_parts)
        return _GraphEdges(
            source=edge_batch * num_candidates + source_local,
            destination=edge_batch * num_candidates + destination_local,
            batch=edge_batch,
            source_local=source_local,
            destination_local=destination_local,
        )

    def _explicit_edges(
        self,
        edge_index: Tensor | Sequence[Tensor],
        edge_features: Tensor | Sequence[Tensor] | None,
        valid: Tensor,
    ) -> _GraphEdges:
        batch_size, num_candidates = valid.shape
        device = valid.device
        indices: list[Tensor]
        attributes: list[Tensor | None]

        if isinstance(edge_index, Tensor):
            edge_index = edge_index.to(device=device, dtype=torch.long)
            if edge_index.ndim == 2 and edge_index.shape[0] == 3:
                edge_batch, source_local, destination_local = edge_index
                aligned_features = (
                    edge_features
                    if isinstance(edge_features, Tensor) and edge_features.ndim == 2
                    else None
                )
                return self._filter_explicit_edges(
                    edge_batch,
                    source_local,
                    destination_local,
                    aligned_features,
                    valid,
                )
            if edge_index.ndim == 2 and edge_index.shape[0] == 2:
                indices = [edge_index for _ in range(batch_size)]
            elif edge_index.ndim == 3 and edge_index.shape[:2] == (batch_size, 2):
                indices = [edge_index[index] for index in range(batch_size)]
            else:
                raise ValueError(
                    "edge_index must have shape [2, E], [3, E], or [B, 2, E]"
                )
        else:
            indices = [item.to(device=device, dtype=torch.long) for item in edge_index]
            if len(indices) != batch_size or any(
                item.ndim != 2 or item.shape[0] != 2 for item in indices
            ):
                raise ValueError(
                    "edge_index sequence must contain one [2, E] tensor per batch"
                )

        if edge_features is None:
            attributes = [None] * batch_size
        elif isinstance(edge_features, Tensor):
            edge_features = edge_features.to(device=device)
            if edge_features.ndim == 2:
                attributes = [edge_features for _ in range(batch_size)]
            elif edge_features.ndim == 3 and edge_features.shape[0] == batch_size:
                attributes = [edge_features[index] for index in range(batch_size)]
            elif edge_features.ndim == 4:
                # Dense edge features are gathered after edge filtering.
                attributes = [None] * batch_size
            else:
                raise ValueError(
                    "explicit edge_features must have shape [E, D], [B, E, D], "
                    "or [B, N, N, D]"
                )
        else:
            attributes = [item.to(device=device) for item in edge_features]
            if len(attributes) != batch_size:
                raise ValueError(
                    "edge_features sequence must have one tensor per batch"
                )

        batch_parts: list[Tensor] = []
        source_parts: list[Tensor] = []
        destination_parts: list[Tensor] = []
        attribute_parts: list[Tensor] = []
        for batch_index, (index, attribute) in enumerate(zip(indices, attributes)):
            if attribute is not None and (
                attribute.ndim != 2 or attribute.shape[0] != index.shape[1]
            ):
                raise ValueError("edge-aligned features must have shape [E, edge_dim]")
            edge_batch = torch.full(
                (index.shape[1],), batch_index, dtype=torch.long, device=device
            )
            filtered = self._filter_explicit_edges(
                edge_batch, index[0], index[1], attribute, valid
            )
            batch_parts.append(filtered.batch)
            source_parts.append(filtered.source_local)
            destination_parts.append(filtered.destination_local)
            if filtered.attributes is not None:
                attribute_parts.append(filtered.attributes)

        if not source_parts:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return _GraphEdges(empty, empty, empty, empty, empty)
        edge_batch = torch.cat(batch_parts)
        source_local = torch.cat(source_parts)
        destination_local = torch.cat(destination_parts)
        combined_attributes = torch.cat(attribute_parts) if attribute_parts else None
        return _GraphEdges(
            source=edge_batch * num_candidates + source_local,
            destination=edge_batch * num_candidates + destination_local,
            batch=edge_batch,
            source_local=source_local,
            destination_local=destination_local,
            attributes=combined_attributes,
        )

    def _filter_explicit_edges(
        self,
        edge_batch: Tensor,
        source_local: Tensor,
        destination_local: Tensor,
        attributes: Tensor | None,
        valid: Tensor,
    ) -> _GraphEdges:
        batch_size, num_candidates = valid.shape
        device = valid.device
        edge_batch = edge_batch.to(device=device, dtype=torch.long)
        source_local = source_local.to(device=device, dtype=torch.long)
        destination_local = destination_local.to(device=device, dtype=torch.long)
        if not (
            edge_batch.ndim == source_local.ndim == destination_local.ndim == 1
            and edge_batch.shape == source_local.shape == destination_local.shape
        ):
            raise ValueError(
                "explicit edge index rows must be one-dimensional and equal length"
            )
        if attributes is not None and (
            attributes.ndim != 2 or attributes.shape[0] != edge_batch.shape[0]
        ):
            raise ValueError("edge-aligned features must have shape [E, edge_dim]")
        in_bounds = (
            (edge_batch >= 0)
            & (edge_batch < batch_size)
            & (source_local >= 0)
            & (source_local < num_candidates)
            & (destination_local >= 0)
            & (destination_local < num_candidates)
        )
        keep = in_bounds.clone()
        if bool(in_bounds.any().item()):
            bounded_batch = edge_batch.clamp(0, max(batch_size - 1, 0))
            bounded_source = source_local.clamp(0, max(num_candidates - 1, 0))
            bounded_destination = destination_local.clamp(0, max(num_candidates - 1, 0))
            keep &= valid[bounded_batch, bounded_source]
            keep &= valid[bounded_batch, bounded_destination]
        if not self.include_self_edges:
            keep &= source_local != destination_local
        edge_batch = edge_batch[keep]
        source_local = source_local[keep]
        destination_local = destination_local[keep]
        filtered_attributes = (
            attributes.to(device=device)[keep] if attributes is not None else None
        )
        return _GraphEdges(
            source=edge_batch * num_candidates + source_local,
            destination=edge_batch * num_candidates + destination_local,
            batch=edge_batch,
            source_local=source_local,
            destination_local=destination_local,
            attributes=filtered_attributes,
        )

    def _attach_edge_features(
        self,
        edges: _GraphEdges,
        edge_features: Tensor | Sequence[Tensor] | None,
        *,
        batch_size: int,
        num_candidates: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor | None:
        if self.edge_dim == 0:
            return None
        if edges.count == 0:
            assert self.edge_encoder is not None
            empty = torch.empty((0, self.edge_dim), device=device, dtype=dtype)
            return self.edge_encoder(empty)
        attributes = edges.attributes
        if attributes is None and isinstance(edge_features, Tensor):
            dense = edge_features.to(device=device, dtype=dtype)
            if dense.ndim == 3 and batch_size == 1:
                dense = dense.unsqueeze(0)
            if dense.ndim == 4:
                expected = (batch_size, num_candidates, num_candidates)
                if dense.shape[:3] != expected:
                    raise ValueError(
                        "dense edge_features must have shape [B, N, N, edge_dim]"
                    )
                attributes = dense[
                    edges.batch, edges.source_local, edges.destination_local
                ]
        if attributes is None:
            raise ValueError("edge_features are required when edge_dim is non-zero")
        attributes = attributes.to(device=device, dtype=dtype)
        if attributes.ndim != 2 or attributes.shape != (edges.count, self.edge_dim):
            raise ValueError(
                f"edge features must have shape [{edges.count}, {self.edge_dim}]"
            )
        assert self.edge_encoder is not None
        return self.edge_encoder(attributes)

    def forward(
        self,
        features: Tensor,
        baseline_scores: Tensor | None = None,
        padding_mask: Tensor | None = None,
        *,
        coordinates: Tensor | None = None,
        edge_index: Tensor | Sequence[Tensor] | None = None,
        edge_features: Tensor | Sequence[Tensor] | None = None,
        raw_edge_inputs: Tensor | None = None,
    ) -> Tensor:
        features, baseline, padding, unbatched = _prepare_candidate_inputs(
            features, baseline_scores, padding_mask
        )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {features.shape[-1]}"
            )
        prepared_raw_edge_inputs: Tensor | None = None
        if raw_edge_inputs is not None:
            if self.graph_type != "knn":
                raise ValueError(
                    "raw_edge_inputs sparse relations are supported only for kNN"
                )
            if self.edge_dim == 0:
                raise ValueError(
                    "raw_edge_inputs are unnecessary when kNN edge_dim is zero"
                )
            if edge_features is not None:
                raise ValueError(
                    "pass either edge_features or raw_edge_inputs, not both"
                )
            if coordinates is not None:
                raise ValueError(
                    "raw_edge_inputs supply kNN coordinates; do not also pass coordinates"
                )
            if unbatched and raw_edge_inputs.ndim == 2:
                raw_edge_inputs = raw_edge_inputs.unsqueeze(0)
            if (
                raw_edge_inputs.ndim != 3
                or raw_edge_inputs.shape[:2] != features.shape[:2]
            ):
                raise ValueError("raw_edge_inputs must have shape [B, N, raw_fields]")
            prepared_raw_edge_inputs = raw_edge_inputs.to(device=features.device)
        if (
            self.edge_dim > 0
            and edge_features is None
            and prepared_raw_edge_inputs is None
        ):
            raise ValueError(
                "edge_features are required when edge_dim is non-zero; build "
                "them from explicit unstandardized candidate data with "
                "build_pairwise_edge_relations or pass kNN raw_edge_inputs"
            )
        batch_size, num_candidates, _ = features.shape
        if num_candidates == 0:
            raw = features.new_zeros((batch_size, 0))
            return self._finish_scores(raw, baseline, padding, unbatched=unbatched)

        valid = ~padding
        if not bool(valid.any().item()):
            raw = features.new_zeros((batch_size, num_candidates))
            return self._finish_scores(raw, baseline, padding, unbatched=unbatched)
        if self.graph_type == "explicit":
            if edge_index is None:
                raise ValueError("edge_index is required for an explicit graph")
            edges = self._explicit_edges(edge_index, edge_features, valid)
        else:
            if edge_index is not None:
                raise ValueError("edge_index is only valid when graph_type='explicit'")
            prepared_coordinates: Tensor | None = None
            prepared_rule_features: Tensor | None = None
            if self.graph_type == "knn":
                if coordinates is None:
                    if prepared_raw_edge_inputs is not None:
                        prepared_coordinates = prepared_raw_edge_inputs[..., :2].to(
                            dtype=features.dtype
                        )
                    elif self.input_dim < 2:
                        raise ValueError(
                            "coordinates are required for kNN when input_dim < 2"
                        )
                    else:
                        prepared_coordinates = features[..., :2]
                else:
                    if unbatched and coordinates.ndim == 2:
                        coordinates = coordinates.unsqueeze(0)
                    if coordinates.ndim != 3 or coordinates.shape[:2] != (
                        batch_size,
                        num_candidates,
                    ):
                        raise ValueError(
                            "coordinates must have shape [B, N, coordinate_dim]"
                        )
                    prepared_coordinates = coordinates.to(
                        device=features.device, dtype=features.dtype
                    )
            elif self.graph_type == "rule":
                if not isinstance(edge_features, Tensor):
                    raise ValueError(
                        "rule graph requires a dense explicit relation tensor"
                    )
                prepared_rule_features = edge_features.to(
                    device=features.device, dtype=features.dtype
                )
                if unbatched and prepared_rule_features.ndim == 3:
                    prepared_rule_features = prepared_rule_features.unsqueeze(0)
                if prepared_rule_features.ndim != 4 or (
                    prepared_rule_features.shape[:3]
                    != (batch_size, num_candidates, num_candidates)
                ):
                    raise ValueError(
                        "rule graph edge_features must have shape [B, N, N, E]"
                    )
            edges = self._complete_or_knn_edges(
                valid, prepared_coordinates, prepared_rule_features
            )

        if prepared_raw_edge_inputs is not None:
            selected_indices = tuple(
                int(index) for index in getattr(self, "edge_feature_indices", ())
            )
            relation_count = len(PAIRWISE_EDGE_RELATION_FIELDS)
            if len(set(selected_indices)) != len(selected_indices) or any(
                index < 0 or index >= relation_count for index in selected_indices
            ):
                raise ValueError("invalid CandidateGNNScorer edge_feature_indices")
            expected_edge_dim = (
                len(selected_indices) if selected_indices else relation_count
            )
            if self.edge_dim != expected_edge_dim:
                raise ValueError(
                    "CandidateGNNScorer edge_dim must match its explicit raw "
                    f"relation selection ({expected_edge_dim}), got {self.edge_dim}"
                )
            sparse_attributes = build_indexed_pairwise_edge_relations(
                prepared_raw_edge_inputs,
                edges.batch,
                edges.source_local,
                edges.destination_local,
                padding,
                relation_indices=selected_indices or None,
            )
            edges = _GraphEdges(
                source=edges.source,
                destination=edges.destination,
                batch=edges.batch,
                source_local=edges.source_local,
                destination_local=edges.destination_local,
                attributes=sparse_attributes,
            )

        edge_hidden = self._attach_edge_features(
            edges,
            edge_features,
            batch_size=batch_size,
            num_candidates=num_candidates,
            dtype=features.dtype,
            device=features.device,
        )
        hidden = self.node_encoder(features).masked_fill(padding.unsqueeze(-1), 0.0)
        flat_hidden = hidden.reshape(batch_size * num_candidates, self.hidden_dim)
        for block in self.message_blocks:
            flat_hidden = block(
                flat_hidden, edges.source, edges.destination, edge_hidden
            )
            flat_hidden = flat_hidden.masked_fill(padding.reshape(-1, 1), 0.0)
        hidden = flat_hidden.reshape(batch_size, num_candidates, self.hidden_dim)
        raw = self.score_head(hidden).squeeze(-1)
        return self._finish_scores(raw, baseline, padding, unbatched=unbatched)


# Concise aliases for experiment registries that use the term "ranker".
DeepSetsRanker = DeepSetsScorer
SetTransformerRanker = SetTransformerScorer
CandidateGNNRanker = CandidateGNNScorer
