"""Strict proposal deduplication with complete provenance preservation."""

from __future__ import annotations

import numpy as np

from .proposal_types import ProposalCandidate


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return 1.0 if union == 0 else float(intersection / union)


def _near_identical(
    left: ProposalCandidate,
    right: ProposalCandidate,
    threshold: float,
    boxes: dict[int, tuple[int, int, int, int] | None],
    areas: dict[int, int],
) -> bool:
    if left.mask.shape != right.mask.shape:
        return False
    if np.array_equal(left.mask, right.mask):
        return True
    left_area = areas[id(left)]
    right_area = areas[id(right)]
    maximum = max(left_area, right_area, 1)
    if abs(left_area - right_area) / maximum > 0.005:
        return False
    left_box = boxes[id(left)]
    right_box = boxes[id(right)]
    if left_box is None or right_box is None:
        return left_box is None and right_box is None
    if max(abs(int(a) - int(b)) for a, b in zip(left_box, right_box, strict=True)) > 2:
        return False
    return mask_iou(left.mask, right.mask) >= threshold


def _merge(duplicate: ProposalCandidate, candidate: ProposalCandidate) -> None:
    duplicate.provenance.extend(candidate.provenance)
    duplicate.deduplication_parent_ids.append(str(candidate.candidate_id))
    duplicate.eligible_final = bool(duplicate.eligible_final or candidate.eligible_final)
    score_left = -np.inf if duplicate.sam_score is None else duplicate.sam_score
    score_right = -np.inf if candidate.sam_score is None else candidate.sam_score
    if score_right > score_left:
        duplicate.sam_score = candidate.sam_score
        duplicate.mask_quality_score = candidate.mask_quality_score
        duplicate.presence_score = candidate.presence_score


def deduplicate_candidates(
    candidates: list[ProposalCandidate],
    *,
    iou_threshold: float = 0.995,
) -> list[ProposalCandidate]:
    if not 0.99 <= float(iou_threshold) <= 1.0:
        raise ValueError("deduplication IoU threshold must be in [0.99,1]")
    # Exact duplicates are common because each raw SAM mask is crossed with
    # multiple instance thresholds and prompt sources. Collapse those in O(N)
    # before the deliberately strict near-duplicate pass.
    # The original HiFi mask is a safety-critical default, not merely another
    # provenance label. If H0 is identical to H1/SAM, H0 owns the canonical
    # candidate ID while every duplicate provenance is still merged into it.
    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (
            item[1].source_family not in {"HIFI_ORIGINAL", "STAGE2_HIFI_FALLBACK"},
            item[0],
        ),
    )
    exact: list[ProposalCandidate] = []
    by_hash: dict[str, ProposalCandidate] = {}
    for _, candidate in ordered:
        digest = candidate.mask_digest
        duplicate = by_hash.get(digest)
        if duplicate is None:
            by_hash[digest] = candidate
            exact.append(candidate)
        elif np.array_equal(duplicate.mask, candidate.mask):
            _merge(duplicate, candidate)
        else:
            # Cryptographic collision guard; do not merge unequal masks.
            exact.append(candidate)

    unique: list[ProposalCandidate] = []
    boxes = {id(item): item.mask_box for item in exact}
    areas = {id(item): item.area for item in exact}
    by_area: dict[int, list[ProposalCandidate]] = {}
    for candidate in exact:
        candidate_area = areas[id(candidate)]
        tolerance = max(1, int(np.ceil(0.005 * max(candidate_area, 1))))
        plausible = (
            item
            for area, items in by_area.items()
            if candidate_area - tolerance <= area <= candidate_area + tolerance
            for item in items
        )
        duplicate = next(
            (
                item
                for item in plausible
                if _near_identical(
                    item,
                    candidate,
                    float(iou_threshold),
                    boxes,
                    areas,
                )
            ),
            None,
        )
        if duplicate is None:
            unique.append(candidate)
            by_area.setdefault(candidate_area, []).append(candidate)
            continue
        _merge(duplicate, candidate)
    return unique


__all__ = ["deduplicate_candidates", "mask_iou"]
