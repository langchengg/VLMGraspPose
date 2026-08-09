"""GT-free construction and persistence of the Stage-1 proposal bank."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

from .proposal_deduplication import deduplicate_candidates, mask_iou
from .proposal_types import ProposalCandidate
from .query_semantics import QuerySemantics
from .sam3_text_proposals import TextPromptSpec
from .sam3_visual_proposals import VisualPromptSpec
from .selective_sam3_vg.io import resize_binary_mask, resize_probability, sha256_file


def _tight_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    yy, xx = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xx):
        raise ValueError("cannot construct a box from an empty mask")
    return (float(xx.min()), float(yy.min()), float(xx.max()), float(yy.max()))


def _expand_box(
    box: tuple[float, float, float, float],
    shape: tuple[int, int],
    fraction: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    height, width = shape
    dx = round((x2 - x1 + 1.0) * float(fraction))
    dy = round((y2 - y1 + 1.0) * float(fraction))
    return (
        float(max(0.0, x1 - dx)),
        float(max(0.0, y1 - dy)),
        float(min(width - 1.0, x2 + dx)),
        float(min(height - 1.0, y2 + dy)),
    )


def _interior_points(
    core: np.ndarray, count: int, separation_px: float
) -> tuple[tuple[float, float], ...]:
    core = np.asarray(core, dtype=bool)
    if not np.any(core):
        raise ValueError("positive-point core is empty")
    distance = ndimage.distance_transform_edt(core)
    score = distance.copy()
    yy, xx = np.ogrid[: core.shape[0], : core.shape[1]]
    points: list[tuple[float, float]] = []
    for _ in range(int(count)):
        flat = int(np.argmax(score))
        if score.flat[flat] <= 0.0:
            break
        y, x = np.unravel_index(flat, score.shape)
        points.append((float(x), float(y)))
        score[(xx - x) ** 2 + (yy - y) ** 2 <= float(separation_px) ** 2] = 0.0
    while len(points) < int(count):
        points.append(points[0])
    if not all(core[int(y), int(x)] for x, y in points):
        raise AssertionError("every positive point must lie in the target core")
    return tuple(points)


def build_text_prompt_specs(semantics: QuerySemantics) -> list[TextPromptSpec]:
    prompts = [
        TextPromptSpec(
            prompt_id="T0",
            source_family="TEXT_FULL_QUERY",
            source_variant="T0_full_query",
            text=semantics.query,
        )
    ]
    seen = {semantics.query.strip().lower()}

    def add(prompt_id: str, family: str, variant: str, text: str | None, eligible: bool) -> None:
        if not text or text.strip().lower() in seen:
            return
        seen.add(text.strip().lower())
        prompts.append(
            TextPromptSpec(
                prompt_id=prompt_id,
                source_family=family,
                source_variant=variant,
                text=text.strip(),
                eligible_final=eligible,
            )
        )

    add(
        "T1",
        "TEXT_TARGET_CATEGORY",
        "T1_target_category",
        semantics.target_category_prompt,
        True,
    )
    add(
        "T2",
        "TEXT_TARGET_ATTRIBUTE",
        "T2_target_attribute",
        semantics.target_attribute_prompt,
        True,
    )
    add(
        "T3",
        "TEXT_TARGET_INSTANCE",
        "T3_target_instance",
        semantics.target_instance_phrase,
        True,
    )
    # Reference proposals remain logically separate even when target and
    # reference share the same category phrase. The frame text cache reuses
    # the neural output, while provenance and final-mask eligibility preserve
    # the target/reference distinction required by relation reasoning.
    if semantics.reference_category_prompt:
        prompts.append(
            TextPromptSpec(
                prompt_id="R1",
                source_family="REFERENCE_TEXT_CATEGORY",
                source_variant="R1_reference_category",
                text=semantics.reference_category_prompt,
                eligible_final=False,
            )
        )
    if (
        semantics.reference_attribute_prompt
        and semantics.reference_attribute_prompt
        != semantics.reference_category_prompt
    ):
        prompts.append(
            TextPromptSpec(
                prompt_id="R2",
                source_family="REFERENCE_TEXT_ATTRIBUTE",
                source_variant="R2_reference_attribute",
                text=semantics.reference_attribute_prompt,
                eligible_final=False,
            )
        )
    return prompts


def build_hifi_candidates(
    sample_id: str,
    probability_352: np.ndarray,
    native_shape: tuple[int, int],
    rgb_sha256: str,
    thresholds: list[float],
) -> tuple[list[ProposalCandidate], np.ndarray, np.ndarray]:
    probability_352 = np.asarray(probability_352, dtype=np.float32)
    native_probability = resize_probability(probability_352, native_shape)
    candidates: list[ProposalCandidate] = []
    original_native = resize_binary_mask(probability_352 >= 0.50, native_shape)
    for threshold in thresholds:
        mask = resize_binary_mask(probability_352 >= float(threshold), native_shape)
        family = "HIFI_ORIGINAL" if float(threshold) == 0.50 else "HIFI_THRESHOLD"
        variant = "H0_probability_ge_0.50" if family == "HIFI_ORIGINAL" else f"H1_probability_ge_{threshold:.2f}"
        candidates.append(
            ProposalCandidate(
                sample_id=sample_id,
                source_family=family,
                source_variant=variant,
                mask=mask,
                probability=native_probability,
                mask_threshold=float(threshold),
                rgb_checksum=rgb_sha256,
                provenance=[
                    {
                        "source": "frozen_hifics_probability_352",
                        "comparison": ">=",
                        "threshold": float(threshold),
                        "native_transform": "nearest_resize_of_binary_352_mask",
                    }
                ],
            )
        )
    return candidates, original_native, native_probability


def build_visual_prompt_specs(
    hifi_mask: np.ndarray,
    native_probability: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[VisualPromptSpec], dict[str, Any]]:
    hifi_mask = np.asarray(hifi_mask, dtype=bool)
    values = native_probability[hifi_mask]
    if not len(values):
        return [], {"status": "EMPTY_HIFI_MASK"}
    threshold = max(
        float(config["core_minimum_probability"]),
        float(np.quantile(values, float(config["core_probability_quantile"]))),
    )
    core = hifi_mask & (native_probability >= threshold)
    if int(np.count_nonzero(core)) < 4:
        eroded = ndimage.binary_erosion(hifi_mask, iterations=1)
        core = eroded if np.any(eroded) else hifi_mask
    points = {
        count: _interior_points(core, count, float(config["point_separation_px"]))
        for count in config["positive_point_counts"]
    }
    tight = _tight_box(hifi_mask)
    prompts: list[VisualPromptSpec] = []
    for fraction in config["box_expansion_ratios"]:
        box = _expand_box(tight, hifi_mask.shape, float(fraction))
        tag = f"expand={100 * float(fraction):.0f}pct"
        prompts.append(VisualPromptSpec(f"V0_{tag}", "VISUAL_HIFI", f"V0_box|{tag}", box_xyxy=box))
        for index, count in enumerate((1, 3, 5), start=1):
            prompts.append(
                VisualPromptSpec(
                    f"V{index}_{tag}",
                    "VISUAL_HIFI",
                    f"V{index}_box_{count}points|{tag}",
                    box_xyxy=box,
                    positive_points_xy=points[count],
                )
            )
    prompts.append(
        VisualPromptSpec(
            "V4",
            "VISUAL_HIFI",
            "V4_mask_prompt",
            input_mask=hifi_mask.astype(np.float32),
        )
    )
    prompts.append(
        VisualPromptSpec(
            "V5",
            "VISUAL_HIFI",
            "V5_mask_3points",
            positive_points_xy=points[3],
            input_mask=hifi_mask.astype(np.float32),
        )
    )
    return prompts, {
        "status": "READY",
        "core_threshold": threshold,
        "core_area_px": int(np.count_nonzero(core)),
        "tight_box_xyxy": tight,
        "positive_points": {str(key): value for key, value in points.items()},
    }


def build_component_prompt_specs(
    hifi_mask: np.ndarray,
    native_probability: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[VisualPromptSpec], list[dict[str, Any]]]:
    labels, count = ndimage.label(np.asarray(hifi_mask, dtype=bool))
    total_mass = max(float(native_probability[hifi_mask].sum(dtype=np.float64)), 1e-12)
    records: list[dict[str, Any]] = []
    ranked: list[tuple[float, int, int, np.ndarray]] = []
    for label in range(1, count + 1):
        component = labels == label
        area = int(np.count_nonzero(component))
        mass = float(native_probability[component].sum(dtype=np.float64))
        fraction = mass / total_mass
        ranked.append((mass, area, label, component))
        records.append(
            {
                "component_id": label,
                "area_px": area,
                "probability_mass": mass,
                "probability_mass_fraction": fraction,
            }
        )
    ranked.sort(key=lambda value: (-value[0], -value[1], value[2]))
    specs: list[VisualPromptSpec] = []
    for rank, (mass, area, label, component) in enumerate(
        ranked[: int(config["maximum_components"])]
    ):
        fraction = mass / total_mass
        meaningful = rank == 0 or (
            area >= int(config["minimum_area_px"])
            and fraction >= float(config["minimum_probability_mass_fraction"])
        )
        records[label - 1].update({"rank": rank, "meaningful": bool(meaningful)})
        if not meaningful:
            continue
        box = _tight_box(component)
        point = _interior_points(component, 1, 1.0)
        prefix = f"component={label}|rank={rank}"
        specs.extend(
            [
                VisualPromptSpec(f"C0_{label}", "COMPONENT", f"C0_box|{prefix}", box_xyxy=box),
                VisualPromptSpec(
                    f"C1_{label}",
                    "COMPONENT",
                    f"C1_point|{prefix}",
                    positive_points_xy=point,
                ),
                VisualPromptSpec(
                    f"C2_{label}",
                    "COMPONENT",
                    f"C2_box_point|{prefix}",
                    box_xyxy=box,
                    positive_points_xy=point,
                ),
            ]
        )
    return specs, records


def morphological_variants(
    candidates: list[ProposalCandidate],
    hifi_mask: np.ndarray,
    depth_m: np.ndarray | None,
    config: dict[str, Any],
) -> list[ProposalCandidate]:
    scored = []
    for candidate in candidates:
        if candidate.source_family.startswith("HIFI") or not candidate.eligible_final:
            continue
        score = -1.0 if candidate.sam_score is None else float(candidate.sam_score)
        scored.append((mask_iou(candidate.mask, hifi_mask), score, str(candidate.candidate_id), candidate))
    scored.sort(key=lambda value: (-value[0], -value[1], value[2]))
    result: list[ProposalCandidate] = []
    structure = np.ones((3, 3), dtype=bool)
    for _, _, _, parent in scored[: int(config["maximum_parent_candidates"])]:
        labels, count = ndimage.label(parent.mask)
        areas = np.bincount(labels.ravel())[1:] if count else np.asarray([], dtype=int)
        substantial = np.zeros_like(parent.mask)
        minimum = max(
            4,
            int(round(parent.area * float(config["minimum_substantial_component_fraction"]))),
        )
        for label, area in enumerate(areas, start=1):
            if int(area) >= minimum:
                substantial |= labels == label
        transformations = {
            "fill_holes": ndimage.binary_fill_holes(parent.mask),
            "retain_substantial_components": substantial,
            "erode_1px": ndimage.binary_erosion(parent.mask, structure=structure),
            "dilate_1px": ndimage.binary_dilation(parent.mask, structure=structure),
        }
        if depth_m is not None and np.any(parent.mask):
            valid = parent.mask & np.isfinite(depth_m) & (depth_m > 0.0)
            if np.any(valid):
                median = float(np.median(depth_m[valid]))
                depth_clean = parent.mask & (
                    ~np.isfinite(depth_m)
                    | (depth_m <= 0.0)
                    | (np.abs(depth_m - median) <= float(config["depth_tolerance_m"]))
                )
                transformations["depth_connected_cleanup"] = depth_clean
        for name, mask in transformations.items():
            mask = np.asarray(mask, dtype=bool)
            if not np.any(mask) or np.array_equal(mask, parent.mask):
                continue
            result.append(
                ProposalCandidate(
                    sample_id=parent.sample_id,
                    source_family="MORPHOLOGY",
                    source_variant=f"{name}|parent={parent.candidate_id}",
                    mask=mask,
                    sam_score=parent.sam_score,
                    model_revision=parent.model_revision,
                    rgb_checksum=parent.rgb_checksum,
                    provenance=[
                        {
                            "transformation": name,
                            "parent_candidate_id": parent.candidate_id,
                        }
                    ],
                    deduplication_parent_ids=[str(parent.candidate_id)],
                )
            )
    return result


def clone_frame_candidates(
    sample_id: str, candidates: list[ProposalCandidate]
) -> list[ProposalCandidate]:
    return [
        ProposalCandidate(
            sample_id=sample_id,
            source_family=item.source_family,
            source_variant=item.source_variant,
            mask=item.mask,
            probability=item.probability,
            sam_score=item.sam_score,
            presence_score=item.presence_score,
            mask_quality_score=item.mask_quality_score,
            box_xyxy=item.box_xyxy,
            model_revision=item.model_revision,
            rgb_checksum=item.rgb_checksum,
            source_rank=item.source_rank,
            provenance=[dict(value) for value in item.provenance],
        )
        for item in candidates
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
            temporary, format="PNG"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _proposal_grid(
    path: Path,
    rgb: Image.Image,
    candidates: list[ProposalCandidate],
    maximum: int = 32,
) -> None:
    thumb_size = (240, 180)
    by_family: dict[str, list[ProposalCandidate]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate.source_family, []).append(candidate)
    for family in by_family:
        by_family[family].sort(
            key=lambda item: (
                -(float(item.sam_score) if item.sam_score is not None else -1.0),
                str(item.candidate_id),
            )
        )
    family_order = sorted(
        by_family,
        key=lambda family: (
            0 if family == "HIFI_ORIGINAL" else 1 if family == "HIFI_THRESHOLD" else 2,
            family,
        ),
    )
    shown: list[ProposalCandidate] = []
    offset = 0
    while len(shown) < int(maximum):
        progress = False
        for family in family_order:
            values = by_family[family]
            if offset < len(values):
                shown.append(values[offset])
                progress = True
                if len(shown) >= int(maximum):
                    break
        if not progress:
            break
        offset += 1
    columns = 4
    rows = max(1, int(np.ceil(len(shown) / columns)))
    canvas = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 26)), "white")
    rgb_array = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    for index, candidate in enumerate(shown):
        overlay = rgb_array.copy()
        overlay[candidate.mask] = (
            0.45 * overlay[candidate.mask] + 0.55 * np.asarray([0, 255, 120])
        ).astype(np.uint8)
        tile = Image.fromarray(overlay).resize(thumb_size, Image.Resampling.BILINEAR)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + 26)
        canvas.paste(tile, (x, y))
        ImageDraw.Draw(canvas).text(
            (x + 3, y + thumb_size[1] + 3),
            f"{index}: {candidate.source_family[:15]} s={candidate.sam_score or 0:.2f}",
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        canvas.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_proposal_bundle(
    output_directory: Path,
    rgb: Image.Image,
    candidates: list[ProposalCandidate],
    original_hifi_mask: np.ndarray,
    prompt_metadata: dict[str, Any],
    runtime: dict[str, Any],
    *,
    deduplication_iou: float,
) -> list[ProposalCandidate]:
    bundle_started = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)
    unique = deduplicate_candidates(candidates, iou_threshold=float(deduplication_iou))
    unique.sort(key=lambda item: str(item.candidate_id))
    mask_path = output_directory / "candidate_masks.npz"
    mask_temporary = mask_path.with_name(f".{mask_path.name}.{os.getpid()}.tmp.npz")
    parquet_path = output_directory / "candidate_index.parquet"
    parquet_temporary = parquet_path.with_name(f".{parquet_path.name}.{os.getpid()}.tmp")
    try:
        np.savez_compressed(
            mask_temporary,
            __mask_shape__=np.asarray(original_hifi_mask.shape, dtype=np.int32),
            __storage_schema__=np.asarray([3], dtype=np.int32),
            __candidate_ids__=np.asarray([str(item.candidate_id) for item in unique]),
            __packed_masks__=np.stack(
                [
                    np.packbits(item.mask.reshape(-1), bitorder="little")
                    for item in unique
                ]
            ),
        )
        pd.DataFrame([item.to_index_record() for item in unique]).to_parquet(
            parquet_temporary, index=False
        )
        mask_temporary.replace(mask_path)
        parquet_temporary.replace(parquet_path)
    finally:
        mask_temporary.unlink(missing_ok=True)
        parquet_temporary.unlink(missing_ok=True)
    _atomic_mask(output_directory / "original_hifi_mask.png", original_hifi_mask)
    _atomic_json(
        output_directory / "candidate_provenance.json",
        {str(item.candidate_id): item.provenance for item in unique},
    )
    _atomic_json(output_directory / "prompt_metadata.json", prompt_metadata)
    _proposal_grid(output_directory / "proposal_grid.png", rgb, unique)
    runtime["bundle_write_seconds"] = time.perf_counter() - bundle_started
    runtime["sample_seconds_total"] = float(
        runtime.get("sample_seconds_before_write", 0.0)
    ) + float(runtime["bundle_write_seconds"])
    _atomic_json(output_directory / "runtime.json", runtime)
    required = [
        "original_hifi_mask.png",
        "candidate_masks.npz",
        "candidate_index.parquet",
        "candidate_provenance.json",
        "proposal_grid.png",
        "prompt_metadata.json",
        "runtime.json",
    ]
    terminal = {
        "status": "COMPLETE",
        "candidate_count_before_deduplication": len(candidates),
        "candidate_count": len(unique),
        "candidate_mask_storage": "single compressed packbits matrix (bitorder=little), schema=3",
        "checksums": {
            name: sha256_file(output_directory / name) for name in required
        },
    }
    _atomic_json(output_directory / "terminal_status.json", terminal)
    return unique


__all__ = [
    "build_component_prompt_specs",
    "build_hifi_candidates",
    "build_text_prompt_specs",
    "build_visual_prompt_specs",
    "clone_frame_candidates",
    "morphological_variants",
    "write_proposal_bundle",
]
