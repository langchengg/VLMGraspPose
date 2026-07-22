"""Deterministic, stratified human-audit sampling and contact sheets."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


MANUAL_FIELDS = (
    "gt_mapping_correct",
    "pred_mask_target_correct",
    "candidate_projection_correct",
    "baseline_top1_target_correct",
    "oracle_candidate_plausible",
    "automatic_primary_class_correct",
    "manual_failure_cause",
    "notes",
    "reviewer",
)


def _hash(seed: int, *values: Any) -> str:
    value = "|".join([str(seed), *(str(item) for item in values)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iou_bin(value: float) -> str:
    edges = (0.25, 0.50, 0.70, 0.90)
    for edge in edges:
        if value < edge:
            return f"lt_{edge:.2f}"
    return "ge_0.90"


def _multiplicity_bin(value: int) -> str:
    if value == 0:
        return "zero"
    if value == 1:
        return "one"
    if value <= 4:
        return "two_to_four"
    return "five_plus"


def stratified_audit_sample(
    samples: pd.DataFrame,
    *,
    per_class: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample every primary class with broad deterministic stratum coverage.

    A greedy first pass selects one item from each composite stratum, then
    fills remaining slots in stable hash order.  This avoids pretending that
    a tiny audit can be proportionally representative of every cross-product.
    """

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    if "primary_failure_class" not in samples:
        raise ValueError("samples require primary_failure_class")
    selected: list[pd.DataFrame] = []
    for primary, source in samples.groupby("primary_failure_class", sort=True):
        group = source.copy()
        group["audit_iou_bin"] = group["mask_iou"].astype(float).map(_iou_bin)
        group["audit_multiplicity_bin"] = (
            group["n_official_candidates"].astype(int).map(_multiplicity_bin)
        )
        group["audit_stratum"] = group[
            [
                "query_type",
                "audit_iou_bin",
                "audit_multiplicity_bin",
                "target_category",
                "scene_id",
            ]
        ].astype(str).agg("|".join, axis=1)
        group["_audit_hash"] = [
            _hash(seed, primary, sample_id) for sample_id in group["sample_id"]
        ]
        group = group.sort_values(["_audit_hash", "dataset_index"])
        target = min(per_class, len(group))
        first = group.drop_duplicates("audit_stratum", keep="first")
        if len(first) >= target:
            picked = first.head(target)
        else:
            remaining = group.loc[~group.index.isin(first.index)]
            picked = pd.concat([first, remaining.head(target - len(first))])
        selected.append(picked)
    result = pd.concat(selected, ignore_index=True) if selected else samples.head(0).copy()
    result = result.sort_values(["primary_failure_class", "_audit_hash"])
    result["audit_index"] = np.arange(1, len(result) + 1)
    return result.drop(columns=["_audit_hash"]).reset_index(drop=True)


def _mask(path: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST))
    return value > 0


def _fit_rgb(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _mark_candidates(
    image: Image.Image,
    candidates: pd.DataFrame,
    *,
    show_pred_filter: bool,
    show_gt: bool,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    sx = image.width / max(1.0, float(candidates.attrs.get("source_width", image.width)))
    sy = image.height / max(1.0, float(candidates.attrs.get("source_height", image.height)))
    for row in candidates.itertuples(index=False):
        if not np.isfinite(row.projected_u) or not np.isfinite(row.projected_v):
            continue
        x, y = float(row.projected_u) * sx, float(row.projected_v) * sy
        if show_gt:
            color = (40, 205, 80) if bool(row.gt_target_positive_primary) else (220, 65, 65)
        elif show_pred_filter:
            color = (40, 160, 245) if bool(row.pred_filter_pass) else (135, 135, 135)
        else:
            color = (255, 205, 40)
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        draw.text((x + 6, y - 7), f"{float(row.vgn_quality):.2f}", fill=color)
        if bool(getattr(row, "is_baseline_top1", False)):
            draw.rectangle((x - 8, y - 8, x + 8, y + 8), outline=(255, 255, 255), width=2)
    positive_mask = (
        candidates["gt_target_positive_primary"].fillna(False).astype(bool)
        if "gt_target_positive_primary" in candidates
        else pd.Series(False, index=candidates.index)
    )
    positive = candidates.loc[positive_mask]
    if show_gt and not positive.empty:
        oracle = positive.sort_values(
            ["vgn_quality", "candidate_index_original"], ascending=[False, True]
        ).iloc[0]
        x, y = float(oracle.projected_u) * sx, float(oracle.projected_v) * sy
        draw.line((x - 9, y, x + 9, y), fill=(255, 0, 255), width=3)
        draw.line((x, y - 9, x, y + 9), fill=(255, 0, 255), width=3)
    return canvas


def render_contact_sheet(
    sample: Mapping[str, Any],
    candidates: pd.DataFrame,
    output_path: Path | str,
    *,
    panel_size: tuple[int, int] = (640, 480),
) -> Path:
    """Render four panels: masks, all candidates, filter, and GT labels."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _fit_rgb(str(sample["rgb_path"]), panel_size)
    source_width = int(sample["intrinsics"]["width"])
    source_height = int(sample["intrinsics"]["height"])
    candidates = candidates.copy()
    candidates.attrs.update(source_width=source_width, source_height=source_height)
    pred = _mask(str(sample["pred_mask_path"]), panel_size)
    gt = _mask(str(sample["gt_mask_path"]), panel_size)
    overlay = np.asarray(rgb).copy()
    overlay[pred] = 0.55 * overlay[pred] + 0.45 * np.array([40, 160, 245])
    overlay[gt] = 0.55 * overlay[gt] + 0.45 * np.array([40, 220, 80])
    panels = [
        Image.fromarray(overlay.astype(np.uint8)),
        _mark_candidates(rgb, candidates, show_pred_filter=False, show_gt=False),
        _mark_candidates(rgb, candidates, show_pred_filter=True, show_gt=False),
        _mark_candidates(rgb, candidates, show_pred_filter=False, show_gt=True),
    ]
    labels = (
        "Predicted mask (blue) + GT mask (green)",
        "All official candidates (quality labels)",
        "Pred-filter pass (blue) / reject (grey)",
        "GT-positive (green), baseline box, oracle cross",
    )
    header_height = 105
    canvas = Image.new("RGB", (panel_size[0] * 2, panel_size[1] * 2 + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    flags = sample.get("secondary_flags", "")
    draw.multiline_text(
        (10, 8),
        f"{sample['sample_id']} | {sample['primary_failure_class']}\n"
        f"Instruction: {sample['instruction']}\nSecondary: {flags}",
        fill="black",
        spacing=3,
    )
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = (index % 2) * panel_size[0]
        y = header_height + (index // 2) * panel_size[1]
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + min(panel_size[0], 430), y + 22), fill=(0, 0, 0))
        draw.text((x + 5, y + 4), label, fill="white")
    temporary = path.with_name(f".{path.name}.tmp")
    canvas.save(temporary, format="PNG")
    temporary.replace(path)
    return path


def build_human_audit(
    samples: pd.DataFrame,
    candidates: pd.DataFrame,
    output_root: Path | str,
    *,
    per_class: int = 30,
    seed: int = 42,
    render: bool = True,
) -> pd.DataFrame:
    """Create pending manual-review rows, instructions and contact sheets."""

    root = Path(output_root)
    contact_root = root / "contact_sheets"
    root.mkdir(parents=True, exist_ok=True)
    audit = stratified_audit_sample(samples, per_class=per_class, seed=seed)
    candidate_groups = {key: value for key, value in candidates.groupby("sample_id")}
    records: list[dict[str, Any]] = []
    for row in audit.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        sheet = contact_root / f"{sample_id}.png"
        if render:
            render_contact_sheet(row, candidate_groups.get(sample_id, candidates.head(0)), sheet)
        record = {
            "audit_index": int(row["audit_index"]),
            "sample_id": sample_id,
            "scene_id": row["scene_id"],
            "instruction": row["instruction"],
            "primary_failure_class": row["primary_failure_class"],
            "secondary_flags": row.get("secondary_flags", ""),
            "query_type": row.get("query_type", "unknown"),
            "mask_iou": row.get("mask_iou"),
            "n_official_candidates": row.get("n_official_candidates"),
            "contact_sheet": str(sheet),
            **{field: "" for field in MANUAL_FIELDS},
        }
        records.append(record)
    output = pd.DataFrame(records)
    output.to_csv(root / "audit_samples.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    output.to_csv(root / "audit_template.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    try:
        output.to_excel(root / "audit_template.xlsx", index=False)
    except (ImportError, ModuleNotFoundError):
        pass
    instructions = """# Human audit instructions

Status: **manual_audit_pending**. No manual labels or agreement values are inferred.

Review each contact sheet using the RGB image and instruction. Blue/green mask overlays are predicted/GT masks. Candidate labels are official processed VGN quality values. A white box is the current modular baseline; a magenta cross is the highest-quality GT-target-consistent same-pool candidate. These are 2-D projected official grasp-pose-origin diagnostics, not physical grasp outcomes.

Fill every manual field with `yes`, `no`, or `uncertain` where applicable. Record a reviewer identity. `manual_failure_cause` should use a short controlled phrase plus details in `notes`. A second independent reviewer enables Cohen's kappa; one reviewer permits agreement only.
"""
    (root / "audit_instructions.md").write_text(instructions, encoding="utf-8")
    return output


__all__ = [
    "MANUAL_FIELDS",
    "build_human_audit",
    "render_contact_sheet",
    "stratified_audit_sample",
]
