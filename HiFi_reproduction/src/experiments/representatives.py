"""Deterministic representative-sample selection for VGN 3-D diagnostics."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.grasping.vgn_pipeline import atomic_write_json


THREE_D_ARTIFACTS = (
    "local_scene_point_cloud.ply",
    "target_point_cloud.ply",
    "grasps_3d.ply",
)


def _finite(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _evenly_spaced(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def select_representatives(
    rows: Sequence[Mapping[str, Any]], *, count: int = 60
) -> list[dict[str, Any]]:
    """Select a deterministic, diverse subset and record why each row was chosen.

    Selection covers every status first, then symbolic query types, then quantiles
    of mask IoU, VGN quality, and candidate counts. Remaining slots are filled
    uniformly over dataset index. No randomness or model score modification is
    involved.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    normalized = [dict(row) for row in rows]
    sample_ids = [str(row.get("sample_id", "")) for row in normalized]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("all metric rows must contain sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("metric rows contain duplicate sample_id values")
    if len(normalized) < count:
        raise ValueError(f"requested {count} representatives from only {len(normalized)} rows")

    chosen: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def add(row: Mapping[str, Any], reason: str) -> None:
        sample_id = str(row["sample_id"])
        if sample_id in chosen:
            chosen[sample_id]["selection_reasons"].append(reason)
        elif len(chosen) < count:
            chosen[sample_id] = {
                "sample_id": sample_id,
                "selection_reasons": [reason],
                "status": row.get("status"),
                "query_type": row.get("query_type"),
                "target_category": row.get("target_category"),
                "mask_iou": _finite(row, "mask_iou"),
                "top1_vgn_quality": _finite(row, "top1_vgn_quality"),
                "official_candidate_count": _finite(row, "official_candidate_count"),
                "target_candidate_count": _finite(row, "target_candidate_count"),
            }

    for status in sorted({str(row.get("status", "")) for row in normalized}):
        group = sorted(
            (row for row in normalized if str(row.get("status", "")) == status),
            key=lambda row: (int(float(row.get("dataset_index") or 0)), str(row["sample_id"])),
        )
        for row in _evenly_spaced(group, 6):
            add(row, f"status:{status}")

    for query_type in sorted({str(row.get("query_type", "")) for row in normalized}):
        group = sorted(
            (row for row in normalized if str(row.get("query_type", "")) == query_type),
            key=lambda row: (str(row.get("target_category", "")), str(row["sample_id"])),
        )
        for row in _evenly_spaced(group, 3):
            add(row, f"query_type:{query_type or 'missing'}")

    quantile_keys = (
        "mask_iou",
        "top1_vgn_quality",
        "official_candidate_count",
        "target_candidate_count",
    )
    for key in quantile_keys:
        group = [row for row in normalized if _finite(row, key) is not None]
        group.sort(key=lambda row: (_finite(row, key), str(row["sample_id"])))
        for quantile_index, row in enumerate(_evenly_spaced(group, 11)):
            add(row, f"{key}_quantile:{quantile_index}/10")

    category_first: dict[str, Mapping[str, Any]] = {}
    for row in sorted(normalized, key=lambda item: str(item["sample_id"])):
        category = str(row.get("target_category", ""))
        if category:
            category_first.setdefault(category, row)
    for category, row in sorted(category_first.items()):
        add(row, f"target_category:{category}")

    ordered_all = sorted(
        normalized,
        key=lambda row: (int(float(row.get("dataset_index") or 0)), str(row["sample_id"])),
    )
    for row in _evenly_spaced(ordered_all, count * 2):
        add(row, "dataset_index_coverage")
    for row in ordered_all:
        add(row, "deterministic_fill")
        if len(chosen) == count:
            break

    if len(chosen) != count:
        raise RuntimeError(f"selected {len(chosen)} representatives, expected {count}")
    return list(chosen.values())


def load_csv_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_representative_manifest(
    source_manifest: Path | str,
    destination: Path | str,
    selection: Sequence[Mapping[str, Any]],
) -> Path:
    """Write an atomic JSONL subset while refusing duplicate or missing rows."""

    selected_ids = [str(row["sample_id"]) for row in selection]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selection contains duplicate sample IDs")
    wanted = set(selected_ids)
    source_rows: dict[str, dict[str, Any]] = {}
    with Path(source_manifest).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id in wanted:
                if sample_id in source_rows:
                    raise ValueError(
                        f"duplicate selected sample_id {sample_id!r} at manifest line {line_number}"
                    )
                source_rows[sample_id] = row
    missing = sorted(wanted - set(source_rows))
    if missing:
        raise ValueError(f"selected sample IDs missing from manifest: {missing[:10]}")

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in selected_ids:
            handle.write(json.dumps(source_rows[sample_id], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def sync_representative_3d(
    selection: Iterable[Mapping[str, Any]],
    *,
    rendered_output: Path | str,
    experiment_output: Path | str,
) -> dict[str, Any]:
    """Atomically copy the three required 3-D PLY files into sample outputs."""

    rendered = Path(rendered_output)
    experiment = Path(experiment_output)
    copied: list[str] = []
    missing: list[dict[str, str]] = []
    for row in selection:
        sample_id = str(row["sample_id"])
        source_dir = rendered / "samples" / sample_id
        destination_dir = experiment / "samples" / sample_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        complete = True
        for name in THREE_D_ARTIFACTS:
            source = source_dir / name
            if not source.is_file():
                missing.append({"sample_id": sample_id, "artifact": name})
                complete = False
                continue
            destination = destination_dir / name
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        if complete:
            copied.append(sample_id)
    result = {
        "requested_count": len(list(selection)) if isinstance(selection, Sequence) else None,
        "complete_3d_sample_count": len(copied),
        "complete_sample_ids": copied,
        "missing_artifacts": missing,
        "artifact_names": list(THREE_D_ARTIFACTS),
    }
    atomic_write_json(experiment / "report" / "representative_3d_sync.json", result)
    return result


__all__ = [
    "THREE_D_ARTIFACTS",
    "load_csv_rows",
    "select_representatives",
    "sync_representative_3d",
    "write_representative_manifest",
]
