from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from failure_analysis.reranking_v2.schema import atomic_write_jsonl

from .schema import read_jsonl


def checkpoint_ensemble_uses_depth(checkpoint_paths: Sequence[str | Path]) -> bool:
    if len(checkpoint_paths) != 3:
        raise ValueError("depth-mode validation requires exactly three checkpoints")
    modes: list[bool] = []
    for path in checkpoint_paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("status") != "complete" or not isinstance(artifact.get("config"), Mapping):
            raise ValueError(f"invalid FCER checkpoint for depth-mode validation: {path}")
        modes.append(bool(artifact["config"].get("use_depth", False)))
    if len(set(modes)) != 1:
        raise ValueError("FCER checkpoint ensemble mixes Native and RGB-D depth modes")
    return modes[0]


def catalog_depth_availability(
    catalog: Any, sample_ids: set[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    locations = getattr(catalog, "locations", None)
    if locations is None:
        return {
            sample_id:{
                "available":True, "reason":None,
                "provenance":"catalog_without_depth_status_interface",
            }
            for sample_id in sorted(sample_ids)
        }
    for sample_id in sorted(sample_ids):
        location = locations.get(sample_id)
        if location is None:
            raise ValueError(f"depth availability is missing catalog sample: {sample_id}")
        status = location.record.get("depth_source")
        if status is None:
            # Compatibility for immutable feature caches produced before explicit
            # depth provenance existed.  This is not interpreted as missing.
            result[sample_id] = {
                "available": True,
                "reason": None,
                "provenance": "legacy_cache_without_depth_status",
            }
            continue
        if not isinstance(status, Mapping) or not isinstance(status.get("available"), bool):
            raise ValueError(f"invalid depth source status: {sample_id}")
        reason = status.get("reason")
        if not status["available"] and not str(reason or "").strip():
            raise ValueError(f"missing depth has no explicit reason: {sample_id}")
        result[sample_id] = dict(status)
    return result


def plan_depth_aware_execution(
    *,
    sample_ids: set[str],
    depth_status: Mapping[str, Mapping[str, Any]],
    use_depth: bool,
    native_fallback_declared: bool,
) -> dict[str, Any]:
    if set(depth_status) != set(sample_ids):
        raise ValueError("depth status must exactly cover the inference cohort")
    missing = {
        sample_id for sample_id in sample_ids
        if depth_status[sample_id].get("available") is not True
    }
    if not use_depth:
        return {
            "model_sample_ids":set(sample_ids),
            "fallback_sample_ids":set(),
            "fallback_source":None,
            "native_model_requires_depth":False,
        }
    return {
        "model_sample_ids":set(sample_ids)-missing,
        "fallback_sample_ids":missing,
        "fallback_source":"native" if native_fallback_declared else "v2",
        "native_model_requires_depth":True,
    }


def _ranking_lookup(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate fallback ranking sample: {sample_id}")
        order = list(map(str, record.get("candidate_order", ())))
        if len(order) != 5 or len(set(order)) != 5:
            raise ValueError(f"fallback ranking candidate order is invalid: {sample_id}")
        result[sample_id] = record
    return result


def merge_depth_fallback_rankings(
    *,
    ordered_sample_ids: Sequence[str],
    method: str,
    primary_ranking_path: str | Path | None,
    missing_depth_ids: set[str],
    v2_ranking_path: str | Path,
    output_path: str | Path,
    native_ranking_path: str | Path | None = None,
    depth_status: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Materialize RGB-D rankings without ever scoring missing-depth samples."""
    ordered = list(map(str, ordered_sample_ids))
    if len(ordered) != len(set(ordered)) or set(depth_status) != set(ordered):
        raise ValueError("depth fallback identities must be unique and exact")
    if not missing_depth_ids <= set(ordered):
        raise ValueError("missing-depth IDs are outside the cohort")
    primary = {} if primary_ranking_path is None else _ranking_lookup(primary_ranking_path)
    if set(primary) != set(ordered)-missing_depth_ids:
        raise ValueError("RGB-D primary rankings must exactly cover depth-available samples")
    fallback_path = native_ranking_path if native_ranking_path is not None else v2_ranking_path
    fallback = _ranking_lookup(fallback_path)
    if not missing_depth_ids <= set(fallback):
        raise ValueError("fallback ranking is missing depth-unavailable samples")
    fallback_name = "native" if native_ranking_path is not None else "v2"
    rows: list[dict[str, Any]] = []
    for sample_id in ordered:
        if sample_id not in missing_depth_ids:
            rows.append(copy.deepcopy(primary[sample_id]))
            continue
        source = copy.deepcopy(fallback[sample_id])
        source["method"] = str(method)
        source["depth_fallback"] = {
            "applied":True,
            "source":fallback_name,
            "reason":str(depth_status[sample_id].get("reason")),
            "rgbd_model_executed":False,
        }
        selection = source.get("selection")
        if isinstance(selection, Mapping):
            source["selection"] = dict(selection)
            source["selection"]["fallback_reason"] = f"missing_depth_{fallback_name}"
        rows.append(source)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination, rows)
    return destination
