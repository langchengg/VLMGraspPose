"""Adapt, freeze, calibrate, and combine G1/C1 candidate pools."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.grasping.common.geometry import rotated_rectangle_iou

from .contracts import (
    candidate_identity_sha256,
    identity_table_sha256,
    validate_frozen_candidates,
)


SOURCE_INFERENCE_COLUMNS = (
    "method",
    "sample_id",
    "scene_id",
    "candidate_id",
    "rank",
    "center_x",
    "center_y",
    "angle_deg",
    "width_px",
    "height_px",
    "score",
    "candidate_metadata_json",
)
SOURCE_LABEL_COLUMNS = (
    "sample_id",
    "candidate_id",
    "candidate_success",
    "best_rectangle_iou",
    "best_angle_difference_deg",
)


def _metadata(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, Mapping):
        raise ValueError("candidate metadata must be an object")
    return parsed


def adapt_source_candidates(
    source: pd.DataFrame,
    *,
    backend: str,
    split: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    missing = sorted(set(SOURCE_INFERENCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"source candidate table missing columns: {missing}")
    if backend not in {"G1", "C1"}:
        raise ValueError("backend must be G1 or C1")
    observed_methods = set(source["method"].astype(str))
    aliases = {
        "G1": lambda value: value == "G1" or "grconvnet" in value.lower(),
        "C1": lambda value: value == "C1" or "ggcnn2" in value.lower(),
    }
    if not observed_methods or not all(aliases[backend](value) for value in observed_methods):
        raise ValueError(
            f"source method/backend mismatch: expected {backend}, got {sorted(observed_methods)}"
        )
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    selected = (
        source.copy()
        if top_k is None
        else source.loc[source["rank"].astype(int) <= int(top_k)].copy()
    )
    metadata = selected["candidate_metadata_json"].map(_metadata)
    selected["backend"] = backend
    selected["split"] = str(split)
    selected["source_candidate_id"] = selected["candidate_id"].astype(str)
    selected["stable_candidate_id"] = (
        backend.lower() + ":" + selected["source_candidate_id"]
    )
    selected["original_rank"] = selected["rank"].astype(int)
    selected["original_score"] = selected["score"].astype(float)
    selected["raw_network_quality"] = metadata.map(
        lambda row: float(row.get("network_quality", math.nan))
    )
    selected["stored_center_mask_support"] = metadata.map(
        lambda row: float(row.get("center_mask_support", math.nan))
    )
    selected["stored_jaw_mask_support"] = metadata.map(
        lambda row: float(row.get("jaw_mask_support", math.nan))
    )
    selected["source_row"] = metadata.map(
        lambda row: int(row.get("source_row", -1))
    )
    selected["source_column"] = metadata.map(
        lambda row: int(row.get("source_column", -1))
    )
    selected["candidate_identity_sha256"] = selected.apply(
        candidate_identity_sha256, axis=1
    )
    columns = [
        "sample_id",
        "scene_id",
        "split",
        "backend",
        "source_candidate_id",
        "stable_candidate_id",
        "original_rank",
        "original_score",
        "center_x",
        "center_y",
        "angle_deg",
        "width_px",
        "height_px",
        "raw_network_quality",
        "stored_center_mask_support",
        "stored_jaw_mask_support",
        "source_row",
        "source_column",
        "candidate_identity_sha256",
    ]
    output = selected.loc[:, columns].reset_index(drop=True)
    validate_frozen_candidates(
        output,
        require_top5=top_k is not None and top_k <= 5,
    )
    return output


def adapt_source_labels(source: pd.DataFrame, *, backend: str) -> pd.DataFrame:
    missing = sorted(set(SOURCE_LABEL_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"source label table missing columns: {missing}")
    output = source.loc[:, SOURCE_LABEL_COLUMNS].copy()
    output["stable_candidate_id"] = (
        str(backend).lower() + ":" + output["candidate_id"].astype(str)
    )
    output = output.rename(columns={"candidate_success": "candidate_correct"})
    return output.drop(columns=["candidate_id"])


def periodic_angle_difference_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 90.0) % 180.0 - 90.0)


def _grasp_view(row: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        center_x=float(row["center_x"]),
        center_y=float(row["center_y"]),
        angle_deg=float(row["angle_deg"]),
        width_px=float(row["width_px"]),
        height_px=float(row["height_px"]),
    )


def cross_backend_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    center = float(
        np.linalg.norm(
            np.array([left["center_x"], left["center_y"]], dtype=float)
            - np.array([right["center_x"], right["center_y"]], dtype=float)
        )
    )
    angle = periodic_angle_difference_deg(left["angle_deg"], right["angle_deg"])
    width = abs(float(left["width_px"]) - float(right["width_px"]))
    iou = float(rotated_rectangle_iou(_grasp_view(left), _grasp_view(right)))
    matched = bool(
        angle <= 10.0
        and ((center <= 4.0 and width <= 5.0) or iou >= 0.5)
    )
    return {
        "matched": matched,
        "cross_backend_rotated_iou": iou,
        "cross_backend_center_distance": center,
        "cross_backend_angle_difference": angle,
        "cross_backend_width_difference": width,
    }


def build_raw_union(g1: pd.DataFrame, c1: pd.DataFrame) -> pd.DataFrame:
    validate_frozen_candidates(g1, require_top5=False)
    validate_frozen_candidates(c1, require_top5=False)
    if set(g1["sample_id"].astype(str)) != set(c1["sample_id"].astype(str)):
        # Empty candidate samples are absent from both candidate tables.  Unequal
        # non-empty coverage is valid, but both must originate from one universe.
        pass
    output = pd.concat([g1, c1], ignore_index=True)
    output["canonical_candidate_id"] = output["stable_candidate_id"]
    output["source_backend"] = output["backend"]
    output["source_score"] = output["original_score"]
    output["source_rank"] = output["original_rank"]
    output["supported_by_g1"] = output["backend"].eq("G1")
    output["supported_by_c1"] = output["backend"].eq("C1")
    output["matched_other_backend_candidate_id"] = None
    output["cross_backend_rotated_iou"] = 0.0
    output["cross_backend_center_distance"] = math.inf
    output["cross_backend_angle_difference"] = 90.0
    output["cross_backend_width_difference"] = math.inf
    output["provenance_json"] = output.apply(
        lambda row: json.dumps(
            [
                {
                    "backend": row["backend"],
                    "source_candidate_id": row["source_candidate_id"],
                    "stable_candidate_id": row["stable_candidate_id"],
                    "source_rank": int(row["original_rank"]),
                    "source_score": float(row["original_score"]),
                }
            ],
            sort_keys=True,
            separators=(",", ":"),
        ),
        axis=1,
    )
    return output


def build_deduplicated_union(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"source_score_calibrated", "canonical_candidate_id"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"calibrated union missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for sample_id, group in raw.groupby("sample_id", sort=False):
        records = group.to_dict(orient="records")
        pair_evidence: dict[tuple[int, int], dict[str, Any]] = {}
        for left in range(len(records)):
            for right in range(left + 1, len(records)):
                if records[left]["backend"] == records[right]["backend"]:
                    continue
                evidence = cross_backend_match(records[left], records[right])
                if evidence["matched"]:
                    pair_evidence[(left, right)] = evidence
        # Deterministic one-to-one cross-backend matching avoids transitive
        # G1-C1-G1 components that could silently delete two distinct grasps
        # from the same backend.
        matched_indices: set[int] = set()
        components: list[list[int]] = []
        for (left, right), evidence in sorted(
            pair_evidence.items(),
            key=lambda item: (
                -float(item[1]["cross_backend_rotated_iou"]),
                float(item[1]["cross_backend_center_distance"]),
                float(item[1]["cross_backend_angle_difference"]),
                str(records[item[0][0]]["stable_candidate_id"]),
                str(records[item[0][1]]["stable_candidate_id"]),
            ),
        ):
            if left in matched_indices or right in matched_indices:
                continue
            components.append([left, right])
            matched_indices.update((left, right))
        components.extend([[index] for index in range(len(records)) if index not in matched_indices])
        for component in components:
            chosen = min(
                component,
                key=lambda index: (
                    -float(records[index]["source_score_calibrated"]),
                    str(records[index]["backend"]),
                    int(records[index]["original_rank"]),
                    str(records[index]["stable_candidate_id"]),
                ),
            )
            record = dict(records[chosen])
            members = [records[index] for index in component]
            record["supported_by_g1"] = any(item["backend"] == "G1" for item in members)
            record["supported_by_c1"] = any(item["backend"] == "C1" for item in members)
            record["canonical_candidate_id"] = str(record["stable_candidate_id"])
            other = [item for item in members if item["backend"] != record["backend"]]
            if other:
                matched = min(
                    other,
                    key=lambda item: (
                        -float(item["source_score_calibrated"]),
                        str(item["stable_candidate_id"]),
                    ),
                )
                record["matched_other_backend_candidate_id"] = matched["stable_candidate_id"]
                evidence = cross_backend_match(record, matched)
                if not evidence["matched"]:
                    raise AssertionError("deduplicated pair lacks a direct NMS match")
                for key, value in evidence.items():
                    if key != "matched":
                        record[key] = value
            record["provenance_json"] = json.dumps(
                [
                    {
                        "backend": item["backend"],
                        "stable_candidate_id": item["stable_candidate_id"],
                        "source_candidate_id": item["source_candidate_id"],
                        "source_rank": int(item["original_rank"]),
                        "source_score": float(item["original_score"]),
                        "source_score_calibrated": float(item["source_score_calibrated"]),
                    }
                    for item in sorted(
                        members,
                        key=lambda item: (str(item["backend"]), int(item["original_rank"])),
                    )
                ],
                sort_keys=True,
                separators=(",", ":"),
            )
            rows.append(record)
    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["sample_id", "source_score_calibrated", "canonical_candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output["union_rank"] = output.groupby("sample_id", sort=False).cumcount() + 1
    output["pool_rank"] = output["union_rank"]
    output["pool_score"] = output["source_score_calibrated"]
    validate_frozen_candidates(output, require_top5=False, allow_union=True)
    return output


def pool_manifest(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "candidate_rows": int(len(frame)),
        "non_empty_samples": int(frame["sample_id"].nunique()),
        "candidate_identity_sha256": identity_table_sha256(frame),
        "maximum_candidates_per_sample": int(
            frame.groupby("sample_id").size().max() if not frame.empty else 0
        ),
    }
