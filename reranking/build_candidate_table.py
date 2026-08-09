"""Build label-separated, frozen-pool candidate tables for CROG and Modular.

The adapters in this module intentionally preserve candidate IDs and geometry.
They only project existing artifacts into one schema; they never generate,
filter, move, rotate, resize, or relabel a grasp candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reranking.data_contracts import streaming_sha256
from reranking.models.tabular import assert_no_forbidden_columns, derive_q_features


LABEL_COLUMNS = (
    "candidate_correct",
    "best_iou_same_gt",
    "best_angle_error_same_gt",
    "baseline_top1_correct",
    "pool_has_positive",
)

# Raw Modular extraction artifacts retain these columns only because labels are
# joined after leakage-safe feature extraction.  They are evaluation targets or
# direct functions of ground truth and must never survive into a model table.
# Keep this deny-list aligned with
# ``HiFi_reproduction/src/grasping/reranking_v1/features.py`` without importing
# that sibling project at runtime.
MODULAR_GT_ALIAS_COLUMNS = frozenset(
    {
        "candidate_positive",
        "best_gt_id",
        "candidate_gt_iou",
        "candidate_gt_angle_error",
        "candidate_gt_angle_error_deg",
        "maximum_rectangle_iou_with_angle_gate",
        "legacy_geq_positive",
        "exact_iou_threshold_pair_count",
        "gt_mask",
        "gt_grasp",
        "first_valid_rank",
        "center_error",
        "angle_error",
        "iou_with_gt",
        "correct_candidate_id",
        "j_at_1",
        "j_at_any",
        "target_gt_object_id",
    }
)

IDENTITY_COLUMNS = (
    "sample_id",
    "scene_id",
    "frame_id",
    "expression_id",
    "candidate_id",
    "candidate_identity_sha256",
    "route",
    "pool_type",
)


class CandidateTableError(ValueError):
    """Raised when a source artifact violates frozen-pool identity."""


@dataclass(frozen=True)
class BuildResult:
    features_path: str
    labels_path: str | None
    query_universe_path: str
    feature_rows: int
    label_rows: int
    query_count: int
    empty_query_count: int
    input_sha256: str
    labels_sha256: str | None
    inference_features_sha256: str | None = None


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CandidateTableError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise CandidateTableError(
                    f"expected JSON object at {path}:{line_number}"
                )
            yield value


def _feature_value(value: Any) -> tuple[float, int, float]:
    """Return value, missing flag, reliability from a CROG feature cell."""

    if isinstance(value, Mapping):
        raw = value.get("value")
        reliability = value.get("reliability", 0.0 if raw is None else 1.0)
    else:
        raw = value
        reliability = 0.0 if raw is None else 1.0
    if raw is None:
        numeric = 0.0
        missing = 1
    else:
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise CandidateTableError(f"non-numeric candidate feature: {raw!r}") from error
        if not math.isfinite(numeric):
            raise CandidateTableError(f"non-finite candidate feature: {raw!r}")
        missing = 0
    try:
        reliability_value = float(reliability)
    except (TypeError, ValueError):
        reliability_value = 0.0
    if not math.isfinite(reliability_value):
        reliability_value = 0.0
    return numeric, missing, reliability_value


def _crog_candidate_row(
    sample: Mapping[str, Any], candidate: Mapping[str, Any], *, pool_type: str
) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id", ""))
    candidate_id = str(candidate.get("candidate_id", ""))
    checksum = str(candidate.get("candidate_checksum", ""))
    if not sample_id or not candidate_id or not checksum:
        raise CandidateTableError("CROG candidate lacks sample/candidate/checksum identity")
    features = candidate.get("features", {})
    if not isinstance(features, Mapping):
        raise CandidateTableError("CROG candidate features must be an object")
    try:
        assert_no_forbidden_columns(tuple(map(str, features)))
    except ValueError as error:
        raise CandidateTableError(
            f"forbidden label-derived CROG candidate feature: {error}"
        ) from error
    diagnostics = candidate.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "scene_id": str(sample.get("scene_id", sample.get("frame_id", sample_id))),
        "frame_id": str(sample.get("scene_id", sample.get("frame_id", sample_id))),
        "expression_id": sample_id,
        "candidate_id": candidate_id,
        "candidate_identity_sha256": checksum,
        "route": "crog_native",
        "pool_type": pool_type,
        "language_instruction": str(sample.get("language_instruction", "")),
        "image_path": str(sample.get("image_path", "")),
        "depth_path": str(sample.get("depth_path", "")),
        "pcd_path": str(sample.get("pcd_path", "")),
        "original_rank": int(candidate.get("q_rank", candidate.get("legacy_rank", 0))) + 1,
        "q_raw": float(candidate.get("q_raw", 0.0)),
        "x_px": float(candidate.get("cx", candidate.get("col", 0.0))),
        "y_px": float(candidate.get("cy", candidate.get("row", 0.0))),
        "z_m": float(diagnostics.get("center_depth_m") or 0.0),
        "z_m_missing": int(diagnostics.get("center_depth_m") is None),
        "angle_rad": float(candidate.get("angle_rad", 0.0)),
        "width_px": float(candidate.get("width_px", 0.0)),
        "width_m": 0.0,
        "width_m_missing": 1,
        "height_px": float(candidate.get("height_px", 0.0)),
    }
    for name, cell in sorted(features.items()):
        value, missing, reliability = _feature_value(cell)
        row[str(name)] = value
        row[f"{name}_missing"] = missing
        row[f"{name}_reliability"] = reliability
    for source, destination in (
        ("angle_concentration", "angle_concentration"),
        ("center_alignment", "center_alignment"),
        ("depth_valid_fraction", "crop_valid_ratio"),
        ("local_depth_median_m", "crop_median"),
        ("nearest_obstacle_distance_px", "border_clearance_px"),
        ("object_width_px", "mask_thickness_px"),
        ("valid_scanline_fraction", "axis_valid_ratio"),
        ("z_ref_m", "z_reference"),
    ):
        value, missing, _ = _feature_value(diagnostics.get(source))
        row[destination] = value
        row[f"{destination}_missing"] = missing
    return row


def _crog_label_rows(
    feature_sample: Mapping[str, Any], label_sample: Mapping[str, Any]
) -> list[dict[str, Any]]:
    sample_id = str(feature_sample.get("sample_id", ""))
    # Corrected CROG test labels use a stable namespaced sample ID while the
    # frozen feature JSONL retains the integer source ID.  The label artifact
    # carries that integer explicitly as ``source_sample_id``; this is the
    # audited join key.  Train/validation labels retain the original ID and
    # therefore fall back to their own ``sample_id``.
    label_source_id = str(
        label_sample.get("source_sample_id", label_sample.get("sample_id", ""))
    )
    if sample_id != label_source_id:
        raise CandidateTableError(
            f"CROG feature/label source sample mismatch: {sample_id!r} != "
            f"{label_source_id!r}"
        )
    output_sample_id = _crog_output_sample_id(feature_sample, label_sample)
    candidates = feature_sample.get("candidates", [])
    labels = label_sample.get("candidate_labels", [])
    if len(candidates) != len(labels):
        raise CandidateTableError(f"CROG candidate count mismatch for {sample_id}")
    by_id = {str(value.get("candidate_id", "")): value for value in labels}
    if len(by_id) != len(labels):
        raise CandidateTableError(f"duplicate CROG label candidate ID for {sample_id}")
    correctness_by_id = {
        candidate_id: _crog_candidate_correct(
            value,
            sample_id=sample_id,
            candidate_id=candidate_id,
        )
        for candidate_id, value in by_id.items()
    }
    pool_has_positive = any(correctness_by_id.values())
    ordered_candidates = sorted(
        candidates,
        key=lambda value: (
            int(value.get("q_rank", value.get("legacy_rank", 0))),
            str(value.get("candidate_id", "")),
        ),
    )
    top_candidate_id = (
        None
        if not ordered_candidates
        else str(ordered_candidates[0].get("candidate_id", ""))
    )
    if top_candidate_id is not None and top_candidate_id not in correctness_by_id:
        raise CandidateTableError(
            f"missing CROG label {sample_id}/{top_candidate_id}"
        )
    derived_top1_correct = (
        False if top_candidate_id is None else correctness_by_id[top_candidate_id]
    )
    declared_top1_correct = label_sample.get("original_top1_correct")
    if declared_top1_correct is not None:
        if not isinstance(declared_top1_correct, bool):
            raise CandidateTableError(
                f"non-boolean CROG original_top1_correct for {sample_id}"
            )
        if declared_top1_correct != derived_top1_correct:
            raise CandidateTableError(
                f"conflicting CROG original_top1_correct for {sample_id}"
            )
    baseline_top1_correct = derived_top1_correct
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("candidate_id", ""))
        label = by_id.get(candidate_id)
        if label is None:
            raise CandidateTableError(f"missing CROG label {sample_id}/{candidate_id}")
        checksum = str(candidate.get("candidate_checksum", ""))
        if checksum != str(label.get("candidate_checksum", "")):
            raise CandidateTableError(f"CROG checksum mismatch {sample_id}/{candidate_id}")
        best = label.get("best_gt") or {}
        output.append(
            {
                "sample_id": output_sample_id,
                "candidate_id": candidate_id,
                "candidate_identity_sha256": checksum,
                "candidate_correct": correctness_by_id[candidate_id],
                "best_iou_same_gt": float(best.get("rectangle_iou", 0.0)),
                "best_angle_error_same_gt": float(
                    best.get("angle_difference_deg", math.inf)
                ),
                "baseline_top1_correct": baseline_top1_correct,
                "pool_has_positive": bool(pool_has_positive),
                "source_candidate_position": index,
                "evaluator_track": str(label_sample.get("evaluator_track", "")),
                "evaluator_version": str(label_sample.get("evaluator_version", "")),
            }
        )
    return output


def _crog_candidate_correct(
    label: Mapping[str, Any], *, sample_id: str, candidate_id: str
) -> bool:
    """Resolve CROG correctness without silently treating schema drift as false."""

    top_present = "candidate_correct" in label
    top_value = label.get("candidate_correct")
    if top_present and not isinstance(top_value, bool):
        raise CandidateTableError(
            f"non-boolean CROG candidate_correct for {sample_id}/{candidate_id}"
        )

    best = label.get("best_gt")
    nested_present = isinstance(best, Mapping) and "joint_success" in best
    nested_value = best.get("joint_success") if isinstance(best, Mapping) else None
    if nested_present and not isinstance(nested_value, bool):
        raise CandidateTableError(
            f"non-boolean CROG best_gt.joint_success for {sample_id}/{candidate_id}"
        )
    if nested_present:
        try:
            geometry_value = bool(
                float(best["rectangle_iou"]) > 0.25
                and float(best["angle_difference_deg"]) <= 30.0
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CandidateTableError(
                "incomplete CROG best_gt geometry for "
                f"{sample_id}/{candidate_id}"
            ) from error
        if nested_value != geometry_value:
            raise CandidateTableError(
                "CROG best_gt.joint_success disagrees with evaluator geometry for "
                f"{sample_id}/{candidate_id}"
            )

    if top_present and nested_present and top_value != nested_value:
        raise CandidateTableError(
            f"conflicting CROG correctness labels for {sample_id}/{candidate_id}"
        )
    if top_present:
        return bool(top_value)
    if nested_present:
        return bool(nested_value)
    raise CandidateTableError(
        f"missing CROG correctness label for {sample_id}/{candidate_id}"
    )


def _crog_output_sample_id(
    feature_sample: Mapping[str, Any], label_sample: Mapping[str, Any]
) -> str:
    """Use corrected stable IDs and namespace legacy split-local integers."""

    source_id = str(feature_sample.get("sample_id", ""))
    label_id = str(label_sample.get("sample_id", source_id))
    if ":" in label_id:
        return label_id
    split = str(feature_sample.get("split", label_sample.get("split", ""))).strip()
    if split:
        try:
            identifier = f"{int(source_id):08d}"
        except ValueError:
            identifier = source_id
        return f"crog:{split}:{identifier}"
    return label_id


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def build_crog_candidate_tables(
    features_jsonl: str | os.PathLike[str],
    output_features: str | os.PathLike[str],
    *,
    labels_jsonl: str | os.PathLike[str] | None = None,
    output_labels: str | os.PathLike[str] | None = None,
    output_query_universe: str | os.PathLike[str] | None = None,
    pool_type: str = "frozen_top5",
) -> BuildResult:
    """Stream CROG nested JSONL into label-separated Parquet tables."""

    source = Path(features_jsonl).resolve()
    label_source = Path(labels_jsonl).resolve() if labels_jsonl is not None else None
    feature_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    label_iterator = _jsonl(label_source) if label_source is not None else None
    for sample in _jsonl(source):
        label_sample: dict[str, Any] | None = None
        source_sample = sample
        if label_iterator is not None:
            try:
                label_sample = next(label_iterator)
            except StopIteration as error:
                raise CandidateTableError("CROG label JSONL ended early") from error
            source_id = str(sample.get("sample_id", ""))
            label_source_id = str(
                label_sample.get(
                    "source_sample_id", label_sample.get("sample_id", "")
                )
            )
            if source_id != label_source_id:
                raise CandidateTableError(
                    f"CROG feature/label source sample mismatch: "
                    f"{source_id!r} != {label_source_id!r}"
                )
            # Normalize the output table to the corrected stable sample ID.
            # Candidate identities and source order remain untouched.
            sample = dict(sample)
            sample["sample_id"] = _crog_output_sample_id(source_sample, label_sample)
        candidates = sample.get("candidates", [])
        if not isinstance(candidates, list):
            raise CandidateTableError("CROG candidates must be a list")
        if pool_type == "frozen_top5" and len(candidates) > 5:
            candidates = sorted(
                candidates,
                key=lambda value: (
                    int(value.get("q_rank", value.get("legacy_rank", 0))),
                    str(value.get("candidate_id", "")),
                ),
            )[:5]
            sample = dict(sample)
            sample["candidates"] = candidates
        rows = [_crog_candidate_row(sample, candidate, pool_type=pool_type) for candidate in candidates]
        if rows:
            derived = derive_q_features(
                pd.DataFrame(rows), query_col="sample_id", q_col="q_raw", rank_col="original_rank"
            )
            feature_rows.extend(derived.to_dict(orient="records"))
        sample_id = str(sample.get("sample_id", ""))
        scene_id = str(sample.get("scene_id", sample_id))
        queries.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "frame_id": scene_id,
                "candidate_count": len(candidates),
                "empty_query": len(candidates) == 0,
            }
        )
        if label_sample is not None:
            label_rows.extend(_crog_label_rows(source_sample, label_sample))
    if label_iterator is not None:
        try:
            next(label_iterator)
        except StopIteration:
            pass
        else:
            raise CandidateTableError("CROG label JSONL has extra samples")
    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.duplicated(["sample_id", "candidate_id"]).any():
        raise CandidateTableError("duplicate CROG candidate key")
    output_features_path = Path(output_features).resolve()
    _atomic_parquet(output_features_path, feature_frame)
    label_path: Path | None = None
    if label_source is not None:
        if output_labels is None:
            raise CandidateTableError("output_labels is required with labels_jsonl")
        label_path = Path(output_labels).resolve()
        label_frame = pd.DataFrame(label_rows)
        if set(map(tuple, feature_frame[["sample_id", "candidate_id"]].to_numpy())) != set(
            map(tuple, label_frame[["sample_id", "candidate_id"]].to_numpy())
        ):
            raise CandidateTableError("CROG feature and label candidate pools differ")
        _atomic_parquet(label_path, label_frame)
    universe_path = Path(
        output_query_universe
        if output_query_universe is not None
        else output_features_path.with_name(f"{output_features_path.stem}_queries.parquet")
    ).resolve()
    _atomic_parquet(universe_path, pd.DataFrame(queries))
    return BuildResult(
        features_path=str(output_features_path),
        labels_path=None if label_path is None else str(label_path),
        query_universe_path=str(universe_path),
        feature_rows=len(feature_frame),
        label_rows=len(label_rows),
        query_count=len(queries),
        empty_query_count=sum(int(value["empty_query"]) for value in queries),
        input_sha256=streaming_sha256(source),
        labels_sha256=None if label_source is None else streaming_sha256(label_source),
    )


def build_modular_candidate_tables(
    candidate_parquet: str | os.PathLike[str],
    output_features: str | os.PathLike[str],
    output_labels: str | os.PathLike[str],
    *,
    pool_type: str = "full_post_filter",
    query_universe: pd.DataFrame | str | os.PathLike[str] | None = None,
    inference_features: str | os.PathLike[str] | None = None,
) -> BuildResult:
    """Project frozen Modular labels and optional rich inference features.

    ``candidate_parquet`` remains the canonical same-GT label/geometry source.
    When ``inference_features`` is supplied, it must cover the exact same
    candidate IDs and agree with the frozen q/rank/pose fields.  This lets the
    retained test pool use the same leakage-safe scalar schema as development
    without copying label columns into the inference extractor.
    """

    source = Path(candidate_parquet).resolve()
    frame = pd.read_parquet(source)
    rename = {
        "gqcnn_q_value": "q_raw",
        "gqcnn_rank": "original_rank",
        "original_gqcnn_rank": "original_rank",
        "center_u_px": "x_px",
        "center_v_px": "y_px",
        "center_depth_m": "z_m",
        "configured_width_px": "width_px",
        "candidate_success": "candidate_correct",
        "joint_success": "candidate_correct",
        "candidate_positive": "candidate_correct",
        "rectangle_iou": "best_iou_same_gt",
        "candidate_gt_iou": "best_iou_same_gt",
        "angle_difference_deg": "best_angle_error_same_gt",
        "candidate_gt_angle_error": "best_angle_error_same_gt",
        "candidate_gt_angle_error_deg": "best_angle_error_same_gt",
    }
    for old, new in rename.items():
        if old in frame.columns and new not in frame.columns:
            frame[new] = frame[old]
    if {"candidate_success", "joint_success"} <= set(frame.columns):
        left = frame["candidate_success"].astype(bool).to_numpy()
        right = frame["joint_success"].astype(bool).to_numpy()
        if not np.array_equal(left, right):
            raise CandidateTableError(
                "conflicting Modular candidate_success and joint_success labels"
            )
    inference_path: Path | None = None
    if inference_features is not None:
        inference_path = Path(inference_features).resolve()
        rich = pd.read_parquet(inference_path)
        for old, new in rename.items():
            if old in rich.columns and new not in rich.columns:
                rich[new] = rich[old]
        forbidden_rich = sorted(
            {
                "candidate_success",
                "joint_success",
                "candidate_positive",
                "candidate_correct",
                "best_iou_same_gt",
                "best_angle_error_same_gt",
                "rectangle_iou",
                "angle_difference_deg",
            }
            & set(rich.columns)
        )
        if forbidden_rich:
            raise CandidateTableError(
                "rich Modular inference table contains labels: "
                f"{forbidden_rich}"
            )
        keys = ["sample_id", "candidate_id"]
        for name, candidate_frame in (("canonical", frame), ("rich", rich)):
            missing_keys = sorted(set(keys) - set(candidate_frame.columns))
            if missing_keys:
                raise CandidateTableError(
                    f"{name} Modular table missing keys: {missing_keys}"
                )
            for key in keys:
                candidate_frame[key] = candidate_frame[key].astype(str)
            if candidate_frame.duplicated(keys).any():
                raise CandidateTableError(
                    f"{name} Modular table has duplicate candidate keys"
                )
        canonical_keys = pd.MultiIndex.from_frame(frame[keys])
        rich_keys = pd.MultiIndex.from_frame(rich[keys])
        if set(canonical_keys) != set(rich_keys):
            raise CandidateTableError(
                "rich/canonical Modular candidate key sets differ"
            )
        frozen_fields = (
            "q_raw",
            "original_rank",
            "x_px",
            "y_px",
            "z_m",
            "angle_rad",
            "width_m",
            "width_px",
        )
        missing_canonical_fields = sorted(set(frozen_fields) - set(frame.columns))
        if missing_canonical_fields:
            raise CandidateTableError(
                "canonical Modular table missing frozen fields: "
                f"{missing_canonical_fields}"
            )
        canonical_projection = frame[[*keys, *frozen_fields]].rename(
            columns={name: f"{name}__canonical" for name in frozen_fields}
        )
        checked = rich.merge(
            canonical_projection,
            on=keys,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        required_rich_fields = {"q_raw", "original_rank", "width_m", "width_px"}
        for name in frozen_fields:
            if name not in rich.columns:
                if name in required_rich_fields:
                    raise CandidateTableError(
                        f"rich Modular table missing frozen field: {name}"
                    )
                checked[name] = checked[f"{name}__canonical"]
            observed = pd.to_numeric(
                checked[name], errors="coerce"
            ).to_numpy(np.float64)
            expected = pd.to_numeric(
                checked[f"{name}__canonical"], errors="coerce"
            ).to_numpy(np.float64)
            if not np.isfinite(observed).all() or not np.isfinite(expected).all():
                raise CandidateTableError(
                    f"non-finite rich/canonical Modular field: {name}"
                )
            if name == "original_rank":
                agrees = np.array_equal(observed.astype(np.int64), expected.astype(np.int64))
            else:
                agrees = np.array_equal(
                    observed.astype(np.float32), expected.astype(np.float32)
                )
            if not agrees:
                raise CandidateTableError(
                    f"rich Modular inference disagrees with canonical {name}"
                )
        rich = checked.drop(
            columns=[f"{name}__canonical" for name in frozen_fields]
        )
        label_columns = (
            "candidate_correct",
            "best_iou_same_gt",
            "best_angle_error_same_gt",
        )
        missing_labels = sorted(set(label_columns) - set(frame.columns))
        if missing_labels:
            raise CandidateTableError(
                f"canonical Modular labels missing: {missing_labels}"
            )
        frame = rich.merge(
            frame[[*keys, *label_columns]],
            on=keys,
            how="left",
            validate="one_to_one",
            sort=False,
        )
    required = {
        "sample_id",
        "scene_id",
        "candidate_id",
        "q_raw",
        "original_rank",
        "x_px",
        "y_px",
        "angle_rad",
        "width_m",
        "width_px",
        "candidate_correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CandidateTableError(f"Modular table missing columns: {missing}")
    if pool_type == "frozen_top5":
        frame = frame.loc[pd.to_numeric(frame["original_rank"]).le(5)].copy()
    frame["frame_id"] = frame.get("frame_id", frame["scene_id"]).astype(str)
    frame["expression_id"] = frame.get("question_index", frame["sample_id"]).astype(str)
    frame["candidate_identity_sha256"] = frame.get(
        "candidate_identity_sha256",
        frame["sample_id"].astype(str) + "/" + frame["candidate_id"].astype(str),
    )
    frame["route"] = "modular_native"
    frame["pool_type"] = pool_type
    frame["z_m"] = frame.get("z_m", 0.0)
    frame["candidate_correct"] = frame["candidate_correct"].astype(bool)
    frame["pool_has_positive"] = frame.groupby("sample_id")["candidate_correct"].transform("max")
    baseline = (
        frame.sort_values(["sample_id", "original_rank", "candidate_id"], kind="mergesort")
        .groupby("sample_id", sort=False)["candidate_correct"]
        .first()
    )
    frame["baseline_top1_correct"] = frame["sample_id"].map(baseline).astype(bool)
    if frame.duplicated(["sample_id", "candidate_id"]).any():
        raise CandidateTableError("duplicate Modular candidate key")
    labels = frame[
        [
            "sample_id",
            "candidate_id",
            "candidate_identity_sha256",
            "candidate_correct",
            "best_iou_same_gt",
            "best_angle_error_same_gt",
            "baseline_top1_correct",
            "pool_has_positive",
        ]
    ].copy()
    forbidden = {
        "candidate_success",
        "joint_success",
        "candidate_positive",
        "best_gt_id",
        "best_gt_index",
        "rectangle_iou",
        "angle_difference_deg",
        "iou_ok",
        "angle_ok",
        "failure_mode",
        "evaluator_version",
        *MODULAR_GT_ALIAS_COLUMNS,
        *LABEL_COLUMNS,
    }
    features = frame.drop(columns=[name for name in forbidden if name in frame.columns])
    features = derive_q_features(
        features, query_col="sample_id", q_col="q_raw", rank_col="original_rank"
    )
    try:
        assert_no_forbidden_columns(tuple(map(str, features.columns)))
    except ValueError as error:
        raise CandidateTableError(
            f"Modular feature table retains ground-truth/label columns: {error}"
        ) from error
    output_features_path = Path(output_features).resolve()
    output_labels_path = Path(output_labels).resolve()
    _atomic_parquet(output_features_path, features)
    _atomic_parquet(output_labels_path, labels)
    observed = (
        frame.groupby("sample_id", sort=True)
        .agg(
            scene_id=("scene_id", "first"),
            frame_id=("frame_id", "first"),
            candidate_count=("candidate_id", "size"),
        )
        .reset_index()
    )
    observed["sample_id"] = observed["sample_id"].astype(str)
    if query_universe is None:
        queries = observed
    else:
        if isinstance(query_universe, pd.DataFrame):
            queries = query_universe.copy()
        else:
            universe_path = Path(query_universe)
            if universe_path.suffix.lower() == ".parquet":
                queries = pd.read_parquet(universe_path)
            elif universe_path.suffix.lower() == ".csv":
                queries = pd.read_csv(universe_path)
            elif universe_path.suffix.lower() in {".jsonl", ".ndjson"}:
                queries = pd.read_json(universe_path, lines=True)
            else:
                raise CandidateTableError(
                    f"unsupported Modular query universe: {universe_path}"
                )
        if "sample_id" not in queries.columns:
            raise CandidateTableError("Modular query universe lacks sample_id")
        queries["sample_id"] = queries["sample_id"].astype(str)
        if queries["sample_id"].eq("").any() or queries.duplicated("sample_id").any():
            raise CandidateTableError(
                "Modular query universe sample IDs must be unique and non-empty"
            )
        missing = sorted(set(observed["sample_id"]) - set(queries["sample_id"]))
        if missing:
            raise CandidateTableError(
                f"Modular query universe omits observed candidates: {missing[:5]}"
            )
        counts = observed.set_index("sample_id")["candidate_count"]
        observed_scene = observed.set_index("sample_id")["scene_id"]
        for column in ("scene_id", "frame_id"):
            if column not in queries.columns:
                queries[column] = queries["sample_id"].map(observed_scene)
            queries[column] = queries[column].where(
                queries[column].notna(), queries["sample_id"]
            ).astype(str)
        queries["candidate_count"] = (
            queries["sample_id"].map(counts).fillna(0).astype(np.int64)
        )
        queries = queries[
            ["sample_id", "scene_id", "frame_id", "candidate_count"]
        ].copy()
    queries["empty_query"] = queries["candidate_count"].eq(0)
    universe_path = output_features_path.with_name(f"{output_features_path.stem}_queries.parquet")
    _atomic_parquet(universe_path, queries)
    return BuildResult(
        features_path=str(output_features_path),
        labels_path=str(output_labels_path),
        query_universe_path=str(universe_path),
        feature_rows=len(features),
        label_rows=len(labels),
        query_count=len(queries),
        empty_query_count=int(queries["empty_query"].sum()),
        input_sha256=streaming_sha256(source),
        labels_sha256=streaming_sha256(source),
        inference_features_sha256=(
            None if inference_path is None else streaming_sha256(inference_path)
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("crog", "modular"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-query-universe", type=Path)
    parser.add_argument("--pool", default="frozen_top5")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.route == "crog":
        result = build_crog_candidate_tables(
            args.input,
            args.output_features,
            labels_jsonl=args.labels,
            output_labels=args.output_labels,
            output_query_universe=args.output_query_universe,
            pool_type=args.pool,
        )
    else:
        result = build_modular_candidate_tables(
            args.input, args.output_features, args.output_labels, pool_type=args.pool
        )
    print(json.dumps(result.__dict__, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
