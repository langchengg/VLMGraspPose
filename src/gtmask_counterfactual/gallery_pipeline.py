"""Canonical two-stage qualitative gallery with no caller-selected cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from HiFi_reproduction.src.grasping.backends.conditioning import (
    resize_probability_to_native,
)

from .candidate_matching import geometry_equivalence
from .contracts import RunState
from .execution_contracts import canonical_semantic_contracts
from .galleries import (
    BOARD_PANELS,
    CASE_CATEGORIES,
    build_eligible_table,
    deterministic_medoid_selection,
    render_case_board,
)
from .independent import gt_corners_to_canonical
from .io import (
    artifact_record,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from .protocol import LOCK_RELATIVE_PATH, load_execution_authority, verify_protocol_lock
from .resource import (
    collect_fresh_three_by_five_gate,
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)
from .visual_assets import validate_visual_asset_registry


EXPECTED_SAMPLE_COUNT = 7_675
QUOTA_PER_ROUTE_CATEGORY = 2
BUILD_RELATIVE_PATH = Path("14_galleries/GALLERY_BUILD_MANIFEST.json")
FINAL_RELATIVE_PATH = Path("14_galleries/GALLERY_MANIFEST.json")
MANUAL_ACCEPTANCE_RELATIVE_PATH = Path("14_galleries/MANUAL_QA_ACCEPTANCE.json")
_GEOMETRY = ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")
_FEATURES = tuple(
    canonical_semantic_contracts()["case_selection"]["feature_vector_schema"]
)


class GalleryContractError(RuntimeError):
    """Canonical gallery evidence is absent, stale, or caller-influenced."""


def _object(path: Path, *, name: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise GalleryContractError(f"{name} is absent or unsafe: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GalleryContractError(f"cannot parse {name}: {source}") from error
    if not isinstance(value, dict):
        raise GalleryContractError(f"{name} must contain one JSON object")
    return value


def _self_hash(value: Mapping[str, Any], *, name: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop("content_sha256", None)
    if observed != canonical_sha256(unsigned):
        raise GalleryContractError(f"{name} content hash differs")


def _record(record: Mapping[str, Any], *, root: Path, name: str) -> Path:
    if not isinstance(record, Mapping):
        raise GalleryContractError(f"{name} is not an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{name} escapes the counterfactual run") from error
    if dict(record) != artifact_record(path):
        raise GalleryContractError(f"{name} artifact differs")
    return path


def _external_record(record: Mapping[str, Any], *, name: str) -> Path:
    if not isinstance(record, Mapping):
        raise GalleryContractError(f"{name} is not an artifact record")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise GalleryContractError(f"{name} is absent or unsafe")
    if (
        sha256_file(path) != record.get("sha256")
        or int(record.get("bytes", -1)) != path.stat().st_size
    ):
        raise GalleryContractError(f"{name} artifact differs")
    return path


def _pipeline(root: Path) -> dict[str, Any]:
    return _object(root / "pipeline_status.json", name="pipeline status")


def _postprocess_graph(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output_path = root / "08_metrics/POSTPROCESS_MANIFEST.json"
    output = _object(output_path, name="postprocess manifest")
    _self_hash(output, name="postprocess manifest")
    if output.get("status") not in {"COMPLETE", "PARTIAL"}:
        raise GalleryContractError("postprocess is not complete enough for galleries")
    inputs_path = _record(
        output.get("postprocess_inputs"), root=root, name="postprocess inputs"
    )
    inputs = _object(inputs_path, name="postprocess inputs")
    _self_hash(inputs, name="postprocess inputs")
    if inputs.get("protocol_lock") != artifact_record(root / LOCK_RELATIVE_PATH):
        raise GalleryContractError("postprocess protocol binding differs")
    return output, inputs


def _read_frame(record: Mapping[str, Any], *, root: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(_record(record, root=root, name=name))


def _normal(value: Any) -> Any:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return _normal(value.item())
    if isinstance(value, np.ndarray):
        return [_normal(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normal(child) for key, child in value.items()}
    return value


def _source_selector_decisions(
    root: Path,
    inputs: Mapping[str, Any],
    *,
    routes: Sequence[str],
    denominator: set[str],
) -> pd.DataFrame:
    authority_path = _record(
        inputs["artifacts"]["final_outcomes_authority"],
        root=root,
        name="final outcomes authority",
    )
    authority = _object(authority_path, name="final outcomes authority")
    _self_hash(authority, name="final outcomes authority")
    selectors = authority.get("frozen_selector_contracts")
    if not isinstance(selectors, Mapping) or set(selectors) != set(routes):
        raise GalleryContractError("frozen selector contracts differ from routes")
    final = _read_frame(
        inputs["artifacts"]["final_outcomes"], root=root, name="final outcomes"
    )
    final["sample_id"] = final["sample_id"].astype(str)
    final["route"] = final["route"].astype(str).str.upper()
    rows: list[pd.DataFrame] = []
    for route in routes:
        contract = selectors[route]
        if not isinstance(contract, Mapping):
            raise GalleryContractError(f"{route} selector contract is malformed")
        source = _external_record(
            contract["source_artifact"], name=f"{route} frozen selector source"
        )
        frame = pd.read_parquet(source)
        system_column = "system_name" if "system_name" in frame else "system"
        frame = frame.loc[
            frame[system_column].astype(str).eq(str(contract["system"]))
        ].copy()
        if "is_selected" in frame and frame["sample_id"].astype(str).duplicated().any():
            keep = frame["is_selected"].astype(bool)
            if "no_output" in frame:
                # Formal candidate bundles carry one row per candidate plus a
                # single no-output decision row.  The latter is deliberately
                # not marked ``is_selected`` and must remain in the exact
                # denominator.
                keep |= frame["no_output"].astype(bool)
            frame = frame.loc[keep].copy()
        required = {"sample_id", "selected_candidate_id", "selected_correct"}
        if not required.issubset(frame.columns):
            raise GalleryContractError(
                f"{route} selector source misses {sorted(required.difference(frame.columns))}"
            )
        frame["sample_id"] = frame["sample_id"].astype(str)
        if frame.duplicated("sample_id").any() or set(frame["sample_id"]) != denominator:
            raise GalleryContractError(f"{route} selector source denominator differs")
        result = pd.DataFrame(
            {
                "sample_id": frame["sample_id"],
                "route": route,
                "system": str(contract["system"]),
                "selected_source_route": (
                    frame["selected_source_route"]
                    if "selected_source_route" in frame
                    else frame.get("selected_route", pd.Series(route, index=frame.index))
                ),
                "selected_candidate_id": frame["selected_candidate_id"],
                "selected_correct": frame["selected_correct"].astype(bool),
                "selector_score": (
                    pd.to_numeric(frame["formal_score"], errors="coerce")
                    if "formal_score" in frame
                    else np.nan
                ),
            }
        )
        result["selected_source_route"] = result["selected_source_route"].fillna("").astype(str).str.upper()
        result["selected_candidate_id"] = result["selected_candidate_id"].fillna("").astype(str)
        bearing = result["selected_candidate_id"].ne("")
        if not result.loc[bearing, "selected_source_route"].eq(route).all() or not result.loc[
            ~bearing, "selected_source_route"
        ].isin({"", route}).all():
            raise GalleryContractError(f"{route} selector source-route identity differs")
        observed = final.loc[final["route"].eq(route), ["sample_id", "final_correct"]]
        compared = observed.merge(
            result[["sample_id", "selected_correct"]],
            on="sample_id",
            how="outer",
            validate="one_to_one",
        )
        if not compared["final_correct"].astype(bool).eq(
            compared["selected_correct"].astype(bool)
        ).all():
            raise GalleryContractError(f"{route} frozen final bits differ")
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _native_by_sample(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    native = frame.loc[pd.to_numeric(frame["native_rank"], errors="coerce").eq(1)].copy()
    if native["sample_id"].astype(str).duplicated().any():
        raise GalleryContractError(f"{label} has duplicate native rank 1")
    return native.set_index(native["sample_id"].astype(str), drop=False)


def _candidate_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "candidate_id",
        "native_rank",
        "native_score",
        *_GEOMETRY,
        "candidate_success",
        "matched_gt_index",
        "best_same_gt_iou",
        "best_same_gt_angle_error_deg",
    ]
    return [
        {column: _normal(row[column]) for column in columns}
        for _, row in frame.sort_values("native_rank").iterrows()
    ]


def _case_id(postprocess_sha: str, sample_id: str, route: str, category: str) -> str:
    return canonical_sha256(
        {
            "schema_version": 1,
            "postprocess_manifest_sha256": postprocess_sha,
            "sample_id": sample_id,
            "route": route,
            "category": category,
        }
    )[:24]


def _first_rank(value: Any) -> float:
    return -1.0 if pd.isna(value) else float(value)


def _categories(
    row: Mapping[str, Any], *, geometry_match: bool, borderline_mapping: bool = False
) -> list[str]:
    if bool(row["technical_failure"]):
        return []
    result: list[str] = []
    taxonomy = str(row["native_taxonomy"])
    if taxonomy == "T4_grounding_limited":
        result.append("clear_grounding_limited")
    if taxonomy in {
        "T5_grounding_plus_selection_within_top5",
        "T6_grounding_plus_deep_ranking",
    }:
        result.append("grounding_plus_selection")
    if taxonomy == "T7_grasper_or_candidate_generation_limited":
        result.append("generator_limited_under_gt")
    if bool(row["pred_all_positive"]) and (
        not bool(row["pred_native_correct"]) or not bool(row["final_correct"])
    ):
        result.append("predicted_mask_ranking_limited")
    if bool(row["GT_mask_regression"]) or (
        not pd.isna(row["pred_first_positive_rank"])
        and not pd.isna(row["gt_first_positive_rank"])
        and float(row["gt_first_positive_rank"])
        > float(row["pred_first_positive_rank"])
    ):
        result.append("gt_mask_regression")
    if bool(row["pred_no_output"]) and not bool(row["gt_no_output"]):
        result.append("no_output_recovered_by_gt_mask")
    if bool(row["pred_native_correct"]) and bool(row["gt_native_correct"]) and geometry_match:
        result.append("no_change_success")
    if bool(row.get("annotation_suspect", False)) or borderline_mapping:
        result.append("borderline_annotation_sensitive")
    return result


def _asset_record(path_value: Any, sha_value: Any, *, name: str) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise GalleryContractError(f"{name} asset is absent or unsafe")
    if sha256_file(path) != str(sha_value):
        raise GalleryContractError(f"{name} asset hash differs")
    return artifact_record(path)


def _asset_metadata(
    visual: Mapping[str, Any], registry: Mapping[str, Any], *, ground_truth_record: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "rgb": _asset_record(visual["source_rgb_path"], visual["source_rgb_sha256"], name="RGB"),
        "depth": _asset_record(visual["source_depth_path"], visual["source_depth_sha256"], name="depth"),
        "pred_mask": _asset_record(visual["predicted_mask_path"], visual["predicted_mask_sha256"], name="predicted mask"),
        "gt_mask": _asset_record(registry["original_gt_mask_path"], registry["original_gt_mask_sha256"], name="GT mask"),
        "ground_truth": dict(ground_truth_record),
        "coordinate_frame": str(visual["coordinate_frame"]),
    }
    probability_status = str(visual["predicted_probability_status"])
    if probability_status == "AVAILABLE":
        result["pred_probability"] = _asset_record(
            visual["predicted_probability_path"],
            visual["predicted_probability_sha256"],
            name="predicted probability",
        )
    elif probability_status == "NOT_AVAILABLE":
        result["pred_probability"] = {"status": "NOT_AVAILABLE"}
    else:
        raise GalleryContractError("predicted probability availability is invalid")
    if result["coordinate_frame"] != "rgb_native":
        raise GalleryContractError("qualitative assets use a noncanonical frame")
    return result


def _canonical_cases(
    root: Path, *, expected_sample_count: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    verify_protocol_lock(root)
    output, inputs = _postprocess_graph(root)
    artifacts = inputs["artifacts"]
    routes = tuple(route for route, status in output["routes"].items() if status == "COMPLETE")
    visual_manifest, visual = validate_visual_asset_registry(
        root, artifacts["visual_assets"], expected_count=expected_sample_count
    )
    sample = _read_frame(artifacts["sample_manifest"], root=root, name="sample manifest")
    prompt_column = "language_prompt" if "language_prompt" in sample else "language"
    required_sample = {"sample_id", prompt_column}
    if not required_sample.issubset(sample.columns):
        raise GalleryContractError("sample manifest lacks qualitative identity fields")
    sample["sample_id"] = sample["sample_id"].astype(str)
    if len(sample) != expected_sample_count or sample["sample_id"].duplicated().any():
        raise GalleryContractError("gallery sample denominator differs")
    ground_truth = _read_frame(artifacts["ground_truth"], root=root, name="ground truth")
    grasp_column = "gt_grasp_rectangles" if "gt_grasp_rectangles" in ground_truth else "gt_grasp_list_json"
    if grasp_column not in ground_truth:
        raise GalleryContractError("ground truth lacks grasp rectangles")
    ground_truth["sample_id"] = ground_truth["sample_id"].astype(str)
    authority = load_execution_authority(root / LOCK_RELATIVE_PATH)
    registry_record = authority.get("gt_mask_registry")
    if not isinstance(registry_record, Mapping):
        raise GalleryContractError("protocol authority lacks GT registry")
    registry_path = _external_record(registry_record, name="GT mask registry")
    registry = pd.read_parquet(registry_path)
    required_registry = {
        "sample_id",
        "original_gt_mask_path",
        "original_gt_mask_sha256",
        "mapping_status",
        "pixel_qa_status",
    }
    if not required_registry.issubset(registry.columns):
        raise GalleryContractError("GT registry lacks visual asset fields")
    registry["sample_id"] = registry["sample_id"].astype(str)
    labels = _read_frame(output["artifacts"]["candidate_labels"], root=root, name="candidate labels")
    post = _read_frame(output["artifacts"]["post_r7_taxonomy"], root=root, name="post-R7 taxonomy")
    decisions = _source_selector_decisions(
        root, inputs, routes=routes, denominator=set(sample["sample_id"])
    )
    post = post.merge(
        decisions[["sample_id", "route", "selected_candidate_id", "selected_source_route", "selector_score"]],
        on=["sample_id", "route"],
        how="left",
        validate="one_to_one",
    )
    visual_index = visual.set_index("sample_id", drop=False)
    sample_index = sample.set_index("sample_id", drop=False)
    registry_index = registry.set_index("sample_id", drop=False)
    gt_index = ground_truth.set_index("sample_id", drop=False)
    ground_truth_record = artifact_record(_record(artifacts["ground_truth"], root=root, name="ground truth"))
    eligible_rows: list[dict[str, Any]] = []
    case_payloads: dict[str, Any] = {}
    for route in routes:
        route_rows = post.loc[post["route"].astype(str).eq(route)].copy()
        pred_route = labels.loc[
            labels["route"].astype(str).eq(route)
            & labels["branch"].astype(str).eq("predicted")
        ].copy()
        gt_route = labels.loc[
            labels["route"].astype(str).eq(route)
            & labels["branch"].astype(str).eq("gt_oracle")
        ].copy()
        pred_native = _native_by_sample(pred_route, label=f"{route} predicted")
        gt_native = _native_by_sample(gt_route, label=f"{route} GT")
        for row in route_rows.to_dict(orient="records"):
            sample_id = str(row["sample_id"])
            registry_row = registry_index.loc[sample_id].to_dict()
            pred_row = pred_native.loc[sample_id] if sample_id in pred_native.index else None
            gt_row = gt_native.loc[sample_id] if sample_id in gt_native.index else None
            geometry_match = bool(
                pred_row is not None
                and gt_row is not None
                and geometry_equivalence(pred_row, gt_row)["matched"]
            )
            categories = _categories(
                row,
                geometry_match=geometry_match,
                borderline_mapping=(
                    registry_row.get("resize_inverse_round_trip_below_reference")
                    is True
                ),
            )
            if not categories:
                continue
            visual_row = visual_index.loc[sample_id].to_dict()
            asset_meta = _asset_metadata(
                visual_row, registry_row, ground_truth_record=ground_truth_record
            )
            prompt = str(sample_index.loc[sample_id, prompt_column]).strip()
            presentation = bool(prompt) and str(registry_row["mapping_status"]) == "PASS" and str(
                registry_row["pixel_qa_status"]
            ) == "P2_MAPPING_QA_PASS"
            pred_sample = pred_route.loc[pred_route["sample_id"].astype(str).eq(sample_id)]
            gt_sample = gt_route.loc[gt_route["sample_id"].astype(str).eq(sample_id)]
            selected_id = str(row.get("selected_candidate_id") or "")
            if selected_id and selected_id not in set(pred_sample["candidate_id"].astype(str)):
                raise GalleryContractError(f"{route}/{sample_id} final candidate is absent")
            pred_native_id = "" if pred_row is None else str(pred_row["candidate_id"])
            gt_native_id = "" if gt_row is None else str(gt_row["candidate_id"])
            feature_values = {
                "pred_candidate_count": float(row["pred_candidate_count"]),
                "gt_candidate_count": float(row["gt_candidate_count"]),
                "pred_positive_candidate_count": float(row["pred_positive_candidate_count"]),
                "gt_positive_candidate_count": float(row["gt_positive_candidate_count"]),
                "pred_first_positive_rank_missing_minus_one": _first_rank(row["pred_first_positive_rank"]),
                "gt_first_positive_rank_missing_minus_one": _first_rank(row["gt_first_positive_rank"]),
                "predicted_mask_iou": float(row["predicted_mask_iou"]),
                "target_area_fraction": float(row["target_area_fraction"]),
                "mask_component_count": float(row["mask_component_count"]),
                "mask_boundary_complexity": float(row["mask_boundary_complexity"]),
                "valid_depth_ratio": float(row["valid_depth_ratio"]),
            }
            if tuple(feature_values) != _FEATURES or not np.isfinite(list(feature_values.values())).all():
                raise GalleryContractError("case feature vector contract differs")
            for category in categories:
                case_id = _case_id(
                    str(output["content_sha256"]), sample_id, route, category
                )
                payload = {
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "route": route,
                    "category": category,
                    "language_prompt": prompt,
                    "assets": asset_meta,
                    "pred_candidates": _candidate_payload(pred_sample),
                    "gt_candidates": _candidate_payload(gt_sample),
                    "gt_grasp_rectangles": _normal(gt_index.loc[sample_id, grasp_column]),
                    "native_candidate_id": pred_native_id,
                    "r7_candidate_id": selected_id,
                    "gt_candidate_id": gt_native_id,
                    "metrics": {str(key): _normal(value) for key, value in row.items()},
                    "selector": {
                        "source_route": str(row.get("selected_source_route") or ""),
                        "candidate_id": selected_id,
                        "score": _normal(row.get("selector_score")),
                    },
                    "coordinate_frame": "rgb_native",
                }
                bundle_hash = canonical_sha256(payload)
                case_payloads[case_id] = payload
                eligible_rows.append(
                    {
                        "case_id": case_id,
                        "sample_id": sample_id,
                        "route": route,
                        "category": category,
                        "predicate_version": "gtmask_case_predicates_v2",
                        "mechanism_pure": True,
                        "presentation_eligible": presentation,
                        "feature_vector_json": json.dumps(list(feature_values.values()), separators=(",", ":")),
                        "asset_bundle_sha256": bundle_hash,
                        "native_taxonomy": str(row["native_taxonomy"]),
                    }
                )
    eligible = build_eligible_table(pd.DataFrame(eligible_rows))
    selected, audit = deterministic_medoid_selection(
        eligible, quota_per_group=QUOTA_PER_ROUTE_CATEGORY
    )
    expected_groups = {(route, category) for route in routes for category in CASE_CATEGORIES}
    observed_groups = set(map(tuple, audit[["route", "category"]].astype(str).to_numpy()))
    if observed_groups != expected_groups:
        missing = sorted(expected_groups.difference(observed_groups))
        raise GalleryContractError(f"gallery has no eligible cases for groups: {missing}")
    if (audit["shortfall"].astype(int) != 0).any():
        short = audit.loc[audit["shortfall"].astype(int).ne(0), ["route", "category"]]
        raise GalleryContractError(
            f"gallery has fewer than two cases for groups: {short.to_dict(orient='records')}"
        )
    context = {
        "output": output,
        "inputs": inputs,
        "case_payloads": case_payloads,
        "visual_manifest": visual_manifest,
        "routes": routes,
    }
    return eligible, selected, {"audit": audit, **context}


def _load_image(record: Mapping[str, Any], *, name: str) -> np.ndarray:
    path = _external_record(record, name=name)
    if path.suffix.lower() in {".npy", ".npz"}:
        value = np.load(path, allow_pickle=False)
        if isinstance(value, np.lib.npyio.NpzFile):
            keys = tuple(value.files)
            if len(keys) != 1:
                value.close()
                raise GalleryContractError(f"{name} NPZ must contain one array")
            array = np.asarray(value[keys[0]])
            value.close()
            return array
        return np.asarray(value)
    with Image.open(path) as image:
        return np.asarray(image)


def _align_probability_to_native(
    probability: np.ndarray, native_shape: tuple[int, int]
) -> np.ndarray:
    """Apply the frozen soft-probability transform used by G1/C1 features."""

    try:
        return resize_probability_to_native(probability, native_shape)
    except ValueError as error:
        raise GalleryContractError(
            "predicted probability cannot be aligned to the RGB-native frame"
        ) from error


def _raw_gt_rectangles(values: Any) -> list[Any]:
    try:
        parsed = json.loads(values) if isinstance(values, str) else values
    except json.JSONDecodeError as error:
        raise GalleryContractError("GT grasp rectangles are not valid JSON") from error
    if not isinstance(parsed, list):
        raise GalleryContractError("GT grasp rectangles must be a list")
    return parsed


def _gt_rectangles(values: Any) -> list[dict[str, float]]:
    parsed = _raw_gt_rectangles(values)
    result = []
    for corners in parsed:
        rectangle = gt_corners_to_canonical(corners)
        result.append(
            {
                "candidate_id": f"gt-{len(result):04d}",
                "cx_px": rectangle["cx_px"],
                "cy_px": rectangle["cy_px"],
                "theta_deg": rectangle["theta_deg"],
                "width_px": rectangle["jaw_width_px"],
                "height_px": rectangle["rectangle_height_px"],
            }
        )
    return result


def _board_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    pred_rows = payload["pred_candidates"]
    selected_id = str(payload["r7_candidate_id"] or payload["native_candidate_id"] or "")
    selected = next(
        (row for row in pred_rows if str(row["candidate_id"]) == selected_id), None
    )
    return {
        "sample_id": payload["sample_id"],
        "route": payload["route"],
        "category": payload["category"],
        "language_prompt": payload["language_prompt"],
        "candidate_count_pred": int(metrics["pred_candidate_count"]),
        "candidate_count_gt": int(metrics["gt_candidate_count"]),
        "positive_count_pred": int(metrics["pred_positive_candidate_count"]),
        "positive_count_gt": int(metrics["gt_positive_candidate_count"]),
        "first_positive_rank_pred": metrics["pred_first_positive_rank"],
        "first_positive_rank_gt": metrics["gt_first_positive_rank"],
        "native_candidate_id": payload["native_candidate_id"],
        "r7_candidate_id": payload["r7_candidate_id"],
        "gt_candidate_id": payload["gt_candidate_id"],
        "native_q": next(
            (
                row["native_score"]
                for row in pred_rows
                if str(row["candidate_id"]) == str(payload["native_candidate_id"])
            ),
            np.nan,
        ),
        "rerank_score": payload["selector"]["score"],
        "rotated_iou": np.nan if selected is None else selected["best_same_gt_iou"],
        "angle_error_deg": np.nan if selected is None else selected["best_same_gt_angle_error_deg"],
        "pass_fail": "NO_OUTPUT" if selected is None else "PASS" if selected["candidate_success"] else "FAIL",
        "earliest_observable_issue": payload["category"],
        "asset_bundle_sha256": canonical_sha256(payload),
    }


def _prepare_gallery(
    run_dir: str | Path,
    *,
    resume: bool = False,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    resource_gate_record: Mapping[str, Any] | None = None,
) -> Path:
    """Recompute eligibility/selection, render, and stop at manual-QA pending."""

    root = Path(run_dir).expanduser().resolve()
    pipeline = _pipeline(root)
    if pipeline.get("status") not in {
        RunState.P8_STATISTICS_COMPLETE.value,
    }:
        raise PermissionError("core gallery preparation requires P8")
    destination = root / BUILD_RELATIVE_PATH
    if destination.exists():
        if not resume:
            raise FileExistsError("gallery build exists; pass --resume")
        verify_gallery_build(root, expected_sample_count=expected_sample_count)
        return destination
    eligible, selected, context = _canonical_cases(
        root, expected_sample_count=expected_sample_count
    )
    audit = context["audit"]
    selection_dir = root / "12_case_selection"
    gallery_dir = root / "14_galleries"
    eligible_path = atomic_parquet(eligible, selection_dir / "eligible_cases.parquet")
    selected_path = atomic_parquet(selected, selection_dir / "selected_cases.parquet")
    selected_csv_path = atomic_csv(selected, selection_dir / "selected_cases.csv")
    selection_rule = canonical_semantic_contracts()["case_selection"]
    selection_rules_path = atomic_json(
        selection_dir / "selection_rules.json", selection_rule
    )
    audit_payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "selection_rule": selection_rule,
        "eligible": artifact_record(eligible_path),
        "selected": artifact_record(selected_path),
        "groups": audit.to_dict(orient="records"),
    }
    audit_payload["content_sha256"] = canonical_sha256(audit_payload)
    audit_path = atomic_json(selection_dir / "selection_audit.json", audit_payload)
    audit_markdown = [
        "# Case-selection audit",
        "",
        "Status: PASS.",
        "",
        "Cases were selected only after constructing the complete eligible table, "
        "using the protocol-bound mechanism-purity and presentation checks, "
        "cluster medoids where features were available, and the locked SHA-256 tie-break.",
        "",
        "| Route | Category | Eligible | Selected |",
        "|---|---|---:|---:|",
    ]
    for row in audit.to_dict(orient="records"):
        audit_markdown.append(
            f"| {row['route']} | {row['category']} | {int(row['eligible_count'])} | "
            f"{int(row['selected_count'])} |"
        )
    audit_markdown_path = atomic_text(
        selection_dir / "case_selection_audit.md", "\n".join(audit_markdown) + "\n"
    )
    board_rows: list[dict[str, Any]] = []
    for selected_row in selected.to_dict(orient="records"):
        case_id = str(selected_row["case_id"])
        payload = context["case_payloads"][case_id]
        assets_meta = payload["assets"]
        rgb = _load_image(assets_meta["rgb"], name=f"{case_id} RGB")
        probability = None
        if assets_meta["pred_probability"].get("status") != "NOT_AVAILABLE":
            probability = _align_probability_to_native(
                _load_image(
                    assets_meta["pred_probability"],
                    name=f"{case_id} probability",
                ),
                rgb.shape[:2],
            )
        pred_rows = list(payload["pred_candidates"])
        gt_rows = list(payload["gt_candidates"])
        raw_gt = _raw_gt_rectangles(_normal(payload["gt_grasp_rectangles"]))
        gt_rectangles = _gt_rectangles(raw_gt)
        selected_id = str(
            payload["r7_candidate_id"] or payload["native_candidate_id"] or ""
        )
        selected_row = next(
            (
                row
                for row in pred_rows
                if str(row["candidate_id"]) == selected_id
            ),
            None,
        )
        matched_gt: list[dict[str, float]] = []
        if selected_row is not None:
            matched_index = selected_row.get("matched_gt_index")
            if (
                matched_index is None
                or int(matched_index) < 0
                or int(matched_index) >= len(gt_rectangles)
            ):
                raise GalleryContractError(
                    f"{case_id} selected candidate has no valid matched GT"
                )
            matched_gt = [gt_rectangles[int(matched_index)]]
        k = 10 if payload["route"] == "D1" else 5
        board_assets = {
            "rgb": rgb,
            "gt_mask": _load_image(assets_meta["gt_mask"], name=f"{case_id} GT mask"),
            "pred_probability": probability,
            "pred_mask": _load_image(assets_meta["pred_mask"], name=f"{case_id} predicted mask"),
            "depth": _load_image(assets_meta["depth"], name=f"{case_id} depth"),
            "pred_all_candidates": pred_rows,
            "pred_top_candidates": [row for row in pred_rows if int(row["native_rank"]) <= k],
            "gt_all_candidates": gt_rows,
            "gt_top_candidates": [row for row in gt_rows if int(row["native_rank"]) <= k],
            "gt_grasps": gt_rectangles,
            "raw_gt_grasp_rectangles": raw_gt,
            "matched_gt_grasps": matched_gt,
            "asset_bundle_payload": payload,
        }
        board_dir = gallery_dir / "boards"
        qa = render_case_board(
            case=_board_case(payload),
            assets=board_assets,
            output_png=board_dir / f"{case_id}.png",
            output_svg=board_dir / f"{case_id}.svg",
        )
        spec_payload = dict(payload)
        spec_payload["board_qa"] = qa
        spec_payload["content_sha256"] = canonical_sha256(spec_payload)
        spec = atomic_json(gallery_dir / "BOARD_SPECS" / f"{case_id}.json", spec_payload)
        board_rows.append({**qa, "case_id": case_id, "spec": artifact_record(spec)})
    ordered_pngs = [
        Path(str(row["png"]["path"]))
        for row in sorted(board_rows, key=lambda row: str(row["case_id"]))
    ]
    case_pdf_path = root / "13_figures/gtmask_counterfactual_cases.pdf"
    case_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = case_pdf_path.with_name(
        f".{case_pdf_path.name}.{os.getpid()}.tmp"
    )
    pages: list[Image.Image] = []
    try:
        for png in ordered_pngs:
            with Image.open(png) as image:
                pages.append(image.convert("RGB"))
        if not pages:
            raise GalleryContractError("case-board PDF requires at least one board")
        pages[0].save(
            temporary_pdf,
            "PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=144.0,
        )
        with temporary_pdf.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_pdf, case_pdf_path)
    finally:
        for page in pages:
            page.close()
        if temporary_pdf.exists():
            temporary_pdf.unlink()
    qa_template = pd.DataFrame(
        [
            {
                "case_id": row["case_id"],
                "sample_id": row["sample_id"],
                "route": row["route"],
                "category": row["category"],
                "status": "PENDING",
                "reviewer": "",
                "reviewed_at_utc": "",
                "notes": "",
                "signature_sha256": "",
            }
            for row in board_rows
        ]
    )
    qa_template_path = atomic_csv(
        qa_template, gallery_dir / "MANUAL_QA_TEMPLATE.csv"
    )
    build: dict[str, Any] = {
        "schema_version": 1,
        "status": "PENDING_MANUAL_QA",
        "postprocess_manifest": artifact_record(
            root / "08_metrics/POSTPROCESS_MANIFEST.json"
        ),
        "protocol_lock": artifact_record(root / LOCK_RELATIVE_PATH),
        "visual_asset_registry": artifact_record(
            root / "04_predicted_replay/VISUAL_ASSET_REGISTRY.json"
        ),
        "eligible": artifact_record(eligible_path),
        "selected": artifact_record(selected_path),
        "selected_csv": artifact_record(selected_csv_path),
        "selection_rules": artifact_record(selection_rules_path),
        "selection_audit": artifact_record(audit_path),
        "case_selection_audit": artifact_record(audit_markdown_path),
        "core_cases_figure": artifact_record(case_pdf_path),
        "manual_qa_template": artifact_record(qa_template_path),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "routes": list(context["routes"]),
        "categories": list(CASE_CATEGORIES),
        "quota_per_route_category": QUOTA_PER_ROUTE_CATEGORY,
        "resource_gate": (
            dict(resource_gate_record)
            if resource_gate_record is not None
            else {"kind": "test_only_synthetic_no_heavy_gate"}
        ),
        "required_panels": list(BOARD_PANELS),
        "boards": board_rows,
    }
    build["content_sha256"] = canonical_sha256(build)
    return atomic_json(destination, build)


def prepare_gallery(
    run_dir: str | Path,
    *,
    resume: bool = False,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    resource_gate: str | Path | None = None,
    collect_resource_gate: bool = False,
    rank1_run_dir: str | Path | None = None,
    test_only_allow_synthetic_without_gate: bool = False,
) -> Path:
    """Run P9 under the repository-global resource lease and a fresh gate."""

    root = Path(run_dir).expanduser().resolve()
    if test_only_allow_synthetic_without_gate:
        if int(expected_sample_count) == EXPECTED_SAMPLE_COUNT:
            raise PermissionError("production-sized gallery cannot bypass its resource gate")
        return _prepare_gallery(
            root,
            resume=resume,
            expected_sample_count=expected_sample_count,
        )
    if bool(resource_gate) == bool(collect_resource_gate):
        raise ValueError("choose exactly one resource gate source")
    if rank1_run_dir is None:
        raise ValueError("rank1_run_dir is required for the live resource interlock")
    rank1 = Path(rank1_run_dir).expanduser().resolve()
    repository = root.parent.parent
    with exclusive_d1_flock(root, purpose="GT-mask P9 gallery build"):
        if collect_resource_gate:
            gate_value = collect_fresh_three_by_five_gate(
                repo_root=repository, rank1_run_dir=rank1
            )
            gate_path = root / "00_audit/resource_gates" / (
                f"p9_gallery_{gate_value['content_sha256'][:20]}.json"
            )
            atomic_json(gate_path, gate_value)
        else:
            gate_path = Path(str(resource_gate)).expanduser().resolve()
            gate_value = _object(gate_path, name="P9 resource gate")
        validate_fresh_gate(gate_value)
        validate_live_resources(
            repo_root=repository,
            rank1_run_dir=rank1,
            prefix="gtmask_p9_gallery_launch",
        )
        validate_fresh_gate(gate_value)
        return _prepare_gallery(
            root,
            resume=resume,
            expected_sample_count=expected_sample_count,
            resource_gate_record=artifact_record(gate_path),
        )


def _compare_frame(saved: pd.DataFrame, expected: pd.DataFrame, *, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            saved.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
        )
    except AssertionError as error:
        raise GalleryContractError(f"saved {name} differs from canonical recompute") from error


def verify_gallery_build(
    run_dir: str | Path, *, expected_sample_count: int = EXPECTED_SAMPLE_COUNT
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    path = root / BUILD_RELATIVE_PATH
    value = _object(path, name="gallery build manifest")
    _self_hash(value, name="gallery build manifest")
    if value.get("status") != "PENDING_MANUAL_QA":
        raise GalleryContractError("gallery build did not stop at manual QA")
    gate = value.get("resource_gate")
    if isinstance(gate, Mapping) and "path" in gate:
        gate_path = _record(gate, root=root, name="P9 resource gate")
        gate_value = _object(gate_path, name="P9 resource gate")
        unsigned_gate = dict(gate_value)
        observed_gate_hash = unsigned_gate.pop("content_sha256", None)
        if (
            observed_gate_hash != canonical_sha256(unsigned_gate)
            or gate_value.get("status") != "PASS"
            or gate_value.get("gate_type")
            != "gtmask_d1_three_continuous_five_minute_windows_v1"
        ):
            raise GalleryContractError("P9 resource gate evidence differs")
    elif not (
        int(expected_sample_count) != EXPECTED_SAMPLE_COUNT
        and gate == {"kind": "test_only_synthetic_no_heavy_gate"}
    ):
        raise GalleryContractError("gallery build lacks a production resource gate")
    eligible, selected, context = _canonical_cases(
        root, expected_sample_count=expected_sample_count
    )
    _compare_frame(
        pd.read_parquet(_record(value["eligible"], root=root, name="eligible cases")),
        eligible,
        name="eligible cases",
    )
    _compare_frame(
        pd.read_parquet(_record(value["selected"], root=root, name="selected cases")),
        selected,
        name="selected cases",
    )
    _compare_frame(
        pd.read_csv(_record(value["selected_csv"], root=root, name="selected cases CSV")),
        selected,
        name="selected cases CSV",
    )
    rules_path = _record(value["selection_rules"], root=root, name="selection rules")
    rules = _object(rules_path, name="selection rules")
    if rules != canonical_semantic_contracts()["case_selection"]:
        raise GalleryContractError("selection rules differ")
    audit_path = _record(value["selection_audit"], root=root, name="selection audit")
    audit = _object(audit_path, name="selection audit")
    _self_hash(audit, name="selection audit")
    if audit.get("groups") != context["audit"].to_dict(orient="records"):
        raise GalleryContractError("selection audit groups differ from recompute")
    _record(value["case_selection_audit"], root=root, name="case-selection audit")
    _record(value["core_cases_figure"], root=root, name="core cases figure")
    selected_ids = set(selected["case_id"].astype(str))
    boards = value.get("boards")
    if not isinstance(boards, Sequence) or isinstance(boards, (str, bytes)):
        raise GalleryContractError("gallery board inventory is absent")
    if {str(row.get("case_id")) for row in boards} != selected_ids:
        raise GalleryContractError("gallery boards do not exactly cover selection")
    for row in boards:
        if row.get("status") != "AUTO_QA_PASS":
            raise GalleryContractError("gallery board auto-QA did not pass")
        for name in ("png", "svg", "spec"):
            _record(row[name], root=root, name=f"board {row['case_id']}.{name}")
        spec = _object(
            Path(str(row["spec"]["path"])), name=f"board {row['case_id']} spec"
        )
        _self_hash(spec, name=f"board {row['case_id']} spec")
        expected_payload = context["case_payloads"][str(row["case_id"])]
        for key, child in expected_payload.items():
            if spec.get(key) != child:
                raise GalleryContractError(
                    f"board {row['case_id']} spec differs: {key}"
                )
    _record(value["manual_qa_template"], root=root, name="manual QA template")
    return value


def manual_qa_signature(build_sha256: str, row: Mapping[str, Any]) -> str:
    fields = {
        "build_sha256": str(build_sha256),
        "case_id": str(row.get("case_id", "")),
        "sample_id": str(row.get("sample_id", "")),
        "route": str(row.get("route", "")),
        "category": str(row.get("category", "")),
        "status": str(row.get("status", "")),
        "reviewer": str(row.get("reviewer", "")),
        "reviewed_at_utc": str(row.get("reviewed_at_utc", "")),
        "notes": str(row.get("notes", "")),
    }
    return canonical_sha256(fields)


def accept_gallery_manual_qa(
    run_dir: str | Path,
    *,
    manual_qa_csv: str | Path,
    resume: bool = False,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
) -> Path:
    """Accept an immutable build only after an exact signed human review."""

    root = Path(run_dir).expanduser().resolve()
    build = verify_gallery_build(root, expected_sample_count=expected_sample_count)
    qa_path = Path(manual_qa_csv).expanduser().resolve()
    if qa_path.is_symlink() or not qa_path.is_file():
        raise GalleryContractError("manual QA CSV is absent or unsafe")
    qa = pd.read_csv(qa_path, keep_default_na=False)
    required = {
        "case_id",
        "sample_id",
        "route",
        "category",
        "status",
        "reviewer",
        "reviewed_at_utc",
        "notes",
        "signature_sha256",
    }
    if set(qa.columns) != required or qa.empty or qa["case_id"].astype(str).duplicated().any():
        raise GalleryContractError("manual QA schema or identity differs")
    selected = pd.read_parquet(
        _record(build["selected"], root=root, name="selected cases")
    )
    expected = selected[["case_id", "sample_id", "route", "category"]].astype(str)
    compared = expected.merge(
        qa,
        on=["case_id", "sample_id", "route", "category"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not compared["_merge"].eq("both").all():
        raise GalleryContractError("manual QA does not exactly cover selected cases")
    for row in qa.to_dict(orient="records"):
        if (
            str(row["status"]) != "PASS"
            or not str(row["reviewer"]).strip()
            or not str(row["notes"]).strip()
        ):
            raise GalleryContractError("manual QA requires explicit reviewer/PASS/notes")
        try:
            reviewed = datetime.fromisoformat(str(row["reviewed_at_utc"]).replace("Z", "+00:00"))
        except ValueError as error:
            raise GalleryContractError("manual QA timestamp is invalid") from error
        if reviewed.tzinfo is None or reviewed > datetime.now(timezone.utc):
            raise GalleryContractError("manual QA timestamp is unzoned or in the future")
        expected_signature = manual_qa_signature(str(build["content_sha256"]), row)
        if str(row["signature_sha256"]) != expected_signature:
            raise GalleryContractError("manual QA row signature differs")
    accepted_csv = atomic_csv(qa, root / "14_galleries/MANUAL_QA.csv")
    acceptance: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "gallery_build": artifact_record(root / BUILD_RELATIVE_PATH),
        "manual_qa": artifact_record(accepted_csv),
        "selected_count": len(selected),
        "exact_selected_coverage": True,
    }
    acceptance["content_sha256"] = canonical_sha256(acceptance)
    acceptance_path = root / MANUAL_ACCEPTANCE_RELATIVE_PATH
    if acceptance_path.exists():
        observed = _object(acceptance_path, name="manual QA acceptance")
        if not resume or observed != acceptance:
            raise GalleryContractError("existing manual QA acceptance differs")
    else:
        atomic_json(acceptance_path, acceptance)
    final: dict[str, Any] = {
        "schema_version": 2,
        "status": "COMPLETE",
        "manual_qa_status": "PASS",
        "manual_qa_coverage_pass": True,
        "selection_rule": "mechanism-purity + presentation eligibility + cluster medoid + SHA256 tie",
        "postprocess_manifest": artifact_record(root / "08_metrics/POSTPROCESS_MANIFEST.json"),
        "gallery_build": artifact_record(root / BUILD_RELATIVE_PATH),
        "manual_qa_acceptance": artifact_record(acceptance_path),
        "eligible": build["eligible"],
        "selected": build["selected"],
        "selected_csv": build["selected_csv"],
        "selection_rules": build["selection_rules"],
        "selection_audit": build["selection_audit"],
        "case_selection_audit": build["case_selection_audit"],
        "core_cases_figure": build["core_cases_figure"],
        "eligible_count": int(build["eligible_count"]),
        "selected_count": int(build["selected_count"]),
        "boards": build["boards"],
        "required_panels": list(BOARD_PANELS),
    }
    final["content_sha256"] = canonical_sha256(final)
    final_path = root / FINAL_RELATIVE_PATH
    if final_path.exists():
        observed = _object(final_path, name="gallery manifest")
        if not resume or observed != final:
            raise GalleryContractError("existing gallery manifest differs")
        return final_path
    return atomic_json(final_path, final)


def verify_complete_gallery(
    run_dir: str | Path, *, expected_sample_count: int = EXPECTED_SAMPLE_COUNT
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    build = verify_gallery_build(root, expected_sample_count=expected_sample_count)
    final = _object(root / FINAL_RELATIVE_PATH, name="gallery manifest")
    _self_hash(final, name="gallery manifest")
    acceptance_path = _record(
        final["manual_qa_acceptance"], root=root, name="manual QA acceptance"
    )
    acceptance = _object(acceptance_path, name="manual QA acceptance")
    _self_hash(acceptance, name="manual QA acceptance")
    if (
        final.get("status") != "COMPLETE"
        or final.get("manual_qa_status") != "PASS"
        or final.get("gallery_build") != artifact_record(root / BUILD_RELATIVE_PATH)
        or acceptance.get("status") != "PASS"
        or acceptance.get("gallery_build") != artifact_record(root / BUILD_RELATIVE_PATH)
        or int(final.get("selected_count", -1)) != int(build["selected_count"])
        or final.get("eligible") != build["eligible"]
        or final.get("selected") != build["selected"]
        or final.get("boards") != build["boards"]
        or final.get("core_cases_figure") != build["core_cases_figure"]
    ):
        raise GalleryContractError("completed gallery contract differs")
    qa_path = _record(acceptance["manual_qa"], root=root, name="manual QA CSV")
    qa = pd.read_csv(qa_path, keep_default_na=False)
    selected = pd.read_parquet(
        _record(build["selected"], root=root, name="selected cases")
    )
    expected_keys = set(selected["case_id"].astype(str))
    if (
        qa.empty
        or qa["case_id"].astype(str).duplicated().any()
        or set(qa["case_id"].astype(str)) != expected_keys
        or not qa["status"].astype(str).eq("PASS").all()
        or qa["reviewer"].astype(str).str.strip().eq("").any()
        or qa["notes"].astype(str).str.strip().eq("").any()
    ):
        raise GalleryContractError("completed gallery manual coverage differs")
    for row in qa.to_dict(orient="records"):
        if manual_qa_signature(str(build["content_sha256"]), row) != str(
            row["signature_sha256"]
        ):
            raise GalleryContractError("completed gallery manual signature differs")
    return final


__all__ = [
    "BUILD_RELATIVE_PATH",
    "EXPECTED_SAMPLE_COUNT",
    "FINAL_RELATIVE_PATH",
    "GalleryContractError",
    "MANUAL_ACCEPTANCE_RELATIVE_PATH",
    "QUOTA_PER_ROUTE_CATEGORY",
    "accept_gallery_manual_qa",
    "manual_qa_signature",
    "prepare_gallery",
    "verify_complete_gallery",
    "verify_gallery_build",
]
