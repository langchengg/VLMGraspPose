"""Deterministic Train/Validation inputs for the P12 four-route extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS
from unified_reranking.hashing import atomic_json, canonical_sha256

from .execution import artifact_record, load_content_manifest
from .four_route import build_top20_union
from .gate_validation import validate_gate_selection_semantics
from .io import atomic_parquet


THREE_ROUTES = ("CROG", "G1", "C1")
ROUTER_SPLITS = ("train", "validation")


def _validation_reference_columns(
    *, three_route_run: Path, sample_ids: pd.Series
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Replay frozen three-route correctness/oracle development evidence."""

    router_path = (
        three_route_run
        / "08_lock/route_router/route_router_validation_decisions.parquet"
    )
    labels_path = (
        three_route_run
        / "03_features/tracks/T5_cross_route/union_validation/candidate_labels.parquet"
    )
    expected = sample_ids.astype(str)
    if expected.empty or expected.duplicated().any():
        raise RuntimeError("P12 Validation reference denominator differs")
    router = pd.read_parquet(router_path, columns=["sample_id", "selected_correct"])
    router["sample_id"] = router["sample_id"].astype(str)
    if (
        router["sample_id"].duplicated().any()
        or set(router["sample_id"]) != set(expected)
    ):
        raise RuntimeError("P12 three-route router Validation coverage differs")
    labels = pd.read_parquet(labels_path, columns=["sample_id", "candidate_success"])
    labels["sample_id"] = labels["sample_id"].astype(str)
    if set(labels["sample_id"]).difference(set(expected)):
        raise RuntimeError("P12 Top15 oracle labels contain a foreign sample")
    oracle = labels.groupby("sample_id", sort=False)["candidate_success"].any()
    references = pd.DataFrame({"sample_id": expected})
    references = references.merge(router, on="sample_id", validate="one_to_one")
    references = references.rename(
        columns={"selected_correct": "three_route_router_correct"}
    )
    references["existing_top15_oracle"] = (
        references["sample_id"].map(oracle).fillna(False).astype(bool)
    )
    return references, {
        "three_route_validation_router_decisions": router_path,
        "three_route_validation_top15_labels": labels_path,
    }


def _same_ids(left: pd.DataFrame, right: pd.DataFrame, *, name: str) -> None:
    left_ids = left["sample_id"].astype(str).tolist()
    right_ids = right["sample_id"].astype(str).tolist()
    if left_ids != right_ids or len(left_ids) != len(set(left_ids)):
        raise RuntimeError(f"{name} sample order/coverage differs")


def _d1_gate_decisions(
    inputs: pd.DataFrame, *, gate_manifest: Path
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    from tools.d1_reranking.apply_locked_test_gate import _apply

    gate = load_content_manifest(
        gate_manifest, name="D1 Validation gate", statuses=("COMPLETE",)
    )
    validated = validate_gate_selection_semantics(gate)
    feature_columns = tuple(
        map(str, gate.get("configuration", {}).get("feature_columns", ()))
    )
    if feature_columns != tuple(SAFE_GATE_FEATURE_COLUMNS):
        raise RuntimeError("D1 gate feature schema differs from the safe contract")
    inference_inputs = inputs.copy()
    if "candidate_count" not in inference_inputs.columns:
        native = inference_inputs["native_candidate_id"].fillna("").astype(str)
        challenger = inference_inputs["challenger_candidate_id"].fillna("").astype(str)
        inference_inputs["candidate_count"] = native.ne("") | challenger.ne("")
        inference_inputs["candidate_count"] = inference_inputs[
            "candidate_count"
        ].astype(int)
    # Development gate inputs predate the Test-only candidate identity fields.
    # The router consumes selected ID/geometry only; keep the unused identity
    # outputs explicitly empty rather than inventing an identity hash.
    for column in ("native_identity_sha256", "challenger_identity_sha256"):
        if column not in inference_inputs.columns:
            inference_inputs[column] = ""
    decisions, _point = _apply(gate, validated.model, inference_inputs)
    return decisions, feature_columns


def _router_frame(
    *, three: pd.DataFrame, d1: pd.DataFrame, gate_manifest: Path, split: str
) -> tuple[pd.DataFrame, list[str]]:
    expected_source = "train_oof" if split == "train" else "validation"
    if set(three["prediction_source"].astype(str)) != {expected_source} or set(
        d1["prediction_source"].astype(str)
    ) != {expected_source}:
        raise RuntimeError(f"P12 {split} router prediction provenance differs")
    _same_ids(three, d1, name=f"P12 {split} router")
    decisions, safe_columns = _d1_gate_decisions(d1, gate_manifest=gate_manifest)
    _same_ids(d1, decisions, name=f"P12 {split} D1 gate")
    selected_correct = np.where(
        decisions["switch"].to_numpy(bool),
        d1["challenger_correct"].to_numpy(bool),
        d1["native_correct"].to_numpy(bool),
    )
    result = three.copy()
    for route in THREE_ROUTES:
        lower = route.lower()
        canonical_geometry = f"{lower}_candidate_geometry_sha256"
        selected_geometry = f"{lower}_selected_candidate_geometry_sha256"
        if canonical_geometry not in result.columns:
            if selected_geometry not in result.columns:
                raise RuntimeError(f"P12 {split} {route} selected geometry is absent")
            result[canonical_geometry] = (
                result[selected_geometry].fillna("").astype(str)
            )
    result["d1_correct"] = selected_correct
    for column in safe_columns:
        result[f"d1_{column}"] = pd.to_numeric(d1[column], errors="raise")
    result["d1_candidate_id"] = (
        decisions["selected_candidate_id"].fillna("").astype(str)
    )
    result["d1_candidate_geometry_sha256"] = (
        decisions["selected_geometry_sha256"].fillna("").astype(str)
    )
    result["d1_margin"] = pd.to_numeric(d1["score_margin"], errors="raise")
    result["d1_reliability"] = pd.to_numeric(
        d1["challenger_reliability"], errors="raise"
    )
    result["d1_stability"] = pd.to_numeric(d1["perturbation_stability"], errors="raise")
    result["d1_candidate_exists"] = decisions["candidate_count"].astype(int).gt(0)
    # The transition router requires an explicit route-level availability
    # signal.  ``challenger_exists_numeric`` is not equivalent: a route can
    # have exactly one native candidate (so the route exists) while having no
    # challenger.  Keep both signals and fit the router with the canonical
    # route-level candidate-existence flag.
    d1_columns = [
        *[f"d1_{column}" for column in safe_columns],
        "d1_candidate_exists",
    ]
    if not np.isfinite(result[d1_columns].to_numpy(float)).all():
        raise RuntimeError(f"P12 {split} D1 router features are non-finite")
    return result, d1_columns


def _enriched_route_top5(
    *,
    candidates_path: Path,
    features_path: Path,
    labels_path: Path,
    feature_columns: tuple[str, ...],
    route: str,
) -> pd.DataFrame:
    candidates = pd.read_parquet(candidates_path)
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    keys = ["sample_id", "candidate_id", "native_rank"]
    required_candidates = (*keys, "candidate_geometry_sha256")
    missing = sorted(set(required_candidates).difference(candidates.columns))
    if missing:
        raise RuntimeError(f"P12 {route} candidates miss columns: {missing}")
    if candidates.duplicated(keys[:2]).any():
        raise RuntimeError(f"P12 {route} candidates contain duplicate keys")
    feature_required = {*keys, *feature_columns}
    label_required = {*keys, "candidate_success", "jacquard_margin"}
    if feature_required.difference(features.columns) or label_required.difference(
        labels.columns
    ):
        raise RuntimeError(f"P12 {route} feature/label schema differs")
    if "candidate_identity_sha256" not in candidates.columns:
        candidates = candidates.copy()
        candidates["candidate_identity_sha256"] = [
            canonical_sha256(
                {
                    "identity_contract": "p12_extension_route_raw_id_geometry_v1",
                    "route": route,
                    "sample_id": str(sample_id),
                    "candidate_id": str(candidate_id),
                    "candidate_geometry_sha256": str(geometry),
                }
            )
            for sample_id, candidate_id, geometry in candidates[
                ["sample_id", "candidate_id", "candidate_geometry_sha256"]
            ].itertuples(index=False, name=None)
        ]
    base_columns = [*required_candidates, "candidate_identity_sha256"]
    for column in (
        "route",
        "native_score",
        "cx_px",
        "cy_px",
        "theta_deg",
        "width_px",
        "height_px",
    ):
        if column in candidates.columns and column not in base_columns:
            base_columns.append(column)
    # Candidate identity/geometry is authoritative in the canonical pool.
    # Several T2 model columns intentionally repeat geometry.  Merging those
    # copies would create ``*_x``/``*_y`` columns and silently remove the
    # canonical names required by the Top20 contract.
    frozen_payload = set(base_columns).difference(keys)
    feature_payload = [
        *keys,
        *[
            column
            for column in feature_columns
            if column not in keys and column not in frozen_payload
        ],
    ]
    result = candidates[base_columns].merge(
        features[feature_payload], on=keys, validate="one_to_one"
    )
    result = result.merge(
        labels[[*keys, "candidate_success", "jacquard_margin"]],
        on=keys,
        validate="one_to_one",
    )
    if len(result) != len(candidates):
        raise RuntimeError(f"P12 {route} enriched Top5 loses membership")
    result["route"] = route
    return result


def _three_feature_columns(three_manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    configuration = three_manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RuntimeError("three-route router configuration is absent")
    columns = configuration.get("feature_columns")
    if not isinstance(columns, Mapping) or set(columns) != {"G1", "C1"}:
        raise RuntimeError("three-route router feature columns differ")
    return {route: list(map(str, columns[route])) for route in ("G1", "C1")}


def assemble_four_route_development_inputs(
    *, d1_run_dir: str | Path, three_route_run: str | Path, resume: bool
) -> dict[str, Any]:
    """Build the exact P12 producer spec without opening any Test input."""

    root = Path(d1_run_dir).expanduser().resolve()
    three_root = Path(three_route_run).expanduser().resolve()
    destination = root / "13_four_route_extension/inputs"
    manifest_path = destination / "manifest.json"
    router_manifest_path = (
        three_root / "07_validation/route_router_inputs/manifest.json"
    )
    three_manifest = json.loads(router_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(three_manifest, dict)
        or three_manifest.get("status") != "COMPLETE"
    ):
        raise RuntimeError("three-route router inputs are not COMPLETE")
    three_columns = _three_feature_columns(three_manifest)
    gate_manifest = root / "07_validation/gate/d1/gate_selection.json"
    d1_gate_input_manifest = root / "07_validation/gate_inputs/d1/manifest.json"
    load_content_manifest(
        d1_gate_input_manifest, name="D1 gate inputs", statuses=("COMPLETE",)
    )
    router_paths: dict[str, Path] = {}
    d1_router_columns: list[str] | None = None
    validation_reference_paths: dict[str, Path] = {}
    for split in ROUTER_SPLITS:
        three_path = (
            three_root
            / f"07_validation/route_router_inputs/{'train_oof' if split == 'train' else 'validation'}.parquet"
        )
        d1_path = (
            root
            / f"07_validation/gate_inputs/d1/{'train_oof' if split == 'train' else 'validation'}.parquet"
        )
        frame, columns = _router_frame(
            three=pd.read_parquet(three_path),
            d1=pd.read_parquet(d1_path),
            gate_manifest=gate_manifest,
            split=split,
        )
        if split == "validation":
            references, validation_reference_paths = _validation_reference_columns(
                three_route_run=three_root, sample_ids=frame["sample_id"]
            )
            frame = frame.merge(references, on="sample_id", validate="one_to_one")
        if d1_router_columns is not None and columns != d1_router_columns:
            raise RuntimeError("P12 D1 router feature columns drift across splits")
        d1_router_columns = columns
        output = (
            destination
            / "router"
            / ("train_oof.parquet" if split == "train" else "validation.parquet")
        )
        router_paths[split] = atomic_parquet(frame, output)

    t2_manifest = load_content_manifest(
        root / "03_features/train/top5/T2_matched_common/manifest.json",
        name="D1 Train Top5 T2",
        statuses=("COMPLETE",),
    )
    feature_columns = tuple(map(str, t2_manifest.get("model_feature_columns", ())))
    if not feature_columns or "base_logit" not in feature_columns:
        raise RuntimeError("P12 union feature schema lacks the frozen D1 T2 columns")
    union_columns = [
        *feature_columns,
        "union_route_crog",
        "union_route_g1",
        "union_route_c1",
        "union_route_d1",
    ]
    union_paths: dict[str, Path] = {}
    for split in ROUTER_SPLITS:
        denominator_path = root / f"01_manifests/d1_paired_{split}.parquet"
        denominator = pd.read_parquet(denominator_path, columns=["sample_id"])
        route_frames: dict[str, pd.DataFrame] = {}
        for route in THREE_ROUTES:
            lower = route.lower()
            route_frames[route] = _enriched_route_top5(
                candidates_path=three_root
                / f"02_candidates/{lower}_{split}_top5.parquet",
                features_path=three_root
                / f"03_features/tracks/T2_matched_common/{lower}_{split}/candidate_features.parquet",
                labels_path=three_root
                / f"03_features/candidate_labels_{lower}_{split}_top5.parquet",
                feature_columns=feature_columns,
                route=route,
            )
        route_frames["D1"] = _enriched_route_top5(
            candidates_path=root / f"02_candidates/{split}/d1_top5_candidates.parquet",
            features_path=root
            / f"03_features/{split}/top5/T2_matched_common/candidate_features.parquet",
            labels_path=root
            / f"03_features/{split}/top5/labels/candidate_labels.parquet",
            feature_columns=feature_columns,
            route="D1",
        )
        union, _audit = build_top20_union(route_frames, denominator, split=split)
        union_paths[split] = atomic_parquet(
            union, destination / "union" / f"{split}_top20.parquet"
        )

    sources = {
        "three_route_router_manifest": artifact_record(router_manifest_path),
        "d1_gate_input_manifest": artifact_record(d1_gate_input_manifest),
        "d1_gate_selection": artifact_record(gate_manifest),
        "d1_t2_manifest": artifact_record(
            root / "03_features/train/top5/T2_matched_common/manifest.json"
        ),
        **{
            name: artifact_record(path)
            for name, path in sorted(validation_reference_paths.items())
        },
        "tool": artifact_record(Path(__file__)),
    }
    artifacts = {
        "router_train_oof": artifact_record(router_paths["train"]),
        "router_validation": artifact_record(router_paths["validation"]),
        "union_train_top20": artifact_record(union_paths["train"]),
        "union_validation_top20": artifact_record(union_paths["validation"]),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name="P12 development inputs", statuses=("COMPLETE",)
        )
        if not resume or existing != result:
            raise RuntimeError("P12 development inputs exist and differ")
    else:
        atomic_json(manifest_path, result)

    spec: dict[str, Any] = {
        "schema_version": 1,
        "status": "VALIDATION_PRODUCER_DECLARED",
        "router": {
            "train_oof": str(router_paths["train"]),
            "validation": str(router_paths["validation"]),
            "feature_columns": {
                "G1": three_columns["G1"],
                "C1": three_columns["C1"],
                "D1": d1_router_columns,
            },
        },
        "union": {
            "train_top20": str(union_paths["train"]),
            "validation_top20": str(union_paths["validation"]),
            "folds": str(root / "04_splits/fold_assignments.parquet"),
            "train_denominator": str(root / "01_manifests/d1_paired_train.parquet"),
            "validation_denominator": str(
                root / "01_manifests/d1_paired_validation.parquet"
            ),
            "feature_columns": union_columns,
        },
        "t4": {
            split: {
                "d1_top5": str(
                    root / f"02_candidates/{split}/d1_top5_candidates.parquet"
                ),
                "t3_manifest": str(
                    root / f"03_features/{split}/top5/T3_route_rich/manifest.json"
                ),
                "peer_top5": {
                    route: str(
                        three_root
                        / f"02_candidates/{route.lower()}_{split}_top5.parquet"
                    )
                    for route in THREE_ROUTES
                },
            }
            for split in ROUTER_SPLITS
        },
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
    }
    spec["content_sha256"] = canonical_sha256(spec)
    spec_path = root / "configs/p12_validation_producer_spec.json"
    if spec_path.exists():
        existing_spec = load_content_manifest(
            spec_path,
            name="P12 Validation producer spec",
            statuses=("VALIDATION_PRODUCER_DECLARED",),
        )
        if not resume or existing_spec != spec:
            raise RuntimeError("P12 Validation producer spec exists and differs")
    else:
        atomic_json(spec_path, spec)
    return {"manifest": result, "spec": spec, "spec_path": str(spec_path)}


__all__ = ["assemble_four_route_development_inputs"]
