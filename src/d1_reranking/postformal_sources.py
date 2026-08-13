"""Deterministic label-free P13 covariate and runtime evidence sources."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from unified_reranking.artifacts import (
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.test_access_guard import append_access_log

from .contracts import assert_label_free_parquet_schema
from .execution import load_content_manifest
from .provenance import load_source_closure
from .ranker_contributions import validate_ranker_contributions


SOURCE_MANIFEST_RELATIVE_PATH = "configs/d1_postformal_sources.json"
OUTPUT_ROOT = "08_lock/postformal_sources"
COVARIATES_RELATIVE_PATH = f"{OUTPUT_ROOT}/sample_covariates.parquet"
CONTRIBUTIONS_RELATIVE_PATH = f"{OUTPUT_ROOT}/ranker_contributions.json"
RUNTIME_COMPONENTS = (
    "dexnet_candidate_generation",
    "feature_extraction",
    "ranker_inference",
    "gate_inference",
    "total_d1_pipeline",
)
FIXED_INPUTS = {
    "paired": "01_manifests/d1_paired_manifest.parquet",
    "candidates": "02_candidates/test_manifest.json",
    "raw_features": "03_features/test/top5/matched_common_raw/manifest.json",
    "t2_features": "03_features/test/top5/T2_matched_common/manifest.json",
    "t3_features": "03_features/test/top5/T3_route_rich/manifest.json",
    "feature_execution": "configs/d1_feature_execution.json",
    "ranker": "08_lock/label_free_test_rankers/d1/manifest.json",
    "gate": "08_lock/label_free_test_gates/d1/manifest.json",
}
COVARIATE_COLUMNS = (
    "sample_id",
    "target_size",
    "relation_query",
    "clutter",
    "depth_missing",
    "predicted_mask_confidence",
    "native_mask_support",
    "selected_mask_support",
)
FORBIDDEN_COVARIATE_TOKENS = (
    "correct",
    "success",
    "oracle",
    "label",
    "ground_truth",
    "gt_",
    "mask_iou",
    "mask_quality",
    "jacquard",
    "angle_error",
    "matched_gt",
)
RELATION_PATTERN = re.compile(
    r"\b(left|right|behind|front|near|next|between|above|below|under|over)\b",
    re.IGNORECASE,
)


def _record(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"D1 postformal source is not a regular file: {source}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return path


def _manifest(
    root: Path, name: str, *, statuses: tuple[str, ...] = ("COMPLETE",)
) -> dict[str, Any]:
    value = load_content_manifest(
        root / FIXED_INPUTS[name],
        name=f"D1 postformal source {name}",
        statuses=statuses,
    )
    if value.get("candidate_test_labels_read") is not False:
        raise PermissionError(f"D1 postformal source {name} is not label-free")
    verify_artifact_records_recursive(
        value,
        name=f"D1 postformal source {name}",
        require_at_least_one=True,
    )
    return value


def _artifact(
    manifest: Mapping[str, Any], names: tuple[str, ...], *, name: str
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"{name} artifact inventory is absent")
    for key in names:
        if isinstance(artifacts.get(key), Mapping):
            return verified_artifact_path(artifacts[key], name=f"{name}/{key}")
    raise RuntimeError(f"{name} misses fixed artifact keys {names}")


def _predicted_mask_fraction(path: Path) -> tuple[float, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"D1 predicted mask asset is absent: {path}")
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2 or not array.size:
        raise RuntimeError(f"D1 predicted mask asset shape differs: {path}")
    return float(np.count_nonzero(array) / array.size), _record(path)


def _finite_nonnegative(value: object) -> float | None:
    if (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    ):
        return float(value)
    return None


def _runtime_payload(
    *,
    component: str,
    sources: Mapping[str, Any],
    latency: float | None,
    peak_memory: float | None,
    measurement_semantics: str,
    parameter_count: int | None = None,
    model_bytes: int | None = None,
    unavailable_reason: str = "",
) -> dict[str, Any]:
    complete = (
        latency is not None and peak_memory is not None and not unavailable_reason
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE" if complete else "NOT_AVAILABLE",
        "component": component,
        "candidate_test_labels_read": False,
        "sources": dict(sources),
        "telemetry": {
            "latency_ms_per_sample": latency,
            "peak_memory_mb": peak_memory,
            "parameter_count": parameter_count,
            "model_bytes": model_bytes,
            "measurement_semantics": measurement_semantics,
        },
    }
    if not complete:
        value["unavailable_reason"] = (
            unavailable_reason or "required persisted telemetry is absent"
        )
    value["content_sha256"] = canonical_sha256(value)
    return value


def _contributions_declaration(root: Path, ranker: Mapping[str, Any]) -> dict[str, Any]:
    selected_method = str(ranker.get("selected_method", ""))
    if not selected_method:
        raise RuntimeError("D1 selected Test ranker method is absent")
    if selected_method.upper() in {"R5", "LAMBDAMART"}:
        manifest_path, manifest = validate_ranker_contributions(root)
        ranker_record = _record(root / FIXED_INPUTS["ranker"])
        contribution_sources = manifest.get("sources")
        if not isinstance(contribution_sources, Mapping):
            raise RuntimeError("D1 R5 contribution sources are absent")
        observed_ranker = contribution_sources.get("ranker_manifest")
        if not isinstance(observed_ranker, Mapping) or any(
            observed_ranker.get(key) != ranker_record.get(key)
            for key in ("path", "sha256", "bytes")
        ):
            raise RuntimeError("D1 R5 contributions do not bind the selected ranker")
        path = verified_artifact_path(
            manifest.get("artifacts", {}).get("candidate_contributions", {}),
            name="D1 R5 candidate contributions",
        )
        columns = assert_label_free_parquet_schema(
            path, name="D1 R5 candidate contributions"
        )
        required = {"sample_id", "candidate_id", "feature_name", "contribution"}
        if required.difference(columns):
            raise RuntimeError("D1 R5 candidate contribution schema differs")
        status = "COMPLETE"
        artifact: Mapping[str, Any] | None = _record(path)
        contribution_manifest: Mapping[str, Any] | None = _record(manifest_path)
    else:
        status = "NOT_APPLICABLE"
        artifact = None
        contribution_manifest = None
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "selected_method": selected_method,
        "reason": (
            "locked native LightGBM pred_contrib values"
            if status == "COMPLETE"
            else f"selected ranker {selected_method} is not LightGBM R5"
        ),
        "candidate_test_labels_read": False,
        "ranker_manifest": _record(root / FIXED_INPUTS["ranker"]),
        "contribution_manifest": contribution_manifest,
        "candidate_contributions": artifact,
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def _append_source_access(
    root: Path, destination: Path, sources: Mapping[str, Any]
) -> None:
    fixed_records = {
        name: sources[name]
        for name in (*FIXED_INPUTS, "source_closure")
        if name in sources
    }
    output = _record(destination)
    append_access_log(
        root,
        {
            "event": "prelock_label_free_test_stage",
            "event_id": canonical_sha256(
                {"stage": "d1_postformal_label_free_sources", "output": output}
            )[:24],
            "stage": "d1_postformal_label_free_sources",
            "purpose": "fixed label-free covariates/runtime evidence",
            "inputs": fixed_records,
            "input_source_signature_sha256": canonical_sha256(sources),
            "output_manifest": str(destination.resolve()),
            "output_manifest_sha256": output["sha256"],
            "candidate_labels_opened_as_table": False,
            "candidate_test_labels_read": False,
        },
    )


def assemble_postformal_sources(
    run_dir: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    """Build fixed prelock evidence sources; no caller-provided data paths are accepted."""

    root = Path(run_dir).expanduser().resolve()
    destination = root / SOURCE_MANIFEST_RELATIVE_PATH
    if any(
        (root / relative).exists()
        for relative in (
            "08_lock/FORMAL_TEST_LOCK.json",
            "FINAL_RUN_LOCK.json",
            "COMPLETE",
        )
    ):
        raise PermissionError("D1 postformal sources must be built before formal lock")
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"D1 postformal sources already exist; pass --resume: {destination}"
            )
        _, existing = load_postformal_sources(root)
        _append_source_access(root, destination, existing["sources"])
        return existing
    closure_path, closure = load_source_closure(root)
    test_inputs = closure.get("canonical_inputs", {}).get("test", {})
    if not isinstance(test_inputs, Mapping) or not isinstance(
        test_inputs.get("paired_manifest"), Mapping
    ):
        raise RuntimeError("D1 source closure Test paired manifest is absent")
    paired_path = root / FIXED_INPUTS["paired"]
    paired_record = _record(paired_path)
    if paired_record["sha256"] != test_inputs["paired_manifest"].get("sha256"):
        raise RuntimeError("D1 local/closure paired Test bytes differ")

    manifests = {
        name: _manifest(root, name) for name in FIXED_INPUTS if name != "paired"
    }
    contribution_value = _contributions_declaration(root, manifests["ranker"])
    raw_path = _artifact(
        manifests["raw_features"], ("candidate_features",), name="D1 raw Top5 features"
    )
    context_path = _artifact(
        manifests["raw_features"], ("sample_context",), name="D1 raw sample context"
    )
    t2_path = _artifact(
        manifests["t2_features"], ("candidate_features",), name="D1 T2 features"
    )
    t3_path = _artifact(
        manifests["t3_features"], ("candidate_features",), name="D1 T3 features"
    )
    candidate_path = _artifact(
        manifests["candidates"], ("top5",), name="D1 Top5 candidates"
    )
    gate_path = _artifact(manifests["gate"], ("decisions",), name="D1 gate decisions")
    ranker_path = _artifact(
        manifests["ranker"], ("per_sample_decisions",), name="D1 ranker decisions"
    )
    for path, name in (
        (raw_path, "raw feature"),
        (context_path, "sample context"),
        (t2_path, "T2 feature"),
        (t3_path, "T3 feature"),
        (candidate_path, "candidate"),
        (gate_path, "gate decision"),
        (ranker_path, "ranker decision"),
    ):
        assert_label_free_parquet_schema(path, name=f"D1 postformal {name}")

    denominator = pd.read_parquet(paired_path, columns=["sample_id"])
    context = pd.read_parquet(
        context_path,
        columns=[
            "sample_id",
            "language",
            "predicted_mask_path",
            "predicted_mask_sha256",
            "candidate_count",
        ],
    )
    raw_columns = [
        "sample_id",
        "candidate_id",
        "native_rank",
        "mask_reliability",
        "rectangle_probability_mean",
        "depth_missing",
        "number_of_nearby_candidates",
    ]
    raw = pd.read_parquet(raw_path, columns=raw_columns)
    t2 = pd.read_parquet(t2_path, columns=["sample_id", "candidate_id"])
    t3 = pd.read_parquet(t3_path, columns=["sample_id", "candidate_id"])
    candidates = pd.read_parquet(
        candidate_path, columns=["sample_id", "candidate_id", "native_rank"]
    )
    gate = pd.read_parquet(
        gate_path, columns=["sample_id", "native_candidate_id", "selected_candidate_id"]
    )
    ranker = pd.read_parquet(
        ranker_path,
        columns=["sample_id", "native_candidate_id", "selected_candidate_id"],
    )
    for frame in (denominator, context, raw, t2, t3, candidates, gate, ranker):
        frame["sample_id"] = frame["sample_id"].astype(str)
    sample_ids = denominator["sample_id"].tolist()
    expected_count = int(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        .get("denominator_contract", {})
        .get("test_samples", -1)
    )
    if (
        expected_count <= 0
        or len(sample_ids) != expected_count
        or len(set(sample_ids)) != expected_count
        or context["sample_id"].duplicated().any()
        or set(context["sample_id"]) != set(sample_ids)
    ):
        raise RuntimeError("D1 postformal source denominator differs")
    candidate_keys = set(
        map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    if (
        set(map(tuple, raw[["sample_id", "candidate_id"]].astype(str).to_numpy()))
        != candidate_keys
        or set(map(tuple, t2[["sample_id", "candidate_id"]].astype(str).to_numpy()))
        != candidate_keys
        or set(map(tuple, t3[["sample_id", "candidate_id"]].astype(str).to_numpy()))
        != candidate_keys
        or gate["sample_id"].duplicated().any()
        or ranker["sample_id"].duplicated().any()
        or set(gate["sample_id"]) != set(sample_ids)
        or set(ranker["sample_id"]) != set(sample_ids)
    ):
        raise RuntimeError("D1 postformal source candidate/decision membership differs")
    # The gate may reject a challenger, but its native identity must still
    # equal the ranker native identity. Gate-selected membership is checked below.
    if (
        not gate["native_candidate_id"]
        .fillna("")
        .astype(str)
        .equals(ranker["native_candidate_id"].fillna("").astype(str))
    ):
        raise RuntimeError("D1 gate/ranker native identity differs")

    raw_index = raw.assign(
        sample_id=raw["sample_id"].astype(str),
        candidate_id=raw["candidate_id"].astype(str),
    ).set_index(["sample_id", "candidate_id"])
    rows = []
    asset_records = []
    context_index = context.set_index("sample_id")
    gate_index = gate.set_index("sample_id")
    for sample_id in sample_ids:
        source = context_index.loc[sample_id]
        mask_fraction, mask_record = _predicted_mask_fraction(
            Path(str(source["predicted_mask_path"])).expanduser().resolve()
        )
        if mask_record["sha256"] != str(source["predicted_mask_sha256"]):
            raise RuntimeError("D1 predicted-mask asset/context hash differs")
        asset_records.append(mask_record)
        native_value = gate_index.loc[sample_id, "native_candidate_id"]
        selected_value = gate_index.loc[sample_id, "selected_candidate_id"]
        native_id = "" if pd.isna(native_value) else str(native_value)
        selected_id = "" if pd.isna(selected_value) else str(selected_value)
        candidate_count = int(source["candidate_count"])
        if candidate_count == 0:
            native = selected = None
        else:
            if (sample_id, native_id) not in raw_index.index or (
                selected_id and (sample_id, selected_id) not in raw_index.index
            ):
                raise RuntimeError(
                    "D1 postformal selected identity is outside Top5 features"
                )
            native = raw_index.loc[(sample_id, native_id)]
            selected = raw_index.loc[(sample_id, selected_id or native_id)]
        rows.append(
            {
                "sample_id": sample_id,
                "target_size": (
                    "small"
                    if mask_fraction < 0.05
                    else "large"
                    if mask_fraction >= 0.20
                    else "medium"
                ),
                "relation_query": bool(
                    RELATION_PATTERN.search(str(source["language"]))
                ),
                "clutter": bool(
                    candidate_count > 0
                    and float(native["number_of_nearby_candidates"]) >= 2.0
                ),
                "depth_missing": bool(
                    candidate_count == 0 or float(native["depth_missing"]) >= 0.5
                ),
                "predicted_mask_confidence": (
                    math.nan
                    if candidate_count == 0
                    else float(native["mask_reliability"])
                ),
                "native_mask_support": (
                    math.nan
                    if candidate_count == 0
                    else float(native["rectangle_probability_mean"])
                ),
                "selected_mask_support": (
                    math.nan
                    if candidate_count == 0
                    else float(selected["rectangle_probability_mean"])
                ),
            }
        )
    covariates = pd.DataFrame(rows, columns=COVARIATE_COLUMNS)
    covariate_path = _atomic_parquet(covariates, root / COVARIATES_RELATIVE_PATH)

    source_records = {
        name: _record(root / relative) for name, relative in FIXED_INPUTS.items()
    }
    source_records["source_closure"] = _record(closure_path)
    runtime_dir = root / OUTPUT_ROOT / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_telemetry = manifests["raw_features"].get("telemetry", {})
    ranker_telemetry = manifests["ranker"].get("telemetry", {})
    feature_latency = _finite_nonnegative(
        manifests["raw_features"].get(
            "feature_extraction_latency_ms",
            raw_telemetry.get("feature_latency_ms")
            if isinstance(raw_telemetry, Mapping)
            else None,
        )
    )
    feature_peak = _finite_nonnegative(
        raw_telemetry.get("peak_memory_mb")
        if isinstance(raw_telemetry, Mapping)
        else None
    )
    ranker_latency = _finite_nonnegative(
        ranker_telemetry.get("ranker_latency_ms")
        if isinstance(ranker_telemetry, Mapping)
        else None
    )
    ranker_peak = _finite_nonnegative(
        ranker_telemetry.get("peak_memory_mb")
        if isinstance(ranker_telemetry, Mapping)
        else None
    )
    parameter_count = (
        int(ranker_telemetry["parameter_count"])
        if isinstance(ranker_telemetry, Mapping)
        and isinstance(ranker_telemetry.get("parameter_count"), (int, float))
        else None
    )
    applications = manifests["ranker"].get("seed_applications")
    model_paths: list[Path] = []
    if isinstance(applications, Mapping):
        for seed, application in applications.items():
            if not isinstance(application, Mapping) or not isinstance(
                application.get("model"), Mapping
            ):
                model_paths = []
                break
            model_paths.append(
                verified_artifact_path(
                    application["model"], name=f"D1 ranker seed {seed} model"
                )
            )
    model_bytes = (
        sum(path.stat().st_size for path in model_paths) if model_paths else None
    )
    runtime_values = {
        "dexnet_candidate_generation": _runtime_payload(
            component="dexnet_candidate_generation",
            sources={"candidate_manifest": source_records["candidates"]},
            latency=None,
            peak_memory=None,
            measurement_semantics="historical_snapshot_candidate_generation_separate_from_reranking",
            unavailable_reason="Snapshot A did not persist Dex-Net candidate-generation wall time or peak memory",
        ),
        "feature_extraction": _runtime_payload(
            component="feature_extraction",
            sources={
                "raw_feature_manifest": source_records["raw_features"],
                "feature_execution": source_records["feature_execution"],
            },
            latency=feature_latency,
            peak_memory=feature_peak,
            measurement_semantics="feature_extraction_not_artifact_loading",
            unavailable_reason=(
                ""
                if feature_latency is not None and feature_peak is not None
                else "persisted extractor latency/peak memory is incomplete"
            ),
        ),
        "ranker_inference": _runtime_payload(
            component="ranker_inference",
            sources={"ranker_manifest": source_records["ranker"]},
            latency=ranker_latency,
            peak_memory=ranker_peak,
            measurement_semantics="persisted ranker inference wall time per candidate",
            parameter_count=parameter_count,
            model_bytes=model_bytes,
            unavailable_reason=(
                ""
                if None
                not in (ranker_latency, ranker_peak, parameter_count, model_bytes)
                else "persisted ranker latency/model telemetry is incomplete"
            ),
        ),
        "gate_inference": _runtime_payload(
            component="gate_inference",
            sources={"gate_manifest": source_records["gate"]},
            latency=None,
            peak_memory=None,
            measurement_semantics="gate inference distinct from ranker inference",
            unavailable_reason="locked gate manifest did not persist gate-only wall time or peak memory",
        ),
        "total_d1_pipeline": _runtime_payload(
            component="total_d1_pipeline",
            sources={
                name: source_records[name]
                for name in ("candidates", "raw_features", "ranker", "gate")
            },
            latency=None,
            peak_memory=None,
            measurement_semantics="candidate generation plus incremental D1 reranking; never inferred from feature loading",
            unavailable_reason="candidate-generation and gate-only telemetry are unavailable; partial times are reported separately",
        ),
    }
    runtime_records = {}
    for component, value in runtime_values.items():
        path = runtime_dir / f"{component}.json"
        atomic_json(path, value)
        runtime_records[component] = _record(path)

    contribution_path = root / CONTRIBUTIONS_RELATIVE_PATH
    atomic_json(contribution_path, contribution_value)
    sources = {
        **source_records,
        "predicted_mask_assets": asset_records,
        "predicted_mask_asset_inventory_sha256": canonical_sha256(asset_records),
        "predicted_mask_asset_count": len(asset_records),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "candidate_test_labels_read": False,
        "selection_used_test_metrics": False,
        "sample_count": len(covariates),
        "covariate_schema": list(COVARIATE_COLUMNS),
        "fixed_source_paths": FIXED_INPUTS,
        "configuration": {
            "target_size_fraction_thresholds": {"small_lt": 0.05, "large_ge": 0.20},
            "clutter_nearby_candidate_threshold": 2,
            "relation_regex": RELATION_PATTERN.pattern,
        },
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": {
            "sample_covariates": _record(covariate_path),
            "runtime_manifests": runtime_records,
            "ranker_contributions": _record(contribution_path),
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(destination, result)
    _append_source_access(root, destination, sources)
    return result


def load_postformal_sources(run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(run_dir).expanduser().resolve()
    path = root / SOURCE_MANIFEST_RELATIVE_PATH
    value = load_content_manifest(
        path, name="D1 postformal sources", statuses=("COMPLETE",)
    )
    if (
        value.get("candidate_test_labels_read") is not False
        or value.get("selection_used_test_metrics") is not False
        or value.get("covariate_schema") != list(COVARIATE_COLUMNS)
        or value.get("fixed_source_paths") != FIXED_INPUTS
        or value.get("source_signature_sha256")
        != canonical_sha256(value.get("sources"))
    ):
        raise RuntimeError("D1 postformal source contract differs")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "sample_covariates",
        "runtime_manifests",
        "ranker_contributions",
    }:
        raise RuntimeError("D1 postformal source artifact inventory differs")
    covariate_path = verified_artifact_path(
        artifacts["sample_covariates"], name="D1 fixed sample covariates"
    )
    try:
        covariate_columns = assert_label_free_parquet_schema(
            covariate_path, name="D1 fixed sample covariates"
        )
    except PermissionError as error:
        raise PermissionError(
            "D1 preformal covariates expose outcome/GT-derived fields"
        ) from error
    forbidden_covariates = [
        column
        for column in covariate_columns
        if any(token in column.lower() for token in FORBIDDEN_COVARIATE_TOKENS)
    ]
    if forbidden_covariates:
        raise PermissionError(
            "D1 preformal covariates expose outcome/GT-derived fields: "
            f"{sorted(forbidden_covariates)}"
        )
    if (
        covariate_path != (root / COVARIATES_RELATIVE_PATH).resolve()
        or covariate_columns != COVARIATE_COLUMNS
    ):
        raise RuntimeError("D1 postformal source covariate artifact differs")
    runtime = artifacts["runtime_manifests"]
    if not isinstance(runtime, Mapping) or set(runtime) != set(RUNTIME_COMPONENTS):
        raise RuntimeError("D1 runtime evidence component inventory differs")
    for component in RUNTIME_COMPONENTS:
        runtime_path = verified_artifact_path(
            runtime[component], name=f"D1 fixed runtime {component}"
        )
        if (
            runtime_path
            != (root / OUTPUT_ROOT / "runtime" / f"{component}.json").resolve()
        ):
            raise RuntimeError(f"D1 fixed runtime path differs: {component}")
        runtime_value = load_content_manifest(
            runtime_path,
            name=f"D1 fixed runtime {component}",
            statuses=("COMPLETE", "NOT_AVAILABLE"),
        )
        if (
            runtime_value.get("component") != component
            or runtime_value.get("candidate_test_labels_read") is not False
        ):
            raise RuntimeError(f"D1 fixed runtime semantics differ: {component}")
    contribution_path = verified_artifact_path(
        artifacts["ranker_contributions"], name="D1 fixed ranker contributions"
    )
    if contribution_path != (root / CONTRIBUTIONS_RELATIVE_PATH).resolve():
        raise RuntimeError("D1 fixed ranker-contribution path differs")
    contribution = load_content_manifest(
        contribution_path,
        name="D1 fixed ranker contributions",
        statuses=("COMPLETE", "NOT_APPLICABLE"),
    )
    if contribution.get("candidate_test_labels_read") is not False:
        raise PermissionError("D1 fixed ranker contributions are not label-free")
    if contribution.get("status") == "COMPLETE":
        if contribution.get("selected_method") not in {"R5", "LAMBDAMART"}:
            raise RuntimeError("D1 fixed contribution method differs")
        contribution_manifest = contribution.get("contribution_manifest")
        if not isinstance(contribution_manifest, Mapping):
            raise RuntimeError("D1 fixed contribution manifest record is absent")
        manifest_path, manifest = validate_ranker_contributions(root)
        expected_manifest = _record(manifest_path)
        if any(
            contribution_manifest.get(key) != expected_manifest.get(key)
            for key in ("path", "sha256", "bytes")
        ):
            raise RuntimeError("D1 fixed contribution manifest differs")
        expected_artifact = manifest.get("artifacts", {}).get(
            "candidate_contributions"
        )
        observed_artifact = contribution.get("candidate_contributions")
        if not isinstance(observed_artifact, Mapping) or any(
            observed_artifact.get(key) != expected_artifact.get(key)
            for key in ("path", "sha256", "bytes")
            if key in expected_artifact
        ):
            raise RuntimeError("D1 fixed contribution artifact differs")
    elif contribution.get("candidate_contributions") is not None:
        raise RuntimeError("D1 non-applicable contribution declaration has an artifact")
    verify_artifact_records_recursive(
        {"sources": value.get("sources"), "artifacts": artifacts},
        name="D1 postformal sources",
        require_at_least_one=True,
    )
    return path, value
