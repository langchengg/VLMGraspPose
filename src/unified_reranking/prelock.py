"""Fail-closed assembly of the label-free P11 pre-lock artifact bundle.

The module deliberately has no Test-label reader.  The predeclared Test label
source is opened only as an opaque byte stream by :func:`sha256_file`; its
Parquet schema and rows remain unavailable until the formal execution claim.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import verify_artifact_records_recursive
from .contracts import assert_model_feature_columns
from .candidates import verify_candidate_contract_hashes
from .hashing import atomic_json, atomic_text, canonical_sha256, sha256_file
from .prelock_validation import (
    validate_application_signature,
    validate_encoder_execution,
    validate_feature_ablation_manifest,
    validate_feature_extraction_benchmark,
    validate_gate_application_outputs,
    validate_gate_selection,
    validate_matrix_phase_execution,
    validate_router_application_outputs,
    validate_router_selection,
    validate_screen_selection,
    validate_scalar_selection,
    validate_union_application_outputs,
    validate_union_headroom,
    validate_union_selection,
)
from .test_bridge import validate_label_free_test_bridge_manifest


ROUTES = ("crog", "g1", "c1")
FORMAL_SEEDS = (42, 123, 2026)
PRIMARY_TRACK = "T2_matched_common"
LABEL_NORMALIZATION = {
    "route_column": "method",
    "variant_column": "variant",
    "include_variants": ["crog_native", "g1", "c1"],
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUPERVISION_TOKENS = (
    "candidate_success",
    "jacquard_margin",
    "matched_gt",
    "ground_truth",
    "best_same_gt",
    "first_positive",
    "native_correct",
    "challenger_correct",
    "selected_correct",
    "oracle",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _regular_file(path: str | Path, name: str) -> Path:
    result = Path(path).expanduser().resolve()
    if result.is_symlink() or not result.is_file():
        raise ValueError(f"{name} is not a regular file: {result}")
    return result


def artifact_record(path: str | Path) -> dict[str, str]:
    result = _regular_file(path, "artifact")
    return {"path": str(result), "sha256": sha256_file(result)}


def _verify_record(record: Mapping[str, Any], name: str) -> Path:
    path = _regular_file(str(record.get("path", "")), name)
    digest = str(record.get("sha256", ""))
    if not _SHA256.fullmatch(digest) or sha256_file(path) != digest:
        raise RuntimeError(f"artifact hash mismatch: {name}")
    return path


def _complete_manifest(path: Path, name: str, statuses: Sequence[str] = ("COMPLETE",)) -> dict[str, Any]:
    value = _read_json(path)
    if str(value.get("status", "")) not in set(statuses):
        raise RuntimeError(f"{name} is not complete: {path}")
    return value


def _assert_label_free_columns(frame: pd.DataFrame, name: str) -> None:
    leaked = sorted(
        column
        for column in map(str, frame.columns)
        if any(token in column.lower() for token in _SUPERVISION_TOKENS)
        or column.lower().startswith("gt_")
        or column.lower().startswith("j_at_")
    )
    if leaked:
        raise PermissionError(f"{name} contains Test supervision columns: {leaked}")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sample_manifest(run_dir: Path) -> tuple[Path, pd.DataFrame]:
    path = _regular_file(run_dir / "01_manifests" / "paired_test.parquet", "paired Test manifest")
    frame = pd.read_parquet(path, columns=["sample_id", "scene_id", "frame_id"])
    for column in ("sample_id", "scene_id", "frame_id"):
        if frame[column].isna().any() or frame[column].astype(str).eq("").any():
            raise ValueError(f"paired Test manifest contains empty {column}")
        frame[column] = frame[column].astype(str)
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError("paired Test sample IDs must be unique and non-empty")
    return path, frame


def _candidate_pool(path: Path, route: str, sample_ids: set[str]) -> pd.DataFrame:
    required = {
        "sample_id",
        "candidate_id",
        "native_rank",
        "candidate_geometry_sha256",
    }
    frame = pd.read_parquet(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{route} candidate pool misses columns: {missing}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["candidate_geometry_sha256"] = frame["candidate_geometry_sha256"].astype(str)
    rank = pd.to_numeric(frame["native_rank"], errors="coerce")
    if (
        rank.isna().any()
        or not np.equal(rank, np.floor(rank)).all()
        or (rank < 1).any()
        or frame["candidate_id"].eq("").any()
        or frame["candidate_geometry_sha256"].eq("").any()
        or not set(frame["sample_id"]).issubset(sample_ids)
        or frame.duplicated(["sample_id", "candidate_id"]).any()
        or frame.duplicated(["sample_id", "native_rank"]).any()
    ):
        raise ValueError(f"{route} candidate pool violates identity/rank contracts")
    frame["native_rank"] = rank.astype(int)
    for sample_id, group in frame.groupby("sample_id", sort=False):
        observed = sorted(group["native_rank"].tolist())
        if observed != list(range(1, len(observed) + 1)):
            raise ValueError(f"{route}/{sample_id} native ranks are not contiguous")
    return frame


def _load_candidate_pools(
    run_dir: Path, samples: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any], dict[str, Path]]:
    sample_ids = set(samples["sample_id"])
    all_pools: dict[str, pd.DataFrame] = {}
    top5_pools: dict[str, pd.DataFrame] = {}
    manifest_records: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for route in ROUTES:
        all_path = _regular_file(
            run_dir / "02_candidates" / f"{route}_test_all.parquet",
            f"{route} Test All pool",
        )
        top5_path = _regular_file(
            run_dir / "02_candidates" / f"{route}_test_top5.parquet",
            f"{route} Test Top-5 pool",
        )
        all_frame = _candidate_pool(all_path, route, sample_ids)
        top5_frame = _candidate_pool(top5_path, route, sample_ids)
        keys = ["sample_id", "candidate_id"]
        expected = all_frame.loc[all_frame["native_rank"].le(5), keys]
        if set(map(tuple, expected.to_numpy())) != set(map(tuple, top5_frame[keys].to_numpy())):
            raise ValueError(f"{route} Top-5 membership is not the native prefix of All")
        checked = top5_frame[keys + ["native_rank", "candidate_geometry_sha256"]].merge(
            all_frame[keys + ["native_rank", "candidate_geometry_sha256"]],
            on=keys,
            validate="one_to_one",
            suffixes=("_top5", "_all"),
        )
        if not checked["native_rank_top5"].equals(checked["native_rank_all"]) or not checked[
            "candidate_geometry_sha256_top5"
        ].equals(checked["candidate_geometry_sha256_all"]):
            raise ValueError(f"{route} Top-5 geometry/native-rank binding differs from All")
        all_pools[route] = all_frame
        top5_pools[route] = top5_frame
        paths[f"{route}_all"] = all_path
        paths[f"{route}_top5"] = top5_path
        manifest_records[route] = {
            "path": str(all_path),
            "sha256": sha256_file(all_path),
            "rows": int(len(all_frame)),
            "samples_with_candidates": int(all_frame["sample_id"].nunique()),
            "no_output_samples": int(len(samples) - all_frame["sample_id"].nunique()),
            "top5": {
                "path": str(top5_path),
                "sha256": sha256_file(top5_path),
                "rows": int(len(top5_frame)),
            },
        }
    return all_pools, top5_pools, manifest_records, paths


def _discover_source_artifact(run_dir: Path, key: str) -> tuple[Path, str | None]:
    audit_path = _regular_file(run_dir / "00_audit" / "fair_source_audit.json", "fair source audit")
    audit = _read_json(audit_path)
    record = dict(audit.get("artifacts", {})).get(key)
    if not isinstance(record, dict):
        raise RuntimeError(f"fair source audit does not declare {key}")
    path = _regular_file(str(record.get("path", "")), key)
    expected = record.get("sha256")
    if expected is not None and sha256_file(path) != expected:
        raise RuntimeError(f"fair source audit hash mismatch: {key}")
    return path, None if expected is None else str(expected)


def discover_candidate_test_labels(run_dir: Path) -> Path:
    """Return the fair source label table path without opening its table content."""

    path, _ = _discover_source_artifact(run_dir, "canonical_candidates_labels_locked")
    return path


def discover_evaluator(run_dir: Path) -> Path:
    copied = run_dir / "configs" / "canonical_evaluator.py"
    source, expected = _discover_source_artifact(run_dir, "evaluator")
    if copied.is_file():
        copied = copied.resolve()
        if sha256_file(copied) != sha256_file(source):
            raise RuntimeError("run-local and fair-source evaluator hashes differ")
        return copied
    if expected is not None and sha256_file(source) != expected:
        raise RuntimeError("frozen evaluator hash differs from fair source audit")
    return source


def collect_code_files(code_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    files: set[Path] = set()
    for raw in code_roots:
        path = Path(raw).expanduser().resolve()
        if path.is_symlink() or not path.exists():
            raise ValueError(f"code root does not exist or is a symlink: {path}")
        if path.is_file():
            files.add(path)
        else:
            files.update(
                candidate.resolve()
                for candidate in path.rglob("*.py")
                if candidate.is_file() and not candidate.is_symlink() and "__pycache__" not in candidate.parts
            )
    if not files:
        raise ValueError("code bundle contains no regular files")
    return tuple(sorted(files, key=str))


def code_bundle(files: Sequence[Path]) -> dict[str, Any]:
    records = [{"path": str(path), "sha256": sha256_file(path)} for path in files]
    return {
        "schema_version": 1,
        "algorithm": "canonical_sha256(path-and-file-sha256-records)",
        "files": records,
        "bundle_sha256": canonical_sha256(records),
    }


def _selected_ranker_contract(run_dir: Path, route: str, selection: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(selection)
    validation_manifest_path = _regular_file(selected.get("validation_manifest", ""), f"{route} Validation ensemble")
    validation_sha256 = str(selected.get("validation_manifest_sha256", ""))
    if not _SHA256.fullmatch(validation_sha256) or sha256_file(validation_manifest_path) != validation_sha256:
        raise RuntimeError(f"{route} selected Validation ensemble hash mismatch")
    validation = _complete_manifest(validation_manifest_path, f"{route} Validation ensemble")
    identity = dict(validation.get("identity", {}))
    if (
        identity.get("route") != route
        or identity.get("track") != PRIMARY_TRACK
        or list(identity.get("seeds", [])) != list(FORMAL_SEEDS)
    ):
        raise ValueError(f"{route} selected ensemble identity violates the primary contract")
    for field in ("method_code", "encoder", "loss"):
        if str(selected.get(field, "")) != str(identity.get(field, "")):
            raise ValueError(f"{route} selected {field} differs from Validation ensemble identity")
    if str(selected.get("ensemble_id", "")) != str(validation.get("ensemble_id", "")):
        raise ValueError(f"{route} selected ensemble_id differs from Validation ensemble")
    oof_manifest_path = _regular_file(
        selected.get("oof_manifest", ""), f"{route} OOF ensemble"
    )
    oof_sha256 = str(selected.get("oof_manifest_sha256", ""))
    if not _SHA256.fullmatch(oof_sha256) or sha256_file(oof_manifest_path) != oof_sha256:
        raise RuntimeError(f"{route} selected OOF ensemble hash mismatch")
    oof = _complete_manifest(oof_manifest_path, f"{route} OOF ensemble")
    oof_identity = dict(oof.get("identity", {}))
    if oof_identity != identity or str(oof.get("ensemble_id", "")) != str(
        validation.get("ensemble_id", "")
    ):
        raise ValueError(f"{route} OOF/Validation ensemble identity mismatch")
    source_records = list(dict(validation.get("sources", {})).get("matrix_manifests", []))
    cells: dict[int, dict[str, Any]] = {}
    cell_records: dict[int, dict[str, str]] = {}
    for index, record in enumerate(source_records):
        path = _verify_record(record, f"{route} Validation cell {index}")
        cell = _complete_manifest(path, f"{route} Validation cell {index}")
        configuration = dict(cell.get("configuration", {}))
        if configuration.get("mode") != "validation":
            continue
        seed = int(configuration.get("seed", -1))
        if seed in cells:
            raise ValueError(f"duplicate {route} Validation cell seed: {seed}")
        cells[seed] = cell
        cell_records[seed] = artifact_record(path)
    if set(cells) != set(FORMAL_SEEDS):
        raise ValueError(f"{route} selected ensemble lacks exactly the three formal seeds")
    feature_sets = {tuple(map(str, cell.get("feature_columns", []))) for cell in cells.values()}
    if len(feature_sets) != 1:
        raise ValueError(f"{route} selected seed cells use different feature schemas")
    features = next(iter(feature_sets))
    if not features:
        raise ValueError(f"{route} selected ranker has no features")
    assert_model_feature_columns(features)
    for seed, cell in cells.items():
        configuration = dict(cell["configuration"])
        if (
            configuration.get("route") != route
            or configuration.get("track") != PRIMARY_TRACK
            or configuration.get("encoder") != identity.get("encoder")
            or configuration.get("loss") != identity.get("loss")
            or int(configuration.get("seed", -1)) != seed
        ):
            raise ValueError(f"{route} seed-{seed} Validation cell differs from selected identity")
    return {
        "selection": selected,
        "validation_manifest": artifact_record(validation_manifest_path),
        "oof_manifest": artifact_record(oof_manifest_path),
        "identity": identity,
        "feature_columns": list(features),
        "cells": cells,
        "cell_manifests": {str(seed): cell_records[seed] for seed in FORMAL_SEEDS},
    }


def _load_development_selections(run_dir: Path) -> dict[str, Any]:
    screen_execution_path = _regular_file(
        run_dir / "05_models/matrix_plans/screen_latest_execution.json",
        "screen latest execution",
    )
    validate_matrix_phase_execution(run_dir, "screen")
    screen_selection_path = _regular_file(
        run_dir / "07_validation/screen_selection_manifest.json",
        "screen selection manifest",
    )
    validate_screen_selection(screen_selection_path)
    screen_finalists_path = _regular_file(
        run_dir / "05_models/screen_finalists.json",
        "screen finalists",
    )
    selected_execution_path = _regular_file(
        run_dir / "05_models/matrix_plans/selected_latest_execution.json",
        "selected latest execution",
    )
    validate_matrix_phase_execution(
        run_dir,
        "selected",
        expected_selection=artifact_record(screen_finalists_path),
    )
    selection_path = _regular_file(
        run_dir / "07_validation" / "selected_primary_ungated.json",
        "primary Validation ranker selection",
    )
    selection = validate_scalar_selection(selection_path)
    if selection.get("primary_track") != PRIMARY_TRACK or set(selection.get("selections", {})) != set(ROUTES):
        raise ValueError("primary ranker selection must contain exactly three T2 routes")
    routes = {
        route: _selected_ranker_contract(run_dir, route, selection["selections"][route])
        for route in ROUTES
    }
    encoder_selection_path = _regular_file(
        run_dir / "05_models" / "encoder_loss_selections.json",
        "encoder loss selection",
    )
    encoder_selection_record = artifact_record(encoder_selection_path)
    encoder_execution_path = _regular_file(
        run_dir / "05_models" / "matrix_plans" / "encoder_latest_execution.json",
        "encoder latest execution",
    )
    validate_encoder_execution(
        encoder_execution_path, expected_selection=encoder_selection_record
    )
    validate_matrix_phase_execution(
        run_dir,
        "encoder",
        expected_selection=encoder_selection_record,
    )
    ablation_path = _regular_file(
        run_dir
        / "07_validation"
        / "ablations"
        / "feature_ablation_manifest.json",
        "feature ablation manifest",
    )
    validate_feature_ablation_manifest(
        ablation_path, expected_selection=artifact_record(selection_path)
    )
    feature_benchmark_path = _regular_file(
        run_dir
        / "07_validation"
        / "telemetry"
        / "feature_extraction_benchmark.json",
        "Validation feature extraction benchmark",
    )
    validate_feature_extraction_benchmark(feature_benchmark_path)
    return {
        "path": selection_path,
        "manifest": selection,
        "routes": routes,
        "screen_execution_record": artifact_record(screen_execution_path),
        "screen_selection_record": artifact_record(screen_selection_path),
        "screen_finalists_record": artifact_record(screen_finalists_path),
        "selected_execution_record": artifact_record(selected_execution_path),
        "encoder_selection_record": encoder_selection_record,
        "encoder_execution_record": artifact_record(encoder_execution_path),
        "feature_ablation_record": artifact_record(ablation_path),
        "feature_benchmark_record": artifact_record(feature_benchmark_path),
    }


def _load_route_prerequisites(run_dir: Path, selected: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route in ROUTES:
        calibration_path = _regular_file(
            run_dir / "05_calibration" / f"{route}_calibration_manifest.json",
            f"{route} calibration",
        )
        calibration = _complete_manifest(calibration_path, f"{route} calibration")
        calibration_application_path = _regular_file(
            run_dir / "05_calibration" / f"{route}_test_application_manifest.json",
            f"{route} label-free Test calibration",
        )
        calibration_application = _complete_manifest(
            calibration_application_path,
            f"{route} label-free Test calibration",
            statuses=("COMPLETE_LABEL_FREE",),
        )
        if calibration_application.get("candidate_labels_loaded") is not False:
            raise PermissionError(f"{route} Test calibration lacks a label-free certificate")
        feature_manifest_path = _regular_file(
            run_dir
            / "03_features"
            / "tracks"
            / PRIMARY_TRACK
            / f"{route}_test"
            / "feature_manifest.json",
            f"{route} T2 Test feature manifest",
        )
        feature_manifest = _complete_manifest(feature_manifest_path, f"{route} T2 Test feature manifest")
        if feature_manifest.get("labels_physically_separate") is not True:
            raise PermissionError(f"{route} Test feature manifest is not certified label-free")
        feature_artifact_path = _verify_record(
            feature_manifest.get("artifact", {}), f"{route} T2 Test candidate features"
        )
        for index, record in enumerate(feature_manifest.get("sources", [])):
            _verify_record(record, f"{route} T2 Test feature source {index}")
        if list(feature_manifest.get("model_feature_columns", [])) != selected["routes"][route]["feature_columns"]:
            raise ValueError(f"{route} selected and Test feature schemas differ")

        ranker_manifest_path = _regular_file(
            run_dir / "08_lock" / "label_free_test_rankers" / route / "manifest.json",
            f"{route} label-free Test ranker",
        )
        ranker = _complete_manifest(ranker_manifest_path, f"{route} label-free Test ranker")
        if ranker.get("candidate_test_labels_read") is not False or ranker.get("label_free_test_inference") is not True:
            raise PermissionError(f"{route} Test ranker lacks a label-free certificate")
        expected_selection = selected["routes"][route]["selection"]
        if dict(ranker.get("selection", {})) != expected_selection:
            raise ValueError(f"{route} Test ranker selection differs from Validation lock")
        ranker_sources = dict(ranker.get("sources", {}))
        ranker_tool = (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "unified_reranking"
            / "apply_selected_test_rankers.py"
        )
        expected_source_records = {
            "selection": artifact_record(selected["path"]),
            "validation_ensemble": selected["routes"][route]["validation_manifest"],
            "candidates": artifact_record(
                run_dir / "02_candidates" / f"{route}_test_top5.parquet"
            ),
            "denominator": artifact_record(
                run_dir / "01_manifests" / "paired_test.parquet"
            ),
            "implementation_tool": artifact_record(ranker_tool),
        }
        if set(ranker_sources) != {*expected_source_records, "applications"}:
            raise ValueError(f"{route} Test ranker source inventory is not exact")
        for source_name, expected_record in expected_source_records.items():
            observed = ranker_sources.get(source_name)
            if not isinstance(observed, dict) or {
                "path": str(Path(str(observed.get("path", ""))).resolve()),
                "sha256": observed.get("sha256"),
            } != expected_record:
                raise ValueError(f"{route} Test ranker {source_name} binding mismatch")
            _verify_record(observed, f"{route} Test ranker {source_name}")
        applications = ranker_sources.get("applications")
        if not isinstance(applications, dict) or set(applications) != {
            str(seed) for seed in FORMAL_SEEDS
        }:
            raise ValueError(f"{route} Test ranker application inventory is incomplete")
        seed_score_frames: dict[int, pd.DataFrame] = {}
        for seed in FORMAL_SEEDS:
            application_path = _verify_record(
                applications[str(seed)], f"{route} seed-{seed} Test application"
            )
            application = _complete_manifest(
                application_path, f"{route} seed-{seed} Test application"
            )
            if (
                application.get("candidate_test_labels_read") is not False
                or application.get("label_free_test_inference") is not True
                or int(application.get("identity", {}).get("seed", -1)) != seed
                or application.get("identity", {}).get("route") != route
                or application.get("identity", {}).get("track") != PRIMARY_TRACK
            ):
                raise ValueError(f"{route} seed-{seed} Test application contract mismatch")
            source_cell = application.get("sources", {}).get("cell_manifest")
            if source_cell != selected["routes"][route]["cell_manifests"][str(seed)]:
                raise ValueError(f"{route} seed-{seed} Test application source cell mismatch")
            if application.get("sources", {}).get("feature_manifest") != artifact_record(
                feature_manifest_path
            ):
                raise ValueError(f"{route} seed-{seed} Test feature binding mismatch")
            verify_artifact_records_recursive(
                {"sources": application.get("sources"), "artifact": application.get("artifact")},
                name=f"{route} seed-{seed} Test application",
                require_at_least_one=True,
            )
            application_scores_path = _verify_record(
                application.get("artifact", {}),
                f"{route} seed-{seed} Test scores",
            )
            application_scores = pd.read_parquet(application_scores_path)
            if not {"sample_id", "candidate_id", "score"}.issubset(
                application_scores.columns
            ):
                raise ValueError(f"{route} seed-{seed} Test scores schema mismatch")
            seed_score_frames[seed] = application_scores[
                ["sample_id", "candidate_id", "score"]
            ].rename(columns={"score": f"score_seed_{seed}"})
        expected_ranker_configuration = {
            "route": route,
            "primary_track": PRIMARY_TRACK,
            "method_code": expected_selection["method_code"],
            "encoder": expected_selection["encoder"],
            "loss": expected_selection["loss"],
            "ensemble_id": expected_selection["ensemble_id"],
            "seeds": list(FORMAL_SEEDS),
            "candidate_test_labels_read": False,
        }
        if ranker.get("configuration") != expected_ranker_configuration:
            raise ValueError(f"{route} Test ranker configuration differs from Validation lock")
        validate_application_signature(ranker, kind="ranker")
        ranker_scores = _verify_record(ranker["artifacts"]["predictions"], f"{route} Test ranker scores")
        ranker_decisions = _verify_record(ranker["artifacts"]["decisions"], f"{route} Test ranker decisions")
        recomputed_scores = pd.read_parquet(
            run_dir / "02_candidates" / f"{route}_test_top5.parquet",
            columns=["sample_id", "candidate_id", "native_rank"],
        )
        for seed in FORMAL_SEEDS:
            recomputed_scores = recomputed_scores.merge(
                seed_score_frames[seed],
                on=["sample_id", "candidate_id"],
                how="left",
                validate="one_to_one",
            )
        score_columns = [f"score_seed_{seed}" for seed in FORMAL_SEEDS]
        if recomputed_scores[score_columns].isna().any().any():
            raise ValueError(f"{route} seed Test scores do not exactly cover candidates")
        recomputed_scores["ensemble_score"] = recomputed_scores[score_columns].mean(
            axis=1
        )
        stored_scores = pd.read_parquet(ranker_scores)
        expected_columns = [
            "sample_id",
            "candidate_id",
            "native_rank",
            *score_columns,
            "ensemble_score",
        ]
        if list(stored_scores.columns) != expected_columns:
            raise ValueError(f"{route} Test ranker score schema differs from producer")
        try:
            pd.testing.assert_frame_equal(
                stored_scores.reset_index(drop=True),
                recomputed_scores.reset_index(drop=True),
                check_dtype=True,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as error:
            raise ValueError(
                f"{route} Test ranker scores differ from bound seed applications"
            ) from error

        gate_selection_path = _regular_file(
            run_dir / "08_lock" / "gates" / route / "gate_selection.json",
            f"{route} gate selection",
        )
        gate = validate_gate_selection(gate_selection_path)
        if gate.get("test_access") != "NONE" or gate.get("decision") not in {"GO", "NO_GO_NATIVE"}:
            raise PermissionError(f"{route} gate is not a development-only locked selection")
        gate_test_path = _regular_file(
            run_dir / "08_lock" / "label_free_test_gates" / route / "manifest.json",
            f"{route} label-free Test gate",
        )
        gate_test = _complete_manifest(gate_test_path, f"{route} label-free Test gate")
        if gate_test.get("candidate_test_labels_read") is not False:
            raise PermissionError(f"{route} Test gate lacks a label-free certificate")
        gate_decisions = _verify_record(gate_test["artifacts"]["decisions"], f"{route} Test gate decisions")
        if gate_test.get("decision") != gate.get("decision"):
            raise ValueError(f"{route} Test gate did not apply the Validation-locked decision")
        expected_operating_point = gate.get("selection", {}).get(
            "selected_operating_point"
        )
        if gate_test.get("selected_operating_point") != expected_operating_point:
            raise ValueError(f"{route} Test gate operating point differs from Validation lock")
        if gate_test.get("feature_columns") != gate.get("configuration", {}).get(
            "feature_columns"
        ):
            raise ValueError(f"{route} Test gate feature schema differs from Validation lock")
        gate_tool = (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "unified_reranking"
            / "apply_locked_test_gates.py"
        )
        expected_gate_sources = {
            "gate_selection": artifact_record(gate_selection_path),
            "ranker_decisions": artifact_record(ranker_decisions),
            "candidate_features": artifact_record(feature_artifact_path),
            "candidates": artifact_record(
                run_dir / "02_candidates" / f"{route}_test_top5.parquet"
            ),
            "sample_denominator": artifact_record(
                run_dir / "01_manifests" / "paired_test.parquet"
            ),
            "implementation_tool": artifact_record(gate_tool),
        }
        if gate_test.get("sources") != expected_gate_sources:
            raise ValueError(f"{route} Test gate source binding mismatch")
        if set(gate_test.get("artifacts", {})) != {"inputs", "decisions"}:
            raise ValueError(f"{route} Test gate artifact inventory is not exact")
        validate_application_signature(gate_test, kind="gate")
        validate_gate_application_outputs(gate_test_path, gate)
        result[route] = {
            "calibration": calibration,
            "calibration_record": artifact_record(calibration_path),
            "calibration_application_record": artifact_record(calibration_application_path),
            "feature_manifest": feature_manifest,
            "feature_manifest_record": artifact_record(feature_manifest_path),
            "feature_artifact_record": artifact_record(feature_artifact_path),
            "ranker_manifest_record": artifact_record(ranker_manifest_path),
            "ranker_scores": ranker_scores,
            "ranker_decisions": ranker_decisions,
            "gate": gate,
            "gate_selection_record": artifact_record(gate_selection_path),
            "gate_test_record": artifact_record(gate_test_path),
            "gate_decisions": gate_decisions,
        }
    return result


def _load_cross_route_prerequisites(
    run_dir: Path, sample_path: Path, evaluator_path: Path
) -> dict[str, Any]:
    router_selection_path = _regular_file(
        run_dir / "08_lock" / "route_router" / "route_router_selection.json",
        "route-router selection",
    )
    router = validate_router_selection(router_selection_path)
    if (
        router.get("test_access") != "NONE"
        or router.get("decision") not in {"GO", "NO_GO_CROG"}
        or dict(router.get("configuration", {})).get("default_route") != "CROG"
        or list(dict(router.get("configuration", {})).get("tie_break", [])) != ["G1", "C1"]
    ):
        raise PermissionError("route router is not a Validation-locked CROG-default policy")
    router_test_path = _regular_file(
        run_dir / "08_lock" / "route_router_test" / "manifest.json",
        "label-free Test route router",
    )
    router_test = _complete_manifest(router_test_path, "label-free Test route router")
    if router_test.get("candidate_test_labels_read") is not False:
        raise PermissionError("Test route router lacks a label-free certificate")
    router_decisions = _verify_record(router_test["artifacts"]["decisions"], "Test router decisions")
    router_model_record = router.get("artifacts", {}).get("transition_models")
    router_input_path = run_dir / "08_lock" / "route_router_inputs" / "test_label_free.parquet"
    router_input_manifest_path = router_input_path.parent / "manifest.json"
    router_tool = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "unified_reranking"
        / "apply_locked_route_router.py"
    )
    expected_router_sources = {
        "router_selection": artifact_record(router_selection_path),
        "router_models": dict(router_model_record or {}),
        "test_router_inputs": artifact_record(router_input_path),
        "test_router_input_manifest": artifact_record(router_input_manifest_path),
        "implementation_tool": artifact_record(router_tool),
    }
    if not isinstance(router_model_record, dict):
        raise ValueError("route router lacks a transition-model artifact")
    _verify_record(router_model_record, "route router transition models")
    if router_test.get("sources") != expected_router_sources:
        raise ValueError("Test route router source binding mismatch")
    expected_router_configuration = {
        "default_route": "CROG",
        "tie_break": ["G1", "C1"],
        "selection_decision": router["decision"],
        "selected_operating_point": router.get("selection", {}).get(
            "selected_operating_point"
        ),
        "feature_columns": router["configuration"]["feature_columns"],
        "candidate_test_labels_read": False,
    }
    if router_test.get("configuration") != expected_router_configuration:
        raise ValueError("Test route router configuration differs from Validation lock")
    validate_application_signature(router_test, kind="router")
    validate_router_application_outputs(router_test_path, router)
    union_path = _regular_file(
        run_dir / "07_validation" / "union_headroom" / "manifest.json",
        "Validation union-headroom decision",
    )
    union = validate_union_headroom(run_dir, union_path)
    if union.get("test_access") != "NONE" or union.get("decision") not in {
        "NO_UNION_HEADROOM",
        "UNION_HEADROOM_AVAILABLE",
    }:
        raise PermissionError("union-headroom decision is not development-only")
    union_selection = None
    union_application = None
    union_selection_record = None
    union_application_record = None
    union_predictions = None
    union_decisions = None
    if union["decision"] == "UNION_HEADROOM_AVAILABLE":
        selection_path = _regular_file(
            run_dir / "08_lock" / "union_ranker" / "selected_union_ranker.json",
            "selected union ranker",
        )
        union_selection = validate_union_selection(selection_path)
        if (
            union_selection.get("test_access") != "NONE"
            or not str(union_selection.get("selected_encoder", ""))
        ):
            raise PermissionError("union ranker is not Validation-locked")
        application_path = _regular_file(
            run_dir / "08_lock" / "union_ranker_test" / "manifest.json",
            "label-free Test union application",
        )
        union_application = _complete_manifest(
            application_path, "label-free Test union application"
        )
        if (
            union_application.get("candidate_test_labels_read") is not False
            or union_application.get("test_access") != "LABEL_FREE_INFERENCE_ONLY"
        ):
            raise PermissionError("union Test application lacks a label-free certificate")
        union_predictions = _verify_record(
            union_application["artifacts"]["predictions"], "union Test predictions"
        )
        union_decisions = _verify_record(
            union_application["artifacts"]["decisions"], "union Test decisions"
        )
        union_feature_manifest_path = _verify_record(
            union_application["sources"]["test_feature_manifest"],
            "union Test feature manifest",
        )
        union_feature_manifest = _complete_manifest(
            union_feature_manifest_path, "union Test feature manifest"
        )
        union_feature_columns = list(
            map(str, union_feature_manifest.get("model_feature_columns", []))
        )
        if not union_feature_columns:
            raise ValueError("union Test feature manifest has no model feature columns")
        selected_validation_record = union_selection.get(
            "selected_validation_manifest"
        )
        selected_validation_path = _verify_record(
            selected_validation_record,
            "selected union Validation ensemble",
        )
        selected_validation = _complete_manifest(
            selected_validation_path, "selected union Validation ensemble"
        )
        expected_union_cells = {
            str(index): dict(record)
            for index, record in enumerate(
                selected_validation.get("sources", {}).get("cells", [])
            )
        }
        observed_union_cells = dict(
            union_application.get("sources", {}).get("cells", {})
        )
        if sorted(observed_union_cells.values(), key=lambda item: item["path"]) != sorted(
            expected_union_cells.values(), key=lambda item: item["path"]
        ):
            raise ValueError("union Test application cell bindings differ from selected winner")
        union_feature_path = _verify_record(
            union_feature_manifest.get("artifacts", {}).get("features", {}),
            "union Test features",
        )
        union_tool = (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "unified_reranking"
            / "apply_locked_union_ranker.py"
        )
        expected_union_sources = {
            "selection": artifact_record(selection_path),
            "test_feature_manifest": artifact_record(union_feature_manifest_path),
            "test_features": artifact_record(union_feature_path),
            "implementation_tool": artifact_record(union_tool),
            "cells": observed_union_cells,
            "denominator": artifact_record(
                run_dir / "01_manifests" / "paired_test.parquet"
            ),
        }
        if union_application.get("sources") != expected_union_sources:
            raise ValueError("union Test application source binding mismatch")
        expected_union_configuration = {
            "encoder": union_selection["selected_encoder"],
            "seeds": list(FORMAL_SEEDS),
            "pool": "primary_union_top15_no_dedup",
            "label_free_test": True,
            "candidate_test_labels_read": False,
        }
        if union_application.get("configuration") != expected_union_configuration:
            raise ValueError("union Test application configuration differs from selected winner")
        validate_application_signature(union_application, kind="union")
        validate_union_application_outputs(application_path, union_selection)
        union_selection_record = artifact_record(selection_path)
        union_application_record = artifact_record(application_path)
    bridges: dict[str, dict[str, str]] = {}
    bridge_tables: list[pd.DataFrame] = []
    for route in ("g1", "c1"):
        for split in ("train", "validation"):
            key = f"{route}_{split}"
            path = _regular_file(
                run_dir
                / "11_attribution_bridge"
                / f"bridge_{route}_{split}_top5_manifest.json",
                f"{route} {split.title()} attribution bridge",
            )
            bridge = _complete_manifest(path, f"{route} {split.title()} attribution bridge")
            if (
                str(bridge.get("split", "")).lower() != split
                or bridge.get("test_access") not in {False, "NONE"}
            ):
                raise PermissionError(f"{route}/{split} bridge is not a development-only result")
            table_path = _verify_record(
                bridge.get("artifacts", {}).get("table", {}),
                f"{route}/{split} bridge table",
            )
            table = pd.read_csv(table_path)
            if table.empty:
                raise ValueError(f"{route}/{split} bridge table is empty")
            table["route"] = route.upper()
            table["split"] = split
            bridge_tables.append(table)
            bridges[key] = artifact_record(path)
    consolidated_bridge_path = run_dir / "07_validation" / "bridge_train_validation.csv"
    consolidated = pd.concat(bridge_tables, ignore_index=True).sort_values(
        ["route", "split"], kind="mergesort"
    )
    _atomic_csv(consolidated_bridge_path, consolidated)
    bridge_alias_path = run_dir / "11_attribution_bridge" / "bridge_train_validation.csv"
    _atomic_csv(bridge_alias_path, consolidated)
    if sha256_file(bridge_alias_path) != sha256_file(consolidated_bridge_path):
        raise RuntimeError("Train/Validation bridge alias differs from canonical table")
    test_bridge_path = _regular_file(
        run_dir
        / "11_attribution_bridge"
        / "test_bridge_input"
        / "manifest.json",
        "label-free Test bridge input manifest",
    )
    test_bridge, _test_bridge_bundle, test_bridge_paths = (
        validate_label_free_test_bridge_manifest(
            test_bridge_path,
            expected_denominator=artifact_record(sample_path),
            expected_evaluator=artifact_record(evaluator_path),
        )
    )
    return {
        "router": router,
        "router_selection_record": artifact_record(router_selection_path),
        "router_test_record": artifact_record(router_test_path),
        "router_decisions": router_decisions,
        "union": union,
        "union_record": artifact_record(union_path),
        "union_selection": union_selection,
        "union_selection_record": union_selection_record,
        "union_application": union_application,
        "union_application_record": union_application_record,
        "union_predictions": union_predictions,
        "union_decisions": union_decisions,
        "union_feature_manifest_record": (
            None
            if union_application is None
            else artifact_record(union_feature_manifest_path)
        ),
        "union_feature_columns": (
            None if union_application is None else union_feature_columns
        ),
        "bridges": bridges,
        "bridge_train_validation_record": artifact_record(consolidated_bridge_path),
        "bridge_train_validation_alias_record": artifact_record(bridge_alias_path),
        "test_bridge": test_bridge,
        "test_bridge_record": artifact_record(test_bridge_path),
        "test_bridge_paths": test_bridge_paths,
    }


def _canonical_decision(
    samples: pd.DataFrame,
    source: pd.DataFrame,
    *,
    name: str,
    selected_column: str = "selected_candidate_id",
) -> pd.DataFrame:
    _assert_label_free_columns(source, name)
    if "sample_id" not in source or selected_column not in source:
        raise ValueError(f"{name} misses sample/selection columns")
    work = source[["sample_id", selected_column]].rename(columns={selected_column: "selected_candidate_id"}).copy()
    work["sample_id"] = work["sample_id"].astype(str)
    if work["sample_id"].duplicated().any() or len(work) != len(samples):
        raise ValueError(f"{name} does not exactly cover the Test denominator")
    output = samples[["sample_id"]].merge(work, on="sample_id", how="left", validate="one_to_one")
    output["selected_candidate_id"] = output["selected_candidate_id"].fillna("").astype(str)
    return output


def _rankings_and_decisions(
    run_dir: Path,
    samples: pd.DataFrame,
    top5_pools: Mapping[str, pd.DataFrame],
    route_inputs: Mapping[str, Any],
    cross_route: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Path]]]:
    output_dir = run_dir / "08_lock" / "formal_label_free"
    systems: list[dict[str, Any]] = []
    route_paths: dict[str, dict[str, Path]] = {}
    gated_ids: dict[str, pd.Series] = {}
    for route in ROUTES:
        pool = top5_pools[route].copy()
        native_ranking = pool[
            ["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"]
        ].rename(columns={"native_rank": "rank"})
        native_ranking["frozen_native_rank"] = native_ranking["rank"].astype(int)
        native_ranking = native_ranking[
            ["sample_id", "candidate_id", "rank", "candidate_geometry_sha256", "frozen_native_rank"]
        ]
        native_decision = samples[["sample_id"]].merge(
            native_ranking.loc[native_ranking["rank"].eq(1), ["sample_id", "candidate_id"]].rename(
                columns={"candidate_id": "selected_candidate_id"}
            ),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        native_decision["selected_candidate_id"] = native_decision["selected_candidate_id"].fillna("").astype(str)

        score_frame = pd.read_parquet(route_inputs[route]["ranker_scores"])
        _assert_label_free_columns(score_frame, f"{route} Test ranker scores")
        required = {"sample_id", "candidate_id", "ensemble_score"}
        missing = sorted(required.difference(score_frame.columns))
        if missing:
            raise ValueError(f"{route} Test ranker scores miss columns: {missing}")
        score_frame = score_frame[list(required)].copy()
        score_frame["sample_id"] = score_frame["sample_id"].astype(str)
        score_frame["candidate_id"] = score_frame["candidate_id"].astype(str)
        scores = pd.to_numeric(score_frame["ensemble_score"], errors="coerce")
        if not np.isfinite(scores.to_numpy(float)).all() or score_frame.duplicated(
            ["sample_id", "candidate_id"]
        ).any():
            raise ValueError(f"{route} Test ranker scores are invalid")
        ranked = pool[
            ["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"]
        ].merge(score_frame, on=["sample_id", "candidate_id"], validate="one_to_one")
        if len(ranked) != len(pool):
            raise ValueError(f"{route} Test ranker scores do not exactly cover Top-5")
        ranked = ranked.sort_values(
            ["sample_id", "ensemble_score", "native_rank", "candidate_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        ranked["rank"] = ranked.groupby("sample_id", sort=False).cumcount() + 1
        ungated_ranking = ranked[
            ["sample_id", "candidate_id", "rank", "candidate_geometry_sha256", "native_rank"]
        ].rename(columns={"native_rank": "frozen_native_rank"})
        ranker_source = pd.read_parquet(route_inputs[route]["ranker_decisions"])
        ungated_decision = _canonical_decision(samples, ranker_source, name=f"{route} Test ranker decisions")
        expected_ungated = samples["sample_id"].map(
            ungated_ranking.loc[ungated_ranking["rank"].eq(1)].set_index("sample_id")["candidate_id"]
        ).fillna("").astype(str)
        if not ungated_decision["selected_candidate_id"].equals(expected_ungated):
            raise ValueError(f"{route} ranker decisions disagree with deterministic ensemble ranking")

        gate_source = pd.read_parquet(route_inputs[route]["gate_decisions"])
        gated_decision = _canonical_decision(samples, gate_source, name=f"{route} Test gate decisions")
        native_ids = native_decision["selected_candidate_id"]
        ungated_ids = ungated_decision["selected_candidate_id"]
        if not (
            (gated_decision["selected_candidate_id"] == native_ids)
            | (gated_decision["selected_candidate_id"] == ungated_ids)
        ).all():
            raise ValueError(f"{route} gate selected outside native/ungated Top-1")
        geometry = pool.set_index(["sample_id", "candidate_id"])["candidate_geometry_sha256"].to_dict()
        for frame in (native_decision, ungated_decision, gated_decision):
            frame["selected_candidate_geometry_sha256"] = [
                "" if not candidate_id else str(geometry.get((sample_id, candidate_id), ""))
                for sample_id, candidate_id in frame[["sample_id", "selected_candidate_id"]].itertuples(index=False)
            ]
            nonempty = frame["selected_candidate_id"].ne("")
            if frame.loc[nonempty, "selected_candidate_geometry_sha256"].eq("").any():
                raise ValueError(f"{route} decision selects outside frozen Top-5")

        paths = {
            "native_ranking": output_dir / f"{route}_native_ranking.parquet",
            "ungated_ranking": output_dir / f"{route}_ungated_ranking.parquet",
            "native_decisions": output_dir / f"{route}_native_decisions.parquet",
            "ungated_decisions": output_dir / f"{route}_ungated_decisions.parquet",
            "gated_decisions": output_dir / f"{route}_gated_decisions.parquet",
        }
        _atomic_parquet(paths["native_ranking"], native_ranking)
        _atomic_parquet(paths["ungated_ranking"], ungated_ranking)
        _atomic_parquet(paths["native_decisions"], native_decision)
        _atomic_parquet(paths["ungated_decisions"], ungated_decision)
        _atomic_parquet(paths["gated_decisions"], gated_decision)
        route_paths[route] = paths
        gated_ids[route] = gated_decision.set_index("sample_id")["selected_candidate_id"]
        native_name = f"{route}_native"
        ungated_name = f"{route}_ungated_primary"
        gated_name = f"{route}_gated_primary"
        systems.extend(
            [
                {
                    "name": native_name,
                    "kind": "native",
                    "route": route,
                    "decisions_path": str(paths["native_decisions"].resolve()),
                    "ranking_path": str(paths["native_ranking"].resolve()),
                    "rank_column": "rank",
                },
                {
                    "name": ungated_name,
                    "kind": "ungated",
                    "route": route,
                    "native_reference": native_name,
                    "hypothesis_family": "primary_order_only_ungated",
                    "decisions_path": str(paths["ungated_decisions"].resolve()),
                    "ranking_path": str(paths["ungated_ranking"].resolve()),
                    "rank_column": "rank",
                },
                {
                    "name": gated_name,
                    "kind": "gated",
                    "route": route,
                    "native_reference": native_name,
                    "hypothesis_family": "primary_order_only_conservative_gate",
                    "decisions_path": str(paths["gated_decisions"].resolve()),
                    "base_ranking_system": ungated_name,
                },
            ]
        )

    router_source = pd.read_parquet(cross_route["router_decisions"])
    _assert_label_free_columns(router_source, "Test route-router decisions")
    required = {"sample_id", "selected_candidate_id", "selected_route"}
    missing = sorted(required.difference(router_source.columns))
    if missing:
        raise ValueError(f"Test router decisions miss columns: {missing}")
    router = samples[["sample_id"]].merge(
        router_source[list(required)], on="sample_id", how="left", validate="one_to_one"
    )
    if len(router_source) != len(samples) or router["selected_route"].isna().any():
        raise ValueError("Test router decisions do not exactly cover the denominator")
    router["selected_route"] = router["selected_route"].astype(str).str.lower()
    router["selected_candidate_id"] = router["selected_candidate_id"].fillna("").astype(str)
    if not set(router["selected_route"]).issubset(ROUTES):
        raise ValueError("Test router selected an unknown route")
    expected = np.asarray(
        [
            gated_ids[route].loc[sample_id]
            for sample_id, route in router[["sample_id", "selected_route"]].itertuples(index=False)
        ],
        dtype=object,
    )
    if not np.array_equal(router["selected_candidate_id"].to_numpy(object), expected):
        raise ValueError("Test router selection differs from the selected route's gated choice")
    router_path = output_dir / "crog_default_router_decisions.parquet"
    _atomic_parquet(router_path, router)
    route_paths["cross_route"] = {"router_decisions": router_path}
    systems.append(
        {
            "name": "crog_default_router",
            "kind": "router",
            "route": "cross_route",
            "native_reference": "crog_gated_primary",
            "hypothesis_family": "primary_cross_route_crog_default_router",
            "decisions_path": str(router_path.resolve()),
        }
    )
    if cross_route["union"]["decision"] == "UNION_HEADROOM_AVAILABLE":
        prediction_source = pd.read_parquet(cross_route["union_predictions"])
        _assert_label_free_columns(prediction_source, "Test union predictions")
        required = {
            "sample_id",
            "candidate_id",
            "native_rank",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
            "ensemble_score",
        }
        missing = sorted(required.difference(prediction_source.columns))
        if missing:
            raise ValueError(f"Test union predictions miss columns: {missing}")
        union_scores = prediction_source[list(required)].copy()
        for column in ("sample_id", "candidate_id", "source_candidate_id"):
            union_scores[column] = union_scores[column].astype(str)
        union_scores["source_route"] = union_scores["source_route"].astype(str).str.lower()
        union_scores["candidate_geometry_sha256"] = union_scores[
            "candidate_geometry_sha256"
        ].astype(str)
        numeric_score = pd.to_numeric(union_scores["ensemble_score"], errors="coerce")
        union_native_rank = pd.to_numeric(union_scores["native_rank"], errors="coerce")
        if (
            not np.isfinite(numeric_score.to_numpy(float)).all()
            or union_native_rank.isna().any()
            or not np.equal(union_native_rank, np.floor(union_native_rank)).all()
            or union_scores.duplicated(["sample_id", "candidate_id"]).any()
        ):
            raise ValueError("Test union predictions have invalid scores/identities")
        union_scores["ensemble_score"] = numeric_score
        union_scores["native_rank"] = union_native_rank.astype(int)
        expected_parts: list[pd.DataFrame] = []
        for route in ROUTES:
            part = top5_pools[route][
                ["sample_id", "candidate_id", "native_rank", "candidate_geometry_sha256"]
            ].copy()
            part["source_route"] = route
            part["source_candidate_id"] = part["candidate_id"].astype(str)
            part["candidate_id"] = route.upper() + ":" + part["source_candidate_id"]
            part = part.rename(columns={"native_rank": "frozen_native_rank"})
            expected_parts.append(part)
        expected_union = pd.concat(expected_parts, ignore_index=True)
        keys = ["sample_id", "candidate_id"]
        if set(map(tuple, union_scores[keys].to_numpy())) != set(
            map(tuple, expected_union[keys].to_numpy())
        ):
            raise ValueError("Test union predictions do not bind the three frozen Top-5 pools")
        checked = union_scores.merge(
            expected_union,
            on=keys,
            validate="one_to_one",
            suffixes=("", "_locked"),
        )
        for column in ("source_route", "source_candidate_id", "candidate_geometry_sha256"):
            if not checked[column].equals(checked[f"{column}_locked"]):
                raise ValueError(f"Test union prediction {column} binding mismatch")
        union_ranking = checked.sort_values(
            ["sample_id", "ensemble_score", "native_rank", "candidate_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).copy()
        union_ranking["rank"] = union_ranking.groupby("sample_id", sort=False).cumcount() + 1
        union_ranking = union_ranking[
            [
                "sample_id",
                "candidate_id",
                "rank",
                "source_route",
                "source_candidate_id",
                "candidate_geometry_sha256",
                "frozen_native_rank",
            ]
        ]
        raw_decisions = pd.read_parquet(cross_route["union_decisions"])
        _assert_label_free_columns(raw_decisions, "Test union decisions")
        union_decision = samples[["sample_id"]].merge(
            raw_decisions[
                [
                    "sample_id",
                    "selected_candidate_id",
                    "source_route",
                    "source_candidate_id",
                    "candidate_geometry_sha256",
                ]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if len(raw_decisions) != len(samples) or union_decision["selected_candidate_id"].isna().any():
            raise ValueError("Test union decisions do not exactly cover the denominator")
        union_decision["selected_candidate_id"] = union_decision[
            "selected_candidate_id"
        ].astype(str)
        union_decision["source_route"] = union_decision["source_route"].astype(str).str.lower()
        union_decision["source_candidate_id"] = union_decision[
            "source_candidate_id"
        ].astype(str)
        union_decision["candidate_geometry_sha256"] = union_decision[
            "candidate_geometry_sha256"
        ].astype(str)
        top = union_ranking.loc[union_ranking["rank"].eq(1)].set_index("sample_id")
        for column in (
            "candidate_id",
            "source_route",
            "source_candidate_id",
            "candidate_geometry_sha256",
        ):
            decision_column = "selected_candidate_id" if column == "candidate_id" else column
            observed = samples["sample_id"].map(top[column]).astype(str)
            if not union_decision[decision_column].equals(observed):
                raise ValueError(f"Test union decision {decision_column} differs from ranking Top-1")
        ranking_path = output_dir / "top15_union_ranking.parquet"
        decision_path = output_dir / "top15_union_decisions.parquet"
        _atomic_parquet(ranking_path, union_ranking)
        _atomic_parquet(decision_path, union_decision)
        route_paths["union"] = {
            "union_ranking": ranking_path,
            "union_decisions": decision_path,
        }
        systems.append(
            {
                "name": "top15_union_primary",
                "kind": "union",
                "route": "cross_route",
                "native_reference": "crog_gated_primary",
                "hypothesis_family": "secondary_cross_route_union",
                "decisions_path": str(decision_path.resolve()),
                "ranking_path": str(ranking_path.resolve()),
                "rank_column": "rank",
                "selection_manifest": cross_route["union_selection_record"],
                "application_manifest": cross_route["union_application_record"],
            }
        )
    return systems, route_paths


def _primary_declaration(
    selected: Mapping[str, Any],
    route_inputs: Mapping[str, Any],
    cross_route: Mapping[str, Any],
) -> str:
    lines = [
        "# Primary method declaration",
        "",
        "Status: **VALIDATION-LOCKED; TEST OUTCOMES NOT READ**.",
        "",
        "The benchmark is a locked retrospective paired Test benchmark. All candidate",
        "membership and geometry are frozen; every single-route selector is order-only.",
        "",
        "## Single-route primary systems",
        "",
    ]
    for route in ROUTES:
        identity = selected["routes"][route]["identity"]
        gate = route_inputs[route]["gate"]
        lines.extend(
            [
                f"### {route.upper()}",
                "",
                f"- Evidence track: `{PRIMARY_TRACK}`.",
                f"- Ungated ranker: `{identity.get('method_code')}`; encoder `{identity.get('encoder')}`; loss `{identity.get('loss')}`.",
                f"- Seed ensemble: `{list(FORMAL_SEEDS)}` (mean candidate score; deterministic native-rank/candidate-ID tie-break).",
                f"- Feature count: `{len(selected['routes'][route]['feature_columns'])}`; exact schema is locked in `selected_features.json`.",
                f"- Gate decision: `{gate.get('decision')}`; operating point `{gate.get('selection', {}).get('selected_operating_point')}`.",
                "- Primary hypothesis: the gated T2 selector improves J@1 over this route's frozen native Top-1 without changing Top-5 membership or geometry.",
                "",
            ]
        )
    router = cross_route["router"]
    union = cross_route["union"]
    lines.extend(
        [
            "## Cross-route primary system",
            "",
            "- Default route: `CROG`; alternative tie-break: `G1`, then `C1`.",
            f"- Router decision: `{router.get('decision')}`; operating point `{router.get('selection', {}).get('selected_operating_point')}`.",
            "- Primary hypothesis: the locked conservative router improves J@1 over the locked gated CROG output.",
            "",
            "## Union and secondary analyses",
            "",
            f"- Validation Top-15 union decision: `{union.get('decision')}`.",
            "- T1, T3, R0-R7 comparisons, encoder comparisons, feature ablations, the 2x2 attribution bridge, and any eligible union ranker are secondary/exploratory and cannot replace the declared T2 primary systems after Test evaluation.",
            "- The post-lock Test attribution bridge is secondary and cannot alter any primary selector.",
            "",
            "## Test-access declaration",
            "",
            "Candidate-level Test labels have not been parsed or materialised by P11 assembly. Only the byte hash of the predeclared immutable fair-source label Parquet is recorded.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    value = {"schema_version": 1, "status": "LOCKED", **dict(payload)}
    atomic_json(path, value)


def _log_hash_only_access(run_dir: Path, label_path: Path, digest: str) -> None:
    log_path = run_dir / "09_formal_test" / "test_access.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    marker = '"event": "prelock_candidate_test_label_hash_only"'
    signature_marker = f'"candidate_labels_sha256": "{digest}"'
    previous = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if marker in previous and signature_marker in previous:
        return
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "prelock_candidate_test_label_hash_only",
                    "path": str(label_path),
                    "candidate_labels_sha256": digest,
                    "candidate_labels_opened_as_table": False,
                    "allowed_access": "opaque_byte_hash_only",
                    "timestamp_utc": _now(),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _log_bridge_ground_truth_hash_only(
    run_dir: Path, ground_truth_path: Path, digest: str
) -> None:
    log_path = run_dir / "09_formal_test" / "test_access.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    marker = '"event": "prelock_historical_test_ground_truth_hash_only"'
    signature_marker = f'"ground_truth_sha256": "{digest}"'
    previous = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if marker in previous and signature_marker in previous:
        return
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "event": "prelock_historical_test_ground_truth_hash_only",
                    "path": str(ground_truth_path.resolve()),
                    "ground_truth_sha256": digest,
                    "ground_truth_rows_opened": False,
                    "allowed_access": "opaque_byte_hash_only",
                    "timestamp_utc": _now(),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _semantic_validate(run_dir: Path, plan_path: Path) -> None:
    # Reuse the exact validators consumed by create_formal_test_lock.  None of
    # these functions opens the candidate-label table before formal execution.
    from tools.unified_reranking.create_formal_test_lock import (
        REQUIRED_LOCK_FILES,
        _validate_p11_snapshot_schemas,
        _validate_selection_contracts,
        validate_formal_evaluation_plan,
    )

    locked = {
        name: run_dir / "08_lock" / filename
        for name, filename in REQUIRED_LOCK_FILES.items()
    }
    plan, _ = validate_formal_evaluation_plan(plan_path)
    _validate_p11_snapshot_schemas(locked, plan)
    _validate_selection_contracts(
        plan,
        locked["candidate_manifests"],
        locked["evaluator_sha256"],
    )
    code_manifest = _read_json(run_dir / "08_lock" / "code_manifest.json")
    files = [Path(record["path"]) for record in code_manifest.get("files", [])]
    recomputed = code_bundle(files)
    expected = (run_dir / "08_lock" / "code_sha256.txt").read_text(encoding="utf-8").strip()
    if recomputed["bundle_sha256"] != expected or code_manifest.get("bundle_sha256") != expected:
        raise ValueError("code bundle digest is not bound to the declared real files")


def assemble_prelock_bundle(
    run_dir: str | Path,
    *,
    candidate_test_labels_path: str | Path | None = None,
    evaluator_path: str | Path | None = None,
    code_roots: Sequence[str | Path] | None = None,
    normalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and validate P11 without creating the formal lock.

    The run manifest is advanced only after every generated snapshot, ranking,
    decision and source hash passes the same semantic checks used by the lock
    command.
    """

    root = Path(run_dir).expanduser().resolve()
    manifest_path = _regular_file(root / "manifest.json", "run manifest")
    if (root / "08_lock" / "FORMAL_TEST_LOCK.json").exists() or (
        root / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    ).exists():
        raise PermissionError("P11 assembly cannot run after formal lock/execution exists")
    manifest = _read_json(manifest_path)
    if int(manifest.get("formal_test_execution_count", -1)) != 0:
        raise PermissionError("formal Test execution count must be zero during P11 assembly")

    labels_path = _regular_file(
        candidate_test_labels_path or discover_candidate_test_labels(root),
        "predeclared candidate-level Test label source",
    )
    evaluator = _regular_file(evaluator_path or discover_evaluator(root), "frozen evaluator")
    folds = _regular_file(root / "04_splits" / "fold_assignments.parquet", "fold assignments")
    code_paths = collect_code_files(
        code_roots
        or (
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parents[2] / "tools" / "unified_reranking",
        )
    )
    code = code_bundle(code_paths)
    sample_path, samples = _sample_manifest(root)
    verify_candidate_contract_hashes(root)
    all_pools, top5_pools, candidate_records, candidate_paths = _load_candidate_pools(root, samples)
    selected = _load_development_selections(root)
    route_inputs = _load_route_prerequisites(root, selected)
    cross_route = _load_cross_route_prerequisites(root, sample_path, evaluator)

    label_hash = sha256_file(labels_path)  # Opaque byte access only.
    evaluator_hash = sha256_file(evaluator)
    fold_hash = sha256_file(folds)
    label_normalization = dict(normalization or LABEL_NORMALIZATION)
    if not str(label_normalization.get("route_column", "")) or not str(
        label_normalization.get("variant_column", "")
    ) or list(label_normalization.get("include_variants", [])) != ["crog_native", "g1", "c1"]:
        raise ValueError("candidate label normalization must lock native CROG/G1/C1 variants")

    source_records: dict[str, dict[str, str]] = {
        "candidate_test_labels": {"path": str(labels_path), "sha256": label_hash},
        "evaluator": artifact_record(evaluator),
        "fold_assignments": artifact_record(folds),
        "paired_test": artifact_record(sample_path),
        "primary_selection": artifact_record(selected["path"]),
        "screen_latest_execution": selected["screen_execution_record"],
        "screen_selection_manifest": selected["screen_selection_record"],
        "screen_finalists": selected["screen_finalists_record"],
        "selected_latest_execution": selected["selected_execution_record"],
        "encoder_loss_selection": selected["encoder_selection_record"],
        "encoder_latest_execution": selected["encoder_execution_record"],
        "feature_ablation_manifest": selected["feature_ablation_record"],
        "feature_extraction_benchmark": selected["feature_benchmark_record"],
        "union_headroom": cross_route["union_record"],
        "route_router_selection": cross_route["router_selection_record"],
        "route_router_test": cross_route["router_test_record"],
    }
    for name, path in candidate_paths.items():
        source_records[f"candidate_{name}"] = artifact_record(path)
    for route in ROUTES:
        source_records[f"{route}_selected_validation_ensemble"] = selected["routes"][
            route
        ]["validation_manifest"]
        source_records[f"{route}_selected_oof_ensemble"] = selected["routes"][route][
            "oof_manifest"
        ]
        for name in (
            "calibration_record",
            "calibration_application_record",
            "feature_manifest_record",
            "feature_artifact_record",
            "ranker_manifest_record",
            "gate_selection_record",
            "gate_test_record",
        ):
            source_records[f"{route}_{name}"] = route_inputs[route][name]
        for seed, record in selected["routes"][route]["cell_manifests"].items():
            source_records[f"{route}_validation_cell_seed_{seed}"] = record
    for bridge_key, record in cross_route["bridges"].items():
        source_records[f"bridge_{bridge_key}"] = record
    source_records["bridge_train_validation"] = cross_route[
        "bridge_train_validation_record"
    ]
    source_records["bridge_train_validation_alias"] = cross_route[
        "bridge_train_validation_alias_record"
    ]
    source_records["test_bridge_manifest"] = cross_route["test_bridge_record"]
    for name, record in sorted(cross_route["test_bridge"]["sources"].items()):
        source_records[f"test_bridge_source_{name}"] = dict(record)
    source_records["test_bridge_candidate_bundle"] = dict(
        cross_route["test_bridge"]["artifacts"]["candidate_bundle"]
    )
    if cross_route["union"]["decision"] == "UNION_HEADROOM_AVAILABLE":
        source_records["union_ranker_selection"] = cross_route["union_selection_record"]
        source_records["union_ranker_test"] = cross_route["union_application_record"]
        source_records["union_test_features"] = cross_route[
            "union_feature_manifest_record"
        ]
    source_signature = canonical_sha256(
        {
            "sources": source_records,
            "code_bundle_sha256": code["bundle_sha256"],
            "normalization": label_normalization,
        }
    )
    marker_path = root / "08_lock" / "prelock_assembly_manifest.json"
    if marker_path.exists():
        marker = _complete_manifest(marker_path, "P11 pre-lock assembly")
        if marker.get("signature_sha256") != source_signature:
            raise RuntimeError("immutable P11 assembly exists with a different source signature")
        for name, record in dict(marker.get("artifacts", {})).items():
            _verify_record(record, f"P11 output {name}")
        plan_path = _verify_record(marker["artifacts"]["formal_evaluation_plan"], "formal evaluation plan")
        _semantic_validate(root, plan_path)
        if manifest.get("test_label_state") != "VALIDATION_SELECTION_COMPLETE":
            manifest["status"] = "VALIDATION_SELECTION_COMPLETE"
            manifest["test_label_state"] = "VALIDATION_SELECTION_COMPLETE"
            manifest["prelock_assembly_sha256"] = sha256_file(marker_path)
            manifest["validation_selection_completed_at_utc"] = _now()
            atomic_json(manifest_path, manifest)
        return marker

    systems, decision_paths = _rankings_and_decisions(
        root, samples, top5_pools, route_inputs, cross_route
    )
    lock_dir = root / "08_lock"
    selected_methods = {
        route: {
            "native_system": f"{route}_native",
            "ungated_system": f"{route}_ungated_primary",
            "gated_system": f"{route}_gated_primary",
            "primary_evidence_track": PRIMARY_TRACK,
            "method_code": selected["routes"][route]["identity"].get("method_code"),
            "encoder": selected["routes"][route]["identity"].get("encoder"),
            "loss": selected["routes"][route]["identity"].get("loss"),
            "seeds": list(FORMAL_SEEDS),
            "validation_ensemble": selected["routes"][route]["validation_manifest"],
            "label_free_test_application": route_inputs[route][
                "ranker_manifest_record"
            ],
        }
        for route in ROUTES
    }
    methods_payload: dict[str, Any] = {"routes": selected_methods}
    features_payload: dict[str, Any] = {
        "routes": {
            route: {
                "evidence_track": PRIMARY_TRACK,
                "feature_columns": selected["routes"][route]["feature_columns"],
                "feature_schema_sha256": canonical_sha256(selected["routes"][route]["feature_columns"]),
                "test_feature_manifest": route_inputs[route]["feature_manifest_record"],
            }
            for route in ROUTES
        }
    }
    hyperparameters_payload: dict[str, Any] = {
        "routes": {
            route: {
                "seed_configurations": {
                    str(seed): selected["routes"][route]["cells"][seed]["configuration"]
                    for seed in FORMAL_SEEDS
                },
                "cell_manifests": selected["routes"][route]["cell_manifests"],
            }
            for route in ROUTES
        }
    }
    if cross_route["union"]["decision"] == "UNION_HEADROOM_AVAILABLE":
        union_selection = cross_route["union_selection"]
        methods_payload["union"] = {
            "system_name": "top15_union_primary",
            "pool": "primary_union_top15_no_dedup",
            "encoder": union_selection["selected_encoder"],
            "seeds": list(FORMAL_SEEDS),
            "selection_manifest": cross_route["union_selection_record"],
            "label_free_test_application": cross_route["union_application_record"],
        }
        features_payload["union"] = {
            "pool": "primary_union_top15_no_dedup",
            "feature_columns": cross_route["union_feature_columns"],
            "feature_schema_sha256": canonical_sha256(
                cross_route["union_feature_columns"]
            ),
            "test_feature_manifest": cross_route["union_feature_manifest_record"],
        }
        hyperparameters_payload["union"] = {
            "fixed_budget": union_selection.get("fixed_budget"),
            "selection_order": union_selection.get("selection_order"),
            "selected_encoder": union_selection["selected_encoder"],
            "selection_manifest": cross_route["union_selection_record"],
        }
    _write_snapshot(lock_dir / "selected_methods.json", methods_payload)
    _write_snapshot(
        lock_dir / "selected_features.json",
        features_payload,
    )
    _write_snapshot(
        lock_dir / "selected_hyperparameters.json",
        hyperparameters_payload,
    )
    _write_snapshot(
        lock_dir / "calibration_manifest.json",
        {
            "routes": {
                route: {
                    "selected_method": route_inputs[route]["calibration"].get("selected_method"),
                    "manifest_path": route_inputs[route]["calibration_record"]["path"],
                    "manifest_sha256": route_inputs[route]["calibration_record"]["sha256"],
                    "label_free_test_application": route_inputs[route]["calibration_application_record"],
                }
                for route in ROUTES
            }
        },
    )
    _write_snapshot(
        lock_dir / "gate_thresholds.json",
        {
            "routes": {
                route: {
                    "decision": route_inputs[route]["gate"]["decision"],
                    "operating_point": route_inputs[route]["gate"].get("selection", {}).get("selected_operating_point"),
                    "selection_manifest": route_inputs[route]["gate_selection_record"],
                    "label_free_test_application": route_inputs[route]["gate_test_record"],
                }
                for route in ROUTES
            }
        },
    )
    _write_snapshot(
        lock_dir / "router_thresholds.json",
        {
            "default_route": "CROG",
            "tie_break": ["G1", "C1"],
            "system_name": "crog_default_router",
            "decision": cross_route["router"]["decision"],
            "operating_point": cross_route["router"].get("selection", {}).get("selected_operating_point"),
            "selection_manifest": cross_route["router_selection_record"],
            "label_free_test_application": cross_route["router_test_record"],
        },
    )
    _write_snapshot(
        lock_dir / "candidate_manifests.json",
        {
            "sample_manifest": artifact_record(sample_path),
            "candidate_pools": candidate_records,
            "candidate_contract": "Test All pools label coverage; rankings restricted to immutable native Top-5 prefixes",
        },
    )
    atomic_text(lock_dir / "fold_assignments_sha256.txt", fold_hash + "\n")
    atomic_text(lock_dir / "evaluator_sha256.txt", evaluator_hash + "\n")
    atomic_json(lock_dir / "code_manifest.json", code)
    atomic_text(lock_dir / "code_sha256.txt", code["bundle_sha256"] + "\n")
    atomic_text(
        lock_dir / "PRIMARY_METHOD_DECLARATION.md",
        _primary_declaration(selected, route_inputs, cross_route),
    )
    label_manifest_path = lock_dir / "candidate_test_label_manifest.json"
    atomic_json(
        label_manifest_path,
        {
            "schema_version": 1,
            "split": "test",
            "provenance": "fair-source canonical candidates, predeclared and hash-only before formal claim",
            "candidate_labels_path": str(labels_path),
            "candidate_labels_sha256": label_hash,
            "candidate_labels_opened_as_table_prelock": False,
            "normalization": label_normalization,
            "evaluator_path": str(evaluator),
            "evaluator_sha256": evaluator_hash,
            "candidate_pools": {
                route: {"path": record["path"], "sha256": record["sha256"]}
                for route, record in candidate_records.items()
            },
        },
    )
    plan_path = lock_dir / "formal_evaluation_plan.json"
    atomic_json(
        plan_path,
        {
            "schema_version": 1,
            "benchmark_description": "locked retrospective paired test benchmark",
            "candidate_test_labels_read": False,
            "sample_manifest": str(sample_path),
            "candidate_label_manifest": str(label_manifest_path),
            "systems": systems,
            "union_contract": {
                "decision": cross_route["union"]["decision"],
                "headroom_manifest": cross_route["union_record"],
                "selection_manifest": cross_route["union_selection_record"],
                "label_free_test_application": cross_route["union_application_record"],
                "identity_mapping": {
                    "formal_candidate_id": "<SOURCE_ROUTE>:<source_candidate_id>",
                    "postclaim_label_join_keys": [
                        "source_route",
                        "sample_id",
                        "source_candidate_id",
                    ],
                    "preclaim_candidate_label_rows_read": False,
                },
            },
            "test_bridge_contract": {
                "manifest": cross_route["test_bridge_record"],
                "candidate_bundle": cross_route["test_bridge"]["artifacts"][
                    "candidate_bundle"
                ],
                "historical_ground_truth": cross_route["test_bridge"]["sources"][
                    "historical_ground_truth"
                ],
                "evaluator": cross_route["test_bridge"]["sources"]["evaluator"],
                "denominator": cross_route["test_bridge"]["sources"]["denominator"],
                "analysis_role": "SECONDARY_POSTLOCK_NO_SELECTION",
                "historical_test_ground_truth_rows_read_preclaim": False,
            },
            "bound_provenance": {
                "fold_assignments": artifact_record(folds),
                "evaluator": artifact_record(evaluator),
                "code_manifest": artifact_record(lock_dir / "code_manifest.json"),
                "code_bundle_sha256": code["bundle_sha256"],
                "primary_selection": artifact_record(selected["path"]),
                "screen_latest_execution": selected["screen_execution_record"],
                "screen_selection_manifest": selected["screen_selection_record"],
                "screen_finalists": selected["screen_finalists_record"],
                "selected_latest_execution": selected[
                    "selected_execution_record"
                ],
                "encoder_loss_selection": selected["encoder_selection_record"],
                "encoder_latest_execution": selected["encoder_execution_record"],
                "feature_ablation_manifest": selected["feature_ablation_record"],
                "feature_extraction_benchmark": selected[
                    "feature_benchmark_record"
                ],
                "union_headroom": cross_route["union_record"],
                "attribution_bridges": cross_route["bridges"],
                "test_bridge_manifest": cross_route["test_bridge_record"],
            },
        },
    )
    _log_hash_only_access(root, labels_path, label_hash)
    _log_bridge_ground_truth_hash_only(
        root,
        Path(cross_route["test_bridge"]["sources"]["historical_ground_truth"]["path"]),
        str(cross_route["test_bridge"]["sources"]["historical_ground_truth"]["sha256"]),
    )
    _semantic_validate(root, plan_path)

    artifact_paths = {
        "selected_methods": lock_dir / "selected_methods.json",
        "selected_features": lock_dir / "selected_features.json",
        "selected_hyperparameters": lock_dir / "selected_hyperparameters.json",
        "calibration_manifest": lock_dir / "calibration_manifest.json",
        "gate_thresholds": lock_dir / "gate_thresholds.json",
        "router_thresholds": lock_dir / "router_thresholds.json",
        "candidate_manifests": lock_dir / "candidate_manifests.json",
        "fold_assignments_sha256": lock_dir / "fold_assignments_sha256.txt",
        "evaluator_sha256": lock_dir / "evaluator_sha256.txt",
        "code_sha256": lock_dir / "code_sha256.txt",
        "code_manifest": lock_dir / "code_manifest.json",
        "primary_method_declaration": lock_dir / "PRIMARY_METHOD_DECLARATION.md",
        "candidate_test_label_manifest": label_manifest_path,
        "formal_evaluation_plan": plan_path,
        "bridge_train_validation": Path(
            cross_route["bridge_train_validation_record"]["path"]
        ),
        "bridge_train_validation_alias": Path(
            cross_route["bridge_train_validation_alias_record"]["path"]
        ),
        "test_bridge_manifest": Path(cross_route["test_bridge_record"]["path"]),
        "test_bridge_candidate_bundle": Path(
            cross_route["test_bridge"]["artifacts"]["candidate_bundle"]["path"]
        ),
    }
    for route, paths in decision_paths.items():
        for name, path in paths.items():
            artifact_paths[f"{route}_{name}"] = path
    output_records = {name: artifact_record(path) for name, path in sorted(artifact_paths.items())}
    # Recheck every source immediately before the state transition. This catches
    # concurrent mutation between discovery and final validation.
    for name, record in source_records.items():
        _verify_record(record, f"P11 source {name}")
    marker = {
        "schema_version": 1,
        "status": "COMPLETE",
        "stage": "P11_PRELOCK",
        "signature_sha256": source_signature,
        "candidate_test_labels_read": False,
        "candidate_test_label_access": "OPAQUE_BYTE_HASH_ONLY",
        "formal_lock_created": False,
        "formal_test_executed": False,
        "sources": source_records,
        "code_bundle_sha256": code["bundle_sha256"],
        "artifacts": output_records,
        "semantic_validation": "PASS",
    }
    atomic_json(marker_path, marker)
    manifest["status"] = "VALIDATION_SELECTION_COMPLETE"
    manifest["test_label_state"] = "VALIDATION_SELECTION_COMPLETE"
    manifest["prelock_assembly_sha256"] = sha256_file(marker_path)
    manifest["validation_selection_completed_at_utc"] = _now()
    atomic_json(manifest_path, manifest)
    return marker


__all__ = [
    "FORMAL_SEEDS",
    "LABEL_NORMALIZATION",
    "PRIMARY_TRACK",
    "ROUTES",
    "artifact_record",
    "assemble_prelock_bundle",
    "code_bundle",
    "collect_code_files",
    "discover_candidate_test_labels",
    "discover_evaluator",
]
