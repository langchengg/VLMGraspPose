"""Post-hoc 4-DoF evaluator, duplicate, and Top-K sensitivity analyses.

The module is intentionally read-only with respect to source runs.  Every
write is rooted below the caller supplied robustness-suite ``run_dir`` and is
performed atomically.  The formal 0.25/30 evaluator setting is a regression
anchor; the remaining threshold cells are descriptive sensitivity analyses.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


THRESHOLD_GRID: tuple[tuple[float, int], ...] = tuple(
    (iou, angle) for iou in (0.20, 0.25, 0.30) for angle in (20, 30, 40)
)
FORMAL_THRESHOLD = (0.25, 30)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260815
RANKER_SEEDS = (20260815, 20260816, 20260817)
ROUTES = ("crog", "g1", "c1", "d1")
FORMAL_METRICS_SHA256 = {
    "primary": "0cb814082fc8754f8217355a48bfdb2719412118d6e179655587374caf6ba625",
    "d1": "c49a2b73ddf5baebdd2c43f01d23c6c91c3ca38c94caa20f95f285dc1a2f6d16",
}


@dataclass(frozen=True)
class RouteSource:
    route: str
    retrospective: bool
    candidates_top5: Path
    candidates_all: Path
    native_decisions: Path | None
    raw_decisions: Path
    gated_decisions: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    serialised = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _analysis_completion_valid(
    repo_root: Path,
    run_dir: Path,
    output: Path,
    filename: str,
    *,
    statuses: set[str],
) -> dict[str, Any] | None:
    path = output / filename
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
        signature = payload.pop("completion_signature_sha256")
        if signature != _sha256_json(payload):
            return None
        if payload.get("status") not in statuses:
            return None
        if payload.get("analysis_code_sha256") != _sha256_file(Path(__file__)):
            return None
        for relative, expected in payload.get("output_hashes", {}).items():
            artifact = output / relative
            if not artifact.is_file() or _sha256_file(artifact) != expected:
                return None
        for relative, expected in payload.get("source_hashes", {}).items():
            artifact = repo_root / relative
            if not artifact.is_file() or _sha256_file(artifact) != expected:
                return None
        for relative, expected in payload.get("run_local_hashes", {}).items():
            artifact = run_dir / relative
            if not artifact.is_file() or _sha256_file(artifact) != expected:
                return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    payload["completion_signature_sha256"] = signature
    return payload


def _write_analysis_completion(
    repo_root: Path,
    run_dir: Path,
    output: Path,
    filename: str,
    *,
    status: str,
    output_paths: Sequence[Path],
    source_paths: Sequence[Path] = (),
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "analysis_code_sha256": _sha256_file(Path(__file__)),
        "output_hashes": {
            str(path.relative_to(output)): _sha256_file(path)
            for path in sorted(output_paths)
        },
        "source_hashes": {
            str(path.resolve().relative_to(repo_root)): _sha256_file(path)
            for path in sorted(source_paths)
        },
        "run_local_hashes": {
            str(path.relative_to(run_dir)): _sha256_file(path)
            for path in (
                run_dir / "PRE_REGISTRATION.md",
                run_dir / "source_artifact_inventory.json",
                run_dir / "SOURCE_AUDIT.md",
            )
            if path.is_file()
        },
        **(metadata or {}),
    }
    payload["completion_signature_sha256"] = _sha256_json(payload)
    _atomic_json(output / filename, payload)
    return payload


def _assert_isolated_run_dir(repo_root: Path, run_dir: Path) -> None:
    """Reject writes outside a dedicated robustness-suite child directory."""

    robustness_root = (repo_root / "artifacts/robustness_suite").resolve()
    resolved_run = run_dir.resolve()
    if resolved_run == robustness_root or robustness_root not in resolved_run.parents:
        raise ValueError(
            "run_dir must be an isolated child of <repo>/artifacts/robustness_suite"
        )


def _temporary(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    return Path(name)


def _atomic_json(destination: Path, value: Any) -> None:
    temporary = _temporary(destination)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(destination: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary(destination)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(destination: Path, frame: pd.DataFrame) -> None:
    temporary = _temporary(destination)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required frozen source artifacts are absent: {missing}")


def _route_sources(repo_root: Path) -> dict[str, RouteSource]:
    unified = repo_root / "runs/fair_unified_reranking_20260809_103012"
    d1 = repo_root / "runs/fair_d1_reranking_extension_20260811T145515Z"
    result: dict[str, RouteSource] = {}
    for route in ("crog", "g1", "c1"):
        locked = unified / "08_lock/formal_label_free"
        result[route] = RouteSource(
            route=route,
            retrospective=False,
            candidates_top5=unified / f"02_candidates/{route}_test_top5.parquet",
            candidates_all=unified / f"02_candidates/{route}_test_all.parquet",
            native_decisions=locked / f"{route}_native_decisions.parquet",
            raw_decisions=locked / f"{route}_ungated_decisions.parquet",
            gated_decisions=locked / f"{route}_gated_decisions.parquet",
        )
    result["d1"] = RouteSource(
        route="d1",
        retrospective=True,
        candidates_top5=d1 / "02_candidates/d1_test_top5.parquet",
        candidates_all=d1 / "02_candidates/d1_test_all.parquet",
        native_decisions=None,
        raw_decisions=d1
        / "08_lock/label_free_test_rankers/d1/per_sample_decisions.parquet",
        gated_decisions=d1
        / "08_lock/label_free_test_gates/d1/gate_test_decisions.parquet",
    )
    return result


def _denominator(repo_root: Path) -> pd.DataFrame:
    path = (
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet"
    )
    frame = pd.read_parquet(path, columns=["sample_id", "scene_id", "frame_id"])
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError("formal 4-DoF denominator is empty or has duplicate sample IDs")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["sequence_id"] = frame["scene_id"].map(sequence_id_from_scene)
    if frame["sequence_id"].eq("").any():
        raise ValueError("sequence clustering identifiers are missing")
    return frame


def sequence_id_from_scene(scene_id: Any) -> str:
    """Return the OCID sequence path, excluding the comma-separated image name."""

    value = str(scene_id).strip()
    return value.rsplit(",", 1)[0].strip() if "," in value else value


def load_frozen_canonical_evaluator(repo_root: Path) -> tuple[ModuleType, Path, str]:
    """Load the byte-frozen evaluator after checking its locked SHA-256."""

    run = repo_root / "runs/fair_unified_reranking_20260809_103012"
    source = run / "configs/canonical_evaluator.py"
    expected = (run / "08_lock/evaluator_sha256.txt").read_text(encoding="utf-8").strip()
    if " " in expected:
        expected = expected.split()[0]
    observed = _sha256_file(source)
    if observed != expected:
        raise RuntimeError(
            f"frozen evaluator hash mismatch: observed={observed} expected={expected}"
        )
    module_name = f"_robustness_frozen_evaluator_{observed[:16]}"
    module = sys.modules.get(module_name)
    if module is None:
        specification = importlib.util.spec_from_file_location(module_name, source)
        if specification is None or specification.loader is None:
            raise ImportError(f"cannot load frozen evaluator: {source}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        except ModuleNotFoundError as error:
            # The required regression command deliberately uses the 6-DoF
            # environment, which does not install Shapely.  Angle-contract
            # checks do not require polygon geometry, so permit the *same
            # byte-frozen module* to load with a fail-closed Polygon sentinel.
            # Any attempt to evaluate IoU still raises instead of silently
            # switching geometry implementations.  Formal threshold analysis
            # runs in the 4-DoF environment and therefore uses real Shapely.
            if error.name not in {"shapely", "shapely.geometry"}:
                sys.modules.pop(module_name, None)
                raise

            class _MissingShapelyPolygon:
                def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                    raise ModuleNotFoundError(
                        "Shapely is required for frozen-evaluator polygon geometry"
                    )

            shapely_stub = ModuleType("shapely")
            geometry_stub = ModuleType("shapely.geometry")
            geometry_stub.Polygon = _MissingShapelyPolygon  # type: ignore[attr-defined]
            shapely_stub.geometry = geometry_stub  # type: ignore[attr-defined]
            sys.modules["shapely"] = shapely_stub
            sys.modules["shapely.geometry"] = geometry_stub
            try:
                specification.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
            finally:
                sys.modules.pop("shapely.geometry", None)
                sys.modules.pop("shapely", None)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return module, source, observed


def threshold_success(
    pairwise_iou: Sequence[float],
    pairwise_angle_error_deg: Sequence[float],
    *,
    iou_threshold: float,
    angle_threshold_deg: float,
) -> bool:
    """Apply the same-GT rule with ``IoU > t`` and ``angle <= a``."""

    iou = np.asarray(pairwise_iou, dtype=float)
    angle = np.asarray(pairwise_angle_error_deg, dtype=float)
    if iou.ndim != 1 or angle.shape != iou.shape:
        raise ValueError("same-GT IoU and angle arrays must be equal 1-D vectors")
    if not np.isfinite(iou).all() or not np.isfinite(angle).all():
        raise ValueError("same-GT pairwise metrics must be finite")
    return bool(np.any((iou > float(iou_threshold)) & (angle <= float(angle_threshold_deg))))


def _candidate_grasp(module: ModuleType, row: Any) -> Any:
    return module.CanonicalGrasp(
        cx_px=float(row.cx_px),
        cy_px=float(row.cy_px),
        theta_deg=float(row.theta_deg),
        jaw_width_px=float(row.width_px),
        rectangle_height_px=float(row.height_px),
        native_score=float(row.native_score),
        native_rank=int(row.native_rank),
        source_method=str(row.route),
        sample_id=str(row.sample_id),
    )


def _normalise_gt(module: ModuleType, rectangles: Any) -> tuple[Any, ...]:
    return tuple(
        module.gt_from_corners(
            np.stack([np.asarray(point, dtype=np.float64) for point in rectangle])
        )
        for rectangle in rectangles
    )


def _raster_iou_from_pixels(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 and second.size == 0:
        return 0.0
    intersection = np.intersect1d(first, second, assume_unique=True).size
    union = int(first.size + second.size - intersection)
    return 0.0 if union <= 0 else float(intersection / union)


def _build_pairwise_cache(
    *,
    candidates_path: Path,
    labels_path: Path,
    evaluator: ModuleType,
    evaluator_sha256: str,
    destination: Path,
    resume: bool,
) -> pd.DataFrame:
    metadata_path = destination.with_suffix(".manifest.json")
    signature = {
        "candidates_path": str(candidates_path.resolve()),
        "candidates_sha256": _sha256_file(candidates_path),
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": _sha256_file(labels_path),
        "evaluator_sha256": evaluator_sha256,
        "contract": "same_gt_raster_iou_and_modulo_pi_angle_v1",
    }
    if resume and destination.is_file() and metadata_path.is_file():
        previous = _read_json(metadata_path)
        if previous.get("signature") == signature and previous.get("status") == "COMPLETE":
            return pd.read_parquet(destination)

    candidates = pd.read_parquet(
        candidates_path,
        columns=[
            "sample_id",
            "candidate_id",
            "route",
            "native_rank",
            "native_score",
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
        ],
    )
    if candidates.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError(f"candidate identities are not unique: {candidates_path}")
    labels = pd.read_parquet(labels_path, columns=["sample_id", "gt_grasp_rectangles"])
    if labels["sample_id"].duplicated().any():
        raise ValueError("formal test labels have duplicate sample IDs")
    gt_by_sample = {
        str(row.sample_id): _normalise_gt(evaluator, row.gt_grasp_rectangles)
        for row in labels.itertuples(index=False)
    }
    missing = sorted(set(candidates["sample_id"].astype(str)).difference(gt_by_sample))
    if missing:
        raise ValueError(f"candidate samples lack formal labels: {missing[:5]}")

    rows: list[dict[str, Any]] = []
    gt_pixel_cache: dict[str, tuple[np.ndarray, ...]] = {}
    for row in candidates.itertuples(index=False):
        sample_id = str(row.sample_id)
        ground_truth = gt_by_sample[sample_id]
        gt_pixels = gt_pixel_cache.get(sample_id)
        if gt_pixels is None:
            gt_pixels = tuple(evaluator._pixels(grasp) for grasp in ground_truth)
            gt_pixel_cache[sample_id] = gt_pixels
        prediction = _candidate_grasp(evaluator, row)
        prediction_pixels = evaluator._pixels(prediction)
        for gt_index, (gt, pixels) in enumerate(zip(ground_truth, gt_pixels, strict=True)):
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": str(row.candidate_id),
                    "native_rank": int(row.native_rank),
                    "gt_index": int(gt_index),
                    "iou": _raster_iou_from_pixels(prediction_pixels, pixels),
                    "angle_error_deg": float(
                        evaluator.periodic_angle_error_deg(
                            prediction.theta_deg, gt.theta_deg
                        )
                    ),
                }
            )
    result = pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "gt_index",
            "iou",
            "angle_error_deg",
        ],
    )
    _atomic_parquet(destination, result)
    _atomic_json(
        metadata_path,
        {
            "status": "COMPLETE",
            "signature": signature,
            "candidate_count": int(len(candidates)),
            "pair_count": int(len(result)),
            "candidate_ids_unique": True,
        },
    )
    return result


def _success_keys(
    pairwise: pd.DataFrame, iou_threshold: float, angle_threshold: int
) -> set[tuple[str, str]]:
    if pairwise.empty:
        return set()
    passed = pairwise.loc[
        pairwise["iou"].gt(float(iou_threshold))
        & pairwise["angle_error_deg"].le(float(angle_threshold)),
        ["sample_id", "candidate_id"],
    ]
    return set(map(tuple, passed.astype(str).drop_duplicates().itertuples(index=False, name=None)))


def _normalise_decisions(path: Path, *, native_from_candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    if native_from_candidates is not None:
        selected = native_from_candidates.loc[
            native_from_candidates["native_rank"].eq(1), ["sample_id", "candidate_id"]
        ].rename(columns={"candidate_id": "selected_candidate_id"})
        return selected.astype(str)
    frame = pd.read_parquet(path)
    required = {"sample_id", "selected_candidate_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"decision artifact lacks fields {missing}: {path}")
    result = frame[["sample_id", "selected_candidate_id"]].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    # Preserve empty/no-output decisions as null; converting them to the string
    # "nan" would turn provenance absence into a candidate identity.
    result["selected_candidate_id"] = result["selected_candidate_id"].map(
        lambda value: None if pd.isna(value) or str(value) == "" else str(value)
    )
    if result["sample_id"].duplicated().any():
        raise ValueError(f"decision artifact has duplicate samples: {path}")
    return result


def _outcomes_for_decisions(
    denominator: pd.DataFrame,
    decisions: pd.DataFrame,
    successes: set[tuple[str, str]],
) -> np.ndarray:
    joined = denominator[["sample_id"]].merge(
        decisions, on="sample_id", how="left", validate="one_to_one"
    )
    return np.asarray(
        [
            bool(candidate is not None and (str(sample), str(candidate)) in successes)
            for sample, candidate in joined[["sample_id", "selected_candidate_id"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=bool,
    )


def _oracle_outcomes(
    denominator: pd.DataFrame,
    candidates: pd.DataFrame,
    successes: set[tuple[str, str]],
) -> np.ndarray:
    success_samples = {
        str(sample_id)
        for sample_id, candidate_id in candidates[["sample_id", "candidate_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
        if (str(sample_id), str(candidate_id)) in successes
    }
    return denominator["sample_id"].isin(success_samples).to_numpy(bool)


def _mcnemar(reference: np.ndarray, challenger: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=bool)
    challenger = np.asarray(challenger, dtype=bool)
    if reference.ndim != 1 or reference.shape != challenger.shape or not len(reference):
        raise ValueError("McNemar inputs must be non-empty paired vectors")
    recovered = int((~reference & challenger).sum())
    harmful = int((reference & ~challenger).sum())
    discordant = recovered + harmful
    pvalue = 1.0 if not discordant else float(binomtest(recovered, discordant, 0.5).pvalue)
    return {
        "recovered": recovered,
        "harmful": harmful,
        "net_recovered": recovered - harmful,
        "discordant": discordant,
        "raw_p": pvalue,
    }


def _cluster_bootstrap(
    reference: Sequence[bool],
    challenger: Sequence[bool],
    clusters: Sequence[Any],
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    reference_array = np.asarray(reference, dtype=bool)
    challenger_array = np.asarray(challenger, dtype=bool)
    cluster_array = np.asarray([str(value) for value in clusters], dtype=object)
    if (
        reference_array.ndim != 1
        or challenger_array.shape != reference_array.shape
        or cluster_array.shape != reference_array.shape
        or not len(reference_array)
    ):
        raise ValueError("cluster bootstrap inputs must be non-empty paired vectors")
    unique, inverse = np.unique(cluster_array, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("cluster bootstrap requires at least two sequences")
    difference = challenger_array.astype(float) - reference_array.astype(float)
    sums = np.bincount(inverse, weights=difference, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    generator = np.random.default_rng(int(seed))
    values = np.empty(int(iterations), dtype=float)
    for start in range(0, int(iterations), 1000):
        stop = min(start + 1000, int(iterations))
        draws = generator.integers(0, len(unique), size=(stop - start, len(unique)))
        values[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "point_estimate": float(difference.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "iterations": int(iterations),
        "seed": int(seed),
        "cluster_count": int(len(unique)),
        "resampling_unit": "sequence",
    }


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("Holm correction requires finite p-values")
    order = np.argsort(values, kind="stable")
    sorted_adjusted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    result = np.empty(len(values), dtype=float)
    result[order] = np.minimum(sorted_adjusted, 1.0)
    return result


def assert_threshold_monotonicity(results: pd.DataFrame) -> None:
    """Reject any success numerator that increases under stricter thresholds."""

    required = {"route", "iou_threshold", "angle_threshold_deg"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"threshold results miss monotonicity fields: {missing}")
    metric_columns = [
        name
        for name in (
            "native_success_num",
            "raw_reranked_success_num",
            "reranked_success_num",
            "oracle_num",
        )
        if name in results.columns
    ]
    for route, route_frame in results.groupby("route", sort=False):
        for metric in metric_columns:
            for angle, group in route_frame.groupby("angle_threshold_deg"):
                values = group.sort_values("iou_threshold")[metric].to_numpy(float)
                if np.any(np.diff(values) > 0):
                    raise RuntimeError(
                        f"threshold monotonicity violated for {route}/{metric}/angle={angle}"
                    )
            for iou, group in route_frame.groupby("iou_threshold"):
                values = group.sort_values("angle_threshold_deg")[metric].to_numpy(float)
                if np.any(np.diff(values) < 0):
                    raise RuntimeError(
                        f"threshold monotonicity violated for {route}/{metric}/iou={iou}"
                    )


def _formal_regression_expected(repo_root: Path, route: str) -> dict[str, int]:
    if route != "d1":
        path = (
            repo_root
            / "runs/fair_unified_reranking_20260809_103012/09_formal_test/formal_test_metrics.json"
        )
        if _sha256_file(path) != FORMAL_METRICS_SHA256["primary"]:
            raise RuntimeError("primary formal metrics hash differs from the locked source")
        metrics = _read_json(path)["systems"]
        return {
            "denominator": int(metrics[f"{route}_native"]["sample_count"]),
            "native": int(metrics[f"{route}_native"]["j_at_1_numerator"]),
            "raw": int(metrics[f"{route}_ungated_primary"]["j_at_1_numerator"]),
            "reranked": int(metrics[f"{route}_gated_primary"]["j_at_1_numerator"]),
            "oracle": int(metrics[f"{route}_native"]["j_at_5_numerator"]),
        }
    path = (
        repo_root
        / "runs/fair_d1_reranking_extension_20260811T145515Z/09_formal_test/formal_test_metrics.json"
    )
    if _sha256_file(path) != FORMAL_METRICS_SHA256["d1"]:
        raise RuntimeError("D1 formal metrics hash differs from the locked source")
    metrics = _read_json(path)["systems"]
    return {
        "denominator": int(metrics["d1_top5_r0"]["sample_count"]),
        "native": int(metrics["d1_top5_r0"]["selected_correct_numerator"]),
        "raw": int(metrics["d1_top5_r7_ungated"]["selected_correct_numerator"]),
        "reranked": int(metrics["d1_top5_r7_gated"]["selected_correct_numerator"]),
        "oracle": int(metrics["d1_top5_r0"]["oracle_numerator"]),
    }


def run_threshold_sensitivity(
    repo_root: Path, run_dir: Path, *, resume: bool = True
) -> dict[str, Any]:
    """Run all 36 route/threshold cells using frozen predictions and geometry."""

    repo_root, run_dir = Path(repo_root).resolve(), Path(run_dir).resolve()
    _assert_isolated_run_dir(repo_root, run_dir)
    output = run_dir / "4d_threshold_sensitivity"
    result_path = output / "results.csv"
    regression_path = output / "regression_check.json"
    if resume:
        completion = _analysis_completion_valid(
            repo_root,
            run_dir,
            output,
            "completion.json",
            statuses={"COMPLETE"},
        )
        if completion is not None:
            return {
                "status": "COMPLETE",
                "cells": int(completion["cells"]),
                "results": str(result_path),
                "completion_signature_sha256": completion[
                    "completion_signature_sha256"
                ],
                "resumed": True,
            }

    evaluator, evaluator_path, evaluator_hash = load_frozen_canonical_evaluator(repo_root)
    labels_path = (
        repo_root
        / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500/manifests/test_labels.parquet"
    )
    sources = _route_sources(repo_root)
    required: list[Path] = [
        evaluator_path,
        labels_path,
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/09_formal_test/formal_test_metrics.json",
        repo_root
        / "runs/fair_d1_reranking_extension_20260811T145515Z/09_formal_test/formal_test_metrics.json",
    ]
    for source in sources.values():
        required.extend(
            [
                source.candidates_top5,
                source.raw_decisions,
                source.gated_decisions,
            ]
        )
        if source.native_decisions is not None:
            required.append(source.native_decisions)
    _require_files(required)
    denominator = _denominator(repo_root)
    result_rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    regression_rows: dict[str, Any] = {}

    for route in ROUTES:
        source = sources[route]
        candidates = pd.read_parquet(
            source.candidates_top5,
            columns=["sample_id", "candidate_id", "native_rank"],
        )
        candidates["sample_id"] = candidates["sample_id"].astype(str)
        candidates["candidate_id"] = candidates["candidate_id"].astype(str)
        pairwise = _build_pairwise_cache(
            candidates_path=source.candidates_top5,
            labels_path=labels_path,
            evaluator=evaluator,
            evaluator_sha256=evaluator_hash,
            destination=output / "pairwise_cache" / f"{route}.parquet",
            resume=resume,
        )
        native_decisions = _normalise_decisions(
            source.native_decisions,
            native_from_candidates=candidates if source.native_decisions is None else None,
        ) if source.native_decisions is not None else _normalise_decisions(
            source.gated_decisions, native_from_candidates=candidates
        )
        raw_decisions = _normalise_decisions(source.raw_decisions)
        gated_decisions = _normalise_decisions(source.gated_decisions)

        route_significance_indexes: list[int] = []
        for iou_threshold, angle_threshold in THRESHOLD_GRID:
            success = _success_keys(pairwise, iou_threshold, angle_threshold)
            native = _outcomes_for_decisions(denominator, native_decisions, success)
            raw = _outcomes_for_decisions(denominator, raw_decisions, success)
            gated = _outcomes_for_decisions(denominator, gated_decisions, success)
            oracle = _oracle_outcomes(denominator, candidates, success)
            test = _mcnemar(native, gated)
            bootstrap = _cluster_bootstrap(
                native, gated, denominator["sequence_id"], seed=BOOTSTRAP_SEED
            )
            n = int(len(denominator))
            native_n, raw_n = int(native.sum()), int(raw.sum())
            reranked_n, oracle_n = int(gated.sum()), int(oracle.sum())
            headroom = oracle_n - native_n
            row = {
                "route": route.upper(),
                "retrospective": bool(source.retrospective),
                "iou_threshold": float(iou_threshold),
                "angle_threshold_deg": int(angle_threshold),
                "formal_cell": bool((iou_threshold, angle_threshold) == FORMAL_THRESHOLD),
                "n": n,
                "native_success_num": native_n,
                "native_j1": native_n / n,
                "raw_reranked_success_num": raw_n,
                "raw_reranked_j1": raw_n / n,
                "reranked_success_num": reranked_n,
                "reranked_j1": reranked_n / n,
                "delta_pp": 100.0 * (reranked_n - native_n) / n,
                "native_jany": oracle_n / n,
                "reranked_jany": oracle_n / n,
                "oracle_num": oracle_n,
                "oracle_at_k": oracle_n / n,
                "recovered": test["recovered"],
                "harmful": test["harmful"],
                "net_recovered": test["net_recovered"],
                "outcome_changing_precision": (
                    test["recovered"] / test["discordant"]
                    if test["discordant"]
                    else math.nan
                ),
                "headroom_recovery": (
                    (reranked_n - native_n) / headroom if headroom > 0 else math.nan
                ),
                "bootstrap_ci_low_pp": 100.0 * bootstrap["ci_low"],
                "bootstrap_ci_high_pp": 100.0 * bootstrap["ci_high"],
                "mcnemar_raw_p": test["raw_p"],
                "mcnemar_holm_p": math.nan,
            }
            result_rows.append(row)
            result_index = len(result_rows) - 1
            route_significance_indexes.append(result_index)
            bootstrap_rows.append(
                {
                    "route": route.upper(),
                    "retrospective": bool(source.retrospective),
                    "iou_threshold": iou_threshold,
                    "angle_threshold_deg": angle_threshold,
                    "point_estimate_pp": 100.0 * bootstrap["point_estimate"],
                    "ci_low_pp": 100.0 * bootstrap["ci_low"],
                    "ci_high_pp": 100.0 * bootstrap["ci_high"],
                    "iterations": bootstrap["iterations"],
                    "seed": bootstrap["seed"],
                    "sequence_count": bootstrap["cluster_count"],
                }
            )
            significance_rows.append(
                {
                    "route": route.upper(),
                    "retrospective": bool(source.retrospective),
                    "iou_threshold": iou_threshold,
                    "angle_threshold_deg": angle_threshold,
                    **test,
                    "holm_adjusted_p": math.nan,
                    "confirmatory": bool(
                        (iou_threshold, angle_threshold) == FORMAL_THRESHOLD
                    ),
                }
            )
            paired_frames.append(
                pd.DataFrame(
                    {
                        "route": route.upper(),
                        "retrospective": bool(source.retrospective),
                        "iou_threshold": iou_threshold,
                        "angle_threshold_deg": angle_threshold,
                        "formal_cell": (iou_threshold, angle_threshold)
                        == FORMAL_THRESHOLD,
                        "sample_id": denominator["sample_id"],
                        "scene_id": denominator["scene_id"],
                        "sequence_id": denominator["sequence_id"],
                        "native_correct": native,
                        "raw_reranked_correct": raw,
                        "reranked_correct": gated,
                        "oracle_at_k": oracle,
                    }
                )
            )

        adjusted = _holm_adjust(
            [result_rows[index]["mcnemar_raw_p"] for index in route_significance_indexes]
        )
        start = len(significance_rows) - len(THRESHOLD_GRID)
        for offset, (index, adjusted_p) in enumerate(
            zip(route_significance_indexes, adjusted, strict=True)
        ):
            result_rows[index]["mcnemar_holm_p"] = float(adjusted_p)
            significance_rows[start + offset]["holm_adjusted_p"] = float(adjusted_p)

        centre = next(
            row
            for row in result_rows
            if row["route"] == route.upper() and row["formal_cell"]
        )
        expected = _formal_regression_expected(repo_root, route)
        observed = {
            "denominator": int(centre["n"]),
            "native": int(centre["native_success_num"]),
            "raw": int(centre["raw_reranked_success_num"]),
            "reranked": int(centre["reranked_success_num"]),
            "oracle": int(centre["oracle_num"]),
        }
        regression_rows[route.upper()] = {
            "retrospective": bool(source.retrospective),
            "expected": expected,
            "observed": observed,
            "exact": observed == expected,
        }

    results = pd.DataFrame(result_rows)
    if len(results) != len(ROUTES) * len(THRESHOLD_GRID):
        raise RuntimeError("threshold grid is incomplete")
    assert_threshold_monotonicity(results)
    regression_ok = all(value["exact"] for value in regression_rows.values())
    regression = {
        "status": "PASS" if regression_ok else "FAIL",
        "formal_contract": {"iou": "> 0.25", "angle": "<= 30 deg"},
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": evaluator_hash,
        "routes": regression_rows,
    }
    _atomic_csv(result_path, results)
    _atomic_parquet(output / "paired_outcomes.parquet", pd.concat(paired_frames, ignore_index=True))
    _atomic_csv(output / "bootstrap_results.csv", pd.DataFrame(bootstrap_rows))
    _atomic_csv(output / "significance_tests.csv", pd.DataFrame(significance_rows))
    _atomic_json(regression_path, regression)
    if not regression_ok:
        raise RuntimeError("formal 0.25/30 threshold cell did not exactly reproduce locked metrics")
    completion = _write_analysis_completion(
        repo_root,
        run_dir,
        output,
        "completion.json",
        status="COMPLETE",
        output_paths=[
            result_path,
            output / "paired_outcomes.parquet",
            output / "bootstrap_results.csv",
            output / "significance_tests.csv",
            regression_path,
            *sorted((output / "pairwise_cache").glob("*.parquet")),
        ],
        source_paths=required,
        metadata={"cells": int(len(results)), "evaluator_sha256": evaluator_hash},
    )
    return {
        "status": "COMPLETE",
        "cells": int(len(results)),
        "routes": list(map(str.upper, ROUTES)),
        "results": str(result_path),
        "regression": regression,
        "completion_signature_sha256": completion[
            "completion_signature_sha256"
        ],
        "resumed": False,
    }


def _duplicate_candidates(repo_root: Path, run_dir: Path) -> list[Path]:
    patterns = (
        "*near_duplicate*.csv",
        "*near_duplicate*.parquet",
        "*near_duplicate*.json",
        "*duplicate_pairs*.csv",
        "*duplicate_pairs*.parquet",
        "*train_test_overlap*.csv",
        "*train_test_overlap*.parquet",
    )
    roots = [
        root
        for root in (
            repo_root / "artifacts",
            repo_root / "runs",
            repo_root / "HiFi_reproduction/reports",
            repo_root / "HiFi_reproduction/runs",
        )
        if root.exists()
    ]
    command = ["rg", "--files", *map(str, roots)]
    for pattern in patterns:
        command.extend(["-g", pattern])
    command.extend(
        [
            "-g",
            "!**/.venv*/**",
            "-g",
            "!**/superseded*/**",
            "-g",
            "!**/candidate_cache/**",
        ]
    )
    completed = subprocess.run(
        command, cwd=repo_root, check=False, capture_output=True, text=True
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"duplicate-map inventory failed: {completed.stderr.strip()}")
    found: set[Path] = set()
    for raw_path in completed.stdout.splitlines():
        path = Path(raw_path).resolve()
        if run_dir not in path.parents and "candidate" not in path.name.lower():
            found.add(path)
    return sorted(found)


def _normalise_duplicate_map(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("pairs") if isinstance(payload, dict) else payload
        frame = pd.DataFrame(rows)
    aliases = {
        "test_sample_id": ("test_sample_id", "test_tuple_id", "test_id"),
        "train_sample_id": ("train_sample_id", "train_tuple_id", "train_id"),
        "similarity_score": ("similarity_score", "similarity", "score"),
    }
    rename: dict[str, str] = {}
    for canonical, options in aliases.items():
        selected = next((name for name in options if name in frame.columns), None)
        if selected is None and canonical != "similarity_score":
            raise ValueError(f"duplicate map lacks {canonical}: {path}")
        if selected is not None:
            rename[selected] = canonical
    result = frame.rename(columns=rename).copy()
    result["test_sample_id"] = result["test_sample_id"].astype(str)
    result["train_sample_id"] = result["train_sample_id"].astype(str)
    if "similarity_score" not in result:
        result["similarity_score"] = math.nan
    if result.empty or result[["test_sample_id", "train_sample_id"]].eq("").any().any():
        raise ValueError("duplicate map contains no valid pairs")
    return result


def _bootstrap_shift(
    frame: pd.DataFrame, removed: set[str], *, iterations: int = BOOTSTRAP_ITERATIONS
) -> dict[str, Any]:
    sequences = frame["sequence_id"].astype(str).to_numpy()
    unique, inverse = np.unique(sequences, return_inverse=True)
    full_delta = frame["reranked_correct"].astype(float) - frame["native_correct"].astype(float)
    keep = ~frame["sample_id"].astype(str).isin(removed).to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    distribution = np.empty(iterations, dtype=float)
    for index in range(iterations):
        chosen = rng.integers(0, len(unique), size=len(unique))
        row_parts = [np.flatnonzero(inverse == item) for item in chosen]
        rows = np.concatenate(row_parts)
        full = float(full_delta.iloc[rows].mean())
        filtered_rows = rows[keep[rows]]
        distribution[index] = (
            math.nan if not len(filtered_rows) else float(full_delta.iloc[filtered_rows].mean() - full)
        )
    finite = distribution[np.isfinite(distribution)]
    if not len(finite):
        raise RuntimeError("duplicate-exclusion shift bootstrap has no finite replicates")
    filtered_delta = float(full_delta.loc[keep].mean())
    return {
        "point_estimate": filtered_delta - float(full_delta.mean()),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "iterations": int(iterations),
        "seed": BOOTSTRAP_SEED,
        "cluster_count": int(len(unique)),
    }


def _apply_duplicate_filter(
    paired: pd.DataFrame, removed_sample_ids: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark and filter duplicate IDs without altering any prediction column."""

    if "sample_id" not in paired.columns:
        raise ValueError("paired outcomes lack sample_id")
    removed = {str(value) for value in removed_sample_ids}
    sample_ids = paired["sample_id"].astype(str)
    unknown = sorted(removed.difference(set(sample_ids)))
    if unknown:
        raise ValueError(f"duplicate map references unknown test tuples: {unknown[:5]}")
    marked = paired.copy()
    marked["excluded"] = sample_ids.isin(removed).to_numpy()
    filtered = marked.loc[~marked["excluded"]].copy()

    # Filtering may remove rows, but it must never recompute or mutate a
    # source prediction.  Preserve both row order and every source column.
    expected = paired.loc[~sample_ids.isin(removed)].reset_index(drop=True)
    observed = filtered[list(paired.columns)].reset_index(drop=True)
    if not observed.equals(expected):
        raise RuntimeError("duplicate filtering changed frozen predictions")
    return marked, filtered


def _common_filtered_sample_ids(
    paired: pd.DataFrame, removed_sample_ids: Iterable[str]
) -> tuple[str, ...]:
    """Return the deterministic post-exclusion intersection across routes."""

    required = {"route", "sample_id"}
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"paired outcomes lack common-intersection fields: {missing}")
    route_sets = [
        set(frame["sample_id"].astype(str))
        for _, frame in paired.groupby("route", sort=True)
    ]
    if not route_sets:
        return ()
    removed = {str(value) for value in removed_sample_ids}
    return tuple(sorted(set.intersection(*route_sets).difference(removed)))


def run_duplicate_exclusion(
    repo_root: Path, run_dir: Path, *, resume: bool = True
) -> dict[str, Any]:
    """Filter only a previously locked near-duplicate map, otherwise fail closed."""

    repo_root, run_dir = Path(repo_root).resolve(), Path(run_dir).resolve()
    _assert_isolated_run_dir(repo_root, run_dir)
    output = run_dir / "4d_near_duplicate_excluded"
    source_record_path = output / "source_duplicate_map.json"
    if resume:
        completion = _analysis_completion_valid(
            repo_root,
            run_dir,
            output,
            "completion.json",
            statuses={"COMPLETE", "FAIL_CLOSED"},
        )
        if completion is not None and source_record_path.is_file():
            previous = _read_json(source_record_path)
            if previous.get("status") == completion["status"]:
                return {
                    **previous,
                    "completion_signature_sha256": completion[
                        "completion_signature_sha256"
                    ],
                    "resumed": True,
                }
    source_audit = run_dir / "SOURCE_AUDIT.md"
    audit_text = (
        source_audit.read_text(encoding="utf-8", errors="ignore")
        if source_audit.is_file()
        else ""
    )
    audited_absence = (
        "Near duplicates:" in audit_text
        and "found no train/test near-duplicate map" in audit_text
        and "This subexperiment is fail-closed" in audit_text
    )
    candidates = [] if audited_absence else _duplicate_candidates(repo_root, run_dir)
    # A map is eligible only when a formal/paper audit names it.  Merely finding
    # an arbitrary similarity file cannot establish the pre-outcome threshold.
    citations: list[tuple[Path, Path]] = []
    audit_files = (
        [
            *repo_root.glob("artifacts/**/*audit*.md"),
            *repo_root.glob("runs/**/*audit*.md"),
            *repo_root.glob("**/SOURCE_AUDIT.md"),
        ]
        if candidates
        else []
    )
    for map_path in candidates:
        for audit in audit_files:
            try:
                text = audit.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if map_path.name in text or str(map_path) in text:
                citations.append((map_path, audit.resolve()))
    eligible = sorted(set(path for path, _audit in citations))
    if len(eligible) != 1:
        reason = (
            "No pre-existing near-duplicate map was explicitly cited by a formal/paper "
            "audit; the threshold and method therefore cannot be reconstructed without "
            "post-outcome invention."
            if not eligible
            else "Multiple formally cited duplicate maps exist and no primary map is designated."
        )
        record = {
            "status": "FAIL_CLOSED",
            "reason": reason,
            "candidate_maps_found": [str(path) for path in candidates],
            "eligible_maps": [str(path) for path in eligible],
            "source_audit_path": str(source_audit) if source_audit.is_file() else None,
            "source_audit_sha256": (
                _sha256_file(source_audit) if source_audit.is_file() else None
            ),
            "post_hoc_robustness_analysis": True,
        }
        _atomic_json(source_record_path, record)
        _atomic_csv(
            output / "exclusion_manifest.csv",
            pd.DataFrame(
                columns=[
                    "test_sample_id",
                    "train_sample_id",
                    "similarity_score",
                    "sequence_id",
                    "removal_reason",
                ]
            ),
        )
        failed = pd.DataFrame([{"status": "FAIL_CLOSED", "reason": reason}])
        _atomic_csv(output / "full_vs_filtered_metrics.csv", failed)
        _atomic_parquet(
            output / "paired_outcomes.parquet",
            pd.DataFrame(
                columns=[
                    "route",
                    "sample_id",
                    "sequence_id",
                    "native_correct",
                    "reranked_correct",
                    "excluded",
                ]
            ),
        )
        _atomic_csv(output / "bootstrap_results.csv", failed)
        _atomic_json(output / "significance_tests.json", record)
        completion = _write_analysis_completion(
            repo_root,
            run_dir,
            output,
            "completion.json",
            status="FAIL_CLOSED",
            output_paths=[
                source_record_path,
                output / "exclusion_manifest.csv",
                output / "full_vs_filtered_metrics.csv",
                output / "paired_outcomes.parquet",
                output / "bootstrap_results.csv",
                output / "significance_tests.json",
            ],
            metadata={
                "reason": reason,
                "source_audit_sha256": record["source_audit_sha256"],
            },
        )
        record["completion_signature_sha256"] = completion[
            "completion_signature_sha256"
        ]
        return record

    map_path = eligible[0]
    duplicate_map = _normalise_duplicate_map(map_path)
    denominator = _denominator(repo_root)
    known_test = set(denominator["sample_id"])
    unknown = sorted(set(duplicate_map["test_sample_id"]).difference(known_test))
    if unknown:
        raise ValueError(f"duplicate map references unknown formal test tuples: {unknown[:5]}")
    removed = set(duplicate_map["test_sample_id"])
    sequence_by_id = denominator.set_index("sample_id")["sequence_id"]
    exclusion = duplicate_map.copy()
    exclusion["sequence_id"] = exclusion["test_sample_id"].map(sequence_by_id)
    exclusion["removal_reason"] = "flagged_train_near_duplicate"
    exclusion = exclusion.sort_values(
        ["test_sample_id", "train_sample_id"], kind="mergesort"
    ).reset_index(drop=True)

    threshold_outcomes = run_dir / "4d_threshold_sensitivity/paired_outcomes.parquet"
    if not threshold_outcomes.is_file():
        raise FileNotFoundError(
            "duplicate exclusion requires completed formal-cell threshold paired outcomes"
        )
    paired = pd.read_parquet(threshold_outcomes)
    paired = paired.loc[paired["formal_cell"].astype(bool)].copy()
    if paired.empty:
        raise RuntimeError("formal-cell paired outcomes are absent")
    paired, _filtered = _apply_duplicate_filter(paired, removed)
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    significance: dict[str, Any] = {"routes": {}}
    for route, frame in paired.groupby("route", sort=False):
        route_tests: dict[str, Any] = {}
        full_gain: float | None = None
        for subset, selected in (
            ("full", frame),
            ("duplicate_excluded", frame.loc[~frame["excluded"]]),
        ):
            if selected.empty:
                raise RuntimeError(f"duplicate exclusion removed every tuple for {route}")
            native = selected["native_correct"].to_numpy(bool)
            reranked = selected["reranked_correct"].to_numpy(bool)
            oracle = selected["oracle_at_k"].to_numpy(bool)
            test = _mcnemar(native, reranked)
            boot = _cluster_bootstrap(native, reranked, selected["sequence_id"])
            gain = float(reranked.mean() - native.mean())
            full_gain = gain if subset == "full" else full_gain
            headroom = float(oracle.mean() - native.mean())
            metric_rows.append(
                {
                    "route": route,
                    "retrospective": bool(selected["retrospective"].iloc[0]),
                    "analysis_set": "within_route",
                    "subset": subset,
                    "n_tuples": int(len(selected)),
                    "n_sequences": int(selected["sequence_id"].nunique()),
                    "native_num": int(native.sum()),
                    "native_j1": float(native.mean()),
                    "reranked_num": int(reranked.sum()),
                    "reranked_j1": float(reranked.mean()),
                    "absolute_gain_pp": 100.0 * gain,
                    "recovered": test["recovered"],
                    "harmful": test["harmful"],
                    "net_recovered": test["net_recovered"],
                    "outcome_changing_precision": (
                        test["recovered"] / test["discordant"]
                        if test["discordant"]
                        else math.nan
                    ),
                    "oracle_num": int(oracle.sum()),
                    "oracle_at_k": float(oracle.mean()),
                    "headroom_recovery": gain / headroom if headroom > 0 else math.nan,
                    "ci_low_pp": 100.0 * boot["ci_low"],
                    "ci_high_pp": 100.0 * boot["ci_high"],
                    "mcnemar_p": test["raw_p"],
                }
            )
            route_tests[subset] = test
            bootstrap_rows.append(
                {
                    "route": route,
                    "analysis": "paired_gain",
                    "subset": subset,
                    "point_estimate_pp": 100.0 * boot["point_estimate"],
                    "ci_low_pp": 100.0 * boot["ci_low"],
                    "ci_high_pp": 100.0 * boot["ci_high"],
                    "iterations": boot["iterations"],
                    "seed": boot["seed"],
                }
            )
        shift = _bootstrap_shift(frame.reset_index(drop=True), removed)
        filtered_gain = next(
            row["absolute_gain_pp"] / 100.0
            for row in reversed(metric_rows)
            if row["route"] == route and row["subset"] == "duplicate_excluded"
        )
        for row in metric_rows:
            if row["route"] == route:
                row["sensitivity_shift_pp"] = 100.0 * (filtered_gain - float(full_gain))
        bootstrap_rows.append(
            {
                "route": route,
                "analysis": "sensitivity_shift",
                "subset": "full_vs_duplicate_excluded",
                "point_estimate_pp": 100.0 * shift["point_estimate"],
                "ci_low_pp": 100.0 * shift["ci_low"],
                "ci_high_pp": 100.0 * shift["ci_high"],
                "iterations": shift["iterations"],
                "seed": shift["seed"],
            }
        )
        route_tests["sensitivity_shift"] = shift
        significance["routes"][route] = route_tests

    # Every route uses the same locked 7,675-tuple universe; nevertheless this
    # is emitted explicitly as the common-intersection analysis.
    common = set(_common_filtered_sample_ids(paired, removed))
    for route, frame in paired.groupby("route", sort=False):
        selected = frame.loc[frame["sample_id"].astype(str).isin(common)]
        native = selected["native_correct"].to_numpy(bool)
        reranked = selected["reranked_correct"].to_numpy(bool)
        oracle = selected["oracle_at_k"].to_numpy(bool)
        test = _mcnemar(native, reranked)
        boot = _cluster_bootstrap(native, reranked, selected["sequence_id"])
        headroom = float(oracle.mean() - native.mean())
        gain = float(reranked.mean() - native.mean())
        metric_rows.append(
            {
                "route": route,
                "retrospective": bool(selected["retrospective"].iloc[0]),
                "analysis_set": "common_intersection",
                "subset": "duplicate_excluded",
                "n_tuples": int(len(selected)),
                "n_sequences": int(selected["sequence_id"].nunique()),
                "native_num": int(native.sum()),
                "native_j1": float(native.mean()),
                "reranked_num": int(reranked.sum()),
                "reranked_j1": float(reranked.mean()),
                "absolute_gain_pp": 100.0 * gain,
                "recovered": test["recovered"],
                "harmful": test["harmful"],
                "net_recovered": test["net_recovered"],
                "outcome_changing_precision": (
                    test["recovered"] / test["discordant"] if test["discordant"] else math.nan
                ),
                "oracle_num": int(oracle.sum()),
                "oracle_at_k": float(oracle.mean()),
                "headroom_recovery": gain / headroom if headroom > 0 else math.nan,
                "ci_low_pp": 100.0 * boot["ci_low"],
                "ci_high_pp": 100.0 * boot["ci_high"],
                "mcnemar_p": test["raw_p"],
                "sensitivity_shift_pp": math.nan,
            }
        )

    record = {
        "status": "COMPLETE",
        "path": str(map_path),
        "sha256": _sha256_file(map_path),
        "citing_audits": [str(audit) for path, audit in citations if path == map_path],
        "number_of_pairs": int(len(duplicate_map)),
        "unique_test_tuples_affected": int(len(removed)),
        "sequences_affected": int(exclusion["sequence_id"].nunique()),
        "selection_rule": "formal/paper-audit-cited map only; no outcome-aware threshold selection",
        "post_hoc_robustness_analysis": True,
    }
    _atomic_json(source_record_path, record)
    _atomic_csv(output / "exclusion_manifest.csv", exclusion)
    _atomic_csv(output / "full_vs_filtered_metrics.csv", pd.DataFrame(metric_rows))
    _atomic_parquet(output / "paired_outcomes.parquet", paired)
    _atomic_csv(output / "bootstrap_results.csv", pd.DataFrame(bootstrap_rows))
    _atomic_json(output / "significance_tests.json", significance)
    completion = _write_analysis_completion(
        repo_root,
        run_dir,
        output,
        "completion.json",
        status="COMPLETE",
        output_paths=[
            source_record_path,
            output / "exclusion_manifest.csv",
            output / "full_vs_filtered_metrics.csv",
            output / "paired_outcomes.parquet",
            output / "bootstrap_results.csv",
            output / "significance_tests.json",
        ],
        source_paths=[map_path],
        metadata={
            "duplicate_map_sha256": record["sha256"],
            "removed_tuple_count": record["unique_test_tuples_affected"],
        },
    )
    record["completion_signature_sha256"] = completion[
        "completion_signature_sha256"
    ]
    return record


def native_prefix(candidates: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Return the true unique native-ranked post-NMS prefix, without padding."""

    required = {"sample_id", "candidate_id", "native_rank"}
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidate prefix input misses columns: {missing}")
    work = candidates.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["candidate_id"] = work["candidate_id"].astype(str)
    if work.duplicated(["sample_id", "candidate_id"]).any():
        raise ValueError("candidate IDs are not unique within sample")
    ranks = pd.to_numeric(work["native_rank"], errors="coerce")
    if ranks.isna().any() or (ranks <= 0).any():
        raise ValueError("native ranks must be positive integers")
    work["native_rank"] = ranks.astype(int)
    for sample_id, group in work.groupby("sample_id", sort=False):
        observed = sorted(group["native_rank"].tolist())
        if observed != list(range(1, len(observed) + 1)):
            raise ValueError(f"native ranks are not contiguous for {sample_id}")
    if k is not None:
        if int(k) <= 0:
            raise ValueError("K must be positive")
        work = work.loc[work["native_rank"].le(int(k))].copy()
    return work.sort_values(
        ["sample_id", "native_rank", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)


def _formal_candidate_labels(repo_root: Path, route: str, candidates: pd.DataFrame) -> pd.DataFrame:
    if route == "d1":
        path = (
            repo_root
            / "runs/fair_d1_reranking_extension_20260811T145515Z/09_formal_test/formal_candidate_outcomes.parquet"
        )
        labels = pd.read_parquet(
            path, columns=["source_route", "sample_id", "candidate_id", "candidate_success"]
        )
        labels = labels.loc[labels["source_route"].eq("D1")].drop(columns="source_route")
    else:
        path = (
            repo_root
            / "runs/fair_unified_reranking_20260809_103012/09_formal_test/per_candidate_scores.parquet"
        )
        labels = pd.read_parquet(
            path,
            columns=["system_name", "sample_id", "candidate_id", "candidate_success"],
        )
        labels = labels.loc[labels["system_name"].eq(f"{route}_native")].drop(
            columns="system_name"
        )
    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    labels = labels.drop_duplicates(["sample_id", "candidate_id"])
    keys = candidates[["sample_id", "candidate_id"]].astype(str)
    joined = keys.merge(labels, on=["sample_id", "candidate_id"], how="left", validate="one_to_one")
    if joined["candidate_success"].isna().any():
        raise ValueError(
            f"formal candidate outcomes do not cover requested {route.upper()} prefix"
        )
    return joined


def _candidate_ceiling_row(
    denominator: pd.DataFrame,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    route: str,
    k_label: str,
    retrospective: bool,
) -> dict[str, Any]:
    labelled = candidates.merge(
        labels, on=["sample_id", "candidate_id"], how="left", validate="one_to_one"
    )
    if labelled["candidate_success"].isna().any():
        raise ValueError("candidate ceiling labels are incomplete")
    labelled["candidate_success"] = labelled["candidate_success"].astype(bool)
    counts = labelled.groupby("sample_id").size()
    positives = labelled.loc[labelled["candidate_success"]]
    first = positives.groupby("sample_id")["native_rank"].min()
    native_ids = set(
        labelled.loc[
            labelled["native_rank"].eq(1) & labelled["candidate_success"], "sample_id"
        ].astype(str)
    )
    oracle_ids = set(positives["sample_id"].astype(str))
    n = len(denominator)
    native = denominator["sample_id"].isin(native_ids)
    oracle = denominator["sample_id"].isin(oracle_ids)
    return {
        "route": route.upper(),
        "retrospective": retrospective,
        "k": k_label,
        "n_groups": int(n),
        "candidate_rows": int(len(labelled)),
        "mean_actual_candidate_count": float(
            denominator["sample_id"].map(counts).fillna(0).mean()
        ),
        "non_empty_pool_num": int(denominator["sample_id"].isin(counts.index).sum()),
        "non_empty_pool_rate": float(denominator["sample_id"].isin(counts.index).mean()),
        "valid_candidate_count": int(labelled["candidate_success"].sum()),
        "oracle_num": int(oracle.sum()),
        "oracle_at_k": float(oracle.mean()),
        "candidate_absence_num": int((~oracle).sum()),
        "candidate_absence_rate": float((~oracle).mean()),
        "native_num": int(native.sum()),
        "native_j1": float(native.mean()),
        "mrr": float(
            denominator["sample_id"].map(first).map(
                lambda rank: 0.0 if pd.isna(rank) else 1.0 / float(rank)
            ).mean()
        ),
        "mean_first_valid_rank": float(first.mean()) if len(first) else math.nan,
    }


def _topk_source_paths(repo_root: Path, route: str, split: str) -> tuple[Path, Path]:
    if route != "d1":
        run = repo_root / "runs/fair_unified_reranking_20260809_103012"
        return (
            run
            / f"03_features/tracks/T2_matched_common/{route}_{split}/candidate_features.parquet",
            run / f"03_features/candidate_labels_{route}_{split}_top5.parquet",
        )
    run = repo_root / "runs/fair_d1_reranking_extension_20260811T145515Z"
    return (
        run
        / f"03_features/{split}/top5/T2_matched_common/candidate_features.parquet",
        run / f"03_features/{split}/top5/labels/candidate_labels.parquet",
    )


def _model_feature_columns(repo_root: Path, route: str) -> tuple[str, ...]:
    if route != "d1":
        selected = _read_json(
            repo_root
            / "runs/fair_unified_reranking_20260809_103012/08_lock/selected_features.json"
        )
        return tuple(map(str, selected["routes"][route]["feature_columns"]))
    manifest = _read_json(
        repo_root
        / "runs/fair_d1_reranking_extension_20260811T145515Z/03_features/train/top5/T2_matched_common/manifest.json"
    )
    return tuple(map(str, manifest["model_feature_columns"]))


def _ranker_hyperparameters(repo_root: Path, route: str) -> dict[str, Any]:
    if route != "d1":
        selected = _read_json(
            repo_root
            / "runs/fair_unified_reranking_20260809_103012/08_lock/selected_hyperparameters.json"
        )["routes"][route]["seed_configurations"]
        configurations = {
            (
                int(value["num_leaves"]),
                float(value["tree_learning_rate"]),
                int(value["n_estimators"]),
            )
            for value in selected.values()
        }
    else:
        trial = _read_json(
            repo_root
            / "runs/fair_d1_reranking_extension_20260811T145515Z/07_validation/primary_selection/trials/f2ed1d3e0611d7d7/manifest.json"
        )["configuration"]
        configurations = {
            (
                int(trial["num_leaves"]),
                float(trial["learning_rate"]),
                int(trial["n_estimators"]),
            )
        }
    if len(configurations) != 1:
        raise RuntimeError(f"formal LambdaMART hyperparameters disagree for {route}")
    leaves, rate, estimators = configurations.pop()
    return {"num_leaves": leaves, "learning_rate": rate, "n_estimators": estimators}


def _topk_contract_paths(repo_root: Path, route: str) -> tuple[Path, ...]:
    source = _route_sources(repo_root)[route]
    paths: list[Path] = [
        source.candidates_all,
        repo_root
        / "runs/fair_unified_reranking_20260809_103012/01_manifests/paired_test.parquet",
    ]
    for split in ("train", "validation", "test"):
        feature_path, label_path = _topk_source_paths(repo_root, route, split)
        paths.append(feature_path)
        if split != "test":
            paths.append(label_path)
    if route == "d1":
        paths.extend(
            [
                repo_root
                / "runs/fair_d1_reranking_extension_20260811T145515Z/09_formal_test/formal_candidate_outcomes.parquet",
                repo_root
                / "runs/fair_d1_reranking_extension_20260811T145515Z/03_features/train/top5/T2_matched_common/manifest.json",
                repo_root
                / "runs/fair_d1_reranking_extension_20260811T145515Z/07_validation/primary_selection/trials/f2ed1d3e0611d7d7/manifest.json",
            ]
        )
    else:
        paths.extend(
            [
                repo_root
                / "runs/fair_unified_reranking_20260809_103012/09_formal_test/per_candidate_scores.parquet",
                repo_root
                / "runs/fair_unified_reranking_20260809_103012/08_lock/selected_features.json",
                repo_root
                / "runs/fair_unified_reranking_20260809_103012/08_lock/selected_hyperparameters.json",
            ]
        )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def _topk_cell_contract(
    repo_root: Path,
    route: str,
    k: int,
    columns: Sequence[str],
    hyperparameters: dict[str, Any],
    preprocessor_artifact: dict[str, Any],
    prefix: pd.DataFrame,
    hash_cache: dict[str, str],
) -> dict[str, Any]:
    source_hashes: dict[str, str] = {}
    for path in _topk_contract_paths(repo_root, route):
        key = str(path)
        if key not in hash_cache:
            hash_cache[key] = _sha256_file(path)
        source_hashes[str(path.relative_to(repo_root))] = hash_cache[key]
    key_records = prefix[["sample_id", "candidate_id", "native_rank"]].astype(
        {"sample_id": str, "candidate_id": str, "native_rank": int}
    )
    candidate_key_sha256 = _sha256_json(key_records.to_dict(orient="records"))
    config = {
        "route": route.upper(),
        "k": int(k),
        "seeds": list(RANKER_SEEDS),
        "feature_columns": list(columns),
        "hyperparameters": hyperparameters,
        "objective": "lambdarank",
        "label_gain": [0, 1],
        "gate": "FAIL_CLOSED_NATIVE_NO_K_SPECIFIC_OOF_GATE",
    }
    contract = {
        "source_sha256": source_hashes,
        "config_sha256": _sha256_json(config),
        "preprocessor_sha256": _sha256_json(preprocessor_artifact),
        "candidate_keys_sha256": candidate_key_sha256,
    }
    contract["cell_signature_sha256"] = _sha256_json(contract)
    return contract


def _load_valid_seed_artifact(
    *,
    seed: int,
    route: str,
    k: int,
    cell_dir: Path,
    score_frame: pd.DataFrame,
    hyperparameters: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, float] | None:
    seed_path = cell_dir / f"seed_{seed}_scores.parquet"
    model_path = cell_dir / f"seed_{seed}_model.txt"
    manifest_path = cell_dir / f"seed_{seed}.json"
    if not all(path.is_file() for path in (seed_path, model_path, manifest_path)):
        return None
    manifest = _read_json(manifest_path)
    core_expected = {
        "status": "COMPLETE",
        "seed": seed,
        "k": k,
        "route": route.upper(),
        "hyperparameters": hyperparameters,
        "preprocessing_fit_split": "train",
        "candidate_ids_frozen": True,
        "candidate_geometry_unchanged": True,
    }
    if any(manifest.get(name) != value for name, value in core_expected.items()):
        return None
    for name, value in contract.items():
        recorded = manifest.get(name)
        if recorded is not None and recorded != value:
            return None
    scores = pd.read_parquet(seed_path)
    keys = ["sample_id", "candidate_id", "native_rank"]
    if (
        list(scores.columns) != [*keys, "score"]
        or len(scores) != len(score_frame)
        or not scores[keys].reset_index(drop=True).equals(
            score_frame[keys].reset_index(drop=True)
        )
        or not np.isfinite(pd.to_numeric(scores["score"], errors="coerce")).all()
    ):
        return None
    score_sha = _sha256_file(seed_path)
    model_sha = _sha256_file(model_path)
    if manifest.get("score_sha256") not in {None, score_sha}:
        return None
    if manifest.get("model_sha256") not in {None, model_sha}:
        return None
    seed_signature = _sha256_json(
        {
            "cell_signature_sha256": contract["cell_signature_sha256"],
            "seed": seed,
            "score_sha256": score_sha,
            "model_sha256": model_sha,
        }
    )
    if manifest.get("seed_signature_sha256") not in {None, seed_signature}:
        return None
    upgraded = {
        **manifest,
        **contract,
        "score_sha256": score_sha,
        "model_sha256": model_sha,
        "seed_signature_sha256": seed_signature,
    }
    if upgraded != manifest:
        _atomic_json(manifest_path, upgraded)
    return scores, float(manifest["ranker_runtime_ms_per_group"])


def _completion_signature_valid(output: Path) -> bool:
    path = output / "completion_signature.json"
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        signature = payload.pop("completion_signature_sha256")
        if signature != _sha256_json(payload):
            return False
        for raw_path, expected in payload["source_sha256"].items():
            source = Path(raw_path)
            if not source.is_file() or _sha256_file(source) != expected:
                return False
        for relative, expected in payload["output_sha256"].items():
            artifact = output / relative
            if not artifact.is_file() or _sha256_file(artifact) != expected:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _list_features_for_prefix(features: pd.DataFrame) -> pd.DataFrame:
    """Exact canonical list-feature equations with batched column assignment.

    The frozen extractor assigns twelve cells through ``DataFrame.loc`` for
    every candidate.  That is mathematically correct but prohibitively slow
    when repeated for a new K prefix.  This implementation keeps its NumPy
    equations and stable native-rank ordering verbatim, then assigns each
    completed column once.
    """

    result = features.copy()
    size = len(result)
    names = (
        "rank_percentile",
        "score_zscore_within_pool",
        "score_percentile_within_pool",
        "delta_to_top1",
        "delta_to_previous",
        "delta_to_next",
        "top1_top2_margin",
        "pool_score_mean",
        "pool_score_std",
        "pool_score_entropy",
        "candidate_count",
        "is_native_top1",
    )
    values = {name: np.empty(size, dtype=float) for name in names}
    for positions in result.groupby("sample_id", sort=False).indices.values():
        index = np.asarray(positions, dtype=int)
        scores = result.iloc[index]["native_score_raw"].to_numpy(float)
        ranks = result.iloc[index]["native_rank"].to_numpy(int)
        order = np.argsort(ranks, kind="stable")
        sorted_scores = scores[order]
        mean, std = float(scores.mean()), float(scores.std())
        softmax = np.exp(scores - scores.max())
        softmax /= softmax.sum()
        entropy = float(-np.sum(softmax * np.log(np.maximum(softmax, 1e-12))))
        for position, local in enumerate(order):
            row_index = index[local]
            rank = position + 1
            percentile = (
                1.0 if len(index) == 1 else (len(index) - rank) / (len(index) - 1)
            )
            values["rank_percentile"][row_index] = percentile
            values["score_zscore_within_pool"][row_index] = (
                (scores[local] - mean) / std if std > 1e-12 else 0.0
            )
            values["score_percentile_within_pool"][row_index] = percentile
            values["delta_to_top1"][row_index] = scores[local] - sorted_scores[0]
            values["delta_to_previous"][row_index] = (
                0.0 if position == 0 else scores[local] - sorted_scores[position - 1]
            )
            values["delta_to_next"][row_index] = (
                0.0
                if position + 1 == len(index)
                else scores[local] - sorted_scores[position + 1]
            )
            values["top1_top2_margin"][row_index] = sorted_scores[0] - (
                sorted_scores[1] if len(index) > 1 else sorted_scores[0]
            )
            values["pool_score_mean"][row_index] = mean
            values["pool_score_std"][row_index] = std
            values["pool_score_entropy"][row_index] = entropy
            values["candidate_count"][row_index] = len(index)
            values["is_native_top1"][row_index] = float(rank == 1)
    for name in names:
        result[name] = values[name]
    return result


def _recompute_pool_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute every pool-relative formal feature after prefix truncation."""

    from unified_reranking.feature_extractors import common as common_features

    work = frame.copy().reset_index(drop=True)
    work = _list_features_for_prefix(work)
    geometry: dict[str, dict[str, Any]] = {}
    for row in work.itertuples(index=False):
        centre = np.asarray([float(row.cx_px), float(row.cy_px)], dtype=float)
        geometry[f"{row.sample_id}\0{row.candidate_id}"] = {
            "center": centre,
            "closing": common_features._axis(float(row.theta_deg))[0],
            "corners": common_features._corners(
                centre,
                float(row.theta_deg),
                float(row.width_px),
                float(row.height_px),
            ),
            "mask_support": float(row.rectangle_probability_mean),
            "depth": float(row.z_center),
            "collision": float(row.finger_sweep_obstacle_max),
        }
    _relations, aggregates = common_features._relations(work, geometry, (480, 640))
    aggregate_columns = [
        "nearest_candidate_distance",
        "nearest_higher_score_distance",
        "number_of_nearby_candidates",
        "candidate_cluster_size",
        "candidate_rank_in_cluster",
        "cluster_best_score",
        "candidate_uniqueness",
        "max_iou_with_other_candidate",
        "mean_iou_with_other_candidate",
        "number_of_overlapping_candidates",
    ]
    work = work.drop(columns=aggregate_columns, errors="ignore").merge(
        aggregates,
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if work[aggregate_columns].isna().any().any():
        raise RuntimeError("pool-relative feature recomputation is incomplete")
    return work


def _load_prefix_features(
    feature_path: Path,
    k: int,
    *,
    derived_cache: Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Load a frozen prefix and optionally cache only its derived K features."""

    source = native_prefix(pd.read_parquet(feature_path), k)
    if k == 5:
        return source
    if resume and derived_cache is not None and derived_cache.is_file():
        derived = pd.read_parquet(derived_cache)
        source_keys = source[["sample_id", "candidate_id", "native_rank"]].reset_index(
            drop=True
        )
        derived_keys = derived[
            ["sample_id", "candidate_id", "native_rank"]
        ].reset_index(drop=True)
        if source_keys.equals(derived_keys):
            _assert_feature_candidate_geometry(source, derived)
            return derived
    derived = _recompute_pool_features(source)
    if derived_cache is not None:
        _atomic_parquet(derived_cache, derived)
    return derived


def _load_ranker_split(
    repo_root: Path,
    route: str,
    split: str,
    k: int,
    *,
    derived_cache: Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    feature_path, label_path = _topk_source_paths(repo_root, route, split)
    _require_files([feature_path, label_path])
    features = _load_prefix_features(
        feature_path, k, derived_cache=derived_cache, resume=resume
    )
    labels = native_prefix(pd.read_parquet(label_path), k)
    required = {"candidate_success"}
    if not required.issubset(labels.columns):
        raise ValueError(f"development labels lack candidate_success: {label_path}")
    joined = features.merge(
        labels[["sample_id", "candidate_id", "candidate_success"]],
        on=["sample_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["candidate_success"].isna().any() or len(joined) != len(labels):
        raise ValueError(f"features/labels do not exactly match for {route}/{split}/K={k}")
    return joined


def _top1_from_scores(scores: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {"sample_id", "candidate_id", "native_rank", score_column}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"scored candidates miss fields: {missing}")
    work = scores.copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="raise")
    if not np.isfinite(work[score_column]).all():
        raise ValueError("ranker scores must be finite")
    return (
        work.sort_values(
            ["sample_id", score_column, "native_rank", "candidate_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates("sample_id")[["sample_id", "candidate_id"]]
        .rename(columns={"candidate_id": "selected_candidate_id"})
    )


def _assert_feature_candidate_geometry(
    candidates: pd.DataFrame, features: pd.DataFrame
) -> None:
    geometry = ("cx_px", "cy_px", "theta_deg", "width_px", "height_px")
    required = {"sample_id", "candidate_id", *geometry}
    for name, frame in (("candidates", candidates), ("features", features)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} cannot prove frozen candidate geometry: {missing}")
    left = candidates[["sample_id", "candidate_id", *geometry]].copy()
    right = features[["sample_id", "candidate_id", *geometry]].copy()
    joined = left.merge(
        right,
        on=["sample_id", "candidate_id"],
        how="outer",
        validate="one_to_one",
        suffixes=("_candidate", "_feature"),
        indicator=True,
    )
    if len(joined) != len(left) or not joined["_merge"].eq("both").all():
        raise RuntimeError("candidate and feature identities differ")
    for column in geometry:
        candidate_value = pd.to_numeric(joined[f"{column}_candidate"], errors="coerce")
        feature_value = pd.to_numeric(joined[f"{column}_feature"], errors="coerce")
        if candidate_value.isna().any() or feature_value.isna().any() or not np.array_equal(
            candidate_value.to_numpy(float), feature_value.to_numpy(float)
        ):
            raise RuntimeError(f"frozen candidate geometry differs in {column}")


def _evaluate_selection(
    denominator: pd.DataFrame, selected: pd.DataFrame, labels: pd.DataFrame
) -> np.ndarray:
    successes = set(
        map(
            tuple,
            labels.loc[labels["candidate_success"].astype(bool), ["sample_id", "candidate_id"]]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )
    return _outcomes_for_decisions(denominator, selected, successes)


def _k1_no_reranking_row(ceiling_row: dict[str, Any]) -> dict[str, Any]:
    """Represent K=1 as a native-only ceiling, never as fake reranking."""

    return {
        **ceiling_row,
        "raw_reranked_num": math.nan,
        "raw_reranked_j1": math.nan,
        "gated_num": math.nan,
        "gated_j1": math.nan,
        "recovered": math.nan,
        "harmful": math.nan,
        "net_recovered": math.nan,
        "headroom_recovery": math.nan,
        "ranker_runtime_ms_per_group": math.nan,
        "seed_mean_j1": math.nan,
        "seed_sd_j1": math.nan,
        "seed_delta_mean_pp": math.nan,
        "seed_delta_sd_pp": math.nan,
        "seed_positive_count": 0,
        "seed_zero_count": 0,
        "seed_negative_count": 0,
        "seed_sign_consistency": False,
        "outcome_changing_precision": math.nan,
        "gate_status": "NOT_APPLICABLE_K1",
    }


def _fit_preprocessor_on_train_only(
    preprocessor_type: Any,
    train: pd.DataFrame,
    columns: Sequence[str],
) -> Any:
    """Fit preprocessing through an API that cannot receive Val or Test rows."""

    return preprocessor_type.fit(train, columns)


def run_topk_sensitivity(
    repo_root: Path, run_dir: Path, *, resume: bool = True
) -> dict[str, Any]:
    """Run native-prefix ceilings and per-K retrained formal LambdaMART models."""

    repo_root, run_dir = Path(repo_root).resolve(), Path(run_dir).resolve()
    _assert_isolated_run_dir(repo_root, run_dir)
    output = run_dir / "4d_topk_sensitivity"
    final_path = output / "ranker_results.csv"
    if resume and final_path.is_file() and _completion_signature_valid(output):
        frame = pd.read_csv(final_path)
        expected = {(route.upper(), k) for route in ROUTES for k in (1, 3, 5)}
        observed = set(zip(frame["route"], frame["k"], strict=False))
        if expected.issubset(observed):
            return {
                "status": "COMPLETE",
                "results": str(final_path),
                "resumed": True,
            }

    denominator = _denominator(repo_root)
    route_sources = _route_sources(repo_root)
    ceiling_rows: list[dict[str, Any]] = []
    ceiling_data: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for route in ROUTES:
        source = route_sources[route]
        all_candidates = native_prefix(pd.read_parquet(source.candidates_all), None)
        requested: list[tuple[str, int | None]] = [("1", 1), ("3", 3), ("5", 5)]
        # Extended D1 has exact frozen formal outcomes for every cached
        # post-NMS candidate.  Other routes' formal labels are Top-5 only, so
        # their K>5 diagnostics fail closed instead of relabelling post hoc.
        if route == "d1":
            requested.extend([("10", 10), ("20", 20), ("all", None)])
        for label, k in requested:
            prefix = native_prefix(all_candidates, k)
            labels = _formal_candidate_labels(repo_root, route, prefix)
            ceiling_rows.append(
                _candidate_ceiling_row(
                    denominator,
                    prefix,
                    labels,
                    route=route,
                    k_label=label,
                    retrospective=source.retrospective,
                )
            )
            if k in (1, 3, 5):
                ceiling_data[(route, int(k))] = (prefix, labels)
    ceiling = pd.DataFrame(ceiling_rows)
    _atomic_csv(output / "candidate_ceiling.csv", ceiling)

    seed_rows: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    ranker_rows: list[dict[str, Any]] = []
    progress_path = output / "progress_manifest.json"
    if resume and progress_path.is_file():
        progress = _read_json(progress_path)
        if not isinstance(progress.get("cells"), dict):
            progress["cells"] = {}
    else:
        progress = {"cells": {}}
    progress["status"] = "IN_PROGRESS"
    _atomic_json(output / "progress_manifest.json", progress)
    source_hash_cache: dict[str, str] = {}

    for route in ROUTES:
        retrospective = route_sources[route].retrospective
        for k in (1, 3, 5):
            ceiling_row = next(
                row
                for row in ceiling_rows
                if row["route"] == route.upper() and row["k"] == str(k)
            )
            prefix, test_labels = ceiling_data[(route, k)]
            native_selection = (
                prefix.loc[prefix["native_rank"].eq(1), ["sample_id", "candidate_id"]]
                .rename(columns={"candidate_id": "selected_candidate_id"})
            )
            native = _evaluate_selection(denominator, native_selection, test_labels)
            if k == 1:
                ranker_rows.append(_k1_no_reranking_row(ceiling_row))
                continue

            cell_dir = output / "models" / route / f"k{k}"
            feature_cache_dir = output / "derived_features" / route / f"k{k}"
            columns = _model_feature_columns(repo_root, route)
            hyperparameters = _ranker_hyperparameters(repo_root, route)
            train = _load_ranker_split(
                repo_root,
                route,
                "train",
                k,
                derived_cache=feature_cache_dir / "train.parquet",
                resume=resume,
            )
            validation = _load_ranker_split(
                repo_root,
                route,
                "validation",
                k,
                derived_cache=feature_cache_dir / "validation.parquet",
                resume=resume,
            )
            test_feature_path, _unused = _topk_source_paths(repo_root, route, "test")
            test_features = _load_prefix_features(
                test_feature_path,
                k,
                derived_cache=feature_cache_dir / "test.parquet",
                resume=resume,
            )
            expected_keys = set(
                map(tuple, prefix[["sample_id", "candidate_id"]].astype(str).itertuples(index=False, name=None))
            )
            observed_keys = set(
                map(
                    tuple,
                    test_features[["sample_id", "candidate_id"]]
                    .astype(str)
                    .itertuples(index=False, name=None),
                )
            )
            if observed_keys != expected_keys:
                raise RuntimeError(f"frozen candidate/feature IDs differ for {route}/K={k}")
            _assert_feature_candidate_geometry(prefix, test_features)
            missing_columns = sorted(set(columns).difference(train.columns))
            if missing_columns:
                raise ValueError(
                    f"locked feature schema is unavailable for {route}/K={k}: {missing_columns}"
                )

            from unified_reranking.datasets import FoldPreprocessor
            from unified_reranking.models import LightGBMLambdaRank

            # The preprocessor is fitted exactly once and solely on Train rows.
            preprocessor = _fit_preprocessor_on_train_only(
                FoldPreprocessor, train, columns
            )
            preprocessor_artifact = preprocessor.artifact()
            _atomic_json(
                cell_dir / "preprocessor.json",
                {
                    **preprocessor_artifact,
                    "fit_split": "train",
                    "fit_sample_count": int(train["sample_id"].nunique()),
                    "validation_used_for_fit": False,
                    "test_used_for_fit": False,
                },
            )
            train_x = preprocessor.transform(train)
            validation_x = preprocessor.transform(validation)
            test_x = preprocessor.transform(test_features)
            score_frame = test_features[["sample_id", "candidate_id", "native_rank"]].copy()
            contract = _topk_cell_contract(
                repo_root,
                route,
                k,
                columns,
                hyperparameters,
                preprocessor_artifact,
                prefix,
                source_hash_cache,
            )
            cell_seed_rows: list[dict[str, Any]] = []
            runtime_values: list[float] = []
            completed_seeds: dict[int, tuple[pd.DataFrame, float]] = {}
            missing_seeds: list[int] = []
            for seed in RANKER_SEEDS:
                completed = (
                    _load_valid_seed_artifact(
                        seed=seed,
                        route=route,
                        k=k,
                        cell_dir=cell_dir,
                        score_frame=score_frame,
                        hyperparameters=hyperparameters,
                        contract=contract,
                    )
                    if resume
                    else None
                )
                if completed is None:
                    missing_seeds.append(seed)
                else:
                    completed_seeds[seed] = completed

            train_labels = train["candidate_success"].astype(int).to_numpy()
            train_groups = train["sample_id"].astype(str).to_numpy()
            validation_labels = (
                validation["candidate_success"].astype(int).to_numpy()
            )
            validation_groups = validation["sample_id"].astype(str).to_numpy()

            def fit_seed(seed: int) -> tuple[int, pd.DataFrame, float]:
                fitted = LightGBMLambdaRank(seed=seed, **hyperparameters).fit(
                    train_x,
                    train_labels,
                    train_groups,
                    eval_set=(
                        validation_x,
                        validation_labels,
                        validation_groups,
                    ),
                )
                started = time.perf_counter_ns()
                values = fitted.predict(test_x)
                elapsed = time.perf_counter_ns() - started
                runtime_ms = elapsed / 1e6 / len(denominator)
                seed_scores = score_frame.copy()
                seed_scores["score"] = values
                seed_path = cell_dir / f"seed_{seed}_scores.parquet"
                seed_manifest_path = cell_dir / f"seed_{seed}.json"
                _atomic_parquet(seed_path, seed_scores)
                model_file = _temporary(cell_dir / f"seed_{seed}_model.txt")
                final_model_file = cell_dir / f"seed_{seed}_model.txt"
                try:
                    fitted.model.booster_.save_model(str(model_file))
                    os.replace(model_file, final_model_file)
                finally:
                    model_file.unlink(missing_ok=True)
                score_sha = _sha256_file(seed_path)
                model_sha = _sha256_file(final_model_file)
                _atomic_json(
                    seed_manifest_path,
                    {
                        "status": "COMPLETE",
                        "seed": seed,
                        "k": k,
                        "route": route.upper(),
                        "hyperparameters": hyperparameters,
                        "preprocessing_fit_split": "train",
                        "ranker_runtime_ms_per_group": runtime_ms,
                        "candidate_rows": int(len(seed_scores)),
                        "candidate_ids_frozen": True,
                        "candidate_geometry_unchanged": True,
                        "independent_seed_training": True,
                        "model_n_jobs": 1,
                        **contract,
                        "score_sha256": score_sha,
                        "model_sha256": model_sha,
                        "seed_signature_sha256": _sha256_json(
                            {
                                "cell_signature_sha256": contract[
                                    "cell_signature_sha256"
                                ],
                                "seed": seed,
                                "score_sha256": score_sha,
                                "model_sha256": model_sha,
                            }
                        ),
                    },
                )
                return seed, seed_scores, runtime_ms

            # Each formal model remains deterministic CPU-only with n_jobs=1;
            # independent pre-registered seeds may train concurrently.
            if missing_seeds:
                with ThreadPoolExecutor(max_workers=len(missing_seeds)) as executor:
                    for seed, seed_scores, runtime_ms in executor.map(
                        fit_seed, missing_seeds
                    ):
                        completed_seeds[seed] = (seed_scores, runtime_ms)

            for seed in RANKER_SEEDS:
                seed_scores, runtime_ms = completed_seeds[seed]
                selected = _top1_from_scores(seed_scores, "score")
                outcome = _evaluate_selection(denominator, selected, test_labels)
                test = _mcnemar(native, outcome)
                seed_row = {
                    "route": route.upper(),
                    "retrospective": retrospective,
                    "k": k,
                    "seed": seed,
                    "n_groups": int(len(denominator)),
                    "native_num": int(native.sum()),
                    "native_j1": float(native.mean()),
                    "reranked_num": int(outcome.sum()),
                    "reranked_j1": float(outcome.mean()),
                    "delta_pp": 100.0 * float(outcome.mean() - native.mean()),
                    **test,
                    "ranker_runtime_ms_per_group": runtime_ms,
                }
                seed_rows.append(seed_row)
                cell_seed_rows.append(seed_row)
                runtime_values.append(runtime_ms)
                score_frame = score_frame.merge(
                    seed_scores[
                        ["sample_id", "candidate_id", "native_rank", "score"]
                    ].rename(
                        columns={"score": f"score_{seed}"}
                    ),
                    on=["sample_id", "candidate_id", "native_rank"],
                    how="left",
                    validate="one_to_one",
                )
            score_columns = [f"score_{seed}" for seed in RANKER_SEEDS]
            if score_frame[score_columns].isna().any().any():
                raise RuntimeError("per-seed ranker scores do not cover frozen candidates")
            score_frame["ensemble_score"] = score_frame[score_columns].mean(axis=1)
            selected = _top1_from_scores(score_frame, "ensemble_score")
            reranked = _evaluate_selection(denominator, selected, test_labels)
            test = _mcnemar(native, reranked)
            bootstrap = _cluster_bootstrap(
                native, reranked, denominator["sequence_id"], seed=BOOTSTRAP_SEED
            )
            oracle = float(ceiling_row["oracle_at_k"])
            native_rate = float(native.mean())
            gain = float(reranked.mean() - native_rate)
            deltas = np.asarray([row["delta_pp"] for row in cell_seed_rows], dtype=float)
            positive_count = int(np.sum(deltas > 0))
            zero_count = int(np.sum(deltas == 0))
            negative_count = int(np.sum(deltas < 0))
            ranker_rows.append(
                {
                    **ceiling_row,
                    "raw_reranked_num": int(reranked.sum()),
                    "raw_reranked_j1": float(reranked.mean()),
                    # A K-specific transition model was not predeclared in the
                    # frozen source.  Secondary gating therefore fails closed
                    # to native and is never tuned on Test outcomes.
                    "gated_num": int(native.sum()),
                    "gated_j1": native_rate,
                    "recovered": test["recovered"],
                    "harmful": test["harmful"],
                    "net_recovered": test["net_recovered"],
                    "outcome_changing_precision": (
                        test["recovered"] / test["discordant"]
                        if test["discordant"]
                        else math.nan
                    ),
                    "headroom_recovery": gain / (oracle - native_rate)
                    if oracle > native_rate
                    else math.nan,
                    "ranker_runtime_ms_per_group": float(np.median(runtime_values)),
                    "seed_mean_j1": float(
                        np.mean([row["reranked_j1"] for row in cell_seed_rows])
                    ),
                    "seed_sd_j1": float(
                        np.std([row["reranked_j1"] for row in cell_seed_rows], ddof=1)
                    ),
                    "seed_delta_mean_pp": float(np.mean(deltas)),
                    "seed_delta_sd_pp": float(np.std(deltas, ddof=1)),
                    "seed_positive_count": positive_count,
                    "seed_zero_count": zero_count,
                    "seed_negative_count": negative_count,
                    "seed_sign_consistency": bool(
                        positive_count == len(deltas) or negative_count == len(deltas)
                    ),
                    "gate_status": "FAIL_CLOSED_NATIVE_NO_K_SPECIFIC_OOF_GATE",
                }
            )
            paired_frames.append(
                pd.DataFrame(
                    {
                        "route": route.upper(),
                        "retrospective": retrospective,
                        "k": k,
                        "sample_id": denominator["sample_id"],
                        "scene_id": denominator["scene_id"],
                        "sequence_id": denominator["sequence_id"],
                        "native_correct": native,
                        "raw_reranked_correct": reranked,
                        "gated_correct": native,
                    }
                )
            )
            bootstrap_rows.append(
                {
                    "route": route.upper(),
                    "retrospective": retrospective,
                    "k": k,
                    "point_estimate_pp": 100.0 * bootstrap["point_estimate"],
                    "ci_low_pp": 100.0 * bootstrap["ci_low"],
                    "ci_high_pp": 100.0 * bootstrap["ci_high"],
                    "iterations": bootstrap["iterations"],
                    "seed": bootstrap["seed"],
                    "sequence_count": bootstrap["cluster_count"],
                    "mcnemar_raw_p": test["raw_p"],
                    "recovered": test["recovered"],
                    "harmful": test["harmful"],
                }
            )
            progress["cells"][f"{route}_k{k}"] = "COMPLETE"
            _atomic_json(output / "progress_manifest.json", progress)

    progress["status"] = "COMPLETE"
    _atomic_csv(final_path, pd.DataFrame(ranker_rows))
    _atomic_csv(output / "seed_results.csv", pd.DataFrame(seed_rows))
    _atomic_csv(output / "bootstrap_results.csv", pd.DataFrame(bootstrap_rows))
    paired = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame(
            columns=[
                "route",
                "retrospective",
                "k",
                "sample_id",
                "sequence_id",
                "native_correct",
                "raw_reranked_correct",
                "gated_correct",
            ]
        )
    )
    _atomic_parquet(output / "paired_outcomes.parquet", paired)
    _atomic_json(output / "progress_manifest.json", progress)
    completion_files = [
        output / "candidate_ceiling.csv",
        final_path,
        output / "seed_results.csv",
        output / "bootstrap_results.csv",
        output / "paired_outcomes.parquet",
        output / "progress_manifest.json",
        *sorted((output / "models").glob("**/*")),
        *sorted((output / "derived_features").glob("**/*")),
    ]
    output_hashes = {
        str(path.relative_to(output)): _sha256_file(path)
        for path in completion_files
        if path.is_file()
    }
    completion: dict[str, Any] = {
        "status": "COMPLETE",
        "expected_cells": [
            f"{route}_k{k}" for route in ROUTES for k in (3, 5)
        ],
        "source_sha256": dict(sorted(source_hash_cache.items())),
        "output_sha256": output_hashes,
    }
    completion["completion_signature_sha256"] = _sha256_json(completion)
    _atomic_json(output / "completion_signature.json", completion)
    return {
        "status": "COMPLETE",
        "results": str(final_path),
        "ranker_cells": int(len(ROUTES) * 2),
        "seed_models": int(len(ROUTES) * 2 * len(RANKER_SEEDS)),
        "gating": "FAIL_CLOSED_NATIVE_NO_K_SPECIFIC_OOF_GATE",
        "resumed": False,
    }


__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "FORMAL_THRESHOLD",
    "RANKER_SEEDS",
    "THRESHOLD_GRID",
    "assert_threshold_monotonicity",
    "load_frozen_canonical_evaluator",
    "native_prefix",
    "run_duplicate_exclusion",
    "run_threshold_sensitivity",
    "run_topk_sensitivity",
    "sequence_id_from_scene",
    "threshold_success",
]
