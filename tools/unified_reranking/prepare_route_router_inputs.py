"""Assemble paired, leakage-safe Train-OOF and Validation route-router inputs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.cross_route_inputs import (
    ALTERNATIVES,
    ROUTES,
    add_router_features,
    apply_gate_probabilities,
    cross_fitted_gate_probabilities,
    gate_operating_point_from_manifest,
    operating_point_audit,
    router_feature_columns,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _load_complete(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE":
        raise RuntimeError(f"incomplete manifest: {path}")
    return value


def discover_gate_manifests(run_dir: Path) -> dict[str, Path]:
    """Find exactly one Validation-locked gate manifest per route."""

    found: dict[str, list[Path]] = {route: [] for route in ROUTES}
    for root in (run_dir / "07_validation", run_dir / "08_lock"):
        if not root.exists():
            continue
        for path in root.rglob("gate_selection.json"):
            value = _load_complete(path)
            route = str(value.get("configuration", {}).get("route", "")).lower()
            if route in found:
                found[route].append(path.resolve())
    invalid = {route: paths for route, paths in found.items() if len(paths) != 1}
    if invalid:
        counts = {route: len(paths) for route, paths in invalid.items()}
        raise RuntimeError(f"expected exactly one locked gate per route: {counts}")
    return {route: paths[0] for route, paths in found.items()}


def _verify_manifest_artifacts(manifest: Mapping[str, Any]) -> None:
    for name, record in dict(manifest.get("artifacts", {})).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"gate artifact hash mismatch: {name}")


def _candidate_catalog(run_dir: Path, route: str, split: str) -> pd.DataFrame:
    candidate_path = run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
    feature_path = (
        run_dir
        / "03_features"
        / "tracks"
        / "T2_matched_common"
        / f"{route}_{split}"
        / "candidate_features.parquet"
    )
    candidates = pd.read_parquet(
        candidate_path,
        columns=[
            "sample_id",
            "candidate_id",
            "cx_px",
            "cy_px",
            "theta_deg",
            "width_px",
            "height_px",
            "candidate_geometry_sha256",
        ],
    )
    features = pd.read_parquet(
        feature_path,
        columns=[
            "sample_id",
            "candidate_id",
            "calibrated_native_probability",
            "overall_feature_reliability",
            "peak_retention_rate",
            "perturbed_valid_fraction",
            "mask_reliability",
        ],
    )
    result = candidates.merge(features, on=["sample_id", "candidate_id"], validate="one_to_one")
    result["stability"] = result[["peak_retention_rate", "perturbed_valid_fraction"]].min(axis=1)
    return result.drop(columns=["peak_retention_rate", "perturbed_valid_fraction"])


def enrich_gated_route(
    gated: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected-candidate observable evidence and frozen geometry."""

    result = gated.copy()
    # The fixed gate schema stores challenger evidence plus challenger-minus-
    # native deltas for these three fields.  Recover the redundant native
    # values exactly at the router boundary instead of widening or rewriting
    # the already locked gate artifacts.
    recoverable_native = {
        "native_overall_reliability": (
            "challenger_overall_reliability",
            "overall_reliability_delta",
        ),
        "native_perturbation_stability": (
            "challenger_perturbation_stability",
            "perturbation_stability_delta",
        ),
        "native_mask_reliability": (
            "challenger_mask_reliability",
            "mask_reliability_delta",
        ),
    }
    for native, (challenger, delta) in recoverable_native.items():
        if challenger not in result or delta not in result:
            continue
        reconstructed = pd.to_numeric(
            result[challenger], errors="coerce"
        ) - pd.to_numeric(result[delta], errors="coerce")
        if native in result:
            declared = pd.to_numeric(result[native], errors="coerce")
            if not np.allclose(
                declared.to_numpy(float),
                reconstructed.to_numpy(float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=False,
            ):
                raise ValueError(f"gated route {native} differs from its locked delta")
        else:
            result[native] = reconstructed

    required = {
        "sample_id",
        "scene_id",
        "prediction_source",
        "native_candidate_id",
        "challenger_candidate_id",
        "gated_candidate_id",
        "gate_switch",
        "gate_probability_recover",
        "gate_probability_harm",
        "gate_utility",
        "score_margin",
        "native_calibrated_probability",
        "challenger_calibrated_probability",
        "native_overall_reliability",
        "challenger_overall_reliability",
        "native_perturbation_stability",
        "challenger_perturbation_stability",
        "native_mask_reliability",
        "challenger_mask_reliability",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"gated route input misses columns: {missing}")
    switches = result["gate_switch"].astype(bool).to_numpy()
    selected_pairs = (
        ("calibrated_probability", "native_calibrated_probability", "challenger_calibrated_probability"),
        ("reliability", "native_overall_reliability", "challenger_overall_reliability"),
        ("stability", "native_perturbation_stability", "challenger_perturbation_stability"),
        ("mask_reliability", "native_mask_reliability", "challenger_mask_reliability"),
    )
    for target, native, challenger in selected_pairs:
        result[f"selected_{target}"] = np.where(
            switches,
            pd.to_numeric(result[challenger], errors="coerce").fillna(0.0),
            pd.to_numeric(result[native], errors="coerce").fillna(0.0),
        )
    result["route_specific_margin"] = np.where(
        switches,
        pd.to_numeric(result["score_margin"], errors="coerce").fillna(0.0),
        np.maximum(
            result["native_calibrated_probability"].fillna(0.0).to_numpy(float)
            - result["challenger_calibrated_probability"].fillna(0.0).to_numpy(float),
            0.0,
        ),
    )
    geometry = catalog.rename(
        columns={
            "candidate_id": "gated_candidate_id",
            **{
                name: f"selected_{name}"
                for name in (
                    "cx_px",
                    "cy_px",
                    "theta_deg",
                    "width_px",
                    "height_px",
                    "candidate_geometry_sha256",
                )
            },
        }
    )[
        [
            "sample_id",
            "gated_candidate_id",
            "selected_cx_px",
            "selected_cy_px",
            "selected_theta_deg",
            "selected_width_px",
            "selected_height_px",
            "selected_candidate_geometry_sha256",
        ]
    ]
    result["gated_candidate_id"] = result["gated_candidate_id"].fillna("").astype(str)
    result = result.merge(
        geometry,
        on=["sample_id", "gated_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    result["selected_candidate_exists"] = (
        result["gated_candidate_id"].ne("")
        & result["selected_candidate_geometry_sha256"].notna()
    )
    result["gate_utility"] = pd.to_numeric(result["gate_utility"], errors="coerce").fillna(0.0)
    keep = [
        "sample_id",
        "scene_id",
        "prediction_source",
        *( ["oof_fold"] if "oof_fold" in result else [] ),
        "gated_candidate_id",
        *( ["gated_correct"] if "gated_correct" in result else [] ),
        "gate_switch",
        "gate_probability_recover",
        "gate_probability_harm",
        "gate_utility",
        "route_specific_margin",
        "selected_calibrated_probability",
        "selected_reliability",
        "selected_stability",
        "selected_mask_reliability",
        "selected_candidate_exists",
        "selected_candidate_geometry_sha256",
        "selected_cx_px",
        "selected_cy_px",
        "selected_theta_deg",
        "selected_width_px",
        "selected_height_px",
    ]
    return result.loc[:, keep]


def _development_gated_route(
    run_dir: Path,
    route: str,
    split: str,
    gate_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[dict[str, object], ...]]:
    input_path = run_dir / "07_validation" / "gate_inputs" / route / (
        "train_oof.parquet" if split == "train" else "validation.parquet"
    )
    frame = pd.read_parquet(input_path)
    point = gate_operating_point_from_manifest(gate_manifest)
    feature_names = tuple(map(str, gate_manifest["configuration"]["feature_columns"]))
    seed = int(gate_manifest["configuration"]["model_seed"])
    if split == "train":
        recover, harm, audits = cross_fitted_gate_probabilities(
            frame, feature_names, seed=seed
        )
        gated = apply_gate_probabilities(frame, recover, harm, point)
    elif split == "validation":
        decision_record = gate_manifest["artifacts"]["validation_decisions"]
        decision_path = Path(decision_record["path"])
        if sha256_file(decision_path) != decision_record["sha256"]:
            raise RuntimeError(f"{route} Validation gate decision hash mismatch")
        locked = frame[["sample_id"]].merge(
            pd.read_parquet(decision_path),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if locked["probability_recover"].isna().any():
            raise RuntimeError(f"{route} locked Validation gate misses paired rows")
        gated = apply_gate_probabilities(
            frame,
            locked["probability_recover"].to_numpy(),
            locked["probability_harm"].to_numpy(),
            point,
        )
        check = gated[["sample_id", "gate_switch", "gated_candidate_id", "gated_correct"]].merge(
            locked[["sample_id", "switch", "selected_candidate_id", "selected_correct"]],
            on="sample_id",
            validate="one_to_one",
        )
        if not (
            np.array_equal(check["gate_switch"].to_numpy(bool), check["switch"].to_numpy(bool))
            and np.array_equal(check["gated_candidate_id"].astype(str), check["selected_candidate_id"].astype(str))
            and np.array_equal(check["gated_correct"].to_numpy(bool), check["selected_correct"].to_numpy(bool))
        ):
            raise RuntimeError(f"{route} locked Validation gate decision is not reproducible")
        audits = ()
    else:
        raise PermissionError("route-router development inputs permit only train/validation")
    return enrich_gated_route(gated, _candidate_catalog(run_dir, route, split)), audits


def _finalize_router_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for route in ROUTES:
        rename[f"{route}_gated_candidate_id"] = f"{route}_candidate_id"
        if f"{route}_gated_correct" in frame:
            rename[f"{route}_gated_correct"] = f"{route}_correct"
    output = frame.rename(columns=rename)
    for route in ALTERNATIVES:
        output[f"{route}_margin"] = output[f"{route}_route_specific_margin"]
        output[f"{route}_reliability"] = output[f"{route}_selected_reliability"]
        output[f"{route}_stability"] = output[f"{route}_selected_stability"]
        output[f"{route}_candidate_exists"] = output[f"{route}_candidate_exists_feature"].astype(bool)
    return output


def _resume(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    value = _load_complete(marker)
    if value.get("signature_sha256") != signature:
        raise RuntimeError("immutable router-input output exists with a different signature")
    for record in value.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("resumable router-input artifact hash mismatch")
    return value


def run(
    run_dir: Path,
    *,
    gate_manifests: Mapping[str, Path] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    gates = dict(gate_manifests or discover_gate_manifests(run_dir))
    if set(gates) != set(ROUTES):
        raise ValueError("gate_manifests must contain crog/g1/c1")
    gate_values: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] = {}
    for route, path in gates.items():
        path = Path(path).resolve()
        value = _load_complete(path)
        if str(value.get("configuration", {}).get("route", "")).lower() != route:
            raise ValueError(f"gate manifest route mismatch: {route}")
        _verify_manifest_artifacts(value)
        gate_values[route] = value
        sources[f"{route}_gate_selection"] = _record(path)
        for split, name in (("train", "train_oof.parquet"), ("validation", "validation.parquet")):
            sources[f"{route}_{split}_gate_input"] = _record(
                run_dir / "07_validation" / "gate_inputs" / route / name
            )
            sources[f"{route}_{split}_candidates"] = _record(
                run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
            )
            sources[f"{route}_{split}_features"] = _record(
                run_dir / "03_features" / "tracks" / "T2_matched_common" / f"{route}_{split}" / "candidate_features.parquet"
            )
    configuration = {
        "outer_gate_cross_fit": True,
        "validation_gates": "validation_locked",
        "default_route": "CROG",
        "route_tie_break": ["G1", "C1"],
        "feature_columns": {
            route.upper(): list(router_feature_columns(route)) for route in ALTERNATIVES
        },
        "test_access": "NONE",
    }
    sources["implementation_tool"] = _record(Path(__file__))
    sources["implementation_primitives"] = _record(
        ROOT / "src" / "unified_reranking" / "cross_route_inputs.py"
    )
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output_dir = (output_dir or run_dir / "07_validation" / "route_router_inputs").resolve()
    marker = output_dir / "manifest.json"
    resumed = _resume(marker, signature)
    if resumed is not None:
        return resumed
    artifacts: dict[str, Any] = {}
    outer_audits: dict[str, Any] = {}
    for split in ("train", "validation"):
        by_route: dict[str, pd.DataFrame] = {}
        for route in ROUTES:
            by_route[route], audits = _development_gated_route(
                run_dir, route, split, gate_values[route]
            )
            if audits:
                outer_audits[route] = list(audits)
        paired = _finalize_router_columns(add_router_features(by_route))
        expected_source = "train_oof" if split == "train" else "validation"
        if set(paired["prediction_source"].astype(str)) != {expected_source}:
            raise RuntimeError("router input provenance mismatch")
        path = output_dir / ("train_oof.parquet" if split == "train" else "validation.parquet")
        _atomic_parquet(path, paired)
        artifacts[expected_source] = _record(path)
    manifest = {
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": configuration,
        "sources": sources,
        "outer_gate_cross_fit_audit": outer_audits,
        "validation_gate_operating_points": {
            route: operating_point_audit(gate_operating_point_from_manifest(gate_values[route]))
            for route in ROUTES
        },
        "artifacts": artifacts,
        "test_access": "NONE",
    }
    atomic_json(marker, manifest)
    return manifest


def _parse_gate(values: list[str]) -> dict[str, Path] | None:
    if not values:
        return None
    result: dict[str, Path] = {}
    for value in values:
        route, separator, path = value.partition("=")
        if not separator or route.lower() not in ROUTES:
            raise ValueError("--gate must have form crog|g1|c1=/path/gate_selection.json")
        result[route.lower()] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--gate", action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P9",
        substage="prepare_route_router_inputs",
        route="cross_route",
        method="outer_cross_fitted_locked_gates",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(
            run_dir,
            gate_manifests=_parse_gate(args.gate),
            output_dir=args.output_dir,
        )
        marker = Path(args.output_dir or run_dir / "07_validation" / "route_router_inputs") / "manifest.json"
        state["artifact_path"] = str(marker.resolve())
        state["artifact_sha256"] = sha256_file(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "discover_gate_manifests",
    "enrich_gated_route",
    "run",
]
