"""Apply locked per-route gates and prepare label-free paired Test router inputs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import pickle
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

from tools.unified_reranking.prepare_gate_inputs import FEATURE_COLUMNS
from tools.unified_reranking.prepare_route_router_inputs import (
    _candidate_catalog,
    _finalize_router_columns,
    _load_complete,
    _record,
    _verify_manifest_artifacts,
    discover_gate_manifests,
    enrich_gated_route,
)
from unified_reranking.cross_route_inputs import (
    ALTERNATIVES,
    ROUTES,
    add_router_features,
    apply_gate_probabilities,
    gate_operating_point_from_manifest,
    router_feature_columns,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log


_TEST_FORBIDDEN_COLUMNS = {
    "candidate_success",
    "jacquard_margin",
    "native_correct",
    "challenger_correct",
    "selected_correct",
    "gated_correct",
    "matched_gt_index",
    "best_same_gt_iou",
    "best_same_gt_angle_error_deg",
}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _assert_label_free(frame: pd.DataFrame, name: str) -> None:
    forbidden = sorted(
        column
        for column in frame.columns
        if column in _TEST_FORBIDDEN_COLUMNS
        or any(
            token in column.lower()
            for token in ("candidate_success", "jacquard_margin", "matched_gt", "ground_truth", "_correct")
        )
    )
    if forbidden:
        raise PermissionError(f"{name} contains forbidden Test supervision: {forbidden}")


def build_label_free_gate_input(
    paired: pd.DataFrame,
    ranker_decisions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct the development gate schema without any Test outcome field."""

    _assert_label_free(paired, "paired Test manifest")
    _assert_label_free(ranker_decisions, "label-free Test ranker decisions")
    output = paired[["sample_id", "scene_id"]].merge(
        ranker_decisions,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if len(output) != len(paired):
        raise RuntimeError("ranker decisions do not preserve the Test denominator")
    required = {
        "native_candidate_id",
        "selected_candidate_id",
        "selected_geometry_sha256",
        "ensemble_score_margin",
        "seed_challenger_votes",
        "challenger_exists",
    }
    missing = sorted(required.difference(output.columns))
    if missing:
        raise ValueError(f"Test ranker decisions miss gate evidence: {missing}")
    output["challenger_candidate_id"] = output["selected_candidate_id"].fillna("").astype(str)
    output["native_candidate_id"] = output["native_candidate_id"].fillna("").astype(str)
    scalar = [
        "calibrated_native_probability",
        "overall_feature_reliability",
        "stability",
        "mask_reliability",
    ]
    native = catalog[["sample_id", "candidate_id", *scalar]].rename(
        columns={
            "candidate_id": "native_candidate_id",
            "calibrated_native_probability": "native_calibrated_probability",
            "overall_feature_reliability": "native_overall_reliability",
            "stability": "native_perturbation_stability",
            "mask_reliability": "native_mask_reliability",
        }
    )
    challenger = catalog[["sample_id", "candidate_id", *scalar, "candidate_geometry_sha256"]].rename(
        columns={
            "candidate_id": "challenger_candidate_id",
            "calibrated_native_probability": "challenger_calibrated_probability",
            "overall_feature_reliability": "challenger_overall_reliability",
            "stability": "challenger_perturbation_stability",
            "mask_reliability": "challenger_mask_reliability",
            "candidate_geometry_sha256": "locked_challenger_geometry_sha256",
        }
    )
    # Raw native score is part of the fixed gate schema.
    # The T2 catalog intentionally keeps the exact calibrated evidence; native
    # score is read from the frozen candidate table by the caller and joined here.
    output = output.merge(
        native,
        on=["sample_id", "native_candidate_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        challenger,
        on=["sample_id", "challenger_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    output["candidate_id_unchanged"] = (
        output["challenger_candidate_id"].ne("")
        & output["locked_challenger_geometry_sha256"].notna()
    )
    output["geometry_hash_unchanged"] = (
        output["selected_geometry_sha256"].fillna("").astype(str)
        == output["locked_challenger_geometry_sha256"].fillna("").astype(str)
    ) & output["candidate_id_unchanged"]
    for column in (
        "native_calibrated_probability",
        "native_overall_reliability",
        "native_perturbation_stability",
        "native_mask_reliability",
        "challenger_calibrated_probability",
        "challenger_overall_reliability",
        "challenger_perturbation_stability",
        "challenger_mask_reliability",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["ranker_score_margin"] = pd.to_numeric(
        output["ensemble_score_margin"], errors="coerce"
    ).fillna(0.0)
    output["calibrated_probability_delta"] = (
        output["challenger_calibrated_probability"]
        - output["native_calibrated_probability"]
    )
    # The only available Test-safe native-score delta is computed from frozen
    # candidate native scores added below.
    output["native_score_delta"] = (
        output["challenger_native_score"] - output["native_native_score"]
        if {"challenger_native_score", "native_native_score"}.issubset(output.columns)
        else 0.0
    )
    output["overall_reliability_delta"] = (
        output["challenger_overall_reliability"] - output["native_overall_reliability"]
    )
    output["perturbation_stability_delta"] = (
        output["challenger_perturbation_stability"] - output["native_perturbation_stability"]
    )
    output["mask_reliability_delta"] = (
        output["challenger_mask_reliability"] - output["native_mask_reliability"]
    )
    output["challenger_exists"] = output["challenger_exists"].fillna(False).astype(bool)
    output["challenger_exists_numeric"] = output["challenger_exists"].astype(float)
    output["score_margin"] = output["ranker_score_margin"]
    output["challenger_reliability"] = output["challenger_overall_reliability"].clip(0.0, 1.0)
    output["perturbation_stability"] = output["challenger_perturbation_stability"].clip(0.0, 1.0)
    output["seed_challenger_votes"] = output["seed_challenger_votes"].fillna(0).astype(int)
    output["prediction_source"] = "test_label_free"
    _assert_label_free(output, "prepared Test gate input")
    if not np.isfinite(output.loc[:, FEATURE_COLUMNS].to_numpy(float)).all():
        raise RuntimeError("Test gate features are not finite")
    return output


def _catalog_with_native_score(run_dir: Path, route: str) -> pd.DataFrame:
    catalog = _candidate_catalog(run_dir, route, "test")
    scores = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_test_top5.parquet",
        columns=["sample_id", "candidate_id", "native_score"],
    ).rename(columns={"native_score": "native_score_raw"})
    return catalog.merge(scores, on=["sample_id", "candidate_id"], validate="one_to_one")


def _add_native_scores(frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    native = catalog[["sample_id", "candidate_id", "native_score_raw"]].rename(
        columns={"candidate_id": "native_candidate_id", "native_score_raw": "native_native_score"}
    )
    challenger = catalog[["sample_id", "candidate_id", "native_score_raw"]].rename(
        columns={"candidate_id": "challenger_candidate_id", "native_score_raw": "challenger_native_score"}
    )
    return frame.merge(native, on=["sample_id", "native_candidate_id"], how="left", validate="one_to_one").merge(
        challenger, on=["sample_id", "challenger_candidate_id"], how="left", validate="one_to_one"
    ).fillna({"native_native_score": 0.0, "challenger_native_score": 0.0})


def _resume(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    value = _load_complete(marker)
    if value.get("signature_sha256") != signature:
        raise RuntimeError("immutable Test router-input output exists with a different signature")
    for record in value.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("resumable Test router-input artifact hash mismatch")
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
    paired_path = run_dir / "01_manifests" / "paired_test.parquet"
    paired = pd.read_parquet(paired_path, columns=["sample_id", "scene_id"])
    sources: dict[str, Any] = {"paired_test_label_free": _record(paired_path)}
    gate_values: dict[str, dict[str, Any]] = {}
    for route in ROUTES:
        gate_path = Path(gates[route]).resolve()
        gate = _load_complete(gate_path)
        _verify_manifest_artifacts(gate)
        if str(gate.get("configuration", {}).get("route", "")).lower() != route:
            raise ValueError(f"gate manifest route mismatch: {route}")
        gate_values[route] = gate
        sources[f"{route}_gate_selection"] = _record(gate_path)
        for name, path in {
            "ranker_manifest": run_dir / "08_lock" / "label_free_test_rankers" / route / "manifest.json",
            "candidates": run_dir / "02_candidates" / f"{route}_test_top5.parquet",
            "features": run_dir / "03_features" / "tracks" / "T2_matched_common" / f"{route}_test" / "candidate_features.parquet",
        }.items():
            sources[f"{route}_{name}"] = _record(path)
    configuration = {
        "label_free_test": True,
        "candidate_test_labels_read": False,
        "default_route": "CROG",
        "route_tie_break": ["G1", "C1"],
        "feature_columns": {
            route.upper(): list(router_feature_columns(route)) for route in ALTERNATIVES
        },
    }
    sources["implementation_tool"] = _record(Path(__file__))
    sources["implementation_primitives"] = _record(
        ROOT / "src" / "unified_reranking" / "cross_route_inputs.py"
    )
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output_dir = (output_dir or run_dir / "08_lock" / "route_router_inputs").resolve()
    marker = output_dir / "manifest.json"
    resumed = _resume(marker, signature)
    if resumed is not None:
        return resumed

    by_route: dict[str, pd.DataFrame] = {}
    artifacts: dict[str, Any] = {}
    for route in ROUTES:
        ranker_manifest_path = run_dir / "08_lock" / "label_free_test_rankers" / route / "manifest.json"
        ranker_manifest = _load_complete(ranker_manifest_path)
        if not ranker_manifest.get("label_free_test_inference") or ranker_manifest.get("candidate_test_labels_read"):
            raise PermissionError(f"{route} Test ranker artifact is not certified label-free")
        decision_record = ranker_manifest["artifacts"]["decisions"]
        decision_path = Path(decision_record["path"])
        if sha256_file(decision_path) != decision_record["sha256"]:
            raise RuntimeError(f"{route} Test ranker decision hash mismatch")
        ranker_decisions = pd.read_parquet(decision_path)
        catalog = _catalog_with_native_score(run_dir, route)
        # Add raw score columns before calculating the fixed gate delta.
        base = paired[["sample_id", "scene_id"]].merge(
            ranker_decisions, on="sample_id", how="left", validate="one_to_one"
        )
        base["challenger_candidate_id"] = base["selected_candidate_id"].fillna("").astype(str)
        base["native_candidate_id"] = base["native_candidate_id"].fillna("").astype(str)
        scored = _add_native_scores(base, catalog)
        gate_input = build_label_free_gate_input(
            paired,
            scored.drop(columns=["scene_id"], errors="ignore"),
            catalog,
        )
        # build_label_free_gate_input preserves the prejoined native-score columns.
        gate_input["native_score_delta"] = (
            gate_input["challenger_native_score"] - gate_input["native_native_score"]
        )
        gate = gate_values[route]
        feature_names = tuple(map(str, gate["configuration"]["feature_columns"]))
        model_record = gate["artifacts"]["transition_model"]
        model_path = Path(model_record["path"])
        if sha256_file(model_path) != model_record["sha256"]:
            raise RuntimeError(f"{route} gate model hash mismatch")
        with model_path.open("rb") as stream:
            model = pickle.load(stream)
        recover, harm = model.predict_probabilities(gate_input.loc[:, feature_names].to_numpy(float))
        gated = apply_gate_probabilities(
            gate_input,
            recover,
            harm,
            gate_operating_point_from_manifest(gate),
        )
        by_route[route] = enrich_gated_route(gated, catalog)
        route_dir = output_dir / "gated_routes" / route
        input_path = route_dir / "gate_input_label_free.parquet"
        decision_path = route_dir / "gated_decisions_label_free.parquet"
        _atomic_parquet(input_path, gate_input)
        _atomic_parquet(decision_path, by_route[route])
        artifacts[f"{route}_gate_input"] = _record(input_path)
        artifacts[f"{route}_gated_decisions"] = _record(decision_path)
    router_input = _finalize_router_columns(add_router_features(by_route))
    _assert_label_free(router_input, "paired Test router input")
    if set(router_input["prediction_source"].astype(str)) != {"test_label_free"}:
        raise RuntimeError("Test router provenance mismatch")
    router_path = output_dir / "test_label_free.parquet"
    _atomic_parquet(router_path, router_input)
    artifacts["test_label_free"] = _record(router_path)
    manifest = {
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": configuration,
        "sources": sources,
        "artifacts": artifacts,
        "candidate_test_labels_read": False,
        "test_access": "LABEL_FREE_FEATURES_ONLY",
    }
    atomic_json(marker, manifest)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "router_test_input_preparation",
            "output_manifest": str(marker.resolve()),
            "output_manifest_sha256": sha256_file(marker),
            "candidate_labels_opened_as_table": False,
        },
    )
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
        stage="P11_PRELOCK",
        substage="label_free_test_gates_and_router_inputs",
        route="cross_route",
        method="locked_gate_application",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(
            run_dir,
            gate_manifests=_parse_gate(args.gate),
            output_dir=args.output_dir,
        )
        marker = Path(args.output_dir or run_dir / "08_lock" / "route_router_inputs") / "manifest.json"
        state["artifact_path"] = str(marker.resolve())
        state["artifact_sha256"] = sha256_file(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_label_free_gate_input", "run"]
