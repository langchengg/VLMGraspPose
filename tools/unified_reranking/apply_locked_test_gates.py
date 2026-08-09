"""Apply Validation-locked conservative gates to label-free Test evidence.

This stage is intentionally pre-lock safe: it reads candidate geometry, ranker
scores, and inference-time features, but it has no path to candidate outcomes.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tools.unified_reranking.prepare_gate_inputs import FEATURE_COLUMNS
from unified_reranking.gate import (
    ConservativeTransitionModel,
    GateEvidence,
    GateOperatingPoint,
    gate_switch_mask,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log


ROUTES = ("crog", "g1", "c1")


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required immutable input is not a regular file: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _verified_artifact(record: dict[str, Any], name: str) -> Path:
    path = Path(str(record.get("path", "")))
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"locked gate artifact failed hash verification: {name}")
    return path


def _candidate_features(run_dir: Path, route: str) -> pd.DataFrame:
    path = (
        run_dir
        / "03_features"
        / "tracks"
        / "T2_matched_common"
        / f"{route}_test"
        / "candidate_features.parquet"
    )
    columns = [
        "sample_id",
        "candidate_id",
        "calibrated_native_probability",
        "native_score_raw",
        "overall_feature_reliability",
        "peak_retention_rate",
        "perturbed_valid_fraction",
        "mask_reliability",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["perturbation_stability"] = frame[
        ["peak_retention_rate", "perturbed_valid_fraction"]
    ].min(axis=1)
    return frame.drop(columns=["peak_retention_rate", "perturbed_valid_fraction"])


def _prefixed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            "candidate_id": f"{prefix}_candidate_id",
            "calibrated_native_probability": f"{prefix}_calibrated_probability",
            "native_score_raw": f"{prefix}_native_score",
            "overall_feature_reliability": f"{prefix}_overall_reliability",
            "perturbation_stability": f"{prefix}_perturbation_stability",
            "mask_reliability": f"{prefix}_mask_reliability",
        }
    )


def build_label_free_test_gate_inputs(run_dir: Path, route: str) -> pd.DataFrame:
    """Rebuild the exact gate feature schema without consulting outcomes."""

    decisions_path = (
        run_dir
        / "08_lock"
        / "label_free_test_rankers"
        / route
        / "per_sample_decisions.parquet"
    )
    decisions = pd.read_parquet(decisions_path)
    forbidden = {
        "candidate_success",
        "native_correct",
        "challenger_correct",
        "selected_correct",
        "jacquard_margin",
        "first_positive_rank",
    }
    leaked = sorted(forbidden.intersection(decisions.columns))
    if leaked:
        raise PermissionError(f"label-free Test ranker decisions contain outcomes: {leaked}")
    denominator = pd.read_parquet(
        run_dir / "01_manifests" / "paired_test.parquet", columns=["sample_id"]
    )
    denominator["sample_id"] = denominator["sample_id"].astype(str)
    if denominator["sample_id"].duplicated().any():
        raise RuntimeError("Test denominator contains duplicate sample IDs")
    output = denominator.merge(decisions, on="sample_id", how="left", validate="one_to_one")
    if output["candidate_count"].isna().any():
        raise RuntimeError("label-free Test ranker decisions do not preserve denominator")

    features = _candidate_features(run_dir, route)
    output = output.merge(
        _prefixed(features, "native"),
        on=["sample_id", "native_candidate_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        _prefixed(features, "challenger"),
        left_on=["sample_id", "selected_candidate_id"],
        right_on=["sample_id", "challenger_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    candidates = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_test_top5.parquet",
        columns=["sample_id", "candidate_id", "candidate_geometry_sha256"],
    ).rename(
        columns={
            "candidate_id": "challenger_candidate_id",
            "candidate_geometry_sha256": "locked_challenger_geometry_sha256",
        }
    )
    output = output.merge(
        candidates,
        on=["sample_id", "challenger_candidate_id"],
        how="left",
        validate="one_to_one",
    )
    exists = output["challenger_exists"].fillna(False).astype(bool)
    output["candidate_id_unchanged"] = (
        output["challenger_candidate_id"].notna()
        & output["locked_challenger_geometry_sha256"].notna()
    )
    output["geometry_hash_unchanged"] = (
        output["selected_geometry_sha256"].fillna("").astype(str)
        == output["locked_challenger_geometry_sha256"].fillna("").astype(str)
    ) & output["candidate_id_unchanged"]
    scalar = [
        "native_calibrated_probability",
        "native_native_score",
        "native_overall_reliability",
        "native_perturbation_stability",
        "native_mask_reliability",
        "challenger_calibrated_probability",
        "challenger_native_score",
        "challenger_overall_reliability",
        "challenger_perturbation_stability",
        "challenger_mask_reliability",
    ]
    output[scalar] = output[scalar].fillna(0.0)
    output["ranker_score_margin"] = output["ensemble_score_margin"].fillna(0.0)
    output["calibrated_probability_delta"] = (
        output["challenger_calibrated_probability"]
        - output["native_calibrated_probability"]
    )
    output["native_score_delta"] = (
        output["challenger_native_score"] - output["native_native_score"]
    )
    output["overall_reliability_delta"] = (
        output["challenger_overall_reliability"]
        - output["native_overall_reliability"]
    )
    output["perturbation_stability_delta"] = (
        output["challenger_perturbation_stability"]
        - output["native_perturbation_stability"]
    )
    output["mask_reliability_delta"] = (
        output["challenger_mask_reliability"] - output["native_mask_reliability"]
    )
    output["challenger_exists_numeric"] = exists.astype(float)
    output["score_margin"] = output["ranker_score_margin"]
    output["challenger_reliability"] = output[
        "challenger_overall_reliability"
    ].clip(0.0, 1.0)
    output["perturbation_stability"] = output[
        "challenger_perturbation_stability"
    ].clip(0.0, 1.0)
    output["prediction_source"] = "test_label_free"
    output["challenger_candidate_id"] = output["challenger_candidate_id"].fillna("")
    output["native_candidate_id"] = output["native_candidate_id"].fillna("")
    result_columns = [
        "sample_id",
        "prediction_source",
        "native_candidate_id",
        "challenger_candidate_id",
        "score_margin",
        "challenger_reliability",
        "perturbation_stability",
        "seed_challenger_votes",
        "candidate_id_unchanged",
        "geometry_hash_unchanged",
        "challenger_exists",
        *FEATURE_COLUMNS,
    ]
    result = output.loc[:, list(dict.fromkeys(result_columns))].copy()
    if not np.isfinite(result.loc[:, FEATURE_COLUMNS].to_numpy(float)).all():
        raise RuntimeError("label-free Test gate features are not finite")
    return result


def run(run_dir: Path, route: str) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    route = str(route).lower()
    if route not in ROUTES:
        raise ValueError("route must be crog, g1, or c1")
    gate_path = run_dir / "08_lock" / "gates" / route / "gate_selection.json"
    gate_record = _record(gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "COMPLETE" or gate.get("test_access") != "NONE":
        raise RuntimeError("gate is not a completed development-only selection")
    configured_features = tuple(gate["configuration"]["feature_columns"])
    if configured_features != tuple(FEATURE_COLUMNS):
        raise RuntimeError("locked gate feature schema differs from predeclared Test schema")
    model_path = _verified_artifact(gate["artifacts"]["transition_model"], "model")
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    if not isinstance(model, ConservativeTransitionModel):
        raise RuntimeError("locked transition model has an unexpected type")

    inputs = build_label_free_test_gate_inputs(run_dir, route)
    matrix = inputs.loc[:, FEATURE_COLUMNS].to_numpy(float)
    probability_recover, probability_harm = model.predict_probabilities(matrix)
    point_payload = gate["selection"].get("selected_operating_point")
    if point_payload is None:
        switches = np.zeros(len(inputs), dtype=bool)
        utility = np.full(len(inputs), np.nan)
        point = None
    else:
        point = GateOperatingPoint(**point_payload)
        evidence = GateEvidence(
            score_margin=inputs["score_margin"].to_numpy(),
            challenger_reliability=inputs["challenger_reliability"].to_numpy(),
            perturbation_stability=inputs["perturbation_stability"].to_numpy(),
            seed_challenger_votes=inputs["seed_challenger_votes"].to_numpy(),
            candidate_id_unchanged=inputs["candidate_id_unchanged"].to_numpy(),
            geometry_hash_unchanged=inputs["geometry_hash_unchanged"].to_numpy(),
            challenger_exists=inputs["challenger_exists"].to_numpy(),
        )
        switches = gate_switch_mask(probability_recover, probability_harm, evidence, point)
        utility = probability_recover - point.lambda_harm * probability_harm
    native = inputs["native_candidate_id"].astype(str).to_numpy()
    challenger = inputs["challenger_candidate_id"].astype(str).to_numpy()
    decisions = pd.DataFrame(
        {
            "sample_id": inputs["sample_id"].astype(str),
            "prediction_source": "test_label_free",
            "probability_recover": probability_recover,
            "probability_harm": probability_harm,
            "utility": utility,
            "switch": switches,
            "native_candidate_id": native,
            "challenger_candidate_id": challenger,
            "selected_candidate_id": np.where(switches, challenger, native),
        }
    )
    source_paths = {
        "gate_selection": gate_record,
        "ranker_decisions": _record(
            run_dir
            / "08_lock"
            / "label_free_test_rankers"
            / route
            / "per_sample_decisions.parquet"
        ),
        "candidate_features": _record(
            run_dir
            / "03_features"
            / "tracks"
            / "T2_matched_common"
            / f"{route}_test"
            / "candidate_features.parquet"
        ),
        "candidates": _record(
            run_dir / "02_candidates" / f"{route}_test_top5.parquet"
        ),
        "sample_denominator": _record(
            run_dir / "01_manifests" / "paired_test.parquet"
        ),
        "implementation_tool": _record(Path(__file__)),
    }
    signature = canonical_sha256(
        {
            "route": route,
            "feature_columns": list(FEATURE_COLUMNS),
            "operating_point": None if point is None else asdict(point),
            "sources": source_paths,
        }
    )
    output = run_dir / "08_lock" / "label_free_test_gates" / route
    marker = output / "manifest.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("signature_sha256") != signature:
            raise RuntimeError("immutable label-free Test gate output signature mismatch")
        for name, record in previous.get("artifacts", {}).items():
            _verified_artifact(record, name)
        append_access_log(
            run_dir,
            {
                "event": "prelock_label_free_test_stage",
                "stage": "gate_test_application",
                "route": route,
                "output_manifest": str(marker.resolve()),
                "output_manifest_sha256": sha256_file(marker),
                "candidate_labels_opened_as_table": False,
                "resumed": True,
            },
        )
        return previous
    input_path = output / "gate_test_inputs.parquet"
    decision_path = output / "gate_test_decisions.parquet"
    _atomic_parquet(input_path, inputs)
    _atomic_parquet(decision_path, decisions)
    manifest = {
        "status": "COMPLETE",
        "route": route,
        "decision": gate["decision"],
        "signature_sha256": signature,
        "feature_columns": list(FEATURE_COLUMNS),
        "selected_operating_point": None if point is None else asdict(point),
        "candidate_test_labels_read": False,
        "prediction_source": "test_label_free",
        "sources": source_paths,
        "artifacts": {
            "inputs": _record(input_path),
            "decisions": _record(decision_path),
        },
    }
    atomic_json(marker, manifest)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "gate_test_application",
            "route": route,
            "output_manifest": str(marker.resolve()),
            "output_manifest_sha256": sha256_file(marker),
            "candidate_labels_opened_as_table": False,
            "resumed": False,
        },
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--route", choices=ROUTES, action="append")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    routes = tuple(args.route or ROUTES)
    for route in routes:
        with ledger_stage(
            run_dir / "run_ledger.sqlite",
            stage="P11_PRELOCK",
            substage=f"label_free_test_gate_{route}",
            route=route,
            method="calibrated_expected_gain_gate",
            command=" ".join(map(str, sys.argv)),
        ) as state:
            run(run_dir, route)
            marker = run_dir / "08_lock" / "label_free_test_gates" / route / "manifest.json"
            state["artifact_path"] = str(marker)
            state["artifact_sha256"] = sha256_file(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_label_free_test_gate_inputs", "run"]
