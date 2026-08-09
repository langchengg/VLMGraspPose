"""Apply the Validation-locked CROG-default router to label-free Test evidence."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for item in (ROOT, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.test_access_guard import append_access_log
from unified_reranking.route_router import (
    DEFAULT_ROUTE_TIE_BREAK,
    RouterEvidence,
    RouterOperatingPoint,
    route_decisions,
    route_utilities,
)


_FORBIDDEN = {
    "candidate_success",
    "jacquard_margin",
    "crog_correct",
    "g1_correct",
    "c1_correct",
    "selected_correct",
    "native_correct",
    "challenger_correct",
}


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "COMPLETE":
        raise RuntimeError(f"incomplete manifest: {path}")
    return value


def _resume(marker: Path, signature: str) -> dict[str, Any] | None:
    if not marker.exists():
        return None
    value = _load(marker)
    if value.get("signature_sha256") != signature:
        raise RuntimeError("immutable Test-router output exists with a different signature")
    for record in value.get("artifacts", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("resumable Test-router artifact hash mismatch")
    return value


def run(
    run_dir: Path,
    *,
    router_selection_path: Path | None = None,
    test_input_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    selection_path = (
        router_selection_path or run_dir / "08_lock" / "route_router" / "route_router_selection.json"
    ).resolve()
    input_path = (
        test_input_path or run_dir / "08_lock" / "route_router_inputs" / "test_label_free.parquet"
    ).resolve()
    selection = _load(selection_path)
    if selection.get("test_access") != "NONE":
        raise PermissionError("router selection is not certified development-only")
    configuration = selection.get("configuration", {})
    if configuration.get("default_route") != "CROG":
        raise ValueError("locked router is not CROG-default")
    if tuple(configuration.get("tie_break", ())) != tuple(DEFAULT_ROUTE_TIE_BREAK):
        raise ValueError("locked router tie order must be exactly G1 then C1")
    input_manifest_path = input_path.parent / "manifest.json"
    input_manifest = _load(input_manifest_path)
    if input_manifest.get("candidate_test_labels_read") is not False:
        raise PermissionError("Test router inputs lack a label-free certificate")
    input_record = input_manifest.get("artifacts", {}).get("test_label_free")
    if not input_record or Path(input_record["path"]).resolve() != input_path:
        raise RuntimeError("Test router input is not the manifest-locked artifact")
    if sha256_file(input_path) != input_record["sha256"]:
        raise RuntimeError("Test router input hash mismatch")
    model_record = selection["artifacts"]["transition_models"]
    model_path = Path(model_record["path"])
    if sha256_file(model_path) != model_record["sha256"]:
        raise RuntimeError("locked route-router model hash mismatch")
    sources = {
        "router_selection": _record(selection_path),
        "router_models": _record(model_path),
        "test_router_inputs": _record(input_path),
        "test_router_input_manifest": _record(input_manifest_path),
        "implementation_tool": _record(Path(__file__)),
    }
    application_configuration = {
        "default_route": "CROG",
        "tie_break": list(DEFAULT_ROUTE_TIE_BREAK),
        "selection_decision": selection["decision"],
        "selected_operating_point": selection["selection"].get("selected_operating_point"),
        "feature_columns": configuration["feature_columns"],
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256(
        {"configuration": application_configuration, "sources": sources}
    )
    output_dir = (output_dir or run_dir / "08_lock" / "route_router_test").resolve()
    marker = output_dir / "manifest.json"
    resumed = _resume(marker, signature)
    if resumed is not None:
        return resumed

    frame = pd.read_parquet(input_path)
    forbidden = sorted(
        column
        for column in frame.columns
        if column in _FORBIDDEN
        or any(
            token in column.lower()
            for token in ("candidate_success", "jacquard_margin", "matched_gt", "ground_truth", "_correct")
        )
    )
    if forbidden:
        raise PermissionError(f"Test router inputs contain supervision: {forbidden}")
    if set(frame["prediction_source"].astype(str)) != {"test_label_free"}:
        raise ValueError("Test router input provenance must be test_label_free")
    with model_path.open("rb") as stream:
        router = pickle.load(stream)
    feature_columns = {
        route: tuple(map(str, configuration["feature_columns"][route]))
        for route in DEFAULT_ROUTE_TIE_BREAK
    }
    probabilities = router.predict_probabilities(
        {route: frame.loc[:, names].to_numpy(float) for route, names in feature_columns.items()}
    )
    evidence = {
        "G1": RouterEvidence(
            route_margin=frame["g1_margin"].to_numpy(),
            reliability=frame["g1_reliability"].to_numpy(),
            perturbation_stability=frame["g1_stability"].to_numpy(),
            candidate_exists=frame["g1_candidate_exists"].to_numpy(),
        ),
        "C1": RouterEvidence(
            route_margin=frame["c1_margin"].to_numpy(),
            reliability=frame["c1_reliability"].to_numpy(),
            perturbation_stability=frame["c1_stability"].to_numpy(),
            candidate_exists=frame["c1_candidate_exists"].to_numpy(),
        ),
    }
    point_value = selection["selection"].get("selected_operating_point")
    if selection["decision"] == "NO_GO_CROG":
        if point_value is not None:
            raise ValueError("NO_GO_CROG router unexpectedly contains an operating point")
        selected_routes = np.full(len(frame), "CROG", dtype=object)
        utilities = {route: np.zeros(len(frame), dtype=float) for route in DEFAULT_ROUTE_TIE_BREAK}
    elif selection["decision"] == "GO" and isinstance(point_value, dict):
        point = RouterOperatingPoint(**point_value)
        selected_routes = route_decisions(
            probabilities,
            evidence,
            point,
            tie_break=DEFAULT_ROUTE_TIE_BREAK,
        )
        utilities = route_utilities(probabilities, lambda_router=point.lambda_router)
    else:
        raise ValueError("router selection has no valid locked decision")
    route_ids = {
        route: frame[f"{route.lower()}_candidate_id"].fillna("").astype(str).to_numpy()
        for route in ("CROG", *DEFAULT_ROUTE_TIE_BREAK)
    }
    route_hashes = {
        route: frame[f"{route.lower()}_selected_candidate_geometry_sha256"].fillna("").astype(str).to_numpy()
        for route in ("CROG", *DEFAULT_ROUTE_TIE_BREAK)
    }
    indexes = np.arange(len(frame))
    selected_ids = np.asarray(
        [route_ids[str(route)][index] for index, route in zip(indexes, selected_routes, strict=True)],
        dtype=object,
    )
    selected_hashes = np.asarray(
        [route_hashes[str(route)][index] for index, route in zip(indexes, selected_routes, strict=True)],
        dtype=object,
    )
    decisions = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str),
            "scene_id": frame["scene_id"].astype(str),
            "prediction_source": "test_label_free",
            "g1_probability_recover": probabilities["G1"][0],
            "g1_probability_harm": probabilities["G1"][1],
            "g1_utility": utilities["G1"],
            "c1_probability_recover": probabilities["C1"][0],
            "c1_probability_harm": probabilities["C1"][1],
            "c1_utility": utilities["C1"],
            "selected_route": selected_routes,
            "switched_from_crog": selected_routes != "CROG",
            "crog_candidate_id": route_ids["CROG"],
            "g1_candidate_id": route_ids["G1"],
            "c1_candidate_id": route_ids["C1"],
            "selected_candidate_id": selected_ids,
            "selected_candidate_geometry_sha256": selected_hashes,
        }
    )
    if _FORBIDDEN.intersection(decisions.columns):
        raise RuntimeError("Test router decisions accidentally contain supervision")
    decision_path = output_dir / "route_router_test_decisions_label_free.parquet"
    _atomic_parquet(decision_path, decisions)
    manifest = {
        "status": "COMPLETE",
        "signature_sha256": signature,
        "configuration": application_configuration,
        "sources": sources,
        "sample_count": int(len(decisions)),
        "route_counts": {
            route: int((selected_routes == route).sum())
            for route in ("CROG", *DEFAULT_ROUTE_TIE_BREAK)
        },
        "artifacts": {"decisions": _record(decision_path)},
        "candidate_test_labels_read": False,
        "test_access": "LABEL_FREE_INFERENCE_ONLY",
    }
    atomic_json(marker, manifest)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "route_router_test_application",
            "output_manifest": str(marker.resolve()),
            "output_manifest_sha256": sha256_file(marker),
            "candidate_labels_opened_as_table": False,
        },
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--router-selection", type=Path)
    parser.add_argument("--test-input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P11_PRELOCK",
        substage="apply_locked_route_router",
        route="cross_route",
        method="crog_default_expected_gain_router",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(
            run_dir,
            router_selection_path=args.router_selection,
            test_input_path=args.test_input,
            output_dir=args.output_dir,
        )
        marker = Path(args.output_dir or run_dir / "08_lock" / "route_router_test") / "manifest.json"
        state["artifact_path"] = str(marker.resolve())
        state["artifact_sha256"] = sha256_file(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
