"""Run source-locked, Validation-only cumulative and leave-family-out ablations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.ablation import ablation_feature_sets
from unified_reranking.artifacts import (
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.metrics import evaluate_order_only
from unified_reranking.training import FORMAL_SEEDS

from tools.unified_reranking.train_matrix_cell import run as train_matrix_cell


ROUTES = ("crog", "g1", "c1")
_BUDGET_FIELDS = (
    "learning_rate",
    "weight_decay",
    "alpha",
    "temperature",
    "beta",
    "epochs",
    "patience",
    "batch_size",
    "num_leaves",
    "tree_learning_rate",
    "n_estimators",
    "num_attention_blocks",
)
_DEFAULTS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "alpha": 0.5,
    "temperature": 1.0,
    "beta": 1.0,
    "epochs": 100,
    "patience": 10,
    "batch_size": 1024,
    "num_leaves": 31,
    "tree_learning_rate": 0.05,
    "n_estimators": 200,
    "num_attention_blocks": 2,
}


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _selected_cells(
    run_dir: Path, route: str, choice: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, list[tuple[dict[str, Any], Path]]]:
    ensemble_path = verified_artifact_path(
        {
            "path": choice.get("validation_manifest"),
            "sha256": choice.get("validation_manifest_sha256"),
        },
        name=f"{route} selected Validation ensemble",
    )
    ensemble = load_verified_json(
        ensemble_path, name=f"{route} selected Validation ensemble"
    )
    records = ensemble.get("sources", {}).get("matrix_manifests", [])
    if not isinstance(records, list):
        raise ValueError(f"{route} ensemble has no matrix-manifest list")
    cells: list[tuple[dict[str, Any], Path]] = []
    for index, record in enumerate(records):
        path = verified_artifact_path(record, name=f"{route} matrix manifest {index}")
        cell = load_verified_json(path, name=f"{route} Validation matrix cell {index}")
        verify_artifact_records_recursive(
            {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
            name=f"{route} Validation matrix cell {index}",
            require_at_least_one=True,
        )
        configuration = cell.get("configuration", {})
        if (
            configuration.get("route") != route
            or configuration.get("mode") != "validation"
            or configuration.get("track") != choice.get("primary_track")
            or configuration.get("encoder") != choice.get("encoder")
            or configuration.get("loss") != choice.get("loss")
        ):
            raise RuntimeError(f"{route} selected cell does not match locked choice")
        cells.append((cell, path))
    by_seed = {int(cell["configuration"]["seed"]): (cell, path) for cell, path in cells}
    if set(by_seed) != set(FORMAL_SEEDS) or len(cells) != len(FORMAL_SEEDS):
        raise RuntimeError(
            f"{route} selected ensemble must contain exactly the formal seeds"
        )
    ordered = [by_seed[seed] for seed in FORMAL_SEEDS]
    reference = ordered[0][0]
    reference_columns = tuple(map(str, reference.get("feature_columns", ())))
    if not reference_columns:
        raise RuntimeError(f"{route} selected cell has no locked feature schema")
    for cell, _ in ordered[1:]:
        if tuple(map(str, cell.get("feature_columns", ()))) != reference_columns:
            raise RuntimeError(f"{route} selected cells disagree on feature schema")
        left = {
            field: reference["configuration"].get(field, _DEFAULTS[field])
            for field in _BUDGET_FIELDS
        }
        right = {
            field: cell["configuration"].get(field, _DEFAULTS[field])
            for field in _BUDGET_FIELDS
        }
        if left != right:
            raise RuntimeError(f"{route} selected cells disagree on fixed budget")
    return ensemble, ensemble_path, ordered


def _args_from_cell(run_dir: Path, cell: Mapping[str, Any]) -> argparse.Namespace:
    config = cell["configuration"]
    values = {field: config.get(field, _DEFAULTS[field]) for field in _BUDGET_FIELDS}
    return argparse.Namespace(
        run_dir=run_dir,
        route=str(config["route"]),
        track=str(config["track"]),
        encoder=str(config["encoder"]),
        loss=str(config["loss"]),
        seed=int(config["seed"]),
        mode="validation",
        fold=None,
        **values,
    )


def _verified_validation_metrics(
    run_dir: Path, cell: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    configuration = cell.get("configuration", {})
    route = str(configuration.get("route", ""))
    if configuration.get("mode") != "validation" or route not in ROUTES:
        raise RuntimeError(f"ablation source is not a Validation cell: {manifest_path}")
    verify_artifact_records_recursive(
        {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
        name=f"ablation Validation cell {manifest_path}",
        require_at_least_one=True,
    )
    prediction_path = verified_artifact_path(
        cell["artifacts"]["predictions"],
        name=f"ablation Validation predictions {manifest_path}",
    )
    candidates = pd.read_parquet(
        run_dir / "02_candidates" / f"{route}_validation_top5.parquet",
        columns=["sample_id", "candidate_id", "native_rank"],
    )
    labels = pd.read_parquet(
        run_dir
        / "03_features"
        / f"candidate_labels_{route}_validation_top5.parquet",
        columns=["sample_id", "candidate_id", "candidate_success"],
    )
    predictions = pd.read_parquet(prediction_path)
    evaluation = candidates.merge(
        labels, on=["sample_id", "candidate_id"], validate="one_to_one"
    ).merge(
        predictions[["sample_id", "candidate_id", "score"]],
        on=["sample_id", "candidate_id"],
        validate="one_to_one",
    )
    denominator = pd.read_parquet(
        run_dir / "01_manifests" / "paired_validation.parquet",
        columns=["sample_id"],
    )["sample_id"].astype(str).tolist()
    metrics, _ = evaluate_order_only(denominator, evaluation, score_column="score")
    if canonical_sha256(cell.get("metrics")) != canonical_sha256(metrics):
        raise RuntimeError(
            f"ablation cell metrics do not match verified predictions: {manifest_path}"
        )
    return metrics


def _aggregate(
    *,
    run_dir: Path,
    route: str,
    choice: Mapping[str, Any],
    spec: Mapping[str, Any],
    selected_cells: list[tuple[dict[str, Any], Path]],
    produced: list[tuple[dict[str, Any], Path]],
    source_signature: str,
    metric_evaluator: Callable[[Path, Mapping[str, Any], Path], Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = [
        metric_evaluator(run_dir, item, path)
        for item, path in produced
    ]
    j_values = np.asarray([row.get("j_at_1") for row in metrics], dtype=float)
    mrr_values = np.asarray([row.get("mrr_at_5") for row in metrics], dtype=float)
    baseline = np.asarray(
        [
            metric_evaluator(run_dir, cell, path)["j_at_1"]
            for cell, path in selected_cells
        ],
        dtype=float,
    )
    if not (
        np.isfinite(j_values).all()
        and np.isfinite(mrr_values).all()
        and np.isfinite(baseline).all()
    ):
        raise RuntimeError("ablation cells contain non-finite Validation metrics")
    budget = {
        field: selected_cells[0][0]["configuration"].get(field, _DEFAULTS[field])
        for field in _BUDGET_FIELDS
    }
    included = list(map(str, spec["included_features"]))
    return {
        "route": route,
        "track": choice["primary_track"],
        "ablation": f"{spec['ablation_type']}:{spec['family']}",
        "ablation_type": spec["ablation_type"],
        "family": spec["family"],
        "family_order": int(spec["family_order"]),
        "seed_count": len(produced),
        "seeds": json.dumps(list(FORMAL_SEEDS), separators=(",", ":")),
        "selected_encoder": choice["encoder"],
        "selected_loss": choice["loss"],
        "fixed_budget_json": json.dumps(budget, sort_keys=True, separators=(",", ":")),
        "fixed_budget_sha256": canonical_sha256(budget),
        "selected_source_signature_sha256": source_signature,
        "included_families_json": json.dumps(
            spec["included_families"], separators=(",", ":")
        ),
        "included_features_json": json.dumps(included, separators=(",", ":")),
        "included_feature_count": len(included),
        "excluded_feature_count": len(selected_cells[0][0]["feature_columns"])
        - len(included),
        "j_at_1": float(j_values.mean()),
        "j_at_1_std": float(j_values.std(ddof=1)),
        "mrr_at_5": float(mrr_values.mean()),
        "mrr_at_5_std": float(mrr_values.std(ddof=1)),
        "delta_j_at_1": float((j_values - baseline).mean()),
        "cell_manifest_paths_json": json.dumps(
            [str(path) for _, path in produced], separators=(",", ":")
        ),
        "cell_manifest_sha256s_json": json.dumps(
            [sha256_file(path) for _, path in produced], separators=(",", ":")
        ),
        "candidate_test_labels_read": False,
        "split": "validation",
    }


def run(
    run_dir: Path,
    *,
    cell_runner: Callable[..., dict[str, object]] = train_matrix_cell,
    metric_evaluator: Callable[
        [Path, Mapping[str, Any], Path], Mapping[str, Any]
    ] = _verified_validation_metrics,
) -> dict[str, Any]:
    """Execute the Validation-only feature-family analyses and persist evidence."""

    run_dir = run_dir.resolve()
    selection_path = run_dir / "07_validation" / "selected_primary_ungated.json"
    selection = load_verified_json(
        selection_path,
        name="selected primary rankers",
        statuses=("VALIDATION_LOCKED",),
    )
    verify_artifact_records_recursive(selection, name="selected primary rankers")
    selections = selection.get("selections")
    if not isinstance(selections, Mapping) or set(selections) != set(ROUTES):
        raise RuntimeError("selected primary rankers do not cover all routes")
    tool_record = _record(Path(__file__))
    output = run_dir / "07_validation" / "ablations"
    cell_output = output / "cells"
    rows: dict[str, list[dict[str, Any]]] = {
        "cumulative": [],
        "leave_one_family_out": [],
    }
    route_sources: dict[str, Any] = {}
    all_cell_records: list[dict[str, str]] = []
    expected_rows: dict[str, dict[str, int]] = {}
    for route in ROUTES:
        choice = selections[route]
        ensemble, ensemble_path, selected_cells = _selected_cells(
            run_dir, route, choice
        )
        columns = tuple(map(str, selected_cells[0][0]["feature_columns"]))
        cumulative, leave_one = ablation_feature_sets(columns)
        expected_rows[route] = {
            "cumulative": len(cumulative),
            "leave_one_family_out": len(leave_one),
        }
        selected_records = [_record(path) for _, path in selected_cells]
        source_contract = {
            "selection": _record(selection_path),
            "ensemble": _record(ensemble_path),
            "selected_cells": selected_records,
            "feature_columns": list(columns),
            "feature_schema_sha256": canonical_sha256(columns),
            "tool": tool_record,
        }
        source_signature = canonical_sha256(source_contract)
        route_sources[route] = source_contract
        for spec in (*cumulative, *leave_one):
            produced: list[tuple[dict[str, Any], Path]] = []
            for selected_cell, _ in selected_cells:
                contract = {
                    "analysis": "validation_feature_family_ablation",
                    "split": "validation",
                    "ablation_type": spec["ablation_type"],
                    "family": spec["family"],
                    "source_signature_sha256": source_signature,
                    "candidate_test_labels_read": False,
                }
                result = cell_runner(
                    _args_from_cell(run_dir, selected_cell),
                    feature_columns_override=spec["included_features"],
                    output_parent=cell_output,
                    analysis_contract=contract,
                )
                if result.get("status") != "COMPLETE":
                    raise RuntimeError("ablation matrix cell did not complete")
                manifest_path = cell_output / str(result["cell_key"]) / "manifest.json"
                if not manifest_path.is_file():
                    raise RuntimeError("ablation cell did not persist its manifest")
                persisted = load_verified_json(
                    manifest_path, name="persisted ablation matrix cell"
                )
                record = _record(manifest_path)
                all_cell_records.append(record)
                produced.append((persisted, manifest_path.resolve()))
            rows[str(spec["ablation_type"])].append(
                _aggregate(
                    run_dir=run_dir,
                    route=route,
                    choice=choice,
                    spec=spec,
                    selected_cells=selected_cells,
                    produced=produced,
                    source_signature=source_signature,
                    metric_evaluator=metric_evaluator,
                )
            )

    cumulative_path = output / "cumulative_feature_ablation.csv"
    leave_one_path = output / "leave_one_family_out_ablation.csv"
    _atomic_csv(cumulative_path, pd.DataFrame(rows["cumulative"]))
    _atomic_csv(leave_one_path, pd.DataFrame(rows["leave_one_family_out"]))
    result = {
        "status": "COMPLETE",
        "schema_version": 1,
        "analysis": "validation_feature_family_ablation",
        "split": "validation",
        "candidate_test_labels_read": False,
        "selection_sha256": sha256_file(selection_path),
        "expected_rows_by_route": expected_rows,
        "sources": {"routes": route_sources, "tool": tool_record},
        "artifacts": {
            "cumulative": _record(cumulative_path),
            "leave_one_family_out": _record(leave_one_path),
            "cell_manifests": all_cell_records,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(output / "feature_ablation_manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P7_P11",
        substage="validation_feature_family_ablations",
        evidence_track="T2_matched_common",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(run_dir)
        artifact = (
            run_dir / "07_validation" / "ablations" / "feature_ablation_manifest.json"
        )
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
