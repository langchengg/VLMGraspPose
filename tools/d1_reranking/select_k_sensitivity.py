"""Ensemble all K cells and publish the non-primary Validation comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.k_replay import (  # noqa: E402
    validate_k_execution_results,
    validate_k_selection,
)
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from d1_reranking.selection import ensemble_seed_scores  # noqa: E402
from unified_reranking.artifacts import verified_artifact_path  # noqa: E402
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.metrics import evaluate_order_only  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _labels_and_denominator(
    root: Path,
    *,
    split: str,
    pool: str,
    plan: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    label_record = plan["sources"]["development_label_manifests"][pool][split]
    label_manifest_path = verified_artifact_path(
        label_record, name=f"D1 K {split}/{pool} label manifest"
    )
    label_manifest = load_content_manifest(
        label_manifest_path,
        name=f"D1 K {split}/{pool} labels",
        statuses=("COMPLETE",),
    )
    label_path = verified_artifact_path(
        label_manifest["artifact"], name=f"D1 K {split}/{pool} labels"
    )
    denominator_record = plan["sources"]["denominators"][split]
    denominator_path = verified_artifact_path(
        denominator_record, name=f"D1 K {split} denominator"
    )
    labels = pd.read_parquet(
        label_path, columns=["sample_id", "candidate_id", "candidate_success"]
    )
    denominator = (
        pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
        .astype(str)
        .tolist()
    )
    return (
        labels,
        denominator,
        {
            "manifest": label_record,
            "labels": artifact_record(label_path),
            "denominator": denominator_record,
        },
    )


def _evaluate(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    denominator: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    evaluation = predictions.merge(
        labels,
        on=["sample_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(evaluation) != len(predictions) or len(evaluation) != len(labels):
        raise RuntimeError("D1 K ensemble/label candidate membership differs")
    return evaluate_order_only(
        denominator, evaluation, score_column="ensemble_score", max_k=5
    )


def _validation_order(scenarios: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        scenarios,
        key=lambda scenario_id: (
            -float(scenarios[scenario_id]["validation_metrics"]["j_at_1"]),
            -float(scenarios[scenario_id]["validation_metrics"]["mrr_at_5"]),
            -float(scenarios[scenario_id]["validation_metrics"]["ndcg_at_5"]),
            scenario_id,
        ),
    )


def run(run_dir: Path, *, resume: bool) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    execution_records, values = validate_k_execution_results(root)
    plan = values["plan"]
    planned = values["planned"]
    results = values["results"]
    output_path = root / "11_k_sensitivity/selection_manifest.json"
    selected_primary_path = root / "07_validation/selected_primary_ungated.json"
    selected_primary = load_content_manifest(
        selected_primary_path,
        name="D1 selected primary for K comparison",
        statuses=("COMPLETE",),
    )
    source_signature = canonical_sha256(
        {
            **execution_records,
            "selected_primary": artifact_record(selected_primary_path),
            "selector": artifact_record(Path(__file__)),
        }
    )
    if output_path.exists():
        existing = load_content_manifest(
            output_path, name="D1 K selection", statuses=("COMPLETE",)
        )
        if not resume:
            raise FileExistsError(f"D1 K selection already exists: {output_path}")
        if existing.get("source_signature_sha256") != source_signature:
            raise RuntimeError("D1 K selection resume source closure differs")
        validate_k_selection(
            root,
            execution_records=execution_records,
            execution_values=values,
        )
        return existing

    scenarios: dict[str, dict[str, Any]] = {}
    label_sources: dict[str, Any] = {}
    for definition in plan["scenario_definitions"]:
        scenario_id = str(definition["scenario_id"])
        pool = str(definition["pool"])
        members = [
            (job_id, job, results[job_id])
            for job_id, job in planned.items()
            if job["configuration"]["scenario_id"] == scenario_id
        ]
        if len(members) != 18:
            raise RuntimeError(f"D1 K {scenario_id} requires exactly 18 cells")
        validation_by_seed: dict[int, pd.DataFrame] = {}
        oof_by_seed: dict[int, list[tuple[int, pd.DataFrame]]] = {
            42: [],
            123: [],
            2026: [],
        }
        cell_job_ids: list[str] = []
        for job_id, job, result in members:
            configuration = job["configuration"]
            prediction_path = verified_artifact_path(
                result["artifacts"]["predictions"],
                name=f"D1 K {scenario_id} cell {job_id} predictions",
            )
            frame = pd.read_parquet(prediction_path)
            seed = int(configuration["seed"])
            if configuration["mode"] == "validation":
                if seed in validation_by_seed:
                    raise RuntimeError(f"D1 K {scenario_id} duplicates Validation seed")
                validation_by_seed[seed] = frame
            else:
                oof_by_seed[seed].append((int(configuration["held_fold"]), frame))
            cell_job_ids.append(job_id)
        if set(validation_by_seed) != {42, 123, 2026} or any(
            sorted(fold for fold, _frame in parts) != list(range(5))
            for parts in oof_by_seed.values()
        ):
            raise RuntimeError(f"D1 K {scenario_id} seed/fold inventory differs")
        validation = ensemble_seed_scores(validation_by_seed)
        oof = ensemble_seed_scores(
            {
                seed: pd.concat(
                    [frame for _fold, frame in sorted(parts)], ignore_index=True
                )
                for seed, parts in oof_by_seed.items()
            }
        )
        train_labels, train_denominator, train_sources = _labels_and_denominator(
            root, split="train", pool=pool, plan=plan
        )
        val_labels, val_denominator, val_sources = _labels_and_denominator(
            root, split="validation", pool=pool, plan=plan
        )
        label_sources[pool] = {
            "train": train_sources,
            "validation": val_sources,
        }
        validation_metrics, validation_decisions = _evaluate(
            validation, val_labels, val_denominator
        )
        oof_metrics, oof_decisions = _evaluate(oof, train_labels, train_denominator)
        scenario_dir = root / "11_k_sensitivity/selection" / scenario_id
        artifacts = {
            "validation_predictions": artifact_record(
                atomic_parquet(
                    validation, scenario_dir / "validation_predictions.parquet"
                )
            ),
            "validation_decisions": artifact_record(
                atomic_parquet(
                    validation_decisions,
                    scenario_dir / "validation_decisions.parquet",
                )
            ),
            "oof_predictions": artifact_record(
                atomic_parquet(oof, scenario_dir / "oof_predictions.parquet")
            ),
            "oof_decisions": artifact_record(
                atomic_parquet(oof_decisions, scenario_dir / "oof_decisions.parquet")
            ),
        }
        scenarios[scenario_id] = {
            "definition": definition,
            "method": plan["selected_primary"]["method"],
            "selected_primary_trial_id": plan["selected_primary"]["trial_id"],
            "seeds": [42, 123, 2026],
            "cell_job_ids": sorted(cell_job_ids),
            "validation_metrics": validation_metrics,
            "oof_metrics": oof_metrics,
            "artifacts": artifacts,
        }

    comparison_rows: list[dict[str, Any]] = [
        {
            "scenario_id": "top5_primary",
            "pool": "top5",
            "track": "T2_matched_common",
            "max_candidates": 5,
            "role": "frozen_primary",
            "method": selected_primary["selected_method"],
            **{
                f"validation_{key}": selected_primary["validation_metrics"][key]
                for key in ("j_at_1", "mrr_at_5", "ndcg_at_5")
            },
            **{
                f"oof_{key}": selected_primary["oof_metrics"][key]
                for key in ("j_at_1", "mrr_at_5", "ndcg_at_5")
            },
        }
    ]
    for scenario_id, scenario in scenarios.items():
        definition = scenario["definition"]
        comparison_rows.append(
            {
                "scenario_id": scenario_id,
                "pool": definition["pool"],
                "track": definition["track"],
                "max_candidates": definition["max_candidates"],
                "role": "sensitivity_only",
                "method": scenario["method"],
                **{
                    f"validation_{key}": scenario["validation_metrics"][key]
                    for key in ("j_at_1", "mrr_at_5", "ndcg_at_5")
                },
                **{
                    f"oof_{key}": scenario["oof_metrics"][key]
                    for key in ("j_at_1", "mrr_at_5", "ndcg_at_5")
                },
            }
        )
    comparison_path = root / "11_k_sensitivity/k_comparison.csv"
    atomic_text(comparison_path, pd.DataFrame(comparison_rows).to_csv(index=False))
    sources = {
        "plan": execution_records["plan"],
        "execution_pointer": execution_records["execution_pointer"],
        "execution_authority": execution_records["execution_authority"],
        "complete_event": execution_records["complete_event"],
        "cells": execution_records["results"],
        "selected_primary": artifact_record(selected_primary_path),
        "development_labels": label_sources,
        "selector": artifact_record(Path(__file__)),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "analysis": "K_sensitivity",
        "source_signature_sha256": source_signature,
        "seed_policy": "fixed mean score ensemble over 42,123,2026; best-seed forbidden",
        "selection_interface": (
            "group exact cells by scenario; concatenate five OOF folds per seed; "
            "mean exact candidate scores across seeds; order sensitivity scenarios "
            "by Validation J@1, MRR@5, nDCG@5, scenario_id"
        ),
        "validation_order": _validation_order(scenarios),
        "primary_replacement_permitted": False,
        "candidate_test_labels_read": False,
        "test_inputs_referenced": False,
        "scenarios": scenarios,
        "sources": sources,
        "artifacts": {"comparison_table": artifact_record(comparison_path)},
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(output_path, result)
    validate_k_selection(
        root,
        execution_records=execution_records,
        execution_values=values,
    )
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    output = root / "11_k_sensitivity/selection_manifest.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P11",
        substage="d1_k_sensitivity_seed_ensemble",
        route="D1",
        pool="top10_allnms",
        evidence_track="T2_T3",
        method="selected_primary_fixed",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(output)
        state["artifact_sha256"] = sha256_file(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
