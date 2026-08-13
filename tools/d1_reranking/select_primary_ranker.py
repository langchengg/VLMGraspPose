"""Replay all 360 D1 cells and select the fixed three-seed primary ranker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from d1_reranking.selection import (  # noqa: E402
    ensemble_seed_scores,
    select_validation_winner,
    trial_configuration,
    trial_id,
)
from d1_reranking.plan import load_active_primary_plan  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)
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
    root: Path, split: str
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    label_manifest_path = (
        root / "03_features" / split / "top5" / "labels" / "manifest.json"
    )
    label_manifest = load_content_manifest(
        label_manifest_path, name=f"D1 {split} labels", statuses=("COMPLETE",)
    )
    label_path = verified_artifact_path(
        label_manifest.get("artifact", {}), name=f"D1 {split} labels"
    )
    denominator_path = (
        root
        / "01_manifests"
        / (
            "d1_paired_train.parquet"
            if split == "train"
            else "d1_paired_validation.parquet"
        )
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
            "label_manifest": artifact_record(label_manifest_path),
            "labels": artifact_record(label_path),
            "denominator": artifact_record(denominator_path),
        },
    )


def _cell_predictions(
    manifest_path: Path, *, job_id: str, planned: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_content_manifest(
        manifest_path, name=f"D1 primary cell {job_id}", statuses=("COMPLETE",)
    )
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError(f"D1 primary cell {job_id} configuration is invalid")
    if configuration.get("planned_job_id") != job_id or any(
        configuration.get(key) != value for key, value in planned.items()
    ):
        raise RuntimeError(f"D1 primary cell {job_id} differs from its plan job")
    verify_artifact_records_recursive(
        {"sources": manifest.get("sources"), "artifacts": manifest.get("artifacts")},
        name=f"D1 primary cell {job_id}",
        require_at_least_one=True,
    )
    prediction_path = verified_artifact_path(
        manifest.get("artifacts", {}).get("predictions", {}),
        name=f"D1 primary cell {job_id} predictions",
    )
    return pd.read_parquet(prediction_path), manifest


def _evaluate_ensemble(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    denominator: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    evaluation = predictions.merge(
        labels,
        on=["sample_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(evaluation) != len(predictions) or len(evaluation) != len(labels):
        raise RuntimeError("D1 ensemble/label candidate membership differs")
    return evaluate_order_only(
        denominator, evaluation, score_column="ensemble_score", max_k=5
    )


def run(run_dir: Path, *, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    plan_path, plan = load_active_primary_plan(root)
    execution_path = root / "07_validation" / "primary_matrix_execution.json"
    execution = load_content_manifest(
        execution_path, name="D1 primary matrix execution", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        plan.get("sources"), name="D1 primary plan sources", require_at_least_one=True
    )
    if execution.get("plan") != artifact_record(plan_path):
        raise RuntimeError("D1 primary execution does not bind the current plan")
    jobs = plan.get("jobs")
    outputs = execution.get("outputs")
    if not isinstance(jobs, list) or not isinstance(outputs, dict):
        raise RuntimeError("D1 primary plan/execution inventory is invalid")
    planned_jobs = {
        str(job["job_id"]): job
        for job in jobs
        if isinstance(job, dict) and isinstance(job.get("configuration"), dict)
    }
    if len(planned_jobs) != 360 or set(outputs) != set(planned_jobs):
        raise RuntimeError("D1 primary execution output/job inventory differs")
    train_labels, train_denominator, train_label_sources = _labels_and_denominator(
        root, "train"
    )
    val_labels, val_denominator, val_label_sources = _labels_and_denominator(
        root, "validation"
    )
    output_manifest_path = root / "07_validation" / "selected_primary_ungated.json"
    source_signature = canonical_sha256(
        {
            "plan": artifact_record(plan_path),
            "execution": artifact_record(execution_path),
            "train_labels": train_label_sources,
            "validation_labels": val_label_sources,
            "selector": artifact_record(Path(__file__)),
        }
    )
    if output_manifest_path.exists():
        existing = load_content_manifest(
            output_manifest_path,
            name="D1 selected primary ungated",
            statuses=("COMPLETE",),
        )
        if resume and existing.get("source_signature_sha256") == source_signature:
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name="D1 selected primary ungated",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("D1 selected primary manifest differs or is corrupt")
    groups: dict[str, list[tuple[str, dict[str, Any], Path]]] = {}
    for job_id, job in planned_jobs.items():
        record = outputs[job_id]
        manifest_path = verified_artifact_path(
            record, name=f"D1 primary execution output {job_id}"
        )
        config = job["configuration"]
        groups.setdefault(trial_id(config), []).append((job_id, config, manifest_path))
    if len(groups) != 20:
        raise RuntimeError("D1 primary matrix must contain exactly 20 trials")
    table_rows: list[dict[str, object]] = []
    trial_manifests: dict[str, dict[str, Any]] = {}
    for identifier, members in sorted(groups.items()):
        seed_validation: dict[int, pd.DataFrame] = {}
        seed_oof_parts: dict[int, list[pd.DataFrame]] = {42: [], 123: [], 2026: []}
        cell_records: list[dict[str, str]] = []
        representative = trial_configuration(members[0][1])
        for job_id, configuration, manifest_path in members:
            if trial_configuration(configuration) != representative:
                raise RuntimeError(f"D1 trial {identifier} mixes configurations")
            frame, _manifest = _cell_predictions(
                manifest_path, job_id=job_id, planned=configuration
            )
            seed = int(configuration["seed"])
            if configuration["mode"] == "validation":
                if seed in seed_validation:
                    raise RuntimeError(
                        f"D1 trial {identifier} duplicates Validation seed"
                    )
                seed_validation[seed] = frame
            else:
                seed_oof_parts[seed].append(frame)
            cell_records.append(artifact_record(manifest_path))
        if any(len(parts) != 5 for parts in seed_oof_parts.values()):
            raise RuntimeError(f"D1 trial {identifier} lacks five OOF folds per seed")
        validation = ensemble_seed_scores(seed_validation)
        oof = ensemble_seed_scores(
            {
                seed: pd.concat(parts, ignore_index=True)
                for seed, parts in seed_oof_parts.items()
            }
        )
        val_metrics, val_decisions = _evaluate_ensemble(
            validation, val_labels, val_denominator
        )
        train_metrics, train_decisions = _evaluate_ensemble(
            oof, train_labels, train_denominator
        )
        trial_dir = root / "07_validation" / "primary_selection" / "trials" / identifier
        artifacts = {
            "validation_predictions": artifact_record(
                atomic_parquet(validation, trial_dir / "validation_predictions.parquet")
            ),
            "validation_decisions": artifact_record(
                atomic_parquet(
                    val_decisions, trial_dir / "validation_decisions.parquet"
                )
            ),
            "oof_predictions": artifact_record(
                atomic_parquet(oof, trial_dir / "oof_predictions.parquet")
            ),
            "oof_decisions": artifact_record(
                atomic_parquet(train_decisions, trial_dir / "oof_decisions.parquet")
            ),
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE",
            "trial_id": identifier,
            "configuration": representative,
            "seed_policy": "fixed mean score ensemble over 42,123,2026",
            "validation_metrics": val_metrics,
            "oof_metrics": train_metrics,
            "sources": {
                "cells": sorted(cell_records, key=lambda row: row["path"]),
                "train_labels": train_label_sources,
                "validation_labels": val_label_sources,
            },
            "artifacts": artifacts,
            "candidate_test_labels_read": False,
        }
        manifest["content_sha256"] = canonical_sha256(manifest)
        manifest_path = trial_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        trial_manifests[identifier] = {
            "manifest": artifact_record(manifest_path),
            "payload": manifest,
        }
        table_rows.append(
            {
                "trial_id": identifier,
                "method": representative["method"],
                "configuration_sha256": canonical_sha256(representative),
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                **{f"oof_{key}": value for key, value in train_metrics.items()},
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    method_winners = {}
    for method in ("R2", "R3", "R4", "R5", "R6"):
        eligible = [row for row in table_rows if row["method"] == method]
        method_winners[method] = dict(
            select_validation_winner(eligible, final_tie_column="trial_id")
        )
    primary_row = dict(
        select_validation_winner(
            [method_winners[method] for method in ("R3", "R5", "R6")],
            final_tie_column="method",
        )
    )
    winner_trial = trial_manifests[str(primary_row["trial_id"])]
    table_path = root / "07_validation" / "tables" / "r2_r6_trials.csv"
    atomic_text(table_path, pd.DataFrame(table_rows).to_csv(index=False))
    selected_table_path = (
        root / "07_validation" / "tables" / "selected_primary_ungated.csv"
    )
    atomic_text(
        selected_table_path,
        pd.DataFrame([method_winners[method] for method in method_winners]).to_csv(
            index=False
        ),
    )
    artifacts = {
        "trial_table": artifact_record(table_path),
        "selected_method_table": artifact_record(selected_table_path),
        "selected_trial_manifest": winner_trial["manifest"],
        "selected_validation_predictions": winner_trial["payload"]["artifacts"][
            "validation_predictions"
        ],
        "selected_validation_decisions": winner_trial["payload"]["artifacts"][
            "validation_decisions"
        ],
        "selected_oof_predictions": winner_trial["payload"]["artifacts"][
            "oof_predictions"
        ],
        "selected_oof_decisions": winner_trial["payload"]["artifacts"]["oof_decisions"],
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_signature_sha256": source_signature,
        "selection_rule": (
            "per trial mean scores over seeds 42,123,2026; within method then "
            "R3/R5/R6 by Validation J@1, MRR@5, nDCG@5, stable code"
        ),
        "eligible_primary_methods": ["R3", "R5", "R6"],
        "selected_method": primary_row["method"],
        "selected_trial_id": primary_row["trial_id"],
        "selected_configuration": winner_trial["payload"]["configuration"],
        "validation_metrics": winner_trial["payload"]["validation_metrics"],
        "oof_metrics": winner_trial["payload"]["oof_metrics"],
        "method_winners": method_winners,
        "candidate_test_labels_read": False,
        "sources": {
            "plan": artifact_record(plan_path),
            "execution": artifact_record(execution_path),
            "train_labels": train_label_sources,
            "validation_labels": val_label_sources,
            "selector": artifact_record(Path(__file__)),
        },
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(output_manifest_path, result)
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    path = root / "07_validation" / "selected_primary_ungated.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P9",
        substage="d1_select_primary_ungated",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R3_R5_R6",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
