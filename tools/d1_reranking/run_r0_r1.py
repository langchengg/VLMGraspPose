"""Evaluate D1 R0 and select the predeclared R1 rule on Validation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.candidates import artifact_record  # noqa: E402
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.plan import R1_GRID, load_active_primary_plan  # noqa: E402
from d1_reranking.rules import r1_score  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    load_verified_json,
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402
from unified_reranking.metrics import evaluate_order_only  # noqa: E402
from unified_reranking.telemetry import (  # noqa: E402
    flatten_telemetry,
    missing_feature_rate,
    telemetry_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _content_manifest(
    path: Path, *, name: str, statuses: tuple[str, ...] = ("COMPLETE",)
) -> dict[str, object]:
    value = load_verified_json(path, name=name, statuses=statuses)
    unsigned = dict(value)
    expected = unsigned.pop("content_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise RuntimeError(f"{name} content hash mismatch")
    return value


def _load_split(
    root: Path, split: str
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    feature_manifest_path = (
        root / "03_features" / split / "top5" / "T2_matched_common" / "manifest.json"
    )
    label_manifest_path = (
        root / "03_features" / split / "top5" / "labels" / "manifest.json"
    )
    feature_manifest = _content_manifest(
        feature_manifest_path, name=f"D1 {split} final T2"
    )
    extraction_latency = float(
        feature_manifest.get("feature_extraction_latency_ms", -1)
    )
    if extraction_latency < 0:
        raise RuntimeError(f"D1 {split} final T2 extraction latency is absent")
    label_manifest = _content_manifest(label_manifest_path, name=f"D1 {split} labels")
    candidate_manifest_path = root / "02_candidates" / split / "manifest.json"
    candidate_manifest = _content_manifest(
        candidate_manifest_path, name=f"D1 {split} candidates"
    )
    candidate_configuration = candidate_manifest.get("configuration")
    if not isinstance(candidate_configuration, dict) or (
        candidate_configuration.get("route") != "D1"
        or candidate_configuration.get("split") != split
        or candidate_manifest.get("candidate_test_labels_read") is not False
    ):
        raise RuntimeError(f"D1 {split} candidate manifest semantics differ")
    verify_artifact_records_recursive(
        candidate_manifest.get("artifacts"),
        name=f"D1 {split} candidate artifacts",
        require_at_least_one=True,
    )
    candidate_record = candidate_manifest.get("artifacts", {}).get("top5", {})
    candidate_path = verified_artifact_path(
        candidate_record, name=f"D1 {split} Top5 candidates"
    )
    candidate_hashes_path = verified_artifact_path(
        candidate_manifest.get("artifacts", {}).get("candidate_hashes", {}),
        name=f"D1 {split} candidate hashes",
    )
    candidate_manifest_record = artifact_record(candidate_manifest_path)
    candidates_record = artifact_record(candidate_path)
    candidate_hashes_record = artifact_record(candidate_hashes_path)
    feature_configuration = feature_manifest.get("configuration")
    feature_sources = feature_manifest.get("sources")
    label_sources = label_manifest.get("sources")
    calibration_manifest_path = (
        root / "05_calibration" / "top5" / "calibration_manifest.json"
    )
    _content_manifest(calibration_manifest_path, name="D1 Top5 calibration")
    if (
        not isinstance(feature_configuration, dict)
        or feature_configuration.get("route") != "D1"
        or feature_configuration.get("split") != split
        or feature_configuration.get("pool") != "top5"
        or feature_configuration.get("track") != "T2_matched_common"
        or not isinstance(feature_sources, dict)
        or feature_sources.get("candidate_manifest") != candidate_manifest_record
        or feature_sources.get("candidates") != candidates_record
        or feature_sources.get("candidate_hashes") != candidate_hashes_record
        or feature_sources.get("calibration_manifest")
        != artifact_record(calibration_manifest_path)
    ):
        raise RuntimeError(f"D1 {split} final T2 source binding differs")
    if (
        label_manifest.get("split") != split
        or label_manifest.get("pool") != "top5"
        or not isinstance(label_sources, dict)
        or label_sources.get("candidate_manifest") != candidate_manifest_record
        or label_sources.get("candidates") != candidates_record
        or label_sources.get("candidate_hashes") != candidate_hashes_record
    ):
        raise RuntimeError(f"D1 {split} label candidate binding differs")
    verify_artifact_records_recursive(
        {"sources": feature_sources, "artifacts": feature_manifest.get("artifacts")},
        name=f"D1 {split} final T2",
        require_at_least_one=True,
    )
    verify_artifact_records_recursive(
        {"sources": label_sources, "artifact": label_manifest.get("artifact")},
        name=f"D1 {split} labels",
        require_at_least_one=True,
    )
    feature_path = verified_artifact_path(
        feature_manifest.get("artifacts", {}).get("candidate_features", {}),
        name=f"D1 {split} T2 features",
    )
    label_path = verified_artifact_path(
        label_manifest.get("artifact", {}), name=f"D1 {split} labels"
    )
    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)
    candidates = pd.read_parquet(
        candidate_path,
        columns=[
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
        ],
    ).rename(columns={"native_rank": "frozen_native_rank"})
    joined = features.merge(
        labels[["sample_id", "candidate_id", "candidate_success"]],
        on=["sample_id", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    joined = joined.merge(
        candidates, on=["sample_id", "candidate_id"], how="inner", validate="one_to_one"
    )
    if (
        len(joined) != len(features)
        or len(joined) != len(labels)
        or len(joined) != len(candidates)
    ):
        raise ValueError(f"D1 {split} R0/R1 feature-label membership differs")
    if (
        not joined["native_rank"]
        .astype(int)
        .equals(joined["frozen_native_rank"].astype(int))
    ):
        raise ValueError(f"D1 {split} R0/R1 native rank differs from candidates")
    joined = joined.drop(columns="frozen_native_rank")
    denominator_path = (
        root
        / "01_manifests"
        / (
            "d1_paired_train.parquet"
            if split == "train"
            else "d1_paired_validation.parquet"
        )
    )
    denominator = (
        pd.read_parquet(denominator_path, columns=["sample_id"])["sample_id"]
        .astype(str)
        .tolist()
    )
    return (
        joined,
        denominator,
        {
            "feature_manifest": artifact_record(feature_manifest_path),
            "feature_extraction_latency_ms": extraction_latency,
            "features": artifact_record(feature_path),
            "label_manifest": artifact_record(label_manifest_path),
            "labels": artifact_record(label_path),
            "candidate_manifest": artifact_record(candidate_manifest_path),
            "candidate_hashes": candidate_hashes_record,
            "candidates": artifact_record(candidate_path),
            "calibration_manifest": artifact_record(calibration_manifest_path),
            "denominator": artifact_record(denominator_path),
        },
    )


def _evaluate(
    frame: pd.DataFrame, denominator: list[str], score_column: str
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    metrics, decisions = evaluate_order_only(
        denominator, frame, score_column=score_column, max_k=5
    )
    predictions = frame[
        [
            "sample_id",
            "candidate_id",
            "native_rank",
            "candidate_identity_sha256",
            "candidate_geometry_sha256",
            score_column,
        ]
    ].rename(columns={score_column: "score"})
    return metrics, decisions, predictions


def run(run_dir: Path, *, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    plan_path, plan = load_active_primary_plan(root)
    verify_artifact_records_recursive(
        plan.get("sources"),
        name="D1 primary matrix plan sources",
        require_at_least_one=True,
    )
    planned_trials = (
        plan.get("method_registry", {}).get("R1", {}).get("trials", [])
        if isinstance(plan.get("method_registry"), dict)
        else []
    )
    if planned_trials != list(R1_GRID):
        raise RuntimeError("D1 R1 trials differ from the frozen primary plan")
    feature_started = time.perf_counter()
    train, train_denominator, train_sources = _load_split(root, "train")
    validation, validation_denominator, validation_sources = _load_split(
        root, "validation"
    )
    processed_rows = len(train) + len(validation)
    feature_latency_ms = (
        (time.perf_counter() - feature_started) * 1000.0 / processed_rows
    )
    sources = {
        "plan": artifact_record(plan_path),
        "train": train_sources,
        "validation": validation_sources,
        "rule": artifact_record(ROOT / "src/d1_reranking/rules.py"),
        "tool": artifact_record(Path(__file__)),
    }
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "pool": "top5",
        "track": "T2_matched_common",
        "r0_score": "native_score_raw",
        "r1_trials": list(R1_GRID),
        "selection": "Validation J@1 desc, MRR@5 desc, nDCG@5 desc, trial name asc",
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    manifest_path = root / "07_validation" / "r0_r1_selection.json"
    if manifest_path.is_file():
        existing = _content_manifest(manifest_path, name="D1 R0/R1 selection")
        if (
            resume
            and existing.get("status") == "COMPLETE"
            and existing.get("source_signature_sha256") == signature
        ):
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name="D1 R0/R1 resume",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("D1 R0/R1 selection contract differs")

    r0_train_metrics, r0_train_decisions, r0_train_predictions = _evaluate(
        train, train_denominator, "native_score_raw"
    )
    r0_val_metrics, r0_val_decisions, r0_val_predictions = _evaluate(
        validation, validation_denominator, "native_score_raw"
    )
    trials = []
    trial_payload: dict[str, dict[str, object]] = {}
    for trial in R1_GRID:
        name = str(trial["name"])
        train_trial = train.copy()
        validation_trial = validation.copy()
        train_trial["r1_score"] = r1_score(train_trial, trial)
        validation_trial["r1_score"] = r1_score(validation_trial, trial)
        train_metrics, train_decisions, train_predictions = _evaluate(
            train_trial, train_denominator, "r1_score"
        )
        val_metrics, val_decisions, val_predictions = _evaluate(
            validation_trial, validation_denominator, "r1_score"
        )
        trial_payload[name] = {
            "configuration": dict(trial),
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "train_decisions": train_decisions,
            "validation_decisions": val_decisions,
            "train_predictions": train_predictions,
            "validation_predictions": val_predictions,
        }
        trials.append(
            {
                "trial": name,
                **dict(trial),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in val_metrics.items()},
            }
        )
    winner = min(
        trial_payload,
        key=lambda name: (
            -float(trial_payload[name]["validation_metrics"]["j_at_1"]),  # type: ignore[index]
            -float(trial_payload[name]["validation_metrics"]["mrr_at_5"]),  # type: ignore[index]
            -float(trial_payload[name]["validation_metrics"]["ndcg_at_5"]),  # type: ignore[index]
            name,
        ),
    )
    inference_started = time.perf_counter()
    selected_validation_score = r1_score(
        validation,
        trial_payload[winner]["configuration"],  # type: ignore[arg-type]
    )
    ranker_latency_ms = (
        (time.perf_counter() - inference_started)
        * 1000.0
        / len(selected_validation_score)
    )
    telemetry = telemetry_payload(
        phase="d1_r1_validation_rule",
        parameter_count=0,
        ranker_latency_ms=ranker_latency_ms,
        feature_latency_ms=feature_latency_ms,
        missing_feature_rate_value=missing_feature_rate(
            validation,
            (
                "base_logit",
                "p_center",
                "rectangle_probability_mean",
                "jaw_probability_min",
            ),
        ),
    )
    train_output = root / "06_oof" / "r0_r1"
    val_output = root / "07_validation" / "r0_r1"
    artifacts = {
        "r0_train_predictions": artifact_record(
            atomic_parquet(
                r0_train_predictions, train_output / "r0_predictions.parquet"
            )
        ),
        "r0_train_decisions": artifact_record(
            atomic_parquet(r0_train_decisions, train_output / "r0_decisions.parquet")
        ),
        "r0_validation_predictions": artifact_record(
            atomic_parquet(r0_val_predictions, val_output / "r0_predictions.parquet")
        ),
        "r0_validation_decisions": artifact_record(
            atomic_parquet(r0_val_decisions, val_output / "r0_decisions.parquet")
        ),
        "r1_train_predictions": artifact_record(
            atomic_parquet(
                trial_payload[winner]["train_predictions"],
                train_output / "r1_predictions.parquet",
            )
        ),  # type: ignore[arg-type]
        "r1_train_decisions": artifact_record(
            atomic_parquet(
                trial_payload[winner]["train_decisions"],
                train_output / "r1_decisions.parquet",
            )
        ),  # type: ignore[arg-type]
        "r1_validation_predictions": artifact_record(
            atomic_parquet(
                trial_payload[winner]["validation_predictions"],
                val_output / "r1_predictions.parquet",
            )
        ),  # type: ignore[arg-type]
        "r1_validation_decisions": artifact_record(
            atomic_parquet(
                trial_payload[winner]["validation_decisions"],
                val_output / "r1_decisions.parquet",
            )
        ),  # type: ignore[arg-type]
    }
    trials_path = root / "07_validation" / "tables" / "r1_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(trials_path, index=False)
    artifacts["r1_trials"] = artifact_record(trials_path)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_signature_sha256": signature,
        "configuration": configuration,
        "selected_r1_trial": winner,
        "selected_r1_configuration": trial_payload[winner]["configuration"],
        "r0_train_metrics": r0_train_metrics,
        "r0_validation_metrics": r0_val_metrics,
        "r1_train_metrics": trial_payload[winner]["train_metrics"],
        "r1_validation_metrics": trial_payload[winner]["validation_metrics"],
        "feature_extraction_latency_ms": validation_sources[
            "feature_extraction_latency_ms"
        ],
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "candidate_test_labels_read": False,
        "sources": sources,
        "artifacts": artifacts,
    }
    result["content_sha256"] = canonical_sha256(result)
    atomic_json(manifest_path, result)
    return result


def main() -> int:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    path = root / "07_validation" / "r0_r1_selection.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P9",
        substage="d1_r0_r1_validation",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R0_R1",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
