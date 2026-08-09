"""Fit route-specific grouped OOF calibration and select it on Validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.calibration import grouped_oof_calibration
from unified_reranking.artifacts import verify_artifact_records_recursive
from unified_reranking.hashing import (
    atomic_json,
    atomic_text,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _joined(run_dir: Path, route: str, split: str) -> pd.DataFrame:
    candidate_path = run_dir / "02_candidates" / f"{route}_{split}_top5.parquet"
    label_path = run_dir / "03_features" / f"candidate_labels_{route}_{split}_top5.parquet"
    candidates = pd.read_parquet(candidate_path, columns=["sample_id", "candidate_id", "native_rank", "native_score"])
    labels = pd.read_parquet(label_path, columns=["sample_id", "candidate_id", "candidate_success"])
    joined = candidates.merge(labels, on=["sample_id", "candidate_id"], how="inner", validate="one_to_one")
    if len(joined) != len(candidates) or len(joined) != len(labels):
        raise ValueError(f"candidate/label identity mismatch for {route}/{split}")
    return joined


def _j1_numerator(frame: pd.DataFrame, score: str | None = None) -> int:
    if score is None:
        top = frame.sort_values(["sample_id", "native_rank", "candidate_id"], kind="mergesort")
    else:
        top = frame.sort_values(["sample_id", score, "native_rank", "candidate_id"], ascending=[True, False, True, True], kind="mergesort")
    selected = top.groupby("sample_id", sort=False).head(1)
    return int(selected["candidate_success"].sum())


def _reliability_rows(frame: pd.DataFrame, probability: str, bins: int = 15) -> list[dict[str, Any]]:
    p = frame[probability].to_numpy(float)
    y = frame["candidate_success"].to_numpy(float)
    indexes = np.minimum(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), bins - 1)
    rows = []
    for index in range(bins):
        selected = indexes == index
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": int(selected.sum()),
                "mean_probability": None if not selected.any() else float(p[selected].mean()),
                "observed_frequency": None if not selected.any() else float(y[selected].mean()),
            }
        )
    return rows


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _formal_lock_exists(run_dir: Path) -> bool:
    return (run_dir / "08_lock" / "FORMAL_TEST_LOCK.json").is_file() or (
        run_dir / "09_formal_test" / "FORMAL_TEST_EXECUTION.json"
    ).is_file()


def run(run_dir: Path, route: str) -> dict[str, Any]:
    route = str(route).lower()
    if route not in {"crog", "g1", "c1"}:
        raise ValueError("route must be crog, g1, or c1")
    output = run_dir / "05_calibration"
    manifest_path = output / f"{route}_calibration_manifest.json"
    train_candidate_path = run_dir / "02_candidates" / f"{route}_train_top5.parquet"
    validation_candidate_path = (
        run_dir / "02_candidates" / f"{route}_validation_top5.parquet"
    )
    train_label_path = (
        run_dir / "03_features" / f"candidate_labels_{route}_train_top5.parquet"
    )
    validation_label_path = (
        run_dir
        / "03_features"
        / f"candidate_labels_{route}_validation_top5.parquet"
    )
    folds_path = run_dir / "04_splits" / "fold_assignments.parquet"
    denominator_path = run_dir / "01_manifests" / "paired_validation.parquet"
    sources = {
        "train_candidates": _record(train_candidate_path),
        "train_labels": _record(train_label_path),
        "validation_candidates": _record(validation_candidate_path),
        "validation_labels": _record(validation_label_path),
        "fold_assignments": _record(folds_path),
        "validation_denominator": _record(denominator_path),
        "tool": _record(Path(__file__)),
    }
    configuration = {
        "route": route,
        "methods": ["platt", "isotonic"],
        "probability_clip": [1e-4, 1 - 1e-4],
        "reliability_bins": 15,
    }
    signature = canonical_sha256(
        {"configuration": configuration, "sources": sources}
    )
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "COMPLETE"
            and previous.get("signature_sha256") == signature
        ):
            verify_artifact_records_recursive(
                previous.get("sources", {}),
                name=f"{route} calibration resumable sources",
                require_at_least_one=True,
            )
            verify_artifact_records_recursive(
                previous.get("artifacts", {}),
                name=f"{route} calibration resumable outputs",
                require_at_least_one=True,
            )
            return previous
        if _formal_lock_exists(run_dir):
            raise RuntimeError(
                f"{route} calibration signature drift after formal locking"
            )
    train = _joined(run_dir, route, "train")
    validation = _joined(run_dir, route, "validation")
    folds = pd.read_parquet(folds_path)
    oof, val, serialized, metadata = grouped_oof_calibration(train, folds, validation)
    native_numerator = _j1_numerator(validation)
    calibrated_numerator = _j1_numerator(val, "calibrated_native_probability")
    if native_numerator != calibrated_numerator:
        raise RuntimeError(
            f"calibration changed Validation baseline: native={native_numerator}, calibrated={calibrated_numerator}"
        )
    oof_path = output / f"{route}_train_oof.parquet"
    validation_path = output / f"{route}_validation.parquet"
    reliability_path = output / f"{route}_validation_reliability.parquet"
    report_path = output / f"{route}_CALIBRATION_AUDIT.md"
    _atomic_parquet(oof_path, oof)
    _atomic_parquet(validation_path, val)
    reliability = []
    for method in ("platt", "isotonic"):
        for row in _reliability_rows(val, f"calibrated_probability_{method}"):
            reliability.append({"method": method, **row})
    _atomic_parquet(reliability_path, pd.DataFrame(reliability))
    report = f"""# {route.upper()} calibration audit

- Selected method: **{metadata['selected_method']}**.
- Selection rule: {metadata['selection_rule']}.
- Validation metrics: `{metadata['validation_metrics']}`.
- Native/calibrated Validation J@1 numerator: {native_numerator}/{calibrated_numerator} (identical).
- Candidate ordering: monotone calibration with native-rank secondary tie-break; verified invariant.
- Probability clip: [1e-4, 1-1e-4].
- Fitting scope: grouped Train OOF for OOF predictions; full Train for Validation transformation.
"""
    atomic_text(report_path, report)
    artifact = {
        "status": "COMPLETE",
        "route": route.upper(),
        "configuration": configuration,
        "signature_sha256": signature,
        **metadata,
        "validation_native_j1_numerator": native_numerator,
        "validation_calibrated_j1_numerator": calibrated_numerator,
        "validation_denominator": int(
            pd.read_parquet(denominator_path, columns=["sample_id"]).shape[0]
        ),
        "calibrators": serialized,
        "fold_assignments_sha256": sources["fold_assignments"]["sha256"],
        "train_candidates_sha256": sources["train_candidates"]["sha256"],
        "validation_candidates_sha256": sources["validation_candidates"]["sha256"],
        "train_labels_sha256": sources["train_labels"]["sha256"],
        "validation_labels_sha256": sources["validation_labels"]["sha256"],
        "sources": sources,
        "artifacts": {
            "train_oof": _record(oof_path),
            "validation": _record(validation_path),
            "validation_reliability": _record(reliability_path),
            "audit_report": _record(report_path),
        },
    }
    atomic_json(manifest_path, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--route", required=True, choices=("crog", "g1", "c1"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P6",
        substage=f"calibration_{args.route}",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(run_dir, args.route)
        artifact = run_dir / "05_calibration" / f"{args.route}_calibration_manifest.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
