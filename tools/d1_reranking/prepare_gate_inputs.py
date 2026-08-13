"""Build exact Train-OOF and Validation inputs for the D1 expected-gain gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record, load_content_manifest  # noqa: E402
from d1_reranking.gate_inputs import build_gate_input_frame  # noqa: E402
from d1_reranking.io import atomic_parquet  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.artifacts import (  # noqa: E402
    verified_artifact_path,
    verify_artifact_records_recursive,
)
from unified_reranking.gate import SAFE_GATE_FEATURE_COLUMNS  # noqa: E402
from unified_reranking.hashing import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    sha256_file,
)
from unified_reranking.ledger import ledger_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _artifact_frame(manifest: dict[str, object], key: str, name: str) -> pd.DataFrame:
    record = manifest.get("artifacts", {}).get(key, {})  # type: ignore[union-attr]
    path = verified_artifact_path(record, name=name)
    return pd.read_parquet(path)


def _split_inputs(
    root: Path,
    *,
    split: str,
    r0_r1: dict[str, object],
    selection: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    prefix = "train" if split == "train" else "validation"
    selected_prefix = "oof" if split == "train" else "validation"
    native_predictions = _artifact_frame(
        r0_r1, f"r0_{prefix}_predictions", f"D1 R0 {split} predictions"
    )
    native_decisions = _artifact_frame(
        r0_r1, f"r0_{prefix}_decisions", f"D1 R0 {split} decisions"
    )
    challenger_predictions = _artifact_frame(
        selection,
        f"selected_{selected_prefix}_predictions",
        f"D1 selected {split} predictions",
    )
    challenger_decisions = _artifact_frame(
        selection,
        f"selected_{selected_prefix}_decisions",
        f"D1 selected {split} decisions",
    )
    feature_manifest_path = (
        root / "03_features" / split / "top5" / "T2_matched_common" / "manifest.json"
    )
    feature_manifest = load_content_manifest(
        feature_manifest_path, name=f"D1 {split} T2", statuses=("COMPLETE",)
    )
    feature_path = verified_artifact_path(
        feature_manifest.get("artifacts", {}).get("candidate_features", {}),
        name=f"D1 {split} T2 candidate features",
    )
    paired_path = (
        root
        / "01_manifests"
        / (
            "d1_paired_train.parquet"
            if split == "train"
            else "d1_paired_validation.parquet"
        )
    )
    folds_path = root / "04_splits" / "fold_assignments.parquet"
    candidate_manifest_path = (
        root / "02_candidates" / "train" / "manifest.json"
        if split == "train"
        else root / "02_candidates" / "validation" / "manifest.json"
    )
    candidate_manifest = load_content_manifest(
        candidate_manifest_path,
        name=f"D1 {split} candidate manifest",
        statuses=("COMPLETE",),
    )
    candidate_path = verified_artifact_path(
        candidate_manifest.get("artifacts", {}).get("top5", {}),
        name=f"D1 {split} Top5 candidates",
    )
    frame = build_gate_input_frame(
        paired=pd.read_parquet(paired_path, columns=["sample_id", "scene_id"]),
        native_predictions=native_predictions,
        native_decisions=native_decisions,
        challenger_predictions=challenger_predictions,
        challenger_decisions=challenger_decisions,
        candidate_features=pd.read_parquet(feature_path),
        candidates=pd.read_parquet(candidate_path),
        prediction_source="train_oof" if split == "train" else "validation",
        folds=(pd.read_parquet(folds_path) if split == "train" else None),
    )
    return frame, {
        "feature_manifest": artifact_record(feature_manifest_path),
        "features": artifact_record(feature_path),
        "paired": artifact_record(paired_path),
        "candidate_manifest": artifact_record(candidate_manifest_path),
        "candidates": artifact_record(candidate_path),
        "folds": artifact_record(folds_path) if split == "train" else None,
    }


def run(run_dir: Path, *, resume: bool) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    r0_r1_path = root / "07_validation" / "r0_r1_selection.json"
    selection_path = root / "07_validation" / "selected_primary_ungated.json"
    r0_r1 = load_content_manifest(
        r0_r1_path, name="D1 R0/R1 selection", statuses=("COMPLETE",)
    )
    selection = load_content_manifest(
        selection_path, name="D1 selected primary", statuses=("COMPLETE",)
    )
    verify_artifact_records_recursive(
        {"r0_r1": r0_r1, "selection": selection},
        name="D1 gate source selections",
        require_at_least_one=True,
    )
    train, train_sources = _split_inputs(
        root, split="train", r0_r1=r0_r1, selection=selection
    )
    validation, validation_sources = _split_inputs(
        root, split="validation", r0_r1=r0_r1, selection=selection
    )
    sources = {
        "r0_r1_selection": artifact_record(r0_r1_path),
        "primary_selection": artifact_record(selection_path),
        "train": train_sources,
        "validation": validation_sources,
        "builder": artifact_record(ROOT / "src/d1_reranking/gate_inputs.py"),
        "tool": artifact_record(Path(__file__)),
    }
    configuration = {
        "schema_version": 1,
        "route": "D1",
        "pool": "top5",
        "track": "T2_matched_common",
        "feature_columns": list(SAFE_GATE_FEATURE_COLUMNS),
        "train_prediction_source": "train_oof",
        "validation_prediction_source": "validation",
        "candidate_test_labels_read": False,
    }
    signature = canonical_sha256({"configuration": configuration, "sources": sources})
    output = root / "07_validation" / "gate_inputs" / "d1"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = load_content_manifest(
            manifest_path, name="D1 gate inputs", statuses=("COMPLETE",)
        )
        if resume and existing.get("source_signature_sha256") == signature:
            verify_artifact_records_recursive(
                {
                    "sources": existing.get("sources"),
                    "artifacts": existing.get("artifacts"),
                },
                name="D1 gate inputs resume",
                require_at_least_one=True,
            )
            return existing
        raise RuntimeError("D1 gate input manifest differs or is corrupt")
    artifacts = {
        "train_oof": artifact_record(
            atomic_parquet(train, output / "train_oof.parquet")
        ),
        "validation": artifact_record(
            atomic_parquet(validation, output / "validation.parquet")
        ),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_signature_sha256": signature,
        "configuration": configuration,
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
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
    path = root / "07_validation" / "gate_inputs" / "d1" / "manifest.json"
    assert_writable_prelock(root)
    with ledger_stage(
        root / "run_ledger.sqlite",
        stage="P11",
        substage="d1_prepare_gate_inputs",
        route="D1",
        pool="top5",
        evidence_track="T2_matched_common",
        method="R7",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        run(root, resume=args.resume)
        state["artifact_path"] = str(path)
        state["artifact_sha256"] = sha256_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
