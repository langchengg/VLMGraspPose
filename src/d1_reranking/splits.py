"""Grouped-fold and split-overlap audit for the D1 extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from unified_reranking.hashing import atomic_json, atomic_text, canonical_sha256

from .candidates import artifact_record
from .io import atomic_copy


def validate_fold_assignments(
    assignments: pd.DataFrame, train_manifest: pd.DataFrame
) -> dict[str, Any]:
    required_assignments = {"sample_id", "scene_id", "rgbd_pair_sha256", "fold"}
    required_train = {"sample_id", "scene_id", "rgbd_pair_sha256"}
    missing_assignments = sorted(required_assignments.difference(assignments.columns))
    missing_train = sorted(required_train.difference(train_manifest.columns))
    if missing_assignments or missing_train:
        raise ValueError(
            f"fold/train schema mismatch: assignments={missing_assignments} train={missing_train}"
        )
    if assignments["sample_id"].astype(str).duplicated().any():
        raise ValueError("fold assignments contain duplicate samples")
    expected = set(train_manifest["sample_id"].astype(str))
    actual = set(assignments["sample_id"].astype(str))
    if expected != actual:
        raise ValueError(
            f"fold assignments differ from D1 Train universe: missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    folds = set(pd.to_numeric(assignments["fold"], errors="raise").astype(int))
    if folds != set(range(5)):
        raise ValueError(f"expected grouped folds 0..4, observed {sorted(folds)}")
    grouped = assignments.groupby("rgbd_pair_sha256")["fold"].nunique()
    if bool((grouped != 1).any()):
        raise ValueError("the same RGB-D frame crosses grouped OOF folds")
    joined = assignments.merge(
        train_manifest[["sample_id", "scene_id", "rgbd_pair_sha256"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("_assignment", "_manifest"),
    )
    for column in ("scene_id", "rgbd_pair_sha256"):
        if (
            not joined[f"{column}_assignment"]
            .astype(str)
            .equals(joined[f"{column}_manifest"].astype(str))
        ):
            raise ValueError(f"fold assignments differ from Train manifest: {column}")
    return {
        "samples": len(assignments),
        "folds": 5,
        "groups": int(assignments["rgbd_pair_sha256"].nunique()),
        "fold_counts": {
            str(key): int(value)
            for key, value in assignments["fold"].value_counts().sort_index().items()
        },
    }


def _overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column].astype(str)).intersection(right[column].astype(str)))


def build_split_audit(
    *,
    run_dir: str | Path,
    unified_run: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    unified = Path(unified_run).resolve()
    paired = {
        "train": pd.read_parquet(run / "01_manifests" / "d1_paired_train.parquet"),
        "validation": pd.read_parquet(
            run / "01_manifests" / "d1_paired_validation.parquet"
        ),
        "test": pd.read_parquet(run / "01_manifests" / "d1_paired_manifest.parquet"),
    }
    source_folds = unified / "04_splits" / "fold_assignments.parquet"
    destination = run / "04_splits" / "fold_assignments.parquet"
    atomic_copy(source_folds, destination)
    assignments = pd.read_parquet(destination)
    fold_summary = validate_fold_assignments(assignments, paired["train"])
    overlaps = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        name = f"{left}_vs_{right}"
        overlaps[name] = {
            "sample_id": _overlap(paired[left], paired[right], "sample_id"),
            "rgbd_pair_sha256": _overlap(
                paired[left], paired[right], "rgbd_pair_sha256"
            ),
            "source_rgb_sha256": _overlap(
                paired[left], paired[right], "source_rgb_sha256"
            ),
            "scene_id": _overlap(paired[left], paired[right], "scene_id"),
        }
    if any(value["sample_id"] for value in overlaps.values()):
        raise RuntimeError("D1 official splits overlap by sample_id")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_WITH_SEQUENCE_OVERLAP_LIMITATION",
        "candidate_test_labels_read": False,
        "fold_assignments": artifact_record(destination),
        "source_fold_assignments": artifact_record(source_folds),
        "paired_manifests": {
            split: {
                "rows": len(frame),
                "sample_identity_sha256": canonical_sha256(
                    sorted(frame["sample_id"].astype(str))
                ),
            }
            for split, frame in paired.items()
        },
        "fold_summary": fold_summary,
        "split_overlaps": overlaps,
        "claim_scope": "locked tuple benchmark; not unseen-sequence generalisation",
    }
    payload["content_sha256"] = canonical_sha256(payload)
    atomic_json(run / "04_splits" / "split_leakage_audit.json", payload)
    table = pd.DataFrame(
        [
            {"split_pair": pair, "identity": identity, "overlap": count}
            for pair, values in overlaps.items()
            for identity, count in values.items()
        ]
    )
    table.to_csv(run / "04_splits" / "split_overlap_counts.csv", index=False)
    report = """# Split leakage audit

- Fold source: byte-identical copy of the completed unified experiment.
- Train OOF: five folds grouped by `rgbd_pair_sha256`; no frame crosses folds.
- Official Train/Validation/Test sample IDs are disjoint.
- Test candidate labels opened: no.

Scene and capture-family overlap is reported rather than hidden. These results
support a locked tuple benchmark; they do not establish unseen-sequence
generalisation.
"""
    atomic_text(run / "04_splits" / "SPLIT_LEAKAGE_AUDIT.md", report)
    atomic_text(
        run / "04_splits" / "SEQUENCE_OVERLAP_LIMITATION.md",
        "# Sequence overlap limitation\n\n"
        "The benchmark contains related capture scenes/sequences across official "
        "splits. Exact sample IDs and RGB-D frames are audited separately. Claims "
        "are restricted to the locked benchmark tuple universe.\n",
    )
    return payload
