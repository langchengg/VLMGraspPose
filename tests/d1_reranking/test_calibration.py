from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from d1_reranking.calibration import (
    join_development_calibration,
    reliability_rows,
    top1_success_numerator,
)
from d1_reranking.candidates import artifact_record
from tools.d1_reranking.fit_calibration import _development_table
from unified_reranking.calibration import grouped_oof_calibration
from unified_reranking.hashing import atomic_json, canonical_sha256


def _development() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = []
    labels = []
    folds = []
    for sample_index in range(10):
        sample_id = f"s{sample_index}"
        folds.append({"sample_id": sample_id, "fold": sample_index % 5})
        for rank, score in ((1, 0.8), (2, 0.2)):
            candidates.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": f"c{rank}",
                    "native_rank": rank,
                    "native_score": score,
                }
            )
            labels.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": f"c{rank}",
                    "candidate_success": int(rank == 1),
                }
            )
    return pd.DataFrame(candidates), pd.DataFrame(labels), pd.DataFrame(folds)


def test_grouped_d1_calibration_preserves_native_order() -> None:
    candidates, labels, folds = _development()
    train = join_development_calibration(candidates, labels)
    validation = train.copy()
    oof, transformed, _models, metadata = grouped_oof_calibration(
        train, folds, validation
    )
    assert metadata["selected_method"] in {"platt", "isotonic"}
    assert top1_success_numerator(validation) == top1_success_numerator(
        transformed, "calibrated_native_probability"
    )
    assert oof["base_logit"].notna().all()
    rows = reliability_rows(
        transformed, f"calibrated_probability_{metadata['selected_method']}"
    )
    assert len(rows) == 15
    assert sum(row["count"] for row in rows) == len(transformed)


def test_d1_calibration_rejects_nonexact_or_nonbinary_labels() -> None:
    candidates, labels, _folds = _development()
    with pytest.raises(ValueError, match="membership differs"):
        join_development_calibration(candidates, labels.iloc[:-1])
    labels.loc[0, "candidate_success"] = 2
    with pytest.raises(ValueError, match="binary"):
        join_development_calibration(candidates, labels)


def test_calibration_loader_binds_labels_to_current_candidate_manifest(
    tmp_path: Path,
) -> None:
    closure = tmp_path / "closure.json"
    closure.write_text("closure", encoding="utf-8")
    candidates, labels, _folds = _development()
    split_dir = tmp_path / "02_candidates" / "train"
    split_dir.mkdir(parents=True)
    candidate_path = split_dir / "d1_top5_candidates.parquet"
    candidate_hashes = split_dir / "candidate_hashes.parquet"
    candidates.to_parquet(candidate_path, index=False)
    pd.DataFrame({"sample_id": candidates["sample_id"].unique()}).to_parquet(
        candidate_hashes, index=False
    )
    candidate_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": {
            "route": "D1",
            "split": "train",
            "source_contract": {"source_closure": artifact_record(closure)},
        },
        "artifacts": {
            "top5": artifact_record(candidate_path),
            "candidate_hashes": artifact_record(candidate_hashes),
        },
    }
    candidate_manifest["content_sha256"] = canonical_sha256(candidate_manifest)
    candidate_manifest_path = split_dir / "manifest.json"
    atomic_json(candidate_manifest_path, candidate_manifest)
    label_dir = tmp_path / "03_features" / "train" / "top5" / "labels"
    label_dir.mkdir(parents=True)
    label_path = label_dir / "candidate_labels.parquet"
    labels.to_parquet(label_path, index=False)
    extra_source = tmp_path / "labels_source.parquet"
    extra_source.write_bytes(b"development labels")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# frozen evaluator\n", encoding="utf-8")
    label_manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "split": "train",
        "pool": "top5",
        "candidate_test_labels_read": False,
        "sources": {
            "source_closure": artifact_record(closure),
            "candidate_manifest": artifact_record(candidate_manifest_path),
            "candidate_hashes": artifact_record(candidate_hashes),
            "candidates": artifact_record(candidate_path),
            "sample_labels": artifact_record(extra_source),
            "evaluator": artifact_record(evaluator),
        },
        "artifact": artifact_record(label_path),
    }
    label_manifest["content_sha256"] = canonical_sha256(label_manifest)
    label_manifest_path = label_dir / "manifest.json"
    atomic_json(label_manifest_path, label_manifest)

    observed, _sources = _development_table(tmp_path, "train", "top5", closure)
    assert len(observed) == len(candidates)

    label_manifest["sources"]["candidate_manifest"] = artifact_record(closure)  # type: ignore[index]
    label_manifest.pop("content_sha256")
    label_manifest["content_sha256"] = canonical_sha256(label_manifest)
    atomic_json(label_manifest_path, label_manifest)
    with pytest.raises(RuntimeError, match="source binding differ"):
        _development_table(tmp_path, "train", "top5", closure)
