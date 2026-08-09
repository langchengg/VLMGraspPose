from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.unified_reranking.apply_locked_calibration import run as apply_calibration
from tools.unified_reranking.fit_calibration import run as fit_calibration


def _candidates(sample_ids: list[str]) -> pd.DataFrame:
    rows = []
    for sample_id in sample_ids:
        rows.extend(
            [
                {
                    "sample_id": sample_id,
                    "candidate_id": f"{sample_id}-a",
                    "native_rank": 1,
                    "native_score": 0.9,
                },
                {
                    "sample_id": sample_id,
                    "candidate_id": f"{sample_id}-b",
                    "native_rank": 2,
                    "native_score": 0.1,
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    for directory in ("01_manifests", "02_candidates", "03_features", "04_splits"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    train_ids = [f"train-{index}" for index in range(10)]
    validation_ids = [f"validation-{index}" for index in range(4)]
    test_ids = [f"test-{index}" for index in range(3)]
    for split, ids in (
        ("train", train_ids),
        ("validation", validation_ids),
        ("test", test_ids),
    ):
        candidates = _candidates(ids)
        candidates.to_parquet(
            run / "02_candidates" / f"g1_{split}_top5.parquet", index=False
        )
        if split != "test":
            labels = candidates[["sample_id", "candidate_id"]].copy()
            labels["candidate_success"] = [
                int((row // 2 + row) % 2 == 0) for row in range(len(labels))
            ]
            labels.to_parquet(
                run
                / "03_features"
                / f"candidate_labels_g1_{split}_top5.parquet",
                index=False,
            )
    pd.DataFrame(
        {"sample_id": train_ids, "fold": [index % 5 for index in range(10)]}
    ).to_parquet(run / "04_splits" / "fold_assignments.parquet", index=False)
    pd.DataFrame({"sample_id": validation_ids}).to_parquet(
        run / "01_manifests" / "paired_validation.parquet", index=False
    )
    return run


def test_calibration_resume_verifies_output_hashes(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    first = fit_calibration(run, "g1")
    artifact = Path(first["artifacts"]["train_oof"]["path"])
    mtime = artifact.stat().st_mtime_ns
    second = fit_calibration(run, "g1")
    assert second["signature_sha256"] == first["signature_sha256"]
    assert artifact.stat().st_mtime_ns == mtime
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fit_calibration(run, "g1")


def test_calibration_source_drift_recomputes_prelock_and_refuses_postlock(
    tmp_path: Path,
) -> None:
    run = _build_run(tmp_path)
    first = fit_calibration(run, "g1")
    labels_path = run / "03_features" / "candidate_labels_g1_train_top5.parquet"
    labels = pd.read_parquet(labels_path)
    labels.loc[0, "candidate_success"] = 1 - int(labels.loc[0, "candidate_success"])
    labels.to_parquet(labels_path, index=False)
    second = fit_calibration(run, "g1")
    assert second["signature_sha256"] != first["signature_sha256"]
    (run / "08_lock").mkdir(parents=True)
    (run / "08_lock" / "FORMAL_TEST_LOCK.json").write_text("{}", encoding="utf-8")
    labels.loc[1, "candidate_success"] = 1 - int(labels.loc[1, "candidate_success"])
    labels.to_parquet(labels_path, index=False)
    with pytest.raises(RuntimeError, match="signature drift after formal locking"):
        fit_calibration(run, "g1")


def test_label_free_test_calibration_resume_and_postlock_drift(tmp_path: Path) -> None:
    run = _build_run(tmp_path)
    fit_calibration(run, "g1")
    first = apply_calibration(run, "g1", "test")
    output = Path(first["artifact"]["path"])
    mtime = output.stat().st_mtime_ns
    second = apply_calibration(run, "g1", "test")
    assert second["signature_sha256"] == first["signature_sha256"]
    assert output.stat().st_mtime_ns == mtime
    candidates_path = run / "02_candidates" / "g1_test_top5.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidates.loc[0, "native_score"] = 0.8
    candidates.to_parquet(candidates_path, index=False)
    (run / "08_lock").mkdir(parents=True)
    (run / "08_lock" / "FORMAL_TEST_LOCK.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="signature drift after formal locking"):
        apply_calibration(run, "g1", "test")

