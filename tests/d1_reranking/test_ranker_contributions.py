from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from d1_reranking import ranker_contributions
from unified_reranking.hashing import canonical_sha256, sha256_file


class _FakeBooster:
    def num_feature(self) -> int:
        return 2

    def predict(self, matrix: np.ndarray, pred_contrib: bool = False) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        if pred_contrib:
            return np.column_stack((values, np.full(len(values), 0.25)))
        return values.sum(axis=1) + 0.25


class _FakeModel:
    def __init__(self) -> None:
        self.model = type("Native", (), {"booster_": _FakeBooster()})()

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.booster_.predict(matrix)


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def test_native_contributions_are_additive_and_long_form() -> None:
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    values = ranker_contributions._native_contribution_matrix(_FakeModel(), matrix)
    keys = pd.DataFrame(
        {"sample_id": ["s0", "s1"], "candidate_id": ["a", "b"]}
    )
    frame = ranker_contributions._long_contribution_frame(
        keys, values, ("feature_a", "feature_b")
    )
    assert list(frame.columns) == [
        "sample_id",
        "candidate_id",
        "feature_name",
        "contribution",
    ]
    assert len(frame) == 6
    assert set(frame["feature_name"]) == {
        "feature_a",
        "feature_b",
        ranker_contributions.BIAS_FEATURE_NAME,
    }
    observed = frame.groupby(["sample_id", "candidate_id"])["contribution"].sum()
    assert observed.tolist() == pytest.approx([3.25, 7.25])


def test_validate_ranker_contributions_rejects_tampered_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / ranker_contributions.ARTIFACT_RELATIVE_PATH
    artifact.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": ["s0"],
            "candidate_id": ["a"],
            "feature_name": [ranker_contributions.BIAS_FEATURE_NAME],
            "contribution": [0.25],
        }
    ).to_parquet(artifact, index=False)
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    sources = {"synthetic": _record(source)}
    value: dict[str, object] = {
        "schema_version": 1,
        "status": "COMPLETE",
        "selected_method": "R5",
        "seeds": list(ranker_contributions.SEEDS),
        "candidate_test_labels_read": False,
        "sources": sources,
        "source_signature_sha256": canonical_sha256(sources),
        "artifacts": {"candidate_contributions": _record(artifact)},
    }
    value["content_sha256"] = canonical_sha256(value)
    manifest = tmp_path / ranker_contributions.MANIFEST_RELATIVE_PATH
    manifest.write_text(json.dumps(value), encoding="utf-8")
    ranker_contributions.validate_ranker_contributions(tmp_path)

    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        ranker_contributions.validate_ranker_contributions(tmp_path)
