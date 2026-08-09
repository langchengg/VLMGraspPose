from __future__ import annotations

import pandas as pd

from unified_reranking.development import _reuse_label_artifact
from unified_reranking.hashing import sha256_file


def test_label_resume_requires_hash_evaluator_and_exact_candidate_identity(tmp_path) -> None:
    candidates = tmp_path / "candidates.parquet"
    labels = tmp_path / "labels.parquet"
    keys = pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "candidate_id": ["b:1", "a:1"],
            "native_rank": [1, 1],
        }
    )
    keys.to_parquet(candidates, index=False)
    keys.iloc[::-1].to_parquet(labels, index=False)
    previous = {
        "sha256": sha256_file(labels),
        "evaluator_sha256": "evaluator",
        "candidate_rows": 2,
    }
    reused = _reuse_label_artifact(candidates, labels, previous, "evaluator")
    assert reused is not None
    assert reused["reused_after_identity_verification"] is True
    assert reused["candidate_manifest_sha256"] == sha256_file(candidates)
    assert _reuse_label_artifact(candidates, labels, previous, "wrong") is None

    changed = keys.copy()
    changed.loc[0, "native_rank"] = 2
    changed.to_parquet(candidates, index=False)
    assert _reuse_label_artifact(candidates, labels, previous, "evaluator") is None

