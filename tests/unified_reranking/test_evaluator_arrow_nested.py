from __future__ import annotations

from pathlib import Path

import numpy as np

from src.unified_reranking.evaluator_adapter import _label_one, load_frozen_evaluator
from src.unified_reranking.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523/config/canonical_evaluator.py"


def test_arrow_style_nested_object_arrays_are_accepted() -> None:
    module = load_frozen_evaluator(EVALUATOR, sha256_file(EVALUATOR))
    candidate = module.CanonicalGrasp(130, 100, 0, 80, 20)
    rectangle = np.asarray(
        [np.asarray([90.0, 90.0]), np.asarray([90.0, 110.0]), np.asarray([170.0, 110.0]), np.asarray([170.0, 90.0])],
        dtype=object,
    )
    result = _label_one(module, candidate, np.asarray([rectangle], dtype=object))
    assert result["candidate_success"] is True
