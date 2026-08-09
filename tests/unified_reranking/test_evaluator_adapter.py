from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.unified_reranking.evaluator_adapter import _label_one, load_frozen_evaluator
from src.unified_reranking.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523/config/canonical_evaluator.py"


def _grasp(module, **changes):
    values = dict(cx_px=130, cy_px=100, theta_deg=0, jaw_width_px=80, rectangle_height_px=20)
    values.update(changes)
    return module.CanonicalGrasp(**values)


def test_frozen_evaluator_hash_is_enforced() -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_evaluator(EVALUATOR, "0" * 64)


def test_failed_evaluator_import_does_not_poison_module_cache(tmp_path: Path) -> None:
    source = tmp_path / "broken_evaluator.py"
    source.write_text("raise RuntimeError('broken evaluator')\n", encoding="utf-8")
    digest = sha256_file(source)
    module_name = f"_unified_reranking_frozen_evaluator_{digest[:16]}"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="broken evaluator"):
            load_frozen_evaluator(source, digest)
        assert module_name not in sys.modules


def test_same_gt_and_margin_label() -> None:
    module = load_frozen_evaluator(EVALUATOR, sha256_file(EVALUATOR))
    candidate = _grasp(module)
    near_bad_angle = module.corners(_grasp(module, theta_deg=30.1)).tolist()
    far_good_angle = module.corners(_grasp(module, cx_px=400)).tolist()
    result = _label_one(module, candidate, [near_bad_angle, far_good_angle])
    assert result["candidate_success"] is False
    assert -1 <= result["jacquard_margin"] <= 1


def test_angle_boundaries_and_invalid_geometry() -> None:
    module = load_frozen_evaluator(EVALUATOR, sha256_file(EVALUATOR))
    assert module.periodic_angle_error_deg(89, -89) == 2
    assert module.periodic_angle_error_deg(0, 30) == 30
    assert module.periodic_angle_error_deg(0, 30 + 1e-9) > 30
    with pytest.raises(ValueError):
        _grasp(module, jaw_width_px=-1)
