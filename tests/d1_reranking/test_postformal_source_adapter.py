from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from d1_reranking.postformal_source_adapter import NativeProbabilityNumpyAdapter
from d1_reranking.postformal_source_adapter import alias_hifics_probability_assets
from d1_reranking.postformal_source_adapter import safe_load_evaluator
from d1_reranking.postformal_source_adapter import select_renderable_cases
from d1_reranking.postformal_source_adapter import exact_d1_allnms_candidates
from unified_reranking.hashing import sha256_file


def test_safe_loader_registers_dataclass_module(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        """from dataclasses import dataclass
IOU_THRESHOLD = 0.25
ANGLE_THRESHOLD_DEG = 30.0
@dataclass
class Grasp:
    value: float
def gt_from_corners(value): return value
def corners(value): return value
""",
        encoding="utf-8",
    )
    module = safe_load_evaluator(
        {
            "path": str(evaluator.resolve()),
            "sha256": sha256_file(evaluator),
            "bytes": evaluator.stat().st_size,
        }
    )
    assert module.Grasp(1.0).value == 1.0
    assert module.IOU_THRESHOLD == 0.25


def test_safe_loader_rejects_hash_drift(tmp_path: Path) -> None:
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("IOU_THRESHOLD=0.25\n", encoding="utf-8")
    record = {
        "path": str(evaluator.resolve()),
        "sha256": sha256_file(evaluator),
        "bytes": evaluator.stat().st_size,
    }
    evaluator.write_text("IOU_THRESHOLD=0.5\n", encoding="utf-8")
    try:
        safe_load_evaluator(record)
    except RuntimeError as error:
        assert "record differs" in str(error)
    else:
        raise AssertionError("hash drift was accepted")


def test_allnms_adapter_excludes_route_qualified_union_aliases(
    tmp_path: Path,
) -> None:
    allnms_path = tmp_path / "allnms.parquet"
    allnms = pd.DataFrame(
        {
            "route": ["D1"],
            "sample_id": ["sample-0"],
            "candidate_id": ["g0004"],
            "candidate_geometry_sha256": ["a" * 64],
        }
    )
    allnms.to_parquet(allnms_path, index=False)
    outcomes = pd.DataFrame(
        {
            "source_route": ["D1", "D1"],
            "sample_id": ["sample-0", "sample-0"],
            "candidate_id": ["g0004", "D1:g0004"],
            "candidate_geometry_sha256": ["a" * 64, "a" * 64],
            "candidate_success": [True, True],
        }
    )
    formal_value = {
        "outcomes": outcomes,
        "plan": {
            "components": {
                "d1_allnms_candidates": {
                    "path": str(allnms_path.resolve()),
                    "sha256": sha256_file(allnms_path),
                    "bytes": allnms_path.stat().st_size,
                }
            }
        },
    }
    selected = exact_d1_allnms_candidates(formal_value)
    assert selected["candidate_id"].tolist() == ["g0004"]
    assert selected["candidate_success"].tolist() == [True]


def test_allnms_adapter_rejects_missing_raw_candidate(tmp_path: Path) -> None:
    allnms_path = tmp_path / "allnms.parquet"
    pd.DataFrame(
        {
            "route": ["D1"],
            "sample_id": ["sample-0"],
            "candidate_id": ["g0004"],
            "candidate_geometry_sha256": ["a" * 64],
        }
    ).to_parquet(allnms_path, index=False)
    value = {
        "outcomes": pd.DataFrame(
            columns=[
                "source_route",
                "sample_id",
                "candidate_id",
                "candidate_geometry_sha256",
            ]
        ),
        "plan": {
            "components": {
                "d1_allnms_candidates": {
                    "path": str(allnms_path.resolve()),
                    "sha256": sha256_file(allnms_path),
                    "bytes": allnms_path.stat().st_size,
                }
            }
        },
    }
    try:
        exact_d1_allnms_candidates(value)
    except RuntimeError as error:
        assert "exactly cover" in str(error)
    else:
        raise AssertionError("missing raw AllNMS candidate was accepted")


def test_visual_probability_authority_aliases_exact_fields(tmp_path: Path) -> None:
    probability = tmp_path / "probability.npy"
    probability.write_bytes(b"probability")
    visual = pd.DataFrame(
        {
            "sample_id": ["sample-0"],
            "predicted_hifics_probability_path": [str(probability)],
            "predicted_hifics_probability_sha256": [sha256_file(probability)],
            "expression": ["Grasp the cup"],
        }
    ).set_index("sample_id", drop=False)
    aliased = alias_hifics_probability_assets(visual)
    assert aliased.loc["sample-0", "hifics_probability_path"] == str(probability)
    assert (
        aliased.loc["sample-0", "hifics_probability_sha256"]
        == sha256_file(probability)
    )
    assert aliased.loc["sample-0", "language_prompt"] == "Grasp the cup"


def test_visual_probability_authority_rejects_conflicting_alias(
    tmp_path: Path,
) -> None:
    probability = tmp_path / "probability.npy"
    probability.write_bytes(b"probability")
    visual = pd.DataFrame(
        {
            "sample_id": ["sample-0"],
            "predicted_hifics_probability_path": [str(probability)],
            "predicted_hifics_probability_sha256": [sha256_file(probability)],
            "hifics_probability_path": [str(tmp_path / "other.npy")],
            "expression": ["Grasp the cup"],
        }
    )
    try:
        alias_hifics_probability_assets(visual)
    except RuntimeError as error:
        assert "alias differs" in str(error)
    else:
        raise AssertionError("conflicting visual probability alias was accepted")


def test_probability_adapter_uses_native_bilinear_contract(tmp_path: Path) -> None:
    probability_path = tmp_path / "probability.npy"
    probability = np.linspace(0.0, 1.0, 352 * 352, dtype=np.float32).reshape(
        352, 352
    )
    np.save(probability_path, probability)
    adapter = NativeProbabilityNumpyAdapter(np, {probability_path: (480, 640)})
    observed = adapter.load(probability_path)
    assert observed.shape == (480, 640)
    assert observed.dtype == np.float32
    assert np.isfinite(observed).all()
    assert float(observed.min()) >= 0.0
    assert float(observed.max()) <= 1.0


def test_probability_adapter_does_not_change_unlisted_numpy(tmp_path: Path) -> None:
    array_path = tmp_path / "unlisted.npy"
    expected = np.asarray([[1.0, 2.0]], dtype=np.float32)
    np.save(array_path, expected)
    adapter = NativeProbabilityNumpyAdapter(np, {})
    assert np.array_equal(adapter.load(array_path), expected)


def test_case_adapter_excludes_e0_without_fabricating_candidates() -> None:
    funnel = pd.DataFrame(
        {
            "sample_id": ["empty", "renderable"],
            "candidate_count": [0, 2],
            "mask_quality": [0.1, 0.2],
        }
    )

    def original(_formal: object, frame: pd.DataFrame, evidence: object) -> pd.DataFrame:
        threshold = float(evidence["mask_quality_threshold"])
        rows = frame.loc[frame["mask_quality"].lt(threshold), "sample_id"]
        return pd.DataFrame(
            {
                "case_category": ["low_quality_grounding_association"] * len(rows),
                "selection_rank": range(1, len(rows) + 1),
                "sample_id": rows,
                "selection_rule": ["old"] * len(rows),
            }
        )

    observed = select_renderable_cases(
        original, {}, funnel, {"mask_quality_threshold": 0.5}
    )
    assert observed["sample_id"].tolist() == ["renderable"]
    assert observed["selection_rule"].tolist() == [
        "lexicographic_sample_id_over_nonempty_allnms_population"
    ]
