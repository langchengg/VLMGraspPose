"""Focused tests for full-run verification and visual review safeguards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_dexnet_full_review import (
    render_case,
    validate_review_paths,
    verified_summary_rows,
)
from scripts.verify_full_dexnet_candidate_run import aggregate_statistics, safe_int


def clean_summary(*, failed: int = 0) -> dict:
    samples = [
        {
            "sample_id": "failed-sample",
            "status": "failed",
            "raw_candidate_count": "",
            "mask_validated_count": "",
            "post_nms_count": "",
            "failure_reason": "broken input",
        }
    ] if failed else []
    return {
        "verification": {
            "expected_samples": len(samples),
            "failed_samples": failed,
            "missing_samples": 0,
            "corrupt_samples": 0,
            "duplicate_sample_ids": [],
            "unexpected_output_directories": [],
            "configuration_hash_mismatches": [],
            "seed_mismatches": [],
            "accounting_identity": len(samples),
            "accounting_identity_expected": len(samples),
        },
        "samples": samples,
    }


def test_failed_empty_counts_do_not_crash_aggregation(tmp_path: Path) -> None:
    failed = clean_summary(failed=1)["samples"][0]
    statistics = aggregate_statistics(
        tmp_path,
        [{"sample_id": "failed-sample"}],
        [failed],
        missing=[],
        corrupt={},
    )
    assert safe_int("") == 0
    assert statistics["total_failures"] == 1
    assert statistics["post_nms_candidate_count"] == {"count": 0}


def test_review_rejects_partial_verification_and_requires_explicit_failures() -> None:
    partial = clean_summary()
    partial["verification"]["missing_samples"] = 1
    with pytest.raises(ValueError, match="not clean"):
        verified_summary_rows(partial, allow_failures=False)

    failed = clean_summary(failed=1)
    with pytest.raises(ValueError, match="--allow-failures"):
        verified_summary_rows(failed, allow_failures=False)
    assert verified_summary_rows(failed, allow_failures=True) == failed["samples"]


@pytest.mark.parametrize("relative", [".", "review", "review/nested"])
def test_review_root_cannot_equal_or_nest_under_candidate_root(
    tmp_path: Path, relative: str
) -> None:
    candidate = (tmp_path / "candidates").resolve()
    review = (candidate / relative).resolve()
    with pytest.raises(ValueError, match="non-nested"):
        validate_review_paths(candidate, review)


def test_failed_unreadable_input_gets_auditable_placeholders(tmp_path: Path) -> None:
    sample_dir = tmp_path / "candidates" / "failed-sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "failure.json").write_text(
        json.dumps({"failure_reason": "missing input"}) + "\n", encoding="utf-8"
    )
    destination = tmp_path / "review" / "failed-sample"
    result = render_case(
        tmp_path / "missing-mask-root",
        sample_dir,
        destination,
        {"sample_id": "failed-sample", "status": "failed"},
    )
    assert result["render_errors"]
    for name in (
        "rgb.png",
        "predicted_hifics_mask.png",
        "depth.png",
        "raw_candidates.png",
        "filtered_candidates.png",
        "topk_candidates.png",
        "failure.json",
        "case.json",
    ):
        assert (destination / name).is_file(), name

