from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from failure_analysis.vlm_safe_rerank.independent_recompute import (
    independent_recompute_p5_validation,
)


def _labels(path: Path, values: dict[str, dict[str, bool]]) -> None:
    lines = []
    for sample_id, candidates in values.items():
        lines.append(json.dumps({
            "sample_id": sample_id,
            "candidate_labels": [
                {"candidate_id": candidate_id, "candidate_correct": correct}
                for candidate_id, correct in candidates.items()
            ],
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_independent_p5_recompute_uses_ids_and_frozen_labels(tmp_path: Path) -> None:
    phase = tmp_path / "p5_validation"
    phase.mkdir()
    rows = []
    for method in ("P5_er2_safe", "P5_flash_safe", "P5_joint_safe"):
        rows.extend([
            {
                "method": method, "sample_id": "s0",
                "baseline_candidate_id": "candidate_0",
                "selected_candidate_id": "candidate_1",
                "baseline_correct": False, "selected_correct": True,
                "legacy_baseline_correct": False, "legacy_selected_correct": False,
            },
            {
                "method": method, "sample_id": "s1",
                "baseline_candidate_id": "candidate_0",
                "selected_candidate_id": "candidate_0",
                "baseline_correct": True, "selected_correct": True,
                "legacy_baseline_correct": True, "legacy_selected_correct": True,
            },
        ])
    pq.write_table(pa.Table.from_pylist(rows), phase / "p5_decisions.parquet")
    method_payload = {
        "corrected": {
            "total": 2, "baseline_successes": 1, "final_successes": 2,
            "recovered": 1, "harmful": 0, "net": 1, "switches": 1,
        },
        "legacy": {
            "total": 2, "baseline_successes": 1, "final_successes": 1,
            "recovered": 0, "harmful": 0, "net": 0, "switches": 1,
        },
    }
    (phase / "P5_VALIDATION_RESULTS.json").write_text(json.dumps({
        "expected_denominator": 2,
        "methods": {method: method_payload for method in (
            "P5_er2_safe", "P5_flash_safe", "P5_joint_safe"
        )},
    }), encoding="utf-8")
    corrected = tmp_path / "corrected.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    _labels(corrected, {
        "s0": {"candidate_0": False, "candidate_1": True},
        "s1": {"candidate_0": True, "candidate_1": False},
    })
    _labels(legacy, {
        "s0": {"candidate_0": False, "candidate_1": False},
        "s1": {"candidate_0": True, "candidate_1": False},
    })

    result = independent_recompute_p5_validation(
        tmp_path, corrected_labels=corrected, legacy_labels=legacy,
    )

    assert result["all_methods_match_main"] is True
    assert result["selected_candidate_ids_verified"] is True
    assert result["methods"]["P5_er2_safe"]["corrected"]["net"] == 1
    assert result["methods"]["P5_er2_safe"]["legacy"]["net"] == 0
    assert result["methods"]["P5_er2_safe"]["selected_ids_match_q_only"] is False
    assert json.loads((tmp_path / "independent_recompute_results.json").read_text())["formal_evidence_used"] is False
