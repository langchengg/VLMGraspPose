from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from failure_analysis.reranking_v2.independent_evaluator import recompute_counts

from .schema import atomic_write_json


def independently_recompute_suite(
    *, features_path: str | Path, corrected_labels_path: str | Path,
    legacy_labels_path: str | Path, methods: Mapping[str, str | Path | None],
    expected_summary_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Recompute top-1/oracle with the independently maintained V2 evaluator."""
    expected = json.loads(Path(expected_summary_path).read_text(encoding="utf-8"))
    tracks = {
        "corrected_scientific": corrected_labels_path,
        "legacy_official_compatibility": legacy_labels_path,
    }
    observed = {}
    for track, labels in tracks.items():
        observed[track] = {}
        for name, predictions in methods.items():
            value = recompute_counts(features=features_path, labels=labels, predictions=predictions)
            target = expected["tracks"][track][name]
            # The independently maintained V2 evaluator intentionally uses a
            # different result vocabulary.  Keep the implementations
            # independent and map their public count fields explicitly.
            if int(value["selected_success_count"]) != int(target["correct"]):
                raise AssertionError(f"independent correct-count mismatch: {track}/{name}")
            if int(value["oracle_success_count"]) != int(target["oracle_correct"]):
                raise AssertionError(f"independent Oracle@5 mismatch: {track}/{name}")
            observed[track][name] = value
    result = {
        "schema_version":"3.0.0","kind":"v3_independent_dual_track_recomputation","status":"passed",
        "methods":list(methods),"tracks":observed,"all_correct_counts_exact":True,
        "all_oracle_counts_exact":True,"independent_implementation":"failure_analysis.reranking_v2.independent_evaluator.recompute_counts",
    }
    atomic_write_json(output_path,result); return result
