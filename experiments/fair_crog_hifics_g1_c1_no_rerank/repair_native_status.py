#!/usr/bin/env python3
"""Deterministically correct pre-v1 input no-output status labels; geometry untouched."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


EXPECTED = {
    "BackendInputError:empty_mask",
    "BackendInputError:mask_too_small",
    "BackendInputError:missing_predicted_mask",
    "BackendInputError:missing_probability_map",
    "BackendInputError:invalid_depth",
    "BackendInputError:invalid_target_depth",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.run_dir.resolve() / "00_audit/native_status_correction.json"
    previous = json.loads(destination.read_text()) if destination.exists() else None
    changes = []
    rank_changes = []
    for variant in ("g1", "c1", "g1_gtmask_oracle", "c1_gtmask_oracle"):
        path = args.run_dir.resolve() / "02_predictions/native_work" / variant / "per_sample.parquet"
        frame = pd.read_parquet(path)
        selected = frame.status.eq("technical_failure") & frame.failure_reason.isin(EXPECTED)
        for row in frame[selected].itertuples(index=False):
            changes.append({"variant": variant, "sample_id": row.sample_id, "old_status": "technical_failure", "new_status": "no_output", "failure_reason": row.failure_reason})
        frame.loc[selected, "status"] = "no_output"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        candidate_path = path.with_name("candidates.parquet")
        candidates = pd.read_parquet(candidate_path)
        for sample_id, indices in candidates.groupby("sample_id", sort=False).groups.items():
            ordered_indices = candidates.loc[indices].sort_values("native_rank").index
            expected_ranks = list(range(1, len(ordered_indices) + 1))
            observed_ranks = candidates.loc[ordered_indices, "native_rank"].astype(int).tolist()
            if observed_ranks != expected_ranks:
                rank_changes.append({"variant": variant, "sample_id": sample_id, "old_ranks": observed_ranks, "new_ranks": expected_ranks})
                candidates.loc[ordered_indices, "native_rank"] = expected_ranks
        temporary_candidates = candidate_path.with_name(f".{candidate_path.name}.{os.getpid()}.tmp")
        candidates.to_parquet(temporary_candidates, index=False)
        os.replace(temporary_candidates, candidate_path)
    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "rule": "BackendInputError deployment-input conditions are protocol no-output, not runtime technical failures",
        "geometry_or_scores_changed": False,
        "candidate_order_changed": False,
        "changes": changes,
        "rank_compaction_only": True,
        "rank_changes": rank_changes,
        "previous_record": previous,
    }
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status_changed": len(changes), "rank_compactions": len(rank_changes), "path": str(destination)}))


if __name__ == "__main__":
    main()
