#!/usr/bin/env python3
"""Compare CPU/MPS candidate topology and numerical geometry on the same smoke set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    result = {
        "schema_version": 1,
        "formal_device_decision": "mps",
        "decision_basis": "same candidate topology/ranks within numerical tolerance; lower smoke latency",
        "candidate_id_note": (
            "IDs hash exact floating geometry, so sub-1e-4 CPU/MPS numerical drift changes IDs "
            "even when rank topology and centers are identical"
        ),
        "methods": {},
    }
    for method in ("G0", "C0"):
        mps = pd.read_parquet(
            run / f"validation/smoke10/{method}/per_candidate_predictions.parquet"
        )
        cpu = pd.read_parquet(
            run / f"validation/smoke10_cpu/{method}/per_candidate_predictions.parquet"
        )
        merged = mps.merge(
            cpu,
            on=["sample_id", "rank"],
            suffixes=("_mps", "_cpu"),
            how="outer",
            indicator=True,
        )
        common = merged[merged["_merge"] == "both"]
        maximum = {
            key: float((common[f"{key}_mps"] - common[f"{key}_cpu"]).abs().max())
            for key in ("center_x", "center_y", "angle_deg", "width_px", "score")
        }
        result["methods"][method] = {
            "mps_candidate_rows": len(mps),
            "cpu_candidate_rows": len(cpu),
            "common_sample_rank_rows": len(common),
            "candidate_topology_equal": len(mps) == len(cpu) == len(common),
            "candidate_ids_exactly_equal": bool(
                (common["candidate_id_mps"] == common["candidate_id_cpu"]).all()
            ),
            "max_absolute_difference": maximum,
            "within_geometry_tolerance": maximum["angle_deg"] < 1e-3
            and maximum["width_px"] < 1e-3
            and maximum["center_x"] == 0.0
            and maximum["center_y"] == 0.0,
            "mps_metrics": json.loads(
                (run / f"validation/smoke10/{method}/metrics.json").read_text()
            ),
            "cpu_metrics": json.loads(
                (run / f"validation/smoke10_cpu/{method}/metrics.json").read_text()
            ),
        }
    if not all(
        row["candidate_topology_equal"] and row["within_geometry_tolerance"]
        for row in result["methods"].values()
    ):
        raise RuntimeError("CPU/MPS candidate topology or geometry diverged")
    output = run / "audit/device_candidate_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
