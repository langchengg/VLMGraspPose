#!/usr/bin/env python3
"""Build canonical sample covariates under a fresh gate and global lease."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.io import artifact_record  # noqa: E402
from gtmask_counterfactual.resource import (  # noqa: E402
    collect_fresh_three_by_five_gate,
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)
from gtmask_counterfactual.sample_covariates import (  # noqa: E402
    write_sample_covariates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--visual-assets", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--resource-gate", type=Path)
    gate.add_argument("--collect-resource-gate", action="store_true")
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    arguments = parser.parse_args()
    run = arguments.run_dir.expanduser().resolve()
    with exclusive_d1_flock(run, purpose="GT-mask sample covariate pixels"):
        if arguments.collect_resource_gate:
            gate_value = collect_fresh_three_by_five_gate(
                repo_root=ROOT, rank1_run_dir=arguments.rank1_run_dir
            )
        else:
            gate_path = arguments.resource_gate.expanduser().resolve()
            gate_value = json.loads(gate_path.read_text(encoding="utf-8"))
        validate_fresh_gate(gate_value)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=arguments.rank1_run_dir,
            prefix="gtmask_sample_covariates_launch",
        )
        output = write_sample_covariates(
            run,
            protocol_lock=arguments.protocol_lock,
            visual_asset_manifest=artifact_record(arguments.visual_assets),
            resource_gate=gate_value,
            resume=arguments.resume,
        )
    print(json.dumps({"status": "COMPLETE", "manifest": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
