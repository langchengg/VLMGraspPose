#!/usr/bin/env python3
"""Analyze six hash-bound saved GT-mask frames; never generate candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.postprocess import (  # noqa: E402
    EXPECTED_SAMPLE_COUNT,
    run_postprocess,
    write_final_outcomes_authority,
    write_postprocess_input_manifest,
    write_route_frame_manifest,
)
from gtmask_counterfactual.statistics import BOOTSTRAP_ITERATIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument(
        "--lock-inputs-json",
        type=Path,
        help="write the canonical input manifest from artifact records, then exit",
    )
    parser.add_argument(
        "--lock-final-outcomes-json",
        type=Path,
        help=(
            "write and verify FINAL_OUTCOMES_AUTHORITY from final_outcomes and "
            "selector_sources artifact records, then exit"
        ),
    )
    parser.add_argument(
        "--lock-route-frame-json",
        type=Path,
        help="write one canonical normalized route-frame manifest, then exit",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("test",), default="test")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    modes = (
        arguments.lock_inputs_json,
        arguments.lock_final_outcomes_json,
        arguments.lock_route_frame_json,
    )
    if sum(value is not None for value in modes) > 1:
        raise ValueError("choose only one authority-writing mode")
    if arguments.lock_route_frame_json is not None:
        declaration_path = arguments.lock_route_frame_json.expanduser().resolve()
        if declaration_path.is_symlink() or not declaration_path.is_file():
            raise ValueError("--lock-route-frame-json must be a regular file")
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        required = {
            "route",
            "branch",
            "candidates",
            "per_sample",
            "source_contract",
        }
        if not isinstance(declaration, dict) or not required.issubset(declaration):
            raise ValueError("route-frame declaration schema differs")
        output = write_route_frame_manifest(
            arguments.run_dir,
            route=declaration["route"],
            branch=declaration["branch"],
            candidates=declaration["candidates"],
            per_sample=declaration["per_sample"],
            source_contract=declaration["source_contract"],
            sample_count=EXPECTED_SAMPLE_COUNT,
        )
        print(json.dumps({"status": "COMPLETE", "route_manifest": str(output)}))
        return 0
    if arguments.lock_final_outcomes_json is not None:
        declaration_path = arguments.lock_final_outcomes_json.expanduser().resolve()
        if declaration_path.is_symlink() or not declaration_path.is_file():
            raise ValueError(
                "--lock-final-outcomes-json must be a regular non-symlink file"
            )
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        if (
            not isinstance(declaration, dict)
            or not isinstance(declaration.get("final_outcomes"), dict)
            or not isinstance(declaration.get("selector_sources"), dict)
        ):
            raise ValueError("final-outcomes declaration schema differs")
        output = write_final_outcomes_authority(
            arguments.run_dir,
            protocol_lock=arguments.protocol_lock,
            final_outcomes=declaration["final_outcomes"],
            selector_sources=declaration["selector_sources"],
            routes=tuple(declaration.get("routes", ("G1", "C1", "D1"))),
        )
        print(json.dumps({"status": "LOCKED", "authority": str(output)}))
        return 0
    if arguments.lock_inputs_json is not None:
        declaration_path = arguments.lock_inputs_json.expanduser().resolve()
        if declaration_path.is_symlink() or not declaration_path.is_file():
            raise ValueError("--lock-inputs-json must be a regular non-symlink file")
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        if not isinstance(declaration, dict):
            raise ValueError("--lock-inputs-json must contain one JSON object")
        output = write_postprocess_input_manifest(
            arguments.run_dir,
            artifacts=declaration,
            protocol_lock=arguments.protocol_lock,
            sample_count=EXPECTED_SAMPLE_COUNT,
        )
        print(json.dumps({"status": "LOCKED", "input_manifest": str(output)}))
        return 0
    result = run_postprocess(
        arguments.run_dir,
        protocol_lock=arguments.protocol_lock,
        input_manifest=arguments.input_manifest,
        resume=arguments.resume,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "scientific_role": "post-formal oracle stage-replacement diagnostic",
                "route_status": str(result),
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "training_or_selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
