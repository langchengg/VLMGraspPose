#!/usr/bin/env python3
"""Print isolated one-sample commands for the requested CPU thread benchmark."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sample-id", default="q0000000_b32eb3299dcd3ae9")
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[2, 4, 8, 15])
    args = parser.parse_args()
    commands = []
    for threads in args.thread_counts:
        command = [
            ".venv-sam3-cpu/bin/python",
            "scripts/run_sam3_cpu_refinement.py",
            "--model-path",
            str(args.model_path),
            "--revision",
            args.revision,
            "--sample-id",
            args.sample_id,
            "--sample-limit",
            "1",
            "--num-threads",
            str(threads),
            "--output-root",
            f"outputs/sam3_cpu_thread_benchmark_{threads}",
        ]
        commands.append({"threads": threads, "command": shlex.join(command)})
    print(json.dumps({"status": "COMMANDS_READY", "runs": commands}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
