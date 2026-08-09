#!/usr/bin/env python3
"""Lifecycle CLI for frozen G1/C1 candidate re-ranking."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.g1_c1_safe_rerank.artifacts import initialize_run, update_phase  # noqa: E402
from src.grasping.g1_c1_safe_rerank.audit import freeze_inference_pools, run_audit  # noqa: E402


DEFAULT_BASE_RUN = PROJECT_ROOT / "runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500"


def _new_run() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "runs" / f"g1_c1_safe_rerank_{stamp}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "audit", "build-dev-candidates", "freeze-pools", "extract-features",
        "oracle-analysis", "train-local", "build-union", "evaluate-oof",
        "api-health", "api-smoke", "api-diagnostic", "api-calibrate", "lock",
        "validate", "formal-test", "recompute", "report", "run-all",
    ))
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--backend", choices=("G1", "C1", "union", "all"), default="all")
    parser.add_argument("--pool", choices=("top5", "top10", "all"), default="top5")
    parser.add_argument("--method")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    parser.add_argument("--allow-new-api", action="store_true")
    parser.add_argument("--allow-formal-api", action="store_true")
    parser.add_argument("--max-api-cost-usd", type=float)
    parser.add_argument("--max-provider-attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--preprocess-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def _resolve_run(args: argparse.Namespace) -> Path:
    if args.run_dir is None:
        if args.command != "audit":
            raise ValueError("--run-dir is required after audit creates the experiment")
        args.run_dir = _new_run()
    return initialize_run(args.run_dir)


def _candidate_command(args: argparse.Namespace, run: Path, backend: str) -> list[str]:
    python = PROJECT_ROOT / ".venv-grasp4dof/bin/python"
    command = [
        str(python), "-m", "tools.g1_c1_rerank.generate_backend_candidates",
        "--base-run", str(args.base_run.expanduser().resolve()),
        "--run-dir", str(run), "--backend", backend,
        "--chunk-size", str(args.chunk_size), "--resume",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def _build_dev(args: argparse.Namespace, run: Path) -> None:
    backends = ("G1", "C1") if args.backend in {"all", "union"} else (args.backend,)
    update_phase(run, "build_dev_candidates", "RUNNING", backends=list(backends))
    for backend in backends:
        subprocess.run(_candidate_command(args, run, backend), cwd=PROJECT_ROOT, check=True)
    update_phase(run, "build_dev_candidates", "COMPLETE", backends=list(backends))


def _api_gate(args: argparse.Namespace, *, formal: bool = False) -> None:
    if args.replay_only:
        return
    if not (args.allow_new_api and os.environ.get("ALLOW_NEW_GEMINI_CALLS") == "1"):
        raise PermissionError("new Gemini calls require --allow-new-api and ALLOW_NEW_GEMINI_CALLS=1")
    budget = args.max_api_cost_usd if args.max_api_cost_usd is not None else os.environ.get("MAX_API_COST_USD")
    if budget is None or float(budget) <= 0:
        raise PermissionError("new Gemini calls require a positive MAX_API_COST_USD")
    if formal and not (args.allow_formal_api and os.environ.get("ALLOW_FORMAL_API_RUN") == "1"):
        raise PermissionError("formal API requires --allow-formal-api and ALLOW_FORMAL_API_RUN=1")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = _resolve_run(args)
    base = args.base_run.expanduser().resolve()
    if args.command == "audit":
        update_phase(run, "audit", "RUNNING")
        result = run_audit(PROJECT_ROOT, base, run)
        update_phase(run, "audit", "COMPLETE", baseline_regression_match=True)
    elif args.command == "freeze-pools":
        update_phase(run, "freeze_pools", "RUNNING")
        result = freeze_inference_pools(base, run)
        update_phase(run, "freeze_pools", "COMPLETE", pools=len(result))
    elif args.command == "build-dev-candidates":
        _build_dev(args, run)
        result = {"status": "COMPLETE", "phase": "build_dev_candidates"}
    elif args.command.startswith("api-"):
        _api_gate(args)
        raise NotImplementedError(f"{args.command} provider execution is not implemented yet")
    elif args.command == "formal-test":
        _api_gate(args, formal=True)
        raise NotImplementedError("formal-test is unavailable until validation GO and lock")
    else:
        raise NotImplementedError(f"{args.command} is not implemented yet")
    print(json.dumps({"run_dir": str(run), "command": args.command, "result": result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
