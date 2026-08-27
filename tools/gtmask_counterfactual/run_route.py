#!/usr/bin/env python3
"""Fail-closed launcher for locked G1/C1 oracle work and D1 Case-B preflight."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.execution import (  # noqa: E402
    artifact_record,
    counterfactual_job,
    import_g1_c1_retrospective,
)
from gtmask_counterfactual.audit import transition_pipeline_status  # noqa: E402
from gtmask_counterfactual.contracts import RunState  # noqa: E402
from gtmask_counterfactual.resource import exclusive_d1_flock  # noqa: E402
from gtmask_counterfactual.protocol import (  # noqa: E402
    verify_protocol_lock,
)
from unified_reranking.hashing import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--route", choices=("g1", "c1"), required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    parser.add_argument(
        "--branch",
        choices=("predicted", "gt_oracle", "gt_shape_only"),
        required=True,
    )
    parser.add_argument("--pool", choices=("top5", "top10", "allnms"), default="allnms")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--gt-registry", required=True, type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--resource-gate", type=Path)
    parser.add_argument("--collect-resource-gate", action="store_true")
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _assert_common(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.split != "test":
        raise ValueError("current frozen GT-oracle sources contain Test only")
    if args.branch != "gt_oracle":
        raise ValueError("this launcher currently authorizes primary GT-oracle only")
    if args.workers != 1:
        raise ValueError("frozen oracle execution is single-worker")
    run_dir = args.run_dir.expanduser().resolve()
    expected_lock = run_dir / "01_protocol_lock/COUNTERFACTUAL_PROTOCOL_LOCK.json"
    if args.protocol_lock.expanduser().resolve() != expected_lock:
        raise ValueError("--protocol-lock is outside --run-dir")
    lock = verify_protocol_lock(run_dir)
    declaration = lock.get("declaration")
    routes = declaration.get("routes") if isinstance(declaration, dict) else None
    if not isinstance(routes, dict) or not isinstance(routes.get(args.route), dict):
        raise ValueError("protocol declaration lacks route contract")
    route_contract = routes[args.route]
    locked_config = route_contract.get(
        "execution_config" if args.route == "d1" else "selected_config"
    )
    if not isinstance(locked_config, dict):
        raise ValueError("route contract lacks its frozen configuration artifact")
    config_record = artifact_record(args.config.expanduser().resolve())
    if (
        Path(str(locked_config.get("path", ""))).expanduser().resolve()
        != Path(config_record["path"])
        or str(locked_config.get("sha256", "")) != config_record["sha256"]
    ):
        raise ValueError("--config differs from protocol lock")
    return run_dir, route_contract


def _job_identity(args: argparse.Namespace, route_contract: dict[str, Any]) -> str:
    """Return a scientific identity independent of resume/gate operations."""

    payload = {
        "tool": str(Path(__file__).resolve()),
        "route": args.route,
        "phase": args.phase,
        "split": args.split,
        "branch": args.branch,
        "pool": args.pool,
        "workers": args.workers,
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "gt_registry_sha256": sha256_file(args.gt_registry),
        "retrospective_source_run": route_contract["retrospective_source_run"],
        "selected_config_sha256": route_contract["selected_config"]["sha256"],
        "execution_source_adapter_sha256": route_contract[
            "c1_pilot_source_adapter"
            if args.phase == "pilot"
            else "execution_source_adapter"
        ]["sha256"],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _assert_route_stage(
    run_dir: Path, *, route: str, phase: str, resume: bool
) -> None:
    """Reject any order except C1 audit -> C1 full import -> G1 full import."""

    pipeline = _load_json(run_dir / "pipeline_status.json")
    observed = str(pipeline.get("status", ""))
    execution_count = int(pipeline.get("counterfactual_execution_count", -1))
    if route == "g1" and phase == "pilot":
        raise ValueError("the preregistered audit pilot is C1-only")
    prior, completed = {
        ("c1", "pilot"): (
            RunState.P3_PROTOCOL_LOCKED.value,
            RunState.P4_C1_PILOT_PASS.value,
        ),
        ("c1", "full"): (
            RunState.P4_C1_PILOT_PASS.value,
            RunState.P5_C1_FULL_COMPLETE.value,
        ),
        ("g1", "full"): (
            RunState.P5_C1_FULL_COMPLETE.value,
            RunState.P5B_G1_FULL_COMPLETE.value,
        ),
    }[(route, phase)]
    if observed == completed:
        if not resume or execution_count != 0:
            raise PermissionError(
                f"{route} completed-stage observation requires --resume and zero "
                "current-run model executions"
            )
        return
    if observed != prior:
        raise PermissionError(
            f"{route} execution requires {prior}; observed {observed or '<missing>'}"
        )
    allowed_counts = {0}
    if execution_count not in allowed_counts:
        raise PermissionError(
            f"{route} execution count differs before launch: {execution_count}"
        )


def _dry_run_plan(
    args: argparse.Namespace, route_contract: dict[str, Any]
) -> dict[str, Any]:
    """Describe a bounded import without claiming or writing any run artifact."""

    import pandas as pd

    record = (
        route_contract["c1_pilot_test_samples"]
        if args.phase == "pilot"
        else route_contract["execution_test_samples"]
    )
    ids = pd.read_parquet(record["path"], columns=["sample_id"])[
        "sample_id"
    ].astype(str).tolist()
    start = 0 if args.start_index is None else args.start_index
    end = len(ids) if args.end_index is None else args.end_index
    if start < 0 or end < start or end > len(ids):
        raise ValueError(
            f"dry-run index window must satisfy 0 <= start <= end <= {len(ids)}"
        )
    window = ids[start:end]
    from unified_reranking.hashing import canonical_sha256

    return {
        "status": "DRY_RUN",
        "writes_performed": False,
        "model_inference_performed": False,
        "execution_mode": route_contract["execution_mode"],
        "route": args.route,
        "phase": args.phase,
        "population_count": len(ids),
        "start_index": start,
        "end_index": end,
        "window_count": len(window),
        "ordered_window_ids_sha256": canonical_sha256(window),
        "retrospective_source_run": route_contract["retrospective_source_run"],
    }


def main() -> int:
    args = parse_args()
    run_dir, route_contract = _assert_common(args)
    _assert_route_stage(
        run_dir, route=args.route, phase=args.phase, resume=args.resume
    )
    if args.start_index is not None or args.end_index is not None:
        if not args.dry_run:
            raise ValueError(
                "--start-index/--end-index are dry-run-only; retrospective imports "
                "publish one exact atomic population"
            )
    if args.python is not None or args.resource_gate is not None or args.collect_resource_gate:
        raise ValueError(
            "--python and resource-gate options are invalid for retrospective import"
        )
    if args.source_run is not None and (
        args.source_run.expanduser().resolve()
        != Path(str(route_contract.get("retrospective_source_run", ""))).resolve()
    ):
        raise ValueError("--source-run differs from the locked retrospective source")
    if not isinstance(route_contract.get("native_manifest_fields"), dict):
        raise ValueError("G1/C1 route lock lacks native manifest fields")
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, route_contract), indent=2, sort_keys=True))
        return 0
    with exclusive_d1_flock(
        run_dir, purpose=f"GT-mask {args.route} retrospective {args.phase} import"
    ):
        stage = {
            ("c1", "pilot"): "P4_C1_PILOT_PASS",
            ("c1", "full"): "P5_C1_FULL_COMPLETE",
            ("g1", "full"): "P5B_G1_FULL_COMPLETE",
        }[(args.route, args.phase)]
        with counterfactual_job(
            run_dir,
            stage=stage,
            route=args.route,
            branch=args.branch,
            command=_job_identity(args, route_contract),
        ) as job:
            manifest = import_g1_c1_retrospective(
                run_dir=run_dir,
                route=args.route,
                scope=args.phase,
                registry_path=args.gt_registry,
                protocol_lock_path=args.protocol_lock,
            )
            job["artifact_path"] = str(manifest)
            job["artifact_sha256"] = sha256_file(manifest)
            target, next_stage = {
                ("c1", "pilot"): (
                    RunState.P4_C1_PILOT_PASS,
                    RunState.P5_C1_FULL_COMPLETE.value,
                ),
                ("c1", "full"): (
                    RunState.P5_C1_FULL_COMPLETE,
                    RunState.P5B_G1_FULL_COMPLETE.value,
                ),
                ("g1", "full"): (
                    RunState.P5B_G1_FULL_COMPLETE,
                    RunState.P7_TAXONOMY_COMPLETE.value,
                ),
            }[(args.route, args.phase)]
            transition_pipeline_status(run_dir, target, first_incomplete_stage=next_stage)
            print(manifest)
            return 0
    raise AssertionError("unreachable route dispatch")


if __name__ == "__main__":
    raise SystemExit(main())
