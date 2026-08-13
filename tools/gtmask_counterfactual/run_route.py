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
    FROZEN_NATIVE_INFERENCE_SHA256,
    NATIVE_INFERENCE,
    artifact_record,
    counterfactual_job,
    run_g1_c1_oracle,
)
from gtmask_counterfactual.audit import transition_pipeline_status  # noqa: E402
from gtmask_counterfactual.contracts import RunState  # noqa: E402
from gtmask_counterfactual.resource import (  # noqa: E402
    collect_fresh_three_by_five_gate,
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)
from gtmask_counterfactual.protocol import (  # noqa: E402
    claim_bulk_execution,
    verify_protocol_lock,
)
from unified_reranking.hashing import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--route", choices=("g1", "c1"), required=True)
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


def _prepare_gate(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    """Collect or load a gate while the caller holds the global heavy lease."""

    if args.collect_resource_gate and args.resource_gate is not None:
        raise ValueError("choose either --collect-resource-gate or --resource-gate")
    if args.collect_resource_gate:
        gate = collect_fresh_three_by_five_gate(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir.expanduser().resolve(),
        )
        gate_path = (
            run_dir / "00_audit/resource_gates" / f"{gate['content_sha256'][:20]}.json"
        )
        if gate_path.exists():
            raise FileExistsError(f"resource gate already exists: {gate_path}")
        _atomic_json(gate_path, gate)
    elif args.resource_gate is not None:
        gate = _load_json(args.resource_gate.expanduser().resolve())
    else:
        raise ValueError("all GT routes require a fresh resource gate")
    validate_fresh_gate(gate)
    return gate


def _ensure_bulk_claim(run_dir: Path, *, resume: bool) -> None:
    claim = run_dir / "01_protocol_lock/COUNTERFACTUAL_EXECUTION.json"
    claim_bulk_execution(run_dir, resume=resume or claim.exists())


def main() -> int:
    args = parse_args()
    run_dir, route_contract = _assert_common(args)
    if args.source_run is None:
        raise ValueError("G1/C1 require --source-run")
    if sha256_file(NATIVE_INFERENCE) != FROZEN_NATIVE_INFERENCE_SHA256:
        raise ValueError("G1/C1 frozen native source hash differs")
    if (
        args.source_run.expanduser().resolve()
        != Path(str(route_contract.get("source_run", ""))).expanduser().resolve()
    ):
        raise ValueError("G1/C1 source run differs from protocol lock")
    if not isinstance(route_contract.get("native_manifest_fields"), dict):
        raise ValueError("G1/C1 route lock lacks native manifest fields")
    label_path = args.source_run.expanduser().resolve() / "manifests/test_labels.parquet"
    label_record = route_contract.get("oracle_label_manifest")
    if (
        not isinstance(label_record, dict)
        or Path(str(label_record.get("path", ""))).expanduser().resolve()
        != label_path
        or str(label_record.get("sha256", "")) != sha256_file(label_path)
    ):
        raise ValueError("G1/C1 oracle label manifest differs from protocol lock")
    with exclusive_d1_flock(run_dir, purpose=f"GT-mask {args.route} bulk execution"):
        # The same kernel lease spans resource observation, claim publication,
        # and execution.  This closes the gate-to-launch race with every D1
        # sibling run that uses the repository-global lock.
        gate = _prepare_gate(args, run_dir)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix=f"gtmask_{args.route}_launch",
        )
        _ensure_bulk_claim(run_dir, resume=args.resume)
        validate_fresh_gate(gate)
        stage = {
            "g1": "P4_G1_COUNTERFACTUAL_COMPLETE",
            "c1": "P5_C1_COUNTERFACTUAL_COMPLETE",
        }[args.route]
        with counterfactual_job(
            run_dir,
            stage=stage,
            route=args.route,
            branch=args.branch,
            command=" ".join(map(str, sys.argv)),
        ) as job:
            manifest = run_g1_c1_oracle(
                run_dir=run_dir,
                route=args.route,
                source_run=args.source_run,
                registry_path=args.gt_registry,
                protocol_lock_path=args.protocol_lock,
                resume=args.resume,
                python=args.python,
            )
            job["artifact_path"] = str(manifest)
            job["artifact_sha256"] = sha256_file(manifest)
            target = (
                RunState.P4_G1_COUNTERFACTUAL_COMPLETE
                if args.route == "g1"
                else RunState.P5_C1_COUNTERFACTUAL_COMPLETE
            )
            next_stage = (
                RunState.P5_C1_COUNTERFACTUAL_COMPLETE.value
                if args.route == "g1"
                else RunState.P6_D1_COUNTERFACTUAL_COMPLETE.value
            )
            transition_pipeline_status(run_dir, target, first_incomplete_stage=next_stage)
            print(manifest)
            return 0
    raise AssertionError("unreachable route dispatch")


if __name__ == "__main__":
    raise SystemExit(main())
