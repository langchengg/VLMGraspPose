"""Resumable entry point for the unified fair reranking experiment."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.audit import audit_fair_source, capture_environment
from unified_reranking.candidates import build_available_candidate_pools
from unified_reranking.hashing import atomic_json, atomic_text, sha256_file
from unified_reranking.ledger import initialize_ledger, ledger_stage
from unified_reranking.manifests import build_paired_manifests


DIRECTORIES = (
    "00_audit",
    "01_manifests",
    "02_candidates",
    "03_features",
    "04_splits",
    "05_calibration",
    "06_oof",
    "07_validation",
    "08_lock",
    "09_formal_test",
    "10_statistics",
    "11_attribution_bridge",
    "12_figures",
    "13_failure_galleries",
    "14_reports",
    "15_independent_recompute",
    "configs",
    "checkpoints",
    "predictions",
    "tables",
    "logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fair-run",
        type=Path,
        default=ROOT / "runs/fair_crog_hifics_g1_c1_no_rerank_20260807_091523",
    )
    parser.add_argument(
        "--modular-run",
        type=Path,
        default=ROOT
        / "HiFi_reproduction/runs/modular_repeatedfilm_4dof_backends_v1_r0corrected_20260803_163500",
    )
    parser.add_argument(
        "--old-feature-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159/features",
    )
    parser.add_argument(
        "--stop-after", choices=("audit", "manifests", "candidates"), default="candidates"
    )
    return parser.parse_args()


def write_reproduce_readme(run_dir: Path) -> Path:
    """Write the honest staged reproduction guide used by the final lock."""

    reproduce = f"""# Reproduce unified fair reranking

`run_all --resume` resumes only the P0-P1 audits, paired manifests, and frozen
candidate pools. It does not execute the complete experiment.

```bash
cd {ROOT.resolve()}
{sys.executable} -m tools.unified_reranking.run_all --run-dir {run_dir.resolve()} --resume
{sys.executable} -m tools.unified_reranking.pipeline_status --run-dir {run_dir.resolve()} --write-status
```

Follow the `next_commands` for the first actionable stage in
`_PIPELINE_STATUS.json`. The guarded terminal commands are:

External fair-run, modular-run, legacy-feature, evaluator, and candidate-label
inputs are path-and-hash bound by `00_audit/fair_source_audit.json`; candidate
pools are bound by `02_candidates/candidate_contract_hashes.json`. Those frozen
records, rather than mutable defaults, are the continuation inputs.

```bash
{sys.executable} -m tools.unified_reranking.prepare_test_bridge_bundle --run-dir {run_dir.resolve()}
{sys.executable} -m tools.unified_reranking.assemble_prelock --run-dir {run_dir.resolve()}
{sys.executable} -m tools.unified_reranking.create_formal_test_lock --run-dir {run_dir.resolve()} --evaluation-plan {run_dir.resolve()}/08_lock/formal_evaluation_plan.json --extra-locked-file code_manifest={run_dir.resolve()}/08_lock/code_manifest.json --extra-locked-file prelock_assembly={run_dir.resolve()}/08_lock/prelock_assembly_manifest.json
{sys.executable} -m tools.unified_reranking.run_formal_test_once --run-dir {run_dir.resolve()}
{sys.executable} -m tools.unified_reranking.independent_recompute --run-dir {run_dir.resolve()}
{sys.executable} -m tools.unified_reranking.build_postformal_artifacts --run-dir {run_dir.resolve()}
```

Candidate-level Test labels remain unavailable until the formal lock is
complete; the formal Test command is an exactly-once execution.
"""
    return atomic_text(run_dir / "README_REPRODUCE.md", reproduce)


def bootstrap(run_dir: Path, resume: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"run directory is non-empty; pass --resume: {run_dir}")
    for directory in DIRECTORIES:
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    initialize_ledger(run_dir / "run_ledger.sqlite")
    if not (run_dir / "manifest.json").exists():
        manifest = {
            "schema_version": 1,
            "protocol": "fair-unified-reranking-v1",
            "status": "IN_PROGRESS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir.resolve()),
            "repo_root": str(ROOT.resolve()),
            "formal_test_execution_count": 0,
            "test_label_state": "LOCKED_PREVALIDATION",
        }
        atomic_json(run_dir / "manifest.json", manifest)
    # P0-P1 bootstrap is intentionally not presented as the full experiment
    # driver. The readiness graph owns the staged, resumable commands.
    write_reproduce_readme(run_dir)
    commands = run_dir / "commands.log"
    if not commands.exists():
        atomic_text(commands, "")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    immutable_markers = tuple(
        path
        for path in (
            run_dir / "08_lock" / "FORMAL_TEST_LOCK.json",
            run_dir / "FINAL_RUN_LOCK.json",
        )
        if path.is_file()
    )
    if immutable_markers:
        joined = ", ".join(str(path) for path in immutable_markers)
        raise PermissionError(
            "run_all is a P0-P1 bootstrap command and refuses to write after an "
            f"immutable lock exists ({joined}); use pipeline_status for read-only verification"
        )
    bootstrap(run_dir, args.resume)
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P0",
        substage="environment_and_fair_source_audit",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        capture_environment(ROOT, run_dir)
        audit = audit_fair_source(
            run_dir, args.fair_run.expanduser().resolve(), args.modular_run.expanduser().resolve()
        )
        if audit["status"] != "PASS":
            raise RuntimeError("fair baseline audit failed; see BASELINE_MISMATCH_REPORT.md")
        artifact = run_dir / "00_audit" / "fair_source_audit.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    if args.stop_after == "audit":
        atomic_json(
            run_dir / "_STATUS.json",
            {"status": "AUDIT_COMPLETE", "next_stage": "manifests", "formal_test_executed": False},
        )
        return 0

    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P1",
        substage="paired_inference_manifests",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        build_paired_manifests(
            run_dir,
            args.modular_run.expanduser().resolve(),
            args.fair_run.expanduser().resolve(),
        )
        artifact = run_dir / "01_manifests" / "paired_manifest_audit.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    if args.stop_after == "manifests":
        atomic_json(
            run_dir / "_STATUS.json",
            {"status": "MANIFESTS_COMPLETE", "next_stage": "candidates", "formal_test_executed": False},
        )
        return 0

    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P1",
        substage="freeze_available_candidates",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        build_available_candidate_pools(
            run_dir,
            args.fair_run.expanduser().resolve(),
            args.old_feature_dir.expanduser().resolve(),
        )
        artifact = run_dir / "02_candidates" / "candidate_contract_hashes.json"
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    status = {
        "status": "AVAILABLE_CANDIDATES_FROZEN",
        "next_stage": "generate fair G1/C1 Train and Validation candidates",
        "formal_test_executed": False,
    }
    atomic_json(run_dir / "_STATUS.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
