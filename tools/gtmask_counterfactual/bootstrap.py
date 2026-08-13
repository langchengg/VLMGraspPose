"""Bootstrap a new independent counterfactual run after full source rehash."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gtmask_counterfactual.audit import bootstrap_run, verify_source_locks  # noqa: E402
from gtmask_counterfactual.contracts import SOURCE_LOCK_EXPECTATIONS  # noqa: E402
from gtmask_counterfactual.io import atomic_json  # noqa: E402
from gtmask_counterfactual.resource import (  # noqa: E402
    collect_fresh_three_by_five_gate,
    exclusive_d1_flock,
    validate_fresh_gate,
    validate_live_resources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--unified-run",
        type=Path,
        default=ROOT / "runs/fair_unified_reranking_20260809_103012",
    )
    parser.add_argument(
        "--d1-run",
        type=Path,
        default=ROOT / "runs/fair_d1_reranking_extension_20260811T145515Z",
    )
    parser.add_argument(
        "--full-rehash-source-inventory",
        action="store_true",
        help="mandatory: independently byte-rehash every source-lock inventory file",
    )
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument("--resource-gate", type=Path)
    gate.add_argument("--collect-resource-gate", action="store_true")
    parser.add_argument(
        "--rank1-run-dir",
        type=Path,
        default=ROOT / "runs/reranking_complete_20260803_094159",
    )
    return parser.parse_args()


def _load_gate(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"resource gate is not a regular file: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("resource gate must contain a JSON object")
    return value


def main() -> int:
    args = parse_args()
    if not args.full_rehash_source_inventory:
        raise SystemExit("--full-rehash-source-inventory is mandatory")
    run_dir = args.run_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = ROOT / "runs" / f"fair_gtmask_counterfactual_g1_c1_d1_{stamp}"
    run_dir = run_dir.expanduser().resolve()
    with exclusive_d1_flock(run_dir, purpose="GT-mask source inventory audit"):
        gate = (
            collect_fresh_three_by_five_gate(
                repo_root=ROOT,
                rank1_run_dir=args.rank1_run_dir.expanduser().resolve(),
            )
            if args.collect_resource_gate
            else _load_gate(args.resource_gate)
        )
        validate_fresh_gate(gate)
        validate_live_resources(
            repo_root=ROOT,
            rank1_run_dir=args.rank1_run_dir,
            prefix="gtmask_source_rehash_before_launch",
        )
        verification = verify_source_locks(
            {
                "unified": (
                    args.unified_run,
                    SOURCE_LOCK_EXPECTATIONS["unified"],
                ),
                "d1": (args.d1_run, SOURCE_LOCK_EXPECTATIONS["d1"]),
            },
            full_inventory_rehash=True,
        )
        bootstrap_run(run_dir, source_verification=verification, resume=args.resume)
        atomic_json(
            run_dir / "00_audit/SOURCE_REHASH_RESOURCE_GATE.json", dict(gate)
        )
    print(Path(run_dir).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
