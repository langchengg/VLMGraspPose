"""Freeze the D1 R7 expected-gain gate grid before gate fitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.execution import artifact_record  # noqa: E402
from d1_reranking.run import assert_writable_prelock  # noqa: E402
from unified_reranking.gate import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
)
from unified_reranking.hashing import atomic_json, canonical_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def run(run_dir: Path) -> dict[str, object]:
    root = run_dir.expanduser().resolve()
    assert_writable_prelock(root)
    path = root / "configs" / "d1_gate_grid.json"
    if path.exists():
        raise FileExistsError(f"D1 gate grid already exists: {path}")
    value: dict[str, object] = {
        "schema_version": 1,
        "status": "PLANNED",
        "lambda_harm": [1.0, 2.0, 4.0],
        "utility_thresholds": [0.0, 0.05, 0.1],
        "score_margin_thresholds": [0.0, 0.05, 0.1],
        "reliability_thresholds": [0.5, 0.75],
        "stability_thresholds": [0.5, 0.75],
        "minimum_seed_votes": 2,
        "selection_rule": (
            "maximum Validation scene-bootstrap 95% lower bound; then mean delta, "
            "fewer harmful, lower switch rate; native if every lower bound <= 0"
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "candidate_test_labels_read": False,
        "sources": {
            "gate_primitive": artifact_record(ROOT / "src/unified_reranking/gate.py"),
            "tool": artifact_record(Path(__file__)),
        },
    }
    value["content_sha256"] = canonical_sha256(value)
    atomic_json(path, value)
    return value


def main() -> int:
    run(parse_args().run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
