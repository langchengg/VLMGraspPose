#!/usr/bin/env python3
"""Independently audit the retained repeated-FiLM test baseline and Oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.repeatedfilm_baseline import (  # noqa: E402
    audit_repeatedfilm_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--nms-candidates", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--mask-metadata", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=7675)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_repeatedfilm_baseline(
        raw_candidates=args.raw_candidates.expanduser().resolve(),
        nms_candidates=args.nms_candidates.expanduser().resolve(),
        scores=args.scores.expanduser().resolve(),
        mask_metadata=args.mask_metadata.expanduser().resolve(),
        test_manifest=args.test_manifest.expanduser().resolve(),
        annotations=args.annotations.expanduser().resolve(),
        evaluation_config_path=args.evaluation_config.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        tmp_root=args.tmp_root.expanduser().resolve(),
        expected_samples=args.expected_samples,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    print(json.dumps(summary["metrics"], sort_keys=True))
    print(f"outputs={args.output_root.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
