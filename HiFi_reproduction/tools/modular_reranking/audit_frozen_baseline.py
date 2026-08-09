#!/usr/bin/env python3
"""Audit the frozen test baseline, strict labels, and Oracle funnel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.baseline_audit import (  # noqa: E402
    assert_expected_baseline,
    audit_frozen_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--limit",
        type=int,
        help="Audit only the first N samples as a smoke test; skips the full hard gate",
    )
    return parser.parse_args()


def _strict_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    protected_roots = [
        args.candidate_root.expanduser().resolve(),
        args.scored_root.expanduser().resolve(),
        args.bundle_manifest.expanduser().resolve().parent,
    ]
    if any(output_dir == root or root in output_dir.parents for root in protected_roots):
        raise ValueError("output-dir must be outside every frozen input root")
    output_dir.mkdir(parents=True, exist_ok=False)
    command = " ".join([sys.executable, *sys.argv])
    (output_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")

    summary, samples, labels = audit_frozen_baseline(
        candidate_root=args.candidate_root.expanduser().resolve(),
        scored_root=args.scored_root.expanduser().resolve(),
        bundle_manifest=args.bundle_manifest.expanduser().resolve(),
        annotation_file=args.annotation_file.expanduser().resolve(),
        evaluation_config=args.evaluation_config.expanduser().resolve(),
        progress_every=args.progress_every,
        limit=args.limit,
    )
    if args.limit is None:
        assert_expected_baseline(summary)
    _strict_json(output_dir / "baseline_audit.json", summary)
    samples.to_parquet(
        output_dir / "per_sample_baseline.parquet",
        index=False,
        compression="zstd",
    )
    labels.to_parquet(
        output_dir / "test_candidate_labels.parquet",
        index=False,
        compression="zstd",
    )
    if "funnel_category" in samples:
        categories = (
            samples.groupby("funnel_category", sort=True)
            .size()
            .rename("count")
            .reset_index()
        )
    else:
        categories = samples
    categories.to_csv(output_dir / "funnel_categories.csv", index=False)
    print(json.dumps(summary["counts"], sort_keys=True))
    print(json.dumps(summary["metrics"], sort_keys=True))
    status = "baseline hard gate passed" if args.limit is None else "smoke audit passed"
    print(f"{status}; outputs={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
