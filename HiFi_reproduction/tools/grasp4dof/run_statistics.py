#!/usr/bin/env python3
"""Run pre-registered paired all-sample J@1 statistical analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.grasping.common.statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_PAIR_SPECS,
    analyze_paired_methods,
    load_aligned_predictions,
    pair_specs_from_json,
)


def _write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        "--per-sample-predictions",
        dest="predictions",
        type=Path,
        required=True,
        help="per_sample_predictions.parquet containing every method/sample row",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pairs-json",
        type=Path,
        help="pre-registered pair list; defaults to the locked benchmark comparisons",
    )
    parser.add_argument(
        "--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    aligned = load_aligned_predictions(args.predictions)
    pairs = (
        DEFAULT_PAIR_SPECS
        if args.pairs_json is None
        else pair_specs_from_json(args.pairs_json)
    )
    statistical_tests, bootstrap_intervals = analyze_paired_methods(
        aligned,
        pairs,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        alpha=args.alpha,
    )
    predictions = args.predictions.expanduser().resolve()
    pair_records = [asdict(pair) for pair in pairs]
    provenance = {
        "predictions_path": str(predictions),
        "predictions_sha256": _sha256(predictions),
        "pair_specifications": pair_records,
        "pair_specifications_sha256": _canonical_sha256(pair_records),
        "pairs_source_path": None
        if args.pairs_json is None
        else str(args.pairs_json.expanduser().resolve()),
        "pairs_source_sha256": None
        if args.pairs_json is None
        else _sha256(args.pairs_json.expanduser().resolve()),
    }
    statistical_tests["provenance"] = provenance
    bootstrap_intervals["provenance"] = provenance
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and output_dir.is_symlink():
        raise ValueError(f"refusing symlink output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / "statistical_tests.json"
    intervals_path = output_dir / "bootstrap_intervals.json"
    existing = [str(path) for path in (statistics_path, intervals_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite statistics outputs: {existing}")
    _write_json_exclusive(statistics_path, statistical_tests)
    _write_json_exclusive(intervals_path, bootstrap_intervals)
    print(
        json.dumps(
            {
                "bootstrap_intervals": str(intervals_path),
                "bootstrap_draws": args.bootstrap_draws,
                "pair_count": len(pairs),
                "statistical_tests": str(statistics_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
