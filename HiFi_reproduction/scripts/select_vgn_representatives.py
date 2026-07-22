"""Select and synchronize deterministic representative VGN 3-D cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.representatives import (
    load_csv_rows,
    select_representatives,
    sync_representative_3d,
    write_representative_manifest,
)
from src.grasping.vgn_pipeline import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select deterministic representative samples for VGN 3-D rendering."
    )
    parser.add_argument("--ocid-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument(
        "--rendered-output",
        type=Path,
        help="After rendering, copy PLY diagnostics from this output into the main samples.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.ocid_output.expanduser().resolve()
    metrics = output / "metrics" / "per_sample.csv"
    if not metrics.is_file():
        raise FileNotFoundError(f"missing per-sample metrics: {metrics}")
    rows = load_csv_rows(metrics)
    selection = select_representatives(rows, count=args.count)
    report_dir = output / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_representative_manifest(
        args.manifest.expanduser().resolve(),
        report_dir / "representative_samples.jsonl",
        selection,
    )
    atomic_write_json(report_dir / "representative_selection.json", selection)
    result: dict[str, object] = {
        "status": "selected",
        "representative_count": len(selection),
        "representative_manifest": str(manifest),
        "selection_metadata": str(report_dir / "representative_selection.json"),
    }
    if args.rendered_output is not None:
        result["sync"] = sync_representative_3d(
            selection,
            rendered_output=args.rendered_output.expanduser().resolve(),
            experiment_output=output,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(json.dumps(run(args), indent=2, ensure_ascii=False))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "error", "reason": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
