#!/usr/bin/env python3
"""Generate the frozen-candidate re-ranking qualitative gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.grasping.reranking_v1.gallery import (  # noqa: E402
    DEFAULT_GALLERY_QUOTAS,
    GalleryError,
    generate_failure_gallery,
)


def _quota(value: str) -> tuple[str, int]:
    try:
        category, raw_count = value.split("=", 1)
        count = int(raw_count)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "quota must use category=integer"
        ) from error
    if category not in DEFAULT_GALLERY_QUOTAS:
        raise argparse.ArgumentTypeError(
            f"unknown category {category!r}; "
            f"choose from {sorted(DEFAULT_GALLERY_QUOTAS)}"
        )
    if count < 0:
        raise argparse.ArgumentTypeError("quota cannot be negative")
    return category, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--hifi-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlm-results", type=Path)
    parser.add_argument("--rgb-root", type=Path)
    parser.add_argument("--predicted-mask-root", type=Path)
    parser.add_argument(
        "--quota",
        action="append",
        type=_quota,
        default=[],
        metavar="CATEGORY=COUNT",
        help="override a default quota; may be repeated",
    )
    parser.add_argument(
        "--allow-quota-shortfall",
        action="store_true",
        help="render all available cases and record deficits instead of failing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quotas = dict(DEFAULT_GALLERY_QUOTAS)
    quotas.update(dict(args.quota))
    try:
        result = generate_failure_gallery(
            per_candidate_path=args.per_candidate,
            per_sample_path=args.per_sample,
            predictions_path=args.predictions,
            method=args.method,
            hifi_manifest_path=args.hifi_manifest,
            candidate_root=args.candidate_root,
            annotation_file=args.annotations,
            evaluation_config_path=args.evaluation_config,
            output_dir=args.output_dir,
            vlm_results_path=args.vlm_results,
            rgb_root=args.rgb_root,
            predicted_mask_root=args.predicted_mask_root,
            quotas=quotas,
            strict_quotas=not args.allow_quota_shortfall,
        )
    except GalleryError as error:
        print(f"gallery generation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
