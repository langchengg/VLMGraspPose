from __future__ import annotations

import argparse

from tqdm import tqdm

from _common import (
    add_common_args,
    log_failure,
    make_index_builder,
    mapping_entries_for_samples,
    resolved_config,
    run_mapping_entry,
)
from main import TargetAwareGraspPipeline, write_summary_csv


def run_split(args: argparse.Namespace) -> int:
    config, dataset_root, output_root, camera, top_k = resolved_config(args)
    builder = make_index_builder(config)
    samples = builder.build_for_split(
        args.split,
        camera,
        dataset_root,
        output_root,
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
    )
    all_targets = not args.one_target_per_frame
    pairs = mapping_entries_for_samples(samples, config, output_root, all_targets=all_targets)
    pipeline = TargetAwareGraspPipeline(config)

    results = []
    for sample, entry in tqdm(pairs, desc=f"{args.split}/{camera}"):
        result = run_mapping_entry(pipeline, sample, entry, output_root, top_k, args.overwrite)
        log_failure(output_root, result)
        results.append(result)

    write_summary_csv(output_root / args.split / "summary.csv", results)
    write_summary_csv(output_root / "summary.csv", results)
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GraspNet split over language-target mapping entries.")
    add_common_args(parser)
    parser.add_argument("--split", required=True, choices=["train", "val", "test_seen", "test_similar", "test_novel"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all-targets-per-frame", action="store_true", default=True)
    parser.add_argument("--one-target-per-frame", action="store_true")
    args = parser.parse_args()
    if args.num_workers != 1:
        print("Mac-compatible prototype uses num-workers=1; ignoring higher values.")
        args.num_workers = 1
    processed = run_split(args)
    print(f"processed_units: {processed}")


if __name__ == "__main__":
    main()
