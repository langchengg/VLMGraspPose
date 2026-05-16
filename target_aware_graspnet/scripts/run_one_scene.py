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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline on one GraspNet scene.")
    add_common_args(parser)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--all-targets-per-frame", action="store_true", default=True)
    parser.add_argument("--one-target-per-frame", action="store_true")
    args = parser.parse_args()

    config, dataset_root, output_root, camera, top_k = resolved_config(args)
    builder = make_index_builder(config)
    samples = builder.build_for_scene(args.scene_id, camera, dataset_root, output_root, max_frames=args.max_frames)
    all_targets = not args.one_target_per_frame
    pairs = mapping_entries_for_samples(samples, config, output_root, all_targets=all_targets)
    pipeline = TargetAwareGraspPipeline(config)

    results = []
    for sample, entry in tqdm(pairs, desc=f"{args.scene_id}/{camera}"):
        result = run_mapping_entry(pipeline, sample, entry, output_root, top_k, args.overwrite)
        log_failure(output_root, result)
        results.append(result)

    split = samples[0].split if samples else "unknown"
    write_summary_csv(output_root / split / args.scene_id / camera / "summary.csv", results)
    write_summary_csv(output_root / "summary.csv", results)
    print(f"processed_units: {len(results)}")


if __name__ == "__main__":
    main()
