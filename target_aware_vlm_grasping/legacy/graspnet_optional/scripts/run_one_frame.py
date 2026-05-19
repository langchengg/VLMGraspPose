from __future__ import annotations

import argparse

from _common import (
    add_common_args,
    log_failure,
    make_index_builder,
    mapping_entries_for_samples,
    resolved_config,
    run_mapping_entry,
)
from main import TargetAwareGraspPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the target-aware RGB-D grasping pipeline on one GraspNet frame.")
    add_common_args(parser)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument("--all-targets-per-frame", action="store_true")
    parser.add_argument("--one-target-per-frame", action="store_true")
    args = parser.parse_args()

    config, dataset_root, output_root, camera, top_k = resolved_config(args)
    builder = make_index_builder(config)
    samples = builder.build_for_scene(args.scene_id, camera, dataset_root, output_root)
    samples = [sample for sample in samples if sample.frame_id == args.frame_id]
    if not samples:
        raise SystemExit(f"No sample found for {args.scene_id}/{camera}/{args.frame_id}.")

    all_targets = bool(args.all_targets_per_frame or args.target_id is not None)
    if args.one_target_per_frame:
        all_targets = False
    pairs = mapping_entries_for_samples(samples, config, output_root, all_targets=all_targets)
    if args.target_id is not None:
        pairs = [(sample, entry) for sample, entry in pairs if entry.target_id == args.target_id]
    if not pairs:
        raise SystemExit("No target mapping entry found for this frame.")

    pipeline = TargetAwareGraspPipeline(config)
    for sample, entry in pairs:
        result = run_mapping_entry(pipeline, sample, entry, output_root, top_k, args.overwrite)
        log_failure(output_root, result)
        print(f"{entry.scene_id}/{entry.camera}/{entry.frame_id}/target_{entry.target_id:03d}: {result.status}")
        if result.best_grasp is not None:
            c = result.best_grasp.candidate
            print(f"  command: {entry.command}")
            print(f"  position: {c.position.tolist()}")
            print(f"  orientation: {c.orientation.tolist()}")
            print(f"  approach: {c.approach_vector.tolist()}")
            print(f"  closing: {c.closing_direction.tolist()}")
            print(f"  width: {c.gripper_width:.4f}")
            print(f"  final_score: {result.best_grasp.final_score:.4f}")
            print(f"  output_dir: {result.sample.output_dir}")
        elif result.error_message:
            print(f"  error: {result.error_message}")


if __name__ == "__main__":
    main()
