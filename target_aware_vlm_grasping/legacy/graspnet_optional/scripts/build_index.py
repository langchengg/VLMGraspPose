from __future__ import annotations

import argparse

from _common import add_common_args, make_index_builder, mapping_entries_for_samples, resolved_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GraspNet sample discovery and optional object-language mappings.")
    add_common_args(parser)
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--build-mapping", action="store_true")
    parser.add_argument("--all-targets-per-frame", action="store_true")
    parser.add_argument("--one-target-per-frame", action="store_true")
    args = parser.parse_args()

    config, dataset_root, output_root, camera, _ = resolved_config(args)
    builder = make_index_builder(config)
    samples = builder.build_for_split(
        args.split,
        camera,
        dataset_root,
        output_root,
        max_scenes=args.max_scenes,
        max_frames=args.max_frames,
    )
    scenes = sorted({sample.scene_id for sample in samples})
    print(f"dataset_root: {dataset_root}")
    print(f"output_root: {output_root}")
    print(f"split: {args.split}")
    print(f"camera: {camera}")
    print(f"scenes: {len(scenes)}")
    print(f"frames: {len(samples)}")
    if samples:
        first = samples[0]
        print(f"first_rgb: {first.rgb_path}")
        print(f"first_depth: {first.depth_path}")
        print(f"first_label: {first.label_path}")

    if args.build_mapping:
        all_targets = args.all_targets_per_frame or not args.one_target_per_frame
        pairs = mapping_entries_for_samples(samples, config, output_root, all_targets=all_targets)
        print(f"mapping_entries: {len(pairs)}")
        print(f"mapping_csv: {output_root / 'mappings' / 'object_language_mapping.csv'}")


if __name__ == "__main__":
    main()
