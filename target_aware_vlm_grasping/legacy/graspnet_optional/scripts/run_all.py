from __future__ import annotations

import argparse

from _common import add_common_args
from run_split import run_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all GraspNet splits in order.")
    add_common_args(parser)
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

    total = 0
    for split in ["train", "val", "test_seen", "test_similar", "test_novel"]:
        args.split = split
        print(f"=== split: {split} ===")
        total += run_split(args)
    print(f"processed_units_total: {total}")


if __name__ == "__main__":
    main()
