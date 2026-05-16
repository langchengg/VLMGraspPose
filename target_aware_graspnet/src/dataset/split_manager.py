from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator


SPLIT_RANGES = {
    "train": (0, 89),
    "val": (90, 99),
    "test_seen": (100, 129),
    "test_similar": (130, 159),
    "test_novel": (160, 189),
}


@dataclass(frozen=True)
class SplitManager:
    split_ranges: dict[str, tuple[int, int]] = None

    def __post_init__(self):
        object.__setattr__(self, "split_ranges", self.split_ranges or SPLIT_RANGES)

    def validate_split(self, split_name: str) -> None:
        if split_name not in self.split_ranges:
            choices = ", ".join(self.split_ranges)
            raise ValueError(f"Invalid split '{split_name}'. Choose from: {choices}")

    def get_scene_ids(self, split_name: str) -> list[str]:
        self.validate_split(split_name)
        start, end = self.split_ranges[split_name]
        return [f"scene_{i:04d}" for i in range(start, end + 1)]

    def get_all_splits(self) -> dict[str, list[str]]:
        return {name: self.get_scene_ids(name) for name in self.split_ranges}

    def scene_to_split(self, scene_id: str) -> str:
        scene_num = parse_scene_id(scene_id)
        for split, (start, end) in self.split_ranges.items():
            if start <= scene_num <= end:
                return split
        raise ValueError(f"Scene id out of known GraspNet range: {scene_id}")

    def iter_scenes(self, split_name: str) -> Iterator[str]:
        yield from self.get_scene_ids(split_name)


def parse_scene_id(scene_id: str) -> int:
    m = re.fullmatch(r"scene_(\d{4})", scene_id)
    if not m:
        raise ValueError(
            f"Invalid scene id '{scene_id}'. Expected format like scene_0100."
        )
    return int(m.group(1))
