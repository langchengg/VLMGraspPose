"""Deterministic scene-disjoint and frame-sampling contracts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_SPLIT_SEED = 20260815
DEFAULT_SPLIT_RATIOS = (0.60, 0.20, 0.20)
DEFAULT_FRAME_COUNT = 256


def _normalize_scene_ids(scene_ids: Sequence[str | int]) -> tuple[str, ...]:
    normalized = tuple(str(scene_id) for scene_id in scene_ids)
    if not normalized:
        raise ValueError("at least one scene is required")
    if any(not scene_id.strip() for scene_id in normalized):
        raise ValueError("scene IDs must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("scene IDs must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class SceneSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int = DEFAULT_SPLIT_SEED
    source_profile: str = "held-out GraspNet training-scene split"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("split seed must be an integer")
        if not self.source_profile.strip():
            raise ValueError("source_profile must be non-empty")
        for name in ("train", "validation", "test"):
            values = tuple(str(item) for item in getattr(self, name))
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate scene in {name} split")
            object.__setattr__(self, name, values)
        assert_scene_split_has_no_overlap(self)

    @property
    def all_scenes(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test

    def split_for(self, scene_id: str | int) -> str:
        normalized = str(scene_id)
        matches = [
            name
            for name in ("train", "validation", "test")
            if normalized in getattr(self, name)
        ]
        if len(matches) != 1:
            raise KeyError(f"scene {normalized!r} occurs in {len(matches)} splits")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "graspnet6d_scene_split_v1",
            "seed": self.seed,
            "source_profile": self.source_profile,
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneSplit":
        if value.get("schema_version") != "graspnet6d_scene_split_v1":
            raise ValueError("unsupported scene split schema")
        split = cls(
            train=tuple(value["train"]),
            validation=tuple(value["validation"]),
            test=tuple(value["test"]),
            seed=int(value["seed"]),
            source_profile=str(value["source_profile"]),
        )
        expected_counts = value.get("counts")
        if expected_counts != {
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        }:
            raise ValueError("scene split counts do not match the scene lists")
        return split


def assert_scene_split_has_no_overlap(split: SceneSplit) -> None:
    train = set(split.train)
    validation = set(split.validation)
    test = set(split.test)
    overlaps = {
        "train_validation": sorted(train & validation),
        "train_test": sorted(train & test),
        "validation_test": sorted(validation & test),
    }
    if any(overlaps.values()):
        raise ValueError(f"scene split overlap: {overlaps}")


def deterministic_scene_split(
    scene_ids: Sequence[str | int],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    min_validation_scenes: int = 5,
    min_test_scenes: int = 5,
    source_profile: str = "held-out GraspNet training-scene split",
) -> SceneSplit:
    """Hash-shuffle scenes and allocate deterministic 60/20/20-style splits."""

    scenes = _normalize_scene_ids(scene_ids)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split seed must be an integer")
    ratio_values = tuple(float(value) for value in ratios)
    if len(ratio_values) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in ratio_values
    ):
        raise ValueError("split ratios must be three finite positive values")
    if not math.isclose(sum(ratio_values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to one")
    for name, value in (
        ("min_validation_scenes", min_validation_scenes),
        ("min_test_scenes", min_test_scenes),
    ):
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    minimum_total = int(min_validation_scenes) + int(min_test_scenes) + 1
    if len(scenes) < minimum_total:
        raise ValueError(
            f"{len(scenes)} scenes cannot provide one train, "
            f"{min_validation_scenes} validation, and {min_test_scenes} test scenes"
        )

    def order_key(scene_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}\0{scene_id}".encode("utf-8")).hexdigest()
        return digest, scene_id

    ordered = tuple(sorted(scenes, key=order_key))
    count = len(ordered)
    validation_count = max(int(min_validation_scenes), int(round(count * ratio_values[1])))
    test_count = max(int(min_test_scenes), int(round(count * ratio_values[2])))
    if validation_count + test_count >= count:
        # This can only occur for non-default constraints after the earlier
        # minimum check; preserve at least one training scene deterministically.
        overflow = validation_count + test_count - (count - 1)
        while overflow > 0 and (
            validation_count > min_validation_scenes or test_count > min_test_scenes
        ):
            if validation_count >= test_count and validation_count > min_validation_scenes:
                validation_count -= 1
            elif test_count > min_test_scenes:
                test_count -= 1
            overflow -= 1
    train_count = count - validation_count - test_count
    if train_count < 1:
        raise ValueError("split constraints leave no training scene")
    split = SceneSplit(
        train=ordered[:train_count],
        validation=ordered[train_count : train_count + validation_count],
        test=ordered[train_count + validation_count :],
        seed=seed,
        source_profile=source_profile,
    )
    if set(split.all_scenes) != set(scenes):
        raise AssertionError("internal error: split does not cover the exact scene universe")
    return split


def deterministic_stratified_scene_split(
    scene_object_ids: Mapping[str | int, Sequence[int]],
    *,
    train_count: int = 20,
    validation_count: int = 5,
    test_count: int = 10,
    seed: int = DEFAULT_SPLIT_SEED,
    source_profile: str = "compact GraspNet training-scene split",
) -> tuple[SceneSplit, tuple[str, ...]]:
    """Allocate fixed scene counts while balancing multi-label object presence.

    Scenes are ordered by object rarity and a seeded SHA-256 tie-break. Each
    scene is greedily assigned to the still-open partition with the largest
    aggregate deficit for its object IDs. Extra scenes enter an explicit
    ``unused`` bin, so selecting 35 scenes cannot favour model outputs.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split seed must be an integer")
    counts = {
        "train": int(train_count),
        "validation": int(validation_count),
        "test": int(test_count),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("fixed train/validation/test counts must be positive")
    normalised: dict[str, tuple[int, ...]] = {}
    for raw_scene, raw_objects in scene_object_ids.items():
        scene = str(raw_scene)
        if not scene.strip() or scene in normalised:
            raise ValueError(
                "scene IDs must be non-empty and unique after string conversion"
            )
        objects = tuple(sorted({int(value) for value in raw_objects}))
        if not objects or any(value < 0 or value > 87 for value in objects):
            raise ValueError(
                f"scene {scene!r} has invalid or empty GraspNet object IDs"
            )
        normalised[scene] = objects
    required = sum(counts.values())
    if len(normalised) < required:
        raise ValueError(
            f"{len(normalised)} scenes cannot provide the fixed {required}-scene split"
        )
    capacities = {**counts, "unused": len(normalised) - required}
    frequencies: dict[int, int] = {}
    for objects in normalised.values():
        for object_id in objects:
            frequencies[object_id] = frequencies.get(object_id, 0) + 1

    def scene_order(scene: str) -> tuple[float, str, str]:
        rarity = sum(1.0 / frequencies[value] for value in normalised[scene])
        digest = hashlib.sha256(f"{seed}\0scene\0{scene}".encode()).hexdigest()
        return -rarity, digest, scene

    ordered = sorted(normalised, key=scene_order)
    desired = {
        partition: {
            object_id: frequency * capacity / len(normalised)
            for object_id, frequency in frequencies.items()
        }
        for partition, capacity in capacities.items()
    }
    assigned: dict[str, list[str]] = {name: [] for name in capacities}
    observed = {
        name: {object_id: 0 for object_id in frequencies} for name in capacities
    }
    for scene in ordered:
        objects = normalised[scene]
        candidates = [
            name
            for name, capacity in capacities.items()
            if len(assigned[name]) < capacity
        ]
        if not candidates:  # pragma: no cover - capacities cover the universe
            raise AssertionError("internal error: no split capacity remains")

        def partition_key(name: str) -> tuple[float, float, str]:
            object_deficit = sum(
                max(0.0, desired[name][object_id] - observed[name][object_id])
                for object_id in objects
            )
            capacity_deficit = capacities[name] - len(assigned[name])
            tie = hashlib.sha256(
                f"{seed}\0partition\0{name}\0{scene}".encode()
            ).hexdigest()
            return -object_deficit, -float(capacity_deficit), tie

        chosen = min(candidates, key=partition_key)
        assigned[chosen].append(scene)
        for object_id in objects:
            observed[chosen][object_id] += 1

    split = SceneSplit(
        train=tuple(sorted(assigned["train"])),
        validation=tuple(sorted(assigned["validation"])),
        test=tuple(sorted(assigned["test"])),
        seed=seed,
        source_profile=source_profile,
    )
    unused = tuple(sorted(assigned["unused"]))
    if (
        len(split.train) != train_count
        or len(split.validation) != validation_count
        or len(split.test) != test_count
    ):
        raise AssertionError("internal error: fixed split counts were not satisfied")
    if set(split.all_scenes) | set(unused) != set(normalised):
        raise AssertionError("internal error: stratified split lost a scene")
    return split, unused


def uniform_frame_indices(
    *,
    total_frames: int = DEFAULT_FRAME_COUNT,
    frames_per_scene: int = 16,
) -> tuple[int, ...]:
    """Sample integer frame IDs uniformly across the full ordered view range."""

    for name, value in (
        ("total_frames", total_frames),
        ("frames_per_scene", frames_per_scene),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if frames_per_scene > total_frames:
        raise ValueError("cannot sample more unique frames than are available")
    if frames_per_scene == 1:
        return (0,)
    last = total_frames - 1
    denominator = frames_per_scene - 1
    # Exact integer nearest-neighbour rounding avoids NumPy-version dependence.
    indices = tuple(
        (2 * index * last + denominator) // (2 * denominator)
        for index in range(frames_per_scene)
    )
    if len(set(indices)) != frames_per_scene:
        raise AssertionError("internal error: uniform sampling produced duplicate frames")
    return indices


def uniform_frame_ids(
    available_frame_ids: Sequence[int], *, frames_per_scene: int
) -> tuple[int, ...]:
    values = tuple(sorted(int(frame_id) for frame_id in available_frame_ids))
    if len(values) != len(set(values)):
        raise ValueError("available frame IDs must be unique")
    positions = uniform_frame_indices(
        total_frames=len(values), frames_per_scene=frames_per_scene
    )
    return tuple(values[position] for position in positions)
