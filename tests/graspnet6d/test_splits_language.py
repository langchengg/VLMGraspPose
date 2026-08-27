"""Unit-only checks for scene splitting and derived language semantics."""

from __future__ import annotations

import pytest

from graspnet6d.language import (
    LanguageExpression,
    VisibleObject,
    assert_query_resolves_uniquely,
    generate_unique_queries_for_target,
    resolve_expression,
    resolve_query,
)
from graspnet6d.splits import (
    DEFAULT_SPLIT_SEED,
    SceneSplit,
    assert_scene_split_has_no_overlap,
    deterministic_scene_split,
    deterministic_stratified_scene_split,
    uniform_frame_ids,
    uniform_frame_indices,
)


def _visible_objects() -> tuple[VisibleObject, ...]:
    return (
        VisibleObject("banana-instance", "banana", (-0.20, 0.00, 0.80), (0.1, 0.1, 0.0)),
        VisibleObject("mug-instance", "mug", (0.00, 0.00, 1.00), (0.2, 0.1, 0.0)),
        VisibleObject("box-instance", "box", (0.20, 0.00, 1.20), (0.3, 0.1, 0.0)),
    )


def test_scene_split_has_no_overlap_and_is_order_independent() -> None:
    scenes = [f"scene_{index:04d}" for index in range(30)]
    first = deterministic_scene_split(scenes)
    second = deterministic_scene_split(list(reversed(scenes)))

    assert first == second
    assert first.seed == DEFAULT_SPLIT_SEED
    assert (len(first.train), len(first.validation), len(first.test)) == (18, 6, 6)
    assert set(first.all_scenes) == set(scenes)
    assert_scene_split_has_no_overlap(first)
    assert SceneSplit.from_dict(first.to_dict()) == first


def test_scene_split_enforces_minimum_validation_and_test_size() -> None:
    scenes = [f"scene_{index:04d}" for index in range(11)]
    split = deterministic_scene_split(scenes)

    assert len(split.train) == 1
    assert len(split.validation) == 5
    assert len(split.test) == 5
    with pytest.raises(ValueError, match="cannot provide"):
        deterministic_scene_split(scenes[:10])


def test_scene_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SceneSplit(train=("scene_0000",), validation=("scene_0000",), test=())


def test_fixed_stratified_split_is_order_independent_and_keeps_unused_scenes() -> None:
    scene_objects = {
        f"scene_{index:04d}": (index % 8, (index * 3) % 11)
        for index in range(50)
    }
    first, first_unused = deterministic_stratified_scene_split(scene_objects)
    second, second_unused = deterministic_stratified_scene_split(
        dict(reversed(list(scene_objects.items())))
    )

    assert first == second
    assert first_unused == second_unused
    assert (len(first.train), len(first.validation), len(first.test)) == (20, 5, 10)
    assert len(first_unused) == 15
    assert set(first.all_scenes).isdisjoint(first_unused)
    assert set(first.all_scenes) | set(first_unused) == set(scene_objects)


def test_fixed_stratified_split_rejects_too_few_or_invalid_scenes() -> None:
    with pytest.raises(ValueError, match="cannot provide"):
        deterministic_stratified_scene_split(
            {f"scene_{index:04d}": (index % 4,) for index in range(34)}
        )
    invalid = {f"scene_{index:04d}": (index % 4,) for index in range(35)}
    invalid["scene_0000"] = ()
    with pytest.raises(ValueError, match="invalid or empty"):
        deterministic_stratified_scene_split(invalid)


def test_uniform_sampling_spans_all_256_views() -> None:
    assert uniform_frame_indices(total_frames=256, frames_per_scene=16) == tuple(
        range(0, 256, 17)
    )
    assert uniform_frame_ids((10, 20, 30, 40, 50), frames_per_scene=3) == (10, 30, 50)


def test_language_query_resolves_uniquely() -> None:
    objects = _visible_objects()
    expression = LanguageExpression(
        "leftmost", {"catalog_name": "banana", "frame": "camera"}
    )
    query = resolve_query(expression, objects)

    assert query.query == "Grasp the leftmost banana."
    assert query.resolver_result == ("banana-instance",)
    assert query.is_unique
    assert query.provenance == "derived"
    assert_query_resolves_uniquely(query, "banana-instance")


def test_language_resolver_keeps_ties_ambiguous() -> None:
    objects = (
        VisibleObject("box-a", "box", (0.2, 0.0, 1.0), (0.0, 0.0, 0.0)),
        VisibleObject("box-b", "box", (0.2, 0.1, 1.1), (0.0, 0.0, 0.0)),
    )
    expression = LanguageExpression(
        "rightmost", {"catalog_name": "box", "frame": "camera"}
    )

    assert resolve_expression(expression, objects) == ("box-a", "box-b")
    assert not resolve_query(expression, objects).is_unique


def test_generated_language_never_embeds_target_id_as_tie_break() -> None:
    objects = _visible_objects()
    queries = generate_unique_queries_for_target(objects, "banana-instance")

    assert queries
    assert all(query.resolver_result == ("banana-instance",) for query in queries)
    assert all("banana-instance" not in query.query for query in queries)
    assert all("target_id" not in query.attributes for query in queries)
    assert all("object_id" not in query.attributes for query in queries)


def test_language_predicate_rejects_hidden_target_identifier() -> None:
    with pytest.raises(ValueError, match="hidden target"):
        LanguageExpression(
            "catalog_name", {"catalog_name": "banana", "target_object_id": 7}
        )


def test_relative_reference_must_be_observably_unique() -> None:
    objects = (
        VisibleObject("banana", "banana", (-0.2, 0.0, 0.8), (0.0, 0.0, 0.0)),
        VisibleObject("box-a", "box", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        VisibleObject("box-b", "box", (0.2, 0.0, 1.1), (0.0, 0.0, 0.0)),
    )
    expression = LanguageExpression(
        "left_of",
        {
            "catalog_name": "banana",
            "reference_catalog_name": "box",
            "frame": "camera",
        },
    )

    assert resolve_expression(expression, objects) == ()
