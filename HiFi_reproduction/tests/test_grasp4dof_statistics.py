"""Focused exact and clustered-bootstrap tests for the 4-DoF analysis."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.grasping.common.statistics import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_PAIR_SPECS,
    PairSpec,
    analyze_paired_methods,
    exact_mcnemar,
    holm_adjust,
    scene_clustered_paired_bootstrap,
    validate_and_align_predictions,
)
from tools.grasp4dof.run_statistics import main as run_statistics_main


def test_default_preregistered_pairs_and_draw_count_are_locked() -> None:
    pair_ids = {pair.pair_id for pair in DEFAULT_PAIR_SPECS}

    assert DEFAULT_BOOTSTRAP_DRAWS >= 10_000
    assert {
        "R0_vs_G1",
        "R0_vs_C1",
        "R0_vs_A0",
        "G1_vs_C1",
        "G1_vs_A0",
        "C1_vs_A0",
        "G0_vs_G1",
        "C0_vs_C1",
        "G0_vs_G0-O",
        "G1_vs_G1-O",
        "C0_vs_C0-O",
        "C1_vs_C1-O",
        "A0_vs_A0-O",
    } == pair_ids


def _table(
    outcomes: dict[str, list[bool]],
    *,
    scenes: list[str] | None = None,
) -> pd.DataFrame:
    sample_count = len(next(iter(outcomes.values())))
    sample_ids = [f"sample-{index:02d}" for index in range(sample_count)]
    scene_ids = scenes or [f"scene-{index // 2}" for index in range(sample_count)]
    return pd.DataFrame(
        [
            {
                "method": method,
                "sample_id": sample_id,
                "scene_id": scene_id,
                "j_at_1": value,
                "non_empty": bool(value),
            }
            for method, values in outcomes.items()
            for sample_id, scene_id, value in zip(
                sample_ids, scene_ids, values, strict=True
            )
        ]
    )


def test_exact_mcnemar_matches_known_two_sided_binomial_value() -> None:
    aligned = validate_and_align_predictions(
        _table(
            {
                "A": [True, True, True, True, True, True],
                "B": [False, False, False, False, False, True],
            }
        )
    )

    result = exact_mcnemar(aligned["A"], aligned["B"], pair_id="A_vs_B")

    assert result["method_a_only_success"] == 5
    assert result["method_b_only_success"] == 0
    assert result["discordant_count"] == 5
    assert result["p_value_exact_two_sided"] == pytest.approx(0.0625)
    assert result["delta_j_at_1_b_minus_a"] == pytest.approx(-5 / 6)


def test_holm_adjustment_is_step_down_monotone_in_sorted_p_order() -> None:
    raw = [0.04, 0.01, 0.03]

    adjusted = holm_adjust(raw)

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])
    ordered = sorted(zip(raw, adjusted))
    assert [item[1] for item in ordered] == sorted(item[1] for item in ordered)
    assert all(adjusted_p >= raw_p for raw_p, adjusted_p in zip(raw, adjusted))


def test_scene_clustered_bootstrap_is_deterministic_and_preserves_clusters() -> None:
    aligned = validate_and_align_predictions(
        _table(
            {
                "A": [False, False, True, True, False, True],
                "B": [True, True, False, True, False, False],
            },
            scenes=["large", "large", "small-a", "small-a", "small-b", "small-b"],
        )
    )

    first = scene_clustered_paired_bootstrap(
        aligned["A"], aligned["B"], pair_id="pair", draws=500, seed=17
    )
    second = scene_clustered_paired_bootstrap(
        aligned["A"], aligned["B"], pair_id="pair", draws=500, seed=17
    )

    assert first == second
    assert first["scene_count"] == 3
    assert first["sample_count"] == 6
    assert first["cluster_contents_preserved"] is True
    assert first["delta_j_at_1"] == pytest.approx(0.0)
    assert first["ci_lower"] <= first["delta_j_at_1"] <= first["ci_upper"]


@pytest.mark.parametrize("failure", ("missing_sample", "scene_mismatch"))
def test_alignment_fails_instead_of_using_unpaired_samples(failure: str) -> None:
    frame = _table({"A": [True, False, False], "B": [False, False, True]})
    if failure == "missing_sample":
        frame = frame.loc[
            ~((frame["method"] == "B") & (frame["sample_id"] == "sample-02"))
        ]
        message = "sample_id alignment failure"
    else:
        frame.loc[
            (frame["method"] == "B") & (frame["sample_id"] == "sample-01"),
            "scene_id",
        ] = "wrong-scene"
        message = "scene_id alignment failure"

    with pytest.raises(ValueError, match=message):
        validate_and_align_predictions(frame)


def test_duplicate_and_non_binary_rows_are_rejected() -> None:
    duplicate = _table({"A": [True, False], "B": [False, True]})
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate method/sample"):
        validate_and_align_predictions(duplicate)

    non_binary = _table({"A": [True, False], "B": [False, True]})
    non_binary["j_at_1"] = non_binary["j_at_1"].astype(object)
    non_binary.loc[0, "j_at_1"] = "False"
    with pytest.raises(ValueError, match="binary"):
        validate_and_align_predictions(non_binary)


def test_all_sample_analysis_retains_empty_prediction_failures() -> None:
    frame = _table({"A": [True, False, False], "B": [True, True, False]})
    frame.loc[frame["sample_id"] == "sample-02", "non_empty"] = False
    aligned = validate_and_align_predictions(frame)

    tests, intervals = analyze_paired_methods(
        aligned,
        [PairSpec("A_vs_B", "A", "B")],
        bootstrap_draws=100,
        seed=5,
    )

    assert tests["alignment"]["sample_count_per_method"] == 3
    assert tests["alignment"]["analysis_population"] == "all_samples"
    assert "never filtered" in tests["alignment"]["empty_prediction_policy"]
    assert intervals["intervals"][0]["sample_count"] == 3


def test_cli_reads_parquet_and_writes_both_machine_readable_outputs(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "per_sample_predictions.parquet"
    _table({"A": [True, False, False], "B": [True, True, False]}).to_parquet(
        predictions, index=False
    )
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "registered_A_vs_B",
                        "method_a": "A",
                        "method_b": "B",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "statistics"

    assert (
        run_statistics_main(
            [
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output_dir),
                "--pairs-json",
                str(pairs),
                "--bootstrap-draws",
                "100",
                "--seed",
                "9",
            ]
        )
        == 0
    )

    tests = json.loads((output_dir / "statistical_tests.json").read_text())
    intervals = json.loads((output_dir / "bootstrap_intervals.json").read_text())
    assert tests["tests"][0]["pair_id"] == "registered_A_vs_B"
    assert tests["tests"][0]["sample_count"] == 3
    assert intervals["bootstrap_draws"] == 100
    assert intervals["intervals"][0]["seed"] == 9
    expected_sha = hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert tests["provenance"] == intervals["provenance"]
    assert tests["provenance"]["predictions_path"] == str(predictions.resolve())
    assert tests["provenance"]["predictions_sha256"] == expected_sha
    assert tests["provenance"]["pair_specifications"][0]["pair_id"] == (
        "registered_A_vs_B"
    )
