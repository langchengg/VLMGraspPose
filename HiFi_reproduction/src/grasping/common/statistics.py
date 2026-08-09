"""Pre-registered paired statistics for all-sample J@1 outcomes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


REQUIRED_COLUMNS = ("method", "sample_id", "scene_id", "j_at_1")
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260803


@dataclass(frozen=True, slots=True)
class PairSpec:
    pair_id: str
    method_a: str
    method_b: str
    family: str = "primary"

    def __post_init__(self) -> None:
        for name in ("pair_id", "method_a", "method_b", "family"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if self.method_a == self.method_b:
            raise ValueError("a paired comparison requires two distinct methods")


DEFAULT_PAIR_SPECS = (
    PairSpec("R0_vs_G1", "R0", "G1", "primary"),
    PairSpec("R0_vs_C1", "R0", "C1", "primary"),
    PairSpec("R0_vs_A0", "R0", "A0", "primary"),
    PairSpec("G1_vs_C1", "G1", "C1", "primary"),
    PairSpec("G1_vs_A0", "G1", "A0", "primary"),
    PairSpec("C1_vs_A0", "C1", "A0", "primary"),
    PairSpec("G0_vs_G1", "G0", "G1", "pretrained_vs_finetuned"),
    PairSpec("C0_vs_C1", "C0", "C1", "pretrained_vs_finetuned"),
    PairSpec("G0_vs_G0-O", "G0", "G0-O", "predicted_vs_oracle"),
    PairSpec("G1_vs_G1-O", "G1", "G1-O", "predicted_vs_oracle"),
    PairSpec("C0_vs_C0-O", "C0", "C0-O", "predicted_vs_oracle"),
    PairSpec("C1_vs_C1-O", "C1", "C1-O", "predicted_vs_oracle"),
    PairSpec("A0_vs_A0-O", "A0", "A0-O", "predicted_vs_oracle"),
)


@dataclass(frozen=True, slots=True)
class MethodOutcomes:
    method: str
    sample_ids: tuple[str, ...]
    scene_ids: tuple[str, ...]
    j_at_1: np.ndarray

    def __post_init__(self) -> None:
        outcomes = np.asarray(self.j_at_1, dtype=bool)
        if outcomes.ndim != 1 or len(outcomes) != len(self.sample_ids):
            raise ValueError("outcome vector length mismatch")
        if len(self.scene_ids) != len(self.sample_ids):
            raise ValueError("scene vector length mismatch")
        object.__setattr__(self, "j_at_1", outcomes)


def _strict_binary(values: pd.Series) -> np.ndarray:
    result: list[bool] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            result.append(bool(value))
        else:
            raise ValueError("j_at_1 must contain only non-null binary values")
    return np.asarray(result, dtype=bool)


def validate_and_align_predictions(
    table: pd.DataFrame,
) -> dict[str, MethodOutcomes]:
    """Require identical sample and scene coverage for every method.

    No row is filtered by ``non_empty`` or any other prediction property;
    empty predictions must be retained explicitly as ``j_at_1=False``.
    """

    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"per-sample predictions missing columns: {missing}")
    frame = table.loc[:, REQUIRED_COLUMNS].copy()
    if frame.empty:
        raise ValueError("per-sample predictions are empty")
    if frame.isna().any().any():
        columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"per-sample predictions contain nulls: {columns}")
    frame["method"] = frame["method"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["scene_id"] = frame["scene_id"].astype(str)
    if any(not value for value in frame["method"]):
        raise ValueError("method cannot be empty")
    if any(not value for value in frame["sample_id"]):
        raise ValueError("sample_id cannot be empty")
    if any(not value for value in frame["scene_id"]):
        raise ValueError("scene_id cannot be empty")
    duplicates = frame.duplicated(["method", "sample_id"], keep=False)
    if bool(duplicates.any()):
        examples = frame.loc[duplicates, ["method", "sample_id"]].head(5)
        raise ValueError(f"duplicate method/sample rows: {examples.to_dict('records')}")

    methods = sorted(frame["method"].unique().tolist())
    reference_ids: tuple[str, ...] | None = None
    reference_scenes: tuple[str, ...] | None = None
    aligned: dict[str, MethodOutcomes] = {}
    for method in methods:
        group = frame.loc[frame["method"] == method].sort_values(
            "sample_id", kind="mergesort"
        )
        sample_ids = tuple(group["sample_id"].tolist())
        scene_ids = tuple(group["scene_id"].tolist())
        if reference_ids is None:
            reference_ids, reference_scenes = sample_ids, scene_ids
        elif sample_ids != reference_ids:
            missing_ids = sorted(set(reference_ids) - set(sample_ids))
            extra_ids = sorted(set(sample_ids) - set(reference_ids))
            raise ValueError(
                f"method sample_id alignment failure for {method}: "
                f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
            )
        elif scene_ids != reference_scenes:
            mismatches = [
                sample_id
                for sample_id, observed, expected in zip(
                    sample_ids, scene_ids, reference_scenes, strict=True
                )
                if observed != expected
            ]
            raise ValueError(
                f"method scene_id alignment failure for {method}: {mismatches[:5]}"
            )
        aligned[method] = MethodOutcomes(
            method=method,
            sample_ids=sample_ids,
            scene_ids=scene_ids,
            j_at_1=_strict_binary(group["j_at_1"]),
        )
    return aligned


def load_aligned_predictions(path: str | Path) -> dict[str, MethodOutcomes]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty per-sample predictions: {source}")
    return validate_and_align_predictions(pd.read_parquet(source))


def exact_mcnemar(
    method_a: MethodOutcomes, method_b: MethodOutcomes, *, pair_id: str
) -> dict[str, Any]:
    """Exact two-sided McNemar test via Binomial(n_discordant, 0.5)."""

    if method_a.sample_ids != method_b.sample_ids:
        raise ValueError("McNemar inputs are not sample-aligned")
    if method_a.scene_ids != method_b.scene_ids:
        raise ValueError("McNemar inputs are not scene-aligned")
    left = method_a.j_at_1
    right = method_b.j_at_1
    both_success = int(np.count_nonzero(left & right))
    both_failure = int(np.count_nonzero(~left & ~right))
    a_only = int(np.count_nonzero(left & ~right))
    b_only = int(np.count_nonzero(~left & right))
    discordant = a_only + b_only
    p_value = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                b_only, n=discordant, p=0.5, alternative="two-sided"
            ).pvalue
        )
    )
    return {
        "pair_id": str(pair_id),
        "method_a": method_a.method,
        "method_b": method_b.method,
        "delta_j_at_1_b_minus_a": float(np.mean(right) - np.mean(left)),
        "sample_count": len(left),
        "both_success": both_success,
        "both_failure": both_failure,
        "method_a_only_success": a_only,
        "method_b_only_success": b_only,
        "discordant_count": discordant,
        "p_value_exact_two_sided": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Holm adjustment requires at least one p-value")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite in [0, 1]")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    adjusted = np.empty_like(values)
    running = 0.0
    family_size = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return [float(value) for value in adjusted]


def scene_clustered_paired_bootstrap(
    method_a: MethodOutcomes,
    method_b: MethodOutcomes,
    *,
    pair_id: str,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Percentile CI from paired scene clusters sampled with replacement."""

    if method_a.sample_ids != method_b.sample_ids:
        raise ValueError("bootstrap inputs are not sample-aligned")
    if method_a.scene_ids != method_b.scene_ids:
        raise ValueError("bootstrap inputs are not scene-aligned")
    if int(draws) <= 0:
        raise ValueError("bootstrap draws must be positive")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    scenes = np.asarray(method_a.scene_ids, dtype=object)
    unique_scenes, inverse = np.unique(scenes, return_inverse=True)
    if unique_scenes.size == 0:
        raise ValueError("bootstrap requires at least one scene")
    differences = method_b.j_at_1.astype(np.int8) - method_a.j_at_1.astype(np.int8)
    cluster_sums = np.bincount(inverse, weights=differences).astype(np.float64)
    cluster_counts = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(draws), dtype=np.float64)
    chunk_size = 2048
    for start in range(0, int(draws), chunk_size):
        stop = min(start + chunk_size, int(draws))
        sampled = rng.integers(
            0,
            len(unique_scenes),
            size=(stop - start, len(unique_scenes)),
            endpoint=False,
        )
        numerator = cluster_sums[sampled].sum(axis=1)
        denominator = cluster_counts[sampled].sum(axis=1)
        deltas[start:stop] = numerator / denominator
    tail = (1.0 - float(confidence_level)) * 0.5
    lower, upper = np.quantile(deltas, [tail, 1.0 - tail], method="linear")
    return {
        "pair_id": str(pair_id),
        "method_a": method_a.method,
        "method_b": method_b.method,
        "delta_definition": "method_b_j_at_1 minus method_a_j_at_1",
        "delta_j_at_1": float(np.mean(differences)),
        "confidence_level": float(confidence_level),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_draws": int(draws),
        "seed": int(seed),
        "sample_count": len(differences),
        "scene_count": len(unique_scenes),
        "resampling_unit": "scene_cluster_with_replacement",
        "cluster_contents_preserved": True,
    }


def pair_specs_from_json(path: str | Path) -> tuple[PairSpec, ...]:
    value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    rows = value.get("pairs") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("pairs JSON must be a non-empty list or {'pairs': [...]} object")
    result: list[PairSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"pairs[{index}] must be an object")
        required = {"method_a", "method_b"}
        if not required.issubset(row):
            raise ValueError(f"pairs[{index}] missing method_a/method_b")
        method_a, method_b = str(row["method_a"]), str(row["method_b"])
        result.append(
            PairSpec(
                pair_id=str(row.get("pair_id", f"{method_a}_vs_{method_b}")),
                method_a=method_a,
                method_b=method_b,
                family=str(row.get("family", "preregistered")),
            )
        )
    pair_ids = [item.pair_id for item in result]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_id values must be unique")
    return tuple(result)


def analyze_paired_methods(
    aligned: Mapping[str, MethodOutcomes],
    pairs: Sequence[PairSpec],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not pairs:
        raise ValueError("at least one pre-registered pair is required")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    missing_methods = sorted(
        {
            method
            for pair in pairs
            for method in (pair.method_a, pair.method_b)
            if method not in aligned
        }
    )
    if missing_methods:
        raise ValueError(f"pre-registered methods missing from predictions: {missing_methods}")
    pair_ids = [pair.pair_id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_id values must be unique")

    tests: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    for pair in pairs:
        left, right = aligned[pair.method_a], aligned[pair.method_b]
        test = exact_mcnemar(left, right, pair_id=pair.pair_id)
        test["family"] = pair.family
        tests.append(test)
        interval = scene_clustered_paired_bootstrap(
            left,
            right,
            pair_id=pair.pair_id,
            draws=bootstrap_draws,
            seed=seed,
        )
        interval["family"] = pair.family
        intervals.append(interval)
    adjusted = holm_adjust([row["p_value_exact_two_sided"] for row in tests])
    for row, adjusted_p in zip(tests, adjusted, strict=True):
        row["p_value_holm"] = adjusted_p
        row["reject_holm_at_alpha"] = bool(adjusted_p <= alpha)

    reference = next(iter(aligned.values()))
    alignment = {
        "methods": sorted(aligned),
        "sample_count_per_method": len(reference.sample_ids),
        "scene_count": len(set(reference.scene_ids)),
        "sample_id_alignment": "exact",
        "scene_id_alignment": "exact_per_sample",
        "analysis_population": "all_samples",
        "empty_prediction_policy": "retained_as_j_at_1_false; never filtered",
    }
    statistical_tests = {
        "schema_version": 1,
        "metric": "j_at_1_all_samples",
        "delta_definition": "method_b minus method_a",
        "alignment": alignment,
        "mcnemar": "exact_two_sided_binomial_on_discordant_pairs",
        "multiple_comparison": {
            "method": "Holm step-down",
            "family_size": len(tests),
            "alpha": float(alpha),
        },
        "tests": tests,
    }
    bootstrap_intervals = {
        "schema_version": 1,
        "metric": "delta_j_at_1_all_samples",
        "delta_definition": "method_b minus method_a",
        "alignment": alignment,
        "bootstrap_draws": int(bootstrap_draws),
        "seed": int(seed),
        "resampling_unit": "scene_cluster_with_replacement",
        "intervals": intervals,
    }
    return statistical_tests, bootstrap_intervals

