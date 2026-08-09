"""Machine-readable input loading and post-lock reporting statistics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reranking.statistics import (
    cluster_bootstrap_difference,
    holm_adjust,
    mcnemar_exact,
)


class ReportingContractError(ValueError):
    """Raised when reporting inputs violate the locked experiment contract."""


@dataclass(frozen=True)
class ReportingInputs:
    stage: Path
    validation_registry_path: Path
    test_registry_path: Path
    lock_path: Path
    validation_registry: pd.DataFrame
    test_registry: pd.DataFrame
    lock: dict[str, Any]
    primary_experiment_id: str
    baseline_name: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingContractError(f"cannot read {path}: {error}") from error


def _rows_from_json(value: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = None
        for key in ("records", "results", "experiments", "rows", "registry"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        if rows is None:
            rows = [value]
    else:
        raise ReportingContractError(f"{path} must contain JSON objects")
    if not all(isinstance(row, dict) for row in rows):
        raise ReportingContractError(f"{path} rows must be JSON objects")
    return [dict(row) for row in rows]


def read_machine_table(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read JSON/JSONL/CSV/TSV/Parquet without interpreting experiment results."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ReportingContractError(f"machine-readable input is not a regular file: {source}")
    suffix = source.suffix.lower()
    if suffix == ".json":
        return pd.DataFrame(_rows_from_json(_read_json(source), source))
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReportingContractError(
                    f"invalid JSONL at {source}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ReportingContractError(f"{source}:{line_number} is not an object")
            rows.append(row)
        return pd.DataFrame(rows)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(source, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ReportingContractError(f"unsupported machine-readable suffix: {source}")


def _find(stage: Path, candidates: tuple[str, ...], label: str) -> Path:
    for relative in candidates:
        path = stage / relative
        if path.is_file() and not path.is_symlink():
            return path
    raise ReportingContractError(
        f"missing required {label}; checked: {[str(stage / value) for value in candidates]}"
    )


def _experiment_id_column(frame: pd.DataFrame) -> str:
    for column in ("experiment_id", "run_id", "configuration_id", "method"):
        if column in frame.columns:
            return column
    raise ReportingContractError("registry has no experiment_id/run_id/configuration_id/method")


def _lock_value(lock: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if lock.get(key) not in (None, ""):
            return lock[key]
    selected = lock.get("selected")
    if isinstance(selected, dict):
        for key in aliases:
            if selected.get(key) not in (None, ""):
                return selected[key]
    return None


def load_reporting_inputs(stage: str | os.PathLike[str]) -> ReportingInputs:
    """Load registries and select primary exclusively from the pre-test lock."""

    root = Path(stage).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ReportingContractError(f"reporting stage is not a regular directory: {root}")
    lock_path = _find(
        root,
        (
            "manifests/PRIMARY_METHOD_LOCK.json",
            "PRIMARY_METHOD_LOCK.json",
            "manifests/primary_method_lock.json",
            "lock.json",
        ),
        "primary lock",
    )
    lock_value = _read_json(lock_path)
    if not isinstance(lock_value, dict):
        raise ReportingContractError("primary lock must be a JSON object")
    if lock_value.get("locked_before_test") is False or lock_value.get("test_used_for_selection") is True:
        raise ReportingContractError("primary lock admits test-based selection")
    primary = _lock_value(
        lock_value,
        ("primary_experiment_id", "selected_experiment_id", "experiment_id", "primary_method", "method"),
    )
    if primary in (None, ""):
        raise ReportingContractError("primary lock does not identify a primary experiment")
    baseline = _lock_value(
        lock_value,
        ("baseline_experiment_id", "baseline_name", "reference_method"),
    ) or "q_only"

    shared_registry = None
    for relative in (
        "metrics/experiment_registry.json",
        "metrics/experiment_registry.parquet",
        "experiment_registry.json",
    ):
        candidate = root / relative
        if candidate.is_file() and not candidate.is_symlink():
            shared_registry = candidate
            break
    validation_path = None
    test_path = None
    for stem, destination in (("validation", "validation"), ("test", "test")):
        found = None
        for relative in (
            f"metrics/{stem}_registry.json",
            f"metrics/{stem}_registry.csv",
            f"metrics/{stem}_registry.parquet",
            f"{stem}_registry.json",
        ):
            candidate = root / relative
            if candidate.is_file() and not candidate.is_symlink():
                found = candidate
                break
        if destination == "validation":
            validation_path = found
        else:
            test_path = found
    if validation_path is None or test_path is None:
        if shared_registry is None:
            raise ReportingContractError("validation/test registries are missing")
        shared = read_machine_table(shared_registry)
        split_col = next(
            (column for column in ("split", "stage", "partition") if column in shared),
            None,
        )
        if split_col is None:
            raise ReportingContractError("shared registry lacks split/stage/partition")
        split = shared[split_col].astype(str).str.lower()
        validation = shared.loc[split.str.contains("val|select|lockcheck")].copy()
        test = shared.loc[split.str.contains("test|formal")].copy()
        validation_path = validation_path or shared_registry
        test_path = test_path or shared_registry
    else:
        validation = read_machine_table(validation_path)
        test = read_machine_table(test_path)
    if validation.empty or test.empty:
        raise ReportingContractError("validation and test registries must both be non-empty")
    test_id_col = _experiment_id_column(test)
    test_ids = test[test_id_col].astype(str)
    if str(primary) not in set(test_ids):
        raise ReportingContractError(
            f"locked primary {primary!r} is absent from the test registry"
        )
    return ReportingInputs(
        stage=root,
        validation_registry_path=validation_path,
        test_registry_path=test_path,
        lock_path=lock_path,
        validation_registry=validation.reset_index(drop=True),
        test_registry=test.reset_index(drop=True),
        lock=lock_value,
        primary_experiment_id=str(primary),
        baseline_name=str(baseline),
    )


def _prediction_path(inputs: ReportingInputs, row: pd.Series) -> Path:
    raw = None
    for column in ("prediction_path", "predictions_path", "per_query_path", "outcomes_path"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            raw = str(row[column])
            break
    if raw is None:
        raise ReportingContractError(
            f"test registry row {row.to_dict()} does not declare a prediction path"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = inputs.stage / path
    path = Path(os.path.abspath(path))
    if path.is_symlink() or not path.is_file():
        raise ReportingContractError(f"prediction artifact is not a regular file: {path}")
    return path


def _binary(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ReportingContractError(f"{name} must contain binary values")
    return values.to_numpy(bool)


def load_query_outcomes(
    path: str | os.PathLike[str],
    *,
    expected_experiment_id: str | None = None,
) -> pd.DataFrame:
    """Load one row per query and normalize reporting-only field aliases."""

    frame = read_machine_table(path)
    aliases = {
        "query_id": "sample_id",
        "sequence_id": "scene_id",
        "q_only_correct": "baseline_correct",
        "reference_correct": "baseline_correct",
        "reranker_correct": "selected_correct",
        "challenger_correct": "selected_correct",
        "oracle_correct": "oracle",
        "q_only_candidate_id": "baseline_candidate_id",
        "reference_candidate_id": "baseline_candidate_id",
        "challenger_candidate_id": "selected_candidate_id",
        "selected_probability": "confidence",
    }
    frame = frame.rename(
        columns={old: new for old, new in aliases.items() if old in frame and new not in frame}
    ).copy()
    required = {
        "sample_id",
        "scene_id",
        "baseline_correct",
        "selected_correct",
        "oracle",
        "baseline_candidate_id",
        "selected_candidate_id",
        "candidate_count",
        "positive_count",
        "first_positive_rank",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ReportingContractError(f"query outcomes missing columns: {missing}")
    for column in ("sample_id", "scene_id"):
        if frame[column].isna().any():
            raise ReportingContractError(f"query outcomes {column} contains null values")
        frame[column] = frame[column].astype(str)
        if frame[column].eq("").any():
            raise ReportingContractError(f"query outcomes {column} contains empty values")
    if frame.duplicated("sample_id").any():
        raise ReportingContractError("query outcomes contain duplicate sample IDs")
    frame["baseline_correct"] = _binary(frame["baseline_correct"], "baseline_correct")
    frame["selected_correct"] = _binary(frame["selected_correct"], "selected_correct")
    frame["oracle"] = _binary(frame["oracle"], "oracle")
    if bool((frame["baseline_correct"] & ~frame["oracle"]).any()) or bool(
        (frame["selected_correct"] & ~frame["oracle"]).any()
    ):
        raise ReportingContractError("a selected success exceeds the frozen-pool oracle")
    if "baseline_oracle" in frame and "selected_oracle" in frame:
        if not _binary(frame["baseline_oracle"], "baseline_oracle").tolist() == _binary(
            frame["selected_oracle"], "selected_oracle"
        ).tolist():
            raise ReportingContractError("reranking changed the frozen-pool oracle")
    if expected_experiment_id is not None:
        for column in ("experiment_id", "configuration_id", "method"):
            if column in frame:
                observed = set(frame[column].astype(str))
                if observed != {str(expected_experiment_id)}:
                    raise ReportingContractError(
                        f"prediction identity differs from registry: {observed}"
                    )
                break
    if "frame_id" not in frame:
        frame["frame_id"] = frame["sample_id"]
    for column in ("candidate_count", "positive_count", "first_positive_rank"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or (frame[column] < 0).any():
            raise ReportingContractError(f"{column} must be non-negative numeric")
        if not np.allclose(frame[column], np.round(frame[column])):
            raise ReportingContractError(f"{column} must contain integers")
        frame[column] = frame[column].astype(int)
    if bool((frame["positive_count"] > frame["candidate_count"]).any()):
        raise ReportingContractError("positive_count exceeds candidate_count")
    has_positive = frame["positive_count"].gt(0)
    if not frame["oracle"].equals(has_positive):
        raise ReportingContractError("oracle disagrees with positive_count")
    if bool((has_positive & frame["first_positive_rank"].lt(1)).any()) or bool(
        (has_positive & frame["first_positive_rank"].gt(frame["candidate_count"])).any()
    ):
        raise ReportingContractError("first_positive_rank is invalid for a positive pool")
    if bool((~has_positive & frame["first_positive_rank"].ne(0)).any()):
        raise ReportingContractError("no-positive queries must use first_positive_rank=0")
    nonempty = frame["candidate_count"].gt(0)
    for column in ("baseline_candidate_id", "selected_candidate_id"):
        invalid = nonempty & (frame[column].isna() | frame[column].astype(str).eq(""))
        if bool(invalid.any()):
            raise ReportingContractError(f"non-empty queries require {column}")
        frame[column] = frame[column].fillna("").astype(str)
    return frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def primary_test_row(inputs: ReportingInputs) -> pd.Series:
    id_col = _experiment_id_column(inputs.test_registry)
    selected = inputs.test_registry.loc[
        inputs.test_registry[id_col].astype(str).eq(inputs.primary_experiment_id)
    ]
    if len(selected) != 1:
        raise ReportingContractError("test registry must contain exactly one locked-primary row")
    return selected.iloc[0]


def summarize_outcomes(frame: pd.DataFrame) -> dict[str, Any]:
    baseline = frame["baseline_correct"].to_numpy(bool)
    selected = frame["selected_correct"].to_numpy(bool)
    oracle = frame["oracle"].to_numpy(bool)
    recovered = (~baseline) & selected
    harmful = baseline & (~selected)
    switched = (
        frame["baseline_candidate_id"].astype(str).to_numpy()
        != frame["selected_candidate_id"].astype(str).to_numpy()
    )
    n = len(frame)
    headroom = int((oracle & ~baseline).sum())
    return {
        "sample_count": n,
        "baseline_correct": int(baseline.sum()),
        "selected_correct": int(selected.sum()),
        "oracle_correct": int(oracle.sum()),
        "baseline_j_at_1": float(baseline.mean()),
        "selected_j_at_1": float(selected.mean()),
        "oracle": float(oracle.mean()),
        "delta_j_at_1": float(selected.mean() - baseline.mean()),
        "recovered": int(recovered.sum()),
        "harmful": int(harmful.sum()),
        "net_recovered": int(recovered.sum() - harmful.sum()),
        "switch_count": int(switched.sum()),
        "headroom_count": headroom,
        "headroom_recovery": (
            0.0 if headroom == 0 else float((recovered.sum() - harmful.sum()) / headroom)
        ),
        "empty_count": int(frame["candidate_count"].eq(0).sum()),
        "no_positive_count": int(frame["positive_count"].eq(0).sum()),
        "rank_gt5_count": int(frame["first_positive_rank"].gt(5).sum()),
    }


def compute_reporting_statistics(
    inputs: ReportingInputs,
    *,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Recompute paired test statistics; never use test rows for selection."""

    id_col = _experiment_id_column(inputs.test_registry)
    raw_p: dict[str, float] = {}
    mcnemar_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    outcomes: dict[str, pd.DataFrame] = {}
    for index, row in inputs.test_registry.iterrows():
        experiment_id = str(row[id_col])
        path = _prediction_path(inputs, row)
        frame = load_query_outcomes(path, expected_experiment_id=experiment_id)
        paired = mcnemar_exact(frame["baseline_correct"], frame["selected_correct"])
        scene = cluster_bootstrap_difference(
            frame["baseline_correct"],
            frame["selected_correct"],
            frame["scene_id"],
            iterations=bootstrap_iterations,
            seed=seed + index,
        )
        raw_p[experiment_id] = float(paired["pvalue"])
        mcnemar_rows.append(
            {
                "experiment_id": experiment_id,
                "reference": inputs.baseline_name,
                "recovered": paired["recovered"],
                "harmful": paired["harmful"],
                "net_recovered": paired["net_recovered"],
                "discordant": paired["discordant"],
                "matched_odds_ratio": paired["matched_odds_ratio"],
                "raw_p": paired["pvalue"],
                "scipy_statsmodels_cross_check": paired["cross_check_passed"],
            }
        )
        bootstrap_rows.append(
            {
                "experiment_id": experiment_id,
                "unit": "scene",
                "point_estimate": scene["point_estimate"],
                "ci_lower": scene["ci95"][0],
                "ci_upper": scene["ci95"][1],
                "iterations": scene["iterations"],
                "cluster_count": scene["cluster_count"],
                "seed": scene["seed"],
            }
        )
        outcomes[experiment_id] = frame
    adjusted = holm_adjust(raw_p)
    holm_rows = [
        {
            "experiment_id": experiment_id,
            "raw_p": raw_p[experiment_id],
            "holm_adjusted_p": adjusted[experiment_id],
        }
        for experiment_id in raw_p
    ]
    primary = outcomes[inputs.primary_experiment_id]
    return {
        "selection_source": str(inputs.lock_path),
        "test_used_for_selection": False,
        "bootstrap_iterations": int(bootstrap_iterations),
        "mcnemar_rows": mcnemar_rows,
        "bootstrap_rows": bootstrap_rows,
        "holm_rows": holm_rows,
        "outcomes": outcomes,
        "primary_outcomes": primary,
        "primary_summary": summarize_outcomes(primary),
    }


def multi_seed_summary(registry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validation seeds only; test rows are never accepted here."""

    frame = registry.copy()
    if "seed" not in frame:
        frame["seed"] = 0
    grouping = [
        column
        for column in ("route", "pool", "configuration", "method", "experiment_family")
        if column in frame
    ]
    if not grouping:
        id_col = _experiment_id_column(frame)
        grouping = [id_col]
    excluded = set(grouping) | {"seed", "fold"}
    numeric_columns = []
    for column in frame.columns:
        if column in excluded:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            frame[column] = converted
            numeric_columns.append(column)
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(grouping, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(grouping, key_values, strict=True))
        row["seed_count"] = int(group["seed"].nunique())
        row["row_count"] = len(group)
        for column in numeric_columns:
            values = group[column].dropna().astype(float)
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_bins(
    outcomes: pd.DataFrame, *, bins: int = 10
) -> pd.DataFrame:
    if "confidence" not in outcomes:
        return pd.DataFrame(columns=["lower", "upper", "count", "confidence", "accuracy"])
    probability = pd.to_numeric(outcomes["confidence"], errors="coerce")
    valid = probability.notna() & probability.between(0.0, 1.0)
    probability = probability.loc[valid]
    truth = outcomes.loc[valid, "selected_correct"].astype(float)
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    for index in range(int(bins)):
        lower, upper = boundaries[index], boundaries[index + 1]
        members = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(members.sum()),
                "confidence": None if not members.any() else float(probability[members].mean()),
                "accuracy": None if not members.any() else float(truth[members].mean()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ReportingContractError",
    "ReportingInputs",
    "compute_reporting_statistics",
    "load_query_outcomes",
    "load_reporting_inputs",
    "multi_seed_summary",
    "primary_test_row",
    "read_machine_table",
    "reliability_bins",
    "summarize_outcomes",
]
