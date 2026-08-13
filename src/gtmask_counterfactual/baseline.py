"""Label-safe replay of frozen post-formal baseline outcomes."""

from __future__ import annotations

import math
import re
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .contracts import BASELINE_TARGETS, BaselineTarget
from .io import sha256_file


class BaselineReplayMismatch(RuntimeError):
    """Raised when recomputed integer metrics differ from frozen authorities."""

    def __init__(self, mismatches: Mapping[str, Mapping[str, tuple[Any, Any]]]):
        self.mismatches = {route: dict(values) for route, values in mismatches.items()}
        super().__init__(f"baseline replay mismatch: {self.mismatches}")


_RAW_TEST_PATTERN = re.compile(
    r"(^|_)(raw_?test|ground_?truth|gt_?mask|gt_?grasp|target_?mask)(_|$)",
    re.IGNORECASE,
)
_SAFE_DERIVED_COLUMNS = {
    "candidate_success",
    "selected_correct",
    "first_positive_rank",
    "best_same_gt_iou",
    "best_same_gt_angle_error_deg",
    "matched_gt_index",
}
_REPLAY_COLUMNS = {
    "system_name",
    "sample_id",
    "selected_correct",
    "candidate_count",
    "first_positive_rank",
    "candidate_success",
    "rank",
    "effective_rank",
    "no_output",
    "row_kind",
}
_FULL_POOL_COLUMNS = {
    "route",
    "sample_id",
    "candidate_count_all",
    "candidate_count_top5",
    "full_pool_positive",
    "top5_positive",
    "first_positive_rank",
    "native_correct",
    "gated_correct",
}


def _verify_locked_source_artifact(source: Path, root: Path) -> None:
    """Require the opened derived artifact to be in the source final inventory."""

    lock_path = root / "FINAL_RUN_LOCK.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("source final lock is unreadable") from error
    inventory = lock.get("inventory") if isinstance(lock, dict) else None
    if not isinstance(inventory, list):
        raise RuntimeError("source final lock has no inventory")
    relative = str(source.relative_to(root))
    matches = [
        row
        for row in inventory
        if isinstance(row, Mapping) and row.get("relative_path") == relative
    ]
    if len(matches) != 1:
        raise PermissionError(f"derived source artifact is not uniquely locked: {relative}")
    record = matches[0]
    if (
        Path(str(record.get("path", ""))).expanduser().resolve() != source
        or record.get("sha256") != sha256_file(source)
        or record.get("bytes") != source.stat().st_size
    ):
        raise RuntimeError(f"derived source artifact differs from final lock: {relative}")


def assert_no_raw_test_supervision(columns: list[str] | tuple[str, ...]) -> None:
    forbidden = sorted(
        column
        for column in map(str, columns)
        if column not in _SAFE_DERIVED_COLUMNS and _RAW_TEST_PATTERN.search(column)
    )
    if forbidden:
        raise PermissionError(f"raw Test label/GT columns are forbidden: {forbidden}")


def load_frozen_formal_parquet(
    path: str | Path,
    *,
    source_run: str | Path,
) -> pd.DataFrame:
    """Open only a frozen 09_formal_test derived artifact after schema preflight."""

    source = Path(path).expanduser().resolve()
    root = Path(source_run).expanduser().resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise PermissionError(
            "baseline artifact is outside its frozen source run"
        ) from error
    if not relative.parts or relative.parts[0] != "09_formal_test":
        raise PermissionError("baseline replay may open only 09_formal_test artifacts")
    if source.is_symlink() or not source.is_file() or source.suffix != ".parquet":
        raise ValueError(f"baseline artifact must be a regular Parquet file: {source}")
    _verify_locked_source_artifact(source, root)
    columns = tuple(map(str, pq.ParquetFile(source).schema_arrow.names))
    assert_no_raw_test_supervision(columns)
    selected = [column for column in columns if column in _REPLAY_COLUMNS]
    return pd.read_parquet(source, columns=selected)


def load_frozen_full_pool_taxonomy(
    path: str | Path,
    *,
    source_run: str | Path,
) -> pd.DataFrame:
    """Load the locked label-derived AllNMS summary without opening raw GT rows."""

    source = Path(path).expanduser().resolve()
    root = Path(source_run).expanduser().resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise PermissionError("full-pool summary is outside its frozen source run") from error
    if relative != Path("13_failure_galleries/failure_taxonomy_per_sample.parquet"):
        raise PermissionError("unexpected full-pool summary path")
    if source.is_symlink() or not source.is_file():
        raise ValueError("full-pool summary must be a regular non-symlink file")
    _verify_locked_source_artifact(source, root)
    all_columns = tuple(map(str, pq.ParquetFile(source).schema_arrow.names))
    assert_no_raw_test_supervision(all_columns)
    columns = set(all_columns)
    missing = sorted(_FULL_POOL_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"full-pool summary lacks required columns: {missing}")
    return pd.read_parquet(source, columns=sorted(_FULL_POOL_COLUMNS))


def _truth(value: Any) -> bool:
    return False if pd.isna(value) else bool(value)


def _rank(value: Any) -> float:
    if value is None or pd.isna(value):
        return math.inf
    observed = float(value)
    return observed if observed >= 1.0 and math.isfinite(observed) else math.inf


def _collapse_per_sample(frame: pd.DataFrame, system: str) -> pd.DataFrame:
    required = {"system_name", "sample_id", "selected_correct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frozen outcome table lacks required columns: {missing}")
    assert_no_raw_test_supervision(tuple(map(str, frame.columns)))
    rows = frame.loc[frame["system_name"].astype(str) == system].copy()
    if rows.empty:
        raise ValueError(f"frozen outcome system is absent: {system}")
    if "first_positive_rank" in rows.columns:
        grouped_rows = []
        for sample_id, sample in rows.groupby("sample_id", sort=True, dropna=False):
            selected = {_truth(value) for value in sample["selected_correct"]}
            if len(selected) != 1:
                raise ValueError(
                    f"selected_correct is inconsistent: {system}/{sample_id}"
                )
            ranks = [_rank(value) for value in sample["first_positive_rank"]]
            finite = [value for value in ranks if math.isfinite(value)]
            candidate_count = (
                int(sample["candidate_count"].max())
                if "candidate_count" in sample.columns
                else int(sample.shape[0])
            )
            no_output = (
                bool(sample["no_output"].map(_truth).all())
                if "no_output" in sample.columns
                else candidate_count == 0
            )
            grouped_rows.append(
                {
                    "sample_id": str(sample_id),
                    "selected_correct": selected.pop(),
                    "first_positive_rank": min(finite, default=math.inf),
                    "candidate_count": candidate_count,
                    "no_output": no_output,
                }
            )
        return pd.DataFrame(grouped_rows)
    rank_column = "effective_rank" if "effective_rank" in rows.columns else "rank"
    required_candidate = {"candidate_success", rank_column}
    missing_candidate = sorted(required_candidate.difference(rows.columns))
    if missing_candidate:
        raise ValueError(
            "frozen outcomes need first_positive_rank or candidate_success/rank: "
            f"{missing_candidate}"
        )
    grouped_rows = []
    for sample_id, sample in rows.groupby("sample_id", sort=True, dropna=False):
        selected = {_truth(value) for value in sample["selected_correct"]}
        if len(selected) != 1:
            raise ValueError(f"selected_correct is inconsistent: {system}/{sample_id}")
        positive_ranks = [
            _rank(rank)
            for rank, success in zip(
                sample[rank_column], sample["candidate_success"], strict=True
            )
            if _truth(success)
        ]
        no_output = (
            bool(sample["no_output"].map(_truth).all())
            if "no_output" in sample.columns
            else sample[rank_column].map(_rank).map(math.isinf).all()
        )
        grouped_rows.append(
            {
                "sample_id": str(sample_id),
                "selected_correct": selected.pop(),
                "first_positive_rank": min(positive_ranks, default=math.inf),
                "candidate_count": int(sample.shape[0]),
                "no_output": no_output,
            }
        )
    return pd.DataFrame(grouped_rows)


def _summarise(native: pd.DataFrame, final: pd.DataFrame) -> dict[str, int]:
    if set(native["sample_id"]) != set(final["sample_id"]):
        raise ValueError("native/final sample IDs are not exactly aligned")
    rank = native["first_positive_rank"].astype(float)
    no_output = native["no_output"].astype(bool)
    return {
        "sample_count": int(native.shape[0]),
        "native_correct": int(native["selected_correct"].astype(bool).sum()),
        "oracle_top5": int((rank <= 5).sum()),
        "oracle_all": int(rank.map(math.isfinite).sum()),
        "no_output": int(no_output.sum()),
        "no_positive_full_pool_excluding_no_output": int(
            ((~rank.map(math.isfinite)) & (~no_output)).sum()
        ),
        "no_positive_full_pool_including_no_output": int(
            (~rank.map(math.isfinite)).sum()
        ),
        "positive_only_below_top5": int(((rank > 5) & rank.map(math.isfinite)).sum()),
        "final_correct": int(final["selected_correct"].astype(bool).sum()),
    }


def _apply_full_pool_summary(
    result: dict[str, int],
    full_pool: pd.DataFrame,
    *,
    route: str,
    native: pd.DataFrame,
    final: pd.DataFrame,
) -> None:
    rows = full_pool.loc[full_pool["route"].astype(str).str.lower() == route].copy()
    sample_ids = set(native["sample_id"].astype(str))
    if rows.shape[0] != len(sample_ids) or rows["sample_id"].astype(str).nunique() != len(
        sample_ids
    ):
        raise ValueError(f"{route} full-pool summary is not one row per sample")
    if set(rows["sample_id"].astype(str)) != sample_ids:
        raise ValueError(f"{route} full-pool summary sample IDs differ")
    for column in ("full_pool_positive", "top5_positive", "native_correct", "gated_correct"):
        if not rows[column].isin([True, False, 0, 1]).all():
            raise ValueError(f"{route} full-pool {column} is not binary")
    if (rows["candidate_count_all"] < rows["candidate_count_top5"]).any():
        raise ValueError(f"{route} AllNMS candidate count is below Top-5 count")
    if (rows["top5_positive"].astype(bool) & ~rows["full_pool_positive"].astype(bool)).any():
        raise ValueError(f"{route} Top-5 positive is absent from AllNMS")
    formal = native.merge(
        final[["sample_id", "selected_correct"]].rename(
            columns={"selected_correct": "final_selected_correct"}
        ),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    aligned = formal.merge(
        rows.rename(columns={"first_positive_rank": "taxonomy_first_positive_rank"}),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("sample_id", kind="mergesort")
    disagreements = {
        "native_correct": aligned["selected_correct"].astype(bool)
        != aligned["native_correct"].astype(bool),
        "gated_correct": aligned["final_selected_correct"].astype(bool)
        != aligned["gated_correct"].astype(bool),
        "top5_positive": aligned["first_positive_rank"].map(math.isfinite)
        != aligned["top5_positive"].astype(bool),
        "candidate_count_top5": aligned["candidate_count"].astype(int)
        != aligned["candidate_count_top5"].astype(int),
    }
    bad = {name: int(mask.sum()) for name, mask in disagreements.items() if mask.any()}
    if bad:
        raise ValueError(f"{route} formal and full-pool per-sample rows disagree: {bad}")
    frozen_top5 = int(rows["top5_positive"].astype(bool).sum())
    frozen_native = int(rows["native_correct"].astype(bool).sum())
    frozen_final = int(rows["gated_correct"].astype(bool).sum())
    if (
        frozen_top5 != result["oracle_top5"]
        or frozen_native != result["native_correct"]
        or frozen_final != result["final_correct"]
    ):
        raise ValueError(f"{route} formal and full-pool summaries disagree")
    full_positive = rows["full_pool_positive"].astype(bool)
    top5_positive = rows["top5_positive"].astype(bool)
    no_output = rows["candidate_count_all"].astype(int).eq(0)
    result.update(
        {
            "oracle_all": int(full_positive.sum()),
            "no_output": int(no_output.sum()),
            "no_positive_full_pool_excluding_no_output": int(
                ((~full_positive) & (~no_output)).sum()
            ),
            "no_positive_full_pool_including_no_output": int((~full_positive).sum()),
            "positive_only_below_top5": int((full_positive & ~top5_positive).sum()),
        }
    )


def _compare(
    actual: Mapping[str, Mapping[str, int]],
    targets: Mapping[str, BaselineTarget],
) -> None:
    mismatches: dict[str, dict[str, tuple[Any, Any]]] = {}
    for route, target in targets.items():
        observed = actual[route]
        expected = {
            "sample_count": target.sample_count,
            "native_correct": target.native_correct,
            "oracle_top5": target.oracle_top5,
            "oracle_all": target.oracle_all,
            "no_output": target.no_output,
            "no_positive_full_pool": target.no_positive_full_pool,
            "positive_only_below_top5": target.positive_only_below_top5,
            "final_correct": target.final_correct,
        }
        observed_comparable = {
            **observed,
            "no_positive_full_pool": observed[
                "no_positive_full_pool_including_no_output"
                if target.no_positive_includes_no_output
                else "no_positive_full_pool_excluding_no_output"
            ],
        }
        if target.oracle_top10 is not None:
            expected["oracle_top10"] = target.oracle_top10
        if target.top10_selected_correct is not None:
            expected["top10_selected_correct"] = target.top10_selected_correct
        if target.all_selected_correct is not None:
            expected["all_selected_correct"] = target.all_selected_correct
        for key, expected_value in expected.items():
            observed_value = observed_comparable.get(key)
            if observed_value != expected_value:
                mismatches.setdefault(route, {})[key] = (
                    observed_value,
                    expected_value,
                )
    if mismatches:
        raise BaselineReplayMismatch(mismatches)


def replay_baselines(
    unified_outcomes: pd.DataFrame,
    d1_outcomes: pd.DataFrame,
    *,
    unified_full_pool: pd.DataFrame | None = None,
    targets: Mapping[str, BaselineTarget] = BASELINE_TARGETS,
) -> dict[str, Any]:
    """Recompute G1/C1/D1 baselines from frozen derived per-sample outcomes."""

    results: dict[str, dict[str, int]] = {}
    for route in ("g1", "c1"):
        native = _collapse_per_sample(unified_outcomes, f"{route}_native")
        final = _collapse_per_sample(unified_outcomes, f"{route}_gated_primary")
        results[route] = _summarise(
            native,
            final,
        )
        if unified_full_pool is None:
            raise ValueError("G1/C1 Oracle@All requires the locked full-pool summary")
        _apply_full_pool_summary(
            results[route],
            unified_full_pool,
            route=route,
            native=native,
            final=final,
        )
    d1_top5 = _collapse_per_sample(d1_outcomes, "d1_top5_r0")
    d1_result = _summarise(
        d1_top5,
        _collapse_per_sample(d1_outcomes, "d1_top5_r7_gated"),
    )
    d1_top10 = _collapse_per_sample(d1_outcomes, "d1_top10_locked")
    d1_all = _collapse_per_sample(d1_outcomes, "d1_allnms_locked")
    if not (
        set(d1_top5["sample_id"])
        == set(d1_top10["sample_id"])
        == set(d1_all["sample_id"])
    ):
        raise ValueError("D1 Top-5/Top-10/AllNMS sample IDs are not aligned")
    d1_result.update(
        {
            "oracle_top10": int(
                d1_top10["first_positive_rank"].map(math.isfinite).sum()
            ),
            "top10_selected_correct": int(
                d1_top10["selected_correct"].astype(bool).sum()
            ),
            "oracle_all": int(d1_all["first_positive_rank"].map(math.isfinite).sum()),
            "all_selected_correct": int(d1_all["selected_correct"].astype(bool).sum()),
            "no_output": int(d1_all["no_output"].astype(bool).sum()),
            "no_positive_full_pool_including_no_output": int(
                (~d1_all["first_positive_rank"].map(math.isfinite)).sum()
            ),
            "no_positive_full_pool_excluding_no_output": int(
                (
                    (~d1_all["first_positive_rank"].map(math.isfinite))
                    & (~d1_all["no_output"].astype(bool))
                ).sum()
            ),
            "positive_only_below_top5": int(
                d1_all["first_positive_rank"].map(math.isfinite).sum()
                - d1_top5["first_positive_rank"].map(math.isfinite).sum()
            ),
        }
    )
    results["d1"] = d1_result
    _compare(results, targets)
    return {
        "schema_version": 1,
        "status": "PASS",
        "source_kind": "frozen_postformal_derived_outcomes",
        "raw_test_ground_truth_rows_read": 0,
        "routes": results,
    }
