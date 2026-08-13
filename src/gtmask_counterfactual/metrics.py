"""Evaluation adapter and denominator-preserving counterfactual metrics.

The GT mask is deliberately absent from every evaluator call in this module.
Correctness depends only on candidate geometry and the locked grasp rectangles.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_K_VALUES = (5, 10)
_CANDIDATE_KEYS = ("sample_id", "route", "branch", "candidate_id")


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} misses columns: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_evaluator(
    path: str | Path, expected_sha256: str | None = None
) -> ModuleType:
    """Load one byte-bound evaluator without importing a mutable repo module."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"locked evaluator is not a regular file: {source}")
    observed = _sha256_file(source)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            "locked evaluator hash mismatch: "
            f"observed={observed} expected={expected_sha256}"
        )
    module_name = f"_gtmask_locked_evaluator_{observed[:20]}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load locked evaluator: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    for attribute in ("CanonicalGrasp", "gt_from_corners", "evaluate_candidate"):
        if not hasattr(module, attribute):
            raise TypeError(f"locked evaluator misses {attribute}")
    return module


def _row_value(row: Any, names: Sequence[str], *, field: str) -> Any:
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    raise ValueError(f"candidate evaluator input misses {field}: tried {list(names)}")


def evaluate_candidate_rows(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    evaluator_path: str | Path,
    evaluator_sha256: str | None = None,
    on_invalid: str = "raise",
) -> pd.DataFrame:
    """Label candidate rows with the locked same-GT 4-DoF evaluator.

    ``on_invalid='mark'`` keeps invalid geometry as an explicit technical row;
    the default fails closed so bulk execution cannot silently turn corruption
    into an ordinary negative candidate.
    """

    if on_invalid not in {"raise", "mark"}:
        raise ValueError("on_invalid must be 'raise' or 'mark'")
    _require_columns(candidates, set(_CANDIDATE_KEYS) | {"native_rank"}, name="candidates")
    _require_columns(
        ground_truth, {"sample_id", "gt_grasp_rectangles"}, name="ground truth"
    )
    if candidates.duplicated(list(_CANDIDATE_KEYS)).any():
        raise ValueError("candidate identities must be unique within branch")
    labels = ground_truth[["sample_id", "gt_grasp_rectangles"]].copy()
    labels["sample_id"] = labels["sample_id"].astype(str)
    if labels["sample_id"].eq("").any() or labels["sample_id"].duplicated().any():
        raise ValueError("ground truth sample IDs must be unique and non-empty")
    work = candidates.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    joined = work.merge(labels, on="sample_id", how="left", validate="many_to_one")
    if joined["gt_grasp_rectangles"].isna().any():
        raise ValueError("candidate rows reference samples without ground truth")
    module = load_locked_evaluator(evaluator_path, evaluator_sha256)
    records: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        try:
            candidate = module.CanonicalGrasp(
                cx_px=float(_row_value(row, ("cx_px",), field="cx_px")),
                cy_px=float(_row_value(row, ("cy_px",), field="cy_px")),
                theta_deg=float(_row_value(row, ("theta_deg",), field="theta_deg")),
                jaw_width_px=float(
                    _row_value(row, ("jaw_width_px", "width_px"), field="width")
                ),
                rectangle_height_px=float(
                    _row_value(
                        row,
                        ("rectangle_height_px", "height_px"),
                        field="height",
                    )
                ),
                native_score=float(getattr(row, "native_score", 0.0)),
                native_rank=int(row.native_rank),
                source_method=str(row.route),
                sample_id=str(row.sample_id),
            )
            rectangles = tuple(
                module.gt_from_corners(np.asarray(rectangle, dtype=np.float64))
                for rectangle in row.gt_grasp_rectangles
            )
            evaluated = module.evaluate_candidate(candidate, rectangles)
            best = evaluated.get("best_success_or_fallback")
            records.append(
                {
                    "evaluator_valid": True,
                    "evaluator_error": "",
                    "candidate_success": bool(evaluated["success"]),
                    "matched_gt_index": None if best is None else int(best["gt_index"]),
                    "best_same_gt_iou": math.nan if best is None else float(best["iou"]),
                    "best_same_gt_angle_error_deg": (
                        math.nan if best is None else float(best["angle_error_deg"])
                    ),
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            if on_invalid == "raise":
                raise ValueError(
                    "invalid evaluator row "
                    f"sample={row.sample_id} candidate={row.candidate_id}: {error}"
                ) from error
            records.append(
                {
                    "evaluator_valid": False,
                    "evaluator_error": f"{type(error).__name__}: {error}",
                    "candidate_success": False,
                    "matched_gt_index": None,
                    "best_same_gt_iou": math.nan,
                    "best_same_gt_angle_error_deg": math.nan,
                }
            )
    return pd.concat(
        [work.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1
    )


def _validate_manifest(sample_manifest: pd.DataFrame) -> pd.DataFrame:
    _require_columns(sample_manifest, {"sample_id"}, name="sample manifest")
    samples = sample_manifest.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    if (
        samples.empty
        or samples["sample_id"].eq("").any()
        or samples["sample_id"].duplicated().any()
    ):
        raise ValueError("sample manifest must contain unique non-empty sample IDs")
    return samples


def branch_sample_outcomes(
    candidates: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    *,
    route: str,
    branch: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    """Collapse one route/branch candidate pool to one row per denominator sample."""

    samples = _validate_manifest(sample_manifest)
    required = set(_CANDIDATE_KEYS) | {"native_rank", "candidate_success"}
    _require_columns(candidates, required, name="candidate labels")
    ks = tuple(sorted({int(value) for value in k_values}))
    if not ks or min(ks) <= 0:
        raise ValueError("k_values must contain positive integers")
    selected = candidates.loc[
        candidates["route"].astype(str).eq(str(route))
        & candidates["branch"].astype(str).eq(str(branch))
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    if selected.duplicated(list(_CANDIDATE_KEYS)).any():
        raise ValueError("candidate identities are not unique")
    unknown = sorted(set(selected["sample_id"]).difference(samples["sample_id"]))
    if unknown:
        raise ValueError(f"candidate rows reference samples outside denominator: {unknown[:5]}")
    if len(selected):
        ranks = pd.to_numeric(selected["native_rank"], errors="coerce").to_numpy(float)
        if not np.isfinite(ranks).all() or not np.equal(ranks, np.floor(ranks)).all():
            raise ValueError("native ranks must be finite integers")
        selected["native_rank"] = ranks.astype(int)
        if selected["native_rank"].le(0).any():
            raise ValueError("native ranks must be positive")
        if selected.duplicated(["sample_id", "native_rank"]).any():
            raise ValueError("native ranks must be unique within each sample")
        minima = selected.groupby("sample_id", sort=False)["native_rank"].min()
        if not minima.eq(1).all():
            raise ValueError("every non-empty candidate pool must begin at native rank 1")
        success_numeric = pd.to_numeric(
            selected["candidate_success"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(success_numeric).all() or not np.isin(
            success_numeric, [0, 1]
        ).all():
            raise ValueError("candidate_success must be finite binary")
        selected["candidate_success"] = success_numeric.astype(bool)

    grouped = selected.groupby("sample_id", sort=False)
    counts = grouped.size().to_dict()
    positives = (
        selected.loc[selected["candidate_success"]]
        .groupby("sample_id", sort=False)["native_rank"]
        .min()
        .to_dict()
    )
    positive_counts = grouped["candidate_success"].sum().astype(int).to_dict()
    rows: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        sample_id = str(sample.sample_id)
        first = positives.get(sample_id)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "route": str(route),
            "branch": str(branch),
            "candidate_count": int(counts.get(sample_id, 0)),
            "no_output": sample_id not in counts,
            "native_correct": first == 1,
            "first_positive_rank": pd.NA if first is None else int(first),
            "positive_candidate_count": int(positive_counts.get(sample_id, 0)),
            "oracle_all": first is not None,
            "reciprocal_rank": 0.0 if first is None else 1.0 / float(first),
        }
        for k in ks:
            hit = bool(first is not None and int(first) <= k)
            row[f"j_at_{k}"] = hit
            row[f"oracle_at_{k}"] = hit
        rows.append(row)
    result = pd.DataFrame.from_records(rows)
    result["first_positive_rank"] = result["first_positive_rank"].astype("Int64")
    return result


def _distribution(values: pd.Series, *, missing_label: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = missing_label if pd.isna(value) else str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_sample_outcomes(outcomes: pd.DataFrame) -> dict[str, Any]:
    """Compute branch metrics while retaining empty/no-output samples."""

    required = {
        "sample_id",
        "route",
        "branch",
        "candidate_count",
        "no_output",
        "native_correct",
        "first_positive_rank",
        "positive_candidate_count",
        "oracle_all",
        "reciprocal_rank",
    }
    _require_columns(outcomes, required, name="sample outcomes")
    if outcomes.empty or outcomes["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample outcomes must be non-empty and unique by sample_id")
    if outcomes["route"].astype(str).nunique() != 1 or outcomes[
        "branch"
    ].astype(str).nunique() != 1:
        raise ValueError("sample outcomes must describe exactly one route and branch")
    count = len(outcomes)
    candidate_count = pd.to_numeric(outcomes["candidate_count"], errors="raise")
    positive_count = pd.to_numeric(
        outcomes["positive_candidate_count"], errors="raise"
    )
    result: dict[str, Any] = {
        "route": str(outcomes["route"].iloc[0]),
        "branch": str(outcomes["branch"].iloc[0]),
        "N": count,
        "no_output": int(outcomes["no_output"].astype(bool).sum()),
        "no_output_rate": float(outcomes["no_output"].astype(bool).mean()),
        "candidate_count_mean": float(candidate_count.mean()),
        "candidate_count_median": float(candidate_count.median()),
        "candidate_count_p95": float(candidate_count.quantile(0.95)),
        "native_j_at_1_numerator": int(outcomes["native_correct"].astype(bool).sum()),
        "native_j_at_1": float(outcomes["native_correct"].astype(bool).mean()),
        "oracle_all_numerator": int(outcomes["oracle_all"].astype(bool).sum()),
        "oracle_all": float(outcomes["oracle_all"].astype(bool).mean()),
        "mrr": float(pd.to_numeric(outcomes["reciprocal_rank"]).mean()),
        "first_positive_rank_distribution": _distribution(
            outcomes["first_positive_rank"], missing_label="no_positive"
        ),
        "positive_candidates_per_sample_mean": float(positive_count.mean()),
        "positive_candidates_per_sample_median": float(positive_count.median()),
        "positive_candidates_per_sample_p95": float(positive_count.quantile(0.95)),
        "positive_candidates_per_sample_distribution": _distribution(
            positive_count, missing_label="0"
        ),
    }
    for column in sorted(
        name for name in outcomes.columns if name.startswith("oracle_at_")
    ):
        suffix = column.removeprefix("oracle_at_")
        values = outcomes[column].astype(bool)
        result[f"oracle_at_{suffix}_numerator"] = int(values.sum())
        result[f"oracle_at_{suffix}"] = float(values.mean())
        result[f"j_at_{suffix}_numerator"] = int(values.sum())
        result[f"j_at_{suffix}"] = float(values.mean())
    return result


def compute_branch_metrics(
    candidates: pd.DataFrame,
    sample_manifest: pd.DataFrame,
    *,
    route: str,
    branch: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcomes = branch_sample_outcomes(
        candidates,
        sample_manifest,
        route=route,
        branch=branch,
        k_values=k_values,
    )
    return outcomes, summarize_sample_outcomes(outcomes)


def compare_branch_outcomes(
    predicted: pd.DataFrame, gt: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair predicted/GT outcomes by identity and compute stage-replacement deltas."""

    for name, frame in (("predicted", predicted), ("gt", gt)):
        _require_columns(
            frame,
            {
                "sample_id",
                "candidate_count",
                "native_correct",
                "first_positive_rank",
                "oracle_all",
            },
            name=f"{name} outcomes",
        )
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} outcomes contain duplicate sample IDs")
    pred = predicted.copy()
    counterfactual = gt.copy()
    pred["sample_id"] = pred["sample_id"].astype(str)
    counterfactual["sample_id"] = counterfactual["sample_id"].astype(str)
    pred_ids, gt_ids = set(pred["sample_id"]), set(counterfactual["sample_id"])
    if pred_ids != gt_ids:
        raise ValueError(
            "paired branches have different sample universes: "
            f"pred_only={sorted(pred_ids - gt_ids)[:5]} "
            f"gt_only={sorted(gt_ids - pred_ids)[:5]}"
        )
    common_binary = sorted(
        ({name for name in pred if name.startswith(("j_at_", "oracle_at_"))})
        & ({name for name in counterfactual if name.startswith(("j_at_", "oracle_at_"))})
    )
    columns = [
        "sample_id",
        "candidate_count",
        "native_correct",
        "first_positive_rank",
        "oracle_all",
        *common_binary,
    ]
    paired = pred[columns].merge(
        counterfactual[columns],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_pred", "_gt"),
        sort=True,
    )
    pred_all = paired["oracle_all_pred"].astype(bool)
    gt_all = paired["oracle_all_gt"].astype(bool)
    recovered = ~pred_all & gt_all
    regressed = pred_all & ~gt_all
    both_positive = pred_all & gt_all
    neither = ~pred_all & ~gt_all
    paired["candidate_count_delta"] = (
        paired["candidate_count_gt"] - paired["candidate_count_pred"]
    )
    paired["first_positive_rank_delta"] = pd.array(
        [
            pd.NA
            if pd.isna(pred_rank) or pd.isna(gt_rank)
            else int(gt_rank) - int(pred_rank)
            for pred_rank, gt_rank in zip(
                paired["first_positive_rank_pred"],
                paired["first_positive_rank_gt"],
                strict=True,
            )
        ],
        dtype="Int64",
    )
    denominator = int((~pred_all).sum())
    summary: dict[str, Any] = {
        "N": len(paired),
        "pred_no_positive_to_gt_positive": int(recovered.sum()),
        "pred_positive_to_gt_no_positive": int(regressed.sum()),
        "both_positive": int(both_positive.sum()),
        "neither_positive": int(neither.sum()),
        "grounding_candidate_recovery_count": int(recovered.sum()),
        "grounding_candidate_recovery_denominator": denominator,
        "grounding_candidate_recovery_rate": (
            None if denominator == 0 else float(recovered.sum() / denominator)
        ),
        "oracle_grounding_ceiling": float(gt_all.mean() - pred_all.mean()),
        "residual_generator_failure_count": int((~gt_all).sum()),
        "residual_generator_failure_rate": float((~gt_all).mean()),
        "candidate_count_delta_mean": float(paired["candidate_count_delta"].mean()),
        "candidate_count_delta_median": float(
            paired["candidate_count_delta"].median()
        ),
        "first_positive_rank_delta_mean_both_positive": (
            None
            if not paired["first_positive_rank_delta"].notna().any()
            else float(paired["first_positive_rank_delta"].dropna().mean())
        ),
    }
    metric_columns = ["native_correct", "oracle_all", *common_binary]
    for column in dict.fromkeys(metric_columns):
        pred_values = paired[f"{column}_pred"].astype(bool)
        gt_values = paired[f"{column}_gt"].astype(bool)
        label = "native_j_at_1" if column == "native_correct" else column
        summary[f"delta_{label}"] = float(gt_values.mean() - pred_values.mean())
    return paired, summary


def metrics_equal(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, atol: float = 0.0
) -> bool:
    """Small recursive equality helper used by independent audit callers."""

    if set(observed) != set(expected):
        return False
    for key in observed:
        left, right = observed[key], expected[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if not metrics_equal(left, right, atol=atol):
                return False
        elif isinstance(left, (float, np.floating)) or isinstance(
            right, (float, np.floating)
        ):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol):
                return False
        elif left != right:
            return False
    return True
