"""Pre-registered scene cohorts and provider-call planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import atomic_json, atomic_parquet, sha256_file


def _class(row: pd.Series) -> str:
    if bool(row["top1_correct"]):
        return "protected_correct"
    return "recoverable_error" if bool(row["recoverable_error"]) else "unrecoverable_error"


def _balanced(frame: pd.DataFrame, n_each: int, seed: int) -> pd.DataFrame:
    parts = []
    for index, name in enumerate(("protected_correct", "recoverable_error", "unrecoverable_error")):
        group = frame.loc[frame["offline_sampling_class"].eq(name)]
        parts.append(group.sample(n=min(n_each, len(group)), random_state=seed+index))
    return pd.concat(parts, ignore_index=True).sort_values(["backend", "sample_id"]).reset_index(drop=True)


def build_stage_cohorts(run_dir: str | Path, *, seed: int = 20260805) -> dict[str, Any]:
    run = Path(run_dir)
    baseline = pd.read_parquet(run / "baseline_per_sample.parquet")
    split = pd.read_csv(run / "DATA_SPLIT.csv")
    validation = baseline.loc[baseline["split"].eq("validation")].merge(
        split[["sample_id", "cohort"]], on="sample_id", how="left", validate="many_to_one"
    )
    validation["offline_sampling_class"] = validation.apply(_class, axis=1)
    cohorts = []
    for backend_index, backend in enumerate(("G1", "C1")):
        prompt = validation.loc[(validation["backend"] == backend) & (validation["cohort"] == "prompt_dev") & validation["candidate_count"].astype(int).ge(2)]
        smoke = _balanced(prompt, 6, seed + backend_index*100)
        smoke["stage"] = "smoke"
        diagnostic = _balanced(prompt, 50, seed + backend_index*100 + 10)
        diagnostic["stage"] = "diagnostic"
        policy = validation.loc[(validation["backend"] == backend) & (validation["cohort"] == "policy_selection")]
        policy = policy.sample(n=min(800, len(policy)), random_state=seed+backend_index).sort_values(["scene_id", "sample_id"]).copy()
        policy["stage"] = "policy_selection"
        untouched = validation.loc[(validation["backend"] == backend) & (validation["cohort"] == "untouched_validation")].copy()
        untouched["stage"] = "untouched_validation"
        cohorts.extend((smoke, diagnostic, policy, untouched))
    table = pd.concat(cohorts, ignore_index=True)
    keep = ["stage", "backend", "sample_id", "scene_id", "candidate_count", "offline_sampling_class"]
    path = run / "STAGE_COHORTS.parquet"
    atomic_parquet(path, table[keep])
    counts = table.groupby(["stage", "backend", "offline_sampling_class"]).size().rename("samples").reset_index()
    summary = {
        "seed": seed, "path": str(path), "sha256": sha256_file(path),
        "counts": counts.to_dict(orient="records"),
        "label_firewall": "offline_sampling_class is cohort construction/evaluation only and is excluded from every provider payload",
    }
    atomic_json(run / "audit/stage_cohort_summary.json", summary)
    return summary
