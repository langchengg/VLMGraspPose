"""Scene-disjoint validation partitioning with the preregistered seed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import SEED
from .io import atomic_json, sha256_file


def build_scene_split(source_run: Path, run_dir: Path) -> dict[str, Any]:
    validation_path = source_run / "manifests/validation_samples.parquet"
    test_path = source_run / "manifests/test_samples.parquet"
    validation = pd.read_parquet(
        validation_path,
        columns=["sample_id", "scene_id", "language", "source_rgb_sha256", "source_depth_sha256"],
    )
    test = pd.read_parquet(test_path, columns=["sample_id", "scene_id"])
    scenes = np.asarray(sorted(validation["scene_id"].astype(str).unique()), dtype=object)
    np.random.default_rng(SEED).shuffle(scenes)
    prompt_count = int(round(0.20 * len(scenes)))
    policy_count = int(round(0.30 * len(scenes)))
    assignments = {
        str(scene): (
            "prompt_dev" if index < prompt_count
            else "policy_selection" if index < prompt_count + policy_count
            else "untouched_validation"
        )
        for index, scene in enumerate(scenes)
    }
    output = validation.copy()
    output["cohort"] = output["scene_id"].astype(str).map(assignments)
    if output["cohort"].isna().any():
        raise AssertionError("validation sample missing scene assignment")
    overlap = output.groupby("scene_id")["cohort"].nunique()
    if (overlap != 1).any():
        raise AssertionError("one scene spans validation cohorts")
    test_overlap = set(output["scene_id"].astype(str)) & set(test["scene_id"].astype(str))
    if test_overlap:
        raise AssertionError("test scene appears in validation cohort")
    path = run_dir / "DATA_SPLIT.csv"
    output.to_csv(path, index=False)
    summary = {
        "seed": SEED,
        "validation_samples": int(len(output)),
        "validation_scenes": int(len(scenes)),
        "cohorts": {
            cohort: {
                "samples": int((output["cohort"] == cohort).sum()),
                "scenes": int(output.loc[output["cohort"] == cohort, "scene_id"].nunique()),
            }
            for cohort in ("prompt_dev", "policy_selection", "untouched_validation")
        },
        "scene_overlap_between_cohorts": 0,
        "test_scene_overlap": 0,
        "data_split_sha256": sha256_file(path),
    }
    atomic_json(run_dir / "DATA_SPLIT_SUMMARY.json", summary)
    return summary
