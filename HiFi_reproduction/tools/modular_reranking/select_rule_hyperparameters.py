#!/usr/bin/env python3
"""Select the q/soft-mask rule weight on the official validation split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.models import (  # noqa: E402
    attach_scores,
    q_only_scores,
    q_softmask_rule_scores,
)
from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
)


def _protected_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".RUN_ACTIVE").is_file():
            return candidate
    raise ValueError(f"path is not inside an active protected run: {resolved}")


def _validate_write_scope(
    output_root: Path, tmp_root: Path
) -> tuple[Path, Path]:
    output = output_root.expanduser().resolve()
    temporary = tmp_root.expanduser().resolve()
    run_root = _protected_run_root(output)
    if _protected_run_root(temporary) != run_root:
        raise ValueError("output-root and tmp-root belong to different active runs")
    configured_tmp = (run_root / "tmp").resolve()
    if temporary != configured_tmp and configured_tmp not in temporary.parents:
        raise ValueError(f"tmp-root must be below {configured_tmp}")
    if output == temporary or temporary in output.parents:
        raise ValueError("final rule selection cannot be stored below tmp-root")
    temporary.mkdir(parents=True, exist_ok=True)
    return output, temporary


def top1(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[frame["reranker_rank"] == 1, ["sample_id", "candidate_positive"]]
        .rename(columns={"candidate_positive": "correct"})
        .sort_values("sample_id")
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-per-candidate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 < args.grid_step <= 1.0:
        parser.error("--grid-step must be in (0, 1]")
    output, tmp_root = _validate_write_scope(
        args.output_root, args.tmp_root
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {output}")
    frame = pd.read_parquet(args.validation_per_candidate.expanduser().resolve())
    if set(frame["split"].astype(str).unique()) - {"val", "validation"}:
        raise ValueError("rule selection may consume official validation only")
    required = {
        "sample_id",
        "candidate_id",
        "candidate_positive",
        "q_raw",
        "q_rank_normalized",
        "p_axis_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"validation table missing columns: {sorted(missing)}")
    baseline = top1(
        attach_scores(frame, q_only_scores(frame), method="q_only")
    ).set_index("sample_id")["correct"].astype(bool)
    alphas = np.unique(
        np.r_[np.arange(0.0, 1.0 + args.grid_step / 2.0, args.grid_step), 1.0]
    )
    rows = []
    for alpha in alphas:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        scored = attach_scores(
            frame,
            q_softmask_rule_scores(frame, alpha=alpha, beta=1.0 - alpha),
            method="q_softmask_rule",
        )
        current = top1(scored).set_index("sample_id")["correct"].astype(bool)
        recovered = int((~baseline & current).sum())
        harmful = int((baseline & ~current).sum())
        changed = recovered + harmful
        rows.append(
            {
                "alpha": alpha,
                "beta": 1.0 - alpha,
                "validation_samples_nonempty": int(len(current)),
                "j_at_1_nonempty": float(current.mean()),
                "recovered": recovered,
                "harmful": harmful,
                "net_gain": recovered - harmful,
                "outcome_changing_precision": (
                    recovered / changed if changed else 0.0
                ),
            }
        )
    sweep = pd.DataFrame(rows).drop_duplicates("alpha")
    # Prefer the most q-conservative weight after outcome metrics tie.
    selected = sweep.sort_values(
        ["net_gain", "j_at_1_nonempty", "outcome_changing_precision", "alpha"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).iloc[0].to_dict()
    staging = (
        tmp_root
        / "rule_hyperparameter_selection"
        / f"{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(parents=True)
    sweep_path = staging / "sweep.csv"
    final_sweep_path = output / "sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    validation_path = args.validation_per_candidate.expanduser().resolve()
    selection = {
        "schema_version": 1,
        **identity_payload(),
        "method": "q_softmask_rule",
        "selection_split": "validation",
        "selection_rule": (
            "max net gain, then J@1, outcome precision, then q-conservative alpha"
        ),
        "selected": selected,
        "validation_per_candidate": str(validation_path),
        "validation_per_candidate_sha256": sha256_file(validation_path),
        "sweep": str(final_sweep_path),
        "sweep_sha256": sha256_file(sweep_path),
        "formal_test_consumed": False,
    }
    (staging / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output)
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
