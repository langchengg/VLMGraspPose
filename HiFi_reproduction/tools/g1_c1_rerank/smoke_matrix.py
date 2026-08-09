#!/usr/bin/env python3
"""Run every locally feasible ranker on exactly 100 Train samples."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
for path in (str(REPOSITORY_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.grasping.g1_c1_safe_rerank.artifacts import atomic_json  # noqa: E402
from src.grasping.g1_c1_safe_rerank.calibration import ScoreCalibrator  # noqa: E402
from src.grasping.g1_c1_safe_rerank.crop_model import CropResidualModel  # noqa: E402
from tools.g1_c1_rerank.run_crop_cnn import load_crops  # noqa: E402
from tools.g1_c1_rerank.run_local_matrix import METHODS, _columns, _fit_model  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_matrix.py RUN_DIR")
    run = Path(sys.argv[1]).expanduser().resolve()
    shard_dir = run / "02_features" / "train" / "g1" / "feature_shards"
    shards = sorted(shard_dir.glob("part-*.parquet"))
    labels_path = run / "data" / "g1_train_candidate_labels.parquet"
    if not shards or not labels_path.is_file():
        raise FileNotFoundError("G1 feature shards and complete labels are required")
    # Candidate generation is ordered by source path, so the first 100 samples can
    # occupy fewer than five scenes.  Select one sample per scene in rounds to make
    # the smoke cohort exercise the formal five-fold grouping contract.
    sample_index = pd.concat(
        [pd.read_parquet(path, columns=["sample_id", "scene_id"]) for path in shards],
        ignore_index=True,
    ).drop_duplicates("sample_id")
    sample_index["within_scene"] = sample_index.groupby("scene_id", sort=False).cumcount()
    sample_ids = (
        sample_index.sort_values(["within_scene", "scene_id", "sample_id"], kind="stable")
        .head(100)["sample_id"]
    )
    selected = set(sample_ids)
    features = pd.concat(
        [
            frame.loc[frame["sample_id"].isin(selected)]
            for frame in (pd.read_parquet(path) for path in shards)
        ],
        ignore_index=True,
    )
    frame = features.loc[features["sample_id"].isin(set(sample_ids))].merge(
        pd.read_parquet(labels_path)[["sample_id", "stable_candidate_id", "candidate_correct"]],
        on=["sample_id", "stable_candidate_id"],
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
    if frame["sample_id"].nunique() != 100:
        raise AssertionError("smoke cohort must contain exactly 100 non-empty samples")
    if frame["scene_id"].nunique() < 5:
        raise AssertionError("smoke cohort must span at least five scenes")
    calibrator = ScoreCalibrator.fit("platt", frame)
    frame["source_score_calibrated"] = np.clip(
        calibrator.predict(frame["original_score"]), 1e-4, 1 - 1e-4
    )
    identity = frame["candidate_identity_sha256"].astype(str).tolist()
    output = run / "00_audit" / "smoke_matrix"
    records: list[dict[str, object]] = []
    for method, spec in METHODS.items():
        checkpoint = output / f"{method}.pt"
        try:
            columns = _columns(frame, through="F6", grare=bool(spec.get("grare")))
            model = _fit_model(
                str(spec["kind"]),
                columns,
                frame,
                seed=17,
                device="cpu",
                checkpoint=checkpoint,
            )
            scores = model.predict_scores(frame)
            if len(scores) != len(frame) or not np.isfinite(scores).all():
                raise AssertionError("candidate score coverage/non-finite failure")
            if frame["candidate_identity_sha256"].astype(str).tolist() != identity:
                raise AssertionError("candidate identity changed during smoke run")
            records.append(
                {
                    "method": method,
                    "rung": spec["rung"],
                    "kind": spec["kind"],
                    "status": "PASS",
                    "sample_count": 100,
                    "candidate_rows": len(frame),
                    "output_rows": len(scores),
                    "finite_scores": True,
                    "feature_count": len(columns),
                    "model": model.artifact(),
                }
            )
        except Exception as error:
            records.append(
                {
                    "method": method,
                    "rung": spec["rung"],
                    "kind": spec["kind"],
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
    try:
        crop = load_crops(run, "g1", "train", frame)
        columns = _columns(frame, through="F6")
        crop_model = CropResidualModel(columns, seed=17, device="cpu", epochs=2).fit(
            frame,
            crop,
            output / "r13_crop_cnn.pt",
        )
        crop_scores = crop_model.predict_scores(frame, crop)
        if len(crop_scores) != len(frame) or not np.isfinite(crop_scores).all():
            raise AssertionError("crop CNN score coverage/non-finite failure")
        records.append(
            {
                "method": "r13_crop_cnn",
                "rung": "R13",
                "kind": "candidate_aligned_crop_cnn",
                "status": "PASS",
                "sample_count": 100,
                "candidate_rows": len(frame),
                "output_rows": len(crop_scores),
                "finite_scores": True,
                "feature_count": len(columns),
                "model": crop_model.artifact(),
            }
        )
    except Exception as error:
        records.append(
            {
                "method": "r13_crop_cnn",
                "rung": "R13",
                "kind": "candidate_aligned_crop_cnn",
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL",
        "sample_count": 100,
        "candidate_rows": len(frame),
        "candidate_count_unchanged": True,
        "label_columns_excluded_from_model_features": True,
        "records": records,
    }
    atomic_json(output / "SMOKE_MATRIX.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "sample_count": payload["sample_count"],
                "candidate_rows": payload["candidate_rows"],
                "scenes": int(frame["scene_id"].nunique()),
                "methods": {row["method"]: row["status"] for row in records},
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
