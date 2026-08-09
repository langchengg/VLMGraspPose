"""Ground-truth-only diagnostics for a frozen proposal bank."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .selective_sam3_vg.metrics import summarize_ious


SOURCE_STAGES: tuple[tuple[str, frozenset[str]], ...] = (
    ("hifi_only", frozenset({"HIFI_ORIGINAL"})),
    ("hifi_thresholds", frozenset({"HIFI_THRESHOLD"})),
    ("full_query_text", frozenset({"TEXT_FULL_QUERY"})),
    (
        "short_target_phrase",
        frozenset(
            {
                "TEXT_TARGET_CATEGORY",
                "TEXT_TARGET_ATTRIBUTE",
                "TEXT_TARGET_INSTANCE",
            }
        ),
    ),
    ("automatic_proposals", frozenset({"AUTOMATIC"})),
    ("hifi_box_points", frozenset({"VISUAL_HIFI"})),
    ("component_prompts", frozenset({"COMPONENT"})),
    ("morphological_variants", frozenset({"MORPHOLOGY"})),
)


def _pil_nearest_indices(source_length: int, target_length: int) -> np.ndarray:
    ramp = np.arange(int(source_length), dtype=np.int32)[None, :]
    resized = Image.fromarray(ramp, mode="I").resize(
        (int(target_length), 1), Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=np.int64)[0]


def _packed_candidate_ious(
    path: Path, target: np.ndarray, *, chunk_size: int = 64
) -> dict[str, float]:
    target = np.asarray(target, dtype=bool)
    archive = np.load(path, allow_pickle=False)
    try:
        stored_shape = (
            tuple(
                int(value)
                for value in np.asarray(archive["__mask_shape__"]).reshape(-1)
            )
            if "__mask_shape__" in archive.files
            else None
        )
        if "__candidate_ids__" in archive.files and "__packed_masks__" in archive.files:
            if stored_shape is None:
                raise ValueError("schema-3 candidate archive lacks its mask shape")
            source_shape = stored_shape
            candidate_ids = [str(value) for value in archive["__candidate_ids__"].tolist()]
            packed = np.asarray(archive["__packed_masks__"], dtype=np.uint8)
        else:
            candidate_ids = [key for key in archive.files if not key.startswith("__")]
            if not candidate_ids:
                raise ValueError("candidate archive contains no masks")
            if stored_shape is not None:
                # Schema 2 stores one packed array per candidate.  Stack only
                # the compact byte arrays; do not materialize all dense masks.
                source_shape = stored_shape
                packed = np.stack(
                    [np.asarray(archive[key], dtype=np.uint8) for key in candidate_ids]
                )
            else:
                # Five protected pilot bundles predate packed persistence.
                # Convert their dense masks once so the same vectorized IoU
                # path and exact PIL-nearest coordinate map are still used.
                dense = np.stack(
                    [np.asarray(archive[key], dtype=bool) for key in candidate_ids]
                )
                source_shape = tuple(int(value) for value in dense.shape[1:])
                packed = np.packbits(
                    dense.reshape(len(candidate_ids), -1),
                    axis=1,
                    bitorder="little",
                )
                del dense
        if len(source_shape) != 2:
            raise ValueError("candidate mask shape must be two-dimensional")
    finally:
        archive.close()
    if packed.shape[0] != len(candidate_ids):
        raise ValueError("packed candidate IDs and masks do not align")
    y_indices = _pil_nearest_indices(source_shape[0], target.shape[0])
    x_indices = _pil_nearest_indices(source_shape[1], target.shape[1])
    values: dict[str, float] = {}
    source_pixels = int(np.prod(source_shape))
    for start in range(0, len(candidate_ids), int(chunk_size)):
        stop = min(len(candidate_ids), start + int(chunk_size))
        native = np.unpackbits(
            packed[start:stop],
            axis=1,
            count=source_pixels,
            bitorder="little",
        ).reshape(stop - start, *source_shape).astype(bool, copy=False)
        aligned = native[:, y_indices[:, None], x_indices[None, :]]
        intersections = np.count_nonzero(aligned & target, axis=(1, 2)).astype(
            np.float32
        )
        unions = np.count_nonzero(aligned | target, axis=(1, 2)).astype(
            np.float32
        )
        ious = np.ones(len(intersections), dtype=np.float32)
        nonempty = unions != 0
        ious[nonempty] = np.asarray(
            intersections[nonempty] / unions[nonempty], dtype=np.float32
        )
        values.update(
            {
                candidate_id: float(iou)
                for candidate_id, iou in zip(
                    candidate_ids[start:stop], ious, strict=True
                )
            }
        )
    return values


class OracleAccumulator:
    """Memory-bounded per-sample accumulator for a large candidate-label bank."""

    def __init__(self) -> None:
        self.per_sample_rows: list[dict[str, Any]] = []
        self.stage_values: dict[str, list[float]] = {
            stage: [] for stage, _ in SOURCE_STAGES
        }
        self.stage_candidate_counts: dict[str, list[int]] = {
            stage: [] for stage, _ in SOURCE_STAGES
        }

    def add(self, frame: pd.DataFrame) -> None:
        required = {"sample_id", "candidate_id", "source_family", "candidate_iou"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"candidate labels missing {sorted(required - set(frame.columns))}"
            )
        sample_ids = frame["sample_id"].astype(str).unique()
        if len(sample_ids) != 1:
            raise ValueError("oracle accumulator expects exactly one sample at a time")
        sample_id = str(sample_ids[0])
        cumulative_sources: set[str] = set()
        for stage, families in SOURCE_STAGES:
            cumulative_sources.update(families)
            eligible = frame[
                frame["eligible_final"]
                & frame["source_family"].isin(cumulative_sources)
            ]
            self.stage_values[stage].append(
                float(eligible["candidate_iou"].max()) if len(eligible) else 0.0
            )
            self.stage_candidate_counts[stage].append(int(len(eligible)))

        eligible = frame[frame["eligible_final"]]
        if eligible.empty:
            raise ValueError(f"sample has no eligible oracle candidate: {sample_id}")
        best = eligible.loc[eligible["candidate_iou"].idxmax()]
        hifi = eligible[eligible["source_family"] == "HIFI_ORIGINAL"]
        if len(hifi) != 1:
            raise ValueError(f"sample lacks exactly one canonical HiFi candidate: {sample_id}")
        hifi_iou = float(hifi.iloc[0]["candidate_iou"])
        p90_sources = sorted(
            set(
                eligible.loc[
                    eligible["candidate_iou"] > 0.90, "source_family"
                ].astype(str)
            )
        )
        self.per_sample_rows.append(
            {
                "sample_id": sample_id,
                "candidate_count": int(len(frame)),
                "eligible_candidate_count": int(len(eligible)),
                "hifi_iou": hifi_iou,
                "best_candidate_iou": float(best["candidate_iou"]),
                "best_candidate_id": str(best["candidate_id"]),
                "best_source_family": str(best["source_family"]),
                "best_source_variant": str(best["source_variant"]),
                "has_p90_candidate": bool(float(best["candidate_iou"]) > 0.90),
                "hifi_is_p90": bool(hifi_iou > 0.90),
                "no_improving_candidate": bool(
                    float(best["candidate_iou"]) <= hifi_iou
                ),
                "p90_source_families_json": json.dumps(p90_sources),
                "unique_p90_source_family": (
                    p90_sources[0] if len(p90_sources) == 1 else None
                ),
            }
        )

    def finalize(self) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        if not self.per_sample_rows:
            raise ValueError("oracle accumulator is empty")
        source_rows: list[dict[str, Any]] = []
        previous_p90 = 0
        previous_mean = 0.0
        for stage_index, (stage, families) in enumerate(SOURCE_STAGES):
            values = self.stage_values[stage]
            candidate_counts = self.stage_candidate_counts[stage]
            metrics = summarize_ious(values)
            p90 = int(metrics["p_at_90_numerator"])
            source_rows.append(
                {
                    "stage_index": stage_index,
                    "source_added": stage,
                    "source_families": ",".join(sorted(families)),
                    "cumulative_candidate_count": int(sum(candidate_counts)),
                    "mean_candidates_per_sample": float(np.mean(candidate_counts)),
                    "oracle_mean_iou": float(metrics["mean_iou"]),
                    "oracle_p70": float(metrics["p_at_70"]),
                    "oracle_p80": float(metrics["p_at_80"]),
                    "oracle_p90": float(metrics["p_at_90"]),
                    "oracle_p90_numerator": p90,
                    "new_p90_successes": p90 - previous_p90,
                    "marginal_p90": float(
                        metrics["p_at_90"]
                        - previous_p90 / len(self.per_sample_rows)
                    ),
                    "marginal_mean_iou": float(
                        metrics["mean_iou"] - previous_mean
                    ),
                }
            )
            previous_p90 = p90
            previous_mean = float(metrics["mean_iou"])
        per_sample = pd.DataFrame(self.per_sample_rows)
        source_contributions = pd.DataFrame(source_rows)
        oracle = summarize_ious(per_sample["best_candidate_iou"])
        summary = {
            **oracle,
            "expanded_proposal_bank_oracle_P@90": float(oracle["p_at_90"]),
            "expanded_proposal_bank_oracle_P@90_numerator": int(
                oracle["p_at_90_numerator"]
            ),
            "expanded_proposal_bank_oracle_P@90_denominator": int(
                oracle["p_at_90_denominator"]
            ),
            "mean_candidate_count": float(per_sample["candidate_count"].mean()),
            "median_candidate_count": float(per_sample["candidate_count"].median()),
            "no_candidate_count": int(
                (per_sample["eligible_candidate_count"] == 0).sum()
            ),
            "no_improving_candidate_count": int(
                per_sample["no_improving_candidate"].sum()
            ),
            "non_deployable_gt_oracle": True,
            "strict_threshold_semantics": "IoU > threshold",
        }
        return summary, per_sample, source_contributions


def candidate_ious(
    proposal_directory: Path,
    target_352: np.ndarray,
) -> pd.DataFrame:
    index = pd.read_parquet(proposal_directory / "candidate_index.parquet")
    ious = _packed_candidate_ious(
        proposal_directory / "candidate_masks.npz", target_352
    )
    rows: list[dict[str, Any]] = []
    for row in index.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        iou = ious[candidate_id]
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "candidate_id": candidate_id,
                "source_family": str(row["source_family"]),
                "source_variant": str(row["source_variant"]),
                "eligible_final": bool(row["eligible_final"]),
                "candidate_iou": iou,
                "y70": bool(iou > 0.70),
                "y80": bool(iou > 0.80),
                "y90": bool(iou > 0.90),
                "continuous_iou": iou,
            }
        )
    return pd.DataFrame(rows)


def summarize_oracle(candidate_labels: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "candidate_id", "source_family", "candidate_iou"}
    if not required.issubset(candidate_labels.columns):
        raise ValueError(f"candidate labels missing {sorted(required - set(candidate_labels))}")
    accumulator = OracleAccumulator()
    for _, frame in candidate_labels.groupby("sample_id", sort=False):
        accumulator.add(frame)
    return accumulator.finalize()


__all__ = [
    "SOURCE_STAGES",
    "OracleAccumulator",
    "candidate_ious",
    "summarize_oracle",
]
