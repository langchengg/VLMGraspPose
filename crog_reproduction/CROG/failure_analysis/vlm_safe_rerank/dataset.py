from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq

from .features import ordered_candidates, select_challengers
from .policy import pair_label


def jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                yield value


def stable_sample_id(feature: Mapping[str, Any]) -> str:
    value = feature.get("sample_id", feature.get("sample_index"))
    if isinstance(value, str) and value.startswith("multiple:"):
        return value
    return f"multiple:{str(feature.get('split', 'train')).lower()}:{int(value):08d}"


def load_feature_index(path: str | Path, sample_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
    result = {}
    for feature in jsonl(path):
        sample_id = stable_sample_id(feature)
        if sample_ids is None or sample_id in sample_ids:
            result[sample_id] = feature
    if sample_ids is not None and set(result) != sample_ids:
        missing = sorted(sample_ids - set(result))[:5]
        raise ValueError(f"frozen features missing requested samples: {missing}")
    return result


def load_label_index(path: str | Path, sample_ids: set[str] | None = None) -> dict[str, dict[str, bool]]:
    result = {}
    for row in jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        result[sample_id] = {
            str(item["candidate_id"]): bool(item["candidate_correct"])
            for item in row["candidate_labels"]
        }
    if sample_ids is not None and set(result) != sample_ids:
        raise ValueError("label rows do not match requested samples")
    return result


def cohort_rows(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def stratified_cohort_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_cohort: Mapping[str, int],
    partition: str = "train",
    seed: int = 20260803,
) -> list[dict[str, Any]]:
    """Fixed scene-aware, query-type round-robin sample without replacement."""

    import numpy as np
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for cohort, count in per_cohort.items():
        candidates = [dict(row) for row in rows if row["partition"] == partition and row["corrected_cohort"] == cohort]
        rng.shuffle(candidates)
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            buckets[(str(row["query_type"]), str(row["group_id"]))].append(row)
        keys = sorted(buckets)
        cursor = 0
        chosen_ids: set[str] = set()
        while len(chosen_ids) < count and keys:
            key = keys[cursor % len(keys)]
            bucket = buckets[key]
            while bucket and str(bucket[-1]["sample_id"]) in chosen_ids:
                bucket.pop()
            if bucket:
                row = bucket.pop()
                chosen_ids.add(str(row["sample_id"])); selected.append(row)
            else:
                keys.remove(key); cursor -= 1
            cursor += 1
        if len(chosen_ids) != count:
            raise ValueError(f"cohort {cohort} has insufficient rows")
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def smoke_sample(rows: Sequence[Mapping[str, Any]], *, seed: int = 20260803) -> list[dict[str, Any]]:
    return stratified_cohort_sample(
        rows,
        per_cohort={"protected_correct": 4, "recoverable_error": 3, "unrecoverable_error": 3},
        seed=seed,
    )


def build_pair_manifests(
    sample_rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, Mapping[str, Any]],
    *,
    corrected_labels: Mapping[str, Mapping[str, bool]] | None = None,
    legacy_labels: Mapping[str, Mapping[str, bool]] | None = None,
    all_challengers: bool,
    maximum_challengers: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return physically separated inference rows and evaluator-only rows."""

    inference: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for cohort in sample_rows:
        sample_id = str(cohort["sample_id"])
        feature = features[sample_id]
        ordered = ordered_candidates(feature)
        baseline_id = str(ordered[0]["candidate_id"])
        challenger_ids = [str(row["candidate_id"]) for row in ordered[1:]] if all_challengers else select_challengers(feature, maximum=maximum_challengers)
        for challenger_id in challenger_ids:
            inference.append({
                "sample_id": sample_id,
                "feature_source_sample_id": int(feature.get("sample_id", feature.get("sample_index"))),
                "baseline_candidate_id": baseline_id,
                "challenger_candidate_id": challenger_id,
                "official_split": str(feature["split"]),
            })
            if corrected_labels is not None:
                corrected = corrected_labels[sample_id]; legacy = legacy_labels[sample_id] if legacy_labels is not None else None
                evaluation.append({
                    "sample_id": sample_id, "baseline_candidate_id": baseline_id,
                    "challenger_candidate_id": challenger_id,
                    "corrected_baseline_correct": corrected[baseline_id],
                    "corrected_challenger_correct": corrected[challenger_id],
                    "corrected_pair_label": pair_label(
                        baseline_correct=corrected[baseline_id], challenger_correct=corrected[challenger_id]
                    ).value,
                    "legacy_baseline_correct": None if legacy is None else legacy[baseline_id],
                    "legacy_challenger_correct": None if legacy is None else legacy[challenger_id],
                    "cohort": str(cohort["corrected_cohort"]),
                    "query_type": str(cohort["query_type"]),
                    "frame_id": str(cohort["frame_id"]),
                    "scene_id": str(cohort["scene_id"]),
                    "bootstrap_sequence_id": str(cohort["group_id"]),
                })
    return inference, evaluation

