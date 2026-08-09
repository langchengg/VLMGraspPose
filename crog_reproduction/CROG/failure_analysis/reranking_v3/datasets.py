from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import read_jsonl, scan_inference_record, stable_sample_id


@dataclass(frozen=True)
class InferenceSample:
    sample_id: str
    record: dict[str, Any]


@dataclass(frozen=True)
class LabelSample:
    sample_id: str
    record: dict[str, Any]


def load_inference_artifact(path: str | Path, *, allowed_ids: set[str] | None = None) -> list[InferenceSample]:
    values = []
    for record in read_jsonl(path):
        scan_inference_record(record)
        sample_id = str(record.get("stable_sample_id") or stable_sample_id(record["split"], record["sample_id"]))
        if allowed_ids is None or sample_id in allowed_ids:
            values.append(InferenceSample(sample_id, record))
    if len(values) != len({value.sample_id for value in values}):
        raise ValueError("duplicate sample ID in inference artifact")
    return values


def load_label_artifact(path: str | Path, *, allowed_ids: set[str] | None = None) -> list[LabelSample]:
    values = []
    for record in read_jsonl(path):
        sample_id = str(record["sample_id"])
        if allowed_ids is None or sample_id in allowed_ids:
            values.append(LabelSample(sample_id, record))
    if len(values) != len({value.sample_id for value in values}):
        raise ValueError("duplicate sample ID in label artifact")
    return values


def explicit_training_join(
    inference: list[InferenceSample], labels: list[LabelSample]
) -> list[tuple[InferenceSample, LabelSample]]:
    by_id = {value.sample_id: value for value in labels}
    if set(by_id) != {value.sample_id for value in inference}:
        missing_labels = sorted({value.sample_id for value in inference} - set(by_id))[:5]
        extra_labels = sorted(set(by_id) - {value.sample_id for value in inference})[:5]
        raise ValueError(f"inference/label cohort mismatch: missing={missing_labels}, extra={extra_labels}")
    result = []
    for sample in inference:
        label = by_id[sample.sample_id]
        inference_ids = [str(value["candidate_id"]) for value in sample.record["candidates"]]
        label_ids = [str(value["candidate_id"]) for value in label.record["candidate_labels"]]
        if inference_ids != label_ids:
            raise ValueError(f"candidate identity mismatch during explicit join: {sample.sample_id}")
        result.append((sample, label))
    return result

