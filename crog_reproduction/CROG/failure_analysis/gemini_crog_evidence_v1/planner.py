from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import MODEL_IDS


MAX_PLANNED_REQUEST_SLOTS = 76_736
P1_PROTOCOL = "p1_full_crog_evidence"
ABLATION_PROTOCOLS = (
    "a0_visual_only",
    "a1_visual_plus_q",
    P1_PROTOCOL,
    "a3_full_evidence_without_q",
)
QUERY_TYPES = ("name", "attribute", "relation", "location", "mixed")
_PURE_RELATION_PROGRAM_TYPES = {
    "scene",
    "ground",
    "filter_category",
    "unique",
    "relate",
    "return",
}


@dataclass(frozen=True)
class CohortSizes:
    pilot: int = 100
    stability: int = 20
    ablation: int = 500
    calibration: int = 9_790
    validation: int = 8_669
    formal_test: int = 17_749

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(int(value) < 1 for value in values.values()):
            raise ValueError("all cohort sizes must be positive")
        if self.stability > self.pilot:
            raise ValueError("stability must be a subset of pilot")
        if self.pilot > self.ablation:
            raise ValueError("pilot must fit inside the ablation cohort")


@dataclass(frozen=True)
class ModelEstimate:
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    latency_seconds: float

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.thought_tokens) < 0:
            raise ValueError("token estimates must be non-negative")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency estimate must be finite and non-negative")


DEFAULT_MODEL_ESTIMATES: dict[str, ModelEstimate] = {
    "gemini-robotics-er-2-preview": ModelEstimate(2620, 851, 924, 9.6235),
    "gemini-3.6-flash": ModelEstimate(2620, 945, 1440, 17.5973),
}


@dataclass(frozen=True)
class PlanAssumptions:
    concurrency: int = 2
    retry_max: int = 5
    flash_input_usd_per_million: float = 1.50
    flash_output_including_thought_usd_per_million: float = 7.50
    er2_input_usd_per_million: float = 2.00
    er2_output_including_thought_usd_per_million: float = 10.00
    flash_cost_cap_per_request_usd: float = 0.10
    er2_cost_cap_per_request_usd: float | None = None
    max_spend_usd: float | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be positive")
        if self.retry_max < 0:
            raise ValueError("retry_max must be non-negative")
        for name in (
            "flash_input_usd_per_million",
            "flash_output_including_thought_usd_per_million",
            "er2_input_usd_per_million",
            "er2_output_including_thought_usd_per_million",
            "flash_cost_cap_per_request_usd",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("er2_cost_cap_per_request_usd", "max_spend_usd"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, order=True)
class LogicalRequestKey:
    namespace: str
    protocol: str
    model_id: str
    sample_id: str
    replicate_id: int = 0

    def encode(self) -> str:
        return "|".join(
            (
                self.namespace,
                self.protocol,
                self.model_id,
                self.sample_id,
                str(int(self.replicate_id)),
            )
        )


@dataclass(frozen=True)
class CacheContract:
    prompt_hash: str | None = None
    schema_hash: str | None = None
    renderer_hash: str | None = None
    evidence_schema_hash: str | None = None
    max_output_tokens: int = 4096
    image_resolution: str = "high"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def with_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["content_sha256"] = _payload_digest(payload)
    return payload


def write_immutable_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Create a deterministic JSON artifact, or verify an identical existing one."""
    path = Path(path)
    payload = with_content_digest(value)
    content = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable artifact already differs: {path}")
    return payload


def annotation_query_type(record: Mapping[str, Any]) -> str:
    filename = str(record.get("template_filename", ""))
    direct = {
        "name.json": "name",
        "attribute.json": "attribute",
        "location.json": "location",
    }
    if filename in direct:
        return direct[filename]
    if filename != "relation.json":
        raise ValueError(f"unknown annotation template: {filename!r}")
    program = record.get("program")
    if not isinstance(program, list):
        raise ValueError("relation annotation is missing its program metadata")
    program_types = {str(item["type"]) for item in program}
    return "relation" if program_types.issubset(_PURE_RELATION_PROGRAM_TYPES) else "mixed"


def load_annotation_query_types(path: str | Path) -> dict[int, str]:
    """Load only question-index to query-type; GT/program payloads are discarded."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"annotation file has no data list: {path}")
    result: dict[int, str] = {}
    for row in rows:
        sample_id = int(row["question_index"])
        if sample_id in result:
            raise ValueError(f"duplicate question_index {sample_id} in {path}")
        result[sample_id] = annotation_query_type(row)
    return result


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _polygon_iou(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    import cv2

    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    intersection, _ = cv2.intersectConvexConvex(left_array, right_array)
    union = (
        abs(float(cv2.contourArea(left_array)))
        + abs(float(cv2.contourArea(right_array)))
        - float(intersection)
    )
    return 0.0 if union <= 0 else float(intersection / union)


def _candidate_scalar(candidate: Mapping[str, Any], name: str) -> float:
    feature = candidate.get("features", {}).get(name, {})
    value = feature.get("value") if isinstance(feature, Mapping) else None
    return 0.0 if value is None else float(value)


def load_development_strata(
    *,
    features_path: str | Path,
    legacy_labels_path: str | Path,
    development_sample_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    """Derive selection-only strata without retaining any evaluator labels."""
    compact: dict[str, dict[str, Any]] = {}
    features = _read_jsonl(features_path)
    labels = _read_jsonl(legacy_labels_path)
    for feature, label in zip(features, labels, strict=True):
        split = str(feature.get("split", "train")).lower()
        sample_id = f"multiple:{split}:{int(feature['sample_id']):08d}"
        if sample_id not in development_sample_ids:
            continue
        candidates = feature["candidates"]
        correctness = [bool(item["candidate_correct"]) for item in label["candidate_labels"]]
        q_values = [float(item["q_raw"]) for item in candidates]
        overlaps = [
            _polygon_iou(candidates[left]["polygon"], candidates[right]["polygon"])
            for left in range(5)
            for right in range(left + 1, 5)
        ]
        compact[sample_id] = {
            "q_status": (
                "q_only_correct"
                if correctness[0]
                else ("q_only_wrong_recoverable" if any(correctness) else "top5_all_wrong")
            ),
            "q_margin": q_values[0] - q_values[1],
            "predicted_mask_area": float(feature.get("predicted_mask_area", 0.0)),
            "predicted_mask_confidence": float(
                np.mean([_candidate_scalar(item, "mask_consistency") for item in candidates])
            ),
            "heavy_overlap": max(overlaps, default=0.0) >= 0.25,
        }
    missing = development_sample_ids - set(compact)
    if missing:
        raise ValueError(f"development profile is missing {len(missing)} samples")

    def quantiles(name: str) -> tuple[float, float]:
        values = np.asarray([row[name] for row in compact.values()], dtype=np.float64)
        return float(np.quantile(values, 0.2)), float(np.quantile(values, 0.8))

    q_low, q_high = quantiles("q_margin")
    area_low, area_high = quantiles("predicted_mask_area")
    mask_low, mask_high = quantiles("predicted_mask_confidence")
    result: dict[str, tuple[str, ...]] = {}
    for sample_id, row in compact.items():
        strata = {str(row["q_status"])}
        if row["q_margin"] <= q_low:
            strata.add("low_q_margin")
        if row["q_margin"] >= q_high:
            strata.add("high_q_margin")
        if row["predicted_mask_area"] <= area_low:
            strata.add("small_target")
        if row["predicted_mask_area"] >= area_high:
            strata.add("large_target")
        if row["predicted_mask_confidence"] <= mask_low:
            strata.add("low_predicted_mask_confidence")
        if row["predicted_mask_confidence"] >= mask_high:
            strata.add("high_predicted_mask_confidence")
        if row["heavy_overlap"]:
            strata.add("heavy_overlap")
        result[sample_id] = tuple(sorted(strata))
    return result


def _selection_hash(seed: int, label: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{sample_id}".encode("utf-8")).hexdigest()


def select_scene_grouped_stratified(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    required_sample_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Select deterministically while spreading picks over query types and scenes."""
    if count < 1 or count > len(rows):
        raise ValueError(f"invalid cohort count {count} for pool of {len(rows)}")
    by_id = {str(row["sample_id"]): dict(row) for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("selection pool contains duplicate sample IDs")
    required = sorted(set(map(str, required_sample_ids)))
    missing = set(required) - set(by_id)
    if missing:
        raise ValueError(f"required samples are absent: {sorted(missing)[:5]}")
    if len(required) > count:
        raise ValueError("required samples exceed cohort size")

    selected = list(required)
    selected_set = set(selected)
    scene_counts = Counter(str(by_id[item]["scene_id"]) for item in selected)
    group_counts = Counter(str(by_id[item]["group_id"]) for item in selected)

    def add(sample_id: str) -> None:
        selected.append(sample_id)
        selected_set.add(sample_id)
        scene_counts[str(by_id[sample_id]["scene_id"])] += 1
        group_counts[str(by_id[sample_id]["group_id"])] += 1

    coverage: dict[str, list[str]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        evaluation = row["evaluation_only"]
        labels = [f"query:{evaluation['query_type']}"] + [
            f"stratum:{item}" for item in evaluation.get("strata", [])
        ]
        for label in labels:
            coverage.setdefault(label, []).append(sample_id)
    for label in sorted(coverage):
        if len(selected) >= count:
            break
        candidates = [item for item in coverage[label] if item not in selected_set]
        if not candidates:
            continue
        choice = min(
            candidates,
            key=lambda item: (
                scene_counts[str(by_id[item]["scene_id"])],
                group_counts[str(by_id[item]["group_id"])],
                _selection_hash(seed, label, item),
                item,
            ),
        )
        add(choice)

    query_types = [
        item
        for item in QUERY_TYPES
        if any(row["evaluation_only"]["query_type"] == item for row in rows)
    ]
    while len(selected) < count:
        made_progress = False
        for query_type in query_types:
            candidates = [
                str(row["sample_id"])
                for row in rows
                if row["evaluation_only"]["query_type"] == query_type
                and str(row["sample_id"]) not in selected_set
            ]
            if not candidates:
                continue
            choice = min(
                candidates,
                key=lambda item: (
                    scene_counts[str(by_id[item]["scene_id"])],
                    group_counts[str(by_id[item]["group_id"])],
                    _selection_hash(seed, f"fill:{query_type}", item),
                    item,
                ),
            )
            add(choice)
            made_progress = True
            if len(selected) >= count:
                break
        if not made_progress:
            raise AssertionError("selection exhausted before reaching requested count")
    return [by_id[item] for item in sorted(selected)]


def _distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _query_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["evaluation_only"]["query_type"]) for row in rows)
    return {name: int(counts.get(name, 0)) for name in QUERY_TYPES}


def _cohort_manifest(
    *,
    name: str,
    partition: str,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    split_manifest_sha256: str,
    annotation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row["sample_id"]))
    return {
        "schema_version": "1.0.0",
        "kind": "gemini_crog_cohort_manifest",
        "cohort": name,
        "partition": partition,
        "seed": int(seed),
        "sample_count": len(ordered),
        "sample_ids_sha256": hashlib.sha256(
            canonical_json([row["sample_id"] for row in ordered]).encode("utf-8")
        ).hexdigest(),
        "scene_distribution": _distribution(ordered, "scene_id"),
        "group_distribution": _distribution(ordered, "group_id"),
        "query_type_distribution": _query_distribution(ordered),
        "selection_contract": {
            "algorithm": "deterministic_scene_grouped_query_and_strata_round_robin_v1",
            "evaluation_only_path": "rows[].evaluation_only",
            "request_builder_must_drop": ["evaluation_only"],
            "annotation_sensitive_fields_persisted": False,
        },
        "sources": {
            "split_manifest_sha256": split_manifest_sha256,
            "annotation_sha256_by_official_split": dict(sorted(annotation_sha256.items())),
        },
        "rows": ordered,
    }


def _load_smoke_ids(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("smoke ID artifact must contain a JSON list")
    result = []
    for value in values:
        text = str(value)
        result.append(text if text.startswith("multiple:") else f"multiple:train:{int(value):08d}")
    return sorted(set(result))


def build_full_run_manifests(
    *,
    run_root: str | Path,
    split_manifest_path: str | Path,
    annotation_paths: Mapping[str, str | Path],
    sizes: CohortSizes = CohortSizes(),
    seed: int = 47,
    smoke_ids_path: str | Path | None = None,
    development_strata: Mapping[str, Sequence[str]] | None = None,
    development_features_path: str | Path | None = None,
    development_legacy_labels_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build all six immutable manifests without persisting annotation programs/GT."""
    required_annotation_splits = {"train", "val", "test"}
    if set(annotation_paths) != required_annotation_splits:
        raise ValueError("annotation_paths must contain exactly train, val, and test")
    annotation_types = {
        split: load_annotation_query_types(path) for split, path in annotation_paths.items()
    }
    annotation_hashes = {
        split: sha256_file(path) for split, path in annotation_paths.items()
    }
    split_payload = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    split_hash = sha256_file(split_manifest_path)
    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "calibration": [],
        "validation": [],
        "test": [],
    }
    for source in split_payload["rows"]:
        partition = str(source["development_partition"])
        if partition not in partitions:
            continue
        official_split = str(source["official_split"])
        local_id = int(source["source_sample_id"])
        try:
            query_type = annotation_types[official_split][local_id]
        except KeyError as exc:
            raise ValueError(
                f"missing annotation metadata for {official_split}:{local_id}"
            ) from exc
        partitions[partition].append(
            {
                "sample_id": str(source["sample_id"]),
                "source_sample_id": local_id,
                "official_split": official_split,
                "development_partition": partition,
                "frame_id": str(source["frame_id"]),
                "scene_id": str(source["scene_id"]),
                "group_id": str(source["sequence_id"]),
                "evaluation_only": {"query_type": query_type, "strata": []},
            }
        )
    expected = {
        "calibration": sizes.calibration,
        "validation": sizes.validation,
        "test": sizes.formal_test,
    }
    for partition, count in expected.items():
        if len(partitions[partition]) != count:
            raise AssertionError(
                f"{partition} count changed: {len(partitions[partition])} != {count}"
            )
    if len(partitions["train"]) < sizes.ablation:
        raise AssertionError("development partition is too small for ablation")

    development_ids = {str(row["sample_id"]) for row in partitions["train"]}
    if development_strata is None:
        if development_features_path is None or development_legacy_labels_path is None:
            raise ValueError(
                "development strata or both development feature/label paths are required"
            )
        development_strata = load_development_strata(
            features_path=development_features_path,
            legacy_labels_path=development_legacy_labels_path,
            development_sample_ids=development_ids,
        )
    missing_strata = development_ids - set(development_strata)
    if missing_strata:
        raise ValueError(f"development strata missing {len(missing_strata)} samples")
    for row in partitions["train"]:
        row["evaluation_only"]["strata"] = sorted(
            set(map(str, development_strata[str(row["sample_id"])]))
        )

    smoke_ids = _load_smoke_ids(smoke_ids_path)
    pilot = select_scene_grouped_stratified(
        partitions["train"],
        count=sizes.pilot,
        seed=seed,
        required_sample_ids=smoke_ids,
    )
    stability = select_scene_grouped_stratified(
        pilot,
        count=sizes.stability,
        seed=seed + 1,
    )
    ablation = select_scene_grouped_stratified(
        partitions["train"],
        count=sizes.ablation,
        seed=seed + 2,
        required_sample_ids=[row["sample_id"] for row in pilot],
    )
    cohorts = {
        "pilot": ("train", pilot),
        "stability": ("train", stability),
        "ablation": ("train", ablation),
        "calibration": ("calibration", partitions["calibration"]),
        "validation": ("validation", partitions["validation"]),
        "formal_test": ("test", partitions["test"]),
    }
    root = Path(run_root)
    result = {}
    for name, (partition, rows) in cohorts.items():
        manifest = _cohort_manifest(
            name=name,
            partition=partition,
            rows=rows,
            seed=seed,
            split_manifest_sha256=split_hash,
            annotation_sha256=annotation_hashes,
        )
        result[name] = write_immutable_json(root / f"{name}_manifest.json", manifest)
    return result


def request_slot_contract(sizes: CohortSizes = CohortSizes()) -> dict[str, int]:
    models = len(MODEL_IDS)
    result = {
        "pilot": sizes.pilot * models,
        "stability": sizes.stability * 3 * models,
        "ablation": sizes.ablation * len(ABLATION_PROTOCOLS) * models,
        "calibration": sizes.calibration * models,
        "validation": sizes.validation * models,
        "formal_test": sizes.formal_test * models,
    }
    result["total"] = sum(result.values())
    return result


def load_exact_p1_cache_keys(
    cache_paths: Iterable[str | Path],
    *,
    contract: CacheContract = CacheContract(),
) -> set[str]:
    """Project only valid, exact fixed-P1 cache rows into logical request keys."""
    result: set[str] = set()
    hash_fields = {
        "prompt_hash": contract.prompt_hash,
        "schema_hash": contract.schema_hash,
        "renderer_hash": contract.renderer_hash,
        "evidence_schema_hash": contract.evidence_schema_hash,
    }
    for path in cache_paths:
        path = Path(path)
        if not path.exists():
            continue
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(responses)")
            }
            required = {
                "sample_id",
                "model_id",
                "valid",
                "generation_config_json",
                *hash_fields,
            }
            if not required.issubset(columns):
                continue
            selected = ",".join(
                ("sample_id", "model_id", "valid", "generation_config_json", *hash_fields)
            )
            for row in connection.execute(f"SELECT {selected} FROM responses"):
                sample_id, model_id, valid, raw_generation, *observed_hashes = row
                if not bool(valid) or str(model_id) not in MODEL_IDS:
                    continue
                generation = json.loads(raw_generation)
                if int(generation.get("max_output_tokens", -1)) != contract.max_output_tokens:
                    continue
                if str(generation.get("image_resolution")) != contract.image_resolution:
                    continue
                if any(
                    expected is not None and str(observed) != str(expected)
                    for observed, expected in zip(observed_hashes, hash_fields.values(), strict=True)
                ):
                    continue
                result.add(
                    LogicalRequestKey(
                        "inference", P1_PROTOCOL, str(model_id), str(sample_id), 0
                    ).encode()
                )
        finally:
            connection.close()
    return result


def _manifest_rows(value: Mapping[str, Any] | str | Path) -> list[dict[str, Any]]:
    payload = (
        json.loads(Path(value).read_text(encoding="utf-8"))
        if isinstance(value, (str, Path))
        else value
    )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("cohort manifest has no rows")
    if int(payload.get("sample_count", -1)) != len(rows):
        raise ValueError("cohort manifest sample_count mismatch")
    return rows


def _phase_keys(
    phase: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[LogicalRequestKey]:
    if phase == "stability":
        return [
            LogicalRequestKey("stability", P1_PROTOCOL, model, str(row["sample_id"]), replicate)
            for row in rows
            for model in MODEL_IDS
            for replicate in (1, 2, 3)
        ]
    protocols = ABLATION_PROTOCOLS if phase == "ablation" else (P1_PROTOCOL,)
    return [
        LogicalRequestKey("inference", protocol, model, str(row["sample_id"]), 0)
        for row in rows
        for protocol in protocols
        for model in MODEL_IDS
    ]


def _usage_and_cost(
    new_by_model: Mapping[str, int],
    *,
    assumptions: PlanAssumptions,
    estimates: Mapping[str, ModelEstimate],
) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for model in MODEL_IDS:
        count = int(new_by_model.get(model, 0))
        estimate = estimates[model]
        usage[model] = {
            "requests": count,
            "input_tokens": count * estimate.input_tokens,
            "output_tokens": count * estimate.output_tokens,
            "thought_tokens": count * estimate.thought_tokens,
            "total_tokens": count
            * (estimate.input_tokens + estimate.output_tokens + estimate.thought_tokens),
        }
    flash = usage["gemini-3.6-flash"]
    flash_cost = (
        flash["input_tokens"] * assumptions.flash_input_usd_per_million
        + (flash["output_tokens"] + flash["thought_tokens"])
        * assumptions.flash_output_including_thought_usd_per_million
    ) / 1_000_000.0
    er2 = usage["gemini-robotics-er-2-preview"]
    er2_cost = (
        er2["input_tokens"] * assumptions.er2_input_usd_per_million
        + (er2["output_tokens"] + er2["thought_tokens"])
        * assumptions.er2_output_including_thought_usd_per_million
    ) / 1_000_000.0
    er2_requests = er2["requests"]
    er2_reserve = (
        None
        if assumptions.er2_cost_cap_per_request_usd is None
        else er2_requests * assumptions.er2_cost_cap_per_request_usd
    )
    flash_reserve = flash["requests"] * assumptions.flash_cost_cap_per_request_usd
    total_reserve = (
        None if er2_reserve is None else float(flash_reserve + er2_reserve)
    )
    return {
        "token_usage": usage,
        "cost": {
            "flash_estimated_usd": float(flash_cost),
            "er2_estimated_usd": float(er2_cost),
            "published_rate_estimated_total_usd": float(flash_cost + er2_cost),
            "flash_conservative_reserve_usd": float(flash_reserve),
            "er2_conservative_reserve_usd": er2_reserve,
            "total_budget_reserve_usd": total_reserve,
            "total_expected_usd": float(flash_cost + er2_cost),
            "er2_public_price_available": True,
            "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        },
    }


def build_full_run_plan(
    *,
    run_root: str | Path,
    manifests: Mapping[str, Mapping[str, Any] | str | Path],
    assumptions: PlanAssumptions,
    existing_cache_keys: Iterable[str] = (),
    model_estimates: Mapping[str, ModelEstimate] = DEFAULT_MODEL_ESTIMATES,
    max_request_slots: int = MAX_PLANNED_REQUEST_SLOTS,
) -> dict[str, Any]:
    """Build an immutable, cache-aware C-H request/cost/runtime plan."""
    phase_order = (
        "pilot",
        "stability",
        "ablation",
        "calibration",
        "validation",
        "formal_test",
    )
    if set(manifests) != set(phase_order):
        raise ValueError("manifests must contain exactly the six C-H cohorts")
    if set(model_estimates) != set(MODEL_IDS):
        raise ValueError("model estimates must match the exact model IDs")
    cache = set(map(str, existing_cache_keys))
    seen: set[str] = set()
    planned_slots = 0
    phases: list[dict[str, Any]] = []
    total_new_by_model = Counter()
    total_existing_hits = total_cross_hits = 0
    for phase in phase_order:
        manifest_value = manifests[phase]
        rows = _manifest_rows(manifest_value)
        keys = _phase_keys(phase, rows)
        planned_slots += len(keys)
        new_by_model = Counter()
        existing_hits = cross_hits = 0
        for key in keys:
            encoded = key.encode()
            if encoded in seen:
                cross_hits += 1
                continue
            seen.add(encoded)
            if encoded in cache:
                existing_hits += 1
            else:
                new_by_model[key.model_id] += 1
                total_new_by_model[key.model_id] += 1
        usage_cost = _usage_and_cost(
            new_by_model, assumptions=assumptions, estimates=model_estimates
        )
        base_wall_seconds = sum(
            new_by_model[model] * model_estimates[model].latency_seconds
            for model in MODEL_IDS
        ) / assumptions.concurrency
        manifest_payload = (
            json.loads(Path(manifest_value).read_text(encoding="utf-8"))
            if isinstance(manifest_value, (str, Path))
            else manifest_value
        )
        new_count = sum(new_by_model.values())
        phases.append(
            {
                "phase": phase,
                "split": str(manifest_payload["partition"]),
                "sample_count": len(rows),
                "sample_manifest": str(
                    (Path(run_root) / f"{phase}_manifest.json").resolve()
                ),
                "sample_manifest_sha256": str(
                    manifest_payload.get("content_sha256", _payload_digest(manifest_payload))
                ),
                "scene_distribution": dict(manifest_payload["scene_distribution"]),
                "group_distribution": dict(manifest_payload["group_distribution"]),
                "query_type_distribution": dict(manifest_payload["query_type_distribution"]),
                "model_ids": list(MODEL_IDS),
                "protocols": list(ABLATION_PROTOCOLS if phase == "ablation" else (P1_PROTOCOL,)),
                "replicates": [1, 2, 3] if phase == "stability" else [0],
                "planned_request_slots": len(keys),
                "existing_cache_hits": existing_hits,
                "cross_phase_cache_hits": cross_hits,
                "cache_hits": existing_hits + cross_hits,
                "new_requests": new_count,
                "new_requests_by_model": {
                    model: int(new_by_model.get(model, 0)) for model in MODEL_IDS
                },
                "expected_retries_upper_bound": new_count * assumptions.retry_max,
                "expected_token_usage": usage_cost["token_usage"],
                "expected_cost": usage_cost["cost"],
                "expected_wall_seconds": float(base_wall_seconds),
                "eta_seconds": float(base_wall_seconds),
                "expected_requests_per_hour": (
                    None
                    if base_wall_seconds == 0
                    else float(new_count * 3600.0 / base_wall_seconds)
                ),
                "expected_wall_seconds_with_retry_upper_bound": float(
                    base_wall_seconds * (1 + assumptions.retry_max)
                ),
            }
        )
        total_existing_hits += existing_hits
        total_cross_hits += cross_hits
    if planned_slots > int(max_request_slots):
        raise AssertionError(
            f"planned request slots exceed cap: {planned_slots} > {max_request_slots}"
        )
    if len(seen) > int(max_request_slots):
        raise AssertionError(
            f"planned unique requests exceed cap: {len(seen)} > {max_request_slots}"
        )
    totals = _usage_and_cost(
        total_new_by_model, assumptions=assumptions, estimates=model_estimates
    )
    total_new = sum(total_new_by_model.values())
    expected_wall = sum(
        total_new_by_model[model] * model_estimates[model].latency_seconds
        for model in MODEL_IDS
    ) / assumptions.concurrency
    expected_total_cost = totals["cost"]["total_budget_reserve_usd"]
    blockers = []
    if assumptions.max_spend_usd is None:
        blockers.append("missing_GEMINI_MAX_SPEND_USD")
    if assumptions.er2_cost_cap_per_request_usd is None:
        blockers.append("missing_GEMINI_ER2_COST_CAP_PER_REQUEST_USD")
    if (
        expected_total_cost is not None
        and assumptions.max_spend_usd is not None
        and expected_total_cost > assumptions.max_spend_usd
    ):
        blockers.append("expected_cost_exceeds_budget")
    plan = {
        "schema_version": "1.0.0",
        "kind": "gemini_crog_full_run_plan",
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "request_cap": int(max_request_slots),
        "model_ids": list(MODEL_IDS),
        "phase_order": list(phase_order),
        "phases": phases,
        "totals": {
            "planned_request_slots": planned_slots,
            "planned_unique_requests": len(seen),
            "existing_cache_hits": total_existing_hits,
            "cross_phase_cache_hits": total_cross_hits,
            "cache_hits": total_existing_hits + total_cross_hits,
            "new_requests": total_new,
            "new_requests_by_model": {
                model: int(total_new_by_model.get(model, 0)) for model in MODEL_IDS
            },
            "expected_retries_upper_bound": total_new * assumptions.retry_max,
            "expected_token_usage": totals["token_usage"],
            "expected_cost": totals["cost"],
            "expected_wall_seconds": float(expected_wall),
            "eta_seconds": float(expected_wall),
            "expected_requests_per_hour": (
                None
                if expected_wall == 0
                else float(total_new * 3600.0 / expected_wall)
            ),
            "expected_wall_seconds_with_retry_upper_bound": float(
                expected_wall * (1 + assumptions.retry_max)
            ),
        },
        "assumptions": {
            "concurrency": assumptions.concurrency,
            "retry_max": assumptions.retry_max,
            "max_spend_usd": assumptions.max_spend_usd,
            "er2_cost_cap_per_request_usd": assumptions.er2_cost_cap_per_request_usd,
            "flash_input_usd_per_million": assumptions.flash_input_usd_per_million,
            "flash_output_including_thought_usd_per_million": assumptions.flash_output_including_thought_usd_per_million,
            "er2_input_usd_per_million": assumptions.er2_input_usd_per_million,
            "er2_output_including_thought_usd_per_million": assumptions.er2_output_including_thought_usd_per_million,
            "flash_cost_cap_per_request_usd": assumptions.flash_cost_cap_per_request_usd,
            "model_estimates": {
                model: asdict(model_estimates[model]) for model in MODEL_IDS
            },
            "token_estimate_provenance": "local Phase-B resolved smoke artifacts",
        },
        "cache_contract": {
            "cross_phase_reuse": "pilot P1 requests are reused by ablation A2",
            "stability_namespace_isolated": True,
            "replicate_id_enters_logical_cache_key": True,
        },
    }
    return write_immutable_json(Path(run_root) / "full_run_plan.json", plan)
