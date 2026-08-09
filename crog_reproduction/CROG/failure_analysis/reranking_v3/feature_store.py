from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .candidate_relations import G10_FEATURE_NAMES, extract_candidate_relation_features
from .depth_geometry import DERIVED_DEPTH_FEATURE_NAMES, derived_depth_features
from .derived_features import DERIVED_HEAD_FEATURE_NAMES, derived_head_features
from .feature_data import ARRAY_KEYS, apply_normalizer
from .schema import artifact_identity, atomic_write_json, read_jsonl, sha256_file, stable_sample_id


@dataclass(frozen=True)
class FeatureLocation:
    root: Path
    record: dict[str, Any]


class FeatureCatalog:
    """Read-only catalog over completed, independently resumable V3 shards."""

    def __init__(
        self,
        artifact_dirs: Sequence[str | Path],
        *,
        head_override_dirs: Sequence[str | Path] = (),
        candidate_feature_paths: Sequence[str | Path] = (),
    ):
        if not artifact_dirs:
            raise ValueError("at least one feature artifact is required")
        self.artifact_dirs = tuple(Path(value).resolve() for value in artifact_dirs)
        self.manifests: list[dict[str, Any]] = []
        self.schema: dict[str, Any] | None = None
        self.locations: dict[str, FeatureLocation] = {}
        self.by_npz: dict[Path, list[FeatureLocation]] = {}
        self.head_override_dirs = tuple(Path(value).resolve() for value in head_override_dirs)
        self.head_overrides: dict[str,np.ndarray] = {}
        self.head_override_manifests: list[dict[str,Any]] = []
        self.candidate_feature_paths = tuple(Path(value).resolve() for value in candidate_feature_paths)
        self.candidate_records: dict[str,dict[str,Any]] = {}
        self.derived_head_cache:dict[str,np.ndarray]={}
        self.derived_depth_cache:dict[str,np.ndarray]={}
        for root in self.artifact_dirs:
            manifest_path = root / "artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise ValueError(f"feature artifact is incomplete: {root}")
            current_schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
            if self.schema is None:
                self.schema = current_schema
            elif current_schema != self.schema:
                raise ValueError("feature schema differs across artifacts")
            self.manifests.append(manifest)
            for record in read_jsonl(root / "index.jsonl"):
                sample_id = str(record["sample_id"])
                if sample_id in self.locations:
                    raise ValueError(f"duplicate sample across feature artifacts: {sample_id}")
                location = FeatureLocation(root, record)
                self.locations[sample_id] = location
                npz_path = root / "shards" / str(record["shard"])
                self.by_npz.setdefault(npz_path, []).append(location)
        if self.schema is None:
            raise AssertionError("feature schema was not loaded")
        self.base_schema = copy.deepcopy(self.schema)
        for root in self.head_override_dirs:
            overlay_manifest=json.loads((root/"artifact_manifest.json").read_text())
            if overlay_manifest.get("status")!="complete" or overlay_manifest.get("artifact_type")!="fullchain_head_feature_correction_overlay": raise ValueError(f"invalid head correction overlay: {root}")
            if json.loads((root/"feature_schema.json").read_text())!=self.schema: raise ValueError("head correction feature schema mismatch")
            overlay_rows=read_jsonl(root/"index.jsonl"); by_shard:dict[str,list[dict[str,Any]]]={}
            for record in overlay_rows: by_shard.setdefault(str(record["shard"]),[]).append(record)
            for shard_name,records in by_shard.items():
                records.sort(key=lambda value:int(value["offset"]))
                with np.load(root/"shards"/shard_name) as payload:
                    sample_ids=np.asarray(payload["sample_ids"]).astype(str); heads=np.asarray(payload["head_features"],dtype=np.float32)
                if len(sample_ids)!=len(records) or heads.shape!=(len(records),5,len(self.schema["head_feature_names"])): raise ValueError("head correction overlay shape mismatch")
                for offset,(sample_id,record) in enumerate(zip(sample_ids,records,strict=True)):
                    if str(record["sample_id"])!=sample_id or sample_id in self.head_overrides: raise ValueError("head correction overlay identity/uniqueness mismatch")
                    location=self.locations.get(sample_id)
                    if location is None or list(map(str,location.record["candidate_ids"]))!=list(map(str,record["candidate_ids"])) or list(map(str,location.record["candidate_checksums"]))!=list(map(str,record["candidate_checksums"])): raise ValueError("head correction candidate identity mismatch")
                    self.head_overrides[sample_id]=heads[offset]
            self.head_override_manifests.append(overlay_manifest)
        if self.head_override_dirs and set(self.head_overrides)!=set(self.locations): raise ValueError("head correction overlay must exactly cover the feature catalog")
        for source in self.candidate_feature_paths:
            for value in read_jsonl(source):
                sample_id=stable_sample_id(str(value["split"]),value["sample_id"])
                if sample_id in self.candidate_records: raise ValueError(f"duplicate frozen candidate record: {sample_id}")
                candidates=value["candidates"]
                expected=self.locations.get(sample_id)
                if expected is None: continue
                ids=[str(item["candidate_id"]) for item in candidates]
                checksums=[str(item["candidate_checksum"]) for item in candidates]
                if ids!=list(map(str,expected.record["candidate_ids"])) or checksums!=list(map(str,expected.record["candidate_checksums"])): raise ValueError(f"frozen candidate identity mismatch: {sample_id}")
                required=("candidate_id","candidate_checksum","q_rank","q_raw","cx","cy","width_px","height_px","angle_deg")
                self.candidate_records[sample_id]={"candidates":[{key:item[key] for key in required} for item in candidates]}
        if self.candidate_feature_paths:
            if set(self.candidate_records)!=set(self.locations): raise ValueError("frozen candidate sources must exactly cover the feature catalog")
            self.schema=copy.deepcopy(self.base_schema)
            self.schema["head_feature_names"]=[*self.base_schema["head_feature_names"],*DERIVED_HEAD_FEATURE_NAMES,*G10_FEATURE_NAMES]
            self.schema["depth_feature_names"]=[*self.base_schema["depth_feature_names"],*DERIVED_DEPTH_FEATURE_NAMES]
            self.schema["derived_head_feature_names"]=list(DERIVED_HEAD_FEATURE_NAMES)
            self.schema["derived_depth_feature_names"]=list(DERIVED_DEPTH_FEATURE_NAMES)
            self.schema["derived_relation_feature_names"]=list(G10_FEATURE_NAMES)
        self.q_index = self.schema["head_feature_names"].index("g0_q_raw")

    def assert_exact_ids(self, expected_ids: set[str]) -> None:
        observed = set(self.locations)
        if observed != expected_ids:
            raise ValueError(
                f"feature cohort mismatch: missing={sorted(expected_ids-observed)[:5]} "
                f"extra={sorted(observed-expected_ids)[:5]}"
            )

    def candidate_identity(self, allowed_ids: set[str]) -> dict[str,dict[str,list[str]]]:
        missing=allowed_ids-set(self.locations)
        if missing: raise ValueError(f"candidate identity request missing IDs: {sorted(missing)[:5]}")
        return {sample_id:{"candidate_ids":list(map(str,self.locations[sample_id].record["candidate_ids"])),"candidate_checksums":list(map(str,self.locations[sample_id].record["candidate_checksums"]))} for sample_id in allowed_ids}

    def provenance(self) -> dict[str, Any]:
        return {
            "artifact_dirs": [str(value) for value in self.artifact_dirs],
            "artifact_manifest_sha256": [sha256_file(value / "artifact_manifest.json") for value in self.artifact_dirs],
            "content_sha256": [value["content_sha256"] for value in self.manifests],
            "row_count": len(self.locations),
            "unique_sample_count": len(self.locations),
            "unique_candidate_count": len(self.locations) * 5,
            "feature_schema_hash": self.manifests[0]["feature_schema_hash"],
            "head_correction_overlays": [str(value) for value in self.head_override_dirs],
            "head_correction_content_sha256": [value["content_sha256"] for value in self.head_override_manifests],
            "candidate_feature_sources": [artifact_identity(value) for value in self.candidate_feature_paths],
            "derived_head_feature_count": len(DERIVED_HEAD_FEATURE_NAMES) if self.candidate_feature_paths else 0,
            "derived_depth_feature_count": len(DERIVED_DEPTH_FEATURE_NAMES) if self.candidate_feature_paths else 0,
            "g10_relation_feature_count": len(G10_FEATURE_NAMES) if self.candidate_feature_paths else 0,
        }

    def iter_batches(
        self,
        *,
        allowed_ids: set[str],
        labels: Mapping[str, np.ndarray] | None,
        priors: Mapping[str, np.ndarray] | None,
        normalizers: Mapping[str, Mapping[str, np.ndarray]] | None,
        batch_size: int,
        seed: int,
        shuffle: bool,
        array_keys: Sequence[str] = ARRAY_KEYS,
    ) -> Iterator[dict[str, Any]]:
        missing = allowed_ids - set(self.locations)
        if missing:
            raise ValueError(f"feature store is missing requested IDs: {sorted(missing)[:5]}")
        shard_paths = [
            path for path, values in self.by_npz.items()
            if any(str(value.record["sample_id"]) in allowed_ids for value in values)
        ]
        rng = np.random.default_rng(int(seed))
        if shuffle:
            rng.shuffle(shard_paths)
        requested_keys = tuple(array_keys)
        unknown = set(requested_keys) - set(ARRAY_KEYS)
        if unknown or "head_features" not in requested_keys:
            raise ValueError(f"invalid feature array selection: unknown={sorted(unknown)}")
        pending: dict[str, list[np.ndarray]] = {name: [] for name in requested_keys}
        pending_ids: list[str] = []
        pending_records: list[dict[str, Any]] = []

        def emit(count: int) -> dict[str, Any]:
            result = {
                name: np.stack(pending[name][:count]) for name in requested_keys
            }
            del_ids = pending_ids[:count]
            del_records = pending_records[:count]
            del pending_ids[:count]
            del pending_records[:count]
            for name in requested_keys:
                del pending[name][:count]
            result["sample_ids"] = np.asarray(del_ids)
            result["records"] = del_records
            result["q"] = result["head_features"][:, :, self.q_index].astype(np.float32)
            if labels is not None:
                result["labels"] = np.stack([labels[value] for value in del_ids]).astype(np.float32)
            if priors is not None:
                result["prior"] = np.stack([priors[value] for value in del_ids]).astype(np.float32)
            else:
                result["prior"] = np.zeros((count, 5, 80), np.float32)
            if normalizers is not None:
                for name in ("head_features", "depth_features", "prior"):
                    if name in result:
                        result[name] = apply_normalizer(result[name], normalizers[name])
            return result

        for shard_path in shard_paths:
            requested = [
                value for value in self.by_npz[shard_path]
                if str(value.record["sample_id"]) in allowed_ids
            ]
            if shuffle:
                rng.shuffle(requested)
            with np.load(shard_path) as shard:
                loaded={name:np.asarray(shard[name]) for name in requested_keys}
                need_crops=self.candidate_feature_paths and any(str(value.record["sample_id"]) not in self.derived_head_cache or ("depth_features" in requested_keys and str(value.record["sample_id"]) not in self.derived_depth_cache) for value in requested)
                if need_crops and "crops" not in loaded: loaded["crops"]=np.asarray(shard["crops"])
                for location in requested:
                    offset = int(location.record["offset"])
                    sample_id = str(location.record["sample_id"])
                    if str(shard["sample_ids"][offset]) != sample_id:
                        raise ValueError(f"feature index/shard identity mismatch: {sample_id}")
                    base_head=self.head_overrides[sample_id] if self.head_overrides else np.asarray(loaded["head_features"][offset])
                    crop=np.asarray(loaded["crops"][offset],dtype=np.float32) if need_crops else None
                    frozen=self.candidate_records.get(sample_id)
                    for name in requested_keys:
                        value=base_head if name=="head_features" else np.asarray(loaded[name][offset])
                        if self.candidate_feature_paths and name=="head_features":
                            augmentation=self.derived_head_cache.get(sample_id)
                            if augmentation is None:
                                candidates=frozen["candidates"]
                                derived=np.stack([derived_head_features(crop[index],candidate,candidates,image_shape=tuple(map(int,location.record["original_size"]))) for index,candidate in enumerate(candidates)])
                                relation_names,relations=extract_candidate_relation_features(candidates,image_shape=tuple(map(int,location.record["original_size"])),head_features=base_head,head_feature_names=self.base_schema["head_feature_names"])
                                if relation_names!=G10_FEATURE_NAMES: raise AssertionError("G10 feature schema changed")
                                augmentation=np.concatenate((derived,relations),axis=-1).astype(np.float32); self.derived_head_cache[sample_id]=augmentation
                            value=np.concatenate((np.asarray(base_head,dtype=np.float32),augmentation),axis=-1)
                        elif self.candidate_feature_paths and name=="depth_features":
                            augmentation=self.derived_depth_cache.get(sample_id)
                            if augmentation is None: augmentation=np.stack([derived_depth_features(value_crop) for value_crop in crop]); self.derived_depth_cache[sample_id]=augmentation
                            value=np.concatenate((np.asarray(value,dtype=np.float32),augmentation),axis=-1)
                        pending[name].append(value)
                    pending_ids.append(sample_id)
                    pending_records.append(location.record)
                    if len(pending_ids) == int(batch_size):
                        yield emit(int(batch_size))
        if pending_ids:
            yield emit(len(pending_ids))


def load_split_ids(
    split_manifest: str | Path,
    *,
    development_partition: str | None = None,
    v3_partition: str | None = None,
) -> set[str]:
    payload = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    result = set()
    for row in payload["rows"]:
        if development_partition is not None and row.get("development_partition") != development_partition:
            continue
        if v3_partition is not None and row.get("v3_partition") != v3_partition:
            continue
        result.add(str(row["sample_id"]))
    if not result:
        raise ValueError("split selection produced an empty cohort")
    return result


def load_label_lookup(path: str | Path, *, allowed_ids: set[str], candidate_identity: Mapping[str,Mapping[str,Sequence[str]]] | None=None) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for record in read_jsonl(path):
        sample_id = str(record["sample_id"])
        if sample_id not in allowed_ids:
            continue
        candidates={str(value["candidate_id"]):value for value in record["candidate_labels"]}
        if len(candidates)!=5: raise ValueError(f"label candidate identity invalid: {sample_id}")
        if candidate_identity is not None:
            expected=candidate_identity.get(sample_id)
            if expected is None: raise ValueError(f"candidate identity missing for label join: {sample_id}")
            order=list(map(str,expected["candidate_ids"])); checksums=list(map(str,expected["candidate_checksums"]))
            if set(candidates)!=set(order): raise ValueError(f"label candidate IDs differ: {sample_id}")
            for candidate_id,checksum in zip(order,checksums,strict=True):
                if str(candidates[candidate_id].get("candidate_checksum"))!=checksum: raise ValueError(f"label candidate checksum differs: {sample_id}/{candidate_id}")
        else: order=[str(value["candidate_id"]) for value in record["candidate_labels"]]
        values = np.asarray([float(candidates[value]["candidate_correct"]) for value in order],dtype=np.float32)
        if values.shape != (5,):
            raise ValueError(f"label shape mismatch: {sample_id}")
        result[sample_id] = values
    missing = allowed_ids - set(result)
    if missing:
        raise ValueError(f"label artifact is missing IDs: {sorted(missing)[:5]}")
    return result


def prior_array_to_lookup(sample_ids: np.ndarray, values: np.ndarray) -> dict[str, np.ndarray]:
    if values.shape != (len(sample_ids), 5, 80):
        raise ValueError(f"V2 prior shape mismatch: {values.shape}")
    return {
        str(sample_id): np.asarray(values[index], dtype=np.float32)
        for index, sample_id in enumerate(sample_ids)
    }


def fit_streaming_normalizers(
    catalog: FeatureCatalog,
    *,
    allowed_ids: set[str],
    priors: Mapping[str, np.ndarray],
    batch_size: int = 64,
) -> dict[str, dict[str, np.ndarray]]:
    totals: dict[str, np.ndarray] = {}
    squares: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for batch in catalog.iter_batches(
        allowed_ids=allowed_ids,
        labels=None,
        priors=priors,
        normalizers=None,
        batch_size=batch_size,
        seed=0,
        shuffle=False,
        array_keys=("head_features", "depth_features"),
    ):
        for name in ("head_features", "depth_features", "prior"):
            values = np.asarray(batch[name], dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"non-finite value while fitting {name}")
            flattened = values.reshape(-1, values.shape[-1])
            totals[name] = totals.get(name, np.zeros(values.shape[-1], np.float64)) + flattened.sum(0)
            squares[name] = squares.get(name, np.zeros(values.shape[-1], np.float64)) + np.square(flattened).sum(0)
            counts[name] = counts.get(name, 0) + len(flattened)
    result = {}
    for name in totals:
        mean = totals[name] / counts[name]
        variance = np.maximum(squares[name] / counts[name] - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-6] = 1.0
        result[name] = {
            "median": mean.astype(np.float32),
            "mean": mean.astype(np.float32),
            "scale": scale.astype(np.float32),
        }
    return result


def write_partition_view(
    catalog: FeatureCatalog,
    *,
    allowed_ids: set[str],
    partition: str,
    output_dir: str|Path,
) -> dict[str,Any]:
    missing=allowed_ids-set(catalog.locations)
    if missing: raise ValueError(f"partition view missing features: {sorted(missing)[:5]}")
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False); index_path=output/"index.jsonl"
    from failure_analysis.reranking_v2.schema import atomic_write_jsonl
    records=[]
    for sample_id in sorted(allowed_ids):
        location=catalog.locations[sample_id]
        records.append({"schema_version":"3.0.0","kind":"fullchain_partition_view_row","partition":partition,"sample_id":sample_id,"parent_artifact":str(location.root),"parent_shard":str(location.record["shard"]),"parent_offset":int(location.record["offset"]),"candidate_ids":location.record["candidate_ids"],"candidate_checksums":location.record["candidate_checksums"]})
    atomic_write_jsonl(index_path,records)
    manifest={"schema_version":"3.0.0","artifact_type":"fullchain_partition_view","status":"complete","partition":partition,"row_count":len(records),"unique_sample_count":len(records),"unique_candidate_count":len(records)*5,"missing_count":0,"fallback_count":0,"parents":catalog.provenance(),"index":artifact_identity(index_path),"storage_semantics":"read-only zero-copy view; arrays remain in immutable parent NPZ shards"}
    atomic_write_json(output/"artifact_manifest.json",manifest); return manifest
