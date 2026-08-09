from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from failure_analysis.reranking_v2.schema import atomic_write_jsonl

from .feature_store import load_split_ids
from .schema import artifact_identity, atomic_write_json, read_jsonl, stable_sample_id


def _feature_identity(path: str|Path) -> dict[str,dict[str,list[str]]]:
    result={}
    for record in read_jsonl(path):
        sample_id=stable_sample_id(str(record["split"]),record["sample_id"]); candidates=record["candidates"]
        result[sample_id]={"candidate_ids":[str(value["candidate_id"]) for value in candidates],"candidate_checksums":[str(value["candidate_checksum"]) for value in candidates]}
    return result


def _partition_track(*, source_path: Path, allowed_ids: set[str], identities: dict[str,dict[str,list[str]]], output_path: Path) -> dict[str,Any]:
    selected=[]
    for record in read_jsonl(source_path):
        sample_id=str(record["sample_id"])
        if sample_id not in allowed_ids: continue
        expected=identities.get(sample_id)
        if expected is None: raise ValueError(f"label partition feature identity missing: {sample_id}")
        labels={str(value["candidate_id"]):value for value in record["candidate_labels"]}
        if set(labels)!=set(expected["candidate_ids"]): raise ValueError(f"label partition candidate IDs differ: {sample_id}")
        for candidate_id,checksum in zip(expected["candidate_ids"],expected["candidate_checksums"],strict=True):
            if str(labels[candidate_id]["candidate_checksum"])!=checksum: raise ValueError(f"label partition checksum differs: {sample_id}/{candidate_id}")
        selected.append(record)
    if {str(value["sample_id"]) for value in selected}!=allowed_ids or len(selected)!=len(allowed_ids): raise ValueError("label partition cohort coverage mismatch")
    selected.sort(key=lambda value:str(value["sample_id"])); atomic_write_jsonl(output_path,selected)
    return artifact_identity(output_path)


def build_v3_label_partitions(*, v2_root: str|Path, v3_split_manifest: str|Path, output_dir: str|Path) -> dict[str,Any]:
    """Create sealed physical label partitions before scientific selection.

    The operation performs only identity-checked mechanical partitioning; no
    correctness value is aggregated, displayed, or returned.
    """
    root=Path(v2_root); output=Path(output_dir)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    v2_split=root/"split_manifest.json"
    partitions={
        "train":(load_split_ids(v2_split,development_partition="train"),root/"base_train/features.jsonl",root/"labels_train"),
        "calibration":(load_split_ids(v2_split,development_partition="calibration"),root/"base_train/features.jsonl",root/"labels_train"),
        "v3_select":(load_split_ids(v3_split_manifest,v3_partition="v3_select"),root/"base_val/features.jsonl",root/"labels_val"),
        "v3_lockcheck":(load_split_ids(v3_split_manifest,v3_partition="v3_lockcheck"),root/"base_val/features.jsonl",root/"labels_val"),
    }
    artifacts={}; identity_cache={}
    for partition,(ids,feature_path,label_root) in partitions.items():
        identities=identity_cache.setdefault(str(feature_path),_feature_identity(feature_path)); artifacts[partition]={}
        for track,source_subdir in (("corrected_scientific","corrected"),("legacy_official_compatibility","legacy_official")):
            target=output/partition/track/"labels.jsonl"; target.parent.mkdir(parents=True,exist_ok=True)
            artifacts[partition][track]=_partition_track(source_path=label_root/source_subdir/"labels.jsonl",allowed_ids=ids,identities=identities,output_path=target)
    result={"schema_version":"3.0.0","kind":"v3_physically_isolated_label_partitions","status":"complete","partitions":artifacts,"counts":{name:len(value[0]) for name,value in partitions.items()},"v2_split":artifact_identity(v2_split),"v3_split":artifact_identity(v3_split_manifest),"operation":"identity-checked mechanical partition only; no metrics or label aggregates computed","test_labels_read":False}
    atomic_write_json(output/"artifact_manifest.json",result); return result
