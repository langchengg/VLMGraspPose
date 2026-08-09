from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from failure_analysis.reranking_v2.schema import atomic_write_jsonl

from .output_map_features import CORRECTED_WIDTH_FEATURE_NAMES, extract_head_features
from .schema import artifact_identity, atomic_write_json, canonical_json, read_jsonl, sha256_bytes, sha256_file, stable_sample_id


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try: np.savez_compressed(temporary,**arrays); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


def build_width_feature_correction(
    *, source_artifact_dir: str|Path, frozen_features_path: str|Path,
    output_dir: str|Path,
) -> dict[str,Any]:
    source=Path(source_artifact_dir).resolve(); output=Path(output_dir).resolve()
    if output.exists(): raise FileExistsError(output)
    manifest=json.loads((source/"artifact_manifest.json").read_text())
    if manifest.get("status")!="complete": raise ValueError("source feature artifact is incomplete")
    schema=json.loads((source/"feature_schema.json").read_text()); names=list(schema["head_feature_names"])
    target_indices=[names.index(value) for value in CORRECTED_WIDTH_FEATURE_NAMES]
    features={stable_sample_id(str(value["split"]),value["sample_id"]):value for value in read_jsonl(frozen_features_path)}
    index=read_jsonl(source/"index.jsonl"); by_shard:dict[str,list[dict[str,Any]]]={}
    for record in index: by_shard.setdefault(str(record["shard"]),[]).append(record)
    output.mkdir(parents=True); (output/"shards").mkdir(); output_records=[]; changed_max={value:0.0 for value in CORRECTED_WIDTH_FEATURE_NAMES}; row_count=0
    for shard_name,records in sorted(by_shard.items()):
        records.sort(key=lambda value:int(value["offset"])); source_path=source/"shards"/shard_name
        with np.load(source_path) as shard:
            sample_ids=np.asarray(shard["sample_ids"]).astype(str); corrected=np.asarray(shard["head_features"],dtype=np.float32).copy(); crops=np.asarray(shard["crops"],dtype=np.float32)
        if [str(value["sample_id"]) for value in records] != sample_ids.tolist(): raise ValueError("source feature index/shard order mismatch")
        for offset,(sample_id,record) in enumerate(zip(sample_ids,records,strict=True)):
            feature=features.get(str(sample_id))
            if feature is None: raise ValueError(f"frozen feature record missing {sample_id}")
            candidates=feature["candidates"]
            ids=[str(value["candidate_id"]) for value in candidates]; checksums=[str(value["candidate_checksum"]) for value in candidates]
            if ids!=list(map(str,record["candidate_ids"])) or checksums!=list(map(str,record["candidate_checksums"])): raise ValueError(f"candidate identity mismatch {sample_id}")
            for candidate_index,candidate in enumerate(candidates):
                original_size=tuple(int(value) for value in record["original_size"])
                if len(original_size)!=2 or min(original_size)<=0:
                    raise ValueError(f"invalid original image size {sample_id}: {original_size}")
                observed_names,values=extract_head_features(crops[offset,candidate_index],candidate,candidates,image_shape=original_size)
                if observed_names!=names: raise ValueError("corrected feature schema changed")
                for target_index,target_name in zip(target_indices,CORRECTED_WIDTH_FEATURE_NAMES,strict=True):
                    new=float(values[target_index]); old=float(corrected[offset,candidate_index,target_index]); changed_max[target_name]=max(changed_max[target_name],abs(new-old)); corrected[offset,candidate_index,target_index]=new
            output_records.append({**record,"shard":shard_name,"offset":offset,"correction":"crog_width_factor_100px_and_opening_axis_mask_thickness_v1"}); row_count+=1
        _write_npz(output/"shards"/shard_name,sample_ids=sample_ids,head_features=corrected)
    atomic_write_jsonl(output/"index.jsonl",output_records)
    atomic_write_json(output/"feature_schema.json",schema)
    outputs=[output/"index.jsonl",output/"feature_schema.json",*sorted((output/"shards").glob("*.npz"))]
    result={
        "schema_version":"3.0.0","artifact_type":"fullchain_head_feature_correction_overlay","status":"complete",
        "created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_artifact":artifact_identity(source/"artifact_manifest.json"),"source_content_sha256":manifest["content_sha256"],
        "frozen_features":artifact_identity(frozen_features_path),"feature_schema_hash":sha256_file(output/"feature_schema.json"),
        "correction":"CROG candidate width is sigmoid(W)*100px; mask thickness measured along opening axis and converted to pixels",
        "corrected_feature_names":list(CORRECTED_WIDTH_FEATURE_NAMES),"maximum_absolute_change":changed_max,
        "row_count":row_count,"unique_sample_count":len({value["sample_id"] for value in output_records}),"unique_candidate_count":row_count*5,
        "missing_count":0,"fallback_count":0,"labels_read":False,"outputs":[artifact_identity(value) for value in outputs],
    }
    result["content_sha256"]=sha256_bytes(canonical_json(result).encode()); atomic_write_json(output/"artifact_manifest.json",result); return result
