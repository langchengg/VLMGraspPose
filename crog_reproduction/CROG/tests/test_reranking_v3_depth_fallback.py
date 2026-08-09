from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from failure_analysis.reranking_v3 import fullchain_extractor
from failure_analysis.reranking_v3.aligned_crops import CROP_CHANNELS, build_fullchain_crop
from failure_analysis.reranking_v3.depth_fallback import (
    checkpoint_ensemble_uses_depth,
    merge_depth_fallback_rankings,
    plan_depth_aware_execution,
)
from failure_analysis.reranking_v3.formal_backend import (
    _verify_source_files,
    build_formal_source_file_manifest,
)
from failure_analysis.reranking_v3.fullchain_extractor import (
    _load_depth_m,
    _normalise_depth_array,
)
from failure_analysis.reranking_v3.schema import read_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in rows))
    return path


def test_uint16_mm_and_float_m_depth_units_are_explicit() -> None:
    millimetres=np.asarray([[0,1000],[2500,3000]],np.uint16)
    depth,status=_normalise_depth_array(millimetres,image_shape=(2,2))
    assert status["available"] is True
    assert status["scale_source"]=="known_uint16_millimetres"
    np.testing.assert_allclose(depth,[[0,1],[2.5,3]])
    metres=np.asarray([[0,1.25],[2.5,3]],np.float32)
    depth,status=_normalise_depth_array(metres,image_shape=(2,2))
    assert status["available"] is True
    assert status["scale_source"]=="known_float_metres"
    np.testing.assert_array_equal(depth,metres)


def test_metadata_scale_is_validated_and_conflicts_fail_safe() -> None:
    raw=np.asarray([[1000]],np.uint16)
    depth,status=_normalise_depth_array(
        raw,image_shape=(1,1),metadata={"depth_unit":"mm","depth_scale_to_m":.001},
    )
    assert status["available"] is True and depth.item()==pytest.approx(1.0)
    depth,status=_normalise_depth_array(
        raw,image_shape=(1,1),metadata={"depth_unit":"m","depth_scale_to_m":.001},
    )
    assert status["available"] is False
    assert status["reason"]=="conflicting_unit_metadata"
    assert np.count_nonzero(depth)==0


@pytest.mark.parametrize(
    ("record_factory","reason"),
    [
        (lambda root: {},"depth_path_missing"),
        (lambda root: {"depth_path":str(root/"absent.png")},"depth_file_missing"),
        (lambda root: {"depth_path":str((root/"corrupt.png").write_bytes(b"bad") and root/"corrupt.png")},"depth_unreadable"),
    ],
)
def test_missing_and_corrupt_depth_emit_zero_evidence(
    tmp_path: Path, record_factory, reason: str,
) -> None:
    record=record_factory(tmp_path)
    depth,status=_load_depth_m(record,image_shape=(4,5))
    assert depth.shape==(4,5) and depth.dtype==np.float32
    assert np.count_nonzero(depth)==0
    assert status["available"] is False
    assert status["reason"]==reason and status["valid_fraction"]==0.0


def test_nan_depth_is_fail_safe_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path=tmp_path/"depth.exr"; path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        fullchain_extractor.cv2,"imread",
        lambda *_args,**_kwargs: np.asarray([[1.0,np.nan],[2.0,3.0]],np.float32),
    )
    depth,status=_load_depth_m({"depth_path":str(path)},image_shape=(2,2))
    assert status["available"] is False and status["reason"]=="nonfinite_values"
    assert np.count_nonzero(depth)==0


def test_zero_fallback_depth_produces_zero_valid_crop_and_fraction() -> None:
    crop,metadata=build_fullchain_crop(
        {"cx":4.0,"cy":4.0,"width_px":4.0,"height_px":2.0,"angle_deg":0.0},
        rgb=torch.zeros((1,3,8,8)),depth_m=torch.zeros((1,1,8,8)),
        raw_heads=tuple(torch.zeros((1,1,4,4)) for _ in range(5)),
        forward_affine=np.asarray([[1,0,0],[0,1,0]],np.float32),output_size=8,
    )
    assert np.count_nonzero(crop[CROP_CHANNELS.index("relative_depth_m")])==0
    assert np.count_nonzero(crop[CROP_CHANNELS.index("depth_valid")])==0
    assert metadata["depth_valid_fraction"]==0.0
    assert metadata["local_depth_median_m"] is None


def test_native_does_not_require_depth_and_rgbd_plans_deterministic_fallback() -> None:
    ids={"a","b"}; status={
        "a":{"available":True,"reason":None},
        "b":{"available":False,"reason":"depth_unreadable"},
    }
    native=plan_depth_aware_execution(
        sample_ids=ids,depth_status=status,use_depth=False,native_fallback_declared=False,
    )
    assert native["model_sample_ids"]==ids and not native["fallback_sample_ids"]
    rgbd=plan_depth_aware_execution(
        sample_ids=ids,depth_status=status,use_depth=True,native_fallback_declared=True,
    )
    assert rgbd["model_sample_ids"]=={"a"}
    assert rgbd["fallback_sample_ids"]=={"b"} and rgbd["fallback_source"]=="native"
    v2=plan_depth_aware_execution(
        sample_ids=ids,depth_status=status,use_depth=True,native_fallback_declared=False,
    )
    assert v2["fallback_source"]=="v2"


def _ranking(sample_id: str, method: str, selected: int) -> dict:
    order=[f"c{selected}",*[f"c{i}" for i in range(5) if i!=selected]]
    return {
        "sample_id":sample_id,"method":method,"candidate_order":order,
        "selection":{"selected_candidate_id":order[0],"selected_index":selected},
    }


@pytest.mark.parametrize("native",[False,True])
def test_rgbd_missing_depth_never_scores_missing_and_uses_exact_fallback_order(
    tmp_path: Path, native: bool,
) -> None:
    primary=_write_jsonl(tmp_path/"primary.jsonl",[_ranking("a","rgbd",2)])
    v2=_write_jsonl(tmp_path/"v2.jsonl",[_ranking("a","v2",1),_ranking("b","v2",1)])
    native_path=(
        _write_jsonl(tmp_path/"native.jsonl",[_ranking("a","native",3),_ranking("b","native",3)])
        if native else None
    )
    output=merge_depth_fallback_rankings(
        ordered_sample_ids=["a","b"],method="rgbd",primary_ranking_path=primary,
        missing_depth_ids={"b"},v2_ranking_path=v2,native_ranking_path=native_path,
        output_path=tmp_path/"merged.jsonl",
        depth_status={"a":{"available":True,"reason":None},"b":{"available":False,"reason":"depth_unreadable"}},
    )
    rows=list(read_jsonl(output))
    assert rows[0]["candidate_order"][0]=="c2"
    assert rows[1]["candidate_order"][0]==("c3" if native else "c1")
    assert rows[1]["depth_fallback"]=={
        "applied":True,"source":"native" if native else "v2",
        "reason":"depth_unreadable","rgbd_model_executed":False,
    }


def test_checkpoint_depth_mode_must_be_consistent(tmp_path: Path) -> None:
    paths=[]
    for index in range(3):
        path=tmp_path/f"model-{index}.pt"
        torch.save({"status":"complete","config":{"use_depth":True}},path); paths.append(path)
    assert checkpoint_ensemble_uses_depth(paths) is True
    torch.save({"status":"complete","config":{"use_depth":False}},paths[-1])
    with pytest.raises(ValueError,match="mixes Native and RGB-D"):
        checkpoint_ensemble_uses_depth(paths)


def test_formal_source_manifest_declares_missing_depth_without_hashing_it(tmp_path: Path) -> None:
    rgb=tmp_path/"rgb.png"; assert cv2.imwrite(str(rgb),np.zeros((2,2,3),np.uint8))
    missing=tmp_path/"missing-depth.png"
    candidate=_write_jsonl(tmp_path/"features.jsonl",[{
        "split":"test","sample_id":0,"image_path":str(rgb),"depth_path":str(missing),
        "candidates":[],
    }])
    source_manifest=tmp_path/"sources.json"
    result=build_formal_source_file_manifest(
        candidate_artifact=candidate,output_path=source_manifest,
    )
    assert result["source_file_count"]==1
    assert result["declared_missing_depth_count"]==1
    assert result["sources"][0]["depth"]=={
        "status":"missing","reason":"depth_file_missing","path":str(missing.resolve()),
    }
    _verify_source_files(source_manifest,candidate)
    missing.write_bytes(b"appeared-after-lock")
    with pytest.raises(ValueError,match="source-file manifest differs"):
        _verify_source_files(source_manifest,candidate)
