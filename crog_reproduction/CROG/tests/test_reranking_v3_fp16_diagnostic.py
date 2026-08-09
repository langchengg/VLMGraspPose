from __future__ import annotations

import numpy as np

from failure_analysis.reranking_v3.fp16_diagnostic import (
    _stored_float32_checks,
    _write_candidate_subset,
)
from failure_analysis.reranking_v3.schema import canonical_json
from failure_analysis.reranking_v3.fullchain_extractor import (
    FLOAT16_STORAGE_FIELDS,
    _materialize_fullchain_shard,
)


def _pending() -> tuple[list[dict[str, str]], dict[str, list[np.ndarray]]]:
    records = [{"sample_id":"development:7"}]
    values = {
        "head_features":[np.asarray([[1.123456]],np.float32)],
        "depth_features":[np.asarray([[2.123456]],np.float32)],
        "token_ids":[np.asarray([1,2],np.int64)],
    }
    for index,name in enumerate(FLOAT16_STORAGE_FIELDS, start=1):
        values[name]=[np.asarray([[index+0.123456]],np.float32)]
    return records,values


def test_precast_materialization_precedes_and_differs_from_float16_cache() -> None:
    records,pending=_pending()
    reference=_materialize_fullchain_shard(records,pending,precast_reference=True)
    cache=_materialize_fullchain_shard(records,pending,precast_reference=False)
    for name in FLOAT16_STORAGE_FIELDS:
        assert reference[name].dtype==np.float32
        assert cache[name].dtype==np.float16
        assert not np.array_equal(reference[name],cache[name].astype(np.float32))
    assert reference["head_features"].dtype==cache["head_features"].dtype==np.float32
    assert reference["depth_features"].dtype==cache["depth_features"].dtype==np.float32
    np.testing.assert_array_equal(reference["head_features"],cache["head_features"])
    np.testing.assert_array_equal(reference["depth_features"],cache["depth_features"])


def test_persisted_float32_checks_report_no_quantization() -> None:
    values={
        "head_features":np.asarray([[[1.25]]],np.float32),
        "depth_features":np.asarray([[[2.5]]],np.float32),
    }
    result=_stored_float32_checks(values,values)
    assert result["head_features"]["bitwise_equal"] is True
    assert result["depth_features"]["max_abs_error"]==0.0
    assert result["head_features"]["cast_to_float16"] is False


def test_candidate_subset_uses_canonical_stable_ids(tmp_path) -> None:
    source=tmp_path/"source.jsonl"
    source.write_text(canonical_json({"split":"train","sample_id":7,"candidates":[]})+"\n")
    destination=tmp_path/"subset.jsonl"
    identity=_write_candidate_subset(
        source,destination,sample_ids={"multiple:train:00000007"},
    )
    assert identity["path"]==str(destination.resolve())
    assert destination.read_text()==source.read_text()


def test_candidate_subset_rejects_nested_label_fields(tmp_path) -> None:
    source=tmp_path/"source.jsonl"
    source.write_text(canonical_json({
        "split":"train","sample_id":7,"candidates":[{"candidate_correct":1}],
    })+"\n")
    try:
        _write_candidate_subset(
            source,tmp_path/"subset.jsonl",sample_ids={"multiple:train:00000007"},
        )
    except ValueError as error:
        assert "candidate_correct" in str(error)
    else:
        raise AssertionError("nested label-like field was accepted")
