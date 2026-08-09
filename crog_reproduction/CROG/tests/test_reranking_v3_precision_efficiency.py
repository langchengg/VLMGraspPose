from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from failure_analysis.reranking_v3.precision_efficiency import (
    _main,
    array_storage_statistics,
    audit_float16_storage_roundtrip,
    audit_npz_float16_storage,
    compare_precision_arrays,
    compare_prediction_precision,
    directory_statistics,
    measure_efficiency,
    write_precision_efficiency_report,
)
from failure_analysis.reranking_v3.schema import canonical_json, sha256_bytes


def test_fp16_array_error_metrics_are_exact_and_aggregated() -> None:
    reference = {
        "a": np.asarray([[1.0, 1.0003], [0.0, -2.0007]], dtype=np.float32),
        "b": np.asarray([[0.25], [4.125]], dtype=np.float32),
    }
    cached = {name: value.astype(np.float16) for name, value in reference.items()}
    result = compare_precision_arrays(reference, cached)
    assert result["array_count"] == 2
    assert result["element_count"] == 6
    assert set(result["per_array"]) == {"a", "b"}
    assert result["per_array"]["a"]["shape"] == [2, 2]
    assert result["per_array"]["a"]["reference_dtype"] == "float32"
    assert result["per_array"]["a"]["cache_dtype"] == "float16"
    assert result["overall"]["max_abs_error"] >= result["overall"]["mean_abs_error"] >= 0.0
    assert result["overall"]["max_relative_error"] >= result["overall"]["mean_relative_error"] >= 0.0
    assert set(result["overall"]["abs_error_quantiles"]) == {"p50", "p90", "p95", "p99", "p99_9"}
    assert result["overall"]["abs_error_quantiles"]["p99_9"] <= result["overall"]["max_abs_error"]


@pytest.mark.parametrize("failure", ["shape", "nonfinite", "dtype", "keys"])
def test_fp16_array_comparison_rejects_invalid_inputs(failure: str) -> None:
    reference = {"x": np.ones((2, 3), dtype=np.float32)}
    cached = {"x": np.ones((2, 3), dtype=np.float16)}
    if failure == "shape":
        cached["x"] = np.ones((3, 2), dtype=np.float16)
    elif failure == "nonfinite":
        cached["x"][0, 0] = np.inf
    elif failure == "dtype":
        cached["x"] = cached["x"].astype(np.float32)
    else:
        cached = {"y": cached["x"]}
    with pytest.raises(ValueError):
        compare_precision_arrays(reference, cached)


def test_prediction_precision_reports_top1_and_full_order_consistency() -> None:
    scores32 = np.asarray([
        [1.0, 0.5, 0.5001],
        [0.5001, 0.5002, 0.0],
    ], dtype=np.float32)
    probabilities32 = np.asarray([
        [0.9, 0.4, 0.4001],
        [0.5001, 0.5002, 0.1],
    ], dtype=np.float32)
    result = compare_prediction_precision(
        reference_scores=scores32,
        cached_scores=scores32.astype(np.float16),
        reference_probabilities=probabilities32,
        cached_probabilities=probabilities32.astype(np.float16),
    )
    score_ranking = result["ranking"]["scores"]
    assert score_ranking["sample_count"] == 2
    assert score_ranking["top1_consistent_count"] == 1
    assert score_ranking["top1_consistency"] == pytest.approx(0.5)
    assert score_ranking["full_order_consistent_count"] == 0
    assert score_ranking["full_order_consistency"] == 0.0
    assert result["precision"]["per_array"]["probabilities"]["max_abs_error"] > 0.0


def test_float16_roundtrip_audits_feature_error_and_float32_prediction_effect() -> None:
    features = {
        "head": np.asarray([
            [[1.0003, 0.5], [0.5001, 0.1], [0.25, -0.1]],
            [[0.5001, 0.2], [0.5002, 0.2], [0.1, 0.0]],
        ], dtype=np.float32),
    }

    def predictor(arrays):
        scores = np.asarray(arrays["head"], dtype=np.float32).sum(axis=-1)
        probabilities = (1.0 / (1.0 + np.exp(-scores))).astype(np.float32)
        return {"scores": scores.astype(np.float32), "probabilities": probabilities}

    result = audit_float16_storage_roundtrip(features, predictor=predictor)
    assert result["labels_read"] is False
    assert result["feature_precision"]["per_array"]["head"]["max_abs_error"] > 0.0
    effect = result["prediction_effect"]
    assert 0.0 <= effect["top_rank_parity"]["scores"] <= 1.0
    assert 0.0 <= effect["top_rank_parity"]["probabilities"] <= 1.0
    assert set(effect["scores"]["abs_error_quantiles"]) == {"p50", "p90", "p95", "p99", "p99_9"}


def test_independent_npz_audit_and_module_entrypoint(tmp_path: Path, capsys) -> None:
    reference_features = tmp_path / "features_f32.npz"
    cached_features = tmp_path / "features_f16.npz"
    reference_predictions = tmp_path / "predictions_f32.npz"
    cached_predictions = tmp_path / "predictions_f16.npz"
    feature = np.asarray([[1.0003, 0.5001], [0.25, -2.0007]], dtype=np.float32)
    scores = np.asarray([[1.0, 0.5001, 0.5], [0.5001, 0.5002, 0.0]], dtype=np.float32)
    probabilities = np.asarray([[0.9, 0.4001, 0.4], [0.5001, 0.5002, 0.1]], dtype=np.float32)
    np.savez_compressed(reference_features, feature=feature)
    np.savez_compressed(cached_features, feature=feature.astype(np.float16))
    np.savez_compressed(reference_predictions, scores=scores, probabilities=probabilities)
    np.savez_compressed(
        cached_predictions, scores=scores.astype(np.float16),
        probabilities=probabilities.astype(np.float16),
    )
    audit = audit_npz_float16_storage(
        reference_features_path=reference_features,
        cached_features_path=cached_features,
        feature_keys=("feature",),
        reference_predictions_path=reference_predictions,
        cached_predictions_path=cached_predictions,
    )
    assert audit["feature_precision"]["overall"]["max_abs_error"] > 0.0
    assert audit["top_rank_parity"]["scores"] == pytest.approx(0.5)
    assert audit["sources"]["reference_features"]["sha256"]

    output = tmp_path / "audit.json"
    assert _main([
        "--reference-features", str(reference_features),
        "--cached-features", str(cached_features),
        "--feature-key", "feature",
        "--reference-predictions", str(reference_predictions),
        "--cached-predictions", str(cached_predictions),
        "--output", str(output),
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert printed["top_rank_parity"] == audit["top_rank_parity"]
    assert report["precision"]["kind"] == "v3_independent_npz_float16_storage_audit"
    assert report["labels_read"] is False


def test_array_and_directory_storage_statistics(tmp_path: Path) -> None:
    arrays = {
        "scores": np.zeros((4, 5), dtype=np.float32),
        "embeddings": np.zeros((4, 5, 3), dtype=np.float16),
    }
    storage = array_storage_statistics(arrays, sample_count=4)
    assert storage["total_bytes"] == arrays["scores"].nbytes + arrays["embeddings"].nbytes
    assert storage["bytes_per_sample"] == storage["total_bytes"] / 4

    (tmp_path / "nested").mkdir()
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / "nested" / "b.json").write_bytes(b"12345")
    disk = directory_statistics(tmp_path, sample_count=4)
    assert disk["file_count"] == 2
    assert disk["directory_count"] == 1
    assert disk["total_bytes"] == 8
    assert disk["bytes_per_sample"] == 2.0
    assert disk["bytes_by_suffix"] == {".bin": 3, ".json": 5}
    assert disk["largest_file"] == {"path": "nested/b.json", "bytes": 5}


def test_efficiency_protocol_is_repeatable_with_injected_clock_and_rss(tmp_path: Path) -> None:
    clock_values = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
    rss_values = iter([1000, 1400])
    calls = []
    (tmp_path / "cache.bin").write_bytes(b"x" * 20)
    result = measure_efficiency(
        lambda: calls.append("operation"),
        sample_count=10,
        warmup=2,
        repeat=3,
        storage_bytes=200,
        disk_directory=tmp_path,
        synchronize=lambda: calls.append("sync"),
        clock=lambda: next(clock_values),
        rss_reader=lambda: next(rss_values),
    )
    assert calls.count("operation") == 5
    assert result["latency_seconds"]["values"] == pytest.approx([0.1, 0.2, 0.3])
    assert result["latency_seconds"]["mean"] == pytest.approx(0.2)
    assert result["throughput_samples_per_second"]["mean"] == pytest.approx(50.0)
    assert result["rss"]["increase_bytes"] == 400
    assert result["storage_bytes_per_sample"] == 20.0
    assert result["disk"]["total_bytes"] == 20


def test_json_report_is_immutable_and_content_hashed(tmp_path: Path) -> None:
    reference = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    precision = compare_precision_arrays(reference, {"x": reference["x"].astype(np.float16)})
    output = tmp_path / "report.json"
    result = write_precision_efficiency_report(
        output,
        precision=precision,
        metadata={"warmup_contract": "synthetic"},
    )
    observed = json.loads(output.read_text(encoding="utf-8"))
    content_sha = observed.pop("content_sha256")
    assert content_sha == sha256_bytes(canonical_json(observed).encode("utf-8"))
    assert result["labels_read"] is False
    assert observed["metadata"] == {"warmup_contract": "synthetic"}
    with pytest.raises(FileExistsError):
        write_precision_efficiency_report(output, precision=precision)
