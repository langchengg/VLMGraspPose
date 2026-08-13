from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from d1_reranking import postformal_sources
from unified_reranking.hashing import canonical_sha256, sha256_file


def _record(path: Path) -> dict[str, object]:
    source = path.resolve()
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _write_content(path: Path, value: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _manifest(
    path: Path,
    *,
    artifacts: dict[str, object],
    **extra: object,
) -> Path:
    return _write_content(
        path,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "candidate_test_labels_read": False,
            "artifacts": artifacts,
            **extra,
        },
    )


def _synthetic_sources(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, selected_method: str = "R3"
) -> dict[str, object]:
    (root / "manifest.json").write_text(
        json.dumps({"denominator_contract": {"test_samples": 2}}),
        encoding="utf-8",
    )
    paired = _write_parquet(
        root / postformal_sources.FIXED_INPUTS["paired"],
        [{"sample_id": "sample-0"}, {"sample_id": "sample-1"}],
    )
    masks = root / "synthetic_assets"
    masks.mkdir(parents=True, exist_ok=True)
    small_mask = masks / "sample-0.png"
    large_mask = masks / "sample-1.png"
    Image.fromarray(np.pad(np.ones((2, 2), dtype=np.uint8) * 255, 9)).save(small_mask)
    Image.fromarray(np.ones((20, 20), dtype=np.uint8) * 255).save(large_mask)

    candidates = _write_parquet(
        root / "synthetic_inputs" / "top5.parquet",
        [
            {"sample_id": "sample-0", "candidate_id": "c0", "native_rank": 1},
            {"sample_id": "sample-0", "candidate_id": "c1", "native_rank": 2},
        ],
    )
    raw = _write_parquet(
        root / "synthetic_inputs" / "raw.parquet",
        [
            {
                "sample_id": "sample-0",
                "candidate_id": "c0",
                "native_rank": 1,
                "mask_reliability": 0.7,
                "rectangle_probability_mean": 0.3,
                "depth_missing": 0.0,
                "number_of_nearby_candidates": 3.0,
            },
            {
                "sample_id": "sample-0",
                "candidate_id": "c1",
                "native_rank": 2,
                "mask_reliability": 0.8,
                "rectangle_probability_mean": 0.6,
                "depth_missing": 0.0,
                "number_of_nearby_candidates": 1.0,
            },
        ],
    )
    context = _write_parquet(
        root / "synthetic_inputs" / "context.parquet",
        [
            {
                "sample_id": "sample-0",
                "language": "object left of cup",
                "predicted_mask_path": str(small_mask.resolve()),
                "predicted_mask_sha256": sha256_file(small_mask),
                "candidate_count": 2,
            },
            {
                "sample_id": "sample-1",
                "language": "plain object",
                "predicted_mask_path": str(large_mask.resolve()),
                "predicted_mask_sha256": sha256_file(large_mask),
                "candidate_count": 0,
            },
        ],
    )
    t2 = _write_parquet(
        root / "synthetic_inputs" / "t2.parquet",
        [
            {"sample_id": "sample-0", "candidate_id": "c0"},
            {"sample_id": "sample-0", "candidate_id": "c1"},
        ],
    )
    t3 = _write_parquet(
        root / "synthetic_inputs" / "t3.parquet",
        [
            {"sample_id": "sample-0", "candidate_id": "c0"},
            {"sample_id": "sample-0", "candidate_id": "c1"},
        ],
    )
    decisions = _write_parquet(
        root / "synthetic_inputs" / "ranker_decisions.parquet",
        [
            {
                "sample_id": "sample-0",
                "native_candidate_id": "c0",
                "selected_candidate_id": "c1",
            },
            {
                "sample_id": "sample-1",
                "native_candidate_id": None,
                "selected_candidate_id": None,
            },
        ],
    )
    gate = _write_parquet(
        root / "synthetic_inputs" / "gate_decisions.parquet",
        pd.read_parquet(decisions).to_dict("records"),
    )
    sentinel = root / "synthetic_inputs" / "feature_execution.txt"
    sentinel.write_text("complete", encoding="utf-8")
    model = root / "synthetic_inputs" / "ranker.model"
    model.write_bytes(b"synthetic-ranker")

    _manifest(
        root / postformal_sources.FIXED_INPUTS["candidates"],
        artifacts={"top5": _record(candidates)},
    )
    _manifest(
        root / postformal_sources.FIXED_INPUTS["raw_features"],
        artifacts={
            "candidate_features": _record(raw),
            "sample_context": _record(context),
        },
        telemetry={"feature_latency_ms": 1.25, "peak_memory_mb": 16.0},
    )
    _manifest(
        root / postformal_sources.FIXED_INPUTS["t2_features"],
        artifacts={"candidate_features": _record(t2)},
    )
    _manifest(
        root / postformal_sources.FIXED_INPUTS["t3_features"],
        artifacts={"candidate_features": _record(t3)},
    )
    _manifest(
        root / postformal_sources.FIXED_INPUTS["feature_execution"],
        artifacts={"execution": _record(sentinel)},
    )
    ranker_extra: dict[str, object] = {
        "selected_method": selected_method,
        "telemetry": {
            "ranker_latency_ms": 0.5,
            "peak_memory_mb": 4.0,
            "parameter_count": 12,
        },
        "seed_applications": {"0": {"model": _record(model)}},
    }
    _manifest(
        root / postformal_sources.FIXED_INPUTS["ranker"],
        artifacts={"per_sample_decisions": _record(decisions)},
        **ranker_extra,
    )
    _manifest(
        root / postformal_sources.FIXED_INPUTS["gate"],
        artifacts={"decisions": _record(gate)},
    )
    closure_path = _write_content(
        root / "synthetic_closure.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "canonical_inputs": {
                "test": {"paired_manifest": _record(paired)},
            },
        },
    )
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        postformal_sources,
        "load_source_closure",
        lambda _root: (closure_path, closure),
    )
    return closure


def test_fixed_label_free_sources_are_deterministic_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _synthetic_sources(tmp_path, monkeypatch)
    result = postformal_sources.assemble_postformal_sources(tmp_path)

    assert result["sample_count"] == 2
    assert result["covariate_schema"] == list(postformal_sources.COVARIATE_COLUMNS)
    covariates = pd.read_parquet(tmp_path / postformal_sources.COVARIATES_RELATIVE_PATH)
    assert list(covariates.columns) == list(postformal_sources.COVARIATE_COLUMNS)
    first = covariates.set_index("sample_id").loc["sample-0"]
    assert first["target_size"] == "small"
    assert bool(first["relation_query"])
    assert bool(first["clutter"])
    assert first["native_mask_support"] == pytest.approx(0.3)
    assert first["selected_mask_support"] == pytest.approx(0.6)
    assert pd.isna(
        covariates.set_index("sample_id").loc["sample-1", "native_mask_support"]
    )

    runtime = result["artifacts"]["runtime_manifests"]
    assert set(runtime) == set(postformal_sources.RUNTIME_COMPONENTS)
    statuses = {
        component: json.loads(Path(record["path"]).read_text(encoding="utf-8"))[
            "status"
        ]
        for component, record in runtime.items()
    }
    assert statuses["feature_extraction"] == "COMPLETE"
    assert statuses["ranker_inference"] == "COMPLETE"
    assert statuses["dexnet_candidate_generation"] == "NOT_AVAILABLE"
    assert statuses["gate_inference"] == "NOT_AVAILABLE"
    assert statuses["total_d1_pipeline"] == "NOT_AVAILABLE"
    contribution = json.loads(
        (tmp_path / postformal_sources.CONTRIBUTIONS_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )
    assert contribution["status"] == "NOT_APPLICABLE"
    access_events = [
        json.loads(line)
        for line in (tmp_path / "09_formal_test" / "test_access.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert access_events[-1]["stage"] == "d1_postformal_label_free_sources"
    assert access_events[-1]["candidate_test_labels_read"] is False

    resumed = postformal_sources.assemble_postformal_sources(tmp_path, resume=True)
    assert resumed == result
    assert len(
        (tmp_path / "09_formal_test" / "test_access.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == len(access_events)

    mask = tmp_path / "synthetic_assets" / "sample-0.png"
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(mask)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        postformal_sources.assemble_postformal_sources(tmp_path, resume=True)


def test_r5_requires_separate_locked_native_contributions_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _synthetic_sources(tmp_path, monkeypatch, selected_method="R5")
    with pytest.raises((FileNotFoundError, RuntimeError), match="candidate contributions"):
        postformal_sources.assemble_postformal_sources(tmp_path)
    assert not (tmp_path / postformal_sources.SOURCE_MANIFEST_RELATIVE_PATH).exists()
    assert not (tmp_path / postformal_sources.COVARIATES_RELATIVE_PATH).exists()
    assert not (tmp_path / postformal_sources.CONTRIBUTIONS_RELATIVE_PATH).exists()
