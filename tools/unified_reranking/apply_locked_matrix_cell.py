"""Apply one Validation-trained matrix cell to label-free frozen Test features."""

from __future__ import annotations

import argparse
import json
import os
import pickletools
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# On macOS, initialize LightGBM's native runtime before PyTorch's OpenMP
# runtime.  Loading a Booster after importing torch can crash in LightGBM's
# C API instead of raising a Python exception.  Keep the dependency optional
# for neural-only environments, but never import it late on the LambdaMART
# path.
try:
    import lightgbm as _lightgbm
except ModuleNotFoundError:  # pragma: no cover - exercised by neural-only installs
    _lightgbm = None

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_reranking.contracts import assert_model_feature_columns
from unified_reranking.artifacts import (
    load_verified_json,
    verified_manifest_artifact,
    verify_artifact_records_recursive,
)
from unified_reranking.datasets import (
    FoldPreprocessor,
    build_inference_query_arrays,
    with_query_edge_features,
)
from unified_reranking.hashing import atomic_json, canonical_sha256, sha256_file
from unified_reranking.ledger import ledger_stage
from unified_reranking.models import (
    CompleteGraphGNNResidualScorer,
    DeepSetsResidualScorer,
    LinearResidualScorer,
    ResidualMLPScorer,
    SetTransformerResidualScorer,
)
from unified_reranking.training import predict_neural_ranker, set_deterministic_cpu
from unified_reranking.telemetry import (
    flatten_telemetry,
    lightgbm_parameter_count,
    missing_feature_rate,
    resolved_track_extraction_latency,
    telemetry_payload,
    torch_parameter_count,
)
from unified_reranking.test_access_guard import append_access_log


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _model(
    encoder: str,
    input_dim: int,
    alpha: float,
    *,
    attention_blocks: int,
    edge_dim: int,
) -> torch.nn.Module:
    if encoder == "linear":
        return LinearResidualScorer(input_dim, alpha=alpha)
    if encoder == "mlp":
        return ResidualMLPScorer(input_dim, alpha=alpha)
    if encoder == "deepsets":
        return DeepSetsResidualScorer(input_dim, alpha=alpha)
    if encoder == "set_transformer":
        return SetTransformerResidualScorer(
            input_dim, alpha=alpha, num_blocks=attention_blocks
        )
    if encoder == "gnn":
        return CompleteGraphGNNResidualScorer(input_dim, edge_dim, alpha=alpha)
    raise ValueError(f"unsupported neural encoder: {encoder}")


def _flat_inference_features(arrays: Any) -> np.ndarray:
    return arrays.features.numpy()[~arrays.padding_mask.numpy()]


def _lambdamart_predictions(model: Any, arrays: Any) -> pd.DataFrame:
    scores = np.asarray(model.predict(_flat_inference_features(arrays)), dtype=float)
    rows = []
    cursor = 0
    for sample_id, candidate_ids in zip(
        arrays.sample_ids, arrays.candidate_ids, strict=True
    ):
        for candidate_id in candidate_ids:
            rows.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "score": float(scores[cursor]),
                }
            )
            cursor += 1
    if cursor != len(scores):
        raise RuntimeError(
            "LambdaMART prediction length does not match candidate tensors"
        )
    return pd.DataFrame(rows)


class _NativeLightGBMRanker:
    """Minimal audited adapter around a natively restored LightGBM Booster."""

    def __init__(self, booster: Any) -> None:
        self.model = type("NativeLightGBMModel", (), {"booster_": booster})()

    def predict(self, features: Any) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        booster = self.model.booster_
        if matrix.ndim != 2 or matrix.shape[1] != int(booster.num_feature()):
            raise ValueError("LightGBM feature width differs from the locked model")
        scores = np.asarray(booster.predict(matrix), dtype=np.float64)
        if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
            raise RuntimeError("LightGBM returned invalid Test scores")
        return scores


def _load_native_lightgbm_ranker(model_path: Path) -> _NativeLightGBMRanker:
    """Restore the Booster string embedded in a verified legacy pickle safely.

    LightGBM's documented portable representation is its native model string.
    Parsing pickle opcodes does not execute constructors, while ``pickle.load``
    invokes ``Booster.__setstate__`` and can crash the process in some native
    library builds.  The training artifact contains exactly one such string.
    """

    candidates = [
        argument
        for _opcode, argument, _position in pickletools.genops(model_path.read_bytes())
        if isinstance(argument, str) and argument.startswith("tree\nversion=")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "locked LambdaMART pickle must contain exactly one native model string"
        )
    if _lightgbm is None:
        raise ModuleNotFoundError("LightGBM is required for LambdaMART Test inference")
    booster = _lightgbm.Booster(model_str=candidates[0])
    if int(booster.num_feature()) <= 0 or int(booster.num_trees()) <= 0:
        raise RuntimeError("locked LambdaMART native model is empty")
    return _NativeLightGBMRanker(booster)


def run(run_dir: Path, cell_manifest_path: Path) -> dict[str, Any]:
    feature_started = time.perf_counter()
    cell_manifest_path = cell_manifest_path.resolve()
    cell = load_verified_json(cell_manifest_path, name="Validation matrix cell")
    configuration = cell.get("configuration", {})
    if cell.get("status") != "COMPLETE" or configuration.get("mode") != "validation":
        raise ValueError(
            "Test inference requires one completed Validation-trained cell"
        )
    verify_artifact_records_recursive(
        {"sources": cell.get("sources"), "artifacts": cell.get("artifacts")},
        name="Validation matrix cell",
        require_at_least_one=True,
    )
    route = str(configuration["route"])
    track = str(configuration["track"])
    encoder = str(configuration["encoder"])
    seed = int(configuration["seed"])
    feature_dir = run_dir / "03_features" / "tracks" / track / f"{route}_test"
    feature_manifest_path = feature_dir / "feature_manifest.json"
    feature_manifest = load_verified_json(
        feature_manifest_path, name=f"{route}/{track}/test feature manifest"
    )
    feature_path = verified_manifest_artifact(
        feature_manifest, name=f"{route}/{track}/test candidate features"
    )
    feature_extraction_latency_ms, feature_extraction_latency_source = (
        resolved_track_extraction_latency(run_dir, track, feature_manifest)
    )
    columns = assert_model_feature_columns(feature_manifest["model_feature_columns"])
    trained_columns = tuple(map(str, cell["feature_columns"]))
    if tuple(columns) != trained_columns:
        raise RuntimeError("locked Train/Test model feature schemas differ")
    features = pd.read_parquet(feature_path)
    feature_missing_rate = missing_feature_rate(features, columns)
    preprocessor = FoldPreprocessor.from_artifact(cell["preprocessor"])
    if preprocessor.columns != tuple(columns):
        raise RuntimeError("persisted preprocessor schema differs from locked features")
    arrays = build_inference_query_arrays(features, preprocessor=preprocessor)

    relation_source: dict[str, Any] | None = None
    if encoder == "gnn":
        relation_metadata = cell.get("relations")
        if not isinstance(relation_metadata, dict):
            raise RuntimeError("GNN cell does not contain a locked relation contract")
        relation_path = (
            run_dir
            / "03_features"
            / "common"
            / f"{route}_test"
            / "candidate_relations.parquet"
        )
        relation_columns = assert_model_feature_columns(relation_metadata["columns"])
        relation_preprocessor = FoldPreprocessor.from_artifact(
            relation_metadata["preprocessor"]
        )
        if relation_preprocessor.columns != tuple(relation_columns):
            raise RuntimeError("persisted relation preprocessor schema differs")
        arrays = with_query_edge_features(
            arrays,
            pd.read_parquet(relation_path),
            preprocessor=relation_preprocessor,
        )
        relation_source = {
            "path": str(relation_path.resolve()),
            "sha256": sha256_file(relation_path),
            "columns": list(relation_columns),
            "schema_sha256": canonical_sha256(relation_columns),
        }

    if len(features) <= 0:
        raise RuntimeError("locked Test feature table has no candidates")
    feature_latency_ms = (
        (time.perf_counter() - feature_started) * 1000.0 / len(features)
    )

    model_path = Path(cell["artifacts"]["model"]["path"]).resolve()
    if sha256_file(model_path) != cell["artifacts"]["model"]["sha256"]:
        raise RuntimeError("locked Validation model hash mismatch")
    if encoder == "lambdamart":
        model = _load_native_lightgbm_ranker(model_path)
        ranker_started = time.perf_counter()
        predictions = _lambdamart_predictions(model, arrays)
        ranker_latency_ms = (
            (time.perf_counter() - ranker_started) * 1000.0 / len(features)
        )
        parameter_count = lightgbm_parameter_count(model)
    else:
        set_deterministic_cpu(seed)
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        model = _model(
            encoder,
            int(payload["input_dim"]),
            float(payload["alpha"]),
            attention_blocks=int(payload.get("num_attention_blocks", 2)),
            edge_dim=int(payload.get("edge_dim", 1)),
        )
        model.load_state_dict(payload["state_dict"])
        ranker_started = time.perf_counter()
        predictions = predict_neural_ranker(model, arrays)
        ranker_latency_ms = (
            (time.perf_counter() - ranker_started) * 1000.0 / len(features)
        )
        parameter_count = torch_parameter_count(model)

    telemetry = telemetry_payload(
        phase="label_free_test_application",
        parameter_count=parameter_count,
        ranker_latency_ms=ranker_latency_ms,
        feature_latency_ms=feature_latency_ms,
        missing_feature_rate_value=feature_missing_rate,
    )

    candidate_path = run_dir / "02_candidates" / f"{route}_test_top5.parquet"
    candidates = pd.read_parquet(
        candidate_path, columns=["sample_id", "candidate_id", "native_rank"]
    )
    expected = set(
        map(tuple, candidates[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    observed = set(
        map(tuple, predictions[["sample_id", "candidate_id"]].astype(str).to_numpy())
    )
    if (
        expected != observed
        or predictions[["sample_id", "candidate_id"]].duplicated().any()
    ):
        raise RuntimeError(
            "Test predictions do not preserve frozen candidate membership"
        )
    predictions = candidates.merge(
        predictions, on=["sample_id", "candidate_id"], validate="one_to_one"
    )
    identity = {
        "source_cell_sha256": sha256_file(cell_manifest_path),
        "route": route,
        "track": track,
        "encoder": encoder,
        "loss": configuration["loss"],
        "seed": seed,
        "split": "test",
        "test_features_sha256": sha256_file(feature_path),
        "test_feature_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_extraction_benchmark_sha256": None
        if feature_extraction_latency_source is None
        else feature_extraction_latency_source["sha256"],
        "test_candidates_sha256": sha256_file(candidate_path),
        "test_relations_sha256": None
        if relation_source is None
        else relation_source["sha256"],
    }
    application_id = canonical_sha256(identity)[:16]
    output = run_dir / "09_formal_test" / "label_free_cell_predictions" / application_id
    prediction_path = output / "predictions.parquet"
    marker = output / "manifest.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "COMPLETE"
            and previous.get("identity") == identity
        ):
            verify_artifact_records_recursive(
                {
                    "sources": previous.get("sources"),
                    "artifact": previous.get("artifact"),
                },
                name=f"completed Test application {application_id}",
                require_at_least_one=True,
            )
            append_access_log(
                run_dir,
                {
                    "event": "prelock_label_free_test_stage",
                    "stage": "matrix_cell_test_application",
                    "application_id": application_id,
                    "output_manifest": str(marker.resolve()),
                    "output_manifest_sha256": sha256_file(marker),
                    "candidate_labels_opened_as_table": False,
                    "resumed": True,
                },
            )
            return previous
    _atomic_parquet(prediction_path, predictions)
    result = {
        "status": "COMPLETE",
        "label_free_test_inference": True,
        "candidate_test_labels_read": False,
        "identity": identity,
        "application_id": application_id,
        "candidate_rows": len(predictions),
        "candidate_bearing_samples": int(predictions["sample_id"].nunique()),
        "telemetry": telemetry,
        **flatten_telemetry(telemetry),
        "feature_extraction_latency_ms": float(feature_extraction_latency_ms),
        "sources": {
            "cell_manifest": {
                "path": str(cell_manifest_path),
                "sha256": sha256_file(cell_manifest_path),
            },
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "features": {
                "path": str(feature_path.resolve()),
                "sha256": sha256_file(feature_path),
            },
            "feature_manifest": {
                "path": str(feature_manifest_path.resolve()),
                "sha256": sha256_file(feature_manifest_path),
            },
            "feature_extraction_benchmark": feature_extraction_latency_source,
            "candidates": {
                "path": str(candidate_path.resolve()),
                "sha256": sha256_file(candidate_path),
            },
            "relations": relation_source,
        },
        "artifact": {
            "path": str(prediction_path.resolve()),
            "sha256": sha256_file(prediction_path),
        },
    }
    atomic_json(marker, result)
    append_access_log(
        run_dir,
        {
            "event": "prelock_label_free_test_stage",
            "stage": "matrix_cell_test_application",
            "application_id": application_id,
            "output_manifest": str(marker.resolve()),
            "output_manifest_sha256": sha256_file(marker),
            "candidate_labels_opened_as_table": False,
            "resumed": False,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cell-manifest", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cell_manifest = args.cell_manifest.resolve()
    cell_key = sha256_file(cell_manifest)[:16]
    with ledger_stage(
        run_dir / "run_ledger.sqlite",
        stage="P11_PRELOCK",
        substage=f"label_free_test_cell_{cell_key}",
        command=" ".join(map(str, sys.argv)),
    ) as state:
        result = run(run_dir, cell_manifest)
        artifact = Path(result["artifact"]["path"])
        state["artifact_path"] = str(artifact)
        state["artifact_sha256"] = sha256_file(artifact)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
