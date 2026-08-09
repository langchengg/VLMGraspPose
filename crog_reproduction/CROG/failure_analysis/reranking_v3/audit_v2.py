from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from failure_analysis.reranking_v2.datasets import RELATIONAL_FEATURES
from failure_analysis.reranking_v2.protocol import verify_lock
from failure_analysis.reranking_v2.schema import SCALAR_FEATURES

from .schema import atomic_write_json, atomic_write_text, sha256_file


USAGE_COLUMNS = (
    "feature_name", "source_stage", "source_tensor", "dimension",
    "candidate_specific_or_query_global", "raw_probability_or_derived",
    "coordinate_frame", "normalization", "train_fitted_parameter",
    "oof_required", "oof_verified", "used_by_locked_primary",
    "used_by_exploratory_method_only", "missing_value_behavior",
    "test_coverage", "inference_allowed", "code_location", "artifact_provenance",
)


def _row(name: str, stage: str, tensor: str, dimension: int, *, used: bool,
         raw: str = "derived", coordinate: str = "candidate/list",
         normalization: str = "none", fitted: str = "none", oof: bool = False,
         exploratory: bool = False, missing: str = "not_applicable",
         location: str, provenance: str) -> dict[str, Any]:
    return {
        "feature_name": name,
        "source_stage": stage,
        "source_tensor": tensor,
        "dimension": int(dimension),
        "candidate_specific_or_query_global": "candidate_specific",
        "raw_probability_or_derived": raw,
        "coordinate_frame": coordinate,
        "normalization": normalization,
        "train_fitted_parameter": fitted,
        "oof_required": bool(oof),
        "oof_verified": bool(oof and used),
        "used_by_locked_primary": bool(used),
        "used_by_exploratory_method_only": bool(exploratory),
        "missing_value_behavior": missing,
        "test_coverage": "V2 focused suite + V3 exact replay/identity gate",
        "inference_allowed": True,
        "code_location": location,
        "artifact_provenance": provenance,
    }


def reconstruct_v2_feature_usage(v2_root: str | Path) -> list[dict[str, Any]]:
    root = Path(v2_root).resolve()
    rows = []
    for name in SCALAR_FEATURES:
        rows.append(_row(
            name, "V1 candidate scalar export", f"candidate.features.{name}", 3,
            used=True, normalization="SetRank mean/std; gate median/mean/std",
            fitted="development-fold normalizers", oof=True,
            missing="value=0,reliability=0,missing=1",
            location="failure_analysis/reranking_v2/datasets.py:31-49",
            provenance=str(root / "oof_base/oof_base_predictions.npz"),
        ))
    for name in RELATIONAL_FEATURES:
        rows.append(_row(
            name, "candidate list context", name, 1, used=True,
            normalization="SetRank mean/std; gate median/mean/std",
            fitted="development-fold normalizers", oof=True,
            location="failure_analysis/reranking_v2/datasets.py:68-114",
            provenance=str(root / "oof_base/oof_base_predictions.npz"),
        ))
    rows.extend([
        _row("rgbd_critic_score", "14-channel aligned critic", "critic_logit", 1, used=True, raw="raw_logit", normalization="model trained", fitted="critic checkpoint", oof=True, location="failure_analysis/reranking_v2/inference.py:141-185", provenance=str(root / "oof_base/oof_base_predictions.npz")),
        _row("rgbd_critic_probability", "14-channel aligned critic", "sigmoid(critic_logit)", 1, used=True, raw="probability", normalization="sigmoid", fitted="critic checkpoint", oof=True, location="failure_analysis/reranking_v2/inference.py:141-185", provenance=str(root / "oof_base/oof_base_predictions.npz")),
        _row("rgbd_critic_embedding", "14-channel aligned critic", "critic_embedding", 64, used=True, normalization="SetRank mean/std", fitted="critic+SetRank checkpoints", oof=True, location="failure_analysis/reranking_v2/inference.py:141-225", provenance=str(root / "oof_base/oof_base_predictions.npz")),
        _row("pre_decoder_latent", "CROG decoder", "latent_pre", 1024, used=False, exploratory=True, coordinate="candidate-aligned ROI", location="failure_analysis/reranking_v2/extract.py:416-545", provenance=str(root / "enhanced_train")),
        _row("post_decoder_latent", "CROG decoder", "latent_post", 1024, used=True, coordinate="candidate-aligned ROI", normalization="latent model internal", fitted="latent residual checkpoint", oof=True, location="failure_analysis/reranking_v2/oof.py:203-234", provenance=str(root / "oof_base/oof_base_predictions.npz")),
        _row("latent_score", "latent residual ranker", "latent_scores", 1, used=True, oof=True, location="failure_analysis/reranking_v2/inference.py:141-225", provenance=str(root / "oof_base/oof_base_predictions.npz")),
        _row("latent_residual", "latent residual ranker", "latent_residuals", 1, used=True, oof=True, location="failure_analysis/reranking_v2/inference.py:141-225", provenance=str(root / "oof_base/oof_base_predictions.npz")),
        _row("setrank_score", "SetRank", "setrank.scores", 1, used=True, raw="derived score", oof=True, location="failure_analysis/reranking_v2/inference.py:188-427", provenance=str(root / "oof_primary/oof_setrank_predictions.npz")),
        _row("setrank_probability", "SetRank", "setrank.probabilities", 1, used=True, raw="probability", oof=True, location="failure_analysis/reranking_v2/inference.py:188-244", provenance=str(root / "oof_primary/oof_setrank_predictions.npz")),
        _row("setrank_internal_embedding", "SetRank", "encoder_state", 64, used=False, exploratory=False, location="failure_analysis/reranking_v2/models/setrank.py:20-66", provenance="not exported"),
        _row("stability_mean", "perturbation stability", "stable_scores(kappa=0)", 1, used=True, location="failure_analysis/reranking_v2/inference.py:247-384", provenance=str(root / "formal_test_primary_v2/stability_test/stability.npz")),
        _row("stability_std", "perturbation stability", "statistics.std", 1, used=False, exploratory=True, location="failure_analysis/reranking_v2/models/uncertainty.py", provenance="statistics JSONL only"),
        _row("stability_min", "perturbation stability", "statistics.minimum", 1, used=False, exploratory=True, location="failure_analysis/reranking_v2/models/uncertainty.py", provenance="statistics JSONL only"),
        _row("ensemble_vote", "three-seed gate", "seed_selected_indices", 1, used=True, raw="derived vote", fitted="required_consensus=2", location="failure_analysis/reranking_v2/inference.py:247-384", provenance=str(root / "formal_test_primary_v2/primary_predictions/predictions.jsonl")),
        _row("crog_raw_output_logits", "CROG projector", "M/Q/W raw logits", 5, used=False, exploratory=False, coordinate="native 104x104", location="model/layers.py:47-132", provenance="not persisted by V1/V2"),
        _row("token_level_text", "CLIP text encoder", "Ft", 512, used=False, exploratory=False, coordinate="query-global tokens", location="model/clip.py:439-456", provenance="not persisted by V1/V2"),
        _row("multiscale_c3_c4_c5_fpn", "CLIP+FPN", "C3/C4/C5/Fm", 0, used=False, exploratory=False, coordinate="multiscale feature maps", location="model/clip.py:207-223; model/layers.py:342-398", provenance="not persisted by V1/V2"),
    ])
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=USAGE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# V2 Locked Primary: Intended vs Actually Consumed Features",
        "",
        "事实优先级为运行代码、checkpoint schema 与 immutable artifact；文档名称不作为消费证据。",
        "",
        "| Feature | Stage | Dim | Locked primary | OOF verified | Provenance |",
        "|---|---|---:|:---:|:---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['feature_name']} | {row['source_stage']} | {row['dimension']} | {'yes' if row['used_by_locked_primary'] else 'no'} | {'yes' if row['oof_verified'] else 'no'} | {row['artifact_provenance']} |")
    lines.extend([
        "", "## 强制结论", "",
        "V2 实际消费 60d scalar/list、14-channel critic score/probability/64d embedding、post-decoder latent score/residual、SetRank score/probability、207d R/H/N gate、三 seed consensus 与 kappa=0 的 stability mean。",
        "",
        "V2 未直接消费 pre-decoder latent、SetRank embedding、stability std/min、五个 CROG raw logits/maps、token-level text、C3/C4/C5/FPN 或 decoder 各层。它不是仅有 ‘Full + critic’，但也不是 full-chain。",
    ])
    atomic_write_text(path, "\n".join(lines) + "\n")


def audit_v2_locked_primary(v2_root: str | Path, output_dir: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    root = Path(v2_root).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    lock = verify_lock(root / "frozen_experiment_manifest.json", repo_root=repo_root)
    rows = reconstruct_v2_feature_usage(root)
    json_path = output / "V2_PRIMARY_FEATURE_USAGE.json"
    csv_path = output / "V2_PRIMARY_FEATURE_USAGE.csv"
    md_path = output / "V2_PRIMARY_FEATURE_USAGE.md"
    atomic_write_json(json_path, {"schema": list(USAGE_COLUMNS), "features": rows})
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    result = {
        "v2_root": str(root),
        "lock_file_sha256": sha256_file(root / "frozen_experiment_manifest.json"),
        "lock_sha256": lock["lock_sha256"],
        "primary_method": lock["primary_method"],
        "feature_rows": len(rows),
        "used_rows": sum(bool(row["used_by_locked_primary"]) for row in rows),
        "outputs": {path.name: sha256_file(path) for path in (json_path, csv_path, md_path)},
    }
    return result


def write_replay_record(output_dir: str | Path, *, expected_path: str | Path, observed_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    output = Path(output_dir)
    expected_sha = sha256_file(expected_path)
    observed_sha = sha256_file(observed_path)
    record = {
        **metadata,
        "expected": {"path": str(Path(expected_path).resolve()), "sha256": expected_sha},
        "observed": {"path": str(Path(observed_path).resolve()), "sha256": observed_sha},
        "byte_exact": expected_sha == observed_sha,
    }
    if not record["byte_exact"]:
        raise AssertionError("V2 replay is not byte-exact; formal training/test is prohibited")
    path = output / "V2_PRIMARY_REPLAY.json"
    atomic_write_json(path, record)
    atomic_write_text(output / "V2_PRIMARY_REPLAY_SHA256.txt", f"{sha256_file(path)}  {path.name}\n")
    return record

