#!/usr/bin/env python3
"""Rank frozen post-NMS Dex-Net candidates without sampling or GQ-CNN."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from ruamel.yaml import YAML


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.geometric_ranker import (  # noqa: E402
    evaluate_planar_annotation_consistency,
    load_frozen_candidates,
    load_intrinsics,
    load_ocid_vlg_grasps,
    load_processed_mask,
    make_final_grasp,
    rank_candidates,
    save_deterministic_npz,
    save_ranked_csv,
    save_ranked_npz,
    save_strict_json,
    sha256_file,
    _ranking_arrays,
)
from src.grasping.grasp_visualization import save_candidate_overlay  # noqa: E402


SUMMARY_FIELDS = (
    "sample_id",
    "query",
    "candidate_count",
    "top1_candidate_id",
    "top1_geometric_score",
    "center_u_px",
    "center_v_px",
    "angle_deg",
    "contact_span_px",
    "top1_2d_rectangle_accuracy",
    "top5_2d_recall",
    "first_2d_matching_rank",
    "ranking_time_ms",
    "source_candidates_npz_sha256",
    "source_npz_unchanged",
    "failure_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_roots",
        nargs="*",
        type=Path,
        default=[
            REPO_ROOT / "outputs" / "dexnet_candidates_one_sample",
            REPO_ROOT / "outputs" / "dexnet_candidates_ten_samples",
        ],
        help="Roots containing existing sample directories (defaults to one and ten outputs)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "dexnet_geometric_ranker.yaml",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT.parent / "crog_reproduction" / "OCID-VLG" / "refer" / "unique" / "test_expressions.json",
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--no-evaluation", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    config = yaml.load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"ranker config must be a mapping: {path}")
    return config


def _sample_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"output root does not exist: {root}")
    samples = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "candidates.npz").is_file()
    )
    if not samples:
        raise FileNotFoundError(f"no frozen candidate sample directories under {root}")
    return samples


def _write_summary(root: Path, rows: list[dict[str, Any]]) -> Path:
    destination = root / "geometric_ranking_summary.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _empty_row(sample_id: str) -> dict[str, Any]:
    return {**{field: "" for field in SUMMARY_FIELDS}, "sample_id": sample_id}


def rank_sample(
    sample_dir: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    annotations: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_npz = sample_dir / "candidates.npz"
    records, candidate_metadata, source_hashes = load_frozen_candidates(source_npz)
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if candidate_metadata and candidate_metadata.get("sample_id") != metadata.get("sample_id"):
        raise ValueError("candidates.json metadata and metadata.json sample IDs disagree")
    depth_m = np.load(sample_dir / "depth_m.npy", allow_pickle=False)
    mask = load_processed_mask(sample_dir / "hifics_mask_processed.png")
    intrinsics = load_intrinsics(sample_dir / "camera.intr")
    ranked, breakdown = rank_candidates(
        records,
        depth_m=depth_m,
        target_mask=mask,
        intrinsics=intrinsics,
        config=config,
    )

    evaluation = None
    if annotations is not None:
        grasps = load_ocid_vlg_grasps(annotations, int(metadata["question_index"]))
        evaluation_config = dict(config["evaluation"])
        evaluation_config["top_k"] = int(config["ranking"]["top_k"])
        evaluation = evaluate_planar_annotation_consistency(ranked, grasps, evaluation_config)
        by_id = {item["candidate_id"]: item for item in evaluation["per_candidate"]}
        for record in ranked:
            record["ocid_vlg_2d_consistency"] = by_id[record["candidate_id"]]

    top1 = ranked[0]
    final = make_final_grasp(top1, camera_frame=intrinsics.frame)
    if evaluation is not None:
        final["ocid_vlg_2d_consistency"] = evaluation["per_candidate"][0]
        final["evaluation_label"] = evaluation["label"]
        final["is_physical_grasp_success"] = False

    save_strict_json(
        sample_dir / "geometrically_ranked_candidates.json",
        {
            "metadata": {
                "schema_version": 1,
                "method": config["method"],
                "source_candidates_npz": str(source_npz),
                "source_candidates_npz_sha256": source_hashes["candidates_npz_sha256"],
                "source_candidates_json_sha256": source_hashes["candidates_json_sha256"],
                "candidate_count": len(ranked),
                "gqcnn_scoring_used": False,
            },
            "candidates": ranked,
        },
    )
    save_ranked_csv(sample_dir / "geometrically_ranked_candidates.csv", ranked)
    save_ranked_npz(sample_dir / "geometrically_ranked_candidates.npz", ranked)
    save_strict_json(sample_dir / "final_grasp.json", final)
    save_deterministic_npz(sample_dir / "final_grasp.npz", _ranking_arrays([top1]))

    rgb = np.asarray(Image.open(sample_dir / "rgb.png").convert("RGB"), dtype=np.uint8)
    save_candidate_overlay(
        rgb,
        [top1],
        sample_dir / "final_grasp_overlay.png",
        mask=mask,
        title=f"Geometric Top-1: {top1['candidate_id']}  score={top1['geometric_score']:.4f}",
        score_field="geometric_score",
        score_label="geom",
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after_hash = sha256_file(source_npz)
    source_unchanged = after_hash == source_hashes["candidates_npz_sha256"]
    if not source_unchanged:
        raise RuntimeError("frozen candidates.npz changed while ranking")
    breakdown.update(
        {
            "sample_id": metadata["sample_id"],
            "query": metadata["query"],
            "source_files": {
                **source_hashes,
                "candidates_npz_sha256_after_ranking": after_hash,
                "source_npz_unchanged": source_unchanged,
                "depth_m_npy_sha256": sha256_file(sample_dir / "depth_m.npy"),
                "processed_mask_sha256": sha256_file(sample_dir / "hifics_mask_processed.png"),
                "camera_intrinsics_sha256": sha256_file(sample_dir / "camera.intr"),
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
            },
            "timing_ms": {"geometric_ranking_and_outputs": elapsed_ms},
            "selected_top1": {
                "candidate_id": top1["candidate_id"],
                "geometric_score": top1["geometric_score"],
                "component_scores": top1["component_scores"],
            },
            "optional_evaluation": evaluation,
        }
    )
    save_strict_json(sample_dir / "ranking_feature_breakdown.json", breakdown)

    row = _empty_row(str(metadata["sample_id"]))
    row.update(
        {
            "query": metadata["query"],
            "candidate_count": len(ranked),
            "top1_candidate_id": top1["candidate_id"],
            "top1_geometric_score": top1["geometric_score"],
            "center_u_px": top1["center_u_px"],
            "center_v_px": top1["center_v_px"],
            "angle_deg": top1["angle_deg"],
            "contact_span_px": top1["contact_span_px_raw"],
            "top1_2d_rectangle_accuracy": "" if evaluation is None else int(evaluation["top1_rectangle_accuracy"]),
            "top5_2d_recall": "" if evaluation is None else int(evaluation["topk_recall"]),
            "first_2d_matching_rank": "" if evaluation is None or evaluation["first_matching_rank"] is None else evaluation["first_matching_rank"],
            "ranking_time_ms": elapsed_ms,
            "source_candidates_npz_sha256": source_hashes["candidates_npz_sha256"],
            "source_npz_unchanged": int(source_unchanged),
            "failure_reason": "",
        }
    )
    return row


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    if args.top_k is not None:
        config["ranking"]["top_k"] = int(args.top_k)
    annotations = None
    if not args.no_evaluation:
        annotations = args.annotations.expanduser().resolve()
        if not annotations.is_file():
            raise FileNotFoundError(f"optional evaluation requested but annotations missing: {annotations}")

    total_failures = 0
    all_results = []
    for root_arg in args.output_roots:
        root = root_arg.expanduser().resolve()
        rows = []
        for sample_dir in _sample_dirs(root):
            try:
                row = rank_sample(
                    sample_dir,
                    config=config,
                    config_path=config_path,
                    annotations=annotations,
                )
                print(json.dumps(row, ensure_ascii=False, sort_keys=True))
            except Exception as error:
                row = _empty_row(sample_dir.name)
                row["failure_reason"] = f"{type(error).__name__}: {error}"
                total_failures += 1
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            rows.append(row)
            all_results.append(row)
        _write_summary(root, rows)
    print(
        json.dumps(
            {
                "status": "DONE" if total_failures == 0 else "PARTIAL",
                "sample_rows": len(all_results),
                "failures": total_failures,
                "output_roots": [str(path.expanduser().resolve()) for path in args.output_roots],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
