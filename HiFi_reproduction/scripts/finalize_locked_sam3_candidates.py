#!/usr/bin/env python3
"""Apply frozen selectors/gate and materialize GT-free benchmark mask bundles."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.conservative_mask_gate import (  # noqa: E402
    GateThresholds,
    apply_gate,
    proposed_alternatives,
)
from src.segmentation.leakage_guard import RuntimeFileAccessGuard  # noqa: E402
from src.segmentation.p90_selector import (  # noqa: E402
    predict_candidates,
    select_scored_candidates,
)
from src.segmentation.proposal_evaluation import deterministic_rule_score  # noqa: E402
from src.segmentation.proposal_types import load_candidate_masks_npz  # noqa: E402
from src.segmentation.query_semantics import parse_query  # noqa: E402
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1",
    )
    return parser.parse_args()


def _load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def _score_final(features: pd.DataFrame, payload: dict) -> pd.DataFrame:
    data = features[features["eligible_final"]].copy().reset_index(drop=True)
    x = payload["encoder"].transform(data)
    raw0 = payload["m0_classifier"].predict_proba(x)[:, 1]
    raw1 = payload["classifier"].predict_proba(x)[:, 1]
    data["m0_p90_calibrated"] = payload["m0_calibrator"].predict(raw0)
    data["m1_p90_calibrated"] = payload["calibrator"].predict(raw1)
    data["m2_predicted_iou"] = np.clip(payload["regressor"].predict(x), 0.0, 1.0)
    data["deterministic_rule_score"] = deterministic_rule_score(data)
    return data


def _visualization(
    rgb_path: Path,
    hifi: np.ndarray,
    final: np.ndarray,
    query: str,
    selected_source: str,
) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8).copy()
    array[np.asarray(hifi, dtype=bool)] = (
        0.70 * array[np.asarray(hifi, dtype=bool)] + 0.30 * np.asarray([255, 180, 0])
    ).astype(np.uint8)
    array[np.asarray(final, dtype=bool)] = (
        0.55 * array[np.asarray(final, dtype=bool)] + 0.45 * np.asarray([0, 220, 160])
    ).astype(np.uint8)
    canvas = Image.new("RGB", (rgb.width, rgb.height + 52), "white")
    canvas.paste(Image.fromarray(array), (0, 52))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), query[:100], fill="black")
    draw.text((8, 28), f"final source: {selected_source}", fill="black")
    return canvas


def _complete(path: Path) -> bool:
    status_path = path / "terminal_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE_GT_FREE_INFERENCE":
        return False
    return all(
        (path / name).is_file() and sha256_file(path / name) == digest
        for name, digest in status.get("file_sha256", {}).items()
    )


def main() -> int:
    args = parse_args()
    experiment = args.experiment_root.expanduser().resolve()
    artifact = args.artifact_root.expanduser().resolve()
    formal_lock = artifact / "formal_lock.json"
    if not formal_lock.is_file():
        raise FileNotFoundError("formal method lock is required before benchmark finalization")
    compact = load_compact_manifest(
        PROJECT_ROOT
        / f"runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/{args.split}/manifest.jsonl",
        expected_split=args.split,
        expected_count=7675,
    )
    stage1_model = _load_pickle(artifact / "stage1_selector/model.pkl")
    final_model = _load_pickle(artifact / "final_selector/model.pkl")
    gate_payload = json.loads((artifact / "final_selector/gate.json").read_text())
    method = str(gate_payload["selected_method"])
    inference_method = (
        "F1_hgb_classifier" if method.startswith("HIFI_FALLBACK") else method
    )
    thresholds = GateThresholds(**gate_payload["thresholds"])
    sample_ids = [row.sample_id for row in compact]
    if args.sample_limit is not None:
        sample_ids = sample_ids[: int(args.sample_limit)]
    by_id = {row.sample_id: row for row in compact}
    output_root = experiment / "locked_benchmark_masks"
    output_root.mkdir(parents=True, exist_ok=True)
    access_log = experiment / "leakage_audit/runtime_file_access_benchmark.jsonl"
    for number, sample_id in enumerate(sample_ids, start=1):
        output = output_root / sample_id
        if args.resume and output.is_dir() and (
            _complete(output) if args.verify_existing else (output / "terminal_status.json").is_file()
        ):
            continue
        started = time.perf_counter()
        row = by_id[sample_id]
        with RuntimeFileAccessGuard(access_log):
            stage1_sample = pd.read_parquet(
                experiment / "features/test_samples" / f"{sample_id}.parquet"
            )
            stage1_scores = predict_candidates(stage1_sample, stage1_model)
            stage1_selected = select_scored_candidates(
                stage1_scores, str(stage1_model["selected_name"])
            ).iloc[0]
            expected_stage1 = json.loads(
                (experiment / "stage2" / sample_id / "selected_stage1_candidate.json").read_text()
            )
            if str(stage1_selected["candidate_id"]) != str(expected_stage1["candidate_id"]):
                raise RuntimeError(f"Stage-1 decision drift for {sample_id}")
            stage2_sample = pd.read_parquet(
                experiment / "features_stage2/test_samples" / f"{sample_id}.parquet"
            )
            final_scores = _score_final(stage2_sample, final_model)
            proposed = proposed_alternatives(final_scores, inference_method)
            final_decision = apply_gate(final_scores, proposed, thresholds).iloc[0]
            final_id = str(final_decision["candidate_id"])
            stage2_root = experiment / "stage2" / sample_id
            stage2_masks = load_candidate_masks_npz(stage2_root / "candidate_masks.npz")
            final_mask = stage2_masks[final_id]
            hifi_rows = final_scores[
                final_scores["source_family"] == "STAGE2_HIFI_FALLBACK"
            ]
            if len(hifi_rows) != 1:
                raise RuntimeError(f"missing unique Stage-2 HiFi fallback for {sample_id}")
            hifi_id = str(hifi_rows.iloc[0]["candidate_id"])
            hifi_mask = stage2_masks[hifi_id]
            temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            Image.fromarray(hifi_mask.astype(np.uint8) * 255, mode="L").save(
                temporary / "original_hifi_mask.png"
            )
            shutil.copy2(row.probability_path, temporary / "original_hifi_probability.npy")
            semantics = parse_query(row.query)
            (temporary / "query_parse.json").write_text(
                json.dumps(semantics.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shutil.copy2(
                experiment / "proposals" / sample_id / "candidate_index.parquet",
                temporary / "proposal_index.parquet",
            )
            stage1_scores.to_parquet(temporary / "stage1_scores.parquet", index=False)
            (temporary / "stage1_decision.json").write_text(
                json.dumps(
                    {
                        "candidate_id": str(stage1_selected["candidate_id"]),
                        "source_family": str(stage1_selected["source_family"]),
                        "selection_score": float(stage1_selected["selection_score"]),
                        "predicted_iou": float(stage1_selected["predicted_iou"]),
                        "uses_ground_truth": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.copy2(stage2_root / "candidate_masks.npz", temporary / "stage2_candidates.npz")
            final_scores.to_parquet(temporary / "stage2_scores.parquet", index=False)
            source_family = str(final_decision["source_family"])
            probability_available = source_family == "STAGE2_HIFI_FALLBACK"
            decision_payload = {
                "candidate_id": final_id,
                "source_family": source_family,
                "gate_accept": bool(final_decision["gate_accept"]),
                "gate_reason": str(final_decision["gate_reason"]),
                "final_probability_available": probability_available,
                "probability_policy": (
                    "original frozen HiFi probability"
                    if probability_available
                    else "unavailable; no binary-to-probability fabrication"
                ),
                "uses_ground_truth": False,
            }
            (temporary / "gate_decision.json").write_text(
                json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            Image.fromarray(final_mask.astype(np.uint8) * 255, mode="L").save(
                temporary / "final_mask.png"
            )
            if probability_available:
                shutil.copy2(row.probability_path, temporary / "final_probability.npy")
            provenance = {
                **decision_payload,
                "sample_id": sample_id,
                "query": row.query,
                "rgb_sha256": str(row.raw["source_rgb_sha256"]),
                "formal_lock_sha256": json.loads(formal_lock.read_text())["formal_lock_sha256"],
                "posthoc_benchmark": True,
            }
            (temporary / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            timing = {
                "final_selection_seconds": time.perf_counter() - started,
                "proposal_runtime_path": str(experiment / "proposals" / sample_id / "runtime.json"),
                "stage2_runtime_path": str(stage2_root / "runtime.json"),
            }
            (temporary / "timing.json").write_text(
                json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            _visualization(
                row.rgb_path, hifi_mask, final_mask, row.query, source_family
            ).save(temporary / "visualization.png")
            hashes = {
                path.name: sha256_file(path)
                for path in sorted(temporary.iterdir())
                if path.is_file()
            }
            (temporary / "terminal_status.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE_GT_FREE_INFERENCE",
                        "sample_id": sample_id,
                        "terminal_outcome": (
                            "selected_hifi"
                            if source_family == "STAGE2_HIFI_FALLBACK"
                            else "selected_stage1_sam3"
                            if source_family == "STAGE2_SELECTED_STAGE1"
                            else "selected_stage2_sam3"
                        ),
                        "file_sha256": hashes,
                        "uses_ground_truth": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if output.exists():
                raise FileExistsError(f"refusing to overwrite locked sample output {output}")
            temporary.replace(output)
        if number % 25 == 0:
            print(f"locked finalization: {number}/{len(sample_ids)}", flush=True)
    print(json.dumps({"status": "COMPLETE", "samples": len(sample_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
