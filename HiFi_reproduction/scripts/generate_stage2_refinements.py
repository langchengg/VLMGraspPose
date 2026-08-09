#!/usr/bin/env python3
"""Generate OOF-selected GT-free target-specific Stage-2 SAM 3 refinements."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.p90_selector import (  # noqa: E402
    predict_candidates,
    select_scored_candidates,
)
from src.segmentation.proposal_types import (  # noqa: E402
    ProposalCandidate,
    load_candidate_masks_npz,
    reference_candidate_ids_from_provenance,
)
from src.segmentation.query_semantics import parse_query  # noqa: E402
from src.segmentation.sam3_embedding_cache import Sam3EmbeddingCache  # noqa: E402
from src.segmentation.sam3_proposal_generator import write_proposal_bundle  # noqa: E402
from src.segmentation.sam3_text_proposals import OfficialSam3TextProposalGenerator  # noqa: E402
from src.segmentation.sam3_visual_proposals import OfficialSam3VisualProposalGenerator  # noqa: E402
from src.segmentation.second_stage_refiner import build_stage2_prompts, tight_box  # noqa: E402
from src.segmentation.selective_sam3_vg.io import (  # noqa: E402
    load_compact_manifest,
    load_probability,
    resize_probability,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--embedding-cache-max-gib", type=float)
    parser.add_argument("--embedding-cache-root", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/sam3_proposal_bank_p90_v1",
    )
    parser.add_argument(
        "--selector-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/sam3_proposal_bank_p90_v1/stage1_selector",
    )
    return parser.parse_args()


def _row_group_by_sample(parquet: pq.ParquetFile) -> dict[str, int]:
    field_index = parquet.schema_arrow.get_field_index("sample_id")
    if field_index < 0:
        raise ValueError("OOF prediction table has no sample_id column")
    result: dict[str, int] = {}
    for group in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(group).column(field_index).statistics
        sample_id = None
        if statistics is not None and statistics.has_min_max:
            minimum = statistics.min
            maximum = statistics.max
            if isinstance(minimum, bytes):
                minimum = minimum.decode("utf-8")
            if isinstance(maximum, bytes):
                maximum = maximum.decode("utf-8")
            if str(minimum) == str(maximum):
                sample_id = str(minimum)
        if sample_id is None:
            values = parquet.read_row_group(
                group, columns=["sample_id"]
            )["sample_id"].to_pylist()
            unique = {str(value) for value in values}
            if len(unique) != 1:
                raise ValueError(f"OOF row group {group} spans multiple samples")
            sample_id = unique.pop()
        if sample_id in result:
            raise ValueError(f"OOF sample spans multiple row groups: {sample_id}")
        result[sample_id] = group
    return result


def _rank_candidates(
    candidates: pd.DataFrame,
    *,
    selected_method: str,
    score_column: str,
) -> pd.DataFrame:
    columns = [score_column]
    if selected_method == "M3_two_head_ensemble":
        columns.append("m2_predicted_iou")
    columns.append("candidate_id")
    return candidates.sort_values(
        columns,
        ascending=[False] * (len(columns) - 1) + [True],
        kind="stable",
    )


def _load_depth_metres(path: Path, shape: tuple[int, int]) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.shape != tuple(shape):
        raise ValueError(f"depth shape {depth.shape} != selected-mask shape {shape}")
    if float(np.nanmax(depth, initial=0.0)) > 20.0:
        depth /= 1000.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _reference_candidates(
    index: pd.DataFrame,
    masks: dict[str, np.ndarray],
    provenance: dict[str, list[dict]],
    *,
    maximum: int = 3,
) -> list[tuple[str, pd.Series, np.ndarray]]:
    reference_ids = reference_candidate_ids_from_provenance(index, provenance)
    rows = index[index.index.astype(str).isin(reference_ids)]
    ranked: list[tuple[str, pd.Series, np.ndarray]] = []
    for candidate_id, candidate in rows.iterrows():
        logical = candidate.copy()
        if not str(logical["source_family"]).startswith("REFERENCE_"):
            logical["source_family"] = "REFERENCE_PROVENANCE_CONTEXT"
        ranked.append((str(candidate_id), logical, masks[str(candidate_id)]))
    ranked.sort(
        key=lambda item: (
            pd.isna(item[1].get("sam_score")),
            -(
                float(item[1].get("sam_score"))
                if not pd.isna(item[1].get("sam_score"))
                else 0.0
            ),
            item[0],
        )
    )
    return ranked[: int(maximum)]


def main() -> int:
    args = parse_args()
    experiment = args.experiment_root.expanduser().resolve()
    selector = args.selector_root.expanduser().resolve()
    selector_metrics = json.loads((selector / "validation_metrics.json").read_text())
    selected_method = str(selector_metrics["selected_model"])
    oof_file: pq.ParquetFile | None = None
    oof_group_by_sample: dict[str, int] = {}
    model_payload = None
    if args.split in {"train", "val"}:
        oof_file = pq.ParquetFile(selector / "oof_predictions.parquet")
        oof_group_by_sample = _row_group_by_sample(oof_file)
        if not selected_method.startswith("HIFI_FALLBACK"):
            decisions = pd.read_parquet(selector / "oof_selected_candidates.parquet")
            decisions = decisions[decisions["method"] == selected_method].copy()
            decisions = decisions[decisions["split"] == args.split].copy()
            decisions = decisions.set_index("sample_id", drop=False)
        else:
            decisions = None
        score_column = (
            "m0_p90_calibrated"
            if selected_method == "M0_logistic"
            else "m2_predicted_iou"
            if selected_method == "M2_hgb_regressor"
            else "m1_p90_calibrated"
        )
    else:
        with (selector / "model.pkl").open("rb") as stream:
            model_payload = pickle.load(stream)
        decisions = None
        score_column = "selection_score"
    compact = load_compact_manifest(
        PROJECT_ROOT
        / f"runs/modular_reranking_repeatedfilm_v1_20260729_203147/compact_inputs/{args.split}/manifest.jsonl",
        expected_split=args.split,
    )
    by_id = {row.sample_id: row for row in compact}
    if args.split in {"train", "val"}:
        missing_oof = [
            row.sample_id for row in compact if row.sample_id not in oof_group_by_sample
        ]
        if missing_oof:
            raise ValueError(
                f"OOF predictions omit {len(missing_oof)} {args.split} samples"
            )
        sample_ids = [row.sample_id for row in compact]
        if decisions is not None:
            if decisions.index.duplicated().any() or set(decisions.index.astype(str)) != set(
                sample_ids
            ):
                raise ValueError(f"selected OOF decisions are incomplete for {args.split}")
    else:
        sample_ids = [row.sample_id for row in compact]
    if args.sample_limit is not None:
        sample_ids = sample_ids[: int(args.sample_limit)]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.num_shards > 1:
        frame_indices: dict[str, int] = {}
        for sample_id in sample_ids:
            scene_id = by_id[sample_id].scene_id
            frame_indices.setdefault(scene_id, len(frame_indices))
        sample_ids = [
            sample_id
            for sample_id in sample_ids
            if frame_indices[by_id[sample_id].scene_id] % args.num_shards
            == args.shard_index
        ]
        if not sample_ids:
            raise ValueError("selected Stage-2 shard is empty")
    proposal_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs/sam3_proposal_bank_p90_v1/proposal_generation.yaml"
        ).read_text()
    )
    stage2_config = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs/sam3_proposal_bank_p90_v1/second_refinement.yaml"
        ).read_text()
    )
    model_path = (PROJECT_ROOT / proposal_config["model"]["local_path"]).resolve()
    cache_root = (
        args.embedding_cache_root.expanduser().resolve()
        if args.embedding_cache_root is not None
        else experiment / "cache/image_embeddings"
    )
    cache = Sam3EmbeddingCache(cache_root)
    visual_generator = OfficialSam3VisualProposalGenerator(
        model_path,
        revision=proposal_config["model"]["revision"],
        cache=cache,
        processor_sha256=proposal_config["model"]["tracker_processor_sha256"],
        num_threads=int(proposal_config["num_threads"]),
    )
    text_generator = OfficialSam3TextProposalGenerator(
        model_path,
        revision=proposal_config["model"]["revision"],
        cache=cache,
        processor_sha256=proposal_config["model"]["pcs_processor_sha256"],
        num_threads=int(proposal_config["num_threads"]),
        micro_batch_size=int(proposal_config["pcs_micro_batch_size"]),
    )
    for number, sample_id in enumerate(sample_ids, start=1):
        output = experiment / "stage2" / sample_id
        if args.resume and (output / "terminal_status.json").is_file():
            terminal = json.loads((output / "terminal_status.json").read_text())
            verified = terminal.get("status") == "COMPLETE"
            if args.verify_existing:
                verified = verified and all(
                    (output / name).is_file() and sha256_file(output / name) == digest
                    for name, digest in terminal.get("stage2_contract_checksums", {}).items()
                )
            if verified:
                continue
        started = time.perf_counter()
        row = by_id[sample_id]
        image = Image.open(row.rgb_path).convert("RGB")
        proposal = experiment / "proposals" / sample_id
        masks = load_candidate_masks_npz(proposal / "candidate_masks.npz")
        index = pd.read_parquet(proposal / "candidate_index.parquet").set_index("candidate_id")
        provenance = json.loads(
            (proposal / "candidate_provenance.json").read_text(encoding="utf-8")
        )
        if args.split in {"train", "val"}:
            if oof_file is None:
                raise RuntimeError("development OOF table was not initialized")
            candidates = oof_file.read_row_group(
                oof_group_by_sample[sample_id]
            ).to_pandas()
            if set(candidates["split"].astype(str)) != {args.split}:
                raise ValueError(f"OOF split drift for {sample_id}")
            if decisions is None:
                fallback = candidates[
                    candidates["source_family"] == "HIFI_ORIGINAL"
                ]
                if len(fallback) != 1:
                    raise ValueError(f"OOF sample requires one HiFi fallback: {sample_id}")
                selected_row = fallback.iloc[0]
            else:
                selected_row = decisions.loc[sample_id]
        else:
            if model_payload is None:
                raise RuntimeError("frozen Stage-1 model was not loaded")
            feature_path = experiment / "features/test_samples" / f"{sample_id}.parquet"
            scored = predict_candidates(pd.read_parquet(feature_path), model_payload)
            candidates = scored.rename(
                columns={
                    "predicted_p90_calibrated": "m1_p90_calibrated",
                    "predicted_iou": "m2_predicted_iou",
                }
            )
            selected_row = select_scored_candidates(scored, selected_method).iloc[0]
        selected_id = str(selected_row["candidate_id"])
        selected_mask = masks[selected_id]
        hifi_row = index[index["source_family"] == "HIFI_ORIGINAL"].iloc[0]
        hifi_id = str(hifi_row.name)
        hifi_mask = masks[hifi_id]
        candidates = _rank_candidates(
            candidates,
            selected_method=selected_method,
            score_column=score_column,
        )
        competitor_ids = [
            str(value)
            for value in candidates["candidate_id"]
            if str(value) != selected_id
        ][:3]
        competitor_masks = [masks[value] for value in competitor_ids]
        competitor_boxes = [tight_box(value) for value in competitor_masks if np.any(value)]
        semantics = parse_query(row.query)
        reference_candidates = [
            value
            for value in _reference_candidates(
                index, masks, provenance, maximum=4
            )
            if value[0] != selected_id
        ][:3]
        use_reference_evidence = semantics.pairwise_relation is not None
        reference_masks = (
            [mask for _, _, mask in reference_candidates if np.any(mask)]
            if use_reference_evidence
            else []
        )
        reference_boxes = [tight_box(mask) for mask in reference_masks]
        depth_m = _load_depth_metres(row.depth_path, selected_mask.shape)
        hifi_probability = resize_probability(
            load_probability(row.probability_path), selected_mask.shape
        )
        target_text = semantics.target_category_prompt or semantics.query
        target_attribute_text = semantics.target_attribute_prompt
        if not np.any(selected_mask):
            visual_candidates = []
            text_candidates = []
            prompt_metadata = {"status": "EMPTY_STAGE1_SELECTION", "uses_ground_truth": False}
            visual_runtime = {"candidate_count": 0}
            text_runtime = {"candidate_count": 0}
        else:
            visual_prompts, text_prompts, prompt_metadata = build_stage2_prompts(
                selected_mask,
                competitor_masks,
                competitor_boxes,
                target_text=target_text,
                target_attribute_text=target_attribute_text,
                reference_masks=reference_masks,
                reference_boxes=reference_boxes,
                depth_m=depth_m,
                hifi_probability=hifi_probability,
            )
            visual_candidates, visual_runtime = visual_generator.generate(
                sample_id,
                image,
                str(row.raw["source_rgb_sha256"]),
                visual_prompts,
                mask_thresholds=tuple(float(x) for x in stage2_config["mask_thresholds"]),
            )
            text_candidates, text_runtime = text_generator.generate(
                sample_id,
                image,
                str(row.raw["source_rgb_sha256"]),
                text_prompts,
                instance_thresholds=(0.10, 0.30, 0.50),
                mask_thresholds=tuple(float(x) for x in stage2_config["mask_thresholds"]),
                maximum_instances_per_prompt=20,
            )
        cache_limit_gib = (
            args.embedding_cache_max_gib
            if args.embedding_cache_max_gib is not None
            else stage2_config.get("embedding_cache_max_gib")
        )
        cache_pruning = (
            cache.prune_to_max_bytes(
                int(float(cache_limit_gib) * 1024**3),
                protected_rgb_sha256=str(row.raw["source_rgb_sha256"]),
            )
            if cache_limit_gib is not None
            else None
        )
        reference_context = []
        for reference_id, reference_row, reference_mask in reference_candidates:
            reference_context.append(
                ProposalCandidate(
                    sample_id,
                    str(reference_row["source_family"]),
                    f"stage1_reference_context|candidate_id={reference_id}",
                    reference_mask,
                    eligible_final=False,
                    sam_score=(
                        None
                        if pd.isna(reference_row.get("sam_score"))
                        else float(reference_row["sam_score"])
                    ),
                    rgb_checksum=str(row.raw["source_rgb_sha256"]),
                )
            )
        mandatory = [
            ProposalCandidate(
                sample_id,
                "STAGE2_HIFI_FALLBACK",
                f"original_hifi|stage1_id={hifi_id}",
                hifi_mask,
                rgb_checksum=str(row.raw["source_rgb_sha256"]),
            ),
            ProposalCandidate(
                sample_id,
                "STAGE2_SELECTED_STAGE1",
                f"selected_stage1|candidate_id={selected_id}",
                selected_mask,
                rgb_checksum=str(row.raw["source_rgb_sha256"]),
            ),
        ]
        runtime = {
            "visual": visual_runtime,
            "text": text_runtime,
            "embedding_cache_pruning": cache_pruning,
            "runtime_seconds_before_write": time.perf_counter() - started,
            "uses_ground_truth": False,
        }
        write_proposal_bundle(
            output,
            image,
            mandatory + reference_context + visual_candidates + text_candidates,
            hifi_mask,
            prompt_metadata,
            runtime,
            deduplication_iou=0.995,
        )
        aliases = {
            "candidate_masks.npz": "refinement_candidates.npz",
            "candidate_index.parquet": "refinement_candidate_index.parquet",
            "proposal_grid.png": "refinement_grid.png",
        }
        for source_name, target_name in aliases.items():
            shutil.copy2(output / source_name, output / target_name)
        (output / "selected_stage1_candidate.json").write_text(
            json.dumps(
                {
                    "candidate_id": selected_id,
                    "source_family": str(selected_row["source_family"]),
                    "oof_method": selected_method,
                    "uses_ground_truth": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "positive_prompts.json").write_text(
            json.dumps(
                {"positive_points": prompt_metadata.get("positive_points"), "selected_box": prompt_metadata.get("selected_box_xyxy")},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "negative_prompts.json").write_text(
            json.dumps(
                {
                    "negative_points": prompt_metadata.get("negative_points"),
                    "competitor_negative_points": prompt_metadata.get(
                        "competitor_negative_points"
                    ),
                    "reference_negative_points": prompt_metadata.get(
                        "reference_negative_points"
                    ),
                    "depth_discontinuity_negative_points": prompt_metadata.get(
                        "depth_discontinuity_negative_points"
                    ),
                    "low_hifi_probability_negative_points": prompt_metadata.get(
                        "low_hifi_probability_negative_points"
                    ),
                    "competitor_boxes": prompt_metadata.get("competitor_boxes"),
                    "reference_boxes": prompt_metadata.get("reference_boxes"),
                    "combined_negative_boxes": prompt_metadata.get(
                        "combined_negative_boxes"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        terminal_path = output / "terminal_status.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        contract_names = (
            "selected_stage1_candidate.json",
            "positive_prompts.json",
            "negative_prompts.json",
            "refinement_candidates.npz",
            "refinement_candidate_index.parquet",
            "refinement_grid.png",
            "runtime.json",
        )
        terminal["stage2_contract_checksums"] = {
            name: sha256_file(output / name) for name in contract_names
        }
        temporary_terminal = terminal_path.with_name(f".{terminal_path.name}.tmp")
        temporary_terminal.write_text(
            json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_terminal.replace(terminal_path)
        if number % 10 == 0:
            print(f"Stage 2: {number}/{len(sample_ids)}", flush=True)
    print(json.dumps({"status": "COMPLETE", "samples": len(sample_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
