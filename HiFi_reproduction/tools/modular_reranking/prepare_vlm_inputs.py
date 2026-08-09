#!/usr/bin/env python3
"""Build GT-free Top-K visual inputs for local VLM reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.vlm_visualization import (  # noqa: E402
    build_vlm_visualization_recipe,
    build_vlm_visualizations,
    canonical_recipe_sha256,
)
from src.grasping.reranking_v1.identity import sha256_file  # noqa: E402
from src.grasping.reranking_v1.artifact_contract import (  # noqa: E402
    identity_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--scored-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=("validation", "formal_test"),
        required=True,
        help=(
            "Explicitly authorize validation preparation or GT-free formal-test "
            "input preparation; this command never performs VLM inference"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--features-parquet", type=Path)
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="JSONL containing only sample_id; filters and orders inputs exactly",
    )
    parser.add_argument(
        "--reuse-visual-root",
        type=Path,
        help="Reuse already audited visual files instead of rendering duplicates",
    )
    parser.add_argument(
        "--visual-storage",
        choices=("on_demand_recipe", "persistent_visuals"),
        default="on_demand_recipe",
        help=(
            "Full validation/formal inputs default to compact recipes. "
            "Persistent PNGs are allowed only for an explicitly selected subset."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.include_metadata and args.features_parquet is None:
        raise ValueError("--include-metadata requires --features-parquet")
    if (
        args.reuse_visual_root is not None
        and args.visual_storage != "persistent_visuals"
    ):
        raise ValueError("--reuse-visual-root requires persistent_visuals")
    if (
        args.visual_storage == "persistent_visuals"
        and args.input_mode == "formal_test"
    ):
        raise ValueError("formal-test VLM inputs must use on-demand recipes")
    if (
        args.visual_storage == "persistent_visuals"
        and args.selection_manifest is None
        and args.limit is None
    ):
        raise ValueError(
            "persistent VLM visuals require an explicit selected subset"
        )
    manifest = args.prediction_manifest.expanduser().resolve()
    candidate_root = args.candidate_root.expanduser().resolve()
    scored_root = args.scored_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rows = read_jsonl(manifest)
    observed_splits = {str(row.get("split", "")) for row in rows}
    allowed_splits = (
        {"val", "validation"}
        if args.input_mode == "validation"
        else {"test"}
    )
    if not rows or observed_splits - allowed_splits:
        raise ValueError(
            f"{args.input_mode} VLM preparation received invalid split values: "
            f"{sorted(observed_splits)}"
        )
    selection_path = None
    if args.selection_manifest is not None:
        selection_path = args.selection_manifest.expanduser().resolve()
        selected_rows = read_jsonl(selection_path)
        if any(set(item) != {"sample_id"} for item in selected_rows):
            raise ValueError(
                "selection manifest must contain only a sample_id field per row"
            )
        selected_ids = [str(item["sample_id"]) for item in selected_rows]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selection manifest contains duplicate sample IDs")
        by_id = {str(item["sample_id"]): item for item in rows}
        missing_ids = [sample_id for sample_id in selected_ids if sample_id not in by_id]
        if missing_ids:
            raise ValueError(f"selection IDs absent from prediction manifest: {missing_ids[:5]}")
        rows = [by_id[sample_id] for sample_id in selected_ids]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no prediction rows selected")
    output_rows: list[dict[str, Any]] = []
    visual_entries: list[dict[str, Any]] = []
    empty_sample_ids: list[str] = []
    aggregate_visual_manifest_path = (
        output_root / "aggregate_visual_manifest.json"
    ).resolve()
    feature_index: dict[tuple[str, str], dict[str, float]] = {}
    if args.features_parquet is not None:
        import numpy as np
        import pandas as pd

        feature_frame = pd.read_parquet(args.features_parquet.expanduser().resolve())
        required = {
            "sample_id",
            "candidate_id",
            "q_percentile_within_sample",
            "p_axis_mean",
            "normalized_width_mismatch",
            "jaw_depth_difference",
            "approach_clearance",
            "pose_cluster_size",
        }
        missing = sorted(required - set(feature_frame.columns))
        if missing:
            raise ValueError(f"VLM metadata features missing: {missing}")
        for item in feature_frame.loc[:, sorted(required)].to_dict(orient="records"):
            key = (str(item["sample_id"]), str(item["candidate_id"]))
            if key in feature_index:
                raise ValueError(f"duplicate feature identity: {key}")
            values = {
                "q_percentile": float(item["q_percentile_within_sample"]),
                "soft_mask_support": float(item["p_axis_mean"]),
                "width_compatibility": float(
                    np.clip(1.0 - item["normalized_width_mismatch"], 0.0, 1.0)
                ),
                "jaw_depth_difference": float(item["jaw_depth_difference"]),
                "clearance_proxy": float(item["approach_clearance"]),
                "candidate_cluster_size": float(item["pose_cluster_size"]),
            }
            if not all(np.isfinite(list(values.values()))):
                raise ValueError(f"non-finite VLM metadata: {key}")
            feature_index[key] = values
    nonempty = 0
    total_candidates = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        source = candidate_root / sample_id
        scored = scored_root / sample_id
        metadata = json.loads(
            (source / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata["sample_id"] != sample_id:
            raise ValueError(f"candidate metadata mismatch: {sample_id}")
        if metadata["query"] != row["query"]:
            raise ValueError(f"query mismatch: {sample_id}")
        scored_path = scored / "gqcnn_scored_candidates.json"
        if not scored_path.is_file():
            marker_path = scored / "_SCORING_COMPLETE.json"
            marker = json.loads(
                marker_path.read_text(encoding="utf-8")
            )
            valid_empty = (
                marker.get("sample_id") == sample_id
                and marker.get("scoring_status") == "skipped_valid_empty"
                and int(marker.get("source_candidate_count", -1)) == 0
                and int(marker.get("gqcnn_scored_count", -1)) == 0
            )
            metadata_path = scored / "scoring_metadata.json"
            metadata_marker = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            if not valid_empty or any(
                metadata_marker.get(key) != marker.get(key)
                for key in (
                    "sample_id",
                    "scoring_status",
                    "source_candidate_count",
                    "gqcnn_scored_count",
                )
            ):
                raise FileNotFoundError(f"scored candidates missing: {sample_id}")
            candidates: list[dict[str, Any]] = []
        else:
            payload = json.loads(scored_path.read_text(encoding="utf-8"))
            candidates = list(payload["candidates"][: args.top_k])
            ranks = [int(candidate["gqcnn_rank"]) for candidate in candidates]
            if ranks != list(range(1, len(candidates) + 1)):
                raise ValueError(f"non-canonical GQ-CNN rank order: {sample_id}")
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate Top-K candidate IDs: {sample_id}")
        output: dict[str, Any] = {
            "sample_id": sample_id,
            "instruction": str(row["query"]),
            "candidate_ids": candidate_ids,
            "original_top1_candidate_id": candidate_ids[0] if candidate_ids else None,
            "include_metadata": bool(args.include_metadata),
            "candidate_metadata": (
                {
                    candidate_id: feature_index[(sample_id, candidate_id)]
                    for candidate_id in candidate_ids
                }
                if args.include_metadata
                else {}
            ),
            "split": str(row["split"]),
            "gt_fields_included": False,
            "visualization_storage_mode": args.visual_storage,
        }
        if candidates:
            if args.visual_storage == "on_demand_recipe":
                source_bindings = (
                    ("source_rgb_path", "source_rgb_sha256"),
                    ("source_depth_path", "source_depth_sha256"),
                    ("native_mask_path", "native_mask_sha256"),
                )
                for path_key, hash_key in source_bindings:
                    source_path = Path(str(row.get(path_key, ""))).resolve()
                    if (
                        not source_path.is_file()
                        or sha256_file(source_path) != row.get(hash_key)
                    ):
                        raise ValueError(
                            f"prediction source provenance changed: "
                            f"{sample_id}:{path_key}"
                        )
                input_config = metadata.get("config", {}).get("input", {})
                recipe = build_vlm_visualization_recipe(
                    sample_id=sample_id,
                    rgb_path=Path(row["source_rgb_path"]),
                    depth_mm_path=Path(row["source_depth_path"]),
                    predicted_mask_path=Path(row["native_mask_path"]),
                    candidates_q_order=candidates,
                    mask_processing=input_config,
                )
                candidate_records_sha256 = str(
                    recipe["candidate_records_sha256"]
                )
                recipe_sha256 = canonical_recipe_sha256(recipe)
                visual_entries.append(
                    {
                        "sample_id": sample_id,
                        "candidate_ids": candidate_ids,
                        "candidate_records_sha256": (
                            candidate_records_sha256
                        ),
                        "visualization_recipe_sha256": recipe_sha256,
                        "storage_mode": "on_demand_recipe",
                    }
                )
                output.update(
                    {
                        "visualization_storage_mode": (
                            "on_demand_recipe"
                        ),
                        "visualization_recipe": recipe,
                        "visualization_recipe_sha256": recipe_sha256,
                    }
                )
            elif args.reuse_visual_root is not None:
                candidate_records_sha256 = canonical_sha256(candidates)
                visual_dir = args.reuse_visual_root.expanduser().resolve() / sample_id
                visual_manifest = json.loads(
                    (visual_dir / "vlm_visualization_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    visual_manifest.get("sample_id") != sample_id
                    or visual_manifest.get("candidate_ids") != candidate_ids
                    or visual_manifest.get("candidate_records_sha256")
                    != candidate_records_sha256
                ):
                    raise ValueError(
                        f"reused visual provenance mismatch: {sample_id}"
                    )
                full_scene_path = visual_dir / "full_scene_overlay.png"
                contact_sheet_path = visual_dir / "candidate_contact_sheet.png"
                manifest_path = visual_dir / "vlm_visualization_manifest.json"
            else:
                candidate_records_sha256 = canonical_sha256(candidates)
                visual = build_vlm_visualizations(
                    sample_id=sample_id,
                    rgb=Path(row["source_rgb_path"]),
                    depth_m=source / "depth_m.npy",
                    predicted_mask=source / "hifics_mask_processed.png",
                    candidates_q_order=candidates,
                    output_dir=output_root / "visuals" / sample_id,
                )
                full_scene_path = visual.full_scene_overlay_path
                contact_sheet_path = visual.candidate_contact_sheet_path
                manifest_path = visual.manifest_path
                visual_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            if args.visual_storage == "persistent_visuals":
                expected_images = {
                    str(Path(item["path"]).resolve()): str(item["sha256"])
                    for item in visual_manifest.get("output_images", [])
                }
                actual_images = {
                    str(full_scene_path.resolve()): sha256_file(
                        full_scene_path
                    ),
                    str(contact_sheet_path.resolve()): sha256_file(
                        contact_sheet_path
                    ),
                }
                if expected_images != actual_images:
                    raise ValueError(
                        f"visual image provenance mismatch: {sample_id}"
                    )
                visual_entries.append(
                    {
                        "sample_id": sample_id,
                        "candidate_ids": candidate_ids,
                        "candidate_records_sha256": (
                            candidate_records_sha256
                        ),
                        "visualization_manifest_path": str(
                            manifest_path.resolve()
                        ),
                        "visualization_manifest_sha256": sha256_file(
                            manifest_path
                        ),
                        "output_images": [
                            {"path": path, "sha256": digest}
                            for path, digest in sorted(
                                actual_images.items()
                            )
                        ],
                        "storage_mode": "persistent_visuals",
                    }
                )
                output.update(
                    {
                        "full_scene_overlay_path": str(
                            full_scene_path.resolve()
                        ),
                        "candidate_contact_sheet_path": str(
                            contact_sheet_path.resolve()
                        ),
                        "visualization_manifest_path": str(
                            manifest_path.resolve()
                        ),
                        "visualization_storage_mode": (
                            "persistent_visuals"
                        ),
                    }
                )
            nonempty += 1
            total_candidates += len(candidates)
        else:
            empty_sample_ids.append(sample_id)
        output["aggregate_visual_manifest_path"] = str(
            aggregate_visual_manifest_path
        )
        output_rows.append(output)
    output_path = output_root / "vlm_inputs.jsonl"
    aggregate_visual_manifest = {
        "schema_version": 2,
        **identity_payload(),
        "manifest_kind": (
            "vlm_visualization_recipes"
            if args.visual_storage == "on_demand_recipe"
            else "vlm_persistent_visuals"
        ),
        "storage_mode": args.visual_storage,
        "gt_free": True,
        "input_mode": args.input_mode,
        "input_split": (
            "validation" if args.input_mode == "validation" else "test"
        ),
        "sample_count": len(output_rows),
        "nonempty_sample_count": len(visual_entries),
        "empty_sample_count": len(empty_sample_ids),
        "empty_sample_ids": empty_sample_ids,
        "visuals": visual_entries,
    }
    atomic_json(aggregate_visual_manifest_path, aggregate_visual_manifest)
    atomic_jsonl(output_path, output_rows)
    summary = {
        "schema_version": 1,
        **identity_payload(),
        "status": "COMPLETED",
        "samples": len(output_rows),
        "nonempty_samples": nonempty,
        "empty_samples": len(output_rows) - nonempty,
        "top_k": args.top_k,
        "visualized_candidates": total_candidates,
        "prediction_manifest": str(manifest),
        "candidate_root": str(candidate_root),
        "scored_root": str(scored_root),
        "gt_fields_included": False,
        "input_mode": args.input_mode,
        "input_split": (
            "validation" if args.input_mode == "validation" else "test"
        ),
        "formal_test_evidence_used": False,
        "include_metadata": bool(args.include_metadata),
        "features_parquet": (
            str(args.features_parquet.expanduser().resolve())
            if args.features_parquet is not None
            else None
        ),
        "selection_manifest": str(selection_path) if selection_path else None,
        "reuse_visual_root": (
            str(args.reuse_visual_root.expanduser().resolve())
            if args.reuse_visual_root is not None
            else None
        ),
        "visual_storage": args.visual_storage,
        "ordinary_pngs_persisted": (
            args.visual_storage == "persistent_visuals"
        ),
        "output_jsonl": str(output_path.resolve()),
        "output_jsonl_sha256": sha256_file(output_path),
        "aggregate_visual_manifest": str(aggregate_visual_manifest_path),
        "aggregate_visual_manifest_sha256": sha256_file(
            aggregate_visual_manifest_path
        ),
    }
    atomic_json(output_root / "summary.json", summary)
    (output_root / "run_command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
