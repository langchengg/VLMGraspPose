"""
scripts/step06_grasp_candidates.py — Generate target-conditioned grasp candidates
================================================================================
Step 6 consumes target grounding and target point-cloud extraction output, then
runs the Open3D-based RGB-D geometric sampler on the target point cloud.

Usage:
    python scripts/step06_grasp_candidates.py
    python scripts/step06_grasp_candidates.py --splits test_seen --top-k 50
    python scripts/step06_grasp_candidates.py --grounding predicted --task seg
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import load_label
from src.grasp_detector import RGBDGeometricGraspSampler
from src.target_point_cloud import build_point_cloud_representation


def _create_detector(detector_type: str, top_k: int):
    """Create a grasp detector.

    Detector types:
        geometric   — local RGB-D geometric sampler (default)
    """
    if detector_type == "geometric":
        print("[step06] Using RGBDGeometricGraspSampler (local RGB-D geometry)")
        return RGBDGeometricGraspSampler(top_k=top_k)

    raise ValueError(
        f"Unknown detector type: {detector_type}. "
        f"Choose from: geometric"
    )


def _load_target_map(split: str, grounding: str, task: str) -> dict:
    """Load TargetRegion records keyed by query sample_id."""
    target_map = {}
    if grounding == "oracle":
        path = config.ORACLE_TARGETS_DIR / f"{split}_oracle.jsonl"
        if not path.exists():
            return target_map
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                target_map[rec["sample_id"]] = {
                    "bbox": rec["gt_bbox"],
                    "mask_path": None,
                    "confidence": 1.0,
                    "label": rec.get("target_label"),
                    "mask_val": rec["gt_mask_val"],
                }
        return target_map

    path = config.GROUNDING_PRED_DIR / f"{split}_grounding_{task}.jsonl"
    if not path.exists():
        return target_map
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            target_map[rec["sample_id"]] = {
                "bbox": rec["pred_bbox"],
                "mask_path": rec.get("pred_mask_path"),
                "confidence": rec.get("grounding_score", 1.0),
                "label": rec.get("target_label") or rec.get("text_query"),
                "mask_val": None,
            }
    return target_map


def _load_target_mask(target: dict, scene_label: np.ndarray = None) -> np.ndarray | None:
    """Load predicted or oracle target mask."""
    if target.get("mask_path"):
        mask_path = Path(target["mask_path"])
        if not mask_path.is_absolute():
            mask_path = config.PROJECT_ROOT / mask_path
        if mask_path.exists():
            return np.load(str(mask_path)).astype(bool)

    if scene_label is not None and target.get("mask_val") is not None:
        return scene_label == int(target["mask_val"])

    return None


def _save_candidates(out_path: Path, candidates: list):
    positions = np.array([c.position for c in candidates], dtype=np.float32)
    rotations = np.array([c.rotation for c in candidates], dtype=np.float32)
    widths = np.array([c.width for c in candidates], dtype=np.float32)
    scores = np.array([c.detector_score for c in candidates], dtype=np.float32)
    sources = np.array([c.source for c in candidates])
    approaches = np.array([c.approach_vector for c in candidates], dtype=np.float32)
    closings = np.array([c.closing_direction for c in candidates], dtype=np.float32)
    grasp_types = np.array([c.grasp_type for c in candidates])

    np.savez_compressed(
        out_path,
        positions=positions,
        rotations=rotations,
        widths=widths,
        detector_scores=scores,
        sources=sources,
        approach_vectors=approaches,
        closing_directions=closings,
        grasp_types=grasp_types,
        num_candidates=len(candidates),
    )


def generate_candidates(
    splits: list = None,
    top_k: int = config.GRASP_TOP_K,
    detector_type: str = config.DEFAULT_DETECTOR,
    grounding: str = "predicted",
    task: str = config.DEFAULT_GROUNDING,
):
    """Run target-conditioned geometric grasp sampling for all queries."""
    if splits is None:
        splits = config.ALL_SPLITS

    detector = _create_detector(detector_type, top_k)

    # Use detector-specific subdirectory to avoid cache pollution
    # e.g. derived/grasp_candidates/geometric/
    candidates_dir = config.GRASP_CANDIDATES_DIR / detector_type
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata for downstream verification
    import json
    meta_path = candidates_dir / "metadata.json"
    meta = {
        "detector_type": detector_type,
        "grounding": grounding,
        "task": task,
        "top_k": top_k,
        "source": detector.__class__.__name__,
        "cache_key": "sample_id",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    for split in splits:
        queries_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        if not queries_path.exists():
            print(f"  [SKIP] {queries_path} not found (run step02 first)")
            continue

        target_map = _load_target_map(split, grounding, task)
        if not target_map:
            print(f"  [SKIP] {split}: missing {grounding} grounding targets")
            continue

        total = 0
        skipped_small_target = 0
        view_cache = {}

        with open(queries_path) as fin:
            lines = fin.readlines()

        for line in tqdm(lines, desc=f"Grasps [{split}]"):
            query = json.loads(line)
            sample_id = query["sample_id"]
            view_sample_id = query["view_sample_id"]
            if sample_id not in target_map:
                continue

            out_path = candidates_dir / f"{sample_id}.npz"
            if out_path.exists():
                total += 1
                continue

            ctx = view_cache.get(view_sample_id)
            if ctx is None:
                pcd_path = config.POINTCLOUDS_DIR / f"{view_sample_id}.npz"
                if not pcd_path.exists():
                    view_cache[view_sample_id] = None
                    continue
                pcd_data = np.load(str(pcd_path))
                ctx = {
                    "scene_points": pcd_data["points"],
                    "scene_pixel_coords": pcd_data["pixel_coords"],
                    "label": None,
                }
                if grounding == "oracle":
                    scene_dir = config.SCENES_DIR / f"scene_{query['scene_id']:04d}"
                    try:
                        ctx["label"] = load_label(scene_dir, query["frame_id"], query["camera"])
                    except Exception:
                        ctx["label"] = None
                view_cache[view_sample_id] = ctx

            if ctx is None:
                continue

            target = target_map[sample_id]
            target_mask = _load_target_mask(target, ctx["label"])
            pcr = build_point_cloud_representation(
                ctx["scene_points"],
                ctx["scene_pixel_coords"],
                target["bbox"],
                target_mask=target_mask,
            )

            if len(pcr.clean_target_points) < config.TARGET_MIN_POINTS:
                skipped_small_target += 1
                continue

            candidates = detector.detect(pcr.clean_target_points, top_k=top_k)

            if not candidates:
                continue

            _save_candidates(out_path, candidates)
            total += 1

        print(f"  [{split}] {total} target-conditioned candidate sets → {candidates_dir}")
        if skipped_small_target:
            print(f"           skipped {skipped_small_target} targets with too few points")


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Generate grasp candidates"
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--top-k", type=int, default=config.GRASP_TOP_K)
    parser.add_argument(
        "--grounding", type=str, default="predicted",
        choices=["predicted", "oracle"],
        help="Target source used before geometric sampling.",
    )
    parser.add_argument(
        "--task", type=str, default=config.DEFAULT_GROUNDING,
        choices=["phrase", "seg"],
        help="Predicted grounding task file to use.",
    )
    parser.add_argument(
        "--detector", type=str, default=config.DEFAULT_DETECTOR,
        choices=["geometric"],
        help="Detector type. Default: geometric (local RGB-D sampler).",
    )
    args = parser.parse_args()

    generate_candidates(
        splits=args.splits,
        top_k=args.top_k,
        detector_type=args.detector,
        grounding=args.grounding,
        task=args.task,
    )


if __name__ == "__main__":
    main()
