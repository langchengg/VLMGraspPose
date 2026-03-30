"""
scripts/step04_florence_grounding.py — Run Florence-2 grounding
=================================================================
Step 4: Use Florence-2-large to predict target bbox & optional mask.

Usage:
    python scripts/step04_florence_grounding.py --splits test_seen
    python scripts/step04_florence_grounding.py --splits train --task phrase
    python scripts/step04_florence_grounding.py --splits test_seen --task seg
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import load_rgb
from src.grounding import get_grounder


def run_grounding(
    splits: list = None,
    task: str = "phrase",
    max_samples: int = None,
):
    """Run Florence-2 grounding on all queries."""
    if splits is None:
        splits = config.TEST_SPLITS

    grounder = get_grounder(task)

    for split in splits:
        queries_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        if not queries_path.exists():
            print(f"  [SKIP] {queries_path} not found (run step02 first)")
            continue

        out_path = config.GROUNDING_PRED_DIR / f"{split}_grounding_{task}.jsonl"
        config.GROUNDING_PRED_DIR.mkdir(parents=True, exist_ok=True)

        total = 0
        failed = 0

        with open(queries_path) as fin, open(out_path, "w") as fout:
            for line in tqdm(fin, desc=f"Grounding [{split}]"):
                if max_samples and total >= max_samples:
                    break

                query = json.loads(line)
                scene_id = query["scene_id"]
                camera = query["camera"]
                frame_id = query["frame_id"]
                text_query = query["text_query"]

                scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"

                try:
                    rgb = load_rgb(scene_dir, frame_id, camera)
                except Exception as e:
                    failed += 1
                    continue

                t0 = time.time()
                result = grounder.ground(rgb, text_query)
                elapsed = time.time() - t0

                if result is None:
                    failed += 1
                    continue

                # Save mask separately if available
                pred_mask_path = None
                if result.mask is not None:
                    mask_dir = config.GROUNDING_PRED_DIR / f"pred_masks_{task}"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_fn = f"{query['sample_id']}_mask.npy"
                    pred_mask_path = str(mask_dir / mask_fn)
                    np.save(pred_mask_path, result.mask)

                record = {
                    "sample_id": query["sample_id"],
                    "text_query": text_query,
                    "pred_bbox": result.bbox,
                    "pred_mask_path": pred_mask_path,
                    "florence_confidence": result.confidence,
                    "grounding_latency": elapsed,
                    "split": split,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

        print(f"  [{split}] {total} predictions → {out_path}")
        if failed:
            print(f"           ({failed} failed)")


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Run Florence-2 grounding"
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Splits to process (default: test splits)"
    )
    parser.add_argument(
        "--task", type=str, default="phrase",
        choices=["phrase", "seg"],
        help="Florence-2 task: phrase (box) or seg (mask)"
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_grounding(
        splits=args.splits,
        task=args.task,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
