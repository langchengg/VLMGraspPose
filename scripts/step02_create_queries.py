"""
scripts/step02_create_queries.py — Generate text queries
==========================================================
Step 2: For each (view, visible_object) pair, generate a class-level
text query from templates.

Usage:
    python scripts/step02_create_queries.py
    python scripts/step02_create_queries.py --splits train val
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def create_queries(splits: list = None):
    """Generate queries from view indexes + object names + templates."""
    if splits is None:
        splits = config.ALL_SPLITS

    name_map = config.get_object_name_map()
    templates = config.get_query_templates()
    class_templates = templates.get("class", ["pick the {obj}"])

    if not name_map:
        print("[ERROR] Object name map is empty.")
        print(f"        Check {config.OBJECT_ID_TO_NAME_PATH}")
        return

    rng = random.Random(42)

    for split in splits:
        views_path = config.SPLITS_DIR / f"{split}_views.jsonl"
        if not views_path.exists():
            print(f"  [SKIP] {views_path} not found (run step01 first)")
            continue

        out_path = config.QUERIES_DIR / f"{split}_queries.jsonl"
        config.QUERIES_DIR.mkdir(parents=True, exist_ok=True)

        total = 0
        with open(views_path) as fin, open(out_path, "w") as fout:
            for line in tqdm(fin, desc=f"Queries [{split}]"):
                view = json.loads(line)
                vis_ids = view.get("visible_object_ids", [])

                for obj_id in vis_ids:
                    obj_name = name_map.get(obj_id)
                    if obj_name is None:
                        continue

                    template = rng.choice(class_templates)
                    text_query = template.format(obj=obj_name)

                    record = {
                        "sample_id": (
                            f"{view['sample_id']}_{obj_id:03d}_{obj_name.replace(' ', '_')}"
                        ),
                        "view_sample_id": view["sample_id"],
                        "scene_id": view["scene_id"],
                        "camera": view["camera"],
                        "frame_id": view["frame_id"],
                        "target_object_id": obj_id,
                        "object_name": obj_name,
                        "query_type": "class",
                        "text_query": text_query,
                        "split": split,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1

        print(f"  [{split}] {total} queries → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Create text queries"
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Splits to process (default: all)"
    )
    args = parser.parse_args()

    create_queries(splits=args.splits)


if __name__ == "__main__":
    main()
