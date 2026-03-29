"""
scripts/download_data.py — Download GraspNet-1Billion dataset
===============================================================
Downloads scene data from Google Drive using gdown.

Usage:
    # Download test_seen only (recommended for quick start)
    python scripts/download_data.py --test-seen

    # Download training data (scenes 0000-0099)
    python scripts/download_data.py --train

    # Download everything
    python scripts/download_data.py --all

Requires: pip install gdown
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Google Drive File IDs ────────────────────────────────────────────
# Source: https://graspnet.net/datasets.html

GDRIVE_FILES = {
    # Train images (scenes 0000-0099) — 4 parts
    "train_1": "1wQx8IJ_Lok3hVK_nchQUzgw88QBGZ5iq",
    "train_2": "1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH",
    "train_3": "1oNcmZno2ymsDUWTmfFOxewMBjTXhL95c",
    "train_4": "1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00",

    # Test images (scenes 0100-0189) — 3 parts
    "test_1": "1_nxiCmHhtsjCgA1IKJn3AuMq4SH_fseW",
    "test_2": "1njgthC-uUvTXgG99qq1fjS-fzofttFms",
    "test_3": "1xixvgY0yK7TEALq3k7JcJk2_SP_6r8nk",

    # 6-DoF grasp labels — 2 parts
    "grasp_label_1": "1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6",
    "grasp_label_2": "1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu",

    # Object 3D models
    "models": "1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5",

    # Dexnet models cache (optional, speeds up evaluation)
    "dex_models": "1RElNqUHNoA9l_muTGNu7yAc3ql_e7pL3",

    # Rectangle grasp labels (optional)
    "rect_labels": "1lR6ZSgtgV1KlqzM14mKlQ8oKhE3UCltO",
}

# ── Scene split mapping ─────────────────────────────────────────────
# GraspNet uses flat scene_XXXX structure. After extraction we
# reorganise into our split-based directories.

SPLIT_RANGES = {
    "train":        (0, 100),    # scene_0000 – scene_0099
    "test_seen":    (100, 130),  # scene_0100 – scene_0129
    "test_similar": (130, 160),  # scene_0130 – scene_0159
    "test_novel":   (160, 190),  # scene_0160 – scene_0189
}


def download_gdrive(file_id: str, output_path: str):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("[ERROR] gdown is required. Install with: pip install gdown")
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading → {output_path}")
    gdown.download(url, output_path, quiet=False)


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file."""
    print(f"  Extracting {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Done. Removing zip...")
    os.remove(zip_path)


def organise_scenes(raw_dir: Path, split: str):
    """Move scenes from flat graspnet structure into split directories.

    GraspNet extracts to: raw_dir/scenes/scene_XXXX/
    We move to: PROJECT_ROOT/<split>/scene_XXXX/
    """
    scenes_dir = raw_dir / "scenes"
    if not scenes_dir.exists():
        # Try without 'scenes' subdirectory
        scenes_dir = raw_dir

    start, end = SPLIT_RANGES[split]
    target_dir = PROJECT_ROOT / split
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for scene_id in range(start, end):
        scene_name = f"scene_{scene_id:04d}"
        src = scenes_dir / scene_name
        dst = target_dir / scene_name

        if src.exists() and not dst.exists():
            src.rename(dst)
            moved += 1
        elif src.exists() and dst.exists():
            print(f"  [SKIP] {scene_name} already exists in {split}/")

    print(f"  Organised {moved} scenes into {split}/")


def download_split(split: str, tmp_dir: Path):
    """Download and organise a specific data split."""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        keys = ["train_1", "train_2", "train_3", "train_4"]
        desc = "Training scenes (0000-0099)"
    elif split in ("test_seen", "test_similar", "test_novel"):
        keys = ["test_1", "test_2", "test_3"]
        desc = "Test scenes (0100-0189)"
    else:
        print(f"[ERROR] Unknown split: {split}")
        return

    print(f"\n{'='*60}")
    print(f"Downloading: {desc}")
    print(f"{'='*60}")

    for key in keys:
        file_id = GDRIVE_FILES[key]
        zip_name = f"{key}.zip"
        zip_path = str(tmp_dir / zip_name)

        # Skip if already downloaded
        if (tmp_dir / zip_name).exists():
            print(f"  [SKIP] {zip_name} already exists")
        else:
            download_gdrive(file_id, zip_path)

        # Extract
        if os.path.exists(zip_path):
            extract_zip(zip_path, str(tmp_dir))

    # Organise into split directory
    organise_scenes(tmp_dir, split)


def download_extras(keys: list, tmp_dir: Path):
    """Download supplementary files (models, labels, etc.)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        file_id = GDRIVE_FILES[key]
        zip_name = f"{key}.zip"
        zip_path = str(tmp_dir / zip_name)

        print(f"\nDownloading: {key}")
        download_gdrive(file_id, zip_path)

        if os.path.exists(zip_path):
            extract_zip(zip_path, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Download GraspNet-1Billion dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick start — test_seen only (~7 GB)
    python scripts/download_data.py --test-seen

    # Full training set (~30 GB)
    python scripts/download_data.py --train

    # Everything including labels and models (~60 GB)
    python scripts/download_data.py --all

Data source: https://graspnet.net/datasets.html
        """,
    )
    parser.add_argument("--test-seen", action="store_true",
                        help="Download test_seen split (scenes 0100-0129)")
    parser.add_argument("--train", action="store_true",
                        help="Download training split (scenes 0000-0099)")
    parser.add_argument("--test-all", action="store_true",
                        help="Download all test splits (seen + similar + novel)")
    parser.add_argument("--grasp-labels", action="store_true",
                        help="Download 6-DoF grasp labels")
    parser.add_argument("--object-models", action="store_true",
                        help="Download object 3D models")
    parser.add_argument("--all", action="store_true",
                        help="Download everything")
    parser.add_argument("--tmp-dir", type=str, default=None,
                        help="Temporary directory for downloads")
    args = parser.parse_args()

    if not any([args.test_seen, args.train, args.test_all,
                args.grasp_labels, args.object_models, args.all]):
        parser.print_help()
        print("\n[ERROR] Specify at least one download option.")
        sys.exit(1)

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else PROJECT_ROOT / "_download_tmp"

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Temp directory: {tmp_dir}")

    # ── Download splits ──────────────────────────────────────────────
    if args.all or args.test_seen or args.test_all:
        download_split("test_seen", tmp_dir)

    if args.all or args.test_all:
        download_split("test_similar", tmp_dir)
        download_split("test_novel", tmp_dir)

    if args.all or args.train:
        download_split("train", tmp_dir)

    # ── Download extras ──────────────────────────────────────────────
    if args.all or args.grasp_labels:
        download_extras(["grasp_label_1", "grasp_label_2"], tmp_dir)

    if args.all or args.object_models:
        download_extras(["models"], tmp_dir)

    # ── Cleanup ──────────────────────────────────────────────────────
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"{'='*60}")

    # Verify
    for split in ["test_seen", "train"]:
        split_dir = PROJECT_ROOT / split
        if split_dir.exists():
            n_scenes = len(list(split_dir.glob("scene_*")))
            print(f"  {split}: {n_scenes} scenes")
        else:
            print(f"  {split}: not downloaded")


if __name__ == "__main__":
    main()
