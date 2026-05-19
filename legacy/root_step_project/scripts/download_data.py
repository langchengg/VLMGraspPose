"""
scripts/download_data.py — Download GraspNet-1Billion dataset
===============================================================
Downloads scene data, grasp labels, collision labels, and object models
from Google Drive to data/raw/graspnet/.

Usage:
    python scripts/download_data.py --test-seen
    python scripts/download_data.py --train
    python scripts/download_data.py --all
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "graspnet"

# ── Google Drive File IDs ────────────────────────────────────────────
GDRIVE_FILES = {
    "train_1": "1wQx8IJ_Lok3hVK_nchQUzgw88QBGZ5iq",
    "train_2": "1b1Z1goPV0o_wdwXZ8qTlHd2TBRU5-CmH",
    "train_3": "1oNcmZno2ymsDUWTmfFOxewMBjTXhL95c",
    "train_4": "1e8Xy7-lFhiXk0ugPOKvHKDiGTparmx00",
    "test_1": "1_nxiCmHhtsjCgA1IKJn3AuMq4SH_fseW",
    "test_2": "1njgthC-uUvTXgG99qq1fjS-fzofttFms",
    "test_3": "1xixvgY0yK7TEALq3k7JcJk2_SP_6r8nk",
    "grasp_label_1": "1FCV6j2J2eQpVk_ddJXljJvjRT1KU3sJ6",
    "grasp_label_2": "1p43sntiN9HJZRDFDNpzaEaEYoPY6IWsu",
    "collision_label": "1lR6ZSgtgV1KlqzM14mKlQ8oKhE3UCltO",
    "models": "1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5",
    "dex_models": "1RElNqUHNoA9l_muTGNu7yAc3ql_e7pL3",
}


def download_gdrive(file_id: str, output_path: str):
    try:
        import gdown
    except ImportError:
        print("[ERROR] gdown required: pip install gdown")
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading → {output_path}")
    gdown.download(url, output_path, quiet=False)


def extract_zip(zip_path: str, extract_to: str):
    print(f"  Extracting {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(zip_path)
    print(f"  Done.")


def download_scenes(keys: list, tmp_dir: Path):
    """Download and extract scene zips."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir = RAW_ROOT / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        file_id = GDRIVE_FILES[key]
        zip_path = str(tmp_dir / f"{key}.zip")

        if (tmp_dir / f"{key}.zip").exists():
            print(f"  [SKIP] {key}.zip already exists")
        else:
            download_gdrive(file_id, zip_path)

        if os.path.exists(zip_path):
            extract_zip(zip_path, str(scenes_dir))


def download_extras(keys: list, tmp_dir: Path):
    """Download supplementary files."""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        file_id = GDRIVE_FILES[key]
        zip_path = str(tmp_dir / f"{key}.zip")

        print(f"\nDownloading: {key}")
        download_gdrive(file_id, zip_path)

        if os.path.exists(zip_path):
            extract_zip(zip_path, str(RAW_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Download GraspNet-1Billion dataset",
        epilog="""
Data is extracted to: data/raw/graspnet/

Examples:
    python scripts/download_data.py --test-seen
    python scripts/download_data.py --all
        """,
    )
    parser.add_argument("--test-seen", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test-all", action="store_true")
    parser.add_argument("--grasp-labels", action="store_true")
    parser.add_argument("--collision-labels", action="store_true")
    parser.add_argument("--object-models", action="store_true")
    parser.add_argument("--dex-models", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default=None)
    args = parser.parse_args()

    if not any([args.test_seen, args.train, args.test_all,
                args.grasp_labels, args.collision_labels,
                args.object_models, args.dex_models, args.all]):
        parser.print_help()
        sys.exit(1)

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else PROJECT_ROOT / "_download_tmp"

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root:    {RAW_ROOT}")

    if args.all or args.test_seen or args.test_all:
        download_scenes(["test_1", "test_2", "test_3"], tmp_dir)
    if args.all or args.train:
        download_scenes(["train_1", "train_2", "train_3", "train_4"], tmp_dir)
    if args.all or args.grasp_labels:
        download_extras(["grasp_label_1", "grasp_label_2"], tmp_dir)
    if args.all or args.collision_labels:
        download_extras(["collision_label"], tmp_dir)
    if args.all or args.object_models:
        download_extras(["models"], tmp_dir)
    if args.all or args.dex_models:
        download_extras(["dex_models"], tmp_dir)

    # Cleanup
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

    print(f"\n{'=' * 60}")
    print(f"Download complete! Data at: {RAW_ROOT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
