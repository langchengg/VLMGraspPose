"""
scripts/download_weights.py — Download pre-trained model weights
=================================================================
Downloads Florence-2-large-ft and GraspNet baseline checkpoint.

Usage:
    python scripts/download_weights.py --all
    python scripts/download_weights.py --florence2
    python scripts/download_weights.py --graspnet
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Florence-2-large fine-tuned
FLORENCE2_MODEL_ID = "microsoft/Florence-2-large-ft"
FLORENCE2_LOCAL_DIR = MODELS_DIR / "florence2"

# GraspNet baseline checkpoints
GRASPNET_CHECKPOINTS = {
    "kinect": {
        "gdrive_id": "1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX",
        "filename": "checkpoint-kn.tar",
    },
    "realsense": {
        "gdrive_id": "1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk",
        "filename": "checkpoint-rs.tar",
    },
}
GRASPNET_LOCAL_DIR = MODELS_DIR / "grasp_detector"


def download_florence2():
    """Download Florence-2-large-ft from HuggingFace Hub."""
    print(f"{'=' * 60}")
    print(f"Downloading Florence-2-large-ft")
    print(f"  Model ID: {FLORENCE2_MODEL_ID}")
    print(f"  Save to:  {FLORENCE2_LOCAL_DIR}")
    print(f"{'=' * 60}")

    if FLORENCE2_LOCAL_DIR.exists() and any(FLORENCE2_LOCAL_DIR.iterdir()):
        print("[SKIP] Already exists. Use --force to re-download.")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"]
        )
        from huggingface_hub import snapshot_download

    FLORENCE2_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=FLORENCE2_MODEL_ID,
            local_dir=str(FLORENCE2_LOCAL_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Downloaded → {FLORENCE2_LOCAL_DIR}")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def download_graspnet(camera: str = "realsense"):
    """Download GraspNet baseline checkpoint."""
    info = GRASPNET_CHECKPOINTS[camera]
    print(f"{'=' * 60}")
    print(f"Downloading GraspNet checkpoint ({camera})")
    print(f"  Save to: {GRASPNET_LOCAL_DIR / info['filename']}")
    print(f"{'=' * 60}")

    GRASPNET_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRASPNET_LOCAL_DIR / info["filename"]

    if out_path.exists():
        print(f"[SKIP] Already exists.")
        return True

    try:
        import gdown
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown"]
        )
        import gdown

    url = f"https://drive.google.com/uc?id={info['gdrive_id']}"
    try:
        gdown.download(url, str(out_path), quiet=False)
        if out_path.suffix == ".tar":
            import tarfile
            with tarfile.open(out_path) as tar:
                tar.extractall(path=str(GRASPNET_LOCAL_DIR))
        print(f"[OK] Downloaded → {out_path}")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def verify():
    """Show status of all model downloads."""
    print(f"\n{'=' * 60}")
    print(f"Model Weight Status")
    print(f"{'=' * 60}")
    models = [
        ("Florence-2-large-ft", FLORENCE2_LOCAL_DIR),
        ("GraspNet (RS)", GRASPNET_LOCAL_DIR / "checkpoint-rs.tar"),
        ("GraspNet (KN)", GRASPNET_LOCAL_DIR / "checkpoint-kn.tar"),
    ]
    for name, path in models:
        exists = path.exists() and (
            path.is_file() or (path.is_dir() and any(path.iterdir()))
        )
        status = "✅" if exists else "❌"
        print(f"  {status}  {name:<25s}  {path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained model weights"
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--florence2", action="store_true")
    parser.add_argument("--graspnet", action="store_true")
    parser.add_argument("--camera", type=str, default="realsense",
                        choices=["kinect", "realsense"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        verify()
        return

    if args.force:
        import shutil
        if (args.florence2 or args.all) and FLORENCE2_LOCAL_DIR.exists():
            shutil.rmtree(FLORENCE2_LOCAL_DIR)
        if (args.graspnet or args.all):
            for ckpt in GRASPNET_CHECKPOINTS.values():
                p = GRASPNET_LOCAL_DIR / ckpt["filename"]
                if p.exists():
                    p.unlink()

    if not (args.all or args.florence2 or args.graspnet):
        parser.print_help()
        verify()
        return

    results = []
    if args.florence2 or args.all:
        results.append(("Florence-2", download_florence2()))
    if args.graspnet or args.all:
        results.append(("GraspNet", download_graspnet(args.camera)))

    print(f"\n{'=' * 60}")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'}  {name}")
    print(f"{'=' * 60}")
    verify()


if __name__ == "__main__":
    main()
