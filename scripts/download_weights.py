"""
scripts/download_weights.py — Download pre-trained model weights
=================================================================
Downloads model weights from HuggingFace / Google Drive to the local
``models/`` directory so that VLMGraspPose pipeline modules can load
them directly.

Supported models
----------------
1. Florence-2-base    — VLM for open-vocabulary target grounding  (Stage 1)
2. Grounding DINO     — Alternative grounding model               (Stage 1)
3. GraspNet baseline  — 6-DoF grasp detection checkpoint          (Stage 2)

Usage
-----
    # Download all models
    python scripts/download_weights.py --all

    # Download only Florence-2
    python scripts/download_weights.py --florence2

    # Download only Grounding DINO
    python scripts/download_weights.py --grounding-dino

    # Download only GraspNet checkpoint (Kinect)
    python scripts/download_weights.py --graspnet --camera kinect

    # List available models
    python scripts/download_weights.py --list
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# =====================================================================
#  Florence-2 (HuggingFace)
# =====================================================================

FLORENCE2_MODEL_ID = "microsoft/Florence-2-base"
FLORENCE2_LOCAL_DIR = MODELS_DIR / "florence-2-base"


def download_florence2():
    """Download Florence-2-base weights from HuggingFace Hub."""
    print("=" * 60)
    print("Downloading Florence-2-base from HuggingFace")
    print(f"  Model ID : {FLORENCE2_MODEL_ID}")
    print(f"  Save to  : {FLORENCE2_LOCAL_DIR}")
    print("=" * 60)

    if FLORENCE2_LOCAL_DIR.exists() and any(FLORENCE2_LOCAL_DIR.iterdir()):
        print("[SKIP] Florence-2-base already exists. Use --force to re-download.")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[INFO] Installing huggingface_hub ...")
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
        print(f"[OK] Florence-2-base downloaded → {FLORENCE2_LOCAL_DIR}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download Florence-2: {e}")
        return False


# =====================================================================
#  Grounding DINO (HuggingFace)
# =====================================================================

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
GDINO_LOCAL_DIR = MODELS_DIR / "grounding-dino-base"


def download_grounding_dino():
    """Download Grounding DINO base weights from HuggingFace Hub."""
    print("=" * 60)
    print("Downloading Grounding DINO base from HuggingFace")
    print(f"  Model ID : {GDINO_MODEL_ID}")
    print(f"  Save to  : {GDINO_LOCAL_DIR}")
    print("=" * 60)

    if GDINO_LOCAL_DIR.exists() and any(GDINO_LOCAL_DIR.iterdir()):
        print("[SKIP] Grounding DINO already exists. Use --force to re-download.")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[INFO] Installing huggingface_hub ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"]
        )
        from huggingface_hub import snapshot_download

    GDINO_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=GDINO_MODEL_ID,
            local_dir=str(GDINO_LOCAL_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] Grounding DINO downloaded → {GDINO_LOCAL_DIR}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download Grounding DINO: {e}")
        return False


# =====================================================================
#  GraspNet Baseline Checkpoint (Google Drive)
# =====================================================================

GRASPNET_CHECKPOINTS = {
    "kinect": {
        "gdrive_id": "1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX",
        "filename": "checkpoint-kn.tar",
        "desc": "Kinect camera (matches project default CAMERA_TYPE='kinect')",
    },
    "realsense": {
        "gdrive_id": "1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk",
        "filename": "checkpoint-rs.tar",
        "desc": "RealSense camera (recommended for better transfer)",
    },
}

GRASPNET_LOCAL_DIR = MODELS_DIR / "graspnet-baseline"


def download_graspnet(camera: str = "kinect"):
    """Download GraspNet baseline pre-trained checkpoint from Google Drive."""
    info = GRASPNET_CHECKPOINTS[camera]
    print("=" * 60)
    print(f"Downloading GraspNet baseline checkpoint ({camera})")
    print(f"  Description : {info['desc']}")
    print(f"  Save to     : {GRASPNET_LOCAL_DIR / info['filename']}")
    print("=" * 60)

    GRASPNET_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GRASPNET_LOCAL_DIR / info["filename"]

    if out_path.exists():
        print(f"[SKIP] {out_path.name} already exists. Use --force to re-download.")
        return True

    try:
        import gdown
    except ImportError:
        print("[INFO] Installing gdown (Google Drive downloader) ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown"]
        )
        import gdown

    url = f"https://drive.google.com/uc?id={info['gdrive_id']}"

    try:
        gdown.download(url, str(out_path), quiet=False)
        print(f"[OK] {info['filename']} downloaded → {out_path}")

        # Extract checkpoint
        if out_path.suffix == ".tar":
            print(f"[INFO] Extracting {info['filename']} ...")
            import tarfile
            with tarfile.open(out_path) as tar:
                tar.extractall(path=str(GRASPNET_LOCAL_DIR))
            print(f"[OK] Extracted to {GRASPNET_LOCAL_DIR}")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to download GraspNet checkpoint: {e}")
        print(f"[TIP]  You can manually download from:")
        print(f"       https://drive.google.com/file/d/{info['gdrive_id']}/view")
        print(f"       and place it at: {out_path}")
        return False


# =====================================================================
#  Verify
# =====================================================================

def verify_models():
    """Print status of all model downloads."""
    print("=" * 60)
    print("Model Weight Status")
    print("=" * 60)

    models = [
        ("Florence-2-base", FLORENCE2_LOCAL_DIR, "Stage 1 — VLM Grounding"),
        ("Grounding DINO",  GDINO_LOCAL_DIR,    "Stage 1 — Alternative Grounding"),
        ("GraspNet (KN)",   GRASPNET_LOCAL_DIR / "checkpoint-kn.tar",
         "Stage 2 — Grasp Generation (Kinect)"),
        ("GraspNet (RS)",   GRASPNET_LOCAL_DIR / "checkpoint-rs.tar",
         "Stage 2 — Grasp Generation (RealSense)"),
    ]

    for name, path, usage in models:
        exists = path.exists() and (
            path.is_file() or (path.is_dir() and any(path.iterdir()))
        )
        status = "✅ READY" if exists else "❌ NOT FOUND"
        print(f"  {status}  {name:<20s}  {path}")
        print(f"           Usage: {usage}")
    print("=" * 60)


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained model weights for VLMGraspPose"
    )
    parser.add_argument("--all", action="store_true",
                        help="Download all models")
    parser.add_argument("--florence2", action="store_true",
                        help="Download Florence-2-base (HuggingFace)")
    parser.add_argument("--grounding-dino", action="store_true",
                        help="Download Grounding DINO base (HuggingFace)")
    parser.add_argument("--graspnet", action="store_true",
                        help="Download GraspNet baseline checkpoint")
    parser.add_argument("--camera", type=str, default="kinect",
                        choices=["kinect", "realsense"],
                        help="GraspNet checkpoint camera type (default: kinect)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    parser.add_argument("--list", action="store_true",
                        help="Show status of all models")
    args = parser.parse_args()

    # Show status
    if args.list:
        verify_models()
        return

    # If --force, remove existing dirs
    if args.force:
        import shutil
        if args.florence2 or args.all:
            if FLORENCE2_LOCAL_DIR.exists():
                shutil.rmtree(FLORENCE2_LOCAL_DIR)
        if args.grounding_dino or args.all:
            if GDINO_LOCAL_DIR.exists():
                shutil.rmtree(GDINO_LOCAL_DIR)
        if args.graspnet or args.all:
            for ckpt in GRASPNET_CHECKPOINTS.values():
                p = GRASPNET_LOCAL_DIR / ckpt["filename"]
                if p.exists():
                    p.unlink()

    # Nothing selected → show help
    if not (args.all or args.florence2 or args.grounding_dino or args.graspnet):
        parser.print_help()
        print("\n")
        verify_models()
        return

    results = []

    if args.florence2 or args.all:
        results.append(("Florence-2", download_florence2()))

    if args.grounding_dino or args.all:
        results.append(("Grounding DINO", download_grounding_dino()))

    if args.graspnet or args.all:
        camera = args.camera if not args.all else "kinect"
        results.append(("GraspNet", download_graspnet(camera)))
        if args.all:
            results.append(("GraspNet (RS)", download_graspnet("realsense")))

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'}  {name}")
    print("=" * 60)

    verify_models()


if __name__ == "__main__":
    main()
