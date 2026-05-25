from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path: Path, depth_scale: float = 1000.0) -> np.ndarray:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Reading .h5 depth files requires h5py.") from exc
        with h5py.File(path, "r") as handle:
            if "depth" not in handle:
                raise KeyError(f"Depth H5 file does not contain a 'depth' dataset: {path}")
            depth_raw = handle["depth"][()]
        return depth_raw.astype(np.float32) / float(depth_scale)
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    return depth_raw.astype(np.float32) / float(depth_scale)


def load_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(f"Could not read label image: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label


def load_intrinsics(path: Path | None, fallback: dict | None = None) -> np.ndarray:
    if path and path.exists():
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(str(path))
            if arr.shape == (3, 3):
                return arr.astype(float)
        if suffix in {".json", ".yaml", ".yml"}:
            with open(path) as f:
                data = json.load(f) if suffix == ".json" else yaml.safe_load(f)
            return intrinsics_from_dict(data)
    if fallback:
        return intrinsics_from_dict(fallback)
    raise FileNotFoundError(f"Missing intrinsics and no fallback supplied: {path}")


def intrinsics_from_dict(data: dict) -> np.ndarray:
    fx, fy = float(data["fx"]), float(data["fy"])
    cx, cy = float(data["cx"]), float(data["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
