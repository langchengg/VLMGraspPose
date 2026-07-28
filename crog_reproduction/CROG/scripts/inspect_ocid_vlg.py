#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

import utils.config as config
from utils.dataset import OCIDVLGDataset


def describe(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            describe(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, torch.Tensor):
        print(f"  {prefix}: tensor shape={tuple(value.shape)} dtype={value.dtype}")
    elif isinstance(value, np.ndarray):
        print(f"  {prefix}: ndarray shape={value.shape} dtype={value.dtype}")
    elif isinstance(value, (list, tuple)):
        print(f"  {prefix}: {type(value).__name__} length={len(value)}")
    else:
        print(f"  {prefix}: {type(value).__name__} value={value!r}")


def sample_paths(dataset, index):
    return {
        "rgb": Path(dataset.root_dir) / dataset.rgb_paths[index],
        "depth": Path(dataset.root_dir) / dataset.depth_paths[index],
        "mask": Path(dataset.root_dir) / dataset.mask_paths[index],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect an OCID-VLG dataset")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    root = Path(cfg.root_path).expanduser().resolve()
    print(f"dataset_root: {root}")
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")

    missing = []
    for split in ("train", "val", "test"):
        print(f"\n[{split}]")
        try:
            dataset = OCIDVLGDataset(
                root_dir=str(root),
                input_size=cfg.input_size,
                word_length=cfg.word_len,
                split=split,
                version=cfg.version,
            )
            print(f"sample_count: {len(dataset)}")
            paths = sample_paths(dataset, 0)
            for kind, path in paths.items():
                exists = path.is_file()
                print(f"{kind}_file: {path} exists={exists}")
                if not exists:
                    missing.append(path)
            sample = dataset[0]
            print(f"returned_keys: {sorted(sample.keys())}")
            print(f"language_expression: {sample['sentence']}")
            describe(sample)
        except Exception as error:
            print(f"ERROR loading {split}: {type(error).__name__}: {error}")
            missing.append(Path(f"<{split} load failed>"))

    if missing:
        print("\nMissing or unreadable dataset entries:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)
    print("\nDataset inspection completed without missing sample files.")


if __name__ == "__main__":
    main()
