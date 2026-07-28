#!/usr/bin/env python3
"""Render paths.local.yaml as shell-safe assignments for repository scripts."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="configs/paths.local.yaml")
    args = parser.parse_args()
    values = yaml.safe_load(Path(args.path).read_text(encoding="utf-8")) or {}
    manifests = values.get("manifests") or {}
    mapping = {
        "OCID_ROOT": values.get("ocid_root"),
        "OCID_API_ROOT": values.get("ocid_api_root"),
        "TRAIN_MANIFEST": manifests.get("train"),
        "VAL_MANIFEST": manifests.get("val"),
        "TEST_MANIFEST": manifests.get("test"),
        "HIFICS_ROOT": values.get("hifi_root"),
        "HIFICS_EXPORT_MANIFEST": values.get("hifi_prediction_manifest"),
    }
    missing = [name for name, value in mapping.items() if not value and name not in {
        "HIFICS_ROOT", "HIFICS_EXPORT_MANIFEST"
    }]
    if missing:
        raise ValueError(f"paths file is missing: {', '.join(missing)}")
    for name, value in mapping.items():
        if value:
            print(f"{name}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
