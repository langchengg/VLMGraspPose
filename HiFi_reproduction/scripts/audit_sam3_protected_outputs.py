#!/usr/bin/env python3
"""Record or verify content hashes for the pre-SAM baseline artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.sam3_serialization import save_strict_json, sha256_file  # noqa: E402


PROTECTED = (
    "runs/hifics_ocidvlg_20260711_112921/predictions",
    "runs/hifics_ocidvlg_20260711_112921/anygrasp_input_predicted_mask",
    "outputs/dexnet_candidates_ten_samples",
    "outputs/gqcnn_original_ranking_evaluation",
)


def _snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in PROTECTED:
        root = REPO_ROOT / relative
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                result[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
            elif path.is_symlink():
                raise ValueError(f"protected output contains symlink: {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "verify"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "outputs" / "sam3_protected_baseline_hashes.json",
    )
    args = parser.parse_args()
    current = _snapshot()
    if args.mode == "record":
        if args.manifest.exists():
            raise FileExistsError(f"refusing to overwrite protected-hash manifest: {args.manifest}")
        save_strict_json(
            args.manifest,
            {"schema_version": 1, "protected_roots": PROTECTED, "files": current},
        )
        print(json.dumps({"status": "RECORDED", "files": len(current)}))
        return 0
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))["files"]
    missing = sorted(set(expected) - set(current))
    extra = sorted(set(current) - set(expected))
    changed = sorted(path for path in set(current) & set(expected) if current[path] != expected[path])
    if missing or extra or changed:
        raise RuntimeError(
            f"protected outputs changed: missing={missing[:5]} extra={extra[:5]} changed={changed[:5]}"
        )
    print(json.dumps({"status": "VERIFIED_UNCHANGED", "files": len(current)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

