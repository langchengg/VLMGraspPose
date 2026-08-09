"""Ground-truth-only visual-grounding evaluation helpers.

Formal inference modules must not import this module. It is intentionally kept
separate so static leakage guards can reject accidental GT dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import load_binary_mask


def load_frozen_ground_truth_manifest(
    manifest_path: str | Path,
    *,
    hifics_root: str | Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != int(expected_count):
        raise ValueError(
            f"ground-truth split manifest must contain {expected_count} rows"
        )
    root = Path(hifics_root).expanduser().resolve()
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        gt_path = Path(str(row["mask_path"]).replace("\\", "/"))
        if not gt_path.is_absolute():
            gt_path = root / gt_path
        gt_path = gt_path.resolve()
        if not gt_path.is_file() or gt_path.is_symlink():
            raise ValueError(f"missing or unsafe GT mask: {gt_path}")
        result.append({**row, "sample_index": index, "gt_mask_path": str(gt_path)})
    return result


def load_ground_truth_mask(row: dict[str, Any]) -> Any:
    return load_binary_mask(row["gt_mask_path"], expected_shape=(352, 352))
