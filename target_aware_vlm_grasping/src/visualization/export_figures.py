from __future__ import annotations

from pathlib import Path

import shutil


def copy_representative_figures(output_root: Path, figure_root: Path, limit: int = 12) -> int:
    figure_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in sorted(output_root.glob("*/*/*/*/visualization_rgb.png")):
        if copied >= limit:
            break
        dst = figure_root / f"{img.parents[3].name}_{img.parents[2].name}_{img.parents[1].name}_{img.parent.name}.png"
        shutil.copyfile(img, dst)
        copied += 1
    return copied
