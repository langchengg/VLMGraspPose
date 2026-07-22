#!/usr/bin/env python3
"""Render the publication architecture schematic with editable SVG text."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def box(ax, xy, width, height, text, *, face, edge, fontsize=7, weight="normal", zorder=3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#17212B",
        linespacing=1.25,
        zorder=zorder + 1,
    )
    return patch


def arrow(ax, start, end, *, color="#51606F", dashed=False, curve=0.0, width=1.2, zorder=2):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=width,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=1,
        shrinkB=1,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def stage_label(ax, x, number, title):
    ax.text(x, 0.92, f"{number}", fontsize=7, fontweight="bold", color="#FFFFFF", ha="center", va="center",
            bbox={"boxstyle": "circle,pad=0.28", "facecolor": "#34495E", "edgecolor": "none"})
    ax.text(x, 0.865, title, fontsize=7.5, fontweight="bold", color="#263746", ha="center", va="center")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.suptitle(
        "Coarse-to-Fine Target Mask Refinement for Grasp Generation",
        fontsize=11,
        fontweight="bold",
        y=0.985,
        color="#17212B",
    )
    stage_label(ax, 0.235, "1", "Language grounding")
    stage_label(ax, 0.475, "2", "Mask refinement")
    stage_label(ax, 0.71, "3", "Candidate generation")
    stage_label(ax, 0.895, "4", "Grasp ranking")

    # Inputs
    box(ax, (0.025, 0.69), 0.115, 0.09, "RGB image", face="#F4F6F7", edge="#8896A3", weight="bold")
    box(ax, (0.025, 0.53), 0.115, 0.09, "Language\nquery", face="#F4F6F7", edge="#8896A3", weight="bold")
    box(ax, (0.025, 0.16), 0.115, 0.09, "Full depth\n(unchanged)", face="#EEF7F7", edge="#4E8D8A", weight="bold")

    # Stage 1
    box(ax, (0.175, 0.57), 0.125, 0.14, "HiFi-CS\ngrounder", face="#E9F2FB", edge="#4B83B6", fontsize=8, weight="bold")
    box(ax, (0.175, 0.39), 0.125, 0.09, "Coarse target\nmask", face="#DCEBFA", edge="#4B83B6", weight="bold")
    arrow(ax, (0.14, 0.735), (0.175, 0.66))
    arrow(ax, (0.14, 0.575), (0.175, 0.62))
    arrow(ax, (0.2375, 0.57), (0.2375, 0.48), color="#4B83B6")

    # Stage 2
    box(ax, (0.34, 0.63), 0.13, 0.13, "Prompt builder\nbox  +  interior points\nnegative points (ablation)", face="#F2ECFA", edge="#7C63A8", fontsize=6.2, weight="bold")
    box(ax, (0.50, 0.63), 0.115, 0.13, "Official SAM 3\nvisual-prompt\nrefinement", face="#EEE6F8", edge="#7C63A8", fontsize=7.2, weight="bold")
    box(ax, (0.50, 0.47), 0.115, 0.085, "Candidate\nrefined masks", face="#F7F3FB", edge="#9A83BE")
    box(ax, (0.34, 0.34), 0.275, 0.075, "Inference-only selector  +  explicit HiFi fallback", face="#FFF4DE", edge="#C98A2E", fontsize=6.4, weight="bold")
    box(ax, (0.42, 0.21), 0.125, 0.08, "Selected refined\ntarget mask", face="#E9F5EC", edge="#4D9562", weight="bold")
    arrow(ax, (0.30, 0.435), (0.34, 0.68), color="#4B83B6", curve=-0.12)
    arrow(ax, (0.47, 0.695), (0.50, 0.695), color="#7C63A8")
    arrow(ax, (0.5575, 0.63), (0.5575, 0.555), color="#7C63A8")
    arrow(ax, (0.5575, 0.47), (0.53, 0.415), color="#7C63A8")
    arrow(ax, (0.405, 0.63), (0.405, 0.415), color="#7C63A8", curve=0.08)
    arrow(ax, (0.475, 0.34), (0.4825, 0.29), color="#4D9562")
    arrow(ax, (0.30, 0.415), (0.34, 0.38), color="#C98A2E", dashed=True)

    # Stage 3
    box(ax, (0.655, 0.30), 0.13, 0.13, "Dex-Net\nantipodal sampler", face="#E9F5EC", edge="#4D9562", fontsize=8, weight="bold")
    box(ax, (0.655, 0.53), 0.13, 0.09, "Planar 4-DoF\ncandidates", face="#F1F8F3", edge="#4D9562", weight="bold")
    arrow(ax, (0.545, 0.25), (0.655, 0.35), color="#4D9562", curve=-0.10)
    arrow(ax, (0.14, 0.205), (0.655, 0.335), color="#4E8D8A", curve=-0.20, width=1.8)
    ax.text(0.39, 0.12, "depth bypasses SAM 3", fontsize=6.5, fontweight="bold", color="#347774", ha="center")
    arrow(ax, (0.72, 0.43), (0.72, 0.53), color="#4D9562")

    # Baseline branch: coarse mask reaches the same unchanged sampler.
    arrow(ax, (0.30, 0.42), (0.655, 0.39), color="#7D8790", dashed=True, curve=0.12)
    ax.text(
        0.49,
        0.445,
        "baseline: coarse-mask branch",
        fontsize=5.8,
        color="#68737D",
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
    )

    # Stage 4
    box(ax, (0.825, 0.63), 0.14, 0.10, "GQ-CNN\nq-value ranking", face="#FCEEDA", edge="#C98A2E", weight="bold")
    box(ax, (0.825, 0.43), 0.14, 0.10, "Target-aware\ngeometric ranking", face="#FCEEDA", edge="#C98A2E", weight="bold")
    box(ax, (0.825, 0.22), 0.14, 0.10, "Top-K / Top-1\nplanar grasp", face="#F8E1C1", edge="#B8751B", weight="bold")
    arrow(ax, (0.785, 0.585), (0.825, 0.68), color="#C98A2E", curve=-0.10)
    arrow(ax, (0.785, 0.57), (0.825, 0.48), color="#C98A2E", curve=0.10)
    arrow(ax, (0.965, 0.68), (0.965, 0.27), color="#C98A2E", curve=0.20)
    arrow(ax, (0.895, 0.43), (0.895, 0.32), color="#C98A2E")

    ax.text(
        0.50,
        0.025,
        "HiFi-CS chooses which instance; SAM 3 refines only its 2D boundary. Ground truth is evaluation-only.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#53616E",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.92)
    fig.savefig(OUTPUT / "coarse_to_fine_mask_architecture.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTPUT / "coarse_to_fine_mask_architecture.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
