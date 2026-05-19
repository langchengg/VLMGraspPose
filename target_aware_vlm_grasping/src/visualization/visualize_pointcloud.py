from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_pointcloud_figure(path: Path, pcd, grasps=None, title: str = "Target point cloud") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(pcd.points) if pcd is not None else np.zeros((0, 3))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
    for grasp in grasps or []:
        c = grasp.candidate if hasattr(grasp, "candidate") else grasp
        p = c.position
        a = c.approach_vector
        d = c.closing_direction
        ax.quiver(p[0], p[1], p[2], a[0], a[1], a[2], length=0.04, color="red")
        ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], length=0.04, color="blue")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
