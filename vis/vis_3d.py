"""
vis/vis_3d.py — 3D Point Cloud + Grasp Pose Visualisation
===========================================================
Uses Matplotlib for 3D scatter plots with gripper frames.
Optionally uses Open3D for interactive viewing if installed.

Usage:
    python -m vis.vis_3d --sample <sample_id>
    python -m vis.vis_3d --sample <sample_id> --backend open3d
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import (
    load_rgb, load_depth, load_label,
    load_camera_intrinsics, get_factor_depth,
)
from src.point_cloud import (
    backproject_depth, crop_point_cloud_by_bbox,
    crop_point_cloud_by_mask, voxel_downsample,
)
from vis.grasp_drawing import (
    to_rotation_matrix, gripper_keypoints, transform_gripper, rank_colour,
)


# ── Matplotlib 3D Backend ───────────────────────────────────────────

def visualise_3d_matplotlib(
    scene_points: np.ndarray,
    target_points: np.ndarray,
    selections: List[Dict],
    sample_id: str,
    output_dir: Path = None,
    scene_rgb: Optional[np.ndarray] = None,
    scene_pixel_coords: Optional[np.ndarray] = None,
    rgb_image: Optional[np.ndarray] = None,
    max_scene_pts: int = 8000,
    max_target_pts: int = 3000,
):
    """Create a 3D scatter plot with gripper poses using Matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

    # ── Scene point cloud (subsampled, grey) ─────────────────────────
    if len(scene_points) > max_scene_pts:
        idx = np.random.choice(len(scene_points), max_scene_pts, replace=False)
        scene_sub = scene_points[idx]
        # Try to colour from RGB image if available
        if (scene_rgb is not None and scene_pixel_coords is not None
                and rgb_image is not None):
            px = scene_pixel_coords[idx]
            H, W = rgb_image.shape[:2]
            valid_px = ((px[:, 0] >= 0) & (px[:, 0] < W)
                        & (px[:, 1] >= 0) & (px[:, 1] < H))
            colours = np.full((len(scene_sub), 3), 0.5)
            colours[valid_px] = rgb_image[
                px[valid_px, 1].astype(int),
                px[valid_px, 0].astype(int)
            ] / 255.0
        else:
            colours = np.full((len(scene_sub), 3), 0.55)
    else:
        scene_sub = scene_points
        colours = np.full((len(scene_sub), 3), 0.55)

    ax.scatter(scene_sub[:, 0], scene_sub[:, 1], scene_sub[:, 2],
               c=colours, s=0.3, alpha=0.3, rasterized=True)

    # ── Target point cloud (highlighted) ─────────────────────────────
    if len(target_points) > max_target_pts:
        idx = np.random.choice(len(target_points), max_target_pts, replace=False)
        target_sub = target_points[idx]
    else:
        target_sub = target_points

    ax.scatter(target_sub[:, 0], target_sub[:, 1], target_sub[:, 2],
               c="#00ff88", s=2, alpha=0.7, label="Target", rasterized=True)

    # ── Draw gripper poses ───────────────────────────────────────────
    for sel in selections:
        rank = sel["rank"]
        pos = sel["position"]
        rot = sel["rotation"]  # flattened 3×3 or quaternion
        width = sel["width"]

        # Get colour — convert BGR to RGB
        bgr = rank_colour(rank)
        rgb_col = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)

        # Transform gripper keypoints to camera frame
        pts = transform_gripper(pos, rot, width)

        # Draw gripper skeleton
        ax.plot3D(*zip(pts[0], pts[1]), color=rgb_col,
                  linewidth=2.5 if rank == 1 else 1.5)
        ax.plot3D(*zip(pts[2], pts[3]), color=rgb_col,
                  linewidth=2.5 if rank == 1 else 1.5)
        ax.plot3D(*zip(pts[4], pts[5]), color=rgb_col,
                  linewidth=2.5 if rank == 1 else 1.5)
        mid = (pts[4] + pts[5]) / 2
        ax.plot3D(*zip(mid, pts[6]), color=rgb_col, linewidth=1.5)

        # Draw approach axis
        R = to_rotation_matrix(rot)
        binormal = R[:, 2]  # z-axis = approach direction
        p = np.array(pos)
        ax.plot3D(*zip(p, p - binormal * 0.03), color=rgb_col, linewidth=1.0)

        # Label
        ax.text(pos[0], pos[1], pos[2] - 0.01,
                f"#{rank}", color=rgb_col, fontsize=8, fontweight='bold')

    # ── Styling ──────────────────────────────────────────────────────
    ax.set_xlabel("X (m)", color="white", fontsize=10)
    ax.set_ylabel("Y (m)", color="white", fontsize=10)
    ax.set_zlabel("Z (m)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=7)

    if len(target_points) > 0:
        center = target_points.mean(axis=0)
        radius = max(np.ptp(target_points, axis=0).max() * 1.5, 0.15)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_title(sample_id, color="white", fontsize=13, pad=15, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, facecolor="#2a2a4e",
              edgecolor="#444", labelcolor="white")
    ax.view_init(elev=-60, azim=-90)

    plt.tight_layout()

    if output_dir is None:
        output_dir = config.PROJECT_ROOT / "vis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sample_id}_3d.png"
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[VIS-3D] Saved → {out_path}")

    return out_path


def _draw_arrow_3d(ax, start, end, colour, linewidth=1.0):
    """Draw a 3D arrow from start to end."""
    ax.plot3D(*zip(start, end), color=colour, linewidth=linewidth)


# ── Open3D Backend (interactive) ────────────────────────────────────

def visualise_3d_open3d(
    scene_points: np.ndarray,
    target_points: np.ndarray,
    selections: List[Dict],
    sample_id: str,
    scene_colours: Optional[np.ndarray] = None,
):
    """Interactive 3D visualisation using Open3D.

    Requires: pip install open3d
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] open3d not installed. Use `pip install open3d`.")
        print("       Falling back to matplotlib backend.")
        return None

    geometries = []

    pcd_scene = o3d.geometry.PointCloud()
    pcd_scene.points = o3d.utility.Vector3dVector(scene_points)
    if scene_colours is not None:
        pcd_scene.colors = o3d.utility.Vector3dVector(scene_colours)
    else:
        pcd_scene.paint_uniform_color([0.6, 0.6, 0.6])
    geometries.append(pcd_scene)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target_points)
    pcd_target.paint_uniform_color([0.0, 1.0, 0.5])
    geometries.append(pcd_target)

    for sel in selections:
        rank = sel["rank"]
        pts = transform_gripper(sel["position"], sel["rotation"], sel["width"])

        bgr = rank_colour(rank)
        rgb_col = [bgr[2] / 255, bgr[1] / 255, bgr[0] / 255]

        lines = [[0, 1], [2, 3], [4, 5]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts[:6])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([rgb_col] * 3)
        geometries.append(line_set)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere.translate(np.array(sel["position"]))
        sphere.paint_uniform_color(rgb_col)
        geometries.append(sphere)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord_frame)

    print(f"[VIS-3D] Open3D viewer for {sample_id}")
    print("  Controls: left-drag=rotate, scroll=zoom, right-drag=pan")
    o3d.visualization.draw_geometries(
        geometries, window_name=f"VLMGraspPose — {sample_id}",
        width=1200, height=800,
    )


# ── Main Entry Point ────────────────────────────────────────────────

def visualise_sample_3d(
    sample_id: str,
    grounder: str = "phrase",
    reranker: str = "rule",
    output_dir: Path = None,
    backend: str = "matplotlib",
) -> Optional[Path]:
    """Load data and create 3D visualisation for a pipeline sample."""
    # Parse sample_id
    parts = sample_id.split("_")
    scene_id = int(parts[1])
    camera = parts[2]
    frame_id = int(parts[3])

    scene_dir = config.SCENES_DIR / f"scene_{scene_id:04d}"
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")

    factor = get_factor_depth(scene_dir, camera)
    rgb = load_rgb(scene_dir, frame_id, camera)
    depth = load_depth(scene_dir, frame_id, camera, factor)
    K = load_camera_intrinsics(scene_dir, camera)

    scene_points, scene_px = backproject_depth(depth, K)

    # Find prediction
    pred_files = glob.glob(
        str(config.RESULTS_DIR / f"predictions_*_{grounder}_{reranker}.json")
    )

    pred = None
    for pf in pred_files:
        with open(pf) as f:
            preds = json.load(f)
        for p in preds:
            if p.get("sample_id") == sample_id:
                pred = p
                break
        if pred:
            break

    selections = pred.get("ranked_grasps", []) if pred else []

    # Target points from predicted bbox
    target_points = np.zeros((0, 3))
    if pred and pred.get("pred_bbox"):
        target_points, _ = crop_point_cloud_by_bbox(
            scene_points, scene_px, pred["pred_bbox"],
        )

    # Note: Don't downsample scene_points separately from scene_px —
    # the indices would go out of sync (BUG-3 fix).  The matplotlib
    # backend already subsamples internally via max_scene_pts.

    if backend == "open3d":
        scene_viz = voxel_downsample(scene_points, 0.005)
        visualise_3d_open3d(scene_viz, target_points, selections, sample_id)
        return None
    else:
        return visualise_3d_matplotlib(
            scene_points, target_points, selections, sample_id,
            output_dir=output_dir,
            scene_rgb=None,
            scene_pixel_coords=scene_px,
            rgb_image=rgb,
        )


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3D Grasp Visualisation")
    parser.add_argument("--sample", type=str, required=True)
    parser.add_argument("--grounder", type=str, default="phrase")
    parser.add_argument("--reranker", type=str, default="rule")
    parser.add_argument("--backend", type=str, default="matplotlib",
                        choices=["matplotlib", "open3d"])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    visualise_sample_3d(
        sample_id=args.sample,
        grounder=args.grounder,
        reranker=args.reranker,
        output_dir=out_dir,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
