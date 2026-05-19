"""
vis/grasp_drawing.py — Core drawing primitives for grasp visualisation
=======================================================================
Low-level helpers to:
  • Draw 2D gripper footprints on RGB images
  • Convert rotation representations to 3×3 matrices
  • Build 3D gripper meshes for Open3D / Matplotlib

The pipeline stores rotation as flattened 3×3: [r11..r33] (9 floats).
Legacy quaternion [qx, qy, qz, qw] support is kept for backward compat.
"""

import numpy as np
from typing import List, Tuple, Optional


# ── Rotation Helpers ────────────────────────────────────────────────

def quat_to_rotation_matrix(q: List[float]) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] → 3×3 rotation matrix."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2),  2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),      1 - 2*(qx**2 + qy**2)],
    ])
    return R


def to_rotation_matrix(rot: List[float]) -> np.ndarray:
    """Convert any rotation list to 3×3.

    Auto-detects format:
      - len == 9  → flattened 3×3 (pipeline default)
      - len == 4  → quaternion [qx, qy, qz, qw]
    """
    if len(rot) == 9:
        return np.array(rot).reshape(3, 3)
    elif len(rot) == 4:
        return quat_to_rotation_matrix(rot)
    else:
        raise ValueError(f"Expected 4 or 9 elements for rotation, got {len(rot)}")


# ── Gripper Keypoints (in gripper-local frame) ──────────────────────

def gripper_keypoints(width: float, depth: float = 0.04,
                      finger_length: float = 0.03) -> np.ndarray:
    """Return 8 keypoints defining a parallel-jaw gripper shape.

    The gripper local frame:
      x-axis = closing direction (between finger tips)
      y-axis = up (palm direction)
      z-axis = approach direction (pointing towards the object)

    Returns (8, 3) array:
      0──1  left finger (base─tip)
      2──3  right finger (tip─base)
      4──5  palm bar left─right
      6     wrist centre
      7     approach point (behind wrist)
    """
    hw = width / 2.0
    fl = finger_length

    pts = np.array([
        # Left finger
        [-hw, 0,  0],       # 0: base
        [-hw, 0, -fl],      # 1: tip
        # Right finger
        [ hw, 0, -fl],      # 2: tip
        [ hw, 0,  0],       # 3: base
        # Palm bar
        [-hw, 0,  0],       # 4: left
        [ hw, 0,  0],       # 5: right
        # Wrist
        [ 0,  0,  depth],   # 6: wrist
        [ 0,  0,  depth*2], # 7: approach
    ], dtype=np.float64)

    return pts


def transform_gripper(position: List[float], rotation: List[float],
                      width: float) -> np.ndarray:
    """Get gripper keypoints transformed to camera frame.

    Parameters
    ----------
    position : [x, y, z] in camera frame
    rotation : flattened 3×3 [r11..r33] or quaternion [qx,qy,qz,qw]
    width : gripper opening width in metres

    Returns
    -------
    (8, 3) keypoints in camera frame
    """
    R = to_rotation_matrix(rotation)
    pts_local = gripper_keypoints(width)
    pts_world = (R @ pts_local.T).T + np.array(position)
    return pts_world


# ── Project Gripper to Image ────────────────────────────────────────

def project_gripper_to_image(
    position: List[float],
    rotation: List[float],
    width: float,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project gripper keypoints onto the image plane.

    Accepts both 9-element (flat rotmat) and 4-element (quaternion) rotation.
    Returns (8, 2) pixel coordinates (u, v).
    """
    pts_3d = transform_gripper(position, rotation, width)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = np.clip(pts_3d[:, 2], 1e-6, None)
    u = pts_3d[:, 0] * fx / z + cx
    v = pts_3d[:, 1] * fy / z + cy

    return np.stack([u, v], axis=-1)


# ── Colour Maps ─────────────────────────────────────────────────────

def score_to_colour(score: float, alpha: int = 200) -> Tuple[int, int, int, int]:
    """Map a score ∈ [0, 1] to a colour (BGRA for OpenCV).

    Low score → red,  High score → green.
    """
    r = int(255 * (1.0 - score))
    g = int(255 * score)
    b = 40
    return (b, g, r, alpha)


def rank_colour(rank: int) -> Tuple[int, int, int]:
    """Return a distinct BGR colour for ranks 1–5+."""
    palette = [
        (0, 255, 0),     # 1: green
        (0, 200, 255),   # 2: yellow
        (0, 140, 255),   # 3: orange
        (255, 100, 0),   # 4: cyan-blue
        (255, 0, 100),   # 5: magenta
    ]
    if rank <= len(palette):
        return palette[rank - 1]
    return (128, 128, 128)
