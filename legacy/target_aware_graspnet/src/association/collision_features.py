from __future__ import annotations

import numpy as np

from utils.geometry import distance_to_plane


def collision_penalty(candidate, pcr) -> float:
    penalties = []
    if pcr.table_plane is not None:
        d = abs(float(distance_to_plane(candidate.position.reshape(1, 3), pcr.table_plane)[0]))
        penalties.append(float(np.clip((0.02 - d) / 0.02, 0.0, 1.0)))
    scene = np.asarray(pcr.scene_pcd.points) if pcr.scene_pcd is not None else np.zeros((0, 3))
    if len(scene):
        if pcr.target_aabb is not None:
            min_bound = np.asarray(pcr.target_aabb.get_min_bound()) - 0.01
            max_bound = np.asarray(pcr.target_aabb.get_max_bound()) + 0.01
            inside_target = np.all((scene >= min_bound) & (scene <= max_bound), axis=1)
            scene = scene[~inside_target]
        dists = np.linalg.norm(scene - candidate.position.reshape(1, 3), axis=1)
        dense = np.mean(dists < max(candidate.gripper_width * 0.25, 0.01))
        penalties.append(float(np.clip(dense * 20.0, 0.0, 1.0)))
    return max(penalties) if penalties else 0.0
