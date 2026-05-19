from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dataset.split_manager import SplitManager
from utils.data_types import GraspNetSample


@dataclass
class PathTemplates:
    rgb: str = "scenes/{scene_id}/{camera}/rgb/{frame_id}.png"
    depth: str = "scenes/{scene_id}/{camera}/depth/{frame_id}.png"
    label: str = "scenes/{scene_id}/{camera}/label/{frame_id}.png"
    annotation: str = "scenes/{scene_id}/{camera}/annotations/{frame_id}.xml"
    camera_intrinsic: str = "scenes/{scene_id}/{camera}/camK.npy"

    @classmethod
    def from_config(cls, cfg: dict | None) -> "PathTemplates":
        allowed = {"rgb", "depth", "label", "annotation", "camera_intrinsic"}
        return cls(**{key: value for key, value in (cfg or {}).items() if key in allowed})

    def path(self, root: Path, key: str, scene_id: str, camera: str, frame_id: str) -> Path:
        template = getattr(self, key)
        return root / template.format(scene_id=scene_id, camera=camera, frame_id=frame_id)


class SampleIndexBuilder:
    def __init__(self, path_templates: PathTemplates | None = None):
        self.templates = path_templates or PathTemplates()
        self.split_manager = SplitManager()

    def build_for_scene(
        self,
        scene_id: str,
        camera: str,
        dataset_root: Path,
        output_root: Path,
        split: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> list[GraspNetSample]:
        dataset_root = Path(dataset_root)
        split = split or self.split_manager.scene_to_split(scene_id)
        rgb_dir = dataset_root / "scenes" / scene_id / camera / "rgb"
        depth_dir = dataset_root / "scenes" / scene_id / camera / "depth"
        rgb_files = sorted(rgb_dir.glob("*.png"))
        depth_frames = {p.stem for p in depth_dir.glob("*.png")}
        if max_frames is not None:
            rgb_files = rgb_files[:max_frames]

        samples = []
        for rgb_path in rgb_files:
            frame_id = rgb_path.stem
            if frame_id not in depth_frames:
                continue
            depth_path = self.templates.path(dataset_root, "depth", scene_id, camera, frame_id)
            annotation_path = self.templates.path(dataset_root, "annotation", scene_id, camera, frame_id)
            label_path = self.templates.path(dataset_root, "label", scene_id, camera, frame_id)
            intrinsic_path = self.templates.path(dataset_root, "camera_intrinsic", scene_id, camera, frame_id)
            sample_out = Path(output_root) / split / scene_id / camera / frame_id
            samples.append(GraspNetSample(
                split=split,
                scene_id=scene_id,
                camera=camera,
                frame_id=frame_id,
                rgb_path=rgb_path,
                depth_path=depth_path,
                annotation_path=annotation_path if annotation_path.exists() else None,
                camera_intrinsic_path=intrinsic_path if intrinsic_path.exists() else None,
                output_dir=sample_out,
                label_path=label_path if label_path.exists() else None,
            ))
        return samples

    def build_for_split(
        self,
        split_name: str,
        camera: str,
        dataset_root: Path,
        output_root: Path,
        max_scenes: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> list[GraspNetSample]:
        scene_ids = self.split_manager.get_scene_ids(split_name)
        if max_scenes is not None:
            scene_ids = scene_ids[:max_scenes]
        samples: list[GraspNetSample] = []
        for scene_id in scene_ids:
            samples.extend(self.build_for_scene(
                scene_id=scene_id,
                camera=camera,
                dataset_root=dataset_root,
                output_root=output_root,
                split=split_name,
                max_frames=max_frames,
            ))
        return samples

    def dry_run(
        self,
        split_name: str,
        camera: str,
        dataset_root: Path,
        max_scenes: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> dict:
        samples = self.build_for_split(
            split_name,
            camera,
            dataset_root,
            output_root=Path("_dry_run"),
            max_scenes=max_scenes,
            max_frames=max_frames,
        )
        return {
            "split": split_name,
            "camera": camera,
            "num_samples": len(samples),
            "num_scenes": len({s.scene_id for s in samples}),
        }
