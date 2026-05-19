import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import scripts.step07_build_labels as step07
import scripts.step08_extract_features as step08
from src.grasp_detector import GraspCandidate


def _candidate(candidate_id: int) -> GraspCandidate:
    return GraspCandidate(
        candidate_id=candidate_id,
        position=[0.0, 0.0, 1.0],
        rotation=np.eye(3, dtype=np.float32).flatten().tolist(),
        width=0.05,
        detector_score=0.9,
        source="geometric",
    )


class TrainingViewCacheTests(unittest.TestCase):
    def test_step07_reuses_view_level_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries_dir = root / "queries"
            oracle_dir = root / "oracle"
            pcd_dir = root / "pcd"
            labels_dir = root / "labels"
            scenes_dir = root / "scenes"
            for d in [queries_dir, oracle_dir, pcd_dir, labels_dir, scenes_dir]:
                d.mkdir(parents=True, exist_ok=True)

            query_records = [
                {
                    "sample_id": "sample_a",
                    "view_sample_id": "view_0001",
                    "scene_id": 0,
                    "camera": "realsense",
                    "frame_id": 0,
                    "target_object_id": 0,
                },
                {
                    "sample_id": "sample_b",
                    "view_sample_id": "view_0001",
                    "scene_id": 0,
                    "camera": "realsense",
                    "frame_id": 0,
                    "target_object_id": 1,
                },
            ]
            oracle_records = [
                {"sample_id": "sample_a", "gt_mask_val": 1},
                {"sample_id": "sample_b", "gt_mask_val": 2},
            ]

            (queries_dir / "train_queries.jsonl").write_text(
                "\n".join(json.dumps(r) for r in query_records) + "\n"
            )
            (oracle_dir / "train_oracle.jsonl").write_text(
                "\n".join(json.dumps(r) for r in oracle_records) + "\n"
            )
            np.savez(
                pcd_dir / "view_0001.npz",
                points=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                pixel_coords=np.array([[0, 0]], dtype=np.int32),
            )

            counts = {"candidates": 0, "pcd": 0, "label": 0, "associate": 0}
            real_np_load = step07.np.load

            def counted_candidates(*args, **kwargs):
                counts["candidates"] += 1
                return [_candidate(0)]

            def counted_np_load(*args, **kwargs):
                counts["pcd"] += 1
                return real_np_load(*args, **kwargs)

            def counted_label(*args, **kwargs):
                counts["label"] += 1
                return np.array([[1]], dtype=np.uint8)

            def counted_associate(*args, **kwargs):
                counts["associate"] += 1
                return 1

            with (
                mock.patch.object(step07.config, "QUERIES_DIR", queries_dir),
                mock.patch.object(step07.config, "ORACLE_TARGETS_DIR", oracle_dir),
                mock.patch.object(step07.config, "POINTCLOUDS_DIR", pcd_dir),
                mock.patch.object(step07.config, "RANK_LABELS_DIR", labels_dir),
                mock.patch.object(step07.config, "SCENES_DIR", scenes_dir),
                mock.patch.object(step07, "load_grasp_candidates", side_effect=counted_candidates),
                mock.patch.object(step07.np, "load", side_effect=counted_np_load),
                mock.patch.object(step07, "load_label", side_effect=counted_label),
                mock.patch.object(step07, "associate_grasp_to_object", side_effect=counted_associate),
            ):
                step07.build_labels(splits=["train"], detector="geometric")

            self.assertEqual(counts["candidates"], 2)
            self.assertEqual(counts["pcd"], 1)
            self.assertEqual(counts["label"], 1)
            self.assertEqual(counts["associate"], 2)
            self.assertTrue((labels_dir / "train_geometric_labels.parquet").exists())

    def test_step08_reuses_view_level_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queries_dir = root / "queries"
            oracle_dir = root / "oracle"
            pcd_dir = root / "pcd"
            features_dir = root / "features"
            scenes_dir = root / "scenes"
            for d in [queries_dir, oracle_dir, pcd_dir, features_dir, scenes_dir]:
                d.mkdir(parents=True, exist_ok=True)

            query_records = [
                {
                    "sample_id": "sample_a",
                    "view_sample_id": "view_0001",
                    "scene_id": 0,
                    "camera": "realsense",
                    "frame_id": 0,
                    "target_object_id": 0,
                },
                {
                    "sample_id": "sample_b",
                    "view_sample_id": "view_0001",
                    "scene_id": 0,
                    "camera": "realsense",
                    "frame_id": 0,
                    "target_object_id": 1,
                },
            ]
            oracle_records = [
                {"sample_id": "sample_a", "gt_bbox": [0, 0, 0, 0], "gt_mask_val": 1},
                {"sample_id": "sample_b", "gt_bbox": [0, 0, 0, 0], "gt_mask_val": 2},
            ]

            (queries_dir / "train_queries.jsonl").write_text(
                "\n".join(json.dumps(r) for r in query_records) + "\n"
            )
            (oracle_dir / "train_oracle.jsonl").write_text(
                "\n".join(json.dumps(r) for r in oracle_records) + "\n"
            )
            np.savez(
                pcd_dir / "view_0001.npz",
                points=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
                pixel_coords=np.array([[0, 0], [1, 0]], dtype=np.int32),
            )

            counts = {"candidates": 0, "pcd": 0, "label": 0, "depth": 0, "intrinsics": 0, "factor": 0}
            real_np_load = step08.np.load

            def counted_candidates(*args, **kwargs):
                counts["candidates"] += 1
                return [_candidate(0)]

            def counted_np_load(*args, **kwargs):
                counts["pcd"] += 1
                return real_np_load(*args, **kwargs)

            def counted_label(*args, **kwargs):
                counts["label"] += 1
                return np.array([[1, 2]], dtype=np.uint8)

            def counted_depth(*args, **kwargs):
                counts["depth"] += 1
                return np.array([[1.0, 1.0]], dtype=np.float32)

            def counted_intrinsics(*args, **kwargs):
                counts["intrinsics"] += 1
                return np.eye(3, dtype=np.float32)

            def counted_factor(*args, **kwargs):
                counts["factor"] += 1
                return 1000.0

            fake_extractor = mock.Mock()
            fake_extractor.extract_batch.side_effect = lambda **kwargs: np.zeros(
                (len(kwargs["candidates"]), len(step08.config.FEATURE_NAMES)), dtype=np.float32
            )

            with (
                mock.patch.object(step08.config, "QUERIES_DIR", queries_dir),
                mock.patch.object(step08.config, "ORACLE_TARGETS_DIR", oracle_dir),
                mock.patch.object(step08.config, "POINTCLOUDS_DIR", pcd_dir),
                mock.patch.object(step08.config, "RANK_FEATURES_DIR", features_dir),
                mock.patch.object(step08.config, "SCENES_DIR", scenes_dir),
                mock.patch.object(step08, "load_grasp_candidates", side_effect=counted_candidates),
                mock.patch.object(step08.np, "load", side_effect=counted_np_load),
                mock.patch.object(step08, "load_label", side_effect=counted_label),
                mock.patch.object(step08, "load_depth", side_effect=counted_depth),
                mock.patch.object(step08, "load_camera_intrinsics", side_effect=counted_intrinsics),
                mock.patch.object(step08, "get_factor_depth", side_effect=counted_factor),
                mock.patch.object(step08, "FeatureExtractor", return_value=fake_extractor),
            ):
                step08.extract_features(splits=["train"], grounding="oracle", detector="geometric")

            self.assertEqual(counts["candidates"], 2)
            self.assertEqual(counts["pcd"], 1)
            self.assertEqual(counts["label"], 1)
            self.assertEqual(counts["depth"], 1)
            self.assertEqual(counts["intrinsics"], 1)
            self.assertEqual(counts["factor"], 1)
            self.assertTrue((features_dir / "train_oracle_geometric_features.parquet").exists())


if __name__ == "__main__":
    unittest.main()
