import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts import step10_inference
from src.grasp_detector import GraspCandidate


class Step10InferenceTests(unittest.TestCase):
    def test_group_predictions_by_view_collects_best_grasps_per_object(self):
        predictions = [
            {
                "sample_id": "viewA_001_mug",
                "view_sample_id": "viewA",
                "scene_id": 100,
                "camera": "realsense",
                "frame_id": 0,
                "target_object_id": 1,
                "target_class": "mug",
                "text_query": "pick the mug",
                "split": "test_seen",
                "grounder": "seg",
                "reranker": "mlp",
                "detector": "geometric",
                "best_grasp": {"candidate_id": 2},
                "failure_reason": None,
            },
            {
                "sample_id": "viewA_002_bottle",
                "view_sample_id": "viewA",
                "scene_id": 100,
                "camera": "realsense",
                "frame_id": 0,
                "target_object_id": 2,
                "target_class": "bottle",
                "text_query": "pick the bottle",
                "split": "test_seen",
                "grounder": "seg",
                "reranker": "mlp",
                "detector": "geometric",
                "best_grasp": {"candidate_id": 4},
                "failure_reason": None,
            },
            {
                "sample_id": "viewB_003_apple",
                "view_sample_id": "viewB",
                "scene_id": 101,
                "camera": "realsense",
                "frame_id": 16,
                "target_object_id": 3,
                "target_class": "apple",
                "text_query": "pick the apple",
                "split": "test_seen",
                "grounder": "seg",
                "reranker": "mlp",
                "detector": "geometric",
                "best_grasp": {"candidate_id": 1},
                "failure_reason": None,
            },
        ]

        grouped = step10_inference.group_predictions_by_view(predictions)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["view_sample_id"], "viewA")
        self.assertEqual(len(grouped[0]["objects"]), 2)
        self.assertEqual(
            [obj["target_object_id"] for obj in grouped[0]["objects"]],
            [1, 2],
        )
        self.assertEqual(grouped[0]["objects"][0]["best_grasp"], {"candidate_id": 2})

    def test_run_inference_reuses_cached_view_context_for_multiple_queries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queries_dir = root / "queries"
            results_dir = root / "results"
            scenes_dir = root / "scenes"
            queries_dir.mkdir(parents=True)
            results_dir.mkdir(parents=True)
            scenes_dir.mkdir(parents=True)

            queries_path = queries_dir / "test_seen_queries.jsonl"
            queries = [
                {
                    "sample_id": "scene_0100_realsense_0000_000_mug",
                    "view_sample_id": "scene_0100_realsense_0000",
                    "scene_id": 100,
                    "camera": "realsense",
                    "frame_id": 0,
                    "target_object_id": 0,
                    "object_name": "mug",
                    "text_query": "pick the mug",
                    "split": "test_seen",
                },
                {
                    "sample_id": "scene_0100_realsense_0000_001_bottle",
                    "view_sample_id": "scene_0100_realsense_0000",
                    "scene_id": 100,
                    "camera": "realsense",
                    "frame_id": 0,
                    "target_object_id": 1,
                    "object_name": "bottle",
                    "text_query": "pick the bottle",
                    "split": "test_seen",
                },
            ]
            with open(queries_path, "w") as f:
                for rec in queries:
                    f.write(json.dumps(rec) + "\n")

            candidate = GraspCandidate(
                candidate_id=0,
                position=[0.0, 0.0, 0.5],
                rotation=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                width=0.04,
                detector_score=0.9,
                source="geometric",
            )
            fake_grounding = SimpleNamespace(
                bbox=[0, 0, 4, 4],
                mask=None,
                confidence=1.0,
            )

            class FakeGrounder:
                def _ensure_loaded(self):
                    return None

                def ground(self, rgb, text_query, **kwargs):
                    return fake_grounding

            class FakeExtractor:
                def extract_batch(self, **kwargs):
                    return np.zeros((1, len(step10_inference.config.FEATURE_NAMES)), dtype=np.float32)

            class FakeReranker:
                def select_top_k(self, features, candidates, k=5):
                    return [{
                        "rank": 1,
                        "candidate_id": 0,
                        "position": candidates[0].position,
                        "rotation": candidates[0].rotation,
                        "width": candidates[0].width,
                        "initial_geometric_score": candidates[0].detector_score,
                        "final_score": 0.95,
                        "approach_vector": candidates[0].approach_vector,
                        "closing_direction": candidates[0].closing_direction,
                        "grasp_type": candidates[0].grasp_type,
                    }]

            points = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
            pixel_coords = np.array([[1, 1]], dtype=np.int32)
            depth = np.ones((5, 5), dtype=np.float32)
            intrinsics = np.array(
                [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            rgb = np.zeros((5, 5, 3), dtype=np.uint8)
            label = np.ones((5, 5), dtype=np.uint8)

            with mock.patch.object(step10_inference.config, "QUERIES_DIR", queries_dir), \
                 mock.patch.object(step10_inference.config, "ORACLE_TARGETS_DIR", root / "oracle"), \
                 mock.patch.object(step10_inference.config, "RESULTS_DIR", results_dir), \
                 mock.patch.object(step10_inference.config, "POINTCLOUDS_DIR", root / "pointclouds"), \
                 mock.patch.object(step10_inference.config, "SCENES_DIR", scenes_dir), \
                 mock.patch.object(step10_inference.config, "TARGET_MIN_POINTS", 1), \
                 mock.patch.object(step10_inference, "get_grounder", return_value=FakeGrounder()), \
                 mock.patch.object(step10_inference, "get_reranker", return_value=FakeReranker()), \
                 mock.patch.object(step10_inference, "FeatureExtractor", return_value=FakeExtractor()), \
                 mock.patch.object(step10_inference, "load_rgb", return_value=rgb) as load_rgb, \
                 mock.patch.object(step10_inference, "get_factor_depth", return_value=1000.0) as get_factor_depth, \
                 mock.patch.object(step10_inference, "load_depth", return_value=depth) as load_depth, \
                 mock.patch.object(step10_inference, "load_camera_intrinsics", return_value=intrinsics) as load_intrinsics, \
                 mock.patch.object(step10_inference, "load_label", return_value=label) as load_label, \
                 mock.patch.object(step10_inference, "load_grasp_candidates", return_value=[candidate]) as load_candidates, \
                 mock.patch.object(step10_inference, "backproject_depth", return_value=(points, pixel_coords)) as backproject_depth, \
                 mock.patch.object(step10_inference, "associate_grasp_to_object", return_value=1):
                step10_inference.run_inference(
                    splits=["test_seen"],
                    grounder_name="seg",
                    reranker_name="rule",
                    detector="geometric",
                )

            self.assertEqual(load_rgb.call_count, 1)
            self.assertEqual(get_factor_depth.call_count, 1)
            self.assertEqual(load_depth.call_count, 1)
            self.assertEqual(load_intrinsics.call_count, 1)
            self.assertEqual(load_label.call_count, 1)
            self.assertEqual(load_candidates.call_count, 2)
            self.assertEqual(backproject_depth.call_count, 1)

            grouped_path = (
                results_dir
                / "top1_by_view_test_seen_seg_rule_geometric.json"
            )
            self.assertTrue(grouped_path.exists())

            with open(grouped_path) as f:
                grouped = json.load(f)

            self.assertEqual(len(grouped), 1)
            self.assertEqual(grouped[0]["view_sample_id"], "scene_0100_realsense_0000")
            self.assertEqual(len(grouped[0]["objects"]), 2)


if __name__ == "__main__":
    unittest.main()
