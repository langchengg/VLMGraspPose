import unittest
from unittest import mock

import numpy as np

from src.feature_extractor import FeatureExtractor
from src.grasp_detector import GraspCandidate


class FeatureExtractorSamplingTests(unittest.TestCase):
    def test_downsamples_scene_points_before_per_candidate_features(self):
        extractor = FeatureExtractor(max_scene_points=20)
        candidate = GraspCandidate(
            candidate_id=0,
            position=[0.0, 0.0, 1.0],
            rotation=np.eye(3, dtype=np.float32).flatten().tolist(),
            width=0.05,
            detector_score=0.9,
            source="antipodal",
        )

        scene_points = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        scene_pixel_coords = np.random.RandomState(1).randint(0, 10, size=(100, 2), dtype=np.int32)
        target_points = np.random.RandomState(2).rand(10, 3).astype(np.float32)
        depth = np.ones((10, 10), dtype=np.float32)
        intrinsics = np.eye(3, dtype=np.float32)
        seen = {}

        def fake_compute_single(c, **kwargs):
            seen["num_scene_points"] = len(kwargs["scene_points"])
            seen["num_pixel_coords"] = len(kwargs["scene_pixel_coords"])
            return [0.0] * 9

        with mock.patch.object(extractor, "_compute_single", side_effect=fake_compute_single):
            features = extractor.extract_batch(
                candidates=[candidate],
                target_bbox=[0, 0, 5, 5],
                target_mask=None,
                target_points=target_points,
                scene_points=scene_points,
                scene_pixel_coords=scene_pixel_coords,
                florence_conf=1.0,
                depth=depth,
                intrinsics=intrinsics,
            )

        self.assertEqual(features.shape, (1, 9))
        self.assertEqual(seen["num_scene_points"], 20)
        self.assertEqual(seen["num_pixel_coords"], 20)


if __name__ == "__main__":
    unittest.main()
