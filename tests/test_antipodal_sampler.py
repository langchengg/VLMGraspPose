import unittest
from unittest import mock

import numpy as np

from src.grasp_detector import AntipodalSampler


class AntipodalSamplerDownsampleTests(unittest.TestCase):
    def test_downsamples_before_estimating_normals_when_cloud_exceeds_limit(self):
        cloud = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        sampler = AntipodalSampler(max_points_for_sampling=20, num_contact_samples=5)
        seen = {}

        def fake_normals(points, k):
            seen["num_points"] = len(points)
            return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1))

        with mock.patch("src.point_cloud.estimate_normals_pca", side_effect=fake_normals):
            sampler.detect(cloud, top_k=5)

        self.assertEqual(seen["num_points"], 20)

    def test_keeps_small_clouds_unchanged(self):
        cloud = np.random.RandomState(1).rand(12, 3).astype(np.float32)
        sampler = AntipodalSampler(max_points_for_sampling=20, num_contact_samples=5)
        seen = {}

        def fake_normals(points, k):
            seen["num_points"] = len(points)
            return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(points), 1))

        with mock.patch("src.point_cloud.estimate_normals_pca", side_effect=fake_normals):
            sampler.detect(cloud, top_k=5)

        self.assertEqual(seen["num_points"], len(cloud))


if __name__ == "__main__":
    unittest.main()
