import inspect
import unittest

import numpy as np

import config
from scripts import download_weights, step06_grasp_candidates, step09_train_reranker, step10_inference
from src.grasp_detector import GraspNetDetector


class SelectedStackDefaultTests(unittest.TestCase):
    def test_config_targets_florence_base_graspnet_mlp_stack(self):
        self.assertEqual(config.FLORENCE2_MODEL_ID, "microsoft/Florence-2-base-ft")
        self.assertEqual(config.DEFAULT_GROUNDING, "seg")
        self.assertEqual(config.DEFAULT_DETECTOR, "graspnet")
        self.assertEqual(config.DEFAULT_RERANKER, "mlp")

    def test_weight_downloader_uses_same_florence_base_model(self):
        self.assertEqual(download_weights.FLORENCE2_MODEL_ID, config.FLORENCE2_MODEL_ID)

    def test_pipeline_function_defaults_match_selected_stack(self):
        self.assertEqual(
            inspect.signature(step06_grasp_candidates.generate_candidates)
            .parameters["detector_type"]
            .default,
            config.DEFAULT_DETECTOR,
        )
        self.assertEqual(
            inspect.signature(step09_train_reranker.train_reranker)
            .parameters["model_name"]
            .default,
            config.DEFAULT_RERANKER,
        )
        run_sig = inspect.signature(step10_inference.run_inference)
        self.assertEqual(run_sig.parameters["grounder_name"].default, config.DEFAULT_GROUNDING)
        self.assertEqual(run_sig.parameters["reranker_name"].default, config.DEFAULT_RERANKER)
        self.assertEqual(run_sig.parameters["detector"].default, config.DEFAULT_DETECTOR)

    def test_graspnet_sampler_returns_fixed_size_point_batch(self):
        detector = GraspNetDetector(num_point=8)
        point_cloud = np.arange(15, dtype=np.float32).reshape(5, 3)
        colors = np.ones((5, 3), dtype=np.float32)

        sampled_points, sampled_colors = detector._sample_points_for_network(
            point_cloud,
            colors,
            rng=np.random.RandomState(0),
        )

        self.assertEqual(sampled_points.shape, (8, 3))
        self.assertEqual(sampled_colors.shape, (8, 3))
        self.assertTrue(np.all(sampled_points[:5] == point_cloud))
        self.assertTrue(np.all(sampled_colors[:5] == colors))


if __name__ == "__main__":
    unittest.main()
