import inspect
import unittest

import config
from scripts import download_weights, step06_grasp_candidates, step09_train_reranker, step10_inference
from src.grasp_detector import RGBDGeometricGraspSampler


class SelectedStackDefaultTests(unittest.TestCase):
    def test_config_targets_florence_base_geometric_mlp_stack(self):
        self.assertEqual(config.FLORENCE2_MODEL_ID, "microsoft/Florence-2-base-ft")
        self.assertEqual(config.DEFAULT_GROUNDING, "seg")
        self.assertEqual(config.DEFAULT_DETECTOR, "geometric")
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

    def test_step06_default_detector_is_rgbd_geometric_sampler(self):
        detector = step06_grasp_candidates._create_detector("geometric", top_k=8)
        self.assertIsInstance(detector, RGBDGeometricGraspSampler)


if __name__ == "__main__":
    unittest.main()
