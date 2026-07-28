"""Integration checks for the pinned official sampler and target filtering."""

from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = REPO_ROOT / "third_party" / "gqcnn-official"
sys.path.insert(0, str(OFFICIAL_ROOT))

from autolab_core import CameraIntrinsics, YamlConfig  # noqa: E402

from src.grasping.dexnet_adapter import (  # noqa: E402
    make_rgbd_and_segmask,
    sample_antipodal_grasps,
)
from src.grasping.dexnet_candidate_generator import (  # noqa: E402
    deduplicate_candidates,
    generate_candidates,
    validate_candidate,
)
from src.grasping.dexnet_scoring import (  # noqa: E402
    GQCNNScoringUnavailable,
    score_fixed_candidates,
)
from src.grasping.ocid_vlg_grasp_adapter import OcidVlgBundleIndex  # noqa: E402
from scripts.run_hifics_dexnet_candidates import derive_sample_seed  # noqa: E402


class OfficialSamplerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sampling = dict(
            YamlConfig(str(REPO_ROOT / "configs" / "dexnet_candidates.yaml"))[
                "sampling"
            ]
        )

    def test_official_bundled_example_returns_multiple_deterministic_grasps(self) -> None:
        example = OFFICIAL_ROOT / "data" / "examples" / "single_object" / "primesense"
        rgb = np.asarray(Image.open(example / "color_0.png").convert("RGB"), dtype=np.uint8)
        depth = np.asarray(
            np.load(example / "depth_0.npy", allow_pickle=False), dtype=np.float32
        ).squeeze()
        mask = np.asarray(Image.open(example / "segmask_0.png")) > 0
        intrinsics = CameraIntrinsics.load(
            str(OFFICIAL_ROOT / "data" / "calib" / "primesense" / "primesense.intr")
        )
        rgbd, segmask = make_rgbd_and_segmask(rgb, depth, mask, frame=intrinsics.frame)

        first = sample_antipodal_grasps(
            rgbd, intrinsics, segmask, self.sampling, num_samples=32, seed=42
        )
        second = sample_antipodal_grasps(
            rgbd, intrinsics, segmask, self.sampling, num_samples=32, seed=42
        )
        self.assertGreaterEqual(len(first), 2)
        self.assertEqual(len(first), len(second))
        first_values = np.asarray(
            [[*grasp.center.data, grasp.depth, grasp.angle, grasp.width] for grasp in first]
        )
        second_values = np.asarray(
            [[*grasp.center.data, grasp.depth, grasp.angle, grasp.width] for grasp in second]
        )
        np.testing.assert_array_equal(first_values, second_values)

    def test_real_hifics_sample_is_deterministic_and_target_filtered(self) -> None:
        run = REPO_ROOT / "runs" / "hifics_ocidvlg_20260711_112921"
        index = OcidVlgBundleIndex(
            REPO_ROOT.parent / "crog_reproduction" / "OCID-VLG",
            run / "anygrasp_input_predicted_mask",
        )
        sample = index.load_sample(
            "q0000000_b32eb3299dcd3ae9",
            camera_frame="ocid_camera_optical",
            min_component_area_px=20,
            retain_largest_component=True,
        )
        filtering = dict(
            YamlConfig(str(REPO_ROOT / "configs" / "dexnet_candidates.yaml"))[
                "filtering"
            ]
        )
        first = generate_candidates(
            sample,
            self.sampling,
            filtering,
            num_samples=64,
            top_k=20,
            seed=42,
        )
        second = generate_candidates(
            sample,
            self.sampling,
            filtering,
            num_samples=64,
            top_k=20,
            seed=42,
        )
        self.assertGreaterEqual(len(first.raw_candidates), 2)
        self.assertGreaterEqual(len(first.deduplicated_candidates), 2)
        first_values = np.asarray(
            [
                [*item["center_uv"], item["center_depth_m"], item["angle_rad"]]
                for item in first.raw_candidates
            ]
        )
        second_values = np.asarray(
            [
                [*item["center_uv"], item["center_depth_m"], item["angle_rad"]]
                for item in second.raw_candidates
            ]
        )
        np.testing.assert_array_equal(first_values, second_values)
        self.assertTrue(
            all(item["centre_inside_mask"] for item in first.deduplicated_candidates)
        )
        self.assertEqual(first.rejection_summary, second.rejection_summary)

    def test_stable_sample_seed_is_repeatable_and_sample_specific(self) -> None:
        first = derive_sample_seed(
            "q0000000_b32eb3299dcd3ae9",
            base_seed=42,
            mode="stable-sha256",
            namespace="formal-v1",
        )
        repeated = derive_sample_seed(
            "q0000000_b32eb3299dcd3ae9",
            base_seed=42,
            mode="stable-sha256",
            namespace="formal-v1",
        )
        other = derive_sample_seed(
            "q0000001_a9a5f9b502546016",
            base_seed=42,
            mode="stable-sha256",
            namespace="formal-v1",
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**32 - 1)
        self.assertEqual(
            derive_sample_seed(
                "q0000000_b32eb3299dcd3ae9",
                base_seed=42,
                mode="fixed",
                namespace="ignored",
            ),
            42,
        )


class FrozenFormalConfigurationTests(unittest.TestCase):
    def test_candidate_config_does_not_refine_predicted_mask(self) -> None:
        config = dict(
            YamlConfig(
                str(
                    REPO_ROOT
                    / "configs"
                    / "dexnet_candidates_formal_no_refinement.yaml"
                )
            )
        )
        self.assertEqual(config["input"]["min_component_area_px"], 0)
        self.assertIs(config["input"]["retain_largest_component"], False)
        self.assertEqual(config["input"]["mask_erode_px"], 0)
        self.assertEqual(config["input"]["mask_dilate_px"], 0)
        self.assertEqual(config["filtering"]["contact_mask_dilation_px"], 0)

    def test_evaluator_config_freezes_strict_corrected_semantics(self) -> None:
        config = dict(
            YamlConfig(
                str(
                    REPO_ROOT
                    / "configs"
                    / "dexnet_grasp_consistency_corrected.yaml"
                )
            )
        )
        self.assertEqual(config["evaluator_version"], "corrected_geometric_v2")
        self.assertEqual(config["iou_threshold"], 0.25)
        self.assertEqual(config["iou_comparison"], ">")
        self.assertEqual(config["angle_threshold_deg"], 30.0)
        self.assertEqual(config["angle_comparison"], "<=")
        self.assertEqual(
            config["coordinate_convention"]["polygon_vertices"], "[x,y]"
        )
        self.assertEqual(
            config["coordinate_convention"]["rasterization"],
            "row=y, column=x",
        )


class FilteringAndScoringBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        self.sample = SimpleNamespace(
            target_mask_processed=mask,
            valid_depth_mask=np.ones_like(mask),
        )
        self.filtering = {
            "require_center_inside_mask": True,
            "min_center_boundary_distance_px": 2.0,
            "endpoint_support_radius_px": 0,
            "contact_mask_dilation_px": 0,
            "min_grasp_axis_mask_support": 0.8,
            "min_valid_depth_support": 1.0,
            "min_gripper_width_m": 0.005,
            "max_gripper_width_m": 0.08,
            "image_boundary_margin_px": 2,
            "nms_center_distance_px": 8.0,
            "nms_angle_distance_deg": 15.0,
        }
        self.candidate = {
            "candidate_id": "g0000",
            "center_uv": [50.0, 50.0],
            "endpoints_uv": [[40.0, 50.0], [60.0, 50.0]],
            "contact_points_uv": None,
            "angle_rad": 0.0,
            "width_m": 0.05,
        }

    def test_center_rejection_and_rejection_reasons(self) -> None:
        accepted = validate_candidate(
            deepcopy(self.candidate), self.sample, self.filtering
        )
        self.assertIsNone(accepted["rejection_reason"])

        outside = deepcopy(self.candidate)
        outside.update(center_uv=[5.0, 5.0], endpoints_uv=[[1.0, 5.0], [9.0, 5.0]])
        rejected = validate_candidate(outside, self.sample, self.filtering)
        self.assertIn("center_outside_target_mask", rejected["rejection_reasons"])
        self.assertIn(
            "center_too_close_to_target_boundary", rejected["rejection_reasons"]
        )

    def test_deduplication_accounts_for_planar_angle_symmetry(self) -> None:
        first = deepcopy(self.candidate)
        second = deepcopy(self.candidate)
        second.update(candidate_id="g0001", center_uv=[53.0, 52.0], angle_rad=np.pi)
        third = deepcopy(self.candidate)
        third.update(candidate_id="g0002", center_uv=[75.0, 75.0], angle_rad=0.0)
        kept, rejected = deduplicate_candidates(
            [first, second, third], self.filtering
        )
        self.assertEqual([item["candidate_id"] for item in kept], ["g0000", "g0002"])
        self.assertEqual(rejected[0]["rejection_reason"], "duplicate_nms")

    def test_scoring_failure_is_explicit_on_native_runtime(self) -> None:
        with self.assertRaisesRegex(GQCNNScoringUnavailable, "TensorFlow<=1.15"):
            score_fixed_candidates(
                None,
                [],
                model_name="not_downloaded",
                model_dir=REPO_ROOT / "models" / "not_downloaded",
                scoring_config={},
                policy_config={},
            )


if __name__ == "__main__":
    unittest.main()
