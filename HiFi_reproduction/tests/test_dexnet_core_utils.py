"""Focused tests for dependency-light Dex-Net integration utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.grasping.camera_geometry import (
    T_CAMERA_GRASP_FIXED_APPROACH_KEY,
    CameraIntrinsicsData,
    backproject_pixel,
    depth_mm_to_meters,
    fixed_approach_pose,
    grasp_endpoints_uv,
    named_fixed_approach_pose,
    width_m_to_pixels,
    width_pixels_to_meters,
)
from src.grasping.grasp_serialization import (
    save_candidate_bundle,
    save_candidates_npz,
)
from src.grasping.grasp_visualization import (
    save_candidate_overlay,
    save_depth_visualization,
)
from src.grasping.mask_processing import (
    binary_dilate,
    binary_erode,
    intersect_valid_depth,
    process_mask_with_diagnostics,
    remove_small_components,
    resize_mask_nearest,
    retain_largest_component,
    to_binary_mask,
)


class MaskProcessingTests(unittest.TestCase):
    def test_threshold_binary_resize_and_alignment(self) -> None:
        probability = np.array([[0.1, 0.6], [np.nan, 1.0]], dtype=np.float32)
        binary = to_binary_mask(probability, threshold=0.6)
        np.testing.assert_array_equal(binary, [[False, True], [False, True]])

        resized = resize_mask_nearest(binary, (4, 4))
        expected = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(resized, expected)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            intersect_valid_depth(binary, np.ones((3, 3), dtype=np.float32))

    def test_components_morphology_and_valid_depth(self) -> None:
        mask = np.zeros((9, 9), dtype=bool)
        mask[1, 1] = True
        mask[3:7, 3:7] = True
        cleaned = remove_small_components(mask, min_size_px=2)
        self.assertFalse(cleaned[1, 1])
        np.testing.assert_array_equal(retain_largest_component(mask), cleaned)
        eroded = binary_erode(cleaned, 1)
        self.assertLess(np.count_nonzero(eroded), np.count_nonzero(cleaned))
        self.assertGreaterEqual(np.count_nonzero(binary_dilate(eroded, 1)), np.count_nonzero(eroded))

        depth = np.ones((9, 9), dtype=np.float32)
        depth[4, 4] = 0.0
        result = process_mask_with_diagnostics(
            mask.astype(np.float32),
            depth,
            threshold=0.5,
            min_component_size_px=2,
            keep_largest_component=True,
            dilate_radius_px=1,
        )
        self.assertTrue(result.original_binary[1, 1])
        self.assertFalse(result.processed[1, 1])
        self.assertFalse(result.processed[4, 4])
        self.assertFalse(result.valid_depth[4, 4])


class CameraGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsicsData(
            frame="ocid_camera",
            fx=500.0,
            fy=400.0,
            cx=320.0,
            cy=240.0,
            skew=0.0,
            height=480,
            width=640,
        )

    def test_depth_conversion_invalid_handling_and_source_preservation(self) -> None:
        source = np.array([[1000.0, 0.0], [np.nan, -5.0]], dtype=np.float32)
        before = source.copy()
        converted = depth_mm_to_meters(source)
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_array_equal(converted, [[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(source, before)

    def test_intrinsics_backprojection_width_and_endpoints(self) -> None:
        point = backproject_pixel(370.0, 200.0, 2.0, self.intrinsics)
        np.testing.assert_allclose(point, [0.2, -0.2, 2.0], atol=1e-7)
        width_px = width_m_to_pixels(0.08, 1.0, self.intrinsics, angle_rad=0.0)
        self.assertAlmostEqual(width_px, 40.0)
        self.assertAlmostEqual(
            width_pixels_to_meters(width_px, 1.0, self.intrinsics, angle_rad=0.0),
            0.08,
        )
        endpoints = grasp_endpoints_uv([100.0, 50.0], 40.0, 0.0)
        np.testing.assert_allclose(endpoints, [[80.0, 50.0], [120.0, 50.0]])

    def test_finite_intrinsics_and_fixed_approach_pose_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            CameraIntrinsicsData("camera", np.nan, 1, 0, 0, 0, 1, 1)
        pose = fixed_approach_pose([0.1, -0.2, 1.0], np.pi / 2)
        np.testing.assert_allclose(pose[:3, 3], [0.1, -0.2, 1.0])
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
        named = named_fixed_approach_pose([0.1, -0.2, 1.0], 0.0)
        self.assertEqual(list(named), [T_CAMERA_GRASP_FIXED_APPROACH_KEY])
        self.assertNotIn("6dof", T_CAMERA_GRASP_FIXED_APPROACH_KEY.lower())

    def test_official_camera_intrinsics_construction(self) -> None:
        official = self.intrinsics.to_perception()
        self.assertEqual(official.frame, "ocid_camera")
        self.assertEqual(official.height, 480)
        self.assertEqual(official.width, 640)
        np.testing.assert_allclose(official.K, self.intrinsics.K)


class SerializationAndVisualizationTests(unittest.TestCase):
    def candidate(self) -> dict:
        return {
            "candidate_id": "g000",
            "sample_id": "sample_001",
            "query": "the red cup",
            "center_uv": [20.0, 15.0],
            "center_depth_m": 1.2,
            "center_camera_xyz_m": [0.0, 0.0, 1.2],
            "angle_rad": 0.0,
            "width_m": 0.06,
            "width_px": 12.0,
            "endpoints_uv": [[14.0, 15.0], [26.0, 15.0]],
            "centre_inside_mask": True,
            "grasp_axis_mask_support": 0.9,
            "rejection_reason": None,
            "camera_frame": "ocid_camera",
            "seed": 42,
            T_CAMERA_GRASP_FIXED_APPROACH_KEY: fixed_approach_pose([0, 0, 1.2], 0),
        }

    def test_deterministic_json_npz_csv_and_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = save_candidate_bundle(
                [self.candidate()],
                json_path=root / "first.json",
                npz_path=root / "first.npz",
                csv_path=root / "first.csv",
                metadata={"seed": 42},
            )
            second = save_candidate_bundle(
                [self.candidate()],
                json_path=root / "second.json",
                npz_path=root / "second.npz",
                csv_path=root / "second.csv",
                metadata={"seed": 42},
            )
            for extension in ("json", "npz", "csv"):
                self.assertEqual(first[extension].read_bytes(), second[extension].read_bytes())
            payload = json.loads(first["json"].read_text())
            self.assertTrue(np.isnan(payload["candidates"][0]["gqcnn_q_value"]))
            with np.load(first["npz"], allow_pickle=False) as arrays:
                self.assertEqual(arrays["center_uv"].shape, (1, 2))
                self.assertTrue(np.isnan(arrays["gqcnn_q_value"][0]))
                self.assertEqual(arrays[T_CAMERA_GRASP_FIXED_APPROACH_KEY].shape, (1, 4, 4))
            with first["csv"].open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["gqcnn_q_value"], "NaN")

    def test_npz_is_byte_deterministic_across_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            one = save_candidates_npz(Path(temporary) / "one.npz", [self.candidate()])
            two = save_candidates_npz(Path(temporary) / "two.npz", [self.candidate()])
            self.assertEqual(hashlib.sha256(one.read_bytes()).digest(), hashlib.sha256(two.read_bytes()).digest())

    def test_rgb_and_depth_overlays_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = np.zeros((30, 40, 3), dtype=np.uint8)
            rgb[..., 1] = 80
            mask = np.zeros((30, 40), dtype=bool)
            mask[8:23, 10:31] = True
            depth = np.full((30, 40), 1.2, dtype=np.float32)
            depth[0, 0] = 0.0
            rgb_path = save_candidate_overlay(
                rgb, [self.candidate()], root / "rgb.png", mask=mask, title="RGB candidates"
            )
            depth_path = save_depth_visualization(
                depth,
                root / "depth.png",
                candidates=[self.candidate()],
                mask=mask,
                title="Depth candidates",
            )
            for path in (rgb_path, depth_path):
                self.assertTrue(path.is_file())
                with Image.open(path) as image:
                    self.assertGreater(image.width, 10)
                    self.assertGreater(image.height, 10)


if __name__ == "__main__":
    unittest.main()
