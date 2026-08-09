import math

import numpy as np
import pytest

from experiments.fair_crog_hifics_g1_c1_no_rerank.geometry import (
    CanonicalGrasp,
    continuous_iou_cv2,
    continuous_iou_shapely,
    periodic_angle_error_deg,
    raster_iou,
)
from HiFi_reproduction.src.grasping.common.geometry import CropTransform


def grasp(**changes):
    values = dict(cx_px=100, cy_px=140, theta_deg=10, jaw_width_px=80, rectangle_height_px=20)
    values.update(changes)
    return CanonicalGrasp(**values)


def test_identical_rectangle_and_180_symmetry():
    assert raster_iou(grasp(), grasp()) == 1.0
    assert periodic_angle_error_deg(10, 190) == 0
    assert periodic_angle_error_deg(89, -89) == 2


def test_angle_boundary_and_radians_degrees():
    assert periodic_angle_error_deg(0, 30) <= 30
    assert periodic_angle_error_deg(0, 30 + 1e-9) > 30
    assert math.degrees(math.radians(37.5)) == pytest.approx(37.5)


def test_xy_swap_matters_in_non_square_frame():
    target = grasp(cx_px=500, cy_px=100)
    swapped = grasp(cx_px=100, cy_px=500)
    assert raster_iou(target, target) == 1
    assert raster_iou(target, swapped) == 0


def test_transform_round_trip():
    transform = CropTransform(50, 20, 240, 240, 300, 300, 640, 480)
    model = transform.native_to_model_pose(190, 120, 35, 75)
    native = transform.model_to_native_pose(*model)
    assert native == pytest.approx((190, 120, 35, 75), abs=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cx_px": np.nan},
        {"cy_px": np.inf},
        {"jaw_width_px": -1},
        {"rectangle_height_px": 0},
    ],
)
def test_invalid_geometry_rejected(kwargs):
    with pytest.raises(ValueError):
        grasp(**kwargs)


def test_out_of_bounds_is_clipped_not_crashed():
    assert 0 <= raster_iou(grasp(cx_px=-20), grasp(cx_px=0)) <= 1


def test_independent_continuous_polygon_implementations_agree():
    a, b = grasp(), grasp(cx_px=117, cy_px=146, theta_deg=-23)
    assert continuous_iou_cv2(a, b) == pytest.approx(continuous_iou_shapely(a, b), abs=2e-6)

