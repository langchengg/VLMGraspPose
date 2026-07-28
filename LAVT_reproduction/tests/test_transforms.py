import numpy as np
import torch

from data.dataset_ocid_vlg_bert import paired_resize


def test_paired_resize_is_aligned_and_binary():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:2, :2] = 255
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1
    resized_image, resized_mask = paired_resize(image, mask, (8, 8))
    assert resized_image.shape == (3, 8, 8)
    assert resized_mask.shape == (8, 8)
    assert set(torch.unique(resized_mask).tolist()) == {0, 1}
    assert resized_mask[:4, :4].all()
    assert not resized_mask[4:, 4:].any()


def test_mask_resize_uses_nearest_not_bilinear():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    _, resized = paired_resize(image, mask, (7, 7))
    assert resized.dtype == torch.int64
    assert set(resized.unique().tolist()) == {0, 1}


def test_source_shape_mismatch_fails_closed():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    try:
        paired_resize(image, mask, 8)
    except ValueError as error:
        assert "unaligned source" in str(error)
    else:
        raise AssertionError("shape mismatch was silently accepted")
