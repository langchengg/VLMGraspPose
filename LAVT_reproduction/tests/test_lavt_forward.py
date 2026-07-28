from argparse import Namespace
from pathlib import Path

import torch
import pytest

from lib.segmentation import lavt_one
from ocid_vlg.losses import build_loss


def test_lavt_one_swin_base_cpu_forward_and_backward_are_finite():
    """Exercise the real LAVT-One graph, not a stand-in segmentation module."""

    swin_weights = Path(
        "pretrained_weights/swin_base_patch4_window12_384_22k.pth"
    )
    bert_weights = Path("pretrained_weights/bert-base-uncased")
    if not swin_weights.is_file() or not bert_weights.is_dir():
        pytest.skip("local official Swin/BERT weights are required for this integration test")

    args = Namespace(
        swin_type="base",
        window12=True,
        mha="",
        fusion_drop=0.0,
        use_checkpoint=True,
        ck_bert=str(bert_weights),
    )
    model = lavt_one(pretrained=str(swin_weights), args=args).cpu().train()
    image = torch.randn(1, 3, 64, 64)
    input_ids = torch.tensor(
        [[101, 1996, 2417, 4874, 102] + [0] * 15], dtype=torch.long
    )
    attention_mask = (input_ids != 0).long()
    target = torch.zeros(1, 64, 64, dtype=torch.long)
    target[:, 16:48, 16:48] = 1

    logits = model(image, input_ids, l_mask=attention_mask)
    loss = build_loss("dice")(logits, target)
    loss.backward()

    assert logits.shape == (1, 2, 64, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
