from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


def _tensor(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"hook expected tensor, got {type(value).__name__}")
    return value.detach()


class FullChainCapture:
    """Non-mutating hooks for the deployed CROG forward chain."""

    def __init__(self, model):
        self.model = model
        self.values: dict[str, Any] = {}
        self._handles = []

    @property
    def base(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _save(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                self.values[name] = tuple(_tensor(item) for item in output)
            else:
                self.values[name] = _tensor(output)
            return None
        return hook

    def _save_attention(self, name: str):
        def hook(_module, _inputs, output):
            self.values[name] = _tensor(output[1])
            return None
        return hook

    def install(self):
        if self._handles:
            raise RuntimeError("full-chain hooks already installed")
        base = self.base
        self._handles.extend([
            base.backbone.visual.register_forward_hook(self._save("visual_multiscale")),
            base.backbone.ln_final.register_forward_hook(self._save("token_features")),
            base.neck.register_forward_hook(self._save("fpn_pre_decoder")),
            base.proj.vis.register_forward_hook(self._save("projector_visual")),
            base.proj.txt.register_forward_hook(self._save("dynamic_text")),
            base.proj.register_forward_hook(self._save("raw_heads")),
        ])
        for index, layer in enumerate(base.decoder.layers):
            self._handles.append(layer.register_forward_hook(self._save(f"decoder_{index + 1}")))
            self._handles.append(layer.multihead_attn.register_forward_hook(self._save_attention(f"cross_attention_{index + 1}")))
        return self

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def suspended(self):
        installed = bool(self._handles)
        if installed:
            self.remove()
        try:
            yield
        finally:
            if installed:
                self.install()

    def clear(self):
        self.values.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.remove()

    def feature_maps(self) -> dict[str, torch.Tensor]:
        required = {
            "visual_multiscale", "token_features", "fpn_pre_decoder",
            "projector_visual", "dynamic_text", "raw_heads",
            "decoder_1", "decoder_2", "decoder_3",
            "cross_attention_1", "cross_attention_2", "cross_attention_3",
        }
        missing = sorted(required - self.values.keys())
        if missing:
            raise RuntimeError(f"full-chain hook did not capture: {missing}")
        c3, c4, c5 = self.values["visual_multiscale"]
        projector = self.values["projector_visual"]
        branches = torch.tensor_split(projector, 5, dim=1)
        result = {
            "c3": c3, "c4": c4, "c5": c5,
            "fpn_pre": self.values["fpn_pre_decoder"],
            "decoder_1": self._decoder_map(self.values["decoder_1"]),
            "decoder_2": self._decoder_map(self.values["decoder_2"]),
            "decoder_3_post": self._decoder_map(self.values["decoder_3"]),
        }
        for name, value in zip(("projector_mask", "projector_quality", "projector_sin", "projector_cos", "projector_width"), branches, strict=True):
            result[name] = value
        return result

    @staticmethod
    def _decoder_map(value: torch.Tensor) -> torch.Tensor:
        # Layer output is [HW,B,C]; post-normalisation happens in decoder, but
        # the per-layer hook intentionally records each actual layer output.
        if value.ndim != 3:
            raise ValueError("decoder layer output must be [HW,B,C]")
        spatial, batch, channels = value.shape
        side = int(round(spatial ** 0.5))
        if side * side != spatial:
            raise ValueError("decoder spatial sequence is not square")
        return value.permute(1, 2, 0).reshape(batch, channels, side, side)

    def tokens(self) -> torch.Tensor:
        return self.values["token_features"]

    def dynamic(self) -> torch.Tensor:
        return self.values["dynamic_text"]

    def raw_heads(self) -> tuple[torch.Tensor, ...]:
        return self.values["raw_heads"]

    def attention_maps(self) -> dict[str, torch.Tensor]:
        result = {}
        for index in range(1, 4):
            value = self.values[f"cross_attention_{index}"]
            if value.ndim != 3:
                raise ValueError("cross-attention weights must be [B,HW,L]")
            batch, spatial, tokens = value.shape
            side = int(round(spatial ** 0.5))
            result[f"attention_{index}"] = value.permute(0, 2, 1).reshape(batch, tokens, side, side)
        return result


def assert_capture_non_mutating(model, inputs: tuple[Any, ...], capture: FullChainCapture) -> dict[str, float]:
    with torch.no_grad():
        with capture.suspended():
            without, _ = model(*inputs)
        capture.clear()
        with_capture, _ = model(*inputs)
    differences = {}
    for name, left, right in zip(("M", "Q", "sin", "cos", "W"), without, with_capture, strict=True):
        differences[name] = float((left - right).abs().max().cpu())
        if not torch.equal(left, right):
            raise AssertionError(f"full-chain hooks changed {name}; max={differences[name]}")
    capture.feature_maps()
    return differences

