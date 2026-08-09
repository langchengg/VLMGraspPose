"""Frozen OpenAI CLIP ViT-B/16 crop semantics for proposal candidates."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import numpy as np
from PIL import Image


EXPECTED_CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
DENSE_FEATURE_SCHEMA_VERSION = 2


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, value, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_savez(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crop_box(mask: np.ndarray, margin_fraction: float = 0.05) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)
    x1, y1, x2, y2 = int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())
    dx = max(2, int(round((x2 - x1 + 1) * margin_fraction)))
    dy = max(2, int(round((y2 - y1 + 1) * margin_fraction)))
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(mask.shape[1] - 1, x2 + dx),
        min(mask.shape[0] - 1, y2 + dy),
    )


class FrozenClipCandidateEncoder:
    def __init__(
        self,
        cache_root: str | Path,
        *,
        batch_size: int = 64,
        num_threads: int = 8,
    ):
        import clip
        import torch

        self.clip = clip
        self.torch = torch
        torch.set_num_threads(int(num_threads))
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("CLIP batch size must be positive")
        weight = Path.home() / ".cache/clip/ViT-B-16.pt"
        if not weight.is_file() or sha256_file(weight) != EXPECTED_CLIP_SHA256:
            raise ValueError("frozen OpenAI CLIP ViT-B/16 weight identity mismatch")
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.model, self.preprocess = clip.load(
            "ViT-B/16",
            device="cpu",
            jit=False,
            download_root=str(weight.parent),
        )
        self.model.float().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def text_embeddings(self, texts: list[str]) -> np.ndarray:
        unique = list(dict.fromkeys(texts))
        values: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for text in unique:
            digest = hashlib.sha256(text.encode()).hexdigest()
            path = self.cache_root / "text" / digest[:2] / f"{digest}.npy"
            if path.is_file():
                values[text] = np.load(path, allow_pickle=False).astype(np.float32)
            else:
                missing.append(text)
        if missing:
            tokens = self.clip.tokenize(missing, truncate=True)
            with self.torch.inference_mode():
                encoded = self.model.encode_text(tokens).float()
                encoded /= encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            for text, value in zip(missing, encoded.cpu().numpy(), strict=True):
                digest = hashlib.sha256(text.encode()).hexdigest()
                path = self.cache_root / "text" / digest[:2] / f"{digest}.npy"
                _atomic_save_npy(path, value.astype(np.float32))
                values[text] = value.astype(np.float32)
        return np.stack([values[text] for text in texts])

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            tensor = self.torch.stack(
                [self.preprocess(image) for image in images[start : start + self.batch_size]]
            )
            with self.torch.inference_mode():
                encoded = self.model.encode_image(tensor).float()
                encoded /= encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            batches.append(encoded.cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)

    def dense_image_embedding(self, rgb: np.ndarray, rgb_sha256: str) -> np.ndarray:
        """Return normalized 14x14 CLIP patch tokens from one frozen frame pass."""

        path = self.cache_root / "dense_images" / rgb_sha256[:2] / f"{rgb_sha256}.npy"
        if path.is_file():
            value = np.load(path, allow_pickle=False).astype(np.float32)
            if value.shape != (14, 14, 512):
                raise ValueError("cached dense CLIP embedding shape mismatch")
            return value
        tensor = self.preprocess(Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB"))[None]
        visual = self.model.visual
        with self.torch.inference_mode():
            x = visual.conv1(tensor)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            class_token = visual.class_embedding.to(x.dtype) + self.torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype
            )
            x = self.torch.cat([class_token, x], dim=1)
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = visual.ln_post(x)
            if visual.proj is not None:
                x = x @ visual.proj
            tokens = x[:, 1:]
            tokens /= tokens.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        value = tokens[0].reshape(14, 14, 512).float().cpu().numpy()
        _atomic_save_npy(path, value.astype(np.float16))
        return value.astype(np.float32)

    @staticmethod
    def _weighted_embedding(tokens: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float32)
        total = float(weights.sum())
        if total <= 1e-8:
            return np.zeros(tokens.shape[-1], dtype=np.float32)
        value = np.sum(tokens * weights[..., None], axis=(0, 1)) / total
        norm = float(np.linalg.norm(value))
        return value / max(norm, 1e-12)

    def dense_candidate_features(
        self,
        rgb: np.ndarray,
        rgb_sha256: str,
        masks: dict[str, np.ndarray],
        candidate_ids: list[str],
        *,
        full_query: str,
        target_category: str | None,
        target_attribute: str | None,
        sample_id: str,
        namespace: str = "dense_candidates",
    ) -> dict[str, np.ndarray]:
        if namespace not in {"dense_candidates", "dense_candidates_stage2"}:
            raise ValueError(f"unsupported CLIP cache namespace: {namespace}")
        cache_path = self.cache_root / namespace / f"{sample_id}.npz"
        identity = hashlib.sha256(
            (
                f"schema={DENSE_FEATURE_SCHEMA_VERSION}\n"
                f"checkpoint={EXPECTED_CLIP_SHA256}\n"
                + rgb_sha256
                + "\n"
                + "\n".join(candidate_ids)
                + "\n"
                + full_query
            ).encode()
        ).hexdigest()
        if cache_path.is_file():
            archive = np.load(cache_path, allow_pickle=False)
            try:
                if str(archive["identity"].item()) == identity:
                    return {
                        key: np.asarray(archive[key])
                        for key in archive.files
                        if key.startswith("feature_") or key == "candidate_id"
                    }
            finally:
                archive.close()
        tokens = self.dense_image_embedding(rgb, rgb_sha256)
        text_values = [
            full_query,
            target_category or full_query,
            target_attribute or target_category or full_query,
        ]
        text_embeddings = self.text_embeddings(text_values)
        masked_embeddings: list[np.ndarray] = []
        box_embeddings: list[np.ndarray] = []
        background_embeddings: list[np.ndarray] = []
        validity: list[bool] = []
        for candidate_id in candidate_ids:
            mask = np.asarray(masks[candidate_id], dtype=np.uint8)
            weights = np.asarray(
                Image.fromarray(mask * 255, mode="L").resize(
                    (14, 14), Image.Resampling.BOX
                ),
                dtype=np.float32,
            ) / 255.0
            validity.append(bool(weights.sum() > 1e-8))
            x1, y1, x2, y2 = _crop_box(mask.astype(bool), margin_fraction=0.0)
            box = np.zeros((14, 14), dtype=np.float32)
            px1 = max(0, min(13, int(np.floor(x1 * 14 / mask.shape[1]))))
            py1 = max(0, min(13, int(np.floor(y1 * 14 / mask.shape[0]))))
            px2 = max(0, min(13, int(np.floor(x2 * 14 / mask.shape[1]))))
            py2 = max(0, min(13, int(np.floor(y2 * 14 / mask.shape[0]))))
            box[py1 : py2 + 1, px1 : px2 + 1] = 1.0
            masked_embeddings.append(self._weighted_embedding(tokens, weights))
            box_embeddings.append(self._weighted_embedding(tokens, box))
            background_embeddings.append(
                self._weighted_embedding(tokens, np.clip(box - weights, 0.0, 1.0))
            )
        masked_array = np.stack(masked_embeddings)
        box_array = np.stack(box_embeddings)
        background_array = np.stack(background_embeddings)
        features = {
            "candidate_id": np.asarray(candidate_ids),
            "feature_clip_full_query_similarity": masked_array @ text_embeddings[0],
            "feature_clip_target_category_similarity": masked_array @ text_embeddings[1],
            "feature_clip_target_attribute_similarity": masked_array @ text_embeddings[2],
            "feature_clip_box_crop_similarity": box_array @ text_embeddings[0],
            "feature_clip_candidate_crop_background_contrast": (
                masked_array @ text_embeddings[0]
                - background_array @ text_embeddings[0]
            ),
            "feature_clip_features_valid": np.asarray(validity, dtype=bool),
            "feature_clip_features_missing": ~np.asarray(validity, dtype=bool),
        }
        _atomic_savez(
            cache_path,
            identity=np.asarray(identity),
            masked_embeddings=masked_array.astype(np.float16),
            box_embeddings=box_array.astype(np.float16),
            background_embeddings=background_array.astype(np.float16),
            **features,
        )
        return features

    def candidate_features(
        self,
        rgb: np.ndarray,
        masks: dict[str, np.ndarray],
        candidate_ids: list[str],
        *,
        full_query: str,
        target_category: str | None,
        target_attribute: str | None,
        sample_id: str,
    ) -> dict[str, np.ndarray]:
        cache_path = self.cache_root / "candidates" / f"{sample_id}.npz"
        identity = hashlib.sha256(
            ("\n".join(candidate_ids) + "\n" + full_query).encode()
        ).hexdigest()
        if cache_path.is_file():
            archive = np.load(cache_path, allow_pickle=False)
            try:
                if str(archive["identity"].item()) == identity:
                    return {
                        key: np.asarray(archive[key])
                        for key in archive.files
                        if key.startswith("feature_") or key == "candidate_id"
                    }
            finally:
                archive.close()

        rgb = np.asarray(rgb, dtype=np.uint8)
        masked_crops: list[Image.Image] = []
        box_crops: list[Image.Image] = []
        for candidate_id in candidate_ids:
            mask = masks[candidate_id]
            x1, y1, x2, y2 = _crop_box(mask)
            crop_rgb = rgb[y1 : y2 + 1, x1 : x2 + 1]
            crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
            masked = crop_rgb.copy()
            masked[~crop_mask] = 127
            masked_crops.append(Image.fromarray(masked, mode="RGB"))
            box_crops.append(Image.fromarray(crop_rgb, mode="RGB"))
        masked_embeddings = self._encode_images(masked_crops)
        box_embeddings = self._encode_images(box_crops)
        text_values = [full_query, target_category or full_query, target_attribute or target_category or full_query]
        text_embeddings = self.text_embeddings(text_values)
        features = {
            "candidate_id": np.asarray(candidate_ids),
            "feature_clip_full_query_similarity": masked_embeddings @ text_embeddings[0],
            "feature_clip_target_category_similarity": masked_embeddings @ text_embeddings[1],
            "feature_clip_target_attribute_similarity": masked_embeddings @ text_embeddings[2],
            "feature_clip_box_crop_similarity": box_embeddings @ text_embeddings[0],
            "feature_clip_candidate_crop_background_contrast": (
                masked_embeddings @ text_embeddings[0]
                - box_embeddings @ text_embeddings[0]
            ),
            "feature_clip_features_valid": np.ones(len(candidate_ids), dtype=bool),
            "feature_clip_features_missing": np.zeros(len(candidate_ids), dtype=bool),
        }
        _atomic_savez(
            cache_path,
            identity=np.asarray(identity),
            masked_embeddings=masked_embeddings.astype(np.float16),
            box_embeddings=box_embeddings.astype(np.float16),
            **features,
        )
        return features


__all__ = ["EXPECTED_CLIP_SHA256", "FrozenClipCandidateEncoder"]
